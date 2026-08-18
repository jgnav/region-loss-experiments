import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .region_loss import RegionLoss


class iBOTLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        patch_out_dim,
        ngcrops,
        nlcrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp2,
        teacher_temp2,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
        center_momentum2=0.9,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=1.0,
        region_warmup_epochs=0,
        region_min_area=0.05,
        mim_start_epoch=0,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.region_loss = RegionLoss(region_min_area)

        if not 0 <= region_warmup_epochs <= nepochs:
            raise ValueError(
                "region_warmup_epochs must be between 0 and the number of epochs"
            )
        if region_warmup_epochs:
            self.region_weight_schedule = np.concatenate(
                (
                    np.linspace(0.0, lambda3, region_warmup_epochs),
                    np.full(nepochs - region_warmup_epochs, lambda3),
                )
            )
        else:
            self.region_weight_schedule = np.full(nepochs, lambda3)

        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp,
                    teacher_temp,
                    warmup_teacher_temp_epochs,
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )
        self.teacher_temp2_schedule = (
            np.concatenate(
                (
                    np.linspace(
                        warmup_teacher_temp2,
                        teacher_temp2,
                        warmup_teacher_temp_epochs,
                    ),
                    np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2,
                )
            )
            if mim_start_epoch == 0
            else np.concatenate(
                (
                    np.ones(mim_start_epoch) * warmup_teacher_temp2,
                    np.linspace(
                        warmup_teacher_temp2,
                        teacher_temp2,
                        warmup_teacher_temp_epochs,
                    ),
                    np.ones(
                        nepochs
                        - warmup_teacher_temp_epochs
                        - mim_start_epoch
                    )
                    * teacher_temp2,
                )
            )
        )

    def forward(
        self,
        student_output,
        teacher_output,
        student_local_cls,
        student_mask,
        crop_boxes,
        epoch,
    ):
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output

        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)

        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax(
            (teacher_patch - self.center2) / temp2, dim=-1
        )
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    loss2 = torch.sum(
                        -teacher_patch_c[q]
                        * F.log_softmax(student_patch_c[v], dim=-1),
                        dim=-1,
                    )
                    mask = student_mask[v].flatten(-2, -1)
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(
                        dim=-1
                    ).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(
                        -teacher_cls_c[q]
                        * F.log_softmax(student_cls_c[v], dim=-1),
                        dim=-1,
                    )
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1

        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        region_stats = self.region_loss(
            student_patch_c,
            teacher_patch_c,
            crop_boxes,
        )
        region_weight = float(self.region_weight_schedule[epoch])
        total_loss3 = region_stats["loss"] * region_weight
        total_loss = {
            "cls": total_loss1,
            "patch": total_loss2,
            "region": total_loss3,
            "region_raw": region_stats["loss"],
            "region_weight": total_loss3.new_tensor(region_weight),
            "region_valid_ratio": region_stats["valid_ratio"],
            "region_intersection_area": region_stats["intersection_area"],
            "loss": total_loss1 + total_loss2 + total_loss3,
        }
        self.update_center(teacher_cls, teacher_patch)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (
            1 - self.center_momentum
        )

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (
            len(teacher_patch) * dist.get_world_size()
        )
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (
            1 - self.center_momentum2
        )
