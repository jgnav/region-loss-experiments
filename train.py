"""Warm-start iBOT pretraining from model weights using one YAML config."""

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import wandb
import yaml
from tensorboardX import SummaryWriter

from data import DataAugmentationiBOT, ImageFolderMask
from losses import iBOTLoss
from model import create_model, iBOTHead
from utils import training as utils
from utils.checkpoint import load_pretrained_weights, read_pretrained_checkpoint
from utils.recipe import get_ibot_recipe


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML training configuration")
    return parser.parse_args()


def load_config(path):
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config.update(get_ibot_recipe(config["arch"]))
    for key in ("data_path", "initial_checkpoint", "output_dir"):
        config[key] = os.path.expandvars(os.path.expanduser(config[key]))
    config["initial_checkpoint"] = Path(config["initial_checkpoint"])
    return SimpleNamespace(**config)


def init_wandb(args):
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        dir=args.output_dir,
        config=config,
        job_type="pretraining-warm-start",
    )


def train_ibot(args, wandb_run=None):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    args.effective_batch_size = args.batch_size_per_gpu * utils.get_world_size()
    checkpoint = read_pretrained_checkpoint(args)
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "effective_batch_size": args.effective_batch_size,
                "start_epoch": 0,
                "final_epoch": args.epochs,
                "source_checkpoint_epoch": args.source_checkpoint_epoch,
            }
        )
    print("\n".join(f"{key}: {value}" for key, value in sorted(vars(args).items())))
    cudnn.benchmark = True

    transform = DataAugmentationiBOT(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
        args.global_crop_size,
        args.local_crop_size,
    )
    dataset = ImageFolderMask(
        args.data_path,
        transform=transform,
        patch_size=args.patch_size,
        pred_ratio=args.pred_ratio,
        pred_ratio_var=args.pred_ratio_var,
        pred_aspect_ratio=args.pred_aspect_ratio,
        pred_shape=args.pred_shape,
        pred_start_epoch=args.pred_start_epoch,
    )
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if len(data_loader) == 0:
        raise ValueError("The distributed data loader is empty")
    print(f"Data loaded: there are {len(dataset)} images.")

    student = create_model(
        args.arch,
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path,
        return_all_tokens=True,
        masked_im_modeling=args.use_masked_im_modeling,
    )
    teacher = create_model(
        args.arch,
        patch_size=args.patch_size,
        return_all_tokens=True,
    )
    embed_dim = student.embed_dim

    student = utils.MultiCropWrapper(
        student,
        iBOTHead(
            embed_dim,
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            norm_last_layer=args.norm_last_layer,
            shared_head=args.shared_head,
        ),
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim,
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        ),
    )
    student, teacher = student.cuda(), teacher.cuda()
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        teacher = nn.parallel.DistributedDataParallel(
            teacher, device_ids=[args.gpu]
        )
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    same_dim = args.shared_head or args.shared_head_teacher
    ibot_loss = iBOTLoss(
        args.out_dim,
        args.out_dim if same_dim else args.patch_out_dim,
        args.global_crops_number,
        args.local_crops_number,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_patch_temp,
        args.teacher_patch_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum,
        center_momentum2=args.center_momentum2,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        region_min_area=args.region_min_area,
        mim_start_epoch=args.pred_start_epoch,
    ).cuda()

    writer = None
    if utils.is_main_process():
        writer = SummaryWriter(logdir=os.path.join(args.output_dir, "tf_logs"))

    params_groups = utils.get_params_groups(student)
    if args.optimizer != "adamw":
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    optimizer = torch.optim.AdamW(params_groups)
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    if args.lr_schedule != "cosine":
        raise ValueError(f"Unsupported learning-rate schedule: {args.lr_schedule}")
    lr_schedule = utils.cosine_scheduler(
        args.lr
        * (args.batch_size_per_gpu * utils.get_world_size())
        / args.reference_batch_size,
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader)
    )
    print("Loss, optimizer and schedulers ready.")

    load_pretrained_weights(checkpoint, student, teacher)
    del checkpoint
    source_epoch = args.source_checkpoint_epoch
    source_description = (
        f" at epoch {source_epoch}" if source_epoch is not None else ""
    )
    print(
        f"Loaded pretrained student and teacher weights{source_description}. "
        f"Starting a fresh {args.epochs}-epoch optimization run."
    )

    start_time = time.time()
    for epoch in range(args.epochs):
        data_loader.sampler.set_epoch(epoch)
        data_loader.dataset.set_epoch(epoch)

        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            ibot_loss,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            fp16_scaler,
            args,
        )

        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
            "ibot_loss": ibot_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        utils.save_on_master(
            save_dict, os.path.join(args.output_dir, "checkpoint.pth")
        )
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
            utils.save_on_master(
                save_dict,
                os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth"),
            )
        log_stats = {
            **{f"train_{key}": value for key, value in train_stats.items()},
            "epoch": epoch,
        }
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as handle:
                handle.write(json.dumps(log_stats) + "\n")
            for key, value in train_stats.items():
                writer.add_scalar(key, value, epoch)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch + 1,
                        "state/global_step": (epoch + 1) * len(data_loader),
                        **{
                            f"train/{key}": value
                            for key, value in train_stats.items()
                        },
                    },
                    step=epoch + 1,
                )

    if writer is not None:
        writer.close()
    total_time = time.time() - start_time
    total_time_string = str(datetime.timedelta(seconds=int(total_time)))
    if wandb_run is not None:
        wandb_run.summary["state/final_epoch"] = args.epochs
        wandb_run.summary["state/training_time_seconds"] = total_time
        wandb_run.summary["state/checkpoint"] = str(
            Path(args.output_dir) / "checkpoint.pth"
        )
    print(f"Training time {total_time_string}")


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    ibot_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    fp16_scaler,
    args,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}/{args.epochs}]"

    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.module.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [
        param_q
        for name_q, param_q in zip(names_q, params_q)
        if name_q in names_common
    ]
    params_k = [
        param_k
        for name_k, param_k in zip(names_k, params_k)
        if name_k in names_common
    ]

    for iteration, (images, _labels, masks, crop_boxes) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        schedule_iteration = len(data_loader) * epoch + iteration
        for index, parameter_group in enumerate(optimizer.param_groups):
            parameter_group["lr"] = lr_schedule[schedule_iteration]
            if index == 0:
                parameter_group["weight_decay"] = wd_schedule[schedule_iteration]

        images = [image.cuda(non_blocking=True) for image in images]
        masks = [mask.cuda(non_blocking=True) for mask in masks]
        crop_boxes = crop_boxes.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[: args.global_crops_number])
            student_output = student(
                images[: args.global_crops_number],
                mask=masks[: args.global_crops_number],
            )

            student.module.backbone.masked_im_modeling = False
            student_local_cls = (
                student(images[args.global_crops_number :])[0]
                if len(images) > args.global_crops_number
                else None
            )
            student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            all_loss = ibot_loss(
                student_output,
                teacher_output,
                student_local_cls,
                masks,
                crop_boxes,
                epoch,
            )
            loss = all_loss.pop("loss")

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training", force=True)
            sys.exit(1)

        probs1 = teacher_output[0].chunk(args.global_crops_number)
        probs2 = student_output[0].chunk(args.global_crops_number)
        pred1 = utils.concat_all_gather(probs1[0].max(dim=1)[1])
        pred2 = utils.concat_all_gather(probs2[1].max(dim=1)[1])
        accuracy = (pred1 == pred2).sum() / pred1.size(0)

        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(
                epoch, student, args.freeze_last_layer
            )
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(
                epoch, student, args.freeze_last_layer
            )
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        with torch.no_grad():
            momentum = momentum_schedule[schedule_iteration]
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(momentum).add_(
                    (1 - momentum) * param_q.detach().data
                )

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        for key, value in all_loss.items():
            metric_logger.update(**{key: value.item()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(acc=accuracy)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


def main():
    cli_args = parse_args()
    args = load_config(cli_args.config)
    output_checkpoint = Path(args.output_dir) / "checkpoint.pth"
    if args.initial_checkpoint.resolve() == output_checkpoint.resolve():
        raise ValueError("The output checkpoint must not overwrite the input checkpoint")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if int(os.environ.get("RANK", "0")) == 0:
        wandb_run = init_wandb(args)
    exit_code = 0
    try:
        train_ibot(args, wandb_run)
    except BaseException:
        exit_code = 1
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish(exit_code=exit_code)
        utils.destroy_process_group()


if __name__ == "__main__":
    main()
