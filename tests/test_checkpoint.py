import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from losses import iBOTLoss
from utils.checkpoint import (
    load_pretrained_state,
    load_resume_state,
    read_pretrained_checkpoint,
    read_resume_checkpoint,
)


def make_loss():
    return iBOTLoss(
        out_dim=3,
        patch_out_dim=3,
        ngcrops=2,
        nlcrops=0,
        warmup_teacher_temp=0.07,
        teacher_temp=0.07,
        warmup_teacher_temp2=0.07,
        teacher_temp2=0.07,
        warmup_teacher_temp_epochs=0,
        nepochs=1,
    )


class PretrainedCheckpointTest(unittest.TestCase):
    def test_loads_model_weights_and_pretrained_centers(self):
        source_student = nn.Linear(2, 2)
        source_teacher = nn.Linear(2, 2)
        source_loss = make_loss()
        source_loss.center.fill_(1.25)
        source_loss.center2.fill_(-0.75)
        checkpoint = {
            "student": source_student.state_dict(),
            "teacher": source_teacher.state_dict(),
            "ibot_loss": source_loss.state_dict(),
        }

        student = nn.Linear(2, 2)
        teacher = nn.Linear(2, 2)
        loss = make_loss()
        load_pretrained_state(checkpoint, student, teacher, loss)

        for expected, actual in zip(
            source_student.parameters(), student.parameters()
        ):
            torch.testing.assert_close(actual, expected)
        for expected, actual in zip(
            source_teacher.parameters(), teacher.parameters()
        ):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(loss.center, source_loss.center)
        torch.testing.assert_close(loss.center2, source_loss.center2)

    def test_rejects_checkpoint_without_loss_state(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pth"
            torch.save(
                {"student": {}, "teacher": {}, "epoch": 800},
                checkpoint_path,
            )
            args = SimpleNamespace(initial_checkpoint=checkpoint_path)

            with self.assertRaisesRegex(ValueError, "official full checkpoint"):
                read_pretrained_checkpoint(args)


class RecordingScaler:
    def __init__(self):
        self.loaded_state = None

    def load_state_dict(self, state):
        self.loaded_state = state


class ResumeCheckpointTest(unittest.TestCase):
    def _make_checkpoint(self):
        student = nn.Linear(2, 2)
        teacher = nn.Linear(2, 2)
        loss = make_loss()
        optimizer = torch.optim.AdamW(student.parameters(), lr=0.0123)
        student(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        return {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": 27,
            "args": SimpleNamespace(
                epochs=50,
                lambda3=0.2,
                use_fp16=True,
                source_checkpoint_epoch=800,
            ),
            "ibot_loss": loss.state_dict(),
            "fp16_scaler": {"scale": 4096.0},
        }

    def test_restores_full_training_state_and_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pth"
            torch.save(self._make_checkpoint(), checkpoint_path)
            args = SimpleNamespace(
                resume_checkpoint=checkpoint_path,
                epochs=50,
                lambda3=0.2,
                use_fp16=True,
            )

            checkpoint = read_resume_checkpoint(args)
            student = nn.Linear(2, 2)
            teacher = nn.Linear(2, 2)
            loss = make_loss()
            optimizer = torch.optim.AdamW(student.parameters(), lr=1.0)
            scaler = RecordingScaler()
            start_epoch = load_resume_state(
                checkpoint,
                student,
                teacher,
                loss,
                optimizer,
                scaler,
            )

            self.assertEqual(start_epoch, 27)
            self.assertEqual(args.resume_epoch, 27)
            self.assertEqual(args.source_checkpoint_epoch, 800)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0123)
            self.assertTrue(optimizer.state)
            self.assertEqual(scaler.loaded_state, {"scale": 4096.0})

    def test_rejects_changed_resume_hyperparameters(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pth"
            torch.save(self._make_checkpoint(), checkpoint_path)
            args = SimpleNamespace(
                resume_checkpoint=checkpoint_path,
                epochs=50,
                lambda3=0.0,
                use_fp16=True,
            )

            with self.assertRaisesRegex(ValueError, "lambda3"):
                read_resume_checkpoint(args)


if __name__ == "__main__":
    unittest.main()
