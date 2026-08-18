import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from losses import iBOTLoss
from utils.checkpoint import load_pretrained_state, read_pretrained_checkpoint


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


if __name__ == "__main__":
    unittest.main()
