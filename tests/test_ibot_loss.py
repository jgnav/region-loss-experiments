import unittest

import numpy as np

from losses import iBOTLoss


def make_loss(**overrides):
    arguments = {
        "out_dim": 3,
        "patch_out_dim": 3,
        "ngcrops": 2,
        "nlcrops": 0,
        "warmup_teacher_temp": 0.07,
        "teacher_temp": 0.07,
        "warmup_teacher_temp2": 0.07,
        "teacher_temp2": 0.07,
        "warmup_teacher_temp_epochs": 0,
        "nepochs": 8,
        "lambda3": 0.1,
        "region_warmup_epochs": 5,
    }
    arguments.update(overrides)
    return iBOTLoss(**arguments)


class AdaptationScheduleTest(unittest.TestCase):
    def test_teacher_temperatures_remain_at_pretrained_value(self):
        loss = make_loss()

        np.testing.assert_allclose(loss.teacher_temp_schedule, 0.07)
        np.testing.assert_allclose(loss.teacher_temp2_schedule, 0.07)

    def test_region_weight_reaches_target_during_fifth_epoch(self):
        loss = make_loss()

        np.testing.assert_allclose(
            loss.region_weight_schedule,
            [0.0, 0.025, 0.05, 0.075, 0.1, 0.1, 0.1, 0.1],
        )

    def test_region_warmup_cannot_exceed_training_length(self):
        with self.assertRaisesRegex(ValueError, "region_warmup_epochs"):
            make_loss(region_warmup_epochs=9)


if __name__ == "__main__":
    unittest.main()
