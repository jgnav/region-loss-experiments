import unittest

import torch

from losses.region_loss import RegionLoss, intersection_patch_weights


class IntersectionPatchWeightsTest(unittest.TestCase):
    def test_full_overlap_covers_every_patch_equally(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0]]]
        )

        weights, valid, area = intersection_patch_weights(boxes, 4, 0.05)

        torch.testing.assert_close(weights, torch.full((1, 2, 4), 0.25))
        self.assertTrue(valid.item())
        torch.testing.assert_close(area, torch.tensor([1.0]))

    def test_intersection_is_projected_into_each_crop(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 0.0], [0.5, 0.0, 1.0, 1.0, 0.0]]]
        )

        weights, valid, area = intersection_patch_weights(boxes, 4, 0.05)

        torch.testing.assert_close(
            weights[:, 0], torch.tensor([[0.0, 0.25, 0.0, 0.25]])
        )
        torch.testing.assert_close(weights[:, 1], torch.full((1, 4), 0.25))
        self.assertTrue(valid.item())
        torch.testing.assert_close(area, torch.tensor([0.5]))

    def test_horizontal_flip_mirrors_patch_coverage(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 1.0], [0.5, 0.0, 1.0, 1.0, 0.0]]]
        )

        weights, _, _ = intersection_patch_weights(boxes, 4, 0.05)

        torch.testing.assert_close(
            weights[:, 0], torch.tensor([[0.25, 0.0, 0.25, 0.0]])
        )

    def test_boundary_patches_receive_fractional_coverage(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 0.0], [0.25, 0.0, 1.0, 1.0, 0.0]]]
        )

        weights, _, _ = intersection_patch_weights(boxes, 4, 0.05)

        torch.testing.assert_close(
            weights[:, 0], torch.tensor([[0.125, 0.25, 0.125, 0.25]])
        )

    def test_overlap_below_threshold_is_skipped(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 0.0], [0.5, 0.0, 1.0, 1.0, 0.0]]]
        )

        weights, valid, _ = intersection_patch_weights(boxes, 4, 0.51)

        self.assertFalse(valid.item())
        self.assertEqual(weights.count_nonzero().item(), 0)

    def test_overlap_at_threshold_is_kept(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 0.0], [0.5, 0.0, 1.0, 1.0, 0.0]]]
        )

        _, valid, _ = intersection_patch_weights(boxes, 4, 0.5)

        self.assertTrue(valid.item())

    def test_zero_area_is_skipped_even_with_zero_threshold(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 0.4, 1.0, 0.0], [0.6, 0.0, 1.0, 1.0, 0.0]]]
        )

        weights, valid, _ = intersection_patch_weights(boxes, 4, 0.0)

        self.assertFalse(valid.item())
        self.assertEqual(weights.count_nonzero().item(), 0)


class RegionLossTest(unittest.TestCase):
    def test_loss_matches_region_across_opposite_views(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0]]]
        )
        teacher = (
            torch.tensor([[[1.0, 0.0]]]).expand(1, 4, 2),
            torch.tensor([[[0.0, 1.0]]]).expand(1, 4, 2),
        )
        matching_student = (
            torch.tensor([[[-10.0, 10.0]]]).expand(1, 4, 2),
            torch.tensor([[[10.0, -10.0]]]).expand(1, 4, 2),
        )
        nonmatching_student = tuple(-logits for logits in matching_student)
        loss_function = RegionLoss(min_area=0.05)

        matching = loss_function(matching_student, teacher, boxes)["loss"]
        nonmatching = loss_function(nonmatching_student, teacher, boxes)["loss"]

        self.assertLess(matching.item(), 1e-6)
        self.assertGreater(nonmatching.item(), 10.0)

    def test_invalid_batch_has_graph_connected_zero_loss(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 0.4, 1.0, 0.0], [0.6, 0.0, 1.0, 1.0, 0.0]]]
        )
        student = tuple(
            torch.randn(1, 4, 3, requires_grad=True) for _ in range(2)
        )
        teacher = tuple(
            torch.softmax(torch.randn(1, 4, 3), dim=-1)
            for _ in range(2)
        )

        result = RegionLoss(min_area=0.0)(student, teacher, boxes)
        result["loss"].backward()

        self.assertEqual(result["loss"].item(), 0.0)
        self.assertEqual(result["valid_ratio"].item(), 0.0)
        for logits in student:
            self.assertEqual(logits.grad.count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
