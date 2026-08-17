from collections.abc import Mapping

import torch


def read_pretrained_checkpoint(args):
    if not args.initial_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.initial_checkpoint}")

    checkpoint = torch.load(
        args.initial_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("The checkpoint root must be a mapping")

    required = {"student", "teacher"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(
            f"The iBOT checkpoint is missing model weights: {missing}"
        )
    if "epoch" in checkpoint and not isinstance(checkpoint["epoch"], int):
        raise ValueError("The checkpoint epoch must be an integer")

    args.source_checkpoint_epoch = checkpoint.get("epoch")
    return checkpoint


def load_pretrained_weights(
    checkpoint,
    student,
    teacher,
):
    objects = {
        "student": student,
        "teacher": teacher,
    }
    for key, value in objects.items():
        incompatible = value.load_state_dict(checkpoint[key], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                f"Checkpoint key '{key}' is incompatible: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
