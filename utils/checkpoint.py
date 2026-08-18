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

    required = {"student", "teacher", "ibot_loss"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(
            "The iBOT checkpoint is missing required training state: "
            f"{missing}. Use the official full checkpoint, not a "
            "backbone-only checkpoint."
        )
    if not isinstance(checkpoint["ibot_loss"], Mapping):
        raise ValueError("The checkpoint key 'ibot_loss' must be a mapping")
    missing_centers = sorted(
        {"center", "center2"} - set(checkpoint["ibot_loss"])
    )
    if missing_centers:
        raise ValueError(
            "The iBOT loss state is missing pretrained centers: "
            f"{missing_centers}"
        )
    if "epoch" in checkpoint and not isinstance(checkpoint["epoch"], int):
        raise ValueError("The checkpoint epoch must be an integer")

    args.source_checkpoint_epoch = checkpoint.get("epoch")
    return checkpoint


def load_pretrained_state(
    checkpoint,
    student,
    teacher,
    ibot_loss,
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

    center_state = {
        key: checkpoint["ibot_loss"][key] for key in ("center", "center2")
    }
    incompatible = ibot_loss.load_state_dict(center_state, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if key in {"center", "center2"}
    ]
    if missing or incompatible.unexpected_keys:
        raise ValueError(
            "Checkpoint iBOT centers are incompatible: "
            f"missing={missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
