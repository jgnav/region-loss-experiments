from collections.abc import Mapping

import torch


def read_full_checkpoint(args):
    if not args.initial_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.initial_checkpoint}")

    checkpoint = torch.load(
        args.initial_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("The checkpoint root must be a mapping")

    required = {"student", "teacher", "optimizer", "epoch", "ibot_loss"}
    if args.use_fp16:
        required.add("fp16_scaler")
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(
            f"The full official iBOT checkpoint is missing keys: {missing}"
        )
    if not isinstance(checkpoint["epoch"], int):
        raise ValueError("The checkpoint epoch must be an integer")

    args.start_epoch = checkpoint["epoch"]
    args.epochs = args.start_epoch + args.epoch
    return checkpoint


def load_training_state(
    checkpoint,
    student,
    teacher,
    optimizer,
    fp16_scaler,
    ibot_loss,
):
    objects = {
        "student": student,
        "teacher": teacher,
        "ibot_loss": ibot_loss,
    }
    for key, value in objects.items():
        incompatible = value.load_state_dict(checkpoint[key], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                f"Checkpoint key '{key}' is incompatible: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    optimizer.load_state_dict(checkpoint["optimizer"])
    if fp16_scaler is not None:
        fp16_scaler.load_state_dict(checkpoint["fp16_scaler"])
