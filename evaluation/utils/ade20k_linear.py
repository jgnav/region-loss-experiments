import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.utils.dense import dense_entrypoint


if __name__ == "__main__":
    dense_entrypoint(
        "evaluation.utils.ade20k_linear",
        "ade20k",
        "linear",
        "ade20k_linear",
    )
