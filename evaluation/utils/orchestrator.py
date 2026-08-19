import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.utils.common import (
    ARCHITECTURES,
    REPO_ROOT,
    checkpoint_fingerprint,
    utc_now,
    write_json,
)


EVALUATIONS = (
    ("pascal_voc_knn", "evaluation.utils.pascal_voc_knn", "pascal_voc"),
    ("pascal_voc_linear", "evaluation.utils.pascal_voc_linear", "pascal_voc"),
    ("imagenet_knn", "evaluation.utils.imagenet_knn", None),
    ("ade20k_knn", "evaluation.utils.ade20k_knn", "ade20k"),
    ("ade20k_linear", "evaluation.utils.ade20k_linear", "ade20k"),
    ("cityscapes_knn", "evaluation.utils.cityscapes_knn", "cityscapes"),
    ("cityscapes_linear", "evaluation.utils.cityscapes_linear", "cityscapes"),
)


def _parser():
    parser = argparse.ArgumentParser(
        description="Run the complete CRISP evaluation suite for one checkpoint"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-key", default="teacher", choices=("teacher", "student")
    )
    parser.add_argument(
        "--arch", default="auto", choices=("auto", *ARCHITECTURES)
    )
    parser.add_argument(
        "--datasets-root", type=Path, default=REPO_ROOT / "dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run directory; defaults to a unique directory in output/evaluation",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="Final table path; defaults to <output-dir>/full_evaluation.json",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-_") or "checkpoint"


def _result_table(results):
    table = []
    pairs = (
        ("PASCAL VOC 2012", "pascal_voc_knn", "pascal_voc_linear"),
        ("ImageNet-1K 10%", "imagenet_knn", None),
        ("ADE20K", "ade20k_knn", "ade20k_linear"),
        ("Cityscapes", "cityscapes_knn", "cityscapes_linear"),
    )
    for dataset, knn_name, linear_name in pairs:
        row = {"dataset": dataset}
        if knn_name in results:
            row["knn"] = results[knn_name].get("metrics", {})
        if linear_name is not None and linear_name in results:
            row["linear"] = results[linear_name].get("metrics", {})
        table.append(row)
    return table


def _write_summary(path, args, started_at, status, results, error=None):
    model = None
    if results:
        model = next(iter(results.values())).get("model")
    updated_at = utc_now()
    summary = {
        "status": status,
        "checkpoint": str(args.checkpoint),
        "checkpoint_key": args.checkpoint_key,
        "architecture": args.arch,
        "model": model,
        "datasets_root": str(args.datasets_root),
        "started_at": started_at,
        "updated_at": updated_at,
        "completed_evaluations": list(results),
        "evaluations": results,
        "table": _result_table(results),
    }
    if status in {"completed", "failed"}:
        summary["finished_at"] = updated_at
    if error is not None:
        summary["error"] = error
    write_json(path, summary)


def _load_completed_result(path, args, evaluation_name):
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    model = result.get("model", {})
    if (
        result.get("status") != "completed"
        or result.get("evaluation") != evaluation_name
        or model.get("checkpoint_fingerprint")
        != checkpoint_fingerprint(args.checkpoint)
        or model.get("checkpoint_key") != args.checkpoint_key
    ):
        return None
    return result


def main(argv=None):
    args = _parser().parse_args(argv)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.datasets_root = args.datasets_root.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.datasets_root.is_dir():
        raise FileNotFoundError(f"Datasets directory not found: {args.datasets_root}")
    if args.output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output_dir = (
            REPO_ROOT
            / "output"
            / "evaluation"
            / f"{_safe_name(args.checkpoint.stem)}-{timestamp}"
        )
    else:
        args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.result_json is None:
        args.result_json = args.output_dir / "full_evaluation.json"
    else:
        args.result_json = args.result_json.expanduser().resolve()
    started_at = utc_now()
    results = {}
    _write_summary(
        args.result_json, args, started_at, "running", results
    )

    print(f"Starting {len(EVALUATIONS)} CRISP evaluations", flush=True)
    for evaluation_index, (name, module, cache_name) in enumerate(
        EVALUATIONS, start=1
    ):
        result_path = args.output_dir / f"{name}.json"
        completed_result = _load_completed_result(result_path, args, name)
        if completed_result is not None:
            results[name] = completed_result
            _write_summary(
                args.result_json, args, started_at, "running", results
            )
            print(
                f"[{evaluation_index}/{len(EVALUATIONS)}] "
                f"Reusing completed {name}",
                flush=True,
            )
            continue
        print(
            f"\n[{evaluation_index}/{len(EVALUATIONS)}] Starting {name}",
            flush=True,
        )
        command = [
            sys.executable,
            "-m",
            module,
            str(args.checkpoint),
            "--checkpoint-key",
            args.checkpoint_key,
            "--arch",
            args.arch,
            "--datasets-root",
            str(args.datasets_root),
            "--output-dir",
            str(args.output_dir),
            "--result-json",
            str(result_path),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
        ]
        if cache_name is not None:
            cache_path = args.output_dir / "feature_cache" / f"{cache_name}.pth"
            command.extend(("--feature-cache", str(cache_path)))
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            error = f"{name} exited with status {completed.returncode}"
            _write_summary(
                args.result_json,
                args,
                started_at,
                "failed",
                results,
                error=error,
            )
            print(f"Full evaluation stopped: {error}", flush=True)
            return completed.returncode
        try:
            with result_path.open("r", encoding="utf-8") as handle:
                results[name] = json.load(handle)
        except (OSError, json.JSONDecodeError) as error_object:
            error = f"{name} did not produce a valid result JSON: {error_object}"
            _write_summary(
                args.result_json,
                args,
                started_at,
                "failed",
                results,
                error=error,
            )
            print(f"Full evaluation stopped: {error}", flush=True)
            return 1
        _write_summary(
            args.result_json, args, started_at, "running", results
        )
        print(f"Completed {name}", flush=True)

    _write_summary(
        args.result_json, args, started_at, "completed", results
    )
    print(
        f"Full evaluation completed. Result table: {args.result_json}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
