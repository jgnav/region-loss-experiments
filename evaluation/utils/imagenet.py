import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms as T
from tqdm.auto import tqdm

from evaluation.utils.common import (
    base_parser,
    cleanup_distributed,
    initialize_distributed,
    is_main_process,
    launch_distributed_if_needed,
    load_backbone,
    prepare_paths,
    utc_now,
    write_json,
)


IMAGENET_NORMALIZE = T.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)
GPU_COUNT = 4
BATCH_SIZE_PER_GPU = 256


def _resolve_imagenet_root(datasets_root):
    candidates = [
        datasets_root / "imagenet",
        datasets_root / "ImageNet",
        datasets_root / "imagenet-1k",
        datasets_root / "ILSVRC2012",
        datasets_root,
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find ImageNet-1K train/ and val/ directories below "
        f"{datasets_root}"
    )


class IndexedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, index


def _eval_transform():
    return T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            IMAGENET_NORMALIZE,
        ]
    )


def _train_transform():
    return T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            IMAGENET_NORMALIZE,
        ]
    )


def _feature_vector(model, images, architecture, knn=False):
    if knn:
        return model.get_intermediate_layers(images, n=1)[0][:, 0]
    if architecture == "vit_small":
        outputs = model.get_intermediate_layers(images, n=4)
        return torch.cat([output[:, 0] for output in outputs], dim=-1)
    outputs = model.get_intermediate_layers(images, n=1)
    return torch.cat(
        [outputs[-1][:, 0], outputs[-1][:, 1:].mean(dim=1)], dim=-1
    )


@torch.inference_mode()
def _extract_distributed_features(model, dataset, args, description):
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=BATCH_SIZE_PER_GPU,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    features = None
    labels = None
    iterator = tqdm(
        loader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        disable=not is_main_process(),
    )
    for images, batch_labels, indices in iterator:
        images = images.cuda(non_blocking=True)
        batch_labels = batch_labels.cuda(non_blocking=True)
        indices = indices.cuda(non_blocking=True)
        batch_features = _feature_vector(
            model, images, args.architecture, knn=True
        ).float()
        gathered_features = [
            torch.empty_like(batch_features) for _ in range(dist.get_world_size())
        ]
        gathered_labels = [
            torch.empty_like(batch_labels) for _ in range(dist.get_world_size())
        ]
        gathered_indices = [
            torch.empty_like(indices) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered_features, batch_features)
        dist.all_gather(gathered_labels, batch_labels)
        dist.all_gather(gathered_indices, indices)
        if is_main_process():
            all_features = torch.cat(gathered_features)
            all_labels = torch.cat(gathered_labels)
            all_indices = torch.cat(gathered_indices)
            if features is None:
                features = torch.empty(
                    len(dataset), all_features.shape[-1], device="cuda"
                )
                labels = torch.empty(
                    len(dataset), dtype=all_labels.dtype, device="cuda"
                )
            features.index_copy_(0, all_indices, all_features)
            labels.index_copy_(0, all_indices, all_labels)
    return features, labels


@torch.inference_mode()
def _weighted_knn(
    train_features,
    train_labels,
    test_features,
    test_labels,
    neighbors,
    temperature=0.07,
    num_classes=1000,
):
    train_features = train_features.T
    top1 = 0.0
    top5 = 0.0
    total = 0
    images_per_chunk = max(1, test_labels.shape[0] // 100)
    ranges = range(0, test_labels.shape[0], images_per_chunk)
    one_hot = torch.zeros(neighbors, num_classes, device="cuda")
    for start in tqdm(
        ranges,
        desc=f"ImageNet weighted k-NN k={neighbors}",
        unit="chunk",
        dynamic_ncols=True,
    ):
        feature = test_features[start : start + images_per_chunk]
        target = test_labels[start : start + images_per_chunk]
        similarity = feature @ train_features
        distances, indices = similarity.topk(neighbors, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(target.shape[0], -1)
        retrieved = torch.gather(candidates, 1, indices)
        one_hot.resize_(target.shape[0] * neighbors, num_classes).zero_()
        one_hot.scatter_(1, retrieved.reshape(-1, 1), 1)
        weights = distances.div(temperature).exp()
        probabilities = torch.sum(
            one_hot.view(target.shape[0], neighbors, num_classes)
            * weights.view(target.shape[0], neighbors, 1),
            dim=1,
        )
        predictions = probabilities.argsort(dim=1, descending=True)
        correct = predictions.eq(target.view(-1, 1))
        top1 += correct[:, :1].sum().item()
        top5 += correct[:, :5].sum().item()
        total += target.shape[0]
    return {
        "top1": 100.0 * top1 / total,
        "top5": 100.0 * top5 / total,
    }


def run_imagenet_knn(args):
    started = utc_now()
    start_time = time.monotonic()
    model, metadata = load_backbone(
        args.checkpoint, args.checkpoint_key, args.arch
    )
    args.architecture = metadata["architecture"]
    model.cuda().eval()
    root = _resolve_imagenet_root(args.datasets_root)
    train_dataset = IndexedImageFolder(root / "train", transform=_eval_transform())
    val_dataset = IndexedImageFolder(root / "val", transform=_eval_transform())
    print(
        f"ImageNet loaded: {len(train_dataset)} train, {len(val_dataset)} val",
        flush=True,
    )
    train_features, train_labels = _extract_distributed_features(
        model, train_dataset, args, "ImageNet train features"
    )
    test_features, test_labels = _extract_distributed_features(
        model, val_dataset, args, "ImageNet val features"
    )
    result = None
    if is_main_process():
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)
        evaluations = {}
        for neighbors in (10, 20, 100, 200):
            evaluations[str(neighbors)] = _weighted_knn(
                train_features,
                train_labels,
                test_features,
                test_labels,
                neighbors,
            )
        result = {
            "evaluation": "imagenet_knn",
            "dataset": "ImageNet-1K 100%",
            "status": "completed",
            "started_at": started,
            "finished_at": utc_now(),
            "elapsed_seconds": time.monotonic() - start_time,
            "model": metadata,
            "dataset_sizes": {
                "train": len(train_dataset),
                "test": len(val_dataset),
            },
            "protocol": {
                "source": "CRISP Table 4 / original iBOT weighted k-NN",
                "input_resolution": 224,
                "training_fraction": 1.0,
                "feature": "final normalized teacher CLS token",
                "feature_l2_normalization": True,
                "temperature": 0.07,
                "neighbors": [10, 20, 100, 200],
                "primary_neighbors": 20,
                "gpu_count": GPU_COUNT,
                "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            },
            "metrics": evaluations["20"],
            "metrics_by_neighbors": evaluations,
        }
        write_json(args.result_json, result)
        print(
            f"ImageNet 100% k-NN (k=20): top-1={result['metrics']['top1']:.3f}, "
            f"top-5={result['metrics']['top5']:.3f}",
            flush=True,
        )
        print(f"Saved results to {args.result_json}", flush=True)
    dist.barrier()
    return result


class LinearClassifier(nn.Module):
    def __init__(self, input_dimension, num_classes=1000):
        super().__init__()
        self.linear = nn.Linear(input_dimension, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, features):
        return self.linear(features.flatten(1))


def _reduce_average(total, count):
    values = torch.tensor([total, count], dtype=torch.float64, device="cuda")
    dist.all_reduce(values)
    return (values[0] / values[1]).item()


def _train_linear_epoch(model, classifier, loader, optimizer, architecture):
    model.eval()
    classifier.train()
    loss_total = 0.0
    sample_count = 0
    for images, targets in loader:
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        with torch.no_grad():
            features = _feature_vector(model, images, architecture, knn=False)
        prediction = classifier(features)
        loss = F.cross_entropy(prediction, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_total += loss.item() * images.shape[0]
        sample_count += images.shape[0]
    return _reduce_average(loss_total, sample_count)


@torch.inference_mode()
def _validate_linear(model, classifier, loader, architecture):
    model.eval()
    classifier.eval()
    loss_total = 0.0
    top1 = 0
    top5 = 0
    sample_count = 0
    for images, targets in tqdm(
        loader,
        desc="ImageNet linear validation",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    ):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        features = _feature_vector(model, images, architecture, knn=False)
        prediction = classifier(features)
        loss_total += F.cross_entropy(prediction, targets).item() * images.shape[0]
        correct = prediction.topk(5, dim=1).indices.eq(targets[:, None])
        top1 += correct[:, :1].sum().item()
        top5 += correct[:, :5].sum().item()
        sample_count += images.shape[0]
    return {
        "loss": loss_total / sample_count,
        "top1": 100.0 * top1 / sample_count,
        "top5": 100.0 * top5 / sample_count,
    }


def run_imagenet_linear(args):
    started = utc_now()
    start_time = time.monotonic()
    model, metadata = load_backbone(
        args.checkpoint, args.checkpoint_key, args.arch
    )
    architecture = metadata["architecture"]
    model.cuda().eval()
    root = _resolve_imagenet_root(args.datasets_root)
    train_dataset = datasets.ImageFolder(root / "train", transform=_train_transform())
    val_dataset = datasets.ImageFolder(root / "val", transform=_eval_transform())
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=BATCH_SIZE_PER_GPU,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = None
    if is_main_process():
        validation_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE_PER_GPU,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )
    feature_dimension = 4 * model.embed_dim if architecture == "vit_small" else 2 * model.embed_dim
    classifier = LinearClassifier(feature_dimension).cuda()
    classifier = DistributedDataParallel(
        classifier, device_ids=[int(torch.cuda.current_device())]
    )
    base_learning_rate = 0.001
    scaled_learning_rate = (
        base_learning_rate * BATCH_SIZE_PER_GPU * dist.get_world_size() / 256.0
    )
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=scaled_learning_rate,
        momentum=0.9,
        weight_decay=0.0,
    )
    epochs = 200
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=0
    )
    state_path = args.output_dir / "linear_probe_checkpoint.pth"
    best_state_path = args.output_dir / "linear_probe_best.pth"
    start_epoch = 0
    best_metrics = {"top1": 0.0, "top5": 0.0, "loss": float("inf")}
    best_epoch = 0
    history = []
    if args.resume and state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("checkpoint_fingerprint") != metadata["checkpoint_fingerprint"]:
            raise ValueError(
                f"Linear-probe resume state belongs to another backbone: {state_path}"
            )
        classifier.module.load_state_dict(state["classifier"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"]
        best_metrics = state["best_metrics"]
        best_epoch = state.get("best_epoch", start_epoch)
        history = state.get("history", [])
        print(f"Resuming ImageNet linear probe at epoch {start_epoch}", flush=True)

    epoch_progress = tqdm(
        range(start_epoch, epochs),
        desc="ImageNet linear probe",
        unit="epoch",
        dynamic_ncols=True,
        disable=not is_main_process(),
    )
    for epoch in epoch_progress:
        train_sampler.set_epoch(epoch)
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        train_loss = _train_linear_epoch(
            model, classifier, train_loader, optimizer, architecture
        )
        scheduler.step()
        dist.barrier()
        metrics = None
        if is_main_process():
            metrics = _validate_linear(
                model, classifier.module, validation_loader, architecture
            )
        objects = [metrics]
        dist.broadcast_object_list(objects, src=0)
        metrics = objects[0]
        record = {
            "epoch": epoch + 1,
            "learning_rate": epoch_learning_rate,
            "train_loss": train_loss,
            **metrics,
        }
        history.append(record)
        is_best = metrics["top1"] >= best_metrics["top1"]
        if is_best:
            best_metrics = dict(metrics)
            best_epoch = epoch + 1
        if is_main_process():
            epoch_progress.set_postfix(
                train_loss=f"{train_loss:.4f}",
                top1=f"{metrics['top1']:.2f}",
                best=f"{best_metrics['top1']:.2f}",
            )
            torch.save(
                {
                    "checkpoint_fingerprint": metadata["checkpoint_fingerprint"],
                    "epoch": epoch + 1,
                    "classifier": classifier.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_metrics": best_metrics,
                    "best_epoch": best_epoch,
                    "history": history,
                },
                state_path,
            )
            if is_best:
                torch.save(
                    {
                        "checkpoint_fingerprint": metadata[
                            "checkpoint_fingerprint"
                        ],
                        "epoch": best_epoch,
                        "classifier": classifier.module.state_dict(),
                        "metrics": best_metrics,
                    },
                    best_state_path,
                )
        dist.barrier()

    result = None
    if is_main_process():
        result = {
            "evaluation": "imagenet_linear",
            "dataset": "ImageNet-1K 100%",
            "status": "completed",
            "started_at": started,
            "finished_at": utc_now(),
            "elapsed_seconds": time.monotonic() - start_time,
            "model": metadata,
            "dataset_sizes": {
                "train": len(train_dataset),
                "test": len(val_dataset),
            },
            "protocol": {
                "source": "CRISP Appendix A.2 with original iBOT feature convention",
                "input_resolution": 224,
                "training_fraction": 1.0,
                "backbone_frozen": True,
                "feature": (
                    "concatenated CLS tokens from final 4 blocks"
                    if architecture == "vit_small"
                    else "final CLS token concatenated with mean patch token"
                ),
                "optimizer": "SGD",
                "momentum": 0.9,
                "weight_decay": 0.0,
                "lr_schedule": "cosine",
                "base_learning_rate": base_learning_rate,
                "scaled_learning_rate": scaled_learning_rate,
                "epochs": epochs,
                "gpu_count": GPU_COUNT,
                "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
                "effective_batch_size": BATCH_SIZE_PER_GPU * GPU_COUNT,
            },
            "metrics": best_metrics,
            "best_epoch": best_epoch,
            "final_metrics": history[-1],
            "history": history,
        }
        write_json(args.result_json, result)
        print(
            f"ImageNet linear probe: best top-1={best_metrics['top1']:.3f}, "
            f"top-5={best_metrics['top5']:.3f}",
            flush=True,
        )
        print(f"Saved results to {args.result_json}", flush=True)
    dist.barrier()
    return result


def imagenet_entrypoint(module, mode):
    launch_distributed_if_needed(module, required_world_size=GPU_COUNT)
    parser = base_parser(f"CRISP ImageNet-1K 100% {mode} evaluation")
    if mode == "linear":
        parser.add_argument(
            "--no-resume",
            action="store_false",
            dest="resume",
            help="Do not resume an incomplete probe in the output directory",
        )
        parser.set_defaults(resume=True)
    args = prepare_paths(parser.parse_args(), f"imagenet_{mode}")
    initialize_distributed(args.seed, allow_tf32=False)
    try:
        if mode == "knn":
            return run_imagenet_knn(args)
        if mode == "linear":
            return run_imagenet_linear(args)
        raise ValueError(f"Unknown ImageNet evaluation mode: {mode}")
    finally:
        cleanup_distributed()
