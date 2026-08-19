# CRISP frozen-backbone evaluation

This folder evaluates one frozen iBOT/region-loss checkpoint with the protocols
reported by CRISP. The default checkpoint entry is `teacher`; the architecture
is inferred from the checkpoint and may be `vit_small`, `vit_base`, or
`vit_large` with patch size 16.

The public entry point is `full-evaluation`. Protocol implementations, dataset
adapters, distributed helpers, and the orchestration code are kept in
`evaluation/utils/`.

## Protocols

Dense segmentation follows CRISP Appendix A.2 and the official CAPI
segmentation evaluator:

- one GPU and 256 x 256 inputs, giving 256 ViT-S/16 patch tokens;
- final normalized teacher patch tokens with the backbone frozen;
- a seeded 90/10 split of the training set for hyperparameter selection;
- `StandardScaler` fit on the training features only;
- k-NN sweep over `k = {1, 3, 10, 30}` and cosine/L2 distance;
- cuML L-BFGS logistic-regression sweep over
  `C = 10 ** linspace(-6, 5, 8)`;
- pixel mIoU and pixel accuracy on the official validation split.

ImageNet-1K 100% k-NN follows the original iBOT weighted k-NN evaluator:

- final teacher CLS token, L2 normalized;
- `k = {10, 20, 100, 200}`, temperature 0.07;
- top-1 and top-5 accuracy, with `k=20` used as the primary table value.

ImageNet-1K 100% linear probing follows CRISP Appendix A.2 and iBOT's feature
convention:

- four GPUs, 224 x 224 inputs, batch size 256 per GPU, 200 epochs;
- base LR 0.001 with iBOT's linear batch scaling, SGD momentum 0.9, no weight
  decay, and cosine decay;
- the final four CLS tokens concatenated for ViT-S; final CLS plus mean patch
  token for ViT-B/L;
- top-1, top-5, loss, training history, and a resumable probe checkpoint.

The implementation is pinned against CAPI commit
[`98b4fa17ee8eec8810c17022df9a27a44845368b`](https://github.com/facebookresearch/capi/tree/98b4fa17ee8eec8810c17022df9a27a44845368b)
and iBOT commit
[`da316d82636a7a7356835ef224b13d5f3ace0489`](https://github.com/bytedance/ibot/tree/da316d82636a7a7356835ef224b13d5f3ace0489).

There is one reproducibility boundary in the available CRISP material. CRISP
says it uses CAPI without modification, but the public CAPI snapshot contains
no Cityscapes loader and marks its experimental VOC loader as not reproducing
CAPI's paper result. CRISP's private dataset wiring is not present in the PDF.
The scripts therefore keep the published CAPI evaluator behavior unchanged and
add deterministic local dataset adapters: VOC uses VOC2012 `train` plus SBD
`train` and `val` masks, and Cityscapes labels are converted to the official 19
train IDs. This is the closest reproducible protocol supported by the released
sources; it should not be described as byte-for-byte use of an unavailable
private loader.

## Environment

For exact dependency versions, create a separate Python 3.11 evaluation
environment:

```bash
python3.11 -m venv .venv-evaluation
source .venv-evaluation/bin/activate
python -m pip install --upgrade pip
python -m pip install -r evaluation/requirements.txt
```

RAPIDS cuML is required only by the three dense linear scripts. The k-NN and
ImageNet scripts do not import cuML. The separate environment uses Python 3.11,
CUDA 12.1 PyTorch, and the versions from the pinned public CAPI setup without
changing the training environment.

## Expected dataset layouts

The resolvers accept common capitalization/nesting variants below `datasets/`.
The canonical layouts are:

```text
datasets/
  imagenet/
    train/<class>/*.JPEG
    val/<class>/*.JPEG
  ADE20K/ADEChallengeData2016/
    images/{training,validation}/*.jpg
    annotations/{training,validation}/*.png
  VOCdevkit/VOC2012/
    JPEGImages/
    SegmentationClass/
    ImageSets/Segmentation/{train,val}.txt
  benchmark_RELEASE/dataset/
    img/*.jpg
    cls/*.mat
    {train,val}.txt
  cityscapes/
    leftImg8bit/{train,val}/<city>/*.png
    gtFine/{train,val}/<city>/*_gtFine_labelIds.png
```

## Evaluation components

The full runner invokes the components in `evaluation/utils/`. They can also be
run separately for debugging. Each command loads the checkpoint and writes a
complete JSON result. Dense components use one GPU. ImageNet components
automatically relaunch themselves with four processes and therefore require at
least four visible GPUs.

```bash
python -m evaluation.utils.pascal_voc_knn CHECKPOINT
python -m evaluation.utils.pascal_voc_linear CHECKPOINT
python -m evaluation.utils.ade20k_knn CHECKPOINT
python -m evaluation.utils.ade20k_linear CHECKPOINT
python -m evaluation.utils.cityscapes_knn CHECKPOINT
python -m evaluation.utils.cityscapes_linear CHECKPOINT
python -m evaluation.utils.imagenet_knn CHECKPOINT
python -m evaluation.utils.imagenet_linear CHECKPOINT
```

Useful shared options are `--checkpoint-key`, `--arch`, `--datasets-root`,
`--output-dir`, `--result-json`, `--num-workers`, and `--seed`. Dense scripts
also accept `--feature-cache`; the full runner supplies this automatically so
k-NN and linear evaluation reuse the same extracted patch features.

## Full evaluation

Expose four GPUs, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 evaluation/full-evaluation CHECKPOINT
```

The runner displays an eight-stage `tqdm` progress bar while each child script
shows its own extraction/training progress. It creates a unique run directory
under `output/evaluation/`, preserves partial results after a failure, and
writes the combined JSON table to `full_evaluation.json`.

On Magerit, submit the complete suite on one four-A100 node from the repository
root:

```bash
sbatch slurm_evaluation.sh CHECKPOINT
```

The optional second and third arguments override the datasets root and run
output directory:

```bash
sbatch slurm_evaluation.sh CHECKPOINT /path/to/datasets /path/to/output
```

The job uses `.venv-evaluation` by default. Set `EVALUATION_VENV` when the
environment lives elsewhere. Its terminal output is written to
`logs/ibot_eval_<job-id>.out`, and evaluation results go to the supplied output
directory or `output/evaluation/slurm-<job-id>/`.

For a ViT-S/16 original iBOT teacher, CRISP reports these reference results:

| Dataset | k-NN | Linear |
| --- | ---: | ---: |
| ADE20K mIoU | 22.0 | 27.0 |
| PASCAL VOC mIoU | 49.7 | 58.0 |
| Cityscapes mIoU | 33.5 | 37.9 |
| ImageNet-1K top-1 | 75.1 | 77.9 |

Validate the evaluator with that checkpoint before comparing the continued
iBOT and region-loss checkpoints.
