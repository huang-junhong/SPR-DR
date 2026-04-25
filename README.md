# SPR

Official research code for SPR-based image super-resolution training.

This repository is a cleaned public release extracted from the experiment workspace. The main entrypoints are:

- `scripts/train_spr.py`: SPR training with two rankers.
- `scripts/train_spr_gc.py`: SPR training with gradient accumulation.
- `scripts/test_spr.py`: checkpoint evaluation and SR image export.

## Installation

Create an environment with Python 3.10 or newer, then install the dependencies:

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from the official PyTorch instructions if the default wheel is not appropriate for your machine.

## Data Format

For `data_config.mode: synthetic`, each training dataset directory must contain:

```text
dataset_root/
  LR_PATH.txt
  HR_PATH.txt
```

Each text file contains one image path per line. Lines in `LR_PATH.txt` and `HR_PATH.txt` must be paired by index.

For validation, each test set root must contain paired subfolders:

```text
val_root/
  LR/
  HR/
```

For `data_config.mode: mri`, each dataset directory must contain `LR2_PATH.txt`, `LR4_PATH.txt`, and `HR_PATH.txt`.

## Training

Edit `configs/train_spr_example.yaml` to point to your datasets and output directory, then run:

```bash
python scripts/train_spr.py --config configs/train_spr_example.yaml
```

For gradient accumulation:

```bash
python scripts/train_spr_gc.py --config configs/train_spr_example.yaml
```

Convenience launchers are also provided:

```bash
bash scripts/train.sh configs/train_spr_example.yaml
bash scripts/train_gc.sh configs/train_spr_example.yaml
```

Outputs are written to `validation.save_root`, including copied config files, TensorBoard logs, and model checkpoints under `models/`.

TensorBoard logging is enabled by default. During training, the scripts record:

- `train/loss/*`: epoch averages for generator and discriminator losses.
- `train/loss_iter/*`: iteration-level losses every `tensorboard.log_interval` steps.
- `train/lr/*`: current learning rates.
- `train/time/epoch_seconds`: epoch wall-clock time.
- `validation/<dataset>/*`: validation metrics.
- `config/train_yaml`: the full YAML config used for the run.

Launch TensorBoard with:

```bash
tensorboard --logdir outputs/spr_experiment
```

## Testing

Testing uses a separate config file. Edit `configs/test_spr_example.yaml` to point to a generator checkpoint and validation dataset, then run:

```bash
python scripts/test_spr.py --config configs/test_spr_example.yaml
```

or:

```bash
bash scripts/test.sh configs/test_spr_example.yaml
```

Before testing, set `checkpoint.path` to an existing generator checkpoint. For example:

```yaml
checkpoint:
  path: ./checkpoints/G.pth
```

Pretrained model filenames, generator names, and checksums are listed in `docs/PRETRAINED_MODELS.md`.

or point it to a checkpoint saved by training:

```yaml
checkpoint:
  path: ./outputs/spr_experiment/models/1/G.pth
```

The test script saves super-resolved images to `test.output_dir` and writes `metrics.csv` when `test.save_csv` is true. Each dataset can use either:

```yaml
test:
  datasets:
    - name: Set5
      root: ./datasets/val/Set5/SRF4
```

where the script expects `LR/` and `HR/` inside `root`, or explicit folders:

```yaml
test:
  datasets:
    - name: Custom
      lr_dir: ./datasets/lr
      hr_dir: ./datasets/hr
```

If `HR` images are absent, only no-reference metrics such as `niqe` are computed.

## Configuration Notes

- `generator_config.generator` supports `srres`, `rrdn`, `real-rrdn`, `swinir-m`, and `swinir-s`.
- `generator_config.pretrain_path` and `discriminator_config.pretrain_path` can be `null` or a checkpoint path.
- `data_config.srf` sets the super-resolution factor.
- `spr_config.mode` supports the modes implemented by `tensor_unit/Patch_Replacement.py`.
- `validation.iqas` are passed to `pyiqa.create_metric`.
- Test configuration is intentionally separate from training configuration: use `checkpoint.path` for the model weights and `test.datasets` for evaluation data.
- Set `tensorboard.log_dir` to override the TensorBoard directory; leave it `null` to use `validation.save_root`.

For SwinIR, use:

```yaml
generator_config:
  generator: swinir-s
  srf: 4
```

Use `swinir-m` for the medium preset and `swinir-s` for the lightweight preset. Advanced SwinIR constructor values can be overridden with `swinir_config`.

## Citation

If this code is useful in your research, please cite 
`
@INPROCEEDINGS{11462166, author={Huang, JunHong and Zang, HangXing and Hu, QingSong and Huang, XingHan and Zhang, Rui}, booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, title={Single Image Super-Resolution with Selective Perceptual Refinement and Distribution-Constancy Ranking}, year={2026}, volume={}, number={}, pages={2646-2650}, keywords={Feedback;Circuits;Pixel;Protocols;HTTP;Digital images;Fuses;Learning (artificial intelligence);Convolutional neural networks;Artificial intelligence;SISR;GAN;Learn-to-Rank;Image hybrid}, doi={10.1109/ICASSP55912.2026.11462166}}
`

## License

This release is provided under the MIT License. See `LICENSE`.
