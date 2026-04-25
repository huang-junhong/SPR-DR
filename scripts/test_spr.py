import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pyiqa
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["TRAIN-ROOT-PATH"] = str(PROJECT_ROOT)

import file_io
import functional
import torch_model


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
NO_REF_METRICS = {"niqe", "brisque", "musiq", "nima", "maniqa", "clipiqa"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an SPR generator checkpoint.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "test_spr_example.yaml"),
        help="Path to a YAML test config.",
    )
    return parser.parse_args()


def resolve_path(path):
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def collect_image_paths(folder):
    return file_io.base.get_file_path_by_suffix(str(folder), IMAGE_EXTS)


def load_image_tensor(path, device):
    image = file_io.reader.read_image(str(path), as_rgb=True)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    tensor = functional.common.image_handle(
        image,
        normlize="zo",
        expand_dim=0,
        chw=True,
        to_tensor=True,
    )
    return tensor.to(device=device, dtype=torch.float32)


def tensor_to_uint8_image(tensor):
    tensor = tensor.detach().float().cpu().clamp(0, 1)
    if tensor.dim() == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for image saving, got {tensor.shape[0]}")
        tensor = tensor[0]
    if tensor.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(tensor.shape)}")
    image = tensor.permute(1, 2, 0).numpy()
    return (image * 255.0).round().astype(np.uint8)


def save_tensor_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_image(tensor)).save(path)


def build_generator(config, device):
    model_config = dict(config["model"])
    model_config["pretrain_path"] = None
    model_config.setdefault("optimizer", "adam")
    model_config.setdefault("init_lr", 0.0)
    model_config.setdefault("decay_schedule", None)

    train_config = {"train_iter": 1}
    model, _, _ = torch_model.init_model(model_config, train_config)
    model.to(device)

    checkpoint_config = config.get("checkpoint", {})
    checkpoint_path = resolve_path(checkpoint_config.get("path"))
    if checkpoint_path is None:
        raise ValueError("checkpoint.path must be set in the test config.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}\n"
            "Set checkpoint.path in the test config to an existing G.pth file. "
            "For example, use a checkpoint produced by training under "
            "validation.save_root/models/<epoch>/G.pth, or place downloaded weights "
            "under ./checkpoints/ and point checkpoint.path there."
        )

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    checkpoint_key = checkpoint_config.get("key")
    if checkpoint_key is not None:
        checkpoint = checkpoint[checkpoint_key]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=bool(checkpoint_config.get("strict", True)))
    model.eval()
    return model


def build_metrics(metric_names, device):
    metrics = []
    for name in metric_names:
        metric_name = str(name).lower()
        if metric_name == "psnr":
            metric = pyiqa.create_metric(
                metric_name,
                device=device,
                as_loss=False,
                test_y_channel=True,
                color_space="ycbcr",
            )
        else:
            metric = pyiqa.create_metric(metric_name, device=device, as_loss=False)
        metrics.append((metric_name, metric))
    return metrics


def evaluate_dataset(model, dataset_config, metrics, test_config, device):
    dataset_name = dataset_config["name"]
    root = resolve_path(dataset_config.get("root"))
    if dataset_config.get("lr_dir"):
        lr_dir = resolve_path(dataset_config.get("lr_dir"))
    elif root is not None:
        lr_dir = root / "LR"
    else:
        raise ValueError(f"Dataset {dataset_name} must set either root or lr_dir.")

    if dataset_config.get("hr_dir"):
        hr_dir = resolve_path(dataset_config.get("hr_dir"))
    elif root is not None:
        hr_dir = root / "HR"
    else:
        hr_dir = None

    lr_paths = collect_image_paths(lr_dir)
    if not lr_paths:
        raise ValueError(f"No LR images found in {lr_dir}")

    hr_paths = collect_image_paths(hr_dir) if hr_dir is not None and hr_dir.exists() else []
    has_hr = len(hr_paths) == len(lr_paths)
    if hr_paths and not has_hr:
        raise ValueError(f"LR/HR image count mismatch in {dataset_name}: {len(lr_paths)} vs {len(hr_paths)}")

    save_images = bool(test_config.get("save_images", True))
    output_dir = resolve_path(test_config.get("output_dir", "./outputs/test_results"))
    save_ext = test_config.get("save_ext", ".png")
    result = {"dataset": dataset_name, "num_images": len(lr_paths)}
    accum = {name: 0.0 for name, _ in metrics}

    with torch.no_grad():
        for index, lr_path in enumerate(lr_paths):
            lr = load_image_tensor(lr_path, device)
            sr = model(lr).clamp(0, 1)
            hr = load_image_tensor(hr_paths[index], device) if has_hr else None

            for metric_name, metric in metrics:
                if metric_name in NO_REF_METRICS:
                    value = metric(sr)
                else:
                    if hr is None:
                        continue
                    value = metric(sr, hr)
                accum[metric_name] += float(value)

            if save_images:
                stem = Path(lr_path).stem
                save_path = output_dir / dataset_name / f"{stem}{save_ext}"
                save_tensor_image(sr, save_path)

    for metric_name, value in accum.items():
        if metric_name in NO_REF_METRICS or has_hr:
            result[metric_name] = value / len(lr_paths)
    return result


def write_csv(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in results for key in row.keys()})
    preferred = ["dataset", "num_images"]
    keys = preferred + [key for key in keys if key not in preferred]
    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    return csv_path


def main():
    args = parse_args()
    config = file_io.reader.read_yaml(args.config)
    test_config = config.get("test", {})

    requested_device = test_config.get("device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    model = build_generator(config, device)
    metrics = build_metrics(test_config.get("metrics", ["psnr", "ssim"]), device)

    results = []
    for dataset_config in test_config["datasets"]:
        result = evaluate_dataset(model, dataset_config, metrics, test_config, device)
        results.append(result)
        print(result)

    if bool(test_config.get("save_csv", True)):
        csv_path = write_csv(results, resolve_path(test_config.get("output_dir", "./outputs/test_results")))
        print(f"Saved metrics to {csv_path}")


if __name__ == "__main__":
    main()
