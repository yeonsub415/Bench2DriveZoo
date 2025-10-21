"""Utility script to dump BEVFormer BEV embeddings and UniAD SDC trajectories.

This helper runs the pretrained BEVFormer and UniAD checkpoints on the
Bench2Drive dataset without performing any training.  It saves the BEV feature
embeddings (`[bs, bev_h*bev_w, embed_dims]`) predicted by BEVFormer and the SDC
trajectory outputs from UniAD to disk for later analysis.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import torch
from mmcv import Config
from mmcv.datasets import build_dataset
from mmcv.parallel import DataContainer, collate
from torch.utils.data import DataLoader

from mmcv.models import build_model
from mmcv.runner import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump BEVFormer BEV embeddings and UniAD SDC trajectories",
    )
    parser.add_argument(
        "--bevformer-config",
        required=True,
        help="Path to the BEVFormer config file",
    )
    parser.add_argument(
        "--bevformer-checkpoint",
        required=True,
        help="Path to the BEVFormer checkpoint (.pth)",
    )
    parser.add_argument(
        "--uniad-config",
        required=True,
        help="Path to the UniAD config file",
    )
    parser.add_argument(
        "--uniad-checkpoint",
        required=True,
        help="Path to the UniAD checkpoint (.pth)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where outputs will be written",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to iterate over",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader worker processes",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples to export",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to run inference on (e.g. cuda:0 or cpu)",
    )
    return parser.parse_args()


def _import_plugins(cfg: Config, repo_root: Path) -> None:
    """Import custom plugin modules referenced by an MMDetection-style config."""
    if getattr(cfg, "plugin", False):
        plugin_dir = getattr(cfg, "plugin_dir", None)
        if plugin_dir is None:
            return
        module_path = plugin_dir.replace(os.sep, ".").strip(".")
        sys.path.insert(0, str(repo_root))
        importlib.import_module(module_path)


def _prepare_dataset_config(cfg: Config, split: str) -> Tuple[Config, int]:
    data_cfg = cfg.data.get(split)
    if data_cfg is None:
        raise ValueError(f"Dataset split '{split}' is not defined in the config")
    data_cfg = data_cfg.copy()
    data_cfg.test_mode = True
    samples_per_gpu = data_cfg.pop("samples_per_gpu", 1)
    data_cfg.pop("workers_per_gpu", None)
    return data_cfg, samples_per_gpu


def _build_dataloader(dataset, samples_per_gpu: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=samples_per_gpu,
        sampler=None,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        collate_fn=lambda batch: collate(batch, samples_per_gpu=samples_per_gpu),
    )


def _unwrap_container(obj: Any) -> Any:
    if isinstance(obj, DataContainer):
        return _unwrap_container(obj.data)
    if isinstance(obj, Mapping):
        return {k: _unwrap_container(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_unwrap_container(v) for v in obj]
        return type(obj)(seq)
    return obj


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_to_device(v, device) for v in obj]
        return type(obj)(seq)
    return obj


def _extract_meta_id(img_metas: Iterable) -> Tuple[Dict[str, Any], str]:
    def _first_meta(meta_obj: Any) -> Dict[str, Any]:
        if isinstance(meta_obj, Mapping):
            return meta_obj
        if isinstance(meta_obj, (list, tuple)) and meta_obj:
            return _first_meta(meta_obj[-1])
        raise ValueError("Unable to parse img_metas for metadata")

    primary_meta = _first_meta(img_metas)
    scene = (
        primary_meta.get("scene_token")
        or primary_meta.get("folder")
        or primary_meta.get("scene_id")
        or "unknown_scene"
    )
    frame = primary_meta.get("frame_idx", primary_meta.get("timestamp", 0))
    if isinstance(frame, (int, float)):
        frame_str = f"{int(frame):06d}"
    else:
        frame_str = str(frame)
    sample_id = f"{scene}_{frame_str}"
    return primary_meta, sample_id


def _build_model(cfg: Config, checkpoint: str, device: torch.device):
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, checkpoint, map_location="cpu")
    model.to(device)
    model.eval()
    return model


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def export_bevformer_embeddings(
    cfg: Config,
    checkpoint: str,
    device: torch.device,
    out_dir: Path,
    split: str,
    workers: int,
    max_samples: int | None,
) -> None:
    data_cfg, samples_per_gpu = _prepare_dataset_config(cfg, split)
    dataset = build_dataset(data_cfg)
    dataloader = _build_dataloader(dataset, samples_per_gpu, workers)
    model = _build_model(cfg, checkpoint, device)

    model.prev_frame_info = {
        "prev_bev": None,
        "scene_token": None,
        "prev_pos": 0,
        "prev_angle": 0,
    }

    for idx, batch in enumerate(dataloader):
        if max_samples is not None and idx >= max_samples:
            break
        unwrapped = _unwrap_container(batch)
        meta, sample_id = _extract_meta_id(unwrapped["img_metas"])
        inputs = _to_device(unwrapped, device)
        with torch.no_grad():
            model(return_loss=False, rescale=True, **inputs)
        bev_embed = model.prev_frame_info.get("prev_bev")
        if bev_embed is None:
            continue
        save_obj = {
            "bev_embed": bev_embed.detach().cpu(),
            "meta": meta,
        }
        torch.save(save_obj, out_dir / f"{sample_id}.pt")


def export_uniad_sdc_traj(
    cfg: Config,
    checkpoint: str,
    device: torch.device,
    out_dir: Path,
    split: str,
    workers: int,
    max_samples: int | None,
) -> None:
    data_cfg, samples_per_gpu = _prepare_dataset_config(cfg, split)
    dataset = build_dataset(data_cfg)
    dataloader = _build_dataloader(dataset, samples_per_gpu, workers)
    model = _build_model(cfg, checkpoint, device)

    model.keep_bev_for_export = False
    model.prev_frame_infos = []
    model.prev_frame_num = getattr(model, "prev_frame_num", 0)
    model.prev_frame_info = {
        "prev_bev": None,
        "scene_token": None,
        "prev_pos": 0,
        "prev_angle": 0,
    }

    for idx, batch in enumerate(dataloader):
        if max_samples is not None and idx >= max_samples:
            break
        unwrapped = _unwrap_container(batch)
        meta, sample_id = _extract_meta_id(unwrapped["img_metas"])
        inputs = _to_device(unwrapped, device)
        with torch.no_grad():
            results = model(return_loss=False, rescale=True, **inputs)
        if not results:
            continue
        planning = results[0].get("planning", {})
        planning_results = planning.get("result_planning", {})
        sdc_traj = planning_results.get("sdc_traj")
        if sdc_traj is None:
            continue
        save_obj = {
            "sdc_traj": sdc_traj.detach().cpu(),
            "meta": meta,
        }
        torch.save(save_obj, out_dir / f"{sample_id}.pt")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    bev_cfg = Config.fromfile(args.bevformer_config)
    uniad_cfg = Config.fromfile(args.uniad_config)
    _import_plugins(bev_cfg, repo_root)
    _import_plugins(uniad_cfg, repo_root)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    bev_out_dir = output_dir / "bevformer"
    uniad_out_dir = output_dir / "uniad"
    _ensure_dir(bev_out_dir)
    _ensure_dir(uniad_out_dir)

    export_bevformer_embeddings(
        bev_cfg,
        args.bevformer_checkpoint,
        device,
        bev_out_dir,
        args.split,
        args.workers,
        args.max_samples,
    )
    export_uniad_sdc_traj(
        uniad_cfg,
        args.uniad_checkpoint,
        device,
        uniad_out_dir,
        args.split,
        args.workers,
        args.max_samples,
    )


if __name__ == "__main__":
    main()