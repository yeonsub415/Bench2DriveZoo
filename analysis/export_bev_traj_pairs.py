"""Export BEVFormer embeddings and trajectories from Bench2Drive clips.

This script runs a UniAD model on the Bench2Drive dataset and stores, for
each frame, the BEV token sequence produced by BEVFormer together with both
model-predicted trajectories and the dataset-provided planning labels.  The
resulting ``.npz`` files can be used as inputs/targets when training VQ-BET
style policies or other downstream modules that expect BEV tokens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from mmcv import Config
from mmcv.datasets import build_dataloader, build_dataset
from mmcv.models import build_model
from mmcv.runner import load_checkpoint
from mmcv.parallel import scatter
from mmcv.utils import ProgressBar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export BEV tokens and trajectories from Bench2Drive clips",
    )
    parser.add_argument(
        "config",
        help="Path to a UniAD stage-2 config (e.g. adzoo/uniad/configs/stage2_e2e/tiny_e2e_b2d.py)",
    )
    parser.add_argument(
        "checkpoint",
        help="Checkpoint containing trained UniAD weights",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where the extracted pairs will be written",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device used for inference",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of frames to export",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip frames whose output files already exist",
    )
    return parser.parse_args()


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach a tensor and move it to numpy."""
    if tensor is None:
        raise ValueError("Expected a tensor but received None")
    return tensor.detach().cpu().numpy()


def _normalise_scene_name(raw: str | None, fallback: str) -> str:
    if raw is None or raw == "":
        return fallback
    # Windows paths may leak backslashes when prepare_B2D.py runs on Windows.
    return raw.replace("\\", "/").strip("/")


def _frame_to_str(frame: Any, index: int) -> str:
    if isinstance(frame, (int, np.integer)):
        return f"{frame:06d}"
    if isinstance(frame, float) and frame.is_integer():
        return f"{int(frame):06d}"
    if frame is None:
        return f"{index:06d}"
    return str(frame)


def main() -> None:
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        dataset = build_dataset(cfg.data.test)
    else:
        raise TypeError("cfg.data.test must be a dataset config dict")

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    if hasattr(model, "keep_bev_for_export"):
        model.keep_bev_for_export = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Any]] = []
    total = len(data_loader)
    if args.max_samples is not None:
        total = min(total, args.max_samples)
    progress = ProgressBar(total)

    for idx, data in enumerate(data_loader):
        if args.max_samples is not None and idx >= args.max_samples:
            break

        scattered = scatter(data, [device])
        batch = scattered[0]

        with torch.no_grad():
            outputs = model(return_loss=False, rescale=True, **batch)

        if not outputs:
            raise RuntimeError("Model returned no outputs for batch")
        result = outputs[0]

        bev_tensor = result.get("bev_embed")
        if bev_tensor is None:
            raise KeyError(
                "Model output did not contain 'bev_embed'. "
                "Ensure the UniAD model exposes BEV features (keep_bev_for_export)."
            )
        if bev_tensor.dim() == 3 and bev_tensor.shape[1] == 1:
            bev_tensor = bev_tensor[:, 0]
        bev_tensor = bev_tensor.to(torch.float32)
        bev_embed = _tensor_to_numpy(bev_tensor)

        planning = result.get("planning", {})
        if "result_planning" not in planning:
            raise KeyError("Planning head output missing from inference results")
        sdc_traj_pred = planning["result_planning"].get("sdc_traj")
        sdc_traj_all_pred = planning["result_planning"].get("sdc_traj_all")
        if sdc_traj_pred is None or sdc_traj_all_pred is None:
            raise KeyError("Planning result missing required trajectory tensors")
        pred_traj = _tensor_to_numpy(sdc_traj_pred)
        pred_traj_all = _tensor_to_numpy(sdc_traj_all_pred)
        if pred_traj.ndim == 3 and pred_traj.shape[0] == 1:
            pred_traj = pred_traj[0]
        if pred_traj_all.ndim == 3 and pred_traj_all.shape[0] == 1:
            pred_traj_all = pred_traj_all[0]

        # Obtain dataset-provided planning labels for the same index.
        sample_info = dataset.get_data_info(idx)
        scene_name = _normalise_scene_name(
            sample_info.get("folder") or sample_info.get("scene_token"),
            fallback=f"scene_{idx:06d}",
        )
        frame_name = _frame_to_str(sample_info.get("frame_idx"), idx)

        gt_traj = np.asarray(sample_info.get("sdc_planning"), dtype=np.float32)
        gt_mask = np.asarray(sample_info.get("sdc_planning_mask"), dtype=bool)
        if gt_traj.ndim == 3 and gt_traj.shape[0] == 1:
            gt_traj = gt_traj[0]
        if gt_mask.ndim == 2 and gt_mask.shape[0] == 1:
            gt_mask = gt_mask[0]
        command_value = sample_info.get("command", -1)
        command = int(np.asarray(command_value).item())

        sample_dir = out_dir / scene_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_path = sample_dir / f"{frame_name}.npz"
        if args.skip_existing and sample_path.exists():
            progress.update()
            continue

        np.savez_compressed(
            sample_path,
            bev_embed=bev_embed,
            sdc_traj_pred=pred_traj,
            sdc_traj_all_pred=pred_traj_all,
            sdc_traj_gt=gt_traj,
            sdc_traj_gt_mask=gt_mask,
            command=command,
        )

        manifest.append(
            {
                "scene": scene_name,
                "frame": frame_name,
                "file": str(sample_path.relative_to(out_dir)),
                "traj_steps_pred": int(pred_traj.shape[0]),
                "traj_steps_gt": int(gt_traj.shape[0]),
            }
        )
        progress.update()

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved {len(manifest)} samples to {out_dir}")


if __name__ == "__main__":
    main()