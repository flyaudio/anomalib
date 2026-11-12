"""Export PatchCore to ONNX for TensorRT / C++ deployment.

This script does one of the following:

1. ``--ckpt_path`` given: loads a trained Lightning checkpoint and exports it.
2. Otherwise: trains PatchCore on MVTec AD (feature extraction + coreset only,
   one epoch, no gradient updates), runs a validation pass to populate the
   post-processor statistics (thresholds, normalization min/max), then exports.

Export design choices (important for TensorRT):

- ``pre_processor`` is REMOVED from the exported graph. The ONNX ``Resize`` op
  with ``antialias=True`` is not well supported by TensorRT, and image
  resize/crop/normalize is cheap and flexible to do in C++ (OpenCV).
  The C++ side must reproduce: resize to 256x256 -> center crop 224x224 ->
  ImageNet normalization.
- ``post_processor`` is KEPT in the graph (pure tensor ops + persisted
  threshold/normalization buffers). Outputs are therefore already normalized:
  ``pred_score`` / ``anomaly_map`` are in [0, 1], and the adaptive threshold
  learned on the validation set maps to 0.5. ``pred_label`` / ``pred_mask``
  (bool) are also exported.
- The PatchCore ``memory_bank`` buffer is baked into the graph as a constant
  initializer, so the nearest-neighbor search runs entirely on GPU inside
  TensorRT (it is implemented as a plain matmul, see
  ``src/anomalib/models/image/patchcore/torch_model.py``).
- Export in FP32; let TensorRT build an FP16 engine with ``trtexec --fp16``.

Usage:
    # train on MVTec AD bottle and export
    python export_patchcore_onnx.py --data_root ./datasets/MVTecAD --category bottle

    # export an existing checkpoint (no dataset needed)
    python export_patchcore_onnx.py --ckpt_path results/patchcore/.../model.ckpt
"""

import argparse
import json
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore

IMAGE_SIZE = 256   # resize before crop (paper setting)
CROP_SIZE = 224    # network input size (paper setting)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path to a trained .ckpt. If given, training is skipped.")
    parser.add_argument("--data_root", type=str, default="./datasets/MVTecAD",
                        help="MVTec AD root (only needed when training from scratch).")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec AD category.")
    parser.add_argument("--output_dir", type=str, default="./exports", help="Export root directory.")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2", help="timm backbone.")
    parser.add_argument("--coreset_sampling_ratio", type=float, default=0.1,
                        help="Coreset ratio. Smaller = smaller memory bank = smaller ONNX/faster NN search.")
    parser.add_argument("--num_neighbors", type=int, default=9, help="PatchCore num_neighbors.")
    parser.add_argument("--no_post_processor", action="store_true",
                        help="Export raw scores (C++ side must normalize / threshold itself).")
    parser.add_argument("--fixed_batch", action="store_true",
                        help="Export a fully static graph with batch=1 (default keeps batch dim dynamic).")
    return parser.parse_args()


def build_or_load_model(args: argparse.Namespace) -> tuple[Patchcore, Engine, MVTecAD | None]:
    """Create the PatchCore model, optionally training it first."""
    # Paper-style pre-processing used for training; removed again before export.
    pre_processor = Patchcore.configure_pre_processor(
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        center_crop_size=(CROP_SIZE, CROP_SIZE),
    )
    model = Patchcore(
        backbone=args.backbone,
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        num_neighbors=args.num_neighbors,
        pre_processor=pre_processor,
        post_processor=not args.no_post_processor,
    )
    engine = Engine()

    datamodule: MVTecAD | None = None
    if args.ckpt_path:
        print(f"[export] Loading checkpoint: {args.ckpt_path}")
        model = Patchcore.load_from_checkpoint(args.ckpt_path)
    else:
        print(f"[export] Training PatchCore on MVTec AD '{args.category}' (feature extraction + coreset)...")
        datamodule = MVTecAD(root=args.data_root, category=args.category,
                             train_batch_size=32, eval_batch_size=32)
        engine.fit(model=model, datamodule=datamodule)
        # Validation pass populates PostProcessor stats (adaptive thresholds, min/max).
        if not args.no_post_processor:
            engine.validate(model=model, datamodule=datamodule)
    return model, engine, datamodule


def dump_stats(model: Patchcore, stats_path: Path) -> None:
    """Dump post-processor statistics next to the ONNX file (for debugging/C++ reference)."""
    pp = model.post_processor
    if pp is None:
        return
    stats = {
        "note": "Normalized outputs are in [0,1]; the adaptive threshold maps to 0.5.",
        "image_threshold_raw": float(pp._image_threshold),  # noqa: SLF001
        "pixel_threshold_raw": float(pp._pixel_threshold),  # noqa: SLF001
        "image_min": float(pp.image_min),
        "image_max": float(pp.image_max),
        "pixel_min": float(pp.pixel_min),
        "pixel_max": float(pp.pixel_max),
        "memory_bank_shape": list(model.model.memory_bank.shape),
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[export] Stats written to {stats_path}: {stats}")


def main() -> None:
    """Export the model and print the graph signature."""
    args = parse_args()
    model, engine, _ = build_or_load_model(args)

    # --- Export-time graph surgery ---
    # Drop pre_processor: resize/crop/normalize are done in C++ (OpenCV).
    model.pre_processor = None
    model.eval()

    onnx_kwargs = {"dynamic_axes": {}} if args.fixed_batch else {}
    onnx_path = engine.export(
        model=model,
        export_type=ExportType.ONNX,
        export_root=args.output_dir,
        model_file_name="patchcore",
        input_size=(CROP_SIZE, CROP_SIZE),
        onnx_kwargs=onnx_kwargs,
    )
    print(f"[export] ONNX written to {onnx_path}")
    dump_stats(model, Path(onnx_path).with_suffix(".stats.json"))

    # --- Verify the graph signature (names the C++ code binds to) ---
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        print("[export] Graph inputs:")
        for inp in onnx_model.graph.input:
            dims = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  - {inp.name}: {dims}")
        print("[export] Graph outputs:")
        for out in onnx_model.graph.output:
            dims = [d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  - {out.name}: {dims}")
    except ImportError:
        print("[export] onnx package not found, skipping graph inspection.")

    print("\nNext step: build the TensorRT engine, e.g.")
    print("  trtexec --onnx exports/weights/onnx/patchcore.onnx --fp16 \\")
    print("      --saveEngine=patchcore_fp16.engine \\")
    print("      --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:8x3x224x224")


if __name__ == "__main__":
    main()
