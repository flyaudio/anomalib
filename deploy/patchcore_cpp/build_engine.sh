#!/bin/bash
# Build a TensorRT 10.11 engine from the exported PatchCore ONNX model.
# Run this ON the target machine (RTX 5090, sm_120) -- engines are GPU-specific.
set -euo pipefail

TRT_DIR=${TRT_DIR:-/opt/TensorRT-10.11.0.33}
ONNX=${1:-exports/weights/onnx/patchcore.onnx}
ENGINE=${2:-patchcore_5090_fp16.engine}

# The ONNX has a dynamic batch dim and static 224x224 spatial dims.
# Adjust maxShapes batch if you plan to run batched inference.
"${TRT_DIR}/bin/trtexec" \
    --onnx="${ONNX}" \
    --saveEngine="${ENGINE}" \
    --fp16 \
    --minShapes=input:1x3x224x224 \
    --optShapes=input:1x3x224x224 \
    --maxShapes=input:8x3x224x224 \
    --builderOptimizationLevel=3 \
    --verbose

echo "Engine written to ${ENGINE}"
echo "Quick benchmark: ${TRT_DIR}/bin/trtexec --loadEngine=${ENGINE} --shapes=input:1x3x224x224"
