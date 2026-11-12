# Build a TensorRT 10.11 engine from the exported PatchCore ONNX model (Windows).
# Run this ON the target machine (RTX 5090, sm_120) -- engines are GPU-specific.

$TrtDir = $env:TRT_DIR
if (-not $TrtDir) { $TrtDir = "D:\TensorRT-10.11.0.33" }

$Onnx   = if ($args[0]) { $args[0] } else { "exports\weights\onnx\patchcore.onnx" }
$Engine = if ($args[1]) { $args[1] } else { "patchcore_5090_fp16.engine" }

# The ONNX has a dynamic batch dim and static 224x224 spatial dims.
# Adjust maxShapes batch if you plan to run batched inference.
& "$TrtDir\bin\trtexec.exe" `
    --onnx="$Onnx" `
    --saveEngine="$Engine" `
    --fp16 `
    --minShapes=input:1x3x224x224 `
    --optShapes=input:1x3x224x224 `
    --maxShapes=input:8x3x224x224 `
    --builderOptimizationLevel=3 `
    --verbose

Write-Host "Engine written to $Engine"
Write-Host "Quick benchmark: $TrtDir\bin\trtexec.exe --loadEngine=$Engine --shapes=input:1x3x224x224"
