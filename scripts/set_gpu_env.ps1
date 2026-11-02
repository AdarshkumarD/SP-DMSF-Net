# Prepare a PowerShell session for TensorFlow 2.10 on an NVIDIA GPU (native Windows).
#
# Usage (activate the virtual environment first, then dot-source this file):
#     .venv\Scripts\Activate.ps1
#     . .\scripts\set_gpu_env.ps1
#
# It only changes the PATH / environment of the CURRENT terminal session.

# 1) CUDA 11.2 runtime and cuDNN 8 DLLs must be found BEFORE any other CUDA version
#    (e.g. a newer toolkit that also sets CUDA_PATH).
$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin"
if (Test-Path $cudaBin) {
    $env:PATH = "$cudaBin;" + $env:PATH
} else {
    Write-Warning "CUDA 11.2 not found at $cudaBin - install CUDA Toolkit 11.2 (see README, Option B)."
}

# 2) ptxas / nvcc 11.8 from the pip package `nvidia-cuda-nvcc-cu11`. Version 11.2's own
#    compiler does not know RTX 40-series / RTX Ada GPUs (compute capability 8.9).
$nvccBin = python -c "import os, nvidia.cuda_nvcc; print(os.path.join(os.path.dirname(nvidia.cuda_nvcc.__file__), 'bin'))"
if ($LASTEXITCODE -eq 0 -and $nvccBin) {
    $env:PATH = "$nvccBin;" + $env:PATH
} else {
    Write-Warning "Python package nvidia-cuda-nvcc-cu11 not importable - is the virtual environment active?"
}

# 3) Keep compiled GPU kernels (first run on a new GPU architecture compiles them; 4 GiB cache).
$env:CUDA_CACHE_MAXSIZE = "4294967296"

Write-Host "GPU environment ready. Verify with:"
Write-Host "  python -c `"import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))`""
