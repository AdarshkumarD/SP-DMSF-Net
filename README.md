# SP-DMSF-Net

Official code for the manuscript **SP-DMSF-Net: Selective-Parallel Deterministic Mask Spatial Fusion Network for Landslide Segmentation**, submitted to *Computers & Geosciences* (Elsevier).

SP-DMSF-Net is a landslide-segmentation network that works on both **3-channel RGB** imagery and **4-channel RGB + DEM** input. It uses a VGG16 encoder with an ASPP bottleneck and a decoder with two changes:

1. **DMSF block (custom up-sampling).** The coarse feature map is up-sampled by zero-insertion (1x1 transposed convolution, stride 2). A fixed even/odd position mask then fills the gaps with the projected skip-connection features, instead of leaving zeros or interpolating. See `dmsf_upsampling` and `SelectiveMerge` in [`src/model.py`](src/model.py).
2. **Parallel convolution block.** Each decoder stage runs a 1x1 and a 3x3 convolution side by side, concatenates them, and fuses them with a 3x3 convolution. This replaces the usual series of 3x3 convolutions. See `parallel_conv_block` in [`src/model.py`](src/model.py).

```
input (RGB or RGB+DEM)
  └─ [4-ch only] 1x1 conv → 3 ch
      └─ VGG16 encoder (taps: block1_conv2 … block5_conv3)
          └─ ASPP (rates 6/12/18 + image pooling)
              └─ 4 × [ DMSF up-sampling → concat skip → parallel conv block ]
                  └─ 1x1 conv + sigmoid → landslide mask
```

## Repository layout

```
SP-DMSF-Net/
├── configs/
│   ├── rgb_3ch.yaml          # 3-channel RGB training configuration
│   └── rgbdem_4ch.yaml       # 4-channel RGB+DEM training configuration
├── src/
│   ├── compat.py             # TF 2.10 compatibility (Adam with weight decay)
│   ├── model.py              # SP-DMSF-Net (DMSF block, parallel conv block, ASPP)
│   ├── augmentation.py       # separate augmentation module (RGB+DEM aware)
│   ├── data.py               # file matching, splits, preprocessing, tf.data pipeline
│   ├── losses_metrics.py     # Dice/BCE/Focal/Tversky losses; F1, IoU, mIoU, PA, ...
│   ├── profiling.py          # params, size, FLOPs/MACs, latency, FPS, GPU memory
│   ├── visualize.py          # training curves, prediction panels
│   └── utils.py
├── train.py                  # train + evaluate + save everything
├── evaluate.py               # evaluate saved weights on val/test
├── tests/
│   ├── test_augmentation.py  # unit tests (NumPy/OpenCV only)
│   └── smoke_test.py         # end-to-end run on synthetic data
├── scripts/
│   └── set_gpu_env.ps1       # PowerShell GPU environment setup (Windows)
├── requirements-windows-gpu.txt  # exact tested package versions (TensorFlow 2.10.0)
└── LICENSE
```

## Installation

This guide describes the setup the code was developed and tested on: **Windows 10/11 with an NVIDIA GPU, without WSL**, using TensorFlow 2.10.0 (the last TensorFlow release with native Windows GPU support). Other platforms (Linux, WSL2, newer TensorFlow versions) have not been tested for this repository. The code uses standard TensorFlow/Keras, so it may work there with a matching CUDA setup, but this is not verified.

### Tested configuration

| Component | Version |
|---|---|
| OS | Windows 10/11 64-bit |
| GPU | NVIDIA RTX 2000 Ada Generation Laptop/Workstation GPU, 16 GB (compute capability 8.9) |
| NVIDIA driver | 596.71 |
| CUDA Toolkit | 11.2 (runtime DLLs) + `nvidia-cuda-nvcc-cu11` 11.8.89 (compiler, via pip) |
| cuDNN | 8.x for CUDA 11.x, `cudnn64_8.dll` in the CUDA 11.2 `bin` folder [TODO: exact version, see Step 3] |
| Python | 3.10.11 |
| TensorFlow / Keras | 2.10.0 / 2.10.0 |
| NumPy / protobuf | 1.26.4 / 3.19.6 |
| h5py / SciPy / pandas | 3.16.0 / 1.15.3 / 2.3.3 |
| scikit-learn / matplotlib | 1.7.2 / 3.10.9 |
| OpenCV (`opencv-python`) / PyYAML | 4.10.0.84 / 6.0.3 |

The exact direct requirements are in [`requirements-windows-gpu.txt`](requirements-windows-gpu.txt).

### Setup

Run every command in **PowerShell** (not `cmd`, not bash: PowerShell does not understand `&&` or `source`).

**Step 1: NVIDIA driver.** Install the current driver for your GPU from [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx). Check it with:

```powershell
nvidia-smi
```

The GPU must be listed. The "CUDA Version" printed there (for example 13.2) is only the maximum the driver supports; it does not have to match the toolkit below.

Also install the **Microsoft Visual C++ Redistributable 2015-2022 (x64)** from [aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) if it is not already present; TensorFlow needs `msvcp140.dll`.

**Step 2: CUDA Toolkit 11.2.** Download CUDA Toolkit **11.2** from the [NVIDIA CUDA archive](https://developer.nvidia.com/cuda-11.2.2-download-archive) and install it to the default location (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`). Other CUDA versions (for example 12.x) may stay installed, but TensorFlow 2.10 needs the 11.2 DLLs, which Step 6 puts first in the PATH. You may deselect the driver component in the installer to keep your newer driver.

**Step 3: cuDNN 8.** Download cuDNN **8.x for CUDA 11.x** from the [NVIDIA cuDNN archive](https://developer.nvidia.com/rdp/cudnn-archive) (a free NVIDIA developer account is required) and unzip it. Copy the files from its `bin`, `include` and `lib` folders into the same-named folders of `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\`. Check:

```powershell
Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\cudnn64_8.dll"   # True
```

Record the exact cuDNN version (for the methods section of a paper) with:

```powershell
(Get-Item "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\cudnn64_8.dll").VersionInfo.FileVersion
```

**Step 4: Python 3.10.** Install Python **3.10.x** (64-bit) from [python.org](https://www.python.org/downloads/windows/) and tick "Add python.exe to PATH". TensorFlow 2.10 does not support Python 3.11 or newer. Check the version:

```powershell
py -3.10 --version     # Python 3.10.x
```

**Step 5: Virtual environment and packages.** From the project folder:

```powershell
cd E:\path\to\SP-DMSF-Net
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-windows-gpu.txt
```

Upgrading pip is optional; the pinned packages install with the pip that ships with Python 3.10. If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and retry. If pip downloads time out on a slow connection, repeat the command with `--default-timeout=120`.

**Step 6: Prepare the session for the GPU.** In every new terminal, after activating the environment, run:

```powershell
. .\scripts\set_gpu_env.ps1
```

This script (see [`scripts/set_gpu_env.ps1`](scripts/set_gpu_env.ps1)) does three things for the current terminal only:

1. puts `...\CUDA\v11.2\bin` first in the `PATH`, so TensorFlow 2.10 finds the CUDA 11.2 and cuDNN 8 DLLs even if a newer CUDA toolkit is installed;
2. adds the `bin` folder of the pip package `nvidia-cuda-nvcc-cu11` (CUDA 11.8 compiler tools) to the `PATH`. TensorFlow 2.10 ships no kernels built for compute capability 8.9 (RTX 40-series / RTX Ada), so CUDA compiles them for your GPU on first use, and the older 11.2 compiler does not know this architecture. This was the setup used for the reported experiments;
3. sets `CUDA_CACHE_MAXSIZE` to 4 GiB so the compiled GPU kernels are cached between runs.

The first epoch on a new GPU can therefore be noticeably slower than later epochs.

**Step 7: Verify the GPU.**

```powershell
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

Expected output: `2.10.0 [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`. An empty list `[]` means TensorFlow runs on the CPU; recheck Steps 2, 3 and 6.

Then run the tests (they use synthetic data and need no dataset):

```powershell
python tests/test_augmentation.py
python tests/smoke_test.py      # ends with "SMOKE TEST PASSED"
```

Notes for this version of TensorFlow: `src/compat.py` supplies Adam with decoupled weight decay (missing in TensorFlow 2.10). Weight decay is ignored for the `sgd` and `rmsprop` options on this version; the default Adam configs are unaffected.

### Memory

On a 16 GB GPU (for example an RTX 2000 Ada), the default `batch_size: 8` at 256 x 256 should fit. If you get an out-of-memory error, add `--set training.batch_size=4`.


## Data preparation

### 4-channel: Bijie landslide dataset (RGB + DEM)

Download: https://gpcv.whu.edu.cn/data/Bijie_landslide_dataset.zip

```text
landslide/
├── image/   RGB images
├── mask/    Binary landslide masks
└── dem/     Digital elevation model (DEM)
```

Files are paired by filename stem (e.g., `001.png` ↔ `001.png` ↔ `001.tif`). Set `data.image_dir`, `data.mask_dir`, and `data.dem_dir` in `configs/rgbdem_4ch.yaml` to the corresponding directories. Replace the paths with your local dataset paths.

### 3-channel: Hokkaido and Jiuzhai Valley datasets (RGB)

**Hokkaido Iburi-Tobu dataset (2023)**

Download: [Hokkaido Iburi-Tobu.zip](https://zenodo.org/records/10294997/files/Hokkaido%20Iburi-Tobu.zip?download=1)

**Jiuzhai Valley (UAV-0.2m) dataset (2023)**

Download: [Jiuzhai valley (UAV-0.2m).zip](https://zenodo.org/records/10294997/files/Jiuzhai%20valley%20(UAV-0.2m).zip?download=1)


```text
hokkaido/
├── img/
└── mask/

jiuzhai/
├── img/
└── mask/
```

Set `data.image_dir` and `data.mask_dir` in `configs/rgb_3ch.yaml` to the appropriate image and mask directories for the dataset being used.

**Dataset reference:** Xu, Y., Ouyang, C., Xu, Q., et al. (2024). *CAS Landslide Dataset: A Large-Scale and Multisensor Dataset for Deep Learning-Based Landslide Detection*. Scientific Data, 11, 12. https://doi.org/10.1038/s41597-023-02847-z

### Preprocessing and splits

Every image is resized to 256 x 256 and min-max normalised per channel. The 4-channel config also applies per-channel histogram equalisation to RGB (`preprocessing.hist_equalize`) and min-max normalises the DEM. Masks are resized with nearest-neighbour interpolation and binarised.

Splits are made in two stages with a fixed seed (`seed: 42`). The exact file lists are written to `splits.json` in every run folder.

| Config | Stage 1 (`holdout`) | Stage 2 (`test_in_holdout`) | Result |
|---|---|---|---|
| `rgbdem_4ch.yaml` | 308 samples held out | 154 of those are test | train = N − 308, val = 154, test = 154 |
| `rgb_3ch.yaml` | 30 % held out | 50 % of those are test | 70 % / 15 % / 15 % |

## Training

```powershell
# 4-channel RGB+DEM (augmentation ON in the config)
python train.py --config configs/rgbdem_4ch.yaml

# 3-channel RGB (augmentation OFF in the config)
python train.py --config configs/rgb_3ch.yaml
```

Any config value can be overridden on the command line:

```powershell
python train.py --config configs/rgbdem_4ch.yaml `
    --set training.epochs=50 training.batch_size=4 data.image_dir=E:/path/to/image
```

### Switching augmentation on or off

The switch is `augmentation.enabled` in each config. You can also set it per run without editing the file:

```powershell
python train.py --config configs/rgbdem_4ch.yaml --aug on
python train.py --config configs/rgbdem_4ch.yaml --aug off
```

Runs go to separate folders (`..._aug-on` / `..._aug-off`), so the two settings never overwrite each other.

The augmentation lives in [`src/augmentation.py`](src/augmentation.py). It is adapted from the 56-variant scheme of *UAV T-YOLO-Rice* (Sangaiah et al., IEEE TNSE 2024): 4 rotations × 2 mirror states × 7 photometric versions (none, Gaussian noise, saturation, uniform noise, blur, brightness up, brightness down).

- Rotation and mirroring are applied identically to RGB, DEM and mask, so labels stay aligned.
- Photometric changes touch the RGB channels only. DEM and mask are never modified.
- Augmentation is applied to the training split only, after the split. Validation and test data are never augmented.
- `mode: online_random` gives each training image one random variant per epoch, so the epoch size is unchanged.
- `mode: offline_full` expands every training image into all variants (56 × longer epochs), so reduce `training.epochs` accordingly.
- Rotations, mirroring, the photometric list and all strengths (`gauss_sigma`, `saturation_factor`, ...) are configurable.

### Configuration summary

| Setting | `rgb_3ch.yaml` | `rgbdem_4ch.yaml` |
|---|---|---|
| Input channels | 3 (RGB) | 4 (RGB + DEM) |
| Histogram equalisation | off | on |
| Augmentation (default) | off | on (`online_random`) |
| Epochs | 60 | 100 |
| Early stopping / ReduceLROnPlateau patience | 7 / 3 | 20 / 7 |
| Optimizer, LR, weight decay | Adam, 1e-4, 1e-3 | Adam, 1e-4, 1e-3 |
| Loss | Dice | Dice |
| Batch size, input size | 8, 256 x 256 | 8, 256 x 256 |

Further options in the configs: optimizer (`adam | adamw | sgd | rmsprop`), loss (`bce | dice | focal | tversky | combined`), `use_multi_gpu` (MirroredStrategy), and `hardware_profiling`.

## Outputs

Each run writes to `outputs/<experiment_name>_aug-<on|off>/`:

| File | Content |
|---|---|
| `config_used.yaml` | Resolved config of the run, including CLI overrides |
| `splits.json` | Exact train/val/test file lists |
| `best.weights.h5` | Weights with the lowest validation loss; these are used for the final evaluation |
| `history.csv` | Per-epoch train/val metrics |
| `metrics.json` | Train, validation and test metrics (Precision, Recall, F1, Dice, IoU, mIoU, PA, loss) |
| `hardware.json` | Parameters, model size, FLOPs/MACs, latency, FPS, throughput, peak GPU memory, train time |
| `training_curves.png` | Train vs validation curves |
| `test_predictions.png` | RGB / DEM / ground truth / prediction for small, median and large landslides |

Evaluate saved weights later (in PowerShell, the backtick continues the command on the next line):

```powershell
python evaluate.py --config configs/rgbdem_4ch.yaml `
    --weights outputs/sp_dmsf_net_rgbdem_4ch_aug-on/best.weights.h5 --split test
```

For the 3-channel (RGB) model, in PowerShell (the backtick continues the command on the next line):

```powershell
python evaluate.py --config configs/rgb_3ch.yaml `
  --weights outputs/sp_dmsf_net_rgb_3ch_aug-off/best.weights.h5 --split test `
  --set data.image_dir=E:/data/hokkaido/img data.mask_dir=E:/data/hokkaido/mask
```

To evaluate on a separate external test set, use `--split all` (every matched sample in the configured folders is evaluated, nothing is augmented):

```powershell
python evaluate.py --config configs/rgb_3ch.yaml `
  --weights outputs/sp_dmsf_net_rgb_3ch_aug-off/best.weights.h5 --split all `
  --out-dir outputs/external_test_rgb `
  --set data.image_dir=E:/ext/image data.mask_dir=E:/ext/mask
```

## Reproducibility notes

- Seeds are fixed for Python, NumPy and TensorFlow (`seed: 42`), and the data split is deterministic. Random augmentation draws and GPU kernels are not bit-wise deterministic, so expect small run-to-run differences.
- The VGG16 encoder is initialised with ImageNet weights (downloaded by Keras on first use). Set `model.pretrained_backbone: false` to train from scratch.
- `metrics.json` reports train, validation and test results, all computed with the best-validation-loss weights in inference mode (no dropout, no augmentation), so the three are directly comparable. The `train_last_epoch_running` entry holds the running training metrics of the last epoch (dropout and augmentation active), as shown in the training curves.
- Software and hardware used for verification: Python 3.10.11, TensorFlow 2.10.0 / Keras 2.10.0, CUDA Toolkit 11.2 with cuDNN 8.x (plus the CUDA 11.8 compiler tools from `nvidia-cuda-nvcc-cu11` 11.8.89), NVIDIA driver 596.71, NVIDIA RTX 2000 Ada Generation GPU (16 GB).


Please also cite the datasets (Bijie, Hokkaido, Jiuzhai Valley) and the works this pipeline builds on (FFS-Net, Liu et al., IEEE TGRS 2023; UAV T-YOLO-Rice, Sangaiah et al., IEEE TNSE 2024).

## License

Released under the MIT License (see [`LICENSE`](LICENSE)).
