# OmniGS Evaluation — RunPod Setup Context

## Goal

Evaluate OmniGS (omnidirectional Gaussian splatting) for photorealistic 3D
reconstruction of manhole interiors from 360° video. OmniGS natively supports
equirectangular images — no perspective decomposition, no quality loss from
reprojection. This is the key advantage over standard 3DGS.

## RunPod Instance

- **SSH**: `ssh root@213.173.111.113 -p 34515 -i ~/.ssh/id_ed25519`
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (98GB VRAM)
- **CUDA**: 12.8 toolkit, driver 580 (CUDA 13.0 capable)
- **Python**: 3.12.3
- **Disk**: 100GB

## Data on this machine

```
/root/omnigs_eval/
├── CONTEXT.md                         # This file
├── data/
│   └── BR03-054/
│       ├── video.mp4                  # Raw 360° equirectangular (3840x1920, 30fps, 43s)
│       │                              # Ricoh Theta Z1, upside-down mount (rotate=180 metadata)
│       │                              # Manhole inspection — camera lowered on cable
│       ├── sparse.ply                 # COLMAP sparse point cloud (510K points)
│       ├── colmap_sparse/             # COLMAP model from perspective reconstruction
│       │   ├── cameras.bin            # PINHOLE 1080x1080, f=540
│       │   ├── images.bin             # 1030 registered images (99.8%)
│       │   └── points3D.bin           # 510K 3D points, 0.79px reproj error
│       └── colmap_to_omnigs.py        # Conversion script (fallback approach)
└── results/                           # (created by training)
```

## Two Approaches

### Approach A: Full Native Pipeline (RECOMMENDED)

Start from the raw video, run openMVG SfM with spherical camera model, then
train OmniGS. This is OmniGS's intended workflow and avoids any perspective
decomposition artifacts.

1. Extract equirectangular frames from video
2. Run openMVG SfM with spherical camera model to get poses + sparse points
3. Train OmniGS on equirectangular frames with spherical poses

### Approach B: Convert COLMAP Poses (FALLBACK)

Use existing COLMAP poses from perspective reconstruction. Faster but poses
may not be perfectly aligned with equirectangular projection.

1. Run `colmap_to_omnigs.py` to convert COLMAP poses to openMVG format
2. Train OmniGS on equirectangular frames with converted poses

## Building OmniGS with CUDA 12.8

OmniGS was developed for CUDA 11.8. Building with CUDA 12.8 requires
adaptation. Key changes needed:

### CUDA API changes (11.8 → 12.8)
- Some CUDA texture APIs deprecated/removed — may need `#include` fixes
- cuDNN 8.x → 9.x API changes (if used)
- PTX forward compatibility should work for the Blackwell GPU (SM 10.0)

### LibTorch version
Use LibTorch 2.5+ with CUDA 12.8 instead of 2.0.1+cu118:
```bash
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124.zip
export Torch_DIR=/root/libtorch
```
Note: cu124 LibTorch works with CUDA 12.8 driver. Check https://download.pytorch.org/libtorch/ for latest cu124 or cu128 builds.

### Step-by-step build

```bash
# 1. System dependencies
apt update && apt install -y \
    build-essential cmake git wget unzip \
    libeigen3-dev libjsoncpp-dev libopengl-dev mesa-utils \
    libglfw3-dev libglm-dev libpng-dev libjpeg-dev libtiff-dev \
    gcc-12 g++-12
export CC=gcc-12 CXX=g++-12

# 2. CUDA paths (already installed at /usr/local/cuda-12.8)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 3. LibTorch (match CUDA 12.x)
cd /root
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124.zip
export Torch_DIR=/root/libtorch

# 4. OpenCV 4.7+ with CUDA (30-45 min compile)
cd /root
git clone --branch 4.10.0 --depth 1 https://github.com/opencv/opencv.git
git clone --branch 4.10.0 --depth 1 https://github.com/opencv/opencv_contrib.git

mkdir opencv/build && cd opencv/build
cmake -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib/modules \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN="8.0;8.6;8.9;9.0;10.0" \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    ..
make -j$(nproc)
make install

# NOTE: If CUDA 12.8 doesn't support SM 10.0 for OpenCV CUDA kernels,
# try removing "10.0" from CUDA_ARCH_BIN. The GPU has forward compat.

# 5. openMVG (for SfM with spherical cameras)
cd /root
git clone --recursive https://github.com/openMVG/openMVG.git
mkdir openMVG_build && cd openMVG_build
cmake -DCMAKE_BUILD_TYPE=Release /root/openMVG/src
make -j$(nproc)
# Binaries at: /root/openMVG_build/Linux-x86_64-Release/

# 6. Clone and build OmniGS
cd /root
git clone --recursive https://github.com/liquorleaf/OmniGS.git
cd OmniGS

# Edit build.sh — set Torch_DIR
sed -i 's|Torch_DIR=.*|Torch_DIR=/root/libtorch|' build.sh

# Edit CMakeLists.txt — update CUDA architectures for 12.8
# Change: set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "75;86")
# To:     set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "80;86;89;90")
# Note: SM 10.0 (Blackwell) may need CUDA 12.8+ arch support in cmake.
# If "100" isn't recognized, use "90" and rely on forward compatibility.
sed -i 's/CUDA_ARCHITECTURES "75;86"/CUDA_ARCHITECTURES "80;86;89;90"/' CMakeLists.txt

# If there are cuDNN version mismatches, you may need to install cuDNN 9.x:
# apt install libcudnn9-dev-cuda-12 OR download from NVIDIA developer site

chmod +x build.sh
./build.sh

# If build fails, common fixes:
# - Missing cuDNN: apt install libcudnn9-dev-cuda-12
# - C++ standard mismatch: add -DCMAKE_CXX_STANDARD=17 to cmake
# - Torch ABI mismatch: ensure LibTorch is cxx11-abi version
# - OpenCV not found: add -DOpenCV_DIR=/usr/local/lib/cmake/opencv4
```

## Data Preparation: Approach A (Native Pipeline)

### Step 1: Extract equirectangular frames

```bash
cd /root/omnigs_eval
pip install opencv-python-headless numpy

python3 -c "
import cv2, os
os.makedirs('data/BR03-054/equirect_frames', exist_ok=True)
cap = cv2.VideoCapture('data/BR03-054/video.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
every = 10  # match the original extraction interval
count = 0
for fi in range(0, total, every):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret: continue
    # Do NOT flip — keep raw orientation consistent with openMVG processing
    cv2.imwrite(f'data/BR03-054/equirect_frames/frame{fi:06d}.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    count += 1
cap.release()
print(f'Extracted {count} frames')
"
```

### Step 2: Run openMVG SfM with spherical camera

This follows OmniGS's data prep scripts. The key difference from standard SfM
is using the `spherical` camera model.

```bash
OPENMVG=/root/openMVG_build/Linux-x86_64-Release
IMAGES=/root/omnigs_eval/data/BR03-054/equirect_frames
SFM_OUT=/root/omnigs_eval/data/BR03-054/sfm_output

mkdir -p $SFM_OUT

# a. Create image listing with spherical camera
$OPENMVG/openMVG_main_SfMInit_ImageListing \
    -i $IMAGES \
    -o $SFM_OUT \
    -c 7 \  # camera model 7 = spherical
    -f 1    # focal (ignored for spherical, but required)

# b. Compute SIFT features
$OPENMVG/openMVG_main_ComputeFeatures \
    -i $SFM_OUT/sfm_data.json \
    -o $SFM_OUT

# c. Compute matches
$OPENMVG/openMVG_main_ComputeMatches \
    -i $SFM_OUT/sfm_data.json \
    -o $SFM_OUT

# d. Incremental SfM
$OPENMVG/openMVG_main_SfM \
    -s INCREMENTAL \
    -i $SFM_OUT/sfm_data.json \
    -m $SFM_OUT \
    -o $SFM_OUT/reconstruction

# e. Triangulate structure from known poses (if using pre-computed poses)
# Only needed for Approach B:
# $OPENMVG/openMVG_main_ComputeStructureFromKnownPoses \
#     -i $SFM_OUT/reconstruction/sfm_data.json \
#     -m $SFM_OUT \
#     -o $SFM_OUT/reconstruction/sfm_data_with_structure.json
```

If openMVG SfM succeeds, the output at `$SFM_OUT/reconstruction/sfm_data.json`
has everything OmniGS needs: camera poses + sparse 3D points + spherical camera model.

### Step 2 (Alternative): Use converted COLMAP poses

If openMVG SfM fails or is too slow:
```bash
pip install scipy
python3 data/BR03-054/colmap_to_omnigs.py \
    --video data/BR03-054/video.mp4 \
    --colmap data/BR03-054/colmap_sparse \
    --output data/BR03-054/omnigs \
    --every 10
```

Then you still need openMVG to triangulate sparse points:
```bash
$OPENMVG/openMVG_main_ComputeFeatures \
    -i data/BR03-054/omnigs/data_openmvg.json \
    -o data/BR03-054/omnigs

$OPENMVG/openMVG_main_ComputeStructureFromKnownPoses \
    -i data/BR03-054/omnigs/data_openmvg.json \
    -m data/BR03-054/omnigs \
    -o data/BR03-054/omnigs/data_openmvg_with_structure.json
```

## Training OmniGS

### Create config

Create `/root/OmniGS/cfg/lonlat/manhole_lonlat.yaml`:

```yaml
Model:
  sh_degree: 3

OptimizationParams:
  iterations: 32000
  position_lr_init: 0.00016
  position_lr_final: 0.0000016
  position_lr_delay_mult: 0.01
  position_lr_max_steps: 32000
  feature_lr: 0.0025
  opacity_lr: 0.05
  scaling_lr: 0.005
  rotation_lr: 0.001
  percent_dense: 0.01
  lambda_dssim: 0.2
  densification_interval: 100
  opacity_reset_interval: 3000
  densify_from_iter: 500
  densify_until_iter: 15000
  densify_grad_threshold: 0.0002

Camera:
  type: "Lonlat"
  width: 3840
  height: 1920

Dataset:
  skip_bottom_ratio: 0.0
```

### Run training

```bash
cd /root/OmniGS

# Using native openMVG SfM output:
./bin/train_openmvg_lonlat \
    cfg/lonlat/manhole_lonlat.yaml \
    /root/omnigs_eval/data/BR03-054/sfm_output/reconstruction \
    /root/omnigs_eval/results/BR03-054 \
    no_viewer

# Or using converted COLMAP poses:
./bin/train_openmvg_lonlat \
    cfg/lonlat/manhole_lonlat.yaml \
    /root/omnigs_eval/data/BR03-054/omnigs \
    /root/omnigs_eval/results/BR03-054 \
    no_viewer
```

Output: `/root/omnigs_eval/results/BR03-054/32000/ply/point_cloud/iteration_32000/point_cloud.ply`

## Rendering / Viewing Results

The built-in viewer requires OpenGL (won't work headless). Render to images instead:

```bash
./bin/test_openmvg_lonlat \
    cfg/lonlat/manhole_lonlat.yaml \
    /root/omnigs_eval/data/BR03-054/sfm_output/reconstruction \
    /root/omnigs_eval/results/BR03-054/32000/ply/point_cloud/iteration_32000/point_cloud.ply \
    /root/omnigs_eval/results/BR03-054/test_renders
```

This renders equirectangular images from the training viewpoints. You can then
view these as 360° images or convert to perspective views.

To transfer results back:
```bash
# From the remote GPU machine (10.227.1.102):
scp -P 34515 -i ~/.ssh/id_ed25519 \
    root@213.173.111.113:/root/omnigs_eval/results/BR03-054/32000/ply/point_cloud/iteration_32000/point_cloud.ply \
    ~/devel/pipeaid/360_video/omnigs_output/BR03-054.ply
```

## Video Details

- **Camera**: Ricoh Theta Z1 (likely, based on iCatchtek 6500 stitching, 3840x1920 resolution)
- **Mounting**: Upside-down (rotate=180 in metadata). OpenCV VideoCapture does NOT auto-apply rotation.
- **Content**: Manhole inspection — camera is lowered on a cable through a standard 48" precast concrete manhole
- **Duration**: ~43 seconds, ~1300 frames at 30fps
- **COLMAP results**: 1030/1032 perspective images registered (99.8%), 0.79px reprojection error, 510K sparse points

## Troubleshooting

- **OmniGS build fails with CUDA arch errors**: Try `CUDA_ARCHITECTURES "80;86;89"` (drop 90/100 if not recognized by cmake)
- **LibTorch version mismatch**: Use the exact cu124 build from PyTorch downloads. cxx11 ABI version required.
- **OpenCV CUDA compile errors**: Try OpenCV 4.10.0 instead of 4.7.0 for better CUDA 12.x support
- **openMVG spherical SfM fails**: Fall back to Approach B (convert COLMAP poses). The COLMAP reconstruction is solid — the equirectangular poses derived from it should be very close.
- **OOM during training**: Unlikely with 98GB VRAM. If it happens, reduce resolution in the config yaml.
- **Blackwell GPU not recognized**: Set CUDA_ARCHITECTURES to "89;90" and rely on forward compatibility. The PTX code compiled for SM 9.0 should run on SM 10.0 via JIT compilation.
