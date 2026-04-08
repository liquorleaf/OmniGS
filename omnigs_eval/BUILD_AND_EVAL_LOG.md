# OmniGS Evaluation on RTX PRO 6000 Blackwell — Build & Pipeline Log

## Overview

Full end-to-end evaluation of [OmniGS](https://github.com/liquorleaf/OmniGS) for
photorealistic 3D reconstruction of a manhole interior from 360° equirectangular
video. OmniGS natively supports spherical (lon-lat) camera models, avoiding the
quality loss from decomposing equirectangular frames into perspective cubemap faces.

**Hardware**: NVIDIA RTX PRO 6000 Blackwell (98 GB VRAM, SM 12.0)  
**CUDA**: 12.8, driver 580  
**OS**: Ubuntu 24.04, GCC 13.3 (host), GCC 12.4 (CUDA host compiler)  
**Date**: 2026-04-08

---

## 1. System Dependencies

```bash
apt update && apt install -y \
    build-essential cmake git wget unzip \
    libeigen3-dev libjsoncpp-dev libopengl-dev mesa-utils \
    libglfw3-dev libglm-dev libpng-dev libjpeg-dev libtiff-dev \
    gcc-12 g++-12
```

GCC 12 is required as the CUDA host compiler because nvcc 12.8 triggers an
internal compiler error (ICE) with GCC 13's `<stdfloat>` header
(`"unsupported float variant!"`).

---

## 2. LibTorch

### Initial attempt: LibTorch 2.5.1+cu124

```bash
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip
```

This failed at runtime: `"no kernel image is available for execution on the device"`.
LibTorch 2.5.1+cu124 was compiled for SM up to 9.0a and does not include kernels
for SM 12.0 (Blackwell).

### Working version: LibTorch 2.7.1+cu128

```bash
wget https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu128.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.7.1+cu128.zip
export Torch_DIR=/root/libtorch
```

LibTorch 2.7.1+cu128 includes native SM 12.0 (Blackwell) kernel support.

---

## 3. OpenCV 4.10.0 with CUDA

```bash
git clone --branch 4.10.0 --depth 1 https://github.com/opencv/opencv.git
git clone --branch 4.10.0 --depth 1 https://github.com/opencv/opencv_contrib.git

cmake -S opencv -B opencv/build \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=opencv_contrib/modules \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN="9.0" \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF

cmake --build opencv/build -j$(nproc)
cmake --install opencv/build
ldconfig
```

**Key decision**: Only SM 9.0 in `CUDA_ARCH_BIN`. Initially tried `"8.0;8.6;8.9;9.0"`
which caused a 45+ minute build. Blackwell runs SM 9.0 PTX via forward compatibility
(JIT compilation), so a single arch is sufficient and cuts build time to ~15 minutes.

---

## 4. openMVG

```bash
git clone --recursive https://github.com/openMVG/openMVG.git
cmake -S openMVG/src -B openMVG_build -DCMAKE_BUILD_TYPE=Release
cmake --build openMVG_build -j$(nproc)
```

Binaries at `/root/openMVG_build/Linux-x86_64-Release/`.

---

## 5. OmniGS — Build with CUDA 12.8 Adaptations

```bash
git clone --recursive https://github.com/liquorleaf/OmniGS.git
```

### 5.1 Source Code Patches

OmniGS was developed for CUDA 11.8 + LibTorch 2.0. Building with
CUDA 12.8 + LibTorch 2.7 on a Blackwell GPU required eight patches:

#### Patch 1: CUDA architectures — SM 12.0

**File**: `CMakeLists.txt`

The original code targets SM 7.5 and 8.6. The Blackwell GPU reports SM 12.0
(compute capability 12.0), which is natively supported by CUDA 12.8's nvcc.

```diff
- set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "75;86")
+ set_target_properties(simple_knn PROPERTIES CUDA_ARCHITECTURES "120")
  ...
+ set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "120")
```

**Note**: SM 9.0 with PTX forward compatibility was attempted first but failed
at runtime with `cudaErrorNoKernelImageForDevice`. The Blackwell architecture
(SM 12.0) is not forward-compatible with SM 9.0 SASS — it requires either
native SM 12.0 compilation or SM 9.0 PTX (which CUDA JIT-compiles). However,
the OmniGS rasterizer uses thrust primitives that require native SASS. Compiling
directly for SM 12.0 resolved this.

#### Patch 2: Missing `<cstdint>` — `uint32_t` / `uintptr_t` undefined

**File**: `cuda_rasterizer/rasterizer_impl.h`

GCC 13 / C++17 no longer implicitly includes `<cstdint>` through transitive headers.

```diff
  #include <iostream>
  #include <vector>
+ #include <cstdint>
  #include "rasterizer.h"
```

#### Patch 3: Missing `<cfloat>` — `FLT_MAX` undefined

**File**: `third_party/simple-knn/simple_knn.cu`

```diff
  #include "cuda_runtime.h"
  #include "device_launch_parameters.h"
+ #include <cfloat>
  #include "simple_knn.h"
```

#### Patch 4: `M_1_PIf32` / `M_2_PIf32` — nvcc ICE with C++23 float types

**Files**: `cuda_rasterizer/auxiliary.h`, `cuda_rasterizer/forward.cu`, `cuda_rasterizer/backward.cu`

GCC 13 defines `M_1_PIf32` and `M_2_PIf32` as `std::float32_t` constants via
`<stdfloat>`. nvcc 12.8 cannot compile code paths involving `std::float32_t`,
triggering: `Internal Compiler Error (codegen): "unsupported float variant!"`.

Replaced all occurrences with standard `M_1_PI` / `M_2_PI` cast to `float`:

```diff
- float W_div_2pi = width * 0.5f * M_1_PIf32;
- float H_div_pi = height * M_1_PIf32;
+ float W_div_2pi = width * 0.5f * (float)M_1_PI;
+ float H_div_pi = height * (float)M_1_PI;
```

```diff
- (lon * M_1_PIf32 + 1.0f) * (width >> 1),
- (lat * M_2_PIf32 + 1.0f) * (height >> 1)
+ (lon * (float)M_1_PI + 1.0f) * (width >> 1),
+ (lat * (float)M_2_PI + 1.0f) * (height >> 1)
```

#### Patch 5: `c10::guts::to_string` removed — optimizer state key API change

**File**: `src/gaussian_model.cpp` (7 occurrences)

In LibTorch 2.5+, the optimizer state map changed from `ska::flat_hash_map<std::string, ...>`
to `ska::flat_hash_map<void*, ...>`. The old string-based key via
`c10::guts::to_string(param.unsafeGetTensorImpl())` was removed.

```diff
- auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
+ auto key = static_cast<void*>(param.unsafeGetTensorImpl());
```

#### Patch 6: `torch::cat` with brace-enclosed initializer list

**File**: `src/gaussian_model.cpp`

LibTorch 2.7 changed `torch::cat` to accept `at::ITensorListRef` which is
ambiguous with brace-init-lists. Explicit `std::vector<torch::Tensor>` resolves this.

```diff
- param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
+ param = torch::cat(std::vector<torch::Tensor>{param, extension_tensor}, /*dim=*/0).requires_grad_();
```

#### Patch 7: `c10Alloc::Stat` / `StatType` — moved to `c10::CachingDeviceAllocator`

**Files**: All 4 example files (`train_openmvg_lonlat.cpp`, `train_360roam.cpp`,
`train_egonerf_omniblender.cpp`, `train_egonerf_ricoh360.cpp`)

```diff
- c10Alloc::Stat reserved_bytes = mem_stats.reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
+ c10::CachingDeviceAllocator::Stat reserved_bytes = mem_stats.reserved_bytes[static_cast<int>(c10::CachingDeviceAllocator::StatType::AGGREGATE)];
```

#### Patch 8: TORCH_CUDA_ARCH_LIST environment variable

LibTorch's bundled FindCUDA auto-detects the GPU and gets "12.0", which its
`select_compute_arch.cmake` doesn't recognize. Setting `TORCH_CUDA_ARCH_LIST`
at cmake time overrides auto-detection:

```bash
TORCH_CUDA_ARCH_LIST="12.0" cmake ...
```

### 5.2 Build Command

```bash
cd OmniGS
# Build Sophus (header-only, generates cmake config)
mkdir -p third_party/Sophus/build && cd third_party/Sophus/build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
cd ../../..

# Build OmniGS
TORCH_CUDA_ARCH_LIST="12.0" cmake -S . -B build \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
    -DTorch_DIR=/root/libtorch/share/cmake/Torch \
    -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -Wno-dev

cmake --build build -j$(nproc)
```

**Result**: All 7 executables and 5 shared libraries built successfully.

---

## 6. Data Preparation

### 6.1 Input Video

- **File**: `/root/omnigs_eval/data/BR03-054/video.mp4`
- **Camera**: Ricoh Theta Z1 (equirectangular, 3840x1920, 30fps)
- **Duration**: ~43 seconds, 1290 frames
- **Content**: Manhole inspection — camera lowered on cable through 48" precast manhole
- **Mounting**: Upside-down (rotate=180 in metadata)

### 6.2 Frame Extraction — Sharpest Frame Selection

Used [Extract_sharpest_frame](https://github.com/Kotohibi/Extract_sharpest_frame)
to select the sharpest frame from each group of 10 consecutive frames, using
Laplacian variance as the sharpness metric.

```bash
git clone https://github.com/Kotohibi/Extract_sharpest_frame.git
pip install opencv-python-headless tqdm

python3 extract_sharpest_frame.py \
    --video /root/omnigs_eval/data/BR03-054/video.mp4 \
    --chunk-size 10 \
    --scale-width 3840 \
    --output-dir /root/omnigs_eval/data/BR03-054/equirect_frames \
    --output-format jpg \
    --jpeg-quality 95 \
    --workers $(nproc)
```

**Result**: 129 frames extracted at full 3840x1920 resolution.

This approach is superior to naive every-Nth-frame extraction because it avoids
motion-blurred frames (common when the camera swings on the cable).

### 6.3 openMVG SfM with Spherical Camera Model

```bash
OPENMVG=/root/openMVG_build/Linux-x86_64-Release
IMAGES=/root/omnigs_eval/data/BR03-054/equirect_frames
SFM_OUT=/root/omnigs_eval/data/BR03-054/sfm_output

# Step 1: Image listing with spherical camera (model 7)
$OPENMVG/openMVG_main_SfMInit_ImageListing \
    -i $IMAGES -o $SFM_OUT -c 7 -f 1

# Step 2: SIFT feature extraction (HIGH preset)
$OPENMVG/openMVG_main_ComputeFeatures \
    -i $SFM_OUT/sfm_data.json -o $SFM_OUT -m SIFT -p HIGH

# Step 3: Putative matching
$OPENMVG/openMVG_main_ComputeMatches \
    -i $SFM_OUT/sfm_data.json -o $SFM_OUT/matches.putative.bin

# Step 4: Geometric filtering
$OPENMVG/openMVG_main_GeometricFilter \
    -i $SFM_OUT/sfm_data.json \
    -m $SFM_OUT/matches.putative.bin \
    -o $SFM_OUT/matches.f.bin

# Step 5: Incremental SfM
$OPENMVG/openMVG_main_SfM \
    -s INCREMENTAL \
    -i $SFM_OUT/sfm_data.json \
    -m $SFM_OUT \
    -o $SFM_OUT/reconstruction
```

**SfM Results**:

| Metric | Value |
|--------|-------|
| Registered cameras | 129 / 129 (100%) |
| 3D points | 221,470 |
| Putative pairs | 8,256 |
| Connected components | 1 |
| Node degree (mean) | 73 matches per image |
| SfM time | 648 seconds |
| Camera model | Spherical (openMVG model 7) |

### 6.4 Export for OmniGS

OmniGS expects a specific directory layout:

```
<dataset>/openMVG/reconstruction/poses.json     # openMVG sfm_data.json
<dataset>/openMVG/reconstruction/colorized.ply   # Colored sparse point cloud
```

```bash
# Convert binary SfM data to JSON
$OPENMVG/openMVG_main_ConvertSfM_DataFormat \
    -i $SFM_OUT/reconstruction/sfm_data.bin \
    -o $SFM_OUT/reconstruction/sfm_data.json

# Create expected directory structure
mkdir -p data/BR03-054/openMVG/reconstruction
ln -s $SFM_OUT/reconstruction/sfm_data.json \
      data/BR03-054/openMVG/reconstruction/poses.json

# Export colorized point cloud
$OPENMVG/openMVG_main_ComputeSfM_DataColor \
    -i $SFM_OUT/reconstruction/sfm_data.bin \
    -o data/BR03-054/openMVG/reconstruction/colorized.ply
```

---

## 7. OmniGS Training

### 7.1 Configuration

File: `OmniGS/cfg/lonlat/manhole_lonlat.yaml` (OpenCV FileStorage `%YAML:1.0` format)

Key parameters:
- **Iterations**: 32,010
- **SH degree**: 3
- **Densification**: iterations 500–15,000, interval 100, gradient threshold 0.0002
- **Learning rates**: position 1.6e-4 → 1.6e-6, feature 2.5e-3, opacity 0.05
- **Loss**: L1 + 0.2 * D-SSIM
- **skip_bottom_ratio**: 0.0 (use full equirectangular frame)

### 7.2 Training Command

```bash
export LD_LIBRARY_PATH=/root/OmniGS/lib:/root/libtorch/lib:/usr/local/lib:$LD_LIBRARY_PATH

./bin/train_openmvg_lonlat \
    cfg/lonlat/manhole_lonlat.yaml \
    /root/omnigs_eval/data/BR03-054 \
    /root/omnigs_eval/results/BR03-054 \
    no_viewer
```

### 7.3 Training Results

| Metric | Value |
|--------|-------|
| Total time | ~37 minutes |
| Speed | ~0.07 s/iteration |
| Final Gaussians | 676,010 (from 221K sparse points) |
| EMA loss @ 10K | 0.0571 |
| EMA loss @ 20K | 0.0497 |
| EMA loss @ 30K | 0.0571 |
| VRAM usage | ~26 GB peak (of 98 GB available) |
| Checkpoints saved | 8K, 16K, 24K, 32K iterations |

---

## 8. Test Rendering & Evaluation

### 8.1 Render Command

```bash
# Create test data symlink (renders from training viewpoints)
ln -s sfm_output/reconstruction/sfm_data.json \
      data/BR03-054/openMVG/data_openmvg_test.json

./bin/test_openmvg_lonlat \
    cfg/lonlat/manhole_lonlat.yaml \
    /root/omnigs_eval/data/BR03-054 \
    /root/omnigs_eval/results/BR03-054/32000/ply/point_cloud/iteration_32000/point_cloud.ply \
    /root/omnigs_eval/results/BR03-054/test_renders
```

### 8.2 Quality Metrics (129 test views)

| Metric | Mean | Median | Min | Max |
|--------|------|--------|-----|-----|
| **PSNR** | 25.30 dB | 25.78 dB | 16.78 dB | 28.72 dB |
| **SSIM** | 0.849 | 0.876 | 0.671 | 0.910 |
| **Render time** | 14.5 ms | 15.0 ms | 10.9 ms | 30.2 ms |
| **Render FPS** | ~70 FPS | — | — | — |

### 8.3 Per-Region Analysis

Lower PSNR on frames 0–5 and 125–128 corresponds to the camera entering and
exiting the manhole (limited overlap, rapid motion). Mid-sequence frames (60–115)
achieve 27–29 dB PSNR, representing the stationary inspection portion with
maximum multi-view coverage.

---

## 9. Output Files

```
/root/omnigs_eval/results/BR03-054/
├── test_renders/0_test/
│   ├── image/                    # 129 rendered equirectangular JPGs (3840x1920)
│   ├── psnr.txt                  # Per-frame PSNR
│   ├── dssim.txt                 # Per-frame SSIM
│   ├── psnr_gaussian_splatting.txt
│   └── render_time.txt           # Per-frame render time (ms)
├── 32000/
│   ├── ply/point_cloud/iteration_32000/
│   │   └── point_cloud.ply       # Final model (160 MB, 676K Gaussians)
│   └── image/                    # Training checkpoint renders
├── 8000/  16000/  24000/         # Intermediate checkpoints
├── 32081_shutdown/               # Final shutdown checkpoint
└── used_times/
    └── keyframe_used_times.txt
```

---

## 10. Lessons Learned

1. **Blackwell is SM 12.0, not SM 10.0.** The RTX PRO 6000 Blackwell reports
   compute capability 12.0. CUDA 12.8 supports `sm_120` natively. Forward
   compatibility from SM 9.0 PTX does not work reliably for all kernel types.

2. **LibTorch version must match GPU generation.** cu124 builds max out at SM 9.0a.
   Blackwell requires cu128 builds (LibTorch 2.7+).

3. **GCC 13 + nvcc 12.8 = ICE.** The `std::float32_t` type from GCC 13's
   `<stdfloat>` header triggers internal compiler errors in nvcc. Use GCC 12
   as the CUDA host compiler.

4. **Single CUDA arch dramatically speeds builds.** OpenCV CUDA with 4 architectures
   takes 45+ minutes. Single arch (SM 9.0 for OpenCV, SM 12.0 for OmniGS) cuts
   this to ~15 minutes.

5. **LibTorch API churn.** Between 2.0 and 2.7, several APIs changed:
   - Optimizer state keys: `string` → `void*`
   - `c10::guts::to_string` removed
   - `c10Alloc::Stat` → `c10::CachingDeviceAllocator::Stat`
   - `torch::cat` initializer list overload resolution changed

6. **Sharpest-frame extraction matters for SfM.** Using Laplacian variance to
   select the sharpest frame per chunk avoids motion-blurred frames that degrade
   feature matching, especially in a cable-lowered inspection scenario.

7. **openMVG spherical SfM works well for 360° video.** 100% camera registration
   with 221K 3D points from 129 equirectangular frames, with no need for
   perspective decomposition.
