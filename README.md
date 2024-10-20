# REAL Video Enhancer
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FTNTwise%2FREAL-Video-enhancer%2F&countColor=%23263759)
![downloads_total](https://img.shields.io/github/downloads/tntwise/REAL-Video-Enhancer/total.svg?label=downloads%40total)
[![pypresence](https://img.shields.io/badge/using-pypresence-00bb88.svg?style=for-the-badge&logo=discord&logoWidth=20)](https://github.com/qwertyquerty/pypresence)
<a href="https://discord.gg/hwGHXga8ck">
      <img src="https://img.shields.io/discord/1041502781808328704?label=Discord" alt="Discord Shield"/></a>

<p align=center>
  <img src="https://github.com/TNTwise/REAL-Video-Enhancer/raw/2.0/icons/logo-v2.svg" width = "25%">
</p>

# Table of Contents
  
* **[Introduction](#introduction)**
* **[Features](#Features)**
* **[Hardware Requirements](#hardware-requirements)**
* **[Benchmarks](#benchmarks)**
  * [NCNN](#rife-ncnn)
  * [TensorRT](#rife-tensorrt-103)
* **[Cloning](#cloning)**
* **[Building](#building)**
* **[Canary build](#canary-build)**
* **[Credits](#credits)**
  * [People](#people) 
  * [Software](#software)
* **[Custom Models](#custom-models)**

# Introduction

<strong>REAL Video Enhancer</strong>  is a redesigned and enhanced version of the original Rife ESRGAN App for Linux. This program offers convenient access to frame interpolation and upscaling functionalities on Linux, and is an alternative to outdated software like <a rel="noopener noreferrer" href="https://nmkd.itch.io/flowframes" target="_blank" >Flowframes</a> or <a rel="noopener noreferrer" href="https://github.com/mafiosnik777/enhancr" target="_blank">enhancr</a> on Windows.

V2 Alpha 2 New Look!:
<p align=center>
  <img src="https://github.com/TNTwise/REAL-Video-Enhancer/blob/2.0/icons/demo.png" width = "100%">
</p>
<h1>Features: </h1>
<ul>
  <li> <strong>NEW!</strong> Windows support. <strong>!!! NOTICE !!!</strong> The bin can be detected as a trojan. This is a false positive caused by pyinstaller.</li>
  <li> MacOS support. (Depricated as 2.0, use 1.2 for now) </li>
  <li> Support for Ubuntu 20.04+ on Executable and Flatpak. </li>
  <li> Discord RPC support for Discord system package and Discord flatpak. </li>
  <li> Scene change detection to preserve sharp transitions. </li>
  <li> Preview that shows latest frame that has been rendered. </li>
  <li> TensorRT and NCNN for efficient inference across many GPUs. </li>
</ul>

# Hardware/Software Requirements

|  | Minimum | Recommended | 
 |--|--|--|
| CPU | Dual Core x64 bit | Quad core x64 bit
| GPU | Vulkan 1.3 capable device | Nvidia RTX GPU (20 series and up)
| RAM | 8 GB | 16 GB
| Storage | 1 GB free (NCNN install only) | 10 GB free (TensorRT install)
| Operating System | Windows 10/11 64bit | Any modern Linux distro (Ubuntu 20.04+)
# Benchmarks:

Benchmarks done with 1920x1080 video, default settings.

### RIFE NCNN


| RX 6650 XT | |
 |--|--|
| rife-v4.6 | 31 fps
| rife-v4.7 - v4.9 | 28 fps
| rife-v4.10 - v4.15 | 23 fps
| rife-v4.16-lite | 31 fps

| RTX 3080 | |
|--|--|
| rife-v4.6 | 81 fps
| rife-v4.7 - v4.9 | 65 fps
| rife-v4.10 - v4.15 | 55 fps
| rife-v4.22 | 50 fps
| rife-v4.22-lite | 63 fps

### RIFE TensorRT 10.3
| RTX 3080 | |
|--|--|
| rife-v4.6 | 270 fps
| rife-v4.7 - v4.9 | 204 fps
| rife-v4.10 - v4.15 | 166 fps
| rife-v4.22 | 140 fps
| rife-v4.22-lite | 192 fps

# Cloning:
```
git clone https://github.com/TNTwise/REAL-Video-Enhancer
```
# Building:
```
python3 build.py --build_exe
```
# Canary Build:
<strong> </strong> <a href="https://github.com/TNTwise/REAL-Video-Enhancer/releases/tag/prerelease">https://github.com/TNTwise/REAL-Video-Enhancer/releases/tag/prerelease</a>

# Credits:
### People:
| Person | For | Link||
|--|--|--|--
| NevermindNilas | Some backend and reference code and working with me on many projects | https://github.com/NevermindNilas/ |
| Styler00dollar | RIFE models (4.1-4.5, 4.7-4.12-lite), Sudo Shuffle Span and benchmarking | https://github.com/styler00dollar |
| HolyWu | TensorRT engine generation code, inference optimizations, and RIFE jagged lines fixes | https://github.com/HolyWu/ |
| Rick Astley | Amazing music | https://www.youtube.com/watch?v=dQw4w9WgXcQ |


### Software: 
| Software Used | For | Link||
|--|--|--|--
| FFmpeg | Multimedia framework for handling video, audio, and other media files |
https://ffmpeg.org/ 
| PyTorch | Neural Network Inference (CUDA/ROCm) | https://pytorch.org/ 
| NCNN | Neural Network Inference (Vulkan) | https://github.com/tencent/ncnn 
| RIFE | Real-Time Intermediate Flow Estimation for Video Frame Interpolation | https://github.com/hzwer/Practical-RIFE 
| rife-ncnn-vulkan | Video frame interpolation implementation using NCNN and Vulkan | https://github.com/nihui/rife-ncnn-vulkan 
| rife ncnn vulkan python | Python bindings for RIFE NCNN Vulkan implementation | https://github.com/media2x/rife-ncnn-vulkan-python 
| ncnn python | Python bindings for NCNN Vulkan framework | https://pypi.org/project/ncnn 
| Real-ESRGAN | Upscaling | https://github.com/xinntao/Real-ESRGAN 
| SPAN | Upscaling | https://github.com/hongyuanyu/SPAN 
| Spandrel | CUDA upscaling model architecture support | https://github.com/chaiNNer-org/spandrel 
| cx_Freeze | Tool for creating standalone executables from Python scripts (Linux build) | https://github.com/marcelotduarte/cx_Freeze 
| PyInstaller | Tool for creating standalone executables from Python scripts (Windows/Mac builds) | https://github.com/marcelotduarte/cx_Freeze 
| Feather Icons | Open source icons library | https://github.com/feathericons/feather 


# Custom models:

| Model | Author | Link |
|--|--|--|
| 4x-SPANkendata | Crustaceous D | [4x-SPANkendata](https://openmodeldb.info/models/4x-SPANkendata) 
| 4x-ClearRealityV1 | Kim2091 | [4x-ClearRealityV1](https://openmodeldb.info/models/4x-ClearRealityV1) 
| 4x-Nomos8k-SPAN series | Helaman | [4x-Nomos8k-SPAN series](https://openmodeldb.info/models/4x-Nomos8k-span-otf-strong) 
| OpenProteus | SiroSky | [OpenProteus](https://github.com/Sirosky/Upscale-Hub/releases/tag/OpenProteus) 
