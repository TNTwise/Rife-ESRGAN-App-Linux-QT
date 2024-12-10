
# RVE 2.1.5
### Added
 - Stopping render, instead of having to kill the entire app.
 - More tiling options.
 - Better checks on imported models.
 - Subtitle passthrough.
 - Changelog view in home menu.
 - Upscale and Interpolate at the same time.
 - Sudo shuffle span for pytorch backend
 - [GIMM-VFI](https://github.com/GSeanCDAT/GIMM-VFI)
 - GMFSS Pro, which helps fix text warping.
 - SloMo mode
### Changed
 - Make RVE smaller by switching to pyside6-essentials. (thanks zeptofine!) 
 - Make GUI more compact.
 - Bump torch to 2.6.0-dev20241206.
 - Remove CUDA install requirement for GMFSS
# RVE 2.1.0
### Added
 - Custom Upscale Model Support (TensorRT/Pytorch/NCNN)
 - GMFSS
 - RIFE 4.25 (TensorRT/Pytorch/NCNN)
 - PySceneDetect for better scene change detections
 - More NCNN models
### Removed
 - MacOS Support fully, not coming back due to changes made by Apple.
### Changed
 - Simplified TensorRT Engine Building
 - Increased RIFE TensorRT speed
 - Better preview that pads instead of stretches
 - Updated PyTorch to 2.6.0.dev20241023
 - Updated TensorRT to 10.6
 - Naming scheme of upscaling models, should be easier to understand


