### This repository is archived for now and no longer maintained. For more implementation, please visit the GPU version: [WebcamVisionGPU](https://github.com/Tom-Ole/WebcamVisionGPU)

# Implementation of Computer Vision / Image Processing Effects on the CPU
This repository contains implementations of various computer vision and image processing effects. 
The goal is to create simple, efficient algorithms that can be run in real-time on Webcam streams.

This is a personal project to learn and experiment with different techniques.

A GPU implementation of these and more Algorithm can be found in my other repository: [WebcamVisionGPU](https://github.com/Tom-Ole/WebcamVisionGPU)




# Ideas Computer Vision / Image Processing Effects to Implement:
- [x] Grayscale Conversion
- [x] Gaussian Blur
- [x] Color filter
- [x] Histogram Equalization
- [x] "Depth Map"
- [] Median Filter
- [] Sepia Tone Filter
- [] Invert Colors
- [x] Sobel Edge Detection
- [x] Canny Edge Detection
- [x] Motion Detection
- [] Background Subtraction
- [] Optical Flow (Lucas–Kanade or Horn–Schunck Lite)
- [] Hough Transform (for Lines or Circles)
- [] Corner Detectors (Harris / Shi–Tomasi)
- [] Depth from Motion (Structure from Motion Lite)
- [] Cartoon / Pencil Drawing Filter
- [] Kuwahara / Oil Painting Filter
- [] Voronoi Mosaic / Cell Shading
- [] Color Object Tracking [Convert to HSV → isolate hue range → centroid of region = object position.]
- [] Feature Matching [Detect keypoints (corners) → extract simple descriptors → match across frames.]
- [] Optical Flow–Based Stabilization

- [] Depth illusion: simulate fake parallax using grayscale brightness as depth
- [] Motion trails: accumulate motion over frames to create ghostly afterimages
- [] Thermal camera: map brightness → color palette.

