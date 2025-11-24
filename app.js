// https://en.wikipedia.org/wiki/Edge_detection

// THis project is about the single algorithm and learning Computer Vision in generell. 
// I know there are some practices which are not really seen as "good"
// I will (maybe) refactor the code and also implement GPU versions of the Algos.


// =================================
// Constants and State
// =================================


const CONFIG = {
    CAM_WIDTH: 1920,
    CAM_HEIGHT: 1080,
    FRAME_RATE: 60,
    SMOOTH_FACTOR: 0.8,
    DEFAULT_DETECTOR: "motiondetection"
}

const KERNELS = {
    SOBEL_X: [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ],
    SOBEL_Y: [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]
}


const state = {
    detector: CONFIG.DEFAULT_DETECTOR,
    previousFrame: null,
    video: null,
    canvas: null,
    ctx: null,
    motionTrail: null
}


// =================================
// Utility Functions
// =================================

function pixelIndex(x, y, width) {
    return (y * width + x) * 4;
}

function rgbToGrayscale(r, g, b) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

function hexToRgb(hex) {
    const hexCode = parseInt(hex.slice(1), 16);
    return {
        r: hexCode >> 16,
        g: (hexCode >> 8) & 255,
        b: hexCode & 255
    };
}

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}


// =================================
// Image Processing Functions
// =================================

function smoothFrame(frameData) {
    const output = new Uint8ClampedArray(frameData.length);

    if (!state.previousFrame) {
        state.previousFrame = new Float32Array(frameData);
        return frameData;
    }

    for (let i = 0; i < frameData.length; i++) {
        state.previousFrame[i] =
            state.previousFrame[i] * CONFIG.SMOOTH_FACTOR +
            frameData[i] * (1 - CONFIG.SMOOTH_FACTOR);
        output[i] = state.previousFrame[i];
    }

    return output;
}

// Create a Gaussian kernel of size `size` for blur
function createGaussianKernel(size, sigma) {
    const halfSize = Math.floor(size / 2);
    const kernel = [];
    let sum = 0;

    // Create gaussian kernel value by index
    const gaussianFunction = (i, j) => {
        const sigma2 = sigma * sigma;
        const coefficient = 1 / (2 * Math.PI * sigma2);
        const exponent = -(i * i + j * j) / (2 * sigma2);
        return coefficient * Math.exp(exponent);
    };

    // Generate kernel values
    for (let j = 0; j < size; j++) {
        const row = [];
        for (let i = 0; i < size; i++) {
            const ii = i - halfSize;
            const jj = j - halfSize;
            const value = gaussianFunction(ii, jj);
            row.push(value);
            sum += value;
        }
        kernel.push(row);
    }

    // Normalize
    return kernel.map(row => row.map(val => val / sum));
}

// Apply Gaussian blur and convert to grayscale
function applyGaussianBlur(pixels, width, height, kernelSize, sigma) {
    const halfSize = Math.floor(kernelSize / 2);
    const kernel = createGaussianKernel(kernelSize, sigma);
    const output = new Uint8ClampedArray(width * height);

    for (let y = halfSize; y < height - halfSize; y++) {
        for (let x = halfSize; x < width - halfSize; x++) {
            let sum = 0;

            for (let ky = -halfSize; ky <= halfSize; ky++) {
                for (let kx = -halfSize; kx <= halfSize; kx++) {
                    const pixelIdx = pixelIndex(x + kx, y + ky, width);
                    const gray = rgbToGrayscale(
                        pixels[pixelIdx],
                        pixels[pixelIdx + 1],
                        pixels[pixelIdx + 2]
                    );
                    sum += gray * kernel[ky + halfSize][kx + halfSize];
                }
            }

            output[y * width + x] = Math.round(sum);
        }
    }

    return output;
}

function applySobel(grayscaleData, width, height) {
    const gradientMagnitude = new Float32Array(width * height);
    const gradientAngle = new Float32Array(width * height);

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let sumX = 0;
            let sumY = 0;

            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const idx = (y + ky) * width + (x + kx);
                    const gray = grayscaleData[idx];

                    sumX += gray * KERNELS.SOBEL_X[ky + 1][kx + 1];
                    sumY += gray * KERNELS.SOBEL_Y[ky + 1][kx + 1];
                }
            }

            const magnitude = Math.sqrt(sumX * sumX + sumY * sumY);
            const angle = Math.atan2(sumY, sumX);

            gradientMagnitude[y * width + x] = magnitude;
            gradientAngle[y * width + x] = angle;
        }
    }

    return { gradientMagnitude, gradientAngle };
}

// =================================
// Edge Detection Algos
// =================================

function applyNonMaxSuppression(gradientMagnitude, gradientAngle, width, height) {
    const output = new Float32Array(width * height).fill(0);

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = y * width + x;
            let angle = gradientAngle[idx];
            const magnitude = gradientMagnitude[idx];

            // Convert angle to degrees and map to [0, 180]
            let degree = angle * (180 / Math.PI);
            if (degree < 0) degree += 180;

            // Determine neighbor positions based on gradient direction
            let neighbor1X, neighbor1Y, neighbor2X, neighbor2Y;

            if ((degree >= 0 && degree < 22.5) || (degree >= 157.5 && degree <= 180)) {
                neighbor1X = x + 1; neighbor1Y = y;
                neighbor2X = x - 1; neighbor2Y = y;
            } else if (degree >= 22.5 && degree < 67.5) {
                neighbor1X = x + 1; neighbor1Y = y - 1;
                neighbor2X = x - 1; neighbor2Y = y + 1;
            } else if (degree >= 67.5 && degree < 112.5) {
                neighbor1X = x; neighbor1Y = y - 1;
                neighbor2X = x; neighbor2Y = y + 1;
            } else {
                neighbor1X = x - 1; neighbor1Y = y - 1;
                neighbor2X = x + 1; neighbor2Y = y + 1;
            }

            const mag1 = gradientMagnitude[neighbor1Y * width + neighbor1X];
            const mag2 = gradientMagnitude[neighbor2Y * width + neighbor2X];

            // Keep only local maxima
            if (magnitude >= mag1 && magnitude >= mag2) {
                output[idx] = magnitude;
            }
        }
    }

    return output;
}

function applyDoubleThreshold(nonMaxSuppressed, width, height, lowRatio, highRatio) {
    const output = new Float32Array(nonMaxSuppressed.length);

    // Find maximum magnitude
    let maxMag = 0;
    for (let i = 0; i < nonMaxSuppressed.length; i++) {
        if (nonMaxSuppressed[i] > maxMag) maxMag = nonMaxSuppressed[i];
    }

    const lowThreshold = maxMag * lowRatio;
    const highThreshold = maxMag * highRatio;

    for (let i = 0; i < nonMaxSuppressed.length; i++) {
        const mag = nonMaxSuppressed[i];

        if (mag <= lowThreshold) {
            output[i] = 0;
        } else if (mag <= highThreshold) {
            output[i] = 0.5; // Weak edge
        } else {
            output[i] = 1; // Strong edge
        }
    }

    return output;
}

function applyHysteresis(edgeMap, width, height) {
    const output = new Float32Array(edgeMap);

    const hasStrongNeighbor = (x, y) => {
        for (let j = -1; j <= 1; j++) {
            for (let i = -1; i <= 1; i++) {
                if (i === 0 && j === 0) continue;

                const nx = x + i;
                const ny = y + j;

                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

                if (output[ny * width + nx] === 1) {
                    return true;
                }
            }
        }
        return false;
    };

    let changed = true;
    while (changed) {
        changed = false;

        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = y * width + x;

                if (output[idx] === 0.5) {
                    if (hasStrongNeighbor(x, y)) {
                        output[idx] = 1;
                        changed = true;
                    }
                }
            }
        }
    }

    // Remove remaining weak edges
    for (let i = 0; i < output.length; i++) {
        if (output[i] === 0.5) {
            output[i] = 0;
        }
    }

    return output;
}


// =================================
// Filters
// =================================

function applyCanny(ctx, video, width, height) {
    ctx.drawImage(video, 0, 0, width, height);

    const frame = ctx.getImageData(0, 0, width, height);
    const pixels = smoothFrame(frame.data);

    // Gaussian blur
    const blurred = applyGaussianBlur(pixels, width, height, 5, 2);

    // Sobel gradient computation
    const { gradientMagnitude, gradientAngle } = applySobel(blurred, width, height);

    // Non-maximum suppression
    const nonMaxSuppressed = applyNonMaxSuppression(
        gradientMagnitude,
        gradientAngle,
        width,
        height
    );

    // Double thresholding
    const lowThreshold = parseFloat(document.getElementById("lowThreshold").value);
    const highThreshold = parseFloat(document.getElementById("highThreshold").value);
    let thresholded = applyDoubleThreshold(
        nonMaxSuppressed,
        width,
        height,
        lowThreshold,
        highThreshold
    );

    // Edge tracking by hysteresis
    thresholded = applyHysteresis(thresholded, width, height);

    // Generate output image
    const output = new Uint8ClampedArray(pixels.length);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const outIdx = pixelIndex(x, y, width);
            const edge = thresholded[y * width + x];

            if (edge === 1) {
                output[outIdx] = 0;
                output[outIdx + 1] = 255;
                output[outIdx + 2] = 0;
            } else if (edge === 0.5) {
                output[outIdx] = 255;
                output[outIdx + 1] = 0;
                output[outIdx + 2] = 0;
            } else {
                output[outIdx] = 0;
                output[outIdx + 1] = 0;
                output[outIdx + 2] = 0;
            }
            output[outIdx + 3] = 255;
        }
    }

    frame.data.set(output);
    ctx.putImageData(frame, 0, 0);
}

function applySobelToOutput(ctx, video, width, height) {
    ctx.drawImage(video, 0, 0, width, height);

    const frame = ctx.getImageData(0, 0, width, height);
    const pixels = frame.data;
    const output = new Uint8ClampedArray(pixels.length);

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let sumX = 0;
            let sumY = 0;

            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const idx = pixelIndex(x + kx, y + ky, width);
                    const gray = rgbToGrayscale(
                        pixels[idx],
                        pixels[idx + 1],
                        pixels[idx + 2]
                    );

                    sumX += gray * KERNELS.SOBEL_X[ky + 1][kx + 1];
                    sumY += gray * KERNELS.SOBEL_Y[ky + 1][kx + 1];
                }
            }

            const magnitude = Math.sqrt(sumX * sumX + sumY * sumY);
            const gradientOrientation = Math.atan2(sumX, sumY);
            const outIdx = pixelIndex(x, y, width);

            const intensity = gradientOrientation * magnitude;
            output[outIdx] = intensity;
            output[outIdx + 1] = intensity;
            output[outIdx + 2] = intensity;
            output[outIdx + 3] = 255;
        }
    }

    frame.data.set(output);
    ctx.putImageData(frame, 0, 0);
}

function applyDepth(ctx, video, width, height) {
    ctx.drawImage(video, 0, 0, width, height);

    const frame = ctx.getImageData(0, 0, width, height);
    const pixels = frame.data;
    const output = new Uint8ClampedArray(pixels.length);

    const alpha = parseFloat(document.getElementById("depthAlpha").value);
    const invAlpha = 1 - alpha;

    for (let y = 1; y < height - 1; y++) {
        const positionDepth = 1 - y / height;

        for (let x = 1; x < width - 1; x++) {
            const idx = pixelIndex(x, y, width);
            const gray = rgbToGrayscale(
                pixels[idx],
                pixels[idx + 1],
                pixels[idx + 2]
            );

            const intensityDepth = 1 - gray / 255;
            const depthValue = alpha * positionDepth + invAlpha * intensityDepth;
            const gammaCorrection = Math.pow(depthValue, 0.8);
            const intensity = clamp(gammaCorrection * 255, 0, 255);

            output[idx] = intensity;
            output[idx + 1] = intensity;
            output[idx + 2] = intensity;
            output[idx + 3] = 255;
        }
    }

    frame.data.set(output);
    ctx.putImageData(frame, 0, 0);
}

function applyGrayscale(ctx, video, width, height) {
    ctx.drawImage(video, 0, 0, width, height);

    const frame = ctx.getImageData(0, 0, width, height);
    const pixels = frame.data;
    const output = new Uint8ClampedArray(pixels.length);

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = pixelIndex(x, y, width);
            const gray = rgbToGrayscale(
                pixels[idx],
                pixels[idx + 1],
                pixels[idx + 2]
            );

            output[idx] = gray;
            output[idx + 1] = gray;
            output[idx + 2] = gray;
            output[idx + 3] = 255;
        }
    }

    frame.data.set(output);
    ctx.putImageData(frame, 0, 0);
}

function applyColorFilter(ctx, video, width, height) {
    ctx.drawImage(video, 0, 0, width, height);

    const frame = ctx.getImageData(0, 0, width, height);
    const pixels = frame.data;
    const output = new Uint8ClampedArray(pixels.length);

    const filterColor = hexToRgb(document.getElementById("colorFilterInput").value);
    const blendStrength = parseFloat(document.getElementById("colorIntensityRange").value);
    const inverseStrength = 1.0 - blendStrength;

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = pixelIndex(x, y, width);
            const r = pixels[idx];
            const g = pixels[idx + 1];
            const b = pixels[idx + 2];

            const gray = rgbToGrayscale(r, g, b);

            const filteredR = (gray * filterColor.r) / 255;
            const filteredG = (gray * filterColor.g) / 255;
            const filteredB = (gray * filterColor.b) / 255;

            output[idx] = r * inverseStrength + filteredR * blendStrength;
            output[idx + 1] = g * inverseStrength + filteredG * blendStrength;
            output[idx + 2] = b * inverseStrength + filteredB * blendStrength;
            output[idx + 3] = 255;
        }
    }

    frame.data.set(output);
    ctx.putImageData(frame, 0, 0);
}


function applyHistogramEq(ctx, video, width, height) {
    ctx.drawImage(video, 0, 0, width, height);

    const frame = ctx.getImageData(0, 0, width, height);
    const pixels = frame.data;

    // Compute luminance and histogram
    const luminance = new Uint8ClampedArray(width * height);
    const histogram = new Uint32Array(256).fill(0);

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = pixelIndex(x, y, width);
            const gray = rgbToGrayscale(
                pixels[idx],
                pixels[idx + 1],
                pixels[idx + 2]
            );
            const lum = Math.round(gray);

            luminance[y * width + x] = lum;
            histogram[lum]++;
        }
    }

    // Compute CDF and mapping
    const cdfMap = new Uint8ClampedArray(256);
    let cumSum = 0;

    const firstNonZero = histogram.find(count => count > 0) || 0;
    const totalPixels = (width - 2) * (height - 2);

    for (let i = 0; i < 256; i++) {
        cumSum += histogram[i];
        const mappedValue = Math.round(
            ((cumSum - firstNonZero) / (totalPixels - firstNonZero)) * 255
        );
        cdfMap[i] = Math.max(0, mappedValue);
    }

    // Apply mapping
    const output = new Uint8ClampedArray(pixels.length);

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = pixelIndex(x, y, width);
            const intensity = cdfMap[luminance[y * width + x]];

            output[idx] = intensity;
            output[idx + 1] = intensity;
            output[idx + 2] = intensity;
            output[idx + 3] = 255;
        }
    }

    frame.data.set(output);
    ctx.putImageData(frame, 0, 0);
}

function applyMotionDetection(ctx, video, width, height) {
    ctx.drawImage(video, 0, 0, width, height);

    const frame = ctx.getImageData(0, 0, width, height);
    const pixels = frame.data;

    const grayScaleFrame = new Uint8ClampedArray(pixels.length);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = pixelIndex(x, y, width);
            const r = pixels[idx];
            const g = pixels[idx + 1];
            const b = pixels[idx + 2];
            const grayscale = rgbToGrayscale(r, g, b);
            grayScaleFrame[idx] = grayScaleFrame[idx + 1] = grayScaleFrame[idx + 2] = grayscale;
            grayScaleFrame[idx + 3] = 255;
        }
    }

    if (!state.previousFrame) {
        state.previousFrame = grayScaleFrame;
        return;
    }

    const diff = new Uint8ClampedArray(pixels.length);

    const threshold = parseFloat(document.getElementById("motionDetectorThreshold").value);

    // Initialize the trail buffer if missing
    if (!state.motionTrail) {
        state.motionTrail = new Uint8ClampedArray(width * height);
    }

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = pixelIndex(x, y, width);
            let d = Math.abs(state.previousFrame[idx] - grayScaleFrame[idx]);
            if (d < threshold) d = 0;
            diff[idx] = diff[idx + 1] = diff[idx + 2] = d;
            diff[idx + 3] = 255;

            const trailIdx = y * width + x;
            state.motionTrail[trailIdx] = Math.min(255, state.motionTrail[trailIdx] + d);
        }
    }

    // Decay the trail buffer to create fading effect
    const decayAmount = 2.5;
    for (let i = 0; i < state.motionTrail.length; i++) {
        state.motionTrail[i] = Math.max(0, state.motionTrail[i] - decayAmount);
    }

    state.previousFrame = grayScaleFrame;

    const output = new Uint8ClampedArray(pixels.length);

    const filterColor = {r: 255, g:0, b: 0};
    const blendStrength = 0.7;
    const inverseStrength = 1.0 - blendStrength;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = pixelIndex(x, y, width);
            const r = pixels[idx];
            const g = pixels[idx + 1];
            const b = pixels[idx + 2];

            const intensity = state.motionTrail[y * width + x];

            const gray = rgbToGrayscale(r, g, b);

            const filteredR = (gray * filterColor.r) / 255;
            const filteredG = (gray * filterColor.g) / 255;
            const filteredB = (gray * filterColor.b) / 255;

            // Color motion trail
            if (intensity > 0) {
                output[idx] = r * inverseStrength + filteredR * blendStrength;
                output[idx + 1] = g * inverseStrength + filteredG * blendStrength;
                output[idx + 2] = b * inverseStrength + filteredB * blendStrength;
                output[idx + 3] = 255;
            } else {
                output[idx] = r;
                output[idx + 1] = g;
                output[idx + 2] = b;
                output[idx + 3] = 255;
            }
        }
    }

    frame.data.set(output);
    ctx.putImageData(frame, 0, 0);

}




// =================================
// UI Events
// =================================

function setupDetectorButtons() {
    const detectorMap = {
        raw: "",
        sobel: "sobel",
        depth: "depth",
        canny: "canny",
        grayscale: "grayscale",
        colorFilter: "colorfilter",
        histogramEq: "histogrameq",
        motiondetection: "motiondetection",
    };


    Object.entries(detectorMap).forEach(([id, detector]) => {
        document.getElementById(id)?.addEventListener("click", () => {
            state.detector = detector;
        });
    });

}

function setupParameterControls() {
    const controls = [
        { sliderId: "depthAlpha", labelId: "alpha" },
        { sliderId: "colorIntensityRange", labelId: "colorIntensity" },
        { sliderId: "highThreshold", labelId: "high" },
        { sliderId: "lowThreshold", labelId: "low" },
        { sliderId: "motionDetectorThreshold", labelId: "motionDet" }
    ];

    controls.forEach(({ sliderId, labelId }) => {
        const slider = document.getElementById(sliderId);
        const label = document.getElementById(labelId);

        if (slider && label) {
            slider.addEventListener("input", () => {
                label.innerText = slider.value;
            });
        }
    });
}

// =================================
// Video/Canvas 
// =================================

async function initializeCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: CONFIG.CAM_WIDTH,
                height: CONFIG.CAM_HEIGHT,
                frameRate: CONFIG.FRAME_RATE
            }
        });

        state.video.srcObject = stream;
        state.video.onloadedmetadata = () => {
            state.video.play();
        };
    } catch (error) {
        console.error("Camera initialization error:", error);
    }
}

function setupCanvas() {
    state.canvas = document.getElementById("canvas");
    state.canvas.width = CONFIG.CAM_WIDTH;
    state.canvas.height = CONFIG.CAM_HEIGHT;

    state.ctx = state.canvas.getContext("2d", {
        willReadFrequently: true
    });
}

function setupVideo() {
    state.video = document.querySelector("video");
    state.video.width = CONFIG.CAM_WIDTH;
    state.video.height = CONFIG.CAM_HEIGHT;
    state.video.style.backgroundColor = "transparent";

    state.video.addEventListener("playing", () => {
        renderLoop();
    }, false);
}

function renderLoop() {
    if (state.video.paused || state.video.ended) return;

    const det = state.detector;

    if (det == "grayscale") {
        applyGrayscale(state.ctx, state.video, CONFIG.CAM_WIDTH, CONFIG.CAM_HEIGHT)
    } else if (det == "colorfilter") {
        applyColorFilter(state.ctx, state.video, CONFIG.CAM_WIDTH, CONFIG.CAM_HEIGHT)
    } else if (det == "depth") {
        applyDepth(state.ctx, state.video, CONFIG.CAM_WIDTH, CONFIG.CAM_HEIGHT)
    } else if (det == "histogrameq") {
        applyHistogramEq(state.ctx, state.video, CONFIG.CAM_WIDTH, CONFIG.CAM_HEIGHT)
    } else if (det == "sobel") {
        applySobelToOutput(state.ctx, state.video, CONFIG.CAM_WIDTH, CONFIG.CAM_HEIGHT)
    } else if (det == "canny") {
        applyCanny(state.ctx, state.video, CONFIG.CAM_WIDTH, CONFIG.CAM_HEIGHT)
    } else if (det == "motiondetection") {
        applyMotionDetection(state.ctx, state.video, CONFIG.CAM_WIDTH, CONFIG.CAM_HEIGHT)
    } else {
        state.ctx.drawImage(state.video, 0, 0, CONFIG.CAM_WIDTH, CONFIG.CAM_HEIGHT);
    }

    requestAnimationFrame(renderLoop);
}

// =================================
// Init 
// =================================


function initialize() {
    setupCanvas();
    setupVideo();
    initializeCamera();
    setupDetectorButtons();
    setupParameterControls();
}

// Start application when DOM is ready
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize);
} else {
    initialize();
}

// ============================



// Ideas by AI:
// Background Subtraction
// Optical Flow (Lucas–Kanade or Horn–Schunck Lite)
// Hough Transform (for Lines or Circles)
// Corner Detectors (Harris / Shi–Tomasi)
// Depth from Motion (Structure from Motion Lite)
// Cartoon / Pencil Drawing Filter
// Kuwahara / Oil Painting Filter
// Voronoi Mosaic / Cell Shading
// Color Object Tracking [Convert to HSV → isolate hue range → centroid of region = object position.]
// Feature Matching [Detect keypoints (corners) → extract simple descriptors → match across frames.]
// Optical Flow–Based Stabilization

// Depth illusion: simulate fake parallax using grayscale brightness as depth
// Motion trails: accumulate motion over frames to create ghostly afterimages
// “Thermal camera”: map brightness → color palette.

