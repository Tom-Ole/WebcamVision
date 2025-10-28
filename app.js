// https://en.wikipedia.org/wiki/Edge_detection


const CAM_WIDTH = 640;
const CAM_HEIGHT = 480;

let detector = "canny";

document.getElementById("raw").addEventListener("click", () => detector = "");
document.getElementById("sobelCPU").addEventListener("click", () => detector = "sobelcpu");
document.getElementById("depth").addEventListener("click", () => detector = "depth");
document.getElementById("sobelDepth").addEventListener("click", () => detector = "sobelDepth");
document.getElementById("canny").addEventListener("click", () => detector = "canny");
document.getElementById("depthAlpha").addEventListener("input", () => {
    document.getElementById("alpha").innerText = document.getElementById("depthAlpha").value
});

const highThresholdEle = document.getElementById("highThreshold");

highThresholdEle.addEventListener("input", () => {
    document.getElementById("high").innerText = highThresholdEle.value
});

const lowThresholdEle = document.getElementById("lowThreshold")

lowThresholdEle.addEventListener("input", () => {
    document.getElementById("low").innerText = lowThresholdEle.value
});

const canvas = document.getElementById("canvas");
canvas.width = CAM_WIDTH;
canvas.height = CAM_HEIGHT;
// canvas.style.backgroundColor = "white";

const ctx = canvas.getContext("2d", {
    willReadFrequently: true
});

let video = document.querySelector("video");
video.style.backgroundColor = "white";

video.width = CAM_WIDTH;
video.height = CAM_HEIGHT;
video.style.backgroundColor = "transparent"

navigator.mediaDevices
    .getUserMedia({
        video: {
            width: CAM_WIDTH,
            height: CAM_HEIGHT,
            frameRate: 30,
        },
    })
    .then((stream) => {
        video.srcObject = stream;
        video.onloadedmetadata = function () {
            video.play();
        };
    })
    .catch((err) => {
        console.error(err);
    });

video.addEventListener("playing", () => { drawToCanvas(); }, false);

function drawToCanvas() {
    if (video.paused || video.ended) return;
    if (detector == "sobelcpu") {
        SobelCPU();
    } else if (detector == "depth") {
        depth();
    } else if (detector == "sobelDepth") {
        SobelAndDepth();
    } else if (detector == "canny") {
        Canny();
    } else {
        ctx.drawImage(video, 0, 0, CAM_WIDTH, CAM_HEIGHT);
    }


    requestAnimationFrame(drawToCanvas);
}


// https://en.wikipedia.org/wiki/Canny_edge_detector
function Canny() {
    ctx.drawImage(video, 0, 0, CAM_WIDTH, CAM_HEIGHT);

    const frame = ctx.getImageData(0, 0, CAM_WIDTH, CAM_HEIGHT);
    const pixels = frame.data;
    const width = frame.width;
    const height = frame.height;



    const k = 3 // odd number
    const halfK = Math.floor(k / 2);
    const sigma = 2;

    // Gausian Matrix function
    function H(i, j) {

        const sigma2 = sigma * sigma
        const a = (1 / (2 * Math.PI * sigma2))
        const b = i * i + j * j
        const c = 2 * sigma2

        return a * Math.exp(-b / c);
    }

    // Create Kernel
    let kernelSum = 0;
    const gausianKernel = [];
    for (let j = 0; j < k; j++) {
        let row = [];
        for (let i = 0; i < k; i++) {
            const ii = i - halfK; // Modern approch with 0-Index
            const jj = j - halfK; // Modern approch with 0-Index
            const value = H(ii, jj);

            row.push(value);
            kernelSum += value;
        }
        gausianKernel.push(row);
    }

    // Normalize the kernel
    for (let j = 0; j < k; j++) {
        for (let i = 0; i < k; i++) {
            gausianKernel[j][i] /= kernelSum;
        }
    }

    // Create Gausian Grayscale image
    let smoothedGrayscale = new Uint8ClampedArray(width * height);
    function idx(x, y) { return (y * width + x) * 4; }


    for (let y = halfK; y < height - halfK; y++) {
        for (let x = halfK; x < width - halfK; x++) {

            let sum = 0;

            // Grayscale
            for (let ky = -halfK; ky <= halfK; ky++) {
                for (let kx = -halfK; kx <= halfK; kx++) {

                    const i = idx(x + kx, y + ky);

                    const r = pixels[i];
                    const g = pixels[i + 1];
                    const b = pixels[i + 2];

                    const gray = 0.299 * r + 0.587 * g + 0.114 * b;

                    sum += gray * gausianKernel[ky + halfK][kx + halfK];

                }
            }



            smoothedGrayscale[y * width + x] = Math.round(sum);
        }
    }

    // Sobel
    const gx = [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ];
    const gy = [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ];

    const gradientMagnitude = new Float32Array(width * height);
    const gradientAngle = new Float32Array(width * height);
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let sumX = 0;
            let sumY = 0;

            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const i = idx(x + kx, y + ky);

                    const smoothedIdx = (y + ky) * width + (x + kx);
                    const gray = smoothedGrayscale[smoothedIdx];;

                    sumX += gray * gx[ky + 1][kx + 1];
                    sumY += gray * gy[ky + 1][kx + 1];

                }
            }

            const mag = Math.sqrt(sumX * sumX + sumY * sumY);
            const angle = Math.atan2(sumY, sumX);

            gradientMagnitude[y * width + x] = mag;
            gradientAngle[y * width + x] = angle;

        }
    }

    // Gradient magnitude thresholding
    const nonMaxSupp = new Float32Array(width * height).fill(0);

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = y * width + x;

            let angle = gradientAngle[idx];
            let magnitude = gradientMagnitude[idx];

            let degree = angle * (180 / Math.PI); // From radiens to degree
            if (degree < 0) degree += 180; // Map to [0, 180]

            let neighbor1X, neighbor1Y, neighbor2X, neighbor2Y;

            if ((degree >= 0 && degree < 22.5) || (degree >= 157.5 && degree <= 180)) {
                // 0 degrees (Horizontal: check W and E)
                neighbor1X = x + 1; neighbor1Y = y;
                neighbor2X = x - 1; neighbor2Y = y;
            } else if (degree >= 22.5 && degree < 67.5) {
                // 45 degrees (Diagonal: check NE and SW)
                neighbor1X = x + 1; neighbor1Y = y - 1;
                neighbor2X = x - 1; neighbor2Y = y + 1;
            } else if (degree >= 67.5 && degree < 112.5) {
                // 90 degrees (Vertical: check N and S)
                neighbor1X = x; neighbor1Y = y - 1;
                neighbor2X = x; neighbor2Y = y + 1;
            } else if (degree >= 112.5 && degree < 157.5) {
                // 135 degrees (Diagonal: check NW and SE)
                neighbor1X = x - 1; neighbor1Y = y - 1;
                neighbor2X = x + 1; neighbor2Y = y + 1;
            }

            const mag1 = gradientMagnitude[neighbor1Y * width + neighbor1X];
            const mag2 = gradientMagnitude[neighbor2Y * width + neighbor2X];

            // apply cutoff
            if (magnitude >= mag1 && magnitude >= mag2) {
                nonMaxSupp[idx] = magnitude; // Local max; keep it
            } else {
                nonMaxSupp[idx] = 0; // cutoff
            }

        }
    }

    // Double Thresholding
    let maxMag = 0;
    for (let i = 0; i < nonMaxSupp.length; i++) {
        if (nonMaxSupp[i] > maxMag) {
            maxMag = nonMaxSupp[i];
        }
    }
    const highTreshold = maxMag * highThresholdEle.value;
    const lowTreshold = maxMag * lowThresholdEle.value;

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = y * width + x;
            const mag = nonMaxSupp[idx];

            if (mag <= lowTreshold) {
                nonMaxSupp[idx] = 0;
            } else if (mag <= highTreshold) {
                nonMaxSupp[idx] = 0.5;
            } else {
                nonMaxSupp[idx] = 1;
            }
        }
    }

    // https://en.wikipedia.org/wiki/Connected-component_labeling
    // hysteresis // Blob analysis
    function checkNeighborhood(x, y) {
        // Iterate over the 3x3 window around the current pixel
        for (let j = -1; j <= 1; j++) {
            for (let i = -1; i <= 1; i++) {
                if (i === 0 && j === 0) continue;

                const nx = x + i;
                const ny = y + j;

                // Check bounds
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

                const neighborIdx = ny * width + nx;

                // If any neighbor is already a STRONG edge, return true
                if (nonMaxSupp[neighborIdx] === 1) {
                    return true;
                }
            }
        }
        return false;
    }

    let changed = true;
    while (changed) {
        changed = false;

        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const mapIdx = y * width + x;

                // Only process WEAK edges (0.5)
                if (nonMaxSupp[mapIdx] === 0.5) {
                    // Check if this weak edge is connected to Strong
                    if (checkNeighborhood(x, y)) {
                        // Promote the Weak Edge
                        nonMaxSupp[mapIdx] = 1;
                        changed = true; // loop again
                    }
                }
            }
        }
    }

    // Remove leftovers
    for (let i = 0; i < nonMaxSupp.length; i++) {
        if (nonMaxSupp[i] === 0.5) {
            nonMaxSupp[i] = 0;
        }
    }



    // Generate final output
    const output = new Uint8ClampedArray(pixels.length);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const outIdx = idx(x, y);
            const mapIdx = y * width + x;
            const edge = nonMaxSupp[mapIdx] || 0;

            let intensity = 0;

            // Note: Weak Edges (0.5) and Suppressed Edges (0) remain strong edges (0)
            if (edge === 1) {
                output[outIdx] = 0
                output[outIdx + 1] = 255
                output[outIdx + 2] = 0
                output[outIdx + 3] = 255;
            } else if (edge === 0.5) {
                output[outIdx] = 255
                output[outIdx + 1] = 0
                output[outIdx + 2] = 0
                output[outIdx + 3] = 255;
            } else {
                output[outIdx] = 0
                output[outIdx + 1] = 0
                output[outIdx + 2] = 0
                output[outIdx + 3] = 255;
            }


        }
    }

    frame.data.set(output);
    ctx.putImageData(frame, 0, 0);

}


// https://en.wikipedia.org/wiki/Edge_detection#Other_first-order_methods
function SobelCPU() {
    ctx.drawImage(video, 0, 0, CAM_WIDTH, CAM_HEIGHT);

    const frame = ctx.getImageData(0, 0, CAM_WIDTH, CAM_HEIGHT);
    const pixels = frame.data;
    const width = frame.width;
    const height = frame.height;

    const output = new Uint8ClampedArray(pixels.length);

    // Sobel kernels
    const gx = [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ];
    const gy = [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ];

    function idx(x, y) { return (y * width + x) * 4; }

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            let sumX = 0;
            let sumY = 0;

            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const i = idx(x + kx, y + ky);

                    const r = pixels[i];
                    const g = pixels[i + 1];
                    const b = pixels[i + 2];
                    const gray = 0.299 * r + 0.587 * g + 0.114 * b;

                    sumX += gray * gx[ky + 1][kx + 1];
                    sumY += gray * gy[ky + 1][kx + 1];

                }
            }

            const mag = Math.sqrt(sumX * sumX + sumY * sumY);
            const outIdx = idx(x, y);

            const gradOri = Math.atan2(sumX, sumY);


            output[outIdx] = output[outIdx + 1] = output[outIdx + 2] = gradOri * mag;
            output[outIdx + 3] = 255; // alpha;

        }
    }

    frame.data.set(output);
    ctx.putImageData(frame, 0, 0);
}


function depth() {
    ctx.drawImage(video, 0, 0, CAM_WIDTH, CAM_HEIGHT);

    const frame = ctx.getImageData(0, 0, CAM_WIDTH, CAM_HEIGHT);
    const pixels = frame.data;
    const width = frame.width;
    const height = frame.height;

    const output = new Uint8ClampedArray(pixels.length);

    function idx(x, y) { return (y * width + x) * 4; }

    const alpha = parseFloat(document.getElementById("depthAlpha").value);
    const invAlpha = 1 - alpha;

    for (let y = 1; y < height - 1; y++) {

        const posDepth = 1 - y / height;

        for (let x = 1; x < width - 1; x++) {
            const i = idx(x, y);

            const r = pixels[i];
            const g = pixels[i + 1];
            const b = pixels[i + 2];

            const grayScale = 0.299 * r + 0.587 * g + 0.114 * b;
            const intensityDepth = 1 - grayScale / 255; // Enhanced depth calculation with better contrast
            const depthValue = alpha * posDepth + invAlpha * intensityDepth;
            const gammeCorrection = Math.pow(depthValue, 0.8);

            output[i] = output[i + 1] = output[i + 2] = Math.min(255, Math.max(0, gammeCorrection * 255));
            output[i + 3] = 255; // alpha;
        }
    }

    frame.data.set(output);
    ctx.putImageData(frame, 0, 0);
}

function SobelAndDepth() {
    ctx.drawImage(video, 0, 0, CAM_WIDTH, CAM_HEIGHT);

    const frame = ctx.getImageData(0, 0, CAM_WIDTH, CAM_HEIGHT);
    const pixels = frame.data;
    const width = frame.width;
    const height = frame.height;

    const output = new Uint8ClampedArray(pixels.length);

    // Sobel kernels
    const gx = [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ];
    const gy = [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ];

    function idx(x, y) { return (y * width + x) * 4; }

    const alpha = parseFloat(document.getElementById("depthAlpha").value);
    const invAlpha = 1 - alpha;

    for (let y = 1; y < height - 1; y++) {

        const posDepth = 1 - y / height;

        for (let x = 1; x < width - 1; x++) {
            let sumX = 0;
            let sumY = 0;

            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const i = idx(x + kx, y + ky);

                    const r = pixels[i];
                    const g = pixels[i + 1];
                    const b = pixels[i + 2];
                    const gray = 0.299 * r + 0.587 * g + 0.114 * b;

                    sumX += gray * gx[ky + 1][kx + 1];
                    sumY += gray * gy[ky + 1][kx + 1];

                }
            }

            const mag = Math.sqrt(sumX * sumX + sumY * sumY);
            const outIdx = idx(x, y);

            const gradOri = Math.atan2(sumX, sumY);


            const i = idx(x, y);

            const r = pixels[i];
            const g = pixels[i + 1];
            const b = pixels[i + 2];

            const grayScale = 0.299 * r + 0.587 * g + 0.114 * b;
            const intensityDepth = 1 - grayScale / 255; // Enhanced depth calculation with better contrast
            const depthValue = alpha * posDepth + invAlpha * intensityDepth;
            const gammeCorrection = Math.pow(depthValue, 0.8);


            output[outIdx] = output[outIdx + 1] = output[outIdx + 2] = gradOri * mag * gammeCorrection;
            output[outIdx + 3] = 255; // alpha;

        }
    }

    frame.data.set(output);
    ctx.putImageData(frame, 0, 0);
}