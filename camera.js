import * as posenet_module from '@tensorflow-models/posenet';
import * as tf from '@tensorflow/tfjs';
import * as paper from 'paper';
import Stats from 'stats.js';
import "babel-polyfill";

import {isMobile, toggleLoadingUI, setStatusText} from './utils/demoUtils';
import {SVGUtils} from './utils/svgUtils'
import {PoseIllustration} from './illustrationGen/illustration';
import {Skeleton} from './illustrationGen/skeleton';
import {FileUtils} from './utils/fileUtils';

import * as JapanTshirt from './resources/illustration/clothes/short-sleeve-final-japan.svg';

// Camera stream video element
let video;
let videoWidth = 1200;
let videoHeight = 800;

// Canvas
let illustration = null;
let canvasScope;
let canvasWidth = 1200;
let canvasHeight = 800;

// ML models
let posenet;
let minPartConfidence = 0.1;
let nmsRadius = 30.0;

// Misc
let mobile = null;
const stats = new Stats();
const avatarSvgs = {
  'Japan': JapanTshirt.default,
};

/**
 * Loads the camera to be used
 *
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: videoWidth,
      height: videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

const defaultPoseNetArchitecture = 'MobileNetV1';
const defaultQuantBytes = 2;
const defaultMultiplier = 1.0;
const defaultStride = 16;
const defaultInputResolution = 200;

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  document.getElementById('main').appendChild(stats.dom);
}

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops to give a real-time effect.
 */
function detectPoseInRealTime(video) {
  const canvas = document.getElementById('output');
  const keypointCanvas = document.getElementById('keypoints');
  const videoCtx = canvas.getContext('2d');
  const keypointCtx = keypointCanvas.getContext('2d');

  canvas.width = videoWidth;
  canvas.height = videoHeight;
  keypointCanvas.width = videoWidth;
  keypointCanvas.height = videoHeight;

  async function poseDetectionFrame() {
    // Begin monitoring code for frames per second
    stats.begin();

    let poses = [];
   
    videoCtx.clearRect(0, 0, videoWidth, videoHeight);
    // Draw video
    videoCtx.save();
    videoCtx.scale(-1, 1);
    videoCtx.translate(-videoWidth, 0);
    videoCtx.drawImage(video, 0, 0, videoWidth, videoHeight);
    videoCtx.restore();

    // Creates a tensor from an image
    const input = tf.browser.fromPixels(canvas);
    let all_poses = await posenet.estimatePoses(video, {
      flipHorizontal: true,
      decodingMethod: 'multi-person',
      maxDetections: 1,
      scoreThreshold: minPartConfidence,
      nmsRadius: nmsRadius
    });

    poses = poses.concat(all_poses);
    keypointCtx.clearRect(0, 0, videoWidth, videoHeight);
    canvasScope.project.clear();

    if (poses.length >= 1 && illustration) {
      Skeleton.flipPose(poses[0]);
      illustration.updateSkeleton(poses[0], null);
      illustration.draw(canvasScope, videoWidth, videoHeight);
    }

    // End monitoring code for fps box
    stats.end();

    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}

function setupCanvas() {
  mobile = isMobile();
  if (mobile) {
    canvasWidth = Math.min(window.innerWidth, window.innerHeight);
    canvasHeight = canvasWidth;
    videoWidth *= 0.7;
    videoHeight *= 0.7;
  }  

  canvasScope = paper.default;
  let canvas = document.querySelector('.illustration-canvas');
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;
  canvasScope.setup(canvas);
}

/**
 * Loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
  setupCanvas();

  toggleLoadingUI(true);
  setStatusText('Loading PoseNet model...');
  posenet = await posenet_module.load({
    architecture: defaultPoseNetArchitecture,
    outputStride: defaultStride,
    inputResolution: defaultInputResolution,
    multiplier: defaultMultiplier,
    quantBytes: defaultQuantBytes
  });


  setStatusText('Loading Avatar file...');
  await parseSVG(Object.values(avatarSvgs)[0]);

  setStatusText('Setting up camera...');
  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this device type is not supported yet, ' +
      'or this browser does not support video capture: ' + e.toString();
    info.style.display = 'block';
    throw e;
  }

  setupFPS();
  
  toggleLoadingUI(false);
  detectPoseInRealTime(video, posenet);
}

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
FileUtils.setDragDropHandler((result) => {parseSVG(result)});

async function parseSVG(target) {
  let svgScope = await SVGUtils.importSVG(target /* SVG string or file path */);
  let skeleton = new Skeleton(svgScope);
  illustration = new PoseIllustration(canvasScope);
  illustration.bindSkeleton(skeleton, svgScope);
}
    
bindPage();
