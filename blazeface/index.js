/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tf from "@tensorflow/tfjs-core";


export const videoWidth = 288;
export const videoHeight = 352;
// export interface Point {
//   x: number;
//   y: number;
// }
// export interface CanvasNode {
//   width: number;
//   height: number;
//   getContext: Function;
//   createImage: Function;
// }

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
export async function detectInRealTime(image, net, mirror) {
  const video = tf.tidy(() => {
    const temp = tf.tensor(new Uint8Array(image.data), [
      image.height,
      image.width,
      4,
    ]);
    return tf.slice(temp, [0, 0, 0], [-1, -1, 3]);
  });

  // since images are being fed from a webcam
  const flipHorizontal = mirror;
  const returnTensors = false;
  const annotateBoxes = true;
  const predictions = await net.estimateFaces(
    video,
    returnTensors,
    flipHorizontal,
    annotateBoxes
  );
  video.dispose();
  return [predictions];
}

export function drawPredictions(page) {
  if (page.predictions == null || page.ctx == null) return;
  const ctx = page.ctx;
  const predictions = page.predictions;

  // For each pose (i.e. person) detected in an image, loop through the poses
  // and draw the resulting skeleton and keypoints if over certain confidence
  // scores
  predictions.forEach((prediction) => {
    const start = prediction[0].topLeft;
    const end = prediction[0].bottomRight;
    const size = [end[0] - start[0], end[1] - start[1]];
    ctx.rect(
      start[0], start[1], size[0], size[1]);
    ctx.strokeStyle = 'red';
    ctx.stroke();
    const landmarks = prediction[0].landmarks;
    ctx.fillStyle = "red";
    for (let j = 0; j < landmarks.length; j++) {
      const x = landmarks[j][0];
      const y = landmarks[j][1];
      ctx.fillRect(x, y, 5, 5);
    }
  });
  ctx.draw();
  return predictions;
}
