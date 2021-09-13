// index.js
// 获取应用实例
import * as blazeface from '@tensorflow-models/blazeface';
import {detectInRealTime, drawPredictions} from '../../blazeface/index';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';

tfjsWasm.setWasmPaths(
  `/miniprogram_npm/@tensorflow/tfjs-backend-wasm/index.js`
  //`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`
  );

const CANVAS_ID = 'image';

const app = getApp()

Page({
  data: {result: ''},
  blazefaceModel: undefined,
  canvas: undefined,
  predictions: undefined,
  ctx: undefined,
  blazeface() {
    if (this.blazefaceModel == null) {
      this.setData({result: 'loading posenet model...'});
      blazeface
          .load({})
          .then((model) => {
            this.blazefaceModel = model;
            this.setData({result: 'model loaded.'});
          });
    }
  },
  executeBlazeface(frame) {
    if (this.blazefaceModel) {
      const start = Date.now();
      detectInRealTime(frame, this.blazefaceModel, false)
          .then((predictions) => {
            this.predictions = predictions;
            drawPredictions(this);
            // const result = `${Date.now() - start}ms`;
            // this.setData({result});
          })
          .catch((err) => {
            console.log(err, err.stack);
          });
    }
  },
  async onReady() {
    console.log('create canvas context for #image...');
    setTimeout(() => {
      this.ctx = wx.createCanvasContext(CANVAS_ID);
      console.log('ctx', this.ctx);
    }, 500);

    await tf.setBackend('wasm');
    
    this.blazeface();

    // Start the camera API to feed the captured images to the models.
    // @ts-ignore the ts definition for this method is worng.
    const context = wx.createCameraContext(this);
    let count = 0;
    const listener = (context).onCameraFrame((frame) => {
      count++;
      if (count === 3) {//3 frame一传
        this.executeBlazeface(frame);
        count = 0;
      }
    });
    listener.start();
  },
  onUnload() {
    if (this.posenetModel) {
      this.posenetModel.dispose();
    }
  }
})


