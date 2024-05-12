import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:hooks_riverpod/hooks_riverpod.dart';
import 'package:image/image.dart' as image_lib;
import 'package:makeat_app/miss_chef_ai/entity/recognition.dart';
import 'package:makeat_app/miss_chef_ai/model/classifier.dart';
import 'package:makeat_app/widgets/image_utils.dart';
import 'package:makeat_app/widgets/logger.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter_tflite/flutter_tflite.dart';

final recognitionsProvider = StateProvider<List<Recognition>>((ref) => []);
int cameraCount = 0;
final mlCameraProvider = FutureProvider.autoDispose.family<MLCamera, Size>((ref, size) async {
  final cameras = await availableCameras();
  final cameraController = CameraController(
    cameras[0],
    ResolutionPreset.high,
    enableAudio: false,
  );
  await cameraController.initialize().then((value) {
    cameraController.startImageStream((image) {
      cameraCount ++ ;
      if(cameraCount % 10 ==0){
        cameraCount = 0;
        detectObject(image);
      }
    },);
  });
  final mlCamera = MLCamera(
    ref.read,
    cameraController,
    size,
  );
  return mlCamera;
});
void detectObject(CameraImage cameraImage) async {
  var detectedValue = await Tflite.runModelOnFrame(
      bytesList: cameraImage.planes.map((e) => e.bytes).toList(),
      imageHeight: cameraImage.height,
      imageWidth: cameraImage.width,
      imageMean: 127.5,
      imageStd: 127.5,
      rotation: 90,
      numResults: 1,
      threshold: 0.4,
      asynch: false);
  if (detectedValue != null) {
    logger.info(detectedValue);
  }
}

class MLCamera {
  MLCamera(
    this._read,
    this.cameraController,
    this.cameraViewSize,
  ) {
    Future(() async {
      classifier = Classifier();
      await initTFliteModel();
      // await cameraController.startImageStream(detectObject);
    });
  }

  final Reader _read;
  final CameraController cameraController;

  final Size cameraViewSize;

  late double ratio = Platform.isAndroid ? cameraViewSize.width / cameraController.value.previewSize!.height : cameraViewSize.width / cameraController.value.previewSize!.width;

  late Size actualPreviewSize = Size(
    cameraViewSize.width,
    cameraViewSize.width * ratio,
  );

  late Classifier classifier;

  bool isPredicting = false;

  Future<void> onCameraAvailable(CameraImage cameraImage) async {
    // logger.log(Level.WARNING, isPredicting);
    if (classifier.interpreter == null) {
      return;
    }

    if (isPredicting) {
      return;
    }

    isPredicting = true;
    final isolateCamImgData = IsolateData(
      cameraImage: cameraImage,
      interpreterAddress: classifier.interpreter!.address,
    );
    List<Recognition> data = await compute(inference, isolateCamImgData);
    _read(recognitionsProvider.notifier).state = data;
    isPredicting = false;
  }

  /// inference function

  static Future<List<Recognition>> inference(IsolateData isolateCamImgData) async {
    var image = ImageUtils.convertYUV420ToImage(
      isolateCamImgData.cameraImage,
    );
    if (Platform.isAndroid) {
      image = image_lib.copyRotate(image, 90);
    }

    final classifier = Classifier(
      interpreter: Interpreter.fromAddress(
        isolateCamImgData.interpreterAddress,
      ),
    );

    return classifier.predict(image);
  }

  Future initTFliteModel() async{
    const String modelFileName = 'assets/miss_chef_ai_model/best_60-fp16.tflite';
    const String labelFileName = 'assets/miss_chef_ai_model/label.txt';

    // const String modelFileName = 'assets/miss_chef_ai_model/mobilenet.tflite';
    // const String labelFileName = 'assets/miss_chef_ai_model/mobilenet_label.txt';
    await Tflite.loadModel(model: modelFileName, labels: labelFileName, isAsset: true, numThreads: 1, useGpuDelegate: false);
  }
}

class IsolateData {
  IsolateData({
    required this.cameraImage,
    required this.interpreterAddress,
  });

  final CameraImage cameraImage;
  final int interpreterAddress;
}
