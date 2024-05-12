import 'dart:math';

import 'package:flutter/cupertino.dart';
import 'package:image/image.dart' as image_lib;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'package:makeat_app/widgets/logger.dart';
import 'package:makeat_app/miss_chef_ai/entity/recognition.dart';
import 'package:tflite_flutter_helper_plus/tflite_flutter_helper_plus.dart';
import 'package:tflite_flutter_plus/src/bindings/types.dart' as tfNew;
import 'package:tflite_flutter/src/bindings/tensorflow_lite_bindings_generated.dart' as tfLiteFNewObj;

class Classifier {
  Classifier({
    Interpreter? interpreter,
  }) {
    loadModel(interpreter);
  }

  late Interpreter? _interpreter;

  Interpreter? get interpreter => _interpreter;

  static const String modelFileName = 'assets/miss_chef_ai_model/best_60-fp16.tflite';

  /// image size into interpreter
  static const int inputSize = 640;

  ImageProcessor? imageProcessor;
  late List<List<int>> _outputShapes;
  late List<tfNew.TfLiteType> _outputTypes;

  static const int clsNum = 40;
  static const double objConfTh = 0.80;
  static const double clsConfTh = 0.80;

  /// load interpreter
  Future<void> loadModel(Interpreter? interpreter) async {
    try {
      _interpreter = interpreter ??
          await Interpreter.fromAsset(
            modelFileName,
            options: InterpreterOptions()..threads = 1,
          );
      final outputTensors = _interpreter!.getOutputTensors();
      print(outputTensors);
      _outputShapes = [];
      _outputTypes = [];
      for (final tensor in outputTensors) {
        _outputShapes.add(tensor.shape);
        _outputTypes.add(tfNew.TfLiteType.float32);
      }
    } on Exception catch (e) {
      print("Error ho gaya Bhai!!!!!!!!!!!!!!!!!!");
      logger.warning(e.toString());
    }
  }

  /// image pre process
  TensorImage getProcessedImage(TensorImage inputImage) {
    final padSize = max(inputImage.height, inputImage.width);

    imageProcessor ??= ImageProcessorBuilder()
        .add(
          ResizeWithCropOrPadOp(
            padSize,
            padSize,
          ),
        )
        .add(
          ResizeOp(
            inputSize,
            inputSize,
            ResizeMethod.bilinear,
          ),
        )
        .build();
    return imageProcessor!.process(inputImage);
  }

  List<Recognition> predict(image_lib.Image image) {
    if (_interpreter == null) {
      return [];
    }

    var inputImage = TensorImage.fromImage(image);
    inputImage = getProcessedImage(inputImage);

    ///  normalize from zero to one
    List<double> normalizedInputImage = [];
    for (var pixel in inputImage.tensorBuffer.getDoubleList()) {
      normalizedInputImage.add(pixel / 255.0);
    }
    var normalizedTensorBuffer = TensorBuffer.createDynamic(tfNew.TfLiteType.float32);
    normalizedTensorBuffer.loadList(normalizedInputImage, shape: [inputSize, inputSize, 3]);

    final inputs = [normalizedTensorBuffer.buffer];

    /// tensor for results of inference
    final outputLocations = TensorBufferFloat(_outputShapes[0]);
    final outputs = {
      0: outputLocations.buffer,
    };
    _interpreter!.runForMultipleInputs(inputs, outputs);

    /// make recognition
    final recognitions = <Recognition>[];
    List<double> results = outputLocations.getDoubleList();
    // results.clear();
    // results.add(outputLocations.getDoubleValue(0)) ;
    for (var i = 0; i < results.length; i += (5 + clsNum)) {
      // check obj conf
      if (results[i + 4] < objConfTh) continue;

      /// check cls conf
      // double maxClsConf = results[i + 5];
      double maxClsConf = results.sublist(i + 5, i + 5 + clsNum - 1).reduce(max);
      if (maxClsConf < clsConfTh) continue;

      /// add detects
      // int cls = 0;
      int cls = results.sublist(i + 5, i + 5 + clsNum - 1).indexOf(maxClsConf) % clsNum;
      Rect outputRect = Rect.fromCenter(
        center: Offset(
          results[i] * inputSize,
          results[i + 1] * inputSize,
        ),
        width: results[i + 2] * inputSize,
        height: results[i + 3] * inputSize,
      );
      Rect transformRect = imageProcessor!.inverseTransformRect(outputRect, image.height, image.width);

      recognitions.add(Recognition(i, cls, maxClsConf, transformRect));
      // logger.info("Observed Object - $i $cls $maxClsConf $transformRect}");
    }
    return recognitions;
  }
}

extension TfLiteTypeFromTensorType on TensorType {
  int getValues() {
    switch (this) {
      case TensorType.float32:
        return tfLiteFNewObj.TfLiteType.kTfLiteFloat32;
      case TensorType.int32:
        return tfLiteFNewObj.TfLiteType.kTfLiteInt32;
      case TensorType.uint8:
        return tfLiteFNewObj.TfLiteType.kTfLiteUInt8;
      case TensorType.int64:
        return tfLiteFNewObj.TfLiteType.kTfLiteInt64;
      case TensorType.string:
        return tfLiteFNewObj.TfLiteType.kTfLiteString;
      case TensorType.boolean:
        return tfLiteFNewObj.TfLiteType.kTfLiteBool;
      case TensorType.int16:
        return tfLiteFNewObj.TfLiteType.kTfLiteInt16;
      case TensorType.complex64:
        return tfLiteFNewObj.TfLiteType.kTfLiteComplex64;
      case TensorType.int8:
        return tfLiteFNewObj.TfLiteType.kTfLiteInt8;
      case TensorType.float16:
        return tfLiteFNewObj.TfLiteType.kTfLiteFloat16;
      case TensorType.float64:
        return tfLiteFNewObj.TfLiteType.kTfLiteFloat64;
      case TensorType.complex128:
        return tfLiteFNewObj.TfLiteType.kTfLiteComplex128;
      case TensorType.uint64:
        return tfLiteFNewObj.TfLiteType.kTfLiteUInt64;
      case TensorType.resource:
        return tfLiteFNewObj.TfLiteType.kTfLiteResource;
      case TensorType.variant:
        return tfLiteFNewObj.TfLiteType.kTfLiteVariant;
      case TensorType.uint32:
        return tfLiteFNewObj.TfLiteType.kTfLiteUInt32;
      case TensorType.uint16:
        return tfLiteFNewObj.TfLiteType.kTfLiteUInt16;
      case TensorType.int4:
        return tfLiteFNewObj.TfLiteType.kTfLiteInt4;
      default:
        return 0;
    }
  }
}