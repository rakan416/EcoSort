# Ultralytics YOLOv5 🚀, AGPL-3.0 license

import os
import platform
import sys
from pathlib import Path

import pandas as pd
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

MACOS = platform.system() == "Darwin"  # macOS environment


class iOSModel(torch.nn.Module):
    def __init__(self, model, im):
        """Initializes an iOS compatible model with normalization based on image dimensions."""
        super().__init__()
        b, c, h, w = im.shape  # batch, channel, height, width
        self.model = model
        self.nc = model.nc  # number of classes
        if w == h:
            self.normalize = 1.0 / w
        else:
            self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])  # broadcast (slower, smaller)


    def forward(self, x):
        """Runs forward pass on the input tensor, returning class confidences and normalized coordinates."""
        xywh, conf, cls = self.model(x)[0].squeeze().split((4, 1, self.nc), 1)
        return cls * conf, xywh * self.normalize  # confidence (3780, 80), coordinates (3780, 4)

 
def export_formats():
    """Returns a DataFrame of supported YOLOv5 model export formats and their properties."""
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlmodel", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False, False],
        ["TensorFlow.js", "tfjs", "_web_model", False, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])

