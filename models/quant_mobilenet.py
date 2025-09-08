import torch
import torch.nn as nn
from torchvision.models.quantization.mobilenetv3 import _mobilenet_v3_model
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf
from torchvision.models import MobileNet_V3_Small_Weights


def load_mobilenet_v3_quant(num_classes=4, n_layers=12, pretrained=True, qat=True):
    """
    Quantized MobileNetV3 Model Loader

    This module provides functionality to load and configure a quantized version of MobileNetV3 Small. 
    Which can either be used for post training quantization or quantization aware training. (QAT is recommended for optimal performance)

    Unlike MobileNetMultilabel, this uses a functional approach rather than a class-based one.
    This was necessary to properly handle quantization and pretrained weights, as the quantized
    architecture requires loading the weights from the original model and then loading them in 
    the model capable of quantization. While functional, this implementation could be refactored 
    in the future to use a more object-oriented design.

    Args:
        num_classes (int): Number of classes to classify
        n_layers (int): Number of layers to use
        pretrained (bool): Whether to use pretrained weights
        qat (bool): Whether to use quantization-aware training

    Returns:
        torch.nn.Module: Quantized MobileNetV3 Small model
    """
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", num_classes=1000)
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model_v3_quant = _mobilenet_v3_model(inverted_residual_setting, last_channel, weights=weights, progress=True,quantize=False, num_classes=1000)



    # Modify first conv layer for single-channel input
    old_first_layer = model_v3_quant.features[0]
    old_conv = old_first_layer[0]  # This is the Conv2d layer
    out_channels = old_conv.out_channels

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    # # Replace the old Conv2d with the new one
    model_v3_quant.features[0][0] = new_conv

    model_v3_quant.features = nn.Sequential(*model_v3_quant.features[:n_layers + 1])

    # Output feature size map (empirically determined)
    feature_size_by_layer = {
        0: 16, 1: 16, 2: 24, 3: 24, 4: 40, 5: 40,
        6: 40, 7: 48, 8: 48, 9: 96, 10: 96, 11: 96, 12: 576
    }
    num_features = feature_size_by_layer[n_layers]
    model_v3_quant.classifier = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )

    if qat :
        print("quantization aware trainign")
        model_v3_quant.fuse_model(is_qat=True)
        model_v3_quant.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
        torch.ao.quantization.prepare_qat(model_v3_quant, inplace=True)

    return model_v3_quant

def load_mobilenet_v3_quant_from_file(path, n_layers=12, num_classes=4):
    """
    Loads a quantized MobileNetV3 model from a file.

    Args:
        path (str): Path to the model file.
        n_layers (int): Number of layers to use.
        num_classes (int): Number of classes to classify.
    """
    checkpoint = torch.load(path, map_location="cpu")
    n_layers = checkpoint.get("n_layers", n_layers)
    num_classes = checkpoint.get("num_classes", num_classes)
    model = load_mobilenet_v3_quant(num_classes=num_classes, n_layers=n_layers, qat=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


import torch


def calibrate(model, calibration_loader, neval_batches):
    """
    Calibrates a model for post training quantization using a calibration loader.

    Args:
        model (torch.nn.Module): The model to calibrate.
        calibration_loader (DataLoader): DataLoader used for calibration.
        neval_batches (int): Number of batches to use for calibration.
    """
    model.eval()
    cnt = 0
    with torch.no_grad():
        for features, labels, metadata in calibration_loader:
            _ = model(features)
            cnt += 1
            if cnt >= neval_batches:
                break


def quantize_model_post_training(model, calibration_loader, neval_batches=10, qengine='x86'):
    """
    Applies post-training static quantization to a model.

    Args:
        model (torch.nn.Module): The model to quantize (must support fuse_model()).
        calibration_loader (DataLoader): DataLoader used for calibration.
        neval_batches (int): Number of batches to use for calibration.
        qengine (str): Quantization backend ('x86', 'fbgemm', 'qnnpack', etc.).
        
    Returns:
        torch.nn.Module: Quantized model.
    """
    # Set backend engine
    torch.backends.quantized.engine = qengine

    # Move to CPU and eval mode
    model.to(torch.device('cpu'))
    model.eval()

    # Fuse layers (e.g., Conv+BN+ReLU)
    model.fuse_model()

    # Configure quantization
    model.qconfig = torch.ao.quantization.get_default_qconfig(qengine)

    # Prepare model (insert observers)
    torch.ao.quantization.prepare(model, inplace=True)

    calibrate(model, calibration_loader, neval_batches)

    # Convert to quantized model
    quantized_model = torch.ao.quantization.convert(model, inplace=True)
    print("Quantization conversion complete.")
    return quantized_model
