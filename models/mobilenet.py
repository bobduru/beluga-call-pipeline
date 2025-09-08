import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

from models.utils import get_best_device

class MobileNetMultilabel(nn.Module):
    def __init__(self, num_classes, pretrained=True, dense_layer_size=256, n_layers=None):
        super(MobileNetMultilabel, self).__init__()
        
        

        best_device = get_best_device()
        self.to(best_device)
        print(f"Loading MobileNetV3_Small model on {best_device}")
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v3_small(weights=weights)

        # Modify first conv layer for single-channel input
        mobilenet.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)

        print(f"Number of feature layers: {len(mobilenet.features)}")
        print(f"Using up to layer {n_layers}")

        # Truncate MobileNet features if requested
        if n_layers is not None:
            if not (0 <= n_layers <= 12):
                raise ValueError("n_layers must be between 0 and 12")
            self.features = nn.Sequential(*mobilenet.features[:n_layers + 1])

            # Output feature size map (empirically determined)
            feature_size_by_layer = {
                0: 16, 1: 16, 2: 24, 3: 24, 4: 40, 5: 40,
                6: 40, 7: 48, 8: 48, 9: 96, 10: 96, 11: 96, 12: 576
            }
            num_features = feature_size_by_layer[n_layers]
        else:
            self.features = mobilenet.features
            num_features = mobilenet.classifier[0].in_features  # usually 576

        print(f"Feature size after backbone: {num_features}")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Final multilabel classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_features, dense_layer_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dense_layer_size, num_classes)
        )


    def forward(self, x):
        features = self.features(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        output = self.classifier(features)
        return output  # raw logits for BCEWithLogitsLoss

    def save_model(self, file_path):
        torch.save({
            'state_dict': self.state_dict(),
        }, file_path)
        print(f"Model weights saved to {file_path}")

    def load_model(self, file_path):
        checkpoint = torch.load(file_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        self.eval()
        self.to(self.device)
        print(f"Model weights loaded from {file_path}")

