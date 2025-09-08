import torch
import torch.nn as nn
from torchvision import models


class ResnetMultilabel(nn.Module):
    def __init__(self, num_classes, pretrained=True, num_head_nodes=256):
        """
        A ResNet-based model for multilabel classification.
        
        Args:
            num_classes (int): Number of output classes for multilabel classification
            pretrained (bool): Whether to use pretrained weights (default: True)
            num_head_nodes (int): Number of nodes in the hidden layer before the output layer
        """
        super(ResnetMultilabel, self).__init__()
        
        # Set up device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Loading resnet18 model on {self.device}")
        self.resnet = models.resnet18(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        
        # Remove the original fully connected layer
        self.resnet.fc = nn.Identity()
        
        # Modify the first convolution layer to accept 1 channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Create the classification head for multilabel
        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_head_nodes),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_head_nodes, num_classes),
        )
        
        # Move model to the detected device
        self.to(self.device)

    def forward(self, x):
        features = self.resnet(x)
        return self.classifier(features)
    
    def save_model(self, file_path):
        """
        Save the model weights to the specified file path.
        
        Args:
            file_path (str): Path to save the model weights.
        """
        torch.save({
            'state_dict': self.state_dict(),
        }, file_path)
        print(f"Model weights saved to {file_path}")

    def load_model(self, file_path):
        """
        Load the model weights from the specified file path.
        
        Args:
            file_path (str): Path to load the model weights from.
        """
        checkpoint = torch.load(file_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        self.eval()  # Set the model to evaluation mode
        print(f"Model weights loaded from {file_path}")