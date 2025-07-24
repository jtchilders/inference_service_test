#!/usr/bin/env python3
"""
PyTorch model definitions for CIFAR-100 classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class CustomCNN(nn.Module):
   """Custom CNN architecture for CIFAR-100."""
   
   def __init__(self, num_classes: int = 100, hidden_size: int = 1024, 
                num_layers: int = 3, dropout_rate: float = 0.2):
      super(CustomCNN, self).__init__()
      
      self.num_classes = num_classes
      self.hidden_size = hidden_size
      self.num_layers = num_layers
      self.dropout_rate = dropout_rate
      
      # Initial convolution layers
      self.features = nn.Sequential(
         nn.Conv2d(3, 64, kernel_size=3, padding=1),
         nn.BatchNorm2d(64),
         nn.ReLU(inplace=True),
         nn.Conv2d(64, 64, kernel_size=3, padding=1),
         nn.BatchNorm2d(64),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(kernel_size=2, stride=2),
         nn.Dropout2d(dropout_rate),
         
         nn.Conv2d(64, 128, kernel_size=3, padding=1),
         nn.BatchNorm2d(128),
         nn.ReLU(inplace=True),
         nn.Conv2d(128, 128, kernel_size=3, padding=1),
         nn.BatchNorm2d(128),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(kernel_size=2, stride=2),
         nn.Dropout2d(dropout_rate),
         
         nn.Conv2d(128, 256, kernel_size=3, padding=1),
         nn.BatchNorm2d(256),
         nn.ReLU(inplace=True),
         nn.Conv2d(256, 256, kernel_size=3, padding=1),
         nn.BatchNorm2d(256),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(kernel_size=2, stride=2),
         nn.Dropout2d(dropout_rate),
      )
      
      # Calculate the size after convolutions
      # CIFAR-100 images are 32x32, after 3 maxpool layers: 32 -> 16 -> 8 -> 4
      conv_output_size = 256 * 4 * 4
      
      # Fully connected layers
      fc_layers = []
      input_size = conv_output_size
      
      for i in range(num_layers):
         output_size = hidden_size if i < num_layers - 1 else num_classes
         fc_layers.extend([
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size) if i < num_layers - 1 else nn.Identity(),
            nn.ReLU(inplace=True) if i < num_layers - 1 else nn.Identity(),
            nn.Dropout(dropout_rate) if i < num_layers - 1 else nn.Identity()
         ])
         input_size = output_size
      
      self.classifier = nn.Sequential(*fc_layers)
      
      # Initialize weights
      self._initialize_weights()
   
   def forward(self, x):
      x = self.features(x)
      x = torch.flatten(x, 1)
      x = self.classifier(x)
      return x
   
   def _initialize_weights(self):
      """Initialize model weights."""
      for m in self.modules():
         if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
               nn.init.constant_(m.bias, 0)
         elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
         elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


class ResNetCIFAR100(nn.Module):
   """ResNet model adapted for CIFAR-100."""
   
   def __init__(self, model_type: str = "resnet18", num_classes: int = 100, 
                dropout_rate: float = 0.2):
      super(ResNetCIFAR100, self).__init__()
      
      # Load pretrained ResNet
      if model_type == "resnet18":
         self.backbone = models.resnet18(weights=None)
      elif model_type == "resnet34":
         self.backbone = models.resnet34(weights=None)
      elif model_type == "resnet50":
         self.backbone = models.resnet50(weights=None)
      else:
         raise ValueError(f"Unsupported ResNet type: {model_type}")
      
      # Modify the first layer for CIFAR-100 (3x32x32 instead of 3x224x224)
      self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
      
      # Remove the maxpool layer since CIFAR-100 images are smaller
      self.backbone.maxpool = nn.Identity()
      
      # Modify the final layer for CIFAR-100 classes
      num_features = self.backbone.fc.in_features
      self.backbone.fc = nn.Sequential(
         nn.Dropout(dropout_rate),
         nn.Linear(num_features, num_classes)
      )
   
   def forward(self, x):
      return self.backbone(x)


class VGGCIFAR100(nn.Module):
   """VGG model adapted for CIFAR-100."""
   
   def __init__(self, model_type: str = "vgg16", num_classes: int = 100, 
                dropout_rate: float = 0.2):
      super(VGGCIFAR100, self).__init__()
      
      # Load VGG without pretrained weights
      if model_type == "vgg16":
         self.backbone = models.vgg16(pretrained=False)
      else:
         raise ValueError(f"Unsupported VGG type: {model_type}")
      
      # Modify the classifier for CIFAR-100
      num_features = self.backbone.classifier[-1].in_features
      self.backbone.classifier[-1] = nn.Sequential(
         nn.Dropout(dropout_rate),
         nn.Linear(num_features, num_classes)
      )
   
   def forward(self, x):
      return self.backbone(x)


class DenseNetCIFAR100(nn.Module):
   """DenseNet model adapted for CIFAR-100."""
   
   def __init__(self, model_type: str = "densenet121", num_classes: int = 100, 
                dropout_rate: float = 0.2):
      super(DenseNetCIFAR100, self).__init__()
      
      # Load DenseNet without pretrained weights
      if model_type == "densenet121":
         self.backbone = models.densenet121(pretrained=False)
      else:
         raise ValueError(f"Unsupported DenseNet type: {model_type}")
      
      # Modify the classifier for CIFAR-100
      num_features = self.backbone.classifier.in_features
      self.backbone.classifier = nn.Sequential(
         nn.Dropout(dropout_rate),
         nn.Linear(num_features, num_classes)
      )
   
   def forward(self, x):
      return self.backbone(x)


def get_model(model_type: str, num_classes: int = 100, hidden_size: int = 1024,
             num_layers: int = 3, dropout_rate: float = 0.2) -> nn.Module:
   """Factory function to create models."""
   
   if model_type == "custom":
      return CustomCNN(
         num_classes=num_classes,
         hidden_size=hidden_size,
         num_layers=num_layers,
         dropout_rate=dropout_rate
      )
   elif model_type.startswith("resnet"):
      return ResNetCIFAR100(
         model_type=model_type,
         num_classes=num_classes,
         dropout_rate=dropout_rate
      )
   elif model_type.startswith("vgg"):
      return VGGCIFAR100(
         model_type=model_type,
         num_classes=num_classes,
         dropout_rate=dropout_rate
      )
   elif model_type.startswith("densenet"):
      return DenseNetCIFAR100(
         model_type=model_type,
         num_classes=num_classes,
         dropout_rate=dropout_rate
      )
   else:
      raise ValueError(f"Unsupported model type: {model_type}")


def count_parameters(model: nn.Module) -> dict:
   """Count the number of parameters in a model."""
   total_params = sum(p.numel() for p in model.parameters())
   trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
   
   return {
      "total": total_params,
      "trainable": trainable_params,
      "non_trainable": total_params - trainable_params
   }


def get_model_summary(model: nn.Module, input_size: tuple = (3, 32, 32)) -> str:
   """Get a summary of the model architecture."""
   def register_hook(module):
      def hook(module, input, output):
         class_name = str(module.__class__).split(".")[-1].split("'")[0]
         module_idx = len(summary)
         
         m_key = f"{class_name}-{module_idx+1}"
         summary[m_key] = OrderedDict()
         summary[m_key]["input_shape"] = list(input[0].size())
         summary[m_key]["input_shape"][0] = -1
         
         if isinstance(output, (list, tuple)):
            summary[m_key]["output_shape"] = [
               [-1] + list(o.size())[1:] for o in output
            ]
         else:
            summary[m_key]["output_shape"] = list(output.size())
            summary[m_key]["output_shape"][0] = -1
         
         summary[m_key]["params"] = sum(p.numel() for p in module.parameters())
         summary[m_key]["trainable_params"] = sum(p.numel() for p in module.parameters() if p.requires_grad)
      
      hooks.append(module.register_forward_hook(hook))
   
   # Create hooks
   summary = OrderedDict()
   hooks = []
   
   # Register hooks
   model.apply(register_hook)
   
   # Make a forward pass
   x = torch.zeros(1, *input_size)
   model(x)
   
   # Remove hooks
   for h in hooks:
      h.remove()
   
   # Create summary string
   summary_str = "Model Summary:\n"
   summary_str += "-" * 80 + "\n"
   summary_str += f"{'Layer (type)':<25} {'Output Shape':<25} {'Param #':<15}\n"
   summary_str += "=" * 80 + "\n"
   
   total_params = 0
   trainable_params = 0
   
   for layer in summary:
      summary_str += f"{layer:<25} {str(summary[layer]['output_shape']):<25} {summary[layer]['params']:<15}\n"
      total_params += summary[layer]["params"]
      trainable_params += summary[layer]["trainable_params"]
   
   summary_str += "=" * 80 + "\n"
   summary_str += f"Total params: {total_params:,}\n"
   summary_str += f"Trainable params: {trainable_params:,}\n"
   summary_str += f"Non-trainable params: {total_params - trainable_params:,}\n"
   summary_str += "-" * 80 + "\n"
   
   return summary_str


# Import OrderedDict for model summary
from collections import OrderedDict 