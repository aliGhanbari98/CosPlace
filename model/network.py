
import torch
import logging
import torchvision
from torch import nn

from model.layers import Flatten, L2Norm, GeM
import augmentations
import parser

CHANNELS_NUM_IN_LAST_CONV = {
        "resnet18": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "vgg16": 512,
    }

args = parser.parse_arguments()

class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone, fc_output_dim):
        super().__init__()
        self.backbone, features_dim = get_backbone(backbone)
        self.aggregation = nn.Sequential(
                L2Norm(),
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),
                L2Norm()
            )
    
    def forward(self, x):
        # for multiscale - addedd by AB
        if args.run_multiscale == 1:
            blur_scales = [augmentations.CustomGaussianBlur(kernel_size=(13,13), sigma=x) for x in (1,3,20)]
            scaled_images = [blur_scales[i](x) for i in range(3)]
            first_stage = [self.backbone(y) for y in scaled_images]
            descriptors = [self.aggregation(z).unsqueeze(0) for z in first_stage]
            descriptors = torch.cat(tuple(descriptors),dim=0)
            #print("the dimension of descriptors after torch cat: " , descriptors.shape)
            #torch.Size([3, 16, 512])
            mean_descriptor = torch.mean(descriptors,dim=0)
            #print("the dimension of descriptors after mean: ", mean_descriptor.shape)
            return mean_descriptor
        else:
            x = self.backbone(x)
            #print("the shape of descriptors after backbone: torch.Size([16, 512, 16, 16])", x.shape)
            x = self.aggregation(x)
            #print("the shape of descriptors after aggreggation: torch.Size([16, 512])", x.shape)
            return x

def get_backbone(backbone_name):
    if backbone_name.startswith("resnet"):
        if backbone_name == "resnet18":
            backbone = torchvision.models.resnet18(pretrained=True)
        elif backbone_name == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=True)
        elif backbone_name == "resnet101":
            backbone = torchvision.models.resnet101(pretrained=True)
        
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    
    elif backbone_name == "vgg16":
        backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        for l in layers[:-5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")
    
    backbone = torch.nn.Sequential(*layers)
    
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim

