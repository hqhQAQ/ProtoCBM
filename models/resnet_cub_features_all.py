import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model


class ResNet_CUB_features(nn.Module):
    def __init__(self, base_architecture, pretrained=True, model_dir='pretrained_models'):
        super(ResNet_CUB_features, self).__init__()
        # resnet18_cub
        self.model = ptcv_get_model(base_architecture, pretrained=pretrained, root=model_dir)
        del self.model.output    # Delete the last layer
        self.features = nn.Sequential(*list(self.model.children()))

    def forward(self, x):
        for layer in self.features[0][:-1]:
            x = layer(x)
        return x
    
    def forward_all(self, x):
        all_feas = []
        for layer_idx, layer in enumerate(self.features[0][:-1]):
            x = layer(x)
            if layer_idx >= 1:
                all_feas.append(x)

        return x, all_feas
    
    # def conv_info(self):
    #     return self.model.kernel_sizes, self.model.strides, self.model.paddings
    

def resnet18_cub_features(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on DET
    """
    model = ResNet_CUB_features(base_architecture='resnet18_cub', pretrained=pretrained)
    
    # features = model.model.features
    # for name in features.named_modules():
    #     print(name[0])

    return model