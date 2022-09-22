from torchvision.models.resnet import resnet50
import torch.nn as nn

def make_resblocks():
    
    # no pretrained weights
    net = resnet50(pretrained=False)

    # Use it if you want to freeze backbone
    # net = resnet50(pretrained=True)
    # for param in net.parameters():
    #     param.requires_grad = False
    
    layer0_name = ['conv1','bn1','relu']
    layer1_name = ['maxpool','layer1']
    layer2_name = ['layer2']
    layer3_name = ['layer3']

    layer0 = nn.Sequential()
    layer1 = nn.Sequential()
    layer2 = nn.Sequential()
    layer3 = nn.Sequential()

    for n,c in net.named_children():
        if n in layer0_name:
            layer0.add_module(n,c)
        elif n in layer1_name:
            layer1.add_module(n,c)
        elif n in layer2_name:
            layer2.add_module(n,c)
        elif n in layer3_name:
            layer3 = c
        else:
            break

    return layer0, layer1, layer2, layer3
