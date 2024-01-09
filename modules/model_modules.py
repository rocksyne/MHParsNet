# system module
import math

# 3d party imports
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class ResNetBackboneNetwork(nn.Module):
    
    def __init__(self, resnet_version:str="resnet34"):
        super(ResNetBackboneNetwork, self).__init__()

        # For variab;e `resnet_versions`, each value of the dictionary is a list of integers.
        # Each intege represents the number of times a residual block should be repeated. 
        # See ../doc/block_repetetion.webp for examples.
        # See also page https://arxiv.org/pdf/1512.03385.pdf, page 5 for the block numbering
        resnet_versions = {"resnet18":[2, 2, 2, 2], "resnet34":[3, 4, 6, 3],
                          "resnet50":[3, 4, 6, 3], "resnet101":[3, 4, 23, 3]}
        
        if resnet_version not in list(resnet_versions.keys()):
            raise ValueError("{} is an invalid resnet version. Pleas choose one of {}".format(resnet_version,resnet_versions.keys()))
        
        self.inplanes = 64
        self.block = BasicBlock if resnet_version in ["resnet18","resnet34"] else Bottleneck
        self.layers = resnet_versions[resnet_version]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=2)
        
        # self.fpn_sizes = [128, 256, 512] for BasicBlock
        if self.block == BasicBlock:
            self.fpn_sizes = [self.layer2[self.layers[1] - 1].conv2.out_channels, self.layer3[self.layers[2] - 1].conv2.out_channels,
                         self.layer4[self.layers[3] - 1].conv2.out_channels]
            
        elif self.block == Bottleneck:
            self.fpn_sizes = [self.layer2[self.layers[1] - 1].conv3.out_channels, self.layer3[self.layers[2] - 1].conv3.out_channels,
                         self.layer4[self.layers[3] - 1].conv3.out_channels]
        
        else:
            raise ValueError(f"Block type {self.block} not understood")
            
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    

    def forward(self,batch_image_input):
        c1 = self.conv1(batch_image_input)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool(c1)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        #return {"c1": c1, "c2": c2, "c3":c3, "c4":c4, "c5":c5}
        return c2,c3,c4,c5

        
class PyramidFeatures(nn.Module):
    def __init__(self, C2_size=64, C3_size=128, C4_size=256, C5_size=512, feature_size=256, pyramid_levels=[2,3,4,5,6,7]):
        super(PyramidFeatures, self).__init__()

        self.pyramid_levels = pyramid_levels

        # 1x1 convolutions for the lateral outputs from resnet
        self.conv1x1_C5 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.conv1x1_C4 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.conv1x1_C3 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.conv1x1_C2 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)

        # upsampling method for all previous top down layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        # 3x3 convolution for top-down layers
        self.conv3x3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs

        # P5 = conv1x1(C5)
        M5 = self.conv1x1_C5(C5)
        P5 = M5

        # P4 = conv3x3(C4_lateral + upsampled(M5))
        C4_lateral = self.conv1x1_C4(C4)
        upsampled_M5 = self._upsample_feature_maps(M5,C4_lateral)
        M4 = C4_lateral + upsampled_M5
        P4 = self.conv3x3(M4)

        # P3 = conv3x3(C3_lateral + upsampled(M4))
        C3_lateral = self.conv1x1_C3(C3)
        upsampled_M4 = self._upsample_feature_maps(M4,C3_lateral)
        M3 = C3_lateral + upsampled_M4
        P3 = self.conv3x3(M3)

        # P2 = conv3x3(C2_lateral + upsampled(M3))
        C2_lateral = self.conv1x1_C2(C2)
        upsampled_M3 = self._upsample_feature_maps(M3,C2_lateral)
        M2 = C2_lateral + upsampled_M3
        P2 = self.conv3x3(M2)

        # not my codes, and I am lazy to change them
        # it works so lets keep it
        # P6
        P6_x = self.P6(C5)

        # P7
        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        out = {"M2":M2,"M3":M3,"M4":M4,"M5":M5,
               "P2":P2, "P3":P3, "P4":P4, "P5":P5,"P6":P6_x,"P7":P7_x}
        
        return out
    
    def _upsample_feature_maps(self,feature_to_be_upsampled,lateral_connection_features, inter_mode='bilinear'):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        
        Documentation from old colde base. May remove >> Todo:
        ==========================================================
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.

        New (updated) documentation
        ==========================================================
        Original codebase uses F.upsample() but this has been modified to F.interpolate()
        because F.upsample() is decapitated.
        '''
        _,_,H,W = lateral_connection_features.size() # use this to keep the size even
        return F.interpolate(feature_to_be_upsampled, size=(H,W), mode=inter_mode)