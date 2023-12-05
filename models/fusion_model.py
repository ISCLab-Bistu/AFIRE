import torch
from torch import nn
from models.common import reflect_conv
from models.ChannelAttention import AdaptiveChannelAttention, DeepAdaptiveFusion
from models.SENet import SELayer, SEEnhancementModule


def CMDAF(vi_feature, ir_feature):
    sigmoid = nn.Sigmoid()
    gap = nn.AdaptiveAvgPool2d(1)
    batch_size, channels, _, _ = vi_feature.size()

    sub_vi_ir = vi_feature - ir_feature
    vi_ir_div = sub_vi_ir * sigmoid(gap(sub_vi_ir))

    sub_ir_vi = ir_feature - vi_feature
    ir_vi_div = sub_ir_vi * sigmoid(gap(sub_ir_vi))

    # 特征加上各自的带有简易通道注意力机制的互补特征
    vi_feature += ir_vi_div
    ir_feature += vi_ir_div

    return vi_feature, ir_feature


def Fusion(vi_out, ir_out):
    return torch.cat([vi_out, ir_out], dim=1)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=16, stride=1, padding=0)
        self.ir_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=16, stride=1, padding=0)

        self.vi_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.ir_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)

        self.vi_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.ir_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)

        self.vi_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.ir_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)

        self.vi_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.ir_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)

        self.vi_se2 = SELayer(16)
        self.ir_se2 = SELayer(16)
        self.vi_se3 = SELayer(32)
        self.ir_se3 = SELayer(32)
        self.vi_se4 = SELayer(64)
        self.ir_se4 = SELayer(64)

        self.seem2 = SEEnhancementModule(16)
        self.seem3 = SEEnhancementModule(32)
        self.seem4 = SEEnhancementModule(64)

    def forward(self, y_vi_image, ir_image):
        activate = nn.LeakyReLU()
        vi_out = activate(self.vi_conv1(y_vi_image))
        ir_out = activate(self.ir_conv1(ir_image))

        vi_out, ir_out = (activate(self.vi_conv2(vi_out)), activate(self.ir_conv2(ir_out)))#(128,16,64,64)
        vi_out, ir_out = (self.vi_se2(vi_out), self.ir_se2(ir_out))
        vi_out, ir_out = self.seem2(vi_out, ir_out)

        vi_out, ir_out = (activate(self.vi_conv3(vi_out)), activate(self.ir_conv3(ir_out)))#(128,32,64,64)
        vi_out, ir_out = (self.vi_se3(vi_out), self.ir_se3(ir_out))

        vi_out, ir_out = (activate(self.vi_conv4(vi_out)), activate(self.ir_conv4(ir_out)))#(128,64,64,64)
        vi_out, ir_out = (self.vi_se4(vi_out), self.ir_se4(ir_out))

        vi_out, ir_out = activate(self.vi_conv5(vi_out)), activate(self.ir_conv5(ir_out))#(128,128,64,64)

        return vi_out, ir_out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = reflect_conv(in_channels=256, kernel_size=3, out_channels=256, stride=1, pad=1)
        self.conv2 = reflect_conv(in_channels=256, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.conv3 = reflect_conv(in_channels=128, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv4 = reflect_conv(in_channels=64, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.conv5 = nn.Conv2d(in_channels=32, kernel_size=1, out_channels=1, stride=1, padding=0)

    def forward(self, x):
        activate = nn.LeakyReLU()
        x = activate(self.conv1(x))
        x = activate(self.conv2(x))
        x = activate(self.conv3(x))
        x = activate(self.conv4(x))#(128,32,64,64)
        x = nn.Tanh()(self.conv5(x)) / 2 + 0.5  # 将范围从[-1,1]转换为[0,1]
        return x


class DAFusion(nn.Module):
    def __init__(self):
        super(DAFusion, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.daf = DeepAdaptiveFusion(256,256)

    def forward(self, y_vi_image, ir_image):
        vi_encoder_out, ir_encoder_out = self.encoder(y_vi_image, ir_image)#(128,128,64,64) (128,128,64,64)
        encoder_out = self.daf(vi_encoder_out, ir_encoder_out)
        fused_image = self.decoder(encoder_out)#输出单通道
        return fused_image
