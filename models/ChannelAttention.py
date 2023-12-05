import torch
import torch.nn as nn

class AdaptiveChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveChannelAttention, self).__init__()

        self.soft_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pooled = self.soft_pooling(x)
        pooled = pooled.squeeze()  # Remove the singleton dimensions
        fc_output = self.fc(pooled)
        weights = self.sigmoid(fc_output)
        weighted_output = x * weights.unsqueeze(-1).unsqueeze(-1)

        return weighted_output

class DeepAdaptiveScore(nn.Module):
    def __init__(self):
        super(DeepAdaptiveScore, self).__init__()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.soft_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.soft_max = nn.AdaptiveMaxPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_channel, _ = torch.max(x, dim=1, keepdim=True)
        avg_channel = torch.mean(x, dim=1, keepdim=True)

        concatenated = torch.cat((max_channel, avg_channel), dim=1)

        reduction = self.conv(concatenated)
        soft_max = self.soft_max(reduction)
        soft_avg = self.soft_avg(reduction)
        added = soft_max + soft_avg

        # deep adaptive score
        das = self.sigmoid(added)

        return das

class DeepAdaptiveFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeepAdaptiveFusion, self).__init__()

        self.vi_score = DeepAdaptiveScore()
        self.ir_score = DeepAdaptiveScore()
        self.channel_attention = AdaptiveChannelAttention(in_channels, out_channels)

    def forward(self, vi, ir):
        vi_score = self.vi_score(vi)
        ir_score = self.ir_score(ir)
        score_fusion = torch.cat([vi_score * vi, ir_score * ir], dim=1)
        out = self.channel_attention(score_fusion)

        return out

# daf = DeepAdaptiveFusion(256,256)
# vi = torch.randn(128,128,64,64)
# ir = torch.randn(128,128,64,64)
# out = daf(vi, ir)
# print(out.shape)