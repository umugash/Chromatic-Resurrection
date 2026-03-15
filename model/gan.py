import torch
from torch import nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))

class SelfAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.query = nn.Conv2d(in_ch, in_ch//8, 1)
        self.key = nn.Conv2d(in_ch, in_ch//8, 1)
        self.value = nn.Conv2d(in_ch, in_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        b, c, h, w = x.size()
        f = self.query(x).view(b, -1, h*w)
        g = self.key(x).view(b, -1, h*w)
        attn = torch.bmm(f.permute(0,2,1), g)
        attn = F.softmax(attn, dim=-1)
        v = self.value(x).view(b, c, -1)
        out = torch.bmm(v, attn.permute(0,2,1)).view(b, c, h, w)
        return self.gamma * out + x

class Generator(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, base=32):
        super().__init__()
        self.enc1 = ResBlock(in_ch, base)
        self.enc2 = ResBlock(base, base*2)
        self.enc3 = ResBlock(base*2, base*4)
        self.enc4 = ResBlock(base*4, base*4)
        self.attn = SelfAttention(base*4)
        self.dec4 = ResBlock(base*4 + base*4, base*2)
        self.dec3 = ResBlock(base*2 + base*2, base)
        self.dec2 = ResBlock(base + base, base)
        self.dec1 = nn.Conv2d(base + in_ch, out_ch, 1)
        self.pool = nn.MaxPool2d(2,2)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.attn(e4)
        d4 = self.dec4(torch.cat([F.interpolate(b, size=e3.shape[2:], mode='bilinear', align_corners=True), e3], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=True), e2], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=True), e1], dim=1))
        out = self.dec1(torch.cat([F.interpolate(d2, size=x.shape[2:], mode='bilinear', align_corners=True), x], dim=1))
        return out
