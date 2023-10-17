import torch
import torch.nn as nn


class Branch(nn.Module):
    """
    Branch
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(Branch, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.IN(x)
        return x
        

class ChannelAttention(nn.Module):
    """
    ChannelAttention(CA)
    """
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    SpatialAttention(SA)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CSAMBlock(nn.Module):
    """
    CSAMBlock(CSAM)
    """
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CSAMBlock, self).__init__()

        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channel, channel, 3, padding=1, bias=False)

        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        conv1_out = self.conv1(x)
        # CA and SA in parallel
        c_out = conv1_out * self.channelattention(conv1_out)
        s_out = conv1_out * self.spatialattention(conv1_out)
        cs_out = c_out + s_out
        conv2_out = self.conv2(cs_out)
        out = conv2_out + x
        return out


class CSAMBlockGroup(nn.Module):
    """
    CSAMBlockGroup
    """
    def __init__(self, dim, nums):
        super(CSAMBlockGroup, self).__init__()

        modules = [CSAMBlock(dim) for _ in range(nums)]
        self.group = nn.Sequential(*modules)

    def forward(self, x):
        out = self.group(x)
        return out


class MLP(nn.Module):
    """
    MLP
    """
    def __init__(self, dim=3136):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 4, bias=False),  # 3136 -> 784
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim // 4, dim // 16, bias=False),  # 784 -> 196
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim // 16, 1, bias=False),  # 196 -> 1
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class AUIQE(nn.Module):
    """
    AUIQE
    """

    def __init__(self, dim=24, nums=6):
        super(AUIQE, self).__init__()

        assert dim % 3 == 0, "dim mod 3 must equal 0"
        self.dim = dim
        # Branch
        self.branch_dim = dim // 3
        self.branch_3 = Branch(in_channels=3, out_channels=self.branch_dim, kernel_size=3)
        self.branch_5 = Branch(in_channels=3, out_channels=self.branch_dim, kernel_size=5)
        self.branch_7 = Branch(in_channels=3, out_channels=self.branch_dim, kernel_size=7)
        # CSAMs
        self.group = CSAMBlockGroup(self.dim, nums=nums)
        # Conv Pool x2
        self.conv_1 = nn.Conv2d(self.dim, self.dim // 4, 3, padding=1, bias=False)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2 = nn.Conv2d(self.dim // 4, 1, 3, padding=1, bias=False)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.LeakyReLU(inplace=True)
        # MLP
        self.mlp = MLP(dim=3136)

    def forward(self, x):
        # Branch
        x1 = self.branch_3(x)  # b, 8, 224, 224
        x2 = self.branch_5(x)  # b, 8, 224, 224
        x3 = self.branch_7(x)  # b, 8, 224, 224
        cat_out = torch.cat([x1, x2, x3], dim=1)  # b, 24, 224, 224
        # CSAMs
        group_out = self.group(cat_out)  # b 24 224 224
        # Conv Pool x2
        x4 = self.act(self.conv_1(group_out))  # B 6 224 224
        x4 = self.max_pool_1(x4)  # B 6 112 112
        x4 = self.act(self.conv_2(x4))  # B 1 112 112
        x4 = self.max_pool_2(x4)  # B 1 56 56
        # MLP
        x4 = torch.flatten(x4, 1)  # B 3136
        out = self.mlp(x4)  # B 1
        return out


if __name__ == '__main__':
    # Test
    input = torch.rand([2, 3, 224, 224])
    net = AUIQE()
    output = net(input)
    print(f"output: {output}")
    print(f"output.shape: {output.shape}")
