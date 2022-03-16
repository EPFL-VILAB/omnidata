import torch.nn as nn
import torch.nn.functional as F
import torch

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels , in_channels // 2,
                kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetSemSeg(nn.Module):
    def __init__(self, n_channels, n_classes1, n_classes2, n_classes3, n_classes4, bilinear=True):
        super(UNetSemSeg, self).__init__()
        self.n_channels = n_channels
        self.n_classes1 = n_classes1
        self.n_classes2 = n_classes2
        self.n_classes3 = n_classes3
        self.n_classes4 = n_classes4
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc1 = OutConv(64, n_classes1)
        self.outc2 = OutConv(64, n_classes2)
        self.outc3 = OutConv(64, n_classes3)
        self.outc4 = OutConv(64, n_classes4)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits1 = self.outc1(x)
        logits2 = self.outc2(x)
        logits3 = self.outc3(x)
        logits4 = self.outc4(x)
        return logits1, logits2, logits3, logits4

class UNetSemSegCombined(nn.Module):
    def __init__(self, n_channels, n_classes1, bilinear=True):
        super(UNetSemSegCombined, self).__init__()
        self.n_channels = n_channels
        self.n_classes1 = n_classes1
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc1 = OutConv(64, n_classes1)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits1 = self.outc1(x)
        return logits1


###################################


class UNet_up_block(nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel, up_sample=True, use_skip=True):
        super().__init__()
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        if use_skip:
            self.conv1 = nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, output_channel)
        self.relu = torch.nn.ReLU()
        self.up_sample = up_sample

    def forward(self, x, prev_feature_map=None):
        if self.up_sample:
            x = self.up_sampling(x)
        if prev_feature_map is not None:
            x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x
 

class UNet_down_block(nn.Module):
    def __init__(self, input_channel, output_channel, down_size=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, output_channel)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        if self.down_size:
            x = self.max_pool(x)
        return x


class UNetSemSeg2(nn.Module):
    def __init__(self, downsample=6, n_channels=3, n_classes1=3, n_classes2=3, patch_size=1):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes1 = n_classes1
        self.n_classes2 = n_classes2
        # self.n_classes3 = n_classes3
        self.downsample = downsample
        self.patch_size = patch_size

        self.down1 = UNet_down_block(n_channels, 16, False)
        self.down_blocks = nn.ModuleList(
            [UNet_down_block(2**(4+i), 2**(5+i), True) for i in range(0, downsample)]
        )

        bottleneck = 2**(4 + downsample)
        self.mid_conv1 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, bottleneck)
        self.mid_conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, bottleneck)
        self.mid_conv3 = torch.nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, bottleneck)

        self.up_blocks_1 = nn.ModuleList(
            [UNet_up_block(2**(4+i), 2**(5+i), 2**(4+i)) for i in range(0, downsample)]
        )

        self.up_blocks_2 = nn.ModuleList(
            [UNet_up_block(2**(4+i), 2**(5+i), 2**(4+i)) for i in range(0, downsample)]
        )

        self.last_conv1_1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_conv1_2 = nn.Conv2d(16, 16, 3, padding=1)

        self.last_bn_1 = nn.GroupNorm(8, 16)
        self.last_bn_2 = nn.GroupNorm(8, 16)

        self.last_conv2_1 = nn.Conv2d(16, n_classes1, 1, padding=0)
        self.last_conv2_2 = nn.Conv2d(16, n_classes2, 1, padding=0)
        # self.last_conv2_3 = nn.Conv2d(16, n_classes3, 1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.down1(x)
        xvals = [x]
        for i in range(0, self.downsample):
            x = self.down_blocks[i](x)
            xvals.append(x)

        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))

        # decoder 1
        for i in range(0, self.downsample)[::-1]:
            if i == self.downsample - 1:
                x1 = self.up_blocks_1[i](x, xvals[i])
            else: 
                x1 = self.up_blocks_1[i](x1, xvals[i])
        x1 = self.relu(self.last_bn_1(self.last_conv1_1(x1)))
        logits1 = self.last_conv2_1(x1)

        # decoder 2
        for i in range(0, self.downsample)[::-1]:
            if i == self.downsample - 1:
                x2 = self.up_blocks_2[i](x, xvals[i])
            else:
                x2 = self.up_blocks_2[i](x2, xvals[i])
        x2 = self.relu(self.last_bn_2(self.last_conv1_2(x2)))
        logits2 = self.last_conv2_2(x2)


        return logits1, logits2 #, logits3