import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg_out + max_out).view(b, c, 1, 1).expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# --- Single Stream Model (Dynamic Input Channels) ---
class SingleStreamCNN(nn.Module):
    def __init__(self, input_channels=3):
        super(SingleStreamCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.attention = CBAM(64)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# --- Dual Stream Model (Dynamic Slicing) ---
class DualStreamRipDetectorWithAttention(nn.Module):
    def __init__(self, rgb_in=3, wav_in=4):
        super(DualStreamRipDetectorWithAttention, self).__init__()
        self.rgb_in = rgb_in
        self.wav_in = wav_in

        # RGB Stream
        self.rgb_conv1 = nn.Sequential(nn.Conv2d(rgb_in, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                                       nn.MaxPool2d(2))
        self.rgb_conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.rgb_conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.rgb_attention = CBAM(64)

        # Wavelet Stream
        self.wavelet_conv1 = nn.Sequential(nn.Conv2d(wav_in, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                                           nn.MaxPool2d(2))
        self.wavelet_conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                           nn.MaxPool2d(2))
        self.wavelet_conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                           nn.MaxPool2d(2))
        self.wavelet_attention = CBAM(64)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, rgb, wavelet):
        # Dynamic slicing based on initialization
        if self.rgb_in == 2:
            rgb_input = rgb[:, 1:3, :, :]  # Drop Red channel (0)
        else:
            rgb_input = rgb[:, :3, :, :]  # All 3 channels

        if self.wav_in == 3:
            wavelet_input = wavelet[:, 1:, :, :]  # Drop LL (0)
        else:
            wavelet_input = wavelet[:, :, :, :]  # All 4 channels

        # RGB processing
        rgb_x = self.global_pool(self.rgb_attention(self.rgb_conv3(self.rgb_conv2(self.rgb_conv1(rgb_input)))))
        # Wavelet processing
        wav_x = self.global_pool(
            self.wavelet_attention(self.wavelet_conv3(self.wavelet_conv2(self.wavelet_conv1(wavelet_input)))))

        return self.classifier(torch.cat([rgb_x, wav_x], dim=1))