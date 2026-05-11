import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, use_bn=True):
    layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def deconv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Generator(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, base=64):
        super().__init__()

        self.enc1 = conv_block(in_channels, base,    use_bn=False)
        self.enc2 = conv_block(base,        base * 2)
        self.enc3 = conv_block(base * 2,    base * 4)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, 3, 1, 1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )

        self.dec1 = deconv_block(base * 8, base * 4)
        self.dec2 = deconv_block(base * 4, base * 2)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base * 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.bottleneck(e3)
        d1 = self.dec1(b)
        d2 = self.dec2(d1)
        out = self.dec3(d2)
        return nn.functional.interpolate(out, size=(70, 70), mode="bilinear", align_corners=False)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, base=64):
        super().__init__()

        self.net = nn.Sequential(
            conv_block(in_channels, base,    use_bn=False),
            conv_block(base,        base * 2),
            conv_block(base * 2,    base * 4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator().to(device)
    D = Discriminator().to(device)
    x = torch.randn(4, 5, 70, 70).to(device)
    out = G(x)
    print("Generator output:", out.shape)
    print("Discriminator output:", D(out).shape)