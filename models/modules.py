import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, channel_img, features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size

        self.disc = nn.Sequential(
            nn.Conv2d(channel_img+1, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self.__block(features_d, features_d*2, 4, 2, 1),
            self.__block(features_d*2, features_d*4, 4, 2, 1),
            self.__block(features_d*4, features_d*8, 4, 2, 1),
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
        )

        self.embed = nn.Embedding(num_classes, img_size*img_size)

    def __block(self, in_channel, out_channel, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=True),
            nn.InstanceNorm2d(out_channel),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, label):
        embedding = self.embed(label).view(label.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_channel, feature_g, num_classes, embed_size):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self.__block(z_dim+embed_size, feature_g*16, 4, 1, 0),
            self.__block(feature_g*16, feature_g*8, 4, 2, 1),
            self.__block(feature_g*8, feature_g*4, 4, 2, 1),
            self.__block(feature_g*4, feature_g*2, 4, 2, 1),
            nn.ConvTranspose2d(feature_g*2, img_channel, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def __block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, label):
        embedding = self.embed(label).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)


def initialise_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    initialise_weights(gen)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success")

