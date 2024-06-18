import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, input1_dir, input2_dir, real_dir, transform=None):
        self.input1_dir = input1_dir
        self.input2_dir = input2_dir
        self.real_dir = real_dir
        self.transform = transform
        self.input1_files = sorted(os.listdir(input1_dir))
        self.input2_files = sorted(os.listdir(input2_dir))
        self.real_files = sorted(os.listdir(real_dir))

    def __len__(self):
        return len(self.input1_files)

    def __getitem__(self, idx):
        input1_path = os.path.join(self.input1_dir, self.input1_files[idx])
        input2_path = os.path.join(self.input2_dir, self.input2_files[idx])
        real_path = os.path.join(self.real_dir, self.real_files[idx])

        input1 = Image.open(input1_path).convert("RGB")
        input2 = Image.open(input2_path).convert("RGB")
        real = Image.open(real_path).convert("RGB")

        if self.transform:
            input1 = self.transform(input1)
            input2 = self.transform(input2)
            real = self.transform(real)

        return input1, input2, real


# 定义生成器和判别器
class DoubleUNetGenerator(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(DoubleUNetGenerator, self).__init__()

        self.unet1 = UNet(in_channels1)
        self.unet2 = UNet(in_channels2)

        self.decoder = nn.ModuleList([
            self.deconv_block(1024, 512, 4, 2, 1),
            self.deconv_block(1024, 512, 4, 2, 1),
            self.deconv_block(1024, 512, 4, 2, 1),
            self.deconv_block(1024, 512, 4, 2, 1),
            self.deconv_block(1024, 256, 4, 2, 1),
            self.deconv_block(512, 128, 4, 2, 1),
            self.deconv_block(256, 64, 4, 2, 1),
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1, bias=False)
        ])
        self.tanh = nn.Tanh()

    def deconv_block(self, in_channels, out_channels, kernel_size, stride, padding, final_layer=False):
        if final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )

    def forward(self, input1, input2):
        enc_features1, encoded1 = self.unet1(input1, return_features=True)
        enc_features2, encoded2 = self.unet2(input2, return_features=True)

        combined_encoded = torch.cat([encoded1, encoded2], dim=1)

        x = combined_encoded
        for idx, dec in enumerate(self.decoder):
            x = dec(x)
            if idx < len(enc_features1):
                x = self.match_size(x, enc_features1[idx])
                x = torch.cat([x, enc_features1[idx]], dim=1)

        return self.tanh(x)

    def match_size(self, x, target):
        _, _, H, W = target.size()
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)


class UNet(nn.Module):
    def __init__(self, in_channels):
        super(UNet, self).__init__()

        self.encoder = nn.ModuleList([
            self.conv_block(in_channels, 64, 4, 2, 1),
            self.conv_block(64, 128, 4, 2, 1),
            self.conv_block(128, 256, 4, 2, 1),
            self.conv_block(256, 512, 4, 2, 1),
            self.conv_block(512, 512, 4, 2, 1),
            self.conv_block(512, 512, 4, 2, 1),
            self.conv_block(512, 512, 4, 2, 1),
            self.conv_block(512, 512, 4, 2, 1, final_layer=True)
        ])

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding, final_layer=False):
        if final_layer:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )

    def forward(self, x, return_features=False):
        enc_features = []

        for enc in self.encoder:
            x = enc(x)
            enc_features.append(x)

        if return_features:
            return enc_features[:-1][::-1], x
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=4, stride=2, padding=1),  # 256*256*64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128*128*128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 64*64*256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 32*32*512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 16*16*512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # 1*1*1
            nn.Sigmoid()
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 创建数据集和数据加载器
dataset = CustomDataset(input1_dir='datasets/background', input2_dir='datasets/defect_map_w', real_dir='datasets/real_images',
                        transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 创建生成器和判别器
generator = DoubleUNetGenerator(in_channels1=3, in_channels2=3, out_channels=3).to(device)
discriminator = Discriminator(in_channels=3).to(device)

# 定义损失函数和优化器
adversarial_loss = nn.BCELoss().to(device)
l1_loss = nn.L1Loss().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 训练循环
num_epochs = 100

for epoch in range(num_epochs):
    for i, (input1, input2, real_samples) in enumerate(dataloader):
        input1 = input1.to(device)
        input2 = input2.to(device)
        real_samples = real_samples.to(device)

        # 生成器前向传播
        fake_samples = generator(input1, input2)

        # 判别器前向传播
        real_validity = discriminator(real_samples, real_samples)
        fake_validity = discriminator(fake_samples.detach(), real_samples)

        # 判别器损失
        real_loss = adversarial_loss(real_validity, torch.ones_like(real_validity))
        fake_loss = adversarial_loss(fake_validity, torch.zeros_like(fake_validity))
        d_loss = (real_loss + fake_loss) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # 生成器损失
        fake_validity = discriminator(fake_samples, real_samples)
        g_adv = adversarial_loss(fake_validity, torch.ones_like(fake_validity))
        g_l1 = l1_loss(fake_samples, real_samples)
        g_loss = g_adv + 100 * g_l1

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        if i % 10 == 0:
            print(
                f'Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] Discriminator Loss: {d_loss.item()} Generator Loss: {g_loss.item()}')

    # 保存模型
    torch.save(generator.state_dict(), f'generator_epoch_{epoch}.pth')
    torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch}.pth')
