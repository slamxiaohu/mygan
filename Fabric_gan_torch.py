import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.utils as utils

# 检查是否有可用的GPU



# 数据预处理和归一化
def normalize_image(image):
    return image / 255.0


def normalize_defect_map(defect_map):
    return defect_map / 255.0


def load_and_preprocess_image(image_path, target_size=(512, 512)):
    print(f"Reading image: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 保留所有通道
    if image is None:
        raise FileNotFoundError(f"Error reading image {image_path}")
    if image.shape[2] == 4:  # 如果有透明度通道，将其转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    image = cv2.resize(image, target_size)
    image = normalize_image(image)
    return image


def load_and_preprocess_defect_map(defect_map_path, target_size=(512, 512)):
    print(f"Reading defect map: {defect_map_path}")
    if not os.path.exists(defect_map_path):
        raise FileNotFoundError(f"File not found: {defect_map_path}")
    defect_map = cv2.imread(defect_map_path, cv2.IMREAD_GRAYSCALE)
    if defect_map is None:
        raise FileNotFoundError(f"Error reading defect map {defect_map_path}")
    defect_map = cv2.resize(defect_map, target_size)
    defect_map = normalize_defect_map(defect_map)
    defect_map = np.expand_dims(defect_map, axis=-1)  # 添加通道维度
    return defect_map


class FabricDataset(Dataset):
    def __init__(self, image_dir, defect_map_dir, real_defect_images_dir, transform=None):
        self.image_dir = image_dir
        self.defect_map_dir = defect_map_dir
        self.real_defect_images_dir = real_defect_images_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.defect_map_files = sorted(os.listdir(defect_map_dir))
        self.real_defect_image_files = sorted(os.listdir(real_defect_images_dir))
        self.transform = transform

        assert len(self.image_files) == len(self.defect_map_files) == len(
            self.real_defect_image_files), "无缺陷织物图像和缺陷定位图的数量不一致"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        defect_path = os.path.join(self.defect_map_dir, self.defect_map_files[idx])
        real_defect_image_path = os.path.join(self.real_defect_images_dir, self.real_defect_image_files[idx])

        image = load_and_preprocess_image(img_path)
        defect_map = load_and_preprocess_defect_map(defect_path)
        real_defect_image = load_and_preprocess_image(real_defect_image_path)

        if self.transform:
            image = self.transform(image)
            defect_map = self.transform(defect_map)
            real_defect_image = self.transform(real_defect_image)

        return image, defect_map, real_defect_image


# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.float())  # 强制转换为float32类型
])

# 示例数据加载
base_dir = 'datasets'
image_dir = os.path.join(base_dir, 'background')
defect_map_dir = os.path.join(base_dir, 'defect_map')
real_defect_images_dir = os.path.join(base_dir, 'real_images')

# 创建数据集和数据加载器
dataset = FabricDataset(image_dir, defect_map_dir, real_defect_images_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(64, 3, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.float()  # 确保输入为 float 类型
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.float()  # 确保输入为 float 类型
        return self.main(x)


# 训练生成对抗网络
def train_gan(generator, discriminator, dataloader, epochs, batch_size):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0001)

    for epoch in range(epochs):
        for i, (texture_images, defect_maps, real_images) in enumerate(dataloader):
            real_images = real_images.to(device)
            texture_images = texture_images.to(device)
            defect_maps = defect_maps.to(device)

            real_labels = torch.ones(batch_size, 1, 32, 32, device=device)  # 512/16 = 32
            fake_labels = torch.zeros(batch_size, 1, 32, 32, device=device)

            # 训练判别器
            optimizer_d.zero_grad()
            generated_images = generator(torch.cat((texture_images, defect_maps), dim=1))
            d_loss_real = criterion(discriminator(real_images), real_labels)
            d_loss_fake = criterion(discriminator(generated_images.detach()), fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            d_loss.backward()
            utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)


            optimizer_d.step()

            # 训练生成器
            optimizer_g.zero_grad()
            g_loss = criterion(discriminator(generated_images), real_labels)
            g_loss.backward()
            optimizer_g.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Discriminator Loss: {d_loss.item()}, Generator Loss: {g_loss.item()}')

    # 保存生成器模型的权重
    torch.save(generator.state_dict(), 'generator_weights1.pth')

if __name__ == "__main__":
    # 构建生成器和判别器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 训练FabricGAN
    train_gan(generator, discriminator, dataloader, epochs=100, batch_size=8)




