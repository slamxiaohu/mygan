import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from Fabric_gan_torch import Generator


if __name__ == "__main__":
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载生成器模型的权重
    generator = Generator().to(device)
    generator.load_state_dict(torch.load('generator_weights1.pth'))
    generator.eval()


    # 数据预处理和归一化
    def normalize_image(image):
        return image / 255.0


    def load_and_preprocess_image(image_path, target_size=(512, 512)):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 保留所有通道
        if image.shape[2] == 4:  # 如果有透明度通道，将其转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = cv2.resize(image, target_size)
        image = normalize_image(image)
        return image


    def load_and_preprocess_defect_map(defect_map_path, target_size=(512, 512)):
        defect_map = cv2.imread(defect_map_path, cv2.IMREAD_GRAYSCALE)
        defect_map = cv2.resize(defect_map, target_size)
        defect_map = normalize_image(defect_map)
        defect_map = np.expand_dims(defect_map, axis=-1)  # 添加通道维度
        return defect_map


    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float())  # 强制转换为float32类型
    ])

    # 示例输入图像和缺陷图像
    example_image_path = 'datasets/background/image_1.png'
    example_defect_map_path = 'datasets/defect_map/IMG_0414(2)_defect.png'

    # 加载和预处理输入数据
    example_image = load_and_preprocess_image(example_image_path)
    example_defect_map = load_and_preprocess_defect_map(example_defect_map_path)

    # 转换为张量并移动到设备
    example_image_tensor = transform(example_image).to(device)
    example_defect_map_tensor = transform(example_defect_map).to(device)

    # 添加批量维度
    example_image_tensor = example_image_tensor.unsqueeze(0)
    example_defect_map_tensor = example_defect_map_tensor.unsqueeze(0)

    # 生成图片
    with torch.no_grad():
        generated_image = generator(torch.cat((example_image_tensor, example_defect_map_tensor), dim=1)).squeeze(0)

    # 将生成的张量转换为图像格式
    generated_image = generated_image.permute(1, 2, 0).cpu().numpy()  # 调整维度顺序为 (H, W, C)
    generated_image = (generated_image + 1) / 2 * 255  # 将 [-1, 1] 转换为 [0, 255]
    generated_image = generated_image.astype(np.uint8)

    # 显示生成的图片
    plt.imshow(cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # 保存生成的图片
    # 保存生成的图片到指定文件夹
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)  # 创建文件夹（如果不存在）
    output_path = os.path.join(output_dir, 'generated_image1.jpg')
    cv2.imwrite(output_path, cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR))
    print(f"Generated image saved to {output_path}")
    # cv2.imwrite('generated_image.jpg', cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR))
