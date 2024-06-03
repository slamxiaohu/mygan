import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, Concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model

# 检查是否有可用的GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

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

# 构建生成对抗网络（GAN）
def build_generator(input_shape=(512, 512, 3), defect_shape=(512, 512, 1)):
    texture_input = Input(shape=input_shape)
    defect_map_input = Input(shape=defect_shape)
    combined_input = Concatenate()([texture_input, defect_map_input])

    x = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(combined_input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Conv2DTranspose(3, (4, 4), padding='same', activation='tanh')(x)

    return Model(inputs=[texture_input, defect_map_input], outputs=x)

def build_discriminator(input_shape=(512, 512, 3)):
    input_image = Input(shape=input_shape)

    x = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(input_image)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Conv2D(512, (4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    x = Conv2D(1, (4, 4), padding='same', activation='sigmoid')(x)

    return Model(inputs=input_image, outputs=x)

def build_gan(generator, discriminator, input_shape=(512, 512, 3), defect_shape=(512, 512, 1)):
    discriminator.trainable = False
    texture_input = Input(shape=input_shape)
    defect_map_input = Input(shape=defect_shape)
    generated_image = generator([texture_input, defect_map_input])
    gan_output = discriminator(generated_image)
    return Model(inputs=[texture_input, defect_map_input], outputs=gan_output)

# 训练生成对抗网络
def train_gan(generator, discriminator, gan, epochs, batch_size, texture_data, defect_maps, real_defect_images):
    for epoch in range(epochs):
        for batch in range(len(texture_data) // batch_size):
            idx = np.random.randint(0, len(texture_data), batch_size)
            real_images = real_defect_images[idx]
            texture_images = texture_data[idx]
            defect_maps_batch = defect_maps[idx]

            generated_images = generator.predict([texture_images, defect_maps_batch])

            real_labels = np.ones((batch_size, 32, 32, 1))  # 512/16 = 32
            fake_labels = np.zeros((batch_size, 32, 32, 1))

            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = gan.train_on_batch([texture_images, defect_maps_batch], real_labels)

        print(f'Epoch {epoch + 1}/{epochs}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}')

# 示例数据加载
def load_dataset(image_dir, defect_map_dir, target_size=(512, 512)):
    image_files = sorted(os.listdir(image_dir))
    defect_map_files = sorted(os.listdir(defect_map_dir))

    assert len(image_files) == len(defect_map_files), "无缺陷织物图像和缺陷定位图的数量不一致"

    images = []
    defect_maps = []

    for img_file, defect_file in zip(image_files, defect_map_files):
        img_path = os.path.join(image_dir, img_file)
        defect_path = os.path.join(defect_map_dir, defect_file)

        image = load_and_preprocess_image(img_path, target_size)
        defect_map = load_and_preprocess_defect_map(defect_path, target_size)

        images.append(image)
        defect_maps.append(defect_map)

    return np.array(images), np.array(defect_maps)

# 示例用法
base_dir = 'datasets'
image_dir = os.path.join(base_dir, 'background')
defect_map_dir = os.path.join(base_dir, 'defect_map')
real_defect_images_dir = os.path.join(base_dir, 'real_images')

# 加载数据
texture_data, defect_maps = load_dataset(image_dir, defect_map_dir)
real_defect_images, _ = load_dataset(real_defect_images_dir, defect_map_dir)  # 假设真实缺陷图像目录结构相同

# 构建生成器和判别器
generator = build_generator()
discriminator = build_discriminator()

# 编译判别器
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# 构建和编译FabricGAN
gan = build_gan(generator, discriminator)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# 训练FabricGAN
train_gan(generator, discriminator, gan, epochs=100, batch_size=1, texture_data=texture_data, defect_maps=defect_maps,
          real_defect_images=real_defect_images)
