from PIL import Image

# 打开图像文件
img = Image.open("output/generated_image1.jpg")

# 将图像转换为灰度图像
gray_img = img.convert("L")

# 保存灰度图像
gray_img.save("gray_image.jpg")

# 显示灰度图像
gray_img.show()
