import cv2
import numpy as np

# 创建一个 512x512 的黑色图像
img = np.zeros((512, 512, 3), np.uint8)

# 在上面画一个蓝色的圆
cv2.circle(img, (256, 256), 100, (255, 0, 0), -1)

# 保存这张图片
cv2.imwrite('test_image.jpg', img)

print("你的第一个 OpenCV 程序已在 WSL2 环境下运行成功。")
print("图片已保存为 test_image.jpg")