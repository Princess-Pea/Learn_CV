import cv2
import os

# 1. 读取刚才拖进去的照片
input_file = 'test.jpg'
img = cv2.imread(input_file)

if img is None:
    print(f"错误：找不到文件 {input_file}，请检查文件名！")
else:
    # 2. 灰度化处理（这是所有视觉识别的第一步，因为颜色有时是干扰）
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 高斯模糊（用来去噪，让图像变得“平滑”）
    # blurred_img = cv2.GaussianBlur(gray_img, (15, 15), 0)
    # 调参：改变模糊程度
    # blurred_img = cv2.GaussianBlur(gray_img, (3, 3), 0) 
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0) 

    # 4. 边缘检测（把图片的轮廓抠出来，这是识别物体的基础）
    # edges = cv2.Canny(blurred_img, 50, 150)
    # 调参：改变 Canny 门槛
    # edges = cv2.Canny(blurred_img, 30, 100)
    edges = cv2.Canny(blurred_img, 40, 120)


    # 5. 保存结果
    cv2.imwrite('test_gray.jpg', gray_img)
    cv2.imwrite('test_edges3.jpg', edges)

    print("--- 实验成功 ---")
    print(f"原始图片大小: {img.shape}")
    print("已生成灰度图: test_gray.jpg")
    print("已生成边缘轮廓图: test_edges3.jpg")