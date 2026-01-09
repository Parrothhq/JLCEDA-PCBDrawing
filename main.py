from time import sleep

import cv2
import numpy as np
import os

def hex_to_BGR(hex_color):
    hex_color = hex_color.lstrip('#')
    bgr_color = np.array([int(hex_color[i:i + 2], 16) for i in (4, 2, 0)])
    return bgr_color

# 定义9种颜色的Hex值
hexColors = {
    "Red":    "#FF0000",
    "Orange": "#FFA500",
    "Yellow": "#FFFF00",
    "Green":  "#3b7e09",
    "Cyan":   "#84fafe",
    "Blue":   "#0000FF",
    "Purple": "#800080",
    "Black":  "#25180f",
    "White":  "#FFFFFF",
}
palette = np.array([hex_to_BGR(v) for v in hexColors.values()])

image = cv2.imread("origin.jpg") # 读取原始图像
if image is None:
    raise FileNotFoundError("找不到origin.jpg，请确保图片在当前目录下。")

scale_factor = 1
target_width = int(image.shape[1] * scale_factor)
target_height = int(image.shape[0] * scale_factor)
image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

h, w, c = image.shape
img_flat = image.reshape(-1, 3)
dists = np.linalg.norm(img_flat[:, None, :] - palette[None, :, :], axis=2)
nearest = np.argmin(dists, axis=1)
result = palette[nearest].reshape(h, w, 3).astype(np.uint8)

cv2.imwrite("output_9color.png", result)

sleep(3)  # 确保文件写入完成

# 颜色分离后输出路径
output_dir = "separated_colors"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取9色图像
img_9color = cv2.imread("output_9color.png")
if img_9color is None:
    raise FileNotFoundError("找不到 output_9color.png，请确保已生成该图片。")

for idx, (color_name, hex_code) in enumerate(hexColors.items()):
    bgr = hex_to_BGR(hex_code)
    # 创建蒙板：与当前颜色相同的像素
    mask = np.all(img_9color == bgr, axis=-1)
    mask_img = np.zeros_like(img_9color)
    mask_img[mask] = bgr
    # 保存蒙板图片
    cv2.imwrite(f"{output_dir}/mask_{color_name}.png", mask_img)
    print(f"已保存: {output_dir}/mask_{color_name}.png")