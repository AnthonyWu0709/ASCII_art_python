import sys

import cv2
import numpy as np


def get_image() -> np.ndarray:
    if len(sys.argv) <= 1:
        print("请在命令后指定一个图片路径！")
        exit(1)

    original_file_path = sys.argv[1]
    original_image = cv2.imread(original_file_path)

    # Debugging
    # cv2.imshow('Original Image', original_image)

    return original_image


def resize_image(original_image: np.ndarray) -> np.ndarray:
    original_height, original_width = original_image.shape[:2]
    # 目标图片尺寸 ： new_width=150
    new_width: int = 150
    new_height = int(original_height * (new_width / original_width) * 0.75)  # 字符一般是长方形，避免控制台输出的字符画拉伸过长
    resized_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Debugging
    # cv2.imshow('Resized Image', resized_image)

    return resized_image


def convert_gray_image_to_ascii(gray_image: np.ndarray) -> str:
    """可以在此自定义像素点灰度对应字符，灰度值由低到高"""
    gray_to_ascii_map = r'▩▦▨▧▥▤◘◙❒❐❑❏▣⊡▢□□▫▫'

    # 对灰度图片缩放灰度值
    min_value = np.min(gray_image)
    max_value = np.max(gray_image)
    scaled_gray_image = ((gray_image - min_value) / (max_value - min_value)) * (len(gray_to_ascii_map) - 1)

    # 缩放过灰度值过后的图片根据gray_to_ascii_map字符串换成长字符串
    ascii_image = ''
    for row in scaled_gray_image:
        ascii_row = ''
        for pixel in row:
            ascii_row += gray_to_ascii_map[int(pixel)]
        ascii_image += ascii_row + '\n'

    return ascii_image


def paint_color_on_ascii_image(ascii_image: str, original_image: np.ndarray) -> str:
    colored_ascii_image = ''
    ascii_image_rows = ascii_image.split('\n')
    for current_ascii_image_row, current_original_image_row in zip(ascii_image_rows, original_image):
        colored_ascii_row = ''
        for char, pixel in zip(current_ascii_image_row, current_original_image_row):
            """注意，OpenCV读取图片默认是BGR格式"""
            b, g, r = pixel
            colored_ascii_row += f"\033[38;2;{r};{g};{b}m{char}\033[0m"
        colored_ascii_image += colored_ascii_row + '\n'

    return colored_ascii_image


# 获取图片
img = get_image()

# 把图片等比例缩放，其中宽度缩放到150px
resized_image = resize_image(img)

img_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
ascii_image = convert_gray_image_to_ascii(img_gray)

print(paint_color_on_ascii_image(ascii_image, resized_image))
