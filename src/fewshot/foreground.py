import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(2)


def center_crop(image, ratio):
    width, height = image.size
    side = int(min(width, height) * ratio)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def grabcut_box(image, iterations=5):
    image = image.convert("RGB")
    width, height = image.size
    scale = 256 / max(width, height)
    if scale < 1.0:
        small = image.resize((int(width * scale), int(height * scale)), Image.BILINEAR)
    else:
        small = image
    array = np.array(small)
    rows, cols = array.shape[:2]

    mask = np.zeros((rows, cols), np.uint8)
    margin_x = int(cols * 0.06)
    margin_y = int(rows * 0.06)
    rect = (margin_x, margin_y, cols - 2 * margin_x, rows - 2 * margin_y)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    cv2.grabCut(array, mask, rect, background, foreground, iterations, cv2.GC_INIT_WITH_RECT)

    keep = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    if keep.mean() < 0.05:
        return None

    row_index = np.flatnonzero(keep.any(axis=1))
    col_index = np.flatnonzero(keep.any(axis=0))
    top, bottom = row_index[0], row_index[-1]
    left, right = col_index[0], col_index[-1]

    side = int(max(right - left + 1, bottom - top + 1) * 1.1)
    min_side = int((0.5 * rows * cols) ** 0.5)
    side = max(side, min_side)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    x0 = max(0, min(cols - side, int(round(center_x - side / 2))))
    y0 = max(0, min(rows - side, int(round(center_y - side / 2))))
    x1 = min(cols, x0 + side)
    y1 = min(rows, y0 + side)

    return (
        int(x0 / cols * width),
        int(y0 / rows * height),
        int(x1 / cols * width),
        int(y1 / rows * height),
    )


def grabcut_crop(image):
    image = image.convert("RGB")
    box = grabcut_box(image)
    if box is None:
        return image
    return image.crop(box)


def crop_foreground(image, mode):
    if mode == "none":
        return image.convert("RGB")
    if mode == "center":
        return center_crop(image.convert("RGB"), 0.70)
    if mode == "grabcut":
        return grabcut_crop(image)
    raise ValueError(mode)
