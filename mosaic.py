import cv2
import numpy as np

def imgBlank():
    return np.zeros((100, 100, 3), np.uint8)

def imgStacking(images, size=(300, 300)):
    blank = imgBlank()
    if len(blank.shape) != 3:
        blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)
    blank = cv2.resize(blank, (size[0], size[1]))

    rows = len(images)
    cols = 0
    for row in images:
        if len(row) > cols:
            cols = len(row)
    mosaic = None
    for row in images:
        part = None
        count = 0
        for img in row:
            if len(img.shape) != 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, (size[0], size[1]))
            if part is None:
                part = img
            else:
                part = np.hstack((part, img))
            count += 1
        for i in range(count, cols):
            part = np.hstack((part, blank))
        if mosaic is None:
            mosaic = part
        else:
            mosaic = np.vstack((mosaic, part))
    return mosaic