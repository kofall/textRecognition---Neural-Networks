import cv2
import numpy as np

def imgBlank():
    return np.zeros((100, 100, 3), np.uint8)

def imgStacking(images, size=(2000, 1000)):
    blank = imgBlank()
    if len(blank.shape) != 3:
        blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)

    rows = len(images)
    cols = 0
    for row in images:
        if len(row) > cols:
            cols = len(row)

    blank = cv2.resize(blank, (int(size[0]/cols), int(size[1]/len(images))))
    mosaic = None
    for row in images:
        part = None
        count = 0
        for img in row:
            if len(img.shape) != 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, (int(size[0]/cols), int(size[1]/len(images))))
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