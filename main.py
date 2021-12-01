import cv2
import mosaic as msc
from preProcess import preProcess
from predict import predict

########################################################################
imgPath = 'data/sample04.png'

########################################################################

def readImg(path):
    return cv2.imread(path)

def showImg(img):
    cv2.imshow('Okno', img)

def projection():
    img = readImg(imgPath)
    pp = preProcess(img)
    while True:
        mosaic = msc.imgStacking([[pp.getImage(), pp.getGrayedImage(), pp.getCanniedImage(), pp.getContouredImage(),
                                   pp.getContouredImageWithBox()]] + pp.getCutImages(),
                                 (1500, 700))
        showImg(pp.getCutImages()[0][0])
        print(predict(cv2.resize(pp.getCutImages()[0][0], (128, 128))))
        pp.updateStatus()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    projection()