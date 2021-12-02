import cv2
import mosaic as msc
from preProcess import preProcess
from predict import predict as prd
from threading import Thread
import time
import numpy as np

########################################################################
imgPath = 'data/img05.png'
#imgPath = 'data/tekst08.png'
windowSize = (1500, 750)
########################################################################
words = None
finish = False
text = ""
########################################################################

def readImg(path):
    return cv2.imread(path)


def showImg(nazwa, img):
    cv2.imshow(nazwa, img)


def saveText(text):
    f = open("text.txt", "w+")
    f.write(text)
    f.close()


def readText():
    printBreak = lambda: print('=' * 50)
    printBreak()

    global words, finish, text
    while words is None and not finish:
        time.sleep(1)
    while not finish:
        # time.sleep(1)
        # continue
        if words is None:
            time.sleep(1)
            continue
        newLine = False
        text = ""
        for word in words:
            for letter in word:
                if letter is None:
                    text += "\n"
                    newLine = True
                else:
                    text += prd(letter)
            if newLine:
                newLine = False
                continue
            text += " "

        text = text.replace("l", "I");
        text = text.upper()
        print("\n" + text)
        saveText(text)
        printBreak()


def projection():
    img = readImg(imgPath)
    pp = preProcess(img)

    thread = Thread(target=readText)
    thread.start()

    global words, finish
    while True:
        words = pp.getCutImages()
        mosaic = msc.imgStacking([[pp.getImage(), pp.getGrayedImage(), pp.getCanniedImage(), pp.getContouredImage(),
                                   pp.getContouredImageWithBox()]], (1500, 400))
        found = msc.imgStacking(np.array(([[pp.imgBlack()]] if words is None else words)), windowSize)

        showImg("Mosaic", mosaic)
        showImg("Letters", found)

        pp.updateStatus()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            finish = True
            break


if __name__ == '__main__':
    projection()
