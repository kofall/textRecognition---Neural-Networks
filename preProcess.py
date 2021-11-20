import cv2
import numpy as np

class preProcess:
    def __init__(self, img):
        self.image = img
        self.contours = None
        self.imgGrayed = None
        self.imgCannied = None
        self.imgDrawWithoutBox = None
        self.imgDrawWithBox = None
        self.cutImages = None

        self.th1 = 128
        self.th2 = 128
        self.areaMin = 0
        self.cutSize = (250, 350)  # (width, height)
        self.__createTrackBar()
        self.__process()

    def __process(self):
        self.__toGray()
        self.__imgEdgeDetectCanny()
        self.__findContours()
        self.__selectByArea()
        self.__sort()
        self.__drawContours()
        self.__cutByContour()

    def __toGray(self):
        self.imgGrayed = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def __imgEdgeDetectCanny(self):
        self.imgCannied = cv2.Canny(self.image, self.th1, self.th2)

    def __findContours(self):
        self.contours, _ = cv2.findContours(self.imgCannied, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def __selectByArea(self):
        cnts = []
        if self.contours is None:
            pass
        else:
            for cnt in self.contours:
                area = cv2.contourArea(cnt)
                if area >= self.areaMin:
                    cnts.append(cnt)
            self.contours = cnts

    def __sort(self):
        lengths = None
        cnts = None
        if self.contours is None:
            pass
        else:
            for i, cnt in enumerate(self.contours):
                peri = cv2.arcLength(cnt, True)  # True means closed
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                l = x**2 + y**2
                if lengths is None:
                    lengths = [l]
                    cnts = [cnt]
                else:
                    for j, length in enumerate(lengths):
                        if l > length:
                            if j == len(lengths) - 1:
                                lengths.append(l)
                                cnts.append(cnt)
                                break;
                            else:
                                continue
                        else:
                            if j == 0:
                                lengths.insert(0, l)
                                cnts.insert(0, cnt)
                            else:
                                lengths.insert(j - 1, l)
                                cnts.insert(j - 1, cnt)
                            break
            self.contours = cnts

    def __drawContours(self):
        img1 = self.image.copy()
        img2 = self.image.copy()

        if self.contours is None:
            self.imgDrawWithoutBox = self.image
            self.imgDrawWithBox = self.image
        else:
            for cnt in self.contours:
                # without box
                area = cv2.contourArea(cnt)
                cv2.drawContours(img1, cnt, -1, (255, 0, 255), 2)
                cv2.drawContours(img2, cnt, -1, (255, 0, 255), 2)
                # with box
                peri = cv2.arcLength(cnt, True)  # True means closed
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 5)
                # cv2.putText(img2, "Points: {}".format(str(len(approx))), (x, y - 25), cv2.FONT_HERSHEY_COMPLEX,
                #           0.7,
                #         (0, 255, 0), 1)
                # cv2.putText(img2, "Area: {}".format(str(int(area))), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                #         (0, 255, 0), 1)
            self.imgDrawWithoutBox = img1
            self.imgDrawWithBox = img2

    def __cutByContour(self):
        images = []
        if self.contours is None:
            self.cutImages = [self.image]
        else:
            for cnt in self.contours:
                peri = cv2.arcLength(cnt, True)  # True means closed
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                x -= 1
                y -= 1
                w += 2
                h += 2
                letterPts = np.float32([(x, y), (x + w, y), (x, y + h), (x + w, y + h)])
                imgPts = np.float32([(0, 0), (self.cutSize[0], 0), (0, self.cutSize[1]), (self.cutSize[0], self.cutSize[1])])
                matrix = cv2.getPerspectiveTransform(letterPts, imgPts)
                images.append(cv2.warpPerspective(self.image, matrix, (self.cutSize[0], self.cutSize[1])))
            self.cutImages = images

    def __empty(self, x):
        pass

    def __createTrackBar(self):
        cv2.namedWindow("Parameters")
        cv2.resizeWindow("Parameters", 500, 150)
        cv2.createTrackbar("TH 1", "Parameters", 128, 255, self.__empty)
        cv2.createTrackbar("TH 2", "Parameters", 128, 255, self.__empty)
        cv2.createTrackbar("AREA MIN", "Parameters", 0, 1000, self.__empty)

    def updateStatus(self):
        self.th1 = cv2.getTrackbarPos("TH 1", "Parameters")
        self.th2 = cv2.getTrackbarPos("TH 2", "Parameters")
        self.areaMin = cv2.getTrackbarPos("AREA MIN", "Parameters")
        self.__process()

    def updateImage(self, img):
        self.image = img
        self.__process()

    def getImage(self):
        return self.image

    def getGrayedImage(self):
        return self.imgGrayed

    def getCanniedImage(self):
        return self.imgCannied

    def getContours(self):
        return self.contours

    def getContouredImage(self):
        return self.imgDrawWithoutBox

    def getContouredImageWithBox(self):
        return self.imgDrawWithBox

    def getCutImages(self):
        return self.cutImages