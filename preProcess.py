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
        self.rowContours = None
        self.cutImages = None

        self.th1 = 128
        self.th2 = 128
        self.areaMin = 0
        self.rowApprox = 5
        self.minGap = 10
        self.cutSize = (250, 350)  # (width, height)
        self.boxApprox = 5

        self.__createTrackBar()
        self.__process()

    def __process(self):
        self.__toGray()
        self.__imgEdgeDetectCanny()
        self.__findContours()
        self.__selectByArea()
        self.__cutByRow()
        self.__sort()
        self.__cutByCol()
        self.__drawContours()

    def __toGray(self):
        self.imgGrayed = 255 - cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def __imgEdgeDetectCanny(self):
        self.imgCannied = cv2.Canny(self.image, self.th1, self.th2)

    def __findContours(self):
        self.contours, _ = cv2.findContours(self.imgGrayed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    def __selectByArea(self):
        cnts = []
        for cnt in self.contours:
            area = cv2.contourArea(cnt)
            if area >= self.areaMin:
                cnts.append(cnt)
        self.contours = cnts

    def __sort(self):
        rows = []
        if self.rowContours is None:
            pass
        else:
            for row in self.rowContours:
                positions = None
                cnts = None
                for cnt in row:
                    peri = cv2.arcLength(cnt, True)  # True means closed
                    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                    x, y, w, h = cv2.boundingRect(approx)
                    if positions is None:
                        positions = [x]
                        cnts = [cnt]
                    else:
                        for i, pos in enumerate(positions):
                            if x < pos:
                                positions.insert(i, x)
                                cnts.insert(i, cnt)
                                break
                            elif i == len(positions) - 1:
                                positions.append(x)
                                cnts.append(cnt)
                                break
                rows.append(cnts)
            self.rowContours = rows

    def __cutByRow(self):
        rowLevels = None
        if len(self.contours) == 0:
            self.rowContours = None
            self.cutImages = [[self.image]]
        else:
            for cnt in self.contours:
                peri = cv2.arcLength(cnt, True)  # True means closed
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                _, y, _, h = cv2.boundingRect(approx)
                if rowLevels is None:
                    rowLevels = [y + h]
                    self.rowContours = [[cnt]]
                else:
                    for i, level in enumerate(rowLevels):
                        if level - self.rowApprox < y + h < level + self.rowApprox:
                            self.rowContours[i].append(cnt)
                            break
                        elif y + h < level:
                            rowLevels.insert(i, y + h)
                            self.rowContours.insert(i, [cnt])
                            break
                        elif i == len(rowLevels) - 1:
                            rowLevels.append(y + h)
                            self.rowContours.append([cnt])
                            break

    def __getPerspective(self, x, y, w, h):
        letterPts = np.float32([(x - self.boxApprox, y - self.boxApprox),
                                (x + w + self.boxApprox, y - self.boxApprox),
                                (x - self.boxApprox, y + h + self.boxApprox),
                                (x + w + self.boxApprox, y + h + self.boxApprox)])
        imgPts = np.float32(
            [(0, 0), (self.cutSize[0], 0), (0, self.cutSize[1]), (self.cutSize[0], self.cutSize[1])])
        matrix = cv2.getPerspectiveTransform(letterPts, imgPts)
        return cv2.warpPerspective(self.image, matrix, (self.cutSize[0], self.cutSize[1]))

    def __cutByCol(self):
        previous = 0
        if self.rowContours is None:
            pass
        else:
            self.cutImages = []
            for row in self.rowContours:
                for i, cnt in enumerate(row):
                    peri = cv2.arcLength(cnt, True)  # True means closed
                    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                    x, y, w, h = cv2.boundingRect(approx)
                    perspective = self.__getPerspective(x, y, w, h)
                    if i == 0:
                        self.cutImages.append([perspective])
                    elif x - previous < self.minGap:
                        self.cutImages[-1].append(perspective)
                    else:
                        self.cutImages.append([perspective])
                    previous = x + w

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

    def __empty(self, x):
        pass

    def __createTrackBar(self):
        cv2.namedWindow("Parameters")
        cv2.resizeWindow("Parameters", 500, 250)
        cv2.createTrackbar("TH 1", "Parameters", 128, 255, self.__empty)
        cv2.createTrackbar("TH 2", "Parameters", 128, 255, self.__empty)
        cv2.createTrackbar("MIN AREA", "Parameters", 0, 10000, self.__empty)
        cv2.createTrackbar("ROW APPROX", "Parameters", 5, 20, self.__empty)
        cv2.createTrackbar("MIN GAP", "Parameters", 10, 100, self.__empty)
        cv2.createTrackbar("BOX APPROX", "Parameters", 5, 20, self.__empty)

    def updateStatus(self):
        self.th1 = cv2.getTrackbarPos("TH 1", "Parameters")
        self.th2 = cv2.getTrackbarPos("TH 2", "Parameters")
        self.areaMin = cv2.getTrackbarPos("MIN AREA", "Parameters")
        self.rowApprox = cv2.getTrackbarPos("ROW APPROX", "Parameters")
        self.minGap = cv2.getTrackbarPos("MIN GAP", "Parameters")
        self.boxApprox = cv2.getTrackbarPos("BOX APPROX", "Parameters")
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
