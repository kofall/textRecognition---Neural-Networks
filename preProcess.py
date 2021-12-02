import cv2
import numpy as np
from skimage import filters


class preProcess:
    def __init__(self, img):
        self.originalImage = img
        self.cap = cv2.VideoCapture(0)
        self.image = None
        self.contours = None
        self.rowContours = None

        self.imgBlured = None
        self.imgGrayed = None
        self.imgCannied = None
        self.imgDilated = None
        self.imgDrawWithoutBox = None
        self.imgDrawWithBox = None
        self.cutImages = None

        self.source = 1
        self.th1 = 128
        self.th2 = 128
        self.areaMin = 50
        self.rowApprox = 5
        self.minGap = 10
        self.backgroundApprox = 10
        self.boxApprox = 0

        self.__createTrackBar()
        self.__process()

    def __process(self):
        self.__checkSource()
        self.__toBlur()
        self.__toGray()
        self.__imgEdgeDetectCanny()
        self.__imgDilation()
        self.__findContours()
        self.__selectByArea()
        self.__cutByRow()
        self.__sort()
        self.__cutByCol()
        self.__drawContours()

    def __checkSource(self):
        if self.source == 1:
            self.image = self.originalImage
        elif self.source == 2:
            self.image = self.originalImage
            self.image = cv2.resize(self.image, (int(self.image.shape[0]/2), int(self.image.shape[1]/2)))
        else:
            _, self.image = self.cap.read()
            self.image = cv2.resize(self.image, (800, 600))

    def __toBlur(self):
        self.imgBlured = cv2.GaussianBlur(self.image, (7, 7), 0)

    def __toGray(self):
        self.imgGrayed = 255 - cv2.cvtColor((self.image if self.source != 3 else self.imgBlured), cv2.COLOR_BGR2GRAY)
        self.imgGrayed = (self.__thrsh(self.imgGrayed) if self.source == 2 else self.imgGrayed)

    def __thrsh(self, img):
        _, th = cv2.threshold(img, self.th1, self.th2, cv2.THRESH_BINARY)
        return th

    def __yenThrsh(self, img):
        thresh = filters.threshold_yen(img)
        img[img > thresh] = 255
        img[img <= thresh] = 0
        return img

    def __imgEdgeDetectCanny(self):
        self.imgCannied = cv2.Canny(self.imgGrayed, self.th1, self.th2)

    def __imgDilation(self, kernel=np.ones((5, 5), np.uint8), iters=1):
        self.imgDilated = cv2.dilate(self.imgCannied, kernel, iterations=iters)

    def __findContours(self):
        self.contours, _ = cv2.findContours((self.imgGrayed if self.source != 3 else self.imgDilated), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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
            self.cutImages = None
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
            [(0, 0), (w, 0), (0, h), (w, h)])
        matrix = cv2.getPerspectiveTransform(letterPts, imgPts)
        img = np.array(cv2.warpPerspective(255 - self.imgGrayed, matrix, (w, h)))
        return self.__yenThrsh(img)

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
                    perspective = self.__fillToSquare(self.__getPerspective(x, y, w, h))
                    if i == 0:
                        self.cutImages.append([perspective])
                    elif x - previous < self.minGap:
                        self.cutImages[-1].append(perspective)
                    else:
                        self.cutImages.append([perspective])
                    previous = x + w
                self.cutImages.append([None])

    def __fillToSquare(self, img):
        filled = 255 - cv2.cvtColor(self.imgWhite(max(img.shape) + 2 * self.backgroundApprox), cv2.COLOR_BGR2GRAY)
        y_offset_up = self.backgroundApprox + int(max(img.shape)/2 - img.shape[0]/2)
        y_offset_down = self.backgroundApprox + int(max(img.shape)/2 + img.shape[0]/2)
        x_offset_left = self.backgroundApprox + int(max(img.shape)/2 - img.shape[1]/2)
        x_offset_right = self.backgroundApprox + int(max(img.shape)/2 + img.shape[1]/2)
        filled[y_offset_up : y_offset_down, x_offset_left : x_offset_right] = img
        return filled


    def __drawContours(self):
        img1 = self.image.copy()
        img2 = self.image.copy()
        if self.contours is None:
            pass
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

    def imgBlack(self, x=100):
        return np.zeros((x, x, 3), np.uint8)

    def imgWhite(self, x=100):
        return np.ones((x, x, 3), np.uint8)

    def __empty(self, x):
        pass

    def __createTrackBar(self):
        cv2.namedWindow("Parameters")
        cv2.resizeWindow("Parameters", 700, 300)
        cv2.createTrackbar("SOURCE", "Parameters", 0, 2, self.__empty)
        cv2.createTrackbar("TH 1", "Parameters", 128, 255, self.__empty)
        cv2.createTrackbar("TH 2", "Parameters", 128, 255, self.__empty)
        cv2.createTrackbar("MIN AREA", "Parameters", 50, 1000, self.__empty)
        cv2.createTrackbar("ROW APPROX", "Parameters", 5, 150, self.__empty)
        cv2.createTrackbar("MIN GAP", "Parameters", 10, 200, self.__empty)
        cv2.createTrackbar("BACK APPROX", "Parameters", 10, 200, self.__empty)
        cv2.createTrackbar("BOX APPROX", "Parameters", 0, 100, self.__empty)

    def updateStatus(self):
        self.source = cv2.getTrackbarPos("SOURCE", "Parameters") + 1
        self.th1 = cv2.getTrackbarPos("TH 1", "Parameters")
        self.th2 = cv2.getTrackbarPos("TH 2", "Parameters")
        self.areaMin = cv2.getTrackbarPos("MIN AREA", "Parameters")
        self.rowApprox = cv2.getTrackbarPos("ROW APPROX", "Parameters")
        self.minGap = cv2.getTrackbarPos("MIN GAP", "Parameters")
        self.backgroundApprox = cv2.getTrackbarPos("BACK APPROX", "Parameters")
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
