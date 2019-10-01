import pytesseract
from PIL import Image
import cv2
import numpy as np


class Preprocess():
    def __init__(self,image,gamma=1.9):
        self.image = image
        self.gamma = gamma
    
    def adjust_gamma(self):
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(self.image, table)

    def sharpening(self,image):
        smoothed = cv2.GaussianBlur(image,(3,3),5)
        unsh = cv2.addWeighted(image,1.5,smoothed,-0.5,0)
        return unsh


    def resizer(self,image):
        img_re = image
        scale_percent = 160 # percent of original size
        width = int(img_re.shape[1] * scale_percent / 100)
        height = int(img_re.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img_re, dim, interpolation = cv2.INTER_CUBIC)
        return resized

    def Filter(self):
        image_gamma = self.adjust_gamma()
        image_sharp = self.sharpening(image_gamma)
        resized = self.resizer(image_sharp)
        return resized
        