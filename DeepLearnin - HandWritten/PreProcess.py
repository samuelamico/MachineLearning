import pytesseract
from PIL import Image
import cv2
import numpy as np


class Preprocess():
    def __init__(self,image):
        self.image = image
    
    def adjust_gamma(self,gamma,image):
        invGamma = 1.0 / gamma
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

    def Shadow_Correction(self):
        rgb_planes = cv2.split(self.image)
        result_planes = []
        result_norm_planes = []
        dst = np.zeros(shape=(5,2))
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((13,13), np.uint8))
            bg_img = cv2.medianBlur(dilated_img,21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img, dst ,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

        result = cv2.merge(result_planes)
        result_norm = cv2.merge(result_norm_planes)
        return result_norm_planes

    def Erosion(self,image):
        img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #Thres
        ret,thr= cv2.threshold(img_gray,100,255,cv2.THRESH_BINARY)
        # Erosion
        kernel = np.ones((3,3), np.uint8) 
        img_erosion = cv2.erode(thr, kernel, iterations=1)
        return img_erosion

        