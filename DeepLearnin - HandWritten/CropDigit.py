import cv2
import numpy as np
from PreProcess import Preprocess

class Cropped():

    def __init__(self,image):
        self.image = image
    
    def CropImage(self,RecP,image):
        # ROI = [y:y+w,x:x+h]
        image_crop = []
        coord_crop = []
        image_crop_more = []
        cnt = 0
        for i in RecP:
            x,y,w,h = (i[0],i[1],i[2],i[3])
            im = image[y:h,abs(x):w]
            if(len(im)!=0):
                preproce = Preprocess(im)
                image_process = preproce.Filter()
                w,h = image_process.shape[:2]
                image_crop.append(Image.fromarray(image_process))
                cnt+=1 
            coord_crop.append((x,y,w,h))
        return image_crop,coord_crop
    
    def FindBox(self):
        image = cv2.imread(self.image)
        orig = image.copy()
        # Convert to Gray
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # Get the Edges:  
        img_sobel = cv2.Sobel(image_gray,cv2.CV_8U,1,0)
        img_threshold = cv2.threshold(img_sobel,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        # Apply Morphology:
        element = cv2.getStructuringElement(cv2.MORPH_RECT,ele_size)
        img_threshold = cv2.morphologyEx(img_threshold[1],cv2.MORPH_CLOSE,element)
        # Find the Edges:
        res = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cv2.__version__.split(".")[0] == '3':
            _, contours, hierarchy = res
        else:
            contours, hierarchy = res

        Rect = [cv2.boundingRect(i) for i in contours if i.shape[0]>100]
        RectP = [(int(i[0]-i[2]*0.08),int(i[1]-i[3]*0.08),int(i[0]+i[2]*1.1),int(i[1]+i[3]*1.1)) for i in Rect]

        # CUT the image => Crop the image
        image_crop,coord_crop,image_crop_more = self.CropImage(RectP,image)

