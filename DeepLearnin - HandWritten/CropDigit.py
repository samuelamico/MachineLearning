import cv2
import numpy as np
from PreProcess import Preprocess

""" 
Author: Samuel Amico 
Contact: sam.fst@gmail.com

Version alpha 1.0 ---
-> Load image in the Images Directory
-> Process Image
-> Crop Rects in Image
-> Return the letters

"""

def FindEdges(image_process,image_orig):
    res = cv2.findContours(image_process.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if cv2.__version__.split(".")[0] == '3':
        im2, contours, hierarchy = res
    else:
        contours, hierarchy = res

    rects = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        if h >= 16:
            # if height is enough
            # create rectangle for bounding
            rect = [x, y, w, h]
            rects.append(rect)
            cv2.rectangle(image_orig, (x, y), (x+w, y+h), (0, 255, 0), 1)
    return image_orig,rect



if __name__ == "__main__":
    # Letter Result Directory
    path_letter = 'ResultLetter/'
    cnt = 0
    # Put the image in the ImageHand directory
    list_img = glob.glob('Images/*')

    # For All images in directory
    for files in list_img:
        img = cv2.imread(files)
        # Call the class, adjust shadow
        preproce = Preprocess(img)
        result_norm_planes = preproce.Shadow_Correction()
        # Gamma correction:
        gamma = 0.3
        gamma_corection = preproce.adjust_gamma(gamma,image)
        # Erodion image or Sharpin:
            #img_erosion = preproce.Erosion(gamma_correction)
        img_erosion = preproce.sharpening(gamma_corection)
        # Find Letter:
        image_orig,rect =FindEdges(img_erosion,img)
        # Save Image:
        cv2.imwrite('Complete'+str(cnt)+'.png',image_orig)
        # Number image
        NI = 0
        for posi in rect:
            x,y,w,h = (posi[0],posi[1],posi[2],posi[3])
            letter_img = img[y:y+h,x:x+w]
            # Interpolation:
            letter_img = preproce.resizer(letter_img)
            cv2.imwrite(path_letter+'Image'+str(cnt)+'Letter'+str(NI)+'.png',letter_img)
            NI+=1
        cnt+=1


