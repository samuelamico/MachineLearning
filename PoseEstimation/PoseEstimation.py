import numpy as np
import cv2
import time


point_selected = False
point_new = False
point = np.array([[[]]],dtype=np.float32)

global pt,xm,ym

b = np.array([[]],dtype=np.float32)

lk_params = lk_params = dict(winSize = (15,15),maxLevel=4,
             criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

capture = cv2.VideoCapture(0)
_,old_img = capture.read()
old_gray = cv2.cvtColor(old_img,cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray,mask=None,**feature_params)
mask = np.zeros_like(old_img)


global cnt
cnt = 0

def on_mouse(event,x,y,flags,params):
    global point,point_selected,old_points,xm,ym,contador,point_new,cnt
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Start Mouse Position:',str(x),str(y))
        xm = np.float32(x) 
        ym = np.float32(y)
        cnt+=1
        point_selected = True
        point_new = True
        


print("Click na imagem para Start")
p1 = np.array([[[10,10]]],dtype=np.float32)
val =  np.array([[0,0]],dtype=np.float32)

cv2.namedWindow('img')
cv2.setMouseCallback('img',on_mouse,0)



# path com as imagens:
arquivo = open('dados.txt','w')

vetor = []
#Salvando em arquivo
def salvar_arquivo(name_classe,vetor):
    #arquivo = open('dados.txt','w')
    arquivo.write(name_classe)
    arquivo.write(",")
    for i in range(len(vetor)):
        arquivo.write(str(vetor[i][0]))
        arquivo.write(",")
        arquivo.write(str(vetor[i][1]))
        arquivo.write(",")
    arquivo.write("\n")






def optical(p1,gray_frame):
    old_gray = gray_frame
    numero_image = 0
    while capture.isOpened():
        global img
        ret,img = capture.read()
        gray_frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        new_point,status,error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame,p1,None,**lk_params)
        old_gray = gray_frame.copy()
        p1 = new_point
        for i in range(new_point.shape[1]):
            cv2.circle(gray_frame,(new_point[0,i,0],new_point[0,i,1]),5,255,-1)
        cv2.imshow('img',gray_frame)
        pressed_key = cv2.waitKey(1) & 0xFF
        ########### Salvar posicoes:
        if pressed_key == ord("b"):
            name = 'Baixo/'+str(numero_image)+'.png'
            cv2.imwrite(name,gray_frame)
            vetor = []
            for i in range(p1.shape[1]):
                x = p1[0,i,0]
                y = p1[0,i,1]
                vetor.append([x,y])
            salvar_arquivo('Baixo',vetor)
            numero_image+=1
        elif pressed_key == ord("c"):
            name = 'Cima/'+str(numero_image)+'.png'
            cv2.imwrite(name,gray_frame)
            vetor = []
            for i in range(p1.shape[1]):
                x = p1[0,i,0]
                y = p1[0,i,1]
                vetor.append([x,y])
            salvar_arquivo('Cima',vetor)
            numero_image+=1
        elif pressed_key == ord("z"):
            break
    

while capture.isOpened():
    global img
    ret,img = capture.read()
    gray_frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if ret is not None:
        if point_new == True:
            b = np.array([[[xm,ym]]],dtype=np.float32)
            val = np.vstack( ( val,np.array([[xm,ym]]) ) )
            p1 = np.array([val],dtype=np.float32)
            point_new = False
    for i in range(p1.shape[1]):
        cv2.circle(gray_frame,(p1[0,i,0],p1[0,i,1]),5,255,-1)
    cv2.imshow('img',gray_frame)
    pressed_key = cv2.waitKey(1) & 0xFF
    if cnt > 11:
        optical(p1,gray_frame)
        break
    
cv2.destroyAllWindows()
capture.release()
arquivo.close()
