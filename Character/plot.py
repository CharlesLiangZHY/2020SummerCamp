import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys

from character import character_dict
from character import margin_dict
# from labels import labels
from lowercase import labels
# from test import labels

from PIL import Image,ImageDraw,ImageFont

A = '.\\A\\'
B = '.\\B\\'



def padding(img):
    h,w = img.shape[0:2]
    new_img = np.zeros([512, 512, 3], dtype=np.uint8)
    new_img[:,:,:] = 255
    new_img[int((512-h)/2) : int(512-(512-h)/2) , int((512-w)/2): int(512-(512-w)/2), :] = img
    return new_img

def resize(img,n):
    h,w = img.shape[0:2]
    return cv.resize(img,(int(w/n), int(h/n)))

def merge(A,B,buf):
    h1,w1,c1 = A.shape # c = 4 r g b alpha channel all 0..255
    h2,w2,c2 = B.shape
    if c1 != c2:
        print("Error")
        return
    Img = np.zeros([512, w1+w2-min(buf,w2), 4], dtype=np.uint8)
    Img = Img + 255
    Img[:,:,3] = 0
    for i in range(512):
        for j in range(w1):
            if A[i,j,3] != 0:
                Img[i,j,0] = 0
                Img[i,j,1] = 0
                Img[i,j,2] = 0
                Img[i,j,3] = A[i,j,3]
    for i in range(512):
        for j in range(min(buf,w2)):
            if B[i,j,3] != 0:
                Img[i,w1-buf+j,0] = 0
                Img[i,w1-buf+j,1] = 0
                Img[i,w1-buf+j,2] = 0
                if B[i,j,3] > Img[i,w1-buf+j,3]:
                    Img[i,w1-buf+j,3] = B[i,j,3]

    for i in range(512):
        for j in range(0,w2-buf):
            if B[i,buf+j,3] != 0:
                Img[i,w1+j,0] = 0
                Img[i,w1+j,1] = 0
                Img[i,w1+j,2] = 0
                Img[i,w1+j,3] = B[i,buf+j,3]
    return Img

def normal_text(filename,W,H):
    name = filename
    if filename[0] == '_':
        name = name[1:]
    text = name

    font = ImageFont.truetype('arialbd.ttf',150)
    w, h = font.getsize(text)
    img = Image.new("RGB",[max(w,W),H],"white")
    draw = ImageDraw.Draw(img)

    draw.textsize(text, font=font)
    

    draw.text([max((W-w),0)/2,(H-h)/2+50],text,'black',font=font) # +50 to align
    img.save(A+filename+'.jpg','jpeg')
    img = cv.imread(A+filename+'.jpg')
    width = img.shape[1]
    img = resize(img, width//512 + 1)
    img = padding(img)
    cv.imwrite(A+filename+'.jpg',img)


def paired_data(filename,plot=False):
    characters = []
    name = filename
    if name[0] == "_":
        name = name[1:]
    for c in name:
        characters.append(cv.imread(character_dict[c], -1))
    Img = characters[0]
    margin_1 = 0
    margin_2 = 0
    h,w,c = Img.shape
    if name[0] > "Z":
        margin_1 = w - margin_dict[name[0]][1]
    else:
        margin_1 = w - margin_dict[name[0]]
    if len(name) > 1:
        for i in range(1,len(name)):
            if name[i] > "Z":
                margin_2 = margin_dict[name[i]][0]
                buf = margin_1 + margin_2
                Img = merge(Img,characters[i],buf)
                h,w,c = characters[i].shape
                if buf < w:
                    margin_1 = w - margin_dict[name[i]][1]
                else:
                    margin_1 = margin_1 - margin_dict[name[i]][1] + margin_dict[name[i]][0]
    else:
        Img = np.zeros([512, w, 4], dtype=np.uint8)
        Img = Img + 255
        Img[:,:,3] = 0
        for i in range(h):
            for j in range(w):
                if characters[0][i,j,3] != 0:
                    Img[i,j,0] = 0
                    Img[i,j,1] = 0
                    Img[i,j,2] = 0
                    Img[i,j,3] = characters[0][i,j,3]

    cv.imwrite(B+filename+".jpg", Img)
    # img = cv.imread(B+filename+'.jpg')
    # width = img.shape[1]
    # img = resize(img, width//512 + 1)
    # img = padding(img)
    # cv.imwrite(B+filename+'.jpg',img)


    h,w,c = Img.shape
    print(filename)
    normal_text(filename,max(w,512),512)
    # normal_text(filename,w,512)
    if plot:
        plt.imshow(Img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        name = sys.argv[1]
        paired_data(name,True)
    elif len(sys.argv) > 2:
        print("Error")
        sys.exit()
    else:
        for word in labels:
            paired_data(word)
