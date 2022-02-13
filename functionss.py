import cv2
import numpy as np
import os
import random

zagotovkaNroslogo = []


def razmetca(mas):
    W = mas.shape[1]
    H = mas.shape[0]
    return W,H

def nrevracenDla(mas, W=64, H=64):
    return mas.reshape(W*H*3)

def nrevracenS(mas,W=64,H=64):
    return mas.reshape((H, W, 3))


def nrevracenDlaMas(mas, W=64, H=64):
    masM=np.zeros((mas.shape[0],W*H*3))
    for i in range(mas.shape[0]):
        masM[i] = mas[i].reshape(W*H*3)
    return masM

def nrevracenSMas(mas,W,H):
    masM = np.zeros((mas.shape[0], W, H, 3))
    for i in range(mas.shape[0]):
        masM[i] = mas[i].mas.reshape((H, W, 3))
    return masM

def trasformMAX(img):
    W, H = razmetca(img)
    return (W-64)*(H-64)

def trasformerirovatFoto(img,N = None,sme = None):
    W,H =razmetca(img)
    if N == None:
        N = (W-64)*(H-64)
    if sme == None:
        sme = (W-64)*(H-64)

    imgM = np.zeros((N,64,64,3),dtype=np.uint8)
    VH = H-64
    VW = W-64

    if sme == 0:
        x, y = 0, 0
    elif sme > VW*VH:
        print("error --> rasformerirovatFoto --> sme")
    elif sme < VW:
        x, y = 0,sme
    elif sme > VW:
        smeZ = sme-VW
        x = smeZ // VW
        y = smeZ %VW
    else:
        print("eror --> rasformerirovatFoto --> sme")

    X, Y = x, y
    del x, y
    global zagotovkaNroslogo

    for x in range(X,VH):
        if x >VH:
            break
        for y in range(Y,VW):
            formul = ((x * VW) + y)
            if formul >= N:
                zagotovkaNroslogo = imgM
                return imgM, W, H
            imgM[formul] = img[x:64+x, y:64+y]
    zagotovkaNroslogo = imgM
    return imgM, W, H

def trasformerirovatFotoX(img,N,sme):
    img = nereobrazovatIMG64(img)
    return trasformerirovatFoto(img,N,sme)

def trasformerirovatFotoMAX(img):
    W, H = razmetca(img)
    VH = H - 64
    VW = W - 64
    return VH*VW

def trasformerirovatFotoXAUTOMAX(img,N):
    maX = trasformerirovatFotoMAX(img)
    Don = 0
    if maX % N != 0:
        Don = 1
    return (maX//N)+Don

def trasformerirovatFotoXAUTO(img,N,sme):
    trebovania = sme*N
    trebovaniaMax = sme * N+N
    max = trasformerirovatFotoXAUTOMAX(img, N)
    if trebovaniaMax <= max:
        return trasformerirovatFotoX(img,N,sme)
    elif trebovaniaMax > max and trebovania<max:
        if trebovania-max == 0:
            print("eror --> trasformerirovatFotoXAUTO --> trebovania")
            return zagotovkaNroslogo
        elif trebovania-max ==1:
            vrem = trasformerirovatFotoX(img, trebovania - max, sme)
            return np.array([vrem,vrem],dtype=np.uint8)
        else:
            return trasformerirovatFotoX(img, trebovania-max, sme)



def nereobrazovatIMG(img,W,H):       ###############################
    WR, HR = razmetca(img)
    if WR > W or HR > H :
        print("error --> nereobrazovatIMG --> H,W 'потеря даних'")
    img2 = np.zeros((H,W,3),dtype=np.uint8)
    for x in range(HR):
        for y in range(WR):
            if x < H and y < W:
                img2[x, y] = img[x, y]
    return img2

def nereobrazovatIMG64(img):
    W, H = razmetca(img)
    W += 64 - (W % 64)
    H += 64 - (H % 64)
    return nereobrazovatIMG(img, W, H)

def razdel64(img):
    img = nereobrazovatIMG64(img)
    W, H = razmetca(img)
    VW, VH = W//64, H//64
    imgM = np.zeros((VW*VH,64,64,3),dtype=np.uint8)
    for x in range(VH):
        for y in range(VW):
            formul = ((x * VW) + y)
            imgM[formul] = img[x*64:x*64+64, y*64:y*64+64]
    return imgM



