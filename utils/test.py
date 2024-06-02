import cv2
import numpy as np
import math


def avgGradient(image):
    width = image.shape[1]
    width = width - 1
    heigt = image.shape[0]
    heigt = heigt - 1
    tmp = 0.0

    for i in range(width):
        for j in range(heigt):
            dx = float(image[i,j+1])-float(image[i,j])
            dy = float(image[i+1,j])-float(image[i,j])
            ds = math.sqrt((dx*dx+dy*dy)/2)
            tmp += ds
    
    imageAG = tmp/(width*heigt)
    return imageAG


#计算空间频率
#根据公式推断出来的，不一定正确
def spatialF(image):
	M = image.shape[0]
	N = image.shape[1]
	
	cf = 0
	rf = 0


	for i in range(1,M-1):
		for j in range(1,N-1):
			dx = float(image[i,j-1])-float(image[i,j])
			rf += dx**2
			dy = float(image[i-1,j])-float(image[i,j])
			cf += dy**2

	RF = math.sqrt(rf/(M*N))
	CF = math.sqrt(cf/(M*N))
	SF = math.sqrt(RF**2+CF**2)

	return SF


def getMI(im1,im2):

    #im1 = im1.astype('float')
    #im2 = im2.astype('float')

    hang, lie = im1.shape
    count = hang*lie
    N = 256
    im1 = (im1*255).astype(np.uint8)
    im2 = (im2*255).astype(np.uint8)
    h = np.zeros((N,N))

    for i in range(hang):
        for j in range(lie):
            h[im1[i,j],im2[i,j]] = h[im1[i,j],im2[i,j]]+1

    h = h/np.sum(h)

    im1_marg = np.sum(h,axis=0)
    im2_marg = np.sum(h, axis=1)

    H_x = 0
    H_y = 0

    for i in range(N):
        if(im1_marg[i]!=0):
            H_x = H_x + im1_marg[i]*math.log2(im1_marg[i])

    for i in range(N):
        if(im2_marg[i]!=0):
            H_x = H_x + im2_marg[i]*math.log2(im2_marg[i])

    H_xy = 0

    for i in range(N):
        for j in range(N):
            if(h[i,j]!=0):
                H_xy = H_xy + h[i,j]*math.log2(h[i,j])

    MI = H_xy-H_x-H_y

    return MI

#图像信息熵也是图像一维熵
def getEn(img):
    img =(img*255).astype(np.uint8)
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if(tmp[i]==0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return(res)