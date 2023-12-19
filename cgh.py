#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 14:59:25 2020

@author: harmonik
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image 
def rgb2gray(rgb):     #rgb2gray
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def normolizatioin(data):
    range_=np.max(data)-np.min(data)
    return(data-np.min(data))/range_

tStart = time.time()#計時開始

lam=532*10**-6                 #設定波長(綠光532紅光633藍光405
Nx=1920                    #水平像素
Ny=1080                    #垂直像素
PS=12.17/1920              #pixel size
z1=1000                    #圖1傳播多遠(mm)
k=2*np.pi/lam              #波數
Op1=lam*z1/PS              #Op=成像面實際投影出來影像大小
a=1/(2*PS)                 #shift零階參數
I=np.zeros((Ny,Nx))
obs=2                      #偏移

im = Image.open('./NCTU.png')  #讀取圖片
newsize = (Nx,Ny) 
im = im.resize(newsize) 
im=np.array(im)          #轉成1920*1080
im=rgb2gray(im)         #轉灰階,黑白圖片要跳過
im=255-im
im=normolizatioin(im)
pic1=(im)/255
pic1=np.fliplr(pic1)    #因為FT後座標會反轉，需將影垂直鏡像90度
pic2=np.fft.fftshift(pic1) #電腦FFT會將象限互換，故用fftshift修正象限
Upic1=np.sqrt(pic2)  

xi1=np.linspace(-(Op1/2)/Nx,(Op1/2)/Nx,Nx)    #實際顯示的成像的x座標陣列
yi1=np.linspace(-(Op1/2)/Ny,(Op1/2)/Ny,Ny)    #實際顯示的成像的y座標陣列
[Xi1,Yi1] = np.meshgrid(xi1,yi1);             #把xi,yi組合成矩陣，Xi直的重複，Yi橫的重複

x=np.linspace(-(Nx/2)*PS,(Nx/2)*PS,Nx)        #電腦全像片LCoS的x座標陣列
y=np.linspace(-(Ny/2)*PS,(Ny/2)*PS,Ny)        #電腦全像面LCoS的y座標
[X,Y] = np.meshgrid(x,y)                      #把x,y組合成矩陣，X直的重複，Y橫的重複
ri1=np.sqrt((Xi1)**2+(Yi1)**2)                #成像面1上影像
r=np.sqrt((X)**2+(Y)**2)                      #LCoS上影像

O1=(np.ones((1080,1920)))*np.exp(1j*np.pi*2*(np.random.rand(1080,1920)))    #初始限制振福1矩陣，以及亂數相位
A1=np.exp(1j*k*ri1**2/(2*z1))*np.exp(1j*k*z1)/(1j*lam*z1)     #近場繞射傅立葉轉換係數
h1=np.exp(1j*k*r**2/(2*z1))       #Fresnel繞射公式傅立葉轉換裡的項U2=A*FT{(U1*h1)}
B1=np.exp(-1j*k*r**2/(2*z1))*np.exp(-1j*k*z1)/(1j*lam*z1)      #近場繞射反傅立葉轉換係數
h11=np.exp(-1j*k*ri1**2/(2*z1))     #Fresnel繞射公式反傅立葉轉換裡的項U1=B*IFT{(U2*h2)}

#IFTA演算法參數
level=256
cir=30
RMS1 = []
for n in range(cir):
    
    O12=A1*(np.fft.fft2(O1*h1))
    O13=Upic1*np.exp(1j*np.angle(O12))
    O14=B1*np.fft.ifft2(O13*h11)
    CGHslevel1=np.round(((np.angle(O14)+np.pi)/(2*np.pi)*level),0)
    O1=np.exp(1j*CGHslevel1*(2*np.pi)/level)
    Urecon1=A1*np.fft.fft2(O1*h1)
    Irecon1=np.fft.fftshift(np.abs(Urecon1)**2)
    NorI_recon1=Irecon1/np.max(np.max(Irecon1))
    Norpic1=pic1/(np.max(np.max(pic1)))
    rms1=np.sqrt(np.mean(np.mean((NorI_recon1-Norpic1)**2)))
    print(n)
    print(rms1)
    RMS1 += [rms1]
    
plt.title("RMS")
x = np.arange(0, cir, 1)
plt.plot(x,RMS1)

print (RMS1)
plt.savefig('RMS.png')
plt.imshow(NorI_recon1,cmap='gray')    #觀看影像
obshift1=O14*np.exp(1j*0*np.pi*a*Y) #將相位偏移，移開零階
CGHslevelf=np.round(((np.angle(obshift1)+2*np.pi)/(2*np.pi)*level),0) 
CGHslevelf[CGHslevelf==level]=0 
#原本round取出的階數共有level+1階，此處將第level階改為0階，共256階
CGHfinal=CGHslevelf/level    
#將角度變成灰階影像，用LCoS顯示

image2=Image.fromarray(np.uint8(CGHfinal*255))    #
print(image2.size)
image2.show() #顯示
tn=time.strftime('%Y_%m_%d_%H_%M_%S')
image2.save("./save.png")  #存擋
tEnd = time.time()#計時結束

finaltime=tEnd-tStart
print("time=",finaltime)#顯示時間
