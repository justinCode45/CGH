import cv2 # pip install opencv-python
import numpy as np # installed with cv2
import time
# import matplotlib as plt
# import screeninfo # pip install screeninfo

# Setup the display window
fileInputName = "kane.JPG" 
fileOutputName = "hologram_kane.bmp" 
fileOutputNameFinal = "final.bmp" 

start_time = time.time()
lam = 532*10**-6 # wavelength of laser
Nx = 1920 # number of pixels in x direction
Ny = 1080 # number of pixels in y direction
Ps = 12.17/1920 # pixel size
z1 = 200 # distance from SLM to object
k = 2*np.pi/lam # wave number
Op1 = lam*z1/Ps # size of object image
a = 1/(2*Ps) #shift

# Read in target image
pic1=cv2.imread ( fileInputName,cv2.IMREAD_GRAYSCALE).astype(np.float64)
print(pic1.shape)
pic1 = cv2.resize(pic1, (Nx, Ny), interpolation = cv2.INTER_AREA)


picl = pic1 + 2*np.pi
pic11 = np.fft.fftshift(pic1)
U_pic1 = np.sqrt(pic11)

xi1 = np.linspace(-(Op1/2)/Nx,(Op1/2)/Nx,Nx)
yi1 = np.linspace(-(Op1/2)/Ny,(Op1/2)/Ny,Ny)
[Xi1,Yi1] = np.meshgrid(xi1,yi1)
x = np.linspace(-(Nx/2)*Ps,(Nx/2)*Ps,Nx)
y = np.linspace(-(Ny/2)*Ps,(Ny/2)*Ps,Ny)
[X,Y] = np.meshgrid(x,y)
ri1 = np.sqrt(Xi1**2+Yi1**2)
r = np.sqrt(X**2+Y**2)

O1 = np.ones((Ny,Nx))*np.exp(1j*2*np.pi*np.random.rand(Ny,Nx))
A1 = np.exp(1j*k*ri1**2/(2*z1))*np.exp(1j*k*z1)/(1j*lam*z1)
h1 = np.exp(1j*k*r**2/(2*z1))
B1 = np.exp(-1j*k*r**2/(2*z1))*np.exp(-1j*k*z1)/(1j*lam*z1)
h11 = np.exp(-1j*k*ri1**2/(2*z1))

level = 256
cir = 30
RMSe = []

for N in range(0,cir):
    O12 = A1*np.fft.fft2(O1*h1)
    O13 = U_pic1*np.exp(1j*np.angle(O12))
    O14 = B1*np.fft.ifft2(O13*h11)
    CGHslevel1 = np.round((np.angle(O14)+np.pi)/(2*np.pi)*level)
    O1 = np.exp(1j*CGHslevel1*2*np.pi/level)

    U_recon1 = A1*np.fft.fft2(O1*h1)
    I_recon1 = np.fft.fftshift(np.abs(U_recon1)**2)
    Norl_recon1 = I_recon1/np.max(np.max(I_recon1))
    Norpic1 = pic1/np.max(np.max(pic1))
    RMS1 = np.sqrt(np.mean(np.mean((Norl_recon1-Norpic1)**2)))
    # print(N)
    # print(RMS1)
    RMSe.append(RMS1)

# fig = plt.figure()
# fig.suptitle('RMS Error vs Iteration')
# plt.plot(RMSe)
# plt.xlabel('Iteration')
# plt.ylabel('RMS Error')
# plt.savefig('RMS Error vs Iteration.png')

Norl_recon1 = cv2.resize(Norl_recon1, (1080,1080), interpolation = cv2.INTER_AREA)
cv2.imwrite(fileOutputName, Norl_recon1)


obshift = O14*np.exp(1j*a*Y*2*np.pi)
CGHslevelf = np.round((np.angle(obshift)+np.pi)/(2*np.pi)*level)

# for i in CGHslevelf:
#     for j in i:
#         if j == 256:
#             j = 0

CGHfinal = CGHslevelf/level
cv2.imwrite(fileOutputNameFinal, CGHfinal)
print("--- %s seconds ---" % (time.time() - start_time))
