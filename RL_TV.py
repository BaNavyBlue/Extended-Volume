# RL deconvolution with TV regularization
# adapted from Matlab devconlucy function

from gettext import npgettext
import cupy
from cupy import fft
import numpy as np
from numpy import float32
import matplotlib.pyplot as plt

def RL_TV(im_extVol, otf, inner_iter, TV_reg, Nz, Ny, Nx):
    sizeI = im_extVol.size
    epsilon = cupy.finfo(cupy.float32).eps
    with cupy.cuda.Device(0):
        J1 = cupy.copy(im_extVol)
        J2 = cupy.copy(J1)
        J3 = 0
        #print('J1.shape: ' + str(J1.shape) + ' J2.shape: ' + str(J2.shape))

        J4 = np.asarray(np.zeros((2, sizeI), np.float32))
        #print('type(J4): ' + str(type(J4)) + ' J4.shape: ' + str(J4.shape) +' J2.size: ' + str(J4.size))

        wI = cupy.maximum(J1,0)
        print('wI.size ' + str(wI.size))

        lamb_duh = 0

        for k in range(inner_iter):
            if k > 2:
                print('J4[0,:].size: ' + str(J4[0,:].size) + ' J4[1,:].size: ' + str(J4[1,:].size))
                print('type(J4): ' + str(type(J4)))
                #with cupy.cuda.Device(0):
                
                numer = cupy.dot(J4[0,:], J4[1,:])
                #numer.replace(cupy.isnan, 0)
                denom = (cupy.dot(J4[1,:],J4[1,:]) + epsilon)
                #denom.replace(cupy.isnan, epsilon)
                lamb_duh = (numer)/(denom)
                lamb_duh = max(min(lamb_duh,1),0) #stability enforcement


            Y = cupy.maximum(J2 + lamb_duh*(J2 - J3), 0) # plus positivity constraint
            print('Y.size: ' + str(Y.size) + ' Y.shape: ' + str(Y.shape))
            print('J2.shape: ' +  str(J2.shape))

            # 3.b Make core for the LR estimation
            #with cupy.cuda.Device(0):
            ReBlurred = cupy.real(fft.ifftn(otf*fft.fftn(Y))).astype(cupy.float32)
            #print(ReBlurred)
            #print(epsilon)
            ReBlurred = cupy.maximum(ReBlurred, epsilon)
            #print('ReBlurred.size: ' + str(ReBlurred.size))

            ImRatio = wI/ReBlurred + epsilon
            #print('ImRatio.size: ' + str(ImRatio.size))

            Ratio = cupy.real(fft.ifftn(cupy.conj(otf)*fft.fftn(ImRatio)))
            if TV_reg != 0: # total variation regularization
                TV_term = computeTV(J2, TV_reg, Nz, Ny, Nx)
                Ratio = Ratio/TV_term
            plt.figure(k)
            plt.imshow(cupy.reshape(ReBlurred, (Nz, Ny, Nx), order='C')[19,:,:].get())
            del ImRatio
            del ReBlurred

            J3 = cupy.copy(J2)
            J2 = cupy.maximum(Y*Ratio, 0)
            #Jtemp = J2 - Y
            #print('J4[:,0].shape: ' + str(J4[:,0].shape) )
            #print('J4[0].shape: ' + str(J4[0].shape) )
            J4[1,:] = cupy.copy(J4[0,:])
            # J4[:,0] = cupy.copy(J2 - Y).get()
            J4[0,:] = cupy.copy(J2 - Y).get()

        return J2 


def computeTV(Image, TV_reg, Nz, Ny, Nx):
    epsilon = cupy.finfo(cupy.float32).eps
    gx = cupy.diff(cupy.reshape(Image, (Nz, Ny, Nx), order='C'), 1, 1)
    Oxp = cupy.pad(gx,((0,0),(0,0),(0,1)), 'constant', const_values = 0 )
    Oxn = cupy.pad(gx, ((0,0),(0,0),(1,0)), 'constant', constant_values = 0)
    mx = (cupy.sign(Oxp) + cupy.sign(Oxn))/2*cupy.minimum(Oxp, Oxn)
    mx = cupy.maximum(mx, epsilon)
    Dx = Oxp/cupy.sqrt(Oxp**2 + mx**2)
    DDx = cupy.diff(Dx, 1 ,1)
    DDx = cupy.pad(DDx, ((0,0),(0,0),(1,0)), 'constant', constant_values = 0)

    del gx
    del Oxp
    del Oxn
    del mx
    del Dx
    
    gy = cupy.diff(cupy.reshape(Image, (Nz, Ny, Nx), order='C'), 1, 2)
    Oyp = cupy.pad(gy, ((0,0),(0,1),(0,0)), 'constant', constant_values = 0)
    Oyn = cupy.pad(gy, ((0,0),(1,0),(0,0)), 'constant', constant_values = 0)
    my = (cupy.sign(Oyp) + cupy.sign(Oyn))/2*cupy.minimum(Oyp, Oyn)
    my = cupy.maximum(my, epsilon)
    Dy = Oyp/cupy.sqrt(Oyp**2 + my**2)
    DDy = cupy.diff(Dy, 1, 2)
    DDy = cupy.pad(DDy, ((0,0),(1,0), (0,0)), 'constant', constant_values = 0)

    del gy
    del Oyp
    del Oyn
    del my
    del Dy

    TV_term = 1 - (DDx + DDy)*TV_reg


    del DDx
    del DDy

    TV_term = cupy.maximum(TV_term, epsilon)
    print('Size TV_term: ' + str(TV_term.size) + ' Shape TV_term: ' + str(TV_term.shape))

    return cupy.reshape(TV_term, (TV_term.size,), order='C')
