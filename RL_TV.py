# RL deconvolution with TV regularization
# adapted from Matlab devconlucy function

from gettext import npgettext
import random
import cupy
from cupy import fft
import numpy as np
from numpy import float32
import matplotlib.pyplot as plt
import math

def RL_TV(im_extVol, otf, inner_iter, TV_reg, Nz, Ny, Nx, cp_stream):
    sizeI = im_extVol.size
    epsilon = cupy.finfo(cupy.float32).eps
    with cp_stream:
        J1 = cupy.copy(im_extVol)

        J2 = cupy.copy(J1)
        J3 = 0
        #print('J1.shape: ' + str(J1.shape) + ' J2.shape: ' + str(J2.shape))

        J4 = cupy.zeros((2, sizeI), dtype=cupy.float32)
        #print('type(J4): ' + str(type(J4)) + ' J4.shape: ' + str(J4.shape) +' J2.size: ' + str(J4.size))
 
        wI = cupy.maximum(J1,0)
        # print('wI.size ' + str(wI.size))

        

        lamb_duh = 0

        for k in range(inner_iter):
            if k > 1:
                lamb_duh = (cupy.sum(J4[0,:].conj()*J4[1,:]))/((cupy.sum(J4[1,:].conj()*J4[1,:]) + epsilon))
                lamb_duh = max(min(lamb_duh, 1),0)
                
            # print('lamb_duh:' + str(lamb_duh))

            Y = cupy.maximum(J2 + lamb_duh*(J2 - J3), 0) # plus positivity constraint

            # 3.b Make core for the LR estimation
            #with cupy.cuda.Device(0):
            ReBlurred = cupy.real(fft.ifftn(otf*fft.fftn(Y))).astype(cupy.float32)

            ReBlurred = cupy.maximum(ReBlurred, epsilon)

            ImRatio = wI/ReBlurred + epsilon
            #print('ImRatio.size: ' + str(ImRatio.size))

            Ratio = cupy.real(fft.ifftn(cupy.conj(otf)*fft.fftn(ImRatio))).astype(cupy.float32)

            if TV_reg != 0: # total variation regularization
                TV_term = computeTV(J2, TV_reg, Nz, Ny, Nx, cp_stream)
                Ratio = Ratio/TV_term

            del ImRatio
            del ReBlurred



            J3 = cupy.copy(J2)
            J2 = cupy.maximum(Y*Ratio, 0)

            J4[1,:] = cupy.copy(J4[0,:])

            J4[0,:] = cupy.copy(cupy.reshape((J2 - Y),(J4[0].size,), order='C'))

        return J2

def computeTV(Image, TV_reg, Nz, Ny, Nx, cp_stream):
    epsilon = cupy.finfo(cupy.float32).eps
    with cp_stream:
        gx = cupy.diff(Image, 1, 2)
        Oxp = cupy.pad(gx,((0,0),(0,0),(0,1)), mode='constant', constant_values = 0 )
        Oxn = cupy.pad(gx, ((0,0),(0,0),(1,0)), mode='constant', constant_values = 0)
        mx = (cupy.sign(Oxp) + cupy.sign(Oxn))/2*cupy.minimum(Oxp, Oxn)
        mx = cupy.maximum(mx, epsilon)
        Dx = Oxp/cupy.sqrt(Oxp**2 + mx**2)
        DDx = cupy.diff(Dx, 1 ,2)
        DDx = cupy.pad(DDx, ((0,0),(0,0),(1,0)), mode='constant', constant_values = 0)

        del gx
        del Oxp
        del Oxn
        del mx
        del Dx
        
        gy = cupy.diff(Image, 1, 1)
        Oyp = cupy.pad(gy, ((0,0),(0,1),(0,0)), mode='constant', constant_values = 0)
        Oyn = cupy.pad(gy, ((0,0),(1,0),(0,0)), mode='constant', constant_values = 0)
        my = (cupy.sign(Oyp) + cupy.sign(Oyn))/2*cupy.minimum(Oyp, Oyn)
        my = cupy.maximum(my, epsilon)
        Dy = Oyp/cupy.sqrt(Oyp**2 + my**2)
        DDy = cupy.diff(Dy, 1, 1)
        DDy = cupy.pad(DDy, ((0,0),(1,0), (0,0)), mode='constant', constant_values = 0)

        del gy
        del Oyp
        del Oyn
        del my
        del Dy

        TV_term = 1 - (DDx + DDy)*TV_reg


        del DDx
        del DDy

        TV_term = cupy.maximum(TV_term, epsilon)

    return TV_term
