## Compute Gaussian

import math
import numpy as np



def ComputeGaussianPSF(NumAp = 1.4, wavelength = 0.515, dx = 0.388, dz = 2, Nx = 608, Nz = 45, refIndex = 1.45):
    #function psf = ComputeGaussianPSF(NA,lambda,dx,dz,Nx,Nz,index)
    # calculate Gaussian-Lorentzian PSF
    z = np.arange(-Nz, Nz + 1, 1)*dz  # z distance, in um

    k = refIndex/wavelength
    delta_k = math.sqrt(2)*NumAp*k

    xi = (z*((math.pi*delta_k**2)/2/k))

    L = dx*Nx
    #x = (-L/2 + dx/2):dx:(L/2 - dx/2);
    x = np.arange((-L/2 + dx/2),(L/2 - dx/2) + dx, dx)

    #[xx,yy] = meshgrid(x, x);
    xx, yy = np.meshgrid(x,x)

    #rho = math.sqrt(xx.^2 + yy.^2)
    vec_sum = xx*xx + yy*yy
    rho = np.sqrt(vec_sum)


    # psf = zeros(Nx,Nx,length(z))
    psf = np.zeros((Nx,Nx,z.size))
    for i in range(z.size):
        psf[:,:,i] = np.pi*(delta_k**2)/(1+(xi[i]**2))*np.exp(-((np.pi*delta_k)**2*(rho*rho))/(1+(xi[i]**2)))
        psf[:,:,i] = psf[:,:,i]/np.sum(psf[:,:,i])



    psf = (psf/np.sum(psf))
    psf = np.swapaxes(psf,2,0)

    return psf

