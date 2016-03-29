import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel

# prepare filter bank kernels
kernels = []
for theta in range(2):
    #theta =0# theta / 4. * np.pi
    for sigma in (15, 15):
        for frequency in (0.005,0.015, 0.025,0.035,0.045, 0.055):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma-theta, sigma_y=sigma+theta))
            kernels.append(kernel)
K1 = kernels[0]
K1 = K1[48,4:-4]
K2 = kernels[2]
K2 = K2[48,4:-4]
K3 = kernels[4]
K3 = K3[48,4:-4]

K4 = kernels[14]
K4 = K4[48,:]
K5 = kernels[17]
K5 = K5[48,:]
K6 = kernels[18]
K6 = K6[48,:]

"""
K3 = kernels[11]
K3 = K3[48,:]
K2 = kernels[1]
K2 = K2[48,4:-4]
K1 = kernels[0]
K1 = K1[48,4:-4]
"""
plt.plot(K1)
dary='symm', mode='same')
imgplot = plt.imshow(np.multiply(K1.reshape(83,1),K3.reshape(1,83)))

