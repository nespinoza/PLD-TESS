import os
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pyfits

d = pyfits.getdata('tess2019169103026-s0013-0000000280097543-0146-s_tp.fits')
ntimes = len(d)
npixels = 6
if os.path.exists('mi.npy'):
    master_image = np.load('mi.npy')
else:
    # First go through all the TPF of the time-series to generate the median image, which will generate our TPF mask:
    firstime = True
    for i in range(ntimes):
        if firstime:
            ss = d[i][3]
            firstime = False
        else:
            ss = np.dstack((ss,d[i][3]))
    master_image = np.median(ss,axis=2)
    np.save('mi.npy',master_image)

if os.path.exists('mask'+str(npixels)+'.npy'):
    mask = np.load('mask'+str(npixels)+'.npy')
    mask_idxs = np.load('mask_idxs_'+str(npixels)+'.npy')
    mask_idys = np.load('mask_idys_'+str(npixels)+'.npy')
else:
    # Generate the TPF mask; an array of zeros and ones. Ones are values that will be added to the aperture photometry. For this, 
    # first identify the values of the npixels brightest pixels in the master/median image:
    cpixels = -1
    flat_mi = master_image.flatten()
    values = np.array([])
    while cpixels < npixels:
        max_val = np.max(flat_mi)
        idx = np.where(flat_mi == max_val)[0]
        values = np.append(values, flat_mi[idx])
        flat_mi = np.delete(flat_mi,idx)
        cpixels += len(idx)
    # With these values, now fill the mask:
    mask = np.zeros(master_image.shape)
    mask_idxs = np.array([])
    mask_idys = np.array([])
    for i in range(len(values)):
        idx = np.where(master_image == values[i])
        mask_idxs = np.append(mask_idxs,idx[0][0])
        mask_idys = np.append(mask_idys,idx[1][0])
        mask[idx] = 1.
    np.save('mask_idxs_'+str(npixels)+'.npy',mask_idxs)
    np.save('mask_idys_'+str(npixels)+'.npy',mask_idys)
    np.save('mask_'+str(npixels)+'.npy',mask)

bkg_flat_idx= np.where(mask.flatten() == 0.)
times, fluxes, flux_pixels = np.array([]), np.array([]), np.array([])
fout = open('lc.dat','w')
fout2 = open('tpf_lc.dat','w')
for i in range(ntimes):
    ct, ctpf = d[i][0]+2457000., d[i][3]
    if ~np.isnan(ct):
        ctpf = ctpf - np.median(ctpf.flatten()[bkg_flat_idx])
        fout.write('{0:.10f} {1:.10f}\n'.format(ct,np.sum(ctpf*mask)))
        s = '{0:.10f}'.format(ct)
        for j in range(len(mask_idxs)):
            s = s + ' {0:.10f}'.format(ctpf[int(mask_idxs[j]),int(mask_idys[j])])
        fout2.write(s+'\n')
fout.close()
fout2.close()
