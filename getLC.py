import matplotlib.pyplot as plt
import numpy as np

from astroquery.mast import Observations

def get_mad(x):
    m = np.median(x)
    mad = np.median(np.abs(x-np.median(x)))
    return 1.4826*mad

def getTPF(object_name, radius = ".02 deg", P = None, t0 = None, tdur = None):
    """
    Given an object name, a period, a transit-time and a transit duration for an exoplanet, this function gets you a PLD corrected 
    lightcurve. PLD coefficients are trained on the out-of-transit data.
    """
    obs_table = Observations.query_object(object_name,radius=radius)
    for i in range(len(obs_table['dataURL'])):

        print(obs_table['dataURL'][i])

def get_phases(t,P,t0):
    """ 
    Given input times, a period (or posterior dist of periods)
    and time of transit center (or posterior), returns the 
    phase at each time t.
    """
    if type(t) is not float:
        phase = ((t - np.median(t0))/np.median(P)) % 1
        ii = np.where(phase>=0.5)[0]
        phase[ii] = phase[ii]-1.0
    else:   
        phase = ((t - np.median(t0))/np.median(P)) % 1
        if phase>=0.5:
            phase = phase - 1.0
    return phase

Ps = [6.61, 9.4544820459]
t0s = [2458659.23114,2458329.2979906704]
tdurs = [10./24.,10./24.]

data = np.genfromtxt('tpf_lc.dat')#np.genfromtxt('TIC280097543_02_pixel-fluxes.dat')
times = data[:,0]
fluxes = data[:,1:]
added_flux = np.sum(fluxes,axis=1)

for i in range(fluxes.shape[0]):
    fluxes[i,:] = fluxes[i,:]/np.sum(fluxes[i,:])

fluxes = np.vstack((fluxes.T,np.ones(fluxes.shape[0]))).T
new_times, new_fluxes, new_added_flux = np.copy(times), np.copy(fluxes), \
                                        np.copy(added_flux)

for P,t0,tdur in zip(Ps,t0s,tdurs):
    phases = get_phases(new_times,P,t0)
    idx = np.where(np.abs(phases)*P>tdur)[0]
    print(len(idx))
    new_times = new_times[idx]
    new_fluxes = new_fluxes[idx,:]
    new_added_flux = new_added_flux[idx]
#plt.plot(new_times,new_added_flux,'o')
#plt.plot(times,added_flux,'-')
#plt.show()
treshold = False
from scipy.signal import medfilt
print(new_fluxes.shape)
while not treshold:
    result = np.linalg.lstsq(new_fluxes,new_added_flux)
    print(new_fluxes.shape,new_added_flux.shape)
    coeffs = result[0]
    print(coeffs)
    prediction = np.dot(coeffs,new_fluxes.T)
    residuals = new_added_flux - prediction
    mf = medfilt(residuals,41)
    residuals = residuals-mf
    sigma = get_mad(residuals)
    idx_out = np.where(np.abs(residuals)>5*sigma)[0]
    if len(idx_out) == 0:
        break
    else:
        idx_in = np.where(np.abs(residuals)<=5*sigma)[0]
        new_fluxes, new_added_flux = new_fluxes[idx_in,:], new_added_flux[idx_in]

prediction = np.dot(coeffs,fluxes.T)
plt.plot(times,added_flux)
plt.plot(times,prediction,alpha=0.5)
plt.show()

t,f,ferr = np.loadtxt('lctess13.dat',unpack=True,usecols=(0,1,2))
plt.errorbar(t,f,ferr,fmt='.')
plt.plot(times,added_flux/prediction,'o')
#plt.plot(times[idx],added_flux[idx]/prediction[idx],'o')
plt.show()

import juliet
f,ferr = (f-1.)*1e6,ferr*1e6
for P,t0,tdur in zip(Ps,t0s,tdurs):
    phases = get_phases(times,P,t0)
    pdc_phases = get_phases(t,P,t0)
    corrected_flux = added_flux/prediction
    corrected_flux = ((corrected_flux/np.median(corrected_flux)) -1.)*1e6
    idx = np.argsort(phases)
    pbin,fbin,fbinerr = juliet.utils.bin_data(phases[idx],corrected_flux[idx],15)
    idx = np.argsort(pdc_phases)
    pbinpdc,fbinpdc,fbinerrpdc = juliet.utils.bin_data(pdc_phases[idx],f[idx],15)
    #plt.errorbar(pdc_phases,f+500,ferr,fmt='.',label='PDC')
    #plt.plot(phases,corrected_flux,'.',label=str(P))
    plt.title(str(P))
    plt.errorbar(pbin,fbin,fbinerr,fmt='o',label='pld')
    plt.errorbar(pbinpdc,fbinpdc,fbinerrpdc,fmt='o',label='pdc')
    plt.legend()
    plt.show()
"""
for i in range(fluxes.shape[1]):
    plt.plot(times,fluxes[:,i]/np.median(fluxes[:,i]))
plt.show()
"""
