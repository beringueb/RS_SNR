import numpy as np 
import matplotlib.pyplot as plt
from noise_calculator import *

LMAX = 4500

primary_tt,primary_ee,primary_te = np.loadtxt('./data/primary.dat', usecols=(1,2,4), unpack=True)
cross_tt,cross_ee,cross_te = np.loadtxt('./data/cross_te_500.dat', usecols=(1,2,4), unpack=True)
cross_et = np.loadtxt('./data/cross_et_500.dat', usecols=(4), unpack=True)
auto_tt,auto_ee,auto_te = np.loadtxt('./data/auto_500.dat', usecols=(1,2,4), unpack=True)

## ##

def _covariance(Experiment,lmax = LMAX):
    """ Compute the noiseless frequency covariance matrix for a given experiment.
    """
    freqs = Experiment.freqs
    N = len(freqs)
    cov = np.zeros((lmax-1,2*N,2*N))
    for i in range(N):
        for j in range(N):
            cov[:,i,j] = (primary_tt + ((freqs[i]/500.)**4 + (freqs[j]/500.)**4)*cross_tt + (freqs[i]/500.)**4 * (freqs[j]/500.)**4 * auto_tt)[:lmax-1]
            cov[:,i+N,j+N] = (primary_ee + ((freqs[i]/500.)**4 + (freqs[j]/500.)**4)*cross_ee + (freqs[i]/500.)**4 * (freqs[j]/500.)**4 * auto_ee)[:lmax-1]
            cov[:,i,j+N] = (primary_te + (freqs[i]/500.)**4*cross_et + (freqs[j]/500.)**4*cross_te + (freqs[i]/500.)**4 * (freqs[j]/500.)**4 * auto_te)[:lmax-1]
            cov[:,i+N,j] = (primary_te + (freqs[i]/500.)**4*cross_te + (freqs[j]/500.)**4*cross_et + (freqs[i]/500.)**4 * (freqs[j]/500.)**4 * auto_te)[:lmax-1]
    return cov
    
def get_fisher(Experiment,lmax=LMAX):
    """ Compute the fisher matrix for the 4 Rayleigh cross spectra (TT,EE,TE,ET) for the noise levels
    defined in the experiment.
    """
    ell = np.linspace(2,lmax,lmax-1)
    freqs = Experiment.freqs
    N = len(freqs)
    Nl = Experiment.compute_noise(lmax)
    cov = _covariance(Experiment,lmax)
    inv_cov = np.linalg.solve(cov+Nl,np.broadcast_to(np.identity(2*N),(lmax-1,2*N,2*N)))
    deriv = np.zeros((4,2*N,2*N))
    for i in range(N):
        for j in range(N):
            deriv[0,i,j] = (freqs[i]/500.)**4 + (freqs[j]/500.)**4 #deriv tt
            deriv[1,i+N,j+N] = (freqs[i]/500.)**4 + (freqs[j]/500.)**4 # deriv ee
            deriv[2,i,j+N] = (freqs[j]/500.)**4 #deriv te
            deriv[2,j+N,i] = deriv[2,i,j+N].copy() #deriv te
            deriv[3,i,j+N] = (freqs[i]/500.)**4 #deriv et
            deriv[3,j+N,i] = deriv[3,i,j+N].copy() #deriv et
    fisher = Experiment.fsky*np.einsum('l,lij,ajk,lkm,bmi -> lab',(2.*ell+1.)/2.,inv_cov,deriv,inv_cov,deriv)
    return fisher

def get_SN2(fisher):
    """ Compute the SNR^2 per ell for the the full Rayleigh signal (ie. combining all the 4 cross spectra)"""
    lmax = int(fisher.shape[0])+1
    signals = np.array([cross_tt,cross_ee,cross_te,cross_et]).T
    SN2_ll = np.einsum('la,lab,lb ->l', signals[:lmax-1,:], fisher, signals[:lmax-1,:])
    print('Full S/N : {:3.2f}'.format(SN2_ll.sum()**.5))
    return SN2_ll
    
def plot_all(fisher,lmax):
    """ Plot the 4 cross spectra, their S/N per mode as well as the cumulative S/N."""
    error_cov = np.linalg.solve(fisher,np.broadcast_to(np.identity(4),(lmax-1,4,4)))
    SN_tt = cross_tt[:lmax-1] / error_cov[:,0,0]**.5
    SN_ee = cross_ee[:lmax-1] / error_cov[:,1,1]**.5
    SN_te = cross_te[:lmax-1] / error_cov[:,2,2]**.5
    SN_et = cross_et[:lmax-1] / error_cov[:,3,3]**.5
    SN2_full = get_SN2(fisher)
    labels = ['Primary T x RS T', 'Primary E x RS E','Primary T x RS E', 'Primary E x RS T']
    SN = np.array([SN_tt,SN_ee,SN_te,SN_et]).T
    signals = np.array([cross_tt,cross_ee,cross_te,cross_et]).T[:lmax-1,:]
    ell = np.linspace(2,lmax,lmax-1)
    for kk in range(4):
        f,[ax1,ax2] = plt.subplots(2,1,sharex = True,gridspec_kw={'height_ratios': [2, 1]},figsize=(10,6))
        plt.subplots_adjust(wspace=0, hspace=0,left=0.1,right=0.9,top = 0.9,bottom=0.1)
        ax1.plot(ell,signals[:,kk], c = 'k')
        ax_cp = ax1.twinx()
        ax_cp.plot(ell,SN[:,kk], c = 'r')
        ax_cp.tick_params(axis='y', labelcolor='r')
        ax_cp.set_ylabel(r'S/N per mode', fontsize = 14, color='r')
        ax1.set_ylabel(r'Cross spectra',fontsize = 14)
        ax2.plot(ell,(np.cumsum(SN[:,kk]**2))**.5, c = 'r')
        ax2.set_xlim(2,LMAX)
        ax2.set_ylabel(r'Cumulative S/N', fontsize = 14)
        ax2.set_xlabel(r'$\ell$', fontsize = 18)
        plt.suptitle('{:s} spectrum'.format(labels[kk]), fontsize = 18)
        ax2.text(LMAX-500,0,r'{:3.1f}$\sigma$'.format(((np.cumsum(SN[:,kk]**2))**.5)[-1]), fontsize = 16)   
    plt.show()


if __name__ == '__main__':
    fisher_CCAT_PLANCK = get_fisher(CCAT_PLANCK)
    plot_all(fisher_CCAT_PLANCK,LMAX) 
    
        
    