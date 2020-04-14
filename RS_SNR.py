import numpy as np 
import matplotlib.pyplot as plt
from noise_calculator import *


## CCAT FIDUCIAL ##
freqs_CCAT = np.array([220.,280.,350.,410.])
Ndet_CCAT_fid = np.array([8e3,1e4,2.1e4,2.1e4]) #Fiducial number of detectors (numbers from Choi et al (arxiv:1908.10451))
NET_det_CCAT = np.array([7.6,14,54,192])*np.sqrt(Ndet_CCAT_fid) # NET per detector (numbers from Choi et al (arxiv:1908.10451))
DT_CCAT = 4000.*3600. #Observing time in seconds (numbers from Choi et al (arxiv:1908.10451))
N_red_CCAT = np.array([1.6e-2,1.1e-1,2.7,1.7e1])
beam_CCAT = np.array([57.,45.,35.,30.])/60.
fsky_CCAT_fid = 15000/(4*180*180/np.pi)

## ##
freqs_PLANCK = np.array([30.,44.,70.,100.,143.,217.,353.])
sensi_PLANCK = np.array([145.,149.,137.,65.,43.,66.,200.])
sensi_PLANCK_pol = np.array([1E25,1E25,450.,103.,81.,134.,406.])
beam_PLANCK = np.array([33.,23.,14.,10.,7.,5.,5.])
fsky_PLANCK = .7
N_white_PLANCK = compute_Nwhite_from_sensi(sensi_PLANCK)
N_white_PLANCK_pol = compute_Nwhite_from_sensi(sensi_PLANCK_pol)
## ##

## ##
LMAX = 4500
CCAT_fiducial = Experiment(freqs_CCAT,fsky_CCAT_fid,N_white=compute_Nwhite_from_NET(NET_det_CCAT,fsky_CCAT_fid,Ndet_CCAT_fid,DT_CCAT),beam=beam_CCAT,N_red=N_red_CCAT,lmax=LMAX)
PLANCK = Experiment(freqs_PLANCK,fsky_PLANCK,N_white=N_white_PLANCK,beam=beam_PLANCK,N_white_pol=N_white_PLANCK_pol,lmax=LMAX)

TOT = PLANCK + CCAT_fiducial

primary_tt,primary_ee,primary_te = np.loadtxt('./data/primary.dat', usecols=(1,2,4), unpack=True)
cross_tt,cross_ee,cross_te = np.loadtxt('./data/cross_te_500.dat', usecols=(1,2,4), unpack=True)
cross_et = np.loadtxt('./data/cross_et_500.dat', usecols=(4), unpack=True)
auto_tt,auto_ee,auto_te = np.loadtxt('./data/auto_500.dat', usecols=(1,2,4), unpack=True)

## ##

def _covariance(Experiment,lmax = LMAX):
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
    ell = np.linspace(2,lmax,lmax-1)
    freqs = Experiment.freqs
    N = len(freqs)
    Nl = Experiment.noise
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
    lmax = int(fisher.shape[0])+1
    signals = np.array([cross_tt,cross_ee,cross_te,cross_et]).T
    SN2_ll = np.einsum('la,lab,lb ->l', signals[:lmax-1,:], fisher, signals[:lmax-1,:])
    return SN2_ll
    
def plot_all(Experiment,lmax):
    fisher = get_fisher(Experiment,lmax)
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
    print((np.cumsum(SN2_full)**.5)[-1])
    plt.show()


if __name__ == '__main__':
    plot_all(TOT,LMAX) 
    
        
    