""" Module that computes noise covariance for different experiments.
"""
import numpy as np
import matplotlib.pyplot as plt


class Experiment:
    """ Definition of a CMB experiment."""
    def __init__(self,freqs,fsky,N_white,beam,N_white_pol=None,N_red=None):
        self.freqs = freqs
        self.fsky = fsky
        self.N_white = N_white
        self.beam = beam
        if N_white_pol is None : 
            self.N_white_pol = 2.*N_white
        else :
            self.N_white_pol = N_white_pol
        if N_red is None :
            self.N_red = np.zeros(len(freqs))
        else : 
            self.N_red = N_red
    
    def __add__(self,other):
        freqs_tot = np.append(self.freqs,other.freqs)
        N_white_tot = np.append(self.N_white,other.N_white)
        N_white_pol_tot = np.append(self.N_white_pol,other.N_white_pol)
        beam_tot = np.append(self.beam,other.beam)
        N_red_tot = np.append(self.N_white,other.N_white)
        fsky_tot = min(self.fsky,other.fsky)
        if self.fsky != other.fsky :
            print("Warning : the two experiments you are combing don't have the same fsky. \n \
            This will return a combined expeiment on the SMALLEST patch. \n \
            You should add the fisher for the experiment with the largest footprint.")
        return Experiment(freqs_tot,fsky_tot,N_white_tot,beam_tot,N_white_pol_tot,N_red_tot)
        
    def compute_noise(self,lmax):
        return _compute_noise(self.freqs,self.N_white,self.beam,self.N_white_pol,self.N_red,lmax)
        
    
def _compute_noise(freqs,N_white,beam,N_white_pol,N_red,lmax):
    """ Compute the noise covariance matrix for a given experiment."""
    N = len(freqs)
    noise = np.zeros((lmax-1,2*N,2*N))
    ell = np.linspace(2,lmax,lmax-1)
    l_knee_T = 1000
    alpha_knee_T = -3.5
    l_knee_P = 700
    alpha_knee_P = -1.4
    for i in range(N):
        noise[:,i,i] = ell*(ell+1.)/2./np.pi * (N_red[i]*(ell/l_knee_T)**alpha_knee_T + N_white[i])*np.exp(ell*(ell+1)*(beam[i]*np.pi/180/60/(8.*np.log(2))**0.5)**2)
        if N_red[i] != 0. :
            N_red_pol = N_white_pol[i]
        else :
            N_red_pol = 0.
        noise[:,i+N,i+N] = ell*(ell+1.)/2./np.pi * (N_red_pol*(ell/l_knee_P)**alpha_knee_P + N_white_pol[i])*np.exp(ell*(ell+1)*(beam[i]*np.pi/180/60/(8.*np.log(2))**0.5)**2)
    return noise


def compute_Nwhite_from_NET(NET,fsky,Ndet,DT_s,Y=1.):
    """ Using sensitivitiy calculation from arxiv:1402.4108 (Eq.1)"""
    fsky_arcmin = fsky*(4*180*180/np.pi)*60**2    
    s = NET*(fsky_arcmin/Ndet/Y/DT_s)**.5
    return (s*np.pi/180/60)**2
    
def compute_Nwhite_from_sensi(sensi):
    """ Compute the white noise levels from detector sensitivities."""
    return (sensi*np.pi/180/60)**2
    
    
## CCAT FIDUCIAL ##
freqs_CCAT = np.array([220.,280.,350.,410.])
Ndet_CCAT_fid = np.array([8e3,1e4,2.1e4,2.1e4]) #Fiducial number of detectors (numbers from Choi et al (arxiv:1908.10451))
NET_det_CCAT = np.array([7.6,14,54,192])*np.sqrt(Ndet_CCAT_fid) # NET per detector (numbers from Choi et al (arxiv:1908.10451))
DT_CCAT = 4000.*3600. #Observing time in seconds (numbers from Choi et al (arxiv:1908.10451))
N_red_CCAT = np.array([1.6e-2,1.1e-1,2.7,1.7e1])
beam_CCAT = np.array([57.,45.,35.,30.])/60.
fsky_CCAT_fid = 15000/(4*180*180/np.pi)

N_white_CCAT = compute_Nwhite_from_NET(NET_det_CCAT,fsky_CCAT_fid,Ndet_CCAT_fid,DT_CCAT)
CCAT = Experiment(freqs_CCAT,fsky_CCAT_fid,N_white_CCAT,beam_CCAT,N_red_CCAT)


## PLANCK ##
freqs_PLANCK = np.array([30.,44.,70.,100.,143.,217.,353.])
sensi_PLANCK = np.array([145.,149.,137.,65.,43.,66.,200.])
sensi_PLANCK_pol = np.array([1E25,1E25,450.,103.,81.,134.,406.])
beam_PLANCK = np.array([33.,23.,14.,10.,7.,5.,5.])
fsky_PLANCK = .7

N_white_PLANCK = compute_Nwhite_from_sensi(sensi_PLANCK)
N_white_PLANCK_pol = compute_Nwhite_from_sensi(sensi_PLANCK_pol)
PLANCK = Experiment(freqs_PLANCK,fsky_PLANCK,N_white_PLANCK,beam_PLANCK,N_white_PLANCK_pol)


## ##
CCAT_PLANCK = PLANCK + CCAT



if __name__ == '__main__':
    print(CCAT_PLANCK.freqs)



  


        
    