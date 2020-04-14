""" Module that computes noise covariance for different experiments.
"""
import numpy as np
import matplotlib.pyplot as plt


class Experiment:
    def __init__(self,freqs,fsky,N_white,beam,N_white_pol=None,N_red=None,lmax=5000):
        self.freqs = freqs
        self.noise = _compute_noise(freqs,N_white,beam,N_white_pol,N_red,lmax)
        self.fsky = fsky
    
    def __add__(self,other):
        freqs_tot = np.append(self.freqs,other.freqs)
        N = len(freqs_tot)
        f1 = self.fsky
        f2 = self.fsky
        if f1 > f2 :
            Nl_tot = _combine_noise(f2/f1*self.noise,other.noise)
            tot =  Experiment([0.],f2,[0.],[0.],[0.],[0.])
            tot.freqs = freqs_tot
            tot.noise = Nl_tot
        else :
            Nl_tot = _combine_noise(self.noise,f1/f2*other.noise)
            tot =  Experiment([0.],f1,[0.],[0.],[0.],[0.])
            tot.freqs = freqs_tot
            tot.noise = Nl_tot
        return tot


        return Experiment()
def _combine_noise(Nl1,Nl2):
    N1 = int(Nl1.shape[1]/2.)
    N2 = int(Nl2.shape[1]/2.)
    lmax = int(Nl1.shape[0]) + 1
    N = N1+N2
    Nl_tot = np.zeros((lmax-1,2*N,2*N))
    Nl_tot[:,:N1,:N1] = Nl1[:,:N1,:N1] 
    Nl_tot[:,N1:N,N1:N] = Nl2[:,:N2,:N2] 
    Nl_tot[:,N:N1+N,N:N+N1] = Nl1[:,N1:,N1:]
    Nl_tot[:,N1+N:,N1+N:] = Nl2[:,N2:,N2:]
    return Nl_tot

        
def _compute_noise(freqs,N_white,beam,N_white_pol=None,N_red=None,lmax=5000):
    N = len(freqs)
    noise = np.zeros((lmax-1,2*N,2*N))
    ell = np.linspace(2,lmax,lmax-1)
    l_knee = 1000
    alpha_knee = -3.5
    for i in range(N):
        if N_red is None:
            noise[:,i,i] = ell*(ell+1.)/2./np.pi * N_white[i] * np.exp(ell*(ell+1)*(beam[i]*np.pi/180/60/(8.*np.log(2))**0.5)**2)
        else : 
            noise[:,i,i] = ell*(ell+1.)/2./np.pi * (N_red[i]*(ell/l_knee)**alpha_knee + N_white[i])*np.exp(ell*(ell+1)*(beam[i]*np.pi/180/60/(8.*np.log(2))**0.5)**2)
    l_knee = 700
    alpha_knee = -1.4
    if N_white_pol is None :
        N_white_pol = 2.*N_white
    for i in range(N):
        if N_red is None:
            noise[:,i+N,i+N] = ell*(ell+1.)/2./np.pi * N_white_pol[i] * np.exp(ell*(ell+1)*(beam[i]*np.pi/180/60/(8.*np.log(2))**0.5)**2)
        else :
            noise[:,i+N,i+N] = ell*(ell+1.)/2./np.pi * (N_white_pol[i]*(ell/l_knee)**alpha_knee + N_white_pol[i])*np.exp(ell*(ell+1)*(beam[i]*np.pi/180/60/(8.*np.log(2))**0.5)**2)
    return noise

def compute_Nwhite_from_NET(NET,fsky,Ndet,DT_s,Y=1.):
    """ Using sensitivitiy calculaiton from arxiv:1402.4108 (Eq.1)"""
    fsky_arcmin = fsky*(4*180*180/np.pi)*60**2    
    s = NET*(fsky_arcmin/Ndet/Y/DT_s)**.5
    return (s*np.pi/180/60)**2
    
def compute_Nwhite_from_sensi(sensi):
    return (sensi*np.pi/180/60)**2


if __name__ == '__main__':
    print()



  


        
    