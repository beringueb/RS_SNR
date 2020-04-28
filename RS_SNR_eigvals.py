import numpy as np 
import matplotlib.pyplot as plt
from noise_calculator import *
from scipy.stats import norm
from scipy import optimize
from scipy.special import erfinv

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
    
def _covariance_primary(Experiment,lmax = LMAX):
    """ Compute the noiseless frequency covariance matrix for a given experiment.
    """
    freqs = Experiment.freqs
    N = len(freqs)
    cov_primary = np.zeros((lmax-1,2*N,2*N))
    for i in range(N):
        for j in range(N):
            cov_primary[:,i,j] = primary_tt[:lmax-1]
            cov_primary[:,i+N,j+N] = primary_ee[:lmax-1]
            cov_primary[:,i,j+N] = primary_te[:lmax-1]
            cov_primary[:,i+N,j] = primary_te[:lmax-1]
    return cov_primary
    
def _gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 2**.5 / stddev)**2)
    
def normalize_eigvect(eigvect):
    """Normalize the eigenvectors, i.e. make sure all the entries are positive"""
    norm = np.ones(eigvect.shape)
    lmax = int(eigvect.shape[0])
    N = eigvect.shape[1]
    for ll in range(lmax-1):
        for i in range(N):
            if np.mean(eigvect[ll,:,i])<0 :
                norm[ll,:,i] = - 1.
    return eigvect.copy()/norm
    
    
def save_eigvals_eigvects(Experiment, lmax, file, normalize = True):
    """ Save the two largest eigenvalues and eigenvector entries corresponding to the 
    largest eigenvalue to 'file'. Stored as 'ell*(ell+1)/2pi * lambda'. """
    cov = _covariance(Experiment, lmax)
    N = len(Experiment.freqs)
    eigvals_T, eigvects_T = np.linalg.eigh(cov[:,:N,:N])
    eigvals_E, eigvects_E = np.linalg.eigh(cov[:,N:,N:])
    
    eigvals_T_save = eigvals_T[:,(-1,-2)]
    eigvals_E_save = eigvals_E[:,(-1,-2)]
    if normalize :
        eigvects_T_save = normalize_eigvect(eigvects_T)[:,:,-1]
        eigvects_E_save = normalize_eigvect(eigvects_E)[:,:,-1]
    else : 
        eigvects_T_save = eigvects_T[:,:,-1]
        eigvects_E_save = eigvects_E[:,:,-1]
    np.savetxt(file + 'eigvals_T', eigvals_T_save, header = 'Temperature : lambda_+, lambda_-')
    np.savetxt(file + 'eigvals_E', eigvals_E_save, header = 'E-mode : lambda_+, lambda_-')
    np.savetxt(file + 'eigvects_T', eigvects_T_save, header = 'Temperature, entries to largest eigvest @ {}'.format(Experiment.freqs))
    np.savetxt(file + 'eigvects_E', eigvects_E_save, header = 'E-mode, entries to largest eigvest @ {}'.format(Experiment.freqs))

    
def estimate_noise_eigvals(cov,Nl):
    """" Estimation of the noise on the different eigenvalues, using an ILC 
    (cf Hirata's paper) """
    N = int(cov.shape[1]/2)
    lmax = int(cov.shape[0])+1
    
    eigvals_T, eigvects_T = np.linalg.eigh(cov[:,:N,:N])
    eigvals_E, eigvects_E = np.linalg.eigh(cov[:,N:,N:])

    A_full = np.zeros((lmax-1,2*N,2*N))
    A_full[:,:N,:N] = eigvects_T
    A_full[:,N:,N:] = eigvects_E

    cov_diag_full = np.einsum('lji,ljk,lkm -> lim',A_full,cov,A_full)
    ix = np.ix_([N-1,N-2,-1,-2],[N-1,N-2,-1,-2])
    A = A_full[:,:,[N-1,N-2,-2,-1]]
    N_inv = np.linalg.inv(Nl)
    Nl_best = np.linalg.solve(np.einsum('lia,lij,ljb->lab',A,N_inv,A),np.repeat(np.identity(4)[np.newaxis,:, :], lmax-1, axis=0))
    cov_diag = np.zeros((lmax-1,4,4))
    for ll in range(lmax-1):
        cov_diag[ll,:,:] = cov_diag_full[ll,...][ix]
    return Nl_best,cov_diag
    
def eigval_primary(cov,cov_primary):
    """ Compute the diagonal covmat obtinaed when there is no RS in the covmat 
    but we are still looking for it int he ILC."""
    N = int(cov.shape[1]/2)
    lmax = int(cov.shape[0])+1
    
    eigvals_T, eigvects_T = np.linalg.eigh(cov[:,:N,:N])
    eigvals_E, eigvects_E = np.linalg.eigh(cov[:,N:,N:])

    A_full = np.zeros((lmax-1,2*N,2*N))
    A_full[:,:N,:N] = eigvects_T
    A_full[:,N:,N:] = eigvects_E

    cov_diag_full = np.einsum('lji,ljk,lkm -> lim',A_full,cov_primary,A_full)
    ix = np.ix_([N-1,N-2,-1,-2],[N-1,N-2,-1,-2])
    A = A_full[:,:,[N-1,N-2,-2,-1]]
    cov_diag_prim = np.zeros((lmax-1,4,4))
    for ll in range(lmax-1):
        cov_diag_prim[ll,:,:] = cov_diag_full[ll,...][ix]
    return cov_diag_prim
    
def get_fisher_eigvals(Experiment,lmax=LMAX):
    """ Compute the fisher matrix for the 8 non zeros elements of the diagonalized covariance matrix.
    """
    ell = np.linspace(2,lmax,lmax-1)
    freqs = Experiment.freqs
    N = len(freqs)
    Nl = Experiment.compute_noise(lmax)
    cov = _covariance(Experiment,lmax)
    Nl_best,cov_diag = estimate_noise_eigvals(cov,Nl)
    inv_cov = np.linalg.solve(cov_diag+Nl_best,np.broadcast_to(np.identity(4),(lmax-1,4,4)))
    elem_list = [[0,0],[1,1],[2,2],[3,3],[0,2],[0,3],[1,2],[1,3]]
    deriv = np.zeros((8,4,4))
    signals = np.zeros((lmax-1,8))
    for kk,elem in enumerate(elem_list):
        ii = elem[0]
        jj = elem[1]
        deriv[kk,ii,jj] = 1.
        deriv[kk,jj,ii] = 1.
        signals[:,kk] = cov_diag[:,jj,ii]
    fisher = Experiment.fsky*np.einsum('l,lij,ajk,lkm,bmi -> lab', (2*ell+1)/2.,inv_cov,deriv,inv_cov,deriv)
    return fisher, signals
    
def print_SN(fisher,signals,lmax):
    """ Print S/N of the eigenvalues, marginalising over the other."""
    elem_list = [[0,0],[1,1],[2,2],[3,3],[0,2],[0,3],[1,2],[1,3]]
    labels = ['T1','T2','E1','E2']
    error_cov = np.linalg.inv(fisher)
    for kk,elem in enumerate(elem_list):
        [ii,jj] = elem
        SN2 = signals[:,kk]**2 / error_cov[:,kk,kk]
        SN = SN2.sum()**.5
        print('SN {:s}{:s} : {:3.2f}'.format(labels[ii], labels[jj], SN))
        
        
def get_SN_corr(Experiment,lmax):
    """ Compute the S/N of the correlated RS signal."""
    cov = _covariance(Experiment,lmax)
    cov_primary = _covariance_primary(Experiment,lmax)
    fisher, signals = get_fisher_eigvals(Experiment, lmax)
    cov_diag_prim = eigval_primary(cov,cov_primary)
    error_cov = np.linalg.inv(fisher)
    N_steps = 10000
    beta_array = np.linspace(1-2,1+2,N_steps)
    chi_sqr_TT = np.zeros((lmax-1,N_steps))
    chi_sqr_TE = np.zeros((lmax-1,N_steps))
    chi_sqr_EE = np.zeros((lmax-1,N_steps))
    for k,beta in enumerate(beta_array):
        chi_sqr_TT[:,k] = ((signals[:,0] - beta*cov_diag_prim[:,0,0])**2 / error_cov[:,0,0])
        chi_sqr_TE[:,k] = ((np.abs(signals[:,4]) - beta * np.abs(cov_diag_prim[:,0,2]))**2 / error_cov[:,4,4])
        chi_sqr_EE[:,k] = ((signals[:,2] - beta * cov_diag_prim[:,2,2])**2 / error_cov[:,2,2])
    SN2 = np.zeros((3,lmax-1))
    for ll in range(lmax-1):
        loglike_TT = np.exp(-chi_sqr_TT[ll,:])
        loglike_TE = np.exp(-chi_sqr_TE[ll,:])
        loglike_EE = np.exp(-chi_sqr_EE[ll,:])
        popt_TT,_ = optimize.curve_fit(_gaussian,beta_array,loglike_TT,p0=[.5,1.,1.], maxfev=5000)
        popt_TE,_ = optimize.curve_fit(_gaussian,beta_array,loglike_TE,p0=[.5,1.,1.], maxfev=5000)
        popt_EE,_ = optimize.curve_fit(_gaussian,beta_array,loglike_EE,p0=[.5,1.,1.], maxfev=5000)
#         print(popt_TT)

        SN2[0,ll] = np.abs(1. - popt_TT[1])/np.abs(popt_TT[2])**2
        SN2[1,ll] = np.abs(1. - popt_TE[1])/np.abs(popt_TE[2])**2
        SN2[2,ll] = np.abs(1. - popt_EE[1])/np.abs(popt_EE[2])**2
#     plt.figure()
#     plt.plot(SN2[0])
#     plt.show()

        
    print('Correlated TT : {:3.2f}'.format(SN2[0].sum()**.5))
    print('Correlated TE : {:3.2f}'.format(SN2[1].sum()**.5))
    print('Correlated EE : {:3.2f}'.format(SN2[2].sum()**.5))


             
def plot_all(fisher,signals,lmax,label):
    """ Plot the  8 non zeros elements of the diagonalized covariance matrix, 
    their S/N per mode as well as the cumulative S/N."""
    error_cov = np.linalg.inv(fisher)
    labels = ['T1','T2','E1','E2']
    elem_list = [[0,0],[1,1],[2,2],[3,3],[0,2],[0,3],[1,2],[1,3]]

    ell = np.linspace(2,lmax,lmax-1)
    for kk,elem in enumerate(elem_list):
        [ii,jj] = elem
        f,[ax1,ax2] = plt.subplots(2,1,sharex = True,gridspec_kw={'height_ratios': [2, 1]},figsize=(10,6))
        plt.subplots_adjust(wspace=0, hspace=0,left=0.1,right=0.9,top = 0.9,bottom=0.1)
        ax1.plot(ell,(signals[:,kk]**2)**.5, c = 'k', lw = 2)
        SN2 = signals[:,kk]**2 / error_cov[:,kk,kk]
        ax_cp = ax1.twinx()
        ax_cp.plot(ell,SN2**.5, c = 'r', ls = 'dashed')
        ax2.plot(ell,(np.cumsum(SN2))**.5, c = 'r')
        ax_cp.tick_params(axis='y', labelcolor='r')
        ax_cp.set_ylabel(r'S/N per mode', fontsize = 14, color='r')
        ax1.set_ylabel(r'',fontsize = 14)
        ax2.set_xlim(2,LMAX)
        ax2.set_ylabel(r'Cumulative S/N', fontsize = 14)
        ax2.set_xlabel(r'$\ell$', fontsize = 18)
        plt.suptitle('{:s}x{:s}'.format(labels[ii],labels[jj]))

    plt.show()
    
if __name__ == '__main__':
    
    print('PLANCK')
    fisher, signals = get_fisher_eigvals(PLANCK,lmax=2500)
    save_eigvals_eigvects(PLANCK, lmax=2500, file='./data/eigvals/PLANCK', normalize = True)
    ok = ru
#     print_SN(fisher, signals,lmax=2500)
    get_SN_corr(PLANCK,2500)
    print()
    print('SO LAT')
    fisher, signals = get_fisher_eigvals(SO_LAT,lmax=4500)
#     print_SN(fisher, signals,lmax=4500)
    get_SN_corr(SO_LAT,4500)
    print()
    print('CCAT')
    fisher, signals = get_fisher_eigvals(CCAT,lmax=4500)
#     print_SN(fisher, signals,lmax=4500)
    get_SN_corr(CCAT,4500)




    
    
#     plot_all(fisher,signals,4500,'SO_LAT')
    
