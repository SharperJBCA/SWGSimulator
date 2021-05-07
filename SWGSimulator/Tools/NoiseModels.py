# NOISE MODELS FOR SKA/MEERKAT RECEIVERS
import numpy as np

def TsysSKA(m):

    # red book estimates, in K
    Trx = 7.5 
    Tcmb= 2.73
    Tspl= 3.

    tsys = (m+Trx + Tcmb + Tspl)#/np.sqrt(dv*hits)
    return tsys

def TsysMeerkat(m):

    # From SKA Science Performance booklet, Oct 2017
    nu = np.linspace(0.95,1.4,1000)
    Trx = np.mean(7.5 + 6.8*np.abs(nu-1.65)**1.5)
    Tcmb= 2.73
    Tspl= 3.

    tsys = (m+Trx + Tcmb + Tspl)
    
    return tsys

def GetNoise(tsys,hits,dv=0.87890625e6):
    rms  = tsys/np.sqrt(dv*hits)
    return np.random.normal(scale=rms)

TsysFuncs = {'SKA':TsysSKA,
             'Meerkat':TsysMeerkat}
