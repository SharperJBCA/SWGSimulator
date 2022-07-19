"""
Created on Thurs 6th Feb 2020

@author: mirfan, scunnington
"""

import numpy as np
import healpy as hp
from astropy.io import fits as pyf
from SWGSimulator.SkyModel.noresolPSbattye import make_ps_nobeam
from matplotlib import pyplot
import os
import h5py 

def planckcorr(freq_ghz):
    """ Takes in frequency in GHZ and produces factor to be applied to temp """

    freq = freq_ghz * 10.**9.
    factor = CST["plancks"] * freq / (CST["kbolt"] * CST["cmb_temp"])
    correction = (np.exp(factor)-1.)**2. / (factor**2. * np.exp(factor))

    return correction

def gen_cl(vmid, ell, fgtype):
    """ Produce Cl's in K^2 for Gaussian realisations of different foregrounds
    
    Spectrum is returned in pixel-space (expects units of K not K^2)
    """

    l_ref = 1000
    v_ref = 130

    if fgtype == 'sync':
        aval = 700e-6
        bval = 2.4
        alpha = 2.80
        xival = 4.0

    if fgtype == 'ps':
        aval = 57e-6
        bval = 1.1
        alpha = 2.07
        xival = 1.0

    if fgtype == 'free':
        aval = 0.088e-6
        bval = 3.0
        alpha = 2.15
        xival = 35.

    if fgtype == 'egfree':
        aval = 0.014e-6
        bval = 1.0
        alpha = 2.10
        xival = 35.

    # return Cl and spectrum separately:
    cl = aval * (l_ref / ell)**bval 
    # * np.exp(-((np.log(vbit1 / vbit2))**2 / (2 * xival**2)))
    cl[0] = cl[1]

    spectrum = (v_ref/vmid)**alpha

    #(v_ref**2 / (vbit1 * vbit2))**alpha #* np.exp(-((np.log(vbit1 / vbit2))**2 / (2 * xival**2)))
    #cl = aval * (l_ref / ell)**bval * (v_ref**2 / (vbit1 * vbit2))**alpha * \
    #     np.exp(-((np.log(vbit1 / vbit2))**2 / (2 * xival**2)))
    return cl, spectrum
    
CST = {"kbolt": 1.3806488e-23, "light": 2.99792458e8, "plancks": 6.626e-34, "cmb_temp": 2.73}

def generate(nside, ffp10loc, vstart, vend, space, freeind=-2.13, smax=0.1, psbeta=-2.7, \
             psdelt=0.2, gauss=False,output_dir='skymodels/'):
    """
    determine the ff, synchrotron and noise contributions for a given MHZ frequency range

    INPUTS: nside - desired nside for output maps
            ffp10loc - location of PSM maps
            vstart - beginning of frequency range (MHz)
            vend - end of frequency range (MHz)
            space - number of frequency bands
            freeInd - spectral index of free-free emission (for temperature not flux)
            Smax - the cut off frequency in Jy for point sources
            psbeta - point source spectral index
            psdelt - spread of ps spectral index
            Gauss - True if you want gaussian diffuse foreground, False if you want to use
                    the planck sky model
    output_dir - output_directory for sky maps

    OUTPUTS:
            map["totsig"] - [n_obs X npix] matrix of foregrounds at beam FWHMs
            map["sync"] - [n_obs X npix] matrix of synchrotron emission
            map["free"] - [n_obs X npix] matrix of free-free emission
            map["beta"] - vector of synchrotron spectral index
            freqs - array of frequencies (MHz)
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    nside  = int(nside)
    vstart = int(vstart)
    vend   = int(vend)
    space  = int(space)

    #admin
    freqs = np.linspace(vstart, vend, space)
    freq_edges = np.linspace(vstart,vend, space+1)
    lenf = len(freqs)

    if gauss is False:

        s217loc = f'{ffp10loc}/COM_SimMap_synchrotron-ffp10-skyinbands-217_2048_R3.00_full.fits'
        f217loc = f'{ffp10loc}/COM_SimMap_freefree-ffp10-skyinbands-217_2048_R3.00_full.fits'
        s353loc = f'{ffp10loc}/COM_SimMap_synchrotron-ffp10-skyinbands-353_2048_R3.00_full.fits'

        #Convert maps from Tcmb to Trj
        sync217 = hp.fitsfunc.read_map(s217loc, field=0, nest=False) / planckcorr(217)
        sync353 = hp.fitsfunc.read_map(s353loc, field=0, nest=False) / planckcorr(353)
        free217 = hp.fitsfunc.read_map(f217loc, field=0, nest=False) / planckcorr(217)

        #get rid of unphysical values
        free217[np.where(free217 < 0.)[0]] = np.percentile(free217, 3)

        syncind = np.log(sync353/ sync217) / np.log(353./217.)

        synca = sync217 * ((freqs[0]/1000.)/217.)**(syncind)
        freea = free217 * ((freqs[0]/1000.)/217.)**(freeind)

        #need to fill in small scale structure for sync ind map which is MAMD's map at 5deg
        els = np.array(range(4000)) + 1.0
        cl5deg = hp.sphtfunc.anafast(np.random.normal(0.0, np.std(syncind), 12*2048*2048), lmax=4000)
        #power spectra taken from https://arxiv.org/pdf/astro-ph/0408515.pdf
        cls = cl5deg[0] * (1000./els)**(2.4)
        np.random.seed(90)
        syncind = syncind + hp.sphtfunc.synfast(cls, 2048)

        #downgrade
        synca = hp.pixelfunc.ud_grade(synca, nside)
        freea = hp.pixelfunc.ud_grade(freea, nside)
        syncind = hp.pixelfunc.ud_grade(syncind, nside)

        syncmap = np.array([synca * (freqs[ii]/freqs[0])**(syncind) for ii in range(lenf)])
        freemap = np.array([freea * (freqs[ii]/freqs[0])**(freeind) for ii in range(lenf)])

        #now have PS
        psmap, tpsmean = make_ps_nobeam(nside, freqs, smax, psbeta, psdelt)

        #write out
        print(synca,type(synca))
        h = h5py.File(f'{output_dir}/SkyModels.hd5','a')
        if 'PSM' in h:
            del h['PSM']
        grp = h.create_group('PSM')
        
        total_map = syncmap + freemap + psmap
        grp.create_dataset('Model_Sky', data= total_map)
        grp.create_dataset('Frequency', data = freqs)
        grp.create_dataset('Model_Sky_Offsets', data = tpsmean)
        grp.create_dataset('Synchrotron',data=syncmap)
        grp.create_dataset('FreeFree',data=freemap)
        grp.create_dataset('PointSouces',data=psmap)

        monopole = np.argmin(total_map,axis=1)
        dset = grp.create_dataset('Monopole', data=monopole)

        if 'Components' in grp:
            components = grp['Components']
        else:
            components = grp.create_group('Components')

        components.create_dataset('Synchrotron_Amplitude',data=synca)
        components.create_dataset('Synchrotron_Index',data=syncind)
        components.create_dataset('FreeFree_Amplitude',data=freea)
        components.create_dataset('FreeFree_Index',data = freeind)

        h.close()
    else:

        if nside < 2048:
            els = np.arange(nside * 3)
        else:
            els = np.arange(4000)

        deltav = freqs[1] - freqs[0]
        vbins = freqs - deltav/2
        vbins = freq_edges #np.append(vbins, freqs[-1]+deltav/2)
        np.random.seed(0)

        h = h5py.File(f'{output_dir}/SkyModels.hd5','a')
        if 'MS05' in h:
            del h['MS05']
        grp = h.create_group('MS05')


        total_map = np.zeros((freqs.size, 12*nside**2))
        for mode in ['sync','free','ps','egfree']:
            cl, spectrum = gen_cl(freqs, els, mode)
            fg_map = hp.synfast(cl, nside, verbose=False)
            dset = grp.create_dataset(mode, data=fg_map)
            dset.attrs['Unit'] = 'K'
            for i in range(freqs.size):
                total_map[i,:] += fg_map*spectrum[i]

        monopole =  16 * (408./freqs)**2.7
        dset = grp.create_dataset('Monopole', data=monopole)

        total_map = total_map + monopole[:,None]
        dset = grp.create_dataset('Total', data=total_map)
        dset.attrs['Unit'] = 'K'

            #pyf.writeto('sky_models/{}mapMS.fits'.format(mode), fg_map,overwrite=True)
        #pyf.writeto('sky_models/totalfluxMS.fits',total_map,overwrite=True)

    return None

def main(parser):

    skyp = parser['SkyModel']

    if skyp['gauss_ms']:
        generate(skyp['nside'], 
                 skyp['ancil_dir'], 
                 skyp['frequency_low'],
                 skyp['frequency_high'],
                 skyp['nfrequencies'], 
                 freeind=skyp['freefree_ind'], 
                 smax=skyp['ps_max'], 
                 psbeta=skyp['ps_beta'],
                 psdelt=skyp['ps_delt'], 
                 output_dir=skyp['output_dir'],
                 gauss=True)
    if skyp['psm']:
        generate(skyp['nside'], 
                 skyp['ancil_dir'], 
                 skyp['frequency_low'],
                 skyp['frequency_high'],
                 skyp['nfrequencies'], 
                 freeind=skyp['freefree_ind'], 
                 smax=skyp['ps_max'], 
                 psbeta=skyp['ps_beta'],
                 psdelt=skyp['ps_delt'], 
                 output_dir=skyp['output_dir'],
                 gauss=False)


if __name__ == "__main__":
    pass
    #running example for SKA foreground challenge
    #pre = './'
    #generate(512, pre, 950, 1400, 512, freeind=-2.1, smax=0.1, psbeta=-2.7, psdelt=0.2, gauss=False)
    #generate(512, pre, 950, 1400, 512, freeind=-2.1, smax=0.1, psbeta=-2.7, psdelt=0.2, gauss=True)
