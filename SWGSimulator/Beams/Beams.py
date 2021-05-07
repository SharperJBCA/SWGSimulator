import h5py 
import numpy as np
from matplotlib import pyplot
from reproject import reproject_from_healpix
from astropy import wcs
import healpy as hp
from tqdm import tqdm
from scipy.special import j0,j1
from scipy.integrate import quad
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from astropy.io import fits
import glob
import os

class Beam:

    def __init__(self,beam_data=None):
        
        self.info = {}
        self.data = {}

        if not isinstance(beam_data, type(None)):
            self.load_beam_data(beam_data)
    
    def load_beam_data(self,beam_data):
        # Load Beams
        for k,v in beam_data.items():
            self.data[k] = v[...]
            if k in v.attrs:
                self.info[k] = v.attrs[k]

            self.__dict__[k] = v[...]

            
        
class AiryBeam(Beam):
    def __init__(self, nside=512, aperture_taper_width=20, dish_diameter=15, thetamax=90,nsamples=2048,**kwargs): 
        """

        """
        super().__init__()
        # constants
        self.c = 3e8
        self.nside = int(nside)

        self.aperture_taper_width = aperture_taper_width 
        self.dish_diameter = dish_diameter
        self.thetamax = thetamax
        self.nsamples = int(nsamples)

    def BeamFunction(self,apfunc, r, sigma, wl, thetamax=5, nsamples=128):
        """
        apfunc - Aperture distribution function
        
        r     - Radius out to D/2 (m)
        sigma - taper width in wavelengths over dish
        wl    - wavelength (m)
        """
    
        # Convert to No. wavelength units
        rho    = r/wl
        drho   = np.abs(rho[1]-rho[0])

        # To store output beam pattern.
        beam   = np.zeros(nsamples)

        # Line-of-sight angles to calculate beam power 
        thetas = np.linspace(0,thetamax,nsamples)*np.pi/180.
    
        # Loop over thetas
        for i, theta in enumerate(thetas):
        
            # Tools of Radio Astronomy 5th Ed. eqn. 6.67
            top = np.sum(apfunc(r,sigma, wl)*j0(2*np.pi*np.sin(theta)*rho)*rho)*drho
            bot = np.sum(apfunc(r,sigma,wl)*rho)*drho
            beam[i] = np.abs(top/bot)**2
        
        return beam, thetas*180/np.pi

    def ApertureFunc(self,r,sigma,wl):
        """
        r     - Radius out to D/2 (m)
        sigma - taper width in wavelengths over dish
        wl    - wavelength (m)
        """
        rho = (r/wl)
    
        f = np.exp(-0.5*(rho/sigma)**2) 
    
        return f

    def __call__(self,frequencies):
        """
        Create Realistic beam models
        """
        lmax = 5*self.nside-1

        self.nfreqs = len(frequencies)
        nsamples = 2048

        self.model = np.zeros((self.nfreqs, nsamples))
        self.tfunc = np.zeros((self.nfreqs, lmax+1))
        self.fwhm = np.zeros(self.nfreqs)
        self.fwhm_frequencies = frequencies*1.

        # setup
        wls = self.c/frequencies/1e6
        r = np.linspace(0,self.dish_diameter/2.,1000) # m

        # First create beam model, then the equivalent transfer function
        for i, wl in enumerate(tqdm(wls)):

            # self.clats in degrees
            self.model[i,:], self.clats   = self.BeamFunction(self.ApertureFunc , 
                                                                   r,
                                                                   self.aperture_taper_width ,
                                                                   wl,
                                                                   thetamax=self.thetamax,nsamples=self.nsamples)
            good = self.model[i,:] > 0.4

            pmdl = interp1d(self.model[i,good], self.clats[good])
            self.fwhm[i] = pmdl(0.5)*2

            self.tfunc[i,:] = hp.beam2bl(self.model[i,:], self.clats*np.pi/180.,lmax=lmax)
            self.tfunc[i,:] /= self.tfunc[i,1]

        self.diffraction_constant = np.polyfit(wls/self.dish_diameter * 180./np.pi, self.fwhm,1)[1]
        self.data['model'] = self.model
        self.data['theta'] = self.clats
        self.data['beam_cl'] = self.tfunc
        self.info['theta'] = {'Unit':'degrees'}
        self.data['diffraction_constant'] = (self.diffraction_constant,)
        self.data['FWHM'] = self.fwhm


class GaussBeam(Beam):
    def __init__(self, nside=512,dish_diameter = 13.5, diffraction_constant=1.1,**kwargs): 
        """

        """
        super().__init__()
        # constants
        self.c = 3e8
        self.nside = int(nside)

        self.dish_diameter = dish_diameter
        self.diffraction_constant = diffraction_constant

    def __call__(self,frequencies,fwhm=None):
        """
        Create the gaussian beam models
        """
        
        gauss = lambda x, sigma : np.exp(-0.5*(x/sigma)**2)

        lmax = 5*self.nside

        self.nfreqs = len(frequencies)
        nsamples = 2048
        thetamax = 5

        self.model = np.zeros((self.nfreqs, nsamples))
        self.tfunc = np.zeros((self.nfreqs, lmax+1))
        wls = self.c/frequencies/1e6
        if isinstance(fwhm,type(None)):
            fwhm = wls/self.dish_diameter * self.diffraction_constant * 180./np.pi
        else:
            self.diffraction_constant = np.polyfit(wls/self.dish_diameter * 180./np.pi, fwhm,1)[1]

        self.clats = np.linspace(0,thetamax,nsamples)
        # First create beam model, then the equivalent transfer function
        for i, wl in enumerate(tqdm(wls)):
            # self.clats are calculate by BeamFunction in degrees, thus FWHM in degrees
            self.model[i,:] = gauss(self.clats, fwhm[i]/2.355)

            self.tfunc[i,:] = hp.beam2bl(self.model[i,:], self.clats*np.pi/180.,lmax=lmax)
            self.tfunc[i,:] /= self.tfunc[i,1]
    

        self.data['model'] = self.model
        self.data['theta'] = self.clats
        self.data['beam_cl'] = self.tfunc
        self.data['FWHM'] = fwhm
        self.info['theta'] = {'Unit':'degrees'}
        self.data['diffraction_constant'] = (self.diffraction_constant,)

def save_beams(beams,output_name,output_dir=''):
    """
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    h = h5py.File(f'{output_dir}/{output_name}','a')

    # Save Beams
    for beam_group, beam_models in beams.items(): # MeerKAT or SKA?
        if not beam_group in h:
            grp = h.create_group(beam_group)
        else:
            grp = h[beam_group]
        for beam_type, beam in beam_models.items(): # Airy/Gauss?
            if not beam_type in grp:
                type_grp = grp.create_group(beam_type)
            else:
                type_grp = grp[beam_type]
            for k,v in beam.data.items(): # Data
                if k in type_grp:
                    del type_grp[k]
                dset = type_grp.create_dataset(k,data=v)
                if k in beam.info:
                    for kinfo, vinfo in beam.info[k].items():
                        dset.attrs[kinfo] = vinfo
    h.close()

def load_beams(filename):
    """
    Load a beam file
    """
    h = h5py.File(f'{filename}','r')
    
    beams = {}
    for beam_group, beam_model in h.items(): #e.g. Meerkat/SKA
        beams[beam_group] = {}
        for beam_type, beam_data in beam_model.items(): # e.g. Gaussian/Airy
            beams[beam_group][beam_type] = Beam(beam_data)
    h.close()

    return beams

def main(parser):

    beam_types = parser['Beams']['beam_types']
    frequencies = np.linspace(parser['Beams']['vstart'],parser['Beams']['vend'],int(parser['Beams']['space']))

    beams = {}
    for beam_type in tqdm(beam_types):
        airy  = AiryBeam(**parser[beam_type])
        gauss = GaussBeam(**parser[beam_type])

        airy(frequencies)
        gauss(frequencies,airy.fwhm)

        beams[beam_type] = {'airy':airy, 'gauss':gauss}

    save_beams(beams,parser['Beams']['beams_outputfile'],parser['Beams']['beams_outputdir'])
