import numpy as np
from matplotlib import pyplot
from SWGSimulator.Tools import Coordinates
from scipy.signal import sawtooth

SECONDS_PER_DAY = 86400.

class Telescope:

    def __init__(self,
                 longitude=0,
                 latitude=0,
                 scanning_speed=1,
                 acceleration=1,
                 sample_rate = 1):

        self.longitude=longitude # degree
        self.latitude =latitude # degree 
        self.scanning_speed=scanning_speed # degree/sec
        self.sample_rate = sample_rate # Hz
        self.acceleration = acceleration # degrees/sec^2

    def get_scan_length(self,
                        dec_centre,
                        start_mjd, 
                        strip_width,
                        start_elevation,nsamples=720):

        
        az_ring = np.linspace(0,180,nsamples)
        el_ring = np.ones(nsamples)*start_elevation
        mjd     = np.ones(nsamples)*start_mjd
        ra,dec= Coordinates.h2e_full(az_ring, 
                                     el_ring,
                                     mjd, 
                                     self.longitude,
                                     self.latitude)

        
        upper,lower = dec_centre + strip_width/2.,dec_centre-strip_width/2.
        itop = np.argmin((dec-upper)**2)
        ibot = np.argmin((dec-lower)**2)

        scan_length = np.abs(az_ring[itop]-az_ring[ibot])
        return scan_length

    def observe(self,
                declination,
                start_mjd, 
                end_mjd,
                strip_width,
                start_azimuth,
                start_elevation):

        obs_length = (end_mjd - start_mjd)*SECONDS_PER_DAY
        obs_samples = int(obs_length/self.sample_rate)


        # Calculate the scan length
        scan_length = self.get_scan_length(declination,
                                           start_mjd, 
                                           strip_width,
                                           start_elevation)

        print(scan_length,strip_width)
        # Wavelength therefore is scan_time * 2
        scan_wavelength = 2*scan_length/self.scanning_speed

        time = np.linspace(0,obs_length,obs_samples)
        #scans = scan_length/2.*sawtooth(2*np.pi*time/scan_wavelength,0.5) + start_azimuth
        scans = scan_length/2.*np.sin(2*np.pi*time/scan_wavelength) + start_azimuth
        mjd = np.linspace(start_mjd,end_mjd,obs_samples)

        return scans,np.ones(obs_samples)*start_elevation,mjd,scan_length

from scipy.interpolate import interp1d
from scipy.optimize import root_scalar,fsolve

class Scheduler:

    def __init__(self):
        pass

    def get_coordinates(self,_ra,_dec,telescope,nsamples=360):

        ra = np.ones(nsamples)*_ra
        dec= np.ones(nsamples)*_dec
        mjd= np.linspace(0,1,nsamples) + Coordinates.MJD_2000

        az,el,ha = Coordinates.e2h_full(ra, 
                                        dec,
                                        mjd, 
                                        telescope.longitude,
                                        telescope.latitude,
                                        return_lha=True)

        target_el = np.max(el)/2.

        model = interp1d(ha,el-target_el)
        xmin  = np.argmin((el-target_el)**2)
        sol   = fsolve(model,x0=ha[xmin])
        
        cross_ha = np.sort( np.array([sol,-sol]) )

        i0 = np.argmin((ha-cross_ha[0])**2)
        i1 = np.argmin((ha-cross_ha[1])**2)
        
        return (az[i0],el[i0],mjd[i0]),(az[i1],el[i1],mjd[i1])

