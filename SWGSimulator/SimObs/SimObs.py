import sys
import numpy as np
from matplotlib import pyplot
from SWGSimulator.Tools import Parser, NoiseModels, HDF5Tools
from SWGSimulator.SkyModel import ska_fgchallenge
from SWGSimulator.Beams import Beams
from SWGSimulator.SimObs import SimObs
import healpy as hp
from astropy.io import fits
import h5py
from tqdm import tqdm
import os


def main(parser):

    beams = Beams.load_beams('{}/{}'.format(parser['Beams']['beams_outputdir'],
                                            parser['Beams']['beams_outputfile']))
                                            

    hits = hp.ud_grade(hp.ud_grade(hp.read_map(parser['SimObs']['hits_map'],1),256,power=-2),512,power=-2)
    mask = (hits > 0)

    frequencies = np.linspace(parser['Beams']['vstart'],parser['Beams']['vend'],int(parser['Beams']['space']))

    common_data = {}
    for telescope in parser['SimObs']['telescopes']:
        common_data[telescope] = {}
        for beam_type, beam_data in beams[telescope].items():
            good = (beam_data.theta < parser['SimObs']['max_theta'])
            common_data[telescope][beam_type] = {'FWHM':[beam_data.FWHM, {'UNIT':'degree'.encode()}],
                                                 'Frequency': [frequencies ,{'UNIT':'MHz'.encode()}],
                                                 'BeamModel': [beam_data.model[:,good],{}],
                                                 'Colatitude': [beam_data.theta[good],{'UNIT':'degree'.encode()}],
                                                 'Hits':[hits[mask]*parser[telescope]['dishes']*parser[telescope]['repeats'],{'UNIT':'seconds'.encode()}]}

        
    sky_maps = h5py.File('{}/SkyModels.hd5'.format(parser['SkyModel']['output_dir']),'r')
    if parser['SimObs']['include_hi']:
        hi_maps  = fits.open(parser['SkyModel']['hi_maps'])[0].data.T/1e3
    else:
        hi_maps = 0


    if not os.path.exists(parser['SimObs']['output_dir']):
        os.makedirs(parser['SimObs']['output_dir'])

    output = h5py.File('{}/swg_fgchallenge2020_withHI:{}_withFGs:{}_{}.hd5'.format(parser['SimObs']['output_dir'],
                                                                                   parser['SimObs']['include_hi'],
                                                                                   parser['SimObs']['include_fgs'],
                                                                                   parser['SimObs']['noise_mode'],
                                                                                   parser['SimObs']['version']),'a')

    x,y,z = np.meshgrid(parser['SimObs']['sky_models'],parser['SimObs']['telescopes'],parser['SimObs']['beam_models'])
    for (skysim, telescope, beamtype) in zip(tqdm(x.flatten()), y.flatten(), z.flatten()):
        if not isinstance(parser['SimObs']['random_seed'],type(None)):
            np.random.seed(int(parser['SimObs']['random_seed']))

        pixelids = np.where((mask))[0]
        good_npixels = len(pixelids)

        if  parser['SimObs']['include_fgs']:
            fg_maps = sky_maps[skysim]['Model_Sky'][...]
        else:
            fg_maps = 0

        maps = fg_maps + hi_maps
        grp  = HDF5Tools.get_group('{}/{}/{}'.format(skysim, telescope, beamtype), output)
        dset = HDF5Tools.create_dataset(grp, (maps.shape[0], good_npixels,), 
                              maps.dtype, 
                              'Maps',
                              UNIT='K'.encode(), NSIDE=512)
        pixel_dset = HDF5Tools.create_dataset(grp, (good_npixels,), pixelids.dtype, 
                                    'Pixels', NSIDE=512)
        pixel_dset[:] = pixelids
        for i in tqdm(range(frequencies.size)):
            # First smooth the map
            alm = hp.map2alm(maps[i,:])
            m = hp.alm2map(hp.almxfl(alm,beams[telescope][beamtype].beam_cl[i,:],inplace=True),int(parser['SkyModel']['nside']),verbose=False)
            # Cut out the sky area not observed
            m = m[mask] 
            m = NoiseModels.TsysFuncs[telescope](m)
            # Set the noise level
            if parser['SimObs']['noise_mode'] == 'white':
                m += NoiseModels.GetNoise(m,common_data[telescope][beamtype]['Hits'][0])
            elif parser['SimObs']['noise_mode'] == 'no_noise':
                pass
            else:
                pass
            dset[i,:] = m
        HDF5Tools.create_ancillary(grp,common_data, skysim, telescope, beamtype)
    output.close()
