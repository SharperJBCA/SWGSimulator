import numpy as np
import h5py

def get_group(name,hdfile):
    if not name in hdfile:
        grp = hdfile.create_group(name)
    else:
        grp = hdfile[name]
        
    return grp

def create_dataset(hdfile, shape,dtype, name, **kwargs):
    if name in hdfile:
        del hdfile[name]
    dset = hdfile.create_dataset(name,shape, dtype=dtype)

    for k,v in kwargs.items():
        dset.attrs[k] = v

    return dset

def create_ancillary(grp,commondata, skysim, telescope, beamtype):

    for k,v in commondata[telescope][beamtype].items():
        dset = create_dataset(grp, v[0].shape, v[0].dtype, k, **v[1])
        dset[...] = v[0]
