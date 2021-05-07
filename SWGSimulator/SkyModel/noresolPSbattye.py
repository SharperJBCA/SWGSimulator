"""
Created on Wed 11th Sep 2019

@author: mirfan
"""
import healpy as hp
import numpy as np
import scipy.integrate as defint

CST = {"kbolt": 1.3806488e-23, "light": 2.99792458e8, "plancks": 6.626e-34, "cmb_temp": 2.73}

def battye(sjy):
    """empirical point source model of battye 2013 for intergrated flux """

    oooh = np.log10(sjy)

    sumbit = 2.593 * oooh**0 + 9.333 * 10**-2 * oooh**1. -4.839 * 10**-4 * oooh**2. \
                + 2.488 * 10**-1 * oooh**3. + 8.995 * 10**-2 * oooh**4. + \
                    8.506 * 10**-3 * oooh**5.

    inte = (10.**sumbit) * sjy**(-2.5) * sjy

    return inte

def pois(sjy):
    """empirical point source model of battye 2013 for poisson power spec """

    oooh = np.log10(sjy)

    sumbit = 2.593 * oooh**0 + 9.333 * 10**-2 * oooh**1. -4.839 * 10**-4 * oooh**2. \
                + 2.488 * 10**-1 * oooh**3. + 8.995 * 10**-2 * oooh**4. + \
                    8.506 * 10**-3 * oooh**5.

    inte = (10.**sumbit) * sjy**(-2.5) * sjy**(2.0)

    return inte

def numcount(sjy):
    """empirical point source model og battye 2013 for source count """

    oooh = np.log10(sjy)

    sumbit = 2.593 * oooh**0 + 9.333 * 10**-2 * oooh**1. -4.839 * 10**-4 * oooh**2. \
                + 2.488 * 10**-1 * oooh**3. + 8.995 * 10**-2 * oooh**4. + \
                    8.506 * 10**-3 * oooh**5.

    inte = (10.**sumbit) * sjy**(-2.5)

    return inte

#Based on Eq 36 from https://arxiv.org/pdf/1209.0343.pdf
#and https://www.research.manchester.ac.uk/portal/files/67403180/FULL_TEXT.PDF p99
# life tip: don't forget 10^-26 conversion from Jy into Wm-2Hz-1

def make_ps_nobeam(nside, freqs, smax, beta, deltbeta):
    """ make a range of ps maps from nside, frequencies, cut-off flux in Jy and
        resolution in arcmin """

    ell = np.arange(nside*3) + 1.0
    npix = 12 * nside * nside
    pixarea = (np.degrees(4 * np.pi) * 60.) / (npix)
    lenf = len(freqs)
    cfact = CST["light"]**2 / (2 * CST["kbolt"] * (1.4e9)**2) * 10.**-26

    ######### first to make the point source map at 1.4 GHz ################
    # Get the mean temperature
    intvals = defint.quad(lambda sjy: battye(sjy), 0., smax)
    tps14 = cfact * (intvals[0] - intvals[1])

    #Get the clustering contribution
    clclust = 1.8 * 10**-4 * ell**-1.2 * tps14**2
    np.random.seed(0)
    clustmap = hp.sphtfunc.synfast(clclust, nside, new=True)

    #Get the poisson contribution
    #under 0.01 Jy poisson contributions behave as gaussians
    clpoislow = np.zeros((len(ell)))
    val = 0
    for ival in np.arange(1e-6, 0.01, (0.01-1e-6)/ len(ell)):
        intvals = defint.quad(lambda sjy: pois(sjy), 0., ival)
        clpoislow[val] = cfact**2 * (intvals[0] - intvals[1])
        val += 1
    np.random.seed(10)
    poislowmap = hp.sphtfunc.synfast(clpoislow, nside, new=True)

    shotmap = np.zeros((npix))
    #over 0.01 Jy you need to inject sources into the sky
    if smax > 0.01:
        for ival in np.arange(0.01, smax, (smax - 0.01)/10.):
            #N is number of sources per steradian per jansky
            numbster = defint.quad(lambda sjy: numcount(sjy), ival - 1e-3, ival + 1e-3)[0]
            numbsky = int(4 * np.pi * numbster * ival)
            tempval = cfact * defint.quad(lambda sjy: battye(sjy), 0.01, ival)[0] / pixarea
            print (numbsky, tempval)
            randind = np.random.choice(range(npix), numbsky)
            shotmap[randind] = tempval

    map14 = tps14 + poislowmap + clustmap + shotmap
    #########################################################################

    ######### scale up to different frequencies ################
    alphas = np.random.normal(beta, scale=deltbeta**2, size=npix)
    maps = np.array([map14 * (freqs[freval]/1400.)**(alphas) for freval in range(lenf)])
    tps_mean = np.array([tps14 * (freqs[ival]/1400.)**(beta) for ival in range(lenf)]).reshape(lenf, 1)
    #########################################################################

    return maps, tps_mean
