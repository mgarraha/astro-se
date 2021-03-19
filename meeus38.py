#!/usr/bin/env python3

# Fits coefficients for Earth perihelion and aphelion dates
# approximated as in Meeus "Astronomical Algorithms" chapter 38.
# Supplementary to https://astronomy.stackexchange.com/a/42016

import skyfield.api as sf
import skyfield.searchlib as sfs
import numpy as np
import scipy.optimize as opt
from math import *

yr0 = 2010
yr1 = 2050
nyr = yr1 - yr0

ts = sf.load.timescale()
t0 = ts.J(yr0 - 0.25)
t1 = ts.J(yr1 - 0.25)

ephem = sf.load('de430t.bsp')
earth = ephem['earth']
sun = ephem['sun']

####

def sun_earth_dist(time):
    pos = sun.at(time).observe(earth)
    lat, lon, dist = pos.ecliptic_latlon()
    return dist.au

sun_earth_dist.rough_period = 182.5

# First approximation for the Earth-Moon barycenter, ignoring perturbations.
def meeus1(k):
    return 2451547.507 + 365.2596358 * k + 1.56e-8 * k**2

# Corrects meeus1 for the effects of the Moon, Venus, and Jupiter.
def meeus2(k, \
        a1p, a2p, a3p, a4p, a5p, a1a, a2a, a3a, a4a, a5a, \
        b1, b2, b3, b4, b5, c1, c2, c3, c4, c5):
    A1 = np.radians(c1 + b1 * k)
    A2 = np.radians(c2 + b2 * k)
    A3 = np.radians(c3 + b3 * k)
    A4 = np.radians(c4 + b4 * k)
    A5 = np.radians(c5 + b5 * k)
    apo = (2 * k).astype(int) % 2
    peri = 1 - apo
    a1 = peri * a1p + apo * a1a
    a2 = peri * a2p + apo * a2a
    a3 = peri * a3p + apo * a3a
    a4 = peri * a4p + apo * a4a
    a5 = peri * a5p + apo * a5a
    return meeus1(k) + a1 * np.sin(A1) + a2 * np.sin(A2) \
            + a3 * np.sin(A3) + a4 * np.sin(A4) + a5 * np.sin(A5)

# Returns a copy of dates, replacing near-duplicate pairs with single values.
# Works around a numerical quirk in skyfield.searchlib.
def dedup(dates):
    n = len(dates)
    retval = []
    skip = False
    for i in range(n-1):
        if skip:
            skip = False
        elif dates[i+1] - dates[i] < 1:
            retval.append((dates[i] + dates[i+1]) / 2)
            skip = True
        else:
            retval.append(dates[i])
    if not skip:
        retval.append(dates[n-1])
    return retval

# Returns a copy of floats, limiting precision for easier comparison.
def rounded(floats, places):
    shift = 10 ** places
    return [round(shift * f) / shift for f in floats]

####

peri_date, peri_dist = sfs.find_minima(t0, t1, sun_earth_dist)
apo_date, apo_dist = sfs.find_maxima(t0, t1, sun_earth_dist)

ephem_dates = np.empty(2 * nyr)
ephem_dates[0::2] = dedup([p.tt for p in peri_date])
ephem_dates[1::2] = dedup([a.tt for a in apo_date])

k0 = np.array([yr0 - 2000 + i / 2. for i in range(2 * nyr)])

print('Published coefficients:')
c0 = [328.41, 316.13, 346.20, 136.95, 249.52]
b0 = [132.788585, 584.903153, 450.380738, 659.306737, 329.653368]
a0p = [1.278, -0.055, -0.091, -0.056, -0.045]
a0a = [-1.352, 0.061, 0.062, 0.029, 0.031]
print(c0)
print(b0)
print(a0p)
print(a0a)

approx0_dates = meeus2(k0, \
        a0p[0], a0p[1], a0p[2], a0p[3], a0p[4], \
        a0a[0], a0a[1], a0a[2], a0a[3], a0a[4], \
        b0[0], b0[1], b0[2], b0[3], b0[4], \
        c0[0], c0[1], c0[2], c0[3], c0[4])
rms0 = 24 * sqrt(np.average(np.square(approx0_dates - ephem_dates)))
print('RMS error [{0}-{1}]: {2:5.3f} hours'.format(yr0, yr1, rms0))

print()
print('Fitted coefficients:')
abc0 = a0p + a0a + b0 + c0
abc1, covar = opt.curve_fit(meeus2, k0, ephem_dates, p0=abc0)
a1p = abc1[0:5]
a1a = abc1[5:10]
b1 = abc1[10:15]
c1 = abc1[15:20]
print(rounded(c1, 2))
print(rounded(b1, 6))
print(rounded(a1p, 3))
print(rounded(a1a, 3))

approx1_dates = meeus2(k0, \
        a1p[0], a1p[1], a1p[2], a1p[3], a1p[4], \
        a1a[0], a1a[1], a1a[2], a1a[3], a1a[4], \
        b1[0], b1[1], b1[2], b1[3], b1[4], \
        c1[0], c1[1], c1[2], c1[3], c1[4])
rms1 = 24 * sqrt(np.average(np.square(approx1_dates - ephem_dates)))
print('RMS error [{0}-{1}]: {2:5.3f} hours'.format(yr0, yr1, rms1))

