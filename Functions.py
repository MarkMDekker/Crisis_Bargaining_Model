# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import scipy.stats
import numpy as np

# ----------------------------------------------------------------- #
# Define functions
# ----------------------------------------------------------------- #


def p3(pars):
    aS, aT, bS, bT, cS, cT = pars
    return 1-scipy.stats.norm(0, np.sqrt(cS[1]**2+bS[1]**2)).cdf(-1-bS[0]+cS[0])

def Es3(pars):
    aS, aT, bS, bT, cS, cT = pars
    p3_ = p3(pars)
    return [p3_*(-cS[0]) + (1-p3_)*(-1-bS[0]),np.sqrt(cS[1]**2+bS[1]**2)]

def Et3(pars):
    aS, aT, bS, bT, cS, cT = pars
    p3_ = p3(pars)
    return [p3_*(-cT[0]) + (1-p3_)*(1+bT[0]),np.sqrt(cT[1]**2+bT[1]**2)]

def p2(pars):
    aS, aT, bS, bT, cS, cT = pars
    Et3_ = Et3(pars)
    return 1-scipy.stats.norm(0, np.sqrt(aT[1]**2+Et3_[1]**2)).cdf(-aT[0]-Et3_[0])

def Es2(pars):
    aS, aT, bS, bT, cS, cT = pars
    p2_ = p2(pars)
    Es3_ = Es3(pars)
    return [p2_*(Es3_[0])+(1-p2_)*(aS[0]),np.sqrt(aS[1]**2+Es3_[1]**2)]

def Et2(pars):
    aS, aT, bS, bT, cS, cT = pars
    p2_ = p2(pars)
    Et3_ = Et3(pars)
    return [p2_*(Et3_[0])+(1-p2_)*(-aT[0]),np.sqrt(aT[1]**2+Et3_[1]**2)]

def p1(pars):
    aS, aT, bS, bT, cS, cT = pars
    Es2_ = Es2(pars)
    return 1-scipy.stats.norm(0, Es2_[1]).cdf(-1-Es2_[0])
