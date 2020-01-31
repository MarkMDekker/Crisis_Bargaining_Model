# --------------------------------------------------------------------------- #
# Preambule
# --------------------------------------------------------------------------- #

import configparser
import numpy as np
from Functions import p1, p2, p3

# --------------------------------------------------------------------------- #
# Class object
# --------------------------------------------------------------------------- #


class CrisisBargainingModel(object):
    def __init__(self):
        ''' Initial function '''

        config = configparser.ConfigParser()
        config.read('config.ini')

        ''' True values '''
        self.aS = np.float(config['PAYOFFS_TRUE']['AUDIENCECOST_A_S'])
        self.aT = np.float(config['PAYOFFS_TRUE']['AUDIENCECOST_A_T'])
        self.bS = np.float(config['PAYOFFS_TRUE']['AUDIENCECOST_B_S'])
        self.bT = np.float(config['PAYOFFS_TRUE']['AUDIENCECOST_B_T'])
        self.cS = np.float(config['PAYOFFS_TRUE']['ECONONOMICCOSTS_S'])
        self.cT = np.float(config['PAYOFFS_TRUE']['ECONONOMICCOSTS_T'])

        '''
        One global standard deviation
        -> only used for opponent's payoffs and own economic costs
        '''
        self.STD = np.float(config['PAYOFFS_UNCERTAINTY']['GLOBAL_STD'])

        ''' Convert into usable list of true values '''
        self.pars_true = [self.aS,
                          self.aT,
                          self.bS,
                          self.bT,
                          self.cS,
                          self.cT]

        ''' Parameters seen by sender '''
        self.pars_s = [[self.aS, 0],
                       [self.aT, self.STD],
                       [self.bS, 0],
                       [self.bT, self.STD],
                       [self.cS, self.STD],
                       [self.cT, self.STD]]

        ''' Parameters seen by target '''
        self.pars_t = [[self.aS, self.STD],
                       [self.aT, 0],
                       [self.bS, self.STD],
                       [self.bT, 0],
                       [self.cS, self.STD],
                       [self.cT, self.STD]]

    def single_run(self):
        ''' Function to do a single run with the given parameters '''
        print('p1                  :', np.round(p1(self.pars_s), 3))
        print('p2                  :', np.round(p2(self.pars_t), 3))
        print('p3                  :', np.round(p3(self.pars_s), 3))
        print('')
        print('Status quo          :', np.round(1-p1(self.pars_s), 3))
        print('Concession target   :', np.round(p1(self.pars_s) *
                                                (1-p2(self.pars_t)), 3))
        print('Backing down sender :', np.round(p1(self.pars_s) *
                                                p2(self.pars_t) *
                                                (1-p3(self.pars_s)), 3))
        print('Sanction imposed    :', np.round(p1(self.pars_s) *
                                                p2(self.pars_t) *
                                                (p3(self.pars_s)), 3))
