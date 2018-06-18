import numpy as np
import astropy.units as u
from .dm import DispersionMeasure
from astropy.extern.configobj import configobj
import sys
sys.path.append('/home/cczhu/temp/baseband/')
from baseband import vdif, mark4, mark5b, dada

class ReadDD:
    """Reader class to return de-dispersed data directy from raw binaries

    Requires a configuration file with data type and frequency information
    of your observation

    Parameters
    ----------
    fname: filename of binary file to be opened
    obsfile: filename of config file for observation
    requires
        srate : int 
            sample_rate in MHz
        dtype : string
            one of 'vdif', 'mark4', 'mark5b', 'dada'
        dm : float
            dispersion measure, in pc/cm**3
        threads:
            IFs to be read
        fedge: list, int
            edge frequencies of IFs in MHz
    optional
        forder: list, int
            list of 1 or -1 indicating upper or lower sideband
            default assumption is upper
        ntrack: int
            ntrack parameter for mark4 data, default = 64
        nIF: int
            total number of IFs, required to open mark5b data
    size: int
        number of samples to read
    """

    def __init__(self, fname=None, obsfile=None, size=2**25):

        obs = {}
        conf = configobj.ConfigObj(r"{0}".format(obsfile))

        for key, val in conf.iteritems():