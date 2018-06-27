# Licensed under the GPLv3 - see LICENSE

# import io
# import warnings
# import numpy as np
# import operator
# from collections import namedtuple
# import astropy.units as u
# from astropy.utils import lazyproperty

import numpy as np
import operator
import pyfftw


class FFTBase(object):
    """Base class for all fast Fourier transforms.

    Provides an `fftfreqs` method to return the sample frequencies of the FFT.

    Actual FFT classes must define a forward and backward transform as `fft`
    and `ifft`, respectively.  They must also support setting and handling the
    `axes` property, the axes over which to perform the FFT, the `complex_data`
    property, which indicates whether data in the time domain is complex or
    completely real, and the `norm` property, which sets the normalization
    convention.
    """

    def __init__(self, axes=-1, complex_data=True, norm=None):
        self._axes = axes
        self._complex_data = complex_data
        self._norm = norm

    @property
    def axes(self):
        """Axes over which to perform the FFT (same as for `numpy.fft`)."""
        return self._axes

    @property
    def complex_data(self):
        """Whether the data are completely real in the time domain."""
        return self._complex_data

    @property
    def norm(self):
        """Normalization convention (same as for `numpy.fft`).

        `None` is an unscaled forward transform and 1 / n scaled inverse
        transform, and 'ortho' is a 1 / sqrt(n) scaling for both.
        """
        return self._norm

    def fft(self, a):
        """Placeholder for forward FFT.

        Parameters
        ----------
        a : array_like
            Data in time domain.

        Returns
        -------
        out : `~numpy.ndarray`
            Fourier transformed input.
        """
        pass

    def ifft(self, a):
        """Placeholder for inverse FFT.

        Parameters
        ----------
        a : array_like
            Data in frequency domain.

        Returns
        -------
        out : `~numpy.ndarray`
            Inverse transformed input.
        """
        pass

    def fftfreq(self, a_length, sample_rate=None, positive_freqs_only=None):
        """Obtains FFT sample frequencies.

        Uses `numpy.fft.fftfreq` or `numpy.fft.rfftfreq`.  As with those, given
        a window of length ``a_length`` and sample rate ``sample_rate``,

            freqs  = [0, 1, ...,   a_length/2-1, -a_length/2,
                      ..., -1] * sample_rate

        if a_length is even

            freqs  = [0, 1, ..., (a_length-1)/2, -(a_length-1)/2,
                      ..., -1] * sample_rate

        if a_length is odd.

        Parameters
        ----------
        a_length : int
            Independent variable of data in the time domain.
        sample_rate : `~astropy.units.Quantity`, optional
            Sample rate.  If `None` (default), output is unitless.
        positive_freqs_only : bool or None, optional
            Whether to return only the positive frequencies.  If `None`
            (default), uses the `complex_data` property.

        Returns
        -------
        freqs : `~numpy.ndarray`
            Fourier-transformed input.
        """
        if positive_freqs_only is None:
            positive_freqs_only = not self.complex_data
        if sample_rate is None:
            sample_rate = 1.
        if positive_freqs_only:
            return np.fft.rfftfreq(operator.index(a_length)) * sample_rate
        return np.fft.fftfreq(operator.index(a_length)) * sample_rate


class NumpyFFT(FFTBase):
    """Numpy FFT class, which wraps `numpy.fft` functions.

    Parameters
    ----------
    complex_data : bool, optional
        Whether data in the time domain is complex or completely real.
        Default: `True`.
    norm : 'ortho' or `None`, optional
        Normalization convention; options are identical as for `numpy.fft`
        functions.  Default: `None`.
    """

    def __init__(self, axes=-1, complex_data=True, norm=None, **kwargs):
        # Get the number of axes to do FFT over.
        try:
            self._transform_ndim = len(axes)
        except TypeError:
            self._transform_ndim = 1
        else:
            assert self._transform_ndim > 0
            # If we pass a length 1 array-like, extract its element.
            if self._transform_ndim == 1:
                axes = axes[0]
        super().__init__(axes=axes, complex_data=complex_data, norm=norm)
        # Select the forward and backward FFT functions to use.
        if self.complex_data:
            self._fft = np.fft.fft
            self._ifft = np.fft.ifft
            self._fftn = np.fft.fftn
            self._ifftn = np.fft.ifftn
        else:
            self._fft = np.fft.rfft
            self._ifft = np.fft.irfft
            self._fftn = np.fft.rfftn
            self._ifftn = np.fft.irfftn

    def fft(self, a):
        """Fourier transform, using the `numpy.fft` functions.

        Parameters
        ----------
        a : array_like
            Data in time domain.

        Returns
        -------
        out : `~numpy.ndarray`
            Fourier transformed input.
        """
        if self._transform_ndim > 1:
            return self._fft(a, axes=self.axes, norm=self.norm)
        return self._fft(a, axis=self.axes, norm=self.norm)

    def ifft(self, a):
        """Inverse Fourier transform, using the `numpy.fft` functions.

        Parameters
        ----------
        a : array_like
            Data in frequency domain.

        Returns
        -------
        out : `~numpy.ndarray`
            Inverse transformed input.
        """
        if self._transform_ndim > 1:
            return self._ifft(a, axes=self.axes, norm=self.norm)
        return self._ifft(a, axis=self.axes, norm=self.norm)


class pyfftwFFT(FFTBase):

    def __init__(self, axes=-1, complex_data=True, norm=None, **kwargs):