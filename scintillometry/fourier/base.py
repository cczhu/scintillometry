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

    def __init__(self, axes=-1, complex_data=True, norm='ortho', **kwargs):
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

        Can be either `None` (unscaled forward transform, 1/n scaled inverse
        transform) or 'ortho' (1/sqrt(n) scaled for both).
        """
        return self._norm

    def fft(self, a):
        """Placeholder for forward FFT.

        Parameters
        ----------
        a : array_like
            Input array, can be complex.
        axis : int or list, optional
            Axis over which to compute the FFT.  If not given, the last axis
            is used.

        Returns
        -------
        out : `~numpy.ndarray`
            Fourier-transformed input.
        """
        pass

    def ifft(self, a, axes=-1):
        """Placeholder for inverse FFT.

        Parameters
        ----------
        a : array_like
            Input array, can be complex.
        axis : int or list, optional
            Axis over which to compute the iFFT.  If not given, the last axis
            is used.

        Returns
        -------
        out : `~numpy.ndarray`
            Fourier-transformed input.
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
        functions.
    """

    def __init__(self, axes=-1, complex_data=True, norm='ortho', **kwargs):
        super().__init__(axes=axes, complex_data=complex_data, norm=norm)
        # Get the number of axes to do FFT over.
        try:
            axes_ndim = len(axes)
        except TypeError:
            axes_ndim = 1
        else:
            assert axes_ndim > 0
            # If we pass a length 1 array-like, extract its element.
            if axes_ndim == 1:
                axes = axes[0]
        # Select the forward and backward FFT functions to use.
        if self.complex_data and axes_ndim > 1:
            self._fft = np.fft.fftn
            self._ifft = np.fft.ifftn
        elif not self.complex_data and axes_ndim > 1:
            self._fft = np.fft.rfftn
            self._ifft = np.fft.irfftn
        elif self.complex_data:
            self._fft = np.fft.fft
            self._ifft = np.fft.ifft
        else:
            self._fft = np.fft.rfft
            self._ifft = np.fft.irfft

    def fft(self, a):
        """FFT, using the `numpy.fft` functions.

        Parameters
        ----------
        a : array_like
            Input array, can be complex.
        axis : int or list, optional
            Axis over which to compute the iFFT.  If not given, the last axis
            is used.

        Returns
        -------
        out : `~numpy.ndarray`
            Fourier-transformed input.
        """
        try:
            axes = operator.index(axes)
        except:
            return self._fft()