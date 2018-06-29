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


def _check_fft_exists(func):
    # Decorator for checking that _fft and _ifft exist.
    def check_fft(self, *args, **kwargs):
        if '_fft' in self.__dict__ and '_ifft' in self.__dict__:
            return func(self, *args, **kwargs)
        raise NotImplementedError('Fourier transform functions have not '
                                  'been linked; run self.setup first.')
    return check_fft


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

    data_format = {'time_shape': None,
                   'time_dtype': None,
                   'fourier_shape': None,
                   'fourier_dtype': None}
    _axes = None
    _norm = None

    def setup(self, a, A, axes=-1, norm=None):
        """Basic transform setup, which just stores transform axes and
        normalization convention.
        """
        # Extract information if user passed actual arrays.
        if isinstance(a, np.ndarray):
            a = {'shape': a.shape, 'dtype': a.dtype.name}
        if isinstance(A, np.ndarray):
            A = {'shape': A.shape, 'dtype': A.dtype.name}
        # Store time and Fourier domain array shapes (for both FFTs and repr).
        self.data_format = {'time_shape': a['shape'],
                            'time_dtype': a['dtype'],
                            'fourier_shape': A['shape'],
                            'fourier_dtype': A['dtype']}

        # If axes is an integer, convert to a 1-element tuple.
        try:
            axes = operator.index(axes)
        except TypeError:
            axes = tuple(axes)
        else:
            axes = (axes,)
        assert len(axes) > 0, "must transform over one or more axes!"
        self._axes = axes
        self._norm = norm if norm == 'ortho' else None

    @property
    def axes(self):
        """Axes over which to perform the FFT (same as for `numpy.fft`)."""
        return self._axes

    @property
    def norm(self):
        """Normalization convention (same as for `numpy.fft`).

        `None` is an unscaled forward transform and 1 / n scaled inverse
        transform, and 'ortho' is a 1 / sqrt(n) scaling for both.
        """
        return self._norm

    def fft(self, a):
        """Placeholder for forward FFT."""
        raise NotImplementedError()

    def ifft(self, a):
        """Placeholder for inverse FFT."""
        raise NotImplementedError()

    def fftfreq(self, a_length, sample_rate=None, positive_freqs_only=False):
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
            Length of array being transformed.
        sample_rate : `~astropy.units.Quantity`, optional
            Sample rate.  If `None` (default), output is unitless.
        positive_freqs_only : bool, optional
            Whether to return only the positive frequencies.  Default: `False`.

        Returns
        -------
        freqs : `~numpy.ndarray`
            Fourier-transformed input.
        """
        if sample_rate is None:
            sample_rate = 1.
        if positive_freqs_only:
            return np.fft.rfftfreq(operator.index(a_length)) * sample_rate
        return np.fft.fftfreq(operator.index(a_length)) * sample_rate

    def __repr__(self):
        return ("<{s.__class__.__name__} time_shape={fmt[time_shape]},"
                " time_dtype={fmt[time_dtype]}\n"
                "    fourier_shape={fmt[fourier_shape]},"
                " fourier_dtype={fmt[fourier_dtype]}\n"
                "    axes={s.axes}, norm={s.norm}>"
                .format(s=self, fmt=self.data_format))


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

    def setup(self, a, A, axes=(-1,), norm=None):
        # Store axes and norm.
        super().setup(a, A, axes=axes, norm=norm)

        complex_data = 'complex' in self.data_format['fourier_dtype']

        # Select the forward and backward FFT functions to use.
        if len(self.axes) == 1:
            if complex_data:
                def fft(a):
                    return np.fft.fftn(a, axes=self.axes, norm=self.norm)

                def ifft(A):
                    return np.fft.ifftn(A, axes=self.axes, norm=self.norm)

            else:
                def fft(a):
                    return np.fft.rfftn(a, axes=self.axes, norm=self.norm)

                def ifft(A):
                    return np.fft.irfftn(A, axes=self.axes, norm=self.norm)

        else:
            if complex_data:
                def fft(a):
                    return np.fft.fft(a, axis=self.axes[0], norm=self.norm)

                def ifft(A):
                    return np.fft.ifft(A, axis=self.axes[0], norm=self.norm)

            else:
                def fft(a):
                    return np.fft.rfft(a, axis=self.axes[0], norm=self.norm)

                def ifft(A):
                    return np.fft.irfft(A, axis=self.axes[0], norm=self.norm)

        self._fft = fft
        self._ifft = ifft

    @_check_fft_exists
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
        return self._fft(a)

    @_check_fft_exists
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
        return self._ifft(a)


class PyfftwFFT(FFTBase):

    def __init__(self, n_simd=None, **kwargs):
        # Set n-byte boundary.
        if n_simd is None:
            n_simd = pyfftw.simd_alignment
        self._n_simd = n_simd

        if 'flags' in kwargs and 'FFTW_DESTROY_INPUT' in kwargs['flags']:
            raise ValueError('Fourier module does not support destroying '
                             'input arrays.')

        self._kwargs = kwargs

        super().__init__()

    def setup(self, a, A, axes=(-1,), norm=None):
        # Store axes and norm.
        super().setup(a, A, axes=axes, norm=norm)

        # Set up normalization keywords.
        if self.norm == 'ortho':
            self._normalise_idft = False
            self._ortho = True
        else:
            self._normalise_idft = True
            self._ortho = False

        # Create empty, byte-aligned arrays.  These will be stored as
        # self._fft.input_array, self._fft.output_array, etc., but we'll be
        # replacing the input and output arrays each time we use a transform.
        a = pyfftw.empty_aligned(self.data_format['time_shape'],
                                 dtype=self.data_format['time_dtype'],
                                 n=self._n_simd)
        A = pyfftw.empty_aligned(self.data_format['fourier_shape'],
                                 dtype=self.data_format['fourier_dtype'],
                                 n=self._n_simd)

        # Create forward and backward transforms.
        self._fft = pyfftw.FFTW(a, A, axes=self.axes, direction='FFTW_FORWARD',
                                **self._kwargs)
        self._ifft = pyfftw.FFTW(A, a, axes=self.axes,
                                 direction='FFTW_BACKWARD', **self._kwargs)

    @_check_fft_exists
    def fft(self, a):
        # Make an empty array to store transform output.
        A = pyfftw.empty_aligned(self.data_format['fourier_shape'],
                                 dtype=self.data_format['fourier_dtype'],
                                 n=self._n_simd)
        # A is returned by self._fft.
        return self._fft(input_array=a, output_array=A,
                         normalise_idft=self._normalise_idft,
                         ortho=self._ortho)

    @_check_fft_exists
    def ifft(self, A):
        # Multi-dimensional real transforms destroy their input arrays, so
        # make a (if necessary, byte-aligned) copy.
        # See https://pyfftw.readthedocs.io/en/latest/source/pyfftw/pyfftw.html#scheme-table
        if self._real_transform and self._axes_ndim > 1:
            A_copy = pyfftw.byte_align(A)
            # If A is already byte-aligned, this doesn't copy anything, so:
            if A_copy is A:
                A_copy = A.copy()
        else:
            A_copy = A
        # Make an empty array to store transform output.
        a = pyfftw.empty_aligned(self.data_format['time_shape'],
                                 dtype=self.data_format['time_dtype'],
                                 n=self._n_simd)
        # a is returned by self._fft.
        return self._ifft(input_array=A_copy, output_array=a,
                          normalise_idft=self._normalise_idft,
                          ortho=self._ortho)
