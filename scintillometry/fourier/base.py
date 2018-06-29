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
from functools import wraps


def _check_fft_exists(func):
    # Decorator for checking that _fft and _ifft exist.
    @wraps(func)
    def check_fft(self, *args, **kwargs):
        if '_fft' in self.__dict__ and '_ifft' in self.__dict__:
            return func(self, *args, **kwargs)
        raise AttributeError('Fourier transform functions have not been '
                             'linked; run self.setup first.')
    return check_fft


class FFTBase(object):
    """Base class for all fast Fourier transforms.

    Provides a `setup` method to store relevant transform information, and a
    `fftfreqs` method to return the sample frequencies of the FFT along one
    axis.

    Actual FFT classes must define a forward and backward transform as `fft`
    and `ifft`, respectively.  They must also support setting and handling the
    `axes` property, the axes over which to perform the FFT, and the `norm`,
    property, which sets the normalization convention.

    Currently does not support Hermitian FFTs (where data is real in Fourier
    space).
    """

    _data_format = {'time_shape': None,
                    'time_dtype': None,
                    'fourier_shape': None,
                    'fourier_dtype': None}
    _axes = None
    _norm = None

    def setup(self, a, A, axes=None, norm=None, verify=True):
        """Store information about arrays, transform axes, and normalization.

        Parameters
        ----------
        a : numpy.ndarray or dict
            Dummy array with the dimensions and dtype of data in the time
            domain.  Can alternatively give a dict with 'shape' and 'dtype'.
        A : numpy.ndarray or dict
            Dummy array with the dimensions and dtype of data in the Fourier
            domain.  Can alternatively give a dict with 'shape' and 'dtype'.
        axes : int, tuple or None, optional
            Axis or axes to transform, as with ``axes`` for `numpy.fft.fftn`.
            If `None` (default), all axes are used.
        norm : 'ortho' or None, optional
            If `None` (default), uses an unscaled forward transform and 1 / n
            scaled inverse transform, and 'ortho' is a 1 / sqrt(n) scaling for
            both.
        verify : bool, optional
            Verify setup is successful and self-consistent.
        """
        # Extract information if user passed actual arrays.
        if isinstance(a, np.ndarray):
            a = {'shape': a.shape, 'dtype': a.dtype.name}
        if isinstance(A, np.ndarray):
            A = {'shape': A.shape, 'dtype': A.dtype.name}
        # Store time and Fourier domain array shapes (for both FFTs and repr).
        self._data_format = {'time_shape': a['shape'],
                             'time_dtype': a['dtype'],
                             'fourier_shape': A['shape'],
                             'fourier_dtype': A['dtype']}

        # If axes is None, cycle through all axes (like with numpy.fft).
        if axes is None:
            axes = tuple(range(len(a['shape'])))
        else:
            # Otherwise, if axes is an integer, convert to a 1-element tuple.
            try:
                axes = operator.index(axes)
            # If not, typecast to a tuple.
            except TypeError:
                axes = tuple(axes)
            else:
                axes = (axes,)
        assert len(axes) > 0, "must transform over one or more axes!"
        self._axes = axes
        self._norm = norm if norm == 'ortho' else None

        if verify:
            self.verify()

    def verify(self):
        """Verify setup is successful and self-consistent."""
        # Check data is complex in Fourier space.
        assert 'complex' in self.data_format['fourier_dtype'], (
            "array for Fourier domain must be complex.")
        # Check that the time and Fourier domain arrays make sense.
        expected_shape = list(self.data_format['time_shape'])
        # If data is real in time domain, halve the relevant axis in Fourier.
        if 'float' in self.data_format['time_dtype']:
            expected_shape[self.axes[-1]] = (
                expected_shape[self.axes[-1]] // 2 + 1)
        assert tuple(expected_shape) == self.data_format['fourier_shape'], (
            "time domain array of shape {df[time_shape]} cannot "
            "be transformed to fourier domain array of shape "
            "{df[fourier_shape]}.".format(df=self.data_format))

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

    @property
    def data_format(self):
        """Shapes and dtypes of arrays expected by FFT.

        'time_' entries are for time domain arrays, and 'fourier_' for Fourier
        domain ones.
        """
        return self._data_format

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

    def setup(self, a, A, axes=(-1,), norm=None, verify=True):
        """Set up FFT.

        Parameters
        ----------
        a : numpy.ndarray or dict
            Dummy array with the dimensions and dtype of data in the time
            domain.  Can alternatively give a dict with 'shape' and 'dtype'.
        A : numpy.ndarray or dict
            Dummy array with the dimensions and dtype of data in the Fourier
            domain.  Can alternatively give a dict with 'shape' and 'dtype'.
        axes : int, tuple or None, optional
            Axis or axes to transform, as with ``axes`` for `numpy.fft.fftn`.
            If `None` (default), all axes are used.
        norm : 'ortho' or None, optional
            If `None` (default), uses an unscaled forward transform and 1 / n
            scaled inverse transform, and 'ortho' is a 1 / sqrt(n) scaling for
            both.
        verify : bool, optional
            Verify setup is successful and self-consistent.
        """
        super().setup(a, A, axes=axes, norm=norm, verify=verify)

        complex_data = 'complex' in self.data_format['time_dtype']

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
        """FFT, using the `numpy.fft` functions.

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
        """Inverse FFT, using the `numpy.fft` functions.

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
