# Licensed under the GPLv3 - see LICENSE

import pyfftw
from .base import FFTBase, _check_fft_exists


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

    def setup(self, a, A, axes=(-1,), norm=None, verify=True):
        """Set up FFT (including FFTW planning).

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
