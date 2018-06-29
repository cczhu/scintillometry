# Licensed under the GPLv3 - see LICENSE

import numpy as np
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
import pytest

from ...fourier import FFTBase, NumpyFFT

FFT_CLASSES = (NumpyFFT,)

# If pyfftw is available, import PyfftwFFT.
try:
    from ...fourier import PyfftwFFT
except ImportError:
    pass
else:
    FFT_CLASSES += (PyfftwFFT,)


class TestFFTBase(object):
    """Class for testing FFTBase's initialization and frequency generator."""

    def test_setup(self):
        """Check that we can set up properties, and they can't be reset."""
        fft = FFTBase()
        a = np.empty((100, 10), dtype='float')
        A = np.empty((100, 6), dtype='complex128')
        fft.setup(a, A, axes=(0, 1), norm='ortho')
        assert fft.axes == (0, 1)
        assert fft.norm == 'ortho'
        assert fft.data_format['time_shape'] == (100, 10)
        assert fft.data_format['time_dtype'] == 'float64'
        assert fft.data_format['fourier_shape'] == (100, 6)
        assert fft.data_format['fourier_dtype'] == 'complex128'
        with pytest.raises(AttributeError):
            fft.norm = None
        # -1 will be converted to a tuple, and normalization to None.
        fft.setup({'shape': (100, 10), 'dtype': 'float64'},
                  {'shape': (100, 6), 'dtype': 'complex128'},
                  axes=-1, norm='not ortho')
        assert fft.axes == (-1,)
        assert fft.norm is None
        assert fft.data_format['time_shape'] == (100, 10)
        assert fft.data_format['time_dtype'] == 'float64'
        assert fft.data_format['fourier_shape'] == (100, 6)
        assert fft.data_format['fourier_dtype'] == 'complex128'
        assert repr(fft).startswith('<FFTBase')

        # Check verification can be turned off.
        fft.setup({'shape': (100, 10), 'dtype': 'float64'},
                  {'shape': (100, 10), 'dtype': 'float64'},
                  verify=False)
        # The above should fail with verify=True.
        with pytest.raises(AssertionError) as excinfo:
            fft.setup({'shape': (100, 10), 'dtype': 'float64'},
                      {'shape': (100, 10), 'dtype': 'float64'},
                      verify=True)
        assert 'must be complex' in str(excinfo)
        # Giving incompatible arrays should also fail.
        with pytest.raises(AssertionError) as excinfo:
            fft.setup({'shape': (100, 10), 'dtype': 'float64'},
                      {'shape': (100, 10), 'dtype': 'complex128'},
                      verify=True)
        assert 'cannot be transformed' in str(excinfo)

    @pytest.mark.parametrize(
        ('nsamp', 'sample_rate'), [(1337, 100. * u.Hz),
                                   (9400, 89. * u.MHz),
                                   (12, 5.2 * u.GHz),
                                   (10000, 23.11)])
    def test_fftfreq(self, nsamp, sample_rate):
        """Test FFT sample frequency generation - including with units."""
        fft = FFTBase()
        fftbasefreqs = fft.fftfreq(nsamp, sample_rate=sample_rate)
        # Override default behaviour and return complex sample frequencies.
        fftbaserealfreqs = fft.fftfreq(nsamp, sample_rate=sample_rate,
                                       positive_freqs_only=True)
        if isinstance(sample_rate, u.Quantity):
            unit = sample_rate.unit
            sample_rate = sample_rate.value
        else:
            unit = None
        npfreqs = np.fft.fftfreq(nsamp, d=(1. / sample_rate))
        # Only test up to nsamp // 2 in the real case, since the largest
        # absolute frequency is positive for rfftfreq, but negative for
        # fftfreq.
        if unit is None:
            assert np.allclose(fftbasefreqs, npfreqs, rtol=1e-14, atol=0.)
            assert np.allclose(fftbaserealfreqs[:-1], npfreqs[:nsamp // 2],
                               rtol=1e-14, atol=0.)
        else:
            assert_quantity_allclose(fftbasefreqs, npfreqs * unit, rtol=1e-14)
            assert_quantity_allclose(
                fftbaserealfreqs[:-1], npfreqs[:nsamp // 2] * unit, rtol=1e-14)


class TestFFTClasses(object):

    def setup(self):
        x = np.linspace(0., 10., 10000)
        # Simple 1D complex sinusoid.
        self.y_exp = np.exp(1.j * 2. * np.pi * x)
        # Simple 1D real sinusoid.
        self.y_rsine = np.sin(2. * np.pi * x)
        # More complex 2D transform.
        self.y_r2D = np.random.uniform(low=-13., high=29., size=(100, 10, 30))
        self.axes_r2D = (0, 1)
        # More complex 3D transform.
        self.y_3D = (np.random.uniform(low=-13., high=29.,
                                       size=(100, 10, 30)) +
                     1.j * np.random.uniform(low=-13., high=29.,
                                             size=(100, 10, 30)))

        # Transforms to be checked against.
        self.Y_exp = np.fft.fft(self.y_exp)
        self.Y_rsine = np.fft.rfft(self.y_rsine)
        self.Y_r2D = np.fft.rfftn(self.y_r2D)
        self.axes_0only = (0,)
        self.Y_r2D_0only = np.fft.rfftn(self.y_r2D, axes=self.axes_0only)
        self.Y_3D = np.fft.fftn(self.y_3D)
        self.axes_3D_12only = (1, 2)
        self.Y_3D_12only = np.fft.fftn(self.y_3D, axes=self.axes_3D_12only)

    @pytest.mark.parametrize('FFTClass', FFT_CLASSES)
    def test_fft(self, FFTClass):
        """Test various FFT implementations, all of which have the same
        interface, against numpy.fft.
        """
        # In the future, may wish to have a custom initialization as a second
        fft = FFTClass()
        assert FFTClass.__name__ in repr(fft)

        # If we haven't set anything up, _fft and _ifft should be None.
        with pytest.raises(AttributeError) as excinfo:
            fft.fft(np.arange(3))
        assert 'have not been linked' in str(excinfo)
        with pytest.raises(AttributeError):
            fft.ifft(np.arange(3))
        assert 'have not been linked' in str(excinfo)

        # 1D real transform.
        fft.setup({'shape': y_r2D.shape, 'dtype': y_r2D.dtype.name},
                  {'shape': (100, 13), 'dtype': 'complex128'})