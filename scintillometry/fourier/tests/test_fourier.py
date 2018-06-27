# Licensed under the GPLv3 - see LICENSE

import numpy as np
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
import pytest

from ...fourier import FFTBase, NumpyFFT


class TestFFTBase(object):
    """Class for testing FFTBase's initialization and frequency generator."""

    def test_properties(self):
        """Somewhat silly - just check that we can initialize properties, and
        they can't be reset.
        """
        fft = FFTBase()
        assert fft.complex_data
        assert fft.norm == 'ortho'
        with pytest.raises(AttributeError):
            fft.complex_data = False
        with pytest.raises(AttributeError):
            fft.complex_data = False

    @pytest.mark.parametrize(
        ('nsamp', 'sample_rate'), [(1337, 100. * u.Hz),
                                   (9400, 89. * u.MHz),
                                   (12, 5.2 * u.GHz),
                                   (10000, 23.11)])
    def test_fftfreq(self, nsamp, sample_rate):
        """Test FFT sample frequency generation - including with units."""
        fft = FFTBase(complex_data=False)
        fftbaserealfreqs = fft.fftfreq(nsamp, sample_rate=sample_rate)
        # Override default behaviour and return complex sample frequencies.
        fftbasefreqs = fft.fftfreq(nsamp, sample_rate=sample_rate,
                                   positive_freqs_only=False)
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
        # More complex 3D transform.
        self.y_3D = (np.random.uniform(low=-13., high=29.,
                                       size=(100, 10, 30)) +
                     1.j * np.random.uniform(low=-13., high=29.,
                                             size=(100, 10, 30)))

        # Transforms to be checked against.
        self.Y_exp = np.fft.fft(self.y_exp)
        self.Y_rsine = np.fft.rfft(self.y_rsine)
        self.Y_r2D = np.fft.rfftn(self.y_r2D)
        self.Y_r2D_0only = np.fft.rfftn(self.y_r2D, axes=(0,))
        self.Y_3D = np.fft.fftn(self.y_3D)
        self.Y_3D_12only = np.fft.fftn(self.y_3D, axes=(1, 2))

    @pytest.mark.parametrize('FFTClass', ('NumpyFFT',))
    def test_fft(self, FFTClass, initdict):
        """Test various FFT implementations, all of which have the same
        interface, against numpy.fft.
        """
        fft = FFTClass(complex_data=False, **kwargs)
        #
