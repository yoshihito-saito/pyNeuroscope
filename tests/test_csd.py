import numpy as np
import pytest

from pyneuroscope.csd import CSD_COLORMAPS, robust_csd_limits, standard_1d_csd


def test_standard_1d_csd_constant_and_linear_are_zero() -> None:
    constant = np.ones((5, 6))
    linear = np.tile(np.arange(6, dtype=float), (5, 1))

    assert np.allclose(standard_1d_csd(constant, range(6)), 0.0)
    assert np.allclose(standard_1d_csd(linear, range(6)), 0.0)


def test_standard_1d_csd_quadratic_has_expected_sign() -> None:
    channels = np.arange(6, dtype=float)
    quadratic = np.tile(channels**2, (4, 1))

    csd = standard_1d_csd(quadratic, range(6), subtract_channel_mean=False)

    assert csd.shape == (4, 4)
    assert np.allclose(csd, -2.0)


def test_standard_1d_csd_uses_nonuniform_depth_spacing() -> None:
    depths = np.asarray([0.0, 1.0, 3.0, 6.0])
    normalized = depths / np.median(np.diff(depths))
    quadratic = np.tile(normalized**2, (3, 1))

    csd = standard_1d_csd(
        quadratic,
        range(4),
        depths=depths,
        subtract_channel_mean=False,
    )

    assert csd.shape == (3, 2)
    assert np.allclose(csd, -2.0)


def test_standard_1d_csd_rejects_invalid_depths() -> None:
    data = np.zeros((3, 3))

    with pytest.raises(ValueError):
        standard_1d_csd(data, [0, 1, 2], depths=[0.0, 1.0, 1.0])


def test_standard_1d_csd_validates_channels() -> None:
    data = np.zeros((3, 2))

    with pytest.raises(ValueError):
        standard_1d_csd(data, [0, 1, 2])


def test_csd_colormap_options_default_to_bwr() -> None:
    assert CSD_COLORMAPS[0] == "bwr"
    assert CSD_COLORMAPS == (
        "bwr",
        "PiYG",
        "PRGn",
        "BrBG",
        "PuOr",
        "RdGy",
        "RdBu",
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
    )


def test_robust_csd_limits_are_symmetric() -> None:
    low, high = robust_csd_limits(np.asarray([[-1.0, 0.0, 2.0]]), percentile=100.0)

    assert (low, high) == (-2.0, 2.0)
