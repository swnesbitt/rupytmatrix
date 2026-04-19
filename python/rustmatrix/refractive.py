"""Refractive-index helpers.

Direct port of ``pytmatrix.refractive``. Provides:

* :func:`mg_refractive` — Maxwell-Garnett effective-medium approximation
  for a multi-component mixture.
* :func:`bruggeman_refractive` — Bruggeman EMA (two-component).
* Tabulated water refractive indices at 0/10/20 °C for the six standard
  radar bands (``m_w_0C``, ``m_w_10C``, ``m_w_20C``).
* :func:`ice_refractive` — factory that returns a callable ``m(wl, rho)``
  interpolating the refractive index of ice/snow against a lookup table.
* ``mi`` — pre-built interpolator using the bundled ``ice_refr.dat`` from
  http://www.atmos.washington.edu/ice_optical_constants/ .
"""

from __future__ import annotations

from os import path

import numpy as np
from scipy import interpolate

from .tmatrix_aux import wl_C, wl_Ka, wl_Ku, wl_S, wl_W, wl_X


def mg_refractive(m, mix):
    """Maxwell-Garnett effective-medium refractive index.

    Parameters
    ----------
    m : tuple of complex
        Complex refractive indices of the constituent media.
    mix : tuple of float
        Volume fractions, ``len(mix) == len(m)``. Renormalised to
        ``sum(mix) == 1`` if needed.

    Returns
    -------
    complex
        Effective complex refractive index of the mixture.

    Notes
    -----
    For two components, the first element is the matrix and the second is
    the inclusion — the approximation is asymmetric. For more components
    the media are mixed recursively from the tail inward.

    Examples
    --------
    Dry snow as ice inclusions in air (10 %% ice by volume):

    >>> mg_refractive((complex(1.0, 0.0), complex(1.78, 3e-4)), (0.9, 0.1))
    """
    if len(m) == 2:
        cF = float(mix[1]) / (mix[0] + mix[1]) * \
            (m[1] ** 2 - m[0] ** 2) / (m[1] ** 2 + 2 * m[0] ** 2)
        er = m[0] ** 2 * (1.0 + 2.0 * cF) / (1.0 - cF)
        return np.sqrt(er)

    m_last = mg_refractive(m[-2:], mix[-2:])
    mix_last = mix[-2] + mix[-1]
    return mg_refractive(m[:-2] + (m_last,), mix[:-2] + (mix_last,))


def bruggeman_refractive(m, mix):
    """Bruggeman effective-medium refractive index (two components only).

    Symmetric counterpart to :func:`mg_refractive`. Takes the same
    arguments but both media are treated equally.
    """
    f1 = mix[0] / sum(mix)
    f2 = mix[1] / sum(mix)
    e1 = m[0] ** 2
    e2 = m[1] ** 2
    a = -2 * (f1 + f2)
    b = (2 * f1 * e1 - f1 * e2 + 2 * f2 * e2 - f2 * e1)
    c = (f1 + f2) * e1 * e2
    e_eff = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return np.sqrt(e_eff)


# Water refractive indices at 0 °C for the six standard radar bands.
m_w_0C = {
    wl_S: complex(9.075, 1.253),
    wl_C: complex(8.328, 2.217),
    wl_X: complex(7.351, 2.785),
    wl_Ku: complex(6.265, 2.993),
    wl_Ka: complex(4.040, 2.388),
    wl_W: complex(2.880, 1.335),
}

# Water refractive indices at 10 °C.
m_w_10C = {
    wl_S: complex(9.019, 0.887),
    wl_C: complex(8.601, 1.687),
    wl_X: complex(7.942, 2.332),
    wl_Ku: complex(7.042, 2.777),
    wl_Ka: complex(4.638, 2.672),
    wl_W: complex(3.117, 1.665),
}

# Water refractive indices at 20 °C.
m_w_20C = {
    wl_S: complex(8.876, 0.653),
    wl_C: complex(8.633, 1.289),
    wl_X: complex(8.208, 1.886),
    wl_Ku: complex(7.537, 2.424),
    wl_Ka: complex(5.206, 2.801),
    wl_W: complex(3.382, 1.941),
}

# Solid-ice density in g/cm^3.
ice_density = 0.9167


def ice_refractive(file):
    """Build a callable interpolator for ice/snow refractive index.

    Parameters
    ----------
    file : str
        Path to a refractive-index lookup table with columns
        ``(wavelength [μm], real, imag)``. The bundled ``ice_refr.dat``
        file comes from the Warren & Brandt (2008) optical-constants
        compilation hosted at
        http://www.atmos.washington.edu/ice_optical_constants/ .

    Returns
    -------
    ref : callable
        ``ref(wl, snow_density)`` where ``wl`` is in mm and
        ``snow_density`` is in g/cm³. Internally applies Maxwell-Garnett
        to mix the ice refractive index with air at the given density.
        Handles scalar or array-like ``wl``.

    Notes
    -----
    The module-level :data:`mi` instance is a pre-built interpolator using
    the bundled data file — use that directly in most cases:

    >>> from rustmatrix.refractive import mi
    >>> from rustmatrix.tmatrix_aux import wl_W
    >>> mi(wl_W, 0.9)   # ice at 0.9 g/cm³ at W-band
    """
    D = np.loadtxt(file)

    log_wl = np.log10(D[:, 0] / 1000.0)
    re = D[:, 1]
    log_im = np.log10(D[:, 2])

    iobj_re = interpolate.interp1d(log_wl, re)
    iobj_log_im = interpolate.interp1d(log_wl, log_im)

    def ref(wl, snow_density):
        lwl = np.log10(wl)
        try:
            len(lwl)
        except TypeError:
            mi_sqr = complex(iobj_re(lwl), 10 ** iobj_log_im(lwl)) ** 2
        else:
            mi_sqr = np.array(
                [complex(a, b) for a, b in zip(iobj_re(lwl), 10 ** iobj_log_im(lwl))]
            ) ** 2

        c = (mi_sqr - 1) / (mi_sqr + 2) * snow_density / ice_density
        return np.sqrt((1 + 2 * c) / (1 - c))

    return ref


_module_path = path.split(path.abspath(__file__))[0]

mi = ice_refractive(path.join(_module_path, "ice_refr.dat"))
