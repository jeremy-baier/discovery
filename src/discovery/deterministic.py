import functools

import numpy as np
import jax
import jax.numpy as jnp

from . import matrix
from . import const


def fpc_fast(pos, gwtheta, gwphi):
    x, y, z = pos

    sin_phi = jnp.sin(gwphi)
    cos_phi = jnp.cos(gwphi)
    sin_theta = jnp.sin(gwtheta)
    cos_theta = jnp.cos(gwtheta)

    m_dot_pos = sin_phi * x - cos_phi * y
    n_dot_pos = -cos_theta * cos_phi * x - cos_theta * sin_phi * y + sin_theta * z
    omhat_dot_pos = -sin_theta * cos_phi * x - sin_theta * sin_phi * y - cos_theta * z

    denom = 1.0 + omhat_dot_pos

    fplus = 0.5 * (m_dot_pos**2 - n_dot_pos**2) / denom
    fcross = (m_dot_pos * n_dot_pos) / denom

    return fplus, fcross


def fpc_cosmu_fast(pos, gwtheta, gwphi):
    """As fpc_fast, but also return cosMu = cos of the angle between the pulsar
    and the GW source (cosMu = -omhat . pos), needed for the pulsar-term delay."""
    x, y, z = pos

    sin_phi = jnp.sin(gwphi)
    cos_phi = jnp.cos(gwphi)
    sin_theta = jnp.sin(gwtheta)
    cos_theta = jnp.cos(gwtheta)

    m_dot_pos = sin_phi * x - cos_phi * y
    n_dot_pos = -cos_theta * cos_phi * x - cos_theta * sin_phi * y + sin_theta * z
    omhat_dot_pos = -sin_theta * cos_phi * x - sin_theta * sin_phi * y - cos_theta * z

    denom = 1.0 + omhat_dot_pos

    fplus = 0.5 * (m_dot_pos**2 - n_dot_pos**2) / denom
    fcross = (m_dot_pos * n_dot_pos) / denom
    cosMu = -omhat_dot_pos

    return fplus, fcross, cosMu


def makedelay_binary(pulsarterm=True, evolve=False):
    """Factory for the circular-binary continuous-wave (CW) delay.

    Two regimes are selected by ``evolve``:

    - ``evolve=False`` (default): the **monochromatic** model. The GW frequency is
      constant over the data span, the amplitude is the strain ``log10_h0``, and the
      pulsar term differs from the Earth term only by a free phase ``phi_psr``.
    - ``evolve=True``: the **frequency-evolving** (chirping) model, mirroring
      ``enterprise_extensions.deterministic.cw_delay`` with ``evolve=True``. The
      binary chirps under the leading radiation-reaction term, so the chirp mass
      ``log10_mc`` becomes an independent parameter and the pulsar distance becomes
      physical: the delay carries per-pulsar ``pdist`` (bound from the Pulsar;
      ``[mean, sigma]`` in kpc) and a sampled offset ``p_dist`` (in units of sigma).
      The amplitude still uses ``log10_h0`` (luminosity distance is derived from h0,
      mc, and fgw, as in enterprise).

    ``evolve`` is resolved at construction time because the two regimes have
    different parameter signatures, which ``makedelay`` introspects.
    """
    if evolve:
        def delay_binary(toas, pos, pdist, log10_mc, log10_h0, log10_f0,
                         ra, sindec, cosinc, psi, phi_earth, phi_psr, p_dist):
            """Evolving BBH residuals from Ellis et. al 2012, 2013 (circular orbit)."""

            mc = 10**log10_mc * const.Tsun      # chirp mass in seconds (geometrized)
            h0 = 10**log10_h0
            fgw = 10**log10_f0                  # GW frequency at tref

            dec, inc = jnp.arcsin(sindec), jnp.arccos(cosinc)

            # antenna pattern + cos(pulsar-source angle); careful with dec -> gwtheta
            fplus, fcross, cosMu = fpc_cosmu_fast(pos, 0.5 * jnp.pi - dec, ra)

            # measured pulsar distance (kpc) -> light-travel time (s); offset in sigma units
            p_dist = (pdist[0] + pdist[1] * p_dist) * const.kpc / const.c

            # luminosity distance (s) from strain, chirp mass, frequency
            dist = 2.0 * mc**(5.0/3.0) * (jnp.pi * fgw)**(2.0/3.0) / h0

            tref = 86400.0 * 51544.5            # MJD J2000 in seconds
            t = toas - tref
            tp = t - p_dist * (1.0 - cosMu)     # retarded (pulsar) time

            w0 = jnp.pi * fgw                   # orbital angular frequency
            phase0 = 0.5 * phi_earth            # GW phase -> orbital phase

            # frequency evolution at earth and pulsar (Peters)
            fac = 256.0 / 5.0 * mc**(5.0/3.0) * w0**(8.0/3.0)
            omega = w0 * (1.0 - fac * t) ** (-3.0/8.0)
            omega_p = w0 * (1.0 - fac * tp) ** (-3.0/8.0)
            omega_p0 = w0 * (1.0 + fac * p_dist * (1.0 - cosMu)) ** (-3.0/8.0)

            # phases
            phase = phase0 + 1.0 / 32.0 / mc**(5.0/3.0) * (w0**(-5.0/3.0) - omega**(-5.0/3.0))

            # earth-term coefficients and amplitude
            At = -0.5 * jnp.sin(2.0 * phase) * (3.0 + jnp.cos(2.0 * inc))
            Bt =  2.0 * jnp.cos(2.0 * phase) * jnp.cos(inc)
            alpha = mc**(5.0/3.0) / (dist * omega**(1.0/3.0))

            rplus  = alpha * (-At * jnp.cos(2.0 * psi) + Bt * jnp.sin(2.0 * psi))
            rcross = alpha * ( At * jnp.sin(2.0 * psi) + Bt * jnp.cos(2.0 * psi))

            if pulsarterm:
                phase_p = (phase0 + phi_psr
                           + 1.0 / 32.0 / mc**(5.0/3.0) * (omega_p0**(-5.0/3.0) - omega_p**(-5.0/3.0)))

                At_p = -0.5 * jnp.sin(2.0 * phase_p) * (3.0 + jnp.cos(2.0 * inc))
                Bt_p =  2.0 * jnp.cos(2.0 * phase_p) * jnp.cos(inc)
                alpha_p = mc**(5.0/3.0) / (dist * omega_p**(1.0/3.0))

                rplus_p  = alpha_p * (-At_p * jnp.cos(2.0 * psi) + Bt_p * jnp.sin(2.0 * psi))
                rcross_p = alpha_p * ( At_p * jnp.sin(2.0 * psi) + Bt_p * jnp.cos(2.0 * psi))

                res = fplus * (rplus_p - rplus) + fcross * (rcross_p - rcross)
            else:
                res = -fplus * rplus - fcross * rcross

            return res

        if not pulsarterm:
            delay_binary = functools.partial(delay_binary, phi_psr=jnp.nan, p_dist=jnp.nan)

        return delay_binary

    def delay_binary(toas, pos, log10_h0, log10_f0, ra, sindec, cosinc, psi, phi_earth, phi_psr):
        """BBH residuals from Ellis et. al 2012, 2013"""

        h0 = 10**log10_h0
        f0 = 10**log10_f0

        dec, inc = jnp.arcsin(sindec), jnp.arccos(cosinc)

        # calculate antenna pattern (note: pos is pulsar sky position unit vector)
        fplus, fcross = fpc_fast(pos, 0.5 * jnp.pi - dec, ra)  # careful with dec -> gwtheta conversion

        if pulsarterm:
            phi_avg = 0.5 * (phi_earth + phi_psr)
        else:
            phi_avg = phi_earth

        tref = 86400.0 * 51544.5  # MJD J2000 in seconds

        phase = phi_avg + 2.0 * jnp.pi * f0 * (toas - tref)
        cphase, sphase = jnp.cos(phase), jnp.sin(phase)

        # fix this for no pulsarterm

        if pulsarterm:
            phi_diff = 0.5 * (phi_earth - phi_psr)
            sin_diff = jnp.sin(phi_diff)

            delta_sin =  2.0 * cphase * sin_diff
            delta_cos = -2.0 * sphase * sin_diff
        else:
            delta_sin = sphase
            delta_cos = cphase

        At = -1.0 * (1.0 + jnp.cos(inc)**2) * delta_sin
        Bt =  2.0 * jnp.cos(inc) * delta_cos

        alpha = h0 / (2 * jnp.pi * f0)

        # calculate rplus and rcross
        rplus  = alpha * (-At * jnp.cos(2 * psi) + Bt * jnp.sin(2 * psi))
        rcross = alpha * ( At * jnp.sin(2 * psi) + Bt * jnp.cos(2 * psi))

        # calculate residuals
        res = -fplus * rplus - fcross * rcross

        return res

    if not pulsarterm:
        delay_binary = functools.partial(delay_binary, phi_psr=jnp.nan)

    return delay_binary


def makedelay_binary_phases(pulsarterm=True):
    """Factory for computing cphase and sphase vectors from binary parameters."""
    def delay_binary_phases(toas, log10_f0):
        """Compute cosine and sine phase vectors.

        Returns:
            cphase: cosine phase vector (toas,)
            sphase: sine phase vector (toas,)
        """
        f0 = 10**log10_f0
        tref = 86400.0 * 51544.5  # MJD J2000 in seconds

        # Compute base phase vectors from frequency evolution only
        phase_base = 2.0 * jnp.pi * f0 * (toas - tref)
        cphase = jnp.cos(phase_base)
        sphase = jnp.sin(phase_base)

        return jnp.array([cphase, sphase])

    return delay_binary_phases


def makedelay_binary_coefficients(pulsarterm=True):
    """Factory for computing coefficients to multiply phase vectors."""
    def delay_binary_coefficients(pos, log10_h0, log10_f0, ra, sindec, cosinc, psi, phi_earth, phi_psr):
        """Compute antenna pattern factors and phase coefficients.

        Returns:
            coeffs: dictionary with keys:
                - 'fplus', 'fcross': antenna pattern factors
                - 'rplus_coeff_c', 'rplus_coeff_s': coefficients for cphase/sphase in rplus
                - 'rcross_coeff_c', 'rcross_coeff_s': coefficients for cphase/sphase in rcross

            Full response is reconstructed as:
            res = -fplus * (rplus_coeff_c * cphase + rplus_coeff_s * sphase)
                  -fcross * (rcross_coeff_c * cphase + rcross_coeff_s * sphase)
        """
        h0 = 10**log10_h0
        f0 = 10**log10_f0

        dec, inc = jnp.arcsin(sindec), jnp.arccos(cosinc)

        # calculate antenna pattern (note: pos is pulsar sky position unit vector)
        fplus, fcross = fpc_fast(pos, 0.5 * jnp.pi - dec, ra)  # careful with dec -> gwtheta conversion

        # Calculate coefficients that multiply cphase and sphase
        # Apply addition theorem: cos(phi_avg + phase_base) = cos(phi_avg)*cos(phase_base) - sin(phi_avg)*sin(phase_base)
        # Original cphase_orig = cos(phi_avg + phase_base)
        # Original sphase_orig = sin(phi_avg + phase_base)
        if pulsarterm:
            phi_avg = 0.5 * (phi_earth + phi_psr)
            phi_diff = 0.5 * (phi_earth - phi_psr)

            cos_avg = jnp.cos(phi_avg)
            sin_avg = jnp.sin(phi_avg)
            sin_diff = jnp.sin(phi_diff)

            # cphase_orig = cos_avg * cphase - sin_avg * sphase
            # sphase_orig = sin_avg * cphase + cos_avg * sphase
            # delta_sin =  2.0 * cphase_orig * sin_diff
            # delta_cos = -2.0 * sphase_orig * sin_diff

            c_coeff_sin = 2.0 * cos_avg * sin_diff    # coefficient for cphase in delta_sin
            s_coeff_sin = -2.0 * sin_avg * sin_diff   # coefficient for sphase in delta_sin
            c_coeff_cos = -2.0 * sin_avg * sin_diff   # coefficient for cphase in delta_cos
            s_coeff_cos = -2.0 * cos_avg * sin_diff   # coefficient for sphase in delta_cos
        else:
            # cphase_orig = cos(phi_earth + phase_base) = cos(phi_earth)*cphase - sin(phi_earth)*sphase
            # sphase_orig = sin(phi_earth + phase_base) = sin(phi_earth)*cphase + cos(phi_earth)*sphase
            # delta_sin = sphase_orig
            # delta_cos = cphase_orig
            cos_earth = jnp.cos(phi_earth)
            sin_earth = jnp.sin(phi_earth)

            c_coeff_sin = sin_earth    # coefficient for cphase in delta_sin
            s_coeff_sin = cos_earth    # coefficient for sphase in delta_sin
            c_coeff_cos = cos_earth    # coefficient for cphase in delta_cos
            s_coeff_cos = -sin_earth   # coefficient for sphase in delta_cos

        # At = -1.0 * (1.0 + cos(inc)^2) * delta_sin
        # Bt = 2.0 * cos(inc) * delta_cos
        cos_inc = jnp.cos(inc)
        At_coeff_c = -1.0 * (1.0 + cos_inc**2) * c_coeff_sin
        At_coeff_s = -1.0 * (1.0 + cos_inc**2) * s_coeff_sin
        Bt_coeff_c = 2.0 * cos_inc * c_coeff_cos
        Bt_coeff_s = 2.0 * cos_inc * s_coeff_cos

        alpha = h0 / (2 * jnp.pi * f0)
        cos_2psi = jnp.cos(2 * psi)
        sin_2psi = jnp.sin(2 * psi)

        # rplus = alpha * (-At * cos(2*psi) + Bt * sin(2*psi))
        # rcross = alpha * (At * sin(2*psi) + Bt * cos(2*psi))

        # Coefficient for cphase in rplus: alpha * (-At_coeff_c * cos(2*psi) + Bt_coeff_c * sin(2*psi))
        rplus_coeff_c = alpha * (-At_coeff_c * cos_2psi + Bt_coeff_c * sin_2psi)
        # Coefficient for sphase in rplus: alpha * (-At_coeff_s * cos(2*psi) + Bt_coeff_s * sin(2*psi))
        rplus_coeff_s = alpha * (-At_coeff_s * cos_2psi + Bt_coeff_s * sin_2psi)

        # Coefficient for cphase in rcross: alpha * (At_coeff_c * sin(2*psi) + Bt_coeff_c * cos(2*psi))
        rcross_coeff_c = alpha * (At_coeff_c * sin_2psi + Bt_coeff_c * cos_2psi)
        # Coefficient for sphase in rcross: alpha * (At_coeff_s * sin(2*psi) + Bt_coeff_s * cos(2*psi))
        rcross_coeff_s = alpha * (At_coeff_s * sin_2psi + Bt_coeff_s * cos_2psi)

        Ac = -fplus * rplus_coeff_c - fcross * rcross_coeff_c
        As = -fplus * rplus_coeff_s - fcross * rcross_coeff_s

        return jnp.array([Ac, As])

    if not pulsarterm:
        delay_binary_coefficients = functools.partial(delay_binary_coefficients, phi_psr=jnp.nan)

    return delay_binary_coefficients



def cos2comp(f, df, A, f0, phi, t0):
    """Project signal A * cos(2pi f t + phi) onto Fourier basis
    cos(2pi k t/T), sin(2pi k t/T) for t in [t0, t0+T]."""

    T = 1.0 / df[0]

    Delta_omega = 2.0 * jnp.pi * (f0 - f[::2])
    Sigma_omega = 2.0 * jnp.pi * (f0 + f[::2])

    phase_Delta_start = phi + Delta_omega * t0
    phase_Delta_end   = phi + Delta_omega * (t0 + T)

    phase_Sigma_start = phi + Sigma_omega * t0
    phase_Sigma_end   = phi + Sigma_omega * (t0 + T)

    ck = (A / T) * (
        (jnp.sin(phase_Delta_end) - jnp.sin(phase_Delta_start)) / Delta_omega +
        (jnp.sin(phase_Sigma_end) - jnp.sin(phase_Sigma_start)) / Sigma_omega
    )

    sk = (A / T) * (
        (jnp.cos(phase_Delta_end) - jnp.cos(phase_Delta_start)) / Delta_omega -
        (jnp.cos(phase_Sigma_end) - jnp.cos(phase_Sigma_start)) / Sigma_omega
    )

    return jnp.stack((sk, ck), axis=1).reshape(-1)


def makefourier_binary(pulsarterm=True):
    def fourier_binary(f, df, mintoa, pos, log10_h0, log10_f0, ra, sindec, cosinc, psi, phi_earth, phi_psr):
        """BBH residuals from Ellis et. al 2012, 2013"""

        h0 = 10**log10_h0
        f0 = 10**log10_f0

        dec, inc = jnp.arcsin(sindec), jnp.arccos(cosinc)

        # calculate antenna pattern (note: pos is pulsar sky position unit vector)
        fplus, fcross = fpc_fast(pos, 0.5 * jnp.pi - dec, ra)  # careful with dec -> gwtheta conversion

        if pulsarterm:
            phi_avg  = 0.5 * (phi_earth + phi_psr)
        else:
            phi_avg = phi_earth

        tref = 86400.0 * 51544.5  # MJD J2000 in seconds

        cphase = cos2comp(f, df, 1.0, f0, phi_avg - 2.0 * jnp.pi * f0 * tref, mintoa)
        sphase = cos2comp(f, df, 1.0, f0, phi_avg - 2.0 * jnp.pi * f0 * tref - 0.5*jnp.pi, mintoa)

        # fix this for no pulsarterm

        if pulsarterm:
            phi_diff = 0.5 * (phi_earth - phi_psr)
            sin_diff = jnp.sin(phi_diff)

            delta_sin =  2.0 * cphase * sin_diff
            delta_cos = -2.0 * sphase * sin_diff
        else:
            delta_sin = sphase
            delta_cos = cphase

        At = -1.0 * (1.0 + jnp.cos(inc)**2) * delta_sin
        Bt =  2.0 * jnp.cos(inc) * delta_cos

        alpha = h0 / (2 * jnp.pi * f0)

        # calculate rplus and rcross
        rplus  = alpha * (-At * jnp.cos(2 * psi) + Bt * jnp.sin(2 * psi))
        rcross = alpha * ( At * jnp.sin(2 * psi) + Bt * jnp.cos(2 * psi))

        # calculate residuals
        res = -fplus * rplus - fcross * rcross

        return res

    if not pulsarterm:
        fourier_binary = functools.partial(fourier_binary, phi_psr=jnp.nan)

    return fourier_binary