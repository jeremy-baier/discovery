"""Tests for the continuous-wave (CW) deterministic delays.

Focused on the frequency-evolving circular-binary model
``makedelay_binary(evolve=True)``: correct factory signature, per-pulsar parameter
binding through ``makedelay``, prior-dict resolution, and finite,
correctly shaped output and gradients. Values are not checked against any
external package."""

import inspect
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import pytest

import discovery as ds
from discovery import prior


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# the source/shared parameters of the evolving CW model and per-pulsar params
_SOURCE_ARGS = ['log10_mc', 'log10_h0', 'log10_f0', 'ra', 'sindec',
                'cosinc', 'psi', 'phi_earth']
_PSR_ARGS = ['phi_psr', 'p_dist']
_CW_COMMON = [f'cw_{a}' for a in _SOURCE_ARGS]   # shared (source) params


class _MockPulsar:
    """Minimal stand-in carrying the attributes makedelay binds from a Pulsar."""
    def __init__(self, name, toas, pos, pdist):
        self.name = name
        self.toas = toas
        self.pos = pos
        self.pdist = pdist


def _mock_pulsar(name='B1937+21', ntoa=200, tspan_years=15):
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    toas = np.sort(rng.uniform(0, tspan_years * 365.25 * 86400, ntoa)) + 53000 * 86400.0
    pos = np.array([0.3, -0.5, 0.81])
    pos = pos / np.linalg.norm(pos)
    pdist = np.array([1.2, 0.2])  # kpc (mean, sigma)
    return _MockPulsar(name, toas, pos, pdist)


def _sample_params():
    # conservative values, well away from coalescence over the observation
    return dict(log10_mc=8.5, log10_h0=-14.0, log10_f0=-8.0, ra=1.1, sindec=0.4,
                cosinc=0.3, psi=0.7, phi_earth=1.3, phi_psr=2.1, p_dist=0.5)


# ---------------------------------------------------------------------------
# Signature tests
# ---------------------------------------------------------------------------

class TestCWEvolveSignature:

    def test_pulsarterm_args(self):
        """The pulsar-term delay exposes the full evolving-model signature."""
        f = ds.makedelay_binary(pulsarterm=True, evolve=True)
        args = inspect.getfullargspec(f).args
        assert args == ['toas', 'pos', 'pdist'] + _SOURCE_ARGS + _PSR_ARGS, \
            f"Got args: {args}"

    def test_earthterm_drops_pulsar_args(self):
        """Earth-only model fixes phi_psr and p_dist, so they are not free args."""
        f = ds.makedelay_binary(pulsarterm=False, evolve=True)
        spec = inspect.getfullargspec(f)
        free = spec.args + [a for a in spec.kwonlyargs
                            if a not in (spec.kwonlydefaults or {})]
        assert 'phi_psr' not in free
        assert 'p_dist' not in free


# ---------------------------------------------------------------------------
# Setup / numerical-sanity tests
# ---------------------------------------------------------------------------

class TestCWEvolveSetup:

    def test_residuals_finite_and_shape(self):
        """Direct evaluation returns a finite residual vector matching toas."""
        psr = _mock_pulsar()
        f = ds.makedelay_binary(pulsarterm=True, evolve=True)
        res = f(jnp.array(psr.toas), jnp.array(psr.pos), jnp.array(psr.pdist),
                **_sample_params())
        assert res.shape == psr.toas.shape
        assert np.all(np.isfinite(np.array(res)))

    def test_earthterm_runs(self):
        """Earth-only model evaluates to a finite vector without pulsar params."""
        psr = _mock_pulsar()
        f = ds.makedelay_binary(pulsarterm=False, evolve=True)
        p = {k: v for k, v in _sample_params().items() if k not in _PSR_ARGS}
        res = f(jnp.array(psr.toas), jnp.array(psr.pos), jnp.array(psr.pdist), **p)
        assert res.shape == psr.toas.shape
        assert np.all(np.isfinite(np.array(res)))

    def test_makedelay_param_names(self):
        """makedelay binds toas/pos/pdist and names source vs per-pulsar params."""
        psr = _mock_pulsar()
        f = ds.makedelay_binary(pulsarterm=True, evolve=True)
        delayfunc = ds.makedelay(psr, f, common=_CW_COMMON, name='cw')

        expected = sorted(_CW_COMMON + [f'{psr.name}_cw_{a}' for a in _PSR_ARGS])
        assert delayfunc.params == expected, f"Got params: {delayfunc.params}"

    def test_makedelay_evaluates(self):
        """The bound delay evaluates to a finite vector at prior-midpoint params."""
        psr = _mock_pulsar()
        f = ds.makedelay_binary(pulsarterm=True, evolve=True)
        delayfunc = ds.makedelay(psr, f, common=_CW_COMMON, name='cw')

        params = {par: 0.5 * sum(prior.getprior_uniform(par)) for par in delayfunc.params}
        res = delayfunc(params)
        assert res.shape == psr.toas.shape
        assert np.all(np.isfinite(np.array(res)))

    def test_prior_dict_resolves(self):
        """Every bound parameter resolves to an entry in the standard prior dict."""
        psr = _mock_pulsar()
        f = ds.makedelay_binary(pulsarterm=True, evolve=True)
        delayfunc = ds.makedelay(psr, f, common=_CW_COMMON, name='cw')

        # raises KeyError if any parameter lacks a prior
        logx = prior.makelogtransform_uniform(delayfunc)
        assert sorted(logx.params) == sorted(delayfunc.params)

    def test_new_prior_keys_present(self):
        """The evolving model's new parameters have explicit prior ranges."""
        assert prior.getprior_uniform('cw_log10_mc') == [7.0, 11.0]
        assert prior.getprior_uniform('B1937+21_cw_p_dist') == [-5.0, 5.0]

    def test_gradient_finite(self):
        """Gradient w.r.t. all parameters is finite (NUTS-readiness)."""
        psr = _mock_pulsar()
        f = ds.makedelay_binary(pulsarterm=True, evolve=True)
        order = _SOURCE_ARGS + _PSR_ARGS
        p0 = jnp.array([_sample_params()[k] for k in order])

        def scal(vec):
            d = dict(zip(order, vec))
            return jnp.sum(f(jnp.array(psr.toas), jnp.array(psr.pos),
                             jnp.array(psr.pdist), **d))

        g = jax.grad(scal)(p0)
        assert bool(jnp.all(jnp.isfinite(g)))
