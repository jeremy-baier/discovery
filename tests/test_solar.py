"""Tests for discovery.solar: dm_solar, make_solardm, fourierbasis_solar_dm,
and makegp_timedomain_solar_dm (with and without a custom interpolation basis)."""

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import pytest

from discovery import const, matrix, solar
from discovery.solar import (
    theta_impact,
    dm_solar,
    make_solardm,
    fourierbasis_solar_dm,
    makegp_timedomain_solar_dm,
)
from discovery.signals import (
    square_exponential_kernel,
    matern_kernel,
    custom_blocked_interpolation_basis,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
AU_LIGHT_SEC = const.AU / const.c   # ~499 s
AU_PC        = const.AU / const.pc  # ~4.85e-6 pc


# ---------------------------------------------------------------------------
# Mock pulsar with solar-system geometry
# ---------------------------------------------------------------------------

class _SolarMockPulsar:
    """Mock pulsar with the minimal attributes required by solar.py.

    Geometry (fixed for all TOAs):
        - Sun at SSB origin
        - Earth at (AU_LIGHT_SEC, 0, 0) in light-seconds
        - Pulsar direction along +z  =>  theta_impact = pi/2 for all TOAs
    """

    def __init__(self, n_toas=60, tspan_years=20.0, seed=0):
        rng = np.random.default_rng(seed)
        tspan_s = tspan_years * 365.25 * 86400

        self.toas          = np.sort(rng.uniform(0, tspan_s, n_toas))
        self.freqs         = rng.uniform(1000., 2000., n_toas)        # MHz
        self.residuals     = rng.normal(0, 1e-6, n_toas)
        self.backend_flags = np.array(['backend_A'] * n_toas)
        self.name          = 'J0000+0000'
        self.pos           = np.array([0., 0., 1.])

        # Sun at SSB => sunssb zero (only first 3 components used)
        self.sunssb    = np.zeros((n_toas, 3))

        # Earth at 1 AU along x axis (in light-seconds); shape (n_toas, 10, 3)
        # Only index [:, 2, :3] is used
        self.planetssb = np.zeros((n_toas, 10, 3))
        self.planetssb[:, 2, 0] = AU_LIGHT_SEC

        # Unit vector from SSB to pulsar at each TOA; +z direction
        self.pos_t = np.tile([0., 0., 1.], (n_toas, 1))

    @property
    def mintoa(self):
        return self.toas.min()

    @property
    def maxtoa(self):
        return self.toas.max()


@pytest.fixture(scope='module')
def solar_psr():
    return _SolarMockPulsar()


# ---------------------------------------------------------------------------
# theta_impact tests
# ---------------------------------------------------------------------------

class TestThetaImpact:

    def test_returns_four_arrays(self, solar_psr):
        result = theta_impact(solar_psr)
        assert len(result) == 4

    def test_theta_shape(self, solar_psr):
        theta, r_earth, b, z = theta_impact(solar_psr)
        n = len(solar_psr.toas)
        assert theta.shape == (n,)
        assert r_earth.shape == (n,)
        assert b.shape == (n,)
        assert z.shape == (n,)

    def test_theta_is_pi_over_2(self, solar_psr):
        """With Earth along x and pulsar along z, theta should be pi/2."""
        theta, _, _, _ = theta_impact(solar_psr)
        np.testing.assert_allclose(theta, np.pi / 2, atol=1e-10)

    def test_r_earth_is_one_au(self, solar_psr):
        """Earth–Sun distance should equal 1 AU in light-seconds."""
        _, r_earth, _, _ = theta_impact(solar_psr)
        np.testing.assert_allclose(r_earth, AU_LIGHT_SEC, rtol=1e-10)

    def test_impact_parameter(self, solar_psr):
        """b = R_earth * sin(theta) = R_earth for theta = pi/2."""
        theta, r_earth, b, _ = theta_impact(solar_psr)
        np.testing.assert_allclose(b, r_earth * np.sin(theta), rtol=1e-10)

    def test_theta_in_range(self, solar_psr):
        theta, _, _, _ = theta_impact(solar_psr)
        assert np.all(theta >= 0) and np.all(theta <= np.pi)


# ---------------------------------------------------------------------------
# dm_solar tests (pure-function)
# ---------------------------------------------------------------------------

class TestDmSolar:

    def test_scalar_input(self):
        """dm_solar should work with scalar inputs."""
        val = dm_solar(7.9, np.pi / 2, AU_LIGHT_SEC)
        assert float(val) > 0

    def test_array_input(self, solar_psr):
        theta, r_earth, _, _ = theta_impact(solar_psr)
        dm = dm_solar(7.9, theta, r_earth)
        assert dm.shape == theta.shape

    def test_linearly_scales_with_n_earth(self):
        """DM should be proportional to n_earth."""
        theta = np.pi / 2
        r = AU_LIGHT_SEC
        dm1 = float(dm_solar(1.0, theta, r))
        dm2 = float(dm_solar(2.0, theta, r))
        np.testing.assert_allclose(dm2, 2.0 * dm1, rtol=1e-10)

    def test_increases_toward_conjunction(self):
        """DM increases as the line of sight approaches the Sun (theta -> 0)."""
        r = AU_LIGHT_SEC
        dm_edge = float(dm_solar(7.9, np.pi / 2, r))
        dm_near = float(dm_solar(7.9, 0.1, r))
        assert dm_near > dm_edge

    def test_close_conjunction_branch(self):
        """At theta very close to pi, the close-approach branch is taken (no error)."""
        r = AU_LIGHT_SEC
        # pi - theta < 1e-5 triggers the close branch
        theta_close = np.pi - 1e-7
        val = float(dm_solar(7.9, theta_close, r))
        assert val > 0

    def test_positive_everywhere(self):
        thetas = np.linspace(0.05, np.pi - 0.05, 50)
        dm = dm_solar(7.9, thetas, AU_LIGHT_SEC)
        assert np.all(np.asarray(dm) > 0)


# ---------------------------------------------------------------------------
# make_solardm tests
# ---------------------------------------------------------------------------

class TestMakeSolardm:

    def test_returns_callable(self, solar_psr):
        solardm = make_solardm(solar_psr)
        assert callable(solardm)

    def test_output_shape(self, solar_psr):
        solardm = make_solardm(solar_psr)
        dm = solardm(7.9)
        assert dm.shape == (len(solar_psr.toas),)

    def test_scales_with_n_earth(self, solar_psr):
        solardm = make_solardm(solar_psr)
        dm1 = np.asarray(solardm(1.0))
        dm2 = np.asarray(solardm(2.0))
        np.testing.assert_allclose(dm2, 2.0 * dm1, rtol=1e-10)

    def test_positive_output(self, solar_psr):
        solardm = make_solardm(solar_psr)
        dm = np.asarray(solardm(7.9))
        assert np.all(dm > 0)

    def test_frequency_dependence(self, solar_psr):
        """make_solardm includes freq^-2 scaling; lower freq => larger DM delay."""
        solardm = make_solardm(solar_psr)
        dm = np.asarray(solardm(7.9))
        # TOA with lowest frequency should have largest DM delay
        i_low  = np.argmin(solar_psr.freqs)
        i_high = np.argmax(solar_psr.freqs)
        assert dm[i_low] > dm[i_high]


# ---------------------------------------------------------------------------
# fourierbasis_solar_dm tests
# ---------------------------------------------------------------------------

class TestFourierbasisSolarDm:

    def test_return_shapes(self, solar_psr):
        n = 14
        f, df, fmat = fourierbasis_solar_dm(solar_psr, n)
        assert f.shape   == (2 * n,)
        assert df.shape  == (2 * n,)
        assert fmat.shape == (len(solar_psr.toas), 2 * n)

    def test_frequencies_positive(self, solar_psr):
        f, _, _ = fourierbasis_solar_dm(solar_psr, 10)
        assert np.all(f > 0)

    def test_explicit_T_matches_auto(self, solar_psr):
        T = solar_psr.maxtoa - solar_psr.mintoa
        f1, df1, fmat1 = fourierbasis_solar_dm(solar_psr, 10)
        f2, df2, fmat2 = fourierbasis_solar_dm(solar_psr, 10, T=T)
        np.testing.assert_allclose(f1, f2, rtol=1e-12)
        np.testing.assert_allclose(fmat1, fmat2, rtol=1e-12)

    def test_solar_dm_scaling_applied(self, solar_psr):
        """solar_dm basis = plain fourierbasis * solar DM weight per TOA."""
        from discovery.signals import fourierbasis
        n = 8
        T = solar_psr.maxtoa - solar_psr.mintoa
        _, _, fmat_plain     = fourierbasis(solar_psr, n, T=T)
        _, _, fmat_solar     = fourierbasis_solar_dm(solar_psr, n, T=T)

        theta, r_earth, _, _ = theta_impact(solar_psr)
        dm_sol = np.asarray(dm_solar(1.0, theta, r_earth))
        dt_DM  = dm_sol * 4.148808e3 / solar_psr.freqs**2

        np.testing.assert_allclose(
            fmat_solar, fmat_plain * dt_DM[:, None], rtol=1e-10
        )

    def test_modes_shape(self, solar_psr):
        """Explicit modes control the number of Fourier components returned."""
        T = solar_psr.maxtoa - solar_psr.mintoa
        modes = np.array([1.0, 2.0, 5.0, 10.0]) / T
        f, df, fmat = fourierbasis_solar_dm(solar_psr, len(modes), modes=modes, T=T)
        assert f.shape   == (2 * len(modes),)
        assert df.shape  == (2 * len(modes),)
        assert fmat.shape == (len(solar_psr.toas), 2 * len(modes))

    def test_modes_frequencies_match(self, solar_psr):
        """Returned frequencies equal the supplied modes (each repeated for sin/cos)."""
        T = solar_psr.maxtoa - solar_psr.mintoa
        modes = np.array([1.5, 3.0, 7.0]) / T
        f, _, _ = fourierbasis_solar_dm(solar_psr, len(modes), modes=modes, T=T)
        np.testing.assert_allclose(f[::2], modes, rtol=1e-12)

    def test_modes_matches_default_grid(self, solar_psr):
        """Modes equal to the default harmonic grid reproduce the no-modes result."""
        n = 6
        T = solar_psr.maxtoa - solar_psr.mintoa
        default_modes = np.arange(1, n + 1, dtype=float) / T
        f1, df1, fmat1 = fourierbasis_solar_dm(solar_psr, n, T=T)
        f2, df2, fmat2 = fourierbasis_solar_dm(solar_psr, n, modes=default_modes, T=T)
        np.testing.assert_allclose(fmat1, fmat2, rtol=1e-12)
        np.testing.assert_allclose(f1, f2,      rtol=1e-12)

    def test_modes_solar_dm_scaling_preserved(self, solar_psr):
        """Solar DM scaling is applied correctly even when modes are supplied."""
        from discovery.signals import fourierbasis as _fourierbasis
        T = solar_psr.maxtoa - solar_psr.mintoa
        modes = np.array([1.0, 3.0, 9.0]) / T

        _, _, fmat_plain = _fourierbasis(solar_psr, len(modes), modes=modes, T=T)
        _, _, fmat_solar = fourierbasis_solar_dm(solar_psr, len(modes), modes=modes, T=T)

        theta, r_earth, _, _ = theta_impact(solar_psr)
        dm_sol = np.asarray(dm_solar(1.0, theta, r_earth))
        dt_DM  = dm_sol * 4.148808e3 / solar_psr.freqs**2

        np.testing.assert_allclose(fmat_solar, fmat_plain * dt_DM[:, None], rtol=1e-10)

    def test_modes_irregular_grid(self, solar_psr):
        """Irregular (non-harmonic) mode grid is accepted and produces correct shape."""
        T = solar_psr.maxtoa - solar_psr.mintoa
        modes = np.array([0.7, 2.3, 4.1, 6.6, 11.0]) / T
        f, df, fmat = fourierbasis_solar_dm(solar_psr, len(modes), modes=modes, T=T)
        assert fmat.shape == (len(solar_psr.toas), 2 * len(modes))
        np.testing.assert_allclose(f[::2], modes, rtol=1e-12)


# ---------------------------------------------------------------------------
# makegp_timedomain_solar_dm tests
# ---------------------------------------------------------------------------

class TestMakegpTimeDomainSolarDm:

    # ---- fixtures -----------------------------------------------------------

    @pytest.fixture
    def sq_exp_cov(self):
        return square_exponential_kernel(log10_sigma_sq_exp=-7., log10_ell=8.)

    @pytest.fixture
    def matern_cov(self):
        return matern_kernel(log10_sigma_matern=-7., log10_ell=8., nu=1.5)

    def _make_custom_basis(self, solar_psr, n_nodes=12):
        """Build a custom node-based Umat and node array for solar_psr."""
        tspan_years = (solar_psr.maxtoa - solar_psr.mintoa) / (365.25 * 86400)
        nodes_mjd = np.linspace(0.0, tspan_years * 365.25, n_nodes)
        Umat_raw, nodes_s = custom_blocked_interpolation_basis(
            solar_psr.toas, nodes_mjd, kind='linear'
        )
        return Umat_raw, nodes_s

    # ---- auto-quantized basis -----------------------------------------------

    def test_returns_variable_gp(self, solar_psr, sq_exp_cov):
        gp = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30)
        assert isinstance(gp, matrix.VariableGP)

    def test_basis_shape_auto(self, solar_psr, sq_exp_cov):
        dt = 86400 * 30
        gp = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=dt)
        assert gp.F.shape[0] == len(solar_psr.toas)
        assert gp.F.shape[1] >= 1

    def test_param_names_auto(self, solar_psr, sq_exp_cov):
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, dt=86400 * 30, name='sw_gp'
        )
        for p in gp.Phi.params:
            assert p.startswith(solar_psr.name), f"Param {p!r} missing pulsar prefix"
            assert 'sw_gp' in p

    def test_phi_square_matrix_auto(self, solar_psr, sq_exp_cov):
        dt = 86400 * 30
        gp = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=dt)
        n_bins = gp.F.shape[1]
        params = {p: jnp.array(-7.0) if 'sigma' in p else jnp.array(8.0)
                  for p in gp.Phi.params}
        phi = gp.Phi.getN(params)
        assert phi.shape == (n_bins, n_bins)

    def test_phi_symmetric_auto(self, solar_psr, sq_exp_cov):
        gp = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30)
        params = {p: jnp.array(-7.0) if 'sigma' in p else jnp.array(8.0)
                  for p in gp.Phi.params}
        phi = gp.Phi.getN(params)
        np.testing.assert_allclose(np.asarray(phi), np.asarray(phi).T, atol=1e-12)

    def test_phi_positive_definite_auto(self, solar_psr, sq_exp_cov):
        gp = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30)
        params = {p: jnp.array(-7.0) if 'sigma' in p else jnp.array(8.0)
                  for p in gp.Phi.params}
        phi = gp.Phi.getN(params)
        eigvals = np.linalg.eigvalsh(np.asarray(phi))
        assert np.all(eigvals > 0)

    def test_index_attribute_auto(self, solar_psr, sq_exp_cov):
        gp = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30)
        assert len(gp.index) == 1
        key = next(iter(gp.index))
        assert key.startswith(solar_psr.name)

    def test_solar_dm_scaling_in_basis_auto(self, solar_psr, sq_exp_cov):
        """Columns of F reflect (freq-dependent) solar wind DM scaling."""
        dt = 86400 * 30
        gp = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=dt)
        theta, r_earth, _, _  = theta_impact(solar_psr)
        dm_sol = np.asarray(dm_solar(1.0, theta, r_earth))
        expected_scale = dm_sol * 4.148808e3 / solar_psr.freqs**2

        F = np.asarray(gp.F)
        # For each active bin find two TOAs and check the weight ratio matches
        # the ratio of their DM scalings.
        for col in range(F.shape[1]):
            rows = np.where(np.abs(F[:, col]) > 0)[0]
            if len(rows) < 2:
                continue
            i, j = rows[0], rows[1]
            if expected_scale[j] == 0:
                continue
            ratio_F  = F[i, col] / F[j, col]
            ratio_dm = expected_scale[i] / expected_scale[j]
            np.testing.assert_allclose(ratio_F, ratio_dm, rtol=1e-10)
            break

    def test_different_kernels_different_phi_auto(self, solar_psr, sq_exp_cov, matern_cov):
        dt = 86400 * 30
        gp_sq = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=dt)
        gp_mt = makegp_timedomain_solar_dm(solar_psr, matern_cov, dt=dt)
        assert gp_sq.F.shape == gp_mt.F.shape

        def _eval(gp):
            params = {p: jnp.array(-7.0) if 'sigma' in p else jnp.array(3.0)
                      for p in gp.Phi.params}
            return gp.Phi.getN(params)

        phi_sq = _eval(gp_sq)
        phi_mt = _eval(gp_mt)
        mask = ~np.eye(phi_sq.shape[0], dtype=bool)
        diff = np.abs(np.asarray(phi_sq)[mask] - np.asarray(phi_mt)[mask])
        scale = np.abs(np.asarray(phi_sq)[mask]) + 1e-300
        assert np.max(diff / scale) > 0.01, \
            "sq-exp and Matérn kernels produced nearly identical covariance matrices"

    # ---- custom basis (Umat + nodes) ----------------------------------------

    def test_returns_variable_gp_custom(self, solar_psr, sq_exp_cov):
        Umat, nodes = self._make_custom_basis(solar_psr)
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, Umat=Umat, nodes=nodes
        )
        assert isinstance(gp, matrix.VariableGP)

    def test_basis_shape_custom(self, solar_psr, sq_exp_cov):
        Umat, nodes = self._make_custom_basis(solar_psr, n_nodes=10)
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, Umat=Umat, nodes=nodes
        )
        assert gp.F.shape[0] == len(solar_psr.toas)
        assert gp.F.shape[1] == Umat.shape[1]

    def test_phi_square_matrix_custom(self, solar_psr, sq_exp_cov):
        Umat, nodes = self._make_custom_basis(solar_psr, n_nodes=8)
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, Umat=Umat, nodes=nodes
        )
        n_nodes = gp.F.shape[1]
        params = {p: jnp.array(-7.0) if 'sigma' in p else jnp.array(8.0)
                  for p in gp.Phi.params}
        phi = gp.Phi.getN(params)
        assert phi.shape == (n_nodes, n_nodes)

    def test_phi_symmetric_custom(self, solar_psr, sq_exp_cov):
        Umat, nodes = self._make_custom_basis(solar_psr)
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, Umat=Umat, nodes=nodes
        )
        params = {p: jnp.array(-7.0) if 'sigma' in p else jnp.array(8.0)
                  for p in gp.Phi.params}
        phi = gp.Phi.getN(params)
        np.testing.assert_allclose(np.asarray(phi), np.asarray(phi).T, atol=1e-12)

    def test_phi_positive_definite_custom(self, solar_psr, sq_exp_cov):
        Umat, nodes = self._make_custom_basis(solar_psr)
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, Umat=Umat, nodes=nodes
        )
        params = {p: jnp.array(-7.0) if 'sigma' in p else jnp.array(8.0)
                  for p in gp.Phi.params}
        phi = gp.Phi.getN(params)
        eigvals = np.linalg.eigvalsh(np.asarray(phi))
        assert np.all(eigvals > 0)

    def test_custom_umat_has_solar_dm_columns(self, solar_psr, sq_exp_cov):
        """Each column of gp.F should be scaled by the solar DM weight."""
        Umat, nodes = self._make_custom_basis(solar_psr)
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, Umat=Umat, nodes=nodes
        )
        theta, r_earth, _, _  = theta_impact(solar_psr)
        dm_sol = np.asarray(dm_solar(1.0, theta, r_earth))
        dt_DM  = dm_sol * 4.148808e3 / solar_psr.freqs**2

        F = np.asarray(gp.F)
        # The stored F equals Umat * dt_DM[:, None]; verify this ratio per col
        Umat_expected = Umat * dt_DM[:, None]
        np.testing.assert_allclose(F, Umat_expected, rtol=1e-10)

    def test_nodes_required_with_custom_umat(self, solar_psr, sq_exp_cov):
        """Providing Umat without nodes should raise AssertionError."""
        Umat, _ = self._make_custom_basis(solar_psr)
        with pytest.raises(AssertionError, match="nodes"):
            makegp_timedomain_solar_dm(
                solar_psr, sq_exp_cov, Umat=Umat, nodes=None
            )

    def test_common_param_custom(self, solar_psr, sq_exp_cov):
        """Parameter in common list should not carry the pulsar-name prefix."""
        import inspect as _inspect
        args = _inspect.getfullargspec(sq_exp_cov).args
        sigma_arg = next(a for a in args if 'sigma' in a)
        Umat, nodes = self._make_custom_basis(solar_psr)
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, Umat=Umat, nodes=nodes,
            name='sw_gp', common=[sigma_arg]
        )
        assert sigma_arg in gp.Phi.params, \
            f"Expected bare common param {sigma_arg!r} in {gp.Phi.params}"

    def test_auto_and_custom_basis_same_node_count(self, solar_psr, sq_exp_cov):
        """Custom basis with n_nodes nodes gives gp.F with n_nodes columns."""
        for n_nodes in [6, 10, 15]:
            Umat, nodes = self._make_custom_basis(solar_psr, n_nodes=n_nodes)
            gp = makegp_timedomain_solar_dm(
                solar_psr, sq_exp_cov, Umat=Umat, nodes=nodes
            )
            assert gp.F.shape[1] == Umat.shape[1]

    # ---- fixed-hyperparameter ("fixed-point") path --------------------------

    @staticmethod
    def _full_noisedict(params):
        """Full noisedict for the square-exponential covariance params."""
        return {p: (-7.0 if 'sigma' in p else 8.0) for p in params}

    def test_empty_noisedict_returns_variable_gp(self, solar_psr, sq_exp_cov):
        gp = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30,
                                        noisedict={})
        assert isinstance(gp, matrix.VariableGP)

    def test_full_noisedict_returns_constant_gp(self, solar_psr, sq_exp_cov):
        gv = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30)
        gc = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, dt=86400 * 30,
            noisedict=self._full_noisedict(gv.Phi.params)
        )
        assert isinstance(gc, matrix.ConstantGP)
        assert isinstance(gc.Phi, matrix.NoiseMatrix2D_novar)

    def test_partial_noisedict_returns_variable_gp(self, solar_psr, sq_exp_cov):
        gv = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30)
        full = self._full_noisedict(gv.Phi.params)
        one_key = next(iter(full))
        gp = makegp_timedomain_solar_dm(
            solar_psr, sq_exp_cov, dt=86400 * 30,
            noisedict={one_key: full[one_key]}
        )
        assert isinstance(gp, matrix.VariableGP)

    def test_cached_covariance_matches_variable(self, solar_psr, sq_exp_cov):
        gv = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30)
        nd = self._full_noisedict(gv.Phi.params)
        gc = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30,
                                        noisedict=nd)
        np.testing.assert_allclose(np.asarray(gc.Phi.N),
                                   np.asarray(gv.Phi.getN(nd)))

    def test_cached_basis_matches_variable(self, solar_psr, sq_exp_cov):
        gv = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30)
        nd = self._full_noisedict(gv.Phi.params)
        gc = makegp_timedomain_solar_dm(solar_psr, sq_exp_cov, dt=86400 * 30,
                                        noisedict=nd)
        np.testing.assert_allclose(np.asarray(gc.F), np.asarray(gv.F))
