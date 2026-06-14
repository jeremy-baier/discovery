Continuous Waves
================

A continuous gravitational wave (CW) is the nearly monochromatic signal from an individual,
slowly evolving supermassive black-hole binary (SMBHB). Discovery models it as a deterministic
delay added to the timing residuals, following the circular-binary formalism of
Ellis et al. (2012, 2013).

Two physical regimes and two likelihood representations are provided:

- :func:`~discovery.deterministic.makedelay_binary` (``evolve=False``, default) — **monochromatic** (non-evolving) time-domain delay.
- :func:`~discovery.deterministic.makedelay_binary` (``evolve=True``) — **frequency-evolving** (chirping) time-domain delay.
- :func:`~discovery.deterministic.makefourier_binary` — **Fourier-domain** representation of the monochromatic model.

All three are JAX functions wrapped for a likelihood with :func:`~discovery.signals.makedelay`:

.. code-block:: python

   cw_func = ds.makedelay_binary(pulsarterm=True)
   delay   = ds.makedelay(psr, cw_func, common=cw_common, name='cw')

Per-pulsar attributes (``toas``, ``pos``, and — for the evolving model — ``pdist``) are bound
automatically from the :class:`~discovery.pulsar.Pulsar`; the remaining arguments become sampled
parameters. Source parameters that are shared across the array (sky location, frequency,
amplitude, …) should be passed in ``common`` so they are not duplicated per pulsar.

The Earth and pulsar terms
--------------------------

A GW induces a timing delay through both the **Earth term** (the wave passing Earth now) and the
**pulsar term** (the wave that passed the pulsar a light-travel time ago). The pulsar term is
delayed by the geometric retardation

.. math::

   t_p = t - \frac{L_p}{c}\,(1 - \cos\mu),

where :math:`L_p` is the pulsar distance and :math:`\mu` is the angle between the pulsar and the
GW source. Set ``pulsarterm=False`` to keep only the Earth term; this also removes the per-pulsar
phase (and distance) parameters.

``evolve=False`` vs ``evolve=True``
-----------------------------------

The key physical choice is whether the binary's orbital frequency is treated as **constant** over
the observation or allowed to **chirp**. This single choice determines which parameters the model
needs.

Monochromatic model (``evolve=False``, default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   cw_func = ds.makedelay_binary(pulsarterm=True)  # evolve=False is the default

The GW frequency :math:`f_0` is assumed fixed over the data span, so the residual is a pure
sinusoid:

.. math::

   \mathrm{phase} = \phi + 2\pi f_0 (t - t_\mathrm{ref}), \qquad
   \alpha = \frac{h_0}{2\pi f_0}.

In this limit the chirp mass and luminosity distance enter the waveform **only** through the
strain amplitude :math:`h_0`, so they are degenerate and are not separate parameters. Likewise,
the pulsar distance only sets the pulsar-term *phase*, which is sampled directly as a free phase
``phi_psr`` rather than derived from a distance.

**Parameters:**

- ``log10_h0`` — log strain amplitude
- ``log10_f0`` — log GW frequency (Hz)
- ``ra`` — right ascension (rad)
- ``sindec`` — sine of declination
- ``cosinc`` — cosine of inclination
- ``psi`` — polarization angle (rad)
- ``phi_earth`` — initial Earth-term phase (rad)
- ``phi_psr`` — initial pulsar-term phase (rad; only if ``pulsarterm=True``)

Evolving model (``evolve=True``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   cw_func = ds.makedelay_binary(pulsarterm=True, evolve=True)

The binary chirps over the observation. The frequency evolves under the leading
(Peters) radiation-reaction term, evaluated separately at Earth and at the pulsar's
retarded time:

.. math::

   \omega(t) = \omega_0 \left(1 - \tfrac{256}{5}\,\mathcal{M}^{5/3}\,
               \omega_0^{8/3}\, t\right)^{-3/8},

with the corresponding chirping phase. Because the frequency now changes, the **chirp mass**
:math:`\mathcal{M}` becomes an independent, measurable parameter (it sets the chirp rate), and the
**pulsar distance** becomes physical: it fixes the pulsar-term frequency offset through
:math:`\mathcal{M}^{5/3}\,\omega_0^{8/3}\,(L_p/c)(1-\cos\mu)`, rather than being absorbed into a
free phase. The luminosity distance is derived internally from :math:`h_0`, :math:`\mathcal{M}`,
and :math:`f_\mathrm{gw}`, so the amplitude is still parameterized by ``log10_h0``.

**Additional parameters** (beyond the monochromatic set):

- ``log10_mc`` — log chirp mass (solar masses); shared source parameter
- ``p_dist`` — per-pulsar distance offset, in units of the measured uncertainty

The measured pulsar distance is taken from ``psr.pdist`` (a ``[mean, sigma]`` pair in kpc) and
combined with the sampled offset as :math:`L_p = (\mathrm{mean} + \sigma \cdot p\_dist)`. The
prior on ``p_dist`` is uniform in sigma units (default :math:`[-5, 5]`); to use it the pulsar
**must** carry a ``pdist`` attribute.

.. note::

   The chirp evolution diverges as the binary approaches coalescence: the factor
   :math:`1 - \tfrac{256}{5}\mathcal{M}^{5/3}\omega_0^{8/3} t` must stay positive over the data
   span. For very large ``log10_mc`` and/or ``log10_f0`` it can go negative and produce ``NaN``;
   keep the priors in a physically sensible range.

When to use which
~~~~~~~~~~~~~~~~~~~

Use the **monochromatic** model when the source is far from merger and its frequency does not
change appreciably over the data span (the usual PTA CW search regime) — it is cheaper and has
fewer parameters. Use the **evolving** model when the binary is massive and/or high-frequency
enough that the frequency drifts over the observation, or when you want to constrain the chirp
mass and exploit the distance-dependent pulsar term.

Choosing between them is also a statement about parameter identifiability: the chirp mass and
pulsar distance are only measurable once the frequency evolves. In the monochromatic limit they
are degenerate with the strain amplitude and a free pulsar phase, respectively, which is why they
are absent from ``makedelay_binary``.

Fourier representation
----------------------

.. code-block:: python

   cw_fourier = ds.makefourier_binary(pulsarterm=True)

The Fourier-domain model projects the (monochromatic) CW waveform directly onto the sine/cosine
Fourier basis used by the rank-reduced / FFT-covariance likelihoods, instead of evaluating the
delay at each TOA. For a basis spanning :math:`[t_0, t_0 + T]`, a component
:math:`A\cos(2\pi f_0 t + \varphi)` is projected analytically onto each basis function via
:func:`~discovery.deterministic.cos2comp`, so no per-TOA evaluation or numerical projection is
needed.

This is useful when the CW is added as a deterministic **mean** of a Fourier-basis GP (for
example as the ``means`` of a global GP), letting the deterministic signal and the stochastic
GP share the same basis. It is most efficient when the likelihood already works in the Fourier
domain.

.. note::

   The Fourier representation is a projection of the **monochromatic** waveform: it assumes a
   single GW frequency. There is no Fourier-domain counterpart to the evolving model, because a
   chirping signal is no longer a single-frequency sinusoid and cannot be projected with the
   closed-form :func:`~discovery.deterministic.cos2comp` integral. Use the time-domain
   ``makedelay_binary(evolve=True)`` when frequency evolution matters.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 28 22 22 28

   * - Model
     - Frequency
     - Domain
     - Distinguishing parameters
   * - ``makedelay_binary`` (``evolve=False``)
     - constant
     - time
     - ``log10_h0``, ``phi_psr``
   * - ``makedelay_binary`` (``evolve=True``)
     - chirping
     - time
     - ``log10_mc``, ``p_dist`` (+ ``log10_h0``)
   * - ``makefourier_binary``
     - constant
     - Fourier
     - (as monochromatic)

See Also
--------

- :doc:`delays` - Deterministic delay models and ``makedelay``
- :doc:`noise_signals` - Stochastic signal components
- :doc:`/api/deterministic` - Deterministic signals API reference
- :func:`~discovery.deterministic.makedelay_binary`
- :func:`~discovery.deterministic.makefourier_binary`
