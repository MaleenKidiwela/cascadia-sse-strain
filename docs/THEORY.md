# Theory and conventions

Everything here is implemented in `src/sse_strain/`; the section headings name
the module.

---

## 1. Slip reconstruction (`solution.py`)

The published solution stores a variational Bayesian ICA decomposition of the
GNSS position time series rather than a slip model. Writing $U$ for the spatial
mixing matrix, $S$ for the component weights and $V$ for the temporal
functions, the inversion recovers, for each retained component $k$, a static
slip pattern $L_{\cdot k}$ from the surface displacement $d_k = U_{\cdot k}$ by
Tarantola–Valette least squares,

$$
\hat{m}_k = m_0 + C_{m}G^{\mathsf T}\left(GC_{m}G^{\mathsf T} + C_{d}\right)^{-1}(d_k - Gm_0),
$$

with a prior covariance whose correlation falls exponentially with inter-patch
distance,

$$
[C_m]_{ij} = E_m^2 \exp\left(-\frac{\|x_i - x_j\|}{\lambda}\right),
\qquad \lambda \approx 21\ \text{km}.
$$

Slip at patch $p$ and epoch $t$ is then

$$
s_p(t) = \sum_{k=1}^{K} L_{pk}\, a_k(t),
\qquad a_k(t) = S_{kk} V_{tk}.
$$

### Why this structure runs through everything

Slip is **low rank**: $K$ static patterns times $K$ scalar time functions,
with $K$ typically 15–30. Strain is linear in slip, so

$$
\varepsilon_{ij}(\mathbf{x}, t) = \sum_k E^{(k)}_{ij}(\mathbf{x})\, a_k(t),
$$

and $K$ forward elastic calculations give the entire 2007–present history.
Epoch-by-epoch evaluation would be several hundred times the work for
identical output. `dataset.component_dataset` stores the two factors
separately; `dataset.evaluate` contracts them for a chosen window.

### Sign and reference conventions

Gualandi removes a secular linear trend before the ICA, so slip is **relative
to the long-term trend**: a patch with negative rate is loading faster than its
secular rate, positive is unloading. There is no positivity constraint,
deliberately, so that a linear sum of components can represent a migrating
source.

`create_model` in the original Julia applies a rake flip to the scalar slip
*magnitude* used for plotting. That operation is nonlinear and is not applied
here — we carry signed strike and dip components, which are linear in the
amplitudes. `Solution.slip_magnitude` reproduces his figures.

### Adding the loading term

Total strain needs the interseismic contribution that the detrending removed.
Two routes:

1. **`fitresult.mat`** holds the trajectory-model fit of the GNSS series, which
   is where that secular trend lives. It is commented out in Gualandi's own
   `driver.jl`. Recovering the loading term from it is plausible and
   unverified — inspect before relying on it.
2. **An independent locking model.** Michel, Gualandi & Avouac (2019, PAGEOPH,
   doi:10.1007/s00024-018-1991-x) derive the Cascadia locking distribution from
   the secular motion of 352 continuous GPS stations, giving two end-member
   models with locked and creeping priors. Back-slip on the interface at
   $\dot{s} = (1 - \phi)\, v_{\text{plate}}$, with $\phi$ the locking fraction,
   then feeds the same Green's functions.

Route 2 is the defensible one for a manuscript; route 1 is worth checking
because it would be internally consistent with the SSE model by construction.

---

## 2. Green's functions (`green.py`)

Displacement and strain use the Nikkhoo & Walter (2015) triangular dislocation
solution for a homogeneous elastic half-space (GJI 201(2), 1119–1141,
doi:10.1093/gji/ggv035), evaluated through `cutde`. The solution is analytic
and artefact-free, and returns the full strain tensor directly, so no separate
Okada implementation or numerical differencing is required.

The only elastic parameter entering the kernels is Poisson's ratio. Gualandi
used $\nu = 0.25$; changing it changes the strain field itself, not merely its
interpretation.

### A convention that must not be got wrong

`cutde.halfspace.disp_free(obs, tris, slips, nu)` and `strain_free` return, for
each observation point, the response summed over **all** triangles. They are
not pairwise `obs[i]`/`tris[i]` evaluations, despite requiring the two arrays
to have equal length. Per-patch Green's function columns therefore require
`disp_matrix`, which returns the unsummed $(n_{\text{obs}}, 3, n_{\text{tri}},
3)$ kernel.

Getting this backwards produces a $G$ whose every column is the whole-fault
response. The inversion still converges and still fits the data, so the error
is invisible in every diagnostic except a direct comparison against
single-triangle forward models. `test_cutde_free_is_summed_not_pairwise` and
`test_displacement_gf_columns_are_single_patch` pin the convention down.

### Verification

Two tests check the physics independently of anything in this package:

- **Free surface.** Tractions $\sigma_{zz}, \sigma_{xz}, \sigma_{yz}$ vanish on
  $z=0$ to $\sim10^{-9}$ of the stress scale, for all three slip directions.
- **Numerical gradient.** The analytic strain equals
  $\tfrac12(\partial_j u_i + \partial_i u_j)$ from central-differenced
  displacements to a relative $2\times10^{-4}$.

---

## 3. Where to evaluate (`megathrust.py`)

Strain exactly on a slipping patch is singular, and near it the field is
controlled by the discretisation. Two distinct effects, often conflated:

**Triangle interior discretisation.** Diagnosed by subdividing every triangle
into four while holding slip fixed. For a smoothly regularised slip model this
changes nothing, because the source is already smooth relative to the patch
scale.

**Piecewise-constant slip.** The real problem. Adjacent patches differ in slip
by 5–8% of peak, and each boundary is a small dislocation edge. Mesh refinement
does *not* remove these, since subdivision leaves the jumps where they were.
Measuring the field's roughness — RMS discrete Laplacian over RMS field —
against evaluation offset shows the effect and where it stops:

| offset above interface | roughness |
|---|---|
| 3 km | 3.14 |
| 5 km | 2.88 |
| 10 km | 2.34 |
| 15 km | 1.93 |
| 20 km | 1.51 |
| 30 km | 1.44 |

The plateau sits near the patch dimension (21.6 km for the fixture). Two
defensible responses, both implemented: evaluate at an offset comparable to the
patch scale, and low-pass the field at the inversion's own prior correlation
length $\lambda$, since the inversion states that nothing finer is resolved.
`resolution_test` re-measures this for any mesh.

Note that this problem largely disappears for the shallow dv/v grid: 1–5 km
depth is already 25–40 km from the interface.

The offset is applied vertically. For a plane dipping at $\delta$, a vertical
offset $h$ is a perpendicular distance $h\cos\delta$; Cascadia dips are 10–20°
in the slow-slip band, so the two differ by under 6%, and the vertical form
keeps the surface single-valued on a lon/lat grid.

### Resolution and grid spacing

The strain field at depth $z$ from slip at depth $h$ varies on a horizontal
scale of order $h$. For Cascadia slow slip at 30–45 km that is 30–50 km, so a
grid much finer than ~10 km resolves nothing the physics contains while costing
quadratically. The same argument sets the scale on which dv/v station pairs
should be binned: pairs whose sensitivity kernels fall inside one strain
wavelength are not independent samples.

---

## 4. Strain and stress (`strain.py`)

East–North–Up throughout, Voigt-like ordering
$[\varepsilon_{xx}, \varepsilon_{yy}, \varepsilon_{zz}, \varepsilon_{xy},
\varepsilon_{xz}, \varepsilon_{yz}]$ with **tensor** off-diagonals
($\varepsilon_{xy} = \tfrac12(\partial_y u_x + \partial_x u_y)$), not
engineering shears. Extension positive, hence tension positive for stress.

$$
\varepsilon_{kk} = \varepsilon_{11}+\varepsilon_{22}+\varepsilon_{33},
\qquad
e_{ij} = \varepsilon_{ij} - \tfrac13 \varepsilon_{kk}\delta_{ij},
$$

$$
J_2' = \tfrac12 e_{ij}e_{ij},
\qquad
\varepsilon_{eq} = \sqrt{\tfrac43 J_2'},
\qquad
\gamma_{\max} = \varepsilon_1 - \varepsilon_3 .
$$

Hooke's law for an isotropic medium, mean stress and pressure, von Mises and
maximum shear:

$$
\sigma_{ij} = \lambda \varepsilon_{kk}\delta_{ij} + 2\mu\varepsilon_{ij},
\qquad
\sigma_m = \tfrac13\sigma_{kk}, \quad p = -\sigma_m,
$$

$$
\sigma_{eq} = \sqrt{3J_2}, \qquad \tau_{\max} = \tfrac12(\sigma_1-\sigma_3).
$$

Coulomb failure stress on a receiver plane with unit normal $n$, slip direction
$s$ and apparent friction $\mu'$:

$$
\Delta \mathrm{CFS} = s_i\sigma_{ij}n_j + \mu'\, n_i\sigma_{ij}n_j ,
$$

with the normal term positive in tension, so unclamping raises $\Delta$CFS. The
result scales linearly with $\mu'$, so report the value used; 0.0–0.4 is the
usual range for a subduction interface.

### Which quantity to compare against dv/v

No single right answer, which is why several are computed and kept in one
Dataset. Laboratory and field work on the stress sensitivity of seismic
velocity generally finds the strongest control through mean or confining stress
acting on crack populations, making `pressure` and `dilatation` the natural
first targets. Deviatoric measures matter if crack opening is
orientation-selective. Keeping all of them means the correlation analysis needs
no recomputation.

---

## 5. Elastic properties (`elastic.py`)

Two parameterisations, deliberately separate:

1. **The half-space for the Green's functions.** Only $\nu$ enters. Use the
   inversion's value.
2. **The moduli for Hooke's law.** $\mu(z)$ and $\lambda(z)$ at the observation
   point, from a velocity model.

Doing (2) with depth dependence while (1) assumes homogeneity is internally
inconsistent: a layered medium would modify the strain field too, not only the
strain-to-stress map. The inconsistency is second order for a smoothly varying
crust and is the standard compromise in the Coulomb stress literature, but it
is a real approximation, and `ElasticModel.metadata()` records the choice so it
travels with the data. `ElasticModel.homogeneous()` is the fully consistent
alternative.

Density and $V_s$ from $V_p$ use the Brocher (2005) Nafe–Drake and $V_p$–$V_s$
regressions (BSSA 95(6), 2081–2092, equations 1 and 9), **transcribed from the
standard published forms and not checked against the paper**. The shipped 1-D
profile is `placeholder_cascadia()`, marked `is_placeholder` and propagated as
`elastic_is_placeholder = 1` in the output.

---

## 6. The measurement operator (`smoothing.py`)

The dv/v series is sampled daily but each sample is a 31-day centred average.
That window is part of the measurement, so the model goes through the same
operator; comparing an unsmoothed strain field against a smoothed dv/v
attributes the filter's effect to the physics.

Because a linear time-invariant operator $W$ commutes with the spatial sum,

$$
W[\varepsilon_{ij}](\mathbf{x},t) = \sum_k E^{(k)}_{ij}(\mathbf{x})\, W[a_k](t),
$$

smoothing the $K$ amplitude functions is exact and costs microseconds.
`test_smoothing_commutes_with_spatial_sum` verifies this to $10^{-10}$.

### Cumulative, not rate

dv/v is a state variable relative to a reference stack; its counterpart is
cumulative strain, which is also what the boxcar treats kindly. A step keeps
its full amplitude and is smeared over one window. A rate pulse of duration
$T < W$ keeps only $T/W$ of its peak:

| episode duration | peak retained (W = 31 d) |
|---|---|
| 7 d | 0.23 |
| 14 d | 0.45 |
| 21 d | 0.68 |
| 25 d | 0.81 |
| ≥31 d | 1.00 |

Since the attenuation depends on duration, a moment–duration or scaling
analysis done on smoothed rates is biased in a way that mimics the physics
being tested.

### The sidelobe

A 31-day boxcar has amplitude response $|H(f)| = |\sin(\pi f W)/(W\sin \pi f)|$,
with zeros at periods of 31.0, 15.5 and 10.3 days and a **first sidelobe of
$-0.218$ at a 21.7-day period**. Cascadia ETS episodes sit on it. In rate space
a three-week transient arrives at about a fifth of amplitude *and with reversed
sign*, so an apparent anticorrelation between dv/v and strain rate at those
periods must be ruled out as a filter artefact. Cumulative space avoids this.

### Referencing

The reference is the last year of the stack. Detrended cumulative slip is close
to a zero-mean saw-tooth, and a reference average removes it exactly only when
the reference length is a whole number of recurrence intervals. Cascadia
recurrence runs 8–22 months along strike, so the residual offset varies
systematically with latitude:

| recurrence | residual offset, % of peak-to-peak |
|---|---|
| 8 mo | 8 |
| 12 mo | 0 |
| 14 mo | 7 |
| 16 mo | 12 |
| 20 mo | 20 |
| 22 mo | 22 |

This is a constant offset, so it biases any regression forced through the
origin, and differently in the north than the south — which would read as a
real along-strike gradient in the strain–dv/v relationship. **Fit an intercept.**
`saw_tooth_reference_bias()` computes the number for a given recurrence.

A reference that varies by pair cannot be folded into the amplitudes, because
the subtracted constant becomes a field; `dataset.reference_field` applies it
after evaluation.

### Degrees of freedom

Neighbouring daily samples of a $W$-day average share $W-1$ inputs. For 5500
daily samples under a 31-day window, roughly **160 independent observations**,
not 5470. `effective_sample_size` computes this from the empirical
autocorrelations by Bartlett's formula, which the $3N/2W$ rule of thumb
overestimates by about 1.6× here. Feed $N_{\text{eff}} - 2$ to any $t$-test on
a correlation coefficient.

Pairs within one strain wavelength are not independent spatially either, so the
naive $N_{\text{pairs}} \times N_{\text{time}}$ overstates precision by two to
three orders of magnitude.

### Fitting the pair offsets

dv/v is defined against a reference stack, so each pair carries an unknown
offset $c_p$. Writing the space–time inversion as $d = Gm + Sc + \varepsilon$
with $S$ the pair-indicator matrix, $S^{\mathsf T}WS$ is diagonal, and
eliminating $c$ gives

$$
(G^{\mathsf T}\tilde{W}G + \Lambda)\hat{m} = G^{\mathsf T}\tilde{W}d,
\qquad
\tilde{W} = W - WS(S^{\mathsf T}WS)^{-1}S^{\mathsf T}W .
$$

$\tilde{W}$ is the projector removing each pair's weighted mean, so solving for
the offsets is algebraically identical to demeaning every pair — the "within"
transformation of a fixed-effects panel regression. Keeping $c$ explicit yields
its posterior covariance, and pairs with few epochs get correspondingly loose
offsets instead of being demeaned by a noisy sample mean.

A static field $f(\mathbf{x})$ with $c_p = -K_p^{\mathsf T}f$ is annihilated by
the augmented operator, so the null space has dimension $M$; the reference-window
constraint supplies exactly $M$ constraints and removes it. Regularisation on
$m$ must not fight that gauge — damp toward the reference-window mean, or apply
the constraint as a hard projection.

**Keep sensitivities global.** One free scale factor per pair absorbs real
signal, and with a few hundred pairs the fit becomes unfalsifiable. Let pairs
differ by offset only, or share sensitivities within a region or depth band if
testing for variation.

---

## References

- Brocher, T. M. (2005). Empirical relations between elastic wavespeeds and density in the Earth's crust. *BSSA* **95**(6), 2081–2092.
- Gualandi, A. (2025). Near real-time Cascadia slow slip events. *GJI* **242**(2), ggaf198. doi:10.1093/gji/ggaf198
- Michel, S., Gualandi, A., & Avouac, J.-P. (2019a). Interseismic coupling and slow slip events on the Cascadia megathrust. *PAGEOPH* **176**, 3867–3891. doi:10.1007/s00024-018-1991-x
- Michel, S., Gualandi, A., & Avouac, J.-P. (2019b). Similar scaling laws for earthquakes and Cascadia slow-slip events. *Nature* **574**, 522–526. doi:10.1038/s41586-019-1673-6
- Nikkhoo, M., & Walter, T. R. (2015). Triangular dislocation: an analytical, artefact-free solution. *GJI* **201**(2), 1119–1141. doi:10.1093/gji/ggv035
