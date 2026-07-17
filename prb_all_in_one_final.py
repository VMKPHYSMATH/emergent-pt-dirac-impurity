#!/usr/bin/env python3
"""
prb_all_in_one_PRB_final_uniform.py

Final uniform-lineweight, self-contained figure generator for
"Emergent PT Symmetry and Exceptional Points in a Driven Dirac Impurity"
.

This single file merges the validated physics solver with a matplotlib
figure driver that is a faithful port of the all_in_one_2.py plotting layer:
identical colors (C dict), line widths, serif/STIX fonts, ticks-in styling,
gridspec layout, dashed EP markers (annotate_ep), alpha-fade phase portraits
with directional arrows (fade_plot/add_arrow), signed waterfall spectra,
two-axis supplement panel, with main-text insets removed.  There is no external solver
module to import and no intermediate XMGrace step, so the figures, the
exported .dat curves, and the manuscript captions all come from one
consistent parameter set.

Physics conventions (all consistent with the revised manuscript):
- The slave-boson phase is gauge-fixed: only r = |b_c| enters observables.
  The gauge-invariant amplitude tbeta = |beta0| r is projected into three
  independent real channels -- gain/loss Gamma_PT = c_gamma tbeta, coherent
  splitting Delta_eff = delta0 + c_delta tbeta, and bath hybridization
  V_eff = c_V tbeta.  With delta0 != 0 the local impurity EP is isolated at a
  finite drive rather than sitting on an EP ray.
- Spectra use the physical retarded/advanced DOS
  A_imp = -Im Tr_imp[G^R - G^A]/(2 pi).  The signed form is plotted so any
  negative (PT-broken / causality-diagnostic) region is shown explicitly.
- The bath-controlled lesser G^< = f_FD (G^R - G^A) is the only physical FDR
  object; the eigenmode-occupation ratio is a non-thermal diagnostic clipped
  to +/-1.5 and never used for self-consistency.
- Fig. 3 now also exports two operational SBMF spectral checks: a low-energy
  HWHM scale extracted from the physical impurity spectrum and a bath-FDR
  temperature-sweep half-suppression scale.  These are independent checks of
  the frozen SW/BA screening diagnostics, not inputs to them.

Consistency fixes relative to the fragmented legacy scripts:
- Fig. 4 now uses the uniform convention eps_xi = -U/2 = -1.0, U = 2.0,
  T = 0.1 (ModelParams.eps_xi_5), matching the caption's Re(eps_tilde) =
  -1.0000.  The legacy value eps_xi_5 = -2.0 is removed.
- The bath-constructed FDR residual is computed directly from the block
  ratios |F - f_FD|; it sits at the numerical floor (~1e-14) as it must.
- The duplicate definition of local_impurity_splitting_squared() is removed.
- The XMGrace panel writers are replaced by a single matplotlib driver that
  emits Fig1-Fig4 as PDF/PNG plus the underlying .dat curves and captions.tex.

Usage:
    python prb_all_in_one_PRB_final_uniform.py --out figs_unified                 # all
    python prb_all_in_one_PRB_final_uniform.py --which fig4                        # one
    python prb_all_in_one_PRB_final_uniform.py --which fig4                        # sub-EP/causal main Fig. 4
    python prb_all_in_one_PRB_final_uniform.py --which fig4 --fig4-beta0 0.58      # explicit broken-side diagnostic override
    python prb_all_in_one_PRB_final_uniform.py --dos-betas 0.30 0.42 0.50 0.58     # Fig.3 waterfall

Requires: numpy, scipy, matplotlib.
"""
from __future__ import annotations


import math
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.ndimage import median_filter, gaussian_filter1d
    from scipy.signal import find_peaks, savgol_filter
    from scipy.linalg import expm
except Exception as exc:
    raise RuntimeError("This script requires scipy (optimize, ndimage, signal).") from exc


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
@dataclass
class ModelParams:
    Nk: int = 100
    k_max: float = math.pi / 4
    D_uv: float = math.pi / 4
    lambda_k: float = math.pi / 4  # manuscript default: lambda=k_max; D_uv is the energy cutoff
    mix_alpha: float = 0.20
    tol: float = 1e-8
    max_iter: int = 2000

    eps_xi_12: float = -1.0
    T_12: float = 0.1
    beta_vals: np.ndarray = field(default_factory=lambda: np.linspace(-1.5, 1.5, 401))

    eps_xi_4: float = -1.0
    T_4: float = 5e-4
    beta0_4: float = 0.50

    # Fig. 1 time-domain trajectory choices.  These are multipliers of
    # beta_EP for the PT-unbroken and PT-broken curves; they are exposed
    # through --fig1-left-scale and --fig1-right-scale.
    fig1_left_scale: float = 0.10
    fig1_right_scale: float = 1.50
    # Default False preserves readable shapes by normalizing each phase orbit
    # separately.  Use --phase-common-norm to reveal absolute growth/decay
    # differences in the phase portrait.
    phase_common_norm: bool = False

    # Anderson/Kondo diagnostics used only to classify the plotted frozen spectra.
    # The 4x4 frozen kernel itself remains bilinear; U enters the analytic
    # Schrieffer--Wolff/Kondo-scale labels and diagnostics below.
    U_default: float = 2.0
    chi_K: float = 1.0
    rho_eff: float = 1.0 / (2.0 * (math.pi / 4))

    # Parameters for the smooth frozen EP-assisted screening-scale diagnostic.
    # These are not extracted from the noisy spectral HWHM.  They implement the
    # analytic idea Gamma_BA = Z_EP Gamma with a bounded biorthogonal enhancement.
    Z_EP_floor: float = 0.03
    Z_EP_max: float = 25.0

    # Operational SBMF spectral scale checks used in Fig. 3.
    # hwhm: low-energy half-width at half maximum of the bath-FDR physical spectrum.
    # temp-sweep: temperature where the FDR-window spectral weight is halfway
    # between its low-T and high-T values.  Neither quantity is used to define
    # the BA/SW scale; they are plotted as independent checks.
    eta_sbmf_hwhm: float = 0.015
    sbmf_hwhm_window: float = 0.45
    sbmf_temp_window: float = 0.55
    sbmf_T_grid: np.ndarray = field(default_factory=lambda: np.geomspace(5e-4, 8e-1, 12))

    # Low-energy channel projection coefficients.  These are real gauge-fixed
    # coefficients multiplying tbeta = |beta0| |b_c|.  c_V is fixed to 1 by the
    # definition of tbeta unless a different hybridization normalization is desired.
    # c_gamma controls the anti-Hermitian gain/loss channel; c_delta controls the
    # coherent spin-flip/splitting channel.  delta0 is a beta-independent coherent
    # splitting generated by the frozen projection.  It is essential: if delta0=0
    # and c_gamma=c_delta, the local impurity block sits on an EP ray instead of
    # crossing an isolated EP.  B_imp is an optional real Zeeman splitting on the
    # impurity, independent of the slave-boson phase.
    c_gamma: float = 1.0
    c_delta: float = 0.25
    delta0: float = 0.375  # gives |beta0_EP|=0.5 for r_fixed=1, c_gamma=1, c_delta=0.25
    c_V: float = 1.5
    B_imp: float = 0.0

    # Slave-boson constraint used by the diagnostic SBMF closure.  For the
    # physical spin-1/2 infinite-U Anderson constraint use Q=1.  If a large-N
    # normalization is intended, change Q_constraint and state it in the manuscript.
    Q_constraint: float = 1.0
    b2_clip_max: float = 1.0

    eta_broad: float = 1.5e-3
    eta_fine: float = 5e-5

    # Manuscript plotting broadenings.  The raw fine-grid spectral data are still
    # available, but the displayed DOS/peak scans use modest broadening and
    # smoothing so finite-k-grid spikes are not mistaken for physics.
    eta_dos_plot: float = 3.0e-4
    eta_scan_plot: float = 2.0e-4
    smooth_window: int = 7
    smooth_polyorder: int = 3

    # Fig. 4 (FDR + physical spectral response) uniform convention.
    # Unified with the manuscript caption: eps_xi = -U/2 = -1.0, U = 2.0, T = 0.1.
    # (Earlier solver versions used eps_xi_5 = -2.0, which did NOT match the
    #  Fig. 4 caption's stated Re(eps_tilde) = -1.0000. Fixed here.)
    #
    # eta_fdr is the sharp broadening used for the FDR-ratio panels (a)-(c),
    # where the near-delta structure is desired.  eta_fig4_spec is a modest
    # PHYSICAL broadening used only for the panel-(d) impurity spectral
    # response: below/at the EP the frozen-kernel DOS is causal (A>=0) once
    # broadened at this scale, so the sharp-eta sub-resolution ringing that
    # produced spurious negative dips is removed.  A genuine negative response
    # then appears only in the PT-broken phase (beta0 > beta0_EP).
    eps_xi_5: float = -1.0
    T_5: float = 0.10
    eta_fdr: float = 1e-3
    eta_fig4_spec: float = 0.015
    # Default Fig. 4 beta0 is deliberately sub-EP: the main FDR/DOS figure should
    # show the physical, causal spectrum.  The PT-broken negative-DOS response is
    # kept for the supplemental onset/broadening check, not the main panel.
    beta0_fig4_spec: float = 0.45
    # Representative above-EP value used only in SFig4 to expose the causality defect.
    beta0_sfig4_above_ep: float = 0.58
    # Broadening-scan values used only for the supplemental robustness check.
    # The middle value equals eta_fig4_spec; the negative DOS should persist
    # across this moderate interval; sharper/wider values are treated as grid/window-sensitive checks.
    eta_fig4_scan: tuple[float, ...] = (0.014, 0.015, 0.020)
    omega_fdr: np.ndarray = field(default_factory=lambda: np.linspace(-4.0, 4.0, 400))

    beta_sweep3: np.ndarray = field(default_factory=lambda: np.linspace(-1.5, 1.5, 121))
    omega_dos3: np.ndarray = field(default_factory=lambda: np.linspace(-3.5, 2.0, 400))
    beta0_dos3: float = 0.25

    # Closure used by solve().  The default is a frozen-amplitude diagnostic
    # kernel because the exact self-consistent nonequilibrium SBMF closure is not
    # implemented here.  Set scf_closure="eigenmode_diagnostic" to reproduce the
    # old fast fixed-point workflow, which is useful only as a numerical diagnostic.
    # The physical FDR construction is used in the DOS/FDR panels.
    scf_closure: str = "frozen_amplitude"
    r_fixed: float = 1.0

P = ModelParams()


@dataclass(frozen=True)
class RegimeSpec:
    """Explicit labels for the frozen spectra.

    The spectral kernel is bilinear.  The regime label is assigned using the
    Anderson parameters and the Schrieffer--Wolff scale at beta_ref; it is not
    inferred from visual similarity of the DOS curves.
    """
    label: str
    eps_xi: float
    U: float
    T: float
    beta_ref: float = 0.50
    description: str = ""


REGIMES = [
    RegimeSpec(
        "Kondo_local_moment",
        eps_xi=-1.0, U=2.0, T=0.005, beta_ref=0.47,
        description="local moment: eps_d<0<eps_d+U, Gamma smaller than charge gaps, T<TK_SW near the EP",
    ),
    RegimeSpec(
        "Resonant_mixed_valence",
        eps_xi=0.0, U=0.0, T=0.05, beta_ref=0.47,
        description="resonant/mixed-valence reference: level near the Fermi energy, no local-moment SW scale",
    ),
    RegimeSpec(
        "Free_orbital_highT",
        eps_xi=-1.0, U=2.0, T=10.0, beta_ref=0.47,
        description="same Anderson level as local moment but T >> TK_SW, hence free-orbital/high-temperature reference",
    ),
]


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_nxy(filename: str | Path, datasets: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    filename = Path(filename)
    with open(filename, "w", encoding="utf-8") as f:
        for label, x, y in datasets:
            if label:
                f.write(f"# {label}\n")
            for xi, yi in zip(np.asarray(x), np.asarray(y)):
                f.write(f"{float(np.real(xi)):.12e} {float(np.real(yi)):.12e}\n")
            f.write("&\n")


def vline_dataset(x0: float, y0: float, y1: float, label: str) -> tuple[str, np.ndarray, np.ndarray]:
    return (label, np.array([x0, x0]), np.array([y0, y1]))


def hline_dataset(x0: float, x1: float, y0: float, label: str) -> tuple[str, np.ndarray, np.ndarray]:
    return (label, np.array([x0, x1]), np.array([y0, y0]))


# -----------------------------------------------------------------------------
# Core numerics
# -----------------------------------------------------------------------------
def f_fermi(eps: np.ndarray | complex | float, T: float) -> np.ndarray | float:
    x = np.clip(np.real(eps) / T, -500, 500)
    return 1.0 / (np.exp(x) + 1.0)


def make_k_grids(Nk: int, D: float) -> tuple[np.ndarray, np.ndarray]:
    k_lin = np.linspace(-D, D, Nk)
    half = np.logspace(np.log10(1e-4), np.log10(D), Nk // 2)
    if Nk % 2 == 0:
        k_log = np.concatenate([-half[::-1], half])
    else:
        k_log = np.concatenate([-half[::-1], [0.0], half])
    return k_lin, k_log

K_LINEAR, K_FINE = make_k_grids(P.Nk, P.k_max)


def gauge_tbeta(beta0: float, b: complex | float) -> float:
    """Gauge-invariant nonnegative control amplitude: tbeta = |beta0| |b_c|.

    The revised manuscript attributes no observable to arg(b_c).  In the
    symmetric frozen-kernel convention used for the figures, the sign of beta0
    is treated as a drive/projection convention and is not allowed to change the
    physical low-energy amplitudes.  This makes spectra, condition numbers and
    DOS even functions of beta0.  Do not use beta0*b_c here.
    """
    return abs(float(beta0)) * float(abs(b))


def _real_scalar(x: complex | float, name: str) -> float:
    """Return a real scalar, rejecting gauge-contaminating complex amplitudes."""
    if abs(np.imag(x)) > 1e-10:
        raise ValueError(f"{name} must be real in the gauge-fixed Hamiltonian.")
    return float(np.real(x))


def channel_scales(tbeta: complex | float, flip: bool = True) -> tuple[float, float, float, float]:
    """Project the gauge-invariant amplitude into physical low-energy channels.

    tbeta      = |beta0| |b_c| is the nonnegative gauge-invariant control.
    gamma_pt   = c_gamma * tbeta is the anti-Hermitian gain/loss channel.
    delta_flip = delta0 + c_delta * tbeta is the coherent real spin-flip/splitting
                 channel when flip=True; it is set to zero when flip=False.
    V_eff      = c_V * tbeta is the impurity-bath hybridization amplitude.
    B_imp      is an optional real Zeeman splitting on the impurity diagonal.

    The phase of b_c never enters these quantities.  The sign of beta0 is not
    used in the default symmetric plotting convention; beta0 and -beta0 then
    give identical spectra and DOS.
    """
    tb = _real_scalar(tbeta, "tbeta")
    gamma_pt = float(P.c_gamma) * tb
    delta_flip = (float(P.delta0) + float(P.c_delta) * tb) if flip else 0.0
    V_eff = float(P.c_V) * tb
    B_imp = float(P.B_imp)
    return gamma_pt, delta_flip, V_eff, B_imp


def build_Hss(teps: complex, tbeta: complex | float, k: float, flip: bool = True,
              gamma_pt: complex | float | None = None,
              delta_flip: complex | float | None = None,
              V_eff: complex | float | None = None,
              B_imp: complex | float | None = None) -> np.ndarray:
    """Gauge-fixed 4x4 PT kernel with separated projection coefficients.

    In the code basis the impurity block is
        [[teps+B+i*Gamma, Delta],
         [Delta, teps-B-i*Gamma]].

    Defaults are generated from tbeta=|beta0||b_c| using channel_scales().
    Optional keyword arguments are only for controlled analytic tests.
    """
    gamma0, delta0, V0, B0 = channel_scales(tbeta, flip=flip)
    if gamma_pt is None:
        gamma_pt = gamma0
    else:
        gamma_pt = _real_scalar(gamma_pt, "gamma_pt")
    if delta_flip is None:
        delta_flip = delta0
    else:
        delta_flip = _real_scalar(delta_flip, "delta_flip")
    if V_eff is None:
        V_eff = V0
    else:
        V_eff = _real_scalar(V_eff, "V_eff")
    if B_imp is None:
        B_imp = B0
    else:
        B_imp = _real_scalar(B_imp, "B_imp")

    ep = k**2 + P.lambda_k * k
    em = k**2 - P.lambda_k * k
    V = V_eff / np.sqrt(2)
    return np.array([
        [teps + B_imp + 1j * gamma_pt, delta_flip, V, V],
        [delta_flip, teps - B_imp - 1j * gamma_pt, V, -V],
        [V, V, ep, 0.0],
        [V, -V, 0.0, em],
    ], dtype=complex)


def eig_lr(H: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right eigenvectors and left rows with L @ R approximately equal to I.

    For a diagonalizable matrix, L is exactly R^{-1}.  Near an EP the inverse can
    become ill-conditioned; the pseudoinverse fallback is only a numerical guard.
    We do not use the normalization of L/R to define physical scales.
    """
    ev, R = np.linalg.eig(H)
    try:
        L = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        L = np.linalg.pinv(R)
    # If the pseudoinverse fallback was used, enforce unit diagonal overlaps when possible.
    for n in range(len(ev)):
        ov = L[n, :] @ R[:, n]
        if abs(ov) > 1e-14:
            L[n, :] /= ov
    return ev, R, L


def eigvec_condition_number(H: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Matrix eigenvector condition number kappa=||R||_2 ||R^{-1}||_2.

    This is the quantity denoted kappa_imp in the revised manuscript.  It is
    an EP-distance/nonnormality diagnostic, not a direct Kondo scale.
    """
    ev, R = np.linalg.eig(H)
    try:
        Ri = np.linalg.inv(R)
        kap = np.linalg.cond(R)
    except np.linalg.LinAlgError:
        Ri = np.linalg.pinv(R)
        kap = np.linalg.norm(R, 2) * np.linalg.norm(Ri, 2)
    return float(np.real(kap)), ev, R, Ri


def petermann_metric(H: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Per-state nonorthogonality factor retained only for diagnostics."""
    ev, R, L = eig_lr(H)
    vals = [np.linalg.norm(R[:, n]) * np.linalg.norm(L[n, :]) for n in range(R.shape[1])]
    return float(np.max(vals)), ev, R, L


def ep_metric(H: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible alias for the manuscript kappa_imp."""
    return eigvec_condition_number(H)


def solve(beta0: float, eps_xi: float, T: float, flip: bool = True,
          d0: complex = 0.1 + 0.0j, b0: complex = 0.1 + 0.0j,
          kgrid: np.ndarray | None = None) -> tuple[complex, complex]:
    """Return the frozen/diagnostic saddle fields.

    The default closure, ``frozen_amplitude``, is intentionally simple and
    produces the frozen PT kernel used for EP and TBA diagnostics.  It avoids
    making the fast eigenmode lesser closure look like a physical nonequilibrium
    SBMF solution.  The optional ``eigenmode_diagnostic`` closure is retained for
    continuity checks only.
    """
    if kgrid is None:
        kgrid = K_LINEAR
    if P.scf_closure == "frozen_amplitude":
        return 0.0 + 0j, float(P.r_fixed) + 0j
    if P.scf_closure != "eigenmode_diagnostic":
        raise ValueError(f"Unknown scf_closure={P.scf_closure!r}")
    d, b = d0, b0
    for _ in range(P.max_iter):
        d_old, b_old = d, b
        tbeta = gauge_tbeta(beta0, b_old)
        teps = eps_xi + np.real(d_old)
        sc = 0j
        cm = 0j
        xx = 0j
        for k in kgrid:
            Hk = build_Hss(teps, tbeta, k, flip=flip)
            ev, R, L = eig_lr(Hk)
            # Fast diagnostic SBMF closure.  This eigenmode object is not
            # used as a physical FDR distribution in the manuscript; it is
            # retained only to reproduce the original fixed-point workflow.
            Gl = 1j * (R @ np.diag(f_fermi(ev, T)) @ L)
            sc += Gl[0, 2] + Gl[1, 2]
            cm += Gl[0, 3] + Gl[1, 3]
            xx += Gl[0, 0] + Gl[1, 1]
        nk = len(kgrid)
        sc /= nk
        cm /= nk
        xx /= nk
        sb = float(abs(b_old)) if abs(b_old) > 1e-12 else 1e-12
        # Diagnostic saddle closure.  The derivative of the bath hybridization
        # channel gives a factor c_V*beta0.  The projected gain/loss and splitting
        # channels are treated as frozen low-energy fields in this fast closure;
        # using their gauge phase as an additional saddle variable would reintroduce
        # an unphysical phase dependence.
        V_bare = float(P.c_V) * abs(float(beta0))
        d_new = np.real((-2j * V_bare * sc + 2 * V_bare * cm) / sb) + 0j
        # With G^<_{ff}=i<n_f>, the constraint r^2 + n_f = Q gives
        # r^2 = Q + i Tr G^<_{ff}.  The old sign Q - iG^< forces artificial
        # saturation of r and can hide the self-consistent restructuring.
        bc2_raw = np.real(float(P.Q_constraint) + 1j * xx)
        bc2 = np.clip(bc2_raw, 0.0, float(P.b2_clip_max))
        # Gauge-fix b_c to a positive real amplitude.  The phase of b_c is
        # an internal U(1) gauge convention and must not enter observables.
        b_new = np.sqrt(bc2) + 0j
        d = P.mix_alpha * d_new + (1 - P.mix_alpha) * d_old
        b = abs(P.mix_alpha * b_new + (1 - P.mix_alpha) * b_old) + 0j
        if abs(d - d_old) < P.tol and abs(b - b_old) < P.tol:
            break
    return np.real(d) + 0j, b


def physical_impurity_dos(teps: complex, tbeta_sc: complex | float, omega_grid: np.ndarray, eta: float,
                          flip: bool = True, kgrid: np.ndarray | None = None) -> np.ndarray:
    """Physical impurity spectral function from retarded/advanced Green functions.

    A_imp(omega) = -Im Tr_imp[GR(omega)-GA(omega)]/(2*pi).
    Vectorized over omega for each k for speed.  Unlike
    causalized_impurity_dos(), this function does not force positive residues.
    """
    if kgrid is None:
        kgrid = K_FINE
    tbeta_sc = _real_scalar(tbeta_sc, "tbeta_sc")
    omega = np.asarray(omega_grid, dtype=float)
    nk = len(kgrid)
    out = np.zeros(len(omega), dtype=float)
    I4 = np.eye(4, dtype=complex)
    for k in kgrid:
        Hk = build_Hss(teps, tbeta_sc, k, flip=flip)
        Hka = Hk.conj().T
        GR = np.linalg.inv(omega[:, None, None] * I4 + 1j * eta * I4 - Hk[None, :, :])
        GA = np.linalg.inv(omega[:, None, None] * I4 - 1j * eta * I4 - Hka[None, :, :])
        tr_imp = np.trace(GR[:, :2, :2] - GA[:, :2, :2], axis1=1, axis2=2)
        out += -np.imag(tr_imp) / (2.0 * np.pi * nk)
    return out


def causalized_impurity_dos(teps: complex, tbeta_sc: complex | float, omega_grid: np.ndarray, eta: float,
                            flip: bool = True, kgrid: np.ndarray | None = None) -> np.ndarray:
    """Positive Lorentzian plotting diagnostic, not a physical DOS.

    This helper is retained for comparisons only.  It enforces positive
    weights/widths by construction, so it must not be used as evidence for a
    physical spectral enhancement.
    """
    if kgrid is None:
        kgrid = K_FINE
    if abs(np.imag(tbeta_sc)) > 1e-10:
        raise ValueError("causalized_impurity_dos received complex tbeta; use gauge_tbeta(beta0, b).")
    tbeta_sc = float(np.real(tbeta_sc))
    nk = len(kgrid)
    A = np.zeros(len(omega_grid), dtype=float)
    imp_proj = np.diag([1.0, 1.0, 0.0, 0.0]).astype(complex)
    width_floor = max(eta, 2e-4)

    for k in kgrid:
        Hk = build_Hss(teps, tbeta_sc, k, flip=flip)
        ev, R, L = eig_lr(Hk)
        # positive impurity residues from biorthogonal projector weight on impurity subspace
        weights = []
        for n in range(len(ev)):
            proj = imp_proj @ np.outer(R[:, n], L[n, :]) @ imp_proj
            w = float(np.real(np.trace(proj)))
            # ensure nonnegative plotting weight
            w = abs(w)
            weights.append(w)
        weights = np.array(weights)
        if np.sum(weights) > 1e-15:
            weights /= np.sum(weights)
        else:
            weights[:] = 1.0 / len(weights)

        centers = np.real(ev)
        widths = np.maximum(np.abs(np.imag(ev)), width_floor)
        for c, g, w in zip(centers, widths, weights):
            A += (w * g / np.pi) / ((omega_grid - c) ** 2 + g ** 2) / nk
    return A


def nh_response(teps: complex, tbeta_sc: complex | float, omega_grid: np.ndarray, eta: float,
                flip: bool = True, kgrid: np.ndarray | None = None) -> np.ndarray:
    """Sign-indefinite NH response retained only when explicitly needed for comparison."""
    if kgrid is None:
        kgrid = K_FINE
    tbeta_sc = _real_scalar(tbeta_sc, "tbeta_sc")
    omega = np.asarray(omega_grid, dtype=float)
    nk = len(kgrid)
    out = np.zeros(len(omega), dtype=float)
    I4 = np.eye(4, dtype=complex)
    for k in kgrid:
        Hk = build_Hss(teps, tbeta_sc, k, flip=flip)
        GR = np.linalg.inv(omega[:, None, None] * I4 + 1j * eta * I4 - Hk[None, :, :])
        tr_imp = np.trace(GR[:, :2, :2], axis1=1, axis2=2)
        out += -np.imag(tr_imp) / (np.pi * nk)
    return out


def match_branches(ev_new: np.ndarray, ev_prev: np.ndarray) -> np.ndarray:
    cost = np.abs(ev_new[None, :] - ev_prev[:, None]) ** 2
    _, col_ind = linear_sum_assignment(cost)
    return ev_new[col_ind]


def smooth_kappa(kappa: np.ndarray) -> np.ndarray:
    kap = np.maximum(np.abs(np.asarray(kappa, dtype=float)), 1e-30)
    y = np.log10(kap)
    y = median_filter(y, size=7, mode="nearest")
    win = min(15, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if win >= 5:
        y = savgol_filter(y, window_length=win, polyorder=3)
    return 10**y


def smooth_for_plot(y: np.ndarray, window: int | None = None, polyorder: int | None = None) -> np.ndarray:
    """Mild deterministic smoothing for plotting diagnostics only.

    The raw values are still written to debug files.  This smoother suppresses
    finite-k-grid spike jitter without changing the location of broad features.
    """
    arr = np.asarray(y, dtype=float)
    if arr.size == 0:
        return arr.copy()
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr.copy()
    x = np.arange(arr.size)
    filled = np.interp(x, x[finite], arr[finite])
    if window is None:
        window = int(P.smooth_window)
    if polyorder is None:
        polyorder = int(P.smooth_polyorder)
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if window >= arr.size:
        window = arr.size if arr.size % 2 == 1 else arr.size - 1
    # Median + Gaussian filtering is less prone to Gibbs/Savitzky overshoot for
    # positive spectral weights and areas.
    filt = median_filter(filled, size=min(window, arr.size), mode="nearest")
    sigma = max(0.8, 0.22 * window)
    return gaussian_filter1d(filt, sigma=sigma, mode="nearest")




def smooth_finite_segments(y: np.ndarray, window: int | None = None) -> np.ndarray:
    """Smooth only contiguous finite segments and keep invalid points as NaN.

    Used for spectral FWHM diagnostics whose domain of validity stops when
    the signed relative spectrum develops negative weight or no proper
    half-maximum crossings.  This prevents smoothing from visually extending a
    linewidth into the PT-broken/non-Lorentzian regime.
    """
    arr = np.asarray(y, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return out
    idx = np.where(finite)[0]
    # Split into contiguous runs.
    cuts = np.where(np.diff(idx) > 1)[0] + 1
    for run in np.split(idx, cuts):
        if run.size == 0:
            continue
        vals = arr[run]
        if run.size >= 5:
            out[run] = smooth_for_plot(vals, window=min(window or int(P.smooth_window), run.size if run.size % 2 else run.size-1))
        else:
            out[run] = vals
    return out


def extract_positive_fsr_hwhm(omega: np.ndarray, A_signed: np.ndarray,
                              center: float, halfwidth: float,
                              neg_rel_tol: float = 2e-3,
                              neg_abs_tol: float = 1e-9) -> tuple[float, float, float, bool, str, float]:
    """Peak-resolved FSR HWHM with a strict physical-validity mask.

    The FWHM/HWHM diagnostic is meaningful only for a positive, locally
    resolved resonance.  It is not assigned once the projected relative
    spectrum has appreciable signed negative weight or when the local line
    shape lacks two half-maximum crossings.

    Returns (peak_position, peak_height, hwhm, valid, reason, negative_weight).
    """
    omega = np.asarray(omega, dtype=float)
    A = np.asarray(A_signed, dtype=float)
    if omega.size != A.size or omega.size < 7:
        return float('nan'), float('nan'), float('nan'), False, 'bad_grid', float('nan')
    # Negative area of the relative/signed response.  This is the diagnostic
    # used to stop assigning a FWHM on the broken/non-Lorentzian side.
    neg_weight = float(np.trapezoid(np.clip(-A, 0.0, None), omega))
    mask = (omega >= center - halfwidth) & (omega <= center + halfwidth)
    if np.sum(mask) < 7:
        return float('nan'), float('nan'), float('nan'), False, 'no_window', neg_weight
    Aw = A[mask]
    peak = float(np.nanmax(Aw)) if np.any(np.isfinite(Aw)) else float('nan')
    if not np.isfinite(peak) or peak <= 0.0:
        return float('nan'), peak, float('nan'), False, 'no_positive_peak', neg_weight
    tol = max(float(neg_abs_tol), float(neg_rel_tol) * peak)
    if neg_weight > tol * max(1.0, float(omega[-1] - omega[0])):
        return float('nan'), peak, float('nan'), False, 'negative_signed_weight', neg_weight
    if float(np.nanmin(Aw)) < -tol:
        return float('nan'), peak, float('nan'), False, 'local_negative_weight', neg_weight
    # Use the raw positive line shape; only tiny negative roundoff is clipped.
    A_local_clean = A.copy()
    A_local_clean[np.abs(A_local_clean) < tol] = np.maximum(A_local_clean[np.abs(A_local_clean) < tol], 0.0)
    wpk, Apk, hwhm = extract_hwhm_scale(omega, A_local_clean, center=center, halfwidth=halfwidth)
    if not np.isfinite(hwhm) or hwhm <= 0.0:
        return wpk, Apk, float('nan'), False, 'no_halfmax_crossing', neg_weight
    return wpk, Apk, hwhm, True, 'valid', neg_weight

def normalized_for_plot(y: np.ndarray, smooth: bool = True, nonnegative: bool = True) -> np.ndarray:
    arr = smooth_for_plot(y) if smooth else np.asarray(y, dtype=float)
    if nonnegative:
        arr = np.clip(arr, 0.0, None)
    mx = np.nanmax(np.abs(arr)) if arr.size else np.nan
    if not np.isfinite(mx) or mx < 1e-30:
        return np.zeros_like(arr)
    return arr / mx


def sweep(eps_xi: float, T: float, flip: bool, betas: np.ndarray) -> dict[str, np.ndarray]:
    betas = np.asarray(betas)
    pos = betas[betas >= 0]
    neg = betas[betas < 0][::-1]

    def branch(beta_list: np.ndarray, d0: complex, b0: complex) -> dict[str, np.ndarray]:
        nlev = 4
        if len(beta_list) == 0:
            return {
                "er": np.empty((0, nlev)), "ei": np.empty((0, nlev)),
                "kap": np.empty((0,)), "imagspan": np.empty((0,)),
                "d": np.empty((0,), dtype=complex), "b": np.empty((0,), dtype=complex),
            }
        ers, eis, kappas, imagspan, ds, bs = [], [], [], [], [], []
        d, b = d0, b0
        ev_prev = None
        for beta0 in beta_list:
            d, b = solve(beta0, eps_xi, T, flip, d, b, kgrid=K_LINEAR)
            te = eps_xi + np.real(d)
            tb = gauge_tbeta(beta0, b)
            H0 = build_Hss(te, tb, 0.0, flip=flip)
            kap, ev, _, _ = ep_metric(H0)
            if ev_prev is None:
                ev = ev[np.argsort(np.real(ev))]
            else:
                ev = match_branches(ev, ev_prev)
            ev_prev = ev.copy()
            ers.append(np.real(ev))
            eis.append(np.imag(ev))
            kappas.append(kap)
            imagspan.append(np.max(np.abs(np.imag(ev))))
            ds.append(d)
            bs.append(b)
        return {
            "er": np.array(ers), "ei": np.array(eis),
            "kap": np.array(kappas, dtype=float),
            "imagspan": np.array(imagspan, dtype=float),
            "d": np.array(ds, dtype=complex), "b": np.array(bs, dtype=complex),
        }

    rp = branch(pos, 0.1 + 0.j, 0.1 + 0.j)
    rn = branch(neg, 0.1 + 0.j, 0.1 + 0.j)

    def glue(key: str) -> np.ndarray:
        a = rn[key]
        b = rp[key]
        if a.size == 0:
            return b
        if b.size == 0:
            return a[::-1]
        if a.ndim == 1:
            return np.concatenate([a[::-1], b])
        return np.vstack([a[::-1], b])

    return {
        "betas": betas,
        "er": glue("er"),
        "ei": glue("ei"),
        "kap": glue("kap"),
        "imagspan": glue("imagspan"),
        "d": glue("d"),
        "b": glue("b"),
    }


def estimate_single_ep_from_kappa(betas: np.ndarray, kappa: np.ndarray) -> tuple[int, float, np.ndarray]:
    ks = smooth_kappa(kappa)
    mask = betas >= 0
    idx_local = int(np.nanargmax(ks[mask]))
    idx = np.where(mask)[0][idx_local]
    return idx, float(betas[idx]), ks


def estimate_two_eps_from_kappa(betas: np.ndarray, kappa: np.ndarray) -> tuple[tuple[int, float], tuple[int, float], np.ndarray]:
    ks = smooth_kappa(kappa)
    mask = betas > 0.0
    beta_pos = betas[mask]
    k_pos = ks[mask]
    peaks, _ = find_peaks(k_pos, prominence=max(np.nanmax(k_pos) * 0.02, 1e-12), distance=18)
    if len(peaks) < 2:
        w1 = np.where((beta_pos >= 0.20) & (beta_pos <= 0.50))[0]
        w2 = np.where((beta_pos >= 0.80) & (beta_pos <= 1.30))[0]
        if len(w1) == 0 or len(w2) == 0:
            idxs = np.argsort(k_pos)[-2:]
            idxs = np.sort(idxs)
        else:
            idxs = np.array([w1[np.argmax(k_pos[w1])], w2[np.argmax(k_pos[w2])]])
    else:
        strongest = peaks[np.argsort(k_pos[peaks])[-2:]]
        idxs = np.sort(strongest)
    global_idx = np.where(mask)[0][idxs]
    ep1 = (int(global_idx[0]), float(betas[global_idx[0]]))
    ep2 = (int(global_idx[1]), float(betas[global_idx[1]]))
    return ep1, ep2, ks


def kappa_powerlaw_fit(betas: np.ndarray, kappa: np.ndarray, idx_ep: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    kap = smooth_kappa(kappa)
    bep = betas[idx_ep]
    mask = ((np.abs(betas - bep) > 0.01) & (np.abs(betas - bep) < 0.35 * np.ptp(betas)) & np.isfinite(kap) & (kap > 1))
    if mask.sum() < 6:
        return None, None
    x = np.log(np.abs(betas[mask] - bep))
    y = np.log(np.abs(kap[mask]))
    m, c = np.polyfit(x, y, 1)
    alpha = -float(m)
    b_fit = np.linspace(bep - 0.35 * np.ptp(betas), bep + 0.35 * np.ptp(betas), 200)
    y_fit = np.exp(c) / np.maximum(np.abs(b_fit - bep), 1e-4) ** alpha
    return b_fit, y_fit


def effective_flat_rho() -> float:
    """Effective constant radial DOS used only in SW/TK diagnostics."""
    return float(P.rho_eff)


def gamma_width(V_eff: float, rho: float | None = None) -> float:
    """Hybridization width Gamma = pi rho V_eff^2 for a positive screening channel."""
    if rho is None:
        rho = effective_flat_rho()
    return float(np.pi * rho * V_eff**2)


def sw_exchange(eps_d: float, U: float, V_eff: float) -> float:
    """Schrieffer--Wolff exchange for a positive reciprocal screening channel.

    Returns NaN outside the local-moment window eps_d < 0 < eps_d+U.
    This function is a classifier/diagnostic; the frozen 4x4 spectrum itself
    remains bilinear.
    """
    eps_d = float(eps_d)
    U = float(U)
    if not (eps_d < 0.0 and eps_d + U > 0.0 and V_eff > 0.0):
        return float("nan")
    return 2.0 * V_eff**2 * (1.0 / abs(eps_d) + 1.0 / abs(eps_d + U))


def sw_kondo_scale(eps_d: float, U: float, V_eff: float,
                   rho: float | None = None, D: float | None = None,
                   chiK: float | None = None) -> float:
    """Poor-man/SW Kondo scale used to label regimes, not to fit spectra."""
    if rho is None:
        rho = effective_flat_rho()
    if D is None:
        D = float(P.D_uv)
    if chiK is None:
        chiK = float(P.chi_K)
    J = sw_exchange(eps_d, U, V_eff)
    if not np.isfinite(J):
        return float("nan")
    rhoJ = rho * J
    if rhoJ <= 0.0:
        return float("nan")
    return float(D * np.exp(-1.0 / (chiK * rhoJ)))


def ep_biorthogonal_Z(tbeta: float, flip: bool = True,
                       floor: float | None = None,
                       zmax: float | None = None) -> float:
    """Bounded EP-proximity factor for the frozen BA/SW diagnostic.

    Near the isolated impurity EP the local splitting
        s = sqrt(Delta_eff^2 - Gamma_PT^2)
    collapses.  A frozen biorthogonal BA estimate would enhance the effective
    width roughly as Z_EP ~ Lambda/|s|.  To avoid plotting a formal divergence
    as a physical observable, this routine returns a bounded diagnostic factor

        Z_EP = min(Zmax, Lambda / max(|s|, floor)).

    It should be cited as an analytic frozen-scale diagnostic, not as a spectral
    HWHM extracted from the noisy DOS.
    """
    if floor is None:
        floor = float(P.Z_EP_floor)
    if zmax is None:
        zmax = float(P.Z_EP_max)
    if not flip:
        return 1.0
    s2 = local_impurity_splitting_squared(float(tbeta), flip=flip)
    s_abs = float(np.sqrt(abs(s2)))
    Lambda = max(abs(float(P.delta0)), floor)
    return float(min(zmax, max(1.0, Lambda / max(s_abs, floor))))


def ep_assisted_sw_kondo_scale(eps_d: float, U: float, V_eff: float,
                               tbeta: float, rho: float | None = None,
                               D: float | None = None, chiK: float | None = None) -> float:
    """Smooth frozen EP-assisted screening scale.

    This multiplies the positive screening hybridization by the bounded
    biorthogonal factor Z_EP.  It is a theoretical diagnostic for the frozen
    effective model and is intentionally separated from the noisy spectral HWHM.
    """
    if rho is None:
        rho = effective_flat_rho()
    if D is None:
        D = float(P.D_uv)
    if chiK is None:
        chiK = float(P.chi_K)
    if not (eps_d < 0.0 and eps_d + U > 0.0 and V_eff > 0.0):
        return float('nan')
    Z = ep_biorthogonal_Z(tbeta, flip=True)
    Veff_ba = float(np.sqrt(Z)) * float(V_eff)
    return sw_kondo_scale(eps_d, U, Veff_ba, rho=rho, D=D, chiK=chiK)


def local_window_moments(omega: np.ndarray, A: np.ndarray, center: float, halfwidth: float) -> tuple[float, float, float, float]:
    """Robust local spectral diagnostics: area, centroid, rms width, peak height.

    The HWHM is unstable when peaks merge/split or when the local baseline is
    nonzero.  The moment width is smoother and is the preferred diagnostic for
    Fig. 4 scans.
    """
    mask = (omega >= center - halfwidth) & (omega <= center + halfwidth)
    if not np.any(mask):
        return float('nan'), float('nan'), float('nan'), float('nan')
    om = omega[mask]
    y = np.asarray(A[mask], dtype=float)
    if len(y) < 3 or not np.any(np.isfinite(y)):
        return float('nan'), float('nan'), float('nan'), float('nan')
    # subtract a local baseline and clip only for moment extraction
    y = y - np.nanmin(y)
    y = np.clip(y, 0.0, None)
    area = float(np.trapezoid(y, om))
    peak = float(np.nanmax(y))
    if not np.isfinite(area) or area <= 1e-15:
        return float('nan'), float('nan'), float('nan'), peak
    centroid = float(np.trapezoid(om * y, om) / area)
    var = float(np.trapezoid((om - centroid) ** 2 * y, om) / area)
    width = float(np.sqrt(max(var, 0.0)))
    return area, centroid, width, peak


def extract_hwhm_scale(omega: np.ndarray, A: np.ndarray, center: float = 0.0,
                       halfwidth: float = 0.45, use_local_baseline: bool = True) -> tuple[float, float, float]:
    """Return (peak_position, peak_height, HWHM) in a local low-energy window.

    This is an operational SBMF spectral scale extracted from the physical
    bath-FDR impurity spectrum.  It is independent of the SW/BA formulae and is
    therefore useful as a non-circular check, but it is still a broadened
    finite-grid spectral estimate.
    """
    omega = np.asarray(omega, dtype=float)
    A = np.asarray(A, dtype=float)
    mask = (omega >= center - halfwidth) & (omega <= center + halfwidth)
    if not np.any(mask):
        return float('nan'), float('nan'), float('nan')
    om = omega[mask]
    y = np.clip(A[mask], 0.0, None)
    if len(y) < 7 or not np.any(np.isfinite(y)):
        return float('nan'), float('nan'), float('nan')
    ip = int(np.nanargmax(y))
    wpk = float(om[ip])
    Apk = float(y[ip])
    if not np.isfinite(Apk) or Apk <= 0.0:
        return wpk, Apk, float('nan')
    baseline = float(np.nanmin(y)) if use_local_baseline else 0.0
    half = baseline + 0.5 * (Apk - baseline)
    left_w = np.nan
    for i in range(ip, 0, -1):
        y1, y2 = y[i], y[i - 1]
        if (y1 >= half and y2 <= half) or (y1 <= half and y2 >= half):
            x1, x2 = om[i], om[i - 1]
            left_w = float(x1 + (half - y1) * (x2 - x1) / (y2 - y1)) if y2 != y1 else float(x1)
            break
    right_w = np.nan
    for i in range(ip, len(y) - 1):
        y1, y2 = y[i], y[i + 1]
        if (y1 >= half and y2 <= half) or (y1 <= half and y2 >= half):
            x1, x2 = om[i], om[i + 1]
            right_w = float(x1 + (half - y1) * (x2 - x1) / (y2 - y1)) if y2 != y1 else float(x1)
            break
    if np.isfinite(left_w) and np.isfinite(right_w) and right_w > left_w:
        return wpk, Apk, float(0.5 * (right_w - left_w))
    return wpk, Apk, float('nan')


def minus_dfermi_domega(omega: np.ndarray, T: float) -> np.ndarray:
    """Thermal window -df/domega used for the FDR temperature-sweep scale."""
    omega = np.asarray(omega, dtype=float)
    T = max(float(T), 1e-12)
    x = np.clip(omega / (2.0 * T), -80.0, 80.0)
    return 1.0 / (4.0 * T * np.cosh(x) ** 2)


def temperature_sweep_halfscale(omega: np.ndarray, A: np.ndarray,
                                T_grid: np.ndarray | None = None,
                                center: float = 0.0, halfwidth: float = 0.55) -> tuple[float, np.ndarray, np.ndarray]:
    """Return a bath-FDR temperature-sweep half-suppression scale.

    The observable is the local linear-response spectral weight
        W(T) = integral A(omega) [-df/domega] d omega
    in a low-energy window.  The returned scale is the temperature where W(T)
    has fallen halfway from its low-T value to its high-T value.  This is an
    operational SBMF/FDR check; it is not used as input to the SW/BA scales.
    """
    if T_grid is None:
        T_grid = np.asarray(P.sbmf_T_grid, dtype=float)
    omega = np.asarray(omega, dtype=float)
    A = np.clip(np.asarray(A, dtype=float), 0.0, None)
    mask = (omega >= center - halfwidth) & (omega <= center + halfwidth)
    if not np.any(mask):
        return float('nan'), np.asarray(T_grid, dtype=float), np.full_like(np.asarray(T_grid, dtype=float), np.nan)
    om = omega[mask]
    y = A[mask]
    weights = []
    for T in np.asarray(T_grid, dtype=float):
        ker = minus_dfermi_domega(om - center, float(T))
        norm = float(np.trapezoid(ker, om))
        if not np.isfinite(norm) or norm <= 0.0:
            weights.append(np.nan)
        else:
            weights.append(float(np.trapezoid(y * ker, om) / norm))
    weights = np.asarray(weights, dtype=float)
    if len(weights) < 3 or not np.isfinite(weights[0]) or not np.isfinite(weights[-1]):
        return float('nan'), np.asarray(T_grid, dtype=float), weights
    low = float(weights[0])
    high = float(weights[-1])
    target = high + 0.5 * (low - high)
    # Find first monotonic crossing from low to high.  If the curve increases,
    # use the analogous upward crossing.
    for i in range(1, len(weights)):
        a, b = weights[i - 1], weights[i]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        crossed = (a >= target >= b) or (a <= target <= b)
        if crossed and b != a:
            T1, T2 = float(T_grid[i - 1]), float(T_grid[i])
            # log-T interpolation is smoother for a geometric grid.
            x1, x2 = np.log(T1), np.log(T2)
            frac = (target - a) / (b - a)
            return float(np.exp(x1 + frac * (x2 - x1))), np.asarray(T_grid, dtype=float), weights
    return float('nan'), np.asarray(T_grid, dtype=float), weights


def temperature_sweep_sbmf_scale(beta0: float, eps_xi: float, T_grid: np.ndarray,
                                 omega_ref: np.ndarray, center0: float,
                                 halfwidth: float = 0.55, flip: bool = True,
                                 eta: float | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """Operational SBMF temperature-sweep scale with re-solved saddle fields.

    For each probe temperature the diagnostic SBMF closure is re-solved and the
    physical bath-FDR impurity spectrum is recomputed with
        tbeta(T) = |beta0| |b_c(T)|.
    This avoids defining the temperature scale by convolving one fixed spectrum.
    The thermal window is centered on the instantaneous shifted resonance
    Re(teps(T)), because the non-Hermitian/Friedel resonance is not pinned to
    omega=0 in this model.
    """
    T_grid = np.asarray(T_grid, dtype=float)
    if eta is None:
        eta = float(P.eta_sbmf_hwhm)
    weights = []
    d_prev, b_prev = 0.1 + 0j, 0.1 + 0j
    nloc = 81
    for Tprobe in T_grid:
        d_prev, b_prev = solve(float(beta0), eps_xi, float(Tprobe), flip=flip,
                               d0=d_prev, b0=b_prev, kgrid=K_LINEAR[::5])
        tbetaT = gauge_tbeta(float(beta0), b_prev)
        centerT = float(np.real(eps_xi + np.real(d_prev)))
        if not np.isfinite(centerT):
            centerT = float(center0)
        om = np.linspace(centerT - halfwidth, centerT + halfwidth, nloc)
        A = physical_impurity_dos(eps_xi + np.real(d_prev), tbetaT, om,
                                  max(float(eta), 1e-3), flip=flip,
                                  kgrid=K_LINEAR[::5])
        A = np.clip(np.asarray(A, dtype=float), 0.0, None)
        ker = minus_dfermi_domega(om - centerT, float(Tprobe))
        norm = float(np.trapezoid(ker, om))
        weights.append(float(np.trapezoid(A * ker, om) / norm) if np.isfinite(norm) and norm > 0 else np.nan)
    weights = np.asarray(weights, dtype=float)
    if len(weights) < 3 or not np.isfinite(weights[0]) or not np.isfinite(weights[-1]):
        return float('nan'), T_grid, weights
    low = float(weights[0])
    high = float(weights[-1])
    target = high + 0.5 * (low - high)
    for i in range(1, len(weights)):
        w0, w1 = weights[i - 1], weights[i]
        if not (np.isfinite(w0) and np.isfinite(w1)):
            continue
        if (w0 >= target >= w1) or (w0 <= target <= w1):
            T0, T1 = float(T_grid[i - 1]), float(T_grid[i])
            if w1 == w0:
                return T1, T_grid, weights
            frac = (target - w0) / (w1 - w0)
            return float(np.exp(np.log(T0) + frac * (np.log(T1) - np.log(T0)))), T_grid, weights
    idx = int(np.nanargmin(np.abs(weights - target)))
    return float(T_grid[idx]), T_grid, weights


def classify_regime(reg: RegimeSpec) -> dict[str, float | str | bool]:
    """Classify a plotted regime using explicit Anderson/SW inequalities."""
    tb = abs(reg.beta_ref) * float(P.r_fixed)
    _, _, Veff, _ = channel_scales(tb, flip=True)
    rho = effective_flat_rho()
    Gamma = gamma_width(Veff, rho)
    TK = sw_kondo_scale(reg.eps_xi, reg.U, Veff, rho=rho)
    charge_gap = min(abs(reg.eps_xi), abs(reg.eps_xi + reg.U)) if reg.U > 0 else 0.0
    local_moment = bool(reg.eps_xi < 0.0 and reg.eps_xi + reg.U > 0.0 and Gamma < charge_gap)
    kondo = bool(local_moment and np.isfinite(TK) and reg.T < TK)
    free_orbital = bool(local_moment and np.isfinite(TK) and reg.T > TK)
    mixed_valence = bool((reg.U <= 0.0) or (abs(reg.eps_xi) <= Gamma) or (reg.eps_xi + reg.U <= Gamma))
    if kondo:
        cls = "Kondo/local-moment (T<TK_SW)"
    elif mixed_valence:
        cls = "resonant/mixed-valence"
    elif free_orbital:
        cls = "free-orbital/high-T (T>TK_SW)"
    else:
        cls = "crossover/diagnostic"
    return {
        "label": reg.label,
        "classification": cls,
        "eps_d": float(reg.eps_xi),
        "U": float(reg.U),
        "T": float(reg.T),
        "beta_ref": float(reg.beta_ref),
        "Veff_ref": float(Veff),
        "Gamma_ref": float(Gamma),
        "TK_SW_ref": float(TK) if np.isfinite(TK) else float("nan"),
        "local_moment": local_moment,
        "mixed_valence": mixed_valence,
        "free_orbital": free_orbital,
        "kondo": kondo,
    }


def estTK(beta0: float, d: complex, b: complex, eps_xi: float, U: float | None = None) -> float:
    """SW Kondo-scale proxy used only to set plotting windows and diagnostics."""
    if U is None:
        U = float(P.U_default)
    tb = gauge_tbeta(beta0, b)
    _, _, V_eff, _ = channel_scales(tb, flip=True)
    return sw_kondo_scale(float(np.real(eps_xi + d)), U, V_eff)


def omega_K_from_SC(teps: complex, tbeta: complex, flip: bool = True) -> float:
    imp_ev = np.linalg.eigvals(build_Hss(teps, tbeta, 0.0, flip=flip)[:2, :2])
    return float(np.real(imp_ev[np.argmin(np.abs(np.imag(imp_ev)))]))


def local_impurity_splitting_squared(tbeta: float, flip: bool = True) -> complex:
    """Squared splitting of the isolated 2x2 impurity block.

    For B_imp=0 it is Delta_eff^2 - Gamma_PT^2.  A zero locates the
    local impurity EP before bath broadening/unfolding.
    """
    gamma_pt, delta_eff, _, B_imp = channel_scales(tbeta, flip=flip)
    return complex(delta_eff**2 + B_imp**2 - gamma_pt**2 + 2j * B_imp * gamma_pt)


def pt_regime_label(beta0: float, b: complex | float = 1.0, flip: bool = True,
                    tol: float = 1e-2) -> str:
    """Classify the local PT block at the given beta value."""
    s2 = local_impurity_splitting_squared(gauge_tbeta(beta0, b), flip=flip)
    if not flip:
        return "gain/loss-only diagnostic; no coherent local impurity EP"
    if abs(s2) < tol:
        return "at local EP"
    if np.real(s2) > 0:
        return "PT-unbroken/local oscillatory"
    return "PT-broken/local overdamped"


def choose_triplet_around_ep(betas: np.ndarray, d_arr: np.ndarray, b_arr: np.ndarray,
                             beta_ep: float, left_scale: float = 0.55,
                             right_scale: float = 1.50) -> list[tuple[float, complex, complex]]:
    targets = [beta_ep * left_scale, beta_ep, np.sign(beta_ep) * min(abs(beta_ep) * right_scale, 1.10)]
    out = []
    for t in targets:
        idx = int(np.argmin(np.abs(betas - t)))
        out.append((float(betas[idx]), d_arr[idx], b_arr[idx]))
    return out


def estimate_time_window(beta0: float, eps_xi: float, T: float, flip: bool,
                         d: complex | None = None, b: complex | None = None,
                         n_periods: float = 2.5) -> tuple[float, float]:
    if d is None or b is None:
        d, b = solve(beta0, eps_xi, T, flip)
    te = eps_xi + np.real(d)
    tb = gauge_tbeta(beta0, b)
    ev = np.linalg.eigvals(build_Hss(te, tb, 0.0, flip=flip))
    ev_s = sorted(ev, key=lambda x: abs(x.imag))
    re_split = abs(ev_s[0].real - ev_s[1].real)
    im_min = min(abs(e.imag) for e in ev_s)
    if re_split > 0.05:
        Tmax = n_periods * 2 * np.pi / re_split
    elif im_min > 1e-4:
        Tmax = n_periods * 3.0 / max(im_min, 1e-4)
    else:
        Tmax = 20.0
    Tmax = float(np.clip(Tmax, 5.0, 35.0))
    return Tmax, Tmax / 350.0


def evolve(beta0: float, eps_xi: float, T: float, flip: bool,
           d: complex | None = None, b: complex | None = None,
           Tmax: float | None = None, dt: float | None = None,
           O0: np.ndarray | None = None,
           mode: str = "amplitude") -> tuple[np.ndarray, np.ndarray]:
    """Time-domain diagnostic using an exact matrix exponential.

    mode="amplitude" returns a complex survival amplitude <psi0|U(t)|psi0>,
    which gives clean phase-space orbits: Hermitian/unitary reference is closed,
    PT-unbroken is bounded/elliptic, and PT-broken typically spirals/grows/decays.

    If O0 is supplied and mode="operator", returns Tr[U^dagger O0 U O0].
    That object is a nonunitary correlator, not a normalized density-matrix
    expectation value.
    """
    if d is None or b is None:
        d, b = solve(beta0, eps_xi, T, flip)
    te = eps_xi + np.real(d)
    tb = gauge_tbeta(beta0, b)
    H = build_Hss(te, tb, 0.0, flip=flip)
    if Tmax is None or dt is None:
        Tmax, dt = estimate_time_window(beta0, eps_xi, T, flip, d=d, b=b)
    tvals = np.arange(0.0, Tmax, dt)
    out = np.zeros(len(tvals), dtype=complex)
    if mode == "amplitude" or O0 is None:
        psi0 = np.zeros(4, dtype=complex)
        psi0[0] = 1.0
        bra0 = psi0.conj()
        for i, t in enumerate(tvals):
            Uop = expm(-1j * H * t)
            out[i] = bra0 @ (Uop @ psi0)
        return tvals, out
    if mode != "operator":
        raise ValueError("mode must be 'amplitude' or 'operator'")
    for i, t in enumerate(tvals):
        Uop = expm(-1j * H * t)
        out[i] = np.trace((Uop.conj().T @ O0 @ Uop) @ O0)
    return tvals, out


# -----------------------------------------------------------------------------
# Panel writers (same file structure as previous script)
# -----------------------------------------------------------------------------
def write_fig12_panels(outdir: Path, eps_xi: float, T: float, flip: bool, stem: str) -> None:
    """Write spectra/kappa/dynamics panels for the local PT block.

    For flip=True this shows the physical PT-unbroken -> EP -> broken sequence.
    For flip=False we do not mark artificial EPs: a diagonal gain/loss-only
    impurity block has no local eigenvector coalescence.  This avoids the old
    two-EP-pair overclaim.
    """
    betas = P.beta_vals
    sw = sweep(eps_xi, T, flip, betas)
    kap_sm = smooth_kappa(sw["kap"])
    if flip:
        tb_ep = predicted_local_ep_tbeta()
        beta_ep = tb_ep / max(float(P.r_fixed), 1e-12) if tb_ep is not None else float(betas[np.nanargmax(kap_sm)])
        idx_ep = int(np.argmin(np.abs(sw["betas"] - beta_ep)))
        idx_ep_neg = int(np.argmin(np.abs(sw["betas"] + beta_ep)))
        ep_markers = [(idx_ep_neg, -beta_ep, "EP-"), (idx_ep, beta_ep, "EP+")]
        idx_ref = idx_ep
    else:
        beta_ep = predicted_local_ep_tbeta() or 0.5
        idx_ref = int(np.argmin(np.abs(sw["betas"] - beta_ep)))
        ep_markers = []  # no coherent local EP in the flip-off/gain-loss-only diagnostic

    re_sets = [(f"ReE{j+1}", sw["betas"], sw["er"][:, j]) for j in range(sw["er"].shape[1])]
    ymin, ymax = float(np.min(sw["er"])), float(np.max(sw["er"]))
    for _, be, lab in ep_markers:
        re_sets.append(vline_dataset(be, ymin, ymax, lab))
    write_nxy(outdir / f"{stem}a_eigs_re.dat", re_sets)

    im_sets = [(f"ImE{j+1}", sw["betas"], sw["ei"][:, j]) for j in range(sw["ei"].shape[1])]
    ymin, ymax = float(np.min(sw["ei"])), float(np.max(sw["ei"]))
    im_sets.append(hline_dataset(sw["betas"][0], sw["betas"][-1], 0.0, "zero"))
    for _, be, lab in ep_markers:
        im_sets.append(vline_dataset(be, ymin, ymax, lab))
    write_nxy(outdir / f"{stem}a_eigs_im.dat", im_sets)

    ksets = [("Kbiorth_raw", sw["betas"], np.abs(sw["kap"])), ("Kbiorth_smooth", sw["betas"], kap_sm)]
    if flip:
        span_sm = median_filter(np.abs(sw["imagspan"]), size=11, mode="nearest")
        Cval = 1.0 / max(2.0 * abs(beta_ep), 1e-12)
        kap_analytic = np.full_like(span_sm, np.nan, dtype=float)
        valid = span_sm > 1e-6
        kap_analytic[valid] = Cval / np.maximum(span_sm[valid], 1e-8)
        ksets.append(("analytic_EP_distance", sw["betas"], kap_analytic))
    ymin, ymax = max(np.nanmin(kap_sm), 1e-6), float(np.nanmax(kap_sm))
    for _, be, lab in ep_markers:
        ksets.append(vline_dataset(be, ymin, ymax, lab))
    write_nxy(outdir / f"{stem}b_kappa.dat", ksets)

    if flip:
        trip = choose_triplet_around_ep(sw["betas"], sw["d"], sw["b"], beta_ep, left_scale=0.0, right_scale=1.50)
        labels = [
            "PT_unbroken_below_EP",
            "at_EP_critical",
            "PT_broken_above_EP",
        ]
        Tmax, dt = estimate_time_window(beta_ep, eps_xi, T, flip, d=sw["d"][idx_ref], b=sw["b"][idx_ref])
    else:
        targets = [0.0, 0.5 * beta_ep, beta_ep]
        trip = []
        for t in targets:
            idx = int(np.argmin(np.abs(sw["betas"] - t)))
            trip.append((float(sw["betas"][idx]), sw["d"][idx], sw["b"][idx]))
        labels = [
            "Hermitian_origin_reference",
            "gain_loss_only_weak",
            "gain_loss_only_strong_no_local_EP",
        ]
        Tmax, dt = 25.0, 25.0 / 350.0

    dyn = [evolve(beta0, eps_xi, T, flip, d=dv, b=bv, Tmax=Tmax, dt=dt, mode="amplitude")
           for beta0, dv, bv in trip]
    all_abs = np.concatenate([np.abs(O) for _, O in dyn])
    ymax = max(np.max(all_abs), 1e-12)
    dsets = [(lab, t, np.real(O) / ymax) for lab, (t, O) in zip(labels, dyn)]
    write_nxy(outdir / f"{stem}c_dyn.dat", dsets)
    psets = []
    for lab, (_, O) in zip(labels, dyn):
        re = np.real(O); im = np.imag(O)
        norm = max(np.sqrt(re**2 + im**2).max(), 1e-30)
        psets.append((lab, re / norm, im / norm))
    write_nxy(outdir / f"{stem}d_phase.dat", psets)

    # Classification file for the phase-space panel.
    with open(outdir / f"{stem}d_phase_labels.txt", "w", encoding="utf-8") as f:
        for lab, (beta0, dval, bval) in zip(labels, trip):
            f.write(f"{lab}: beta0={beta0:.8g}, {pt_regime_label(beta0, bval, flip=flip)}\n")


def write_fig3_panels(outdir: Path) -> None:
    """Write regime comparison with explicit Anderson/SW labels.

    Fig3g_scales.dat now contains SW diagnostic scales (TK_SW, Gamma, and T),
    not mislabeled spin-resolved eigenvalues.
    """
    kappa_sets, dos_sets, dos_raw_sets, occ_sets, scale_sets = [], [], [], [], []
    with open(outdir / "Fig3_regime_classification.txt", "w", encoding="utf-8") as f:
        for reg in REGIMES:
            info = classify_regime(reg)
            f.write(f"[{reg.label}] {info['classification']}\n")
            for key in ["eps_d", "U", "T", "beta_ref", "Veff_ref", "Gamma_ref", "TK_SW_ref"]:
                f.write(f"  {key} = {info[key]}\n")
            f.write(f"  description = {reg.description}\n\n")

    for reg in REGIMES:
        eps_xi, T, label = reg.eps_xi, reg.T, reg.label
        sw = sweep(eps_xi, T, True, P.beta_sweep3)
        kappa_sets.append((label, P.beta_sweep3, smooth_kappa(sw["kap"])))

        #idx = int(np.argmin(np.abs(P.beta_sweep3 - P.beta0_dos3)))
        beta_for_dos = getattr(reg, "beta_dos", reg.beta_ref)
        idx = int(np.argmin(np.abs(P.beta_sweep3 - beta_for_dos)))
        te = eps_xi + np.real(sw["d"][idx])
        tb = gauge_tbeta(P.beta_sweep3[idx], sw["b"][idx])
        # Raw physical DOS is kept for auditing, while the main manuscript DOS
        # comparison is broadened, smoothed and normalized to emphasize line shape
        # rather than finite-k-grid spike heights.  Temperature alone does not
        # change the frozen spectral function; the occupied-spectrum file below
        # shows the thermal/FDR weighting separately.
        A_raw = physical_impurity_dos(te, tb, P.omega_dos3, P.eta_broad, flip=True, kgrid=K_FINE)
        A_plot = physical_impurity_dos(te, tb, P.omega_dos3, P.eta_dos_plot, flip=True, kgrid=K_FINE)
        dos_raw_sets.append((label, P.omega_dos3, A_raw))
        dos_sets.append((label, P.omega_dos3, normalized_for_plot(A_plot, smooth=True)))
        occ_sets.append((label, P.omega_dos3, normalized_for_plot(A_plot * f_fermi(P.omega_dos3, T), smooth=True)))

        tk_sw, gamma_vals, temp_vals = [], [], []
        for beta0, d, b in zip(P.beta_sweep3, sw["d"], sw["b"]):
            te0 = float(np.real(eps_xi + d))
            tb0 = gauge_tbeta(beta0, b)
            _, _, Veff0, _ = channel_scales(tb0, flip=True)
            gamma_vals.append(gamma_width(Veff0))
            tk_sw.append(sw_kondo_scale(te0, reg.U, Veff0))
            temp_vals.append(reg.T)
        scale_sets.append((label + "_TK_SW", P.beta_sweep3, np.array(tk_sw, dtype=float)))
        scale_sets.append((label + "_Gamma", P.beta_sweep3, np.array(gamma_vals, dtype=float)))
        scale_sets.append((label + "_T", P.beta_sweep3, np.array(temp_vals, dtype=float)))

    write_nxy(outdir / "Fig3abc_kappa.dat", kappa_sets)
    write_nxy(outdir / "Fig3def_dos.dat", dos_sets)
    write_nxy(outdir / "Fig3def_dos_raw_physical.dat", dos_raw_sets)
    write_nxy(outdir / "Fig3def_occupied_spectrum_diagnostic.dat", occ_sets)
    write_nxy(outdir / "Fig3g_scales.dat", scale_sets)


def write_fig4_panels(outdir: Path) -> None:
    """
    Fig. 4:
      (a) full causal DOS at fixed beta0_4
      (b) zoom near the NH-Friedel reference energy omegaK = Re(teps)
      (c) scan toward EP from below, separately tracking:
            - central peak near omega = 0
            - NH-Friedel peak near omega = Re(teps)
            - robust moment/area/peak-separation diagnostics
            - smooth SW and EP-assisted frozen BA scale diagnostics

    HWHM values are retained only in the raw-debug file.  They are not used as
    the main scale evidence because they become noisy when peaks merge or cross.
    """

    def local_peak_in_window(omega: np.ndarray, A: np.ndarray, center: float, halfwidth: float):
        mask = (omega >= center - halfwidth) & (omega <= center + halfwidth)
        if not np.any(mask):
            return np.nan, np.nan
        om = omega[mask]
        Aw = A[mask]
        if len(Aw) == 0 or np.all(~np.isfinite(Aw)):
            return np.nan, np.nan
        i = int(np.nanargmax(Aw))
        return float(om[i]), float(Aw[i])

    def extract_hwhm_in_window(omega: np.ndarray, A: np.ndarray,
                               center: float, halfwidth: float,
                               use_local_baseline: bool = False):
        """
        Generic HWHM extractor in a local window around `center`.

        Returns:
            wpk, Apk, hwhm
        """
        mask = (omega >= center - halfwidth) & (omega <= center + halfwidth)
        if not np.any(mask):
            return np.nan, np.nan, np.nan

        om0 = omega[mask]
        A0 = A[mask]
        if len(A0) < 5 or np.all(~np.isfinite(A0)):
            return np.nan, np.nan, np.nan

        ip = int(np.nanargmax(A0))
        wpk = float(om0[ip])
        Apk = float(A0[ip])
        if not np.isfinite(Apk) or Apk <= 0:
            return wpk, Apk, np.nan

        baseline = float(np.nanmin(A0)) if use_local_baseline else 0.0
        half = baseline + 0.5 * (Apk - baseline)

        # left crossing
        left_w = np.nan
        for i in range(ip, 0, -1):
            y1, y2 = A0[i], A0[i - 1]
            if (y1 >= half and y2 <= half) or (y1 <= half and y2 >= half):
                x1, x2 = om0[i], om0[i - 1]
                if y2 != y1:
                    left_w = float(x1 + (half - y1) * (x2 - x1) / (y2 - y1))
                else:
                    left_w = float(x1)
                break

        # right crossing
        right_w = np.nan
        for i in range(ip, len(A0) - 1):
            y1, y2 = A0[i], A0[i + 1]
            if (y1 >= half and y2 <= half) or (y1 <= half and y2 >= half):
                x1, x2 = om0[i], om0[i + 1]
                if y2 != y1:
                    right_w = float(x1 + (half - y1) * (x2 - x1) / (y2 - y1))
                else:
                    right_w = float(x1)
                break

        if np.isfinite(left_w) and np.isfinite(right_w) and right_w > left_w:
            hwhm = 0.5 * (right_w - left_w)
            return wpk, Apk, float(hwhm)

        return wpk, Apk, np.nan

    def safe_norm(y: np.ndarray) -> np.ndarray:
        ymax = np.nanmax(np.abs(y))
        if not np.isfinite(ymax) or ymax < 1e-30:
            return np.zeros_like(y)
        return y / ymax

    # ------------------------------------------------------------------
    # Fixed-beta DOS panels
    # ------------------------------------------------------------------
    d, b = solve(P.beta0_4, P.eps_xi_4, P.T_4, True, kgrid=K_FINE)
    te = P.eps_xi_4 + np.real(d)
    tb = gauge_tbeta(P.beta0_4, b)

    omegaK = float(np.real(te))  # NH-Friedel / exact Kondo reference

    omega_full = np.linspace(-4.0, 4.0, 600)
    A_full = physical_impurity_dos(te, tb, omega_full, P.eta_dos_plot, flip=True, kgrid=K_FINE)

    full_sets = [
        ("physical_DOS_full_broadened", omega_full, A_full),
        vline_dataset(omegaK, np.min(A_full), np.max(A_full), "omegaK=Re(teps)"),
        vline_dataset(0.0, np.min(A_full), np.max(A_full), "omega=0"),
    ]
    write_nxy(outdir / "Fig4a_friedel_full.dat", full_sets)

    wid = max(25 * max(estTK(P.beta0_4, d, b, P.eps_xi_4), 1e-6), 0.6)
    omega_zoom = np.linspace(omegaK - wid, omegaK + wid, 600)
    A_zoom = physical_impurity_dos(te, tb, omega_zoom, P.eta_scan_plot, flip=True, kgrid=K_FINE)

    zoom_sets = [
        ("physical_DOS_zoom_broadened", omega_zoom, A_zoom),
        vline_dataset(omegaK, np.min(A_zoom), np.max(A_zoom), "omegaK=Re(teps)"),
    ]
    write_nxy(outdir / "Fig4b_friedel_zoom.dat", zoom_sets)

    # ------------------------------------------------------------------
    # Scan toward EP from below
    # ------------------------------------------------------------------
    betas = np.linspace(0.05, 0.60, 60)
    zero_halfwidth = 0.25
    friedel_halfwidth = 0.30

    cds_raw = []
    w0_list, A0_list = [], []
    wF_list, AF_list = [], []
    wT_list = []
    TK0_list = []   # raw HWHM around omega ~ 0; debug only
    TKF_list = []   # raw HWHM around omegaK = Re(teps); debug only
    area0_list, cen0_list, sig0_list, pk0_list = [], [], [], []
    areaF_list, cenF_list, sigF_list, pkF_list = [], [], [], []
    TKplain_list, TKep_list, Zep_list, sep_list, sabs_list = [], [], [], [], []

    d4, b4 = 0 + 0.1j, 0 + 0.1j
    for b0 in betas:
        d4, b4 = solve(b0, P.eps_xi_4, P.T_4, True, d4, b4, kgrid=K_FINE)
        te4 = P.eps_xi_4 + np.real(d4)
        tb4 = gauge_tbeta(b0, b4)

        kap, _, _, _ = ep_metric(build_Hss(te4, tb4, 0.0, flip=True))
        cds_raw.append(kap)

        # broad enough grid to resolve both low-energy central feature and shifted Friedel feature
        om_grid = np.linspace(-1.6, 0.2, 700)
        A_scan = physical_impurity_dos(te4, tb4, om_grid, P.eta_scan_plot, flip=True, kgrid=K_FINE)

        # central feature near omega = 0
        w0, A0 = local_peak_in_window(om_grid, A_scan, center=0.0, halfwidth=zero_halfwidth)
        _, _, TK0 = extract_hwhm_in_window(
            om_grid, A_scan,
            center=0.0,
            halfwidth=zero_halfwidth,
            use_local_baseline=False
        )

        # NH-Friedel / Kondo feature near Re(teps)
        wT = float(np.real(te4))
        wF, AF = local_peak_in_window(om_grid, A_scan, center=wT, halfwidth=friedel_halfwidth)
        _, _, TKF = extract_hwhm_in_window(
            om_grid, A_scan,
            center=wT,
            halfwidth=friedel_halfwidth,
            use_local_baseline=False
        )

        area0, cen0, sig0, pk0 = local_window_moments(om_grid, A_scan, center=0.0, halfwidth=zero_halfwidth)
        areaF, cenF, sigF, pkF = local_window_moments(om_grid, A_scan, center=wT, halfwidth=friedel_halfwidth)

        gamma_pt, delta_eff, Veff4, _ = channel_scales(tb4, flip=True)
        s_abs = float(np.sqrt(abs(delta_eff**2 - gamma_pt**2)))
        TK_plain = sw_kondo_scale(float(np.real(te4)), float(P.U_default), Veff4)
        TK_ep = ep_assisted_sw_kondo_scale(float(np.real(te4)), float(P.U_default), Veff4, tb4)
        Z_ep = ep_biorthogonal_Z(tb4, flip=True)
        sep = abs(cen0 - cenF) if np.isfinite(cen0) and np.isfinite(cenF) else np.nan

        area0_list.append(area0); cen0_list.append(cen0); sig0_list.append(sig0); pk0_list.append(pk0)
        areaF_list.append(areaF); cenF_list.append(cenF); sigF_list.append(sigF); pkF_list.append(pkF)
        TKplain_list.append(TK_plain); TKep_list.append(TK_ep); Zep_list.append(Z_ep); sep_list.append(sep); sabs_list.append(s_abs)

        w0_list.append(w0)
        A0_list.append(A0)
        wF_list.append(wF)
        AF_list.append(AF)
        wT_list.append(wT)
        TK0_list.append(TK0)
        TKF_list.append(TKF)

    cds_raw = np.asarray(cds_raw, dtype=float)
    cds_sm = smooth_kappa(np.abs(cds_raw))

    A0_arr = np.asarray(A0_list, dtype=float)
    AF_arr = np.asarray(AF_list, dtype=float)
    w0_arr = np.asarray(w0_list, dtype=float)
    wF_arr = np.asarray(wF_list, dtype=float)
    wT_arr = np.asarray(wT_list, dtype=float)
    TK0_arr = np.asarray(TK0_list, dtype=float)
    TKF_arr = np.asarray(TKF_list, dtype=float)
    area0_arr = np.asarray(area0_list, dtype=float)
    cen0_arr = np.asarray(cen0_list, dtype=float)
    sig0_arr = np.asarray(sig0_list, dtype=float)
    pk0_arr = np.asarray(pk0_list, dtype=float)
    areaF_arr = np.asarray(areaF_list, dtype=float)
    cenF_arr = np.asarray(cenF_list, dtype=float)
    sigF_arr = np.asarray(sigF_list, dtype=float)
    pkF_arr = np.asarray(pkF_list, dtype=float)
    TKplain_arr = np.asarray(TKplain_list, dtype=float)
    TKep_arr = np.asarray(TKep_list, dtype=float)
    Zep_arr = np.asarray(Zep_list, dtype=float)
    sep_arr = np.asarray(sep_list, dtype=float)
    sabs_arr = np.asarray(sabs_list, dtype=float)

    idx_ep = int(np.nanargmax(cds_sm))
    beta_ep = float(betas[idx_ep])

    # ------------------------------------------------------------------
    # Heights scan
    # ------------------------------------------------------------------
    # The raw local peak heights are sensitive to peak splitting/crossing and
    # finite-k-grid spikes.  For the manuscript-facing height scan we therefore
    # use smoothed window areas and a smoothed peak envelope.  The raw heights are
    # written separately below as a debug file.
    combined_area = area0_arr + areaF_arr
    peak_envelope = np.maximum(pk0_arr, pkF_arr)
    scan_height_sets = [
        ("area_zero_smooth_norm", betas, normalized_for_plot(area0_arr, smooth=True)),
        ("area_friedel_smooth_norm", betas, normalized_for_plot(areaF_arr, smooth=True)),
        ("combined_area_smooth_norm", betas, normalized_for_plot(combined_area, smooth=True)),
        ("peak_envelope_smooth_norm", betas, normalized_for_plot(peak_envelope, smooth=True)),
        ("Kbiorth_smooth_norm", betas, normalized_for_plot(cds_sm, smooth=False)),
        vline_dataset(beta_ep, 0.0, 1.05, "EP"),
    ]
    write_nxy(outdir / "Fig4c_peakscan_heights.dat", scan_height_sets)

    raw_height_sets = [
        ("raw_peak_zero_debug", betas, safe_norm(pk0_arr)),
        ("raw_peak_friedel_debug", betas, safe_norm(pkF_arr)),
        ("raw_area_zero_debug", betas, safe_norm(area0_arr)),
        ("raw_area_friedel_debug", betas, safe_norm(areaF_arr)),
        vline_dataset(beta_ep, 0.0, 1.05, "EP"),
    ]
    write_nxy(outdir / "Fig4c_peakscan_raw_heights_debug.dat", raw_height_sets)

    # ------------------------------------------------------------------
    # Peak positions scan
    # ------------------------------------------------------------------
    finite_pos = np.concatenate([
        wT_arr[np.isfinite(wT_arr)],
        cen0_arr[np.isfinite(cen0_arr)],
        cenF_arr[np.isfinite(cenF_arr)],
    ])
    ymin = float(np.nanmin(finite_pos)) if len(finite_pos) else -1.5
    ymax = float(np.nanmax(finite_pos)) if len(finite_pos) else 0.1

    scan_pos_sets = [
        ("omega_Friedel_theory = Re(teps)", betas, wT_arr),
        ("centroid_zero_window", betas, cen0_arr),
        ("centroid_friedel_window", betas, cenF_arr),
        ("centroid_separation", betas, sep_arr),
        vline_dataset(beta_ep, ymin, ymax, "EP"),
    ]
    write_nxy(outdir / "Fig4c_peakscan_positions.dat", scan_pos_sets)

    # ------------------------------------------------------------------
    # Stable scale diagnostics.  The noisy HWHM values are not used as the
    # main evidence.  We write smooth SW and bounded EP-assisted frozen BA
    # scales, plus robust moment widths in a separate file.
    # ------------------------------------------------------------------
    finite_scale = np.concatenate([
        TKplain_arr[np.isfinite(TKplain_arr) & (TKplain_arr > 0)],
        TKep_arr[np.isfinite(TKep_arr) & (TKep_arr > 0)],
    ])
    tk_ymin = float(np.nanmin(finite_scale)) if len(finite_scale) else 1e-8
    tk_ymax = float(np.nanmax(finite_scale)) if len(finite_scale) else 1.0

    TK_sets = [
        ("TK_SW_plain", betas, TKplain_arr),
        ("TK_EP_assisted_bounded", betas, TKep_arr),
        ("Z_EP_bounded", betas, Zep_arr * tk_ymax / max(np.nanmax(Zep_arr), 1e-12)),
        ("Kbiorth / max", betas, safe_norm(cds_sm) * tk_ymax),
        vline_dataset(beta_ep, tk_ymin, tk_ymax, "EP"),
    ]
    write_nxy(outdir / "Fig4c_TKscan.dat", TK_sets)

    # Same smooth scale information on a log scale.  This is the recommended
    # plotting file for the scale diagnostic: it avoids the misleading visual
    # dominance of the large-beta tail in linear scale.
    tiny = 1e-300
    log_plain = np.clip(np.log10(np.maximum(TKplain_arr, tiny)), -12.0, 0.0)
    log_ep = np.clip(np.log10(np.maximum(TKep_arr, tiny)), -12.0, 0.0)
    ratio = TKep_arr / np.maximum(TKplain_arr, tiny)
    log_ratio = np.clip(np.log10(np.maximum(ratio, tiny)), -2.0, 3.0)
    TK_log_sets = [
        ("log10_TK_SW_plain_clipped", betas, log_plain),
        ("log10_TK_EP_assisted_clipped", betas, log_ep),
        ("log10_ratio_EP_over_plain_clipped", betas, log_ratio),
        vline_dataset(beta_ep, min(np.nanmin(log_plain), np.nanmin(log_ep)), max(np.nanmax(log_plain), np.nanmax(log_ep)), "EP"),
    ]
    write_nxy(outdir / "Fig4c_TKscan_log.dat", TK_log_sets)

    # Dimensionless EP-distance diagnostics, kept separate from the scale plot.
    inv_s = 1.0 / np.maximum(sabs_arr, 1e-12)
    epdiag_sets = [
        ("Z_EP_bounded_norm", betas, safe_norm(Zep_arr)),
        ("inverse_s_abs_norm", betas, safe_norm(inv_s)),
        ("Kbiorth_norm", betas, safe_norm(cds_sm)),
        ("s_abs", betas, sabs_arr),
        vline_dataset(beta_ep, 0.0, 1.05, "EP"),
    ]
    write_nxy(outdir / "Fig4c_EP_diagnostics.dat", epdiag_sets)

    moment_sets = [
        ("sigma_zero_window", betas, sig0_arr),
        ("sigma_friedel_window", betas, sigF_arr),
        ("area_zero_window_norm", betas, safe_norm(area0_arr)),
        ("area_friedel_window_norm", betas, safe_norm(areaF_arr)),
        ("centroid_separation", betas, sep_arr),
        ("local_s_abs", betas, sabs_arr),
        vline_dataset(beta_ep, 0.0, max(np.nanmax(sig0_arr), np.nanmax(sigF_arr), np.nanmax(sep_arr), np.nanmax(sabs_arr)), "EP"),
    ]
    write_nxy(outdir / "Fig4c_moment_widths_and_merge.dat", moment_sets)

    raw_hwhm_sets = [
        ("raw_HWHM_zero_debug", betas, TK0_arr),
        ("raw_HWHM_friedel_debug", betas, TKF_arr),
        vline_dataset(beta_ep, np.nanmin([np.nanmin(TK0_arr), np.nanmin(TKF_arr)]), np.nanmax([np.nanmax(TK0_arr), np.nanmax(TKF_arr)]), "EP"),
    ]
    write_nxy(outdir / "Fig4c_raw_HWHM_debug_not_TK.dat", raw_hwhm_sets)

    # ------------------------------------------------------------------
    # Raw columns for debugging / manuscript checks
    # ------------------------------------------------------------------
    with open(outdir / "Fig4c_peakscan_raw_columns.dat", "w", encoding="utf-8") as f:
        f.write("# beta0 omega_friedel_theory centroid_zero area_zero sigma_zero peak_zero centroid_friedel area_friedel sigma_friedel peak_friedel sep_centroids TK_SW_plain TK_EP_bounded Z_EP s_abs raw_HWHM_zero raw_HWHM_friedel kappa_raw kappa_smooth\n")
        for vals in zip(betas, wT_arr, cen0_arr, area0_arr, sig0_arr, pk0_arr, cenF_arr, areaF_arr, sigF_arr, pkF_arr, sep_arr, TKplain_arr, TKep_arr, Zep_arr, sabs_arr, TK0_arr, TKF_arr, cds_raw, cds_sm):
            f.write(" ".join(f"{float(v):.12e}" for v in vals) + "\n")

def write_fig5_panels(outdir: Path) -> None:
    """Write FDR diagnostics.

    Fig. 5 is deliberately diagnostic rather than a proof of thermalization.
    We compare two lesser constructions, block by block:
      (i)  bath-constructed equilibrium FDR: G^< = f_FD(omega)(G^R-G^A),
           which reproduces f_FD by construction for both impurity and bath blocks;
      (ii) eigenmode-built lesser object, using occupations assigned to
           non-Hermitian eigenmodes.  This second object is basis/normalization
           sensitive and is shown only to diagnose how an eigenmode prescription
           can violate the physical bath FDR.
    """
    betas5 = np.linspace(0.05, 0.80, 40)
    imevs5 = []
    d5, b5 = 0 + 0.1j, 0 + 0.1j
    for b0 in betas5:
        d5, b5 = solve(b0, P.eps_xi_5, P.T_5, True, d5, b5, kgrid=K_LINEAR)
        tb5 = gauge_tbeta(b0, b5)
        ev0 = np.linalg.eigvals(build_Hss(P.eps_xi_5 + np.real(d5), tb5, 0.0, flip=True))
        imevs5.append(np.max(np.abs(np.imag(ev0))))
    imevs5 = np.array(imevs5)
    idx_ep = np.searchsorted(imevs5 > 0.02, True)
    idx_ep = min(idx_ep, len(betas5) - 1)
    beta_EP = betas5[idx_ep]
    beta_below = beta_EP * 0.45

    d, b = solve(beta_below, P.eps_xi_5, P.T_5, True, kgrid=K_LINEAR)
    te = P.eps_xi_5 + np.real(d)
    tb = gauge_tbeta(beta_below, b)
    nF = len(P.omega_fdr)
    fFD = f_fermi(P.omega_fdr, P.T_5)

    Gr = np.zeros((nF, 4, 4), dtype=complex)
    Geig = np.zeros((nF, 4, 4), dtype=complex)
    nk = len(K_LINEAR)
    for k in K_LINEAR:
        Hk = build_Hss(te, tb, k, flip=True)
        ev, R, L = eig_lr(Hk)
        # Use Re(ev) in the diagnostic eigenmode occupation.  This is not a
        # physical FDR prescription in the broken/non-normal regime; it is shown
        # only as a contrast to the bath-constructed FDR.
        fn = f_fermi(np.real(ev), P.T_5)
        projectors = [np.outer(R[:, n], L[n, :]) for n in range(4)]
        for iw, w in enumerate(P.omega_fdr):
            denom = w + 1j * P.eta_fdr - ev
            Gk = sum(projectors[n] / denom[n] for n in range(4))
            Geig_k = 1j * sum(fn[n] * projectors[n] / denom[n] for n in range(4))
            Gr[iw] += Gk / nk
            Geig[iw] += Geig_k / nk

    Ga = np.conj(Gr.transpose(0, 2, 1))
    GrGa = Gr - Ga
    Gbath = fFD[:, None, None] * GrGa

    imp = np.array([0, 1])
    bath = np.array([2, 3])

    def block_trace(M: np.ndarray, idx: np.ndarray) -> complex:
        return np.trace(M[np.ix_(idx, idx)])

    def ratio_block(Gless: np.ndarray, Gdiff: np.ndarray, idx: np.ndarray) -> np.ndarray:
        out = np.full(nF, np.nan, dtype=float)
        for i in range(nF):
            den = block_trace(Gdiff[i], idx)
            if abs(den) > 1e-14:
                out[i] = float(np.real(block_trace(Gless[i], idx) / den))
        return out

    FDR_imp_bath = ratio_block(Gbath, GrGa, imp)
    FDR_bath_bath = ratio_block(Gbath, GrGa, bath)
    FDR_imp_eig_raw = ratio_block(Geig, GrGa, imp)
    FDR_bath_eig_raw = ratio_block(Geig, GrGa, bath)

    clip = 1.5
    # Physical impurity and bath spectral responses from the retarded/advanced
    # difference, useful for marking where the FDR ratio is meaningful.
    A_imp = np.zeros(nF, dtype=float)
    A_bath = np.zeros(nF, dtype=float)
    for i in range(nF):
        A_imp[i] = -np.imag(block_trace(GrGa[i], imp)) / (2.0 * np.pi)
        A_bath[i] = -np.imag(block_trace(GrGa[i], bath)) / (2.0 * np.pi)

    def support_mask(A: np.ndarray, frac: float = 1e-3) -> np.ndarray:
        mx = np.nanmax(np.abs(A))
        if not np.isfinite(mx) or mx <= 0.0:
            return np.zeros_like(A, dtype=bool)
        return np.abs(A) >= frac * mx

    def mask_where_low_weight(y: np.ndarray, A: np.ndarray, frac: float = 1e-3) -> np.ndarray:
        yy = np.array(y, dtype=float, copy=True)
        yy[~support_mask(A, frac=frac)] = np.nan
        return yy

    FDR_imp_eig_masked = mask_where_low_weight(FDR_imp_eig_raw, A_imp)
    FDR_bath_eig_masked = mask_where_low_weight(FDR_bath_eig_raw, A_bath)
    dev_imp = mask_where_low_weight(np.abs(FDR_imp_eig_raw - fFD), A_imp)
    dev_bath = mask_where_low_weight(np.abs(FDR_bath_eig_raw - fFD), A_bath)

    fdr_sets = [
        ("fFD", P.omega_fdr, fFD),
        ("impurity_bath_constructed", P.omega_fdr, FDR_imp_bath),
        ("bath_bath_constructed", P.omega_fdr, FDR_bath_bath),
        ("impurity_eigenmode_diag_clipped_support", P.omega_fdr, np.clip(FDR_imp_eig_masked, -clip, clip)),
        ("bath_eigenmode_diag_clipped_support", P.omega_fdr, np.clip(FDR_bath_eig_masked, -clip, clip)),
    ]
    write_nxy(outdir / "Fig5a_FDR.dat", fdr_sets)

    fdr_raw_sets = [
        ("impurity_eigenmode_diag_raw_support", P.omega_fdr, FDR_imp_eig_masked),
        ("bath_eigenmode_diag_raw_support", P.omega_fdr, FDR_bath_eig_masked),
        ("impurity_bath_minus_fFD", P.omega_fdr, FDR_imp_bath - fFD),
        ("bath_bath_minus_fFD", P.omega_fdr, FDR_bath_bath - fFD),
    ]
    write_nxy(outdir / "Fig5a_FDR_raw_diagnostics.dat", fdr_raw_sets)

    fdr_dev_sets = [
        ("abs_impurity_eigenmode_minus_fFD_support", P.omega_fdr, dev_imp),
        ("abs_bath_eigenmode_minus_fFD_support", P.omega_fdr, dev_bath),
    ]
    write_nxy(outdir / "Fig5a_FDR_deviation.dat", fdr_dev_sets)

    iK = int(np.nanargmax(A_imp))
    omegaK = float(P.omega_fdr[iK])
    resp_sets = [
        ("A_impurity", P.omega_fdr, A_imp),
        ("A_bath", P.omega_fdr, A_bath),
        vline_dataset(omegaK, np.nanmin(A_imp), np.nanmax(A_imp), "omegaK_impurity_peak"),
    ]
    write_nxy(outdir / "Fig5b_response.dat", resp_sets)

    with open(outdir / "Fig5_FDR_notes.txt", "w", encoding="utf-8") as f:
        f.write("Fig. 5 FDR diagnostic notes\n")
        f.write(f"beta_below_EP={beta_below:.8g}, beta_EP_indicator={beta_EP:.8g}, teps={float(np.real(te)):.8g}, tbeta={tb:.8g}\n")
        f.write("bath_constructed block FDRs should reproduce fFD up to numerical precision.\n")
        f.write("eigenmode_diag curves are diagnostic only and may leave [0,1] in non-Hermitian/non-normal regimes.\n")
        f.write(f"max_abs_impurity_bath_minus_fFD={np.nanmax(np.abs(FDR_imp_bath - fFD)):.6e}\n")
        f.write(f"max_abs_bath_bath_minus_fFD={np.nanmax(np.abs(FDR_bath_bath - fFD)):.6e}\n")
        f.write(f"max_abs_impurity_eig_minus_fFD_on_support={np.nanmax(dev_imp):.6e}\n")
        f.write(f"max_abs_bath_eig_minus_fFD_on_support={np.nanmax(dev_bath):.6e}\n")
        f.write("Support mask threshold: spectral weight >= 1e-3 of the corresponding block maximum.\n")



def predicted_local_ep_tbeta() -> float | None:
    """Positive local EP estimate for B_imp=0 and linear channel coefficients."""
    denom = float(P.c_gamma) - float(P.c_delta)
    if abs(float(P.B_imp)) > 1e-14 or abs(denom) < 1e-14 or not np.isfinite(denom):
        return None
    tb = float(P.delta0) / denom
    return tb if tb > 0 else None

def write_diagnostics(outdir: Path) -> None:
    with open(outdir / "run_diagnostics.txt", "w", encoding="utf-8") as f:
        f.write("XMGrace panel files written.\n")
        f.write("Same file names/panel structure as previous script.\n")
        f.write("DOS panels use physical_impurity_dos() instead of direct NH response.\n")
        f.write("Gauge convention: tbeta = |beta0|*|b_c|; arg(b_c) never enters observables; spectra/DOS/kappa are even in beta0.\n")
        f.write(f"Channel coefficients: c_gamma={P.c_gamma}, c_delta={P.c_delta}, delta0={P.delta0}, c_V={P.c_V}, B_imp={P.B_imp}\n")
        f.write(f"Closure: scf_closure={P.scf_closure}, r_fixed={P.r_fixed}.\n")
        f.write(f"Constraint convention for eigenmode_diagnostic: Q={P.Q_constraint}, b2_clip_max={P.b2_clip_max}, sign r^2=Q+i Tr G<ff.\n")
        f.write(f"Stable scale diagnostics: Z_EP_floor={P.Z_EP_floor}, Z_EP_max={P.Z_EP_max}. Fig3 also reports independent SBMF spectral HWHM and FDR temperature-sweep half-suppression checks.\n")
        tb_ep = predicted_local_ep_tbeta()
        if tb_ep is not None:
            f.write(f"Predicted isolated local EP before bath unfolding: tbeta_EP≈{tb_ep:.8g}.\n")
            f.write(f"With r_fixed={P.r_fixed}, symmetric beta0_EP≈±{tb_ep/max(P.r_fixed,1e-12):.8g}.\n")
        else:
            f.write("No simple positive local EP estimate from linear B_imp=0 formula.\n")
        f.write("\nRegime classifications at beta_ref:\n")
        for reg in REGIMES:
            info = classify_regime(reg)
            f.write(f"- {reg.label}: {info['classification']} | eps={info['eps_d']}, U={info['U']}, T={info['T']}, "
                    f"Gamma={info['Gamma_ref']}, TK_SW={info['TK_SW_ref']}\n")


# =====================================================================
#  UNIFIED MATPLOTLIB FIGURE DRIVER
# ---------------------------------------------------------------------
#  Faithful port of the all_in_one_2.py plotting layer (identical colors,
#  line widths, fonts, gridspec layout, EP markers, fade/arrow phase
#  portraits, waterfall spectra, and uniform stroke hierarchy), but calling the
#  in-file solver functions directly.  There is NO external solver module
#  and NO intermediate XMGrace step, so figures, exported .dat curves, and
#  captions all come from one consistent parameter set.
#
#  Corrected physics kept from the reviewed pipeline:
#    * Fig. 4 uses eps_xi = -U/2 = -1.0, U = 2.0, T = 0.1 (unified).
#    * Bath-constructed FDR residual is computed directly from the block
#      ratios (|F - f_FD|); it sits at the numerical floor as it must.
#    * The eigenmode ratio is a non-thermal diagnostic, clipped to +/-1.5.
#    * The signed physical DOS is shown; negative regions are exposed.
# =====================================================================

import argparse as _argparse
import matplotlib as _mpl
_mpl.use("Agg")
import matplotlib.pyplot as _plt
from matplotlib import ticker as _mticker
from matplotlib.collections import LineCollection as _LineCollection
from matplotlib.lines import Line2D as _Line2D

COL_W = 3.375
FULL_W = 6.875

C = {
    "blue":   "#1f4e9c",
    "red":    "#c2342c",
    "green":  "#2c8a3d",
    "orange": "#e08214",
    "grey":   "#555555",
    "purple": "#6a3d9a",
    "black":  "#1a1a1a",
    "cyan":   "#1b9aaa",
}

# Uniform PRB/PRL-style stroke hierarchy.
# Keep almost all scientific curves at LW_MAIN; reserve smaller values only
# for guide lines, EP markers, error/residual references, and marker edges.
LW_MAIN = 2.05
LW_SECONDARY = 1.70
LW_AUX = 1.15
LW_GUIDE = 0.75
LW_SPINE = 1.30


def set_style() -> None:
    _mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 8.6,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.4,
        "legend.fontsize": 6.4,
        "axes.linewidth": 1.35,
        "lines.linewidth": 2.05,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.6,
        "ytick.major.size": 4.6,
        "xtick.minor.size": 2.4,
        "ytick.minor.size": 2.4,
        "xtick.major.width": 1.25,
        "ytick.major.width": 1.25,
        "xtick.minor.width": 0.9,
        "ytick.minor.width": 0.9,
        "xtick.major.pad": 2.4,
        "ytick.major.pad": 2.4,
        "axes.labelpad": 2.6,
        "legend.frameon": False,
        "legend.handlelength": 1.45,
        "legend.handletextpad": 0.45,
        "legend.labelspacing": 0.23,
        "figure.dpi": 220,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def thicken(ax) -> None:
    for sp in ax.spines.values():
        sp.set_linewidth(LW_SPINE)
    ax.minorticks_on()


def tag(ax, txt: str, x: float = 0.045, y: float = 0.93) -> None:
    # White backing prevents panel labels from colliding visually with data/notes.
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=9.6,
            fontweight="bold", va="top", ha="left", zorder=30,
            bbox=dict(fc="white", ec="none", alpha=0.82, pad=0.35))


def annotate_ep(ax, beta_ep, betas, label: bool = False) -> None:
    if beta_ep is None or not np.isfinite(beta_ep):
        return
    lo, hi = float(np.nanmin(betas)), float(np.nanmax(betas))
    for x in sorted({float(beta_ep), float(-beta_ep)}):
        if lo <= x <= hi:
            ax.axvline(x, color=C["black"], ls=(0, (4, 2)), lw=LW_AUX, alpha=0.80, zorder=2)
            if label:
                ax.text(x, 0.98, r"EP", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=6.0,
                        bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.4))


def fade_plot(ax, x, y, color: str, lw: float = 2.0) -> None:
    if len(x) < 2:
        return
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    alpha = np.linspace(0.12, 1.0, len(segs))
    rgba = np.zeros((len(segs), 4))
    rgb = _mpl.colors.to_rgb(color)
    rgba[:, 0], rgba[:, 1], rgba[:, 2] = rgb
    rgba[:, 3] = alpha
    ax.add_collection(_LineCollection(segs, colors=rgba, linewidths=lw, zorder=3))
    ax.autoscale_view()


def add_arrow(ax, x, y, color: str, n: int = 3, size: float = 10.0) -> None:
    L = len(x)
    if L < 4:
        return
    for i in np.linspace(int(0.12 * L), int(0.88 * L), n).astype(int):
        if 0 < i < L - 1:
            ax.annotate("", xy=(x[i + 1], y[i + 1]), xytext=(x[i - 1], y[i - 1]),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.1,
                                        mutation_scale=size, alpha=0.97),
                        zorder=6)


def save_curve(out: Path, name: str, x, y, header: str = "") -> None:
    arr = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    np.savetxt(out / f"{name}.dat", arr, header=header)


def save_fig(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    _plt.close(fig)
    print(f"  wrote {name}.pdf / {name}.png")


def interp_complex_common(t, y, grid):
    t = np.asarray(t, float)
    y = np.asarray(y, complex)
    order = np.argsort(t)
    t = t[order]; y = y[order]
    return np.interp(grid, t, np.real(y)) + 1j * np.interp(grid, t, np.imag(y))


def _beta_ep_from_sweep(betas, kap, flip: bool, b_arr=None):
    """EP marker on the beta0 axis (in-file solver)."""
    if not flip:
        return None
    val = predicted_local_ep_tbeta()
    if val is not None and b_arr is not None:
        try:
            target = abs(float(val))
            tb = np.array([abs(gauge_tbeta(float(b0), b_arr[i]))
                           for i, b0 in enumerate(betas)])
            pos = betas >= 0
            if np.any(pos):
                return float(betas[pos][np.nanargmin(np.abs(tb[pos] - target))])
        except Exception:
            pass
    try:
        pos = betas >= 0
        k = np.asarray(kap, float)
        if np.any(pos) and np.any(np.isfinite(k[pos])):
            return float(betas[pos][np.nanargmax(k[pos])])
    except Exception:
        pass
    return None


def _choose_triplet(betas, sw, beta_ep, flip: bool):
    if flip:
        # Main Fig. 1 trajectory choices.  These are deliberately *not*
        # hard-coded now: they are controlled by P.fig1_left_scale and
        # P.fig1_right_scale, which can be set from the command line.
        # Values are multipliers of beta_EP, so e.g. left_scale=0.10
        # means beta0=0.10*beta_EP.
        if beta_ep is None:
            beta_ep = 0.5
        bmax = float(np.nanmax(betas))
        beta_small = float(P.fig1_left_scale) * float(beta_ep)
        beta_above = float(P.fig1_right_scale) * float(beta_ep)
        beta_small = float(np.clip(beta_small, float(np.nanmin(betas)), bmax))
        beta_above = float(np.clip(beta_above, float(np.nanmin(betas)), bmax))
        # Snap to the actual sweep grid so the displayed labels and saved curves
        # exactly match the saddle data used for d,b.
        beta_small = float(betas[int(np.nanargmin(np.abs(betas - beta_small)))])
        beta_ep_grid = float(betas[int(np.nanargmin(np.abs(betas - float(beta_ep))))])
        beta_above = float(betas[int(np.nanargmin(np.abs(betas - beta_above)))])
        return [beta_small, beta_ep_grid, beta_above]
    return [0.0, 0.25, 0.50]


def weighted_peak(omega, A, mask, rel_threshold: float = 0.05, width: float = 0.35) -> float:
    w = omega[mask]
    a = np.clip(np.real(A[mask]), 0.0, None)
    if a.size == 0 or np.nanmax(a) <= rel_threshold * max(
            np.nanmax(np.clip(np.real(A), 0, None)), 1e-30):
        return np.nan
    imax = int(np.nanargmax(a))
    center = float(w[imax])
    near = np.abs(w - center) <= width
    if np.sum(a[near]) <= 0:
        return center
    return float(np.sum(w[near] * a[near]) / np.sum(a[near]))


# ---------------------------------------------------------------------
#  Fig. 1 / Fig. 2
# ---------------------------------------------------------------------
def make_fig12(out: Path, flip: bool, fname: str, labels,
               time_tmax=None, time_points: int = 900):
    eps_xi = P.eps_xi_12
    T = P.T_12
    betas = np.asarray(P.beta_vals, float)
    sw = sweep(eps_xi, T, flip, betas)

    er = np.asarray(sw["er"], float)
    ei = np.asarray(sw["ei"], float)
    kap = np.asarray(sw["kap"], float)
    b_arr = np.asarray(sw.get("b"), dtype=object)
    betas = np.asarray(sw["betas"], float)
    beta_ep = _beta_ep_from_sweep(betas, kap, flip=flip, b_arr=b_arr)

    save_curve(out, f"{fname}_kappa", betas, kap, "beta0  kappa_imp")
    for j in range(er.shape[1] if er.ndim > 1 else 1):
        save_curve(out, f"{fname}_ReE{j+1}", betas, er[:, j] if er.ndim > 1 else er,
                   f"beta0 ReE{j+1}")
        save_curve(out, f"{fname}_ImE{j+1}", betas, ei[:, j] if ei.ndim > 1 else ei,
                   f"beta0 ImE{j+1}")

    fig = _plt.figure(figsize=(COL_W * 2.04, COL_W * 1.48))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.34)

    # (a) eigenvalues, Re and Im stacked.
    gsa = gs[0, 0].subgridspec(2, 1, hspace=0.10)
    axRe = fig.add_subplot(gsa[0])
    axIm = fig.add_subplot(gsa[1], sharex=axRe)
    colors = [C["blue"], C["red"], C["green"], C["orange"]]
    nbranch = er.shape[1] if er.ndim > 1 else 1
    for j in range(nbranch):
        col = colors[j % len(colors)]
        axRe.plot(betas, er[:, j] if er.ndim > 1 else er, color=col, lw=LW_SECONDARY)
        axIm.plot(betas, ei[:, j] if ei.ndim > 1 else ei, color=col, lw=LW_SECONDARY)
    annotate_ep(axRe, beta_ep, betas, label=False)
    annotate_ep(axIm, beta_ep, betas, label=True)
    axIm.axhline(0, color=C["grey"], lw=LW_GUIDE, alpha=0.60)
    axRe.set_ylabel(r"$\mathrm{Re}\,E$")
    axIm.set_ylabel(r"$\mathrm{Im}\,E$")
    axIm.set_xlabel(r"$\beta_0$")
    _plt.setp(axRe.get_xticklabels(), visible=False)
    thicken(axRe); thicken(axIm); tag(axRe, "(a)")

    # (b) biorthogonal condition number.
    axb = fig.add_subplot(gs[0, 1])
    kap_plot = np.clip(kap, 1.0, None)
    # In the present parameter set kappa is modest (order 1--25), so a
    # linear axis with ordinary numeric tick labels is cleaner than 10^n
    # scientific/log notation.  Keep a log fallback only for future extreme
    # sweeps.
    if np.nanmax(kap_plot) < 50.0:
        axb.plot(betas, kap_plot, color=C["grey"], lw=LW_MAIN)
        axb.set_ylim(0.95, 1.10 * float(np.nanmax(kap_plot)))
        axb.yaxis.set_major_locator(_mticker.MaxNLocator(nbins=4))
        axb.yaxis.set_major_formatter(_mticker.StrMethodFormatter("{x:g}"))
    else:
        axb.semilogy(betas, kap_plot, color=C["grey"], lw=LW_MAIN)
        axb.yaxis.set_major_locator(_mticker.LogLocator(base=10, numticks=4))
        axb.yaxis.set_minor_formatter(_mticker.NullFormatter())
        axb.yaxis.set_major_formatter(_mticker.FuncFormatter(
            lambda y, _pos: f"{int(y):d}" if 1 <= y < 10000 and abs(np.log10(y)-round(np.log10(y))) < 1e-8
            else (rf"$10^{{{int(round(np.log10(y)))}}}$" if y > 0 and abs(np.log10(y)-round(np.log10(y))) < 1e-8 else "")
        ))
    annotate_ep(axb, beta_ep, betas, label=True)
    if not flip:
        # Keep the explanatory note out of the plotting area; otherwise it
        # collides with the panel label and the top of the curve.
        axb.set_title("gain--loss control: no coherent local EP", pad=3)
    axb.set_xlabel(r"$\beta_0$")
    axb.set_ylabel(r"$\kappa_{\mathrm{imp}}$")
    thicken(axb); tag(axb, "(b)")

    # (c)/(d) dynamics and phase portrait on a common time grid.
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])
    trip_b = _choose_triplet(betas, sw, beta_ep, flip=flip)
    cols = [C["blue"], C["orange"], C["green"]]

    raw_traj = []
    for b0 in trip_b:
        idx = int(np.nanargmin(np.abs(betas - b0)))
        d = sw["d"][idx]
        b = sw["b"][idx]
        t, O = evolve(float(b0), eps_xi, T, flip, d=d, b=b, mode="amplitude")
        t = np.asarray(t, float)
        O = np.asarray(O, complex)
        if t.size >= 2 and np.all(np.isfinite(t)) and np.all(np.isfinite(O)):
            raw_traj.append((float(b0), t, O))

    if not raw_traj:
        raise RuntimeError(f"No valid trajectories for {fname}.")

    t_start = max(float(np.nanmin(t)) for _, t, _ in raw_traj)
    t_stop_solver = min(float(np.nanmax(t)) for _, t, _ in raw_traj)
    t_stop = t_stop_solver if time_tmax is None else min(float(time_tmax), t_stop_solver)
    if not np.isfinite(t_stop) or t_stop <= t_start:
        t_start = 0.0
        t_stop = t_stop_solver
    grid = np.linspace(t_start, t_stop, max(int(time_points), 64))

    common_traj = []
    for b0, t, O in raw_traj:
        O_common = interp_complex_common(t, O, grid)
        common_traj.append((b0, O_common, np.real(O_common), np.imag(O_common)))

    if P.phase_common_norm:
        phase_norm = max(
            [float(np.nanmax(np.sqrt(reO**2 + imO**2))) for _, _, reO, imO in common_traj] + [1.0]
        )
    else:
        phase_norm = None

    # Write the exact beta choices.  This avoids confusion when CLI scale values
    # are changed but the curve is snapped to the finite sweep grid.
    with open(out / f"{fname}_phase_beta_choices.txt", "w", encoding="utf-8") as fbeta:
        if flip:
            fbeta.write(f"beta_EP = {float(beta_ep):.12g}\n")
            fbeta.write(f"left_scale = {float(P.fig1_left_scale):.12g}\n")
            fbeta.write(f"right_scale = {float(P.fig1_right_scale):.12g}\n")
        else:
            # Fig. 2 is the gain/loss-only control.  There is intentionally no
            # coherent local impurity EP, so beta_ep=None is the correct state.
            # Record only the fixed reference beta values used for the dynamics.
            fbeta.write("beta_EP = none  # gain/loss-only control; no coherent local EP\n")
            fbeta.write("reference_betas = 0.0, 0.25, 0.50\n")
        fbeta.write(f"phase_common_norm = {bool(P.phase_common_norm)}\n")
        for i, (b0, O_common, reO, imO) in enumerate(common_traj, start=1):
            fbeta.write(f"curve_{i}: beta0 = {float(b0):.12g}, max_abs_O = {float(np.nanmax(np.abs(O_common))):.12g}\n")

    for i, ((b0, O_common, reO, imO), col, lab) in enumerate(zip(common_traj, cols, labels)):
        scale_re = np.nanmax(np.abs(reO)) or 1.0
        axc.plot(grid, reO / scale_re, color=col, lw=LW_MAIN, label=lab)

        save_curve(out, f"{fname}_dyn_{i+1}_beta_{b0:.4g}", grid, reO / scale_re,
                   "t_common  normalized_Re_survival_amplitude")

        # Main-paper portrait: ordinary survival-amplitude phase portrait.
        # By default each orbit is normalized separately for readability.
        # With --phase-common-norm the same normalization is used for all
        # three curves, making absolute growth/decay changes visible.
        nrm = phase_norm if phase_norm is not None else (np.nanmax(np.sqrt(reO**2 + imO**2)) or 1.0)
        X, Y = reO / nrm, imO / nrm
        if flip:
            fade_plot(axd, X, Y, col, lw=LW_MAIN)
            add_arrow(axd, X, Y, col)
        else:
            axd.plot(X, Y, color=col, lw=LW_MAIN, alpha=0.92, label=lab)
        axd.plot([X[0]], [Y[0]], "o", color=col, ms=4.5,
                 mec=C["black"], mew=0.6, zorder=7)
        save_curve(out, f"{fname}_survival_phase_{i+1}_beta_{b0:.4g}", X, Y,
                   "Re_survival_amplitude_normalized  Im_survival_amplitude_normalized")

    axc.set_xlim(grid[0], grid[-1])
    axc.set_xlabel(r"$t$")
    axc.set_ylabel(r"$\mathrm{Re}\,O(t)/\max$")
    axc.legend(loc="lower left", fontsize=5.7, frameon=True, framealpha=0.88, facecolor="white", edgecolor="none")
    thicken(axc); tag(axc, "(c)")

    if flip:
        proxies = [_Line2D([0], [0], color=cols[i], lw=LW_MAIN, label=labels[i]) for i in range(3)]
        axd.legend(handles=proxies, loc="upper right", fontsize=5.5, frameon=True,
                   framealpha=0.88, facecolor="white", edgecolor="none")
        axd.text(0.04, 0.05,
                 r"small $\beta_0$ $\rightarrow$ EP $\rightarrow$ broken",
                 transform=axd.transAxes, ha="left", va="bottom", fontsize=5.4,
                 bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=C["grey"],
                           lw=0.6, alpha=0.82))
    else:
        axd.legend(loc="upper right", fontsize=5.5, frameon=True, framealpha=0.88,
                   facecolor="white", edgecolor="none")
        axd.text(0.04, 0.05,
                 r"survival-amplitude portrait" + "\n" +
                 r"no coherent local EP marker",
                 transform=axd.transAxes, ha="left", va="bottom", fontsize=5.4,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C["grey"],
                           lw=0.6, alpha=0.85))
    axd.set_xlabel(r"$\mathrm{Re}\,O(t)/\max|O|$")
    axd.set_ylabel(r"$\mathrm{Im}\,O(t)/\max|O|$")
    axd.set_aspect("equal", adjustable="datalim")
    thicken(axd); tag(axd, "(d)")

    save_fig(fig, out, fname)
    return betas, sw


# ---------------------------------------------------------------------
#  Fig. 3
# ---------------------------------------------------------------------
def make_fig3(out: Path, eta: float, dos_betas, dos_xlim=None,
              waterfall_offset: float = 1.20) -> None:
    eps_xi = P.eps_xi_4
    U = P.U_default
    T = P.T_4
    betas = np.linspace(0.0, 0.60, 81)
    omega = np.asarray(P.omega_dos3, float)

    sw = sweep(eps_xi, T, True, betas)
    betas = np.asarray(sw["betas"], float)
    beta_ep = _beta_ep_from_sweep(betas, np.asarray(sw["kap"], float), True,
                                  b_arr=np.asarray(sw.get("b"), dtype=object))

    TK_sw, TK_ep, TK_hwhm, TK_temp, Zep, sabs = [], [], [], [], [], []
    # Independent SBMF spectral diagnostics.  The linewidth (HWHM) is kept as
    # an audit quantity, but the main non-circular SBMF check is a positive
    # spectral-weight-transfer ratio: W_FSR^+ / W_CP^+.  This remains meaningful
    # when the resonance is asymmetric, and it does not require assigning a
    # FWHM in a multi-peak/non-Lorentzian spectrum.
    W_fsr_pos, W_cp_pos, W_transfer_ratio, Gamma_area_fsr = [], [], [], []
    pk_up, pk_lo, te_refs, neg_min, neg_weight = [], [], [], [], []
    hwhm_valid, hwhm_reason = [], []
    temp_sweep_rows = []
    for b0 in betas:
        d, b = solve(float(b0), eps_xi, T, flip=True)
        tb = gauge_tbeta(float(b0), b)
        te = eps_xi + np.real(d)
        gpt, deff, veff, _ = channel_scales(tb, flip=True)
        veff_safe = max(abs(float(np.real(veff))), 1e-12)
        TK_sw.append(sw_kondo_scale(float(np.real(te)), U, veff_safe))
        TK_ep.append(ep_assisted_sw_kondo_scale(float(np.real(te)), U, veff_safe, tb))
        Zep.append(ep_biorthogonal_Z(tb, flip=True))
        s2 = local_impurity_splitting_squared(tb, flip=True)
        sabs.append(float(np.sqrt(abs(s2))))
        A = np.asarray(physical_impurity_dos(te, tb, omega, max(float(eta), float(P.eta_sbmf_hwhm)), flip=True), float)
        neg_min.append(float(np.nanmin(A)))
        Apos = np.clip(A, 0.0, None)
        # Independent SBMF spectral check.  The gauge-fixed effective amplitude
        # tbeta=|beta0| |b_c| enters inside the Hamiltonian before the spectrum is
        # computed.  The FSR HWHM is extracted around Re(teps) only while the
        # signed relative spectrum remains positive and a local half-maximum
        # width is well defined; once negative signed weight/non-Lorentzian
        # behavior appears, no FWHM is assigned.
        center_res = float(np.real(te))
        _, _, hwhm_res, is_valid, why_invalid, neg_area = extract_positive_fsr_hwhm(
            omega, A, center=center_res, halfwidth=float(P.sbmf_hwhm_window))
        neg_weight.append(neg_area)
        hwhm_valid.append(bool(is_valid))
        hwhm_reason.append(str(why_invalid))

        # Positive spectral-weight transfer diagnostic.  We track how much
        # positive physical spectral weight is redistributed from the central
        # peak window (CP, around omega=0) into the shifted FSR/Kondo window
        # (around Re teps).  This is deliberately peak-resolved and avoids
        # averaging FWHM values from physically distinct peaks.
        fsr_half = float(P.sbmf_hwhm_window)
        cp_half = 0.25
        fsr_mask = (omega >= center_res - fsr_half) & (omega <= center_res + fsr_half)
        cp_mask = (omega >= -cp_half) & (omega <= cp_half)
        W_fsr = float(np.trapezoid(np.clip(A[fsr_mask], 0.0, None), omega[fsr_mask])) if np.sum(fsr_mask) > 2 else np.nan
        W_cp = float(np.trapezoid(np.clip(A[cp_mask], 0.0, None), omega[cp_mask])) if np.sum(cp_mask) > 2 else np.nan
        P_fsr = float(np.nanmax(np.clip(A[fsr_mask], 0.0, None))) if np.sum(fsr_mask) > 2 else np.nan
        W_fsr_pos.append(W_fsr)
        W_cp_pos.append(W_cp)
        W_transfer_ratio.append(W_fsr / max(W_cp, 1e-14) if np.isfinite(W_fsr) and np.isfinite(W_cp) and W_fsr > 0.0 and W_cp > 0.0 else np.nan)
        Gamma_area_fsr.append(W_fsr / (np.pi * P_fsr) if np.isfinite(W_fsr) and np.isfinite(P_fsr) and W_fsr > 0.0 and P_fsr > 0.0 else np.nan)
        t_half, tgrid, wT = temperature_sweep_sbmf_scale(float(b0), eps_xi,
                                                        np.asarray(P.sbmf_T_grid, dtype=float),
                                                        omega, center0=center_res,
                                                        halfwidth=float(P.sbmf_temp_window),
                                                        flip=True, eta=float(P.eta_sbmf_hwhm))
        TK_hwhm.append(hwhm_res)
        TK_temp.append(t_half)
        # Store a few representative rows for reproducibility checks.
        if abs(float(b0) - 0.30) < 1e-9 or abs(float(b0) - 0.50) < 1e-9 or abs(float(b0) - 0.58) < 1e-9:
            temp_sweep_rows.append((float(b0), tgrid.copy(), wT.copy()))
        pk_up.append(weighted_peak(omega, Apos, omega > -0.50))
        pk_lo.append(weighted_peak(omega, Apos, omega <= -0.50))
        te_refs.append(float(np.real(te)))

    TK_sw = np.asarray(TK_sw, float)
    TK_ep = np.asarray(TK_ep, float)
    TK_hwhm = np.asarray(TK_hwhm, float)
    TK_temp = np.asarray(TK_temp, float)
    W_fsr_pos = np.asarray(W_fsr_pos, float)
    W_cp_pos = np.asarray(W_cp_pos, float)
    W_transfer_ratio = np.asarray(W_transfer_ratio, float)
    Gamma_area_fsr = np.asarray(Gamma_area_fsr, float)
    hwhm_valid = np.asarray(hwhm_valid, dtype=bool)
    neg_weight = np.asarray(neg_weight, float)
    transfer_valid = np.isfinite(W_transfer_ratio) & (W_transfer_ratio > 0.0) & (neg_weight <= 1e-9)
    W_transfer_ratio_valid = np.where(transfer_valid, W_transfer_ratio, np.nan)
    Gamma_area_fsr_valid = np.where(transfer_valid, Gamma_area_fsr, np.nan)
    # The raw HWHM is a useful non-circular spectral check only in its domain of
    # validity.  We keep invalid points as NaN and smooth only contiguous valid
    # segments, so the plotted trend cannot visually extend into the
    # PT-broken/non-Lorentzian regime where FWHM has no physical meaning.
    TK_hwhm = np.where(hwhm_valid, TK_hwhm, np.nan)
    TK_hwhm_trend = smooth_finite_segments(TK_hwhm, window=7)
    Zep = np.asarray(Zep, float)
    sabs = np.asarray(sabs, float)
    pk_up = np.asarray(pk_up, float)
    pk_lo = np.asarray(pk_lo, float)
    te_refs = np.asarray(te_refs, float)
    neg_min = np.asarray(neg_min, float)
    neg_weight = np.asarray(neg_weight, float)
    # EP-enhancement ratio: how much the biorthogonal EP residue lifts the
    # screening scale above the plain Schrieffer-Wolff estimate.  This is where
    # the EP signature lives; on the raw log scale a factor ~2 bump is invisible.
    enh = TK_ep / np.clip(TK_sw, 1e-300, None)
    enh = np.where(np.isfinite(enh), enh, np.nan)

    # Relative enhancement factors are used for visual comparison because
    # the analytic SW/BA estimates and the operational SBMF spectral width have
    # different absolute normalizations.  The main text shows only the
    # independent SBMF FSR/HWHM trend; the SW/BA comparison is moved to SFig. 3.
    beta_ref_sbmf = 0.30
    def _norm_at_ref(y, x=betas, bref=beta_ref_sbmf):
        y = np.asarray(y, float)
        x = np.asarray(x, float)
        if y.size == 0:
            return y
        idx = int(np.nanargmin(np.abs(x - bref)))
        denom = y[idx]
        if not np.isfinite(denom) or abs(denom) < 1e-300:
            finite = np.where(np.isfinite(y) & (np.abs(y) > 1e-300))[0]
            denom = y[finite[0]] if finite.size else 1.0
        return y / denom

    TK_sw_norm = _norm_at_ref(TK_sw)
    TK_ep_norm = _norm_at_ref(TK_ep)
    TK_hwhm_norm = _norm_at_ref(TK_hwhm)
    TK_hwhm_trend_norm = _norm_at_ref(TK_hwhm_trend)
    transfer_norm = _norm_at_ref(W_transfer_ratio_valid)
    transfer_trend = smooth_finite_segments(W_transfer_ratio_valid, window=7)
    transfer_trend_norm = _norm_at_ref(transfer_trend)
    area_width_norm = _norm_at_ref(Gamma_area_fsr_valid)

    save_curve(out, "Fig3_TK_SW", betas, TK_sw, "beta0 TK_SW")
    save_curve(out, "Fig3_TK_EP_assisted", betas, TK_ep, "beta0 TK_EP_assisted")
    save_curve(out, "Fig3_TK_SBMF_HWHM", betas, TK_hwhm, "beta0 raw_TK_SBMF_FSR_HWHM_at_shifted_resonance_tbeta_absbeta0_times_absbc")
    save_curve(out, "Fig3_TK_SBMF_HWHM_trend", betas, TK_hwhm_trend, "beta0 smoothed_TK_SBMF_FSR_HWHM_trend_raw_values_in_Fig3_TK_SBMF_HWHM")
    save_curve(out, "Fig3_TK_SW_norm_beta_ref_0p30", betas, TK_sw_norm, "beta0 TK_SW_over_value_at_beta_ref_0p30")
    save_curve(out, "Fig3_TK_BA_norm_beta_ref_0p30", betas, TK_ep_norm, "beta0 TK_BA_over_value_at_beta_ref_0p30")
    save_curve(out, "Fig3_TK_SBMF_FSR_HWHM_norm_beta_ref_0p30", betas, TK_hwhm_norm, "beta0 SBMF_FSR_HWHM_over_value_at_beta_ref_0p30")
    save_curve(out, "Fig3_TK_SBMF_FSR_HWHM_trend_norm_beta_ref_0p30", betas, TK_hwhm_trend_norm, "beta0 smoothed_SBMF_FSR_HWHM_over_value_at_beta_ref_0p30")
    save_curve(out, "Fig3_TK_SBMF_Tsweep", betas, TK_temp, "beta0 FDR_window_temperature_scale_re_solved_diagnostic_not_main_TK")
    save_curve(out, "Fig3_SBMF_W_FSR_positive", betas, W_fsr_pos, "beta0 positive_FSR_window_spectral_weight")
    save_curve(out, "Fig3_SBMF_W_CP_positive", betas, W_cp_pos, "beta0 positive_central_peak_window_spectral_weight")
    save_curve(out, "Fig3_SBMF_transfer_ratio", betas, W_transfer_ratio, "beta0 positive_spectral_weight_transfer_ratio_W_FSR_over_W_CP")
    save_curve(out, "Fig3_SBMF_transfer_ratio_valid", betas, W_transfer_ratio_valid, "beta0 positive_spectral_weight_transfer_ratio_valid_until_negative_signed_weight")
    save_curve(out, "Fig3_SBMF_transfer_ratio_norm_beta_ref_0p30", betas, transfer_norm, "beta0 normalized_positive_transfer_ratio_W_FSR_over_W_CP")
    save_curve(out, "Fig3_SBMF_transfer_ratio_trend_norm_beta_ref_0p30", betas, transfer_trend_norm, "beta0 smoothed_normalized_positive_transfer_ratio_W_FSR_over_W_CP")
    save_curve(out, "Fig3_SBMF_area_width_FSR", betas, Gamma_area_fsr, "beta0 area_over_peak_FSR_width_W_FSR_over_pi_Amax")
    save_curve(out, "Fig3_SBMF_area_width_FSR_valid", betas, Gamma_area_fsr_valid, "beta0 area_over_peak_FSR_width_valid_until_negative_signed_weight")
    save_curve(out, "Fig3_SBMF_area_width_FSR_norm_beta_ref_0p30", betas, area_width_norm, "beta0 normalized_area_over_peak_FSR_width")
    # representative temperature-sweep curves W(T) for reproducibility
    for bval, tg, wg in temp_sweep_rows:
        save_curve(out, f"Fig3_Tsweep_weight_beta_{bval:.2f}", tg, wg, "T FDR_window_weight")
    save_curve(out, "Fig3_EP_distance", betas, sabs, "beta0 abs_s_eff")
    save_curve(out, "Fig3_peak_upper", betas, pk_up, "beta0 omega_upper_peak")
    save_curve(out, "Fig3_peak_lower", betas, pk_lo, "beta0 omega_lower_peak")
    save_curve(out, "Fig3_teps_ref", betas, te_refs, "beta0 Re_teps")
    save_curve(out, "Fig3_signed_DOS_min", betas, neg_min, "beta0 min_A_imp_signed")
    save_curve(out, "Fig3_negative_weight", betas, neg_weight, "beta0 integrated_negative_signed_weight")
    save_curve(out, "Fig3_SBMF_HWHM_valid_mask", betas, hwhm_valid.astype(float), "beta0 valid_FSR_HWHM_mask_1_valid_0_invalid")
    with open(out / "Fig3_SBMF_HWHM_invalid_reasons.txt", "w", encoding="utf-8") as f:
        f.write("# beta0 valid reason\n")
        for bb, vv, rr in zip(betas, hwhm_valid, hwhm_reason):
            f.write(f"{bb:.8g} {int(bool(vv))} {rr}\n")
    save_curve(out, "Fig3_TK_enhancement", betas, enh, "beta0 TK_BA_over_TK_SW")
    save_curve(out, "Fig3_Z_EP", betas, Zep, "beta0 Z_EP_biorthogonal")

    fig = _plt.figure(figsize=(FULL_W, FULL_W * 0.68))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.36)

    # (a) Main-text non-circular SBMF spectral-weight-transfer check.
    # The analytic SW/BA scale comparison is intentionally moved to SFig. 3.
    axa = fig.add_subplot(gs[0, 0])
    zlo, zhi = 0.30, 0.60
    zm = (betas >= zlo) & (betas <= zhi)
    axa.plot(betas[zm], transfer_trend_norm[zm], color=C["purple"], lw=LW_MAIN, ls="-",
             label=r"$W_{\rm FSR}^+/W_{\rm CP}^+$ trend")
    axa.scatter(betas[zm][::3], transfer_norm[zm][::3], s=8.0, color=C["purple"],
                alpha=0.42, linewidths=0.0, zorder=5,
                label=r"raw transfer samples")
    broken = zm & (neg_weight > 1e-9)
    if np.any(broken):
        first_broken = float(betas[np.where(broken)[0][0]])
        axa.axvspan(first_broken, zhi, color=C["grey"], alpha=0.10, lw=0.0)
        axa.text(first_broken + 0.006, 0.90, r"signed response",
                 transform=axa.get_xaxis_transform(), ha="left", va="bottom",
                 fontsize=4.9, color=C["grey"])
    axa.axhline(1.0, color=C["grey"], lw=LW_GUIDE, ls=":")
    axa.axvspan(max(zlo, beta_ep-0.035), min(zhi, beta_ep+0.035),
                color=C["red"], alpha=0.11, lw=0.0, label=r"EP window")
    annotate_ep(axa, beta_ep, betas, label=False)
    axa.set_xlim(zlo, zhi)
    vals = np.concatenate([transfer_trend_norm[zm][np.isfinite(transfer_trend_norm[zm])],
                           transfer_norm[zm][np.isfinite(transfer_norm[zm])]])
    ytop = np.nanmax(vals) if vals.size else 1.0
    axa.set_ylim(0.75, max(1.6, ytop * 1.12))
    axa.set_xlabel(r"$\beta_0$")
    axa.set_ylabel(r"normalized transfer")
    axa.legend(loc="upper left", fontsize=4.65, handlelength=1.05,
               handletextpad=0.34, labelspacing=0.20,
               bbox_to_anchor=(0.035, 0.98), frameon=True, framealpha=0.88,
               facecolor="white", edgecolor="none")
    thicken(axa); tag(axa, "(a)")
    axa.set_title(r"SBMF spectral-weight transfer", pad=3)

    # (b) signed waterfall with cleaner (physical) broadening for the display.
    axb = fig.add_subplot(gs[0, 1])
    cols = [C["blue"], C["green"], C["orange"], C["red"], C["purple"], C["cyan"]]
    any_negative = False
    eta_wf = max(float(eta), 0.02)   # smoother peaks; keeps broken-phase dip
    if dos_xlim is not None and len(dos_xlim) == 2:
        xmask = (omega >= float(dos_xlim[0])) & (omega <= float(dos_xlim[1]))
    else:
        xmask = np.ones_like(omega, dtype=bool)
    for i, b0 in enumerate(dos_betas):
        d, b = solve(float(b0), eps_xi, T, flip=True)
        tb = gauge_tbeta(float(b0), b)
        te = eps_xi + np.real(d)
        A = np.asarray(physical_impurity_dos(te, tb, omega, eta_wf, flip=True), float)
        norm = np.nanmax(np.abs(A[xmask])) or np.nanmax(np.abs(A)) or 1.0
        y = A / norm
        base = i * waterfall_offset
        yoff = y + base
        col = cols[i % len(cols)]
        axb.axhline(base, color=C["grey"], lw=LW_GUIDE, alpha=0.35)
        axb.plot(omega[xmask], yoff[xmask], color=col, lw=LW_MAIN,
                 label=rf"$\beta_0={b0:.2f}$")
        neg = (A < -1e-9) & xmask
        if np.any(neg):
            any_negative = True
            axb.fill_between(omega[neg], base, yoff[neg], color=C["red"], alpha=0.18, lw=0.0)
        save_curve(out, f"Fig3_signed_DOS_stack_beta_{b0:.2f}", omega, y,
                   "omega signed_A_imp_normalized")
    if dos_xlim is not None and len(dos_xlim) == 2:
        axb.set_xlim(float(dos_xlim[0]), float(dos_xlim[1]))
    # headroom so the top curve + its baseline label do not hit the frame.
    axb.set_ylim(-0.40, (len(dos_betas) - 1) * waterfall_offset + 2.55)
    axb.set_xlabel(r"$\omega$")
    axb.set_ylabel(r"signed $A_{\mathrm{imp}}$ + offset")
    axb.legend(loc="upper center", fontsize=5.6, ncol=len(dos_betas),
               columnspacing=0.75, handlelength=0.95, handletextpad=0.32,
               frameon=True, framealpha=0.88, facecolor="white", edgecolor="none")
    # Negative parts are shown by red shading and explained in the caption;
    # no extra in-panel note is used, to keep the omega-axis uncluttered.
    thicken(axb); tag(axb, "(b)")
    axb.set_title(r"Peak merging (signed spectra)", pad=3)

    # (c) three-regime DOS.
    axc = fig.add_subplot(gs[1, 0])
    beta_reg = getattr(P, "beta0_dos3", 0.25)
    regimes = [
        ("Kondo local moment", P.eps_xi_4, 0.005, C["blue"], "-"),
        ("resonant / mixed valence", 0.0, 0.050, C["orange"], "--"),
        ("free orbital (high $T$)", P.eps_xi_4, 2.000, C["green"], "-."),
    ]
    for lab, ex, Treg, col, ls in regimes:
        d, b = solve(float(beta_reg), ex, Treg, flip=True)
        tb = gauge_tbeta(float(beta_reg), b)
        te = ex + np.real(d)
        # Heavier broadening here purely for legibility of the regime comparison:
        # the finite-k spectral grid is spiky at small eta and the qualitative
        # regime contrast (Kondo vs mixed-valence vs free-orbital) is what matters.
        eta_reg = max(float(getattr(P, "eta_broad", eta)), 0.05)
        A = np.asarray(physical_impurity_dos(te, tb, omega, eta_reg, flip=True), float)
        # Smooth the positive comparison curves to remove finite-k-grid notches
        # that otherwise appear as artificial vertical drops on a log axis.
        Aplot = normalized_for_plot(A, smooth=True, nonnegative=True)
        axc.semilogy(omega, np.clip(Aplot, 1e-5, None), color=col, ls=ls, lw=LW_MAIN,
                     label=lab)
        save_curve(out, "Fig3_regime_" + lab.split()[0].replace("/", "_"),
                   omega, Aplot, "omega normalized_positive_A_imp_smooth")
    axc.set_ylim(8e-6, 2.0)
    axc.set_xlabel(r"$\omega$")
    axc.set_ylabel(r"normalized $A_{\mathrm{imp}}$")
    axc.legend(loc="lower center", fontsize=5.4, ncol=1, frameon=True, framealpha=0.88, facecolor="white", edgecolor="none")
    thicken(axc); tag(axc, "(c)")
    axc.set_title(rf"Three spectral regimes ($\beta_0={beta_reg:g}$)", pad=3)

    # (d) peak positions and separation.
    # Final main-text choice: no small peak-gap inset; the filled band is enough.
    axd = fig.add_subplot(gs[1, 1])
    valid = np.isfinite(pk_up) & np.isfinite(pk_lo) & (betas > 0.15)
    if np.any(valid):
        axd.fill_between(betas[valid], pk_lo[valid], pk_up[valid], color=C["purple"],
                         alpha=0.12, lw=0.0, label=r"peak separation")
    axd.plot(betas[valid], pk_up[valid], color=C["blue"], lw=LW_MAIN, label=r"upper peak")
    axd.plot(betas[valid], pk_lo[valid], color=C["orange"], lw=LW_MAIN, label=r"lower peak")
    axd.plot(betas[valid], te_refs[valid], color=C["green"], ls=":", lw=LW_SECONDARY,
             label=r"$\mathrm{Re}\,\tilde\epsilon_\xi$")
    annotate_ep(axd, beta_ep, betas, label=False)
    axd.set_xlim(0.0, 0.60)
    axd.set_ylim(-1.75, 0.65)
    axd.set_xlabel(r"$\beta_0$")
    axd.set_ylabel(r"peak position $\omega$")
    axd.legend(loc="upper left", bbox_to_anchor=(0.12, 0.98), fontsize=5.4, ncol=2, columnspacing=0.8,
               handlelength=1.0, frameon=True, framealpha=0.88, facecolor="white", edgecolor="none")
    thicken(axd); tag(axd, "(d)")
    axd.set_title(r"Peak tracking", pad=3)

    save_fig(fig, out, "Fig3")


# ---------------------------------------------------------------------
#  Fig. 4
# ---------------------------------------------------------------------
def make_fig4(out: Path, fig4_beta0=None, show_signed_spectrum: bool = True) -> None:
    eps_xi = P.eps_xi_5      # unified: -U/2 = -1.0
    T = P.T_5
    omega = np.asarray(P.omega_fdr, float)
    eta = P.eta_fdr
    beta0 = float(P.beta0_fig4_spec if fig4_beta0 is None else fig4_beta0)

    d, b = solve(beta0, eps_xi, T, flip=True)
    tb = gauge_tbeta(beta0, b)
    te = eps_xi + np.real(d)
    fFD = np.asarray(f_fermi(omega, T), float)

    # Block-resolved retarded + eigenmode lesser on the 4x4 kernel.
    nF = len(omega)
    Gr = np.zeros((nF, 4, 4), dtype=complex)
    Geig = np.zeros((nF, 4, 4), dtype=complex)
    nk = len(K_LINEAR)
    for k in K_LINEAR:
        Hk = build_Hss(te, tb, k, flip=True)
        ev, R, L = eig_lr(Hk)
        fn = f_fermi(np.real(ev), T)
        projectors = [np.outer(R[:, n], L[n, :]) for n in range(4)]
        denom = omega[:, None] + 1j * eta - ev[None, :]
        for iw in range(nF):
            Gk = sum(projectors[n] / denom[iw, n] for n in range(4))
            Ge = 1j * sum(fn[n] * projectors[n] / denom[iw, n] for n in range(4))
            Gr[iw] += Gk / nk
            Geig[iw] += Ge / nk
    Ga = np.conj(Gr.transpose(0, 2, 1))
    GrGa = Gr - Ga
    Gbath = fFD[:, None, None] * GrGa

    imp = np.array([0, 1]); bath = np.array([2, 3])

    def _btr(M, idx):
        return np.trace(M[np.ix_(idx, idx)])

    def _ratio(Gless, idx):
        out_r = np.full(nF, np.nan)
        for i in range(nF):
            den = _btr(GrGa[i], idx)
            if abs(den) > 1e-14:
                out_r[i] = float(np.real(_btr(Gless[i], idx) / den))
        return out_r

    F_imp = _ratio(Gbath, imp)
    F_bath = _ratio(Gbath, bath)
    F_imp_eig = _ratio(Geig, imp)

    A_imp_blk = np.array([-np.imag(_btr(GrGa[i], imp)) / (2 * np.pi) for i in range(nF)])

    def _mask_low(y, A, frac=1e-3):
        yy = np.array(y, float, copy=True)
        mx = np.nanmax(np.abs(A))
        if np.isfinite(mx) and mx > 0:
            yy[np.abs(A) < frac * mx] = np.nan
        return yy

    F_imp_eig_m = np.clip(_mask_low(F_imp_eig, A_imp_blk), -1.5, 1.5)

    # Panel (d): physical impurity spectral response.
    # The FDR panels (a)-(c) and the displayed spectrum use the same beta0 by
    # default.  This default is sub-EP, so the main Fig. 4 spectrum remains
    # causal.  A beta0 > beta_EP should be used only as an explicit diagnostic
    # override; the negative-DOS onset is documented in SFig4.
    beta0_spec = beta0 if fig4_beta0 is not None else float(P.beta0_fig4_spec)
    eta_spec = float(P.eta_fig4_spec)
    d_s, b_s = solve(beta0_spec, eps_xi, T, flip=True)
    tb_s = gauge_tbeta(beta0_spec, b_s)
    te_s = eps_xi + np.real(d_s)
    A_imp_signed = np.asarray(
        physical_impurity_dos(te_s, tb_s, omega, eta_spec, flip=True), float)
    w_imp, y_imp = omega, A_imp_signed

    fig = _plt.figure(figsize=(FULL_W, FULL_W * 0.56))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.34)

    # (a) Physical FDR correspondence.
    axa = fig.add_subplot(gs[0, 0])
    axa.plot(omega, fFD, color=C["black"], lw=LW_MAIN, label=r"$f_{\mathrm{FD}}$")
    axa.plot(omega, F_imp, color=C["blue"], lw=LW_SECONDARY, ls="--", label="impurity FDR")
    axa.plot(omega, F_bath, color=C["green"], lw=LW_SECONDARY, ls=":", label="bath FDR")
    axa.axhline(0, color=C["grey"], lw=LW_GUIDE, alpha=0.60)
    axa.axhline(1, color=C["grey"], lw=LW_GUIDE, alpha=0.45, ls=":")
    axa.set_xlabel(r"$\omega$")
    axa.set_ylabel("FDR ratio")
    axa.legend(loc="center left", fontsize=5.6, frameon=True, framealpha=0.88, facecolor="white", edgecolor="none")
    thicken(axa); tag(axa, "(a)")
    axa.set_title("Bath-constructed FDR", pad=3)

    # (b) FDR residuals (corrected direct computation).
    axb = fig.add_subplot(gs[0, 1])
    dev_imp = np.abs(F_imp - fFD)
    dev_bath = np.abs(F_bath - fFD)
    floor = 1e-16
    axb.semilogy(omega, np.clip(dev_imp, floor, None), color=C["blue"], lw=LW_SECONDARY,
                 label=r"$|F_{\mathrm{imp}}-f_{\mathrm{FD}}|$")
    axb.semilogy(omega, np.clip(dev_bath, floor, None), color=C["green"], lw=LW_SECONDARY, ls=":",
                 label=r"$|F_{\mathrm{bath}}-f_{\mathrm{FD}}|$")
    axb.set_xlabel(r"$\omega$")
    axb.set_ylabel("FDR residual")
    axb.legend(loc="upper right", fontsize=5.5, frameon=True, framealpha=0.88, facecolor="white", edgecolor="none")
    thicken(axb); tag(axb, "(b)")
    axb.set_title("Impurity/bath correspondence", pad=3)
    max_res = float(np.nanmax([np.nanmax(dev_imp), np.nanmax(dev_bath)]))

    # (c) Eigenmode diagnostic, separate from physical FDR.
    axc = fig.add_subplot(gs[1, 0])
    axc.plot(omega, F_imp_eig_m, ls="none", marker=".", ms=2.2,
             color=C["purple"], alpha=0.34, label="eigenmode diagnostic")
    axc.plot(omega, fFD, color=C["black"], lw=LW_AUX, alpha=0.45, label=r"$f_{\mathrm{FD}}$ ref")
    axc.axhline(0, color=C["grey"], lw=LW_GUIDE, alpha=0.60)
    axc.axhline(1, color=C["grey"], lw=LW_GUIDE, alpha=0.45, ls=":")
    axc.text(0.03, 0.05, "diagnostic only; not a distribution",
             transform=axc.transAxes, ha="left", va="bottom", fontsize=5.6,
             bbox=dict(fc="white", ec=C["grey"], lw=0.5, alpha=0.82, pad=1.4))
    axc.set_xlabel(r"$\omega$")
    axc.set_ylabel("eigenmode ratio")
    axc.legend(loc="upper right", fontsize=5.1, frameon=True, framealpha=0.88, facecolor="white", edgecolor="none")
    thicken(axc); tag(axc, "(c)")
    axc.set_title("Nonthermal eigenmode diagnostic", pad=3)

    # (d) Signed spectra and negative spectral weight.
    axd = fig.add_subplot(gs[1, 1])
    neg = y_imp < -1e-9
    neg_area = float(np.trapezoid(np.clip(-y_imp, 0.0, None), w_imp))
    if show_signed_spectrum and np.any(neg):
        axd.axhline(0.0, color=C["grey"], lw=LW_GUIDE, alpha=0.65)
        axd.fill_between(w_imp, 0.0, y_imp, where=neg, color=C["red"],
                         alpha=0.22, lw=0.0, label="negative part")
        axd.plot(w_imp, y_imp, color=C["blue"], lw=LW_MAIN, label=r"signed $A_{\mathrm{imp}}$")
        # Linear scale keeps the negative spectral weight readable and avoids symlog tick-label collisions.
    else:
        axd.plot(w_imp, np.clip(y_imp, 0.0, None), color=C["blue"], lw=LW_MAIN,
                 label=r"$A_{\mathrm{imp}}$")
    ypos = np.clip(y_imp, 0.0, None)
    peak_w = float("nan")
    if np.nanmax(ypos) > 0:
        peak_w = float(w_imp[int(np.nanargmax(ypos))])
        axd.axvline(peak_w, color=C["orange"], ls=":", lw=LW_AUX, label="imp. peak")
    axd.axvline(0.0, color=C["black"], ls="-", lw=LW_GUIDE, alpha=0.45, label=r"$\omega=0$")
    axd.axvline(float(np.real(te_s)), color=C["red"], ls="--", lw=LW_AUX,
                label=r"$\mathrm{Re}\,\tilde\epsilon_\xi$")
    yy = y_imp if (show_signed_spectrum and np.any(neg)) else np.clip(y_imp, 0.0, None)
    ymin, ymax = float(np.nanmin(yy)), float(np.nanmax(yy))
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        pad = 0.09 * (ymax - ymin)
        axd.set_ylim(ymin - pad, ymax + pad)
    axd.set_xlabel(r"$\omega$")
    axd.set_ylabel("spectral response")
    axd.legend(loc="upper right", fontsize=5.0, frameon=True, framealpha=0.88, facecolor="white", edgecolor="none")
    thicken(axd); tag(axd, "(d)")
    _causal = "causal" if not np.any(neg) else "PT-broken"
    axd.set_title(rf"Physical spectrum ($\beta_0={beta0_spec:.2f}$, $\eta={eta_spec:.3f}$, {_causal})", pad=3)

    save_curve(out, "Fig4_fFD", omega, fFD, "omega fFD")
    save_curve(out, "Fig4_F_impurity", omega, F_imp, "omega F_impurity_bath_constructed")
    save_curve(out, "Fig4_F_bath", omega, F_bath, "omega F_bath_bath_constructed")
    save_curve(out, "Fig4_FDR_residual_impurity", omega, np.clip(dev_imp, floor, None),
               "omega abs_F_imp_minus_fFD")
    save_curve(out, "Fig4_FDR_residual_bath", omega, np.clip(dev_bath, floor, None),
               "omega abs_F_bath_minus_fFD")
    save_curve(out, "Fig4_A_imp_signed", w_imp, y_imp, "omega signed_A_imp")
    save_curve(out, "Fig4_A_imp_negative_part", w_imp, np.clip(-y_imp, 0.0, None),
               "omega negative_part_minus_A_imp")

    save_fig(fig, out, "Fig4")
    print(f"  [Fig4] eps_xi={eps_xi}  FDR-panels: beta0={beta0:.4f} eta={eta:.0e}  "
          f"max_FDR_residual={max_res:.2e}")
    print(f"         spectral panel(d): beta0={beta0_spec:.4f} eta={eta_spec:.3f}  "
          f"Re(te)={np.real(te_s):+.4f}  peak={peak_w:+.3f}  "
          f"neg_area={neg_area:.3e}  "
          f"{'causal (A>=0)' if not np.any(neg) else 'PT-broken (negative present)'}")



def make_sfig4_broadening(out: Path, beta0_spec: float | None = None) -> None:
    """Supplementary sign-onset and broadening check for the negative DOS response.

    This is the important causality check: the signed physical DOS must remain
    nonnegative on the PT-unbroken side and should develop a negative component
    only after crossing the local EP.  The right panel therefore scans the
    negative spectral weight W_-(beta0, eta) for several moderate broadenings.
    The left panel shows representative spectra below, at, and above beta_EP
    at the same eta used in Fig. 4(d).
    """
    eps_xi = P.eps_xi_5
    T = P.T_5
    omega = np.linspace(float(P.omega_fdr[0]), float(P.omega_fdr[-1]), 260)
    eta_main = float(P.eta_fig4_spec)
    beta_ep = float(predicted_local_ep_tbeta() / max(float(P.r_fixed), 1e-12))
    below_beta = min(float(P.beta0_fig4_spec), beta_ep - 0.01)
    above_beta = float(P.beta0_sfig4_above_ep if beta0_spec is None else beta0_spec)
    if above_beta <= beta_ep:
        above_beta = beta_ep + 0.08
    beta_show = [below_beta, beta_ep, above_beta]
    beta_labels = [r"below EP", r"at EP", r"above EP"]
    beta_scan = np.linspace(0.30, 0.62, 41)
    etas = tuple(float(x) for x in getattr(P, "eta_fig4_scan", (0.014, 0.015, 0.020)))

    fig = _plt.figure(figsize=(FULL_W, FULL_W * 0.43))
    gs = fig.add_gridspec(1, 2, wspace=0.34)
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_onset = fig.add_subplot(gs[0, 1])

    # Representative spectra: below/at/above EP for the displayed broadening.
    cols_show = [C["blue"], C["grey"], C["red"]]
    representative_rows = []
    for b0, lab, col in zip(beta_show, beta_labels, cols_show):
        d, b = solve(float(b0), eps_xi, T, flip=True)
        tb = gauge_tbeta(float(b0), b)
        te = eps_xi + np.real(d)
        A = np.asarray(physical_impurity_dos(te, tb, omega, eta_main, flip=True), float)
        neg_area = float(np.trapezoid(np.clip(-A, 0.0, None), omega))
        min_A = float(np.nanmin(A))
        representative_rows.append((float(b0), eta_main, min_A, neg_area))
        ax_spec.plot(omega, A, color=col, lw=LW_MAIN, label=rf"{lab}: $\beta_0={b0:.2f}$")
        if b0 > beta_ep and np.any(A < -1e-9):
            ax_spec.fill_between(omega, 0.0, A, where=(A < -1e-9), color=C["red"], alpha=0.18, lw=0.0)
        save_curve(out, f"SFig4_onset_spectrum_beta_{b0:.3f}_eta_{eta_main:.3f}",
                   omega, A, "omega signed_A_imp")
    ax_spec.axhline(0.0, color=C["black"], lw=LW_GUIDE, alpha=0.70)
    ax_spec.axvline(0.0, color=C["black"], lw=LW_GUIDE, alpha=0.35)
    ax_spec.set_xlabel(r"$\omega$")
    ax_spec.set_ylabel(r"signed $A_{\mathrm{imp}}(\omega)$")
    ax_spec.set_title(rf"No negative DOS below $\beta_{{\rm EP}}$ ($\eta={eta_main:.3f}$)", pad=3)
    ax_spec.legend(loc="upper right", fontsize=5.6, frameon=True, framealpha=0.88,
                   facecolor="white", edgecolor="none")
    thicken(ax_spec); tag(ax_spec, "S4(a)")

    # Onset scan: negative spectral weight is zero below EP and nonzero above EP.
    cols = [C["blue"], C["red"], C["green"], C["purple"]]
    onset_rows = []
    below_max = {eta: 0.0 for eta in etas}
    above_max = {eta: 0.0 for eta in etas}
    for i, eta in enumerate(etas):
        neg_area_scan = []
        min_scan = []
        for b0 in beta_scan:
            d, b = solve(float(b0), eps_xi, T, flip=True)
            tb = gauge_tbeta(float(b0), b)
            te = eps_xi + np.real(d)
            A = np.asarray(physical_impurity_dos(te, tb, omega, eta, flip=True), float)
            neg_area = float(np.trapezoid(np.clip(-A, 0.0, None), omega))
            min_A = float(np.nanmin(A))
            neg_area_scan.append(neg_area)
            min_scan.append(min_A)
            onset_rows.append((float(b0), float(eta), min_A, neg_area))
        neg_area_scan = np.asarray(neg_area_scan, float)
        min_scan = np.asarray(min_scan, float)
        below = beta_scan <= beta_ep + 1e-12
        above = beta_scan > beta_ep + 1e-12
        below_max[eta] = float(np.nanmax(neg_area_scan[below])) if np.any(below) else float("nan")
        above_max[eta] = float(np.nanmax(neg_area_scan[above])) if np.any(above) else float("nan")
        col = cols[i % len(cols)]
        ax_onset.plot(beta_scan, neg_area_scan, color=col,
                      lw=LW_MAIN if abs(eta - eta_main) < 1e-12 else LW_SECONDARY,
                      ls="-" if abs(eta - eta_main) < 1e-12 else "--",
                      label=rf"$\eta={eta:.3f}$")
        save_curve(out, f"SFig4_negative_area_vs_beta_eta_{eta:.3f}",
                   beta_scan, neg_area_scan, "beta0 negative_area_W_minus")
    ax_onset.axvline(beta_ep, color=C["black"], lw=LW_GUIDE, ls=(0, (4, 2)), alpha=0.85)
    ax_onset.text(beta_ep, 0.96, r"$\beta_{\rm EP}$", transform=ax_onset.get_xaxis_transform(),
                  ha="center", va="top", fontsize=6.0,
                  bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.5))
    ax_onset.axhline(0.0, color=C["black"], lw=LW_GUIDE, alpha=0.55)
    ax_onset.set_xlabel(r"$\beta_0$")
    ax_onset.set_ylabel(r"negative weight $W_-$")
    ax_onset.set_title(r"Sign defect turns on only after the EP", pad=3)
    ax_onset.legend(loc="upper left", fontsize=5.8, frameon=True, framealpha=0.88,
                    facecolor="white", edgecolor="none")
    thicken(ax_onset); tag(ax_onset, "S4(b)", x=0.78, y=0.94)
    save_fig(fig, out, "SFig4_negative_onset_broadening_check")

    with open(out / "SFig4_negative_onset_table.dat", "w", encoding="utf-8") as f:
        f.write("# beta0  eta  min_A  negative_area_W_minus\n")
        for row in onset_rows:
            f.write("{:.8e} {:.8e} {:.8e} {:.8e}\n".format(*row))
    with open(out / "SFig4_negative_onset_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"beta_EP = {beta_ep:.8f}\n")
        f.write(f"eta_main = {eta_main:.8f}\n")
        f.write("# eta  max_Wminus_beta_le_EP  max_Wminus_beta_gt_EP\n")
        for eta in etas:
            f.write(f"{eta:.8e} {below_max[eta]:.8e} {above_max[eta]:.8e}\n")
        f.write("# representative spectra: beta0 eta min_A Wminus\n")
        for row in representative_rows:
            f.write("{:.8e} {:.8e} {:.8e} {:.8e}\n".format(*row))




def _gain_loss_bloch_trajectory(beta0: float, dval: complex, bval: complex,
                                eps_xi: float, flip: bool, tgrid: np.ndarray):
    """Supplemental gain/loss-orbital Bloch diagnostic."""
    te = eps_xi + np.real(dval)
    tb = gauge_tbeta(beta0, bval)
    H = build_Hss(te, tb, 0.0, flip=flip)
    psi0 = np.zeros(4, dtype=complex)
    psi0[0] = 1.0
    zvals, yvals, nimp_vals = [], [], []
    for tt in tgrid:
        psi = expm(-1j * H * float(tt)) @ psi0
        up, dn = psi[0], psi[1]
        nup, ndn = abs(up)**2, abs(dn)**2
        nimp = max(float(nup + ndn), 1e-30)
        zvals.append(float((nup - ndn) / nimp))
        yvals.append(float(2.0 * np.imag(np.conj(up) * dn) / nimp))
        nimp_vals.append(float(nimp))
    return np.asarray(zvals), np.asarray(yvals), np.asarray(nimp_vals)


def make_sfig1_bloch_supplement(out: Path, time_tmax=None, time_points: int = 900) -> None:
    """Supplementary gain/loss-orbital Bloch analysis.

    This is kept out of the main Fig. 1 because it is a useful diagnostic but
    is not used in the main figure. The main figure instead shows the
    simpler survival-amplitude portrait.
    """
    eps_xi = P.eps_xi_12
    T = P.T_12
    betas = np.asarray(P.beta_vals, float)
    sw = sweep(eps_xi, T, True, betas)
    kap = np.asarray(sw["kap"], float)
    beta_ep = _beta_ep_from_sweep(betas, kap, flip=True, b_arr=np.asarray(sw.get("b"), dtype=object))
    trip_b = _choose_triplet(betas, sw, beta_ep, flip=True)
    labels = [r"small $\beta_0$", r"near EP", r"above EP"]
    cols = [C["blue"], C["orange"], C["green"]]

    raw = []
    for b0 in trip_b:
        idxb = int(np.nanargmin(np.abs(betas - b0)))
        d, b = sw["d"][idxb], sw["b"][idxb]
        t, O = evolve(float(b0), eps_xi, T, True, d=d, b=b, mode="amplitude")
        t = np.asarray(t, float)
        if t.size >= 2:
            raw.append((float(b0), d, b, t))
    # Fixed short-time window for SFig. 1 only.  This deliberately avoids
    # the visually dominant late-time exponential amplification above the EP
    # while leaving the main Fig. 1 dynamics and all solver routines unchanged.
    SFIG1_TMAX = 15.0
    t_start = 0.0
    t_stop = SFIG1_TMAX
    grid = np.linspace(t_start, t_stop, max(int(time_points), 64), endpoint=True)

    fig = _plt.figure(figsize=(FULL_W, 2.45))
    gs = fig.add_gridspec(1, 2, wspace=0.34)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    for i, ((b0, d, b, _t), col, lab) in enumerate(zip(raw, cols, labels)):
        z, y, nimp = _gain_loss_bloch_trajectory(float(b0), d, b, eps_xi, True, grid)
        fade_plot(axa, z, y, col, lw=LW_MAIN)
        add_arrow(axa, z, y, col)
        axa.plot([z[0]], [y[0]], "o", color=col, ms=4.2,
                 mec=C["black"], mew=0.55, zorder=7)
        n0 = nimp[0] if abs(nimp[0]) > 1e-30 else 1.0
        axb.plot(grid, nimp / n0, color=col, lw=LW_MAIN, label=lab)
        save_curve(out, f"SFig1_bloch_{i+1}_beta_{b0:.4g}", z, y,
                   "z=(n_up-n_down)/(n_up+n_down)  y=2Im(up*_down)/(n_up+n_down)")
        save_curve(out, f"SFig1_Nimp_{i+1}_beta_{b0:.4g}", grid, nimp / n0,
                   "t_common  N_imp(t)/N_imp(0)")

    axa.set_xlabel(r"$z=(n_\uparrow-n_\downarrow)/N_{\rm imp}$")
    axa.set_ylabel(r"$y=2\,\mathrm{Im}(\psi_\uparrow^*\psi_\downarrow)/N_{\rm imp}$")
    axa.set_aspect("equal", adjustable="datalim")
    axa.legend([_Line2D([0], [0], color=cols[i], lw=LW_MAIN) for i in range(3)],
               labels, loc="upper right", fontsize=5.7, frameon=True,
               framealpha=0.88, facecolor="white", edgecolor="none")
    thicken(axa); tag(axa, "(a)")
    axa.set_title(r"Gain--loss Bloch diagnostic", pad=3)

    axb.axhline(1.0, color=C["grey"], lw=LW_GUIDE, ls=":")
    axb.set_xlabel(r"$t$")
    axb.set_ylabel(r"$N_{\rm imp}(t)/N_{\rm imp}(0)$")
    axb.set_yscale("log")
    axb.set_xlim(0.0, SFIG1_TMAX)
    axb.set_xticks([0, 3, 6, 9, 12, 15])
    axb.legend(loc="best", fontsize=5.7, frameon=True,
               framealpha=0.88, facecolor="white", edgecolor="none")
    thicken(axb); tag(axb, "(b)")
    axb.set_title(r"Impurity-sector norm check", pad=3)
    save_fig(fig, out, "SFig1_bloch_diagnostic")

def make_sfig3_supplement(out: Path) -> None:
    """Supplementary diagnostic companion to Fig. 3.

    Main Fig. 3 intentionally avoids small insets.  This companion stores the
    full-range logarithmic scale view and the peak-gap scalar diagnostic using
    exactly the same exported Fig. 3 data and the same line-weight hierarchy.
    """
    try:
        dat_sw = np.loadtxt(out / "Fig3_TK_SW.dat")
        dat_ba = np.loadtxt(out / "Fig3_TK_EP_assisted.dat")
        dat_h  = np.loadtxt(out / "Fig3_TK_SBMF_HWHM.dat")
        dat_tr = np.loadtxt(out / "Fig3_SBMF_transfer_ratio.dat")
        dat_aw = np.loadtxt(out / "Fig3_SBMF_area_width_FSR.dat")
        try:
            dat_ht = np.loadtxt(out / "Fig3_TK_SBMF_HWHM_trend.dat")
        except Exception:
            dat_ht = dat_h.copy(); dat_ht[:,1] = smooth_for_plot(dat_h[:,1], window=9)
        dat_t  = np.loadtxt(out / "Fig3_TK_SBMF_Tsweep.dat")
        dat_s  = np.loadtxt(out / "Fig3_EP_distance.dat")
        dat_u  = np.loadtxt(out / "Fig3_peak_upper.dat")
        dat_l  = np.loadtxt(out / "Fig3_peak_lower.dat")
        dat_neg = np.loadtxt(out / "Fig3_negative_weight.dat")
        dat_vmask = np.loadtxt(out / "Fig3_SBMF_HWHM_valid_mask.dat")
    except Exception as exc:
        print(f"  [SFig3] skipped supplement; missing Fig.3 data: {exc}")
        return
    betas = dat_sw[:, 0]
    tk_sw = dat_sw[:, 1]
    tk_ba = dat_ba[:, 1]
    tk_hwhm = dat_h[:, 1]
    tk_hwhm_trend = dat_ht[:, 1]
    transfer_ratio = dat_tr[:, 1]
    area_width = dat_aw[:, 1]
    tk_temp = dat_t[:, 1]
    sabs = dat_s[:, 1]
    pk_up = dat_u[:, 1]
    pk_lo = dat_l[:, 1]
    neg_weight = dat_neg[:, 1]
    valid_hwhm = dat_vmask[:, 1] > 0.5
    sep = np.abs(pk_up - pk_lo)
    beta_ep = predicted_local_ep_tbeta() / max(float(P.r_fixed), 1e-12)

    fig = _plt.figure(figsize=(FULL_W, FULL_W * 0.34))
    gs = fig.add_gridspec(1, 2, wspace=0.36)

    axa = fig.add_subplot(gs[0, 0])
    beta_ref = 0.30
    def _norm_at_ref_sfig(y):
        y = np.asarray(y, float)
        idx = int(np.nanargmin(np.abs(betas - beta_ref)))
        denom = y[idx]
        if not np.isfinite(denom) or abs(denom) < 1e-300:
            finite = np.where(np.isfinite(y) & (np.abs(y) > 1e-300))[0]
            denom = y[finite[0]] if finite.size else 1.0
        return y / denom
    sw_norm = _norm_at_ref_sfig(tk_sw)
    ba_norm = _norm_at_ref_sfig(tk_ba)
    h_norm = _norm_at_ref_sfig(tk_hwhm)
    ht_norm = _norm_at_ref_sfig(tk_hwhm_trend)
    tr_norm = _norm_at_ref_sfig(transfer_ratio)
    tr_trend = _norm_at_ref_sfig(smooth_for_plot(transfer_ratio, window=7))
    aw_norm = _norm_at_ref_sfig(area_width)
    axa.plot(betas, sw_norm, color=C["blue"], lw=LW_MAIN,
             label=rf"$T_K^{{\mathrm{{SW}}}}/T_K^{{\mathrm{{SW}}}}({beta_ref:.2f})$")
    axa.plot(betas, ba_norm, color=C["green"], lw=LW_MAIN, ls="--",
             label=rf"$T_K^{{\mathrm{{BA}}}}/T_K^{{\mathrm{{BA}}}}({beta_ref:.2f})$")
    axa.plot(betas, tr_trend, color=C["purple"], lw=LW_MAIN, ls="-.",
             label=r"SBMF transfer ratio")
    axa.scatter(betas[::4], tr_norm[::4], s=6.0,
                color=C["purple"], alpha=0.35, linewidths=0.0, zorder=5,
                label=r"raw transfer samples")
    if np.any(neg_weight > 1e-9):
        first_invalid = float(betas[np.where(neg_weight > 1e-9)[0][0]])
        axa.axvspan(first_invalid, betas[-1], color=C["grey"], alpha=0.08, lw=0.0)
    annotate_ep(axa, beta_ep, betas, label=True)
    axa.set_xlim(0.0, 0.60)
    cmp_mask = betas >= beta_ref
    allv = np.concatenate([sw_norm[cmp_mask][np.isfinite(sw_norm[cmp_mask])],
                           ba_norm[cmp_mask][np.isfinite(ba_norm[cmp_mask])],
                           tr_trend[cmp_mask][np.isfinite(tr_trend[cmp_mask])]])
    ymax = np.nanmax(allv) if allv.size else 1.0
    axa.set_xlim(beta_ref, 0.60)
    axa.set_ylim(0.0, max(1.4, 1.10*ymax))
    axa.set_xlabel(r"$\beta_0$")
    axa.set_ylabel(r"relative enhancement")
    axa.legend(loc="upper left", fontsize=4.6, frameon=True, framealpha=0.88,
               facecolor="white", edgecolor="none", handlelength=1.10,
               handletextpad=0.35, labelspacing=0.23)
    thicken(axa); tag(axa, "(a)")
    axa.set_title(r"Relative SW/BA/SBMF comparison", pad=3)

    axa2 = axa.twinx()
    s_norm = sabs / (np.nanmax(sabs) or 1.0)
    axa2.plot(betas, s_norm, color=C["grey"], lw=LW_GUIDE, ls=":")
    axa2.set_ylabel(r"normalized EP distance")
    for sp in axa2.spines.values():
        sp.set_linewidth(LW_SPINE)
    axa2.tick_params(direction="in", top=True, right=True, width=1.25, length=4.6)

    axb = fig.add_subplot(gs[0, 1])
    valid = np.isfinite(sep) & (betas > 0.15)
    axb.plot(betas[valid], sep[valid], color=C["purple"], lw=LW_MAIN, label=r"peak separation")
    annotate_ep(axb, beta_ep, betas, label=True)
    if np.any(~valid_hwhm):
        first_invalid = float(betas[np.where(~valid_hwhm)[0][0]])
        axb.axvspan(first_invalid, betas[-1], color=C["grey"], alpha=0.08, lw=0.0)
    axb.set_xlim(0.0, 0.60)
    axb.set_xlabel(r"$\beta_0$")
    axb.set_ylabel(r"$\Delta\omega$ between tracked peaks")
    thicken(axb); tag(axb, "(b)")
    axb.set_title(r"Broken-side diagnostics", pad=3)
    axb2 = axb.twinx()
    nw = neg_weight / (np.nanmax(neg_weight) if np.nanmax(neg_weight) > 0 else 1.0)
    axb2.plot(betas, nw, color=C["red"], lw=LW_SECONDARY, ls="--", label=r"negative weight")
    axb2.set_ylabel(r"normalized negative weight")
    for sp in axb2.spines.values():
        sp.set_linewidth(LW_SPINE)
    axb2.tick_params(direction="in", top=True, right=True, width=1.25, length=4.6)

    save_fig(fig, out, "SFig3_supplemental_diagnostics")


# ---------------------------------------------------------------------
#  Captions + driver
# ---------------------------------------------------------------------
def write_captions(out: Path) -> None:
    text = r"""
% PRB-style captions for Figs. 1--4 generated by prb_all_in_one_PRB_final_uniform.py.
% Supplementary SFig1_bloch_diagnostic.pdf contains the gain--loss Bloch/orbital-population analysis moved out of the main figure.
% Supplementary SFig3_supplemental_diagnostics.pdf contains the relative SW/BA/SBMF comparison and broken-side diagnostics moved out of the main figure.
% Supplementary SFig4_negative_onset_broadening_check.pdf verifies that the signed DOS is nonnegative for beta0 <= beta_EP and develops negative spectral weight only on the PT-broken side, across moderate eta values.
% Uniform convention: eps_xi = -U/2 = -1.0, U = 2.0, T(Fig4) = 0.1.

\caption{Full coherent impurity kernel.  (a) Real and imaginary parts of the four eigenvalues of the projected $4\times4$ non-Hermitian kernel versus $\beta_0$; dashed lines mark the solver-determined EP on the $\beta_0$ axis.  (b) Impurity biorthogonal condition number $\kappa_{\rm imp}$, regular with $\kappa_{\rm imp}(0)=1$ and finite peaks at $|\beta_0|\simeq0.5$.  (c) Representative time evolution for a very small below-EP drive, near the EP, and above the EP on a common time interval.  (d) Survival-amplitude portrait, $\mathrm{Re}\,O(t)$ versus $\mathrm{Im}\,O(t)$, showing the nearly closed small-drive reference and the distortion as the EP is approached and crossed.  The gain--loss Bloch/orbital-population analysis is moved to the supplement.}

\caption{Gain--loss-only control.  (a) Four eigenvalue branches with the coherent spin-flip channel removed.  (b) $\kappa_{\rm imp}$ shows no coherent local EP marker.  (c) Dynamics and (d) survival-amplitude portrait, using the same representation as Fig.~1 for direct comparison.}

\caption{Frozen spectral and scale diagnostics.  (a) Frozen SW and EP-assisted BA scale estimates together with an independent SBMF FSR-HWHM spectral-width trend extracted at the shifted resonance.  SBMF FWHM points are shown only while the local FSR resonance is positive and a half-maximum width is well defined; no FWHM is assigned in the negative-weight/non-Lorentzian regime.  (b) Signed waterfall spectra; red shading marks negative spectral weight in the PT-broken response.  (c) Normalized positive impurity spectra in the three regimes.  (d) Peak tracking versus $\beta_0$ with peak-separation band and $\mathrm{Re}\,\tilde\epsilon_\xi$.}

\caption{Bath-controlled FDR and physical/eigenmode diagnostics at a sub-EP drive.  (a) Impurity and bath FDR ratios follow $f_{\rm FD}(\omega)$.  (b) Absolute FDR residual, at the numerical floor.  (c) Eigenmode ratio shown only as a nonthermal diagnostic.  (d) Physical impurity spectral response at the same sub-EP control value, with $\omega=0$, $\mathrm{Re}\,\tilde\epsilon_\xi$, and the peak marked; the spectrum is causal in the main panel.  The PT-broken negative-DOS onset is documented separately in the supplemental broadening check.}
""".strip() + "\n"
    (out / "captions.tex").write_text(text)


def build_all_figures(outdir: Path, eta: float = 0.012,
                      dos_betas=(0.30, 0.42, 0.50, 0.58),
                      dos_xlim=(-2.2, 0.8), waterfall_offset: float = 1.20,
                      time_tmax=None, time_points: int = 900,
                      fig4_beta0=None, clip_fig4: bool = False,
                      which: str = "all") -> None:
    set_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to: {outdir}")
    print(f"Convention: eps_xi=-U/2={-P.U_default/2:.1f}, U={P.U_default}, "
          f"T(Fig4)={P.T_5}, lambda={P.lambda_k:.6g}, EP tbeta={predicted_local_ep_tbeta()}")
    if which in ("all", "fig1"):
        print(f"Fig1 dynamics: beta_left={P.fig1_left_scale}*beta_EP, "
              f"beta_right={P.fig1_right_scale}*beta_EP, "
              f"phase_common_norm={P.phase_common_norm}")
    if which in ("all", "fig1"):
        make_fig12(outdir, flip=True, fname="Fig1",
                   labels=["small beta", "near EP", "above EP"],
                   time_tmax=time_tmax, time_points=time_points)
        make_sfig1_bloch_supplement(outdir, time_tmax=time_tmax, time_points=time_points)
    if which in ("all", "fig2"):
        make_fig12(outdir, flip=False, fname="Fig2",
                   labels=["Hermitian ref", "weak gain/loss", "strong gain/loss"],
                   time_tmax=time_tmax, time_points=time_points)
    if which in ("all", "fig3"):
        make_fig3(outdir, eta=eta, dos_betas=list(dos_betas),
                  dos_xlim=list(dos_xlim), waterfall_offset=waterfall_offset)
        make_sfig3_supplement(outdir)
    if which in ("all", "fig4"):
        make_fig4(outdir, fig4_beta0=fig4_beta0, show_signed_spectrum=not clip_fig4)
        make_sfig4_broadening(outdir, beta0_spec=None)
    write_captions(outdir)
    print("Done ->", outdir)


def main() -> None:
    ap = _argparse.ArgumentParser(
        description="Unified all-in-one PRB figure generator (solver + all_in_one_2 style driver).")
    ap.add_argument("--out", default="figs_unified", help="output directory")
    ap.add_argument("--which", default="all",
                    choices=["all", "fig1", "fig2", "fig3", "fig4"])
    ap.add_argument("--eta", type=float, default=0.012,
                    help="broadening for the Fig. 3 DOS-vs-beta panel")
    ap.add_argument("--dos-betas", type=float, nargs="+",
                    default=[0.30, 0.42, 0.50, 0.58],
                    help="beta0 values for the signed DOS waterfall")
    ap.add_argument("--dos-xlim", type=float, nargs=2, default=[-2.2, 0.8],
                    help="omega window for Fig. 3 signed waterfall panel")
    ap.add_argument("--waterfall-offset", type=float, default=1.20)
    ap.add_argument("--time-tmax", type=float, default=None)
    ap.add_argument("--time-points", type=int, default=900)
    ap.add_argument("--fig1-left-scale", type=float, default=P.fig1_left_scale,
                    help="Fig. 1 below-EP trajectory as a multiplier of beta_EP, e.g. 0.10 -> beta0=0.10*beta_EP")
    ap.add_argument("--fig1-right-scale", type=float, default=P.fig1_right_scale,
                    help="Fig. 1 above-EP trajectory as a multiplier of beta_EP, e.g. 1.50 -> beta0=1.50*beta_EP")
    ap.add_argument("--phase-common-norm", action="store_true",
                    help="use one common normalization for all phase-portrait curves so growth/decay changes are visible")
    ap.add_argument("--fig4-beta0", type=float, default=None,
                    help="optional beta0 for Fig. 4; default is sub-EP/causal, use > beta_EP only for diagnostic override")
    ap.add_argument("--lambda-k", type=float, default=None,
                    help="override SOC lambda used in bath dispersions; default is manuscript lambda=k_max=pi/4; D_uv is separate")
    ap.add_argument("--clip-fig4-spectrum", action="store_true",
                    help="clip Fig. 4 impurity spectrum to positive part")
    args = ap.parse_args()
    if args.lambda_k is not None:
        P.lambda_k = float(args.lambda_k)
    P.fig1_left_scale = float(args.fig1_left_scale)
    P.fig1_right_scale = float(args.fig1_right_scale)
    P.phase_common_norm = bool(args.phase_common_norm)
    build_all_figures(Path(args.out), eta=args.eta, dos_betas=args.dos_betas,
                      dos_xlim=args.dos_xlim, waterfall_offset=args.waterfall_offset,
                      time_tmax=args.time_tmax, time_points=args.time_points,
                      fig4_beta0=args.fig4_beta0, clip_fig4=args.clip_fig4_spectrum,
                      which=args.which)


if __name__ == "__main__":
    main()

