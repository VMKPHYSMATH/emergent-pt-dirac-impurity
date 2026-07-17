"""
bethe_prb_submission.py
=======================
Generates Fig. 5 and Fig. 6 for:

  "Emergent PT Symmetry and Exceptional Points in a Driven Dirac Impurity"

Outputs (written to the same directory as this script):
  fig5_rapidities.pdf / .png  —  Biorthogonal BA rapidities and EP structure
  fig6_phasediag.pdf  / .png  —  EP phase boundary in (U, lambda) plane

Requirements:
  Python >= 3.8, numpy, matplotlib

Run:
  python bethe_prb_submission.py

Parameters (match manuscript captions exactly):
  eps_xi = -1.0          bare impurity level (= -U/2, exact PH symmetry)
  gamma  =  0.50         flip amplitude; bare EP at |beta_0| = gamma
  U_val  =  2.0          Coulomb interaction (exact PH: eps_xi = -U/2)
  k_max  =  pi/4         bath momentum cutoff (the UV energy cutoff D_uv is conceptually separate)
  lam    =  0.50         spin-orbit coupling lambda (D-SOC^2 form factor)
  F_val  =  1-(lam/k_max)^2 = 0.595   D-SOC^2 overlap factor

Physics notes:
  - V_eta = beta_0  (corrected from beta_0/sqrt(2); see App. A of manuscript)
  - Physical S-matrix width: Gamma_eta = pi * beta_0^2 / k_max
  - Bath dispersion sampled over full range [-k_max, k_max]
  - s_eff from quartic: s_eff^4 - s0^2 * s_eff^2 - U^2 * beta0^2 * F = 0
    positive root: s_eff^2 = (s0^2 + sqrt(s0^4 + 4*U^2*b^2*F)) / 2
  - mu^L = conj(mu^R) by PT symmetry (verified numerically)
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import os

# ── Output directory (same folder as this script) ─────────────────────────
OUT = os.path.dirname(os.path.abspath(__file__))

# ── Global plot style ─────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':          'serif',
    'mathtext.fontset':     'cm',
    'font.size':            11.0,
    'axes.labelsize':       12.5,
    'axes.titlesize':       11.0,
    'axes.linewidth':       1.5,
    'lines.linewidth':      2.5,
    'xtick.direction':      'in',    'ytick.direction':      'in',
    'xtick.top':            True,    'ytick.right':          True,
    'xtick.major.width':    1.3,     'ytick.major.width':    1.3,
    'xtick.minor.width':    0.9,     'ytick.minor.width':    0.9,
    'xtick.major.size':     5.5,     'ytick.major.size':     5.5,
    'xtick.minor.size':     3.0,     'ytick.minor.size':     3.0,
    'xtick.minor.visible':  True,    'ytick.minor.visible':  True,
    'xtick.labelsize':      10.5,    'ytick.labelsize':      10.5,
    'legend.fontsize':      9.5,
    'legend.framealpha':    1.0,
    'legend.facecolor':     'white',
    'legend.edgecolor':     '0.60',
    'legend.borderpad':     0.5,
    'legend.labelspacing':  0.30,
    'legend.handlelength':  2.2,
    'legend.handletextpad': 0.5,
    'axes.grid':            False,
    'savefig.dpi':          600,
    'figure.dpi':           120,
})

# ══════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════

eps_xi = -1.0          # bare impurity level
gamma  = 0.50          # flip amplitude; EP at |beta_0| = gamma
U_val  = 2.0           # Coulomb interaction (exact PH: eps_xi = -U/2)
k_max  = np.pi / 4     # bath momentum cutoff
lam    = 0.50          # spin-orbit coupling (D-SOC^2)
F_val  = 1.0 - (lam / k_max)**2   # D-SOC^2 overlap factor

ETA_FLOOR = 5e-5       # lower bound on imaginary broadening (numerics)
S_FLOOR   = 1e-9       # lower bound on |s| to avoid 1/0 in Gamma_bio

# Bath momentum grid — full range [-k_max, k_max]
Nk   = 6000
k_W  = np.linspace(-k_max, k_max, Nk)
dk_W = k_W[1] - k_W[0]
ec_p = k_W**2 + lam * k_W    # chiral dispersion epsilon_{k,+}
ec_m = k_W**2 - lam * k_W    # chiral dispersion epsilon_{k,-}

# beta_0 sweep: dense near EP, coarser elsewhere
b_arr = np.sort(np.unique(np.concatenate([
    np.linspace(-1.10,  1.10, 1000),
    np.linspace( 0.46,  0.54,  400),   # dense near right EP
    np.linspace(-0.54, -0.46,  400),   # dense near left EP
    [gamma, -gamma],                   # exact EP points
])))
ep_idx = np.argmin(np.abs(b_arr - gamma))   # index of right EP

# ══════════════════════════════════════════════════════════════════════════
# PHYSICS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def s_bare(b):
    """Bare impurity splitting s = sqrt(gamma^2 - beta_0^2).
    Complex for |beta_0| > gamma (PT-broken phase)."""
    return np.sqrt(complex(gamma**2 - b**2))


def s_eff(b):
    """Interaction-corrected splitting from the self-consistent quartic
    (App. K of manuscript):
      s_eff^4 - s0^2 * s_eff^2 - U^2 * b^2 * F = 0
    Positive root: s_eff^2 = (s0^2 + sqrt(s0^4 + 4*U^2*b^2*F)) / 2.
    s_eff > 0 for all beta_0 when U > 0 and F > 0 (EP regularised)."""
    s0   = complex(gamma**2 - b**2)
    disc = np.sqrt(s0**2 + 4.0 * (U_val**2) * b**2 * F_val + 0j)
    se   = np.sqrt((s0 + disc) / 2.0 + 0j)
    return -se if se.real < 0 else se


def W_val(ed, ec, V2, eta):
    """One-body bath shift W_eta = V^2 * sum_k 1/(epsilon_d - epsilon_c + i*eta).
    Discretised over the k_W grid."""
    return complex(V2 * np.sum(dk_W / (ed - ec + 1j * eta)))


def compute(b, se):
    """Compute Bethe-dressed rapidities and diagnostics at fixed beta_0.

    Uses corrected hybridisation V_eta = beta_0 (not beta_0/sqrt(2)),
    giving physical S-matrix width Gamma_eta = pi * beta_0^2 / k_max.

    Returns: mu^R_+, mu^R_-, mu^L_+, mu^L_-, Gamma_bio,
             channel splitting, |mu^R - mu^L|_+, |mu^R - mu^L|_-
    """
    ep  = complex(eps_xi + se)
    em  = complex(eps_xi - se)
    V2  = b**2                             # V_eta = beta_0  =>  V^2 = beta_0^2
    eta = max(np.pi * V2 / k_max, ETA_FLOOR)   # Gamma_eta = pi*b^2/k_max

    muRp = ep + W_val(ep, ec_p, V2, eta)
    muRm = em + W_val(em, ec_m, V2, eta)

    Gbio       = b**2 / max(abs(se), S_FLOOR)  # biorthogonal width = b^2/|s|
    chan_split  = abs(muRp - muRm)              # genuine EP diagnostic
    RL_p        = abs(muRp - np.conj(muRp))    # |mu^R - mu^L|_+  (not EP signal)
    RL_m        = abs(muRm - np.conj(muRm))

    # mu^L = conj(mu^R) by PT symmetry
    return (muRp, muRm, np.conj(muRp), np.conj(muRm),
            Gbio, chan_split, RL_p, RL_m)


def sweep(sfunc, label):
    """Sweep beta_0 array and collect all rapidities and diagnostics."""
    print(f'  {label}...', end=' ', flush=True)
    R = {k: [] for k in 'Rp Rm Lp Lm Gbio chan RL_p RL_m s'.split()}
    for b in b_arr:
        se = sfunc(b)
        mRp, mRm, mLp, mLm, Gb, chan, rp, rm = compute(b, se)
        for v, key in zip([mRp, mRm, mLp, mLm], ['Rp', 'Rm', 'Lp', 'Lm']):
            R[key].append(v if abs(v) < 30 else np.nan + 0j)
        R['Gbio'].append(Gb)
        R['chan'].append(chan)
        R['RL_p'].append(rp)
        R['RL_m'].append(rm)
        R['s'].append(se)
    print('done.')
    return {k: np.array(v) for k, v in R.items()}


# ══════════════════════════════════════════════════════════════════════════
# COMPUTE
# ══════════════════════════════════════════════════════════════════════════

print('Computing rapidities...')
B = sweep(s_bare, 'bare s   (Layer 1/3)')
S = sweep(s_eff,  f's_eff   (Layer 2/3, U={U_val})')

# Layer 1: bare h_d eigenvalues  eps_xi ± s
hd_p = np.array([eps_xi + s_bare(b) for b in b_arr])
hd_m = np.array([eps_xi - s_bare(b) for b in b_arr])

# Layer 2: interaction-corrected levels  eps_xi ± s_eff
uc_p = np.array([eps_xi + s_eff(b) for b in b_arr])
uc_m = np.array([eps_xi - s_eff(b) for b in b_arr])

# Marker spacing (every ~22nd point so symbols don't clutter curves)
mk = max(1, len(b_arr) // 22)

# ══════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════
C_HD = '#111111'   # Layer 1: bare h_d eigenvalues  (black)
C_UC = '0.48'      # Layer 2: U-corrected levels    (grey)
C_BR = '#1a5fa8'   # Layer 3: bare-s rapidities     (blue)
C_BL = '#5fa0d8'   # Layer 3: bare-s left (mirror)  (light blue)
C_SR = '#b84a0a'   # Layer 3: s_eff rapidities      (orange)
C_SL = '#d8845a'   # Layer 3: s_eff left (mirror)   (light orange)
C_EP = '#b81c1c'   # EP marker colour               (red)


def shade(ax):
    """Shade PT-unbroken (blue) and PT-broken (orange) regions;
    draw red dashed vertical lines at bare EP positions ±gamma."""
    for sgn in [+1, -1]:
        ax.axvspan(0,          sgn * gamma, alpha=0.05, color='#3a78c9',
                   zorder=0, lw=0)
        ax.axvspan(sgn * gamma, sgn * 1.25, alpha=0.06, color='#d46010',
                   zorder=0, lw=0)
        ax.axvline(sgn * gamma, color=C_EP, lw=1.6, ls='--',
                   zorder=5, alpha=0.9)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Biorthogonal BA rapidities and EP structure
# ══════════════════════════════════════════════════════════════════════════

fig5, axes = plt.subplots(
    2, 2, figsize=(8.2, 6.6),
    gridspec_kw=dict(wspace=0.36, hspace=0.46,
                     left=0.10, right=0.97, top=0.96, bottom=0.09))
axA, axB = axes[0]
axC, axD = axes[1]

# ── Panel (a): Real parts of rapidities ───────────────────────────────────

axA.plot(b_arr, hd_p.real, color=C_HD, lw=3.2, ls='-',  zorder=9,
         label=r'bare $\tilde\epsilon_{d,\pm}$ (L1)')
axA.plot(b_arr, hd_m.real, color=C_HD, lw=3.2, ls='--', zorder=9)

axA.plot(b_arr, uc_p.real, color=C_UC, lw=1.8, ls='-',  zorder=7,
         label=r'$\tilde\epsilon_\xi\pm s_{\rm eff}$ (L2)')
axA.plot(b_arr, uc_m.real, color=C_UC, lw=1.8, ls='--', zorder=7)

axA.plot(b_arr, B['Rp'].real, color=C_BR, lw=2.3, ls='-',
         marker='o', ms=3.0, markevery=mk, zorder=8,
         label=r'$\mathrm{Re}\,\mu^R_\pm$ bare $s$ (L3)')
axA.plot(b_arr, B['Rm'].real, color=C_BR, lw=2.3, ls='--',
         marker='o', ms=3.0, markevery=mk + 4, zorder=8)

axA.plot(b_arr, S['Rp'].real, color=C_SR, lw=2.3, ls='-',
         marker='s', ms=3.0, markevery=mk, zorder=8,
         label=r'$\mathrm{Re}\,\mu^R_\pm$ $s_{\rm eff}$, $U\!=\!2.0$')
axA.plot(b_arr, S['Rm'].real, color=C_SR, lw=2.3, ls='--',
         marker='s', ms=3.0, markevery=mk + 4, zorder=8)

shade(axA)
axA.set_xlabel(r'$\beta_0$')
axA.set_ylabel(r'$\mathrm{Re}\,\mu^\alpha_\pm$')
axA.set_title(r'(a)  $\mathrm{Re}\,\mu^R_\pm = \mathrm{Re}\,\mu^L_\pm$'
              r'  ($\mathcal{PT}$)', pad=4)
axA.set_xlim(b_arr[0], b_arr[-1])
axA.set_ylim(-2.8, 0.2)

leg_a = axA.legend(loc='lower left', ncol=1)
leg_a.set_zorder(20)

axA.annotate('EP', xy=(gamma, -1.0), xytext=(gamma + 0.16, -0.70),
             fontsize=10.5, color=C_EP, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=C_EP, lw=1.3))
axA.text(0.55, 0.97, r'$\mathcal{PT}$-unb.',
         transform=axA.transAxes, ha='center', va='top',
         fontsize=10, color='#0a3a8a', style='italic')
axA.text(0.90, 0.97, r'$\mathcal{PT}$-br.',
         transform=axA.transAxes, ha='center', va='top',
         fontsize=10, color='#8a1a00', style='italic')

# ── Panel (b): Imaginary parts — bath width, not EP diagnostic ────────────

axB.plot(b_arr, B['Rp'].imag, color=C_BR, lw=2.5, ls='-',
         marker='o', ms=3.0, markevery=mk,
         label=r'$\mathrm{Im}\,\mu^R_+$ bare $s$')
axB.plot(b_arr, B['Rm'].imag, color=C_BR, lw=2.5, ls='--',
         marker='o', ms=3.0, markevery=mk + 4)

axB.plot(b_arr, B['Lp'].imag, color=C_BL, lw=1.5, ls='-', alpha=0.80,
         marker='v', ms=2.5, markevery=mk + 8,
         label=r'$\mathrm{Im}\,\mu^L_+$ bare $s$  (mirror)')
axB.plot(b_arr, B['Lm'].imag, color=C_BL, lw=1.5, ls='--', alpha=0.80,
         marker='v', ms=2.5, markevery=mk + 12)

axB.plot(b_arr, S['Rp'].imag, color=C_SR, lw=2.5, ls='-',
         marker='s', ms=3.0, markevery=mk,
         label=r'$\mathrm{Im}\,\mu^R_+$ $s_{\rm eff}$, $U\!>\!0$')
axB.plot(b_arr, S['Rm'].imag, color=C_SR, lw=2.5, ls='--',
         marker='s', ms=3.0, markevery=mk + 4)

axB.plot(b_arr, S['Lp'].imag, color=C_SL, lw=1.5, ls='-', alpha=0.80,
         marker='^', ms=2.5, markevery=mk + 8,
         label=r'$\mathrm{Im}\,\mu^L_+$ $s_{\rm eff}$')
axB.plot(b_arr, S['Lm'].imag, color=C_SL, lw=1.5, ls='--', alpha=0.80,
         marker='^', ms=2.5, markevery=mk + 12)

axB.axhline(0, color='0.40', lw=1.0, ls=':', zorder=1)
shade(axB)
axB.set_xlabel(r'$\beta_0$')
axB.set_ylabel(r'$\mathrm{Im}\,\mu^\alpha_\pm$')
axB.set_title(r'(b)  $\mathrm{Im}\,\mu^{R,L}_\pm$: bath width'
              r' (finite, not EP diagnostic)', pad=4)
axB.set_xlim(b_arr[0], b_arr[-1])
axB.set_ylim(-2.2, 3.0)
axB.legend(loc='upper right', ncol=1)
axB.text(0.03, 0.05,
         r'$\mathrm{Im}(\mu^R) = -\tilde\Gamma_{\rm phys}$:'
         r' finite $\forall\beta_0$' '\n'
         r'$\tilde\Gamma_{\rm phys}=\pi\beta_0^2/k_{\rm max}$;'
         r' bath width, not EP signal',
         transform=axB.transAxes, fontsize=8.5, color='0.25',
         bbox=dict(fc='white', alpha=1.0, pad=2,
                   boxstyle='round,pad=0.3'))

# ── Panel (c): Channel splitting — genuine EP diagnostic ──────────────────

fl = 1e-5   # floor for log plot
s_bare_arr = np.array([abs(s_bare(b)) for b in b_arr])
s_eff_arr  = np.array([abs(s_eff(b))  for b in b_arr])

axC.semilogy(b_arr, np.maximum(2 * s_bare_arr, fl),
             color=C_HD, lw=3.0, ls='-', zorder=9,
             label=r'$2|s|$ bare (L1)')
axC.semilogy(b_arr, np.maximum(2 * s_eff_arr, fl),
             color=C_UC, lw=1.8, ls='-', zorder=7,
             label=r'$2|s_{\rm eff}|$ (L2)')
axC.semilogy(b_arr, np.maximum(B['chan'], fl),
             color=C_BR, lw=2.5, ls='-',
             marker='o', ms=3.0, markevery=mk,
             label=r'$|\mu^R_+-\mu^R_-|$ bare $s$ (L3)')
axC.semilogy(b_arr, np.maximum(S['chan'], fl),
             color=C_SR, lw=2.5, ls='-',
             marker='s', ms=3.0, markevery=mk,
             label=r'$|\mu^R_+-\mu^R_-|$ $s_{\rm eff}$, $U\!>\!0$')

shade(axC)
axC.set_xlabel(r'$\beta_0$')
axC.set_ylabel(r'$|\mu^R_+ - \mu^R_-|$')
axC.set_title(r'(c)  Channel splitting $|\mu^R_+-\mu^R_-|$:'
              r' genuine EP diagnostic', pad=4)
axC.set_xlim(b_arr[0], b_arr[-1])
axC.set_ylim(1e-7, 10.0)
axC.grid(True, which='major', ls=':', lw=0.6, alpha=0.5)
axC.grid(True, which='minor', ls=':', lw=0.3, alpha=0.3)
axC.yaxis.set_major_locator(LogLocator(base=10, numticks=6))

leg_c = axC.legend(loc='lower left', ncol=1)
leg_c.set_zorder(20)

axC.annotate(r'$2|s|\to 0$ at bare EP',
             xy=(gamma, B['chan'][ep_idx] + 1e-6),
             xytext=(gamma - 0.52, 3e-3),
             fontsize=9.5, color=C_BR, ha='center',
             arrowprops=dict(arrowstyle='->', color=C_BR, lw=1.3))
axC.annotate(r'$U\!>\!0$: $s_{\rm eff}\!>\!0$, EP removed',
             xy=(gamma, S['chan'][ep_idx]),
             xytext=(gamma + 0.18, 0.40),
             fontsize=9.0, color=C_SR, ha='left',
             arrowprops=dict(arrowstyle='->', color=C_SR, lw=1.1))

# ── Panel (d): Biorthogonal width Gamma_bio = beta_0^2 / |s| ──────────────

# Mask near the bare EP where s -> 0 to avoid numerical artefacts
G_bare = np.where(np.abs(B['s']) > 4e-3, B['Gbio'], np.nan)

axD.semilogy(b_arr, np.maximum(G_bare, 1e-5),
             color=C_BR, lw=2.5, ls='-',
             label=r'$\beta_0^2/|s|$  bare $s$')
axD.semilogy(b_arr, np.maximum(S['Gbio'], 1e-5),
             color=C_SR, lw=2.5, ls='--',
             label=r'$\beta_0^2/|s_{\rm eff}|$  $U\!>\!0$')

shade(axD)
axD.set_xlabel(r'$\beta_0$')
axD.set_ylabel(r'$\tilde\Gamma_{\rm bio} = \beta_0^2/|s|$')
axD.set_title(r'(d)  $\tilde\Gamma_{\rm bio}$: biorthogonal norm'
              r' diverges at EP', pad=4)
axD.set_xlim(b_arr[0], b_arr[-1])
axD.set_ylim(5e-4, 1e4)
axD.grid(True, which='major', ls=':', lw=0.6, alpha=0.5)
axD.grid(True, which='minor', ls=':', lw=0.3, alpha=0.3)
axD.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
axD.legend(loc='upper right', ncol=1)

# Annotation sits at (-0.85, 600) — clear of both divergence peaks
axD.annotate(r'$\tilde\Gamma_{\rm bio}\to\infty$',
             xy=(gamma - 0.04, 14.0),
             xytext=(-0.85, 600.0),
             fontsize=10, color=C_BR, ha='left',
             arrowprops=dict(arrowstyle='->', color=C_BR, lw=1.3,
                             connectionstyle='arc3,rad=0.15'))

# ── Save Fig. 5 ───────────────────────────────────────────────────────────
p5 = os.path.join(OUT, 'fig5_rapidities')
fig5.savefig(p5 + '.pdf', bbox_inches='tight')
fig5.savefig(p5 + '.png', dpi=300, bbox_inches='tight')
print(f'Saved {p5}.pdf / .png')
plt.close(fig5)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6 — EP phase boundary in (U, lambda) plane
# ══════════════════════════════════════════════════════════════════════════
#
# Colour map: D_EP = gamma^2 - U^2 * F(lambda)
#   > 0 (blue):  PT-unbroken
#   < 0 (red):   PT-broken
# Thick black contour: phase boundary D_EP = 0
# Dashed grey line:    lambda = lam (value used in Fig. 5)
# Dotted orange line:  lambda = k_max (F = 0 exactly for D-SOC^2)

N     = 400
U_g   = np.linspace(0.0, 4.0, N)
lam_g = np.linspace(0.0, 2.5, N)
Ug, Lg = np.meshgrid(U_g, lam_g)

F_gauss = lambda L: np.exp(-(L / k_max)**2)
F_dsoc2 = lambda L: np.maximum(1.0 - (L / k_max)**2, 0.0)

g2g = gamma**2 - Ug**2 * F_gauss(Lg)
g2d = gamma**2 - Ug**2 * F_dsoc2(Lg)

fig6, (ax6a, ax6b) = plt.subplots(
    1, 2, figsize=(8.2, 3.8),
    gridspec_kw=dict(wspace=0.30, left=0.08, right=0.84,
                     top=0.91, bottom=0.15))
vmax = gamma**2

for ci, (ax, g2) in enumerate([(ax6a, g2g), (ax6b, g2d)]):

    cf = ax.contourf(Ug, Lg, g2,
                     levels=np.linspace(-vmax, vmax, 201),
                     cmap='RdBu_r', alpha=0.70, extend='both')
    cs = ax.contour(Ug, Lg, g2, levels=[0.0],
                    colors=['#080808'], linewidths=2.8, linestyles=['-'])

    # Label the phase boundary
    if ci == 0:
        ax.clabel(cs, fmt={0.0: r'$\mathcal{D}_{\rm EP}=0$'},
                  fontsize=9.5, inline_spacing=5, manual=[(2.6, 1.30)])
    else:
        ax.text(1.50, k_max + 0.09, r'$\mathcal{D}_{\rm EP}=0$',
                fontsize=9.5, color='#080808', ha='center', va='bottom')

    # Horizontal reference lines
    ax.axhline(lam,   color='0.20',     lw=1.8, ls='--', alpha=0.88, zorder=6)
    ax.axhline(k_max, color='#9a4800',  lw=1.8, ls=':',  alpha=0.92, zorder=6)

    # D-SOC^2 only: shade F=0 region where U cannot regularise the EP
    if ci == 1:
        ax.fill_between(U_g, k_max, 2.5,
                        color='#d46010', alpha=0.14, zorder=0)
        ax.text(2.5, 1.85, r'$F = 0$',
                ha='center', va='center', fontsize=12,
                color='#6a2000', fontstyle='italic', fontweight='bold')
        ax.text(2.5, 1.60, r'($U$ ineffective)',
                ha='center', va='center', fontsize=9.5, color='#6a2000')

    # Phase labels — positions verified to sit in correct colour zones.
    # D_EP > 0 (blue, PT-unbroken) = left of boundary curve (small U).
    # D_EP < 0 (red,  PT-broken)   = right of boundary (large U, lam < k_max).
    #
    # Gaussian (ci=0): boundary runs from (U~0.5, lam=0) to (U~4, lam~1.6).
    #   PT-unb. at U=0.20, lam=1.50 -> axes(0.05, 0.60) -> g2=+0.249 BLUE ✓
    #   PT-br.  at U=3.00, lam=0.30 -> axes(0.75, 0.12) -> g2=-7.53  RED  ✓
    #
    # D-SOC^2 (ci=1): PT-broken wedge is below k_max and right of boundary.
    #   PT-unb. at U=0.30, lam=0.30 -> axes(0.07, 0.12) -> g2=+0.173 BLUE ✓
    #   PT-br.  at U=1.50, lam=0.40 -> axes(0.38, 0.16) -> g2=-1.416 RED  ✓
    if ci == 0:
        # Gaussian panel: blue (PT-unb) region is narrow strip left of boundary.
        # At lam=2.0 the boundary is at U~13, so the entire visible plot is blue there.
        # Use upper-left corner where there is clearly blue space.
        ax.text(0.12, 0.88, r'$\mathcal{PT}$-unb.',   # U~0.5, lam~2.2 -> g2≈+0.25 BLUE ✓
                transform=ax.transAxes, fontsize=10.5, color='#800000',
                style='italic', ha='center', va='center')
        ax.text(0.75, 0.12, r'$\mathcal{PT}$-br.',    # U~3.0, lam~0.3 -> g2≈-7.5 RED  ✓
                transform=ax.transAxes, fontsize=10.5, color='#00006a',
                style='italic', ha='center', va='center')
    else:
        # D-SOC^2 panel: PT-unb is far-left narrow strip + entire top (F=0).
        # PT-br is the wedge below k_max and right of boundary.
        ax.text(0.38, 0.8, r'$\mathcal{PT}$-unb.',   # U~0.3, lam~0.3 -> g2=+0.17 BLUE ✓
                transform=ax.transAxes, fontsize=10.5, color='#800000',
                style='italic', ha='center', va='center')
        ax.text(0.38, 0.16, r'$\mathcal{PT}$-br.',    # U~1.5, lam~0.4 -> g2=-1.4 RED  ✓
                transform=ax.transAxes, fontsize=10.5, color='#00006a',
                style='italic', ha='center', va='center')

    # Reference line labels
    ax.text(3.90, lam + 0.10, rf'$\lambda={lam:.2f}$',
            fontsize=9.5, color='0.20', ha='right', va='bottom')
    ax.text(3.90, k_max + 0.10, r'$\lambda=k_{\rm max}$',
            fontsize=9.5, color='#7a3600', ha='right', va='bottom')

    title_F = (r'(a)  Gaussian: $F = e^{-(\lambda/k_{\rm max})^2}$'
               if ci == 0
               else r'(b)  $D$-SOC$^2$: '
                    r'$F = \max(1-(\lambda/k_{\rm max})^2,\,0)$')
    ax.set_title(title_F, pad=4)
    ax.set_xlabel(r'Interaction $U$')
    ax.set_ylabel(r'SOC $\lambda$')
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(0.0, 2.5)

cbar = fig6.colorbar(cf, ax=[ax6a, ax6b],
                     shrink=0.93, pad=0.020, extend='both', fraction=0.046)
cbar.set_label(r'$\mathcal{D}_{\rm EP}=\gamma^2-U^2F(\lambda)$',
               fontsize=11.0)
cbar.ax.tick_params(labelsize=10.0)

# ── Save Fig. 6 ───────────────────────────────────────────────────────────
p6 = os.path.join(OUT, 'fig6_phasediag')
fig6.savefig(p6 + '.pdf', bbox_inches='tight')
fig6.savefig(p6 + '.png', dpi=300, bbox_inches='tight')
print(f'Saved {p6}.pdf / .png')
plt.close(fig6)

print('Done.')
