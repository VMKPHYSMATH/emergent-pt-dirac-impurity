from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_control import dimer

root=Path(__file__).resolve().parent
rows=json.loads((root/'results/checks.json').read_text())+json.loads((root/'results/six_sites.json').read_text())
v=json.loads((root/'results/verification.json').read_text())
summary={
    'number_of_exact_cases':len(rows),
    'maximum_sumrule_error':max(r['sumrule_error'] for r in rows),
    'maximum_ground_residual':max(r['ground_residual'] for r in rows),
    'maximum_residue_coefficient_error':max(c['coefficient_error'] for r in rows for c in r['contours']),
    'maximum_third_pole_coefficient':max(c['third_pole_coefficient_norm'] for r in rows for c in r['contours']),
    'maximum_direct_schur_error':max(r['direct_schur_error'] for r in rows),
    'maximum_U0_onebody_error':max(r.get('U0_onebody_green_error',0) for r in rows),
    'maximum_nilpotent_direction_error':max(c['direction_error'] for r in rows for c in r['contours']),
    'maximum_invariant_observable_error':max(r['invariant_observable_error'] for r in rows),
    'all_selected_blocks_have_nullities_1_2':all(r['jordan_nullities']==[1,2] for r in rows),
    'all_loop_exchanges':all(r['exchange'] for r in v['loops']),
    'loops_winding':[r['winding'] for r in v['loops']],
    'two_sided_smallest_offset_relative_error':max(abs(complex(*r['ratio'])-1) for r in v['two_sided'] if r['relative_offset']==1e-6),
}
(root/'results/summary.json').write_text(json.dumps(summary,indent=2))
with (root/'table.csv').open('w') as f:
    names=['sites','bath_sites','U','nd','double_occupancy','P1','omega','weight','slope','B_coefficient']
    w=csv.DictWriter(f,fieldnames=names);w.writeheader()
    for r in rows:
        w.writerow({**{k:r[k] for k in names[:6]},'P1':r['P1'],**{k:r['addition'][k] for k in ['omega','weight','slope']},
                    'B_coefficient':r['addition']['predicted_B_coefficient']})

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,
                     'savefig.bbox':'tight'})
fig,ax=plt.subplots(2,2,figsize=(8.0,5.7),layout='constrained')
U=np.linspace(0,16,300)
ax[0,0].plot(U,[dimer(u)['B_coefficient'] for u in U],color='#176389',label='Dimer: analytic')
for n,marker,color in [(2,'o','#176389'),(4,'s','#b95b24'),(6,'^','#6264a7')]:
    rr=[r for r in rows if r['sites']==n]
    ax[0,0].plot([r['U'] for r in rr],[r['addition']['predicted_B_coefficient'] for r in rr],marker,
                 ms=5,mfc='white',mec=color,label=f'{n-1} bath site'+('s' if n>2 else '')+': exact resolvent')
ax[0,0].set(xlabel='$U$ (hopping $V=0.5$)',ylabel=r'Double-pole coefficient $Z_\ell m_\ell$',title='(a) Nonzero interacting residue')
ax[0,0].legend(fontsize=7,frameon=False)
for n,marker,color in [(2,'o','#176389'),(4,'s','#b95b24'),(6,'^','#6264a7')]:
    rr=[r for r in rows if r['sites']==n]
    ax[0,1].plot([r['U'] for r in rr],[r['P1'] for r in rr],marker+'-',color=color,label=f'{n-1} bath site'+('s' if n>2 else ''))
ax[0,1].set(xlabel='$U$',ylabel=r'$P_1=n_d-2D$',ylim=(.45,1.025),title='(b) Suppressed charge fluctuations')
ax[0,1].legend(fontsize=8,frameon=False)
for U,color in [(4.,'#176389'),(8.,'#b95b24')]:
    rr=[r for r in v['two_sided'] if r['U']==U]
    x=np.array([r['gamma']**2-.01 for r in rr]); y=np.array([complex(*r['squared_splitting']).real for r in rr])
    order=np.argsort(x)
    ax[1,0].plot(x[order]/1e-4,y[order]/1e-4,'o-',ms=3,color=color,label=f'$U={U:g}$')
ax[1,0].axhline(0,color='.6',lw=.6);ax[1,0].axvline(0,color='.6',lw=.6)
ax[1,0].set(xlabel=r'$(\gamma^2-\Delta^2)/10^{-4}$',ylabel=r'$(\omega_+-\omega_-)^2/10^{-4}$',title='(c) Two-sided square-root splitting')
ax[1,0].legend(fontsize=8,frameon=False)
loop=next(r for r in v['loops'] if r['U']==8. and r['points']==256)
p=np.array(loop['poles']);p=p[...,0]+1j*p[...,1]
center=next(r for r in rows if r['sites']==4 and r['U']==8.)['addition']['omega']
for j,c in [(0,'#176389'),(1,'#b95b24')]:
    ax[1,1].plot((p[:,j].real-center)*1e3,p[:,j].imag*1e3,color=c,label=f'Branch {j+1}')
    ax[1,1].plot((p[0,j].real-center)*1e3,p[0,j].imag*1e3,'o',ms=4,color=c)
ax[1,1].plot(0,0,'k*',ms=9,label='EP')
ax[1,1].set(xlabel=r'$10^3\operatorname{Re}(\omega-\omega_{\rm EP})$',ylabel=r'$10^3\operatorname{Im}\omega$',title='(d) Pole exchange, $U=8$')
ax[1,1].legend(fontsize=7,frameon=False,loc='center',bbox_to_anchor=(.5,.73))
for a in ax.ravel():a.grid(alpha=.15)
fig.savefig(root/'pt_control_checks.pdf');fig.savefig(root/'pt_control_checks.png');plt.close(fig)

table='\n'.join(f"| {r['bath_sites']} | {r['U']:g} | {r['nd']:.6f} | {r['double_occupancy']:.8f} | {r['addition']['omega']:.8f} | {r['addition']['predicted_B_coefficient']:.8f} |" for r in rows)
report=r'''# Exact interacting EP in a spin-symmetric Anderson control

28 October 2026. New calculation; the manuscript has not been edited.

## Result and scope

The proposed restricted route works. A half-filled Anderson dimer with a local
PT field has an impurity Green-function double pole with a strictly nonzero
coefficient for every finite repulsive U and nonzero hopping V. This is an
analytic finite-system result, independently checked by untruncated charge-sector
resolvents. Finite controls with three and five bath sites also pass the checks
listed below. No Jordan or PT form is fitted to the self-energy or Green function.

The model is H_A + d† N d, where H_A is a spin-rotation-invariant Anderson
Hamiltonian and N=Delta sigma_x+i gamma sigma_z. At gamma=Delta>0, N²=0.
The antiunitary symmetry is internal spin exchange followed by complex
conjugation; it is not an assertion about the physical spin time-reversal
operator of a generic SOC host. The bath in this control is spin independent.

This is the imposed non-Hermitian impurity-core control already discussed in
the Supplement, not the original passive full-k driven continuum model.
The state is the biorthogonal analytic continuation of an isolated Hermitian
singlet. Its scalar nd and D agree with those of that Hermitian reference.
No claim about a physical pumped steady-state occupation is made.

## What the proof adds

Complex-spin covariance gives G(z,N)=g0(z) I+g1(z) N at the null field, but
alignment alone does not guarantee a double pole. For an isolated spin-doublet
addition transition with weight Z and local-field slope m, differentiation of
the Hermitian Zeeman response gives the double-pole coefficient B=Z m N.
Consequently Z*m != 0 closes the missing amplitude condition. No arbitrary
nonzero-residue theorem for all baths or all continuum resonances is claimed.

This also supplies the nonlinear inverse-kernel certificate: at the pole,
K=G^-1 tends to -(m/Z)N, with rank one; det(K) has a double zero with
second derivative 2/Z² != 0. A double-pole amplitude and a coincident pair
of scalar eigenvalues are not being substituted for this kernel condition.

For the half-filled dimer eps_d=-U/2, bath energy zero, and hopping V>0, set
A=sqrt(U²/16+V²), C=sqrt(U²/16+4V²). The lower and upper positive transition
energies are w_l=C-A and w_h=C+A. The exact formulas are

    Z_l=(w_l²-9V²)/[2(w_l²-w_h²)]
    m_l=(1+U/(4A))/2
    D=(1-U/sqrt(U²+64V²))/4
    B_l=Z_l*m_l*N

For every finite U>=0, 0<w_l<=V and w_h>w_l, so Z_l>0 and m_l>=1/2.
Thus the selected pole is genuinely defective at gamma=Delta. The symmetry
pins this control condition, while U changes the pole energy and its residue.
At fixed V and large U, w_l~6V²/U and Z_l*m_l~18V²/U²: the amplitude is not
bounded away from zero in the infinite-U limit.

## Exact numerical results

All entries use hopping V=0.5, Delta=gamma=0.1 and eps_d=-U/2; the chemical
potential is zero. Bath sites are spinful sites, not single fermion modes.
The tabulated w is the lowest positive addition pole, and B=(Z*m)N.

| Bath sites | U | nd | D | w | Z*m |
|---:|---:|---:|---:|---:|---:|
@@TABLE@@

This is finite-system evidence of suppressed charge fluctuations and a local
moment amplitude. It does not establish continuum Kondo screening or an
unscreened local-moment fixed point. The change of w and residue with bath size
is not a continuum convergence certificate.

## Verification

- Exact finite charge sectors; no NRG truncation, mean field or tail fitting.
- Every selected addition doublet has algebraic multiplicity 2; nullities of
  H-E and (H-E)² are 1 and 2 respectively.
- Direct contour integration extracts a nonzero double-pole coefficient, not
  merely its normalized direction. Halving the contour radius and doubling
  its mesh reproduces the coefficient; the third-order coefficient vanishes
  within numerical precision.
- Analytic dimer pole, weight, field slope, double occupancy and scalar Green
  function agree with the independent Fock-space calculation.
- Independent direct and Schur resolvents agree. At U=0 the many-body response
  agrees with the one-body inverse. The equal-time anticommutator sum rule holds.
- At U=4 and 8, four total sites, the squared splitting approaches
  4*m²*(Delta²-gamma²) from both sides of the EP.
- Two control loops (radii 1e-4 and 5e-5, meshes 128 and 256) at each U yield
  squared-splitting winding -1 and pole exchange. The second loop control is
  a real impurity sigma_z field, intentionally leaving the PT line.
- A real bath sigma_x field of 0.005 preserves the internal PT symmetry but
  splits the poles at the old EP condition. This shows the additional role of
  spin covariance; PT alone is insufficient to pin the EP.

Machine-readable maximum residuals are in results/summary.json.

## Self-energy interpretation

At the null field the interacting self-energy has the form Sigma0(z) I+
Sigma1(z) N. Sigma1 is not generally zero. From G=g0 I+g1 N one finds
Sigma1=g1/g0²-1. At the selected pole its limiting value is m/Z-1.
For the dimer the scalar self-energy is U/2+(U²/4)z/(z²-9V²).
Therefore this calculation includes a dynamical interaction self-energy and
a nontrivial local spin vertex. Freezing the self-energy to a scalar function
would give B=Z² N instead of the correct Z*m*N.

## Manuscript use

This could strengthen the existing structural-control subsection from
"nilpotent direction retained" to "nonzero double pole established in a
specified finite interacting model." It should be presented as a controlled
comparison with the full driven problem, not as evidence that the original
occupied full-k EP has converged. It does not justify replacing the original
spin-dependent bath by a symmetric fit and calling that a solution.

Interaction-stable symmetry-protected EPs already exist in the literature:
R. Schaefer, J. C. Budich and D. J. Luitz, Physical Review Research 4, 033181
(2022), https://doi.org/10.1103/PhysRevResearch.4.033181. Novelty of this precise
impurity residue formulation needs a dedicated comparison before claiming it.

## Reproduction

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python pt_control.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python pt_control.py --sites 6 --U 4 8 --output results/six_sites.json
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python verify_control.py
    python build_report.py
    pdflatex -interaction=nonstopmode -halt-on-error PT_Interacting_Control_Note.tex

Dependencies: Python 3, NumPy, SciPy, Matplotlib; LaTeX for the optional note.
All energies use a common arbitrary unit, with hopping set to 0.5. The five-bath-site checks use
924 states in the half-filled sector and 792 states in each response sector.
'''.replace('@@TABLE@@',table)
(root/'REPORT.md').write_text(report)
print(json.dumps(summary,indent=2))
