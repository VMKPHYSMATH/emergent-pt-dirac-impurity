import os
"""U=0 impurity density matrix on the archived z-shifted 12-site block-Wilson chains (read-only inputs),
compared with the continuum one, and the first-order Fock slope each implies."""
import csv, json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sbmf_fullk import EP0
from sbmf_fullk2 import density_matrix_T0, solve_ep_extra, P0
base = (Path(os.environ["EPT_REPO"]) / "reproducibility/full_k_finite_u/finiteU_fullk_reproducibility/preflight_lowbands_z4" if "EPT_REPO" in os.environ else Path(__file__).resolve().parents[3] / "reproducibility/filling/reference/submitted_chains")
man = json.loads((base / "z_chain_manifest.json").read_text())

def read_chain(csv_path, summary_path):
    onsite, hopping = [], []
    with open(csv_path) as h:
        for row in csv.DictReader(h):
            m = lambda pre: np.array([complex(float(row[f"{pre}{a}{b}_re"]), float(row[f"{pre}{a}{b}_im"]))
                                      for a, b in ((1, 1), (1, 2), (2, 1), (2, 2))]).reshape(2, 2)
            onsite.append(m("a")); hopping.append(m("b"))
    hopping = hopping[:-1]
    enc = json.loads(Path(summary_path).read_text())["block_chain"]["impurity_to_chain_matrix"]
    return onsite, hopping, np.array([[complex(v[0], v[1]) for v in r] for r in enc])

def one_body(onsite, hopping, imp, eps, sites):
    n = 2 * (sites + 1); M = np.zeros((n, n), complex)
    M[0:2, 0:2] = eps * np.eye(2); M[0:2, 2:4] = imp; M[2:4, 0:2] = imp.conj().T
    for s in range(sites):
        a = 2 + 2 * s; M[a:a+2, a:a+2] = onsite[s]
        if s + 1 < sites:
            M[a:a+2, a+2:a+4] = hopping[s]; M[a+2:a+4, a:a+2] = hopping[s].conj().T
    return 0.5 * (M + M.conj().T)

eps = EP0["eps"]
rhos = []
for c in man["chains"]:
    on, hop, imp = read_chain(base / c["chain"], base / c["summary"])
    for sites in (10, 12):
        E, V = np.linalg.eigh(one_body(on, hop, imp, eps, sites))
        occ = E < 0
        rho = (V[0:2, occ] @ V[0:2, occ].conj().T).real
        if sites == 12:
            rhos.append(rho)
        print("z=%.3f sites=%d: occupied one-body orbitals=%d  lowest E=%s  rho_imp=%s Tr=%.5f"
              % (c["z_shift"], sites, occ.sum(), np.round(E[:3], 5), np.round(rho, 5).tolist(), np.trace(rho)))
rbar = np.mean(rhos, axis=0)
rc = density_matrix_T0(eps, EP0["VO"], P0)
print("z-averaged chain rho =", np.round(rbar, 6).tolist(), " continuum rho =", np.round(rc, 6).tolist())
x0 = np.array([EP0["z"].real, EP0["z"].imag, 0.10])
for lab, rho in (("continuum rho", rc), ("chain rho (z-avg, 12 sites)", rbar)):
    o = []
    for U in (1e-6, -1e-6):
        ok, x, e = solve_ep_extra(EP0["VO"], U * (np.trace(rho) * np.eye(2) - rho), x0, unknown="W"); o.append((e, x[2]))
    print("first-order Fock with %-28s: d eps_d/dU=%+.4f  dW/dU=%+.4f" % (lab, (o[0][0]-o[1][0])/2e-6, (o[0][1]-o[1][1])/2e-6))
print("converged block-Wilson centers (quadratic fit to U=0.001..0.0025): d eps_d/dU=-0.3470  dW/dU=-4.459")
