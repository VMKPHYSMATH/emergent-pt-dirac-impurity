#!/usr/bin/env python3
"""Physical-sheet checks at and away from an EP; finite grid, not a proof."""
import json
import numpy as np
import two_sided_ep as t

data=json.loads((t.ROOT/'results/cert_occ8.json').read_text())
v=np.asarray(data['solution']['center'])
out={'scope':'sampled physical-sheet passivity checks, not a whole-axis or continuum theorem','cases':[]}
omega=np.unique(np.r_[np.linspace(-.025,.025,65),np.linspace(-.3,.3,25)])
for label,U,eps,W in [('EP',.0005,v[2],v[3]),
                       ('minus_level_offset',.0005,v[2]-1e-4,v[3]),
                       ('plus_level_offset',.0005,v[2]+1e-4,v[3]),
                       ('occupied_strong_U_reference',.2,.0151703497834-.1,.1)]:
    eng=t.Engine(U,.012,8)
    states=eng.states(eps,W)
    row={'label':label,'U':U,'epsilon_d':eps,'W':W,'filling':eng.filling(eps,W),'grids':[]}
    for eta in [1e-5,3e-4,3e-3]:
        records=[]
        for w in omega:
            z=complex(w,eta)
            b=t.c.fullk.analytic_physical_self_energy(z+eng.mu,eng.cfg['model']['VO'],t.c.parameters(eng.cfg['model'],W))
            s=t.c.interaction_sigma(states,z)
            G=np.linalg.inv((z+eng.mu-eps)*np.eye(2)-b-s)
            rate=1j*(b+s-(b+s).conj().T)
            A=-(G-G.conj().T)/(2j*np.pi)
            records.append([float(w),float(np.linalg.eigvalsh(rate)[0]),float(np.linalg.eigvalsh(A)[0])])
        records=np.asarray(records)
        row['grids'].append({'eta':eta,'frequency_points':len(omega),
              'minimum_rate_eigenvalue':float(records[:,1].min()),
              'minimum_spectral_eigenvalue':float(records[:,2].min()),
              'pass':bool(records[:,1].min()>-1e-9 and records[:,2].min()>-1e-9),
              'samples':records.tolist()})
    if label in ['minus_level_offset','plus_level_offset']:
        pp,info=t.pair(eng,eps,W,complex(*v[:2]))
        row['distinct_resonance_poles']=[t.enc(p) for p in pp]
    out['cases'].append(row)
    t.save(t.ROOT/'results/physical_sheet_checks.json',out)
    print(label,[(g['eta'],g['pass'],g['minimum_rate_eigenvalue'],g['minimum_spectral_eigenvalue']) for g in row['grids']],flush=True)
