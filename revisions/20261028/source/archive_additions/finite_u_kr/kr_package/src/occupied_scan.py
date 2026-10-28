"""Occupied gate path; this is not particle-hole symmetry of the bath."""
import json
from pathlib import Path
import numpy as np
import matrix_saddle as m
from validate import pole

s=m.Saddle(.012,32);x=s.initial();rows=[]
for U in np.r_[np.linspace(0,.02,21),np.arange(.025,.205,.005)]:
    r=s.solve(U,m.EPS0-U/2,m.VO0,x);rows.append(r)
    if not r['success']:break
    x=np.array(r['x'])
for target in [.02,.08,.2]:
    r=min(rows,key=lambda r:abs(r['U']-target));x=np.array(r['x'])
    roots=[]
    for er in np.r_[np.linalg.eigvalsh(np.array(r['lambda_matrix'])),np.linspace(-.12,.16,12)]:
        for im in [-.001,-.004,-.012]:
            z,err=pole(complex(er,im),x,r['auxiliary_vertex'],m.P)
            if err<1e-10 and not any(abs(z-rr)<1e-7 for rr in roots):roots.append(z)
    r['candidate_simple_poles']=[dict(z=[z.real,z.imag],F=abs(m.determinant(z,x,r['auxiliary_vertex'])),F_z=abs(m.ddet(z,x,r['auxiliary_vertex'])),open_channels=len(m.bath.open_channels(z.real,r['auxiliary_vertex'],m.P))) for z in roots]
Path('results/occupied_gate_path.json').write_text(json.dumps(rows,indent=2))
print('Completed occupied gate path:',len(rows),'points; endpoint U=',rows[-1]['U'])
