"""Targeted retarded upper-rim convergence and quadrature checks for main Fig. 4 (splitting diagnostic)."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import full_k_two_cut_ep_certificate as c
root=Path(__file__).resolve().parent.parent
v=1.833184476734544
rows=[]
for soc,drive,w in [(.1,60,.020357014816949),(.05,40,.018),(.15,70,.04),(.1,0,.02),(.18,75,.08)]:
 p=replace(c.P,soc=soc,drive_frequency=drive)
 a=c.analytic_physical_self_energy(w+1e-4j,v,p)
 q=c.quadrature_physical_self_energy(w+1e-4j,v,p)
 err=float(np.max(abs(a-q)))
 values=[c.analytic_physical_self_energy(w+eta*1j,v,p) for eta in [1e-8,1e-10,1e-12]]
 diffs=[float(np.max(abs(values[i]-values[-1]))) for i in range(2)]
 h=-(values[-1]-values[-1].conj().T)/(2j)
 rate=np.linalg.eigvalsh(h)
 assert err<1e-10 and diffs[-1]<1e-8 and rate.min()>-1e-10
 rows.append({'soc':soc,'Omega':drive,'omega':w,'quadrature_eta':1e-4,'quadrature_max_absolute_error':err,'eta_1e8_to_1e12':diffs[0],'eta_1e10_to_1e12':diffs[1],'halfwidths':rate.tolist()})
a=np.load(root/'archive_additions/splitting_diagnostic_data.npz')
report={'scope':'five resolved positive-frequency points; not an interval proof of the entire grid','points':rows,'grid_points':int(a['RES'].size),'grid_finite_points':int(np.isfinite(a['RES']).sum()),'grid_min_halfwidth':float(np.nanmin(a['min_halfwidth'])),'grid_eta':float(a['eta'])}
(root/'validation/retarded_map_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
