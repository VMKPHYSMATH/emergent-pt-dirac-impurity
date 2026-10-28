from pathlib import Path
import json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
SRC=HERE.parent/'finite_u_kr'
sys.path.insert(0,str(SRC/'scripts'));sys.path.insert(0,str(SRC/'kr_package/src'))
import matrix_saddle as m
import kr_certificate as c
import validate as v
r=json.loads((HERE/'new_results_recheck.json').read_text())
row=min(r['fixed_mu_beyond_constrained_fold'],key=lambda a:abs(a['U']-.0032))
s=m.Saddle(row['mu'],64)
rr=m.joint_ep(s,row['U'],np.array(row['y']),None,'V_O')
st=v.stability(rr,64)
print('point',json.dumps({k:rr[k] for k in ['U','success','n_d','double_occupancy','F','F_z','F_zz','singular_values','joint_residual_max']}),flush=True)
print('stability',json.dumps(st),flush=True)
summ,recs=c.certificate(rr,'V_O',2e-6,2e-4,samples=128,order=64)
print('loop',json.dumps(summ),flush=True)
(HERE/'beyond_fold_certificate.json').write_text(json.dumps(dict(ep=rr,stability=st,summary=summ,records=recs),indent=2))
