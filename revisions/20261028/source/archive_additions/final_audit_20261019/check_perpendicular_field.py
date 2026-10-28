"""Selected perpendicular-field continuations, not a global absence search."""
from pathlib import Path
import json,numpy as np
from scipy.optimize import root
from scipy.optimize._numdiff import approx_derivative
import check_field_temperature as q
HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'field_temperature_recheck.json').read_text())
out={}
for W in [.1,.4]:
 p=q.bath.replace(q.P,impurity_hybridization=W)
 e=min(data['driven_W_branch'],key=lambda r:abs(r['W']-W))
 v=np.r_[np.array(e['z'])/.01,e['controls'][0]/.01,e['bath_field_zero']/.01]
 rows=[]
 for ratio in np.arange(0,.201,.005):
  def kernel(z,ct):return q.static_kernel(z,ct[0],(0.,-ct[1]),(0.,-ratio*ct[1]),p)
  def f(v):
   z=complex(v[0],v[1])*.01;ct=np.array(v[2:])*.01
   fun=lambda zz:complex(np.linalg.det(kernel(zz,ct)))
   a=fun(z);b=q.der(fun,z);return [a.real/1e-5,a.imag/1e-5,b.real/.01,b.imag/.01]
  sol=root(f,v,jac=lambda x:approx_derivative(f,x,method='3-point',abs_step=1e-5),tol=1e-10)
  z=complex(sol.x[0],sol.x[1])*.01;ct=np.array(sol.x[2:])*.01
  res=float(np.max(np.abs(f(sol.x))));row=q.describe(kernel,z,ct);nc=int(q.nch_static(z.real,(0.,-ct[1]),p))
  row.update(W=W,impurity_to_bath_g_ratio=float(ratio),scaled_residual=res,channels=nc,solver_success=bool(sol.success))
  rows.append(row)
  if res>1e-8 or z.imag>=0 or nc!=2 or ct[1]<=0:
   print('First unsuccessful continuation step, not absence proof:',json.dumps(row),flush=True);break
  v=sol.x
  if abs(ratio-.05)<1e-8:
   print('Ratio 20 candidate:',json.dumps(row),flush=True)
   rad=q.circularizing_control_loop(kernel,z,ct,[1e-4,1e-4])
   cert=q.certificate(kernel,z,ct,rad,128);cert['control_loop_matrix']=rad.tolist()
   cert['all_poles_two_cut']=all(q.nch_static(zv[0],(0.,-rec['controls'][1]),p)==2 for rec in cert['records'] for zv in rec['poles'])
   out[f'certificate_W{W}_ratio20']=cert
   print('Ratio 20 certificate:',json.dumps(cert['summary']),flush=True)
 out[f'continuation_W{W}']=rows
 (HERE/'perpendicular_field_continuation.json').write_text(json.dumps(out,indent=2))
