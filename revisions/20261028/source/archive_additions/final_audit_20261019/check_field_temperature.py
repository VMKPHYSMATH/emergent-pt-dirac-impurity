"""Independent U=0 field/width checks; no submitted source is changed.

Static fields are coefficients b.sigma, so the Zeeman splitting is 2|b|.
The general tilted static bath retains BOTH field components. Its radial
integral is evaluated by quartic partial fractions and checked separately.
"""
from pathlib import Path
import sys,json,math
import numpy as np
from scipy.optimize import root,linear_sum_assignment
from scipy.optimize._numdiff import approx_derivative
from scipy.integrate import quad_vec
from scipy.constants import h,k as kb,c as lightspeed

HERE=Path(__file__).resolve().parent
SRC=HERE.parent/'finite_u_kr/kr_package/src'
sys.path.insert(0,str(SRC))
import full_k_two_cut_ep_certificate as bath

I,X,Z=bath.I2,bath.SX,bath.SZ
P=bath.replace(bath.Parameters(),soc=.05)
out={}
def enc(z):return [float(z.real),float(z.imag)]
def save(): (HERE/'field_temperature_recheck.json').write_text(json.dumps(out,indent=2))
def emit(label,obj):print(label,json.dumps(obj),flush=True)
def der(fun,z):return bath.derivative(fun,z,2e-6)

def static_roots(z,bc,p):
    bx,bz=bc
    return np.roots([1.,0.,-2*z-p.soc**2,-2*p.soc*bx,z*z-bx*bx-bz*bz])

def static_integral(z,bc,p,continued=True):
    roots=static_roots(z,bc,p);bx,bz=bc
    residues=[]
    for r in roots:
        den=4*r**3-2*(2*z+p.soc**2)*r-2*p.soc*bx
        residues.append(r*((z-r*r)*I+(p.soc*r+bx)*X+bz*Z)/den)
    residues=np.array(residues)
    integrals=np.log(p.cutoff-roots)-np.log(-roots)
    ans=p.impurity_hybridization**2/(2*np.pi)*np.einsum('r,rij->ij',integrals,residues)
    if continued:
        real_roots=static_roots(z.real,bc,p)
        rows,cols=linear_sum_assignment(abs(real_roots[:,None]-roots[None,:]))
        mapping=dict(zip(rows,cols))
        for j,rr in enumerate(real_roots):
            if abs(rr.imag)<1e-8 and 1e-10<rr.real<p.cutoff-1e-10:
                vel=2*rr.real+p.soc*(p.soc*rr.real+bx)/(z.real-rr.real**2)
                ans+=1j*p.impurity_hybridization**2*residues[mapping[j]]*np.sign(vel)
    return ans

def nch_static(zreal,bc,p):
    return sum(abs(r.imag)<1e-8 and 1e-10<r.real<p.cutoff-1e-10 for r in static_roots(zreal,bc,p))

def static_kernel(z,eps,bc,bd,p):
    return (z-eps)*I-bd[0]*X-bd[1]*Z-static_integral(z,bc,p)

def describe(kernel,z,controls):
    fun=lambda a:complex(np.linalg.det(kernel(a,controls)))
    f=fun(z);fz=der(fun,z);fzz=der(lambda a:der(fun,a),z)
    return dict(z=enc(z),controls=list(map(float,controls)),F=abs(f),F_z=abs(fz),F_zz=enc(fzz),singular_values=np.linalg.svd(kernel(z,controls),compute_uv=False).tolist())

def certificate(kernel,z0,controls,radii,samples=128):
    cs=np.array(controls,float);records=[];zc=z0;tracked=[];assign=0.;bad=0;flags=0
    for theta in np.linspace(0,2*np.pi,samples+1):
        rmat=np.array(radii)
        if rmat.ndim==1:rmat=np.diag(rmat)
        ct=cs+rmat@np.array([np.cos(theta),np.sin(theta)])
        f=lambda z:complex(np.linalg.det(kernel(z,ct)))
        def cr(v):
            q=der(f,complex(*v));return [q.real,q.imag]
        sol=root(cr,enc(zc),tol=1e-10);zc=complex(*sol.x);flags+=not sol.success
        fz=abs(der(f,zc));D=f(zc);fzz=der(lambda a:der(f,a),zc)
        split=np.sqrt(-2*D/fzz);new=[];pres=0.
        for zg in [zc+split,zc-split]:
            ss=root(lambda v:[f(complex(*v)).real,f(complex(*v)).imag],enc(zg),tol=1e-10)
            flags+=not ss.success;zz=complex(*ss.x);new.append(zz);pres=max(pres,abs(f(zz)))
        new=np.array(new)
        if tracked:
            cost=abs(tracked[-1][:,None]-new[None,:]);rr,cc=linear_sum_assignment(cost)
            ordered=np.empty(2,complex);ordered[rr]=new[cc]
            sep=.5*min(abs(tracked[-1][1]-tracked[-1][0]),abs(ordered[1]-ordered[0]))
            assign=max(assign,float(np.max(cost[rr,cc])/sep));new=ordered
        tracked.append(new)
        bad+=bool(fz>1e-10 or pres>1e-14 or abs(new[1]-new[0])<1e-9)
        records.append(dict(theta=float(theta),controls=ct.tolist(),D=enc(D),F_z=fz,F_zz=enc(fzz),poles=[enc(zz) for zz in new],pole_residual=pres))
    arr=np.array(tracked);D=np.array([complex(*r['D']) for r in records]);inc=np.angle(D[1:]/D[:-1]);sep=arr[:,1]-arr[:,0]
    rr,cc=linear_sum_assignment(abs(arr[-1][:,None]-arr[0][None,:]));perm=np.empty(2,int);perm[rr]=cc
    summary=dict(winding=float(inc.sum()/(2*np.pi)),half_winding=float(np.angle(sep[1:]/sep[:-1]).sum()/(2*np.pi)),permutation=perm.tolist(),max_phase_increment=float(abs(inc).max()),minimum_boundary_modulus=float(abs(D).min()),minimum_pole_separation=float(abs(sep).min()),max_assignment_ratio=assign,maximum_pole_residual=max(r['pole_residual'] for r in records),maximum_critical_residual=max(r['F_z'] for r in records),failures=int(bad),solver_flags=int(flags),exchange_error=float(abs(arr[-1]-arr[0,::-1]).max()))
    summary['numerical_gates_pass']=bool(bad==0 and abs(abs(summary['winding'])-1)<1e-7 and abs(abs(summary['half_winding'])-.5)<1e-7 and perm.tolist()==[1,0] and summary['max_phase_increment']<np.pi/2 and assign<.5)
    return dict(summary=summary,records=records)

def circularizing_control_loop(kernel,z,controls,max_excursions):
    def local(c):
        f=complex(np.linalg.det(kernel(z,c)));return [f.real,f.imag]
    J=approx_derivative(local,np.array(controls),method='3-point',abs_step=1e-7)
    inv=np.linalg.inv(J)
    amplitude=min(np.array(max_excursions)/np.linalg.norm(inv,axis=1))
    return amplitude*inv

if __name__ == "__main__":
    # Exact radial-integral validation away from the physical cuts.
    errors=[]
    for bc in [(0.,-.00818),(.0005,.0004)]:
        for zz in [.012+.004j,.02+.01j,-.01+.03j]:
            a=static_integral(zz,bc,P,False)
            def integ(k):return P.impurity_hybridization**2*k/(2*np.pi)*np.linalg.inv((zz-k*k)*I-(P.soc*k+bc[0])*X-bc[1]*Z)
            b,err=quad_vec(integ,0,P.cutoff,epsabs=1e-13,epsrel=1e-11)
            errors.append(float(abs(a-b).max()))
            if bc[0]==0:errors.append(float(abs(a-bath.analytic_physical_self_energy_from_q(zz,bc[1],P)).max()))
    out['static_integral_max_quadrature_error']=max(errors);emit('static integral error',max(errors))
    
    # Driven EP branch, continuously retuned in W.
    v=np.array([.011972834346/.01,-.002807216274/.01,.0151703497834/.01,1.21342200961])
    out['driven_W_branch']=[]
    for W in np.arange(.1,.751,.005):
        p=bath.replace(P,impurity_hybridization=float(W))
        kern=lambda z,ct:(z-ct[0])*I-bath.resonance_self_energy(z,ct[1],p)
        def res(v):
            z=complex(v[0],v[1])*.01;ct=[v[2]*.01,v[3]]
            f=lambda zz:complex(np.linalg.det(kern(zz,ct)))
            a=f(z);b=der(f,z);return [a.real/1e-5,a.imag/1e-5,b.real/.01,b.imag/.01]
        sol=root(res,v,jac=lambda vv:approx_derivative(res,vv,method='3-point',abs_step=1e-5),tol=1e-10);resid=float(np.max(np.abs(res(sol.x))))
        z=complex(sol.x[0],sol.x[1])*.01;ct=[sol.x[2]*.01,sol.x[3]]
        nc=len(bath.open_channels(z.real,ct[1],p))
        if resid>1e-8 or z.imag>=0 or nc!=2:
            out['driven_first_failed_step']=dict(W=float(W),residual=resid,z=enc(z),channels=nc,solver_success=bool(sol.success));break
        v=sol.x
        b0=abs(bath.auxiliary_terms(0.,ct[1],p)[1]);threshold=bath.upper_threshold(ct[1],p)
        row=describe(kern,z,ct);row.update(W=float(W),Gamma=W*W/4,Tstar_width_proxy=.4*W*W/4,Tstar_pole_proxy=.4*abs(z.imag),bath_field_zero=float(b0),threshold=threshold,threshold_gap_widths=(z.real-threshold)/abs(z.imag),x=ct[1]/60,channels=nc,scaled_residual=resid,solver_success=bool(sol.success))
        out['driven_W_branch'].append(row)
        if any(abs(W-w)<1e-6 for w in [.1,.2,.3,.4,.7]):emit('W branch',row)
        if abs(W-.4)<1e-6:
            cert=certificate(kern,z,ct,[1e-4,1e-2],samples=128)
            cert['all_poles_two_cut']=all(len(bath.open_channels(zz[0],rec['controls'][1],p))==2 for rec in cert['records'] for zz in rec['poles'])
            out['driven_W04_certificate']=cert;emit('W=.4 certificate',cert['summary'])
        save()
    
    # Static bath-only field, same exact full radial integral, no auxiliary modes.
    kern=lambda z,ct:static_kernel(z,ct[0],(0.,-ct[1]),(0.,0.),P)
    v0=np.array([.01197/.01,-.002807/.01,.01517/.01,.00818/.01])
    def res(v):
        z=complex(v[0],v[1])*.01;ct=np.array(v[2:])*.01
        f=lambda zz:complex(np.linalg.det(kern(zz,ct)))
        a=f(z);b=der(f,z);return [a.real/1e-5,a.imag/1e-5,b.real/.01,b.imag/.01]
    sol=root(res,v0,tol=1e-11);z=complex(sol.x[0],sol.x[1])*.01;ct=np.array(sol.x[2:])*.01
    row=describe(kern,z,ct);row['scaled_residual']=float(np.max(np.abs(res(sol.x))));row['solver_success']=bool(sol.success)
    out['static_bath_only']=row;emit('static bath only',row)
    out['static_bath_only_certificate']=certificate(kern,z,ct,[1e-4,1e-4],128)
    emit('static bath certificate',out['static_bath_only_certificate']['summary']);save()
    
    # Tilted physical field: obtain points at selected Re z, including both bath
    # components when bath_ratio=1. Then test a loop in gate and field strength
    # at the fixed field direction of one selected point.
    out['tilted_field_points']=[]
    for ratio in [0.,1.]:
        for E in [.006,.012,.03]:
            sg=bath.pauli_components(static_integral(E-.0025j,(0.,0.),P))
            guess=np.array([-.0025,E-sg[0].real,-sg[1].real,abs(sg[1].imag)])/.01
            def res(v):
                zi,eps,bx,bz=v*.01;zz=E+1j*zi
                f=lambda zz:complex(np.linalg.det(static_kernel(zz,eps,(ratio*bx,ratio*bz),(bx,bz),P)))
                a=f(zz);b=der(f,zz);return [a.real/1e-5,a.imag/1e-5,b.real/.01,b.imag/.01]
            sol=root(res,guess,tol=1e-10);resid=float(np.max(np.abs(res(sol.x))))
            zi,eps,bx,bz=sol.x*.01;zz=E+1j*zi;bmag=float(np.hypot(bx,bz));direction=np.array([bx,bz])/bmag
            kern=lambda z,ct:static_kernel(z,ct[0],tuple(ratio*ct[1]*direction),tuple(ct[1]*direction),P)
            row=describe(kern,zz,[eps,bmag]);row.update(bath_to_impurity_ratio=ratio,bx=float(bx),bz=float(bz),angle_from_SOC_x_deg=float(np.degrees(np.arctan2(bz,bx))),field_over_Gamma=bmag/.0025,scaled_residual=resid,solver_success=bool(sol.success),channels=int(nch_static(E,(ratio*bx,ratio*bz),P)),Tstar_proxy_over_b=.001/bmag)
            out['tilted_field_points'].append(row);emit('tilted field',row)
            if ratio==1. and E==.012 and resid<1e-8:
                radii=circularizing_control_loop(kern,zz,[eps,bmag],[1e-5,bmag*.01])
                cert=certificate(kern,zz,[eps,bmag],radii,128)
                cert['control_loop_matrix']=radii.tolist()
                cert['all_poles_two_cut']=all(nch_static(zv[0],tuple(ratio*rec['controls'][1]*direction),P)==2 for rec in cert['records'] for zv in rec['poles'])
                out['equal_g_full_tilt_certificate']=cert
                emit('full tilted certificate',cert['summary'])
            save()
    
    checks=[]
    for a in out['tilted_field_points']:
        if a['bath_to_impurity_ratio']!=1 or abs(a['z'][0]-.012)>1e-9:continue
        bc=(a['bx'],a['bz']);E=a['z'][0];z=complex(*a['z']);eps=a['controls'][0]
        upper=static_integral(E+1e-9j,bc,P,False);lower=static_integral(E-1e-9j,bc,P,True)
        rates=np.linalg.eigvalsh(-(upper-upper.conj().T)/(2j))
        fun=lambda zz:complex(np.linalg.det(static_kernel(zz,eps,bc,bc,P)))
        steps=[]
        for hs in [1e-5,5e-6,2e-6,1e-6]:
            fz=bath.derivative(fun,z,hs);fzz=bath.derivative(lambda zz:bath.derivative(fun,zz,hs),z,hs)
            steps.append(dict(h=hs,F_z=abs(fz),F_zz=enc(fzz)))
        checks.append(dict(half_width_eigenvalues=rates.tolist(),continued_lower_upper_difference=float(abs(lower-upper).max()),derivative_steps=steps))
    out['equal_g_tilt_extra_checks']=checks
    base=out['driven_W_branch'][0];w4=min(out['driven_W_branch'],key=lambda r:abs(r['W']-.4))
    assert abs(w4['W']-.4)<1e-8, 'Do not report W=.4 conversions unless that point was reached'
    out['conversions']=dict(h_over_k=h/kb,muB_over_k=0.67171381472,
        f_THz1=dict(energy_eV=h*1e12/1.602176634e-19,Tbase=.001/60*h*1e12/kb,TW04_barewidth=w4['Tstar_width_proxy']/60*h*1e12/kb,TW04_polewidth=w4['Tstar_pole_proxy']/60*h*1e12/kb),
        wavelength_nm800=dict(energy_eV=h*lightspeed/(800e-9)/1.602176634e-19,Tbase=.001/60*h*lightspeed/(800e-9)/kb,TW04_barewidth=w4['Tstar_width_proxy']/60*h*lightspeed/(800e-9)/kb,TW04_polewidth=w4['Tstar_pole_proxy']/60*h*lightspeed/(800e-9)/kb))
    emit('conversions',out['conversions']);save()
