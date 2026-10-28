"""Independent numerical checks and self-consistent pole monodromy."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import root
from scipy.optimize._numdiff import approx_derivative
from numpy.polynomial.legendre import leggauss
import matrix_saddle as m

Y=np.array([[0,-1j],[1j,0]])

def stability(row,order=40):
    s=m.Saddle(row['mu'],order,p=m.replace(m.P,impurity_hybridization=row.get('impurity_hybridization',.1)))
    x=np.array(row['x']);U=row['U'];eps=row['epsilon'];vo=row['auxiliary_vertex']
    def gradient(x):
        f=s.residual(x,U,eps,vo);prob=m.unpack(x)[0];pv=prob[1:]
        J=np.diag(pv)-np.outer(pv,pv)
        ns=(prob[1]-prob[2])/2
        return np.r_[J@(f[:3]*.02),f[3]*.02*ns,2*f[4:]]
    H=approx_derivative(gradient,x,method='3-point',abs_step=np.r_[np.repeat(2e-5,4),np.repeat(2e-7,3)])
    he=H[:4,:4]-H[:4,4:]@np.linalg.solve(H[4:,4:],H[4:,:4])
    eig=np.linalg.eigvalsh((he+he.T)/2)
    # The omitted imaginary spin direction is odd under complex conjugation
    # and decouples from the real sector at this saddle. Test its curvature
    # after eliminating the corresponding y-component constraint multiplier.
    prob,R,n,rv,lam=m.unpack(x)
    r0,rs=(rv[0]+rv[1])/2,(rv[0]-rv[1])/2
    n0,ns=np.trace(n)/2,(prob[1]-prob[2])/2
    direction=np.sin(x[3])*m.X+np.cos(x[3])*m.Z
    delta=s.delta(vo)
    def transverse_gradient(v):
        chi,ly=v
        dire=np.cos(chi)*direction+np.sin(chi)*Y
        direp=-np.sin(chi)*direction+np.cos(chi)*Y
        R=r0*m.I+rs*dire;n=n0*m.I+ns*dire;la=lam+ly*Y
        G=m.inv2(s.omega[:,None,None]*m.I-la-R@delta@R)
        Gh=(G+G.conj().transpose(0,2,1))/2
        nq=m.I/2+np.einsum('w,wij->ij',s.w,Gh)-(la-s.mu*m.I)/(np.pi*s.ymax)
        b=delta@R@G+G@R@delta
        B=np.einsum('w,wij->ij',s.w,(b+b.conj().transpose(0,2,1))/2)
        mass=s.p.impurity_hybridization**2*s.p.cutoff**2/(4*np.pi)
        B-=2*mass*R/(np.pi*s.ymax)
        return np.array([np.trace((B*rs-la*ns)@direp).real,np.trace(Y@(nq-n)).real])
    ht=approx_derivative(transverse_gradient,[0.,0.],method='3-point',abs_step=[2e-5,2e-7])
    et=ht[0,0]-ht[0,1]*ht[1,0]/ht[1,1]
    return dict(hessian_symmetry_relative=float(np.linalg.norm(H-H.T)/np.linalg.norm(H)),constrained_curvature_eigenvalues=eig.tolist(),transverse_curvature=float(et),local_minimum=bool(min(eig)>0 and et>0))

def star_check():
    nk=48;k,w=leggauss(nk);k=(k+1)*m.P.cutoff/2;w=w*m.P.cutoff/2*k/(2*np.pi);v=m.P.impurity_hybridization*np.sqrt(w)
    mu=.012;vo=m.VO0;R=np.array([[.85,.07],[.07,.73]]);lam=np.array([[.018,.0007],[.0007,.013]])
    braw=-vo**2*m.P.auxiliary_energy/(m.P.auxiliary_energy**2-m.P.drive_frequency**2)
    H=np.zeros((2+4*nk,2+4*nk));H[:2,:2]=lam
    for j in range(nk):
        a=2+4*j;H[a:a+2,a:a+2]=(k[j]**2-braw)*m.I+m.P.soc*k[j]*m.X
        H[a+2:a+4,a+2:a+4]=m.P.auxiliary_energy*m.I-m.P.drive_frequency*m.Z
        H[a:a+2,a+2:a+4]=vo*m.I;H[a+2:a+4,a:a+2]=vo*m.I
        H[:2,a:a+2]=v[j]*R;H[a:a+2,:2]=v[j]*R
    E,V=np.linalg.eigh(H);occ=V[:,E<mu];D=occ@occ.T
    B=sum(v[j]*(D[2+4*j:4+4*j,:2]+D[:2,2+4*j:4+4*j]) for j in range(nk))
    s=m.Saddle(mu,48);ev,odd=m.bath.auxiliary_terms(s.omega,vo,m.P);q=s.omega-ev
    kk=(q[:,None]-k[None,:]**2)[...,None,None]*m.I-m.P.soc*k[None,:,None,None]*m.X-odd[:,None,None,None]*m.Z
    s._vo=vo;s._delta=np.einsum('k,wkij->wij',v*v,m.inv2(kk))
    nq,Bq=s.qp(R,lam,vo)
    return dict(star_dimension=len(H),density_error=float(np.max(abs(nq-D[:2,:2]))),force_error=float(np.max(abs(Bq-B))))

def pole(z0,x,vo,p):
    def f(a):
        v=m.determinant(complex(*a),x,vo,p)
        return [v.real,v.imag]
    def jac(a):
        v=m.ddet(complex(*a),x,vo,p)
        return [[v.real,-v.imag],[v.imag,v.real]]
    sol=root(f,[z0.real,z0.imag],jac=jac,tol=1e-10)
    z=complex(*sol.x)
    return z,float(abs(m.determinant(z,x,vo,p)))

def exchange(row,samples=128,gate_radius=5e-6,vertex_radius=.001):
    p=m.replace(m.P,impurity_hybridization=row.get('impurity_hybridization',.1))
    s=m.Saddle(row['mu'],40,p=p);x=np.array(row['x']);z0=complex(*row['z'])
    pair=None;records=[]
    for phi in np.linspace(0,2*np.pi,samples+1):
        eps=row['epsilon']+gate_radius*np.cos(phi)
        vo=row['auxiliary_vertex']+vertex_radius*np.sin(phi)
        sr=s.solve(row['U'],eps,vo,x)
        if not sr['success']:raise RuntimeError('Saddle failed on exchange contour')
        x=np.array(sr['x'])
        if pair is None:
            f=m.determinant(z0,x,vo,p);fz=m.ddet(z0,x,vo,p)
            fzz=complex(*row['F_zz'])
            center=z0-fz/fzz;split=np.sqrt((fz/fzz)**2-2*f/fzz)
            pair=np.array([center+split,center-split])
        solved=[pole(z,x,vo,p) for z in pair]
        new=np.array([zz for zz,_ in solved])
        if abs(new[0]-new[1])<1e-8:raise RuntimeError('Roots collapsed during tracking')
        if sum(abs(new-pair))>sum(abs(new[::-1]-pair)):new=new[::-1]
        pair=new
        records.append(dict(phi=float(phi),epsilon=eps,auxiliary_vertex=vo,n_d=sr['n_d'],saddle_residual=sr['residual_max'],poles=[[zz.real,zz.imag] for zz in pair],pole_residual=max(err for _,err in solved),channels=[len(m.bath.open_channels(zz.real,vo,p)) for zz in pair]))
    arr=np.array([[complex(*zz) for zz in r['poles']] for r in records])
    sep=arr[:,0]-arr[:,1];half=np.unwrap(np.angle(sep));wind=(half[-1]-half[0])/(2*np.pi)
    return dict(samples=samples,gate_radius=gate_radius,vertex_radius=vertex_radius,pole_exchange_error=float(np.max(abs(arr[-1]-arr[0,::-1]))),return_without_exchange_error=float(np.max(abs(arr[-1]-arr[0]))),half_winding=float(wind),pole_residual_max=max(r['pole_residual'] for r in records),saddle_residual_max=max(r['saddle_residual'] for r in records),minimum_separation=float(min(abs(sep))),all_two_cut=all(v==2 for r in records for v in r['channels']),records=records)

def main():
    result={'star_check':star_check()}
    points=[1e-5j,1e-3j,.1j,1j,10j,1000j]
    analytic=m.hybridization_array(np.array(points),m.VO0)
    result['radial_integral_error']=max(float(np.max(abs(a-m.bath.quadrature_physical_self_energy(z,m.VO0,m.P)))) for a,z in zip(analytic,points))
    selected=json.load(open('results/ep_n1_U0025.json'))
    refinements=[]
    for order in [20,32,48,64]:
        s=m.Saddle(selected['mu'],order)
        r=m.joint_ep(s,selected['U'],np.array(selected['y']),1.)
        refinements.append(r)
    result['refinements']=refinements
    selected=refinements[-1]
    result['selected']=selected
    result['selected_stability']=stability(selected,64)
    result['exchange']=exchange(selected)
    result['derivative_steps']=[]
    zz=complex(*selected['z']);xx=np.array(selected['x']);vo=selected['auxiliary_vertex']
    for h in [1e-5,5e-6,2e-6,1e-6,5e-7]:
        fz=m.ddet(zz,xx,vo,h=h);fzz=m.bath.derivative(lambda z:m.ddet(z,xx,vo,h=h),zz,h)
        result['derivative_steps'].append(dict(h=h,F_z=abs(fz),F_zz=[fzz.real,fzz.imag]))
    # High-frequency canonical spinon normalization; the coherent physical
    # Green function has weight R R^dagger, not the full unit spectral sum.
    prob,R,n,rv,lam=m.unpack(xx);zs=1e6j
    df=m.hybridization_array(np.array([zs]),vo)[0];G=m.inv2(zs*m.I-lam-R@df@R)
    result['spinon_high_frequency_error']=float(np.max(abs(zs*G-m.I)))
    result['coherent_weight']= (R@R.T).tolist()
    occupied=json.load(open('results/occupied_gate_path.json'))
    result['occupied_checks']=[]
    for target in [.02,.08,.2]:
        row=min(occupied,key=lambda r:abs(r['U']-target))
        s=m.Saddle(row['mu'],56);rr=s.solve(row['U'],row['epsilon'],row['auxiliary_vertex'],np.array(row['x']))
        rr['stability']=stability(rr,56)
        result['occupied_checks'].append(rr)
    # Re-solving from perturbed seeds tests reproducibility of this local saddle.
    rng=np.random.default_rng(1701);s=m.Saddle(selected['mu'],48);seeds=[]
    for i in range(6):
        seed=np.array(selected['x']);seed[:4]+=rng.normal(size=4)*.15;seed[4:]+=rng.normal(size=3)*.0002
        rr=s.solve(selected['U'],selected['epsilon'],selected['auxiliary_vertex'],seed)
        seeds.append(dict(success=rr['success'],n_d=rr['n_d'],double_occupancy=rr['double_occupancy'],R_difference=float(np.max(abs(np.array(rr['R'])-np.array(selected['R'])))),lambda_difference=float(np.max(abs(np.array(rr['lambda_matrix'])-np.array(selected['lambda_matrix'])))),residual=rr['residual_max']))
    result['multiple_seeds']=seeds
    Path('results/validation.json').write_text(json.dumps(result,indent=2))
    concise={k:v for k,v in result.items() if k not in ['refinements','exchange','selected','occupied_checks']}
    concise['exchange']={k:v for k,v in result['exchange'].items() if k!='records'}
    concise['occupied_checks']=[{k:r[k] for k in ['U','n_d','double_occupancy','density_eigenvalues','stability']} for r in result['occupied_checks']]
    print(json.dumps(concise,indent=2),flush=True)

if __name__=='__main__':main()
