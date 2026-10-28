#!/usr/bin/env python3
"""Passive finite-star fit of the full matrix bath, with held-out validation.

Only the physical sheet is fitted. This is not an interacting continuum EP
certificate and does not approximate an outgoing sheet with real poles.
"""
from __future__ import annotations
import json,sys,time,argparse
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.special import roots_legendre
from scipy.linalg import svd,qr
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'solver'
if not SOURCE.exists():
    SOURCE=ROOT.parent/'bath_convergence_20261028/Local_Moment_ED_NRG_KR_Bath_Checks_20261028/solver'
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(SOURCE/'kr'))
import filling_campaign as fc
import charge_solver as cs
import matrix_saddle as ms

MU=.012;EPS=.0151703497834;W=.1;VO=1.2134220096136235
P=replace(fc.fullk.Parameters(),soc=.05,impurity_hybridization=W)
I=np.eye(2)
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def radial(cutoff,count):
    x,w=roots_legendre(count);k=.5*cutoff*(x+1)
    return k,k*.5*cutoff*w/(2*np.pi)
def microscopic(nodes=8192):
    fc.mapping.radial_quadrature=radial
    settings=replace(fc.mapping.Settings(),quadrature_nodes=nodes,auxiliary_vertex=VO,soc=.05,impurity_hybridization=W,low_bands_only=False)
    e,v=fc.mapping.microscopic_modes(P,settings)
    if abs(v.imag).max()>1e-12:raise RuntimeError('real microscopic bath expected')
    return e,v.real
def physical(z):return ms.hybridization_array(np.asarray(z,dtype=complex),VO,P)
def occupied_bath_moments(order=128):
    # Split exactly where microscopic bath eigenvalues cross the Fermi level.
    # This removes the step-function quadrature error of a global radial grid.
    even,odd=fc.fullk.auxiliary_terms(MU+0j,VO,P)
    roots=fc.fullk.radial_roots(MU-even,odd,P)
    cuts=[0.,P.cutoff]
    for r in roots:
        if abs(r.imag)<1e-12 and 0<r.real<P.cutoff**2:cuts.append(float(np.sqrt(r.real)))
    cuts=sorted(cuts);out=[np.zeros((2,2)) for _ in range(2)]
    xx,ww=roots_legendre(order)
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        kk=(hi+lo)/2+(hi-lo)*xx/2
        weights=ww*(hi-lo)/2*kk/(2*np.pi)*W**2
        for k,w in zip(kk,weights):
            e,V=np.linalg.eigh(fc.mapping.microscopic_bath_matrix(k,VO,P))
            mask=e<MU;v=V[:2,mask].real
            for j in [0,1]:out[j]+=w*(v*e[mask]**j)@v.T
    return out
def hybridization(z,e,v):
    z=np.atleast_1d(z)
    return np.einsum('al,bl,wl->wab',v,v,1/(z[:,None]-e[None,:]))
def one_body(e,v,eps=EPS,Wnew=W,mu=MU):
    return np.block([[(eps-mu)*I,v*(Wnew/W)],[v.T*(Wnew/W),np.diag(e-mu)]])
def packed(a):
    return np.stack([a[...,0,0],a[...,1,1],np.sqrt(2)*a[...,0,1]],axis=-1)
def training_grid(floor=.0002,real_weight=.55):
    imag=MU+1j*np.geomspace(floor,1e3,64)
    real=np.linspace(-.004,.048,35)+.0025j
    return np.r_[imag,real],np.r_[np.ones(len(imag)),np.full(len(real),real_weight)]
def projection_seed(e,v,size,floor=.0002):
    # Orthonormal projection preserves real energies and positive residues.
    # Including v.T preserves bath zeroth and first moments exactly.
    Q,_=qr(v.T,mode='economic')
    columns=[]
    for z in MU+1j*np.geomspace(floor,1.,18):
        a=v.T/(z-e[:,None])
        for b in [a.real,a.imag]:
            b=b-Q@(Q.T@b)
            norm=np.linalg.norm(b,axis=0)
            columns.append(b/np.maximum(norm,1e-30))
    X=np.column_stack(columns)
    L,s,_=svd(X,full_matrices=False)
    B=np.column_stack([Q,L[:,:size-2]])
    B,_=qr(B,mode='economic')
    h=B.T@(e[:,None]*B)
    ee,V=np.linalg.eigh((h+h.T)/2)
    return ee,(v@B)@V
def fit(seed_e,seed_v,targets,moments,maxeval=1200,floor=.0002,real_weight=.55,occupied_moments=None,spin_weight=1.):
    z,weights=training_grid(floor,real_weight);target=targets
    norm=np.maximum(np.linalg.norm(target,axis=(1,2)),1e-7)
    factor=weights/norm
    es=np.maximum(abs(seed_e-MU),.02)
    vs=np.sqrt(np.trace(moments[0])/2)
    n=len(seed_e)
    def pk(a):
        # Orthogonal Pauli components: equal weights recover Frobenius norm.
        return np.stack([(a[...,0,0]+a[...,1,1])/np.sqrt(2),
                         spin_weight*(a[...,0,0]-a[...,1,1])/np.sqrt(2),
                         spin_weight*np.sqrt(2)*a[...,0,1]],axis=-1)
    x0=np.r_[seed_e/es,(seed_v/vs).ravel()]
    mscales=[np.linalg.norm(a) for a in moments]
    occupied=seed_e<MU
    if occupied_moments is not None:
        occupied_scales=[np.linalg.norm(a) for a in occupied_moments]
    def unpack(x):return x[:n]*es,x[n:].reshape(2,n)*vs
    def fun(x):
        e,v=unpack(x);d=(pk(hybridization(z,e,v)-target)*factor[:,None]).ravel()
        mr=[]
        for order,weight in [(0,10.),(1,3.)]:
            m=(v*(e**order)[None,:])@v.T
            mr.extend((weight*pk(m-moments[order])/mscales[order]).tolist())
        if occupied_moments is not None:
            for order,weight in [(0,5.),(1,2.)]:
                m=(v[:,occupied]*(e[occupied]**order)[None,:])@v[:,occupied].T
                mr.extend((weight*pk(m-occupied_moments[order])/occupied_scales[order]).tolist())
        return np.r_[d.real,d.imag,mr]
    def jac(x):
        e,v=unpack(x);den=1/(z[:,None]-e)
        J=np.zeros((len(z),3,3*n),complex)
        for l in range(n):
            R=np.outer(v[:,l],v[:,l])
            J[:,:,l]=pk(R)[None,:]*den[:,l,None]**2*es[l]
            for a in range(2):
                dr=np.zeros((2,2));dr[a,:]+=v[:,l];dr[:,a]+=v[:,l]
                J[:,:,n+a*n+l]=pk(dr)[None,:]*den[:,l,None]*vs
        J=(J*factor[:,None,None]).reshape(-1,3*n)
        tail=[]
        for order,weight in [(0,10.),(1,3.)]:
            K=np.zeros((3,3*n))
            for l in range(n):
                if order:K[:,l]=pk(np.outer(v[:,l],v[:,l]))*es[l]
                for a in range(2):
                    dr=np.zeros((2,2));dr[a,:]+=v[:,l];dr[:,a]+=v[:,l]
                    K[:,n+a*n+l]=pk(dr)*e[l]**order*vs
            tail.append(weight*K/mscales[order])
        if occupied_moments is not None:
            for order,weight in [(0,5.),(1,2.)]:
                K=np.zeros((3,3*n))
                for l in np.where(occupied)[0]:
                    if order:K[:,l]=pk(np.outer(v[:,l],v[:,l]))*es[l]
                    for a in range(2):
                        dr=np.zeros((2,2));dr[a,:]+=v[:,l];dr[:,a]+=v[:,l]
                        K[:,n+a*n+l]=pk(dr)*e[l]**order*vs
                tail.append(weight*K/occupied_scales[order])
        return np.vstack([J.real,J.imag,*tail])
    # A compressed pole can lie in a spectral gap. Constrain only the full
    # spectral range; imposing separate low/remote bands can reject valid seeds.
    lb=np.r_[np.full(n,-.009)/es,np.full(2*n,-4.)]
    ub=np.r_[np.full(n,181.)/es,np.full(2*n,4.)]
    if occupied_moments is not None:
        lb[:n]=np.where(occupied,-.009,MU+1e-8)/es
        ub[:n]=np.where(occupied,MU-1e-8,181.)/es
    sol=least_squares(fun,x0,jac=jac,bounds=(lb,ub),max_nfev=maxeval,ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale='jac')
    e,v=unpack(sol.x);ix=np.argsort(e);e=e[ix];v=v[:,ix]
    return e,v,{'success':bool(sol.success),'message':str(sol.message),'nfev':sol.nfev,'cost':float(sol.cost),'gradient_optimality':float(sol.optimality),'training_relative_rms':float(np.sqrt(np.mean(fun(sol.x)[:6*len(z)]**2))),'low_frequency_training_floor':floor,'real_axis_training_weight':real_weight,'occupied_moment_constraints':occupied_moments is not None,'spin_component_weight':spin_weight}
def validate(e,v,moments):
    groups=[]
    for eta in [.001,.003,.01]:
        z=np.linspace(-.0063,.0507,101)+1j*eta
        target=physical(z);approx=hybridization(z,e,v)
        errors=np.linalg.norm(approx-target,axis=(1,2))/np.linalg.norm(target,axis=(1,2))
        mask=(z.real>.006)&(z.real<.021)
        groups.append({'eta':eta,'rms_relative':float(np.sqrt(np.mean(errors**2))),'max_relative':float(max(errors)),
                       'resonance_window_rms_relative':float(np.sqrt(np.mean(errors[mask]**2)))})
    z=MU+1j*np.geomspace(.00031,300.,91)
    errors=np.linalg.norm(hybridization(z,e,v)-physical(z),axis=(1,2))/np.linalg.norm(physical(z),axis=(1,2))
    filling=[]
    for eps,width in [(EPS,.1),(EPS-.002,.1),(EPS+.002,.1),(.018197601233595,.131445451398)]:
        rho=ms.Saddle(MU,48,replace(P,impurity_hybridization=width)).qp(I,eps*I,VO)[0]
        d=cs.gaussian_observables(one_body(e,v,eps,width))
        fitted=cs.decode_matrix(d['density_matrix'])
        filling.append({'epsilon_d':eps,'W':width,'continuum_nd':float(np.trace(rho).real),'fit_nd':d['nd'],'absolute_nd_error':float(abs(d['nd']-np.trace(rho).real)),'density_frobenius_error':float(np.linalg.norm(fitted-rho))})
    residue_min=min(float(np.linalg.eigvalsh(np.outer(v[:,l],v[:,l]))[0]) for l in range(len(e)))
    return {'held_out_real_frequency':groups,'held_out_imaginary_frequency_rms_relative':float(np.sqrt(np.mean(errors**2))),
            'zeroth_moment_relative_error':float(np.linalg.norm(v@v.T-moments[0])/np.linalg.norm(moments[0])),
            'first_moment_relative_error':float(np.linalg.norm((v*e)@v.T-moments[1])/np.linalg.norm(moments[1])),
            'minimum_residue_eigenvalue':residue_min,'filling':filling,
            'bath_orbitals_below_mu':int(np.sum(e<MU))}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sizes',default='6,8,10,12,14,16');ap.add_argument('--floor',type=float,default=.0002)
    ap.add_argument('--name',default='primary');ap.add_argument('--maxeval',type=int,default=1200);ap.add_argument('--real-weight',type=float,default=.55)
    ap.add_argument('--occupied-moments',action='store_true');ap.add_argument('--exact-occupied-moments',action='store_true');ap.add_argument('--seed-family');ap.add_argument('--spin-weight',type=float,default=1.);a=ap.parse_args()
    t=time.monotonic();e,v=microscopic()
    moments=[(v*(e**k)[None,:])@v.T for k in [0,1]]
    emask=e<MU
    occupied_moments=[(v[:,emask]*(e[emask]**k)[None,:])@v[:,emask].T for k in [0,1]]
    occ_audit={}
    if a.exact_occupied_moments:
        refined=occupied_bath_moments(256);split=occupied_bath_moments(128)
        occ_audit={'split_128_vs_256_relative':[float(np.linalg.norm(x-y)/np.linalg.norm(y)) for x,y in zip(split,refined)],
                   'global_8192_vs_split_relative':[float(np.linalg.norm(x-y)/np.linalg.norm(y)) for x,y in zip(occupied_moments,refined)]}
        occupied_moments=refined
    z,_=training_grid(a.floor,a.real_weight);target=physical(z)
    qe=float(np.max(np.linalg.norm(hybridization(z,e,v)-target,axis=(1,2))/np.linalg.norm(target,axis=(1,2))))
    out={'model':{'mu':MU,'epsilon_d_U0':EPS,'W':W,'VO':VO,'soc':P.soc},'scope':'passive physical-sheet bath compression; no finite-U or EP claim',
         'microscopic_radial_nodes':8192,'microscopic_vs_analytic_max_training_relative_error':qe,'occupied_moment_quadrature_audit':occ_audit,'fits':[]}
    for size in map(int,a.sizes.split(',')):
        se,sv=projection_seed(e,v,size,a.floor)
        if a.seed_family:
            sd=json.loads((ROOT/'results'/f'fits_{a.seed_family}.json').read_text())
            sr=next(r for r in sd['fits'] if r['bath_orbitals']==size)
            se,sv=np.asarray(sr['energies']),np.asarray(sr['couplings'])
        fe,fv,info=fit(se,sv,target,moments,a.maxeval,a.floor,a.real_weight,occupied_moments if a.occupied_moments else None,a.spin_weight)
        val=validate(fe,fv,moments)
        mask=fe<MU
        val['occupied_zeroth_moment_relative_error']=float(np.linalg.norm(fv[:,mask]@fv[:,mask].T-occupied_moments[0])/np.linalg.norm(occupied_moments[0]))
        item={'bath_orbitals':size,'energies':fe.tolist(),'couplings':fv.tolist(),'fit':info,'validation':val,
              'projection_seed_validation':validate(se,sv,moments),'seconds':time.monotonic()-t}
        out['fits'].append(item);save(ROOT/'results'/('fits_'+a.name+'.json'),out)
        print(a.name,size,'cost',info['cost'],'calls',info['nfev'],'nd',val['filling'][0]['fit_nd'],'imag_err',val['held_out_imaginary_frequency_rms_relative'],'real003',val['held_out_real_frequency'][1]['rms_relative'],'nminus',val['bath_orbitals_below_mu'],flush=True)
if __name__=='__main__':main()
