#!/usr/bin/env python3
"""Two-sided EP tests of analytic U=0 and exact finite-chain self-energy embedding."""
from __future__ import annotations
import argparse,json,time,sys,math,copy
from pathlib import Path
from functools import lru_cache
from dataclasses import replace
from collections import OrderedDict
import numpy as np
from scipy.optimize import root,least_squares,linear_sum_assignment
from scipy import sparse

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/"solver"
if not SOURCE.is_dir():
    raise FileNotFoundError("Use the bundled sibling solver/ directory")
sys.path.insert(0,str(SOURCE))
import filling_campaign as c
import charge_solver as charge
import pole_derivatives as pd
import exact_block_chain_response as exact

def enc(z):return [float(complex(z).real),float(complex(z).imag)]
def save(path,obj):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
class Engine:
    def __init__(self,U=0.,mu=0.,N=6,zs=(.5,),nodes=512,lam=1.5):
        self.cfg=json.loads((SOURCE/"configs/submitted_baseline.json").read_text())
        self.cfg["mapping"].update(z_shifts=list(zs),quadrature_nodes=nodes,Lambda=lam)
        self.cfg["current_mu"]=mu
        self.U=U;self.mu=mu;self.N=N;self.cache=OrderedDict();self.builds=0
        self.chains=c.make_chains(self.cfg,mu,ROOT/"chain_cache") if U else []
        self.h=1e-6;self.q=.0025
    def states(self,eps,W):
        key=(float(eps),float(W))
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        states=c.build_states(self.cfg,self.chains,eps,W,self.U,self.N)
        self.builds+=1;self.cache[key]=states
        while len(self.cache)>6:self.cache.popitem(last=False)
        return states
    def evaluate(self,z,eps,W,states=None,h=None):
        p=c.parameters(self.cfg["model"],W);vo=self.cfg["model"]["VO"]
        bfun=lambda zz:c.fullk.resonance_self_energy(zz,vo,p)
        b0,b1,b2=c.h_derivatives(bfun,z,h or self.h)
        if self.U:
            states=self.states(eps,W) if states is None else states
            s0,s1,s2=pd.sigma_derivatives(states,z-self.mu)
        else:s0=s1=s2=np.zeros((2,2),complex)
        K=(z-eps)*np.eye(2)-b0-s0
        D1=np.eye(2)-b1-s1;D2=-b2-s2
        f=np.linalg.det(K)
        fz=D1[0,0]*K[1,1]+K[0,0]*D1[1,1]-D1[0,1]*K[1,0]-K[0,1]*D1[1,0]
        fzz=(D2[0,0]*K[1,1]+2*D1[0,0]*D1[1,1]+K[0,0]*D2[1,1]
             -D2[0,1]*K[1,0]-2*D1[0,1]*D1[1,0]-K[0,1]*D2[1,0])
        return complex(f),complex(fz),complex(fzz),K
    def determinant(self,z,eps,W,states=None):
        p=c.parameters(self.cfg["model"],W);vo=self.cfg["model"]["VO"]
        b=c.fullk.resonance_self_energy(z,vo,p)
        if self.U:
            states=self.states(eps,W) if states is None else states
            s=c.interaction_sigma(states,z-self.mu)
        else:s=0.
        return complex(np.linalg.det((z-eps)*np.eye(2)-b-s))
    def describe(self):
        return {"U":self.U,"mu":self.mu,"bath_blocks":self.N,
                "z_shifts":self.cfg["mapping"]["z_shifts"],"Lambda":self.cfg["mapping"]["Lambda"],
                "quadrature_nodes":self.cfg["mapping"]["quadrature_nodes"],"VO":self.cfg["model"]["VO"],
                "scope":"analytic continuum U=0" if self.U==0 else "exact finite-chain self-energy embedding"}
    def filling(self,eps,W):
        if not self.U:return None
        return c.fillings(self.states(eps,W))

def center(engine,seed,bounded=False,maxeval=90):
    scale=np.array([.01,.003,.01,.1])
    def fun(y):
        v=y*scale;z=complex(*v[:2]);eps,W=v[2:]
        if W<.002 or W>.8 or not (-.5<eps<.5) or not(-.15<z.imag<-1e-7):
            return [100.+abs(float(y[0])),100.+abs(float(y[1])),100.+abs(float(y[2])),100.+abs(float(y[3]))]
        f,fz,_,_=engine.evaluate(z,eps,W)
        return np.array([f.real/engine.q**2,f.imag/engine.q**2,fz.real/engine.q,fz.imag/engine.q])
    t=time.monotonic()
    if bounded:
        sol=least_squares(fun,np.asarray(seed)/scale,bounds=(np.array([-.15,-.149,-.499,.00201])/scale,np.array([.3,-1.01e-7,.499,.799])/scale),max_nfev=maxeval,xtol=1e-10,ftol=1e-10,gtol=1e-10,diff_step=1e-5)
    else:
        sol=root(fun,np.asarray(seed)/scale,tol=1e-9,options={"maxfev":maxeval,"eps":1e-10})
    v=np.asarray(sol.x)*scale
    residual=float(np.max(abs(fun(sol.x))))
    out={"center":v.tolist(),"solver_success":bool(sol.success),"message":str(sol.message),
         "scaled_residual":residual,"nfev":int(sol.nfev),"seconds":time.monotonic()-t,
         "center_numerical_pass":bool(residual<3e-7 and v[3]>.002 and v[1]<0)}
    if out["center_numerical_pass"]:
        z=complex(*v[:2]);eps,W=v[2:];f,fz,fzz,K=engine.evaluate(z,eps,W)
        f2,fz2,fzz2,_=engine.evaluate(z,eps,W,h=engine.h/2)
        ss=np.linalg.svd(K,compute_uv=False)
        channels=c.fullk.open_channels(z.real,engine.cfg["model"]["VO"],c.parameters(engine.cfg["model"],W))
        out["checks"]={"F":abs(f),"F_z":abs(fz),"F_zz":enc(fzz),"singular_values":ss.tolist(),
                       "Fz_derivative_halving_difference":abs(fz-fz2),
                       "Fzz_derivative_halving_relative_difference":abs(fzz-fzz2)/max(abs(fzz),1e-30),
                       "open_channel_count":len(channels)}
        out["center_numerical_pass"]=bool(out["center_numerical_pass"] and abs(fzz)>1e-4 and ss[0]>1e-5 and ss[1]<1e-8 and len(channels)==2 and out["checks"]["Fzz_derivative_halving_relative_difference"]<.01)
        if engine.U:
            states=engine.states(eps,W)
            out["filling"]=c.fillings(states)
            axis=c.physical_diagnostics(engine.cfg,states,eps,W)
            out["physical_axis_checks"]=axis;out["physical_axis_pass"]=c.physical_checks_pass(axis)
            out["center_numerical_pass"] &= out["physical_axis_pass"]
            out["ground_charges"]=[charge.observables(s)["ground_charges"] for s in states]
            # Check exact resolvent derivatives against differences of the actual ratio.
            h=2e-6
            sf=lambda zz:c.interaction_sigma(states,zz-engine.mu)
            _,sd1,sd2=c.h_derivatives(sf,z,h)
            _,sa1,sa2=pd.sigma_derivatives(states,z-engine.mu)
            out["selfenergy_derivative_checks"]={"first_relative":float(np.linalg.norm(sd1-sa1)/max(np.linalg.norm(sa1),1e-12)),
                                                "second_relative":float(np.linalg.norm(sd2-sa2)/max(np.linalg.norm(sa2),1e-12))}
    return out

def pair(engine,eps,W,zseed):
    states=engine.states(eps,W) if engine.U else None
    def crit(v):
        f,fz,fzz,K=engine.evaluate(complex(*v),eps,W,states)
        return [fz.real/engine.q,fz.imag/engine.q]
    sol=root(crit,[zseed.real,zseed.imag],tol=1e-10)
    z=complex(*sol.x);f,fz,fzz,K=engine.evaluate(z,eps,W,states)
    if abs(fzz)<1e-6 or abs(fz)>1e-7:raise RuntimeError("critical-point search failed")
    delta=np.sqrt(-2*f/fzz)
    poles=[]
    for guess in [z+delta,z-delta]:
        fun=lambda v: [engine.determinant(complex(*v),eps,W,states).real/engine.q**2,
                       engine.determinant(complex(*v),eps,W,states).imag/engine.q**2]
        s=root(fun,[guess.real,guess.imag],tol=1e-10)
        pole=complex(*s.x);err=abs(engine.determinant(pole,eps,W,states))
        if err>1e-11 or pole.imag>=0:raise RuntimeError("separated-pole solve failed")
        poles.append(pole)
    if abs(poles[0]-poles[1])<1e-8:raise RuntimeError("unresolved pole separation")
    channels=[len(c.fullk.open_channels(p.real,engine.cfg["model"]["VO"],c.parameters(engine.cfg["model"],W))) for p in poles]
    return np.array(poles),{"zcritical":enc(z),"Fcritical":enc(f),"Fz_modulus":abs(fz),"Fzz":enc(fzz),"open_channels":channels,
                            "ground_charges":[] if not engine.U else [charge.observables(s)["ground_charges"] for s in states]}

def sided(engine,reference):
    v=np.array(reference);c0=v[2:]+np.array([4e-5,-3e-4]);zseed=complex(*v[:2]);rows=[]
    for step in range(4):
        h=np.array([1e-4,1e-3])/(2**step);ds=[];critical=[];details=[]
        for j in range(2):
            axis=[]
            for sign in [-1,1]:
                cc=c0.copy();cc[j]+=sign*h[j]
                poles,d=pair(engine,*cc,zseed)
                axis.append((poles[0]-poles[1])**2);critical.append(complex(*d["zcritical"]))
                details.append({"direction":j,"sign":sign,"controls":cc.tolist(),"poles":[enc(p) for p in poles],**d})
            ds.append(axis)
        D=np.mean(ds);J=np.column_stack([np.array([((a[1]-a[0])/(2*h[j])).real,((a[1]-a[0])/(2*h[j])).imag]) for j,a in enumerate(ds)])
        correction=np.linalg.solve(J,-np.array([D.real,D.imag]))
        c0+=correction;zseed=sum(critical)/len(critical)
        rows.append({"step":step,"half_steps":h.tolist(),"estimated_controls":c0.tolist(),
                     "error_from_independent_double_zero":(c0-v[2:]).tolist(),"jacobian_condition":float(np.linalg.cond(J*np.array([.01,.1])[None,:])),
                     "samples":details})
    fitted=center(engine,[zseed.real,zseed.imag,*c0])
    return {"steps":rows,"final_double_zero":fitted,"center_difference":(np.asarray(fitted["center"])-v).tolist()}

def loop(engine,reference,radius=1.,samples=64):
    v=np.array(reference);zseed=complex(*v[:2]);previous=None;allp=[];rows=[];maxratio=0
    for theta in np.linspace(0,2*np.pi,samples+1):
        eps=v[2]+radius*1e-4*np.cos(theta);W=v[3]+radius*1e-3*np.sin(theta)
        pp,d=pair(engine,eps,W,zseed)
        zseed=complex(*d["zcritical"])
        if previous is not None:
            _,ix=linear_sum_assignment(abs(previous[:,None]-pp[None,:]));pp=pp[ix]
            maxratio=max(maxratio,float(np.max(abs(pp-previous))/(abs(pp[0]-pp[1])/2)))
        allp.append(pp);previous=pp
        rows.append({"angle":float(theta),"epsilon_d":float(eps),"W":float(W),"poles":[enc(p) for p in pp],**d})
    allp=np.array(allp);sep=allp[:,0]-allp[:,1];D=sep**2
    phase=np.angle(D[1:]/D[:-1]);winding=float(np.sum(phase)/(2*np.pi))
    half=float(np.sum(np.angle(sep[1:]/sep[:-1]))/(2*np.pi))
    _,perm=linear_sum_assignment(abs(allp[0,:,None]-allp[-1,None,:]))
    charges={json.dumps(row["ground_charges"]) for row in rows}
    passed=abs(abs(winding)-1)<.01 and abs(abs(half)-.5)<.01 and perm.tolist()==[1,0] and max(abs(phase))<np.pi/2 and maxratio<.5 and len(charges)==1 and all(x==2 for row in rows for x in row["open_channels"])
    return {"pass":bool(passed),"winding":winding,"half_winding":half,"permutation":perm.tolist(),"radius":radius,"samples":samples,
            "max_phase_step":float(max(abs(phase))),"max_assignment_to_half_separation_ratio":maxratio,
            "minimum_pole_separation":float(min(abs(sep))),"unique_ground_charge_patterns":len(charges),"rows":rows}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mu",type=float,default=0.);ap.add_argument("--N",type=int,default=6)
    ap.add_argument("--zs",default=".5");ap.add_argument("--nodes",type=int,default=512)
    ap.add_argument("--U",default="0,.0005,.001,.0025,.005,.01,.02,.05,.1,.2")
    ap.add_argument("--name",required=True);ap.add_argument("--certify",action="store_true")
    a=ap.parse_args();seed=[.011972834346,-.002807216274,.0151703497834,.1];results=[]
    for U in [float(x) for x in a.U.split(",")]:
        eng=Engine(U,a.mu,a.N,tuple(float(z) for z in a.zs.split(",")),a.nodes)
        item={"settings":eng.describe()}
        t=time.monotonic()
        try:
            rr=center(eng,seed);item["solution"]=rr
            print(a.name,"U",U,"pass",rr["center_numerical_pass"],"center",rr["center"],"nd",rr.get("filling",{}).get("nd"),"calls",rr["nfev"],flush=True)
            if rr["center_numerical_pass"]:
                seed=rr["center"]
                if a.certify:
                    item["two_sided"]=sided(eng,seed)
                    item["loop"]=loop(eng,seed)
                    print("certificate",item["loop"]["pass"],item["loop"]["winding"],item["loop"]["half_winding"],flush=True)
            else:
                item["failure_scope"]="This continuation did not certify a center; it does not prove absence."
        except Exception as e:
            item["error"]=type(e).__name__+": "+str(e)
            print(a.name,"U",U,"ERROR",item["error"],flush=True)
        item["seconds"]=time.monotonic()-t;item["state_builds"]=eng.builds;results.append(item)
        save(ROOT/"results"/(a.name+".json"),{"scope":"tests of analytic continuum U=0 and finite-chain interacting embedding, not continuum interacting certification","results":results})
        if item.get("error") or not item.get("solution",{}).get("center_numerical_pass",False):break
if __name__=="__main__":main()

