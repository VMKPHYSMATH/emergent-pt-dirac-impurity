#!/usr/bin/env python3
"""Matched finite-chain exact/NRG/finite-U KR benchmark. No EP search."""
from pathlib import Path
import sys,json,time
from dataclasses import replace
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent / "solver"
sys.path.insert(0,str(SOURCE))
import filling_campaign as fc
import charge_solver as cs
import exact_block_chain_response as exact
sys.path.insert(0,str(SOURCE/"kr"))
import matrix_saddle as m

class ChainSaddle(m.Saddle):
    """Evaluate the KR quasiparticle functional by diagonalizing the same finite bath."""
    def __init__(self,onsite,hopping,coupling,mu,order=32,p=m.P):
        super().__init__(mu,order,p)
        self.N=len(onsite)
        # Input chain is relative to mu; restore absolute energies consistently.
        bare=exact.build_one_body(onsite,hopping,coupling,0.,p.impurity_hybridization,p.impurity_hybridization,self.N)
        self.Hb=bare[2:,2:]+mu*np.eye(2*self.N)
        self.C=bare[:2,2:]
        self.M=self.C@self.C.conj().T
        self.last_levels=None
    def full_qp(self,R,lam):
        H=np.block([[lam,R@self.C],[self.C.conj().T@R,self.Hb]])
        ev,V=np.linalg.eigh(H)
        f=np.where(ev<self.mu-1e-12,1.,np.where(ev>self.mu+1e-12,0.,.5))
        rho=(V*f)@V.conj().T
        return ev,rho
    def qp(self,R,lam,vo):
        ev,rho=self.full_qp(R,lam)
        self.last_levels=ev
        n=rho[:2,:2]
        B=self.C@rho[2:,:2]+rho[:2,2:]@self.C.conj().T
        if max(np.max(np.abs(n.imag)),np.max(np.abs(B.imag)))>1e-9:
            raise RuntimeError("non-real impurity density requires a larger saddle ansatz")
        return n.real,B.real
    def delta(self,vo):
        inv=np.linalg.inv(self.omega[:,None,None]*np.eye(len(self.Hb))-self.Hb)
        return self.C@inv@self.C.conj().T
    def quadrature_check(self,x,vo):
        _,R,_,_,lam=m.unpack(x)
        d=self.delta(vo)
        G=m.inv2(self.omega[:,None,None]*m.I-lam-R@d@R)
        n=m.I/2+np.einsum("w,wij->ij",self.w,G.real)-(lam-self.mu*m.I)/(np.pi*self.ymax)
        B=np.einsum("w,wij->ij",self.w,(d@R@G+G@R@d).real)-(self.M.real@R+R@self.M.real)/(np.pi*self.ymax)
        ne,Be=self.qp(R,lam,vo)
        return {"density_max_difference":float(np.max(np.abs(n-ne))),"hybridization_force_max_difference":float(np.max(np.abs(B-Be)))}

def simple(row):
    ev=np.array(row["density_eigenvalues"])
    return {"nd":row["n_d"],"D":row["double_occupancy"],"P1":row["n_d"]-2*row["double_occupancy"],
            "spin_magnitude":float((ev[-1]-ev[0])/2),"density_eigenvalues":ev.tolist(),
            "Z_eigenvalues":row["Z_eigenvalues"],"success":row["success"],"residual_max":row["residual_max"]}

def main():
    cfg=json.loads((SOURCE/"configs/submitted_baseline.json").read_text())
    cfg["mapping"]["z_shifts"]=[.5]
    mu=.012; cfg["current_mu"]=mu
    on,hop,C,mapping=fc.make_chains(cfg,mu,ROOT/"cache")[0]
    N=8; W=.1; VO=cfg["model"]["VO"]; EPS=.01517035
    p=replace(m.P,impurity_hybridization=W)
    chain=ChainSaddle(on[:N],hop[:N-1],C,mu,32,p)
    continuum=m.Saddle(mu,32,p)
    report={"scope":"matched static finite-chain and continuum mean-field comparison; no EP certificate",
            "settings":{"mu":mu,"W":W,"VO":VO,"epsilon_d_path":"0.01517035-U/2","bath_blocks":N,"Lambda":1.5,"z_shift":.5},
            "mapping":mapping,"chain_path":[],"comparison":[]}
    targets=[0.,.002,.02,.08,.2]
    ugrid=np.unique(np.round(np.r_[np.linspace(0,.02,21),np.arange(.025,.205,.005),targets],12))
    archived=json.loads((SOURCE/"kr/occupied_gate_path.json").read_text())
    prev=None
    t=time.monotonic()
    for U in ugrid:
        eps=EPS-U/2
        # Continue the finite-chain saddle; retry from independently stored continuum solution.
        seeds=[prev,np.array(min(archived,key=lambda r:abs(r["U"]-U))["x"])]
        attempts=[]
        for seed in seeds:
            with np.errstate(all="ignore"):
                r=chain.solve(float(U),eps,VO,seed)
            attempts.append({"success":r["success"],"residual_max":r["residual_max"],"message":r["message"]})
            if r["success"]:break
        r["attempts"]=attempts
        if not np.isfinite(np.asarray(r["x"])).all(): raise RuntimeError("nonfinite saddle")
        report["chain_path"].append(r)
        print("chain",U,r["success"],r["n_d"],r["double_occupancy"],r["residual_max"],flush=True)
        if not r["success"]:
            report["stop_reason"]="finite-chain KR continuation failed; not evidence of absent physical solution"
            break
        prev=np.array(r["x"])
        if min(abs(U-a) for a in targets)<1e-10:
            h=exact.build_one_body(on,hop,C,eps-mu,W,W,N)
            state=cs.solve(h,float(U),50000,1e-11)
            obs=cs.observables(state)
            exactrow={"nd":obs["nd"],"D":obs["double_occupancy"],"P1":obs["single_occupancy_probability"],
                      "spin_magnitude":float(np.diff(obs["density_eigenvalues"])[0]/2),"density_eigenvalues":obs["density_eigenvalues"],
                      "ground_charge":obs["ground_charges"],"observables":obs}
            seed=np.array(min(archived,key=lambda r:abs(r["U"]-U))["x"])
            cr=continuum.solve(float(U),eps,VO,seed)
            cross=chain.quadrature_check(prev,VO)
            item={"U":float(U),"epsilon_d":float(eps),"ED_same_chain":exactrow,"KR_same_chain":simple(r),
                  "KR_continuum":simple(cr),"chain_quadrature_crosscheck":cross,"continuum_saddle":cr}
            # Compare actual finite-chain energy with this KR stationary functional, without claiming a finite-size variational bound.
            _,R,n,_,lam=m.unpack(prev)
            ev,_=chain.full_qp(R,lam)
            item["KR_chain_functional"]=float(np.minimum(ev-mu,0).sum()+np.trace((eps*m.I-lam)@n)+U*r["double_occupancy"])
            item["ED_chain_grand_energy"]=obs["grand_ground_energy"]
            report["comparison"].append(item)
            print("COMPARE",json.dumps({k:v for k,v in item.items() if k not in ["ED_same_chain","continuum_saddle"]}),flush=True)
        (ROOT/"comparison_results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    report["seconds"]=time.monotonic()-t
    (ROOT/"comparison_results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")

if __name__=="__main__":main()

