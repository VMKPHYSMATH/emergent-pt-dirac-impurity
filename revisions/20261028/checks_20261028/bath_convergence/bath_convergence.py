#!/usr/bin/env python3
"""U=0 bath convergence: exact one-body projectors, Wilson shifts, length and Lambda."""
import json,sys,time
from pathlib import Path
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

def rho_of(H):
    e,V=np.linalg.eigh(H)
    f=np.where(e < -1e-12,1.,np.where(e > 1e-12,0.,.5))
    return ((V*f)@V.conj().T)[:2,:2],int(np.count_nonzero(e < -1e-12))
def summarize(rho):
    ev=np.linalg.eigvalsh(rho)
    return dict(nd=float(np.trace(rho).real),D=float(np.linalg.det(rho).real),
                spin_magnitude=float((ev[-1]-ev[0])/2),density=cs.encode_matrix(rho))
def run():
    mu=.012;eps=.01517035;VO=1.2134220096136235;W=.1
    p=replace(m.P,impurity_hybridization=W)
    ref=m.Saddle(mu,48,p).qp(m.I,eps*m.I,VO)[0]
    ref2=m.Saddle(mu,72,p).qp(m.I,eps*m.I,VO)[0]
    report={"scope":"U=0 one-body bath convergence only; no interacting NRG or EP search",
            "settings":{"mu":mu,"epsilon_d":eps,"VO":VO,"W":W,"U":0.,"soc":p.soc,"cutoff":p.cutoff,
                        "auxiliary_energy":p.auxiliary_energy,"drive_frequency":p.drive_frequency,"low_bands_only":False},
            "continuum":summarize(ref),"reference_grid_difference":float(np.max(abs(ref-ref2))),
            "runs":[],"averages":[]}
    t=time.monotonic()
    for nodes in [512,2048]:
        settings=replace(fc.mapping.Settings(),quadrature_nodes=nodes,auxiliary_vertex=VO,soc=p.soc,impurity_hybridization=W,low_bands_only=False)
        energies,couplings=fc.mapping.microscopic_modes(p,settings)
        energies=energies-mu
        for lam in [2.,1.5,1.25]:
            for z in [.125,.25,.375,.5,.625,.75,.875,1.]:
                es,vs=fc.mapping.logarithmic_star(energies,couplings,lam,z,96,1e-13)
                v0,on,hop,orth=fc.mapping.block_lanczos(es,vs,64)
                if orth>1e-8:raise RuntimeError("block orthogonality failed")
                Hstar=np.block([[(eps-mu)*np.eye(2),vs],[vs.conj().T,np.diag(es)]])
                rs,qs=rho_of(Hstar)
                entry={"quadrature_nodes":nodes,"Lambda":lam,"z_shift":z,"star_modes":len(es),
                       "mapped_blocks":len(on),"orthogonality_error":orth,
                       "star":{**summarize(rs),"ground_charge":qs,"rho_error":float(np.linalg.norm(rs-ref))},
                       "chains":[]}
                for N in [8,12,16,24,32,48,64]:
                    if N>len(on):continue
                    H=exact.build_one_body(on,hop,v0,eps-mu,W,W,N)
                    rho,q=rho_of(H)
                    entry["chains"].append({"bath_blocks":N,**summarize(rho),"ground_charge":q,
                                            "rho_error":float(np.linalg.norm(rho-ref)),
                                            "rho_error_vs_star":float(np.linalg.norm(rho-rs))})
                report["runs"].append(entry)
            group=[a for a in report["runs"] if a["quadrature_nodes"]==nodes and a["Lambda"]==lam]
            for Nz,zset in [(1,[.5]),(2,[.25,.75]),(4,[.25,.5,.75,1.]),(8,[.125,.25,.375,.5,.625,.75,.875,1.])]:
                selected=[a for a in group if a["z_shift"] in zset]
                for N in ["star",8,12,16,24,32,48,64]:
                    members=[a["star"] if N=="star" else next((b for b in a["chains"] if b["bath_blocks"]==N),None) for a in selected]
                    if any(x is None for x in members):continue
                    rho=np.mean([cs.decode_matrix(x["density"]) for x in members],axis=0)
                    # Average observables per mesh; D(avg rho) differs from average D.
                    avg={"quadrature_nodes":nodes,"Lambda":lam,"Nz":Nz,"z_shifts":zset,"bath_blocks":N,
                         "nd":float(np.mean([x["nd"] for x in members])),
                         "D":float(np.mean([x["D"] for x in members])),
                         "spin_magnitude_of_average":summarize(rho)["spin_magnitude"],"density_average":cs.encode_matrix(rho),
                         "rho_error":float(np.linalg.norm(rho-ref)),
                         "nd_error":float(np.mean([x["nd"] for x in members])-np.trace(ref).real),
                         "nd_spread":float(np.ptp([x["nd"] for x in members]))}
                    report["averages"].append(avg)
            selected=[a for a in report["averages"] if a["quadrature_nodes"]==nodes and a["Lambda"]==lam and a["Nz"]==8]
            print(json.dumps({"nodes":nodes,"Lambda":lam,"mapped_blocks":[a["mapped_blocks"] for a in group],"Nz8":selected}),flush=True)
            report["seconds"]=time.monotonic()-t
            (ROOT/"bath_convergence.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
if __name__=="__main__":run()

