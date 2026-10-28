#!/usr/bin/env python3
"""Finer U=0 star-discretization check, using equivalent scipy Gauss-Legendre nodes."""
from pathlib import Path
import sys,json,time
from dataclasses import replace
import numpy as np
from scipy.special import roots_legendre
from scipy.linalg import eigh
import bath_convergence as b

def gaussian_nodes(cutoff,count):
    x,w=roots_legendre(count)
    k=.5*cutoff*(x+1)
    return k,k*.5*cutoff*w/(2*np.pi)
b.fc.mapping.radial_quadrature=gaussian_nodes

def main():
    p=replace(b.m.P,impurity_hybridization=.1)
    mu=.012;eps=.01517035;vo=1.2134220096136235
    ref=b.m.Saddle(mu,48,p).qp(b.m.I,eps*b.m.I,vo)[0]
    out={"scope":"U=0 full-star checks only; no many-body truncation or finite-chain tail",
         "node_generator":"scipy.special.roots_legendre; same Gauss-Legendre quadrature as base mapping",
         "continuum":b.summarize(ref),"runs":[],"averages":[]}
    t=time.monotonic()
    for nodes in [2048,8192]:
        settings=replace(b.fc.mapping.Settings(),quadrature_nodes=nodes,auxiliary_vertex=vo,soc=p.soc,impurity_hybridization=.1,low_bands_only=False)
        energies,vs=b.fc.mapping.microscopic_modes(p,settings);energies-=mu
        for lam in [1.5,1.25,1.1,1.05]:
            members=[]
            for z in [.125,.25,.375,.5,.625,.75,.875,1.]:
                es,v=b.fc.mapping.logarithmic_star(energies,vs,lam,z,400,1e-15)
                H=np.block([[(eps-mu)*np.eye(2),v],[v.conj().T,np.diag(es)]])
                ev,V=eigh(H,driver="evr")
                f=(ev<0).astype(float)
                rho=((V*f)@V.conj().T)[:2,:2]
                x={"quadrature_nodes":nodes,"Lambda":lam,"z_shift":z,"star_modes":len(es),**b.summarize(rho)}
                out["runs"].append(x);members.append(x)
            rho=np.mean([b.cs.decode_matrix(x["density"]) for x in members],axis=0)
            row={"quadrature_nodes":nodes,"Lambda":lam,"Nz":8,"nd":float(np.trace(rho).real),"nd_error":float(np.trace(rho-ref).real),
                 "rho_error":float(np.linalg.norm(rho-ref)),"nd_spread":float(np.ptp([x["nd"] for x in members])),
                 "D_averaged":float(np.mean([x["D"] for x in members])),"spin_magnitude_of_average":b.summarize(rho)["spin_magnitude"]}
            out["averages"].append(row);print(json.dumps(row),flush=True)
            out["seconds"]=time.monotonic()-t
            (b.ROOT/"fine_star_convergence.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
if __name__=="__main__":main()

