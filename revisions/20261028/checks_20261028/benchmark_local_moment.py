#!/usr/bin/env python3
"""Small feasibility benchmark; does not establish a continuum resonance EP."""
import json, sys, time, platform
from pathlib import Path
import numpy as np
import scipy

OUT = Path(__file__).resolve().parent
BASE = OUT / "solver"
sys.path.insert(0, str(BASE))
import filling_campaign as fc
import charge_solver as cs
import kept_filling as kf
import python_fullk_nrg_pilot as pilot
import exact_block_chain_response as exact

cfg=json.loads((BASE/"configs/submitted_baseline.json").read_text())
cfg["mapping"]["z_shifts"]=[0.5]
cfg["current_mu"]=0.012
U=0.2
ed=0.01517035-U/2
start=time.monotonic()
chains=fc.make_chains(cfg,cfg["current_mu"],OUT/"cache")
onsite,hopping,coupling,mapping=chains[0]
report={"scope":"finite-chain occupancy/truncation feasibility only; no EP search",
        "parameters":{"U":U,"epsilon_d_absolute":ed,"mu":cfg["current_mu"],"W":cfg["model"]["W"],"VO":cfg["model"]["VO"],"Lambda":cfg["mapping"]["Lambda"],"z_shift":0.5},
        "versions":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
        "mapping":mapping,"runs":[]}
def commit(item):
    report["runs"].append(item)
    (OUT/"feasibility_results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in item.items() if k not in ["observables","flow"]}),flush=True)
def summarize(obs):
    rho=cs.decode_matrix(obs["density_matrix"])
    eig=np.linalg.eigvalsh(rho)
    return {"nd":obs["nd"],"D":obs["double_occupancy"],"P1":obs["single_occupancy_probability"],
            "density_eigenvalues":eig.tolist(),"spin_magnitude":float((eig[1]-eig[0])/2)}
for N in [6, 8]:
    t=time.monotonic()
    h=exact.build_one_body(onsite,hopping,coupling,ed-cfg["current_mu"],cfg["model"]["W"],cfg["model"]["W"],N)
    state=cs.solve(h,U,50000,1e-11)
    obs=cs.observables(state)
    g,q,s=cs.response(state,0.003j)
    commit({"method":"exact_charge_sectors","bath_blocks":N,**summarize(obs),"sum_rule_error":float(np.linalg.norm(s-np.eye(2))),"seconds":time.monotonic()-t,"observables":obs})
for N,keep in [(6,64),(6,128),(6,256),(8,256),(8,512)]:
    t=time.monotonic()
    state,flow=kf.run(onsite,hopping,coupling,ed-cfg["current_mu"],U,N,keep)
    obs=kf.observables(state)
    g,q,s=pilot.response(state,0.003j)
    commit({"method":"kept_space_pilot","bath_blocks":N,"nkeep":keep,**summarize(obs),"sum_rule_error":float(np.linalg.norm(s-np.eye(2))),"seconds":time.monotonic()-t,"observables":obs,"flow":flow})
report["total_seconds"]=time.monotonic()-start
(OUT/"feasibility_results.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")

