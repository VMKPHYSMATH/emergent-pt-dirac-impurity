"""Double EP-loop sampling; saddle is re-solved at every point."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parent;KR=ROOT.parent/'finite_u_kr'
sys.path.insert(0,str(KR/'scripts'));sys.path.insert(0,str(KR/'kr_package/src'))
import kr_certificate as c
rows=[]
for name,path,control,h1,h2 in [('low_fill_U002','results/cert_mu0_VO_U0.02.json','V_O',1e-4,1e-2),('low_fill_U02','results/cert_mu0_VO_U0.2.json','V_O',1e-4,1e-2),('W_U00025','results/cert_cont_W_U0.0025.json','W',1e-4,1e-3),('n1_U00025','kr_package/results/ep_n1_U0025.json','V_O',5e-6,1e-3)]:
    data=json.loads((KR/path).read_text());ep=data.get('ep',data)
    sm,records=c.certificate(ep,control,h1,h2,samples=256,order=64)
    sv=ep['singular_values'];rank_ok=sv[1]<1e-12 and sv[0]>1e-6
    passed=bool(sm['certified'] and sm['minimum_F_zz']>1e-2 and rank_ok)
    rows.append({'name':name,'summary':sm,'center_singular_values':sv,'passed':passed})
    (ROOT/'results'/f'loop256_{name}.json').write_text(json.dumps({'ep':ep,'summary':sm,'records':records},indent=2))
    print(name,sm,'passed',passed,flush=True)
out={'points':rows,'passed':all(x['passed'] for x in rows)}
(ROOT/'results/loop_resolution.json').write_text(json.dumps(out,indent=2))
raise SystemExit(0 if out['passed'] else 1)
