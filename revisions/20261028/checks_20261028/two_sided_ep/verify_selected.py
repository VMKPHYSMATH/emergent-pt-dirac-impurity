#!/usr/bin/env python3
"""Selected EP topology and bath refinement checks; no manuscript edits."""
import argparse, json, time, traceback
from pathlib import Path
import two_sided_ep as t

ap=argparse.ArgumentParser()
ap.add_argument('--name',required=True)
ap.add_argument('--source',required=True)
ap.add_argument('--U',type=float,required=True)
ap.add_argument('--mu',type=float,default=.012)
ap.add_argument('--N',type=int,default=8)
ap.add_argument('--zs',default='.5')
ap.add_argument('--nodes',type=int,default=512)
ap.add_argument('--lam',type=float,default=1.5)
ap.add_argument('--topology',action='store_true')
a=ap.parse_args()
source=json.loads((t.ROOT/'results'/a.source).read_text())
if 'results' in source:
    row=next(r for r in source['results'] if abs(r['settings']['U']-a.U)<1e-12)
    seed=row['solution']['center']
else:seed=source['solution']['center']
eng=t.Engine(a.U,a.mu,a.N,tuple(float(s) for s in a.zs.split(',')),a.nodes,a.lam)
out={'settings':eng.describe(),'source_seed':seed,'scope':'finite-chain self-energy embedding; no interacting continuum claim'}
target=t.ROOT/'results'/(a.name+'.json')
try:
    out['solution']=t.center(eng,seed,maxeval=160)
    t.save(target,out)
    print(a.name,'center',out['solution']['center'],'pass',out['solution']['center_numerical_pass'],flush=True)
    if a.topology and out['solution']['center_numerical_pass']:
        out['two_sided']=t.sided(eng,out['solution']['center'])
        t.save(target,out)
        print('two-sided',out['two_sided']['center_difference'],flush=True)
        out['loops']=[]
        for radius,samples in [(1.,64),(.5,128)]:
            loop=t.loop(eng,out['solution']['center'],radius,samples)
            out['loops'].append(loop);t.save(target,out)
            print('loop',radius,samples,loop['pass'],loop['winding'],loop['half_winding'],flush=True)
except Exception as exc:
    out['error']=str(exc);out['traceback']=traceback.format_exc()
    print(out['traceback'],flush=True)
out['state_builds']=eng.builds
t.save(target,out)
