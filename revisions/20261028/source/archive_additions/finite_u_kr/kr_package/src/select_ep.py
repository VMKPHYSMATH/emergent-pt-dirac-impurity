"""Select the reproducible n=1, U=.0025 benchmark from a continued branch."""
import json
from pathlib import Path
import numpy as np
import matrix_saddle as m
rows=json.load(open('results/ep_n1.json'))
r=min([r for r in rows if r['success']],key=lambda r:abs(r['U']-.0025))
s=m.Saddle(r['mu'],40)
r=m.joint_ep(s,.0025,np.array(r['y']),1.)
assert r['success']
Path('results/ep_n1_U0025.json').write_text(json.dumps(r,indent=2))
print('Selected EP:',r['z'],'n=',r['n_d'],'D=',r['double_occupancy'])
