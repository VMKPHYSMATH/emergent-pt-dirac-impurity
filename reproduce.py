#!/usr/bin/env python3
"""Verify the shipped snapshot or rerun its declared numerical checks."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'revisions/20261028/source'
AUD=SRC/'archive_additions/slave_boson_audit_20261024'
KR=SRC/'archive_additions/finite_u_kr'
ROOT_FILES=['README.md', 'RELEASE_NOTES.md', 'VALIDATION_REPORT.md', 'UPLOAD_GUIDE.md', 'Slave_Boson_Convergence_Audit.pdf', 'LICENSE', 'LICENSE-DATA', 'CITATION.cff', 'CITATION.bib', 'requirements.txt', 'environment.json', 'zenodo_metadata_notes.json', 'reproduce.py']

def release_files():
    files=[ROOT/p for p in ROOT_FILES if (ROOT/p).is_file()]
    files += [p for p in (ROOT/'revisions/20261028').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.log','.blg','.out','.fls','.fdb_latexmk')]
    return sorted(files)

def hashes(write=False):
    manifest=ROOT/'SHA256SUMS.txt'
    if write:
        rows=[f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}' for p in release_files()]
        manifest.write_text('\n'.join(rows)+'\n')
        (ROOT/'MANIFEST.txt').write_text('Current candidate files only; historical repository files are outside this manifest.\n'+'\n'.join(str(p.relative_to(ROOT)) for p in release_files())+'\n')
        print(f'Wrote hashes for {len(rows)} files.');return
    if not manifest.exists():raise SystemExit('Missing SHA256SUMS.txt')
    count=0
    for line in manifest.read_text().splitlines():
        digest,rel=line.split('  ',1);p=ROOT/rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise SystemExit(f'Hash mismatch: {rel}')
        count+=1
    print(f'PASS: {count} reference-file hashes.')

def require(ok,msg):
    if not ok:raise SystemExit('FAIL: '+msg)
    print('PASS:',msg)

def gates():
    load=lambda p:json.loads(p.read_text())
    a=load(AUD/'results/kr_paths.json');rows=[x for p in a['paths'].values() for x in p['checked']]
    require(len(rows)==98 and all(x['passed'] and x['order96_held_state_residual']<1e-8 for x in rows),'98 accepted KR points and finer-grid residuals')
    require(all(x['density_change_1e5_to1e6']<1e-9 and x['force_change_1e5_to1e6']<1e-9 for x in a['tail_checks']),'KR tail cutoff')
    c=load(AUD/'results/coleman.json')
    require(c['passed'] and all(x['relative_r_error']<2e-6 for x in c['grid'] if x['refinement']==4),'scalar r versus independent finite-cutoff reference')
    require(all(x['converged']==1 for x in c['production_solver']),'corrected scalar production solver')
    s=load(AUD/'results/stationary_coleman.json');require(s['passed'],'stationary-bath Coleman integration and Matsubara sums')
    f=load(AUD/'results/variational_functional.json');require(all(x['derivative_discrepancy']<1e-8 for p in f['points'] for x in p['records']),'independent log-determinant functional derivatives')
    v=load(KR/'kr_package/results/validation.json')
    require(v['selected_stability']['local_minimum'] and all(x['success'] for x in v['multiple_seeds']),'selected KR local stability and six perturbed seeds')
    require(v['exchange']['all_two_cut'] and abs(abs(v['exchange']['half_winding'])-.5)<1e-6,'128-sample filling-one pole exchange')
    loops=load(AUD/'results/loop_resolution.json')
    require(all(x['passed'] and x['summary']['failures']==0 and x['summary']['minimum_F_zz']>0 and x['summary']['endpoint_permutation']==[1,0] for x in loops['points']),'four 256-sample EP loops, rank and nonzero second derivative')
    cb=load(KR/'results/compare_bath.json')
    require(all(r['saddles_ok'] and r['ep_residual']<1e-9 and r['last_step']<1e-10 for r in cb['kr_chain']),'four finite-chain KR comparison points (ED values are reference inputs)')
    u0=load(SRC/'validation/recheck_20261024/u0/full_k_two_cut_ep_summary.json')
    require(u0['ep']['certified'] and u0['local_topology']['certified'],'U=0 full-radial-bath EP certificate')

    controls=ROOT/'revisions/20261028/checks_20261028'
    pt=load(controls/'pt_interacting_control/results/summary.json')
    require(pt['number_of_exact_cases']==14 and pt['all_selected_blocks_have_nullities_1_2'] and pt['all_loop_exchanges'] and pt['maximum_residue_coefficient_error']<1e-12, '14 exact scalar-bath control cases (separate model)')
    bath=load(controls/'optimized_bath/summary.json')
    require(bath['checks']['largest_sumrule_error']<1e-10 and bath['checks']['minimum_sampled_spectral_eigenvalue']>0, 'optimized-bath sampled sum-rule and positivity checks')
    drift=bath['families']['occupied_exact']['Sigma_change_10_12_over_Gamma']
    require(abs(drift-0.3298602705340236)<1e-10, 'reported self-energy nonconvergence diagnostic is retained')

def run(mode):
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    logdir=ROOT/'run_logs';logdir.mkdir(exist_ok=True)
    tasks=[('kr_paths',AUD/'check_kr_paths.py',[],AUD),('kr_functional',AUD/'check_variational_functional.py',[],AUD),('u0',SRC/'archive_additions/full_k_two_cut_ep_certificate.py',['--soc','.05','--out',str(SRC/'validation/recheck_20261024/u0')],SRC),('contact_algebra',SRC/'archive_additions/verify_finiteU_contact_algebra.py',[],SRC)]
    pt=ROOT/'revisions/20261028/checks_20261028/pt_interacting_control'
    tasks.append(('pt_control',pt/'verify_control.py',[],pt))
    if mode=='full':tasks += [('kr_validation',KR/'kr_package/src/validate.py',[],KR/'kr_package'),('kr_certificates',KR/'scripts/kr_certificate.py',[],KR/'scripts'),('kr_chain_comparison',KR/'scripts/compare_bath.py',[],KR/'scripts'),('fold',SRC/'archive_additions/final_audit_20261019/recheck_new_results.py',[],SRC),('beyond_fold',SRC/'archive_additions/final_audit_20261019/check_beyond_fold.py',[],SRC),('loop_resolution',AUD/'check_loop_resolution.py',[],AUD),('coleman',AUD/'check_coleman.py',[],AUD),('stationary_coleman',AUD/'check_stationary_coleman.py',[],AUD)]
    for name,script,args,cwd in tasks:
        print('Running',name,flush=True)
        with (logdir/(name+'.txt')).open('w') as log:
            result=subprocess.run([sys.executable,str(script),*args],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(f'FAIL: {name}; see run_logs/{name}.txt')
    gates()
    print('Completed',mode,'rerun. Reference hashes describe pre-rerun bytes; inspect changes before regenerating hashes.')

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for opt in ('verify-stored','quick','full','write-hashes'):g.add_argument('--'+opt,action='store_true')
    a=p.parse_args()
    if a.write_hashes:hashes(True)
    elif a.verify_stored:hashes();gates()
    else:run('full' if a.full else 'quick')
