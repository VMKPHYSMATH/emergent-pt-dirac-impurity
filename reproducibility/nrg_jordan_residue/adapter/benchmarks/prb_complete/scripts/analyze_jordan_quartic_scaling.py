#!/usr/bin/env python3
"""Branch-tracked Jordan/quartic scaling analysis.

The script uses the full 2x2 transition residue matrices and physical pole
columns exported by PTDiracNHNRG.  The exported pole columns already include
the NRG iteration scale; no further Wilson rescaling is applied.

For each model, detuning, and particle-addition/removal sector it:
  1. constructs supported pole-pair candidates at a fixed NRG iteration;
  2. tracks one branch continuously in U;
  3. decomposes B = (W+ - W-)(z+ - z-)/2 into Pauli components;
  4. extrapolates y=s^2 to delta_coh -> 0;
  5. fits |y(U)-y(0)| ~ U^p and evaluates a measured-baseline quartic ratio.

The measured-baseline quartic residual is
    Q = y(U)^2 - y(0) y(U),
which avoids silently identifying the NRG U=0 discriminant with the bare local
one.  Ratios are reported both with the raw control beta0 and with the physical
local NH matrix element Gamma_PT, since those are distinct in this adapter.
"""
from __future__ import annotations

import argparse
import itertools
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

KEYS = ("11", "12", "21", "22")


@dataclass(frozen=True)
class Params:
    beta0: float
    U: float
    eps_d: float
    delta_eff: float
    gamma_pt: float
    delta_coh: float
    soc_lambda: float
    F_lambda: float
    scale_lambda: float


def fnum(text: str, default=math.nan):
    try:
        return float(text.strip())
    except Exception:
        return default


def parse_summary(path: Path) -> Params:
    d = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for item in raw.split(";"):
            if "=" in item:
                k, v = item.split("=", 1)
                d[k.strip()] = fnum(v)
    return Params(
        beta0=d["beta0"], U=d["U"], eps_d=d["eps_d"],
        delta_eff=d["Delta_eff"], gamma_pt=d["Gamma_PT"],
        delta_coh=d["Delta_coh"], soc_lambda=d.get("soc_lambda", 0.0),
        F_lambda=d.get("F_lambda", 1.0), scale_lambda=d.get("Lambda", math.nan),
    )


def matrix(row: pd.Series) -> np.ndarray:
    prefix = "add" if int(row.charge) == int(row.ground_charge) + 1 else "rem"
    return np.array([
        complex(row[f"{prefix}{k}_real"], row[f"{prefix}{k}_imag"]) for k in KEYS
    ], dtype=np.complex128).reshape(2, 2)


def components(B):
    return (
        0.5*np.trace(B),
        0.5*(B[0,1] + B[1,0]),
        (B[1,0] - B[0,1])/(2j),
        0.5*(B[0,0] - B[1,1]),
    )


def candidates(df, params, args):
    out=[]
    for sector, g0 in df.groupby("sector"):
        g = g0.sort_values(["matrix_support_abs", "matrix_rank"], ascending=[False, True]).head(args.pair_pool)
        mx = float(g.matrix_support_abs.max()) if len(g) else 0.0
        g = g[g.matrix_support_abs >= args.support_fraction*mx]
        rows=[r for _,r in g.iterrows()]
        N=np.array([[params.delta_eff,1j*params.gamma_pt],[1j*params.gamma_pt,-params.delta_eff]],complex)
        nn=np.linalg.norm(N)
        for a,b in itertools.combinations(rows,2):
            za=complex(a.pole_real,a.pole_imag); zb=complex(b.pole_real,b.pole_imag)
            # Deterministic orientation. B itself is invariant under pair swap.
            if (za.real,za.imag) <= (zb.real,zb.imag):
                rm,rp,zm,zp=a,b,za,zb
            else:
                rm,rp,zm,zp=b,a,zb,za
            Wm=matrix(rm); Wp=matrix(rp)
            dz=zp-zm; s=0.5*dz; B=0.5*(Wp-Wm)*dz
            b0,bx,by,bz=components(B)
            bn=float(np.linalg.norm(B)); support=min(float(rm.matrix_support_abs),float(rp.matrix_support_abs))
            align=float(abs(np.vdot(N,B))/(nn*bn)) if nn>0 and bn>0 else math.nan
            tf=float(abs(np.trace(B))/(math.sqrt(2)*bn)) if bn>0 else math.nan
            out.append(dict(
                sector=int(sector), rank_minus=int(rm.matrix_rank), rank_plus=int(rp.matrix_rank),
                z_minus=zm,z_plus=zp,z_center=0.5*(zp+zm),s=s,y=s*s,B=B,
                B0=b0,Bx=bx,By=by,Bz=bz,Bnorm=bn,alignment=align,trace_fraction=tf,
                support=support,max_residual=max(float(rm.max_residual),float(rp.max_residual)),
                biorth_error=max(float(rm.biorth_error),float(rp.biorth_error)),
            ))
    return out


def orient_s(s, prev):
    if prev is None: return s
    return s if abs(s-prev) <= abs(-s-prev) else -s


def normalized_matrix_distance(A,B):
    na=np.linalg.norm(A); nb=np.linalg.norm(B)
    if na==0 or nb==0: return 1.0
    # B is phase-sensitive as a residue matrix. Allow only a sign ambiguity inherited
    # from an occasional pair orientation convention.
    a=A/na; b=B/nb
    return float(min(np.linalg.norm(a-b),np.linalg.norm(a+b)))


def choose_seed(cands):
    reliable=[c for c in cands if np.isfinite(c["alignment"]) and c["support"]>0]
    if not reliable: return None
    maxsup=max(c["support"] for c in reliable)
    # No alignment cut in selection. Pick a near-degenerate, well-supported pair.
    return min(reliable,key=lambda c:(abs(c["s"])/(1e-14+math.sqrt(maxsup)), -c["support"]))


def choose_next(cands, prev):
    if not cands: return None
    scale_z=max(abs(prev["z_center"]),abs(prev["s"]),1e-4)
    scale_s=max(abs(prev["s"]),1e-5)
    def cost(c):
        ss=orient_s(c["s"],prev["s"])
        return (
            1.0*abs(c["z_center"]-prev["z_center"])/scale_z
            +1.5*abs(ss-prev["s"])/scale_s
            +0.75*normalized_matrix_distance(c["B"],prev["B"])
            +0.10*abs(math.log10((c["support"]+1e-300)/(prev["support"]+1e-300)))
        )
    got=min(cands,key=cost).copy()
    got["s"]=orient_s(got["s"],prev["s"]); got["y"]=got["s"]**2
    got["tracking_cost"]=cost(got)
    return got


def load_runs(root, iteration, args):
    runs=[]
    for summary in root.rglob("RUN_SUMMARY.txt"):
        run=summary.parent
        csv=run/"impurity_transition_residue_matrix.csv"
        if not csv.is_file(): continue
        p=parse_summary(summary)
        d=pd.read_csv(csv)
        d=d[d.iteration==iteration].copy()
        if d.empty: continue
        d["sector"]=d.charge.astype(int)-d.ground_charge.astype(int)
        runs.append((run,p,candidates(d,p,args)))
    return runs


def model_name(p):
    return "scalar" if abs(p.soc_lambda)<1e-14 else f"soc_lambda_{p.soc_lambda:.6f}"


def track(runs, args):
    grouped={}
    for run,p,c in runs:
        grouped.setdefault((model_name(p),p.delta_coh),[]).append((p.U,run,p,c))
    rows=[]
    models=sorted({key[0] for key in grouped})
    for model in models:
        deltas=sorted([key[1] for key in grouped if key[0]==model], reverse=True)
        all_sectors=sorted({cc["sector"] for delta in deltas for *_,cs in grouped[(model,delta)] for cc in cs})
        for sector in all_sectors:
            previous_delta_seed=None
            for delta in deltas:
                items=sorted(grouped[(model,delta)],key=lambda x:x[0])
                prev=None
                first_at_delta=None
                for U,run,p,cs in items:
                    avail=[x for x in cs if x["sector"]==sector]
                    if prev is None:
                        cur=choose_seed(avail) if previous_delta_seed is None else choose_next(avail,previous_delta_seed)
                    else:
                        cur=choose_next(avail,prev)
                    if cur is None: continue
                    if prev is None and previous_delta_seed is None:
                        cur=cur.copy(); cur["tracking_cost"]=0.0
                    if first_at_delta is None:
                        first_at_delta=cur.copy()
                    b0,bx,by,bz=cur["B0"],cur["Bx"],cur["By"],cur["Bz"]
                    reliable=(cur["max_residual"]<=args.max_residual and
                              cur["biorth_error"]<=args.max_biorth_error and
                              cur["support"]>=args.min_support)
                    rows.append(dict(
                        run=str(run),model=model,delta_coh=delta,U=U,sector=sector,
                        beta0=p.beta0,Gamma_PT=p.gamma_pt,Delta_eff=p.delta_eff,
                        F_lambda=p.F_lambda,soc_lambda=p.soc_lambda,
                        z_center_real=cur["z_center"].real,z_center_imag=cur["z_center"].imag,
                        s_real=cur["s"].real,s_imag=cur["s"].imag,s_abs=abs(cur["s"]),
                        y_real=cur["y"].real,y_imag=cur["y"].imag,y_abs=abs(cur["y"]),
                        B_frobenius=cur["Bnorm"],alignment=cur["alignment"],
                        trace_fraction=cur["trace_fraction"],support=cur["support"],
                        max_residual=cur["max_residual"],biorth_error=cur["biorth_error"],
                        reliable=bool(reliable),tracking_cost=cur["tracking_cost"],
                        B0_real=b0.real,B0_imag=b0.imag,Bx_real=bx.real,Bx_imag=bx.imag,
                        By_real=by.real,By_imag=by.imag,Bz_real=bz.real,Bz_imag=bz.imag,
                        Bx_abs=abs(bx),By_abs=abs(by),Bz_abs=abs(bz),
                        nilpotent_mismatch=abs(bx-1j*bz)/(abs(bx)+abs(bz)+1e-300),
                        rank_minus=cur["rank_minus"],rank_plus=cur["rank_plus"],
                    ))
                    prev=cur
                if first_at_delta is not None:
                    previous_delta_seed=first_at_delta
    return pd.DataFrame(rows)

def extrapolate_zero_delta(tracked):
    rows=[]
    for (model,sector,U),g in tracked.groupby(["model","sector","U"]):
        g=g[g.reliable].sort_values("delta_coh")
        if len(g)<2: continue
        x=g.delta_coh.to_numpy(float)
        # Linear extrapolation of complex y and B components. y=s^2 is the
        # analytic variable at a second-order EP; s itself is not.
        def intercept(col):
            y=g[col].to_numpy(float)
            deg=1 if len(g)>=2 else 0
            return float(np.polyfit(x,y,deg)[-1])
        yr,yi=intercept("y_real"),intercept("y_imag")
        bx=complex(intercept("Bx_real"),intercept("Bx_imag"))
        bz=complex(intercept("Bz_real"),intercept("Bz_imag"))
        representative=g.iloc[g.delta_coh.argmin()]
        rows.append(dict(
            model=model,sector=int(sector),U=float(U),
            y0_real=yr,y0_imag=yi,y0_abs=abs(complex(yr,yi)),
            s0_abs=math.sqrt(abs(complex(yr,yi))),
            Bx0_real=bx.real,Bx0_imag=bx.imag,Bx0_abs=abs(bx),
            Bz0_real=bz.real,Bz0_imag=bz.imag,Bz0_abs=abs(bz),
            nilpotent_mismatch0=abs(bx-1j*bz)/(abs(bx)+abs(bz)+1e-300),
            beta0=float(representative.beta0),Gamma_PT=float(representative.Gamma_PT),
            F_lambda=float(representative.F_lambda),soc_lambda=float(representative.soc_lambda),
            n_delta=len(g),max_tracking_cost=float(g.tracking_cost.max()),
            max_residual=float(g.max_residual.max()),max_biorth_error=float(g.biorth_error.max()),
            min_support=float(g.support.min()),median_alignment=float(g.alignment.median()),
            median_trace_fraction=float(g.trace_fraction.median()),
        ))
    out=pd.DataFrame(rows)
    if out.empty:return out
    enriched=[]
    for (model,sector),g in out.groupby(["model","sector"]):
        g=g.sort_values("U").copy()
        zero=g.iloc[np.argmin(abs(g.U.to_numpy()))]
        ybare=complex(zero.y0_real,zero.y0_imag)
        for _,r in g.iterrows():
            y=complex(r.y0_real,r.y0_imag); U=float(r.U)
            dy=y-ybare; Q=y*y-ybare*y
            d=r.to_dict(); d.update(
                ybare_real=ybare.real,ybare_imag=ybare.imag,
                delta_y_real=dy.real,delta_y_imag=dy.imag,delta_y_abs=abs(dy),
                Q_real=Q.real,Q_imag=Q.imag,Q_abs=abs(Q),
                C_beta0_abs=(abs(Q)/(U*U*r.beta0*r.beta0*r.F_lambda)) if U>0 and r.F_lambda>0 else math.nan,
                C_gamma_abs=(abs(Q)/(U*U*r.Gamma_PT*r.Gamma_PT*r.F_lambda)) if U>0 and r.F_lambda>0 else math.nan,
                C_beta0_phase=np.angle(Q) if U>0 else math.nan,
            ); enriched.append(d)
    return pd.DataFrame(enriched)


def fit_power(g,col="delta_y_abs"):
    h=g[(g.U>0)&(g[col]>0)&np.isfinite(g[col])].sort_values("U")
    if len(h)<3:return math.nan,math.nan,0
    # Favor the controlled small-U half of the scan.
    h=h.iloc[:max(3,int(math.ceil(len(h)*0.65)))]
    x=np.log(h.U.to_numpy()); y=np.log(h[col].to_numpy())
    p,loga=np.polyfit(x,y,1)
    pred=loga+p*x
    ssr=np.sum((y-pred)**2); sst=np.sum((y-y.mean())**2)
    r2=1-ssr/sst if sst>0 else math.nan
    return float(p),float(r2),len(h)


def plots(ext,out):
    if plt is None or ext.empty:return
    # |Delta y| power law
    fig,ax=plt.subplots(figsize=(7.2,4.8))
    for (model,sector),g in ext.groupby(["model","sector"]):
        h=g[(g.U>0)&(g.delta_y_abs>0)].sort_values("U")
        ax.loglog(h.U,h.delta_y_abs,marker="o",label=f"{model}, sector {sector:+d}")
    ax.set_xlabel("U");ax.set_ylabel(r"$|s^2(U)-s^2(0)|_{\delta\to0}$")
    ax.grid(True,which="both",alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(out/"Delta_s2_vs_U.png",dpi=180);fig.savefig(out/"Delta_s2_vs_U.pdf");plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,4.8))
    for (model,sector),g in ext.groupby(["model","sector"]):
        h=g[(g.U>0)&np.isfinite(g.C_beta0_abs)].sort_values("U")
        ax.semilogx(h.U,h.C_beta0_abs,marker="o",label=f"{model}, sector {sector:+d}")
    ax.set_xlabel("U");ax.set_ylabel(r"$|Q|/(U^2\beta_0^2F)$")
    ax.grid(True,which="both",alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(out/"Quartic_ratio_beta0.png",dpi=180);fig.savefig(out/"Quartic_ratio_beta0.pdf");plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,4.8))
    for (model,sector),g in ext.groupby(["model","sector"]):
        h=g[g.U>0].sort_values("U")
        ax.loglog(h.U,h.Bx0_abs,marker="o",label=f"|Bx| {model} {sector:+d}")
        ax.loglog(h.U,h.Bz0_abs,marker="s",linestyle="--",label=f"|Bz| {model} {sector:+d}")
    ax.set_xlabel("U");ax.set_ylabel("extrapolated Jordan residue component")
    ax.grid(True,which="both",alpha=.25);ax.legend(fontsize=7,ncol=2);fig.tight_layout()
    fig.savefig(out/"Jordan_components_vs_U.png",dpi=180);fig.savefig(out/"Jordan_components_vs_U.pdf");plt.close(fig)


def report(tracked,ext,out,args):
    lines=["# Branch-tracked Jordan/quartic scaling audit","",
           f"Fixed NRG iteration: **{args.iteration}**.","",
           "The analysis uses the exported `pole_real/pole_imag` columns, which already contain the physical NRG iteration scale.",
           "It tracks full 2x2 residue-matrix pole pairs in U and extrapolates the analytic variable `y=s^2` linearly to `delta_coh -> 0`.","",
           "## Reliability and scaling summary","",
           "| model | sector | U points | fitted p in |Delta s^2|~U^p | R2 | median alignment | median trace fraction | max biorth error |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    if ext.empty:
        lines += ["| none | 0 | 0 | nan | nan | nan | nan | nan |",""]
    else:
        for (model,sector),g in ext.groupby(["model","sector"]):
            p,r2,n=fit_power(g)
            lines.append(f"| {model} | {int(sector):+d} | {len(g)} | {p:.6g} ({n} fit points) | {r2:.5g} | {g.median_alignment.median():.6g} | {g.median_trace_fraction.median():.6g} | {g.max_biorth_error.max():.3g} |")
        lines += ["","## Interpretation gates","",
                  "1. Generic Jordan-vector perturbation at the exact EP requires `p ~= 1`, equivalently `|s| ~ U^(1/2)` after subtracting the U=0 discriminant.",
                  "2. The full quartic claim additionally requires `C_beta0_abs = |Q|/(U^2 beta0^2 F)` to approach a nonzero plateau with stable phase.",
                  "3. `C_gamma_abs` is also exported because in this adapter `beta0` is a control parameter whereas `Gamma_PT` is the physical local NH matrix element.",
                  "4. The SOC-overlap mode remains an effective control model, not the microscopic k^2 +/- lambda k bath.",""]
    (out/"JORDAN_QUARTIC_SCALING_AUDIT.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("output_root",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--iteration",type=int,default=5)
    ap.add_argument("--pair-pool",type=int,default=12)
    ap.add_argument("--support-fraction",type=float,default=.02)
    ap.add_argument("--max-residual",type=float,default=1e-8)
    ap.add_argument("--max-biorth-error",type=float,default=1e-6)
    ap.add_argument("--min-support",type=float,default=1e-4)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    runs=load_runs(args.output_root.resolve(),args.iteration,args)
    tracked=track(runs,args)
    tracked.to_csv(args.out/"branch_tracked_pairs.csv",index=False)
    ext=extrapolate_zero_delta(tracked)
    ext.to_csv(args.out/"zero_detuning_scaling.csv",index=False)
    plots(ext,args.out);report(tracked,ext,args.out,args)
    print(f"runs={len(runs)} tracked_rows={len(tracked)} extrapolated_rows={len(ext)}")
    print(args.out/"JORDAN_QUARTIC_SCALING_AUDIT.md")

if __name__=="__main__": main()
