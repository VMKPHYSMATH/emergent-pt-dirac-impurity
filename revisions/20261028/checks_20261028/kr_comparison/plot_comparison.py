#!/usr/bin/env python3
"""Plot the five computed comparison points; connecting lines guide the eye."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
results=json.loads((root/"comparison_results.json").read_text())["comparison"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.labelsize":11,
                     "legend.fontsize":9,"pdf.fonttype":42,"ps.fonttype":42})
fig,axs=plt.subplots(1,3,figsize=(11,3.5),layout="constrained")
styles=[("ED_same_chain","Exact finite chain","#b24428","o","-"),
        ("KR_same_chain","KR, same finite chain","#246a9b","s","--"),
        ("KR_continuum","KR, continuum bath","#535659","^",":")]
for i,(ax,key,label) in enumerate(zip(
        axs,["nd","D","spin_magnitude"],
        [r"$n_d$",r"$D=\langle n_\uparrow n_\downarrow\rangle$",r"$|\langle\mathbf{S}_d\rangle|$"])):
    for method,legend,color,marker,ls in styles:
        ax.plot([x["U"] for x in results],[x[method][key] for x in results],
                color=color,marker=marker,ms=6 if method=="KR_same_chain" else 5,
                markerfacecolor="none" if method=="KR_same_chain" else color,
                lw=1.3,ls=ls,label=legend)
    ax.set_xlabel(r"$U$");ax.set_ylabel(label);ax.set_xlim(-.004,.207)
    ax.grid(alpha=.18)
    ax.text(.90 if i==1 else .04,.93,"("+chr(97+i)+")",transform=ax.transAxes,fontweight="bold")
    if key=="D":ax.set_yscale("log")
axs[0].set_ylim(.82,1.008);axs[2].set_ylim(.02,.52)
fig.legend(*axs[0].get_legend_handles_labels(),loc="outside upper center",ncol=3,frameon=False)
fig.savefig(root/"ED_KR_comparison.pdf",bbox_inches="tight")
fig.savefig(root/"ED_KR_comparison.png",dpi=200,bbox_inches="tight")

