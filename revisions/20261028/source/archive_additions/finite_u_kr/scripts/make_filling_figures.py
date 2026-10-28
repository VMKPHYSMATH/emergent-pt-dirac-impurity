"""Render main Fig. 7 and SM Fig. S5 from unchanged archived KR results.

Main: former S5(a-c), mean-filling-one EP and constrained continuation.
Supplement: former S5(d), a separate occupied path without an EP search.
No saddle or pole calculation is performed here.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
PK=HERE.parent/'kr_package/results'
OUT=HERE.parent/'fig'
OUT.mkdir(exist_ok=True)
SOURCE=HERE.parents[2]
G0=0.0025
C1,C2,C3,CK='#1f4e8c','#c8511b','#2e7d32','#333333'
mpl.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral','DejaVu Serif'],
 'mathtext.fontset':'stix','font.size':9,'axes.labelsize':9.6,'axes.titlesize':9.4,
 'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.2,'axes.linewidth':.9,
 'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
 'pdf.fonttype':42,'ps.fonttype':42})

val=json.loads((PK/'validation.json').read_text())
rec=val['exchange']['records']
zep=complex(*val['selected']['z'])
P=np.array([[complex(*p) for p in r['poles']] for r in rec])

def branch(arch,ext,fold,key):
    a=json.loads((PK/arch).read_text())
    e=[r for r in json.loads((HERE.parent/'results'/ext).read_text()) if r.get('success',True) and not r.get('failed_attempt')]
    f=[r for r in json.loads((HERE.parent/'results'/fold).read_text()) if r['ok']]
    up=[(r[key],r['U']) for r in a]+[(r[key],r['U']) for r in e[1:]]
    down=[(r['par'],r['U']) for r in f]
    return np.array(up),np.array(down)

upV,dnV=branch('ep_n1.json','ep_n1_extended_VO.json','fold_n1_VO.json','auxiliary_vertex')
upW,dnW=branch('ep_n1_W.json','ep_n1_extended_W.json','fold_n1_W.json','impurity_hybridization')
fig,axs=plt.subplots(1,3,figsize=(7.4,2.85),layout='constrained')
ax=axs[0]
for j,(color,lab) in enumerate([(C1,'tracked $z_1$'),(C2,'tracked $z_2$')]):
    w=1e5*(P[:,j]-zep)
    ax.plot(w.real,w.imag,color=color,lw=1.3,ls='-' if j==0 else '--',label=lab)
    k=len(w)//3
    ax.annotate('',xy=(w[k+1].real,w[k+1].imag),xytext=(w[k].real,w[k].imag),
                arrowprops=dict(arrowstyle='-|>',color=color,lw=1,mutation_scale=8))
    ax.plot(w[0].real,w[0].imag,'o',mfc='white',mec=color,ms=4.5,zorder=5)
    ax.plot(w[-1].real,w[-1].imag,'s',color=color,ms=3.1,zorder=6)
ax.plot(0,0,'*',color=CK,ms=8,zorder=7)
ax.set_xlabel(r'$10^5\,\mathrm{Re}(z-z_{\rm EP})$')
ax.set_ylabel(r'$10^5\,\mathrm{Im}(z-z_{\rm EP})$')
ax.set_title('(a) Pole exchange\n'+r'$\langle n_d\rangle=1,\ U=\bar\Gamma_0$')
ax.legend(loc='lower left',frameon=False,handlelength=1.8,borderpad=.1,labelspacing=.25)

for ax,up,dn,xlabel,title in [(axs[1],upV,dnV,r'$V_O$',r'(b) Fixed $W=0.10$'),
                             (axs[2],upW,dnW,r'$W$',r'(c) Fixed $V_O$')]:
    ax.plot(up[:,0],up[:,1]/G0,'-o',color=C1,ms=2.8,lw=1.2,label='$U$ continuation')
    ax.plot(dn[:,0],dn[:,1]/G0,'--s',color=C2,ms=2.3,lw=1.1,label='control continuation')
    i=np.argmax(dn[:,1]);fold=dn[i,1]/G0
    ax.axhline(fold,color='.5',lw=.7,ls=':')
    ax.text(.05,.58,r'$U_c/\bar\Gamma_0=%.2f$'%fold,transform=ax.transAxes,fontsize=8)
    ax.set_xlabel(xlabel);ax.set_ylabel(r'$U/\bar\Gamma_0$')
    ax.set_title(title+'\n'+r'$\langle n_d\rangle=1$ constraint')
    ax.legend(loc='upper left',frameon=True,facecolor='white',edgecolor='none',
              framealpha=1,handlelength=1.6,borderpad=.2,labelspacing=.3)
    ins=ax.inset_axes([.58,.08,.39,.25])
    sel=up[:,1]/G0>.97*fold
    ins.plot(up[sel,0],up[sel,1]/G0,'-o',color=C1,ms=2,lw=1)
    ins.plot(dn[:,0],dn[:,1]/G0,'--s',color=C2,ms=1.8,lw=1)
    ins.axhline(fold,color='.5',lw=.6,ls=':')
    ins.tick_params(labelsize=6.2,pad=1)
    ins.margins(x=.12,y=.10)
    if xlabel==r'$V_O$':
        ins.set_xticks([.86,.88]);ins.set_yticks([1.20,1.23])
        ax.invert_xaxis();ins.invert_xaxis()
    else:
        ins.set_xticks([.184,.186]);ins.set_yticks([3.14,3.18])
fig.savefig(OUT/'Fig6_kr_filling_one.pdf')
fig.savefig(OUT/'Fig6_kr_filling_one.png',dpi=220)
plt.close(fig)

occ=json.loads((PK/'occupied_gate_path.json').read_text())
U=np.array([r['U'] for r in occ]);nd=np.array([r['n_d'] for r in occ])
D=np.array([r['double_occupancy'] for r in occ])
mn=np.array([min(r['density_eigenvalues']) for r in occ])
fig,ax=plt.subplots(figsize=(4.5,3.0),layout='constrained')
ax.semilogy(U/G0,D,'-',color=C1,lw=1.4,label=r'$D_d$')
ax.semilogy(U/G0,mn,'--',color=C2,lw=1.4,label='minority density eigenvalue')
ax.semilogy(U/G0,1-nd,':',color=C3,lw=1.6,label=r'$1-\langle n_d\rangle$')
ax.set_xlabel(r'$U/\bar\Gamma_0$');ax.set_ylabel('Dimensionless occupation measures')
ax.set_title(r'Occupied path, $\mu=0.012$ (no EP search)')
ax.legend(loc='upper right',frameon=False)
fig.savefig(OUT/'SFig5_kr_occupied_path.pdf')
fig.savefig(OUT/'SFig5_kr_occupied_path.png',dpi=220)
plt.close(fig)

for name in ['Fig6_kr_filling_one.pdf','SFig5_kr_occupied_path.pdf']:
    (SOURCE/name).write_bytes((OUT/name).read_bytes())
print('Rendered main Fig. 7 and SM Fig. S5 from unchanged numerical data.')
