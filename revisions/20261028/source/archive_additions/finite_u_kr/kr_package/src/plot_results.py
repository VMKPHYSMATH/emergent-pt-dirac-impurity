import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

v=json.load(open('results/validation.json'));ep=v['selected']
a=json.load(open('results/ep_n1.json'));a=[r for r in a if r['success'] and r['joint_residual_max']<1e-9]
b=json.load(open('results/occupied_gate_path.json'));b=[r for r in b if r['success']]
records=v['exchange']['records'];poles=np.array([[complex(*z) for z in r['poles']] for r in records]);zep=complex(*ep['z'])
Path('figures').mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.labelsize':10,'axes.titlesize':10,'legend.fontsize':8.3,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(7.1,6.1),layout='constrained')
colors=['#1f5b96','#c95824']
ax=axs[0,0]
for j in range(2):
    z=(poles[:,j]-zep)*1e5
    ax.plot(z.real,z.imag,color=colors[j],lw=1.8,label=f'Pole {j+1}')
    ax.scatter(z[0].real,z[0].imag,s=27,facecolor='white',edgecolor=colors[j],zorder=5)
    for idx in [27,85]:
        ax.annotate('',xy=(z[idx+4].real,z[idx+4].imag),xytext=(z[idx].real,z[idx].imag),arrowprops=dict(arrowstyle='->',color=colors[j],lw=1.4))
ax.scatter(0,0,marker='*',s=100,c='#20242b',zorder=7)
ax.annotate('EP',(0,0),xytext=(6,7),textcoords='offset points',fontsize=9)
ax.set(xlabel=r'$10^5\,\mathrm{Re}(z-z_{\mathrm{EP}})$',ylabel=r'$10^5\,\mathrm{Im}(z-z_{\mathrm{EP}})$',title='(a) Self-consistent pole exchange')
ax.set_aspect('equal',adjustable='datalim');ax.legend(loc='upper right',frameon=False)
ax=axs[0,1];U=np.array([r['U'] for r in a]);zz=np.array([complex(*r['z']) for r in a]);mu=np.array([r['mu'] for r in a])
ax.plot(U*1e3,zz.real*1e3,'o-',color=colors[0],ms=3,label=r'$\mathrm{Re}\,z_{\mathrm{EP}}$')
ax.plot(U*1e3,-zz.imag*1e3,'s--',color=colors[1],ms=3,label=r'$-\mathrm{Im}\,z_{\mathrm{EP}}$')
ax.set(xlabel=r'$10^3 U$',ylabel=r'Energy $\times\,10^3$',title=r'(b) Retuned EP branch, $n_d=1$')
ax.legend(frameon=False);ax.grid(alpha=.17)
ax=axs[1,0];U=np.array([r['U'] for r in b]);n=np.array([r['n_d'] for r in b]);d=np.array([r['double_occupancy'] for r in b])
ax.plot(U,n,color=colors[0],lw=1.8,label=r'$n_d$')
ax.plot(U,n-2*d,'--',color=colors[1],lw=1.8,label=r'$P_{\mathrm{single}}=n_d-2D$')
ax.set(xlabel=r'$U$',ylabel='Occupation / probability',ylim=(.45,1.02),title='(c) Occupied gate path')
ax.legend(frameon=False,loc='lower right');ax.grid(alpha=.17)
ax=axs[1,1]
ax.semilogy(U,d,color=colors[0],lw=1.8,label=r'$D=\langle n_\uparrow n_\downarrow\rangle$')
minor=np.array([r['density_eigenvalues'][0] for r in b]);ax.semilogy(U,minor,'--',color=colors[1],lw=1.8,label=r'Minority density eigenvalue')
ax.set(xlabel=r'$U$',ylabel='Probability',title='(d) Suppression and polarization')
ax.legend(frameon=False);ax.grid(alpha=.17)
fig.savefig('figures/Stationary_SB_Checks.pdf')
fig.savefig('figures/Stationary_SB_Checks.png',dpi=200)
plt.close(fig)
with open('results/benchmark_table.csv','w') as f:
    writer=csv.DictWriter(f,fieldnames=['case','U','n_d','D','P_single','mu','epsilon','V_O','W','z_re','z_im'])
    writer.writeheader()
    for r in [ep]+v['occupied_checks']:
        z=r.get('z',[None,None]);writer.writerow(dict(case='retuned_EP' if 'z' in r else 'occupied_gate',U=r['U'],n_d=r['n_d'],D=r['double_occupancy'],P_single=r['n_d']-2*r['double_occupancy'],mu=r['mu'],epsilon=r['epsilon'],V_O=r['auxiliary_vertex'],W=r.get('impurity_hybridization',.1),z_re=z[0],z_im=z[1]))
print('Figure and benchmark table written.')
