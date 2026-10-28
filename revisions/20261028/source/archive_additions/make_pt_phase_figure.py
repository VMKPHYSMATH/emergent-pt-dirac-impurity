#!/usr/bin/env python3
"""Render the corrected retarded splitting diagnostic (historical filename)."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
data,out=sys.argv[1:3]
d=np.load(data); x,y=np.meshgrid(d['Oms'],d['lams'])
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix',
 'font.size':10,'axes.labelsize':11,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(5.5,3.9),layout='constrained')
im=ax.pcolormesh(x,y,d['cosine_phase'],vmin=-1,vmax=1,cmap='RdBu',shading='auto',rasterized=True)
ax.contour(x,y,d['cosine_phase'],levels=[0],colors='black',linewidths=1.1)
th=d['thresh'];ok=np.isfinite(th[:,1]);ax.plot(th[ok,1],th[ok,0],'k--',lw=1)
ax.plot(60,.10,'o',mfc='white',mec='black',ms=6)
ax.annotate('pole-EP controls',xy=(60,.1),xytext=(77,.125),fontsize=9,
 arrowprops={'arrowstyle':'-','lw':.8})
ax.set(xlabel=r'drive frequency $\Omega$',ylabel=r'SOC strength $\lambda$',
 xlim=(d['Oms'].min(),d['Oms'].max()),ylim=(d['lams'].min(),d['lams'].max()))
cb=fig.colorbar(im,ax=ax,pad=.025,ticks=[-1,-.5,0,.5,1])
cb.set_label(r'$C_s=\mathrm{Re}\,s^2/|s^2|$')
fig.savefig(out,dpi=400);fig.savefig(out.replace('.pdf','.png'),dpi=200)
print(out)
