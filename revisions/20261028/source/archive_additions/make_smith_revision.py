#!/usr/bin/env python3
"""Smith diagnostic normalized by the slowest pole half-width.
Independent matrix calculation for the declared Markov benchmark only.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
I=np.eye(2,dtype=complex);sx=np.array([[0,1],[1,0]],complex);sz=np.diag([1,-1])
gamma=.12
def poles(D,G):
    # Analytic labels avoid floating-point swapping after real-part coalescence.
    s=np.sqrt(complex(D*D-G*G))
    return np.array([-s,s])-1j*gamma
def calc(w,D,G):
    R=np.linalg.inv(w[:,None,None]*I-D*sx+1j*gamma*I-1j*G*sz)
    B=np.diag(np.sqrt([2*(gamma-G),2*(gamma+G)]))
    A=R.conj().transpose(0,2,1);Q=B@A@R@B
    S=I-1j*B@R@B;dS=1j*B@R@R@B
    Qder=-1j*S.conj().transpose(0,2,1)@dS
    return np.linalg.eigvalsh(Q)[:,-1],float(np.max(abs(Q-Qder))),float(np.max(abs(S.conj().transpose(0,2,1)@S-I)))
def main():
    dest=Path(__file__).resolve().parent.parent
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10,'pdf.fonttype':42})
    fig,ax=plt.subplots(2,2,figsize=(8.6,6.4),layout='constrained')
    betas=np.linspace(.35,.55,161);w=np.linspace(-.6,.6,12001)
    records=[];maxerr=maxunit=0.
    for control,ls,label in [(False,'-','active'),(True,'--',r'matched $G=0$')]:
        pp=[];norm=[]
        for b in betas:
            D=.075+.05*b;G=0 if control else .2*b
            po=poles(D,G);pp.append(po)
            q,e,u=calc(w,D,G);maxerr=max(maxerr,e);maxunit=max(maxunit,u)
            slow=float(min(-po.imag));norm.append(slow*max(q))
            records.append({'beta0':float(b),'control':control,'D':float(D),'G':float(G),
              'slowest_pole_halfwidth':slow,'proper_peak':float(max(q)),'normalized_peak':float(norm[-1])})
        pp=np.array(pp)
        for j,c in [(0,'tab:blue'),(1,'tab:orange')]:
            ax[0,0].plot(betas,pp[:,j].real,color=c,ls=ls,label=label if j==0 else None)
            ax[0,1].plot(betas,-pp[:,j].imag,color=c,ls=ls)
        ax[1,1].plot(betas,norm,ls,label=label,lw=1.6)
    control_values={}
    for D,G,style,label in [(.1,.1,'-','EP'),(.1,0,'--','Hermitian'),(.05,.1,'-.','off EP')]:
        q,e,u=calc(w,D,G);slow=float(min(-poles(D,G).imag))
        ax[1,0].plot(w,q,style,label=label,lw=1.5)
        control_values[label]={'peak':float(max(q)),'slowest_halfwidth':slow,'normalized_peak':float(slow*max(q))}
    off=control_values['off EP']['normalized_peak'];ax[1,1].plot(.5,off,'D',color='purple',label='off-EP control')
    for a in [ax[0,0],ax[0,1],ax[1,1]]:a.axvline(.5,color='gray',ls=':',lw=1);a.set_xlabel(r'$\beta_0$')
    ax[0,0].set(title='(a) Pole coalescence',ylabel=r'$\mathrm{Re}\,z$');ax[0,0].legend(frameon=False,fontsize=9)
    ax[0,1].set(title='(b) Pole half-widths',ylabel=r'$-\mathrm{Im}\,z$')
    ax[1,0].set(title='(c) Proper delay and controls',xlabel=r'$\omega$',ylabel=r'$q_{\max}(\omega)$',xlim=(-.35,.35));ax[1,0].legend(frameon=False,fontsize=9)
    ax[1,1].set(title='(d) Pole-width normalization',ylabel=r'$\gamma_{\rm slow}q_{\max}^{\rm peak}$');ax[1,1].legend(frameon=False,fontsize=9)
    for a in ax.flat:a.grid(alpha=.2,lw=.5)
    fig.savefig(dest/'SFig2_smith_delay.pdf');fig.savefig(dest/'SFig2_smith_delay.png',dpi=180)
    out={'controls':control_values,'max_smith_identity_residual':maxerr,'max_unitarity_residual':maxunit,'records':records}
    (dest/'validation/smith_revision.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
if __name__=='__main__':main()
