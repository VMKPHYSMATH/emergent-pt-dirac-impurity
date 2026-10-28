"""Independent analytic, finite-detuning, loop, and symmetry-breaking checks."""
import json
from pathlib import Path
import numpy as np
from scipy import linalg as la
from pt_control import Response, basis, onebody, hamiltonian, spin_flip, SX, SZ, encode

def dimer(U, V=.5):
    a = np.sqrt(U*U/16 + V*V)
    b = np.sqrt(U*U/16 + 4*V*V)
    low, high = b-a, b+a
    Z = (low*low-9*V*V)/(2*(low*low-high*high))
    m = .5*(1+U/(4*a))
    return dict(omega=low, upper_pole=high, weight=Z, slope=m,
                B_coefficient=Z*m, double_occupancy=.25*(1-U/np.sqrt(U*U+64*V*V)),
                nilpotent_selfenergy_coefficient_at_pole=m/Z-1)

def poles(sites, U, delta, gamma, longitudinal=0., bath_field=0., anchor=None):
    h = onebody(sites, U, field=delta*SX+(longitudinal+1j*gamma)*SZ, bath_field=bath_field)
    e = la.eigvals(hamiltonian(h,U,basis(2*sites,sites)))
    ep = la.eigvals(hamiltonian(h,U,basis(2*sites,sites+1)))
    if anchor is None:
        ezero = la.eigvalsh(hamiltonian(onebody(sites,U),U,basis(2*sites,sites)))[0]
        eplus = la.eigvalsh(hamiltonian(onebody(sites,U),U,basis(2*sites,sites+1)))[0]
    else:
        ezero,eplus=anchor
    ground = e[np.argmin(abs(e-ezero))]
    pair = ep[np.argsort(abs(ep-eplus))[:2]] - ground
    return pair

def run():
    out = {'analytic_dimer':[], 'two_sided':[], 'loops':[], 'negative_controls':[]}
    for U in [0,.5,2,4,8,16,32]:
        R=Response(2,U)
        a=R.lowest_addition(); exact=dimer(U)
        errs={k:float(abs(a[k]-exact[k])) for k in ['omega','weight','slope']}
        errs['D']=float(abs(R.observable(lambda s:int(bool(s&1))*int(bool(s&2)))-exact['double_occupancy']))
        greens=[]
        for z in [.31+.21j, -.73+.14j, 1.3+.42j]:
            scalar=1/(z-.25/z-U*U/4*z/(z*z-2.25))
            greens.append(float(abs(np.trace(R.green(z))/2-scalar)))
        errs['scalar_green']=max(greens)
        assert max(errs.values())<1e-10
        out['analytic_dimer'].append(dict(U=U,**exact,errors=errs))
    for U in [4.,8.]:
        R=Response(4,U); a=R.lowest_addition(); delta=R.delta
        anchor=R.E0,a['energy']
        for eta in [.001,.0005,.00025]:
            kp=la.inv(R.green(a['omega']+1j*eta))
            km=la.inv(R.green(a['omega']-1j*eta))
            k0=(kp+km)/2
            fzz=-(la.det(kp)+la.det(km))/eta**2
            out.setdefault('inverse_kernel_checks',[]).append(dict(U=U,eta=eta,
                K_limit_error=float(la.norm(k0+a['slope']/a['weight']*R.field)),
                Fzz=encode(fzz),Fzz_expected=2/a['weight']**2,
                Fzz_relative_error=float(abs(fzz/(2/a['weight']**2)-1))))
        for relative in [1e-2,1e-3,1e-4,1e-5,1e-6]:
            for side in [-1,1]:
                gamma=delta*(1+side*relative)
                p=poles(4,U,delta,gamma,anchor=anchor)
                measured=(p[0]-p[1])**2
                prediction=4*a['slope']**2*(delta*delta-gamma*gamma)
                out['two_sided'].append(dict(U=U,gamma=gamma,relative_offset=relative,side=side,
                    poles=encode(p),squared_splitting=encode(measured),prediction=prediction,
                    ratio=encode(measured/prediction)))
        for radius, points in [(1e-4,128),(5e-5,256)]:
            path=[]
            for theta in np.linspace(0,2*np.pi,points+1):
                p=poles(4,U,delta,delta+radius*np.cos(theta),radius*np.sin(theta),anchor=anchor)
                if path and la.norm(p-path[-1])>la.norm(p[::-1]-path[-1]):
                    p=p[::-1]
                path.append(p)
            path=np.array(path); difference=path[:,0]-path[:,1]
            winding=(np.unwrap(np.angle(difference*difference))[-1]-np.unwrap(np.angle(difference*difference))[0])/(2*np.pi)
            exchanged=la.norm(path[-1]-path[0,::-1])<1e-9
            assert abs(abs(winding)-1)<1e-7 and exchanged
            out['loops'].append(dict(U=U,radius=radius,points=points,winding=float(winding),
                                     exchange=bool(exchanged),poles=encode(path)))
        # Original PT and triangular forms are unitarily equivalent, also for G.
        pt=Response(4,U,form='pt')
        rotation=(np.eye(2)-1j*SX)/np.sqrt(2)
        rotation_error=max(la.norm(pt.green(z)-rotation@R.green(z)@rotation.conj().T)
                           for z in [.21+.17j,-.2+.1j])
        assert rotation_error<1e-8
        out.setdefault('basis_checks',[]).append(dict(U=U,green_rotation_error=float(rotation_error)))
        # PT is preserved by a real bath sigma_x field, but complex-spin covariance is lost.
        field=.005
        h=onebody(4,U,field=delta*SX+1j*delta*SZ,bath_field=field)
        states=basis(8,5); H=hamiltonian(h,U,states); P=spin_flip(states,4)
        p=poles(4,U,delta,delta,bath_field=field,anchor=anchor)
        out['negative_controls'].append(dict(U=U,bath_sigma_x=field,
            pt_error=float(la.norm(P@H.conj()@P-H)),poles_at_old_EP=encode(p),
            splitting_at_old_EP=float(abs(p[0]-p[1]))))
    # At U>0, freezing the self-energy to its zero-field scalar part misses the vertex.
    for U in [4.,8.]:
        R=Response(2,U);z=.31+.21j;g=R.green(z);gi=la.inv(g)
        sigma=(z+U/2-.25/z)*np.eye(2)-R.field-gi
        out.setdefault('selfenergy_checks',[]).append(dict(U=U,z=encode(z),
            scalar_error=float(abs(np.trace(sigma)/2-(U/2+U*U/4*z/(z*z-2.25)))),
            nilpotent_coefficient=encode(sigma[0,1]/(.2))))
    for U in [4.,8.]:
        responses=[Response(4,U,delta=d) for d in [.05,.1,.2]]
        zero=Response(4,U,delta=0.)
        z=.31+.21j
        scalars=[np.trace(r.green(z))/2 for r in responses]
        vectors=[r.green(z)[0,1]/(2*r.delta) for r in responses]
        scalar_error=max(abs(s-np.trace(zero.green(z))/2) for s in scalars)
        vector_error=max(abs(v-vectors[0]) for v in vectors)
        assert scalar_error<1e-9 and vector_error<1e-9
        out.setdefault('nilpotent_amplitude_checks',[]).append(dict(U=U,
            deltas=[.05,.1,.2],scalar_error=float(scalar_error),vector_error=float(vector_error)))
    Path('results/verification.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v if k not in ('loops','two_sided') else [
        {kk:vv for kk,vv in r.items() if kk!='poles'} for r in v]
        for k,v in out.items()},indent=2))

if __name__=='__main__':run()
