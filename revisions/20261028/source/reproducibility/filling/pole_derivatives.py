"""Exact resolvent derivatives for the finite-chain pole search.

No spectral truncation: the same full charge-sector matrices and ground
ensemble as charge_solver are used. Each LU factorization serves all three
derivative orders. The final certificate also compares finite differences.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
import charge_solver as charge
import filling_campaign as c


def setup(state,member):
    if member.response_state is None:
        pb,pi,hp=charge.sector(state.one_body,state.interaction,member.charge+1,state.max_sector_dimension)
        mb,mi,hm=charge.sector(state.one_body,state.interaction,member.charge-1,state.max_sector_dimension)
        def vectors(index,creation,composite=False):
            return tuple(charge.exact.operator_vector(member.basis,member.vector,index,a,creation,composite) for a in range(2))
        member.response_state=hp,hm,vectors(pi,True),vectors(mi,False),vectors(pi,True,True),vectors(mi,False,True)
    return member.response_state


def response_derivatives(state,z):
    result=[(np.zeros((2,2),complex),np.zeros((2,2),complex)) for _ in range(3)]
    for member in state.members:
        hp,hm,pd,md,pf,mf=setup(state,member)
        pd,md,pf,mf=[np.column_stack(v) for v in (pd,md,pf,mf)]
        for h,shift,sign,right,left_g,left_f,hole in [
            (hp,z+state.energy,-1,pd,pd,pf,False),
            (hm,z-state.energy,1,np.column_stack([md,mf]),md,md,True)]:
            if not h.shape[0]:continue
            lu=splu(shift*sparse.eye(h.shape[0],format='csc')+sign*h.tocsc())
            projected=right
            for order,factor in enumerate([1.,-1.,2.]):
                projected=lu.solve(projected)
                if hole:
                    g=(left_g.conj().T@projected[:,:2]).T
                    f=(left_f.conj().T@projected[:,2:]).T
                else:
                    g=left_g.conj().T@projected
                    f=left_f.conj().T@projected
                result[order][0][:]+=member.weight*factor*g
                result[order][1][:]+=member.weight*factor*f
    return result


def sigma_derivatives(states,z):
    if states[0].interaction==0:return [np.zeros((2,2),complex) for _ in range(3)]
    response=[response_derivatives(s,z) for s in states]
    g=[sum(r[k][0] for r in response)/len(states) for k in range(3)]
    f=[sum(r[k][1] for r in response)/len(states) for k in range(3)]
    if np.linalg.cond(g[0])>1e12:raise RuntimeError('ill-conditioned impurity Green function in analytic derivative')
    u=states[0].interaction
    right_solve=lambda a:np.linalg.solve(g[0].T,a.T).T
    s0=right_solve(u*f[0])
    s1=right_solve(u*f[1]-s0@g[1])
    s2=right_solve(u*f[2]-2*s1@g[1]-s0@g[2])
    return s0,s1,s2


def determinant_derivatives(config,job,states,z,width,vertex):
    import resonance_sheets
    h=config['ep']['derivative_step']
    bath=lambda value:resonance_sheets.bath(config,value+job['mu'],vertex,width)
    b0,b1,b2=c.h_derivatives(bath,z,h)
    s0,s1,s2=sigma_derivatives(states,z)
    k=(z+job['mu']-job['epsilon_d'])*np.eye(2)-b0-s0
    first=np.eye(2)-b1-s1
    second=-b2-s2
    f=np.linalg.det(k)
    fz=first[0,0]*k[1,1]+k[0,0]*first[1,1]-first[0,1]*k[1,0]-k[0,1]*first[1,0]
    fzz=(second[0,0]*k[1,1]+2*first[0,0]*first[1,1]+k[0,0]*second[1,1]
         -second[0,1]*k[1,0]-2*first[0,1]*first[1,0]-k[0,1]*second[1,0])
    return f,fz,fzz
