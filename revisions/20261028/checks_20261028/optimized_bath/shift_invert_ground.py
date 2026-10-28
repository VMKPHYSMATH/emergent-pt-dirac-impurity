"""Optional exact-sector eigensolver acceleration; no state truncation.

The first solve shifts below a rigorous Gershgorin bound. Reconstruction
shifts immediately below the known ground energy and resolves degeneracy.
"""
import numpy as np
from scipy.sparse.linalg import eigsh

def install(cs):
    original=cs.exact.lowest_eigenpair
    original_vectors=cs.ground_vectors
    def real_if_possible(A):
        return A.real if np.max(np.abs(A.data.imag),initial=0)<1e-14 else A
    def lowest(A):
        if A.shape[0]<=256:return original(A)
        M=real_if_possible(A)
        diagonal=M.diagonal().real
        radius=np.asarray(abs(M).sum(axis=1)).ravel()-abs(diagonal)
        lower=float(np.min(diagonal-radius))-1e-8
        e,v=eigsh(M,k=1,sigma=lower,which='LM',tol=1e-12,maxiter=10000)
        residual=np.linalg.norm(M@v[:,0]-e[0]*v[:,0])
        if residual>1e-9:raise RuntimeError(f'shift-invert residual {residual}')
        return float(e[0]),v[:,0].astype(complex)
    def vectors(A,target,tolerance):
        if A.shape[0]<=256:return original_vectors(A,target,tolerance)
        M=real_if_possible(A);k=4
        while True:
            e,V=eigsh(M,k=min(k,A.shape[0]-2),sigma=target-1e-8,which='LM',tol=1e-12,maxiter=10000)
            ix=np.argsort(e);e,V=e[ix],V[:,ix]
            if e[0]<target-max(10*tolerance,1e-9):raise RuntimeError('ground energy inconsistent')
            if e[-1]>target+tolerance:break
            k*=2
            if k>64:raise RuntimeError('unresolved ground degeneracy')
        keep=np.abs(e-target)<=tolerance
        if not np.any(keep):raise RuntimeError('ground reconstruction failed')
        V,_=np.linalg.qr(V[:,keep])
        return [V[:,j].astype(complex) for j in range(V.shape[1])]
    cs.exact.lowest_eigenpair=lowest
    cs.ground_vectors=vectors
