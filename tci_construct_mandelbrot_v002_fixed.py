#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stable Construct→Mandelbrot TCI flow with safe metrics
"""

import numpy as np, matplotlib.pyplot as plt, json, time
from numpy.linalg import svd
from scipy.spatial.distance import cdist, directed_hausdorff
from scipy.spatial import KDTree

# ---------- CONFIG ----------
np.random.seed(7)
construct_ns = list(range(20, 301, 20))
mandelbrot_grid = 600
mandelbrot_samples = 25000
escape_R, max_iter = 250, 250
grid_bins = 128
domain = (-2.25, 1.25, -1.75, 1.75)
alpha, T, eps = 0.2, 60, 1e-12
sinkhorn_eps, sinkhorn_iter = 0.8, 600
# ----------------------------

def lucas_companion(n):
    A = np.zeros((n,n)); A[0,:]=1; A[1:,:-1]=np.eye(n-1); return A

def construct_points(ns):
    pts=[]
    for n in ns:
        vals=np.linalg.eigvals(lucas_companion(n))
        vals=vals[np.abs(vals)>1e-10]
        pts+=list(1/vals)
    return np.array(pts)

def mandelbrot_distance_estimator(c):
    z=np.zeros_like(c); dz=np.ones_like(c); esc=np.zeros(c.shape,bool); last=np.zeros_like(c)
    with np.errstate(over='ignore', invalid='ignore'):
        for _ in range(max_iter):
            dz=2*z*dz+1; z=z*z+c
            mask=(np.abs(z)>escape_R)&(~esc)
            esc|=mask; last[mask]=z[mask]
    d=np.zeros(c.shape)
    m=esc; z_,dz_=last[m],dz[m]
    with np.errstate(over='ignore', invalid='ignore'):
        d[m]=np.log(np.abs(z_))*np.abs(z_)/np.maximum(np.abs(2*z_*dz_),eps)
    d=np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return esc,d,last

def sample_mandelbrot_boundary():
    xs=np.linspace(domain[0],domain[1],mandelbrot_grid)
    ys=np.linspace(domain[2],domain[3],mandelbrot_grid)
    X,Y=np.meshgrid(xs,ys); C=X+1j*Y
    esc,d,_=mandelbrot_distance_estimator(C)
    if not np.any(esc): raise RuntimeError("No escape points")
    q=np.quantile(d[esc],0.25)
    pts=C[(esc)&(d<=q)].ravel()
    if pts.size>mandelbrot_samples:
        pts=np.random.choice(pts,mandelbrot_samples,replace=False)
    return pts

# --- simplified Sinkhorn (robust) ---
def entropic_ot_alignment(X,Y):
    n,m=len(X),len(Y)
    if n>m: X=np.random.choice(X,m,replace=False)
    if m>n: Y=np.random.choice(Y,n,replace=False)
    M=cdist(np.c_[X.real,X.imag],np.c_[Y.real,Y.imag])
    M=M/M.mean()
    K=np.exp(-M/sinkhorn_eps)
    K=np.nan_to_num(K)
    match=np.argmax(K,axis=1)
    return np.array([Y[j] for j in match]), X

def procrustes_align_no_scale(Xc,Yc):
    X=np.c_[Xc.real,Xc.imag]; Y=np.c_[Yc.real,Yc.imag]
    X0,Y0=X-X.mean(0),Y-Y.mean(0)
    U,_,Vt=svd(Y0.T@X0,full_matrices=False)
    R=U@Vt; Xal=(X0@R)+Y.mean(0)
    return Xal[:,0]+1j*Xal[:,1]

def to_prob(cloud,bins=grid_bins):
    H,_,_=np.histogram2d(cloud.real,cloud.imag,
        bins=(bins,bins),
        range=[[domain[0],domain[1]],[domain[2],domain[3]]])
    H=np.maximum(H,eps); return H/H.sum()

def KL(P,X):
    P_=np.clip(P,eps,None); X_=np.clip(X,eps,None)
    return float(np.sum(P_*(np.log(P_)-np.log(X_))))

def tci_flow(P,X0):
    X=X0.copy(); kls=[KL(P,X)]; traj=[X]
    for _ in range(T):
        X=(1-alpha)*X+alpha*P
        kls.append(KL(P,X)); traj.append(X.copy())
    return np.array(kls),traj

def hausdorff(A,B): 
    return max(directed_hausdorff(A,B)[0], directed_hausdorff(B,A)[0])

def local_curvature(pts,k=6):
    tree=KDTree(np.c_[pts.real,pts.imag]); curv=[]
    for p in pts:
        _,idx=tree.query([p.real,p.imag],k=k)
        neigh=pts[idx]; z=neigh-np.mean(neigh)
        cov=np.cov(np.c_[z.real,z.imag].T)
        eig=np.linalg.eigvalsh(cov)
        curv.append(eig.min()/np.sum(eig))
    return np.array(curv)

def spectral_distance(X,Y,K=30,sigma=0.05):
    from scipy.spatial.distance import pdist,squareform
    D1=squareform(pdist(np.c_[X.real,X.imag]))
    D2=squareform(pdist(np.c_[Y.real,Y.imag]))
    K1=np.exp(-D1**2/(2*sigma**2))
    K2=np.exp(-D2**2/(2*sigma**2))
    w1=np.sort(np.real(np.linalg.eigvals(K1)))[-K:]
    w2=np.sort(np.real(np.linalg.eigvals(K2)))[-K:]
    return np.linalg.norm(w1-w2)/np.sqrt(K)

if __name__=="__main__":
    t0=time.time()
    print("Generating Construct and Mandelbrot samples...")
    Cpts=construct_points(construct_ns)
    Mpts=sample_mandelbrot_boundary()
    print("OT + Procrustes alignment...")
    Mmatch,Ctrim=entropic_ot_alignment(Cpts,Mpts)
    Caligned=procrustes_align_no_scale(Ctrim,Mmatch)

    print("Computing correspondences (robustly)...")
    try:
        # Downsample both to the same length for safe correlations
        n=min(len(Caligned),len(Mpts))
        Csub=np.random.choice(Caligned,n,replace=False)
        Msub=np.random.choice(Mpts,n,replace=False)
        h0=hausdorff(np.c_[Csub.real,Csub.imag],np.c_[Msub.real,Msub.imag])
        curv_corr=np.corrcoef(local_curvature(Csub),local_curvature(Msub))[0,1]
    except Exception as e:
        print("Warning: curvature correlation failed:",e)
        h0,curv_corr=np.nan,np.nan

    try:
        dspec=spectral_distance(Caligned,Mpts)
    except Exception as e:
        print("Warning: spectral distance failed:",e)
        dspec=np.nan

    print("Running TCI flow...")
    P_M=to_prob(Mpts); X_C=to_prob(Caligned)
    kls,traj=tci_flow(P_M,X_C)

    out={"Hausdorff_before":float(h0),
         "Curvature_corr":float(curv_corr),
         "Spectral_L2":float(dspec),
         "KL_initial":float(kls[0]),
         "KL_final":float(kls[-1]),
         "runtime_sec":time.time()-t0}
    json.dump(out,open("tci_results.json","w"),indent=2)

    plt.figure()
    plt.plot(kls)
    plt.xlabel("t"); plt.ylabel("D_KL"); plt.title("KL descent (TCI flow)")
    plt.tight_layout(); plt.savefig("KL_descent.png",dpi=150)

    plt.figure()
    plt.imshow(traj[-1],origin="lower",extent=domain)
    plt.title("Final histogram X_T")
    plt.tight_layout(); plt.savefig("XT_final.png",dpi=150)
    plt.close("all")

    print("Done. Results:",out)
