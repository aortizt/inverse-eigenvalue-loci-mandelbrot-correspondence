#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-resolution Construct→Mandelbrot TCI flow experiment.
Reproducible, standalone version for local execution (Spyder, CLI, etc.).
"""

import numpy as np, matplotlib.pyplot as plt, json, time
from numpy.linalg import svd
from scipy.spatial.distance import cdist, directed_hausdorff
from scipy.spatial import KDTree
from multiprocessing import Pool, cpu_count

# ---------- CONFIG ----------
np.random.seed(7)
construct_ns = list(range(20, 301, 20))
mandelbrot_grid = 600
mandelbrot_samples = 25000
escape_R, max_iter = 4.0, 250
grid_bins = 128
domain = (-2.25, 1.25, -1.75, 1.75)
alpha, T, eps = 0.2, 60, 1e-12
sinkhorn_eps, sinkhorn_iter = 0.05, 1000
use_parallel = True
# ----------------------------

# === Construct ===
def lucas_companion(n:int):
    A=np.zeros((n,n)); A[0,:]=1; A[1:,:-1]=np.eye(n-1); return A
def construct_points(ns):
    pts=[]; 
    for n in ns:
        vals=np.linalg.eigvals(lucas_companion(n))
        vals=vals[np.abs(vals)>1e-10]; pts+=list(1/vals)
    return np.array(pts,dtype=complex)

# === Mandelbrot ===
def mandelbrot_distance_estimator(c):
    z=np.zeros_like(c); dz=np.ones_like(c); esc=np.zeros(c.shape,bool); last=np.zeros_like(c)
    for _ in range(max_iter):
        dz=2*z*dz+1; z=z*z+c
        mask=(np.abs(z)>escape_R)&(~esc); esc|=mask; last[mask]=z[mask]
    d=np.zeros(c.shape); m=esc
    z_,dz_=last[m],dz[m]
    d[m]=np.log(np.abs(z_))*np.abs(z_)/np.maximum(np.abs(2*z_*dz_),eps)
    return esc,d,last

def sample_mandelbrot_boundary():
    xs=np.linspace(domain[0],domain[1],mandelbrot_grid)
    ys=np.linspace(domain[2],domain[3],mandelbrot_grid)
    X,Y=np.meshgrid(xs,ys); C=X+1j*Y
    esc,d,_=mandelbrot_distance_estimator(C)
    q=np.quantile(d[esc],0.25)
    pts=C[(esc)&(d<=q)].ravel()
    if pts.size>mandelbrot_samples:
        pts=np.random.choice(pts,mandelbrot_samples,replace=False)
    return pts

# === Entropic OT (Sinkhorn) ===
def sinkhorn(a,b,M,eps=0.05,iters=500):
    K=np.exp(-M/eps); u=np.ones_like(a); v=np.ones_like(b)
    for _ in range(iters):
        u=a/(K@v); v=b/(K.T@u)
    Γ=np.diag(u)@K@np.diag(v); return Γ

def entropic_ot_alignment(X,Y):
    n,m=len(X),len(Y)
    a,b=np.ones(n)/n,np.ones(m)/m
    M=cdist(np.c_[X.real,X.imag],np.c_[Y.real,Y.imag])**2
    Γ=sinkhorn(a,b,M,sinkhorn_eps,sinkhorn_iter)
    match=np.argmax(Γ,axis=1)
    return np.array([Y[j] for j in match])

# === Procrustes ===
def procrustes_align_no_scale(Xc,Yc):
    X=np.c_[Xc.real,Xc.imag]; Y=np.c_[Yc.real,Yc.imag]
    X0,Y0=X-X.mean(0),Y-Y.mean(0)
    U,_,Vt=svd(Y0.T@X0,full_matrices=False)
    R=U@Vt; Xal=(X0@R)+Y.mean(0)
    return Xal[:,0]+1j*Xal[:,1]

# === Histograms and KL ===
def to_prob(cloud,bins=grid_bins):
    H,_,_=np.histogram2d(cloud.real,cloud.imag,bins=(bins,bins),
                         range=[[domain[0],domain[1]],[domain[2],domain[3]]])
    H=np.maximum(H,eps); return H/H.sum()
def KL(P,X): P_=np.clip(P,eps,None); X_=np.clip(X,eps,None)
    return float(np.sum(P_*(np.log(P_)-np.log(X_))))

def tci_flow(P,X0):
    X=X0.copy(); kls=[KL(P,X)]; traj=[X]
    for _ in range(T):
        X=(1-alpha)*X+alpha*P; kls.append(KL(P,X)); traj.append(X.copy())
    return np.array(kls),traj

# === Quantitative correspondences ===
def hausdorff(A,B): return max(directed_hausdorff(A,B)[0], directed_hausdorff(B,A)[0])

def local_curvature(pts,k=6):
    tree=KDTree(np.c_[pts.real,pts.imag]); curv=[]
    for p in pts:
        _,idx=tree.query([p.real,p.imag],k=k)
        neigh=pts[idx]; z=neigh-np.mean(neigh)
        cov=np.cov(np.c_[z.real,z.imag].T); eig=np.linalg.eigvalsh(cov)
        curv.append(eig.min()/np.sum(eig))
    return np.array(curv)

def spectral_distance(X,Y,K=30,sigma=0.05):
    from scipy.spatial.distance import pdist,squareform
    for pts,name in [(X,"X"),(Y,"Y")]:
        D=squareform(pdist(np.c_[pts.real,pts.imag]))
        Kmat=np.exp(-D**2/(2*sigma**2)); Dsum=Kmat.sum(1)
        L=np.diag(1/Dsum)@Kmat; w,_=np.linalg.eig(L)
        yield np.sort(np.real(w))[-K:]

# === MAIN ===
t0=time.time()
Cpts=construct_points(construct_ns)
Mpts=sample_mandelbrot_boundary()

# OT + Procrustes
Mmatch=entropic_ot_alignment(Cpts,Mpts)
Caligned=procrustes_align_no_scale(Cpts,Mmatch)

# Initial metrics
h0=hausdorff(np.c_[Caligned.real,Caligned.imag],
             np.c_[Mpts.real,Mpts.imag])
curv_corr=np.corrcoef(local_curvature(Caligned),
                      local_curvature(Mpts))[0,1]
specC,specM=spectral_distance(Caligned,Mpts)
dspec=np.linalg.norm(specC-specM)/np.sqrt(len(specC))

# Flow
P_M=to_prob(Mpts); X_C=to_prob(Caligned)
kls,traj=tci_flow(P_M,X_C)
hT=KL(P_M,traj[-1])

# Output
out={"Hausdorff_before":h0,"Curvature_corr":float(curv_corr),
     "Spectral_L2":float(dspec),
     "KL_initial":float(kls[0]),"KL_final":float(kls[-1]),
     "runtime_sec":time.time()-t0}
json.dump(out,open("tci_results.json","w"),indent=2)

# Figures
plt.figure(); plt.plot(kls); plt.xlabel("t"); plt.ylabel("D_KL"); 
plt.title("KL descent (TCI flow)"); plt.savefig("KL_descent.png",dpi=150)
plt.figure(); plt.imshow(traj[-1],origin="lower",extent=domain);
plt.title("Final histogram X_T"); plt.savefig("XT_final.png",dpi=150)
plt.close("all")
print("Done. Results:",out)
