import time
import numpy as np
from ieye import methods as M
from ieye.data import TwoViewModel
from ieye.metrics import amari_to_sources, subspace_angle, source_r2

rng = np.random.default_rng(0)
model = TwoViewModel.make([.8,.75,.7,.65,.6], [2,-1,2,-1,2], 10, rng)
_, views = model.sample(2000, rng)
fits = {
    "cca": M.cca, "ey": M.ey, "cca_fastica": M.cca_fastica, "mvica": M.multiview_ica,
    "rot_only": M.rotation_only,
    "joint": lambda v,k,s: M.joint(v,k,s,mu=1.0),
}
for name, f in fits.items():
    t=time.time(); w=f(views,5,0); dt=time.time()-t
    print(f"{name:12s} amari={amari_to_sources(model,w):.3f} angle={subspace_angle(model,w):.2f} r2={source_r2(model,w,20000,np.random.default_rng(1)):.3f} t={dt:.1f}s")
