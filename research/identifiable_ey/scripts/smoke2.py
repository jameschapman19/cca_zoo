import numpy as np, torch
from ieye import methods as M
from ieye.data import TwoViewModel
from ieye.metrics import amari_to_sources, subspace_angle
torch.set_num_threads(2)
for n in (500, 2000):
  for seed in range(3):
    rng = np.random.default_rng(seed)
    model = TwoViewModel.make([.8,.75,.7,.65,.6], [2,-1,2,-1,2], 10, rng)
    _, views = model.sample(n, rng)
    out = {"cca_fastica": M.cca_fastica(views,5,seed), "mvica": M.multiview_ica(views,5,seed), "rot": M.rotation_only(views,5,seed,restarts=4,max_iter=200)}
    for mu in (1, 10, 100): out[f"joint{mu}"] = M.joint(views,5,seed,mu=mu,restarts=4,max_iter=200)
    print(n, seed, "  ".join(f"{k}={amari_to_sources(model,w):.3f}/{subspace_angle(model,w):.1f}" for k,w in out.items()), flush=True)
