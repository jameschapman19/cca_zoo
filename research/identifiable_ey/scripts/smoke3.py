import time, numpy as np, torch
from ieye.data import TwoViewModel
from ieye import ssl
torch.set_num_threads(1)
rng = np.random.default_rng(0)
rho = np.linspace(0.9, 0.5, 8)
model = TwoViewModel.make(rho, np.zeros(8), 64, rng)
_, views = model.sample(20000, rng)
for d, batch in [(16, 32), (16, 256), (64, 32)]:
    for arm in ssl.ARMS:
        t = time.time()
        w = ssl.train(arm, views, d, batch, 2000, 3e-3, np.random.default_rng(1))
        print(d, batch, f"{arm:28s} capture={ssl.subspace_capture(model, w):.3f} probeR2={ssl.probe_r2(model, w, np.random.default_rng(2)):.3f} t={time.time()-t:.0f}s", flush=True)
