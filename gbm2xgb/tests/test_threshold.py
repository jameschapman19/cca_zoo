import numpy as np

from gbm2xgb._xgb import xgb_threshold


def test_threshold_matches_double_comparison_on_all_nearby_float32():
    rng = np.random.default_rng(0)
    for t in map(float, np.concatenate([rng.normal(size=2000), rng.normal(size=2000) * 1e5, [0.0, 1e-35]])):  # python floats, as parsed from LightGBM JSON
        c = np.float32(xgb_threshold(t))
        f = np.float32(t)
        x = f
        neighbours = [x]
        for _ in range(3):
            neighbours.append(np.nextafter(neighbours[-1], np.float32(np.inf)))
        x = f
        for _ in range(3):
            x = np.nextafter(x, np.float32(-np.inf))
            neighbours.append(x)
        for x in neighbours:
            assert (float(x) <= t) == (x < c), (t, x, c)
