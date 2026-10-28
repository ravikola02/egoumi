"""Simple matplotlib renderer for named cq shapes."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
           "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#2F4B7C", "#A05195"]


def tris(shape, tol=0.4):
    v, f = shape.tessellate(tol, 0.3)
    v = np.array([[p.x, p.y, p.z] for p in v])
    return v[np.array(f)] if len(f) else np.zeros((0, 3, 3))


def render(parts, path, views=((25, -60),), title="", colors=None, alpha=1.0, size=6, labels=True):
    """parts: list of (name, shape). views: list of (elev, azim)."""
    data = []
    for i, (nm, s) in enumerate(parts):
        c = (colors or {}).get(nm, PALETTE[i % len(PALETTE)])
        data.append((nm, tris(s), c))
    allv = np.concatenate([t.reshape(-1, 3) for _, t, _ in data if len(t)])
    lo, hi = allv.min(0), allv.max(0)
    ctr, rng = (lo + hi) / 2, (hi - lo).max() / 2
    light = np.array([0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
    fig = plt.figure(figsize=(size * len(views), size))
    for k, (el, az) in enumerate(views):
        ax = fig.add_subplot(1, len(views), k + 1, projection="3d")
        for nm, t, c in data:
            if not len(t):
                continue
            n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            sh = 0.35 + 0.65 * np.abs(n @ light)
            base = np.array(matplotlib.colors.to_rgb(c))
            fc = np.clip(base[None] * sh[:, None], 0, 1)
            pc = Poly3DCollection(t, facecolors=fc, edgecolors="none", alpha=alpha)
            ax.add_collection3d(pc)
        for i in range(3):
            getattr(ax, "set_" + "xyz"[i] + "lim")(ctr[i] - rng, ctr[i] + rng)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(el, az)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(f"elev {el}, azim {az}", fontsize=9)
    if labels:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in data]
        fig.legend(handles, [nm for nm, _, _ in data], loc="lower center",
                   ncol=min(6, len(data)), fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)
