"""Smooth 2D double well with a curved, tilted reaction channel.

Phasepoint positions are (y, x): y is the order parameter (index 0), x the transverse
coordinate, as in the other 2D toytis potentials.

    s = (y - y0) / w                                  channel coordinate, wells at s = -1, +1
    c(s) = x_A + (x_B - x_A) (s + 1) / 2 + b (1 - s^2)  channel centre line: from x_A (well A) to
                                                      x_B (well B), bowed out by b at the saddle
    n = x - c(s)                                      distance from the centre line
    V(y, x) = V_b (s^2 - 1)^2 + eps s + (k / 2) (1 + a s^2)^(-1) n^2

- V_b: barrier height (at eps = 0); eps tilts the two wells (eps > 0: B higher).
- k: transverse stiffness at the saddle; a > 0 makes the channel wider in the wells
  than at the saddle (an entropic bottleneck), a = 0 keeps the width constant.
- The centre line runs diagonally and bends, so the reaction direction at the saddle
  is not along y: at fixed y the committor depends on x, and the isocommittor lines are
  tilted and curved. There are no walls, kinks or flat regions: the potential is a smooth
  polynomial-rational function, confining in both directions.

Defaults (kT = 0.05): barrier 4 kT, transverse thermal width sqrt(kT/k) ~ 0.06, wells at
y = 0.1 and 0.9, the channel moving from x = -0.15 (A) to x = +0.15 (B) and bowing out to
x = +0.2 at the saddle. Suggested states A: y <= 0.15, B: y >= 0.85.

`potential_and_force_xy(y, x, xp)` evaluates many points at once (xp = numpy or cupy); the
scalar methods used by the toytis engine call it.
"""
import numpy as np


class CurvedChannelDoubleWell:
    def __init__(self, Vb=0.2, eps=0.0, y0=0.5, w=0.4, xA=-0.15, xB=0.15, b=0.2, k=15.0, a=0.0):
        self.Vb, self.eps, self.y0, self.w = Vb, eps, y0, w
        self.xA, self.xB, self.b, self.k, self.a = xA, xB, b, k, a

    def __repr__(self):
        # Logged with the run settings, so an analysis can rebuild the exact potential. No '=' in it:
        # tistools reads any log line holding 'zero_left' and '=' as the zero_left setting.
        return "CurvedChannelDoubleWell({})".format(", ".join(
            "{}: {!r}".format(k, getattr(self, k)) for k in ("Vb", "eps", "y0", "w", "xA", "xB", "b", "k", "a")))

    # ------------------------------------------------------------------
    # vectorized core
    # ------------------------------------------------------------------
    def potential_and_force_xy(self, y, x, xp=np):
        """V and the force (F_y, F_x) at arrays of points y, x (numpy or cupy via xp)."""
        s = (y - self.y0) / self.w
        c = self.xA + (self.xB - self.xA) * (s + 1) / 2 + self.b * (1 - s * s)
        dc_ds = (self.xB - self.xA) / 2 - 2 * self.b * s
        n = x - c
        g = 1.0 / (1 + self.a * s * s)                 # transverse stiffness profile k * g(s)
        dg_ds = -2 * self.a * s * g * g
        v = self.Vb * (s * s - 1) ** 2 + self.eps * s + 0.5 * self.k * g * n * n
        dv_ds = 4 * self.Vb * s * (s * s - 1) + self.eps + 0.5 * self.k * dg_ds * n * n \
            - self.k * g * n * dc_ds
        dv_dx = self.k * g * n
        return v, -dv_ds / self.w, -dv_dx

    # ------------------------------------------------------------------
    # toytis interface (one phasepoint)
    # ------------------------------------------------------------------
    def potential_and_force(self, ph):
        pos = np.asarray(ph[0], dtype=float)
        v, fy, fx = self.potential_and_force_xy(pos[0], pos[1])
        return float(v), np.array([fy, fx])

    def potential(self, ph):
        return self.potential_and_force(ph)[0]

    def force(self, ph):
        return self.potential_and_force(ph)[1]

    def effective_potential(self, ph):
        """The potential whose gradient is the force (here the potential itself)."""
        return self.potential(ph)

    def centre_line(self, y):
        """x of the channel centre at order parameter y."""
        s = (np.asarray(y) - self.y0) / self.w
        return self.xA + (self.xB - self.xA) * (s + 1) / 2 + self.b * (1 - s * s)

    def kick_range(self, lam=None):
        """Transverse range for initial kicks: the channel centre +- 0.05 at lambda."""
        xc = float(self.centre_line(0.5 if lam is None else lam))
        return (xc - 0.05, xc + 0.05)

    def grid(self, y_range=(-0.1, 1.1), x_range=None, ny=241, nx=121):
        """(ys, xs, V[iy, ix]) on a regular grid; x_range defaults to the channel +- 0.35."""
        if x_range is None:
            lo = min(self.xA, self.xB, self.xA + self.b) - 0.35
            hi = max(self.xA, self.xB, self.xB + self.b, self.b + (self.xA + self.xB) / 2) + 0.35
            x_range = (lo, hi)
        ys, xs = np.linspace(*y_range, ny), np.linspace(*x_range, nx)
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        return ys, xs, self.potential_and_force_xy(Y, X)[0]

    def stationary_points(self):
        """Minima and saddle along the channel (s = -1, +1 and the maximum in between)."""
        ss = np.linspace(-1.5, 1.5, 30001)
        ys = self.y0 + self.w * ss
        xs = self.centre_line(ys)
        v = self.potential_and_force_xy(ys, xs)[0]          # V along the centre line (n = 0)
        i_a = np.argmin(np.where(ss < 0, v, np.inf))
        i_b = np.argmin(np.where(ss > 0, v, np.inf))
        i_s = i_a + np.argmax(v[i_a:i_b + 1])
        return {name: (ys[i], xs[i], v[i]) for name, i in (("A", i_a), ("saddle", i_s), ("B", i_b))}

    def barrier_info(self):
        p = self.stationary_points()
        return ("well A at (y, x) = ({:.3f}, {:.3f}), V = {:.4f}; saddle at ({:.3f}, {:.3f}), V = {:.4f}; "
                "well B at ({:.3f}, {:.3f}), V = {:.4f}".format(*p["A"], *p["saddle"], *p["B"]))

    # ------------------------------------------------------------------
    # plotting (same names as the other potentials)
    # ------------------------------------------------------------------
    def plot_potential(self, ax, vmax=None):
        ys, xs, V = self.grid()
        vmax = vmax if vmax is not None else self.Vb * 3
        g = ax.contourf(ys, xs, np.clip(V, None, vmax).T, 40)
        ax.plot(ys, self.centre_line(ys), "w--", lw=0.8)
        ax.set(xlabel="y (order parameter)", ylabel="x")
        return g

    def get_potential_plot(self, N=100):
        ys, xs, V = self.grid(ny=N, nx=N)
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        return X, Y, V
