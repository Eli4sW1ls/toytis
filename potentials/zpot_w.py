"""Z potential with two mirrored channels, selected by a slow coordinate w (positions (y, x, w)).

The tunnelling test of the parametric order parameter (optimalrcs examples/param_lambda): a slow degree of
freedom orthogonal to y that changes the committor.

    U(y, x, w) = C(y, x) + s(w) R(y, x) + (1 - s(w)) R(y, Lx - x) + V_w(w)

* C: everything of the Z potential (sjoelbak_angle.RectangularGridWithAngledBarrierPotential) except its
  ridge: walls, state wells, slope, gravity (force only, as in the Z class). Symmetric in x -> Lx - x.
* R: the tilted cosine ridge of the Z potential (angle 45: the line x + y = 0.6). R(y, Lx - x) is its mirror
  image (the line y - x = 0.3).
* s(w) = (1 - tanh(w / delta)) / 2 switches between them: channel w < 0 has the Z ridge, channel w > 0 the
  mirrored one. s(-w) = 1 - s(w).
* V_w(w) = dV_w ((w / w0)^2 - 1)^2: a double well in w (minima at +-w0, barrier dV_w at w = 0) that makes w slow.

The two ridges cross in an X. Between them (0.3 < y < 0.6) a point is on the A side of one channel's ridge
and on the B side of the other's, so the committor there depends on w, which neither y nor (y, x) resolves.
A path can also pass a ridge "through" w: climb it in one channel and hop into the other, where the ridge is
elsewhere. In the (y, x) projection such a path crosses the ridge without paying for it.

Symmetry: U(y, x, w) = U(y, Lx - x, -w). In A and B (state wells, no ridge) the two channels have the same
energy, so each is populated half the time, and the A -> B rate is the same in both.

`potential_and_force` is written out with the math module (the engine calls it every step); `reference_*`
composes the Z class itself, for the tests.
"""
import math

import numpy as np

from potentials.sjoelbak_angle import RectangularGridWithAngledBarrierPotential


class ZChannelsW:
    """Two mirrored Z channels selected by a slow coordinate w. Phasepoint positions (y, x, w)."""

    def __init__(self, w0=0.1, dV_w=0.15, delta=0.02, kick_w_sd=0.02, Lx=0.3, angle=45.0, **z_kwargs):
        """
        Parameters
        ----------
        w0 : float
            Positions of the two w minima (+-w0).
        dV_w : float
            Barrier of the w double well at w = 0 (energy units; kT = 0.05 in the Z runs).
        delta : float
            Width of the channel switch s(w) = (1 - tanh(w / delta)) / 2.
        kick_w_sd : float
            SD of w around +-w0 for the kick points of initial paths (channel chosen at random).
        Lx, angle, z_kwargs
            Passed to RectangularGridWithAngledBarrierPotential (the Z runs: Lx = 0.3, angle = 45).
        """
        z = RectangularGridWithAngledBarrierPotential(Lx=Lx, angle=angle, **z_kwargs)
        self.z = z
        self.w0, self.dV_w, self.delta, self.kick_w_sd = float(w0), float(dV_w), float(delta), float(kick_w_sd)
        self.Lx, self.Ly, self.k = z.Lx, z.Ly, z.k
        # ridge: D = |a x - y + b| / norm < L/4, pot A cos(2 pi D / L)   (internal (x, y) as in the Z class)
        self._a, self._b = z.r, z.b
        self._norm = math.sqrt(z.r ** 2 + 1)
        self._A, self._L = z.A, z.L
        self._states = list(zip(z.state_y, z.state_A, z.state_L))
        self._slope = (z.slope_start_y, z.slope_end_y, z.slope_strength_y)
        self.gravity = z.gravity

    # ------------------------------------------------------------------
    def kick_range(self, lam=None):
        """Transverse (x) range for initial kicks: the full channel width."""
        return (0.0, self.Lx)

    def kick_extra(self, lam=None):
        """Coordinates after (y, x) of a kick point: w near one of the two minima, channel at random."""
        sign = 1.0 if np.random.rand() < 0.5 else -1.0
        return np.array([sign * self.w0 + self.kick_w_sd * np.random.randn()])

    def switch(self, w):
        """(s, ds/dw)."""
        t = math.tanh(w / self.delta)
        return 0.5 * (1.0 - t), -0.5 * (1.0 - t * t) / self.delta

    def info(self):
        return dict(w0=self.w0, dV_w=self.dV_w, delta=self.delta, kick_w_sd=self.kick_w_sd,
                    z=self.z.barrier_info())

    # ------------------------------------------------------------------
    def _ridge(self, x, y):
        """Ridge of the Z class at internal (x, y): (pot, fx, fy)."""
        a, b, norm = self._a, self._b, self._norm
        u = a * x - y + b
        D = abs(u) / norm
        q = self._L / 4
        if D >= q:
            return 0.0, 0.0, 0.0
        arg = 2 * math.pi * D / self._L
        pot = self._A * math.cos(arg)
        dV_dD = -2 * math.pi * self._A / self._L * math.sin(arg)
        sgn = 1.0 if (y - a * x - b) > 0 else (-1.0 if (y - a * x - b) < 0 else 0.0)
        return pot, sgn * dV_dD * a / norm, -sgn * dV_dD / norm

    def potential_and_force(self, ph):
        """Potential (without the gravity term, as in the Z class) and force (fy, fx, fw) at ph = (pos, vel)."""
        pos = ph[0]
        y, x, w = float(pos[0]), float(pos[1]), float(pos[2])
        k = self.k
        pot = fx = fy = 0.0
        # walls
        if x < 0:
            pot += 0.5 * k * x * x; fx += -k * x
        elif x > self.Lx:
            d = x - self.Lx; pot += 0.5 * k * d * d; fx += -k * d
        if y < 0:
            pot += 0.5 * k * y * y; fy += -k * y
        elif y > self.Ly:
            d = y - self.Ly; pot += 0.5 * k * d * d; fy += -k * d
        # state wells
        for Y, amp, ell in self._states:
            D = abs(y - Y)
            if D < ell / 4:
                arg = 2 * math.pi * D / ell
                pot += amp * math.cos(arg)
                sg = 1.0 if y > Y else (-1.0 if y < Y else 0.0)
                fy += amp * 2 * math.pi / ell * math.sin(arg) * sg
        # gravity (force only) and slope
        fy -= self.gravity
        y0, y1, g = self._slope
        if y >= y0:
            if y <= y1:
                pot += g * (y - y0) / (y1 - y0); fy += -g / (y1 - y0)
            else:
                pot += g
        # the two ridges, switched by w
        s, ds = self.switch(w)
        p1, f1x, f1y = self._ridge(x, y)
        p2, f2x, f2y = self._ridge(self.Lx - x, y)               # mirror image: d/dx flips sign
        pot += s * p1 + (1 - s) * p2
        fx += s * f1x - (1 - s) * f2x
        fy += s * f1y + (1 - s) * f2y
        # w: switch and double well
        r = w / self.w0
        pot += self.dV_w * (r * r - 1) ** 2
        fw = -ds * (p1 - p2) - 4 * self.dV_w * r * (r * r - 1) / self.w0
        return pot, np.array([fy, fx, fw])

    def potential(self, ph):
        return self.potential_and_force(ph)[0]

    def force(self, ph):
        return self.potential_and_force(ph)[1]

    def effective_potential(self, ph):
        """Potential consistent with the force (gravity added back, as in the Z class)."""
        return self.potential(ph) + self.gravity * float(ph[0][0])

    # ------------------------------------------------------------------
    def reference_potential_and_force(self, ph):
        """The same from the Z class itself (slow; for the tests)."""
        y, x, w = (float(v) for v in ph[0])
        zero = np.zeros(2)
        p1, f1 = self.z.potential_and_force((np.array([y, x]), zero))
        p2, f2 = self.z.potential_and_force((np.array([y, self.Lx - x]), zero))
        s, ds = self.switch(w)
        r = w / self.w0
        pot = s * p1 + (1 - s) * p2 + self.dV_w * (r * r - 1) ** 2
        fy = s * f1[0] + (1 - s) * f2[0]
        fx = s * f1[1] - (1 - s) * f2[1]
        fw = -ds * (p1 - p2) - 4 * self.dV_w * r * (r * r - 1) / self.w0
        return pot, np.array([fy, fx, fw])
