# -*- coding: utf-8 -*-
"""Piecewise 2D potential for a Langevin particle around a rectangular obstacle.

Geometry (matching the sketch):

    A -------- I -------- J -------- B
    |          |  upper   |          |
    |          E----------F          |
    |          | forbidden|          |
    |          G----------H          |
    |          |  lower   |          |
    D -------- K -------- L -------- C

The background potential is zero in the left strip AIDK and the right strip
JBCL. In the upper channel IEJF it decreases linearly from 0 at IE to
V_top_mid at the x-midpoint, then increases linearly back to 0 at JF.
In the lower channel GKHL it increases linearly from 0 at GK to V_bottom_mid
at the x-midpoint, then decreases linearly back to 0 at HL.

Outer and inner wall barriers are added on top of that background landscape.
The wall barrier is Gaussian in the signed distance d:

    U_wall(d) = a * exp(-(d - b)^2 / (2*c^2))

With the default b = 0, the barrier is centered on the geometric wall, has a
maximum value a, and decays smoothly with distance from the wall. In
particular, the wall energy is bounded by a; it cannot blow up exponentially
inside the forbidden region when the full grid is plotted.

The force is always computed analytically as -grad(V), except at the intended
piecewise-linear kinks and signed-distance medial lines, where the force is
not differentiable by construction.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


class PotentialRectangularChannels:
    """2D piecewise-linear channel potential with rectangular wall barriers."""

    def __init__(self, 
                 x_outer_left=-3.0, x_outer_right=3.0, y_outer_bottom=-2.0, y_outer_top=2.0,
                 x_inner_left=-1.0, x_inner_right=1.0, y_inner_bottom=-0.6, y_inner_top=0.6,
                 V_top_mid=-0.5, V_bottom_mid=0.5,
                 wall_a=100.0, wall_b=0.0, wall_c=0.15,                    
                 desc="Rectangular two-channel potential"):
        """Set up the potential.

        Parameters are stored in ``self.params`` so they can be modified in the
        same style as the supplied example class.

        Geometry parameters
        -------------------
        x_outer_left, x_outer_right : float
            x coordinates of AD and BC.
        y_outer_bottom, y_outer_top : float
            y coordinates of DC and AB.
        x_inner_left, x_inner_right : float
            x coordinates of EG/IK and FH/JL.
        y_inner_bottom, y_inner_top : float
            y coordinates of GH and EF.

        Landscape parameters
        --------------------
        V_top_mid : float
            Potential at the midpoint of the upper channel. Usually negative.
        V_bottom_mid : float
            Potential at the midpoint of the lower channel. Usually positive.

        Wall parameters
        ---------------
        wall_a : float
            Amplitude of the Gaussian wall barrier. For wall_b = 0 this is
            the maximum wall energy at d = 0.
        wall_b : float
            Center of the Gaussian in signed-distance coordinates. Normally 0.
        wall_c : float
            Standard deviation / width of the Gaussian barrier.
        """
        self.desc = desc

        # self.params = {
        #     # Outer rectangle ABCD
        #     "x_outer_left": -3.0,
        #     "x_outer_right": 3.0,
        #     "y_outer_bottom": -2.0,
        #     "y_outer_top": 2.0,

        #     # Inner forbidden rectangle EFGH
        #     "x_inner_left": -1.0,
        #     "x_inner_right": 1.0,
        #     "y_inner_bottom": -0.6,
        #     "y_inner_top": 0.6,

        #     # Background landscape
        #     "V_top_mid": -.5,
        #     "V_bottom_mid": .5,

        #     # Gaussian wall barrier
        #     # U(d) = a * exp(-(d-b)^2 / (2*c^2))
        #     "wall_a": 100.0,
        #     "wall_b": 0.0,
        #     "wall_c": 0.15,
        # }
        
        self.params = {
            # Outer rectangle ABCD
            "x_outer_left": x_outer_left,
            "x_outer_right": x_outer_right,
            "y_outer_bottom": y_outer_bottom,
            "y_outer_top": y_outer_top,
            
            # Inner forbidden rectangle EFGH
            "x_inner_left": x_inner_left,
            "x_inner_right": x_inner_right,
            "y_inner_bottom": y_inner_bottom,
            "y_inner_top": y_inner_top,

            # Background landscape
            "V_top_mid": V_top_mid,
            "V_bottom_mid": V_bottom_mid,

            # Gaussian wall barrier
            "wall_a": wall_a,
            "wall_b": wall_b,
            "wall_c": wall_c,
        }

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _xy_from_ph(ph):
        """Extract x and y using the same phase-point convention as the example."""
        x = float(ph[0][0])
        y = float(ph[0][1])
        return x, y

    def _validate_geometry(self):
        """Raise ValueError if the rectangle geometry is inconsistent."""
        p = self.params
        if not (p["x_outer_left"] < p["x_inner_left"] < p["x_inner_right"] < p["x_outer_right"]):
            raise ValueError("Require x_outer_left < x_inner_left < x_inner_right < x_outer_right")
        if not (p["y_outer_bottom"] < p["y_inner_bottom"] < p["y_inner_top"] < p["y_outer_top"]):
            raise ValueError("Require y_outer_bottom < y_inner_bottom < y_inner_top < y_outer_top")
        if p["wall_a"] < 0.0:
            raise ValueError("wall_a must be non-negative")
        if p["wall_c"] <= 0.0:
            raise ValueError("wall_c must be positive")

    @staticmethod
    def distance_from_wall_segment(point, endpoint_a, endpoint_b):
        """Distance from a point to a finite line segment.

        Parameters
        ----------
        point : array-like, shape (2,)
        endpoint_a, endpoint_b : array-like, shape (2,)

        Returns
        -------
        distance : float
            Euclidean distance to the closest point on the segment.
        direction : np.ndarray, shape (2,)
            Unit vector from the closest point on the segment toward ``point``.
            It is [0, 0] when the point lies exactly on the segment.
        closest : np.ndarray, shape (2,)
            Closest point on the segment.
        """
        point = np.asarray(point, dtype=float)
        a = np.asarray(endpoint_a, dtype=float)
        b = np.asarray(endpoint_b, dtype=float)

        ab = b - a
        ab2 = np.dot(ab, ab)
        if ab2 == 0.0:
            raise ValueError("A wall segment must have non-zero length")

        t = np.dot(point - a, ab) / ab2
        t = np.clip(t, 0.0, 1.0)
        closest = a + t * ab

        delta = point - closest
        distance = float(np.linalg.norm(delta))
        if distance > 0.0:
            direction = delta / distance
        else:
            direction = np.zeros(2, dtype=float)

        return distance, direction, closest

    # ------------------------------------------------------------------
    # Background piecewise-linear landscape
    # ------------------------------------------------------------------
    def _background_potential_and_force(self, x, y):
        """Return the background potential and force, excluding all walls."""
        p = self.params
        xl = p["x_inner_left"]
        xr = p["x_inner_right"]
        yb = p["y_inner_bottom"]
        yt = p["y_inner_top"]
        xm = 0.5 * (xl + xr)

        v = 0.0
        force = np.zeros(2, dtype=float)

        # AIDK and JBCL are flat at V = 0.
        if not (xl < x < xr):
            return v, force

        # Upper channel IEJF: 0 -> V_top_mid -> 0.
        if y >= yt:
            vmid = p["V_top_mid"]
            if x <= xm:
                slope = vmid / (xm - xl)
                v = slope * (x - xl)
            else:
                slope = -vmid / (xr - xm)
                v = vmid + slope * (x - xm)
            force[0] = -slope
            return v, force

        # Lower channel GKHL: 0 -> V_bottom_mid -> 0.
        if y <= yb:
            vmid = p["V_bottom_mid"]
            if x <= xm:
                slope = vmid / (xm - xl)
                v = slope * (x - xl)
            else:
                slope = -vmid / (xr - xm)
                v = vmid + slope * (x - xm)
            force[0] = -slope
            return v, force

        # Inside the forbidden rectangle the background is set to zero.
        # The inner wall barrier dominates there.
        return v, force

    # ------------------------------------------------------------------
    # Wall model
    # ------------------------------------------------------------------
    @staticmethod
    def exponential(d, a, b, c):
        """Exponential profile retained as a utility function."""
        return a * np.exp(b - c * d)

    @staticmethod
    def exponential_force(d, a, b, c):
        """Derivative dV/dd of :meth:`exponential`."""
        return -a * c * np.exp(b - c * d)

    @staticmethod
    def gaussian(d, a, b, c):
        """Gaussian wall profile as a function of distance d."""
        return a * np.exp(-((d - b) ** 2) / (2.0 * c ** 2))

    @staticmethod
    def gaussian_force(d, a, b, c):
        """Derivative dV/dd of :meth:`gaussian`.

        This follows the convention in the requested helper function. The
        spatial force is therefore ``-gaussian_force(...) * grad(d)``.
        """
        return (
            -a / c ** 2
            * (d - b)
            * np.exp(-((d - b) ** 2) / (2.0 * c ** 2))
        )

    def _repel_from_wall_potential_and_force(self, signed_distance, grad_distance):
        """Gaussian wall energy and force from a signed distance.

        ``signed_distance`` is positive on the allowed side and negative on
        the forbidden side. ``grad_distance`` is grad(d). With the default
        ``wall_b = 0``, the Gaussian is centered on the wall and is bounded by
        ``wall_a``.

        ``gaussian_force`` returns dV/dd, so the Cartesian force is

            F = -(dV/dd) * grad(d).
        """
        a = self.params["wall_a"]
        b = self.params["wall_b"]
        c = self.params["wall_c"]

        v = self.gaussian(signed_distance, a, b, c)
        dV_dd = self.gaussian_force(signed_distance, a, b, c)
        force = -dV_dd * np.asarray(grad_distance, dtype=float)
        return float(v), force

    def _outer_wall_potential_and_force(self, x, y):
        """Sum barriers from the four outer walls of ABCD."""
        p = self.params
        v = 0.0
        force = np.zeros(2, dtype=float)

        # AD: allowed side is +x.
        dv, df = self._repel_from_wall_potential_and_force(
            x - p["x_outer_left"], np.array([1.0, 0.0])
        )
        v += dv
        force += df

        # BC: allowed side is -x.
        dv, df = self._repel_from_wall_potential_and_force(
            p["x_outer_right"] - x, np.array([-1.0, 0.0])
        )
        v += dv
        force += df

        # DC: allowed side is +y.
        dv, df = self._repel_from_wall_potential_and_force(
            y - p["y_outer_bottom"], np.array([0.0, 1.0])
        )
        v += dv
        force += df

        # AB: allowed side is -y.
        dv, df = self._repel_from_wall_potential_and_force(
            p["y_outer_top"] - y, np.array([0.0, -1.0])
        )
        v += dv
        force += df

        return v, force

    def _inner_signed_distance_and_gradient(self, x, y):
        """Signed distance to the forbidden rectangle EFGH.

        Returns
        -------
        signed_distance : float
            Positive outside EFGH, zero on its boundary, negative inside.
        grad_distance : np.ndarray, shape (2,)
            Gradient of the signed distance. It points away from the forbidden
            rectangle, i.e. toward the accessible region.

        Notes
        -----
        The closest inner wall is found using ``distance_from_wall_segment``.
        Force continuity at exact corner/medial-line ties is not required here.
        """
        p = self.params
        xl = p["x_inner_left"]
        xr = p["x_inner_right"]
        yb = p["y_inner_bottom"]
        yt = p["y_inner_top"]

        point = np.array([x, y], dtype=float)

        # E, F, H, G in the notation of the sketch.
        e = np.array([xl, yt])
        f = np.array([xr, yt])
        h = np.array([xr, yb])
        g = np.array([xl, yb])

        # Segment name, endpoints, outward normal from the forbidden rectangle.
        walls = (
            (e, f, np.array([0.0, 1.0])),   # EF, top
            (f, h, np.array([1.0, 0.0])),   # FH, right
            (h, g, np.array([0.0, -1.0])),  # HG, bottom
            (g, e, np.array([-1.0, 0.0])),  # GE, left
        )

        distances = []
        directions = []
        for a, b, outward_normal in walls:
            dist, direction, _ = self.distance_from_wall_segment(point, a, b)
            distances.append(dist)
            directions.append((direction, outward_normal))

        idx = int(np.argmin(distances))
        min_dist = float(distances[idx])
        direction, outward_normal = directions[idx]

        inside = (xl < x < xr) and (yb < y < yt)

        if inside:
            # Inside the obstacle, d is negative and grad(d) is the outward
            # normal of the nearest wall.
            return -min_dist, outward_normal.copy()

        # Outside: ordinary distance to the closest boundary point. Its
        # gradient points away from the obstacle. Exactly on the wall, use
        # the wall's known outward normal because the radial direction is zero.
        if min_dist > 0.0:
            return min_dist, direction.copy()
        return 0.0, outward_normal.copy()

    def _inner_wall_potential_and_force(self, x, y):
        """Barrier produced by the inner forbidden rectangle EFGH."""
        signed_distance, grad_distance = self._inner_signed_distance_and_gradient(x, y)
        return self._repel_from_wall_potential_and_force(
            signed_distance, grad_distance
        )

    # ------------------------------------------------------------------
    # Public API, matching the structure of the supplied example
    # ------------------------------------------------------------------
    def potential(self, ph):
        """Evaluate the total potential energy."""
        self._validate_geometry()
        x, y = self._xy_from_ph(ph)

        v_bg, _ = self._background_potential_and_force(x, y)
        v_outer, _ = self._outer_wall_potential_and_force(x, y)
        v_inner, _ = self._inner_wall_potential_and_force(x, y)

        return float(v_bg + v_outer + v_inner)

    def force(self, ph):
        """Evaluate the total force vector = -grad(V)."""
        self._validate_geometry()
        x, y = self._xy_from_ph(ph)

        _, f_bg = self._background_potential_and_force(x, y)
        _, f_outer = self._outer_wall_potential_and_force(x, y)
        _, f_inner = self._inner_wall_potential_and_force(x, y)

        return f_bg + f_outer + f_inner

    def potential_and_force(self, ph):
        """Evaluate potential and force in one call."""
        self._validate_geometry()
        x, y = self._xy_from_ph(ph)

        v_bg, f_bg = self._background_potential_and_force(x, y)
        v_outer, f_outer = self._outer_wall_potential_and_force(x, y)
        v_inner, f_inner = self._inner_wall_potential_and_force(x, y)

        return (
            float(v_bg + v_outer + v_inner),
            f_bg + f_outer + f_inner,
        )

    # ------------------------------------------------------------------
    # Optional plotting helper
    # ------------------------------------------------------------------
    def plot_pot(self, ngrid=300, wall_clip=None):
        """Plot the potential and force field over the outer rectangle.

        Parameters
        ----------
        ngrid : int
            Grid resolution along each axis.
        wall_clip : float or None
            If given, clip displayed potential values to this maximum. This is
            useful if the Gaussian wall barriers are chosen much higher than
            the channel landscape.
        """
        self._validate_geometry()
        p = self.params

        xs = np.linspace(p["x_outer_left"], p["x_outer_right"], ngrid)
        ys = np.linspace(p["y_outer_bottom"], p["y_outer_top"], ngrid)
        xx, yy = np.meshgrid(xs, ys)

        vv = np.empty_like(xx)
        fx = np.empty_like(xx)
        fy = np.empty_like(xx)

        for j in range(ngrid):
            for i in range(ngrid):
                ph = (np.array([xx[j, i], yy[j, i]]), np.array([0.0, 0.0]))
                vv[j, i], ff = self.potential_and_force(ph)
                fx[j, i], fy[j, i] = ff

        vv_display = np.minimum(vv, wall_clip) if wall_clip is not None else vv

        fig1, ax1 = plt.subplots()
        c1 = ax1.pcolormesh(xx, yy, vv_display, shading="auto")
        fig1.colorbar(c1, ax=ax1, label="V(x, y)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_title("Potential energy surface")
        ax1.set_aspect("equal")

        # Draw inner obstacle.
        ax1.plot(
            [p["x_inner_left"], p["x_inner_right"], p["x_inner_right"],
             p["x_inner_left"], p["x_inner_left"]],
            [p["y_inner_top"], p["y_inner_top"], p["y_inner_bottom"],
             p["y_inner_bottom"], p["y_inner_top"]],
            "k-",
        )

        fig2, ax2 = plt.subplots()
        stride = max(1, ngrid // 30)
        ax2.quiver(
            xx[::stride, ::stride], yy[::stride, ::stride],
            fx[::stride, ::stride], fy[::stride, ::stride]
        )
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("Force field")
        ax2.set_aspect("equal")

        ax2.plot(
            [p["x_inner_left"], p["x_inner_right"], p["x_inner_right"],
             p["x_inner_left"], p["x_inner_left"]],
            [p["y_inner_top"], p["y_inner_top"], p["y_inner_bottom"],
             p["y_inner_bottom"], p["y_inner_top"]],
            "k-",
        )

        plt.show()


if __name__ == "__main__":
    pot = PotentialRectangularChannels()

    # Example parameter modifications:
    # pot.params["V_top_mid"] = -8.0
    # pot.params["V_bottom_mid"] = 8.0
    # pot.params["wall_a"] = 200.0
    # pot.params["wall_b"] = 0.0
    # pot.params["wall_c"] = 0.05

    pot.plot_pot(ngrid=250, wall_clip=20.0)