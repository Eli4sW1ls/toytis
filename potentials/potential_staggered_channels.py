# -*- coding: utf-8 -*-
"""Two-channel potential with staggered barriers (REPPTIS vs staple test case).

Same box, obstacle and Gaussian walls as
:class:`potentials.potential_rectangular_channels.PotentialRectangularChannels`;
only the background landscape inside the two channels differs. With
``xl``/``xr`` the channel ends and ``a = x_kink``:

    upper channel (IEJF):  0 at xl -> +V_peak at xl+a -> -V_dip at xr-a -> 0 at xr
    lower channel (GKHL):  0 at xl -> -V_dip at xl+a -> +V_peak at xr-a -> 0 at xr

with each piece a half-cosine in x (zero slope at every node, so the force
is continuous), flat in y. The two strips stay flat at V = 0.

Why this shape
--------------
* Every real path climbs ``V_peak`` in its own channel, so the A -> B rate goes
  as ~exp(-beta V_peak).
* At every x one channel goes uphill and the other downhill, so the type of a
  turn (maximum or minimum of x) tells which channel it happened in. Staple
  paths, which are concatenated per turn, keep the channel identity.
* At each x the lower of the two channels dominates the equilibrium
  population: the lower channel before x = 0, the upper channel after it.
  The projected profile F(x) = -kT ln(exp(-beta V_up) + exp(-beta V_low)) only
  peaks where the two cross, at x = 0, at (V_peak - V_dip)/2 - kT ln 2.
  REPPTIS, which loses memory at every interface, sees that much lower
  barrier: it concatenates "up the lower channel" with "down the upper
  channel", a path that cannot occur.
* The landscape maps onto itself under x -> -x with the channels swapped, so
  A -> B and B -> A are equivalent and both channels are equally likely.

Defaults (T = 0.07): V_peak = 7.1 kT, V_dip = 2.1 kT, projected barrier
1.8 kT, so REPPTIS can overestimate the rate by up to ~200x when the
interfaces inside the channels are dense (spacing ~0.2).

Tested RETIS benchmark settings (same landscape in kT units, ~10x shorter
paths than T = 0.07):
    PotentialStaggeredChannels(V_peak=0.5*0.25/0.07, V_dip=3*0.25)
    temperature 0.25, friction 1, dt 0.02
    interfaces [-1.6, -1.0, -0.8, ..., 0.8, 1.0, 1.6]
(see notebooks/staggered_channels_run.py). Friction 1 gives a ballistic
length of ~2.5 interface spacings: at friction 5 the low channel's diffusive
counter-slope turns swamp the other channel's turn states at its barrier top,
so staple paths would tunnel as well.
State A/B at x = -/+1.6, clear of the obstacle wall: an interface at
x = -/+1.0 can only be crossed at the narrow channel mouths, which makes
[0-] paths very long.
"""

import numpy as np

from potentials.potential_rectangular_channels import PotentialRectangularChannels


class PotentialStaggeredChannels(PotentialRectangularChannels):
    """Rectangular two-channel potential with staggered barriers and dips."""

    # Transverse (y) range for initial kicks and the RETIS dummy path: the
    # middle of the lower channel, clear of the walls.
    kick_range = (-1.4, -1.2)

    def __init__(self, V_peak=0.5, V_dip=0.15, x_kink=0.4,
                 desc="Staggered two-channel potential", **kwargs):
        """Set up the potential.

        Parameters
        ----------
        V_peak : float
            Barrier height in each channel.
        V_dip : float
            Depth of the dip in each channel, opposite the other channel's
            barrier. Keep it to a few kT, or it becomes a metastable trap.
        x_kink : float
            Distance from the channel ends to the barrier top / dip bottom.
        **kwargs
            Geometry and wall parameters of PotentialRectangularChannels.
        """
        super().__init__(desc=desc, **kwargs)
        self.params["V_peak"] = V_peak
        self.params["V_dip"] = V_dip
        self.params["x_kink"] = x_kink

    def _background_potential_and_force(self, x, y):
        """Return the background potential and force, excluding all walls."""
        p = self.params
        xl = p["x_inner_left"]
        xr = p["x_inner_right"]

        # The strips and the forbidden rectangle are flat at V = 0.
        if not (xl < x < xr) or p["y_inner_bottom"] < y < p["y_inner_top"]:
            return 0.0, np.zeros(2, dtype=float)

        nodes = [xl, xl + p["x_kink"], xr - p["x_kink"], xr]
        if y >= p["y_inner_top"]:
            values = [0.0, p["V_peak"], -p["V_dip"], 0.0]
        else:
            values = [0.0, -p["V_dip"], p["V_peak"], 0.0]

        # Half-cosine between consecutive nodes: zero slope at every node.
        k = min(int(np.searchsorted(nodes, x)), 3)
        width = nodes[k] - nodes[k - 1]
        dv = values[k] - values[k - 1]
        t = np.pi * (x - nodes[k - 1]) / width
        v = values[k - 1] + 0.5 * dv * (1.0 - np.cos(t))
        slope = 0.5 * dv * np.pi / width * np.sin(t)
        return v, np.array([-slope, 0.0])
