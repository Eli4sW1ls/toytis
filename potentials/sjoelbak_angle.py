"""Sjoelbak potential with a barrier of controllable tilt angle.

This is the ``sjoelbak`` potential (two stable wells at the ends of a
rectangular channel, separated by a straight cosine ridge) rewritten so that
the *angle* of the ridge is the control parameter instead of the raw slope
``r``.

Geometry / conventions
----------------------
The channel runs along ``y`` (length ``Ly``), which is also the order
parameter used by :class:`order.OrderX`.  The transverse direction is ``x``
(width ``Lx``); ``Lx`` is what is drawn as the *height* of the grid by
``plot_potential``.

The tilt ``angle`` is measured from the transverse (``x``) axis, i.e. from
the orientation of a barrier that is perpendicular to the order parameter:

* ``angle = 0``    -> flat ridge, ``y = y_center``, perpendicular to lambda.
* ``angle = 45``   -> line slope ``a = -1``, i.e. the original sjoelbak.
* ``angle -> 90``  -> ridge parallel to the channel (degenerate, forbidden).

The line slope is ``a = -tan(angle)``, so positive angles descend with ``x``
exactly like the original ``r = -1``.  Negative angles mirror the barrier.

Fitting the barrier in the box
------------------------------
The ridge is an infinite line, so it always blocks the channel completely;
the real constraint is that its *footprint in y* must stay clear of the two
state wells.  Over the channel width the line itself spans
``Lx * |tan(angle)|`` in ``y``, and the cosine ridge adds a half-thickness of
``(L/4) / cos(angle)`` on each side (the perpendicular half-width ``L/4``
seen along ``y``).  With the barrier centred at ``y_center`` and ``h`` the
distance from ``y_center`` to the nearest state-well edge, the barrier fits
as long as::

    (Lx/2) * tan(angle) + (L/4) / cos(angle) <= h

which is solved by :meth:`max_angle_for_width`.  Beyond that threshold angle
the barrier cannot fit at the requested width.  With ``auto_shrink=True``
(default) the channel width ``Lx`` -- the grid height in the usual plot -- is
then reduced to the largest value that still fits,
:meth:`width_for_angle`, and a warning is logged.  Above
:attr:`angle_hard_limit` the ridge half-thickness alone exceeds ``h`` and no
width can help; that raises a ``ValueError``.

A note on gravity
-----------------
As in the original potential, the constant ``gravity`` pull is added to the
*force* only and is deliberately left out of :meth:`potential`, so
``force != -grad(potential)`` by a constant ``(0, -gravity)``.  This is kept
for bit-for-bit compatibility with ``sjoelbak.py``.  Use
:meth:`effective_potential` to get the potential the dynamics actually
integrates, ``potential + gravity * y``; that is the one to use for free
energy / barrier-height estimates.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


class RectangularGridWithAngledBarrierPotential:
    """Sjoelbak potential with a barrier of adjustable tilt angle."""

    def __init__(self, Lx=0.239, Ly=0.9, k=2000, angle=45.0, A=0.3, L=0.2,
                 y_center=0.45, degrees=True,
                 slope_start_y=0.1, slope_end_y=0.825, slope_strength_y=0.2,
                 state_y=(0.025, 0.875), state_A=(-0.15, -0.15),
                 state_L=(0.2, 0.2), gravity=0.2,
                 clearance=0.0, auto_shrink=True):
        """Initialize the potential with given parameters.

        Parameters
        ----------
        Lx : float
            The requested width of the grid in the x direction (the "height"
            of the grid in the standard plot).  May be shrunk, see
            `auto_shrink`.
        Ly : float
            The length of the grid in the y direction.
        k : float
            The coefficient for the harmonic potential term outside the
            boundaries.
        angle : float
            Tilt of the barrier, measured from the x axis (0 = perpendicular
            to the order parameter y, 45 = the original sjoelbak).  Must
            satisfy |angle| < 90.
        A : float
            The amplitude (height) of the barrier ridge.
        L : float
            The length scale for the barrier potential; the ridge has a
            perpendicular half-width of L/4.
        y_center : float
            The y coordinate the barrier line passes through at x = Lx/2.
        degrees : bool
            Whether `angle` is given in degrees (default) or radians.
        slope_start_y, slope_end_y, slope_strength_y : float
            The gradual slope between the two states, as in `sjoelbak.py`.
        state_y, state_A, state_L : tuple of float
            Centre, amplitude and length scale of the state A / state B
            cosine wells.  These were hardcoded in `sjoelbak.py`; they are
            parameters here because they set how much room the barrier has.
        gravity : float
            Constant force -gravity along y.  Not included in `potential`,
            see the module docstring.
        clearance : float
            Extra margin in y that the barrier must keep from the state wells.
        auto_shrink : bool
            If True (default), shrink `Lx` when the requested angle does not
            fit; if False, raise a ValueError instead.
        """
        self.Ly = Ly
        self.k = k
        self.A = A
        self.L = L
        self.y_center = y_center
        self.slope_start_y = slope_start_y
        self.slope_end_y = slope_end_y
        self.slope_strength_y = slope_strength_y
        self.state_y = tuple(state_y)
        self.state_A = tuple(state_A)
        self.state_L = tuple(state_L)
        self.gravity = gravity
        self.clearance = clearance

        # --- angle bookkeeping ------------------------------------------
        self.angle = float(np.radians(angle) if degrees else angle)
        if not np.isfinite(self.angle) or abs(self.angle) >= 0.5 * np.pi:
            raise ValueError(
                'angle must satisfy |angle| < 90 degrees, got {}'.format(angle)
            )
        self.r = -np.tan(self.angle)  # line slope, r = -1 for angle = 45 deg

        # --- room available for the barrier in the y direction ----------
        # The barrier may not eat into the state wells.
        self.band = (self.state_y[0] + self.state_L[0] / 4 + clearance,
                     self.state_y[1] - self.state_L[1] / 4 - clearance)
        if not self.band[0] < self.y_center < self.band[1]:
            raise ValueError(
                'y_center = {} lies outside the usable band {}'
                .format(self.y_center, self.band)
            )
        # half-room: distance from the barrier centre to the nearest well
        self.half_room = min(self.y_center - self.band[0],
                             self.band[1] - self.y_center)
        if self.half_room <= self.L / 4:
            raise ValueError(
                'the ridge half-thickness L/4 = {:.4g} does not even fit in '
                'the available half-room {:.4g}; reduce L or move the states'
                .format(self.L / 4, self.half_room)
            )

        # Above this angle no channel width whatsoever fits the barrier,
        # because the ridge half-thickness (L/4)/cos(angle) alone exceeds the
        # available room.
        self.angle_hard_limit = np.arccos((self.L / 4) / self.half_room)

        # --- fit the barrier, shrinking Lx if we must -------------------
        self.Lx_requested = Lx
        self.angle_threshold = self.max_angle_for_width(Lx)
        if abs(self.angle) <= self.angle_threshold:
            self.Lx = Lx
            self.shrunk = False
        else:
            if abs(self.angle) >= self.angle_hard_limit:
                raise ValueError(
                    'angle {:.4g} deg is at or beyond the hard limit {:.4g} '
                    'deg: the ridge is thicker than the room between the '
                    'states no matter how narrow the channel is. Reduce L, '
                    'move the states apart, or use a smaller angle.'
                    .format(np.degrees(self.angle),
                            np.degrees(self.angle_hard_limit))
                )
            if not auto_shrink:
                raise ValueError(
                    'angle {:.4g} deg exceeds the threshold {:.4g} deg for '
                    'Lx = {:.4g}; pass auto_shrink=True or use Lx <= {:.4g}'
                    .format(np.degrees(self.angle),
                            np.degrees(self.angle_threshold), Lx,
                            self.width_for_angle(self.angle))
                )
            self.Lx = self.width_for_angle(self.angle)
            self.shrunk = True
            logger.warning(
                'Barrier angle %.4g deg exceeds the threshold %.4g deg for '
                'Lx = %.4g; shrinking the grid height to Lx = %.4g so the '
                'barrier still clears the states.',
                np.degrees(self.angle), np.degrees(self.angle_threshold),
                Lx, self.Lx
            )

        # Line through M = (Lx/2, y_center) with slope r: y = r*x + b.
        self.M = (self.Lx / 2, self.y_center)
        self.b = self.M[1] - self.r * self.M[0]

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------
    @property
    def angle_deg(self):
        """The barrier tilt in degrees."""
        return np.degrees(self.angle)

    @property
    def angle_threshold_deg(self):
        """Largest tilt (deg) that fits at the *requested* width."""
        return np.degrees(self.angle_threshold)

    @property
    def angle_hard_limit_deg(self):
        """Tilt (deg) beyond which no channel width fits the barrier."""
        return np.degrees(self.angle_hard_limit)

    def max_angle_for_width(self, Lx):
        """Largest |angle| (radians) whose barrier fits at width `Lx`.

        Solves ``(Lx/2) tan(t) + (L/4)/cos(t) = half_room`` by rewriting it
        as ``half_room cos(t) - (Lx/2) sin(t) = L/4``, i.e.
        ``R cos(t + phi) = L/4`` with ``R = hypot(half_room, Lx/2)`` and
        ``phi = atan2(Lx/2, half_room)``.
        """
        rad = np.hypot(self.half_room, 0.5 * Lx)
        phi = np.arctan2(0.5 * Lx, self.half_room)
        ratio = (self.L / 4) / rad
        if ratio >= 1.0:  # not even a flat barrier fits
            return 0.0
        return max(0.0, np.arccos(ratio) - phi)

    def width_for_angle(self, angle, degrees=False):
        """Largest channel width `Lx` whose barrier fits at `angle`."""
        ang = abs(np.radians(angle) if degrees else angle)
        if ang >= self.angle_hard_limit:
            return 0.0
        if ang < 1e-12:  # flat barrier: any width fits
            return np.inf
        return 2.0 * (self.half_room - (self.L / 4) / np.cos(ang)) / np.tan(ang)

    @property
    def angle_gap(self):
        """Tilt (radians) above which a free corridor opens in the y projection.

        The ridge covers an x interval of width ``(L/2)/sin(angle)`` at any
        given ``y``.  As soon as that is narrower than the channel,
        i.e. ``sin(angle) > L / (2 Lx)``, there is *no* value of ``y`` at
        which the ridge blocks the whole width.  Every path still has to
        cross the line, but the order parameter ``lambda = y`` no longer
        resolves the barrier: the projected free energy ``F(y)`` loses the
        ridge almost entirely.  Returns ``pi/2`` (never) if the channel is so
        narrow that the ridge always spans it.
        """
        ratio = self.L / (2 * self.Lx) if self.Lx > 0 else np.inf
        if ratio >= 1.0:
            return 0.5 * np.pi
        return np.arcsin(ratio)

    @property
    def angle_gap_deg(self):
        """:attr:`angle_gap` in degrees."""
        return np.degrees(self.angle_gap)

    def free_corridor(self):
        """``(y_from_A, y_from_B)``: how far each state reaches barrier-free.

        ``y_from_A`` is the highest ``y`` reachable from state A without
        touching the ridge (hugging the wall where the line is highest), and
        ``y_from_B`` the lowest ``y`` reachable from state B.  When
        ``y_from_A > y_from_B`` the two overlap, which happens exactly above
        :attr:`angle_gap`; interfaces placed in the overlap are crossed with
        probability ~1 from either side and carry no information.
        """
        thickness = (self.L / 4) * np.sqrt(self.r**2 + 1)
        ends = (self.b, self.r * self.Lx + self.b)
        return max(ends) - thickness, min(ends) + thickness

    def barrier_footprint(self):
        """Return ``(y_low, y_high)``, the y extent of the ridge in the box."""
        thickness = (self.L / 4) * np.sqrt(self.r**2 + 1)
        ends = (self.b, self.r * self.Lx + self.b)
        return min(ends) - thickness, max(ends) + thickness

    def barrier_info(self):
        """Summary of the barrier geometry, handy for logging/plots."""
        low, high = self.barrier_footprint()
        return {
            'angle_deg': self.angle_deg,
            'slope_r': self.r,
            'intercept_b': self.b,
            'Lx_requested': self.Lx_requested,
            'Lx': self.Lx,
            'shrunk': self.shrunk,
            'angle_threshold_deg': self.angle_threshold_deg,
            'angle_hard_limit_deg': self.angle_hard_limit_deg,
            'angle_gap_deg': self.angle_gap_deg,
            'free_corridor': self.free_corridor(),
            'usable_band': self.band,
            'half_room': self.half_room,
            'footprint_y': (low, high),
            'ridge_length': self.Lx / np.cos(self.angle),
            'line_y_at_x0': self.b,
            'line_y_at_xLx': self.r * self.Lx + self.b,
        }

    # ------------------------------------------------------------------
    # potential / force
    # ------------------------------------------------------------------
    def _walls(self, x):
        """Harmonic confinement of the rectangular grid. Returns (pot, f)."""
        pot_x = pot_y = fx = fy = 0.0
        if x[0] < 0:
            pot_x = 0.5 * self.k * (0 - x[0])**2
            fx = self.k * (0 - x[0])
        elif x[0] > self.Lx:
            pot_x = 0.5 * self.k * (x[0] - self.Lx)**2
            fx = -self.k * (x[0] - self.Lx)

        if x[1] < 0:
            pot_y = 0.5 * self.k * (0 - x[1])**2
            fy = self.k * (0 - x[1])
        elif x[1] > self.Ly:
            pot_y = 0.5 * self.k * (x[1] - self.Ly)**2
            fy = -self.k * (x[1] - self.Ly)

        return pot_x + pot_y, np.array([fx, fy])

    def _barrier(self, x):
        """Tilted cosine ridge. Returns (pot, f)."""
        a = self.r
        b = self.b
        norm = np.sqrt(a**2 + 1)
        D = abs(a * x[0] - x[1] + b) / norm
        if D >= self.L / 4:
            return 0.0, np.zeros(2)
        Dsgn = np.sign(x[1] - a * x[0] - b)
        pot = self.A * np.cos(2 * np.pi * D / self.L)
        # Force due to the barrier is by definition perpendicular to the line
        dV_dD = -2 * np.pi * self.A / self.L * np.sin(2 * np.pi * D / self.L)
        # the slope of the line is a, so the gradient is (-a, 1)
        f = Dsgn * dV_dD * np.array([a / norm, -1 / norm])
        return pot, f

    def _states(self, x):
        """The two cosine wells defining state A and state B."""
        pot = 0.0
        f = np.zeros(2)
        for Y, amp, ell in zip(self.state_y, self.state_A, self.state_L):
            D = np.abs(x[1] - Y)
            if D < ell / 4:
                pot += amp * np.cos(2 * np.pi * D / ell)
                dV_dD = amp * 2 * np.pi / ell * np.sin(2 * np.pi * D / ell)
                f += dV_dD * np.array([0.0, np.sign(x[1] - Y)])
        return pot, f

    def _slope(self, x):
        """Global slope between the two states."""
        if x[1] < self.slope_start_y:
            return 0.0, 0.0
        span = self.slope_end_y - self.slope_start_y
        if x[1] <= self.slope_end_y:
            return (self.slope_strength_y * (x[1] - self.slope_start_y) / span,
                    -self.slope_strength_y / span)
        # Above state B: constant potential (no force)
        return self.slope_strength_y, 0.0

    def potential_and_force(self, ph):
        """
        Calculate the potential and force at the given phasepoint.

        Parameters
        ----------
        ph : tuple (x, v) of numpy float arrays
            The phasepoint at which to calculate.

        Returns
        -------
        pot : float
            The potential at ph
        f : numpy float array
            The force vector acting in ph
        """
        x, _ = ph

        pot, f = self._walls(x)

        pot_b, f_b = self._barrier(x)
        pot += pot_b
        f = f + f_b

        pot_s, f_s = self._states(x)
        pot += pot_s
        f = f + f_s

        f += np.array([0., -self.gravity])  # gravity (force only, see module doc)

        pot_g, fy_g = self._slope(x)
        pot += pot_g
        f[1] += fy_g

        return pot, f

    def potential(self, ph):
        """
        Calculate the potential at the given phasepoint.

        Parameters
        ----------
        ph : tuple (x, v) of numpy float arrays
            The phasepoint at which to calculate.

        Returns
        -------
        pot : float
            The potential at ph
        """
        return self.potential_and_force(ph)[0]

    def force(self, ph):
        """
        Calculate the force at the given phasepoint.

        Parameters
        ----------
        ph : tuple (x, v) of numpy float arrays
            The phasepoint at which to calculate.

        Returns
        -------
        f : numpy float array
            The force vector acting in ph
        """
        return self.potential_and_force(ph)[1]

    def effective_potential(self, ph):
        """Potential that is consistent with :meth:`force`.

        `potential` omits gravity (kept that way for compatibility with
        `sjoelbak.py`); this adds it back, so that
        ``force == -grad(effective_potential)``.
        """
        return self.potential(ph) + self.gravity * ph[0][1]

    def barrier_top_profile(self, N=400):
        """Effective potential along the ridge line, inside the channel.

        Returns ``(s, y, U_eff)`` where `s` is the arc length along the ridge
        measured from ``x = 0``.  The minimum of `U_eff` is the cheapest
        crossing point, which for a tilted barrier sits against the wall at
        the low-y end of the ridge.
        """
        xs = np.linspace(0.0, self.Lx, N)
        ys = self.r * xs + self.b
        s = xs / np.cos(self.angle)
        u = np.array([self.effective_potential((np.array([xi, yi]),
                                                np.zeros(2)))
                      for xi, yi in zip(xs, ys)])
        return s, ys, u

    # ------------------------------------------------------------------
    # plotting (same interface as sjoelbak.py)
    # ------------------------------------------------------------------
    def plot_potential(self, ax):
        """Plots the potential.

        Parameters
        ----------
        ax : matplotlib axis object
            Axis on which to plot the potential

        """
        xvals, yvals = np.meshgrid(np.linspace(-.3*self.Ly, 1.3*self.Ly, 1000),
                                   np.linspace(-.3*self.Lx, 1.3*self.Lx, 1000))
        potvals = np.array([[self.potential_and_force((np.array([x, y]),
            np.array([0, 0])))[0] for x in yvals[:, 0]] for y in xvals[0]]).T
        g = ax.contourf(xvals, yvals, potvals)
        ax.contour(xvals, yvals, potvals, levels=[-5, 0, 5, 10, 15, 20], colors="black")
        return g

    def get_potential_plot(self, N=100):
        """Returns U(x,y) for given x,y grid."""
        xvals, yvals = np.meshgrid(np.linspace(-.1*self.Lx, 1.1*self.Lx, N),
                                   np.linspace(-.1*self.Ly, 1.1*self.Ly, N))
        potvals = np.array([[self.potential_and_force((np.array([x, y]),
            np.array([0, 0])))[0] for x in xvals[0]] for y in yvals[:, 0]])
        return xvals, yvals, potvals

    def plot_pot(self, intfs):
        fig0, ax0 = plt.subplots()
        self.plot_potential(ax0)
        plt.show()

        fig1, ax1 = plt.subplots()
        x_y = np.zeros([500, 500])
        i = 0
        for y in np.linspace(-.2*self.Lx, 1.2*self.Lx, 500):
            x_y[i] = np.array([self.potential((np.array([y, xx]), np.array([0, 0])))
                               for xx in np.linspace(-.2*self.Ly, 1.2*self.Ly, 500)])
            i += 1
        c1 = ax1.pcolorfast((-.2*self.Ly, 1.2*self.Ly), (-.2*self.Lx, 1.2*self.Lx), x_y, vmax=2)
        for intf in intfs:
            ax1.axvline(intf, ymin=-.1*self.Ly, ymax=1.1*self.Ly, color='orange', linewidth=0.5)
        fig1.colorbar(c1)
        plt.show()

        fig2, ax2 = plt.subplots()
        i = 0
        for y in np.linspace(-.2*self.Lx, 1.2*self.Lx, 500):
            x_y[i] = np.array([np.average(self.force((np.array([y, xx]), np.array([0, 0]))))
                               for xx in np.linspace(-.2*self.Ly, 1.2*self.Ly, 500)])
            i += 1
        c2 = ax2.pcolorfast((-.2*self.Ly, 1.2*self.Ly), (-.2*self.Lx, 1.2*self.Lx), x_y)
        for intf in intfs:
            ax2.axvline(intf, ymin=-.2*self.Lx, ymax=1.2*self.Lx, color='red')
        fig2.colorbar(c2)
        plt.show(block=True)
