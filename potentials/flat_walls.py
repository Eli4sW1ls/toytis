# -*- coding: utf-8 -*-
# Copyright (c) 2015, PyRETIS Development Team.
# Distributed under the LGPLv2.1+ License. See LICENSE for more info.
# adapted by An Ghysels, March 12, 2019
"""This is a 1D example potential"""
import logging
import numpy as np
logger = logging.getLogger(__name__)  # pylint: disable=C0103
logger.addHandler(logging.NullHandler())


class FlatWall1D():
    # TODO description
    r"""FlatWall(PotentialFunction).

    # TODO   rewrite all this documentation !!!

    This class defines a 2D dimensional potential with two
    stable states.
    The potential energy (:math:`V_\text{pot}`) is given by

    .. math::

       V_\text{pot}(x, y) = \gamma_1 (x^2 + y^2)^2 +
       \gamma_2 \exp(\alpha_1 (x - x_0)^2 + \alpha_2 (y - y_0)^2) +
       \gamma_3 \exp(\beta_1 (x + x_0)^2 + \beta_2(y + y_0)^2)

    where :math:`x` and :math:`y` gives the positions and :math:`\gamma_1`,
    :math:`\gamma_2`, :math:`\gamma_3`, :math:`\alpha_1`, :math:`\alpha_2`,
    :math:`\beta_1`, :math:`\beta_2`, :math:`x_0` and :math:`y_0` are
    potential parameters.

    Attributes
    ----------
    params : dict
        Contains the parameters. The keys are:

        * `gamma1`: The :math:`\gamma_1` parameter for the potential.
        * `gamma2`: The :math:`\gamma_2` parameter for the potential.
        * `gamma3`: The :math:`\gamma_3` parameter for the potential.
        * `alpha1`: The :math:`\alpha_1` parameter for the potential.
        * `alpha2`: The :math:`\alpha_2` parameter for the potential.
        * `beta1`: The :math:`\beta_1` parameter for the potential.
        * `beta2`: The :math:`\beta_2` parameter for the potential.
        * `x0`: The :math:`x_0` parameter for the potential.
        * `y0`: The :math:`y_0` parameter for the potential.
    """

    def __init__(self, desc='1D flat plus 2 walls'):
        """Initiate the potential.

        Parameters
        ----------
        a : float, optional
            Parameter for the potential.
        b : float, optional
            Parameter for the potential.
        c : float, optional
            Parameter for the potential.
        desc : string, optional
            Description of the force field.
        """
        # these are the default parameters
        self.params = {'xleft': -0.2, 'xright': 1., 'k': 100.}

    def potential(self, ph):
        """Evaluate the potential.

        Parameters
        ----------
        system : object like `System`
            The system we evaluate the potential for. Here, we
            make use of the positions only.

        Returns
        -------
        out : float
            The potential energy.
        """
        x = ph[0]  # this is array, assume 1D (x)
        xleft  = self.params['xleft']
        xright = self.params['xright']
        k = self.params['k']
        if x < xleft:
            v_pot = k*(x-xleft)**2/2.
        elif x > xright:
            v_pot = k*(x-xright)**2/2.
        else: v_pot = np.zeros_like(x)
        return v_pot.sum()

    def force(self, ph):
        """Evaluate forces.

        Parameters
        ----------
        system : object like `System`
            The system we evaluate the potential for. Here, we
            make use of the positions only.

        Returns
        -------
        out[0] : numpy.array
            The calculated force.
        out[1] : numpy.array
            The virial, currently not implemented for this potential
        """
        x = ph[0]
        xleft  = self.params['xleft']
        xright = self.params['xright']
        k = self.params['k']

        if x < xleft:
            force = -k*(x-xleft)
        elif x > xright:
            force = -k*(x-xright)
        else:
            force = 0.

        return force

    def potential_and_force(self, ph):
        """Evaluate the potential and the force.

        Parameters
        ----------
        system : object like `System`
            The system we evaluate the potential for. Here, we
            make use of the positions only.

        Returns
        -------
        out[0] : float
            The potential energy as a float.
        out[1] : numpy.array
            The force as a numpy.array of the same shape as the
            positions in `particles.pos`.
        out[2] : numpy.array
            The virial, currently not implemented for this potential.
        """
        v_pot = self.potential(ph)
        force = self.force(ph)
         
        return v_pot, force
