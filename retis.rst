TIS 1D example
==============

Simulation
----------
task = retis
steps = 30000
interfaces = [-1, -0.75, -0.5, -0.25, 0.]
#zero_left = -0.2
permeability = False

System
------
units = reduced
dimensions = 1
temperature = 1.

Box
---
periodic = [True]
low = [-1000.]
high = [1000.]

Engine
------
class = Langevin
timestep = 0.01
gamma = 10.
high_friction = False
seed = 0

TIS
---
freq =  0.
maxlength = 200000  # 200000
aimless = True
allowmaxlength = False
zero_momentum = False
rescale_energy = False
sigma_v = -1
seed = 0


RETIS settings
--------------
swapfreq = 0.1
relative_shoots = None
nullmoves = True
swapsimul = True

Initial-path
------------
method = kick
kick-from = previous

Particles
---------
position = {'input_file': 'initial.xyz'}
velocity = {'generate': 'maxwell',
            'momentum': False,
            'seed': 0}
mass = {'Ar': 1.0}
name = ['Ar']
type = [0]

Forcefield
----------
description = 1D flat with walls, no force

Orderparameter
--------------
class = Position
dim = x
index = 0
periodic = True

Potential
---------
class = FlatWall1D
module = flat-potential-walls.py
parameter k = 100
parameter xleft = -0.2
parameter xright = 0.4

Output
------
screen = 1000
energy-file = 1000
order-file = 1

Analysis
--------
tau_ref_bin = [-0.12,-0.1]

