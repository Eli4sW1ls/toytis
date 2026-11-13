"""Run toytis i* in more than two dimensions, without changing the sampler.

Two things in toytis assume dim <= 2:

* order.OrderX.calculate returns [pos[0], pos[1]], so order.txt would lose every coordinate after x.
  OrderXN returns all positions [y, x, w, ...] (numpy float64, as OrderX); op[0] = y stays the order parameter
  of the interfaces, turns and states. len(orders) = dim, so v_ord works too (the Z runs used v_ord = False).
* pgmoves.kick_star (i* ensembles) and moves.kick_retis (state-A ensembles, e.g. [0-] of an i* run; RETIS,
  REPPTIS) build the kick point of an initial path only for dim 1 and 2 (for dim 3 their loop never sets it).
  kick_star_nd and kick_retis_nd are the original functions with two edits, made on their source text and
  checked: the dim-2 branch is taken for every dim >= 2, and the kick point (y, x) is then extended by
  potential.kick_extra(lambda) (the coordinates after x). Everything else is the original code.

    import nd_patches
    nd_patches.install()          # before Simulation(settings); also before a restart (restart.pkl refers to
                                  # nd_patches.OrderXN)

Composes with optimalrcs examples/param_lambda/toytis_adapter.install (a learned lambda): call this first; that
adapter then wraps kick_star_nd and replaces OrderXN.
"""
import inspect
import textwrap

import numpy as np

import ensemble
import moves
import pgmoves


class OrderXN:
    """All positions of the phase point: [y, x, w, ...] (op[0] = y is the order parameter)."""

    def __init__(self, params=None):
        self.params = params

    def calculate(self, ph):
        pos = ph[0]
        return [pos[i] for i in range(len(pos))]


def _extend_kick(ens, sh):
    """(y, x) kick point -> (y, x, extra...) with the potential's kick_extra (dim > 2)."""
    dim = ens.settings['dim']
    pos = np.asarray(sh[0], dtype=float)
    if dim <= 2 or len(pos) == dim:
        return sh
    lam = ens.intfs['R'] if ens.id == 0 else ens.intfs['L'] if ens.id == 1 else ens.intfs['M']
    extra = np.asarray(ens.engine.potential.kick_extra(lam), dtype=float)
    pos = np.concatenate([pos, extra])
    if len(pos) != dim:
        raise ValueError(f'kick point has {len(pos)} coordinates, dim = {dim}: check potential.kick_extra')
    return (pos, sh[1])


def _patched(module, name):
    """module.<name> (a kick function) with the dim-2 branch taken for every dim >= 2 and the kick point extended
    by _extend_kick; built from its source text, which must contain each edited line exactly once."""
    src = textwrap.dedent(inspect.getsource(getattr(module, name)))
    edits = [
        ('elif ens.settings["dim"] == 2:', 'elif ens.settings["dim"] >= 2:'),
        ('        shootpoint_op = ens.orderparameter.calculate(sh)\n',
         '        sh = _extend_kick(ens, sh)\n        shootpoint_op = ens.orderparameter.calculate(sh)\n'),
    ]
    for old, new in edits:
        if src.count(old) != 1:
            raise RuntimeError(f'nd_patches: {module.__name__}.{name} changed, cannot find exactly one {old!r}')
        src = src.replace(old, new)
    src = src.replace(f'def {name}(ens):', f'def {name}_nd(ens):', 1)
    namespace = dict(module.__dict__)          # propagate, Path, logger, np, ... as in the module
    namespace['_extend_kick'] = _extend_kick
    exec(compile(src, f'<nd_patches: {module.__name__}.{name} for dim >= 2>', 'exec'), namespace)
    fn = namespace[f'{name}_nd']
    fn.__doc__ = f'{module.__name__}.{name} for any dim >= 2 (see nd_patches).'
    return fn


kick_star_nd = _patched(pgmoves, 'kick_star')      # i* body ensembles and [0*]
kick_retis_nd = _patched(moves, 'kick_retis')      # state-A ensembles ([0-] of an i* run), RETIS, REPPTIS


def install():
    """Patch ensemble.OrderX, ensemble.kick_star and ensemble.kick_retis in this process."""
    ensemble.OrderX = OrderXN
    ensemble.kick_star = kick_star_nd
    ensemble.kick_retis = kick_retis_nd
