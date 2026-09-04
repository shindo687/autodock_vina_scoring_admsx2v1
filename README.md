# vina-ad

`vina-ad` is a separately installable sidecar for AutoDock Vina. It exposes an
explicit ChainRules-compatible API for all three upstream scoring families:
`vina`, `vinardo`, and `ad4`. The coordinate entry point is
`vina_ad.score_coordinates`; `score` and `energy` remain aliases. A small
`ScoringFunction` facade mirrors the upstream family/`set_weights` contract.

```python
import vina_ad
coordinates = [[0., 0., 0.], [3., 0., 0.], [0., 4., 0.]]
atom_types = [0, 0, 0]  # XS_TYPE_C_H values from upstream atom_constants.h
value, tangent = vina_ad.jvp(
    vina_ad.score_coordinates, coordinates,
    atom_types,
    tangents={"coordinates": [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]},
)
value, gradients = vina_ad.value_and_grad(
    vina_ad.score_coordinates, coordinates, atom_types, wrt="coordinates"
)
```

Select another family explicitly (AD4 atom types are used in the second
example):

```python
vina_ad.score_coordinates(coordinates[:2], [0, 0], sf_name="vinardo")
vina_ad.score_coordinates(
    coordinates[:2], [0, 3], sf_name="ad4", charges=[0.2, -0.3]
)
```

`vina_ad.potential_terms` returns the unweighted source potential sums. For
the six/five potentials, `vina_ad.recombine_terms` performs the complete
upstream weighted accumulation and torsion correction on caller-precomputed
interactions. `vina_ad.score_terms` returns weighted contributions (including
the torsion contribution) whose ordinary sum is exactly the composed score;
`term_values` returns raw feature terms plus the torsion count. AD4 takes
AutoDock atom types and optional per-atom charges (omitted charges mean zero).

The formulas map Vina's and Vinardo's Gaussians, repulsion, hydrophobic,
hydrogen-bond, macrocycle glue, and torsion correction, plus AD4's capped
van der Waals, hydrogen-bond, electrostatic, desolvation, glue, and torsion
terms from the immutable upstream source. Atom types, fixed interacting pairs,
charges, and torsion count are state inputs; coordinates, weights, and the
precomputed term vector have registered JVP/VJP paths. The sidecar remains
usable without the compiled upstream `vina_wrapper`; maps/grid interpolation,
pose optimization, and stochastic docking are intentionally outside this
contract. See `SPEC.md`, `api_inventory.json`, and `vina_ad/requirements.md`.
`python -m vina_ad.workflow` runs the sourced workflow, including public
`value_and_grad`/`jvp` calls and quantitative primal/derivative metrics, when a
real binding and PDBQT files are available.
In an installed environment pass the source inputs explicitly:

```bash
python -m vina_ad.workflow \
  --receptor /path/to/1iep_receptor.pdbqt \
  --ligand /path/to/1iep_ligand.pdbqt
```

If the real binding or inputs are unavailable it reports the capability as
deferred; the labelled `run_demo` is only a toy diagnostic.
