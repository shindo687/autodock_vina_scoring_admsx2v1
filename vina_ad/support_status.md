# Support status (vina-ad 0.2.0)

| API family | Status | Evidence / reason |
| --- | --- | --- |
| `score_coordinates` (`score`, `energy` aliases) | implemented | Explicit `sf_name="vina"`, `"vinardo"`, or `"ad4"`; coordinate pair replay and source-level family formulas in `core.py`; tests cover all families and independent finite differences |
| `potential_terms` / `term_values` | implemented | Public unweighted potential decomposition; AD4 has AD types, charge input, capped electrostatic/desolvation/H-bond terms; coordinate JVP/VJP |
| `score_terms` / `weighted_terms` | implemented | Weighted contribution vector, one element per public weight, including the torsion contribution; its sum equals the composed score; coordinate/weight JVP/VJP |
| `recombine_terms` | implemented | Complete upstream potential-weight accumulation over six Vina or five Vinardo/AD4 precomputed interaction sums, with torsion division/addition; term/weight JVP/VJP |
| `ScoringFunction` | implemented facade | Explicit family selection, family defaults, `set_weights`, `get_weights`, `score`, and `terms`; pure functions remain the AD boundary |
| JVP/VJP/grad/value_and_grad | implemented | Coordinate and weight paths for scalar score and weighted term vectors; term-vector path for precomputed recombination; finite-difference, duality, zero, and cutoff tests |
| `vina.Vina.score` maps/grid layer | deferred by scope | Grid interpolation and receptor map generation are not inferred from a coordinate pair list; callers can pass their precomputed potential sums to `recombine_terms` |
| `vina.Vina.optimize`, `dock`, `randomize` | not suitable/deferred | Iterative, stochastic, or discrete search control flow has no stable derivative contract |

The pinned upstream formulas are in `upstream/src/lib/scoring_function.h`,
`potentials.h`, `atom_constants.h`, and `conf_independent.cpp`. For Vina and
Vinardo, upstream first forms `E = sum(weight_i * potential_i)` and returns
`E / (1 + weight_rot * torsion_count)`. For AD4 it returns
`sum(weight_i * potential_i) + weight_rot * torsion_count`. The final element
of `score_terms` is the corresponding correction/addition so that the vector
is an exact weighted decomposition. `term_values` is the raw feature vector;
its final element is the torsion count.

Unsupported active inputs are rejected with contextual `UnsupportedWrt`; the
dependency-free fallback preserves the same ChainRules 0.1.0 protocol. Pair
cutoffs, slope knots, caps, and coincident coordinates raise
`NonDifferentiablePoint` only for coordinate-active rules. File I/O, maps,
pose selection, and search remain outside this sidecar.
