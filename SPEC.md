# AutoDock-Vina AD sidecar specification (full scoring surface)

This scope was committed before implementation in commit `0923ce7`; the
implementation and tests follow in the round-2 fix commit. The reviewed
implementation was `f00535b` (review round 2).

## Upstream and boundary

- Official source: `https://github.com/ccsb-scripps/AutoDock-Vina.git`, commit
  `3c65c0b3e6c2c1d183f6a175ecb65e3c5ba91645`, copied verbatim under
  `upstream/` by import commit `ac64216`.
- The snapshot's Python binding imports the compiled `vina.vina_wrapper`,
  which is absent from the snapshot. Full `vina.Vina` docking is therefore
  deferred. A labelled toy diagnostic is never used as workflow evidence.
- The sidecar implements the source-level potential/recombination surface for
  official `SF_VINA`, `SF_VINARDO`, and `SF_AD42`. Coordinate calls remain a
  fixed pair-list replay (grid/map values can instead be passed as precomputed
  potential sums). Vina/Vinardo require X-Score atom types; AD4 requires AD4
  atom types and accepts PDBQT charges (default zero). Coordinates, weights,
  and precomputed term vectors have registered derivative rules; atom types,
  charges, pair topology, and torsion count are fixed state.

## Scoring-family callables

`vina_ad.score_coordinates(coordinates, atom_types, *, pairs=None,
weights=None, torsion_count=0.0, sf_name="vina", charges=None)` returns pair
energy in kcal/mol. `pairs=None` means every `i < j`; otherwise every `(i, j)`
is a fixed pair. The family selects the source potential set and default
weight vector. `potential_terms` exposes unweighted potential sums;
`recombine_terms` applies the complete scorer to six Vina or five Vinardo/AD4
precomputed terms. `score_terms` exposes one weighted contribution per public
weight and sums exactly to `score_coordinates`.

Vina weights are `(gaussian1, gaussian2, repulsion, hydrophobic, hydrogen,
glue, rot)`; Vinardo weights are `(gaussian, repulsion, hydrophobic, hydrogen,
glue, rot)`; AD4 weights are `(vdw, hydrogen, electrostatic, desolvation,
glue, rot)`. AD4 uses integer `AD_TYPE_*` values 0 through 30 and per-atom
charges.

For Vina, every pair with `r = ||x_i-x_j|| < 20`, let
`d = r - optimal_distance(xs_i,xs_j)` and
`g(o,w)=exp(-((d-o)/w)**2)`. The six terms are exactly the upstream classes:

1. `vina_gaussian(0, 0.5, 8)` -> `g(0, 0.5)`;
2. `vina_gaussian(3, 2, 8)` -> `g(3, 2)`;
3. `vina_repulsion(0, 8)` -> `d*d` when `d <= 0`, else zero;
4. `vina_hydrophobic(0.5, 1.5, 8)` -> `slope_step(1.5, 0.5, d)` for
   hydrophobic X-Score type pairs;
5. `vina_non_dir_h_bond(-0.7, 0, 8)` -> `slope_step(0, -0.7, d)` for donor /
   acceptor pairs;
6. `linearattraction(20)` -> `r` only for matching macrocycle glue pairs.

Vinardo uses five terms: `vinardo_gaussian(0, 0.8, 8)`,
`vinardo_repulsion(0, 8)`, `vinardo_hydrophobic(0, 2.5, 8)`,
`vinardo_non_dir_h_bond(-0.6, 0, 8)`, and the same glue attraction. Its
optimal distances use `xs_vinardo_vdw_radii`.

AD4 uses `ad4_vdw(0.5, 100000, 8)`, `ad4_hb(0.5, 100000, 8)`,
`ad4_electrostatic(100, 20.48)`, `ad4_solvation(3.6, 0.01097, true, 20.48)`,
and glue attraction. AD4 van der Waals and H-bond terms use the tabulated
AD4 radii/depths and the electrostatic/desolvation terms use atom charges.

The potential weights multiply their terms. Vina/Vinardo then apply
`E/(1 + weight_rot*torsion_count)`; AD4 adds `weight_rot*torsion_count`. The
mapping is sourced to `upstream/src/lib/scoring_function.h:48-85`,
`upstream/src/lib/potentials.h:134-514`, and
`upstream/src/lib/conf_independent.cpp:146-179`.

JVP/VJP/gradient rules are real-linear in coordinates and weights. Coincident
radii, the 8/20 A cutoffs, and piecewise knots are reported as
`NonDifferentiablePoint` only when coordinates are an active input. A
weights-only JVP/VJP computes term-value derivatives without coordinate
singularity checks. Invalid values and unsupported active inputs fail with
contextual errors. Rules call the public primal and share one analytic pair
linearisation, pruned by the requested active inputs.

## Oracle and workflow

Tests compare independent source-formula transcriptions to `ScoringFunction`
values (absolute error <= `1e-12`) and run a real installed `vina.Vina`
Python binding on a sourced one-atom receptor/ligand PDBQT pair. The binding
rounds to three decimals and interpolates maps; that one-atom restricted pair
oracle agrees within `0.05` kcal/mol. Every exposed AD entry point has an
executed finite-difference or analytic oracle: JVP and VJP are checked in both
coordinate and weight directions, their real inner-product duality is asserted,
and `grad`/`value_and_grad` are compared for both active inputs. Zero JVP
directions and zero VJP cotangents are also checked. The official
`upstream/example/python_scripting/first_example.py` source files are run with
the installed real binding, and `run_official_workflow` invokes public
`value_and_grad` and `jvp` on all sourced receptor/ligand cross pairs. It
reports the real Vina score, restricted primal, gradient norms, directional
finite-difference error, duality error, pair count, and remaining full-grid /
search coverage; the two primal values are intentionally not treated as the
same model for multi-atom inputs.

## Complete API inventory and decisions

All 19 public `vina.Vina` methods are recorded in `api_inventory.json`. Every
entry has primal semantics, differentiable and fixed inputs, derivative
interface, mathematical convention, reusable computation/dependency chain,
oracle, and explicit decision evidence. Constructor/configuration, file I/O,
map generation, pose selection, optimization, and stochastic docking remain
deferred or not suitable for AD.

## Acceptance thresholds and remaining scope

- Source-term oracle: absolute error <= `1e-12`.
- Real binding smoke oracle: absolute deviation <= `0.05` kcal/mol.
- Analytic derivatives: finite-difference error <= `2e-5` away from knots;
  JVP/VJP duality <= `1e-10`.
- Fresh no-dependency installs must preserve ChainRules 0.1.0 tangent/wrt
  validation and contextual `UnsupportedWrt` attributes.

The complete C++ grid/search engine, grid interpolation derivatives, topology
construction, and all deferred/not-suitable methods are outside this scope.
