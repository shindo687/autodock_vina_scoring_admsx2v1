"""Differentiable, source-level AutoDock-Vina scoring families.

The upstream ``ScoringFunction`` first evaluates a set of pair/grid
interactions, then recombines the resulting potential sums with its public
weights.  This module makes both halves explicit.  ``potential_terms`` and
``score_coordinates`` are a coordinate pair replay (maps remain an explicit
caller-supplied boundary), while ``recombine_terms`` is the complete
weight/torsion recombination over any precomputed interaction vector.

The formulas and constants are transcribed from the pinned snapshot in
``upstream/src/lib/{scoring_function,potentials,atom_constants,conf_independent}.``
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from .protocol import NonDifferentiablePoint, UnsupportedWrt, ZERO, rules


DEFAULT_VINA_WEIGHTS = (-0.035579, -0.005156, 0.840245, -0.035069, -0.587439, 50.0, 0.05846)
DEFAULT_VINARDO_WEIGHTS = (-0.045, 0.8, -0.035, -0.600, 50.0, 0.05846)
DEFAULT_AD4_WEIGHTS = (0.1662, 0.1209, 0.1406, 0.1322, 50.0, 0.2983)

FAMILY_DEFAULT_WEIGHTS = {"vina": DEFAULT_VINA_WEIGHTS, "vinardo": DEFAULT_VINARDO_WEIGHTS, "ad4": DEFAULT_AD4_WEIGHTS}
DEFAULT_WEIGHTS = FAMILY_DEFAULT_WEIGHTS
FAMILY_TERM_NAMES = {
    "vina": ("gaussian1", "gaussian2", "repulsion", "hydrophobic", "hydrogen_bond", "glue", "torsion"),
    "vinardo": ("gaussian", "repulsion", "hydrophobic", "hydrogen_bond", "glue", "torsion"),
    "ad4": ("vdw", "hydrogen_bond", "electrostatic", "desolvation", "glue", "torsion"),
}
FAMILY_POTENTIAL_NAMES = {name: values[:-1] for name, values in FAMILY_TERM_NAMES.items()}

_XS_RADII = (1.9, 1.9, 1.8, 1.8, 1.8, 1.8, 1.7, 1.7, 1.7, 1.7, 2.0, 2.1, 1.5, 1.8, 2.0, 2.2, 2.2, 2.3, 1.2, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9, 0.0, 0.0)
_VINARDO_RADII = (2.0, 2.0, 1.7, 1.7, 1.7, 1.7, 1.6, 1.6, 1.6, 1.6, 2.0, 2.1, 1.5, 1.8, 2.0, 2.2, 2.2, 2.3, 1.2, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0)
_XS_TYPE_SIZE = 32
_HYDROPHOBIC_TYPES = frozenset((0, 12, 13, 14, 15))
_DONOR_TYPES = frozenset((3, 5, 7, 9, 18))
_ACCEPTOR_TYPES = frozenset((4, 5, 8, 9))

# AD_TYPE_* table (radius, Lennard-Jones depth, HB depth, HB radius,
# solvation parameter, volume) from atom_constants.h.
_AD4_DATA = (
    (2.00000, 0.15000, 0.0, 0.0, -0.00143, 33.51030), (2.00000, 0.15000, 0.0, 0.0, -0.00052, 33.51030),
    (1.75000, 0.16000, 0.0, 0.0, -0.00162, 22.44930), (1.60000, 0.20000, 0.0, 0.0, -0.00251, 17.15730),
    (2.10000, 0.20000, 0.0, 0.0, -0.00110, 38.79240), (2.00000, 0.20000, 0.0, 0.0, -0.00214, 33.51030),
    (1.00000, 0.02000, 0.0, 0.0, 0.00051, 0.00000), (1.54500, 0.08000, 0.0, 0.0, -0.00110, 15.44800),
    (2.36000, 0.55000, 0.0, 0.0, -0.00110, 55.05850), (1.75000, 0.16000, -5.0, 1.9, -0.00162, 22.44930),
    (1.60000, 0.20000, -5.0, 1.9, -0.00251, 17.15730), (2.00000, 0.20000, -1.0, 2.5, -0.00214, 33.51030),
    (1.00000, 0.02000, 1.0, 0.0, 0.00051, 0.00000), (0.65000, 0.87500, 0.0, 0.0, -0.00110, 1.56000),
    (0.65000, 0.87500, 0.0, 0.0, -0.00110, 2.14000), (0.74000, 0.55000, 0.0, 0.0, -0.00110, 1.70000),
    (0.99000, 0.55000, 0.0, 0.0, -0.00110, 2.77000), (0.65000, 0.01000, 0.0, 0.0, -0.00110, 1.84000),
    (2.04500, 0.27600, 0.0, 0.0, -0.00110, 35.82350), (2.16500, 0.38900, 0.0, 0.0, -0.00110, 42.56610),
    (2.30000, 0.20000, 0.0, 0.0, -0.00143, 50.96500), (2.40000, 0.55000, 0.0, 0.0, -0.00110, 57.90580),
    (0.00000, 0.00000, 0.0, 0.0, 0.00000, 0.00000), (0.00000, 0.00000, 0.0, 0.0, 0.00000, 0.00000),
    (0.00000, 0.00000, 0.0, 0.0, 0.00000, 0.00000), (0.00000, 0.00000, 0.0, 0.0, 0.00000, 0.00000),
    (2.00000, 0.15000, 0.0, 0.0, -0.00143, 33.51030), (2.00000, 0.15000, 0.0, 0.0, -0.00143, 33.51030),
    (2.00000, 0.15000, 0.0, 0.0, -0.00143, 33.51030), (2.00000, 0.15000, 0.0, 0.0, -0.00143, 33.51030),
    (0.00000, 0.00000, 0.0, 0.0, 0.00000, 0.00000),
)
_AD4_TYPE_SIZE = len(_AD4_DATA)
_AD4_GLUE_PARTNERS = {22: 26, 23: 27, 24: 28, 25: 29}


def _family(sf_name: Any) -> str:
    if not isinstance(sf_name, str):
        raise TypeError("sf_name must be one of 'vina', 'vinardo', or 'ad4'")
    value = sf_name.lower()
    if value not in FAMILY_DEFAULT_WEIGHTS:
        raise ValueError(f"sf_name must be one of 'vina', 'vinardo', or 'ad4'; got {sf_name!r}")
    return value


def family_term_names(sf_name: str = "vina") -> tuple[str, ...]:
    """Return ordered public term names for a scoring family."""
    return FAMILY_TERM_NAMES[_family(sf_name)]


def _rows(coordinates: Any, *, tangent: bool = False) -> tuple[list[list[float]], str]:
    if isinstance(coordinates, (str, bytes)) or not isinstance(coordinates, Sequence):
        try:
            coordinates = coordinates.tolist()
        except AttributeError as exc:
            raise TypeError("coordinates must be a numeric (N, 3) sequence") from exc
    try:
        raw = list(coordinates)
    except TypeError as exc:
        raise TypeError("coordinates must be a numeric (N, 3) sequence") from exc
    if len(raw) < 2 and not tangent:
        raise ValueError("coordinates must contain at least two atoms")
    out: list[list[float]] = []
    for row in raw:
        if isinstance(row, (str, bytes)):
            raise TypeError("each coordinate must have exactly three real values")
        try:
            values = list(row)
        except TypeError as exc:
            raise TypeError("each coordinate must have exactly three real values") from exc
        if len(values) != 3:
            raise ValueError("coordinates must have shape (N, 3)")
        converted = []
        for value in values:
            if isinstance(value, bool):
                raise TypeError("coordinates must contain real numbers")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError("coordinates must contain real numbers") from exc
            if not math.isfinite(number):
                raise ValueError("coordinates must be finite")
            converted.append(number)
        out.append(converted)
    return out, ("tuple" if isinstance(coordinates, tuple) else "list")


def _integer_types(values: Any, n_atoms: int, *, family: str) -> tuple[int, ...]:
    label = "AD4" if family == "ad4" else "X-Score"
    limit = _AD4_TYPE_SIZE if family == "ad4" else _XS_TYPE_SIZE
    if values is None:
        raise TypeError(f"atom_types is required: pass one {label} atom type per atom")
    if isinstance(values, (str, bytes)):
        raise TypeError(f"atom_types must be a length-N integer sequence of {label} values")
    try:
        raw = list(values)
    except TypeError as exc:
        raise TypeError(f"atom_types must be a length-N integer sequence of {label} values") from exc
    if len(raw) != n_atoms:
        raise ValueError("atom_types must have one atom type per atom")
    result = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("atom_types must contain integer atom type values")
        value = int(value)
        if value < 0 or value >= limit:
            raise ValueError(f"atom type must be in [0, {limit}) for {family}")
        result.append(value)
    return tuple(result)


def _weights(weights: Any, family: str) -> tuple[float, ...]:
    expected = FAMILY_DEFAULT_WEIGHTS[family]
    if weights is None:
        return expected
    if isinstance(weights, (str, bytes)):
        raise TypeError(f"weights must be a length-{len(expected)} real sequence for {family}")
    try:
        raw = list(weights)
    except TypeError as exc:
        raise TypeError(f"weights must be a length-{len(expected)} real sequence for {family}") from exc
    if len(raw) != len(expected):
        raise ValueError(f"weights must have length {len(expected)} for {family}")
    result = []
    for value in raw:
        if isinstance(value, bool):
            raise TypeError("weights must contain real numbers")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("weights must contain real numbers") from exc
        if not math.isfinite(number):
            raise ValueError("weights must be finite")
        result.append(number)
    return tuple(result)


def _pairs(pairs: Any, n_atoms: int) -> tuple[tuple[int, int], ...]:
    if pairs is None:
        return tuple((i, j) for i in range(n_atoms - 1) for j in range(i + 1, n_atoms))
    if isinstance(pairs, (str, bytes)):
        raise TypeError("pairs must be a sequence of (i, j) index pairs")
    try:
        raw_pairs = list(pairs)
    except TypeError as exc:
        raise TypeError("pairs must be a sequence of (i, j) index pairs") from exc
    result = []
    for pair in raw_pairs:
        try:
            values = list(pair)
        except TypeError as exc:
            raise TypeError("each pair must contain two atom indices") from exc
        if len(values) != 2 or any(isinstance(v, bool) or not isinstance(v, Integral) for v in values):
            raise TypeError("each pair must contain two atom indices")
        i, j = (int(values[0]), int(values[1]))
        if i < 0 or j < 0 or i >= n_atoms or j >= n_atoms or i == j:
            raise ValueError("pair indices must be distinct and in range")
        result.append((i, j))
    return tuple(result)


def _torsion_count(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("torsion_count must be a finite non-negative real")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("torsion_count must be a finite non-negative real") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError("torsion_count must be a finite non-negative real")
    return result


def _charges(charges: Any, n_atoms: int, family: str) -> tuple[float, ...]:
    if family != "ad4":
        if charges is not None:
            raise ValueError("charges are only accepted for sf_name='ad4'")
        return (0.0,) * n_atoms
    # PDBQT charges default to zero in the upstream atom representation.  An
    # explicit vector is recommended for electrostatic/desolvation parity, but
    # omitting it remains useful for neutral AD4 probes and keeps family
    # selection orthogonal to charge data.
    if charges is None:
        return (0.0,) * n_atoms
    if isinstance(charges, (str, bytes)):
        raise TypeError("charges must be a length-N real sequence")
    try:
        raw = list(charges)
    except TypeError as exc:
        raise TypeError("charges must be a length-N real sequence") from exc
    if len(raw) != n_atoms:
        raise ValueError("charges must have one value per atom")
    result = []
    for value in raw:
        if isinstance(value, bool):
            raise TypeError("charges must contain real numbers")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("charges must contain real numbers") from exc
        if not math.isfinite(number):
            raise ValueError("charges must be finite")
        result.append(number)
    return tuple(result)


def _is_glued_xs(type_i: int, type_j: int) -> bool:
    partners = {21: (19, 20), 24: (22, 23), 27: (25, 26), 30: (28, 29)}
    return type_j in partners.get(type_i, ()) or type_i in partners.get(type_j, ())


def _is_glued_ad(type_i: int, type_j: int) -> bool:
    return _AD4_GLUE_PARTNERS.get(type_i) == type_j or _AD4_GLUE_PARTNERS.get(type_j) == type_i


def _slope_step(bad: float, good: float, x: float) -> tuple[float, float]:
    if bad < good:
        if x <= bad or x >= good:
            return (0.0 if x <= bad else 1.0), 0.0
    else:
        if x >= bad or x <= good:
            return (0.0 if x >= bad else 1.0), 0.0
    return (x - bad) / (good - bad), 1.0 / (good - bad)


def _pair_terms_vina(type_i: int, type_j: int, radius: float, *, vinardo: bool = False, differentiate: bool = False) -> tuple[tuple[float, ...], tuple[float, ...]]:
    radii = _VINARDO_RADII if vinardo else _XS_RADII
    optimal = 0.0 if type_i in (21, 24, 27, 30) or type_j in (21, 24, 27, 30) else radii[type_i] + radii[type_j]
    delta = radius - optimal
    if differentiate and math.isclose(radius, 8.0, rel_tol=0.0, abs_tol=1e-14):
        raise NonDifferentiablePoint("8 A potential cutoff is non-differentiable")
    if radius < 8.0:
        if differentiate and abs(delta) <= 1e-14:
            raise NonDifferentiablePoint("repulsion knot is non-differentiable")
        if differentiate and type_i in _HYDROPHOBIC_TYPES and type_j in _HYDROPHOBIC_TYPES:
            knot = (0.0, 2.5) if vinardo else (0.5, 1.5)
            if min(abs(delta - knot[0]), abs(delta - knot[1])) <= 1e-14:
                raise NonDifferentiablePoint("hydrophobic slope-step knot is non-differentiable")
        donor_acceptor = (type_i in _DONOR_TYPES and type_j in _ACCEPTOR_TYPES) or (type_j in _DONOR_TYPES and type_i in _ACCEPTOR_TYPES)
        if differentiate and donor_acceptor:
            knot = (0.0, -0.6) if vinardo else (0.0, -0.7)
            if min(abs(delta - knot[0]), abs(delta - knot[1])) <= 1e-14:
                raise NonDifferentiablePoint("hydrogen-bond slope-step knot is non-differentiable")
    glued = _is_glued_xs(type_i, type_j)
    if differentiate and math.isclose(radius, 20.0, rel_tol=0.0, abs_tol=1e-14) and glued:
        raise NonDifferentiablePoint("20 A macrocycle glue cutoff is non-differentiable")
    n = 5 if vinardo else 6
    if radius >= 20.0:
        return (0.0,) * n, (0.0,) * n
    values, derivatives = [0.0] * n, [0.0] * n
    if radius < 8.0:
        if vinardo:
            value = math.exp(-((delta / 0.8) ** 2))
            values[0] = value
            if differentiate:
                derivatives[0] = -2.0 * delta * value / (0.8 * 0.8)
            if delta <= 0:
                values[1] = delta * delta
                if differentiate:
                    derivatives[1] = 2.0 * delta
            if type_i in _HYDROPHOBIC_TYPES and type_j in _HYDROPHOBIC_TYPES:
                values[2], derivatives[2] = _slope_step(2.5, 0.0, delta)
            if ((type_i in _DONOR_TYPES and type_j in _ACCEPTOR_TYPES) or (type_j in _DONOR_TYPES and type_i in _ACCEPTOR_TYPES)):
                values[3], derivatives[3] = _slope_step(0.0, -0.6, delta)
        else:
            for index, (offset, width) in enumerate(((0.0, 0.5), (3.0, 2.0))):
                displacement = delta - offset
                value = math.exp(-((displacement / width) ** 2))
                values[index] = value
                if differentiate:
                    derivatives[index] = -2.0 * displacement * value / (width * width)
            if delta <= 0:
                values[2] = delta * delta
                if differentiate:
                    derivatives[2] = 2.0 * delta
            if type_i in _HYDROPHOBIC_TYPES and type_j in _HYDROPHOBIC_TYPES:
                values[3], derivatives[3] = _slope_step(1.5, 0.5, delta)
            if ((type_i in _DONOR_TYPES and type_j in _ACCEPTOR_TYPES) or (type_j in _DONOR_TYPES and type_i in _ACCEPTOR_TYPES)):
                values[4], derivatives[4] = _slope_step(0.0, -0.7, delta)
    if glued:
        values[-1] = radius
        if differentiate:
            derivatives[-1] = 1.0
    return tuple(values), tuple(derivatives)


def _smoothen(radius: float, optimal: float, smoothing: float = 0.5) -> tuple[float, float]:
    half = smoothing * 0.5
    if radius > optimal + half:
        return radius - half, 1.0
    if radius < optimal - half:
        return radius + half, 1.0
    return optimal, 0.0


def _ad4_pair_terms(type_i: int, type_j: int, charge_i: float, charge_j: float, radius: float, *, differentiate: bool = False) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if differentiate and (math.isclose(radius, 8.0, abs_tol=1e-14, rel_tol=0.0) or (_is_glued_ad(type_i, type_j) and math.isclose(radius, 20.0, abs_tol=1e-14, rel_tol=0.0)) or math.isclose(radius, 20.48, abs_tol=1e-14, rel_tol=0.0)):
        raise NonDifferentiablePoint("AD4 potential cutoff is non-differentiable")
    if radius >= 20.48:
        return (0.0,) * 5, (0.0,) * 5
    ri, ei, hdi, hri, si, vi = _AD4_DATA[type_i]
    rj, ej, hdj, hrj, sj, vj = _AD4_DATA[type_j]
    values, derivatives = [0.0] * 5, [0.0] * 5
    hb_depth = hdi * hdj
    if radius < 8.0 and hb_depth >= 0:
        rij = ri + rj
        effective, smooth_derivative = _smoothen(radius, rij)
        depth = math.sqrt(ei * ej)
        c12, c6 = rij ** 12 * depth, rij ** 6 * depth * 2.0
        r6, r12 = effective ** 6, effective ** 12
        raw = c12 / r12 - c6 / r6 if r12 > 1e-12 and r6 > 1e-12 else 100000.0
        values[0] = min(100000.0, raw)
        if differentiate:
            if math.isclose(raw, 100000.0, abs_tol=1e-10, rel_tol=1e-12):
                raise NonDifferentiablePoint("AD4 van der Waals cap is non-differentiable")
            derivatives[0] = (-12.0 * c12 / effective ** 13 + 6.0 * c6 / effective ** 7) * smooth_derivative
    elif radius < 8.0:
        rij = hri + hrj
        effective, smooth_derivative = _smoothen(radius, rij)
        depth = -hb_depth
        c12, c10 = rij ** 12 * depth * 5.0, rij ** 10 * depth * 6.0
        r10, r12 = effective ** 10, effective ** 12
        raw = c12 / r12 - c10 / r10 if r12 > 1e-12 and r10 > 1e-12 else 100000.0
        values[1] = min(100000.0, raw)
        if differentiate:
            if math.isclose(raw, 100000.0, abs_tol=1e-10, rel_tol=1e-12):
                raise NonDifferentiablePoint("AD4 hydrogen-bond cap is non-differentiable")
            derivatives[1] = (-12.0 * c12 / effective ** 13 + 10.0 * c10 / effective ** 11) * smooth_derivative
    qprod = charge_i * charge_j * 332.0
    B, lB = 86.9525, -86.9525 * 0.003627
    exp_term = math.exp(lB * radius)
    diel = -8.5525 + B / (1.0 + 7.7839 * exp_term)
    diel_prime = -B * 7.7839 * lB * exp_term / (1.0 + 7.7839 * exp_term) ** 2
    if radius < 1e-6:
        values[2] = qprod * 100.0 / diel
        if differentiate:
            derivatives[2] = -qprod * 100.0 * diel_prime / (diel * diel)
    else:
        inv = 1.0 / (radius * diel)
        values[2] = qprod * min(100.0, inv)
        if differentiate:
            if math.isclose(inv, 100.0, abs_tol=1e-10, rel_tol=1e-12):
                raise NonDifferentiablePoint("AD4 electrostatic cap is non-differentiable")
            derivatives[2] = qprod * (-(diel + radius * diel_prime) / (radius * diel) ** 2) if inv < 100.0 else 0.0
    sigma, solvation_q = 3.6, 0.01097
    coefficient = (si + solvation_q * abs(charge_i)) * vj + (sj + solvation_q * abs(charge_j)) * vi
    values[3] = coefficient * math.exp(-0.5 * (radius / sigma) ** 2)
    if differentiate:
        derivatives[3] = -radius / (sigma * sigma) * values[3]
    if _is_glued_ad(type_i, type_j) and radius < 20.0:
        values[4] = radius
        if differentiate:
            derivatives[4] = 1.0
    return tuple(values), tuple(derivatives)


def _raw_terms_and_jacobian(rows: list[list[float]], types: tuple[int, ...], pairs: tuple[tuple[int, int], ...], family: str, charges: tuple[float, ...], *, active_coordinates: bool) -> tuple[tuple[float, ...], list[list[list[float]]]]:
    n_terms = len(FAMILY_POTENTIAL_NAMES[family])
    sums = [0.0] * n_terms
    jac = [[[0.0, 0.0, 0.0] for _ in rows] for _ in range(n_terms)]
    for i, j in pairs:
        delta_xyz = [rows[i][k] - rows[j][k] for k in range(3)]
        radius_squared = sum(component * component for component in delta_xyz)
        if radius_squared == 0.0 and active_coordinates:
            raise NonDifferentiablePoint("coincident atom coordinates have no finite derivative")
        radius = math.sqrt(radius_squared)
        if family == "ad4":
            terms, derivatives = _ad4_pair_terms(types[i], types[j], charges[i], charges[j], radius, differentiate=active_coordinates)
        else:
            terms, derivatives = _pair_terms_vina(types[i], types[j], radius, vinardo=family == "vinardo", differentiate=active_coordinates)
        for term_index, term in enumerate(terms):
            sums[term_index] += term
            if active_coordinates and derivatives[term_index] and radius_squared:
                scale = derivatives[term_index] / radius
                for k, component in enumerate(delta_xyz):
                    jac[term_index][i][k] += scale * component
                    jac[term_index][j][k] -= scale * component
    return tuple(sums), jac


def _recombine(raw: tuple[float, ...], weights: tuple[float, ...], torsions: float, family: str) -> tuple[float, tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    n = len(raw)
    if family in ("vina", "vinardo"):
        denominator = 1.0 + weights[-1] * torsions
        if denominator <= 0.0:
            raise ValueError("1 + weight_rot*torsion_count must be positive")
        energy = sum(weights[i] * raw[i] for i in range(n))
        score_value = energy / denominator
        contributions = tuple(weights[i] * raw[i] / denominator for i in range(n)) + (score_value - energy,)
        raw_gradient = tuple(weights[i] / denominator for i in range(n))
        weight_gradient = tuple(raw[i] / denominator for i in range(n)) + (-energy * torsions / (denominator * denominator),)
        return score_value, contributions, raw_gradient, weight_gradient
    score_value = sum(weights[i] * raw[i] for i in range(n)) + weights[-1] * torsions
    contributions = tuple(weights[i] * raw[i] for i in range(n)) + (weights[-1] * torsions,)
    return score_value, contributions, tuple(weights[:-1]), tuple(raw) + (torsions,)


def _coordinates_state(coordinates: Any, atom_types: Any, pairs: Any, weights: Any, torsion_count: Any, sf_name: Any, charges: Any):
    family = _family(sf_name)
    rows, _ = _rows(coordinates)
    types = _integer_types(atom_types, len(rows), family=family)
    pair_indices = _pairs(pairs, len(rows))
    coefficients = _weights(weights, family)
    torsions = _torsion_count(torsion_count)
    charge_values = _charges(charges, len(rows), family)
    return family, rows, types, pair_indices, coefficients, torsions, charge_values


def potential_terms(coordinates: Any, atom_types: Any = None, *, pairs: Any = None, sf_name: str = "vina", charges: Any = None) -> tuple[float, ...]:
    """Return unweighted potential sums for a fixed pair topology."""
    family, rows, types, pair_indices, _, _, charge_values = _coordinates_state(coordinates, atom_types, pairs, None, 0.0, sf_name, charges)
    return _raw_terms_and_jacobian(rows, types, pair_indices, family, charge_values, active_coordinates=False)[0]


@rules.jvp_for(potential_terms)
def _potential_terms_jvp(tangents: dict[str, Any], coordinates: Any, atom_types: Any = None, *, pairs: Any = None, sf_name: str = "vina", charges: Any = None) -> tuple[tuple[float, ...], Any]:
    unsupported = set(tangents) - {"coordinates"}
    if unsupported:
        raise UnsupportedWrt(potential_terms, unsupported, supported={"coordinates"})
    family, rows, types, pair_indices, _, _, charge_values = _coordinates_state(coordinates, atom_types, pairs, None, 0.0, sf_name, charges)
    value = potential_terms(coordinates, atom_types, pairs=pairs, sf_name=sf_name, charges=charges)
    tangent = tangents.get("coordinates", ZERO)
    if tangent is ZERO:
        return value, ZERO
    tangent_rows = _tangent_rows(tangent, len(rows))
    _, jac = _raw_terms_and_jacobian(rows, types, pair_indices, family, charge_values, active_coordinates=True)
    return value, _directional_raw(jac, tangent_rows)


@rules.vjp_for(potential_terms)
def _potential_terms_vjp(wrt: tuple[str, ...], coordinates: Any, atom_types: Any = None, *, pairs: Any = None, sf_name: str = "vina", charges: Any = None) -> tuple[tuple[float, ...], Any]:
    supported = {"coordinates"}
    unsupported = set(wrt) - supported
    if unsupported:
        raise UnsupportedWrt(potential_terms, unsupported, supported=supported)
    family, rows, types, pair_indices, _, _, charge_values = _coordinates_state(coordinates, atom_types, pairs, None, 0.0, sf_name, charges)
    value = potential_terms(coordinates, atom_types, pairs=pairs, sf_name=sf_name, charges=charges)
    _, jac = _raw_terms_and_jacobian(rows, types, pair_indices, family, charge_values, active_coordinates="coordinates" in wrt)
    def pullback(cotangent: Any) -> dict[str, Any]:
        cot = tuple(float(component) for component in cotangent)
        if len(cot) != len(value):
            raise ValueError(f"potential_terms cotangent must have length {len(value)}")
        gradient = [[0.0, 0.0, 0.0] for _ in rows]
        for term_index, coefficient in enumerate(cot):
            for i in range(len(rows)):
                for k in range(3):
                    gradient[i][k] += coefficient * jac[term_index][i][k]
        return {"coordinates": _restore_gradient(coordinates, gradient)}
    return value, pullback


def score_coordinates(coordinates: Any, atom_types: Any = None, *, pairs: Any = None, weights: Any = None, torsion_count: Any = 0.0, sf_name: str = "vina", charges: Any = None) -> float:
    """Evaluate a coordinate pair replay for ``sf_name``."""
    family, rows, types, pair_indices, coefficients, torsions, charge_values = _coordinates_state(coordinates, atom_types, pairs, weights, torsion_count, sf_name, charges)
    raw, _ = _raw_terms_and_jacobian(rows, types, pair_indices, family, charge_values, active_coordinates=False)
    return _recombine(raw, coefficients, torsions, family)[0]


def score_terms(coordinates: Any, atom_types: Any = None, *, pairs: Any = None, weights: Any = None, torsion_count: Any = 0.0, sf_name: str = "vina", charges: Any = None) -> tuple[float, ...]:
    """Return weighted per-term contributions whose sum is the composed score."""
    family, rows, types, pair_indices, coefficients, torsions, charge_values = _coordinates_state(coordinates, atom_types, pairs, weights, torsion_count, sf_name, charges)
    raw, _ = _raw_terms_and_jacobian(rows, types, pair_indices, family, charge_values, active_coordinates=False)
    return _recombine(raw, coefficients, torsions, family)[1]


weighted_terms = score_terms


def term_values(coordinates: Any, atom_types: Any = None, *, pairs: Any = None, torsion_count: Any = 0.0, sf_name: str = "vina", charges: Any = None) -> tuple[float, ...]:
    """Return unweighted per-term features, padded with the torsion input.

    The first values are the raw potential sums used by the upstream weighted
    accumulator.  The final value is ``torsion_count`` for all families.  For
    Vina/Vinardo the scorer applies the documented torsion division after this
    linear term vector, so ``sum(weight * term_values)`` is the pre-division
    potential energy; use :func:`score_terms` for exact post-division
    weighted contributions.
    """
    family, rows, types, pair_indices, _, torsions, charge_values = _coordinates_state(coordinates, atom_types, pairs, None, torsion_count, sf_name, charges)
    raw, _ = _raw_terms_and_jacobian(rows, types, pair_indices, family, charge_values, active_coordinates=False)
    return tuple(raw) + (torsions,)


def _precomputed_terms(terms: Any, expected: int) -> tuple[float, ...]:
    """Validate a term vector, or sum a matrix of per-interaction vectors."""
    if isinstance(terms, (str, bytes)):
        raise TypeError(f"terms must be a length-{expected} real sequence or an (M, {expected}) matrix")
    try:
        outer = list(terms)
    except TypeError as exc:
        raise TypeError(f"terms must be a length-{expected} real sequence or an (M, {expected}) matrix") from exc
    try:
        values = tuple(float(value) for value in outer)
        scalar_vector = len(values) == expected
    except (TypeError, ValueError):
        scalar_vector = False
        values = ()
    if not scalar_vector:
        sums = [0.0] * expected
        for row in outer:
            if isinstance(row, (str, bytes)):
                raise TypeError(f"terms must be a length-{expected} real sequence or an (M, {expected}) matrix")
            try:
                values_row = list(row)
            except TypeError as exc:
                raise TypeError(f"terms must be a length-{expected} real sequence or an (M, {expected}) matrix") from exc
            if len(values_row) != expected:
                raise ValueError(f"each precomputed interaction must have length {expected}")
            for index, value in enumerate(values_row):
                sums[index] += float(value)
        values = tuple(sums)
    if len(values) != expected or not all(math.isfinite(value) for value in values):
        raise ValueError(f"terms must be a finite length-{expected} real sequence")
    return values


@rules.jvp_for(term_values)
def _term_values_jvp(tangents: dict[str, Any], coordinates: Any, atom_types: Any = None, *, pairs: Any = None, torsion_count: Any = 0.0, sf_name: str = "vina", charges: Any = None) -> tuple[tuple[float, ...], Any]:
    unsupported = set(tangents) - {"coordinates"}
    if unsupported:
        raise UnsupportedWrt(term_values, unsupported, supported={"coordinates"})
    value = term_values(coordinates, atom_types, pairs=pairs, torsion_count=torsion_count, sf_name=sf_name, charges=charges)
    tangent = tangents.get("coordinates", ZERO)
    if tangent is ZERO:
        return value, ZERO
    family, rows, types, pair_indices, _, _, charge_values = _coordinates_state(coordinates, atom_types, pairs, None, torsion_count, sf_name, charges)
    tangent_rows = _tangent_rows(tangent, len(rows))
    _, jac = _raw_terms_and_jacobian(rows, types, pair_indices, family, charge_values, active_coordinates=True)
    return value, _directional_raw(jac, tangent_rows) + (0.0,)


@rules.vjp_for(term_values)
def _term_values_vjp(wrt: tuple[str, ...], coordinates: Any, atom_types: Any = None, *, pairs: Any = None, torsion_count: Any = 0.0, sf_name: str = "vina", charges: Any = None) -> tuple[tuple[float, ...], Any]:
    supported = {"coordinates"}
    unsupported = set(wrt) - supported
    if unsupported:
        raise UnsupportedWrt(term_values, unsupported, supported=supported)
    family, rows, types, pair_indices, _, _, charge_values = _coordinates_state(coordinates, atom_types, pairs, None, torsion_count, sf_name, charges)
    value = term_values(coordinates, atom_types, pairs=pairs, torsion_count=torsion_count, sf_name=sf_name, charges=charges)
    _, jac = _raw_terms_and_jacobian(rows, types, pair_indices, family, charge_values, active_coordinates=True)
    def pullback(cotangent: Any) -> dict[str, Any]:
        cot = tuple(float(component) for component in cotangent)
        if len(cot) != len(value):
            raise ValueError(f"term_values cotangent must have length {len(value)}")
        gradient = [[0.0, 0.0, 0.0] for _ in rows]
        for term_index, coefficient in enumerate(cot[:-1]):
            for i in range(len(rows)):
                for k in range(3):
                    gradient[i][k] += coefficient * jac[term_index][i][k]
        return {"coordinates": _restore_gradient(coordinates, gradient)}
    return value, pullback


def recombine_terms(terms: Any, *, weights: Any = None, torsion_count: Any = 0.0, sf_name: str = "vina") -> float:
    """Recombine precomputed unweighted potential sums with family weights."""
    family = _family(sf_name)
    expected = len(FAMILY_POTENTIAL_NAMES[family])
    values = _precomputed_terms(terms, expected)
    return _recombine(values, _weights(weights, family), _torsion_count(torsion_count), family)[0]


score = score_coordinates
energy = score_coordinates
score_family = score_coordinates


def _restore_gradient(original: Any, gradient: list[list[float]]) -> Any:
    if isinstance(original, tuple):
        return tuple(tuple(row) for row in gradient)
    try:
        import numpy as np
        if hasattr(original, "shape"):
            return np.asarray(gradient, dtype=float).reshape(original.shape)
    except ImportError:
        pass
    return gradient


def _restore_vector(original: Any, values: tuple[float, ...]) -> Any:
    if isinstance(original, tuple):
        return tuple(values)
    try:
        import numpy as np
        if hasattr(original, "shape"):
            return np.asarray(values, dtype=float).reshape(original.shape)
    except ImportError:
        pass
    return list(values)


def _tangent_rows(value: Any, n_rows: int) -> list[list[float]]:
    rows, _ = _rows(value, tangent=True)
    if len(rows) != n_rows:
        raise ValueError("coordinates tangent must have shape (N, 3)")
    return rows


def _tangent_vector(value: Any, expected: int, label: str) -> tuple[float, ...]:
    try:
        raw = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} tangent must be a length-{expected} real sequence") from exc
    if len(raw) != expected or not all(math.isfinite(component) for component in raw):
        raise ValueError(f"{label} tangent must be a finite length-{expected} real sequence")
    return raw


def _directional_raw(jac: list[list[list[float]]], tangent_rows: list[list[float]]) -> tuple[float, ...]:
    return tuple(sum(jac[t][i][k] * tangent_rows[i][k] for i in range(len(tangent_rows)) for k in range(3)) for t in range(len(jac)))


@rules.jvp_for(score_coordinates)
def _score_coordinates_jvp(tangents: dict[str, Any], coordinates: Any, atom_types: Any = None, *, pairs: Any = None, weights: Any = None, torsion_count: Any = 0.0, sf_name: str = "vina", charges: Any = None) -> tuple[float, Any]:
    unsupported = set(tangents) - {"coordinates", "weights"}
    if unsupported:
        raise UnsupportedWrt(score_coordinates, unsupported, supported={"coordinates", "weights"})
    family, rows, types, pair_indices, coefficients, torsions, charge_values = _coordinates_state(coordinates, atom_types, pairs, weights, torsion_count, sf_name, charges)
    value = score_coordinates(coordinates, atom_types, pairs=pairs, weights=weights, torsion_count=torsion_count, sf_name=sf_name, charges=charges)
    coordinate_tangent, weight_tangent = tangents.get("coordinates", ZERO), tangents.get("weights", ZERO)
    if coordinate_tangent is ZERO and weight_tangent is ZERO:
        return value, ZERO
    tangent_rows = _tangent_rows(coordinate_tangent, len(rows)) if coordinate_tangent is not ZERO else None
    tangent_weights = _tangent_vector(weight_tangent, len(coefficients), "weights") if weight_tangent is not ZERO else None
    raw, jac = _raw_terms_and_jacobian(rows, types, pair_indices, family, charge_values, active_coordinates=tangent_rows is not None)
    raw_direction = _directional_raw(jac, tangent_rows) if tangent_rows is not None else (0.0,) * len(raw)
    if family in ("vina", "vinardo"):
        denominator = 1.0 + coefficients[-1] * torsions
        energy_value = sum(coefficients[i] * raw[i] for i in range(len(raw)))
        d_energy = sum(coefficients[i] * raw_direction[i] for i in range(len(raw)))
        d_denominator = 0.0 if tangent_weights is None else tangent_weights[-1] * torsions
        if tangent_weights is not None:
            d_energy += sum(tangent_weights[i] * raw[i] for i in range(len(raw)))
        directional = (d_energy * denominator - energy_value * d_denominator) / denominator**2
    else:
        directional = sum(coefficients[i] * raw_direction[i] for i in range(len(raw)))
        if tangent_weights is not None:
            directional += sum(tangent_weights[i] * raw[i] for i in range(len(raw))) + tangent_weights[-1] * torsions
    return value, directional


@rules.vjp_for(score_coordinates)
def _score_coordinates_vjp(wrt: tuple[str, ...], coordinates: Any, atom_types: Any = None, *, pairs: Any = None, weights: Any = None, torsion_count: Any = 0.0, sf_name: str = "vina", charges: Any = None) -> tuple[float, Any]:
    supported = {"coordinates", "weights"}
    unsupported = set(wrt) - supported
    if unsupported:
        raise UnsupportedWrt(score_coordinates, unsupported, supported=supported)
    family, rows, types, pair_indices, coefficients, torsions, charge_values = _coordinates_state(coordinates, atom_types, pairs, weights, torsion_count, sf_name, charges)
    value = score_coordinates(coordinates, atom_types, pairs=pairs, weights=weights, torsion_count=torsion_count, sf_name=sf_name, charges=charges)
    raw, jac = _raw_terms_and_jacobian(rows, types, pair_indices, family, charge_values, active_coordinates="coordinates" in wrt)
    _, _, raw_gradient, weight_gradient = _recombine(raw, coefficients, torsions, family)
    def pullback(cotangent: Any) -> dict[str, Any]:
        factor = float(cotangent)
        result: dict[str, Any] = {}
        if "coordinates" in wrt:
            coordinate_gradient = [[0.0, 0.0, 0.0] for _ in rows]
            for term_index, coefficient in enumerate(raw_gradient):
                for i in range(len(rows)):
                    for k in range(3):
                        coordinate_gradient[i][k] += coefficient * jac[term_index][i][k]
            result["coordinates"] = _restore_gradient(coordinates, [[factor * component for component in row] for row in coordinate_gradient])
        if "weights" in wrt:
            result["weights"] = _restore_vector(weights if weights is not None else coefficients, tuple(factor * component for component in weight_gradient))
        return result
    return value, pullback


@rules.jvp_for(score_terms)
def _score_terms_jvp(tangents: dict[str, Any], coordinates: Any, atom_types: Any = None, *, pairs: Any = None, weights: Any = None, torsion_count: Any = 0.0, sf_name: str = "vina", charges: Any = None) -> tuple[tuple[float, ...], Any]:
    unsupported = set(tangents) - {"coordinates", "weights"}
    if unsupported:
        raise UnsupportedWrt(score_terms, unsupported, supported={"coordinates", "weights"})
    family, rows, types, pair_indices, coefficients, torsions, charge_values = _coordinates_state(coordinates, atom_types, pairs, weights, torsion_count, sf_name, charges)
    coordinate_tangent, weight_tangent = tangents.get("coordinates", ZERO), tangents.get("weights", ZERO)
    values = score_terms(coordinates, atom_types, pairs=pairs, weights=weights, torsion_count=torsion_count, sf_name=sf_name, charges=charges)
    if coordinate_tangent is ZERO and weight_tangent is ZERO:
        return values, ZERO
    tangent_rows = _tangent_rows(coordinate_tangent, len(rows)) if coordinate_tangent is not ZERO else None
    tangent_weights = _tangent_vector(weight_tangent, len(coefficients), "weights") if weight_tangent is not ZERO else (0.0,) * len(coefficients)
    raw, jac = _raw_terms_and_jacobian(rows, types, pair_indices, family, charge_values, active_coordinates=tangent_rows is not None)
    raw_direction = _directional_raw(jac, tangent_rows) if tangent_rows is not None else (0.0,) * len(raw)
    if family in ("vina", "vinardo"):
        denominator = 1.0 + coefficients[-1] * torsions
        energy_value = sum(coefficients[i] * raw[i] for i in range(len(raw)))
        d_energy = sum(coefficients[i] * raw_direction[i] + tangent_weights[i] * raw[i] for i in range(len(raw)))
        d_denominator = tangent_weights[-1] * torsions
        d_score = (d_energy * denominator - energy_value * d_denominator) / denominator**2
        out = [((tangent_weights[i] * raw[i] + coefficients[i] * raw_direction[i]) * denominator - coefficients[i] * raw[i] * d_denominator) / denominator**2 for i in range(len(raw))]
        out.append(d_score - sum(out))
        return values, tuple(out)
    out = tuple(tangent_weights[i] * raw[i] + coefficients[i] * raw_direction[i] for i in range(len(raw))) + (tangent_weights[-1] * torsions,)
    return values, out


@rules.vjp_for(score_terms)
def _score_terms_vjp(wrt: tuple[str, ...], coordinates: Any, atom_types: Any = None, *, pairs: Any = None, weights: Any = None, torsion_count: Any = 0.0, sf_name: str = "vina", charges: Any = None) -> tuple[tuple[float, ...], Any]:
    supported = {"coordinates", "weights"}
    unsupported = set(wrt) - supported
    if unsupported:
        raise UnsupportedWrt(score_terms, unsupported, supported=supported)
    family, rows, types, pair_indices, coefficients, torsions, charge_values = _coordinates_state(coordinates, atom_types, pairs, weights, torsion_count, sf_name, charges)
    values = score_terms(coordinates, atom_types, pairs=pairs, weights=weights, torsion_count=torsion_count, sf_name=sf_name, charges=charges)
    raw, jac = _raw_terms_and_jacobian(rows, types, pair_indices, family, charge_values, active_coordinates="coordinates" in wrt)
    def pullback(cotangent: Any) -> dict[str, Any]:
        cot = tuple(float(v) for v in cotangent)
        if len(cot) != len(values):
            raise ValueError(f"score_terms cotangent must have length {len(values)}")
        result: dict[str, Any] = {}
        if family in ("vina", "vinardo"):
            denominator = 1.0 + coefficients[-1] * torsions
            energy_value = sum(coefficients[i] * raw[i] for i in range(len(raw)))
            numerator = sum(cot[i] * coefficients[i] * raw[i] for i in range(len(raw))) + cot[-1] * energy_value
            scale = tuple(cot[i] / denominator + cot[-1] * (1.0 / denominator - 1.0) for i in range(len(raw)))
            raw_coefficients = tuple(coefficients[i] * scale[i] for i in range(len(raw)))
            weight_gradient = tuple(raw[i] * scale[i] for i in range(len(raw))) + (-numerator * torsions / denominator**2,)
        else:
            raw_coefficients = tuple(coefficients[i] * cot[i] for i in range(len(raw)))
            weight_gradient = tuple(raw[i] * cot[i] for i in range(len(raw))) + (cot[-1] * torsions,)
        if "coordinates" in wrt:
            gradient = [[0.0, 0.0, 0.0] for _ in rows]
            for term_index, coefficient in enumerate(raw_coefficients):
                for i in range(len(rows)):
                    for k in range(3):
                        gradient[i][k] += coefficient * jac[term_index][i][k]
            result["coordinates"] = _restore_gradient(coordinates, gradient)
        if "weights" in wrt:
            result["weights"] = _restore_vector(weights if weights is not None else coefficients, weight_gradient)
        return result
    return values, pullback


@rules.jvp_for(recombine_terms)
def _recombine_terms_jvp(tangents: dict[str, Any], terms: Any, *, weights: Any = None, torsion_count: Any = 0.0, sf_name: str = "vina") -> tuple[float, Any]:
    unsupported = set(tangents) - {"terms", "weights"}
    if unsupported:
        raise UnsupportedWrt(recombine_terms, unsupported, supported={"terms", "weights"})
    family = _family(sf_name)
    expected = len(FAMILY_POTENTIAL_NAMES[family])
    raw = _precomputed_terms(terms, expected)
    coefficients = _weights(weights, family)
    torsions = _torsion_count(torsion_count)
    value = _recombine(raw, coefficients, torsions, family)[0]
    tangent_terms = _precomputed_terms(tangents["terms"], expected) if tangents.get("terms", ZERO) is not ZERO else (0.0,) * expected
    tangent_weights = _tangent_vector(tangents["weights"], len(coefficients), "weights") if tangents.get("weights", ZERO) is not ZERO else (0.0,) * len(coefficients)
    if family in ("vina", "vinardo"):
        denominator = 1.0 + coefficients[-1] * torsions
        energy_value = sum(coefficients[i] * raw[i] for i in range(expected))
        d_energy = sum(tangent_weights[i] * raw[i] + coefficients[i] * tangent_terms[i] for i in range(expected))
        return value, (d_energy * denominator - energy_value * tangent_weights[-1] * torsions) / denominator**2
    return value, sum(tangent_weights[i] * raw[i] + coefficients[i] * tangent_terms[i] for i in range(expected)) + tangent_weights[-1] * torsions


@rules.vjp_for(recombine_terms)
def _recombine_terms_vjp(wrt: tuple[str, ...], terms: Any, *, weights: Any = None, torsion_count: Any = 0.0, sf_name: str = "vina") -> tuple[float, Any]:
    supported = {"terms", "weights"}
    unsupported = set(wrt) - supported
    if unsupported:
        raise UnsupportedWrt(recombine_terms, unsupported, supported=supported)
    family = _family(sf_name)
    expected = len(FAMILY_POTENTIAL_NAMES[family])
    raw = _precomputed_terms(terms, expected)
    coefficients = _weights(weights, family)
    torsions = _torsion_count(torsion_count)
    value, _, raw_gradient, weight_gradient = _recombine(raw, coefficients, torsions, family)
    def pullback(cotangent: Any) -> dict[str, Any]:
        factor = float(cotangent)
        result = {}
        if "terms" in wrt:
            result["terms"] = tuple(factor * item for item in raw_gradient)
        if "weights" in wrt:
            result["weights"] = _restore_vector(weights if weights is not None else coefficients, tuple(factor * item for item in weight_gradient))
        return result
    return value, pullback


@dataclass
class ScoringFunction:
    """Python facade mirroring Vina's explicit family and ``set_weights`` API."""

    sf_name: str = "vina"
    weights: Any = None

    def __post_init__(self) -> None:
        self.sf_name = _family(self.sf_name)
        self.weights = _weights(self.weights, self.sf_name)

    @property
    def term_names(self) -> tuple[str, ...]:
        return family_term_names(self.sf_name)

    def set_weights(self, weights: Any) -> None:
        self.weights = _weights(weights, self.sf_name)

    def get_weights(self) -> tuple[float, ...]:
        return tuple(self.weights)

    def score(self, coordinates: Any, atom_types: Any = None, *, pairs: Any = None, torsion_count: Any = 0.0, charges: Any = None) -> float:
        return score_coordinates(coordinates, atom_types, pairs=pairs, weights=self.weights, torsion_count=torsion_count, sf_name=self.sf_name, charges=charges)

    def terms(self, coordinates: Any, atom_types: Any = None, *, pairs: Any = None, torsion_count: Any = 0.0, charges: Any = None) -> tuple[float, ...]:
        return score_terms(coordinates, atom_types, pairs=pairs, weights=self.weights, torsion_count=torsion_count, sf_name=self.sf_name, charges=charges)
