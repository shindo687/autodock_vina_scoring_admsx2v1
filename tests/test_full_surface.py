import math

import pytest

import vina_ad


def _fd_score(coordinates, atom_types, *, sf_name, charges=None, weights=None, index=0, h=1e-6, torsion_count=0.0):
    kwargs = {"sf_name": sf_name, "torsion_count": torsion_count}
    if charges is not None:
        kwargs["charges"] = charges
    if weights is not None:
        kwargs["weights"] = weights
    plus = [list(row) for row in coordinates]
    minus = [list(row) for row in coordinates]
    plus[1][0] += h
    minus[1][0] -= h
    return (vina_ad.score_coordinates(plus, atom_types, **kwargs) - vina_ad.score_coordinates(minus, atom_types, **kwargs)) / (2 * h)


@pytest.mark.parametrize("family, expected", [("vina", 7), ("vinardo", 6), ("ad4", 6)])
def test_family_defaults_and_weighted_term_reconstruction(family, expected):
    coords = ((0.0, 0.0, 0.0), (3.1, 0.2, 0.0))
    if family == "ad4":
        types, charges = (0, 3), (0.25, -0.35)
    else:
        types, charges = (0, 0), None
    kwargs = {"sf_name": family}
    if charges is not None:
        kwargs["charges"] = charges
    terms = vina_ad.score_terms(coords, types, **kwargs)
    score = vina_ad.score_coordinates(coords, types, **kwargs)
    assert len(terms) == expected
    assert sum(terms) == pytest.approx(score, abs=1e-12)
    assert len(vina_ad.potential_terms(coords, types, **kwargs)) == expected - 1
    assert len(vina_ad.family_term_names(family)) == expected


@pytest.mark.parametrize("family", ["vina", "vinardo", "ad4"])
def test_recombine_precomputed_terms_matches_coordinate_surface(family):
    coords = ((0.0, 0.0, 0.0), (3.1, 0.2, 0.0))
    if family == "ad4":
        types, charges = (0, 3), (0.25, -0.35)
    else:
        types, charges = (0, 0), None
    kwargs = {"sf_name": family, "torsion_count": 2.0}
    if charges is not None:
        kwargs["charges"] = charges
    raw = vina_ad.potential_terms(coords, types, sf_name=family, **({"charges": charges} if charges is not None else {}))
    assert vina_ad.recombine_terms(raw, **{k: v for k, v in kwargs.items() if k != "charges"}) == pytest.approx(vina_ad.score_coordinates(coords, types, **kwargs))


def test_vinardo_and_ad4_weight_vjps_match_finite_difference():
    coords = ((0.0, 0.0, 0.0), (3.1, 0.2, 0.0))
    for family, types, charges in (("vinardo", (0, 0), None), ("ad4", (0, 3), (0.25, -0.35))):
        kwargs = {"sf_name": family, "torsion_count": 1.0}
        if charges is not None:
            kwargs["charges"] = charges
        weights = list(vina_ad.FAMILY_DEFAULT_WEIGHTS[family])
        value, pullback = vina_ad.vjp(vina_ad.score_coordinates, coords, types, wrt=("coordinates", "weights"), weights=weights, **kwargs)
        gradients = pullback(1.0)
        assert value == pytest.approx(vina_ad.score_coordinates(coords, types, weights=weights, **kwargs))
        h = 1e-6
        perturbed = weights[:]
        perturbed[0] += h
        plus = vina_ad.score_coordinates(coords, types, weights=perturbed, **kwargs)
        perturbed[0] -= 2 * h
        minus = vina_ad.score_coordinates(coords, types, weights=perturbed, **kwargs)
        assert gradients["weights"][0] == pytest.approx((plus - minus) / (2 * h), abs=2e-6)
        plus_kwargs = {"sf_name": family, "charges": charges, "weights": weights, "torsion_count": 1.0}
        assert gradients["coordinates"][1][0] == pytest.approx(_fd_score(coords, types, **plus_kwargs), abs=2e-6)


def test_public_scoring_function_set_weights_facade():
    scorer = vina_ad.ScoringFunction("vinardo")
    assert scorer.get_weights() == vina_ad.DEFAULT_VINARDO_WEIGHTS
    scorer.set_weights((1, 2, 3, 4, 5, 6))
    assert scorer.get_weights() == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert scorer.score(((0., 0., 0.), (3., 0., 0.)), (0, 0)) == pytest.approx(
        vina_ad.score_coordinates(((0., 0., 0.), (3., 0., 0.)), (0, 0), sf_name="vinardo", weights=(1, 2, 3, 4, 5, 6))
    )


def test_recombine_rules_have_term_and_weight_derivatives():
    terms = (1.2, -0.5, 0.25, 0.1, 0.0)
    weights = vina_ad.DEFAULT_VINARDO_WEIGHTS
    value, tangent = vina_ad.jvp(vina_ad.recombine_terms, terms, sf_name="vinardo", weights=weights, torsion_count=2.0, tangents={"terms": (0.1, 0.2, 0.3, 0.4, 0.5)})
    h = 1e-6
    plus = tuple(x + h * d for x, d in zip(terms, (0.1, 0.2, 0.3, 0.4, 0.5)))
    minus = tuple(x - h * d for x, d in zip(terms, (0.1, 0.2, 0.3, 0.4, 0.5)))
    assert tangent == pytest.approx((vina_ad.recombine_terms(plus, sf_name="vinardo", weights=weights, torsion_count=2.0) - vina_ad.recombine_terms(minus, sf_name="vinardo", weights=weights, torsion_count=2.0)) / (2 * h), abs=2e-6)
    assert math.isfinite(value)


def test_recombine_accepts_precomputed_interaction_matrix():
    rows = ((1.0, 2.0, 3.0, 4.0, 5.0), (0.5, -1.0, 0.25, 0.0, 2.0))
    summed = tuple(sum(row[i] for row in rows) for i in range(5))
    assert vina_ad.recombine_terms(rows, sf_name="vinardo") == pytest.approx(
        vina_ad.recombine_terms(summed, sf_name="vinardo")
    )
