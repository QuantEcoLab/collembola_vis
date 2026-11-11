"""Analytic geometry/guard checks; no models, dataset writes or inference."""
import math

import numpy as np
from scipy.spatial import cKDTree

from scripts.experimental_centerline import Config, analyze, graph_diameter_path, prune_branches, skeleton_graph


def capsule():
    y, x = np.mgrid[:200, :260]
    return (x - np.clip(x, 60, 180))**2 + (y - 100)**2 <= 16**2


def test_straight_capsule_length_volume_and_exact_section_integration():
    mask = capsule()
    before = mask.copy()
    result = analyze(mask, 8.)
    assert np.array_equal(mask, before)
    assert abs(result['length_px'] - 152) < 3
    exact_volume = math.pi * 16**2 * 120 + 4 / 3 * math.pi * 16**3
    assert abs(result['volume_px3'] / exact_volume - 1) < .12
    assert math.isclose(result['volume_px3'], sum(s['volume_px3'] for s in result['cross_sections']), rel_tol=1e-12)
    assert math.isclose(result['length_px'], sum(s['delta_l_px'] for s in result['cross_sections']), rel_tol=1e-12)
    assert math.isclose(result['length_mm'], result['length_px'] * .008)
    assert math.isclose(result['volume_mm3'], result['volume_px3'] * .008**3)
    points = np.array(result['centerline_xy'])
    for tangent, section in zip(np.diff(points, axis=0), result['cross_sections']):
        assert abs(np.dot(tangent, section['normal_xy'])) < 1e-9
    assert not result['geometry_unreliable'], result['warnings']
    assert not result['aggregate_included']


def test_curved_capsule_follows_arc_not_chord():
    angles = np.linspace(-1., 2., 1000)
    points = np.column_stack([140 + 80 * np.cos(angles), 140 + 80 * np.sin(angles)])
    y, x = np.mgrid[:280, :280]
    mask = cKDTree(points).query(np.column_stack([x.ravel(), y.ravel()]))[0].reshape(x.shape) <= 12
    result = analyze(mask)
    assert result['length_px'] is not None, result['warnings']
    assert abs(result['length_px'] - (80 * 3 + 24)) / 264 < .05
    assert result['length_px'] > 1.3 * result['endpoint_chord_px']
    expected = math.pi * 12**2 * 240 + 4 / 3 * math.pi * 12**3
    assert abs(result['volume_px3'] / expected - 1) < .12
    assert result['length_mm'] is None and result['volume_mm3'] is None


def test_thin_legs_and_antenna_do_not_become_main_path_or_width():
    mask = capsule()
    mask[40:100, 108:113] = True
    mask[98:103, 25:60] = True
    result = analyze(mask)
    path = np.asarray(result['centerline_xy'])
    assert path[:, 1].min() > 90
    assert path[:, 1].max() < 110
    assert path[:, 0].min() > 40
    assert result['width_max_px'] < 38
    assert 145 < result['length_px'] < 160


def test_multiple_components_crop_edge_and_loops_are_excluded():
    mask = capsule()
    mask[5:8, 5:8] = True
    result = analyze(mask, 8)
    assert 'MULTIPLE_COMPONENTS' in result['warnings']
    assert result['geometry_unreliable'] and not result['aggregate_included']
    assert len(result['component_areas_px']) == 2
    clipped = analyze(capsule()[90:, :])
    assert 'CROP_EDGE_CONTACT' in clipped['warnings']
    y, x = np.mgrid[:180, :180]
    ring = (np.hypot(x - 90, y - 90) - 50)**2 <= 8**2
    result = analyze(ring)
    assert 'SKELETON_LOOP_AMBIGUITY' in result['warnings']
    assert 'NO_RELIABLE_MAIN_PATH' in result['warnings']
    assert result['length_px'] is None and result['volume_px3'] is None
    assert not result['aggregate_included']


def test_empty_and_tiny_masks_do_not_produce_bogus_physical_measurements():
    assert analyze(np.zeros((20, 20), dtype=bool))['length_px'] is None
    tiny = np.zeros((20, 20), dtype=bool)
    tiny[8, 8] = True
    result = analyze(tiny, 8)
    assert result['volume_mm3'] is None
    assert result['geometry_unreliable']


def test_short_thin_side_branch_is_pruned_without_removing_body_ends():
    skeleton = np.zeros((60, 60), dtype=bool)
    skeleton[30, 10:51] = True
    skeleton[20:31, 30] = True
    coords, graph = skeleton_graph(skeleton)
    radii = np.ones(skeleton.shape) * 2
    radii[30, 10:51] = 8
    clean, removed = prune_branches(coords, graph, radii, 8, Config())
    path, length, endpoints = graph_diameter_path(coords, clean, 24)
    assert len(removed) == 1 and removed[0]['reason'] == 'short_thin'
    assert len(endpoints) == 2
    assert length == 40
    assert np.all(path[:, 1] == 30)


def test_branch_pixels_not_only_branch_skeleton_are_excluded_from_width_mask():
    mask = capsule()
    mask[55:101, 104:118] = True  # survives scale opening, but is a short side limb
    result = analyze(mask)
    assert result['removed_branches']
    assert result['branch_territory_removed_area_px'] > 0
    assert not result['body_mask'][65, 110]
    assert result['width_max_px'] < 38
    assert all(abs(p[1] - 100) < 10 for p in result['centerline_xy'])
