"""Experimental 2-D body centerline and circular-section volume. Pure, no I/O/models.

Body/appendage separation is a scale-based geometric hypothesis, not anatomical
classification. Every result requires visual biological review before adoption.
"""
from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from math import hypot, pi

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class Config:
    opening_radius_fraction: float = 0.35
    minimum_opening_radius_px: float = 1.0
    reference_radius_quantile: float = 0.99
    short_branch_local_radius_factor: float = 3.0
    short_branch_radius_fraction: float = 0.75
    long_thin_branch_radius_fraction: float = 0.45
    smoothing_sigma_px: float = 1.5
    sampling_step_px: float = 1.0
    ray_step_px: float = 0.5
    maximum_body_area_loss_fraction: float = 0.35
    maximum_endpoints: int = 24


def largest_component(mask):
    labels, count = ndi.label(mask, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())[1:]
    if not count:
        return np.zeros_like(mask, dtype=bool), []
    return labels == (int(np.argmax(sizes)) + 1), sorted(map(int, sizes), reverse=True)


def distance(mask):
    # Pad even a fully filled crop so distance to the exterior is well-defined.
    return ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]


def body_opening(mask, radius):
    padding = int(np.ceil(radius)) + 3
    padded = np.pad(mask, padding)
    eroded = ndi.distance_transform_edt(padded) > radius
    if not eroded.any():
        return np.zeros_like(mask, dtype=bool)
    opened = (ndi.distance_transform_edt(~eroded) <= radius) & padded
    return opened[padding:-padding, padding:-padding]


def skeleton_graph(skeleton):
    coords = np.column_stack(np.nonzero(skeleton))  # y, x
    lookup = {tuple(p): i for i, p in enumerate(coords)}
    graph = {i: {} for i in range(len(coords))}
    for i, (y, x) in enumerate(coords):
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            j = lookup.get((y + dy, x + dx))
            if j is None:
                continue
            # Suppress diagonal shortcuts/triangles when an orthogonal link exists.
            if dy and dx and ((y + dy, x) in lookup or (y, x + dx) in lookup):
                continue
            graph[i][j] = graph[j][i] = hypot(dy, dx)
    return coords, graph


def prune_branches(coords, graph, radii, reference, config):
    graph = {i: dict(edges) for i, edges in graph.items()}
    removed = []
    for _ in range(30):
        candidates = []
        for endpoint in [i for i in graph if len(graph[i]) == 1]:
            chain = [endpoint]
            previous = None
            current = endpoint
            length = 0.0
            while True:
                next_nodes = [j for j in graph[current] if j != previous]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                length += graph[current][nxt]
                chain.append(nxt)
                previous, current = current, nxt
                if len(graph[current]) != 2:
                    break
            if len(graph[current]) < 3 or len(chain) < 2:
                continue
            branch = chain[:-1]
            values = np.array([radii[tuple(coords[i])] for i in branch])
            junction_radius = float(radii[tuple(coords[current])])
            short = length <= max(4, config.short_branch_local_radius_factor * junction_radius)
            thin = float(np.median(values)) < config.short_branch_radius_fraction * reference
            distal = values[:max(1, int(len(values) * 0.8))]
            long_thin = float(np.quantile(distal, .9)) < config.long_thin_branch_radius_fraction * reference
            if (short and thin) or long_thin:
                candidates.append((length, branch, chain, 'short_thin' if short and thin else 'long_thin'))
        if not candidates:
            break
        # Remove one branch, then recompute topology; do not accidentally remove
        # both remaining body ends using stale junction degrees.
        _, branch, chain, reason = min(candidates, key=lambda row: row[0])
        removed.append({'reason': reason, 'points_xy': coords[chain][:, ::-1].astype(float).tolist()})
        for i in branch:
            for j in list(graph[i]):
                graph[j].pop(i, None)
            graph.pop(i)
    return graph, removed


def dijkstra(graph, start):
    distances = {start: 0.0}
    previous = {}
    heap = [(0.0, start)]
    while heap:
        dist, i = heappop(heap)
        if dist != distances[i]:
            continue
        for j, weight in graph[i].items():
            candidate = dist + weight
            if candidate < distances.get(j, float('inf')):
                distances[j] = candidate
                previous[j] = i
                heappush(heap, (candidate, j))
    return distances, previous


def graph_diameter_path(coords, graph, max_endpoints):
    endpoints = sorted(i for i in graph if len(graph[i]) == 1)
    if len(endpoints) < 2 or len(endpoints) > max_endpoints:
        return None, None, endpoints
    best = None
    for start in endpoints:
        distances, previous = dijkstra(graph, start)
        for end in endpoints:
            if end != start and end in distances and (best is None or distances[end] > best[0]):
                nodes = [end]
                while nodes[-1] != start:
                    nodes.append(previous[nodes[-1]])
                best = distances[end], nodes[::-1]
    if best is None:
        return None, None, endpoints
    return coords[best[1]][:, ::-1].astype(float), float(best[0]), endpoints


def signed_distance(mask):
    padded = np.pad(mask, 2)
    return (ndi.distance_transform_edt(padded) - ndi.distance_transform_edt(~padded))[2:-2, 2:-2]


def sample(field, points):
    points = np.asarray(points)
    return ndi.map_coordinates(field, [points[:, 1], points[:, 0]], order=1, mode='constant', cval=-1)


def first_boundary(field, origin, direction, step, limit):
    intervals = np.arange(0, limit + step, step)
    values = sample(field, origin + intervals[:, None] * direction)
    if values[0] < 0:
        return None
    exits = np.flatnonzero(values <= 0)
    if not len(exits):
        return None
    i = int(exits[0])
    if i == 0:
        return 0.0
    t = values[i - 1] / (values[i - 1] - values[i])
    return float(intervals[i - 1] + t * step)


def resample_path(points, step):
    length = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, length > 1e-9]
    points = points[keep]
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if s[-1] <= 0:
        return points
    targets = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.column_stack([np.interp(targets, s, points[:, j]) for j in (0, 1)])


def analyze(mask, calibration_um_per_px=None, config=Config()):
    """Return diagnostic candidates and all intermediate geometry. Never mutate input.

    Numerical candidates on flagged masks are NOT accepted measurements. No
    aggregate inclusion is enabled, even for an unflagged pending-review pilot.
    """
    raw = np.asarray(mask, dtype=bool).copy()
    if raw.ndim != 2:
        raise ValueError('Expected a 2-D binary mask')
    if calibration_um_per_px is not None and (not np.isfinite(calibration_um_per_px) or calibration_um_per_px <= 0):
        raise ValueError('Calibration must be positive µm/px')
    warnings = []
    main, component_areas = largest_component(raw)
    if len(component_areas) > 1:
        warnings.append('MULTIPLE_COMPONENTS')
    if raw.any() and (raw[0].any() or raw[-1].any() or raw[:, 0].any() or raw[:, -1].any()):
        warnings.append('CROP_EDGE_CONTACT')
    out = {'method': 'experimental_centerline_circular_sections_v2', 'config': asdict(config),
           'component_areas_px': component_areas, 'warnings': warnings,
           'aggregate_included': False, 'biological_review': 'pending',
           'length_px': None, 'length_mm': None, 'volume_px3': None, 'volume_mm3': None,
           'main_component_mask': main, 'body_mask': np.zeros_like(raw),
           'raw_skeleton_mask': np.zeros_like(raw), 'removed_skeleton_mask': np.zeros_like(raw),
           'body_skeleton_mask': np.zeros_like(raw),
           'remaining_off_path_mask': np.zeros_like(raw), 'removed_branches': [],
           'centerline_xy': [], 'cross_sections': []}

    def finish():
        out['geometry_unreliable'] = bool(warnings)
        return out

    if not main.any():
        warnings.append('EMPTY_MASK')
        return finish()
    radii = distance(main)
    out['raw_skeleton_mask'] = skeletonize(raw)
    reference = float(np.quantile(radii[main], config.reference_radius_quantile))
    radius = max(config.minimum_opening_radius_px, config.opening_radius_fraction * reference)
    body, opened_areas = largest_component(body_opening(main, radius))
    out.update(reference_radius_px=reference, opening_radius_px=radius, body_mask=body,
               body_area_px=int(body.sum()), original_area_px=int(raw.sum()), opening_component_areas_px=opened_areas)
    if not body.any():
        warnings.append('BODY_EXTRACTION_EMPTY')
        return finish()
    area_loss = 1 - float(body.sum()) / float(main.sum())
    out['body_area_loss_fraction'] = area_loss
    if area_loss > config.maximum_body_area_loss_fraction:
        warnings.append('EXCESSIVE_BODY_REMOVAL')
    if len(opened_areas) > 1 and opened_areas[1] > .05 * opened_areas[0]:
        warnings.append('BODY_EXTRACTION_SPLIT')
    if 2 * reference < 6:
        warnings.append('BODY_TOO_THIN_FOR_STABLE_WIDTH')
    skeleton = skeletonize(body)
    coords, graph = skeleton_graph(skeleton)
    out['body_skeleton_mask'] = skeleton
    if len(graph) < 3:
        warnings.append('NO_RELIABLE_MAIN_PATH')
        return finish()
    clean, removed = prune_branches(coords, graph, distance(body), reference, config)
    out['removed_branches'] = removed
    removed_mask = out['raw_skeleton_mask'] & ~body
    out['scale_removed_skeleton_pixels'] = int(removed_mask.sum())
    for i in set(graph) - set(clean):
        removed_mask[tuple(coords[i])] = True
    out['removed_skeleton_mask'] = removed_mask
    if not clean:
        warnings.append('NO_RELIABLE_MAIN_PATH')
        return finish()
    # Count cycles separately from branch degree; loops are never silently treated
    # as a unique anatomical head-to-tail path.
    components = 0
    unseen = set(clean)
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            for j in clean[stack.pop()]:
                if j in unseen:
                    unseen.remove(j)
                    stack.append(j)
    cycles = sum(len(e) for e in clean.values()) // 2 - len(clean) + components
    out['skeleton_cycles'] = cycles
    if cycles:
        warnings.append('SKELETON_LOOP_AMBIGUITY')
    if components > 1:
        warnings.append('DISCONNECTED_BODY_SKELETON')
    path, graph_length, endpoints = graph_diameter_path(coords, clean, config.maximum_endpoints)
    out['endpoint_count_after_pruning'] = len(endpoints)
    out['skeleton_geodesic_length_px'] = graph_length
    if path is None:
        warnings.append('NO_RELIABLE_MAIN_PATH')
        return finish()
    if len(endpoints) != 2:
        warnings.append('MAIN_PATH_BRANCH_AMBIGUITY')
    off_path = np.zeros_like(raw)
    chosen_pixels = {tuple(map(int, point[::-1])) for point in path}
    for i in clean:
        if tuple(coords[i]) not in chosen_pixels:
            off_path[tuple(coords[i])] = True
    out['remaining_off_path_mask'] = off_path
    # Skeleton pruning alone must not leave a thick limb root inside a width
    # transect. Assign body pixels to their nearest body-skeleton seed: retained
    # main path versus discarded branch. Width rays use the retained territory,
    # not the original opened silhouette. This cut is explicit/visualized.
    path_seeds = np.zeros_like(raw)
    for point in path:
        path_seeds[int(point[1]), int(point[0])] = True
    discarded_seeds = skeleton & ~path_seeds
    territory_removed = np.zeros_like(raw)
    if discarded_seeds.any():
        territory = body & (ndi.distance_transform_edt(~path_seeds) <= ndi.distance_transform_edt(~discarded_seeds))
        refined, territory_components = largest_component(territory)
        territory_removed = body & ~refined
        if (path_seeds & ~refined).any() or len(territory_components) > 1:
            warnings.append('BRANCH_TERRITORY_CUT_AMBIGUOUS')
        if territory_removed.sum() > .15 * body.sum():
            warnings.append('SUBSTANTIAL_BRANCH_TERRITORY_REMOVAL')
        body = refined
        out['removed_skeleton_mask'] |= discarded_seeds
    out['branch_territory_removed_mask'] = territory_removed
    out['branch_territory_removed_area_px'] = int(territory_removed.sum())
    out['body_mask'] = body
    out['body_area_px'] = int(body.sum())
    out['body_area_loss_fraction'] = 1 - float(body.sum()) / float(main.sum())
    if out['body_area_loss_fraction'] > config.maximum_body_area_loss_fraction and 'EXCESSIVE_BODY_REMOVAL' not in warnings:
        warnings.append('EXCESSIVE_BODY_REMOVAL')
    # Smooth only the selected path, never grow the mask into a limb/antenna.
    path = resample_path(path, 1.)
    if len(path) >= 5:
        path = ndi.gaussian_filter1d(path, config.smoothing_sigma_px, axis=0, mode='nearest')
    field = signed_distance(body)
    if (sample(field, path) <= 0).any():
        warnings.append('SMOOTHED_PATH_OUTSIDE_BODY')
    # Extend each end to the first boundary of the *cleaned body*, not raw mask.
    extensions = []
    for index, neighbor in ((0, min(5, len(path) - 1)), (-1, max(0, len(path) - 6))):
        direction = path[index] - path[neighbor]
        norm = np.linalg.norm(direction)
        length = first_boundary(field, path[index], direction / norm, config.ray_step_px, max(raw.shape)) if norm > 0 else None
        if length is None:
            warnings.append('BODY_ENDPOINT_UNCERTAIN')
            extensions.append(path[index])
        else:
            extensions.append(path[index] + direction / norm * length)
    path = resample_path(np.vstack([extensions[0], path, extensions[1]]), config.sampling_step_px)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    out['centerline_xy'] = path.tolist()
    out['length_px'] = float(lengths.sum())
    out['endpoint_chord_px'] = float(np.linalg.norm(path[-1] - path[0]))
    out['tortuosity'] = out['length_px'] / max(out['endpoint_chord_px'], 1e-9)
    position = 0.
    valid_widths = True
    sections = []
    for i, dl in enumerate(lengths):
        midpoint = (path[i] + path[i + 1]) / 2
        tangent = (path[i + 1] - path[i]) / dl
        normal = np.array([-tangent[1], tangent[0]])
        left = first_boundary(field, midpoint, normal, config.ray_step_px, max(raw.shape))
        right = first_boundary(field, midpoint, -normal, config.ray_step_px, max(raw.shape))
        width = left + right if left is not None and right is not None else None
        valid_widths &= width is not None and width > 0
        sections.append({'s_px': position + dl / 2, 'x_px': float(midpoint[0]), 'y_px': float(midpoint[1]),
                         'normal_xy': normal.tolist(), 'left_px': left, 'right_px': right,
                         'left_xy': (midpoint + normal * left).tolist() if left is not None else None,
                         'right_xy': (midpoint - normal * right).tolist() if right is not None else None,
                         'width_px': width, 'delta_l_px': float(dl),
                         'volume_px3': float(pi * (width / 2)**2 * dl) if width is not None else None})
        position += dl
    out['cross_sections'] = sections
    widths = np.array([s['width_px'] for s in sections if s['width_px'] is not None])
    if valid_widths and len(widths):
        out['volume_px3'] = float(sum(s['volume_px3'] for s in sections))
        median = ndi.median_filter(widths, size=max(5, int(len(widths) * .1) | 1), mode='nearest')
        # Unresolved lateral spikes may be limb roots or a poor centerline. Keep
        # the diagnostic geometry visible; never silently clamp measured widths.
        if (widths > np.maximum(1.4 * median, median + 3)).any():
            warnings.append('LOCAL_WIDTH_SPIKE_POSSIBLE_APPENDAGE')
        asymmetry = np.array([max(s['left_px'], s['right_px']) / max(.5, min(s['left_px'], s['right_px'])) for s in sections])
        if np.mean(asymmetry > 2.5) > .15:
            warnings.append('OFF_CENTER_PATH_OR_APPENDAGE_WIDTH')
        out['width_min_px'] = float(widths.min())
        out['width_median_px'] = float(np.median(widths))
        out['width_max_px'] = float(widths.max())
    else:
        warnings.append('INVALID_PERPENDICULAR_CROSS_SECTIONS')
    if calibration_um_per_px is not None:
        scale = calibration_um_per_px / 1000
        out['length_mm'] = out['length_px'] * scale
        out['volume_mm3'] = out['volume_px3'] * scale**3 if out['volume_px3'] is not None else None
    return finish()
