#!/usr/bin/env python3
"""Draw independent, single-target LabelMe reference polygons over clean images."""

import argparse
import io
import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def normalize_polygon_vertices(points):
    """Remove zero-length edges and an explicit closing copy, never reshape."""
    if not isinstance(points, list) or not 3 <= len(points) <= 20000:
        raise ValueError('Poligon mora imati 3–20000 vrhova.')
    cleaned = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError('Očekuju se vrhovi [x,y].')
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in point):
            raise ValueError('Koordinate moraju biti konačni brojevi.')
        if not cleaned or point != cleaned[-1]:
            cleaned.append(list(point))
    if len(cleaned) > 1 and cleaned[-1] == cleaned[0]:
        cleaned.pop()
    return cleaned


def validate_polygon(points, width, height):
    """Reject invalid geometry instead of inventing, clipping or smoothing it."""
    if not isinstance(points, list) or not 3 <= len(points) <= 20000:
        raise ValueError('Polygon must have 3-20000 vertices')
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError('Expected [x,y] vertices')
        x, y = point
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in point):
            raise ValueError('Coordinates must be finite numbers')
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError('A vertex is outside the annotation image')
    seen = {}
    for index, point in enumerate(points, 1):
        if tuple(point) in seen:
            raise ValueError(f'Vrhovi {seen[tuple(point)]} i {index} imaju iste koordinate. '
                             'To nije samo zatvaranje poligona; pomakni ili ukloni ponovljeni vrh.')
        seen[tuple(point)] = index
    area = abs(sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(points, points[1:]+points[:1]))) / 2
    if area <= 1:
        raise ValueError('Polygon has no meaningful area')

    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])

    def on_segment(a, b, c):
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    def intersects(a, b, c, d):
        values = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
        if values[0]*values[1] < 0 and values[2]*values[3] < 0:
            return True
        return any(abs(v) < 1e-9 and on_segment(p, q, r) for v, p, q, r in (
            (values[0], a, b, c), (values[1], a, b, d), (values[2], c, d, a), (values[3], c, d, b)))

    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            if j == i+1 or (i == 0 and j == n-1):
                continue
            if intersects(points[i], points[(i+1) % n], points[j], points[(j+1) % n]):
                raise ValueError('Polygon intersects itself')
    return area


class AnnotationPack:
    def __init__(self, root):
        self.root = root.resolve()
        self.cases = json.loads((self.root / 'manifest.json').read_text())
        if not self.cases or len({row['case_id'] for row in self.cases}) != len(self.cases):
            raise ValueError('Expected a nonempty pack of unique cases')
        if any(Path(row['case_id']).name != row['case_id'] for row in self.cases):
            raise ValueError('Case IDs cannot contain directories')
        info_path = self.root / 'pack_info.json'
        self.info = json.loads(info_path.read_text()) if info_path.exists() else {}
        if self.info.get('case_count', len(self.cases)) != len(self.cases):
            raise ValueError('Pack count does not match manifest')
        self.options = {
            'title': self.info.get('title', 'Neovisne ručne referentne konture'),
            'annotation_mode': self.info.get('annotation_mode', 'EXPANDED_SOURCE_CROP'),
            'autosave_on_close': self.info.get('autosave_on_close', False),
        }
        self.lock = threading.RLock()
        for row in self.cases:
            if self.info.get('source_allowlist') and row['source_id'] not in self.info['source_allowlist']:
                raise ValueError('Source is outside the permitted annotation sources')
            with Image.open(self.file(row, 'image')) as image:
                if list(image.size) != row['annotation_size']:
                    raise ValueError('Image/manifest dimensions differ')

    def file(self, row, kind):
        if kind not in {'image', 'original', 'context', 'v5', 'v7'}:
            raise ValueError('Unknown image kind')
        path = (self.root / row['files'][kind]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Path outside annotation pack')
        return path

    def annotation_file(self, row):
        # No file is created until the user explicitly saves a manual polygon.
        return self.root / 'annotations' / (row['case_id'] + '.json')

    def load(self, row):
        path = self.annotation_file(row)
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        if payload['imageWidth'] != row['annotation_size'][0] or payload['imageHeight'] != row['annotation_size'][1]:
            raise ValueError('Annotation dimensions do not match the clean image')
        if len(payload['shapes']) != 1 or payload['shapes'][0]['shape_type'] != 'polygon':
            raise ValueError('Exactly one target-body polygon is expected')
        validate_polygon(payload['shapes'][0]['points'], *row['annotation_size'])
        return payload

    def state(self):
        annotations = [self.load(row) for row in self.cases]
        return {'cases': self.cases, 'annotations': annotations, 'options': self.options,
                'completed': sum(a is not None for a in annotations),
                'next_pending': next((i for i, a in enumerate(annotations) if a is None), None)}

    def save(self, index, points, notes=''):
        if type(index) is not int or not 0 <= index < len(self.cases):
            raise ValueError('Invalid case index')
        if not isinstance(notes, str) or len(notes) > 10000:
            raise ValueError('Notes must be text, up to 10000 characters')
        row = self.cases[index]
        original_vertex_count = len(points) if isinstance(points, list) else 0
        points = normalize_polygon_vertices(points)
        area = validate_polygon(points, *row['annotation_size'])
        payload = {
            'version': '5.6.1', 'flags': {'manual_reference': True, 'model_mask_imported': False},
            'shapes': [{'label': 'trunk', 'points': points, 'group_id': None,
                        'shape_type': 'polygon', 'flags': {}, 'description': notes}],
            'imagePath': '../' + row['files']['image'], 'imageData': None,
            'imageWidth': row['annotation_size'][0], 'imageHeight': row['annotation_size'][1],
            'reference_metadata': {
                'source_id': row['source_id'], 'detection_id': row['detection_id'],
                'annotation_box_source_xyxy': row['annotation_box_source_xyxy'],
                'original_crop_box_source_xyxy': row['original_crop_box_source_xyxy'] if 'original_crop_box_source_xyxy' in row else row['annotation_box_source_xyxy'],
                'original_dimensions': row.get('original_dimensions', row['annotation_size']),
                'purpose': 'INDEPENDENT_MANUAL_REFERENCE_ONLY_NOT_ADDED_TO_DATASET',
                'saved_at': datetime.now(timezone.utc).isoformat(timespec='microseconds'),
                'area_px': area,
                'redundant_vertices_removed': original_vertex_count - len(points),
            },
        }
        content = json.dumps(payload, indent=2, ensure_ascii=False)
        with self.lock:
            # Preserve every manual save in addition to the current LabelMe file.
            history = self.root / 'annotation_history' / f"{row['case_id']}__{time.time_ns()}.json"
            with history.open('x', encoding='utf-8') as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            path = self.annotation_file(row)
            temp = path.with_suffix('.json.tmp')
            with temp.open('w', encoding='utf-8') as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        return payload


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, pack):
        self.pack = pack
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def body(self, content, mime='application/json; charset=utf-8', status=200):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def json(self, value, status=200):
        self.body(json.dumps(value, ensure_ascii=False).encode(), status=status)

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            if parsed.path == '/':
                self.body(Path(__file__).with_suffix('.html').read_bytes(), 'text/html; charset=utf-8')
            elif parsed.path == '/api/state':
                self.json(self.server.pack.state())
            elif parsed.path == '/asset':
                query = parse_qs(parsed.query)
                index = int(query['index'][0])
                if not 0 <= index < len(self.server.pack.cases):
                    raise ValueError('Invalid case')
                self.body(self.server.pack.file(self.server.pack.cases[index], query['kind'][0]).read_bytes(), 'image/png')
            else:
                self.json({'error': 'Not found'}, 404)
        except (ValueError, KeyError, IndexError) as exc:
            self.json({'error': str(exc)}, 400)
        except OSError as exc:
            self.json({'error': str(exc)}, 500)

    def do_POST(self):
        if self.path != '/api/annotation':
            self.json({'error': 'Not found'}, 404)
            return
        try:
            size = int(self.headers.get('Content-Length', 0))
            if self.headers.get('Content-Type') != 'application/json' or not 0 < size < 1000000:
                raise ValueError('Expected JSON annotation')
            data = json.loads(self.rfile.read(size))
            self.server.pack.save(data['index'], data['points'], data.get('notes', ''))
            self.json(self.server.pack.state())
        except (ValueError, KeyError, TypeError) as exc:
            self.json({'error': str(exc)}, 400)
        except OSError as exc:
            self.json({'error': str(exc)}, 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, default=ROOT / 'outputs/ruler_v7_reference12_v1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    pack = AnnotationPack(args.pack)
    if args.check:
        print(f"OK: {len(pack.cases)} cases, {pack.state()['completed']} manual polygons saved; no writes")
        return
    server = Server(('127.0.0.1', args.port), pack)
    print(f'Independent polygon annotation: http://127.0.0.1:{server.server_port}', flush=True)
    print(f'New LabelMe files only: {pack.root / "annotations"}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
