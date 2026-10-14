"""Amap walking routes with resumable, directed, coordinate-specific SQLite cache."""
import hashlib
import json
import math
import os
import sqlite3
import time
from datetime import datetime, timezone
import requests

ENDPOINT = 'https://restapi.amap.com/v3/direction/walking'


def parse_response(payload):
    """Distinguish a successful no-route result from an API or malformed-data error."""
    if str(payload.get('status')) != '1':
        # Do not print URLs or arbitrary response content: URLs contain the key.
        raise RuntimeError('Amap rejected request; error code ' + str(payload.get('infocode', 'unknown')))
    paths = payload.get('route', {}).get('paths', [])
    if not paths:
        return {'status': 'no_route', 'duration_s': None, 'distance_m': None}
    candidates = []
    for p in paths:
        duration, distance = float(p['duration']), float(p['distance'])
        if not math.isfinite(duration) or not math.isfinite(distance) or duration < 0 or distance < 0:
            raise ValueError('Invalid route time/distance')
        candidates.append((duration, distance))
    duration, distance = min(candidates)
    return {'status': 'ok', 'duration_s': duration, 'distance_m': distance}


class AmapRouter:
    def __init__(self, cache_path, key=None, offline=False, interval=0.35):
        self.key = key or os.environ.get('AMAP_API_KEY')
        self.offline = offline
        self.interval = interval
        self.db = sqlite3.connect(cache_path)
        self.db.execute('CREATE TABLE IF NOT EXISTS routes (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.session = requests.Session()

    def close(self):
        self.session.close()
        self.db.close()

    def route(self, origin, destination):
        """Return the fastest reported directed route, reusing a matching cache entry."""
        coords = [float(v) for v in (*origin, *destination)]
        if not all(math.isfinite(v) for v in coords) or not all(-180 <= coords[i] <= 180 and -90 <= coords[i+1] <= 90 for i in [0,2]):
            raise ValueError('Invalid GCJ-02 longitude/latitude')
        o, d = ','.join(f'{v:.6f}' for v in origin), ','.join(f'{v:.6f}' for v in destination)
        # Origin and destination order matters; the API key is never stored.
        cache_key = hashlib.sha256(f'{ENDPOINT}|GCJ02|{o}|{d}|fastest'.encode()).hexdigest()
        row = self.db.execute('SELECT value FROM routes WHERE key=?', (cache_key,)).fetchone()
        if row:
            return json.loads(row[0])
        if self.offline:
            raise RuntimeError('Route missing from cache; offline mode cannot infer reachability')
        if not self.key:
            raise RuntimeError('Set AMAP_API_KEY in the environment')
        for attempt in range(3):
            time.sleep(self.interval)
            try:
                response = self.session.get(ENDPOINT, params={'key': self.key, 'origin': o, 'destination': d, 'output': 'JSON'}, timeout=30)
                response.raise_for_status()
                result = parse_response(response.json())
                break
            except requests.RequestException:
                if attempt == 2:
                    raise RuntimeError('Amap transport failed after 3 attempts; no Euclidean fallback used') from None
                time.sleep(2 ** attempt)
        result.update({'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'endpoint': ENDPOINT, 'crs': 'GCJ-02', 'origin': o, 'destination': d})
        self.db.execute('INSERT OR REPLACE INTO routes VALUES (?, ?)', (cache_key, json.dumps(result)))
        self.db.commit()
        return result
