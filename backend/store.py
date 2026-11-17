import json
import sqlite3
import uuid
from contextlib import contextmanager
import numpy as np
from .config import DATA, MODEL_ID
from .fusion import quality_band, quality_basis


class Store:
    def __init__(self, root=DATA):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / 'facewatch.sqlite3'
        with self.db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS persons (
              id TEXT PRIMARY KEY, name TEXT NOT NULL, selected INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE IF NOT EXISTS images (
              id TEXT PRIMARY KEY, person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
              path TEXT NOT NULL, embedding BLOB NOT NULL, model TEXT NOT NULL, quality TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS model_embeddings (
              image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
              model TEXT NOT NULL, embedding BLOB NOT NULL, quality TEXT NOT NULL,
              PRIMARY KEY(image_id,model));
            CREATE TABLE IF NOT EXISTS calibrations (
              model TEXT NOT NULL, quality_band TEXT NOT NULL, config TEXT NOT NULL,
              PRIMARY KEY(model,quality_band));
            CREATE TABLE IF NOT EXISTS diagnostics (
              id TEXT PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              scenario TEXT NOT NULL, person_id TEXT, expected_present INTEGER NOT NULL,
              state TEXT NOT NULL, summary TEXT);
            CREATE TABLE IF NOT EXISTS diagnostic_frames (
              run_id TEXT NOT NULL REFERENCES diagnostics(id), frame_id INTEGER NOT NULL,
              elapsed REAL NOT NULL, metrics TEXT NOT NULL, PRIMARY KEY(run_id,frame_id));
            CREATE TABLE IF NOT EXISTS diagnostic_analysis (
              run_id TEXT PRIMARY KEY REFERENCES diagnostics(id), version INTEGER NOT NULL,
              result TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS diagnostic_ground_truth (
              run_id TEXT PRIMARY KEY REFERENCES diagnostics(id), details TEXT NOT NULL,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS video_ground_truth (
              source_hash TEXT PRIMARY KEY, filename TEXT NOT NULL, duration REAL NOT NULL,
              annotations TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS video_evaluations (
              run_id TEXT PRIMARY KEY, source_hash TEXT NOT NULL, result TEXT NOT NULL,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS video_face_ground_truth (
              run_id TEXT NOT NULL, source_hash TEXT NOT NULL, frame_id INTEGER NOT NULL,
              face_index INTEGER NOT NULL, details TEXT NOT NULL,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(run_id,frame_id,face_index));
            CREATE TABLE IF NOT EXISTS model_probes (
              run_id TEXT PRIMARY KEY, model TEXT NOT NULL, source_hash TEXT NOT NULL,
              protocol TEXT NOT NULL, result TEXT NOT NULL,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS video_measurements (
              run_id TEXT NOT NULL, frame_id INTEGER NOT NULL, timestamp REAL NOT NULL,
              metrics TEXT NOT NULL, PRIMARY KEY(run_id,frame_id));
            ''')
            if 'context' not in {r['name'] for r in db.execute('PRAGMA table_info(diagnostics)')}:
                db.execute('ALTER TABLE diagnostics ADD COLUMN context TEXT')

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def people(self):
        with self.db() as db:
            people = [dict(r) for r in db.execute('SELECT * FROM persons ORDER BY name')]
            for p in people:
                p['selected'] = bool(p['selected'])
                p['images'] = [dict(r) for r in db.execute(
                    'SELECT id,model,quality FROM images WHERE person_id=? ORDER BY rowid', (p['id'],))]
                for i in p['images']:
                    i['url'] = f"/api/images/{i['id']}"
                    i['quality'] = json.loads(i['quality'])
                    i['models'] = [r[0] for r in db.execute('SELECT model FROM model_embeddings WHERE image_id=?', (i['id'],))]
            return people

    def exists(self, person_id):
        with self.db() as db:
            return db.execute('SELECT 1 FROM persons WHERE id=?', (person_id,)).fetchone() is not None

    def add(self, name, samples, person_id=None):
        new = person_id is None
        person_id = person_id or uuid.uuid4().hex
        written = []
        try:
            with self.db() as db:
                if new:
                    db.execute('INSERT INTO persons(id,name) VALUES (?,?)', (person_id, name))
                elif not db.execute('SELECT 1 FROM persons WHERE id=?', (person_id,)).fetchone():
                    raise ValueError('Không tìm thấy hồ sơ.')
                for image, embedding, quality in samples:
                    image_id = uuid.uuid4().hex
                    path = self.root / 'people' / f'{image_id}.jpg'
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(image)
                    written.append(path)
                    derived = embedding if isinstance(embedding, dict) else {}
                    legacy = np.empty(0, np.float32) if derived else np.asarray(embedding, np.float32)
                    db.execute('INSERT INTO images VALUES (?,?,?,?,?,?)',
                               (image_id, person_id, str(path), legacy.tobytes(),
                                MODEL_ID, json.dumps(quality)))
                    for model, vector in derived.items():
                        db.execute('INSERT INTO model_embeddings VALUES (?,?,?,?)',
                            (image_id, model, np.asarray(vector, np.float32).tobytes(), json.dumps(quality)))
            return person_id
        except Exception:
            for path in written:
                path.unlink(missing_ok=True)
            raise

    def update(self, person_id, name=None, selected=None):
        with self.db() as db:
            if not db.execute('SELECT 1 FROM persons WHERE id=?', (person_id,)).fetchone():
                raise ValueError('Không tìm thấy hồ sơ.')
            if name is not None:
                db.execute('UPDATE persons SET name=? WHERE id=?', (name, person_id))
            if selected is not None:
                db.execute('UPDATE persons SET selected=? WHERE id=?', (int(selected), person_id))

    def image_path(self, image_id):
        with self.db() as db:
            row = db.execute('SELECT path FROM images WHERE id=?', (image_id,)).fetchone()
            return row['path'] if row else None

    def delete_image(self, person_id, image_id):
        with self.db() as db:
            row = db.execute('SELECT path FROM images WHERE id=? AND person_id=?', (image_id, person_id)).fetchone()
            if not row:
                raise ValueError('Không tìm thấy ảnh.')
            if db.execute('SELECT COUNT(*) FROM images WHERE person_id=?', (person_id,)).fetchone()[0] <= 1:
                raise ValueError('Hồ sơ cần ít nhất một ảnh. Hãy thêm ảnh trước hoặc xóa cả hồ sơ.')
            db.execute('DELETE FROM images WHERE id=?', (image_id,))
        from pathlib import Path
        Path(row['path']).unlink(missing_ok=True)

    def delete(self, person_id):
        with self.db() as db:
            paths = [r['path'] for r in db.execute('SELECT path FROM images WHERE person_id=?', (person_id,))]
            db.execute('DELETE FROM persons WHERE id=?', (person_id,))
        from pathlib import Path
        for path in paths:
            Path(path).unlink(missing_ok=True)

    def gallery(self, include_unselected=False):
        with self.db() as db:
            return [(r['person_id'], r['name'], np.frombuffer(r['embedding'], np.float32).copy())
                    for r in db.execute('''SELECT images.*, persons.name FROM images JOIN persons
                      ON persons.id=images.person_id WHERE (persons.selected=1 OR ?=1) AND images.model=?''',
                      (int(include_unselected), MODEL_ID))]

    def selected_ids(self):
        with self.db() as db:
            return {r['id'] for r in db.execute('SELECT id FROM persons WHERE selected=1')}

    def gallery_model(self, version):
        with self.db() as db:
            return [(r['person_id'], r['name'], np.frombuffer(r['embedding'], np.float32).copy())
                    for r in db.execute('''SELECT p.id person_id,p.name,e.embedding FROM model_embeddings e
                    JOIN images i ON i.id=e.image_id JOIN persons p ON p.id=i.person_id WHERE e.model=?''', (version,))]

    def reindex(self, engine):
        """Only add derived embeddings missing for the current model version. Originals are immutable."""
        errors = []
        with self.db() as db:
            rows = [dict(r) for r in db.execute('SELECT id,path FROM images')]
        for row in rows:
            with self.db() as db:
                present = {r[0] for r in db.execute('SELECT model FROM model_embeddings WHERE image_id=?', (row['id'],))}
            needed = {v for k,v in engine.versions.items() if k in ('lvface','antelopev2','petalface')} - present
            if not needed:
                continue
            try:
                from pathlib import Path
                _, vectors, quality = engine.register(Path(row['path']).read_bytes(), self.settings())
                with self.db() as db:
                    for version, vector in vectors.items():
                        if version in needed:
                            db.execute('INSERT OR IGNORE INTO model_embeddings VALUES (?,?,?,?)',
                                (row['id'], version, vector.astype(np.float32).tobytes(), json.dumps(quality)))
            except Exception as exc:
                errors.append({'image_id': row['id'], 'error': str(exc)})
        return errors

    def calibration(self, version, face_pixels, alignment_band='regular', face_height=None):
        band = quality_band(face_pixels,alignment_band,face_height)
        with self.db() as db:
            row = db.execute('SELECT config FROM calibrations WHERE model=? AND quality_band=?', (version,band)).fetchone()
            generation = self.label_generation(db)
            config = json.loads(row[0]) if row else None
            if config and config.get('identity_gates'):
                gallery_images = sorted(r[0] for r in db.execute(
                    'SELECT image_id FROM model_embeddings WHERE model=?', (version,)))
                enrollment_images = sorted(r[0] for r in db.execute('SELECT id FROM images'))
                if gallery_images != config.get('identity_gates_gallery_images') or enrollment_images != gallery_images:
                    config = dict(config)
                    config.pop('identity_gates', None)
        if not config or config.get('enabled') is False or config.get('label_generation', generation) != generation:
            return None
        if alignment_band!='regular' and config.get('quality_basis')!=quality_basis(alignment_band):
            return None
        minimum=config.get('minimum_native_face_pixels')
        maximum=config.get('maximum_native_face_pixels')
        if (minimum is not None and face_pixels<minimum) or (maximum is not None and face_pixels>maximum):
            return None
        return config

    @staticmethod
    def label_generation(db):
        row = db.execute("SELECT value FROM settings WHERE key='calibration_generation'").fetchone()
        return json.loads(row[0]) if row else 0

    @staticmethod
    def labels_changed(db):
        db.execute('UPDATE diagnostic_analysis SET version=0')
        generation = Store.label_generation(db)+1
        db.execute("INSERT OR REPLACE INTO settings VALUES ('calibration_generation',?)", (json.dumps(generation),))

    def calibrate(self, version, band, config):
        with self.db() as db:
            db.execute('INSERT OR REPLACE INTO calibrations VALUES (?,?,?)', (version,band,json.dumps(config)))

    def settings(self, versions=None):
        result = dict(similarity=0.62, margin=0.08, confirm_frames=3, absence_seconds=3.0,
                      calibrated=False, min_face_pixels=28, min_inference_face_pixels=14, min_sharpness=18.0)
        with self.db() as db:
            for r in db.execute('SELECT * FROM settings'):
                result[r['key']] = json.loads(r['value'])
            generation = self.label_generation(db)
            coverage, profiles = {}, []
            active = set(versions.values()) if versions is not None else None
            for row in db.execute('SELECT * FROM calibrations'):
                config = json.loads(row['config'])
                if (active is not None and row['model'] not in active) or config.get('enabled') is False or config.get('label_generation',generation) != generation:
                    continue
                if row['quality_band'].endswith('_high_residual') and config.get('quality_basis')!=quality_basis('high_residual'):
                    continue
                key = row['model'].split(':')[0]
                coverage.setdefault(row['quality_band'],[]).append(key)
                profiles.append(dict(model=key, version=row['model'], band=row['quality_band'], **config))
            result['calibration_coverage'] = coverage
            result['calibrated'] = any(len(set(models)) >= 2 for models in coverage.values())
            result['calibration_status'] = 'preliminary' if result['calibrated'] else 'unavailable'
            result['calibration_profiles'] = profiles
        return result

    def save_settings(self, values):
        with self.db() as db:
            for key, value in values.items():
                db.execute('INSERT OR REPLACE INTO settings VALUES (?,?)', (key, json.dumps(value)))
