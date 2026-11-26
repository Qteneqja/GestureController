"""Continue EXP-003 from its saved epoch-16 model on a local runtime.

The original run and first continuation are read-only inputs. The new
continuation writes to distinct paths in the mounted project folder.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import keras
import numpy as np
import tensorflow as tf
import yaml

PROJECT = Path('/content/GestureController')
CODE = PROJECT / 'checkpoints/local_runtime_code'
FRAMES = Path('/content/jester_frames')
ROOT = PROJECT / 'experiments/cnn_gru_baseline'
CHECKPOINTS = PROJECT / 'checkpoints'
FIRST = ROOT / 'EXP-003-retry-01'
PREVIOUS = ROOT / 'EXP-003-retry-01-continuation-01'
RUN_ID = 'EXP-003-retry-01-continuation-02-local'
OUTPUT = ROOT / RUN_ID
LATEST = CHECKPOINTS / 'cnn_gru_EXP-003-retry-01-continuation-01_last.keras'
OLD_BEST = CHECKPOINTS / 'cnn_gru_EXP-003-retry-01-continuation-01_best.keras'
NEW_BEST = CHECKPOINTS / f'cnn_gru_{RUN_ID}_best.keras'
NEW_LAST = CHECKPOINTS / f'cnn_gru_{RUN_ID}_last.keras'


def main():
    if OUTPUT.exists() or NEW_BEST.exists() or NEW_LAST.exists():
        raise RuntimeError('Local continuation files already exist; refusing to overwrite.')
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('No GPU detected; check the local runtime.')

    with (FIRST / 'provenance.json').open(encoding='utf-8') as f:
        provenance = json.load(f)['original_provenance']
    rows = []
    for directory in (FIRST, PREVIOUS):
        with (directory / 'history.csv').open(newline='', encoding='utf-8') as f:
            rows.extend(csv.DictReader(f))
    if [int(row['epoch']) for row in rows] != list(range(16)):
        raise RuntimeError('Saved histories do not contain exactly epochs 1–16.')
    best_index = min(range(len(rows)), key=lambda i: float(rows[i]['val_loss']))
    best_loss = float(rows[best_index]['val_loss'])
    wait = len(rows) - best_index - 1

    data = provenance['data_settings'].copy()
    data['frames_folder'] = str(FRAMES)
    for split in ('train', 'validation'):
        split_file = CODE / data[f'{split}_csv']
        # Windows checkout/archive may use CRLF; the Colab hashes use LF.
        actual = hashlib.sha256(split_file.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != provenance['split_sha256'][split]:
            raise RuntimeError(f'{split} split does not match the original experiment.')
        with split_file.open(newline='', encoding='utf-8') as f:
            missing = [r['video_id'] for r in csv.DictReader(f)
                       if not (FRAMES / r['video_id']).is_dir()]
        if missing:
            raise RuntimeError(f'{split} is missing {len(missing)} video folders.')

    os.chdir(CODE)
    sys.path.insert(0, str(CODE))
    from src.gesture_recognition.training.train_cnn import make_real_datasets, save_curves

    settings = provenance['training_settings']
    keras.utils.set_random_seed(settings['seed'])
    config_path = Path('/tmp/gesture-exp003-local-data.yaml')
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    train, validation = make_real_datasets(
        str(config_path), settings['train_limit'], settings['validation_limit']
    )
    model = keras.models.load_model(LATEST)
    best_model = keras.models.load_model(OLD_BEST)
    if model.get_layer('gru').units != 64 or int(model.optimizer.iterations.numpy()) != 16000:
        raise RuntimeError('The latest checkpoint is not the expected epoch-16 model.')
    expected_best_steps = (best_index + 1) * 1000
    if int(best_model.optimizer.iterations.numpy()) != expected_best_steps:
        raise RuntimeError('The saved best checkpoint does not match the best history epoch.')
    if '--check' in sys.argv:
        clips, labels = next(iter(train))
        probabilities = np.asarray(model(clips, training=False))
        if tuple(clips.shape) != (8, 16, 112, 112, 3):
            raise RuntimeError(f'Unexpected batch shape: {clips.shape}')
        if probabilities.shape != (8, 7) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError('The checkpoint produced invalid predictions.')
        print(f'Preflight passed: {len(rows)} completed epochs; best epoch {best_index + 1}; '
              f'patience count {wait}; batch shape {clips.shape}; prediction shape {probabilities.shape}.')
        return
    best_weights = best_model.get_weights()
    del best_model

    class ContinuedEarlyStopping(keras.callbacks.EarlyStopping):
        def on_train_begin(self, logs=None):
            super().on_train_begin(logs)
            self.best = best_loss
            self.wait = wait
            self.best_weights = best_weights
            self.best_epoch = best_index

    OUTPUT.mkdir(parents=True)
    info = {
        'continuation_id': RUN_ID,
        'resumed_from': str(LATEST),
        'original_best_checkpoint': str(OLD_BEST),
        'completed_epochs_before_resume': len(rows),
        'original_best_epoch': best_index + 1,
        'original_best_validation_loss': best_loss,
        'epochs_without_improvement': wait,
        'optimizer_steps_before_resume': 16000,
        'hardware': [d.name for d in tf.config.list_physical_devices('GPU')],
        'tensorflow_version': tf.__version__,
        'keras_version': keras.__version__,
        'numpy_version': np.__version__,
        'note': 'Random augmentation and shuffle state restart in this local continuation.',
        'original_provenance': provenance,
    }
    (OUTPUT / 'continuation_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    callbacks = [
        keras.callbacks.ModelCheckpoint(NEW_BEST, monitor='val_loss', save_best_only=True,
                                        initial_value_threshold=best_loss),
        keras.callbacks.ModelCheckpoint(NEW_LAST),
        ContinuedEarlyStopping(monitor='val_loss', patience=settings['early_stopping_patience'],
                               restore_best_weights=True),
        keras.callbacks.CSVLogger(OUTPUT / 'history.csv'),
    ]
    print(f'Continuing at epoch {len(rows)+1}; previous best epoch {best_index+1}, '
          f'validation loss {best_loss:.4f}; patience count {wait}.', flush=True)
    start = time.monotonic()
    history = model.fit(train, validation_data=validation, initial_epoch=len(rows),
                        epochs=settings['epochs'], callbacks=callbacks, shuffle=False)
    if history.history:
        save_curves(history.history, OUTPUT / 'curves.png')
    info['continuation_minutes'] = (time.monotonic() - start) / 60
    info['epochs_run_in_continuation'] = len(history.history.get('loss', []))
    info['new_best_checkpoint_created'] = NEW_BEST.exists()
    (OUTPUT / 'continuation_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'Continuation finished. Results: {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
