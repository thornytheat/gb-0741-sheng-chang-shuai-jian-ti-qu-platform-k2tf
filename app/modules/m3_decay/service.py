from dataclasses import replace
from math import ceil

from app.modules.m2_calibration.service import calibration
from app.modules.m3_decay.integral import integrate, start_index
from app.modules.m8_onset.service import onsets


def render_curve(curve, limit=900):
    signal = curve.signal
    start = start_index(signal)
    stride = max(1, ceil(len(curve.db) / limit))
    indices = list(range(0, len(curve.db), stride))
    if indices[-1] != len(curve.db) - 1:
        indices.append(len(curve.db) - 1)
    return {'band_hz': signal.hz, 'sample_rate_hz': signal.rate, 'peak_index': signal.peak_index,
            'onset_index': start, 'onset_line': signal.line_numbers[start], 'dropped_prefix_samples': start,
            'tail_start_s': curve.tail_from / signal.rate, 'initial_energy_pa2_s': curve.first_energy / signal.rate,
            'noise_power_pa2': curve.noise_power, 'floor_db': curve.floor_db, 'error': curve.error,
            'full_points': len(curve.db), 'points': [{'t': index / signal.rate, 'db': curve.db[index],
                         'source_line': signal.line_numbers[start + index]} for index in indices]}


def decay(data):
    signals, calibrated = calibration(data)
    detected = onsets(signals)
    resolved = {hz: replace(signal, onset_index=detected[hz].index if detected[hz].index is not None else 0)
                for hz, signal in signals.items()}
    curves = {hz: integrate(signal) for hz, signal in resolved.items()}
    return curves, calibrated, {'bands': [render_curve(curve) for curve in curves.values()]}
