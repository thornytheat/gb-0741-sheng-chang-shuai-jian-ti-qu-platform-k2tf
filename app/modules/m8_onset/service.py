from app.modules.m2_calibration.service import calibration
from app.modules.m8_onset.alignment import align
from app.modules.m8_onset.detection import detect
from app.reasons import reason


def onsets(signals):
    return {hz: detect(signal) for hz, signal in signals.items()}


def band_entry(signal, onset):
    prefix = onset.index if onset.index is not None else 0
    return {'band_hz': signal.hz, 'status': 'detected' if onset.index is not None else 'none',
            'onset_index': onset.index,
            'onset_s': onset.index / signal.rate if onset.index is not None else None,
            'onset_line': signal.line_numbers[onset.index] if onset.index is not None else None,
            'finding': None if onset.index is not None else reason('ONSET_NONE'),
            'peak_index': signal.peak_index, 'peak_s': signal.peak_index / signal.rate,
            'peak_line': signal.line_numbers[signal.peak_index],
            'peak_is_onset': signal.peak_index == onset.index,
            'prefix': {'samples': prefix, 'from_line': signal.line_numbers[0] if prefix else None,
                       'to_line': signal.line_numbers[prefix - 1] if prefix else None,
                       **reason('LEADING_SILENCE')},
            'spikes': [{**event, **reason(event['code'])} for event in onset.spikes],
            'floor_power_pa2': onset.floor_power, 'threshold_power_pa2': onset.threshold_power}


def onset_report(data):
    signals, _ = calibration(data)
    detected = onsets(signals)
    bands = [band_entry(signals[hz], detected[hz]) for hz in sorted(signals)]
    return {'bands': bands, **align(bands),
            'isolated_bands': [row['band_hz'] for row in data.rejected_bands]}
