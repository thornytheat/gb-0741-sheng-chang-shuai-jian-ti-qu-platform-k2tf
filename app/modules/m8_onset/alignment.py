from math import isclose
from statistics import median

from app.reasons import reason

ALIGN_TOLERANCE_S = 0.002


def align(bands):
    detected = [band for band in bands if band['onset_index'] is not None]
    consensus = median(band['onset_s'] for band in detected) if detected else None
    divergent = []
    for band in bands:
        if band['onset_index'] is None:
            band['deviation_s'] = None
            band['aligned'] = None
            continue
        deviation = band['onset_s'] - consensus
        band['deviation_s'] = deviation
        distance = abs(deviation)
        band['aligned'] = (distance <= ALIGN_TOLERANCE_S
                           or isclose(distance, ALIGN_TOLERANCE_S, rel_tol=0, abs_tol=1e-12))
        if not band['aligned']:
            band['finding'] = reason('ONSET_DIVERGENT')
            divergent.append(band['band_hz'])
    status = 'RETEST' if divergent else 'DELIVERABLE'
    return {'status': status, 'reason': reason('POINT_RETEST') if divergent else None,
            'point_onset_s': consensus, 'tolerance_s': ALIGN_TOLERANCE_S,
            'checked_bands': len(detected), 'divergent_bands': divergent}
