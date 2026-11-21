import pytest

from app.config import SAMPLES
from app.evidence import canonical, sha256
from app.modules.m1_intake.reader import read
from app.modules.m6_reconcile.service import evaluate
from app.modules.m8_onset.alignment import align
from app.modules.m8_onset.detection import detect


def decaying(arrival, n=500, noise=0.002, peak=1.0, ratio=0.99):
    values = [noise] * n
    for k in range(n - arrival):
        values[arrival + k] = (1 if k % 2 == 0 else -1) * (peak * ratio ** k + noise)
    return values


def test_leading_silence_stripped_and_prefix_kept(signal):
    onset = detect(signal(decaying(60), peak=60))
    assert onset.index == 60 and onset.spikes == ()


def test_isolated_spike_before_arrival_is_not_onset(signal):
    values = decaying(40)
    values[10] = 1.5  # 电磁尖峰早于真实到达
    onset = detect(signal(values, peak=10))
    assert onset.index == 40
    assert [(event['index'], event['line'], event['samples']) for event in onset.spikes] == [(10, 12, 1)]


def test_spike_without_sustained_arrival_has_no_onset(signal):
    values = [0.002] * 200
    values[50] = 2.0
    onset = detect(signal(values, peak=50))
    assert onset.index is None and len(onset.spikes) == 1


def test_stronger_late_reflection_does_not_move_onset(signal):
    values = decaying(30, peak=0.5)
    for k in range(200):
        values[100 + k] += ((-1) ** k) * 1.3 * 0.99 ** k
    onset = detect(signal(values, peak=100))
    assert onset.index == 30


def test_leading_segment_louder_than_tail_is_not_arrival(signal):
    onset = detect(signal([0.002] * 6 + [(-1) ** k * 0.98 ** k for k in range(494)], peak=6))
    assert onset.index == 6


def test_zero_energy_and_digital_silence_prefix(signal):
    assert detect(signal([0] * 100)).index is None
    values = [0.0] * 25 + [(-1) ** k * 0.95 ** k for k in range(175)]
    assert detect(signal(values, peak=25)).index == 25


def band(hz, onset_s):
    return {'band_hz': hz, 'onset_index': None if onset_s is None else round(onset_s * 1000), 'onset_s': onset_s}


def test_consensus_is_median_and_divergence_flagged():
    bands = [band(250, .025), band(500, .025), band(1000, .025), band(2000, .085)]
    result = align(bands)
    assert result['status'] == 'RETEST' and result['divergent_bands'] == [2000]
    assert result['point_onset_s'] == pytest.approx(.025) and result['reason']['code'] == 'POINT_RETEST'
    assert bands[3]['aligned'] is False and bands[3]['finding']['code'] == 'ONSET_DIVERGENT'
    assert bands[3]['deviation_s'] == pytest.approx(.06)


def test_deviation_at_exact_tolerance_stays_aligned():
    aligned = align([band(250, .025), band(500, .029)])
    divergent = align([band(250, .025), band(500, .0295)])
    assert aligned['status'] == 'DELIVERABLE' and divergent['status'] == 'RETEST'


def test_point_without_detected_onsets_is_not_blocked():
    result = align([band(250, None), band(500, None)])
    assert (result['status'], result['point_onset_s'], result['checked_bands']) == ('DELIVERABLE', None, 0)


@pytest.fixture(scope='module')
def retest():
    return evaluate(read(SAMPLES['retest'].read_bytes()))


def test_retest_point_sent_back_for_remeasurement(retest):
    onset = retest['onset']
    assert onset['status'] == 'RETEST' and onset['divergent_bands'] == [2000]
    assert onset['reason']['code'] == 'POINT_RETEST' and onset['point_onset_s'] == pytest.approx(.025)
    assert retest['reconcile']['status'] == 'MATCHED' and retest['reconcile']['totals']['fitted'] == 6


def test_divergent_band_keeps_own_onset_and_prefix_as_evidence(retest):
    row = next(band for band in retest['onset']['bands'] if band['band_hz'] == 2000)
    assert row['aligned'] is False and row['finding']['code'] == 'ONSET_DIVERGENT'
    assert row['deviation_s'] == pytest.approx(.06) and row['onset_line'] == 3387
    assert row['prefix']['samples'] == 85 and row['prefix']['code'] == 'LEADING_SILENCE'
    assert (row['prefix']['from_line'], row['prefix']['to_line']) == (3302, 3386)


def test_spike_band_reports_rejected_transient(retest):
    row = next(band for band in retest['onset']['bands'] if band['band_hz'] == 500)
    assert row['onset_index'] == 25 and row['peak_index'] == 3 and not row['peak_is_onset']
    assert [(event['index'], event['code']) for event in row['spikes']] == [(3, 'SPIKE_REJECTED')]
    assert row['spikes'][0]['peak_pa'] == pytest.approx(1.5)


def test_reflection_peak_is_not_adopted_as_onset(retest):
    row = next(band for band in retest['onset']['bands'] if band['band_hz'] == 1000)
    assert row['onset_index'] == 25 and row['peak_index'] == 80 and not row['peak_is_onset']


def test_decay_curve_starts_at_adopted_onset(retest):
    curve = next(band for band in retest['decay']['bands'] if band['band_hz'] == 250)
    assert curve['onset_index'] == 25 and curve['dropped_prefix_samples'] == 25
    assert curve['full_points'] == 1075 and curve['points'][0]['source_line'] == curve['onset_line']


def test_hall_point_is_deliverable(report):
    onset = report['onset']
    assert onset['status'] == 'DELIVERABLE' and onset['divergent_bands'] == []
    assert onset['point_onset_s'] == pytest.approx(.006)


def test_onset_api_returns_verdict_and_prefix_evidence(client, source):
    reply = client.get(f'/api/onset/{source}')
    assert reply.status_code == 200
    body = reply.json()
    assert body['status'] == 'DELIVERABLE'
    row = next(band for band in body['bands'] if band['band_hz'] == 500)
    assert row['onset_index'] == 6 and row['prefix']['samples'] == 6


def test_archive_records_delivery_verdict(client):
    run = client.post('/api/runs', json={'sample': 'retest'}).json()
    assert run['result']['onset']['status'] == 'RETEST'
    exported = client.get(f"/api/runs/{run['id']}/export").json()
    assert exported['result']['onset']['divergent_bands'] == [2000]
    retry = client.post(f"/api/runs/{run['id']}/rerun").json()
    assert retry['reused'] and len(client.get('/api/runs').json()['runs']) == 1


def test_onset_rule_change_invalidates_old_archive_but_keeps_it_readable(client):
    old = client.post('/api/runs', json={'sample': 'reference'}).json()
    source_id = old['manifest']['source_id']
    stale_id = sha256(canonical({'source_id': source_id, 'version': 'rt-schroeder-1'}))
    with client.app.state.store.connect() as connection:
        connection.execute("UPDATE runs SET id=?, version='rt-schroeder-1' WHERE id=?", (stale_id, old['id']))
    rerun = client.post(f'/api/runs/{stale_id}/rerun')
    assert (rerun.status_code, rerun.json()['code']) == (409, 'VERSION_CHANGED')
    assert client.get(f'/api/runs/{stale_id}').status_code == 200
    fresh = client.post('/api/runs', json={'source_id': source_id}).json()
    assert fresh['id'] != stale_id and not fresh['reused']
    again = client.post(f"/api/runs/{fresh['id']}/rerun").json()
    assert again['reused'] and again['id'] == fresh['id']
    assert len(client.get('/api/runs').json()['runs']) == 2
