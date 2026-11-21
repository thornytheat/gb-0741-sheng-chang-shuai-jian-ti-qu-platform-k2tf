from dataclasses import dataclass
from statistics import median

from app.modules.m3_decay.integral import tail_power

MARGIN_DB = 10
COARSE_DB = 40
BLOCK_S = 0.005
SUSTAIN_BLOCKS = 2


@dataclass(frozen=True)
class Onset:
    index: int | None
    floor_power: float
    threshold_power: float
    block: int
    spikes: tuple[dict, ...]


def loud(power, threshold):
    return power > 0 and power >= threshold


def prefix_sums(power):
    sums = [0.0]
    for value in power:
        sums.append(sums[-1] + value)
    return sums


def sustained(power, sums, block, threshold):
    for start in range(len(power) - SUSTAIN_BLOCKS * block + 1):
        first = (sums[start + block] - sums[start]) / block
        second = (sums[start + SUSTAIN_BLOCKS * block] - sums[start + block]) / block
        if loud(first, threshold) and loud(second, threshold):
            return next(index for index in range(start, start + block) if loud(power[index], threshold))
    return None


def find_onset(power, block, tail_noise):
    count = len(power)
    if count < SUSTAIN_BLOCKS * block:
        return None, tail_noise
    sums = prefix_sums(power)
    blocks = [(sums[start + block] - sums[start]) / block for start in range(0, count - block + 1, block)]
    strongest = max(blocks)
    if strongest <= 0:
        return None, tail_noise
    coarse = sustained(blocks, prefix_sums(blocks), 1, strongest / 10 ** (COARSE_DB / 10))
    if coarse is None:
        return None, tail_noise
    floor = median(blocks[:coarse]) if coarse else tail_noise
    return sustained(power, sums, block, floor * 10 ** (MARGIN_DB / 10)), floor


def rejected_spikes(power, line_numbers, pa, stop, threshold):
    events = []
    index = 0
    while index < stop:
        if not loud(power[index], threshold):
            index += 1
            continue
        run = index
        while run + 1 < stop and loud(power[run + 1], threshold):
            run += 1
        peak = max(range(index, run + 1), key=lambda at: abs(pa[at]))
        events.append({'index': index, 'line': line_numbers[index], 'samples': run - index + 1,
                       'peak_pa': abs(pa[peak]), 'peak_line': line_numbers[peak], 'code': 'SPIKE_REJECTED'})
        index = run + 1
    return tuple(events)


def detect(signal):
    power = tuple(value * value for value in signal.pa)
    _, tail_noise = tail_power(power)
    block = max(4, round(signal.rate * BLOCK_S))
    index, floor = find_onset(power, block, tail_noise)
    threshold = floor * 10 ** (MARGIN_DB / 10)
    spikes = rejected_spikes(power, signal.line_numbers, signal.pa,
                             index if index is not None else len(power), threshold)
    return Onset(index, floor, threshold, block, spikes)
