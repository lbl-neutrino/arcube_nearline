#!/usr/bin/env python3
"""
Whole-tile-trigger (WTT) tagging for one 2x2 flow file.

A tile in an event is tagged as WTT when:
  * the event has > min_hits_per_event hits, the tile's IO group
    > min_hits_per_io_group packets and the tile > min_hits_per_tile packets,
    from >= min_chips_per_tile distinct chips;
  * its packet-time histogram in bin_size_us bins has an adjacent-bin step of
    more than min_bin_difference packets AND a bin-to-bin ratio beyond
    max_bin_ratio (tested at two bin phases, 0 and half a bin);
  * the 1 us histogram centred on the busiest 1 us bin within +-10 us of the
    largest step is peaked: excess kurtosis > kurt20_cut in a +-20 us window
    OR > kurt10_cut in a +-10 us window. 

Hits are charge/calib_prompt_hits; the time, tile and chip of each hit come
from its packet (one packet per prompt hit).
"""

import math

import numpy as np

HITS_DSET = 'calib_prompt_hits'

DEFAULT_CUTS = dict(min_hits_per_event=2000, min_hits_per_io_group=1000,
                    min_hits_per_tile=1000, min_chips_per_tile=95,
                    bin_size_us=10, min_bin_difference=500, max_bin_ratio=0.5,
                    kurt20_cut=0.0, kurt10_cut=-0.5)


def io_channel_to_tile(io_channel):
    """Map io_channel -> tile id (1-8). Works on arrays."""
    # cast first: io_channel is uint8 in flow files, and uint8(0) - 1 wraps to 255
    io_channel = np.asarray(io_channel, dtype=np.int64)
    return (io_channel - 1 - ((io_channel - 1) % 4)) // 4 + 1


def ref_children(region, ref, parent_ids):
    """(parent_id, child_id) ref rows belonging to parent_ids, via ref_region."""
    parent_ids = np.atleast_1d(np.asarray(parent_ids))
    if parent_ids.size == 0:
        return np.empty((0, 2), dtype=int)
    sel = region[np.sort(parent_ids)]
    starts = np.asarray(sel['start'], dtype=np.int64)
    lengths = np.clip(np.asarray(sel['stop'], dtype=np.int64) - starts, 0, None)
    total = int(lengths.sum())
    if total == 0:
        return np.empty((0, 2), dtype=int)
    cum = np.cumsum(lengths) - lengths
    rows = np.unique(np.repeat(starts, lengths) + (np.arange(total) - np.repeat(cum, lengths)))
    pairs = np.asarray(ref[rows])
    return pairs[np.isin(pairs[:, 0], parent_ids)]


def beam_event_ids(f):
    """Event ids with a beam external trigger (ext_trig io group 5), or None."""
    try:
        exttrigs = f['charge/ext_trigs/data']
        ref = np.asarray(f['charge/events/ref/charge/ext_trigs/ref'])
    except (KeyError, OSError):
        return None
    names = exttrigs.dtype.names
    iog_field = 'iogroup' if 'iogroup' in names else ('io_group' if 'io_group' in names else None)
    if iog_field is None or ref.size == 0:
        return None
    trig_ids = np.asarray(exttrigs['id']) if 'id' in names else np.arange(len(exttrigs))
    beam_trigs = trig_ids[np.asarray(exttrigs[iog_field]) == 5]
    return np.unique(ref[np.isin(ref[:, 1], beam_trigs), 0])


def _coarse_hists(ts, bin_size_us):
    """(edges, hist) at bin phases 0 and half a bin."""
    out = []
    for phase in (0.0, bin_size_us / 2.0):
        lo = ts.min() - phase
        n_bins = int(math.ceil((ts.max() + bin_size_us - lo) / bin_size_us))
        if n_bins < 2:
            continue
        edges = lo + bin_size_us * np.arange(n_bins + 1)
        out.append((edges, np.histogram(ts, bins=edges)[0].astype(np.int64)))
    return out


def tile_spike(ts, bin_size_us, min_bin_difference, max_bin_ratio):
    """Step test: a big and sharp step between adjacent coarse bins."""
    for _, hist in _coarse_hists(ts, bin_size_us):
        ratio = (hist[1:] + 1.0) / (hist[:-1] + 1.0)        # +1 avoids /0
        sharp = (ratio.min() < max_bin_ratio) or ((1.0 / ratio).min() < max_bin_ratio)
        if sharp and np.abs(np.diff(hist)).max() > min_bin_difference:
            return True
    return False


def largest_jump_edge(ts, bin_size_us):
    """Time of the bin edge with the largest |step| over both bin phases."""
    best_edge, best_step = float(ts.min()), -1
    for edges, hist in _coarse_hists(ts, bin_size_us):
        steps = np.abs(np.diff(hist))
        i = int(np.argmax(steps))
        if steps[i] > best_step:
            best_edge, best_step = float(edges[i + 1]), int(steps[i])
    return best_edge


def fine_peak(ts, centre_us, search_us=10.0):
    """Centre of the busiest 1 us bin within centre +- search_us."""
    edges = np.arange(centre_us - search_us, centre_us + search_us + 0.5, 1.0)
    i = int(np.argmax(np.histogram(ts, bins=edges)[0]))
    return float(edges[i] + 0.5)


def excess_kurtosis(ts, centre_us, half_width_us):
    """Excess kurtosis of the 1 us histogram in centre +- half_width
    (bin centres weighted by counts; uniform = -1.2)."""
    edges = np.arange(centre_us - half_width_us, centre_us + half_width_us + 0.5, 1.0)
    counts = np.histogram(ts, bins=edges)[0]
    n = counts.sum()
    if n < 3:
        return float('nan')
    x = 0.5 * (edges[1:] + edges[:-1])
    d = x - np.sum(counts * x) / n
    var = np.sum(counts * d ** 2) / n
    if var <= 0:
        return float('nan')
    return float(np.sum(counts * d ** 4) / n / var ** 2 - 3.0)


def tag_file(f, cuts=None):
    """Run the WTT selection over an open h5py flow file.

    Returns a dict with event counts, run time, the number of tiles passing
    each stage (hit/chip cuts, step test, kurtosis cut), and one record per
    tagged tile: (event_id, is_beam, io_group, tile, kurt20, kurt10, unix_ts).
    """
    c = dict(DEFAULT_CUTS, **(cuts or {}))
    events = np.asarray(f['charge/events/data'])
    hits_ref = f['charge/events/ref/charge/%s/ref' % HITS_DSET]
    hits_region = f['charge/events/ref/charge/%s/ref_region' % HITS_DSET]
    packets_full = f['charge/packets/data']
    packets_ref = f['charge/%s/ref/charge/packets/ref' % HITS_DSET]
    packets_region = f['charge/%s/ref/charge/packets/ref_region' % HITS_DSET]

    beam_ids = beam_event_ids(f)
    if beam_ids is None:
        beam_ids = np.empty(0, dtype=np.int64)
    unix_ts = np.asarray(events['unix_ts'], dtype=np.int64)

    out = dict(n_events=int(len(events)),
               n_beam_events=int(np.isin(events['id'], beam_ids).sum()),
               first_unix_ts=int(unix_ts.min()) if unix_ts.size else None,
               last_unix_ts=int(unix_ts.max()) if unix_ts.size else None,
               n_candidate_tiles=0, n_step_tiles=0, tagged_tiles=[])

    for event in events[events['nhit'] > c['min_hits_per_event']]:
        ev_id = int(event['id'])
        hit_ids = np.unique(ref_children(hits_region, hits_ref, [ev_id])[:, 1])
        if hit_ids.size == 0:
            continue
        pkt_pairs = ref_children(packets_region, packets_ref, hit_ids)
        if pkt_pairs.shape[0] == 0:
            continue
        packets = np.asarray(packets_full[np.sort(pkt_pairs[:, 1])])
        ts_all = np.asarray(packets['timestamp'], dtype=np.int64)
        tile_of_packet = io_channel_to_tile(packets['io_channel'])
        iog_of_packet = np.asarray(packets['io_group'])

        io_groups, iog_counts = np.unique(iog_of_packet, return_counts=True)
        for iog in io_groups[iog_counts > c['min_hits_per_io_group']]:
            iog_mask = iog_of_packet == iog
            tiles, tile_counts = np.unique(tile_of_packet[iog_mask], return_counts=True)
            for tile in tiles[tile_counts > c['min_hits_per_tile']]:
                mask = iog_mask & (tile_of_packet == tile)
                if np.unique(packets['chip_id'][mask]).size < c['min_chips_per_tile']:
                    continue
                out['n_candidate_tiles'] += 1
                raw = ts_all[mask]
                ts = (raw - raw.min()) / 10.0          # us, relative to first packet
                if ts.max() <= 0:
                    continue
                if not tile_spike(ts, c['bin_size_us'], c['min_bin_difference'], c['max_bin_ratio']):
                    continue
                out['n_step_tiles'] += 1
                peak = fine_peak(ts, largest_jump_edge(ts, c['bin_size_us']))
                k20 = excess_kurtosis(ts, peak, 20.0)
                k10 = excess_kurtosis(ts, peak, 10.0)
                if not (k20 > c['kurt20_cut'] or k10 > c['kurt10_cut']):
                    continue
                out['tagged_tiles'].append(dict(
                    event_id=ev_id, is_beam=bool(np.isin(ev_id, beam_ids)),
                    io_group=int(iog), tile=int(tile),
                    kurt20=round(k20, 3), kurt10=round(k10, 3),
                    unix_ts=int(event['unix_ts'])))
    return out
