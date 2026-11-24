#!/usr/bin/env python3
"""Ingest Tesla dashcam clips.

Reads data/clips/<clip>/<clip>-<cam>.mp4 and writes data/drive/:
  frames/<cam>/<n:06d>.jpg   every decoded frame, n = running index per camera
  cams/<cam>.json            per-frame table: n, clip, idx, pts, seq, t, SEI fields, ego pose
  timeline.npz               master clock (front camera): t, seq, x, y, yaw, speed, ...
  meta.json                  fps, origin lat/lon, per-camera seq offsets, image sizes

Every camera's frames are put on the front camera's clock. Where a camera counts the same frames as the
front (all of HW4's, HW3's repeaters) that is a constant counter offset; HW3's back camera has a counter
of its own and is timed from its file timestamps instead (see free_clock). HW3 also reports its heading
about 0.6 s late, which heading_lag measures and main() undoes before integrating the trajectory.

Each packet (access unit) carries its own SEI, so metadata is paired with the
frame it was muxed with, not by counting NALs (some files hold SEIs with no
slice, and ffmpeg's image2 muxer drops frames with duplicate dts).

Conventions: world = ENU metres (x east, y north, z up), origin at the first
front frame. yaw = counter-clockwise from east, radians. Tesla heading_deg is
clockwise from north.
"""
import av, cv2, json, math, os, struct, sys, glob, collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # run by path
from pipeline.common import hardware  # noqa: E402

# Every camera name a dashcam file can carry. HW4 records all six, HW3 the first four; main() and the
# runner's inventory work with whichever of them the footage has.
CAMS = ['front', 'back', 'left_repeater', 'right_repeater', 'left_pillar', 'right_pillar']
NAMES = {1: 'ver', 2: 'gear', 3: 'seq', 4: 'speed', 5: 'accel_pedal', 6: 'steer', 7: 'blink_l',
         8: 'blink_r', 9: 'brake', 10: 'ap', 11: 'lat', 12: 'lon', 13: 'heading',
         14: 'ax', 15: 'ay', 16: 'az'}


def unescape(b):
    out = bytearray(); z = 0
    for x in b:
        if z >= 2 and x == 3:
            z = 0; continue
        out.append(x); z = z + 1 if x == 0 else 0
    return bytes(out)


def varint(b, i):
    v = 0; s = 0
    while True:
        x = b[i]; i += 1; v |= (x & 0x7f) << s; s += 7
        if not x & 0x80:
            return v, i


def parse_proto(b):
    f = {}; i = 0
    try:
        while i < len(b):
            k, i = varint(b, i); fn = k >> 3; wt = k & 7
            if wt == 0: v, i = varint(b, i)
            elif wt == 5: v = struct.unpack('<f', b[i:i + 4])[0]; i += 4
            elif wt == 1: v = struct.unpack('<d', b[i:i + 8])[0]; i += 8
            elif wt == 2:
                l, i = varint(b, i); i += l; continue
            else: break
            f[NAMES.get(fn, str(fn))] = v
    except (IndexError, struct.error):
        pass
    return f


def parse_packet(buf):
    """Return (sei dict or None, has_slice) for one AVCC access unit."""
    p = 0; sei = None; has_slice = False; n = len(buf)
    while p + 4 <= n:
        ln = struct.unpack('>I', buf[p:p + 4])[0]
        if ln == 0 or p + 4 + ln > n: break
        nal = buf[p + 4:p + 4 + ln]; t = nal[0] & 0x1f
        if t in (1, 5): has_slice = True
        elif t == 6 and sei is None:
            b = unescape(nal); k = 3
            while k < len(b) and b[k] == 0x42: k += 1
            if k > 3 and k < len(b) and b[k] == 0x69:
                sei = parse_proto(b[k + 1:-1])
        p += 4 + ln
    return sei, has_slice


def do_file(args):
    path, outdir, start_n, quality = args
    os.makedirs(outdir, exist_ok=True)
    c = av.open(path); st = c.streams.video[0]
    st.thread_type = 'AUTO'
    rows = []; pending = collections.defaultdict(collections.deque); n = start_n; idx = 0
    def emit(frame):
        nonlocal n, idx
        # pair by pts: a few packets never produce a frame, so order alone would shift the metadata
        sei, pts = pending[frame.pts].popleft()
        img = frame.to_ndarray(format='bgr24')
        cv2.imwrite(os.path.join(outdir, f'{n:06d}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        r = {'n': n, 'idx': idx, 'pts': pts}; r.update(sei or {})
        rows.append(r); n += 1; idx += 1
    for pkt in c.demux(st):
        if pkt.size:
            sei, has_slice = parse_packet(bytes(pkt))
            if not has_slice: continue
            pending[pkt.pts].append((sei, float(pkt.pts * st.time_base)))
        for fr in pkt.decode():
            emit(fr)
    lost = sum(len(q) for q in pending.values())
    return path, rows, (st.width, st.height), lost


def smooth(p, half):
    """Local straight-line fit of p over 2*half+1 samples: a moving average away from the ends."""
    n = len(p); w = 2 * half + 1; i = np.arange(n, dtype=np.float64)
    if n < w:
        return np.polyval(np.polyfit(i, p, 1), i) if n > 1 else np.array(p, dtype=np.float64)
    out = np.empty(n)
    out[half:n - half] = np.convolve(p, np.ones(w) / w, 'valid')
    out[:half] = np.polyval(np.polyfit(i[:w], p[:w], 1), i[:half])
    out[n - half:] = np.polyval(np.polyfit(i[n - w:], p[n - w:], 1), i[n - half:])
    return out


def offset_drift(rows, fk, key, mode, fps):
    """(frames per minute, minutes) by which a camera's counter offset from the front's drifts, from a
    straight line through the matched frames.

    A camera that counts the front's frames holds its offset: the mean wobbles by a tenth of a frame as the
    telemetry beats against the shutters, with no trend. HW3's back camera loses half a frame to a frame a
    minute even on a drive without a single stop, which a histogram of offsets does not show: three frames
    of drift in six minutes still sit within two of the mode.
    """
    d = np.array([(r['seq'], r['seq'] - s) for r in rows if 'seq' in r for s in fk.get(key(r), [])
                  if abs(r['seq'] - s - mode) <= 2], dtype=np.float64).reshape(-1, 2)
    if len(d) < 600: return 0.0, 0.0
    minutes = (d[:, 0] - d[0, 0]) / fps / 60
    return float(np.polyfit(minutes, d[:, 1], 1)[0]), float(minutes[-1])


def free_clock(rows, front, fk, key, seq0, fps):
    """{n: t} for a camera whose frame counter is not the front camera's.

    HW3's back camera counts its own frames, a little faster or slower than the others and at 30 instead
    of 36 fps whenever the car stands still, so no constant offset turns its counter into time. Inside one
    file its timestamps are a clock, though:
    smoothed (they jitter by a few ms), shifted onto the front clock by the frames that carry the same
    telemetry tuple as a front frame, and following the front's own slow drift of counter against
    timestamps, they agree with telemetry to about 2 ms, through a stop as well.
    """
    files = []
    for clip in sorted({r['clip'] for r in rows}):
        rs = [r for r in rows if r['clip'] == clip]
        fr = [r for r in front if r['clip'] == clip]
        p = smooth(np.array([r['pts'] for r in rs]), 12)
        # the front's counter time against its own timestamps in this file: a start time and a slow drift
        tf = (np.array([r['seq'] for r in fr], dtype=np.float64) - seq0) / fps
        lead = tf - smooth(np.array([r['pts'] for r in fr]), 12)
        start = float(np.median(lead))
        drift = smooth(lead, 36) - start
        hit = np.array([key(r) in fk for r in rs])
        shift = None
        if hit.any():
            tm = np.array([(np.mean(fk[key(r)]) - seq0) / fps for r in rs if key(r) in fk])
            d = tm - p[hit] - np.interp(tm, tf, drift)
            shift = float(np.median(d))
            near = np.abs(d - shift) < 1.5 / fps
            if near.any(): shift = float(d[near].mean())
        files.append((rs, p, tf, drift, start, shift))
    known = [shift - start for *_, start, shift in files if shift is not None]
    if not known:
        raise ValueError('a camera with its own frame counter shares no telemetry with the front camera')
    out = {}
    for rs, p, tf, drift, start, shift in files:
        if shift is None:
            # nothing moved in this file, so nothing matched: start it as far from the front's file as
            # the others do. That is some tens of ms out, which a parked car does not show.
            shift = start + float(np.median(known))
        t = p + shift
        for r, x in zip(rs, t + np.interp(t, tf, drift)): out[r['n']] = float(x)
    return out


# Steering leads the turning it causes by about this much. Measured on one HW3 drive, where the heading
# trailed the steering by 0.64 s and the yaw recovered from the images by 0.58 s.
STEER_LEAD = 0.06


def heading_lag(t, heading, speed, steer):
    """Seconds by which the telemetry heading trails the car's turning, or 0.0 when it is current.

    HW3 writes a heading that is about 0.6 s old into each frame, so a trajectory integrated from it turns
    late: degrees of yaw error and up to a metre of position error through every corner. Speed and steering
    angle are current, and steering angle x speed is proportional to yaw rate, so the lag is the shift that
    best lines the heading's rate up with it. That proportion only holds over a stretch of similar driving
    (a fast straight with the wheel a hair off centre breaks it), so it is measured minute by minute and
    the median taken over the minutes that turn enough to tell. HW4 measures 0.10 to 0.16 s this way, most
    of it the steering lead, and is left exactly as it is; so is a drive with too little turning.
    """
    dt = 0.02
    u = np.arange(t[0], t[-1], dt)
    if len(u) < 1000 or not np.any(steer): return 0.0
    box = np.ones(15) / 15
    sm = lambda x: np.convolve(np.interp(u, t, x), box, 'same')
    v = sm(speed); rate = np.gradient(sm(heading), dt); want = sm(steer) * v
    ok = v > 2.0
    span = min(3000, len(u) - 75)
    lags = []
    for a in range(0, len(u) - 75 - span + 1, 1500):
        y = want[a:a + span]; my = ok[a:a + span]
        if my.sum() < 500 or np.abs(rate[a:a + span][my]).max() < 0.1: continue
        best = (0.0, 0.0)
        for sh in range(76):
            m = my & ok[a + sh:a + sh + span]
            x = rate[a + sh:a + sh + span][m]
            if len(x) < 500 or x.std() == 0 or y[m].std() == 0: continue
            c = abs(np.corrcoef(x, y[m])[0, 1])
            if c > best[1]: best = (sh * dt, c)
        if best[1] >= 0.95: lags.append(best[0])
    lag = float(np.median(lags)) if lags else 0.0
    print(f'heading trails steering by {lag:.2f} s (median of {len(lags)} one-minute stretches'
          + (f', {min(lags):.2f} to {max(lags):.2f} s)' if lags else ')'), flush=True)
    return round(lag - STEER_LEAD, 3) if lag > 0.3 else 0.0


def main(root, heading_late=None):
    clips = sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, 'clips', '*')) if os.path.isdir(p))
    out = os.path.join(root, 'drive'); os.makedirs(out, exist_ok=True)
    counts_path = os.path.join(out, 'raw_tables.json')
    if os.path.exists(counts_path) and '--force' not in sys.argv:
        raw = json.load(open(counts_path))
    else:
        # the cameras this car records: those with a file in the first clip
        cams = [c for c in CAMS if clips and os.path.exists(os.path.join(root, 'clips', clips[0], f'{clips[0]}-{c}.mp4'))]
        # pass 1: count frames per file so running indices are known up front
        jobs = []
        for cam in cams:
            n = 0
            for clip in clips:
                path = os.path.join(root, 'clips', clip, f'{clip}-{cam}.mp4')
                c = av.open(path); st = c.streams.video[0]
                cnt = sum(1 for pkt in c.demux(st) if pkt.size and parse_packet(bytes(pkt))[1])
                # running indices leave a gap where a packet yields no frame
                jobs.append((path, os.path.join(out, 'frames', cam), n, 95)); n += cnt
        raw = {cam: [] for cam in cams}; sizes = {}
        with ProcessPoolExecutor(12) as ex:
            for path, rows, size, lost in ex.map(do_file, jobs):
                clip = os.path.basename(os.path.dirname(path)); cam = os.path.basename(path)[len(clip) + 1:-4]
                for r in rows: r['clip'] = clip
                raw[cam] += rows; sizes[cam] = size
                print('decoded', os.path.basename(path), len(rows), 'undecodable packets', lost, flush=True)
        for cam in cams: raw[cam].sort(key=lambda r: r['n'])
        raw['_sizes'] = sizes
        json.dump(raw, open(counts_path, 'w'))
    sizes = raw['_sizes']
    cams = [c for c in CAMS if c in raw]
    if 'front' not in cams:
        raise ValueError('no front camera clips: the front camera is the master clock')

    # ---- master clock from the front camera
    F = [r for r in raw['front'] if 'seq' in r and 'lat' in r]
    seq = np.array([r['seq'] for r in F], dtype=np.int64)
    # fps from pts inside clips
    num = den = 0.0
    for clip in clips:
        rs = [r for r in F if r['clip'] == clip]
        num += rs[-1]['seq'] - rs[0]['seq']; den += rs[-1]['pts'] - rs[0]['pts']
    fps = num / den
    seq0 = int(seq[0]); t = (seq - seq0) / fps
    g = lambda k, d=0.0: np.array([r.get(k, d) for r in F], dtype=np.float64)
    speed = g('speed'); heading = np.unwrap(np.radians(g('heading')))
    # heading_late comes from the runner, which measures it once over the whole drive; alone, measure it here
    late = heading_lag(t, heading, speed, g('steer')) if heading_late is None else float(heading_late)
    if late:
        heading = np.interp(t + late, t, heading)
        print(f'heading moved {late:.2f} s earlier', flush=True)
    lat = g('lat'); lon = g('lon')
    R = 6378137.0; lat0 = lat[0]; lon0 = lon[0]
    gx = np.radians(lon - lon0) * R * math.cos(math.radians(lat0)); gy = np.radians(lat - lat0) * R
    dt = np.diff(t, prepend=t[0])
    vm = np.concatenate([[speed[0]], 0.5 * (speed[1:] + speed[:-1])])
    hm = np.concatenate([[heading[0]], 0.5 * (heading[1:] + heading[:-1])])
    x = np.cumsum(vm * dt * np.sin(hm)); y = np.cumsum(vm * dt * np.cos(hm))
    yaw = math.pi / 2 - heading
    np.savez(os.path.join(out, 'timeline.npz'), t=t, seq=seq, x=x, y=y, yaw=yaw, speed=speed,
             heading=heading, gps_x=gx, gps_y=gy, lat=lat, lon=lon, steer=g('steer'),
             ax=g('ax'), ay=g('ay'), az=g('az'), ap=g('ap'), brake=g('brake'),
             blink_l=g('blink_l'), blink_r=g('blink_r'), accel_pedal=g('accel_pedal'), gear=g('gear'))

    # ---- per-camera seq offset relative to front, from identical telemetry tuples
    key = lambda r: (r.get('lat'), r.get('lon'), r.get('speed'), r.get('heading'), r.get('ax'))
    fk = collections.defaultdict(list)
    for r in F:
        if r.get('speed', 0) > 1.0: fk[key(r)].append(r['seq'])
    offsets = {'front': 0.0}; clocks = {}
    known = hardware.own_counter(cams, sizes)
    for cam in cams[1:]:
        cnt = collections.Counter()
        for r in raw[cam]:
            for s in fk.get(key(r), []): cnt[r['seq'] - s] += 1
        mode = cnt.most_common(1)[0][0]
        near = {k: v for k, v in cnt.items() if abs(k - mode) <= 2}
        offsets[cam] = sum(k * v for k, v in near.items()) / sum(near.values())
        print(cam, 'seq offset', round(offsets[cam], 3), dict(sorted(near.items())))
        share = sum(near.values()) / sum(cnt.values()); drift, minutes = offset_drift(raw[cam], fk, key, mode, fps)
        # This camera does not count the front's frames if its layout says so, if one offset does not explain
        # the matches, or if the offset drifts. Drift needs two minutes to tell from the wobble of a camera
        # that does count them (under 0.02 frames a minute on HW4 and on HW3's repeaters).
        if cam in known or share < 0.9 or (minutes >= 2 and abs(drift) > 0.05 and abs(drift) * minutes > 0.15):
            clocks[cam] = free_clock([r for r in raw[cam] if 'seq' in r], F, fk, key, seq0, fps)
            print(cam, f'has its own frame counter (one offset explains {share:.0%} of the telemetry matches and '
                  f'drifts {drift:+.2f} frames a minute): timed from file timestamps')

    os.makedirs(os.path.join(out, 'cams'), exist_ok=True)
    for cam in cams:
        rows = []
        for r in raw[cam]:
            if 'seq' not in r: continue
            tc = clocks[cam][r['n']] if cam in clocks else (r['seq'] - offsets[cam] - seq0) / fps
            row = dict(r); row['t'] = tc
            row['ego_x'] = float(np.interp(tc, t, x)); row['ego_y'] = float(np.interp(tc, t, y))
            row['ego_yaw'] = float(np.interp(tc, t, yaw)); row['ego_speed'] = float(np.interp(tc, t, speed))
            rows.append(row)
        json.dump(rows, open(os.path.join(out, 'cams', cam + '.json'), 'w'))
        print(cam, 'frames', len(raw[cam]), 'with sei', len(rows), 't', round(rows[0]['t'], 3), round(rows[-1]['t'], 3))
    meta = {'fps': fps, 'seq0': seq0, 'origin_lat': lat0, 'origin_lon': lon0, 'seq_offsets': offsets,
            'sizes': sizes, 'cams': cams, 'clips': clips, 'duration_s': float(t[-1]),
            'distance_m': float(np.sum(vm * dt))}
    # seq_offsets means nothing for a camera listed here: its rows' t is the only clock it has
    if clocks: meta['free_clocks'] = sorted(clocks)
    if late: meta['heading_late_s'] = late
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print('fps', fps, 'duration', t[-1], 'distance', np.sum(vm * dt), 'DR end', x[-1], y[-1], 'GPS end', gx[-1], gy[-1])


if __name__ == '__main__':
    main(sys.argv[1])
