"""Read handwriting with Grape, the FineNotes handwriting model, in plain Python.

Needs only numpy. It computes exactly what FineNotes computes on the device:
the int8 network, the character language model stored in the same file, and
the beam search with the file's own settings.

    python reader.py grape-1.0.bin line.json

`line.json` holds ONE line of handwriting:

    {"strokes": [{"points": [[x, y], [x, y], ...], "t0": 1712345678901},
                 ...]}

- x, y: pen positions in any unit (pixels are fine), y pointing DOWN as on a
  screen. Only the shape matters: the line is normalised by its own size.
- t0 (optional): when the stroke started, in milliseconds. With start times
  the strokes are read in writing order and the pauses between them count;
  without, strokes are read left to right.

From Python:

    import reader
    model = reader.load("grape-1.0.bin")
    text = model.read([[(10, 50), (12, 40), ...], ...], t0s=[0, 420, ...])

The file format is described in FORMAT.md.
Licence: Apache 2.0 (see LICENSE and NOTICE).
"""

from __future__ import annotations

import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MAGIC = b"FNHW"
BOS = "\x02"
NEG = -1e30


# ------------------------------------------------------------------ the file


def _dequantise(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale[:, None]


class _LM:
    """The 6-gram character model (interpolated Witten-Bell) as stored in the
    file: a trie of contexts read from the most recent character backwards.
    A context the file does not hold ends the walk (it was pruned)."""

    def __init__(self, meta: dict, payload: bytes):
        self.n = meta["order"]
        self.chars = meta["chars"]
        self.code = {c: i for i, c in enumerate(self.chars)}
        self.V = meta["V"]
        nodes, preds, count_bytes = struct.unpack_from("<III", payload, 0)
        o = 12
        self.node_char = np.frombuffer(payload, np.uint8, nodes, o)
        o += nodes
        self.children = np.frombuffer(payload, np.uint8, nodes, o).astype(np.int64)
        o += nodes
        self.npreds = np.frombuffer(payload, np.uint8, nodes, o).astype(np.int64)
        o += nodes
        self.pred_char = np.frombuffer(payload, np.uint8, preds, o)
        o += preds
        counts, pos = [], o
        for _ in range(preds):
            v = shift = 0
            while True:
                b = payload[pos]
                pos += 1
                v |= (b & 0x7F) << shift
                shift += 7
                if b < 0x80:
                    break
            counts.append(v)
        self.counts = np.array(counts, np.int64)
        self.first_child = np.concatenate([[1], 1 + np.cumsum(self.children)[:-1]])
        self.first_pred = np.concatenate([[0], np.cumsum(self.npreds)[:-1]])
        csum = np.concatenate([[0], np.cumsum(self.counts)])
        self.tot = csum[self.first_pred + self.npreds] - csum[self.first_pred]
        self._cache: dict = {}

    @staticmethod
    def _find(keys: np.ndarray, start: int, n: int, code: int) -> int:
        lo, hi = start, start + n
        while lo < hi:
            mid = (lo + hi) >> 1
            if keys[mid] < code:
                lo = mid + 1
            else:
                hi = mid
        return lo if lo < start + n and keys[lo] == code else -1

    def logp(self, hist: str, c: str) -> float:
        key = (hist[-(self.n - 1):], c)
        if key in self._cache:
            return self._cache[key]
        cc = self.code.get(c, -1)
        p = 1.0 / self.V
        node = 0
        for k in range(self.n):
            if k:
                ch = hist[-k] if k <= len(hist) else BOS
                node = self._find(self.node_char, int(self.first_child[node]), int(self.children[node]), self.code.get(ch, -1))
                if node < 0:
                    break
            typ = int(self.npreds[node])
            tot = int(self.tot[node])
            j = self._find(self.pred_char, int(self.first_pred[node]), typ, cc)
            cnt = int(self.counts[j]) if j >= 0 else 0
            p = (cnt + typ * p) / (tot + typ)
        lp = math.log(p)
        self._cache[key] = lp
        return lp


class _Net:
    """The bidirectional LSTM: int8 weights de-quantised per row, then a
    linear layer and log-softmax. Gate order: input, forget, cell, output."""

    def __init__(self, meta: dict, buf: bytes):
        I, H, L, C = meta["inputs"], meta["hidden"], meta["layers"], meta["classes"]
        pos = 0

        def take(dtype, count):
            nonlocal pos
            a = np.frombuffer(buf, dtype, count, pos)
            pos += a.nbytes + (-a.nbytes % 4)
            return a

        self.mean = self.std = None
        if meta["standardize"]:
            self.mean, self.std = take(np.float32, I), take(np.float32, I)
        self.layers = []
        for k in range(L):
            n_in = I if k == 0 else 2 * H
            dirs = []
            for _ in range(2):
                s_ih, s_hh, bias = take(np.float32, 4 * H), take(np.float32, 4 * H), take(np.float32, 4 * H)
                w_ih = _dequantise(take(np.int8, 4 * H * n_in).reshape(4 * H, n_in), s_ih)
                w_hh = _dequantise(take(np.int8, 4 * H * H).reshape(4 * H, H), s_hh)
                dirs.append((w_ih, w_hh, bias.astype(np.float32)))
            self.layers.append(dirs)
        s_out, self.b_out = take(np.float32, C), take(np.float32, C)
        self.w_out = _dequantise(take(np.int8, C * 2 * H).reshape(C, 2 * H), s_out)

    @staticmethod
    def _run(x, w_ih, w_hh, b, reverse):
        T, H = x.shape[0], w_hh.shape[1]
        h = np.zeros(H, np.float32)
        c = np.zeros(H, np.float32)
        out = np.zeros((T, H), np.float32)
        pre = x @ w_ih.T + b
        sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))  # noqa: E731
        for t in (range(T - 1, -1, -1) if reverse else range(T)):
            z = pre[t] + w_hh @ h
            i, f, g, o = z[:H], z[H : 2 * H], z[2 * H : 3 * H], z[3 * H :]
            c = sig(f) * c + sig(i) * np.tanh(g)
            h = sig(o) * np.tanh(c)
            out[t] = h
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = x.astype(np.float32)
        if self.mean is not None:
            h = (h - self.mean) / self.std
        for fwd, bwd in self.layers:
            h = np.concatenate([self._run(h, *fwd, False), self._run(h, *bwd, True)], axis=1)
        z = h @ self.w_out.T + self.b_out
        z -= z.max(1, keepdims=True)
        return z - np.log(np.exp(z).sum(1, keepdims=True))


# ------------------------------------------------------------------ features


def _extrema_min(strokes: list[np.ndarray]) -> np.ndarray:
    out = []
    for s in strokes:
        y = s[:, 1]
        if len(s) < 3:
            out.append(s[np.argmin(y), :2])
            continue
        idx = np.nonzero((y[1:-1] <= y[:-2]) & (y[1:-1] <= y[2:]))[0] + 1
        out.extend(s[idx, :2])
    return np.array(out, dtype=float).reshape(-1, 2)


def _robust_line(pts: np.ndarray) -> tuple[float, float]:
    if len(pts) < 2 or np.ptp(pts[:, 0]) <= 0:
        return 0.0, float(np.median(pts[:, 1])) if len(pts) else 0.0
    keep = np.ones(len(pts), bool)
    a, b = 0.0, float(np.median(pts[:, 1]))
    for _ in range(3):
        if keep.sum() < 2 or np.ptp(pts[keep, 0]) <= 0:
            break
        a, b = np.polyfit(pts[keep, 0], pts[keep, 1], 1)
        r = pts[:, 1] - (a * pts[:, 0] + b)
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-9
        keep = np.abs(r - np.median(r[keep])) <= 1.5 * 1.4826 * mad
    return float(a), float(b)


def _timed(strokes: list[np.ndarray], t0s: list[float | None], heights_per_second: float) -> list[np.ndarray]:
    """FineNotes stores one start time per stroke and no time per point: the
    pen moves at a constant speed inside a stroke (heights_per_second line
    heights per second); between strokes the stored pause counts."""
    ys = np.concatenate([s[:, 1] for s in strokes])
    speed = heights_per_second * max(float(np.ptp(ys)), 1.0)
    timed, t_end, prev_end = [], 0.0, None
    for i, xy in enumerate(strokes):
        if prev_end is None:
            t_start = 0.0
        elif t0s[i] is not None and t0s[i - 1] is not None and t0s[i] > t0s[i - 1]:
            t_start = max(t_end + 0.01, timed[-1][0, 2] + (t0s[i] - t0s[i - 1]) / 1000.0)
        else:
            t_start = t_end + np.hypot(*(xy[0] - prev_end)) / speed + 0.15
        seg = np.sqrt((np.diff(xy, axis=0) ** 2).sum(1)) if len(xy) > 1 else np.zeros(0)
        t = t_start + np.concatenate([[0.0], np.cumsum(seg)]) / speed
        timed.append(np.column_stack([xy, t]))
        t_end, prev_end = t[-1], xy[-1]
    return timed


def _normalise(timed: list[np.ndarray], core_units: float) -> list[np.ndarray]:
    """Baseline to y = 0, x from 0, and the core (x-height) zone scaled to
    core_units. The model was trained without levelling or deslanting."""
    pts = np.concatenate(timed)
    full = max(float(np.ptp(pts[:, 1])), 1e-6)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    out = [np.column_stack([s[:, 0] - cx, s[:, 1] - cy, s[:, 2]]) for s in timed]
    mins = _extrema_min(out)
    base = _robust_line(mins)[1] if len(mins) >= 2 else float(np.concatenate(out)[:, 1].min())
    step, ys = full / 60.0, []
    for st in out:
        seg = np.hypot(*np.diff(st[:, :2], axis=0).T) if len(st) > 1 else np.zeros(0)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(int(arc[-1] / step), 1)
        ys.append(np.interp(np.linspace(0, arc[-1], n + 1), arc, st[:, 1]) if arc[-1] > 0 else st[:1, 1])
    ys = np.concatenate(ys)
    core = (np.percentile(ys, 80) - np.percentile(ys, 20)) / 0.6
    if not (0.2 * full <= core <= 0.9 * full):
        core = 0.45 * full
    scale = core_units / max(core, 1e-6)
    res = [np.column_stack([r[:, 0], r[:, 1] - base, r[:, 2]]) for r in out]
    x0 = min(r[:, 0].min() for r in res)
    return [np.column_stack([(r[:, 0] - x0) * scale, r[:, 1] * scale, r[:, 2]]) for r in res]


def features(strokes, t0s=None, settings: dict | None = None) -> np.ndarray:
    """One line of strokes (each a sequence of (x, y), y down) -> the
    model's input frames (dx, dy, dt, new stroke)."""
    st = settings or {"pointsPerUnit": 20.0, "heightsPerSecond": 5.3, "coreUnits": 0.4}
    xy = [np.asarray(s, dtype=float)[:, :2] * np.array([1.0, -1.0]) for s in strokes if len(s)]
    t0s = list(t0s) if t0s is not None else [None] * len(xy)
    if all(t is not None for t in t0s):
        order = sorted(range(len(xy)), key=lambda i: t0s[i])
    else:
        order = sorted(range(len(xy)), key=lambda i: xy[i][:, 0].min())
    xy, t0s = [xy[i] for i in order], [t0s[i] for i in order]
    norm = _normalise(_timed(xy, t0s, st["heightsPerSecond"]), st["coreUnits"])
    xs, ys, ts, ns = [], [], [], []
    for k, s in enumerate(norm):
        x, y, t = s[:, 0], s[:, 1], s[:, 2]
        if len(s) == 1:
            xs.append(x), ys.append(y), ts.append(t), ns.append([k])
            continue
        n = max(int(np.ceil(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2).sum() * st["pointsPerUnit"])), 2)
        tn = (t - t.min()) / max(1e-9, t.max() - t.min())
        grid = np.linspace(0, 1, n)
        xs.append(np.interp(grid, tn, x)), ys.append(np.interp(grid, tn, y))
        ts.append(np.interp(grid, tn, t)), ns.append([k] * n)
    x, y, t, stroke = (np.concatenate(v) for v in (xs, ys, ts, ns))
    f = np.zeros((len(x), 4), np.float32)
    f[1:, 0], f[1:, 1], f[1:, 2] = np.diff(x), np.diff(y), np.diff(t)
    f[:, 2] = np.clip(f[:, 2], 0.0, 2.0)
    f[0, 3] = 1.0
    f[1:, 3] = np.diff(stroke)
    return f


# ------------------------------------------------------------------ decoding


def _lse(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    return a if b <= NEG else a + math.log1p(math.exp(b - a))


def greedy(logp: np.ndarray, chars: list[str]) -> str:
    out, prev = [], 0
    for k in logp.argmax(1):
        if k != 0 and k != prev:
            out.append(chars[k])
        prev = k
    return "".join(out)


def beam_decode(logp: np.ndarray, chars: list[str], lm: _LM | None, alpha: float, beta: float,
                beam: int = 16, topk: int = 8, prune: float = -9.0) -> str:
    """CTC prefix beam search; each new character adds alpha * log P_lm + beta."""
    beams = {"": (0.0, NEG)}
    for t in range(logp.shape[0]):
        lp = logp[t]
        cand = [int(c) for c in np.argsort(lp)[::-1][: topk + 1] if c != 0 and lp[c] > prune][:topk]
        nxt: dict[str, list[float]] = defaultdict(lambda: [NEG, NEG])
        for prefix, (pb, pnb) in beams.items():
            ptot = _lse(pb, pnb)
            e = nxt[prefix]
            e[0] = _lse(e[0], ptot + float(lp[0]))
            last = prefix[-1] if prefix else None
            for c in cand:
                ch, p = chars[c], float(lp[c])
                bonus = (alpha * lm.logp(prefix, ch) + beta) if lm else 0.0
                if ch == last:
                    e[1] = _lse(e[1], pnb + p)
                    n = nxt[prefix + ch]
                    n[1] = _lse(n[1], pb + p + bonus)
                else:
                    n = nxt[prefix + ch]
                    n[1] = _lse(n[1], ptot + p + bonus)
        beams = dict(sorted(nxt.items(), key=lambda kv: -_lse(*kv[1]))[:beam])
    final = lambda kv: _lse(*kv[1]) + (alpha * lm.logp(kv[0], "\n") if lm else 0.0)  # noqa: E731
    return max(beams.items(), key=final)[0]


# ------------------------------------------------------------------ the model


class Model:
    def __init__(self, path: str | Path):
        data = Path(path).read_bytes()
        if data[:4] != MAGIC:
            raise ValueError("not a FineNotes handwriting model file")
        _, meta_len = struct.unpack_from("<II", data, 4)
        self.meta = json.loads(data[12 : 12 + meta_len].decode("utf8"))
        o, sections = 12 + meta_len, {}
        while o < len(data):
            tag = data[o : o + 4].decode("ascii")
            (size,) = struct.unpack_from("<I", data, o + 4)
            sections[tag] = data[o + 8 : o + 8 + size]
            o += 8 + size + (-size % 4)
        self.chars = ["<blank>"] + list(self.meta["alphabet"])
        self.net = _Net(self.meta["net"], sections["NET1"])
        self.lm = _LM(self.meta["lm"], sections["LM01"]) if "LM01" in sections else None
        self.decoder = self.meta["decoder"]

    def logprobs(self, strokes, t0s=None) -> np.ndarray:
        return self.net(features(strokes, t0s, self.meta["features"]))

    def read(self, strokes, t0s=None, use_lm: bool = True) -> str:
        """One line of strokes -> its text."""
        lp = self.logprobs(strokes, t0s)
        if not use_lm or self.lm is None:
            return greedy(lp, self.chars)
        d = self.decoder
        return beam_decode(lp, self.chars, self.lm, d["alpha"], d["beta"], d["beam"], d["topk"], d["prune"])


def load(path: str | Path) -> Model:
    return Model(path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    model = load(sys.argv[1])
    line = json.loads(Path(sys.argv[2]).read_text(encoding="utf8"))
    strokes = [s["points"] for s in line["strokes"]]
    t0s = [s.get("t0") for s in line["strokes"]]
    print(model.read(strokes, t0s if all(t is not None for t in t0s) else None))
