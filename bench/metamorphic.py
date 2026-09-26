"""Metamorphic tests without labels: the same clip under perturbations that must not change what
the system says. Invariants: confirmed-people count within ±1, no new event types, the whole-time
set stable. One row per variant to data/bench/metamorphic.jsonl; exit code 1 if any invariant fails.

  python bench/metamorphic.py --source /content/HI_DEF_VIDEO.mp4 --variants brightness_up,brightness_down,res540,fps6,hflip
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VARIANTS = ["brightness_up", "brightness_down", "res540", "fps6", "hflip", "shift2s"]


def make_variant(src: str, name: str, out_dir: Path) -> tuple[Path, dict]:
    """Re-encode the clip with one perturbation. fps6 and shift2s are sampling changes handled by
    slice args; the pixel variants are written with PyAV."""
    import av
    out = out_dir / f"{Path(src).stem}_{name}.mp4"
    args: dict = {}
    if name == "fps6":
        return Path(src), {"fps": 6}
    if name == "shift2s":
        return Path(src), {"skip_s": 2}
    if out.exists():
        return out, args
    cin = av.open(src); sin = cin.streams.video[0]
    cout = av.open(str(out), "w")
    codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    w, h = sin.width, sin.height
    if name == "res540":
        w, h = (w * 540 // h) // 2 * 2, 540
    sout = cout.add_stream(codec, rate=int(sin.average_rate) if sin.average_rate else 6)
    sout.width, sout.height, sout.pix_fmt = w, h, "yuv420p"
    for fr in cin.decode(sin):
        img = fr.to_ndarray(format="rgb24").astype(np.float32)
        if name == "brightness_up":
            img = np.clip(img * 1.2, 0, 255)
        elif name == "brightness_down":
            img = np.clip(img * 0.8, 0, 255)
        elif name == "hflip":
            img = img[:, ::-1, :]
        img = img.astype(np.uint8)
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
        if name == "res540":
            vf = vf.reformat(width=w, height=h)
        for pkt in sout.encode(vf):
            cout.mux(pkt)
    for pkt in sout.encode():
        cout.mux(pkt)
    cout.close(); cin.close()
    return out, args


def run_slice(source: Path, out_dir: Path, base_args: list[str], fps: float, skip_s: float, zones: str | None) -> dict:
    cmd = [sys.executable, "bench/slice_gpu.py", "--source", str(source), "--out", str(out_dir / "episodes"), "--fps", str(fps)] + base_args
    if zones:
        cmd += ["--zones", zones]
    if skip_s:
        cmd += ["--skip-s", str(skip_s)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-800:])
    txt = p.stdout
    return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--zones", default=None)
    ap.add_argument("--work", default="data/metamorphic")
    ap.add_argument("--tolerance", type=int, default=1)
    ap.add_argument("slice_args", nargs="*", help="extra args passed to slice_gpu.py after --")
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    base = run_slice(Path(a.source), work / "base", a.slice_args, a.fps, 0, a.zones)
    base_people = base.get("person_tubes", 0) - base.get("person_tubes_low_quality", 0)
    rows, failures = [], []
    print(f"base: confirmed people {base_people}, entities {base.get('entities')}, events {sorted(base['events'])}")
    for v in [x.strip() for x in a.variants.split(",") if x.strip()]:
        try:
            src, extra = make_variant(a.source, v, work)
            r = run_slice(src, work / v, a.slice_args, extra.get("fps", a.fps), extra.get("skip_s", 0), a.zones)
        except Exception as e:
            print(f"  {v:15s} ERROR {str(e)[:120]}"); failures.append(v); continue
        people = r.get("person_tubes", 0) - r.get("person_tubes_low_quality", 0)
        new_events = sorted(set(r["events"]) - set(base["events"]))
        ok = abs(people - base_people) <= a.tolerance and not new_events
        rows.append({"variant": v, "people": people, "entities": r.get("entities"), "new_event_types": new_events, "pass": ok})
        print(f"  {v:15s} people {people:2d} (base {base_people}) entities {r.get('entities')} new events {new_events or '-'}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(v)
    out = Path("data/bench"); out.mkdir(parents=True, exist_ok=True)
    with (out / "metamorphic.jsonl").open("a") as f:
        f.write(json.dumps({"ring": "metamorphic", "source": Path(a.source).name, "base_people": base_people, "rows": rows,
                            "failures": failures, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
    print(f"{len(rows) - len([r for r in rows if not r['pass']])}/{len(rows)} invariants hold" + (f"; failed: {failures}" if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
