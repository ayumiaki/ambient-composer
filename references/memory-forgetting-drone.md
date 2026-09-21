# memory-forgetting drone — working fallback (2026-09-16)

Ship's `compose.py` throws a broadcast bug on ≥150s (stereo array sizing). This
stdlib-only numpy script works around it and produces the same "memory and
forgetting" arc.

## Run

```bash
mkdir -p /home/rynardt/songs
/home/rynardt/hermes-agent/venv/bin/python3 /home/rynardt/.hermes/learn_by_building/builds/memory_forget_drone.py
```

Output: `/home/rynardt/songs/where-memory-rest.mp3` (150s, 44.1kHz stereo, ~1.2MB).

## Design notes

- Fundamental D₂ 73.4 Hz, harmonic stack that erodes section-by-section:
  emerge (full) → persist (full) → erode (1,2,3) → dissolve (1 only).
- Stereo detuning ±0.18 Hz per harmonic + 0.04 Hz pan LFO for beating/width.
- Brown-noise lowpass floor, cutoff 800→200 Hz sweep.
- Master fade-out last 5s, normalised to −1 dB.
