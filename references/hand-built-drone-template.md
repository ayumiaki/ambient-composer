# Hand-Built Drone Composition Template

Generalized pattern for composing ambient drone pieces with stdlib+numpy, avoiding the `compose.py` broadcast bug on stereo ≥150s.

## The Pattern

### 1. Choose Fundamental + Theme

| Theme | Fundamental | Harmonic character | Arc |
|-------|-------------|-------------------|-----|
| memory/forgetting | D₂ 73.4 Hz | Full stack → strip to bass | emerge → persist → erode → dissolve |
| distance/absence | A₁ 55 Hz | Sparse (1,3,5,8,13) | expand → plateau → contract → vanish |
| solitude | C₂ 65.4 Hz | Single + slow harmonics | hush → presence → solitude → hush |
| becoming | E₂ 82.4 Hz | Rising harmonic count | seed → grow → bloom → transform |

### 2. Section Structure

Each section is a dict with:
```python
{"name": "...", "dur": 45, "harmonics": [1,3,5], "cutoff_start": 400, "cutoff_end": 200, "gain": 0.15, "detune": 0.12}
```

- `harmonics`: list of integer multiples of the fundamental
- `cutoff_start/end`: brown-noise lowpass sweep range (Hz)
- `gain`: peak amplitude for this section's harmonic stack
- `detune`: stereo detuning in Hz per harmonic

### 3. Build Per-Section

```python
n = int(SR * dur)
mono = np.zeros(n)
for h in harmonics:
    freq = FUND * h
    tone = np.sin(2 * np.pi * freq * np.arange(n) / SR)
    mono += tone / (h ** 0.7)  # amplitude rolloff
mono = mono / np.max(np.abs(mono)) * gain
```

### 4. Noise Floor

Brown noise through time-varying first-order IIR lowpass:

```python
white = np.random.randn(n)
brown = np.cumsum(white)
brown = brown / (np.max(np.abs(brown)) + 1e-9) * 0.3
# Sweep cutoff frame-by-frame (frame_size=512)
for i in range(0, n, frame_size):
    co = cutoff_env[i]
    frame = lowpass(brown[i:i+frame_size], co)
    noise_filtered[i:i+len(frame)] = frame[:min(frame_size, n-i)]
```

⚠️ Frame-by-frame with exact sizing avoids the mono/stereo shape mismatch that causes the broadcast bug.

### 5. Stereo Width

```python
stereo = np.zeros((n, 2))
# Left: detuned up, Right: detuned down
left = sum(sin(2π * freq * (1+detune) * t) / h^0.7 for h in harmonics)
right = sum(sin(2π * freq * (1-detune) * t) / h^0.7 for h in harmonics)
# Pan LFO for breathing
pan = 0.5 + 0.4 * sin(2π * 0.03 * t)
stereo[:, 0] = left * pan
stereo[:, 1] = right * (1-pan)
```

### 6. Master

- 8s linear fade-out on the tail
- Normalize to −1 dB (`target = 10**(-1/20)` ≈ 0.891)
- Write 16-bit stereo WAV via stdlib `wave`
- Convert to MP3 via `ffmpeg -qscale:a 3`

## Anti-Broadcast Rule

The compose.py bug: delay/overlap sizing assumes mono, stereo doubles one channel's array. Fix:
- Build harmonics into `np.zeros(n)` (mono) first
- Then assign to pre-allocated `np.zeros((n, 2))` stereo array
- Never do `stereo[:, 0] = some_stereo_sized_array`

## Verification

```bash
file output.mp3  # should show: Audio file with ID3, MPEG ADTS, layer III, 44.1 kHz, Stereo
stat --printf='%s\n' output.mp3  # expect ~1.5MB per 90s
```

## Session History

- 2026-09-17 — Piece 2 "distance as absence" (A₁, 180s) composed using this pattern for the trilogy goal. File: `songs/piece2-distance-as-absence.mp3`.
- 2026-09-16 — Piece 1 "where-memory-rest" (D₂, 150s) via the memory-forgetting-drone.md fallback.
