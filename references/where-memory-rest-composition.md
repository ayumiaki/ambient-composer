# Where Memory Rests — Composition Parameters

Concrete instance of the `ambient-composer` theme "memory and forgetting", generated 2026-09-12.

## Theme Mapping

| Section | Start | End | Frequencies | Noise | Detuning | LFO |
|---------|-------|-----|-------------|-------|----------|-----|
| Emergence | 0s | 45s | 73 Hz (only) | -60 dB, LP 80 Hz | 0 Hz | Amp LFO 0.005 Hz ±10% |
| Persistence | 45s | 90s | 73, 110, 147, 175, 220, 294, 349 Hz | -23 dB, LP 400 Hz | ±0.3 Hz | Amp LFO 0.02 Hz ±15% |
| Erosion | 90s | 135s | 73, 110, 147, 175, 220 Hz (top 2 fading) | -18 dB, LP sweep 500→120 Hz | ±0.8 Hz | Filter LFO 0.015 Hz ±30% |
| Dissolution | 135s | 180s | 73 Hz only (steep x^1.5/x^2.0 fades) | -60 dB, LP 80 Hz | 0 Hz | Amp LFO 0.008 Hz ±20% |

## Harmonic Weights (Persistence section)

| Freq (Hz) | Note | Weight | Stereo placement |
|-----------|------|--------|-----------------|
| 73.42 | D₂ | 25% | Center |
| 110.0 | A₂ | 18% | Center |
| 147.83 | D₃ | 14% | Center |
| 174.61 | F₃ | 11% | Center |
| 220.0 | A₃ | 8% | Center |
| 293.66 | D₄ | 5% | Fading (starts at 90s) |
| 349.23 | F₄ | 3% | Fading (starts at 90s) |

## Key Technique Notes (discovered during this session)

1. **IIR filter for brown noise** — vectorized first-order low-pass is sufficient for ambient texture. No need for FFT-based filtering.
2. **Stereo beating** — detune left channel +X Hz, right channel -X Hz. The resulting beat frequency = 2X Hz. ±0.3 Hz detuning → 0.6 Hz beating (slow enough to feel like breath, not enough to sound like a phaser).
3. **Brown noise generation** — cumulative sum of white noise, then normalised to unit RMS. Sounds warmer than white noise (1/f² spectrum).
4. **Curved fade-out** — apply x^1.5 (slow start) or x^2.0 (accelerating end) to the amplitude envelope, not a linear fade. Makes the final dissolution feel natural rather than truncated.
5. **Seed determinism** — seed=42 for reproducible noise. Critical for re-generation matching.
6. **Peak normalisation** — normalise to -1 dBFS (not 0) to prevent inter-sample clipping during MP3 encoding.

## Verification Output

```
$ ffprobe /home/rynardt/songs/where-memory-rest.mp3
Duration: 180.000s, 44100 Hz, stereo, MP3, peak=-1.84 dB
```
