# modulation-as-return — Composition Parameters

Third piece of the ambient drone trilogy "memory and forgetting", composed 2026-09-17.

## Theme Mapping

The trilogy completes: 
1. **where-memory-rest** (piece 1) — memory at rest, place, stillness
2. **distance-as-absence** (piece 2) — memory receding, distance, vanishing  
3. **modulation-as-return** (piece 3) — memory returning *changed*, modulation as transformation

Piece 3 argues: forgetting isn't an ending. What returns is not what left. Time modulates everything.

## Technical Design

| Section | Start | End | Frequencies | Noise | Detuning | FM Depth | FM Rate |
|---------|-------|-----|-------------|-------|----------|----------|---------|
| Scatter | 0s | 45s | 82, 330, 577, 906 Hz | LP 200→800 Hz | ±0.15 Hz | 8.0 Hz | 0.07 Hz |
| Gather | 45s | 90s | 82, 165, 247, 412, 659 Hz | LP 800→1500 Hz | ±0.18 Hz | 5.0 Hz | 0.05 Hz |
| Transform | 90s | 135s | 82, 165, 247, 412, 659, 1071 Hz | LP 1500→2000 Hz | ±0.20 Hz | 6.0 Hz | 0.09 Hz |
| Transcend | 135s | 180s | 82, 247, 412 Hz | LP 2000→300 Hz | ±0.10 Hz | 3.0 Hz | 0.04 Hz |

- **Fundamental:** E₂ 82.4 Hz — warmer, more present than pieces 1 (D₂ 73.4) and 2 (A₁ 55.0)
- **FM (frequency modulation):** Each section has its own vibrato depth and rate, creating an evolving sense of "the same note but different"
- **Stereo:** Left/right detuned ±section_detune, 20ms Haas delay on right noise channel for width
- **Harmonic scatter:** Fibonacci-spaced harmonics (1,2,3,5,8,13) create organic non-tonal clusters
- **Arc:** Sparse/bright → dense/warm → fullest/brightest again → stripping back but *different* from start

## Key Techniques

1. **FM vibrato** — `phase = 2π·f·t - (depth/rate)·cos(2π·rate·t)` gives a natural pitch waversections that evolves per section
2. **Fibonacci harmonics** — spacing of 1,2,3,5,8,13 gives non-integer-related partials that feel organic rather than chordal
3. **Haas effect** — 20ms delay on right channel's noise floor creates width without phasiness
4. **Filter sweep as metaphor** — cutoff rising through scatter/gather/transform, then falling in transcend = "opening up, then letting go differently"

## Verification

```
Duration: 180.0s, Codec: mp3, Rate: 44100Hz, Channels: 2, Size: 5.8MB
```

## Trilogy Summary

| # | Title | Duration | Fundamental | Character |
|---|-------|----------|-------------|-----------|
| 1 | where-memory-rest | 150s | D₂ 73.4 Hz | Resting, stillness, emergence→dissolution |
| 2 | distance-as-absence | 180s | A₁ 55.0 Hz | Receding, expanding→vanishing |
| 3 | modulation-as-return | 180s | E₂ 82.4 Hz | Returning changed, scatter→transcend |

All pieces saved to `~/songs/`.
