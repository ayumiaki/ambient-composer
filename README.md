# Ambient Composer

Pure-numpy ambient drone generator with a musical motif system.

## Architecture

`scripts/motif.py` — the core:
- **Motif data model**: intervals, rhythmic ratios, accent, contour (step-sizes), harmonic roles, identity strength
- **Transformations**: transpose, invert, augment, diminish, fragment, omit, ornament, register displacement, call-response, cadential mutation
- **Similarity scoring**: weighted sum of transposition-invariant interval distance (0.35), inversion-symmetric contour distance (0.45), rhythm distance (0.20)
- **Composition**: A → B → A′ → Outro with inter-section crossfades

## Results

| Class | Min | Max | Mean |
|-------|-----|-----|------|
| Transformed variants | 0.553 | 1.000 | **0.856** |
| Random motifs | 0.125 | 0.322 | **0.202** |

Separation: **0.653**. Worst fragment variant (0.553) beats best random (0.322).

## Run

```bash
cd scripts
uv run --with numpy python3 test_motif.py
```

## Output

`output/becoming_motif_demo.wav` — 120s stereo 16-bit @ 44.1kHz
