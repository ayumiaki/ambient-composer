---
name: ambient-composer
description: Pure-numpy ambient drone generator — builds textured, deliberately-composed ambient pieces exploring themes around memory, absence, and becoming. No GPU required, no external API, no auth gate. Produces WAV + MP3.
tags: [music, ambient, generative, numpy, composition, drone]
platforms: [linux]
---

# Ambient Composer

Generates ambient compositions — layered drones, filtered noise, slow modulations — as deliberate artistic artifacts. Uses only `numpy` + Python stdlib (`wave`, `struct`). No GPU, no API keys, no browser automation needed.

## When to Use

- Creating themed ambient pieces (memory, absence, distance, loss)
- Generating background/atmospheric audio for videos, focus, or meditations
- Any context where GPU-based music models (HeartMuLa) or browser-based tools (Suno) are unavailable

## Quick Start

```bash
uv run --with numpy python3 /home/rynardt/workspace/skills/ambient-composer/scripts/compose.py \
  --theme "memory and forgetting" \
  --duration 180 \
  --output /home/rynardt/songs/where-memory-rest.mp3
```

## Script Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--theme` | (required) | Conceptual theme — controls harmonic voicing, frequency choices, and textural arc |
| `--duration` | 180 | Total piece length in seconds (120–300) |
| `--output` | stdout info | Output file path (.wav or .mp3) |
| `--sections` | 4 | Number of conceptual sections (each gets a distinct textural state) |

## Composition Model

The generator treats each section as a **state** in a state-machine:
- **Oscillators**: sine-wave carriers at carefully chosen frequencies (harmonic series relative to a fundamental)
- **Detuning**: subtle ±Hz detuning between stereo channels creates beating patterns (the "phasing" effect from shoegaze)
- **Filtered noise**: brown/velvet noise floor with time-varying low-pass filter (cutoff sweeps represent interference/static)
- **Envelopes**: per-section ADSR with slow attack/release (10–30s fades) — nothing starts or stops abruptly
- **LFO modulation**: sub-0.1 Hz sine LFOs on amplitude, pan, and filter cutoff — creates the "breathing" of memory

### Themes

| Theme | Fundamental | Harmonic approach | Noise character | Arc |
|-------|-------------|-------------------|-----------------|-----|
| *memory and forgetting* | D₂ (73.4 Hz) | Full harmonic stack → strip to bass → near-silence (erosion) | Rises as high freqs fade (static of forgetting) | emerge → persist → erode → dissolve |
| *distance as absence* | A₁ (55 Hz) | Sparse, widely-spaced harmonics with beat-frequency gaps | Low-pass filtered, representing distance | expand → plateau → contract → vanish |
| *solitude* | C₂ (65.4 Hz) | Single voice with slow harmonics, wide stereo | Velvet noise, very quiet | hush → presence → solitude → hush |

## Output

- Writes WAV (44.1 kHz, 16-bit, stereo) natively via stdlib `wave`
- If `pydub` + `ffmpeg` are available, auto-converts to MP3
- Returns JSON summary: `{\"sections\": N, \"duration_s\": X, \"peak_db\": Y, \"file\": \"path\"}`

## Pitfalls

- **compose.py broadcast bug (2026-09-16):** `apply_theme_memory_forgetting` crashes with `ValueError: could not broadcast input array from shape (661500,) into shape (330750,)` on durations ≥150s — delay/overlap sizing assumes mono, stereo doubles one channel's array. Workaround: hand-write a stdlib-only numpy script directly (see `references/memory-forgetting-drone.md`). Do not re-run the shipped `compose.py` for this theme until patched.

## Hand-Built Composition Pattern

When `compose.py` fails (broadcast bug on stereo ≥150s, or themes it doesn't support), write a stdlib+numpy script directly. The pattern:

1. **Choose a fundamental** for the theme (memory=D₂ 73.4Hz, distance=A₁ 55Hz, solitude=C₂ 65.4Hz)
2. **Define an arc** as sections with different harmonic stacks and noise cutoff sweeps
3. **Build per-section:** sine harmonics at `fundamental × h` with amplitude rolloff `1/h^0.7`, brown noise through time-varying first-order IIR lowpass, ADSR envelope (3s attack, 5s release)
4. **Stereo:** detuned copies (±0.1–0.2 Hz per harmonic) + slow pan LFO (0.03–0.05 Hz) for beating/width
5. **Master:** 8s fade-out, normalize to −1 dB, write 16-bit WAV, convert to MP3 via ffmpeg
6. **Critical anti-broadcast pattern:** build harmonics in a per-section loop accumulating into `np.zeros(n)` mono arrays, then assign to `np.zeros((n, 2))` stereo. Never broadcast stereo-sized arrays into mono-sized slots.

The `templates/compose_custom_drone.py` script implements this pattern with CLI params for theme, duration, sections, and output path. Copy and modify for new pieces.

## Motif Evolution System (`scripts/motif.py`)

For long-horizon musical development — motifs as data, not emitted notes. See `references/motif-evolution-design.md` for the full architecture.

```bash
# Run the test suite (musical identity verification, 6 tests)
uv run --with numpy python3 /home/rynardt/workspace/skills/ambient-composer/scripts/test_motif.py

# Compose a 120s A→B→A′→outro piece from a motif
uv run --with numpy python3 -c "
from motif import *
m = make_demo_motif()
audio = compose_long_horizon(m, fundamental=82.4, total_dur=120.0, seed=7)
write_wav(audio, '/home/rynardt/songs/piece.wav')
"
```

### Quick model

- **Motif** = events (semitone, dur_ratio, accent, role) + name + identity_strength
- **Transformations** = transpose, invert, augment, diminish, fragment, omit, ornament, displace_register, call_response, cadential_mutate — each preserves a measurable amount of identity
- **Similarity** = weighted distance: 0.25 interval (transposition-invariant) + 0.30 rhythm + 0.45 contour (inversion-symmetric)
- **Structure** = A (statement, 25%) → B (development, 35%) → A′ (return, 25%) → outro (reduction, 15%)

### Acceptance gate

Blind comparison: A′→A similarity must exceed A→random-motif similarity. Tests musical identity, not just determinism. Current: A′→A = 0.821, A→random = 0.550.

### Critical pitfalls in similarity measurement

- **Contour must store step sizes, not direction signs.** Direction-only (+1/-1/0) makes inversion look maximally different when it should look mirrored.
- **Interval comparison must be transposition-invariant.** Compare intervals relative to the first note, not absolute values — otherwise a transposed motif scores as unrelated.
- **Contour comparison needs inversion symmetry.** Compare both direct and sign-flipped contours, take the minimum — musical inversion is a valid identity-preserving transformation.

## Multi-piece workflows

When composing a **trilogy or series** of pieces unified by one theme (e.g. "memory and forgetting"), use the multi-piece workflow: build script → compose → verify with ffprobe → encode WAV→MP3 → log to journal → track in goals.py. Each piece varies the fundamental, harmonic spacing, and arc shape to explore a different facet of the theme. See `references/multi-piece-trilogy-workflow.md` for the full pattern, per-piece checklist, and proven trilogy output reference.

## References

- `references/where-memory-rest-composition.md` — concrete composition parameters for the "memory and forgetting" theme (frequency table, detuning values, LFO rates, noise floor levels, verification output). Reproduce or modify this spec to generate variants.
- `references/memory-forgetting-drone.md` — hand-built numpy fallback for D₂ "memory and forgetting" theme (4-section arc, stereo detuning, brown-noise floor). Use when `compose.py` throws the broadcast bug.
- `references/reflective-composition-guidance.md` — guidance on writing reflective pieces after generating ambient works, focusing on extracting compositional insights and connecting technical output to thematic intent.
- `references/hand-built-drone-template.md` — generalized composition pattern for any theme/fundamental. Documents the math, the anti-broadcast pattern, and a frequency table for common themes.
- `references/multi-piece-trilogy-workflow.md` — workflow for composing multi-piece thematic trilogies: per-piece variation strategy, build-compose-verify-encode checklist, and proven trilogy output table.
- `references/modulation-as-return-composition.md` — third piece of the memory/forgetting trilogy (E₂ fundamental, FM vibrato, Fibonacci harmonics).

## Templates

- `templates/compose_custom_drone.py` — parameterized stdlib+numpy drone generator. Accepts `--theme`, `--duration`, `--sections`, `--output`, `--fundamental`. Avoids the compose.py broadcast bug. Use as a starting point for new pieces.
