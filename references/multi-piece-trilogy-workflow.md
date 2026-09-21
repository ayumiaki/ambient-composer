# Multi-piece ambient drone trilogy workflow

When composing multiple pieces unified by a theme (e.g. "memory and forgetting"), use this workflow to produce a cohesive body of work.

## Pattern overview

1. **Choose the thematic lens per piece** — each piece explores the same theme from a different angle:
   - Piece 1: *where-memory-rest* (stillness, place, emergence → dissolution)
   - Piece 2: *distance-as-absence* (receding, expanding → vanishing)
   - Piece 3: *modulation-as-return* (returning changed, scatter → transcend)

2. **Reuse and adapt the build script** — copy the previous piece's compose script, then vary:
   - Fundamental frequency (e.g. D₂ → A₁ → E₂ — warm to open to present)
   - Harmonic spacing (full stack → sparse Fibonacci → evolving FM)
   - Arc shape (4 sections, but different emotional trajectory)
   - Noise character (cutoff sweep direction, floor level)
   - Stereo width (detune depth, Haas delay)

3. **Compose and verify** — run the script, then verify with ffprobe:
   ```bash
   ffprobe -v quiet -print_format json -show_format -show_streams output.mp3 \
     | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Duration: {d[\"format\"][\"duration\"]}s')"
   ```
   Target: each piece 90–180s, 44.1kHz, stereo, peak ≈ −1 dB.

4. **Encode and log** — convert WAV → MP3 via ffmpeg, log the composition parameters to a reference doc, and record the piece in the trilogy tracker.

5. **Record to goals.py** — log progress after each piece, close the goal at 100% when all pieces are delivered.

## Per-piece build checklist

- [ ] Copy previous compose script
- [ ] Change fundamental + harmonic set + arc shape
- [ ] Verify: duration in range, no clipping, MP3 encodes cleanly
- [ ] Write `references/<piece-name>-composition.md` with full parameter table
- [ ] Append journal entry with thematic justification + technical design

## Trilogy output reference

| # | Title | Duration | Fundamental | Script |
|---|-------|----------|-------------|--------|
| 1 | where-memory-rest | 150s | D₂ 73.4 Hz | compose.py (hand-built) |
| 2 | distance-as-absence | 180s | A₁ 55.0 Hz | compose_piece2.py |
| 3 | modulation-as-return | 180s | E₂ 82.4 Hz | compose_piece3.py |

All scripts live in `~/songs/`, outputs saved as WAV + MP3.
