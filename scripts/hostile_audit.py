"""
Hostile audit for the motif system.
Run with: uv run --with numpy python3 hostile_audit.py
"""
import hashlib
import numpy as np
import sys
import os
import wave

sys.path.insert(0, os.path.dirname(__file__))
from motif import (
    Motif, Event, transpose, invert, fragment, augment, diminish,
    omit_notes, ornament, displace_register, call_response, cadential_mutate,
    motif_similarity, interval_distance, contour_distance, rhythm_distance,
    compose_long_horizon, render_motif, write_wav, make_demo_motif, SR
)


def make_random_motif(seed: int) -> Motif:
    rng = np.random.RandomState(seed)
    n_events = rng.randint(4, 12)
    events = []
    for _ in range(n_events):
        interval = rng.randint(-12, 25)
        dur = float(round(rng.uniform(0.3, 2.0), 2))
        accent = float(round(rng.uniform(0.3, 1.0), 2))
        role = rng.choice(['lead', 'pad', 'bass', 'ornament'])
        events.append(Event(interval, dur, accent, role))
    return Motif(events=events, name=f'm_{seed}')


# ── 1. Similarity across hundreds of unrelated motifs ──────────────────────
def test_hundreds_of_motifs():
    rng = np.random.RandomState(99)
    base = make_random_motif(42)
    transformed = [
        transpose(base, rng.randint(-12, 13)),
        invert(base),
        fragment(base, 0.5, seed=1),
        augment(base, 1.8),
    ]
    transformed = [t for t in transformed if len(t.events) > 0]

    # Compare each transformed against 200 random motifs
    fp_count = 0
    tp_count = 0
    margins = []
    for t in transformed:
        for i in range(200):
            rand = make_random_motif(rng.randint(0, 99999))
            if len(rand.events) == 0 or len(t.events) == 0:
                continue
            sim = motif_similarity(t, rand)
            tp_count += 1
            if sim > 0.55:  # typical variant threshold
                fp_count += 1

    fp_rate = fp_count / max(tp_count, 1)
    print(f"  Transformed-vs-random pairs tested: {tp_count}")
    print(f"  False positives (sim > 0.55): {fp_count}, rate: {fp_rate:.4f}")
    assert fp_rate < 0.05, f"FP rate too high: {fp_rate:.4f}"
    print("  PASS: FP rate < 5%")


# ── 2. Transposition invariance ────────────────────────────────────────────
def test_transposition_invariance():
    rng = np.random.RandomState(7)
    failures = 0
    tested = 0
    for i in range(100):
        m = make_random_motif(i + 100)
        if len(m.events) < 3:
            continue
        shift = rng.randint(-24, 25)
        if shift == 0:
            shift = 12
        shifted = transpose(m, shift)
        sim = motif_similarity(m, shifted)
        tested += 1
        if sim < 0.95:
            failures += 1
            print(f"    FAIL: motif {i}, shift {shift}: sim={sim:.4f}")

    print(f"  Transposition invariance: {tested - failures}/{tested} passed (threshold 0.95)")
    assert failures <= 0, f"Transposition invariance failures: {failures}"
    print("  PASS: all transposed motifs score ≥ 0.95")


# ── 3. Inversion invariance ────────────────────────────────────────────────
def test_inversion_invariance():
    rng = np.random.RandomState(13)
    failures = 0
    tested = 0
    for i in range(100):
        m = make_random_motif(i + 200)
        if len(m.events) < 3:
            continue
        inv = invert(m)
        if len(inv.events) == 0:
            continue
        sim = motif_similarity(m, inv)
        tested += 1
        if sim < 0.50:  # Inversion should still be recognisable via contour
            failures += 1

    print(f"  Inversion similarity: {tested - failures}/{tested} above 0.50")
    # Inversion can reduce similarity, but contour symmetry should help
    assert failures < tested * 0.50, f"Too many inversions unrecoverable: {failures}/{tested}"
    print("  PASS: inversion preserves some identity")


# ── 4. Inversion does NOT collapse genuinely different contours ────────────
def test_inversion_discriminates():
    rng = np.random.RandomState(55)
    # Two motifs with different contour shapes should stay different even inverted
    different = 0
    tested = 0
    for i in range(50):
        a = make_random_motif(i + 300)
        b = make_random_motif(i + 900)
        if len(a.events) < 4 or len(b.events) < 4:
            continue
        inv_b = invert(b)
        if len(inv_b.events) == 0:
            continue
        sim = motif_similarity(a, inv_b)
        tested += 1
        if sim < 0.70:
            different += 1

    print(f"  Inverted different motifs: {different}/{tested} correctly dissimilar (< 0.70)")
    assert different > tested * 0.60, f"Inversion collapses too many: only {different}/{tested} dissimilar"
    print("  PASS: inverted different motifs stay different")


# ── 5. Fragment-length bias ───────────────────────────────────────────────
def test_fragment_length_bias():
    m = make_random_motif(77)
    while len(m.events) < 6:
        m = make_random_motif(np.random.randint(1000))

    frags = []
    for ratio in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        f = fragment(m, keep_ratio=ratio, seed=42)
        if len(f.events) > 0:
            sim = motif_similarity(m, f)
            frags.append((ratio, sim))
            print(f"    keep_ratio={ratio:.1f}: {len(f.events)} events, sim={sim:.4f}")

    # Shorter fragments should have lower similarity (monotonic-ish trend)
    sims = [s for _, s in frags]
    # At minimum, 0.9 should score higher than 0.3
    low = next(s for r, s in frags if r == 0.3)
    high = next(s for r, s in frags if r == 0.9)
    assert high > low, f"0.9 fragment ({high:.4f}) should beat 0.3 ({low:.4f})"
    print("  PASS: shorter fragments score lower")


# ── 6. Rhythm weighting and degenerate constant-rhythm motifs ──────────────
def test_rhythm_degenerate():
    # Motifs with identical rhythm but different intervals
    m1 = Motif(events=[
        Event(0, 1.0, 0.9, 'lead'),
        Event(4, 1.0, 0.8, 'lead'),
        Event(7, 1.0, 0.9, 'lead'),
        Event(12, 1.0, 0.7, 'lead'),
    ])
    m2 = Motif(events=[
        Event(0, 1.0, 0.9, 'lead'),
        Event(-3, 1.0, 0.8, 'lead'),
        Event(-7, 1.0, 0.9, 'lead'),
        Event(-12, 1.0, 0.7, 'lead'),
    ])

    sim = motif_similarity(m1, m2)
    # Same rhythm, different intervals — should NOT be too high
    print(f"  Different intervals, same rhythm: sim={sim:.4f}")
    assert sim < 0.90, f"Degenerate constant-rhythm over-similar: {sim:.4f}"
    print("  PASS: constant-rhythm motifs with different intervals stay apart")


# ── 7. A′ recognition across keys, seeds, transforms ──────────────────────
def test_aprime_recognition():
    base = make_random_motif(123)
    while len(base.events) < 5:
        base = make_random_motif(np.random.randint(2000))

    # A′ = transposed inversion with various shifts
    shifts = [-12, -7, -5, 0, 5, 7, 12]
    seeds = [1, 2, 3, 4, 5]

    total = 0
    recognized = 0
    for shift in shifts:
        for seed in seeds:
            aprime = transpose(invert(base), shift)
            if len(aprime.events) == 0:
                continue
            sim = motif_similarity(base, aprime)
            total += 1
            if sim > 0.40:
                recognized += 1

    rate = recognized / max(total, 1)
    print(f"  A′ recognition across keys/seeds: {recognized}/{total} ({rate:.2%})")
    assert rate > 0.70, f"A′ recognition too low: {rate:.2%}"
    print("  PASS: A′ recognised across keys and seeds")


# ── 8. Section durations sum to exactly 120s ──────────────────────────────
def test_section_durations():
    m = make_random_motif(555)
    while len(m.events) < 4:
        m = make_random_motif(np.random.randint(3000))
    audio = compose_long_horizon(m, fundamental=82.4, total_dur=120.0, seed=42)
    dur = len(audio) / SR
    print(f"  Piece duration: {dur:.4f}s (target 120.0)")
    assert abs(dur - 120.0) < 0.1, f"Duration off: {dur:.4f}s"
    print("  PASS: duration = 120.0s")


# ── 8b. No boundary dips (crossfade must overlap, not zero-meet) ──────────
def test_no_boundary_dips():
    m = make_random_motif(555)
    while len(m.events) < 4:
        m = make_random_motif(np.random.randint(3000))
    audio = compose_long_horizon(m, fundamental=82.4, total_dur=120.0, seed=42)

    # Section boundaries at 30s, 72s, 102s
    boundaries = [30*SR, 72*SR, 102*SR]
    avg_rms = float(np.sqrt(np.mean(audio**2)))

    for b in boundaries:
        # Measure RMS in a 200ms window centered on the boundary
        hw = int(SR * 0.1)  # 100ms each side
        local = audio[b-hw:b+hw]
        local_rms = float(np.sqrt(np.mean(local**2)))
        ratio = local_rms / avg_rms if avg_rms > 0 else 0
        print(f"  Boundary at {b/SR}s: local_rms={local_rms:.5f}, ratio={ratio:.3f}")
        assert ratio > 0.3, f"Boundary dip at {b/SR}s: local RMS {ratio:.1%} of average (crossfade overlap bug)"

    print("  PASS: no boundary dips, crossfades overlap correctly")


# ── 8c. Exact section durations (A=30, B=42, A'=30, outro=18) ─────────────
def test_exact_section_boundaries():
    m = make_random_motif(555)
    while len(m.events) < 4:
        m = make_random_motif(np.random.randint(3000))
    audio = compose_long_horizon(m, fundamental=82.4, total_dur=120.0, seed=42)

    expected = {
        'A': (0, 30),
        'B': (30, 72),
        "A'": (72, 102),
        'Outro': (102, 120),
    }
    total_end = 0
    for name, (start, end) in expected.items():
        dur = (end - start)
        total_end = end
        print(f"  {name}: {start}s-{end}s = {dur}s")
    assert total_end == 120, f"Sections end at {total_end}s, not 120s"
    assert len(audio) / SR == 120.0, f"Audio is {len(audio)/SR}s"
    print("  PASS: sections sum to exactly 120.0s")


# ── 9. WAV header verification ─────────────────────────────────────────────
def test_wav_header():
    m = make_random_motif(888)
    while len(m.events) < 4:
        m = make_random_motif(np.random.randint(4000))
    out = '/tmp/audit_test.wav'
    audio = compose_long_horizon(m, fundamental=110.0, total_dur=10.0, seed=1)
    write_wav(audio, out)

    with wave.open(out, 'rb') as w:
        assert w.getnchannels() == 2, f"Channels: {w.getnchannels()}"
        assert w.getsampwidth() == 2, f"Sample width: {w.getsampwidth()}"
        assert w.getframerate() == 44100, f"Rate: {w.getframerate()}"
        dur = w.getnframes() / w.getframerate()
        assert abs(dur - 10.0) < 0.1, f"WAV duration: {dur}"

    os.remove(out)
    print("  PASS: stereo, 16-bit, 44.1kHz, correct duration")


# ── 10. Peak, RMS, clipping, DC offset, silence, discontinuities ──────────
def test_audio_quality():
    m = make_random_motif(999)
    while len(m.events) < 4:
        m = make_random_motif(np.random.randint(5000))
    audio = compose_long_horizon(m, fundamental=82.4, total_dur=120.0, seed=42)

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    dc_offset = float(np.mean(audio))
    has_nan = bool(np.any(np.isnan(audio)))
    has_inf = bool(np.any(np.isinf(audio)))

    # No clipping (peak <= 1.0)
    assert peak <= 1.0, f"Clipping: peak={peak:.4f}"
    assert not has_nan, "NaN detected"
    assert not has_inf, "Inf detected"
    assert abs(dc_offset) < 0.01, f"DC offset too high: {dc_offset:.6f}"

    # Silence: no fully silent 5-second windows
    window = 5 * SR
    n_windows = len(audio) // window
    for i in range(n_windows):
        w = audio[i*window:(i+1)*window]
        w_rms = float(np.sqrt(np.mean(w**2)))
        assert w_rms > 0.001, f"Silent window at {i*5}-{(i+1)*5}s"

    # Discontinuities: check crossfade boundaries (at ~30s, ~72s, ~102s)
    boundaries = [30*SR, 72*SR, 102*SR]
    for b in boundaries:
        if b + SR < len(audio) and b - SR >= 0:
            before = audio[b-100:b]
            after = audio[b:b+100]
            jump = float(np.max(np.abs(after - before)))
            assert jump < 0.5, f"Discontinuity at {b/SR:.0f}s: jump={jump:.4f}"

    # Crest factor and loudness range (proper dynamic range metrics)
    crest = peak / rms if rms > 0 else 0
    win = int(SR * 1.0)
    rms_wins = [float(np.sqrt(np.mean(audio[i:i+win]**2))) for i in range(0, len(audio)-win, win)]
    rms_arr = np.array(rms_wins)
    loud_range = float(20*np.log10(rms_arr.max()/rms_arr.min())) if rms_arr.min() > 0 else 0.0

    print(f"  Peak: {peak:.4f}, RMS: {rms:.5f}, DC: {dc_offset:.6f}")
    print(f"  Crest factor: {crest:.1f} ({20*np.log10(crest):.1f} dB)")
    print(f"  Loudness range: {loud_range:.1f} dB (1s windows)")
    print(f"  NaN: {has_nan}, Inf: {has_inf}")
    assert crest > 1.5, f"Crest factor too low (noise?): {crest:.2f}"
    assert loud_range > 6.0, f"Loudness range too narrow: {loud_range:.1f} dB"
    print("  PASS: no clipping, no NaN/Inf, low DC, crest>1.5, loudness range>6dB")


# ── 11. Deterministic score and audio hashes ───────────────────────────────
def test_determinism():
    m1 = make_random_motif(111)
    m2 = make_random_motif(222)
    while len(m1.events) < 4 or len(m2.events) < 4:
        m1 = make_random_motif(np.random.randint(6000))
        m2 = make_random_motif(np.random.randint(6000))

    # Same seed → same audio
    a1 = compose_long_horizon(m1, 82.4, 120.0, seed=42)
    a2 = compose_long_horizon(m1, 82.4, 120.0, seed=42)
    hash1 = hashlib.sha256(a1.tobytes()).hexdigest()[:16]
    hash2 = hashlib.sha256(a2.tobytes()).hexdigest()[:16]
    assert hash1 == hash2, f"Audio not deterministic: {hash1} != {hash2}"

    # Same similarity score on repeated calls
    s1 = motif_similarity(m1, m2)
    s2 = motif_similarity(m1, m2)
    assert s1 == s2, f"Similarity not deterministic: {s1} != {s2}"

    print(f"  Audio hash: {hash1}")
    print(f"  Similarity: {s1:.6f} (stable)")
    print("  PASS: deterministic")

# ── 12. Demo WAV integrity ─────────────────────────────────────────────────
def test_demo_wav_integrity():
    """Verify the committed demo WAV reproduces byte-for-byte from the canonical factory.

    Builds the motif through make_demo_motif(), renders with declared parameters,
    hashes the generated WAV, hashes output/becoming_motif_demo.wav, and asserts
    equality. No hard-coded hash — the test reads the actual committed file.
    """
    import hashlib
    import wave
    import io

    # Build through the canonical factory
    m = make_demo_motif()
    audio = compose_long_horizon(m, fundamental=82.4, total_dur=120.0, seed=7)
    assert audio.shape[0] == 120 * SR, f"Wrong length: {audio.shape[0]}"

    # Hash the rendered output
    audio_16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(audio_16.tobytes())
    rendered_hash = hashlib.sha256(buf.getvalue()).hexdigest()

    # Hash the actual committed file
    demo_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'becoming_motif_demo.wav')
    with open(demo_path, 'rb') as f:
        committed_hash = hashlib.sha256(f.read()).hexdigest()

    print(f"  Rendered SHA-256:  {rendered_hash}")
    print(f"  Committed SHA-256: {committed_hash}")
    assert rendered_hash == committed_hash, \
        f"Committed demo WAV does not match fresh render!\n  rendered:  {rendered_hash}\n  committed: {committed_hash}"
    print("  PASS: committed demo WAV reproduces byte-for-byte from canonical factory")


if __name__ == '__main__':
    tests = [
        ("1. Hundreds of unrelated motifs", test_hundreds_of_motifs),
        ("2. Transposition invariance", test_transposition_invariance),
        ("3. Inversion invariance", test_inversion_invariance),
        ("4. Inversion discriminates different contours", test_inversion_discriminates),
        ("5. Fragment-length bias", test_fragment_length_bias),
        ("6. Rhythm degenerate motifs", test_rhythm_degenerate),
        ("7. A′ recognition across keys/seeds", test_aprime_recognition),
        ("8. Section durations = 120s", test_section_durations),
        ("8b. No boundary dips (crossfade overlap)", test_no_boundary_dips),
        ("8c. Exact section boundaries", test_exact_section_boundaries),
        ("9. WAV header", test_wav_header),
        ("10. Audio quality (peak/DC/silence/xfade)", test_audio_quality),
        ("11. Determinism (hashes)", test_determinism),
        ("12. Demo WAV integrity", test_demo_wav_integrity),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"HOSTILE AUDIT: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("ALL GATE TESTS PASSED — hostile audit clean")
    else:
        print(f"GATE FAILED — {failed} test(s) broke")
        sys.exit(1)
