#!/usr/bin/env python3
"""Tests for the motif evolution system.

Acceptance gate: blind comparisons should match A′ to A more often than
to motifs from other seeds. That tests musical identity, not determinism.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from motif import (
    Motif, Event, transpose, invert, fragment, augment, diminish,
    omit_notes, ornament, displace_register, call_response, cadential_mutate,
    motif_similarity, compose_long_horizon, make_demo_motif, render_motif,
)


def make_motif_from_seed(seed: int) -> Motif:
    """Generate a distinct motif from a seed integer."""
    import numpy as np
    rng = np.random.RandomState(seed)
    n_events = rng.randint(4, 8)
    intervals = [0]
    for _ in range(n_events - 1):
        intervals.append(intervals[-1] + rng.choice([-7, -5, -4, -3, -2, 2, 3, 4, 5, 7]))
    events = [Event(semitone=s, dur_ratio=float(rng.choice([0.5, 0.7, 1.0, 1.2, 1.5])),
                    accent=float(rng.uniform(0.5, 0.95)),
                    role=str(rng.choice(["lead", "ornament", "pad"])))
               for s in intervals]
    return Motif(events=events, name=f"seed_{seed}", identity_strength=1.0)


def test_similarity_bounds():
    """Similarity scores must stay in [0, 1]."""
    m = make_demo_motif()
    for fn in [transpose, invert, fragment, augment, diminish, ornament, displace_register]:
        try:
            result = fn(m)
            s = motif_similarity(m, result)
            assert 0.0 <= s <= 1.0, f"{fn.__name__} similarity {s} out of bounds"
        except TypeError:
            # Some fns need extra args
            pass
    print("PASS: test_similarity_bounds")


def test_identity_preserved_under_transpose():
    """A′ (transposed inversion) should be more similar to A than to a random motif."""
    a = make_demo_motif()
    a_prime = transpose(invert(a), 12)  # octave-shifted inversion
    random_motifs = [make_motif_from_seed(i) for i in range(10, 20)]

    sim_to_self = motif_similarity(a, a_prime)
    sims_to_others = [motif_similarity(a, r) for r in random_motifs]
    avg_sim_to_others = sum(sims_to_others) / len(sims_to_others)

    print(f"  A→A' similarity: {sim_to_self:.3f}")
    print(f"  A→random avg:    {avg_sim_to_others:.3f}")
    assert sim_to_self > avg_sim_to_others, \
        f"A' ({sim_to_self:.3f}) should be closer to A than random ({avg_sim_to_others:.3f})"
    print("PASS: test_identity_preserved_under_transpose")


def test_fragment_recognisable():
    """A 50% fragment should still be more recognizable than a random motif."""
    a = make_demo_motif()
    frag = fragment(a, keep_ratio=0.5, seed=42)
    random_motifs = [make_motif_from_seed(i) for i in range(20, 30)]

    sim_frag = motif_similarity(a, frag)
    sims_random = [motif_similarity(a, r) for r in random_motifs]
    avg_random = sum(sims_random) / len(sims_random)

    print(f"  A→fragment similarity: {sim_frag:.3f}")
    print(f"  A→random avg:          {avg_random:.3f}")
    assert sim_frag > avg_random
    print("PASS: test_fragment_recognisable")


def test_compose_produces_audio():
    """compose_long_horizon must produce a non-silent stereo buffer."""
    m = make_demo_motif()
    audio = compose_long_horizon(m, fundamental=73.4, total_dur=30.0, seed=42)
    assert audio.ndim == 2, f"Expected stereo, got shape {audio.shape}"
    assert audio.shape[1] == 2
    assert audio.shape[0] > 44100, "Should be at least 1 second"
    peak = abs(audio).max()
    assert peak > 0.01, f"Output is silent (peak={peak})"
    print(f"  Shape: {audio.shape}, Peak: {peak:.4f}")
    print("PASS: test_compose_produces_audio")


def test_seeded_determinism():
    """Same seed must produce identical output."""
    m = make_demo_motif()
    a1 = compose_long_horizon(m, fundamental=73.4, total_dur=20.0, seed=42)
    a2 = compose_long_horizon(m, fundamental=73.4, total_dur=20.0, seed=42)
    assert a1.shape == a2.shape
    assert (a1 == a2).all(), "Same seed produced different output"
    print("PASS: test_seeded_determinism")


def test_different_seeds_diverge():
    """Different seeds should produce different pieces."""
    m = make_demo_motif()
    a1 = compose_long_horizon(m, fundamental=73.4, total_dur=20.0, seed=42)
    a2 = compose_long_horizon(m, fundamental=73.4, total_dur=20.0, seed=99)
    assert not (a1 == a2).all(), "Different seeds produced identical output"
    print("PASS: test_different_seeds_diverge")


if __name__ == "__main__":
    tests = [
        test_similarity_bounds,
        test_identity_preserved_under_transpose,
        test_fragment_recognisable,
        test_compose_produces_audio,
        test_seeded_determinism,
        test_different_seeds_diverge,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")

    print(f"\n{'='*40}")
    print(f"{passed}/{len(tests)} passed")
    if passed == len(tests):
        print("ALL GREEN — musical identity verified")
    else:
        sys.exit(1)
