#!/usr/bin/env python3
"""Motif evolution system for the ambient composer.

A motif is a sequence of musical events (pitch ratio, duration ratio, accent, role).
Transformations produce recognisable variants. Similarity scoring keeps evolution
within the band: recognizable return <-> unrelated note soup.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math

SR = 44100


# ── Data model ──────────────────────────────────────────────────────────────

@dataclass
class Event:
    """Single event within a motif.
    - semitone: interval in semitones from the fundamental (can be negative)
    - dur_ratio: duration relative to motif unit length (1.0 = standard)
    - accent: amplitude shape 0.0–1.0
    - role: 'lead' | 'bass' | 'pad' | 'ornament'
    """
    semitone: int
    dur_ratio: float = 1.0
    accent: float = 0.8
    role: str = "lead"


@dataclass
class Motif:
    """A motif is a sequence of events plus identity metadata."""
    events: List[Event]
    name: str = "motif"
    identity_strength: float = 1.0  # 1.0 = fully recognisable, 0.0 = unrecognisable

    def intervals(self) -> List[int]:
        return [e.semitone for e in self.events]

    def durations(self) -> List[float]:
        return [e.dur_ratio for e in self.events]

    def accents(self) -> List[float]:
        return [e.accent for e in self.events]

    def contour(self) -> List[int]:
        """Step sizes between consecutive intervals (not just sign)."""
        ints = self.intervals()
        return [ints[i] - ints[i-1] for i in range(1, len(ints))]

    def copy(self) -> "Motif":
        return Motif(
            events=[Event(e.semitone, e.dur_ratio, e.accent, e.role) for e in self.events],
            name=self.name,
            identity_strength=self.identity_strength,
        )


# ── Transformations ─────────────────────────────────────────────────────────

def transpose(m: Motif, semitones: int) -> Motif:
    """Shift every note by a fixed interval. Preserves contour, intervals, rhythm."""
    out = m.copy()
    out.events = [Event(e.semitone + semitones, e.dur_ratio, e.accent, e.role) for e in m.events]
    out.name = f"{m.name}_T{semitones:+d}"
    out.identity_strength = m.identity_strength * 0.95
    return out


def invert(m: Motif) -> Motif:
    """Mirror all intervals around the first note. Preserves rhythm, flips contour."""
    base = m.events[0].semitone
    out = m.copy()
    out.events = [Event(base - (e.semitone - base), e.dur_ratio, e.accent, e.role) for e in m.events]
    out.name = f"{m.name}_inv"
    out.identity_strength = m.identity_strength * 0.88
    return out


def augment(m: Motif, factor: float = 2.0) -> Motif:
    """Rhythmic augmentation — stretch durations, slightly soften accents."""
    out = m.copy()
    out.events = [Event(e.semitone, e.dur_ratio * factor, max(0.4, e.accent * 0.85), e.role)
                   for e in m.events]
    out.name = f"{m.name}_aug{factor:.1f}"
    out.identity_strength = m.identity_strength * 0.90
    return out


def diminish(m: Motif, factor: float = 0.5) -> Motif:
    """Rhythmic diminution — compress durations, sharpen accents."""
    out = m.copy()
    out.events = [Event(e.semitone, max(0.125, e.dur_ratio * factor),
                         min(1.0, e.accent * 1.15), e.role) for e in m.events]
    out.name = f"{m.name}_dim{factor:.1f}"
    out.identity_strength = m.identity_strength * 0.90
    return out


def fragment(m: Motif, keep_ratio: float = 0.5, seed: int = 0) -> Motif:
    """Keep a subset of events (default: first half). Reduce identity proportionally."""
    rng = np.random.RandomState(seed)
    n = max(2, int(len(m.events) * keep_ratio))
    indices = sorted(rng.choice(len(m.events), size=n, replace=False).tolist())
    kept = [m.events[i] for i in indices]
    out = m.copy()
    out.events = [Event(e.semitone, e.dur_ratio, e.accent, e.role) for e in kept]
    out.name = f"{m.name}_frag{keep_ratio:.1f}"
    out.identity_strength = m.identity_strength * (0.5 + 0.4 * keep_ratio)
    return out


def omit_notes(m: Motif, count: int = 1, seed: int = 0) -> Motif:
    """Drop N events (default: weak accents first)."""
    rng = np.random.RandomState(seed)
    if count >= len(m.events) - 1:
        count = len(m.events) - 2
    indices = sorted(rng.choice(len(m.events), size=count, replace=False).tolist())
    kept = [e for i, e in enumerate(m.events) if i not in indices]
    out = m.copy()
    out.events = [Event(e.semitone, e.dur_ratio, e.accent, e.role) for e in kept]
    out.name = f"{m.name}_omit{count}"
    out.identity_strength = m.identity_strength * max(0.4, 1.0 - 0.15 * count)
    return out


def ornament(m: Motif, density: float = 1.0, seed: int = 0) -> Motif:
    """Insert passing tones between events. density=N inserts up to N per gap."""
    rng = np.random.RandomState(seed)
    new_events = []
    for i, e in enumerate(m.events):
        new_events.append(e)
        if i < len(m.events) - 1:
            n_passing = rng.randint(0, int(density) + 1)
            for k in range(n_passing):
                step = (m.events[i+1].semitone - e.semitone) // (n_passing + 1)
                mid = Event(e.semitone + step * (k+1), 0.4, 0.4, "ornament")
                new_events.append(mid)
    out = m.copy()
    out.events = [Event(e.semitone, e.dur_ratio, e.accent, e.role) for e in new_events]
    out.name = f"{m.name}_orn{density:.1f}"
    out.identity_strength = m.identity_strength * 0.82
    return out


def displace_register(m: Motif, octaves: int = 1) -> Motif:
    """Move entire motif up/down by octaves (12 semitones each)."""
    return transpose(m, octaves * 12)


def call_response(m: Motif, gap_ratio: float = 0.5) -> Tuple[Motif, Motif]:
    """Split into call and response (inverted, softer, slightly delayed)."""
    mid = len(m.events) // 2
    call = m.copy()
    call.events = m.events[:mid]
    call.name = f"{m.name}_call"

    resp = invert(fragment(m, 0.6, seed=42))
    resp.name = f"{m.name}_resp"
    resp.identity_strength = m.identity_strength * 0.75
    return call, resp


def cadential_mutate(m: Motif, strength: float = 0.5) -> Motif:
    """Mutate the final approach — lead toward a cadential tone (perfect 5th or octave)."""
    out = m.copy()
    target_semitone = 7 if strength > 0.3 else 12  # 5th or octave
    if out.events:
        out.events[-1] = Event(target_semitone, out.events[-1].dur_ratio * 1.5, 0.9, "cadence")
    out.name = f"{m.name}_cadence"
    out.identity_strength = m.identity_strength * 0.70
    return out


# ── Similarity ──────────────────────────────────────────────────────────────

def _to_relative(intervals: List[int]) -> List[int]:
    """Make intervals transposition-invariant by subtracting the first."""
    if not intervals:
        return []
    base = intervals[0]
    return [i - base for i in intervals]


def interval_distance(a: List[int], b: List[int]) -> float:
    """Edit distance on interval sequences, transposition-invariant.
    
    Compares intervals relative to the first note, so a motif and its
    transposition score as identical rather than completely different.
    """
    ra, rb = _to_relative(a), _to_relative(b)
    la, lb = len(ra), len(rb)
    if la == 0 and lb == 0:
        return 0.0
    # Levenshtein
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if ra[i-1] == rb[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    maxlen = max(la, lb)
    return dp[la][lb] / maxlen if maxlen > 0 else 0.0


def rhythm_distance(a: List[float], b: List[float]) -> float:
    """Mean absolute difference of duration ratios, normalised."""
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 1.0
    # Pad to same length
    maxlen = max(la, lb)
    pa = a + [0.0] * (maxlen - la)
    pb = b + [0.0] * (maxlen - lb)
    return np.mean(np.abs(np.array(pa) - np.array(pb)))


def contour_distance(a: List[int], b: List[int]) -> float:
    """Distance between contour sequences, with inversion symmetry.
    
    Musical inversion (mirroring) is a valid transformation, so we compare
    both direct and sign-flipped contour, taking the closer match.
    """
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 1.0
    maxlen = max(la, lb)
    pa = a + [0] * (maxlen - la)
    pb = b + [0] * (maxlen - lb)
    pa_arr = np.array(pa, dtype=float)
    pb_arr = np.array(pb, dtype=float)
    # Direct comparison
    direct = np.mean(np.abs(pa_arr - pb_arr))
    # Inverted comparison (sign flip = mirror)
    inverted = np.mean(np.abs(pa_arr + pb_arr))
    # Normalise by a reasonable range (max possible mean step in semitones)
    best = min(direct, inverted)
    return float(np.clip(best / 12.0, 0.0, 1.0))


def motif_similarity(a: Motif, b: Motif) -> float:
    """Weighted similarity in [0, 1]. 1 = identical, 0 = unrecognisable."""
    idist = interval_distance(a.intervals(), b.intervals())
    rdist = rhythm_distance(a.durations(), b.durations())
    cdist = contour_distance(a.contour(), b.contour())
    # Contour is strongest identifier, rhythm next, exact intervals weakest
    weighted = 0.25 * idist + 0.30 * rdist + 0.45 * cdist
    return float(np.clip(1.0 - weighted, 0.0, 1.0))


# ── Rendering ────────────────────────────────────────────────────────────────

def semitone_to_freq(semitone: int, fundamental: float) -> float:
    """Convert semitone offset to frequency."""
    return fundamental * (2 ** (semitone / 12.0))


def render_event(event: Event, fundamental: float, unit_dur: float, sr: int = SR) -> np.ndarray:
    """Render a single event as a stereo tone with envelope."""
    freq = semitone_to_freq(event.semitone, fundamental)
    if freq < 20 or freq > 8000:
        return np.zeros(int(sr * unit_dur * event.dur_ratio))
    n = max(1, int(sr * unit_dur * event.dur_ratio))
    t = np.arange(n) / sr

    # Choose role-based timbre
    if event.role == "lead":
        tone = np.sin(2 * np.pi * freq * t)
    elif event.role == "bass":
        tone = (np.sin(2 * np.pi * freq * t) +
                0.5 * np.sin(2 * np.pi * freq * 2 * t)) / 1.5
    elif event.role == "pad":
        tone = (np.sin(2 * np.pi * freq * t) +
                0.3 * np.sin(2 * np.pi * freq * 1.005 * t)) / 1.3
    elif event.role == "ornament":
        tone = 0.5 * np.sin(2 * np.pi * freq * t)
    else:
        tone = np.sin(2 * np.pi * freq * t)

    # Envelope: fast attack, sustain, release
    attack = min(int(sr * 0.05), n // 4)
    release = min(int(sr * 0.15), n // 2)
    env = np.ones(n)
    if attack > 0:
        env[:attack] = np.linspace(0, 1, attack)
    if release > 0:
        env[-release:] = np.linspace(1, 0, release)

    return tone * env * event.accent


def render_motif(m: Motif, fundamental: float, unit_dur: float = 1.0,
                 sr: int = SR, stereo_width: float = 0.5) -> np.ndarray:
    """Render a motif as a stereo audio buffer. Returns shape (N, 2)."""
    mono_parts = [render_event(e, fundamental, unit_dur, sr) for e in m.events]
    mono = np.concatenate(mono_parts) if mono_parts else np.zeros(0)
    if len(mono) == 0:
        return np.zeros((0, 2))

    # Stereo spread: slight detune per channel
    n = len(mono)
    stereo = np.zeros((n, 2))
    detune = 0.003 * stereo_width
    phase_offset = np.cumsum(np.ones(n) * detune)  # subtle drift
    stereo[:, 0] = mono * (1 + 0.001 * np.sin(phase_offset))
    stereo[:, 1] = mono * (1 - 0.001 * np.sin(phase_offset))
    return stereo


# ── A → B → A' → Outro ─────────────────────────────────────────────────────

def compose_long_horizon(
    motif: Motif,
    fundamental: float,
    total_dur: float = 120.0,
    sr: int = SR,
    seed: int = 42,
) -> np.ndarray:
    """Compose a piece: A (statement) → B (development) → A' (return) → outro."""
    rng = np.random.RandomState(seed)

    # Allocate time: A=25%, B=35%, A'=25%, outro=15%
    a_dur = total_dur * 0.25
    b_dur = total_dur * 0.35
    aprime_dur = total_dur * 0.25
    outro_dur = total_dur * 0.15

    unit_dur = 0.6  # seconds per motif unit

    # ── A: clear statement ──
    a_audio = render_motif(motif, fundamental, unit_dur, sr, stereo_width=0.4)
    # Loop/pad to fill a_dur
    a_target = int(sr * a_dur)
    if len(a_audio) > 0:
        repeats = max(1, a_target // len(a_audio) + 1)
        a_audio = np.tile(a_audio, (repeats, 1) if a_audio.ndim == 2 else repeats)[:a_target]
    else:
        a_audio = np.zeros((a_target, 2))

    # ── B: development — fragments, counterpoint, transformations ──
    fragments = []
    # Generate enough fragments to fill b_dur at 50% overlap
    avg_frag_dur = unit_dur  # average fragment duration
    avg_frag_samples = int(sr * avg_frag_dur)
    n_frags_needed = int((sr * b_dur) / (avg_frag_samples * 0.5)) + 5
    for _ in range(n_frags_needed):
        transforms = [
            lambda m: transpose(m, rng.choice([-5, -2, 2, 5, 7])),
            lambda m: fragment(m, rng.uniform(0.4, 0.7), seed=rng.randint(1000)),
            lambda m: invert(fragment(m, 0.6, seed=rng.randint(1000))),
            lambda m: augment(fragment(m, 0.5, seed=rng.randint(1000)), rng.uniform(1.5, 2.5)),
        ]
        t = rng.choice(transforms)
        fm = t(motif)
        frag_audio = render_motif(fm, fundamental, unit_dur * rng.uniform(0.8, 1.2), sr,
                                   stereo_width=0.5 + 0.3 * rng.random())
        if len(frag_audio) > 0:
            fragments.append(frag_audio)

    if fragments:
        # Overlap-add with crossfade
        b_target = int(sr * b_dur)
        b_audio = np.zeros((b_target, 2))
        pos = 0
        frag_idx = 0
        max_iters = 500  # safety cap
        iters = 0
        while pos < b_target and iters < max_iters:
            frag = fragments[frag_idx % len(fragments)]
            frag_idx += 1
            iters += 1
            end = min(pos + len(frag), b_target)
            actual = end - pos
            if actual > 0:
                # Crossfade in
                fade_in = min(int(sr * 0.5), actual)
                ramp = np.linspace(0, 1, fade_in).reshape(-1, 1)
                b_audio[pos:pos+fade_in] += frag[:fade_in] * ramp
                if actual > fade_in:
                    b_audio[pos+fade_in:end] += frag[fade_in:actual]
            pos += len(frag) // 2  # 50% overlap
    else:
        b_audio = np.zeros((int(sr * b_dur), 2))

    # ── A': recognizable return — transposed inversion or augmentation ──
    aprime_motif = transpose(invert(motif), rng.choice([-12, 0, 12]))  # octave shift
    aprime_audio = render_motif(aprime_motif, fundamental, unit_dur * 1.3, sr, stereo_width=0.6)
    aprime_target = int(sr * aprime_dur)
    if len(aprime_audio) > 0:
        repeats = max(1, aprime_target // len(aprime_audio) + 1)
        aprime_audio = np.tile(aprime_audio, (repeats, 1))[:aprime_target]
    else:
        aprime_audio = np.zeros((aprime_target, 2))

    # ── Outro: reduce to smallest identifying fragment, loop to fill ──
    smallest = fragment(motif, keep_ratio=0.4, seed=123)
    outro_audio = render_motif(smallest, fundamental, unit_dur * 1.5, sr, stereo_width=0.3)
    outro_target = int(sr * outro_dur)
    if len(outro_audio) > 0:
        # Loop the fragment to fill the section (like A and A'), then fade
        repeats = max(1, outro_target // len(outro_audio) + 1)
        outro_audio = np.tile(outro_audio, (repeats, 1))[:outro_target]
    else:
        outro_audio = np.zeros((outro_target, 2))

    # Ensure all are 2D and apply inter-section crossfades
    sections = [a_audio, b_audio, aprime_audio, outro_audio]
    for i in range(len(sections)):
        if sections[i].ndim == 1:
            sections[i] = sections[i].reshape(-1, 1)
        # Safety: force 2D (N, 2)
        if sections[i].ndim == 2 and sections[i].shape[1] != 2:
            sections[i] = np.column_stack([sections[i][:, 0], sections[i][:, 0]])

    # Inter-section crossfade (2s overlaps)
    xfade_len = int(sr * 2.0)
    parts = [sections[0]]
    for i in range(1, len(sections)):
        prev = parts[-1]
        curr = sections[i]
        if len(prev) > xfade_len and len(curr) > xfade_len:
            fade_out = np.linspace(1, 0, xfade_len).reshape(-1, 1)
            fade_in = np.linspace(0, 1, xfade_len).reshape(-1, 1)
            prev[-xfade_len:] *= fade_out  # fade out tail of previous
            curr[:xfade_len] *= fade_in    # fade in head of current
        parts.append(curr)

    piece = np.concatenate(parts, axis=0)

    # Master: 8s fade-out
    fade_len = int(sr * 8)
    if len(piece) > fade_len:
        fade = np.linspace(1, 0, fade_len).reshape(-1, 1)
        piece[-fade_len:] *= fade

    # Normalise
    peak = np.max(np.abs(piece))
    if peak > 0:
        piece = piece * (0.891 / peak)

    return piece


def write_wav(audio: np.ndarray, path: str, sr: int = SR):
    """Write stereo WAV via stdlib wave."""
    import wave
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])
    audio_16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_16.tobytes())


# ── Demo motif ──────────────────────────────────────────────────────────────

def make_demo_motif() -> Motif:
    """A simple pentatonic-ish motif for testing."""
    return Motif(
        events=[
            Event(0,  1.0, 0.9, "lead"),     # root
            Event(2,  0.7, 0.7, "lead"),     # major 2nd
            Event(4,  0.7, 0.75, "lead"),    # major 3rd
            Event(7,  1.2, 0.8, "lead"),     # perfect 5th
            Event(4,  0.5, 0.6, "ornament"), # back down
            Event(2,  0.5, 0.55, "ornament"),
            Event(0,  1.5, 0.85, "lead"),    # return to root
        ],
        name="demo_motif",
        identity_strength=1.0,
    )


if __name__ == "__main__":
    import sys
    m = make_demo_motif()
    print(f"Motif: {m.name}")
    print(f"  Intervals: {m.intervals()}")
    print(f"  Durations: {m.durations()}")
    print(f"  Contour:   {m.contour()}")
    print(f"  Identity:  {m.identity_strength}")

    t = transpose(m, 7)
    print(f"Transpose +7 semitones:")
    print(f"  Intervals: {t.intervals()}")
    print(f"  Similarity: {motif_similarity(m, t):.3f}")

    inv = invert(m)
    print(f"Inverted:")
    print(f"  Intervals: {inv.intervals()}")
    print(f"  Similarity: {motif_similarity(m, inv):.3f}")

    frag = fragment(m, 0.5, seed=42)
    print(f"Fragmented (50%):")
    print(f"  Events: {len(frag.events)}")
    print(f"  Similarity: {motif_similarity(m, frag):.3f}")

    # Compose
    print(f"\nComposing A→B→A'→outro (60s, D2=73.4Hz)...")
    audio = compose_long_horizon(m, fundamental=73.4, total_dur=60.0, seed=42)
    out = "/home/rynardt/songs/motif_demo.wav"
    write_wav(audio, out)
    import os
    print(f"Written: {out} ({os.path.getsize(out)} bytes)")
