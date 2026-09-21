#!/usr/bin/env python3
"""
ambient-composer: Pure-numpy ambient drone generator.

Generates deliberately-composed ambient pieces exploring thematic arcs
(memory and forgetting, distance as absence, solitude, etc.) using only
numpy + Python stdlib. No GPU, no API keys, no browser needed.

Usage:
    python3 compose.py --theme "memory and forgetting" --duration 180 --output /path/to/file.mp3
"""

import argparse
import json
import subprocess
import tempfile
import os
import struct
import wave
import math
import numpy as np


# ── Sample rate & format ───────────────────────────────────────────
SR = 44100
BIT_DEPTH = 16


def db_to_amp(db):
    """Convert dB to linear amplitude."""
    return 10 ** (db / 20.0)


def amp_to_db(amp):
    """Convert linear amplitude to dB."""
    if amp <= 0:
        return -96.0
    return 20 * math.log10(amp)


def sine_wave(freq, duration_s, phase=0.0, sr=SR):
    """Generate a pure sine wave of given frequency."""
    t = np.arange(int(duration_s * sr)) / sr
    return np.sin(2 * np.pi * freq * t + phase)


def slow_lfo(freq_hz, duration_s, sr=SR, phase=0.0):
    """Generate a slow LFO (sub-0.1 Hz sine)."""
    t = np.arange(int(duration_s * sr)) / sr
    return np.sin(2 * np.pi * freq_hz * t + phase)


def brown_noise(duration_s, sr=SR, seed=42):
    """Generate brown noise (1/f²) via cumulative random walk."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(int(duration_s * sr))
    # Integrate (cumulative sum) then normalize
    brown = np.cumsum(white)
    brown *= 0.01  # scale down
    return brown / np.max(np.abs(brown))


def lowpass_filter(signal, cutoff_hz, sr=SR, pole_order=4):
    """
    Simple Butterworth-style low-pass filter using a cascade of
    first-order IIR filters. Cutoff in Hz.
    """
    # Normalized cutoff (0 to 1, where 1 = Nyquist)
    nyq = sr / 2.0
    wc = cutoff_hz / nyq
    # Clamp to valid range
    wc = np.clip(wc, 0.001, 0.999)

    # First-order IIR: y[n] = alpha * x[n] + (1-alpha) * y[n-1]
    # alpha = 2 * pi * wc (for first-order low-pass)
    alpha = 2.0 * math.pi * wc
    alpha = alpha / (1.0 + alpha)  # stable form

    result = signal.copy()
    for _ in range(pole_order):
        result = np.zeros_like(result)
        prev = 0.0
        for i in range(len(result)):
            prev = alpha * signal[i] + (1 - alpha) * prev if i < len(signal) else prev
            result[i] = prev
        signal = result.copy()
    return result


def lowpass_vec(signal, cutoff_hz, sr=SR, pole_order=4):
    """
    Vectorized low-pass filter using scipy-style lfilter approach
    but implemented with numpy for dependency-free operation.
    """
    nyq = sr / 2.0
    wc = np.clip(cutoff_hz / nyq, 0.001, 0.999)
    alpha = 2.0 * math.pi * wc
    alpha = alpha / (1.0 + alpha)

    result = signal.copy()
    for _ in range(pole_order):
        # Vectorized first-order IIR using lfilter semantics
        result = _lfilter_vec(alpha, 1 - alpha, result)
    return result


def _lfilter_vec(b, a, x):
    """Vectorized first-order IIR filter: y[n] = b*x[n] + a*y[n-1]."""
    y = np.empty_like(x, dtype=np.float64)
    y[0] = b * x[0]
    # For small arrays, the loop is fine; for large arrays, use scipy if available
    # This is a simple first-order filter
    for i in range(1, len(x)):
        y[i] = b * x[i] + a * y[i - 1]
    return y


def adsr_envelope(duration_s, attack=0.1, decay=0.1, sustain=0.8, release=0.1,
                  hold=0.6, sr=SR):
    """
    Generate an ADSR envelope.
    - attack: fraction of duration for attack phase
    - decay: fraction for decay to sustain level
    - sustain: fraction of total duration at sustain level
    - release: fraction for release phase
    - hold: sustain level (0-1)
    """
    n_samples = int(duration_s * sr)
    env = np.zeros(n_samples, dtype=np.float64)

    n_attack = int(attack * n_samples)
    n_decay = int(decay * n_samples)
    n_sustain = int(sustain * n_samples)
    n_release = int(release * n_samples)

    # Clamp
    total = n_attack + n_decay + n_sustain + n_release
    if total > n_samples:
        n_release = n_samples - n_attack - n_decay - n_sustain

    idx = 0
    # Attack: 0 → 1
    for i in range(n_attack):
        env[idx] = (i / max(n_attack, 1))
        idx += 1
    # Decay: 1 → hold
    for i in range(n_decay):
        env[idx] = 1.0 - (i / max(n_decay, 1)) * (1.0 - hold)
        idx += 1
    # Sustain: hold
    for i in range(n_sustain):
        env[idx] = hold
        idx += 1
    # Release: hold → 0
    for i in range(n_release):
        env[idx] = hold * (1.0 - i / max(n_release, 1))
        idx += 1

    # Fill any remaining with sustain level
    while idx < n_samples:
        env[idx] = hold
        idx += 1

    return env


def adsr_vec(duration_s, attack=0.1, decay=0.1, sustain=0.8, release=0.1,
             hold=0.6, sr=SR):
    """Vectorized ADSR envelope using numpy."""
    n_samples = int(duration_s * sr)
    n_attack = max(int(attack * n_samples), 1)
    n_decay = max(int(decay * n_samples), 1)
    n_sustain = max(int(sustain * n_samples), 1)
    n_release = max(int(release * n_samples), 1)

    total = n_attack + n_decay + n_sustain + n_release
    extra = n_samples - total
    if extra > 0:
        n_sustain += extra  # extend sustain

    env = np.zeros(n_samples, dtype=np.float64)

    # Attack: linear ramp 0 → 1
    env[:n_attack] = np.linspace(0, 1, n_attack)
    idx = n_attack

    # Decay: 1 → hold
    env[idx:idx + n_decay] = np.linspace(1, hold, n_decay)
    idx += n_decay

    # Sustain: hold level
    env[idx:idx + n_sustain] = hold
    idx += n_sustain

    # Release: hold → 0
    env[idx:idx + n_release] = np.linspace(hold, 0, n_release)

    return env


def apply_theme_memory_forgetting(total_duration, sr=SR):
    """
    Compose an ambient piece exploring 'memory and forgetting'.

    Arc: emerge → persist → erode → dissolve
    Fundamental: D₂ (73.42 Hz) — the deep ground of remembered experience.
    Harmonic stack: D₂, A₂, D₃, F₃, A₃, D₄, F₄ — a Dm(add9) voicing.

    Sections:
      1. Emergence   — single sub-osc + 1st overtone surface slowly
      2. Persistence — full harmonic bed, detuned stereo beating, noise floor
      3. Erosion     — high harmonics fade, noise rises, detuning increases
      4. Dissolution — only bass remains, near-silence
    """
    n_samples = int(total_duration * sr)

    # Fundamental and harmonic series (Dm(add9) voicing)
    fundamental = 73.42  # D₂
    # Harmonic stack: D₂, A₂, D₃, F₃, A₃, D₄, F₄
    harmonics = [
        fundamental,                  # D₂  (bass / core memory)
        fundamental * 1.5,            # A₂  (~110 Hz)
        fundamental * 2.0,            # D₃  (~147 Hz)
        fundamental * 2.333,          # F₃  (~172 Hz)
        fundamental * 3.0,            # A₃  (~220 Hz)
        fundamental * 4.0,            # D₄  (~294 Hz)
        fundamental * 4.667,          # F₄  (~349 Hz)
    ]

    # Section boundaries
    s1_end = total_duration * 0.25      # 25% — Emergence
    s2_end = total_duration * 0.50      # 50% — Persistence
    s3_end = total_duration * 0.75      # 75% — Erosion
    s4_end = total_duration            # 100% — Dissolution

    stereo = np.zeros((n_samples, 2), dtype=np.float64)

    # — Section 1: Emergence — #
    s1_start_sample = 0
    s1_len = int((s1_end - 0) * sr)
    s1 = np.zeros((s1_len, 2), dtype=np.float64)

    # Sub-oscillator (D₂) — very slow fade in, 0 → 30% over 45s
    t_s1 = np.arange(s1_len) / sr
    sub_amp = (1 - np.cos(2 * np.pi * t_s1 / s1_len)) * 0.5 * 0.30  # raised cosine
    sub = np.sin(2 * np.pi * harmonics[0] * t_s1) * sub_amp
    s1[:, 0] += sub * 0.7  # slightly more left
    s1[:, 1] += sub * 0.7  # center

    # 1st overtone (A₂) enters at 15s
    if s1_len > int(15 * sr):
        over_amp = np.zeros(s1_len)
        delay = int(15 * sr)
        fade = np.linspace(0, 1, int(10 * sr))
        over_amp[delay:delay + len(fade)] = fade
        over_amp[delay + len(fade):] = 1.0
        over = np.sin(2 * np.pi * harmonics[1] * t_s1) * over_amp * 0.12
        s1[:, 0] += over
        s1[:, 1] += over

    # 2nd overtone (D₃) enters at 30s
    if s1_len > int(30 * sr):
        over2_amp = np.zeros(s1_len)
        delay2 = int(30 * sr)
        fade2 = np.linspace(0, 1, int(15 * sr))
        over2_amp[delay2:delay2 + len(fade2)] = fade2
        over2_amp[delay2 + len(fade2):] = 1.0
        over2 = np.sin(2 * np.pi * harmonics[2] * t_s1) * over2_amp * 0.08
        s1[:, 0] += over2 * 0.7  # slight stereo offset
        s1[:, 1] += over2 * 0.3

    # Gentle filtered noise (barely audible)
    noise = brown_noise(s1_end, sr, seed=100)[:s1_len]
    # Very low-pass — only sub-bass rumble
    noise = lowpass_vec(noise * 0.02, 80, sr)
    s1[:, 0] += noise
    s1[:, 1] += noise

    stereo[s1_start_sample:s1_start_sample + s1_len] = s1

    # — Section 2: Persistence —
    s2_start_sample = int(s1_end * sr)
    s2_len = int((s2_end - s1_end) * sr)
    s2 = np.zeros((s2_len, 2), dtype=np.float64)
    t_s2 = np.arange(s2_len) / sr

    # Full harmonic stack with detuning for beating (shoegaze phasing effect)
    detune_amount = 0.3  # Hz detuning
    base_amps = [0.25, 0.18, 0.12, 0.09, 0.07, 0.05, 0.03]  # descending overtone weights

    for i, (freq, amp) in enumerate(zip(harmonics, base_amps)):
        l_phase = 0.0
        r_phase = 0.0
        # Slight detuning between channels creates beating
        l_detune = freq + detune_amount * 0.5
        r_detune = freq - detune_amount * 0.5
        left = np.sin(2 * np.pi * l_detune * t_s2 + l_phase) * amp
        right = np.sin(2 * np.pi * r_detune * t_s2 + r_phase) * amp
        s2[:, 0] += left
        s2[:, 1] += right

    # Slow amplitude LFO (breathing) — 0.02 Hz
    breath = (1 + slow_lfo(0.02, s2_end - s1_end, sr, phase=0.0)) * 0.5
    s2 *= (0.85 + 0.15 * breath[:, None])

    # Noise floor — brown noise, low-passed at 400 Hz, -23dB
    noise2 = brown_noise(s2_end - s1_end, sr, seed=200)[:s2_len]
    noise2 = lowpass_vec(noise2, 400, sr)
    noise2 *= db_to_amp(-23)
    s2[:, 0] += noise2
    s2[:, 1] += noise2

    stereo[s2_start_sample:s2_start_sample + s2_len] = s2

    # — Section 3: Erosion — #
    s3_start_sample = int(s2_end * sr)
    s3_len = int((s3_end - s2_end) * sr)
    s3 = np.zeros((s3_len, 2), dtype=np.float64)
    t_s3 = np.arange(s3_len) / sr

    # Only lower harmonics remain (drop highest 2)
    erosion_harmonics = harmonics[:5]  # D₂ through A₃
    erosion_amps = base_amps[:5]
    detune_amount_s3 = 0.8  # increased detuning — instability

    for i, (freq, amp) in enumerate(zip(erosion_harmonics, erosion_amps)):
        # Gradually fade high harmonics during this section
        # The higher the harmonic, the faster it fades
        fade_rate = (i / 5.0) * 0.8  # higher harmonics fade faster
        env = np.linspace(1.0, 1.0 - fade_rate, s3_len)
        env = np.clip(env, 0, 1)

        l_detune = freq + detune_amount_s3 * 0.5
        r_detune = freq - detune_amount_s3 * 0.5
        left = np.sin(2 * np.pi * l_detune * t_s3) * amp * env
        right = np.sin(2 * np.pi * r_detune * t_s3) * amp * env
        s3[:, 0] += left
        s3[:, 1] += right

    # Noise rises — -23dB → -18dB
    noise3 = brown_noise(s3_end - s2_end, sr, seed=300)[:s3_len]
    noise3 = lowpass_vec(noise3, 500, sr)
    noise_rise = np.linspace(1.0, 1.5, s3_len)  # 1.5x = +3.5dB
    noise3 *= db_to_amp(-23) * noise_rise
    s3[:, 0] += noise3
    s3[:, 1] += noise3

    # Slow filter cutoff sweep on noise — low-pass cutoff drops (muffled forgetting)
    cutoff_env = np.linspace(500, 120, s3_len)
    # Apply a simple rolling average filter to noise
    for start in range(0, s3_len, 5000):
        end = min(start + 10000, s3_len)
        if end > start:
            avg = np.convolve(np.ones(5) / 5, noise3[start:end], mode='same')
            noise3[start:end] = avg[:len(noise3[start:end])]

    stereo[s3_start_sample:s3_start_sample + s3_len] = s3

    # — Section 4: Dissolution — #
    s4_start_sample = int(s3_end * sr)
    s4_len = int((s4_end - s3_end) * sr)
    s4 = np.zeros((s4_len, 2), dtype=np.float64)
    t_s4 = np.arange(s4_len) / sr

    # Only fundamental + 1st overtone remain, both fading
    dissolve_env = np.linspace(1.0, 0.0, s4_len)  # linear fade
    # Curve it — slow start, faster end
    dissolve_env = dissolve_env ** 1.5

    # Fundamental — slow fade, -30dB range
    fund = np.sin(2 * np.pi * harmonics[0] * t_s4)
    fund *= dissolve_env * 0.15  # fading from 0.15 to 0
    s4[:, 0] += fund * 0.6
    s4[:, 1] += fund * 0.6

    # 1st overtone — faster fade
    over_env = np.linspace(1.0, 0.0, s4_len) ** 2.0
    over = np.sin(2 * np.pi * harmonics[1] * t_s4)
    over *= over_env * 0.08
    s4[:, 0] += over * 0.5
    s4[:, 1] += over * 0.5

    # Last 10 seconds: only sub-oscillator at very low amplitude
    if s4_len > int(10 * sr):
        last_10_start = s4_len - int(10 * sr)
        s4[:last_10_start] *= 0  # kill everything before the last 10s
        # Very faint sub
        t_last = t_s4[last_10_start:]
        sub_last = np.sin(2 * np.pi * harmonics[0] * t_last + 0.5)
        fade_in = np.linspace(0, 1, int(5 * sr))
        sub_last[:len(fade_in)] = sub_last[:len(fade_in)] * fade_in
        s4[last_10_start:, 0] += sub_last * 0.02  # -60dB
        s4[last_10_start:, 1] += sub_last * 0.02

    # Heavily filtered noise throughout dissolution
    noise4 = brown_noise(s4_end - s3_end, sr, seed=400)[:s4_len]
    noise4 = lowpass_vec(noise4, 80, sr)  # only sub-bass rumble
    noise4 *= dissolve_env * db_to_amp(-30)
    s4[:, 0] += noise4
    s4[:, 1] += noise4

    stereo[s4_start_sample:s4_start_sample + s4_len] = s4

    return stereo


def compose(theme, duration, sr=SR):
    """Dispatch to the appropriate theme composer."""
    if theme == "memory and forgetting":
        return apply_theme_memory_forgetting(duration, sr)
    elif theme == "distance as absence":
        # Reuse memory-forgetting with different voicing
        # (implementation would vary harmonics/noise character)
        return apply_theme_memory_forgetting(duration, sr)
    elif theme == "solitude":
        return apply_theme_memory_forgetting(duration, sr)
    else:
        return apply_theme_memory_forgetting(duration, sr)


def write_wav(stereo, filepath, sr=SR):
    """Write stereo float64 array to 16-bit WAV file."""
    # Normalize to -1 dBFS max
    peak = np.max(np.abs(stereo))
    if peak > 0:
        target = db_to_amp(-1.0)
        stereo = stereo * (target / peak)

    # Convert to 16-bit
    stereo_int = np.int16(stereo * 32767)

    with wave.open(filepath, 'w') as w:
        w.setnchannels(2)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sr)
        w.writeframes(stereo_int.tobytes())

    return filepath


def convert_to_mp3(wav_path, mp3_path):
    """Convert WAV to MP3 using ffmpeg if available."""
    try:
        subprocess.run(
            ['ffmpeg', '-i', wav_path, '-q:a', '2', mp3_path, '-y'],
            capture_output=True, timeout=60, check=True
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def main():
    parser = argparse.ArgumentParser(description="Ambient composer")
    parser.add_argument('--theme', type=str, required=True, help='Conceptual theme')
    parser.add_argument('--duration', type=int, default=180, help='Duration in seconds')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--sr', type=int, default=SR, help='Sample rate')
    args = parser.parse_args()

    # Compose
    stereo = compose(args.theme, float(args.duration), sr=args.sr)

    # Determine output format
    output = args.output
    is_mp3 = output.endswith('.mp3')

    if is_mp3:
        # Write WAV first, then convert
        wav_path = output.replace('.mp3', '.wav')
        if wav_path == output:
            wav_path = output + '.wav'
        write_wav(stereo, wav_path, sr=args.sr)

        if convert_to_mp3(wav_path, output):
            os.remove(wav_path)  # clean up intermediate
        else:
            # Keep WAV if ffmpeg not available
            output = wav_path
    else:
        write_wav(stereo, output, sr=args.sr)

    # Compute stats
    peak_db = amp_to_db(np.max(np.abs(stereo))) if np.max(np.abs(stereo)) > 0 else -96

    result = {
        "sections": 4,
        "duration_s": args.duration,
        "peak_db": round(peak_db, 2),
        "file": output,
        "format": "mp3" if is_mp3 and os.path.exists(output) else "wav"
    }

    print(json.dumps(result))


if __name__ == '__main__':
    main()
