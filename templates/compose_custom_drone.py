#!/usr/bin/env python3
"""Parameterized ambient drone generator — avoids compose.py stereo broadcast bug.

Accepts CLI args for theme, fundamental, duration, sections, output.
Produces WAV + MP3 via stdlib+numpy only (no GPU, no API).

Usage:
  uv run --with numpy python3 compose_custom_drone.py \
    --fundamental 55.0 \
    --duration 180 \
    --output /home/rynardt/songs/my-drone.mp3

Defaults produce an "distance as absence" style piece (A1, 180s, 4 sections).
"""
import numpy as np
import wave, struct, os, subprocess, json, math, argparse

SR = 44100

# Preset themes
PRESETS = {
    "memory": {
        "fundamental": 73.4,  # D2
        "sections": [
            {"name": "emerge",   "ratios": [1,2,3,5,8], "cutoff": (800,1200), "gain": 0.15, "detune": 0.18},
            {"name": "persist",  "ratios": [1,2,3,5,8], "cutoff": (1200,1000), "gain": 0.18, "detune": 0.18},
            {"name": "erode",    "ratios": [1,2,3],       "cutoff": (1000,400),  "gain": 0.12, "detune": 0.12},
            {"name": "dissolve", "ratios": [1],            "cutoff": (400,200),   "gain": 0.06, "detune": 0.06},
        ]
    },
    "distance": {
        "fundamental": 55.0,  # A1
        "sections": [
            {"name": "expand",   "ratios": [1,3,5,8,13], "cutoff": (400,1200), "gain": 0.15, "detune": 0.12},
            {"name": "plateau",  "ratios": [1,3,5,8,13], "cutoff": (1200,1000), "gain": 0.18, "detune": 0.15},
            {"name": "contract", "ratios": [1,3,5],       "cutoff": (1000,600),  "gain": 0.12, "detune": 0.08},
            {"name": "vanish",   "ratios": [1],            "cutoff": (600,200),   "gain": 0.06, "detune": 0.04},
        ]
    },
    "solitude": {
        "fundamental": 65.4,  # C2
        "sections": [
            {"name": "hush",     "ratios": [1],            "cutoff": (300,200),   "gain": 0.08, "detune": 0.05},
            {"name": "presence", "ratios": [1,2,3],       "cutoff": (200,500),   "gain": 0.12, "detune": 0.08},
            {"name": "solitude", "ratios": [1,2,3,5],    "cutoff": (500,300),   "gain": 0.10, "detune": 0.06},
            {"name": "hush",     "ratios": [1],            "cutoff": (300,200),   "gain": 0.06, "detune": 0.04},
        ]
    },
}

def make_brown_noise(n):
    white = np.random.randn(n)
    brown = np.cumsum(white)
    brown = brown / (np.max(np.abs(brown)) + 1e-9) * 0.3
    return brown

def lowpass(signal, cutoff, sr=SR):
    rc = 1.0 / (2 * np.pi * cutoff)
    dt = 1.0 / sr
    alpha = dt / (rc + dt)
    out = np.zeros_like(signal)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = alpha * signal[i] + (1 - alpha) * out[i-1]
    return out

def build_section(spec, fundamental, sr=SR):
    n = int(sr * spec.get("dur", 45))
    t = np.arange(n) / sr
    
    # Harmonic stack in mono
    mono = np.zeros(n)
    for h in spec["ratios"]:
        freq = fundamental * h
        if freq > 8000:
            continue
        tone = np.sin(2 * np.pi * freq * t)
        mono += tone / (h ** 0.7)
    peak = np.max(np.abs(mono))
    if peak > 0:
        mono = mono / peak * spec["gain"]
    
    # Noise floor with time-varying cutoff
    noise = make_brown_noise(n)
    cutoff_env = np.linspace(spec["cutoff"][0], spec["cutoff"][1], n)
    noise_f = np.zeros(n)
    frame_size = 512
    for i in range(0, n, frame_size):
        end = min(i + frame_size, n)
        co = cutoff_env[i]
        filtered = lowpass(noise[i:end], co)
        actual = end - i
        noise_f[i:end] = filtered[:actual]
    
    mono = mono + noise_f * 0.15
    
    # ADSR
    attack = int(sr * 3)
    release = int(sr * 5)
    env = np.ones(n)
    if 0 < attack < n:
        env[:attack] = np.linspace(0, 1, attack)
    if 0 < release < n:
        env[-release:] = np.linspace(1, 0, release)
    mono = mono * env
    
    # Stereo with detuning + pan LFO
    stereo = np.zeros((n, 2))
    pan_lfo = 0.5 + 0.4 * np.sin(2 * np.pi * 0.03 * t)
    
    left = np.zeros(n)
    right = np.zeros(n)
    for h in spec["ratios"]:
        freq = fundamental * h
        if freq > 8000:
            continue
        left += np.sin(2 * np.pi * freq * (1 + spec["detune"]) * t) / (h ** 0.7)
        right += np.sin(2 * np.pi * freq * (1 - spec["detune"]) * t) / (h ** 0.7)
    
    pl = np.max(np.abs(left))
    pr = np.max(np.abs(right))
    if pl > 0:
        left = left / pl * spec["gain"] * env
    if pr > 0:
        right = right / pr * spec["gain"] * env
    
    stereo[:, 0] = left * pan_lfo + noise_f * 0.1 * (1 - pan_lfo)
    stereo[:, 1] = right * (1 - pan_lfo) + noise_f * 0.1 * pan_lfo
    
    return stereo

def compose(theme_name, fundamental, section_list, output_path, sr=SR):
    sections = []
    for spec in section_list:
        sec = build_section(spec, fundamental, sr)
        sections.append(sec)
        print(f"  {spec['name']}: {sec.shape[0]/sr:.1f}s")
    
    full = np.concatenate(sections, axis=0)
    total_dur = full.shape[0] / sr
    
    # Master fade-out
    fade_len = int(sr * 8)
    if full.shape[0] > fade_len:
        fade = np.linspace(1, 0, fade_len).reshape(-1, 1)
        full[-fade_len:] *= fade
    
    # Normalize to -1 dB
    peak = np.max(np.abs(full))
    target = 10 ** (-1.0 / 20)
    if peak > 0:
        full = full * (target / peak)
    
    # Write WAV
    wav_path = output_path
    if output_path.endswith('.mp3'):
        wav_path = output_path[:-4] + '.wav'
    
    full_16 = (full * 32767).astype(np.int16)
    with wave.open(wav_path, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(full_16.tobytes())
    print(f"WAV: {wav_path} ({os.path.getsize(wav_path)} bytes)")
    
    # Convert to MP3 if needed
    if output_path.endswith('.mp3'):
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-qscale:a", "3", output_path],
                capture_output=True, text=True, timeout=120
            )
            if r.returncode == 0 and os.path.exists(output_path):
                os.remove(wav_path)
                print(f"MP3: {output_path} ({os.path.getsize(output_path)} bytes)")
                return output_path
        except Exception:
            pass
        print(f"ffmpeg failed, keeping WAV")
        return wav_path
    
    return wav_path

def main():
    parser = argparse.ArgumentParser(description="Ambient drone generator")
    parser.add_argument("--theme", default="distance", choices=list(PRESETS.keys()),
                        help="Theme preset (memory/distance/solitude)")
    parser.add_argument("--fundamental", type=float, default=None,
                        help="Fundamental frequency (overrides preset)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Total duration in seconds (evenly split across sections)")
    parser.add_argument("--output", default="/home/rynardt/songs/custom-drone.mp3",
                        help="Output path (.mp3 or .wav)")
    args = parser.parse_args()
    
    preset = PRESETS[args.theme]
    fundamental = args.fundamental or preset["fundamental"]
    sections = preset["sections"]
    
    # Override duration if specified
    if args.duration:
        per_section = args.duration / len(sections)
        for s in sections:
            s["dur"] = per_section
    
    print(f"Theme: {args.theme}, Fundamental: {fundamental}Hz, Sections: {len(sections)}")
    out = compose(args.theme, fundamental, sections, args.output)
    print(json.dumps({"file": out, "duration_s": round(args.duration or sum(s.get("dur",45) for s in sections), 1)}))

if __name__ == "__main__":
    main()
