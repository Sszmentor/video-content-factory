#!/usr/bin/env python3
"""
Measure audio sync offset between HeyGen video audio track and original audio.
Uses cross-correlation to find the precise time shift.
"""

import subprocess
import numpy as np
import sys
import json
import os
import tempfile


def extract_audio_to_numpy(file_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Extract audio from any media file and return as numpy array."""
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            'ffmpeg', '-y', '-i', file_path,
            '-vn',  # no video
            '-ac', '1',  # mono
            '-ar', str(sample_rate),  # resample
            '-f', 's16le',  # raw PCM
            '-acodec', 'pcm_s16le',
            tmp_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        audio = np.fromfile(tmp_path, dtype=np.int16).astype(np.float32)
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        return audio
    finally:
        os.unlink(tmp_path)


def measure_offset(reference_audio: np.ndarray, target_audio: np.ndarray,
                   sample_rate: int = 16000, max_offset_ms: int = 500) -> dict:
    """
    Measure time offset between reference and target audio using cross-correlation.

    Returns dict with:
        offset_samples: int (positive = target is ahead / needs delay)
        offset_ms: float
        confidence: float (0-1, higher = more confident)
    """
    # Limit search range
    max_offset_samples = int(max_offset_ms * sample_rate / 1000)

    # Trim to same length for correlation
    min_len = min(len(reference_audio), len(target_audio))
    ref = reference_audio[:min_len]
    tgt = target_audio[:min_len]

    # Use smaller chunks for faster computation
    # Analyze first 10 seconds (most reliable for offset detection)
    chunk_len = min(min_len, sample_rate * 10)
    ref_chunk = ref[:chunk_len]
    tgt_chunk = tgt[:chunk_len]

    # Cross-correlate using FFT
    n = len(ref_chunk) + len(tgt_chunk) - 1
    fft_size = 1
    while fft_size < n:
        fft_size *= 2

    ref_fft = np.fft.rfft(ref_chunk, fft_size)
    tgt_fft = np.fft.rfft(tgt_chunk, fft_size)
    correlation = np.fft.irfft(ref_fft * np.conj(tgt_fft))

    # Find peak within allowed range
    # Correlation indices: 0..max_offset = target delayed, -(max_offset)..0 = target early
    valid_range_pos = correlation[:max_offset_samples]
    valid_range_neg = correlation[-max_offset_samples:]
    combined = np.concatenate([valid_range_pos, valid_range_neg])

    peak_idx = np.argmax(np.abs(combined))
    if peak_idx < max_offset_samples:
        offset_samples = peak_idx
    else:
        offset_samples = -(len(combined) - peak_idx)

    offset_ms = offset_samples * 1000.0 / sample_rate

    # Confidence: ratio of peak to mean
    peak_val = np.abs(combined[peak_idx])
    mean_val = np.mean(np.abs(combined))
    confidence = float(peak_val / (mean_val + 1e-8))
    # Normalize to 0-1
    confidence = min(confidence / 20.0, 1.0)

    return {
        'offset_samples': int(offset_samples),
        'offset_ms': round(offset_ms, 2),
        'offset_frames_25fps': round(offset_ms / 40.0, 2),
        'confidence': round(confidence, 3),
        'direction': 'audio_leads' if offset_ms > 0 else 'audio_lags' if offset_ms < 0 else 'in_sync'
    }


def analyze_energy_onset(audio: np.ndarray, sample_rate: int = 16000,
                         threshold: float = 0.05) -> float:
    """Find the onset time (first moment of significant energy)."""
    window_size = int(sample_rate * 0.01)  # 10ms windows
    energy = np.array([
        np.mean(audio[i:i+window_size]**2)
        for i in range(0, len(audio) - window_size, window_size)
    ])
    # Find first window above threshold
    for i, e in enumerate(energy):
        if e > threshold * np.max(energy):
            return i * 0.01  # seconds
    return 0.0


def main():
    base_dir = "/Users/sergeyzinenko/Yandex.Disk.localized/Рабочее/DeFi-гедонист/Контент/Shorts/26.02.23"

    files = [
        {
            'name': 'A2',
            'video': f"{base_dir}/03-heygen/A2.mp4",
            'original_audio': f"{base_dir}/02-enhanced/A2_enhanced.mp3",
        },
        {
            'name': 'A3',
            'video': f"{base_dir}/03-heygen/A3.mp4",
            'original_audio': f"{base_dir}/02-enhanced/A3_enhanced.mp3",
        },
    ]

    sr = 16000
    results = {}

    for f in files:
        print(f"\n{'='*60}")
        print(f"Analyzing {f['name']}")
        print(f"{'='*60}")

        # Extract audio from video (HeyGen's compressed version)
        print(f"  Extracting audio from video...")
        video_audio = extract_audio_to_numpy(f['video'], sr)

        # Load original audio
        print(f"  Loading original audio...")
        original_audio = extract_audio_to_numpy(f['original_audio'], sr)

        # Measure offset
        print(f"  Computing cross-correlation...")
        offset = measure_offset(original_audio, video_audio, sr, max_offset_ms=500)

        # Analyze onset times
        video_onset = analyze_energy_onset(video_audio, sr)
        original_onset = analyze_energy_onset(original_audio, sr)
        onset_diff_ms = round((video_onset - original_onset) * 1000, 2)

        print(f"\n  Results for {f['name']}:")
        print(f"    Cross-correlation offset: {offset['offset_ms']} ms")
        print(f"    Offset in frames (25fps): {offset['offset_frames_25fps']}")
        print(f"    Direction: {offset['direction']}")
        print(f"    Confidence: {offset['confidence']}")
        print(f"    Energy onset (video audio): {video_onset:.3f}s")
        print(f"    Energy onset (original):    {original_onset:.3f}s")
        print(f"    Onset difference: {onset_diff_ms} ms")

        # Duration comparison
        video_dur = len(video_audio) / sr
        orig_dur = len(original_audio) / sr
        drift_ms = round((video_dur - orig_dur) * 1000, 2)
        print(f"    Duration (video audio): {video_dur:.3f}s")
        print(f"    Duration (original):    {orig_dur:.3f}s")
        print(f"    Duration drift: {drift_ms} ms")

        if abs(drift_ms) > 100:
            drift_rate = drift_ms / (video_dur * 1000)
            print(f"    WARNING: Progressive drift detected! Rate: {drift_rate:.6f}")

        results[f['name']] = {
            'offset': offset,
            'onset_diff_ms': onset_diff_ms,
            'duration_drift_ms': drift_ms,
            'video_duration_s': round(video_dur, 3),
            'original_duration_s': round(orig_dur, 3),
        }

    # Save results
    output_path = f"{base_dir}/03-heygen/temp/sync_analysis.json"
    with open(output_path, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    main()
