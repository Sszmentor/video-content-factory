#!/usr/bin/env python3
"""
Create sync-corrected versions of HeyGen Avatar videos.
For each video, generates variants with different audio offsets
and replaces compressed audio with high-quality original.
"""

import subprocess
import os
import sys


BASE_DIR = "/Users/sergeyzinenko/Yandex.Disk.localized/Рабочее/DeFi-гедонист/Контент/Shorts/26.02.23"
OUTPUT_DIR = f"{BASE_DIR}/03-heygen/temp/sync_test"


def run_ffmpeg(cmd: list[str], description: str):
    """Run ffmpeg command with error handling."""
    print(f"  {description}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return False
    return True


def create_sync_variant(video_path: str, audio_path: str, name: str,
                        offset_ms: float, fps: int = 30):
    """
    Create a sync-corrected video variant.

    offset_ms > 0: audio plays later (use when lips move before sound)
    offset_ms < 0: audio plays earlier (use when lips lag behind sound)
    """
    output_path = f"{OUTPUT_DIR}/{name}_offset{offset_ms:+.0f}ms.mp4"
    offset_sec = offset_ms / 1000.0

    if offset_ms == 0:
        # Simple audio replacement + frame rate normalization
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-map', '0:v',
            '-map', '1:a',
            '-c:v', 'libx264',
            '-preset', 'slow',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),
            '-c:a', 'aac',
            '-b:a', '256k',
            '-ar', '48000',
            '-shortest',
            output_path
        ]
    elif offset_ms > 0:
        # Delay audio: add silence at the beginning of audio
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-itsoffset', str(offset_sec),
            '-i', audio_path,
            '-map', '0:v',
            '-map', '1:a',
            '-c:v', 'libx264',
            '-preset', 'slow',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),
            '-c:a', 'aac',
            '-b:a', '256k',
            '-ar', '48000',
            '-shortest',
            output_path
        ]
    else:
        # Advance audio: offset the video instead
        cmd = [
            'ffmpeg', '-y',
            '-itsoffset', str(-offset_sec),
            '-i', video_path,
            '-i', audio_path,
            '-map', '0:v',
            '-map', '1:a',
            '-c:v', 'libx264',
            '-preset', 'slow',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),
            '-c:a', 'aac',
            '-b:a', '256k',
            '-ar', '48000',
            '-shortest',
            output_path
        ]

    label = f"{name} offset={offset_ms:+.0f}ms → {fps}fps"
    return run_ffmpeg(cmd, label), output_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [
        {
            'name': 'A2',
            'video': f"{BASE_DIR}/03-heygen/A2.mp4",
            'audio': f"{BASE_DIR}/02-enhanced/A2_enhanced.mp3",
        },
        {
            'name': 'A3',
            'video': f"{BASE_DIR}/03-heygen/A3.mp4",
            'audio': f"{BASE_DIR}/02-enhanced/A3_enhanced.mp3",
        },
    ]

    # Offsets to try (in ms)
    # Negative = audio earlier (lip movement lags)
    # Positive = audio later (lip movement leads)
    offsets = [-80, -40, 0, +40, +80]

    results = []
    for f in files:
        print(f"\n{'='*60}")
        print(f"Processing {f['name']}")
        print(f"{'='*60}")

        for offset in offsets:
            ok, path = create_sync_variant(
                f['video'], f['audio'], f['name'], offset
            )
            if ok:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                results.append((f['name'], offset, path, size_mb))
                print(f"    OK: {os.path.basename(path)} ({size_mb:.1f} MB)")
            else:
                print(f"    FAILED: {f['name']} offset={offset}")

    print(f"\n{'='*60}")
    print(f"Summary: {len(results)} variants created in {OUTPUT_DIR}")
    print(f"{'='*60}")
    for name, offset, path, size in results:
        print(f"  {name} {offset:+4d}ms  →  {os.path.basename(path)}  ({size:.1f} MB)")

    print(f"\nКак тестировать:")
    print(f"  1. Откройте все варианты одного клипа (напр. A2)")
    print(f"  2. Смотрите на плозивы (/п/, /б/) — губы должны смыкаться точно на звук")
    print(f"  3. Если губы ОПЕРЕЖАЮТ звук → выберите + offset")
    print(f"  4. Если губы ОТСТАЮТ от звука → выберите - offset")
    print(f"  5. 0ms = просто замена аудио на оригинал + 30fps")


if __name__ == '__main__':
    main()
