#!/usr/bin/env python3
"""
HeyGen Sync Pipeline — от сырого аудио до синхронизированного видео.

Полный pipeline без конфликтов:
  1. enhance  — EQ + compressor (БЕЗ loudnorm, чтобы не делать дважды)
  2. trim     — energy onset detection (–55dB, 150мс safety margin)
  3. loudnorm — two-pass -16 LUFS (один раз, после trim)
  4. upload   — загрузка в HeyGen API
  5. generate — запуск генерации видео с аватаром
  6. poll     — ожидание готовности
  7. download — скачивание видео
  8. sync-fix — auto_sync_fix (detect offset → fix если >30мс)

Ключевое отличие от наивного pipeline:
- loudnorm НЕ дублируется (enhance идёт без loudnorm)
- sync-fix НЕ слепой +80мс, а автодетекция через mediapipe

Usage:
  # Полный pipeline:
  python heygen_sync_pipeline.py run \
    --audio raw_recording.m4a \
    --avatar green \
    --duration 15

  # Только подготовка аудио (без HeyGen):
  python heygen_sync_pipeline.py prepare \
    --audio raw_recording.m4a \
    --duration 15

  # Только sync-fix (уже есть видео):
  python heygen_sync_pipeline.py sync \
    --video heygen_output.mp4 \
    --audio prepared_audio.mp3

Dependencies: mediapipe, numpy, ffmpeg, curl
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AVATARS = {
    'green': {
        'id': 'f9a08bfd4e6a4c59abc8a88ddec62bd2',
        'name': 'Верт. Зелён шёлк рубашка V2',
        'width': 1080, 'height': 1920,  # portrait
    },
    'blue': {
        'id': '01ddf9ea190f47bb8c8426264b4417fb',
        'name': 'Синяя рубашка Armani',
        'width': 1920, 'height': 1080,  # landscape
    },
}

# HeyGen EQ preset — БЕЗ loudnorm (loudnorm будет отдельным шагом)
HEYGEN_EQ_CHAIN = (
    "highpass=f=65:poles=2,"
    "equalizer=f=115:t=h:width_type=s:width=0.7:g=3.5,"    # Bass +3.5dB
    "equalizer=f=210:t=h:width_type=s:width=0.8:g=2.5,"    # Warmth +2.5dB
    "equalizer=f=400:t=h:width_type=s:width=1.2:g=-3.0,"   # Mud cut -3dB
    "equalizer=f=3000:t=h:width_type=s:width=1.5:g=2.0,"   # Presence +2dB
    "equalizer=f=6000:t=h:width_type=s:width=2.0:g=-1.5,"  # De-esser -1.5dB
    "treble=g=-1.5:f=8000:t=s:width_type=s:width=0.5,"     # Treble shelf -1.5dB
    "acompressor=threshold=-20dB:ratio=3.5:attack=10:release=150:knee=6:makeup=2dB"
)


# ---------------------------------------------------------------------------
# Step 1: Enhance (EQ + compression, NO loudnorm)
# ---------------------------------------------------------------------------

def step_enhance(input_path: str, output_path: str, duration: float = None) -> str:
    """Apply EQ + compressor WITHOUT loudnorm.

    loudnorm is a SEPARATE step (step 3) — after trimming.
    This avoids double normalization which degrades quality.
    """
    log.info("Step 1: Enhance (EQ + compressor, без loudnorm)")

    cmd = ['ffmpeg', '-y', '-i', input_path]
    if duration:
        cmd.extend(['-t', str(duration)])
    cmd.extend([
        '-af', HEYGEN_EQ_CHAIN,
        '-ar', '48000', '-ac', '1',
        '-c:a', 'pcm_s24le',
        '-v', 'error',
        output_path
    ])
    subprocess.run(cmd, check=True)
    log.info(f"  → {Path(output_path).name}")
    return output_path


# ---------------------------------------------------------------------------
# Step 2: Onset trim (energy detection, NOT silenceremove)
# ---------------------------------------------------------------------------

def step_trim(input_path: str, output_path: str,
              threshold_db: float = -55, safety_margin_ms: float = 150) -> str:
    """Trim leading/trailing silence using energy onset detection.

    ⚠️ НЕ использовать silenceremove для leading trim — он режет /с/, /ц/, /ф/.
    """
    log.info("Step 2: Onset trim (–55dB, 150мс safety margin)")
    sr = 48000

    # Extract raw PCM for analysis
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run([
        'ffmpeg', '-y', '-i', input_path,
        '-f', 's16le', '-ac', '1', '-ar', str(sr),
        '-v', 'error', tmp_path
    ], check=True)

    audio = np.fromfile(tmp_path, dtype=np.int16).astype(np.float32)
    os.unlink(tmp_path)
    audio_norm = audio / (np.max(np.abs(audio)) + 1e-8)

    # Energy onset detection (two-stage: coarse → fine)
    window_ms = 5
    window_samples = int(sr * window_ms / 1000)
    n_windows = len(audio_norm) // window_samples
    energy = np.array([
        np.sqrt(np.mean(audio_norm[i * window_samples:(i + 1) * window_samples] ** 2))
        for i in range(n_windows)
    ])

    peak_energy = np.max(energy)

    # Stage 1: Coarse onset — find actual speech start (-25dB from peak)
    # This ignores noise/room tone and finds real speech
    coarse_threshold = peak_energy * (10 ** (-25 / 20))  # ~5.6% of peak
    coarse_onset = 0
    for i, e in enumerate(energy):
        if e > coarse_threshold:
            coarse_onset = i
            break

    # Stage 2: Fine onset — search backward from coarse for quiet consonants
    # (-55dB from peak catches /с/, /ц/, /ф/)
    fine_threshold = peak_energy * (10 ** (threshold_db / 20))
    fine_onset = coarse_onset
    # Search backward up to 500ms (100 windows) — consonants are short
    max_lookback = min(coarse_onset, int(500 / window_ms))
    for i in range(coarse_onset - 1, coarse_onset - max_lookback - 1, -1):
        if i < 0:
            break
        if energy[i] < fine_threshold:
            fine_onset = i + 1  # first window above threshold
            break
    else:
        fine_onset = max(coarse_onset - max_lookback, 0)

    onset_ms = fine_onset * window_ms
    trim_ms = max(onset_ms - safety_margin_ms, 0)
    trim_sec = trim_ms / 1000.0

    log.info(f"  Coarse onset at {coarse_onset * window_ms}ms, "
             f"fine onset at {onset_ms}ms, trim at {trim_ms}ms")

    # Apply trim: leading (by onset) + trailing (gentle silenceremove)
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-ss', str(trim_sec),
        '-af', 'areverse,silenceremove=start_periods=1:start_duration=0.3:'
               'start_threshold=-50dB:start_silence=0.25,areverse',
        '-ar', '48000', '-ac', '1', '-c:a', 'pcm_s24le',
        '-v', 'error',
        output_path
    ]
    subprocess.run(cmd, check=True)
    log.info(f"  → {Path(output_path).name}")
    return output_path


# ---------------------------------------------------------------------------
# Step 3: Two-pass loudnorm (ОДИН раз, после trim)
# ---------------------------------------------------------------------------

def step_loudnorm(input_path: str, output_path: str,
                  target_i: float = -16, target_tp: float = -1.5,
                  target_lra: float = 11) -> str:
    """Two-pass EBU R128 loudness normalization → MP3 320kbps.

    Этот шаг выполняется ОДИН раз, после enhance и trim.
    """
    log.info("Step 3: Two-pass loudnorm → MP3 320kbps")

    # Pass 1: Measure
    measure_cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-af', f'loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:print_format=json',
        '-f', 'null', '-',
    ]
    result = subprocess.run(measure_cmd, capture_output=True, text=True)
    stderr = result.stderr

    # Parse JSON from stderr
    json_start = stderr.rfind('{')
    json_end = stderr.rfind('}') + 1
    if json_start < 0:
        raise RuntimeError(f"Could not parse loudnorm JSON from: {stderr[-500:]}")
    measured = json.loads(stderr[json_start:json_end])

    log.info(f"  Pass 1: input_i={measured['input_i']}, input_tp={measured['input_tp']}")

    # Pass 2: Apply with linear=true
    af = (
        f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:"
        f"measured_I={measured['input_i']}:"
        f"measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:"
        f"measured_thresh={measured['input_thresh']}:"
        f"linear=true"
    )
    apply_cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-af', af,
        '-codec:a', 'libmp3lame', '-b:a', '320k',
        '-ar', '48000', '-ac', '1',
        '-v', 'error',
        output_path
    ]
    subprocess.run(apply_cmd, check=True)

    log.info(f"  → {Path(output_path).name} ({os.path.getsize(output_path)//1024} KB)")
    return output_path


# ---------------------------------------------------------------------------
# Step 4-7: HeyGen upload → generate → poll → download
# ---------------------------------------------------------------------------

def get_heygen_key() -> str:
    """Get HeyGen API key from environment."""
    key = os.environ.get('HEYGEN_API_KEY', '')
    if not key:
        raise RuntimeError("HEYGEN_API_KEY not set. Run: export HEYGEN_API_KEY=$(op read ...)")
    return key


def step_upload(audio_path: str) -> str:
    """Upload audio to HeyGen, return asset URL."""
    log.info("Step 4: Upload to HeyGen")
    key = get_heygen_key()

    result = subprocess.run([
        'curl', '-s', '-X', 'POST',
        'https://upload.heygen.com/v1/asset',
        '-H', f'X-API-KEY: {key}',
        '-H', 'Content-Type: audio/mpeg',
        '--data-binary', f'@{audio_path}'
    ], capture_output=True, text=True, check=True)

    data = json.loads(result.stdout)
    if data.get('code') != 100:
        raise RuntimeError(f"Upload failed: {data}")

    url = data['data']['url']
    asset_id = data['data']['id']
    log.info(f"  → asset_id={asset_id}")
    return url


def step_generate(audio_url: str, avatar_key: str) -> str:
    """Start HeyGen video generation, return video_id."""
    log.info(f"Step 5: Generate video ({avatar_key})")
    key = get_heygen_key()
    avatar = AVATARS[avatar_key]

    payload = {
        "video_inputs": [{
            "character": {
                "type": "avatar",
                "avatar_id": avatar['id'],
                "avatar_style": "normal"
            },
            "voice": {
                "type": "audio",
                "audio_url": audio_url
            }
        }],
        "dimension": {
            "width": avatar['width'],
            "height": avatar['height']
        },
        "test": False
    }

    result = subprocess.run([
        'curl', '-s', '-X', 'POST',
        'https://api.heygen.com/v2/video/generate',
        '-H', f'X-Api-Key: {key}',
        '-H', 'Content-Type: application/json',
        '-d', json.dumps(payload)
    ], capture_output=True, text=True, check=True)

    data = json.loads(result.stdout)
    if data.get('error'):
        raise RuntimeError(f"Generate failed: {data}")

    video_id = data['data']['video_id']
    log.info(f"  → video_id={video_id}, avatar={avatar['name']}")
    return video_id


def step_poll(video_id: str, timeout: int = 600, interval: int = 15) -> str:
    """Poll HeyGen until video is ready, return download URL."""
    log.info(f"Step 6: Polling video status (timeout={timeout}s)")
    key = get_heygen_key()
    start = time.time()

    while time.time() - start < timeout:
        result = subprocess.run([
            'curl', '-s',
            f'https://api.heygen.com/v1/video_status.get?video_id={video_id}',
            '-H', f'X-Api-Key: {key}',
        ], capture_output=True, text=True, check=True)

        data = json.loads(result.stdout)
        status = data.get('data', {}).get('status', 'unknown')
        elapsed = int(time.time() - start)
        log.info(f"  [{elapsed}s] status={status}")

        if status == 'completed':
            url = data['data']['video_url']
            log.info(f"  → Ready! URL obtained")
            return url
        elif status == 'failed':
            error = data.get('data', {}).get('error', 'unknown')
            raise RuntimeError(f"Generation failed: {error}")

        time.sleep(interval)

    raise TimeoutError(f"Video not ready after {timeout}s")


def step_download(video_url: str, output_path: str) -> str:
    """Download generated video."""
    log.info("Step 7: Download video")
    subprocess.run([
        'curl', '-s', '-L', '-o', output_path, video_url
    ], check=True)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log.info(f"  → {Path(output_path).name} ({size_mb:.1f} MB)")
    return output_path


# ---------------------------------------------------------------------------
# Step 8: Auto sync-fix (detect + correct)
# ---------------------------------------------------------------------------

def step_sync_fix(video_path: str, audio_path: str,
                  threshold_ms: float = 30.0,
                  output_path: str = None) -> dict:
    """Detect lip-sync offset and apply correction if needed.

    Uses auto_sync_fix module (mediapipe FaceMesh + cross-correlation).
    Returns dict with detection results.
    """
    log.info("Step 8: Auto sync-fix (detect + correct)")

    # Import from auto_sync_fix.py (same directory)
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    from auto_sync_fix import auto_sync_fix, SyncResult

    result = auto_sync_fix(
        video_path=video_path,
        audio_path=audio_path,
        threshold_ms=threshold_ms,
        min_confidence=0.3,
        output_path=output_path,
    )

    log.info(f"  Offset: {result.offset_ms:+.1f}ms, confidence: {result.confidence:.3f}")
    log.info(f"  Direction: {result.direction}")
    log.info(f"  Corrected: {result.correction_applied}")
    if result.corrected_file:
        log.info(f"  → {Path(result.corrected_file).name}")

    return {
        'offset_ms': result.offset_ms,
        'confidence': result.confidence,
        'direction': result.direction,
        'correction_applied': result.correction_applied,
        'corrected_file': result.corrected_file,
        'reason': result.reason,
    }


# ---------------------------------------------------------------------------
# High-level commands
# ---------------------------------------------------------------------------

def cmd_prepare(args):
    """Подготовить аудио: enhance → trim → loudnorm → MP3."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.audio).stem

    # Step 1: Enhance (EQ + comp, NO loudnorm)
    enhanced = str(output_dir / f"{stem}_enhanced.wav")
    step_enhance(args.audio, enhanced, duration=args.duration)

    # Step 2: Trim
    trimmed = str(output_dir / f"{stem}_trimmed.wav")
    step_trim(enhanced, trimmed)

    # Step 3: Loudnorm → MP3
    final = str(output_dir / f"{stem}_optimized.mp3")
    step_loudnorm(trimmed, final)

    # Cleanup intermediate WAVs (keep final MP3)
    if not args.keep_intermediate:
        os.unlink(enhanced)
        os.unlink(trimmed)

    log.info(f"\n✅ Audio ready: {final}")
    return final


def cmd_run(args):
    """Полный pipeline: prepare → upload → generate → poll → download → sync-fix."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.audio).stem
    avatar_key = args.avatar

    # Steps 1-3: Prepare audio
    enhanced = str(output_dir / f"{stem}_enhanced.wav")
    step_enhance(args.audio, enhanced, duration=args.duration)

    trimmed = str(output_dir / f"{stem}_trimmed.wav")
    step_trim(enhanced, trimmed)

    final_audio = str(output_dir / f"{stem}_optimized.mp3")
    step_loudnorm(trimmed, final_audio)

    # Steps 4-7: HeyGen
    audio_url = step_upload(final_audio)
    video_id = step_generate(audio_url, avatar_key)
    video_url = step_poll(video_id)

    video_path = str(output_dir / f"{stem}_{avatar_key}_heygen.mp4")
    step_download(video_url, video_path)

    # Step 8: Auto sync-fix
    synced_path = str(output_dir / f"{stem}_{avatar_key}_synced.mp4")
    sync_result = step_sync_fix(video_path, final_audio, output_path=synced_path)

    # Cleanup
    if not args.keep_intermediate:
        for f in [enhanced, trimmed]:
            if os.path.exists(f):
                os.unlink(f)

    # Summary
    final_video = sync_result.get('corrected_file') or video_path
    log.info(f"\n{'='*60}")
    log.info(f"✅ Pipeline complete!")
    log.info(f"   Audio:  {final_audio}")
    log.info(f"   Video:  {final_video}")
    log.info(f"   Offset: {sync_result['offset_ms']:+.1f}ms ({sync_result['direction']})")
    log.info(f"   Fixed:  {sync_result['correction_applied']}")
    log.info(f"{'='*60}")

    return {
        'audio': final_audio,
        'video_heygen': video_path,
        'video_final': final_video,
        'sync': sync_result,
    }


def cmd_sync(args):
    """Только sync-fix: detect + fix уже готового видео."""
    sync_result = step_sync_fix(
        args.video, args.audio,
        threshold_ms=args.threshold,
    )
    log.info(f"\n{json.dumps(sync_result, indent=2, ensure_ascii=False)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HeyGen Sync Pipeline — от аудио до синхронизированного видео",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline stages:
  enhance → trim → loudnorm → upload → generate → poll → download → sync-fix

Examples:
  %(prog)s prepare --audio recording.m4a --duration 15
  %(prog)s run --audio recording.m4a --avatar green --duration 15
  %(prog)s run --audio recording.m4a --avatar blue --duration 15
  %(prog)s sync --video heygen_output.mp4 --audio optimized.mp3
        """
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # --- prepare ---
    p_prep = sub.add_parser('prepare', help='Подготовить аудио (без HeyGen)')
    p_prep.add_argument('--audio', required=True, help='Входной аудиофайл')
    p_prep.add_argument('--duration', type=float, help='Вырезать первые N секунд')
    p_prep.add_argument('--output-dir', default='.', help='Папка для результатов')
    p_prep.add_argument('--keep-intermediate', action='store_true')

    # --- run ---
    p_run = sub.add_parser('run', help='Полный pipeline: аудио → синхронизированное видео')
    p_run.add_argument('--audio', required=True, help='Входной аудиофайл')
    p_run.add_argument('--avatar', choices=list(AVATARS.keys()), default='green',
                       help='Аватар: green (зелёная рубашка portrait) или blue (синяя Armani landscape)')
    p_run.add_argument('--duration', type=float, help='Вырезать первые N секунд')
    p_run.add_argument('--output-dir', default='.', help='Папка для результатов')
    p_run.add_argument('--keep-intermediate', action='store_true')

    # --- sync ---
    p_sync = sub.add_parser('sync', help='Только sync-fix (уже есть видео)')
    p_sync.add_argument('--video', required=True, help='HeyGen видео')
    p_sync.add_argument('--audio', required=True, help='Оригинальный аудиофайл')
    p_sync.add_argument('--threshold', type=float, default=30.0)

    args = parser.parse_args()

    if args.command == 'prepare':
        cmd_prepare(args)
    elif args.command == 'run':
        cmd_run(args)
    elif args.command == 'sync':
        cmd_sync(args)


if __name__ == '__main__':
    main()
