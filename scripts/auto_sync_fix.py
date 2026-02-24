#!/usr/bin/env python3
"""
Automatic lip-sync detection and correction for HeyGen Avatar IV videos.

Detects visual lip-audio offset using mediapipe FaceMesh + cross-correlation,
then corrects via ffmpeg audio replacement with measured offset.

Algorithm:
  1. Extract video frames → mediapipe FaceMesh → Mouth Aspect Ratio (MAR) signal
  2. Extract audio → RMS energy envelope (matching frame rate)
  3. FFT cross-correlation (MAR vs energy) → offset in ms + confidence
  4. Decision: |offset| > threshold AND confidence > min_confidence?
     YES → ffmpeg -itsoffset (replace audio + delay)
     NO  → leave video as-is

Usage:
  python auto_sync_fix.py detect --video 03-heygen/A2.mp4
  python auto_sync_fix.py fix --video 03-heygen/A2.mp4 --audio 02-enhanced/A2_enhanced.mp3
  python auto_sync_fix.py batch --heygen-dir 03-heygen/ --enhanced-dir 02-enhanced/

Dependencies: mediapipe, numpy, ffmpeg (CLI)
"""

import argparse
import json
import logging
import os
import subprocess
import struct
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SyncResult:
    """Result of lip-sync detection and optional correction."""
    offset_ms: float            # positive = lips lead audio (need to delay audio)
    confidence: float           # 0-1, peak-to-mean ratio of cross-correlation
    direction: str              # "lips_lead" | "lips_lag" | "in_sync" | "unknown"
    correction_applied: bool
    corrected_file: Optional[str]
    reason: str                 # human-readable explanation
    face_detection_rate: float  # fraction of frames with detected face (0-1)
    mar_signal_length: int      # number of frames analyzed


# ---------------------------------------------------------------------------
# SyncDetector — measures visual-audio offset
# ---------------------------------------------------------------------------

class SyncDetector:
    """Detect lip-sync offset by cross-correlating visual mouth motion with audio energy."""

    def __init__(self, analysis_fps: int = 25, sample_rate: int = 16000,
                 max_offset_ms: int = 200):
        """
        Args:
            analysis_fps: Frame rate for visual analysis (default: 25, native HeyGen rate)
            sample_rate: Audio sample rate for energy analysis (default: 16kHz)
            max_offset_ms: Maximum offset to search for (default: ±200ms)
        """
        self.analysis_fps = analysis_fps
        self.sample_rate = sample_rate
        self.max_offset_ms = max_offset_ms

    def extract_frames(self, video_path: str) -> list[np.ndarray]:
        """
        Extract video frames at analysis_fps using ffmpeg pipe.

        Returns list of RGB numpy arrays, each shape (height, width, 3).
        """
        # Get video dimensions
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            video_path
        ]
        probe = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        info = json.loads(probe.stdout)
        width = info['streams'][0]['width']
        height = info['streams'][0]['height']
        frame_size = width * height * 3  # RGB24

        log.info(f"Extracting frames: {width}x{height} @ {self.analysis_fps}fps")

        # Extract frames via pipe
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f'fps={self.analysis_fps}',
            '-pix_fmt', 'rgb24',
            '-f', 'rawvideo',
            '-v', 'error',
            'pipe:1'
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        frames = []
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            frames.append(frame)

        proc.wait()
        log.info(f"Extracted {len(frames)} frames ({len(frames)/self.analysis_fps:.1f}s)")
        return frames

    def compute_mar_signal(self, frames: list[np.ndarray]) -> tuple[np.ndarray, float]:
        """
        Compute Mouth Aspect Ratio (MAR) for each frame using mediapipe FaceLandmarker.

        MAR = vertical lip opening / horizontal lip width
        Higher MAR = mouth more open (correlates with louder speech)

        Returns:
            (mar_signal, face_detection_rate)
            mar_signal: zero-mean normalized array, length = len(frames)
            face_detection_rate: fraction of frames where face was detected (0-1)
        """
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options

        # Find model file
        model_path = self._find_face_landmarker_model()

        options = vision.FaceLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=model_path),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)

        mar_values = []
        detected_count = 0

        for frame in frames:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                lm = result.face_landmarks[0]  # list of NormalizedLandmark
                detected_count += 1

                # Lip landmarks (mediapipe 478-point mesh):
                #   13 = upper lip center (top of outer lip)
                #   14 = lower lip center (bottom of outer lip)
                #   78 = left mouth corner
                #  308 = right mouth corner
                top = lm[13]
                bottom = lm[14]
                left = lm[78]
                right = lm[308]

                vertical = abs(top.y - bottom.y)
                horizontal = abs(left.x - right.x) + 1e-8
                mar = vertical / horizontal
                mar_values.append(mar)
            else:
                mar_values.append(np.nan)

        landmarker.close()

        mar_signal = np.array(mar_values, dtype=np.float64)

        # Interpolate NaN values (frames where face was not detected)
        nan_mask = np.isnan(mar_signal)
        if nan_mask.any() and not nan_mask.all():
            indices = np.arange(len(mar_signal))
            mar_signal[nan_mask] = np.interp(
                indices[nan_mask],
                indices[~nan_mask],
                mar_signal[~nan_mask]
            )
        elif nan_mask.all():
            # No face detected at all
            return mar_signal, 0.0

        detection_rate = detected_count / len(frames) if frames else 0.0

        # Normalize to zero-mean for cross-correlation
        mar_signal = mar_signal - np.mean(mar_signal)

        log.info(f"MAR signal: {len(mar_signal)} frames, "
                 f"face detected in {detection_rate:.0%} of frames")
        return mar_signal, detection_rate

    def extract_audio_energy(self, video_path: str) -> np.ndarray:
        """
        Extract audio from video and compute RMS energy envelope
        aligned to the visual analysis frame rate.

        Returns zero-mean normalized energy array, length = n_frames.
        """
        # Extract audio to numpy (reuses pattern from measure_sync.py)
        audio = self._extract_audio_to_numpy(video_path)

        # Compute RMS energy per frame-aligned window
        samples_per_frame = self.sample_rate // self.analysis_fps
        n_frames = len(audio) // samples_per_frame

        energy = np.array([
            np.sqrt(np.mean(
                audio[i * samples_per_frame:(i + 1) * samples_per_frame] ** 2
            ))
            for i in range(n_frames)
        ])

        # Normalize to zero-mean
        energy = energy - np.mean(energy)

        log.info(f"Audio energy: {n_frames} frames from {len(audio)/self.sample_rate:.1f}s audio")
        return energy

    def cross_correlate(self, visual_signal: np.ndarray,
                        audio_signal: np.ndarray) -> tuple[float, float]:
        """
        FFT-based cross-correlation of visual MAR with audio energy.

        Returns (offset_ms, confidence).
            offset_ms > 0: visual leads audio (lips move before sound)
            offset_ms < 0: visual lags audio (lips move after sound)
            confidence: 0-1 (peak-to-mean ratio, normalized)
        """
        # Trim to same length
        min_len = min(len(visual_signal), len(audio_signal))
        vis = visual_signal[:min_len]
        aud = audio_signal[:min_len]

        # FFT cross-correlation (same approach as measure_sync.py)
        n = len(vis) + len(aud) - 1
        fft_size = 1
        while fft_size < n:
            fft_size *= 2

        vis_fft = np.fft.rfft(vis, fft_size)
        aud_fft = np.fft.rfft(aud, fft_size)
        # aud × conj(vis): positive lag = audio delayed relative to visual = lips lead
        correlation = np.fft.irfft(aud_fft * np.conj(vis_fft))

        # Search within ±max_offset_ms
        max_offset_frames = int(self.max_offset_ms * self.analysis_fps / 1000)
        ms_per_frame = 1000.0 / self.analysis_fps

        # Positive lags: correlation[0..max_offset_frames]
        # = visual leads audio by 0..max_offset_frames frames
        valid_pos = correlation[:max_offset_frames + 1]

        # Negative lags: correlation[-(max_offset_frames)..]
        # = visual lags audio by 1..max_offset_frames frames
        valid_neg = correlation[-max_offset_frames:]

        combined = np.concatenate([valid_pos, valid_neg])

        peak_idx = np.argmax(np.abs(combined))
        if peak_idx <= max_offset_frames:
            offset_frames = peak_idx
        else:
            offset_frames = -(len(combined) - peak_idx)

        offset_ms = offset_frames * ms_per_frame

        # Confidence: peak-to-mean ratio, normalized to 0-1
        # Note: visual-audio correlation is weaker than audio-audio,
        # so we use normalization factor 5 (not 20 as in measure_sync.py)
        peak_val = np.abs(combined[peak_idx])
        mean_val = np.mean(np.abs(combined))
        confidence = float(peak_val / (mean_val + 1e-8))
        confidence = min(confidence / 5.0, 1.0)

        log.info(f"Cross-correlation: offset={offset_ms:.1f}ms, confidence={confidence:.3f}")
        return offset_ms, confidence

    def detect(self, video_path: str) -> SyncResult:
        """
        Full detection pipeline: extract frames → MAR → audio energy → correlate.

        Returns SyncResult with measured offset and confidence.
        Does NOT apply any correction (use SyncFixer for that).
        """
        log.info(f"Analyzing: {video_path}")

        # Step 1: Extract frames
        frames = self.extract_frames(video_path)
        if len(frames) < int(2 * self.analysis_fps):  # < 2 seconds
            return SyncResult(
                offset_ms=0, confidence=0, direction="unknown",
                correction_applied=False, corrected_file=None,
                reason=f"Video too short ({len(frames)} frames, need ≥{2*self.analysis_fps})",
                face_detection_rate=0, mar_signal_length=len(frames)
            )

        # Step 2: Compute MAR signal
        mar_signal, detection_rate = self.compute_mar_signal(frames)
        if detection_rate < 0.5:
            return SyncResult(
                offset_ms=0, confidence=0, direction="unknown",
                correction_applied=False, corrected_file=None,
                reason=f"Face detected in only {detection_rate:.0%} of frames (need ≥50%)",
                face_detection_rate=detection_rate, mar_signal_length=len(frames)
            )

        # Step 3: Extract audio energy
        audio_energy = self.extract_audio_energy(video_path)

        # Step 4: Cross-correlate
        offset_ms, confidence = self.cross_correlate(mar_signal, audio_energy)

        # Determine direction
        if abs(offset_ms) <= 1.0:
            direction = "in_sync"
        elif offset_ms > 0:
            direction = "lips_lead"
        else:
            direction = "lips_lag"

        return SyncResult(
            offset_ms=round(offset_ms, 1),
            confidence=round(confidence, 3),
            direction=direction,
            correction_applied=False,
            corrected_file=None,
            reason=f"Detected: {offset_ms:.1f}ms ({direction}), confidence={confidence:.3f}",
            face_detection_rate=round(detection_rate, 3),
            mar_signal_length=len(mar_signal)
        )

    @staticmethod
    def _find_face_landmarker_model() -> str:
        """Find the face_landmarker.task model file.

        Searches in order:
        1. Same directory as this script
        2. mediapipe package directory
        3. ~/.cache/mediapipe/
        """
        candidates = [
            # Next to this script
            str(Path(__file__).parent / "face_landmarker.task"),
            # In mediapipe package
            str(Path(__import__('mediapipe').__path__[0]) / "face_landmarker.task"),
            # In user cache
            str(Path.home() / ".cache" / "mediapipe" / "face_landmarker.task"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            "face_landmarker.task not found. Download from:\n"
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task\n"
            f"Place in: {candidates[0]}"
        )

    def _extract_audio_to_numpy(self, file_path: str) -> np.ndarray:
        """Extract audio from any media file and return as normalized numpy array.

        Adapted from measure_sync.py extract_audio_to_numpy().
        """
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                'ffmpeg', '-y', '-i', file_path,
                '-vn', '-ac', '1',
                '-ar', str(self.sample_rate),
                '-f', 's16le', '-acodec', 'pcm_s16le',
                '-v', 'error',
                tmp_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            audio = np.fromfile(tmp_path, dtype=np.int16).astype(np.float32)
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            return audio
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# SyncFixer — applies correction
# ---------------------------------------------------------------------------

class SyncFixer:
    """Apply lip-sync correction by replacing audio with offset.

    Uses audio filters (adelay/atrim) instead of -itsoffset for reliable timing.
    -itsoffset is unreliable with -c:v copy (observed: 58ms instead of 80ms).
    """

    def fix(self, video_path: str, audio_path: str,
            offset_ms: float, output_path: str) -> str:
        """
        Replace HeyGen audio with original high-quality audio, applying measured offset.

        Uses -c:v copy (no video re-encoding) + audio filters for precise timing.

        Offset convention:
          offset > 0 (lips_lead): lips move before sound → trim audio start
          offset < 0 (lips_lag):  lips move after sound → pad audio with silence

        Args:
            video_path: Path to HeyGen video
            audio_path: Path to original audio (MP3/WAV)
            offset_ms: Offset in milliseconds (positive = lips lead)
            output_path: Where to save corrected video

        Returns:
            Path to corrected video file.
        """
        abs_ms = abs(offset_ms)
        abs_sec = abs_ms / 1000.0

        if offset_ms > 0:
            # Lips lead audio → advance audio (trim start so audio plays earlier)
            # This makes sound arrive earlier to match the early lip movement
            af = f"atrim=start={abs_sec:.4f},asetpts=PTS-STARTPTS"
        elif offset_ms < 0:
            # Lips lag audio → delay audio (pad with silence so audio plays later)
            # This makes sound wait for the late lip movement
            af = f"adelay={int(abs_ms)}:all=1"
        else:
            af = None  # just replace audio, no offset

        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-map', '0:v',
            '-map', '1:a',
            '-c:v', 'copy',
        ]
        if af:
            cmd.extend(['-af', af])
        cmd.extend([
            '-c:a', 'aac', '-b:a', '256k', '-ar', '48000',
            '-shortest',
            '-v', 'error',
            output_path
        ])

        log.info(f"Applying sync-fix: offset={offset_ms:+.1f}ms → {Path(output_path).name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg sync-fix failed: {result.stderr[:500]}")

        log.info(f"Saved: {output_path}")
        return output_path


# ---------------------------------------------------------------------------
# High-level function
# ---------------------------------------------------------------------------

def auto_sync_fix(video_path: str,
                  audio_path: Optional[str] = None,
                  threshold_ms: float = 30.0,
                  min_confidence: float = 0.3,
                  output_path: Optional[str] = None,
                  analysis_fps: int = 25) -> SyncResult:
    """
    Detect lip-sync offset and apply correction if needed.

    Args:
        video_path: Path to HeyGen video
        audio_path: Path to original audio (required for correction)
        threshold_ms: Minimum offset to trigger correction (default: 30ms)
        min_confidence: Minimum confidence to trust detection (default: 0.5)
        output_path: Where to save corrected video (default: {stem}_synced.mp4)
        analysis_fps: Frame rate for analysis (default: 25)

    Returns:
        SyncResult with detection details and correction status.
    """
    # Step 1: Detect
    detector = SyncDetector(analysis_fps=analysis_fps)
    result = detector.detect(video_path)

    # Step 2: Decide
    if result.direction == "unknown":
        log.warning(f"Detection failed: {result.reason}")
        return result

    if abs(result.offset_ms) <= threshold_ms:
        result.reason = (f"Offset {result.offset_ms:.1f}ms within ±{threshold_ms}ms threshold — "
                         f"no correction needed")
        log.info(result.reason)
        return result

    if result.confidence < min_confidence:
        result.reason = (f"Confidence {result.confidence:.3f} below {min_confidence} — "
                         f"skipping correction (offset was {result.offset_ms:.1f}ms)")
        log.warning(result.reason)
        return result

    if audio_path is None:
        result.reason = (f"Offset {result.offset_ms:.1f}ms detected but no audio provided — "
                         f"detect-only mode")
        log.info(result.reason)
        return result

    # Step 3: Fix
    if output_path is None:
        stem = Path(video_path).stem
        parent = Path(video_path).parent
        output_path = str(parent / f"{stem}_synced.mp4")

    fixer = SyncFixer()
    fixer.fix(video_path, audio_path, result.offset_ms, output_path)

    result.correction_applied = True
    result.corrected_file = output_path
    result.reason = (f"Corrected: offset {result.offset_ms:+.1f}ms "
                     f"(confidence {result.confidence:.3f}) → {Path(output_path).name}")
    return result


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def find_matching_audio(video_stem: str, enhanced_dir: str) -> Optional[str]:
    """Find matching enhanced audio file for a video.

    Tries patterns:
        A2 → A2_enhanced.mp3, A2_enhanced.wav, A2.mp3, A2.wav
    """
    candidates = [
        f"{video_stem}_enhanced.mp3",
        f"{video_stem}_enhanced.wav",
        f"{video_stem}.mp3",
        f"{video_stem}.wav",
    ]
    for candidate in candidates:
        path = os.path.join(enhanced_dir, candidate)
        if os.path.exists(path):
            return path
    return None


def batch_sync_fix(heygen_dir: str, enhanced_dir: str,
                   threshold_ms: float = 30.0,
                   min_confidence: float = 0.5,
                   report_path: Optional[str] = None) -> dict:
    """
    Process all MP4 files in heygen_dir, matching to audio in enhanced_dir.

    Returns dict of {filename: SyncResult}.
    """
    heygen_path = Path(heygen_dir)
    videos = sorted(heygen_path.glob("*.mp4"))

    # Filter out already-corrected files and temp directories
    videos = [v for v in videos if '_synced' not in v.stem and v.is_file()]

    if not videos:
        log.warning(f"No MP4 files found in {heygen_dir}")
        return {}

    log.info(f"Found {len(videos)} videos to analyze in {heygen_dir}")
    results = {}

    for video in videos:
        audio = find_matching_audio(video.stem, enhanced_dir)
        if audio:
            log.info(f"\n{'='*60}")
            log.info(f"Processing: {video.name} ↔ {Path(audio).name}")
            log.info(f"{'='*60}")
        else:
            log.warning(f"No matching audio for {video.name} — detect-only mode")

        try:
            result = auto_sync_fix(
                str(video), audio,
                threshold_ms=threshold_ms,
                min_confidence=min_confidence
            )
            results[video.name] = result
            log.info(f"→ {result.reason}")
        except Exception as e:
            log.error(f"Error processing {video.name}: {e}")
            results[video.name] = SyncResult(
                offset_ms=0, confidence=0, direction="error",
                correction_applied=False, corrected_file=None,
                reason=f"Error: {str(e)}", face_detection_rate=0,
                mar_signal_length=0
            )

    # Save report
    if report_path is None:
        report_path = os.path.join(heygen_dir, "sync_report.json")

    report = {name: asdict(r) for name, r in results.items()}
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info(f"\nReport saved: {report_path}")

    # Summary
    corrected = sum(1 for r in results.values() if r.correction_applied)
    skipped = len(results) - corrected
    log.info(f"Summary: {corrected} corrected, {skipped} skipped out of {len(results)} videos")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto lip-sync detection and correction for HeyGen videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s detect --video 03-heygen/A2.mp4
  %(prog)s fix --video 03-heygen/A2.mp4 --audio 02-enhanced/A2_enhanced.mp3
  %(prog)s batch --heygen-dir 03-heygen/ --enhanced-dir 02-enhanced/
  %(prog)s batch --heygen-dir 03-heygen/ --enhanced-dir 02-enhanced/ --threshold 0
        """
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- detect ---
    p_detect = subparsers.add_parser('detect', help='Detect lip-sync offset (no correction)')
    p_detect.add_argument('--video', required=True, help='Path to HeyGen video')
    p_detect.add_argument('--fps', type=int, default=25, help='Analysis FPS (default: 25)')

    # --- fix ---
    p_fix = subparsers.add_parser('fix', help='Detect and fix lip-sync')
    p_fix.add_argument('--video', required=True, help='Path to HeyGen video')
    p_fix.add_argument('--audio', required=True, help='Path to original audio')
    p_fix.add_argument('--threshold', type=float, default=30.0,
                       help='Correction threshold in ms (default: 30)')
    p_fix.add_argument('--confidence', type=float, default=0.3,
                       help='Min confidence (default: 0.3)')
    p_fix.add_argument('--output', help='Output path (default: {stem}_synced.mp4)')
    p_fix.add_argument('--fps', type=int, default=25, help='Analysis FPS (default: 25)')

    # --- batch ---
    p_batch = subparsers.add_parser('batch', help='Batch process all videos in directory')
    p_batch.add_argument('--heygen-dir', required=True, help='Directory with HeyGen videos')
    p_batch.add_argument('--enhanced-dir', required=True, help='Directory with enhanced audio')
    p_batch.add_argument('--threshold', type=float, default=30.0,
                         help='Correction threshold in ms (default: 30)')
    p_batch.add_argument('--confidence', type=float, default=0.3,
                         help='Min confidence (default: 0.3)')
    p_batch.add_argument('--report', help='Report output path (default: heygen-dir/sync_report.json)')

    args = parser.parse_args()

    if args.command == 'detect':
        detector = SyncDetector(analysis_fps=args.fps)
        result = detector.detect(args.video)
        print(f"\n{'='*50}")
        print(f"  Offset:     {result.offset_ms:+.1f} ms")
        print(f"  Direction:  {result.direction}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Face rate:  {result.face_detection_rate:.0%}")
        print(f"  Frames:     {result.mar_signal_length}")
        print(f"  Verdict:    {result.reason}")
        print(f"{'='*50}")

    elif args.command == 'fix':
        result = auto_sync_fix(
            args.video, args.audio,
            threshold_ms=args.threshold,
            min_confidence=args.confidence,
            output_path=args.output,
            analysis_fps=args.fps
        )
        print(f"\n{'='*50}")
        print(f"  Offset:     {result.offset_ms:+.1f} ms")
        print(f"  Direction:  {result.direction}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Corrected:  {result.correction_applied}")
        if result.corrected_file:
            print(f"  Output:     {result.corrected_file}")
        print(f"  Verdict:    {result.reason}")
        print(f"{'='*50}")

    elif args.command == 'batch':
        batch_sync_fix(
            args.heygen_dir, args.enhanced_dir,
            threshold_ms=args.threshold,
            min_confidence=args.confidence,
            report_path=args.report
        )


if __name__ == '__main__':
    main()
