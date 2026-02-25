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
    discriminability: float     # 0-1, (peak - 2nd_peak) / peak — higher = more certain
    direction: str              # "lips_lead" | "lips_lag" | "in_sync" | "unknown"
    correction_applied: bool
    corrected_file: Optional[str]
    reason: str                 # human-readable explanation
    face_detection_rate: float  # fraction of frames with detected face (0-1)
    mar_signal_length: int      # number of frames analyzed
    verified_residual_ms: Optional[float] = None  # residual after self-verification


@dataclass
class AdaptiveSyncResult:
    """Result of adaptive (per-chunk) lip-sync detection and correction."""
    global_offset_ms: float       # mean offset across all chunks (for comparison)
    chunk_offsets: list           # [{time_sec, offset_ms, confidence}, ...]
    offset_range_ms: float        # max - min offset (measures drift)
    correction_applied: bool
    corrected_file: Optional[str]
    method: str                   # "adaptive" | "global_fallback"
    reason: str
    face_detection_rate: float
    n_chunks: int
    global_discriminability: float = 0.0  # discriminability of global offset (0-1)


# ---------------------------------------------------------------------------
# SyncDetector — measures visual-audio offset
# ---------------------------------------------------------------------------

class SyncDetector:
    """Detect lip-sync offset by cross-correlating visual mouth motion with audio energy."""

    def __init__(self, analysis_fps: int = 25, sample_rate: int = 16000,
                 max_offset_ms: int = 400):
        """
        Args:
            analysis_fps: Frame rate for visual analysis (default: 25, native HeyGen rate)
            sample_rate: Audio sample rate for energy analysis (default: 16kHz)
            max_offset_ms: Maximum offset to search for (default: ±400ms)
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
                        audio_signal: np.ndarray) -> tuple[float, float, float]:
        """
        FFT-based cross-correlation of visual MAR with audio energy.

        Returns (offset_ms, confidence, discriminability).
            offset_ms > 0: visual leads audio (lips move before sound)
            offset_ms < 0: visual lags audio (lips move after sound)
            confidence: 0-1 (peak-to-mean ratio, normalized)
            discriminability: 0-1 ((peak - 2nd_peak) / peak) — higher = more certain
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
        # vis × conj(aud): R[k] = Σ aud[n]×vis[n+k]
        # positive k = visual is AHEAD of audio = lips_lead
        # negative k = visual is BEHIND audio = lips_lag
        correlation = np.fft.irfft(vis_fft * np.conj(aud_fft))

        # Search within ±max_offset_ms
        max_offset_frames = int(self.max_offset_ms * self.analysis_fps / 1000)
        ms_per_frame = 1000.0 / self.analysis_fps

        # Positive lags: correlation[0..max_offset_frames]
        # = visual leads audio by 0..max_offset_frames frames (lips move first)
        valid_pos = correlation[:max_offset_frames + 1]

        # Negative lags: correlation[-(max_offset_frames)..]
        # = visual lags audio by 1..max_offset_frames frames (lips move late)
        valid_neg = correlation[-max_offset_frames:]

        combined = np.concatenate([valid_pos, valid_neg])
        abs_combined = np.abs(combined)

        peak_idx = int(np.argmax(abs_combined))
        if peak_idx <= max_offset_frames:
            offset_frames = peak_idx
        else:
            offset_frames = -(len(combined) - peak_idx)

        offset_ms = offset_frames * ms_per_frame

        # Confidence: peak-to-mean ratio, normalized to 0-1
        peak_val = float(abs_combined[peak_idx])
        mean_val = float(np.mean(abs_combined))
        confidence = float(peak_val / (mean_val + 1e-8))
        confidence = min(confidence / 5.0, 1.0)

        # Discriminability: (peak - 2nd_peak) / peak
        # Find 2nd highest peak NOT adjacent to the main peak (±2 frames away)
        sorted_indices = np.argsort(-abs_combined)
        second_peak_val = 0.0
        second_peak_lag = 0
        for idx in sorted_indices:
            if idx <= max_offset_frames:
                lag = idx
            else:
                lag = -(len(combined) - idx)
            if abs(lag - offset_frames) > 2:
                second_peak_val = float(abs_combined[idx])
                second_peak_lag = lag
                break
        discriminability = (peak_val - second_peak_val) / (peak_val + 1e-8)

        log.info(f"Cross-correlation: offset={offset_ms:.1f}ms, confidence={confidence:.3f}, "
                 f"discriminability={discriminability:.3f}")

        # Log diagnostic info when discriminability is low
        if discriminability < 0.15:
            log.warning(f"  Low discriminability: peak at {offset_ms:+.0f}ms barely above "
                        f"2nd peak at {second_peak_lag * ms_per_frame:+.0f}ms "
                        f"(diff={discriminability:.1%})")
            # Show top-5 peaks for diagnostic
            top5 = []
            for idx in sorted_indices[:5]:
                if idx <= max_offset_frames:
                    lag = idx
                else:
                    lag = -(len(combined) - idx)
                top5.append(f"{lag * ms_per_frame:+.0f}ms({abs_combined[idx]:.4f})")
            log.warning(f"  Top-5 peaks: {' '.join(top5)}")

        return offset_ms, confidence, discriminability

    def self_verify(self, mar_signal: np.ndarray, audio_energy: np.ndarray,
                    detected_offset_ms: float) -> float:
        """Verify detection by shifting audio and re-detecting.

        Shifts audio by detected offset, then re-runs cross-correlation.
        Expects residual ~0ms if the detection is correct.

        Returns residual_ms (ideally close to 0).
        At 25fps, residuals < 40ms are expected due to frame quantization.
        """
        ms_per_frame = 1000.0 / self.analysis_fps
        shift_frames = int(round(detected_offset_ms / ms_per_frame))
        min_len = min(len(mar_signal), len(audio_energy))
        mar = mar_signal[:min_len]
        aud = audio_energy[:min_len]

        if shift_frames >= 0:
            shifted = np.concatenate([aud[shift_frames:], np.zeros(shift_frames)])
        else:
            shifted = np.concatenate([np.zeros(-shift_frames), aud[:min_len + shift_frames]])

        residual_ms, _, _ = self.cross_correlate(mar, shifted[:min_len])
        log.info(f"Self-verification: shift={detected_offset_ms:+.1f}ms → residual={residual_ms:+.1f}ms "
                 f"({'OK' if abs(residual_ms) <= ms_per_frame else 'CHECK'})")
        return residual_ms

    # ------------------------------------------------------------------
    # Adaptive sync methods (sub-frame precision, per-chunk analysis)
    # ------------------------------------------------------------------

    @staticmethod
    def _parabolic_interpolation(correlation: np.ndarray, peak_idx: int) -> float:
        """Sub-sample peak detection via parabolic interpolation.

        Fits parabola through 3 points around peak.
        Returns fractional index of true peak (~20ms precision at 25fps).
        """
        if peak_idx <= 0 or peak_idx >= len(correlation) - 1:
            return float(peak_idx)

        y_prev = float(correlation[peak_idx - 1])
        y_peak = float(correlation[peak_idx])
        y_next = float(correlation[peak_idx + 1])

        denom = y_prev - 2 * y_peak + y_next
        if abs(denom) < 1e-10:
            return float(peak_idx)

        delta = 0.5 * (y_prev - y_next) / denom
        # Clamp delta to ±0.5 (parabola vertex should be within ±0.5 samples)
        delta = max(-0.5, min(0.5, delta))
        return peak_idx + delta

    def cross_correlate_subframe(self, visual_signal: np.ndarray,
                                 audio_signal: np.ndarray) -> tuple[float, float, float]:
        """Cross-correlation with sub-frame precision via parabolic interpolation.

        Same as cross_correlate() but adds parabolic peak refinement.
        Returns (offset_ms, confidence, discriminability) with ~20ms precision at 25fps.
        """
        min_len = min(len(visual_signal), len(audio_signal))
        vis = visual_signal[:min_len]
        aud = audio_signal[:min_len]

        n = len(vis) + len(aud) - 1
        fft_size = 1
        while fft_size < n:
            fft_size *= 2

        vis_fft = np.fft.rfft(vis, fft_size)
        aud_fft = np.fft.rfft(aud, fft_size)
        # vis × conj(aud): R[k] = Σ aud[n]×vis[n+k]
        # positive k = lips_lead, negative k = lips_lag
        correlation = np.fft.irfft(vis_fft * np.conj(aud_fft))

        max_offset_frames = int(self.max_offset_ms * self.analysis_fps / 1000)
        ms_per_frame = 1000.0 / self.analysis_fps

        valid_pos = correlation[:max_offset_frames + 1]
        valid_neg = correlation[-max_offset_frames:]
        combined = np.concatenate([valid_pos, valid_neg])

        peak_idx = int(np.argmax(np.abs(combined)))

        # Integer offset
        if peak_idx <= max_offset_frames:
            offset_frames_int = peak_idx
        else:
            offset_frames_int = -(len(combined) - peak_idx)

        # Sub-frame refinement via parabolic interpolation
        abs_combined = np.abs(combined)
        refined_idx = self._parabolic_interpolation(abs_combined, peak_idx)

        if refined_idx <= max_offset_frames:
            offset_frames = refined_idx
        else:
            offset_frames = -(len(combined) - refined_idx)

        offset_ms = offset_frames * ms_per_frame

        # Confidence
        peak_val = float(abs_combined[peak_idx])
        mean_val = float(np.mean(abs_combined))
        confidence = float(peak_val / (mean_val + 1e-8))
        confidence = min(confidence / 5.0, 1.0)

        # Discriminability
        sorted_indices = np.argsort(-abs_combined)
        second_peak_val = 0.0
        for idx in sorted_indices:
            if idx <= max_offset_frames:
                lag = idx
            else:
                lag = -(len(combined) - idx)
            if abs(lag - offset_frames_int) > 2:
                second_peak_val = float(abs_combined[idx])
                break
        discriminability = (peak_val - second_peak_val) / (peak_val + 1e-8)

        return offset_ms, confidence, discriminability

    def _cross_correlate_constrained(self, visual_signal: np.ndarray,
                                     audio_signal: np.ndarray,
                                     baseline_ms: float,
                                     search_radius_frames: int = 2) -> tuple[float, float]:
        """Cross-correlation constrained to ±search_radius around baseline offset.

        Phase 2 of adaptive detection: refines the global offset per chunk.
        Searches only in a narrow window (±2 frames = ±80ms at 25fps) around
        the known global offset, then applies parabolic interpolation for
        sub-frame precision (~20ms).

        Args:
            visual_signal: Zero-mean MAR signal (chunk)
            audio_signal: Zero-mean energy signal (chunk)
            baseline_ms: Global offset to center the search around
            search_radius_frames: ±frames to search around baseline (default: 2)

        Returns:
            (offset_ms, confidence) with sub-frame precision.
        """
        min_len = min(len(visual_signal), len(audio_signal))
        vis = visual_signal[:min_len]
        aud = audio_signal[:min_len]

        ms_per_frame = 1000.0 / self.analysis_fps
        baseline_frames = baseline_ms / ms_per_frame  # can be fractional

        # Full FFT cross-correlation
        n = len(vis) + len(aud) - 1
        fft_size = 1
        while fft_size < n:
            fft_size *= 2

        vis_fft = np.fft.rfft(vis, fft_size)
        aud_fft = np.fft.rfft(aud, fft_size)
        # vis × conj(aud): R[k] = Σ aud[n]×vis[n+k]
        # positive k = lips_lead, negative k = lips_lag
        correlation = np.fft.irfft(vis_fft * np.conj(aud_fft))

        # Build search window centered on baseline
        # baseline_frames > 0 → positive lag (indices 0..N in correlation)
        # baseline_frames < 0 → negative lag (indices from end of correlation)
        center_int = int(round(baseline_frames))
        search_min = center_int - search_radius_frames
        search_max = center_int + search_radius_frames

        # Collect (correlation_index, lag_frames) pairs within search window
        candidates = []
        for lag in range(search_min, search_max + 1):
            if lag >= 0:
                idx = lag
            else:
                idx = len(correlation) + lag
            if 0 <= idx < len(correlation):
                candidates.append((idx, lag, abs(float(correlation[idx]))))

        if not candidates:
            return baseline_ms, 0.0

        # Find best peak within constrained window
        best = max(candidates, key=lambda x: x[2])
        best_idx, best_lag, peak_val = best

        # Parabolic interpolation for sub-frame precision
        abs_corr = np.abs(correlation)
        refined_idx = self._parabolic_interpolation(abs_corr, best_idx)

        # Convert refined index back to lag in frames
        if best_lag >= 0:
            refined_lag = refined_idx  # positive lag: index = lag
        else:
            refined_lag = refined_idx - len(correlation)  # negative lag

        offset_ms = refined_lag * ms_per_frame

        # Confidence: peak relative to mean in the search window
        window_vals = [c[2] for c in candidates]
        mean_val = np.mean(window_vals) if window_vals else 1e-8
        confidence = float(peak_val / (mean_val + 1e-8))
        confidence = min(confidence / 3.0, 1.0)  # normalize (narrower window → higher ratio)

        return offset_ms, confidence

    def detect_adaptive(self, video_path: str,
                        chunk_sec: float = 2.0,
                        overlap: float = 0.5,
                        min_chunk_confidence: float = 0.2) -> AdaptiveSyncResult:
        """Two-phase adaptive lip-sync detection with sub-frame precision.

        Phase 1: Compute global offset using full signal (reliable baseline).
        Phase 2: Per-chunk constrained refinement — search within ±2 frames
                 of global offset, then parabolic interpolation for ~20ms precision.

        If per-chunk results are too noisy (std > 30ms), falls back to
        uniform global offset applied across all chunks.

        Args:
            video_path: Path to HeyGen video
            chunk_sec: Chunk duration in seconds (default: 2.0)
            overlap: Overlap fraction between chunks (default: 0.5 = 50%)
            min_chunk_confidence: Minimum confidence for a chunk to be trusted

        Returns:
            AdaptiveSyncResult with per-chunk offsets and smoothed curve.
        """
        log.info(f"Adaptive analysis: {video_path}")

        # Step 1: Extract full MAR signal and audio energy
        frames = self.extract_frames(video_path)
        if len(frames) < int(2 * self.analysis_fps):
            return AdaptiveSyncResult(
                global_offset_ms=0, chunk_offsets=[], offset_range_ms=0,
                correction_applied=False, corrected_file=None,
                method="global_fallback",
                reason=f"Video too short ({len(frames)} frames)",
                face_detection_rate=0, n_chunks=0
            )

        mar_signal, detection_rate = self.compute_mar_signal(frames)
        if detection_rate < 0.5:
            return AdaptiveSyncResult(
                global_offset_ms=0, chunk_offsets=[], offset_range_ms=0,
                correction_applied=False, corrected_file=None,
                method="global_fallback",
                reason=f"Face detected in only {detection_rate:.0%} of frames",
                face_detection_rate=detection_rate, n_chunks=0
            )

        audio_energy = self.extract_audio_energy(video_path)

        # Phase 1: Global offset (reliable baseline)
        global_offset_ms, global_confidence, global_disc = self.cross_correlate_subframe(
            mar_signal, audio_energy
        )
        log.info(f"Phase 1 — Global: {global_offset_ms:+.1f}ms "
                 f"(confidence={global_confidence:.3f}, discriminability={global_disc:.3f})")

        # Phase 2: Per-chunk constrained refinement
        chunk_frames = int(chunk_sec * self.analysis_fps)
        step_frames = int(chunk_frames * (1 - overlap))
        min_len = min(len(mar_signal), len(audio_energy))

        chunk_offsets = []
        i = 0
        while i + chunk_frames <= min_len:
            chunk_mar = mar_signal[i:i + chunk_frames]
            chunk_aud = audio_energy[i:i + chunk_frames]
            center_time = (i + chunk_frames / 2) / self.analysis_fps

            # Constrained search: ±2 frames around global offset
            offset_ms, confidence = self._cross_correlate_constrained(
                chunk_mar, chunk_aud,
                baseline_ms=global_offset_ms,
                search_radius_frames=2
            )

            chunk_offsets.append({
                'time_sec': round(center_time, 3),
                'offset_ms': round(offset_ms, 1),
                'confidence': round(confidence, 3),
                'frame_start': i,
                'frame_end': i + chunk_frames,
            })
            i += step_frames

        if not chunk_offsets:
            return AdaptiveSyncResult(
                global_offset_ms=round(global_offset_ms, 1),
                chunk_offsets=[], offset_range_ms=0,
                correction_applied=False, corrected_file=None,
                method="global_fallback",
                reason="No chunks could be analyzed",
                face_detection_rate=detection_rate, n_chunks=0
            )

        # Step 3: Analyze chunk consistency
        offsets = np.array([c['offset_ms'] for c in chunk_offsets])
        offset_std = float(np.std(offsets))
        offset_range = float(np.max(offsets) - np.min(offsets))

        # Choose smoothing strategy based on per-chunk variance:
        #   std ≤ 30ms:  mild smoothing (median window=3) — per-chunk variation is real
        #   30 < std ≤ 80ms: heavy smoothing (median window=5) — mix of real drift + noise
        #   std > 80ms:  fall back to uniform global offset — too noisy to trust
        if offset_std > 80.0:
            log.info(f"Per-chunk std={offset_std:.1f}ms > 80ms — "
                     f"too noisy, using uniform global offset {global_offset_ms:+.1f}ms")
            for c in chunk_offsets:
                c['offset_smoothed_ms'] = round(global_offset_ms, 1)
            method = "global_uniform"
        else:
            # Median filter to remove outliers, window size depends on noise level
            median_window = 5 if offset_std > 30.0 else 3
            if offset_std > 30.0:
                log.info(f"Per-chunk std={offset_std:.1f}ms > 30ms — "
                         f"using wider smoothing (median window={median_window})")

            if len(offsets) >= median_window:
                try:
                    from scipy.ndimage import median_filter
                    offsets_smoothed = median_filter(offsets, size=median_window)
                except (ImportError, ModuleNotFoundError):
                    hw = median_window // 2
                    offsets_smoothed = np.array([
                        np.median(offsets[max(0, j-hw):j+hw+1])
                        for j in range(len(offsets))
                    ])
            else:
                offsets_smoothed = offsets.copy()

            # Rate-of-change limiter: prevent sharp offset jumps that cause
            # audible pitch artifacts in the time-warped audio.
            # Max 20ms change per chunk (~1s step) = ~2% speed change = inaudible.
            # Without this, a 90ms jump over 1s = 9% speed change = very audible.
            max_rate_ms = 20.0  # max offset change per chunk step
            chunk_step_sec = chunk_sec * (1 - overlap)
            max_rate_per_step = max_rate_ms  # per chunk step

            rate_limited = np.array(offsets_smoothed, dtype=np.float64)
            # Forward pass: limit how fast offset can increase or decrease
            for j in range(1, len(rate_limited)):
                delta = rate_limited[j] - rate_limited[j - 1]
                if abs(delta) > max_rate_per_step:
                    rate_limited[j] = rate_limited[j - 1] + np.sign(delta) * max_rate_per_step
            # Backward pass: symmetric smoothing from the end
            for j in range(len(rate_limited) - 2, -1, -1):
                delta = rate_limited[j] - rate_limited[j + 1]
                if abs(delta) > max_rate_per_step:
                    rate_limited[j] = rate_limited[j + 1] + np.sign(delta) * max_rate_per_step

            offsets_smoothed = rate_limited
            for j, c in enumerate(chunk_offsets):
                c['offset_smoothed_ms'] = round(float(offsets_smoothed[j]), 1)
            method = "adaptive"

        # Log results
        log.info(f"Phase 2 — {len(chunk_offsets)} chunks, "
                 f"global={global_offset_ms:+.1f}ms, "
                 f"std={offset_std:.1f}ms, range={offset_range:.1f}ms, "
                 f"method={method}")
        for c in chunk_offsets:
            smoothed = c.get('offset_smoothed_ms', c['offset_ms'])
            log.info(f"  t={c['time_sec']:.1f}s: raw={c['offset_ms']:+.1f}ms "
                     f"→ smoothed={smoothed:+.1f}ms (conf={c['confidence']:.3f})")

        return AdaptiveSyncResult(
            global_offset_ms=round(global_offset_ms, 1),
            chunk_offsets=chunk_offsets,
            offset_range_ms=round(offset_range, 1),
            correction_applied=False,
            corrected_file=None,
            method=method,
            reason=(f"{method}: global={global_offset_ms:+.1f}ms, "
                    f"std={offset_std:.1f}ms, range={offset_range:.1f}ms, "
                    f"{len(chunk_offsets)} chunks"),
            face_detection_rate=round(detection_rate, 3),
            n_chunks=len(chunk_offsets),
            global_discriminability=round(global_disc, 3)
        )

    def detect(self, video_path: str) -> SyncResult:
        """
        Full detection pipeline: extract frames → MAR → audio energy → correlate → verify.

        Returns SyncResult with measured offset, confidence, discriminability,
        and self-verification residual.
        Does NOT apply any correction (use SyncFixer for that).
        """
        log.info(f"Analyzing: {video_path}")

        # Step 1: Extract frames
        frames = self.extract_frames(video_path)
        if len(frames) < int(2 * self.analysis_fps):  # < 2 seconds
            return SyncResult(
                offset_ms=0, confidence=0, discriminability=0, direction="unknown",
                correction_applied=False, corrected_file=None,
                reason=f"Video too short ({len(frames)} frames, need ≥{2*self.analysis_fps})",
                face_detection_rate=0, mar_signal_length=len(frames)
            )

        # Step 2: Compute MAR signal
        mar_signal, detection_rate = self.compute_mar_signal(frames)
        if detection_rate < 0.5:
            return SyncResult(
                offset_ms=0, confidence=0, discriminability=0, direction="unknown",
                correction_applied=False, corrected_file=None,
                reason=f"Face detected in only {detection_rate:.0%} of frames (need ≥50%)",
                face_detection_rate=detection_rate, mar_signal_length=len(frames)
            )

        # Step 3: Extract audio energy
        audio_energy = self.extract_audio_energy(video_path)

        # Step 4: Cross-correlate
        offset_ms, confidence, discriminability = self.cross_correlate(mar_signal, audio_energy)

        # Step 5: Self-verification (shift audio and re-detect → expect ~0ms)
        residual_ms = self.self_verify(mar_signal, audio_energy, offset_ms)

        # Determine direction
        if abs(offset_ms) <= 1.0:
            direction = "in_sync"
        elif offset_ms > 0:
            direction = "lips_lead"
        else:
            direction = "lips_lag"

        reason = (f"Detected: {offset_ms:.1f}ms ({direction}), "
                  f"confidence={confidence:.3f}, discriminability={discriminability:.3f}")
        if abs(residual_ms) > 1000.0 / self.analysis_fps:
            reason += f", residual={residual_ms:+.1f}ms (check)"

        return SyncResult(
            offset_ms=round(offset_ms, 1),
            confidence=round(confidence, 3),
            discriminability=round(discriminability, 3),
            direction=direction,
            correction_applied=False,
            corrected_file=None,
            reason=reason,
            face_detection_rate=round(detection_rate, 3),
            mar_signal_length=len(mar_signal),
            verified_residual_ms=round(residual_ms, 1)
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

# ---------------------------------------------------------------------------
# Audio time-warp for adaptive sync
# ---------------------------------------------------------------------------

def apply_adaptive_offset(audio: np.ndarray, sr: int,
                          offset_curve_ms: np.ndarray,
                          frame_times: np.ndarray) -> np.ndarray:
    """Apply variable offset to audio via sample-level interpolation.

    For each output sample, computes where in the original audio to read from
    based on the offset curve. Uses linear interpolation — zero artifacts.

    Args:
        audio: Original audio as float32 numpy array
        sr: Sample rate (e.g., 48000)
        offset_curve_ms: Offset in ms at each frame time (positive = delay audio)
        frame_times: Time of each frame in seconds

    Returns:
        Warped audio array (same dtype as input).
    """
    n_samples = len(audio)
    output_times = np.arange(n_samples, dtype=np.float64) / sr

    # Interpolate offset curve to every audio sample
    offset_per_sample = np.interp(output_times, frame_times, offset_curve_ms)

    # Source time: where in original audio to read from
    # If offset > 0 (lips lead): advance audio → read from LATER position (skip start)
    #   source = output + offset → positive shift → skips beginning
    # If offset < 0 (lips lag): delay audio → read from EARLIER position (before start = silence)
    #   source = output + offset → negative shift → source < 0 at start → zero-padded
    source_times = output_times + offset_per_sample / 1000.0
    source_indices = source_times * sr

    # Zero-pad for negative indices (silence before audio starts when delaying)
    negative_mask = source_indices < 0
    source_indices = np.clip(source_indices, 0, n_samples - 2)
    idx_floor = source_indices.astype(np.int64)
    frac = (source_indices - idx_floor).astype(np.float32)

    # Linear interpolation between adjacent samples
    output = audio[idx_floor] * (1 - frac) + audio[np.minimum(idx_floor + 1, n_samples - 1)] * frac

    # Zero out samples where source was before audio start (true silence)
    output[negative_mask] = 0.0

    return output.astype(audio.dtype)


# ---------------------------------------------------------------------------
# SyncFixer — applies global correction (single offset)
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

        # Get video duration to limit output (prevents black frames
        # when audio is longer than video)
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=duration',
            '-of', 'csv=p=0',
            video_path
        ]
        probe = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        video_duration = float(probe.stdout.strip())

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
        ])
        # Limit output to video duration — prevents black frames
        # when audio file is longer than video (e.g., V1: 15s video + 455s audio)
        cmd.extend(['-t', f'{video_duration:.6f}'])
        cmd.extend(['-v', 'error', output_path])

        log.info(f"Applying sync-fix: offset={offset_ms:+.1f}ms → {Path(output_path).name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg sync-fix failed: {result.stderr[:500]}")

        log.info(f"Saved: {output_path}")
        return output_path


# ---------------------------------------------------------------------------
# AdaptiveSyncFixer — applies per-chunk correction via audio time-warp
# ---------------------------------------------------------------------------

class AdaptiveSyncFixer:
    """Apply adaptive lip-sync correction via audio time-warping.

    Instead of a single global offset, applies a smooth offset curve
    that varies over time. Uses numpy sample-level interpolation —
    zero artifacts, no rubberband/atempo needed.
    """

    def fix(self, video_path: str, audio_path: str,
            adaptive_result: 'AdaptiveSyncResult',
            output_path: str, audio_sr: int = 48000) -> str:
        """Apply adaptive correction by time-warping the audio.

        1. Load original audio into numpy
        2. Build smooth offset curve from chunk offsets
        3. Apply time-warp via apply_adaptive_offset()
        4. Save warped audio to temp WAV
        5. Mux with video: ffmpeg -i video -i warped.wav -c:v copy

        Args:
            video_path: Path to HeyGen video
            audio_path: Path to original audio
            adaptive_result: Result from detect_adaptive()
            output_path: Where to save corrected video
            audio_sr: Sample rate for audio processing (default: 48000)

        Returns:
            Path to corrected video.
        """
        # Step 1: Load audio at high sample rate
        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
            tmp_raw = tmp.name
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', audio_path,
                '-vn', '-ac', '1', '-ar', str(audio_sr),
                '-f', 's16le', '-acodec', 'pcm_s16le',
                '-v', 'error', tmp_raw
            ], capture_output=True, check=True)
            audio = np.fromfile(tmp_raw, dtype=np.int16).astype(np.float32)
        finally:
            if os.path.exists(tmp_raw):
                os.unlink(tmp_raw)

        # Step 2: Build offset curve from chunk offsets
        reliable = [c for c in adaptive_result.chunk_offsets
                    if c.get('offset_smoothed_ms') is not None]
        if not reliable:
            reliable = [c for c in adaptive_result.chunk_offsets
                        if c['confidence'] >= 0.2]
        if not reliable:
            # No reliable data — fall back to global offset
            log.warning("No reliable chunks for adaptive fix, using global offset")
            reliable = [{'time_sec': 0, 'offset_smoothed_ms': adaptive_result.global_offset_ms}]

        frame_times = np.array([c['time_sec'] for c in reliable])
        offsets_ms = np.array([c.get('offset_smoothed_ms', c['offset_ms']) for c in reliable])

        # Extend curve to cover full audio duration
        audio_duration = len(audio) / audio_sr
        if frame_times[0] > 0:
            frame_times = np.insert(frame_times, 0, 0.0)
            offsets_ms = np.insert(offsets_ms, 0, offsets_ms[0])
        if frame_times[-1] < audio_duration:
            frame_times = np.append(frame_times, audio_duration)
            offsets_ms = np.append(offsets_ms, offsets_ms[-1])

        log.info(f"Offset curve: {len(frame_times)} points, "
                 f"range [{np.min(offsets_ms):.1f}, {np.max(offsets_ms):.1f}] ms")

        # Step 3: Apply time-warp
        warped = apply_adaptive_offset(audio, audio_sr, offsets_ms, frame_times)

        # Step 4: Save warped audio to temp WAV
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_wav = tmp.name
        try:
            # Write raw PCM, then convert to WAV via ffmpeg
            with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp2:
                tmp_raw2 = tmp2.name
            warped.astype(np.int16).tofile(tmp_raw2)

            subprocess.run([
                'ffmpeg', '-y',
                '-f', 's16le', '-ar', str(audio_sr), '-ac', '1',
                '-i', tmp_raw2,
                '-c:a', 'pcm_s16le',
                '-v', 'error',
                tmp_wav
            ], capture_output=True, check=True)
            os.unlink(tmp_raw2)

            # Step 5: Mux video + warped audio
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', tmp_wav,
                '-map', '0:v', '-map', '1:a',
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '256k', '-ar', '48000',
                '-shortest',
                '-v', 'error',
                output_path
            ]
            log.info(f"Muxing adaptive-synced video → {Path(output_path).name}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg mux failed: {result.stderr[:500]}")

            log.info(f"Saved: {output_path}")
            return output_path
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)


# ---------------------------------------------------------------------------
# High-level functions
# ---------------------------------------------------------------------------

def auto_sync_fix(video_path: str,
                  audio_path: Optional[str] = None,
                  threshold_ms: float = 30.0,
                  min_confidence: float = 0.3,
                  min_discriminability: float = 0.1,
                  output_path: Optional[str] = None,
                  analysis_fps: int = 25,
                  force: bool = False) -> SyncResult:
    """
    Detect lip-sync offset and apply correction if needed.

    Args:
        video_path: Path to HeyGen video
        audio_path: Path to original audio (required for correction)
        threshold_ms: Minimum offset to trigger correction (default: 30ms)
        min_confidence: Minimum confidence to trust detection (default: 0.3)
        min_discriminability: Minimum discriminability to trust detection (default: 0.1).
            When discriminability is below this threshold, the cross-correlation peak
            is barely distinguishable from noise and the detected offset may be wrong.
            Use --force to override.
        output_path: Where to save corrected video (default: {stem}_synced.mp4)
        analysis_fps: Frame rate for analysis (default: 25)
        force: If True, apply correction even with low discriminability (default: False)

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

    # Discriminability safeguard: when the peak is barely above noise,
    # the detected offset is unreliable and may cause WORSE sync.
    # V3_G1 case: disc=0.027, detected +320ms, true offset was -160ms.
    if result.discriminability < min_discriminability and not force:
        result.reason = (
            f"⚠ UNRELIABLE: discriminability {result.discriminability:.3f} below {min_discriminability} — "
            f"the detected offset {result.offset_ms:+.1f}ms may be WRONG. "
            f"Cross-correlation peak is barely above noise. "
            f"Use --force to override, or manually test offsets with the fix command."
        )
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


def adaptive_sync_fix(video_path: str,
                      audio_path: Optional[str] = None,
                      threshold_ms: float = 30.0,
                      chunk_sec: float = 2.0,
                      overlap: float = 0.5,
                      min_chunk_confidence: float = 0.2,
                      min_discriminability: float = 0.1,
                      output_path: Optional[str] = None,
                      analysis_fps: int = 25,
                      detect_only: bool = False,
                      force: bool = False) -> AdaptiveSyncResult:
    """
    Detect per-chunk lip-sync offset and apply adaptive correction.

    Uses 2-second overlapping chunks with sub-frame precision (~20ms at 25fps).
    Correction is applied via audio time-warp (sample-level interpolation).

    Falls back to global offset if:
      - Video too short (<4s)
      - All chunks below confidence threshold
      - Fewer than 2 reliable chunks

    Args:
        video_path: Path to HeyGen video
        audio_path: Path to original audio (required for correction)
        threshold_ms: Minimum mean offset to trigger correction (default: 30ms)
        chunk_sec: Chunk duration in seconds (default: 2.0)
        overlap: Overlap fraction between chunks (default: 0.5)
        min_chunk_confidence: Min confidence for a chunk (default: 0.2)
        min_discriminability: Minimum discriminability to trust detection (default: 0.1)
        output_path: Where to save corrected video (default: {stem}_adaptive_synced.mp4)
        analysis_fps: Frame rate for analysis (default: 25)
        detect_only: If True, only detect — don't apply correction
        force: If True, apply correction even with low discriminability

    Returns:
        AdaptiveSyncResult with per-chunk offsets and correction status.
    """
    # Step 1: Detect
    detector = SyncDetector(analysis_fps=analysis_fps)
    result = detector.detect_adaptive(
        video_path,
        chunk_sec=chunk_sec,
        overlap=overlap,
        min_chunk_confidence=min_chunk_confidence
    )

    # Step 2: Decide
    if result.method == "global_fallback" and result.n_chunks == 0:
        log.warning(f"Adaptive detection failed: {result.reason}")
        return result

    if abs(result.global_offset_ms) <= threshold_ms:
        result.reason = (f"Mean offset {result.global_offset_ms:.1f}ms within "
                         f"±{threshold_ms}ms threshold — no correction needed. "
                         f"Range: {result.offset_range_ms:.1f}ms across {result.n_chunks} chunks")
        log.info(result.reason)
        return result

    # Discriminability safeguard
    if result.global_discriminability < min_discriminability and not force:
        result.reason = (
            f"⚠ UNRELIABLE: discriminability {result.global_discriminability:.3f} below "
            f"{min_discriminability} — the detected offset {result.global_offset_ms:+.1f}ms "
            f"may be WRONG. Cross-correlation peak is barely above noise. "
            f"Use --force to override, or manually test offsets."
        )
        log.warning(result.reason)
        return result

    if detect_only:
        result.reason = (f"Adaptive detect: mean={result.global_offset_ms:.1f}ms, "
                         f"range={result.offset_range_ms:.1f}ms, "
                         f"{result.n_chunks} chunks (detect-only mode)")
        log.info(result.reason)
        return result

    if audio_path is None:
        result.reason = (f"Mean offset {result.global_offset_ms:.1f}ms detected "
                         f"but no audio provided — detect-only mode")
        log.info(result.reason)
        return result

    # Step 3: Fix via adaptive time-warp
    if output_path is None:
        stem = Path(video_path).stem
        parent = Path(video_path).parent
        output_path = str(parent / f"{stem}_adaptive_synced.mp4")

    fixer = AdaptiveSyncFixer()
    fixer.fix(video_path, audio_path, result, output_path)

    result.correction_applied = True
    result.corrected_file = output_path
    result.reason = (f"Adaptive corrected: mean={result.global_offset_ms:+.1f}ms, "
                     f"range={result.offset_range_ms:.1f}ms, "
                     f"{result.n_chunks} chunks → {Path(output_path).name}")
    log.info(result.reason)
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
                offset_ms=0, confidence=0, discriminability=0, direction="error",
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
# Sweep — generate test files for manual offset selection
# ---------------------------------------------------------------------------

def sweep_sync(video_path: str, audio_path: str,
               range_ms: int = 200, step_ms: int = 40,
               output_dir: Optional[str] = None) -> list[str]:
    """
    Generate test videos with different offsets for manual comparison.

    Creates N files from -range_ms to +range_ms (step_ms increments).
    User listens and picks the best-sounding offset.

    Offset convention:
      negative = lips_lag (audio delayed with adelay, -c:v copy)
      positive = lips_lead (audio trimmed with atrim, -c:v copy)
      zero = no offset, just replace audio

    Args:
        video_path: Path to HeyGen video
        audio_path: Path to original audio
        range_ms: Maximum offset to test (default: ±200ms)
        step_ms: Step between offsets (default: 40ms)
        output_dir: Where to save test files (default: same directory as video)

    Returns:
        List of generated file paths.
    """
    if output_dir is None:
        output_dir = str(Path(video_path).parent)

    stem = Path(video_path).stem
    files = []

    # Get video duration to limit output (prevents black frames
    # when audio is longer than video)
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=duration',
        '-of', 'csv=p=0',
        video_path
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    video_duration = float(probe.stdout.strip())

    offsets = list(range(-range_ms, range_ms + 1, step_ms))
    log.info(f"Sweep: generating {len(offsets)} test files for {Path(video_path).name}")
    log.info(f"  Range: {-range_ms:+d}ms to {+range_ms:+d}ms, step {step_ms}ms")

    for offset_ms in offsets:
        if offset_ms < 0:
            label = f"neg{abs(offset_ms)}ms"
            af = f"adelay={abs(offset_ms)}:all=1"
        elif offset_ms > 0:
            label = f"pos{offset_ms}ms"
            abs_sec = offset_ms / 1000.0
            af = f"atrim=start={abs_sec:.4f},asetpts=PTS-STARTPTS"
        else:
            label = "0ms"
            af = None

        out_path = os.path.join(output_dir, f"{stem}_sweep_{label}.mp4")

        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-map', '0:v', '-map', '1:a',
            '-c:v', 'copy',
        ]
        if af:
            cmd.extend(['-af', af])
        cmd.extend([
            '-c:a', 'aac', '-b:a', '256k', '-ar', '48000',
            '-t', f'{video_duration:.6f}',
            '-v', 'error',
            out_path
        ])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"  Failed: {label} — {result.stderr[:200]}")
            continue

        files.append(out_path)

    log.info(f"Generated {len(files)} test files in {output_dir}")
    return files


def manual_sync_fix(video_path: str, audio_path: str,
                    offset_ms: float, output_path: Optional[str] = None) -> str:
    """
    Apply a manually specified offset without detection.

    Uses -c:v copy (no re-encoding) + audio filter for precise timing.
    For negative offsets (lips_lag), uses adelay (audio delayed, full audio preserved).
    For positive offsets (lips_lead), uses atrim (audio trimmed from start).

    Args:
        video_path: Path to HeyGen video
        audio_path: Path to original audio
        offset_ms: Offset in ms (negative = delay audio, positive = trim audio)
        output_path: Output path (default: {stem}_synced.mp4)

    Returns:
        Path to corrected video file.
    """
    if output_path is None:
        stem = Path(video_path).stem
        parent = Path(video_path).parent
        output_path = str(parent / f"{stem}_synced.mp4")

    fixer = SyncFixer()
    fixer.fix(video_path, audio_path, offset_ms, output_path)
    return output_path


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
  %(prog)s fix --video 03-heygen/A2.mp4 --audio 02-enhanced/A2.mp3
  %(prog)s fix --video 03-heygen/A2.mp4 --audio 02-enhanced/A2.mp3 --offset -80
  %(prog)s sweep --video 03-heygen/A2.mp4 --audio 02-enhanced/A2.mp3
  %(prog)s sweep --video 03-heygen/A2.mp4 --audio 02-enhanced/A2.mp3 --range 160 --step 20
  %(prog)s batch --heygen-dir 03-heygen/ --enhanced-dir 02-enhanced/
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
    p_fix.add_argument('--min-disc', type=float, default=0.1,
                       help='Min discriminability to trust detection (default: 0.1)')
    p_fix.add_argument('--force', action='store_true',
                       help='Apply correction even with low discriminability')
    p_fix.add_argument('--offset', type=float, default=None,
                       help='Manual offset in ms (skip detection). Negative = delay audio, positive = trim audio')
    p_fix.add_argument('--output', help='Output path (default: {stem}_synced.mp4)')
    p_fix.add_argument('--fps', type=int, default=25, help='Analysis FPS (default: 25)')

    # --- sweep ---
    p_sweep = subparsers.add_parser('sweep',
        help='Generate test files with different offsets for manual comparison')
    p_sweep.add_argument('--video', required=True, help='Path to HeyGen video')
    p_sweep.add_argument('--audio', required=True, help='Path to original audio')
    p_sweep.add_argument('--range', type=int, default=200, dest='sweep_range',
                         help='Max offset to test in ms (default: ±200ms)')
    p_sweep.add_argument('--step', type=int, default=40,
                         help='Step between offsets in ms (default: 40ms)')
    p_sweep.add_argument('--output-dir', help='Directory for test files (default: same as video)')

    # --- adaptive ---
    p_adaptive = subparsers.add_parser('adaptive',
        help='Adaptive per-chunk sync detection + correction (20ms precision)')
    p_adaptive.add_argument('--video', required=True, help='Path to HeyGen video')
    p_adaptive.add_argument('--audio', help='Path to original audio (omit for detect-only)')
    p_adaptive.add_argument('--threshold', type=float, default=30.0,
                            help='Mean offset threshold in ms (default: 30)')
    p_adaptive.add_argument('--chunk-sec', type=float, default=2.0,
                            help='Chunk duration in seconds (default: 2.0)')
    p_adaptive.add_argument('--overlap', type=float, default=0.5,
                            help='Chunk overlap fraction (default: 0.5)')
    p_adaptive.add_argument('--min-confidence', type=float, default=0.2,
                            help='Min chunk confidence (default: 0.2)')
    p_adaptive.add_argument('--min-disc', type=float, default=0.1,
                            help='Min discriminability to trust detection (default: 0.1)')
    p_adaptive.add_argument('--force', action='store_true',
                            help='Apply correction even with low discriminability')
    p_adaptive.add_argument('--output', help='Output path (default: {stem}_adaptive_synced.mp4)')
    p_adaptive.add_argument('--fps', type=int, default=25, help='Analysis FPS (default: 25)')
    p_adaptive.add_argument('--detect-only', action='store_true',
                            help='Only detect offsets, do not apply correction')

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
        print(f"  Offset:          {result.offset_ms:+.1f} ms")
        print(f"  Direction:       {result.direction}")
        print(f"  Confidence:      {result.confidence:.3f}")
        print(f"  Discriminability:{result.discriminability:.3f}"
              f"{'  ⚠ LOW' if result.discriminability < 0.1 else ''}")
        print(f"  Face rate:       {result.face_detection_rate:.0%}")
        print(f"  Frames:          {result.mar_signal_length}")
        if result.verified_residual_ms is not None:
            residual_ok = abs(result.verified_residual_ms) <= 40
            print(f"  Verification:    {result.verified_residual_ms:+.1f}ms"
                  f" {'✓' if residual_ok else '⚠ CHECK'}")
        print(f"  Verdict:         {result.reason}")
        print(f"{'='*50}")

    elif args.command == 'fix':
        if args.offset is not None:
            # Manual offset — skip detection entirely
            output_path = args.output
            if output_path is None:
                stem = Path(args.video).stem
                parent = Path(args.video).parent
                output_path = str(parent / f"{stem}_synced.mp4")

            out = manual_sync_fix(args.video, args.audio, args.offset, output_path)
            direction = "lips_lag" if args.offset < 0 else ("lips_lead" if args.offset > 0 else "in_sync")
            print(f"\n{'='*50}")
            print(f"  Mode:            manual offset")
            print(f"  Offset:          {args.offset:+.1f} ms")
            print(f"  Direction:       {direction}")
            print(f"  Output:          {out}")
            print(f"{'='*50}")
        else:
            # Auto-detect offset
            result = auto_sync_fix(
                args.video, args.audio,
                threshold_ms=args.threshold,
                min_confidence=args.confidence,
                min_discriminability=args.min_disc,
                output_path=args.output,
                analysis_fps=args.fps,
                force=args.force
            )
            print(f"\n{'='*50}")
            print(f"  Offset:          {result.offset_ms:+.1f} ms")
            print(f"  Direction:       {result.direction}")
            print(f"  Confidence:      {result.confidence:.3f}")
            print(f"  Discriminability:{result.discriminability:.3f}"
                  f"{'  ⚠ LOW' if result.discriminability < 0.1 else ''}")
            print(f"  Corrected:       {result.correction_applied}")
            if result.corrected_file:
                print(f"  Output:          {result.corrected_file}")
            if result.verified_residual_ms is not None:
                residual_ok = abs(result.verified_residual_ms) <= 40
                print(f"  Verification:    {result.verified_residual_ms:+.1f}ms"
                      f" {'✓' if residual_ok else '⚠ CHECK'}")
            print(f"  Verdict:         {result.reason}")
            print(f"{'='*50}")

    elif args.command == 'adaptive':
        result = adaptive_sync_fix(
            args.video,
            audio_path=args.audio,
            threshold_ms=args.threshold,
            chunk_sec=args.chunk_sec,
            overlap=args.overlap,
            min_chunk_confidence=args.min_confidence,
            min_discriminability=args.min_disc,
            output_path=args.output,
            analysis_fps=args.fps,
            detect_only=args.detect_only,
            force=args.force
        )
        print(f"\n{'='*60}")
        print(f"  Method:       {result.method}")
        print(f"  Mean offset:  {result.global_offset_ms:+.1f} ms")
        print(f"  Discriminability: {result.global_discriminability:.3f}"
              f"{'  ⚠ LOW' if result.global_discriminability < 0.1 else ''}")
        print(f"  Offset range: {result.offset_range_ms:.1f} ms (drift)")
        print(f"  Chunks:       {result.n_chunks}")
        print(f"  Face rate:    {result.face_detection_rate:.0%}")
        print(f"  Corrected:    {result.correction_applied}")
        if result.corrected_file:
            print(f"  Output:       {result.corrected_file}")
        print(f"  Verdict:      {result.reason}")
        print(f"{'='*60}")
        # Print per-chunk table
        if result.chunk_offsets:
            print(f"\n  {'Time':>6s}  {'Offset':>8s}  {'Smoothed':>9s}  {'Conf':>6s}  {'Status'}")
            print(f"  {'─'*6}  {'─'*8}  {'─'*9}  {'─'*6}  {'─'*6}")
            for c in result.chunk_offsets:
                smoothed = c.get('offset_smoothed_ms', '—')
                if isinstance(smoothed, (int, float)):
                    smoothed_str = f"{smoothed:+.1f}ms"
                else:
                    smoothed_str = str(smoothed)
                status = "OK" if c['confidence'] >= args.min_confidence else "LOW"
                print(f"  {c['time_sec']:6.1f}s  {c['offset_ms']:+7.1f}ms"
                      f"  {smoothed_str:>9s}  {c['confidence']:6.3f}  {status}")

    elif args.command == 'sweep':
        files = sweep_sync(
            args.video, args.audio,
            range_ms=args.sweep_range,
            step_ms=args.step,
            output_dir=args.output_dir
        )
        stem = Path(args.video).stem
        print(f"\n{'='*60}")
        print(f"  SWEEP: {len(files)} test files generated")
        print(f"  Range: {-args.sweep_range:+d}ms to {+args.sweep_range:+d}ms, step {args.step}ms")
        print(f"{'='*60}")
        print()
        print("  Listen to each file and find the best sync:")
        print()
        for f in files:
            name = Path(f).name
            # Extract offset from filename
            if '_sweep_neg' in name:
                ms = name.split('_sweep_neg')[1].replace('ms.mp4', '')
                print(f"    {name:45s}  offset: -{ms}ms")
            elif '_sweep_pos' in name:
                ms = name.split('_sweep_pos')[1].replace('ms.mp4', '')
                print(f"    {name:45s}  offset: +{ms}ms")
            elif '_sweep_0ms' in name:
                print(f"    {name:45s}  offset:  0ms")
        print()
        print("  Once you pick the best offset, create the final file:")
        print(f"    python {sys.argv[0]} fix --video {args.video} --audio {args.audio} --offset <OFFSET>")
        print()
        print("  Then clean up sweep files:")
        print(f"    rm {Path(files[0]).parent}/{stem}_sweep_*.mp4")
        print(f"{'='*60}")

    elif args.command == 'batch':
        batch_sync_fix(
            args.heygen_dir, args.enhanced_dir,
            threshold_ms=args.threshold,
            min_confidence=args.confidence,
            report_path=args.report
        )


if __name__ == '__main__':
    main()
