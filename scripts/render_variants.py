#!/usr/bin/env python3
"""
Render 3 style variants for comparison.
Uses existing shorts_subtitles.py pipeline for audio/transcribe/parse,
then generates different ASS styles and burns them.

Variants:
  V6A "Clean Karaoke"  — karaoke white→yellow, purple solid overlays, 3 words, minimal outline
  V6B "Yellow Accent"  — same karaoke + overlay key words highlighted in yellow
  V6C "Purple Pill"    — karaoke on purple pill background, purple overlays

Usage:
  python3 render_variants.py --video A2.mp4 --script scenario.md --output ./temp/ --code A2
"""

import sys
import os
import re
from pathlib import Path

# Import functions from the main pipeline
sys.path.insert(0, str(Path(__file__).parent))
from shorts_subtitles import (
    extract_audio, transcribe_words, parse_scenario, align_overlays,
    group_words_into_phrases, postprocess_subtitle_text, burn_subtitles,
    Word, INTER_GAP_MS,
)


# ─── Brand colors (BGR for ASS) ──────────────────────────────────
# ASS uses &HBBGGRR format
# Purple #7B2FBE = R=123 G=47 B=190 → &HBE2F7B&
# Yellow #FFD700 = R=255 G=215 B=0 → &H00D7FF&

def _ass_color(r, g, b, a=0):
    """Create pysubs2.Color from RGB values."""
    import pysubs2
    return pysubs2.Color(r, g, b, a)


def _build_karaoke_line(phrase, phrases, pi, max_words):
    """Build karaoke tag line from phrase words with grammar postprocessing."""
    is_last = (pi == len(phrases) - 1)

    raw_parts = []
    for j, w in enumerate(phrase):
        if j < len(phrase) - 1:
            dur_cs = max(int((phrase[j + 1].start - w.start) * 100), 10)
        else:
            dur_cs = max(int((w.end - w.start) * 100), 10)
        raw_parts.append((dur_cs, w.text))

    full_text = " ".join(t for _, t in raw_parts)
    full_text = postprocess_subtitle_text(full_text, is_last_phrase=is_last)
    corrected_words = full_text.split()

    karaoke_parts = []
    ci, ri = 0, 0
    while ri < len(raw_parts) and ci < len(corrected_words):
        cw = corrected_words[ci]
        merged_dur = raw_parts[ri][0]
        peek = ri + 1
        while peek < len(raw_parts):
            next_raw = raw_parts[peek][1].strip(".,!?;:")
            if next_raw and next_raw.lower() in cw.lower():
                merged_dur += raw_parts[peek][0]
                peek += 1
            else:
                break
        karaoke_parts.append((merged_dur, cw))
        ri = peek
        ci += 1
    while ci < len(corrected_words):
        karaoke_parts.append((10, corrected_words[ci]))
        ci += 1

    return karaoke_parts


def generate_v6a(words, overlays, output_path, max_words=3):
    """V6A: Clean Karaoke — white→yellow sweep, purple solid overlays, minimal outline."""
    import pysubs2

    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = "1080"
    subs.info["PlayResY"] = "1920"
    subs.info["ScaledBorderAndShadow"] = "yes"

    # Karaoke style: white → yellow, thin outline, italic
    style_sub = pysubs2.SSAStyle()
    style_sub.fontname = "Arial"
    style_sub.fontsize = 68
    style_sub.bold = True
    style_sub.italic = True
    style_sub.primarycolor = _ass_color(255, 255, 255, 0)      # white (before)
    style_sub.secondarycolor = _ass_color(0, 215, 255, 0)      # yellow (after highlight)
    style_sub.outlinecolor = _ass_color(0, 0, 0, 0)
    style_sub.backcolor = _ass_color(0, 0, 0, 180)
    style_sub.outline = 1.5
    style_sub.shadow = 2
    style_sub.alignment = 2
    style_sub.marginv = 180
    style_sub.marginl = 60
    style_sub.marginr = 60
    subs.styles["Karaoke"] = style_sub

    # Overlay: PURPLE solid background (#7B2FBE)
    style_ov = pysubs2.SSAStyle()
    style_ov.fontname = "Arial"
    style_ov.fontsize = 58
    style_ov.bold = True
    style_ov.primarycolor = _ass_color(255, 255, 255, 0)       # white text
    style_ov.outlinecolor = _ass_color(123, 47, 190, 0)        # purple outline (creates border)
    style_ov.backcolor = _ass_color(123, 47, 190, 0)           # purple solid box
    style_ov.outline = 12                                       # thick = padding effect
    style_ov.shadow = 0
    style_ov.alignment = 8
    style_ov.marginv = 160
    style_ov.marginl = 40
    style_ov.marginr = 40
    style_ov.borderstyle = 3                                    # opaque box
    subs.styles["Overlay"] = style_ov

    # CTA: green on purple
    style_cta = pysubs2.SSAStyle()
    style_cta.fontname = "Arial"
    style_cta.fontsize = 54
    style_cta.bold = True
    style_cta.primarycolor = _ass_color(50, 255, 50, 0)
    style_cta.outlinecolor = _ass_color(123, 47, 190, 0)
    style_cta.backcolor = _ass_color(123, 47, 190, 0)
    style_cta.outline = 12
    style_cta.shadow = 0
    style_cta.alignment = 8
    style_cta.marginv = 160
    style_cta.marginl = 40
    style_cta.marginr = 40
    style_cta.borderstyle = 3
    subs.styles["CTA"] = style_cta

    # --- Karaoke events ---
    phrases = group_words_into_phrases(words, max_words=max_words, min_words=1)
    for pi, phrase in enumerate(phrases):
        line_start_ms = int(phrase[0].start * 1000)
        line_end_ms = int(phrase[-1].end * 1000) + 150
        if pi > 0:
            prev_end = int(phrases[pi - 1][-1].end * 1000) + 150
            if line_start_ms < prev_end + INTER_GAP_MS:
                line_start_ms = prev_end + INTER_GAP_MS

        parts = _build_karaoke_line(phrase, phrases, pi, max_words)
        line_text = " ".join(f"{{\\kf{d}}}{w}" for d, w in parts)
        subs.events.append(pysubs2.SSAEvent(
            start=line_start_ms, end=line_end_ms, style="Karaoke", text=line_text
        ))

    # --- Overlay events ---
    for oi, ov in enumerate(overlays):
        is_last = (oi == len(overlays) - 1)
        is_cta = is_last and any(kw in ov["text"].upper() for kw in ["ПОДПИШИСЬ", "SUBSCRIBE"])
        if is_cta:
            text = f"{{\\fad(300,0)}}↓↓↓ {ov['text']} ↓↓↓"
            subs.events.append(pysubs2.SSAEvent(
                start=int(ov["start"]*1000), end=int(ov["end"]*1000), style="CTA", text=text
            ))
        else:
            subs.events.append(pysubs2.SSAEvent(
                start=int(ov["start"]*1000), end=int(ov["end"]*1000), style="Overlay",
                text=f"{{\\fad(200,200)}}{ov['text']}"
            ))

    subs.events.sort(key=lambda e: e.start)
    subs.save(output_path)
    print(f"  V6A saved: {output_path}")


def generate_v6b(words, overlays, output_path, max_words=3):
    """V6B: Yellow Accent — karaoke + overlays with yellow-highlighted key words."""
    import pysubs2

    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = "1080"
    subs.info["PlayResY"] = "1920"
    subs.info["ScaledBorderAndShadow"] = "yes"

    # Same karaoke as V6A
    style_sub = pysubs2.SSAStyle()
    style_sub.fontname = "Arial"
    style_sub.fontsize = 68
    style_sub.bold = True
    style_sub.italic = True
    style_sub.primarycolor = _ass_color(255, 255, 255, 0)
    style_sub.secondarycolor = _ass_color(0, 215, 255, 0)       # yellow
    style_sub.outlinecolor = _ass_color(0, 0, 0, 0)
    style_sub.backcolor = _ass_color(0, 0, 0, 180)
    style_sub.outline = 1.5
    style_sub.shadow = 2
    style_sub.alignment = 2
    style_sub.marginv = 180
    style_sub.marginl = 60
    style_sub.marginr = 60
    subs.styles["Karaoke"] = style_sub

    # Overlay: purple solid, but key words will have inline yellow highlight
    style_ov = pysubs2.SSAStyle()
    style_ov.fontname = "Arial"
    style_ov.fontsize = 58
    style_ov.bold = True
    style_ov.primarycolor = _ass_color(255, 255, 255, 0)
    style_ov.outlinecolor = _ass_color(123, 47, 190, 0)
    style_ov.backcolor = _ass_color(123, 47, 190, 0)
    style_ov.outline = 12
    style_ov.shadow = 0
    style_ov.alignment = 8
    style_ov.marginv = 160
    style_ov.marginl = 40
    style_ov.marginr = 40
    style_ov.borderstyle = 3
    subs.styles["Overlay"] = style_ov

    # Overlay with yellow highlight box style
    style_ov_yellow = pysubs2.SSAStyle()
    style_ov_yellow.fontname = "Arial"
    style_ov_yellow.fontsize = 58
    style_ov_yellow.bold = True
    style_ov_yellow.primarycolor = _ass_color(255, 255, 255, 0)
    style_ov_yellow.outlinecolor = _ass_color(0, 180, 255, 0)   # yellow-orange outline
    style_ov_yellow.backcolor = _ass_color(0, 180, 255, 0)      # yellow-orange box
    style_ov_yellow.outline = 12
    style_ov_yellow.shadow = 0
    style_ov_yellow.alignment = 8
    style_ov_yellow.marginv = 160
    style_ov_yellow.marginl = 40
    style_ov_yellow.marginr = 40
    style_ov_yellow.borderstyle = 3
    subs.styles["OverlayYellow"] = style_ov_yellow

    # CTA
    style_cta = pysubs2.SSAStyle()
    style_cta.fontname = "Arial"
    style_cta.fontsize = 54
    style_cta.bold = True
    style_cta.primarycolor = _ass_color(50, 255, 50, 0)
    style_cta.outlinecolor = _ass_color(123, 47, 190, 0)
    style_cta.backcolor = _ass_color(123, 47, 190, 0)
    style_cta.outline = 12
    style_cta.shadow = 0
    style_cta.alignment = 8
    style_cta.marginv = 160
    style_cta.marginl = 40
    style_cta.marginr = 40
    style_cta.borderstyle = 3
    subs.styles["CTA"] = style_cta

    # Karaoke events (same as V6A)
    phrases = group_words_into_phrases(words, max_words=max_words, min_words=1)
    for pi, phrase in enumerate(phrases):
        line_start_ms = int(phrase[0].start * 1000)
        line_end_ms = int(phrase[-1].end * 1000) + 150
        if pi > 0:
            prev_end = int(phrases[pi - 1][-1].end * 1000) + 150
            if line_start_ms < prev_end + INTER_GAP_MS:
                line_start_ms = prev_end + INTER_GAP_MS

        parts = _build_karaoke_line(phrase, phrases, pi, max_words)
        line_text = " ".join(f"{{\\kf{d}}}{w}" for d, w in parts)
        subs.events.append(pysubs2.SSAEvent(
            start=line_start_ms, end=line_end_ms, style="Karaoke", text=line_text
        ))

    # --- Overlay events with yellow accent on key words ---
    # Key words to highlight: numbers, legal refs, strong words
    HIGHLIGHT_PATTERNS = [
        r'\d+[\-]?ФЗ',       # 522-ФЗ
        r'\d+\s*МЛРД',       # 29 МЛРД
        r'\d+[\.,]?\d*%',    # 5.9%
        r'БЕЗ СУДА',
        r'УКРАДЕНО',
        r'ЗАМОРОЗИТ',
        r'ПОДПИШИСЬ',
    ]

    def highlight_overlay_text(text):
        """Wrap key words in yellow highlight using inline ASS overrides."""
        result = text
        # Yellow text color in ASS: {\1c&H00D7FF&} (BGR)
        for pattern in HIGHLIGHT_PATTERNS:
            result = re.sub(
                f'({pattern})',
                r'{\\1c&H00D7FF&}\1{\\1c&HFFFFFF&}',
                result,
                flags=re.IGNORECASE
            )
        return result

    for oi, ov in enumerate(overlays):
        is_last = (oi == len(overlays) - 1)
        is_cta = is_last and any(kw in ov["text"].upper() for kw in ["ПОДПИШИСЬ", "SUBSCRIBE"])
        if is_cta:
            text = f"{{\\fad(300,0)}}↓↓↓ {ov['text']} ↓↓↓"
            subs.events.append(pysubs2.SSAEvent(
                start=int(ov["start"]*1000), end=int(ov["end"]*1000), style="CTA", text=text
            ))
        else:
            highlighted = highlight_overlay_text(ov["text"])
            subs.events.append(pysubs2.SSAEvent(
                start=int(ov["start"]*1000), end=int(ov["end"]*1000), style="Overlay",
                text=f"{{\\fad(200,200)}}{highlighted}"
            ))

    subs.events.sort(key=lambda e: e.start)
    subs.save(output_path)
    print(f"  V6B saved: {output_path}")


def generate_v6c(words, overlays, output_path, max_words=3):
    """V6C: Purple Pill — karaoke on purple pill background, purple overlays."""
    import pysubs2

    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = "1080"
    subs.info["PlayResY"] = "1920"
    subs.info["ScaledBorderAndShadow"] = "yes"

    # Karaoke ON PURPLE PILL: borderstyle=3 with purple backcolor
    style_sub = pysubs2.SSAStyle()
    style_sub.fontname = "Arial"
    style_sub.fontsize = 64
    style_sub.bold = True
    style_sub.italic = True
    style_sub.primarycolor = _ass_color(255, 255, 255, 0)       # white
    style_sub.secondarycolor = _ass_color(0, 215, 255, 0)       # yellow fill sweep
    style_sub.outlinecolor = _ass_color(123, 47, 190, 0)        # purple outline
    style_sub.backcolor = _ass_color(123, 47, 190, 0)           # purple box
    style_sub.outline = 10
    style_sub.shadow = 0
    style_sub.alignment = 2
    style_sub.marginv = 180
    style_sub.marginl = 60
    style_sub.marginr = 60
    style_sub.borderstyle = 3                                    # opaque box!
    subs.styles["Karaoke"] = style_sub

    # Overlay: lighter purple (more contrast)
    style_ov = pysubs2.SSAStyle()
    style_ov.fontname = "Arial"
    style_ov.fontsize = 58
    style_ov.bold = True
    style_ov.primarycolor = _ass_color(255, 255, 255, 0)
    style_ov.outlinecolor = _ass_color(100, 30, 160, 0)         # darker purple
    style_ov.backcolor = _ass_color(100, 30, 160, 0)
    style_ov.outline = 14
    style_ov.shadow = 0
    style_ov.alignment = 8
    style_ov.marginv = 160
    style_ov.marginl = 40
    style_ov.marginr = 40
    style_ov.borderstyle = 3
    subs.styles["Overlay"] = style_ov

    # CTA
    style_cta = pysubs2.SSAStyle()
    style_cta.fontname = "Arial"
    style_cta.fontsize = 54
    style_cta.bold = True
    style_cta.primarycolor = _ass_color(50, 255, 50, 0)
    style_cta.outlinecolor = _ass_color(100, 30, 160, 0)
    style_cta.backcolor = _ass_color(100, 30, 160, 0)
    style_cta.outline = 14
    style_cta.shadow = 0
    style_cta.alignment = 8
    style_cta.marginv = 160
    style_cta.marginl = 40
    style_cta.marginr = 40
    style_cta.borderstyle = 3
    subs.styles["CTA"] = style_cta

    # Karaoke events
    phrases = group_words_into_phrases(words, max_words=max_words, min_words=1)
    for pi, phrase in enumerate(phrases):
        line_start_ms = int(phrase[0].start * 1000)
        line_end_ms = int(phrase[-1].end * 1000) + 150
        if pi > 0:
            prev_end = int(phrases[pi - 1][-1].end * 1000) + 150
            if line_start_ms < prev_end + INTER_GAP_MS:
                line_start_ms = prev_end + INTER_GAP_MS

        parts = _build_karaoke_line(phrase, phrases, pi, max_words)
        line_text = " ".join(f"{{\\kf{d}}}{w}" for d, w in parts)
        subs.events.append(pysubs2.SSAEvent(
            start=line_start_ms, end=line_end_ms, style="Karaoke", text=line_text
        ))

    # Overlay events
    for oi, ov in enumerate(overlays):
        is_last = (oi == len(overlays) - 1)
        is_cta = is_last and any(kw in ov["text"].upper() for kw in ["ПОДПИШИСЬ", "SUBSCRIBE"])
        if is_cta:
            text = f"{{\\fad(300,0)}}↓↓↓ {ov['text']} ↓↓↓"
            subs.events.append(pysubs2.SSAEvent(
                start=int(ov["start"]*1000), end=int(ov["end"]*1000), style="CTA", text=text
            ))
        else:
            subs.events.append(pysubs2.SSAEvent(
                start=int(ov["start"]*1000), end=int(ov["end"]*1000), style="Overlay",
                text=f"{{\\fad(200,200)}}{ov['text']}"
            ))

    subs.events.sort(key=lambda e: e.start)
    subs.save(output_path)
    print(f"  V6C saved: {output_path}")


def generate_v7(words, overlays, output_path, max_words=3):
    """V7: Center Stage — V6B base + corrected positioning/font to match top shorts.

    Key changes vs V6B:
    - Subtitles: bigger (76px), NO italic, thicker outline (3px), raised from bottom (marginv 380)
    - Overlays: moved from TOP to CENTER of frame (alignment=5), bigger font (66px)
    - CTA: centered in frame
    - Yellow highlights on key words preserved from V6B
    """
    import pysubs2

    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = "1080"
    subs.info["PlayResY"] = "1920"
    subs.info["ScaledBorderAndShadow"] = "yes"

    # Karaoke: bigger, bolder, NO italic, raised from bottom
    style_sub = pysubs2.SSAStyle()
    style_sub.fontname = "Arial"
    style_sub.fontsize = 76
    style_sub.bold = True
    style_sub.italic = False                                         # NO italic!
    style_sub.primarycolor = _ass_color(255, 255, 255, 0)           # white
    style_sub.secondarycolor = _ass_color(0, 215, 255, 0)           # yellow sweep
    style_sub.outlinecolor = _ass_color(0, 0, 0, 0)                 # black outline
    style_sub.backcolor = _ass_color(0, 0, 0, 180)
    style_sub.outline = 3.0                                          # thicker (was 1.5)
    style_sub.shadow = 3                                             # deeper shadow
    style_sub.alignment = 2                                          # bottom-center
    style_sub.marginv = 380                                          # raised (was 180)
    style_sub.marginl = 50
    style_sub.marginr = 50
    subs.styles["Karaoke"] = style_sub

    # Overlay: CENTER of frame (alignment=5), bigger, yellow accents
    style_ov = pysubs2.SSAStyle()
    style_ov.fontname = "Arial"
    style_ov.fontsize = 66                                           # bigger (was 58)
    style_ov.bold = True
    style_ov.primarycolor = _ass_color(255, 255, 255, 0)
    style_ov.outlinecolor = _ass_color(123, 47, 190, 0)             # purple
    style_ov.backcolor = _ass_color(123, 47, 190, 0)                # purple box
    style_ov.outline = 14                                            # more padding (was 12)
    style_ov.shadow = 0
    style_ov.alignment = 5                                           # CENTER! (was 8=top)
    style_ov.marginv = 100                                           # slight offset above center
    style_ov.marginl = 40
    style_ov.marginr = 40
    style_ov.borderstyle = 3
    subs.styles["Overlay"] = style_ov

    # CTA: centered in frame
    style_cta = pysubs2.SSAStyle()
    style_cta.fontname = "Arial"
    style_cta.fontsize = 60                                          # bigger
    style_cta.bold = True
    style_cta.primarycolor = _ass_color(50, 255, 50, 0)
    style_cta.outlinecolor = _ass_color(123, 47, 190, 0)
    style_cta.backcolor = _ass_color(123, 47, 190, 0)
    style_cta.outline = 14
    style_cta.shadow = 0
    style_cta.alignment = 5                                          # CENTER
    style_cta.marginv = 100
    style_cta.marginl = 40
    style_cta.marginr = 40
    style_cta.borderstyle = 3
    subs.styles["CTA"] = style_cta

    # --- Karaoke events ---
    phrases = group_words_into_phrases(words, max_words=max_words, min_words=1)
    for pi, phrase in enumerate(phrases):
        line_start_ms = int(phrase[0].start * 1000)
        line_end_ms = int(phrase[-1].end * 1000) + 150
        if pi > 0:
            prev_end = int(phrases[pi - 1][-1].end * 1000) + 150
            if line_start_ms < prev_end + INTER_GAP_MS:
                line_start_ms = prev_end + INTER_GAP_MS

        parts = _build_karaoke_line(phrase, phrases, pi, max_words)
        line_text = " ".join(f"{{\\kf{d}}}{w}" for d, w in parts)
        subs.events.append(pysubs2.SSAEvent(
            start=line_start_ms, end=line_end_ms, style="Karaoke", text=line_text
        ))

    # --- Overlay events with yellow accent (same as V6B) ---
    HIGHLIGHT_PATTERNS = [
        r'\d+[\-]?ФЗ',
        r'\d+\s*МЛРД',
        r'\d+[\.,]?\d*%',
        r'БЕЗ СУДА',
        r'УКРАДЕНО',
        r'ЗАМОРОЗИТ',
        r'ПОДПИШИСЬ',
    ]

    def highlight_overlay_text(text):
        result = text
        for pattern in HIGHLIGHT_PATTERNS:
            result = re.sub(
                f'({pattern})',
                r'{\\1c&H00D7FF&}\1{\\1c&HFFFFFF&}',
                result,
                flags=re.IGNORECASE
            )
        return result

    for oi, ov in enumerate(overlays):
        is_last = (oi == len(overlays) - 1)
        is_cta = is_last and any(kw in ov["text"].upper() for kw in ["ПОДПИШИСЬ", "SUBSCRIBE"])
        if is_cta:
            text = f"{{\\fad(300,0)}}↓↓↓ {ov['text']} ↓↓↓"
            subs.events.append(pysubs2.SSAEvent(
                start=int(ov["start"]*1000), end=int(ov["end"]*1000), style="CTA", text=text
            ))
        else:
            highlighted = highlight_overlay_text(ov["text"])
            subs.events.append(pysubs2.SSAEvent(
                start=int(ov["start"]*1000), end=int(ov["end"]*1000), style="Overlay",
                text=f"{{\\fad(200,200)}}{highlighted}"
            ))

    subs.events.sort(key=lambda e: e.start)
    subs.save(output_path)
    print(f"  V7 saved: {output_path}")


def _v8_highlight_patterns():
    """Key words to highlight yellow in overlays."""
    return [
        r'\d+[\-]?ФЗ',
        r'\d+\s*МЛРД',
        r'\d+[\.,]?\d*%',
        r'БЕЗ СУДА',
        r'УКРАДЕНО',
        r'ЗАМОРОЗИТ',
        r'ПОДПИШИСЬ',
        r'НЕ ВАШИ',
        r'ЦЕНТРОБАНК',
        r'ЦБ',
    ]


def _v8_highlight_text(text):
    """Highlight key words with yellow inline override, rest stays white."""
    result = text
    for pattern in _v8_highlight_patterns():
        result = re.sub(
            f'({pattern})',
            r'{\\1c&H00D7FF&}\1{\\1c&HFFFFFF&}',
            result,
            flags=re.IGNORECASE
        )
    return result


def generate_v8a(words, overlays, output_path, max_words=3):
    """V8A: Dual Layer — big static subtitles at bottom + keyword overlay at center.

    Based on analysis of top shorts + external viral shorts:
    - Bottom: BIG white bold static subtitles (no karaoke, no background)
    - Center: Overlay key phrase (yellow highlights on key words, no background box)
    - NO italic, NO borderstyle=3 boxes
    """
    import pysubs2

    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = "1080"
    subs.info["PlayResY"] = "1920"
    subs.info["ScaledBorderAndShadow"] = "yes"

    # BOTTOM: Big static subtitles — white, bold, no background
    style_sub = pysubs2.SSAStyle()
    style_sub.fontname = "Arial"
    style_sub.fontsize = 80
    style_sub.bold = True
    style_sub.italic = False
    style_sub.primarycolor = _ass_color(255, 255, 255, 0)       # white
    style_sub.secondarycolor = _ass_color(255, 255, 255, 0)
    style_sub.outlinecolor = _ass_color(0, 0, 0, 0)             # black outline
    style_sub.backcolor = _ass_color(0, 0, 0, 120)              # subtle shadow
    style_sub.outline = 3.5
    style_sub.shadow = 3
    style_sub.alignment = 2                                       # bottom-center
    style_sub.marginv = 250                                       # raised from very bottom
    style_sub.marginl = 50
    style_sub.marginr = 50
    subs.styles["BigSubs"] = style_sub

    # CENTER: Keyword overlay — yellow/white, no box, floating text
    style_kw = pysubs2.SSAStyle()
    style_kw.fontname = "Arial"
    style_kw.fontsize = 68
    style_kw.bold = True
    style_kw.italic = False
    style_kw.primarycolor = _ass_color(255, 255, 255, 0)        # white (default)
    style_kw.outlinecolor = _ass_color(60, 20, 120, 0)          # dark purple outline
    style_kw.backcolor = _ass_color(0, 0, 0, 150)
    style_kw.outline = 3
    style_kw.shadow = 4
    style_kw.alignment = 5                                        # center
    style_kw.marginv = 80                                         # slightly above center
    style_kw.marginl = 40
    style_kw.marginr = 40
    subs.styles["Keyword"] = style_kw

    # CTA: green floating text at center
    style_cta = pysubs2.SSAStyle()
    style_cta.fontname = "Arial"
    style_cta.fontsize = 62
    style_cta.bold = True
    style_cta.primarycolor = _ass_color(50, 255, 50, 0)         # green
    style_cta.outlinecolor = _ass_color(60, 20, 120, 0)
    style_cta.backcolor = _ass_color(0, 0, 0, 150)
    style_cta.outline = 3
    style_cta.shadow = 4
    style_cta.alignment = 5
    style_cta.marginv = 80
    style_cta.marginl = 40
    style_cta.marginr = 40
    subs.styles["CTA"] = style_cta

    # --- Static subtitle events (no karaoke!) ---
    phrases = group_words_into_phrases(words, max_words=max_words, min_words=1)
    for pi, phrase in enumerate(phrases):
        is_last = (pi == len(phrases) - 1)
        line_start_ms = int(phrase[0].start * 1000)
        line_end_ms = int(phrase[-1].end * 1000) + 150
        if pi > 0:
            prev_end = int(phrases[pi - 1][-1].end * 1000) + 150
            if line_start_ms < prev_end + INTER_GAP_MS:
                line_start_ms = prev_end + INTER_GAP_MS

        # Build plain text (no karaoke tags)
        raw_text = " ".join(w.text for w in phrase)
        text = postprocess_subtitle_text(raw_text, is_last_phrase=is_last)

        subs.events.append(pysubs2.SSAEvent(
            start=line_start_ms, end=line_end_ms, style="BigSubs", text=text
        ))

    # --- Center keyword overlay events ---
    for oi, ov in enumerate(overlays):
        is_last = (oi == len(overlays) - 1)
        is_cta = is_last and any(kw in ov["text"].upper() for kw in ["ПОДПИШИСЬ", "SUBSCRIBE"])
        if is_cta:
            text = f"{{\\fad(300,0)}}↓↓↓ {ov['text']} ↓↓↓"
            subs.events.append(pysubs2.SSAEvent(
                start=int(ov["start"]*1000), end=int(ov["end"]*1000), style="CTA", text=text
            ))
        else:
            highlighted = _v8_highlight_text(ov["text"])
            subs.events.append(pysubs2.SSAEvent(
                start=int(ov["start"]*1000), end=int(ov["end"]*1000), style="Keyword",
                text=f"{{\\fad(200,200)}}{highlighted}"
            ))

    subs.events.sort(key=lambda e: e.start)
    subs.save(output_path)
    print(f"  V8A saved: {output_path}")


def generate_v8b(words, overlays, output_path, max_words=3):
    """V8B: Triple Layer — small transcription at top + keyword center + big subs bottom.

    Same as V8A but adds a THIRD layer: small running transcription text
    at the top of the screen (like the viral external shorts pattern).
    """
    import pysubs2

    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = "1080"
    subs.info["PlayResY"] = "1920"
    subs.info["ScaledBorderAndShadow"] = "yes"

    # Hold last subtitle of each layer until video end
    # (ffmpeg clips to actual video duration, so overshoot is safe)
    VIDEO_HOLD_END_MS = int(words[-1].end * 1000) + 10000

    # TOP: Small transcription — WHITE text on BLACK opaque background
    # Like viral shorts: always readable regardless of video content behind
    style_trans = pysubs2.SSAStyle()
    style_trans.fontname = "Arial"
    style_trans.fontsize = 32
    style_trans.bold = False
    style_trans.italic = False
    style_trans.primarycolor = _ass_color(255, 255, 255, 0)      # pure white text
    style_trans.outlinecolor = _ass_color(0, 0, 0, 0)            # black outline = box color
    style_trans.backcolor = _ass_color(0, 0, 0, 0)               # FULLY OPAQUE black box
    style_trans.outline = 8                                        # padding around text
    style_trans.shadow = 0
    style_trans.alignment = 7                                      # top-LEFT (typewriter: text grows right)
    style_trans.marginv = 230
    style_trans.marginl = 150
    style_trans.marginr = 150
    style_trans.borderstyle = 3                                    # opaque background box
    subs.styles["SmallTrans"] = style_trans

    # BOTTOM: Big static subtitles (same as V8A)
    style_sub = pysubs2.SSAStyle()
    style_sub.fontname = "Arial"
    style_sub.fontsize = 80
    style_sub.bold = True
    style_sub.italic = False
    style_sub.primarycolor = _ass_color(255, 255, 255, 0)
    style_sub.secondarycolor = _ass_color(255, 255, 255, 0)
    style_sub.outlinecolor = _ass_color(0, 0, 0, 0)
    style_sub.backcolor = _ass_color(0, 0, 0, 120)
    style_sub.outline = 3.5
    style_sub.shadow = 3
    style_sub.alignment = 2
    style_sub.marginv = 250
    style_sub.marginl = 50
    style_sub.marginr = 50
    subs.styles["BigSubs"] = style_sub

    # CENTER: Keyword overlay (same as V8A)
    style_kw = pysubs2.SSAStyle()
    style_kw.fontname = "Arial"
    style_kw.fontsize = 68
    style_kw.bold = True
    style_kw.italic = False
    style_kw.primarycolor = _ass_color(255, 255, 255, 0)
    style_kw.outlinecolor = _ass_color(60, 20, 120, 0)
    style_kw.backcolor = _ass_color(0, 0, 0, 150)
    style_kw.outline = 3
    style_kw.shadow = 4
    style_kw.alignment = 5
    style_kw.marginv = 80
    style_kw.marginl = 40
    style_kw.marginr = 40
    subs.styles["Keyword"] = style_kw

    # CTA
    style_cta = pysubs2.SSAStyle()
    style_cta.fontname = "Arial"
    style_cta.fontsize = 62
    style_cta.bold = True
    style_cta.primarycolor = _ass_color(50, 255, 50, 0)
    style_cta.outlinecolor = _ass_color(60, 20, 120, 0)
    style_cta.backcolor = _ass_color(0, 0, 0, 150)
    style_cta.outline = 3
    style_cta.shadow = 4
    style_cta.alignment = 5
    style_cta.marginv = 80
    style_cta.marginl = 40
    style_cta.marginr = 40
    subs.styles["CTA"] = style_cta

    # --- Layer 1 (TOP): Small running transcription ---
    # Typewriter effect: words appear one-by-one, never jump between lines.
    # Uses explicit \N line breaks so ASS soft-wrap never shifts words.
    CHARS_PER_LINE = 34     # chars per line (~780px at 32px Arial)
    CHARS_HARD_MAX = 42     # absolute max before forced break (allows number+unit overflow)
    MAX_LINES = 2           # max 2 lines, then reset block
    STEP_SIZE = 1           # word-by-word typewriter (attracts eye)
    AVG_CHAR_PX = 18        # estimated pixel width per char at 32px Arial regular

    def _is_sentence_end(word_text):
        """Check if word ends a sentence."""
        stripped = word_text.rstrip()
        return stripped.endswith(('.', '!', '?', '…'))

    def _is_number_group(word):
        """Check if word is a number, unit, preposition before number, or % sign.
        These should stick together on one line."""
        w = word.rstrip('.,!?;:')
        if re.match(r'^\d+[\-.,]?\d*%?$', w):          # 29, 6%, 522, 10
            return True
        if re.match(r'^\d+[\-]?[А-Яа-яA-Za-z]+', w):   # 522-ФЗ
            return True
        if w.lower() in ('на', 'до', 'от', 'за', 'по', 'в', 'с', 'к', 'не', 'менее', 'более', 'меньше', 'больше'):
            return True  # short prepositions that should stick to next word
        if w in ('млрд', 'млн', 'тыс', 'руб', 'рублей', 'миллиардов', 'миллионов', 'процентов', 'дней', 'лет', 'месяцев'):
            return True  # units after numbers
        return False

    def _layout_lines(display_words, chars_per_line, chars_hard_max):
        """Arrange words into lines with explicit breaks.
        Keeps number+unit groups together even if slightly over chars_per_line.
        Only breaks at chars_hard_max."""
        lines = []
        current_line_words = []
        current_len = 0
        for i, w in enumerate(display_words):
            space = 1 if current_line_words else 0
            new_len = current_len + space + len(w)
            if current_line_words and new_len > chars_per_line:
                # Over soft limit — but keep number groups together
                is_group = _is_number_group(w)
                prev_is_group = current_line_words and _is_number_group(current_line_words[-1])
                if (is_group or prev_is_group) and new_len <= chars_hard_max:
                    # Allow overflow to keep group on same line
                    current_line_words.append(w)
                    current_len = new_len
                else:
                    # Break to new line
                    lines.append(" ".join(current_line_words))
                    current_line_words = [w]
                    current_len = len(w)
            else:
                current_line_words.append(w)
                current_len = new_len
        if current_line_words:
            lines.append(" ".join(current_line_words))
        return lines

    # Step 1: Split words into sentences
    sentences = []
    current_sentence = []
    for w in words:
        current_sentence.append(w)
        if _is_sentence_end(w.text):
            sentences.append(current_sentence)
            current_sentence = []
    if current_sentence:
        sentences.append(current_sentence)

    # Step 2: Group sentences into blocks (max 2 lines, no mid-sentence splits)
    max_block_chars = CHARS_PER_LINE * MAX_LINES
    blocks = []
    current_block_words = []
    current_chars = 0
    for sent in sentences:
        sent_chars = sum(len(w.text) + 1 for w in sent)
        if current_block_words and current_chars + sent_chars >= max_block_chars:
            blocks.append(current_block_words)
            current_block_words = []
            current_chars = 0
        if sent_chars > max_block_chars and not current_block_words:
            chunk = []
            chunk_chars = 0
            for w in sent:
                w_chars = len(w.text) + 1
                if chunk and chunk_chars + w_chars > max_block_chars:
                    blocks.append(chunk)
                    chunk = []
                    chunk_chars = 0
                chunk.append(w)
                chunk_chars += w_chars
            if chunk:
                current_block_words = chunk
                current_chars = chunk_chars
        else:
            current_block_words.extend(sent)
            current_chars += sent_chars
    if current_block_words:
        blocks.append(current_block_words)

    # Step 3: Generate typewriter subtitle events per block
    for bi, block_words in enumerate(blocks):
        is_final_block = (block_words[-1] is words[-1])
        if bi + 1 < len(blocks):
            block_max_end = int(blocks[bi + 1][0].start * 1000)
        else:
            block_max_end = VIDEO_HOLD_END_MS  # last block holds until video end

        # Pre-calculate block position: center the FINAL (max-width) text
        final_raw = " ".join(w.text for w in block_words)
        final_processed = postprocess_subtitle_text(final_raw, is_last_phrase=is_final_block)
        final_display = final_processed.split()
        final_lines = _layout_lines(final_display, CHARS_PER_LINE, CHARS_HARD_MAX)
        max_line_chars = max(len(line) for line in final_lines)
        max_line_px = max_line_chars * AVG_CHAR_PX
        x_pos = max(30, int((1080 - max_line_px) / 2))
        y_pos = style_trans.marginv  # use configured marginv

        # Build steps: word-by-word, but group short prepositions/numbers with next word
        def _is_glue_word(w_text):
            """Words that should NOT appear alone — they stick to the NEXT word."""
            t = w_text.rstrip('.,!?;:').lower()
            # Short prepositions and particles
            if t in ('в', 'на', 'с', 'к', 'у', 'о', 'за', 'до', 'от', 'по', 'из', 'не', 'ни', 'же'):
                return True
            # Bare numbers (should appear with their unit: "29 миллиардов", "10 дней")
            if re.match(r'^\d+$', t):
                return True
            return False

        steps = []
        wi = 0
        while wi < len(block_words):
            # Peek ahead: if current word is a glue word, skip to next
            step_end = wi + 1
            while step_end < len(block_words) and _is_glue_word(block_words[step_end - 1].text):
                step_end += 1
            step_end = min(step_end, len(block_words))
            start_ms = int(block_words[wi].start * 1000)
            steps.append((step_end, start_ms))
            wi = step_end

        for i in range(len(steps)):
            word_count, start_ms = steps[i]
            if i + 1 < len(steps):
                end_ms = steps[i + 1][1]
            else:
                end_ms = block_max_end  # holds until next block or video end

            # Build display text with postprocessing
            visible = block_words[:word_count]
            raw_text = " ".join(w.text for w in visible)
            processed = postprocess_subtitle_text(raw_text, is_last_phrase=is_final_block)

            # Layout into lines with explicit \N breaks (words never jump)
            display_words = processed.split()
            lines = _layout_lines(display_words, CHARS_PER_LINE, CHARS_HARD_MAX)
            text_body = "\\N".join(lines)
            # Position block so its center = frame center (540px)
            text = f"{{\\pos({x_pos},{y_pos})}}" + text_body

            subs.events.append(pysubs2.SSAEvent(
                start=start_ms, end=end_ms, style="SmallTrans", text=text
            ))

    # --- Layer 2 (BOTTOM): Big static subtitles ---
    phrases = group_words_into_phrases(words, max_words=max_words, min_words=1)
    for pi, phrase in enumerate(phrases):
        is_last = (pi == len(phrases) - 1)
        line_start_ms = int(phrase[0].start * 1000)
        # Last phrase holds until video end (viewer remembers final message)
        line_end_ms = VIDEO_HOLD_END_MS if is_last else int(phrase[-1].end * 1000) + 150
        if pi > 0:
            prev_end = int(phrases[pi - 1][-1].end * 1000) + 150
            if line_start_ms < prev_end + INTER_GAP_MS:
                line_start_ms = prev_end + INTER_GAP_MS

        raw_text = " ".join(w.text for w in phrase)
        text = postprocess_subtitle_text(raw_text, is_last_phrase=is_last)

        subs.events.append(pysubs2.SSAEvent(
            start=line_start_ms, end=line_end_ms, style="BigSubs", text=text
        ))

    # --- Layer 3 (CENTER): Keyword overlays ---
    for oi, ov in enumerate(overlays):
        is_last = (oi == len(overlays) - 1)
        is_cta = is_last and any(kw in ov["text"].upper() for kw in ["ПОДПИШИСЬ", "SUBSCRIBE"])
        if is_cta:
            # CTA holds until video end — viewer must remember call to action
            text = f"{{\\fad(300,0)}}↓↓↓ {ov['text']} ↓↓↓"
            subs.events.append(pysubs2.SSAEvent(
                start=int(ov["start"]*1000), end=VIDEO_HOLD_END_MS, style="CTA", text=text
            ))
        else:
            highlighted = _v8_highlight_text(ov["text"])
            subs.events.append(pysubs2.SSAEvent(
                start=int(ov["start"]*1000), end=int(ov["end"]*1000), style="Keyword",
                text=f"{{\\fad(200,200)}}{highlighted}"
            ))

    subs.events.sort(key=lambda e: e.start)
    subs.save(output_path)
    print(f"  V8B saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Render style variants for comparison")
    parser.add_argument("--video", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--output", default=".")
    parser.add_argument("--code", default="A2")
    parser.add_argument("--words-json", help="Reuse existing words.json (skip transcription)")
    parser.add_argument("--variant", help="Render specific variant: v6a, v6b, v6c, v7, v8a, v8b")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1-2: Audio + Transcribe (or reuse)
    if args.words_json and Path(args.words_json).exists():
        import json
        print("[1-2/6] Reusing existing transcription...")
        with open(args.words_json) as f:
            data = json.load(f)
        words = [Word(w["text"], w["start"], w["end"]) for w in data]
        print(f"  Words: {len(words)}")
    else:
        print("[1/6] Extracting audio...")
        audio_path = extract_audio(args.video, f"{args.code}_audio")
        print("[2/6] Transcribing...")
        words = transcribe_words(audio_path)

    # Step 3-4: Parse scenario + align overlays
    print("[3/6] Parsing scenario...")
    scenario = parse_scenario(args.script)
    print("[4/6] Aligning overlays...")
    overlays = align_overlays(scenario, words)

    # Determine which variants to render
    all_variants = {
        "v6a": generate_v6a,
        "v6b": generate_v6b,
        "v6c": generate_v6c,
        "v7": generate_v7,
        "v8a": generate_v8a,
        "v8b": generate_v8b,
        "v8c": generate_v8b,  # V8C = V8B with sentence-aware SmallTrans
        "v8d": generate_v8b,  # V8D = narrower SmallTrans (150+150, marginv 170)
        "v8e": generate_v8b,  # V8E = medium SmallTrans (120+120, marginv 150)
        "v8f": generate_v8b,  # V8F = V8D + left-align typewriter (alignment=7)
        "v8g": generate_v8b,  # V8G = V8F + hold last subs/CTA until video end
    }
    if args.variant:
        selected = {args.variant: all_variants[args.variant]}
    else:
        selected = all_variants

    # Step 5: Generate ASS variants
    print(f"[5/6] Generating {len(selected)} ASS variant(s)...")
    ass_files = {}
    for label, gen_func in selected.items():
        ass_path = str(out / f"{args.code}_{label}.ass")
        gen_func(words, overlays, ass_path)
        ass_files[label] = ass_path

    # Step 6: Burn
    print(f"[6/6] Burning {len(selected)} variant(s)...")
    for label, ass_path in ass_files.items():
        mp4_path = str(out / f"{args.code}_{label}.mp4")
        burn_subtitles(args.video, ass_path, mp4_path)
        size_mb = os.path.getsize(mp4_path) / 1024 / 1024
        print(f"  {label}: {size_mb:.1f} MB")

    print(f"\n=== Done! {len(selected)} variant(s) ready ===")


if __name__ == "__main__":
    main()
