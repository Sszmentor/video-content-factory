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
    style_trans.alignment = 8                                      # top-center
    style_trans.marginv = 100
    style_trans.marginl = 40
    style_trans.marginr = 40
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
    # Accumulating text: words add step-by-step until ~2 lines filled,
    # then reset. Each step REPLACES the previous (non-overlapping timing).
    WORDS_PER_BLOCK = 14  # ~2 lines at 32px on 1080px width
    STEP_SIZE = 3          # add 3 words per step
    block_start_idx = 0
    while block_start_idx < len(words):
        block_end_idx = min(block_start_idx + WORDS_PER_BLOCK, len(words))
        block_words = words[block_start_idx:block_end_idx]

        # Build list of steps: [(visible_words_count, start_ms, end_ms)]
        steps = []
        for step_end in range(STEP_SIZE, len(block_words) + STEP_SIZE, STEP_SIZE):
            step_end = min(step_end, len(block_words))
            # Start: when the newest batch of words begins
            new_start = max(0, step_end - STEP_SIZE)
            start_ms = int(block_words[new_start].start * 1000)
            steps.append((step_end, start_ms))

        # Set end times: each step ends when the next one starts
        for i in range(len(steps)):
            word_count, start_ms = steps[i]
            if i + 1 < len(steps):
                end_ms = steps[i + 1][1]  # next step's start
            else:
                end_ms = int(block_words[-1].end * 1000) + 400  # hold last step

            visible = block_words[:word_count]
            raw_text = " ".join(w.text for w in visible)
            text = postprocess_subtitle_text(raw_text, is_last_phrase=(block_end_idx >= len(words)))

            subs.events.append(pysubs2.SSAEvent(
                start=start_ms, end=end_ms, style="SmallTrans", text=text
            ))

        block_start_idx = block_end_idx

    # --- Layer 2 (BOTTOM): Big static subtitles ---
    phrases = group_words_into_phrases(words, max_words=max_words, min_words=1)
    for pi, phrase in enumerate(phrases):
        is_last = (pi == len(phrases) - 1)
        line_start_ms = int(phrase[0].start * 1000)
        line_end_ms = int(phrase[-1].end * 1000) + 150
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
