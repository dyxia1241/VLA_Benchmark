#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


CARD_WIDTH = 1760
CARD_MARGIN = 40
CARD_GAP = 24
CARD_BG = (244, 241, 235)
CARD_PANEL_BG = (252, 251, 247)
CARD_INK = (28, 28, 28)
CARD_MUTED = (92, 92, 92)
CARD_ACCENT = (26, 92, 158)
CARD_ACCENT_BG = (230, 239, 249)
FRAME_GUTTER = 24
FRAME_LABEL_BAR_H = 52
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]
FONT_CANDIDATES_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render compact AIST pilot audit cards from sampled JSONL and extracted frames.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_items(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = FONT_CANDIDATES_BOLD if bold else FONT_CANDIDATES
    for cand in candidates:
        path = Path(cand)
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except Exception:
            continue
    family = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(family, size=size)
    except Exception:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def _line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    return _text_size(draw, "Ag", font)[1] + 8


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    raw = " ".join(str(text or "").split())
    if not raw:
        return []
    words = raw.split()
    if not words:
        return [raw]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if _text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def frame_indices_for_item(item: dict[str, Any]) -> list[int]:
    if isinstance(item.get("frame_indices"), list):
        return [int(x) for x in item["frame_indices"]]
    if "frame_index" in item:
        return [int(item["frame_index"])]
    return []


def resolve_frame_paths(item: dict[str, Any], frame_dir: Path) -> list[Path]:
    recording_id = str(item.get("recording_id") or item.get("task_id") or "").strip()
    camera = str(item.get("camera", "cam_high")).strip()
    out: list[Path] = []
    for frame_index in frame_indices_for_item(item):
        out.append(frame_dir / f"{recording_id}_{int(frame_index)}_{camera}.jpg")
    return out


def relative_frame_labels(item: dict[str, Any], num_frames: int) -> list[str]:
    if not isinstance(item.get("frame_indices"), list) or "frame_index" not in item:
        return []
    center = int(item["frame_index"])
    labels: list[str] = []
    for frame_index in item["frame_indices"][:num_frames]:
        offset = int(frame_index) - center
        if offset == 0:
            labels.append("t0")
        elif offset > 0:
            labels.append(f"t+{offset}")
        else:
            labels.append(f"t{offset}")
    return labels


def frame_labels(item: dict[str, Any], num_frames: int) -> list[str]:
    task_type = str(item.get("task_type", ""))
    if task_type == "T_temporal":
        labels = [str(x) for x in item.get("shuffled_labels", [])]
        return labels[:num_frames]
    if task_type == "T_binary":
        labels = [str(x) for x in item.get("display_labels", [])]
        return labels[:num_frames]
    if task_type.startswith("T3") or task_type == "T4":
        labels = relative_frame_labels(item, num_frames)
        if labels:
            return labels
    if task_type in {"T_progress", "T6"} and num_frames == 5:
        return ["t-6", "t-3", "t0", "t+3", "t+6"]
    if num_frames == 4:
        return ["t-6", "t-3", "t0", "t+3"]
    if num_frames == 3:
        return ["frame 1", "frame 2", "frame 3"]
    if num_frames == 2:
        return ["frame 1", "frame 2"]
    return ["frame"]


def choice_lines(item: dict[str, Any]) -> list[str]:
    choices = item.get("choices")
    if not isinstance(choices, dict):
        return []
    return [f"{k}: {v}" for k, v in sorted(choices.items())]


def gt_answer_text(item: dict[str, Any]) -> str:
    if item.get("needs_human_direction_label"):
        return "Annotate human_label in CSV: left / right / top / bottom / unclear"
    answer = str(item.get("answer", ""))
    choices = item.get("choices")
    if isinstance(choices, dict) and answer in choices:
        return f"{answer}: {choices[answer]}"
    return answer


def gt_section_title(item: dict[str, Any]) -> str:
    if item.get("needs_human_direction_label"):
        return "Human Label Needed"
    return "Benchmark GT"


def side_notes(item: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    task_desc = str(item.get("task_meta_description", "")).strip()
    taxonomy = str(item.get("taxonomy", "")).strip()
    query_arm = str(item.get("query_arm", "")).strip()
    if task_desc:
        lines.append(f"Task: {task_desc}")
    if taxonomy:
        lines.append(f"Taxonomy: {taxonomy}")
    if query_arm:
        lines.append(f"Query arm: {query_arm}")
    if "phase_label" in item:
        lines.append(f"Phase label: {item['phase_label']}")
    if "label" in item:
        lines.append(f"Contact label: {item['label']}")
    if "progress_bin" in item:
        lines.append(f"Progress bin: {item['progress_bin']}")
    if "speed_level_label" in item:
        lines.append(f"Speed label: {item['speed_level_label']}")
    if "low_level_text_raw" in item:
        lines.append(f"Current low-level action: {item['low_level_text_raw']}")
    if "current_low_level_text_raw" in item:
        lines.append(f"Current low-level action: {item['current_low_level_text_raw']}")
    if "next_low_level_text_raw" in item:
        lines.append(f"Next low-level action: {item['next_low_level_text_raw']}")
    if "auto_qpos_direction_text" in item:
        lines.append(f"Auto qpos direction candidate: {item['auto_qpos_direction_text']}")
    if "delta_x" in item and "delta_y" in item:
        lines.append(f"qpos delta x/y: {item['delta_x']} / {item['delta_y']}")
    if "local_chain_context" in item:
        chain = " > ".join(str(x) for x in item["local_chain_context"])
        lines.append(f"Local chain: {chain}")
    return lines


def open_or_placeholder(path: Path, size: tuple[int, int]) -> Image.Image:
    if path.exists():
        with Image.open(path) as raw:
            return raw.convert("RGB")
    img = Image.new("RGB", size, (238, 236, 231))
    draw = ImageDraw.Draw(img)
    font = _load_font(28, bold=True)
    text = f"Missing frame\n{path.name}"
    lines = text.splitlines()
    total_h = 0
    dims: list[tuple[str, int, int]] = []
    for line in lines:
        tw, th = _text_size(draw, line, font)
        dims.append((line, tw, th))
        total_h += th + 8
    y = (size[1] - total_h) // 2
    for line, tw, th in dims:
        draw.text(((size[0] - tw) // 2, y), line, font=font, fill=CARD_MUTED)
        y += th + 8
    return img


def draw_frame_panel(img: Image.Image, label: str, panel_w: int, panel_h: int) -> Image.Image:
    outer = Image.new("RGB", (panel_w, panel_h), CARD_BG)
    draw = ImageDraw.Draw(outer)
    draw.rounded_rectangle([0, 0, panel_w - 1, panel_h - 1], radius=24, fill=CARD_PANEL_BG, outline=CARD_INK, width=3)

    if label:
        bar = [18, 18, panel_w - 18, 18 + FRAME_LABEL_BAR_H]
        draw.rounded_rectangle(bar, radius=14, fill=CARD_INK)
        font = _load_font(24, bold=True)
        tw, th = _text_size(draw, label, font)
        tx = bar[0] + (bar[2] - bar[0] - tw) // 2
        ty = bar[1] + (bar[3] - bar[1] - th) // 2 - 1
        draw.text((tx, ty), label, font=font, fill=(246, 246, 244))
        image_top = bar[3] + 16
    else:
        image_top = 18

    box = [18, image_top, panel_w - 18, panel_h - 18]
    draw.rounded_rectangle(box, radius=16, fill=(255, 255, 255), outline=CARD_INK, width=2)
    inner_w = box[2] - box[0] - 20
    inner_h = box[3] - box[1] - 20
    fitted = ImageOps.contain(img, (inner_w, inner_h), method=Image.Resampling.LANCZOS)
    px = box[0] + 10 + (inner_w - fitted.width) // 2
    py = box[1] + 10 + (inner_h - fitted.height) // 2
    outer.paste(fitted, (px, py))
    return outer


def build_visual_strip(item: dict[str, Any], frame_paths: list[Path]) -> Image.Image:
    num_frames = max(1, len(frame_paths))
    available_w = CARD_WIDTH - CARD_MARGIN * 2
    if num_frames <= 1:
        panel_w = min(available_w, 1180)
        panel_h = 860
    elif num_frames == 2:
        panel_w = (available_w - FRAME_GUTTER) // 2
        panel_h = 620
    elif num_frames == 3:
        panel_w = (available_w - 2 * FRAME_GUTTER) // 3
        panel_h = 500
    elif num_frames == 4:
        panel_w = (available_w - 3 * FRAME_GUTTER) // 4
        panel_h = 430
    else:
        panel_w = (available_w - 4 * FRAME_GUTTER) // 5
        panel_h = 390

    labels = frame_labels(item, num_frames)
    placeholder_size = (panel_w - 64, panel_h - 120)
    panels: list[Image.Image] = []
    for idx in range(num_frames):
        frame_path = frame_paths[idx]
        label = labels[idx] if idx < len(labels) else f"frame {idx + 1}"
        raw = open_or_placeholder(frame_path, placeholder_size)
        panels.append(draw_frame_panel(raw, label, panel_w, panel_h))

    strip_w = len(panels) * panel_w + max(0, len(panels) - 1) * FRAME_GUTTER
    strip_h = panel_h
    strip = Image.new("RGB", (strip_w, strip_h), CARD_BG)
    x = 0
    for panel in panels:
        strip.paste(panel, (x, 0))
        x += panel_w + FRAME_GUTTER
    return strip


def write_card(output_dir: Path, item: dict[str, Any], frame_dir: Path, item_index: int) -> Path:
    cards_dir = output_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = resolve_frame_paths(item, frame_dir)
    visual = build_visual_strip(item, frame_paths)

    title_font = _load_font(40, bold=True)
    meta_font = _load_font(24, bold=False)
    heading_font = _load_font(28, bold=True)
    body_font = _load_font(24, bold=False)
    gt_font = _load_font(26, bold=True)

    measure = Image.new("RGB", (CARD_WIDTH, 300), CARD_BG)
    draw = ImageDraw.Draw(measure)
    text_w = CARD_WIDTH - CARD_MARGIN * 2
    note_lines: list[str] = []
    for line in side_notes(item):
        note_lines.extend(_wrap_text(draw, line, body_font, text_w - 36))
    question_lines = _wrap_text(draw, str(item.get("question", "")), body_font, text_w)
    choice_wrapped: list[str] = []
    for line in choice_lines(item):
        choice_wrapped.extend(_wrap_text(draw, line, body_font, text_w - 24))
    gt_lines = _wrap_text(draw, gt_answer_text(item), gt_font, text_w - 32)

    title_h = _text_size(draw, "Ag", title_font)[1]
    meta_h = _text_size(draw, "Ag", meta_font)[1]
    heading_h = _text_size(draw, "Ag", heading_font)[1]
    body_h = _line_height(draw, body_font)
    gt_h = _line_height(draw, gt_font)

    note_box_h = 0
    if note_lines:
        note_box_h = 30 + heading_h + max(1, len(note_lines)) * body_h

    total_h = (
        CARD_MARGIN
        + title_h
        + 8
        + meta_h
        + CARD_GAP
        + note_box_h
        + (CARD_GAP if note_box_h else 0)
        + visual.height
        + CARD_GAP
        + heading_h
        + 12
        + max(1, len(question_lines)) * body_h
        + 16
        + heading_h
        + 12
        + max(1, len(choice_wrapped) if choice_wrapped else 1) * body_h
        + 20
        + 86
        + max(1, len(gt_lines)) * gt_h
        + CARD_MARGIN
    )

    card = Image.new("RGB", (CARD_WIDTH, total_h), CARD_BG)
    draw = ImageDraw.Draw(card)
    y = CARD_MARGIN

    title = f"AIST Pilot Card {item_index:03d}"
    recording_id = str(item.get("recording_id") or item.get("task_id") or "")
    meta = f"{item.get('task_type', '')} | recording={recording_id} | camera={item.get('camera', '')}"
    draw.text((CARD_MARGIN, y), title, font=title_font, fill=CARD_INK)
    y += title_h + 8
    draw.text((CARD_MARGIN, y), meta, font=meta_font, fill=CARD_MUTED)
    y += meta_h + CARD_GAP

    if note_box_h:
        note_box = [CARD_MARGIN, y, CARD_WIDTH - CARD_MARGIN, y + note_box_h]
        draw.rounded_rectangle(note_box, radius=20, fill=CARD_ACCENT_BG, outline=CARD_ACCENT, width=2)
        draw.text((CARD_MARGIN + 18, y + 14), "Context", font=heading_font, fill=CARD_INK)
        note_y = y + 14 + heading_h + 6
        for line in note_lines:
            draw.text((CARD_MARGIN + 18, note_y), line, font=body_font, fill=CARD_INK)
            note_y += body_h
        y += note_box_h + CARD_GAP

    vx = CARD_MARGIN + (CARD_WIDTH - CARD_MARGIN * 2 - visual.width) // 2
    draw.rounded_rectangle(
        [vx - 10, y - 10, vx + visual.width + 10, y + visual.height + 10],
        radius=24,
        fill=CARD_BG,
        outline=CARD_MUTED,
        width=2,
    )
    card.paste(visual, (vx, y))
    y += visual.height + CARD_GAP

    draw.text((CARD_MARGIN, y), "Question", font=heading_font, fill=CARD_INK)
    y += heading_h + 12
    for line in question_lines:
        draw.text((CARD_MARGIN, y), line, font=body_font, fill=CARD_INK)
        y += body_h

    y += 16
    draw.text((CARD_MARGIN, y), "Choices", font=heading_font, fill=CARD_INK)
    y += heading_h + 12
    if not choice_wrapped:
        choice_wrapped = ["No explicit choices recorded."]
    for line in choice_wrapped:
        draw.text((CARD_MARGIN, y), line, font=body_font, fill=CARD_INK)
        y += body_h

    y += 20
    gt_box = [CARD_MARGIN, y, CARD_WIDTH - CARD_MARGIN, y + 86 + max(1, len(gt_lines)) * gt_h]
    draw.rounded_rectangle(gt_box, radius=20, fill=(232, 238, 228), outline=(113, 132, 106), width=2)
    draw.text((CARD_MARGIN + 18, y + 14), gt_section_title(item), font=heading_font, fill=CARD_INK)
    gt_y = y + 14 + heading_h + 6
    for line in gt_lines:
        draw.text((CARD_MARGIN + 18, gt_y), line, font=gt_font, fill=CARD_INK)
        gt_y += gt_h

    stem = f"{item_index:03d}_{item.get('task_type', 'UNK')}_{recording_id}".replace("/", "_")
    out_path = cards_dir / f"{stem}.jpg"
    card.save(out_path, format="JPEG", quality=95)
    return out_path


def main() -> None:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl)
    frame_dir = Path(args.frame_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cards_dir = output_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    # Keep the audit pack deterministic across rebuilds by clearing stale cards.
    for old_card in cards_dir.glob("*.jpg"):
        old_card.unlink()

    items = load_items(input_jsonl)
    manifest_rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        card_path = write_card(output_dir, item, frame_dir, idx)
        manifest_rows.append(
            {
                "item_index": idx,
                "task_type": item.get("task_type", ""),
                "recording_id": item.get("recording_id") or item.get("task_id") or "",
                "card_path": str(card_path),
            }
        )

    summary = {
        "input_jsonl": str(input_jsonl),
        "frame_dir": str(frame_dir),
        "output_dir": str(output_dir),
        "num_items": len(items),
        "num_cards": len(manifest_rows),
    }
    (output_dir / "render_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "cards_manifest.jsonl").open("w", encoding="utf-8") as fh:
        for row in manifest_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
