#!/usr/bin/env python3
"""Render the project Markdown report to LibreOffice-friendly HTML.

The generated HTML is an intermediate document. LibreOffice then converts it
to DOCX and embeds all referenced local figures into the Word package.
"""

from __future__ import annotations

import argparse
import html
import re
import struct
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
IMAGE_RE = re.compile(r"^!\[([^]]*)\]\(([^)]+)\)\s*$")
ORDERED_RE = re.compile(r"^\d+\.\s+(.+)$")
TABLE_SEPARATOR_RE = re.compile(r"^\|(?:\s*:?-+:?\s*\|)+\s*$")
FIGURE_CAPTION_RE = re.compile(r"^图\s*\d+\s+")


def inline_markup(value: str) -> str:
    """Escape text and render the small inline subset used by the report."""
    escaped = html.escape(value, quote=False)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def split_table_row(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def html_image_size(image_path: Path) -> tuple[int, int]:
    """Return a conservative HTML image size that LibreOffice imports well."""
    width = 1200
    height = 800
    if image_path.suffix.lower() == ".png":
        payload = image_path.read_bytes()[:24]
        if payload[:8] == b"\x89PNG\r\n\x1a\n" and len(payload) >= 24:
            width, height = struct.unpack(">II", payload[16:24])
    scale = min(600.0 / width, 610.0 / height, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def render_markdown(markdown_path: Path) -> str:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    title = "黄线赛道视觉避障与规划算法实验总结"
    date_text = "2026 年 8 月 25 日"

    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()

    body_start = 1
    for index in range(1, min(len(lines), 8)):
        stripped = lines[index].strip()
        if stripped.startswith("日期："):
            date_text = stripped.split("：", 1)[1].strip()
            body_start = index + 1
            break

    headings = []
    for line in lines[body_start:]:
        match = re.match(r"^##\s+(.+)$", line)
        if match:
            headings.append(match.group(1).strip())

    output = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "@page { size: A4; margin: 2.45cm 2.35cm 2.35cm 2.45cm; }",
        "body { font-family: 'Noto Serif CJK SC', 'SimSun', '宋体', serif; "
        "font-size: 11pt; line-height: 1.65; color: #111; }",
        ".cover { height: 23cm; display: table; width: 100%; "
        "page-break-after: always; text-align: center; }",
        ".cover-inner { display: table-cell; vertical-align: middle; }",
        ".cover h1 { font-family: 'Noto Sans CJK SC', 'SimHei', '黑体', sans-serif; "
        "font-size: 26pt; line-height: 1.45; margin: 0 0 2.2cm 0; }",
        ".cover .subtitle { font-size: 15pt; margin-bottom: 3.4cm; }",
        ".cover .meta { font-size: 12pt; margin: 0.35cm 0; text-indent: 0; }",
        ".toc { page-break-after: always; }",
        ".toc h2 { text-align: center; border: 0; font-size: 20pt; margin-bottom: 1cm; }",
        ".toc p { text-indent: 0; margin: 0.28cm 1.2cm; font-size: 12pt; }",
        "h1, h2, h3 { font-family: 'Noto Sans CJK SC', 'SimHei', '黑体', sans-serif; "
        "color: #111; page-break-after: avoid; }",
        "h2 { font-size: 17pt; margin-top: 0.85cm; margin-bottom: 0.4cm; "
        "border-bottom: 1px solid #777; padding-bottom: 0.12cm; }",
        "h3 { font-size: 13.5pt; margin-top: 0.62cm; margin-bottom: 0.25cm; }",
        "p { margin: 0.18cm 0; text-align: justify; text-indent: 2em; }",
        "ul, ol { margin: 0.18cm 0 0.28cm 0.9cm; padding-left: 0.55cm; }",
        "li { margin: 0.08cm 0; text-align: justify; }",
        "code { font-family: 'DejaVu Sans Mono', 'Consolas', monospace; "
        "font-size: 9.5pt; background: #f2f2f2; padding: 0 0.08cm; }",
        "pre { font-family: 'DejaVu Sans Mono', 'Consolas', monospace; "
        "font-size: 9.3pt; line-height: 1.4; background: #f5f5f5; "
        "border: 1px solid #d0d0d0; padding: 0.35cm; white-space: pre-wrap; "
        "page-break-inside: avoid; }",
        "table { width: 100%; border-collapse: collapse; margin: 0.35cm 0 0.5cm 0; "
        "font-size: 9.5pt; page-break-inside: avoid; }",
        "th, td { border: 1px solid #777; padding: 0.16cm 0.18cm; "
        "vertical-align: middle; text-align: left; }",
        "th { background: #e9eef4; font-family: 'Noto Sans CJK SC', 'SimHei', '黑体', sans-serif; }",
        ".figure { text-align: center; margin: 0.45cm 0 0.15cm 0; "
        "page-break-inside: avoid; }",
        ".figure img { width: 15.3cm; height: auto; }",
        ".caption { text-align: center; text-indent: 0; font-size: 9.5pt; "
        "margin: 0.08cm 0 0.45cm 0; color: #333; }",
        ".note { text-indent: 0; font-size: 9.5pt; color: #444; }",
        ".manual-break { page-break-before: always; font-size: 1pt; "
        "line-height: 1pt; margin: 0; text-indent: 0; }",
        "</style>",
        "</head>",
        "<body>",
        '<div class="cover"><div class="cover-inner">',
        f"<h1>{html.escape(title)}</h1>",
        '<p class="subtitle">侦察小车黄线走廊双圆片避障</p>',
        '<p class="meta">项目模块：视觉感知、路径规划与实车迁移</p>',
        f'<p class="meta">日期：{html.escape(date_text)}</p>',
        "</div></div>",
        '<p class="manual-break">&nbsp;</p>',
        '<div class="toc">',
        "<h2>目录</h2>",
    ]

    for heading in headings:
        output.append(f"<p>{inline_markup(heading)}</p>")
    output.extend(["</div>", '<p class="manual-break">&nbsp;</p>'])

    index = body_start
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        heading = HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            if level == 1:
                index += 1
                continue
            output.append(f"<h{level}>{inline_markup(heading.group(2))}</h{level}>")
            index += 1
            continue

        image_match = IMAGE_RE.match(stripped)
        if image_match:
            image_path = (markdown_path.parent / image_match.group(2)).resolve()
            image_width, image_height = html_image_size(image_path)
            output.extend([
                '<div class="figure">',
                f'<img src="{image_path.as_uri()}" '
                f'width="{image_width}" height="{image_height}" '
                f'alt="{html.escape(image_match.group(1))}">',
                "</div>",
            ])
            index += 1
            continue

        if stripped.startswith("```"):
            code_lines = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            output.append(f"<pre>{html.escape(chr(10).join(code_lines))}</pre>")
            continue

        if stripped.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            rows = [split_table_row(item) for item in table_lines]
            if len(rows) >= 2 and TABLE_SEPARATOR_RE.match(table_lines[1]):
                header = rows[0]
                data_rows = rows[2:]
            else:
                header = []
                data_rows = rows
            output.append("<table>")
            if header:
                output.append("<thead><tr>")
                for cell in header:
                    output.append(f"<th>{inline_markup(cell)}</th>")
                output.append("</tr></thead>")
            output.append("<tbody>")
            for row in data_rows:
                output.append("<tr>")
                for cell in row:
                    output.append(f"<td>{inline_markup(cell)}</td>")
                output.append("</tr>")
            output.append("</tbody></table>")
            continue

        if stripped.startswith("- "):
            output.append("<ul>")
            while index < len(lines) and lines[index].strip().startswith("- "):
                output.append(f"<li>{inline_markup(lines[index].strip()[2:])}</li>")
                index += 1
            output.append("</ul>")
            continue

        if ORDERED_RE.match(stripped):
            output.append("<ol>")
            while index < len(lines):
                match = ORDERED_RE.match(lines[index].strip())
                if not match:
                    break
                output.append(f"<li>{inline_markup(match.group(1))}</li>")
                index += 1
            output.append("</ol>")
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate:
                break
            if (
                HEADING_RE.match(lines[index])
                or IMAGE_RE.match(candidate)
                or candidate.startswith("```")
                or candidate.startswith("|")
                or candidate.startswith("- ")
                or ORDERED_RE.match(candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        paragraph = " ".join(paragraph_lines)
        css_class = "caption" if FIGURE_CAPTION_RE.match(paragraph) else ""
        class_attr = f' class="{css_class}"' if css_class else ""
        output.append(f"<p{class_attr}>{inline_markup(paragraph)}</p>")

    output.extend(["</body>", "</html>"])
    return "\n".join(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("markdown", type=Path)
    parser.add_argument("html", type=Path)
    args = parser.parse_args()
    args.html.write_text(render_markdown(args.markdown.resolve()), encoding="utf-8")


if __name__ == "__main__":
    main()
