#!/usr/bin/env python3
"""Replace external local-image links in a DOCX with embedded media files."""

from __future__ import annotations

import argparse
import html
import os
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse


RELATIONSHIP_RE = re.compile(r"<Relationship\b[^>]*/>")


def local_path_from_target(target: str) -> Path:
    parsed = urlparse(target)
    if parsed.scheme != "file":
        raise ValueError(f"Only file URLs can be embedded: {target}")
    return Path(unquote(parsed.path))


def embed_images(docx_path: Path) -> int:
    temp_path = docx_path.with_suffix(".embedded.tmp.docx")
    with zipfile.ZipFile(docx_path, "r") as source:
        source_infos = source.infolist()
        entries = {name: source.read(name) for name in source.namelist()}

    rels_name = "word/_rels/document.xml.rels"
    document_name = "word/document.xml"
    rels_xml = entries[rels_name].decode("utf-8")
    document_xml = entries[document_name].decode("utf-8")
    embedded = []

    def replace_relationship(match: re.Match[str]) -> str:
        nonlocal document_xml
        tag = match.group(0)
        if "/image\"" not in tag or 'TargetMode="External"' not in tag:
            return tag
        id_match = re.search(r'\bId="([^"]+)"', tag)
        target_match = re.search(r'\bTarget="([^"]+)"', tag)
        if id_match is None or target_match is None:
            return tag
        rel_id = id_match.group(1)
        source_path = local_path_from_target(html.unescape(target_match.group(1)))
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        image_number = len(embedded) + 1
        suffix = source_path.suffix.lower() or ".png"
        archive_name = f"word/media/image{image_number}{suffix}"
        document_xml = document_xml.replace(
            f'r:link="{rel_id}"', f'r:embed="{rel_id}"'
        )
        embedded.append((archive_name, source_path.read_bytes(), suffix[1:]))
        tag = re.sub(
            r'\bTarget="[^"]+"',
            f'Target="media/image{image_number}{suffix}"',
            tag,
            count=1,
        )
        tag = re.sub(r'\s+TargetMode="External"', "", tag, count=1)
        return tag

    rels_xml = RELATIONSHIP_RE.sub(replace_relationship, rels_xml)
    entries[rels_name] = rels_xml.encode("utf-8")
    entries[document_name] = document_xml.encode("utf-8")

    for archive_name, payload, _extension in embedded:
        entries[archive_name] = payload

    with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as target:
        source_names = set()
        for info in source_infos:
            source_names.add(info.filename)
            target.writestr(info, entries[info.filename])
        for name, payload in entries.items():
            if name not in source_names:
                target.writestr(name, payload)

    os.replace(temp_path, docx_path)
    return len(embedded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    args = parser.parse_args()
    count = embed_images(args.docx.resolve())
    print(f"embedded_images={count}")


if __name__ == "__main__":
    main()
