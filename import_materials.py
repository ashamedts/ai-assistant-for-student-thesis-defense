# -*- coding: utf-8 -*-
"""导入讲稿材料：解析 materials/ 目录下的 pptx/docx/pdf/txt，切片为 JSON 知识库。"""
import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MATERIALS_DIR = os.path.join(BASE_DIR, "materials")
OUT_FILE = os.path.join(BASE_DIR, "materials_data.json")
CHUNK = 600


def _clean(s):
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_pptx(path):
    from pptx import Presentation
    prs = Presentation(path)
    parts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                parts.append(shape.text)
    return parts


def _parse_docx(path):
    from docx import Document
    doc = Document(path)
    return [p.text for p in doc.paragraphs if p.text.strip()]


def _parse_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        t = page.extract_text()
        if t and t.strip():
            parts.append(t)
    return parts


def _parse_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [f.read()]


PARSERS = {".pptx": _parse_pptx, ".docx": _parse_docx, ".pdf": _parse_pdf, ".txt": _parse_txt}


def main():
    if not os.path.isdir(MATERIALS_DIR):
        os.makedirs(MATERIALS_DIR)
        print(f"已创建 materials 目录：{MATERIALS_DIR}\n请放入讲稿文件后重新运行。")
        return
    chunks = []
    for name in sorted(os.listdir(MATERIALS_DIR)):
        path = os.path.join(MATERIALS_DIR, name)
        ext = os.path.splitext(name)[1].lower()
        if ext not in PARSERS:
            continue
        try:
            parts = PARSERS[ext](path)
        except Exception as e:
            print(f"跳过 {name}: {e}")
            continue
        text = "\n".join(_clean(p) for p in parts if _clean(p))
        for i in range(0, len(text), CHUNK):
            chunks.append({"src": name, "text": text[i:i + CHUNK]})
        print(f"{name}: {len(parts)} 段 -> {len(text)//CHUNK + 1} 块")
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"完成：共 {len(chunks)} 块，已写入 {OUT_FILE}")


if __name__ == "__main__":
    main()
