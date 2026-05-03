# Official Paper Excellent-Style Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the 2026 contest's official submission structure while rewriting the paper's body style and section logic to match the 2023 excellent-paper examples.

**Architecture:** `paper/main.tex` remains the single authoritative paper source for PDF builds. The official full version keeps the eight required parts, and the anonymous PDF removes only the cover and acknowledgement. The writing style changes are limited to title numbering, section organization, abstract/introduction phrasing, and result-discussion logic; computed tables, figures, and quantitative conclusions remain unchanged.

**Tech Stack:** XeLaTeX/ctex, Python `python-docx` exporter, PyMuPDF verification, existing generated figures and LaTeX tables.

---

## File Structure

- Modify: `paper/main.tex`
  - Add official Chinese section numbering: `一、` and `（一）`.
  - Keep cover, abstract, contents, list of tables/figures, body, references, appendix, and acknowledgement.
  - Rewrite body headings and paragraphs to follow excellent-paper logic: problem framing, model design, empirical analysis, robustness, mechanism explanation, conclusions and policy suggestions.
- Modify: `scripts/35_build_word_paper.py`
  - Ensure the Word exporter still reads section/subsection headings and table macros after the LaTeX style change.
  - Keep Word tables editable.
- Modify: `docs/final_submission_checklist.md`
  - Record that the official structure is retained while body numbering is converted to Chinese contest style.
- Create: `docs/paper_style_reference_notes.md`
  - Summarize the observed excellent-paper format and the official-format constraints used for this rewrite.

## Task 1: Capture Style Reference

**Files:**
- Create: `docs/paper_style_reference_notes.md`

- [ ] **Step 1: Record observed excellent-paper traits**

Write notes covering:

```text
2023 excellent papers usually start with title, author line, abstract, keywords, and direct entry into "一、引言"; they use Chinese section hierarchy and dense result-oriented paragraphs. The 2026 official attachment, however, requires Word full version with eight parts and anonymous PDF with six parts. Therefore, this project keeps the official parts but rewrites the body to excellent-paper style.
```

- [ ] **Step 2: Commit style notes with paper rewrite**

Run after all rewrite work:

```bash
git add docs/paper_style_reference_notes.md
git commit -m "docs: document excellent paper style reference"
```

Expected: Style reference notes are tracked with the rewrite commit.

## Task 2: Convert LaTeX Body Numbering

**Files:**
- Modify: `paper/main.tex`

- [ ] **Step 1: Add ctex heading configuration**

Add `\ctexset` after document spacing/table macros so the body uses:

```tex
section={name={,、},number=\chinese{section},format=\heiti\zihao{-3},aftername=\hspace{0.5em}},
subsection={name={（,）},number=\chinese{subsection},format=\kaishu\zihao{4},aftername=\hspace{0.5em}},
subsubsection={format=\songti\zihao{-4}\bfseries}
```

- [ ] **Step 2: Build full PDF**

Run:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=paper/build paper/main.tex
```

Expected: `paper/build/main.pdf` builds and the body headings display as Chinese numbering.

## Task 3: Rewrite Body Structure

**Files:**
- Modify: `paper/main.tex`

- [ ] **Step 1: Replace current section headings**

Use this official-compliant excellent-paper structure:

```text
一、引言
二、研究设计与数据基础
三、指标体系与统计模型构建
四、南海海表增暖的统计测度
五、海洋热浪风险演化与空间分异
六、气候驱动因子解释与稳健性讨论
七、结论与建议
```

- [ ] **Step 2: Rewrite section openings**

Each section opening should follow:

```text
本部分回答什么问题 -> 使用什么证据 -> 该部分在全文论证链条中的作用。
```

- [ ] **Step 3: Preserve all numerical claims**

Keep the existing computed values:

```text
SST trend 0.191 °C/10a
SSTA trend 0.174 °C/10a
MHW frequency trend 1.308 events/10a
MHW total days trend 13.381 days/10a
MHW cumulative intensity trend 15.176 °C·d/10a
North Shelf HRI 0.574 and high-risk share 16.6%
Wind speed coefficients -0.363 and -0.558
```

- [ ] **Step 4: Keep all tables and figures**

The final source must still contain:

```text
10 table environments
13 figure environments
16 includegraphics commands
```

## Task 4: Update Word Export If Needed

**Files:**
- Modify: `scripts/35_build_word_paper.py`

- [ ] **Step 1: Check exported heading text**

Run:

```bash
/home/lz/miniconda3/envs/pytorch/bin/python scripts/35_build_word_paper.py --tex paper/main.tex --output paper/build/作品全文-组别-作品编号.docx
```

Expected: Word still has cover, abstract, contents, figure/table list, body, references, appendix, acknowledgement, 10 tables, and 16 images.

- [ ] **Step 2: Patch exporter only if parsing fails**

If the Word exporter misses headings or tables, update regex handling in `scripts/35_build_word_paper.py`. Do not change data outputs.

## Task 5: Final Verification

**Files:**
- Read: `paper/build/main.pdf`
- Read: `paper/build/main_anonymous.pdf`
- Read: `paper/build/作品全文-组别-作品编号.docx`

- [ ] **Step 1: Build full PDF**

Run:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=paper/build paper/main.tex
```

Expected: exit code 0.

- [ ] **Step 2: Build anonymous PDF**

Run twice:

```bash
xelatex -interaction=nonstopmode -halt-on-error -output-directory=paper/build -jobname=main_anonymous "\def\ANONYMOUS{1}\input{paper/main.tex}"
```

Expected: exit code 0 and `paper/build/main_anonymous.pdf` exists.

- [ ] **Step 3: Build Word**

Run:

```bash
/home/lz/miniconda3/envs/pytorch/bin/python scripts/35_build_word_paper.py --tex paper/main.tex --output paper/build/作品全文-组别-作品编号.docx
```

Expected: exit code 0 and `paper/build/作品全文-组别-作品编号.docx` exists.

- [ ] **Step 4: Check anonymous leakage**

Run:

```bash
/home/lz/miniconda3/envs/pytorch/bin/python -c "import fitz; p='paper/build/main_anonymous.pdf'; text=''.join(page.get_text() for page in fitz.open(p)); bad=['华南农业大学','参赛学校','参赛队员','指导老师','致谢','作品编号','TJJM2026']; print({b:(b in text) for b in bad})"
```

Expected: all values are `False`.

- [ ] **Step 5: Check Word structure**

Run:

```bash
/home/lz/miniconda3/envs/pytorch/bin/python -c "from docx import Document; p='paper/build/作品全文-组别-作品编号.docx'; d=Document(p); print(len(d.paragraphs), len(d.tables), len(d.inline_shapes))"
```

Expected: `tables == 10`, `images == 16`.

- [ ] **Step 6: Commit and push**

Run:

```bash
git add paper/main.tex scripts/35_build_word_paper.py docs/final_submission_checklist.md docs/paper_style_reference_notes.md docs/superpowers/plans/2026-05-03-official-paper-excellent-style-rewrite.md
git commit -m "docs: align paper style with excellent examples"
git push origin main
```

Expected: GitHub `main` includes the rewrite.

## Self-Review

- Spec coverage: The plan keeps official Word/PDF structure and rewrites the body style to match excellent-paper examples.
- Placeholder scan: No TBD/TODO placeholders remain.
- Scope check: The plan touches only paper source, Word export compatibility, and documentation. Data processing and generated analysis outputs are explicitly out of scope.
