# ICLR 2027 Draft Template

This project is a clean, general-purpose draft scaffold for an ICLR 2027 submission.
Because the official ICLR 2027 style has not yet been released, it currently uses
the ICLR 2026 conference style and bibliography format.

## Compile

```bash
latexmk -pdf draft.tex
```

The main file is `draft.tex`. Paper content is split across `sections/`, and
references belong in `references.bib`.

Keep `\iclrfinalcopy` commented while preparing an anonymous submission. When the
official ICLR 2027 template becomes available, replace the 2026 `.sty` and `.bst`
files and update their names in `draft.tex`.
