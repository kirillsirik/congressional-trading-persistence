"""Exports: LaTeX / Markdown / JSON resolved from the ledger only (JC-023).

Shells carry ledger keys; render-time lookup is the only way a number reaches
an export. There is no API accepting a literal display value.
"""

from __future__ import annotations

import json
from pathlib import Path

from .numbers_ledger import NumbersLedger
from .shells import Shell


class ExportError(RuntimeError):
    pass


def _resolve_cell(column: str, value, ledger: NumbersLedger) -> str:
    if value is None or value == "" or (isinstance(value, float) and value != value):
        return ""
    if column == "ledger_key" or column.endswith("_key"):
        entry = ledger.require(str(value))
        return f"{entry.rounded_value:g}"
    return str(value)


def _rendered_rows(shell: Shell, ledger: NumbersLedger) -> list[list[str]]:
    shell.validate()
    ledger.assert_shell_covered(shell.frame)
    return [
        [_resolve_cell(col, row[col], ledger) for col in shell.frame.columns]
        for _, row in shell.frame.iterrows()
    ]


def to_markdown(shell: Shell, ledger: NumbersLedger) -> str:
    rows = _rendered_rows(shell, ledger)
    header = list(shell.frame.columns)
    lines = [
        f"### {shell.caption}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    lines += ["", *[f"*Note: {n}*" for n in shell.notes]]
    return "\n".join(lines) + "\n"


def to_latex(shell: Shell, ledger: NumbersLedger) -> str:
    rows = _rendered_rows(shell, ledger)
    header = list(shell.frame.columns)
    colspec = "l" * len(header)
    lines = [
        "\\begin{table}[htbp]\\centering",
        f"\\caption{{{_tex_escape(shell.caption)}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(_tex_escape(h) for h in header) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(_tex_escape(c) for c in r) + " \\\\" for r in rows]
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        *[f"\\par\\footnotesize {_tex_escape(n)}" for n in shell.notes],
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def to_json(shell: Shell, ledger: NumbersLedger) -> str:
    rows = _rendered_rows(shell, ledger)
    return json.dumps(
        {
            "output_id": shell.spec.output_id,
            "caption": shell.caption,
            "notes": shell.notes,
            "columns": list(shell.frame.columns),
            "rows": rows,
            "ledger_hash": ledger.content_hash(),
        },
        indent=2,
    )


def export_all(
    shells: dict[str, Shell], ledger: NumbersLedger, out_dir: Path
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for output_id, shell in sorted(shells.items()):
        base = out_dir / output_id
        base.with_suffix(".md").write_text(to_markdown(shell, ledger))
        base.with_suffix(".tex").write_text(to_latex(shell, ledger))
        base.with_suffix(".json").write_text(to_json(shell, ledger))
        written[output_id] = str(base)
    return written


def _tex_escape(text: str) -> str:
    for char in "&%$#_{}":
        text = text.replace(char, "\\" + char)
    return text
