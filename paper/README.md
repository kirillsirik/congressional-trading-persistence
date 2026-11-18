# Paper

| File | Contents |
|---|---|
| `Congressional_Trading_Complete.pdf` | Main text, references and Appendices A–G |
| `Congressional_Trading_Reading_Copy.pdf` | Main text and references |
| `Congressional_Trading_Appendix.pdf` | Appendices A–G and their references |

Keep the three PDFs in one folder: cross-references between them link by file name.

The rest of this folder holds the programs that turn the stored results in `../data/` into the tables, figures
and macros that the paper prints, and those exhibits:

| Folder | Contents | Written by |
|---|---|---|
| `writing_sample_generated/` | The table bodies and figure wrappers of the main text and of Appendices C and F | `make tables`, `make figures` |
| `generated/` | The macro files whose macros the paper prints | `make macros` |
| `revised_generated/` | The row macros of Tables B1, C1 and D1 | `make tables` |
| `figures/` | The six figures of the paper, and the stored validation record of Figure G1 | `make figures` |
| `data/` | Stored results of the average-return analysis read by `generate_assets.py`, and quoted literature values | inputs |
| `scripts/`, `generate_assets.py` | The generators | |

The generators also write intermediate files into these folders; `.gitignore` keeps them out of the repository.
See the repository README for the map from each exhibit to its code and data.
