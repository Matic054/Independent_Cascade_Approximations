# Paper analysis

This directory contains the final analysis layer used to turn held-out benchmark
CSVs into paper-facing tables and figures.

## Files

- `paper_analysis.ipynb`: readable, section-by-section analysis notebook.
- `paper_analysis.py`: reusable analysis functions used by the notebook.
- `figures/`: generated paper-facing figures.
- `tables/`: generated numerical summaries and paired-comparison tables.

The analysis intentionally reads only the final held-out benchmark outputs and
small selected-hyperparameter files. It does **not** require the very large
`calibration_curves.csv` files.

Run from the repository root with:

```bash
jupyter lab analysis/paper_analysis.ipynb
```

or generate all standard outputs non-interactively with:

```bash
python -c "from analysis.paper_analysis import generate_all; generate_all('.')"
```

Historical result archives may not contain every newer method. The analysis
checks method availability before running paired comparisons, so missing ALE2 or
IPL2 results are reported through `tables/method_availability.csv` rather than
causing the notebook to fail.
