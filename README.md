# CSV Analysis Assignment

`analysis.py` performs CSV loading, missing-value filling, grouped statistics,
plotting, process capability calculations, and distribution fitting for Sections 1–4.

## Environment setup

Use Python 3.10 or later. Open a terminal in this project directory, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If `.venv` already exists and its dependencies are installed, only run
`source .venv/bin/activate`. In your IDE, select `.venv/bin/python` as the Python
interpreter.

The analysis uses pandas, NumPy, Matplotlib, and SciPy. `pathlib` is part of the
Python standard library. `requirements.txt` also includes scikit-learn from an
older implementation; the current script does not use it.

## Run the script

With the virtual environment activated:

```bash
python analysis.py
```

Alternatively, run directly without activating the environment:

```bash
.venv/bin/python analysis.py
```

The script has no command-line arguments. Configure it by editing the constants
at the top of `analysis.py`. To save the printed results as a text file:

```bash
python analysis.py > analysis_results.txt
```

For an environment without a graphical display, use
`MPLBACKEND=Agg python analysis.py`.

## Parameters to change

The current settings at the top of `analysis.py` are:

```python
CSV_FILE = BASE_DIR / "validation/validate_data.csv"
OUTPUT_DIR = BASE_DIR / "validation/output/data/"

THICKNESS_LOWERLIMIT = 550.0
THICKNESS_UPPERLIMIT = 650.0
TARGET_TEMPERATURE_CPK = 2.0
```

| Parameter | Purpose |
| --- | --- |
| `CSV_FILE` | Input CSV file. |
| `OUTPUT_DIR` | Directory for generated PNG figures; created automatically. |
| `THICKNESS_LOWERLIMIT` | Lower specification limit (LSL), used in Sections 3 and 4. |
| `THICKNESS_UPPERLIMIT` | Upper specification limit (USL), used in Sections 3 and 4. |
| `TARGET_TEMPERATURE_CPK` | Target Cpk used to calculate temperature limits in Section 3. |

`BASE_DIR` is the directory containing `analysis.py`; keep it unchanged. Relative
input and output paths are resolved from this directory. The thickness lower
limit must be less than the upper limit, and the temperature target must be
positive. Temperature limits are calculated symmetrically around the sample mean.

For example, to analyze the complete dataset:

```python
CSV_FILE = BASE_DIR / "data/Example Dataset.csv"
OUTPUT_DIR = BASE_DIR / "data/output"
```

Available input files:

| File | Use |
| --- | --- |
| `data/Example Dataset.csv` | Complete assignment dataset. |
| `tests/test_data.csv` | Odd-numbered data records from the complete dataset. |
| `validation/validate_data.csv` | Even-numbered data records from the complete dataset. |
| `validation/validate_norm.csv` | 100 generated normal samples. |
| `validation/validate_norm2.csv` | 100 generated two-component normal-mixture samples. |
| `validation/validate_norm3.csv` | 100 generated three-component normal-mixture samples. |
| `validation/validate_norm_lognormal.csv` | 100 generated lognormal samples. |

Odd/even numbering excludes the header. Each input is analyzed independently.
Choose a different `OUTPUT_DIR` for each dataset to retain earlier figures.

## Input format

For the full analysis, the CSV must contain these exact, case-sensitive columns:

```csv
timestamp,product_id,Thickness,Temperature
2024/11/19 08:30,type0,600,4.2
2024/11/19 09:15,type1,610,4.3
```

Additional columns such as `Unit ID` are allowed. Timestamps must use
`YYYY/MM/DD HH:MM`. To use a different timestamp format, change the `format`
argument in `set_time_index()`.

Missing thickness and temperature values are filled with their respective column
means before analysis. Missing or invalid timestamps and entirely empty measurement
columns cause an error. The default four-model comparison requires positive,
finite, nonconstant thickness values, with at least two observations after filling.

A CSV containing **only** the `Thickness` column automatically runs Section 4
only. This supports the generated `validate_norm*.csv` files above.

## Optional changes in `main()`

Change the arguments to `plot_data()` to select fields and plot types:

```python
# Line plot: x and y, with optional product grouping.
plot_data(data, "line", x="timestamp", y="Thickness", group_by="product_id")

# Histogram: x is the measured variable; y is the count.
plot_data(data, "hist", x="Temperature")

# Box plot: y is the measured variable; x optionally defines groups.
plot_data(data, "box", y="Thickness")
plot_data(data, "box", x="product_id", y="Thickness")
```

The additional time comparisons use hourly means for each product and date:

```python
hourly = calculate_hourly_means(data, column="Thickness", group_by="product_id")
plot_hourly_comparison(hourly, "same_hour")
plot_hourly_comparison(hourly, "same_day")
```

`same_hour` compares measurement dates in a separate panel for each hour, with
dates equally spaced. `same_day` compares hours in a separate panel for each date.
Unmeasured time slots remain empty and break the lines.

To change the candidate distributions, edit the `results` list in `main()`:

```python
results = [
    fit(values, normal),
    fit(values, norm2),
    fit(values, norm3),
    fit(values, lognormal),
]
```

The script selects the candidate with the lowest BIC and calculates its
percentile-based Cpk using the thickness specification limits above.

## Outputs

Statistics, fitted parameters, model comparisons, and capability results are
printed to the terminal. Figures are saved as PNG files in `OUTPUT_DIR`; no plot
window opens. Repeated runs overwrite figures with the same filenames.
The source CSV is not modified. CSV and JSON result files are not exported
by the current script.
