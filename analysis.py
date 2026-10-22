import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from pathlib import Path
from scipy import optimize, stats
from scipy.special import logsumexp

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "data/Example Dataset.csv"
OUTPUT_DIR = BASE_DIR / "output/"

THICKNESS_LOWERLIMIT = 550.0 # LSL
THICKNESS_UPPERLIMIT = 650.0 # USL
TARGET_TEMPERATURE_CPK = 2.0


def load_data(file_path: Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    return data


def fill_missing_data(data, columns):
    if isinstance(columns, str):
        columns = [columns]

    print("\nMissing values in each column:")
    print(data.isna().sum())

    data = data.copy()
    fill_values = data[columns].mean()
    empty_columns = fill_values[fill_values.isna()].index.tolist()
    if empty_columns:
        raise ValueError(f"Cannot calculate mean for empty columns: {empty_columns}")
    data[columns] = data[columns].fillna(fill_values)

    print("\nMean values used to fill missing data:")
    print(fill_values)
    return data


def set_time_index(data):
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y/%m/%d %H:%M")
    if data["timestamp"].isna().any():
        raise ValueError("The timestamp column contains missing values.")
    data = data.set_index("timestamp")

    print("\nFirst five rows with timestamp as index:")
    print(data.head(5))
    return data


def calculate_statistics(data, columns, group_by=None, statistics="mean", freq=None):
    group_keys = []
    if group_by is not None:
        group_keys = [group_by] if isinstance(group_by, str) else list(group_by)
    if freq is not None:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Time grouping requires a DatetimeIndex.")
        if data.index.hasnans:
            raise ValueError("Time grouping requires non-missing timestamps.")
        group_keys.append(pd.Grouper(freq=freq))

    if group_keys:
        values = data.groupby(group_keys, dropna=False)[columns]
    else:
        values = data[columns]
    result = values.agg(statistics)

    print(f"\nStatistics for {columns} ({statistics}):")
    print(result)
    return result


def save_figure(figure, output_dir: Path, filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    if not figure.get_constrained_layout():
        figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"\nPlot saved to: {output_path}")


def plot_data(data, plot_type, x=None, y=None, group_by=None, output_dir=OUTPUT_DIR):
    if plot_type not in ("line", "hist", "box"):
        raise ValueError("plot_type must be 'line', 'hist', or 'box'.")
    if plot_type == "line" and (x is None or y is None):
        raise ValueError("A line plot requires both x and y.")
    if plot_type == "hist" and (x is None or y is not None or group_by is not None):
        raise ValueError("A histogram uses x only; its y-axis is the count.")
    if plot_type == "box" and (y is None or group_by is not None):
        raise ValueError("A box plot requires y; use x for optional grouping.")

    plot_frame = data.reset_index()
    for field in (x, y, group_by):
        if field is not None and field not in plot_frame.columns:
            raise ValueError(f"Column not found: {field}")

    time_line = plot_type == "line" and pd.api.types.is_datetime64_any_dtype(plot_frame[x])
    if plot_type == "line":
        groups = list(plot_frame.groupby(group_by)) if group_by else [(None, plot_frame)]
        if not groups:
            raise ValueError("No data available for the line plot.")

    figure, axis = plt.subplots(figsize=(12, 5) if time_line else (10, 5))
    histogram = None

    if plot_type == "line":
        for index, (label, group) in enumerate(groups):
            group = group.sort_values(x, kind="stable")
            if time_line:
                dates = group[x].dt.normalize()
                gaps = group[x].diff()
                breaks = (dates.ne(dates.shift()) | gaps.le(pd.Timedelta(0))
                          | gaps.gt(pd.Timedelta(hours=1)))
                segments = group.groupby(breaks.cumsum(), sort=False)
            else:
                segments = [(None, group)]

            for segment_index, (_, segment) in enumerate(segments):
                axis.plot(
                    segment[x], segment[y], color=f"C{index % 10}",
                    marker="o", markersize=2.2, linewidth=0.6, alpha=0.6,
                    label=label if segment_index == 0 else None,
                )
        axis.set_ylabel(y)
        axis.set_xlabel(x)
        title = f"{y} vs. {x}"
        if time_line:
            first, last = plot_frame[x].min(), plot_frame[x].max()
            date_range = f"{first:%Y-%m-%d}"
            if first.normalize() != last.normalize():
                date_range += f" to {last:%Y-%m-%d}"
            title += f" ({date_range})"
        axis.set_title(title)
        if group_by:
            axis.legend(title=group_by)
        if time_line:
            locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    elif plot_type == "hist":
        _, bin_edges, _ = axis.hist(plot_frame[x].dropna(), bins="auto", edgecolor="black")
        histogram = {
            "bin_edges": bin_edges,
            "xlim": axis.get_xlim(),
            "ylim": axis.get_ylim(),
        }
        axis.set_xlabel(x)
        axis.set_ylabel("Count")
        axis.set_title(f"{x} Histogram")

    elif plot_type == "box":
        if x is None:
            axis.boxplot(plot_frame[y].dropna())
            axis.set_xticks([1], ["All data"])
        else:
            labels = []
            values = []
            for label, group in plot_frame.groupby(x):
                labels.append(str(label))
                values.append(group[y].dropna())
            axis.boxplot(values)
            axis.set_xticks(range(1, len(labels) + 1), labels)
            axis.set_xlabel(x)
        axis.set_ylabel(y)
        axis.set_title(f"{y} Box Plot")

    axis.grid(axis="y", alpha=0.3)
    filename_parts = [plot_type, x, y]
    if group_by:
        filename_parts.extend(["by", group_by])
    filename = "_".join(part for part in filename_parts if part is not None)
    filename = filename.replace(" ", "_").lower() + ".png"
    save_figure(figure, output_dir, filename)
    return histogram


def calculate_hourly_means(data, column="Thickness", group_by="product_id"):
    """Calculate each product's mean separately for each date and hour."""
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.hasnans:
        raise ValueError("Hourly comparison requires a non-missing DatetimeIndex.")
    frame = data[[group_by, column]].copy()
    frame["date"] = data.index.normalize()
    frame["hour"] = data.index.hour
    return frame.groupby([group_by, "date", "hour"], dropna=False)[column].mean().reset_index()


def plot_hourly_comparison(hourly, view, y="Thickness", group_by="product_id",
                           output_dir=OUTPUT_DIR):
    """Compare dates at a fixed hour, or hours within a fixed date."""
    if view not in ("same_hour", "same_day"):
        raise ValueError("view must be 'same_hour' or 'same_day'.")
    if hourly.empty:
        raise ValueError("No hourly data available for comparison.")

    if view == "same_hour":
        panel_column, x_column = "hour", "date"
        x_values = pd.DatetimeIndex(hourly["date"].unique()).sort_values()
        columns = 4
        title = f"{y}: hourly means across dates (one panel per hour)"
        x_label = "Measurement date (equally spaced)"
    else:
        panel_column, x_column = "date", "hour"
        x_values = pd.Index(range(24))
        columns = 2
        title = f"{y}: hourly means within each date"
        x_label = "Hour of day"

    panels = sorted(hourly[panel_column].unique())
    columns = min(columns, len(panels))
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 3.3 * rows + 0.8),
        sharex=True, sharey=True, squeeze=False, layout="constrained",
    )
    axes = axes.ravel()
    groups = list(hourly.groupby(group_by, dropna=False))
    positions = np.arange(len(x_values))

    for index, panel in enumerate(panels):
        axis = axes[index]
        for color_index, (label, group) in enumerate(groups):
            selected = group.loc[group[panel_column] == panel]
            # Reindexing leaves unmeasured combinations as NaN and breaks the lines.
            values = selected.set_index(x_column)[y].reindex(x_values)
            axis.plot(positions, values.to_numpy(), marker="o", markersize=3,
                      linewidth=1, alpha=0.8, color=f"C{color_index % 10}", label=label)

        if view == "same_hour":
            axis.set_title(f"{panel:02d}:00–{panel:02d}:59", fontsize=10, loc="left")
            axis.set_xticks(positions, x_values.strftime("%Y-%m-%d"),
                            rotation=55, ha="right", fontsize=8)
            axis.set_xlim(-0.35, len(x_values) - 0.65)
        else:
            axis.set_title(f"{pd.Timestamp(panel):%Y-%m-%d}", fontsize=10, loc="left")
            ticks = np.arange(0, 24, 3)
            axis.set_xticks(ticks, [f"{hour:02d}:00" for hour in ticks], fontsize=8)
            axis.set_xlim(-0.5, 23.5)

        axis.tick_params(axis="x", labelbottom=index + columns >= len(panels))
        axis.grid(axis="y", alpha=0.25)

    for axis in axes[len(panels):]:
        figure.delaxes(axis)
    axes[0].legend(title=group_by, fontsize=8, title_fontsize=8, loc="upper left")
    figure.suptitle(title, fontsize=14)
    figure.supxlabel(x_label)
    figure.supylabel(f"Hourly mean {y}")
    filename = f"hourly_{y.lower()}_{view}_by_{group_by}.png"
    save_figure(figure, output_dir, filename)


def calculate_mean_std(data, column):
    mean = data[column].mean()
    std = data[column].std(ddof=1)
    if pd.isna(std) or std <= 0:
        raise ValueError(f"{column} must have a positive sample standard deviation.")
    return mean, std


def calculate_cpk(data, column, lower_limit, upper_limit):
    if not lower_limit < upper_limit:
        raise ValueError("The lower limit must be less than the upper limit.")

    mean, std = calculate_mean_std(data, column)

    cpk = min((mean - lower_limit) / (3 * std),
              (upper_limit - mean) / (3 * std))

    print(f"\n{column} capability:")
    print(f"Mean: {mean:.6f}")
    print(f"Sample standard deviation: {std:.6f}")
    print(f"Cpk: {cpk:.6f}")
    return cpk


def calculate_symmetric_limits(data, column, target_cpk):
    if not target_cpk > 0:
        raise ValueError("The target Cpk must be positive.")

    mean, std = calculate_mean_std(data, column)

    half_width = 3 * target_cpk * std
    lower_limit = mean - half_width
    upper_limit = mean + half_width

    print(f"\n{column} symmetric limits for target Cpk = {target_cpk:g}:")
    print(f"Lower specification limit: {lower_limit:.6f}")
    print(f"Upper specification limit: {upper_limit:.6f}")
    return lower_limit, upper_limit

# Fit model definition : Norm, Norm2, Norm3, Lognormal
def normal(x, mean, std, *, log=False):
    density = stats.norm.logpdf(x, loc=mean, scale=std)
    return density if log else np.exp(density)


def normal_mixture(x, means, stds, weights, *, log=False):
    component_logs = stats.norm.logpdf(np.asarray(x)[..., None], loc=means, scale=stds)
    with np.errstate(divide="ignore"):
        density = logsumexp(component_logs + np.log(weights), axis=-1)
    return density if log else np.exp(density)


def norm2(x, mean1, std1, mean2, std2, weight, *, log=False):
    return normal_mixture(x, [mean1, mean2], [std1, std2],
                          [weight, 1 - weight], log=log)


def norm3(x, mean1, std1, mean2, std2, mean3, std3,
          weight1, remaining_fraction, *, log=False):
    weights = [weight1, (1 - weight1) * remaining_fraction,
               (1 - weight1) * (1 - remaining_fraction)]
    return normal_mixture(x, [mean1, mean2, mean3], [std1, std2, std3],
                          weights, log=log)


def lognormal(x, shape, scale, *, log=False):
    density = stats.lognorm.logpdf(x, s=shape, loc=0, scale=scale)
    return density if log else np.exp(density)


def mixture_initials(values, count):
    starts = []
    for edge in (0.1, 0.25, 0.4):
        means = np.quantile(values, np.linspace(edge, 1 - edge, count))
        for width in (0.5, 1.0):
            parameters = []
            for mean in means:
                parameters.extend([mean, values.std() * width])
            parameters.extend(1 / (count - index) for index in range(count - 1))
            starts.append(parameters)
    return starts


def normal_bounds(values, count):
    minimum_std = max(values.std() * 1e-3, 1e-8)
    return [(None, None), (minimum_std, None)] * count + [(0, 1)] * (count - 1)


def normal_mixture_quantile(probability, means, stds, weights):
    if not 0 < probability < 1:
        raise ValueError("The quantile probability must be between zero and one.")
    means, stds, weights = (np.asarray(values, dtype=float)
                           for values in (means, stds, weights))
    active = weights > 0
    means, stds, weights = means[active], stds[active], weights[active]
    component_quantiles = stats.norm.ppf(probability, loc=means, scale=stds)
    lower, upper = component_quantiles.min(), component_quantiles.max()
    if lower == upper:
        return float(lower)

    def cdf_difference(value):
        return np.sum(weights * stats.norm.cdf(value, loc=means, scale=stds)) - probability

    return optimize.brentq(cdf_difference, lower, upper)


# Default fitting configuration for each model: 
MODEL_SETTINGS = {
    normal: {
        "initial": lambda values: [[values.mean(), values.std()]],
        "bounds": lambda values: normal_bounds(values, 1),
        "scale_indices": [1],
        "quantile": lambda probability, parameters: stats.norm.ppf(
            probability, loc=parameters[0], scale=parameters[1]),
    },
    norm2: {
        "initial": lambda values: mixture_initials(values, 2),
        "bounds": lambda values: normal_bounds(values, 2),
        "scale_indices": [1, 3],
        "quantile": lambda probability, parameters: normal_mixture_quantile(
            probability, parameters[0:4:2], parameters[1:4:2],
            [parameters[4], 1 - parameters[4]]),
    },
    norm3: {
        "initial": lambda values: mixture_initials(values, 3),
        "bounds": lambda values: normal_bounds(values, 3),
        "scale_indices": [1, 3, 5],
        "quantile": lambda probability, parameters: normal_mixture_quantile(
            probability, parameters[0:6:2], parameters[1:6:2],
            [parameters[6], (1 - parameters[6]) * parameters[7],
             (1 - parameters[6]) * (1 - parameters[7])]),
    },
    lognormal: {
        "initial": lambda values: [[np.log(values).std(), np.exp(np.log(values).mean())]],
        "bounds": lambda values: [(1e-8, None), (1e-8, None)],
        "positive_data": True,
        "scale_indices": [0, 1],
        "quantile": lambda probability, parameters: stats.lognorm.ppf(
            probability, s=parameters[0], loc=0, scale=parameters[1]),
    },
}


def fit(values, model):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Fitting requires a finite, one-dimensional sample of at least two values.")
    if values.std() <= 0:
        raise ValueError("Fitting requires nonconstant data.")
    if model not in MODEL_SETTINGS:
        raise ValueError("Add the model's initial parameters and bounds to MODEL_SETTINGS.")

    settings = MODEL_SETTINGS[model]
    if settings.get("positive_data", False) and (values <= 0).any():
        raise ValueError(f"{model.__name__} requires positive data.")
    bounds = settings["bounds"](values)

    def negative_log_likelihood(parameters):
        log_density = model(values, *parameters, log=True)
        if not np.isfinite(log_density).all():
            return np.inf
        return -float(np.sum(log_density))

    best = None
    for initial in settings["initial"](values):
        result = optimize.minimize(
            negative_log_likelihood, initial, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 50},
        )
        if result.success and np.isfinite(result.fun):
            if best is None or result.fun < best.fun:
                best = result
    if best is None:
        raise RuntimeError(f"No successful likelihood fit for {model.__name__}.")

    parameter_count = len(best.x)
    scale_bound_hit = any(
        best.x[index] <= bounds[index][0] * 1.01
        for index in settings.get("scale_indices", [])
    )
    fitted = {
        "model": model, "name": model.__name__, "parameters": best.x,
        "parameter_count": parameter_count, "log_likelihood": -best.fun,
        "bic": 2 * best.fun + parameter_count * np.log(len(values)),
        "sample_count": len(values), "scale_bound_hit": scale_bound_hit,
    }
    print(f"\n{model.__name__} fitted parameters: {np.round(best.x, 6)}")
    return fitted


def compare_fits(results):
    if not results or len({result["sample_count"] for result in results}) != 1:
        raise ValueError("Compare fits made with the same number of observations.")
    comparison = pd.DataFrame([{
        "Model": result["name"], "Parameters": result["parameter_count"],
        "Log likelihood": result["log_likelihood"], "BIC": result["bic"],
        "Scale at bound": result["scale_bound_hit"],
    } for result in results]).sort_values("BIC").reset_index(drop=True)
    comparison["Delta BIC"] = comparison["BIC"] - comparison["BIC"].min()
    best = min(results, key=lambda result: result["bic"])

    print("\nDistribution comparison:")
    print(comparison.to_string(index=False, float_format="%.6f"))
    if best["scale_bound_hit"]:
        print("The lowest-BIC fit reaches its scale bound; model selection is unreliable.")
    else:
        print(f"Preferred candidate: {best['name']} (lowest BIC).")
    print("BIC balances likelihood against parameter count.")
    return comparison, best


def calculate_fitted_cpk(fitted, lower_limit, upper_limit):
    if not np.isfinite([lower_limit, upper_limit]).all() or lower_limit >= upper_limit:
        raise ValueError("Specification limits must be finite and lower must be less than upper.")
    quantile = MODEL_SETTINGS[fitted["model"]].get("quantile")
    if quantile is None:
        raise ValueError("Add the model's quantile function to MODEL_SETTINGS.")

    tail_probability = stats.norm.cdf(-3)
    lower, median, upper = [
        float(quantile(probability, fitted["parameters"]))
        for probability in (tail_probability, 0.5, 1 - tail_probability)
    ]
    if not np.isfinite([lower, median, upper]).all() or not lower < median < upper:
        raise ValueError("The fitted quantiles must be finite and strictly increasing.")

    cpl = (median - lower_limit) / (median - lower)
    cpu = (upper_limit - median) / (upper - median)
    cpk = min(cpl, cpu)

    print(f"\nCapability based on fitted {fitted['name']}:")
    print(f"Specification limits: [{lower_limit:g}, {upper_limit:g}]")
    print(f"Fitted quantiles (0.13499%, 50%, 99.86501%): {lower:.6f}, {median:.6f}, {upper:.6f}")
    print(f"Cpl: {cpl:.6f}; Cpu: {cpu:.6f}")
    print(f"Fitted-distribution Cpk (percentile method): {cpk:.6f}")
    return cpk


def plot_fits(values, results, output_dir=OUTPUT_DIR, histogram=None):
    label = getattr(values, "name", None) or "Value"
    values = np.asarray(values, dtype=float)
    figure, axis = plt.subplots(figsize=(10, 5))
    bins = "auto" if histogram is None else histogram["bin_edges"]
    _, bin_edges, _ = axis.hist(values, bins=bins, alpha=0.4, edgecolor="black", label="Data")
    if histogram is None:
        histogram = {"xlim": axis.get_xlim(), "ylim": axis.get_ylim()}

    bin_widths = np.diff(bin_edges)
    if not np.allclose(bin_widths, bin_widths[0]):
        plt.close(figure)
        raise ValueError("Fitted count curves require equal-width histogram bins.")

    axis.set_xlim(histogram["xlim"])
    axis.set_ylim(histogram["ylim"])
    axis.set_autoscale_on(False)
    grid = np.linspace(*histogram["xlim"], 700)
    for result in results:
        density = result["model"](grid, *result["parameters"])
        fitted_counts = density * len(values) * bin_widths[0]
        axis.plot(grid, fitted_counts, label=result["name"], clip_on=True)
    axis.set(xlabel=label, ylabel="Count", title=f"{label} Distribution Fits (n={len(values)})")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    save_figure(figure, output_dir, f"{label.lower()}_distribution_fit.png")


def main():

    # Section 1
    data = load_data(CSV_FILE)

    print("\nFirst five rows before filling missing values:")
    print(data.head(5))

    if list(data.columns) == ["Thickness"]:
        print("\nThickness-only validation data: running Section 4.")
        data = fill_missing_data(data, columns="Thickness")
    else:
        data = fill_missing_data(data, columns=["Thickness", "Temperature"])
        data = set_time_index(data)
        calculate_statistics(
            data, columns=["Thickness", "Temperature"], group_by="product_id"
        )
        calculate_statistics(
            data, columns="Thickness", group_by="product_id",
            statistics=["min", "max"], freq="D"
        )

    # Section 2
        plot_data(data, "line", x="timestamp", y="Thickness", group_by="product_id")
        hourly = calculate_hourly_means(data, column="Thickness", group_by="product_id")
        plot_hourly_comparison(hourly, "same_hour")
        plot_hourly_comparison(hourly, "same_day")
        plot_data(data, "box", y="Thickness")
        plot_data(data, "hist", x="Temperature")
        plot_data(data, "box", x="product_id", y="Thickness")

    # Section 3
        calculate_cpk(data, "Thickness", THICKNESS_LOWERLIMIT, THICKNESS_UPPERLIMIT)
        calculate_symmetric_limits(data, "Temperature", TARGET_TEMPERATURE_CPK)

    # Section 4
    histogram = plot_data(data, "hist", x="Thickness")
    values = data["Thickness"]

    results = [
        fit(values, normal),
        fit(values, norm2),
        fit(values, norm3),
        fit(values, lognormal),
    ]

    _, best = compare_fits(results)
    plot_fits(values, results, histogram=histogram)
    calculate_fitted_cpk(best, THICKNESS_LOWERLIMIT, THICKNESS_UPPERLIMIT)


if __name__ == "__main__":
    main()
