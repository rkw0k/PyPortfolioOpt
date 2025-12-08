"""
The ``plotting`` module houses all the functions to generate various plots.

Currently implemented:

  - ``plot_covariance`` - plot a correlation matrix
  - ``plot_dendrogram`` - plot the hierarchical clusters in a portfolio
  - ``plot_efficient_frontier`` – plot the efficient frontier from an EfficientFrontier or CLA object
  - ``plot_weights`` - bar chart of weights
"""

import warnings

import numpy as np
import scipy.cluster.hierarchy as sch

from . import CLA, EfficientFrontier, exceptions, risk_models


def _import_matplotlib():
    """Helper function to import matplotlib only when needed"""
    try:
        import matplotlib.pyplot as plt

        return plt
    except (ModuleNotFoundError, ImportError):  # pragma: no cover
        raise ImportError("Please install matplotlib via pip or poetry")


def _get_plotly():
    """Helper function to import plotly only when needed"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        return go, make_subplots
    except (ModuleNotFoundError, ImportError):
        raise ImportError(
            "Please install plotly via pip or poetry to use interactive plots"
        )


def _plot_io(**kwargs):
    """
    Helper method to optionally save the figure to file.

    :param filename: name of the file to save to, defaults to None (doesn't save)
    :type filename: str, optional
    :param dpi: dpi of figure to save or plot, defaults to 300
    :type dpi: int (between 50-500)
    :param showfig: whether to plt.show() the figure, defaults to False
    :type showfig: bool, optional
    """
    plt = _import_matplotlib()

    filename = kwargs.get("filename", None)
    showfig = kwargs.get("showfig", False)
    dpi = kwargs.get("dpi", 300)

    plt.tight_layout()
    if filename:
        plt.savefig(fname=filename, dpi=dpi)
    if showfig:  # pragma: no cover
        plt.show()


def plot_covariance(cov_matrix, plot_correlation=False, show_tickers=True, **kwargs):
    """
    Generate a basic plot of the covariance (or correlation) matrix, given a
    covariance matrix.

    :param cov_matrix: covariance matrix
    :type cov_matrix: pd.DataFrame or np.ndarray
    :param plot_correlation: whether to plot the correlation matrix instead, defaults to False.
    :type plot_correlation: bool, optional
    :param show_tickers: whether to use tickers as labels (not recommended for large portfolios),
                        defaults to True
    :type show_tickers: bool, optional

    :return: matplotlib axis
    :rtype: matplotlib.axes object
    """
    plt = _import_matplotlib()

    if plot_correlation:
        matrix = risk_models.cov_to_corr(cov_matrix)
    else:
        matrix = cov_matrix
    fig, ax = plt.subplots()

    cax = ax.imshow(matrix)
    fig.colorbar(cax)

    if show_tickers:
        ax.set_xticks(np.arange(0, matrix.shape[0], 1))
        ax.set_xticklabels(matrix.index)
        ax.set_yticks(np.arange(0, matrix.shape[0], 1))
        ax.set_yticklabels(matrix.index)
        plt.xticks(rotation=90)

    _plot_io(**kwargs)

    return ax


def plot_portfolio_summary(
    values,
    initial_value: float = 1_000_000,
    sample_trials: int | None = None,
    inflation_rate: float | None = None,
    inflation_series=None,
    trading_periods_per_year: int = 252,
    ax=None,
    figsize=(12, 9),
    showfig=False,
    filename: str | None = None,
    **kwargs,
):
    """Plot portfolio summary charts: portfolio value, inflation-adjusted value,
    annual returns and drawdowns.

    Parameters
    - values: pd.Series (portfolio values) or pd.DataFrame (trials or multi-ticker prices).
      If DataFrame and values appear normalized (start ~100), they will be scaled by
      `initial_value` as in `plot_simulation_results`.
    - initial_value: used to scale normalized series when DataFrame of trials provided.
    - sample_trials: if DataFrame of trials provided, number of individual trials to overlay.
    - inflation_rate / inflation_series: same semantics as `plot_simulation_results`.
    - trading_periods_per_year: used when index is integer-based to convert periods->years.
    """
    plt = _import_matplotlib()
    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd is None:
        raise TypeError("pandas is required for plot_portfolio_summary")

    # Normalize inputs to a DataFrame of monetary values (index as time)
    if isinstance(values, pd.DataFrame):
        df = values.copy()
        # detect normalized (start ~= 100) heuristically
        first_row = df.iloc[0].dropna()
        if not first_row.empty and (first_row.abs() > 0).all() and (first_row.mean() < 200):
            vals = df / 100.0 * float(initial_value)
        else:
            vals = df.astype(float)
        # choose median series for summary
        value_series = vals.median(axis=1)
    elif isinstance(values, pd.Series):
        value_series = values.astype(float).copy()
        vals = None
    else:
        raise TypeError("values must be a pandas Series or DataFrame")

    # Apply inflation adjustment if requested (same logic as plot_simulation_results)
    inflation_applied = False
    if inflation_series is not None:
        try:
            inf = pd.Series(inflation_series)
            inf = inf.reindex(value_series.index)
            if inf.isnull().any():
                inf = pd.Series(inflation_series).reset_index(drop=True)
                inf.index = value_series.index
            deflator = inf / float(inf.iloc[0])
            real_series = value_series.divide(deflator.values, axis=0)
            inflation_applied = True
        except Exception:
            inflation_applied = False
            real_series = value_series
    elif inflation_rate is not None:
        t = np.arange(len(value_series))
        per_period = (1.0 + float(inflation_rate)) ** (t / float(trading_periods_per_year))
        real_series = value_series / per_period
        inflation_applied = True
    else:
        real_series = value_series

    # Compute annual returns
    if hasattr(value_series.index, "year"):
        # Datetime-like index
        annual = value_series.resample("A").last().pct_change().dropna()
        annual_real = real_series.resample("A").last().pct_change().dropna()
        x = annual.index
    else:
        # integer index: bucket by trading_periods_per_year
        years = (np.arange(len(value_series)) // trading_periods_per_year).astype(int)
        annual = value_series.groupby(years).apply(lambda s: float(s.iloc[-1]) / float(s.iloc[0]) - 1 if len(s) > 1 else np.nan).dropna()
        annual_real = real_series.groupby(years).apply(lambda s: float(s.iloc[-1]) / float(s.iloc[0]) - 1 if len(s) > 1 else np.nan).dropna()
        x = annual.index

    # Compute drawdown series
    running_max = value_series.cummax()
    drawdown = value_series / running_max - 1

    # Create subplots: value (top), annual returns (middle), drawdown (bottom)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=False)
    ax_val, ax_ann, ax_dd = axes

    # Plot nominal and real portfolio values
    ax_val.plot(value_series.index, value_series, label="Nominal", color="#1f77b4", linewidth=2)
    if inflation_applied:
        ax_val.plot(real_series.index, real_series, label="Inflation-adjusted", color="#ff7f0e", linewidth=2)
    # If trials provided and sample_trials, overlay some
    if vals is not None and sample_trials and sample_trials > 0:
        ncols = vals.shape[1]
        sample = vals.sample(n=min(sample_trials, ncols), axis=1)
        for col in sample.columns:
            ax_val.plot(sample.index, sample[col], lw=0.8, color="#444444", alpha=0.5)

    ax_val.set_title("Portfolio Value" + (" (inflation-adjusted)" if inflation_applied else ""))
    ax_val.set_ylabel("Value")
    ax_val.legend()

    # Annual returns bar chart (show nominal and real side-by-side if both exist)
    width = 0.35
    if isinstance(x, pd.DatetimeIndex):
        ax_ann.bar(annual.index, annual.values * 100, width=200, label="Nominal")
        if inflation_applied:
            ax_ann.bar(annual_real.index + pd.DateOffset(days=40), annual_real.values * 100, width=200, label="Real")
    else:
        idx = np.arange(len(annual))
        ax_ann.bar(idx - width / 2, annual.values * 100, width=width, label="Nominal")
        if inflation_applied:
            ax_ann.bar(idx + width / 2, annual_real.values * 100, width=width, label="Real")
        ax_ann.set_xticks(idx)
        ax_ann.set_xticklabels([str(i) for i in x])

    ax_ann.set_ylabel("Annual Return (%)")
    ax_ann.set_title("Annual Returns")
    ax_ann.legend()

    # Drawdown plot
    ax_dd.fill_between(value_series.index, drawdown * 100, color="tab:red", alpha=0.6)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_title("Drawdowns")
    ax_dd.axhline(0, color="#000000", linewidth=0.8)

    plt.tight_layout()

    # Save/show
    if filename:
        plt.savefig(filename)
    if showfig:  # pragma: no cover
        plt.show()

    return (ax_val, ax_ann, ax_dd)


def plot_dendrogram(hrp, ax=None, show_tickers=True, **kwargs):
    """
    Plot the clusters in the form of a dendrogram.

    :param hrp: HRPpt object that has already been optimized.
    :type hrp: object
    :param show_tickers: whether to use tickers as labels (not recommended for large portfolios),
                        defaults to True
    :type show_tickers: bool, optional
    :param filename: name of the file to save to, defaults to None (doesn't save)
    :type filename: str, optional
    :param showfig: whether to plt.show() the figure, defaults to False
    :type showfig: bool, optional
    :return: matplotlib axis
    :rtype: matplotlib.axes object
    """
    plt = _import_matplotlib()

    ax = ax or plt.gca()

    if hrp.clusters is None:
        warnings.warn(
            "hrp param has not been optimized.  Attempting optimization.",
            RuntimeWarning,
        )
        hrp.optimize()

    if show_tickers:
        sch.dendrogram(hrp.clusters, labels=hrp.tickers, ax=ax, orientation="top")
        ax.tick_params(axis="x", rotation=90)
        plt.tight_layout()
    else:
        sch.dendrogram(hrp.clusters, no_labels=True, ax=ax)

    _plot_io(**kwargs)

    return ax


def _plot_cla(cla, points, ax, show_assets, show_tickers, interactive):
    """
    Helper function to plot the efficient frontier from a CLA object
    """
    if interactive:
        go, _ = _get_plotly()

    if cla.weights is None:
        cla.max_sharpe()
    optimal_ret, optimal_risk, sharpe_max = cla.portfolio_performance()
    opt_weights = cla.weights
    if cla.frontier_values is None:
        cla.efficient_frontier(points=points)

    mus, sigmas, weights = cla.frontier_values

    if interactive:
        # Create the label
        hovertemplate = "Risk: %{x}<br>Return: %{y}<extra>"
        # Loop over each asset and append its information
        for i, ticker in enumerate(cla.tickers):
            hovertemplate += f"{ticker}: %{{customdata[{i}]:.4%}}<br>"
        hovertemplate += "</extra>"

        ax.add_trace(
            go.Scatter(
                x=sigmas,
                y=mus,
                mode="lines",
                line=dict(color="lightskyblue", width=2),
                name="Efficient frontier",
                customdata=weights,
                hovertemplate=hovertemplate,
            )
        )
        ax.add_trace(
            go.Scatter(
                x=[optimal_risk],
                y=[optimal_ret],
                customdata=[opt_weights, [sharpe_max]],
                mode="markers",
                name="Max Sharpe Portfolio",
                marker=dict(size=12, symbol="x", color="coral"),
                hovertemplate="Sharpe: %{{customdata[1]:.4}}<br>" + hovertemplate,
            )
        )
    else:
        ax.plot(sigmas, mus, label="Efficient frontier")
        ax.scatter(
            optimal_risk, optimal_ret, marker="x", s=100, color="r", label="optimal"
        )

    asset_mu = cla.expected_returns
    asset_sigma = np.sqrt(np.diag(cla.cov_matrix))
    if show_assets:
        if interactive:
            ax.add_trace(
                go.Scatter(
                    x=asset_sigma,
                    y=asset_mu,
                    mode="markers",
                    name="Assets",
                    marker=dict(size=10, symbol="star-diamond", color="silver"),
                    hovertemplate="Risk: %{x}<br>Return: %{y}<extra></extra>",
                )
            )
        else:
            ax.scatter(
                asset_sigma,
                asset_mu,
                s=30,
                color="k",
                label="assets",
            )
            if show_tickers:
                for i, label in enumerate(cla.tickers):
                    ax.annotate(label, (asset_sigma[i], asset_mu[i]))
    return ax


def _ef_default_returns_range(ef, points):
    """
    Helper function to generate a range of returns from the GMV returns to
    the maximum (constrained) returns
    """
    ef_minvol = ef.deepcopy()
    ef_maxret = ef.deepcopy()

    ef_minvol.min_volatility()
    min_ret = ef_minvol.portfolio_performance()[0]
    max_ret = ef_maxret._max_return()
    return np.linspace(min_ret, max_ret - 0.0001, points)


def _plot_ef(ef, ef_param, ef_param_range, ax, show_assets, show_tickers, interactive):
    """
    Helper function to plot the efficient frontier from an EfficientFrontier object
    """
    if interactive:
        go, _ = _get_plotly()

    mus, sigmas = [], []

    # Create a portfolio for each value of ef_param_range
    for param_value in ef_param_range:
        try:
            if ef_param == "utility":
                ef.max_quadratic_utility(param_value)
            elif ef_param == "risk":
                ef.efficient_risk(param_value)
            elif ef_param == "return":
                ef.efficient_return(param_value)
            else:
                raise NotImplementedError(
                    "ef_param should be one of {'utility', 'risk', 'return'}"
                )
        except exceptions.OptimizationError:
            continue
        except ValueError:
            warnings.warn(
                "Could not construct portfolio for parameter value {:.3f}".format(
                    param_value
                )
            )

        ret, sigma, _ = ef.portfolio_performance()
        mus.append(ret)
        sigmas.append(sigma)

    if interactive:
        ax.add_trace(
            go.Scatter(
                x=sigmas,
                y=mus,
                mode="lines",
                name="Efficient frontier",
                line=dict(width=2, color="lightskyblue"),
            )
        )
    else:
        ax.plot(sigmas, mus, label="Efficient frontier")

    asset_mu = ef.expected_returns
    asset_sigma = np.sqrt(np.diag(ef.cov_matrix))
    if show_assets:
        if interactive:
            ax.add_trace(
                go.Scatter(
                    x=asset_sigma,
                    y=asset_mu,
                    mode="markers",
                    marker=dict(size=10, symbol="star-diamond", color="silver"),
                )
            )
        else:
            ax.scatter(
                asset_sigma,
                asset_mu,
                s=30,
                color="k",
                label="assets",
            )
            if show_tickers:
                for i, label in enumerate(ef.tickers):
                    ax.annotate(label, (asset_sigma[i], asset_mu[i]))
    return ax


def plot_efficient_frontier(
    opt,
    ef_param="return",
    ef_param_range=None,
    points=100,
    ax=None,
    show_assets=True,
    show_tickers=False,
    interactive=False,
    **kwargs,
):
    """
    Plot the efficient frontier based on either a CLA or EfficientFrontier object.

    :param opt: an instantiated optimizer object BEFORE optimising an objective
    :type opt: EfficientFrontier or CLA
    :param ef_param: [EfficientFrontier] whether to use a range over utility, risk, or return.
                     Defaults to "return".
    :type ef_param: str, one of {"utility", "risk", "return"}.
    :param ef_param_range: the range of parameter values for ef_param.
                           If None, automatically compute a range from min->max return.
    :type ef_param_range: np.array or list (recommended to use np.arange or np.linspace)
    :param points: number of points to plot, defaults to 100. This is overridden if
                   an `ef_param_range` is provided explicitly.
    :type points: int, optional
    :param show_assets: whether we should plot the asset risks/returns also, defaults to True
    :type show_assets: bool, optional
    :param show_tickers: whether we should annotate each asset with its ticker, defaults to False
    :type show_tickers: bool, optional
    :param interactive: Switch rendering engine between Plotly and Matplotlib
    :type show_tickers: bool, optional
    :param filename: name of the file to save to, defaults to None (doesn't save)
    :type filename: str, optional
    :param showfig: whether to plt.show() the figure, defaults to False
    :type showfig: bool, optional
    :return: matplotlib axis
    :rtype: matplotlib.axes object
    """
    plt = _import_matplotlib()

    if interactive:
        go, _ = _get_plotly()
        ax = go.Figure()
    else:
        ax = ax or plt.gca()

    if isinstance(opt, CLA):
        ax = _plot_cla(
            opt,
            points,
            ax=ax,
            show_assets=show_assets,
            show_tickers=show_tickers,
            interactive=interactive,
        )
    elif isinstance(opt, EfficientFrontier):
        if ef_param_range is None:
            ef_param_range = _ef_default_returns_range(opt, points)

        ax = _plot_ef(
            opt,
            ef_param,
            ef_param_range,
            ax=ax,
            show_assets=show_assets,
            show_tickers=show_tickers,
            interactive=interactive,
        )
    else:
        raise NotImplementedError("Please pass EfficientFrontier or CLA object")

    if interactive:
        ax.update_layout(
            xaxis_title="Volatility",
            yaxis_title="Return",
        )
    else:
        ax.legend()
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Return")

        _plot_io(**kwargs)
    return ax


def plot_weights(weights, ax=None, **kwargs):
    """
    Plot the portfolio weights as a horizontal bar chart

    :param weights: the weights outputted by any PyPortfolioOpt optimizer
    :type weights: {ticker: weight} dict
    :param ax: ax to plot to, optional
    :type ax: matplotlib.axes
    :return: matplotlib axis
    :rtype: matplotlib.axes
    """
    plt = _import_matplotlib()

    ax = ax or plt.gca()

    desc = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in desc]
    vals = [i[1] for i in desc]

    y_pos = np.arange(len(labels))

    ax.barh(y_pos, vals)
    ax.set_xlabel("Weight")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    _plot_io(**kwargs)
    return ax


def plot_simulation_results(
    sim_df,
    initial_value=1_000_000,
    percentiles=(5, 25, 50, 75, 95),
    sample_trials=30,
    ax=None,
    figsize=(10, 6),
    showfig=False,
    inflation_rate: float | None = None,
    inflation_series=None,
    trading_periods_per_year: int = 252,
    **kwargs,
):
    """Plot Monte Carlo simulation results.

    The input `sim_df` is expected to be a DataFrame where each column is a
    trial and rows are time-ordered normalized values (start=100). This
    function scales values to `initial_value` and plots median and percentile
    bands, optionally overlaying a sample of individual trials.

    :param sim_df: DataFrame with trials as columns and normalized values (start=100)
    :type sim_df: pd.DataFrame
    :param initial_value: starting portfolio value to scale normalized series
    :type initial_value: float, optional
    :param percentiles: tuple/list of percentiles to compute and plot (must include median 50)
    :type percentiles: iterable of ints, optional
    :param sample_trials: number of individual trial paths to overlay (random sample)
    :type sample_trials: int, optional
    :param ax: matplotlib axis to plot to, optional
    :type ax: matplotlib.axes, optional
    :param figsize: figure size passed to matplotlib if ax not provided
    :type figsize: tuple, optional
    :param showfig: whether to call plt.show(); passed to _plot_io via kwargs
    :type showfig: bool, optional
    :return: matplotlib axis
    :rtype: matplotlib.axes
    """
    plt = _import_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Validate sim_df
    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd is None or not hasattr(sim_df, "columns"):
        raise TypeError("sim_df must be a pandas DataFrame with trials as columns")

    # Scale normalized series (start=100) to monetary values
    vals = sim_df / 100.0 * float(initial_value)

    # Apply inflation adjustment if requested. Two options:
    # - inflation_series: pandas Series (same index as sim_df) giving CPI or price index
    # - inflation_rate: annual inflation rate (e.g. 0.02 for 2%) applied continuously per period
    inflation_applied = False
    if inflation_series is not None:
        try:
            import pandas as pd

            # Align and compute deflator (normalize to 1 at first observation)
            inf = pd.Series(inflation_series)
            inf = inf.reindex(vals.index)
            if inf.isnull().any():
                # try to align by position if index mismatch
                inf = pd.Series(inflation_series).reset_index(drop=True)
                inf.index = vals.index
            deflator = inf / float(inf.iloc[0])
            vals = vals.divide(deflator.values, axis=0)
            inflation_applied = True
        except Exception:
            # if something fails, ignore inflation adjustment
            inflation_applied = False
    elif inflation_rate is not None:
        # inflation_rate is annual; convert to per-period compounding
        # t is number of periods since start
        t = np.arange(len(vals))
        per_period = (1.0 + float(inflation_rate)) ** (t / float(trading_periods_per_year))
        vals = vals / per_period[:, None]
        inflation_applied = True

    # Compute percentiles
    pct = vals.quantile([p / 100.0 for p in percentiles], axis=1)

    # Ensure median present at 0.5
    if 0.5 not in pct.index:
        pct.loc[0.5] = vals.median(axis=1)
        pct = pct.sort_index()

    # Plot percentile fills (outer to inner) and collect legend handles
    sorted_pcts = sorted(percentiles)
    # use a bold/classic colormap for bands (Tableau/tab10) for stronger colors
    cmap = plt.get_cmap("tab10")
    band_handles = []
    band_labels = []
    n_bands = len(sorted_pcts) // 2
    # sample the colormap evenly across available band slots
    for i in range(n_bands):
        low = sorted_pcts[i]
        high = sorted_pcts[-(i + 1)]
        frac = float(i) / max(1, n_bands - 1) if n_bands > 1 else 0.0
        color = cmap(frac)
        ax.fill_between(
            vals.index,
            pct.loc[low / 100.0],
            pct.loc[high / 100.0],
            color=color,
            alpha=0.35,
        )
        from matplotlib.patches import Patch

        band_handles.append(Patch(facecolor=color, alpha=0.35))
        band_labels.append(f"{low} - {high} pct band")

    # Plot median
    median = pct.loc[0.5]
    from matplotlib.lines import Line2D

    # use a strong black for median to contrast clearly with bold bands
    median_line, = ax.plot(vals.index, median, color="#000000", linewidth=2)

    # Overlay sample trials
    sample_handle = None
    if sample_trials and sample_trials > 0:
        ncols = vals.shape[1]
        sample = vals.sample(n=min(sample_trials, ncols), axis=1)
        for col in sample.columns:
            ax.plot(vals.index, sample[col], lw=0.8, color="#444444", alpha=0.6)
        sample_handle = Line2D([0], [0], color="#444444", lw=1, alpha=0.6)

    title = "Monte Carlo Simulation Results"
    if inflation_applied:
        title += " (inflation-adjusted)"
    ax.set_title(title)
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Portfolio Value (real terms)" if inflation_applied else "Portfolio Value")

    # Build legend: bands (outer->inner), median, sample trials
    handles = []
    labels = []
    # add band handles in same order
    handles.extend(band_handles[::-1])
    labels.extend(band_labels[::-1])
    handles.append(median_line)
    labels.append("Median")
    if sample_handle is not None:
        handles.append(sample_handle)
        labels.append(f"Sample trials (n={min(sample_trials, vals.shape[1])})")

    ax.legend(handles=handles, labels=labels)

    # Delegate save/show behavior
    kwargs.setdefault("showfig", showfig)
    _plot_io(**kwargs)

    return ax
