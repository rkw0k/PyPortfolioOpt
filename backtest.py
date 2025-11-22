import logging
import pandas as pd
import numpy as np

# create module logger (no default output unless configured by caller)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def monte_carlo_historical(historical_data, num_trials, simulation_years, verbose=False):
    """Run Monte Carlo using historical rolling windows.

    - Reduce prints and use the logger instead.
    - Choose first trading day of each year as possible start dates,
      and fallback to any trading day with enough remaining history.
    """
    # compute last allowable start date so sim fits in history
    # Use DateOffset(years=...) so we don't rely on Jan 1 specifically
    last_possible_start = historical_data.index.max() - pd.DateOffset(years=simulation_years)

    # pick first trading day of each year as canonical "Jan 1st" start
    first_trading_day_each_year = historical_data.groupby(historical_data.index.year).apply(
        lambda df: df.index.min()
    )
    # convert to an index
    first_trading_day_each_year = pd.DatetimeIndex(first_trading_day_each_year.values)

    # valid starts are first trading days on or before last_possible_start
    valid_start_dates = first_trading_day_each_year[first_trading_day_each_year <= last_possible_start]

    # fallback: if none of the "first trading day of year" candidates work,
    # use any trading day that still allows a full simulation window.
    if valid_start_dates.empty:
        # Use all trading days less-than-or-equal to last_possible_start
        candidate_starts = historical_data[historical_data.index <= last_possible_start].index
        if candidate_starts.empty:
            # more helpful message; include dates to guide resolution
            raise ValueError(
                "Historical data range is too short for the requested simulation_years. "
                f"Needed end >= {simulation_years} years before {historical_data.index.max().date()}. "
                "Either reduce simulation_years or provide longer historical_data."
            )
        valid_start_dates = candidate_starts

    if verbose:
        logger.info(f"Running Monte Carlo Simulation with {num_trials} trials...")
        logger.info(
            "Valid start range %s -> %s (using {len(valid_start_dates)} candidate starts)",
            valid_start_dates.min().date(),
            valid_start_dates.max().date(),
        )

    # run trials
    results = []
    for _ in range(num_trials):
        start_date = np.random.choice(valid_start_dates)
        end_date = start_date + pd.DateOffset(years=simulation_years)
        # ...existing code that runs trials, but replace prints below with logger.* and respect verbose...
        # e.g.:
        # if verbose: logger.debug("Trial %d / %d", i, num_trials)
        # ...existing code...