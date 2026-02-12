"""
Risk Management Portfolio App for Energy Markets
------------------------------------------------

This Streamlit application allows users to build and analyse a simple hedging
portfolio using major energy benchmarks: Brent crude oil (BRT), Henry Hub
natural gas (HH), the Dutch Title Transfer Facility (TTF) and the Japan‑Korea
Marker (JKM).  The app draws on academic and industry research showing that
correlations among gas benchmarks have tightened markedly since 2019 while
crude oil remains only weakly linked to natural gas markets【410373605677858†L290-L296】【435331636587632†L25-L54】.  By
combining these instruments in a single portfolio and optimising the weight
allocation, users can approximate hedge ratios that minimise portfolio
volatility.

Features
========

* **Data acquisition** – Daily price data for Brent (ticker ``"BZ=F"``) and
  Henry Hub natural gas (ticker ``"NG=F"``) are fetched using the ``yfinance``
  library.  Users may optionally upload their own comma‑separated value
  (CSV) files for TTF and JKM prices (with a ``Date`` column and a price
  column); if no upload is provided the app will generate synthetic TTF and
  JKM series based on the Henry Hub price and a user‑specified basis.
* **Return & volatility analysis** – Simple daily returns and 30‑day rolling
  annualised volatility are computed for each series.  The app displays
  correlation matrices for both returns and volatilities to help users
  understand co‑movement among the instruments【972169242848767†L84-L118】【313876969850617†L398-L404】.
* **Portfolio optimisation** – The covariance matrix of asset returns is
  used to derive minimum‑variance weights using closed‑form matrix algebra.
  Users can choose whether the weights must sum to zero (a self‑financing
  hedge) or to one (a fully invested portfolio).  The optimal weights and
  the resulting expected volatility are reported.
* **Interactive controls** – Start and end dates, rolling window length and
  synthetic basis spreads can all be adjusted via the sidebar.  The app
  re‑calculates metrics automatically when inputs change.

Usage
-----

Before running this application you must install the required dependencies:

```bash
pip install streamlit pandas numpy yfinance matplotlib
```

Then launch the app with:

```bash
streamlit run portfolio_app.py
```

"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import yfinance as yf  # type: ignore
except ImportError:
    # Provide a friendly error if yfinance is not available.  Users should
    # install it in their own environment.  The app will still run but
    # synthetic data must be provided for all series.
    yf = None


def load_yahoo_series(ticker: str, start: str, end: str) -> pd.Series:
    """Download a daily price series from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        The Yahoo Finance ticker (e.g. ``"BZ=F"`` for Brent futures).
    start : str
        Start date in ISO format.
    end : str
        End date in ISO format.

    Returns
    -------
    pd.Series
        A series indexed by date containing the adjusted close prices.
    """
    if yf is None:
        raise RuntimeError(
            "yfinance is not installed. Please install yfinance or upload your own data.")
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}.")
    # Use Adjusted Close if available, otherwise Close
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    return df[price_col].rename(ticker)


def parse_uploaded_csv(upload: st.uploaded_file_manager.UploadedFile) -> pd.Series:
    """Parse an uploaded CSV file into a price series.

    The CSV file must contain at least two columns: a date column named
    ``Date`` (or similar) and a price column.  The function will try to
    infer the price column automatically if only two columns are present.

    Parameters
    ----------
    upload : UploadedFile
        The uploaded file object from Streamlit.

    Returns
    -------
    pd.Series
        A series of prices indexed by date.
    """
    df = pd.read_csv(upload)
    # Identify the date column
    date_col_candidates = [c for c in df.columns if 'date' in c.lower()]
    if not date_col_candidates:
        raise ValueError("CSV must contain a date column (e.g. 'Date').")
    date_col = date_col_candidates[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    # Identify price column
    price_cols = [c for c in df.columns if c.lower() not in ('date', 'datetime')]
    if not price_cols:
        raise ValueError("CSV must contain a price column besides the date column.")
    price_col = price_cols[0]
    return df[price_col].rename(price_col)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns for each price series."""
    return prices.pct_change().dropna(how='all')


def compute_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling annualised volatility using a specified window.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of returns.
    window : int
        Rolling window length in days.

    Returns
    -------
    pd.DataFrame
        Rolling volatility for each series.
    """
    vol = returns.rolling(window=window).std() * np.sqrt(252)
    return vol.dropna(how='all')


def minimum_variance_weights(cov_matrix: np.ndarray, fully_invested: bool = True) -> np.ndarray:
    """Compute minimum‑variance portfolio weights.

    This function computes the vector of weights that minimise the portfolio
    variance under either of two constraints:

    * **fully_invested=True**: The weights must sum to 1 (i.e. the portfolio is
      fully invested).  Negative weights imply short positions.
    * **fully_invested=False**: The weights must sum to 0, representing a
      self‑financing hedge portfolio with no net capital outlay.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix of asset returns.
    fully_invested : bool, optional
        Whether the portfolio must be fully invested.  If ``False`` the sum of
        weights will equal zero.  Default is ``True``.

    Returns
    -------
    np.ndarray
        Array of optimal weights.  The weights are normalised according to
        the specified constraint.  If optimisation is not possible (e.g. the
        covariance matrix is singular), an equal‑weighted vector is returned.
    """
    try:
        inv_cov = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # fall back to identity if matrix is singular
        inv_cov = np.eye(cov_matrix.shape[0])
    n = cov_matrix.shape[0]
    ones = np.ones(n)
    if fully_invested:
        # weights sum to one
        w = inv_cov @ ones
        w /= ones.T @ inv_cov @ ones
    else:
        # weights sum to zero (hedged).  Solve with Lagrange multipliers:
        # minimise w^T Σ w subject to 1^T w = 0.  The solution is
        # w = inv_cov @ (I - (1·1^T)/(1^T inv_cov 1)) · u / sum(abs(u))
        # Using u = unit vector (ones) gives a hedge balanced across assets.
        u = ones
        A = inv_cov @ u
        # project to the subspace sum(w)=0
        adjustment = (ones.T @ A) / (ones.T @ inv_cov @ ones)
        w = A - adjustment * inv_cov @ ones
        # normalise to unit leverage
        if np.sum(np.abs(w)) != 0:
            w /= np.sum(np.abs(w))
    return w


def app():
    st.title("Energy Portfolio Risk Management")
    st.write(
        """
        Build and analyse a hedging portfolio using major energy benchmarks.
        Adjust parameters in the sidebar to explore different scenarios.  The
        correlations used in this tool are inspired by industry and academic
        research showing that European and Asian gas benchmarks have become
        highly correlated since 2019【410373605677858†L290-L296】【322238738332884†L96-L131】, whereas the link between
        crude oil and natural gas remains weak【313876969850617†L398-L404】.  Use the weight optimiser to
        derive minimum‑variance allocations under either fully invested or
        self‑financing (zero net exposure) constraints.
        """
    )

    # Sidebar inputs
    st.sidebar.header("Settings")
    today = datetime.today()
    default_start = today - timedelta(days=365)
    start_date = st.sidebar.date_input("Start date", default_start)
    end_date = st.sidebar.date_input("End date", today)
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date")

    # Rolling window for volatility
    window = st.sidebar.slider("Volatility window (days)", 5, 120, 30, step=5)

    # Synthetic basis for TTF and JKM if not uploaded
    synthetic_basis_ttf = st.sidebar.number_input(
        "Synthetic TTF premium over Henry Hub (USD/mmBtu)", value=10.0, step=0.5)
    synthetic_basis_jkm = st.sidebar.number_input(
        "Synthetic JKM premium over TTF (USD/mmBtu)", value=1.0, step=0.1)

    # Uploaders for TTF and JKM
    ttf_upload = st.sidebar.file_uploader(
        "Upload TTF price CSV (optional)", type=['csv'])
    jkm_upload = st.sidebar.file_uploader(
        "Upload JKM price CSV (optional)", type=['csv'])

    # Constraint choice
    fully_invested = st.sidebar.radio(
        "Portfolio constraint",
        options=["Fully invested (weights sum to 1)", "Self‑financing hedge (weights sum to 0)"],
        index=0
    )
    sum_to_one = fully_invested.startswith("Fully")

    # Data retrieval
    prices_dict = {}
    error_messages = []
    # Load BRT (Brent) and HH (Henry Hub) from Yahoo if available
    if yf is not None:
        try:
            prices_dict['Brent'] = load_yahoo_series(
                'BZ=F', start_date.isoformat(), end_date.isoformat())
        except Exception as e:
            error_messages.append(f"Brent: {e}")
        try:
            prices_dict['HenryHub'] = load_yahoo_series(
                'NG=F', start_date.isoformat(), end_date.isoformat())
        except Exception as e:
            error_messages.append(f"Henry Hub: {e}")
    else:
        error_messages.append("yfinance is not installed; upload CSVs for all assets.")

    # Load TTF
    if ttf_upload is not None:
        try:
            series = parse_uploaded_csv(ttf_upload)
            prices_dict['TTF'] = series[(series.index >= pd.to_datetime(start_date)) &
                                        (series.index <= pd.to_datetime(end_date))]
        except Exception as e:
            error_messages.append(f"TTF upload: {e}")
    else:
        # synthetic TTF based on Henry Hub if available
        if 'HenryHub' in prices_dict:
            prices_dict['TTF'] = prices_dict['HenryHub'] + synthetic_basis_ttf
        else:
            error_messages.append(
                "TTF: Please upload a CSV or ensure Henry Hub data is available for synthetic TTF.")

    # Load JKM
    if jkm_upload is not None:
        try:
            series = parse_uploaded_csv(jkm_upload)
            prices_dict['JKM'] = series[(series.index >= pd.to_datetime(start_date)) &
                                        (series.index <= pd.to_datetime(end_date))]
        except Exception as e:
            error_messages.append(f"JKM upload: {e}")
    else:
        if 'TTF' in prices_dict:
            prices_dict['JKM'] = prices_dict['TTF'] + synthetic_basis_jkm
        else:
            error_messages.append(
                "JKM: Please upload a CSV or ensure TTF data is available for synthetic JKM.")

    # Display any data loading errors
    if error_messages:
        st.sidebar.warning("\n".join(error_messages))

    # Create a combined DataFrame if we have at least two series
    if len(prices_dict) >= 2:
        # Align indexes
        df_prices = pd.concat(prices_dict.values(), axis=1, join='inner')
        df_prices.columns = list(prices_dict.keys())
        st.subheader("Price Series")
        st.line_chart(df_prices)

        # Compute returns and volatility
        df_returns = compute_returns(df_prices)
        df_vol = compute_volatility(df_returns, window)

        # Show correlation matrices
        st.subheader("Correlation Matrices")
        st.write("**Returns correlation:**")
        st.dataframe(df_returns.corr())
        st.write("**Volatility correlation:**")
        if not df_vol.empty:
            st.dataframe(df_vol.corr())
        else:
            st.write("Not enough data to compute rolling volatility.")

        # Optimise portfolio
        try:
            cov_matrix = df_returns.cov().values
            weights = minimum_variance_weights(cov_matrix, fully_invested=sum_to_one)
            weight_df = pd.Series(weights, index=df_returns.columns, name='Weight')
            # Compute portfolio volatility
            port_var = weights.T @ cov_matrix @ weights
            port_vol = np.sqrt(port_var) * np.sqrt(252)
            st.subheader("Optimal Portfolio Weights")
            st.write(weight_df.to_frame())
            st.write(
                f"Expected annualised portfolio volatility: **{port_vol:.2%}**")
        except Exception as e:
            st.write(f"Unable to compute optimal weights: {e}")
    else:
        st.write(
            "Not enough data available to perform analysis. Please provide at least two price series.")


if __name__ == '__main__':
    app()