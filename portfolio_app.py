import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import yfinance as yf  # type: ignore
except ImportError:
    yf = None


DEFAULT_TICKERS = {
    "Brent": "BZ=F",
    "HenryHub": "NG=F",
    "TTF": "TTF=F",
    "JKM": "JKM=F",
}


def load_yahoo_series(ticker: str, start: str, end: str) -> pd.Series:
    if yf is None:
        raise RuntimeError("yfinance is not installed. Please install yfinance or upload data.")

    df = pd.DataFrame()
    dl = getattr(yf, "download", None)
    if callable(dl):
        try:
            df = dl(ticker, start=start, end=end, progress=False)
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        tk = yf.Ticker(ticker)
        df = tk.history(start=start, end=end, auto_adjust=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if price_col not in df.columns:
        raise ValueError(f"Ticker {ticker} missing Close/Adj Close column.")

    return df[price_col].rename(ticker)


def parse_uploaded_csv(upload: st.runtime.uploaded_file_manager.UploadedFile) -> pd.Series:
    df = pd.read_csv(upload)
    date_col_candidates = [c for c in df.columns if "date" in c.lower()]
    if not date_col_candidates:
        raise ValueError("CSV must contain a date-like column.")
    date_col = date_col_candidates[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    price_cols = [c for c in df.columns if c.lower() not in ("date", "datetime")]
    if not price_cols:
        raise ValueError("CSV must contain a price column besides date.")
    return df[price_cols[0]].rename(upload.name)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")


def compute_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    vol = returns.rolling(window=window).std() * np.sqrt(252)
    return vol.dropna(how="any")


def minimum_variance_weights(cov_matrix: np.ndarray, fully_invested: bool = True) -> np.ndarray:
    n = cov_matrix.shape[0]
    ones = np.ones(n)
    regularizer = 1e-8 * np.eye(n)
    inv_cov = np.linalg.pinv(cov_matrix + regularizer)

    if fully_invested:
        w = inv_cov @ ones
        denom = ones.T @ inv_cov @ ones
        return w / denom if denom != 0 else np.repeat(1 / n, n)

    raw = inv_cov @ ones
    centered = raw - np.mean(raw)
    denom = np.sum(np.abs(centered))
    return centered / denom if denom != 0 else np.zeros(n)


def optimize_hedge_overlay(
    cov_matrix: np.ndarray,
    base_weights: np.ndarray,
    lambda_reg: float,
    net_constraint: float = 0.0,
) -> np.ndarray:
    """
    Solve min_x (x+b)'Σ(x+b) + λx'x, s.t. 1'x = net_constraint.
    x is hedge overlay, b is existing book weights.
    """
    n = len(base_weights)
    ones = np.ones(n)
    A = cov_matrix + lambda_reg * np.eye(n)
    A_inv = np.linalg.pinv(A + 1e-10 * np.eye(n))

    rhs = cov_matrix @ base_weights
    c1 = ones.T @ A_inv @ ones
    c2 = ones.T @ A_inv @ rhs
    nu = -(net_constraint + c2) / c1 if c1 != 0 else 0.0
    x = -A_inv @ (rhs + nu * ones)
    return x


def simulate_dynamic_hedge(
    returns: pd.DataFrame,
    base_weights: np.ndarray,
    lookback: int,
    rebalance_days: int,
    lambda_reg: float,
    net_constraint: float,
) -> pd.DataFrame:
    records = []
    cols = returns.columns
    for t in range(lookback, len(returns), rebalance_days):
        cov = returns.iloc[t - lookback : t].cov().values
        overlay = optimize_hedge_overlay(cov, base_weights, lambda_reg, net_constraint)
        next_t = min(t + rebalance_days, len(returns))
        period_ret = returns.iloc[t:next_t]
        base_pnl = period_ret.values @ base_weights
        hedged_pnl = period_ret.values @ (base_weights + overlay)
        for i, dt in enumerate(period_ret.index):
            records.append(
                {
                    "Date": dt,
                    "Base": base_pnl[i],
                    "Hedged": hedged_pnl[i],
                    **{f"overlay_{c}": overlay[j] for j, c in enumerate(cols)},
                }
            )
    out = pd.DataFrame(records).set_index("Date") if records else pd.DataFrame()
    return out


def app():
    st.title("能源组合风险管理与套保优化（BRT / HH / TTF / JKM）")
    st.caption("支持‘最小方差’与‘最小套保后净风险’两种逻辑，并可做滚动动态再平衡。")

    st.sidebar.header("1) 时间与波动参数")
    today = datetime.today()
    default_start = today - timedelta(days=365 * 3)
    start_date = st.sidebar.date_input("开始日期", default_start)
    end_date = st.sidebar.date_input("结束日期", today)
    if start_date >= end_date:
        st.error("开始日期必须早于结束日期。")
        return

    window = st.sidebar.slider("滚动波动率窗口(天)", 10, 180, 30, 5)

    st.sidebar.header("2) 数据源")
    use_yahoo_for_all = st.sidebar.checkbox("尝试使用yfinance抓取四个品种", value=True)
    custom_tickers = {
        k: st.sidebar.text_input(f"{k} ticker", v) for k, v in DEFAULT_TICKERS.items()
    }

    st.sidebar.markdown("若TTF/JKM抓取失败，可上传CSV（含Date和价格列）")
    ttf_upload = st.sidebar.file_uploader("TTF CSV", type=["csv"])
    jkm_upload = st.sidebar.file_uploader("JKM CSV", type=["csv"])

    st.sidebar.header("3) 公司现货/期货盘位映射(可正可负)")
    base_pos = {
        "Brent": st.sidebar.number_input("Brent基准暴露", value=1.0, step=0.1),
        "HenryHub": st.sidebar.number_input("HH基准暴露", value=0.0, step=0.1),
        "TTF": st.sidebar.number_input("TTF基准暴露", value=0.0, step=0.1),
        "JKM": st.sidebar.number_input("JKM基准暴露", value=0.0, step=0.1),
    }

    st.sidebar.header("4) 套保优化参数")
    objective = st.sidebar.radio(
        "目标函数",
        ["最小方差组合(不考虑当前盘位)", "最小化当前盘位套保后的净波动"],
        index=1,
    )
    fully_invested = st.sidebar.checkbox("最小方差模式下：权重和=1", value=True)
    lambda_reg = st.sidebar.slider("套保调整惩罚 λ", 0.0, 5.0, 0.5, 0.1)
    net_constraint = st.sidebar.slider("套保overlay净敞口约束(权重和)", -1.0, 1.0, 0.0, 0.1)

    st.sidebar.header("5) 动态再平衡")
    dyn_lookback = st.sidebar.slider("协方差回看窗口(天)", 20, 252, 90, 5)
    rebalance_days = st.sidebar.slider("再平衡频率(天)", 5, 60, 20, 5)

    prices_dict, errors = {}, []
    for asset in ["Brent", "HenryHub", "TTF", "JKM"]:
        loaded = False
        if use_yahoo_for_all and yf is not None:
            try:
                prices_dict[asset] = load_yahoo_series(
                    custom_tickers[asset], start_date.isoformat(), end_date.isoformat()
                )
                loaded = True
            except Exception as e:
                errors.append(f"{asset} yfinance失败: {e}")

        if not loaded and asset == "TTF" and ttf_upload is not None:
            try:
                prices_dict[asset] = parse_uploaded_csv(ttf_upload)
                loaded = True
            except Exception as e:
                errors.append(f"TTF CSV失败: {e}")

        if not loaded and asset == "JKM" and jkm_upload is not None:
            try:
                prices_dict[asset] = parse_uploaded_csv(jkm_upload)
                loaded = True
            except Exception as e:
                errors.append(f"JKM CSV失败: {e}")

    if errors:
        st.sidebar.warning("\n".join(errors))

    if len(prices_dict) < 2:
        st.warning("可用价格序列不足，至少需要两个品种。")
        return

    df_prices = pd.concat(prices_dict.values(), axis=1, join="inner")
    df_prices.columns = list(prices_dict.keys())
    df_prices = df_prices[(df_prices.index >= pd.to_datetime(start_date)) & (df_prices.index <= pd.to_datetime(end_date))]
    df_returns = compute_returns(df_prices)
    if df_returns.empty:
        st.warning("收益率数据为空，请调整时间范围或数据源。")
        return

    st.subheader("价格走势")
    st.line_chart(df_prices)

    st.subheader("相关性（收益率 + 波动率）")
    st.write("收益率相关")
    st.dataframe(df_returns.corr())
    df_vol = compute_volatility(df_returns, window)
    st.write("滚动波动率相关")
    st.dataframe(df_vol.corr() if not df_vol.empty else pd.DataFrame())

    cov = df_returns.cov().values
    cols = df_returns.columns.tolist()

    st.subheader("优化结果")
    if objective.startswith("最小方差组合"):
        w = minimum_variance_weights(cov, fully_invested=fully_invested)
        final_w = pd.Series(w, index=cols, name="weight")
        port_vol = float(np.sqrt(w.T @ cov @ w) * np.sqrt(252))
        st.write("最优权重")
        st.dataframe(final_w.to_frame())
        st.metric("年化波动率", f"{port_vol:.2%}")
    else:
        base_vec = np.array([base_pos.get(c, 0.0) for c in cols], dtype=float)
        if np.allclose(base_vec, 0):
            st.warning("当前盘位全为0，无法体现套保效果。请输入至少一个非零暴露。")
            return

        overlay = optimize_hedge_overlay(cov, base_vec, lambda_reg, net_constraint)
        hedged = base_vec + overlay

        base_vol = float(np.sqrt(base_vec.T @ cov @ base_vec) * np.sqrt(252))
        hedged_vol = float(np.sqrt(hedged.T @ cov @ hedged) * np.sqrt(252))
        eff = 1 - (hedged_vol**2 / base_vol**2) if base_vol > 0 else np.nan

        res = pd.DataFrame(
            {
                "BaseExposure": base_vec,
                "Overlay": overlay,
                "PostHedgeExposure": hedged,
            },
            index=cols,
        )
        st.dataframe(res)
        c1, c2, c3 = st.columns(3)
        c1.metric("套保前年化波动", f"{base_vol:.2%}")
        c2.metric("套保后年化波动", f"{hedged_vol:.2%}")
        c3.metric("方差口径套保效率", f"{eff:.2%}")

        if len(df_returns) > dyn_lookback + rebalance_days:
            sim = simulate_dynamic_hedge(
                df_returns, base_vec, dyn_lookback, rebalance_days, lambda_reg, net_constraint
            )
            if not sim.empty:
                st.subheader("动态再平衡表现")
                nav = (1 + sim[["Base", "Hedged"]]).cumprod()
                st.line_chart(nav)
                sim_eff = 1 - (sim["Hedged"].var() / sim["Base"].var()) if sim["Base"].var() != 0 else np.nan
                st.metric("动态样本内套保效率", f"{sim_eff:.2%}")
                overlay_cols = [c for c in sim.columns if c.startswith("overlay_")]
                st.write("Overlay路径")
                st.line_chart(sim[overlay_cols])


if __name__ == "__main__":
    app()
