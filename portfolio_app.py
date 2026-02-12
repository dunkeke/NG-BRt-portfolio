import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import yfinance as yf  # type: ignore
except ImportError:
    yf = None

ASSETS = ["Brent", "HH", "TTF", "JKM"]
DEFAULT_TICKERS = {
    "Brent": "BZ=F",
    "HH": "NG=F",  # use NG=F as HH proxy on Yahoo
    "TTF": "TTF=F",
    "JKM": "",
}


DEFAULT_TICKERS = {
    "Brent": "BZ=F",
    "HenryHub": "NG=F",
    "TTF": "TTF=F",
    "JKM": "JKM=F",
}


def load_yahoo_series(ticker: str, start: str, end: str) -> pd.Series:
    if not ticker:
        raise ValueError("Empty ticker.")
    if yf is None:
        raise RuntimeError("yfinance is not installed.")

    df = pd.DataFrame()
    dl = getattr(yf, "download", None)
    if callable(dl):
        try:
            df = dl(ticker, start=start, end=end, progress=False)
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if price_col not in df.columns:
        raise ValueError(f"Ticker {ticker} missing Close/Adj Close.")

    out = df[price_col].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.rename(ticker)


def parse_uploaded_csv(upload: st.runtime.uploaded_file_manager.UploadedFile) -> pd.Series:
    df = pd.read_csv(upload)
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col is None:
        raise ValueError("CSV needs date column.")
    price_col = next((c for c in df.columns if c != date_col), None)
    if price_col is None:
        raise ValueError("CSV needs price column.")
    df[date_col] = pd.to_datetime(df[date_col])
    s = df.set_index(date_col).sort_index()[price_col]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.astype(float)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")


def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    x = returns.values
    n, k = x.shape
    if n < 2:
        return np.cov(x.T)
    cov = np.cov(x[: min(20, n)].T)
    for i in range(n):
        v = x[i].reshape(-1, 1)
        cov = lam * cov + (1 - lam) * (v @ v.T)
    return cov


def shrink_cov(sample_cov: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    diag = np.diag(np.diag(sample_cov))
    return (1 - alpha) * sample_cov + alpha * diag


def estimate_covariance(returns: pd.DataFrame, method: str, ewma_lambda: float, shrink_alpha: float) -> np.ndarray:
    sample = returns.cov().values
    if method == "Sample":
        cov = sample
    elif method == "EWMA":
        cov = ewma_cov(returns, lam=ewma_lambda)
    else:
        cov = shrink_cov(sample, alpha=shrink_alpha)
    return cov + 1e-8 * np.eye(cov.shape[0])


def project_to_box_simplex(v: np.ndarray, lb: np.ndarray, ub: np.ndarray, target_sum: float = 1.0) -> np.ndarray:
    w = np.clip(v.copy(), lb, ub)
    for _ in range(200):
        diff = target_sum - w.sum()
        if abs(diff) < 1e-9:
            break
        free = (w > lb + 1e-12) & (w < ub - 1e-12)
        if not np.any(free):
            break
        w[free] += diff / free.sum()
        w = np.clip(w, lb, ub)
    return w


def solve_minvar(cov: np.ndarray, lb: np.ndarray | None = None, ub: np.ndarray | None = None) -> np.ndarray:
    n = cov.shape[0]
    ones = np.ones(n)
    inv = np.linalg.pinv(cov)
    w = inv @ ones
    denom = ones @ inv @ ones
    w = w / denom if denom != 0 else np.repeat(1 / n, n)
    if lb is None or ub is None:
        return w
    return project_to_box_simplex(w, lb, ub, 1.0)


def risk_contributions(cov: np.ndarray, w: np.ndarray) -> np.ndarray:
    mrc = cov @ w
    port_var = max(float(w @ mrc), 1e-12)
    return w * mrc / port_var


def solve_risk_parity(cov: np.ndarray, lb: np.ndarray, ub: np.ndarray, steps: int = 1200, lr: float = 0.03) -> np.ndarray:
    n = cov.shape[0]
    w = np.repeat(1 / n, n)
    target = np.repeat(1 / n, n)
    for _ in range(steps):
        rc = risk_contributions(cov, w)
        grad = rc - target
        w = w * np.exp(-lr * grad)
        w = project_to_box_simplex(w, lb, ub, 1.0)
    return w


def pca_factor_loadings(returns: pd.DataFrame, n_factors: int = 3) -> np.ndarray:
    corr = returns.corr().values
    vals, vecs = np.linalg.eigh(corr)
    idx = np.argsort(vals)[::-1]
    vecs = vecs[:, idx]
    return vecs[:, :n_factors]


def solve_factor_cover(
    cov: np.ndarray,
    B: np.ndarray,
    b_star: np.ndarray,
    gamma: float,
    lam_factor: float,
    lb: np.ndarray,
    ub: np.ndarray,
    steps: int = 2000,
    lr: float = 0.02,
) -> np.ndarray:
    n = cov.shape[0]
    w = np.repeat(1 / n, n)
    L = lam_factor * np.eye(len(b_star))
    for _ in range(steps):
        diff = B.T @ w - b_star
        grad = 2 * (B @ (L @ diff)) + 2 * gamma * (cov @ w)
        w = w - lr * grad
        w = project_to_box_simplex(w, lb, ub, 1.0)
    return w


def rolling_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    return returns.rolling(window).std() * np.sqrt(252)


def app():
    st.title("能源组合风险管理（Brent / HH(NG) / TTF / JKM）")
    st.info("HH 在 Yahoo 默认采用 NG=F（避免 HH=F look-alik 混用）。")

    st.sidebar.header("参数")
    end = datetime.today()
    start = end - timedelta(days=365 * 3)
    start_date = st.sidebar.date_input("开始", start)
    end_date = st.sidebar.date_input("结束", end)
    if start_date >= end_date:
        st.error("开始日期需早于结束日期")
        return

    cov_method = st.sidebar.selectbox("协方差估计", ["Sample", "EWMA", "Shrinkage"], index=1)
    ewma_lambda = st.sidebar.slider("EWMA λ", 0.80, 0.99, 0.94, 0.01)
    shrink_alpha = st.sidebar.slider("Shrinkage α", 0.0, 0.8, 0.2, 0.05)
    vol_window = st.sidebar.slider("滚动波动窗口", 10, 180, 30, 5)

    st.sidebar.subheader("数据源")
    tickers = {}
    for a in ASSETS:
        tickers[a] = st.sidebar.text_input(f"{a} ticker", DEFAULT_TICKERS[a])
    upload_jkm = st.sidebar.file_uploader("JKM CSV（优先上传）", type=["csv"])

    st.sidebar.subheader("优化模式")
    mode = st.sidebar.radio("模式", ["MinVar(无约束)", "MinVar(带约束)", "RiskParity", "FactorCover+RiskBudget"])

    st.sidebar.subheader("权重约束")
    lb = {}
    ub = {}
    default_lb = {"Brent": 0.0, "HH": 0.1, "TTF": 0.05, "JKM": 0.1}
    default_ub = {"Brent": 0.6, "HH": 0.8, "TTF": 0.8, "JKM": 0.8}
    for a in ASSETS:
        lb[a] = st.sidebar.number_input(f"{a} 下限", min_value=0.0, max_value=1.0, value=default_lb[a], step=0.01)
        ub[a] = st.sidebar.number_input(f"{a} 上限", min_value=0.0, max_value=1.0, value=default_ub[a], step=0.01)

    st.sidebar.subheader("因子覆盖目标（仅FactorCover模式）")
    gamma = st.sidebar.slider("波动惩罚 γ", 0.0, 5.0, 1.0, 0.1)
    lam_factor = st.sidebar.slider("因子贴合权重 λ_f", 0.0, 10.0, 3.0, 0.1)
    b1 = st.sidebar.slider("目标因子1(Global Energy)", -2.0, 2.0, 0.5, 0.1)
    b2 = st.sidebar.slider("目标因子2(Atlantic-Pacific Gas)", -2.0, 2.0, 0.0, 0.1)
    b3 = st.sidebar.slider("目标因子3(Oil-specific)", -2.0, 2.0, 0.2, 0.1)

    prices = {}
    errs = []
    for a in ASSETS:
        try:
            if a == "JKM" and upload_jkm is not None:
                prices[a] = parse_uploaded_csv(upload_jkm)
            else:
                if not tickers[a]:
                    raise ValueError("请填写ticker或上传CSV")
                prices[a] = load_yahoo_series(tickers[a], start_date.isoformat(), end_date.isoformat())
        except Exception as e:
            errs.append(f"{a} 数据失败: {e}")

    if errs:
        st.warning("\n".join(errs))

    if len(prices) < 2:
        st.error("有效品种不足，无法优化。")
        return

    df = pd.concat(prices.values(), axis=1).sort_index()
    df.columns = list(prices.keys())
    df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
    df = df.ffill(limit=3).dropna(how="any")
    rets = compute_returns(df)
    if rets.empty:
        st.error("收益率为空，请检查数据。")
        return

    st.subheader("价格与相关性")
    st.line_chart(df)
    st.write("收益率相关")
    st.dataframe(rets.corr())
    st.write("滚动波动率相关")
    st.dataframe(rolling_vol(rets, vol_window).dropna().corr())

    cov = estimate_covariance(rets, cov_method, ewma_lambda, shrink_alpha)
    assets = rets.columns.tolist()
    lb_arr = np.array([lb.get(a, 0.0) for a in assets])
    ub_arr = np.array([ub.get(a, 1.0) for a in assets])

    if lb_arr.sum() > 1 + 1e-9 or ub_arr.sum() < 1 - 1e-9:
        st.error("约束不可行：下限和>1 或 上限和<1。")
        return

    if mode == "MinVar(无约束)":
        w = solve_minvar(cov)
    elif mode == "MinVar(带约束)":
        w = solve_minvar(cov, lb_arr, ub_arr)
    elif mode == "RiskParity":
        w = solve_risk_parity(cov, lb_arr, ub_arr)
    else:
        B = pca_factor_loadings(rets, n_factors=3)
        b_star = np.array([b1, b2, b3])
        w = solve_factor_cover(cov, B, b_star, gamma, lam_factor, lb_arr, ub_arr)

    port_vol = float(np.sqrt(w.T @ cov @ w) * np.sqrt(252))
    out = pd.Series(w, index=assets, name="weight")

    st.subheader("优化结果")
    st.dataframe(out.to_frame())
    st.metric("组合年化波动", f"{port_vol:.2%}")

    if mode in ["RiskParity", "FactorCover+RiskBudget"]:
        rc = pd.Series(risk_contributions(cov, w), index=assets, name="risk_contribution")
        st.write("风险贡献")
        st.dataframe(rc.to_frame())

    if mode == "FactorCover+RiskBudget":
        B = pca_factor_loadings(rets, n_factors=3)
        exposure = pd.Series(B.T @ w, index=["F1", "F2", "F3"], name="realized_factor_exposure")
        target = pd.Series([b1, b2, b3], index=["F1", "F2", "F3"], name="target_factor_exposure")
        st.write("因子暴露：目标 vs 实现")
        st.dataframe(pd.concat([target, exposure], axis=1))


if __name__ == "__main__":
    app()
