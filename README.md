# NG-BRt Portfolio App

这是一个基于 Streamlit 的油气组合风险管理工具，覆盖 `Brent / Henry Hub / TTF / JKM` 四个基准，支持：

- 收益率相关性与滚动波动率相关性分析；
- 最小方差组合；
- 基于当前盘位暴露的套保 overlay 优化；
- 动态再平衡与套保效率展示。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run portfolio_app.py
```

## Streamlit 部署说明

如果通过 Streamlit Community Cloud 部署，通常**不需要单独前端工程文件**（如 React/Vue 项目）。

最小必需项是：

1. `portfolio_app.py`（应用入口）；
2. `requirements.txt`（依赖清单）。

可选增强：

- `.streamlit/config.toml`：主题、端口、日志等配置；
- `packages.txt`：系统级依赖（本项目当前不需要）。

## 数据说明

- 默认通过 `yfinance` 抓取数据（可在侧边栏调整 ticker）；
- 若 TTF / JKM 抓取失败，可上传包含 `Date` 与价格列的 CSV。


## 数据口径提醒

- HH 腿在 Yahoo 默认使用 `NG=F`（NYMEX Natural Gas）作为风险管理代理；不建议混用 `HH=F` look-alik。
- JKM 建议优先上传本地 CSV（date, price）。
- 支持三类优化：最小方差（无约束/带约束）、风险平价、因子覆盖+风险预算。
