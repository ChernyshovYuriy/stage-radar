"""
Market Stage Analyzer
=====================
Detects current stock market state by sector:
  - Whether each sector is trending Up / Down / Neutral
  - Which Weinstein Stage (1–4) each sector is in

Stage Definitions (Stan Weinstein Stage Analysis):
  Stage 1 – Basing / Accumulation : Price is near or below a flat MA150
  Stage 2 – Advancing / Uptrend   : Price is above a rising MA150
  Stage 3 – Topping / Distribution: Price is near or above a flattening/rolling-over MA150
  Stage 4 – Declining / Downtrend : Price is below a declining MA150

Requirements:
    pip install yfinance pandas numpy tabulate

Usage:
    from market_stage_analyzer import MarketAnalyzer

    tickers = ["AAPL", "MSFT", "XOM", "JPM", ...]
    analyzer = MarketAnalyzer(tickers)
    report   = analyzer.run()
    print(report.summary())
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SHORT_MA = 50  # days – fast moving average
LONG_MA = 150  # days – Weinstein's 30-week proxy
SLOPE_WIN = 20  # days used to measure MA slope
RSI_PERIOD = 14
HISTORY = "2y"  # yfinance period string

# Slope thresholds (% change of MA over SLOPE_WIN trading days)
RISING_THRESHOLD = 1.5
DECLINING_THRESHOLD = -1.5

# Performance windows (trading days)
PERF_WINDOWS = {"1W": 5, "1M": 21, "3M": 63}


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TickerResult:
    symbol: str
    sector: str
    stage: int  # 1-4
    stage_label: str
    price: float
    ma_50: float
    ma_150: float
    ma_150_slope: float  # % change
    rsi: float
    perf_1w: float  # % return
    perf_1m: float
    perf_3m: float
    price_vs_ma150: float  # % deviation
    pct_from_52w_high: float
    is_up: bool  # net bias: True = bullish
    error: Optional[str] = None


@dataclass
class SectorResult:
    sector: str
    stage: int  # median/modal stage across constituents
    stage_label: str
    trend: str  # "Up" | "Down" | "Neutral"
    avg_rsi: float
    avg_perf_1w: float
    avg_perf_1m: float
    avg_perf_3m: float
    pct_stage2: float  # % of tickers in Stage 2
    pct_stage4: float  # % of tickers in Stage 4
    tickers: List[TickerResult] = field(default_factory=list)


@dataclass
class MarketReport:
    sectors: Dict[str, SectorResult]
    overall_trend: str
    bull_sectors: List[str]
    bear_sectors: List[str]
    timestamp: str

    # ── pretty printing ────────────────────────────────────────────────────
    def summary(self, show_tickers: bool = False) -> str:
        try:
            from tabulate import tabulate
            _tabulate = tabulate
        except ImportError:
            _tabulate = _simple_table

        lines: List[str] = []
        lines.append("=" * 72)
        lines.append(f"  MARKET STAGE REPORT  —  {self.timestamp}")
        lines.append(f"  Overall Market Trend: {self.overall_trend}")
        lines.append("=" * 72)

        # ── sector table ──────────────────────────────────────────────────
        sector_rows = []
        for name, s in sorted(self.sectors.items()):
            sector_rows.append([
                name,
                f"Stage {s.stage} – {s.stage_label}",
                s.trend,
                f"{s.avg_rsi:.1f}",
                _fmt_pct(s.avg_perf_1w),
                _fmt_pct(s.avg_perf_1m),
                _fmt_pct(s.avg_perf_3m),
                f"{s.pct_stage2:.0f}% / {s.pct_stage4:.0f}%",
            ])

        lines.append("")
        lines.append(_tabulate(
            sector_rows,
            headers=["Sector", "Stage", "Trend", "RSI",
                     "1W %", "1M %", "3M %", "S2% / S4%"],
            tablefmt="rounded_outline" if _tabulate is not _simple_table else "simple",
        ))

        # ── bull / bear summary ───────────────────────────────────────────
        lines.append("")
        if self.bull_sectors:
            lines.append(f"  🟢 Bullish sectors : {', '.join(self.bull_sectors)}")
        if self.bear_sectors:
            lines.append(f"  🔴 Bearish sectors : {', '.join(self.bear_sectors)}")

        # ── optional ticker detail ─────────────────────────────────────────
        if show_tickers:
            for name, s in sorted(self.sectors.items()):
                lines.append("")
                lines.append(f"  ── {name} ──")
                t_rows = []
                for t in sorted(s.tickers, key=lambda x: x.stage):
                    t_rows.append([
                        t.symbol,
                        f"Stage {t.stage}",
                        _fmt_pct(t.perf_1w),
                        _fmt_pct(t.perf_1m),
                        _fmt_pct(t.perf_3m),
                        f"{t.rsi:.1f}",
                        _fmt_pct(t.ma_150_slope, suffix=" slope"),
                    ])
                lines.append(_tabulate(
                    t_rows,
                    headers=["Ticker", "Stage", "1W", "1M", "3M", "RSI", "MA150 slope"],
                    tablefmt="simple",
                ))

        lines.append("")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Core engine
# ──────────────────────────────────────────────────────────────────────────────
class MarketAnalyzer:
    """
    Fetch OHLCV data and sector info for a list of tickers,
    then classify each ticker and aggregate per sector.

    Parameters
    ----------
    tickers : list[str]
        E.g. ["AAPL", "MSFT", "XOM", "JPM", ...]
    period  : str
        yfinance period string (default "2y").  Needs ≥ LONG_MA trading days.
    """

    def __init__(self, tickers: List[str], period: str = HISTORY):
        self.tickers = [t.upper().strip() for t in tickers]
        self.period = period

    # ── public entry point ─────────────────────────────────────────────────
    def run(self) -> MarketReport:
        """Download data, analyse every ticker, aggregate by sector."""
        print(f"[MarketAnalyzer] Fetching data for {len(self.tickers)} tickers…")
        prices_df, info_map = self._fetch_data()

        ticker_results: List[TickerResult] = []
        for sym in self.tickers:
            result = self._analyse_ticker(sym, prices_df, info_map)
            ticker_results.append(result)
            status = f"Stage {result.stage}" if not result.error else f"ERROR: {result.error}"
            print(f"  {sym:<8} {result.sector:<30} {status}")

        sector_results = self._aggregate_sectors(ticker_results)
        return self._build_report(sector_results)

    # ── data fetching ──────────────────────────────────────────────────────
    def _fetch_data(self) -> Tuple[pd.DataFrame, Dict[str, dict]]:
        """Return (Close price DataFrame, info dict keyed by ticker)."""
        import yfinance as yf

        # Bulk download for efficiency
        raw = yf.download(
            self.tickers,
            period=self.period,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )

        # Normalise: extract Close prices into a clean DataFrame
        if len(self.tickers) == 1:
            prices_df = raw[["Close"]].rename(columns={"Close": self.tickers[0]})
        else:
            try:
                prices_df = raw["Close"]
            except KeyError:
                # fallback for single-level column
                prices_df = raw.xs("Close", level=1, axis=1)

        # Fetch sector info per ticker (one by one — small overhead)
        info_map: Dict[str, dict] = {}
        for sym in self.tickers:
            try:
                info_map[sym] = yf.Ticker(sym).info
            except Exception as e:
                info_map[sym] = {"sector": "Unknown", "longName": sym}

        return prices_df, info_map

    # ── single-ticker analysis ─────────────────────────────────────────────
    def _analyse_ticker(
            self,
            symbol: str,
            prices_df: pd.DataFrame,
            info_map: Dict[str, dict],
    ) -> TickerResult:

        sector = info_map.get(symbol, {}).get("sector") or "Unknown"

        # Guard: ticker missing from download
        if symbol not in prices_df.columns:
            return TickerResult(
                symbol=symbol, sector=sector,
                stage=0, stage_label="No data",
                price=0, ma_50=0, ma_150=0,
                ma_150_slope=0, rsi=50,
                perf_1w=0, perf_1m=0, perf_3m=0,
                price_vs_ma150=0, pct_from_52w_high=0,
                is_up=False, error="Not in download"
            )

        prices = prices_df[symbol].dropna()

        if len(prices) < LONG_MA + SLOPE_WIN:
            return TickerResult(
                symbol=symbol, sector=sector,
                stage=0, stage_label="Insufficient data",
                price=0, ma_50=0, ma_150=0,
                ma_150_slope=0, rsi=50,
                perf_1w=0, perf_1m=0, perf_3m=0,
                price_vs_ma150=0, pct_from_52w_high=0,
                is_up=False, error="Insufficient history"
            )

        try:
            stage, label, metrics = _compute_stage(prices)
        except Exception as exc:
            return TickerResult(
                symbol=symbol, sector=sector,
                stage=0, stage_label="Calc error",
                price=0, ma_50=0, ma_150=0,
                ma_150_slope=0, rsi=50,
                perf_1w=0, perf_1m=0, perf_3m=0,
                price_vs_ma150=0, pct_from_52w_high=0,
                is_up=False, error=str(exc)
            )

        # Performance returns
        perfs = {}
        for key, days in PERF_WINDOWS.items():
            if len(prices) > days:
                perfs[key] = (prices.iloc[-1] / prices.iloc[-days - 1] - 1) * 100
            else:
                perfs[key] = 0.0

        is_up = stage in (2,) or (stage == 3 and perfs["1M"] > 0)

        return TickerResult(
            symbol=symbol, sector=sector,
            stage=stage, stage_label=label,
            price=metrics["price"],
            ma_50=metrics["ma_50"],
            ma_150=metrics["ma_150"],
            ma_150_slope=metrics["ma_150_slope"],
            rsi=metrics["rsi"],
            perf_1w=perfs["1W"],
            perf_1m=perfs["1M"],
            perf_3m=perfs["3M"],
            price_vs_ma150=metrics["price_vs_ma150"],
            pct_from_52w_high=metrics["pct_from_52w_high"],
            is_up=is_up,
        )

    # ── sector aggregation ─────────────────────────────────────────────────
    @staticmethod
    def _aggregate_sectors(
            ticker_results: List[TickerResult],
    ) -> Dict[str, SectorResult]:

        sectors: Dict[str, List[TickerResult]] = {}
        for r in ticker_results:
            sectors.setdefault(r.sector, []).append(r)

        out: Dict[str, SectorResult] = {}
        for sec_name, members in sectors.items():
            valid = [m for m in members if m.error is None and m.stage > 0]
            if not valid:
                continue

            stages = [m.stage for m in valid]
            n = len(stages)
            modal = max(set(stages), key=stages.count)
            pct_s2 = stages.count(2) / n * 100
            pct_s4 = stages.count(4) / n * 100

            avg_rsi = np.mean([m.rsi for m in valid])
            avg_1w = np.mean([m.perf_1w for m in valid])
            avg_1m = np.mean([m.perf_1m for m in valid])
            avg_3m = np.mean([m.perf_3m for m in valid])

            # Trend signal: RSI above/below 50 + 1-month performance
            if avg_rsi > 55 and avg_1m > 0:
                trend = "Up"
            elif avg_rsi < 45 and avg_1m < 0:
                trend = "Down"
            elif avg_1m > 1.5:
                trend = "Up"
            elif avg_1m < -1.5:
                trend = "Down"
            else:
                trend = "Neutral"

            out[sec_name] = SectorResult(
                sector=sec_name,
                stage=modal,
                stage_label=_STAGE_LABELS[modal],
                trend=trend,
                avg_rsi=avg_rsi,
                avg_perf_1w=avg_1w,
                avg_perf_1m=avg_1m,
                avg_perf_3m=avg_3m,
                pct_stage2=pct_s2,
                pct_stage4=pct_s4,
                tickers=valid,
            )

        return out

    # ── report builder ─────────────────────────────────────────────────────
    @staticmethod
    def _build_report(sectors: Dict[str, SectorResult]) -> MarketReport:
        from datetime import datetime

        bull = [s for s, v in sectors.items() if v.trend == "Up"]
        bear = [s for s, v in sectors.items() if v.trend == "Down"]

        if len(bull) > len(bear) * 1.5:
            overall = "Bullish"
        elif len(bear) > len(bull) * 1.5:
            overall = "Bearish"
        else:
            overall = "Mixed / Neutral"

        return MarketReport(
            sectors=sectors,
            overall_trend=overall,
            bull_sectors=sorted(bull),
            bear_sectors=sorted(bear),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Technical helpers
# ──────────────────────────────────────────────────────────────────────────────
_STAGE_LABELS = {
    1: "Basing",
    2: "Advancing",
    3: "Topping",
    4: "Declining",
}


def _compute_stage(
        prices: pd.Series,
) -> Tuple[int, str, dict]:
    """
    Core Weinstein-style stage detection.

    Stage logic (multi-signal, not just price vs MA):

      Stage 2 – Advancing : price > MA150, MA150 rising, momentum intact
      Stage 3 – Topping   : price > MA150, BUT MA150 slope decelerating or RSI falling
      Stage 4 – Declining : price < MA150, MA150 declining, momentum weak
      Stage 1 – Basing    : price < MA150, BUT decline losing steam OR price recovering

    Returns (stage_int, stage_label, metrics_dict)
    """
    min_needed = LONG_MA + max(SLOPE_WIN, 40)  # 150 + 40 = 190 bars
    if len(prices) < min_needed:
        raise ValueError(
            f"Need ≥ {min_needed} trading days of history; got {len(prices)}."
        )

    ma_50 = prices.rolling(SHORT_MA).mean()
    ma_150 = prices.rolling(LONG_MA).mean()

    cur_price = float(prices.iloc[-1])
    cur_ma50 = float(ma_50.iloc[-1])
    cur_ma150 = float(ma_150.iloc[-1])

    # MA slopes at different horizons (used to detect acceleration / deceleration)
    ma150_slope_10d = _slope(ma_150, 10)  # ~2-week (recent)
    ma150_slope_20d = _slope(ma_150, 20)  # ~1-month (medium)
    ma150_slope_40d = _slope(ma_150, 40)  # ~2-month (older baseline)

    ma50_slope = _slope(ma_50, 10)

    # Primary slope = 20-day (headline metric shown in reports)
    ma150_slope = ma150_slope_20d

    # Slope deceleration: negative value means MA is losing upward momentum
    deceleration = ma150_slope_20d - ma150_slope_40d

    # Price deviation from MA150
    price_vs_ma150 = (cur_price - cur_ma150) / cur_ma150 * 100

    # RSI-14
    rsi = _rsi(prices, RSI_PERIOD)

    # Short-term price momentum (5-day return)
    price_mom_5d = (float(prices.iloc[-1]) / float(prices.iloc[-6]) - 1) * 100 \
        if len(prices) > 5 else 0.0

    # 52-week stats
    tail_252 = prices.tail(252)
    high_52w = float(tail_252.max())
    pct_off_high = (cur_price / high_52w - 1) * 100  # negative = below high

    # ── Stage classification ───────────────────────────────────────────────
    #
    # PRIMARY RULE: price vs MA150 × MA150 slope direction
    #
    #   price > MA150  +  slope rising  →  candidate Stage 2 or 3 (check deceleration)
    #   price > MA150  +  slope flat/dn →  Stage 3 (or early Stage 4)
    #   price < MA150  +  slope declining→ candidate Stage 4 or 1 (check recovery)
    #   price < MA150  +  slope flat/up →  Stage 1
    #
    # SECONDARY SIGNALS differentiate 2 vs 3 and 4 vs 1:
    #   Deceleration (slope_20d << slope_40d) with RSI < 60 → lean Stage 3
    #   Slope recovery  (slope_10d > slope_20d + buffer) + RSI > 40 → lean Stage 1

    if cur_price >= cur_ma150:
        if ma150_slope >= RISING_THRESHOLD:
            # Is the uptrend losing steam? Check deceleration + RSI
            slope_decelerating = deceleration < -2.5  # MA150 slowing fast
            rsi_fading = rsi < 58 and rsi < 65  # momentum waning
            price_below_ma50 = cur_price < cur_ma50  # near-term breakdown
            if (slope_decelerating and rsi_fading) or price_below_ma50:
                stage = 3  # Topping — momentum fading despite still above MA150
            else:
                stage = 2  # Advancing — healthy uptrend
        else:
            # MA150 flat or declining while price still above it → classic Stage 3
            stage = 3
            # If MA50 is also declining and price slipped under it → early Stage 4
            if ma50_slope < DECLINING_THRESHOLD and cur_price < cur_ma50:
                stage = 4

    else:  # cur_price < cur_ma150
        if ma150_slope <= DECLINING_THRESHOLD:
            # Is the downtrend losing steam? Check if recent slope is recovering
            recent_vs_medium = ma150_slope_10d - ma150_slope_20d  # positive = improving
            slope_recovering = recent_vs_medium > 0.8
            rsi_stabilising = rsi > 42
            if slope_recovering and rsi_stabilising:
                stage = 1  # Basing — downtrend slowing, accumulation possible
            else:
                stage = 4  # Declining — still in downtrend
        else:
            # MA150 flat or gently rising, price just pulled back under it → Stage 1
            stage = 1

    label = _STAGE_LABELS[stage]

    metrics = {
        "price": cur_price,
        "ma_50": cur_ma50,
        "ma_150": cur_ma150,
        "ma_150_slope": ma150_slope,
        "ma_150_slope_10d": ma150_slope_10d,
        "ma_150_slope_40d": ma150_slope_40d,
        "deceleration": deceleration,
        "rsi": rsi,
        "price_vs_ma150": price_vs_ma150,
        "pct_from_52w_high": pct_off_high,
    }

    return stage, label, metrics


def _slope(series: pd.Series, window: int) -> float:
    """Percentage change of series over `window` trailing bars."""
    clean = series.dropna()
    if len(clean) < window:
        return 0.0
    past = float(clean.iloc[-window])
    present = float(clean.iloc[-1])
    if past == 0:
        return 0.0
    return (present - past) / past * 100


def _rsi(prices: pd.Series, period: int = 14) -> float:
    """Wilder's RSI — returns 50.0 when price has no variance."""
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    val = float(rsi.iloc[-1])
    return val if np.isfinite(val) else 50.0


# ──────────────────────────────────────────────────────────────────────────────
# Fallback table renderer (used when tabulate not installed)
# ──────────────────────────────────────────────────────────────────────────────
def _simple_table(rows, headers, **kwargs) -> str:
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)
    hdr = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    lines = [hdr, sep]
    for row in rows:
        lines.append("  ".join(str(v).ljust(w) for v, w in zip(row, widths)))
    return "\n".join(lines)


def _fmt_pct(v: float, suffix: str = "") -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.2f}%{suffix}"


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: build a DataFrame export
# ──────────────────────────────────────────────────────────────────────────────
def report_to_dataframe(report: MarketReport) -> pd.DataFrame:
    """Flatten all ticker results into a tidy DataFrame for further analysis."""
    rows = []
    for s in report.sectors.values():
        for t in s.tickers:
            rows.append({
                "sector": t.sector,
                "ticker": t.symbol,
                "stage": t.stage,
                "stage_label": t.stage_label,
                "trend": s.trend,
                "price": t.price,
                "ma_50": t.ma_50,
                "ma_150": t.ma_150,
                "ma_150_slope": t.ma_150_slope,
                "rsi": t.rsi,
                "perf_1w": t.perf_1w,
                "perf_1m": t.perf_1m,
                "perf_3m": t.perf_3m,
                "price_vs_ma150": t.price_vs_ma150,
                "pct_from_52w_high": t.pct_from_52w_high,
                "is_up": t.is_up,
            })
    return pd.DataFrame(rows).sort_values(["sector", "stage", "ticker"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# CLI usage
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("../res/can_tickers", "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    analyzer = MarketAnalyzer(tickers)
    report = analyzer.run()

    # ── Print sector summary ───────────────────────────────────────────────
    print(report.summary(show_tickers=True))

    # ── Export to CSV ──────────────────────────────────────────────────────
    df = report_to_dataframe(report)
    df.to_csv("market_stages.csv", index=False)
    print("Detailed results saved to market_stages.csv")
