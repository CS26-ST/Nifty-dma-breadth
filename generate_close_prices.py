import argparse
import os
import time
from datetime import date
from typing import List

import pandas as pd
import yfinance as yf


def read_symbols(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    symbols = df[col].astype(str).str.strip()

    # Normalize: remove NSE: and .NS then add .NS
    cleaned = (
        symbols.str.replace("NSE:", "", regex=False)
        .str.replace(".NS", "", regex=False)
        .str.strip()
    )
    return sorted({f"{s}.NS" for s in cleaned if s and s != "nan"})


def extract_close(downloaded: pd.DataFrame) -> pd.DataFrame:
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()

    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" in downloaded.columns.get_level_values(0):
            close = downloaded["Close"].copy()
        else:
            close_cols = [c for c in downloaded.columns if len(c) == 2 and c[1] == "Close"]
            if not close_cols:
                return pd.DataFrame()
            close = downloaded[close_cols].copy()
            close.columns = [c[0] for c in close.columns]
    else:
        if "Close" not in downloaded.columns:
            return pd.DataFrame()
        close = downloaded[["Close"]].copy()
        close.columns = ["SINGLE_TICKER"]

    return close


def download_close_prices(
    yf_symbols: List[str],
    period: str,
    chunk_size: int,
    pause: float,
    max_retries: int,
    auto_adjust: bool,
) -> pd.DataFrame:
    all_close = []

    for i in range(0, len(yf_symbols), chunk_size):
        chunk = yf_symbols[i:i + chunk_size]

        for attempt in range(max_retries + 1):
            try:
                downloaded = yf.download(
                    tickers=chunk,
                    period=period,
                    interval="1d",
                    auto_adjust=auto_adjust,
                    threads=True,
                    progress=False,
                )
                close = extract_close(downloaded)
                if not close.empty:
                    all_close.append(close)
                break
            except Exception as e:
                if attempt == max_retries:
                    print(f"[ERROR] chunk {i//chunk_size + 1} failed: {e}")
                else:
                    wait = pause * (attempt + 2)
                    print(f"[WARN] retry {attempt+1}/{max_retries} wait {wait}s: {e}")
                    time.sleep(wait)

        time.sleep(pause)

    if not all_close:
        return pd.DataFrame()

    close_df = pd.concat(all_close, axis=1)
    close_df = close_df.dropna(axis=1, how="all")
    close_df = close_df.loc[:, ~close_df.columns.duplicated()]
    close_df = close_df.sort_index()

    # Clean columns: remove .NS
    close_df.columns = [str(c).replace(".NS", "") for c in close_df.columns]
    return close_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True, help="Universe name e.g. nifty500, nifty750")
    ap.add_argument("--symbols_csv", required=True, help="Path to universe CSV")
    ap.add_argument("--period", default="2y", help="yfinance period, e.g. 2y, 3y")
    ap.add_argument("--chunk_size", type=int, default=50)
    ap.add_argument("--pause", type=float, default=1.0)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--auto_adjust", action="store_true", help="Use adjusted prices")
    args = ap.parse_args()

    out_dir = os.path.join("data", args.universe)
    os.makedirs(out_dir, exist_ok=True)

    yf_symbols = read_symbols(args.symbols_csv)
    print(f"Universe: {args.universe} | Symbols: {len(yf_symbols)} | Period: {args.period}")

    close_df = download_close_prices(
        yf_symbols=yf_symbols,
        period=args.period,
        chunk_size=args.chunk_size,
        pause=args.pause,
        max_retries=args.max_retries,
        auto_adjust=args.auto_adjust,
    )

    if close_df.empty:
        raise RuntimeError("No close prices downloaded.")

    out_csv = os.path.join(out_dir, "close.csv")
    close_df.to_csv(out_csv, index=True)

    print(f"Saved: {out_csv} | rows={close_df.shape[0]} | cols={close_df.shape[1]}")


if __name__ == "__main__":
    main()
