import argparse
import os
import time
from typing import List

import pandas as pd
import yfinance as yf


def read_symbols(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    col = "Symbol" if "Symbol" in df.columns else df.columns[0]

    symbols = (
        df[col]
        .astype(str)
        .str.replace("NSE:", "", regex=False)
        .str.replace(".NS", "", regex=False)
        .str.strip()
    )

    return sorted({f"{s}.NS" for s in symbols if s and s != "nan"})


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
        close.columns = ["SINGLE"]

    return close


def download_close_prices(
    symbols: List[str],
    period: str,
    chunk_size: int,
    pause: float,
):
    frames = []

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        print(f"Fetching chunk {i//chunk_size + 1}")

        data = yf.download(
            tickers=chunk,
            period=period,
            interval="1d",
            auto_adjust=True,
            threads=True,
            progress=False,
        )

        close = extract_close(data)
        if not close.empty:
            frames.append(close)

        time.sleep(pause)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1)
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_index()

    df.columns = [c.replace(".NS", "") for c in df.columns]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", required=True, help="e.g. nifty500")
    parser.add_argument("--symbols_csv", required=True, help="e.g. universes/nifty500.csv")
    parser.add_argument("--period", default="2y", help="yfinance period")
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--pause", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(f"data/{args.universe}", exist_ok=True)

    symbols = read_symbols(args.symbols_csv)
    print(f"Universe: {args.universe}, Symbols: {len(symbols)}")

    close_df = download_close_prices(
        symbols=symbols,
        period=args.period,
        chunk_size=args.chunk_size,
        pause=args.pause,
    )

    if close_df.empty:
        raise RuntimeError("No close prices downloaded")

    out_path = f"data/{args.universe}/close.csv"
    close_df.to_csv(out_path)

    print(f"Saved close prices → {out_path}")


if __name__ == "__main__":
    main()
