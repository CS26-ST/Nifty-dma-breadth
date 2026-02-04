import time
import pandas as pd
import yfinance as yf
import os
from datetime import date

def fetch_nifty500_symbols():
    df = pd.read_csv("nifty500_symbols.csv")
    symbols = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .tolist()
    )
    # Convert to Yahoo tickers
    return sorted(list(set(s + ".NS" for s in symbols if s)))


def download_close_prices(symbols, period="2y", chunk_size=50, pause=1):
    all_close = []

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        df = yf.download(
            chunk,
            period=period,
            auto_adjust=True,
            threads=True,
            progress=False
        )

        if df is not None and not df.empty and "Close" in df.columns:
            all_close.append(df["Close"].copy())

        time.sleep(pause)

    if not all_close:
        return pd.DataFrame()

    close_prices = pd.concat(all_close, axis=1)
    close_prices = close_prices.dropna(axis=1, how="all")
    close_prices = close_prices.sort_index()

    return close_prices


def main():
    symbols = fetch_nifty500_symbols()
    close_prices = download_close_prices(symbols)

    if close_prices.empty:
        raise RuntimeError("No price data downloaded")

    # Folder: data/nifty500
    base_dir = "data/nifty500"
    os.makedirs(base_dir, exist_ok=True)

    # Date-stamped filename
    today = date.today().isoformat()
    filename = f"{base_dir}/nifty500_close_prices_1y_{today}.xlsx"

    # Save Excel
    close_prices.to_excel(filename)

    print("Saved:", filename)
    print("Rows (trading days):", close_prices.shape[0])
    print("Stocks:", close_prices.shape[1])


if __name__ == "__main__":
    main()
