import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True, help="e.g. nifty500, nifty750")
    ap.add_argument("--close_csv", required=True, help="e.g. data/nifty500/close.csv")
    ap.add_argument("--fast", type=int, default=50)
    ap.add_argument("--slow", type=int, default=200)
    ap.add_argument("--min_coverage", type=float, default=0.80)
    args = ap.parse_args()

    os.makedirs("output", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    df = pd.read_csv(args.close_csv)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    df.columns = [str(c).replace("NSE:", "").replace(".NS", "").strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")

    coverage = df.notna().mean(axis=0)
    df = df[coverage[coverage >= args.min_coverage].index]

    dma_fast = df.rolling(args.fast, min_periods=args.fast).mean()
    dma_slow = df.rolling(args.slow, min_periods=args.slow).mean()

    above_fast = (df > dma_fast).where(dma_fast.notna())
    above_slow = (df > dma_slow).where(dma_slow.notna())

    valid_fast = above_fast.notna().sum(axis=1).astype(float)
    valid_fast = valid_fast.replace(0, pd.NA)
    valid_slow = above_slow.notna().sum(axis=1).astype(float)
    valid_slow = valid_slow.replace(0, pd.NA)

    breadth_fast = (above_fast.sum(axis=1) / valid_fast) * 100
    breadth_slow = (above_slow.sum(axis=1) / valid_slow) * 100

    out = pd.DataFrame({
        f"pct_above_{args.fast}dma": breadth_fast,
        f"pct_above_{args.slow}dma": breadth_slow,
        f"valid_stocks_{args.fast}dma": valid_fast,
        f"valid_stocks_{args.slow}dma": valid_slow,
    }).dropna()

    out_csv = os.path.join("output", f"{args.universe}_dma_breadth.csv")
    out_img = os.path.join("images", f"{args.universe}_dma_breadth.png")

    out.to_csv(out_csv)

    plt.figure(figsize=(14, 6))
    plt.plot(out.index, out[f"pct_above_{args.fast}dma"], label=f"% Above {args.fast}DMA")
    plt.plot(out.index, out[f"pct_above_{args.slow}dma"], label=f"% Above {args.slow}DMA")
    for lvl in [20, 50, 80]:
        plt.axhline(lvl, linestyle="--", linewidth=1)

    plt.ylim(0, 100)
    plt.title(f"{args.universe.upper()} DMA Breadth")
    plt.xlabel("Date")
    plt.ylabel("Breadth (%)")
    plt.grid(True, linewidth=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_img, dpi=200)

    print("Saved:", out_csv)
    print("Saved:", out_img)


if __name__ == "__main__":
    main()
