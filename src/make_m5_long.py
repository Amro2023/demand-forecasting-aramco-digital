import pandas as pd
import os

IN_PATH = "/Users/amroosman/Downloads/sales_train_validation.csv"
OUT_PATH = "data/raw/m5_demand.csv"

def main():
    os.makedirs("data/raw", exist_ok=True)

    df = pd.read_csv(IN_PATH)

    # M5 columns: id, item_id, dept_id, cat_id, store_id, state_id, d_1 ... d_1913
    day_cols = [c for c in df.columns if c.startswith("d_")]

    long_df = df.melt(
        id_vars=["store_id", "item_id"],
        value_vars=day_cols,
        var_name="day",
        value_name="demand"
    )

    # Convert "d_1" -> 1 (you can map to real dates later; for now use synthetic timeline)
    long_df["day_num"] = long_df["day"].str.replace("d_", "", regex=False).astype(int)

    # Create a synthetic date index starting at 2011-01-29 (common M5 start date)
    start = pd.Timestamp("2011-01-29")
    long_df["date"] = start + pd.to_timedelta(long_df["day_num"] - 1, unit="D")

    out = long_df[["date", "demand", "store_id", "item_id"]]
    out.to_csv(OUT_PATH, index=False)
    print(f"✅ Wrote: {OUT_PATH} rows={len(out)} cols={list(out.columns)}")

if __name__ == "__main__":
    main()
