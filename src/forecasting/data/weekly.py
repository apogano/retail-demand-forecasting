"""Turn clean sales lines into a weekly demand table per product.

Steps:
1. Flag bulk lines   - one-off wholesale orders far above what the item usually sells
                       per line. Kept, but in a separate column, so the modelling
                       step can choose its target.
2. Aggregate weekly  - weeks run Monday to Sunday; partial weeks at the start and
                       end of the data are dropped (they would look like a sales drop).
3. Select products   - enough history to learn from, then the top N by volume.
4. Fill the grid     - one row per product per week from its first sale to the end,
                       with 0 for weeks without sales (no row is not the same as no data).
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from forecasting.data.clean import PROCESSED_PATH as SALES_PATH

WEEKLY_PATH = Path("data/processed/weekly.parquet")


@dataclass(frozen=True)
class WeeklyConfig:
    bulk_factor: float = 20.0  # a line is bulk if quantity > factor x the item's median line...
    bulk_min_quantity: int = 500  # ...and also above this absolute floor
    min_history_weeks: int = 52  # product must be on sale at least this long before the end
    top_n: int = 500


def flag_bulk(sales: pd.DataFrame, cfg: WeeklyConfig) -> pd.Series:
    """True for lines far above the item's typical line quantity."""
    median_line = sales.groupby("stock_code")["quantity"].transform("median")
    return (sales["quantity"] > cfg.bulk_factor * median_line) & (
        sales["quantity"] >= cfg.bulk_min_quantity
    )


def week_start(dates: pd.Series) -> pd.Series:
    """Monday 00:00 of the week each date falls in."""
    return dates.dt.to_period("W-SUN").dt.start_time


def complete_weeks(dates: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last week (Monday) fully covered by the data.

    Data starts on Tue 2009-12-01 and ends on Fri 2011-12-09: both of those
    weeks are partial and would show up as false drops in demand.
    """
    first_day, last_day = dates.min().normalize(), dates.max().normalize()
    first = week_start(pd.Series([first_day])).iloc[0]
    if first < first_day:
        first += pd.Timedelta(weeks=1)
    last = week_start(pd.Series([last_day])).iloc[0]
    if last + pd.Timedelta(days=6) > last_day:
        last -= pd.Timedelta(weeks=1)
    return first, last


def aggregate(sales: pd.DataFrame, bulk: pd.Series) -> pd.DataFrame:
    """One row per (product, week) that had sales."""
    df = sales.assign(
        week=week_start(sales["invoice_date"]),
        bulk_quantity=sales["quantity"].where(bulk, 0),
        revenue=sales["quantity"] * sales["price"],
    )
    weekly = df.groupby(["stock_code", "week"]).agg(
        quantity=("quantity", "sum"),
        bulk_quantity=("bulk_quantity", "sum"),
        revenue=("revenue", "sum"),
        n_lines=("invoice", "size"),
        n_customers=("customer_id", "nunique"),
    )
    weekly["avg_price"] = weekly["revenue"] / weekly["quantity"]
    return weekly.drop(columns="revenue").reset_index()


def select_products(weekly: pd.DataFrame, last_week: pd.Timestamp, cfg: WeeklyConfig) -> pd.Index:
    """Products with enough history, ranked by volume excluding bulk orders."""
    first_sale = weekly.groupby("stock_code")["week"].min()
    long_enough = first_sale[first_sale <= last_week - pd.Timedelta(weeks=cfg.min_history_weeks - 1)].index

    volume = (
        weekly[weekly["stock_code"].isin(long_enough)]
        .assign(base=lambda d: d["quantity"] - d["bulk_quantity"])
        .groupby("stock_code")["base"]
        .sum()
    )
    return volume.nlargest(cfg.top_n).index


def fill_grid(weekly: pd.DataFrame, last_week: pd.Timestamp) -> pd.DataFrame:
    """Every week from each product's first sale to the end; missing weeks get 0."""
    first_sale = weekly.groupby("stock_code")["week"].min()
    grid = pd.concat(
        pd.DataFrame({"stock_code": code, "week": pd.date_range(start, last_week, freq="W-MON")})
        for code, start in first_sale.items()
    )
    out = grid.merge(weekly, on=["stock_code", "week"], how="left")
    counts = ["quantity", "bulk_quantity", "n_lines", "n_customers"]
    out[counts] = out[counts].fillna(0).astype("int64")
    # avg_price stays NaN in weeks without sales: there was no price that week
    return out.sort_values(["stock_code", "week"]).reset_index(drop=True)


def build_weekly(sales: pd.DataFrame, cfg: WeeklyConfig | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or WeeklyConfig()
    bulk = flag_bulk(sales, cfg)
    first, last = complete_weeks(sales["invoice_date"])

    weekly = aggregate(sales, bulk)
    weekly = weekly[(weekly["week"] >= first) & (weekly["week"] <= last)]
    products = select_products(weekly, last, cfg)
    selected = fill_grid(weekly[weekly["stock_code"].isin(products)], last)

    in_range = (sales["invoice_date"] >= first) & (sales["invoice_date"] < last + pd.Timedelta(weeks=1))
    total_units = sales.loc[in_range, "quantity"].sum()
    report = {
        "weeks": f"{first.date()} to {last.date()} ({(last - first).days // 7 + 1} weeks)",
        "bulk lines": int(bulk.sum()),
        "bulk share of units": f"{sales.loc[bulk, 'quantity'].sum() / sales['quantity'].sum():.2%}",
        "products in data": int(sales["stock_code"].nunique()),
        "products selected": len(products),
        "selected share of units": f"{selected['quantity'].sum() / total_units:.2%}",
        "selected item-weeks": len(selected),
        "zero weeks among selected": f"{(selected['quantity'] == 0).mean():.2%}",
    }
    return selected, report


def main() -> None:
    sales = pd.read_parquet(SALES_PATH)
    weekly, report = build_weekly(sales)
    for key, value in report.items():
        print(f"{key:<28}{value}")
    WEEKLY_PATH.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(WEEKLY_PATH, index=False)
    print(f"\nSaved {len(weekly):,} rows to {WEEKLY_PATH}")


if __name__ == "__main__":
    main()
