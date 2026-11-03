"""Turn clean sales lines into a weekly demand table per product.

Steps:
1. Flag bulk lines   - one-off wholesale orders far above what the item usually sells
                       per line. Kept, but in a separate column, so the modelling
                       step can choose its target.
2. Aggregate weekly  - weeks run Monday to Sunday; partial weeks at the start and
                       end of the data are dropped (they would look like a sales drop).
3. Trading days      - days per week with any sale at all, across all products. The shop
                       closes around Christmas and on bank holidays; that is known in
                       advance, so it is a legitimate feature for the future too.
4. Select products   - using ONLY the weeks before the test window, so the choice of
                       products does not peek at the period we will evaluate on:
                       enough history, still selling recently, then the top N by volume.
5. Fill the grid     - one row per product per week from its first sale to the end,
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
    test_weeks: int = 12  # the final weeks held out for evaluation; selection never sees them
    min_history_weeks: int = 52  # on sale at least this long before the test window
    recent_weeks: int = 12  # at least one sale in this many weeks before the test window
    top_n: int = 500


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


def first_test_week(last_week: pd.Timestamp, cfg: WeeklyConfig) -> pd.Timestamp:
    """Monday of the first test week."""
    return last_week - pd.Timedelta(weeks=cfg.test_weeks - 1)


def flag_bulk(sales: pd.DataFrame, cfg: WeeklyConfig, before: pd.Timestamp | None = None) -> pd.Series:
    """True for lines far above the item's typical line quantity.

    The typical line (median) is learned from lines before `before` when given,
    so the threshold does not depend on the test period.
    """
    learn = sales if before is None else sales[sales["invoice_date"] < before]
    median_line = learn.groupby("stock_code")["quantity"].median()
    # products first sold in the test period: fall back to all their lines
    median_line = median_line.reindex(sales["stock_code"].unique()).fillna(
        sales.groupby("stock_code")["quantity"].median()
    )
    threshold = sales["stock_code"].map(median_line) * cfg.bulk_factor
    return (sales["quantity"] > threshold) & (sales["quantity"] >= cfg.bulk_min_quantity)


def trading_days(sales: pd.DataFrame) -> pd.Series:
    """Days with at least one sale, per week, across all products."""
    days = sales["invoice_date"].dt.normalize()
    return days.groupby(week_start(sales["invoice_date"])).nunique().rename("trading_days")


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


def select_products(weekly: pd.DataFrame, cutoff: pd.Timestamp, cfg: WeeklyConfig) -> pd.Index:
    """Products chosen from the weeks before `cutoff` (the first test week) only."""
    seen = weekly[weekly["week"] < cutoff]
    first_sale = seen.groupby("stock_code")["week"].min()
    last_sale = seen.groupby("stock_code")["week"].max()

    long_enough = first_sale <= cutoff - pd.Timedelta(weeks=cfg.min_history_weeks)
    still_selling = last_sale >= cutoff - pd.Timedelta(weeks=cfg.recent_weeks)
    eligible = long_enough[long_enough & still_selling].index

    volume = (
        seen[seen["stock_code"].isin(eligible)]
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
    first, last = complete_weeks(sales["invoice_date"])
    cutoff = first_test_week(last, cfg)
    bulk = flag_bulk(sales, cfg, before=cutoff)

    weekly = aggregate(sales, bulk)
    weekly = weekly[(weekly["week"] >= first) & (weekly["week"] <= last)]
    products = select_products(weekly, cutoff, cfg)
    selected = fill_grid(weekly[weekly["stock_code"].isin(products)], last)

    days = trading_days(sales)
    selected = selected.merge(days, left_on="week", right_index=True, how="left")
    selected["trading_days"] = selected["trading_days"].fillna(0).astype("int64")  # shop closed

    all_weeks = pd.date_range(first, last, freq="W-MON")
    days = days.reindex(all_weeks, fill_value=0)
    in_range = (sales["invoice_date"] >= first) & (sales["invoice_date"] < last + pd.Timedelta(weeks=1))
    total_units = sales.loc[in_range, "quantity"].sum()
    report = {
        "weeks": f"{first.date()} to {last.date()} ({len(all_weeks)} weeks)",
        "test window": f"{cutoff.date()} to {last.date()} ({cfg.test_weeks} weeks)",
        "weeks with no sales at all": ", ".join(str(d.date()) for d in days[days == 0].index) or "none",
        "trading days per week": days.value_counts().sort_index().to_dict(),
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
