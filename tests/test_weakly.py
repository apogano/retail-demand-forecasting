import pandas as pd

from forecasting.data.weekly import (
    WeeklyConfig,
    build_weekly,
    complete_weeks,
    fill_grid,
    flag_bulk,
    week_start,
)


def sale(code, qty, date, price=1.0, customer=100, invoice="1"):
    return {
        "invoice": invoice,
        "stock_code": code,
        "description": "ITEM",
        "quantity": qty,
        "invoice_date": pd.Timestamp(date),
        "price": price,
        "customer_id": customer,
        "country": "United Kingdom",
    }


def frame(*rows):
    df = pd.DataFrame(rows)
    df["customer_id"] = df["customer_id"].astype("Int64")
    return df


def test_weeks_start_on_monday():
    dates = pd.Series(pd.to_datetime(["2010-01-04 09:00", "2010-01-10 23:59", "2010-01-11 00:00"]))
    assert week_start(dates).dt.strftime("%Y-%m-%d").tolist() == ["2010-01-04", "2010-01-04", "2010-01-11"]


def test_partial_first_and_last_weeks_are_excluded():
    # Same span as the real data: Tue 2009-12-01 to Fri 2011-12-09
    dates = pd.Series(pd.to_datetime(["2009-12-01 07:45", "2011-12-09 12:50"]))
    first, last = complete_weeks(dates)
    assert first == pd.Timestamp("2009-12-07")  # first full Monday-Sunday week
    assert last == pd.Timestamp("2011-11-28")  # week of 12-05 has no data after Friday


def test_week_is_complete_when_data_starts_monday_and_ends_sunday():
    dates = pd.Series(pd.to_datetime(["2010-01-04 08:00", "2010-01-17 18:00"]))
    assert complete_weeks(dates) == (pd.Timestamp("2010-01-04"), pd.Timestamp("2010-01-11"))


def test_bulk_needs_both_factor_and_floor():
    cfg = WeeklyConfig(bulk_factor=20, bulk_min_quantity=500)
    sales = frame(
        *[sale("A", 12, "2010-01-04") for _ in range(9)],
        sale("A", 19152, "2010-01-05"),  # 1,596x the median: bulk
        *[sale("B", 1, "2010-01-04") for _ in range(9)],
        sale("B", 100, "2010-01-05"),  # 100x the median, but only 100 units: not bulk
    )
    assert flag_bulk(sales, cfg).tolist() == [False] * 9 + [True] + [False] * 10


def test_grid_fills_missing_weeks_with_zero_from_first_sale():
    weekly = pd.DataFrame({
        "stock_code": ["A", "A"],
        "week": pd.to_datetime(["2010-01-11", "2010-01-25"]),
        "quantity": [5, 3], "bulk_quantity": [0, 0], "n_lines": [1, 1], "n_customers": [1, 1],
        "avg_price": [1.0, 1.0],
    })
    out = fill_grid(weekly, pd.Timestamp("2010-02-01"))
    assert out["week"].dt.strftime("%m-%d").tolist() == ["01-11", "01-18", "01-25", "02-01"]
    assert out["quantity"].tolist() == [5, 0, 3, 0]
    assert out["avg_price"].isna().tolist() == [False, True, False, True]


def test_build_weekly_selects_by_volume_excluding_bulk_and_needs_history():
    cfg = WeeklyConfig(top_n=1, min_history_weeks=4)
    sales = frame(
        # A: steady 10/week for 6 weeks = 60 units
        *[sale("A", 10, pd.Timestamp("2010-01-04") + pd.Timedelta(weeks=w)) for w in range(6)],
        # B: small lines plus one huge bulk order; without bulk only 6 units
        *[sale("B", 1, pd.Timestamp("2010-01-04") + pd.Timedelta(weeks=w)) for w in range(6)],
        sale("B", 5000, "2010-01-20"),
        # C: big but new, only 2 weeks of history
        sale("C", 500, "2010-02-01"), sale("C", 500, "2010-02-08 10:00"),
        sale("A", 1, "2010-02-14 18:00"),  # data ends on a Sunday
    )
    weekly, report = build_weekly(sales, cfg)
    assert weekly["stock_code"].unique().tolist() == ["A"]
    assert weekly["quantity"].tolist() == [10, 10, 10, 10, 10, 11]
    assert report["bulk lines"] == 1


def test_weekly_counts_lines_customers_and_price():
    sales = frame(
        sale("A", 10, "2010-01-04", price=2.0, customer=1, invoice="1"),
        sale("A", 30, "2010-01-06", price=1.0, customer=2, invoice="2"),
        sale("A", 5, "2010-01-07", price=1.0, customer=2, invoice="3"),
        sale("A", 1, "2010-01-10 18:00"),
    )
    weekly, _ = build_weekly(sales, WeeklyConfig(min_history_weeks=1))
    row = weekly.iloc[0]
    assert (row["quantity"], row["n_lines"], row["n_customers"]) == (46, 4, 3)
    assert row["avg_price"] == (20 + 30 + 5 + 1) / 46  # weighted by quantity
