import pandas as pd

from forecasting.data.weekly import (
    WeeklyConfig,
    build_weekly,
    complete_weeks,
    fill_grid,
    first_test_week,
    flag_bulk,
    select_products,
    trading_days,
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


def weekly_rows(code, weeks, qty=10, bulk=0):
    """Pre-aggregated weekly rows, for testing selection directly."""
    return pd.DataFrame({
        "stock_code": code, "week": pd.to_datetime(weeks), "quantity": qty, "bulk_quantity": bulk,
    })


def mondays(start, n):
    return pd.date_range(start, periods=n, freq="W-MON")


# --- calendar --------------------------------------------------------------


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


def test_test_window_is_the_last_n_weeks():
    assert first_test_week(pd.Timestamp("2011-11-28"), WeeklyConfig(test_weeks=12)) == pd.Timestamp("2011-09-12")


def test_trading_days_counts_distinct_days_across_all_products():
    sales = frame(
        sale("A", 1, "2010-01-04 09:00"), sale("B", 1, "2010-01-04 15:00"),  # same Monday
        sale("A", 1, "2010-01-05"), sale("A", 1, "2010-01-10"),
    )
    assert trading_days(sales).to_dict() == {pd.Timestamp("2010-01-04"): 3}


# --- bulk ------------------------------------------------------------------


def test_bulk_needs_both_factor_and_floor():
    cfg = WeeklyConfig(bulk_factor=20, bulk_min_quantity=500)
    sales = frame(
        *[sale("A", 12, "2010-01-04") for _ in range(9)],
        sale("A", 19152, "2010-01-05"),  # 1,596x the median: bulk
        *[sale("B", 1, "2010-01-04") for _ in range(9)],
        sale("B", 100, "2010-01-05"),  # 100x the median, but only 100 units: not bulk
    )
    assert flag_bulk(sales, cfg).tolist() == [False] * 9 + [True] + [False] * 10


def test_bulk_threshold_learned_before_the_test_window():
    cfg = WeeklyConfig(bulk_factor=20, bulk_min_quantity=500)
    sales = frame(
        *[sale("A", 10, "2010-01-04") for _ in range(3)],  # before: median 10 -> threshold 200
        *[sale("A", 1000, "2010-06-01") for _ in range(5)],  # test period, would move the median
    )
    flags = flag_bulk(sales, cfg, before=pd.Timestamp("2010-05-01"))
    assert flags.tolist() == [False] * 3 + [True] * 5


# --- selection -------------------------------------------------------------


def test_selection_ignores_the_test_window():
    cfg = WeeklyConfig(top_n=1, min_history_weeks=4, recent_weeks=4)
    cutoff = pd.Timestamp("2010-03-01")
    weekly = pd.concat([
        weekly_rows("A", mondays("2010-01-04", 8), qty=10),  # 80 before the cutoff
        weekly_rows("B", mondays("2010-01-04", 8), qty=5),  # 40 before...
        weekly_rows("B", mondays("2010-03-01", 4), qty=1000),  # ...but huge in the test window
    ])
    assert select_products(weekly, cutoff, cfg).tolist() == ["A"]


def test_selection_excludes_discontinued_products():
    cfg = WeeklyConfig(top_n=5, min_history_weeks=4, recent_weeks=4)
    cutoff = pd.Timestamp("2010-06-07")
    weekly = pd.concat([
        weekly_rows("ACTIVE", mondays("2010-01-04", 22), qty=1),
        weekly_rows("STOPPED", mondays("2010-01-04", 10), qty=100),  # last sale in March
    ])
    assert select_products(weekly, cutoff, cfg).tolist() == ["ACTIVE"]


def test_selection_needs_history_and_ranks_without_bulk():
    cfg = WeeklyConfig(top_n=1, min_history_weeks=4, recent_weeks=4)
    cutoff = pd.Timestamp("2010-02-15")
    weekly = pd.concat([
        weekly_rows("A", mondays("2010-01-04", 6), qty=10),
        weekly_rows("B", mondays("2010-01-04", 6), qty=1000, bulk=999),  # 6 units without bulk
        weekly_rows("NEW", mondays("2010-02-01", 2), qty=500),  # only 2 weeks of history
    ])
    assert select_products(weekly, cutoff, cfg).tolist() == ["A"]


# --- grid and end to end ---------------------------------------------------


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


def test_build_weekly_end_to_end_with_closed_week():
    cfg = WeeklyConfig(top_n=1, min_history_weeks=2, recent_weeks=2, test_weeks=1)
    sales = frame(
        sale("A", 10, "2010-01-04"), sale("A", 10, "2010-01-11"),
        # week of 2010-01-18: shop closed, no sales at all
        sale("A", 10, "2010-01-25"), sale("A", 10, "2010-02-01"),
        sale("A", 4, "2010-02-07 18:00"),  # data ends on a Sunday
    )
    weekly, report = build_weekly(sales, cfg)
    assert weekly["quantity"].tolist() == [10, 10, 0, 10, 14]
    assert weekly["trading_days"].tolist() == [1, 1, 0, 1, 2]
    assert report["weeks with no sales at all"] == "2010-01-18"


def test_weekly_counts_lines_customers_and_price():
    sales = frame(
        sale("A", 10, "2010-01-04", price=2.0, customer=1, invoice="1"),
        sale("A", 30, "2010-01-06", price=1.0, customer=2, invoice="2"),
        sale("A", 5, "2010-01-07", price=1.0, customer=2, invoice="3"),
        sale("A", 1, "2010-01-11"), sale("A", 1, "2010-01-17 18:00"),
    )
    cfg = WeeklyConfig(min_history_weeks=1, recent_weeks=1, test_weeks=1)
    weekly, _ = build_weekly(sales, cfg)
    row = weekly.iloc[0]
    assert (row["quantity"], row["n_lines"], row["n_customers"]) == (45, 3, 2)
    assert row["avg_price"] == (20 + 30 + 5) / 45  # weighted by quantity
