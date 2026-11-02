import pandas as pd
import pytest

from forecasting.data.clean import CleaningReport, basic_filters, clean

T0 = pd.Timestamp("2010-01-04 10:00")


def line(invoice, code="22423", qty=1, days=0, price=2.5, customer=100, description="ITEM"):
    return {
        "invoice": invoice,
        "stock_code": code,
        "description": description,
        "quantity": qty,
        "invoice_date": T0 + pd.Timedelta(days=days),
        "price": price,
        "customer_id": customer,
        "country": "United Kingdom",
    }


def frame(*lines):
    df = pd.DataFrame(lines)
    df["customer_id"] = df["customer_id"].astype("Int64")
    return df


def quantities(sales):
    """{invoice: quantity} of what survived cleaning."""
    return dict(zip(sales["invoice"], sales["quantity"]))


# --- rules 1-5 -------------------------------------------------------------


def test_lowercase_codes_become_uppercase():
    out = basic_filters(frame(line("1", code="85123a")), CleaningReport())
    assert out["stock_code"].tolist() == ["85123A"]


def test_exact_duplicates_dropped_but_different_quantities_kept():
    df = frame(line("1", qty=6), line("1", qty=6), line("1", qty=6), line("1", qty=12))
    out = basic_filters(df, CleaningReport())
    assert sorted(out["quantity"]) == [6, 12]


@pytest.mark.parametrize("code", ["POST", "DOT", "M", "C2", "BANK CHARGES", "gift_0001_20", "TEST001", "DCGS0058"])
def test_non_product_codes_dropped(code):
    assert basic_filters(frame(line("1", code=code)), CleaningReport()).empty


@pytest.mark.parametrize("code", ["22423", "85123A", "15056BL"])
def test_product_codes_kept(code):
    assert len(basic_filters(frame(line("1", code=code)), CleaningReport())) == 1


def test_zero_or_negative_price_dropped():
    df = frame(line("1", price=0.0), line("A563185", price=-11062.06), line("2"))
    assert basic_filters(df, CleaningReport())["invoice"].tolist() == ["2"]


def test_negative_quantity_without_C_is_an_adjustment():
    df = frame(line("1", qty=-720, description="damaged"), line("C2", qty=-1), line("3"))
    assert sorted(basic_filters(df, CleaningReport())["invoice"]) == ["3", "C2"]


# --- rule 6: cancellations ---------------------------------------------------


def test_full_cancellation_removes_the_order():
    # The 80,995-unit order cancelled 12 minutes later
    df = frame(line("581483", qty=80995), line("C581484", qty=-80995, days=0.01))
    sales, report = clean(df)
    assert sales.empty
    assert report.cancellations["exact match"] == 1


def test_exact_match_takes_the_most_recent_identical_invoice():
    df = frame(line("1", qty=10, days=0), line("2", qty=10, days=30), line("C3", qty=-10, days=31))
    assert quantities(clean(df)[0]) == {"1": 10}


def test_exact_match_wins_over_allocation():
    # FIFO would take 10 from invoice 1; the exact match says invoice 2 was cancelled
    df = frame(line("1", qty=20, days=0), line("2", qty=10, days=5), line("C3", qty=-10, days=6))
    assert quantities(clean(df, "fifo")[0]) == {"1": 20}


def test_partial_return_fifo_reduces_oldest():
    df = frame(line("1", qty=10, days=0), line("2", qty=10, days=5), line("C3", qty=-4, days=6))
    assert quantities(clean(df, "fifo")[0]) == {"1": 6, "2": 10}


def test_partial_return_lifo_reduces_newest():
    df = frame(line("1", qty=10, days=0), line("2", qty=10, days=5), line("C3", qty=-4, days=6))
    assert quantities(clean(df, "lifo")[0]) == {"1": 10, "2": 6}


def test_allocation_spans_several_invoices():
    df = frame(line("1", qty=3, days=0), line("2", qty=5, days=1), line("C3", qty=-7, days=2))
    assert quantities(clean(df, "fifo")[0]) == {"2": 1}


def test_cancellation_never_touches_later_invoices():
    df = frame(line("C1", qty=-5, days=0), line("2", qty=5, days=1))
    sales, report = clean(df)
    assert quantities(sales) == {"2": 5}
    assert report.cancellations["units unmatched"] == 5


def test_customer_and_item_must_both_match():
    df = frame(
        line("1", qty=5, customer=100, code="22423"),
        line("C2", qty=-5, customer=200, code="22423", days=1),  # other customer
        line("C3", qty=-5, customer=100, code="85123A", days=1),  # other item
    )
    sales, report = clean(df)
    assert quantities(sales) == {"1": 5}
    assert report.cancellations["units unmatched"] == 10


def test_cancellation_without_customer_is_dropped_and_counted():
    df = frame(line("1", qty=5, customer=None), line("C2", qty=-5, customer=None, days=1))
    sales, report = clean(df)
    assert quantities(sales) == {"1": 5}
    assert report.cancellations["no customer"] == 1


def test_leftover_beyond_what_was_bought_is_unmatched():
    df = frame(line("1", qty=3), line("C2", qty=-10, days=1))
    sales, report = clean(df)
    assert sales.empty
    assert report.cancellations["units matched"] == 3
    assert report.cancellations["units unmatched"] == 7
