"""Turn raw transaction lines into clean sales lines that represent real demand.

Rules, in order (each one is documented in the README with the evidence behind it):

1. Stock codes to upper case       - "85123a" and "85123A" are the same product.
2. Drop exact duplicate lines      - same invoice, item, quantity, price and minute;
                                     most likely an export join that multiplied rows.
3. Keep product codes only         - 5 digits + optional letters; drops POST, DOT, M, D,
                                     BANK CHARGES, gift vouchers, TEST001, etc.
4. Drop price <= 0                 - warehouse adjustments ("damaged", "check", "?"),
                                     bad-debt entries.
5. Drop non-cancellation qty <= 0  - same adjustments; all have price 0 and no customer.
6. Match cancellations to invoices - same customer and item, earlier or same time:
     a. exact quantity match first, most recent invoice wins;
     b. otherwise allocate the cancelled quantity across invoices, FIFO or LIFO;
     c. whatever cannot be matched (no customer, invoice before the data starts)
        is dropped and counted.
   Invoice lines reduced to zero are removed: a cancelled order was never demand.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from forecasting.data.load import INTERIM_PATH

PROCESSED_PATH = Path("data/processed/sales.parquet")
PRODUCT_CODE = r"\d{5}[A-Z]*"

Strategy = Literal["fifo", "lifo"]


@dataclass
class CleaningReport:
    """How many lines each rule removed, so every number in the README is reproducible."""

    lines_in: int = 0
    steps: dict[str, int] = field(default_factory=dict)
    cancellations: dict[str, int] = field(default_factory=dict)
    lines_out: int = 0

    def __str__(self) -> str:
        out = [f"Lines in:  {self.lines_in:>10,}"]
        out += [f"  {name:<34}{-n:>10,}" for name, n in self.steps.items()]
        out += ["Cancellations:"]
        out += [f"  {name:<34}{n:>10,}" for name, n in self.cancellations.items()]
        out += [f"Lines out: {self.lines_out:>10,}"]
        return "\n".join(out)


def _drop(df: pd.DataFrame, mask: pd.Series, name: str, report: CleaningReport) -> pd.DataFrame:
    report.steps[name] = int(mask.sum())
    return df.loc[~mask]


def basic_filters(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Rules 1-5: everything except the cancellation matching."""
    report.lines_in = len(df)
    df = df.assign(stock_code=df["stock_code"].str.upper())

    df = _drop(df, df.duplicated(), "exact duplicates", report)
    df = _drop(df, ~df["stock_code"].str.fullmatch(PRODUCT_CODE, na=False), "non-product codes", report)
    df = _drop(df, df["price"] <= 0, "price <= 0", report)

    is_cancel = df["invoice"].str.startswith("C")
    df = _drop(df, ~is_cancel & (df["quantity"] <= 0), "adjustments (qty <= 0, no C)", report)
    return df


def match_cancellations(
    df: pd.DataFrame, strategy: Strategy, report: CleaningReport
) -> pd.DataFrame:
    """Rule 6: net cancellations against the invoices they cancel.

    Returns the invoice lines only, with `quantity` reduced by what was cancelled
    and lines that reached zero removed.
    """
    is_cancel = df["invoice"].str.startswith("C")
    sales = df.loc[~is_cancel].copy()
    cancels = df.loc[is_cancel].sort_values("invoice_date")

    # remaining[i] = quantity of sales line i still not cancelled
    remaining = sales["quantity"].to_dict()

    # For each (customer, item): its sales lines as (date, index), oldest first
    candidates: dict[tuple, list[tuple]] = defaultdict(list)
    keyed = sales.dropna(subset=["customer_id"]).sort_values("invoice_date")
    for idx, cust, code, date in zip(
        keyed.index, keyed["customer_id"], keyed["stock_code"], keyed["invoice_date"]
    ):
        candidates[(cust, code)].append((date, idx))

    counts = {"total": len(cancels), "exact match": 0, f"{strategy} allocation": 0,
              "no customer": 0, "units matched": 0, "units unmatched": 0}

    for cust, code, date, qty in zip(
        cancels["customer_id"], cancels["stock_code"], cancels["invoice_date"], cancels["quantity"]
    ):
        to_cancel = -qty
        if pd.isna(cust):
            counts["no customer"] += 1
            counts["units unmatched"] += to_cancel
            continue

        # Only invoices issued before (or at) the cancellation, with something left
        open_lines = [i for d, i in candidates.get((cust, code), []) if d <= date and remaining[i] > 0]

        # a. exact quantity match, most recent first
        exact = [i for i in reversed(open_lines) if remaining[i] == to_cancel]
        if exact:
            remaining[exact[0]] = 0
            counts["exact match"] += 1
            counts["units matched"] += to_cancel
            continue

        # b. allocate across invoices
        order = open_lines if strategy == "fifo" else list(reversed(open_lines))
        if order:
            counts[f"{strategy} allocation"] += 1
        for i in order:
            take = min(remaining[i], to_cancel)
            remaining[i] -= take
            to_cancel -= take
            counts["units matched"] += take
            if to_cancel == 0:
                break
        counts["units unmatched"] += to_cancel  # c. leftover is dropped

    report.cancellations = {k: int(v) for k, v in counts.items()}
    sales["quantity"] = pd.Series(remaining)
    fully_cancelled = sales["quantity"] == 0
    report.steps["invoice lines fully cancelled"] = int(fully_cancelled.sum())
    report.steps["cancellation lines (netted)"] = len(cancels)
    return sales.loc[~fully_cancelled]


def clean(df: pd.DataFrame, strategy: Strategy = "fifo") -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport()
    df = basic_filters(df, report)
    sales = match_cancellations(df, strategy, report)
    report.lines_out = len(sales)
    return sales.reset_index(drop=True), report


def main() -> None:
    raw = pd.read_parquet(INTERIM_PATH)
    sales, report = clean(raw, strategy="fifo")
    print(report)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\nLargest remaining lines:")
    print(sales.nlargest(10, "quantity"))

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    sales.to_parquet(PROCESSED_PATH, index=False)
    print(f"\nSaved {len(sales):,} lines to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
