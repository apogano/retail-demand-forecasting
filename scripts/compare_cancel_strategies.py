"""Does FIFO vs LIFO cancellation matching change weekly demand per item?

Only partial returns go through FIFO/LIFO (exact matches are handled first),
so we expect a small difference. This measures it instead of assuming it.
"""

import pandas as pd

from forecasting.data.clean import clean
from forecasting.data.load import INTERIM_PATH


def weekly(sales: pd.DataFrame) -> pd.Series:
    week = sales["invoice_date"].dt.to_period("W-SUN").dt.start_time  # weeks start on Monday
    return sales.groupby([sales["stock_code"], week])["quantity"].sum()


raw = pd.read_parquet(INTERIM_PATH)
fifo = weekly(clean(raw, "fifo")[0])
lifo = weekly(clean(raw, "lifo")[0])

both = pd.concat({"fifo": fifo, "lifo": lifo}, axis=1).fillna(0)
diff = (both["fifo"] - both["lifo"]).abs()

print(f"Item-weeks:                  {len(both):,}")
print(f"Item-weeks that differ:      {(diff > 0).sum():,} ({(diff > 0).mean():.2%})")
print(f"Units moved between weeks:   {diff.sum() / 2:,.0f} of {both['fifo'].sum():,.0f} "
      f"({diff.sum() / 2 / both['fifo'].sum():.3%})")
print("\nLargest differences:")
print(both.assign(diff=diff).nlargest(10, "diff"))
