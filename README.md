# Retail Demand Forecasting

Weekly demand forecasting per product on real retail invoice data, with proper backtesting, explainability and a served, monitored model.

> Work in progress. Sections below are filled in as each step is completed.

## Data

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI, CC BY 4.0): 1,067,371 invoice lines from a UK online gift-ware retailer, December 2009 to December 2011. Many customers are wholesalers, so the data behaves much like B2B sales lines in an ERP.

```bash
# put online_retail_II.xlsx in data/raw/, then:
uv run python -m forecasting.data.load    # xlsx -> data/interim/transactions.parquet
uv run python -m forecasting.data.clean   # -> data/processed/sales.parquet
```

## Cleaning

The goal is a table of **demand**: what customers actually ordered. Not net sales, not stock movements. Rules run in this order, because the order matters. For example, duplicates must go before cancellation matching, or a duplicated cancellation would cancel twice.

| # | Rule | Lines removed | Why |
|---|------|--------------:|-----|
| 1 | Stock codes to upper case | – | 3,366 lines use lower-case suffixes (`85123a`); the same product would otherwise appear under two codes |
| 2 | Drop exact duplicates | 34,335 | Identical invoice, item, quantity, price and minute, sometimes three copies of a line. Most likely an export join (e.g. against lots or serial numbers) that multiplied rows, not real repeat lines |
| 3 | Keep product codes only (`\d{5}[A-Z]*`) | 5,980 | Postage (`POST`, `DOT`, `C2`), discounts (`D`), manual entries (`M`), bank and Amazon fees, gift vouchers, test codes |
| 4 | Drop price ≤ 0 | 5,928 | Warehouse adjustments ("check", "damages", "?", "thrown away") and bad-debt entries |
| 5 | Drop quantity ≤ 0 that is not a cancellation | 0 | A safety net: all 3,457 such lines have price 0 and no customer, so rule 4 already removed them |
| 6 | Net cancellations against invoices | 17,914 cancellations, 7,178 invoice lines | See below |

**996,036 lines remain**, about 10.7M units.

### Cancellations

A cancelled order was never demand. Simply dropping the 19,494 cancellation lines would keep their original orders. One of them is a single order of 80,995 units, cancelled 12 minutes later. So each cancellation is matched to the invoice it cancels:

1. Customer and item must match, and the invoice must come before the cancellation.
2. **Exact quantity match first**, taking the most recent invoice. This catches full cancellations: 6,354 lines, including both huge orders (invoices 541431 and 581483).
3. **Otherwise allocate FIFO** across that customer's invoices for the item, which handles partial returns: 9,648 lines.
4. What cannot be matched is dropped and counted: 328 cancellations without a customer, plus returns of goods bought before the data starts. That is 17,480 of 467,739 cancelled units (3.7%).

**FIFO or LIFO?** Only partial returns go through allocation, so the choice should barely matter. Measured with `scripts/compare_cancel_strategies.py`, switching to LIFO moves 18,743 units between weeks: 0.18% of all units, in 3% of item-weeks. FIFO is kept.

**Open assumption:** a return weeks later also reduces demand in the week of the original order. Arguably that demand was real at the time, since the goods left the warehouse. This remains to be tested.

### Not removed: bulk orders

The largest remaining lines are real one-off wholesale orders. Examples: 19,152 mugs and about 12,500 each of four party-tableware sets for one Danish customer at £0.10 each, and 4 × 10,000 tissue packs for one UK customer. They are real sales, so the cleaning step keeps them. They are not repeatable demand, though, so they are handled at the modelling stage (see Features).
