"""Load the raw Online Retail II workbook and cache it as parquet.

The workbook has one sheet per year (2009-2010, 2010-2011). We read it row by
row with calamine (a fast Rust xlsx reader) so we can show a progress bar,
then keep a parquet copy that every later step reads in seconds.
"""

from pathlib import Path

import pandas as pd
from python_calamine import CalamineWorkbook
from tqdm import tqdm

RAW_PATH = Path("data/raw/online_retail_II.xlsx")
INTERIM_PATH = Path("data/interim/transactions.parquet")

# Original column name -> our snake_case name
COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def as_code(value) -> str | None:
    """Invoice and stock codes are labels, not numbers.

    Excel stores 85048 as a number, so the reader returns 85048.0;
    turn it back into "85048". Text codes like "85123A" pass through.
    """
    if value is None or value == "":
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def read_sheet(workbook: CalamineWorkbook, name: str) -> pd.DataFrame:
    """Read one sheet row by row, with a progress bar."""
    sheet = workbook.get_sheet_by_name(name)
    rows = sheet.iter_rows()
    header = next(rows)

    missing = set(COLUMNS) - set(header)
    if missing:
        raise ValueError(f"Sheet {name!r}: missing columns {sorted(missing)}")

    data = list(tqdm(rows, total=sheet.height - 1, desc=name, unit=" rows", unit_scale=True))
    return pd.DataFrame(data, columns=header)


def read_workbook(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read every sheet of the workbook and stack them into one frame."""
    workbook = CalamineWorkbook.from_path(str(path))
    frames = [read_sheet(workbook, name) for name in workbook.sheet_names]
    df = pd.concat(frames, ignore_index=True)

    df = df.rename(columns=COLUMNS)[list(COLUMNS.values())]
    df = df.replace("", None)  # calamine returns "" for empty cells
    df["invoice"] = df["invoice"].map(as_code)
    df["stock_code"] = df["stock_code"].map(as_code)
    # A few descriptions are typed as numbers in Excel (e.g. 20713); parquet needs one type per column
    df["description"] = df["description"].map(as_code)
    df["country"] = df["country"].map(as_code)
    df["quantity"] = df["quantity"].astype("int64")
    df["price"] = df["price"].astype("float64")
    df["customer_id"] = pd.to_numeric(df["customer_id"]).astype("Int64")  # nullable integer
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def main() -> None:
    df = read_workbook()
    INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INTERIM_PATH, index=False)
    print(f"Saved {len(df):,} rows to {INTERIM_PATH}")


if __name__ == "__main__":
    main()
