"""First look at the raw transactions: what is in there before we clean anything."""

import pandas as pd

from forecasting.data.load import INTERIM_PATH

pd.set_option("display.width", 120)

df = pd.read_parquet(INTERIM_PATH)

print("Rows, columns:", df.shape)
print("\nTypes:\n", df.dtypes)
print("\nDate range:", df["invoice_date"].min(), "->", df["invoice_date"].max())
print("\nMissing values per column:\n", df.isna().sum())
print("\nExact duplicate rows:", df.duplicated().sum())

# Invoices: cancellations and other prefixes
cancel = df["invoice"].str.startswith("C")
print("\nCancellation lines (invoice starts with C):", cancel.sum())
print("Lines with quantity <= 0:", (df["quantity"] <= 0).sum())
print("  ...of which are NOT cancellations:", ((df["quantity"] <= 0) & ~cancel).sum())
print("Lines with price <= 0:", (df["price"] <= 0).sum())
prefixes = df["invoice"].str.extract(r"^([A-Za-z]+)")[0].value_counts()
print("Invoice prefixes:", prefixes.to_dict())

# Stock codes: real products look like 5 digits, sometimes followed by letters
print("\nDistinct stock codes:", df["stock_code"].nunique())
odd = df.loc[~df["stock_code"].str.fullmatch(r"\d{5}[A-Za-z]*"), "stock_code"]
print(f"Stock codes NOT of the form 5 digits + optional letters ({odd.nunique()} distinct):")
print(odd.value_counts().head(25))

print("\nQuantity:\n", df["quantity"].describe())
print("\nPrice:\n", df["price"].describe())
print("\nTop 10 countries by lines:\n", df["country"].value_counts().head(10))

numeric_desc = df["description"].str.fullmatch(r"\d+", na=False)
print("\nLines with a numeric description:", numeric_desc.sum())
print(df.loc[numeric_desc, ["invoice", "stock_code", "description", "quantity", "price", "customer_id"]].head(15))


cols = ["invoice", "stock_code", "description", "quantity", "price", "customer_id", "invoice_date"]

print("\nHuge lines (|quantity| > 10,000):")
print(df.loc[df["quantity"].abs() > 10_000, cols].sort_values("invoice_date"))

adj = (df["quantity"] <= 0) & ~df["invoice"].str.startswith("C")
print("\nNon-cancellation lines with quantity <= 0:")
print("  with price > 0:", (df.loc[adj, "price"] > 0).sum(), "| with customer:", df.loc[adj, "customer_id"].notna().sum())
print(df.loc[adj, "description"].value_counts().head(15))

lower = df["stock_code"].str.contains(r"[a-z]", na=False) & df["stock_code"].str.fullmatch(r"\d{5}[A-Za-z]*", na=False)
print("\nProduct codes with lowercase letters:", lower.sum(), df.loc[lower, "stock_code"].unique()[:15])

print("\nA few exact duplicates:")
print(df[df.duplicated(keep=False)].sort_values(["invoice", "stock_code"])[cols].head(10))
