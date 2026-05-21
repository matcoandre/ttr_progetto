from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "data" / "data_oecd.csv"
VALUE_COLUMNS = ("OBS_VALUE", "Observation Value")


def _to_numeric(series):
    return pd.to_numeric(
        series.astype("string")
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )


def carica_e_pulisci(path=DATA_PATH, impute_missing=True):
    """Carica il CSV OECD e restituisce una tabella paese x indicatore."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File dati non trovato: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required_columns = {"Country", "Indicator"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Colonne mancanti nel CSV: {sorted(missing_columns)}")

    if "INEQUALITY" in df.columns:
        total_mask = df["INEQUALITY"].astype("string").str.upper().eq("TOT")
    elif "Inequality" in df.columns:
        total_mask = df["Inequality"].astype("string").str.contains(
            "total", na=False, case=False
        )
    else:
        raise ValueError(
            "Colonna di disaggregazione non trovata: serve INEQUALITY o Inequality"
        )

    filtered = df.loc[total_mask].copy()

    value_columns = [col for col in VALUE_COLUMNS if col in filtered.columns]
    if not value_columns:
        raise ValueError(f"Nessuna colonna valore trovata tra: {VALUE_COLUMNS}")

    values = None
    for col in value_columns:
        parsed = _to_numeric(filtered[col])
        values = parsed if values is None else values.combine_first(parsed)

    filtered["v_clean"] = values
    filtered = filtered.dropna(subset=["v_clean", "Country", "Indicator"])

    pivot = filtered.pivot_table(
        index="Country",
        columns="Indicator",
        values="v_clean",
        aggfunc="mean",
    ).sort_index()

    if impute_missing:
        pivot = pivot.fillna(pivot.mean(numeric_only=True))

    return pivot


if __name__ == "__main__":
    res = carica_e_pulisci()
    print(res.index.tolist())
