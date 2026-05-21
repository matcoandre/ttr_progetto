import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pulizia_dati import carica_e_pulisci
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGET = "Life satisfaction"
EXCLUDED_COUNTRIES = ["OECD - Total", "Brazil", "Russia", "South Africa", "Colombia"]
FEATURES = [
    "Employment rate",
    "Educational attainment",
    "Household net adjusted disposable income",
    "Homicide rate",
    "Self-reported health",
    "Employees working very long hours",
    "Quality of support network",
]


def prepara_dataset():
    df = carica_e_pulisci(impute_missing=False)
    df = df.drop(index=[country for country in EXCLUDED_COUNTRIES if country in df.index])

    required = [TARGET, *FEATURES]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Indicatori mancanti nel dataset: {missing}")

    df = df.dropna(subset=[TARGET])
    return df, df[FEATURES], df[TARGET]


def crea_modello():
    return make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(20, 20),
            max_iter=10000,
            alpha=0.5,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
    )


def addestra_e_valuta(X, y):
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_mae = -cross_val_score(
        crea_modello(),
        X,
        y,
        cv=cv,
        scoring="neg_mean_absolute_error",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    model = crea_modello()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    metrics = {
        "train_mae": mean_absolute_error(y_train, train_pred),
        "test_mae": mean_absolute_error(y_test, test_pred),
        "cv_mae_mean": cv_mae.mean(),
        "cv_mae_std": cv_mae.std(),
        "test_countries": set(X_test.index),
    }
    return model, metrics


def crea_risultati(df, y, predictions, test_countries):
    results = pd.DataFrame(
        {
            "reale": y,
            "predetto": predictions,
        },
        index=df.index,
    )
    results["errore"] = (results["reale"] - results["predetto"]).abs()
    results["set"] = [
        "test" if country in test_countries else "train" for country in results.index
    ]
    return results


def stampa_risultati(results, metrics):
    print(
        f"{'Paese':<25} | {'Set':<5} | {'Reale':<7} | "
        f"{'Predetto':<8} | {'Err':<6}"
    )
    print("-" * 65)

    for country, row in results.iterrows():
        print(
            f"{country:<25} | {row['set']:<5} | {row['reale']:<7.1f} | "
            f"{row['predetto']:<8.1f} | {row['errore']:<6.2f}"
        )

    print("-" * 65)
    print(f"MAE train: {metrics['train_mae']:.4f}")
    print(f"MAE test : {metrics['test_mae']:.4f}")
    print(
        f"MAE CV 5-fold: {metrics['cv_mae_mean']:.4f} "
        f"+/- {metrics['cv_mae_std']:.4f}"
    )
    print(f"MAE totale descrittivo: {results['errore'].mean():.4f}")


def mostra_grafico(results, save_path=None):
    colors = results["set"].map({"train": "tab:blue", "test": "tab:orange"})

    plt.figure(figsize=(10, 7))
    plt.scatter(
        results["reale"],
        results["predetto"],
        c=colors,
        edgecolors="black",
        alpha=0.75,
    )

    for country, row in results.iterrows():
        plt.annotate(country, (row["reale"], row["predetto"]), fontsize=7)

    min_value = min(results["reale"].min(), results["predetto"].min())
    max_value = max(results["reale"].max(), results["predetto"].max())
    plt.plot([min_value, max_value], [min_value, max_value], "r--")
    plt.xlabel("Reale")
    plt.ylabel("Predetto")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160)
        print(f"Grafico salvato in: {save_path}")
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predizione della Life Satisfaction con una MLP su dati OECD."
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Non mostrare il grafico finale.",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        help="Salva il grafico nel percorso indicato.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df, X, y = prepara_dataset()
    model, metrics = addestra_e_valuta(X, y)
    predictions = model.predict(X)
    results = crea_risultati(df, y, predictions, metrics["test_countries"])

    stampa_risultati(results, metrics)

    if args.save_plot:
        mostra_grafico(results, args.save_plot)
    elif not args.no_plot:
        mostra_grafico(results)


if __name__ == "__main__":
    main()
