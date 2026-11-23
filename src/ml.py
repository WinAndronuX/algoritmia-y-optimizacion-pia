"""Train and evaluate the selection hyper-heuristic classifier."""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CLASSES = ["2opt_hill_climbing", "sa_2opt", "mst_2opt"]
COST_COLS = {
    "2opt_hill_climbing": "cost_2opt_hill_climbing",
    "sa_2opt": "cost_sa_2opt",
    "mst_2opt": "cost_mst_2opt",
}


def load(split):

    path = os.path.join(RESULTS_DIR, f"{split}_labeled.csv")
    df = pd.read_csv(path)

    return df


def features(df):
    X = pd.get_dummies(df[["n", "distribution", "mean_pairwise_dist",  # type: ignore[arg-type]
                           "std_coords", "mean_centroid_dist", "mean_nn_dist"]],
                       columns=["distribution"], drop_first=False)
    # ensure consistent column order
    return X


def main():

    train = load("train")
    test = load("test")

    X_train = features(train)
    X_test = features(test)
    # align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    y_train = train["best_algorithm"]
    y_test = test["best_algorithm"]

    # Baseline: majority class
    majority = y_train.value_counts().idxmax()
    baseline_acc = (y_test == majority).mean()

    print("=== Balance de clases (train) ===")
    print(y_train.value_counts(normalize=True).round(4))
    print("\nMajority (baseline):", majority, f"({baseline_acc:.4f} on test)")

    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=20),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=200),
    }

    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)

        # regret
        regrets = []

        for i, p in enumerate(pred):
            chosen_cost = test.iloc[i][COST_COLS[p]]
            best_cost = min(float(test.iloc[i][c]) for c in COST_COLS.values())
            regrets.append(chosen_cost - best_cost)

        mean_regret = float(np.mean(regrets))
        results[name] = {"model": model, "pred": pred, "acc": acc,
                        "regret": mean_regret}

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}  (baseline {baseline_acc:.4f}, "
              f"delta {acc - baseline_acc:+.4f})")
        print(f"Mean regret: {mean_regret:.6f}")
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(y_test, pred, labels=CLASSES))
        print(classification_report(y_test, pred, labels=CLASSES, zero_division=0))  # type: ignore[arg-type]

    # Feature importance (tree)
    dt = results["DecisionTree"]["model"]

    print("\n=== Decision Tree rules ===")
    print(export_text(dt, feature_names=list(X_train.columns), max_depth=4))
    print("\n=== Feature importances (DecisionTree) ===")

    for feat, imp in sorted(zip(X_train.columns, dt.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"  {feat:32s} {imp:.4f}")

    rf = results["RandomForest"]["model"]

    print("\n=== Feature importances (RandomForest) ===")

    for feat, imp in sorted(zip(X_train.columns, rf.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"  {feat:32s} {imp:.4f}")

    # Save confusion matrix figure for best model by accuracy
    best_name = max(results, key=lambda k: results[k]["acc"])
    best_pred = results[best_name]["pred"]

    cm = confusion_matrix(y_test, best_pred, labels=CLASSES)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))

    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix - {best_name} (acc={results[best_name]['acc']:.3f})")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im)
    fig.tight_layout()

    fig.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Save metrics summary
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:

        f.write(f"majority_baseline: {majority} acc={baseline_acc:.4f}\n")

        for name, r in results.items():
            f.write(f"{name}: acc={r['acc']:.4f} regret={r['regret']:.6f}\n")

    print(f"\nBest model: {best_name} (acc={results[best_name]['acc']:.4f})")
    print("Figures ->", FIGURES_DIR)

    # persist best model
    joblib.dump(results[best_name]["model"],
                os.path.join(RESULTS_DIR, "best_model.joblib"))


if __name__ == "__main__":
    main()
