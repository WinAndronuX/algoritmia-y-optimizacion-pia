import csv
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from algorithms import ALGORITHMS, tour_cost

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "hh_tsp_dataset", "data")
INSTANCES_DIR = os.path.join(DATA_DIR, "instances")
OUT_DIR = os.path.join(ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)


def load_points(instance_id):

    path = os.path.join(INSTANCES_DIR, f"{instance_id}.csv")
    pts = []

    with open(path) as f:

        r = csv.reader(f)
        next(r)  # header

        for row in r:

            if not row:
                continue

            pts.append((float(row[0]), float(row[1])))

    return pts


def label_split(split_name):

    in_path = os.path.join(DATA_DIR, f"{split_name}.csv")
    out_path = os.path.join(OUT_DIR, f"{split_name}_labeled.csv")

    with open(in_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    out_fields = list(fieldnames if fieldnames is not None else []) + [
        "cost_2opt_hill_climbing", "cost_sa_2opt", "cost_mst_2opt", "best_algorithm",
    ]

    t0 = time.time()
    out_rows = []

    for idx, row in enumerate(rows, 1):

        iid = row["instance_id"]
        points = load_points(iid)

        seed = int(iid.split("_")[1])
        costs = {}

        for algo_name, algo_fn in ALGORITHMS.items():

            if algo_name == "sa_2opt":
                tour = algo_fn(points, random.Random(seed))
            else:
                tour = algo_fn(points)

            costs[algo_name] = tour_cost(tour, points)

        best = min(costs, key=lambda k: costs[k])
        row = dict(row)

        row["cost_2opt_hill_climbing"] = repr(costs["2opt_hill_climbing"])
        row["cost_sa_2opt"] = repr(costs["sa_2opt"])
        row["cost_mst_2opt"] = repr(costs["mst_2opt"])
        row["best_algorithm"] = best

        out_rows.append(row)

        if idx % 25 == 0 or idx == len(rows):
            print(f"  {split_name}: {idx}/{len(rows)}  ({time.time()-t0:.1f}s)", flush=True)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(out_rows)

    print(f"  -> {out_path}")

    return out_rows


if __name__ == "__main__":

    for split in ("train", "test"):
        print(f"Labeling {split}...")
        label_split(split)

    print("Done.")
