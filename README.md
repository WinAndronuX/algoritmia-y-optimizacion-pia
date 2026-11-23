# Proyecto Final – Hiper-heurística de Selección para TSP

Genera las etiquetas (corriendo 2-opt hill climbing, SA+2-opt y MST+2-opt
sobre las 500 instancias) y entrena un clasificador que predice qué algoritmo
será el mejor para una instancia nueva.

## Ejecutar

```bash
cd PIA
python3 src/label.py          # ~1 min, genera results/*_labeled.csv
python3 src/ml.py             # métricas + figuras
```

Usa Python 3.13 con scikit-learn/pandas/matplotlib instalados.

## Algoritmos y parámetros

- **2-opt hill climbing**: solución inicial por vecino más cercano, búsqueda
  local hasta óptimo local (delta O(1)).
- **SA + 2-opt**: T0 = 15.0, enfriamiento geométrico α = 0.995,
  iters = max(500, 150·n), solución inicial NN, semilla derivada del
  `instance_id`. (T0 coherente con el T* ≈ 15.46 de Tarea 3.)
- **MST + 2-opt**: Prim + preorder DFS + 2-opt.

## Salidas

- `results/train_labeled.csv`, `results/test_labeled.csv` — dataset etiquetado.
- `results/metrics.txt`, `results/best_model.joblib`.
- `figures/confusion_matrix.png`.
