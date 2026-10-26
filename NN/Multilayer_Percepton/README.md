# Entrenamiento MLP actual

La ejecución vigente es la escalera de 100 en 100 chunks, con LayerNorm,
cosine, shuffled, scaler de todo TRAIN y evaluación local/global 50/50.

| Archivo o carpeta | Uso |
|---|---|
| `dataset.py` | Lectura de datos, muestreo y normalización |
| `mlp_layernorm_shuffled_ladder100_dualeval.py` | Entrenamiento vigente |
| `run_chunk_ladder100_dualeval.sh` | Lanzador de etapas 100, 200, ... |
| `README_ladder100_dualeval.md` | Configuración y criterios de evaluación |
| `plot_training_metrics.py` | Gráficas de métricas, sin medias por defecto |
| `tests/` | Pruebas de muestreo, memoria, recuperación y selección de best |
| `analysis_scaler_2_vs_all/` | Scaler completo y estadísticas de comparación |
| `compare_scaler_all_chunks.py` | Código que calculó el scaler completo |
| `chunk_ladder100_layernorm_cosine_shuffled_allscaler_dualeval_100epochs_1GPU/` | Resultados, muestras y checkpoints actuales |
| `chunk_ladder100_dualeval_launcher.log` | Log del lanzador activo |
| `archive/` | Experimentos anteriores comprimidos y manifiesto de limpieza |

Log de la primera etapa:
`chunk_ladder100_layernorm_cosine_shuffled_allscaler_dualeval_100epochs_1GPU/chunks_100/run.log`.

El scaler completo sigue en su ruta original para que el proceso activo y
las siguientes etapas puedan cargarlo sin cambios. Los experimentos históricos
se conservan comprimidos; los scripts de pruebas redundantes se han retirado.
