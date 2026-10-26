# Escalera de 100 chunks y evaluación local/global

Entrenamiento: `mlp_layernorm_shuffled_ladder100_dualeval.py`.
Lanzador: `run_chunk_ladder100_dualeval.sh`.
Resultados: `chunk_ladder100_layernorm_cosine_shuffled_allscaler_dualeval_100epochs_1GPU/`.

Etapas: 100, 200, 300, ..., 14.500, 14.503 chunks anidados con seed 42.
Cada etapa realiza 100 épocas. La primera empieza con pesos aleatorios.
Las siguientes cargan los pesos de `best` de la etapa anterior y reinician
optimizador y scheduler, como la escalera original.

Se conservan LayerNorm, dropout 0.30, AdamW lr=0.001 y weight_decay=0.0001,
batch TRAIN 8192 con balanceo 50/50, shuffled por época y cosine warm
restarts T0=20, Tmult=1, min_lr=0.00001. Se carga el scaler ya calculado
sobre todo TRAIN, sin recalcularlo. Una sola GPU, física 2 por defecto.

Después de cada chunk se imprimen TRAIN ONLINE y cuatro evaluaciones:
- TRAIN EVAL STAGE: filas TRAIN de los chunks de la etapa.
- TEST EVAL STAGE: filas TEST de los chunks de la etapa.
- TRAIN EVAL ALL: filas TRAIN de todos los chunks físicos.
- TEST EVAL ALL: filas TEST de todos los chunks físicos.

Cada evaluación usa filas fijas, únicas y balanceadas globalmente 50/50.
El objetivo es 29.006 positivas y 29.006 negativas por evaluación,
configurable con --eval-target-per-class. Las cuotas se reparten entre chunks;
si no hay suficientes filas se reduce la muestra. No es evaluación de todas
las filas ni muestreo ponderado por tamaño de chunk. Cada línea imprime
exactamente positive, negative y n. Los conteos también se obtienen de
tp+fn y tn+fp en el historial JSONL. Los manifiestos registran las filas exactas.

Las muestras STAGE permanecen fijas dentro de una etapa y cambian al ampliarla.
Las muestras ALL se comparten entre todas las etapas, para comparar el progreso
sobre la misma población. TRAIN ALL incluye datos aún no usados para entrenar;
no se mezclan filas TEST con las de entrenamiento.

BEST se elige únicamente por menor TEST EVAL STAGE loss al cerrar cada época.
Las métricas ALL son diagnósticas y no cambian la selección de pesos.
Se guardan best y last al cerrar cada época y recovery cada 100 chunks.
Al cerrar la época se reutilizan las cuatro evaluaciones del último chunk;
no hay evaluación adicional de todas las filas ni ordenación global de AUC.
Los AUC por chunk y por muestra son exactos; el agregado TRAIN ONLINE de la
época conserva solo contadores, sin AUC acumulado, para limitar memoria.
Las cachés de embeddings están en disco y la evaluación usa lotes de 1024.

Los resultados son exploratorios: TEST STAGE selecciona modelos a petición
del usuario. Para una estimación final independiente habría que reservar otro
conjunto o seleccionar con VALIDATION.

Pruebas CPU:
- `python tests/test_ladder100_dualeval.py`
- `python tests/test_ladder100_sampling.py`

Gráficas (desde la raíz del proyecto):
```
python NN/Multilayer_Percepton/plot_training_metrics.py \
 --input NN/Multilayer_Percepton/chunk_ladder100_layernorm_cosine_shuffled_allscaler_dualeval_100epochs_1GPU/chunks_100 \
 --chunks-per-epoch 100 --metrics loss \
 --series train_eval_stage test_eval_stage train_eval_all test_eval_all \
 --separate-series
```
