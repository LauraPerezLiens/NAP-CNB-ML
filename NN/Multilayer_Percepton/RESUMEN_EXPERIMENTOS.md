# Registro de pruebas del MLP: desde BatchNorm hasta la escalera de 100 chunks

Fecha de recopilación: 26 de octubre de 2026. Este documento es una instantánea,
no un informe de 100 épocas completadas en todos los experimentos.
Los valores históricos se han reconstruido de historiales JSONL archivados;
BatchNorm procede del registro técnico conservado antes de borrar sus salidas.

## 1. Configuración de referencia y cómo interpretar las cifras

Se ha trabajado con clasificación binaria sobre embeddings precomputados:
ProtT5, 1.024 dimensiones, y SecondaryBERT, 768 dimensiones: 1.792 en total.
La arquitectura de referencia es 1792 → 512 → 128 → 1, con normalización,
ReLU y dropout 0,30. Se usa BCEWithLogitsLoss, AdamW, learning rate inicial
0,001, weight decay 0,0001 y batches de 8.192 filas, 4.096 por clase.
El scheduler de las escaleras es cosine con warm restarts: T0=20 épocas,
Tmult=1 y learning rate mínimo 0,00001. Semilla 42; 100 épocas por etapa,
sin early stopping, sobre una GPU por entrenamiento.

Los datos están preparados en 11 datasets y 14.503 chunks físicos. TRAIN,
VALIDATION y TEST se distinguen por split_id. La política acordada es dividir
por filas: se permite compartir grupos/proteínas entre splits. No se ha
redefinido esta política como un split por grupos. Los resultados no prueban
generalización a proteínas o grupos totalmente nuevos.

En las escaleras se reutilizan los pesos del best de la etapa anterior y se
reinician optimizador y scheduler. Por tanto, son entrenamientos acumulativos,
no modelos independientes entrenados desde cero para cada tamaño.

Salvo que se indique lo contrario, las tablas muestran el checkpoint con menor
TEST loss de la etapa, no necesariamente la última época ni la máxima accuracy.
TEST participa en la selección: estos resultados son exploratorios. VALIDATION
no ha sustituido a TEST como criterio de selección en las pruebas realizadas.

## 2. BatchNorm: baseline y diagnóstico de estadísticas de inferencia

Se completaron un baseline y una ejecución diagnóstica de 10 chunks y 100 épocas.
El síntoma fue una discrepancia entre el comportamiento en entrenamiento y en
modo evaluación. Se evaluaron los mismos pesos y muestras con las estadísticas
acumuladas de BatchNorm y, como diagnóstico, con estadísticas del batch,
desactivando dropout y evitando modificar el modelo original.

Resultados de TEST en época 100 conservados en la documentación:

| Normalización / evaluación | Loss | Accuracy | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| BatchNorm, estadísticas acumuladas | 1,880200 | 76,9167 % | 53,8600 % | 0,986282 |
| BatchNorm, estadísticas del batch — diagnóstico | 0,050603 | 98,5450 % | 98,3391 % | 0,998883 |
| LayerNorm, evaluación normal | 0,003121 | 99,9170 % | 99,8538 % | 0,999991 |

Interpretación: el modo de normalizar en inferencia afectaba fuertemente al
resultado de BatchNorm. Una AUC alta con accuracy/recall y loss peores también
indica que ordenar ejemplos y clasificarlos con el umbral fijo no es lo mismo.
La evaluación con estadísticas del batch fue un diagnóstico, no una validación
de un modelo independiente de la composición del batch en inferencia.

También se inició una escalera BatchNorm, que se detuvo durante la etapa de
20 chunks. Sus salidas y scripts se eliminaron por decisión del usuario; no
queda una tabla numérica fiable de esa etapa. Se decidió continuar con LayerNorm.

## 3. Sustitución de BatchNorm por LayerNorm

Se sustituyeron las dos capas de normalización por LayerNorm. Esta normaliza
las activaciones de cada muestra, sin running mean/variance dependientes de
los batches. El primer entrenamiento de 10 chunks completó 100 épocas.

- Mejor checkpoint: época 98, TEST loss 0,002926; accuracy 99,9269 %;
  recall 99,8738 %; AUC 0,999990; 3 falsos positivos y 19 falsos negativos.
- Último checkpoint: época 100, loss 0,003121; accuracy 99,9170 %;
  3 falsos positivos y 22 falsos negativos.
- TEST evaluado: 30.104 filas, 15.052 por clase.

Conclusión operativa: LayerNorm resolvió en esta prueba la fuerte diferencia
asociada a las estadísticas de BatchNorm y se adoptó para los siguientes ensayos.
La comparación fue con una semilla; no demuestra universalmente que una
normalización sea superior en cualquier población.

## 4. Escalera de 10 en 10, LayerNorm y cosine, sin shuffled entre épocas

Etapas 10, 20, 30…; scaler ajustado con TRAIN de 2 chunks; orden de chunks fijo
entre épocas. Las muestras se barajaban dentro del entrenamiento, pero no el
orden de visita de los chunks. TRAIN EVAL y TEST EVAL balanceaban globalmente
con todas las positivas del split de la etapa y otras tantas negativas fijas.

| Chunks | Épocas completadas | Época best | TEST loss | Accuracy | Recall | ROC-AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 100 | 98 | 0.002926 | 99.9269 % | 99.8738 % | 0.999990 |
| 20 | 100 | 100 | 0.003898 | 99.8859 % | 99.7902 % | 0.999991 |
| 30 | 100 | 100 | 0.005635 | 99.8248 % | 99.7578 % | 0.999966 |
| 40 | 100 | 100 | 0.010749 | 99.6793 % | 99.5837 % | 0.999907 |
| 50 | 100 | 100 | 0.015396 | 99.5222 % | 99.2825 % | 0.999882 |
| 60 | 100 | 100 | 0.041782 | 98.5328 % | 97.7107 % | 0.999421 |
| 70 | 100 | 100 | 0.048620 | 98.2409 % | 97.1894 % | 0.999216 |
| 80 | 100 | 100 | 0.048334 | 98.2566 % | 97.2959 % | 0.999075 |
| 90 | 71 | 60 | 0.055978 | 97.9806 % | 96.9819 % | 0.998624 |

La etapa de 90 chunks se detuvo durante la época 72, dejando 71 completas.
Al ampliar la escalera empeoraron loss y recall; la accuracy de best pasó de
99,93 % en 10 chunks a 97,98 % en la etapa parcial de 90.
Esto no permite concluir que añadir datos sea perjudicial: cambian la población
de evaluación, su heterogeneidad y el historial acumulado de entrenamiento.

Se revisó además la composición de los chunks con un análisis auxiliar. Se
plantearon diferencias entre datasets y efectos del orden como explicaciones;
no se realizó una intervención controlada que aislase una única causa.

## 5. Barajar el orden de los chunks en cada época: shuffled

Se creó una variante que cambia únicamente el orden de visita TRAIN de los
chunks por época, de forma determinista con semilla y época. Se mantuvieron
el subconjunto de chunks de cada etapa y las muestras de evaluación fijas.
La primera etapa comenzó desde cero; las siguientes heredaron su propio best.

| Chunks | Épocas completadas | Época best | TEST loss | Accuracy | Recall | ROC-AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 100 | 93 | 0.002078 | 99.9502 % | 99.9269 % | 0.999995 |
| 20 | 100 | 95 | 0.002680 | 99.9121 % | 99.8550 % | 0.999994 |
| 30 | 100 | 95 | 0.003949 | 99.8973 % | 99.8422 % | 0.999974 |
| 40 | 100 | 96 | 0.006852 | 99.8031 % | 99.7665 % | 0.999947 |
| 50 | 100 | 77 | 0.009908 | 99.6910 % | 99.6000 % | 0.999930 |
| 60 | 8 | 8 | 0.039510 | 98.5424 % | 98.0706 % | 0.999097 |

La etapa de 60 se detuvo durante la época 9; sus valores proceden de solo
8 épocas completas y no equivalen a un entrenamiento de 100 épocas.

Comparación de loss best con y sin shuffled, ambas etapas completas:

| Chunks | Sin shuffled | Con shuffled | Reducción relativa de loss |
|---:|---:|---:|---:|
| 10 | 0.002926 | 0.002078 | 29.0 % |
| 20 | 0.003898 | 0.002680 | 31.3 % |
| 30 | 0.005635 | 0.003949 | 29.9 % |
| 40 | 0.010749 | 0.006852 | 36.3 % |
| 50 | 0.015396 | 0.009908 | 35.6 % |

La mejora fue consistente en 10–50 chunks. En 40 chunks los errores pasaron de
456 a 280; en 50, de 903 a 584. El efecto incluye los pesos acumulados desde
etapas anteriores; no son cinco experimentos independientes desde cero.

Se observó que muchos best aparecían cerca de la época 100. Se planteó prolongar
hasta 200 épocas, pero no se ejecutó esa comparación. Los mínimos tienden a
coincidir con el final de los ciclos cosine de 20 épocas.

## 6. StandardScaler: 2 chunks, prueba de 10 y cálculo de todos los chunks

El scaler original se ajustaba con todas las filas TRAIN de 2 chunks físicos,
independientes de los chunks seleccionados para cada etapa del MLP.
Se preparó una comparación con 10 chunks; se terminó retirando su carpeta al
optar por calcular el scaler completo. No se conservan aquí cifras verificables
de esa comparación 2 frente a 10 y no se inventa una conclusión cuantitativa.

El scaler completo terminó con 14.503 chunks y **1.159.882.361 filas TRAIN**.
No utilizó filas de TEST ni VALIDATION para estimar medias y escalas.

| Diferencia frente al scaler de 2 chunks | Primary, 1.024 variables | Secondary, 768 variables |
|---|---:|---:|
| Desplazamiento mediano de medias, en sigma del scaler completo | 0,110 | 0,215 |
| Diferencia mediana relativa de escala | 5,11 % | 4,70 % |
| Diferencia máxima relativa de escala | 28,87 % | 18,65 % |
| Variables con diferencia de escala mayor del 10 % | 169 —16,5 %— | 76 —9,9 %— |
| Diferencia RMS entre valores normalizados sobre todo TRAIN | 0,179 | 0,236 |

Cambian las medias y escalas de las 1.792 características. La diferencia
justificaba probar el entrenamiento, pero no demostraba de antemano una mejora.
El scaler completo es el que usa la ejecución actual.

## 7. Escalera shuffled con el scaler completo

Se creó una copia del entrenamiento shuffled que carga el scaler completo sin
calcularlo y usa una carpeta independiente. La etapa de 10 comenzó desde cero.

| Chunks | Épocas completadas | Época best | TEST loss | Accuracy | Recall | ROC-AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 100 | 96 | 0.003006 | 99.9336 % | 99.9070 % | 0.999988 |
| 20 | 100 | 95 | 0.002944 | 99.9167 % | 99.8581 % | 0.999996 |
| 30 | 20 | 20 | 0.010867 | 99.6532 % | 99.4496 % | 0.999915 |

Las etapas de 10 y 20 completaron 100 épocas. La de 30 se detuvo durante la
época 21, con 20 completas. Frente al shuffled con scaler de 2 chunks:

- 10 chunks: loss 0,002078 → 0,003005; accuracy 99,9502 % → 99,9336 %.
- 20 chunks: loss 0,002680 → 0,002944; accuracy 99,9121 % → 99,9167 %.

Por tanto, **no se observó una mejora uniforme al usar el scaler completo**:
la loss fue mayor en las dos etapas comparables; la accuracy mejoró ligeramente
en 20. Una semilla y diferencias pequeñas no establecen superioridad general.
No se compara la etapa parcial de 30 como si estuviese terminada.

## 8. Entrenamiento directo con los 14.503 chunks

Se lanzó un entrenamiento desde cero con LayerNorm, shuffled, scaler completo,
100 épocas objetivo y todos los chunks desde la primera época. TRAIN ONLINE
seguía siendo 50/50. Se añadieron TRAIN EVAL y TEST EVAL tras cada chunk sobre
muestras globales fijas de 58.012 filas: 29.006 positivas y 29.006 negativas.
Se eligió este monitor para seguir la misma población mientras pasaban más datos.

La primera versión conservaba índices y muestras en RAM: se observaron unos
10 GiB durante preparación. Se detuvo y se cambió a lectura por lotes y cachés
en disco. Más adelante se eliminó la acumulación global de logits/predicciones,
manteniendo contadores de la época y AUC exacto en los monitores. Se observaron
aproximadamente 1–1,7 GiB de RAM y unos 563 MiB de VRAM en entrenamiento;
son mediciones puntuales, no un límite garantizado.

Inicialmente se conservó una evaluación de todas las filas balanceadas al final
de época. Tras terminar los 14.503 chunks de la época 1, esa fase seguía ocupando
aproximadamente un día sin mensajes de progreso. Se detuvo, se recuperó desde
el chunk 14.500, se repitieron los últimos 3 y se sustituyó por los monitores
fijos ya calculados. Así se guardaron best/last y pudo arrancar la época 2.

| Momento | TEST loss | Accuracy | Recall | Especificidad | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Cierre de época 1, checkpoint guardado | 0,649533 | 61,47 % | 56,53 % | 66,42 % | 0,665027 |
| Último monitor antes de detener, época 2/chunk 601 | 0,645867 | 62,06 % | 70,67 % | 53,45 % | 0,670063 |

Se detuvo con 1 época completa y un checkpoint recuperable hasta el chunk 600
de la época 2. El best guardado corresponde a la época 1: no se guardaba best
por cada chunk. No se completaron 100 épocas en esta prueba.

Interpretación: aprendió respecto al inicio, pero el rendimiento global se
estabilizó aproximadamente en AUC 0,67, con recall/especificidad oscilantes.
No se alcanzaron los ~99 % de las pruebas pequeñas; esas evaluaciones no tienen
la misma población ni el mismo número de épocas, por lo que no son comparables
directamente. Se descartó continuar por coste y rendimiento insuficiente para
justificar ese experimento, no por un fallo de ejecución demostrado.

## 9. Rendimiento, concurrencia y cambios operativos

Se inspeccionaron GPU, CPU, disco y RAM. Los tres entrenamientos simultáneos
usaban GPU distintas. No se observó saturación global de CPU; había bastante
trabajo de sistema/gestión de datos y baja actividad instantánea de GPU.

Se detuvieron las otras dos escaleras para comprobar interferencia. En la
muestra corta posterior hubo chunks de 7–9 segundos, pero también un pico de
69 segundos; no se obtuvo una comparación controlada suficiente para atribuir
una aceleración sostenida a eliminar la concurrencia.

Las estimaciones por época variaron según el tramo: alrededor de 40–79 horas,
más la evaluación global final cuando existía. No eran tiempos medidos de
100 épocas completas. El campo de 23 segundos en el cierre recuperado de la
época 1 mide únicamente el segmento repetido, no la época entera.

Se añadieron checkpoints atómicos, estado aleatorio PyTorch/CUDA, recuperación
intraépoca, separación de best/last/recovery y limpieza de historiales duplicados
tras reanudar. Las comprobaciones automáticas validaron balanceo, muestras fijas,
reproducibilidad de la evaluación y selección correcta del best. Son pruebas
del software, no experimentos adicionales de calidad predictiva.

## 10. Experimento actual: escalera de 100 en 100 y evaluación doble

Etapas previstas: 100, 200, 300… 14.500 y 14.503; 100 épocas por etapa.
Se mantienen LayerNorm, shuffled, scaler completo, cosine y el resto de la
configuración. La primera etapa empieza desde cero; la siguiente recibe el
best de TEST STAGE de la anterior, reiniciando optimizador/scheduler.

Se evalúan cuatro conjuntos fijos 50/50 después de cada chunk:

| Evaluación, etapa actual de 100 chunks | Positivas | Negativas | Total |
|---|---:|---:|---:|
| TRAIN EVAL STAGE —100 chunks— | 29.006 | 29.006 | 58.012 |
| TEST EVAL STAGE —100 chunks— | 28.873 | 28.873 | 57.746 |
| TRAIN EVAL ALL —14.503 chunks— | 29.006 | 29.006 | 58.012 |
| TEST EVAL ALL —14.503 chunks— | 29.006 | 29.006 | 58.012 |

Las muestras ALL se reutilizan entre etapas. Las STAGE permanecen fijas dentro
de cada etapa y cambian al ampliar los chunks. Se seleccionan filas sin repetición
con cuotas por chunk; no se recorren todas las filas ni se pondera necesariamente
cada chunk según su tamaño. TEST STAGE selecciona best; ALL solo diagnostica.
TRAIN ALL incluye chunks todavía no entrenados en esa etapa, no es solo error
sobre los ejemplos ya usados para ajustar pesos.

Instantánea al recopilar datos: 22 épocas completas; época 23/chunk 38 en curso.
El mejor cierre de época por TEST STAGE loss es el 20:

| Evaluación, época 20 | Loss | Accuracy | Recall | Especificidad | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| TRAIN STAGE | 0.317686 | 85.44 % | 80.49 % | 90.39 % | 0.936854 |
| TEST STAGE | 0.333323 | 84.83 % | 79.38 % | 90.29 % | 0.930583 |
| TRAIN ALL | 1.079969 | 54.03 % | 22.47 % | 85.60 % | 0.590878 |
| TEST ALL | 1.072312 | 54.28 % | 22.78 % | 85.77 % | 0.593038 |

El TEST local pasó de loss 0,672914, accuracy 58,26 % y AUC 0,634662 en época 1
a loss 0,333323, accuracy 84,83 % y AUC 0,930583 en época 20. Sin embargo,
el TEST ALL pasó de loss 0,686453 y AUC 0,611053 a loss 1,072312 y AUC 0,593038.

Este es un resultado relevante: **mejorar dentro de los 100 chunks no está
implicando mejorar sobre todos los chunks**. En esa época el recall global fue
22,78 %, con especificidad 85,77 %: hay una tendencia a clasificar como negativos
muchos positivos del conjunto global. No se explica solamente por diferencia
TRAIN frente a TEST local; puede reflejar especialización al subconjunto o
cambios de distribución entre chunks. No se ha aislado causalmente el mecanismo.

Tras el reinicio cosine en época 21 empeoró temporalmente TEST STAGE: loss
0,426520 y AUC 0,882531; en época 22 volvió a mejorar a 0,388790 y 0,907267.
No se ha completado la primera etapa ni comenzado la de 200 en esta instantánea.

## 11. Propuestas que todavía no equivalen a experimentos realizados

- Comparar una y dos GPU con el mismo batch global: propuesto, no ejecutado como
  comparación de calidad. Usar GPU distintas para ejecuciones independientes no
  es entrenar un modelo con dos GPU.
- Evaluar o entrenar con prevalencia natural, sin balancear: discutido; finalmente
  se pidió mantener 50/50. No hay resultados de una comparación balanceado/natural.
- Extender una etapa a 200 épocas: propuesto al ver best tardíos; no realizado.
- Reducir el learning rate o actualizar cosine por minibatch/chunk: hipótesis,
  no una prueba independiente con resultados disponibles.
- Mezclar/intercalar minibatches de varios chunks con un búfer pequeño: propuesto
  para investigar adaptación al chunk reciente; no implementado como experimento.
- Seleccionar checkpoints con VALIDATION: recomendado para una evaluación final
  independiente; se mantuvo TEST STAGE por decisión del usuario.

## 12. Conclusiones acumuladas y límites

1. El diagnóstico de BatchNorm mostró un problema fuerte al cambiar las
   estadísticas de inferencia; LayerNorm mejoró claramente el ensayo de 10 chunks.
2. Shuffled mejoró la loss en las etapas completas comparables de 10–50 chunks.
3. El scaler completo cambió apreciablemente la normalización, pero no mejoró
   uniformemente las métricas de las primeras etapas frente al scaler de 2 chunks.
4. Los resultados casi perfectos de subconjuntos pequeños no se trasladaron al
   entrenamiento directo ni a la evaluación sobre todos los chunks.
5. La evaluación doble actual distingue aprendizaje local de rendimiento global:
   de momento hay una mejora local notable y una transferencia global limitada.
6. Se ha reducido considerablemente el uso de RAM y el coste del cierre de época;
   esto mejora viabilidad, pero no corrige por sí solo la calidad predictiva.
7. No hay comparación multisemilla, experimento controlado de LR/mezcla de chunks,
   prueba de calidad con DDP ni evaluación independiente por grupos nuevos.

## 13. Fuentes conservadas

- `archive/legacy_experiments_2026-10-26.tar.gz`: EXPERIMENTS.md, escaleras antiguas,
  historiales por época y archivo del experimento con todos los chunks.
- `archive/auxiliary_tools_and_scaler_analysis.tar.gz`: comparación de scalers,
  conteos/procedencia y herramientas de comprobación.
- Carpeta actual `chunk_ladder100_layernorm_cosine_shuffled_allscaler_dualeval_100epochs_1GPU/`:
  historial por época, monitor por chunk, logs y checkpoints activos.

Los historiales contienen más decimales que el log. Los números se presentan
redondeados. Las etapas interrumpidas están identificadas y no se extrapolan
sus resultados a 100 épocas. Que un archivo auxiliar se haya archivado o
eliminado no implica por sí mismo que el experimento fracasara científicamente.
