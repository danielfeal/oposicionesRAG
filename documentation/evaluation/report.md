# Evaluación del RAG

Generado: 2026-09-07T11:08:23.292453+00:00


## Dataset

| exam   |   off_topic |   standard |   with_options |   with_ref |
|:-------|------------:|-----------:|---------------:|-----------:|
| A2     |           0 |          9 |              9 |          8 |
| C1     |           0 |          9 |              9 |          8 |
| C2     |           0 |          8 |              8 |          8 |
| OT     |          10 |          0 |              0 |          0 |


## Retrieval

Hit Rate@k del retriever, sin ninguna reordenación posterior.

|   k |   n_rows |   hit_rate |
|----:|---------:|-----------:|
|   1 |       76 |   0.605263 |
|   2 |       76 |   0.855263 |
|   3 |       76 |   0.894737 |
|   4 |       76 |   0.907895 |
|   5 |       76 |   0.921053 |
|  10 |       76 |   0.934211 |
|  20 |       76 |   0.947368 |

### Por tipo de pregunta

| question_type   |   k |   n_rows |   hit_rate |
|:----------------|----:|---------:|-----------:|
| standard        |   1 |       26 |   0.576923 |
| standard        |   2 |       26 |   0.846154 |
| standard        |   3 |       26 |   0.923077 |
| standard        |   4 |       26 |   0.923077 |
| standard        |   5 |       26 |   0.961538 |
| standard        |  10 |       26 |   0.961538 |
| standard        |  20 |       26 |   0.961538 |
| with_options    |   1 |       26 |   0.538462 |
| with_options    |   2 |       26 |   0.884615 |
| with_options    |   3 |       26 |   0.923077 |
| with_options    |   4 |       26 |   0.923077 |
| with_options    |   5 |       26 |   0.923077 |
| with_options    |  10 |       26 |   0.961538 |
| with_options    |  20 |       26 |   0.961538 |
| with_ref        |   1 |       24 |   0.708333 |
| with_ref        |   2 |       24 |   0.833333 |
| with_ref        |   3 |       24 |   0.833333 |
| with_ref        |   4 |       24 |   0.875    |
| with_ref        |   5 |       24 |   0.875    |
| with_ref        |  10 |       24 |   0.875    |
| with_ref        |  20 |       24 |   0.916667 |


## Generación (LLM como juez: gemini-3.7-flash)

Puntuaciones de 0 a 3. Las filas que el pipeline no llegó a responder puntúan 0 y siguen contando en el denominador.

- Filas evaluadas: **76**
- Puntuadas a cero por fallo del pipeline: **3**
- Correctness: **2.53** / 3 (0.842)

| grupo                      |   n_rows |   n_zeroed |   0 |   1 |   2 |   3 |   avg |
|:---------------------------|---------:|-----------:|----:|----:|----:|----:|------:|
| question_type=standard     |       26 |          1 |   1 |   1 |   1 |  23 |  2.77 |
| question_type=with_options |       26 |          1 |   5 |   0 |   0 |  21 |  2.42 |
| question_type=with_ref     |       24 |          1 |   4 |   1 |   1 |  18 |  2.38 |
| total                      |       76 |          3 |  10 |   2 |   2 |  62 |  2.53 |

- Faithfulness: **2.84** / 3 (0.947)

| grupo                      |   n_rows |   n_zeroed |   0 |   1 |   2 |   3 |   avg |
|:---------------------------|---------:|-----------:|----:|----:|----:|----:|------:|
| question_type=standard     |       26 |          1 |   1 |   0 |   0 |  25 |  2.88 |
| question_type=with_options |       26 |          1 |   2 |   0 |   0 |  24 |  2.77 |
| question_type=with_ref     |       24 |          1 |   1 |   0 |   0 |  23 |  2.88 |
| total                      |       76 |          3 |   4 |   0 |   0 |  72 |  2.84 |

### Motivos de puntuación cero

|                                                                                                                                                                                                                                                                              |   count |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------:|
| judge_error=ClientError("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Resource exhausted. Please try again later. Please refer to https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more details.', 'status': 'RESOURCE_EXHAUSTED'}}") |       3 |

### Estados del pipeline

|    |   count |
|:---|--------:|
| OK |      76 |


## Query processing (filtrado off-topic)

Clase positiva: off-topic. `UNKNOWN` cuenta como *predicho relevante*, igual que en producción (fail-open).

|                    | pred. off-topic | pred. relevante |
|--------------------|-----------------|-----------------|
| **real off-topic** | 4 | 6 |
| **real relevante** | 0 | 76 |

- Precision: **1.000**
- Accuracy: **0.930**
- Veredictos `UNKNOWN`: **0**


## Latencia

Etapas del pipeline completo, solo sobre las filas que alcanzan cada una:

| stage      |   n |   avg_s |   max_s |
|:-----------|----:|--------:|--------:|
| processing |  86 |    0.75 |    1.29 |
| retrieve   |  76 |    0.88 |    2.1  |
| generate   |  76 |    2.12 |   13.67 |
| total      |  76 |    3    |   14.31 |
