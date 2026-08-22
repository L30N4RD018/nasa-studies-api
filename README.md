# NASA Studies API

API REST construida con **FastAPI** para explorar y sintetizar estudios de
biología espacial del [NASA Open Science Data Repository (OSDR)](https://osdr.nasa.gov/).

Indexa **1119 estudios** repartidos en 15 organismos y 11 tipos de proyecto, y
genera títulos y descripciones para cualquier subconjunto filtrado usando un
**LLM local en formato GGUF** (`llama.cpp`), con caída automática a un método
heurístico cuando el modelo no está disponible.

Proyecto nacido en el NASA Space Apps Challenge 2025.

## Características

- **Búsqueda facetada** por organismo, tipo de proyecto, palabras clave y texto libre.
- **Generación con LLM local** — sin claves de API ni servicios externos: el
  modelo cuantizado (Q5_K_M) corre en CPU vía `llama-cpp-python`.
- **Degradación elegante** — si `llama-cpp-python` o el `.gguf` no están
  presentes, la API arranca igual y responde en *modo heurístico*. Todos los
  endpoints siguen operativos salvo la síntesis por LLM.
- **Corrector ortográfico** y expansión de consultas sobre el vocabulario del corpus.
- **Caché en memoria** de payloads por combinación de filtros.

## Endpoints

| Método | Ruta                  | Descripción                                        |
|--------|-----------------------|----------------------------------------------------|
| GET    | `/health`             | Estado del servicio y nº de estudios cargados      |
| GET    | `/docs`               | Swagger UI (automático de FastAPI)                 |
| GET    | `/facets`             | Valores disponibles para cada faceta               |
| GET    | `/studies`            | Listado paginado con filtros                       |
| GET    | `/studies/{study_id}` | Detalle de un estudio                              |
| POST   | `/studies/search`     | Búsqueda completa (payload con artículos y tópicos)|
| POST   | `/studies/generate`   | Genera título y descripción del subconjunto        |
| GET    | `/spell-check`        | Corrección ortográfica de una consulta             |
| GET    | `/llm/status`         | Disponibilidad y estado de carga del modelo        |
| POST   | `/llm/reload`         | Recarga el modelo en caliente                      |
| POST   | `/reload`             | Recarga el corpus desde disco                      |

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

La API queda en `http://127.0.0.1:8000` y la documentación en `/docs`.

### Con el modelo LLM

El `.gguf` **no se versiona** (pesa ~869 MB). Colócalo en
`app/models/odr_model_q5_k_m.gguf` e instala el runtime:

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

El índice de wheels precompilados evita compilar `llama.cpp` desde fuente.

### Con Docker

```bash
docker build -t nasa-studies-api .
docker run -p 80:80 nasa-studies-api
```

## Variables de entorno

| Variable          | Por defecto        | Descripción                                      |
|-------------------|--------------------|--------------------------------------------------|
| `GGUF_MODEL_PATH` | `/app/app/models/…`| Ruta al modelo `.gguf`                           |
| `LLM_N_THREADS`   | nº de CPUs         | Hilos de inferencia                              |
| `LLM_N_CTX`       | `2048`             | Ventana de contexto                              |
| `LLM_N_BATCH`     | `512`              | Tamaño de batch; bajar si hay presión de memoria |
| `LLM_USE_MLOCK`   | `0`                | `mlock` requiere `CAP_IPC_LOCK`; falla en la mayoría de contenedores |

## Despliegue

Incluye `render.yaml` para [Render](https://render.com). El plan gratuito
(512 MB de RAM) ejecuta la API en modo heurístico sin problema: el consumo
medido en reposo es de ~119 MB.

Para servir el modelo hacen falta ~2 GB de RAM, fuera del alcance de los
planes gratuitos actuales de PaaS.

## Estructura

```
app/
  main.py              # rutas FastAPI y carga del corpus
  models/payload.py    # esquemas de respuesta
  services/
    pipeline.py        # orquestación y caché
    filters.py         # filtrado y expansión de consultas
    ranking.py         # ordenación por relevancia
    generation.py      # LLM GGUF + fallback heurístico
    spell_checker.py   # corrección ortográfica
odr/                   # corpus OSDR en JSON por organismo
config/                # catálogo de tópicos
streamlit_app.py       # frontend de demostración
```

## Licencia

MIT — ver [LICENSE](LICENSE).
