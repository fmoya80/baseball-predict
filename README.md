# baseball-predict

Proyecto de análisis y predicción pregame para partidos de MLB.

El objetivo de este proyecto es construir un pipeline de datos que permita:

1. Obtener información histórica y futura de partidos MLB.
2. Construir features pregame para cada enfrentamiento.
3. Mostrar contexto de cada partido en un dashboard de Streamlit.
4. Preparar la base para modelos predictivos de carreras y posibles ganadores.

---

## Objetivo del proyecto

`baseball-predict` busca transformar datos de MLB en una vista útil antes de cada partido, integrando:

- contexto ofensivo reciente de ambos equipos
- contexto del pitcher abridor
- calendario de juegos
- dataset pregame consolidado
- visualización simple en Streamlit
- futura capa de modelado predictivo

La idea central es que, para cada partido programado, exista una ficha previa que permita entender el matchup y, más adelante, estimar carreras esperadas o probabilidad de victoria.

---

## Alcance actual

Actualmente el proyecto está orientado a construir la base de datos pregame y dejar una primera versión operativa del dashboard.

### Enfoque del MVP
- consumir datos desde MLB Stats API
- construir tablas incrementales
- actualizar la información de forma diaria
- mostrar partidos y contexto relevante en Streamlit
- dejar preparado el terreno para modelos predictivos

---

## Roadmap general

### Fase 1 — Data Pipeline
Construcción de datasets base:

- `games_schedule`
- `team_batting_logs`
- `starter_logs`
- `pregame_dataset` o tabla consolidada de features

### Fase 2 — Visualización
Dashboard en Streamlit para:

- ver partidos por fecha
- comparar ambos equipos
- comparar ofensiva reciente vs pitcher abridor
- explorar métricas pregame

### Fase 3 — Modelado
Construcción de un modelo predictivo para:

- predecir carreras anotadas por equipo
- estimar ganador probable
- evaluar potencial uso de variables externas en el futuro

---

## Stack

- **Python**
- **pandas**
- **requests**
- **Streamlit**
- **MLB Stats API**
- **GitHub**
- **Railway**

---

## Estructura esperada del proyecto

```bash
baseball-predict/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── master/
│
├── src/
│   ├── config.py
│   ├── update_master.py
│   ├── schedule.py
│   ├── team_batting.py
│   ├── starter_logs.py
│   ├── pregame_builder.py
│   └── utils.py
│
└── notebooks/