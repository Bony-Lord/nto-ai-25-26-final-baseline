# Baseline Pipeline

Репозиторий содержит воспроизводимый baseline для кейса «Потеряшки»:  
из входных CSV формируется валидный `submission.csv` для `NDCG@20`.

## TL;DR

- Пайплайн состоит из 5 стадий: `prepare_data -> build_features -> generate_candidates -> rank_and_select -> make_submission`.
- На выходе создается `artifacts/submission.csv`.
- Поддерживается перезапуск по стадиям и кэширование шагов через fingerprint входов.
- Прогресс выполнения виден в stdout и в лог-файле текущего запуска.

## Быстрый старт

1. Подготовьте данные в `data/` (см. [`SETUP.md`](SETUP.md)).
2. Запустите полный пайплайн:
   `uv run python -m src.cli run --config configs/base.yaml`
3. Проверьте результат: `artifacts/submission.csv`.

## Что читать дальше

- **Как запускать и дебажить**: [`SETUP.md`](SETUP.md)
- **Как дорабатывать baseline под себя**: [`ONBOARDING.md`](ONBOARDING.md)

## Что в репозитории важно

- `src/pipeline/orchestrator.py` — оркестрация стадий и запуск `PipelineRunner`.
- `src/pipeline/stages/` — отдельные классы стадий (`PrepareDataStage`, `BuildFeaturesStage`, ...).
- `src/pipeline/workflows/local_validation.py` — локальная валидация через псевдо-инцидент.
- `src/candidates/` — генераторы кандидатов (главная зона для улучшений).
- `src/ranking/simple_blend.py` — простое объединение источников.
- `configs/base.yaml` — ключевые параметры запуска и генераторов.

## Что лучше не трогать и не тратить время

- Не переписывайте логику кэширования и атомарной записи в `src/core/artifacts.py` и `src/io/hashing.py` без реальной необходимости.
- Не меняйте контракт итогового сабмита в `src/core/validate.py`: это зона, где проще всего случайно сломать отправку.
- Не начинайте с рефакторинга оркестрации/стадий в `src/pipeline/`: быстрее и эффективнее улучшать генераторы в `src/candidates/` и параметры в `configs/base.yaml`.
- Не тратьте время на сложный ML “с нуля” в самом начале: сначала выжмите максимум из candidate generation + blending/weights.

## Что делать дальше

Перейдите в [`SETUP.md`](SETUP.md) и выполните команды запуска из раздела «Быстрый запуск».