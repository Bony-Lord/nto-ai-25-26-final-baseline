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
   `uv run python -m src.platform.cli.entrypoint run --config configs/experiments/baseline.yaml`
3. Проверьте результат: `artifacts/submission.csv`.

## Что читать дальше

- **[Documentation Index](docs/INDEX.md)**: Единая точка навигации по всем документам проекта.
- **Как запускать и дебажить**: [`SETUP.md`](docs/baseline/SETUP.md)
- **Как дорабатывать baseline под себя**: [`ONBOARDING.md`](docs/baseline/ONBOARDING.md)

## Что в репозитории важно

- `src/competition/solution/` — участковая зона (features/generators/ranking/validation).
- `configs/experiments/` — конфиги экспериментов участника.
- `src/platform/pipeline/` — оркестрация и стадии (техническое ядро).
- `src/platform/core/` — кэширование, валидация контракта, логирование.
- `src/platform/infra/` — инфраструктурные IO/utility-компоненты.

## Что лучше не трогать и не тратить время

- Не переписывайте логику кэширования и атомарной записи в `src/platform/core/artifacts.py` и `src/platform/infra/hashing.py` без реальной необходимости.
- Не меняйте контракт итогового сабмита в `src/platform/core/submission_contract.py`.
- Не рефакторьте `src/platform/` ради улучшения метрики: используйте `src/competition/solution/` и `configs/experiments/*.yaml`.
- Не тратьте время на сложный ML “с нуля” в самом начале: сначала выжмите максимум из candidate generation + blending/weights.

## Что делать дальше

Перейдите в [`SETUP.md`](docs/baseline/SETUP.md) и выполните команды запуска из раздела «Быстрый запуск».