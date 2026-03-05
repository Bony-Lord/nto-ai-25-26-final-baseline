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
   `uv run python -m src.cli run --config configs/experiments/baseline.yaml`
3. Проверьте результат: `artifacts/submission.csv`.

## Что читать дальше

- **Как запускать и дебажить**: [`SETUP.md`](SETUP.md)
- **Как дорабатывать baseline под себя**: [`ONBOARDING.md`](ONBOARDING.md)
- **Какие файлы участнику менять, а какие нет**: [`docs/PARTICIPANT_SURFACE.md`](docs/PARTICIPANT_SURFACE.md)

## Что в репозитории важно

- `src/participants/` — участковая зона (features/generators/ranking/validation).
- `src/candidates/` — реализации генераторов кандидатов.
- `configs/experiments/` — конфиги экспериментов участника.
- `src/pipeline/` — оркестрация и стадии (техническое ядро).
- `src/core/` — кэширование, валидация контракта, логирование.

## Что лучше не трогать и не тратить время

- Не переписывайте логику кэширования и атомарной записи в `src/core/artifacts.py` и `src/io/hashing.py` без реальной необходимости.
- Не меняйте контракт итогового сабмита в `src/core/validate.py`.
- Не рефакторьте `src/pipeline/` ради улучшения метрики: используйте `src/participants/` и `configs/experiments/*.yaml`.
- Не тратьте время на сложный ML “с нуля” в самом начале: сначала выжмите максимум из candidate generation + blending/weights.

## Что делать дальше

Перейдите в [`SETUP.md`](SETUP.md) и выполните команды запуска из раздела «Быстрый запуск».