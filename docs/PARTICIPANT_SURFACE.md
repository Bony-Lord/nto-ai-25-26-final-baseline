# PARTICIPANT SURFACE

Этот документ фиксирует границы между зоной участника и техническим ядром baseline.

## Что менять участнику (N файлов)

Основной путь улучшений:

1. `src/competition/solution/features.py` — сборка признаков baseline.
2. `src/competition/solution/generators/registry.py` — регистрация доступных генераторов.
3. `src/competition/solution/generators/*.py` — реализация самих генераторов.
4. `src/competition/solution/ranking.py` — вызов/настройка ранжирования.
5. `src/competition/solution/validation.py` — participant-уровень проверок.
6. `configs/experiments/*.yaml` — параметры экспериментов.

Этого набора достаточно для большинства улучшений качества.

## Что не менять участнику (M файлов)

Техническое ядро организаторов:

- `src/platform/pipeline/*` — оркестрация, стадии и маршрутизация.
- `src/platform/core/*` — кэш, артефакты, строгая проверка сабмита, логирование.
- `src/platform/infra/*` — fingerprinting, IO и инфраструктурные utility.
- `scoring.py` — платформенное оценивание.

Эти файлы отвечают за воспроизводимость и стабильность пайплайна.

## Быстрый путь участника

1. Скопировать `configs/experiments/baseline.yaml` в новый конфиг эксперимента.
2. Добавить/изменить генератор в `src/competition/solution/generators/` и зарегистрировать его.
3. Подкрутить веса и параметры в `configs/experiments/*.yaml`.
4. Проверить локально:
   - `uv run python -m src.platform.cli.entrypoint run --config configs/experiments/<name>.yaml`
   - `uv run python -m src.platform.cli.entrypoint validate --config configs/experiments/<name>.yaml`
