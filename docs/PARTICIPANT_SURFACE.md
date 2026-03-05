# PARTICIPANT SURFACE

Этот документ фиксирует границы между зоной участника и техническим ядром baseline.

## Что менять участнику (N файлов)

Основной путь улучшений:

1. `src/participants/features.py` — сборка признаков baseline.
2. `src/participants/generators/registry.py` — регистрация доступных генераторов.
3. `src/candidates/*.py` — реализация самих генераторов.
4. `src/participants/ranking.py` — вызов/настройка ранжирования.
5. `src/participants/validation.py` — participant-уровень проверок.
6. `configs/experiments/*.yaml` — параметры экспериментов.

Этого набора достаточно для большинства улучшений качества.

## Что не менять участнику (M файлов)

Техническое ядро организаторов:

- `src/pipeline/*` — оркестрация, стадии и маршрутизация.
- `src/core/*` — кэш, артефакты, строгая проверка сабмита, логирование.
- `src/io/*` — fingerprinting и файловые адаптеры.
- `scoring.py` — платформенное оценивание.

Эти файлы отвечают за воспроизводимость и стабильность пайплайна.

## Быстрый путь участника

1. Скопировать `configs/experiments/baseline.yaml` в новый конфиг эксперимента.
2. Добавить/изменить генератор в `src/candidates/` и зарегистрировать его.
3. Подкрутить веса и параметры в `configs/experiments/*.yaml`.
4. Проверить локально:
   - `uv run python -m src.cli run --config configs/experiments/<name>.yaml`
   - `uv run python -m src.cli validate --config configs/experiments/<name>.yaml`
