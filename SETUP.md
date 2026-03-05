# SETUP: запуск и проверка baseline

Этот файл про практику: как быстро запустить baseline, какие есть команды и что делать при типовых ошибках.

## Prerequisites

- Установлен `uv`.
- Доступен Python версии из `pyproject.toml`.
- Вы запускаете команды из корня репозитория.
- В `data/` лежат входные CSV:

```text
data/
  interactions.csv
  targets.csv
  editions.csv
  authors.csv
  genres.csv
  book_genres.csv
  users.csv
```

## Быстрый запуск

Полный прогон:

```bash
uv run python -m src.cli run --config configs/experiments/baseline.yaml
```

Перезапуск с конкретной стадии (с автодогоном зависимостей):

```bash
uv run python -m src.cli run --config configs/experiments/baseline.yaml --stage generate_candidates
```

Локальная валидация (псевдо-инцидент):

```bash
uv run python -m src.cli validate --config configs/experiments/baseline.yaml
```

## CLI: команды и опции

Точка входа: `python -m src.cli`

- `run`
  - `--config` (по умолчанию `configs/base.yaml`; рекомендуемый — `configs/experiments/baseline.yaml`)
  - `--stage` (`prepare_data | build_features | generate_candidates | rank_and_select | make_submission`)
- `validate`
  - `--config` (по умолчанию `configs/base.yaml`; рекомендуемый — `configs/experiments/baseline.yaml`)

## Где смотреть результаты

- Итоговый сабмит: `artifacts/submission.csv`
- Промежуточные артефакты: `artifacts/*.parquet`
- Метаданные запуска:
  - `artifacts/_meta/run.json`
  - `artifacts/_meta/step_status.json`
- Логи: `logs/`

## Типовые проблемы и быстрые фиксы

### 1) Ошибка про отсутствующий файл

Пример: `Pipeline failed: Required file is missing: ...`

Что делать:
- проверить структуру `data/`;
- проверить имя файла и расширение;
- убедиться, что запускаете из корня репозитория.

### 2) Сабмит невалиден по формату

Проверить:
- для каждого `user_id` ровно `k` строк;
- `rank` уникален и в диапазоне `1..k`;
- `edition_id` не повторяется в рамках одного пользователя.

### 3) Шаги пропускаются, но вы ожидали пересчет

Причина: cache-hit по fingerprint.

Что делать:
- изменить входные данные/конфиг или
- удалить соответствующие артефакты в `artifacts/` и запустить снова.

### 4) Долгий запуск

Что смотреть:
- текущую стадию и ETA в stdout;
- детали в `logs/run_*.log`;
- длительность и статистику шагов в `artifacts/_meta/step_status.json`.

## Что делать дальше

После первого успешного запуска переходите в [`ONBOARDING.md`](ONBOARDING.md), чтобы понять, как правильно улучшать baseline.
