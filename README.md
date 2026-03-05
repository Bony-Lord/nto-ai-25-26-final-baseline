# Baseline Pipeline (ADR-001)

Бейзлайн реализован как модульный офлайн-пайплайн из 5 стадий:
`prepare_data -> build_features -> generate_candidates -> rank_and_select -> make_submission`.

## Данные

Ожидаемая структура `data/`:

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

## Запуск (сценарий 1)

```bash
uv run python -m src.cli run --config configs/base.yaml
```

Результат:
- создаются артефакты в `artifacts/`
- итоговый файл `artifacts/submission.csv`

## Перезапуск конкретного шага (сценарий 2)

```bash
uv run python -m src.cli run --config configs/base.yaml --stage generate_candidates
```

Поведение:
- будут выполнены зависимости выбранной стадии
- готовые стадии с совпадающим fingerprint пропускаются

## Локальная валидация (сценарий 3)

```bash
uv run python -m src.cli validate --config configs/base.yaml
```

Вывод:
- `mean_ndcg@20`
- квантили (`q25`, `q50`, `q75`)

## Примечания

- Логи запуска пишутся в `logs/`.
- Метаданные кэша шагов пишутся в `artifacts/_meta/`.
- При любой записи артефактов используется атомарный `os.replace`.