# ONBOARDING: как улучшать baseline быстро и безопасно

Этот файл объясняет, где участнику выгодно тратить время, а где лучше не трогать код на старте.

## 1) Карта baseline

- `src/competition/solution/` — главная зона участника (features/generators/ranking/validation).
- `configs/experiments/` — конфиги экспериментов.
- `src/platform/pipeline/orchestrator.py` и `src/platform/pipeline/stages/` — техническая оркестрация.
- `src/platform/pipeline/workflows/local_validation.py` — `PseudoIncidentValidationWorkflow`.
- `src/platform/core/submission_contract.py` — строгая проверка формата `submission.csv`.
- `configs/system.yaml` — системные параметры (пути, логи).
- `tests/` — минимальные тесты контрактов и smoke-сценарий.

## 2) Что менять в первую очередь (high impact)

1. **Новые генераторы кандидатов** в `src/competition/solution/generators/`.
2. **Регистрация генераторов** в `src/competition/solution/generators/registry.py`.
3. **Параметры эксперимента** в `configs/experiments/*.yaml`.
4. **Весы источников** в `ranking.source_weights`.
5. **Ширина кандидатов**: `candidates.per_generator_k`.

Это обычно дает лучший прирост качества при минимальном риске сломать пайплайн.

## 3) Куда не лезть сначала

- Механизм кэширования и атомарной записи (`src/platform/core/artifacts.py`, `src/platform/infra/hashing.py`).
- Контракт итогового сабмита и его валидация (`src/platform/core/submission_contract.py`).
- Базовая схема оркестрации и стадий в `src/platform/pipeline/`, если нет явной причины.

Причина: эти части отвечают за стабильность и воспроизводимость baseline.

## 4) Мини-гайд: добавить генератор за 10-15 минут

### Шаг 1. Создать файл генератора

Пример `src/competition/solution/generators/my_generator.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from src.platform.core.dataset import Dataset


class MyGenerator:
    name = "my_generator"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def generate(
        self,
        dataset: Dataset,
        user_ids: np.ndarray,
        features: pd.DataFrame,
        k: int,
        seed: int,
    ) -> pd.DataFrame:
        rows: list[dict[str, int | float | str]] = []
        # TODO: заполните rows своей логикой
        return pd.DataFrame(rows, columns=["user_id", "edition_id", "score", "source"])
```

### Шаг 2. Зарегистрировать генератор

Добавить его в реестр `src/competition/solution/generators/registry.py`:

```python
GENERATOR_REGISTRY["my_generator"] = (
    lambda params, tqdm_enabled: MyGenerator(alpha=float(params.get("alpha", 1.0)))
)
```

### Шаг 3. Подключить в конфиг

В `configs/experiments/<your_exp>.yaml`:

```yaml
candidates:
  generators:
    - name: my_generator
      params:
        alpha: 1.0
```

### Шаг 4. Проверить контракт

Генератор обязан возвращать колонки:
- `user_id`
- `edition_id`
- `score`
- `source` (везде равен `name`)

### Шаг 5. Прогнать тесты и запуск

- `uv run pytest -q`
- `uv run python -m src.platform.cli.entrypoint run --config configs/experiments/<your_exp>.yaml --stage generate_candidates`

## 5) Быстрый checklist перед экспериментом

- Изменения ограничены генераторами и конфигом.
- Контракт генератора не нарушен.
- Сабмит проходит строгую валидацию.
- Повторный запуск корректно использует cache-hit.

## 6) Что делать, если мало времени

1. Не трогать ядро пайплайна.
2. Добавить 1-2 сильных генератора.
3. Подобрать `source_weights` и `per_generator_k`.
4. Проверять метрики через `validate`.

## Что делать дальше

Вернитесь в [`README.md`](README.md) как в навигационную точку и фиксируйте лучшие конфиги в `configs/experiments/`.
