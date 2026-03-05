# ARCHITECTURE: TWO-ZONE

Архитектура baseline разделена на две зоны с жесткой границей ответственности.

## Zones

- `src/competition/solution/*` — editable зона участника.
- `src/platform/*` — техническое ядро организаторов.

## Принцип разделения

- `competition` отвечает за **логику решения**:
  - features
  - candidate generators
  - ranking
  - solution-level validation
- `platform` отвечает за **выполнение и надежность**:
  - CLI
  - orchestration
  - cache/fingerprint
  - IO
  - строгий контракт сабмита

## Import policy

Разрешено:

- `src.platform.pipeline.*` -> `src.competition.solution.*`
- `src.platform.pipeline.*` -> `src.platform.core.*`
- `src.platform.pipeline.*` -> `src.platform.infra.*`

Запрещено:

- `src.competition.solution.*` -> `src.platform.pipeline.*`
- любые legacy-импорты из `src.pipeline`, `src.core`, `src.participants`, `src.io`, `src.utils`

## Migration map (old -> new)

- `src/participants/*` -> `src/competition/solution/*`
- `src/candidates/*` -> `src/competition/solution/generators/*`
- `src/ranking/*` -> `src/competition/solution/ranking.py`
- `src/pipeline/*` -> `src/platform/pipeline/*`
- `src/core/*` -> `src/platform/core/*`
- `src/io/*` + `src/utils/*` -> `src/platform/infra/*`
- `src/cli.py` -> `src/platform/cli/entrypoint.py`
- `src/core/validate.py` -> `src/platform/core/submission_contract.py`

## Каноничный CLI

Используется только:

`python -m src.platform.cli.entrypoint`

