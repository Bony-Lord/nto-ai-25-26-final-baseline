Ваш запрос
/kaggle/input/datasets/artemnazemtsev/nto-ai/authors.csv Пример пути, решение не раотает выведи в однйо ячеекей .py кодом, надо пробить 0.1 Описание данных (Финал)

Схема данных

см. файл data_description.md в репозитории бейзлайна

Описание файлов

interactions.csv

Наблюдаемый лог взаимодействий пользователей после частичной потери событий. Каждая строка — одно событие.

FieldDescriptionCommentuser_idидентификатор пользователя (FK → users.user_id)-edition_idидентификатор издания (FK → editions.edition_id)-event_typeтип события: 1 — wishlist, 2 — read-ratingрейтинг (только для read, иначе NULL)-event_tsдата-время события-

targets.csv

Список пользователей, для которых нужно восстановить потерянные позитивные взаимодействия.

FieldDescriptionCommentuser_idидентификатор пользователяодин столбец

editions.csv

Справочник изданий (edition-level). Каждое издание привязано к одной книге.

FieldDescriptionCommentedition_idидентификатор издания (PK)-book_idидентификатор книги-author_idидентификатор автора (FK → authors.author_id)-publication_yearгод публикациииздания, не книгиage_restrictionвозрастное ограничениенапример, 18+language_idидентификатор языкасправочник отсутствуетpublisher_idидентификатор издателясправочник отсутствуетtitleназвание-descriptionописаниетекст

authors.csv

Справочник авторов.

FieldDescriptionCommentauthor_idидентификатор автора (PK)-author_nameимя автораФИО или псевдоним

genres.csv

Справочник жанров.

FieldDescriptionCommentgenre_idидентификатор жанра (PK)-genre_nameназвание жанра-

book_genres.csv

Связь многие-ко-многим между книгами и жанрами.

FieldDescriptionCommentbook_idидентификатор книги-genre_idидентификатор жанра (FK → genres.genre_id)-

users.csv

Справочник пользователей с демографическими признаками.

FieldDescriptionCommentuser_idидентификатор пользователя (PK)-genderпол: 1, 2 или NULL-ageвозраст (может быть NULL)- Сгегнерирцуй решение с нуля, следи за логикой чтобы скомпилипровалось Промпт для генерации топового решения

Role: Act as a Senior Machine Learning Engineer and Kaggle Grandmaster specializing in Recommender Systems (RecSys).

Task: Design and implement a high-performance two-stage recommendation pipeline for a book/edition recommendation task. The goal is to maximize NDCG@20 and MRR@20.

Dataset Context: - interactions.csv (user_id, edition_id, event_type [1: click/wish, 2: read], event_ts).



editions.csv (edition_id, author_id, title, description, text_data).

targets.csv (user_id for prediction).

Constraints & Requirements:



Validation Strategy: Implement a robust Time-Split Cross-Validation (sliding window). Avoid data leakage by ensuring the training set only contains interactions strictly prior to the validation window.

Stage 1: Multi-Source Retrieval (Recall):

Collaborative Filtering: Weighted Matrix Factorization (ALS) via implicit library with exponential time decay.

Item-to-Item: Co-occurrence/Co-visitation matrix (Items bought together) using sparse matrix multiplication ($A^T A$).

User-Author Affinity: Retrieve new editions from authors the user has previously interacted with.

Content-Based: TF-IDF + TruncatedSVD (64-128 components) on text_data with cosine similarity.

Trend-Based: Top-K popular items within the last 7/14 days.

Stage 2: Ranking (Feature Engineering):

User Features: Activity density, mean interaction weight, preference stability (entropy of genres/authors).

Item Features: Global popularity, recent momentum (slope of sales), age restriction encoding, author's average rating/popularity.

Affinity (Cross) Features: User-Author interaction count, User-Genre overlap, time since the user's last interaction with this specific author.

Temporal Features: Log-transformed "Time since last seen", "Time since item release".

Model: Use CatBoostRanker.

Loss function: YetiRank or NDCG:top=20.

Optimization: Implement Hard Negative Mining (using candidates that the model almost ranked high but were false).

Technical Specs: * Use memory_efficient data types (int32, float32).

Handle Cold Start using a popularity-based fallback.

Ensure strict grouping by group_id (user_id) for CatBoost Pool.

Output: Provide a clean, modular Python script using pandas, scipy.sparse, implicit, and catboost. Include a function for inference that generates a submission.csv with user_id, edition_id, and rank (1-20).  Для того чтобы выжать максимум из задачи RecSys (рекомендательных систем), тебе нужен промпт, который заставит нейронку мыслить категориями SOTA (State-of-the-Art) решений.Этот промпт составлен на языке Senior ML-инженера. Он структурирован так, чтобы модель не просто «написала код», а спроектировала двухстадийную систему ранжирования с продвинутым фиче-инжинирингом.Промпт для генерации топового решенияRole: Ты — Senior Machine Learning Engineer и Kaggle Grandmaster, специализирующийся на рекомендательных системах (RecSys). Твоя задача — спроектировать и реализовать архитектуру для задачи предсказания книжных рекомендаций (NDCG@20).Objective: Построить гибридную двухстадийную систему (Retrieval + Ranking), которая минимизирует проблему холодного старта и максимизирует точность ранжирования.Data Schema:interactions.csv (user_id, edition_id, event_type [1: click, 2: read], event_ts).editions.csv (edition_id, author_id, title, description, text_data).targets.csv (user_id для инференса).Requirements & Architecture:Validation Framework:Реализуй Time-Split Cross-Validation (sliding window). Обучение строго на данных до момента времени $T$, валидация — на окне $[T, T+7 \text{ дней}]$. Никакого data leakage.Stage 1: Multi-Source Retrieval (Recall):Collaborative Filtering: Weighted Matrix Factorization (ALS) через библиотеку implicit. Используй exponential time decay для весов взаимодействий.Item-to-Item (I2I): Рассчитай матрицу со-встречаемости (Co-occurrence matrix) через векторное перемножение разреженных матриц ($A^T A$).Content-Based: Используй предобученные эмбеддинги (или TF-IDF + SVD) для text_data для поиска семантически близких книг.Affinity Retrieval: Обязательный реколл по «Любимым авторам» (новые книги авторов, которых пользователь читал ранее).Stage 2: Feature Engineering (Ranking):User Features: Интенсивность потребления контента, средний вес взаимодействия, энтропия интересов по авторам.Item Features: Глобальная популярность, «импульс» (рост популярности за последние 3 дня), средний возраст читателя.Cross-Features (Affinity): Количество книг данного автора в истории пользователя, время с последнего взаимодействия с этим автором/жанром.Temporal Features: Логарифм времени с последнего клика пользователя, «свежесть» книги (время с даты выхода).Model & Training:Модель: CatBoostRanker.Loss Function: YetiRank (или NDCG:top=20).Обязательно внедри Hard Negative Mining: выбирай в качестве негативных примеров те айтемы, которые выдал Retrieval, но которых нет в покупках.Примени строгую группировку по user_id для Pool.Optimization:Используй memory_efficient типы данных (int32, float32).Реализуй автоматическую «добивку» (fallback) популярными айтемами для «холодных» пользователей, отсутствующих в Recall-стадии.Output: Напиши чистый, модульный код на Python. Код должен включать классы для Recall-моделей, функцию генерации признаков и финальный цикл инференса. Финал. Профиль ИИ НТО 2025-2026

Необходимо восстановить наиболее вероятные потерянные позитивные взаимодействия пользователей с книгами, основываясь на неполных логах и метаданных.



Принимая участие в соревновании, вы соглашаетесь с правилами проведения финала профиля ИИ НТО



Финал НТО ИИ. Кейс «Потеряшки»

1. Описание кейса

1.1. Легенда

Ночной релиз книжного сервиса прошёл с ошибкой. Во время миграции схемы событий часть продовых воркеров начала писать логи в новую схему, а часть осталась на старой. Из-за ошибки в консьюмере и дедупликации часть позитивных событий не попала в итоговое хранилище.



В результате история взаимодействий пользователей стала неполной. Это ухудшает персонализацию, рекомендации и продуктовую аналитику.



Ваша роль — команда рекомендательной системы, которой нужно восстановить наиболее вероятные потерянные позитивные взаимодействия по наблюдаемым данным.



1.2. Что требуется от решения

Для каждого пользователя из targets.csv нужно сформировать ранжированный список из 20 изданий (edition_id), которые с наибольшей вероятностью являются потерянными позитивными взаимодействиями.



Решение заключается в построении ML-пайплайна.



Генерация кандидатов — поиск потенциально релевантных изданий по всему доступному каталогу.

Ранжирование — отбор топ-20 наиболее вероятных потеряшек.

Формирование итогового списка — выдача топ-20 по каждому пользователю.

Подробное описание структуры файлов и полей приведено в разделе "Данные" и data_description.md в репозитории бейзлайна.



1.3. Что считается позитивным взаимодействием и потеряшкой

Позитивные события в логах.



event_type = 1 — добавление в вишлист (wishlist)

event_type = 2 — чтение (read)

Оценивание ведётся на уровне уникальной пары (user_id, edition_id), а не на уровне строк событий.



«Потеряшка» — это скрытая (потерянная) позитивная пара (user_id, edition_id) из окна инцидента. Тип позитивного события (wishlist/read) в решении предсказывать не требуется.



1.4. Что известно о характере потерь

Потеря позитивных взаимодействий частичная: в окне инцидента скрывается порядка 20% позитивных пар (user_id, edition_id).

Потери неравномерны по пользователям и времени.

Возможна различная затронутость типов событий (wishlist и read).

Точный протокол формирования скрытых потеряшек не раскрывается.

1.5. Временные окна

Для формирования кейса используется период в 214 дней, в нём есть Окно инцидента (от 2025-10-01 00:00:00 до 2025-11-01 00:00:00) — период, в котором произошел сбой логирования и часть событий была потеряна.



Участники получают наблюдаемый лог за весь период (180 дней), где в окне инцидента применены искусственные потери.



1.6. Контекст задачи

Неполные логи — реалистичная проблема продовых систем. Потери событий могут возникать из-за ошибок миграции, сбоев ETL, рассинхронизации схем и проблем в контурах доставки сообщений.



В этом кейсе моделируется аварийное восстановление сигналов для рекомендательной системы. Цель — построить воспроизводимый ML-пайплайн, который по наблюдаемым данным восстанавливает наиболее вероятные потерянные позитивные взаимодействия.



Важно. Восстановленные события в реальных системах обычно рассматриваются как модельный слой, а не как замена первичных логов.



2. Оценивание

2.1. Основная метрика

Итоговый скор считается как среднее значение NDCG@20 по всем пользователям теста.





Метрика награждает отдельно за:



попадание в потеряшку

правильное ранжирование

2.2. Предсказание участника и скрытые данные

Для каждого user_id из targets.csv участник предсказывает 20 строк с полями edition_id и rank (1..20).



Организаторы имеют скрытую разметку is_lost ∈ {0,1} для пар (user_id, edition_id) (1 — пара является потеряшкой, 0 — нет).



2.3. Формулы метрики

Определения:



rel(rank) = 1, если издание на позиции rank является потеряшкой для данного пользователя, иначе rel(rank) = 0.



Пусть relevant_items — все скрытые потеряшки пользователя.



В IDCG предполагаем идеальную выдачу, где на первых позициях стоят все потеряшки.






2.4. Примеры подсчёта метрики

Примеры вынесены в отдельный документ: metric_examples.md (см. репозиторий бейзлайна).



2.5. Пояснения к оцениванию

Усреднение выполняется по пользователям.

Метрика оценивает только скрытые потерянные позитивные взаимодействия.

Оценивание ведётся по парам (user_id, edition_id).

Public leaderboard и Private leaderboard используют один и тот же принцип оценивания на разных скрытых подмножествах пользователей.

Разбиение на Public и Private выполняется по пользователям (user_id) в пропорции 50/50.

3. Формат и требования к решению

3.1. Формат сабмита

Необходимо отправить файл submission.csv следующего вида:



user_id,edition_id,rank

3.2. Технические требования

Для каждого user_id из targets.csv должно быть ровно 20 строк.

rank должен быть целым числом от 1 до 20.

Внутри одного user_id значения rank не должны повторяться.

Внутри одного user_id значения edition_id не должны повторяться.

Файл должен содержать предсказания для всех пользователей из targets.csv.

Порядок строк в файле не важен.

3.3. Пример сабмита (sample_submission.csv)

Пример корректного файла решения.



Field Description Comment

user_id идентификатор пользователя -

edition_id восстановленное издание -

rank позиция в топ-20 значения 1..20, уникальны в рамках user_id

Бейзлайн

Расположен по ссылке https://github.com/Orange-Hack/nto-ai-25-26-final-baseline /kaggle/input/datasets/artemnazemtsev/nto-ai/authors.csv Пример пути, решение не раотает выведи в однйо ячеекей .py кодом, надо пробить 0.1 Описание данных (Финал)

Схема данных

см. файл data_description.md в репозитории бейзлайна

Описание файлов

interactions.csv

Наблюдаемый лог взаимодействий пользователей после частичной потери событий. Каждая строка — одно событие.

FieldDescriptionCommentuser_idидентификатор пользователя (FK → users.user_id)-edition_idидентификатор издания (FK → editions.edition_id)-event_typeтип события: 1 — wishlist, 2 — read-ratingрейтинг (только для read, иначе NULL)-event_tsдата-время события-

targets.csv

Список пользователей, для которых нужно восстановить потерянные позитивные взаимодействия.

FieldDescriptionCommentuser_idидентификатор пользователяодин столбец

editions.csv

Справочник изданий (edition-level). Каждое издание привязано к одной книге.

FieldDescriptionCommentedition_idидентификатор издания (PK)-book_idидентификатор книги-author_idидентификатор автора (FK → authors.author_id)-publication_yearгод публикациииздания, не книгиage_restrictionвозрастное ограничениенапример, 18+language_idидентификатор языкасправочник отсутствуетpublisher_idидентификатор издателясправочник отсутствуетtitleназвание-descriptionописаниетекст

authors.csv

Справочник авторов.

FieldDescriptionCommentauthor_idидентификатор автора (PK)-author_nameимя автораФИО или псевдоним

genres.csv

Справочник жанров.

FieldDescriptionCommentgenre_idидентификатор жанра (PK)-genre_nameназвание жанра-

book_genres.csv

Связь многие-ко-многим между книгами и жанрами.

FieldDescriptionCommentbook_idидентификатор книги-genre_idидентификатор жанра (FK → genres.genre_id)-

users.csv

Справочник пользователей с демографическими признаками.

FieldDescriptionCommentuser_idидентификатор пользователя (PK)-genderпол: 1, 2 или NULL-ageвозраст (может быть NULL)- Сгегнерирцуй решение с нуля, следи за логикой чтобы скомпилипровалось Промпт для генерации топового решения

Role: Act as a Senior Machine Learning Engineer and Kaggle Grandmaster specializing in Recommender Systems (RecSys).

Task: Design and implement a high-performance two-stage recommendation pipeline for a book/edition recommendation task. The goal is to maximize NDCG@20 and MRR@20.

Dataset Context: - interactions.csv (user_id, edition_id, event_type [1: click/wish, 2: read], event_ts).



editions.csv (edition_id, author_id, title, description, text_data).

targets.csv (user_id for prediction).

Constraints & Requirements:



Validation Strategy: Implement a robust Time-Split Cross-Validation (sliding window). Avoid data leakage by ensuring the training set only contains interactions strictly prior to the validation window.

Stage 1: Multi-Source Retrieval (Recall):

Collaborative Filtering: Weighted Matrix Factorization (ALS) via implicit library with exponential time decay.

Item-to-Item: Co-occurrence/Co-visitation matrix (Items bought together) using sparse matrix multiplication ($A^T A$).

User-Author Affinity: Retrieve new editions from authors the user has previously interacted with.

Content-Based: TF-IDF + TruncatedSVD (64-128 components) on text_data with cosine similarity.

Trend-Based: Top-K popular items within the last 7/14 days.

Stage 2: Ranking (Feature Engineering):

User Features: Activity density, mean interaction weight, preference stability (entropy of genres/authors).

Item Features: Global popularity, recent momentum (slope of sales), age restriction encoding, author's average rating/popularity.

Affinity (Cross) Features: User-Author interaction count, User-Genre overlap, time since the user's last interaction with this specific author.

Temporal Features: Log-transformed "Time since last seen", "Time since item release".

Model: Use CatBoostRanker.

Loss function: YetiRank or NDCG:top=20.

Optimization: Implement Hard Negative Mining (using candidates that the model almost ranked high but were false).

Technical Specs: * Use memory_efficient data types (int32, float32).

Handle Cold Start using a popularity-based fallback.

Ensure strict grouping by group_id (user_id) for CatBoost Pool.

Output: Provide a clean, modular Python script using pandas, scipy.sparse, implicit, and catboost. Include a function for inference that generates a submission.csv with user_id, edition_id, and rank (1-20). 
