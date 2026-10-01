# Системы искусственного интеллекта и машинное обучение

# Лабораторная работа №1

## Линейная регрессия и факторный анализ

### Цель работы

Изучить применение методов линейной регрессии для прогнозирования числового целевого признака, исследовать взаимосвязи между факторами, выявить мультиколлинеарность и оценить влияние снижения размерности методом главных компонент (PCA) на качество моделей.

### Задачи

1. Загрузить и исследовать исходный набор данных.
2. Провести первичный анализ данных и визуализацию распределений признаков.
3. Выполнить предварительную обработку данных.
4. Исследовать корреляции между признаками.
5. Проверить наличие мультиколлинеарности с помощью VIF.
6. Построить модели:

   * Linear Regression;
   * Ridge Regression;
   * Lasso Regression.
7. Оценить качество моделей по метрикам RMSE, R² и MAPE.
8. Выполнить снижение размерности с помощью PCA.
9. Повторно обучить модели на главных компонентах.
10. Сравнить результаты до и после применения PCA.

---

# 1. Введение

Линейная регрессия является одним из базовых методов машинного обучения для решения задач регрессии. Она позволяет описывать зависимость целевой переменной от одного или нескольких факторов.

В данной работе рассматривается задача прогнозирования стоимости жилого дома на основе его характеристик. В качестве набора данных используется **House Prices — Advanced Regression Techniques**, опубликованный на Kaggle.

Целевой переменной является `SalePrice` — фактическая цена продажи дома.

В работе дополнительно рассматриваются методы регуляризации **Ridge** и **Lasso**, которые позволяют уменьшить влияние мультиколлинеарности и предотвратить переобучение модели.

Для снижения размерности признакового пространства применяется **PCA (Principal Component Analysis)** — метод главных компонент.

---

# 2. Описание набора данных

Для исследования используется набор данных **House Prices — Advanced Regression Techniques**.

Источник:

[Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

Основной файл:

```text
train.csv
```

Целевая переменная:

```text
SalePrice
```

Она представляет собой стоимость продажи жилого объекта.

Набор данных содержит характеристики домов, среди которых:

* площадь участка;
* площадь жилого помещения;
* количество комнат;
* качество отделки;
* количество гаражных мест;
* год постройки;
* год реконструкции;
* площадь подвала;
* характеристики кухни;
* характеристики гаража;
* характеристики района и другие параметры.

---

# 3. Импорт библиотек

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings("ignore")
```

---

# 4. Загрузка данных

Файл `train.csv` должен находиться в той же директории, что и ноутбук.

```python
df = pd.read_csv("train.csv")

df.head()
```

Проверим размер набора данных:

```python
print("Размер набора данных:", df.shape)
```

Проверим общую информацию:

```python
df.info()
```

Получим статистические характеристики числовых признаков:

```python
df.describe().T
```

---

# 5. Первичный анализ данных

Проверим количество пропущенных значений:

```python
missing = df.isnull().sum()

missing = missing[missing > 0].sort_values(ascending=False)

missing
```

В наборе данных присутствует большое количество пропущенных значений. При этом пропуск может означать не только отсутствие информации, но и отсутствие соответствующего объекта, например гаража, бассейна или подвала.

Для подготовки данных в дальнейшем используются стратегии заполнения пропусков:

* медианой — для числовых признаков;
* наиболее частым значением — для категориальных признаков.

---

# 6. Анализ целевой переменной

Целевой переменной является `SalePrice`.

Построим распределение стоимости домов:

```python
plt.figure(figsize=(10, 6))

sns.histplot(df["SalePrice"], kde=True)

plt.title("Распределение стоимости домов")
plt.xlabel("SalePrice")
plt.ylabel("Количество объектов")

plt.show()
```

Также построим boxplot:

```python
plt.figure(figsize=(10, 4))

sns.boxplot(x=df["SalePrice"])

plt.title("Boxplot целевой переменной")
plt.xlabel("SalePrice")

plt.show()
```

### Интерпретация

Распределение `SalePrice` является правосторонне асимметричным: большая часть объектов находится в относительно среднем ценовом диапазоне, однако присутствуют дорогие объекты, формирующие длинный правый хвост.

Наличие выбросов может оказывать влияние на линейные модели, поэтому их следует учитывать при интерпретации результатов.

---

# 7. Анализ числовых признаков

Рассмотрим распределение нескольких наиболее важных числовых признаков.

```python
numeric_features_for_plot = [
    "GrLivArea",
    "OverallQual",
    "YearBuilt",
    "TotalBsmtSF",
    "GarageCars"
]

df[numeric_features_for_plot].hist(
    figsize=(12, 8),
    bins=30
)

plt.suptitle("Распределение основных числовых признаков")

plt.show()
```

---

# 8. Визуализация зависимости факторов от целевой переменной

## 8.1. Общая жилая площадь

```python
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="GrLivArea",
    y="SalePrice"
)

plt.title("Зависимость стоимости от жилой площади")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")

plt.show()
```

Можно наблюдать положительную зависимость: увеличение жилой площади в целом связано с увеличением стоимости объекта.

---

## 8.2. Качество объекта

```python
plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="OverallQual",
    y="SalePrice"
)

plt.title("Зависимость стоимости от качества дома")
plt.xlabel("OverallQual")
plt.ylabel("SalePrice")

plt.show()
```

Признак `OverallQual` также демонстрирует выраженную связь с ценой дома.

---

# 9. Корреляционный анализ

Для числовых признаков рассчитывается матрица корреляций.

```python
numeric_df = df.select_dtypes(include=np.number)

corr_matrix = numeric_df.corr()
```

Визуализируем матрицу:

```python
plt.figure(figsize=(16, 12))

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0
)

plt.title("Корреляционная матрица")

plt.show()
```

Получим признаки с наиболее высокой корреляцией с `SalePrice`:

```python
target_corr = (
    corr_matrix["SalePrice"]
    .sort_values(ascending=False)
)

target_corr.head(15)
```

### Интерпретация

Наиболее заметная связь с целевой переменной наблюдается у характеристик, связанных с:

* общим качеством дома;
* жилой площадью;
* площадью подвала;
* площадью гаража;
* количеством гаражных мест;
* возрастом и состоянием объекта.

Однако высокая корреляция признака с целевой переменной не означает причинно-следственную связь.

---

# 10. Подготовка данных

Идентификатор объекта `Id` не несёт полезной информации для прогнозирования и удаляется.

```python
data = df.drop(columns=["Id"])
```

Разделим признаки и целевую переменную:

```python
X = data.drop(columns=["SalePrice"])
y = data["SalePrice"]
```

Определим числовые и категориальные признаки:

```python
numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

print("Числовых признаков:", len(numeric_features))
print("Категориальных признаков:", len(categorical_features))
```

---

# 11. Разделение на обучающую и тестовую выборки

Используем разделение:

* 80% — обучение;
* 20% — тестирование.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)
```

---

# 12. Предварительная обработка признаков

Для числовых признаков:

1. пропущенные значения заменяются медианой;
2. выполняется стандартизация.

Для категориальных признаков:

1. пропуски заменяются наиболее частым значением;
2. выполняется One-Hot Encoding.

```python
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            )
        )
    ]
)
```

Создадим общий `ColumnTransformer`:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)
```

Использование `Pipeline` позволяет избежать утечки данных между обучающей и тестовой выборками.

---

# 13. Метрики качества

Для оценки моделей используются три метрики.

## RMSE

Root Mean Squared Error:

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat y_i)^2
}
$$

Чем меньше RMSE, тем лучше модель.

---

## R²

Коэффициент детерминации:

$$
R^2 =
1 -
\frac{
\sum (y_i-\hat y_i)^2
}{
\sum (y_i-\bar y)^2
}
$$

Чем ближе значение к 1, тем большую долю вариации целевой переменной объясняет модель.

---

## MAPE

Средняя абсолютная процентная ошибка:

$$
MAPE =
\frac{100\%}{n}
\sum_{i=1}^{n}
\left|
\frac{y_i-\hat y_i}{y_i}
\right|
$$

Чем меньше MAPE, тем лучше качество прогноза.

---

# 14. Функция оценки модели

```python
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    mape = mean_absolute_percentage_error(
        y_test,
        y_pred
    ) * 100

    return {
        "RMSE": rmse,
        "R2": r2,
        "MAPE (%)": mape
    }
```

---

# 15. Линейная регрессия

Создадим Pipeline:

```python
linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)
```

Обучение:

```python
linear_model.fit(X_train, y_train)
```

Оценка:

```python
linear_results = evaluate_model(
    linear_model,
    X_test,
    y_test
)

linear_results
```

---

# 16. Ridge Regression

Ridge Regression использует L2-регуляризацию.

Обучим модель с подбором параметра `alpha`.

```python
ridge_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge())
    ]
)
```

Зададим сетку параметров:

```python
ridge_params = {
    "model__alpha": [
        0.01,
        0.1,
        1,
        10,
        100,
        1000
    ]
}
```

Используем 5-кратную кросс-валидацию:

```python
ridge_search = GridSearchCV(
    ridge_pipeline,
    ridge_params,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

ridge_search.fit(X_train, y_train)
```

Лучшее значение параметра:

```python
print(
    "Лучший alpha:",
    ridge_search.best_params_
)
```

Оценка модели:

```python
ridge_results = evaluate_model(
    ridge_search.best_estimator_,
    X_test,
    y_test
)

ridge_results
```

---

# 17. Lasso Regression

Lasso использует L1-регуляризацию.

```python
lasso_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", Lasso(max_iter=20000))
    ]
)
```

Сетка параметров:

```python
lasso_params = {
    "model__alpha": [
        0.0001,
        0.001,
        0.01,
        0.1,
        1,
        10
    ]
}
```

Обучение с кросс-валидацией:

```python
lasso_search = GridSearchCV(
    lasso_pipeline,
    lasso_params,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

lasso_search.fit(X_train, y_train)
```

Лучший параметр:

```python
print(
    "Лучший alpha:",
    lasso_search.best_params_
)
```

Оценка:

```python
lasso_results = evaluate_model(
    lasso_search.best_estimator_,
    X_test,
    y_test
)

lasso_results
```

---

# 18. Сравнение моделей до PCA

Сформируем таблицу результатов:

```python
results_before_pca = pd.DataFrame(
    [
        linear_results,
        ridge_results,
        lasso_results
    ],
    index=[
        "Linear Regression",
        "Ridge",
        "Lasso"
    ]
)

results_before_pca
```

Визуализация RMSE:

```python
results_before_pca["RMSE"].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("RMSE моделей до PCA")
plt.ylabel("RMSE")
plt.xticks(rotation=0)

plt.show()
```

Визуализация R²:

```python
results_before_pca["R2"].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("R² моделей до PCA")
plt.ylabel("R²")
plt.xticks(rotation=0)

plt.show()
```

Визуализация MAPE:

```python
results_before_pca["MAPE (%)"].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("MAPE моделей до PCA")
plt.ylabel("MAPE (%)")
plt.xticks(rotation=0)

plt.show()
```

### Место для результатов

> **Заполнить после запуска ноутбука:**
> Здесь необходимо привести таблицу с фактическими значениями RMSE, R² и MAPE для трёх моделей.

---

# 19. Проверка мультиколлинеарности

Для оценки мультиколлинеарности используется **VIF (Variance Inflation Factor)**.

VIF рассчитывается по формуле:

$$
VIF_j =
\frac{1}{1-R_j^2}
$$

где $R_j^2$ — коэффициент детерминации при регрессии $j$-го признака на остальные признаки.

Чем выше VIF, тем сильнее признак связан с другими факторами.

Для анализа первоначально рассмотрим числовые признаки.

```python
X_numeric = X[numeric_features].copy()

imputer = SimpleImputer(strategy="median")

X_numeric_imputed = imputer.fit_transform(
    X_numeric
)

X_numeric_imputed = pd.DataFrame(
    X_numeric_imputed,
    columns=numeric_features
)
```

Расчёт VIF:

```python
vif_data = pd.DataFrame()

vif_data["Feature"] = X_numeric_imputed.columns

vif_data["VIF"] = [
    variance_inflation_factor(
        X_numeric_imputed.values,
        i
    )
    for i in range(
        X_numeric_imputed.shape[1]
    )
]

vif_data.sort_values(
    "VIF",
    ascending=False
).head(20)
```

### Интерпретация

Высокие значения VIF указывают на наличие мультиколлинеарности между некоторыми признаками.

Например, признаки, связанные с площадями помещений, количеством комнат или другими характеристиками одного и того же объекта, могут быть сильно взаимосвязаны.

Мультиколлинеарность может приводить к нестабильности коэффициентов обычной линейной регрессии.

Ridge частично решает эту проблему за счёт L2-регуляризации, а Lasso дополнительно может занулять некоторые коэффициенты.

---

# 20. Метод главных компонент PCA

Для снижения размерности используется метод главных компонент.

PCA преобразует исходное пространство признаков в новое пространство взаимно некоррелированных компонент.

Перед применением PCA признаки должны быть стандартизированы.

Сначала получим обработанные данные:

```python
X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)
```

Поскольку после One-Hot Encoding матрица может быть разреженной, преобразуем её в плотный массив:

```python
X_train_processed = X_train_processed.toarray()
X_test_processed = X_test_processed.toarray()
```

---

# 21. Полный PCA

Сначала построим PCA со всеми возможными компонентами:

```python
pca_full = PCA()

X_train_pca_full = pca_full.fit_transform(
    X_train_processed
)
```

Получим объяснённую дисперсию:

```python
explained_variance = (
    pca_full.explained_variance_ratio_
)

cumulative_variance = np.cumsum(
    explained_variance
)
```

---

# 22. Scree plot

Построим график объяснённой дисперсии:

```python
plt.figure(figsize=(12, 6))

plt.plot(
    range(
        1,
        len(explained_variance) + 1
    ),
    explained_variance
)

plt.xlabel("Номер главной компоненты")
plt.ylabel("Доля объяснённой дисперсии")

plt.title("Scree plot")

plt.show()
```

---

# 23. Накопленная объяснённая дисперсия

```python
plt.figure(figsize=(12, 6))

plt.plot(
    range(
        1,
        len(cumulative_variance) + 1
    ),
    cumulative_variance
)

plt.axhline(
    0.95,
    linestyle="--"
)

plt.xlabel("Количество компонент")
plt.ylabel("Накопленная объяснённая дисперсия")

plt.title(
    "Накопленная объяснённая дисперсия PCA"
)

plt.show()
```

Определим количество компонент, сохраняющих не менее 95% исходной дисперсии:

```python
n_components = np.argmax(
    cumulative_variance >= 0.95
) + 1

print(
    "Количество компонент для 95% дисперсии:",
    n_components
)
```

### Место для результата

> **Количество главных компонент:** `___`
>
> **Доля объяснённой дисперсии:** `___ %`

---

# 24. PCA с выбранным количеством компонент

Создадим PCA:

```python
pca = PCA(
    n_components=n_components
)

X_train_pca = pca.fit_transform(
    X_train_processed
)

X_test_pca = pca.transform(
    X_test_processed
)
```

Проверим размерность:

```python
print(
    "Исходное количество признаков:",
    X_train_processed.shape[1]
)

print(
    "Количество признаков после PCA:",
    X_train_pca.shape[1]
)
```

Таким образом, PCA позволяет заменить исходное большое количество признаков меньшим числом независимых компонент.

---

# 25. Линейная регрессия после PCA

```python
linear_pca = LinearRegression()

linear_pca.fit(
    X_train_pca,
    y_train
)

linear_pca_results = evaluate_model(
    linear_pca,
    X_test_pca,
    y_test
)

linear_pca_results
```

---

# 26. Ridge после PCA

```python
ridge_pca = GridSearchCV(
    Ridge(),
    {
        "alpha": [
            0.01,
            0.1,
            1,
            10,
            100,
            1000
        ]
    },
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

ridge_pca.fit(
    X_train_pca,
    y_train
)
```

Оценка:

```python
ridge_pca_results = evaluate_model(
    ridge_pca.best_estimator_,
    X_test_pca,
    y_test
)

ridge_pca_results
```

Лучший параметр:

```python
print(
    "Лучший alpha:",
    ridge_pca.best_params_
)
```

---

# 27. Lasso после PCA

```python
lasso_pca = GridSearchCV(
    Lasso(max_iter=20000),
    {
        "alpha": [
            0.0001,
            0.001,
            0.01,
            0.1,
            1,
            10
        ]
    },
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

lasso_pca.fit(
    X_train_pca,
    y_train
)
```

Оценка:

```python
lasso_pca_results = evaluate_model(
    lasso_pca.best_estimator_,
    X_test_pca,
    y_test
)

lasso_pca_results
```

Лучший параметр:

```python
print(
    "Лучший alpha:",
    lasso_pca.best_params_
)
```

---

# 28. Сравнение моделей после PCA

```python
results_after_pca = pd.DataFrame(
    [
        linear_pca_results,
        ridge_pca_results,
        lasso_pca_results
    ],
    index=[
        "Linear Regression + PCA",
        "Ridge + PCA",
        "Lasso + PCA"
    ]
)

results_after_pca
```

---

# 29. Итоговое сравнение

Объединим результаты:

```python
all_results = pd.concat(
    [
        results_before_pca,
        results_after_pca
    ]
)

all_results
```

---

## 29.1. Сравнение RMSE

```python
all_results["RMSE"].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Сравнение RMSE до и после PCA"
)

plt.ylabel("RMSE")
plt.xticks(rotation=45, ha="right")

plt.show()
```

---

## 29.2. Сравнение R²

```python
all_results["R2"].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Сравнение R² до и после PCA"
)

plt.ylabel("R²")
plt.xticks(rotation=45, ha="right")

plt.show()
```

---

## 29.3. Сравнение MAPE

```python
all_results["MAPE (%)"].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Сравнение MAPE до и после PCA"
)

plt.ylabel("MAPE (%)")
plt.xticks(rotation=45, ha="right")

plt.show()
```

---

# 30. Итоговая таблица

После выполнения всех ячеек должна быть получена таблица следующего вида:

| Модель                  |  RMSE |    R² | MAPE (%) |
| ----------------------- | ----: | ----: | -------: |
| Linear Regression       | `...` | `...` |    `...` |
| Ridge                   | `...` | `...` |    `...` |
| Lasso                   | `...` | `...` |    `...` |
| Linear Regression + PCA | `...` | `...` |    `...` |
| Ridge + PCA             | `...` | `...` |    `...` |
| Lasso + PCA             | `...` | `...` |    `...` |

---

# 31. Анализ результатов

При анализе необходимо обратить внимание на следующие аспекты:

1. Как изменился RMSE после применения PCA?
2. Как изменился коэффициент R²?
3. Изменилось ли значение MAPE?
4. Сколько признаков было в исходном пространстве?
5. Сколько компонент осталось после PCA?
6. Какую долю исходной дисперсии сохранили выбранные компоненты?
7. Повлияло ли снижение размерности на качество моделей?
8. Как регуляризация Ridge и Lasso повлияла на результаты?
9. Какие признаки имели высокий VIF?
10. Удалось ли PCA устранить линейную зависимость между преобразованными компонентами?

### Место для анализа

> В исходном пространстве было **___** признаков.
> После применения PCA осталось **___** главных компонент, которые объясняют **___%** дисперсии.
>
> До применения PCA значения RMSE, R² и MAPE составляли соответственно **___**, **___** и **___**.
>
> После применения PCA показатели составили **___**, **___** и **___**.
>
> Таким образом, применение PCA **[улучшило / ухудшило / практически не изменило]** качество прогнозирования. Полученный результат можно объяснить тем, что PCA уменьшает размерность признакового пространства и устраняет линейную корреляцию между компонентами, однако при этом часть информации исходных признаков может быть потеряна.

---

# 32. Вывод

В ходе лабораторной работы был исследован набор данных о стоимости жилых домов и построены модели машинного обучения для прогнозирования цены объекта.

На первом этапе был проведён первичный анализ данных, изучены распределения признаков и целевой переменной, а также построена корреляционная матрица.

Для подготовки данных были обработаны пропущенные значения, категориальные признаки преобразованы методом One-Hot Encoding, а числовые признаки стандартизированы.

Были построены три модели регрессии:

* Linear Regression;
* Ridge Regression;
* Lasso Regression.

Качество моделей оценивалось по метрикам RMSE, R² и MAPE с использованием разделения данных на обучающую и тестовую выборки и кросс-валидации при подборе параметров регуляризации.

Дополнительно был проведён анализ мультиколлинеарности с использованием VIF. Он показал наличие взаимосвязей между некоторыми исходными признаками.

Для снижения размерности был применён метод главных компонент PCA. Количество компонент выбиралось таким образом, чтобы сохранить не менее 95% исходной дисперсии.

После PCA модели были обучены повторно, а их результаты сопоставлены с результатами моделей, построенных на исходном признаковом пространстве.

### Итоговые результаты

> **Количество исходных признаков:** `___`
> **Количество компонент PCA:** `___`
> **Сохранённая дисперсия:** `___%`
>
> **Лучшая по RMSE модель до PCA:** `___`
> **Лучшая по RMSE модель после PCA:** `___`
>
> **RMSE до PCA:** `___`
> **RMSE после PCA:** `___`
>
> **R² до PCA:** `___`
> **R² после PCA:** `___`
>
> **MAPE до PCA:** `___%`
> **MAPE после PCA:** `___%`

---

# 33. Источники

1. Kaggle — House Prices: Advanced Regression Techniques
   https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

2. Scikit-learn — Linear Regression
   https://scikit-learn.org/stable/modules/linear_model.html

3. Scikit-learn — Ridge Regression
   https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression

4. Scikit-learn — Lasso Regression
   https://scikit-learn.org/stable/modules/linear_model.html#lasso

5. Scikit-learn — Principal Component Analysis
   https://scikit-learn.org/stable/modules/decomposition.html#pca

6. Scikit-learn — Cross-validation
   https://scikit-learn.org/stable/modules/cross_validation.html

7. Statsmodels — Variance Inflation Factor
   https://www.statsmodels.org/
