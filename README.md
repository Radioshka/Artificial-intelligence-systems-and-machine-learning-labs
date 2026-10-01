# Penguin Regression + PCA

Учебный ML-проект по линейной регрессии, регуляризации и PCA.

## Датасет

Используется локальный CSV-файл `data/penguins.csv`, предоставленный для этой лабораторной работы.

В проекте **не требуется участвовать в каких-либо Kaggle-соревнованиях** и не требуется отправлять решения на Kaggle.

### Целевая переменная

`body_mass_g` — масса тела пингвина в граммах.

### Признаки

Числовые:
- `culmen_length_mm`
- `culmen_depth_mm`
- `flipper_length_mm`

Категориальные:
- `species`
- `island`
- `sex`

Целевая переменная `species` намеренно не используется как target, поскольку тогда задача была бы классификационной. В данной лабораторной требуется регрессия с RMSE, R² и MAPE, поэтому целевой является `body_mass_g`.

## Что выполняет программа

1. Загружает CSV.
2. Показывает размер датасета и пропуски.
3. Строит распределения признаков и target.
4. Строит scatter plots и boxplot массы по видам.
5. Строит корреляционную матрицу.
6. Рассчитывает VIF для числовых предикторов.
7. Делит данные на train/test в пропорции 80/20.
8. Обрабатывает пропуски.
9. Кодирует категориальные признаки через One-Hot Encoding.
10. Обучает:
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
11. Использует 5-fold cross-validation.
12. Рассчитывает:
    - RMSE
    - R²
    - MAPE
13. Выполняет стандартизацию всех преобразованных признаков и PCA.
14. Сохраняет компоненты, объясняющие не менее 95% дисперсии.
15. Повторяет Linear/Ridge/Lasso после PCA.
16. Сравнивает результаты до и после PCA.

## Структура

```text
penguin-regression-pca/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── data/
│   └── penguins.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── visualization.py
│   └── pca_analysis.py
├── results/
│   ├── figures/
│   ├── correlation_matrix.csv
│   ├── vif.csv
│   ├── results_before_pca.csv
│   ├── results_after_pca.csv
│   └── results.csv
└── report/
    └── lab1_linear_regression_pca.md
```

## Запуск

Создать виртуальное окружение:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

Запустить:

```bash
python main.py
```

После запуска результаты появятся в `results/`.

## Важный момент про PCA

PCA выполняется после:
1. заполнения пропусков;
2. One-Hot Encoding категориальных переменных;
3. стандартизации всех полученных признаков.

Все преобразования обучаются только на тренировочной выборке, чтобы не допускать утечки информации из test set.
