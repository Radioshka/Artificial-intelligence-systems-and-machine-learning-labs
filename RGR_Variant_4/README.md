# РГР, вариант 4. Роль ИИ в финансах и трейдинге

Источник по заданию: Forbes.com, темы Finance & Investments.

## Как повторить расчёт

```
pip install -r requirements.txt
python -u collect.py
python -u fill_bodies.py
python -u collect_extra.py
python -u analyze.py
python -u build_report.py
```

`collect.py` собирает заголовки через RSS Google News с ограничением `site:forbes.com`, восстанавливает адрес статьи на Forbes и читает текст со страницы. Если Forbes отвечает отказом, текст берётся из открытого снимка Wayback Machine.

`collect_extra.py` добавляет аналогичные открытые источники, которые допускает задание: аннотации arXiv (раздел q-fin) и полные статьи TechCrunch. Уже сохранённый более длинный текст Forbes не заменяется.

`analyze.py` делает токенизацию, удаление стоп-слов, стемминг, лемматизацию, считает частоты терминов, TF-IDF и обучает наивный байесовский классификатор.

Ноутбук с тем же расчётом: `rgr_variant4.ipynb`. Его нужно открывать из этой папки.
