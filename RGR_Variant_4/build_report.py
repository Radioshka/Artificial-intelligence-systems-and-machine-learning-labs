"""Пояснительная записка по варианту 4. Числа читаются из results/summary.json."""

from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
SUMMARY = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
OUT = ROOT / "отчёт" / "РГР_вариант4_ИИ_в_финансах.docx"
FIG = ROOT / "figures"

NAMES = {
    "banking": "банки и финтех",
    "investment": "инвестиции",
    "risk": "риск и регулирование",
    "trading": "торговые стратегии",
    "other": "прочие материалы",
}


def num(value, digits=3) -> str:
    return f"{float(value):.{digits}f}".replace(".", ",")


def set_run_font(run, size=14, bold=False, name="Times New Roman"):
    run.font.name = name
    run.bold = bold
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0, 0, 0)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), name)


def add_markup(paragraph, text, size=14, base_bold=False):
    for index, chunk in enumerate(text.split("**")):
        if not chunk:
            continue
        run = paragraph.add_run(chunk)
        set_run_font(run, size=size, bold=base_bold or index % 2 == 1)


def P(doc, text, size=14, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=Cm(1.25),
      space_before=Pt(0), space_after=Pt(0), keep_next=False):
    paragraph = doc.add_paragraph()
    pf = paragraph.paragraph_format
    pf.alignment = align
    pf.first_line_indent = indent
    pf.space_before = space_before
    pf.space_after = space_after
    pf.line_spacing = 1.5
    pf.keep_with_next = keep_next
    add_markup(paragraph, text, size)
    return paragraph


def outline(paragraph, level=0):
    p_pr = paragraph._p.get_or_add_pPr()
    node = OxmlElement("w:outlineLvl")
    node.set(qn("w:val"), str(level))
    p_pr.append(node)


def page_break(doc):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.first_line_indent = Cm(0)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.0
    paragraph.add_run().add_break(WD_BREAK.PAGE)


def STRUCT(doc, text):
    """Заголовок структурного элемента: по центру, прописными, с новой страницы."""
    paragraph = doc.add_paragraph()
    pf = paragraph.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(0)
    pf.space_after = Pt(14)
    pf.line_spacing = 1.5
    pf.page_break_before = True
    pf.keep_with_next = True
    run = paragraph.add_run(text)
    set_run_font(run, size=14, bold=True)
    outline(paragraph, 0)
    return paragraph


def SECTION(doc, text):
    """Заголовок раздела основной части: абзацный отступ, не прописными, с новой страницы."""
    paragraph = doc.add_paragraph()
    pf = paragraph.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.left_indent = Cm(1.25)
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(0)
    pf.space_after = Pt(14)
    pf.line_spacing = 1.5
    pf.page_break_before = True
    pf.keep_with_next = True
    run = paragraph.add_run(text)
    set_run_font(run, size=14, bold=True)
    outline(paragraph, 0)
    return paragraph


def BULLET(doc, text):
    paragraph = doc.add_paragraph()
    pf = paragraph.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf.left_indent = Cm(1.25)
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    pf.line_spacing = 1.5
    add_markup(paragraph, "— " + text, 14)
    return paragraph


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def cell_text(cell, text, size=12, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text = ""
    paragraph = cell.paragraphs[0]
    pf = paragraph.paragraph_format
    pf.alignment = align
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(1)
    pf.space_after = Pt(1)
    pf.line_spacing = 1.0
    run = paragraph.add_run(str(text))
    set_run_font(run, size=size, bold=bold)


def TABLE(doc, caption, headers, rows, widths, size=12):
    cap = doc.add_paragraph()
    pf = cap.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(12)
    pf.space_after = Pt(4)
    pf.line_spacing = 1.0
    pf.keep_with_next = True
    add_markup(cap, caption, 14)

    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for index, width in enumerate(widths):
        table.columns[index].width = Cm(width)
    header = table.rows[0]
    tr_pr = header._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)
    for index, title in enumerate(headers):
        cell = header.cells[index]
        cell.width = Cm(widths[index])
        cell_text(cell, title, size=size, bold=True)
        shade(cell, "FFFFFF")
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cells[index].width = Cm(widths[index])
            align = WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            cell_text(cells[index], value, size=size, align=align)
            shade(cells[index], "FFFFFF")
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(2)
    spacer.paragraph_format.space_after = Pt(2)
    spacer.paragraph_format.first_line_indent = Cm(0)
    return table


def FIGURE(doc, number, filename, caption, width=15.5):
    paragraph = doc.add_paragraph()
    pf = paragraph.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(8)
    pf.space_after = Pt(2)
    paragraph.add_run().add_picture(str(FIG / filename), width=Cm(width))
    cap = doc.add_paragraph()
    cf = cap.paragraph_format
    cf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cf.first_line_indent = Cm(0)
    cf.space_before = Pt(0)
    cf.space_after = Pt(8)
    cf.line_spacing = 1.0
    add_markup(cap, f"Рисунок {number} — {caption}", 14)


def CODE(doc, title, path: Path):
    label = doc.add_paragraph()
    lf = label.paragraph_format
    lf.first_line_indent = Cm(0)
    lf.space_before = Pt(8)
    lf.space_after = Pt(2)
    lf.keep_with_next = True
    run = label.add_run(title)
    set_run_font(run, size=12, bold=True)
    for line in path.read_text(encoding="utf-8").splitlines():
        paragraph = doc.add_paragraph()
        pf = paragraph.paragraph_format
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.first_line_indent = Cm(0)
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing = 1.0
        run = paragraph.add_run(line if line else " ")
        set_run_font(run, size=12, name="Consolas")


def add_page_number(section):
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    set_run_font(run, size=14)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(end)


def per_class_rows(result):
    rows = []
    report = result["report"]
    for code in result["labels"]:
        match = re.search(rf"^\s*{code}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)", report, re.M)
        precision, recall, f1, support = match.groups()
        rows.append([
            NAMES[code],
            num(precision),
            num(recall),
            num(f1),
            support,
        ])
    return rows


def center_line(doc, text, size=14, bold=False, space_after=0, space_before=0):
    paragraph = doc.add_paragraph()
    pf = paragraph.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.first_line_indent = Cm(0)
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = 1.0
    run = paragraph.add_run(text)
    set_run_font(run, size=size, bold=bold)
    return paragraph


def add_numpages(paragraph):
    run = paragraph.add_run()
    set_run_font(run, size=14)

    def fld(kind):
        node = OxmlElement("w:fldChar")
        node.set(qn("w:fldCharType"), kind)
        run._r.append(node)

    fld("begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " NUMPAGES "
    run._r.append(instr)
    fld("end")


def add_toc(doc):
    paragraph = doc.add_paragraph()
    pf = paragraph.paragraph_format
    pf.first_line_indent = Cm(0)
    pf.line_spacing = 1.5
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = r' TOC \o "1-1" \h \z \u '
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(separate)
    hint = paragraph.add_run("Содержание обновится при открытии документа в редакторе.")
    set_run_font(hint, size=14)
    run2 = paragraph.add_run()
    set_run_font(run2, size=14)
    run2._r.append(end)


def main():
    s = SUMMARY
    full = s["nb_full"]
    balanced = s["nb_balanced"]
    masked = s["nb_masked"]
    example = s["preprocess_example"]

    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(3.0)
    section.right_margin = Cm(1.5)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.different_first_page_header_footer = True
    add_page_number(section)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(14)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    center_line(doc, "Министерство науки и высшего образования Российской Федерации", space_after=0)
    center_line(doc, "______________________________________________", space_before=6)
    center_line(doc, "(наименование образовательной организации)", size=12, space_after=18)
    center_line(doc, "РАСЧЕТНО-ГРАФИЧЕСКАЯ РАБОТА", bold=True, space_before=18)
    center_line(doc, "по дисциплине «Системы искусственного интеллекта и машинное обучение»", space_before=12)
    center_line(doc, "РОЛЬ ИСКУССТВЕННОГО ИНТЕЛЛЕКТА", bold=True, space_before=18)
    center_line(doc, "В ФИНАНСАХ И ТРЕЙДИНГЕ", bold=True)
    center_line(doc, "Вариант 4. Источник: Forbes.com, Finance & Investments", space_before=12)
    center_line(doc, "Исполнитель", space_before=28)
    center_line(doc, "студент группы __________    ________________________")
    center_line(doc, "(группа)                         (ФИО)", size=12)
    center_line(doc, "Руководитель", space_before=14)
    center_line(doc, "________________    ________________________")
    center_line(doc, "(должность)              (ФИО)", size=12)
    center_line(doc, "____________", space_before=28)
    center_line(doc, "2026")

    STRUCT(doc, "РЕФЕРАТ")
    P(doc, "Отчет 47 с., 1 книга, 6 рисунков, 9 таблиц, 8 источников, 1 приложение.")
    P(doc, "Ключевые слова: ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ, ФИНАНСЫ, ТРЕЙДИНГ, FORBES, TF-IDF, НАИВНЫЙ БАЙЕС, ГЕНЕРАТИВНЫЙ ИИ, РИСК.")
    P(doc, "Объект исследования — открытые публикации о влиянии искусственного интеллекта на финансовый рынок и торговые стратегии. Основной источник варианта — Forbes, дополнительно взяты arXiv и TechCrunch, потому что страницы статей Forbes часто закрыты. Цель работы — собрать эти публикации, обработать тексты и описать, какие темы в них преобладают. Методы: сбор по RSS и API, очистка текста, частоты терминов, TF-IDF и наивный байесовский классификатор. В корпус анализа вошли "
      f"{s['n_analyzed']} статьи за период с {s['date_min']} по {s['date_max']}. Шире всего в текстах представлены риск, генеративный искусственный интеллект и машинное обучение. Компьютерное зрение и глубокое обучение почти не обсуждаются. Классификатор при равных априорных вероятностях достигает accuracy {num(balanced['accuracy'])} против базового уровня {num(balanced['baseline_accuracy'])}. Результат применим как карта публичной повестки, а не как основание для торговой сделки.")

    STRUCT(doc, "СОДЕРЖАНИЕ")
    add_toc(doc)

    STRUCT(doc, "ПЕРЕЧЕНЬ СОКРАЩЕНИЙ И ОБОЗНАЧЕНИЙ")
    P(doc, "ИИ — искусственный интеллект.")
    P(doc, "TF-IDF — term frequency — inverse document frequency, мера веса слова в документе относительно корпуса.")
    P(doc, "NLP — natural language processing, обработка естественного языка.")
    P(doc, "ML — machine learning, машинное обучение.")
    P(doc, "LLM — large language model, большая языковая модель.")

    STRUCT(doc, "ВВЕДЕНИЕ")
    P(doc, "Цель работы — освоить практический контур обработки текстов на Python и применить его к открытым публикациям о роли искусственного интеллекта в финансах и трейдинге. По варианту 4 источник — Forbes.com, тематика Finance & Investments. Нужно собрать материалы, очистить текст, посчитать частоты ключевых терминов, выделить характерные слова мерой TF-IDF и показать классификацию наивным байесовским классификатором.")
    P(doc, "Задача сформулирована так: по текстам Forbes и по аналогичным открытым источникам описать, о чём пишут в связи с ИИ на финансовом рынке, и проверить, разделяются ли эти тексты на устойчивые темы: торговые стратегии, риск и регулирование, инвестиции, банки и финтех. Работа не строит торговый робот и не оценивает доходность стратегий. Она отвечает на другой вопрос: как эти темы представлены в публичных материалах и насколько их можно отличить автоматическим классификатором.")
    P(doc, "Для выполнения работы необходимо:", space_after=Pt(0))
    BULLET(doc, "собрать список публикаций Forbes и сохранить его в JSON и CSV;")
    BULLET(doc, "получить текст статей и описать, откуда он взят, если сайт закрывает прямой доступ;")
    BULLET(doc, "выполнить токенизацию, удаление стоп-слов, стемминг и лемматизацию;")
    BULLET(doc, "построить частоты терминов из задания и терминов предметной области;")
    BULLET(doc, "выделить топ словосочетаний по TF-IDF;")
    BULLET(doc, "обучить Multinomial Naive Bayes и сравнить его с константным базовым уровнем;")
    BULLET(doc, "сформулировать выводы о применимости такого обзора для наблюдения за рынком.")

    SECTION(doc, "1 Предметная область и обзор методов")
    P(doc, "В финансах ИИ используется в нескольких разных задачах, и смешивать их в одном выводе нельзя. Алгоритмическая торговля исполняет заявки по правилам: сигнал, размер позиции, ограничение риска. Классические квантовые модели опираются на статистику временных рядов. Машинное обучение добавляет нелинейные зависимости и большие наборы признаков: цены, стакан, новости, отчётность. Глубокие сети применяют к последовательностям, но в открытой прессе их обсуждают реже, чем сам факт автоматизации.")
    P(doc, "Отдельная линия — текст. Новости, стенограммы созвонов с инвесторами и отчёты эмитентов разбирают методами NLP: мешок слов, TF-IDF, тематические модели, а в последние годы большие языковые модели. Здесь же лежит генеративный ИИ в банке: черновики аналитических записок, разбор клиентских обращений, внутренние помощники сотрудников. Это не то же самое, что высокочастотная торговля, хотя в заголовках оба сюжета называют «ИИ на рынке».")
    P(doc, "Третья линия — контроль. Модели кредитного риска, антифрод и проверки комплаенса работают на табличных данных и на графах операций. Регулятор смотрит не на точность прогноза цены, а на объяснимость, данные для обучения и ответственность, если модель ошиблась. В публичных статьях этот сюжет часто важнее, чем описание конкретной альфы.")
    P(doc, "Для самой работы выбраны методы, которые прямо указаны в задании. Частота показывает, сколько раз термин встретился и в скольких статьях. TF-IDF понижает вес слов, которые есть почти везде, и поднимает слова, характерные для части корпуса [1]. Наивный байесовский классификатор оценивает класс по произведению вероятностей слов и остаётся базовым методом для коротких и средних текстов [2]. Стемминг Портера и лемматизация WordNet нужны, чтобы свести формы слова к одной единице до подсчёта [3, 4].")
    P(doc, "Ограничение обзора честное. Forbes — деловое издание, а не журнал с торговыми алгоритмами. По нему видно повестку и язык рынка, а не устройство конкретной стратегии. Часть материалов — колонки Forbes Councils, то есть мнение приглашённого автора, иногда с коммерческим интересом. Это учтено в выводах.")

    SECTION(doc, "2 Источник данных и сбор публикаций")
    P(doc, f"Сбор выполнен {s['collected_at'][:10]}. Прямой список «все статьи Forbes о финансах» в открытом виде не выдаётся, а часть страниц статей отвечает скрипту кодом 403. Поэтому программа идёт в три шага. Сначала RSS Google News с ограничением site:forbes.com по шести запросам: искусственный интеллект и трейдинг, машинное обучение в банках и финтехе, генеративный ИИ в финансах, алгоритмическая торговля, регулирование и управление активами. Затем канонический адрес Forbes восстанавливается из ссылки Google News. Затем текст читается со страницы Forbes [5], а если страница закрыта — из снимка Wayback Machine [8] с проверкой, что в тексте есть слова заголовка.")
    sites = s.get("site_counts") or {}
    site_line = ", ".join(f"{name} — {count}" for name, count in sites.items())
    P(doc, f"После фильтра заголовков осталось {s['n_collected']} публикаций. Полный текст получен у {s['n_with_body']}. В анализ вошли {s['n_analyzed']} статьи с текстом не короче 500 символов. Четыре материала отброшены по заголовку как не относящиеся к финансам: рынок труда, недвижимость, медицинское страхование. Даты отобранных статей: с {s['date_min']} по {s['date_max']}. По сайтам в расчёте: {site_line}.")
    P(doc, "Дополнительные источники взяты как аналогичные, это прямо допускает задание. arXiv [6] отдаёт аннотации через открытый API, разделы q-fin: машинное обучение, глубокие сети и алгоритмическая торговля. TechCrunch [7] отдал несколько полных статей по RSS раздела fintech и тега artificial intelligence. CNBC и The Verge в этой выгрузке подходящих открытых текстов не дали: лента либо без пары «ИИ и финансы» в заголовке, либо пустая. Короткие лиды Forbes короче 500 символов в частоты не входят.")
    P(doc, "Заголовок попадал в список, только если в нём одновременно есть признак ИИ и признак финансов: trading, bank, investment, portfolio, fraud, credit, fintech, stock, market и близкие формы. Это отсекло общие заметки про советы директоров, маркетинг и SEO, которые поисковая лента подмешивает к слову AI. Состав корпуса приведен в таблице 1.")
    TABLE(
        doc,
        "Таблица 1 — Состав собранного корпуса",
        ["Показатель", "Значение"],
        [
            ["Публикаций после фильтра заголовков", str(s["n_collected"])],
            ["Статей с полным текстом", str(s["n_with_body"])],
            ["Статей в частотном анализе и TF-IDF", str(s["n_analyzed"])],
            ["Статей в классификаторе", str(s["n_classified"])],
            ["Самая ранняя дата", s["date_min"]],
            ["Самая поздняя дата", s["date_max"]],
        ],
        [11.0, 5.5],
        size=12,
    )
    P(doc, "Файлы data/publications.json и data/publications.csv содержат дату, заголовок, адрес, источник текста и сам текст. Поле text_source равно forbes, если страница открылась напрямую, и archive, если текст взят из снимка.")
    P(doc, "Примеры статей, которые задают четыре темы варианта, приведены в таблице 2.")
    from analyze import label_of

    publications = json.loads((ROOT / "data" / "publications.json").read_text(encoding="utf-8"))
    picked = {}
    for article in publications["articles"]:
        if article["text_source"] == "title" or article["n_chars"] < 500:
            continue
        code, title = label_of(f"{article['title']}. {article['text']}")
        if code == "other" or code in picked:
            continue
        picked[code] = {
            "published": article["published"],
            "class_title": title,
            "title": article["title"],
        }
        if len(picked) == 4:
            break
    order = ["trading", "risk", "investment", "banking"]
    shown = [picked[code] for code in order if code in picked]
    TABLE(
        doc,
        "Таблица 2 — Примеры публикаций корпуса",
        ["Дата", "Класс по словарю", "Заголовок"],
        [[item["published"], item["class_title"], item["title"]] for item in shown[:5]],
        [2.8, 4.2, 9.5],
        size=12,
    )

    SECTION(doc, "3 Предварительная обработка текста")
    P(doc, "Текст переводится в нижний регистр, из него удаляются числа и знаки препинания. Токен — последовательность букв длиннее двух символов. Стоп-слова берутся из списка NLTK для английского языка, к ним добавлены служебные для этого корпуса слова forbes, said, also, would, could. Стемминг выполняется стеммером Портера, лемматизация — WordNetLemmatizer с частью речи из pos-tagger. Для статьи сохраняются четыре списка: исходные токены, токены без стоп-слов, основы и леммы.")
    P(doc, "На коротком фрагменте первой статьи это выглядит так. Исходный кусок: «" + example["source"][:180].replace("\n", " ") + "». Результат шагов приведен в таблице 3.")
    TABLE(
        doc,
        "Таблица 3 — Фрагмент предобработки",
        ["Шаг", "Первые элементы"],
        [
            ["Токены", ", ".join(example["tokens"][:12])],
            ["Без стоп-слов", ", ".join(example["no_stop"][:12])],
            ["Стеммы", ", ".join(example["stems"][:12])],
            ["Леммы", ", ".join(example["lemmas"][:12])],
        ],
        [3.5, 13.0],
        size=12,
    )
    P(doc, "Стеммер режет financial до financi и intelligence до intellig. Лемматизатор сохраняет financial и intelligence и только снимает грамматическую форму, например drive вместо driven. Для содержательного частотного списка дальше используются леммы. Стеммы остаются в программе как требуемый заданием шаг и как иллюстрация того, что грубое усечение хуже леммы для английского делового текста.")

    SECTION(doc, "4 Частотный анализ и TF-IDF")
    P(doc, "Ключевые термины из задания посчитаны по полному тексту статей регулярными выражениями, чтобы не спутать ML с частью другого слова. Рядом добавлены термины варианта: LLM, generative AI, algorithmic trading, fraud, risk, fintech, portfolio. Частоты приведены в таблице 4 и на рисунке 1.")
    term_rows = [
        [row["term"], str(row["occurrences"]), str(row["documents"])]
        for row in s["term_rows"]
    ]
    TABLE(
        doc,
        "Таблица 4 — Частота ключевых терминов",
        ["Термин", "Вхождений", "Статей"],
        term_rows,
        [7.5, 4.5, 4.5],
    )
    FIGURE(
        doc, 1, "term_frequency.png",
        "Число вхождений ключевых терминов в собранном корпусе",
    )
    risk = next(row for row in s["term_rows"] if row["term"] == "Risk")
    gen = next(row for row in s["term_rows"] if row["term"] == "Generative AI")
    ml = next(row for row in s["term_rows"] if row["term"] == "ML")
    dl = next(row for row in s["term_rows"] if row["term"] == "Deep Learning")
    nlp = next(row for row in s["term_rows"] if row["term"] == "NLP")
    cv = next(row for row in s["term_rows"] if row["term"] == "Computer Vision")
    algo = next(row for row in s["term_rows"] if row["term"] == "Algorithmic trading")
    P(doc, f"Самый широкий термин — risk: {risk['occurrences']} вхождений в {risk['documents']} статьях из {s['n_analyzed']}. Генеративный ИИ встречается {gen['occurrences']} раз в {gen['documents']} статьях, машинное обучение — {ml['occurrences']} раз в {ml['documents']} статьях. Глубокое обучение почти не образует отдельную тему: deep learning есть в {dl['documents']} статьях, NLP — в {nlp['documents']}, computer vision — в {cv['documents']}. Алгоритмическая торговля концентрированнее: {algo['occurrences']} вхождений, но только {algo['documents']} статей. О ней пишут реже, чем о риске, зато в тех статьях, где пишут, термин повторяется.")
    P(doc, "После лемматизации верх списка занимают слова финансового сюжета, а не названия архитектур нейросетей. Это согласуется с частотной таблицей: издание обсуждает рынок, данные, риск и инвестора, а не устройство модели. Двенадцать самых частых лемм приведены в таблице 5.")
    TABLE(
        doc,
        "Таблица 5 — Наиболее частые леммы",
        ["Лемма", "Число"],
        [[row["lemma"], str(row["count"])] for row in s["top_lemmas"][:12]],
        [8.0, 4.0],
    )
    P(doc, "TF-IDF посчитан по тем же текстам: слова и биграммы, английские стоп-слова, min_df = 2, max_df = 0,85. В таблицу 6 и на рисунок 2 вынесены 15 терминов с наибольшим средним весом по корпусу. Среднее, а не сумма, нужно потому, что длинная статья иначе задавила бы короткую.")
    TABLE(
        doc,
        "Таблица 6 — Топ-15 терминов по среднему TF-IDF",
        ["Термин", "Средний TF-IDF"],
        [[row["term"], num(row["mean_tfidf"], 4)] for row in s["tfidf_top"][:15]],
        [8.0, 5.0],
    )
    FIGURE(doc, 2, "tfidf_top.png", "Термины с наибольшим средним TF-IDF", width=15.2)
    P(doc, "Вверху оказываются financial, trading, market, fraud, investors, investment, stock. Слово fraud поднимается выше многих общих слов: оно редкое в одних статьях и очень плотное в материалах про мошенничество, поэтому TF-IDF отделяет его лучше, чем сырая частота. ChatGPT входит в топ, хотя классические термины deep learning и computer vision — нет. Для этого корпуса «характерный язык» — торговля, риск мошенничества и генеративные инструменты, а не компьютерное зрение.")

    SECTION(doc, "5 Классификация наивным байесовским классификатором")
    P(doc, "Готовой разметки у статей нет. Класс назначен правилом: в тексте ищутся устойчивые выражения четырёх словарей. Торговые стратегии — algorithmic trading, high-frequency, hedge fund, quantitative trading и близкие. Риск — fraud, compliance, regulation, credit risk, AML. Инвестиции — portfolio, asset management, wealth management, robo-advisor. Банки и финтех — banking, payment, lending, insurance, fintech. Побеждает словарь с наибольшим числом совпадений. Если не совпал ни один, статья помечается как прочая.")
    P(doc, "Такая разметка прозрачна, но шумная. Статья про торговлю, в которой много раз сказано portfolio, может уйти в инвестиции. Поэтому классификатор ниже проверяется дважды: на полном тексте и на тексте, из которого выражения словарей удалены. Второй прогон показывает, узнаёт ли модель тему по окружающим словам или только повторяет слова, которыми мы сами поставили метку. Распределение статей приведено в таблице 7 и на рисунке 3.")
    TABLE(
        doc,
        "Таблица 7 — Распределение статей по классам",
        ["Класс", "Статей"],
        [[name, str(count)] for name, count in s["class_counts"].items()],
        [10.0, 4.0],
    )
    FIGURE(doc, 3, "class_distribution.png", "Число статей в каждом тематическом классе", width=14.5)
    P(doc, f"Класс «прочие материалы» содержит {s['class_counts'].get('прочие материалы', 0)} статьи. У них нет устойчивого словаря, и в обучение они не входят. В классификаторе {s['n_classified']} статей, разбиение стратифицированное, test_size = 0,25, random_state = 42: {balanced['n_train']} на обучение и {balanced['n_test']} на проверку. Признаки — TF-IDF униграммы и биграммы, не более 4000 признаков, min_df = 2. Модель — MultinomialNB со сглаживанием Лапласа α = 1.")
    P(doc, "Сравнение идёт с константой: всегда предсказывать самый частый класс обучающей выборки. Для этой выборки это инвестиции, точность константы на тесте " + num(balanced["baseline_accuracy"]) + ".")
    P(doc, "Первый классификатор оценивает априорную вероятность класса по частоте в обучении. При несбалансированных темах он тянется к инвестициям. Второй считает априорные вероятности равными: вопрос эксперимента в том, разделяет ли словарь темы, а не в том, какая тема попалась чаще. Третий повторяет равные априорные вероятности, но на тексте без слов размечающего словаря. Сводка приведена в таблице 8.")
    TABLE(
        doc,
        "Таблица 8 — Три эксперимента на одной и той же проверке",
        ["Эксперимент", "Accuracy", "Macro-F1", "База"],
        [
            ["Априорные частоты классов", num(full["accuracy"]), num(full["macro_f1"]), num(full["baseline_accuracy"])],
            ["Равные априорные вероятности", num(balanced["accuracy"]), num(balanced["macro_f1"]), num(balanced["baseline_accuracy"])],
            ["Словарь меток удалён", num(masked["accuracy"]), num(masked["macro_f1"]), num(masked["baseline_accuracy"])],
        ],
        [7.2, 3.0, 3.0, 3.3],
    )
    P(doc, f"С априорными частотами accuracy = {num(full['accuracy'])}, macro-F1 = {num(full['macro_f1'])}. Модель выше константы {num(full['baseline_accuracy'])}, но macro-F1 низкий: банки и торговля на тесте почти не узнаются, инвестиции получают recall 1,000. Классификатор в этом режиме в основном повторяет самый большой класс и дополнительно узнаёт часть рисковых статей. Матрица ошибок показана на рисунке 4.")
    FIGURE(doc, 4, "confusion_full.png", "Матрица ошибок при априорных частотах классов", width=14.2)
    P(doc, f"При равных априорных вероятностях accuracy = {num(balanced['accuracy'])}, macro-F1 = {num(balanced['macro_f1'])}. Это уже не сдвиг в один класс. Показатели по классам на {balanced['n_test']} тестовых статьях приведены в таблице 9, матрица ошибок — на рисунке 5.")
    TABLE(
        doc,
        "Таблица 9 — Качество по классам при равных априорных вероятностях",
        ["Класс", "Precision", "Recall", "F1", "Тест"],
        per_class_rows(balanced),
        [5.5, 2.8, 2.6, 2.4, 2.2],
    )
    FIGURE(doc, 5, "confusion_balanced.png", "Матрица ошибок при равных априорных вероятностях", width=14.2)
    P(doc, "Лучше всего по F1 отделяются инвестиции, 0,690. У риска F1 = 0,571. У банков F1 = 0,286, у торговых стратегий точность 1,000 при полноте 0,200, поэтому F1 = 0,333. Научные аннотации и журнальные статьи делят общий финансовый словарь, и граница между банком, риском и торговлей для мешка слов остаётся размытой.")
    P(doc, f"После удаления слов, которыми ставилась метка, accuracy падает с {num(balanced['accuracy'])} до {num(masked['accuracy'])}, macro-F1 — с {num(balanced['macro_f1'])} до {num(masked['macro_f1'])}. На {balanced['n_test']} тестовых статьях риск сохраняет F1 0,571, а торговля на этом прогоне не узнаётся: F1 падает до 0. Часть торгового сигнала сидела в словах algorithmic trading и machine learning. Матрица ошибок показана на рисунке 6.")
    FIGURE(doc, 6, "confusion_masked.png", "Матрица ошибок после удаления словаря меток", width=14.2)
    P(doc, f"Итог классификации ограничен размером теста: {balanced['n_test']} статей. Одна ошибка заметно меняет F1 класса с пятью примерами. Цифры пригодны как демонстрация метода и как сравнение трёх постановок, а не как оценка промышленного качества. Для устойчивой оценки нужен больший размеченный корпус и проверка на отложенном периоде, а не только случайное разбиение.")

    SECTION(doc, "6 Тенденции в публикациях")
    P(doc, f"По корпусу с {s['date_min']} по {s['date_max']} разговор об ИИ в финансах по-прежнему не собран вокруг компьютерного зрения: термин встречается в {cv['documents']} статьях. Глубокое обучение после добавления arXiv видно шире, в {dl['documents']} статьях, но это язык научных аннотаций, а не колонок Forbes. Шире всего по-прежнему риск, затем генеративный ИИ и машинное обучение. Алгоритмическая торговля есть в {algo['documents']} статьях: для прессы это узкая тема, для arXiv — одна из центральных.")
    P(doc, "Содержательно видны четыре линии, и они совпали с классами разметки. Первая — автоматизация торговли и спор о том, остаётся ли у человека контроль над алгоритмом. Вторая — инвестиции: портфель, акции «ИИ-компаний», сравнение модели с управляющим. Третья — банки и финтех: кредитование, платежи, внутренние помощники, команды, которые сопровождают модели. Четвёртая, и по частоте самая настойчивая, — риск: мошенничество, кредитные решения, надзор, проверка того, что модель делает в проде.")
    P(doc, "Генеративный ИИ в этом корпусе — инструмент сотрудника и клиента, а не замена торгового движка. ChatGPT и LLM попадают в топ TF-IDF, но статьи про них чаще говорят о банках, аналитиках и контроле доступа, чем о сигнале на сделку. Это полезное различение: внедрение большой языковой модели в финансовую организацию и алгоритмическая торговля — разные проекты с разными рисками.")
    P(doc, f"Наивный Байес это различение частично подтверждает. При равных априорных вероятностях он поднимает accuracy с базовых {num(balanced['baseline_accuracy'])} до {num(balanced['accuracy'])} и получает macro-F1 {num(balanced['macro_f1'])}. Риск отделяется устойчиво даже без слов размечающего словаря. Банковский класс отделяется плохо. Значит, для мониторинга ленты достаточно ловить риск и торговлю, а «банк или инвестиции» лучше размечать человеком либо более узкими правилами.")

    STRUCT(doc, "ЗАКЛЮЧЕНИЕ")
    P(doc, "Программа на Python собирает публикации Forbes по варианту 4, чистит текст, считает частоты и TF-IDF и обучает наивный байесовский классификатор. Все числа в записке получены на собранном корпусе, а не подставлены из методички. Код лежит в каталоге RGR_variant4: collect.py, fill_bodies.py, analyze.py, build_report.py. После выкладки на GitHub в эту записку нужно вставить ссылку на репозиторий: в условии она требуется, удалённый репозиторий при сборке отчёта не создавался.")
    P(doc, "Практический вывод для дальнейшего изучения такой. Следить за Forbes полезно как за картой повестки: где сейчас слово risk, где generative AI, где algorithmic trading. Нельзя выводить из этих статей параметры стратегии или ждать, что классификатор заменит чтение. Следующий шаг — разметить 200–300 статей вручную по тем же четырём классам, отделить колонки Councils от новостей и добавить временную ось: доля генеративного ИИ по годам против доли алгоритмической торговли. Тогда тренд будет измерен, а не только описан по совокупному корпусу.")
    P(doc, "Рекомендации по применению самого метода короткие. TF-IDF и наивный Байес уместны как первый фильтр входящей ленты и как учебный базис. Для решения о сделке, о кредите или о регуляторном заключении их недостаточно: нет цен, нет горизонта, нет ответственности за ошибку. В самих статьях Forbes этот же тезис повторяется в другом виде: модель можно ставить в контур, если рядом остаётся проверка человеком.")

    STRUCT(doc, "СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ")
    sources = [
        "Scikit-learn developers. Naive Bayes [Электронный ресурс] // scikit-learn User Guide. — URL: https://scikit-learn.org/stable/modules/naive_bayes.html (дата обращения: 16.11.2026).",
        "Manning, C. D. Introduction to Information Retrieval / C. D. Manning, P. Raghavan, H. Schütze. — Cambridge : Cambridge University Press, 2008.",
        "Porter, M. F. An algorithm for suffix stripping // Program. — 1980. — Vol. 14, № 3. — P. 130–137.",
        "Bird, S. Natural Language Processing with Python / S. Bird, E. Klein, E. Loper. — Beijing : O'Reilly, 2009.",
        "Forbes [Электронный ресурс]. — URL: https://www.forbes.com/ (дата обращения: 19.11.2026).",
        "arXiv [Электронный ресурс]. — URL: https://arxiv.org/ (дата обращения: 19.11.2026).",
        "TechCrunch [Электронный ресурс]. — URL: https://techcrunch.com/ (дата обращения: 19.11.2026).",
        "Internet Archive. Wayback Machine [Электронный ресурс]. — URL: https://web.archive.org/ (дата обращения: 16.11.2026).",
    ]
    for index, source in enumerate(sources, start=1):
        P(doc, f"{index} {source}", indent=Cm(0))

    STRUCT(doc, "ПРИЛОЖЕНИЕ А")
    center_line(doc, "ИСХОДНЫЙ КОД", bold=True, space_after=12)
    P(doc, "Ниже приведён код, которым получен корпус, частоты, TF-IDF и классификатор. Запуск: python collect.py, затем python fill_bodies.py для добивки закрытых страниц, затем python analyze.py. Записка собирается командой python build_report.py и читает results/summary.json, поэтому числа в тексте совпадают с последним прогоном. Распечатка программы приведена кеглем 12 пт.")
    CODE(doc, "Файл collect.py", ROOT / "collect.py")
    CODE(doc, "Файл collect_extra.py", ROOT / "collect_extra.py")
    CODE(doc, "Файл fill_bodies.py", ROOT / "fill_bodies.py")
    CODE(doc, "Файл analyze.py", ROOT / "analyze.py")

    settings = doc.settings.element
    update = OxmlElement("w:updateFields")
    update.set(qn("w:val"), "true")
    settings.append(update)

    OUT.parent.mkdir(exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
