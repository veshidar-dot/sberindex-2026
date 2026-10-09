"""Привязка заголовков к МО и субъектам и разметка типа события; помесячные агрегаты.

    python -m sbi.news_geo

Вход - out/news/headlines.parquet (sbi.news_crawl). Выход:
  out/news/headlines_annotated.parquet - заголовки с типами, субъектами и МО (в git не кладутся);
  ref/news/headlines_monthly_mo.csv     - число заголовков «МО x месяц x тип» (только счётчики, в git);
  ref/news/headlines_monthly_region.csv - то же по субъектам;
  out/news/geo_check_sample.csv         - случайная выборка привязок для ручной проверки.

Правила - по образцу src/sbi/newsgeo.py репозитория a-amik/sberindex-2026 (лицензия MIT, атрибуция - ref/news/SOURCES.md):
МО находится по названию центра (если оно одно на страну или его субъект назван в том же заголовке) либо по шаблону
«<имя> район / округ»; субъект - по прилагательному перед «область / край / республика» или по прямому имени.
Слова приводятся к начальной форме (pymorphy3). Тип события - по словам-маркерам; у заголовка может быть несколько типов.
Справочник центров и субъектов МО - ref/mo_dict.csv (справочник МО СберИндекса).
"""
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import pymorphy3

from .config import load_config, out_path, resolve

MORPH = pymorphy3.MorphAnalyzer()

EVENTS = {
    "flood": {"паводок", "наводнение", "подтопление", "подтопить", "затопить", "затопление", "половодье",
              "дамба", "разлив", "паводковый"},
    "fire": {"пожар", "возгорание", "загореться", "сгореть", "пожарище"},
    "emergency": {"чс", "чрезвычайный", "эвакуация", "эвакуировать", "взрыв", "обрушение", "обрушиться",
                  "авария", "ураган", "землетрясение", "смерч", "буря", "непогода", "обесточить", "блэкаут"},
    "attack": {"беспилотник", "бпла", "дрон", "атака", "обстрел", "пво", "ракета"},
    "plant": {"завод", "предприятие", "комбинат", "фабрика", "нпз", "шахта", "рудник"},
    "layoff": {"сокращение", "увольнение", "банкротство", "закрытие", "простой", "задолженность"},
    "payments": {"выплата", "компенсация", "пособие", "материальный"},
    "transport": {"мост", "трасса", "перекрыть", "паром", "переправа", "аэропорт", "рейс"},
}
REGION_WORDS = {"область", "край", "республика", "округ", "ао"}
# центры МО, у которых есть знаменитый тёзка вне набора: заголовок про тёзку не должен уходить в маленький МО
STOP = {"ростов", "мирный", "советский", "заречный", "октябрьский", "первомайский", "ленинский", "курск", "курская",
        "белгород", "брянск", "воронеж", "краснодар", "симферополь", "севастополь", "волчанск", "харьков",
        "донецк", "луганск", "херсон", "мариуполь", "москва", "петербург",
        # проверка выборки: «Патриарх Кирилл» - не город Кириллов, Первомайск - тёзка в зоне боёв
        "кирилл", "первомайск", "суровикин", "мантуров", "неман", "игра"}
WATER = {"море", "пролив", "залив", "озеро", "река", "водохранилище", "бухта"}   # «в Охотском море» - не Охотский район
# центры-прилагательные (с. Казанское, пгт Некрасовское, г. Белый) путаются с «Казанским вокзалом», «Белым домом»:
# по названию центра такие МО не ищем, только по шаблону «<имя> район / округ»
ADJ_END = ("ский", "цкий", "ый", "ий", "ой", "ая", "ое")
ENDINGS = ("ами", "ями", "ах", "ях", "ов", "ев", "ом", "ем", "ой", "ей", "ам", "ям", "а", "я", "у", "ю", "е", "ы", "и")
ALIASES = {"татарстан": "татарстан", "башкирия": "башкортостан", "башкортостан": "башкортостан",
           "якутия": "саха", "кузбасс": "кемеровская", "чувашия": "чувашия", "удмуртия": "удмуртская",
           "мордовия": "мордовия", "дагестан": "дагестан", "чечня": "чеченская", "ингушетия": "ингушетия",
           "бурятия": "бурятия", "тыва": "тыва", "хакасия": "хакасия", "карелия": "карелия", "коми": "коми",
           "калмыкия": "калмыкия", "адыгея": "адыгея", "югра": "ханты", "ямал": "ямало", "москва": "москва",
           "петербург": "санкт", "подмосковье": "московская", "приморье": "приморский", "кубань": "краснодарский",
           "забайкалье": "забайкальский", "камчатка": "камчатский", "чукотка": "чукотский", "сахалин": "сахалинская"}


@lru_cache(maxsize=300_000)
def lemma(word):
    return MORPH.parse(word)[0].normal_form


@lru_cache(maxsize=300_000)
def is_geo(word):
    return any("Geox" in p.tag for p in MORPH.parse(word)[:3])


@lru_cache(maxsize=300_000)
def is_person(word):
    """Первый разбор - имя или фамилия: «Суровикина», «Гагарина», «Владимира» в заголовке чаще люди, чем города."""
    tag = MORPH.parse(word)[0].tag
    return any(g in tag for g in ("Name", "Surn", "Patr"))


def stems(word):
    """Основы без падежного окончания: словарь знает не все топонимы («Ишима» -> «ишим»)."""
    return [word[: -len(e)] for e in ENDINGS if word.endswith(e) and len(word) - len(e) >= 3]


def tokens(text):
    return [(w, lemma(w.lower().replace("ё", "е")), w[0].isupper()) for w in re.findall(r"[А-Яа-яЁё][А-Яа-яЁё-]*", text)]


def region_key(name):
    """Ключ субъекта, совместимый с ALIASES: «Оренбургская область» -> «оренбургская», «Республика Саха (Якутия)» -> «саха»."""
    n = name.lower().replace("ё", "е")
    special = {"чувашская": "чувашия", "москва": "москва", "санкт-петербург": "санкт", "ханты-мансийский": "ханты",
               "ямало-ненецкий": "ямало"}
    first = n.split()[0]
    if first in special:
        return special[first]
    if first == "республика":
        return n.split()[1]
    return first.split("-")[0] if first.startswith(("ханты", "ямало")) else first


def gazetteer(mo):
    city, district = {}, {}
    for r in mo.itertuples():
        rk = region_key(r.region_name)
        if isinstance(r.center, str):
            name = re.sub(r"^(г|пгт|рп|п|с|ст-ца|аул|д|х|сл|кп|дп|снт|село|поселок)\.?\s+", "", r.center.strip())
            if len(name) >= 4 and " " not in name:
                city.setdefault(lemma(name.lower().replace("ё", "е")), []).append((r.territory_id, rk))
        if isinstance(r.name_short, str) and r.name_short.endswith(("ский", "цкий", "ной", "ный")):
            district.setdefault(lemma(r.name_short.lower().replace("ё", "е")), []).append((r.territory_id, rk))
    regions = {lemma(region_key(n)): region_key(n) for n in mo.region_name.unique()}
    is_city = dict(zip(mo.territory_id, mo.mo_type.eq("городской округ")))
    return {"city": city, "district": district, "regions": regions, "is_city": is_city}


def locate(text, gz):
    """Субъекты и МО, названные в заголовке."""
    tk = tokens(text)
    regs, mos = set(), set()
    for i, (w, lm, cap) in enumerate(tk):
        nxt = tk[i + 1][1] if i + 1 < len(tk) else ""
        if lm in ALIASES and cap:
            regs.add(ALIASES[lm])
        if lm in gz["regions"] and nxt in REGION_WORDS:
            regs.add(gz["regions"][lm])
    for i, (w, lm, cap) in enumerate(tk):
        nxt = tk[i + 1][1] if i + 1 < len(tk) else ""
        low = w.lower().replace("ё", "е")
        key = lm if lm in gz["city"] else next((x for x in stems(low) if x in gz["city"]), None)
        first_word_ok = i > 0 or is_geo(low)            # «Мирный протест…» в начале - не город Мирный
        cands = []
        if (cap and key and key not in STOP and not key.endswith(ADJ_END) and first_word_ok
                and nxt not in REGION_WORDS and nxt not in WATER and not is_person(low)):
            cands = gz["city"][key]
        elif lm in gz["district"] and nxt in {"район", "округ"}:
            cands = gz["district"][lm]
        if regs:                                        # субъект назван явно: МО другого субъекта не берём
            cands = [(t, r) for t, r in cands if r in regs]  # («Даниловский район Москвы» - не Ярославская обл.)
        if len(cands) == 1 and not (lm in gz["regions"] and nxt in REGION_WORDS):
            mos.add(cands[0][0])
            regs.add(cands[0][1])
        elif len(cands) > 1:
            pick = [t for t, r in cands if r in regs] or ([t for t, _ in cands] if len({r for _, r in cands}) == 1 else [])
            if len(pick) > 1:                          # город - центр и своего округа, и соседнего района
                pick = [t for t in pick if gz["is_city"].get(t)] or pick
            if len(pick) == 1:
                mos.add(pick[0])
                regs.update(r for t, r in cands if t == pick[0])
    return regs, mos


def classify(text):
    lms = {lm for _, lm, _ in tokens(text)}
    out = [k for k, words in EVENTS.items() if lms & words]
    if lms & EVENTS["flood"] or ("чс" in lms and "режим" in lms):   # крупное стихийное событие
        out.append("disaster")
    return out


def main():
    cfg = load_config()
    h = pd.read_parquet(out_path(cfg, "news", "headlines.parquet"))
    mo = pd.read_csv(resolve("ref/mo_dict.csv"))
    gz = gazetteer(mo)
    rows = []
    for k, title in enumerate(h.title):
        regs, mos = locate(title, gz)
        rows.append((classify(title), sorted(regs), sorted(mos)))
        if k % 50_000 == 0:
            print(f"  {k:,}/{len(h):,}", flush=True)
    h["types"], h["regions"], h["mos"] = zip(*rows)
    h["month"] = h.date.str[:7]
    h.to_parquet(out_path(cfg, "news", "headlines_annotated.parquet"), index=False)

    def agg(col, key):
        e = h[h[col].map(len) > 0].explode(col)
        e["any"] = 1
        typ = e.explode("types").dropna(subset=["types"])
        a = e.groupby([col, "month"]).size().rename("all")
        t = typ.groupby([col, "month", "types"]).size().unstack("types", fill_value=0)
        return pd.concat([a, t], axis=1).fillna(0).astype(int).reset_index().rename(columns={col: key})

    mo_m, rg_m = agg("mos", "territory_id"), agg("regions", "region_key")
    mo_m.to_csv(resolve("ref/news/headlines_monthly_mo.csv"), index=False, encoding="utf-8")
    rg_m.to_csv(resolve("ref/news/headlines_monthly_region.csv"), index=False, encoding="utf-8")
    loc = h[h.mos.map(len) > 0]
    names = mo.set_index("territory_id").name_short
    smp = loc.sample(min(60, len(loc)), random_state=cfg["seed"])[["date", "source", "title", "mos"]]
    smp["mo_names"] = smp.mos.map(lambda ids: "; ".join(names[i] for i in ids))
    smp.to_csv(out_path(cfg, "news", "geo_check_sample.csv"), index=False, encoding="utf-8")
    typed = h.types.map(len) > 0
    print(f"заголовков {len(h):,}; с типом события {typed.mean():.1%}; с субъектом {(h.regions.map(len) > 0).mean():.1%}; "
          f"с МО {(h.mos.map(len) > 0).mean():.1%}; МО с хотя бы одной новостью: {mo_m.territory_id.nunique()}")
    print(h.explode("types").types.value_counts().to_string())


if __name__ == "__main__":
    main()
