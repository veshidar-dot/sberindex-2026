"""Заголовки федеральных лент из дневных архивов (только метаданные: дата, время, заголовок, рубрика, адрес).

    python -m sbi.news_crawl

Тексты статей не скачиваются. Страницы дневных архивов разрешены robots.txt обеих лент; между запросами пауза
(news_crawl.pause). Каждый день кэшируется в out/news/raw/<источник>_<дата>.json, повторный запуск докачивает
только недостающие дни. Итог - out/news/headlines.parquet. Сами заголовки в репозиторий не кладутся: в git
только агрегаты «МО x месяц x тип события» (sbi.news_geo).

Разбор страниц - по образцу src/sbi/sources/news.py репозитория a-amik/sberindex-2026 (лицензия MIT, атрибуция -
ref/news/SOURCES.md).
    Интерфакс  https://www.interfax.ru/news/ГГГГ/ММ/ДД[/page_N]
    Лента      https://lenta.ru/rubrics/{russia,economics}/ГГГГ/ММ/ДД/[page/N/]
"""
import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests

from .config import load_config, out_path

IFX_ITEM = re.compile(r'<div data-id="(\d+)">\s*<span>(\d{2}:\d{2})</span>\s*<a href="([^"]+)">\s*<h3>(.*?)</h3>', re.S)
LENTA_ITEM = re.compile(r'<a class="card-full-news[^"]*" href="(/news/[^"]+)"><h3 class="card-full-news__title">(.*?)</h3>'
                        r'.*?card-full-news__date">(\d{2}:\d{2})[^<]*</time>(?:<span[^>]*rubric">([^<]*)</span>)?', re.S)


def _session(ua):
    s = requests.Session()
    s.headers["User-Agent"] = ua
    return s


def _pages(source, d, s, pause, base):
    out, seen = [], set()
    for page in range(1, 40):
        url = (base + (f"/page_{page}" if page > 1 else "")) if source == "interfax" else (base + (f"page/{page}/" if page > 1 else ""))
        r = None
        for _ in range(3):                                   # два повтора на сетевой сбой
            try:
                r = s.get(url, timeout=60)
                break
            except requests.RequestException:
                time.sleep(5)
        if r is None or r.status_code != 200:                # страница за последней - конец дня
            break
        time.sleep(pause)
        if source == "interfax":
            items = [{"id": i, "time": t, "url": "https://www.interfax.ru" + u, "title": h, "rubric": u.split("/")[1]}
                     for i, t, u, h in IFX_ITEM.findall(r.text)]
        else:                                                # страницы дня у Ленты захватывают соседние дни
            items = [{"id": u, "time": t, "url": "https://lenta.ru" + u, "title": h, "rubric": rb}
                     for u, h, t, rb in LENTA_ITEM.findall(r.text) if u.startswith(f"/news/{d:%Y/%m/%d}/")]
        new = [x for x in items if x["id"] not in seen]
        if not new:
            break
        seen.update(x["id"] for x in new)
        out += new
    return out


def _day(source, d, s, pause):
    if source == "lenta":                                    # местные события у Ленты - в «Россия» и «Экономика»
        out = []
        for rub in ("russia", "economics"):
            out += _pages(source, d, s, pause, f"https://lenta.ru/rubrics/{rub}/{d:%Y/%m/%d}/")
    else:
        out = _pages(source, d, s, pause, f"https://www.interfax.ru/news/{d:%Y/%m/%d}")
    for x in out:
        x["title"] = html.unescape(re.sub(r"<[^>]+>", "", x["title"])).strip()
        x["date"] = d.isoformat()
        x["source"] = source
    return out


def main():
    cfg = load_config()
    p = cfg["news_crawl"]
    raw = out_path(cfg, "news", "raw", "x").parent
    d0, d1 = date.fromisoformat(p["start"]), date.fromisoformat(p["end"])
    days = [d0 + timedelta(i) for i in range((d1 - d0).days + 1)]

    def run(job):
        source, d = job
        f = raw / f"{source}_{d.isoformat()}.json"
        if f.exists():
            return 0
        items = _day(source, d, _session(p["user_agent"]), p["pause"])
        if items:                                            # пустой день не кэшируем: возможно, сбой
            f.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        return len(items)

    jobs = [(src, d) for d in days for src in p["sources"]]
    done = 0
    with ThreadPoolExecutor(p["workers_per_source"] * len(p["sources"])) as ex:
        for k, n in enumerate(ex.map(run, jobs), 1):
            done += n
            if k % 100 == 0:
                print(f"  {k}/{len(jobs)} дней-источников, новых заголовков {done:,}", flush=True)
    rows = [r for f in sorted(raw.glob("*.json")) for r in json.loads(f.read_text(encoding="utf-8"))]
    df = pd.DataFrame(rows).drop_duplicates(["source", "id"])
    df.to_parquet(out_path(cfg, "news", "headlines.parquet"), index=False)
    print(f"заголовков: {len(df):,} ({df.groupby('source').size().to_dict()}), дней: {df.date.nunique()}")


if __name__ == "__main__":
    main()
