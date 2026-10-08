# Источники данных в ref/news

Файлы взяты из открытых репозиториев других участников конкурса под лицензией MIT. Мы используем
их как данные (агрегаты и реестры), а не как код решения. Привязку мест к МО проверили на случайной
выборке из 40 мест: все 29 мест, чьи МО есть в панели, привязаны к правильному МО (остальные 11 -
МО вне панели, например Белгородская и Курская области).

| Файл | Исходный путь | Репозиторий, коммит |
|---|---|---|
| `gdelt_monthly_mo.parquet` | `data/external/news_monthly_mo.parquet` | maloyan/sberindex-2026-forecasting, `beebf5b365` |
| `gdelt_monthly_region.parquet` | `data/external/news_monthly_region.parquet` | maloyan/sberindex-2026-forecasting, `beebf5b365` |
| `gdelt_location_match.csv` | `data/external/news_location_match.csv` | maloyan/sberindex-2026-forecasting, `beebf5b365` |
| `event_registry_maloyan.csv` | `data/external/event_registry.csv` | maloyan/sberindex-2026-forecasting, `beebf5b365` |
| `rosstat_retail_monthly.csv` | `data/external/rosstat_monthly.csv` | maloyan/sberindex-2026-forecasting, `beebf5b365` |
| `event_registry_rav11l.csv` | `data/events/registry.csv` | rav11l/sberindex-shocks, `956bbf10a6` |

Что внутри:
- `gdelt_monthly_*` - помесячные счётчики сообщений GDELT 2.0 (Global Database of Events, Language and Tone,
  открытые данные) по МО и по регионам: число документов и статей, суммарная тональность, число негативных
  документов, тематические счётчики (катастрофы, паводки, пожары, экономика, цены, инфраструктура, безопасность).
  Покрытие - 811 МО, 2022-12 ... 2024-12.
- `gdelt_location_match.csv` - сопоставление мест из GDELT (название, координаты) с `territory_id` и кодом региона.
- `rosstat_retail_monthly.csv` - месячный оборот розничной торговли (всего, продовольственные, непродовольственные)
  и общественного питания, 2000-2025. По описанию в исходном репозитории это данные Росстата; мы проверили шов: с 2018-12 значения совпадают
  с национальным рядом СберИндекса «Потребительские расходы» (`ref/consumer_spending_national.csv`) до тысячных.
- `event_registry_*` - реестры событий со ссылками на первоисточники (решения Банка России по ключевой ставке,
  региональные события), датой объявления и ожидаемым знаком эффекта.

## Лицензии

maloyan/sberindex-2026-forecasting:

```
MIT License

Copyright (c) 2026 Narek Maloyan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

rav11l/sberindex-shocks: та же лицензия MIT, `Copyright (c) 2026 Ravil Akhtyamov`.
