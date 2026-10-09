"""Презентация в PDF: те же результаты, что в отчёте и на лендинге, на 16 слайдах 16:9.

    python -m sbi.deck

Шаблон - docs/deck/template.html (вёрстка и подписи); данные подставляются из out/ в одну JSON-строку,
графики рисует сам шаблон. Итог - docs/deck/presentation.html и docs/presentation.pdf (печать headless Chrome,
путь к браузеру - deck.browser в config.yaml).
"""
import json
import subprocess

import numpy as np
import pandas as pd

from .config import load_config, out_path, resolve


def _records(df):
    return json.loads(df.to_json(orient="records", force_ascii=False))


def collect(cfg):
    o = lambda f: out_path(cfg, f)
    d = {}
    m = pd.read_csv(o("metrics.csv"))
    d["metrics"] = _records(m[m.category == "Итого"][["model", "h", "MAE", "R2"]])
    d["metrics_cat"] = _records(m[m.model.isin(["ensemble", "prophet"])][["model", "h", "category", "MAE"]])
    for name, f in (("dm", "dm_vs_baseline.csv"), ("dm_seasonal", "dm_vs_prophet_seasonal.csv")):
        t = pd.read_csv(o(f))
        d[name] = _records(t[t.model == "ensemble"][["h", "mean_diff", "p"]])
    a = pd.read_csv(o("ablation_national_fixed.csv"))
    d["ablation"] = _records(a[a.category == "Итого"][["model", "h", "MAE"]])
    s = pd.read_csv(o("selection_mcs.csv"))
    d["mcs"] = _records(s[(s.category == "Итого") & s.in_mcs][["h", "model", "MAE"]])
    d["rank"] = _records(pd.read_csv(o("selection_rank_forecast.csv")))
    d["rank_detect"] = _records(pd.read_csv(o("selection_rank_detect.csv")))
    d["cumulative"] = _records(pd.read_csv(o("metrics_cumulative.csv")))
    ds = pd.read_csv(o("detection_summary.csv"))
    d["detect"] = _records(ds[ds.group == "все"][["detector", "recall", "recall_ci", "precision", "delay", "pauc"]])
    d["labeled"] = _records(pd.read_csv(o("detection_real_labeled.csv")))
    ev = pd.read_csv(o("detection_real_events.csv"))
    big = ev[ev.shift_log.abs() > 0.1]
    d["events"] = {"rows": len(ev), "big": len(big),
                   **{k: {"all": float((ev[k] >= 0).mean()), "big": float((big[k] >= 0).mean())}
                      for k in ("cusum", "pelt", "bocpd", "ssm_lr", "ssm_lr_news")}}
    g = pd.read_csv(o("gdelt_event_study.csv"))
    d["gdelt"] = _records(g[(g.category == "все") & (g.news == "hazard")][["scope", "lag_news_vs_shift", "lift", "ci_low", "ci_high"]])
    fmc = pd.read_csv(o("fm_compare.csv"))
    d["fm_sub"] = _records(fmc[fmc.category == "Итого"][["model", "h", "MAE"]])
    iv = o("intervals.csv")
    if iv.exists():
        t = pd.read_csv(iv)
        d["intervals"] = _records(t[t.category == "Итого"])
        d["intervals_cat"] = _records(t[(t.rows == "все") & (t.level == 0.8)][["h", "category", "coverage"]])
    n = pd.read_csv(o("national_oos_summary.csv"))
    d["national_oos"] = _records(n)
    d["forward"] = json.loads(o("forward/manifest.json").read_text(encoding="utf-8"))
    d["cases"] = json.loads(o("cases.json").read_text(encoding="utf-8"))
    return d


def main():
    cfg = load_config()
    data = json.dumps(collect(cfg), ensure_ascii=False, default=float).replace("</", "<\\/")
    tpl = resolve("docs/deck/template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*DATA*/null", data)
    page = resolve("docs/deck/presentation.html")
    page.write_text(html, encoding="utf-8")
    pdf = resolve("docs/presentation.pdf")
    browser = cfg["deck"]["browser"]
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                    f"--print-to-pdf={pdf}", page.as_uri()], check=True, capture_output=True, timeout=180)
    print(pdf, pdf.stat().st_size // 1024, "КБ")


if __name__ == "__main__":
    main()
