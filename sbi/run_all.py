"""Полный прогон проекта одной командой.

    python -m sbi.run_all                  # всё с нуля (CPU 16 ядер: около 12 ч, из них два Prophet ~7 ч)
    python -m sbi.run_all --skip-existing  # не пересчитывать прогнозы, которые уже лежат в out/forecasts

Порядок: каталог событий -> прогнозы всех моделей (ансамбль после своих членов) -> метрики и тесты DM ->
абляции -> детекция (синтетика, реальные события) -> новости (GDELT, эффект на прогноз) -> выбор лучшей модели
и детектора -> проверка общей траектории вне выборки -> прогноз на 2025 г. -> данные лендинга.
"""
import argparse
import subprocess
import sys
import time

from .config import load_config, out_path

MODELS = ["naive", "seasonal_naive", "panel_ssm", "panel_ssm_news", "panel_ssm_automap", "panel_seasonal",
          "lgbm", "lgbm_news", "timesfm_local", "chronos2_local", "ensemble",
          "chronos2", "chronos2_cross", "timesfm", "panel_ssm_chronos", "chronos2_ft_local", "tirex_local",
          "prophet", "prophet_seasonal"]

STEPS = [
    ["sbi.report"],
    ["sbi.ablation", "--ext-map", "fixed"],
    ["sbi.ablation", "--ext-map", "auto"],
    ["sbi.detect_eval"],
    ["sbi.detect_real"],
    ["sbi.gdelt"],
    ["sbi.news_eval"],
    ["sbi.studies"],
    ["sbi.fm_compare"],
    ["sbi.selection"],
    ["sbi.national_oos"],
    ["sbi.forward"],
    ["sbi.export_landing"],
]


def run(args):
    t = time.time()
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-W", "ignore", "-m", *args], check=True)
    print(f"   {time.time() - t:.0f} с", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    run(["sbi.news"])
    for m in MODELS:
        if a.skip_existing and out_path(cfg, "forecasts", f"{m}.npz").exists() and m != "ensemble":
            print(f"-- {m}: прогноз уже есть, пропускаю", flush=True)
            continue
        run(["sbi.forecast", "--models", m])
    for s in STEPS:
        run(s)


if __name__ == "__main__":
    main()
