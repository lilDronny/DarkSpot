"""
darkspot_integration — связующий слой проекта DarkSpot.

Соединяет три независимые ветки проекта, не переписывая их логику:
  1. Ветка ЦА (анализ целевой аудитории)      → спрос по районам
  2. Ветка конкурентов (2GIS)                  → плотность конкуренции по районам
  3. Ветка недвижимости (ЦИАН)                 → стоимость аренды по районам

Каждая ветка в конце своего ноутбука вызывает соответствующий export_* и
сбрасывает свой финальный артефакт в общую папку DATA_DIR. Оркестрирующий
ноутбук читает эти артефакты через load_* и собирает их в единую модель.

Модуль — единственный источник истины для:
  - схемы обмена между ветками
  - канонического списка районов с центроидами
  - модели аренды и итогового скоринга локаций
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error

DATA_DIR = "darkspot_data"

DEMAND_CSV = "demand.csv"
COMPETITORS_CSV = "competitors.csv"
REALESTATE_DISTRICT_CSV = "realestate_district.csv"
REALESTATE_LISTINGS_CSV = "realestate_listings.csv"

# Соответствие формата бизнеса колонке спроса из ветки ЦА и параметрам экономики
FORMATS = {
    "coffeeshop":   {"ru": "Кофейня",       "ca_col": "Кофейня",       "queries": ["кофейня", "кофе"]},
    "nail_salon":   {"ru": "Салон красоты", "ca_col": "Салон красоты", "queries": ["маникюр", "ногтевая студия"]},
    "pickup_point": {"ru": "ПВЗ",           "ca_col": "ПВЗ",           "queries": ["пункт выдачи", "пвз"]},
}

OKRUG_INCOME_INDEX = {
    "ЦАО": 1.35, "ЗАО": 1.20, "СЗАО": 1.10, "ЮЗАО": 1.08, "САО": 1.05,
    "СВАО": 0.95, "ВАО": 0.90, "ЮАО": 0.88, "ЮВАО": 0.85, "ЗелАО": 0.82,
}

# Канонический список районов Москвы с центроидами (name, okrug, lat, lon).
# Нужен для карты и для запросов конкуренции по районам в ветке 2GIS.
MOSCOW_DISTRICTS = [
    ("Хамовники","ЦАО",55.728,37.58),("Арбат","ЦАО",55.749,37.592),("Тверской","ЦАО",55.767,37.606),
    ("Пресненский","ЦАО",55.76,37.57),("Замоскворечье","ЦАО",55.735,37.63),("Якиманка","ЦАО",55.73,37.61),
    ("Басманный","ЦАО",55.765,37.67),("Таганский","ЦАО",55.74,37.66),("Мещанский","ЦАО",55.778,37.633),
    ("Красносельский","ЦАО",55.78,37.66),("Аэропорт","САО",55.8,37.53),("Сокол","САО",55.805,37.515),
    ("Беговой","САО",55.785,37.56),("Тимирязевский","САО",55.82,37.57),("Савёловский","САО",55.795,37.585),
    ("Войковский","САО",55.82,37.5),("Коптево","САО",55.84,37.52),("Головинский","САО",55.85,37.49),
    ("Левобережный","САО",55.86,37.47),("Дмитровский","САО",55.87,37.54),("Бескудниковский","САО",55.87,37.56),
    ("Западное Дегунино","САО",55.88,37.52),("Восточное Дегунино","САО",55.89,37.56),("Ховрино","САО",55.87,37.49),
    ("Хорошёвский","САО",55.78,37.53),("Молжаниновский","САО",55.95,37.43),("Останкинский","СВАО",55.82,37.61),
    ("Бабушкинский","СВАО",55.87,37.66),("Свиблово","СВАО",55.86,37.63),("Алексеевский","СВАО",55.81,37.64),
    ("Марьина роща","СВАО",55.8,37.615),("Бутырский","СВАО",55.815,37.585),("Ростокино","СВАО",55.835,37.66),
    ("Ярославский","СВАО",55.87,37.7),("Лосиноостровский","СВАО",55.875,37.7),("Северное Медведково","СВАО",55.895,37.65),
    ("Южное Медведково","СВАО",55.875,37.64),("Бибирево","СВАО",55.89,37.605),("Отрадное","СВАО",55.865,37.605),
    ("Алтуфьевский","СВАО",55.89,37.585),("Лианозово","СВАО",55.9,37.58),("Марфино","СВАО",55.83,37.595),
    ("Северный","СВАО",55.92,37.56),("Сокольники","ВАО",55.79,37.68),("Измайлово","ВАО",55.79,37.78),
    ("Преображенское","ВАО",55.795,37.715),("Богородское","ВАО",55.815,37.71),("Гольяново","ВАО",55.815,37.77),
    ("Метрогородок","ВАО",55.82,37.73),("Восточное Измайлово","ВАО",55.79,37.81),("Северное Измайлово","ВАО",55.81,37.8),
    ("Соколиная Гора","ВАО",55.775,37.72),("Перово","ВАО",55.75,37.78),("Новогиреево","ВАО",55.75,37.81),
    ("Ивановское","ВАО",55.77,37.86),("Вешняки","ВАО",55.72,37.82),("Косино-Ухтомский","ВАО",55.7,37.86),
    ("Новокосино","ВАО",55.745,37.86),("Восточный","ВАО",55.815,37.87),("Люблино","ЮВАО",55.68,37.76),
    ("Марьино","ЮВАО",55.65,37.74),("Текстильщики","ЮВАО",55.705,37.73),("Кузьминки","ЮВАО",55.7,37.77),
    ("Печатники","ЮВАО",55.69,37.715),("Рязанский","ЮВАО",55.72,37.785),("Нижегородский","ЮВАО",55.73,37.715),
    ("Лефортово","ЮВАО",55.76,37.7),("Южнопортовый","ЮВАО",55.715,37.68),("Выхино-Жулебино","ЮВАО",55.7,37.83),
    ("Капотня","ЮВАО",55.64,37.8),("Некрасовка","ЮВАО",55.7,37.91),("Даниловский","ЮАО",55.71,37.63),
    ("Донской","ЮАО",55.705,37.61),("Нагатинский Затон","ЮАО",55.68,37.67),("Чертаново Центральное","ЮАО",55.63,37.61),
    ("Чертаново Северное","ЮАО",55.65,37.605),("Чертаново Южное","ЮАО",55.6,37.605),("Нагатино-Садовники","ЮАО",55.675,37.62),
    ("Нагорный","ЮАО",55.67,37.62),("Москворечье-Сабурово","ЮАО",55.64,37.66),("Царицыно","ЮАО",55.62,37.68),
    ("Бирюлёво Восточное","ЮАО",55.59,37.66),("Бирюлёво Западное","ЮАО",55.58,37.62),("Орехово-Борисово Северное","ЮАО",55.62,37.73),
    ("Орехово-Борисово Южное","ЮАО",55.6,37.73),("Зябликово","ЮАО",55.61,37.745),("Братеево","ЮАО",55.635,37.755),
    ("Гагаринский","ЮЗАО",55.7,37.57),("Академический","ЮЗАО",55.69,37.57),("Обручевский","ЮЗАО",55.66,37.54),
    ("Коньково","ЮЗАО",55.63,37.52),("Ломоносовский","ЮЗАО",55.69,37.54),("Котловка","ЮЗАО",55.665,37.595),
    ("Зюзино","ЮЗАО",55.655,37.58),("Черёмушки","ЮЗАО",55.67,37.56),("Тёплый Стан","ЮЗАО",55.62,37.49),
    ("Ясенево","ЮЗАО",55.6,37.53),("Северное Бутово","ЮЗАО",55.57,37.56),("Южное Бутово","ЮЗАО",55.54,37.54),
    ("Раменки","ЗАО",55.7,37.5),("Дорогомилово","ЗАО",55.74,37.55),("Кунцево","ЗАО",55.73,37.43),
    ("Крылатское","ЗАО",55.76,37.41),("Фили-Давыдково","ЗАО",55.73,37.47),("Филёвский Парк","ЗАО",55.745,37.5),
    ("Можайский","ЗАО",55.725,37.41),("Очаково-Матвеевское","ЗАО",55.69,37.46),("Тропарёво-Никулино","ЗАО",55.66,37.48),
    ("Проспект Вернадского","ЗАО",55.675,37.5),("Внуково","ЗАО",55.64,37.29),("Солнцево","ЗАО",55.645,37.39),
    ("Ново-Переделкино","ЗАО",55.64,37.35),("Строгино","СЗАО",55.8,37.4),("Хорошёво-Мнёвники","СЗАО",55.78,37.47),
    ("Щукино","СЗАО",55.81,37.46),("Покровское-Стрешнево","СЗАО",55.82,37.45),("Северное Тушино","СЗАО",55.855,37.44),
    ("Южное Тушино","СЗАО",55.83,37.43),("Митино","СЗАО",55.84,37.36),("Куркино","СЗАО",55.89,37.39),
]


def districts_table():
    df = pd.DataFrame(MOSCOW_DISTRICTS, columns=["district", "okrug", "lat", "lon"])
    df["income_index"] = df["okrug"].map(OKRUG_INCOME_INDEX)
    df["_key"] = df["district"].map(_norm_name)
    return df


def _norm_name(s):
    s = str(s).lower().replace("ё", "е")
    s = re.sub(r"\bрайон\b", "", s)
    s = re.sub(r"[^а-яa-z0-9 -]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================ EXPORT (ветки → диск)

def export_demand(df_ca, path=None):
    """Ветка ЦА: нормализует свой df и сохраняет спрос по районам.
    Ожидает колонки: district, okrug, population, avg_salary, income_index
    и колонки спроса по форматам (Кофейня / Салон красоты / ПВЗ)."""
    _ensure_dir()
    keep = ["district", "okrug", "population", "avg_salary", "income_index"]
    fmt_cols = [c for c in ["Кофейня", "Салон красоты", "ПВЗ"] if c in df_ca.columns]
    out = df_ca[keep + fmt_cols].copy()
    rename = {"Кофейня": "demand_coffee", "Салон красоты": "demand_beauty", "ПВЗ": "demand_pickup"}
    out = out.rename(columns=rename)
    path = path or os.path.join(DATA_DIR, DEMAND_CSV)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[export_demand] {len(out)} районов → {path}")
    return path


def export_competitors(df_comp, path=None):
    """Ветка конкурентов: сохраняет агрегат по районам.
    Ожидает колонки: district, n_competitors, avg_comp_rating."""
    _ensure_dir()
    cols = ["district", "n_competitors", "avg_comp_rating"]
    out = df_comp[cols].copy()
    path = path or os.path.join(DATA_DIR, COMPETITORS_CSV)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[export_competitors] {len(out)} районов → {path}")
    return path


def export_realestate(df_cian, district_path=None, listings_path=None):
    """Ветка ЦИАН: сохраняет и агрегат по районам, и сырые объявления.
    Ожидает колонки: district, okrug, price_sqm_month, area_sqm, metro_walk_min,
    price_month, address, object_type, url (часть может отсутствовать)."""
    _ensure_dir()
    df = df_cian.copy()
    df = df[df["district"].notna()]

    agg = (df.groupby("district")
             .agg(median_rent_sqm=("price_sqm_month", "median"),
                  n_listings=("price_sqm_month", "count"),
                  avg_metro_min=("metro_walk_min", "mean"))
             .round(1).reset_index())

    district_path = district_path or os.path.join(DATA_DIR, REALESTATE_DISTRICT_CSV)
    listings_path = listings_path or os.path.join(DATA_DIR, REALESTATE_LISTINGS_CSV)
    agg.to_csv(district_path, index=False, encoding="utf-8-sig")

    listing_cols = [c for c in ["district", "okrug", "address", "object_type", "area_sqm",
                                "price_sqm_month", "price_month", "metro_walk_min", "floor", "url"]
                    if c in df.columns]
    df[listing_cols].to_csv(listings_path, index=False, encoding="utf-8-sig")
    print(f"[export_realestate] {len(agg)} районов → {district_path}; {len(df)} объявлений → {listings_path}")
    return district_path, listings_path


# ============================================================ LOAD (диск → оркестратор)

def _require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Не найден артефакт: {path}. Запусти соответствующую ветку с её export_* ячейкой."
        )
    return path


def load_demand(business_type):
    path = _require(os.path.join(DATA_DIR, DEMAND_CSV))
    df = pd.read_csv(path)
    col = {"coffeeshop": "demand_coffee", "nail_salon": "demand_beauty",
           "pickup_point": "demand_pickup"}[business_type]
    df["demand"] = df[col]
    df["_key"] = df["district"].map(_norm_name)
    return df


def load_competitors():
    path = _require(os.path.join(DATA_DIR, COMPETITORS_CSV))
    df = pd.read_csv(path)
    df["_key"] = df["district"].map(_norm_name)
    return df[["_key", "n_competitors", "avg_comp_rating"]]


def load_realestate():
    dpath = _require(os.path.join(DATA_DIR, REALESTATE_DISTRICT_CSV))
    lpath = _require(os.path.join(DATA_DIR, REALESTATE_LISTINGS_CSV))
    dd = pd.read_csv(dpath); dd["_key"] = dd["district"].map(_norm_name)
    ll = pd.read_csv(lpath); ll["_key"] = ll["district"].map(_norm_name)
    return dd[["_key", "median_rent_sqm", "n_listings", "avg_metro_min"]], ll


# ============================================================ MASTER + МОДЕЛЬ

FEATURES = ["income_index", "demand", "n_competitors", "avg_comp_rating", "avg_metro_min"]
TARGET = "median_rent_sqm"


def build_master(business_type):
    """Сводит три ветки на уровне района через нормализованный ключ названия."""
    demand = load_demand(business_type)
    comp = load_competitors()
    rent_d, _ = load_realestate()
    geo = districts_table()[["_key", "lat", "lon"]]

    master = (demand
              .merge(comp, on="_key", how="left")
              .merge(rent_d, on="_key", how="left")
              .merge(geo, on="_key", how="left"))

    before = len(master)
    master = master.dropna(subset=[TARGET]).reset_index(drop=True)
    master["avg_comp_rating"] = master["avg_comp_rating"].fillna(master["avg_comp_rating"].median())
    master["n_competitors"] = master["n_competitors"].fillna(0)
    print(f"[build_master] районов с полными данными: {len(master)} / {before}")
    return master


class RentModel:
    """Линейная регрессия справедливой арендной ставки района по фундаментальным признакам.
    Остаток (реальная - предсказанная ставка) трактуется как недооценённость локации."""

    def __init__(self):
        self.pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
        self.metrics_ = {}

    def fit(self, master):
        X, y = master[FEATURES], master[TARGET]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
        self.pipe.fit(Xtr, ytr)
        pred = self.pipe.predict(Xte)
        cv = cross_val_score(self.pipe, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
        self.metrics_ = {"r2": r2_score(yte, pred),
                         "mae": mean_absolute_error(yte, pred),
                         "cv_mean": cv.mean(), "cv": cv}
        # дообучаем на всех данных для предсказаний по всем районам
        self.pipe.fit(X, y)
        return self

    def coefficients(self):
        return pd.Series(self.pipe.named_steps["lr"].coef_, index=FEATURES).sort_values(key=abs, ascending=False)

    def annotate(self, master):
        df = master.copy()
        df["pred_rent"] = self.pipe.predict(df[FEATURES]).round(1)
        df["rent_residual"] = (df[TARGET] - df["pred_rent"]).round(1)
        return df


# ============================================================ СКОРИНГ

def _minmax(s):
    s = s.astype(float)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def score_opportunities(master_annotated, weights=None, budget_rent_month=None, area_sqm=60):
    w = weights or {"demand": 0.40, "low_comp": 0.30, "underpriced": 0.30}
    df = master_annotated.copy()
    df["s_demand"] = _minmax(df["demand"])
    df["s_low_comp"] = 1 - _minmax(df["n_competitors"])
    df["s_underpriced"] = 1 - _minmax(df["rent_residual"])
    raw = (w["demand"] * df["s_demand"] +
           w["low_comp"] * df["s_low_comp"] +
           w["underpriced"] * df["s_underpriced"])
    df["opportunity_score"] = (_minmax(raw) * 100).round(1)
    if budget_rent_month is not None:
        df["fits_budget"] = df["median_rent_sqm"] * area_sqm <= budget_rent_month
    else:
        df["fits_budget"] = True
    return df.sort_values("opportunity_score", ascending=False).reset_index(drop=True)


def find_spots(ranking, listings, area_sqm=60, budget_rent_month=None, top_districts=6, top_n=10):
    top = ranking[ranking["fits_budget"]].head(top_districts)
    keys = set(top["_key"])
    cand = listings[listings["_key"].isin(keys)].copy()
    cand = cand.merge(ranking[["_key", "opportunity_score"]], on="_key", how="left")

    cand = cand[cand["area_sqm"].between(area_sqm * 0.5, area_sqm * 2.0)]
    if budget_rent_month is not None and "price_month" in cand:
        cand = cand[cand["price_month"] <= budget_rent_month]
    if cand.empty:
        return cand

    cand["s_district"] = _minmax(cand["opportunity_score"])
    cand["s_cheap"] = 1 - _minmax(cand["price_sqm_month"])
    cand["s_metro"] = 1 - _minmax(cand["metro_walk_min"].fillna(cand["metro_walk_min"].median()))
    raw = 0.5 * cand["s_district"] + 0.3 * cand["s_cheap"] + 0.2 * cand["s_metro"]
    cand["spot_score"] = (_minmax(raw) * 100).round(1)
    return cand.sort_values("spot_score", ascending=False).head(top_n).reset_index(drop=True)
