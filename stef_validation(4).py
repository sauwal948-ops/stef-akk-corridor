#!/usr/bin/env python3
"""
STEF validation pipeline: one script, two stages, fully reproducible from cached samples.

    python stef_validation.py download   # Stage A. Needs Google Earth Engine. Writes data/ (cached samples).
    python stef_validation.py analyse    # Stage B. NO Earth Engine needed. Reads data/, writes results/.
    python stef_validation.py selftest   # Smoke test of Stage B on SYNTHETIC data (numbers are meaningless).

Why two stages: Earth Engine, WorldPop and VIIRS calls can change over time. Stage A is run once and its
output (data/) is archived with the code (Zenodo). Anyone, including a reviewer, can then rerun Stage B
and obtain the reported numbers exactly, without an Earth Engine account.

Scope: this is a classical-ML (Random Forest) land-cover transfer experiment, Sentinel-2 vs ESA WorldCover v200
(both 2021). It is NOT a GeoFM experiment and does NOT predict infrastructure risk.

Design decisions frozen BEFORE the full run (do not retune after seeing results):
  * the E (ERA5) scales, the LST scale, the thresholds in classify_pathway, equal TS weights,
  * the predictor family and outcomes tested (Bonferroni uses the full family size).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import re
import shutil
import subprocess
import sys
import time
import warnings
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================================
# Configuration
# ============================================================================================
CFG = dict(
    year=2021,                                   # imagery year = ESA WorldCover v200 year
    bands=["B2", "B3", "B4", "B5", "B8", "B11", "B12"],
    n_pixels=800,                                # labelled pixels requested per region
    n_shift=2000,                                # unlabelled pixels per region for shift statistics
    min_per_class=15,
    class_weight=None,                           # None (main run) or "balanced" (robustness run: --balanced)
    classes_allowed=[10, 30, 40, 50],            # the chapter's four classes; others (e.g. water 80) are ignored in every pair
    seeds=list(range(10)),
    budgets=[50, 100, 200],                      # target labels for adaptation / target-only baseline
    block_deg=0.1,                               # spatial block size (about 11 km)
    holdout=0.3,                                 # share of target pixels (whole blocks) held out
    n_trees=200,
    radius_m=15000,                              # equal sampling frame (disc) for all non-corridor regions
    corridor_buffer_m=5000,                      # corridor buffer (as in the chapter)
    e_scale_T=5.0, e_scale_P=500.0,              # E kernel scales (frozen; illustrative, uncalibrated)
    lst_scale_K=5.0,                             # III_LST = 1 - exp(-SD_LST / 5 K) (frozen)
    n_boot=4000, n_weight_draws=5000,
    master_seed=20261031,
    sources=["Rhine-Ruhr DE", "Po Valley IT"],   # two European source regions
)


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


# ============================================================================================
# Regions. Centres are approximate and editable. Non-corridor regions are 15 km-radius discs
# (equal sampling frame, addressing the "bounding box vs 5 km buffer" criticism).
# 'grp' marks Europe/US vs elsewhere; 'kind' drives which analysis set a region belongs to.
# ============================================================================================
def _city(lon, lat, grp="other", kind="city"):
    return dict(kind=kind, lon=lon, lat=lat, grp=grp)


REGIONS: dict[str, dict] = {
    # --- Europe / US controls (also usable as sources) ---
    "Rhine-Ruhr DE": _city(7.20, 51.45, "EU/US", "control"),
    "Bavaria DE": _city(11.40, 48.20, "EU/US", "control"),
    "Po Valley IT": _city(9.40, 45.20, "EU/US", "control"),
    "Houston US": _city(-95.35, 29.80, "EU/US", "control"),
    # --- Africa ---
    "Lagos NG": _city(3.45, 6.55), "Ibadan NG": _city(3.90, 7.40), "Abuja NG": _city(7.50, 9.05),
    "Kano NG": _city(8.55, 12.00), "Kaduna NG": _city(7.45, 10.52), "Port Harcourt NG": _city(7.00, 4.82),
    "Accra GH": _city(-0.15, 5.65), "Nairobi KE": _city(36.90, -1.25), "Johannesburg ZA": _city(28.10, -26.15),
    "Cairo EG": _city(31.30, 30.05), "Addis Ababa ET": _city(38.78, 9.02),
    "Dar es Salaam TZ": _city(39.25, -6.80), "Kampala UG": _city(32.58, 0.32),
    # --- South and South-East Asia, Latin America ---
    "Pune IN": _city(73.85, 18.55), "Chennai IN": _city(80.15, 13.05), "Jaipur IN": _city(75.80, 26.85),
    "Ahmedabad IN": _city(72.60, 23.05), "Surat IN": _city(72.82, 21.20), "Mumbai IN": _city(72.95, 19.10),
    "Delhi NCR IN": _city(77.20, 28.55), "Dhaka BD": _city(90.40, 23.80), "Karachi PK": _city(67.10, 24.90),
    "Jakarta ID": _city(106.85, -6.25), "Sao Paulo BR": _city(-46.60, -23.60),
}

# Corridor polylines [lon, lat], traced from published maps (approximate; as-built centrelines are not public)
AKK_NAMES = ["Ajaokuta", "Baro", "Suleja", "Kaduna", "Zaria", "Kano"]
AKK_PTS = [[6.6667, 7.5000], [6.7500, 8.5500], [7.1806, 9.1817], [7.4383, 10.5222], [7.7000, 11.0667], [8.5920, 12.0022]]
DMIC_NAMES = ["Dadri", "Alwar", "Jaipur", "Ahmedabad", "Dholera", "Surat", "JNPT-Mumbai"]
DMIC_PTS = [[77.5537, 28.5546], [76.6346, 27.5530], [75.7873, 26.9124], [72.5714, 23.0225],
            [72.1833, 22.2500], [72.8311, 21.1702], [72.9525, 18.9490]]


# Named nodes and midpoints used in the chapter's Tables 8 and 9 (from the Visual Evidence document; given there as lat, lon).
# They are used ONLY to test the "tautology" criticism quantitatively against systematic points.
CORRIDOR_NODES = {
    "AKK (Nigeria)": [("node", "Baro", 8.550, 6.750), ("node", "Suleja", 9.182, 7.181), ("node", "Kaduna", 10.522, 7.438),
                      ("node", "Zaria", 11.067, 7.700), ("node", "Kano", 12.002, 8.592),
                      ("midpoint", "Baro-Suleja", 8.870, 6.960), ("midpoint", "Kaduna-Zaria", 10.790, 7.570),
                      ("midpoint", "Zaria-Kano", 11.530, 8.150)],
    "DMIC (India)": [("node", "Dadri", 28.555, 77.554), ("node", "Alwar", 27.553, 76.635), ("node", "Jaipur", 26.912, 75.787),
                     ("node", "Ahmedabad/Gandhinagar", 23.023, 72.571), ("node", "Dholera SIR", 22.250, 72.183),
                     ("node", "Surat", 21.170, 72.831), ("node", "JNPT/Mumbai", 18.949, 72.953),
                     ("midpoint", "Jaipur-Ahmedabad", 24.900, 73.900)],
}
PROFILE_STEP_KM = 10      # systematic sampling interval along the traced line
PROFILE_RADIUS_M = 1000   # built-up fraction measured in a 1 km radius disc


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def densify(line, step_km):
    """Equally spaced points along a polyline [[lon, lat], ...]: (lon, lat, km_along). Linear in lon/lat per leg."""
    pts, total, nxt = [], 0.0, 0.0
    for (lo1, la1), (lo2, la2) in zip(line[:-1], line[1:]):
        d = haversine_km(lo1, la1, lo2, la2)
        while d > 0 and nxt <= total + d + 1e-9:
            f = (nxt - total) / d
            pts.append((lo1 + f * (lo2 - lo1), la1 + f * (la2 - la1), nxt))
            nxt += step_km
        total += d
    return pts


def _add_corridor(label, names, pts):
    REGIONS[label] = dict(kind="corridor", line=pts, grp="other")
    for i in range(len(pts) - 1):
        REGIONS[f"{label.split(' ')[0]} seg{i + 1} {names[i]}-{names[i + 1]}"] = dict(
            kind="segment", line=[pts[i], pts[i + 1]], grp="other", parent=label)


_add_corridor("AKK (Nigeria)", AKK_NAMES, AKK_PTS)
_add_corridor("DMIC (India)", DMIC_NAMES, DMIC_PTS)


# ============================================================================================
# STAGE A: download (Earth Engine) -> data/
# ============================================================================================
def cmd_download(args):
    import ee  # imported here so Stage B never needs it

    project = os.environ.get("EE_PROJECT")
    try:
        from kaggle_secrets import UserSecretsClient  # optional (Kaggle)
        project = project or UserSecretsClient().get_secret("EE_PROJECT")
    except Exception:
        pass
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate(auth_mode="notebook")
        ee.Initialize(project=project)

    data = Path(args.data_dir)
    (data / "samples").mkdir(parents=True, exist_ok=True)
    (data / "meta").mkdir(parents=True, exist_ok=True)
    yr, bands = CFG["year"], CFG["bands"]

    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    viirs = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate(f"{yr}-01-01", f"{yr + 1}-01-01")
             .select("avg_rad").mean())
    era = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate(f"{yr}-01-01", f"{yr + 1}-01-01")

    def retry(fn, tries=4):
        for i in range(tries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                if i == tries - 1:
                    raise
                print(f"   retry {i + 1} after error: {str(e)[:100]}")
                time.sleep(5 * 2 ** i)

    def geom(r):
        if "line" in r:
            return ee.Geometry.LineString(r["line"]).buffer(CFG["corridor_buffer_m"])
        return ee.Geometry.Point([r["lon"], r["lat"]]).buffer(CFG["radius_m"])

    def s2(aoi, cloud_filter=True):
        def mask(img):
            scl = img.select("SCL")
            return img.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))
        c = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi).filterDate(f"{yr}-01-01", f"{yr}-12-31"))
        if cloud_filter:
            c = c.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)).map(mask)
        return c

    def lst(aoi):
        def prep(img):
            qa = img.select("QA_PIXEL")
            ok = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
            return img.select("ST_B10").multiply(0.00341802).add(149.0).updateMask(ok)
        return (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(aoi)
                .filterDate(f"{yr}-01-01", f"{yr}-12-31").map(prep).median())

    for name, r in REGIONS.items():
        sl = slug(name)
        meta_p = data / "meta" / f"{sl}.json"
        if meta_p.exists() and (data / "samples" / f"{sl}_labelled.csv").exists() and not args.force:
            print(f"[skip] {name} (cached)")
            continue
        print(f"[download] {name}")
        aoi = geom(r)
        sd = zlib.crc32(name.encode()) % 10000
        coll = s2(aoi)
        img = coll.median().clip(aoi)

        stacked = img.select(bands).addBands(wc.rename("label"))
        feats = retry(lambda: stacked.sample(region=aoi, scale=30, numPixels=CFG["n_pixels"], seed=sd,
                                             geometries=True, tileScale=8).getInfo()["features"])
        rows = []
        for f in feats:
            p = f["properties"]
            if p.get("label") is None or any(p.get(b) is None for b in bands):
                continue
            rows.append([f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1]]
                        + [p[b] for b in bands] + [int(p["label"])])
        lab = pd.DataFrame(rows, columns=["lon", "lat"] + bands + ["label"])

        sf = retry(lambda: img.select(bands).sample(region=aoi, scale=20, numPixels=CFG["n_shift"], seed=sd + 1,
                                                    geometries=False, tileScale=8).getInfo()["features"])
        shift = pd.DataFrame([[f["properties"].get(b) for b in bands] for f in sf], columns=bands).dropna()

        def red(image, scale, reducer=None):
            reducer = reducer or ee.Reducer.mean()
            return retry(lambda: image.reduceRegion(reducer, aoi, scale, maxPixels=1e9, bestEffort=True).getInfo())

        rad = (red(viirs, 500).get("avg_rad") or 0.0)
        wp = (ee.ImageCollection("WorldPop/GP/100m/pop").filterBounds(aoi).filterDate("2020-01-01", "2021-01-01")
              .mosaic().clip(aoi))
        pst = red(wp, 100, ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True))
        lst_st = red(lst(aoi), 100, ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True))
        clim = red(era.select("temperature_2m").mean().addBands(era.select("total_precipitation_sum").sum()), 11132)
        n_clear = red(coll.select("B2").count(), 100).get("B2")
        meta = dict(
            name=name, kind=r["kind"], grp=r["grp"], parent=r.get("parent"),
            viirs_rad=rad,
            pop_mean=pst.get("population_mean"), pop_sd=pst.get("population_stdDev"),
            lst_mean_K=lst_st.get("ST_B10_mean"), lst_sd_K=lst_st.get("ST_B10_stdDev"),
            T_C=(clim["temperature_2m"] - 273.15) if clim.get("temperature_2m") is not None else None,
            P_mm=(clim["total_precipitation_sum"] * 1000.0) if clim.get("total_precipitation_sum") is not None else None,
            n_scenes_lt40cloud=int(retry(lambda: coll.size().getInfo())),
            n_scenes_all=int(retry(lambda: s2(aoi, cloud_filter=False).size().getInfo())),
            mean_clear_obs_per_pixel=n_clear,
            n_labelled=int(len(lab)), n_shift=int(len(shift)), year=yr,
        )
        lab.to_csv(data / "samples" / f"{sl}_labelled.csv", index=False)
        shift.to_csv(data / "samples" / f"{sl}_shift.csv", index=False)
        meta_p.write_text(json.dumps(clean(meta), indent=2))
        print(f"   {len(lab)} labelled px; classes {dict(Counter(lab['label']))}")

    # Quantitative replacement for the chapter's Tables 8/9: WorldCover built-up fraction (class 50, 10 m) in a 1 km disc
    # at equally spaced points along each traced corridor, plus the chapter's named nodes/midpoints for comparison.
    prof_p = data / "corridor_builtup_profile.csv"
    if not prof_p.exists() or args.force:
        built = wc.eq(50).rename("built")
        recs = []
        for label in ["AKK (Nigeria)", "DMIC (India)"]:
            pts = [("systematic", f"km{k:.0f}", lo, la, k) for lo, la, k in densify(REGIONS[label]["line"], PROFILE_STEP_KM)]
            pts += [(t, n, lon, lat, np.nan) for t, n, lat, lon in CORRIDOR_NODES[label]]
            fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lo, la]).buffer(PROFILE_RADIUS_M), {"i": i})
                                       for i, (_, _, lo, la, _) in enumerate(pts)])
            out = retry(lambda: built.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10, tileScale=4).getInfo()["features"])
            val = {f["properties"]["i"]: f["properties"].get("mean") for f in out}
            for i, (t, n, lo, la, k) in enumerate(pts):
                recs.append(dict(corridor=label, point_type=t, name=n, lon=lo, lat=la, km_along=k, builtup_frac_1km=val.get(i)))
        pd.DataFrame(recs).to_csv(prof_p, index=False)
        print("   corridor built-up profile:", len(recs), "points")
    print("Download finished. Archive the data/ folder together with this script.")


# ============================================================================================
# Shift measures and transfer experiment (functions taken from the original notebooks)
# ============================================================================================
def clean(o):
    """JSON-safe: NaN/inf -> null, numpy -> python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if np.isfinite(v) else None
    return o


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(clean(obj), f, indent=2, allow_nan=False)


def bhattacharyya(a, b, bins=50):
    """Returns (SSI_b, D_B). BC = sum sqrt(p q) is the Bhattacharyya COEFFICIENT (not a distance).
    D_B = -ln BC is the Bhattacharyya DISTANCE. SSI_b = 1 - exp(-D_B) = 1 - BC in [0,1]
    (this is also the squared Hellinger distance)."""
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 0.0, 0.0
    pa, _ = np.histogram(a, bins=bins, range=(lo, hi))
    pb, _ = np.histogram(b, bins=bins, range=(lo, hi))
    pa, pb = pa / max(pa.sum(), 1), pb / max(pb.sum(), 1)
    bc = float(np.clip(np.sum(np.sqrt(pa * pb)), 1e-12, 1.0))
    return 1.0 - bc, float(-np.log(bc))


def _rf(seed, n=None, **kw):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=n or CFG["n_trees"], random_state=seed, n_jobs=-1,
                                  class_weight=CFG["class_weight"], **kw)


def domain_auc(Xs, Xt, seed=0, n_max=800):
    """Cross-validated AUC of a classifier separating source from target pixels (0.5 = indistinguishable).
    Random CV: spatial autocorrelation can inflate it; use for relative comparison only."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    rng = np.random.default_rng(seed)
    m = min(len(Xs), len(Xt), n_max)
    if m < 10:
        return np.nan
    a, b = Xs[rng.choice(len(Xs), m, replace=False)], Xt[rng.choice(len(Xt), m, replace=False)]
    X, y = np.vstack([a, b]), np.r_[np.zeros(m), np.ones(m)]
    p = cross_val_predict(_rf(seed, n=max(20, CFG["n_trees"] // 2)), X, y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def domain_auc_class_matched(Xs, ys, Xt, yt, classes, seed=0):
    vals = []
    for c in classes:
        a, b = Xs[ys == c], Xt[yt == c]
        if len(a) >= CFG["min_per_class"] and len(b) >= CFG["min_per_class"]:
            vals.append(domain_auc(a, b, seed=seed, n_max=300))
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def mmd_rbf(Xs, Xt, n=500, seed=0):
    from scipy.spatial.distance import pdist
    from sklearn.metrics.pairwise import rbf_kernel
    rng = np.random.default_rng(seed)
    a = Xs[rng.choice(len(Xs), min(n, len(Xs)), replace=False)]
    b = Xt[rng.choice(len(Xt), min(n, len(Xt)), replace=False)]
    Z = np.vstack([a, b])
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    a, b = (a - mu) / sd, (b - mu) / sd
    g = 1.0 / (2 * np.median(pdist(np.vstack([a, b]))) ** 2 + 1e-12)
    return float(rbf_kernel(a, a, gamma=g).mean() + rbf_kernel(b, b, gamma=g).mean() - 2 * rbf_kernel(a, b, gamma=g).mean())


def classify_pathway(ssi, ntl):
    """Ordered, exhaustive ILLUSTRATIVE rule from the chapter (thresholds uncalibrated; evaluated, not assumed)."""
    if ssi >= 0.5 or ntl >= 0.75:
        return "C"
    if ssi < 0.2 and ntl < 0.4:
        return "A"
    return "B"


def env_similarity(ms, mt):
    if any(v is None for v in (ms.get("T_C"), mt.get("T_C"), ms.get("P_mm"), mt.get("P_mm"))):
        return np.nan
    dT, dP = ms["T_C"] - mt["T_C"], ms["P_mm"] - mt["P_mm"]
    return float(np.exp(-0.5 * ((dT / CFG["e_scale_T"]) ** 2 + (dP / CFG["e_scale_P"]) ** 2)))


def block_ids(lon, lat, deg):
    return (np.floor(np.asarray(lon) / deg).astype(np.int64) * 1000003 + np.floor(np.asarray(lat) / deg).astype(np.int64))


def block_holdout(groups, frac, rng):
    ug = np.unique(groups).copy()
    rng.shuffle(ug)
    hold, n = [], 0
    for g in ug:
        if n >= frac * len(groups):
            break
        hold.append(g)
        n += int((groups == g).sum())
    mask = np.isin(groups, hold)
    return np.where(~mask)[0], np.where(mask)[0]          # (pool, holdout)


def macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0))


def run_transfer(Xs, ys, gs, Xt, yt, gt, classes):
    """Spatially blocked transfer. Per seed: block hold-out in source and target; zero-shot; adaptation
    (source + n target labels) and target-only baseline for each budget n. Confusion matrices and per-class F1
    are POOLED / AVERAGED over seeds."""
    from sklearn.metrics import confusion_matrix, f1_score
    seeds, budgets, frac = CFG["seeds"], CFG["budgets"], CFG["holdout"]
    L = len(classes)
    cm_zs = np.zeros((L, L), int)
    cm_ad = {n: np.zeros((L, L), int) for n in budgets}
    pc_zs, pc_ad, rows = [], {n: [] for n in budgets}, []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        s_pool, s_hold = block_holdout(gs, frac, rng)
        f1_in = np.nan
        if len(s_hold) and len(s_pool):
            f1_in = macro_f1(ys[s_hold], _rf(seed).fit(Xs[s_pool], ys[s_pool]).predict(Xs[s_hold]))
        t_pool, t_hold = block_holdout(gt, frac, rng)
        if len(t_hold) == 0 or len(t_pool) == 0:
            continue
        p_zs = _rf(seed).fit(Xs, ys).predict(Xt[t_hold])
        row = dict(seed=seed, f1_in_source_blocked=f1_in, f1_zeroshot=macro_f1(yt[t_hold], p_zs), n_holdout=len(t_hold))
        cm_zs += confusion_matrix(yt[t_hold], p_zs, labels=classes)
        pc_zs.append(f1_score(yt[t_hold], p_zs, labels=classes, average=None, zero_division=0))
        for n in budgets:
            idx = rng.choice(t_pool, min(n, len(t_pool)), replace=False)
            p_ad = _rf(seed).fit(np.vstack([Xs, Xt[idx]]), np.r_[ys, yt[idx]]).predict(Xt[t_hold])
            row[f"f1_adapted_{n}"] = macro_f1(yt[t_hold], p_ad)
            w = np.r_[np.ones(len(ys)), np.full(len(idx), len(ys) / len(idx))]       # frozen: target total weight = source total weight
            p_w = _rf(seed).fit(np.vstack([Xs, Xt[idx]]), np.r_[ys, yt[idx]], sample_weight=w).predict(Xt[t_hold])
            row[f"f1_adaptedw_{n}"] = macro_f1(yt[t_hold], p_w)
            row[f"f1_targetonly_{n}"] = (macro_f1(yt[t_hold], _rf(seed).fit(Xt[idx], yt[idx]).predict(Xt[t_hold]))
                                         if len(np.unique(yt[idx])) >= 2 else np.nan)
            cm_ad[n] += confusion_matrix(yt[t_hold], p_ad, labels=classes)
            pc_ad[n].append(f1_score(yt[t_hold], p_ad, labels=classes, average=None, zero_division=0))
        rows.append(row)
    det = dict(
        classes=[int(c) for c in classes], pooled_over_seeds=True, n_seeds=len(rows),
        cm_zeroshot=cm_zs.tolist(), cm_adapted={str(n): cm_ad[n].tolist() for n in budgets},
        f1_per_class_zeroshot=(np.mean(pc_zs, 0).tolist() if pc_zs else None),
        f1_per_class_adapted={str(n): (np.mean(pc_ad[n], 0).tolist() if pc_ad[n] else None) for n in budgets})
    return pd.DataFrame(rows), det


# ============================================================================================
# Transferability score (predictors only) and statistics
# ============================================================================================
def ts_components(ssi, ntl, iii, E):
    return dict(D=1 - ssi, E=E, Q=1 - ntl, C=1 - iii)


def ts_pre(comp, w=(0.25, 0.25, 0.25, 0.25)):
    v = np.array([comp["D"], comp["E"], comp["Q"], comp["C"]], float)
    w = np.array(w, float)
    ok = ~np.isnan(v)
    return float((w[ok] * v[ok]).sum() / w[ok].sum()) if ok.any() else np.nan


def spearman_boot(x, y, n_boot=None, seed=0):
    from scipy.stats import spearmanr
    n_boot = n_boot or CFG["n_boot"]
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(rho=np.nan, p=np.nan, lo=np.nan, hi=np.nan, n=int(len(x)))
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) and np.ptp(y[i]):
            bs.append(spearmanr(x[i], y[i])[0])
    return dict(rho=float(rho), p=float(p), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=int(len(x)))


PREDICTORS = ["SSI", "SSI_7B", "MMD", "domain_auc", "domain_auc_cm", "E", "NTL_PROXY", "III_POP", "III_2C", "TS_pre", "TS_min"]
OUTCOMES = ["f1_zeroshot", "drop_zeroshot", "f1_adapted_50", "gain_50_vs_targetonly", "gainw_50_vs_targetonly"]


def weight_sensitivity(tab, outcome):
    from scipy.stats import spearmanr
    M = np.c_[1 - tab.SSI, tab.E, 1 - tab.NTL_PROXY, 1 - tab.III_POP]
    ok = ~np.isnan(M).any(1) & ~tab[outcome].isna().values
    M, y = M[ok], tab[outcome].values[ok]
    if len(y) < 6:
        return None
    rng = np.random.default_rng(CFG["master_seed"])
    rhos = np.array([spearmanr(M @ w, y)[0] for w in rng.dirichlet(np.ones(4), CFG["n_weight_draws"])])
    rhos = rhos[~np.isnan(rhos)]
    abl = {nm: float(spearmanr(M[:, cols].mean(1), y)[0]) for nm, cols in
           {"D_only": [0], "D+E": [0, 1], "D+Q+C": [0, 2, 3], "D+C": [0, 3], "E+Q+C (no spectral)": [1, 2, 3]}.items()}
    return dict(n=int(len(y)), median=float(np.median(rhos)), p05=float(np.percentile(rhos, 5)), p95=float(np.percentile(rhos, 95)),
                min=float(rhos.min()), max=float(rhos.max()), share_same_sign_as_equal_weights=float(
                    (np.sign(rhos) == np.sign(spearmanr(M.mean(1), y)[0])).mean()), ablations=abl)


def loo_range(tab, col, outcome):
    from scipy.stats import spearmanr
    d = tab[[col, outcome]].dropna()
    v = [spearmanr(d[col].drop(d.index[i]), d[outcome].drop(d.index[i]))[0] for i in range(len(d))]
    return (float(min(v)), float(max(v))) if v else (np.nan, np.nan)


# ============================================================================================
# STAGE B: analyse (no Earth Engine)
# ============================================================================================
def load_region(data_dir: Path, name: str):
    sl = slug(name)
    lp, sp, mp = data_dir / "samples" / f"{sl}_labelled.csv", data_dir / "samples" / f"{sl}_shift.csv", data_dir / "meta" / f"{sl}.json"
    if not (lp.exists() and sp.exists() and mp.exists()):
        return None
    lab, shift, meta = pd.read_csv(lp), pd.read_csv(sp), json.loads(mp.read_text())
    sig = hashlib.md5(b"".join(p_.read_bytes() for p_ in (lp, sp, mp))).hexdigest()
    return dict(name=name, sig=sig, X=lab[CFG["bands"]].values.astype(float), y=lab["label"].values.astype(int),
                g=block_ids(lab["lon"].values, lab["lat"].values, CFG["block_deg"]),
                shift=shift[CFG["bands"]].values.astype(float), meta=meta)


def region_characteristics(reg):
    m = reg["meta"]
    pop_cv = (m.get("pop_sd") or 0) / ((m.get("pop_mean") or 0) + 1e-9)
    lst_sd = m.get("lst_sd_K")
    iii_pop = 1 - np.exp(-pop_cv)
    iii_lst = (1 - np.exp(-lst_sd / CFG["lst_scale_K"])) if lst_sd is not None else np.nan
    return dict(viirs_rad=m.get("viirs_rad"), pop_cv=pop_cv, III_POP=iii_pop, lst_sd_K=lst_sd, III_LST=iii_lst,
                III_2C=float(np.nanmean([iii_pop, iii_lst])), T_C=m.get("T_C"), P_mm=m.get("P_mm"),
                n_scenes_lt40cloud=m.get("n_scenes_lt40cloud"), n_scenes_all=m.get("n_scenes_all"),
                mean_clear_obs_per_pixel=m.get("mean_clear_obs_per_pixel"))


def common_classes(src, tgt):
    cs, ct = Counter(src["y"]), Counter(tgt["y"])
    return sorted(c for c in cs if c in CFG["classes_allowed"] and cs[c] >= CFG["min_per_class"]
                  and ct.get(c, 0) >= CFG["min_per_class"])


def run_pair(src, tgt):
    classes = common_classes(src, tgt)
    if len(classes) < 2:
        print("   too few shared classes:", tgt["name"])
        return None
    ms, mt = np.isin(src["y"], classes), np.isin(tgt["y"], classes)
    Xs, ys, gs, Xt, yt, gt = src["X"][ms], src["y"][ms], src["g"][ms], tgt["X"][mt], tgt["y"][mt], tgt["g"][mt]
    ib = {b: i for i, b in enumerate(CFG["bands"])}
    per = {b: bhattacharyya(src["shift"][:, ib[b]], tgt["shift"][:, ib[b]]) for b in CFG["bands"]}
    cs, ct = region_characteristics(src), region_characteristics(tgt)
    ratio = (ct["viirs_rad"] or 0) / ((cs["viirs_rad"] or 0) + 1e-9)
    idx = dict(
        SSI=float(np.mean([per["B5"][0], per["B11"][0]])), SSI_B5=per["B5"][0], SSI_B11=per["B11"][0],
        DB_B5B11=float(np.mean([per["B5"][1], per["B11"][1]])), SSI_7B=float(np.mean([v[0] for v in per.values()])),
        domain_auc=domain_auc(Xs, Xt), domain_auc_cm=domain_auc_class_matched(Xs, ys, Xt, yt, classes),
        MMD=mmd_rbf(Xs, Xt), E=env_similarity(src["meta"], tgt["meta"]),
        NTL_RATIO=ratio, NTL_PROXY=float(np.clip(1 - ratio, 0, 1)),     # night-light ACTIVITY proxy (not label scarcity)
        III_POP=ct["III_POP"], III_LST=ct["III_LST"], III_2C=ct["III_2C"])
    comp = ts_components(idx["SSI"], idx["NTL_PROXY"], idx["III_POP"], idx["E"])
    idx.update(TS_pre=ts_pre(comp), TS_min=float(np.nanmin(list(comp.values()))))
    idx["pathway_rule"] = classify_pathway(idx["SSI"], idx["NTL_PROXY"])
    df, det = run_transfer(Xs, ys, gs, Xt, yt, gt, classes)
    return dict(indices=idx, classes=classes, seed_results=df, details=det, n_source=int(len(ys)), n_target=int(len(yt)))


def pair_row(source, tgt, res):
    df, i = res["seed_results"], res["indices"]
    pc = lambda c, f=np.nanmean: float(f(df[c])) if c in df and df[c].notna().any() else np.nan  # noqa: E731
    r = dict(source=source, target=tgt["name"], kind=tgt["meta"]["kind"], grp=tgt["meta"]["grp"], parent=tgt["meta"].get("parent"),
             n_classes=len(res["classes"]), classes="/".join(map(str, res["classes"])), n_source=res["n_source"], n_target=res["n_target"], **i)
    r["ng_in"] = bool(tgt["name"].endswith((" NG", " IN")) or tgt["meta"]["kind"] in ("segment", "corridor"))
    r.update(f1_in=pc("f1_in_source_blocked"), f1_zeroshot=pc("f1_zeroshot"),
             f1_zs_lo=float(np.nanpercentile(df.f1_zeroshot, 2.5)), f1_zs_hi=float(np.nanpercentile(df.f1_zeroshot, 97.5)))
    for n in CFG["budgets"]:
        a, t = df[f"f1_adapted_{n}"], df[f"f1_targetonly_{n}"]
        d = (a - t).dropna()
        r.update({f"f1_adapted_{n}": float(a.mean()), f"f1_targetonly_{n}": float(t.mean()),
                  f"gain_{n}_vs_targetonly": float(d.mean()) if len(d) else np.nan,
                  f"gain_{n}_lo": float(np.percentile(d, 2.5)) if len(d) else np.nan,
                  f"gain_{n}_hi": float(np.percentile(d, 97.5)) if len(d) else np.nan,
                  f"gain_{n}_pos_seeds": int((d > 0).sum()), f"gain_{n}_n_seeds": int(len(d)),
                  f"gain_{n}_vs_zeroshot": float(a.mean() - df.f1_zeroshot.mean())})
        aw = df[f"f1_adaptedw_{n}"]
        dw = (aw - t).dropna()
        r.update({f"f1_adaptedw_{n}": float(aw.mean()), f"gainw_{n}_vs_targetonly": float(dw.mean()) if len(dw) else np.nan,
                  f"gainw_{n}_lo": float(np.percentile(dw, 2.5)) if len(dw) else np.nan,
                  f"gainw_{n}_hi": float(np.percentile(dw, 97.5)) if len(dw) else np.nan,
                  f"gainw_{n}_pos_seeds": int((dw > 0).sum())})
    r["drop_zeroshot"] = r["f1_in"] - r["f1_zeroshot"]
    return r


def analyse_source(source, regions, out: Path):
    src = regions[source]
    rows, perclass, cms = [], [], {}
    (out / "seed_results").mkdir(exist_ok=True, parents=True)
    for name, tgt in regions.items():
        if name == source:
            continue
        print(f"[pair] {source} -> {name}")
        key = hashlib.md5((json.dumps({k: CFG[k] for k in ["seeds", "budgets", "block_deg", "holdout", "n_trees", "min_per_class", "classes_allowed", "class_weight"]},
                                      sort_keys=True) + src["sig"] + tgt["sig"]).encode()).hexdigest()
        cf = out / "pair_cache" / f"{slug(source)}__{slug(name)}.pkl"
        cf.parent.mkdir(parents=True, exist_ok=True)
        res = None
        if cf.exists():
            try:
                cached = pickle.loads(cf.read_bytes())
                if cached.get("key") == key:
                    res = cached["res"]
                    print("   (resumed from cache)")
            except Exception:  # noqa: BLE001
                res = None
        if res is None:
            res = run_pair(src, tgt)
            if res is None:
                continue
            cf.write_bytes(pickle.dumps(dict(key=key, res=res)))
        rows.append(pair_row(source, tgt, res))
        res["seed_results"].to_csv(out / "seed_results" / f"{slug(source)}__{slug(name)}.csv", index=False)
        det = res["details"]
        cms[name] = det
        for k, c in enumerate(det["classes"]):
            rec = dict(source=source, target=name, worldcover_class=c, f1_zeroshot=det["f1_per_class_zeroshot"][k] if det["f1_per_class_zeroshot"] else np.nan)
            for n in CFG["budgets"]:
                v = det["f1_per_class_adapted"][str(n)]
                rec[f"f1_adapted_{n}"] = v[k] if v else np.nan
            perclass.append(rec)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / f"pair_table__{slug(source)}.csv", index=False)      # partial result, safe if the session stops
    tab["in_set_segments"] = tab.kind.isin(["city", "control", "segment"])    # primary analysis set
    tab["in_set_whole"] = tab.kind.isin(["city", "control", "corridor"])      # sensitivity: whole corridors
    return tab, pd.DataFrame(perclass), cms


def correlations(tab, source):
    rows, wsens, loo = [], {}, {}
    for set_name, col in [("segments", "in_set_segments"), ("whole_corridors", "in_set_whole")]:
        d0 = tab[tab[col]]
        subsets = {"all": d0, "chapter_four_classes": d0[d0.classes == "10/30/40/50"],
                   "four_classes_cities_controls": d0[(d0.classes == "10/30/40/50") & d0.kind.isin(["city", "control"])],
                   "four_classes_corridor_segments": d0[(d0.classes == "10/30/40/50") & (d0.kind == "segment")],
                   "four_classes_Nigeria_India": d0[(d0.classes == "10/30/40/50") & d0.ng_in], "non_EU_US": d0[d0.grp == "other"],
                   "Nigeria_India_only": d0[d0.ng_in]}
        for sub, d in subsets.items():
            for p in PREDICTORS:
                for o in OUTCOMES:
                    r = spearman_boot(d[p], d[o], seed=CFG["master_seed"])
                    rows.append(dict(source=source, analysis_set=set_name, subset=sub, predictor=p, outcome=o, **r,
                                     p_bonferroni=min(1.0, r["p"] * len(PREDICTORS)) if not np.isnan(r["p"]) else np.nan))
        wsens[set_name] = {o: weight_sensitivity(d0, o) for o in ["f1_zeroshot", "f1_adapted_50"]}
        loo[set_name] = {p: loo_range(d0, p, "f1_zeroshot") for p in ["SSI", "MMD", "TS_pre"]}
    return pd.DataFrame(rows), wsens, loo


def pathway_eval(tab, source):
    d = tab[tab.in_set_segments & (tab.classes == "10/30/40/50")].copy()
    d["ssi_tier"] = pd.cut(d.SSI, [-1, 0.2, 0.5, 2], labels=["<0.2", "0.2-0.5", ">=0.5"], right=False)
    out = []
    for key in ["pathway_rule", "ssi_tier"]:
        g = d.groupby(key, observed=True).agg(n=("target", "size"), mean_zeroshot_f1=("f1_zeroshot", "mean"),
                                              mean_adapted_50=("f1_adapted_50", "mean"),
                                              mean_gain_vs_targetonly_50=("gain_50_vs_targetonly", "mean"),
                                              mean_gain_vs_targetonly_200=("gain_200_vs_targetonly", "mean"),
                                              mean_gainw_vs_targetonly_50=("gainw_50_vs_targetonly", "mean"),
                                              mean_gainw_vs_targetonly_200=("gainw_200_vs_targetonly", "mean")).reset_index()
        g.insert(0, "grouping", key)
        g = g.rename(columns={key: "group"})
        g.insert(0, "source", source)
        out.append(g)
    return pd.concat(out)


def make_figures(tab, source, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = tab[tab.in_set_segments & (tab.classes == "10/30/40/50")]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
    for a, col, lab in zip(ax, ["SSI", "MMD", "domain_auc_cm", "TS_pre"],
                           ["SSI (1 - BC, B5/B11)", "MMD (RBF)", "Class-matched domain AUC", "TS_pre (similarity)"]):
        for grp, mk in [("EU/US", "s"), ("other", "o")]:
            s = d[d.grp == grp]
            a.errorbar(s[col], s.f1_zeroshot, yerr=[s.f1_zeroshot - s.f1_zs_lo, s.f1_zs_hi - s.f1_zeroshot], fmt=mk, capsize=2, label=grp)
        r = spearman_boot(d[col], d.f1_zeroshot, n_boot=500)
        a.set_title(f"rho = {r['rho']:+.2f}, n = {r['n']}", fontsize=9)
        a.set_xlabel(lab)
        a.set_ylabel("Zero-shot macro-F1 (blocked hold-out)")
        a.legend(fontsize=7)
    fig.suptitle(f"Source: {source}. Error bars: 2.5-97.5 percentile over {len(CFG['seeds'])} seeds.", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"fig_predictors_vs_zeroshot__{slug(source)}.png", dpi=200)
    plt.close(fig)
    # learning curves for the two corridors and the European control
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for a, name in zip(axs, ["AKK (Nigeria)", "DMIC (India)", "Bavaria DE"]):
        t = tab[tab.target == name]
        if t.empty:
            a.set_visible(False)
            continue
        t = t.iloc[0]
        b = CFG["budgets"]
        a.axhline(t.f1_zeroshot, ls=":", c="grey", label="zero-shot")
        a.plot(b, [t[f"f1_adapted_{n}"] for n in b], "o-", label="source + target labels")
        a.plot(b, [t[f"f1_targetonly_{n}"] for n in b], "s--", label="target labels only")
        a.set_title(name, fontsize=10)
        a.set_xlabel("Target labels")
    axs[0].set_ylabel("Macro-F1")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"fig_learning_curves__{slug(source)}.png", dpi=200)
    plt.close(fig)


def corridor_profile_summary(data: Path, out: Path):
    """Is 'high built-up density at named nodes' an artefact of choosing cities? Compare nodes vs systematic points."""
    p = data / "corridor_builtup_profile.csv"
    if not p.exists():
        print("No corridor_builtup_profile.csv (run download).")
        return None
    from scipy.stats import mannwhitneyu
    d = pd.read_csv(p)
    d.to_csv(out / "corridor_builtup_profile.csv", index=False)
    rows = []
    for c, g in d.groupby("corridor"):
        sysv = g[g.point_type == "systematic"].builtup_frac_1km.dropna()
        for t in ["systematic", "node", "midpoint"]:
            v = g[g.point_type == t].builtup_frac_1km.dropna()
            if len(v):
                rows.append(dict(corridor=c, point_type=t, n=len(v), mean=float(v.mean()), median=float(v.median()),
                                 p25=float(v.quantile(.25)), p75=float(v.quantile(.75)), share_gt_10pct=float((v > .10).mean())))
        nod = g[g.point_type == "node"].builtup_frac_1km.dropna()
        if len(nod) >= 3 and len(sysv) >= 3:
            rows.append(dict(corridor=c, point_type="node_vs_systematic_MannWhitney_p", n=len(nod) + len(sysv),
                             mean=float(mannwhitneyu(nod, sysv).pvalue)))
    t = pd.DataFrame(rows)
    t.to_csv(out / "corridor_builtup_summary.csv", index=False)
    return t


def versions():
    import scipy
    import sklearn
    v = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
    try:
        import matplotlib
        v["matplotlib"] = matplotlib.__version__
    except Exception:  # noqa: BLE001
        pass
    return v


def sha256(p: Path):
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def cmd_analyse(args):
    data, out = Path(args.data_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = list(REGIONS)
    regions = {n: r for n in names if (r := load_region(data, n)) is not None}
    missing = [n for n in names if n not in regions]
    if missing:
        print("WARNING: no cached data for:", missing)
    sources = [s for s in CFG["sources"] if s in regions]
    if not sources:
        sys.exit("No source region data found. Run `download` first.")

    # characteristics of every region, including the SOURCES (reviewer: source values were never computed)
    pd.DataFrame([dict(region=n, kind=r["meta"]["kind"], grp=r["meta"]["grp"], **region_characteristics(r)) for n, r in regions.items()]
                 ).to_csv(out / "region_characteristics.csv", index=False)

    all_tabs, all_pc, all_corr, all_path, summary = [], [], [], [], {}
    for source in sources:
        tab, pcl, cms = analyse_source(source, regions, out)
        corr, ws, loo = correlations(tab, source)
        all_tabs.append(tab)
        all_pc.append(pcl)
        all_corr.append(corr)
        all_path.append(pathway_eval(tab, source))
        dump_json(cms, out / f"confusion_matrices__{slug(source)}.json")
        make_figures(tab, source, out)
        sel = lambda sub: corr[(corr.analysis_set == "segments") & (corr["subset"] == sub) & (corr.outcome == "f1_zeroshot")  # noqa: E731
                               ].set_index("predictor")[["rho", "lo", "hi", "p", "p_bonferroni", "n"]].to_dict("index")
        prim = sel("chapter_four_classes")
        summary[source] = dict(n_pairs=int(len(tab)), n_primary_set=int((tab.in_set_segments & (tab.classes == "10/30/40/50")).sum()),
                               primary_correlations_vs_zeroshot_f1=prim,
                               secondary_all_common_classes_vs_zeroshot_f1=sel("all"),
                               strata_vs_zeroshot_f1={k: sel(k) for k in ["four_classes_cities_controls", "four_classes_corridor_segments", "four_classes_Nigeria_India"]},
                               adaptation_four_classes={f"budget_{b}": dict(
                                   mean_gain_vs_targetonly=float(tab[tab.in_set_segments & (tab.classes == "10/30/40/50")][f"gain_{b}_vs_targetonly"].mean()),
                                   mean_gainw_vs_targetonly=float(tab[tab.in_set_segments & (tab.classes == "10/30/40/50")][f"gainw_{b}_vs_targetonly"].mean()),
                                   n_pairs_adapted_beats_targetonly=int((tab[tab.in_set_segments & (tab.classes == "10/30/40/50")][f"gain_{b}_vs_targetonly"] > 0).sum()),
                                   n_pairs_weighted_beats_targetonly=int((tab[tab.in_set_segments & (tab.classes == "10/30/40/50")][f"gainw_{b}_vs_targetonly"] > 0).sum()))
                                   for b in CFG["budgets"]},
                               weight_sensitivity=ws, leave_one_out_rho_range=loo)
    tab_all = pd.concat(all_tabs, ignore_index=True)
    tab_all.to_csv(out / "pair_table.csv", index=False)
    pd.concat(all_pc).to_csv(out / "per_class_f1.csv", index=False)
    pd.concat(all_corr).to_csv(out / "correlations.csv", index=False)
    pd.concat(all_path).to_csv(out / "pathway_rule_evaluation.csv", index=False)

    # robustness across the two sources
    if len(sources) == 2:
        a, b = [t[t.in_set_segments].set_index("target") for t in all_tabs]
        j = a.join(b, lsuffix="_A", rsuffix="_B", how="inner")
        from scipy.stats import spearmanr
        summary["source_consistency"] = dict(
            n_common_targets=int(len(j)), sources=sources,
            rho_zeroshot_f1_between_sources=float(spearmanr(j.f1_zeroshot_A, j.f1_zeroshot_B)[0]),
            rho_SSI_between_sources=float(spearmanr(j.SSI_A, j.SSI_B)[0]))

    corridor_profile_summary(data, out)
    dump_json(summary, out / "results_summary.json")
    manifest = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), config=CFG, versions=versions(),
                    command=" ".join(sys.argv), regions_with_data=list(regions), regions_missing=missing,
                    data_sha256={str(p.relative_to(data)): sha256(p) for p in sorted(data.rglob("*")) if p.is_file()})
    try:
        manifest["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        manifest["git_commit"] = None
    dump_json(manifest, out / "run_manifest.json")
    print("\nDone. Outputs in", out)
    z = shutil.make_archive(str(out), "zip", str(out))
    print("Zipped results (download this file):", z)
    for s in sources:
        print(f"\n== {s}: PRIMARY (chapter's four classes 10/30/40/50) correlations with zero-shot macro-F1 (n={summary[s]['n_primary_set']}) ==")
        for p, v in summary[s]["primary_correlations_vs_zeroshot_f1"].items():
            print(f"  {p:14s} rho={v['rho']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] p={v['p']:.4f} p_bonf={v['p_bonferroni']:.3f}")


# ============================================================================================
# SELFTEST (synthetic data: verifies the code path only; the numbers are meaningless)
# ============================================================================================
def make_synthetic(data_dir: Path, seed=0):
    rng = np.random.default_rng(seed)
    (data_dir / "samples").mkdir(parents=True, exist_ok=True)
    (data_dir / "meta").mkdir(parents=True, exist_ok=True)
    base = {10: 0.05, 30: 0.12, 40: 0.20, 50: 0.30}
    cls_vec = {c: rng.uniform(0.02, 0.4, 7) for c in base}
    for name, r in REGIONS.items():
        sh = 0.0 if r["grp"] == "EU/US" else rng.uniform(0.05, 0.6)
        off = sh * rng.normal(0, 0.08, 7)
        cl = [10, 30, 40, 50] if rng.random() > 0.3 else [10, 40, 50]
        lon0, lat0 = (r["line"][0] if "line" in r else (r["lon"], r["lat"]))
        rows = []
        for c in cl:
            for _ in range(int(rng.integers(60, 400))):
                x = cls_vec[c] + off + rng.normal(0, 0.04 + 0.03 * sh, 7)
                rows.append([lon0 + rng.uniform(-0.4, 0.4), lat0 + rng.uniform(-0.4, 0.4), *x, c])
        lab = pd.DataFrame(rows, columns=["lon", "lat"] + CFG["bands"] + ["label"]).sample(frac=1, random_state=seed)
        lab.to_csv(data_dir / "samples" / f"{slug(name)}_labelled.csv", index=False)
        lab[CFG["bands"]].sample(min(len(lab), 600), random_state=seed).to_csv(data_dir / "samples" / f"{slug(name)}_shift.csv", index=False)
        meta = dict(name=name, kind=r["kind"], grp=r["grp"], parent=r.get("parent"), viirs_rad=float(rng.lognormal(0.5, 1.0)),
                    pop_mean=float(rng.uniform(5, 80)), pop_sd=float(rng.uniform(5, 200)), lst_mean_K=300.0, lst_sd_K=float(rng.uniform(1, 8)),
                    T_C=float(rng.uniform(10, 28)), P_mm=float(rng.uniform(300, 2000)), n_scenes_lt40cloud=40, n_scenes_all=120,
                    mean_clear_obs_per_pixel=20.0, n_labelled=len(lab), n_shift=600, year=CFG["year"])
        (data_dir / "meta" / f"{slug(name)}.json").write_text(json.dumps(meta))


def make_synthetic_profile(data_dir: Path, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for label in ["AKK (Nigeria)", "DMIC (India)"]:
        for lo, la, k in densify(REGIONS[label]["line"], PROFILE_STEP_KM):
            recs.append(dict(corridor=label, point_type="systematic", name=f"km{k:.0f}", lon=lo, lat=la, km_along=k, builtup_frac_1km=rng.beta(1, 8)))
        for t, n, lat, lon in CORRIDOR_NODES[label]:
            recs.append(dict(corridor=label, point_type=t, name=n, lon=lon, lat=lat, km_along=np.nan,
                             builtup_frac_1km=rng.beta(4, 3) if t == "node" else rng.beta(1, 12)))
    pd.DataFrame(recs).to_csv(data_dir / "corridor_builtup_profile.csv", index=False)


def cmd_selftest(args):
    import tempfile
    CFG.update(seeds=[0, 1, 2], n_trees=30, n_boot=300, n_weight_draws=200, sources=["Rhine-Ruhr DE"])
    tmp = Path(tempfile.mkdtemp(prefix="stef_selftest_"))
    make_synthetic(tmp / "data")
    make_synthetic_profile(tmp / "data")
    args.data_dir, args.out_dir = str(tmp / "data"), str(tmp / "results")
    cmd_analyse(args)
    need = ["pair_table.csv", "per_class_f1.csv", "correlations.csv", "pathway_rule_evaluation.csv", "results_summary.json",
            "run_manifest.json", "region_characteristics.csv", "corridor_builtup_summary.csv"]
    missing = [f for f in need if not (Path(args.out_dir) / f).exists()]
    json.loads((Path(args.out_dir) / "results_summary.json").read_text())      # must be strict-valid JSON (no NaN)
    print("\nSELFTEST", "FAILED, missing: " + str(missing) if missing else "PASSED (synthetic data; ignore the numbers)", "| outputs:", args.out_dir)


# ============================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["download", "analyse", "selftest"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--force", action="store_true", help="download: re-fetch regions already cached")
    ap.add_argument("--sources", default=None, help='analyse: comma-separated source regions, e.g. "Bavaria DE" or "Rhine-Ruhr DE,Po Valley IT"')
    ap.add_argument("--balanced", action="store_true", help="analyse: class_weight='balanced' in every Random Forest (robustness run)")
    args = ap.parse_args()
    if args.sources:
        CFG["sources"] = [x.strip() for x in args.sources.split(",")]
    if args.balanced:
        CFG["class_weight"] = "balanced"
    {"download": cmd_download, "analyse": cmd_analyse, "selftest": cmd_selftest}[args.command](args)


if __name__ == "__main__":
    main()
