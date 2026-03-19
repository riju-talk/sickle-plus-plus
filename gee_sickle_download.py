#!/usr/bin/env python3
"""
Download 32x32 satellite patches for every grid cell in the KML.
Sentinel-2 (12 bands), Sentinel-1 (2 bands), Landsat-8 (8 bands).
"""

import io, os, time
from datetime import datetime

import ee
import fiona
import geopandas as gpd
import numpy as np
import requests

# ── CONFIG ────────────────────────────────────────────────────────────────────

START_DATE = "2016-01-01"
END_DATE   = "2021-03-01"
MAX_IMAGES = 512

KML_FILE  = "SSSE.kml"

S2_BANDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]
S1_BANDS = ["VV","VH"]
L8_BANDS = ["SR_B1","SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7","ST_B10"]

S2_SCALE = 10   # 320m / 10m = 32 pixels
S1_SCALE = 10
L8_SCALE = 10   # resample L8 to 10m so output is also 32x32

MAX_RETRIES = 5
RETRY_WAIT  = 10

# ── PATHS ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = os.path.join(SCRIPT_DIR, "SICKLE_DATASET", "images")
os.makedirs(OUT_DIR, exist_ok=True)

# ── EARTH ENGINE ──────────────────────────────────────────────────────────────

ee.Authenticate()
ee.Initialize(project="sickle-plus-plus")
print("EE ready\n")

# ── LOAD GRID BOXES FROM KML ──────────────────────────────────────────────────

def load_grid_cells():
    kml_path = os.path.join(SCRIPT_DIR, KML_FILE)
    layers   = fiona.listlayers(kml_path)
    print("Layers:", layers)

    # Try every layer, collect all polygon geometries
    boxes = []
    for layer in layers:
        try:
            gdf = gpd.read_file(kml_path, driver="KML", layer=layer)
        except Exception as e:
            print(f"  skip layer '{layer}': {e}")
            continue

        for i, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            # Only keep actual polygon / box geometries
            if geom.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            bounds = geom.bounds   # (minx, miny, maxx, maxy) in lon/lat
            boxes.append({
                "id":     f"{layer}_{i}",
                "bounds": bounds,
            })

    # Deduplicate by bounding box (same box appears in multiple layers)
    seen   = set()
    unique = []
    for b in boxes:
        key = tuple(round(v, 6) for v in b["bounds"])
        if key not in seen:
            seen.add(key)
            unique.append(b)

    print(f"Unique grid boxes found: {len(unique)}")
    return unique


# ── BUILD EE REGION FROM BOUNDING BOX ────────────────────────────────────────

def bounds_to_ee_region(bounds):
    minx, miny, maxx, maxy = bounds
    return ee.Geometry.BBox(minx, miny, maxx, maxy)


# ── DOWNLOAD ONE IMAGE → (C, H, W) float32 ───────────────────────────────────

def image_to_numpy(image, bands, region, scale):
    url = image.getDownloadURL({
        "bands":  bands,
        "scale":  scale,
        "region": region,
        "format": "NPY",
        "crs":    "EPSG:4326",
    })
    wait = RETRY_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            break
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"      retry {attempt} in {wait}s ({exc})")
            time.sleep(wait)
            wait *= 2

    structured = np.load(io.BytesIO(r.content))
    return np.stack([structured[b].astype(np.float32) for b in bands], axis=0)


# ── COLLECTION → (T, C, H, W) ────────────────────────────────────────────────

def collection_to_timeseries(col, bands, region, scale):
    size = min(col.size().getInfo(), MAX_IMAGES)
    if size == 0:
        raise RuntimeError("empty")

    img_list       = col.toList(size)
    arrays, dates  = [], []

    for i in range(size):
        try:
            img  = ee.Image(img_list.get(i))
            ms   = img.get("system:time_start").getInfo()
            date = datetime.utcfromtimestamp(ms/1000).strftime("%Y-%m-%d")
            arr  = image_to_numpy(img, bands, region, scale)
            arrays.append(arr)
            dates.append(date)
        except Exception as exc:
            print(f"      img {i} skip: {exc}")

    if not arrays:
        raise RuntimeError("no images downloaded")

    return np.stack(arrays, axis=0), dates   # (T,C,H,W), [dates]


# ── S2 CLOUD MASK ─────────────────────────────────────────────────────────────

def mask_s2_clouds(img):
    scl  = img.select("SCL")
    return img.updateMask(scl.gte(4).And(scl.lte(6)))


# ── RESUME CHECK ─────────────────────────────────────────────────────────────

def done(uid):
    return all(
        os.path.exists(os.path.join(OUT_DIR, s, uid, "ts.npz"))
        for s in ("S2","S1","L8")
    )


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    boxes = load_grid_cells()

    for idx, box in enumerate(boxes, 1):
        uid    = box["id"]
        bounds = box["bounds"]
        print(f"\n[{idx}/{len(boxes)}] {uid}  bounds={[round(v,4) for v in bounds]}")

        if done(uid):
            print("  done, skip")
            continue

        region = bounds_to_ee_region(bounds)

        try:
            s2_col = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .select(S2_BANDS + ["SCL"])
                .filterBounds(region)
                .filterDate(START_DATE, END_DATE)
                .map(mask_s2_clouds)
                .select(S2_BANDS)
            )
            s1_col = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filter(ee.Filter.eq("instrumentMode","IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VH"))
                .select(S1_BANDS)
                .filterBounds(region)
                .filterDate(START_DATE, END_DATE)
            )
            l8_col = (
                ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                .select(L8_BANDS)
                .filterBounds(region)
                .filterDate(START_DATE, END_DATE)
            )

            s2_n, s1_n, l8_n = (
                s2_col.size().getInfo(),
                s1_col.size().getInfo(),
                l8_col.size().getInfo(),
            )
            print(f"  S2={s2_n} S1={s1_n} L8={l8_n}")

            if s2_n == s1_n == l8_n == 0:
                print("  no images, skip")
                continue

            for sensor, col, bands, scale, n in [
                ("S2", s2_col, S2_BANDS, S2_SCALE, s2_n),
                ("S1", s1_col, S1_BANDS, S1_SCALE, s1_n),
                ("L8", l8_col, L8_BANDS, L8_SCALE, l8_n),
            ]:
                if n == 0:
                    continue
                print(f"  [{sensor}] {n} images…")
                try:
                    data, dates = collection_to_timeseries(col, bands, region, scale)
                    out = os.path.join(OUT_DIR, sensor, uid)
                    os.makedirs(out, exist_ok=True)
                    np.savez_compressed(
                        os.path.join(out, "ts.npz"),
                        data  = data,
                        bands = np.array(bands, dtype=object),
                        dates = np.array(dates, dtype=object),
                        uid   = np.array(uid),
                    )
                    print(f"    saved shape={data.shape}")
                except Exception as exc:
                    print(f"    [{sensor}] ERROR: {exc}")

        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\nAll done. Output: {OUT_DIR}")

if __name__ == "__main__":
    main()