"""
generate_kml.py
---------------
Generates KML files from the CIMMYT NUE survey dataset.

Two KML files are produced:
  1. nue_all_plots.kml       – one placemark per unique (LAT, LONG) in the full
                               NUE survey dataset, coloured by Region.
  2. nue_odisha_matched.kml  – Odisha Rabi rows that have a matching entry in
                               the Odisha Kharif-2018 supplementary file,
                               showing plot-level details.

Run from the cimmyt_data folder:
    python generate_kml.py

Requirements:
    pip install pandas openpyxl
"""

import os
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ---------------------------------------------------------------------------
# Paths (relative to this script)
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
NUE_CSV    = os.path.join(BASE_DIR, "NUE_survey_dataset.csv")
ODISHA_XLS = None   # set below if file found

# The Odisha supplementary XLSX used in dataset_prep.ipynb is not stored in
# the repo; skip matched KML if it is absent.
_candidate = os.path.join(BASE_DIR, "Odisha_Kharif2018.xlsx")
if os.path.isfile(_candidate):
    ODISHA_XLS = _candidate

OUT_ALL     = os.path.join(BASE_DIR, "nue_all_plots.kml")
OUT_ODISHA  = os.path.join(BASE_DIR, "nue_odisha_matched.kml")

# ---------------------------------------------------------------------------
# Region → colour mapping  (KML colours: aabbggrr)
# ---------------------------------------------------------------------------
REGION_COLOURS = {
    "Odisha":        "ff0000ff",   # red
    "Bihar":         "ff00ff00",   # green
    "Jharkhand":     "ffff0000",   # blue
    "West Bengal":   "ff00ffff",   # yellow
    "Chhattisgarh":  "ffff00ff",   # magenta
    "Madhya Pradesh":"ff008800",   # dark green
    "Uttar Pradesh": "ff880000",   # dark blue
    "Punjab":        "ff888800",   # teal
    "Haryana":       "ff008888",   # olive
}
DEFAULT_COLOUR = "ff888888"       # grey for any unlisted region


# ---------------------------------------------------------------------------
# KML helpers
# ---------------------------------------------------------------------------

def _prettify(element: ET.Element) -> str:
    """Return a pretty-printed XML string for the Element."""
    rough = ET.tostring(element, encoding="unicode")
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ")


def _new_kml_doc(name: str) -> tuple:
    """Return (kml_root, Document_element)."""
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = name
    return kml, doc


def _add_style(doc: ET.Element, style_id: str, colour: str, scale: float = 1.0):
    """Add an icon style to the KML Document."""
    style = ET.SubElement(doc, "Style", id=style_id)
    icon_style = ET.SubElement(style, "IconStyle")
    ET.SubElement(icon_style, "color").text = colour
    ET.SubElement(icon_style, "scale").text = str(scale)
    icon = ET.SubElement(icon_style, "Icon")
    ET.SubElement(icon, "href").text = (
        "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
    )


def _add_placemark(doc: ET.Element, name: str, lat: float, lon: float,
                   description: str, style_id: str):
    """Add a Placemark to the Document."""
    pm = ET.SubElement(doc, "Placemark")
    ET.SubElement(pm, "name").text = name
    ET.SubElement(pm, "description").text = description
    ET.SubElement(pm, "styleUrl").text = f"#{style_id}"
    point = ET.SubElement(pm, "Point")
    ET.SubElement(point, "coordinates").text = f"{lon},{lat},0"


# ---------------------------------------------------------------------------
# KML 1 – All plots (unique lat/lon)
# ---------------------------------------------------------------------------

def build_all_plots_kml(df: pd.DataFrame, out_path: str):
    print(f"Building {os.path.basename(out_path)} ...")

    kml, doc = _new_kml_doc("NUE Survey – All Plots")

    # Build one style per region
    regions = df["Region"].dropna().unique().tolist()
    for region in regions:
        colour = REGION_COLOURS.get(region, DEFAULT_COLOUR)
        _add_style(doc, f"style_{region.replace(' ', '_')}", colour, scale=1.0)
    _add_style(doc, "style_unknown", DEFAULT_COLOUR, scale=0.8)

    # Deduplicate by (Region, LAT, LONG) and keep a count + Season list
    grp = (
        df.groupby(["Region", "LAT", "LONG"], dropna=True)
          .agg(
              Count=("Merged_ID", "count"),
              Seasons=("SEASON", lambda x: ", ".join(sorted(x.dropna().unique()))),
          )
          .reset_index()
    )

    for _, row in grp.iterrows():
        region  = row["Region"]
        lat     = row["LAT"]
        lon     = row["LONG"]
        count   = row["Count"]
        seasons = row["Seasons"]

        style_id = f"style_{region.replace(' ', '_')}"
        if region not in regions:
            style_id = "style_unknown"

        desc = (
            f"<b>Region:</b> {region}<br/>"
            f"<b>Records:</b> {count}<br/>"
            f"<b>Season(s):</b> {seasons}<br/>"
            f"<b>Lat/Lon:</b> {lat}, {lon}"
        )
        _add_placemark(doc, f"{region} ({lat:.3f},{lon:.3f})", lat, lon, desc, style_id)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_prettify(kml))
    print(f"  -> Written {grp.shape[0]} placemarks to {out_path}")


# ---------------------------------------------------------------------------
# KML 2 – Odisha Rabi matched to supplementary file
# ---------------------------------------------------------------------------

def build_odisha_matched_kml(df: pd.DataFrame, odisha_xls: str, out_path: str):
    print(f"Building {os.path.basename(out_path)} ...")

    # ---- reproduce dataset_prep.ipynb logic --------------------------------
    try:
        odisha_supp = pd.read_excel(odisha_xls, sheet_name="Sheet1")
    except Exception as exc:
        print(f"  Could not read Odisha XLS: {exc}")
        return

    nue_odisha = df[(df["Region"] == "Odisha") & (df["SEASON"] == "Rabi")].copy()
    odisha_supp["Latitude_round"]  = odisha_supp["Latitude"].round(2)
    odisha_supp["Longitude_round"] = odisha_supp["Longitude"].round(2)
    odisha_supp = odisha_supp.reset_index().rename(columns={"index": "id"})

    lat_long_to_ids = (
        odisha_supp
        .groupby(["Latitude_round", "Longitude_round"])["id"]
        .apply(list)
        .to_dict()
    )

    nue_odisha["matched_ids"] = nue_odisha.apply(
        lambda r: lat_long_to_ids.get((r["LAT"], r["LONG"]), []), axis=1
    )

    # Convert acres → hectares
    acre_col = "C-q306_cropLarestAreaAcre"
    ha_col   = "C-q306_cropLarestAreaHectare"
    if acre_col in odisha_supp.columns:
        odisha_supp[ha_col] = odisha_supp[acre_col] / 2.47105
    else:
        odisha_supp[ha_col] = float("nan")

    def find_best_match(row):
        ids = row["matched_ids"]
        if not ids:
            return None, None
        sub = odisha_supp.loc[odisha_supp["id"].isin(ids), ["id", ha_col]]
        if sub.empty or sub[ha_col].isna().all():
            return float(ids[0]), None
        crlparha = row.get("CRLPARHA", float("nan"))
        best = sub.iloc[(sub[ha_col] - crlparha).abs().argmin()]
        return best["id"], abs(best[ha_col] - crlparha)

    nue_odisha[["chosen_mapping_id", "mapping_area_diff"]] = nue_odisha.apply(
        lambda r: pd.Series(find_best_match(r)), axis=1
    )

    matched = nue_odisha[nue_odisha["chosen_mapping_id"].notna()].copy()
    print(f"  Matched {len(matched)} / {len(nue_odisha)} Odisha Rabi rows")

    # ---- build KML ---------------------------------------------------------
    kml, doc = _new_kml_doc("NUE Survey – Odisha Rabi Matched")
    _add_style(doc, "matched",   "ff0000ff", scale=1.2)   # red
    _add_style(doc, "unmatched", "ff888888", scale=0.8)   # grey

    # Unmatched
    unmatched = nue_odisha[nue_odisha["chosen_mapping_id"].isna()]
    for _, row in unmatched.iterrows():
        desc = f"<b>Merged_ID:</b> {row.get('Merged_ID','')}<br/>No matching supplementary record found."
        _add_placemark(doc, str(row.get("Merged_ID", "")),
                       row["LAT"], row["LONG"], desc, "unmatched")

    # Matched
    for _, row in matched.iterrows():
        area_diff = row.get("mapping_area_diff", None)
        diff_str  = f"{area_diff:.4f} ha" if pd.notna(area_diff) else "N/A"
        desc = (
            f"<b>Merged_ID:</b> {row.get('Merged_ID','')}<br/>"
            f"<b>Chosen supp. ID:</b> {int(row['chosen_mapping_id'])}<br/>"
            f"<b>Area difference:</b> {diff_str}<br/>"
            f"<b>CRLPARHA:</b> {row.get('CRLPARHA','')}<br/>"
            f"<b>Lat/Lon:</b> {row['LAT']}, {row['LONG']}"
        )
        _add_placemark(doc, str(row.get("Merged_ID", "")),
                       row["LAT"], row["LONG"], desc, "matched")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_prettify(kml))
    print(f"  -> Written {len(nue_odisha)} placemarks to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.isfile(NUE_CSV):
        raise FileNotFoundError(f"Cannot find NUE CSV at: {NUE_CSV}")

    print(f"Reading {NUE_CSV} …")
    df = pd.read_csv(NUE_CSV, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")

    # KML 1 – all unique locations
    build_all_plots_kml(df, OUT_ALL)

    # KML 2 – Odisha matched (only if supplementary XLS is present)
    if ODISHA_XLS:
        build_odisha_matched_kml(df, ODISHA_XLS, OUT_ODISHA)
    else:
        print(
            f"\nNote: Odisha_Kharif2018.xlsx not found in {BASE_DIR}.\n"
            f"  Place the file there and re-run to generate {os.path.basename(OUT_ODISHA)}."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
