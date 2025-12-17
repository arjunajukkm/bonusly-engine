import io
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Callable

import pandas as pd
import polars as pl
import streamlit as st

# =======================
# Config
# =======================
SOURCE_SHEET = "Base Data"
CATEGORIES = ["Collaboration", "Execution", "Disruption", "Integrity", "Inclusion"]
BAD_RATINGS = {
    "APPROACHING EXPECTATIONS",
    "UNSATISFACTORY",
    "LOA",
    "TOO NEW TO RATE",
}

REGION_EMEA = "Europe, Middle East & Africa"
REGION_JAPAC = "JAPAC"
MUST_EMEA = ["France", "Germany", "United Kingdom", "Spain", "Netherlands"]
MUST_JAPAC = ["Singapore", "India", "China", "Australia", "Japan"]

# Columns required in the Source Sheet
MUST_HAVE_COLUMNS = [
    "Message", "toLocation", "toEmail",
    "FromEmail", "toFirstName", "toLastName",
    "Recognition", "Gender", "Team"
]


# =======================
# Helpers
# =======================
def norm_col(s: str) -> str:
    return str(s or "").strip().lower()

def safe_str(v) -> str:
    if v is None:
        return ""
    return str(v).strip()

def parse_fy_number(fy: str) -> int:
    fy = safe_str(fy).upper()
    return int(re.sub(r"^FY", "", fy))

def fy_quarter_to_index(fy_num: int, q_num: int) -> int:
    return fy_num * 4 + (q_num - 1)

def require_columns_polars(df: pl.DataFrame, required: List[str], sheet_name: str) -> Dict[str, str]:
    cols = df.columns
    norm_map = {norm_col(c): c for c in cols}
    missing = [c for c in required if norm_col(c) not in norm_map]
    if missing:
        raise ValueError(f'Missing required columns in "{sheet_name}": {missing}')
    return norm_map

def build_country_to_region_map() -> Dict[str, str]:
    # (Same map as before)
    pairs = [
        ("Europe, Middle East & Africa", "Algeria"), ("LATAM", "Argentina"), ("JAPAC", "Australia"),
        ("Europe, Middle East & Africa", "Austria"), ("Europe, Middle East & Africa", "Azerbaijan"),
        ("JAPAC", "Bangladesh"), ("Europe, Middle East & Africa", "Belgium"), ("LATAM", "Brazil"),
        ("Europe, Middle East & Africa", "Bulgaria"), ("Americas", "Canada"), ("LATAM", "Chile"),
        ("JAPAC", "China"), ("LATAM", "Colombia"), ("LATAM", "Costa Rica"),
        ("Europe, Middle East & Africa", "Côte d'Ivoire"), ("Europe, Middle East & Africa", "Croatia"),
        ("Europe, Middle East & Africa", "Czechia"), ("Europe, Middle East & Africa", "Denmark"),
        ("LATAM", "Dominican Republic"), ("LATAM", "Ecuador"), ("Europe, Middle East & Africa", "Egypt"),
        ("Europe, Middle East & Africa", "Finland"), ("Europe, Middle East & Africa", "France"),
        ("Europe, Middle East & Africa", "Germany"), ("Europe, Middle East & Africa", "Greece"),
        ("Europe, Middle East & Africa", "Guernsey"), ("JAPAC", "Hong Kong"),
        ("Europe, Middle East & Africa", "Hungary"), ("JAPAC", "India"), ("JAPAC", "Indonesia"),
        ("Europe, Middle East & Africa", "Ireland"), ("Europe, Middle East & Africa", "Israel"),
        ("Europe, Middle East & Africa", "Italy"), ("JAPAC", "Japan"),
        ("Europe, Middle East & Africa", "Kazakhstan"), ("Europe, Middle East & Africa", "Kenya"),
        ("JAPAC", "Korea, Republic of"), ("Europe, Middle East & Africa", "Kuwait"),
        ("Europe, Middle East & Africa", "Latvia"), ("Europe, Middle East & Africa", "Lithuania"),
        ("Europe, Middle East & Africa", "Luxembourg"), ("JAPAC", "Malaysia"), ("LATAM", "Mexico"),
        ("Europe, Middle East & Africa", "Morocco"), ("Europe, Middle East & Africa", "Netherlands"),
        ("JAPAC", "New Zealand"), ("Europe, Middle East & Africa", "Nigeria"),
        ("Europe, Middle East & Africa", "Norway"), ("Europe, Middle East & Africa", "Oman"),
        ("LATAM", "Panama"), ("LATAM", "Peru"), ("JAPAC", "Philippines"),
        ("Europe, Middle East & Africa", "Poland"), ("Europe, Middle East & Africa", "Portugal"),
        ("Europe, Middle East & Africa", "Qatar"), ("Europe, Middle East & Africa", "Romania"),
        ("Europe, Middle East & Africa", "Saudi Arabia"), ("Europe, Middle East & Africa", "Senegal"),
        ("Europe, Middle East & Africa", "Serbia"), ("JAPAC", "Singapore"),
        ("Europe, Middle East & Africa", "Slovakia"), ("Europe, Middle East & Africa", "Slovenia"),
        ("Europe, Middle East & Africa", "South Africa"), ("Europe, Middle East & Africa", "Spain"),
        ("JAPAC", "Sri Lanka"), ("Europe, Middle East & Africa", "Sweden"),
        ("Europe, Middle East & Africa", "Switzerland"), ("JAPAC", "Taiwan"), ("JAPAC", "Thailand"),
        ("Europe, Middle East & Africa", "Tunisia"), ("Europe, Middle East & Africa", "Türkiye"),
        ("Europe, Middle East & Africa", "Ukraine"), ("Europe, Middle East & Africa", "United Arab Emirates"),
        ("Europe, Middle East & Africa", "United Kingdom"), ("Americas", "United States of America"),
        ("JAPAC", "Vietnam"),
    ]
    return {safe_str(country).lower(): region for region, country in pairs}


@dataclass
class Candidate:
    email_lower: str
    total: float
    toEmail: str
    toLocation: str
    gender: str
    team: str
    mandatory: bool = False


def select_emails_for_region_category(region: str, candidates_sorted: List[Candidate], cutoff: float) -> List[str]:
    # (Logic identical to previous version)
    def in_base(c: Candidate) -> bool:
        return cutoff >= 0 and c.total >= cutoff

    base_pool = [c for c in candidates_sorted if in_base(c)]
    outside_pool = [c for c in candidates_sorted if not in_base(c)]
    sel: List[Candidate] = []
    sel_set: Set[str] = set()

    def add_sel(c: Candidate, mandatory: bool = False) -> bool:
        if c.email_lower in sel_set:
            return False
        sel.append(Candidate(**{**c.__dict__, "mandatory": mandatory or c.mandatory}))
        sel_set.add(c.email_lower)
        return True

    def remove_one_non_mandatory_lowest() -> bool:
        idx = -1
        worst_score = float("inf")
        worst_email = ""
        for i, c in enumerate(sel):
            if c.mandatory: continue
            if (c.total < worst_score) or (c.total == worst_score and c.email_lower > worst_email):
                worst_score = c.total
                worst_email = c.email_lower
                idx = i
        if idx >= 0:
            removed = sel.pop(idx)
            sel_set.remove(removed.email_lower)
            return True
        return False

    def pick_best(predicate) -> Optional[Candidate]:
        for pool in (base_pool, outside_pool):
            for c in pool:
                if c.email_lower not in sel_set and predicate(c):
                    return c
        return None

    for c in base_pool: add_sel(c)

    required_list = MUST_EMEA if region == REGION_EMEA else MUST_JAPAC if region == REGION_JAPAC else []
    for loc in required_list:
        if not any(x.toLocation == loc for x in sel):
            cand = pick_best(lambda x: x.toLocation == loc)
            if cand:
                if len(sel) >= 5 and not remove_one_non_mandatory_lowest(): pass
                add_sel(cand, mandatory=True)
        else:
            for x in sel:
                if x.toLocation == loc: x.mandatory = True; break

    if not any(x.toLocation == "Israel" for x in sel):
        cand = pick_best(lambda x: x.toLocation == "Israel")
        if cand:
            if len(sel) >= 5 and not remove_one_non_mandatory_lowest(): pass
            add_sel(cand, mandatory=True)

    def female_count() -> int: return sum(1 for x in sel if safe_str(x.gender).lower() == "female")
    while female_count() < 2:
        cand = pick_best(lambda x: safe_str(x.gender).lower() == "female")
        if not cand: break
        if len(sel) >= 5 and not remove_one_non_mandatory_lowest(): break
        add_sel(cand)

    have_overflow = cutoff >= 0 and len(sel) > 5
    want = len(sel) if have_overflow else 5
    team_set = {x.team for x in sel}
    loc_set = {x.toLocation for x in sel}
    predicates = [
        lambda x: x.team not in team_set and x.toLocation not in loc_set,
        lambda x: x.team not in team_set,
        lambda x: x.toLocation not in loc_set,
        lambda x: True,
    ]
    for pred in predicates:
        while len(sel) < want:
            cand = pick_best(lambda x, p=pred: p(x))
            if not cand: break
            add_sel(cand)
            team_set.add(cand.team); loc_set.add(cand.toLocation)

    while len(sel) > 5 and not (cutoff >= 0 and all(x.total >= cutoff for x in sel)):
        if not remove_one_non_mandatory_lowest(): break

    sel.sort(key=lambda x: (-x.total, x.email_lower))
    return [x.email_lower for x in sel]


# =======================
# Validation Function
# =======================
def validate_excel_structure(file_bytes: bytes) -> List[str]:
    """Checks for missing sheets and missing columns without loading full data."""
    errors = []
    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
        sheet_names = set(xls.sheet_names)

        # 1. Check Sheet Existence
        if SOURCE_SHEET not in sheet_names:
            return [f"CRITICAL: Missing sheet '{SOURCE_SHEET}'"]

        # 2. Check Column Existence (Read headers only)
        # We perform a light read (nrows=0) just to get columns
        df_base = pd.read_excel(xls, SOURCE_SHEET, nrows=0)
        actual_cols = set(norm_col(c) for c in df_base.columns)
        
        missing_base = []
        for required in MUST_HAVE_COLUMNS:
            if norm_col(required) not in actual_cols:
                missing_base.append(required)
        
        if missing_base:
            errors.append(f"Sheet '{SOURCE_SHEET}' missing columns: {', '.join(missing_base)}")

        # Optional: Check Previous Winners headers if sheet exists
        if "Previous Winners" in sheet_names:
            df_pw = pd.read_excel(xls, "Previous Winners", nrows=0)
            pw_cols = set(norm_col(c) for c in df_pw.columns)
            missing_pw = [c for c in ["toEmail", "FY", "Quarter"] if norm_col(c) not in pw_cols]
            if missing_pw:
                errors.append(f"Sheet 'Previous Winners' missing columns: {', '.join(missing_pw)}")

    except Exception as e:
        errors.append(f"File Error: {str(e)}")
    
    return errors


# =======================
# Core processing (FIXED)
# =======================
def process_excel_to_categories_only(file_bytes: bytes, curr_fy: str, curr_q: int, progress_callback: Callable[[float, str], None] = None) -> Tuple[bytes, dict]:
    
    def update_prog(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    # --- STEP 1: PARSING ---
    update_prog(0.1, "Parsing Excel sheets...")
    
    xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
    sheet_names = set(xls.sheet_names)

    if SOURCE_SHEET not in sheet_names:
        raise ValueError(f'Sheet "{SOURCE_SHEET}" not found.')

    # Load dataframes
    base_pd = xls.parse(SOURCE_SHEET)
    prev_pd = xls.parse("Previous Winners") if "Previous Winners" in sheet_names else None
    sn_pd = xls.parse("Serving Notice") if "Serving Notice" in sheet_names else None
    rating_pd = xls.parse("Rating") if "Rating" in sheet_names else None

    base = pl.from_pandas(base_pd)
    prev = pl.from_pandas(prev_pd) if prev_pd is not None else None
    sn = pl.from_pandas(sn_pd) if sn_pd is not None else None
    rating = pl.from_pandas(rating_pd) if rating_pd is not None else None

    # --- STEP 2: JOINS & PREP ---
    update_prog(0.3, "Joining Reference Data (Regions, History)...")

    bmap = require_columns_polars(base, MUST_HAVE_COLUMNS, SOURCE_SHEET)
    
    # Map raw columns to standardized names
    col_message = bmap["message"]
    col_loc = bmap["tolocation"]
    col_email = bmap["toemail"]
    col_rec = bmap["recognition"]
    col_gender = bmap["gender"]
    col_team = bmap["team"]
    col_from = bmap["fromemail"]
    col_first = bmap["tofirstname"]
    col_last = bmap["tolastname"]

    country_to_region = build_country_to_region_map()
    region_lut = pl.DataFrame({
        "loc_key": list(country_to_region.keys()),
        "Region": list(country_to_region.values()),
    })

    base = base.with_columns([
        pl.col(col_loc).cast(pl.Utf8).fill_null("").str.strip_chars().str.to_lowercase().alias("loc_key"),
        pl.col(col_email).cast(pl.Utf8).fill_null("").str.strip_chars().str.to_lowercase().alias("email_lower"),
        pl.col(col_message).cast(pl.Utf8).fill_null("").alias("Message_norm"),
        pl.col(col_rec).cast(pl.Utf8).fill_null("").alias("Recognition_norm"),
        pl.col(col_gender).cast(pl.Utf8).fill_null("").alias("Gender_norm"),
        pl.col(col_team).cast(pl.Utf8).fill_null("").alias("Team_norm"),
    ]).join(region_lut, on="loc_key", how="left").with_columns([
        pl.col("Region").fill_null("").alias("Region")
    ])

    msg = pl.col("Message_norm")
    base = base.with_columns([
        msg.str.to_lowercase().str.count_matches(r"#collaboration\b").alias("Collaboration"),
        msg.str.to_lowercase().str.count_matches(r"#execution\b").alias("Execution"),
        msg.str.to_lowercase().str.count_matches(r"#disruption\b").alias("Disruption"),
        msg.str.to_lowercase().str.count_matches(r"#integrity\b").alias("Integrity"),
        msg.str.to_lowercase().str.count_matches(r"#inclusion\b").alias("Inclusion"),
        msg.str.count_matches(r"(^|[^\w])@([A-Za-z0-9_]+)").alias("Count of Recipients"),
    ])

    # Previous Winners Logic
    curr_idx = fy_quarter_to_index(parse_fy_number(curr_fy), int(curr_q))
    prev_map_df = None
    if prev is not None and prev.height > 0:
        pmap = require_columns_polars(prev, ["toEmail", "FY", "Quarter"], "Previous Winners")
        prev = prev.with_columns([
            pl.col(pmap["toemail"]).cast(pl.Utf8).fill_null("").str.strip_chars().str.to_lowercase().alias("email_lower"),
            pl.col(pmap["fy"]).cast(pl.Utf8).fill_null("").str.strip_chars().str.to_uppercase().alias("FY_norm"),
            pl.col(pmap["quarter"]).cast(pl.Int64).alias("Q_norm"),
        ]).filter(
            (pl.col("email_lower") != "") & (pl.col("FY_norm").str.contains(r"^FY\d{2}$")) & (pl.col("Q_norm").is_between(1, 4))
        ).with_columns([
            pl.col("FY_norm").str.replace(r"^FY", "").cast(pl.Int64).alias("FY_num"),
        ]).with_columns([
            (pl.lit(curr_idx) - (pl.col("FY_num") * 4 + (pl.col("Q_norm") - 1))).alias("diff"),
        ]).with_columns([
            ((pl.col("diff") >= 0) & (pl.col("diff") <= 8)).alias("within_2y"),
        ])
        prev_map_df = prev.group_by("email_lower").agg(pl.any("within_2y").alias("PrevWinnerWithin2Y"))

    # Serving Notice Logic
    sn_df = None
    if sn is not None and sn.height > 0:
        smap = require_columns_polars(sn, ["toEmail"], "Serving Notice")
        sn_df = sn.with_columns([
            pl.col(smap["toemail"]).cast(pl.Utf8).fill_null("").str.strip_chars().str.to_lowercase().alias("email_lower"),
        ]).filter(pl.col("email_lower") != "").select(["email_lower"]).unique().with_columns([pl.lit(True).alias("ServingNotice")])

    # Rating Logic
    rating_df = None
    if rating is not None and rating.height > 0:
        rmap = require_columns_polars(rating, ["toEmail", "Year End Rating", "Mid Year Rating"], "Rating")
        rating = rating.with_columns([
            pl.col(rmap["toemail"]).cast(pl.Utf8).fill_null("").str.strip_chars().str.to_lowercase().alias("email_lower"),
            pl.col(rmap["year end rating"]).cast(pl.Utf8).fill_null("").str.strip_chars().str.to_uppercase().alias("YE"),
            pl.col(rmap["mid year rating"]).cast(pl.Utf8).fill_null("").str.strip_chars().str.to_uppercase().alias("MY"),
        ]).filter(pl.col("email_lower") != "").with_columns([
            ((pl.col("YE").is_in(list(BAD_RATINGS))) | (pl.col("MY").is_in(list(BAD_RATINGS)))).alias("is_bad")
        ])
        rating_df = rating.group_by("email_lower").agg(pl.any("is_bad").alias("AnyBad")).with_columns([
            pl.when(pl.col("AnyBad")).then(pl.lit("No")).otherwise(pl.lit("Yes")).alias("RatingFlag")
        ]).select(["email_lower", "RatingFlag"])

    # Merging Lookups
    if prev_map_df is not None:
        base = base.join(prev_map_df, on="email_lower", how="left").with_columns([
            pl.when(pl.col("PrevWinnerWithin2Y") == True).then(pl.lit("Yes")).otherwise(pl.lit("No")).alias("Previous Winner")
        ])
    else:
        base = base.with_columns([pl.lit("No").alias("Previous Winner")])

    if sn_df is not None:
        base = base.join(sn_df, on="email_lower", how="left").with_columns([
            pl.when(pl.col("ServingNotice") == True).then(pl.lit("Yes")).otherwise(pl.lit("No")).alias("Serving Notice")
        ])
    else:
        base = base.with_columns([pl.lit("No").alias("Serving Notice")])

    if rating_df is not None:
        base = base.join(rating_df, on="email_lower", how="left").with_columns([pl.col("RatingFlag").fill_null("Yes").alias("Rating")])
    else:
        base = base.with_columns([pl.lit("Yes").alias("Rating")])

    # --- STEP 3: ELIGIBILITY ---
    update_prog(0.5, "Calculating Eligibility...")

    base = base.with_columns([
        pl.when(
            (pl.col("Recognition_norm") == "Peer to Peer") &
            (pl.col("Previous Winner") == "No") &
            (pl.col("Serving Notice") == "No") &
            (pl.col("Rating") == "Yes") &
            (pl.col("Count of Recipients") <= 3)
        ).then(pl.lit("Eligible")).otherwise(pl.lit("Non Eligible")).alias("Eligibility")
    ])

    eligible = base.filter(pl.col("Eligibility") == "Eligible")

    # --- STEP 4: SELECTION ---
    update_prog(0.7, "Running Diversity Selection Logic...")

    attrs = eligible.select([
        pl.col("email_lower"),
        pl.col(col_email).cast(pl.Utf8).fill_null("").alias("toEmail"),
        pl.col(col_loc).cast(pl.Utf8).fill_null("").alias("toLocation"),
        pl.col("Gender_norm").alias("gender"),
        pl.col("Team_norm").alias("team"),
    ]).group_by("email_lower").agg([
        pl.first("toEmail").alias("toEmail"),
        pl.first("toLocation").alias("toLocation"),
        pl.first("gender").alias("gender"),
        pl.first("team").alias("team"),
    ])

    allow: Dict[str, Dict[str, Set[str]]] = {c: {} for c in CATEGORIES}

    for cat in CATEGORIES:
        agg = eligible.group_by(["Region", "email_lower"]).agg(
            pl.sum(cat).alias("total")
        ).filter(pl.col("total") > 0).join(attrs, on="email_lower", how="left")

        regions = agg.select("Region").unique().to_series().to_list()
        for region in regions:
            reg_df = agg.filter(pl.col("Region") == region).select(
                ["email_lower", "total", "toEmail", "toLocation", "gender", "team"]
            ).sort(["total", "email_lower"], descending=[True, False])

            rows = reg_df.to_dicts()
            if not rows: continue

            candidates = [
                Candidate(
                    email_lower=r["email_lower"],
                    total=float(r["total"] or 0),
                    toEmail=safe_str(r.get("toEmail", "")),
                    toLocation=safe_str(r.get("toLocation", "")),
                    gender=safe_str(r.get("gender", "")),
                    team=safe_str(r.get("team", "")),
                )
                for r in rows
            ]

            k = min(5, len(candidates))
            cutoff = candidates[k - 1].total if candidates else -1
            selected = select_emails_for_region_category(region, candidates, cutoff)
            allow[cat][region] = set(selected)

    # --- STEP 5: WRITING ---
    update_prog(0.9, "Generating Output Excel...")

    out_tables: Dict[str, pl.DataFrame] = {}
    for cat in CATEGORIES:
        allow_rows = []
        for region, emails in allow[cat].items():
            for e in emails: allow_rows.append({"Region": region, "email_lower": e})

        if allow_rows:
            allow_df = pl.DataFrame(allow_rows).with_columns([pl.col("Region").cast(pl.Utf8), pl.col("email_lower").cast(pl.Utf8)])
        else:
            allow_df = pl.DataFrame(schema={"Region": pl.Utf8, "email_lower": pl.Utf8})

        cat_rows = eligible.filter(pl.col(cat) > 0).join(
            allow_df, on=["Region", "email_lower"], how="inner"
        ).select([
            pl.col(col_from).cast(pl.Utf8).fill_null("").alias("FromEmail"),
            pl.col(col_first).cast(pl.Utf8).fill_null("").alias("toFirstName"),
            pl.col(col_last).cast(pl.Utf8).fill_null("").alias("toLastName"),
            pl.col(col_email).cast(pl.Utf8).fill_null("").alias("toEmail"),
            pl.col(col_loc).cast(pl.Utf8).fill_null("").alias("toLocation"),
            pl.col(col_message).cast(pl.Utf8).fill_null("").alias("Message"),
            pl.col(col_rec).cast(pl.Utf8).fill_null("").alias("Recognition"),
            pl.col("Region").alias("Region"),
        ]).sort(pl.col("toEmail").str.to_lowercase())

        out_tables[cat] = cat_rows

    base_rows = base.height
    eligible_rows = eligible.height

    summary_rows = [
        {"Metric": "Base Data rows", "Value": int(base_rows)},
        {"Metric": "Eligible rows", "Value": int(eligible_rows)},
    ]
    total_out = 0
    for cat in CATEGORIES:
        n = int(out_tables[cat].height)
        total_out += n
        summary_rows.append({"Metric": f"{cat} rows", "Value": n})
    summary_rows.append({"Metric": "Total rows across category sheets", "Value": int(total_out)})
    
    summary_df = pd.DataFrame(summary_rows)
    # FIX: Renaming variable to region_counts_pd to match the usage below
    region_counts_pd = eligible.group_by("Region").agg(pl.len().alias("EligibleRows")).sort("EligibleRows", descending=True).to_pandas()

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        region_counts_pd.to_excel(writer, index=False, sheet_name="Summary_Region")
        for cat in CATEGORIES:
            out_tables[cat].to_pandas().to_excel(writer, index=False, sheet_name=cat)

    output.seek(0)
    stats = {
        "base_rows": int(base_rows),
        "eligible_rows": int(eligible_rows),
        **{f"{c}_rows": int(out_tables[c].height) for c in CATEGORIES},
    }
    
    update_prog(1.0, "Complete")
    return output.getvalue(), stats

# =======================
# UI IMPLEMENTATION
# =======================
st.set_page_config(page_title="Recognition Logic", page_icon="💠", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    header[data-testid="stHeader"] { visibility: hidden; }
    footer { visibility: hidden; }

    /* Titles */
    h1 { font-weight: 700; margin-bottom: 0.5rem; color: #F9FAFB; }
    .subtitle { color: #9CA3AF; font-size: 1.1rem; margin-bottom: 2rem; }
    h3 { color: #E5E7EB; font-size: 1rem; font-weight: 600; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 0.05em; }

    /* Stats */
    .stat-container {
        display: flex; flex-direction: column; align-items: center; padding: 15px;
        background: rgba(255, 255, 255, 0.05); border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stat-value { font-size: 1.8rem; font-weight: 700; color: #60A5FA; }
    .stat-label { font-size: 0.85rem; color: #9CA3AF; margin-top: 4px; text-align: center; }

    /* Alerts */
    .success-box {
        background-color: rgba(16, 185, 129, 0.1); border: 1px solid #10B981; color: #34D399;
        padding: 16px; border-radius: 8px; margin-bottom: 1rem; text-align: center; font-weight: 600;
    }
    .error-box {
        background-color: rgba(239, 68, 68, 0.1); border: 1px solid #EF4444; color: #F87171;
        padding: 16px; border-radius: 8px; margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>Bonusly Engine</h1>", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    with st.container(border=True):
        st.markdown("<h3>1. Configuration</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fy = st.text_input("Fiscal Year", value="FY25", placeholder="e.g. FY25")
        with c2:
            q = st.selectbox("Quarter", options=[1, 2, 3, 4], index=0)
        out_name = st.text_input("Output Filename", value="processed_results.xlsx")

    with st.container(border=True):
        st.markdown("<h3>2. Data Source</h3>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload Excel File", type=["xlsx"], label_visibility="collapsed")
        if uploaded:
            st.success(f"Loaded: {uploaded.name}")
        else:
            st.caption("Required: 'Base Data' sheet.")

with col_right:
    with st.container(border=True):
        st.markdown("<h3>3. Action</h3>", unsafe_allow_html=True)
        
        # Action Buttons Layout
        btn_col1, btn_col2 = st.columns([1, 1])
        
        with btn_col1:
            validate_btn = st.button("🔍 Validate File", disabled=(uploaded is None), use_container_width=True)
        with btn_col2:
            process_btn = st.button("🚀 Run Processing", type="primary", disabled=(uploaded is None), use_container_width=True)
        
        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
        
        # --- VALIDATION LOGIC ---
        if validate_btn:
            with st.spinner("Checking file structure..."):
                file_bytes = uploaded.read()
                uploaded.seek(0) # Reset pointer
                errors = validate_excel_structure(file_bytes)
            
            if not errors:
                st.markdown('<div class="success-box">File Validated Successfully!</div>', unsafe_allow_html=True)
                st.info("Structure looks good. Sheets and columns match requirements.")
            else:
                st.markdown('<div class="error-box">Validation Failed</div>', unsafe_allow_html=True)
                for err in errors:
                    st.error(err)

        # --- PROCESSING LOGIC ---
        if process_btn:
            try:
                fy_clean = safe_str(fy).upper()
                if not re.fullmatch(r"FY\d{2}", fy_clean):
                    st.error('Format Error: Fiscal Year must be "FY" followed by 2 digits.')
                    st.stop()

                if not out_name.lower().endswith(".xlsx"):
                    out_name += ".xlsx"

                file_bytes = uploaded.read()
                uploaded.seek(0)
                
                # Progress Bar setup
                prog_bar = st.progress(0, text="Initializing...")
                
                def update_ui_progress(pct, msg):
                    prog_bar.progress(pct, text=msg)
                    time.sleep(0.1) # Tiny sleep to make updates visible to user

                out_bytes, stats = process_excel_to_categories_only(
                    file_bytes, fy_clean, int(q), progress_callback=update_ui_progress
                )

                # Results UI
                st.markdown('<div class="success-box">Processing Complete</div>', unsafe_allow_html=True)
                
                st.markdown("<h3>Results Summary</h3>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""<div class="stat-container"><div class="stat-value">{stats['base_rows']:,}</div><div class="stat-label">Total Input</div></div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""<div class="stat-container"><div class="stat-value">{stats['eligible_rows']:,}</div><div class="stat-label">Eligible</div></div>""", unsafe_allow_html=True)
                with m3:
                    total_cat = sum(stats[f"{c}_rows"] for c in CATEGORIES)
                    st.markdown(f"""<div class="stat-container"><div class="stat-value" style="color:#34D399">{total_cat:,}</div><div class="stat-label">Selected</div></div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                
                st.download_button(
                    label="Download Results (.xlsx)",
                    data=out_bytes,
                    file_name=out_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error: {e}")