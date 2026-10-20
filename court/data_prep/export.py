"""Steps 7-8: assemble phase tables, rare categories, D8 exclusion assert, export per court/SCHEMA.md."""
import fnmatch
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .segments import DAY0

M_CAT = ["case_type", "city", "nature_suit", "District_Judge", "Magistrate_Judge",
         "Defendants_highest_offense_opening"]
M_NUM = ["is_mdl", "Party_Amicus", "Party_Counter_Claimant", "Party_Counter_Defendant",
         "Party_Court_Monitor", "Party_Intervenor", "Party_Material_Witness",
         "Party_Third_Party_Defendant", "Party_Third_Party_Plaintiff", "Party_Trustee",
         "Other_courts", "related_case_count", "plaintiffs_share_ind", "plaintiffs_share_pro_se",
         "Defendants_share_ind", "Defendants_share_pro_se"]
# D8 suspect (possibly end-of-case values): sensitivity view S only
S_NUM = ["plaintiffs_count", "plaintiffs_counsels_count", "Defendants_count",
         "Defendants_counsels_count", "plaintiffs_share_pro_hac_vice", "Defendants_share_pro_hac_vice"]
SUSPECT_PATTERNS = ["*counsels_count", "plaintiffs_count", "Defendants_count", "*_share_pro_hac_vice"]
# D8 outcome columns: never features in any run (patterns on raw names, case-insensitive)
OUTCOME_PATTERNS = ["resolution_type", "disposition_group", "is_case_*", "activity_count_*", "total_*",
                    "*terminated*", "phase*_exit_type", "phase*_los", "phase*_activities",
                    "phase*_completed", "phase*_entered", "phase*_entry_date", "phase*_exit_date",
                    "case_status", "terminating_date", "*pending_counts", "unique_*_judge_count"]
EXCLUSION_REASONS = {
    "resolution_type": "D8 outcome column", "disposition_group": "D8 outcome column",
    "is_case_*": "D8 outcome column", "activity_count_*": "D8 outcome column (end-of-case counts)",
    "total_*": "D8 outcome column (end-of-case totals)",
    "Defendants_highest_offense_terminated": "D8 outcome column (terminated)",
    "Defendants_terminated_counts": "D8 outcome column (terminated counts)",
    "Defendants_pending_counts": "end-of-case value (pending at termination); conservative exclusion",
    "unique_district_judge_count": "end-of-case value (judges over the whole case); conservative exclusion",
    "unique_magistrate_judge_count": "end-of-case value; conservative exclusion",
    "phase*_exit_type / phase*_los / phase*_activities / phase*_completed": "D8 outcome columns",
    "phase*_entered / phase*_entry_date / phase*_exit_date": "used only to build segments/labels (never features)",
    "case_status": "D8 outcome column", "terminating_date": "D8 outcome column",
    "filing_date": "exported only as filing_year (never feature)",
    "event-log case-level columns": "dropped at ingest (usecols); include outcome columns",
    **{c: "D8 suspect (possibly end-of-case): view S only (sensitivity run)" for c in S_NUM},
}
FIXED = ["segment_id", "ucid", "Activity", "judge", "timestamp", "label", "remaining_in_phase",
         "in_phase", "n_hist", "entry_date", "exit_date", "filing_year",
         "timesincelastevent", "timesincecasestart", "event_nr", "month", "weekday", "open_cases"]
TIME_COLS = ["timesincelastevent", "timesincecasestart", "event_nr", "month", "weekday"]
NEVER = ["ucid", "in_phase", "n_hist", "entry_date", "exit_date", "filing_year", "remaining_in_phase", "label"]
RARE_MIN = 10


def _raw_name(col):
    for p in ("M_", "S_", "PP_", "P_", "G_"):
        if col.startswith(p):
            return col[len(p):]
    return col


def is_outcome(col):
    raw = _raw_name(col).lower()
    return any(fnmatch.fnmatch(raw, p.lower()) for p in OUTCOME_PATTERNS)


def is_suspect(col):
    raw = _raw_name(col)
    return any(fnmatch.fnmatch(raw, p) for p in SUSPECT_PATTERNS)


def assert_no_leaky_features(colspec):
    """D8: no outcome column in any view/feature list; suspect columns only in view S."""
    feats = set()
    for k in ("static_cat", "static_num", "dynamic_cat", "dynamic_num"):
        feats |= set(colspec[k])
    for v, cols in colspec["views"].items():
        feats |= set(cols)
        for c in cols:
            assert not is_outcome(c), f"D8 violation: outcome column {c} in view {v}"
            if v != "S":
                assert not is_suspect(c), f"D8 violation: suspect column {c} in view {v}"
    bad = [c for c in feats if is_outcome(c)]
    assert not bad, f"D8 violation: outcome columns among features: {bad}"
    bad = [c for c in feats if c in NEVER]
    assert not bad, f"never-feature columns among features: {bad}"


def rare_to_other(df, static_cat, dynamic_cat, report):
    """Levels with < RARE_MIN occurrences -> "other" (static: # segments, dynamic: # event rows)."""
    rep = {}
    seg_first = df.drop_duplicates("segment_id")
    for c in static_cat:
        vc = seg_first[c].value_counts()
        rare = vc.index[vc < RARE_MIN]
        rep[c] = int(len(rare))
        if len(rare):
            df[c] = df[c].where(~df[c].isin(rare), "other")
    for c in dynamic_cat:
        if c == "Activity":
            continue
        vc = df[c].value_counts()
        rare = vc.index[vc < RARE_MIN]
        rep[c] = int(len(rare))
        if len(rare):
            df[c] = df[c].where(~df[c].isin(rare), "other")
    report.update(rep)
    return df


def day_str(d):
    return (DAY0 + pd.to_timedelta(np.asarray(d, dtype=np.int64), unit="D")).strftime("%Y-%m-%d")


def assemble_phase(k, seg, ev_idx, seg_idx, ev, EF, G, win, attr_cols, pp_dyn):
    """One row per trace event. seg: kept segments of phase k; EF: event features aligned to ev."""
    wi = win.set_index("ucid")
    meta = wi.loc[seg.ucid]
    d = ev.d.values[ev_idx]
    cols = {}
    cols["segment_id"] = seg.segment_id.values[seg_idx]
    cols["ucid"] = seg.ucid.values[seg_idx]
    cols["Activity"] = ev.Activity.values[ev_idx]
    cols["judge"] = EF.judge.values[ev_idx]
    cols["timestamp"] = day_str(d)
    cols["label"] = np.where(seg.label01.values[seg_idx] == 1, "deviant", "regular")
    ex = seg.exit.values.astype(np.int64)[seg_idx]
    en = seg.entry.values.astype(np.int64)[seg_idx]
    cols["remaining_in_phase"] = (ex - d).astype(np.int32)
    cols["in_phase"] = (d >= en).astype(np.int8)
    cols["n_hist"] = seg.n_hist.values[seg_idx].astype(np.int32)
    ed, xd = day_str(seg.entry.values), day_str(seg.exit.values)
    cols["entry_date"] = np.asarray(ed)[seg_idx]
    cols["exit_date"] = np.asarray(xd)[seg_idx]
    cols["filing_year"] = seg.filing_year.values[seg_idx].astype(np.int16)
    for c in TIME_COLS:
        cols[c] = EF[c].values[ev_idx]
    cols["open_cases"] = G["open_cases"].values
    for c in M_CAT:
        cols["M_" + c] = meta[c].fillna("missing").astype(str).values[seg_idx]
    for c in M_NUM:
        cols["M_" + c] = meta[c].astype("float64").fillna(0).values[seg_idx]
    for c in attr_cols:
        cols[c] = EF[c].values[ev_idx]
    cols["P_judge_is_magistrate"] = EF["P_judge_is_magistrate"].values[ev_idx]
    for c in pp_dyn:
        cols[c] = EF[c].values[ev_idx]
    for c in ("PP_prev_phase_days_1", "PP_prev_phase_days_2"):
        cols[c] = seg[c].values[seg_idx].astype(np.int32)
    for c in G.columns:
        if c != "open_cases":
            cols[c] = G[c].values
    for c in S_NUM:
        cols["S_" + c] = meta[c].astype("float64").fillna(0).values[seg_idx]
    df = pd.DataFrame(cols)
    assert (df.remaining_in_phase >= 1).all(), "D7 violated: event on/after exit"
    return df


def column_spec(df, attr_cols, pp_dyn):
    g_cols = [c for c in df.columns if c.startswith("G_")]
    m_cat = ["M_" + c for c in M_CAT]
    m_num = ["M_" + c for c in M_NUM]
    s_num = ["S_" + c for c in S_NUM]
    pp_static = ["PP_prev_phase_days_1", "PP_prev_phase_days_2"]
    p_dyn = TIME_COLS + attr_cols + ["P_judge_is_magistrate"]
    spec = {
        "case_id_col": "segment_id", "activity_col": "Activity", "resource_col": "judge",
        "timestamp_col": "timestamp", "label_col": "label", "regression_label_col": "remaining_in_phase",
        "static_cat": m_cat, "static_num": m_num + pp_static + s_num,
        "dynamic_cat": ["Activity", "judge"],
        "dynamic_num": p_dyn + pp_dyn + ["open_cases"] + g_cols,
        "views": {"M": m_cat + m_num, "P": ["Activity", "judge"] + p_dyn, "PP": pp_dyn + pp_static,
                  "B0G": ["open_cases"], "G": g_cols, "S": s_num},
        "never_features": NEVER,
        "excluded": EXCLUSION_REASONS,
        "notes": {
            "P_attr_duplicates": "count of collapsed duplicate entries of the event (>=1), not a 0/1 flag",
            "G_definition": "open at t: filing <= t < termination (state at end of day t), this case excluded",
            "rare_categories": f"levels with < {RARE_MIN} occurrences -> 'other' (static: segments, dynamic: rows), per phase file",
        },
    }
    assert_no_leaky_features(spec)
    exported = set(df.columns)
    for k in ("static_cat", "static_num", "dynamic_cat", "dynamic_num"):
        missing = set(spec[k]) - exported
        assert not missing, missing
    return spec


def write_phase(df, path_base):
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    pacsv.write_csv(tbl, path_base + ".csv", write_options=pacsv.WriteOptions(delimiter=";", quoting_style="needed"))
    pq.write_table(tbl, path_base + ".parquet")
    return os.path.getsize(path_base + ".csv"), os.path.getsize(path_base + ".parquet")


def write_columns_json(spec, path):
    with open(path, "w") as f:
        json.dump(spec, f, indent=1)
