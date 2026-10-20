"""Step 1: ingest and check the raw inputs.

* MD5 + line count of the full event log must equal PLAN.md §0.3 (else stop).
  The check result is cached in data/interim/raw_check.json, keyed by size + mtime.
* The event log is read with only the event columns (case-level columns repeated on every
  row, which include outcome columns, are never loaded) and filtered to case-table cases.
* Output: data/interim/events.parquet with a global `row` index (original file order).
"""
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

DATA = os.environ.get("COURT_DATA_DIR", "data")
RAW = os.path.join(DATA, "raw")
INTERIM = os.path.join(DATA, "interim")

EVENT_LOG = "df_event_log.csv"
CASE_TABLE = "case_level_df.csv"
PHASE_LOG = "phases_event_log_cln.csv"

EXPECTED = {
    EVENT_LOG: {"md5": "0f18ebeb01d366898a29896d69e45091", "lines": 4811484},
    CASE_TABLE: {"md5": "97951dac7164a31a38e617503216c57a", "lines": None},  # no trailing newline: wc -l = 155,525,
    PHASE_LOG: {"md5": "fced1e192da99a5dec2adff4e1542e89", "lines": None},
}
EVENT_BASE_COLS = ["ucid", "date_filed", "Activity", "event_judge"]


def _md5(path):
    try:
        out = subprocess.run(["md5", "-q", path], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        return h.hexdigest()


def _lines(path):
    out = subprocess.run(["wc", "-l", path], capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0])


def check_inputs(raw_dir=RAW, interim_dir=INTERIM, log=print):
    """MD5 + line counts of all three inputs (cached). Raises if the event log differs."""
    os.makedirs(interim_dir, exist_ok=True)
    cache_path = os.path.join(interim_dir, "raw_check.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    res = {}
    for name, exp in EXPECTED.items():
        p = os.path.join(raw_dir, name)
        st = os.stat(p)  # follows the symlink, read-only
        key = f"{st.st_size}:{int(st.st_mtime)}"
        if cache.get(name, {}).get("key") == key:
            r = cache[name]
        else:
            r = {"key": key, "size": st.st_size, "md5": _md5(p), "lines": _lines(p)}
        r["md5_ok"] = r["md5"] == exp["md5"]
        r["lines_ok"] = exp["lines"] is None or r["lines"] == exp["lines"]
        res[name] = r
        log(f"[ingest] {name}: md5={r['md5']} ok={r['md5_ok']} lines={r['lines']} ok={r['lines_ok']}")
    json.dump(res, open(cache_path, "w"), indent=1)
    ev = res[EVENT_LOG]
    if not (ev["md5_ok"] and ev["lines_ok"]):
        raise SystemExit(f"STOP: event log MD5/line count differ from PLAN.md §0.3: {ev}")
    if not (res[CASE_TABLE]["md5_ok"] and res[CASE_TABLE]["lines_ok"]):
        raise SystemExit(f"STOP: case table MD5/line count differ: {res[CASE_TABLE]}")
    return res


def read_case_table(path):
    c = pd.read_csv(path, low_memory=False)
    c = c.loc[:, ~c.columns.str.startswith("Unnamed")]
    for col in c.columns:  # TRUE/FALSE strings -> bool
        if c[col].dtype == object or str(c[col].dtype).startswith("str"):
            vals = set(c[col].dropna().unique().tolist())
            if vals and vals <= {"TRUE", "FALSE", "True", "False"}:
                c[col] = c[col].map({"TRUE": True, "FALSE": False, "True": True, "False": False})
    for col in [x for x in c.columns if x.endswith("_date")]:
        c[col] = pd.to_datetime(c[col])
    return c


def event_columns(path):
    with open(path) as f:
        header = f.readline().rstrip("\n").split(",")
    attr = [h for h in header if h.startswith("attribute_")]
    return header, attr


def read_events(path, keep_ucids=None, log=print):
    """Read only event columns with pyarrow; add `row` = original 0-based row order."""
    header, attr = event_columns(path)
    flags = [a for a in attr if a != "attribute_duplicates"]
    types = {"ucid": pa.string(), "date_filed": pa.string(), "Activity": pa.string(),
             "event_judge": pa.string(), "attribute_duplicates": pa.int32()}
    types.update({a: pa.bool_() for a in flags})
    cols = EVENT_BASE_COLS + attr
    tbl = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=1 << 26),
        convert_options=pacsv.ConvertOptions(
            include_columns=cols, column_types=types,
            true_values=["True", "TRUE", "true"], false_values=["False", "FALSE", "false"],
            strings_can_be_null=True),
    )
    n_all = tbl.num_rows
    tbl = tbl.append_column("row", pa.array(np.arange(n_all, dtype=np.int64)))
    if keep_ucids is not None:
        import pyarrow.compute as pc
        mask = pc.is_in(tbl["ucid"], value_set=pa.array(list(keep_ucids), type=pa.string()))
        tbl = tbl.filter(mask)
    log(f"[ingest] event log rows: {n_all:,}; kept (case-table cases): {tbl.num_rows:,}")
    return tbl, n_all, flags


def run(raw_dir=RAW, interim_dir=INTERIM, log=print, report=None):
    report = report if report is not None else {}
    chk = check_inputs(raw_dir, interim_dir, log)
    report["inputs"] = chk
    cases = read_case_table(os.path.join(raw_dir, CASE_TABLE))
    assert cases.ucid.is_unique
    report["n_cases_table"] = len(cases)
    out = os.path.join(interim_dir, "events.parquet")
    if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(os.path.join(raw_dir, EVENT_LOG)):
        log(f"[ingest] reusing {out}")
        ev = pd.read_parquet(out)
        meta = json.load(open(os.path.join(interim_dir, "events_meta.json")))
        n_all = meta["n_all"]
    else:
        tbl, n_all, _ = read_events(os.path.join(raw_dir, EVENT_LOG), set(cases.ucid), log)
        pq.write_table(tbl, out)
        json.dump({"n_all": n_all}, open(os.path.join(interim_dir, "events_meta.json"), "w"))
        ev = tbl.to_pandas()
        del tbl
    report["n_events_full_log"] = n_all
    report["n_events_case_table_cases"] = len(ev)
    report["n_cases_in_log"] = int(ev.ucid.nunique())

    # phase log: only check that its case ids are case-table ids
    ph = pd.read_csv(os.path.join(raw_dir, PHASE_LOG), usecols=["ucid"])
    report["phase_log_rows"] = len(ph)
    report["phase_log_cases"] = int(ph.ucid.nunique())
    report["phase_log_cases_not_in_table"] = int((~pd.Index(ph.ucid.unique()).isin(cases.ucid)).sum())
    report["table_cases_not_in_log"] = int((~cases.ucid.isin(ev.ucid.unique())).sum())

    # sum(attribute_duplicates) == total_activity_count per case
    dup = ev.groupby("ucid", sort=False)["attribute_duplicates"].sum()
    nrows = ev.groupby("ucid", sort=False).size()
    tac = cases.set_index("ucid")["total_activity_count"]
    dup = dup.reindex(tac.index).fillna(0)
    bad = int((dup != tac).sum())
    report["dup_sum_mismatch_cases"] = bad
    assert bad == 0, f"{bad} cases with sum(attribute_duplicates) != total_activity_count"
    diff = (nrows.reindex(tac.index).fillna(0) - tac)
    report["rows_minus_total_activity"] = {
        "share_lt0": float((diff < 0).mean()), "share_eq0": float((diff == 0).mean()),
        "mean": float(diff.mean()), "median": float(diff.median()),
        "q01": float(diff.quantile(0.01)), "q99": float(diff.quantile(0.99)),
        "min": float(diff.min()), "max": float(diff.max())}
    log(f"[ingest] dup-sum check ok; rows - total_activity_count: {report['rows_minus_total_activity']}")
    return cases, ev, report
