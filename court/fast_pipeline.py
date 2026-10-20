"""C9 fast path: prefix feature matrices without materialising the prefix log.

Upstream builds a prefix log (``DatasetManager.generate_prefix_data``: one copy of the
first ``n`` events of every case for every prefix length ``n``) and encodes it with
``FeatureUnion([StaticTransformer, AggregateTransformer])``. At court scale that is
3-10M rows x >100 columns. Here running aggregates are computed once per event,
segment-chunk by segment-chunk, and only the rows at prefix ends are kept.

The output must equal the upstream encoding of the same prefixes (tests/test_fast_pipeline
checks against a literal head(n)+groupby reference). Semantics mirrored from upstream:

* event order inside a segment: stable sort by (timestamp, activity), as in
  ``split_data_strict`` (``sorting_cols``) -> ``prepare_log``;
* prefixes (C6): length ``n_hist + j`` for j = 1, 1+gap, ... <= min(L, in-phase events);
  ``prefix_nr = j``; prefix case id ``seg`` for j == 1 else ``"%s_%s" % (seg, n_hist + j)``
  (upstream ``"%s_%s" % (x, nr_events)``); output rows sorted by that id, which is the
  row order upstream's ``groupby(case_id)`` produces (matters for XGBoost subsampling);
* static part (``StaticTransformer``): ``groupby.first()`` (first non-null value in the
  prefix) of static num cols, then ``get_dummies`` of static cat cols;
* aggregate part (``AggregateTransformer``, boolean=False): ``get_dummies`` of dynamic cat
  cols summed per prefix (cols in kind order, values sorted), then mean/max/min/sum/std of
  dynamic num cols with NaN skipped; std is pandas' (ddof=1), NaN for < 2 values;
* C8 (G columns only): last non-null value, named ``last_<col>``;
* ``fillna(0)`` everywhere; dummy columns exist only if the value occurs in the training
  (bucket) prefixes -> ``fit_columns`` keeps dummy columns that are non-zero in train.

The numeric-aggregate column layout of upstream depends on how A1 ports the old
``agg({'mean': np.mean, ...})`` call; ``NUM_AGG_LAYOUT`` switches between
``stat_major`` (``mean_<col>``, all cols per stat; old-pandas dict-renamer layout) and
``col_major`` (``<col>_mean``; ``agg([...])`` layout). Phase 2 pins it to the port.
"""
import numpy as np
import pandas as pd

STATS = ["mean", "max", "min", "sum", "std"]
NUM_AGG_LAYOUT = "stat_major"


def agg_feature_names(num_cols, layout=None):
    layout = layout or NUM_AGG_LAYOUT
    if layout == "stat_major":
        return [("%s_%s" % (s, c), c, s) for s in STATS for c in num_cols]
    return [("%s_%s" % (c, s), c, s) for c in num_cols for s in STATS]


def prepare_log(df, case_col="segment_id", ts_col="timestamp", act_col="Activity"):
    """Rows of each segment contiguous, inside a segment stable-sorted by (timestamp, activity).

    Segments appear in order of first appearance (irrelevant for the output, which is sorted
    by prefix case id)."""
    df = df.reset_index(drop=True)
    if not np.issubdtype(df[ts_col].dtype, np.datetime64):
        df[ts_col] = pd.to_datetime(df[ts_col])
    seg_order = pd.Series(pd.factorize(df[case_col])[0], index=df.index)
    order = np.lexsort((np.arange(len(df)), df[act_col].astype(str).to_numpy(),
                        df[ts_col].to_numpy(), seg_order.to_numpy()))
    return df.iloc[order].reset_index(drop=True)


def segment_bounds(df, case_col="segment_id"):
    codes = pd.factorize(df[case_col])[0]
    change = np.flatnonzero(np.diff(codes) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(df)]])
    if len(np.unique(codes[starts])) != len(starts):
        raise ValueError("segments are not contiguous; call prepare_log first")
    return starts, ends


def prefix_ends(df, max_len, gap=1, case_col="segment_id", n_hist_col="n_hist", min_len=1):
    """Row positions (into prepared ``df``) of prefix ends, and their j (= prefix_nr)."""
    starts, ends = segment_bounds(df, case_col)
    n = ends - starts
    h = df[n_hist_col].to_numpy()[starts].astype(np.int64)
    m = np.minimum(n - h, max_len)
    k = np.where(m >= min_len, (m - min_len) // gap + 1, 0)
    seg_rep = np.repeat(np.arange(len(starts)), k)
    within = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
    j = min_len + within * gap
    rows = starts[seg_rep] + h[seg_rep] + j - 1
    return rows.astype(np.int64), j.astype(np.int64), h[seg_rep]


class PrefixMatrix:
    """Feature matrix + per-row metadata (sorted by prefix case id)."""

    def __init__(self, X, names, kinds, sources, meta):
        self.X, self.names, self.kinds, self.sources, self.meta = X, list(names), list(kinds), list(sources), meta

    def subset_rows(self, mask):
        idx = np.asarray(mask)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return PrefixMatrix(self.X[idx], self.names, self.kinds, self.sources,
                            self.meta.iloc[idx].reset_index(drop=True))

    def subset_cols(self, cols_mask):
        idx = np.flatnonzero(cols_mask)
        return PrefixMatrix(self.X[:, idx], [self.names[i] for i in idx], [self.kinds[i] for i in idx],
                            [self.sources[i] for i in idx], self.meta)

    def to_frame(self):
        return pd.DataFrame(self.X, columns=self.names, index=self.meta["prefix_case_id"].to_numpy())


def fit_columns(pm):
    """Upstream column set for training matrix ``pm``: dummy columns present iff non-zero."""
    nz = (pm.X != 0).any(axis=0) if len(pm.X) else np.zeros(pm.X.shape[1], bool)
    return np.array([k != "dummy" for k in pm.kinds]) | nz


def apply_columns(pm, train_names):
    """Align ``pm`` to the train column list (missing -> 0, extra dropped), like upstream transform."""
    pos = {n: i for i, n in enumerate(pm.names)}
    X = np.zeros((len(pm.X), len(train_names)), dtype=pm.X.dtype)
    kinds, sources = [], []
    for out_i, n in enumerate(train_names):
        if n in pos:
            X[:, out_i] = pm.X[:, pos[n]]
            kinds.append(pm.kinds[pos[n]]); sources.append(pm.sources[pos[n]])
        else:
            kinds.append("dummy"); sources.append(None)
    return PrefixMatrix(X, train_names, kinds, sources, pm.meta)


class FastEncoder:
    def __init__(self, spec, case_col="segment_id", act_col="Activity", ts_col="timestamp",
                 label_col="label", reg_label_col="remaining_in_phase", pos_label="deviant",
                 n_hist_col="n_hist", dtype=np.float32, max_chunk_cells=4e7, layout=None):
        self.spec = spec
        self.case_col, self.act_col, self.ts_col = case_col, act_col, ts_col
        self.label_col, self.reg_label_col, self.pos_label = label_col, reg_label_col, pos_label
        self.n_hist_col = n_hist_col
        self.dtype = dtype
        self.max_chunk_cells = max_chunk_cells
        self.layout = layout or NUM_AGG_LAYOUT
        self.vocab = None

    # ---- vocabulary (all values seen in the given log; pruned later by fit_columns) ----
    def build_vocab(self, df):
        self.vocab = {}
        for c in self.spec.static_cat + self.spec.dynamic_cat:
            vals = df[c].dropna().astype(str).unique()
            self.vocab[c] = sorted(vals)
        return self

    def _layout(self):
        sp = self.spec
        names, kinds, sources = [], [], []
        for c in sp.static_num:
            names.append(c); kinds.append("static_num"); sources.append(c)
        for c in sp.static_cat:
            for v in self.vocab[c]:
                names.append("%s_%s" % (c, v)); kinds.append("dummy"); sources.append(c)
        for c in sp.dynamic_cat:
            for v in self.vocab[c]:
                names.append("%s_%s" % (c, v)); kinds.append("dummy"); sources.append(c)
        for n, c, s in agg_feature_names(sp.dynamic_num, self.layout):
            names.append(n); kinds.append("agg_" + s); sources.append(c)
        for c in sp.last_cols:
            names.append("last_%s" % c); kinds.append("last"); sources.append(c)
        return names, kinds, sources

    def encode(self, df, max_len, gap=1, prepared=False):
        """Encode all prefixes of ``df`` (a SCHEMA-shaped event log)."""
        if not prepared:
            df = prepare_log(df, self.case_col, self.ts_col, self.act_col)
        if self.vocab is None:
            self.build_vocab(df)
        names, kinds, sources = self._layout()
        rows, j, h = prefix_ends(df, max_len, gap, self.case_col, self.n_hist_col)
        X = np.zeros((len(rows), len(names)), dtype=self.dtype)
        starts, ends = segment_bounds(df, self.case_col)
        n_cells_per_row = max(1, len(names))
        chunk_rows = int(max(1000, self.max_chunk_cells // n_cells_per_row))
        # chunk = consecutive segments with <= chunk_rows events (at least one segment)
        out_pos = 0
        s = 0
        while s < len(starts):
            e = s
            while e < len(starts) and (ends[e] - starts[s] <= chunk_rows or e == s):
                e += 1
            lo, hi = starts[s], ends[e - 1]
            sel = (rows >= lo) & (rows < hi)
            r_local = rows[sel] - lo
            block = self._encode_chunk(df.iloc[lo:hi], r_local)
            X[out_pos:out_pos + len(r_local)] = block
            out_pos += len(r_local)
            s = e
        seg = df[self.case_col].to_numpy()[rows]
        plen = h + j
        meta = pd.DataFrame({
            "segment_id": seg,
            "prefix_nr": j,
            "prefix_len": plen,
            "prefix_case_id": [s_ if jj == 1 else "%s_%s" % (s_, pl) for s_, jj, pl in zip(seg, j, plen)],
            "last_ts": df[self.ts_col].to_numpy()[rows],
        })
        lab = df.groupby(self.case_col, sort=False)[self.label_col].transform("first").to_numpy()[rows] \
            if self.label_col in df else np.full(len(rows), None)
        meta["y_clf"] = (lab == self.pos_label).astype(int)
        if self.reg_label_col in df:
            ff = df.groupby(self.case_col, sort=False)[self.reg_label_col].ffill().to_numpy()
            meta["y_reg"] = ff[rows].astype(float)
        order = np.argsort(meta["prefix_case_id"].to_numpy().astype(str), kind="stable")
        meta = meta.iloc[order].reset_index(drop=True)
        return PrefixMatrix(X[order], names, kinds, sources, meta)

    def _encode_chunk(self, d, r):
        sp = self.spec
        seg = pd.factorize(d[self.case_col])[0]
        seg_s = pd.Series(seg)
        parts = []
        # static num: first non-null value within the prefix
        if sp.static_num:
            parts.append(self._first_nonnull(d[sp.static_num].to_numpy(dtype=float), seg, r))
        for c in sp.static_cat:
            codes = self._codes(d[c], c)
            first = self._first_nonnull(np.where(codes >= 0, codes, np.nan).astype(float)[:, None], seg, r)[:, 0]
            oh = np.zeros((len(r), len(self.vocab[c])))
            ok = ~np.isnan(first)
            oh[np.flatnonzero(ok), first[ok].astype(int)] = 1
            parts.append(oh)
        # dynamic cat: cumulative one-hot counts (exact integer cumsum, reset per segment)
        seg_start = np.zeros(len(d), dtype=np.int64)
        bounds = np.flatnonzero(np.diff(seg) != 0) + 1
        seg_start[bounds] = bounds
        seg_start = np.maximum.accumulate(seg_start)
        for c in sp.dynamic_cat:
            codes = self._codes(d[c], c)
            V = len(self.vocab[c])
            oh = np.zeros((len(d), V), dtype=np.int32)
            ok = codes >= 0
            oh[np.flatnonzero(ok), codes[ok]] = 1
            cs = np.cumsum(oh, axis=0)
            before = np.where(seg_start[r] > 0, seg_start[r] - 1, -1)
            cnt = cs[r] - np.where(before[:, None] >= 0, cs[np.maximum(before, 0)], 0)
            parts.append(cnt)
            del oh, cs
        # dynamic num: running mean/max/min/sum/std (NaN skipped), shifted by first value for stability
        if sp.dynamic_num:
            A = d[sp.dynamic_num].to_numpy(dtype=float)
            valid = ~np.isnan(A)
            g = pd.DataFrame(A).groupby(seg)
            x0 = g.transform("first").to_numpy()
            Z = np.where(valid, A - np.nan_to_num(x0), 0.0)
            stack = pd.DataFrame(np.hstack([valid.astype(float), Z, Z * Z])).groupby(seg).cumsum().to_numpy()[r]
            k = A.shape[1]
            cnt, s1, s2 = stack[:, :k], stack[:, k:2 * k], stack[:, 2 * k:]
            x0r = np.nan_to_num(x0[r])
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(cnt > 0, x0r + s1 / cnt, 0.0)
                tot = np.where(cnt > 0, x0r * cnt + s1, 0.0)
                var = (s2 - s1 * s1 / cnt) / (cnt - 1)
                std = np.where(cnt > 1, np.sqrt(np.maximum(var, 0.0)), 0.0)
            mx = pd.DataFrame(np.where(valid, A, -np.inf)).groupby(seg).cummax().to_numpy()[r]
            mn = pd.DataFrame(np.where(valid, A, np.inf)).groupby(seg).cummin().to_numpy()[r]
            mx = np.where(np.isfinite(mx), mx, 0.0)
            mn = np.where(np.isfinite(mn), mn, 0.0)
            stats = {"mean": mean, "max": mx, "min": mn, "sum": tot, "std": std}
            for _, c, s in agg_feature_names(list(range(k)), self.layout):
                parts.append(stats[s][:, c:c + 1])
        if sp.last_cols:
            L = d[sp.last_cols].to_numpy(dtype=float)
            ff = pd.DataFrame(L).groupby(seg).ffill().to_numpy()[r]
            parts.append(np.nan_to_num(ff, nan=0.0))
        if not parts:
            return np.zeros((len(r), 0))
        out = np.hstack([p.reshape(len(r), -1).astype(float) for p in parts])
        return np.nan_to_num(out, nan=0.0)

    def _codes(self, s, c):
        codes = pd.Categorical(s.astype(str), categories=self.vocab[c]).codes.astype(np.int64)
        codes[s.isna().to_numpy()] = -1
        return codes

    @staticmethod
    def _first_nonnull(A, seg, r):
        g = pd.DataFrame(A).groupby(seg)
        # position of the first non-null value of each column within the segment
        pos = np.arange(len(A))[:, None].repeat(A.shape[1], 1).astype(float)
        pos[np.isnan(A)] = np.nan
        first_pos = pd.DataFrame(pos).groupby(seg).transform("min").to_numpy()
        first_val = g.transform("first").to_numpy()
        fp = first_pos[r]
        return np.where(~np.isnan(fp) & (fp <= r[:, None]), first_val[r], np.nan)
