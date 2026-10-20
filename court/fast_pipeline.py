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
                 n_hist_col="n_hist", dtype=np.float32, max_chunk_cells=1.5e7, layout=None):
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
        # output rows sorted by prefix case id (upstream groupby order); blocks are written straight
        # to their sorted position, so the matrix is never copied
        order = np.argsort(meta["prefix_case_id"].to_numpy().astype(str), kind="stable")
        dest = np.empty(len(order), dtype=np.int64)
        dest[order] = np.arange(len(order))
        meta = meta.iloc[order].reset_index(drop=True)
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
            X[dest[out_pos:out_pos + len(r_local)]] = block
            out_pos += len(r_local)
            s = e
        return PrefixMatrix(X, names, kinds, sources, meta)

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
        # dynamic num: running mean/max/min/sum/std, bit-identical to pandas groupby (running_num_stats)
        if sp.dynamic_num:
            A = d[sp.dynamic_num].to_numpy(dtype=float)
            k = A.shape[1]
            stats = running_num_stats(A, seg, r)
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
        codes = pd.Index(self.vocab[c]).get_indexer(s.astype(str)).astype(np.int64)
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


_SPLIT = 134217729.0  # 2**27 + 1 (Dekker split)


def _fma(a, b, c):
    """Correctly rounded a*b + c with numpy float64 (error-free transformations).

    pandas' cython ``group_var`` is compiled with floating-point contraction, so its Welford update
    ``M2 += (val - mean) * (val - oldmean)`` is a fused multiply-add; emulating it keeps the running
    std bit-identical to ``groupby().std()`` (checked against ``math.fma`` on 2M random triples)."""
    p = a * b
    t = _SPLIT * a; ah = t - (t - a); al = a - ah
    t = _SPLIT * b; bh = t - (t - b); bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    s = p + c; bb = s - p; err = (p - (s - bb)) + (c - bb)
    return s + (err + e)


def running_num_stats(A, seg, r):
    """mean/max/min/sum/std of the numeric columns ``A`` over each prefix ending at local row r.

    Rows of a segment are contiguous and in event order. Accumulators follow pandas' cython groupby
    kernels step by step (NaN skipped): Kahan-compensated sum for sum and mean (``group_sum``,
    ``group_mean``), Welford with FMA for var (``group_var``, ddof=1), so the result at a prefix end
    equals ``groupby(prefix).agg(...)`` on the materialised prefix bit for bit. Empty -> 0 (fillna)."""
    n, k = A.shape
    nseg = int(seg.max()) + 1 if n else 0
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]]) if n else np.array([], int)
    pos = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
    order = np.argsort(pos, kind="stable")
    bnd = np.searchsorted(pos[order], np.arange(pos.max() + 2 if n else 1))
    r_pos = pos[r]
    r_order = np.argsort(r_pos, kind="stable")
    rbnd = np.searchsorted(r_pos[r_order], np.arange(pos.max() + 2 if n else 1))
    sumx = np.zeros((nseg, k)); comp = np.zeros((nseg, k)); nobs = np.zeros((nseg, k))
    mean = np.zeros((nseg, k)); M2 = np.zeros((nseg, k))
    mx = np.full((nseg, k), -np.inf); mn = np.full((nseg, k), np.inf)
    o_sum = np.zeros((len(r), k)); o_n = np.zeros((len(r), k)); o_m2 = np.zeros((len(r), k))
    o_mx = np.zeros((len(r), k)); o_mn = np.zeros((len(r), k))
    last_p = int(r_pos.max()) if len(r) else -1
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for p in range(last_p + 1):
            rows = order[bnd[p]:bnd[p + 1]]
            sg = seg[rows]; v = A[rows]; ok = ~np.isnan(v)
            nb = nobs[sg] + ok
            nobs[sg] = nb
            sx, cp = sumx[sg], comp[sg]
            y = v - cp; t = sx + y; c = (t - sx) - y
            c = np.where(c != c, 0.0, c)
            comp[sg] = np.where(ok, c, cp); sumx[sg] = np.where(ok, t, sx)
            om = mean[sg]; nm = om + (v - om) / nb
            m2 = _fma(v - nm, v - om, M2[sg])
            mean[sg] = np.where(ok, nm, om); M2[sg] = np.where(ok, m2, M2[sg])
            mx[sg] = np.where(ok, np.fmax(mx[sg], v), mx[sg]); mn[sg] = np.where(ok, np.fmin(mn[sg], v), mn[sg])
            sel = r_order[rbnd[p]:rbnd[p + 1]]
            if len(sel):
                ss = seg[r[sel]]
                o_sum[sel] = sumx[ss]; o_n[sel] = nobs[ss]; o_m2[sel] = M2[ss]
                o_mx[sel] = mx[ss]; o_mn[sel] = mn[ss]
        has = o_n > 0
        return {"mean": np.where(has, o_sum / o_n, 0.0),
                "sum": np.where(has, o_sum, 0.0),
                "max": np.where(has, o_mx, 0.0),
                "min": np.where(has, o_mn, 0.0),
                "std": np.where(o_n > 1, np.sqrt(o_m2 / (o_n - 1)), 0.0)}
