"""A3 false-winner statistics on block-shift NULL panels (plan §7.1), computed before the tag.

Every household-level quantity here is a statistic of shifted (placebo) outcomes from N's N1/N2
panels; no actual household mean is ever formed. Per draw and household (n_h >= min_n purchases in
the household's ACTUAL count over 2015-2025, the plan's full-sample set): n, mean, iid t and p,
anchor-date-clustered t and p, and the normal-normal EB posterior mean (`estimators.eb`). Per draw:
largest mean, largest t, top-decile mean, count of nominal p < 0.05, winner identities. Across draws:
winner-identity turnover (consecutive change share; entropy of the argmax distribution).
Units: basis points (stored panels are decimal returns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from trading_on_disclosure_v4.estimators import eb as eb_mod
from trading_on_disclosure_v4.null import block_shift as bs

BP = 1e4
ALPHA = 0.05
MIN_N_GRID = (20, 10, 50)
READ_COLUMNS = ["draw", "household_id", "y_h20", "shifted_anchor_date"]


def household_set(clock: str, min_n: int, drop_duplicates: bool = True) -> tuple[pd.Index, dict]:
    """Households with >= min_n ACTUAL purchases over A u B in the null generator's input (counts only)."""
    events, report = bs.load_events_for_null(clock, drop_duplicates=drop_duplicates)
    counts = events.groupby("household_id").size()
    keep = counts.index[counts >= min_n]
    info = {"clock": clock, "min_n": int(min_n), "n_households_in": int(counts.size),
            "n_households_set": int(len(keep)), "n_events_in": int(report["n_events_out"]),
            "n_events_set": int(counts.loc[keep].sum())}
    return keep, info


def _clustered_se(df: pd.DataFrame, key: list[str], cluster: str = "shifted_anchor_date") -> pd.DataFrame:
    """Cluster-robust pieces for the household mean (intercept-only regression, G/(G-1) correction)."""
    resid = df["y"] - df.groupby(key, sort=False)["y"].transform("mean")
    cl = resid.groupby([df[k] for k in key] + [df[cluster]], sort=False).sum()
    cl = cl.rename("s").reset_index()
    agg = cl.groupby(key, sort=True).agg(ss=("s", lambda s: float(np.sum(s.to_numpy() ** 2))),
                                         n_clusters=("s", "size"))
    return agg


def _attach_cluster_stats(g: pd.DataFrame, df: pd.DataFrame, key: list[str], cluster: str, suffix: str) -> None:
    cl = _clustered_se(df, key, cluster).reindex(g.index)
    G = cl["n_clusters"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        se_cl = np.sqrt(cl["ss"].to_numpy() * G / (G - 1)) / g["n"].to_numpy(float)
        g[f"se_{suffix}"] = np.where(G > 1, se_cl, np.nan)
        g[f"t_{suffix}"] = np.where(g[f"se_{suffix}"] > 0, g["mean"] / g[f"se_{suffix}"], np.nan)
    g[f"p_{suffix}"] = 2 * stats.t.sf(np.abs(g[f"t_{suffix}"]), df=np.maximum(G - 1, 1))
    g[f"n_clusters_{suffix}"] = G.astype(int)


def draw_household_stats(batch: pd.DataFrame, households: pd.Index, y: str = "y_h20",
                         calendar: pd.DatetimeIndex | None = None, block_length: int = 20) -> pd.DataFrame:
    """Per (draw, household) n, mean, sd, se, t, p (iid, n-1 dof); clustered t, p (G-1 dof) by shifted
    anchor date (`_cl`) and, when a calendar is given, by `block_length`-session calendar bin of the
    shifted anchor (`_blk`, overlapping outcome windows share a bin); bp."""
    df = batch.loc[batch["household_id"].isin(households), ["draw", "household_id", y, "shifted_anchor_date"]]
    df = df.rename(columns={y: "y"})
    df = df[np.isfinite(df["y"])].copy()
    df["y"] = df["y"] * BP
    key = ["draw", "household_id"]
    g = df.groupby(key, sort=True)["y"].agg(n="size", mean="mean", sd=lambda s: s.std(ddof=1))
    g = g[g["n"] >= 2].copy()
    g["se"] = g["sd"] / np.sqrt(g["n"])
    with np.errstate(divide="ignore", invalid="ignore"):
        g["t"] = np.where(g["se"] > 0, g["mean"] / g["se"], np.nan)
    g["p"] = 2 * stats.t.sf(np.abs(g["t"]), df=g["n"] - 1)
    _attach_cluster_stats(g, df, key, "shifted_anchor_date", "cl")
    g["n_clusters"] = g["n_clusters_cl"]
    if calendar is not None:
        df["blk"] = calendar.get_indexer(pd.DatetimeIndex(df["shifted_anchor_date"])) // block_length
        _attach_cluster_stats(g, df, key, "blk", "blk")
    return g.reset_index()


def _top_decile_mean(values: np.ndarray) -> tuple[float, int]:
    k = max(1, int(np.ceil(0.1 * len(values))))
    return float(np.sort(values)[-k:].mean()), k


def eb_shrink(hs: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """EB posterior means for one draw's household table (marginal ML; prior mean linear in log(1+n))."""
    fit = eb_mod.fit_eb(hs["mean"].to_numpy(), hs["se"].to_numpy(), hs["n"].to_numpy())
    return fit.posterior_mean, {"eb_tau": fit.tau, "eb_at_zero_boundary": bool(fit.at_zero_boundary),
                                "eb_prior_slope": fit.b}


def winner_row(hs: pd.DataFrame, alpha: float = ALPHA, with_eb: bool = True) -> dict:
    """One draw's false-winner statistics from its household table (rows of `draw_household_stats`)."""
    hs = hs.reset_index(drop=True)
    means, ts = hs["mean"].to_numpy(), hs["t"].to_numpy()
    top_mean, k = _top_decile_mean(means)
    i_mean, i_t = int(np.argmax(means)), int(np.nanargmax(ts))
    row = {"draw": int(hs["draw"].iloc[0]), "n_households": int(len(hs)), "n_events": int(hs["n"].sum()),
           "max_mean": float(means.max()), "max_t": float(np.nanmax(ts)), "max_t_cl": float(np.nanmax(hs["t_cl"])),
           "min_mean": float(means.min()), "top_decile_mean": top_mean, "top_decile_size": k,
           "n_p05_iid": int((hs["p"] < alpha).sum()), "n_p05_cl": int((hs["p_cl"] < alpha).sum()),
           "share_p05_iid": float((hs["p"] < alpha).mean()), "share_p05_cl": float((hs["p_cl"] < alpha).mean()),
           "median_n_clusters": float(hs["n_clusters"].median()),
           "winner_by_mean": str(hs["household_id"].iloc[i_mean]), "winner_by_t": str(hs["household_id"].iloc[i_t])}
    if "p_blk" in hs:
        row.update({"n_p05_blk": int((hs["p_blk"] < alpha).sum()), "share_p05_blk": float((hs["p_blk"] < alpha).mean()),
                    "median_n_clusters_blk": float(hs["n_clusters_blk"].median())})
    if with_eb:
        post, info = eb_shrink(hs)
        top_post, _ = _top_decile_mean(post)
        row.update(info)
        row.update({"max_eb_mean": float(post.max()), "top_decile_eb_mean": top_post,
                    "winner_by_eb": str(hs["household_id"].iloc[int(np.argmax(post))])})
    return row


def batch_winner_rows(batch: pd.DataFrame, households: pd.Index, y: str = "y_h20", with_eb: bool = True,
                      calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Per-draw rows for every draw in one stored batch."""
    hs = draw_household_stats(batch, households, y, calendar)
    return pd.DataFrame([winner_row(g, with_eb=with_eb) for _, g in hs.groupby("draw", sort=True)])


def turnover(identities) -> dict:
    """Consecutive-draw change share; entropy (bits, and normalized by log2 of the household count)."""
    ids = np.asarray([str(i) for i in identities])
    if len(ids) == 0:
        raise ValueError("no identities")
    _, counts = np.unique(ids, return_counts=True)
    p = counts / counts.sum()
    ent = float(-(p * np.log2(p)).sum())
    return {"n_draws": int(len(ids)), "consecutive_change_share": float(np.mean(ids[1:] != ids[:-1])) if len(ids) > 1 else 0.0,
            "n_distinct_winners": int(len(counts)), "top_winner_share": float(p.max()), "entropy_bits": ent}


QUANTILES = (0.50, 0.90, 0.95, 0.99)
DIST_STATS = {"max_mean": "bp", "max_t": "t", "max_t_cl": "t", "top_decile_mean": "bp", "min_mean": "bp",
              "n_p05_iid": "count", "n_p05_cl": "count", "share_p05_iid": "share", "share_p05_cl": "share",
              "n_p05_blk": "count", "share_p05_blk": "share", "median_n_clusters": "count", "median_n_clusters_blk": "count",
              "max_eb_mean": "bp", "top_decile_eb_mean": "bp", "eb_tau": "bp", "eb_at_zero_boundary": "share",
              "eb_prior_slope": "bp_per_log_n", "n_households": "count", "n_events": "count"}
WINNER_COLS = ("winner_by_mean", "winner_by_t", "winner_by_eb")


def summarize_draws(per_draw: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Tidy summary (mean, sd, q50/90/95/99, min, max) per statistic plus turnover rows; no identities."""
    rows = []
    for stat, units in DIST_STATS.items():
        if stat not in per_draw.columns:
            continue
        v = per_draw[stat].astype(float).to_numpy()
        q = np.nanquantile(v, QUANTILES)
        rows.append({**meta, "kind": "distribution", "statistic": stat, "units": units,
                     "n_draws": int(np.isfinite(v).sum()), "mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)),
                     "q50": q[0], "q90": q[1], "q95": q[2], "q99": q[3], "min": float(np.nanmin(v)), "max": float(np.nanmax(v))})
    h = int(per_draw["n_households"].max())
    for col in WINNER_COLS:
        if col not in per_draw.columns:
            continue
        t = turnover(per_draw[col].to_numpy())
        for k in ("consecutive_change_share", "n_distinct_winners", "top_winner_share", "entropy_bits"):
            rows.append({**meta, "kind": "turnover", "statistic": f"{col}|{k}", "units": "share" if "share" in k else k,
                         "n_draws": t["n_draws"], "mean": float(t[k])})
        rows.append({**meta, "kind": "turnover", "statistic": f"{col}|entropy_normalized", "units": "share",
                     "n_draws": t["n_draws"], "mean": float(t["entropy_bits"] / np.log2(h)) if h > 1 else np.nan})
    return pd.DataFrame(rows)
