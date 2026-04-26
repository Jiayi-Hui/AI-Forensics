import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from sonics.utils.config import cfg2dict


MANIFEST_VERSION = "1.0"
RUNS_DIR = Path("manifests") / "runs"
AVAILABLE_STATUSES = {"available", "provided", "validated"}
STRING_COLUMNS = [
    "id",
    "song_id",
    "filename",
    "title",
    "artist",
    "lyrics",
    "youtube_id",
    "label",
    "artist_overlap",
    "split",
    "source",
    "binary_label",
    "source_platform",
    "filepath",
    "provenance_url",
    "download_status",
    "download_error",
    "audio_sha256",
    "container_format",
    "manifest_version",
    "request_time",
    "raw_media_url",
    "yt_dlp_version",
    "ffmpeg_version",
    "ffprobe_version",
    "download_args",
    "normalized_at",
    "data_origin",
    "algorithm",
    "style",
    "lyrics_feature",
    "lyrics_features",
    "topic",
    "genre",
    "mood",
]


def _to_path(path):
    return Path(path).expanduser()


def utcnow_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_table(path):
    path = _to_path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _write_table(df, path):
    path = _to_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _namespace_to_dict(value):
    if isinstance(value, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: _namespace_to_dict(v) for k, v in value.items()}
    return value


def _listify(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _default_dataset_root(value):
    if value:
        return _to_path(value)
    return Path("dataset")


def _normalize_common_columns(df):
    df = df.copy()
    for col in STRING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), "").astype(str)
    for col in ["skip_time", "duration", "year", "sample_rate", "channels"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["target"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def _attach_file_status(df):
    df = df.copy()
    if "filepath" not in df.columns:
        return df
    exists = df["filepath"].fillna("").map(lambda x: bool(x) and Path(x).exists())
    if "download_status" not in df.columns:
        df["download_status"] = ""
    df.loc[exists & df["download_status"].eq(""), "download_status"] = "available"
    df.loc[~exists & df["download_status"].eq(""), "download_status"] = "missing"
    df["file_exists"] = exists
    return df


def build_master_manifest(
    real_csv,
    fake_csv,
    output_path,
    dataset_root="dataset",
    manifest_version=MANIFEST_VERSION,
):
    dataset_root = _default_dataset_root(dataset_root)
    real_df = pd.read_csv(real_csv, low_memory=False).copy()
    fake_df = pd.read_csv(fake_csv, low_memory=False).copy()

    if "id" not in real_df.columns:
        real_df["id"] = real_df["filename"]
    if "source" not in real_df.columns:
        real_df["source"] = "youtube"

    real_df["song_id"] = real_df["id"].astype(str)
    real_df["binary_label"] = "real"
    real_df["source_platform"] = "youtube"
    real_df["filepath"] = (
        dataset_root / "real_songs"
    ).as_posix() + "/" + real_df["filename"].astype(str) + ".mp3"
    real_df["provenance_url"] = real_df["youtube_id"].fillna("").map(
        lambda x: f"https://www.youtube.com/watch?v={x}" if x else ""
    )
    real_df["download_status"] = "pending"
    real_df["download_error"] = ""
    real_df["audio_sha256"] = ""
    real_df["container_format"] = "mp3"
    real_df["sample_rate"] = pd.NA
    real_df["channels"] = pd.NA
    real_df["manifest_version"] = manifest_version
    real_df["request_time"] = ""
    real_df["raw_media_url"] = ""
    real_df["yt_dlp_version"] = ""
    real_df["ffmpeg_version"] = ""
    real_df["download_args"] = ""
    real_df["normalized_at"] = ""
    real_df["data_origin"] = "sonics_real_csv"

    fake_df["song_id"] = fake_df["id"].astype(str)
    fake_df["binary_label"] = "fake"
    fake_df["source_platform"] = fake_df.get("source", "").fillna("unknown")
    fake_df["youtube_id"] = fake_df.get("youtube_id", "")
    fake_df["title"] = fake_df.get("title", fake_df["filename"])
    fake_df["artist"] = fake_df.get("artist", "")
    fake_df["year"] = fake_df.get("year", pd.NA)
    fake_df["artist_overlap"] = fake_df.get("artist_overlap", "")
    fake_df["filepath"] = (
        dataset_root / "fake_songs"
    ).as_posix() + "/" + fake_df["filename"].astype(str) + ".mp3"
    fake_df["provenance_url"] = ""
    fake_df["download_status"] = "provided"
    fake_df["download_error"] = ""
    fake_df["audio_sha256"] = ""
    fake_df["container_format"] = "mp3"
    fake_df["sample_rate"] = pd.NA
    fake_df["channels"] = pd.NA
    fake_df["manifest_version"] = manifest_version
    fake_df["request_time"] = ""
    fake_df["raw_media_url"] = ""
    fake_df["yt_dlp_version"] = ""
    fake_df["ffmpeg_version"] = ""
    fake_df["download_args"] = ""
    fake_df["normalized_at"] = ""
    fake_df["data_origin"] = "sonics_fake_csv"

    manifest_df = pd.concat([real_df, fake_df], ignore_index=True, sort=False)
    manifest_df = _normalize_common_columns(manifest_df)
    manifest_df = _attach_file_status(manifest_df)
    _write_table(manifest_df, output_path)
    return manifest_df


def load_manifest(path):
    df = _read_table(path)
    df = _normalize_common_columns(df)
    if "manifest_version" not in df.columns:
        df["manifest_version"] = MANIFEST_VERSION
    if "binary_label" not in df.columns and "target" in df.columns:
        df["binary_label"] = df["target"].map({0: "real", 1: "fake"}).fillna("unknown")
    if "source_platform" not in df.columns:
        df["source_platform"] = df.get("source", "")
    return _attach_file_status(df)


def load_subset_definition(path):
    path = _to_path(path)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        subset_manifest_path = data.get("subset_manifest_path")
        if not subset_manifest_path:
            raise ValueError(f"subset json missing subset_manifest_path: {path}")
        subset_manifest_path = (path.parent / subset_manifest_path).resolve()
        return data, load_manifest(subset_manifest_path)
    return {"subset_manifest_path": str(path)}, load_manifest(path)


def apply_manifest_filters(df, filters=None):
    if filters is None:
        return df.copy()
    filters = _namespace_to_dict(filters)
    filtered = df.copy()

    splits = _listify(filters.get("splits"))
    if splits:
        filtered = filtered[filtered["split"].isin(splits)]

    binary_labels = _listify(filters.get("binary_labels") or filters.get("labels"))
    if binary_labels and "binary_label" in filtered.columns:
        filtered = filtered[filtered["binary_label"].isin(binary_labels)]

    source_platforms = _listify(filters.get("source_platforms"))
    if source_platforms and "source_platform" in filtered.columns:
        filtered = filtered[filtered["source_platform"].isin(source_platforms)]

    targets = _listify(filters.get("targets"))
    if targets and "target" in filtered.columns:
        target_values = {int(x) for x in targets}
        filtered = filtered[filtered["target"].isin(target_values)]

    statuses = _listify(filters.get("download_statuses"))
    if statuses and "download_status" in filtered.columns:
        filtered = filtered[filtered["download_status"].isin(statuses)]

    min_duration = filters.get("min_duration")
    if min_duration is not None and "duration" in filtered.columns:
        filtered = filtered[filtered["duration"] >= float(min_duration)]

    max_duration = filters.get("max_duration")
    if max_duration is not None and "duration" in filtered.columns:
        filtered = filtered[filtered["duration"] <= float(max_duration)]

    return filtered.reset_index(drop=True)


def filter_downloaded_rows(df, require_downloaded=False):
    if not require_downloaded:
        return df.copy()
    downloaded = df[
        df["download_status"].isin(AVAILABLE_STATUSES) & df["file_exists"].fillna(False)
    ]
    return downloaded.reset_index(drop=True)


def summarize_dataframe(df):
    summary = {
        "num_rows": int(len(df)),
        "by_split": {},
        "by_target": {},
        "by_source_platform": {},
    }
    if "split" in df.columns:
        summary["by_split"] = {
            str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()
        }
    if "target" in df.columns:
        summary["by_target"] = {
            str(k): int(v) for k, v in df["target"].value_counts().sort_index().items()
        }
    if "source_platform" in df.columns:
        summary["by_source_platform"] = {
            str(k): int(v)
            for k, v in df["source_platform"].fillna("").value_counts().sort_index().items()
        }
    return summary


def split_dataframe(df):
    return {
        split_name: split_df.reset_index(drop=True)
        for split_name, split_df in df.groupby("split", dropna=False)
    }


def export_split_csvs(df, output_dir):
    output_dir = _to_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = {}
    for split_name, split_df in split_dataframe(df).items():
        filename = output_dir / f"{split_name}.csv"
        split_df.to_csv(filename, index=False)
        exported[split_name] = str(filename.resolve())
    return exported


def resolve_dataset_frames(cfg):
    dataset_cfg = getattr(cfg, "dataset", SimpleNamespace())
    manifest_path = getattr(dataset_cfg, "manifest_path", None)
    subset_path = getattr(dataset_cfg, "subset_path", None)
    require_downloaded = bool(getattr(dataset_cfg, "require_downloaded", False))
    filters = getattr(dataset_cfg, "filters", None)

    meta = {
        "manifest_path": manifest_path,
        "subset_path": subset_path,
        "source": "legacy_csv",
    }

    if subset_path:
        subset_info, df = load_subset_definition(subset_path)
        meta["source"] = "subset"
        meta["subset_definition"] = subset_info
        if not manifest_path:
            manifest_path = subset_info.get("manifest_path")
            meta["manifest_path"] = manifest_path
    elif manifest_path:
        df = load_manifest(manifest_path)
        meta["source"] = "manifest"
    else:
        train_df = pd.read_csv(dataset_cfg.train_dataframe)
        valid_df = pd.read_csv(dataset_cfg.valid_dataframe)
        test_df = pd.read_csv(dataset_cfg.test_dataframe)
        return {
            "train": train_df.reset_index(drop=True),
            "valid": valid_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True),
        }, meta

    df = apply_manifest_filters(df, filters)
    df = filter_downloaded_rows(df, require_downloaded=require_downloaded)
    meta["summary"] = summarize_dataframe(df)

    frames = {"train": pd.DataFrame(), "valid": pd.DataFrame(), "test": pd.DataFrame()}
    for split_name, split_df in split_dataframe(df).items():
        if split_name in frames:
            frames[split_name] = split_df

    for split_name in frames:
        frames[split_name] = frames[split_name].reset_index(drop=True)
    return frames, meta


def materialize_run_snapshot(
    cfg,
    mode,
    frames,
    output_dir,
    dataset_meta=None,
    extra=None,
):
    run_id = f"{mode}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    output_dir = _to_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_paths = {}
    for split_name, split_df in frames.items():
        path = output_dir / f"resolved_{mode}_{split_name}.csv"
        split_df.to_csv(path, index=False)
        resolved_paths[split_name] = str(path.resolve())

    snapshot = {
        "run_id": run_id,
        "mode": mode,
        "created_at": utcnow_iso(),
        "config": cfg2dict(cfg),
        "dataset": {
            "meta": dataset_meta or {},
            "resolved_paths": resolved_paths,
            "summary": {k: summarize_dataframe(v) for k, v in frames.items()},
        },
    }
    if extra:
        snapshot["extra"] = extra

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = RUNS_DIR / f"{run_id}.json"
    with open(snapshot_path, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2, ensure_ascii=False)
    return run_id, snapshot_path


def build_subset_manifest(
    manifest_path,
    subset_manifest_path,
    subset_meta_path,
    seed=42,
    splits=None,
    binary_labels=None,
    source_platforms=None,
    fraction=None,
    max_real=None,
    max_fake=None,
):
    df = load_manifest(manifest_path)
    filters = {
        "splits": splits,
        "binary_labels": binary_labels,
        "source_platforms": source_platforms,
    }
    filtered = apply_manifest_filters(df, filters)

    rng = pd.Series(filtered.index).sample(frac=1.0, random_state=seed).tolist()
    filtered = filtered.loc[rng].reset_index(drop=True)

    selected_parts = []
    layer_counts = defaultdict(int)
    group_cols = ["split", "binary_label", "source_platform"]

    for label_name, cap in [("real", max_real), ("fake", max_fake)]:
        part = filtered[filtered["binary_label"] == label_name].copy()
        if fraction is not None:
            chosen = (
                part.groupby(group_cols, dropna=False, group_keys=False)
                .apply(
                    lambda g: g.sample(
                        n=min(
                            len(g),
                            max(0, int(len(g) * fraction)),
                        )
                        if len(g)
                        else 0,
                        random_state=seed,
                    )
                    if len(g)
                    else g
                )
                .reset_index(drop=True)
            )
        elif cap is not None:
            pieces = []
            total = len(part)
            if total:
                group_sizes = part.groupby(group_cols, dropna=False).size().to_dict()
                allocated = {}
                remaining = int(min(cap, total))
                fractions = []
                for key, size in group_sizes.items():
                    raw = size / total * remaining
                    base = min(size, int(raw))
                    allocated[key] = base
                    fractions.append((raw - base, key))
                assigned = sum(allocated.values())
                for _, key in sorted(fractions, reverse=True):
                    if assigned >= remaining:
                        break
                    if allocated[key] < group_sizes[key]:
                        allocated[key] += 1
                        assigned += 1
                for key, group_df in part.groupby(group_cols, dropna=False):
                    take = allocated.get(key, 0)
                    if take:
                        pieces.append(group_df.sample(n=take, random_state=seed))
                chosen = pd.concat(pieces, ignore_index=True) if pieces else part.iloc[0:0]
            else:
                chosen = part.iloc[0:0]
        else:
            chosen = part

        for key, count in chosen.groupby(group_cols, dropna=False).size().items():
            layer_counts["|".join("" if pd.isna(v) else str(v) for v in key)] += int(count)
        selected_parts.append(chosen)

    subset_df = pd.concat(selected_parts, ignore_index=True).reset_index(drop=True)
    subset_df = subset_df.sort_values(["split", "target", "source_platform", "song_id"]).reset_index(
        drop=True
    )
    _write_table(subset_df, subset_manifest_path)

    subset_meta = {
        "created_at": utcnow_iso(),
        "manifest_path": str(_to_path(manifest_path).resolve()),
        "subset_manifest_path": os.path.relpath(
            str(_to_path(subset_manifest_path).resolve()),
            start=str(_to_path(subset_meta_path).resolve().parent),
        ),
        "seed": seed,
        "filters": {
            "splits": _listify(splits),
            "binary_labels": _listify(binary_labels),
            "source_platforms": _listify(source_platforms),
        },
        "fraction": fraction,
        "max_real": max_real,
        "max_fake": max_fake,
        "layer_counts": dict(layer_counts),
        "summary": summarize_dataframe(subset_df),
        "manifest_version": subset_df["manifest_version"].iloc[0]
        if len(subset_df)
        else MANIFEST_VERSION,
    }
    _to_path(subset_meta_path).parent.mkdir(parents=True, exist_ok=True)
    with open(subset_meta_path, "w", encoding="utf-8") as fh:
        json.dump(subset_meta, fh, indent=2, ensure_ascii=False)
    return subset_df, subset_meta
