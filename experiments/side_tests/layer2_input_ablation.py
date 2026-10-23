"""Frozen-classifier input/window ablations; validation first, no main changes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re

import numpy as np

from .layer2_threshold_sweep import evaluate_point, select_fpr, write_csv

METHODS = ["raw_max", "raw_mean", "raw_confirm2", "json_max", "json_mean", "json_confirm2", "html_max", "html_mean", "html_confirm2", "paired_agreement"]


def unwrap_json(text, depth=0):
    """Parse valid transport JSON only; keep all leaves and non-wrapper keys."""
    if depth > 12:
        return str(text)
    if isinstance(text, str):
        try:
            value = json.loads(text)
        except (ValueError, TypeError):
            if text.lstrip().startswith(('{\\"', '[\\"', '[{\\"')):
                try:
                    value = json.loads(json.loads('"' + text + '"'))
                except (ValueError, TypeError):
                    return text
            else:
                return text
        if value == text:
            return text
        return unwrap_json(value, depth+1)
    if isinstance(text, dict):
        return "\n".join(("" if key in {"result", "text"} else str(key) + ": ") + unwrap_json(value, depth+1)
                         for key, value in text.items())
    if isinstance(text, list):
        return "\n".join(unwrap_json(value, depth+1) for value in text)
    return json.dumps(text, ensure_ascii=False)


class EvidenceHTML(HTMLParser):
    """Linearize markup while retaining hidden/script/style/comment/attribute text."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        self.parts.append(tag + " " + " ".join(f"{key}: {value}" for key, value in attrs))

    def handle_endtag(self, tag):
        self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)

    def handle_comment(self, data):
        self.parts.append(data)

    def handle_decl(self, decl):
        self.parts.append(decl)

    def unknown_decl(self, data):
        self.parts.append(data)


def linearize_html(text):
    if not re.search(r"<[!/A-Za-z][^>]*>", text):
        return text
    parser = EvidenceHTML()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        return text
    return "\n".join(parser.parts)


def aggregate_windows(values, mode):
    if not values:
        return 0.
    if mode == "mean":
        return float(np.mean(values))
    if mode == "confirm2" and len(values) > 1:
        return float(sorted(values)[-2])
    return float(max(values))


def payload_score(raw, json_values, html, method):
    if method == "paired_agreement":
        return min(aggregate_windows(raw, "max"), aggregate_windows(html, "max"))
    view, mode = method.split("_", 1)
    return aggregate_windows({"raw": raw, "json": json_values, "html": html}[view], mode)


class WindowScorer:
    def __init__(self, detector):
        self.detector = detector
        self.cache = {}

    def score(self, texts):
        import torch
        from tqdm.auto import tqdm
        unique = list(dict.fromkeys(text for text in texts if text not in self.cache))
        windows, owners = [], []
        for owner, text in enumerate(unique):
            encoded = self.detector.tokenizer(text, max_length=self.detector.max_length, truncation=True,
                stride=self.detector.window_overlap, return_overflowing_tokens=True, padding=False)
            for index in range(len(encoded["input_ids"])):
                windows.append({key: encoded[key][index] for key in self.detector.tokenizer.model_input_names if key in encoded})
                owners.append(owner)
            self.cache[text] = []
        with torch.inference_mode():
            for start in tqdm(range(0, len(windows), 16), desc="Ablation windows"):
                inputs = self.detector.tokenizer.pad(windows[start:start+16], padding=True, return_tensors="pt")
                logits = self.detector.model(**{key: value.to(self.detector.device) for key, value in inputs.items()}).logits
                scores = torch.softmax(logits.float(), dim=-1)[:, self.detector.malicious_index].cpu().numpy()
                for owner, score in zip(owners[start:start+16], scores):
                    self.cache[unique[owner]].append(float(score))


def score_rows(rows, scorer, output_path):
    from models.detectors.channel_payloads import channel_payloads
    parts = [channel_payloads(row["sequence"]) for row in rows]
    originals = list(dict.fromkeys(text for blocks in parts for _, text in blocks))
    views = {text: (text, unwrap_json(text), linearize_html(unwrap_json(text))) for text in originals}
    scorer.score([value for triple in views.values() for value in triple])
    lookup = {text: {method: payload_score(*(scorer.cache[value] for value in triple), method)
                     for method in METHODS} for text, triple in views.items()}
    result = {method: np.array([max((lookup[text][method] for _, text in blocks), default=0.) for blocks in parts]) for method in METHODS}
    with output_path.open("w") as stream:
        for i, row in enumerate(rows):
            stream.write(json.dumps({"id": row["id"], "label": row["label"],
                "scores": {method: float(result[method][i]) for method in METHODS},
                "payloads": len(parts[i]), "raw_windows": sum(len(scorer.cache[text]) for _, text in parts[i])}) + "\n")
    return result, parts, views, lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/layer2_channelwise/20261021_071550_378314"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from data.data_preprocess import ROOT, read_jsonl, sha256, write_json
    from data.prepare_training import load_prepared
    from training.layer1_common import directory_hash
    from training.layer1_pipeline import gpu_preflight
    gpu_preflight("0")
    source = json.loads((args.source / "config.json").read_text())
    if directory_hash(source["snapshot"]) != source["checkpoint_sha256"]:
        raise ValueError("Original classifier checkpoint changed")
    output = args.output_dir or ROOT / "results/side_layer2_input_ablation" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    val = load_prepared("encoder", "val")
    if sha256(ROOT / "data/training/encoder/val.jsonl") != source["validation_sha256"]:
        raise ValueError("Prepared validation changed")
    l1_run = Path(source["l1_run_dir"])
    selected = json.loads((l1_run / "autoencoder/selection_frozen.json").read_text())
    val_error_path = Path(selected["best"]["checkpoint"]).parent / "validation_errors.npy"
    if sha256(val_error_path) != source["validation_errors_sha256"]:
        raise ValueError("L1 validation scores changed")
    val_errors = np.load(val_error_path)
    from models.detectors.layer2_detector import Layer2Detector
    detector = Layer2Detector(**dict(source["model"], model_path=source["snapshot"]), device="cuda:0", local_files_only=True)
    scorer = WindowScorer(detector)
    write_json(output / "config.json", {"source_results": str(args.source.resolve()), "methods": METHODS,
        "model": source["model"], "checkpoint_sha256": source["checkpoint_sha256"],
        "validation_sha256": source["validation_sha256"], "test_sha256": source["test_sha256"],
        "physical_gpu": 0, "source_sha256": sha256(Path(__file__)), "fixed_L2_threshold": .5,
        "notes": ["no training/main changes/L3", "all source channels retained", "no evidence matching/removal by attack labels",
                  "HTML retains script/style/hidden/comments/attributes", "mean and confirm2 may dilute localized injection",
                  "paired agreement is two representations of same model, not independent evidence",
                  "test comparisons are exploratory; operational negative labels are not a semantic oracle"]})
    val_scores, _, _, _ = score_rows(val, scorer, output / "validation_predictions.jsonl")
    val_labels = np.array([row["label"] for row in val])
    selections = {method: select_fpr(val_labels, val_scores[method], .01) for method in METHODS}
    best = max(METHODS, key=lambda method: (selections[method]["validation"]["tp"],
        -selections[method]["validation"]["fp"], -METHODS.index(method)))
    write_json(output / "selection_frozen.json", {"objective": "max_validation_TP_under_1pct_standalone_FPR",
        "selections": selections, "selected_method": best,
        "validation_scores_sha256": sha256(output / "validation_predictions.jsonl")})
    print("VALIDATION_FROZEN " + best, flush=True)
    processed = ROOT / "data/processed/test_processed.jsonl"
    if sha256(processed) != source["test_sha256"]:
        raise ValueError("Test data changed")
    test, old = read_jsonl(processed), read_jsonl(args.source / "predictions.jsonl")
    if len(test) != len(old) or any(a["id"] != b["id"] or a["label"] != b["label"] for a, b in zip(test, old)):
        raise ValueError("Test scores are not aligned")
    test_scores, parts, views, lookup = score_rows(test, scorer, output / "test_predictions.jsonl")
    baseline = np.array([row["l2_probability"] for row in old])
    if not np.array_equal(test_scores["raw_max"] > .5, baseline > .5) or not np.allclose(test_scores["raw_max"], baseline, atol=2e-5):
        raise ValueError("Instrumented raw-max does not reproduce cached baseline")
    labels = np.array([row["label"] for row in test])
    errors = np.array([row["l1_error"] for row in old])
    fixed_val, fixed_test, calibrated_test = [], [], []
    for method in METHODS:
        for gate in [None] + source["thresholds"]:
            fixed_val.append({"method": method, **evaluate_point(val_labels, val_errors, val_scores[method], gate, .5)})
            fixed_test.append({"method": method, **evaluate_point(labels, errors, test_scores[method], gate, .5)})
            calibrated_test.append({"method": method, **evaluate_point(labels, errors, test_scores[method], gate, selections[method]["threshold"])})
    fp = (labels == 0) & (test_scores["raw_max"] > .5)
    # Delta-only is diagnostic, not a deployable replacement for L1-gated auditing.
    seen, delta = defaultdict(set), []
    for row, blocks in zip(test, parts):
        prior = seen[row["original_trace_id"]]
        new = [text for _, text in blocks if text not in prior]
        delta.append(max((lookup[text]["raw_max"] for text in new), default=0.))
        prior.update(text for _, text in blocks)
    full_episode, delta_episode = defaultdict(bool), defaultdict(bool)
    for row, full, incremental in zip(test, test_scores["raw_max"], delta):
        full_episode[row["original_trace_id"]] |= bool(full > .5)
        delta_episode[row["original_trace_id"]] |= bool(incremental > .5)
    assert full_episode == delta_episode
    exposure = Counter()
    for i, blocks in enumerate(parts):
        if labels[i] == 0:
            windows = sum(len(scorer.cache[text]) for _, text in blocks)
            band = "1" if windows <= 1 else "2-4" if windows <= 4 else "5-16" if windows <= 16 else "17+"
            exposure[(band, "negative")] += 1
            exposure[(band, "false_positive")] += int(fp[i])
    audit = {"baseline_fp_prefixes": int(fp.sum()),
        "baseline_fp_episodes": len({test[i]["original_trace_id"] for i in np.where(fp)[0]}),
        "unique_tool_payloads_in_FP_prefixes": len({text for i in np.where(fp)[0] for channel, text in parts[i] if channel == "tool"}),
        "delta_only_prefix_metrics": evaluate_point(labels, errors, np.array(delta), None, .5),
        "delta_vs_cumulative_episode_any_block_identical": full_episode == delta_episode,
        "window_exposure": {band: {key: exposure[(band, key)] for key in ["negative", "false_positive"]} for band in ["1", "2-4", "5-16", "17+"]}}
    examples = []
    for i in np.where(fp)[0]:
        row = test[i]
        changed = [method for method in METHODS if test_scores[method][i] <= .5]
        if changed and len(examples) < 12:
            examples.append({"id": row["id"], "label": row["label"], "corrected_by": changed,
                "scores": {method: float(test_scores[method][i]) for method in METHODS},
                "tool_views": [{"raw": text[:1600], "json": views[text][1][:1600], "html": views[text][2][:1600]}
                               for channel, text in parts[i] if channel == "tool"]})
    with (output / "payload_window_audit.jsonl").open("w") as stream:
        for raw, triple in views.items():
            stream.write(json.dumps({"raw": raw, "json": triple[1], "html": triple[2],
                "window_probabilities": {name: scorer.cache[text] for name, text in zip(["raw", "json", "html"], triple)}}) + "\n")
    write_csv(output / "validation_fixed.csv", fixed_val)
    write_csv(output / "test_fixed.csv", fixed_test)
    write_csv(output / "test_calibrated.csv", calibrated_test)
    write_json(output / "examples.json", examples)
    write_json(output / "report.json", {"validation_selected_method": best, "audit": audit,
        "validation_fixed": fixed_val, "test_fixed": fixed_test, "test_calibrated": calibrated_test,
        "selections": selections, "baseline_reproduced": True,
        "test_per_scenario": {method: {kind: evaluate_point(labels[mask], errors[mask], test_scores[method][mask], None, .5)
            for kind in sorted({row.get("attack_type") or "benign" for row in test})
            for mask in [np.array([(row.get("attack_type") or "benign") == kind for row in test])]}
            for method in METHODS}})
    print("COMPLETED " + str(output), flush=True)


if __name__ == "__main__":
    main()
