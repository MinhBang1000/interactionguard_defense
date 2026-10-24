from __future__ import annotations
import json
import numpy as np

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


def aggregate_windows(values, mode):
    if not values:
        return 0.
    if mode == "mean":
        return float(np.mean(values))
    if mode == "confirm2" and len(values) > 1:
        return float(sorted(values)[-2])
    return float(max(values))


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


LABEL_REFERENCE = "https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/responsible_ai/prompt_guard/inference.py"

def malicious_index(labels, key):
    matches = [int(index) for index, name in labels.items() if str(name).lower() in {"injection", "malicious", "jailbreak", "unsafe"}]
    if len(matches) == 1:
        return matches[0], "explicit_id2label"
    if key == "prompt_guard_2" and set(str(value).upper() for value in labels.values()) == {"LABEL_0", "LABEL_1"}:
        return 1, LABEL_REFERENCE
    raise ValueError(f"No verified malicious label mapping for {key}: {labels}")

