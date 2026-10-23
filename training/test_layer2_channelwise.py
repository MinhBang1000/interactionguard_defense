import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from models.detectors.channel_payloads import channel_payloads
from experiments.layer2_channelwise_test import routing_summary


class ChannelTests(unittest.TestCase):
    def test_payloads_only(self):
        sequence = "[PROMPT]\nuser\n[SEP]\n[RETRIEVE]\n[Doc 1]\nfirst\n[Doc 2]\nsecond\n[SEP]\n[AGENT]\nnot a payload\n[SEP]\n[TOOL: search]\ntool data\n[EOS]"
        self.assertEqual(channel_payloads(sequence), [("prompt", "user"), ("retrieval", "first"), ("retrieval", "second"), ("tool", "tool data")])

    def test_raw_and_empty(self):
        self.assertEqual(channel_payloads("plain text"), [("prompt", "plain text")])
        self.assertEqual(channel_payloads("[AGENT]\nreasoning\n[EOS]"), [])

    def test_strict_routing_and_pending(self):
        summary = routing_summary(np.array([1, 1, 0, 0]), np.array([.1, .3, .4, .2]),
                                  np.array([.99, .8, .4, .99]), .2, .5)
        self.assertEqual(summary["l1_l2_only"]["tp"], 1)
        self.assertEqual(summary["l1_l2_only"]["fn"], 1)
        self.assertEqual(summary["l2_processed"], 2)
        self.assertEqual(summary["l3_pending"], 1)
        self.assertEqual(summary["l3_pending_negative"], 1)
        self.assertEqual(summary["l1_l2_only"]["fp"], 0)

    def test_native_tokenizer_windows_and_aggregation(self):
        try:
            import torch
            from transformers import AutoTokenizer
            from models.detectors.layer2_detector import Layer2Detector
        except ImportError:
            self.skipTest("Torch/Transformers not installed")
        snapshot = Path.home() / ".cache/huggingface/hub/models--protectai--deberta-v3-base-prompt-injection-v2/snapshots/90c9989b1a342275dd0d1a95aad283c04e075671"
        if not snapshot.exists():
            self.skipTest("Optional cached HF tokenizer not available")
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        tail_id = tokenizer(" dangerous", add_special_tokens=False)["input_ids"][-1]

        class TailModel:
            def __call__(self, input_ids, **kwargs):
                positive = (input_ids == tail_id).any(dim=1).float()
                return SimpleNamespace(logits=torch.stack((torch.zeros_like(positive), positive * 5), dim=1))

        detector = Layer2Detector.__new__(Layer2Detector)
        detector.tokenizer, detector.model = tokenizer, TailModel()
        detector.device, detector.threshold = torch.device("cpu"), .5
        detector.max_length, detector.window_overlap, detector.malicious_index = 512, 64, 1
        long_text = "hello " * 1000 + " dangerous"
        sequence = "[PROMPT]\nhello\n[SEP]\n[TOOL: search]\n" + long_text + "\n[EOS]"
        result = detector._predict_channels([sequence, sequence, "[AGENT]\n dangerous\n[EOS]"], 2, True)
        self.assertEqual(result["unique_payloads"], 2)
        self.assertGreater(result["windows"], 2)
        self.assertEqual(result["predictions"].tolist(), [1, 1, 0])
        self.assertGreater(result["channel_probabilities"][0]["tool"], .9)
        self.assertEqual(detector._predict_channels([], 2, True)["predictions"].size, 0)


if __name__ == "__main__":
    unittest.main()
