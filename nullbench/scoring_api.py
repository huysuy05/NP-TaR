import numpy as np
from typing import Callable, Dict, List

from .metrics.dcb import compute_dcb
from .metrics.entropy import compute_entropy
from .metrics.null_score import compute_null_score

class NullBench:
    """
    Null Robustness Benchmark.

    Usage:
        bench = NullBench(task, generators)
        scores = bench.evaluate(predict_proba_fn)
    """
    def __init__(self, task, generators, abstention_threshold: float = 0.3):
        self.task = task
        self.generators = generators
        self.abstention_threshold = abstention_threshold

    def _ensure_probs(self, probs):
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim != 2:
            raise ValueError(f"Expected 2D array, got {probs.shape}")
        
        # Normalize just in case
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return probs / row_sums
    
    def evaluate(self, predict_proba_fn: Callable):
        """
        Method to return a dict of metrics
        Metrics included are DCB, Entropy, and NRS
        """
        texts = self.task.get_test_texts()
        true_labels = np.array(self.task.get_test_labels(), dtype=int)
        K = len(self.task._labels_)
        N = len(texts)
        class_labels = self._get_class_labels(K)

        formatter = getattr(self.task, "format_with_instruction", None)
        if formatter is None:
            def formatter(x):
                return x

        def apply_instructions(batch):
            return [formatter(t) for t in batch]

        formatted_texts = apply_instructions(texts)
        probs_test = self._ensure_probs(predict_proba_fn(formatted_texts))
        preds = probs_test.argmax(axis=1)
        accuracy = float((preds == true_labels).mean())
        recall_per_class = self._compute_recall_per_class(true_labels, preds, K)
        default_class_test = self._summarize_default_class(probs_test.mean(axis=0), class_labels)

        results = {
            "task": self.task.name,
            "num_classes": K,
            "num_test_examples": N,
            "accuracy_test": accuracy,
            "recall_per_class": recall_per_class,
            "default_class_test": default_class_test,
        }

        # 2. null / low-signal families
        all_null_probs = []

        for name, gen_fn in self.generators.items():
            try:
                # some generators: fn(n)
                null_inputs = gen_fn(N)
            except TypeError:
                # others: fn(texts)
                null_inputs = gen_fn(texts)

            formatted_null_inputs = apply_instructions(null_inputs)
            probs_null = self._ensure_probs(predict_proba_fn(formatted_null_inputs))
            default_class_summary = self._summarize_default_class(probs_null.mean(axis=0), class_labels)

            dcb = compute_dcb(probs_null)
            entropy = compute_entropy(probs_null)
            abstention = float((probs_null.max(axis=1) < self.abstention_threshold).mean())
            nrs = compute_null_score(dcb, entropy, abstention, K)

            results[f"dcb_{name}"] = dcb
            results[f"entropy_{name}"] = entropy
            results[f"abstention_{name}"] = abstention
            results[f"nrs_{name}"] = nrs
            results[f"default_class_{name}"] = default_class_summary

            all_null_probs.append(probs_null)

        # 3. overall aggregate
        if all_null_probs:
            stacked = np.concatenate(all_null_probs, axis=0)
            dcb_overall = compute_dcb(stacked)
            entropy_overall = compute_entropy(stacked)
            abstention_overall = float((stacked.max(axis=1) < self.abstention_threshold).mean())
            nrs_overall = compute_null_score(dcb_overall, entropy_overall, abstention_overall, K)
            default_class_overall = self._summarize_default_class(stacked.mean(axis=0), class_labels)

            results["dcb_overall"] = dcb_overall
            results["null_entropy_overall"] = entropy_overall
            results["abstention_overall"] = abstention_overall
            results["nrs_overall"] = nrs_overall
            results["default_class_overall"] = default_class_overall

        return results

    def _get_class_labels(self, num_classes: int) -> List[str]:
        if hasattr(self.task, "labels"):
            labels = list(self.task.labels)
        elif hasattr(self.task, "_labels_"):
            labels = list(self.task._labels_)
        else:
            labels = [str(i) for i in range(num_classes)]
        return labels

    @staticmethod
    def _compute_recall_per_class(true_labels: np.ndarray, preds: np.ndarray, num_classes: int) -> List[float]:
        recalls: List[float] = []
        for cls_idx in range(num_classes):
            mask = true_labels == cls_idx
            denom = int(mask.sum())
            if denom == 0:
                recalls.append(0.0)
            else:
                correct = int(((preds == cls_idx) & mask).sum())
                recalls.append(float(correct / denom))
        return recalls

    @staticmethod
    def _summarize_default_class(mean_probs: np.ndarray, class_labels: List[str]) -> Dict[str, object]:
        idx = int(mean_probs.argmax())
        label = class_labels[idx] if idx < len(class_labels) else str(idx)
        return {
            "index": idx,
            "label": label,
            "mean_probability": float(mean_probs[idx]),
        }
        
