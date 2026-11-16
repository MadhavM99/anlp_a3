from collections import Counter

from rapidfuzz.distance import Levenshtein


def word_spans(text):
    spans, start = [], None
    for index, ch in enumerate(text):
        if ch != " " and start is None:
            start = index
        if ch == " " and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(text)))
    return spans


class DecryptionMetrics:
    def __init__(self):
        self.chars = 0
        self.correct_chars = 0
        self.words = 0
        self.correct_words = 0
        self.distance = 0
        self.normalized_distance = 0.0
        self.docs = 0
        self.true_positive = Counter()
        self.predicted = Counter()
        self.actual = Counter()

    def update(self, prediction, gold):
        self.docs += 1
        self.chars += len(gold)
        for p, g in zip(prediction, gold):
            self.predicted[p] += 1
            self.actual[g] += 1
            if p == g:
                self.correct_chars += 1
                self.true_positive[g] += 1
        for start, end in word_spans(gold):
            self.words += 1
            self.correct_words += prediction[start:end] == gold[start:end]
        distance = Levenshtein.distance(prediction, gold)
        self.distance += distance
        self.normalized_distance += distance / max(1, len(gold))

    def compute(self):
        labels = sorted(self.actual)
        precisions, recalls, f1s = [], [], []
        for label in labels:
            precision = self.true_positive[label] / self.predicted[label] if self.predicted[label] else 0.0
            recall = self.true_positive[label] / self.actual[label]
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        count = max(1, len(labels))
        return {
            "char_accuracy": self.correct_chars / max(1, self.chars),
            "word_accuracy": self.correct_words / max(1, self.words),
            "precision": sum(precisions) / count,
            "recall": sum(recalls) / count,
            "f1": sum(f1s) / count,
            "levenshtein_mean": self.distance / max(1, self.docs),
            "levenshtein_normalized": self.normalized_distance / max(1, self.docs),
            "docs": self.docs,
            "chars": self.chars,
        }
