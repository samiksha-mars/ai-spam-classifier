"""Spam classifier using a multinomial Naive Bayes model written from scratch."""

import argparse
import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_DATA = Path(__file__).with_name("dataset.csv")
TOKEN_RE = re.compile(r"[a-z0-9$£€]+")


def tokenize(text):
    """Lowercase the text and split it into word tokens."""
    return TOKEN_RE.findall(text.lower())


class NaiveBayesClassifier:
    """Multinomial Naive Bayes with Laplace (add-alpha) smoothing."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_counts = Counter()
        self.word_counts = {}
        self.total_words = Counter()
        self.vocabulary = set()

    def fit(self, samples):
        """Train on an iterable of (text, label) pairs."""
        for text, label in samples:
            self.class_counts[label] += 1
            counts = self.word_counts.setdefault(label, Counter())
            for token in tokenize(text):
                counts[token] += 1
                self.total_words[label] += 1
                self.vocabulary.add(token)
        if not self.class_counts:
            raise ValueError("cannot train on an empty dataset")
        return self

    def _log_scores(self, text):
        tokens = tokenize(text)
        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocabulary)
        scores = {}
        for label, doc_count in self.class_counts.items():
            score = math.log(doc_count / total_docs)
            denominator = self.total_words[label] + self.alpha * vocab_size
            counts = self.word_counts[label]
            for token in tokens:
                if token in self.vocabulary:
                    score += math.log((counts[token] + self.alpha) / denominator)
            scores[label] = score
        return scores

    def predict_proba(self, text):
        """Return a dict mapping each label to its probability."""
        scores = self._log_scores(text)
        top = max(scores.values())
        exps = {label: math.exp(s - top) for label, s in scores.items()}
        total = sum(exps.values())
        return {label: value / total for label, value in exps.items()}

    def predict(self, text):
        """Return (label, probability) for the most likely label."""
        probabilities = self.predict_proba(text)
        label = max(probabilities, key=probabilities.get)
        return label, probabilities[label]


def load_dataset(path):
    """Load (text, label) pairs from a CSV with `label,text` columns."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"label", "text"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{path}: expected a header row with 'label' and 'text' columns")
        samples = []
        for row in reader:
            if not row["label"] or row["text"] is None:
                raise ValueError(f"{path}, line {reader.line_num}: missing label or text")
            samples.append((row["text"], row["label"]))
        return samples


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify a message as spam or ham.")
    parser.add_argument("text", nargs="?", help="message to classify (reads stdin if omitted)")
    parser.add_argument("--data", default=DEFAULT_DATA, help="training CSV with label,text columns")
    args = parser.parse_args(argv)

    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        parser.error("no message given")

    try:
        model = NaiveBayesClassifier().fit(load_dataset(args.data))
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    label, probability = model.predict(text)
    print(f"{label} ({probability:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
