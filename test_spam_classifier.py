import unittest

from spam_classifier import DEFAULT_DATA, NaiveBayesClassifier, load_dataset, tokenize


class TokenizeTest(unittest.TestCase):

    def test_lowercases_and_splits(self):
        self.assertEqual(tokenize("Hello, WORLD!"), ["hello", "world"])

    def test_keeps_currency_and_numbers(self):
        self.assertEqual(tokenize("Win $1000 now"), ["win", "$1000", "now"])


class NaiveBayesTest(unittest.TestCase):

    def setUp(self):
        self.model = NaiveBayesClassifier().fit([
            ("free prize click now", "spam"),
            ("win free cash", "spam"),
            ("lunch tomorrow at noon", "ham"),
            ("see you at the meeting", "ham"),
        ])

    def test_predicts_spam(self):
        label, _ = self.model.predict("free cash prize")
        self.assertEqual(label, "spam")

    def test_predicts_ham(self):
        label, _ = self.model.predict("meeting at noon")
        self.assertEqual(label, "ham")

    def test_probabilities_sum_to_one(self):
        probabilities = self.model.predict_proba("anything at all")
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_unknown_words_fall_back_to_priors(self):
        probabilities = self.model.predict_proba("zzz qqq")
        self.assertAlmostEqual(probabilities["spam"], 0.5)

    def test_empty_training_set_raises(self):
        with self.assertRaises(ValueError):
            NaiveBayesClassifier().fit([])


class BundledDatasetTest(unittest.TestCase):

    def test_classifies_obvious_examples(self):
        model = NaiveBayesClassifier().fit(load_dataset(DEFAULT_DATA))
        self.assertEqual(model.predict("Claim your free cash prize now")[0], "spam")
        self.assertEqual(model.predict("Can we move our meeting to Friday?")[0], "ham")


if __name__ == "__main__":
    unittest.main()
