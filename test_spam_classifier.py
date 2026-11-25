import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from spam_classifier import DEFAULT_DATA, NaiveBayesClassifier, load_dataset, main, tokenize


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


class LoadDatasetTest(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def write(self, content):
        path = self.dir / "data.csv"
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_text_label_pairs(self):
        path = self.write("label,text\nspam,win cash\nham,see you soon\n")
        self.assertEqual(load_dataset(path), [("win cash", "spam"), ("see you soon", "ham")])

    def test_accepts_byte_order_mark(self):
        path = self.dir / "excel.csv"
        path.write_text("label,text\nspam,win cash\n", encoding="utf-8-sig")
        self.assertEqual(load_dataset(path), [("win cash", "spam")])

    def test_missing_columns_raise_value_error(self):
        with self.assertRaises(ValueError):
            load_dataset(self.write("category,message\nspam,win cash\n"))

    def test_short_row_raises_value_error(self):
        with self.assertRaises(ValueError):
            load_dataset(self.write("label,text\nspam\n"))


class CliTest(unittest.TestCase):

    def run_main(self, argv):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_classifies_message(self):
        code, out, _ = self.run_main(["Claim your free cash prize now"])
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("spam ("))

    def test_missing_data_file_is_reported(self):
        code, _, err = self.run_main(["hello", "--data", "does-not-exist.csv"])
        self.assertEqual(code, 1)
        self.assertIn("Error:", err)

    def test_header_only_data_file_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "empty.csv")
            path.write_text("label,text\n", encoding="utf-8")
            code, _, err = self.run_main(["hello", "--data", str(path)])
        self.assertEqual(code, 1)
        self.assertIn("empty dataset", err)


if __name__ == "__main__":
    unittest.main()
