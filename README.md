# AI Spam Classifier

A tiny spam/ham text classifier built on a **multinomial Naive Bayes** model
implemented from scratch. Uses only the Python standard library.

## Features

- Naive Bayes with Laplace smoothing, written from scratch
- Ships with a small labelled dataset (`dataset.csv`) so it works out of the box
- Prints the predicted label and its probability
- Train on your own CSV with `--data`

## How it works

Each message is tokenized into lowercase words. During training the model counts
how often each word appears in spam and ham messages. To classify, it combines the
class prior with the (smoothed) likelihood of every known word, working in log space
to avoid underflow, and normalizes the scores into probabilities.

## Usage

```bash
python3 spam_classifier.py "Claim your free cash prize now"
# spam (100.0%)

python3 spam_classifier.py "Are we still on for lunch tomorrow?"
# ham (99.3%)

# Read the message from stdin
echo "You won a free cruise" | python3 spam_classifier.py

# Train on your own data (CSV with label,text columns)
python3 spam_classifier.py "some message" --data my_messages.csv
```

## Tests

```bash
python3 -m unittest test_spam_classifier.py
```
