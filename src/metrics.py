"""
OCR metrics: Character Error Rate (CER) and Word Error Rate (WER).

Both are based on the "edit distance" (Levenshtein distance): the smallest
number of single edits needed to turn the prediction into the correct text.
An edit is one of:
    substitution  (wrong item)      "worid" -> "world"   (i -> l)
    deletion      (extra item)      "helllo" -> "hello"
    insertion     (missing item)    "helo"  -> "hello"

CER = character edits / number of characters in the correct text
WER = word edits      / number of words in the correct text

0.0 means perfect. 0.10 means "about 1 mistake per 10 characters (or words)".
They can be above 1.0 if the prediction is much longer than the real text.
"""


def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (strings or lists of words)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,                       # deletion
                    current[j - 1] + 1,                    # insertion
                    previous[j - 1] + (item_a != item_b),  # substitution (0 if equal)
                )
            )
        previous = current
    return previous[-1]


def cer(actual: str, predicted: str) -> float:
    """Character Error Rate for one example."""
    if len(actual) == 0:
        return float(len(predicted) > 0)
    return edit_distance(actual, predicted) / len(actual)


def wer(actual: str, predicted: str) -> float:
    """Word Error Rate for one example (words are split on spaces)."""
    actual_words, predicted_words = actual.split(), predicted.split()
    if len(actual_words) == 0:
        return float(len(predicted_words) > 0)
    return edit_distance(actual_words, predicted_words) / len(actual_words)


def corpus_cer(actuals, predictions) -> float:
    """CER over a whole dataset: total character edits / total characters."""
    edits = sum(edit_distance(a, p) for a, p in zip(actuals, predictions))
    total = sum(len(a) for a in actuals)
    return edits / max(total, 1)


def corpus_wer(actuals, predictions) -> float:
    """WER over a whole dataset: total word edits / total words."""
    edits = sum(edit_distance(a.split(), p.split()) for a, p in zip(actuals, predictions))
    total = sum(len(a.split()) for a in actuals)
    return edits / max(total, 1)


def align(actual: str, predicted: str):
    """List the individual character edits between `actual` and `predicted`.

    Returns tuples like ("substitution", "l", "i"), ("deletion", "e", ""),
    ("insertion", "", "x"). Used for the error analysis (which letters get confused).
    """
    n, m = len(actual), len(predicted)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist[i][j] = min(dist[i - 1][j] + 1, dist[i][j - 1] + 1,
                             dist[i - 1][j - 1] + (actual[i - 1] != predicted[j - 1]))
    # walk back from the end to recover which edits were made
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + (actual[i - 1] != predicted[j - 1]):
            if actual[i - 1] != predicted[j - 1]:
                ops.append(("substitution", actual[i - 1], predicted[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            ops.append(("deletion", actual[i - 1], ""))     # model missed this character
            i -= 1
        else:
            ops.append(("insertion", "", predicted[j - 1]))  # model added an extra character
            j -= 1
    return ops[::-1]


if __name__ == "__main__":
    # Small demonstration (the same example as in the README)
    actual, predicted = "hello world", "helo world"
    print(f"Actual   : {actual!r}")
    print(f"Predicted: {predicted!r}")
    print(f"CER = {cer(actual, predicted):.3f}   (1 missing 'l' / 11 characters)")
    print(f"WER = {wer(actual, predicted):.3f}   (1 wrong word / 2 words)")
