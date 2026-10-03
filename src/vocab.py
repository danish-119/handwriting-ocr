"""
Character vocabulary: turns text into numbers (for training) and back (for predictions).

A neural network can only work with numbers, so every character gets an ID:

    ID 0  -> CTC "blank"   (special: means "no character at this position")
    ID 1  -> [UNK]         (special: any character never seen in the training data)
    ID 2+ -> the real characters found in the TRAINING transcriptions: ' ', '!', ... 'A', ... 'z'

IMPORTANT: the vocabulary is built only from the training labels. Using the test
labels here would leak information about the test set into the model.
"""

import json
from collections import Counter
from pathlib import Path

BLANK = "[BLANK]"
UNK = "[UNK]"
UNK_DISPLAY = "\N{REPLACEMENT CHARACTER}"  # how an unknown character is shown in predictions


class Vocabulary:
    def __init__(self, characters):
        self.characters = list(characters)                 # the real characters
        self.id_to_char = [BLANK, UNK] + self.characters   # integer -> character
        self.char_to_id = {c: i for i, c in enumerate(self.id_to_char)}  # character -> integer
        self.blank_id = 0
        self.unk_id = 1

    @property
    def size(self) -> int:
        """Number of classes the model must predict (characters + blank + unknown)."""
        return len(self.id_to_char)

    # ---- building / saving ------------------------------------------------
    @classmethod
    def from_texts(cls, texts):
        """Build the vocabulary from a list of (training!) transcriptions."""
        counts = Counter("".join(texts))
        return cls(sorted(counts))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"characters": self.characters}, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Vocabulary file not found: {path}\n"
                "It is created during training (src/train.py or the notebook)."
            )
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["characters"])

    # ---- text <-> numbers -------------------------------------------------
    def encode(self, text: str):
        """'Hi!' -> [ID of 'H', ID of 'i', ID of '!']. Unknown characters become UNK."""
        return [self.char_to_id.get(c, self.unk_id) for c in text]

    def decode(self, ids):
        """[IDs] -> text. Blanks are skipped; UNK is shown as a replacement character."""
        chars = []
        for i in ids:
            i = int(i)
            if i == self.blank_id or i < 0 or i >= self.size:
                continue
            chars.append(UNK_DISPLAY if i == self.unk_id else self.id_to_char[i])
        return "".join(chars)

    def __repr__(self):
        return f"Vocabulary({self.size} classes: blank + unk + {len(self.characters)} characters)"

