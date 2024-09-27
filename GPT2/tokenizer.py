import regex
import json
from collections import Counter
from typing import List, Dict, Tuple
from tqdm import tqdm


class Tokenizer:
    """Byte Pair Encoding implementation with regex pre-tokenization."""

    INITIAL_VOCAB_SIZE = 256

    def __init__(self, text: str = None, vocab_size: int = None):
        """Initialize BPE with text and desired vocabulary size."""
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.inverse_vocab = {}
        self.regex_pattern = self._compile_regex_pattern()

        if text is not None and vocab_size is not None:
            if vocab_size <= self.INITIAL_VOCAB_SIZE:
                raise ValueError(
                    f"vocab_size must be greater than {self.INITIAL_VOCAB_SIZE}"
                )
            self._train(text)

    @staticmethod
    def _compile_regex_pattern():
        """Compile the regex pattern for pre-tokenization."""
        pat_str = "|".join(
            [
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                r"\p{N}{1,3}",
                r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
                r"\s*[\r\n]+",
                r"\s+(?!\S)",
                r"\s+",
            ]
        )
        return regex.compile(pat_str)

    def _pre_tokenize(self, text: str) -> List[int]:
        """Pre-tokenize the text using the regex pattern."""
        return [
            ord(char)
            for match in self.regex_pattern.findall(text)
            for char in match
        ]

    def _train(self, text: str):
        """Train BPE on the given text."""
        tokens = self._pre_tokenize(text)
        self.vocab = {i: bytes([i]) for i in range(self.INITIAL_VOCAB_SIZE)}
        self.inverse_vocab = {
            bytes([i]): i for i in range(self.INITIAL_VOCAB_SIZE)
        }

        next_token = self.INITIAL_VOCAB_SIZE

        with tqdm(
            total=self.vocab_size - next_token, desc="Training BPE"
        ) as pbar:
            while next_token < self.vocab_size:
                pair = Counter(zip(tokens, tokens[1:])).most_common(1)[0][0]
                new_token = self.vocab[pair[0]] + self.vocab[pair[1]]
                self.vocab[next_token] = new_token
                self.inverse_vocab[new_token] = next_token
                self.merges[pair] = next_token

                tokens = self._merge(tokens, pair, next_token)
                next_token += 1
                pbar.update(1)

    @staticmethod
    def _merge(tokens: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """Merge paired tokens in the list of tokens."""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(idx)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of token IDs."""
        tokens = self._pre_tokenize(text)
        encoded = []
        i = 0
        while i < len(tokens):
            substr = bytes(tokens[i : i + 2])
            if substr in self.inverse_vocab:
                encoded.append(self.inverse_vocab[substr])
                i += 2
            else:
                encoded.append(tokens[i])
                i += 1
        return encoded

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into text."""
        tokens = [self.vocab[idx] for idx in token_ids]
        return b"".join(tokens).decode("utf-8", errors="replace")

    def save(self, path: str):
        """Save the tokenizer to a JSON file."""
        data = {
            "vocab_size": self.vocab_size,
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load the tokenizer from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.vocab_size = data["vocab_size"]
        tokenizer.merges = {
            tuple(map(int, k.split(","))): v for k, v in data["merges"].items()
        }
        tokenizer.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tokenizer.inverse_vocab = {
            bytes(v): int(k) for k, v in data["vocab"].items()
        }

        return tokenizer

    def __repr__(self) -> str:
        """Return a string representation of the BPE object."""
        return f"BPE(vocab_size={self.vocab_size}, merges={len(self.merges)})"
