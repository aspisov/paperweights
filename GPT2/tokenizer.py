import regex
import json
from collections import Counter
from typing import List, Dict, Tuple


class Tokenizer:
    """Byte Pair Encoding implementation with regex pre-tokenization, EOT special token, and save/load functionality."""

    INITIAL_VOCAB_SIZE = 256

    def __init__(self, text: str = None, vocab_size: int = None):
        """Initialize BPE with text and desired vocabulary size.

        Args:
            text: Input text to train on. If None, the tokenizer will be initialized empty.
            vocab_size: Desired size of the vocabulary. If None, the tokenizer will be initialized empty.
        """
        self.special_tokens = {"<|endoftext|>": 0}
        self.special_tokens_reverse = {
            v: k for k, v in self.special_tokens.items()
        }

        if text is not None and vocab_size is not None:
            if vocab_size <= self.INITIAL_VOCAB_SIZE + len(self.special_tokens):
                raise ValueError(
                    f"vocab_size must be greater than {self.INITIAL_VOCAB_SIZE + len(self.special_tokens)}"
                )

            self.vocab_size = vocab_size
            self.regex_pattern = self._compile_regex_pattern()
            initial_tokens = self._pre_tokenize(text)
            self.merges = self._train(initial_tokens)
            self.vocab = {
                idx: bytes([idx]) for idx in range(self.INITIAL_VOCAB_SIZE)
            }
            for pair, idx in self.merges.items():
                self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

            # Add EOT token to vocab
            self.vocab[self.special_tokens["<|endoftext|>"]] = (
                "<|endoftext|>".encode("utf-8")
            )
        else:
            self.vocab_size = None
            self.regex_pattern = None
            self.merges = {}
            self.vocab = {}

    @staticmethod
    def _compile_regex_pattern():
        """Compile the regex pattern for pre-tokenization."""
        pat_str = "|".join(
            [
                r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                r"""\p{N}{1,3}""",
                r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
                r"""\s*[\r\n]+""",
                r"""\s+(?!\S)""",
                r"""\s+""",
            ]
        )
        return regex.compile(pat_str)

    def _pre_tokenize(self, text: str) -> List[int]:
        """Pre-tokenize the text using the regex pattern."""
        tokens = []
        for match in self.regex_pattern.findall(text):
            tokens.extend(list(match.encode("utf-8")))
        return tokens

    def _train(self, tokens: List[int]) -> Dict[Tuple[int, int], int]:
        """Train BPE on the given tokens."""
        merges = {}
        next_token = self.INITIAL_VOCAB_SIZE + len(self.special_tokens)
        while next_token < self.vocab_size:
            pair = Counter(zip(tokens, tokens[1:])).most_common(1)[0][0]
            tokens = self._merge(tokens, pair, next_token)
            merges[pair] = next_token
            next_token += 1
        return merges

    @staticmethod
    def _merge(tokens: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """Merge paired tokens in the list of tokens."""
        new_ids = [tokens[0]]
        for i in range(1, len(tokens)):
            if (tokens[i - 1], tokens[i]) == pair:
                new_ids[-1] = idx
            else:
                new_ids.append(tokens[i])
        return new_ids

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of token IDs."""
        tokens = self._pre_tokenize(text)
        while True:
            new_tokens = []
            i = 0
            merges_count = 0
            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and (tokens[i], tokens[i + 1]) in self.merges
                ):
                    new_tokens.append(self.merges[tokens[i], tokens[i + 1]])
                    i += 2
                    merges_count += 1
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            if merges_count == 0:
                break
            tokens = new_tokens

        # Add EOT token at the end
        tokens.append(self.special_tokens["<|endoftext|>"])
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into text."""
        tokens = []
        for idx in token_ids:
            if idx in self.special_tokens_reverse:
                if idx == self.special_tokens["<|endoftext|>"]:
                    break  # Stop decoding at EOT token
                tokens.append(self.special_tokens_reverse[idx].encode("utf-8"))
            else:
                tokens.append(self.vocab[idx])
        text = b"".join(tokens).decode("utf-8", errors="replace")
        return text

    def save(self, path: str):
        """Save the tokenizer to a JSON file."""
        data = {
            "vocab_size": self.vocab_size,
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            "vocab": {k: list(v) for k, v in self.vocab.items()},
            "special_tokens": self.special_tokens,
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
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.special_tokens_reverse = {
            v: k for k, v in tokenizer.special_tokens.items()
        }
        tokenizer.regex_pattern = tokenizer._compile_regex_pattern()

        return tokenizer

    def __repr__(self) -> str:
        """Return a string representation of the BPE object."""
        return f"BPE(vocab_size={self.vocab_size}, merges={len(self.merges)})"
