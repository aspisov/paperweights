from collections import Counter
from typing import List, Dict, Tuple


class BPE:
    """Byte Pair Encoding implementation."""

    INITIAL_VOCAB_SIZE = 256

    def __init__(self, text: str, vocab_size: int):
        """Initialize BPE with text and desired vocabulary size.

        Args:
            text: Input text to train on.
            vocab_size: Desired size of the vocabulary.
        """
        if vocab_size <= self.INITIAL_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be greater than {self.INITIAL_VOCAB_SIZE}"
            )

        self.vocab_size = vocab_size
        initial_tokens = list(map(int, text.encode("utf-8")))
        self.merges = self._train(initial_tokens)
        self.vocab = {
            idx: bytes([idx]) for idx in range(self.INITIAL_VOCAB_SIZE)
        }
        for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

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

    def _train(self, tokens: List[int]) -> Dict[Tuple[int, int], int]:
        """Train BPE on the given tokens."""
        merges = {}
        next_token = self.INITIAL_VOCAB_SIZE
        while next_token < self.vocab_size:
            pair = Counter(zip(tokens, tokens[1:])).most_common(1)[0][0]
            tokens = self._merge(tokens, pair, next_token)
            merges[pair] = next_token
            next_token += 1
        return merges

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into text."""
        tokens = b"".join(self.vocab[idx] for idx in token_ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of token IDs."""
        tokens = list(text.encode("utf-8"))
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
                return new_tokens
            tokens = new_tokens

    def __repr__(self) -> str:
        """Return a string representation of the BPE object."""
        return f"BPE(vocab_size={self.vocab_size}, merges={len(self.merges)})"
