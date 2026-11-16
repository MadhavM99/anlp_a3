import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

PUNCTUATION = frozenset("!'+,-./:;<=>?[]_{}")
PAD = "<pad>"
UNK = "<unk>"
IGNORE = -100


def strip_cipher(text):
    return "".join(ch for ch in text if ch not in PUNCTUATION)


def text_of(record):
    if "text" in record:
        return record["text"]
    return max((value for key, value in record.items() if isinstance(value, str) and key != "id"), key=len)


class Vocab:
    def __init__(self, chars, specials):
        self.itos = list(specials) + sorted(set(chars) - set(specials))
        self.stoi = {ch: index for index, ch in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text, unknown):
        return [self.stoi.get(ch, unknown) for ch in text]

    def decode(self, ids):
        return "".join(self.itos[index] for index in ids)

    def to_list(self):
        return list(self.itos)

    @classmethod
    def from_list(cls, itos):
        vocab = cls.__new__(cls)
        vocab.itos = list(itos)
        vocab.stoi = {ch: index for index, ch in enumerate(vocab.itos)}
        return vocab


class JsonlIndex:
    def __init__(self, path, limit=None):
        self.path = Path(path)
        self.offsets = []
        with open(self.path, "rb") as handle:
            while limit is None or len(self.offsets) < limit:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        self.handle = None

    def __len__(self):
        return len(self.offsets)

    def read(self, index):
        if self.handle is None:
            self.handle = open(self.path, "rb")
        self.handle.seek(self.offsets[index])
        return json.loads(self.handle.readline())


class PairCorpus:
    def __init__(self, cipher_path, plain_path, limit=None, drop_misaligned=True):
        self.cipher = JsonlIndex(cipher_path, limit)
        self.plain = JsonlIndex(plain_path, limit) if plain_path is not None and Path(plain_path).exists() else None
        self.docs, self.dropped = [], 0
        self.cipher_chars, self.plain_chars = set(), set()
        for index in range(len(self.cipher)):
            record = self.cipher.read(index)
            cipher_text = strip_cipher(text_of(record))
            self.cipher_chars.update(cipher_text)
            if self.plain is None:
                self.docs.append((index, record.get("id", index), len(cipher_text)))
                continue
            plain_text = text_of(self.plain.read(index))
            if len(plain_text) != len(cipher_text):
                self.dropped += 1
                if drop_misaligned:
                    continue
            self.plain_chars.update(plain_text)
            self.docs.append((index, record.get("id", index), len(cipher_text)))

    def __len__(self):
        return len(self.docs)

    def get(self, position):
        index, doc_id, _ = self.docs[position]
        cipher_text = strip_cipher(text_of(self.cipher.read(index)))
        plain_text = text_of(self.plain.read(index)) if self.plain is not None else None
        return doc_id, cipher_text, plain_text


def split_paths(root, split):
    folder = Path(root) / split
    return folder / "cipher.jsonl", folder / "cleaned.jsonl"


def heldout_paths(path):
    path = Path(path)
    if path.is_file():
        return path, None
    cipher = path / "cipher.jsonl"
    if not cipher.exists():
        cipher = next(candidate for candidate in sorted(path.glob("*.jsonl")) if candidate.name != "cleaned.jsonl")
    plain = path / "cleaned.jsonl"
    return cipher, plain if plain.exists() else None


def windows(length, patch):
    return list(range(0, max(length, 1), patch))


def encode_window(cipher_ids, plain_ids, start, patch, context, pad_id):
    left = max(0, start - context)
    x = [pad_id] * (context - (start - left)) + cipher_ids[left:start + patch]
    y = [IGNORE] * context
    if plain_ids is not None:
        y += plain_ids[start:start + patch]
    else:
        y += [IGNORE] * len(cipher_ids[start:start + patch])
    width = context + patch
    x += [pad_id] * (width - len(x))
    y += [IGNORE] * (width - len(y))
    return x, y


class PatchDataset(Dataset):
    def __init__(self, corpus, cipher_vocab, plain_vocab, patch, context):
        self.corpus = corpus
        self.cipher_vocab = cipher_vocab
        self.plain_vocab = plain_vocab
        self.patch = patch
        self.context = context
        self.items = [(position, start) for position, (_, _, length) in enumerate(corpus.docs) for start in windows(length, patch)]
        self.cache = (None, None, None)

    def __len__(self):
        return len(self.items)

    def encoded(self, position):
        if self.cache[0] != position:
            _, cipher_text, plain_text = self.corpus.get(position)
            cipher_ids = self.cipher_vocab.encode(cipher_text, self.cipher_vocab.stoi[UNK])
            plain_ids = self.plain_vocab.encode(plain_text, IGNORE)
            self.cache = (position, cipher_ids, plain_ids)
        return self.cache[1], self.cache[2]

    def __getitem__(self, index):
        position, start = self.items[index]
        cipher_ids, plain_ids = self.encoded(position)
        x, y = encode_window(cipher_ids, plain_ids, start, self.patch, self.context, self.cipher_vocab.stoi[PAD])
        return torch.tensor(x), torch.tensor(y)


def build_vocabs(corpus):
    cipher_vocab = Vocab(corpus.cipher_chars, [PAD, UNK])
    plain_vocab = Vocab(corpus.plain_chars, [])
    return cipher_vocab, plain_vocab
