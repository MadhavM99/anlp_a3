import argparse
import json
import statistics
from collections import Counter
from itertools import islice
from pathlib import Path

PUNCTUATION = set("!'+,-./:;<=>?[]_{}")


def read_jsonl(path, limit=None):
    with open(path, encoding="utf-8") as handle:
        for line in islice(handle, limit):
            yield json.loads(line)


def text_of(record):
    if "text" in record:
        return record["text"]
    return max((value for key, value in record.items() if isinstance(value, str) and key != "id"), key=len)


def strip(text):
    return "".join(ch for ch in text if ch not in PUNCTUATION)


def probe_split(root, split, limit):
    cipher_path, plain_path = root / split / "cipher.jsonl", root / split / "cleaned.jsonl"
    first_cipher = next(read_jsonl(cipher_path, 1))
    first_plain = next(read_jsonl(plain_path, 1))
    cipher_chars, plain_chars, removed = Counter(), Counter(), Counter()
    lengths, matches, total, id_matches = [], 0, 0, 0
    shift_by_position, pair_map = Counter(), {}
    for cipher, plain in zip(read_jsonl(cipher_path, limit), read_jsonl(plain_path, limit)):
        c_text, p_text = text_of(cipher), text_of(plain)
        stripped = strip(c_text)
        removed.update(ch for ch in c_text if ch in PUNCTUATION)
        cipher_chars.update(stripped)
        plain_chars.update(p_text)
        lengths.append(len(p_text))
        total += 1
        id_matches += cipher.get("id") == plain.get("id")
        if len(stripped) == len(p_text):
            matches += 1
            for index, (c, p) in enumerate(zip(stripped[:200], p_text[:200])):
                if c.isalpha() and p.isalpha():
                    shift_by_position[(index % 8, (ord(c.lower()) - ord(p.lower())) % 26)] += 1
                pair_map.setdefault(p, Counter())[c] += 1
    deterministic = sum(max(counter.values()) for counter in pair_map.values()) / max(1, sum(sum(c.values()) for c in pair_map.values()))
    return {
        "cipher_keys": {k: (v[:120] if isinstance(v, str) else v) for k, v in first_cipher.items()},
        "plain_keys": {k: (v[:120] if isinstance(v, str) else v) for k, v in first_plain.items()},
        "docs": total,
        "id_match_rate": id_matches / total,
        "aligned_after_strip": matches / total,
        "plain_len": {"mean": statistics.mean(lengths), "median": statistics.median(lengths), "max": max(lengths), "min": min(lengths), "total": sum(lengths)},
        "plain_charset": "".join(sorted(plain_chars)),
        "cipher_charset_after_strip": "".join(sorted(cipher_chars)),
        "removed_punctuation": dict(removed.most_common()),
        "plain_to_cipher_determinism": deterministic,
        "top_shift_by_position_mod8": {str(pos): Counter({s: n for (p, s), n in shift_by_position.items() if p == pos}).most_common(3) for pos in range(8)},
        "plain_to_cipher_top": {p: counter.most_common(4) for p, counter in sorted(pair_map.items())[:30]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/part1")
    parser.add_argument("--train-limit", type=int, default=100000)
    parser.add_argument("--out", default="results/part1/probe.json")
    args = parser.parse_args()
    root = Path(args.data)
    report = {split: probe_split(root, split, args.train_limit if split == "train" else None) for split in ("train", "validation", "test")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
