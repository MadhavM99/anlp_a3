import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.part1.probe import read_jsonl, strip, text_of


def determinism(table):
    total = sum(sum(counter.values()) for counter in table.values())
    return round(sum(max(counter.values()) for counter in table.values()) / max(1, total), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/part1/train")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--out", default="results/part1/probe_key.json")
    args = parser.parse_args()
    root = Path(args.data)
    period = defaultdict(lambda: defaultdict(Counter))
    period_prev = defaultdict(lambda: defaultdict(Counter))
    prefix = defaultdict(lambda: defaultdict(Counter))
    window = defaultdict(lambda: defaultdict(Counter))
    absolute = defaultdict(Counter)
    examples = []
    pairs = zip(read_jsonl(root / "cipher.jsonl", args.limit), read_jsonl(root / "cleaned.jsonl", args.limit))
    for cipher_record, plain_record in pairs:
        raw = text_of(cipher_record)
        cipher, plain = strip(raw), text_of(plain_record)
        if len(cipher) != len(plain):
            continue
        if len(examples) < 8 and 20 <= len(plain) <= 60:
            examples.append({"cipher_raw": raw, "cipher": cipher, "plain": plain})
        for i in range(len(plain)):
            if i < 12:
                prefix[i][plain[: i + 1]][cipher[i]] += 1
            if i < 400:
                absolute[(i, plain[i])][cipher[i]] += 1
            for m in range(1, 41):
                period[m][(i % m, plain[i])][cipher[i]] += 1
                if i >= 1:
                    period_prev[m][(i % m, plain[i], plain[i - 1])][cipher[i]] += 1
            for k in range(1, 7):
                if i >= k:
                    window[k][plain[i - k: i + 1]][cipher[i]] += 1
    report = {
        "cipher_given_plain_prefix_at_position": {str(i): determinism(table) for i, table in sorted(prefix.items())},
        "cipher_given_last_k_plain_and_current": {str(k): determinism(table) for k, table in sorted(window.items())},
        "cipher_given_position_and_plain_upto_400": determinism(absolute),
        "cipher_given_position_mod_m_and_plain": {str(m): determinism(table) for m, table in sorted(period.items())},
        "cipher_given_position_mod_m_plain_and_prev_plain": {str(m): determinism(table) for m, table in sorted(period_prev.items())},
        "examples": examples,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
