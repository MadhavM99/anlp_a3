import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.part1.probe import read_jsonl, strip, text_of


def determinism(table):
    total = sum(sum(counter.values()) for counter in table.values())
    return sum(max(counter.values()) for counter in table.values()) / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/part1/train")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--out", default="results/part1/probe_dependency.json")
    args = parser.parse_args()
    root = Path(args.data)
    tables = defaultdict(lambda: defaultdict(Counter))
    by_position = defaultdict(lambda: defaultdict(Counter))
    pairs = zip(read_jsonl(root / "cipher.jsonl", args.limit), read_jsonl(root / "cleaned.jsonl", args.limit))
    for cipher_record, plain_record in pairs:
        cipher, plain = strip(text_of(cipher_record)), text_of(plain_record)
        if len(cipher) != len(plain):
            continue
        for i in range(len(plain)):
            target = plain[i]
            tables["c_i"][cipher[i]][target] += 1
            if i < 8:
                by_position[i][cipher[i]][target] += 1
            for lag in range(1, args.max_lag + 1):
                if i >= lag:
                    tables[f"c_i,c_i-{lag}"][(cipher[i], cipher[i - lag])][target] += 1
                    tables[f"c_i,p_i-{lag}"][(cipher[i], plain[i - lag])][target] += 1
            if i >= 2:
                tables["c_i,c_i-1,c_i-2"][(cipher[i], cipher[i - 1], cipher[i - 2])][target] += 1
            if i >= 3:
                tables["c_i..c_i-3"][(cipher[i], cipher[i - 1], cipher[i - 2], cipher[i - 3])][target] += 1
                tables["c_i,p_i-1..p_i-3"][(cipher[i], plain[i - 1], plain[i - 2], plain[i - 3])][target] += 1
    report = {
        "determinism_of_plain_given": {name: round(determinism(table), 4) for name, table in sorted(tables.items())},
        "contexts": {name: len(table) for name, table in sorted(tables.items())},
        "determinism_given_c_i_at_position": {str(pos): round(determinism(table), 4) for pos, table in sorted(by_position.items())},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
