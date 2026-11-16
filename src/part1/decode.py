import time

import torch

from src.part1.data import PAD, UNK, encode_window, windows
from src.part1.metrics import DecryptionMetrics


@torch.no_grad()
def decrypt_text(model, cipher_text, cipher_vocab, plain_vocab, cfg, device, max_chunks=None):
    cipher_ids = cipher_vocab.encode(cipher_text, cipher_vocab.stoi[UNK])
    starts = windows(len(cipher_ids), cfg["patch"])
    if max_chunks is not None:
        starts = starts[:max_chunks]
    pad_id = cipher_vocab.stoi[PAD]
    pieces = []
    for offset in range(0, len(starts), cfg["eval_batch_size"]):
        batch = [encode_window(cipher_ids, None, start, cfg["patch"], cfg["context"], pad_id)[0] for start in starts[offset:offset + cfg["eval_batch_size"]]]
        x = torch.tensor(batch, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg["amp"] and device.type == "cuda"):
            logits = model(x)
        predictions = logits[:, cfg["context"]:].argmax(dim=-1).tolist()
        for start, row in zip(starts[offset:offset + cfg["eval_batch_size"]], predictions):
            pieces.append(plain_vocab.decode(row[:max(0, min(cfg["patch"], len(cipher_ids) - start))]))
    return "".join(pieces)


def decrypt_corpus(model, corpus, cipher_vocab, plain_vocab, cfg, device, max_docs=None, max_chunks=None, max_minutes=None):
    model.eval()
    metrics = DecryptionMetrics()
    records = []
    started = time.time()
    total = len(corpus) if max_docs is None else min(max_docs, len(corpus))
    for position in range(total):
        if max_minutes is not None and time.time() - started > max_minutes * 60:
            break
        doc_id, cipher_text, plain_text = corpus.get(position)
        prediction = decrypt_text(model, cipher_text, cipher_vocab, plain_vocab, cfg, device, max_chunks)
        records.append({"id": doc_id, "text": prediction})
        if plain_text is not None:
            gold = plain_text[:len(prediction)]
            metrics.update(prediction, gold)
    return records, metrics.compute() if metrics.docs else None
