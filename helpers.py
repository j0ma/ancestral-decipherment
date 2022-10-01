from collections import Counter


def encrypt(msg, key, sep=""):
    return sep.join(key[c] for c in msg)


def gen_order_stat_map(plaintext: str) -> dict[str, int]:
    return {sym: ix for ix, (sym, _) in enumerate(Counter(plaintext).most_common())}


def reverse(d):
    return {v: k for k, v in d.items()}


def decrypt(ciphertext, key):
    return encrypt(ciphertext, reverse(key))


def infer_key(plaintext, ciphertext):
    k_plain = gen_order_stat_map(plaintext)
    k_cipher_inv = {v: k for k, v in gen_order_stat_map(ciphertext).items()}

    return {sym: k_cipher_inv[freq] for sym, freq in k_plain.items()}


def read_txt(f):
    with open(f, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line
