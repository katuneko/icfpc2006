#!/usr/bin/env python3
import argparse
import struct

MAGIC = b"UMD0"


def read_u32_be(f):
    data = f.read(4)
    if len(data) != 4:
        raise EOFError
    return struct.unpack(">I", data)[0]


def iter_arrays(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"bad magic in {path}: {magic!r}")
        count = read_u32_be(f)
        for _ in range(count):
            arr_id = read_u32_be(f)
            active = read_u32_be(f)
            length = read_u32_be(f)
            data = f.read(length * 4)
            if len(data) != length * 4:
                raise EOFError
            yield arr_id, active, length, data


def first_word_diff_idx(a: bytes, b: bytes) -> int | None:
    if len(a) != len(b):
        return 0
    # Scan 4-byte words.
    for off in range(0, len(a), 4):
        if a[off : off + 4] != b[off : off + 4]:
            return off // 4
    return None


def decode_words(data: bytes):
    if not data:
        return ()
    return struct.unpack(">" + "I" * (len(data) // 4), data)


def main():
    ap = argparse.ArgumentParser(description="Compare two UMD0 dumps array-by-array")
    ap.add_argument("dump_a")
    ap.add_argument("dump_b")
    ap.add_argument("--max-arrays", type=int, default=0, help="Compare only first N arrays (0=all)")
    ap.add_argument(
        "--decode-max-len",
        type=int,
        default=2048,
        help="If an array differs and len<=N, decode and show word diffs",
    )
    ap.add_argument(
        "--max-word-diffs",
        type=int,
        default=40,
        help="Show at most this many per-array word diffs",
    )
    args = ap.parse_args()

    it_a = iter_arrays(args.dump_a)
    it_b = iter_arrays(args.dump_b)

    diffs = []
    total = 0

    while True:
        if args.max_arrays and total >= args.max_arrays:
            break
        try:
            ida, acta, lena, dataa = next(it_a)
        except StopIteration:
            ida = None
        try:
            idb, actb, lenb, datab = next(it_b)
        except StopIteration:
            idb = None

        if ida is None and idb is None:
            break
        if ida is None or idb is None:
            diffs.append(("count-mismatch", ida, idb, acta if ida is not None else None, actb if idb is not None else None, lena if ida is not None else None, lenb if idb is not None else None, None))
            break
        if ida != idb:
            diffs.append(("id-mismatch", ida, idb, acta, actb, lena, lenb, None))
            break

        total += 1

        if acta != actb or lena != lenb:
            diffs.append(("meta", ida, idb, acta, actb, lena, lenb, None))
            continue
        if dataa == datab:
            continue

        first = first_word_diff_idx(dataa, datab)
        entry = ["data", ida, idb, acta, actb, lena, lenb, first]

        if lena <= args.decode_max_len:
            wa = decode_words(dataa)
            wb = decode_words(datab)
            wd = []
            for i, (x, y) in enumerate(zip(wa, wb)):
                if x != y:
                    wd.append((i, x, y))
                    if len(wd) >= args.max_word_diffs:
                        break
            entry.append(wd)
        diffs.append(tuple(entry))

    print(f"compared arrays: {total}")
    print(f"diff arrays: {len(diffs)}")

    for d in diffs:
        kind = d[0]
        if kind in ("id-mismatch", "count-mismatch"):
            print(d)
            continue
        if kind == "meta":
            _, ida, _idb, acta, actb, lena, lenb, _ = d
            print(f"array {ida}: meta differs active {acta}->{actb} len {lena}->{lenb}")
            continue
        if kind == "data":
            if len(d) == 8:
                _, ida, _idb, _acta, _actb, lena, _lenb, first = d
                print(f"array {ida}: data differs len={lena} first_diff_idx={first}")
            else:
                _, ida, _idb, _acta, _actb, lena, _lenb, first, wd = d
                print(f"array {ida}: data differs len={lena} first_diff_idx={first} diffs={len(wd)}")
                for i, x, y in wd:
                    print(f"  [{i}] {x:#010x} -> {y:#010x}")


if __name__ == "__main__":
    main()
