#!/usr/bin/env python3
import argparse
import re
import struct
from pathlib import Path

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
            raise ValueError(f"bad magic: {magic!r}")
        count = read_u32_be(f)
        for _ in range(count):
            arr_id = read_u32_be(f)
            active = read_u32_be(f)
            length = read_u32_be(f)
            data = f.read(length * 4)
            if len(data) != length * 4:
                raise EOFError
            yield arr_id, active, data


def scan_packed_bytes(data, pattern):
    hits = set()
    for m in pattern.finditer(data):
        hits.add(m.group(0).decode("ascii"))
    return hits


def scan_word_chars(data, pattern):
    hits = set()
    if not data:
        return hits
    words = struct.unpack(">" + "I" * (len(data) // 4), data)
    buf = []
    for val in words:
        if val <= 0x7f:
            buf.append(val)
        else:
            if len(buf) >= 10:
                text = bytes(buf).decode("ascii", errors="ignore")
                for m in pattern.finditer(text):
                    hits.add(m.group(0))
            buf = []
    if len(buf) >= 10:
        text = bytes(buf).decode("ascii", errors="ignore")
        for m in pattern.finditer(text):
            hits.add(m.group(0))
    return hits


def iter_codes(data, bits, msb_first):
    mask = (1 << bits) - 1
    buf = 0
    count = 0
    if msb_first:
        for byte in data:
            buf = (buf << 8) | byte
            count += 8
            while count >= bits:
                shift = count - bits
                yield (buf >> shift) & mask
                count -= bits
                buf &= (1 << count) - 1 if count else 0
    else:
        for byte in data:
            buf |= byte << count
            count += 8
            while count >= bits:
                yield buf & mask
                buf >>= bits
                count -= bits


def iter_decoded_chars(data, bits, msb_first, alphabet=None, offset=0, ascii_mode=False):
    for code in iter_codes(data, bits, msb_first):
        if ascii_mode:
            if 32 <= code <= 126:
                yield chr(code)
            else:
                yield "?"
            continue
        idx = code - offset
        if 0 <= idx < len(alphabet):
            yield alphabet[idx]
        else:
            yield "?"


def scan_stream_for_keywords(char_iter, keywords, max_hits, context, filters):
    hits = []
    found = set()
    max_kw = max(len(k) for k in keywords) if keywords else 0
    carry = ""
    chunk = []
    chunk_size = 4096
    for ch in char_iter:
        chunk.append(ch)
        if len(chunk) < chunk_size:
            continue
        text = carry + "".join(chunk)
        lower = text.lower()
        for kw in keywords:
            if kw in found:
                continue
            idx = lower.find(kw)
            if idx != -1:
                start = max(0, idx - context)
                end = min(len(text), idx + len(kw) + context)
                snippet = text[start:end]
                if is_plausible_snippet(snippet, filters):
                    hits.append((kw, snippet))
                    found.add(kw)
                    if len(hits) >= max_hits:
                        return hits
        carry = text[-(max_kw - 1) :] if max_kw > 1 else ""
        chunk = []
    if chunk:
        text = carry + "".join(chunk)
        lower = text.lower()
        for kw in keywords:
            if kw in found:
                continue
            idx = lower.find(kw)
            if idx != -1:
                start = max(0, idx - context)
                end = min(len(text), idx + len(kw) + context)
                snippet = text[start:end]
                if is_plausible_snippet(snippet, filters):
                    hits.append((kw, snippet))
                    found.add(kw)
                    if len(hits) >= max_hits:
                        return hits
    return hits


def scan_stream_for_regex(char_iter, pattern, max_hits, context):
    hits = []
    seen = set()
    carry = ""
    chunk = []
    chunk_size = 4096
    carry_len = max(context * 2, 64)
    for ch in char_iter:
        chunk.append(ch)
        if len(chunk) < chunk_size:
            continue
        text = carry + "".join(chunk)
        for m in pattern.finditer(text):
            start = max(0, m.start() - context)
            end = min(len(text), m.end() + context)
            snippet = text[start:end]
            if snippet in seen:
                continue
            hits.append((m.group(0), snippet))
            seen.add(snippet)
            if len(hits) >= max_hits:
                return hits
        carry = text[-carry_len:] if len(text) > carry_len else text
        chunk = []
    if chunk:
        text = carry + "".join(chunk)
        for m in pattern.finditer(text):
            start = max(0, m.start() - context)
            end = min(len(text), m.end() + context)
            snippet = text[start:end]
            if snippet in seen:
                continue
            hits.append((m.group(0), snippet))
            seen.add(snippet)
            if len(hits) >= max_hits:
                break
    return hits


def is_plausible_snippet(snippet, filters):
    if not snippet:
        return False
    letters = sum(1 for c in snippet if c.isalpha())
    spaces = snippet.count(" ")
    qmarks = snippet.count("?")
    if letters < filters["min_letters"]:
        return False
    if spaces / len(snippet) < filters["min_space_ratio"]:
        return False
    if qmarks / len(snippet) > filters["max_q_ratio"]:
        return False
    return True


def parse_id_filter(spec):
    if not spec:
        return set()
    ids = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                start, end = end, start
            ids.update(range(start, end + 1))
        else:
            ids.add(int(part))
    return ids


def build_cons_nodes(dump_path):
    nodes = {}
    referenced = set()
    with open(dump_path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError("bad magic in dump")
        count = read_u32_be(f)
        for _ in range(count):
            arr_id = read_u32_be(f)
            _active = read_u32_be(f)
            length = read_u32_be(f)
            data = f.read(length * 4)
            if length != 3:
                continue
            nxt, mid, val = struct.unpack(">III", data)
            if val == 10 or 32 <= val <= 126:
                nodes[arr_id] = (nxt, mid, val)
                if nxt != 0:
                    referenced.add(nxt)
    heads = [nid for nid in nodes if nid not in referenced]
    return nodes, heads


def iter_cons_strings(nodes, heads, max_len, min_len):
    for head in heads:
        chars = []
        seen = set()
        cur = head
        while cur in nodes and cur not in seen:
            seen.add(cur)
            nxt, _mid, val = nodes[cur]
            if val == 10:
                chars.append("\n")
            else:
                chars.append(chr(val))
            if max_len and len(chars) >= max_len:
                break
            cur = nxt
        if min_len and len(chars) < min_len:
            continue
        yield head, "".join(chars)


def iter_cons_string_from(nodes, start, max_len):
    chars = []
    seen = set()
    cur = start
    while cur in nodes and cur not in seen:
        seen.add(cur)
        nxt, _mid, val = nodes[cur]
        if val == 10:
            chars.append("\n")
        else:
            chars.append(chr(val))
        if max_len and len(chars) >= max_len:
            break
        cur = nxt
    return "".join(chars)


def is_plain_pattern(pattern):
    return re.search(r"[.^$*+?{}\\[\\]|()\\\\]", pattern) is None


def match_cons_literal(nodes, start, codes):
    cur = start
    for code in codes:
        node = nodes.get(cur)
        if node is None:
            return False
        nxt, _mid, val = node
        if val != code:
            return False
        cur = nxt
    return True


def main():
    parser = argparse.ArgumentParser(description="Scan UM dump for part numbers.")
    parser.add_argument("dump", help="Path to UMD0 dump")
    parser.add_argument(
        "--pattern",
        default=r"[A-Z]-[0-9]{4}-[A-Z]{3}",
        help="Regex pattern for ASCII matches",
    )
    parser.add_argument(
        "--scan-packed",
        action="store_true",
        help="Scan packed 5/6/7-bit text for keywords",
    )
    parser.add_argument(
        "--scan-packed-pattern",
        action="store_true",
        help="Scan packed 5/6/7-bit text for --pattern matches",
    )
    parser.add_argument(
        "--scan-cons",
        action="store_true",
        help="Scan cons-list strings for regex matches",
    )
    parser.add_argument(
        "--cons-pattern",
        default="",
        help="Regex pattern for cons-list string scan (defaults to --pattern)",
    )
    parser.add_argument(
        "--cons-any",
        action="store_true",
        help="Scan cons-list strings starting from every node (not just heads)",
    )
    parser.add_argument(
        "--cons-max-len",
        type=int,
        default=0,
        help="Maximum cons string length to walk (0 = no limit)",
    )
    parser.add_argument(
        "--cons-min-len",
        type=int,
        default=0,
        help="Minimum cons string length to report",
    )
    parser.add_argument(
        "--cons-limit",
        type=int,
        default=50,
        help="Maximum cons string hits to report",
    )
    parser.add_argument(
        "--arrays",
        default="",
        help="Comma-separated array ids (or ranges like 0-10) to scan",
    )
    parser.add_argument(
        "--variants",
        default="",
        help="Comma-separated variant names to scan (ex: 5bit-msb-space26)",
    )
    parser.add_argument(
        "--keywords",
        default="you are,from here,there is,history of,museum of,adventure,sequent,robber,package,pousse,race car",
        help="Comma-separated keywords for packed scans",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=2,
        help="Minimum distinct keyword hits to report an array/variant",
    )
    parser.add_argument(
        "--max-hits",
        type=int,
        default=5,
        help="Maximum keyword hits per encoding variant",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=40,
        help="Context characters to show around packed hits",
    )
    parser.add_argument(
        "--pattern-context",
        type=int,
        default=30,
        help="Context characters to show around packed pattern hits",
    )
    parser.add_argument(
        "--pattern-max-hits",
        type=int,
        default=5,
        help="Maximum pattern hits per encoding variant",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=1024,
        help="Minimum array byte size for packed scans",
    )
    parser.add_argument(
        "--min-letters",
        type=int,
        default=10,
        help="Minimum alphabetic characters in a hit snippet",
    )
    parser.add_argument(
        "--min-space-ratio",
        type=float,
        default=0.05,
        help="Minimum space ratio in a hit snippet",
    )
    parser.add_argument(
        "--max-q-ratio",
        type=float,
        default=0.2,
        help="Maximum '?' ratio in a hit snippet",
    )
    args = parser.parse_args()

    byte_pattern = re.compile(args.pattern.encode("ascii"))
    text_pattern = re.compile(args.pattern)
    keywords = [k.strip().lower() for k in args.keywords.split(",") if k.strip()]
    array_filter = parse_id_filter(args.arrays)
    variant_filter = {v.strip() for v in args.variants.split(",") if v.strip()}

    variants = [
        ("5bit-msb-space0", 5, True, " ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?'", 0, False),
        ("5bit-msb-space26", 5, True, "ABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'", 0, False),
        ("5bit-lsb-space0", 5, False, " ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?'", 0, False),
        ("5bit-lsb-space26", 5, False, "ABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'", 0, False),
        ("6bit-msb-base64", 6, True, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", 0, False),
        ("6bit-msb-space0", 6, True, " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.", 0, False),
        ("6bit-msb-space62", 6, True, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .", 0, False),
        ("6bit-lsb-base64", 6, False, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", 0, False),
        ("6bit-lsb-space0", 6, False, " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.", 0, False),
        ("6bit-lsb-space62", 6, False, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .", 0, False),
        ("7bit-msb-ascii", 7, True, None, 0, True),
        ("7bit-lsb-ascii", 7, False, None, 0, True),
    ]

    packed_hits = set()
    word_hits = set()
    for arr_id, active, data in iter_arrays(args.dump):
        _ = active  # unused for now
        packed_hits.update(scan_packed_bytes(data, byte_pattern))
        word_hits.update(scan_word_chars(data, text_pattern))

    print("packed hits:")
    for item in sorted(packed_hits):
        print(item)
    print("word hits:")
    for item in sorted(word_hits):
        print(item)

    if args.scan_cons:
        cons_pat = args.cons_pattern or args.pattern
        cons_re = re.compile(cons_pat)
        print("cons-list hits:")
        nodes, heads = build_cons_nodes(args.dump)
        hit_count = 0
        start_nodes = nodes.keys() if args.cons_any else heads
        if args.cons_any and is_plain_pattern(cons_pat):
            codes = [10 if ch == "\n" else ord(ch) for ch in cons_pat]
            for node_id, (_nxt, _mid, val) in nodes.items():
                if not codes:
                    break
                if val != codes[0]:
                    continue
                if not match_cons_literal(nodes, node_id, codes):
                    continue
                text = iter_cons_string_from(nodes, node_id, args.cons_max_len)
                if args.cons_min_len and len(text) < args.cons_min_len:
                    continue
                print(f"head {node_id} len {len(text)}")
                snippet = cons_pat
                if text:
                    snippet = cons_re.sub(lambda m: f"<<{m.group(0)}>>", text, count=1)
                print(snippet)
                hit_count += 1
                if hit_count >= args.cons_limit:
                    break
        else:
            for head, text in iter_cons_strings(nodes, start_nodes, args.cons_max_len, args.cons_min_len):
                if cons_re.search(text):
                    print(f"head {head} len {len(text)}")
                    snippet = cons_re.sub(lambda m: f"<<{m.group(0)}>>", text, count=1)
                    print(snippet)
                    hit_count += 1
                    if hit_count >= args.cons_limit:
                        break

    if args.scan_packed:
        filters = {
            "min_letters": args.min_letters,
            "min_space_ratio": args.min_space_ratio,
            "max_q_ratio": args.max_q_ratio,
        }
        print("packed text hits:")
        for arr_id, active, data in iter_arrays(args.dump):
            if array_filter and arr_id not in array_filter:
                continue
            if len(data) < args.min_bytes:
                continue
            for name, bits, msb_first, alphabet, offset, ascii_mode in variants:
                if variant_filter and name not in variant_filter:
                    continue
                char_iter = iter_decoded_chars(
                    data, bits, msb_first, alphabet=alphabet, offset=offset, ascii_mode=ascii_mode
                )
                hits = scan_stream_for_keywords(
                    char_iter, keywords, args.max_hits, args.context, filters
                )
                if len(hits) < args.min_hits:
                    continue
                print(f"array {arr_id} active={active} {name}")
                for kw, snippet in hits:
                    cleaned = snippet.replace("\\n", " ")
                    print(f"  {kw}: {cleaned}")

    if args.scan_packed_pattern:
        print("packed pattern hits:")
        for arr_id, active, data in iter_arrays(args.dump):
            if array_filter and arr_id not in array_filter:
                continue
            if len(data) < args.min_bytes:
                continue
            for name, bits, msb_first, alphabet, offset, ascii_mode in variants:
                if variant_filter and name not in variant_filter:
                    continue
                char_iter = iter_decoded_chars(
                    data, bits, msb_first, alphabet=alphabet, offset=offset, ascii_mode=ascii_mode
                )
                hits = scan_stream_for_regex(
                    char_iter, text_pattern, args.pattern_max_hits, args.pattern_context
                )
                if not hits:
                    continue
                print(f"array {arr_id} active={active} {name}")
                for match, snippet in hits:
                    cleaned = snippet.replace("\\n", " ")
                    print(f"  {match}: {cleaned}")


if __name__ == "__main__":
    main()
