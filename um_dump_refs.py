#!/usr/bin/env python3
import argparse
import struct

MAGIC = b"UMD0"


def read_u32_be(f):
    data = f.read(4)
    if len(data) != 4:
        raise EOFError
    return struct.unpack(">I", data)[0]


def read_dump(path):
    arrays = {}
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError("bad magic in dump")
        count = read_u32_be(f)
        for _ in range(count):
            arr_id = read_u32_be(f)
            active = read_u32_be(f)
            length = read_u32_be(f)
            data = f.read(length * 4)
            if len(data) != length * 4:
                raise EOFError
            if length:
                words = struct.unpack(">" + "I" * length, data)
            else:
                words = ()
            arrays[arr_id] = (active, words)
    return arrays


def build_cons_nodes(arrays):
    nodes = {}
    for arr_id, (_active, words) in arrays.items():
        if len(words) != 3:
            continue
        nxt, mid, val = words
        if val == 10 or 32 <= val <= 126:
            nodes[arr_id] = (nxt, mid, val)
    return nodes


def cons_string(nodes, head, max_len):
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
    return "".join(chars)


def is_string_obj(words, nodes):
    if len(words) < 3:
        return False
    head = words[0]
    if head not in nodes:
        return False
    if words[1] != 4:
        return False
    length = words[2]
    if length <= 0:
        return False
    if len(words) < 3 + length:
        return False
    seq = words[3 : 3 + length]
    if all(v in nodes for v in seq):
        return True
    return False


def is_string_list(words, arrays, nodes):
    if len(words) < 3:
        return False
    if words[1] != 4:
        return False
    count = words[2]
    if count < 0:
        return False
    if len(words) != 3 + count:
        return False
    for v in words[3:]:
        arr = arrays.get(v)
        if arr is None:
            return False
        if not is_string_obj(arr[1], nodes):
            return False
    return True


def array_info(arr_id, arrays, nodes, max_len, max_list):
    if arr_id not in arrays:
        return None
    active, words = arrays[arr_id]
    length = len(words)
    if arr_id in nodes:
        nxt, _mid, val = nodes[arr_id]
        ch = "\n" if val == 10 else chr(val)
        return {
            "id": arr_id,
            "active": active,
            "length": length,
            "type": "cons-node",
            "detail": f"val={val} ch={repr(ch)} next={nxt}",
        }
    if is_string_obj(words, nodes):
        head = words[0]
        strlen = words[2]
        text = cons_string(nodes, head, max_len or strlen)
        return {
            "id": arr_id,
            "active": active,
            "length": length,
            "type": "string",
            "detail": f"head={head} len={strlen} text={text!r}",
        }
    if is_string_list(words, arrays, nodes):
        count = words[2]
        texts = []
        for v in words[3 : 3 + count][:max_list]:
            head = arrays[v][1][0]
            strlen = arrays[v][1][2]
            text = cons_string(nodes, head, max_len or strlen)
            texts.append(text)
        return {
            "id": arr_id,
            "active": active,
            "length": length,
            "type": "string-list",
            "detail": f"count={count} texts={texts!r}",
        }
    if length >= 3 and words[1] == 4 and len(words) == 3 + words[2]:
        count = words[2]
        return {
            "id": arr_id,
            "active": active,
            "length": length,
            "type": "list",
            "detail": f"count={count} values={words[3:3 + min(count, max_list)]}",
        }
    return {
        "id": arr_id,
        "active": active,
        "length": length,
        "type": "raw",
        "detail": f"values={words[:min(length, max_list)]}",
    }


def find_string_objects(arrays, nodes, substring, ignore_case, max_len):
    matches = []
    for arr_id, (_active, words) in arrays.items():
        if not is_string_obj(words, nodes):
            continue
        head = words[0]
        strlen = words[2]
        text = cons_string(nodes, head, max_len or strlen)
        hay = text.lower() if ignore_case else text
        needle = substring.lower() if ignore_case else substring
        if needle in hay:
            matches.append((arr_id, head, strlen, text))
    return matches


def find_refs(arrays, targets):
    refs = {t: [] for t in targets}
    target_set = set(targets)
    for arr_id, (_active, words) in arrays.items():
        if not words:
            continue
        for v in words:
            if v in target_set:
                refs[v].append(arr_id)
        # no early exit; some arrays may reference multiple targets
    return refs


def trace_refs(arrays, nodes, start_ids, depth, max_len, max_list, max_refs):
    current = set(start_ids)
    visited = set(start_ids)
    for level in range(1, depth + 1):
        refs = find_refs(arrays, current)
        next_ids = set()
        for target in sorted(current):
            arr_ids = refs.get(target, [])
            if not arr_ids:
                continue
            print(f"refs for {target} (depth {level}):")
            for arr_id in arr_ids[:max_refs]:
                info = array_info(arr_id, arrays, nodes, max_len, max_list)
                if info is None:
                    continue
                print(
                    f"  array {info['id']} active={info['active']} len={info['length']} "
                    f"type={info['type']} {info['detail']}"
                )
                next_ids.add(arr_id)
        current = next_ids - visited
        visited |= next_ids
        if not current:
            break


def main():
    parser = argparse.ArgumentParser(description="Inspect references in UM dump.")
    parser.add_argument("dump", help="Path to UMD0 dump")
    parser.add_argument("--find-string", default="", help="Substring to search in string objects")
    parser.add_argument("--ignore-case", action="store_true", help="Case-insensitive string search")
    parser.add_argument("--trace-string", default="", help="Trace refs from string object match")
    parser.add_argument("--trace-id", type=int, default=None, help="Trace refs from array id")
    parser.add_argument("--depth", type=int, default=2, help="Trace depth")
    parser.add_argument("--max-len", type=int, default=200, help="Max string length to decode")
    parser.add_argument("--max-list", type=int, default=10, help="Max list entries to show")
    parser.add_argument("--max-refs", type=int, default=20, help="Max refs to show per target")
    args = parser.parse_args()

    arrays = read_dump(args.dump)
    nodes = build_cons_nodes(arrays)

    if args.find_string:
        matches = find_string_objects(
            arrays, nodes, args.find_string, args.ignore_case, args.max_len
        )
        for arr_id, head, strlen, text in matches:
            print(f"string array {arr_id} head {head} len {strlen}: {text!r}")

    if args.trace_string:
        matches = find_string_objects(
            arrays, nodes, args.trace_string, args.ignore_case, args.max_len
        )
        for arr_id, head, strlen, text in matches:
            print(f"trace for string array {arr_id} head {head} len {strlen}: {text!r}")
            trace_refs(
                arrays, nodes, [arr_id, head], args.depth, args.max_len, args.max_list, args.max_refs
            )

    if args.trace_id is not None:
        trace_refs(
            arrays, nodes, [args.trace_id], args.depth, args.max_len, args.max_list, args.max_refs
        )


if __name__ == "__main__":
    main()
