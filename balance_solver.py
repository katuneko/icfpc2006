#!/usr/bin/env python3
import itertools
from typing import Iterable, List, Tuple, Optional


# Balance interpreter and puzzle helpers.
# Register order: sR0, sR1, sR2, sR3, dR0, dR1


def _imm5(byte: int) -> int:
    value = byte & 0x1F
    return value - 0x20 if value & 0x10 else value


def enc_science(imm: int) -> int:
    return (0b000 << 5) | (imm & 0x1F)


def enc_math(d: int, s1: int, s2: int) -> int:
    return (0b001 << 5) | ((d & 1) << 4) | ((s1 & 3) << 2) | (s2 & 3)


def enc_logic(d: int, s1: int, s2: int) -> int:
    return (0b010 << 5) | ((d & 1) << 4) | ((s1 & 3) << 2) | (s2 & 3)


def enc_physics(imm: int) -> int:
    return (0b011 << 5) | (imm & 0x1F)


def decode_program(hex_str: str) -> List[int]:
    hex_str = hex_str.strip()
    if len(hex_str) % 2:
        raise ValueError("hex string length must be even")
    return [int(hex_str[i:i + 2], 16) for i in range(0, len(hex_str), 2)]


def encode_program(bytes_list: Iterable[int]) -> str:
    return "".join(f"{b:02x}" for b in bytes_list)


def _physics_step(state: Tuple[int, int, int, int, int, int], imm: int) -> Tuple[int, int, int, int, int, int]:
    sr0, sr1, sr2, sr3, dr0, dr1 = state
    sr0_new = (sr0 + imm) & 0xFF
    mask = imm & 0x1F
    regs = [("d", 1), ("d", 0), ("s", 3), ("s", 2), ("s", 1)]
    c_regs = [regs[i] for i in range(5) if (mask >> i) & 1]

    if not c_regs:
        return (sr0_new, sr1, sr2, sr3, dr0, dr1)

    old_sr = [sr0_new, sr1, sr2, sr3]
    old_dr = [dr0, dr1]

    def get(reg):
        typ, idx = reg
        return old_dr[idx] if typ == "d" else old_sr[idx]

    cs = [sr0_new] + [get(r) for r in c_regs]
    new_sr = [sr0_new, sr1, sr2, sr3]
    new_dr = [dr0, dr1]
    for reg, val in zip(c_regs, cs[:-1]):
        typ, idx = reg
        if typ == "d":
            new_dr[idx] = val & 0xFF
        else:
            new_sr[idx] = val & 0xFF
    new_sr[0] = cs[-1] & 0xFF
    return (new_sr[0], new_sr[1], new_sr[2], new_sr[3], new_dr[0], new_dr[1])


def search_physics_sequence(
    target: Tuple[int, int, int, int, int, int],
    max_len: int = 4,
    start: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    allowed_imms: Optional[Iterable[int]] = None,
) -> Optional[List[int]]:
    if allowed_imms is None:
        allowed_imms = range(32)
    start_state = start
    if start_state == target:
        return []
    from collections import deque

    queue = deque([(start_state, [])])
    seen = {start_state}
    for _ in range(max_len):
        for _ in range(len(queue)):
            state, path = queue.popleft()
            for imm_raw in allowed_imms:
                imm = _imm5(imm_raw)
                new_state = _physics_step(state, imm)
                if new_state in seen:
                    continue
                new_path = path + [imm]
                if new_state == target:
                    return new_path
                seen.add(new_state)
                queue.append((new_state, new_path))
    return None


class BalanceMachine:
    def __init__(self, code: List[int], mem: List[int], sR: List[int], dR: List[int]):
        if not code:
            raise ValueError("code must not be empty")
        self.code = code[:]
        self.mem = mem[:]
        self.sR = sR[:]
        self.dR = dR[:]
        self.ip = 0
        self.ispeed = 1

    def step(self) -> Optional[bool]:
        byte = self.code[self.ip]
        op = (byte >> 5) & 0x7

        if op == 0b001:  # MATH
            d = (byte >> 4) & 1
            s1 = (byte >> 2) & 3
            s2 = byte & 3
            s1p = (s1 + 1) & 3
            s2p = (s2 + 1) & 3
            addr_d = self.dR[d]
            addr_d2 = self.dR[(d + 1) & 1]
            a = self.mem[self.sR[s1]]
            b = self.mem[self.sR[s2]]
            c = self.mem[self.sR[s1p]]
            dval = self.mem[self.sR[s2p]]
            self.mem[addr_d2] = (c - dval) & 0xFF
            self.mem[addr_d] = (a + b) & 0xFF
        elif op == 0b010:  # LOGIC
            d = (byte >> 4) & 1
            s1 = (byte >> 2) & 3
            s2 = byte & 3
            s1p = (s1 + 1) & 3
            s2p = (s2 + 1) & 3
            addr_d = self.dR[d]
            addr_d2 = self.dR[(d + 1) & 1]
            a = self.mem[self.sR[s1]]
            b = self.mem[self.sR[s2]]
            c = self.mem[self.sR[s1p]]
            dval = self.mem[self.sR[s2p]]
            self.mem[addr_d2] = (c ^ dval) & 0xFF
            self.mem[addr_d] = (a & b) & 0xFF
        elif op == 0b000:  # SCIENCE
            imm = _imm5(byte)
            if self.mem[self.sR[0]] != 0:
                self.ispeed = imm
            if self.ispeed == 0:
                return True
        elif op == 0b011:  # PHYSICS
            imm = _imm5(byte)
            self.sR[0], self.sR[1], self.sR[2], self.sR[3], self.dR[0], self.dR[1] = _physics_step(
                (self.sR[0], self.sR[1], self.sR[2], self.sR[3], self.dR[0], self.dR[1]), imm
            )
        else:
            return False

        self.ip = ((self.ip + self.ispeed) & 0xFFFFFFFF) % len(self.code)
        return None

    def run(self, max_steps: int = 100000) -> Tuple[bool, int]:
        for step in range(max_steps):
            halted = self.step()
            if halted is not None:
                return halted, step + 1
        return False, max_steps


def puzzle_state(name: str, params: Tuple[int, ...]) -> Tuple[List[int], List[int], List[int]]:
    if name == "copymem":
        (a,) = params
        mem = [0] * 256
        mem[0] = a
        mem[1] = 1
        sR = [0, 0, 0, 0]
        dR = [0, 0]
        return mem, sR, dR
    if name == "copyreg":
        (a,) = params
        mem = [0] * 256
        for i, v in enumerate([1, 2, 4, 8, 16, 32, 64, 128]):
            mem[i] = v
        sR = [a & 0xFF, 0, 1, 2]
        dR = [3, 4]
        return mem, sR, dR
    if name == "multmem":
        (a, b) = params
        mem = [0] * 256
        mem[0] = a
        mem[1] = b
        sR = [0, 1, 2, 3]
        dR = [4, 5]
        return mem, sR, dR
    if name == "fillmem":
        a, i, j = params
        mem = [0] * 256
        mem[0] = a
        mem[1] = i
        mem[2] = j
        mem[4:8] = [1, 2, 4, 8]
        sR = [0, 1, 2, 3]
        dR = [4, 5]
        return mem, sR, dR
    if name == "clearreg":
        mem = list(range(256))
        sR = [0, 1, 2, 3]
        dR = [4, 5]
        return mem, sR, dR
    raise ValueError(f"unknown puzzle {name}")


def check_puzzle(name: str, machine: BalanceMachine, params: Tuple[int, ...]) -> bool:
    if name == "copymem":
        (a,) = params
        regs = machine.sR + machine.dR
        return any(r == a for r in regs)
    if name == "copyreg":
        (a,) = params
        return a in machine.mem
    if name == "multmem":
        a, b = params
        return machine.mem[2] == (a * b) & 0xFF
    if name == "fillmem":
        a, i, j = params
        for idx in range(8, i):
            if machine.mem[idx] != 0:
                return False
        for idx in range(i, j):
            if machine.mem[idx] != a:
                return False
        for idx in range(j, 256):
            if machine.mem[idx] != 0:
                return False
        return True
    if name == "clearreg":
        return machine.sR == [0, 0, 0, 0] and machine.dR == [0, 0]
    return False


def verify_program(name: str, program: List[int], samples: Optional[int] = None) -> bool:
    if name == "copymem":
        values = range(1, 256)
        if samples:
            values = list(range(1, 256))
            values = values[:samples]
        params_list = [(a,) for a in values]
    elif name == "copyreg":
        values = range(1, 256)
        if samples:
            values = list(range(1, 256))
            values = values[:samples]
        params_list = [(a,) for a in values]
    elif name == "multmem":
        params_list = [(a, b) for a in range(1, 256) for b in range(1, 256)]
        if samples:
            params_list = params_list[:samples]
    elif name == "fillmem":
        params_list = [(a, i, j) for a in range(1, 256) for i in range(8, 255) for j in range(i + 1, 256)]
        if samples:
            params_list = params_list[:samples]
    elif name == "clearreg":
        params_list = [tuple()]
    else:
        raise ValueError(f"unknown puzzle {name}")

    for params in params_list:
        mem, sR, dR = puzzle_state(name, params)
        machine = BalanceMachine(program, mem, sR, dR)
        halted, _ = machine.run()
        if not halted:
            return False
        if not check_puzzle(name, machine, params):
            return False
    return True


def clearreg_candidate() -> List[int]:
    # Found by meet-in-the-middle over PHYSICS to zero registers, plus MATH to set M[0] nonzero.
    return [
        enc_physics(-5),
        enc_physics(-1),
        enc_math(1, 0, 1),
        enc_physics(-2),
        enc_physics(-4),
        enc_physics(-3),
        enc_physics(-5),
        enc_physics(5),
        enc_science(0),
    ]


def _apply_physics_sequence(
    state: Tuple[int, int, int, int, int, int],
    seq: Iterable[int],
) -> Tuple[int, int, int, int, int, int]:
    for imm in seq:
        state = _physics_step(state, imm)
    return state


def search_increment_sequence(
    target: str,
    max_len: int = 6,
    allowed_imms: Optional[Iterable[int]] = None,
    samples: Tuple[int, ...] = (0, 1, 2),
) -> Optional[List[int]]:
    if allowed_imms is None:
        allowed_imms = range(32)
    target_map = {
        "sr0": 0,
        "sr1": 1,
        "sr2": 2,
        "sr3": 3,
        "dr0": 4,
        "dr1": 5,
    }
    if target not in target_map:
        raise ValueError(f"unknown target {target}")
    idx = target_map[target]

    def make_state(value: int) -> Tuple[int, int, int, int, int, int]:
        base = [0, 1, 0, 0, 0, 0]
        base[idx] = value & 0xFF
        return tuple(base)  # type: ignore[return-value]

    start_states = tuple(make_state(v) for v in samples)

    def is_target(states: Tuple[Tuple[int, int, int, int, int, int], ...]) -> bool:
        for v, state in zip(samples, states):
            if state[idx] != ((v + 1) & 0xFF):
                return False
            for reg_idx, reg_val in enumerate(state):
                if reg_idx == idx:
                    continue
                base = 1 if reg_idx == 1 else 0
                if reg_val != base:
                    return False
        return True

    from collections import deque

    queue = deque([(start_states, [])])
    seen = {start_states}
    for _ in range(max_len):
        for _ in range(len(queue)):
            states, path = queue.popleft()
            if is_target(states):
                return path
            for imm_raw in allowed_imms:
                imm = _imm5(imm_raw)
                next_states = tuple(_apply_physics_sequence(state, [imm]) for state in states)
                if next_states in seen:
                    continue
                seen.add(next_states)
                queue.append((next_states, path + [imm]))
    for states, path in queue:
        if is_target(states):
            return path
    return None


def build_sr0_samples(
    sr0_values: Iterable[int],
    reg_sets: Iterable[Tuple[int, int, int, int, int]],
) -> Tuple[Tuple[int, int, int, int, int, int], ...]:
    samples: List[Tuple[int, int, int, int, int, int]] = []
    for sr0 in sr0_values:
        for sr1, sr2, sr3, dr0, dr1 in reg_sets:
            samples.append((sr0 & 0xFF, sr1, sr2, sr3, dr0, dr1))
    return tuple(samples)


def search_sr0_delta_sequence(
    delta: int,
    max_len: int = 8,
    allowed_imms: Optional[Iterable[int]] = None,
    samples: Optional[Tuple[Tuple[int, int, int, int, int, int], ...]] = None,
    sr0_values: Tuple[int, ...] = (0, 1, 2),
    reg_sets: Tuple[Tuple[int, int, int, int, int], ...] = (
        (1, 0, 0, 0, 0),
        (7, 5, 11, 13, 17),
    ),
) -> Optional[List[int]]:
    if allowed_imms is None:
        allowed_imms = range(32)
    if samples is None:
        samples = build_sr0_samples(sr0_values, reg_sets)
    start_states = samples

    def is_target(
        states: Tuple[Tuple[int, int, int, int, int, int], ...]
    ) -> bool:
        for state, start in zip(states, start_states):
            if state[0] != ((start[0] + delta) & 0xFF):
                return False
            if state[1:] != start[1:]:
                return False
        return True

    from collections import deque

    queue = deque([(start_states, [])])
    seen = {start_states}
    for _ in range(max_len):
        for _ in range(len(queue)):
            states, path = queue.popleft()
            if is_target(states):
                return path
            for imm_raw in allowed_imms:
                imm = _imm5(imm_raw)
                next_states = tuple(_apply_physics_sequence(state, [imm]) for state in states)
                if next_states in seen:
                    continue
                seen.add(next_states)
                queue.append((next_states, path + [imm]))
    for states, path in queue:
        if is_target(states):
            return path
    return None


def search_copyreg_a_lt_8_loop(max_steps: int = 200) -> None:
    setup = [
        enc_logic(0, 0, 0),  # mem[3] = M[a], mem[4] = 0
        enc_physics(-16),
        enc_physics(1),
        enc_physics(4),
        enc_physics(7),
        enc_physics(1),  # sR0=9, sR2=1, sR3=3, dR0=1, dR1=9
    ]

    math_ops = [enc_math(d, s1, s2) for d in (0, 1) for s1 in range(4) for s2 in range(4)]
    logic_ops = [enc_logic(d, s1, s2) for d in (0, 1) for s1 in range(4) for s2 in range(4)]
    science_ops = [enc_science(imm) for imm in (0, 1, -1)]

    def works(program: List[int]) -> bool:
        for a in range(1, 8):
            mem, sR, dR = puzzle_state("copyreg", (a,))
            machine = BalanceMachine(program, mem, sR, dR)
            halted, _ = machine.run(max_steps=max_steps)
            if not halted:
                return False
            if not check_puzzle("copyreg", machine, (a,)):
                return False
        return True

    # Try 2-instruction loops: (MATH|LOGIC) + SCIENCE
    for op in math_ops + logic_ops:
        for sci in science_ops:
            program = setup + [op, sci]
            if works(program):
                print("found loop len2:", encode_program(program))
                return

    # Try 3-instruction loops: MATH + LOGIC + SCIENCE and LOGIC + MATH + SCIENCE
    for op1 in math_ops:
        for op2 in logic_ops:
            for sci in science_ops:
                program = setup + [op1, op2, sci]
                if works(program):
                    print("found loop len3 (MATH+LOGIC):", encode_program(program))
                    return
    for op1 in logic_ops:
        for op2 in math_ops:
            for sci in science_ops:
                program = setup + [op1, op2, sci]
                if works(program):
                    print("found loop len3 (LOGIC+MATH):", encode_program(program))
                    return

    print("no loop found for a<8 within search space")


def search_copyreg_a_lt_8_templates(
    max_steps: int = 200,
    templates: Optional[List[Tuple[str, List[List[int]]]]] = None,
    setups: Optional[List[Tuple[str, List[int]]]] = None,
    math_sources: Tuple[int, ...] = (0, 1),
    logic_sources: Tuple[int, ...] = (0, 1),
    physics_imms: Tuple[int, ...] = (-16, -4, -2, -1, 1, 2, 4, 7),
    science_imms: Tuple[int, ...] = (0, 1, -1, 2, -2),
    max_checks: Optional[int] = None,
    max_seconds: Optional[float] = None,
    random_samples: Optional[int] = None,
    random_seed: Optional[int] = 0,
) -> None:
    import time
    import random

    if setups is None:
        setups = [
            ("const1", [-8, 10, 8, -2]),
            ("const2", [9, 1, -9, -15]),
            ("const3", [-14, 3, 14, -13]),
            ("const4", [14, 3, -14, -13]),
            ("sr0_0_sr1_a", [-8, 8, 8, -8, 8]),
        ]

    setup_codes = [(name, [enc_physics(imm) for imm in seq]) for name, seq in setups]

    math_ops = [
        enc_math(d, s1, s2)
        for d in (0, 1)
        for s1 in math_sources
        for s2 in math_sources
    ]
    logic_ops = [
        enc_logic(d, s1, s2)
        for d in (0, 1)
        for s1 in logic_sources
        for s2 in logic_sources
    ]
    physics_ops = [enc_physics(imm) for imm in physics_imms]
    science_ops = [enc_science(imm) for imm in science_imms]
    ml_ops = math_ops + logic_ops

    if templates is None:
        templates = [
            ("ml+sci", [ml_ops, science_ops]),
            ("ml+ml+sci", [ml_ops, ml_ops, science_ops]),
            ("ml+phy+sci", [ml_ops, physics_ops, science_ops]),
            ("phy+ml+sci", [physics_ops, ml_ops, science_ops]),
            ("phy+phy+sci", [physics_ops, physics_ops, science_ops]),
        ]

    def works(program: List[int]) -> bool:
        for a in range(1, 8):
            mem, sR, dR = puzzle_state("copyreg", (a,))
            machine = BalanceMachine(program, mem, sR, dR)
            halted, _ = machine.run(max_steps=max_steps)
            if not halted:
                return False
            if not check_puzzle("copyreg", machine, (a,)):
                return False
        return True

    start_time = time.time()
    checks = 0

    def should_stop() -> bool:
        if max_checks is not None and checks >= max_checks:
            return True
        if max_seconds is not None and (time.time() - start_time) >= max_seconds:
            return True
        return False

    for setup_name, setup_code in setup_codes:
        for label, pools in templates:
            if random_samples is not None:
                rng = random.Random(random_seed)
                for _ in range(random_samples):
                    seq = [rng.choice(pool) for pool in pools]
                    program = setup_code + list(seq)
                    checks += 1
                    if works(program):
                        print(
                            "found template:",
                            setup_name,
                            label,
                            encode_program(program),
                        )
                        return
                    if should_stop():
                        print("stopped after", checks, "checks (time/limit)")
                        return
            else:
                for seq in itertools.product(*pools):
                    program = setup_code + list(seq)
                    checks += 1
                    if works(program):
                        print(
                            "found template:",
                            setup_name,
                            label,
                            encode_program(program),
                        )
                        return
                    if should_stop():
                        print("stopped after", checks, "checks (time/limit)")
                        return

    print("no template program found for a<8 within search space")


def _run_straightline(
    program: List[int],
    mem: List[int],
    sR: List[int],
    dR: List[int],
) -> Tuple[bool, Optional[int], Optional[int]]:
    dest_addr = None
    sr0_at_science = None
    for byte in program:
        op = (byte >> 5) & 0x7
        if op == 0b001:  # MATH
            d = (byte >> 4) & 1
            s1 = (byte >> 2) & 3
            s2 = byte & 3
            s1p = (s1 + 1) & 3
            s2p = (s2 + 1) & 3
            addr_d = dR[d]
            addr_d2 = dR[(d + 1) & 1]
            a = mem[sR[s1]]
            b = mem[sR[s2]]
            c = mem[sR[s1p]]
            dval = mem[sR[s2p]]
            mem[addr_d2] = (c - dval) & 0xFF
            mem[addr_d] = (a + b) & 0xFF
        elif op == 0b010:  # LOGIC
            d = (byte >> 4) & 1
            s1 = (byte >> 2) & 3
            s2 = byte & 3
            s1p = (s1 + 1) & 3
            s2p = (s2 + 1) & 3
            addr_d = dR[d]
            addr_d2 = dR[(d + 1) & 1]
            a = mem[sR[s1]]
            b = mem[sR[s2]]
            c = mem[sR[s1p]]
            dval = mem[sR[s2p]]
            mem[addr_d2] = (c ^ dval) & 0xFF
            mem[addr_d] = (a & b) & 0xFF
            dest_addr = addr_d
        elif op == 0b000:  # SCIENCE
            sr0_at_science = sR[0]
            imm = _imm5(byte)
            if mem[sR[0]] != 0 and imm == 0:
                return True, dest_addr, sr0_at_science
        elif op == 0b011:  # PHYSICS
            imm = _imm5(byte)
            sR[0], sR[1], sR[2], sR[3], dR[0], dR[1] = _physics_step(
                (sR[0], sR[1], sR[2], sR[3], dR[0], dR[1]), imm
            )
        else:
            return False, dest_addr, sr0_at_science
    return False, dest_addr, sr0_at_science


def search_copyreg_cmp_science(
    max_physics_len: int = 2,
    physics_imms: Tuple[int, ...] = (-16, -8, -4, -2, -1, 1, 2, 4, 7, 8, 15),
    s1_choices: Tuple[int, ...] = (1,),
    s2_choices: Tuple[int, ...] = (0, 1, 2, 3),
    prefixes: Optional[List[Tuple[str, List[int]]]] = None,
    require_dest_match: bool = True,
    interesting_sizes: Tuple[int, ...] = (1, 2, 3),
    max_results: int = 20,
) -> None:
    base_setup = [-8, 8, 8, -8, 8]
    if prefixes is None:
        prefixes = [
            ("base", []),
            ("base+2,1", [2, 1]),
            ("base+7,15", [7, 15]),
            ("base+7,15+2,1", [7, 15, 2, 1]),
            ("base+2,1+7,15", [2, 1, 7, 15]),
        ]

    prefix_codes = [
        (label, [enc_physics(imm) for imm in base_setup + seq]) for label, seq in prefixes
    ]
    logic_ops = [
        enc_logic(d, s1, s2)
        for d in (0, 1)
        for s1 in s1_choices
        for s2 in s2_choices
    ]

    physics_seqs = [[]]
    for length in range(1, max_physics_len + 1):
        for seq in itertools.product(physics_imms, repeat=length):
            physics_seqs.append([enc_physics(imm) for imm in seq])

    results = 0
    for prefix_label, prefix in prefix_codes:
        for logic_op in logic_ops:
            for physics_seq in physics_seqs:
                program = prefix + [logic_op] + physics_seq + [enc_science(0)]
                halts = []
                dest_ok = True
                for a in range(1, 8):
                    mem, sR, dR = puzzle_state("copyreg", (a,))
                    halted, dest_addr, sr0_at_science = _run_straightline(
                        program, mem, sR, dR
                    )
                    if require_dest_match:
                        if dest_addr is None or sr0_at_science != dest_addr:
                            dest_ok = False
                            break
                    if halted:
                        halts.append(a)
                if not dest_ok:
                    continue
                if not halts or len(halts) == 7:
                    continue
                if interesting_sizes and len(halts) not in interesting_sizes:
                    continue
                print(
                    "subset",
                    halts,
                    "prefix",
                    prefix_label,
                    "logic",
                    hex(logic_op),
                    "phy",
                    [hex(b) for b in physics_seq],
                )
                results += 1
                if results >= max_results:
                    return

def build_sr1_dr1_samples(
    sr0: int,
    sr2: int,
    sr3: int,
    dr0: int,
    sr1_values: Iterable[int],
    dr1_values: Iterable[int],
) -> Tuple[Tuple[int, int, int, int, int, int], ...]:
    samples: List[Tuple[int, int, int, int, int, int]] = []
    for sr1 in sr1_values:
        for dr1 in dr1_values:
            samples.append(
                (sr0 & 0xFF, sr1 & 0xFF, sr2 & 0xFF, sr3 & 0xFF, dr0 & 0xFF, dr1 & 0xFF)
            )
    return tuple(samples)


def search_physics_move_register(
    source_idx: int,
    target_idx: int,
    max_len: int = 5,
    allowed_imms: Optional[Iterable[int]] = None,
    samples: Optional[Tuple[Tuple[int, int, int, int, int, int], ...]] = None,
    preserve_idxs: Tuple[int, ...] = (1,),
) -> Optional[List[int]]:
    if allowed_imms is None:
        allowed_imms = range(32)
    if samples is None:
        raise ValueError("samples must be provided")
    start_states = samples

    def is_target(
        states: Tuple[Tuple[int, int, int, int, int, int], ...]
    ) -> bool:
        for state, start in zip(states, start_states):
            if state[target_idx] != start[source_idx]:
                return False
            for idx in preserve_idxs:
                if state[idx] != start[idx]:
                    return False
        return True

    from collections import deque

    queue = deque([(start_states, [])])
    seen = {start_states}
    for _ in range(max_len):
        for _ in range(len(queue)):
            states, path = queue.popleft()
            if is_target(states):
                return path
            for imm_raw in allowed_imms:
                imm = _imm5(imm_raw)
                next_states = tuple(_apply_physics_sequence(state, [imm]) for state in states)
                if next_states in seen:
                    continue
                seen.add(next_states)
                queue.append((next_states, path + [imm]))
    for states, path in queue:
        if is_target(states):
            return path
    return None


def search_copyreg_dr1_to_sr(
    max_len: int = 4,
    allowed_imms: Optional[Iterable[int]] = None,
    sr1_values: Tuple[int, ...] = (1, 2, 3, 7),
    dr1_values: Tuple[int, ...] = (0, 1, 2, 4, 7, 15, 31),
) -> None:
    setups = {
        "const1": (1, 2, 10, 1),
        "const2": (2, 4, 3, 2),
        "const3": (3, 2, 4, 3),
        "const4": (4, 2, 3, 4),
    }
    if allowed_imms is None:
        allowed_imms = (-16, -8, -4, -2, -1, 1, 2, 4, 7, 8, 15)

    for name, (sr0, sr2, sr3, dr0) in setups.items():
        samples = build_sr1_dr1_samples(sr0, sr2, sr3, dr0, sr1_values, dr1_values)
        seq_sr2 = search_physics_move_register(
            5,
            2,
            max_len=max_len,
            allowed_imms=allowed_imms,
            samples=samples,
            preserve_idxs=(0, 1, 4),
        )
        seq_sr3 = search_physics_move_register(
            5,
            3,
            max_len=max_len,
            allowed_imms=allowed_imms,
            samples=samples,
            preserve_idxs=(0, 1, 4),
        )
        print(name, "dR1->sR2:", seq_sr2, "dR1->sR3:", seq_sr3)

def main() -> None:
    prog = clearreg_candidate()
    print("clearreg candidate:", encode_program(prog))
    ok = verify_program("clearreg", prog)
    print("clearreg verify:", ok)


if __name__ == "__main__":
    main()
