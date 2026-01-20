use std::collections::VecDeque;
use std::env;
use std::fs;
use std::io::{self, IsTerminal, Read, Write};

fn read_program(path: &str) -> Result<Vec<u32>, String> {
    let data = fs::read(path).map_err(|e| format!("read error: {}", e))?;
    if data.len() % 4 != 0 {
        return Err(format!(
            "program size is not a multiple of 4 bytes: {}",
            data.len()
        ));
    }
    let mut words = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let word = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        words.push(word);
    }
    Ok(words)
}

fn write_u32_be<W: Write>(out: &mut W, val: u32) -> Result<(), String> {
    out.write_all(&val.to_be_bytes())
        .map_err(|e| format!("dump write error: {}", e))
}

fn dump_arrays(path: &str, arrays: &[Vec<u32>], active: &[u8]) -> Result<(), String> {
    let mut file = fs::File::create(path).map_err(|e| format!("dump open error: {}", e))?;
    file.write_all(b"UMD0")
        .map_err(|e| format!("dump write error: {}", e))?;
    write_u32_be(&mut file, arrays.len() as u32)?;
    for (id, arr) in arrays.iter().enumerate() {
        let is_active = active.get(id).copied().unwrap_or(0);
        write_u32_be(&mut file, id as u32)?;
        write_u32_be(&mut file, is_active as u32)?;
        write_u32_be(&mut file, arr.len() as u32)?;
        for &word in arr {
            write_u32_be(&mut file, word)?;
        }
    }
    Ok(())
}

fn maybe_dump(
    dump_path: &Option<String>,
    dumped: &mut bool,
    arrays: &[Vec<u32>],
    active: &[u8],
) -> Result<(), String> {
    if *dumped {
        return Ok(());
    }
    if let Some(path) = dump_path.as_deref() {
        dump_arrays(path, arrays, active)?;
    }
    *dumped = true;
    Ok(())
}

fn line_has_publication(line: &[u8]) -> bool {
    let text = match std::str::from_utf8(line) {
        Ok(s) => s,
        Err(_) => return false,
    };
    if !text.contains('=') || !text.contains('@') || !text.contains('|') {
        return false;
    }
    for (idx, ch) in text.char_indices() {
        if ch != '|' {
            continue;
        }
        let start = idx + 1;
        let bytes = text.as_bytes();
        let mut end = start;
        while end < bytes.len() && bytes[end].is_ascii_hexdigit() {
            end += 1;
        }
        let len = end - start;
        if len >= 31 && len <= 32 {
            return true;
        }
    }
    false
}

fn dump_output_trace(
    path: &str,
    line: &[u8],
    trace: &VecDeque<(usize, usize, u8, [u32; 8])>,
) -> Result<(), String> {
    let mut file = fs::File::create(path).map_err(|e| format!("trace open error: {}", e))?;
    let line_text = String::from_utf8_lossy(line);
    writeln!(&mut file, "publication line: {}", line_text)
        .map_err(|e| format!("trace write error: {}", e))?;
    for (prog_id, pc, byte, regs) in trace {
        writeln!(
            &mut file,
            "{:08x} {:08x} {:02x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}",
            prog_id,
            pc,
            byte,
            regs[0],
            regs[1],
            regs[2],
            regs[3],
            regs[4],
            regs[5],
            regs[6],
            regs[7],
        )
        .map_err(|e| format!("trace write error: {}", e))?;
    }
    Ok(())
}

struct ArrayWatch {
    id: usize,
    idx: usize,
    log: fs::File,
}

fn run<R: Read, W: Write>(
    program: Vec<u32>,
    input: &mut R,
    output: &mut W,
    step_limit: Option<u64>,
    output_is_tty: bool,
    dump_path: &Option<String>,
    dump_on_pub: &Option<String>,
    trace_on_pub: &Option<String>,
    dump_on_substr: &Option<(Vec<u8>, String)>,
    trace_on_substr: &Option<(Vec<u8>, String)>,
    output_on_substr: &Option<Vec<u8>>,
    jump_log: &mut Option<fs::File>,
    jump_log_after: &mut Option<(u64, fs::File)>,
    log_substr_step: &mut Option<(Vec<u8>, fs::File)>,
    trace_pc: &mut Option<(usize, fs::File)>,
    array_watch: &mut Vec<ArrayWatch>,
    array_patches: &mut Vec<(usize, usize, u32, bool)>,
    patches: &[(usize, u32)],
    patch_after_substr: &mut Vec<(Vec<u8>, usize, u32, bool)>,
    pc_patches: &mut Vec<(usize, u32, bool)>,
    override_on_substr: &mut Option<(Vec<u8>, usize, u32, bool)>,
    override_on_blank: &mut Option<(usize, u32, bool)>,
    oob_zero: bool,
) -> Result<(), String> {
    let mut regs = [0u32; 8];
    let mut arrays: Vec<Vec<u32>> = Vec::new();
    arrays.push(program);
    let mut active: Vec<u8> = vec![1];
    let mut free_ids: Vec<usize> = Vec::new();
    let mut pc: usize = 0;
    let mut dumped = false;

    let mut prog_len = arrays[0].len();
    let mut prog_ptr = arrays[0].as_ptr();

    let mut input_buf = [0u8; 4096];
    let mut input_pos: usize = 0;
    let mut input_len: usize = 0;
    let mut last_input_was_newline = false;

    let mut line_buf: Vec<u8> = Vec::with_capacity(256);
    let mut pub_dumped = false;
    let mut substr_dumped = false;
    let mut pub_traced = false;
    let mut substr_traced = false;
    let mut step_count: u64 = 0;
    let mut output_trace: VecDeque<(usize, usize, u8, [u32; 8])> =
        VecDeque::with_capacity(4096);
    let mut current_prog_id: usize = 0;

    if !patches.is_empty() {
        let prog = &mut arrays[0];
        for &(addr, value) in patches {
            if addr < prog.len() {
                prog[addr] = value;
            }
        }
        prog_len = arrays[0].len();
        prog_ptr = arrays[0].as_ptr();
    }

    macro_rules! exec_loop {
        ($step_check:block) => {{
            loop {
                if pc >= prog_len {
                    return Err(format!("execution finger out of bounds: {}", pc));
                }
                let cur_pc = pc;
                let mut override_inst: Option<u32> = None;
                if let Some((_, addr, value, armed)) = override_on_substr.as_mut() {
                    if *armed && *addr == cur_pc {
                        override_inst = Some(*value);
                    }
                }
                if let Some((addr, value, armed)) = override_on_blank.as_mut() {
                    if *armed && *addr == cur_pc {
                        override_inst = Some(*value);
                    }
                }
                if !pc_patches.is_empty() {
                    for (addr, value, applied) in pc_patches.iter_mut() {
                        if !*applied && *addr == cur_pc && *addr < arrays[0].len() {
                            arrays[0][*addr] = *value;
                            prog_ptr = arrays[0].as_ptr();
                            prog_len = arrays[0].len();
                            *applied = true;
                        }
                    }
                }
                if !array_patches.is_empty() {
                    for (arr_id, idx, value, applied) in array_patches.iter_mut() {
                        if !*applied
                            && *arr_id < arrays.len()
                            && active.get(*arr_id).copied().unwrap_or(0) != 0
                            && *idx < arrays[*arr_id].len()
                        {
                            arrays[*arr_id][*idx] = *value;
                            if *arr_id == 0 {
                                prog_ptr = arrays[0].as_ptr();
                                prog_len = arrays[0].len();
                            }
                            *applied = true;
                        }
                    }
                }
                // Safety: pc is bounds-checked against prog_len above.
                let inst = match override_inst {
                    Some(v) => v,
                    None => unsafe { *prog_ptr.add(cur_pc) },
                };
                if let Some((addr, log)) = trace_pc.as_mut() {
                    if *addr == cur_pc {
                        writeln!(
                            log,
                            "{:016x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x}",
                            step_count,
                            cur_pc,
                            inst,
                            regs[0],
                            regs[1],
                            regs[2],
                            regs[3],
                            regs[4],
                            regs[5],
                            regs[6],
                            regs[7],
                        )
                        .map_err(|e| format!("trace pc error: {}", e))?;
                    }
                }
                pc += 1;

                let op = inst >> 28;
                if op == 13 {
                    let a = ((inst >> 25) & 7) as usize;
                    regs[a] = inst & 0x1ffffff;
                    $step_check
                    continue;
                }

                let a = ((inst >> 6) & 7) as usize;
                let b = ((inst >> 3) & 7) as usize;
                let c = (inst & 7) as usize;

                match op {
                    0 => {
                        if regs[c] != 0 {
                            regs[a] = regs[b];
                        }
                    }
                    1 => {
                        let id = regs[b] as usize;
                        if id >= arrays.len() || active[id] == 0 {
                            if oob_zero {
                                regs[a] = 0;
                                step_count = step_count.wrapping_add(1);
                                $step_check
                                continue;
                            }
                            return Err(format!(
                                "array index from inactive array id {} pc={:#x} step={:#x}",
                                id, cur_pc, step_count
                            ));
                        }
                        let arr = &arrays[id];
                        let idx = regs[c] as usize;
                        if idx >= arr.len() {
                            if oob_zero {
                                regs[a] = 0;
                                step_count = step_count.wrapping_add(1);
                                $step_check
                                continue;
                            }
                            return Err(format!(
                                "array index out of bounds: id={} idx={} pc={:#x} step={:#x}",
                                id, idx, cur_pc, step_count
                            ));
                        }
                        // Safety: idx is bounds-checked above.
                        regs[a] = unsafe { *arr.get_unchecked(idx) };
                    }
                    2 => {
                        let id = regs[a] as usize;
                        if id >= arrays.len() || active[id] == 0 {
                            if oob_zero {
                                step_count = step_count.wrapping_add(1);
                                $step_check
                                continue;
                            }
                            return Err(format!(
                                "array amend inactive array id {} pc={:#x} step={:#x}",
                                id, cur_pc, step_count
                            ));
                        }
                        let arr = &mut arrays[id];
                        let idx = regs[b] as usize;
                        if idx >= arr.len() {
                            if oob_zero {
                                step_count = step_count.wrapping_add(1);
                                $step_check
                                continue;
                            }
                            return Err(format!(
                                "array amend out of bounds: id={} idx={} pc={:#x} step={:#x}",
                                id, idx, cur_pc, step_count
                            ));
                        }
                        // Safety: idx is bounds-checked above.
                        unsafe {
                            *arr.get_unchecked_mut(idx) = regs[c];
                        }
                        if !array_watch.is_empty() {
                            for watch in array_watch.iter_mut() {
                                if watch.id == id && watch.idx == idx {
                                    writeln!(
                                        watch.log,
                                        "{:016x} {:08x} {:08x} {:08x} {:08x}",
                                        step_count,
                                        cur_pc,
                                        id,
                                        idx,
                                        regs[c]
                                    )
                                    .map_err(|e| format!("array watch error: {}", e))?;
                                }
                            }
                        }
                    }
                    3 => {
                        regs[a] = regs[b].wrapping_add(regs[c]);
                    }
                    4 => {
                        regs[a] = regs[b].wrapping_mul(regs[c]);
                    }
                    5 => {
                        if regs[c] == 0 {
                            return Err("division by zero".to_string());
                        }
                        regs[a] = regs[b] / regs[c];
                    }
                    6 => {
                        regs[a] = !(regs[b] & regs[c]);
                    }
                    7 => {
                        output.flush().map_err(|e| format!("flush error: {}", e))?;
                        maybe_dump(dump_path, &mut dumped, &arrays, &active)?;
                        return Ok(());
                    }
                    8 => {
                        let size = regs[c] as usize;
                        let id = if let Some(id) = free_ids.pop() {
                            let arr = &mut arrays[id];
                            arr.clear();
                            arr.resize(size, 0);
                            active[id] = 1;
                            id
                        } else {
                            arrays.push(vec![0u32; size]);
                            active.push(1);
                            arrays.len() - 1
                        };
                        regs[b] = id as u32;
                    }
                    9 => {
                        let id = regs[c] as usize;
                        if id == 0 {
                            return Err("attempted to abandon array 0".to_string());
                        }
                        if id >= arrays.len() || active[id] == 0 {
                            return Err(format!("abandon inactive array id {}", id));
                        }
                        active[id] = 0;
                        arrays[id].clear();
                        free_ids.push(id);
                    }
                    10 => {
                        let val = regs[c];
                        if val > 255 {
                            return Err(format!("output value out of range: {}", val));
                        }
                        let byte = val as u8;
                        if output_trace.len() >= 4096 {
                            output_trace.pop_front();
                        }
                        output_trace.push_back((current_prog_id, pc - 1, byte, regs));
                        if output_on_substr.is_none() {
                            output
                                .write_all(&[byte])
                                .map_err(|e| format!("output error: {}", e))?;
                            if output_is_tty {
                                output.flush().map_err(|e| format!("flush error: {}", e))?;
                            }
                        }
                        if byte == b'\n' || byte == b'\r' {
                            if let Some(needle) = output_on_substr {
                                if line_buf.windows(needle.len()).any(|w| w == needle) {
                                    output
                                        .write_all(&line_buf)
                                        .map_err(|e| format!("output error: {}", e))?;
                                    output
                                        .write_all(&[byte])
                                        .map_err(|e| format!("output error: {}", e))?;
                                    if output_is_tty {
                                        output.flush()
                                            .map_err(|e| format!("flush error: {}", e))?;
                                    }
                                }
                            }
                            if !patch_after_substr.is_empty() {
                                for (needle, addr, value, applied) in patch_after_substr.iter_mut()
                                {
                                    if !*applied
                                        && line_buf.windows(needle.len()).any(|w| w == needle)
                                    {
                                        if *addr < arrays[0].len() {
                                            arrays[0][*addr] = *value;
                                            prog_ptr = arrays[0].as_ptr();
                                            prog_len = arrays[0].len();
                                            *applied = true;
                                        }
                                    }
                                }
                            }
                            if let Some((needle, _addr, _value, armed)) =
                                override_on_substr.as_mut()
                            {
                                if !*armed
                                    && line_buf.windows(needle.len()).any(|w| w == needle)
                                {
                                    *armed = true;
                                }
                            }
                            if !pub_dumped {
                                if let Some(path) = dump_on_pub.as_deref() {
                                    if line_has_publication(&line_buf) {
                                        eprintln!(
                                            "publication line detected: {}",
                                            String::from_utf8_lossy(&line_buf)
                                        );
                                        dump_arrays(path, &arrays, &active)?;
                                        pub_dumped = true;
                                    }
                                }
                            }
                            if !substr_dumped {
                                if let Some((needle, path)) = dump_on_substr {
                                    if line_buf.windows(needle.len()).any(|w| w == needle) {
                                        dump_arrays(path, &arrays, &active)?;
                                        substr_dumped = true;
                                    }
                                }
                            }
                            if !pub_traced {
                                if let Some(path) = trace_on_pub.as_deref() {
                                    if line_has_publication(&line_buf) {
                                        dump_output_trace(path, &line_buf, &output_trace)?;
                                        pub_traced = true;
                                    }
                                }
                            }
                            if !substr_traced {
                                if let Some((needle, path)) = trace_on_substr {
                                    if line_buf.windows(needle.len()).any(|w| w == needle) {
                                        dump_output_trace(path, &line_buf, &output_trace)?;
                                        substr_traced = true;
                                    }
                                }
                            }
                            if let Some((needle, log)) = log_substr_step.as_mut() {
                                if line_buf.windows(needle.len()).any(|w| w == needle) {
                                    let line_text = String::from_utf8_lossy(&line_buf);
                                    writeln!(log, "{:016x} {}", step_count, line_text)
                                        .map_err(|e| format!("substr step log error: {}", e))?;
                                }
                            }
                            line_buf.clear();
                        } else if line_buf.len() < 4096 {
                            line_buf.push(byte);
                        } else {
                            line_buf.clear();
                        }
                    }
                    11 => {
                        let prev_input_was_newline = last_input_was_newline;
                        if input_pos >= input_len {
                            match input.read(&mut input_buf) {
                                Ok(0) => {
                                    input_pos = 0;
                                    input_len = 0;
                                    regs[c] = 0xffff_ffff;
                                }
                                Ok(n) => {
                                    input_pos = 1;
                                    input_len = n;
                                    regs[c] = input_buf[0] as u32;
                                }
                                Err(e) => return Err(format!("input error: {}", e)),
                            }
                        } else {
                            regs[c] = input_buf[input_pos] as u32;
                            input_pos += 1;
                        }
                        let current_is_newline = regs[c] == b'\n' as u32;
                        if let Some((_addr, _value, armed)) = override_on_blank.as_mut() {
                            if prev_input_was_newline && current_is_newline {
                                *armed = true;
                            }
                        }
                        last_input_was_newline = current_is_newline;
                    }
                    12 => {
                        let op_pc = pc.wrapping_sub(1);
                        let id = regs[b] as usize;
                        if id != 0 {
                            if id >= arrays.len() || active[id] == 0 {
                                return Err(format!("load program from inactive array id {}", id));
                            }
                            let (head, tail) = arrays.split_at_mut(id);
                            let dst = &mut head[0];
                            let src = &tail[0];
                            dst.resize(src.len(), 0);
                            dst.copy_from_slice(src);
                            if !patches.is_empty() {
                                for &(addr, value) in patches {
                                    if addr < dst.len() {
                                        dst[addr] = value;
                                    }
                                }
                            }
                            prog_ptr = dst.as_ptr();
                            prog_len = dst.len();
                            current_prog_id = id;
                        }
                        pc = regs[c] as usize;
                        if let Some(log) = jump_log.as_mut() {
                            writeln!(
                                log,
                                "{:016x} {:08x} {:08x} {:08x}",
                                step_count,
                                op_pc,
                                id,
                                pc
                            )
                                .map_err(|e| format!("jump log error: {}", e))?;
                        }
                        if let Some((threshold, log)) = jump_log_after.as_mut() {
                            if step_count >= *threshold {
                                writeln!(
                                    log,
                                    "{:016x} {:08x} {:08x} {:08x}",
                                    step_count,
                                    op_pc,
                                    id,
                                    pc
                                )
                                .map_err(|e| format!("jump log error: {}", e))?;
                            }
                        }
                    }
                    _ => {
                        maybe_dump(dump_path, &mut dumped, &arrays, &active)?;
                        return Err(format!("invalid operator: {}", op));
                    }
                }

                step_count = step_count.wrapping_add(1);
                $step_check
            }
        }};
    }

    if let Some(mut steps_left) = step_limit {
        exec_loop!({
            steps_left -= 1;
            if steps_left == 0 {
                maybe_dump(dump_path, &mut dumped, &arrays, &active)?;
                return Err("step limit reached".to_string());
            }
        });
    } else {
        exec_loop!({});
    }
}

fn main() {
    let mut step_limit: Option<u64> = None;
    let mut dump_path: Option<String> = None;
    let mut dump_on_pub: Option<String> = None;
    let mut trace_on_pub: Option<String> = None;
    let mut dump_on_substr: Option<(Vec<u8>, String)> = None;
    let mut trace_on_substr: Option<(Vec<u8>, String)> = None;
    let mut output_on_substr: Option<Vec<u8>> = None;
    let mut jump_log: Option<fs::File> = None;
    let mut jump_log_after: Option<(u64, fs::File)> = None;
    let mut log_substr_step: Option<(Vec<u8>, fs::File)> = None;
    let mut trace_pc: Option<(usize, fs::File)> = None;
    let mut array_watch: Vec<ArrayWatch> = Vec::new();
    let mut array_patches: Vec<(usize, usize, u32, bool)> = Vec::new();
    let mut patches: Vec<(usize, u32)> = Vec::new();
    let mut patch_after_substr: Vec<(Vec<u8>, usize, u32, bool)> = Vec::new();
    let mut pc_patches: Vec<(usize, u32, bool)> = Vec::new();
    let mut override_on_substr: Option<(Vec<u8>, usize, u32, bool)> = None;
    let mut override_on_blank: Option<(usize, u32, bool)> = None;
    let mut oob_zero = false;
    let mut program_path: Option<String> = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "-steps" {
            let val = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let parsed: u64 = match val.parse() {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("invalid step count: {}", val);
                    std::process::exit(2);
                }
            };
            if parsed > 0 {
                step_limit = Some(parsed);
            }
        } else if arg == "-dump" {
            let val = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            dump_path = Some(val);
        } else if arg == "-dump-on-pub" {
            let val = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            dump_on_pub = Some(val);
        } else if arg == "-trace-on-pub" {
            let val = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            trace_on_pub = Some(val);
        } else if arg == "-dump-on-substr" {
            let needle = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let path = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            dump_on_substr = Some((needle.into_bytes(), path));
        } else if arg == "-trace-on-substr" {
            let needle = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let path = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            trace_on_substr = Some((needle.into_bytes(), path));
        } else if arg == "-output-on-substr" {
            let needle = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            output_on_substr = Some(needle.into_bytes());
        } else if arg == "-log-substr-step" {
            let needle = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let path = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let file = match fs::File::create(&path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("substr step log open error: {}", e);
                    std::process::exit(1);
                }
            };
            log_substr_step = Some((needle.into_bytes(), file));
        } else if arg == "-log-jumps" {
            let path = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let file = match fs::File::create(&path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("jump log open error: {}", e);
                    std::process::exit(1);
                }
            };
            jump_log = Some(file);
        } else if arg == "-log-jumps-after-step" {
            let step_s = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let path = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let threshold = match step_s.parse::<u64>() {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("invalid step for jump log: {}", step_s);
                    std::process::exit(2);
                }
            };
            let file = match fs::File::create(&path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("jump log open error: {}", e);
                    std::process::exit(1);
                }
            };
            jump_log_after = Some((threshold, file));
        } else if arg == "-trace-pc" {
            let addr_s = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let path = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let addr = match u32::from_str_radix(addr_s.trim_start_matches("0x"), 16)
                .or_else(|_| addr_s.parse())
            {
                Ok(v) => v as usize,
                Err(_) => {
                    eprintln!("invalid trace pc addr: {}", addr_s);
                    std::process::exit(2);
                }
            };
            let file = match fs::File::create(&path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("trace pc open error: {}", e);
                    std::process::exit(1);
                }
            };
            trace_pc = Some((addr, file));
        } else if arg == "-watch-array" {
            let id_s = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let idx_s = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let path = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let arr_id = match u32::from_str_radix(id_s.trim_start_matches("0x"), 16)
                .or_else(|_| id_s.parse())
            {
                Ok(v) => v as usize,
                Err(_) => {
                    eprintln!("invalid watch array id: {}", id_s);
                    std::process::exit(2);
                }
            };
            let idx_val = match u32::from_str_radix(idx_s.trim_start_matches("0x"), 16)
                .or_else(|_| idx_s.parse())
            {
                Ok(v) => v as usize,
                Err(_) => {
                    eprintln!("invalid watch array idx: {}", idx_s);
                    std::process::exit(2);
                }
            };
            let file = match fs::File::create(&path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("watch array open error: {}", e);
                    std::process::exit(1);
                }
            };
            array_watch.push(ArrayWatch {
                id: arr_id,
                idx: idx_val,
                log: file,
            });
        } else if arg == "-patch-array" {
            let id_s = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let idx_s = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let val_s = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let arr_id = match u32::from_str_radix(id_s.trim_start_matches("0x"), 16)
                .or_else(|_| id_s.parse())
            {
                Ok(v) => v as usize,
                Err(_) => {
                    eprintln!("invalid patch array id: {}", id_s);
                    std::process::exit(2);
                }
            };
            let idx_val = match u32::from_str_radix(idx_s.trim_start_matches("0x"), 16)
                .or_else(|_| idx_s.parse())
            {
                Ok(v) => v as usize,
                Err(_) => {
                    eprintln!("invalid patch array idx: {}", idx_s);
                    std::process::exit(2);
                }
            };
            let value = match u32::from_str_radix(val_s.trim_start_matches("0x"), 16)
                .or_else(|_| val_s.parse())
            {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("invalid patch array value: {}", val_s);
                    std::process::exit(2);
                }
            };
            array_patches.push((arr_id, idx_val, value, false));
        } else if arg == "-patch" {
            let addr_s = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let val_s = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let addr = match u32::from_str_radix(addr_s.trim_start_matches("0x"), 16)
                .or_else(|_| addr_s.parse())
            {
                Ok(v) => v as usize,
                Err(_) => {
                    eprintln!("invalid patch addr: {}", addr_s);
                    std::process::exit(2);
                }
            };
            let value = match u32::from_str_radix(val_s.trim_start_matches("0x"), 16)
                .or_else(|_| val_s.parse())
            {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("invalid patch value: {}", val_s);
                    std::process::exit(2);
                }
            };
            patches.push((addr, value));
        } else if arg == "-patch-after-substr" {
            let needle = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let addr_s = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let val_s = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let addr = match u32::from_str_radix(addr_s.trim_start_matches("0x"), 16)
                .or_else(|_| addr_s.parse())
            {
                Ok(v) => v as usize,
                Err(_) => {
                    eprintln!("invalid patch addr: {}", addr_s);
                    std::process::exit(2);
                }
            };
            let value = match u32::from_str_radix(val_s.trim_start_matches("0x"), 16)
                .or_else(|_| val_s.parse())
            {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("invalid patch value: {}", val_s);
                    std::process::exit(2);
                }
            };
            patch_after_substr.push((needle.into_bytes(), addr, value, false));
        } else if arg == "-patch-on-pc" {
            let addr_s = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let val_s = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let addr = match u32::from_str_radix(addr_s.trim_start_matches("0x"), 16)
                .or_else(|_| addr_s.parse())
            {
                Ok(v) => v as usize,
                Err(_) => {
                    eprintln!("invalid patch addr: {}", addr_s);
                    std::process::exit(2);
                }
            };
            let value = match u32::from_str_radix(val_s.trim_start_matches("0x"), 16)
                .or_else(|_| val_s.parse())
            {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("invalid patch value: {}", val_s);
                    std::process::exit(2);
                }
            };
            pc_patches.push((addr, value, false));
        } else if arg == "-override-on-substr" {
            let needle = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let addr_s = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let val_s = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let addr = match u32::from_str_radix(addr_s.trim_start_matches("0x"), 16)
                .or_else(|_| addr_s.parse())
            {
                Ok(v) => v as usize,
                Err(_) => {
                    eprintln!("invalid override addr: {}", addr_s);
                    std::process::exit(2);
                }
            };
            let value = match u32::from_str_radix(val_s.trim_start_matches("0x"), 16)
                .or_else(|_| val_s.parse())
            {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("invalid override value: {}", val_s);
                    std::process::exit(2);
                }
            };
            override_on_substr = Some((needle.into_bytes(), addr, value, false));
        } else if arg == "-override-on-blank-line" {
            let addr_s = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let val_s = match args.next() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
                    );
                    std::process::exit(2);
                }
            };
            let addr = match u32::from_str_radix(addr_s.trim_start_matches("0x"), 16)
                .or_else(|_| addr_s.parse())
            {
                Ok(v) => v as usize,
                Err(_) => {
                    eprintln!("invalid override addr: {}", addr_s);
                    std::process::exit(2);
                }
            };
            let value = match u32::from_str_radix(val_s.trim_start_matches("0x"), 16)
                .or_else(|_| val_s.parse())
            {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("invalid override value: {}", val_s);
                    std::process::exit(2);
                }
            };
            override_on_blank = Some((addr, value, false));
        } else if arg == "-oob-zero" {
            oob_zero = true;
        } else if program_path.is_none() {
            program_path = Some(arg);
        } else {
            eprintln!(
                "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
            );
            std::process::exit(2);
        }
    }

    let path = match program_path {
        Some(p) => p,
        None => {
            eprintln!(
                "usage: um [-steps N] [-dump PATH] [-dump-on-pub PATH] [-trace-on-pub PATH] [-dump-on-substr STR PATH] [-trace-on-substr STR PATH] [-output-on-substr STR] [-log-substr-step STR PATH] [-log-jumps PATH] [-log-jumps-after-step N PATH] [-trace-pc ADDR PATH] [-watch-array ID IDX PATH] [-patch-array ID IDX VALUE] [-patch ADDR VALUE] [-patch-after-substr STR ADDR VALUE] [-patch-on-pc ADDR VALUE] [-override-on-substr STR ADDR VALUE] [-override-on-blank-line ADDR VALUE] [-oob-zero] program.um"
            );
            std::process::exit(2);
        }
    };

    let program = match read_program(&path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("load error: {}", e);
            std::process::exit(1);
        }
    };

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = io::BufReader::new(stdin.lock());
    let mut output = io::BufWriter::new(stdout.lock());

    let output_is_tty = stdout.is_terminal();
    if let Err(e) = run(
        program,
        &mut input,
        &mut output,
        step_limit,
        output_is_tty,
        &dump_path,
        &dump_on_pub,
        &trace_on_pub,
        &dump_on_substr,
        &trace_on_substr,
        &output_on_substr,
        &mut jump_log,
        &mut jump_log_after,
        &mut log_substr_step,
        &mut trace_pc,
        &mut array_watch,
        &mut array_patches,
        &patches,
        &mut patch_after_substr,
        &mut pc_patches,
        &mut override_on_substr,
        &mut override_on_blank,
        oob_zero,
    ) {
        eprintln!("runtime error: {}", e);
        std::process::exit(1);
    }
}
