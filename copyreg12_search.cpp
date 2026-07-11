#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <unordered_set>
#include <vector>

using Code = std::array<uint8_t, 12>;

struct Score { int ok, halt; long long steps; };

static inline int imm5(uint8_t b) {
    int v = b & 31;
    return (v & 16) ? v - 32 : v;
}

static Score evaluate(const Code &code, int max_steps = 1600) {
    Score out{0, 0, 0};
    for (int input = 1; input < 256; ++input) {
        uint8_t mem[256]{};
        for (int i = 0; i < 8; ++i) mem[i] = uint8_t(1u << i);
        uint8_t sr[4] = {uint8_t(input), 0, 1, 2};
        uint8_t dr[2] = {3, 4};
        unsigned ip = 0;
        int speed = 1;
        bool halted = false;
        int step = 0;
        for (; step < max_steps; ++step) {
            uint8_t b = code[ip];
            int op = b >> 5;
            if (op == 1 || op == 2) {
                int dd = (b >> 4) & 1, s1 = (b >> 2) & 3, s2 = b & 3;
                uint8_t a = mem[sr[s1]], bb = mem[sr[s2]];
                uint8_t c = mem[sr[(s1 + 1) & 3]], d = mem[sr[(s2 + 1) & 3]];
                mem[dr[(dd + 1) & 1]] = op == 1 ? uint8_t(c - d) : uint8_t(c ^ d);
                mem[dr[dd]] = op == 1 ? uint8_t(a + bb) : uint8_t(a & bb);
            } else if (op == 0) {
                if (mem[sr[0]] != 0) speed = imm5(b);
                if (speed == 0) { halted = true; ++step; break; }
            } else if (op == 3) {
                int im = imm5(b), mask = im & 31;
                sr[0] = uint8_t(sr[0] + im);
                int idx[5] = {5, 4, 3, 2, 1}; // dr1, dr0, sr3, sr2, sr1
                uint8_t all[6] = {sr[0], sr[1], sr[2], sr[3], dr[0], dr[1]};
                int chosen[5], n = 0;
                for (int i = 0; i < 5; ++i) if ((mask >> i) & 1) chosen[n++] = idx[i];
                if (n) {
                    uint8_t first = all[0];
                    for (int i = 0; i < n; ++i) {
                        uint8_t next = all[chosen[i]];
                        all[chosen[i]] = first;
                        first = next;
                    }
                    all[0] = first;
                    for (int i = 0; i < 4; ++i) sr[i] = all[i];
                    dr[0] = all[4]; dr[1] = all[5];
                }
            }
            // The reference interpreter wraps the signed speed addition to
            // uint32 before reducing it modulo the program length.
            ip = uint32_t(int64_t(ip) + speed) % 12;
        }
        out.steps += step;
        if (!halted) continue;
        ++out.halt;
        for (uint8_t v : mem) if (v == uint8_t(input)) { ++out.ok; break; }
    }
    return out;
}

static bool better(const Score &a, const Score &b) {
    if (a.ok != b.ok) return a.ok > b.ok;
    if (a.halt != b.halt) return a.halt > b.halt;
    return a.steps < b.steps;
}

static void print_code(const Code &c) {
    for (auto b : c) std::printf("%02x", b);
}

int main(int argc, char **argv) {
    int seconds = argc > 1 ? std::atoi(argv[1]) : 300;
    unsigned seed = argc > 2 ? unsigned(std::strtoul(argv[2], nullptr, 10)) : 1;
    std::mt19937 rng(seed);
    const std::array<uint8_t, 13> base = {0x29,0x22,0x65,0x0b,0x00,0x2e,0x29,0x55,0x03,0x67,0x59,0x77,0x0a};
    struct Item { Code c; Score s; };
    std::vector<Item> pop;
    auto add = [&](Code c) { pop.push_back({c, evaluate(c)}); };
    for (int drop = 0; drop < 13; ++drop) {
        Code c{}; int j = 0;
        for (int i = 0; i < 13; ++i) if (i != drop) c[j++] = base[i];
        add(c);
        for (int k = 0; k < 100; ++k) {
            Code d = c;
            for (int m = 0, n = 1 + int(rng() % 3); m < n; ++m) d[rng()%12] = uint8_t(rng()%128);
            add(d);
        }
    }
    Score best{-1,-1,0}; Code bestc{};
    auto started = std::chrono::steady_clock::now();
    int generation = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count() < seconds) {
        ++generation;
        std::sort(pop.begin(), pop.end(), [](const Item&a,const Item&b){ return better(a.s,b.s); });
        if (better(pop[0].s, best)) {
            best=pop[0].s; bestc=pop[0].c;
            std::printf("gen=%d ok=%d halt=%d steps=%lld hex=",generation,best.ok,best.halt,best.steps);
            print_code(bestc); std::printf("\n"); std::fflush(stdout);
        }
        const int elite = std::min<int>(200, pop.size());
        std::vector<Item> next(pop.begin(), pop.begin()+elite);
        next.reserve(1800);
        while (next.size() < 1800) {
            Code c = pop[rng()%elite].c;
            unsigned mode=rng()%100;
            if (mode < 65) {
                int n = 1 + (rng()%100 < 18) + (rng()%100 < 4);
                while(n--) c[rng()%12]=uint8_t(rng()%128);
            } else if (mode < 78) {
                std::swap(c[rng()%12],c[rng()%12]);
            } else if (mode < 90) {
                int from=rng()%12,to=rng()%12; uint8_t v=c[from];
                if(from<to) for(int i=from;i<to;++i)c[i]=c[i+1];
                else for(int i=from;i>to;--i)c[i]=c[i-1];
                c[to]=v;
            } else {
                const Code &d=pop[rng()%elite].c;
                int lo=rng()%12, hi=lo+rng()%(13-lo);
                for(int i=lo;i<hi;++i)c[i]=d[i];
            }
            // A zero-speed SCIENCE instruction is necessary for a normal halt.
            if (std::find(c.begin(),c.end(),uint8_t(0))==c.end()) c[rng()%12]=0;
            next.push_back({c,evaluate(c)});
        }
        pop.swap(next);
    }
    std::printf("final ok=%d halt=%d hex=",best.ok,best.halt); print_code(bestc); std::printf("\n");
}
