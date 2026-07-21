#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

struct Rect { const char *name; int w, h; };

static std::array<Rect, 15> rs = {{{"E",55,22},{"main",51,8},{"H",35,9},
    {"l",49,19},{"M",38,17},{"B",49,11},{"Z",50,19},{"A",42,19},
    {"r",49,19},{"L",59,21},{"T",34,9},{"R",49,14},{"D",33,9},
    {"Q",52,23},{"X",90,25}}};

struct Layout { int w, h; std::array<int,15> x, y; };

static Layout evaluate(const std::array<int,15>& p, const std::array<int,15>& q) {
    std::array<int,15> pp{}, pq{}, x{}, y{};
    for (int k=0;k<15;k++) pp[p[k]]=k, pq[q[k]]=k;
    for (int ai=0;ai<15;ai++) {
        int a=p[ai];
        for (int bi=ai+1;bi<15;bi++) {
            int b=p[bi];
            if (pq[a] < pq[b]) x[b]=std::max(x[b],x[a]+rs[a].w);
        }
    }
    for (int ai=0;ai<15;ai++) {
        int a=q[ai];
        for (int bi=ai+1;bi<15;bi++) {
            int b=q[bi];
            if (pp[a] > pp[b]) y[b]=std::max(y[b],y[a]+rs[a].h);
        }
    }
    int w=0,h=0;
    for(int i=0;i<15;i++) w=std::max(w,x[i]+rs[i].w), h=std::max(h,y[i]+rs[i].h);
    return {w,h,x,y};
}

static int target_w = 141;
static int target_h = 92;

static long long cost(const Layout& z) {
    long long overflow = std::max(0,z.w-target_w) + std::max(0,z.h-target_h);
    return overflow*100000000LL + 1LL*z.w*z.h;
}

int main(int argc, char **argv) {
    long long iterations = argc > 1 ? std::stoll(argv[1]) : 20000000;
    unsigned seed = argc > 2 ? std::stoul(argv[2]) : 1;
    if (argc > 3) target_w = std::stoi(argv[3]);
    if (argc > 4) target_h = std::stoi(argv[4]);
    for (int ai=5; ai<argc; ai++) {
        std::string spec=argv[ai];
        auto p1=spec.find(':'), p2=spec.find(':',p1+1);
        if(p1==std::string::npos || p2==std::string::npos) continue;
        std::string name=spec.substr(0,p1);
        int dw=std::stoi(spec.substr(p1+1,p2-p1-1));
        int dh=std::stoi(spec.substr(p2+1));
        for(auto &r:rs) if(name==r.name) r.w+=dw,r.h+=dh;
    }
    std::mt19937 rng(seed);
    std::array<int,15> p{},q{}; std::iota(p.begin(),p.end(),0); q=p;
    std::shuffle(p.begin(),p.end(),rng); std::shuffle(q.begin(),q.end(),rng);
    auto cur=evaluate(p,q), best=cur; auto bp=p,bq=q;
    double temp=2e8;
    for(long long it=0;it<iterations;it++) {
        auto np=p,nq=q;
        int a=rng()%15,b=rng()%15;
        if(rng()&1) std::swap(np[a],np[b]); else std::swap(nq[a],nq[b]);
        auto nxt=evaluate(np,nq); long long dc=cost(nxt)-cost(cur);
        double t=std::max(1000.0,temp*(1.0-double(it%200000)/200000.0));
        if(dc<=0 || std::generate_canonical<double,32>(rng)<std::exp(-double(dc)/t)) p=np,q=nq,cur=nxt;
        if(cost(cur)<cost(best)) {
            best=cur;bp=p;bq=q;
            std::cerr << "best "<<best.w<<"x"<<best.h<<" area="<<best.w*best.h<<" at "<<it<<"\n";
            if(best.w<=target_w && best.h<=target_h) break;
        }
        if(it%200000==199999) { p=bp;q=bq;cur=best; temp*=0.97; }
    }
    std::cout<<"SIZE "<<best.w<<" "<<best.h<<"\n";
    for(int i=0;i<15;i++) std::cout<<rs[i].name<<" "<<best.x[i]<<" "<<best.y[i]<<" "<<rs[i].w<<" "<<rs[i].h<<"\n";
}
