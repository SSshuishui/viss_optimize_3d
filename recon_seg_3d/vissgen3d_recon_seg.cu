// vissgen3d_unified_blockage_nside.cu
// Unified 1M/10M version with runtime blockage switch (blockage=1 uses FOV+penalty, blockage=0 disables both), WITHOUT thrust.
// - nside param; npix=12*nside*nside (NOT read from B file)
// - B/theta/phi: B_<btag>.txt theta_heal_<btag>.txt phi_heal_<btag>.txt (no header)
// - Multi-GPU pixel-chunk split
// - Unified timing output per day
//
// Build:
//   nvcc -O3 -lineinfo -Xcompiler -fopenmp -std=c++14 vissgen3d_unified_blockage_nside.cu -o vissgen3d_unified_blockage_nside
//
// Run example:
//   ./vissgen3d_unified_blockage_nside --btag=10M --nside=4096 --start_day=432 --end_day=450 \
//       --in_dir=/data/zhaox/earth_10Mhz --out_dir=./3dunified10M/ --gpus=0,1,2,3 --uvw_max=450000 --blockage=1

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>

#include <cuda_runtime.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CHECK_CUDA(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while(0)

struct HostTimer {
  using clock = std::chrono::high_resolution_clock;
  clock::time_point t0;
  void tic() { t0 = clock::now(); }
  double toc_s() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - t0).count();
  }
};

static inline std::string get_arg(int argc, char** argv, const std::string& key, const std::string& defv) {
  for (int i=1;i<argc;i++) {
    std::string s(argv[i]);
    if (s.rfind(key + "=", 0) == 0) return s.substr(key.size()+1);
  }
  return defv;
}
static inline int to_int(const std::string& s, int defv) {
  try { return std::stoi(s); } catch(...) { return defv; }
}
static inline std::vector<int> parse_gpus(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) out.push_back(std::stoi(tok));
  }
  if (out.empty()) out.push_back(0);
  return out;
}
static inline std::string norm_dir(std::string p) {
  while (!p.empty() && p.back() == '/') p.pop_back();
  return p;
}
static inline void ensure_dir(const std::string& path) {
  std::string cmd = "mkdir -p " + path;
  std::system(cmd.c_str());
}

// B/theta/phi：无表头，每行一个 float（或空白分隔也行）
static bool load_single_noheader(const std::string& path, float* out, long long n) {
  FILE* fp = fopen(path.c_str(), "r");
  if (!fp) return false;
  char line[256];
  long long i = 0;
  while (i < n && fgets(line, sizeof(line), fp)) {
    char* p = line;
    out[i] = strtof(p, &p);
    i++;
  }
  fclose(fp);
  return (i == n);
}

// updated_uvw / xyz：第一行表头，后面每行若干列 float


static bool load_single_skip_first(
    const std::string& path,
    float* a,
    int maxN,
    int& outN
) {
  FILE* fp = fopen(path.c_str(), "r");
  if (!fp) return false;
  char line[256];
  if (!fgets(line, sizeof(line), fp)) { fclose(fp); return false; }

  int n = 0;
  while (n < maxN && fgets(line, sizeof(line), fp)) {
    char* p = line;
    a[n] = strtof(p, &p);
    n++;
  }
  fclose(fp);
  outN = n;
  return true;
}

static bool load_quad_skip_first(
    const std::string& path,
    float* a, float* b, float* c, float* d,
    int maxN,
    int& outN
) {
  FILE* fp = fopen(path.c_str(), "r");
  if (!fp) return false;
  char line[256];
  if (!fgets(line, sizeof(line), fp)) { fclose(fp); return false; }

  int n = 0;
  while (n < maxN && fgets(line, sizeof(line), fp)) {
    char* p = line;
    a[n] = strtof(p, &p);
    b[n] = strtof(p, &p);
    c[n] = strtof(p, &p);
    d[n] = strtof(p, &p);
    n++;
  }
  fclose(fp);
  outN = n;
  return true;
}

static bool load_triplets_skip_first(
    const std::string& path,
    float* a, float* b, float* c,
    int maxN,
    int& outN
) {
  FILE* fp = fopen(path.c_str(), "r");
  if (!fp) return false;
  char line[256];
  if (!fgets(line, sizeof(line), fp)) { fclose(fp); return false; }

  int n = 0;
  while (n < maxN && fgets(line, sizeof(line), fp)) {
    char* p = line;
    a[n] = strtof(p, &p);
    b[n] = strtof(p, &p);
    c[n] = strtof(p, &p);
    n++;
  }
  fclose(fp);
  outN = n;
  return true;
}

__device__ __forceinline__ void sincos_fast(float x, float* s, float* c) { __sincosf(x, s, c); }
__device__ __forceinline__ float norm3(float x, float y, float z) { return sqrtf(x*x + y*y + z*z); }

// chunk theta/phi -> lmn
__global__ void healpix_lmn_from_theta_phi_chunk(
    const float* __restrict__ theta,
    const float* __restrict__ phi,
    float* __restrict__ l,
    float* __restrict__ m,
    float* __restrict__ n,
    long long n_chunk
) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_chunk) return;

  float theta_val = theta[idx];
  float phi_val   = phi[idx];

  theta_val = (float)M_PI * 0.5f - theta_val;
  if (phi_val > (float)M_PI) phi_val -= 2.0f * (float)M_PI;
  phi_val = -phi_val;

  float st, ct, sp, cp;
  sincos_fast(theta_val, &st, &ct);
  sincos_fast(phi_val,   &sp, &cp);

  l[idx] = ct * cp;
  m[idx] = ct * sp;
  n[idx] = st;
}



__global__ void ceilAndScale_opt(
    const float* __restrict__ bll,
    int* __restrict__ gs,
    int size,
    int nr
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;

  float x = (bll[idx] - 0.25f) * 2.0f;
  int g = __float2int_ru(x);
  if (g < 0) g = 0;
  if (g > nr) g = nr;
  gs[idx] = g;
}

__global__ void phase_correct_and_dg(
    float2* __restrict__ Viss,
    const float* __restrict__ w,
    const float* __restrict__ bll,
    float* __restrict__ dg,
    int uvw_index
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= uvw_index) return;

  float wv = w[idx];
  float bv = bll[idx];

  float ang = -2.0f * (float)M_PI * wv;
  float s, c;
  sincos_fast(ang, &s, &c);

  float a = Viss[idx].x;
  float b = Viss[idx].y;

  float re = a * c - b * s;
  float im = a * s + b * c;
  Viss[idx] = make_float2(re, im);

  float ratio = (bv != 0.0f) ? (wv / bv) : 0.0f;
  ratio = fminf(1.0f, fmaxf(-1.0f, ratio));

  float gamma = asinf(ratio);
  float sg = sinf(gamma);
  float cg = cosf(gamma);

  float inside = 1.0f - 4.0f * sg * sg;
  float mag = sqrtf(fabsf(inside));

  float denom = fabsf(cg);
  if (denom < 1e-12f) denom = 1e-12f;

  dg[idx] = mag / denom * 1.5f;
}

__global__ void precompute_weight(
    float* __restrict__ weight,
    const float* __restrict__ dcf,
    const float* __restrict__ dg,
    const int* __restrict__ gs,
    int uvw_index,
    int nr
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= uvw_index) return;

  int g = gs[idx];
  if (g < 0) g = 0;
  if (g > nr) g = nr;

  float val = dcf[g] * dg[idx];
  if (val > (1.0f / 8.0f)) val = (1.0f / 8.0f);
  weight[idx] = val;
}

// Viss partial (per GPU chunk), then host reduce
template<int TILE_PIX>
__global__ void viss_partial(
    const float* __restrict__ B,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    long long n_chunk,

    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,

    const float* __restrict__ xyz1a, const float* __restrict__ xyz1b, const float* __restrict__ xyz1c,
    const float* __restrict__ xyz2a, const float* __restrict__ xyz2b, const float* __restrict__ xyz2c,
    float phi,

    int amount,
    float2* __restrict__ Vpart
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= amount) return;

  float u0 = u[i];
  float v0 = v[i];
  float w0 = w[i];

  float x1a0=xyz1a[i], x1b0=xyz1b[i], x1c0=xyz1c[i];
  float x2a0=xyz2a[i], x2b0=xyz2b[i], x2c0=xyz2c[i];
  float cosphi = cosf(phi);
  float thr1 = norm3(x1a0,x1b0,x1c0) * cosphi;
  float thr2 = norm3(x2a0,x2b0,x2c0) * cosphi;

  float acc_re = 0.0f;
  float acc_im = 0.0f;
  const float k = -2.0f * (float)M_PI;

  extern __shared__ float smem[];
  float* sB = smem;
  float* sL = sB + TILE_PIX;
  float* sM = sL + TILE_PIX;
  float* sN = sM + TILE_PIX;

  for (long long base = 0; base < n_chunk; base += TILE_PIX) {
    int t0 = threadIdx.x;
    if (t0 < TILE_PIX) {
      long long p = base + t0;
      if (p < n_chunk) {
        sB[t0] = B[p];
        sL[t0] = l[p];
        sM[t0] = m[p];
        sN[t0] = n[p];
      } else {
        sB[t0]=0.0f; sL[t0]=0.0f; sM[t0]=0.0f; sN[t0]=0.0f;
      }
    }
    int t1 = threadIdx.x + blockDim.x;
    if (t1 < TILE_PIX) {
      long long p = base + t1;
      if (p < n_chunk) {
        sB[t1] = B[p];
        sL[t1] = l[p];
        sM[t1] = m[p];
        sN[t1] = n[p];
      } else {
        sB[t1]=0.0f; sL[t1]=0.0f; sM[t1]=0.0f; sN[t1]=0.0f;
      }
    }
    __syncthreads();

    int tileN = (int)min((long long)TILE_PIX, n_chunk - base);
    #pragma unroll 4
    for (int t=0;t<tileN;t++) {
      float lp=sL[t], mp=sM[t], npv=sN[t];
      float dot1 = lp*x1a0 + mp*x1b0 + npv*x1c0;
      float dot2 = lp*x2a0 + mp*x2b0 + npv*x2c0;
      if (dot1 < thr1 || dot2 < thr2) continue;

      float phase = u0*lp + v0*mp + w0*(npv - 1.0f);
      float ang = k * phase;
      float s, c;
      sincos_fast(ang, &s, &c);
      float bp = sB[t];
      acc_re += bp * c;
      acc_im += bp * s;
    }

    __syncthreads();
  }

  Vpart[i] = make_float2(acc_re, acc_im);
}

// compute C real (per GPU chunk), uvw tiling, with MATLAB weight + optional recon blockage
template<int TILE_UVW>
__global__ void computeC_real_chunk(
    long long n_chunk,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,

    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ weight,

    const float* __restrict__ xyz1a, const float* __restrict__ xyz1b, const float* __restrict__ xyz1c,
    const float* __restrict__ xyz2a, const float* __restrict__ xyz2b, const float* __restrict__ xyz2c,
    float phi,
    int use_blockage,

    const float2* __restrict__ Viss,
    int uvw_index,

    float* __restrict__ out_real
) {
  long long pix = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (pix >= n_chunk) return;

  float lp = l[pix];
  float mp = m[pix];
  float npv = n[pix];

  float acc_re = 0.0f;
  float acc_im = 0.0f;

  float cosphi = cosf(phi);

  extern __shared__ unsigned char smem_raw[];
  float* su = (float*)smem_raw;
  float* sv = su + TILE_UVW;
  float* sw = sv + TILE_UVW;
  float* sweight = sw + TILE_UVW;

  float* sx1a = sweight + TILE_UVW;
  float* sx1b = sx1a + TILE_UVW;
  float* sx1c = sx1b + TILE_UVW;

  float* sx2a = sx1c + TILE_UVW;
  float* sx2b = sx2a + TILE_UVW;
  float* sx2c = sx2b + TILE_UVW;

  float* sInv1 = sx2c + TILE_UVW;
  float* sInv2 = sInv1 + TILE_UVW;

  float2* sV = (float2*)(sInv2 + TILE_UVW);

  for (int base=0; base<uvw_index; base += TILE_UVW) {
    int i = base + threadIdx.x;

    if (threadIdx.x < TILE_UVW) {
      if (i < uvw_index) {
        su[threadIdx.x] = u[i];
        sv[threadIdx.x] = v[i];
        sw[threadIdx.x] = w[i];
        sweight[threadIdx.x] = weight[i];

        float x1a0=xyz1a[i], x1b0=xyz1b[i], x1c0=xyz1c[i];
        float x2a0=xyz2a[i], x2b0=xyz2b[i], x2c0=xyz2c[i];
        sx1a[threadIdx.x]=x1a0; sx1b[threadIdx.x]=x1b0; sx1c[threadIdx.x]=x1c0;
        sx2a[threadIdx.x]=x2a0; sx2b[threadIdx.x]=x2b0; sx2c[threadIdx.x]=x2c0;
        sInv1[threadIdx.x]=rsqrtf(x1a0*x1a0+x1b0*x1b0+x1c0*x1c0+1e-20f);
        sInv2[threadIdx.x]=rsqrtf(x2a0*x2a0+x2b0*x2b0+x2c0*x2c0+1e-20f);

        sV[threadIdx.x] = Viss[i];
      } else {
        su[threadIdx.x]=0.0f; sv[threadIdx.x]=0.0f; sw[threadIdx.x]=0.0f; sweight[threadIdx.x]=0.0f;
        sx1a[threadIdx.x]=0.0f; sx1b[threadIdx.x]=0.0f; sx1c[threadIdx.x]=0.0f;
        sx2a[threadIdx.x]=0.0f; sx2b[threadIdx.x]=0.0f; sx2c[threadIdx.x]=0.0f;
        sInv1[threadIdx.x]=0.0f; sInv2[threadIdx.x]=0.0f;
        sV[threadIdx.x]=make_float2(0.0f,0.0f);
      }
    }
    __syncthreads();

    int tileN = min(TILE_UVW, uvw_index - base);
    #pragma unroll 4
    for (int t=0;t<tileN;t++) {
      float wt = sweight[t];
      if (wt == 0.0f) continue;

      if (use_blockage) {
        float dot1 = lp*sx1a[t] + mp*sx1b[t] + npv*sx1c[t];
        float dot2 = lp*sx2a[t] + mp*sx2b[t] + npv*sx2c[t];
        float c1 = dot1 * sInv1[t];
        float c2 = dot2 * sInv2[t];
        if (!(c1 >= cosphi && c2 >= cosphi)) continue;
      }

      float phase = su[t]*lp + sv[t]*mp + sw[t]*npv;
      float ang = 2.0f * (float)M_PI * phase;
      float s, c;
      sincos_fast(ang, &s, &c);

      float a = sV[t].x;
      float b = sV[t].y;

      acc_re += wt * (a * c - b * s);
      acc_im += wt * (a * s + b * c);
    }
    __syncthreads();
  }

  out_real[pix] = acc_re;
}

struct GpuCtx {
  int dev = 0;
  cudaStream_t stream = nullptr;

  long long pix0 = 0;
  long long pix1 = 0;
  long long n_chunk = 0;

  float* d_B=nullptr;
  float* d_theta=nullptr;
  float* d_phi=nullptr;
  float* d_l=nullptr;
  float* d_m=nullptr;
  float* d_n=nullptr;

  int UVW_MAX = 450000;
  float* d_u=nullptr; float* d_v=nullptr; float* d_w=nullptr;
  float* d_bll=nullptr;
  int*   d_gs=nullptr;
  float* d_dg=nullptr;
  float* d_weight=nullptr;
  float* d_dcf=nullptr;

  float* d_xyz1a=nullptr; float* d_xyz1b=nullptr; float* d_xyz1c=nullptr;
  float* d_xyz2a=nullptr; float* d_xyz2b=nullptr; float* d_xyz2c=nullptr;

  float2* d_Viss=nullptr;  // uvw_index

  float* d_Creal=nullptr;

  float2* h_Vpart=nullptr; // pinned, max amount
  float*  h_chunk=nullptr; // pinned
};

int main(int argc, char** argv) {
  std::string btag = get_arg(argc, argv, "--btag", "10M");
  int nside = to_int(get_arg(argc, argv, "--nside", "0"), 0);
  int start_day = to_int(get_arg(argc, argv, "--start_day", "432"), 432);
  int end_day   = to_int(get_arg(argc, argv, "--end_day", "450"), 450);
  int dcf_start_day = to_int(get_arg(argc, argv, "--dcf_start_day", std::to_string(start_day)), start_day);
  int dcf_end_day   = to_int(get_arg(argc, argv, "--dcf_end_day", std::to_string(end_day)), end_day);
  std::string in_dir  = norm_dir(get_arg(argc, argv, "--in_dir", ""));
  std::string out_dir = get_arg(argc, argv, "--out_dir", "");
  std::string gpus_s  = get_arg(argc, argv, "--gpus", "0");
  int uvw_max = to_int(get_arg(argc, argv, "--uvw_max", "450000"), 450000);
  int blockage = to_int(get_arg(argc, argv, "--blockage", "1"), 1);

  if (btag != "1M" && btag != "10M") {
    std::cerr << "ERROR: --btag must be 1M or 10M\n";
    return 1;
  }
  if (nside <= 0) nside = (btag == "1M") ? 512 : 4096;

  long long npix = 12LL * (long long)nside * (long long)nside;
  if (npix <= 0) {
    std::cerr << "ERROR: invalid npix\n";
    return 1;
  }

  if (in_dir.empty()) {
    in_dir = (btag == "1M") ? "../earth_1Mhz" : "../earth_10Mhz";
  }
  if (out_dir.empty()) {
    out_dir = std::string("./3dunified") + btag + ((blockage != 0) ? "_block/" : "_noblock/");
  }
  if (!out_dir.empty() && out_dir.back() != '/') out_dir.push_back('/');
  ensure_dir(out_dir);

  auto gpus = parse_gpus(gpus_s);

  int devCount=0;
  CHECK_CUDA(cudaGetDeviceCount(&devCount));
  for (int d : gpus) {
    if (d < 0 || d >= devCount) {
      std::cerr << "ERROR: gpu id " << d << " out of range, devCount=" << devCount << "\n";
      return 1;
    }
  }

  std::cout << "btag=" << btag << " nside=" << nside << " npix=" << npix << "\n";
  std::cout << "days=[" << start_day << "," << end_day << "] dcf_days=[" << dcf_start_day << "," << dcf_end_day << "] in_dir=" << in_dir << " out_dir=" << out_dir << "\n";
  std::cout << "gpus=" << gpus_s << " uvw_max=" << uvw_max << " blockage=" << blockage << "\n";

  // blockage phi（严格按 MATLAB）
  float phi0 = 0.0f;
  float Rmoon = 1737.1e3f;
  float h_moon = 300e3f;
  float theta0 = asinf(Rmoon / (Rmoon + h_moon));
  phi0 = (float)M_PI - theta0;
  std::cout << "blockage phi=" << phi0 << "\n";

  float frequency = (btag == "1M") ? 1.0e6f : 1.0e7f;
  float lamda = 3.0e8f / frequency;
  float bl_max = 100.0e3f;
  int nr = (int)ceilf(bl_max / lamda * 2.0f);
  std::cout << "dcf nr=" << nr << "\n";

  HostTimer t_total; t_total.tic();

  // Load B/theta/phi
  HostTimer t_io; t_io.tic();
  std::vector<float> hB(npix);
  std::vector<float> hTheta(npix);
  std::vector<float> hPhi(npix);

  std::string fB = in_dir + "/B_" + btag + ".txt";
  std::string fT = in_dir + "/theta_heal_" + btag + ".txt";
  std::string fP = in_dir + "/phi_heal_" + btag + ".txt";

  if (!load_single_noheader(fB, hB.data(), npix) ||
      !load_single_noheader(fT, hTheta.data(), npix) ||
      !load_single_noheader(fP, hPhi.data(), npix)) {
    std::cerr << "ERROR reading B/theta/phi for npix=" << npix << "\n"
              << fB << "\n" << fT << "\n" << fP << "\n";
    return 1;
  }

  float s_scale = 4.0f * (float)M_PI / (double)npix;
  for (long long i=0;i<npix;i++) hB[i] *= s_scale;

  std::cout << "load B/theta/phi OK, s=" << s_scale << ", time=" << t_io.toc_s() << " s\n";

  HostTimer t_dcf; t_dcf.tic();
  std::vector<float> h_dcf((size_t)nr + 1, 0.0f);
  std::vector<unsigned long long> mb((size_t)nr, 0ULL);
  std::vector<float> h_bll_scan((size_t)uvw_max);
  std::string bll_suf = "day" + btag + ".txt";

  for (int day = dcf_start_day; day <= dcf_end_day; ++day) 
  {
    std::string fbll = in_dir + "/bll" + std::to_string(day) + bll_suf;
    int bll_n = 0;
    if (!load_single_skip_first(fbll, h_bll_scan.data(), uvw_max, bll_n)) {
      std::cerr << "[dcf] ERROR read " << fbll << "\n";
      return 1;
    }
    for (int i=0; i<bll_n; ++i) {
      float x = (h_bll_scan[i] - 0.25f) * 2.0f;
      int g = (int)ceilf(x);
      if (g < 0) g = 0;
      if (g > nr) g = nr;
      if (g >= 1 && g <= nr) mb[(size_t)g - 1]++;
    }
  }

  h_dcf[0] = 1.0f / ((float)M_PI * 4.0f);
  for (int idx=1; idx<=nr; ++idx) {
    unsigned long long cnt = mb[(size_t)idx - 1];
    if (cnt == 0ULL) {
      h_dcf[(size_t)idx] = 0.0f;
      continue;
    }
    float x = 0.5f * (float)idx + 0.25f;
    float diff = 1.5f * x * x - 0.75f * x + 0.125f;
    h_dcf[(size_t)idx] = (2.0f / 3.0f) * (float)M_PI * diff / (float)cnt;
  }
  std::cout << "dcf precompute time=" << t_dcf.toc_s() << " s\n";

  int G = (int)gpus.size();
  std::vector<GpuCtx> ctx(G);

  long long chunk_size = (npix + G - 1) / G;

  // init per GPU
  HostTimer t_init; t_init.tic();
  #pragma omp parallel for num_threads(G)
  for (int gi=0; gi<G; ++gi) {
    int dev = gpus[gi];
    CHECK_CUDA(cudaSetDevice(dev));

    long long pix0 = (long long)gi * chunk_size;
    long long pix1 = std::min(npix, pix0 + chunk_size);
    long long n_chunk = std::max(0LL, pix1 - pix0);

    ctx[gi].dev = dev;
    ctx[gi].pix0 = pix0;
    ctx[gi].pix1 = pix1;
    ctx[gi].n_chunk = n_chunk;
    ctx[gi].UVW_MAX = uvw_max;

    CHECK_CUDA(cudaStreamCreateWithFlags(&ctx[gi].stream, cudaStreamNonBlocking));

    CHECK_CUDA(cudaMalloc(&ctx[gi].d_B,     (size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_theta, (size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_phi,   (size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_l,     (size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_m,     (size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_n,     (size_t)n_chunk*sizeof(float)));

    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_B,     hB.data() + pix0,     (size_t)n_chunk*sizeof(float), cudaMemcpyHostToDevice, ctx[gi].stream));
    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_theta, hTheta.data() + pix0, (size_t)n_chunk*sizeof(float), cudaMemcpyHostToDevice, ctx[gi].stream));
    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_phi,   hPhi.data() + pix0,   (size_t)n_chunk*sizeof(float), cudaMemcpyHostToDevice, ctx[gi].stream));

    const int BLOCK = 256;
    int grid = (int)((n_chunk + BLOCK - 1) / BLOCK);
    healpix_lmn_from_theta_phi_chunk<<<grid, BLOCK, 0, ctx[gi].stream>>>(ctx[gi].d_theta, ctx[gi].d_phi, ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n, n_chunk);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaStreamSynchronize(ctx[gi].stream));

    CHECK_CUDA(cudaFree(ctx[gi].d_theta)); ctx[gi].d_theta=nullptr;
    CHECK_CUDA(cudaFree(ctx[gi].d_phi));   ctx[gi].d_phi=nullptr;

    CHECK_CUDA(cudaMalloc(&ctx[gi].d_u, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_v, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_w, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_bll, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_gs, (size_t)uvw_max*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_dg, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_weight, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_dcf, ((size_t)nr + 1)*sizeof(float)));
    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_dcf, h_dcf.data(), ((size_t)nr + 1)*sizeof(float), cudaMemcpyHostToDevice, ctx[gi].stream));

    CHECK_CUDA(cudaMalloc(&ctx[gi].d_xyz1a, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_xyz1b, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_xyz1c, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_xyz2a, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_xyz2b, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_xyz2c, (size_t)uvw_max*sizeof(float)));

    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Viss, (size_t)uvw_max*sizeof(float2)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Creal, (size_t)n_chunk*sizeof(float)));

    int max_amount = uvw_max;
    CHECK_CUDA(cudaMallocHost(&ctx[gi].h_Vpart, (size_t)max_amount*sizeof(float2)));

    const int CHUNK = 1 << 20;
    CHECK_CUDA(cudaMallocHost(&ctx[gi].h_chunk, (size_t)CHUNK*sizeof(float)));

  }
  std::cout << "GPU init time=" << t_init.toc_s() << " s\n";

  // free host sky arrays after uploading
  hB.clear(); hB.shrink_to_fit();
  hTheta.clear(); hTheta.shrink_to_fit();
  hPhi.clear(); hPhi.shrink_to_fit();

  // host day buffers
  std::vector<float> hu(uvw_max), hv(uvw_max), hw(uvw_max), hbll(uvw_max), tmpf(uvw_max);
  std::vector<float> hxyz1a(uvw_max), hxyz1b(uvw_max), hxyz1c(uvw_max);
  std::vector<float> hxyz2a(uvw_max), hxyz2b(uvw_max), hxyz2c(uvw_max);

  std::vector<float2> hViss(uvw_max);
  std::vector<float> hWeight(uvw_max);

  static const size_t OUT_BUF_SZ = 8 << 20;
  static thread_local std::vector<char> outbuf(OUT_BUF_SZ);

  for (int day = start_day; day <= end_day; ++day) {
    HostTimer t_day; t_day.tic();

    // read day files
    HostTimer t_day_io; t_day_io.tic();

    std::string suf = "day" + btag + ".txt";
    std::string fuvw  = in_dir + "/updated_uvw" + std::to_string(day) + suf;
    std::string fxyz1 = in_dir + "/xyza" + std::to_string(day) + suf;
    std::string fxyz2 = in_dir + "/xyzb" + std::to_string(day) + suf;
    std::string fbll  = in_dir + "/bll" + std::to_string(day) + suf;

    int uvw_index=0, xyz1_index=0, xyz2_index=0, bll_index=0;

    if (!load_quad_skip_first(fuvw, hu.data(), hv.data(), hw.data(), tmpf.data(), uvw_max, uvw_index)) {
      std::cerr << "[day " << day << "] ERROR read " << fuvw << "\n";
      continue;
    }
    if (!load_triplets_skip_first(fxyz1, hxyz1a.data(), hxyz1b.data(), hxyz1c.data(), uvw_max, xyz1_index) ||
        !load_triplets_skip_first(fxyz2, hxyz2a.data(), hxyz2b.data(), hxyz2c.data(), uvw_max, xyz2_index) ||
        !load_single_skip_first(fbll, hbll.data(), uvw_max, bll_index)) {
      std::cerr << "[day " << day << "] ERROR read xyz/bll files\n";
      continue;
    }
    if (uvw_index <= 0 || xyz1_index != uvw_index || xyz2_index != uvw_index || bll_index != uvw_index) {
      std::cerr << "[day " << day << "] ERROR index mismatch uvw=" << uvw_index
                << " xyz1=" << xyz1_index << " xyz2=" << xyz2_index << " bll=" << bll_index << "\n";
      continue;
    }
    int amount = uvw_index;

    double day_io_s = t_day_io.toc_s();

    // H2D broadcast uvw
    HostTimer t_h2d; t_h2d.tic();
    #pragma omp parallel for num_threads(G)
    for (int gi=0; gi<G; ++gi) {
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      cudaStream_t stream = ctx[gi].stream;

      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_u, hu.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_v, hv.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_w, hw.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_bll, hbll.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));

      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_xyz1a, hxyz1a.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_xyz1b, hxyz1b.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_xyz1c, hxyz1c.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_xyz2a, hxyz2a.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_xyz2b, hxyz2b.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_xyz2c, hxyz2c.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));

      CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    double h2d_s = t_h2d.toc_s();

    // Viss partial + reduce
    HostTimer t_viss; t_viss.tic();
    #pragma omp parallel for num_threads(G)
    for (int gi=0; gi<G; ++gi) {
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      cudaStream_t stream = ctx[gi].stream;

      float2* d_Vpart = ctx[gi].d_Viss; // scratch

      constexpr int BASE_BLOCK = 256;
      int grid = (amount + BASE_BLOCK - 1) / BASE_BLOCK;
      constexpr int TILE_PIX = 256;
      size_t shmem = (size_t)TILE_PIX * 4 * sizeof(float);

      viss_partial<TILE_PIX><<<grid, BASE_BLOCK, shmem, stream>>>(
        ctx[gi].d_B, ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n, ctx[gi].n_chunk,
        ctx[gi].d_u, ctx[gi].d_v, ctx[gi].d_w,
        ctx[gi].d_xyz1a, ctx[gi].d_xyz1b, ctx[gi].d_xyz1c,
        ctx[gi].d_xyz2a, ctx[gi].d_xyz2b, ctx[gi].d_xyz2c,
        phi0,
        amount,
        d_Vpart
      );
      CHECK_CUDA(cudaPeekAtLastError());
      CHECK_CUDA(cudaStreamSynchronize(stream));

      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].h_Vpart, d_Vpart, (size_t)amount*sizeof(float2), cudaMemcpyDeviceToHost, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    for (int i=0;i<amount;i++) {
      double re = 0.0, im = 0.0;
      for (int gi=0; gi<G; ++gi) {
        re += (double)ctx[gi].h_Vpart[i].x;
        im += (double)ctx[gi].h_Vpart[i].y;
      }
      hViss[i] = make_float2((float)re, (float)im);
    }
    double viss_s = t_viss.toc_s();

    // MATLAB: Viss *= exp(-i*2*pi*w), dg, gs, weight=min(dcf(gs)*dg,1/8), then broadcast
    HostTimer t_vfix; t_vfix.tic();
    {
      int gi0 = 0;
      CHECK_CUDA(cudaSetDevice(ctx[gi0].dev));
      cudaStream_t stream = ctx[gi0].stream;

      CHECK_CUDA(cudaMemcpyAsync(ctx[gi0].d_Viss, hViss.data(), (size_t)uvw_index*sizeof(float2), cudaMemcpyHostToDevice, stream));

      const int BLOCK = 256;
      int grid = (uvw_index + BLOCK - 1) / BLOCK;

      phase_correct_and_dg<<<grid, BLOCK, 0, stream>>>(ctx[gi0].d_Viss, ctx[gi0].d_w, ctx[gi0].d_bll, ctx[gi0].d_dg, uvw_index);
      CHECK_CUDA(cudaPeekAtLastError());

      ceilAndScale_opt<<<grid, BLOCK, 0, stream>>>(ctx[gi0].d_bll, ctx[gi0].d_gs, uvw_index, nr);
      CHECK_CUDA(cudaPeekAtLastError());

      precompute_weight<<<grid, BLOCK, 0, stream>>>(ctx[gi0].d_weight, ctx[gi0].d_dcf, ctx[gi0].d_dg, ctx[gi0].d_gs, uvw_index, nr);
      CHECK_CUDA(cudaPeekAtLastError());

      CHECK_CUDA(cudaStreamSynchronize(stream));

      CHECK_CUDA(cudaMemcpyAsync(hViss.data(), ctx[gi0].d_Viss, (size_t)uvw_index*sizeof(float2), cudaMemcpyDeviceToHost, stream));
      CHECK_CUDA(cudaMemcpyAsync(hWeight.data(), ctx[gi0].d_weight, (size_t)uvw_index*sizeof(float), cudaMemcpyDeviceToHost, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    #pragma omp parallel for num_threads(G)
    for (int gi=0; gi<G; ++gi) {
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      cudaStream_t stream = ctx[gi].stream;
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_Viss, hViss.data(), (size_t)uvw_index*sizeof(float2), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_weight, hWeight.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    double vfix_s = t_vfix.toc_s();

    // computeC
    HostTimer t_C; t_C.tic();
    #pragma omp parallel for num_threads(G)
    for (int gi=0; gi<G; ++gi) {
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      cudaStream_t stream = ctx[gi].stream;

      const int BLOCK = 256;
      int grid = (int)((ctx[gi].n_chunk + BLOCK - 1) / BLOCK);
      constexpr int TILE_UVW = 256;

      // su,sv,sw,weight (4) + xyz1(3) + xyz2(3) + inv1,inv2 (2) = 12 float arrays + Viss float2
      size_t shmem = (size_t)TILE_UVW * (12 * sizeof(float) + sizeof(float2));

      computeC_real_chunk<TILE_UVW><<<grid, BLOCK, shmem, stream>>>(
        ctx[gi].n_chunk, ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n,
        ctx[gi].d_u, ctx[gi].d_v, ctx[gi].d_w,
        ctx[gi].d_weight,
        ctx[gi].d_xyz1a, ctx[gi].d_xyz1b, ctx[gi].d_xyz1c,
        ctx[gi].d_xyz2a, ctx[gi].d_xyz2b, ctx[gi].d_xyz2c,
        phi0, blockage,
        ctx[gi].d_Viss, uvw_index,
        ctx[gi].d_Creal
      );
      CHECK_CUDA(cudaPeekAtLastError());
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    double C_s = t_C.toc_s();

    // write output
    HostTimer t_w; t_w.tic();
    std::string out_path = out_dir + "C" + std::to_string(day) + "day" + btag + ".txt";
    std::ofstream ofs(out_path);
    if (!ofs.is_open()) {
      std::cerr << "[day " << day << "] ERROR open output: " << out_path << "\n";
    } else {
      ofs.rdbuf()->pubsetbuf(outbuf.data(), outbuf.size());
      const int CHUNK = 1 << 20;

      for (int gi=0; gi<G; ++gi) {
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        for (long long off=0; off<ctx[gi].n_chunk; off += CHUNK) {
          int cur = (int)std::min((long long)CHUNK, ctx[gi].n_chunk - off);
          CHECK_CUDA(cudaMemcpy(ctx[gi].h_chunk, ctx[gi].d_Creal + off, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost));
          for (int i=0;i<cur;i++) ofs << ctx[gi].h_chunk[i] << "\n";
        }
      }
      ofs.close();
    }
    double w_s = t_w.toc_s();

    double day_s = t_day.toc_s();
    std::cout << "[day " << day << "] uvw_index=" << uvw_index
              << " io=" << day_io_s << "s"
              << " h2d=" << h2d_s << "s"
              << " Vpart+reduce=" << viss_s << "s"
              << " Vphase+bcast=" << vfix_s << "s"
              << " C=" << C_s << "s"
              << " write=" << w_s << "s"
              << " total=" << day_s << "s\n";
  }

  // cleanup
  #pragma omp parallel for num_threads(G)
  for (int gi=0; gi<G; ++gi) {
    CHECK_CUDA(cudaSetDevice(ctx[gi].dev));

    if (ctx[gi].h_chunk) CHECK_CUDA(cudaFreeHost(ctx[gi].h_chunk));
    if (ctx[gi].h_Vpart) CHECK_CUDA(cudaFreeHost(ctx[gi].h_Vpart));

    if (ctx[gi].d_Creal) CHECK_CUDA(cudaFree(ctx[gi].d_Creal));
    if (ctx[gi].d_Viss)  CHECK_CUDA(cudaFree(ctx[gi].d_Viss));

    if (ctx[gi].d_xyz2c) CHECK_CUDA(cudaFree(ctx[gi].d_xyz2c));
    if (ctx[gi].d_xyz2b) CHECK_CUDA(cudaFree(ctx[gi].d_xyz2b));
    if (ctx[gi].d_xyz2a) CHECK_CUDA(cudaFree(ctx[gi].d_xyz2a));
    if (ctx[gi].d_xyz1c) CHECK_CUDA(cudaFree(ctx[gi].d_xyz1c));
    if (ctx[gi].d_xyz1b) CHECK_CUDA(cudaFree(ctx[gi].d_xyz1b));
    if (ctx[gi].d_xyz1a) CHECK_CUDA(cudaFree(ctx[gi].d_xyz1a));
    if (ctx[gi].d_dcf)   CHECK_CUDA(cudaFree(ctx[gi].d_dcf));
    if (ctx[gi].d_weight) CHECK_CUDA(cudaFree(ctx[gi].d_weight));
    if (ctx[gi].d_dg)    CHECK_CUDA(cudaFree(ctx[gi].d_dg));
    if (ctx[gi].d_gs)    CHECK_CUDA(cudaFree(ctx[gi].d_gs));
    if (ctx[gi].d_bll)   CHECK_CUDA(cudaFree(ctx[gi].d_bll));

    if (ctx[gi].d_w) CHECK_CUDA(cudaFree(ctx[gi].d_w));
    if (ctx[gi].d_v) CHECK_CUDA(cudaFree(ctx[gi].d_v));
    if (ctx[gi].d_u) CHECK_CUDA(cudaFree(ctx[gi].d_u));

    if (ctx[gi].d_n) CHECK_CUDA(cudaFree(ctx[gi].d_n));
    if (ctx[gi].d_m) CHECK_CUDA(cudaFree(ctx[gi].d_m));
    if (ctx[gi].d_l) CHECK_CUDA(cudaFree(ctx[gi].d_l));
    if (ctx[gi].d_B) CHECK_CUDA(cudaFree(ctx[gi].d_B));

    if (ctx[gi].stream) CHECK_CUDA(cudaStreamDestroy(ctx[gi].stream));
  }

  std::cout << "TOTAL time=" << t_total.toc_s() << " s\n";
  return 0;
}
