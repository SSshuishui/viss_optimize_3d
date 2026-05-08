// recon_3d_direct_viss_conj.cu
//
// Based on recon_seg_clip_viss_conj_viss_seg.cu, modified as follows:
//   Stage 1: keep the same full-precision 3D Viss simulation idea,
//            and still exploit the 56-row half-symmetry:
//            each 56-row time group contains 28 baselines and their negatives,
//            so only the first half is integrated, and the second half is obtained
//            by conjugation after phase correction.
//   Stage 2: replace the segmented short-time 2D uv-grid reconstruction with
//            direct 3D back-projection corresponding to the provided MATLAB code:
//              C(p) += sum_i Viss(i) * dcf(gs_i) * gdg_i * vismask(p,i)
//                               * exp(+j 2*pi * (u_i*l_p + v_i*m_p + w_i*n_p))
//            where Viss has already been phase-corrected by exp(-j 2*pi w).
//
// Notes:
//   1) This file reconstructs ONE DAY at a time, but dcf can be computed from
//      multiple days through --dcf_days. This mirrors the MATLAB logic where mb
//      is accumulated over many days before reconstructing one day.
//   2) bll is NOT loaded from file; it is derived as sqrt(u^2 + v^2 + w^2).
//   3) Default output is the real part only (matching most existing image-output
//      usage in the current CUDA pipeline). If you later need complex C exactly,
//      the stage-2 kernel can be extended to accumulate imag as well.
//
// Build:
//   nvcc -O3 --use_fast_math -lineinfo -Xcompiler -fopenmp -std=c++17 \
//     recon_3d_direct_viss_conj.cu -o recon_3d_direct_viss_conj
//
// Example:
//   ./recon_3d_direct_viss_conj --btag=1M --nside=512 --day=1 --dcf_days=450 \
//      --in_dir=./earth_1Mhz_cuda --sky_dir=./earth_1Mhz --out_dir=./out3d/ \
//      --gpus=0,1,2,3 --uvw_max=4500000 --write_viss=1
//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <system_error>
#include <limits>

#include <cuda_runtime.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CHECK_CUDA(call) do { \
  cudaError_t e=(call); \
  if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while(0)

static constexpr int SIGNED_BASELINES_PER_T = 56;
static constexpr int UNIQUE_BASELINES_PER_T = 28;

struct HostTimer {
  using clock=std::chrono::high_resolution_clock;
  clock::time_point t0;
  void tic(){t0=clock::now();}
  double toc_s() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(clock::now()-t0).count();
  }
};

static inline std::string get_arg(int argc,char** argv,const std::string& key,const std::string& defv){
  for(int i=1;i<argc;i++){
    std::string s(argv[i]);
    if(s.rfind(key+"=",0)==0) return s.substr(key.size()+1);
  }
  return defv;
}
static inline int to_int(const std::string& s,int defv){ try{return std::stoi(s);}catch(...){return defv;} }
static inline std::vector<int> parse_gpus(const std::string& s){
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while(std::getline(ss,tok,',')) if(!tok.empty()) out.push_back(std::stoi(tok));
  if(out.empty()) out.push_back(0);
  return out;
}

static inline std::string norm_dir(std::string p){
  while(!p.empty() && p.back()=='/') p.pop_back();
  return p;
}
static inline void ensure_dir(const std::string& path){
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  if(ec && !std::filesystem::exists(path)){
    std::cerr << "ERROR: create_directories failed for " << path
              << ", message=" << ec.message() << "\n";
    std::exit(1);
  }
}

static bool load_single_auto_np(const std::string& path, float* out, long long n_expected) {
  FILE* fp = fopen(path.c_str(), "r");
  if (!fp) return false;
  char line[256];
  if (!fgets(line, sizeof(line), fp)) { fclose(fp); return false; }

  char* p = line;
  float first = strtof(p, &p);
  long long i = 0;

  long long maybeN = (long long)llround((double)first);
  bool is_header = (std::llabs(maybeN - n_expected) == 0);
  if (!is_header) out[i++] = first;

  while (i < n_expected && fgets(line, sizeof(line), fp)) {
    char* q = line;
    out[i++] = strtof(q, &q);
  }
  fclose(fp);
  return (i == n_expected);
}

static bool load_triplets(const std::string& path,float* a,float* b,float* c,int maxN,int& outN){
  FILE* fp=fopen(path.c_str(),"r");
  if(!fp) return false;
  char line[512];
  int n=0;
  while(n<maxN && fgets(line,sizeof(line),fp)){
    char* p=line;
    a[n]=strtof(p,&p);
    b[n]=strtof(p,&p);
    c[n]=strtof(p,&p);
    n++;
  }
  fclose(fp);
  outN=n;
  return true;
}

static inline int half_to_full_pos_idx(int ih){
  int group = ih / UNIQUE_BASELINES_PER_T;
  int j = ih - group * UNIQUE_BASELINES_PER_T;
  return group * SIGNED_BASELINES_PER_T + j;
}
static inline int half_to_full_neg_idx(int ih){
  return half_to_full_pos_idx(ih) + UNIQUE_BASELINES_PER_T;
}

static inline bool nearly_neg_pair(float a, float b, float atol=1e-5f, float rtol=1e-5f){
  float ref = fmaxf(fabsf(a), fabsf(b));
  float tol = atol + rtol * ref;
  return fabsf(a + b) <= tol;
}

static bool validate_uvw_halfsym_layout(const std::vector<float>& u,
                                        const std::vector<float>& v,
                                        const std::vector<float>& w,
                                        int N)
{
  if(N % SIGNED_BASELINES_PER_T != 0) return false;
  int T = N / SIGNED_BASELINES_PER_T;
  for(int t=0; t<T; ++t){
    int base = t * SIGNED_BASELINES_PER_T;
    for(int j=0; j<UNIQUE_BASELINES_PER_T; ++j){
      int i0 = base + j;
      int i1 = base + UNIQUE_BASELINES_PER_T + j;
      if(!nearly_neg_pair(u[i0], u[i1]) ||
         !nearly_neg_pair(v[i0], v[i1]) ||
         !nearly_neg_pair(w[i0], w[i1])){
        std::cerr << "ERROR: uvw half-symmetry check failed at group=" << t
                  << " local=" << j
                  << " u=(" << u[i0] << "," << u[i1] << ")"
                  << " v=(" << v[i0] << "," << v[i1] << ")"
                  << " w=(" << w[i0] << "," << w[i1] << ")\n";
        return false;
      }
    }
  }
  return true;
}

static void phase_correct_viss_halfsym_host(float2* Viss_half, const float* w_full, int N_half){
  for(int ih=0; ih<N_half; ++ih){
    int i = half_to_full_pos_idx(ih);
    float ang = -2.0f*(float)M_PI*w_full[i];
    float c = std::cos(ang);
    float s = std::sin(ang);
    float a = Viss_half[ih].x, b = Viss_half[ih].y;
    Viss_half[ih] = make_float2(a*c - b*s, a*s + b*c);
  }
}

static void expand_viss_halfsym_to_full(const std::vector<float2>& hViss_half,
                                        std::vector<float2>& hViss_full)
{
  int N_half = (int)hViss_half.size();
  for(int ih=0; ih<N_half; ++ih){
    int ip = half_to_full_pos_idx(ih);
    int in = half_to_full_neg_idx(ih);
    float2 z = hViss_half[ih];
    hViss_full[ip] = z;
    hViss_full[in] = make_float2(z.x, -z.y);
  }
}

__device__ __forceinline__ void sincos_fast(float x,float* s,float* c){ __sincosf(x,s,c); }

__global__ void healpix_lmn_from_theta_phi_chunk(
    const float* __restrict__ theta,
    const float* __restrict__ phi,
    float* __restrict__ l,
    float* __restrict__ m,
    float* __restrict__ n,
    long long n_chunk)
{
  long long idx=(long long)blockIdx.x*blockDim.x + threadIdx.x;
  if(idx>=n_chunk) return;
  float th=theta[idx];
  float ph=phi[idx];
  th=(float)M_PI*0.5f - th;
  if(ph>(float)M_PI) ph-=2.0f*(float)M_PI;
  ph=-ph;
  float st,ct,sp,cp;
  sincos_fast(th,&st,&ct);
  sincos_fast(ph,&sp,&cp);
  l[idx]=ct*cp;
  m[idx]=ct*sp;
  n[idx]=st;
}

__global__ void invnorm3_kernel(const float* x,const float* y,const float* z,float* invn,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){
    float a=x[i], b=y[i], c=z[i];
    invn[i]=rsqrtf(a*a+b*b+c*c);
  }
}

template<int TILE_PIX, bool DO_BLOCKAGE>
__global__ void viss_partial_all_halfsym(
    const float* __restrict__ B,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    long long n_chunk,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ x1,
    const float* __restrict__ y1,
    const float* __restrict__ z1,
    const float* __restrict__ invn1,
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    const float* __restrict__ invn2,
    int N_half,
    float cosphi,
    float2* __restrict__ Vpart)
{
  int ih = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = (ih < N_half);

  int i = 0;
  float u0 = 0.0f, v0 = 0.0f, w0 = 0.0f;
  float x1i = 0.0f, y1i = 0.0f, z1i = 0.0f;
  float x2i = 0.0f, y2i = 0.0f, z2i = 0.0f;
  float in1 = 0.0f, in2 = 0.0f;

  if(active){
    int group = ih / UNIQUE_BASELINES_PER_T;
    int j     = ih - group * UNIQUE_BASELINES_PER_T;
    i         = group * SIGNED_BASELINES_PER_T + j;

    u0 = u[i];
    v0 = v[i];
    w0 = w[i];

    if constexpr (DO_BLOCKAGE){
      x1i = x1[i]; y1i = y1[i]; z1i = z1[i]; in1 = invn1[i];
      x2i = x2[i]; y2i = y2[i]; z2i = z2[i]; in2 = invn2[i];
    }
  }

  float acc_re = 0.0f, acc_im = 0.0f;
  const float k = -2.0f * (float)M_PI;

  extern __shared__ float smem[];
  float* sB = smem;
  float* sL = sB + TILE_PIX;
  float* sM = sL + TILE_PIX;
  float* sN = sM + TILE_PIX;

  for(long long p0 = 0; p0 < n_chunk; p0 += TILE_PIX){
    for(int lane = threadIdx.x; lane < TILE_PIX; lane += blockDim.x){
      long long p = p0 + lane;
      if(p < n_chunk){
        sB[lane] = B[p];
        sL[lane] = l[p];
        sM[lane] = m[p];
        sN[lane] = n[p];
      }else{
        sB[lane] = 0.0f;
        sL[lane] = 0.0f;
        sM[lane] = 0.0f;
        sN[lane] = 0.0f;
      }
    }
    __syncthreads();

    if(active){
      int tileN = (int)min((long long)TILE_PIX, n_chunk - p0);

      #pragma unroll 4
      for(int k0 = 0; k0 < tileN; k0++){
        float lp  = sL[k0];
        float mp  = sM[k0];
        float npv = sN[k0];

        if constexpr (DO_BLOCKAGE){
          float c1 = (lp * x1i + mp * y1i + npv * z1i) * in1;
          float c2 = (lp * x2i + mp * y2i + npv * z2i) * in2;
          if(c1 < cosphi || c2 < cosphi) continue;
        }

        float phase = u0 * lp + v0 * mp + w0 * (npv - 1.0f);
        float ang   = k * phase;

        float s, c;
        sincos_fast(ang, &s, &c);

        float bp = sB[k0];
        acc_re += bp * c;
        acc_im += bp * s;
      }
    }
    __syncthreads();
  }

  if(active){
    Vpart[ih] = make_float2(acc_re, acc_im);
  }
}

__global__ void compute_bll_and_weight(
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    int N,
    const float* __restrict__ dcf,
    int dcf_len,
    float* __restrict__ bll,
    float* __restrict__ base_weight)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= N) return;

  float uu = u[i], vv = v[i], ww = w[i];
  float bl = sqrtf(uu*uu + vv*vv + ww*ww);
  bll[i] = bl;

  if(!isfinite(bl) || bl <= 1e-20f){
    base_weight[i] = (dcf_len > 0 ? dcf[0] : 0.0f);
    return;
  }

  int gs = (int)ceilf((bl - 0.25f) / 0.5f) + 1;  // MATLAB: ceil((bll-1/4)/0.5)+1
  if(gs < 0) gs = 0;
  if(gs >= dcf_len) gs = dcf_len - 1;

  float s = ww / bl;
  s = fminf(1.0f, fmaxf(-1.0f, s));
  float c = sqrtf(fmaxf(0.0f, 1.0f - s*s));

  float a = 1.0f - 4.0f * s * s;
  float mag_sqrt = sqrtf(fabsf(a));
  float gdg = 0.0f;
  if(c > 1e-12f) gdg = 1.5f * mag_sqrt / c;

  float wgt = dcf[gs] * gdg;
  if(!isfinite(wgt) || wgt < 0.0f) wgt = 0.0f;
  base_weight[i] = wgt;
}

template<int TILE_BL>
__global__ void recon_3d_direct_real(
    long long n_chunk,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ x1,
    const float* __restrict__ y1,
    const float* __restrict__ z1,
    const float* __restrict__ invn1,
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    const float* __restrict__ invn2,
    const float* __restrict__ base_weight,
    const float2* __restrict__ Viss,
    int N,
    float cosphi,
    float* __restrict__ Cout)
{
  __shared__ float su[TILE_BL], sv[TILE_BL], sw[TILE_BL];
  __shared__ float sx1[TILE_BL], sy1[TILE_BL], sz1[TILE_BL], sin1[TILE_BL];
  __shared__ float sx2[TILE_BL], sy2[TILE_BL], sz2[TILE_BL], sin2[TILE_BL];
  __shared__ float swgt[TILE_BL];
  __shared__ float2 sV[TILE_BL];

  long long pix = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(pix >= n_chunk) return;

  float lp = l[pix], mp = m[pix], npv = n[pix];
  const float TWO_PI = 6.2831853071795864769f;
  double acc_re = 0.0;

  for(int b0 = 0; b0 < N; b0 += TILE_BL){
    int tileN = min(TILE_BL, N - b0);
    for(int t = threadIdx.x; t < tileN; t += blockDim.x){
      int bi = b0 + t;
      su[t]   = u[bi];
      sv[t]   = v[bi];
      sw[t]   = w[bi];
      sx1[t]  = x1[bi]; sy1[t]  = y1[bi]; sz1[t]  = z1[bi]; sin1[t] = invn1[bi];
      sx2[t]  = x2[bi]; sy2[t]  = y2[bi]; sz2[t]  = z2[bi]; sin2[t] = invn2[bi];
      swgt[t] = base_weight[bi];
      sV[t]   = Viss[bi];
    }
    __syncthreads();

    #pragma unroll 4
    for(int t = 0; t < tileN; ++t){
      float c1 = (lp * sx1[t] + mp * sy1[t] + npv * sz1[t]) * sin1[t];
      float c2 = (lp * sx2[t] + mp * sy2[t] + npv * sz2[t]) * sin2[t];
      if(c1 < cosphi || c2 < cosphi) continue;

      float wgt = swgt[t];
      if(wgt <= 0.0f) continue;
      if(wgt > 0.125f) wgt = 0.125f;   // MATLAB: gdcf2(gdcf2>1/8)=1/8

      float phase = TWO_PI * (su[t] * lp + sv[t] * mp + sw[t] * npv);
      float s, c;
      sincos_fast(phase, &s, &c);

      float vr = sV[t].x;
      float vi = sV[t].y;
      if(!isfinite(vr) || !isfinite(vi)) continue;

      acc_re += (double)wgt * ((double)vr * (double)c - (double)vi * (double)s);
    }
    __syncthreads();
  }

  Cout[pix] = (float)acc_re;
}

struct GpuCtx {
  int dev=0;
  cudaStream_t stream=nullptr;
  long long pix0=0,pix1=0,n_chunk=0;

  float* d_B=nullptr;
  float* d_theta=nullptr;
  float* d_phi=nullptr;
  float* d_l=nullptr;
  float* d_m=nullptr;
  float* d_n=nullptr;

  int N=0;
  int N_half=0;
  float* d_u=nullptr; float* d_v=nullptr; float* d_w=nullptr;
  float* d_x1=nullptr; float* d_y1=nullptr; float* d_z1=nullptr;
  float* d_x2=nullptr; float* d_y2=nullptr; float* d_z2=nullptr;
  float* d_invn1=nullptr; float* d_invn2=nullptr;
  float* d_bll=nullptr;
  float* d_basew=nullptr;

  float2* d_Vpart=nullptr;
  float2* d_Viss=nullptr;
  float* d_dcf=nullptr;

  float* d_Cacc=nullptr;

  float2* h_Vpart=nullptr;
  float* h_chunk=nullptr;
};

static void compute_dcf_from_mb(const std::vector<long long>& mb, std::vector<float>& dcf){
  int nr = (int)mb.size();
  dcf.assign((size_t)nr + 1, 0.0f);
  dcf[0] = 4.0f / (float)M_PI; // MATLAB: 1/(pi/4)
  for(int p=1; p<=nr; ++p){
    double R = 0.5 * (double)p + 0.25;
    double num = (2.0/3.0) * M_PI * (R*R*R - (R - 0.5)*(R - 0.5)*(R - 0.5));
    long long cnt = mb[p-1];
    dcf[p] = (cnt > 0) ? (float)(num / (double)cnt) : 0.0f;
  }
}

static bool accumulate_mb_from_uvw_files(
    const std::string& in_dir,
    const std::string& btag,
    int days_total,
    int uvw_max,
    float bl_max,
    float lamda,
    std::vector<long long>& mb)
{
  int nr = (int)std::ceil(bl_max / lamda * 2.0f);
  mb.assign(nr, 0);

  std::vector<float> tu(uvw_max), tv(uvw_max), tw(uvw_max);
  for(int k=1; k<=days_total; ++k){
    std::string fuvw = in_dir + "/uvw" + std::to_string(k) + "day" + btag + ".txt";
    int Nk = 0;
    if(!load_triplets(fuvw, tu.data(), tv.data(), tw.data(), uvw_max, Nk)){
      std::cerr << "ERROR read for dcf: " << fuvw << "\n";
      return false;
    }
    for(int i=0; i<Nk; ++i){
      double uu = (double)tu[i], vv = (double)tv[i], ww = (double)tw[i];
      double bll = std::sqrt(uu*uu + vv*vv + ww*ww);
      int s = (int)std::ceil((bll - 0.25) / 0.5); // MATLAB s=ceil((bll-1/4)/0.5)
      if(s >= 1 && s <= nr) mb[s-1]++;
    }
    std::cout << "[dcf] scanned day " << k << "/" << days_total << ", Nk=" << Nk << "\n";
  }
  return true;
}

int main(int argc,char** argv){
  std::string btag=get_arg(argc,argv,"--btag","10M");
  int nside=to_int(get_arg(argc,argv,"--nside","0"),0);
  int day=to_int(get_arg(argc,argv,"--day","1"),1);
  int dcf_days=to_int(get_arg(argc,argv,"--dcf_days","1"),1);
  int write_viss=to_int(get_arg(argc,argv,"--write_viss","1"),1);

  std::string in_dir=norm_dir(get_arg(argc,argv,"--in_dir",""));
  std::string sky_dir=norm_dir(get_arg(argc,argv,"--sky_dir",""));
  std::string out_dir=get_arg(argc,argv,"--out_dir","./out3d/");
  std::string gpus_s=get_arg(argc,argv,"--gpus","0");

  int uvw_max=to_int(get_arg(argc,argv,"--uvw_max","4500000"),4500000);
  int print_stats=to_int(get_arg(argc,argv,"--print_stats","1"),1);

  if(btag!="1M" && btag!="10M" && btag!="30M"){
    std::cerr<<"ERROR --btag\n";
    return 1;
  }
  if(nside<=0){
    if(btag=="1M") nside=512;
    else if(btag=="10M") nside=4096;
    else nside=16384;
  }

  if(in_dir.empty()){
    if(btag=="1M") in_dir="./earth_1Mhz_cuda";
    else if(btag=="10M") in_dir="./earth_10Mhz_cuda";
    else in_dir="./earth_30Mhz_cuda";
  }
  if(sky_dir.empty()) sky_dir=(btag=="1M")? "./earth_1Mhz" : (btag=="10M"? "./earth_10Mhz":"./earth_30Mhz");
  if(!out_dir.empty() && out_dir.back()!='/') out_dir.push_back('/');
  ensure_dir(out_dir);

  // Interpret --gpus as PHYSICAL GPU ids, equivalent to CUDA_VISIBLE_DEVICES.
  // Example: --gpus=4,5,6,7  => program sees 4 logical devices: 0,1,2,3.
  auto physical_gpus=parse_gpus(gpus_s);
  if(physical_gpus.empty()) physical_gpus.push_back(0);
  for(int d: physical_gpus){
    if(d < 0){
      std::cerr << "ERROR negative gpu id in --gpus: " << d << "\n";
      return 1;
    }
  }
  setenv("CUDA_VISIBLE_DEVICES", gpus_s.c_str(), 1);

  int devCount=0;
  CHECK_CUDA(cudaGetDeviceCount(&devCount));
  int G=(int)physical_gpus.size();
  if(devCount != G){
    std::cerr << "WARNING: after applying CUDA_VISIBLE_DEVICES=" << gpus_s
              << ", cudaGetDeviceCount()=" << devCount
              << " while --gpus count=" << G << ". Continuing with the visible logical devices.\n";
    G = std::min(devCount, G);
  }
  if(G <= 0){
    std::cerr << "ERROR: no visible CUDA devices after applying --gpus=" << gpus_s << "\n";
    return 1;
  }
  std::vector<int> gpus(G);
  for(int i=0;i<G;++i) gpus[i]=i;

  long long npix=12LL*(long long)nside*(long long)nside;
  float R=1737.1e3f, h=300e3f;
  float theta=asinf(R/(R+h));
  float phi=(float)M_PI-theta;
  float cosphi=cosf(phi);

  float frequency=(btag=="1M")? 1e6f : (btag=="10M"? 1e7f : 3e7f);
  float lamda=3e8f/frequency;
  float bl_max=100e3f;

  std::cout<<"btag="<<btag<<" nside="<<nside<<" npix="<<npix<<" day="<<day
           <<" dcf_days="<<dcf_days<<"\n";
  std::cout<<"in_dir="<<in_dir<<" sky_dir="<<sky_dir<<" out_dir="<<out_dir
           <<" gpus="<<gpus_s<<" uvw_max="<<uvw_max<<"\n";
  std::cout<<"theta="<<theta<<" phi="<<phi<<" cosphi="<<cosphi<<" lamda="<<lamda<<"\n";

  // ---------------- load sky ----------------
  HostTimer t_io;
  t_io.tic();
  std::vector<float> hB(npix), hTheta(npix), hPhi(npix);

  std::string fB1=sky_dir+"/B_"+btag+".txt";
  std::string fT1=sky_dir+"/theta_heal_"+btag+".txt";
  std::string fP1=sky_dir+"/phi_heal_"+btag+".txt";
  std::string fB2=sky_dir+"/B.txt";
  std::string fT2=sky_dir+"/theta_heal.txt";
  std::string fP2=sky_dir+"/phi_heal.txt";

  bool ok=load_single_auto_np(fB1,hB.data(),npix)
       && load_single_auto_np(fT1,hTheta.data(),npix)
       && load_single_auto_np(fP1,hPhi.data(),npix);
  if(!ok){
    ok=load_single_auto_np(fB2,hB.data(),npix)
      && load_single_auto_np(fT2,hTheta.data(),npix)
      && load_single_auto_np(fP2,hPhi.data(),npix);
  }
  if(!ok){
    std::cerr<<"ERROR reading B/theta/phi\n";
    return 1;
  }
  std::cout<<"Loaded sky in "<<t_io.toc_s()<<" s\n";

  // C=single(B.*s);  
  const double pix_area = 4.0 * M_PI / (double)npix;
  #pragma omp parallel for
  for (long long i = 0; i < npix; ++i) {
    hB[i] = (float)((double)hB[i] * pix_area);
  }
  std::cout << " pix_area=" << pix_area
            << " inv_pix_area=" << (1.0 / pix_area)
            << "\n";

  // ---------------- load one-day baselines ----------------
  HostTimer t_bas;
  t_bas.tic();
  std::vector<float> hu(uvw_max), hv(uvw_max), hw(uvw_max);
  std::vector<float> hx1(uvw_max), hy1(uvw_max), hz1(uvw_max);
  std::vector<float> hx2(uvw_max), hy2(uvw_max), hz2(uvw_max);

  auto make_path=[&](const std::string& dir,const std::string& pre){
    return dir+"/"+pre+std::to_string(day)+"day"+btag+".txt";
  };
  std::string fuvw=make_path(in_dir,"uvw");
  std::string fxy1=make_path(in_dir,"xyza");
  std::string fxy2=make_path(in_dir,"xyzb");

  int N=0,N1=0,N2=0;
  if(!load_triplets(fuvw,hu.data(),hv.data(),hw.data(),uvw_max,N)){
    std::cerr<<"ERROR read "<<fuvw<<"\n";
    return 1;
  }
  if(!load_triplets(fxy1,hx1.data(),hy1.data(),hz1.data(),uvw_max,N1) ||
     !load_triplets(fxy2,hx2.data(),hy2.data(),hz2.data(),uvw_max,N2)){
    std::cerr<<"ERROR read xyz\n";
    return 1;
  }
  if(N<=0 || N1!=N || N2!=N){
    std::cerr<<"ERROR baseline count mismatch uvw="<<N<<" xyz1="<<N1<<" xyz2="<<N2<<"\n";
    return 1;
  }
  std::cout<<"Loaded baselines N="<<N<<" in "<<t_bas.toc_s()<<" s\n";

  if(N % SIGNED_BASELINES_PER_T != 0){
    std::cerr<<"ERROR: N="<<N<<" not divisible by "<<SIGNED_BASELINES_PER_T
             <<"; cannot use the 56-row half-symmetry assumption.\n";
    return 1;
  }
  if(!validate_uvw_halfsym_layout(hu, hv, hw, N)){
    std::cerr<<"ERROR: uvw rows do not satisfy per-56 half-symmetry; stop.\n";
    return 1;
  }

  int N_half = (N / SIGNED_BASELINES_PER_T) * UNIQUE_BASELINES_PER_T;
  std::cout << "N_half=" << N_half << " (using half-symmetry for Viss stage)\n";

  // ---------------- compute dcf from mb ----------------
  HostTimer t_dcf;
  t_dcf.tic();
  std::vector<long long> mb;
  if(!accumulate_mb_from_uvw_files(in_dir, btag, dcf_days, uvw_max, bl_max, lamda, mb)){
    return 1;
  }
  std::vector<float> hdcf;
  compute_dcf_from_mb(mb, hdcf);
  std::cout << "Computed dcf in " << t_dcf.toc_s() << " s, dcf_len=" << hdcf.size() << "\n";

  // ---------------- init per GPU ----------------
  std::vector<GpuCtx> ctx(G);
  long long chunk_size=(npix + G - 1)/G;
  HostTimer t_init;
  t_init.tic();

  #pragma omp parallel for num_threads(G)
  for(int gi=0; gi<G; ++gi){
    int dev=gpus[gi];
    CHECK_CUDA(cudaSetDevice(dev));

    long long pix0=(long long)gi*chunk_size;
    long long pix1=std::min(npix,pix0+chunk_size);
    long long n_chunk=std::max(0LL,pix1-pix0);

    ctx[gi].dev=dev;
    ctx[gi].pix0=pix0;
    ctx[gi].pix1=pix1;
    ctx[gi].n_chunk=n_chunk;
    ctx[gi].N=N;
    ctx[gi].N_half=N_half;
    CHECK_CUDA(cudaStreamCreateWithFlags(&ctx[gi].stream,cudaStreamNonBlocking));

    CHECK_CUDA(cudaMalloc(&ctx[gi].d_B,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_theta,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_phi,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_l,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_m,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_n,(size_t)n_chunk*sizeof(float)));

    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_B,hB.data()+pix0,(size_t)n_chunk*sizeof(float),cudaMemcpyHostToDevice,ctx[gi].stream));
    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_theta,hTheta.data()+pix0,(size_t)n_chunk*sizeof(float),cudaMemcpyHostToDevice,ctx[gi].stream));
    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_phi,hPhi.data()+pix0,(size_t)n_chunk*sizeof(float),cudaMemcpyHostToDevice,ctx[gi].stream));

    int BLOCK=256;
    int grid=(int)((n_chunk+BLOCK-1)/BLOCK);
    healpix_lmn_from_theta_phi_chunk<<<grid,BLOCK,0,ctx[gi].stream>>>(
      ctx[gi].d_theta,ctx[gi].d_phi,ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,n_chunk);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaStreamSynchronize(ctx[gi].stream));
    CHECK_CUDA(cudaFree(ctx[gi].d_theta)); ctx[gi].d_theta=nullptr;
    CHECK_CUDA(cudaFree(ctx[gi].d_phi));   ctx[gi].d_phi=nullptr;

    CHECK_CUDA(cudaMalloc(&ctx[gi].d_u,(size_t)N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_v,(size_t)N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_w,(size_t)N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_x1,(size_t)N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_y1,(size_t)N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_z1,(size_t)N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_x2,(size_t)N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_y2,(size_t)N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_z2,(size_t)N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_invn1,(size_t)N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_invn2,(size_t)N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_bll,(size_t)N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_basew,(size_t)N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_dcf,(size_t)hdcf.size()*sizeof(float)));

    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_u,hu.data(),(size_t)N*sizeof(float),cudaMemcpyHostToDevice,ctx[gi].stream));
    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_v,hv.data(),(size_t)N*sizeof(float),cudaMemcpyHostToDevice,ctx[gi].stream));
    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_w,hw.data(),(size_t)N*sizeof(float),cudaMemcpyHostToDevice,ctx[gi].stream));
    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_x1,hx1.data(),(size_t)N*sizeof(float),cudaMemcpyHostToDevice,ctx[gi].stream));
    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_y1,hy1.data(),(size_t)N*sizeof(float),cudaMemcpyHostToDevice,ctx[gi].stream));
    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_z1,hz1.data(),(size_t)N*sizeof(float),cudaMemcpyHostToDevice,ctx[gi].stream));
    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_x2,hx2.data(),(size_t)N*sizeof(float),cudaMemcpyHostToDevice,ctx[gi].stream));
    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_y2,hy2.data(),(size_t)N*sizeof(float),cudaMemcpyHostToDevice,ctx[gi].stream));
    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_z2,hz2.data(),(size_t)N*sizeof(float),cudaMemcpyHostToDevice,ctx[gi].stream));
    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_dcf,hdcf.data(),(size_t)hdcf.size()*sizeof(float),cudaMemcpyHostToDevice,ctx[gi].stream));

    int gN=(N+255)/256;
    invnorm3_kernel<<<gN,256,0,ctx[gi].stream>>>(ctx[gi].d_x1,ctx[gi].d_y1,ctx[gi].d_z1,ctx[gi].d_invn1,N);
    invnorm3_kernel<<<gN,256,0,ctx[gi].stream>>>(ctx[gi].d_x2,ctx[gi].d_y2,ctx[gi].d_z2,ctx[gi].d_invn2,N);
    compute_bll_and_weight<<<gN,256,0,ctx[gi].stream>>>(
      ctx[gi].d_u, ctx[gi].d_v, ctx[gi].d_w,
      N,
      ctx[gi].d_dcf, (int)hdcf.size(),
      ctx[gi].d_bll, ctx[gi].d_basew);
    CHECK_CUDA(cudaPeekAtLastError());

    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Vpart,(size_t)N_half*sizeof(float2)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Viss,(size_t)N*sizeof(float2)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Cacc,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Cacc, 0, (size_t)n_chunk*sizeof(float), ctx[gi].stream));

    CHECK_CUDA(cudaMallocHost(&ctx[gi].h_Vpart,(size_t)N_half*sizeof(float2)));
    CHECK_CUDA(cudaMallocHost(&ctx[gi].h_chunk,(size_t)std::min(1LL<<20, n_chunk>0?n_chunk:1LL)*sizeof(float)));

    CHECK_CUDA(cudaStreamSynchronize(ctx[gi].stream));
  }
  std::cout << "GPU init done in " << t_init.toc_s() << " s\n";

  // ---------------- stage 1: Viss ----------------
  HostTimer t_viss;
  t_viss.tic();
  constexpr int TILE_PIX = 256;
  const int BLOCKV = 256;
  int gridV = (N_half + BLOCKV - 1) / BLOCKV;
  size_t shmem = (size_t)TILE_PIX * 4 * sizeof(float);

  #pragma omp parallel for num_threads(G)
  for(int gi=0; gi<G; ++gi){
    CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
    cudaStream_t stream=ctx[gi].stream;

    viss_partial_all_halfsym<TILE_PIX,true><<<gridV,BLOCKV,shmem,stream>>>(
      ctx[gi].d_B,ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,ctx[gi].n_chunk,
      ctx[gi].d_u, ctx[gi].d_v, ctx[gi].d_w,
      ctx[gi].d_x1, ctx[gi].d_y1, ctx[gi].d_z1, ctx[gi].d_invn1,
      ctx[gi].d_x2, ctx[gi].d_y2, ctx[gi].d_z2, ctx[gi].d_invn2,
      N_half, cosphi,
      ctx[gi].d_Vpart
    );
    CHECK_CUDA(cudaPeekAtLastError());

    CHECK_CUDA(cudaMemcpyAsync(
      ctx[gi].h_Vpart,
      ctx[gi].d_Vpart,
      (size_t)N_half*sizeof(float2),
      cudaMemcpyDeviceToHost,
      stream
    ));
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  std::vector<float2> hViss_half(N_half);
  for(int i=0;i<N_half;i++){
    double re=0.0, im=0.0;
    for(int gi=0; gi<G; ++gi){
      re += (double)ctx[gi].h_Vpart[i].x;
      im += (double)ctx[gi].h_Vpart[i].y;
    }
    hViss_half[i]=make_float2((float)re,(float)im);
  }

  phase_correct_viss_halfsym_host(hViss_half.data(), hw.data(), N_half);

  std::vector<float2> hViss_full(N, make_float2(0,0));
  expand_viss_halfsym_to_full(hViss_half, hViss_full);

  #pragma omp parallel for num_threads(G)
  for(int gi=0; gi<G; ++gi){
    CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
    CHECK_CUDA(cudaMemcpyAsync(
      ctx[gi].d_Viss,
      hViss_full.data(),
      (size_t)N*sizeof(float2),
      cudaMemcpyHostToDevice,
      ctx[gi].stream
    ));
    CHECK_CUDA(cudaStreamSynchronize(ctx[gi].stream));
  }
  std::cout << "Stage-1 Viss done in " << t_viss.toc_s() << " s\n";

  if(write_viss){
    std::string fv=out_dir+"Viss"+std::to_string(day)+"day"+btag+".txt";
    std::ofstream ofsViss(fv);
    if(!ofsViss.is_open()){
      std::cerr<<"ERROR open "<<fv<<"\n";
      return 1;
    }
    static thread_local std::vector<char> vissbuf(8<<20);
    ofsViss.rdbuf()->pubsetbuf(vissbuf.data(), vissbuf.size());
    for(int i=0;i<N;i++) ofsViss << hViss_full[i].x << " " << hViss_full[i].y << "\n";
    ofsViss.close();
    std::cout << "Wrote phase-corrected Viss to " << fv << "\n";
  }

  if(print_stats){
    float umin=std::numeric_limits<float>::infinity(), vmin=umin, wmin=umin;
    float umax=-umin, vmax=-umin, wmax=-umin;
    for(int i=0;i<N;i++){
      umin=std::min(umin, hu[i]); umax=std::max(umax, hu[i]);
      vmin=std::min(vmin, hv[i]); vmax=std::max(vmax, hv[i]);
      wmin=std::min(wmin, hw[i]); wmax=std::max(wmax, hw[i]);
    }
    std::cout << "uvw stats: u=[" << umin << "," << umax << "] "
              << "v=[" << vmin << "," << vmax << "] "
              << "w=[" << wmin << "," << wmax << "]\n";
  }

  // ---------------- stage 2: direct 3D reconstruction ----------------
  HostTimer t_rec;
  t_rec.tic();
  #pragma omp parallel for num_threads(G)
  for(int gi=0; gi<G; ++gi){
    CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
    cudaStream_t stream = ctx[gi].stream;
    int BLOCK = 256;
    int grid = (int)((ctx[gi].n_chunk + BLOCK - 1) / BLOCK);

    recon_3d_direct_real<128><<<grid,BLOCK,0,stream>>>(
      ctx[gi].n_chunk,
      ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n,
      ctx[gi].d_u, ctx[gi].d_v, ctx[gi].d_w,
      ctx[gi].d_x1, ctx[gi].d_y1, ctx[gi].d_z1, ctx[gi].d_invn1,
      ctx[gi].d_x2, ctx[gi].d_y2, ctx[gi].d_z2, ctx[gi].d_invn2,
      ctx[gi].d_basew,
      ctx[gi].d_Viss,
      N,
      cosphi,
      ctx[gi].d_Cacc
    );
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }
  std::cout << "Stage-2 direct 3D recon done in " << t_rec.toc_s() << " s\n";

  // ---------------- write C ----------------
  HostTimer t_w;
  t_w.tic();
  std::string outC=out_dir+"C"+std::to_string(day)+"day"+btag+".txt";
  std::ofstream ofs(outC);
  if(!ofs.is_open()){
    std::cerr<<"ERROR open "<<outC<<"\n";
    return 1;
  }
  static thread_local std::vector<char> outbuf(8<<20);
  ofs.rdbuf()->pubsetbuf(outbuf.data(), outbuf.size());

  const long long CH=1LL<<20;
  for(int gi=0; gi<G; ++gi){
    CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
    for(long long off=0; off<ctx[gi].n_chunk; off+=CH){
      int cur=(int)std::min(CH, ctx[gi].n_chunk-off);
      CHECK_CUDA(cudaMemcpy(ctx[gi].h_chunk, ctx[gi].d_Cacc+off, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost));
      for(int i=0;i<cur;i++) ofs << ctx[gi].h_chunk[i] << "\n";
    }
  }
  ofs.close();
  std::cout << "Wrote " << outC << " in " << t_w.toc_s() << " s\n";

  // ---------------- cleanup ----------------
  #pragma omp parallel for num_threads(G)
  for(int gi=0; gi<G; ++gi){
    CHECK_CUDA(cudaSetDevice(ctx[gi].dev));

    if(ctx[gi].h_chunk) cudaFreeHost(ctx[gi].h_chunk);
    if(ctx[gi].h_Vpart) cudaFreeHost(ctx[gi].h_Vpart);

    if(ctx[gi].d_Cacc) cudaFree(ctx[gi].d_Cacc);
    if(ctx[gi].d_dcf) cudaFree(ctx[gi].d_dcf);
    if(ctx[gi].d_Viss) cudaFree(ctx[gi].d_Viss);
    if(ctx[gi].d_Vpart) cudaFree(ctx[gi].d_Vpart);
    if(ctx[gi].d_basew) cudaFree(ctx[gi].d_basew);
    if(ctx[gi].d_bll) cudaFree(ctx[gi].d_bll);

    if(ctx[gi].d_invn2) cudaFree(ctx[gi].d_invn2);
    if(ctx[gi].d_invn1) cudaFree(ctx[gi].d_invn1);
    if(ctx[gi].d_z2) cudaFree(ctx[gi].d_z2);
    if(ctx[gi].d_y2) cudaFree(ctx[gi].d_y2);
    if(ctx[gi].d_x2) cudaFree(ctx[gi].d_x2);
    if(ctx[gi].d_z1) cudaFree(ctx[gi].d_z1);
    if(ctx[gi].d_y1) cudaFree(ctx[gi].d_y1);
    if(ctx[gi].d_x1) cudaFree(ctx[gi].d_x1);
    if(ctx[gi].d_w) cudaFree(ctx[gi].d_w);
    if(ctx[gi].d_v) cudaFree(ctx[gi].d_v);
    if(ctx[gi].d_u) cudaFree(ctx[gi].d_u);

    if(ctx[gi].d_n) cudaFree(ctx[gi].d_n);
    if(ctx[gi].d_m) cudaFree(ctx[gi].d_m);
    if(ctx[gi].d_l) cudaFree(ctx[gi].d_l);
    if(ctx[gi].d_phi) cudaFree(ctx[gi].d_phi);
    if(ctx[gi].d_theta) cudaFree(ctx[gi].d_theta);
    if(ctx[gi].d_B) cudaFree(ctx[gi].d_B);

    if(ctx[gi].stream) cudaStreamDestroy(ctx[gi].stream);
  }

  return 0;
}
