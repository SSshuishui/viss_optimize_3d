// optimized_viss_recon_multi_freq.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <chrono>
#include <omp.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include "error.cuh"

#include <thrust/complex.h>
#include <thrust/device_vector.h>

#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using Complex = thrust::complex<float>;

// 你给的约束：uvw_index < 400000
static constexpr int UVW_MAX = 450000;

// ===============================
// 设备侧小工具
// ===============================
__device__ __forceinline__ void sincos_fast(float x, float* s, float* c) {
    __sincosf(x, s, c);
}
__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}
__device__ __forceinline__ float norm3(float x, float y, float z) {
    return sqrtf(x * x + y * y + z * z);
}

// ===============================
// Kernel 1：一次性生成 l,m,n（不回写 theta/phi；B 缩放放到 Host 侧一次完成）
// ===============================
__global__ void healpix_moonback_pre_opt(
    const float* __restrict__ theta_heal,
    const float* __restrict__ phi_heal,
    float* __restrict__ l,
    float* __restrict__ m,
    float* __restrict__ n,
    int npix
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npix) return;

    float theta_val = theta_heal[idx];
    float phi_val   = phi_heal[idx];

    theta_val = (float)M_PI * 0.5f - theta_val;
    if (phi_val > (float)M_PI) phi_val -= 2.0f * (float)M_PI;
    phi_val = -phi_val;

    float st, ct;
    float sp, cp;
    sincos_fast(theta_val, &st, &ct); // st=sin, ct=cos
    sincos_fast(phi_val,   &sp, &cp);

    l[idx] = ct * cp;
    m[idx] = ct * sp;
    n[idx] = st;
}

// ===============================
// Kernel 2：计算 Viss（去 acosf；去 complex exp；用 sincosf）
// 条件 beta<=phi <=> dot/norm >= cos(phi)
// exp(-i*2*pi*phase) 用 sincos
// ===============================
__global__ void healpix_moonback_viss_opt(
    const float* __restrict__ B,
    Complex* __restrict__ Viss,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ xyz1a,
    const float* __restrict__ xyz1b,
    const float* __restrict__ xyz1c,
    const float* __restrict__ xyz2a,
    const float* __restrict__ xyz2b,
    const float* __restrict__ xyz2c,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    int uvw_index,
    int npix,
    float phi
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= uvw_index) return;

    float u0 = u[i];
    float v0 = v[i];
    float w0 = w[i];

    float x1a = xyz1a[i], x1b = xyz1b[i], x1c = xyz1c[i];
    float x2a = xyz2a[i], x2b = xyz2b[i], x2c = xyz2c[i];

    float norm1 = norm3(x1a, x1b, x1c);
    float norm2 = norm3(x2a, x2b, x2c);

    float cosphi = cosf(phi);
    float thr1 = norm1 * cosphi;
    float thr2 = norm2 * cosphi;

    float acc_re = 0.0f;
    float acc_im = 0.0f;

    const float k = -2.0f * (float)M_PI;

    for (int p = 0; p < npix; ++p) {
        float lp = l[p], mp = m[p], np = n[p];

        float dot1 = lp * x1a + mp * x1b + np * x1c;
        float dot2 = lp * x2a + mp * x2b + np * x2c;

        if (dot1 >= thr1 && dot2 >= thr2) {
            float phase = u0 * lp + v0 * mp + w0 * (np - 1.0f);
            float angle = k * phase;

            float s, c;
            sincos_fast(angle, &s, &c); // exp(i*angle)=c+i*s

            float bp = B[p];
            acc_re += bp * c;
            acc_im += bp * s;
        }
    }

    Complex out(acc_re, acc_im);
    Viss[i] = out;
}

// ===============================
// Kernel 3：ceilAndScale -> gs（直接得到 ring index；去 sort）
// ceil((bll-1/4)/0.5) = ceil((bll-0.25)*2)
// 用 __float2int_ru 实现 ceil（对正数）
// 加 clamp 到 [0,nr]
// ===============================
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

// ===============================
// Kernel 4：O(size) histogram 统计 mb（替代原 O(size*nr)）
// 统计 g in [1,nr] -> mb[g-1]++
// ===============================
__global__ void histogram_gs(
    const int* __restrict__ gs,
    unsigned int* __restrict__ hist,
    int size,
    int nr
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int g = gs[idx];
    if (g >= 1 && g <= nr) {
        atomicAdd(&hist[g - 1], 1u);
    }
}

// ===============================
// Kernel 5：计算 dcf（去 pow，闭式多项式）
// dcf[0] = 1/(pi*4)
// 对 idx=1..nr：x = idx/2 + 1/4
// x^3-(x-0.5)^3 = 1.5x^2 - 0.75x + 0.125
// dcf[idx] = 2/3*pi*diff / mb[idx-1]
// ===============================
__global__ void calculateDcf_opt(
    float* __restrict__ dcf,             // nr+1
    const unsigned int* __restrict__ mb, // nr
    int nr
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (nr + 1)) return;

    if (idx == 0) {
        dcf[0] = 1.0f / ((float)M_PI * 4.0f);
        return;
    }

    unsigned int cnt = mb[idx - 1];
    if (cnt == 0u) {
        dcf[idx] = 0.0f;
        return;
    }

    float x = 0.5f * (float)idx + 0.25f;
    float diff = 1.5f * x * x - 0.75f * x + 0.125f;

    dcf[idx] = (2.0f / 3.0f) * (float)M_PI * diff / (float)cnt;
}

// ===============================
// Kernel 6：Viss 相位修正 + dg（去 complex sqrt/abs；改实数等价）
// Viss *= exp(-i*2*pi*w)
// gamma=asin(w/bll), dg = sqrt(|1-4*sin^2(gamma)|)/|cos(gamma)|*3/2
// ===============================
__global__ void viss_dg_trans_opt(
    Complex* __restrict__ Viss,
    const float* __restrict__ w,
    const float* __restrict__ bll,
    float* __restrict__ dg,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float wv = w[idx];
    float bv = bll[idx];

    // Viss *= exp(-i*2*pi*w)
    float angle = -2.0f * (float)M_PI * wv;
    float s, c;
    sincos_fast(angle, &s, &c);

    float a = Viss[idx].real();
    float b = Viss[idx].imag();
    float re = a * c - b * s;
    float im = a * s + b * c;
    Viss[idx] = Complex(re, im);

    float ratio = (bv != 0.0f) ? (wv / bv) : 0.0f;
    ratio = clampf(ratio, -1.0f, 1.0f);
    float gamma = asinf(ratio);

    float sg = sinf(gamma);
    float cg = cosf(gamma);

    float inside = 1.0f - 4.0f * sg * sg;
    float mag = sqrtf(fabsf(inside));

    float denom = fabsf(cg);
    if (denom < 1e-12f) denom = 1e-12f;

    dg[idx] = mag / denom * 1.5f;
}

// ===============================
// Kernel 7：预计算 weight[i] = min(dcf[gs[i]]*dg[i], 1/8)
// ===============================
__global__ void precompute_weight(
    float* __restrict__ weight,
    const float* __restrict__ dcf,   // nr+1
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

// ===============================
// Kernel 8：computeC（shared memory tiling, no blockage）
// C[pix] = sum_i Viss[i]*weight[i]*exp(i*2*pi*(u*l + v*m + w*n))
// ===============================
__global__ void computeC_tiled(
    int npix,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const float* __restrict__ weight,
    const Complex* __restrict__ Viss,
    Complex* __restrict__ C,
    int uvw_index
) {
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= npix) return;

    float lp = l[pix];
    float mp = m[pix];
    float np = n[pix];

    float acc_re = 0.0f;
    float acc_im = 0.0f;

    extern __shared__ unsigned char smem[];
    float* su = (float*)smem;
    float* sv = su + blockDim.x;
    float* sw = sv + blockDim.x;
    float* sW = sw + blockDim.x;
    float2* sV = (float2*)(sW + blockDim.x);

    for (int base = 0; base < uvw_index; base += blockDim.x) {
        int i = base + threadIdx.x;

        if (i < uvw_index) {
            su[threadIdx.x] = u[i];
            sv[threadIdx.x] = v[i];
            sw[threadIdx.x] = w[i];
            sW[threadIdx.x] = weight[i];
            sV[threadIdx.x] = make_float2(Viss[i].real(), Viss[i].imag());
        } else {
            su[threadIdx.x] = 0.0f;
            sv[threadIdx.x] = 0.0f;
            sw[threadIdx.x] = 0.0f;
            sW[threadIdx.x] = 0.0f;
            sV[threadIdx.x] = make_float2(0.0f, 0.0f);
        }

        __syncthreads();

        int tileN = min(blockDim.x, uvw_index - base);
        #pragma unroll 4
        for (int t = 0; t < tileN; ++t) {
            float phase = su[t] * lp + sv[t] * mp + sw[t] * np;
            float angle = 2.0f * (float)M_PI * phase;

            float s, c;
            sincos_fast(angle, &s, &c);

            float wt = sW[t];
            float a = sV[t].x;
            float b = sV[t].y;

            acc_re += wt * (a * c - b * s);
            acc_im += wt * (a * s + b * c);
        }

        __syncthreads();
    }

    C[pix] = Complex(acc_re, acc_im);
}

// ===============================
// Kernel 9：computeC（shared memory tiling, with blockage in recon only）
// MATLAB 对应：
//   gdcf2 = gdcf(gs).*gdg;
//   beta1/beta2 超出 phi => gdcf2=0
//   之后再做 1/8 截断
// 这里 weight 已经等于 min(gdcf(gs).*gdg, 1/8)，
// 因此只需在累加前做遮挡 mask：blocked ? 0 : weight
// ===============================
__global__ void computeC_tiled_blockage(
    int npix,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const float* __restrict__ xyz1a,
    const float* __restrict__ xyz1b,
    const float* __restrict__ xyz1c,
    const float* __restrict__ xyz2a,
    const float* __restrict__ xyz2b,
    const float* __restrict__ xyz2c,
    const float* __restrict__ weight,
    const Complex* __restrict__ Viss,
    Complex* __restrict__ C,
    int uvw_index,
    float cosphi
) {
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= npix) return;

    float lp = l[pix];
    float mp = m[pix];
    float np = n[pix];

    float acc_re = 0.0f;
    float acc_im = 0.0f;

    extern __shared__ unsigned char smem[];
    float* su    = (float*)smem;
    float* sv    = su    + blockDim.x;
    float* sw    = sv    + blockDim.x;
    float* sW    = sw    + blockDim.x;
    float2* sV   = (float2*)(sW + blockDim.x);
    float* sx1a  = (float*)(sV   + blockDim.x);
    float* sx1b  = sx1a  + blockDim.x;
    float* sx1c  = sx1b  + blockDim.x;
    float* sx2a  = sx1c  + blockDim.x;
    float* sx2b  = sx2a  + blockDim.x;
    float* sx2c  = sx2b  + blockDim.x;
    float* sthr1 = sx2c  + blockDim.x;
    float* sthr2 = sthr1 + blockDim.x;

    for (int base = 0; base < uvw_index; base += blockDim.x) {
        int i = base + threadIdx.x;

        if (i < uvw_index) {
            float x1a = xyz1a[i], x1b = xyz1b[i], x1c = xyz1c[i];
            float x2a = xyz2a[i], x2b = xyz2b[i], x2c = xyz2c[i];

            su[threadIdx.x]   = u[i];
            sv[threadIdx.x]   = v[i];
            sw[threadIdx.x]   = w[i];
            sW[threadIdx.x]   = weight[i];
            sV[threadIdx.x]   = make_float2(Viss[i].real(), Viss[i].imag());

            sx1a[threadIdx.x] = x1a;
            sx1b[threadIdx.x] = x1b;
            sx1c[threadIdx.x] = x1c;
            sx2a[threadIdx.x] = x2a;
            sx2b[threadIdx.x] = x2b;
            sx2c[threadIdx.x] = x2c;

            sthr1[threadIdx.x] = norm3(x1a, x1b, x1c) * cosphi;
            sthr2[threadIdx.x] = norm3(x2a, x2b, x2c) * cosphi;
        } else {
            su[threadIdx.x]   = 0.0f;
            sv[threadIdx.x]   = 0.0f;
            sw[threadIdx.x]   = 0.0f;
            sW[threadIdx.x]   = 0.0f;
            sV[threadIdx.x]   = make_float2(0.0f, 0.0f);

            sx1a[threadIdx.x] = 0.0f;
            sx1b[threadIdx.x] = 0.0f;
            sx1c[threadIdx.x] = 0.0f;
            sx2a[threadIdx.x] = 0.0f;
            sx2b[threadIdx.x] = 0.0f;
            sx2c[threadIdx.x] = 0.0f;
            sthr1[threadIdx.x] = 0.0f;
            sthr2[threadIdx.x] = 0.0f;
        }

        __syncthreads();

        int tileN = min(blockDim.x, uvw_index - base);
        #pragma unroll 4
        for (int t = 0; t < tileN; ++t) {
            float dot1 = lp * sx1a[t] + mp * sx1b[t] + np * sx1c[t];
            float dot2 = lp * sx2a[t] + mp * sx2b[t] + np * sx2c[t];

            if (dot1 < sthr1[t] || dot2 < sthr2[t]) continue;

            float phase = su[t] * lp + sv[t] * mp + sw[t] * np;
            float angle = 2.0f * (float)M_PI * phase;

            float s, c;
            sincos_fast(angle, &s, &c);

            float wt = sW[t];
            float a = sV[t].x;
            float b = sV[t].y;

            acc_re += wt * (a * c - b * s);
            acc_im += wt * (a * s + b * c);
        }

        __syncthreads();
    }

    C[pix] = Complex(acc_re, acc_im);
}

// ===============================
// Kernel 10：提取 C 实部（便于更快写文件）
// ===============================
__global__ void extract_real(
    const Complex* __restrict__ C,
    float* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = C[idx].real();
}

// ===============================
// 更快的文本读取（FILE + fgets + strtof）
// ===============================
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

static bool load_single(
    const std::string& path,
    float* a,
    int maxN,
    int& outN
) {
    FILE* fp = fopen(path.c_str(), "r");
    if (!fp) return false;
    char line[256];
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

// ===============================
// 频率 -> 路径/文件名（1MHz / 10MHz）
// 你要求：差异只有文件路径 + main 调用频率
// ===============================
static inline bool is_1mhz(float f) {
    // 允许少量浮动误差
    return (fabsf(f - 1.0e6f) < 1.0e5f);
}

static inline void build_paths(float frequency, std::string& base_dir, std::string& tag_short, std::string& tag_file) {
    // base_dir：输入文件夹
    // tag_short：输出目录后缀（1M / 10M）
    // tag_file：输入 B/theta/phi 的文件名后缀（1Mhz / 10Mhz）
    if (is_1mhz(frequency)) {
        base_dir = "./earth_1Mhz/";
        tag_short = "1M";
        tag_file  = "1M";
    } else {
        base_dir = "/data/zhaox/earth_10Mhz/";
        tag_short = "10M";
        tag_file  = "10M";
    }
}

// ===============================
// 计时
// ===============================
static timeval start_tv, finish_tv;
static float total_time = 0.0f;

// ===============================
// 主流程
// ===============================
int vissGen(float frequency, int blockage)
{
    gettimeofday(&start_tv, NULL);

    int nDevices = 0;
    CHECK(cudaGetDeviceCount(&nDevices));
    omp_set_num_threads(nDevices);

    cout << "devices: " << nDevices << endl;
    cout << "frequency: " << frequency << endl;

    int days = 1;
    int start_day = 0;

    // 频率相关路径/标签（1MHz/10MHz）
    std::string address, tag_short, tag_file;
    build_paths(frequency, address, tag_short, tag_file);

    // 读取 B/theta/phi（CPU）
    string address_B          = address + "B_" + tag_file + ".txt";
    string address_theta_heal = address + "theta_heal_" + tag_file + ".txt";
    string address_phi_heal   = address + "phi_heal_" + tag_file + ".txt";

    ifstream BFile(address_B);
    ifstream thetaFile(address_theta_heal);
    ifstream phiFile(address_phi_heal);

    if (!BFile.is_open() || !thetaFile.is_open() || !phiFile.is_open()) {
        cout << "ERROR: cannot open B/theta/phi files:\n"
             << address_B << "\n" << address_theta_heal << "\n" << address_phi_heal << endl;
        return -1;
    }

    int npix = 0;
    BFile >> npix;
    cout << "npix: " << npix << endl;

    vector<float> cB(npix), ctheta_heal(npix), cphi_heal(npix);
    for (int i = 0; i < npix; i++) {
        BFile >> cB[i];
        thetaFile >> ctheta_heal[i];
        phiFile >> cphi_heal[i];
    }
    BFile.close();
    thetaFile.close();
    phiFile.close();
    cout << "load B/theta/phi in CPU success" << endl;

    // 计算 nside（你说 1MHz=512、10MHz=4096；这里由 npix 自动推出）
    int nside = (int)lround(sqrt((double)npix / 12.0));
    float s   = 4.0f * (float)M_PI / (float)npix;
    float res = sqrtf(4.0f * (float)M_PI / (float)npix);

    // 遮挡参数
    float R = 1737.1e3f;
    float h = 300e3f;
    float theta = asinf(R / (R + h));
    float phi   = (float)M_PI - theta;

    // recon 参数
    float bl_max = 100e3f;
    float lamda  = 3e8f / frequency;
    int   nr     = (int)ceil(bl_max / lamda * 2.0f);

    cout << "address: " << address << endl;
    cout << "nside: " << nside << endl;
    cout << "s: " << s << endl;
    cout << "res: " << res << endl;
    cout << "theta: " << theta << endl;
    cout << "phi: " << phi << endl;
    cout << "lamda: " << lamda << endl;
    cout << "nr: " << nr << endl;
    cout << "recon blockage: " << blockage << endl;

    // Host 侧先把 B *= s（避免每 day 重复缩放）
    vector<float> cB_scaled = cB;
    for (int i = 0; i < npix; ++i) cB_scaled[i] *= s;

    // 输出目录（按频率分开）
    // 你原 10MHz：3dnoblockage10M/...
    // 这里 1MHz：3dnoblockage1M/...
    std::string out_dir = (blockage ? "3dblockage" : "3dnoblockage") + tag_short + "/";

    // OpenMP：每线程控制一张 GPU
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        CHECK(cudaSetDevice(tid));
        CHECK(cudaDeviceSynchronize());
        cout << "Thread " << tid << " is running on device " << tid << endl;

        cudaStream_t stream;
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        // 常驻：B + lmn + C + Creal
        thrust::device_vector<float> d_B(cB_scaled.begin(), cB_scaled.end());
        thrust::device_vector<float> d_theta(ctheta_heal.begin(), ctheta_heal.end());
        thrust::device_vector<float> d_phi(cphi_heal.begin(), cphi_heal.end());
        thrust::device_vector<float> d_l(npix), d_m(npix), d_n(npix);

        // 一次性生成 l,m,n
        {
            constexpr int BLOCK = 256;
            int grid = (npix + BLOCK - 1) / BLOCK;
            healpix_moonback_pre_opt<<<grid, BLOCK, 0, stream>>>(
                thrust::raw_pointer_cast(d_theta.data()),
                thrust::raw_pointer_cast(d_phi.data()),
                thrust::raw_pointer_cast(d_l.data()),
                thrust::raw_pointer_cast(d_m.data()),
                thrust::raw_pointer_cast(d_n.data()),
                npix
            );
            CHECK(cudaPeekAtLastError());
            CHECK(cudaStreamSynchronize(stream));
        }

        // theta/phi 释放（不再需要）
        d_theta.clear(); d_theta.shrink_to_fit();
        d_phi.clear();   d_phi.shrink_to_fit();

        thrust::device_vector<Complex> d_C(npix);
        thrust::device_vector<float>   d_Creal(npix);

        // 复用：Host 缓冲（UV）
        std::vector<float> cu(UVW_MAX), cv(UVW_MAX), cw(UVW_MAX);
        std::vector<float> cxyz1a(UVW_MAX), cxyz1b(UVW_MAX), cxyz1c(UVW_MAX);
        std::vector<float> cxyz2a(UVW_MAX), cxyz2b(UVW_MAX), cxyz2c(UVW_MAX);
        std::vector<float> cbll(UVW_MAX);

        // 复用：Device 缓冲
        thrust::device_vector<float> d_u(UVW_MAX), d_v(UVW_MAX), d_w(UVW_MAX);
        thrust::device_vector<float> d_xyz1a(UVW_MAX), d_xyz1b(UVW_MAX), d_xyz1c(UVW_MAX);
        thrust::device_vector<float> d_xyz2a(UVW_MAX), d_xyz2b(UVW_MAX), d_xyz2c(UVW_MAX);
        thrust::device_vector<float> d_bll(UVW_MAX);

        thrust::device_vector<int> d_gs(UVW_MAX);
        thrust::device_vector<unsigned int> d_mb(nr);
        thrust::device_vector<float> d_dcf(nr + 1);
        thrust::device_vector<float> d_dg(UVW_MAX);
        thrust::device_vector<float> d_weight(UVW_MAX);

        // Viss 
        thrust::device_vector<Complex> d_Viss(UVW_MAX);

        for (int p = tid + start_day; p < days; p += nDevices) {
            cout << "GPU " << tid << " processing day " << (p + 1) << endl;

            int uvw_index = 0, xyz1_index = 0, xyz2_index = 0, bll_index = 0;

            // 读取 uvw/xyza/xyzb/bll
            {
                string address_uvw  = address + "updated_uvw"  + to_string(p + 1) + "day1M.txt";
                string address_xyz1 = address + "xyza" + to_string(p + 1) + "day1M.txt";
                string address_xyz2 = address + "xyzb" + to_string(p + 1) + "day1M.txt";
                string address_bll  = address + "bll"  + to_string(p + 1) + "day1M.txt";

                if (!load_triplets(address_uvw, cu.data(), cv.data(), cw.data(), UVW_MAX, uvw_index)) {
                    cout << "ERROR open/read " << address_uvw << endl; continue;
                }
                if (!load_triplets(address_xyz1, cxyz1a.data(), cxyz1b.data(), cxyz1c.data(), UVW_MAX, xyz1_index)) {
                    cout << "ERROR open/read " << address_xyz1 << endl; continue;
                }
                if (!load_triplets(address_xyz2, cxyz2a.data(), cxyz2b.data(), cxyz2c.data(), UVW_MAX, xyz2_index)) {
                    cout << "ERROR open/read " << address_xyz2 << endl; continue;
                }
                if (!load_single(address_bll, cbll.data(), UVW_MAX, bll_index)) {
                    cout << "ERROR open/read " << address_bll << endl; continue;
                }

                if (uvw_index <= 0 || xyz1_index != uvw_index || xyz2_index != uvw_index || bll_index != uvw_index) {
                    cout << "ERROR: index mismatch. uvw=" << uvw_index
                         << " xyz1=" << xyz1_index << " xyz2=" << xyz2_index
                         << " bll=" << bll_index << endl;
                    continue;
                }
            }

            // H2D memcpy（async）
            {
                CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_u.data()), cu.data(),
                                      uvw_index * sizeof(float), cudaMemcpyHostToDevice, stream));
                CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_v.data()), cv.data(),
                                      uvw_index * sizeof(float), cudaMemcpyHostToDevice, stream));
                CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_w.data()), cw.data(),
                                      uvw_index * sizeof(float), cudaMemcpyHostToDevice, stream));

                CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_xyz1a.data()), cxyz1a.data(),
                                      uvw_index * sizeof(float), cudaMemcpyHostToDevice, stream));
                CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_xyz1b.data()), cxyz1b.data(),
                                      uvw_index * sizeof(float), cudaMemcpyHostToDevice, stream));
                CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_xyz1c.data()), cxyz1c.data(),
                                      uvw_index * sizeof(float), cudaMemcpyHostToDevice, stream));

                CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_xyz2a.data()), cxyz2a.data(),
                                      uvw_index * sizeof(float), cudaMemcpyHostToDevice, stream));
                CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_xyz2b.data()), cxyz2b.data(),
                                      uvw_index * sizeof(float), cudaMemcpyHostToDevice, stream));
                CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_xyz2c.data()), cxyz2c.data(),
                                      uvw_index * sizeof(float), cudaMemcpyHostToDevice, stream));

                CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_bll.data()), cbll.data(),
                                      uvw_index * sizeof(float), cudaMemcpyHostToDevice, stream));
                CHECK(cudaStreamSynchronize(stream));
            }

            // Viss
            {
                constexpr int BLOCK = 256;
                int grid = (uvw_index + BLOCK - 1) / BLOCK;

                healpix_moonback_viss_opt<<<grid, BLOCK, 0, stream>>>(
                    thrust::raw_pointer_cast(d_B.data()),
                    thrust::raw_pointer_cast(d_Viss.data()),
                    thrust::raw_pointer_cast(d_u.data()),
                    thrust::raw_pointer_cast(d_v.data()),
                    thrust::raw_pointer_cast(d_w.data()),
                    thrust::raw_pointer_cast(d_xyz1a.data()),
                    thrust::raw_pointer_cast(d_xyz1b.data()),
                    thrust::raw_pointer_cast(d_xyz1c.data()),
                    thrust::raw_pointer_cast(d_xyz2a.data()),
                    thrust::raw_pointer_cast(d_xyz2b.data()),
                    thrust::raw_pointer_cast(d_xyz2c.data()),
                    thrust::raw_pointer_cast(d_l.data()),
                    thrust::raw_pointer_cast(d_m.data()),
                    thrust::raw_pointer_cast(d_n.data()),
                    uvw_index, npix, phi
                );
                CHECK(cudaPeekAtLastError());
            }

            // gs
            {
                constexpr int BLOCK = 256;
                int grid = (uvw_index + BLOCK - 1) / BLOCK;

                ceilAndScale_opt<<<grid, BLOCK, 0, stream>>>(
                    thrust::raw_pointer_cast(d_bll.data()),
                    thrust::raw_pointer_cast(d_gs.data()),
                    uvw_index, nr
                );
                CHECK(cudaPeekAtLastError());
            }

            // histogram mb
            {
                CHECK(cudaMemsetAsync(thrust::raw_pointer_cast(d_mb.data()), 0,
                                      nr * sizeof(unsigned int), stream));

                constexpr int BLOCK = 256;
                int grid = (uvw_index + BLOCK - 1) / BLOCK;

                histogram_gs<<<grid, BLOCK, 0, stream>>>(
                    thrust::raw_pointer_cast(d_gs.data()),
                    thrust::raw_pointer_cast(d_mb.data()),
                    uvw_index, nr
                );
                CHECK(cudaPeekAtLastError());
            }

            // dcf
            {
                constexpr int BLOCK = 256;
                int grid = ((nr + 1) + BLOCK - 1) / BLOCK;

                calculateDcf_opt<<<grid, BLOCK, 0, stream>>>(
                    thrust::raw_pointer_cast(d_dcf.data()),
                    thrust::raw_pointer_cast(d_mb.data()),
                    nr
                );
                CHECK(cudaPeekAtLastError());
            }

            // Viss phase + dg
            {
                constexpr int BLOCK = 256;
                int grid = (uvw_index + BLOCK - 1) / BLOCK;

                viss_dg_trans_opt<<<grid, BLOCK, 0, stream>>>(
                    thrust::raw_pointer_cast(d_Viss.data()),
                    thrust::raw_pointer_cast(d_w.data()),
                    thrust::raw_pointer_cast(d_bll.data()),
                    thrust::raw_pointer_cast(d_dg.data()),
                    uvw_index
                );
                CHECK(cudaPeekAtLastError());
            }

            // weight
            {
                constexpr int BLOCK = 256;
                int grid = (uvw_index + BLOCK - 1) / BLOCK;

                precompute_weight<<<grid, BLOCK, 0, stream>>>(
                    thrust::raw_pointer_cast(d_weight.data()),
                    thrust::raw_pointer_cast(d_dcf.data()),
                    thrust::raw_pointer_cast(d_dg.data()),
                    thrust::raw_pointer_cast(d_gs.data()),
                    uvw_index, nr
                );
                CHECK(cudaPeekAtLastError());
            }

            // computeC (optional blockage in recon stage only)
            {
                CHECK(cudaMemsetAsync(thrust::raw_pointer_cast(d_C.data()), 0,
                                      (size_t)npix * sizeof(Complex), stream));

                constexpr int BLOCK = 256;
                int grid = (npix + BLOCK - 1) / BLOCK;

                if (blockage) {
                    size_t shmem = (size_t)BLOCK * (12 * sizeof(float) + sizeof(float2));

                    computeC_tiled_blockage<<<grid, BLOCK, shmem, stream>>>(
                        npix,
                        thrust::raw_pointer_cast(d_u.data()),
                        thrust::raw_pointer_cast(d_v.data()),
                        thrust::raw_pointer_cast(d_w.data()),
                        thrust::raw_pointer_cast(d_l.data()),
                        thrust::raw_pointer_cast(d_m.data()),
                        thrust::raw_pointer_cast(d_n.data()),
                        thrust::raw_pointer_cast(d_xyz1a.data()),
                        thrust::raw_pointer_cast(d_xyz1b.data()),
                        thrust::raw_pointer_cast(d_xyz1c.data()),
                        thrust::raw_pointer_cast(d_xyz2a.data()),
                        thrust::raw_pointer_cast(d_xyz2b.data()),
                        thrust::raw_pointer_cast(d_xyz2c.data()),
                        thrust::raw_pointer_cast(d_weight.data()),
                        thrust::raw_pointer_cast(d_Viss.data()),
                        thrust::raw_pointer_cast(d_C.data()),
                        uvw_index,
                        cosf(phi)
                    );
                } else {
                    size_t shmem = (size_t)BLOCK * (4 * sizeof(float) + sizeof(float2));

                    computeC_tiled<<<grid, BLOCK, shmem, stream>>>(
                        npix,
                        thrust::raw_pointer_cast(d_u.data()),
                        thrust::raw_pointer_cast(d_v.data()),
                        thrust::raw_pointer_cast(d_w.data()),
                        thrust::raw_pointer_cast(d_l.data()),
                        thrust::raw_pointer_cast(d_m.data()),
                        thrust::raw_pointer_cast(d_n.data()),
                        thrust::raw_pointer_cast(d_weight.data()),
                        thrust::raw_pointer_cast(d_Viss.data()),
                        thrust::raw_pointer_cast(d_C.data()),
                        uvw_index
                    );
                }
                CHECK(cudaPeekAtLastError());
            }

            // 提取 C real
            {
                constexpr int BLOCK = 256;
                int grid = (npix + BLOCK - 1) / BLOCK;

                extract_real<<<grid, BLOCK, 0, stream>>>(
                    thrust::raw_pointer_cast(d_C.data()),
                    thrust::raw_pointer_cast(d_Creal.data()),
                    npix
                );
                CHECK(cudaPeekAtLastError());
                CHECK(cudaStreamSynchronize(stream));
            }

            // 保存（分块 D2H，避免一次性 host 超大内存）
            {
                // 输出文件名：保持你原风格，只把目录与后缀按频率区分
                // 10MHz：.../C{day}day10M.txt
                // 1MHz ：.../C{day}day1M.txt
                string address_C = out_dir + "C" + to_string(p + 1) + "day" + tag_short + ".txt";
                cout << "GPU " << tid << " saving: " << address_C << endl;

                std::ofstream file(address_C);
                if (!file.is_open()) {
                    cout << "ERROR open output file: " << address_C << endl;
                } else {
                    // 增大输出缓冲
                    static const size_t OUT_BUF_SZ = 8 << 20;
                    static thread_local std::vector<char> outbuf(OUT_BUF_SZ);
                    file.rdbuf()->pubsetbuf(outbuf.data(), outbuf.size());

                    // pinned host chunk
                    float* h_chunk = nullptr;
                    const int CHUNK = 1 << 20; // ~4MB
                    CHECK(cudaMallocHost(&h_chunk, (size_t)CHUNK * sizeof(float)));

                    float* dptr = thrust::raw_pointer_cast(d_Creal.data());

                    for (int off = 0; off < npix; off += CHUNK) {
                        int cur = std::min(CHUNK, npix - off);
                        CHECK(cudaMemcpy(h_chunk, dptr + off, (size_t)cur * sizeof(float), cudaMemcpyDeviceToHost));

                        for (int i = 0; i < cur; ++i) {
                            file << h_chunk[i] << "\n";
                        }
                    }

                    CHECK(cudaFreeHost(h_chunk));
                    file.close();
                    cout << "GPU " << tid << " saved success." << endl;
                }
            }
        }

        CHECK(cudaStreamDestroy(stream));
    }

    gettimeofday(&finish_tv, NULL);
    total_time = ((finish_tv.tv_sec - start_tv.tv_sec) * 1000000 + (finish_tv.tv_usec - start_tv.tv_usec)) / 1000000.0f;
    cout << "total time: " << total_time << "s" << endl;

    return 0;
}

int main(int argc, char** argv)
{
    float freq = 1e6f;
    int blockage = 0;

    if (argc >= 2) freq = strtof(argv[1], nullptr);
    if (argc >= 3) blockage = atoi(argv[2]);

    cout << "usage: " << argv[0] << " [frequency] [blockage]" << endl;
    cout << "example: " << argv[0] << " 1e6 0" << endl;
    cout << "example: " << argv[0] << " 1e6 1" << endl;

    return vissGen(freq, blockage);
}
