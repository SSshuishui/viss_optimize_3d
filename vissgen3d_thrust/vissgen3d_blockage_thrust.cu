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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using Complex = thrust::complex<float>;

// 你给的实际约束：uvw_index < 400000
static constexpr int UVW_MAX = 450000;

// ===============================
// device utils
// ===============================
__device__ __forceinline__ void sincos_fast(float x, float* s, float* c) { __sincosf(x, s, c); }
__device__ __forceinline__ float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
__device__ __forceinline__ float norm3(float x, float y, float z) { return sqrtf(x * x + y * y + z * z); }

// ===============================
// Kernel: healpix pre (生成 l,m,n；不回写 theta/phi；B 已在 host 缩放)
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
    sincos_fast(theta_val, &st, &ct);
    sincos_fast(phi_val,   &sp, &cp);

    l[idx] = ct * cp;
    m[idx] = ct * sp;
    n[idx] = st;
}

// ===============================
// Kernel: Viss (加入遮挡的版本仍然用同样的 Viss 定义：对 sky 积分)
// 关键优化：
// - acos 判定改为 dot >= norm*cos(phi)
// - exp(-i*2*pi*phase) 用 sincos
// - 对 sky(pix) 维度做 shared tiling：每个 block 只加载一次 B/l/m/n tile，供多条基线复用
// ===============================
template<int TILE_PIX>
__global__ void healpix_moonback_viss_tiled(
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

    float n1 = norm3(x1a, x1b, x1c);
    float n2 = norm3(x2a, x2b, x2c);

    float cosphi = cosf(phi);
    float thr1 = n1 * cosphi;
    float thr2 = n2 * cosphi;

    float acc_re = 0.0f;
    float acc_im = 0.0f;

    // exp(-i 2*pi*phase)
    const float k = -2.0f * (float)M_PI;

    extern __shared__ float smem[];
    float* sB = smem;                   // TILE_PIX
    float* sL = sB + TILE_PIX;          // TILE_PIX
    float* sM = sL + TILE_PIX;          // TILE_PIX
    float* sN = sM + TILE_PIX;          // TILE_PIX

    // sky 维度分块
    for (int base = 0; base < npix; base += TILE_PIX) {
        // blockDim 可能 < TILE_PIX，用两次加载补齐
        int t0 = threadIdx.x;
        if (t0 < TILE_PIX) {
            int p = base + t0;
            if (p < npix) {
                sB[t0] = B[p];
                sL[t0] = l[p];
                sM[t0] = m[p];
                sN[t0] = n[p];
            } else {
                sB[t0] = 0.0f; sL[t0] = 0.0f; sM[t0] = 0.0f; sN[t0] = 0.0f;
            }
        }
        int t1 = threadIdx.x + blockDim.x;
        if (t1 < TILE_PIX) {
            int p = base + t1;
            if (p < npix) {
                sB[t1] = B[p];
                sL[t1] = l[p];
                sM[t1] = m[p];
                sN[t1] = n[p];
            } else {
                sB[t1] = 0.0f; sL[t1] = 0.0f; sM[t1] = 0.0f; sN[t1] = 0.0f;
            }
        }
        __syncthreads();

        int tileN = min(TILE_PIX, npix - base);
        #pragma unroll 4
        for (int t = 0; t < tileN; ++t) {
            float lp = sL[t], mp = sM[t], np_ = sN[t];

            float dot1 = lp * x1a + mp * x1b + np_ * x1c;
            float dot2 = lp * x2a + mp * x2b + np_ * x2c;

            if (dot1 >= thr1 && dot2 >= thr2) {
                float phase = u0 * lp + v0 * mp + w0 * (np_ - 1.0f);
                float ang = k * phase;
                float s, c;
                sincos_fast(ang, &s, &c);

                float bp = sB[t];
                acc_re += bp * c;
                acc_im += bp * s;
            }
        }

        __syncthreads();
    }

    Complex out(acc_re, acc_im);
    Viss[i] = out;
}

// ===============================
// Kernel: Viss phase correction
// 原逻辑：Viss *= exp(-i*2*pi*w)
// 目的：把 Viss 中的 w*(n-1) 形式修正为 w*n 形式（与后续 exp(i*2*pi*(u l + v m + w n)) 对齐）
// ===============================
__global__ void phase_correct_viss(
    Complex* __restrict__ Viss,
    const float* __restrict__ w,
    int uvw_index
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= uvw_index) return;

    float ang = -2.0f * (float)M_PI * w[idx];
    float s, c;
    sincos_fast(ang, &s, &c);

    float a = Viss[idx].real();
    float b = Viss[idx].imag();

    // (a+ib)*(c+is)
    float re = a * c - b * s;
    float im = a * s + b * c;
    Viss[idx] = Complex(re, im);
}

// ===============================
// Kernel: computeC with blockage (penalty = f[i], 并对每个 (pix,i) 做 FOV 判定；不做 acos)
// 判定：acos(dot/norm) <= phi  <=> dot/norm >= cos(phi)
// 这里对 xyz1/xyz2 的 norm 用 rsqrt 加速
// 对 uvw 做 shared tiling，减少全局访存
// ===============================
template<int TILE_UVW>
__global__ void computeC_blockage_tiled(
    int npix,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ f,        // penalty（已在 host 侧做 1/f_point）
    const float* __restrict__ xyz1a,
    const float* __restrict__ xyz1b,
    const float* __restrict__ xyz1c,
    const float* __restrict__ xyz2a,
    const float* __restrict__ xyz2b,
    const float* __restrict__ xyz2c,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const Complex* __restrict__ Viss,   // 已做 phase_correct
    Complex* __restrict__ C,
    float phi,
    int uvw_index
) {
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= npix) return;

    float lp = l[pix];
    float mp = m[pix];
    float np_ = n[pix];

    float acc_re = 0.0f;
    float acc_im = 0.0f;

    float cosphi = cosf(phi);

    // shared: u,v,w,f + xyz1/xyz2 + invnorm + Viss
    extern __shared__ unsigned char smem_raw[];
    float* su   = (float*)smem_raw;                 // TILE_UVW
    float* sv   = su + TILE_UVW;
    float* sw   = sv + TILE_UVW;
    float* sf   = sw + TILE_UVW;

    float* sx1a = sf + TILE_UVW;
    float* sx1b = sx1a + TILE_UVW;
    float* sx1c = sx1b + TILE_UVW;

    float* sx2a = sx1c + TILE_UVW;
    float* sx2b = sx2a + TILE_UVW;
    float* sx2c = sx2b + TILE_UVW;

    float* sInv1 = sx2c + TILE_UVW;
    float* sInv2 = sInv1 + TILE_UVW;

    float2* sV  = (float2*)(sInv2 + TILE_UVW);

    for (int base = 0; base < uvw_index; base += TILE_UVW) {
        int i = base + threadIdx.x;

        if (threadIdx.x < TILE_UVW) {
            if (i < uvw_index) {
                float x1a = xyz1a[i], x1b = xyz1b[i], x1c = xyz1c[i];
                float x2a = xyz2a[i], x2b = xyz2b[i], x2c = xyz2c[i];

                su[threadIdx.x] = u[i];
                sv[threadIdx.x] = v[i];
                sw[threadIdx.x] = w[i];
                sf[threadIdx.x] = f[i];

                sx1a[threadIdx.x] = x1a;
                sx1b[threadIdx.x] = x1b;
                sx1c[threadIdx.x] = x1c;

                sx2a[threadIdx.x] = x2a;
                sx2b[threadIdx.x] = x2b;
                sx2c[threadIdx.x] = x2c;

                float inv1 = rsqrtf(x1a * x1a + x1b * x1b + x1c * x1c + 1e-20f);
                float inv2 = rsqrtf(x2a * x2a + x2b * x2b + x2c * x2c + 1e-20f);
                sInv1[threadIdx.x] = inv1;
                sInv2[threadIdx.x] = inv2;

                sV[threadIdx.x] = make_float2(Viss[i].real(), Viss[i].imag());
            } else {
                su[threadIdx.x] = 0.0f; sv[threadIdx.x] = 0.0f; sw[threadIdx.x] = 0.0f; sf[threadIdx.x] = 0.0f;
                sx1a[threadIdx.x] = 0.0f; sx1b[threadIdx.x] = 0.0f; sx1c[threadIdx.x] = 0.0f;
                sx2a[threadIdx.x] = 0.0f; sx2b[threadIdx.x] = 0.0f; sx2c[threadIdx.x] = 0.0f;
                sInv1[threadIdx.x] = 0.0f; sInv2[threadIdx.x] = 0.0f;
                sV[threadIdx.x] = make_float2(0.0f, 0.0f);
            }
        }

        __syncthreads();

        int tileN = min(TILE_UVW, uvw_index - base);
        #pragma unroll 4
        for (int t = 0; t < tileN; ++t) {
            // FOV 判定
            float dot1 = lp * sx1a[t] + mp * sx1b[t] + np_ * sx1c[t];
            float dot2 = lp * sx2a[t] + mp * sx2b[t] + np_ * sx2c[t];

            float c1 = dot1 * sInv1[t];
            float c2 = dot2 * sInv2[t];

            float pen = (c1 >= cosphi && c2 >= cosphi) ? sf[t] : 0.0f;
            if (pen == 0.0f) continue;

            // exp(i 2*pi*(u*l+v*m+w*n))
            float phase = su[t] * lp + sv[t] * mp + sw[t] * np_;
            float ang = 2.0f * (float)M_PI * phase;

            float s, c;
            sincos_fast(ang, &s, &c);

            float a = sV[t].x;
            float b = sV[t].y;

            acc_re += pen * (a * c - b * s);
            acc_im += pen * (a * s + b * c);
        }

        __syncthreads();
    }

    C[pix] = Complex(acc_re, acc_im);
}

// ===============================
// Kernel: extract real(C) 便于写文件
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
// 更快的文本读取：FILE + fgets + strtof
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
// 频率 -> 输入路径/标签（1MHz / 10MHz）
// ===============================
static inline bool is_1mhz(float f) { return (fabsf(f - 1.0e6f) < 1.0e5f); }

static inline void build_paths(float frequency, std::string& base_dir, std::string& tag_short, std::string& tag_file) {
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
// timer
// ===============================
static timeval start_tv, finish_tv;
static float total_time = 0.0f;

// ===============================
// main pipeline
// ===============================
int vissGen(float frequency)
{
    gettimeofday(&start_tv, NULL);

    int nDevices = 0;
    CHECK(cudaGetDeviceCount(&nDevices));
    omp_set_num_threads(nDevices);

    cout << "devices: " << nDevices << endl;
    cout << "frequency: " << frequency << endl;

    int startday = 0;
    int days = 1;
    cout << "startday: " << startday << endl;
    cout << "days: " << days << endl;

    // 频率相关路径/标签
    std::string address, tag_short, tag_file;
    build_paths(frequency, address, tag_short, tag_file);
    cout << "address: " << address << endl;

    // 读取 B/theta/phi
    string address_B          = address + "B_" + tag_file + ".txt";
    string address_theta_heal = address + "theta_heal_" + tag_file + ".txt";
    string address_phi_heal   = address + "phi_heal_" + tag_file + ".txt";

    ifstream BFile(address_B);
    ifstream thetaFile(address_theta_heal);
    ifstream phiFile(address_phi_heal);

    if (!BFile.is_open() || !thetaFile.is_open() || !phiFile.is_open()) {
        cout << "ERROR: cannot open B/theta/phi:\n"
             << address_B << "\n" << address_theta_heal << "\n" << address_phi_heal << endl;
        return -1;
    }

    int npix = 0;
    BFile >> npix;
    cout << "npix: " << npix << endl;

    vector<float> cB(npix), ctheta(npix), cphi(npix);
    for (int i = 0; i < npix; ++i) {
        BFile >> cB[i];
        thetaFile >> ctheta[i];
        phiFile >> cphi[i];
    }
    BFile.close(); thetaFile.close(); phiFile.close();
    cout << "load B/theta/phi success" << endl;

    // nside/s/res
    int nside = (int)lround(sqrt((double)npix / 12.0));
    float s   = 4.0f * (float)M_PI / (float)npix;
    float res = sqrtf(4.0f * (float)M_PI / (float)npix);

    // 遮挡半视场角
    float R = 1737.1e3f;
    float h = 300e3f;
    float theta = asinf(R / (R + h));
    float phi   = (float)M_PI - theta;

    cout << "nside: " << nside << endl;
    cout << "s: " << s << endl;
    cout << "res: " << res << endl;
    cout << "theta: " << theta << endl;
    cout << "phi: " << phi << endl;

    // host 上先缩放 B（避免每 day 再做一次）
    vector<float> cB_scaled = cB;
    for (int i = 0; i < npix; ++i) cB_scaled[i] *= s;

    // 输出目录：10M -> 3dblockage10M, 1M -> 3dblockage1M
    std::string out_dir = "3dblockage" + tag_short + "/";

    // OpenMP：每线程一张 GPU
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        CHECK(cudaSetDevice(tid));
        CHECK(cudaDeviceSynchronize());
        cout << "Thread " << tid << " on device " << tid << endl;

        cudaStream_t stream;
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        // 常驻：B + lmn + C + Creal（每 GPU 只算一次 lmn）
        thrust::device_vector<float> d_B(cB_scaled.begin(), cB_scaled.end());
        thrust::device_vector<float> d_theta(ctheta.begin(), ctheta.end());
        thrust::device_vector<float> d_phi(cphi.begin(), cphi.end());

        thrust::device_vector<float> d_l(npix), d_m(npix), d_n(npix);

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

        // theta/phi 不再需要
        d_theta.clear(); d_theta.shrink_to_fit();
        d_phi.clear();   d_phi.shrink_to_fit();

        thrust::device_vector<Complex> d_C(npix);
        thrust::device_vector<float>   d_Creal(npix);

        // uvw/xyz host buffers（按 UVW_MAX 复用）
        std::vector<float> cu(UVW_MAX), cv(UVW_MAX), cw(UVW_MAX), cf(UVW_MAX);
        std::vector<float> cxyz1a(UVW_MAX), cxyz1b(UVW_MAX), cxyz1c(UVW_MAX);
        std::vector<float> cxyz2a(UVW_MAX), cxyz2b(UVW_MAX), cxyz2c(UVW_MAX);

        // device buffers（复用）
        thrust::device_vector<float> d_u(UVW_MAX), d_v(UVW_MAX), d_w(UVW_MAX), d_f(UVW_MAX);
        thrust::device_vector<float> d_xyz1a(UVW_MAX), d_xyz1b(UVW_MAX), d_xyz1c(UVW_MAX);
        thrust::device_vector<float> d_xyz2a(UVW_MAX), d_xyz2b(UVW_MAX), d_xyz2c(UVW_MAX);
        thrust::device_vector<Complex> d_Viss(UVW_MAX);

        // pinned host chunk for save
        float* h_chunk = nullptr;
        const int CHUNK = 1 << 20; // ~4MB
        CHECK(cudaMallocHost(&h_chunk, (size_t)CHUNK * sizeof(float)));

        for (int p = tid + startday; p < days; p += nDevices) {
            int day = p + 1;
            cout << "GPU " << tid << " processing day " << day << endl;

            int uvw_index = 0, xyz1_index = 0, xyz2_index = 0;

            // 文件名：10M/1M 只差 tag_short
            // uvw：updated_uvw{day}day{tag_short}.txt（4列 u v w f_point）
            // xyz：xyza/xyzb{day}day{tag_short}.txt（3列）
            std::string suf = "day" + tag_short + ".txt";
            std::string address_uvw  = address + "updated_uvw" + to_string(day) + suf;
            std::string address_xyz1 = address + "xyza" + to_string(day) + suf;
            std::string address_xyz2 = address + "xyzb" + to_string(day) + suf;

            // 读 uvw + f_point（host 上做 cf=1/f_point）
            {
                // 临时读到 cf 里先放 f_point，再转 1/f_point
                std::vector<float> tmpf(UVW_MAX);

                if (!load_quad(address_uvw, cu.data(), cv.data(), cw.data(), tmpf.data(), UVW_MAX, uvw_index)) {
                    cout << "ERROR read " << address_uvw << endl;
                    continue;
                }
                for (int i = 0; i < uvw_index; ++i) {
                    float fp = tmpf[i];
                    cf[i] = (fp != 0.0f) ? (1.0f / fp) : 0.0f;
                }
            }

            // 读 xyz1/xyz2
            if (!load_triplets(address_xyz1, cxyz1a.data(), cxyz1b.data(), cxyz1c.data(), UVW_MAX, xyz1_index)) {
                cout << "ERROR read " << address_xyz1 << endl;
                continue;
            }
            if (!load_triplets(address_xyz2, cxyz2a.data(), cxyz2b.data(), cxyz2c.data(), UVW_MAX, xyz2_index)) {
                cout << "ERROR read " << address_xyz2 << endl;
                continue;
            }

            if (uvw_index <= 0 || xyz1_index != uvw_index || xyz2_index != uvw_index) {
                cout << "ERROR: index mismatch uvw=" << uvw_index
                     << " xyz1=" << xyz1_index << " xyz2=" << xyz2_index << endl;
                continue;
            }
            if (uvw_index > UVW_MAX) {
                cout << "ERROR: uvw_index exceeds UVW_MAX" << endl;
                continue;
            }

            // H2D async
            CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_u.data()), cu.data(),
                                  uvw_index * sizeof(float), cudaMemcpyHostToDevice, stream));
            CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_v.data()), cv.data(),
                                  uvw_index * sizeof(float), cudaMemcpyHostToDevice, stream));
            CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_w.data()), cw.data(),
                                  uvw_index * sizeof(float), cudaMemcpyHostToDevice, stream));
            CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_f.data()), cf.data(),
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

            CHECK(cudaStreamSynchronize(stream));

            // --- Viss ---
            {
                // baseline 维度线程块（建议 128：给每线程更多寄存器，且 shared sky tile 复用效果好）
                constexpr int BASE_BLOCK = 128;
                int grid = (uvw_index + BASE_BLOCK - 1) / BASE_BLOCK;

                // sky tile = 256
                constexpr int TILE_PIX = 256;
                size_t shmem = (size_t)TILE_PIX * 4 * sizeof(float); // B,l,m,n

                healpix_moonback_viss_tiled<TILE_PIX><<<grid, BASE_BLOCK, shmem, stream>>>(
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

            // --- Viss phase correction（只对 0..uvw_index-1 需要，因为 computeC 用 uvw_index） ---
            {
                constexpr int BLOCK = 256;
                int grid = (uvw_index + BLOCK - 1) / BLOCK;
                phase_correct_viss<<<grid, BLOCK, 0, stream>>>(
                    thrust::raw_pointer_cast(d_Viss.data()),
                    thrust::raw_pointer_cast(d_w.data()),
                    uvw_index
                );
                CHECK(cudaPeekAtLastError());
            }

            // --- computeC (blockage) ---
            {
                CHECK(cudaMemsetAsync(thrust::raw_pointer_cast(d_C.data()), 0,
                                      (size_t)npix * sizeof(Complex), stream));

                constexpr int BLOCK = 256;
                int grid = (npix + BLOCK - 1) / BLOCK;

                constexpr int TILE_UVW = 256;
                // shared bytes：
                //  su,sv,sw,sf (4) + xyz1(3) + xyz2(3) + inv1,inv2 (2) = 12 float arrays
                //  + Viss float2
                size_t shmem =
                    (size_t)TILE_UVW * (12 * sizeof(float) + sizeof(float2));

                computeC_blockage_tiled<TILE_UVW><<<grid, BLOCK, shmem, stream>>>(
                    npix,
                    thrust::raw_pointer_cast(d_u.data()),
                    thrust::raw_pointer_cast(d_v.data()),
                    thrust::raw_pointer_cast(d_w.data()),
                    thrust::raw_pointer_cast(d_f.data()),
                    thrust::raw_pointer_cast(d_xyz1a.data()),
                    thrust::raw_pointer_cast(d_xyz1b.data()),
                    thrust::raw_pointer_cast(d_xyz1c.data()),
                    thrust::raw_pointer_cast(d_xyz2a.data()),
                    thrust::raw_pointer_cast(d_xyz2b.data()),
                    thrust::raw_pointer_cast(d_xyz2c.data()),
                    thrust::raw_pointer_cast(d_l.data()),
                    thrust::raw_pointer_cast(d_m.data()),
                    thrust::raw_pointer_cast(d_n.data()),
                    thrust::raw_pointer_cast(d_Viss.data()),
                    thrust::raw_pointer_cast(d_C.data()),
                    phi,
                    uvw_index
                );
                CHECK(cudaPeekAtLastError());
            }

            // extract real
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

            // save (分块 D2H)
            {
                string out_path = out_dir + "C" + to_string(day) + "day" + tag_short + "_half_freq.txt";
                cout << "GPU " << tid << " saving: " << out_path << endl;

                std::ofstream file(out_path);
                if (!file.is_open()) {
                    cout << "ERROR open output file: " << out_path << endl;
                } else {
                    // 提升输出缓冲
                    static const size_t OUT_BUF_SZ = 8 << 20;
                    static thread_local std::vector<char> outbuf(OUT_BUF_SZ);
                    file.rdbuf()->pubsetbuf(outbuf.data(), outbuf.size());

                    float* dptr = thrust::raw_pointer_cast(d_Creal.data());

                    for (int off = 0; off < npix; off += CHUNK) {
                        int cur = std::min(CHUNK, npix - off);
                        CHECK(cudaMemcpy(h_chunk, dptr + off, (size_t)cur * sizeof(float), cudaMemcpyDeviceToHost));
                        for (int i = 0; i < cur; ++i) {
                            file << h_chunk[i] << "\n";
                        }
                    }
                    file.close();
                }
            }
        }

        CHECK(cudaFreeHost(h_chunk));
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
    return vissGen(freq);
}
