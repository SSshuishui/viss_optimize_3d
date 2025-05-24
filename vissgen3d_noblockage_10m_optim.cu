#include <cstdio>
#include <iostream>
#include <ctime>
#include <string>
#include <cmath>
#include <omp.h>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "error.cuh"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <thrust/complex.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>

#define _USE_MATH_DEFINES
#define EXP 0.0000000000

using namespace std;
using Complex = thrust::complex<float>;
const int uvw_presize = 4000000;


// complexExp 函数的实现
__device__ thrust::complex<float> complexExp(Complex d) {
    return thrust::exp(d);
}

// complexAbs 函数的实现
__device__ thrust::complex<float> ComplexAbs(const Complex &d) {
    // 复数的模定义为 sqrt(real^2 + imag^2)
    return thrust::complex<float>(sqrt(d.real() * d.real() + d.imag() * d.imag()));
}

__device__ float norm(float x, float y, float z) {
    return sqrtf(x * x + y * y + z * z);
}

struct timeval start, finish;
float total_time;

string address = "/data/zhaox/earth_10Mhz/";


__global__ void healpix_moonback_pre(
    float * __restrict__ theta_heal, 
    float * __restrict__ phi_heal,
    float * __restrict__ l, 
    float * __restrict__ m, 
    float * __restrict__ n,
    float * __restrict__ B, 
    int npix, 
    float s)
{
    // 获取线程索引
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= npix) return;

    // 预先加载频繁使用的数据到寄存器
    float theta_val = theta_heal[index];
    float phi_val = phi_heal[index];

    // 优化角度计算
    theta_val = M_PI / 2 - theta_val;
    if (phi_val > M_PI) {
        phi_val -= 2 * M_PI;
    }
    phi_val = -phi_val;

    // 计算l, m, n
    float cos_theta = cosf(theta_val);
    l[index] = cos_theta * cosf(phi_val);
    m[index] = cos_theta * sinf(phi_val);
    n[index] = sinf(theta_val);

    // 更新theta_heal和phi_heal
    theta_heal[index] = theta_val;
    phi_heal[index] = phi_val;

    // 缩放B
    B[index] *= s;
}

void launch_healpix_moonback_pre(
    float *d_theta_heal, 
    float *d_phi_heal,
    float *d_l, 
    float *d_m, 
    float *d_n,
    float *d_B, 
    int npix, 
    float s)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, healpix_moonback_pre, 0, 0);
    int blocksPerGrid = floor(npix + threadsPerBlock - 1) / threadsPerBlock;

    // 设置缓存配置以优化共享内存的使用
    cudaFuncSetCacheConfig(healpix_moonback_pre, cudaFuncCachePreferL1);

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 启动核函数
    healpix_moonback_pre<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_theta_heal, d_phi_heal, d_l, d_m, d_n, d_B, npix, s);

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 销毁流
    cudaStreamDestroy(stream);
}


__global__ void healpix_moonback_viss(
    float * __restrict__ B, 
    Complex * __restrict__ Viss,
    float * __restrict__ u, 
    float * __restrict__ v, 
    float * __restrict__ w,
    float * __restrict__ xyz1a, 
    float * __restrict__ xyz1b, 
    float * __restrict__ xyz1c, 
    float * __restrict__ xyz2a, 
    float * __restrict__ xyz2b, 
    float * __restrict__ xyz2c,
    float * __restrict__ l, 
    float * __restrict__ m, 
    float * __restrict__ n, 
    int amount, 
    int npix, 
    float phi,
    Complex zero, 
    Complex I1, 
    Complex two, 
    Complex CPI)  
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= amount) return;

    // 预先加载频繁使用的数据到寄存器
    float u_val = u[i];
    float v_val = v[i];
    float w_val = w[i];
    float xyz1a_val = xyz1a[i];
    float xyz1b_val = xyz1b[i];
    float xyz1c_val = xyz1c[i];
    float xyz2a_val = xyz2a[i];
    float xyz2b_val = xyz2b[i];
    float xyz2c_val = xyz2c[i];
    float norm1 = norm(xyz1a_val, xyz1b_val, xyz1c_val);
    float norm2 = norm(xyz2a_val, xyz2b_val, xyz2c_val);

    Complex acc = zero;
    for (int index = 0; index < npix; index++) {
        // 天空每个点与视场中心的夹角
        float gb1_comp = l[index] * xyz1a_val + m[index] * xyz1b_val + n[index] * xyz1c_val;
        float gb2_comp = l[index] * xyz2a_val + m[index] * xyz2b_val + n[index] * xyz2c_val;
        float beta1 = acosf(gb1_comp / norm1);
        float beta2 = acosf(gb2_comp / norm2);

        if (beta1 <= phi && beta2 <= phi) {
            float phase = u_val * l[index] + v_val * m[index] + w_val * (n[index] - 1.0f);
            Complex vari(phase, 0.0f);
            acc += Complex(B[index], 0) * thrust::exp((zero - I1) * two * CPI * vari);
        }
    }

    Viss[i] = acc;
    Viss[i + amount] = thrust::conj(acc);
}

// 启动核函数的包装函数
void launch_healpix_moonback_viss(
    float *d_B, 
    Complex *d_Viss,
    float *d_u, 
    float *d_v, 
    float *d_w,
    float *d_xyz1a, 
    float *d_xyz1b, 
    float *d_xyz1c, 
    float *d_xyz2a, 
    float *d_xyz2b, 
    float *d_xyz2c,
    float *d_l, 
    float *d_m, 
    float *d_n, 
    int amount, 
    int npix, 
    float phi,
    Complex zero, 
    Complex I1, 
    Complex two, 
    Complex CPI)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, healpix_moonback_viss, 0, 0);
    int blocksPerGrid = floor(amount + threadsPerBlock - 1) / threadsPerBlock;


    // 设置缓存配置以优化共享内存的使用
    cudaFuncSetCacheConfig(healpix_moonback_viss, cudaFuncCachePreferL1);

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 启动核函数
    healpix_moonback_viss<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_B, d_Viss, d_u, d_v, d_w, d_xyz1a, d_xyz1b, d_xyz1c, 
        d_xyz2a, d_xyz2b, d_xyz2c, d_l, d_m, d_n, amount, npix, phi, 
        zero, I1, two, CPI);

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 销毁流
    cudaStreamDestroy(stream);
}


__global__ void ceilAndScale(
    float* __restrict__ bll, 
    int* __restrict__ sort_bll, 
    int* __restrict__ gs, 
    int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // 预先加载频繁使用的数据到寄存器
    float bll_val = bll[idx];

    // 计算排序后的bll并缩放
    int sort_bll_val = ceilf((bll_val - 1.0f / 4.0f) / 0.5f);
    sort_bll[idx] = sort_bll_val;
    gs[idx] = sort_bll_val;
}

// 启动核函数的包装函数
void launch_ceilAndScale(
    float* d_bll, 
    int* d_sort_bll, 
    int* d_gs, 
    int size)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, ceilAndScale, 0, 0);
    int blocksPerGrid = floor(size + threadsPerBlock - 1) / threadsPerBlock;

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 启动核函数
    ceilAndScale<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_bll, d_sort_bll, d_gs, size);

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 销毁流
    cudaStreamDestroy(stream);
}

__global__ void countOccurrences(
    int* __restrict__ s, 
    float* __restrict__ mb, 
    int size, 
    int nr) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // 预先加载频繁使用的数据到寄存器
    int s_val = s[idx];

    for (int p = 1; p <= nr; ++p) {
        if (s_val == p) {
            atomicAdd(&mb[p-1], 1.0f);
        }
    }
}

// 启动核函数的包装函数
void launch_countOccurrences(
    int* d_s, 
    float* d_mb, 
    int size, 
    int nr)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, countOccurrences, 0, 0);
    int blocksPerGrid = floor(size + threadsPerBlock - 1) / threadsPerBlock;

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 启动核函数
    countOccurrences<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_s, d_mb, size, nr);

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 销毁流
    cudaStreamDestroy(stream);
}

__global__ void calculateR(float* __restrict__ R, int nr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nr) {
        R[idx] = (idx + 1) / 2.0f + 1.0f / 4.0f;
    }
}

__global__ void calculateDcf(float* __restrict__ dcf, float* __restrict__ R, float* __restrict__ mb, int nr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nr) {
        if (idx == 0){
            dcf[idx] = 1.0f / M_PI / 4.0f;
        }else{
            dcf[idx] = 2.0f / 3.0f * M_PI * (pow(R[idx-1], 3) - pow(R[idx-1] - 0.5, 3)) / mb[idx-1];
        }
    }
}


__global__ void viss_gamma_trans(
    Complex* __restrict__ Viss, 
    float* __restrict__ w, 
    float* __restrict__ bll, 
    float* __restrict__ gamma, 
    float* __restrict__ dg, 
    int size, 
    Complex zero, 
    Complex two, 
    Complex CPI, 
    Complex I1) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // 预先加载频繁使用的数据到寄存器
    float w_val = w[idx];
    float bll_val = bll[idx];

    // gViss = gViss * exp(-1i * 2 * M_PI * gw)
    Complex cw(w_val, 0.0f);
    Viss[idx] = Viss[idx] * complexExp((zero - I1) * two * CPI * cw);
    
    // ggamma = asin(gw / gbll)
    float gamma_val = asinf(w_val / bll_val);
    gamma[idx] = gamma_val;

    // gdg = abs((complex(1-4*(sin(ggamma)).^2)).^0.5./cos(ggamma)*3/2)
    float sin_gamma = sinf(gamma_val);
    float cos_gamma = cosf(gamma_val);
    thrust::complex<float> value(1.0f - 4.0f * sin_gamma * sin_gamma, 0.0f);
    thrust::complex<float> sqrt_result = thrust::sqrt(value);
    float magnitude = thrust::abs(sqrt_result);
    dg[idx] = magnitude / fabsf(cos_gamma) * 3.0f / 2.0f;
}

// 启动核函数的包装函数
void launch_viss_gamma_trans(
    Complex* d_Viss, 
    float* d_w, 
    float* d_bll, 
    float* d_gamma, 
    float* d_dg, 
    int size, 
    Complex zero, 
    Complex two, 
    Complex CPI, 
    Complex I1)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, viss_gamma_trans, 0, 0);
    int blocksPerGrid = floor(size + threadsPerBlock - 1) / threadsPerBlock;

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 启动核函数
    viss_gamma_trans<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_Viss, d_w, d_bll, d_gamma, d_dg, size, zero, two, CPI, I1);

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 销毁流
    cudaStreamDestroy(stream);
}


__global__ void computeC(
    int npix, 
    float* __restrict__ dcf, 
    float* __restrict__ dg, 
    int* __restrict__ gs, 
    float* __restrict__ u, 
    float* __restrict__ v, 
    float* __restrict__ w,
    float* __restrict__ l, 
    float* __restrict__ m, 
    float* __restrict__ n, 
    Complex* __restrict__ C, 
    Complex* __restrict__ Viss, 
    Complex I1, 
    Complex two, 
    Complex CPI, 
    int uvw_index) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npix) return;

    // 预先加载频繁使用的数据到寄存器
    float l_val = l[idx];
    float m_val = m[idx];
    float n_val = n[idx];

    Complex acc = C[idx];

    for (int i = 0; i < uvw_index; i++) {
        float dcf_val = fminf(dcf[gs[i]] * dg[i], 1.0f / 8.0f);
        float phase = u[i] * l_val + v[i] * m_val + w[i] * n_val;
        Complex PhaseDifference(phase, 0.0f);
        acc += Viss[i] * Complex(dcf_val, 0.0f) * complexExp(two * CPI * I1 * PhaseDifference);
    }

    C[idx] = acc;
}

// 启动核函数的包装函数
void launch_computeC(
    int npix, 
    float* d_dcf, 
    float* d_dg, 
    int* d_gs, 
    float* d_u, 
    float* d_v, 
    float* d_w,
    float* d_l, 
    float* d_m, 
    float* d_n, 
    Complex* d_C, 
    Complex* d_Viss, 
    Complex I1, 
    Complex two, 
    Complex CPI, 
    int uvw_index)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, computeC, 0, 0);
    int blocksPerGrid = floor(npix + threadsPerBlock - 1) / threadsPerBlock;


    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 启动核函数
    computeC<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        npix, d_dcf, d_dg, d_gs, d_u, d_v, d_w, d_l, d_m, d_n, d_C, d_Viss, I1, two, CPI, uvw_index);

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 销毁流
    cudaStreamDestroy(stream);
}


int vissGen(float frequency) 
{   
    gettimeofday(&start, NULL);

    int nDevices;
    // 设置节点数量（gpu显卡数量）
    CHECK(cudaGetDeviceCount(&nDevices));
    // 设置并行区中的线程数
    omp_set_num_threads(nDevices);
    cout << "devices: " << nDevices << endl;

    cout << "frequency: " << frequency << endl;

    int days = 226;
    int start_day = 223;
    Complex I1(0.0, 1.0);
    Complex zero(0.0, 0.0);
    Complex one(1.0, 0.0);
    Complex two(2.0, 0.0);
    Complex CPI(M_PI, 0.0);
    cout << "days: " << days << endl;
    cout << "start day: " << start_day << endl;

    // 读取 B.txt, theta_heal.txt, phi_heal.txt 文件
    string address_B = address + "B_10Mhz.txt";
    string address_theta_heal = address + "theta_heal_10Mhz.txt";
    string address_phi_heal = address + "phi_heal_10Mhz.txt";
    ifstream BFile, thetaFile, phiFile;
    BFile.open(address_B);
    thetaFile.open(address_theta_heal);
    phiFile.open(address_phi_heal);

    int npix = 0;
    BFile >> npix;  // 读取第一行的数据，也就是总数据行数
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

    cout << "load B.txt, theta_heal.txt, phi_heal.txt in CPU success" << endl;
    
    int nside=round(sqrt(npix/12));
    float s=4*M_PI/npix;
    float res=sqrt(4*M_PI/npix);

    // f_pix2and_nest 函数调用，获得每个点的
    thrust::device_vector<int> jrll = {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    thrust::device_vector<int> jpll = {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};

    float R=1737.1e3;  // 月球半径
    float h=300e3;     // 卫星轨道高度
    float theta= asinf(R/(R+h));  // 单颗卫星的月背遮挡区的半视场角
    float phi= M_PI-theta;   // 每条基线的半视场角，每条基线对应一个半视场角
    cout << "theta: " << theta << endl;
    cout << "phi: " << phi << endl;

    // recon的参数
    float bl_max=100e3;
    float lamda = 3e8 / frequency;
    float nr=ceil(bl_max/lamda*2); // 环的数量 半个波长一个环

    cout << "nside: " << nside << endl;
    cout << "s: " << s << endl;
    cout << "res: " << res << endl;
    cout << "theta: " << theta << endl;
    cout << "lamda: " << lamda << endl;
    cout << "nr: " << nr << endl;


    // 开启cpu线程并行
    // 一个线程处理1个GPU
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        cudaSetDevice(tid);
        CHECK(cudaDeviceSynchronize());
        std::cout << "Thread " << tid << " is running on device " << tid << endl;

        // 遍历所有开启的线程处理， 一个线程控制一个GPU 处理一个id*amount/total的块
        for (int p = tid + start_day; p < days; p += nDevices) {
            cout << "for loop: " << p+1 << endl;

            // 将 B, theta_heal, phi_heal 数据从CPU搬到GPU上        
            thrust::device_vector<float> B(cB.begin(), cB.end());
            thrust::device_vector<float> theta_heal(ctheta_heal.begin(), ctheta_heal.end());
            thrust::device_vector<float> phi_heal(cphi_heal.begin(), cphi_heal.end());

            // 创建临时变量
            thrust::device_vector<float> l(npix), m(npix), n(npix);

            std::vector<float> cu(uvw_presize), cv(uvw_presize), cw(uvw_presize);
            thrust::device_vector<float> u(uvw_presize), v(uvw_presize), w(uvw_presize);

            std::vector<float> cxyz1a(uvw_presize), cxyz1b(uvw_presize), cxyz1c(uvw_presize);
            thrust::device_vector<float> xyz1a(uvw_presize), xyz1b(uvw_presize), xyz1c(uvw_presize);

            std::vector<float> cxyz2a(uvw_presize), cxyz2b(uvw_presize), cxyz2c(uvw_presize);
            thrust::device_vector<float> xyz2a(uvw_presize), xyz2b(uvw_presize), xyz2c(uvw_presize);

            std::vector<float> cbll(uvw_presize);
            thrust::device_vector<float> bll(uvw_presize);

            thrust::device_vector<Complex> Viss(uvw_presize);
            
            // 存储计算后的到的最终结果
            thrust::device_vector<Complex> C(npix);

            int uvw_index, xyz1_index, xyz2_index, bll_index; 
            #pragma omp critical
            {   
                // 读取 uvw
                string address_uvw = address + "uvw" + to_string(p+1) + "day1M.txt";
                cout << "address_uvw: " << address_uvw << endl;
                ifstream uvwFile(address_uvw);
                uvw_index = 0;
                float u_point, v_point, w_point;
                if (uvwFile.is_open()) {
                    uvwFile >> u_point >> v_point >> w_point; // 读取第一行，删除
                    while (uvwFile >> u_point >> v_point >> w_point) {
                        // cu, cv, cw 需要存储原始坐标
                        cu[uvw_index] = u_point;
                        cv[uvw_index] = v_point;
                        cw[uvw_index] = w_point;
                        uvw_index++;
                    }
                }
                cout << "uvw_index: " << uvw_index << endl;
                // 复制到GPU上
                thrust::copy(cu.begin(), cu.begin() + uvw_index, u.begin());
                thrust::copy(cv.begin(), cv.begin() + uvw_index, v.begin());
                thrust::copy(cw.begin(), cw.begin() + uvw_index, w.begin());
                
                // 读取 xyz1(xyza)
                string address_xyz1 = address + "xyza" + to_string(p+1) + "day1M.txt";
                cout << "address_xyz1: " << address_xyz1 << endl;
                ifstream xyz1File(address_xyz1);
                xyz1_index = 0;
                float a_point, b_point, c_point;
                if (xyz1File.is_open()) {
                    xyz1File >> a_point >> b_point >> c_point;
                    while (xyz1File >> a_point >> b_point >> c_point) {
                        cxyz1a[xyz1_index] = a_point;
                        cxyz1b[xyz1_index] = b_point;
                        cxyz1c[xyz1_index] = c_point;
                        xyz1_index++;
                    }
                }
                cout << "xyz1_index: " << xyz1_index << endl;
                // 复制到GPU上
                thrust::copy(cxyz1a.begin(), cxyz1a.begin() + xyz1_index, xyz1a.begin());
                thrust::copy(cxyz1b.begin(), cxyz1b.begin() + xyz1_index, xyz1b.begin());
                thrust::copy(cxyz1c.begin(), cxyz1c.begin() + xyz1_index, xyz1c.begin());
                
                // 读取 xyz2(xyzb)
                string address_xyz2 = address + "xyzb" + to_string(p+1) + "day1M.txt";
                cout << "address_xyz2: " << address_xyz2 << endl;
                ifstream xyz2File(address_xyz2);
                xyz2_index = 0;
                if (xyz2File.is_open()) {
                    xyz2File >> a_point >> b_point >> c_point;
                    while (xyz2File >> a_point >> b_point >> c_point) {
                        cxyz2a[xyz2_index] = a_point;
                        cxyz2b[xyz2_index] = b_point;
                        cxyz2c[xyz2_index] = c_point;
                        xyz2_index++;
                    }
                }
                cout << "xyz2_index: " << xyz2_index << endl;
                // 复制到GPU上
                thrust::copy(cxyz2a.begin(), cxyz2a.begin() + xyz2_index, xyz2a.begin());
                thrust::copy(cxyz2b.begin(), cxyz2b.begin() + xyz2_index, xyz2b.begin());
                thrust::copy(cxyz2c.begin(), cxyz2c.begin() + xyz2_index, xyz2c.begin());

                // 读取 bll
                string address_bll = address + "bll" + to_string(p+1) + "day1M.txt";
                cout << "address_bll: " << address_bll << endl;
                ifstream bllFile(address_bll);
                bll_index = 0;
                if (bllFile.is_open()) {
                    bllFile >> a_point;
                    while (bllFile >> a_point) {
                        cbll[bll_index] = a_point;
                        bll_index++;
                    }
                }
                cout << "bll_index: " << bll_index << endl;
                // 复制到GPU上
                thrust::copy(cbll.begin(), cbll.begin() + bll_index, bll.begin());

                // // 读取Viss，测试时使用
                // int viss_index;
                // string address_viss = address + "Viss" + to_string(p+1) + "day1M.txt";
                // cout << "address_viss: " << address_viss << endl;
                // ifstream vissFile(address_viss);
                // viss_index = 0;
                // if (vissFile.is_open()) {
                //     vissFile >> a_point >> b_point;
                //     while (vissFile >> a_point >> b_point) {
                //         cViss[viss_index].real(a_point);
                //         cViss[viss_index].imag(b_point);
                //         viss_index++;
                //     }
                // }
                // cout << "viss_index: " << viss_index << endl;
                // // 复制到GPU上
                // thrust::copy(cViss.begin(), cViss.begin() + viss_index, Viss.begin());
            }

            // 计算可见度
            int amount = ceil(uvw_index/2);
            cout << "amount: " << amount << endl;
            
            launch_healpix_moonback_pre(
                thrust::raw_pointer_cast(theta_heal.data()), 
                thrust::raw_pointer_cast(phi_heal.data()),
                thrust::raw_pointer_cast(l.data()),
                thrust::raw_pointer_cast(m.data()),
                thrust::raw_pointer_cast(n.data()),
                thrust::raw_pointer_cast(B.data()),
                npix, s);
            CHECK(cudaDeviceSynchronize());


            
            printf("Viss Computing... Here is gpu %d running process %d\n", omp_get_thread_num(), p+1);
            launch_healpix_moonback_viss(
                thrust::raw_pointer_cast(B.data()),
                thrust::raw_pointer_cast(Viss.data()),
                thrust::raw_pointer_cast(u.data()),
                thrust::raw_pointer_cast(v.data()),
                thrust::raw_pointer_cast(w.data()),
                thrust::raw_pointer_cast(xyz1a.data()),
                thrust::raw_pointer_cast(xyz1b.data()),
                thrust::raw_pointer_cast(xyz1c.data()),
                thrust::raw_pointer_cast(xyz2a.data()),
                thrust::raw_pointer_cast(xyz2b.data()),
                thrust::raw_pointer_cast(xyz2c.data()),
                thrust::raw_pointer_cast(l.data()),
                thrust::raw_pointer_cast(m.data()),
                thrust::raw_pointer_cast(n.data()),
                amount, npix, phi,
                zero, I1, two, CPI);
            CHECK(cudaDeviceSynchronize());
            cout << "Period " << p+1 << " Viss Computing Success!" << endl;
            for (int i=0; i<=2; i++){
                cout << "Viss[" << i << "]: " << Viss[i] << endl;
            }


            // 图像重构
            thrust::device_vector<float> mb(nr);
            thrust::device_vector<int> sort_bll(bll_index);
            thrust::device_vector<int> gs(bll_index);
            launch_ceilAndScale(
                thrust::raw_pointer_cast(bll.data()),
                thrust::raw_pointer_cast(sort_bll.data()), 
                thrust::raw_pointer_cast(gs.data()), 
                bll_index);
            CHECK(cudaDeviceSynchronize());
            cout << "Period " << p+1 << " ceilAndScale..." << endl;


            thrust::sort(sort_bll.begin(), sort_bll.end());   // s=sort(ceil((bll-1/4)/0.5));
            cout << "Period " << p+1 << " Sort..." << endl;
            
            launch_countOccurrences(
                thrust::raw_pointer_cast(sort_bll.data()), 
                thrust::raw_pointer_cast(mb.data()), 
                sort_bll.size(), nr);
            CHECK(cudaDeviceSynchronize());
            cout << "Period " << p+1 << " countOccurrences..." << endl;


            thrust::device_vector<float> dcfR(nr);
            // R=[1:nr]';   R=R/2+1/4;
            calculateR<<<(nr + 255) / 256, 256>>>(thrust::raw_pointer_cast(dcfR.data()), nr);
            CHECK(cudaDeviceSynchronize());
            cout << "Period " << p+1 << " calculateR..." << endl;

            int minGridSize, blockSize, gridSize;
            thrust::device_vector<float> dcf(nr+1);
            // dcf=2/3*pi*(R.^3-(R-1/2).^3)./mb;    dcf=[1/pi/4;dcf];
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculateDcf, 0, 0);
            gridSize = floor((nr+1) + blockSize - 1) / blockSize;
            calculateDcf<<<gridSize, blockSize>>>(
                thrust::raw_pointer_cast(dcf.data()), 
                thrust::raw_pointer_cast(dcfR.data()), 
                thrust::raw_pointer_cast(mb.data()), 
                nr+1);
            CHECK(cudaDeviceSynchronize());
            cout << "Period " << p+1 << " calculateDcf..." << endl;


            thrust::device_vector<float> gamma(uvw_index);
            thrust::device_vector<float> dg(uvw_index);
            // gViss = gViss.*exp(-1i*2*pi*gw);     ggamma=asin(gw./gbll);
            launch_viss_gamma_trans(
                thrust::raw_pointer_cast(Viss.data()),
                thrust::raw_pointer_cast(w.data()),
                thrust::raw_pointer_cast(bll.data()),
                thrust::raw_pointer_cast(gamma.data()),
                thrust::raw_pointer_cast(dg.data()),
                uvw_index, zero, two, CPI, I1);
            CHECK(cudaDeviceSynchronize());
            cout << "Period " << p+1 << " viss and gamma..." << endl;


            launch_computeC(
                npix, 
                thrust::raw_pointer_cast(dcf.data()), 
                thrust::raw_pointer_cast(dg.data()), 
                thrust::raw_pointer_cast(gs.data()), 
                thrust::raw_pointer_cast(u.data()),
                thrust::raw_pointer_cast(v.data()),
                thrust::raw_pointer_cast(w.data()),
                thrust::raw_pointer_cast(l.data()),
                thrust::raw_pointer_cast(m.data()),
                thrust::raw_pointer_cast(n.data()),
                thrust::raw_pointer_cast(C.data()),
                thrust::raw_pointer_cast(Viss.data()),
                I1, two, CPI, uvw_index);
            CHECK(cudaDeviceSynchronize());
            cout << "Period " << p+1 << " compute C success" << endl;

            for (int i=0; i<=2; i++){
                cout << "C[" << i << "]: " << C[i] << endl;
            }


            // 创建一个临界区，用于保存结果
            #pragma omp critical
            {   
                // 将数据从设备内存复制到主机内存
                std::vector<Complex> host_C(C.size());
                CHECK(cudaMemcpy(host_C.data(), thrust::raw_pointer_cast(C.data()), C.size() * sizeof(Complex), cudaMemcpyDeviceToHost));
                CHECK(cudaDeviceSynchronize());
                // 打开文件
                string address_C = "3dnoblockage10M/C" + to_string(p+1) + "day10M.txt";
                cout << "Period " << p+1 << " save address_C: " << address_C << endl;
                std::ofstream file(address_C);
                if (file.is_open()) {
                    // 按照指定格式写入文件
                    for(const Complex& value : host_C)
                    {
                        file << value.real() << std::endl;
                    }
                }
                // 关闭文件
                file.close();
                std::cout << "Period " << p+1 << " save C success!" << std::endl;
            }
        }
    }
    
    gettimeofday(&finish, NULL);
    total_time = ((finish.tv_sec - start.tv_sec) * 1000000 + (finish.tv_usec - start.tv_usec)) / 1000000.0;
    cout << "total time: " << total_time << "s" << endl;
    return 0;
}


int main()
{
    vissGen(1e7);
}

