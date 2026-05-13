// healpix_pix2ang_cuda.cu
// nvcc -O3 -std=c++17 -arch=sm_86 healpix_pix2ang_cuda.cu -o healpix_pix2ang
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(call) do {                                \
  cudaError_t err = (call);                                  \
  if (err != cudaSuccess) {                                  \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                \
            __FILE__, __LINE__, cudaGetErrorString(err));    \
    std::exit(1);                                            \
  }                                                          \
} while(0)

static inline uint64_t npix_from_nside(int nside) {
    return 12ull * (uint64_t)nside * (uint64_t)nside;
}

// -------------------------
// MATLAB round: ties away from zero
// -------------------------
__host__ __device__ __forceinline__
long long matlab_round(double x) {
    return (x >= 0.0) ? (long long)floor(x + 0.5) : (long long)ceil(x - 0.5);
}

// -------------------------
// NEST helpers: deinterleave (compact 1 by 1)
// x = compact1by1(v) uses even bits; y = compact1by1(v>>1) uses odd bits
// -------------------------
__device__ __forceinline__ uint32_t compact1by1(uint32_t x) {
    x &= 0x55555555u;
    x = (x | (x >> 1)) & 0x33333333u;
    x = (x | (x >> 2)) & 0x0F0F0F0Fu;
    x = (x | (x >> 4)) & 0x00FF00FFu;
    x = (x | (x >> 8)) & 0x0000FFFFu;
    return x;
}

// -------------------------
// pix2ang_nest kernel
// Input: global ipix in [start, start+count)
// Output: theta/phi arrays of size 'count' (local indexing)
// -------------------------
__global__ void k_pix2ang_nest(
    int nside,
    uint64_t start_ipix,
    int count,
    float* __restrict__ theta_out,
    float* __restrict__ phi_out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint64_t ipix = start_ipix + (uint64_t)tid;

    // constants from MATLAB
    const int jrll[12] = {2,2,2,2,3,3,3,3,4,4,4,4};
    const int jpll[12] = {1,3,5,7,0,2,4,6,1,3,5,7};

    uint64_t npface = (uint64_t)nside * (uint64_t)nside;
    int face_num = (int)(ipix / npface);     // 0..11
    uint32_t ipf = (uint32_t)(ipix - (uint64_t)face_num * npface); // 0..npface-1

    // deinterleave to get ix,iy on face
    // For nside <= 2^13 (8192), ipf fits in 26 bits, safe in uint32.
    uint32_t ix = compact1by1(ipf);
    uint32_t iy = compact1by1(ipf >> 1);

    int jrt = (int)(ix + iy);     // 0..2*(nside-1)
    int jpt = (int)(ix - iy);     // -(nside-1)..+(nside-1)

    int nl4 = 4 * nside;
    int jr  = jrll[face_num] * nside - jrt - 1;

    // z/nr/kshift
    double fact1 = 1.0 / (3.0 * (double)nside * (double)nside);
    double fact2 = 2.0 / (3.0 * (double)nside);

    int nr = 0;
    int kshift = 0;
    double z = 0.0;

    if (jr < nside) {
        // north polar
        nr = jr;
        z = 1.0 - (double)nr * (double)nr * fact1;
        kshift = 0;
    } else if (jr <= 3 * nside) {
        // equatorial
        nr = nside;
        z = (double)(2 * nside - jr) * fact2;
        kshift = (jr - nside) & 1;
    } else {
        // south polar
        nr = nl4 - jr;
        z = -1.0 + (double)nr * (double)nr * fact1;
        kshift = 0;
    }

    // theta = acos(z)
    double zz = fmax(-1.0, fmin(1.0, z));
    double theta = acos(zz);

    // jp = fix(((jpll(face+1).*nr) + jpt + 1 + kshift)/2);
    // note: C/C++ int division truncs toward 0, matches MATLAB fix for integer numerator.
    int jp = (jpll[face_num] * nr + jpt + 1 + kshift) / 2;
    if (jp > nl4) jp -= nl4;
    if (jp < 1)   jp += nl4;

    // phi = (pi/2)*(jp -(kshift+1)/2)./nr;
    double sub = 0.5 * (double)(kshift + 1); // 0.5 or 1.0
    double phi = (M_PI * 0.5) * ((double)jp - sub) / (double)nr;

    theta_out[tid] = (float)theta;
    phi_out[tid]   = (float)phi;
}

// -------------------------
// correct_ring_phi from MEALPix (MATLAB)
// delta(iphi<0)=1; delta(iphi>4*iring)=-1;
// iring = iring - location*delta;
// iphi  = iphi + delta*(4*iring);
// -------------------------
__device__ __forceinline__
void correct_ring_phi_dev(int location, long long& iring, long long& iphi) {
    long long delta = 0;
    if (iphi < 0) delta = 1;
    if (iphi > 4 * iring) delta = -1;
    if (delta != 0) {
        iring = iring - (long long)location * delta;
        iphi  = iphi  + delta * (4 * iring);
    }
}

// -------------------------
// pix2ang_ring kernel
// -------------------------
__global__ void k_pix2ang_ring(
    int nside,
    uint64_t start_ipix,
    int count,
    float* __restrict__ theta_out,
    float* __restrict__ phi_out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint64_t ipix = start_ipix + (uint64_t)tid;

    const long long nl2  = 2ll * (long long)nside;
    const long long nl4  = 4ll * (long long)nside;
    const long long npix = (long long)(12ll * (long long)nside * (long long)nside);
    const long long nCap = nl2 * ((long long)nside - 1ll);

    double theta = 0.0, phi = 0.0;

    if ((long long)ipix < nCap) {
        // North polar cap
        long long iPixM = (long long)ipix;
        double x = ((double)(iPixM + 1ll)) / 2.0;
        long long iRing = (long long)floor(sqrt(x) + 0.5); // round
        long long iPhi  = iPixM - 2ll * iRing * (iRing - 1ll);
        correct_ring_phi_dev(+1, iRing, iPhi);

        double t = ((double)iRing / (double)nside);
        double z = 1.0 - (t * t) / 3.0;
        z = fmax(-1.0, fmin(1.0, z));
        theta = acos(z);
        phi   = (M_PI * 0.5) * ((double)iPhi + 0.5) / (double)iRing;

    } else if ((long long)ipix < (npix - nCap)) {
        // Equatorial region
        long long ipM = (long long)ipix - nCap;
        long long iRing = (ipM / nl4) + (long long)nside;
        long long iPhi  = ipM % nl4;

        double fodd = 0.5 * (double)((iRing + (long long)nside + 1ll) & 1ll); // 0 or 0.5

        double z = ((double)(nl2 - iRing)) / (1.5 * (double)nside);
        z = fmax(-1.0, fmin(1.0, z));
        theta = acos(z);
        phi   = (M_PI * 0.5) * ((double)iPhi + fodd) / (double)nside;

    } else {
        // South polar cap
        long long ipM = (long long)npix - (long long)ipix;
        double x = ((double)ipM) / 2.0;
        long long iRing = (long long)floor(sqrt(x) + 0.5); // round
        long long iPhi  = 2ll * iRing * (iRing + 1ll) - ipM;
        correct_ring_phi_dev(-1, iRing, iPhi);

        double t = ((double)iRing / (double)nside);
        double z = (t * t) / 3.0 - 1.0;
        z = fmax(-1.0, fmin(1.0, z));
        theta = acos(z);
        phi   = (M_PI * 0.5) * ((double)iPhi + 0.5) / (double)iRing;
    }

    theta_out[tid] = (float)theta;
    phi_out[tid]   = (float)phi;
}

// -------------------------
// Buffered write helpers (no header line)
// -------------------------
static void write_float_lines(FILE* fp, const float* a, int n) {
    for (int i = 0; i < n; ++i) {
        // 9-10 位有效数字对 float 足够；你后续用 >> float 读也没问题
        fprintf(fp, "%.9g\n", a[i]);
    }
}

// -------------------------
// Main: chunked generation
// -------------------------
int main(int argc, char** argv) {
    int nside = 512;
    std::string order = "ring";
    std::string out_theta = "theta_heal.txt";
    std::string out_phi   = "phi_heal.txt";
    int chunk = 1 << 20; // 1,048,576 pixels per chunk by default

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--nside" && i + 1 < argc) nside = std::atoi(argv[++i]);
        else if (a == "--order" && i + 1 < argc) order = argv[++i];
        else if (a == "--out_theta" && i + 1 < argc) out_theta = argv[++i];
        else if (a == "--out_phi" && i + 1 < argc) out_phi = argv[++i];
        else if (a == "--chunk" && i + 1 < argc) chunk = std::max(1024, std::atoi(argv[++i]));
    }

    uint64_t npix = npix_from_nside(nside);
    std::cout << "nside=" << nside << " npix=" << npix << " order=" << order
              << " chunk=" << chunk << "\n";

    FILE* fth = fopen(out_theta.c_str(), "wt");
    FILE* fph = fopen(out_phi.c_str(), "wt");
    if (!fth || !fph) {
        perror("fopen");
        return 1;
    }
    // large stdio buffers
    static std::vector<char> buf1(8 << 20), buf2(8 << 20);
    setvbuf(fth, buf1.data(), _IOFBF, buf1.size());
    setvbuf(fph, buf2.data(), _IOFBF, buf2.size());

    // device buffers for one chunk
    float *d_theta = nullptr, *d_phi = nullptr;
    CUDA_CHECK(cudaMalloc(&d_theta, (size_t)chunk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_phi,   (size_t)chunk * sizeof(float)));

    // pinned host buffers for one chunk
    float *h_theta = nullptr, *h_phi = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_theta, (size_t)chunk * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_phi,   (size_t)chunk * sizeof(float)));

    const int BS = 256;

    for (uint64_t start = 0; start < npix; start += (uint64_t)chunk) {
        int cur = (int)std::min<uint64_t>((uint64_t)chunk, npix - start);
        int GS = (cur + BS - 1) / BS;

        if (order == "nest") {
            k_pix2ang_nest<<<GS, BS>>>(nside, start, cur, d_theta, d_phi);
        } else if (order == "ring") {
            k_pix2ang_ring<<<GS, BS>>>(nside, start, cur, d_theta, d_phi);
        } else {
            std::cerr << "ERROR: --order must be nest or ring\n";
            return 2;
        }
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_theta, d_theta, (size_t)cur * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_phi,   d_phi,   (size_t)cur * sizeof(float), cudaMemcpyDeviceToHost));

        write_float_lines(fth, h_theta, cur);
        write_float_lines(fph, h_phi, cur);

        if ((start / (uint64_t)chunk) % 64ull == 0ull) {
            std::cout << "progress " << start << "/" << npix << "\n";
        }
    }

    CUDA_CHECK(cudaFree(d_theta));
    CUDA_CHECK(cudaFree(d_phi));
    CUDA_CHECK(cudaFreeHost(h_theta));
    CUDA_CHECK(cudaFreeHost(h_phi));
    fclose(fth);
    fclose(fph);

    std::cout << "done.\n";
    return 0;
}
