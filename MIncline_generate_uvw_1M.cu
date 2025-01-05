#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>

#include "error.cuh"
#include <omp.h>
#include <cuda_runtime.h>


// Define constants
const int satnum = 8;
const double mu_e = 4.902793455e3; // unit in km^3/s^2
const double d2r = M_PI / 180;
const double a = 2038.14;
const double e = 0;
const double incl = 30 * d2r;
const double argp = 0 * d2r;

const double u0 = 100e3 / 23;
const std::vector<double> r1 = {0, u0, 4*u0, 10*u0, 16*u0, 18*u0, 21*u0, 23*u0};
const std::vector<double> r2 = {0, 0.1e3, 0.4*u0, 1*u0, 1.6*u0, 1.8*u0, 2.1*u0, 2.3*u0};


// LinSpace函数
struct linspace_functor {
    double a, step;
    linspace_functor(double _a, double _step) : a(_a), step(_step) {}
    __host__ __device__ double operator()(const int& x) const { 
        return a + step * x;
    }
};

void generate_linspace(thrust::device_vector<double>& d_vec, double start, double end, int num) {
    // Calculate step size
    double step = (end - start) / double(num - 1);
    // Generate sequence [0, 1, 2, ..., num-1]
    thrust::sequence(d_vec.begin(), d_vec.end());
    // Transform sequence to linear space
    thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), linspace_functor(start, step));
}

struct calculate_dM {
    double a; // 常数 a
    calculate_dM(double _a) : a(_a) {}
    __host__ __device__
    double operator()(double d) const {
        return asin(d / (2 * a * 1e3)) * 2;
    }
};

struct SortCompare {
    __host__ __device__
    bool operator()(const thrust::tuple<double, double, double>& a, const thrust::tuple<double, double, double>& b) const {
        if (thrust::get<0>(a) != thrust::get<0>(b)) {
            return thrust::get<0>(a) < thrust::get<0>(b); // 首先按 u 坐标排序
        } else if (thrust::get<1>(a) != thrust::get<1>(b)) {
            return thrust::get<1>(a) < thrust::get<1>(b); // 然后按 v 坐标排序
        } else {
            return thrust::get<2>(a) < thrust::get<2>(b); // 最后按 w 坐标排序
        }
    }
};

struct UniqueCompare {
    __host__ __device__
    bool operator()(const thrust::tuple<double, double, double>& a, const thrust::tuple<double, double, double>& b) const {
        return thrust::get<0>(a) == thrust::get<0>(b) &&
               thrust::get<1>(a) == thrust::get<1>(b) &&
               thrust::get<2>(a) == thrust::get<2>(b); // 检查 uvw 坐标是否都相等
    }
};


__global__ void calculateOrbit(double* x, double* y, double* z, double* Mt, int OrbitRes, double d2r, double e, double a, double argp, double incl, int day, int g) {
    // 计算全局索引
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    double raan = (g-1) * d2r * 0.08 + (k / OrbitRes) * d2r * 0.08 + (day - 1) * d2r * 0.08;
    double M = Mt[k];
    double E0 = M;
    for (int i = 1; i < 100; ++i) {
        double M0 = E0 - e * sin(E0);
        double error = M - M0;
        if (abs(error) < 1e-15) {
            E = E0;
            break;
        }
        E0 = E0 + error / (1 - e * cos(E0));
    }
    double temp = tan(E0 / 2) / sqrt((1 - e) / (1 + e));
    double theta = atan(temp) * 2;
    double r = a * (1 - e * e) / (1 + e * cos(theta));
    double w = theta + argp;
    int tmp = ((g-1)*OrbitRes) + k;
    x[tmp] = r * (cos(w) * cos(raan) - sin(w) * cos(incl) * sin(raan));
    y[tmp] = r * (cos(w) * sin(raan) + sin(w) * cos(incl) * cos(raan));
    z[tmp] = r * (sin(w) * sin(incl));
}


__global__ void uvwPosition(double* x, double* y, double* z, double* xt, double* yt, double* zt, int OrbitCounts, int OrbitRes, int interval, int span, int sat) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;   // OrbitCounts * OrbitRes / 3  大小和xt yt zt的维度一样
    int k = globalIdx / (OrbitRes/3) + 1;  // 第几圈

    int offset = 0;
    int tmp = sat*OrbitRes*OrbitCounts;  // 计算是用的x的哪一段
    if(k <= OrbitCounts){
        offset = (k-1)*interval;
        int endIndexWithoutMin = (k-1)*OrbitRes + offset + span;
        int startIdx = (k-1)*OrbitRes + offset; 
        if ((k-1)*(OrbitRes+interval) > OrbitRes*OrbitCounts) {
            int reset_index = (OrbitRes*OrbitCounts) / (OrbitRes+interval) + 1;
            // 当超出范围之后，从头开始采样，但是偏置还是存在
            startIdx = (k-1-reset_index)*OrbitRes + offset;
            endIndexWithoutMin = (k-1-reset_index)*OrbitRes + offset + span;
        }

        int circleIdx = startIdx + (globalIdx-(k-1)*span);  // 每一圈中对应的向量下标
        // 超出的部分就设置为0
        if(endIndexWithoutMin > OrbitRes * OrbitCounts){
            // startIdx 到 最后长度设置为正常值，超出部分设置为0
            if(circleIdx > OrbitRes*OrbitCounts){
                xt[tmp/3 + globalIdx] = 0;
                yt[tmp/3 + globalIdx] = 0;
                zt[tmp/3 + globalIdx] = 0;
            }else{
                xt[tmp/3 + globalIdx] = x[tmp + circleIdx];
                yt[tmp/3 + globalIdx] = y[tmp + circleIdx];
                zt[tmp/3 + globalIdx] = z[tmp + circleIdx];
            }
        }
        // 没有超出的部分就全部设置为正常值
        else{
            xt[tmp/3 + globalIdx] = x[tmp + circleIdx];
            yt[tmp/3 + globalIdx] = y[tmp + circleIdx];
            zt[tmp/3 + globalIdx] = z[tmp + circleIdx];
        }
    }
}

__global__ void computeUVW(double *xt, double *yt, double *zt, thrust::tuple<double, double, double> *UVWHash, int OrbitCounts, int OrbitRes, double lambda, int m, int n) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;    // OrbitCounts*OrbitRes/3  大小和xt yt zt的1/8维度一样
    int tmp = OrbitRes*OrbitCounts/3;
    double u_val = ceil((xt[tmp * n + globalIdx] - xt[tmp * m + globalIdx]) * 1e3 / lambda);
    double v_val = ceil((yt[tmp * n + globalIdx] - yt[tmp * m + globalIdx]) * 1e3 / lambda);
    double w_val = ceil((zt[tmp * n + globalIdx] - zt[tmp * m + globalIdx]) * 1e3 / lambda);
    UVWHash[globalIdx] = thrust::make_tuple(u_val, v_val, w_val);
}



void writeToFile(const thrust::device_vector<double>& device_vector, const std::string& filename, int satnum, int columns) {
    // 将数据从设备内存复制到主机内存
    std::vector<double> host_vector(device_vector.size());
    thrust::copy(device_vector.begin(), device_vector.end(), host_vector.begin());
    // 打开文件
    std::ofstream file(filename);
    if (file.is_open()) {
        // 按照指定格式写入文件
        for (int i = 0; i < columns; ++i) {
            for (int s = 0; s < satnum; ++s) {
                file << host_vector[s * columns + i] << " ";
            }
            file << std::endl; // 换行，开始新的一组数据
        }
    }
    // 关闭文件
    file.close();
}


void Step(int day, double frequency, int gpu_id) {
    int lambda = 3e8 / frequency;
    int OrbitRes = ceil(2 * M_PI * 100e3 / lambda);
    int OrbitCounts = round(24/2.3);

    std::cout << "lambda " << lambda << std::endl;
    std::cout << "OrbitRes " << OrbitRes << std::endl;
    std::cout << "OrbitRes * OrbitCounts * satnum " << OrbitRes * OrbitCounts * satnum << std::endl;

    cudaSetDevice(gpu_id);
    // 在GPU上创建xyz和Mt
    thrust::device_vector<double> x(OrbitRes * OrbitCounts * satnum, 0);
    thrust::device_vector<double> y(OrbitRes * OrbitCounts * satnum, 0);
    thrust::device_vector<double> z(OrbitRes * OrbitCounts * satnum, 0);
    thrust::device_vector<double> Mt(OrbitRes);

    generate_linspace(Mt, 1/OrbitRes, 2*M_PI+M_PI/200, OrbitRes);
    std::cout << "GPU " << gpu_id << " Mt_all compute successe! " << std::endl;

    // 记录循环事件
    cudaEvent_t satstart, satstop;
    CHECK(cudaEventCreate(&satstart));
    CHECK(cudaEventCreate(&satstop));
    CHECK(cudaEventRecord(satstart));

    // 计算第一颗卫星
    for (int g = 0; g < OrbitCounts; g++) 
    {
        int minGridSize; // 最小网格大小
        int posBlockSize;
        int posGridSize;    // 实际网格大小
        CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &posBlockSize, calculateOrbit, 0, 0));
        posGridSize = floor(OrbitRes + posBlockSize - 1) / posBlockSize;
        std::cout << "GPU" << gpu_id << " Calculate position, blockSize: " << posBlockSize << std::endl;
        std::cout << "GPU" << gpu_id << " Calculate position, girdSize: " << posGridSize << std::endl;
        calculateOrbit<<<posGridSize, posBlockSize>>>(
            thrust::raw_pointer_cast(x.data()),
            thrust::raw_pointer_cast(y.data()),
            thrust::raw_pointer_cast(z.data()), 
            thrust::raw_pointer_cast(Mt.data()), 
            OrbitRes, d2r, e, a, argp, incl, day, g
        );

        std::cout << "Sat 1 " << "Count " << g << "calculateOrbit compute success!" << std::endl;
        
        CHECK(cudaDeviceSynchronize());  // 线程同步，等前面完成了再进行下一步    
    }

    std::cout << "Sat 1 all calculateOrbit !!!" << std::endl;

    // 计算剩余几颗
    thrust::device_vector<double> vx(x[], 0);
    thrust::device_vector<double> vy(OrbitRes * OrbitCounts * satnum, 0);
    thrust::device_vector<double> vz(OrbitRes * OrbitCounts * satnum, 0);

        // 调用uvw_position核函数
        // 记录uvw开始事件
        cudaEvent_t uvwstart, uvwstop;
        CHECK(cudaEventCreate(&uvwstart));
        CHECK(cudaEventCreate(&uvwstop));
        CHECK(cudaEventRecord(uvwstart));
        int uvwBlockSize;
        int uvwGridSize;    // 实际网格大小
        CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &uvwBlockSize, uvwPosition, 0, 0));
        uvwGridSize = floor(OrbitCounts * OrbitRes / 3 + uvwBlockSize - 1) / uvwBlockSize;
        std::cout << "GPU" << gpu_id << " UVW position, blockSize: " << uvwBlockSize << std::endl;
        std::cout << "GPU" << gpu_id << " UVW position, girdSize: " << uvwGridSize << std::endl;
        uvwPosition<<<uvwGridSize, uvwBlockSize>>>(
            thrust::raw_pointer_cast(x.data()),
            thrust::raw_pointer_cast(y.data()),
            thrust::raw_pointer_cast(z.data()), 
            thrust::raw_pointer_cast(xt.data()),
            thrust::raw_pointer_cast(yt.data()),
            thrust::raw_pointer_cast(zt.data()), 
            OrbitCounts, OrbitRes, interval, span, ss
        );
        std::cout << "GPU" << gpu_id << " uvwPosition compute success!" << std::endl;
        // 记录uvw结束事件
        CHECK(cudaEventRecord(uvwstop));
        CHECK(cudaEventSynchronize(uvwstop));
        // 计算经过的时间
        float uvwMS = 0;
        CHECK(cudaEventElapsedTime(&uvwMS, uvwstart, uvwstop));
        printf("Frequency-%f Index-%d Sat-%d On GPU-%d Compute xt yt zt Cost Time is: %f s\n", frequency, index, ss+1, gpu_id, uvwMS/1000);
        std::cout << "Will Save! Sat " << ss+1 << " uvw position compute successe! on GPU " << gpu_id << std::endl;
        CHECK(cudaDeviceSynchronize());  // 线程同步，等前面完成了再进行下一步    
    }
    
    // writeToFile(x, "x_output.txt", satnum, OrbitRes*OrbitCounts);
    // writeToFile(y, "y_output.txt", satnum, OrbitRes*OrbitCounts);
    // writeToFile(z, "z_output.txt", satnum, OrbitRes*OrbitCounts);

    // writeToFile(xt, "xt.txt", satnum, OrbitRes*OrbitCounts/3);
    // writeToFile(yt, "yt.txt", satnum, OrbitRes*OrbitCounts/3);
    // writeToFile(zt, "zt.txt", satnum, OrbitRes*OrbitCounts/3);

    // 记录循环结束事件
    CHECK(cudaEventRecord(satstop));
    CHECK(cudaEventSynchronize(satstop));
    // 计算经过的时间
    float satMS = 0;
    CHECK(cudaEventElapsedTime(&satMS, satstart, satstop));
    printf("Frequency-%f Index-%d On GPU-%d All satellite Calculate Cost Time is: %f s\n", frequency, index, gpu_id, satMS/1000);
    // 销毁事件
    cudaEventDestroy(satstart);
    cudaEventDestroy(satstop);

    // 用完xyz后，释放它们的内存
    x.clear();
    x.shrink_to_fit();
    y.clear();
    y.shrink_to_fit();
    z.clear();
    z.shrink_to_fit();

    // 记录duplicate开始事件
    cudaEvent_t dupstart, dupstop;
    CHECK(cudaEventCreate(&dupstart));
    CHECK(cudaEventCreate(&dupstop));
    CHECK(cudaEventRecord(dupstart));

    thrust::device_vector<thrust::tuple<double, double, double>> UVWHash(OrbitRes*OrbitCounts/3, thrust::make_tuple(0.0, 0.0, 0.0));  // 用来存储uvw坐标的哈希值
    thrust::device_vector<thrust::tuple<double, double, double>> globalUVW;   // 用来存储全局UVW
    
    int blockSize;
    int minGridSize; // 最小网格大小
    int gridSize;    // 实际网格大
    int loopIndex = 0;
    for (int m=0; m<satnum; m++) {
        for (int n=0; n<satnum; n++)
        {
            if (m != n) {
                loopIndex++;
                std::cout << "loop: " << loopIndex << std::endl;

                CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeUVW, 0, 0));
                gridSize = floor(OrbitCounts*OrbitRes/3 + blockSize - 1) / blockSize;
                std::cout << "GPU" << gpu_id << "Calculate hash, blockSize: " << blockSize << std::endl;
                std::cout << "GPU" << gpu_id << "Calculate hash, girdSize: " << gridSize << std::endl;
                thrust::tuple<double, double, double>* d_UVWHash = thrust::raw_pointer_cast(UVWHash.data());
                computeUVW<<<gridSize, blockSize>>>( 
                    thrust::raw_pointer_cast(xt.data()),
                    thrust::raw_pointer_cast(yt.data()),
                    thrust::raw_pointer_cast(zt.data()), 
                    d_UVWHash,
                    OrbitCounts, OrbitRes, lambda, m, n
                );
                CHECK(cudaDeviceSynchronize());  // 线程同步，等前面完成了再进行下一步
                std::cout << "size of UVWHash: " << UVWHash.size() << std::endl;
                std::cout << "m:" << m << " n:" << n << " computed UVW" << std::endl;

                // 根据坐标值对数据进行排序
                thrust::sort(UVWHash.begin(), UVWHash.end(), SortCompare());
                std::cout << "m:" << m << " n:" << n << " sorted" << std::endl;

                // 去除连续重复元素
                auto new_end = thrust::unique(thrust::device, UVWHash.begin(), UVWHash.end(), UniqueCompare());
                // UVWHash.erase(new_end, UVWHash.end());
                int lengthOfUniqueUVW = thrust::distance(UVWHash.begin(), new_end);
                std::cout << "size of Unique UVWHash: " << lengthOfUniqueUVW << std::endl;
                std::cout << "m:" << m << " n:" << n << " uniqued" << std::endl;

                // 将去重后的 UVWHash 添加到 globalUVW 的后面
                globalUVW.insert(globalUVW.end(), UVWHash.begin(), new_end);
                std::cout << "m:" << m << " n:" << n << " globalUVW" << std::endl;

                CHECK(cudaDeviceSynchronize());  // 线程同步，等前面完成了再进行下一步    
            }
        }
    }
    CHECK(cudaDeviceSynchronize());  // 线程同步，等前面完成了再进行下一步    

    // 对全局的uvw进行排序去重
    thrust::sort(thrust::device, globalUVW.begin(), globalUVW.end(), SortCompare());
    auto new_end = thrust::unique(thrust::device, globalUVW.begin(), globalUVW.end(), UniqueCompare());
    globalUVW.erase(new_end, globalUVW.end());
    std::cout << "size of globalUVW: " << globalUVW.size() << std::endl;
    CHECK(cudaDeviceSynchronize());  // 线程同步，等前面完成了再进行下一步 

     // 记录duplicate结束事件
    CHECK(cudaEventRecord(dupstop));
    CHECK(cudaEventSynchronize(dupstop));
    // 计算经过的时间
    float dupMS = 0;
    CHECK(cudaEventElapsedTime(&dupMS, dupstart, dupstop));
    printf("Frequency %f Index %d On GPU %d Duplicate Cost Time is: %f s\n", frequency, index, gpu_id, dupMS/1000);
    // 销毁事件
    cudaEventDestroy(dupstart);
    cudaEventDestroy(dupstop);

    // 记录save开始事件
    cudaEvent_t savestart, savestop;
    CHECK(cudaEventCreate(&savestart));
    CHECK(cudaEventCreate(&savestop));
    CHECK(cudaEventRecord(savestart));
    // 复制到本地，然后保存
    thrust::host_vector<thrust::tuple<double, double, double>> h_globalUVW = globalUVW;
    // 写入到文件中
    std::ostringstream fname;
    fname << "frequency_1M/uvw" << index << "frequency1M.txt";
    std::ofstream file(fname.str());
    for(size_t i = 0; i < h_globalUVW.size(); i++) {
        auto& t = h_globalUVW[i];
        file << thrust::get<0>(t) << " " << thrust::get<1>(t) << " " << thrust::get<2>(t) << std::endl;
    }
    file.close();

    // 记录save结束事件
    CHECK(cudaEventRecord(savestop));
    CHECK(cudaEventSynchronize(savestop));
    // 计算经过的时间
    float saveMS = 0;
    CHECK(cudaEventElapsedTime(&saveMS, savestart, savestop));
    printf("Frequency %f Index %d On GPU %d Save txt Cost Time is: %f s\n", frequency, index, gpu_id, saveMS/1000);
    // 销毁事件
    cudaEventDestroy(savestart);
    cudaEventDestroy(savestop);
    std::cout << "GPU Compute position successe!" << std::endl;
}


int main()
{
    int days = 450;
    double frequency = 1e6;    

    // auto pos_start = std::chrono::high_resolution_clock::now();

    for (int index=1; index<=periods; index++){
        Step(index, frequency, 0); 
    }
    
    // auto pos_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> pos_ms = pos_end - pos_start;
    // std::cout << "Position Generated in " << pos_ms.count()/1000 << "s" << std::endl;

    // int nDevices;
    // CHECK(cudaGetDeviceCount(&nDevices));
    // int periodsPerGPU = periods / nDevices; // 每个 GPU 处理的周期数
    // std::cout << "Num of GPU: " << nDevices << std::endl;

    // // 设置并行区中的线程数
    // omp_set_num_threads(nDevices);
    // #pragma omp parallel 
    // {   
    //     int tid = omp_get_thread_num();
    //     cudaSetDevice(tid);
    //     // 计算每个 GPU 处理的周期范围
    //     int startPeriod = tid * periodsPerGPU+1;
    //     int endPeriod = (tid + 1) * periodsPerGPU+1;
    //     if (tid == nDevices - 1) {
    //         // 确保最后一个 GPU 处理所有剩余的周期
    //         endPeriod = periods+1;
    //     }
    //     // 循环遍历每个线程负责的周期
    //     for (int index = startPeriod; index < endPeriod; index++) {
    //         MIncline(index, frequency, stride, tid);
    //     }
    // }
    
    return 0;
}
