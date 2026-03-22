#include "common.hpp"
#include "orbit_online.cuh"

int main(int argc, char** argv){
  std::string btag=get_arg(argc,argv,"--btag","10M");
  int dcf_days=to_int(get_arg(argc,argv,"--dcf_days","450"),450);
  int segs=to_int(get_arg(argc,argv,"--segs","0"),0);
  float bl_max=to_float(get_arg(argc,argv,"--bl_max","100000"),100000.0f);
  std::string gpus_s=get_arg(argc,argv,"--gpus","0");
  int gen_gpu_index=to_int(get_arg(argc,argv,"--gen_gpu_index","0"),0);
  uint64_t orbit_seed=to_u64(get_arg(argc,argv,"--orbit_seed","42"), 42ULL);
  std::string out_bin=get_arg(argc,argv,"--out","");

  if(!gpus_s.empty()) setenv("CUDA_VISIBLE_DEVICES", gpus_s.c_str(), 1);

  if(btag!="1M" && btag!="10M" && btag!="30M"){
    std::cerr << "ERROR --btag must be 1M / 10M / 30M\n";
    return 1;
  }
  if(segs<=0){
    if(btag=="1M") segs=1;
    else if(btag=="10M") segs=10;
    else segs=30;
  }

  float frequency=(btag=="1M")? 1e6f : (btag=="10M"? 1e7f : 3e7f);
  float lambda_m=3e8f/frequency;
  if(out_bin.empty()) out_bin = default_dcf_bin_name(btag, dcf_days, orbit_seed);

  int devCount=0;
  CHECK_CUDA(cudaGetDeviceCount(&devCount));
  if(devCount <= 0){
    std::cerr << "ERROR no visible CUDA devices after applying CUDA_VISIBLE_DEVICES=" << gpus_s << "\n";
    return 1;
  }
  if(gen_gpu_index < 0 || gen_gpu_index >= devCount){
    std::cerr << "ERROR --gen_gpu_index out of range, visible count=" << devCount << "\n";
    return 1;
  }

  std::cout << "CUDA_VISIBLE_DEVICES=" << gpus_s << "\n";
  std::cout << "dcf generator on logical GPU=" << gen_gpu_index
            << " btag=" << btag
            << " dcf_days=" << dcf_days
            << " segs=" << segs
            << " bl_max=" << bl_max
            << " lambda=" << lambda_m
            << " orbit_seed=" << orbit_seed << "\n";

  HostTimer timer;
  timer.tic();
  std::vector<long long> mb;
  if(!accumulate_mb_from_online_orbit(gen_gpu_index, lambda_m, bl_max, dcf_days, segs, orbit_seed, mb)){
    return 1;
  }
  std::vector<float> dcf;
  compute_dcf_from_mb(mb, dcf);
  double elapsed = timer.toc_s();

  DcfMbBinPayload payload;
  payload.btag = btag;
  payload.lambda_m = lambda_m;
  payload.bl_max = bl_max;
  payload.dcf_days = dcf_days;
  payload.orbit_seed = orbit_seed;
  payload.mb = mb;
  payload.dcf = dcf;

  if(!save_dcf_mb_bin(out_bin, payload)){
    std::cerr << "ERROR writing dcf bin: " << out_bin << "\n";
    return 1;
  }

  std::cout << "Generated dcf+mb in " << elapsed << " s\n";
  std::cout << "mb_len=" << mb.size() << " dcf_len=" << dcf.size() << "\n";
  std::cout << "Saved to " << out_bin << "\n";
  return 0;
}
