#include "common.hpp"
#include "orbit_online.cuh"
#include "viss_recon_kernel.cuh"

#include <array>
#include <limits>

static constexpr int RECON_TILE_PIX_HOST = 256;
static constexpr int RECON_TILE_BL_HOST  = 128;


struct Stage1SegMetrics {
  int dayid = 0;
  int seg_idx = 0;
  int t0 = 0;
  int tlen = 0;
  int segN = 0;
  int segN_half = 0;
  double T_gen = 0.0;
  double T_bcast = 0.0;
  double T_viss_kernel = 0.0;
  double T_reduce = 0.0;
  double T_phase = 0.0;
  double T_output_stage = 0.0;
  double T_wall = 0.0;
  double T_stage1_total = 0.0;
  double overlap_efficiency = 0.0;
  double multi_gpu_scaling_efficiency = 1.0;
};

int main(int argc,char** argv){
  std::string btag=get_arg(argc,argv,"--btag","10M");
  int nside=to_int(get_arg(argc,argv,"--nside","4096"),4096);
  int day_start=to_int(get_arg(argc,argv,"--day_start","1"),1);
  int day_count=to_int(get_arg(argc,argv,"--day_count","1"),1);
  int segs=to_int(get_arg(argc,argv,"--segs","0"),0);
  int dcf_days=to_int(get_arg(argc,argv,"--dcf_days","1"),1);
  std::string dcf_bin=get_arg(argc,argv,"--dcf_bin","");

  int viss_tile_pix=to_int(get_arg(argc,argv,"--viss_tile_pix","256"),256);
  if(viss_tile_pix!=256 && viss_tile_pix!=512){ std::cerr<<"ERROR --viss_tile_pix supports 256 or 512\n"; return 1; }
  int blockage=to_int(get_arg(argc,argv,"--blockage","1"),1);
  int write_viss=to_int(get_arg(argc,argv,"--write_viss","0"),0);
  int write_baseline_txt=to_int(get_arg(argc,argv,"--write_baseline_txt","0"),0);
  std::string B_mode=get_arg(argc,argv,"--B_mode","auto");
  std::string C_mode=get_arg(argc,argv,"--C_mode","bin");
  int gen_gpu_index=to_int(get_arg(argc,argv,"--gen_gpu_index","0"),0);
  uint64_t orbit_seed=to_u64(get_arg(argc,argv,"--orbit_seed","42"), 12345ULL);

  std::string sky_dir=norm_dir(get_arg(argc,argv,"--sky_dir",""));
  std::string out_dir=get_arg(argc,argv,"--out_dir","./out_direct3d/");
  std::string gpus_s=get_arg(argc,argv,"--gpus","0");

  if(!gpus_s.empty()) setenv("CUDA_VISIBLE_DEVICES", gpus_s.c_str(), 1);

  if(btag!="1M" && btag!="10M" && btag!="30M"){
    std::cerr<<"ERROR --btag\n";
    return 1;
  }
  if(nside<=0){
    if(btag=="1M") nside=512;
    else if(btag=="10M") nside=4096;
    else nside=16384;
  }
  if(segs<=0){
    if(btag=="1M") segs=1;
    else if(btag=="10M") segs=10;
    else segs=30;
  }
  if(sky_dir.empty()) sky_dir=(btag=="1M")? "./earth_1Mhz" : (btag=="10M"? "./earth_10Mhz":"./earth_30Mhz");
  if(!out_dir.empty() && out_dir.back()!='/') out_dir.push_back('/');
  ensure_dir(out_dir);

  auto visible_gpus = parse_gpus(gpus_s);
  int requested_G = (int)visible_gpus.size();

  int devCount=0;
  CHECK_CUDA(cudaGetDeviceCount(&devCount));
  if(devCount <= 0){
    std::cerr<<"ERROR no visible CUDA devices after applying CUDA_VISIBLE_DEVICES="<<gpus_s<<"\n";
    return 1;
  }

  int G = devCount;
  if(requested_G > 0 && requested_G != devCount){
    std::cerr<<"WARNING requested "<<requested_G
             <<" GPUs via --gpus="<<gpus_s
             <<", but CUDA sees "<<devCount
             <<". Use visible count = "<<devCount<<".\n";
  }

  std::vector<int> gpus(G);
  for(int i=0;i<G;i++) gpus[i] = i;

  std::cout<<"CUDA_VISIBLE_DEVICES="<<gpus_s<<"\n";
  std::cout<<"Visible logical GPUs: ";
  for(int i=0;i<G;i++){ std::cout<<gpus[i]; if(i+1<G) std::cout<<","; }
  std::cout<<"\n";

  long long npix=12LL*(long long)nside*(long long)nside;
  float R=1737.1e3f, h=300e3f;
  float theta=asinf(R/(R+h));
  float phi=(float)M_PI-theta;
  float cosphi=cosf(phi);

  float frequency=(btag=="1M")? 1e6f : (btag=="10M"? 1e7f : 3e7f);
  float lamda=3e8f/frequency;
  float bl_max=100e3f;

  std::cout<<"btag="<<btag<<" nside="<<nside<<" npix="<<npix
           <<" day_start="<<day_start<<" day_count="<<day_count<<" segs="<<segs
           <<" dcf_days="<<dcf_days<<"\n";
  std::cout<<"blockage="<<blockage
           <<"  [Stage-1 Viss=tilecone+blockage(always ON, vissfast), Stage-2 Recon=direct 3D half-sym tilecone]\n";
  std::cout<<" sky_dir="<<sky_dir<<" out_dir="<<out_dir
           <<" gpus="<<gpus_s<<" C_mode="<<C_mode
           <<" viss_tile_pix="<<viss_tile_pix<<"\n";
  std::cout<<"theta="<<theta<<" phi="<<phi<<" cosphi="<<cosphi<<" lamda="<<lamda<<"\n";

  HostTimer t_io;
  t_io.tic();
  std::vector<float> hB(npix);
  std::string used_B_path;
  if(!load_B_auto(sky_dir, btag, hB.data(), npix, B_mode, &used_B_path)){
    std::cerr<<"ERROR reading B in mode="<<B_mode<<" from "<<sky_dir<<"\n";
    return 1;
  }
  std::cout<<"Loaded B from "<<used_B_path<<" in "<<t_io.toc_s()<<" s\n";

  float lambda_m = lamda;
  int OrbitRes = (int)std::ceil((double)(2.0 * M_PI) * (double)(100e3 / (double)lambda_m));
  int ProcessionCount = round_away_from_zero_host(24.0f / ORBIT_HOURS);
  int segLen = OrbitRes / 3;
  int T = ProcessionCount * segLen;
  int N = T * SIGNED_BASELINES_PER_T;

  if(segs > T) segs = T;

  std::vector<int> seg_t0(segs), seg_tlen(segs);
  int baseT = T / segs;
  int remT = T % segs;
  int curT = 0;
  for(int k=0;k<segs;k++){
    int len = baseT + (k<remT?1:0);
    seg_t0[k]=curT;
    seg_tlen[k]=len;
    curT += len;
  }

  int max_tlen = 0;
  for(int k=0;k<segs;k++) max_tlen = std::max(max_tlen, seg_tlen[k]);
  int max_segN = max_tlen * SIGNED_BASELINES_PER_T;
  int max_segN_half = max_tlen * UNIQUE_BASELINES_PER_T;

  std::cout << "Online orbit plan: OrbitRes=" << OrbitRes
            << " ProcessionCount=" << ProcessionCount
            << " segLen=" << segLen
            << " T=" << T
            << " N=" << N << "\n";
  std::cout << "Segment plan: max_tlen=" << max_tlen
            << " max_segN=" << max_segN
            << " max_segN_half=" << max_segN_half << "\n";

  HostTimer t_dcf;
  t_dcf.tic();
  std::vector<long long> mb;
  std::vector<float> hdcf;
  if(gen_gpu_index < 0 || gen_gpu_index >= G){
    std::cerr << "ERROR --gen_gpu_index out of range\n";
    return 1;
  }
  if(!dcf_bin.empty()){
    DcfMbBinPayload payload;
    if(!load_dcf_mb_bin(dcf_bin, payload)){
      std::cerr << "ERROR loading dcf bin: " << dcf_bin << "\n";
      return 1;
    }
    mb = payload.mb;
    hdcf = payload.dcf;
    if(hdcf.empty()){
      std::cerr << "ERROR dcf bin has empty dcf array: " << dcf_bin << "\n";
      return 1;
    }
    std::cout << "Loaded dcf bin from " << dcf_bin
              << " in " << t_dcf.toc_s() << " s, dcf_len=" << hdcf.size()
              << ", mb_len=" << mb.size()
              << ", src_btag=" << payload.btag
              << ", src_days=" << payload.dcf_days
              << ", src_seed=" << payload.orbit_seed << "\n";
    if(payload.btag != btag){
      std::cout << "WARNING dcf bin btag=" << payload.btag << " but current btag=" << btag << "\n";
    }
  } else {
    if(!accumulate_mb_from_online_orbit(gpus[gen_gpu_index], lambda_m, bl_max, dcf_days, segs, orbit_seed, mb)){
      return 1;
    }
    compute_dcf_from_mb(mb, hdcf);
    std::cout << "Computed dcf in " << t_dcf.toc_s() << " s, dcf_len=" << hdcf.size() << "\n";
  }

  std::vector<GpuCtx> ctx(G);
  std::vector<GpuDirectExtra> ext(G);
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
    ctx[gi].N_half=max_segN_half;
    CHECK_CUDA(cudaStreamCreateWithFlags(&ctx[gi].compute_stream,cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&ctx[gi].xfer_stream,cudaStreamNonBlocking));

    CHECK_CUDA(cudaMalloc(&ctx[gi].d_B,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_l,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_m,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_n,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_nm1,(size_t)n_chunk*sizeof(float)));
    ctx[gi].ntile_viss = (int)((n_chunk + viss_tile_pix - 1) / viss_tile_pix);
    ctx[gi].ntile_recon = (int)((n_chunk + RECON_TILE_PIX_HOST - 1) / RECON_TILE_PIX_HOST);
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cx_viss,(size_t)ctx[gi].ntile_viss*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cy_viss,(size_t)ctx[gi].ntile_viss*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cz_viss,(size_t)ctx[gi].ntile_viss*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cosA_viss,(size_t)ctx[gi].ntile_viss*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_sinA_viss,(size_t)ctx[gi].ntile_viss*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cx_recon,(size_t)ctx[gi].ntile_recon*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cy_recon,(size_t)ctx[gi].ntile_recon*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cz_recon,(size_t)ctx[gi].ntile_recon*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cosA_recon,(size_t)ctx[gi].ntile_recon*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_sinA_recon,(size_t)ctx[gi].ntile_recon*sizeof(float)));

    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_B,hB.data()+pix0,(size_t)n_chunk*sizeof(float),cudaMemcpyHostToDevice,ctx[gi].xfer_stream));

    int BLOCK=256;
    int grid=(int)((n_chunk+BLOCK-1)/BLOCK);
    pix2lmn_nest_kernel<<<grid,BLOCK,0,ctx[gi].xfer_stream>>>(
      nside, (unsigned int)pix0, (int)n_chunk, ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n);
    CHECK_CUDA(cudaPeekAtLastError());
    int grid_nm1=(int)((n_chunk+255)/256);
    build_nm1_kernel<<<grid_nm1,256,0,ctx[gi].xfer_stream>>>(ctx[gi].d_n, ctx[gi].d_nm1, ctx[gi].n_chunk);    launch_build_tile_cone_meta_runtime(viss_tile_pix, ctx[gi].xfer_stream,
      ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n, ctx[gi].n_chunk,
      ctx[gi].d_tile_cx_viss, ctx[gi].d_tile_cy_viss, ctx[gi].d_tile_cz_viss,
      ctx[gi].d_tile_cosA_viss, ctx[gi].d_tile_sinA_viss, ctx[gi].ntile_viss);
    launch_build_tile_cone_meta_runtime(RECON_TILE_PIX_HOST, ctx[gi].xfer_stream,
      ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n, ctx[gi].n_chunk,
      ctx[gi].d_tile_cx_recon, ctx[gi].d_tile_cy_recon, ctx[gi].d_tile_cz_recon,
      ctx[gi].d_tile_cosA_recon, ctx[gi].d_tile_sinA_recon, ctx[gi].ntile_recon);
    CHECK_CUDA(cudaPeekAtLastError());

    CHECK_CUDA(cudaMalloc(&ext[gi].d_dcf,(size_t)hdcf.size()*sizeof(float)));
    CHECK_CUDA(cudaMemcpyAsync(ext[gi].d_dcf, hdcf.data(), (size_t)hdcf.size()*sizeof(float), cudaMemcpyHostToDevice, ctx[gi].xfer_stream));

    for(int slot=0; slot<2; ++slot){
      GpuSegSlot &sg = ctx[gi].slots[slot];
      CHECK_CUDA(cudaMalloc(&sg.d_pos, (size_t)max_tlen * (3*SATNUM) * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_satx, (size_t)max_tlen * SATNUM * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_saty, (size_t)max_tlen * SATNUM * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_satz, (size_t)max_tlen * SATNUM * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_u, (size_t)max_segN * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_v, (size_t)max_segN * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_w, (size_t)max_segN * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_x1, (size_t)max_segN * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_y1, (size_t)max_segN * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_z1, (size_t)max_segN * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_x2, (size_t)max_segN * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_y2, (size_t)max_segN * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_z2, (size_t)max_segN * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_invn1, (size_t)max_segN * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_invn2, (size_t)max_segN * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_Vpart, (size_t)max_segN_half * sizeof(float2)));
      CHECK_CUDA(cudaMalloc(&sg.d_Viss, (size_t)max_segN_half * sizeof(float2)));
      CHECK_CUDA(cudaMalloc(&ext[gi].d_pairw_half[slot], (size_t)max_segN_half * sizeof(float)));
      CHECK_CUDA(cudaMallocHost(&sg.h_Vpart, (size_t)max_segN_half * sizeof(float2)));
      CHECK_CUDA(cudaEventCreate(&sg.bcast_start));
      CHECK_CUDA(cudaEventCreate(&sg.ready));
      CHECK_CUDA(cudaEventCreate(&sg.viss_start));
      CHECK_CUDA(cudaEventCreate(&sg.viss_stop));
    }

    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Cacc,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Cacc, 0, (size_t)n_chunk*sizeof(float), ctx[gi].compute_stream));
    CHECK_CUDA(cudaMallocHost(&ctx[gi].h_chunk,(size_t)std::min(1LL<<20, n_chunk>0?n_chunk:1LL)*sizeof(float)));

    CHECK_CUDA(cudaStreamSynchronize(ctx[gi].xfer_stream));
    CHECK_CUDA(cudaStreamSynchronize(ctx[gi].compute_stream));
  }
  std::cout << "GPU init done in " << t_init.toc_s() << " s\n";

  BaselineBroadcastInfo binfo;
  query_peer_access(binfo, gpus[gen_gpu_index], ctx);
  std::cout << "Generator logical GPU=" << gpus[gen_gpu_index]
            << ", peer-staging-needed=" << (binfo.need_host_stage ? "YES" : "NO") << "\n";

  bool need_all_host = write_baseline_txt || write_viss || binfo.need_host_stage;

  for(int dayid=day_start; dayid<day_start+day_count; ++dayid){
    std::cout << "===== DAY " << dayid << " =====\n";

    for(int gi=0; gi<G; ++gi){
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Cacc, 0, (size_t)ctx[gi].n_chunk*sizeof(float), ctx[gi].compute_stream));
      CHECK_CUDA(cudaStreamSynchronize(ctx[gi].compute_stream));
    }

    std::ofstream ofsViss;
    std::ofstream ofsUVW, ofsXYZa, ofsXYZb;
    if(write_viss){
      std::string fv = out_dir + "Viss" + std::to_string(dayid) + "day" + btag + ".txt";
      ofsViss.open(fv);
      if(!ofsViss.is_open()){
        std::cerr << "ERROR open " << fv << "\n";
        return 1;
      }
    }
    if(write_baseline_txt){
      std::string fuvw = out_dir + "uvw"  + std::to_string(dayid) + "day" + btag + ".txt";
      std::string fxy1 = out_dir + "xyza" + std::to_string(dayid) + "day" + btag + ".txt";
      std::string fxy2 = out_dir + "xyzb" + std::to_string(dayid) + "day" + btag + ".txt";
      ofsUVW.open(fuvw); ofsXYZa.open(fxy1); ofsXYZb.open(fxy2);
      if(!ofsUVW.is_open() || !ofsXYZa.is_open() || !ofsXYZb.is_open()){
        std::cerr << "ERROR open baseline txt outputs\n";
        return 1;
      }
    }

    OrbitGenCtx gen;
    orbit_gen_init(gen, gpus[gen_gpu_index], lambda_m, dayid, max_tlen, orbit_seed);

    HostTimer t_stage1_pipeline;
    t_stage1_pipeline.tic();

    orbit_gen_make_segment_async(gen, 0, seg_t0[0], seg_tlen[0], need_all_host);
    broadcast_segment_to_gpus_async(gen, 0, binfo, ctx, need_all_host);
    if(segs > 1) orbit_gen_make_segment_async(gen, 1, seg_t0[1], seg_tlen[1], need_all_host);

    double stage1_sum_gen = 0.0;
    double stage1_sum_bcast = 0.0;
    double stage1_sum_viss_kernel = 0.0;
    double stage1_sum_reduce = 0.0;
    double stage1_sum_phase = 0.0;
    double stage1_sum_output = 0.0;
    double stage1_sum_wall_active = 0.0;
    double stage1_sum_serial_equiv = 0.0;
    double stage1_sum_gpu_kernel = 0.0;
    std::vector<Stage1SegMetrics> stage1_metrics;
    stage1_metrics.reserve((size_t)segs);

    HostTimer t_rec;
    t_rec.tic();
    for(int s=0; s<segs; ++s){
      int cur = s & 1;
      int nxt = cur ^ 1;

      HostTimer t_stage1_wall;
      t_stage1_wall.tic();

      orbit_gen_wait_slot(gen, cur);
      const OrbitSegSlot &gcur = gen.slots[cur];

      int seg_t0_cur = gcur.t0;
      int tlen = gcur.tlen;
      int segN = gcur.segN;
      int segN_half = tlen * UNIQUE_BASELINES_PER_T;
      if(segN <= 0 || segN_half <= 0) continue;

      if(!validate_uvw_halfsym_layout_ptr(gcur.h_u, gcur.h_v, gcur.h_w, segN)){
        std::cerr << "ERROR: generated segment uvw does not satisfy half-sym at seg " << s << " day " << dayid << "\n";
        return 1;
      }

      if(write_baseline_txt){
        write_txt_3cols_stream(ofsUVW,  gcur.h_u,  gcur.h_v,  gcur.h_w,  (size_t)segN);
        write_txt_3cols_stream(ofsXYZa, gcur.h_x1, gcur.h_y1, gcur.h_z1, (size_t)segN);
        write_txt_3cols_stream(ofsXYZb, gcur.h_x2, gcur.h_y2, gcur.h_z2, (size_t)segN);
      }

      double T_gen = 0.0;
      CHECK_CUDA(cudaSetDevice(gen.dev));
      T_gen = cuda_event_elapsed_s(gcur.gen_start, gcur.ready);

      const int BLOCKV=256;
      int gridV=(segN_half + BLOCKV - 1)/BLOCKV;
      size_t shmem=(size_t)viss_tile_pix*5*sizeof(float);

      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        cudaStream_t stream=ctx[gi].compute_stream;
        GpuSegSlot &slot = ctx[gi].slots[cur];
        CHECK_CUDA(cudaStreamWaitEvent(stream, slot.ready, 0));
        CHECK_CUDA(cudaEventRecord(slot.viss_start, stream));
        launch_vissfast_halfsym_tilecone_runtime<true>(viss_tile_pix, gridV, BLOCKV, shmem, stream,
          ctx[gi].d_B,ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,ctx[gi].d_nm1,ctx[gi].n_chunk,
          ctx[gi].d_tile_cx_viss, ctx[gi].d_tile_cy_viss, ctx[gi].d_tile_cz_viss,
          ctx[gi].d_tile_cosA_viss, ctx[gi].d_tile_sinA_viss, ctx[gi].ntile_viss,
          slot.d_u, slot.d_v, slot.d_w,
          slot.d_x1, slot.d_y1, slot.d_z1, slot.d_invn1,
          slot.d_x2, slot.d_y2, slot.d_z2, slot.d_invn2,
          segN_half, cosphi,
          slot.d_Vpart);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaEventRecord(slot.viss_stop, stream));
        CHECK_CUDA(cudaMemcpyAsync(slot.h_Vpart, slot.d_Vpart,
                                   (size_t)segN_half*sizeof(float2),
                                   cudaMemcpyDeviceToHost, stream));
      }

      HostTimer t_reduce;
      t_reduce.tic();
      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        CHECK_CUDA(cudaStreamSynchronize(ctx[gi].compute_stream));
      }

      double T_bcast = 0.0;
      double T_viss_kernel = 0.0;
      double sum_gpu_kernel = 0.0;
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        const GpuSegSlot &slot = ctx[gi].slots[cur];
        double tb = cuda_event_elapsed_s(slot.bcast_start, slot.ready);
        double tk = cuda_event_elapsed_s(slot.viss_start, slot.viss_stop);
        T_bcast = std::max(T_bcast, tb);
        T_viss_kernel = std::max(T_viss_kernel, tk);
        sum_gpu_kernel += tk;
      }

      std::vector<float2> hViss_half(segN_half);
      for(int i=0;i<segN_half;i++){
        float re=0.0f, im=0.0f;
        for(int gi=0; gi<G; ++gi){
          re += ctx[gi].slots[cur].h_Vpart[i].x;
          im += ctx[gi].slots[cur].h_Vpart[i].y;
        }
        hViss_half[i]=make_float2(re,im);
      }
      double T_reduce = t_reduce.toc_s();

      HostTimer t_phase;
      t_phase.tic();
      phase_correct_viss_halfsym_host(hViss_half.data(), gcur.h_w, segN_half);
      double T_phase = t_phase.toc_s();

      HostTimer t_output_stage;
      t_output_stage.tic();
      std::vector<float2> hViss_seg;
      if(write_viss){
        hViss_seg.assign(segN, make_float2(0,0));
        expand_viss_halfsym_to_full(hViss_half, hViss_seg);
        for(int i=0;i<segN;i++) ofsViss << hViss_seg[i].x << " " << hViss_seg[i].y << "\n";
      }

      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        cudaStream_t stream=ctx[gi].compute_stream;
        GpuSegSlot &slot = ctx[gi].slots[cur];
        CHECK_CUDA(cudaMemcpyAsync(slot.d_Viss, hViss_half.data(),
                                   (size_t)segN_half*sizeof(float2),
                                   cudaMemcpyHostToDevice, stream));
      }
      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        CHECK_CUDA(cudaStreamSynchronize(ctx[gi].compute_stream));
      }
      double T_output_stage = t_output_stage.toc_s();
      double T_wall = t_stage1_wall.toc_s();

      double serial_equiv = T_gen + T_bcast + T_viss_kernel + T_reduce + T_phase + T_output_stage;
      double overlap_eff = (serial_equiv > 0.0) ? clamp01d((serial_equiv - T_wall) / serial_equiv) : 0.0;
      double mgpu_scaling_eff = (G > 0 && T_viss_kernel > 0.0) ? clamp01d(sum_gpu_kernel / ((double)G * T_viss_kernel)) : 1.0;

      stage1_sum_gen += T_gen;
      stage1_sum_bcast += T_bcast;
      stage1_sum_viss_kernel += T_viss_kernel;
      stage1_sum_reduce += T_reduce;
      stage1_sum_phase += T_phase;
      stage1_sum_output += T_output_stage;
      stage1_sum_wall_active += T_wall;
      stage1_sum_serial_equiv += serial_equiv;
      stage1_sum_gpu_kernel += sum_gpu_kernel;

      double T_stage1_total = T_wall;

      stage1_metrics.push_back(Stage1SegMetrics{
        dayid, s+1, seg_t0_cur, tlen, segN, segN_half,
        T_gen, T_bcast, T_viss_kernel, T_reduce, T_phase, T_output_stage, T_wall, T_stage1_total,
        overlap_eff, mgpu_scaling_eff
      });

      if(s+1 < segs){
        orbit_gen_wait_slot(gen, nxt);
        broadcast_segment_to_gpus_async(gen, nxt, binfo, ctx, need_all_host);
      }
      if(s+2 < segs){
        orbit_gen_make_segment_async(gen, cur, seg_t0[s+2], seg_tlen[s+2], need_all_host);
      }

      HostTimer t_rseg;
      t_rseg.tic();
      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        cudaStream_t stream=ctx[gi].compute_stream;
        GpuSegSlot &slot = ctx[gi].slots[cur];
        int gw = (segN_half + 255) / 256;
        compute_pair_weight_half_kernel<<<gw,256,0,stream>>>(
          slot.d_u, slot.d_v, slot.d_w,
          segN_half,
          ext[gi].d_dcf, (int)hdcf.size(),
          ext[gi].d_pairw_half[cur]);
        CHECK_CUDA(cudaPeekAtLastError());

        int grid = ctx[gi].ntile_recon;
        if(blockage==1){
          recon_3d_direct_halfsym_tilecone_real<RECON_TILE_PIX_HOST, RECON_TILE_BL_HOST, true><<<grid, RECON_TILE_PIX_HOST, 0, stream>>>(
            ctx[gi].n_chunk,
            ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n,
            ctx[gi].d_tile_cx_recon, ctx[gi].d_tile_cy_recon, ctx[gi].d_tile_cz_recon,
            ctx[gi].d_tile_cosA_recon, ctx[gi].d_tile_sinA_recon, ctx[gi].ntile_recon,
            slot.d_u, slot.d_v, slot.d_w,
            slot.d_x1, slot.d_y1, slot.d_z1, slot.d_invn1,
            slot.d_x2, slot.d_y2, slot.d_z2, slot.d_invn2,
            ext[gi].d_pairw_half[cur],
            slot.d_Viss,
            segN_half,
            cosphi,
            ctx[gi].d_Cacc
          );
        }else{
          recon_3d_direct_halfsym_tilecone_real<RECON_TILE_PIX_HOST, RECON_TILE_BL_HOST, false><<<grid, RECON_TILE_PIX_HOST, 0, stream>>>(
            ctx[gi].n_chunk,
            ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n,
            ctx[gi].d_tile_cx_recon, ctx[gi].d_tile_cy_recon, ctx[gi].d_tile_cz_recon,
            ctx[gi].d_tile_cosA_recon, ctx[gi].d_tile_sinA_recon, ctx[gi].ntile_recon,
            slot.d_u, slot.d_v, slot.d_w,
            slot.d_x1, slot.d_y1, slot.d_z1, slot.d_invn1,
            slot.d_x2, slot.d_y2, slot.d_z2, slot.d_invn2,
            ext[gi].d_pairw_half[cur],
            slot.d_Viss,
            segN_half,
            cosphi,
            ctx[gi].d_Cacc
          );
        }
        CHECK_CUDA(cudaPeekAtLastError());
      }
      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        CHECK_CUDA(cudaStreamSynchronize(ctx[gi].compute_stream));
      }

      std::cout << "[day " << dayid << " seg " << (s+1) << "/" << segs
                << "] t0=" << seg_t0_cur
                << " segN=" << segN
                << " segN_half=" << segN_half
                << " Stage1{"
                << "T_gen=" << T_gen << "s"
                << ", T_bcast=" << T_bcast << "s"
                << ", T_viss_kernel=" << T_viss_kernel << "s"
                << ", T_reduce=" << T_reduce << "s"
                << ", T_phase=" << T_phase << "s"
                << ", T_output_stage=" << T_output_stage << "s"
                << ", T_stage1_total=" << T_wall << "s"
                << ", T_wall=" << T_wall << "s"
                << ", overlap efficiency=" << overlap_eff
                << ", multi-GPU scaling efficiency=" << mgpu_scaling_eff
                << "}"
                << " ReconSegTime=" << t_rseg.toc_s() << "s\n";
    }

    double T_wall_pipeline = t_stage1_pipeline.toc_s();
    double overlap_eff_day = (stage1_sum_serial_equiv > 0.0) ? clamp01d((stage1_sum_serial_equiv - T_wall_pipeline) / stage1_sum_serial_equiv) : 0.0;
    double mgpu_scaling_eff_day = (G > 0 && stage1_sum_viss_kernel > 0.0) ? clamp01d(stage1_sum_gpu_kernel / ((double)G * stage1_sum_viss_kernel)) : 1.0;

    std::cout << "[day " << dayid << " Stage-1 summary] "
              << "T_gen=" << stage1_sum_gen << "s "
              << "T_bcast=" << stage1_sum_bcast << "s "
              << "T_viss_kernel=" << stage1_sum_viss_kernel << "s "
              << "T_reduce=" << stage1_sum_reduce << "s "
              << "T_phase=" << stage1_sum_phase << "s "
              << "T_output_stage=" << stage1_sum_output << "s "
              << "T_stage1_total=" << T_wall_pipeline << "s "
              << "T_wall=" << T_wall_pipeline << "s "
              << "T_wall_active_sum=" << stage1_sum_wall_active << "s "
              << "T_serial_equiv=" << stage1_sum_serial_equiv << "s "
              << "overlap efficiency=" << overlap_eff_day << " "
              << "multi-GPU scaling efficiency=" << mgpu_scaling_eff_day << "\n";

    {
      std::string metrics_csv = out_dir + "stage1_metrics_day" + std::to_string(dayid) + "_" + btag + ".csv";
      std::ofstream mofs(metrics_csv);
      if(mofs.is_open()){
        mofs << "day,seg_idx,t0,tlen,segN,segN_half,T_gen,T_bcast,T_viss_kernel,T_reduce,T_phase,T_output_stage,T_stage1_total,T_wall,overlap_efficiency,multi_gpu_scaling_efficiency\n";
        mofs << std::setprecision(10);
        for(const auto& m : stage1_metrics){
          mofs << m.dayid << ','
               << m.seg_idx << ','
               << m.t0 << ','
               << m.tlen << ','
               << m.segN << ','
               << m.segN_half << ','
               << m.T_gen << ','
               << m.T_bcast << ','
               << m.T_viss_kernel << ','
               << m.T_reduce << ','
               << m.T_phase << ','
               << m.T_output_stage << ','
               << m.T_stage1_total << ','
               << m.T_wall << ','
               << m.overlap_efficiency << ','
               << m.multi_gpu_scaling_efficiency << '\n';
        }
        mofs << "summary,0,0,0,0,0,"
             << stage1_sum_gen << ','
             << stage1_sum_bcast << ','
             << stage1_sum_viss_kernel << ','
             << stage1_sum_reduce << ','
             << stage1_sum_phase << ','
             << stage1_sum_output << ','
             << T_wall_pipeline << ','
             << overlap_eff_day << ','
             << mgpu_scaling_eff_day << '\n';
      }
    }

    std::cout<<"Direct 3D recon done in "<<t_rec.toc_s()<<" s\n";

    HostTimer t_w;
    t_w.tic();
    std::string outC = out_dir+"C"+std::to_string(dayid)+"day"+btag + ((C_mode=="txt")? ".txt" : ".bin");
    if(C_mode=="txt"){
      std::ofstream ofs(outC);
      if(!ofs.is_open()){
        std::cerr<<"ERROR open "<<outC<<"\n";
        return 1;
      }
      static thread_local std::vector<char> outbuf(8<<20);
      ofs.rdbuf()->pubsetbuf(outbuf.data(), outbuf.size());
      const int CH=1<<20;
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        for(long long off=0; off<ctx[gi].n_chunk; off+=CH){
          int cur=(int)std::min((long long)CH, ctx[gi].n_chunk-off);
          CHECK_CUDA(cudaMemcpy(ctx[gi].h_chunk, ctx[gi].d_Cacc+off, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost));
          for(int i=0;i<cur;i++) ofs<<ctx[gi].h_chunk[i]<<"\n";
        }
      }
      ofs.close();
    } else {
      std::ofstream ofs(outC, std::ios::binary);
      if(!ofs.is_open()){
        std::cerr<<"ERROR open "<<outC<<"\n";
        return 1;
      }
      const int CH=1<<20;
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        for(long long off=0; off<ctx[gi].n_chunk; off+=CH){
          int cur=(int)std::min((long long)CH, ctx[gi].n_chunk-off);
          CHECK_CUDA(cudaMemcpy(ctx[gi].h_chunk, ctx[gi].d_Cacc+off, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost));
          ofs.write(reinterpret_cast<const char*>(ctx[gi].h_chunk), (std::streamsize)cur*sizeof(float));
        }
      }
      ofs.close();
    }
    std::cout<<"Wrote "<<outC<<" in "<<t_w.toc_s()<<" s\n";

    if(write_viss) ofsViss.close();
    if(write_baseline_txt){ ofsUVW.close(); ofsXYZa.close(); ofsXYZb.close(); }
    orbit_gen_destroy(gen);
  }

  #pragma omp parallel for num_threads(G)
  for(int gi=0; gi<G; ++gi){
    CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
    if(ctx[gi].h_chunk) cudaFreeHost(ctx[gi].h_chunk);

    if(ext[gi].d_pairw_half[1]) cudaFree(ext[gi].d_pairw_half[1]);
    if(ext[gi].d_pairw_half[0]) cudaFree(ext[gi].d_pairw_half[0]);
    if(ext[gi].d_dcf) cudaFree(ext[gi].d_dcf);

    if(ctx[gi].d_Cacc) cudaFree(ctx[gi].d_Cacc);

    for(int slot=0; slot<2; ++slot){
      GpuSegSlot &sg = ctx[gi].slots[slot];
      if(sg.viss_stop) cudaEventDestroy(sg.viss_stop);
      if(sg.viss_start) cudaEventDestroy(sg.viss_start);
      if(sg.ready) cudaEventDestroy(sg.ready);
      if(sg.bcast_start) cudaEventDestroy(sg.bcast_start);
      if(sg.h_Vpart) cudaFreeHost(sg.h_Vpart);
      if(sg.d_Viss) cudaFree(sg.d_Viss);
      if(sg.d_Vpart) cudaFree(sg.d_Vpart);
      if(sg.d_invn2) cudaFree(sg.d_invn2);
      if(sg.d_invn1) cudaFree(sg.d_invn1);
      if(sg.d_z2) cudaFree(sg.d_z2);
      if(sg.d_y2) cudaFree(sg.d_y2);
      if(sg.d_x2) cudaFree(sg.d_x2);
      if(sg.d_z1) cudaFree(sg.d_z1);
      if(sg.d_y1) cudaFree(sg.d_y1);
      if(sg.d_x1) cudaFree(sg.d_x1);
      if(sg.d_w) cudaFree(sg.d_w);
      if(sg.d_v) cudaFree(sg.d_v);
      if(sg.d_u) cudaFree(sg.d_u);
      if(sg.d_satz) cudaFree(sg.d_satz);
      if(sg.d_saty) cudaFree(sg.d_saty);
      if(sg.d_satx) cudaFree(sg.d_satx);
      if(sg.d_pos) cudaFree(sg.d_pos);
    }

    if(ctx[gi].d_tile_sinA_viss) cudaFree(ctx[gi].d_tile_sinA_viss);
    if(ctx[gi].d_tile_cosA_viss) cudaFree(ctx[gi].d_tile_cosA_viss);
    if(ctx[gi].d_tile_cz_viss) cudaFree(ctx[gi].d_tile_cz_viss);
    if(ctx[gi].d_tile_cy_viss) cudaFree(ctx[gi].d_tile_cy_viss);
    if(ctx[gi].d_tile_cx_viss) cudaFree(ctx[gi].d_tile_cx_viss);
    if(ctx[gi].d_tile_sinA_recon) cudaFree(ctx[gi].d_tile_sinA_recon);
    if(ctx[gi].d_tile_cosA_recon) cudaFree(ctx[gi].d_tile_cosA_recon);
    if(ctx[gi].d_tile_cz_recon) cudaFree(ctx[gi].d_tile_cz_recon);
    if(ctx[gi].d_tile_cy_recon) cudaFree(ctx[gi].d_tile_cy_recon);
    if(ctx[gi].d_tile_cx_recon) cudaFree(ctx[gi].d_tile_cx_recon);
    if(ctx[gi].d_n) cudaFree(ctx[gi].d_n);
    if(ctx[gi].d_nm1) cudaFree(ctx[gi].d_nm1);
    if(ctx[gi].d_m) cudaFree(ctx[gi].d_m);
    if(ctx[gi].d_l) cudaFree(ctx[gi].d_l);
    if(ctx[gi].d_B) cudaFree(ctx[gi].d_B);

    if(ctx[gi].xfer_stream) cudaStreamDestroy(ctx[gi].xfer_stream);
    if(ctx[gi].compute_stream) cudaStreamDestroy(ctx[gi].compute_stream);
  }

  return 0;
}
