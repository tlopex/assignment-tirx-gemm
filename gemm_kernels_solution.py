import tvm
from tvm.script import tirx as Tx

from tvm.tirx.op_schedule.cuda.common import tma_shared_layout, SwizzleMode
from tvm.tir.layout import TileLayout, S, TLane, TCol, tid_in_wg as axis_tid_in_wg
from tvm.tirx.tile_scheduler import ClusterPersistentScheduler2D
from tvm.tirx.pipeline import PipelineState, MBarrier, TMABar, TCGen05Bar

SM_COUNT = 148  # B200
F16_SIZE = 2

# ======================================================================
# Step 1: Single-tile synchronous GEMM
#   M=128, N=128, K=64 — exactly one tile, no loops.
#   All threads sync-load GMEM→SMEM, one MMA, sync writeback.
# ======================================================================


def hgemm_v1(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            bx, by = Tx.cta_id([M // BLK_M, N // BLK_N], parent="kernel")
            wg_id = Tx.warpgroup_id([1], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            mma_bar = pool.alloc((1,), "uint64", align=8)
            pool.move_base_to(1024)
            Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
            pool.commit()

            if warp_id == 0:
                if lane_id == 0:
                    Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=0,
                                  layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            m_st = Tx.meta_var(bx * BLK_M)
            n_st = Tx.meta_var(by * BLK_N)

            phase_mma: Tx.int32
            phase_mma = 0

            with Tx.cta():
                Tx.copy(Asmem[:],A[:])
                Tx.copy(Bsmem[:],B[:])
            
            Tx.cuda.cta_sync()
            Tx.ptx.tcgen05.fence.after_thread_sync()

            if warp_id==0:
                with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:], Bsmem[:],
                                          accum=False, dispatch="tcgen05", cta_group=1)
                    Tx.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            Tx.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)

            reg = Tx.alloc_local((BLK_N,),acc_type)
            reg_f16 = Tx.alloc_local((BLK_N,),d_type)

            reg_wg = reg.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1@axis_tid_in_wg, 1)]))

            with Tx.warpgroup():
                Tx.copy(reg_wg[:],tmem[:, :BLK_N])

            with Tx.thread():
                Tx.cast(reg_f16,reg)
                m_thr = Tx.meta_var(m_st+32*warp_id+lane_id)
                Tx.copy(D[m_thr,  :],reg_f16[:])


            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 2: K-loop — accumulate in TMEM
#   M=128, N=128, K=any multiple of 64.
#   Loop over K dimension with accumulation.
# ======================================================================

def hgemm_v2(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            bx, by = Tx.cta_id([1, 1], parent="kernel")
            wg_id = Tx.warpgroup_id([1], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            mma_bar = pool.alloc((1,), "uint64", align=8)
            pool.move_base_to(1024)
            Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
            pool.commit()

            if warp_id == 0:
                if lane_id == 0:
                    Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=0,
                                  layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            phase_mma: Tx.int32
            phase_mma = 0


            m_st = Tx.meta_var(bx * BLK_M)
            n_st = Tx.meta_var(by * BLK_N)

            for i in range(K_TILES):
                with Tx.cta():
                    Tx.copy(Asmem[:,:],A[:,i*BLK_K:(i+1)*BLK_K])
                    Tx.copy(Bsmem[:,:],B[:,i*BLK_K:(i+1)*BLK_K])

                Tx.cuda.cta_sync()
                Tx.ptx.tcgen05.fence.after_thread_sync()

                if warp_id==0:
                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        Tx.gemm_async(tmem[:, :BLK_N], Asmem[:], Bsmem[:],
                                            accum=(i!=0), dispatch="tcgen05", cta_group=1)
                        Tx.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

                Tx.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                phase_mma^=1

            reg = Tx.alloc_local((BLK_N,),acc_type)
            reg_f16 = Tx.alloc_local((BLK_N,),d_type)

            reg_wg = reg.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1@axis_tid_in_wg, 1)]))

            with Tx.warpgroup():
                Tx.copy(reg_wg[:],tmem[:, :BLK_N])
                Tx.cuda.cta_sync()
                Tx.ptx.tcgen05.fence.after_thread_sync()

            with Tx.thread():
                Tx.cast(reg_f16[:],reg[:])
                m_thr = Tx.meta_var(m_st+32*warp_id+lane_id)
                Tx.copy(D[m_thr, :],reg_f16[:])


            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 3: Spatial tiling — multi-CTA
#   M, N any multiples of 128, K any multiple of 64.
#   Grid of (M/128)×(N/128) CTAs.
# ======================================================================

def hgemm_v3(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            bx, by = Tx.cta_id([M//BLK_M, N//BLK_N], parent="kernel")
            wg_id = Tx.warpgroup_id([1], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            mma_bar = pool.alloc((1,), "uint64", align=8)
            pool.move_base_to(1024)
            Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
            pool.commit()

            if warp_id == 0:
                if lane_id == 0:
                    Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=0,
                                  layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            phase_mma: Tx.int32
            phase_mma = 0


            m_st = Tx.meta_var(bx * BLK_M)
            n_st = Tx.meta_var(by * BLK_N)

            for i in range(K_TILES):
                with Tx.cta():
                    Tx.copy(Asmem[:,:],A[m_st:m_st+BLK_M,i*BLK_K:(i+1)*BLK_K])
                    Tx.copy(Bsmem[:,:],B[n_st:n_st+BLK_N,i*BLK_K:(i+1)*BLK_K])

                Tx.cuda.cta_sync()
                Tx.ptx.tcgen05.fence.after_thread_sync()

                if warp_id==0:
                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        Tx.gemm_async(tmem[:, :BLK_N], Asmem[:], Bsmem[:],
                                            accum=(i!=0), dispatch="tcgen05", cta_group=1)
                        Tx.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

                Tx.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                phase_mma^=1

            reg = Tx.alloc_local((BLK_N,),acc_type)
            reg_f16 = Tx.alloc_local((BLK_N,),d_type)

            reg_wg = reg.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1@axis_tid_in_wg, 1)]))

            with Tx.warpgroup():
                Tx.copy(reg_wg[:],tmem[:, :BLK_N])
                Tx.cuda.cta_sync()
                Tx.ptx.tcgen05.fence.after_thread_sync()

            with Tx.thread():
                Tx.cast(reg_f16[:],reg[:])
                m_thr = Tx.meta_var(m_st+32*warp_id+lane_id)
                Tx.copy(D[m_thr, n_st:n_st+BLK_N],reg_f16[:])


            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 4: TMA async load + TMA store
#   Replace sync GMEM→SMEM load with TMA (single-thread dispatch, mbarrier sync).
#   Writeback: TMEM → RF → SMEM → TMA store → GMEM.
# ======================================================================

def hgemm_v4(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    # Note: Constants like TMEM_LD_N, EPI_N, MMA_N must be defined outside
    # @Tx.prim_func (alongside BLK_M, BLK_K, etc.). Variables assigned inside
    # the kernel function are treated as TIR variables (dynamic), not Python
    # constants — this causes dispatch failures when used in buffer slicing.
    TMEM_LD_N:Tx.int
    TMEM_LD_N=8
    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            bx, by = Tx.cta_id([M // BLK_M, N // BLK_N], parent="kernel")
            wg_id = Tx.warpgroup_id([1], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            tma_bar = pool.alloc((1,), "uint64", align=8)
            mma_bar = pool.alloc((1,), "uint64", align=8)
            pool.move_base_to(1024)
            Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
            Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
            pool.commit()

            if warp_id == 0:
                if lane_id == 0:
                    Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                    Tx.ptx.mbarrier.init(tma_bar.ptr_to([0]), 1)
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=0,
                                  layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            m_st = Tx.meta_var(bx * BLK_M)
            n_st = Tx.meta_var(by * BLK_N)

            phase_tma: Tx.int32
            phase_mma: Tx.int32
            phase_tma = 0
            phase_mma = 0

            @Tx.inline
            def tma_load(i):
                Tx.copy_async(Asmem[:,:],A[m_st:m_st+BLK_M,i*BLK_K:(i+1)*BLK_K],dispatch="tma",cta_group=1,mbar=tma_bar.ptr_to([0]))
                Tx.copy_async(Bsmem[:,:],B[n_st:n_st+BLK_N,i*BLK_K:(i+1)*BLK_K],dispatch="tma",cta_group=1,mbar=tma_bar.ptr_to([0]))
                Tx.ptx.mbarrier.arrive.expect_tx(tma_bar.ptr_to([0]),(BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

            @Tx.inline
            def mma(i):
                Tx.ptx.mbarrier.try_wait(tma_bar.ptr_to([0]), phase_tma)
                phase_tma^=1
                Tx.ptx.tcgen05.fence.after_thread_sync()
                Tx.gemm_async(tmem[:, :BLK_N], Asmem[:], Bsmem[:],
                                            accum=(i!=0), dispatch="tcgen05", cta_group=1)
                Tx.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
                Tx.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                phase_mma^=1

            if warp_id==0:
                with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                    for k in range(K_TILES):
                        tma_load(k)
                        mma(k)

            # Warp 0 runs the TMA+MMA loop inside elect_sync scope as a single thread.
            # The other 3 warps are not blocked by `if warp_id == 0:`, so they skip
            # straight to the writeback code. At that point, warp 0's MMA may not
            # have finished yet — TMEM may still be empty or mid-write. cta_sync
            # makes warps 1/2/3 wait here until warp 0 finishes the loop, ensuring
            # all MMAs are complete before anyone reads TMEM.
            Tx.cuda.cta_sync()
            Tx.ptx.tcgen05.fence.after_thread_sync()

            reg_f16 = Tx.alloc_local((BLK_N,),d_type)

            for no in Tx.unroll(BLK_N // TMEM_LD_N):
                reg = Tx.alloc_local((TMEM_LD_N,), acc_type)
                reg_wg = reg.view(128, TMEM_LD_N, layout=TileLayout(S[(128, TMEM_LD_N) : (1@axis_tid_in_wg, 1)]))
                with Tx.warpgroup():
                    Tx.copy(reg_wg[:, :], tmem[:, no * TMEM_LD_N : no * TMEM_LD_N + TMEM_LD_N])
                with Tx.thread():
                    Tx.cast(reg_f16[no * TMEM_LD_N : no * TMEM_LD_N + TMEM_LD_N], reg[:])

            with Tx.thread():
                Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.warpgroup_sync(10)
            with Tx.thread(parent="warpgroup")[Tx.ptx.elect_sync()]:
                n_st_epi = Tx.meta_var(n_st)
                Tx.copy_async(D[m_st : m_st + BLK_M, n_st_epi : n_st_epi + BLK_N], Dsmem[:, :], dispatch="tma")

            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel


# ======================================================================
# Step 5: Software pipeline
#   PIPE_DEPTH=2 double-buffered SMEM. Prefetch + overlap.
# ======================================================================


def hgemm_v5(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
    PRE_NUM = min(PIPE_DEPTH, K_TILES)
    TMEM_LD_N = 8

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            bx, by = Tx.cta_id([M // BLK_M, N // BLK_N], parent="kernel")
            wg_id = Tx.warpgroup_id([1], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
            mma_bar = pool.alloc((1,), "uint64", align=8)
            pool.move_base_to(1024)
            Asmem = pool.alloc((PIPE_DEPTH,BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((PIPE_DEPTH,BLK_N, BLK_K), b_type, layout=B_layout)
            Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
            pool.commit()

            if warp_id == 0:
                if lane_id == 0:
                    Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                    for i in range(PIPE_DEPTH):
                        Tx.ptx.mbarrier.init(tma_bar.ptr_to([i]), 1)
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=0,
                                  layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            m_st = Tx.meta_var(bx * BLK_M)
            n_st = Tx.meta_var(by * BLK_N)

            phase_tma: Tx.int32
            phase_mma: Tx.int32
            phase_tma = 0
            phase_mma = 0

            @Tx.inline
            def tma_load(k,i):
                k_tile = k*PIPE_DEPTH+i
                Tx.copy_async(Asmem[i,:,:],A[m_st:m_st+BLK_M,k_tile*BLK_K:(k_tile+1)*BLK_K],dispatch="tma",cta_group=1,mbar=tma_bar.ptr_to([i]))
                Tx.copy_async(Bsmem[i,:,:],B[n_st:n_st+BLK_N,k_tile*BLK_K:(k_tile+1)*BLK_K],dispatch="tma",cta_group=1,mbar=tma_bar.ptr_to([i]))
                Tx.ptx.mbarrier.arrive.expect_tx(tma_bar.ptr_to([i]),(BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

            @Tx.inline
            def mma(k,i):
                Tx.ptx.mbarrier.try_wait(tma_bar.ptr_to([i]), phase_tma)
                k_tile = k*PIPE_DEPTH+i
                if i == PIPE_DEPTH - 1:
                    phase_tma^=1
                Tx.ptx.tcgen05.fence.after_thread_sync()
                Tx.gemm_async(tmem[:, :BLK_N], Asmem[i,:,:], Bsmem[i,:,:],
                                            accum=(k_tile!=0), dispatch="tcgen05", cta_group=1)
                Tx.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
                Tx.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                phase_mma^=1

            if warp_id==0:
                with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                    for i in range(PRE_NUM):
                        tma_load(0,i)

                    k1: Tx.int32
                    k1=PRE_NUM
                    for k in range(K_TILES):
                        mma(k//PIPE_DEPTH, k%PIPE_DEPTH)
                        if k1<K_TILES:
                            tma_load(k1//PIPE_DEPTH, k1%PIPE_DEPTH)
                            k1+=1

            Tx.cuda.cta_sync()
            Tx.ptx.tcgen05.fence.after_thread_sync()

            reg_f16 = Tx.alloc_local((BLK_N,),d_type)

            for i in Tx.unroll(BLK_N // TMEM_LD_N):
                reg = Tx.alloc_local((TMEM_LD_N,), acc_type)
                reg_wg = reg.view(128, TMEM_LD_N, layout=TileLayout(S[(128, TMEM_LD_N) : (1@axis_tid_in_wg, 1)]))
                with Tx.warpgroup():
                    Tx.copy(reg_wg[:], tmem[:, i*TMEM_LD_N:(i+1)*TMEM_LD_N])
                    Tx.cuda.cta_sync()
                    Tx.ptx.tcgen05.fence.after_thread_sync()

                with Tx.thread():
                    Tx.cast(reg_f16[i*TMEM_LD_N:(i+1)*TMEM_LD_N], reg[:])

            with Tx.thread():
                Tx.copy(Dsmem[warp_id*32+lane_id, :], reg_f16[:])
                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.cuda.warpgroup_sync(10)

            with Tx.thread(parent="warpgroup")[Tx.ptx.elect_sync()]:
                Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N], Dsmem[:, :], dispatch="tma")
                Tx.ptx.cp_async.bulk.commit_group()
                Tx.ptx.cp_async.bulk.wait_group(0)
            Tx.cuda.warpgroup_sync(10)

            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)


    return kernel


# ======================================================================
# Step 6: Persistent kernel + tile scheduler
#   Fixed SM_COUNT CTAs, loop over tiles with L2-friendly ordering.
# ======================================================================


def hgemm_v6(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
    PRE_NUM = min(PIPE_DEPTH, K_TILES)

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, ( BLK_N, BLK_M))


    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            wg_id = Tx.warpgroup_id([1], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
            mma_bar = pool.alloc((1,), "uint64", align=8)
            pool.move_base_to(1024)
            Asmem = pool.alloc((PIPE_DEPTH,BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((PIPE_DEPTH,BLK_N, BLK_K), b_type, layout=B_layout)
            Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
            pool.commit()

            tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // BLK_M, num_n_tiles=N // BLK_N,
            l2_group_size=8, num_clusters=SM_COUNT)

            if warp_id == 0:
                if lane_id == 0:
                    Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                    for i in range(PIPE_DEPTH):
                        Tx.ptx.mbarrier.init(tma_bar.ptr_to([i]), 1)
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=0,
                                  layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            phase_tma = Tx.alloc_local((1,),"int32")
            phase_mma = Tx.alloc_local((1,),"int32")
            phase_tma[0] = 0
            phase_mma[0] = 0
            tile_scheduler.init(bx)

            while tile_scheduler.valid():
                m_st = Tx.meta_var(tile_scheduler.m_idx * BLK_M)
                n_st = Tx.meta_var(tile_scheduler.n_idx * BLK_N)

                @Tx.inline
                def tma_load(k,i):
                    k_tile = k*PIPE_DEPTH+i
                    Tx.copy_async(Asmem[i,:,:],A[m_st:m_st+BLK_M,k_tile*BLK_K:(k_tile+1)*BLK_K],dispatch="tma",cta_group=1,mbar=tma_bar.ptr_to([i]))
                    Tx.copy_async(Bsmem[i,:,:],B[n_st:n_st+BLK_N,k_tile*BLK_K:(k_tile+1)*BLK_K],dispatch="tma",cta_group=1,mbar=tma_bar.ptr_to([i]))
                    Tx.ptx.mbarrier.arrive.expect_tx(tma_bar.ptr_to([i]),(BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

                @Tx.inline
                def mma(k,i):
                    Tx.ptx.mbarrier.try_wait(tma_bar.ptr_to([i]), phase_tma[0])
                    k_tile = k*PIPE_DEPTH+i
                    if i == PIPE_DEPTH - 1:
                        phase_tma[0]^=1
                    Tx.ptx.tcgen05.fence.after_thread_sync()
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[i,:,:], Bsmem[i,:,:],
                                                accum=(k_tile!=0), dispatch="tcgen05", cta_group=1)
                    Tx.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
                    Tx.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma[0])
                    phase_mma[0]^=1

                if warp_id==0:
                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        for i in range(PRE_NUM):
                            tma_load(0,i)

                        k1: Tx.int32
                        k1=PRE_NUM
                        for k in range(K_TILES):
                            mma(k//PIPE_DEPTH, k%PIPE_DEPTH)
                            if k1<K_TILES:
                                tma_load(k1//PIPE_DEPTH, k1%PIPE_DEPTH)
                                k1+=1

                Tx.cuda.cta_sync()
                Tx.ptx.tcgen05.fence.after_thread_sync()

                reg_f16 = Tx.alloc_local((BLK_N,),d_type)
                reg = Tx.alloc_local((BLK_N,),acc_type)

                reg_wg = reg.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1@axis_tid_in_wg, 1)]))

                with Tx.warpgroup():
                    Tx.copy(reg_wg[:],tmem[:, :BLK_N])
                    Tx.cuda.cta_sync()
                    Tx.ptx.tcgen05.fence.after_thread_sync()

                with Tx.thread():
                    Tx.cast(reg_f16[:],reg[:])

                with Tx.thread():
                    Tx.copy(Dsmem[warp_id*32+lane_id,:],reg_f16[:])
                    Tx.ptx.fence.proxy_async("shared::cta")
                    Tx.cuda.warpgroup_sync(10)

                with Tx.thread(parent="warpgroup")[Tx.ptx.elect_sync()]:
                    n_st_epi = Tx.meta_var(n_st)
                    Tx.copy_async(D[m_st : m_st + BLK_M, n_st_epi : n_st_epi + BLK_N], Dsmem[:, :], dispatch="tma")
                    Tx.ptx.cp_async.bulk.commit_group()
                    Tx.ptx.cp_async.bulk.wait_group(0)
                Tx.cuda.warpgroup_sync(10)

                Tx.cuda.cta_sync()
                tile_scheduler.next_tile()

            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)


    return kernel


# ======================================================================
# Step 7: Warp specialization
#   WG1: warp0 (MMA) + warp3 (TMA producer)
#   WG0: writeback (TMEM → RF → SMEM → GMEM)
#   4 barrier types: tma2mma, mma2tma, mma2ld, ld2mma
#   PIPE_DEPTH=2 (same as step 6, focus on warp spec structure)
# ======================================================================


def hgemm_v7(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
    WG_NUMBER = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            tma2mma = TMABar(pool, PIPE_DEPTH, "tma2mma")
            mma2tma = TCGen05Bar(pool, PIPE_DEPTH, "mma2tma")
            mma2ld = TCGen05Bar(pool, 1, "mma2ld")
            ld2mma = MBarrier(pool, 1, "ld2mma")
            pool.move_base_to(1024)
            Asmem = pool.alloc((PIPE_DEPTH,BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((PIPE_DEPTH,BLK_N, BLK_K), b_type, layout=B_layout)
            Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)

            tma2mma.init(1)
            mma2tma.init(1)
            mma2ld.init(1)
            ld2mma.init(128)

            pool.commit()

            tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // BLK_M, num_n_tiles=N // BLK_N,
            l2_group_size=8, num_clusters=SM_COUNT)

            if wg_id == 0:
                if warp_id == 0:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=0,
                                  layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            tile_scheduler.init(bx)
            m_st = Tx.meta_var(tile_scheduler.m_idx * BLK_M)
            n_st = Tx.meta_var(tile_scheduler.n_idx * BLK_N)

            if wg_id==1:
                if warp_id==3:
                    tma_phase = PipelineState("tma", PIPE_DEPTH)
                    tma_phase.init(is_producer=True)

                    @Tx.inline
                    def tma_load(k,i):
                        k_tile = k*PIPE_DEPTH+i
                        Tx.copy_async(Asmem[i,:,:],A[m_st:m_st+BLK_M,k_tile*BLK_K:(k_tile+1)*BLK_K],dispatch="tma",cta_group=1,mbar=tma2mma.ptr_to([tma_phase.stage]))
                        Tx.copy_async(Bsmem[i,:,:],B[n_st:n_st+BLK_N,k_tile*BLK_K:(k_tile+1)*BLK_K],dispatch="tma",cta_group=1,mbar=tma2mma.ptr_to([tma_phase.stage]))

                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            for k in range(K_TILES):
                                mma2tma.wait(tma_phase.stage,tma_phase.phase)
                                tma_load(k//PIPE_DEPTH, k%PIPE_DEPTH)
                                tma2mma.arrive(tma_phase.stage, (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                                tma_phase.move_to_next_stage()
                            tile_scheduler.next_tile()

                elif warp_id ==0:
                    mma_phase = PipelineState("mma", PIPE_DEPTH)
                    mma_phase.init(is_producer=False)
                    ld_phase  = PipelineState("ld", 1)
                    ld_phase.init(is_producer=True)

                    @Tx.inline
                    def mma(k,i):
                        k_tile = k*PIPE_DEPTH+i
                        Tx.ptx.tcgen05.fence.after_thread_sync()
                        Tx.gemm_async(tmem[:, :BLK_N], Asmem[i,:,:], Bsmem[i,:,:],
                                                    accum=(k_tile!=0), dispatch="tcgen05", cta_group=1)

                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            ld2mma.wait(ld_phase.stage,ld_phase.phase)
                            ld_phase.move_to_next_stage()

                            for k in range(K_TILES):
                                tma2mma.wait(mma_phase.stage,mma_phase.phase)
                                mma(k//PIPE_DEPTH, k%PIPE_DEPTH)
                                mma2tma.arrive(mma_phase.stage, cta_group=1, cta_mask=0)
                                mma_phase.move_to_next_stage()
                            mma2ld.arrive(0, cta_group=1, cta_mask=0)

                            tile_scheduler.next_tile()

            elif wg_id ==0:
                wb_phase = PipelineState("wb", 1)
                wb_phase.init(is_producer=False)

                reg_f16 = Tx.alloc_local((BLK_N,),d_type)
                reg = Tx.alloc_local((BLK_N,),acc_type)

                reg_wg = reg.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1@axis_tid_in_wg, 1)]))

                while tile_scheduler.valid():
                    mma2ld.wait(wb_phase.stage,wb_phase.phase)
                    wb_phase.move_to_next_stage()
                    with Tx.warpgroup():
                        Tx.copy(reg_wg[:],tmem[:, :BLK_N])

                    ld2mma.arrive(0, cta_id=0, pred=True)

                    with Tx.thread():
                        Tx.cast(reg_f16[:],reg[:])

                    with Tx.thread():
                        Tx.copy(Dsmem[warp_id*32+lane_id,:],reg_f16[:])
                        Tx.ptx.fence.proxy_async("shared::cta")
                        Tx.cuda.warpgroup_sync(10)

                    with Tx.thread(parent="warpgroup")[Tx.ptx.elect_sync()]:
                        n_st_epi = Tx.meta_var(n_st)
                        Tx.copy_async(D[m_st : m_st + BLK_M, n_st_epi : n_st_epi + BLK_N], Dsmem[:, :], dispatch="tma")
                        Tx.ptx.cp_async.bulk.commit_group()
                        Tx.ptx.cp_async.bulk.wait_group(0)
                    Tx.cuda.warpgroup_sync(10)
                    tile_scheduler.next_tile()

            Tx.cuda.cta_sync()
            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)


    return kernel


# ======================================================================
# Step 8: Deeper pipeline (PIPE_DEPTH=4)
#   Same warp-specialized structure as v7, but with 4-stage pipeline
#   to better hide TMA latency. Only changes: PIPE_DEPTH=2 → 4,
#   which affects barrier array sizes and Asmem/Bsmem stage dimensions.
# ======================================================================

def hgemm_v8(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    WG_NUMBER = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            tma2mma = TMABar(pool, PIPE_DEPTH, "tma2mma")
            mma2tma = TCGen05Bar(pool, PIPE_DEPTH, "mma2tma")
            mma2ld = TCGen05Bar(pool, 1, "mma2ld")
            ld2mma = MBarrier(pool, 1, "ld2mma")
            pool.move_base_to(1024)
            Asmem = pool.alloc((PIPE_DEPTH,BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((PIPE_DEPTH,BLK_N, BLK_K), b_type, layout=B_layout)
            Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)

            tma2mma.init(1)
            mma2tma.init(1)
            mma2ld.init(1)
            ld2mma.init(128)

            pool.commit()

            tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // BLK_M, num_n_tiles=N // BLK_N,
            l2_group_size=8, num_clusters=SM_COUNT)

            if wg_id == 0:
                if warp_id == 0:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=0,
                                  layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            tile_scheduler.init(bx)
            m_st = Tx.meta_var(tile_scheduler.m_idx * BLK_M)
            n_st = Tx.meta_var(tile_scheduler.n_idx * BLK_N)

            if wg_id==1:
                if warp_id==3:
                    tma_phase = PipelineState("tma", PIPE_DEPTH)
                    tma_phase.init(is_producer=True)

                    @Tx.inline
                    def tma_load(k,i):
                        k_tile = k*PIPE_DEPTH+i
                        Tx.copy_async(Asmem[i,:,:],A[m_st:m_st+BLK_M,k_tile*BLK_K:(k_tile+1)*BLK_K],dispatch="tma",cta_group=1,mbar=tma2mma.ptr_to([tma_phase.stage]))
                        Tx.copy_async(Bsmem[i,:,:],B[n_st:n_st+BLK_N,k_tile*BLK_K:(k_tile+1)*BLK_K],dispatch="tma",cta_group=1,mbar=tma2mma.ptr_to([tma_phase.stage]))

                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            for k in range(K_TILES):
                                mma2tma.wait(tma_phase.stage,tma_phase.phase)
                                tma_load(k//PIPE_DEPTH, k%PIPE_DEPTH)
                                tma2mma.arrive(tma_phase.stage, (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                                tma_phase.move_to_next_stage()
                            tile_scheduler.next_tile()

                elif warp_id ==0:
                    mma_phase = PipelineState("mma", PIPE_DEPTH)
                    mma_phase.init(is_producer=False)
                    ld_phase  = PipelineState("ld", 1)
                    ld_phase.init(is_producer=True)

                    @Tx.inline
                    def mma(k,i):
                        k_tile = k*PIPE_DEPTH+i
                        Tx.ptx.tcgen05.fence.after_thread_sync()
                        Tx.gemm_async(tmem[:, :BLK_N], Asmem[i,:,:], Bsmem[i,:,:],
                                                    accum=(k_tile!=0), dispatch="tcgen05", cta_group=1)

                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            ld2mma.wait(ld_phase.stage,ld_phase.phase)
                            ld_phase.move_to_next_stage()

                            for k in range(K_TILES):
                                tma2mma.wait(mma_phase.stage,mma_phase.phase)
                                mma(k//PIPE_DEPTH, k%PIPE_DEPTH)
                                mma2tma.arrive(mma_phase.stage, cta_group=1, cta_mask=0)
                                mma_phase.move_to_next_stage()
                            mma2ld.arrive(0, cta_group=1, cta_mask=0)

                            tile_scheduler.next_tile()

            elif wg_id ==0:
                wb_phase = PipelineState("wb", 1)
                wb_phase.init(is_producer=False)

                reg_f16 = Tx.alloc_local((BLK_N,),d_type)
                reg = Tx.alloc_local((BLK_N,),acc_type)

                reg_wg = reg.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1@axis_tid_in_wg, 1)]))

                while tile_scheduler.valid():
                    mma2ld.wait(wb_phase.stage,wb_phase.phase)
                    wb_phase.move_to_next_stage()
                    with Tx.warpgroup():
                        Tx.copy(reg_wg[:],tmem[:, :BLK_N])

                    ld2mma.arrive(0, cta_id=0, pred=True)

                    with Tx.thread():
                        Tx.cast(reg_f16[:],reg[:])

                    with Tx.thread():
                        Tx.copy(Dsmem[warp_id*32+lane_id,:],reg_f16[:])
                        Tx.ptx.fence.proxy_async("shared::cta")
                        Tx.cuda.warpgroup_sync(10)

                    with Tx.thread(parent="warpgroup")[Tx.ptx.elect_sync()]:
                        n_st_epi = Tx.meta_var(n_st)
                        Tx.copy_async(D[m_st : m_st + BLK_M, n_st_epi : n_st_epi + BLK_N], Dsmem[:, :], dispatch="tma")
                        Tx.ptx.cp_async.bulk.commit_group()
                        Tx.ptx.cp_async.bulk.wait_group(0)
                    Tx.cuda.warpgroup_sync(10)
                    tile_scheduler.next_tile()

            Tx.cuda.cta_sync()
            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)


    return kernel


# ======================================================================
# Step 9: Cluster — 2-CTA cooperation
#   CTA_GROUP=2, MMA_M=MMA_N=256, cross-CTA TMEM sharing.
# ======================================================================

def hgemm_v9(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_N = BLK_N * CTA_GROUP
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    WG_NUMBER = 2
    DTYPE_SIZE = a_type.bits // 8

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))


    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            cbx, cby = Tx.cta_id([CTA_GROUP, 1], parent="cluster")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            tma2mma = TMABar(pool, PIPE_DEPTH, "tma2mma")
            mma2tma = TCGen05Bar(pool, PIPE_DEPTH, "mma2tma")
            mma2ld = TCGen05Bar(pool, 1, "mma2ld")
            ld2mma = MBarrier(pool, 1, "ld2mma")
            pool.move_base_to(1024)
            Asmem = pool.alloc((PIPE_DEPTH,BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((PIPE_DEPTH,BLK_N, BLK_K), b_type, layout=B_layout)
            Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)

            tma2mma.init(1)
            mma2tma.init(1)
            mma2ld.init(1)
            ld2mma.init(128 * CTA_GROUP)

            pool.commit()

            # Cluster of CTA_GROUP CTAs per tile; each cluster owns an MMA_N × MMA_N output.
            tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // MMA_N, num_n_tiles=N // MMA_N,
            l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)

            if wg_id == 0:
                if warp_id == 0:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=2)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=0,
                                  layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            tile_scheduler.init(bx // CTA_GROUP)
            m_idx = Tx.meta_var(tile_scheduler.m_idx)
            n_idx = Tx.meta_var(tile_scheduler.n_idx)
            m_st = Tx.meta_var((m_idx * CTA_GROUP + cbx) * BLK_M)
            n_st = Tx.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

            tma2mma_cta0 = tma2mma.remote_view(0)

            if wg_id==1:
                if warp_id==3:
                    tma_phase = PipelineState("tma", PIPE_DEPTH)
                    tma_phase.init(is_producer=True)

                    @Tx.inline
                    def tma_load(k,i):
                        k_tile = k*PIPE_DEPTH+i
                        Tx.copy_async(Asmem[i,:,:],A[m_st:m_st+BLK_M,k_tile*BLK_K:(k_tile+1)*BLK_K],dispatch="tma",cta_group=2,mbar=tma2mma_cta0.ptr_to([tma_phase.stage]))
                        Tx.copy_async(Bsmem[i,:,:],B[n_st:n_st+BLK_N,k_tile*BLK_K:(k_tile+1)*BLK_K],dispatch="tma",cta_group=2,mbar=tma2mma_cta0.ptr_to([tma_phase.stage]))

                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            for k in range(K_TILES):
                                mma2tma.wait(tma_phase.stage,tma_phase.phase)
                                tma_load(k//PIPE_DEPTH, k%PIPE_DEPTH)
                                if cbx == 0:
                                    tma2mma_cta0.arrive(tma_phase.stage, CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * DTYPE_SIZE)
                                tma_phase.move_to_next_stage()
                            tile_scheduler.next_tile()

                elif warp_id ==0:
                    mma_phase = PipelineState("mma", PIPE_DEPTH)
                    mma_phase.init(is_producer=False)
                    ld_phase  = PipelineState("ld", 1)
                    ld_phase.init(is_producer=True)

                    @Tx.inline
                    def mma(k,i):
                        k_tile = k*PIPE_DEPTH+i
                        Tx.ptx.tcgen05.fence.after_thread_sync()
                        Tx.gemm_async(tmem[:, :MMA_N], Asmem[i,:,:], Bsmem[i,:,:],
                                                    accum=(k_tile!=0), dispatch="tcgen05", cta_group=2)
                    if cbx == 0:
                        with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                            while tile_scheduler.valid():
                                ld2mma.wait(ld_phase.stage,ld_phase.phase)
                                ld_phase.move_to_next_stage()

                                for k in range(K_TILES):
                                    tma2mma.wait(mma_phase.stage,mma_phase.phase)
                                    mma(k//PIPE_DEPTH, k%PIPE_DEPTH)
                                    mma2tma.arrive(mma_phase.stage, cta_group=2, cta_mask=3)
                                    mma_phase.move_to_next_stage()
                                mma2ld.arrive(0, cta_group=2, cta_mask=3)

                                tile_scheduler.next_tile()

            elif wg_id ==0:
                wb_phase = PipelineState("wb", 1)
                wb_phase.init(is_producer=False)

                # Writeback MMA_N columns in BLK_N-wide chunks (register budget).
                while tile_scheduler.valid():
                    mma2ld.wait(wb_phase.stage,wb_phase.phase)

                    wb_phase.move_to_next_stage()
                    Tx.ptx.tcgen05.fence.after_thread_sync()
                    reg_f16 = Tx.alloc_local((BLK_N,),d_type)
                    for no in Tx.unroll(MMA_N // BLK_N):
                        reg = Tx.alloc_local((BLK_N,),acc_type)
                        reg_wg = reg.view(BLK_M, BLK_N, layout=TileLayout(S[(BLK_M, BLK_N) : (1@axis_tid_in_wg, 1)]))

                        with Tx.warpgroup():
                            Tx.copy(reg_wg[:], tmem[:, no*BLK_N:(no+1)*BLK_N])
                        with Tx.thread():
                            Tx.cast(reg_f16[:],reg[:])

                        with Tx.thread():
                            Tx.copy(Dsmem[warp_id*32+lane_id,:],reg_f16[:])
                            Tx.ptx.fence.proxy_async("shared::cta")
                            Tx.cuda.warpgroup_sync(10)

                        with Tx.thread(parent="warpgroup")[Tx.ptx.elect_sync()]:
                            n_st_epi = Tx.meta_var(n_idx * MMA_N + no * BLK_N)
                            Tx.copy_async(D[m_st : m_st + BLK_M, n_st_epi : n_st_epi + BLK_N], Dsmem[:, :], dispatch="tma")
                            Tx.ptx.cp_async.bulk.commit_group()
                            Tx.ptx.cp_async.bulk.wait_group(0)
                        Tx.cuda.warpgroup_sync(10)
                    ld2mma.arrive(0, cta_id=0, pred=True)
                    tile_scheduler.next_tile()


            Tx.cuda.cluster_sync()
            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=2)


    return kernel


# ======================================================================
# Step 10: 2-consumer warp specialization
#   NUM_CONSUMER=2, WG2 (TMA+MMA), WG0/WG1 (writeback).
#   This is the final optimized kernel.
# ======================================================================

def hgemm_v10(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    NUM_CONSUMER = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_N = BLK_N * CTA_GROUP
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    WG_NUMBER = 3
    DTYPE_SIZE = a_type.bits // 8

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (NUM_CONSUMER, BLK_M, EPI_N))


    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        # fmt: off
        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            cbx, cby = Tx.cta_id([CTA_GROUP, 1], parent="cluster")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            tma2mma = TMABar(pool, PIPE_DEPTH, "tma2mma")
            mma2tma = TCGen05Bar(pool, PIPE_DEPTH, "mma2tma")
            mma2ld = TCGen05Bar(pool, NUM_CONSUMER, "mma2ld")
            ld2mma = MBarrier(pool, NUM_CONSUMER, "ld2mma")
            pool.move_base_to(1024)
            Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
            Dsmem = pool.alloc((NUM_CONSUMER, BLK_M, EPI_N), d_type, layout=D_layout)

            tma2mma.init(1)
            mma2tma.init(NUM_CONSUMER)
            mma2ld.init(1)
            ld2mma.init(128 * CTA_GROUP)

            pool.commit()

            tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // MMA_N // NUM_CONSUMER, num_n_tiles=N // MMA_N,
            l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)

            if wg_id == 0:
                if warp_id == 0:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=2)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer((128, 512), "float32", scope="tmem", allocated_addr=0,
                                  layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            tile_scheduler.init(bx // CTA_GROUP)
            m_idx = Tx.meta_var(tile_scheduler.m_idx)
            n_idx = Tx.meta_var(tile_scheduler.n_idx)
            m_st = Tx.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M)
            n_st = Tx.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

            tma2mma_cta0 = tma2mma.remote_view(0)

            if wg_id==2:
                if warp_id==3:
                    tma_phase = PipelineState("tma", PIPE_DEPTH)
                    tma_phase.init(is_producer=True)

                    # Two consumers need different M-row blocks of A but share
                    # the same K-chunk and B (multicast).
                    #   consumer 0: A[m_st         : m_st+BLK_M,        K_chunk_z]
                    #   consumer 1: A[m_st+CTA_GRP : m_st+CTA_GRP+BLK_M, K_chunk_z]
                    @Tx.inline
                    def tma_load(k,i):
                        k_tile = k*PIPE_DEPTH+i
                        m_st_c1 = Tx.meta_var(m_st + CTA_GROUP * BLK_M)
                        Tx.copy_async(Asmem[i,0,:,:],A[m_st:m_st+BLK_M,k_tile*BLK_K:(k_tile+1)*BLK_K],dispatch="tma",cta_group=2,mbar=tma2mma_cta0.ptr_to([tma_phase.stage]))
                        Tx.copy_async(Asmem[i,1,:,:],A[m_st_c1:m_st_c1+BLK_M,k_tile*BLK_K:(k_tile+1)*BLK_K],dispatch="tma",cta_group=2,mbar=tma2mma_cta0.ptr_to([tma_phase.stage]))
                        Tx.copy_async(Bsmem[i,:,:],B[n_st:n_st+BLK_N,k_tile*BLK_K:(k_tile+1)*BLK_K],dispatch="tma",cta_group=2,mbar=tma2mma_cta0.ptr_to([tma_phase.stage]))

                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            for k in range(K_TILES):
                                mma2tma.wait(tma_phase.stage,tma_phase.phase)
                                tma_load(k//PIPE_DEPTH, k%PIPE_DEPTH)
                                if cbx == 0:
                                    tma2mma_cta0.arrive(tma_phase.stage, CTA_GROUP * (2 * BLK_M * BLK_K + BLK_N * BLK_K) * DTYPE_SIZE)
                                tma_phase.move_to_next_stage()
                            tile_scheduler.next_tile()

                elif warp_id < 2:
                    mma_phase = PipelineState("mma", PIPE_DEPTH)
                    mma_phase.init(is_producer=False)
                    ld_phase  = PipelineState("ld", 1)
                    ld_phase.init(is_producer=True)

                    # Per-consumer barrier stage: warp 0 uses stage 0, warp 1 uses stage 1.
                    @Tx.inline
                    def mma(k,i):
                        k_tile = k*PIPE_DEPTH+i
                        Tx.ptx.tcgen05.fence.after_thread_sync()
                        Tx.gemm_async(tmem[:,warp_id*MMA_N:warp_id*MMA_N+MMA_N], Asmem[i,warp_id,:,:], Bsmem[i,:,:],
                                                    accum=(k_tile!=0), dispatch="tcgen05", cta_group=2)
                    if cbx == 0:
                        with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                            while tile_scheduler.valid():
                                ld2mma.wait(warp_id,ld_phase.phase)
                                ld_phase.move_to_next_stage()

                                for k in range(K_TILES):
                                    tma2mma.wait(mma_phase.stage,mma_phase.phase)
                                    mma(k//PIPE_DEPTH, k%PIPE_DEPTH)
                                    mma2tma.arrive(mma_phase.stage, cta_group=2, cta_mask=3)
                                    mma_phase.move_to_next_stage()
                                mma2ld.arrive(warp_id, cta_group=2, cta_mask=3)

                                tile_scheduler.next_tile()

            elif wg_id < 2:
                wb_phase = PipelineState("wb", 1)
                wb_phase.init(is_producer=False)
                # Each WG consumes its own barrier stage (WG0↔warp0, WG1↔warp1).
                while tile_scheduler.valid():
                    mma2ld.wait(wg_id,wb_phase.phase)

                    wb_phase.move_to_next_stage()
                    Tx.ptx.tcgen05.fence.after_thread_sync()
                    reg_f16 = Tx.alloc_local((EPI_N,),d_type)
                    for i in Tx.unroll(MMA_N // EPI_N):
                        reg = Tx.alloc_local((EPI_N,),acc_type)
                        reg_wg = reg.view(128, EPI_N, layout=TileLayout(S[(128, EPI_N) : (1@axis_tid_in_wg, 1)]))

                        with Tx.warpgroup():
                            col_st = Tx.meta_var(wg_id * MMA_N + i * EPI_N)
                            col_end = Tx.meta_var(wg_id * MMA_N + i * EPI_N + EPI_N)
                            Tx.copy(reg_wg[:], tmem[:, col_st : col_end])
                        with Tx.thread():
                            Tx.cast(reg_f16[:],reg[:])

                        with Tx.thread():
                            Tx.copy(Dsmem[wg_id,warp_id*32+lane_id,:],reg_f16[:])
                            Tx.ptx.fence.proxy_async("shared::cta")
                            Tx.cuda.warpgroup_sync(wg_id + 10)

                        with Tx.thread(parent="warpgroup")[Tx.ptx.elect_sync()]:
                            m_st_epi = Tx.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M)
                            n_st_epi = Tx.meta_var(n_idx * MMA_N + i * EPI_N)
                            Tx.copy_async(D[m_st_epi : m_st_epi + BLK_M, n_st_epi : n_st_epi + EPI_N], Dsmem[wg_id, :, :], dispatch="tma")
                            Tx.ptx.cp_async.bulk.commit_group()
                            Tx.ptx.cp_async.bulk.wait_group(0)
                        Tx.cuda.warpgroup_sync(wg_id + 10)
                    ld2mma.arrive(wg_id, cta_id=0, pred=True)
                    tile_scheduler.next_tile()


            Tx.cuda.cluster_sync()
            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=2)


    return kernel

