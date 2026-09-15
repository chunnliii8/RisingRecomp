#include "vk_renderer.h"
#include "../cpu/fence_wait.h"

#include "pm4.h"
#include "pump_split.h"   // part 117: the draw context under the two-core pump
#include "pump_stats.h"
#include "shader_translator.h"
#include "drawid_ps_spv.h"
#include "null_ps_spv.h"
#include "rt_factor_spv.h"
#include "rt_shadow_spv.h"
#include "xenos.h"
#include "../kernel/xlive_overlay_glue.h"
#include "pit_gravel_tex.h"
#include "../host/bug_report.h"
#include "../host/host_paths.h"
#include "../host/settings.h"
#include "../host/window.h"
#include "../cpu/guest_thread.h"
#include "../cpu/thread_budget.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// <unistd.h> was here for one readlink("/proc/self/exe"); host/host_paths.h owns
// that now (release A.1) and nothing else in this file is POSIX.

// ===================================================================================
// THE INTERFACE THE TRANSLATED SHADERS PRESENT
// ===================================================================================
// This is not a design choice on our side â€” it is what XenosRecomp emits, and getting
// it wrong produces a device-lost or a black frame rather than a compile error. Read it
// out of the generated HLSL (`XenosRecomp/shader_common.h`) rather than from here if
// anything ever disagrees; this is a transcription.
//
// PUSH CONSTANTS (24 bytes, both stages): three uint64 GPU device addresses â€”
//   +0  VertexShaderConstants   256 float4  (the guest's ALU constants 0..255)
//   +8  PixelShaderConstants    224 float4  (the guest's ALU constants 256..479)
//   +16 SharedConstants         the block laid out below
//
// SHARED CONSTANTS. Every offset here appears verbatim in the generated shaders, which
// is the only reason to trust it:
//   +0   .. +63   Texture2D   descriptor index, one uint per sampler slot (16 slots)
//   +64  .. +127  Texture3D   descriptor index
//   +128 .. +191  TextureCube descriptor index
//   +192 .. +255  Sampler     descriptor index
//   +256          g_Booleans          (the 16+16 named bool scheme)
//   +260          g_SwappedTexcoords  (bit N: TEXCOORD N needs a YXWZ unswizzle)
//   +264          g_HalfPixelOffset   (float2)
//   +272          g_AlphaThreshold    (float)
//   +276          g_ParamGenMask      (uint)
//   +280          g_TessGrid          (uint)
//   +288 .. +351  Texture1D   descriptor index
//   +352          g_PosScale          (float2)
//   +360          g_PosOffset         (float2)
//   +384 .. +511  the 32 Xenos loop constants
//   +512 .. +543  the 256-bit Xenos bool constant file
//   +544 ..       per-fetch-slot dependent-vfetch table, 16 bytes each
//
// DESCRIPTOR SETS. One unbounded array each, matching the HLSL register spaces:
//   set 0 = Texture2D[]   set 1 = Texture3D[]   set 2 = TextureCube[]
//   set 3 = Sampler[]     set 4 = Texture1D[]
// ===================================================================================

namespace {

constexpr uint32_t kSharedTex2D = 0;
constexpr uint32_t kSharedTex3D = 64;
constexpr uint32_t kSharedTexCube = 128;
constexpr uint32_t kSharedSampler = 192;
constexpr uint32_t kSharedBooleans = 256;
constexpr uint32_t kSharedSwappedTexcoords = 260;
constexpr uint32_t kSharedHalfPixelOffset = 264;
constexpr uint32_t kSharedAlphaThreshold = 272;
constexpr uint32_t kSharedParamGenMask = 276;
constexpr uint32_t kSharedTessGrid = 280;
// ALPHA-TO-MASK, part 46. 1 when RB_COLORCONTROL bit 4 is set AND the guest surface is
// 4x MSAA â€” the only configuration in which our sample-per-sample dither is the right
// emulation. The last free dword before the 1D alias table at 288.
constexpr uint32_t kSharedAlphaToMask = 284;
constexpr uint32_t kSharedTex1D = 288;
constexpr uint32_t kSharedPosScale = 352;
constexpr uint32_t kSharedPosOffset = 360;
constexpr uint32_t kSharedLoopConstants = 384;
constexpr uint32_t kSharedBoolFile = 512;
constexpr uint32_t kSharedVfetchTable = 544;
// USER CLIP PLANES (part 57). Six float4 plane equations, dotted against the raw
// clip-space position by the vertex shader's XE_USER_CLIP_PLANES epilogue â€” Vulkan has
// no fixed-function clip planes, so the distance must be computed and exported as
// ClipDistance by the shader itself. The shared block is memset to zero every draw,
// and a zero plane dots to distance 0, which Vulkan KEEPS â€” so a draw with no planes
// enabled clips nothing by construction, and only the enabled planes are ever written.
constexpr uint32_t kSharedClipPlanes = 544 + 96 * 16;                 // 2080

// ROUTE (B)'s SCREEN-SPACE SHADOW FACTOR (part 65). Four words:
//   +0  uint  descriptor index of the factor image in the 2D bindless heap (0 = the
//             white dummy, which reads as LIT â€” the honest failure)
//   +4  uint  sampler index
//   +8  float 1 / factor-image width      SV_Position.xy * these two = the lookup uv
//   +12 float 1 / factor-image height
// Read by the helper tools/patch_rt_shadow_hlsl.py injects into the 126 shaders the
// census found sampling the cascade atlas. THE OFFSET IS DUPLICATED IN THAT TOOL
// (RT_BLOCK) and a disagreement does not fail â€” it reads a neighbouring word as a
// descriptor index, which is a valid index into the same heap, so the shader samples a
// real but wrong texture. Change both or neither.
constexpr uint32_t kSharedRtShadow = kSharedClipPlanes + 6 * 16;      // 2176
constexpr uint32_t kSharedSize = kSharedRtShadow + 16;                // 2192
// The stride of one shared block in the part-111 pre-zeroed sub-arena. 256 because that
// is `ArenaAlloc`'s default alignment and the address goes into a push constant the
// shaders reach with `vk::RawBufferLoad`; keeping the alignment identical to the old
// allocation means the arm and its control hand the shader the same shape of address.
constexpr uint32_t kSharedStride = (kSharedSize + 255u) & ~255u;      // 2304

// BOTH stages get 256 float4 registers, and the pixel shader's 256 is load-bearing.
//
// XenosRecomp's README documents the pixel shader window as 224 float4 (3584 bytes) and
// this file believed it. The generated shaders do not: the macro they emit is
// `pc(INDEX) = select(INDEX < 256, RawBufferLoad(PixelShaderConstants + min(INDEX,255)*16), 0)`,
// so a shader reading c255 loads from offset 4080 â€” 512 bytes past a 224-register
// buffer, i.e. into whatever this arena allocated next.
//
// Case Zero's scene pixel shaders read c255 in their FINAL instructions, as the
// tone-map's scale and bias:
//     mul  r0.xyz, r0.xyz, c255.wwww
//     mad  r0.xyz, r0.xyz, c14.wwww, c255.xxxx
//     max  r0.xyz, r0.xyz, c255.zzzz
//     mul  r0.xyz, r0.xyz, c255.yyyy
// so a wrong c255 does not tint the scene â€” it collapses every pixel to a constant.
// That is what "930 draws producing three distinct colours" was.
//
// The guest states the true size itself and it is not 224: SQ_PS_CONST reads
// base=256 size=255, i.e. ALU float4 registers 256..511, which is 256 registers.
// Sizing a constant buffer from a tool's documentation rather than from the guest's
// own register is the whole mistake.
constexpr uint32_t kVsConstBytes = 256 * 16;
constexpr uint32_t kPsConstBytes = 256 * 16;

// The bindless heap's size, and the number this renderer ran out of.
//
// It was 4096 with the comment "the frontend uses a few dozen", which was true of
// every screen this port could reach at the time and stopped being true the moment it
// reached Still Creek. Slots are handed out monotonically and never recycled
// (`entry.slot = R->nextTextureSlot++`), so a long session in a texture-rich area
// fills the heap and every texture after that is served slot 0 â€” the 1x1 WHITE dummy
// â€” with only a counter to say so. `R->nextTextureSlot` read exactly 4096 out of a
// live game via `gdb -p ... print`, with white buildings, a white NPC, white road
// decals, white blood and white inventory icons on screen.
//
// The rule the operator's pictures establish is worth keeping, because it is not the
// obvious one: it is NOT "things that appear late go white", it is "anything needing
// a NEW SLOT after the heap filled goes white". This title streams textures BY
// DISTANCE, so walking toward a building requests a higher-resolution version â€” a new
// fetch constant, a new cache entry, a new slot â€” which is why EVERY building whitens
// on approach while its distant version stays correct.
//
// THE REAL FIX IS NOW IN (imported from Case West c24176f, 2026-09-05, where their
// operator hit all 65536 slots on a release completion run): ReclaimTextureSlot is
// the LRU the rest of this comment called for â€” when the heap fills, the least-
// recently-used slot that no in-flight frame can still reference is evicted (its
// image retired behind the same fence window as RetiredImage) and handed to the new
// texture, instead of serving the 1x1 white dummy forever. Case West validated it at
// CW_VK_MAX_TEXTURES=256 under sync validation: 133 recycles, 0 hazards. This cap is
// now a VRAM ceiling / churn knob, not a correctness limit; CZ_VK_NO_TEX_LRU=1
// restores the old white-on-full behaviour as the same-binary control arm.
//
// Sized from the DEVICE rather than from a new magic number, clamped to something
// sane, because "how many sampled images may a shader see" is a property of the host
// and not something to guess twice.
uint32_t g_maxDescriptors = 4096;   // replaced at init from the device's own limit

// --- diagnostics --------------------------------------------------------------------
// Every path that declines to do something increments one of these. The alternative â€”
// returning quietly â€” is what makes a renderer that draws 80% of a frame look exactly
// like one that draws all of it, and this project has already paid for that lesson in
// the command processor (gotcha 84: a parser that stops early must say so).
std::map<std::string, uint64_t> g_stats;
void Count(const char* name) { ++g_stats[name]; }

// ...and the same counters, reached without paying for them, for the handful of call
// sites that run on EVERY draw.
//
// `Count` constructs a `std::string` from the literal (a heap allocation for any name
// over 15 characters, and most of ours are) and walks a red-black tree comparing
// strings, per call. That is fine at a few hundred calls and it is not fine at the
// ~5 unconditional counters a draw times 6,600 draws a frame: it is instrumentation
// overhead inside the phase being instrumented, which is the same defect that made
// part 18's state cache first measure as a dead heat (`docs/perf-cpu-plan.md` Â§1a,
// hypothesis B â€” the one item there filed as needing no measurement to justify).
//
// A `std::map` node is stable for the life of the map and `g_stats` is never cleared,
// so the address of a counter can be resolved ONCE per call site and then incremented
// directly. `COUNT(literal)` does exactly that with a function-local static; the
// printing interface, the names and the ordering are untouched, so nothing downstream
// of `VkRenderer_DumpStats` can tell the difference.
//
// `Count` stays, and is still the right thing for every path that declines a draw:
// those run rarely, and a cold call site is not worth a static.
//
// HOW TO FIND THE SITES THAT ARE WORTH CONVERTING, because part 52 got this wrong from
// the code and right from the data. `perf` put `std::map<std::string, uint64_t>::
// operator[]` at 2.30% of the pump thread, and `perf-plan-part52.md` Â§4 priced the fix
// as "28 `Count(` sites inside `DoDraw`'s body" â€” counted by reading the source. The
// counters' OWN DUMP says something different: of ~62.5 M plain-`Count` calls in a
// 200-second outdoor run, **52.9 M â€” 84.6% â€” are the single site in
// `VkRenderer_Draw`**, and ten sites are 99.2% of the total. Most of the 28 in `DoDraw`
// are decline paths that fire a few hundred times an hour.
//
// So the rule is: `VkRenderer_DumpStats` already prints the call count of every counter,
// which is the exact statistic that ranks these sites, and reading the source instead
// ranks them by how alarming they look. Sort the dump before converting anything
// (`docs/phase5-notes.md` Â§6ci).
uint64_t* CounterSlot(const char* name) { return &g_stats[name]; }

// Is anything going to READ `Renderer::snapshotsSampledThisPass` this run? Three
// instruments do, and all three are env-rooted: `CZ_VK_PSBIND`, `CZ_VK_DRAW_CENSUS`
// (armed by F9, but only ever when the variable is set) and `CZ_VK_RESOLVE_TRACE`.
// Maintaining the list costs a linear scan per snapshot fetch â€” ~2,070 a frame â€” so it
// is worth asking the question once instead of paying for the answer 8.2 M times a
// session. Defined here rather than as a local static because the readers live several
// thousand lines apart and must agree; set once in VkRenderer_Init.
bool g_passInputsWanted = false;

// CZ_VK_COPY_CENSUS=1 â€” the resolve-copy produced-vs-sampled census (part 90 item 2,
// perf-plan-part90.md Â§2). The question part 80 Â§1 item 5 left unasked: how many of
// the 50.4 resolve copies a frame (0.741 ms of GPU) are DEAD â€” their destination
// region overwritten by the next copy of the same (snapshot, rect) before anything
// consumed it? Consumption is marked at every snapshot reader: the draw-path fetch
// (BEFORE the right-sized-view early return, which the pass-inputs list misses), the
// present, the frame-stats surface, and the cube-face assembly. The count is
// CONSERVATIVE toward "live": a sample of the snapshot marks all its rects even
// though it may have read only one, so a large dead share is real. A DIAGNOSTIC ARM,
// off by default; one bool test per site when off.
bool g_copyCensusOn = false;
struct CcRect
{
    int32_t x, y;
    uint32_t w, h;
    bool sampled;
    uint64_t px;
    uint64_t deadN, deadPx;   // cumulative, for the per-key verdict rows
};
std::unordered_map<uint32_t, std::vector<CcRect>> g_ccMap;
uint64_t g_ccCopies = 0, g_ccDead = 0, g_ccPixels = 0, g_ccDeadPixels = 0;
uint64_t g_ccSampleMarks = 0;

void CopyCensusCopy(uint32_t key, int32_t x, int32_t y, uint32_t w, uint32_t h,
                    uint64_t px)
{
    auto& rects = g_ccMap[key];
    ++g_ccCopies;
    g_ccPixels += px;
    for (auto& r : rects)
        if (r.x == x && r.y == y && r.w == w && r.h == h)
        {
            if (!r.sampled)
            {
                ++g_ccDead;
                g_ccDeadPixels += r.px;
                ++r.deadN;
                r.deadPx += r.px;
            }
            r.sampled = false;
            r.px = px;
            return;
        }
    rects.push_back(CcRect{ x, y, w, h, false, px, 0, 0 });
}

void CopyCensusSampled(uint32_t key)
{
    auto it = g_ccMap.find(key);
    if (it == g_ccMap.end())
        return;
    ++g_ccSampleMarks;
    for (auto& r : it->second)
        r.sampled = true;
}
#define COUNT(lit)                                                                     \
    do                                                                                 \
    {                                                                                  \
        static uint64_t* _czSlot = CounterSlot(lit);                                   \
        ++*_czSlot;                                                                    \
    } while (0)

// --- CZ_VK_PROFILE=N â€” where a FRAME's CPU time actually goes ------------------------
//
// Every frame-rate number this project owned before today divided a whole run's frames
// by its wall time, and every one of them was taken at the TITLE SCREEN. "Gameplay runs
// at 8-12 fps" was an operator's stopwatch, and the three suspects on the board (the
// synchronous submit, the per-frame readback, the per-draw constant upload) had never
// been separated â€” so any work on them would have been optimising whichever one came to
// mind first. This splits the frame into named phases and prints milliseconds.
//
// The phases are CPU wall time on the thread that records the frame, which is the right
// quantity while the renderer is CPU-bound and would be the wrong one if it were not.
// `submit` is the honest check on that: it is the wait for the GPU to finish, so a
// frame whose time is in `submit` is GPU-bound and the rest of this table is noise.
//
// It costs one clock read per phase entry and exit. At ~2,000 draws a frame that is a
// few thousand vDSO reads, well under a millisecond of a ~100 ms frame â€” but it is off
// by default anyway, because an instrument expensive enough to change the thing it
// measures reports its own overhead (gotcha 7).
struct ProfilePhases
{
    uint64_t constants = 0;   // the per-draw ALU constant copy into mapped memory
    // ...and `constants` SPLIT FIVE WAYS (part 75), for the same reason `record` and
    // `drawOther` were split before it. Part 75's re-baseline made it the single biggest
    // term in a crowd frame by a wide margin â€” **7.96 ms of an 18.64 ms frame at
    // 5,000-7,000 draws, 10.88 of 25.68 at 7,000-9,000** â€” i.e. roughly half of everything
    // the sixteen phases account for. A number that large with no breakdown is not an
    // item, it is a place to start guessing, and the scope holds four unrelated things:
    // the vertex gather, the pixel gather, the projection patches, and a 2,192-byte
    // memset of the shared block that runs on EVERY draw whether the memo hit or not.
    //
    // The vertex half then turned out to be all of it, so it is split AGAIN â€” and the
    // answer is the PATCH, not the copy everyone would have looked at first.
    //
    // EXCLUSIVE of each other, like every scope here, so `constants` keeps its meaning as
    // the residual: the memo bookkeeping, the exposure min/max and the counters.
    uint64_t constVs = 0;      // the VERTEX window: the RESIDUAL after the two below
    uint64_t constVsCopy = 0;  // ...its CopyConstWindow gather
    uint64_t constVsPatch = 0; // ...its fov/wide projection patch, which READS BACK
    uint64_t constPs = 0;      // the PIXEL window's gather
    uint64_t constShared = 0;  // memset(shared, 0, kSharedSize) â€” every draw, memo or not
    uint64_t streams = 0;     // vertex/index stream copy + dword swap
    uint64_t textures = 0;    // texture untile + upload
    uint64_t record = 0;      // the vkCmd calls of a draw
    // ...and `record` SPLIT THREE WAYS (part 47). It is 15.2 ms of the operator's
    // 42.8 ms frame â€” 2.17 microseconds per draw, against ~1.3 on the headless route â€”
    // and it was completely uninstrumented inside, which is precisely the state the PM4
    // walk was in when "the walk is 11 ms" supported no hypothesis about what to change.
    // A number with no breakdown is not an item, it is a place to start guessing.
    //
    // These are EXCLUSIVE of each other and subtract from `record`, like every other
    // scope here, so `record` keeps its meaning as the residual: the shared-constant
    // writes, the A2M block and the draw fingerprint.
    uint64_t recordState = 0;    // pipeline / viewport / scissor / blend / sets / push
    uint64_t recordVertex = 0;   // the vertex-stream walk and its binds
    uint64_t recordIndex = 0;    // the index setup, its bind, and the vkCmdDraw* itself
    // THE STREAM CONTENT GUARD, split out of `record` in part 52 to price plan item 1.4.
    //
    // It was never in `streams`: `ProfScope(streams)` deliberately wraps only the
    // `CopySwapped`, so a cross-frame HIT costs the `streams` column nothing â€” which is
    // exactly the design, and exactly why `streams` reads 0.02 ms while `GuardFold` is
    // the biggest symbol in the pump. The hash was therefore being charged to whichever
    // scope encloses `UploadStream`, and that is `recordVertex` and `recordIndex`.
    //
    // That matters for more than tidiness. Item 1.4 (parallel command RECORDING) is
    // priced off `record`, and item 1.1 (parallel content GUARDS) is priced off
    // `GuardFold` â€” and without this split the same milliseconds were counted in both.
    // A profiler phase names a SCOPE, not a subsystem.
    uint64_t streamGuard = 0;
    uint64_t submit = 0;      // vkQueueSubmit + the fence wait (i.e. the GPU)
    // ...and that split in two, because they are different subsystems wearing one
    // number. `submitCall` is the driver translating a command buffer of ~1,900 draws
    // on THIS cpu; `fenceWait` is the GPU actually executing it. 24 ms for that many
    // small draws on an RTX 3070 is far too slow to be fill, so which half it is
    // decides whether the next question is about barriers or about the host.
    uint64_t submitCall = 0;
    uint64_t fenceWait = 0;
    uint64_t readback = 0;    // image -> host buffer -> window
    // CZ_VK_FRAME_STATS's OWN COST, and it exists because part 50 wrote the rule and
    // then stopped one instrument short of applying it. Â§6cg Â§6: "an instrument that
    // can only be read through another instrument cannot measure that one" â€” said of
    // `CZ_VK_PROFILE`, whose 2-4 ms bill was found by reading `CZ_VK_FRAME_STATS`
    // instead. Nobody then asked what THAT one costs, and it is not small: per
    // PRESENTED frame it zeroes a 2 MB bitmap and walks all 921,600 pixels with a
    // random-access bit test. Every performance number in this project since part 18 â€”
    // including the operator's whole-map lap and part 50's own profiler A/B â€” was
    // recorded with it enabled, so it is a floor under every frame time ever quoted
    // here. EXCLUSIVE, like every scope; it is not part of `readback`.
    uint64_t frameStats = 0;
    // DoDraw's own untimed work â€” register decode, the pipeline-key build and its
    // lookup, the fetch-constant walk, and the always-on censuses. EXCLUSIVE of every
    // phase above, which it was not until part 20; see the ProfScope comment.
    uint64_t drawOther = 0;
    // ...and `drawOther` SPLIT THREE WAYS (part 48), for exactly the reason `record` was
    // split in part 47: it is 4.19 ms of the operator's frame and it was four different
    // things wearing one number, one of which had been named a suspect for two parts
    // without anyone being able to price it. The part-47 split is what found the stream
    // guard â€” 81.65 MB hashed in one frame, charged to a phase whose name did not
    // mention it (gotchas 325, 326) â€” so this is the same move, one phase over.
    //
    // EXCLUSIVE of each other and of the named phases nested inside them, like every
    // scope here, so `drawOther` keeps its meaning as the residual.
    uint64_t otherKey = 0;      // register decode + the PipelineKey build + the censuses
    uint64_t otherPipeline = 0; // the std::map<PipelineKey, VkPipeline> probe (and, on a
                                // miss, the creation `pipelineNs` reports separately)
    uint64_t otherFetch = 0;    // the texture/sampler fetch-constant walk and its binds,
                                // minus `textures` (UploadTexture has its own scope)
    // ...and the RESIDUAL split again, because the first split said the residual was the
    // largest part of `other` (45%, 329 ns/draw) and a residual names nothing. Gotcha
    // 327: splitting a phase has now found three items in two parts and reading the code
    // has found none, so the answer to "what is in the residual" is another split.
    uint64_t otherShader = 0;   // the two shader-hash lookups and the draw's early guards
    uint64_t otherBegin = 0;    // BeginFrame + BeginRendering + the three arena allocs
    uint64_t otherTail = 0;     // bool/loop constants, the viewport decode, the censuses
    uint64_t draws = 0;       // how many draws those numbers are spread over
    // RT stage 2 (part 64): the whole ray-traced-shadow path â€” BLAS/TLAS builds
    // recorded and the trace pass â€” measured from day one per the plan's rule.
    // EXCLUSIVE like every scope; zero in any run at tier OG.
    uint64_t rt = 0;
    uint64_t scopes = 0;      // ProfScope closes â€” the profiler's own bill, see ProfScope
    // Pipeline creation, which lives INSIDE `drawOther` and is the only thing in there
    // that costs milliseconds. Separated because a first-visit stutter and a per-draw
    // overhead are different defects that were sharing one column. See GetPipeline.
    uint64_t pipelineNs = 0;
    uint64_t pipelinesCreated = 0;
};
ProfilePhases g_prof;
bool g_profileOn = false;

inline uint64_t ProfNow()
{
    return g_profileOn ? uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count())
                       : 0;
}
// THE SAME CLOCK, ALWAYS READ. `ProfNow` returns 0 unless `CZ_VK_PROFILE` is armed, and
// the profiler costs 2-4 ms a frame â€” so anything it can measure is invisible in the runs
// that matter, which are the operator's uninstrumented soaks. This is for the handful of
// events a run has that are worth timing unconditionally because they happen ~500 times
// in a whole session rather than 33,000 times a frame (part 71: pipeline creation).
inline uint64_t NowNs()
{
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count());
}
// --- WHAT THE PROFILER ITSELF COSTS, and why that is a phase's worth of nanoseconds ----
//
// `docs/perf-plan-part50.md` Â§4 calls `other`'s residual â€” 206 ns/draw, unnamed after two
// splits â€” "the highest-yield-per-hour item in the document". Before splitting it a third
// time, there is a candidate that no split could ever name, because it is not IN any of
// the code being split: **the clock reads this profiler performs.**
//
// Every `ProfScope` reads the clock twice, and DoDraw opens a dozen of them. Worse, the
// exclusivity accounting puts that overhead somewhere very specific. A child's measured
// interval starts AFTER its constructor's `ProfNow()` has returned and ends AT its
// `Close()`'s read, so the cost of both reads falls outside the child's interval and
// inside its parent's â€” and it is not subtracted, because `childNs` only ever receives
// the child's own measured total. `drawOther` is the outermost per-draw scope. So **every
// clock read every nested scope in DoDraw performs lands in `other`'s residual, and
// nowhere else.**
//
// If that is what the residual is, it is not an item at all: it is absent from any run
// without `CZ_VK_PROFILE`, so "removing" it would save a frame time nobody is paying.
// This is gotcha 7 â€” a probe expensive enough to distort what it reports â€” in the one
// place it is hardest to see, because the probe is the thing doing the reporting.
//
// So the cost is MEASURED rather than argued: `CalibrateProfNow` times the call, the
// scope count per draw is counted, and the profile print multiplies the two and puts the
// product next to the residual it is meant to explain. A prediction that can be wrong.
uint64_t g_profNowNs10 = 0;   // ns per ProfNow() call, x10, so a sub-ns cost is visible
uint32_t g_extraScopes = 0;   // CZ_VK_PROFILE_EXTRA_SCOPES â€” the control; see DoDraw

// Scoped accumulator, EXCLUSIVE of any scope nested inside it. Compiles to nothing
// measurable when the profile is off, because every call short-circuits on one
// already-hot bool.
//
// THE EXCLUSIVITY IS A PART-20 BUG FIX, and it silently re-ordered this project's own
// performance plan. These scopes nest: `record` opens partway down DoDraw and lives to
// the end of it, so the three `UploadStream` calls below it ran INSIDE `record` and
// their `streams` time was counted twice â€” once in `streams` and once in `record`.
// `submit` encloses `submitCall` and `fenceWait` the same way. The print then computed
// DoDraw's residual as `drawTotal - (constants + streams + textures + record)`, which
// subtracts `streams` twice, so the residual came out that much too small.
//
// The arithmetic is not subtle once written down, but the effect was: it made `record`
// look like the draw path's dominant term and the residual look like its smallest, and
// `docs/perf-cpu-plan.md` was written on those numbers â€” filing the residual as "the
// cheapest item in this document".
//
// The mechanism is worth stating generally, because a profiler is exactly the kind of
// code whose defects are invisible: a nested scope moves time from the OUTER scope's
// residual into the inner scope's name, and both columns still add up to the same
// total. Nothing looks wrong. The fix is for every scope to subtract what its children
// took, which makes each column mean "time in THIS phase and no other" â€” and then the
// columns and the total become independent statements that can disagree.
//
// `g_profileOn` is set once at init, before any draw, and never changes; the ctor and
// dtor would otherwise have to agree about a flag that moved between them.
struct ProfScope
{
    uint64_t* sink;
    uint64_t t0 = 0;
    uint64_t childNs = 0;   // what scopes opened inside this one consumed
    ProfScope* parent = nullptr;
    static thread_local ProfScope* current;

    explicit ProfScope(uint64_t* s) : sink(s)
    {
        if (!g_profileOn)
            return;
        t0 = ProfNow();
        parent = current;
        current = this;
    }
    // Stop accounting NOW rather than at the closing brace. Needed where a phase
    // boundary does not coincide with a scope boundary â€” part 47 split `record` into
    // three, and its vertex section is not braced, so bracing it would have moved a
    // dozen locals that the index section reads. Idempotent, and the destructor becomes
    // a no-op afterwards, so a Close() plus the normal scope exit cannot double-count.
    void Close()
    {
        if (!g_profileOn || !sink)
            return;
        const uint64_t total = ProfNow() - t0;
        *sink += total - childNs;
        if (parent)
            parent->childNs += total;
        current = parent;
        sink = nullptr;
        ++g_prof.scopes;   // one add against this scope's two clock reads; see above
    }
    // The check inlined, the body not: with the profiler OFF â€” every shipped run â€” the
    // destructor was still a call per scope, ~10 scopes a draw, 0.5% of the pump
    // (part 117's profile). Now it is one hot-bool test.
    ~ProfScope()
    {
        if (__builtin_expect(g_profileOn, 0))
            Close();
    }
};
thread_local ProfScope* ProfScope::current = nullptr;

// Time `ProfNow()`. Called once, before the first draw, with the profile already on so
// the calibrated call is the same call the scopes make â€” including its `g_profileOn`
// test, which is part of what a scope pays.
//
// The MINIMUM of many batches, not the mean: this runs while the rest of the process is
// starting up, so a batch can be interrupted by anything, and every such interruption can
// only make a batch slower. The floor is the number that describes the call.
void CalibrateProfNow()
{
    uint64_t best = ~0ull;
    volatile uint64_t sink = 0;
    for (int batch = 0; batch < 32; ++batch)
    {
        const uint64_t t0 = ProfNow();
        for (int i = 0; i < 1000; ++i)
            sink += ProfNow();
        const uint64_t t1 = ProfNow();
        best = std::min(best, t1 - t0);
    }
    g_profNowNs10 = (best + 50) / 100;   // best is ns for 1000 calls => x10 per call
    fprintf(stderr,
            "[vkprof] a clock read costs %.1f ns; every ProfScope makes TWO and their "
            "cost lands in the residual of the scope AROUND them (see g_profNowNs10)\n",
            double(g_profNowNs10) / 10.0);
}

#if CZ_WHOLEFUNC
// --- WHOLE-FUNCTION SAMPLED TIMERS (part 110, item A.2) ------------------------------
//
// THE DEFECT THEY ANSWER. A `ProfScope` measures a REGION OF CODE, and this renderer's
// regions do not cover the functions they are named after. `streams` reads 0.3% of the
// frame while the SYMBOL `UploadStream` is 9.4% of the pump thread and 13.1% with
// `PersistFind` â€” a factor of thirty, and it has now misled three separate parts about
// the same function (22 closed the stream cache on it, 55 re-opened it from a `perf`
// symbol profile, 109 read it and wrote "almost certainly dead on price" four hours
// before its own symbol profile said otherwise). Gotcha 343 was written about exactly
// this and the table went on saying it.
//
// WHY NOT JUST SCOPE THE WHOLE FUNCTION. `UploadStream` runs ~46,000 times a crowd frame
// and a `ProfScope` is two clock reads at ~21 ns, so a per-call scope is ~2 ms a frame â€”
// larger than several of the phases it would be separating. An instrument that big does
// not measure the function, it replaces it (gotchas 7, 223). So: time one call in
// `kWfPeriod` and scale.
//
// THE PERIOD IS 17 AND THAT IS DELIBERATE. Part 89's resolve-split census samples every
// 16th DRAW, which is fine for a per-draw mean but is a power of two sitting on top of a
// renderer whose work batches in powers of two â€” a sampler aligned with the thing it
// samples measures a phase of it rather than a mean of it. A prime period cannot align
// with any power-of-two batching, and it is sampled on a per-CALL counter rather than a
// per-draw one, because one draw's streams are not interchangeable with another's.
//
// THEY ARE INCLUSIVE OF CALLEES, unlike every `ProfScope` here, and that is the point:
// the question is what a SUBSYSTEM costs, so `UploadStream`'s number includes
// `PersistFind` and the guard, and `DoDraw`'s includes all of it. **That means they must
// be compared with a `perf` symbol GROUP and never with one symbol's self time** â€”
// `tools/phase_vs_perf.py` computes exactly those groups.
//
// **IT IS NOT FREE WHEN OFF, AND THAT IS A MEASUREMENT, NOT A CONCESSION.** This was
// written expecting one predictable branch on an already-hot global â€” tested before the
// counter is even incremented â€” and its own pre-registered identity gate refuted that:
// three runs an arm, both binaries alternated in one session, matched draw bands,
// **+0.65 / +0.67 / +0.43 / +0.29 / +0.42 ms of pump CPU** in every band with real
// sample counts, against a bar of +-0.10 (part 110 Â§6.8). Roughly 6 ns per call across
// ~78,000 calls a frame, which is twenty times what a predicted branch costs.
//
// The mechanism is the RAII object, not the branch: a non-trivial destructor on
// `UploadTexture` â€” a wrapper whose body is a tail call â€” forces a real call and a stack
// frame, and on `DoDraw` it puts one on every early return. **A probe changes codegen
// even when its body never runs**, so "free when off" has to be MEASURED and cannot be
// argued from the source.
//
// So the three call sites are behind `-DCZ_WHOLEFUNC=1` and a default build carries no
// code at all. `CZ_VK_NO_WHOLEFUNC=1` is the runtime control INSIDE such a build; it
// cannot refund the 0.5 ms, which is why the build announces itself and says to read its
// shares rather than its milliseconds.
//
// SINGLE-THREADED BY CONSTRUCTION: all three functions run on the pump thread only (the
// parallel-record workers replay captured commands and call none of them), so these are
// plain adds and not atomics. If any of them is ever moved to a worker, this comment is
// the thing that has to change first.
struct WholeFunc
{
    uint64_t ns = 0;        // summed over the SAMPLED calls only
    uint64_t sampled = 0;   // how many calls were timed
    uint64_t calls = 0;     // how many calls happened while armed
};
WholeFunc g_wfStream, g_wfTexture, g_wfDraw;
bool g_wholeFunc = false;
constexpr uint64_t kWfPeriod = 17;

struct WfScope
{
    WholeFunc* w = nullptr;
    uint64_t t0 = 0;
    explicit WfScope(WholeFunc* wf)
    {
        if (!g_wholeFunc)
            return;
        if (++wf->calls % kWfPeriod)
            return;
        w = wf;
        t0 = NowNs();
    }
    ~WfScope()
    {
        if (!w)
            return;
        w->ns += NowNs() - t0;
        ++w->sampled;
    }
};

#endif // CZ_WHOLEFUNC

// CZ_VK_TEX_CENSUS=1 â€” per texture ADDRESS, where its pixels came from.
//
// The aggregate counters above say how many fetches took each path; they cannot say
// WHICH surface took which, and that is the whole question behind a black rectangle on
// screen. A surface our renderer resolved to and then served from guest memory is
// serving pixels nobody ever wrote there (gotcha 113: a resolve becomes a host image,
// not guest bytes), and the symptom is a filled black quad four layers away. The
// `zero` column is the one that matters: an upload whose every byte is zero is this
// runtime saying out loud that it had nothing to give.
//
// Gated on the env var because the `snapshot` column is hit ~500,000 times a run and a
// probe expensive enough to change the frame rate manufactures what it reports
// (gotcha 7).
struct TexSource
{
    uint32_t width = 0, height = 0, format = 0;
    uint64_t uploads = 0, zeroUploads = 0, fromSnapshot = 0, snapshotTooOld = 0;
    uint64_t maxAge = 0;
    bool everResolved = false;
    // Enough to re-read the same bytes at report time. "This upload was black" and
    // "this upload was black AND the guest has filled it in since" are completely
    // different defects â€” the first says the data was never there, the second says we
    // cached a texture that arrived late â€” and only a re-read separates them.
    const uint8_t* src = nullptr;
    uint64_t srcBytes = 0;
};
bool g_texCensus = false;

// --- CZ_VK_TEX_GUARD / CZ_VK_TEX_REVALIDATE ------------------------------------------
//
// THE OPERATOR'S REPORT THIS EXISTS FOR: "almost all the textures in the game are wrong
// and got the texture of something else â€” a building getting the repeated texture of a
// moose head item". That is not a scaling defect and not the bindless heap running out
// (which serves the white dummy); it is object A being drawn with object B's PIXELS,
// which means the cache handed out an image that no longer belongs to that fetch.
//
// The mechanism the code already half-admits, in the CZ_VK_TEX_REFRESH comment: the
// texture cache is keyed on the fetch constant's six dwords and is NEVER INVALIDATED.
// Those dwords are a descriptor â€” address, extent, format, tiling, swizzle, pitch â€” and
// this title STREAMS ITS TEXTURES BY DISTANCE (open-items 3z), so it is constantly
// loading a new texture into a heap address a previous one has been freed from. When
// the new occupant has the same extent and format as the old one, every dword matches,
// the key matches, and the draw gets the old image forever.
//
// `CZ_VK_TEX_REFRESH` was built for one instance of this (a font atlas the CPU keeps
// writing) and it takes an ADDRESS, so it could only ever answer about a texture
// somebody had already identified. The question here is a census over all of them.
//
// GUARD is the measurement: on a cache HIT, hash a bounded sample of the guest bytes the
// entry was uploaded from and compare it with the hash taken at upload. A mismatch is
// the cache serving pixels the guest has since replaced. Counted globally and per
// address, so the answer is a number and a list rather than an impression.
//
// REVALIDATE is the repair the measurement would justify: on a mismatch, re-upload into
// the SAME image and the SAME bindless slot â€” which is exact and allocation-free,
// because the dimensions are part of the key. Kept behind its own switch and OFF until
// the census says the mismatch is real, because a per-fetch re-upload is expensive and
// "it looked better" is not a reason to ship one.
//
// POISON is the control (gotcha 30). It folds the frame number into the guard so every
// hit MUST mismatch: a census that cannot report 100% has not been shown capable of
// reporting anything, and this project has shipped a comparison that only ever read
// 100% before (gotcha 234) â€” so both ends need exercising.
bool g_texGuard = false;
bool g_texRevalidate = false;
bool g_texGuardPoison = false;

// --- GOLDEN TEXTURE STORE (part 94) --------------------------------------------------
// This title streams a shared detail texture (the gas-station rooftop's gravel floor,
// `0E522000` 32x32 DXT1) into a recycled heap address whose bytes are PRESENT only during
// a brief streaming window: we upload them to the GPU when we happen to catch them, then
// the guest frees the address back to zero (a live read at a rooftop where the gravel
// renders CORRECTLY finds `0E522000` all-zero in every address space â€” the picture comes
// from our cached image, not from memory). Whichever surface we upload for FIRST wins: a
// rooftop reached while the bytes are live caches real gravel; the recessed pit, reached
// while they are zero, caches an opaque-black BC1 block and â€” since the bytes never change
// again â€” the content guard never re-fires and it stays black forever (Â§6aa's shape, now
// explained). The two share the fetch descriptor, so they are the same texture.
//
// The store closes that gap without guessing at pixels: the FIRST time a signature uploads
// non-zero, keep the decoded bytes; any later all-zero upload of the same signature is
// served those bytes instead of black, and a cache entry already frozen black is refreshed
// from them on its next hit. It can only ever replace an all-zero (black) upload, so it
// cannot change any surface that renders today. Bounded to small textures (the streamed
// detail maps this class appears in) so the memory cost is negligible.
// CZ_VK_NO_GOLDEN_TEX=1 is the same-binary control arm (brings the black pit back).
bool g_noGolden = false;
constexpr size_t kGoldenTexCap = 64 * 1024;   // only remember detail-sized textures
std::unordered_map<uint64_t, std::vector<uint8_t>> g_goldenTex;
uint64_t g_goldenStored = 0, g_goldenServed = 0;
inline uint64_t GoldenSig(uint32_t address, uint32_t w, uint32_t h, uint32_t fmt)
{
    return (uint64_t(address & 0x1FFFFFFFu) << 32) ^ (uint64_t(w) << 20)
           ^ (uint64_t(h) << 8) ^ uint64_t(fmt);
}
// The gravel is present in memory only during a streaming race we may lose in any given
// session (Â§6aa), so an in-memory store alone would fix the pit only in the lucky sessions
// that happened to catch the bytes. Persist each captured signature to disk and preload it,
// so once ANY session catches the real texture the surface is correct in every session
// after â€” and a shipped build can carry a pre-warmed set. Files are <sig>.bin under the
// same cache root as the pipeline cache; the filename IS the key.
std::filesystem::path GoldenDir()
{
    std::error_code ec;
    std::filesystem::path base;
    if (const char* x = getenv("CZ_GOLDEN_DIR"); x && *x)
        base = x;
#if defined(_WIN32)
    else if (const char* la = getenv("LOCALAPPDATA"); la && *la)
        base = std::filesystem::path(la) / "cz-recomp";
#else
    // XDG_CACHE_HOME first, like the pipeline cache beside it â€” until part 102 this
    // read HOME/.cache directly, so a "fresh machine" run that parked XDG_CACHE_HOME
    // still preloaded 29,000 golden files from the real cache and persisted nothing:
    // the Linux control arm for the czamd stutter was silently warm.
    else if (const char* x = getenv("XDG_CACHE_HOME"); x && *x)
        base = std::filesystem::path(x) / "cz-recomp";
    else if (const char* h = getenv("HOME"); h && *h)
        base = std::filesystem::path(h) / ".cache" / "cz-recomp";
#endif
    else
        base = std::filesystem::temp_directory_path(ec) / "cz-recomp";
    base /= "golden";
    std::filesystem::create_directories(base, ec);
    return base;
}
// --- THE PACK (part 104 item 2) ---------------------------------------------------------
// The per-file store's boot walk was the cost, measured before it was touched: one
// open+read per file on the boot path before the first frame â€” czamd 5,923-5,958 files in
// 1,048-1,095 ms (176-185 us a file on NTFS behind a real-time scanner), the dev box
// 29,932 in 1,290 ms â€” and growing with every session that captured anything.
// `golden.pack` is the same bytes as ONE file: an 8-byte header, then
// {signature u64, length u32, bytes} entries to end-of-file, read back with one read at
// boot and APPENDED to as textures are captured (the writer thread keeps it open and
// flushes after every entry).
//
// Append-only rather than "rewritten from the map at exit", for two reasons that are the
// same reason: a platform that ends the process with TerminateProcess never runs the exit
// path (czamd's harness, gotcha 519 â€” and a crash anywhere is the same case), and a clean
// exit would otherwise rewrite the whole store every time. An append survives a kill up to
// its last flushed entry, and the one thing a kill can leave â€” a torn tail entry â€” is
// detected at load by its own length field and cut off before the next append.
//
// Loose <sig>.bin files (an install that predates the pack, or a session under the
// per-file arm) are folded in on the next boot: read into the map as before, then handed
// to the writer, which appends each to the pack and DELETES the file. So the directory is
// walked once more and then it is empty and the walk is free. A store that grew two ways
// forever would carry both costs.
//
// CZ_VK_NO_GOLDEN_PACK=1 is the same-binary control arm: the per-file store exactly as it
// was â€” the walk at boot, one file per capture, the pack ignored. It is what the preload
// time is measured against, and it is also how a pack a reader distrusts can be bypassed.
constexpr uint32_t kGoldenPackMagic   = 0x5047435Au; // "ZCGP"
constexpr uint32_t kGoldenPackVersion = 1;
bool g_noGoldenPack = false;
size_t   g_goldenPackEntries = 0;  // entries the pack held at load
uint64_t g_goldenPackBytes   = 0;  // and its byte length after the torn-tail cut
std::filesystem::path GoldenPackPath() { return GoldenDir() / "golden.pack"; }

// THE WRITE IS OFF THE FRAME THREAD SINCE PART 102 â€” and it was the czamd session-one
// stutter. GoldenPersist ran synchronously inside the texture DECODE scope, one
// create+write+rename per new small texture, unnamed by any of the decode split's
// timers (it read as RESIDUAL 79.9%, 2,759 ms, on czamd against 0.0% here). On Linux a
// 64 KB file in ~/.cache costs tens of microseconds and never showed; on Windows the
// same write into %LOCALAPPDATA% costs ~2 ms (NTFS + the real-time scanner), so a frame
// that streamed 25 zombie detail maps stalled 50 ms â€” 11-35 textures, 9-600 KB, 25-90 ms
// of "decode" in the czamd trace, every one a first sight of that texture on that
// install. A new player on Windows paid it on every texture of session one. The bytes
// are already copied into g_goldenTex, so the file can be written whenever: a queue and
// one writer thread, drained at exit. The queue is bounded; past the bound a persist is
// dropped and counted (the next session captures it again â€” it is a cache).
namespace goldenwriter
{
struct Job
{
    uint64_t sig = 0;
    std::vector<uint8_t> px;        // the bytes to persist; EMPTY for a fold-in, which
                                    // the writer reads from `loose` itself (the load
                                    // thread already read it once into the map, and a
                                    // second copy of a 361 MB store is not a cost to
                                    // pay for a migration)
    std::filesystem::path loose;    // a loose <sig>.bin to delete once the pack holds it
};
std::mutex mx;
std::condition_variable cv;
std::deque<Job> queue;                                          // guarded by mx
bool up = false;                                                // frame thread only
bool stop = false;                                              // guarded by mx
std::thread thread;
constexpr size_t kMaxQueued = 8192;
uint64_t dropped = 0;                                           // frame thread only
std::mutex packMx;                                              // the pack file below
std::ofstream packOut;                                          // guarded by packMx
uint64_t packAppended = 0, packAppendedBytes = 0, looseRemoved = 0; // guarded by packMx

// The per-file form: the pre-part-104 store, kept whole as the control arm.
void WriteOne(uint64_t sig, const std::vector<uint8_t>& px)
{
    char name[32];
    snprintf(name, sizeof name, "%016llx.bin", (unsigned long long)sig);
    std::error_code ec;
    const std::filesystem::path dir = GoldenDir();
    const std::filesystem::path tmp = dir / (std::string(name) + ".tmp");
    { std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
      f.write(reinterpret_cast<const char*>(px.data()), std::streamsize(px.size())); }
    std::filesystem::rename(tmp, dir / name, ec);
}

// One entry onto the end of the pack. The stream is opened on the first append and
// stays open for the writer's life (an open per entry is the 2 ms Windows cost the
// writer thread was built to hide â€” off the frame now, but still 2 ms of the writer's
// time each); flushed after every entry so a kill loses at most the entry in flight,
// which the loader's torn-tail cut handles. A store that ends up torn is a cache short
// one texture, and the next session captures it again.
bool PackAppend(uint64_t sig, const std::vector<uint8_t>& px)
{
    if (px.empty() || px.size() > kGoldenTexCap)
        return false;
    std::lock_guard<std::mutex> lk(packMx);
    if (!packOut.is_open())
    {
        std::error_code ec;
        const std::filesystem::path p = GoldenPackPath();
        const bool fresh = !std::filesystem::exists(p, ec) || std::filesystem::file_size(p, ec) == 0;
        packOut.open(p, std::ios::binary | std::ios::app);
        if (!packOut)
            return false;
        if (fresh)
        {
            const uint32_t hdr[2] = { kGoldenPackMagic, kGoldenPackVersion };
            packOut.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
        }
    }
    const uint32_t len = uint32_t(px.size());
    packOut.write(reinterpret_cast<const char*>(&sig), sizeof sig);
    packOut.write(reinterpret_cast<const char*>(&len), sizeof len);
    packOut.write(reinterpret_cast<const char*>(px.data()), std::streamsize(px.size()));
    packOut.flush();
    if (!packOut)
        return false;
    ++packAppended;
    packAppendedBytes += 12 + px.size();
    return true;
}

void Run(const Job& job)
{
    if (g_noGoldenPack)
    {
        WriteOne(job.sig, job.px);
        return;
    }
    if (!job.px.empty())
    {
        PackAppend(job.sig, job.px);
    }
    else if (!job.loose.empty() && job.sig)
    {
        // A fold-in the pack did not already hold: read the loose file here, on the
        // writer's time, and append it. If the append fails the file is KEPT â€” it is
        // still the only copy â€” and the next boot tries again.
        std::ifstream f(job.loose, std::ios::binary);
        std::vector<uint8_t> px((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
        if (!PackAppend(job.sig, px))
            return;
    }
    if (!job.loose.empty())
    {
        std::error_code ec;
        if (std::filesystem::remove(job.loose, ec))
        {
            std::lock_guard<std::mutex> lk(packMx);
            ++looseRemoved;
        }
    }
}

void Worker()
{
    ThreadBudget_NameSelf("cz-pipepack");
    for (;;)
    {
        Job job;
        {
            std::unique_lock<std::mutex> lk(mx);
            cv.wait(lk, [] { return stop || !queue.empty(); });
            if (queue.empty())
                return; // stop, and nothing left
            job = std::move(queue.front());
            queue.pop_front();
        }
        Run(job);
    }
}

void Start()
{
    if (up)
        return;
    up = true;
    thread = std::thread(Worker);
    ThreadBudget_Note("golden", 1,
                      "golden texture writer; blocked except while persisting a file");
    ThreadBudget_Report();
}

// `force` is the migration's flag: the fold-in of an old per-file store can be tens of
// thousands of jobs (29,932 here) against a bound sized for a session's captures, and a
// migration that dropped 21,000 of them would leave those files to be walked again on
// every boot â€” the cost this exists to remove. The jobs carry no bytes, so an unbounded
// migration queue is 30,000 paths, not 361 MB.
bool Enqueue(Job&& job, bool force)
{
    {
        std::lock_guard<std::mutex> lk(mx);
        if (!force && queue.size() >= kMaxQueued)
        {
            ++dropped;
            return false;
        }
        queue.emplace_back(std::move(job));
    }
    cv.notify_one();
    return true;
}

// Exit: let the writer finish what is queued (bounded â€” a cache is not worth a hang).
void Drain()
{
    if (!up)
        return;
    {
        std::lock_guard<std::mutex> lk(mx);
        stop = true;
    }
    cv.notify_all();
    if (thread.joinable())
        thread.join();
    up = false;
    std::lock_guard<std::mutex> lk(packMx);
    if (packOut.is_open())
        packOut.close();
}
} // namespace goldenwriter

void GoldenLoad()
{
    if (g_noGolden) return;
    g_noGoldenPack = getenv("CZ_VK_NO_GOLDEN_PACK") != nullptr;
    // TIMED (part 103 item 5). This runs on the boot path before the first frame. A boot
    // cost with no number beside it cannot be ranked, so the line below carries the wall
    // time and both populations â€” the pack's entries and the loose files walked â€” on
    // every platform.
    const auto t0 = std::chrono::steady_clock::now();
    size_t files = 0, folded = 0, packEntries = 0;
    uint64_t torn = 0;
    std::error_code ec;
    const std::filesystem::path dir = GoldenDir();

    // 1. The pack: one read, then a walk over the buffer. An entry whose length runs past
    //    the end is the torn tail a kill leaves; everything before it is good, and the
    //    file is cut back to that point so the next append lands on an entry boundary.
    if (!g_noGoldenPack)
    {
        const std::filesystem::path pp = GoldenPackPath();
        // One fread into an uninitialised buffer, then the entries are copied out into
        // their own vectors. MEASURED, not assumed (part 104, dev box, 345 MB / 29,933
        // entries): the read is ~390 ms and the fill ~270 ms, and swapping an ifstream
        // into a zero-filled vector for this form changed neither â€” the read is bound by
        // first-touch page faults on the fresh 345 MB, not by the stream. A store this
        // size is the dev box's oddity (every small texture ever seen, 3070 sessions);
        // czamd's is ~70 MB. If a boot ever needs the last 400 ms back, the next step is
        // an mmap the map's entries point INTO (no copy, no heap), not a faster copy.
        const uintmax_t fsize = std::filesystem::exists(pp, ec) ? std::filesystem::file_size(pp, ec) : 0;
#ifdef _WIN32
        FILE* f = _wfopen(pp.c_str(), L"rb");
#else
        FILE* f = fopen(pp.c_str(), "rb");
#endif
        if (f)
        {
            std::unique_ptr<uint8_t[]> buf(fsize ? new uint8_t[size_t(fsize)] : nullptr);
            const size_t size = fsize ? fread(buf.get(), 1, size_t(fsize), f) : 0;
            fclose(f);
            auto rd32 = [&](size_t o) { uint32_t v; memcpy(&v, buf.get() + o, 4); return v; };
            auto rd64 = [&](size_t o) { uint64_t v; memcpy(&v, buf.get() + o, 8); return v; };
            size_t off = 0;
            if (size >= 8 && rd32(0) == kGoldenPackMagic && rd32(4) == kGoldenPackVersion)
            {
                off = 8;
                g_goldenTex.reserve(g_goldenTex.size() + size / 8192); // ~a rehash or two, not thirty
                while (off + 12 <= size)
                {
                    const uint64_t sig = rd64(off);
                    const uint32_t len = rd32(off + 8);
                    if (len == 0 || len > kGoldenTexCap || off + 12 + len > size)
                        break; // torn (or foreign) tail â€” stop here, cut below
                    auto& slot = g_goldenTex[sig];
                    if (slot.empty())
                        slot.assign(buf.get() + off + 12, buf.get() + off + 12 + len);
                    ++packEntries;
                    off += 12 + len;
                }
                if (off != size)
                {
                    torn = size - off;
                    std::filesystem::resize_file(pp, off, ec);
                }
            }
            else if (size)
            {
                // Not ours. Set it aside rather than overwrite it â€” a file with that name
                // that we did not write is a question, and a cache is not worth destroying
                // someone's answer to it.
                std::filesystem::rename(pp, dir / "golden.pack.foreign", ec);
                fprintf(stderr, "[vk] golden texture store: %s had a foreign header â€” moved "
                                "to golden.pack.foreign, starting a fresh pack\n",
                        pp.string().c_str());
            }
            g_goldenPackEntries = packEntries;
            g_goldenPackBytes = off;
        }
    }

    // 2. Loose files. Under the per-file arm this is the store; under the pack it is the
    //    fold-in of whatever predates it or was written under the arm, and each file is
    //    handed to the writer to append (if the pack lacks it) and delete.
    for (auto& e : std::filesystem::directory_iterator(dir, ec))
    {
        if (ec) break;
        if (!e.is_regular_file()) continue;
        if (e.path().extension() != ".bin") continue; // the pack, .tmp leftovers, .foreign
        const uint64_t sig = strtoull(e.path().stem().string().c_str(), nullptr, 16);
        if (!sig) continue;
        ++files;
        auto& slot = g_goldenTex[sig];
        const bool fresh = slot.empty();
        if (fresh)
        {
            std::ifstream f(e.path(), std::ios::binary);
            std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)),
                                      std::istreambuf_iterator<char>());
            if (data.empty() || data.size() > kGoldenTexCap)
            {
                g_goldenTex.erase(sig);
                continue;
            }
            slot = std::move(data);
        }
        if (!g_noGoldenPack)
        {
            // Fresh: the writer reads the file and appends it, then deletes it. Not
            // fresh (the pack already holds this signature): sig 0 says "delete only".
            goldenwriter::Job job;
            job.sig = fresh ? sig : 0;
            job.loose = e.path();
            goldenwriter::Start();
            goldenwriter::Enqueue(std::move(job), /*force=*/true);
            ++folded;
        }
    }

    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    if (!g_goldenTex.empty() || files || packEntries)
        fprintf(stderr,
                "[vk] golden texture store: preloaded %zu signatures in %.1f ms from %s â€” "
                "pack %zu entries (%.1f MB%s), %zu loose files walked%s%s\n",
                g_goldenTex.size(), ms, dir.string().c_str(), packEntries,
                double(g_goldenPackBytes) / (1024.0 * 1024.0),
                torn ? ", torn tail cut" : "", files,
                folded ? " and folded into the pack" : "",
                g_noGoldenPack ? " [CZ_VK_NO_GOLDEN_PACK: per-file store, pack ignored]" : "");
    if (torn)
        fprintf(stderr, "[vk] golden texture store: cut %llu torn bytes off the end of the "
                        "pack (a previous session was ended mid-write)\n",
                (unsigned long long)torn);
}

void GoldenPersist(uint64_t sig, const std::vector<uint8_t>& px)
{
    if (g_noGolden || px.empty()) return;
    static const bool syncArm = [] {
        const char* v = getenv("CZ_VK_GOLDEN_SYNC");
        return v && *v && strcmp(v, "0") != 0;
    }();
    if (syncArm)
    {
        // The pre-part-102 behaviour â€” the file written on the frame thread â€” kept as
        // the same-binary control arm for the czamd stutter. Under the pack it is the
        // append done synchronously; under the per-file arm the file.
        goldenwriter::Job job;
        job.sig = sig;
        job.px = px;
        goldenwriter::Run(job);
        return;
    }
    goldenwriter::Start();
    goldenwriter::Job job;
    job.sig = sig;
    job.px = px;
    goldenwriter::Enqueue(std::move(job), /*force=*/false);
}
struct TexGuardStats
{
    uint64_t hits = 0;        // cache hits the guard was computed for
    uint64_t changed = 0;     // ...whose guest bytes had changed since upload
    uint64_t reuploaded = 0;  // ...and were re-uploaded (REVALIDATE only)
    uint64_t guardBytes = 0;  // what the guard itself read, i.e. its own cost
    // ONCE PER FRAME PER ENTRY (part 47). Cache hits the guard was NOT computed for
    // because this same cache entry had already been validated in this frame â€” i.e.
    // what the policy below saves. It is the numerator of the whole item: hits +
    // skippedSameFrame is what the pre-part-47 renderer would have hashed.
    uint64_t skippedSameFrame = 0;
} g_texGuardStats;

// CZ_VK_TEX_GUARD_EVERY_FETCH=1 â€” the same-binary CONTROL ARM for that policy, i.e. the
// pre-part-47 behaviour: revalidate on every fetch rather than once per frame per entry.
//
// WHY THE POLICY EXISTS. The operator's own profiled frame charged **26.5 ms of 61.7 to
// `textures`**, and the guard is most of it: over one session it read **366 GB â€” 92.9 MB
// a frame â€” to catch 986 real changes in 26.8 M checks (0.0037%)**
// (`docs/perf-plan-part47.md` Â§1.1). The mechanism is load-bearing and stays: part 38
// built it because a streaming-recycled address served its first occupant forever (the
// tanker cylinder wearing a brick wall). What does not have to stay is buying that
// exactness once per FETCH when the same texture is fetched by many draws of one frame.
//
// WHAT IT COSTS, stated so a run can refute it: a texture the guest rewrites mid-frame
// is now served stale for the REST OF THAT FRAME instead of from the first draw after
// the write â€” at most one frame of latency, against a picture that only updates once a
// frame anyway. The falsifiable claim is that `changed` does not fall: if a change is
// real it is still there at the next frame's first fetch, so the two arms should report
// the same population. A drop means the policy is losing detections and the number says
// how many.
bool g_texGuardEveryFetch = false;
// Per address, because "17% of hits are stale" and "one atlas is stale every frame" are
// completely different defects and a single ratio cannot tell them apart.
struct TexGuardAddr
{
    uint64_t hits = 0, changed = 0;
    uint32_t width = 0, height = 0, format = 0;
    uint64_t srcBytes = 0;   // what one check of this texture costs
};
// Where the guard's bytes go, by SOURCE size: bucket b is [1 KB << b, 1 KB << (b+1)),
// bucket 0 everything below 1 KB, the last bucket everything above. See the comment at
// the increment for why this exists â€” it is the price list for a bounded-prefix guard.
constexpr size_t kTexGuardHistBuckets = 12;
uint64_t g_texGuardHistCount[kTexGuardHistBuckets]{};
uint64_t g_texGuardHistBytes[kTexGuardHistBuckets]{};

// CZ_VK_DIM_CENSUS=1 â€” WHERE IS THE DIMENSION IN THE TEXTURE FETCH CONSTANT?
//
// The renderer needs the dimension twice over and from two different places. The SHADER
// says which descriptor heap a slot is sampled from (its fetch instruction picks
// `tfetch2D` or `tfetchCube`), and part 25 put that in the sidecar. But the GUEST has to
// say how the image is laid out in memory â€” a cube map is six faces, and reading one is
// reading a sixth of it â€” and that can only come from the fetch constant, whose
// `dimension` field this renderer never decoded at all: `DecodeTextureFetch` hardcoded
// `t.dimension = 1`.
//
// The bit position is not something to remember. It is something to MEASURE, and this
// title provides the oracle for free: the shader-declared dimension partitions every
// fetch into two classes that must differ in exactly the bits of that field. So for each
// class this accumulates the AND and the OR of all six dwords across every fetch. A bit
// that is set in every fetch of a class has AND=1; one clear in every fetch has OR=0.
// The field is where the two classes' always-set / always-clear patterns disagree â€” and
// if no two-bit field disagrees cleanly, the dimension is not in the fetch constant here
// and the shader is the only source, which is an answer too.
//
// It also censuses dword2's top six bits, which Xenia's layout says is the stack DEPTH
// for a stacked or cube surface: the prediction is 5 (six faces, stored minus one) for
// every cube fetch and 0 for every 2D one. A prediction stated before the run, so the
// run can refute it (the project's own evidence rule).
bool g_dimCensus = false;
struct DimClass
{
    uint64_t fetches = 0;
    uint32_t andMask[6] = { ~0u, ~0u, ~0u, ~0u, ~0u, ~0u };
    uint32_t orMask[6] = {};
    std::map<uint32_t, uint64_t> d2Top;   // dword2 >> 26 -> count
};
std::map<uint32_t, DimClass> g_dimClasses;  // shader-declared dimension -> the census

// CZ_VK_DIM_DISAGREE=N â€” PRINT THE FIRST N SHADER-VERSUS-CONSTANT DISAGREEMENTS IN FULL.
//
// The cross-check in `UploadTexture` says a disagreement HAPPENED â€” ~14,670 cube fetches
// a run declined to the white dummy because the shader indexes the cube array while the
// fetch constant describes a 2D surface. It cannot say WHY, and the round-2 captures
// turned that from a curiosity into a defect with a known victim: over the gas-station
// frame, 414 of 414 cube-declared draws on HARDWARE read stack depth 5 and dimension 3,
// with no disagreement at all (docs/open-items.md 00g). So the disagreement is ours, and
// it is the confirmed mechanism behind the white glass and the blown-out bathroom window.
//
// Two candidate causes, and they need different fixes: our register file has LOST a
// constant the guest set (so the slot holds something else â€” stale, or another slot's
// texture), or our dimension DECODE misreads a case the capture does not contain. This
// prints what separates them: the six raw dwords of the offending slot, and then the
// whole 32-slot fetch-constant file as we hold it at that draw, so an off-by-one in the
// slot index or a stale neighbour is visible rather than inferred.
// The one-liners are capped, because the useful part is the CENSUS underneath them: which
// shaders disagree, at which slot, about which texture. The first version printed only
// the first 25 and every one of them was the same shader at the same slot on two frames â€”
// which reads as "there is one case" and is equally consistent with "the cap was reached
// inside one draw batch". A capped log line is not a count (gotcha 109), so the census is
// unbounded and prints at shutdown.
bool g_dimDisagree = false;
int g_dimDisagreeLeft = 0;
struct DimDisagree
{
    uint64_t psHash = 0, vsHash = 0;
    uint32_t slot = 0, shaderDim = 0, constDim = 0, addr = 0, w = 0, h = 0, fmt = 0;
    uint64_t fetches = 0;
};
std::map<uint64_t, DimDisagree> g_dimDisagreements;   // keyed on shader+slot+address

// CZ_VK_STREAM_CENSUS=1|2 â€” what the per-frame vertex/index stream cache actually does.
//
// `streams` is the largest draw-path term in a real crowd (12.3-14.3% of a frame, twice
// what the headless recipe shows), and the plan could not say what to do about it,
// because two opposite readings fit the same millisecond count:
//
//   * nearly all HITS â€” then the cost is the lookup and the fix is a cheaper key;
//   * mostly MISSES â€” then it is real copying and the fix is a different cache LIFETIME.
//
// No amount of reading the code decides that, so this counts it. One thing the code DOES
// decide: `ProfScope(streams)` wraps only the `CopySwapped`, so the `streams` column is
// copy time and a hit costs it nothing â€” a hit's lookup is charged to `other`.
//
// Level 2 additionally answers the question a cross-frame cache stands or falls on:
// whether the bytes at a repeated (address, size, endian) are the SAME bytes next frame.
// A persistent cache keyed on the address is wrong if the guest rewrites the buffer in
// place, and "the key repeats" is not evidence that "the content repeats". Level 2 hashes
// every stream, which costs about as much as the copy â€” it is a diagnostic run, never a
// frame-time measurement (gotcha 223).
int g_streamCensus = 0;
struct StreamCensus
{
    uint64_t hits = 0, misses = 0;      // per-draw cache outcomes
    uint64_t bytesCopied = 0;           // what the misses actually swapped
    uint64_t bytesHit = 0;              // what the hits avoided swapping
    // Of the misses, the ones whose key was present in the PREVIOUS frame â€” i.e. what a
    // cache that survived the frame boundary would have hit instead.
    uint64_t prevFrameKeyHits = 0;
    uint64_t prevFrameKeyBytes = 0;
    // ...and of THOSE, the ones whose bytes were unchanged since last frame (level 2).
    // The gap between this and the line above is the guest rewriting a buffer in place,
    // which is exactly the traffic a persistent cache would serve stale.
    uint64_t prevFrameSameContent = 0;
    uint64_t prevFrameSameBytes = 0;
    // Of the streams the full hash says DID change, the ones the cross-frame store served
    // from its own copy anyway â€” i.e. the ones its bounded-cost guard missed. This is the
    // correctness measurement for the whole persistent cache and it must read zero.
    uint64_t guardMissed = 0;
    // Copied bytes and misses split by what the stream IS: [0] declared vertex binding,
    // [1] index buffer, [2] shader-side dependent fetch.
    uint64_t kindBytes[3] = { 0, 0, 0 };
    uint64_t kindMisses[3] = { 0, 0, 0 };
    uint64_t kindRepeatBytes[3] = { 0, 0, 0 };
};
StreamCensus g_streamCensus_c;

// WHICH streams change in place, not just how many. Level 2 only, cumulative over the
// whole run rather than per profile window.
//
// Part 21 established that 164 of 10,154,820 repeated keys really do get rewritten by the
// guest â€” 0.0016%, and it described the changing set as "a recurring ~26". That number
// decides how a persistent cache invalidates, and the two answers cost wildly different
// amounts of work: if the rewritten streams are a NAMED set â€” one address range, or one
// `kind` â€” then invalidation is an exclusion rule and costs nothing, while if they are
// scattered across the geometry it has to be guest-page write tracking (`mprotect` plus a
// `SIGSEGV` handler sharing the process with `cpu/crash_report.cpp`'s). The census
// already detects each mismatch; it just threw the identity away. This keeps it.
//
// Cumulative on purpose: the claim under test is that the same keys recur, and a counter
// reset every five seconds cannot show recurrence. Unbounded in principle, bounded in
// practice by the finding it is testing â€” and if it is NOT bounded, that is the finding.
struct StreamChange
{
    uint64_t times = 0;   // how many frames this key was seen rewritten
    uint64_t bytes = 0;   // its size, which is part of the key and so constant
    int kind = 0;         // 0 vertex binding, 1 index buffer, 2 dependent fetch
    uint64_t firstFrame = 0, lastFrame = 0;
};
std::unordered_map<uint64_t, StreamChange> g_streamChanged;

// Last frame's keys, and (level 2 only) a content hash for each. Rebuilt in BeginFrame
// out of the cache that is about to be cleared, so the draw path never walks it.
std::unordered_map<uint64_t, uint64_t> g_prevStreamKeys;
// This frame's hashes, level 2 only. Separate from `streamCache` because that map is on
// the hot path and must not grow a field an instrument is the only reader of.
std::unordered_map<uint64_t, uint64_t> g_streamHashes;

// CZ_VK_STREAM_CENSUS_POISON=1 â€” THE CONTROL ARM FOR THE CONTENT CHECK, and it exists
// because that check reports 100.0% and nothing else.
//
// The level-2 line says "of the streams whose key repeated last frame, N of N had
// identical bytes". On this title it reads 149,925 of 149,925 in a crowd, which is either
// a real and very useful fact about the guest's geometry or a comparison that cannot
// fail. Those are indistinguishable from the output. With this on, the hash is salted
// with the FRAME NUMBER, so the same bytes hash differently in consecutive frames and
// the line MUST read 0.0%. If it still reads 100%, the instrument is broken and the
// finding built on it is worthless (gotchas 30, 94, 158).
bool g_streamPoison = false;

// FNV-1a over a stream's GUEST bytes. Level 2 only, and deliberately over the source
// rather than the arena copy: the question is whether the guest rewrote the buffer, and
// the arena copy is our own output.
uint64_t StreamHash(const uint8_t* p, size_t bytes, uint64_t salt)
{
    uint64_t h = 1469598103934665603ull ^ salt;
    for (size_t i = 0; i < bytes; ++i)
        h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

// THE GUARD: a bounded-cost fingerprint of a stream's guest bytes, and the thing that
// makes a cross-frame cache a cache rather than an assumption.
//
// The problem it solves is measured, not hypothetical. 30 distinct streams a run really
// are rewritten in place by the guest at an address a previous frame already cached, and
// serving the old copy draws the wrong mesh. Detecting that by hashing the whole stream
// costs about what the copy costs, which would give the saving straight back.
//
// So the cost is capped at `kGuardBytes` regardless of stream size:
//
//   * a stream of `kGuardBytes` or fewer is hashed IN FULL, so the check is exact for it;
//   * a larger one is hashed at eight evenly spread blocks, always including the first
//     and the last, so a rewrite has to miss all eight windows to go unnoticed.
//
// The threshold is not arbitrary. Every rewritten stream this title has ever been seen
// to produce is EXACTLY 80 bytes (30 of them, in two narrow guest ranges, all declared
// vertex bindings â€” see the census's REWRITTEN IN PLACE line), so 512 puts the entire
// observed population on the exact branch with a 6x margin. But "we have never seen a
// big one change" is a zero, and a zero is a detection failure until something could
// have detected it (gotcha 3) â€” hence the sampled branch rather than a size cutoff that
// simply declines to check. `CZ_VK_STREAM_CENSUS=2` measures what the sampling misses:
// it computes the FULL hash as well and counts every change the guard did not catch.
//
// Cost in a crowd frame: ~2,000 first-touch streams x <=512 B = under 1 MB, against the
// 61-77 MB of copying it is there to avoid.
// THE EXACT BOUND, AND WHY IT IS NO LONGER 512.
//
// 512 was fitted to a census that found every rewritten stream was exactly 80 bytes â€”
// but that census could only see streams rewritten between two consecutive frames in a
// recipe that never changed a HUD number, so the bound was fitted to the population the
// instrument could reach (gotcha 235, second instance). The streams it could not see
// are the ones that broke: a HUD is batched into one multi-KB vertex buffer in which
// only the digit quads change, those quads fall outside the 8x64 sampled windows, and
// the store serves the previous frame's numbers. That is open item 00c, and the
// operator confirmed the exact guard fixes it outright across several minutes and two
// weapons.
//
// Exact-everywhere is not the ship-able form. Measured on the outdoor recipe:
//
//   guard read   0.41 MB/frame -> 30.70 MB/frame   (75x)
//   `record`     4.8% -> 16.7% of frame            (+11.9 points, ~3.8 ms)
//
// So the bound is raised rather than removed: hash EXACTLY up to kGuardBytes, sample
// above it. Cost is bounded by (streams/frame x kGuardBytes) â€” about 1,000 x 16 KB
// worst case here, against the 26-30 MB/frame of copying the store avoids.
//
// 16 KB is deliberately generous rather than tuned. The HUD buffer's true size has not
// been measured, and the failure mode of a bound that is too small is the defect coming
// back silently, while the failure mode of one too large is frame time this title's
// vblank floor absorbed entirely at 2,000 draws (32.0 ms on BOTH arms). Given that
// asymmetry, guess high. `CZ_VK_STREAM_GUARD_BYTES=N` overrides it without a rebuild so
// the bound can be tuned against the defect instead of against a model of it, and
// `CZ_VK_STREAM_GUARD_EXACT=1` still means unlimited.
//
// NOT MEASURED: the crowd. The A/B above parked at ~2,000 draws; the recipe peaks near
// 8,300-8,700 and there is no profile window at that depth. A crowd frame moves 61-77 MB
// of stream bytes, so if anything is going to hurt it is there â€” measure before calling
// this free.
constexpr size_t kGuardBytesDefault = 16384;
size_t g_guardBytes = kGuardBytesDefault;
// Streams that exceeded the bound and were therefore only SAMPLED, cumulative.
uint64_t g_guardSampled = 0;
constexpr size_t kGuardBlocks = 8;

// THE SIZE HISTOGRAM OF THE EXPOSED POPULATION, and why a count was not enough.
//
// The counter above says how MANY streams the bound leaves sampled â€” 6,405 a frame in
// the operator's session â€” and that number cannot choose a new bound, because raising
// the bound costs in proportion to the BYTES of the streams it newly covers, not their
// count. Item 00c's fix was parked twice for exactly this: "raise the bound" was the
// recommendation both times and nobody could say to what, because a bound is only
// choosable against the distribution it has to separate. If the UI text buffer is 40 KB
// and the crowd's meshes are 400 KB, one number fixes the picture for almost nothing;
// if they overlap, size is the wrong discriminator entirely and that is worth knowing
// before building anything on it.
//
// Powers of two from 16 KB up, so a bound can be read straight off it.
constexpr size_t kGuardHistBuckets = 8;   // 16K,32K,64K,128K,256K,512K,1M,>1M
uint64_t g_guardHistCount[kGuardHistBuckets] = {};
uint64_t g_guardHistBytes[kGuardHistBuckets] = {};

// EIGHT BYTES A STEP, not one, and the reason is the dependency chain rather than the
// memory. FNV-1a is `h = (h ^ byte) * prime`, so every byte waits on the previous byte's
// MULTIPLY â€” about four cycles each, serial, no matter how fast the loads are. At 512
// bytes that is ~0.55 us per stream and ~1 ms per crowd frame, a fifth of what the store
// saves, spent inside the check that makes the store safe. Folding a whole uint64 per
// step keeps the mixing (xor then multiply by an odd constant is injective in the input
// word either way) and cuts the chain to 64 steps.
// CZ_VK_GUARD_FOLD_SERIAL=1 restores the single-accumulator fold â€” the same-binary
// control arm for the part-47 widening below.
inline bool GuardFoldSerial()
{
    // getenv rather than Env(): this sits above Env's definition, and moving Env up
    // would drag the whole environment-arm block above the profiler it is used by.
    static const bool serial = getenv("CZ_VK_GUARD_FOLD_SERIAL") != nullptr;
    return serial;
}

// FOUR INDEPENDENT LANES, because this loop is LATENCY-bound and not bandwidth-bound.
//
// The single-accumulator form is `h = (h ^ v) * PRIME` per 8 bytes, and the multiply is
// on the critical path: ~5 cycles of latency per 8 bytes is ~1.6 bytes/cycle, roughly
// 6 GB/s, however much load bandwidth the machine has. That did not matter when this
// was hashing a few hundred KB. It matters now: **the stream guard reads 81.65 MB in
// one frame of the operator's session** (60.8 of it the exact-hash promotion that fixed
// the UI text in part 46), and at ~6 GB/s that alone is most of the 15.2 ms `record`
// phase â€” which the part-47 split shows is 1,327 ns per draw in its vertex section.
//
// Four lanes give the out-of-order engine four independent multiply chains, so the loop
// becomes load-bound instead. **The bytes read, the coverage and therefore the detection
// power are all unchanged** â€” this is the same hash over the same input, computed
// faster. The VALUE changes, which is safe because a guard is only ever compared with
// another guard computed by this same code in this same process; none is persisted or
// compared with anything external.
//
// The tail is folded serially into lane 0 so a buffer under 32 bytes still mixes every
// byte, and the lanes are combined with distinct rotations so that two lanes swapping
// contents cannot cancel.
// CZ_VK_GUARD_NTA=1 (part 118): non-temporal PREFETCH ahead of the fold, so the ~37 MB
// a frame the guard streams through does not displace the GUEST's working set from the
// shared L3. The guard reads every vertex stream once a frame, sequentially, and never
// again until the next frame â€” the textbook non-temporal access â€” and part 118's PMU
// pair on the Main Thread says our renderer's presence turns ~3,700 of its L3 hits a
// frame into DRAM fills (31.1k vs 27.4k), on a thread whose cost is serialised misses.
// x86 has no non-temporal LOAD for write-back memory; PREFETCHNTA is the nearest thing
// (AMD: fills L1 only, not written back into L2/L3 on eviction). An arm because the
// benefit lands on ANOTHER thread and only a frame-time A/B can price it.
inline bool GuardFoldNta()
{
    static const bool nta = getenv("CZ_VK_GUARD_NTA") != nullptr;
    return nta;
}

inline uint64_t GuardFold(uint64_t h, const uint8_t* p, size_t n)
{
    constexpr uint64_t P = 1099511628211ull;
    size_t i = 0;
    if (!GuardFoldSerial())
    {
        uint64_t h0 = h, h1 = h ^ 0x9E3779B97F4A7C15ull, h2 = h ^ 0xC2B2AE3D27D4EB4Full,
                 h3 = h ^ 0x165667B19E3779F9ull;
        const bool nta = GuardFoldNta();
        if (nta)   // the first lines, which the loop's own prefetch would arrive too late for
            for (size_t k = 0; k < 512 && k < n; k += 64)
                __builtin_prefetch(p + k, 0, 0);
        for (; i + 32 <= n; i += 32)
        {
            if (nta && (i & 63) == 0)
                __builtin_prefetch(p + i + 512, 0, 0);   // locality 0 = prefetchnta
            uint64_t v0, v1, v2, v3;
            memcpy(&v0, p + i, 8);        // unaligned-safe; each compiles to one load
            memcpy(&v1, p + i + 8, 8);
            memcpy(&v2, p + i + 16, 8);
            memcpy(&v3, p + i + 24, 8);
            h0 = (h0 ^ v0) * P;
            h1 = (h1 ^ v1) * P;
            h2 = (h2 ^ v2) * P;
            h3 = (h3 ^ v3) * P;
        }
        h = h0 ^ ((h1 << 13) | (h1 >> 51)) ^ ((h2 << 27) | (h2 >> 37)) ^
            ((h3 << 41) | (h3 >> 23));
    }
    for (; i + 8 <= n; i += 8)
    {
        uint64_t v;
        memcpy(&v, p + i, 8);   // unaligned-safe and compiles to one load
        h = (h ^ v) * P;
    }
    for (; i < n; ++i)
        h = (h ^ p[i]) * P;
    return h;
}

// CZ_VK_STREAM_GUARD_EXACT=1 â€” hash EVERY byte, whatever the stream's size.
//
// The sampling below is exact only up to 512 bytes; above that it reads 8 blocks of 64
// and can therefore miss a small edit in a large stream. That is precisely the shape of
// the HUD defect in open item 00c: a UI vertex buffer of a few KB in which only the two
// quads carrying the ammo digits change, and 512 sampled bytes that never land on them.
// A missed change means the guard calls the stream unchanged and the store serves the
// PREVIOUS frame's buffer â€” the old number â€” which is what "flickers between 26 and 27
// regardless of the real ammo" looks like when the guest double-buffers its UI.
//
// This is the arm that separates "the store is guilty" from "the store's GUARD is
// guilty", which `CZ_VK_NO_PERSIST_STREAMS=1` cannot: that one disables the store
// wholesale and costs 4.7 ms of a crowd frame, so it can never be the fix even when it
// makes the symptom go away. If the symptom goes away HERE, the fix is a better guard
// for UI-sized streams and the 4.7 ms stays bought.
//
// Not the default, because the cost is unbounded in stream size and unmeasured on a
// crowd frame â€” establish the picture first, then decide what it is worth.
bool g_guardExact = false;

// Streams promoted to an exact guard because they were caught changing, and what that
// promotion costs â€” both reported, because an adaptive policy nobody can price is how
// "raise the bound" stayed unactionable for two parts.
uint64_t g_guardDynamic = 0;
uint64_t g_guardDynamicBytes = 0;

// ...AND WHY EACH ONE WAS PROMOTED, which is the split part 50 item 2a turns on.
//
// The policy has three doors to the exact guard and they have completely different
// prices. `needsExact` â€” the sampled guard has been caught missing a real change â€” is
// UNBUDGETED and PERMANENT, because that is the UI-text case item 00c was about and
// serving a stale HUD to save bandwidth is not a trade this renderer makes. The other
// two, a dynamic stream still accruing its proof and a newly-met stream being probed,
// share one 4 MB per-frame toll.
//
// Which means the whole promotion cost can be read off these three numbers, and part 46
// left a prediction to check: `needsExact` should be "the UI text buffers and almost
// nothing else". If it is, the promotion cannot exceed the 4 MB budget by much and item
// 2a is somewhere else entirely. If it is not â€” if a permanent, unbudgeted latch is
// firing on ordinary geometry â€” then the promotion is unbounded by construction and
// grows for as long as the player keeps meeting new streams, which no counter here would
// have shown. Splitting a cost by its REASON is what found three items in two parts
// (gotcha 327); this is the same move one subsystem over.
uint64_t g_guardProven = 0,  g_guardProvenBytes = 0;   // needsExact: unbudgeted, forever
uint64_t g_guardSpec = 0,    g_guardSpecBytes = 0;     // dynamic, still accruing proof
uint64_t g_guardProbe = 0,   g_guardProbeBytes = 0;    // newly met, bootstrap probe
// How many entries have EVER latched `needsExact`, against the store's size. The latch
// never unlatches, so this is a ratchet and its trend is the thing to read.
uint64_t g_guardProvenEntries = 0;
// ...and the number that decides what to DO about it: of the observations that came
// through the proven door, how many found the stream actually CHANGED?
//
// The guard exists to avoid a copy. For a stream that changes on nearly every frame it
// avoids nothing and costs a whole extra read of the buffer: we hash N bytes to learn
// what we are about to find out anyway, and then copy N bytes. Always copying such a
// stream is cheaper (one pass instead of two) AND strictly safer â€” a stream that is
// always copied can never be served stale, which is the defect this whole mechanism
// exists to prevent. For a stream that rarely changes the guard is the win it was built
// to be. So the change RATE within the proven set is the whole decision, and it has
// never been measured.
uint64_t g_guardProvenObs = 0, g_guardProvenChanged = 0;

// The folding itself, at a CALLER-CHOSEN bound: exact up to `bound`, and above it eight
// blocks of `bound/8` spread over the whole buffer â€” the first starting exactly at 0 and
// the last ending exactly on the final byte.
//
// Split out in part 47 so the TEXTURE guard can have its own bound without borrowing the
// stream guard's counters. They are two different questions with two different answers:
// a stream guard is looking for a small edit inside a batched UI buffer (item 00c, where
// missing one shows the previous frame's ammo count), and a texture guard is looking for
// an ADDRESS THE STREAMING SYSTEM RECYCLED â€” an entirely different texture written over
// the old one, which a spread sample sees at essentially any bound. Sharing the constant
// between them was never a decision anyone made.
uint64_t GuardOver(const uint8_t* p, size_t bytes, size_t bound, size_t* readOut)
{
    uint64_t h = (1469598103934665603ull ^ bytes) * 1099511628211ull;
    if (bytes <= bound)
    {
        if (readOut)
            *readOut += bytes;
        return GuardFold(h, p, bytes);
    }
    const size_t block = bound / kGuardBlocks;
    const size_t span = bytes - block;
    for (size_t b = 0; b < kGuardBlocks; ++b)
        h = GuardFold(h, p + (span * b) / (kGuardBlocks - 1), block);
    if (readOut)
        *readOut += bound;
    return h;
}

// The bound the TEXTURE content guard folds at, in bytes. `CZ_VK_TEX_GUARD_BYTES=N`.
//
// It defaults to the stream guard's 16 KB, which is what it has silently been since part
// 38 â€” so the default is not a change and every earlier texture-guard number stays
// comparable. It is a knob because the guard costs 92.9 MB a frame and the histogram
// printed with the stats says exactly what each bound would buy; choosing one is then a
// measurement rather than a guess, which is the whole reason "raise the bound" stayed
// unactionable for two parts on the stream side.
size_t g_texGuardBytes = 16384;

uint64_t TextureGuard(const uint8_t* p, size_t bytes, size_t* readOut)
{
    return GuardOver(p, bytes, g_texGuardBytes, readOut);
}

// SPLIT IN TWO for part 53 item 1.1, and the split is the whole reason the parallel
// guard can be shown to change nothing but WHERE the hash runs.
//
// `StreamGuardCount` is the census â€” the promotion counters and the sampled-size
// histogram â€” and it stays on the pump thread whoever does the hashing, because those
// numbers are what make the two arms comparable at all. `StreamGuardHash` is the pure
// part: guest bytes in, a `uint64_t` out, no globals written. Only the pure half moves.
//
// Keeping them callable together as `StreamGuard` means the serial path is byte-for-byte
// the code it always was, which is what makes it a usable oracle rather than a rewrite
// that happens to agree.
void StreamGuardCount(size_t bytes, bool forceExact)
{
    if (g_guardExact || forceExact)
    {
        if (forceExact && !g_guardExact)
        {
            ++g_guardDynamic;
            g_guardDynamicBytes += bytes;
        }
        return;
    }
    // Above the bound the guard is a SAMPLE and can therefore miss a small edit. Count
    // the exposure rather than leaving it silent: this is the population that item 00c's
    // defect lived in, and a bound raised until this counter is zero for the streams that
    // matter is a bound chosen by measurement instead of by guess.
    if (bytes > g_guardBytes)
    {
        ++g_guardSampled;
        // Which power-of-two bucket this stream would need the bound raised to.
        size_t b = 0;
        for (size_t lim = 32768; b + 1 < kGuardHistBuckets && bytes >= lim; lim <<= 1)
            ++b;
        ++g_guardHistCount[b];
        g_guardHistBytes[b] += bytes;
    }
}

// How many bytes a guard over `bytes` at `bound` READS â€” derivable from the sizes alone,
// so the byte census stays exact even when the hash itself ran on another thread and
// never reported back. Mirrors GuardOver's own two cases.
inline uint64_t GuardReadBytes(uint64_t bytes, uint64_t bound)
{
    return bytes <= bound ? bytes : bound;
}

uint64_t StreamGuardHash(const uint8_t* p, size_t bytes, size_t* readOut, bool forceExact)
{
    if (g_guardExact || forceExact)
    {
        if (readOut)
            *readOut += bytes;
        return GuardFold((1469598103934665603ull ^ bytes) * 1099511628211ull, p, bytes);
    }
    // The size is folded in first (inside GuardOver). Without it a stream that shrinks to
    // a prefix of itself would hash the same, and size is part of the key only for the
    // streams the key came from â€” a re-copy into the same slot keeps the slot's size.
    return GuardOver(p, bytes, g_guardBytes, readOut);
}

uint64_t StreamGuard(const uint8_t* p, size_t bytes, size_t* readOut, bool forceExact)
{
    StreamGuardCount(bytes, forceExact);
    return StreamGuardHash(p, bytes, readOut, forceExact);
}

bool g_active = false;
bool g_initTried = false;

// Which feed owns the renderer this run. False = the PM4 executor (CZ_VKDRAW,
// phase 5); true = the D3D draw service (CZ_D3D_DRAW, phase C). Set once at init and
// never changed: the entries belonging to the other feed check it and return, so a
// run can never have both feeds drawing into one EDRAM image.
bool g_d3dMode = false;

const char* Env(const char* n) { return getenv(n); }
bool EnvOn(const char* n) { return getenv(n) != nullptr; }

// Take one from a "print only the first N of these" countdown, STOPPING at zero.
//
// The obvious spelling -- decrementing inside the comparison -- is a bug, and it shipped
// in v1.0.2 on the hottest path in the project (pm4.cpp's [pm4draw] trace): it decrements
// on EVERY call whether it prints or not, so the counter runs past zero and, at 2^31, wraps
// to INT_MAX â€” at which point a trace NOBODY ARMED turns itself on and never stops. On a
// per-draw path that took about 80 minutes of play: the frame rate roughly halved and
// cz_runtime.log reached 98 GB. Every countdown in this file had the same shape; none of
// the others could reach 2^31 because they only tick on rare or defect paths, which is
// luck rather than design. This makes the safe form the easy one to write.
bool TakeOne(int& left)
{
    if (left <= 0)
        return false;
    --left;
    return true;
}

// ===================================================================================
// PART 81 Â§1.1: THE DEVICE COMMAND TABLE â€” one indirection off every vkCmd* call
// ===================================================================================
//
// WHAT THIS REMOVES, AND WHY IT IS WORTH A NAMED CHANGE.
//
// Every `vkCmd*` symbol a program links against is the Vulkan LOADER's exported
// trampoline, not the driver's function. Calling it costs a PLT hop into libvulkan, a
// load of the dispatch table out of the dispatchable handle's first word, and an
// indirect jump through that table into the driver â€” and only then does the driver's own
// work begin. `vkGetDeviceProcAddr` hands back the driver's entry point directly, so a
// call through a stored pointer skips the trampoline entirely.
//
// Part 80 measured the whole driver call chain at **251 ns a draw = 2.33 ms a frame** at
// the operator's crowd load, over **4.83 `vkCmd*` calls a draw** â€” 52 ns a call
// (`phase5-notes.md` Â§6ed). The trampoline is a small constant fraction of each of those,
// which is why the item's ceiling is stated as 0.12-0.35 ms/frame and why the plan
// pre-registers what to do if it lands under the noise floor: KEEP IT AND REPORT A NULL.
// It is strictly less work for identical behaviour.
//
// THE ONE FAILURE MODE, AND IT IS LOUD ON PURPOSE. `vkGetDeviceProcAddr` returns
// `nullptr` for a name the device does not support or that is misspelled, and calling a
// null function pointer is a segfault with no message attached to it. So every name is
// resolved in ONE place, the resolved count is printed, and a null ABORTS with the name
// in the message. A silent fallback to the loader symbol would be worse than a crash: it
// would leave the arm partially engaged and make the measurement a blend of two
// configurations, which is gotcha 151 exactly.
//
// THE CONTROL ARM IS `CZ_VK_NO_DEVICE_PFN=1`, and it is a genuinely single-variable one.
// It resolves the SAME names through `vkGetInstanceProcAddr`, which for a device-level
// command returns the loader's trampoline â€” i.e. the function the exported symbol IS.
// Both arms therefore call through a stored pointer with identical argument marshalling,
// and the only difference between them is whether that pointer points at the trampoline
// or at the driver. (Today's code calls the exported symbol through the PLT, so the
// control arm is a hair cheaper than the code it stands in for; that direction is the
// safe one â€” it can only UNDER-state the saving.)
struct DeviceCommands
{
    PFN_vkCmdBindPipeline BindPipeline = nullptr;
    PFN_vkCmdSetViewport SetViewport = nullptr;
    PFN_vkCmdSetScissor SetScissor = nullptr;
    PFN_vkCmdSetBlendConstants SetBlendConstants = nullptr;
    PFN_vkCmdSetStencilReference SetStencilReference = nullptr;
    PFN_vkCmdSetStencilCompareMask SetStencilCompareMask = nullptr;
    PFN_vkCmdSetStencilWriteMask SetStencilWriteMask = nullptr;
    PFN_vkCmdBindDescriptorSets BindDescriptorSets = nullptr;
    PFN_vkCmdPushConstants PushConstants = nullptr;
    PFN_vkCmdBindVertexBuffers BindVertexBuffers = nullptr;
    PFN_vkCmdBindIndexBuffer BindIndexBuffer = nullptr;
    PFN_vkCmdDrawIndexed DrawIndexed = nullptr;
    PFN_vkCmdDraw Draw = nullptr;
};
DeviceCommands g_dfn;

// Called once, immediately after the device and queue exist. Before this point the
// macros below would dereference nulls, so nothing may record commands until it has run
// â€” which is structurally true anyway (there is no command buffer yet).
void ResolveDeviceCommands(VkDevice device, VkInstance instance)
{
    const bool viaLoader = EnvOn("CZ_VK_NO_DEVICE_PFN");
    int resolved = 0, missing = 0;
    auto get = [&](const char* name) -> PFN_vkVoidFunction {
        PFN_vkVoidFunction f = viaLoader ? vkGetInstanceProcAddr(instance, name)
                                         : vkGetDeviceProcAddr(device, name);
        if (f)
            ++resolved;
        else
        {
            ++missing;
            fprintf(stderr, "[vk] FATAL: could not resolve device command '%s'\n", name);
        }
        return f;
    };
#define CZ_RESOLVE(member, name) \
    g_dfn.member = reinterpret_cast<PFN_vk##name>(get("vk" #name))
    CZ_RESOLVE(BindPipeline, CmdBindPipeline);
    CZ_RESOLVE(SetViewport, CmdSetViewport);
    CZ_RESOLVE(SetScissor, CmdSetScissor);
    CZ_RESOLVE(SetBlendConstants, CmdSetBlendConstants);
    CZ_RESOLVE(SetStencilReference, CmdSetStencilReference);
    CZ_RESOLVE(SetStencilCompareMask, CmdSetStencilCompareMask);
    CZ_RESOLVE(SetStencilWriteMask, CmdSetStencilWriteMask);
    CZ_RESOLVE(BindDescriptorSets, CmdBindDescriptorSets);
    CZ_RESOLVE(PushConstants, CmdPushConstants);
    CZ_RESOLVE(BindVertexBuffers, CmdBindVertexBuffers);
    CZ_RESOLVE(BindIndexBuffer, CmdBindIndexBuffer);
    CZ_RESOLVE(DrawIndexed, CmdDrawIndexed);
    CZ_RESOLVE(Draw, CmdDraw);
#undef CZ_RESOLVE
    if (missing)
    {
        fprintf(stderr,
                "[vk] FATAL: %d of %d device commands did not resolve â€” refusing to run "
                "with a partially engaged command table\n",
                missing, resolved + missing);
        abort();
    }
    // AN ARM WITH NO COUNTER CANNOT BE SHOWN TO HAVE ENGAGED (gotcha 151), so both arms
    // say which one they are, unconditionally, on one line.
    fprintf(stderr,
            "[vk] device command table: %d commands resolved via %s%s\n", resolved,
            viaLoader ? "vkGetInstanceProcAddr â€” THE LOADER TRAMPOLINE"
                      : "vkGetDeviceProcAddr â€” the driver directly",
            viaLoader ? " (CZ_VK_NO_DEVICE_PFN=1, the control arm)" : "");
}

// From here to the end of the file every `vkCmd*` in the record path goes through the
// table. Done with macros RATHER than by editing 40 call sites for one reason: a macro
// cannot miss one. An edited call site that was overlooked would leave a call on the
// loader path, and the arm would be partially engaged with nothing to say so.
#define vkCmdBindPipeline g_dfn.BindPipeline
#define vkCmdSetViewport g_dfn.SetViewport
#define vkCmdSetScissor g_dfn.SetScissor
#define vkCmdSetBlendConstants g_dfn.SetBlendConstants
#define vkCmdSetStencilReference g_dfn.SetStencilReference
#define vkCmdSetStencilCompareMask g_dfn.SetStencilCompareMask
#define vkCmdSetStencilWriteMask g_dfn.SetStencilWriteMask
#define vkCmdBindDescriptorSets g_dfn.BindDescriptorSets
#define vkCmdPushConstants g_dfn.PushConstants
#define vkCmdBindVertexBuffers g_dfn.BindVertexBuffers
#define vkCmdBindIndexBuffer g_dfn.BindIndexBuffer
#define vkCmdDrawIndexed g_dfn.DrawIndexed
#define vkCmdDraw g_dfn.Draw

// ===================================================================================
// INTERNAL RESOLUTION SCALE â€” CZ_VK_RES / CZ_VK_RES_SCALE
// ===================================================================================
//
// The title renders at 1280x720 and nothing changes that: its vertex positions, its
// viewports, its scissors, its resolve extents and its texture fetches are all in guest
// pixels, and they are its own numbers. What this scales is the RASTERISATION TARGET â€”
// the host images the guest's draws land in â€” so the same geometry is sampled at more
// points. That is why it produces a sharper picture rather than a different one, and it
// is why the whole change fits on one side of a line that already exists in this file.
//
// **THE INVARIANT, and every edit below is an instance of it: a surface whose pixels
// come from the RENDER PIPELINE scales; a surface whose pixels come from GUEST MEMORY
// does not.** The EDRAM stand-in, the resolve snapshots, their right-sized views and the
// rendered cube map are the first kind. An uploaded texture is the second â€” there is no
// more data in guest memory than the guest put there, and inventing some would be a
// different feature (an upscaler) wearing this one's name.
//
// The consequence is that every guest coordinate has to be multiplied on its way to
// Vulkan and nowhere else. `edramWidth`/`edramHeight` therefore stay in GUEST pixels â€”
// they are the denominator of the window-coordinate-to-NDC mapping, which is
// resolution-independent by construction â€” while `R->color.width` is the host image's
// and is scaled. The two used to be the same number, so anywhere the old code compared a
// guest coordinate against `R->color.width` it now compares against `edramWidth`; those
// substitutions are identities at scale 1, which is what makes the default arm provably
// unchanged.
//
// WHAT THIS DOES NOT FIX, said out loud because it will be noticed. The title's post
// chain computes its blur taps from texel offsets IT supplies, in units of a 1280-wide
// surface. Those are NORMALISED offsets, so a tap still lands the same fraction of the
// screen away and a blur keeps its screen-space size â€” it is simply sampled at fewer
// taps per pixel than the artist intended. That is the ordinary, accepted outcome of
// resolution scaling and not a defect to chase.
//
// INTEGER MULTIPLES ONLY, and an unsupported request is refused loudly rather than
// rounded (gotcha 5). A non-integer scale would put a fractional factor into the tile
// scissors â€” this title renders in two 640-wide halves â€” and half a pixel of scissor
// error is a seam down the middle of the screen that no counter would report.
//
//   CZ_VK_RES=2560x1440   the resolution, said the way a player says it
//   CZ_VK_RES_SCALE=2     the same thing as a multiplier
// THE INTERNAL RESOLUTION â€” an explicit WIDTH x HEIGHT since operator revision 3
// (the menu lists the display's own modes, and 1920x1080 is no integer multiple of
// 1280x720). The renderer scales RATIONALLY and TRUNCATING: Y extents by H/720, X
// extents by W/1280 â€” floor(a)+floor(b) <= floor(a+b) holds for any fixed rational
// (gotcha 373), so a converted offset+extent can never overrun a converted surface,
// and the title's two 640-wide tile scissors stay exact for any EVEN width
// (640*W/1280 = W/2). 16:9 integer multiples reproduce the old integer path bit for
// bit. Latched at boot; changed only by ApplyPendingRenderScale between frames.
// 0 until the first InternalRes() call computes the boot value.
std::atomic<uint32_t> g_internalW{ 0 }, g_internalH{ 0 };
// True when an env var chose the resolution: the measurement arm wins and menu
// requests are refused loudly rather than silently overriding an A/B.
bool g_resScaleLocked = false;
std::atomic<uint32_t> g_resScalePending{ 0 };
// The explicit W x H form of the same request (part 91) â€” one word so a torn
// half-update cannot exist. 0 = nothing pending.
std::atomic<uint64_t> g_resWHPending{ 0 };

void InternalRes(uint32_t& w, uint32_t& h)
{
    w = g_internalW.load(std::memory_order_relaxed);
    h = g_internalH.load(std::memory_order_relaxed);
    if (w && h)
        return;
    uint32_t bw = 0, bh = 0;
    // CZ_VK_RES=WxH â€” any resolution the store validates (even width, H 720..2880,
    // at least 16:10 since part 108). The old integer-multiple forms still parse, so every recipe
    // in the docs is unchanged.
    if (const char* r = Env("CZ_VK_RES"))
    {
        uint32_t pw = 0, ph = 0;
        if (sscanf(r, "%ux%u", &pw, &ph) == 2 && Settings_ValidInternalRes(pw, ph))
        {
            g_resScaleLocked = true;
            bw = pw;
            bh = ph;
            // SAY SO. This arm changes every rasterisation target in the renderer and
            // printed NOTHING when it worked â€” only when it was rejected â€” so a session
            // quoting it had no way to show it engaged, and part 71's harness gate for it
            // could not have matched anything (gotcha 408, found by writing the gate
            // first). The settings path a few lines below has always announced itself.
            fprintf(stderr, "[vk] internal resolution %ux%u from CZ_VK_RES (env wins over "
                            "cz_settings.txt)\n", bw, bh);
        }
        else
            fprintf(stderr, "[vk] CZ_VK_RES=%s is not a resolution this renderer can "
                            "produce (even width, height 720..2880, at least 16:10) â€” "
                            "IGNORED, rendering at 1280x720.\n", r);
    }
    if (!bw)
        if (const char* n = Env("CZ_VK_RES_SCALE"))
        {
            const long v = strtol(n, nullptr, 10);
            if (v >= 1 && v <= 4)
            {
                g_resScaleLocked = true;
                bw = 1280 * uint32_t(v);
                bh = 720 * uint32_t(v);
            }
            else
                fprintf(stderr, "[vk] CZ_VK_RES_SCALE=%s is out of range (1..4) â€” "
                                "IGNORED.\n", n);
        }
    if (!bw)
    {
        // The persisted setting from the PC options screen. Env wins above so an
        // A/B arm can never be silently overridden by whatever the menu last wrote.
        Settings_InternalRes(bw, bh);
        // Clamped to the DISPLAY on load: a settings file written on one monitor
        // must not strand an impossible size on another. Env arms are deliberately
        // not clamped (measurement wins); unknown display (headless) clamps nothing.
        uint32_t dw = 0, dh = 0;
        if (Host_DisplaySize(&dw, &dh) && (bw > dw || bh > dh))
        {
            fprintf(stderr, "[vk] %ux%u from cz_settings.txt is larger than the "
                            "%ux%u display â€” clamped to 1280x720\n", bw, bh, dw, dh);
            bw = 1280;
            bh = 720;
        }
        if (bw != 1280 || bh != 720)
            fprintf(stderr, "[vk] internal resolution %ux%u from cz_settings.txt\n",
                    bw, bh);
    }
    // The legacy wide arms, applied ON TOP so their headless A/B recipes survive:
    // CZ_VK_WIDE=1 widens to the exact-21:9 width (the night's verified 21/16),
    // CZ_VK_WIDE=0 forces 16:9, CZ_VK_WIDE_NUM=n picks the numerator over 32.
    if (const char* e = Env("CZ_VK_WIDE"))
    {
        uint32_t num = atoi(e) != 0 ? 42u : 32u;
        if (const char* wn = Env("CZ_VK_WIDE_NUM"))
        {
            const long v = strtol(wn, nullptr, 10);
            if (v >= 33 && v <= 64 && num != 32)
                num = uint32_t(v);
        }
        bw = ((uint64_t(bh) * 1280 / 720) * num / 32) & ~1u;
        fprintf(stderr, "[vk] CZ_VK_WIDE=%s%s â€” internal %ux%u (env wins)\n", e,
                num != 32 && num != 42 ? " (CZ_VK_WIDE_NUM)" : "", bw, bh);
    }
    if (!Settings_ValidInternalRes(bw, bh))
    {
        bw = 1280;
        bh = 720;
    }
    g_internalW.store(bw, std::memory_order_relaxed);
    g_internalH.store(bh, std::memory_order_relaxed);
    w = bw;
    h = bh;
}
inline uint32_t InternalW() { uint32_t w, h; InternalRes(w, h); return w; }
inline uint32_t InternalH() { uint32_t w, h; InternalRes(w, h); return h; }

// The LEGACY integer view, for the parked live-scale seam and a few logs: the
// nearest whole multiple of 720 rows. New code converts extents with RS/RSX or the
// per-pass forms below, never with this.
uint32_t ResScale()
{
    return std::min(4u, std::max(1u, (InternalH() + 360) / 720));
}

// A guest extent, in host pixels â€” Y by H/720, X by W/1280, truncating (see the
// header comment above). The Pass* forms take an explicit internal resolution, for
// the shadow tier's per-pass size and for a snapshot's own build size.
inline uint32_t PassY(uint32_t v, uint32_t ph) { return uint32_t(uint64_t(v) * ph / 720); }
inline int32_t PassYi(int32_t v, uint32_t ph)
{
    return int32_t(int64_t(v) * int64_t(ph) / 720);
}
inline uint32_t PassX(uint32_t v, uint32_t pw)
{
    return uint32_t(uint64_t(v) * pw / 1280);
}
inline int32_t PassXi(int32_t v, uint32_t pw)
{
    return int32_t(int64_t(v) * int64_t(pw) / 1280);
}
inline uint32_t RS(uint32_t v) { return PassY(v, InternalH()); }
inline int32_t RSi(int32_t v) { return PassYi(v, InternalH()); }
inline uint32_t RSX(uint32_t v) { return PassX(v, InternalW()); }
inline int32_t RSXi(int32_t v) { return PassXi(v, InternalW()); }

// Wider than 16:9? Exact comparison on purpose: a 16:9 internal resolution must
// take the bit-identical legacy path (no projection patch, no UCP compensation).
inline bool WideMode()
{
    uint32_t w, h;
    InternalRes(w, h);
    return uint64_t(w) * 9 > uint64_t(h) * 16;
}
// NARROWER than 16:9 (part 108: the Steam Deck's 1280x800 and the 16:10 desktop
// modes). Exact comparison for the same reason: 16:9 stays on the untouched path.
inline bool NarrowMode()
{
    uint32_t w, h;
    InternalRes(w, h);
    return uint64_t(w) * 9 < uint64_t(h) * 16;
}
// 0 = 16:9, 1 = wide, 2 = narrow â€” the patch memo's key byte and every gate below.
inline uint8_t AspectPatchMode() { return WideMode() ? 1 : NarrowMode() ? 2 : 0; }
inline bool AspectPatchActive() { return AspectPatchMode() != 0; }
// The horizontal fov factor the wide frame carries relative to 16:9 at the same
// height: k = (W/1280) / (H/720) = 9W/16H. 1.0 at 16:9 by construction; BELOW 1 in
// narrow mode, where every use below reads it as the vertical factor's reciprocal.
inline float WideFovFactor()
{
    uint32_t w, h;
    InternalRes(w, h);
    return (9.0f * float(w)) / (16.0f * float(h));
}


// THE 16:9 SCENE PROJECTION, recognized structurally in a VS constant window's c0..c3.
// The shape is the one part 58's pose work measured (tools/pose_read.py,
// clip_plane_space.py): row 3 exactly (0,0,1,0) â€” w_clip = z_view, a perspective â€”
// rows 0/1 diagonal, and |xscale/yscale| = 9/16 exactly, because this title's scene
// cameras are all 16:9 whatever their fov. The shadow orthos (row 3 = (0,0,0,1)) and
// the CUBE FACE cameras (1:1 ratio) fail the test and stay untouched, which is what
// keeps shadows and reflections correct in wide mode.
inline bool Is169Perspective(const uint32_t* c)
{
    float m[16];
    memcpy(m, c, sizeof m);
    if (m[12] != 0.0f || m[13] != 0.0f || m[14] != 1.0f || m[15] != 0.0f)
        return false;
    if (m[1] != 0.0f || m[2] != 0.0f || m[3] != 0.0f || m[4] != 0.0f ||
        m[6] != 0.0f || m[7] != 0.0f || m[8] != 0.0f || m[9] != 0.0f)
        return false;
    if (m[0] == 0.0f || m[5] == 0.0f)
        return false;
    const float ratio = std::fabs(m[0] / m[5]);
    return std::fabs(ratio - 0.5625f) <= 0.002f;   // 9/16, small float slack
}

// THE SECOND FORM, found in part 62 by the miss-dump after the operator's live A/B
// proved the fov slider moved ONLY the UI: **the world's draws do not carry the raw
// projection at c0..c3 â€” they carry the full VIEW-PROJECTION COMPOSITE P*V**, with
// V's translation in the fourth column. Only ~2% of draws (UI, frontend scenes, a
// few effects) use the raw form; every world mesh rides the composite, which is why
// part 60's wide patch â€” raw-form only â€” left GAMEPLAY geometry stretched at 21:9
// while the frontend (where it was verified) was correct. The composite is
// recognizable because P's structure survives the product:
//   row0 = A * v0        (v0 = view rotation row, unit)  -> ||row0|| = A_eff
//   row1 = B * v1                                        -> ||row1|| = B_eff
//   row2 = p22 * v2 + p23 * (0,0,0,1)                    -> row2.xyz ~ 1.0001*row3.xyz
//   row3 = v2            (unit; w_clip = z_view)
// so: unit row3, ||row0||/||row1|| = 9/16 exactly, rows 0/1 orthogonal to row3, and
// row2.xyz proportional to row3.xyz with factor ~zf/(zf-zn) ~ 1. The shadow orthos
// and skinning affines fail on row3 = (0,0,0,1) (zero xyz norm) and the six cube
// face cameras fail on ratio 1:1 â€” measured in the same dump, entries 2-6/8-14.
// A_eff/B_eff vary per composite (the game applies a uniform xy zoom on top of the
// 45Â° base), which is why the fov math must use the composite's OWN B_eff.
//
// Returns 0 = not a scene transform, 1 = raw projection, 2 = composite; bEff is
// the |B| the fov arithmetic should use for this window.
inline int SceneXformForm(const uint32_t* c, float& bEff)
{
    if (Is169Perspective(c))
    {
        float b;
        memcpy(&b, c + 5, 4);
        bEff = std::fabs(b);
        return 1;
    }
    float m[16];
    memcpy(m, c, sizeof m);
    auto dot3 = [&](int r, int s) {
        return m[r * 4 + 0] * m[s * 4 + 0] + m[r * 4 + 1] * m[s * 4 + 1] +
               m[r * 4 + 2] * m[s * 4 + 2];
    };
    const float n3sq = dot3(3, 3);
    // 0.01, not 0.004 (part 108, item 0ad): the DOOR TRANSITION camera's view row has
    // norm 1.0024 (n3sq 1.0048) â€” the title scales that camera's view by a quarter of
    // a percent â€” and at 0.004 every world draw of the transition read "not a scene
    // transform", the wide patch never ran, and the whole doorway rendered as the
    // 16:9 frustum stretched to 21:9 until the roaming camera (norm 1.0000 exactly)
    // came back on the first movement. 1,161 of 1,161 world draws xf=0 in the
    // stretched F9, 1,153 of 1,153 xf=2 after the step. Orthos and affines read 0
    // here and the cube faces fail the 9/16 ratio below, so the wider band admits
    // nothing new. CZ_VK_XFORM_STRICT=1 restores 0.004, the control arm.
    static const float unitSlack = Env("CZ_VK_XFORM_STRICT") ? 0.004f : 0.01f;
    if (std::fabs(n3sq - 1.0f) > unitSlack)  // unit view row; orthos/affines are 0
        return 0;
    const float n0 = std::sqrt(dot3(0, 0)), n1 = std::sqrt(dot3(1, 1));
    if (!(n0 > 0.0f) || !(n1 > 0.0f))
        return 0;
    if (std::fabs(n0 / n1 - 0.5625f) > 0.002f)   // 9/16; cube faces read 1.0
        return 0;
    if (std::fabs(dot3(0, 3)) > 0.01f * n0 || std::fabs(dot3(1, 3)) > 0.01f * n1)
        return 0;                                // rows 0/1 must be âŠ¥ the view row
    const float f = dot3(2, 3);                  // row2.xyz = f * row3.xyz, f ~ p22
    if (f < 0.9f || f > 1.1f)
        return 0;
    for (int i = 0; i < 3; i++)
        if (std::fabs(m[8 + i] - f * m[12 + i]) > 0.01f)
            return 0;
    bEff = n1;
    return 2;
}

// Make a recognized scene transform correct for the wide surface. TWO FORMS, TWO
// TREATMENTS since the culling fix (part 62, Â§6cu):
//
// RAW form (UI, frontend scenes): divide A â€” the classic vert-plus widen, which
// is also what self-centers the UI (part 60). Unchanged.
//
// COMPOSITE form (the world): MULTIPLY ROW1 BY k (narrow the vertical) instead
// of dividing row0. Proportions on the wide surface are identical either way
// (A/B ends at 9/(16k) both ways); the difference is which absolute fov the
// surface shows â€” and the game-side fov substitution (cpu/camera_fov.cpp) now
// OVER-WIDENS the roaming camera by k in tan space, so narrowing the vertical
// back yields the same picture as before WITH the game's own 16:9 culling
// frustum covering the whole 21:9 view (the operator's flank pop-in, twice
// reported). Cameras that are NOT over-widened (cutscenes, minigames) come out
// as a constant-horizontal 21:9 CROP â€” the standard cinematic treatment, and
// crucially one whose visible region the game's frustum always covers, so the
// pre-existing cutscene flank gap closes too. Row1 scales WHOLE (translation
// component included â€” row1 = B * v1-with-translation).
//
// NARROW MODE (part 108, 16:10 â€” the Steam Deck's own panel) IS THE SAME DESIGN WITH
// THE AXES SWAPPED, and k < 1 there. RAW form: MULTIPLY B (the y scale) by k â€” the
// 16:9 frame occupies the central 9/10 of the height at full width, so the UI is
// letterboxed rather than cropped at the flanks, and a frontend scene reveals a band
// at top and bottom the way wide mode reveals the flanks. COMPOSITE form: DIVIDE ROW0
// BY k (narrow the horizontal back) â€” the game-side substitution widens the roaming
// camera by 1/k in tan space, so the horizontal returns to the 16:9 fov and the
// vertical keeps the extra 1/k: a vert-plus picture whose visible region the game's
// widened frustum covers exactly. Cameras that are NOT widened (cutscenes) come out as
// a constant-VERTICAL crop, again inside their own 16:9 frustum. Proportions are
// aspect-correct either way: on a W x H surface with a 16:9 projection the picture is
// stretched vertically by 1/k, and scaling x by 1/k or y by k both undo it.
inline int PatchWideProjection(uint32_t* c)
{
    float bEff;
    const int form = SceneXformForm(c, bEff);
    if (form == 0)
        return 0;
    const float k = WideFovFactor();
    const bool narrow = k < 1.0f;
    if (form == 1)
    {
        // Raw: wide divides A (x scale, c[0]); narrow multiplies B (y scale, c[5]).
        const int at = narrow ? 5 : 0;
        float a;
        memcpy(&a, c + at, 4);
        a = narrow ? a * k : a / k;
        memcpy(c + at, &a, 4);
        return 1;
    }
    // Composite: wide multiplies row1 by k; narrow divides row0 by k. Whole rows,
    // translation included.
    const int row = narrow ? 0 : 4;
    float m[4];
    memcpy(m, c + row, sizeof m);
    for (int i = 0; i < 4; i++)
        m[i] = narrow ? m[i] / k : m[i] * k;
    memcpy(c + row, m, sizeof m);
    return 2;
}

// The ratio B'/B the slider applies to a recognized projection's y scale:
// the game's vertical half-fov is atan(1/|B|), the patched one adds halfRad, and
// B' = 1/tan(patched half-fov). Clamped so the patched half-fov stays inside
// (0.5Â°, 89Â°) â€” outside that a tan pole would flip or explode the projection.
// Sign-safe: computed on |B|, and a RATIO carries no sign, so a y-flipped
// projection keeps its flip.
inline float FovScaleRatio(float b, float halfRad)
{
    const float ab = std::fabs(b);
    if (!(ab > 0.0f))
        return 1.0f;
    float half = std::atan(1.0f / ab) + halfRad;
    constexpr float kMinHalf = 0.0087266f;   // 0.5 degrees
    constexpr float kMaxHalf = 1.5533430f;   // 89 degrees
    if (half < kMinHalf)
        half = kMinHalf;
    if (half > kMaxHalf)
        half = kMaxHalf;
    return (1.0f / std::tan(half)) / ab;
}

// CZ_VK_FOV_CENSUS=1 â€” print every DISTINCT recognized projection (A, B, and the
// z-row terms m10/m11, which encode zn/zf) the first time it is seen. The question
// it answers (part 61): does the title's UI ride the SAME projection as the 3D
// scene, or a structurally distinguishable one? The wide patch WANTS to catch UI
// (that is the part-60 self-centering); a fov slider that catches UI shrinks the
// HUD, so if the two populations separate on a structural field, the fov patch can
// exempt the UI's. Env-gated and first-occurrence-only: free when off, and a
// bounded number of lines when on.
// CZ_VK_FOV_MISS=N â€” print the first N DISTINCT unrecognized VS windows' c0..c3
// as a 4x4, with row norms and the ||row0||/||row1|| ratio. The question (part 62):
// the operator's live A/B proved the fov patch does NOT move the world â€” so what
// DO the world draws carry at c0..c3? If it is a composed view-projection built
// from the known 45Â° P, then rows 0/1 are A- and B-scaled ROTATION rows:
// ||row0||/||row1|| = 9/16 exactly and row3 is unit â€” recognizable and patchable.
inline void FovMissDump(const uint32_t* c, uint32_t depthControl)
{
    static const long want = [] {
        const char* e = Env("CZ_VK_FOV_MISS");
        return e ? strtol(e, nullptr, 10) : 0;
    }();
    if (want <= 0)
        return;
    static std::mutex mu;
    static std::set<std::array<uint32_t, 4>> seen;
    std::lock_guard<std::mutex> lock(mu);
    if (long(seen.size()) >= want)
        return;
    const std::array<uint32_t, 4> key{ c[0], c[5], c[10], c[15] };
    if (!seen.insert(key).second)
        return;
    float m[16];
    memcpy(m, c, sizeof m);
    auto norm = [&](int r) {
        return std::sqrt(m[r * 4 + 0] * m[r * 4 + 0] + m[r * 4 + 1] * m[r * 4 + 1] +
                         m[r * 4 + 2] * m[r * 4 + 2]);
    };
    const float n0 = norm(0), n1 = norm(1), n3 = norm(3);
    fprintf(stderr,
            "[fov-miss] #%zu depth=%X ratio01=%.4f n0=%.4f n1=%.4f n3=%.4f\n"
            "  [% .4f % .4f % .4f % .4f]\n  [% .4f % .4f % .4f % .4f]\n"
            "  [% .4f % .4f % .4f % .4f]\n  [% .4f % .4f % .4f % .4f]\n",
            seen.size(), depthControl & 0x7, n1 > 0 ? n0 / n1 : 0.0f, n0, n1, n3,
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
            m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
}

// The ARM, hoisted out of `FovCensus` so part 71's per-frame hook fold can ask whether
// this census is live without paying for a call into it on every one of the ~33,000
// draws a frame at the operator's soak. Same static, same one-time `Env` read.
inline bool FovCensusArmed()
{
    static const bool on = Env("CZ_VK_FOV_CENSUS") != nullptr;
    return on;
}

inline void FovCensus(const uint32_t* c, uint32_t depthControl)
{
    if (!FovCensusArmed())
        return;
    float bEff = 0.0f;
    const int form = SceneXformForm(c, bEff);
    if (form == 0)
    {
        FovMissDump(c, depthControl);
        return;
    }
    static std::mutex mu;
    static std::map<std::array<uint32_t, 5>, uint64_t> seen;
    static uint64_t calls = 0;
    // Key: for the RAW form, the projection's A/B and z-row terms (bit-stable and
    // few). For the COMPOSITE form the matrix values change with the camera every
    // frame, so keying on them would grow the map without bound â€” composites
    // aggregate under a form marker. Both keys carry the draw's depth-test and
    // depth-write enables (RB_DEPTHCONTROL bits 1 and 2).
    const std::array<uint32_t, 5> key =
        form == 1 ? std::array<uint32_t, 5>{ c[0], c[5], c[10], c[11],
                                             depthControl & 0x6 }
                  : std::array<uint32_t, 5>{ 0xC0320051u, 0, 0, 0,
                                             depthControl & 0x6 };
    std::lock_guard<std::mutex> lock(mu);
    // The COMPOSITE's fov over time (part 108, item 0ad): composites aggregate under
    // one marker above, so a camera class with a different fov â€” the door transition
    // camera â€” was invisible to this census. Print when the composite's bEff CHANGES
    // by more than 0.5% from the last one printed, at most 20 lines a second (the
    // roaming camera smooths its fov during an aim, which would otherwise be a line
    // per draw). A door that renders at the wrong ratio should show up here as a
    // bEff the roaming camera never carries.
    if (form == 2)
    {
        static float lastB = 0.0f;
        static double secStart = 0.0;
        static int linesThisSec = 0;
        if (std::fabs(bEff - lastB) > 0.005f * std::max(bEff, 1e-6f))
        {
            const double t = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now().time_since_epoch()).count();
            if (t - secStart >= 1.0) { secStart = t; linesThisSec = 0; }
            if (linesThisSec < 20)
            {
                ++linesThisSec;
                fprintf(stderr, "[fov-composite] bEff %.4f -> %.4f (vfov %.2f -> %.2f deg)\n",
                        lastB, bEff, lastB > 0 ? 2.0 * std::atan(1.0 / lastB) * 57.29578 : 0.0,
                        2.0 * std::atan(1.0 / bEff) * 57.29578);
            }
            lastB = bEff;
        }
    }
    const bool fresh = ++seen[key] == 1;
    ++calls;
    if (fresh)
        fprintf(stderr, "[fov-census] #%zu form=%s bEff=%.6f (vfov=%.2f deg) "
                        "ztest=%u zwrite=%u\n", seen.size(),
                form == 1 ? "raw" : "composite", bEff,
                2.0 * std::atan(1.0 / bEff) * 57.29578,
                (depthControl >> 1) & 1, (depthControl >> 2) & 1);
    // The population counts, dumped periodically â€” first-occurrence lines name the
    // variants; this says how many draws each variant actually carries.
    if (calls % 200000 == 0)
        for (const auto& [k, n] : seen)
            fprintf(stderr, "[fov-census] count: A=%08X B=%08X z=%08X/%08X "
                            "depth=%X -> %llu draws\n", k[0], k[1], k[2], k[3],
                    k[4], (unsigned long long)n);
}

// Apply the slider to a recognized scene transform: both fov-carrying rows scale by
// one ratio so the aspect is untouched â€” which is also what keeps recognition true
// afterwards, so this composes with PatchWideProjection (ORDER: fov first, wide
// second; the wide patch then scales the x row alone). Same call sites and same
// memo/verify discipline as the wide patch above. The ratio is computed from the
// window's OWN effective B (composites carry a per-camera zoom on top of the 45Â°
// base), so the slider means the same thing whatever the game's camera is doing.
// Raw form: scale m[0] and m[5]. Composite: scale rows 0 and 1 whole (8 floats â€”
// the translation components are A- and B-scaled too, measured in the miss dump).
inline int PatchFovProjection(uint32_t* c, float halfRad)
{
    if (halfRad == 0.0f)
        return 0;
    float bEff;
    const int form = SceneXformForm(c, bEff);
    // COMPOSITE ONLY by default (part 62): the world rides the composite and the
    // HUD/UI ride the raw form, so patching only form 2 gives the slider exactly
    // the scope the operator asked for â€” the world changes, the HUD does not.
    // (Wide mode still patches both: its raw-form UI centering is a wanted
    // feature.) `CZ_VK_FOV_RAW=1` restores the part-61 both-forms behaviour â€” the
    // A/B arm, and the fallback if a raw-form SCENE element (sky? an effect?)
    // turns up misregistered against the fov-shifted world.
    static const bool rawToo = Env("CZ_VK_FOV_RAW") != nullptr;
    if (form == 0 || (form == 1 && !rawToo))
        return 0;
    const float r = FovScaleRatio(bEff, halfRad);
    if (r == 1.0f)
        return 0;
    float m[8];
    memcpy(m, c, sizeof m);
    if (form == 1)
    {
        m[0] *= r;
        m[5] *= r;
    }
    else
        for (int i = 0; i < 8; i++)
            m[i] *= r;
    memcpy(c, m, sizeof m);
    return form;
}

// ===================================================================================
// THE CONTENT GUARDS, ON OTHER CORES â€” part 53, plan item 1.1
// ===================================================================================
//
// WHY THIS EXISTS. Every performance plan this project has written makes the pump
// thread's work SMALLER. Part 52 shipped four such items and the operator confirmed
// them, and at the end of it the process was still using 2.24 of 16 cores with the pump
// 97.5% busy â€” thirteen cores idle while one thread was the frame. `GuardFold` is the
// largest symbol in that thread (26.3% of it with the instruments off, measured on the
// outdoor route at the open of this part; 20.0% on the operator's machine), and it is
// the one big cost here that is PURE: it reads guest memory and returns a `uint64_t`,
// calls no Vulkan, and mutates no renderer state. So it is the first thing to move.
//
// THE OBSTACLE, AND THE SHAPE OF THE ANSWER. The pump discovers which streams and
// textures a frame touches by walking the packet stream, so the work is discovered
// exactly when it is needed â€” there is nothing to hand a worker in advance. What makes
// it tractable is that the working set barely changes: part 22 measured 94-97% of stream
// BYTES byte-identical frame to frame, so the SET is stable even when the contents are
// not. Every guard the pump computes therefore also files a job for NEXT frame, and at
// the swap the whole list is dispatched to the pool. When the pump reaches that stream
// again it finds a finished hash instead of computing one.
//
// CORRECTNESS NEVER DEPENDS ON THE PREDICTION. A miss â€” a stream never seen before, a
// worker that has not finished, an entry that wanted the exact hash when only the
// sampled one was pre-computed â€” falls straight back to hashing inline. The prediction
// buys performance and nothing else.
//
// THE RACE, STATED HONESTLY, BECAUSE IT IS THE REAL COST OF THIS ITEM. The guard reads
// guest memory while the guest may be writing it, and that is true today: the inline
// guard has always raced. What changes is the WINDOW. Inline, the bytes are hashed at
// the moment the draw needs them; pre-hashed, they are hashed up to a frame earlier. A
// torn read still produces a different hash and reads as "changed", which is safe. What
// is NEW is that the pre-hash can see a COHERENT OLD state that matches the stored
// guard, where the inline hash would have seen the new bytes and re-copied â€” so a stream
// the guest rewrote mid-frame can be served one frame stale. That is the part-46 defect
// class (a stale mesh, a stale HUD), so it is not waved away: `CZ_VK_VERIFY_PARALLEL_GUARD=1`
// measures how often it actually happens, and it is what must be read before this is
// called free. `CZ_VK_NO_PARALLEL_GUARD=1` is the same-binary control arm.
//
// WHAT A WORKER MAY AND MAY NOT TOUCH. Workers never look at `persistCache` or
// `textures`: an `unordered_map` node's address is stable but an ERASE is not, and a
// worker holding a pointer into a cache the pump is editing is a use-after-free waiting
// for an unlucky frame. Instead each job is self-contained â€” a guest pointer, a length,
// a bound â€” and each result lands in a slot the pool owns. The cache entry keeps only
// an INDEX into that array plus the frame it belongs to, both written by the pump.

struct GuardJob
{
    const uint8_t* p = nullptr;
    uint64_t bytes = 0;
    uint32_t bound = 0;      // the sampled bound this subsystem folds at
    uint8_t wantExact = 0;   // also fold the WHOLE buffer (a promoted dynamic stream)
};

struct GuardOut
{
    uint64_t sampled = 0;
    uint64_t exact = 0;
    // THE DESCRIPTOR THE WORKER ACTUALLY HASHED, echoed back. A slot mix-up â€” entry A
    // handed entry B's hash â€” is the one failure here that would be silent and would
    // draw a wrong mesh, so the consumer checks that the slot it read describes the
    // buffer it asked about. Gotcha 342: ask what the wrong answer IS, not how likely.
    const uint8_t* p = nullptr;
    uint64_t bytes = 0;
    uint32_t haveExact = 0;
    std::atomic<uint32_t> done{ 0 };
};

// Heap-allocated once and never freed on purpose: the workers are detached and outlive
// static destruction, so a global with a destructor would be a teardown crash.
struct GuardPool
{
    std::mutex mx;
    std::condition_variable wake;    // workers wait here for a dispatch
    std::condition_variable idle;    // the pump waits here to drain before re-dispatching
    std::vector<GuardJob> jobs;      // the dispatch the workers are chewing on
    std::vector<GuardJob> gather;    // next frame's list, appended to as the pump walks
    GuardOut* out = nullptr;
    size_t outCap = 0;
    size_t count = 0;
    uint64_t generation = 0;
    std::atomic<size_t> next{ 0 };
    std::atomic<size_t> finished{ 0 };
    // Wall time the workers spent CHEWING, summed across the pool â€” part 89 step 0c's
    // occupancy input. Two unconditional clock reads per worker per DISPATCH (once a
    // frame, like the drain's), because "552 dispatches and 0 blocked" is a dispatch
    // count, not an occupancy figure, and the record-sharing question needs the second
    // (gotcha 151's shape: an idle-LOOKING count is not a measurement of idleness).
    std::atomic<uint64_t> busyNs{ 0 };
    unsigned workers = 0;
    bool started = false;
};
GuardPool* g_gp = nullptr;

// The census, printed with CZ_VK_PROFILE. `served` over `requests` is the hit rate the
// plan told this part to PRE-REGISTER: below ~80% the item is not working and any
// frame-time number taken from it is noise.
struct GuardPoolStats
{
    uint64_t requests = 0;     // guards the pump wanted
    uint64_t served = 0;       // ...answered by a finished pre-hash
    uint64_t bytesServed = 0;  // ...and the bytes that did not get hashed on the pump
    uint64_t missUnknown = 0;  // no job was filed for this buffer last frame
    uint64_t missPending = 0;  // filed, dispatched, but the worker has not finished
    uint64_t missVariant = 0;  // filed, but the exact hash was wanted and only sampled ran
    uint64_t mixups = 0;       // the slot described a DIFFERENT buffer â€” a real defect
    uint64_t dispatches = 0;
    uint64_t drainBlocked = 0; // dispatches that had to wait for the previous one
    uint64_t drainNs = 0;
    uint64_t verifyChecked = 0;
    uint64_t verifyStale = 0;  // the pre-hash disagreed with an inline hash taken now
};
GuardPoolStats g_gpStats;

// CZ_VK_VERIFY_PARALLEL_GUARD=1 â€” hash inline as well and compare, on every served
// guard. A disagreement is NOT necessarily a bug: it is the widened race above, and this
// is the instrument that says how wide. Costs the whole saving and then some, so it is
// an arm, never a default.
// CZ_VK_VERIFY_PARALLEL_GUARD_POISON=1 â€” perturb every pre-hash so the check MUST fire.
// A verify arm that has never been seen to fail has not been shown capable of it
// (gotcha 30).
bool g_gpVerify = false;
bool g_gpVerifyPoison = false;

void GuardRunJob(const GuardJob& j, GuardOut& o)
{
    const uint64_t seed = (1469598103934665603ull ^ j.bytes) * 1099511628211ull;
    o.sampled = GuardOver(j.p, size_t(j.bytes), j.bound, nullptr);
    if (j.wantExact)
        o.exact = GuardFold(seed, j.p, size_t(j.bytes));
    o.haveExact = j.wantExact;
    o.p = j.p;
    o.bytes = j.bytes;
    if (g_gpVerifyPoison)
    {
        o.sampled ^= 0x9E3779B97F4A7C15ull;
        o.exact ^= 0x9E3779B97F4A7C15ull;
    }
    o.done.store(1, std::memory_order_release);
}

// The parallel-record queue's worker hooks â€” defined with the rest of the machinery
// after the Renderer (they record Vulkan commands); the pool only needs these two.
bool ParRec_PendingChunks();
void ParRec_WorkerDrain(uint32_t workerIdx);
// ...and the same two hooks for the part-111 shared-block pre-zero (item B1).
bool Prezero_Pending();
void Prezero_WorkerDrain();

void GuardWorker(unsigned workerIdx)
{
    { char n[16]; snprintf(n, sizeof n, "cz-guard%u", workerIdx); ThreadBudget_NameSelf(n); }
    GuardPool& gp = *g_gp;
    uint64_t seen = 0;
    for (;;)
    {
        {
            std::unique_lock<std::mutex> lk(gp.mx);
            gp.wake.wait(lk, [&] {
                return gp.generation != seen || ParRec_PendingChunks() ||
                       Prezero_Pending();
            });
            seen = gp.generation;
        }
        const uint64_t busy0 = NowNs();
        // Record chunks FIRST: a chunk blocks THIS frame's submit where a guard job
        // is a prediction for the next frame â€” the priorities are not symmetric.
        ParRec_WorkerDrain(workerIdx);
        // ...then the pre-zero, which is for THIS frame and which the pump will start
        // asking for at its first draw. A guard job is a prediction for the NEXT frame,
        // so it is the lowest priority of the three.
        Prezero_WorkerDrain();
        for (;;)
        {
            const size_t i = gp.next.fetch_add(1, std::memory_order_relaxed);
            if (i >= gp.count)
                break;
            GuardRunJob(gp.jobs[i], gp.out[i]);
            // The pump polls `done` per slot and never waits on this, so the only
            // consumer of `finished` is the drain at the next dispatch.
            if (gp.finished.fetch_add(1, std::memory_order_acq_rel) + 1 == gp.count)
            {
                std::lock_guard<std::mutex> lk(gp.mx);
                gp.idle.notify_all();
            }
            // ...and between guard jobs, so a chunk never waits behind a whole
            // dispatch of hashes.
            if (ParRec_PendingChunks())
                ParRec_WorkerDrain(workerIdx);
            if (Prezero_Pending())
                Prezero_WorkerDrain();
        }
        // A worker that woke to an already-drained list adds its ~0, which is correct:
        // that IS its busy time for the dispatch.
        gp.busyNs.fetch_add(NowNs() - busy0, std::memory_order_relaxed);
    }
}

// 0 = off (the control arm). Otherwise the number of workers.
unsigned GuardPoolWorkers()
{
    static const unsigned n = [] () -> unsigned {
        if (EnvOn("CZ_VK_NO_PARALLEL_GUARD"))
            return 0;
        // FOUR, not sixteen, and the reason is in the plan: the PM4 walk is inherently
        // serial, so this pool is not a way to use the machine â€” it is a way to hide one
        // memory-latency-bound loop behind the walk. Four independent streams of misses
        // is most of what a single core's line-fill buffers cannot overlap; past that
        // the dispatch and the cache traffic start to cost more than they hide.
        //
        // FOUR IS NOW A REQUEST RATHER THAN A DECISION (part 55). The number this pool
        // gets is whatever the runtime-wide budget can spare, because part 55 proposes
        // two more pools and three pools each sizing themselves off
        // `hardware_concurrency()` is how a six-core machine ends up running twelve
        // workers. On the operator's 8-physical-core box the budget is 3, so this asks
        // for 4 and is CLAMPED to 3 â€” a deliberate, measured cost of leaving the machine
        // usable, and `CZ_VK_GUARD_WORKERS=4` is the arm that restores part 53's number.
        // See runtime/cpu/thread_budget.h for the policy and why it counts PHYSICAL
        // cores; the old line here was `hw >= 6 ? 4 : (hw >= 3 ? 2 : 0)` against a
        // logical count, which on this machine read 16 for 8 cores.
        return ThreadBudget_Take("guard", 4, "CZ_VK_GUARD_WORKERS");
    }();
    return n;
}

// Wait for the previous dispatch to finish. This blocks the PUMP, so it is counted:
// a pool that is still working when the next frame swaps is a pool that is too small,
// and that has to be visible rather than merely slow.
void GuardPoolDrain()
{
    GuardPool& gp = *g_gp;
    if (gp.finished.load(std::memory_order_acquire) >= gp.count)
        return;
    ++g_gpStats.drainBlocked;
    // An unconditional clock read, not ProfNow(): this is a stall the pump pays whether
    // or not anyone asked for a profile, and a number that only exists under an
    // instrument cannot be used to size the pool.
    const auto t0 = std::chrono::steady_clock::now();
    {
        std::unique_lock<std::mutex> lk(gp.mx);
        gp.idle.wait(lk,
                     [&] { return gp.finished.load(std::memory_order_acquire) >= gp.count; });
    }
    g_gpStats.drainNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - t0)
                                      .count());
}

// Called from the swap, right after the frame counter moves. Hands the list the frame
// just ended built to the workers and starts them.
void GuardPoolDispatch()
{
    const unsigned n = GuardPoolWorkers();
    // Report once the grant exists. `ThreadBudget_Report` prints only when something has
    // changed since the last call, so this is a single predictable branch per swap and
    // the line that matters â€” the final allocation, with this pool's share in it â€” is
    // the last `[threads]` line in the log rather than the first.
    ThreadBudget_Report();
    if (!n)
        return;
    if (!g_gp)
    {
        g_gp = new GuardPool();
        g_gp->workers = n;
        g_gpVerify = EnvOn("CZ_VK_VERIFY_PARALLEL_GUARD");
        g_gpVerifyPoison = EnvOn("CZ_VK_VERIFY_PARALLEL_GUARD_POISON");
    }
    GuardPool& gp = *g_gp;
    if (gp.gather.empty())
        return;
    GuardPoolDrain();
    gp.jobs.swap(gp.gather);
    gp.gather.clear();
    gp.count = gp.jobs.size();
    if (gp.count > gp.outCap)
    {
        // Grown, never shrunk, and only here â€” where the pool is provably idle, because
        // the drain above just ran. `GuardOut` holds an atomic and is therefore neither
        // copyable nor movable, so this is a plain array rather than a vector.
        delete[] gp.out;
        gp.outCap = gp.count + gp.count / 2 + 64;
        gp.out = new GuardOut[gp.outCap];
    }
    for (size_t i = 0; i < gp.count; ++i)
        gp.out[i].done.store(0, std::memory_order_relaxed);
    gp.next.store(0, std::memory_order_relaxed);
    gp.finished.store(0, std::memory_order_release);
    ++g_gpStats.dispatches;
    {
        std::lock_guard<std::mutex> lk(gp.mx);
        ++gp.generation;
    }
    gp.wake.notify_all();
    if (!gp.started)
    {
        gp.started = true;
        for (unsigned i = 0; i < n; ++i)
            std::thread(GuardWorker, i).detach();
    }
}

// File a job for NEXT frame and return the slot it will land in. Called by the pump at
// the moment it computes (or reuses) a guard, which is exactly the order the next frame
// will ask for them in â€” so the shared job counter hands the earliest-needed buffers out
// first, and a worker pool that cannot finish the whole list still finishes the useful
// end of it.
uint32_t GuardPoolFile(const uint8_t* p, uint64_t bytes, uint32_t bound, bool wantExact)
{
    if (!GuardPoolWorkers() || !g_gp)
        return UINT32_MAX;
    GuardPool& gp = *g_gp;
    if (gp.gather.size() >= 65535)
        return UINT32_MAX;
    gp.gather.push_back(GuardJob{ p, bytes, bound, uint8_t(wantExact) });
    return uint32_t(gp.gather.size() - 1);
}

// The consumer side. Returns the finished result for `slot`, or nullptr with the reason
// counted. `stamp` is the frame the slot was filed FOR; a slot from any other frame is
// not this frame's answer.
const GuardOut* GuardPoolTake(uint32_t slot, uint64_t stamp, uint64_t frame,
                              const uint8_t* p, uint64_t bytes, bool needExact)
{
    if (!GuardPoolWorkers() || !g_gp)
        return nullptr;
    ++g_gpStats.requests;
    GuardPool& gp = *g_gp;
    if (slot == UINT32_MAX || stamp != frame || slot >= gp.count)
    {
        ++g_gpStats.missUnknown;
        return nullptr;
    }
    GuardOut& o = gp.out[slot];
    if (!o.done.load(std::memory_order_acquire))
    {
        ++g_gpStats.missPending;
        return nullptr;
    }
    if (o.p != p || o.bytes != bytes)
    {
        // A slot describing another buffer means the index bookkeeping is wrong, and the
        // consequence would be a real, wrong hash â€” the shape of the part-52 memo defect.
        // Loud, capped, and it falls back to hashing inline so the picture stays right.
        if (g_gpStats.mixups < 8)
            fprintf(stderr,
                    "[vk] PARALLEL GUARD SLOT MIX-UP #%llu: slot %u was filed for "
                    "%p/%llu but was claimed for %p/%llu\n",
                    (unsigned long long)g_gpStats.mixups + 1, slot, (const void*)o.p,
                    (unsigned long long)o.bytes, (const void*)p,
                    (unsigned long long)bytes);
        ++g_gpStats.mixups;
        return nullptr;
    }
    if (needExact && !o.haveExact)
    {
        ++g_gpStats.missVariant;
        return nullptr;
    }
    ++g_gpStats.served;
    return &o;
}

// ===================================================================================
// Guest memory, again
// ===================================================================================
// Same convention as pm4.cpp and for the same reason: the PPC_LOAD macros need a
// `base` named exactly that in scope, and keeping the accessors local is what lets
// this file be read without the recompiled image in view.
constexpr uint32_t kPhysArenaBase = 0xA0000000u;
constexpr uint32_t kPhysArenaEnd = 0xBFFF0000u;

inline uint32_t PhysToVa(uint32_t addr) { return kPhysArenaBase | (addr & 0x1FFFFFFFu); }

// True when [va, va+bytes) is inside the physical arena. Every guest pointer the
// register file hands us goes through this: a fetch constant left over from a previous
// frame can name anything at all, and a memcpy from it is a host segfault attributed
// to our renderer rather than to the stale register it came from.
// The pose capture's player half â€” defined in cpu/debug_tunables.cpp, which owns the
// guest pointer and guest memory access. Declared rather than headered because that
// translation unit has no header and one function does not justify inventing one.
extern "C" uint32_t CZ_DebugWritePlayerObject(FILE* f, uint32_t bytes);
// The player's world position, read through the title's own `getplayerinfo` path on a
// GUEST thread and cached; this returns the cache plus its age, because the render
// thread must not make guest calls. Age travels with the value so a stale one cannot
// pass for fresh.
extern "C" int CZ_DebugPlayerPos(float out[3], long long* ageMs);

bool GuestRangeOk(uint32_t va, uint64_t bytes)
{
    return bytes && va >= kPhysArenaBase && uint64_t(va) + bytes <= kPhysArenaEnd;
}

// The GPU's per-address endian swizzle: 0 none, 1 = 8-in-16, 2 = 8-in-32, 3 = 16-in-32.
void CopySwapped(uint8_t* dst, const uint8_t* src, size_t bytes, uint32_t endian)
{
    switch (endian & 3)
    {
        case 1:
            for (size_t i = 0; i + 1 < bytes; i += 2)
            {
                dst[i] = src[i + 1];
                dst[i + 1] = src[i];
            }
            break;
        case 2:
            for (size_t i = 0; i + 3 < bytes; i += 4)
            {
                uint32_t v;
                memcpy(&v, src + i, 4);
                v = __builtin_bswap32(v);
                memcpy(dst + i, &v, 4);
            }
            break;
        case 3:
            for (size_t i = 0; i + 3 < bytes; i += 4)
            {
                uint32_t v;
                memcpy(&v, src + i, 4);
                v = (v >> 16) | (v << 16);
                memcpy(dst + i, &v, 4);
            }
            break;
        default:
            memcpy(dst, src, bytes);
            break;
    }
}

inline float F32(uint32_t bits)
{
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// ===================================================================================
// Vertex formats
// ===================================================================================
// The Xenos format code, as the vertex fetch instruction carries it, to the Vulkan
// vertex format that reads the same bytes after the whole stream has been dword
// swapped. `numFormat` 0 means a normalized fraction and 1 means an integer kept as a
// float value, which is a different Vulkan format, not a shader-side difference.
//
// UNDEFINED IS NOT A FALLBACK. An unmapped format used to fall through as
// VK_FORMAT_UNDEFINED on the previous port and the pipeline drew anyway â€” undefined
// behaviour that eventually took the device down. Here it refuses the pipeline and
// names the format once, which turns "some geometry is missing" into a line of log.
VkFormat XenosVertexFormat(uint32_t fmt, bool isSigned, bool isInteger)
{
    // `numFormat` 1 â€” "integer kept as a float value" â€” is NOT a shader-side detail,
    // and treating it as one is a silent, total corruption of whatever the attribute
    // carries. A normalized format divides by the type's range, so an integer 32
    // arrives as 32/255 = 0.125, and a shader that does `floor()` on it to index
    // something reads element 0 every time. Case Zero has 15 such attributes; the
    // meshes that use them collapse to a vanishing point, which reads as scrambled
    // geometry rather than as a vertex-format bug.
    //
    // USCALED/SSCALED are exactly this concept in Vulkan â€” an integer in memory
    // delivered as its own value in a float input â€” so the shader needs no change and
    // the input stays float-typed, which a *_UINT format would not.
    switch (fmt)
    {
        case 6:
            if (isInteger)
                return isSigned ? VK_FORMAT_R8G8B8A8_SSCALED : VK_FORMAT_R8G8B8A8_USCALED;
            return isSigned ? VK_FORMAT_R8G8B8A8_SNORM : VK_FORMAT_R8G8B8A8_UNORM;
        case 25:
            if (isInteger)
                return isSigned ? VK_FORMAT_R16G16_SSCALED : VK_FORMAT_R16G16_USCALED;
            return isSigned ? VK_FORMAT_R16G16_SNORM : VK_FORMAT_R16G16_UNORM;
        case 26:
            if (isInteger)
                return isSigned ? VK_FORMAT_R16G16B16A16_SSCALED
                                : VK_FORMAT_R16G16B16A16_USCALED;
            return isSigned ? VK_FORMAT_R16G16B16A16_SNORM
                            : VK_FORMAT_R16G16B16A16_UNORM;
        case 31: return VK_FORMAT_R16G16_SFLOAT;
        case 32: return VK_FORMAT_R16G16B16A16_SFLOAT;
        // There is no 32-bit-normalized Vulkan vertex format, so both flavours of
        // k_32 get the integer type. Better than rejecting the draw.
        case 33: return isSigned ? VK_FORMAT_R32_SINT : VK_FORMAT_R32_UINT;
        // k_10_11_11 packed normals are decoded IN the shader, which takes the raw
        // dword â€” so the input must deliver the untouched 32 bits, not a normalized
        // format that would pre-decode them wrongly.
        //
        // R32_SFLOAT, not R32_UINT, because of what the SHADER-side input is typed as.
        // This title wraps every packed normal in a TEXCOORD usage, whose input variable
        // is float4; binding R32_UINT against that is a pipeline type mismatch
        // (VUID-VkGraphicsPipelineCreateInfo-Input-08733, ten pipelines on the outdoor
        // route) whose practical effect was the packed dword's bits read AS a float â€”
        // NaN whenever bits 30..23 are all ones. Those NaNs, laundered by the tone
        // epilogue's max/saturate into exactly rgb(180,180,180), were the white-surface
        // plateau (open-items 00f). An SFLOAT attribute is a plain 32-bit load, the bits
        // arrive intact, and the emitter's XeUnpack_10_11_11 recovers them with asuint â€”
        // see XenosRecomp shader_common.h. If a title ever declares fmt16 under a uint4
        // usage (Fable 2 wraps normals as NORMAL), that path wants R32_UINT again.
        case 16: return VK_FORMAT_R32_SFLOAT;
        case 7:
            if (isInteger)
                return VK_FORMAT_R32_UINT;
            return isSigned ? VK_FORMAT_A2B10G10R10_SNORM_PACK32
                            : VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case 36: return VK_FORMAT_R32_SFLOAT;
        case 37: return VK_FORMAT_R32G32_SFLOAT;
        case 57: return VK_FORMAT_R32G32B32_SFLOAT;
        case 38: return VK_FORMAT_R32G32B32A32_SFLOAT;
        default: return VK_FORMAT_UNDEFINED;
    }
}

// Dwords occupied by one element of a vertex format. Used only for bounds checking,
// and it is deliberately NOT the stride: a stream's last vertex only has to reach
// `offset + element`, and buffers are commonly sized exactly that tightly, so a
// `vertices * stride` bound reports an overrun on perfectly good geometry.
uint32_t VertexFormatDwords(uint32_t fmt)
{
    switch (fmt)
    {
        case 6: case 7: case 16: case 25: case 31: case 33: case 36: return 1;
        case 26: case 32: case 37: return 2;
        case 57: return 3;
        case 38: return 4;
        default: return 0;
    }
}

// CZ_VK_FETCH_SLOT_INVERT=1 â€” read vertex fetch constants at `95 - slot`.
//
// The arm for the one convention this renderer cannot derive. A vfetch's constant index
// is `const_index * 3 + const_index_sel`, and Xenia's disassembly prints the same
// shaders' fetches as vf0/vf1/vf2 where that formula gives 95/94/93 â€” so one of the two
// is a display convention. The first attempt to settle it (dumping the populated slots)
// was WEAKER than it looked: the shader it happened to catch asked for slot 0 twice,
// and both slot 0 and slot 95 were populated, so the observation was consistent with
// either reading. This is the version that cannot be ambiguous â€” invert it and look at
// the geometry.
uint32_t FetchSlot(uint32_t slot)
{
    static const bool invert = getenv("CZ_VK_FETCH_SLOT_INVERT") != nullptr;
    return invert ? (slot <= 95 ? 95 - slot : slot) : slot;
}

// ===================================================================================
// The shader cache
// ===================================================================================
struct VertexAttribute
{
    int32_t location = -1;   // -1 = a dependent fetch, read in-shader, not an input
    uint32_t fetchSlot = 0;
    uint32_t format = 0;
    uint32_t isSigned = 0;
    uint32_t isInteger = 0;
    uint32_t strideDwords = 0;
    uint32_t offsetDwords = 0;
    uint32_t indirect = 0;
};

struct ShaderMeta
{
    VkShaderModule module = VK_NULL_HANDLE;
    // ROUTE (B)'s VARIANT (part 65), from assets/shader_spv_rt: the same shader with
    // its shadow-atlas taps redirected to our screen-space factor. Null for the 323
    // shaders the census found do NOT sample the atlas, and null for every shader when
    // the RT cache is absent â€” the pipeline key only sets its RT bit when this is
    // non-null, so a missing variant cache degrades to the stock renderer rather than
    // to a shader that is not there.
    VkShaderModule moduleRt = VK_NULL_HANDLE;
    bool isVertex = false;
    std::vector<VertexAttribute> attributes; // vertex shaders only
    // HOW MUCH OF THE VFETCH TABLE THIS SHADER CAN READ (part 109). Derived from
    // `attributes` when the sidecar is parsed, never at draw time.
    //
    // The shared constant block is zeroed on EVERY draw â€” 2,192 bytes into
    // write-combined arena memory, ~20 MB a frame at the operator's crowd, and `perf`
    // puts `__memset_avx2` at 4.2% of the pump thread = 0.44 ms of a 10.8 ms frame.
    // 1,536 of those 2,192 bytes are the dependent-vertex-fetch table (96 slots x 16),
    // and **only a slot this shader actually declares can ever be read**: XenosRecomp's
    // `XeVfetchDep` addresses the table by the slot baked into the shader, so a slot no
    // attribute names is dead memory. 45 of this title's 67 vertex shaders declare NO
    // dependent fetch at all and read none of it.
    //
    // Zeroing is still load-bearing for the slots that ARE declared: the publish loop
    // `continue`s on a bad range or a failed upload, and the shader's own bounds check
    // then sees size 0 and returns float4(0,0,0,0) â€” the mesh collapses to the origin
    // rather than reading a stale address. So this is the highest declared slot plus
    // one, not the count of declared slots, and a shader with none gets zero.
    // The declared slots themselves, sorted and deduped â€” not just the highest one. A
    // PREFIX up to the highest slot saved 940 of 2,192 bytes a draw (42.9%); the shaders
    // that do use dependent fetches declare a handful of slots scattered up to ~37, so
    // zeroing the entries rather than the span is most of the rest.
    std::vector<uint16_t> vfetchSlots;
    std::vector<uint32_t> interpolators;
    std::vector<uint32_t> tfetchConsts;
    // PARALLEL to tfetchConsts: 0 = 1D, 1 = 2D, 2 = 3D, 3 = cube, and it decides which
    // of the four descriptor-index arrays in the shared constants this slot's index has
    // to be published into. Empty when the sidecar predates part 25, which the binder
    // counts rather than papering over â€” see the note at bindTextures.
    std::vector<uint32_t> tfetchDims;
    // THE OBJECT->WORLD TRANSFORM'S CONSTANT ROWS (part 67), from
    // config/rt_world_xform.json, which tools/rt_world_xform_census.py reads out of this
    // shader's own microcode.
    //
    // Why this has to exist at all: `rtshadow::Collect` gates a draw on
    // `SceneXformForm(c0..c3) == 2`, and Â§6cs read that as "so the position stream is
    // world-space". It is not. c0..c3 is the CAMERA's view-projection, and that is the
    // same matrix whether the shader feeds it a world position or an object position it
    // transformed one line earlier â€” which is what this title's world shaders do, from a
    // row-major 4x3 at vc(8..10). Measured over the twenty `.xtr` world traces, 46,820
    // accepted draws: 100% of them carry a NON-IDENTITY world translation, and the
    // fraction whose bounding box intersects the frustum it was drawn into goes from
    // 0.1% untransformed to 97.8% placed. Our BLASes held local geometry and every TLAS
    // instance carried an identity transform, so the whole town was piled at the origin
    // â€” which is exactly the "97.3% of receivers fully open" part 66 measured.
    //
    // Two shapes exist and both are read from the microcode rather than assumed: a
    // `direct` 4x3 at vc(base..base+2), and a `palette` blend of vc(base + i) entries by
    // three per-vertex weights (whose entry 0 is what the static world draws use â€” see
    // xfPalette, which is counted so the approximation is never invisible). A shader
    // with no entry is NOT placed at the origin on a guess; it is declined and counted.
    uint8_t xfCount = 0;        // 0 = no table entry (or a genuinely world-space stream)
    uint8_t xfBase[2] = {};     // innermost stage first
    uint8_t xfPalette = 0;      // bit i set = stage i is a palette blend
    bool xfKnown = false;       // the table HAD an entry (an empty one means "camera")
    // THE PALETTE BLEND'S OWN INPUTS (`"blend"` in the same table).
    //
    // "entry 0 with unit weight" is not an approximation that is right for the static
    // world and wrong for actors, which is what Â§6cx assumed. Part 69's census of
    // hardware's own index streams reads ZERO of 2,786 palette draws referencing a
    // single matrix, with a median of 19 distinct entries â€” so every batched prop in
    // such a draw is being placed on top of whichever prop happens to be bone 0. The
    // placement is PER VERTEX, and these four fields are what makes doing it possible:
    // where in the vertex buffer the weights and the matrix indices live.
    //
    // Layout, not binding: twelve of the bank's eighteen palette shaders take these
    // bytes as declared attributes and six through a dependent fetch, and it is one
    // interleaved buffer either way.
    // PERF ITEM C â€” the ALU constant registers this shader actually READS
    // (tools/alu_const_sidecar.py, from the XenosRecomp HLSL at cache-build time).
    //
    // The renderer copies the guest's whole 256-float4 window per stage per draw: 4,096
    // bytes, and the constant memo reaches only ~61% of pixel windows and 2.9% of vertex
    // ones because the guest rewrites a world matrix per object. Over all 449 modules the
    // median shader reads **25** registers and the MAXIMUM is **56** â€” so the gather is
    // bounded at 896 bytes against 4,096, and that bound is a fact about the bank rather
    // than an average to hope for.
    //
    // `aluDynamic` marks the 22 vertex shaders that index `a0`-relatively (the bone
    // palette, `vc(8/9/10 + a0)`). An `a0` read can land anywhere, so no list can bound it
    // and those keep the full copy.
    //
    // THIS LIST IS LOAD-BEARING FOR CORRECTNESS: a register it omits is one that is never
    // copied, so the shader would read whatever the bump arena left in that slot â€” garbage,
    // not a stale value. `CZ_VK_VERIFY_CONST_GATHER=1` is the arm that makes that
    // believable and `CZ_VK_GATHER_POISON=1` is what proves the arm can fire.
    std::vector<uint32_t> aluConsts;
    bool aluDynamic = true;      // absent sidecar entry => full copy, never a guess
    // Whether the sidecar CARRIED a list at all. An empty list from a sidecar that has
    // the key means "reads no constants" (11 modules); an empty list from one that does
    // not means "we do not know". The copy path must not merge them.
    bool aluListKnown = false;
    // Part 88: true when EVERY dynamic expression in the sidecar is `N + a0` with N in
    // [8, 10] â€” the bone palette. Only these take the write-extent-bounded copy; the
    // one outlier (`vc(209+a0)`) and any sidecar carrying no exprs keep the full copy,
    // because the bound is the extent of writes ANCHORED AT c8 and says nothing about
    // a palette living anywhere else in the window.
    bool aluDynPalette = false;
    uint8_t blendSlot = 0, blendStrideDw = 0, blendWeightOffDw = 0, blendIndexOffDw = 0;
    uint8_t blendBytes[4] = {};   // which component of each dword, per influence
    uint8_t blendCount = 0;       // 0 = no descriptor; the entry-0 fallback stands
};

// A deliberately small JSON reader for a file this project writes itself.
//
// Pulling in a JSON library for four key names would be the larger risk: the sidecar's
// shape is fixed by tools/synth_shader_container.py, both ends live in this repo, and a
// malformed sidecar is a build-pipeline bug that should be loud here rather than
// tolerated. Anything unrecognised is ignored, and a missing sidecar drops the shader
// with a message (never silently â€” an orphaned .spv cost the previous port 25,364
// draws a run before anyone noticed the module count was seven short).
struct Json
{
    const std::string& s;
    size_t p = 0;

    explicit Json(const std::string& text) : s(text) {}

    void Skip()
    {
        while (p < s.size() && (isspace(uint8_t(s[p])) || s[p] == ',' || s[p] == ':'))
            ++p;
    }
    bool Find(const char* key, size_t from = 0)
    {
        const std::string pat = std::string("\"") + key + "\"";
        const size_t at = s.find(pat, from);
        if (at == std::string::npos)
            return false;
        p = at + pat.size();
        Skip();
        return true;
    }
    long Number()
    {
        Skip();
        return strtol(s.c_str() + p, nullptr, 10);
    }
};

long JsonIntField(const std::string& obj, const char* key, long fallback)
{
    Json j(obj);
    return j.Find(key) ? j.Number() : fallback;
}

std::vector<uint32_t> JsonIntArray(const std::string& s, const char* key)
{
    std::vector<uint32_t> out;
    Json j(s);
    if (!j.Find(key))
        return out;
    const size_t open = s.find('[', j.p);
    const size_t close = s.find(']', open);
    if (open == std::string::npos || close == std::string::npos)
        return out;
    const char* c = s.c_str() + open + 1;
    const char* end = s.c_str() + close;
    while (c < end)
    {
        while (c < end && !isdigit(uint8_t(*c)) && *c != '-')
            ++c;
        if (c >= end)
            break;
        out.push_back(uint32_t(strtol(c, const_cast<char**>(&c), 10)));
    }
    return out;
}

// THE WORLD-TRANSFORM TABLE (part 67): shader hash -> the VS constant rows carrying its
// object->world matrix. One file for ALL caches, because the table is a property of the
// microcode and not of a variant â€” and because six sibling caches each carrying their own
// copy is exactly the drift gotcha 390 cost three parts of operator sessions.
//
// Read with the same deliberately small reader the sidecars use: find the shader's own
// key, then the next "stages" string, which is `kind@base` pairs innermost first.
std::string g_worldXformText;
bool g_worldXformLoaded = false;
uint32_t g_worldXformEntries = 0;

void LoadWorldXformTable(const std::filesystem::path& shaderDir)
{
    if (g_worldXformLoaded)
        return;
    g_worldXformLoaded = true;
    // HostPaths::Config() first (release-plan A.1: anchored to the executable, not the
    // CWD). The sibling-of-the-shader-cache candidate stays because CZ_SHADER_SPV can
    // point the cache somewhere else entirely, and the CWD-relative ones stay because
    // they cost nothing and several recorded recipes rely on them.
    std::vector<std::filesystem::path> candidates = {
        HostPaths::Config() / "rt_world_xform.json",
        shaderDir.parent_path().parent_path() / "config" / "rt_world_xform.json",
        "config/rt_world_xform.json",
        "../config/rt_world_xform.json",
        "../../config/rt_world_xform.json",
    };
    for (const auto& c : candidates)
    {
        std::ifstream f(c);
        if (!f)
            continue;
        g_worldXformText.assign((std::istreambuf_iterator<char>(f)), {});
        for (size_t at = g_worldXformText.find("\"stages\""); at != std::string::npos;
             at = g_worldXformText.find("\"stages\"", at + 1))
            ++g_worldXformEntries;
        fprintf(stderr, "[vk] world-transform table: %s (%u shaders)\n",
                c.string().c_str(), g_worldXformEntries);
        return;
    }
    // Loud, once, because a missing table is not a missing feature: it is RT shadows
    // silently reverting to the part-66 behaviour of piling the town at the origin.
    fprintf(stderr,
            "[vk] no config/rt_world_xform.json â€” RT shadow occluders cannot be PLACED. "
            "Regenerate with tools/rt_world_xform_census.py\n");
}

// Fill in this shader's stages from the table. Absent = declined, never defaulted.
void ApplyWorldXform(const std::string& name, ShaderMeta& meta)
{
    if (g_worldXformText.empty())
        return;
    const std::string pat = "\"" + name + "\"";
    const size_t at = g_worldXformText.find(pat);
    if (at == std::string::npos)
        return;
    const size_t st = g_worldXformText.find("\"stages\"", at);
    if (st == std::string::npos)
        return;
    const size_t open = g_worldXformText.find('"', st + 8);
    const size_t close = open == std::string::npos
                             ? std::string::npos
                             : g_worldXformText.find('"', open + 1);
    if (close == std::string::npos)
        return;
    meta.xfKnown = true;
    const std::string stages = g_worldXformText.substr(open + 1, close - open - 1);
    size_t p = 0;
    while (p < stages.size() && meta.xfCount < 2)
    {
        const size_t comma = stages.find(',', p);
        const std::string tok = stages.substr(p, comma == std::string::npos
                                                     ? std::string::npos
                                                     : comma - p);
        const size_t at2 = tok.find('@');
        if (at2 != std::string::npos)
        {
            if (tok.compare(0, at2, "palette") == 0)
                meta.xfPalette |= uint8_t(1u << meta.xfCount);
            meta.xfBase[meta.xfCount++] =
                uint8_t(strtoul(tok.c_str() + at2 + 1, nullptr, 10));
        }
        if (comma == std::string::npos)
            break;
        p = comma + 1;
    }
    // "blend": "slot:strideDw:weightOffDw:indexOffDw:bytes" â€” five fields, the last a
    // digit per influence. Parsed positionally rather than by name because it is one
    // short generated string with both ends in this repo, the same contract as "stages".
    // Bounded by THIS entry's closing brace, not by a character count. The first
    // version used `close + 64` and that is not safe: a shader with no blend sits about
    // forty characters from the NEXT shader's, so it would have inherited its
    // neighbour's vertex layout â€” a silently misplaced mesh with nothing to say so.
    const size_t brace = g_worldXformText.find('}', close);
    const size_t bl = g_worldXformText.find("\"blend\"", close);
    if (bl == std::string::npos || brace == std::string::npos || bl > brace)
        return;
    const size_t bo = g_worldXformText.find('"', bl + 7);
    const size_t bc = bo == std::string::npos ? std::string::npos
                                             : g_worldXformText.find('"', bo + 1);
    if (bc == std::string::npos)
        return;
    const std::string blend = g_worldXformText.substr(bo + 1, bc - bo - 1);
    unsigned f[4] = {};
    size_t q = 0;
    int got = 0;
    for (; got < 4 && q < blend.size(); ++got)
    {
        f[got] = unsigned(strtoul(blend.c_str() + q, nullptr, 10));
        const size_t colon = blend.find(':', q);
        if (colon == std::string::npos)
            break;
        q = colon + 1;
    }
    if (got != 4 || q >= blend.size())
        return;
    meta.blendSlot = uint8_t(f[0]);
    meta.blendStrideDw = uint8_t(f[1]);
    meta.blendWeightOffDw = uint8_t(f[2]);
    meta.blendIndexOffDw = uint8_t(f[3]);
    for (size_t k = q; k < blend.size() && meta.blendCount < 4; ++k)
        if (blend[k] >= '0' && blend[k] <= '3')
            meta.blendBytes[meta.blendCount++] = uint8_t(blend[k] - '0');
}

bool LoadShaderMeta(const std::filesystem::path& path, ShaderMeta& meta)
{
    std::ifstream f(path);
    if (!f)
        return false;
    const std::string text((std::istreambuf_iterator<char>(f)), {});

    meta.isVertex = text.find("\"vs\"") != std::string::npos;
    meta.interpolators = JsonIntArray(text, "interpolators");
    // Perf item C. TWO DIFFERENT STATES THAT LOOK THE SAME THROUGH `JsonIntArray`, and
    // conflating them was a real defect in the first version of this:
    //
    //   * the KEY IS ABSENT â€” a sidecar predating part 72. We know nothing, so the shader
    //     keeps the full copy. Safe, and counted at the copy site rather than silently
    //     taken.
    //   * the KEY IS PRESENT AND THE LIST IS EMPTY â€” the shader reads NO ALU constants at
    //     all, which 11 of the 449 modules do. The correct copy for those is ZERO bytes,
    //     and treating "empty" as "unknown" gave them the full 4,096 instead â€” the feature
    //     quietly not applying to exactly the shaders it should help most.
    //
    // `JsonIntArray` returns an empty vector for both, so the presence of the key has to
    // be tested separately.
    const bool haveAluList = text.find("\"aluConsts\"") != std::string::npos;
    meta.aluConsts = JsonIntArray(text, "aluConsts");
    meta.aluDynamic =
        !(haveAluList && text.find("\"aluDynamic\": false") != std::string::npos);
    meta.aluListKnown = haveAluList;
    // The dynamic EXPRESSIONS, recorded at cache-build time for exactly this decision
    // (alu_const_sidecar.py's comment says so). Every element must parse as `N + a0`
    // with N in [8, 10] or the shader is NOT a palette shader and keeps the full copy â€”
    // an unparseable expression fails CLOSED, never into the bounded path.
    if (meta.aluDynamic)
    {
        const size_t dk = text.find("\"aluDynamicExprs\"");
        if (dk != std::string::npos)
        {
            const size_t open = text.find('[', dk);
            const size_t close = open == std::string::npos ? std::string::npos
                                                           : text.find(']', open);
            if (open != std::string::npos && close != std::string::npos)
            {
                bool all = false;
                for (size_t q = text.find('"', open); q != std::string::npos && q < close;
                     q = text.find('"', q))
                {
                    const size_t q2 = text.find('"', q + 1);
                    if (q2 == std::string::npos || q2 > close)
                    {
                        all = false;
                        break;
                    }
                    const std::string e = text.substr(q + 1, q2 - q - 1);
                    char* endp = nullptr;
                    const long n = strtol(e.c_str(), &endp, 10);
                    if (n < 8 || n > 10 || std::string(endp) != " + a0")
                    {
                        all = false;
                        break;
                    }
                    all = true;
                    q = q2 + 1;
                }
                meta.aluDynPalette = all;
            }
        }
    }

    meta.tfetchConsts = JsonIntArray(text, "tfetchConsts");
    meta.tfetchDims = JsonIntArray(text, "tfetchDims");
    // A sidecar written before part 25 has no dimensions at all, and one whose arrays
    // disagree in length is a build-pipeline defect rather than something to index into.
    // Both cases are dropped to "no dimension information" here, ONCE, so the binder's
    // fallback is reached deliberately and the message names the shader.
    if (!meta.tfetchDims.empty() && meta.tfetchDims.size() != meta.tfetchConsts.size())
    {
        fprintf(stderr,
                "[vk] %s: tfetchDims has %zu entries for %zu tfetchConsts â€” the arrays "
                "are POSITIONAL; ignoring the dimensions and treating every slot as 2D\n",
                path.filename().string().c_str(), meta.tfetchDims.size(),
                meta.tfetchConsts.size());
        meta.tfetchDims.clear();
    }

    // The attribute array is objects, so it is walked object by object rather than
    // with the flat integer-array reader.
    size_t at = text.find("\"attributes\"");
    if (at != std::string::npos)
    {
        size_t open = text.find('[', at);
        size_t cursor = open;
        while (cursor != std::string::npos)
        {
            const size_t objOpen = text.find('{', cursor);
            if (objOpen == std::string::npos)
                break;
            const size_t objClose = text.find('}', objOpen);
            if (objClose == std::string::npos)
                break;
            const std::string obj = text.substr(objOpen, objClose - objOpen + 1);
            VertexAttribute a;
            a.location = int32_t(JsonIntField(obj, "location", -1));
            a.fetchSlot = uint32_t(JsonIntField(obj, "fetchSlot", 0));
            a.format = uint32_t(JsonIntField(obj, "format", 0));
            a.isSigned = uint32_t(JsonIntField(obj, "signed", 0));
            a.isInteger = uint32_t(JsonIntField(obj, "integer", 0));
            a.strideDwords = uint32_t(JsonIntField(obj, "strideDwords", 0));
            a.offsetDwords = uint32_t(JsonIntField(obj, "offsetDwords", 0));
            a.indirect = uint32_t(JsonIntField(obj, "indirect", 0));
            if (a.indirect && a.fetchSlot < 96 &&
                std::find(meta.vfetchSlots.begin(), meta.vfetchSlots.end(),
                          uint16_t(a.fetchSlot)) == meta.vfetchSlots.end())
                meta.vfetchSlots.push_back(uint16_t(a.fetchSlot));
            meta.attributes.push_back(a);
            cursor = objClose + 1;
            const size_t nextBrace = text.find('{', cursor);
            const size_t arrayEnd = text.find(']', cursor);
            if (nextBrace == std::string::npos || nextBrace > arrayEnd)
                break;
        }
    }
    return true;
}

// ===================================================================================
// Pipeline key
// ===================================================================================
// Everything that has to be baked into a VkPipeline. Kept as a POD compared with
// memcmp so that adding a field cannot be forgotten in an equality operator â€” the
// classic way to get two different states sharing one pipeline, which renders as a
// draw quietly using the previous draw's blend mode.
constexpr uint32_t kPassDrawId = 1u << 0;
constexpr uint32_t kPassRtShadow = 1u << 1;

struct PipelineKey
{
    uint64_t vsHash;
    uint64_t psHash;
    uint32_t topology;
    uint32_t blendControl;
    uint32_t colorMask;
    uint32_t depthControl;
    uint32_t modeControl;
    uint32_t primRestart;
    // RB_COLORCONTROL's alpha test, as a pipeline dimension because the generated
    // shaders implement it behind a SPECIALIZATION constant (SPEC_CONSTANT_ALPHA_TEST
    // -> clip(oC0.w - g_AlphaThreshold)), and a spec constant is baked at pipeline
    // creation. 1 = the clip is compiled in. The THRESHOLD stays per-draw (shared
    // constants +272), so one pipeline serves every ref value.
    uint32_t alphaTest;
    // WHICH FRAGMENT MODULE THIS DRAW GETS, as a bitfield. A pipeline dimension for
    // the same reason alphaTest is: the module and the blend state are baked at
    // creation. Both bits are off by default.
    //   bit 0  kPassDrawId (part 39) â€” the fragment stage is replaced by drawid_ps.hlsl,
    //          which writes the draw's own index instead of its colour, for exactly one
    //          armed frame.
    //   bit 1  kPassRtShadow (part 65) â€” the fragment stage is the assets/shader_spv_rt
    //          variant, whose shadow-atlas taps read our screen-space factor instead.
    // A BITFIELD RATHER THAN A SECOND FIELD because PipelineKey has to stay
    // padding-free (see PipelineKeyHash) and 56 bytes is what that costs today.
    uint32_t passFlags;
    // THE POLYGON OFFSET, as a pipeline dimension rather than dynamic state (part 56).
    // Dynamic would have been cheaper in principle â€” the value varies per draw â€” but a
    // declared dynamic state MUST be set before EVERY draw with that pipeline, and this
    // renderer has draw paths (the resolve blits, the draw-ID pass, the overlay) that do
    // not pass through the per-draw state block. Declaring it produced 40
    // `VUID-vkCmdDraw-None-08608` from exactly those. The title uses only two or three
    // distinct offsets, so baking them costs a handful of pipelines and no correctness.
    // Stored as the raw register bits so the key stays trivially comparable.
    uint32_t polyOffsetScale;
    uint32_t polyOffsetOffset;

    bool operator<(const PipelineKey& o) const
    {
        return memcmp(this, &o, sizeof(*this)) < 0;
    }
    bool operator==(const PipelineKey& o) const
    {
        return memcmp(this, &o, sizeof(*this)) == 0;
    }
};

// THE LOOKUP CONTAINER IS A HASH TABLE, NOT A TREE (part 52 item 3.2).
//
// This map is probed once per DRAW â€” ~5,500 times a frame outdoors â€” and the profiler's
// `other` split prices that probe at **108-113 ns/draw**, the largest term in `other`
// after the residual that part 50 showed is the profiler measuring itself. A
// `std::map` of ~413 entries is ~9 levels of pointer chasing, and every level is a
// separate cache line holding a red-black node, so the cost is 9 likely misses and 9
// 48-byte `memcmp`s. A hash table is one hash of a fixed 48 bytes and one bucket.
//
// The key is a POD with no padding (two `uint64_t` then TEN `uint32_t` = exactly 56
// bytes â€” the polygon offset added two in part 56), which is what makes both the
// `memcmp` comparison and this hash correct â€”
// hashing raw bytes of a struct WITH padding would hash uninitialised memory and give
// two equal keys different hashes. If a field is ever added, keep that property or the
// table silently starts missing.
struct PipelineKeyHash
{
    size_t operator()(const PipelineKey& k) const
    {
        static_assert(sizeof(PipelineKey) == 56, "PipelineKey must stay padding-free");
        uint64_t h = 0xCBF29CE484222325ull;
        uint64_t w[7];
        memcpy(w, &k, sizeof w);
        for (uint64_t v : w)
        {
            h ^= v;
            h *= 0x100000001B3ull;
            h ^= h >> 29;
        }
        return size_t(h);
    }
};

// ===================================================================================
// The renderer
// ===================================================================================
struct Buffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    uint8_t* mapped = nullptr;
    VkDeviceSize size = 0;
    // For a DEVICE-LOCAL twin of a host buffer with identical offsets (the stream store
    // mirror, part 106): the HOST buffer's mapping, so the few CPU readers of a stream
    // (`StreamLoc::bytes()`) read the bytes the twin was copied from rather than
    // dereferencing memory that has no mapping at all.
    uint8_t* shadowMapped = nullptr;
};

// Where UploadStream put a stream's bytes. Two buffers are now possible â€” the per-frame
// arena and the cross-frame store â€” and every consumer needs the handle, the device
// address and the host pointer, so returning a bare offset no longer says enough.
// `buf == nullptr` is the failure return that `VkDeviceSize(-1)` used to be.
struct StreamLoc
{
    Buffer* buf = nullptr;
    VkDeviceSize at = 0;
    bool ok() const { return buf != nullptr; }
    VkBuffer handle() const { return buf->buffer; }
    uint8_t* bytes() const
    {
        return (buf->mapped ? buf->mapped : buf->shadowMapped) + at;
    }
    VkDeviceAddress address() const { return buf->address + at; }
    VkDeviceSize capacity() const { return buf->size; }
};

// ===================================================================================
// A FLAT, OPEN-ADDRESSED CACHE â€” because the largest single cost on the pump thread
// turned out to be `std::unordered_map`, not any of the work it was caching
// ===================================================================================
//
// HOW THIS WAS FOUND, because the method transfers further than the fix. Part 55's plan
// filed item C as "split `UploadStream` before assuming anything about it": the symbol is
// 14.74% of the pump thread and appears in no performance plan this project has written,
// because part 22 closed the stream cache on the strength of `ProfScope(streams)` reading
// 0.0% â€” and a scope is a region of code, not a subsystem (gotcha 343).
//
// It could not be split with `ProfScope`. The hot path here is taken ~33,000 times in a
// crowd frame and a scope costs two clock reads at ~20 ns, so instrumenting it would add
// 1.3 ms a frame â€” larger than several of the phases it would be separating. An
// instrument that big does not measure the function, it replaces it (gotcha 7). So the
// split was done with `perf` and the DWARF line table instead (`tools/part55_srcline.py`),
// at zero cost to the thing being measured, and the answer was not what any amount of
// reading the code would have suggested:
//
//     41.24%  stl_function.h:378     std::equal_to<uint64_t> â€” the key compare
//     13.99%  hashtable.h:2263   \
//     12.79%  hashtable.h:0       |  _M_find_before_node: the bucket's node chain
//     10.14%  hashtable.h:2257   /
//     10.49%  hashtable_policy.h:585 _Mod_range_hashing â€” the PRIME MODULO, a division
//      1.53%  vk_renderer.cpp:6851   ...the first line of our own function
//
// **Eighty-nine percent of `UploadStream` is the hash-map lookup**, i.e. 13.1% of the
// whole pump thread â€” larger than any single item in part 55's plan, and it is not
// parallel work waiting for a thread. It is work that should not exist.
//
// WHY `std::unordered_map` IS THIS SLOW HERE, stated so the reasoning can be checked
// rather than trusted. It is a chained hash table: every entry is a separately
// `malloc`ed node, so a lookup is a bucket-array load, then a dependent pointer chase to
// a node that was allocated at an unrelated address, then a compare of a key that lives
// in that node â€” two dependent cache misses whose latency cannot be overlapped, and the
// compare is charged with the second one, which is why `equal_to` reads as the single
// hottest line. On top of that the standard mandates prime-modulo bucketing, so every
// lookup performs a 64-bit division (~20-26 cycles, unpipelined), and `std::hash` for an
// integer is the IDENTITY, so nothing is mixed before that division.
//
// WHAT REPLACES IT. Open addressing with linear probing, keys and values in two flat
// arrays sized to a power of two: one masked load, and a probe sequence that walks
// forward through memory the prefetcher can see coming. The division is gone; the node
// allocations are gone (`_int_malloc` is 2.19% of this thread); a hit is typically one
// cache line.
//
// TWO DETAILS THAT WOULD BE SILENT DEFECTS IF GOT WRONG.
//
//  1. **The key must be MIXED, not used raw.** With prime bucketing, a structured key is
//     survivable. With a power-of-two mask the low bits ARE the bucket, and this cache's
//     key is `(va << 32) | (bytes << 2) | endian` â€” its low bits are an endian code and
//     the bottom of a byte count, so the identity hash would pile every stream in a
//     frame into a handful of buckets and turn a probe into a linear scan. A splitmix64
//     finalizer costs three multiplies and fixes it. This is not a hypothetical: it is
//     the standard way a power-of-two table is slower than the chained one it replaced.
//
//  2. **Clearing is a GENERATION BUMP, not a memset.** `streamCache` is cleared every
//     frame; zeroing a 4,096-entry table each time would hand back part of the saving.
//     Each slot carries the generation it was written in, and a slot from any older
//     generation reads as empty. That is correct with linear probing for a reason worth
//     writing down: every live entry was inserted AFTER the last bump, so its own probe
//     sequence also treated stale slots as free â€” no live key's chain can pass over one.
//
// AND IT IS AN ARM WITH A VERIFIER, because the wrong answer here is not a crash.
// `UploadStream`'s key is an identity by construction (part 22 fixed a version whose
// fields overlapped) and a lookup that returns the WRONG entry hands a draw another
// mesh's vertex stream â€” triangles between unrelated vertices, on some frames, in some
// places. `CZ_VK_NO_FLAT_CACHE=1` keeps the `std::unordered_map` and is the same-binary
// control arm; `CZ_VK_VERIFY_FLAT_CACHE=1` maintains BOTH structures and compares every
// lookup, and `CZ_VK_VERIFY_FLAT_CACHE_POISON=1` perturbs the flat one so the check can
// be seen to fire â€” a verifier that has never failed has not been shown capable of it
// (gotcha 30). Same shape as part 53's slot-mix-up check, which is why that item was
// trustworthy.
bool g_flatCacheOff = false;
bool g_flatCacheVerify = false;
bool g_flatCacheVerifyPoison = false;
uint64_t g_flatCacheChecked = 0;
uint64_t g_flatCacheDisagreed = 0;
uint64_t g_flatCacheProbes = 0;     // probe steps taken, to price the load factor
uint64_t g_flatCacheLookups = 0;
// GROW ACCOUNTING, always on, because the operator's first soak A/B reported stuttering
// in transit that the settled soak did not have, and a doubling table is the obvious
// suspect: a grow re-inserts every live entry and the cross-frame store reaches tens of
// thousands of them. "Obvious suspect" is not evidence, and the headless campaigns say
// the opposite (the flat arm had FEWER frames over 50 ms than the map arm), so this
// counts and TIMES every grow instead of anyone reasoning about it. Two clock reads per
// grow â€” of which there are a handful in a whole run â€” so it is free.
// The constant memo (part 55). Hit rate is the claim, not a side note: the item is
// worthless below ~30% and the counter is how a run says so.
// How many draws asked for a polygon offset. THE ARM NEEDS A COUNTER: "the decals look
// better" is a judgement, and this is the number that says the code ran at all â€” on the
// title backdrop it should read ~80 a frame and `CZ_VK_NO_POLY_OFFSET=1` must take it to 0.
uint64_t g_polyOffsetDraws = 0;
// THE SCOPED SHARED-BLOCK ZERO (part 109), counted so the arm proves it engaged. Bytes
// NOT written rather than a boolean: "the fast path is on" and "the fast path is reached"
// are different claims, and this title's shader mix decides the second (gotcha 408).
uint64_t g_sharedZeroDraws = 0, g_sharedZeroSaved = 0;
// Draws that enabled the STENCIL TEST. ~18% of a gameplay frame on the operator's own
// captures, and this renderer honoured none of them until part 56.
uint64_t g_stencilDraws = 0;
bool g_constMemoOff = false;
bool g_psConstScaleActive = false;   // CZ_VK_PS_CONST_SCALE mutates in place; see its use
bool g_constMemoVerify = false;
bool g_constMemoVerifyPoison = false;
uint64_t g_constMemoHits = 0;
uint64_t g_constMemoVsHits = 0;
// Run totals, never reset by the profile print â€” the windowed counters above are zeroed
// each window, and a run summary built from those would report only the last one.
uint64_t g_constMemoRunHits = 0;
uint64_t g_constMemoRunMisses = 0;
uint64_t g_constMemoRunVsHits = 0;
uint64_t g_constMemoRunPsHits = 0;
uint64_t g_constMemoPsHits = 0;
uint64_t g_constMemoMisses = 0;
uint64_t g_constMemoChecked = 0;
uint64_t g_constMemoStale = 0;
// The vertex-fetch memo census (Â§6eb Â§4). A plain global compare on the hot path, not a
// static-init guard, which is the shape part 76 had to take back off the per-draw path
// (gotcha 453).
bool g_fetchMemoCensus = false;
uint64_t g_fetchMemoHits = 0, g_fetchMemoMisses = 0;
uint64_t g_fetchMemoShaderMiss = 0, g_fetchMemoVersionMiss = 0;
uint64_t g_fetchMemoExactHits = 0, g_fetchMemoExactMisses = 0;
bool g_streamDedupCensus = false;
uint64_t g_dedupLookups = 0, g_dedupRepeats = 0, g_dedupOverflow = 0;

// ---- CZ_VK_RESOLVE_SPLIT_CENSUS=1 â€” part 89 step 0a: how much of `recordVertex` and
// `recordIndex` is UploadStream (the RESOLVE half â€” flat-cache lookup, content guard,
// cross-frame store: shared mutable state that must stay serial in any parallel-record
// design) versus the pure per-draw work around it (decode + bind recording, which is
// what secondary command buffers could move to workers)?
//
// SAMPLED, 1 DRAW IN 16, because the per-call instrument is disqualified by this
// file's own FlatCache comment: UploadStream runs ~40-50k times a crowd frame and a
// ProfScope per call is ~2 ms/frame of clock reads â€” larger than the split it would
// be measuring (gotchas 7, 223). Raw NowNs() pairs on every 16th draw cost ~0.13 ms
// and are unbiased across draws; the census prints its own clock bill per draw so the
// reader subtracts it rather than trusting the raw number (gotcha 7's discipline).
// A DIAGNOSTIC ARM: never quote a frame time from a run carrying it.
bool g_resolveSplitCensus = false;
uint64_t g_resolveVNs = 0, g_resolveINs = 0;        // sampled UploadStream ns, by side
uint64_t g_resolveVCalls = 0, g_resolveICalls = 0;  // sampled call counts
uint64_t g_resolveSampledDraws = 0;

// ---- CZ_VK_REUSE_CENSUS=1 â€” WOULD CROSS-FRAME COMMAND REUSE BE SERVED? (part 87) ----
//
// Case West's lead 2: their part-7 campaign left the frame partitioned into ranges with
// per-range secondaries and guard hashes over every input, and a range whose inputs hash
// identical to last frame's could REPLAY last frame's recorded commands and skip record
// entirely. This renderer has none of that structure (the 2a/2b port is the wave-2
// candidate and ~10k diverged lines), so per the rule that a hit rate is measured before
// the thing that depends on it is written (gotchas 428, 434, 470), this census asks the
// question the port would turn on: HOW MANY DRAWS ARE INPUT-IDENTICAL TO THE SAME-ORDINAL
// DRAW OF THE PREVIOUS FRAME, AND IN HOW LONG A RUN?
//
// A draw's fingerprint folds everything the recorded commands depend on, mirrored from
// what the renderer itself reads rather than from a model of it:
//   * the full PipelineKey (56 B, padding-free â€” PipelineKeyHash already depends on that);
//   * the draw arguments and the index buffer's address/endian;
//   * the dynamic state the pipeline does not bake (scissors, window offset, viewport,
//     blend constants, alpha/stencil ref, VTE, cull mode, clip control, VGT_INDX_OFFSET)
//     and the render-target block (surface/color/depth info) â€” a draw into a different
//     target is never the same draw;
//   * the vertex-fetch constants of every declared attribute and the texture-fetch
//     constants of every declared tfetch slot, both stages;
//   * the ALU constants THE SHADERS ACTUALLY SEE, with the gather's own semantics
//     (c0..c3 always + the sidecar list; the whole window when the gather is off or the
//     shader is dynamic) â€” mirroring CopyConstWindow is what stops this census answering
//     a different question than the upload path asks.
//
// Stream CONTENT is the part a register fingerprint cannot see, and it is folded via the
// renderer's own verdict: a draw is dirty if any stream it touched was COPIED this frame
// (fresh, guard-caught stale, or arena-pathed). A persist hit with a passing guard counts
// as unchanged â€” including a SAMPLED guard, deliberately, because that is the semantics
// the shipping renderer already lives by (it serves the stored bytes on a sampled pass).
//
// TWO STATED CEILINGS, both biased the safe way. Texture CONTENT changes are NOT folded
// (the fetch constants are; a texture rewritten in place under an unchanged fetch
// constant counts identical), so the number is a ceiling. And the comparison is by
// ORDINAL: a frame whose draw list shifts by one misaligns every draw after the shift,
// which under-reports reuse â€” the census fails toward the null (gotcha 30's neighbour).
//
// A DIAGNOSTIC ARM (gotcha 7): it hashes ~0.5-1 KB of registers per draw on the pump
// thread. Never quote a frame time from a run carrying it. Free when off â€” one plain
// global compare per draw, not a static-init guard (gotcha 453).
bool g_reuseCensus = false;
bool g_reuseDrawDirty = false;
uint64_t g_reuseFpDraw = 0;
std::unordered_set<uint64_t> g_reuseCopiedKeys;   // stream keys copied THIS frame
namespace reusecensus
{
struct Fp
{
    uint64_t fp;       // the full fingerprint
    uint64_t fpNoAlu;  // the same, EXCLUDING the ALU constant windows
    uint8_t dirty;
};
std::vector<Fp> prev, cur;
// THE DIAGNOSIS CHANNELS (added after the first crowd run). The ordinal channel read
// 1.7-1.9% at the crowd with ZERO draws failing on stream bytes alone, which leaves two
// mechanisms that want opposite responses: ORDINAL MISALIGNMENT (one inserted zombie
// draw shifts every later ordinal â€” a range-partitioned scheme could resync, so the lead
// survives) versus PER-FRAME CONSTANTS (an engine time/wind register in every shader's
// window â€” reuse is dead at its mechanism). A set-membership channel is alignment-
// insensitive, and a no-ALU fingerprint isolates the constants; the four cells of
// (ordinal|set) x (full|noAlu) name the killer.
std::unordered_set<uint64_t> prevFull, prevNoAlu;
uint32_t run = 0;            // current run of identical draws
// Totals since the run started, plus a snapshot at the last periodic print so every
// line reports WINDOW rates â€” part 72's census printed a cumulative mean and the mean
// was a transient (gotcha 428); this one never prints C/n as a rate.
struct Tot
{
    uint64_t frames = 0, draws = 0, identical = 0, fpOnlyDirty = 0, misaligned = 0;
    uint64_t runs = 0, runDraws8 = 0, runDraws32 = 0;
    uint64_t setFull = 0, ordNoAlu = 0, setNoAlu = 0;
    // Every draw that touched a stream copied this frame, unconditionally â€” the proof
    // the dirty plumbing is alive at all. The `fpOnlyDirty` conjunction read 0.0 in all
    // 28 windows of the first diagnosis run, and a channel that has never been seen to
    // fire cannot have its zeros believed (gotcha 151).
    uint64_t dirtyDraws = 0;
} t, last;

inline void FlushRun()
{
    if (!run)
        return;
    ++t.runs;
    if (run >= 8)
        t.runDraws8 += run;
    if (run >= 32)
        t.runDraws32 += run;
    run = 0;
}

inline void EndDraw(uint64_t fp, uint64_t fpNoAlu, bool dirty)
{
    const size_t idx = cur.size();
    const bool fpMatch = idx < prev.size() && prev[idx].fp == fp;
    ++t.draws;
    if (dirty)
        ++t.dirtyDraws;
    if (fpMatch && !dirty)
    {
        ++t.identical;
        ++run;
    }
    else
    {
        if (fpMatch)
            ++t.fpOnlyDirty;
        if (idx >= prev.size())
            ++t.misaligned;
        FlushRun();
    }
    if (!dirty)
    {
        if (prevFull.count(fp))
            ++t.setFull;
        if (prevNoAlu.count(fpNoAlu))
            ++t.setNoAlu;
    }
    if (idx < prev.size() && prev[idx].fpNoAlu == fpNoAlu && !dirty)
        ++t.ordNoAlu;
    cur.push_back(Fp{ fp, fpNoAlu, uint8_t(dirty) });
}

void Print(const char* tag)
{
    const Tot d{ t.frames - last.frames, t.draws - last.draws,
                 t.identical - last.identical, t.fpOnlyDirty - last.fpOnlyDirty,
                 t.misaligned - last.misaligned, t.runs - last.runs,
                 t.runDraws8 - last.runDraws8, t.runDraws32 - last.runDraws32,
                 t.setFull - last.setFull, t.ordNoAlu - last.ordNoAlu,
                 t.setNoAlu - last.setNoAlu, t.dirtyDraws - last.dirtyDraws };
    last = t;
    if (!d.frames || !d.draws)
        return;
    const double f = double(d.frames);
    const double dpf = double(d.draws) / f;
    const double idf = double(d.identical) / f;
    const double r8f = double(d.runDraws8) / f;
    fprintf(stderr,
            "[reuse] %llu frames (+%llu%s) â€” ALL FIGURES ARE PER-FRAME RATES OVER THAT "
            "WINDOW\n"
            "[reuse]   draws %.0f   IDENTICAL to the same-ordinal draw of the previous "
            "frame: %.1f (%.2f%%)\n"
            "[reuse]     in runs >=8: %.1f/frame (%.2f%%)   >=32: %.1f/frame   runs "
            "%.1f/frame (mean identical-run length %.1f)\n"
            "[reuse]   register-identical but stream bytes copied this frame: %.1f/frame "
            "(dirty draws at all: %.1f/frame â€” the channel's liveness proof)   past "
            "previous frame's end (misaligned): %.1f/frame\n"
            "[reuse]   ceiling at 524 ns/draw of record (Â§6ec Â§1): identical %.3f ms, "
            "runs>=8 %.3f ms â€” texture CONTENT is not folded and ordinal misalignment "
            "under-reports, so read as a CEILING on a run-shaped saving\n",
            (unsigned long long)t.frames, (unsigned long long)d.frames, tag,
            dpf, idf, dpf > 0 ? 100.0 * idf / dpf : 0.0,
            r8f, dpf > 0 ? 100.0 * r8f / dpf : 0.0,
            double(d.runDraws32) / f, double(d.runs) / f,
            d.runs ? double(d.identical) / double(d.runs) : 0.0,
            double(d.fpOnlyDirty) / f, double(d.dirtyDraws) / f, double(d.misaligned) / f,
            idf * 524e-6, r8f * 524e-6);
    // The four-cell diagnosis: (ordinal|set) x (full|noAlu). Set >> ordinal at the same
    // fingerprint means the draws exist but MOVE (alignment is the killer, a resyncing
    // range scheme survives); noAlu >> full at the same matcher means the ALU constants
    // are the killer (a per-frame engine register, and reuse dies at its mechanism).
    fprintf(stderr,
            "[reuse]   diagnosis/frame: ordinal full %.1f (%.2f%%) | set full %.1f "
            "(%.2f%%) | ordinal noALU %.1f (%.2f%%) | set noALU %.1f (%.2f%%)\n",
            idf, dpf > 0 ? 100.0 * idf / dpf : 0.0,
            double(d.setFull) / f, dpf > 0 ? 100.0 * double(d.setFull) / f / dpf : 0.0,
            double(d.ordNoAlu) / f, dpf > 0 ? 100.0 * double(d.ordNoAlu) / f / dpf : 0.0,
            double(d.setNoAlu) / f, dpf > 0 ? 100.0 * double(d.setNoAlu) / f / dpf : 0.0);
}

inline void FrameBoundary()
{
    FlushRun();
    prev.swap(cur);
    cur.clear();
    prevFull.clear();
    prevNoAlu.clear();
    for (const Fp& e : prev)
    {
        prevFull.insert(e.fp);
        prevNoAlu.insert(e.fpNoAlu);
    }
    g_reuseCopiedKeys.clear();
    ++t.frames;
    if (t.frames == 30 || t.frames % 600 == 0)
        Print(" since the last line");
}
} // namespace reusecensus

// ---- PART 81 Â§2: WHERE IS THE GUARD'S 86.2 MB A FRAME CHARGED? ----------------------
//
// `[vkprof]` reports **guard read 86.21 MB/frame** while the prehash pool reports **96.2%
// served, 27.2 MB/frame moved off the pump**. Subtracting says 59 MB a frame is still
// read ON the pump â€” which at any plausible rate is milliseconds, and NO PROFILER PHASE
// SHOWS IT: `streams` reads 0.1-0.2% and `record`'s GUARD column is 10 ns a draw. It is
// the largest number in this frame that has never been placed, and Â§6ec's whole argument
// is that the guards ARE the remaining CPU cost â€” so it decides whether that argument has
// an item behind it.
//
// Exactly one of three things is true and they want different work:
//   1. the pump reads far less than the subtraction suggests, because the two counters
//      count different POPULATIONS (96.2% of requests served can still be 31% of bytes
//      if the pool serves the small streams and the pump keeps the big ones);
//   2. the reading is real and charged to a scope nobody has attributed it to;
//   3. the counter means something other than its name.
//
// So: the bytes split by WHO READ THEM, and a clock over the pump's own half. The clock
// is the part that separates (1) from (2) â€” bytes alone cannot, because "59 MB on the
// pump" and "59 MB nowhere near the pump" produce the same subtraction.
//
// Census-gated rather than unconditional, on one plain global compare, because the clock
// pair is ~43 ns on a path that runs thousands of times a frame and an instrument that
// is not free when off is one this project has had to take back off a hot path once
// already (gotcha 453). It answers its question in one run and is off in every other.
bool g_guardCensus = false;
uint64_t g_gcPumpBytes = 0, g_gcPumpCount = 0, g_gcPumpNs = 0;
uint64_t g_gcPoolBytes = 0, g_gcPoolCount = 0;

// ---- PART 81 Â§1.0: THE BIND-RUN CENSUS ---------------------------------------------
//
// `BindVertexBufferCached` issues one `vkCmdBindVertexBuffers` per binding, and the API
// takes a CONTIGUOUS RANGE. The bind loop assigns `binding` with `++binding` as it walks
// the shader's attributes, so the bindings themselves are contiguous by construction â€”
// but the CHANGED ones need not be, and the whole item turns on which:
//
//   Hypothesis A, ALL-OR-NOTHING: a draw reuses the previous mesh or replaces it whole,
//     so the changed bindings form ONE run and batching collapses 1.725 calls/draw to
//     ~0.52 â€” 0.58 ms/frame at the operator's load. The item lives.
//   Hypothesis B, SCATTERED: changed bindings interleave with unchanged ones, a run is
//     usually one binding long, batched calls/draw stays ~1.725 and the item is worth
//     nothing.
//
// A mean of "runs per draw" alone cannot separate a few long runs from many short ones,
// and this project has been caught by a mean standing in for a distribution three times
// (gotchas 428, 434, 470) â€” so the run-LENGTH HISTOGRAM is printed beside it.
//
// A plain global compare on the hot path, not a static-init guard, which is the shape
// part 76 had to take back off the per-draw path (gotcha 453).
bool g_bindRunCensus = false;
uint64_t g_brDraws = 0;         // draws that entered the bind loop
uint64_t g_brOffered = 0;       // bindings offered (tracked + untracked)
uint64_t g_brChanged = 0;       // tracked bindings that actually issued a call
uint64_t g_brRuns = 0;          // contiguous runs of changed TRACKED bindings
uint64_t g_brUntracked = 0;     // binds above kMaxTrackedBindings: always issued, never
                                // batched, and an unbudgeted exception is how a ceiling
                                // becomes wrong
uint64_t g_brUntrackedDraws = 0;
uint64_t g_brRunHist[9] = {};   // run length 1..8, [8] = 8 or more
// Per-draw run state. -1 = no changed binding seen yet in this draw.
int32_t g_brPrevChanged = -1;
uint32_t g_brCurRun = 0;
bool g_brDrawHadUntracked = false;

// Close the run in progress, if any, and file its length.
inline void BindRunCensusCloseRun()
{
    if (!g_brCurRun)
        return;
    ++g_brRuns;
    g_brRunHist[g_brCurRun < 8 ? g_brCurRun : 8]++;
    g_brCurRun = 0;
}

// Called at the top of every draw's bind loop, where `binding` is reset to 0.
inline void BindRunCensusBeginDraw()
{
    BindRunCensusCloseRun();
    if (g_brDrawHadUntracked)
        ++g_brUntrackedDraws;
    g_brDrawHadUntracked = false;
    g_brPrevChanged = -1;
    ++g_brDraws;
}

// ---- PART 71: THE PIPELINE-CREATION CENSUS -----------------------------------------
//
// WHY IT IS UNCONDITIONAL. The header comment on the creation site says this hypothesis
// -- "first arrival somewhere costs a spike in `other`, and pipeline creation is the only
// candidate with that shape" -- had been inferred THREE TIMES and never measured, and
// then failed a pre-registered prediction at the casino. It gained a timer at that point,
// gated on `CZ_VK_PROFILE`... which costs 2-4 ms a frame, so it is off in every session
// whose stutter anyone has ever reported. Part 71's operator soak then produced a
// **3,891 ms frame** and three more above 800 ms, all in the first fifty seconds, all in
// the two arms that loaded a COLD shader set -- and the timer could say nothing, again.
//
// So this reads the clock unconditionally. The bill is two `steady_clock` reads per
// pipeline creation and a run creates ~500 of them; that is ~20 microseconds in a
// five-minute session, i.e. below anything this project can measure. An instrument that
// is only armed when nobody is looking is not an instrument (gotcha 7's inverse).
//
// The TOP-FRAME table is the load-bearing part and not the totals: a session that spends
// 4 seconds compiling pipelines spread evenly over 20,000 frames is invisible to a player,
// and one that spends the same 4 seconds inside a single frame is the thing the operator
// reported. Only a per-frame roll-up can tell those apart (gotcha 237 again, one level up
// from frame times).
uint64_t g_pipeCount = 0, g_pipeNs = 0, g_pipeWorstNs = 0, g_pipeWorstFrame = 0;
uint64_t g_pipeCurFrame = ~0ull, g_pipeCurNs = 0;
uint32_t g_pipeCurCount = 0;
struct PipeFrameRec { uint64_t frame = 0; uint64_t ns = 0; uint32_t count = 0; };
PipeFrameRec g_pipeTop[12];

// ===================================================================================
// THE CONSTANT-SLOT RACE DETECTOR (part 74) â€” what has to be clean before the gather
// can be re-enabled
// ===================================================================================
//
// WHY THIS EXISTS. Part 72 shipped the constant gather, its verifier read **0
// disagreements over 17,948,265 gathers**, and the operator still saw a half-screen sky
// flicker. Part 74 discriminated it â€” gather ON flickered twice including a deliberate
// positive control, gather OFF was clean twice â€” so the gather is off by default and the
// verifier was *right and not covering the defect*. It checks that the gather copied what
// the shader's list NAMES. That is not the feature's blast radius (gotcha 440).
//
// THE INVARIANT THIS CHECKS INSTEAD, and it is deliberately hypothesis-free: **the bytes a
// draw's constant window holds at RECORD time must equal what they hold at SUBMIT time.**
//
// That is not a style rule, it is forced by how constants are bound. The window is handed
// to the shader as a **buffer device address in a push constant** (`R->arena.address +
// vsConstAt`), so the shader dereferences it when the GPU executes, not when we record. The
// constant memo hands MANY draws the same arena offset, and the gather's top-up then writes
// that slot **in place** for each new shader. Any such write is applied RETROACTIVELY to
// every earlier draw of the frame that shares the offset â€” which is a mechanism for one
// group of draws rendering with another group's constants, and on a title that tiles
// LEFT/RIGHT (gotcha 265) for one half of the screen differing from the other.
//
// It makes no assumption that the defect IS the tiling or IS the projection: it records the
// tile and the c0..c3 block only so the report can say which, if it fires at all.
//
// COST, stated because it is not small: it hashes both 4 KB windows per draw and keeps
// ~100 bytes per draw. That is the traffic the gather exists to avoid, so this is an ARMED
// instrument and never a default â€” `CZ_VK_CONST_RACE=1`. It is a correctness gate to be run
// deliberately, not a probe to leave on (gotcha 7: a probe expensive enough to stall the
// game manufactures the stability it reports).
struct ConstRef
{
    uint32_t draw = 0;
    uint32_t windowOffset = 0;          // kPaScWindowOffset â€” the TILE identity
    VkDeviceSize vsAt = 0, psAt = 0;
    const ShaderMeta* vs = nullptr;
    const ShaderMeta* ps = nullptr;
    uint32_t c0[16] = {};               // the projection block, kept for the diff
    // ONLY THE REGISTERS THIS DRAW'S SHADER READS, not the whole 4 KB window. The first
    // cut stored and hashed both full windows per draw and was ~45x slower than the game â€”
    // so slow that the autonomous route never reached the outdoor world and the harness
    // correctly refused to report. An instrument too expensive to run where the defect
    // lives is not an instrument (gotcha 7). The median shader reads 26 registers and the
    // maximum is 56, so this is <=896 bytes against 4,096, and it is exactly the scope the
    // `affected` verdict needs. The 22 a0-relative shaders keep the whole window, because
    // they can index anywhere in it.
    std::vector<uint32_t> vsRegs;
    // AND THE PIXEL WINDOW'S. The first cut skipped this on the reasoning that "the pixel
    // window carries no projection" â€” true, and irrelevant: the pixel window is GATHERED
    // too, and a pixel shader served a register that moved is a SHADING defect, which is
    // what a flickering sky actually is. Skipping it made the detector blind to the more
    // likely half of its own subject, which is gotcha 440 for the third time in two parts.
    std::vector<uint32_t> psRegs;
};
std::vector<ConstRef> g_constRefs;
uint64_t g_raceDraws = 0, g_raceVsChanged = 0, g_racePsChanged = 0, g_raceProjChanged = 0,
         g_raceCrossTile = 0, g_raceFrames = 0, g_raceDirtyFrames = 0;
// **THE NUMBER THE DECISION TURNS ON.** A slot changing in registers the recorded draw
// never READS is harmless â€” the gather deliberately leaves those holding arena garbage, so
// a later top-up filling them in is the system working, not a defect. `affected` counts
// only draws whose OWN register list moved. Splitting the two is the same discipline that
// gotcha 440 says the gather's verifier failed to apply to itself: match the instrument's
// scope to the blast radius, in both directions.
uint64_t g_raceAffected = 0, g_raceAffectedProj = 0, g_raceAffectedFrames = 0;
// **THE PATCH READS THE WINDOW TOO, AND IT IS NOT A SHADER.** `PatchFovProjection` calls
// `SceneXformForm`, which reads c0..c3 of the copy â€” sixteen floats â€” to decide whether the
// window even IS a scene projection. Under the gather, a shader whose list does not name
// registers 0..3 leaves those holding whatever the bump arena last put there. So this
// decision is being made on ARENA RESIDUE, which varies frame to frame: the same shader can
// be recognized as a projection in one frame and not the next, and when it is recognized we
// WRITE to c0..c3. That is a frame-to-frame nondeterminism inside the renderer with no
// shader involved, and it is the shape an intermittent artifact has.
//
// Counted rather than argued: how often the patch runs on a window whose c0..c3 were not
// gathered, and how often it then RECOGNIZED something there.
uint64_t g_patchResidue = 0, g_patchResidueRecognized = 0, g_patchGathered = 0;
// Part 75's cached-patch shortcut and its verifier. See the long comment at the patch
// site: the arena is write-combined, so reading c0..c3 back out of it to recognize a
// projection was ~36% of the whole frame.
uint64_t g_patchSrcChecked = 0, g_patchSrcBad = 0;
const bool g_patchSrcVerify = [] {
    const bool o = getenv("CZ_VK_VERIFY_PATCH_SRC") != nullptr;
    if (o)
        fprintf(stderr, "[vk] CZ_VK_VERIFY_PATCH_SRC=1 â€” every patched projection is "
                        "recomputed from the ARENA copy and compared. Expensive on "
                        "purpose: it performs the uncached reads the change removes.\n");
    return o;
}();
const bool g_patchSrcVerifyPoison = [] {
    const bool o = getenv("CZ_VK_VERIFY_PATCH_SRC_POISON") != nullptr;
    if (o)
        fprintf(stderr, "[vk] CZ_VK_VERIFY_PATCH_SRC_POISON=1 â€” one float is perturbed; "
                        "CZ_VK_VERIFY_PATCH_SRC MUST then report mismatches. A zero here "
                        "means the verifier is blind, not that the patch is right.\n");
    return o;
}();
// CZ_VK_SKY_ASYM's accumulators â€” see the measurement in the present path.
// ===================================================================================
// THE PER-FRAME CPU/GPU PROFILER (part 74) â€” attributing a stutter to a side
// ===================================================================================
//
// WHAT WAS MISSING. Part 74's decomposition splits a frame into `walk + sleep + RESIDUAL`
// and put every hitch inside `walk` â€” which is `Pm4_Execute`: the command processor, the
// renderer's recording, AND the GPU fence wait, all in one number. So "the hitch is in the
// renderer" was as far as it could go, and the two halves want opposite fixes: CPU
// recording time is ours to shorten, GPU time is not.
//
// **This project has never measured GPU time directly.** The nearest thing is
// `tools/gpu_clock_sample.py`, which samples clocks and utilisation from outside the
// process and cannot be aligned to a frame, let alone to the ONE frame that stuttered.
//
// So: two timestamps written into each frame's own command buffer, and the fence wait
// clocked unconditionally on the CPU side. That gives, per frame:
//
//     wall = walk + sleep + residual        (part 74's decomposition)
//     walk = record + fenceWait             (this: CPU recording vs waiting for the GPU)
//     gpu  = the GPU's own execution time for that frame's command buffer
//
// READ BACK DEFERRED, NEVER WAITED ON. The results for a slot are collected right after
// that slot's fence has been waited â€” by then they are guaranteed available, so this adds
// no stall. That is the same discipline the RT coverage query already uses; a
// `VK_QUERY_RESULT_WAIT_BIT` here would make the instrument create the stall it reports
// (gotcha 7).
uint64_t g_gpuFrameNs = 0, g_gpuFrames = 0, g_fenceWaitNs = 0;
double g_timestampPeriodNs = 0.0;      // 0 = the device cannot timestamp; say so, never guess
VkQueryPool g_tsPool = VK_NULL_HANDLE;
// Per slot: the frame number whose timestamps that slot holds, and the last GPU time read
// back. `~0ull` means the slot has never been written.
uint64_t g_tsFrameOf[8] = { ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull };
uint64_t g_gpuNsOfFrame = 0;           // the most recently READ-BACK frame's GPU time
uint64_t g_gpuNsFrameNo = ~0ull;

uint64_t g_markCount = 0;      // F7 presses â€” the operator's felt-stutter markers
uint64_t g_skyFrames = 0, g_skyFlips = 0;
// WHICH SHADERS leave c0..c3 ungathered â€” i.e. whose windows the residue lives in. Only a
// handful of the 449 can be in this set, and naming them is far cheaper than bisecting.
std::unordered_map<const void*, uint64_t> g_residueByShader;
double g_skyAbsSum = 0.0, g_skyStepSum = 0.0, g_skyStepMax = 0.0;
bool g_raceReported = false;

bool ConstRaceOn()
{
    static const bool on = [] {
        const bool o = EnvOn("CZ_VK_CONST_RACE");
        if (o)
            fprintf(stderr,
                    "[vk] CZ_VK_CONST_RACE=1 â€” the constant-slot race detector is ARMED. "
                    "It hashes both 4 KB constant windows PER DRAW and re-checks them at "
                    "submit, so never quote a frame time from this run.\n");
        return o;
    }();
    return on;
}

// ...AND THE PROOF THAT IT CAN FIRE. A detector for an intermittent defect that has never
// been seen to scream is indistinguishable from a defect that did not trigger â€” which is
// the operator's own objection and the whole reason part 72's fix could not be confirmed
// (gotcha 30). This mutates ONE dword of ONE memo slot after the draws referencing it were
// recorded, which is exactly the shape being hunted, so a clean run of the real thing means
// something only if the poisoned run reports it.
bool ConstRacePoison()
{
    static const bool on = EnvOn("CZ_VK_CONST_RACE_POISON");
    return on;
}

// ===================================================================================
// THE SLOW-FRAME TABLE (part 72, open item 0w) â€” the same shape, asking a wider question
// ===================================================================================
//
// The pipeline table above found the 3,891 ms frame because it recorded a per-frame
// roll-up of ONE candidate. Item 0w needs the next candidate and does not know which it
// is: binning the operator's eight arms says heavy gameplay is smooth (`>2x med` 0.0%
// across 51 windows) and every recurring hitch is below ~2,000 draws â€” menu and load,
// where the median window still holds a ~290 ms frame against a 5.7 ms median.
//
// So rather than instrument one guess, this records the WORST FRAMES BY WALL TIME and
// what happened inside each: draws, textures uploaded, pipelines created. Whichever
// candidate is responsible names itself, and a frame that is slow with none of them
// elevated is itself the finding â€” it would say the cost is outside the renderer
// (streaming, file I/O, the guest) and stop anyone instrumenting the draw path further.
//
// Free: the frame's wall time is already measured for `CZ_FPS_LOG`, and the two counters
// it differences are single increments.
// ===================================================================================
// THE RESOLVE/BEGIN CYCLE CLOCK (part 73) â€” plan Â§4b's one number, before any fix
// ===================================================================================
//
// Part 72's pass histogram found **41 near-empty render passes per frame** â€” 80% of all
// passes, carrying 0.6% of the draws â€” and filed them as a LEAD rather than an item,
// because nobody knows what one of them COSTS. `CZ_VK_PROFILE` cannot answer it: it is
// 2-4 ms a frame, the same order as the thing being measured (gotcha 7). So this is the
// shape that worked for the pipeline census â€” one unconditional clock, always on, whose
// bill is two `steady_clock` reads per resolve.
//
// WHAT THE DECISION TURNS ON, said out loud first because part 72 got this wrong four
// times (gotchas 428/433/434): the question is **not** "what do resolves cost" â€” it is
// "what would be recovered by not issuing the 41 near-empty ones". So the time is split
// by the size of the pass that just ENDED, and the near-empty class is reported on its
// own. A total would be dominated by the 1.35 big resolves a frame, which snapshot a
// full-screen surface and are not removable.
//
// AND IT CHECKS THE LEAD'S OWN PREMISE. Â§4b asserts every near-empty pass is an
// `EndRendering` + resolve + `BeginRendering` cycle. Reading the code says that may be
// false: `DoResolve` only ends the render scope when it actually takes a snapshot or
// performs a clear, and `BeginRendering` returns immediately when the scope is still
// open. A near-empty pass that does neither costs bookkeeping and nothing else. So the
// count of resolves that genuinely BROKE the scope is printed beside the time, and if it
// is small for the near-empty class then Â§4b's premise is wrong and the item is dead
// before anyone writes a fix.
//
// LIMIT, stated because the number will be quoted: these are `vkCmd*` RECORDING costs on
// the pump thread. They do not measure what an extra render-pass instance costs the GPU.
// That is the right scope for this port â€” ~93% of the heavy frame is CPU (part 71) â€” but
// it is not the whole cost and this instrument cannot see the rest.
enum { kCycNearEmpty = 0, kCycSmall = 1, kCycBig = 2, kCycClasses = 3 };
uint64_t g_cycN[kCycClasses] = {}, g_cycResolveNs[kCycClasses] = {},
         g_cycBeginNs[kCycClasses] = {}, g_cycBroke[kCycClasses] = {};
// BeginRendering's real work is charged to the pass it OPENS, whose size is not known
// until that pass ends â€” so it is parked here and flushed by the next resolve.
uint64_t g_cycPendBeginNs = 0;

inline uint64_t CycNow()
{
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count());
}

// ---- CZ_VK_PARDRAW_CENSUS=1 â€” part 111 Â§3: THE ASK-FIRST STEP FOR ITEM B ------------
//
// Item B moves per-draw renderer work off the pump onto the idle cores. Before any
// threading code is written, two questions have to be answered PER SUBSYSTEM, and part
// 110 Â§1 is the reason there are two rather than one:
//
//   1. WHAT MUTATES SHARED STATE â€” the part that must serialise or shard, and whose cost
//      `S` sits on the pump forever however good the sharding is; and
//   2. WHAT READS STATE THE PUMP WILL OVERWRITE before a deferred job could run â€” the
//      part that cannot be deferred at all. That is what killed the constants stage
//      before it was designed: `CopyConstWindow`'s source is `g_regs`, which the pump's
//      own walk rewrites, and the VS window changes on 98.3% of draws.
//
// The encouraging prior this is testing: 47,000 stream lookups a frame against ~2,000
// frame-first touches, i.e. 96% of that traffic is READS. If that holds, B2's shared
// tables can be read-mostly with a small serial insert queue and the sharding question
// never becomes the hard one.
//
// THE HIGH-FREQUENCY OPERATIONS ARE COUNTED AND THE LOW-FREQUENCY ONES ARE TIMED, on
// purpose. A `steady_clock` read is ~25 ns, and the arena bump runs ~3 times a draw
// (~28,000 times a crowd frame) â€” timing that would add 1.4 ms to the frame and measure
// the instrument. The mutations that matter for `S` run ~2,000 times a frame, where two
// clock reads cost ~0.1 ms and the reading survives. A DIAGNOSTIC ARM either way: never
// quote a frame time from a run carrying it.
bool g_pardrawCensus = false;
struct ParDrawCensus
{
    // --- the per-frame arena (a bump pointer, ours, serial by construction) ---
    uint64_t arenaAllocs = 0;      // ArenaAlloc calls
    uint64_t arenaBytes = 0;
    uint64_t persistAllocs = 0;    // PersistAlloc calls (the cross-frame store's bump)
    uint64_t persistAllocBytes = 0;
    // --- the frame stream cache (FlatCache<StreamLoc>, reset every swap) ---
    uint64_t streamFinds = 0;      // lookups â€” the READ traffic
    uint64_t streamInserts = 0;    // ...and the mutations
    uint64_t streamInsertNs = 0;
    // --- the cross-frame persist store (FlatCache<PersistEntry>, survives the swap) ---
    uint64_t persistFinds = 0;
    uint64_t persistInserts = 0;
    uint64_t persistUpdates = 0;   // in-place edits of a found entry (guard/frame stamps)
    uint64_t mirrorPushes = 0;     // MirrorMark's vector append â€” a second shared mutable
    uint64_t persistMutNs = 0;     // inserts + mirror pushes
    // --- the texture table ---
    uint64_t texFinds = 0;         // ...every one of which WRITES `lastUsedFrame`, so a
                                   // "read" here is a read-modify-write and the sharding
                                   // question is different from the stream cache's
    uint64_t texInserts = 0;
    uint64_t texInsertNs = 0;
    // --- Vulkan calls that cannot leave the pump without external synchronisation ---
    uint64_t descWrites = 0;       // vkUpdateDescriptorSets on the per-draw path
    uint64_t descWriteNs = 0;
    // --- the pipeline cache ---
    uint64_t pipeFinds = 0;
    uint64_t pipeInserts = 0;
    // --- question 2: what reads `g_regs` on the per-draw path ---
    uint64_t constWindowCopies = 0;  // CopyConstWindow calls â€” reads g_regs, cannot defer
    uint64_t fetchWalks = 0;         // fetch-constant walks â€” also reads g_regs
};
ParDrawCensus g_pdc;
// One scope for the timed sites, so a site is two tokens and cannot forget its close.
struct PdcScope
{
    uint64_t* sink;
    uint64_t t0;
    explicit PdcScope(uint64_t* s) : sink(s), t0(s ? CycNow() : 0) {}
    ~PdcScope() { if (sink) *sink += CycNow() - t0; }
};
#define PDC_SCOPE(field) PdcScope _pdcScope(g_pardrawCensus ? &g_pdc.field : nullptr)

// ===================================================================================
// B1 â€” THE SHARED-BLOCK PRE-ZERO, ON THE GUARD POOL'S WORKERS (part 111 Â§4)
// ===================================================================================
//
// WHY THIS IS THE FIRST STAGE OF ITEM B EVEN THOUGH IT IS THE SMALLEST. Part 110
// measured the pump's serial floor at 2.49 ms against a 10.93 ms pump, so ~8.4 ms of
// per-draw work is movable in principle and item B is worth ~2.3 ms in practice (the
// wall has a second floor at 8.8 ms â€” the title's own recompiled code â€” so the useful
// headroom is ~2.1 ms, not 5.6). Every remaining stage of item B has a hazard to argue
// about: the constants read `g_regs`, which the pump rewrites between draws, and the
// streams and textures mutate caches the pump also reads. THIS ONE HAS NEITHER. The
// per-draw `memset(shared, 0, 2192)` has no source at all; it writes a constant into
// memory only this draw owns.
//
// So it is the job to build the machinery on: a work queue the pump posts to, workers
// taking from the existing budget, a drain point, an engagement counter, and a control
// arm. If the dispatch overhead cannot pay for a job THIS clean, it cannot pay for the
// harder stages either, and the whole design is refuted for one day's work. That is the
// pre-registered kill in Â§4.3 and it is the outcome worth paying for.
//
// WHERE THE THREADS COME FROM: the guard pool, unchanged, per Â§7. It runs at ~34% busy
// per worker and its work is front-loaded (dispatched at the swap, done early in the
// frame), so an idle guard worker picking up zero chunks costs the thread budget
// nothing. The hook is the same one `ParRec` already uses â€” `Prezero_Pending()` in the
// worker's wait predicate and `Prezero_WorkerDrain()` between guard jobs.
//
// CORRECTNESS NEVER DEPENDS ON THE PREDICTION, exactly as the parallel guard's does
// not: a slot whose chunk is not finished falls back to an inline `memset`, counted. A
// fallback rate above a few percent means the workers are not keeping up and the item
// is HALF-ENGAGED â€” which would otherwise read as a weak win rather than as a broken
// arm (gotcha 151).
//
// THE HAZARD THAT IS REAL, AND HOW IT IS CLOSED. A worker must never zero memory that
// the GPU is still reading or that this frame has already filled. Two things make that
// true by construction rather than by argument:
//   * the dispatch happens in `DoSwapImpl` AFTER `R->frameSlot` advances â€” i.e. after
//     `RetireOldestFrame` has observed the new slot's fence, which is the same moment
//     that already makes reusing that slot's command buffer and arena region legal; and
//   * a new dispatch DRAINS the previous one first, with the pump helping (it claims
//     and zeroes whatever is unclaimed rather than blocking on it), so a straggler from
//     the previous frame can never be writing into the region the new frame is handing
//     out. With `framesInFlight=1` those two regions are the SAME memory, which is
//     exactly the case a generation counter alone would not have covered.
constexpr uint32_t kPzSlotsPerChunk = 128;    // 128 * 2,304 = 294,912 B a chunk
constexpr uint32_t kPzMaxChunks = 512;        // 65,536 slots = draws per frame region

// The queue. `g_pzQueued` is posted by the pump under the drain above, so it only ever
// grows while no worker is claiming; `g_pzClaim` is the workers' shared cursor and
// `g_pzDone` is what the drain waits on.
std::atomic<uint32_t> g_pzQueued{ 0 }, g_pzClaim{ 0 }, g_pzDone{ 0 };

// PER-CHUNK OWNERSHIP, AND IT IS NOT A READINESS FLAG. The first version of this was one
// `ready` byte per chunk, and it had a real defect: a chunk the pump reached BEFORE a
// worker got to it took the inline path, filled its constants â€” and then the worker
// claimed that same chunk and zeroed the constants underneath it. The symptom would have
// been intermittent wrong descriptor indices and cleared clip planes, i.e. a picture bug
// that appears only when the workers fall behind, which is exactly the load where nobody
// is looking at correctness.
//
// So the chunk is OWNED, by a compare-exchange, and only the owner writes it:
//   FREE  -> the first of the two to claim it wins
//   BUSY  -> a worker is inside the memset. The pump must WAIT here; it cannot go inline,
//            because the worker's zeros would land after the pump's constants. Bounded by
//            one chunk's memset (~295 KB, tens of microseconds) and counted.
//   READY -> zeroed; the pump takes the fast path
//   PUMP  -> the pump got there first and owns it for the rest of the frame; every slot
//            in it takes the inline memset and no worker will touch it
constexpr uint8_t kPzFree = 0, kPzBusy = 1, kPzReady = 2, kPzPump = 3;
std::atomic<uint8_t> g_pzChunkState[kPzMaxChunks];
uint8_t* g_pzBase = nullptr;                  // the region's mapped address, stamped at
                                              // dispatch so a worker never reads `R`

bool g_prezeroOff = false;      // CZ_VK_NO_PREZERO=1 â€” the same-binary control arm
bool g_prezeroPoison = false;   // CZ_VK_PREZERO_POISON=1 â€” the positive control
// The bill, both sides of it (gotcha 344: part 53 moved 13.1 points off the pump and
// 33.2 points appeared on the workers; a measurement that reports one side is not one).
uint64_t g_pzBytesPre = 0;      // bytes a worker zeroed
uint64_t g_pzBytesInline = 0;   // ...and bytes the pump had to zero itself
uint64_t g_pzHits = 0, g_pzMisses = 0, g_pzOverflow = 0;
std::atomic<uint64_t> g_pzWorkerNsA{ 0 };
uint64_t g_pzDrainNs = 0, g_pzDrainHelped = 0, g_pzDrainWaits = 0;
uint64_t g_pzDispatches = 0, g_pzChunksPosted = 0;
uint64_t g_pzPumpClaims = 0;      // chunks the pump got to first
// ...and the workers therefore skipped. ATOMIC, and it is the only counter here that has
// to be: `Prezero_RunChunk` runs on a worker AND on the pump (which helps its own drain),
// so several threads can reach this increment at once. Every other counter in this block
// is touched by the pump alone. A plain `uint64_t` here would be a real data race â€” small
// in consequence, but a race argument that says "these two threads never overlap" has to
// actually be true for every counter it covers.
std::atomic<uint64_t> g_pzWorkerSkips{ 0 };
uint64_t g_pzWaits = 0, g_pzWaitNs = 0;   // the pump waiting on a BUSY chunk
uint64_t g_pzBeyondWatermark = 0;         // slots past what this frame's dispatch posted

bool Prezero_Pending()
{
    return g_pzClaim.load(std::memory_order_relaxed) <
           g_pzQueued.load(std::memory_order_acquire);
}

// Zero (or poison) one chunk. Called on a worker, and on the PUMP when it is helping a
// drain â€” which is why it takes no worker index and touches no per-worker state.
void Prezero_RunChunk(uint32_t c)
{
    uint8_t expect = kPzFree;
    if (!g_pzChunkState[c].compare_exchange_strong(expect, kPzBusy,
                                                   std::memory_order_acq_rel))
    {
        // The pump reached this chunk first and owns it. Leaving it alone is the whole
        // point of the handshake; zeroing it here would wipe constants already written.
        g_pzWorkerSkips.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const uint64_t t0 = CycNow();
    uint8_t* p = g_pzBase + uint64_t(c) * kPzSlotsPerChunk * kSharedStride;
    const size_t n = size_t(kPzSlotsPerChunk) * kSharedStride;
    // The POISON arm writes 0xAA where the zero would go. It is the gotcha-30 test for
    // this whole item: the claim is "the block a draw reads is the one a worker
    // prepared", and "the picture looks the same" cannot tell that apart from "the fast
    // path never engaged and the inline memset did the work". Under poison the picture
    // MUST break; if it does not, the arm is inert and the measurement is meaningless.
    std::memset(p, g_prezeroPoison ? 0xAA : 0x00, n);
    g_pzChunkState[c].store(kPzReady, std::memory_order_release);
    g_pzWorkerNsA.fetch_add(CycNow() - t0, std::memory_order_relaxed);
}

void Prezero_WorkerDrain()
{
    for (;;)
    {
        uint32_t c = g_pzClaim.load(std::memory_order_relaxed);
        const uint32_t q = g_pzQueued.load(std::memory_order_acquire);
        if (c >= q)
            return;
        if (!g_pzClaim.compare_exchange_weak(c, c + 1, std::memory_order_acq_rel))
            continue;
        Prezero_RunChunk(c);
        g_pzDone.fetch_add(1, std::memory_order_release);
    }
}

// `g_texFetchResolves` counts CALLS to UploadTexture â€” one per texture fetch per draw,
// i.e. ~2.3 per draw. **It was the slow-frame table's "texture uploads" column for one
// run and it was useless there**: every row read 2.24-2.43x its own draw count, so the
// column was the draw count in another unit and carried no independent information. It is
// kept because it is the denominator every cube/texture share in this file divides by; it
// is NOT the upload count. The run total for real uploads is ~2,300; this counter reaches
// 15,000 in a single frame. Gotcha: a counter named for the function it sits in measures
// the CALL, not the WORK the function usually declines to do.
uint64_t g_texFetchResolves = 0;
// The real ones: an upload is a decode + a staging memcpy + a RunImmediate, which is a
// submit and a fence wait. That is the per-frame variable cost open item 0w is hunting,
// so it is counted, weighed AND timed â€” a count alone cannot tell 6,000 small uploads
// from 6 huge ones, and neither can tell either from a stall inside RunImmediate.
uint64_t g_texRealUploads = 0, g_texUploadBytes = 0, g_texUploadNs = 0;
// ...and the LARGEST single upload, which a mean cannot give and which sizes the staging
// arena. Part 79 needed it to partition that arena into per-slot segments without
// narrowing the "larger than the staging buffer" decline the callers already have: a
// segment smaller than the biggest real texture would silently start dropping textures
// that used to upload. Free, and there was no way to ask before.
uint64_t g_texUploadMaxBytes = 0;
// Untiling, endian swap and image creation â€” everything BEFORE the staging copy.
uint64_t g_texDecodeNs = 0;
// ...AND ITS DECOMPOSITION (part 77). `g_texDecodeNs` is 469 ms of a texture-burst run and
// 66% of the whole upload path (Â§6dn Â§4), and part 75/76/77's board all say "parallelise or
// cache the untile loop" â€” which is a plan built on the assumption that the untile loop is
// where the time is. NOBODY HAS MEASURED THAT. The decode scope contains at least seven
// distinct things: a zero-filled allocation of the whole destination, the base-level untile,
// the mip chain's untile, TWO whole-buffer validation scans over each mip level (the
// mostly-empty test and the endpoint-luma divergence test), two content scans over the
// finished buffer, a source hash for the cache guard, and `vkCreateImage` + a view.
//
// So this is the same shape part 74 had to build three times over: a decomposition where
// every microsecond lands somewhere, rather than one number with a name that describes only
// the operation somebody expected to find (gotcha 439, and part 75's whole finding â€” a cost
// hid under a phase NAME that described the other thing in the same scope). The residual is
// PRINTED, so a wrong split shows up as a large unattributed column instead of as a
// confident wrong ranking.
//
// Free when nothing is uploading: eight `CycNow()` reads per REAL upload (~2,300 a run),
// never per unit and never per draw â€” the gotcha-223 trap this file has fallen into before.
uint64_t g_texDecAllocNs = 0;    // the zero-filled `pixels` vector
uint64_t g_texDecBaseNs = 0;     // level 0, all faces: Tiled2DOffset + CopySwapped
uint64_t g_texDecMipNs = 0;      // the mip chain's own untile loops
uint64_t g_texDecMipChkNs = 0;   // the mostly-empty and endpoint-luma guards
uint64_t g_texDecScanNs = 0;     // the all-black and uniform-block content scans
uint64_t g_texDecGuardNs = 0;    // TextureGuard over the SOURCE, for the cache entry
uint64_t g_texDecImageNs = 0;    // CreateImage (VkImage + view + allocation) + NameImage
uint64_t g_texDecGoldenNs = 0;   // the golden store: store/serve + (part 102) the queue push.
                                 // Named because it was the 79.9% RESIDUAL on czamd
// How many units the base level untiled, so the base column can be quoted per unit â€”
// a millisecond total cannot distinguish "the loop is slow" from "there are a lot of units".
uint64_t g_texDecBaseUnits = 0;
// PART 77's MIP-GUARD VERIFICATION. Both halves of that change are identity-preserving by
// construction â€” the carried `prevLuma` is the same function over the same bytes, and the
// word-at-a-time zero test is the same predicate â€” but "by construction" is an argument and
// this project's standard is a measurement (part 75 verified its arena patch at 0 of
// 47,352,900 draws). `CZ_VK_VERIFY_MIP_GUARD=1` recomputes the old way alongside the new and
// counts every disagreement; `_POISON=1` perturbs the NEW answer so the verifier must read
// 100%, because a checker that has never failed has not been shown capable of failing
// (gotcha 30).
uint64_t g_mgChecked = 0, g_mgDisagree = 0;
size_t g_mgEmptyWas = 0;
bool g_mgVerifyEmptyInit = false;
// CreateImage's own four driver calls â€” see the comment at the call sites. Counted for
// EVERY image this renderer makes, not only textures, because a snapshot or a dummy
// allocating the same way is the same finding one level over.
uint64_t g_ciN = 0, g_ciCreateNs = 0, g_ciReqNs = 0, g_ciAllocNs = 0, g_ciBindNs = 0,
         g_ciViewNs = 0, g_ciAllocBytes = 0;
// AND THE SPLIT THAT DECIDES WHAT A FIX WOULD BE. An upload measured at 249 us for a
// 55 KB copy is not a copy â€” but "not a copy" is an inference, and the two candidates
// want opposite fixes: if the time is the memcpy/decode, no synchronisation change helps;
// if it is `vkQueueWaitIdle`, batching or a transfer queue removes nearly all of it. So
// RunImmediate times its submit-and-wait separately from its allocate/record/free.
uint64_t g_immN = 0, g_immTotalNs = 0, g_immWaitNs = 0;
struct SlowFrameRec
{
    uint64_t frame = 0;
    uint32_t us = 0, draws = 0, tex = 0, pipes = 0;
    uint32_t texKB = 0, texUs = 0;
    // THE DECOMPOSITION, added in part 74 so the table stops reporting an ABSENCE.
    // `walkUs` is time inside Pm4_Execute â€” the command processor, the whole renderer and
    // the GPU fence wait. `sleepUs` is the pump asleep at the top of its loop. The
    // RESIDUAL is what is left of the frame's wall time after both, and it is the column
    // the whole instrument exists for: part 73 concluded "the cost is outside the
    // renderer" from three candidate columns all reading zero, which is the weakest kind
    // of finding. Every millisecond now lands somewhere by construction.
    uint32_t walkUs = 0, sleepUs = 0;
    int32_t residualUs = 0;
    // `walk` split one level: CPU recording versus waiting for the GPU. Plus the GPU's own
    // execution time for the frame, from its command buffer's own timestamps.
    uint32_t fenceUs = 0, gpuUs = 0;
    uint32_t texDecUs = 0;   // untile + endian swap + image creation
};
SlowFrameRec g_slowTop[12];

void SlowFrameNote(uint64_t frame, uint32_t us, uint32_t draws, uint32_t tex,
                   uint32_t pipes, uint32_t texKB, uint32_t texUs, uint32_t walkUs,
                   uint32_t sleepUs, int32_t residualUs, uint32_t fenceUs, uint32_t gpuUs,
                   uint32_t texDecUs)
{
    uint32_t worst = 0;
    for (uint32_t i = 1; i < 12; ++i)
        if (g_slowTop[i].us < g_slowTop[worst].us)
            worst = i;
    if (us <= g_slowTop[worst].us)
        return;
    g_slowTop[worst] = SlowFrameRec{ frame,  us,      draws,   tex,     pipes,
                                     texKB,  texUs,   walkUs,  sleepUs, residualUs,
                                     fenceUs, gpuUs, texDecUs };
}

// Close the frame currently being accumulated and keep it if it is among the worst.
void PipeFrameFlush()
{
    if (!g_pipeCurCount)
        return;
    uint32_t worst = 0;
    for (uint32_t i = 1; i < 12; ++i)
        if (g_pipeTop[i].ns < g_pipeTop[worst].ns)
            worst = i;
    if (g_pipeCurNs > g_pipeTop[worst].ns)
        g_pipeTop[worst] = { g_pipeCurFrame, g_pipeCurNs, g_pipeCurCount };
    g_pipeCurCount = 0;
    g_pipeCurNs = 0;
}

void NotePipelineCreate(uint64_t ns, uint64_t frame)
{
    ++g_pipeCount;
    g_pipeNs += ns;
    if (ns > g_pipeWorstNs)
    {
        g_pipeWorstNs = ns;
        g_pipeWorstFrame = frame;
    }
    if (g_pipeCurFrame != frame)
    {
        PipeFrameFlush();
        g_pipeCurFrame = frame;
    }
    g_pipeCurNs += ns;
    ++g_pipeCurCount;
}

// PART 71's hook fold â€” see the `hooksDraw` comment in the Renderer struct. These are
// ENGAGEMENT counters: `folded` counts the per-draw hook blocks skipped and `foldedFetch`
// the fetch-constant decodes skipped, and both must read ZERO in a run with RT on, which
// is the fold's identity gate.
uint64_t g_hookFoldFolded = 0;
uint64_t g_hookFoldLive = 0;
uint64_t g_hookFoldFetchFolded = 0;
uint64_t g_hookFoldFetchLive = 0;

uint64_t g_flatGrows = 0;
uint64_t g_flatGrowNs = 0;
uint64_t g_flatGrowWorstNs = 0;
uint32_t g_flatGrowLoud = 0;

// splitmix64's finalizer. Three multiplies and three shifts; enough avalanche that the
// low bits of a structured key are usable as a bucket index.
inline uint64_t FlatMix(uint64_t x)
{
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Key 0 is usable: emptiness is decided by the generation stamp, never by a sentinel
// key. That matters because these keys are guest-derived and a reserved value would be a
// landmine nobody would find until a stream landed at address 0.
//
// THE SLOT STATE LIVES IN THE GENERATION WORD, in its top bit, so a probe reads one
// array and not two. Low 31 bits = the generation the slot was written in; top bit set =
// this slot held an entry that was ERASED, and a probe must step over it rather than
// stop. Three cases and they are exhaustive:
//
//    (gens[i] & kGenMask) != gen    -> EMPTY: this slot is from an older generation (or
//                                     was never written). Stop probing.
//    gens[i] == (gen | kTomb)       -> TOMBSTONE: keep probing, do not match.
//    gens[i] == gen                 -> live; compare the key.
//
// A tombstone from an older generation therefore reads as empty for free, which is the
// property that lets `Clear()` stay a single increment even for a table that erases.
constexpr uint32_t kFlatTomb = 0x80000000u;
constexpr uint32_t kFlatGenMask = 0x7FFFFFFFu;

template <class V>
struct FlatCache
{
    std::vector<uint64_t> keys;
    std::vector<uint32_t> gens;
    std::vector<V> vals;
    uint32_t mask = 0;
    uint32_t gen = 1;
    uint32_t live = 0;
    uint32_t tombs = 0;

    uint32_t Size() const { return live; }

    // Bump the generation rather than clearing. Wrapping would make an ancient slot read
    // as live, so the wrap re-arms the table properly instead of pretending it cannot
    // happen: at one bump a frame this is over a year of continuous play, and the branch
    // costs nothing measurable against a frame.
    void Clear()
    {
        if (gen >= kFlatGenMask)
        {
            std::fill(gens.begin(), gens.end(), 0u);
            gen = 1;
        }
        else
        {
            ++gen;
        }
        live = 0;
        tombs = 0;
    }

    V* Find(uint64_t key)
    {
        if (!mask)
            return nullptr;
        ++g_flatCacheLookups;
        // The poison arm looks the WRONG KEY up, so a lookup that should hit misses and
        // the verifier must see it. It is deliberately not a wrong VALUE: the table is
        // still serving the frame while poisoned, and a miss costs a re-copy where a
        // wrong hit would draw a wrong mesh. What is being tested is the check's power,
        // not the renderer's tolerance for corruption.
        const uint64_t k = g_flatCacheVerifyPoison ? (key ^ 1ull) : key;
        uint32_t i = uint32_t(FlatMix(k)) & mask;
        for (;;)
        {
            ++g_flatCacheProbes;
            const uint32_t g = gens[i];
            if ((g & kFlatGenMask) != gen)
                return nullptr;
            if (g == gen && keys[i] == k)
                return &vals[i];
            i = (i + 1) & mask;
        }
    }

    // Insert or overwrite. Grows at a 0.7 load factor counting TOMBSTONES as occupied,
    // because a table full of tombstones probes exactly as slowly as a full one; the
    // rehash inside `Grow` is what drops them.
    V* Insert(uint64_t key, const V& v)
    {
        if (!mask || live + tombs + 1 > ((mask + 1) * 7) / 10)
            Grow();
        uint32_t i = uint32_t(FlatMix(key)) & mask;
        uint32_t firstTomb = UINT32_MAX;
        for (;;)
        {
            const uint32_t g = gens[i];
            if ((g & kFlatGenMask) != gen)
                break;                                  // empty: the key is not present
            if (g == gen && keys[i] == key)
            {
                vals[i] = v;
                return &vals[i];
            }
            if (firstTomb == UINT32_MAX && g == (gen | kFlatTomb))
                firstTomb = i;                          // reusable, but only once the
            i = (i + 1) & mask;                         // key is known to be absent
        }
        if (firstTomb != UINT32_MAX)
        {
            i = firstTomb;
            --tombs;
        }
        keys[i] = key;
        gens[i] = gen;
        vals[i] = v;
        ++live;
        return &vals[i];
    }

    // Erase by key. Tombstones rather than backward-shift deletion: erases are rare in
    // every user of this table (a stream the store could not re-home), and a shift moves
    // OTHER entries, which would invalidate a `V*` a caller is still holding â€” a defect
    // that would appear only under memory pressure and would look like a wrong mesh.
    void Erase(uint64_t key)
    {
        if (!mask)
            return;
        uint32_t i = uint32_t(FlatMix(key)) & mask;
        for (;;)
        {
            const uint32_t g = gens[i];
            if ((g & kFlatGenMask) != gen)
                return;
            if (g == gen && keys[i] == key)
            {
                gens[i] = gen | kFlatTomb;
                --live;
                ++tombs;
                return;
            }
            i = (i + 1) & mask;
        }
    }

    // Size the table up front so a doubling never lands inside a frame the player is
    // looking at. Measured on the outdoor route before this existed: 20 grows, 31.41 ms
    // in total, the worst three 3.38 / 8.85 / 15.04 ms as the cross-frame store filled â€”
    // i.e. three visible hitches during streaming, which is exactly when a player is
    // moving and exactly when the operator reported stuttering. That report turned out
    // to have a larger cause (see phase5-notes Â§6cl), but this part of it is real, it is
    // ours, and it is removable for the price of one allocation at start-up.
    void Reserve(uint32_t entries)
    {
        uint32_t cap = 16;
        while (uint64_t(entries) * 10 > uint64_t(cap) * 7)
            cap <<= 1;
        if (mask && cap <= mask + 1)
            return;                       // already at least this big
        std::vector<uint64_t> ok;
        std::vector<V> ov;
        for (uint32_t i = 0; mask && i <= mask; ++i)
            if (gens[i] == gen)
            {
                ok.push_back(keys[i]);
                ov.push_back(vals[i]);
            }
        keys.assign(cap, 0);
        gens.assign(cap, 0);
        vals.assign(cap, V{});
        mask = cap - 1;
        live = 0;
        tombs = 0;
        for (size_t k = 0; k < ok.size(); ++k)
            Insert(ok[k], ov[k]);
    }

    // Find, or default-construct in place. `std::map::operator[]`'s semantics, which is
    // what the census tables that use this were written against.
    V& FindOrInsert(uint64_t key)
    {
        if (V* v = Find(key))
            return *v;
        return *Insert(key, V{});
    }

    // Walk the live entries. Used by the stream census, which is off by default.
    template <class F>
    void ForEach(F f) const
    {
        for (uint32_t i = 0; i <= mask && mask; ++i)
            if (gens[i] == gen)
                f(keys[i], vals[i]);
    }

  private:
    void Grow()
    {
        const auto growT0 = std::chrono::steady_clock::now();
        const uint32_t oldCap = mask ? mask + 1 : 0;
        std::vector<uint64_t> ok;
        std::vector<V> ov;
        ok.reserve(live);
        ov.reserve(live);
        for (uint32_t i = 0; i < oldCap; ++i)
            if (gens[i] == gen)
            {
                ok.push_back(keys[i]);
                ov.push_back(vals[i]);
            }
        // Doubling only when the LIVE population justifies it: a table whose growth was
        // triggered by tombstones is rehashed at the same size instead, which is what
        // stops an erase-heavy user from growing without bound.
        uint32_t cap = oldCap ? oldCap : 1024;
        while (live + 1 > (cap * 7) / 10)
            cap *= 2;
        keys.assign(cap, 0);
        gens.assign(cap, 0);
        vals.assign(cap, V{});
        mask = cap - 1;
        live = 0;
        tombs = 0;
        for (size_t k = 0; k < ok.size(); ++k)
            Insert(ok[k], ov[k]);
        const uint64_t ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - growT0)
                                         .count());
        ++g_flatGrows;
        g_flatGrowNs += ns;
        if (ns > g_flatGrowWorstNs)
            g_flatGrowWorstNs = ns;
        // Loud on anything a player could see as a hitch. Capped, and unconditional â€”
        // a cost that only appears under an instrument cannot be blamed or exonerated
        // by a run the operator drives, which is the position this counter exists to
        // get out of.
        if (ns > 2000000 && g_flatGrowLoud < 12)
        {
            ++g_flatGrowLoud;
            fprintf(stderr,
                    "[vk] flat cache grow #%llu took %.2f ms (%u live entries into %u "
                    "slots) â€” this is a per-frame HITCH if it lands mid-frame\n",
                    (unsigned long long)g_flatGrows, double(ns) / 1e6, live, mask + 1);
        }
    }
};

// THE TWO ALWAYS-ON TEXTURE CENSUSES, moved down here in part 55 so they can be flat.
// They were `std::map<uint32_t, ...>` â€” red-black trees â€” and `g_texGuardAddrs` is
// touched once per GUARDED TEXTURE FETCH, i.e. on the hot path of a function that was
// 11.87% of the pump thread with ~72% of that in container machinery. A census nobody
// reads unless a profile window prints it should not cost a tree insert per fetch. The
// print sites sort a copy, so the output is byte-identical to the ordered map's.
FlatCache<TexSource> g_texSources;
FlatCache<TexGuardAddr> g_texGuardAddrs;

struct Image
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0;
    // ARRAY LAYERS, and it is here because `Barrier` needs it. A barrier's
    // subresourceRange had `layerCount = 1` hardcoded, which is correct for every image
    // this renderer had until cube maps arrived and then silently wrong: the five faces
    // past layer 0 would never leave TRANSFER_DST, so the sampler would read them in the
    // wrong layout. That is undefined behaviour whose most likely presentation is a cube
    // with one correct face, which reads as a decode bug rather than a barrier one.
    uint32_t layers = 1;
    // MIP LEVELS, here for exactly the reason `layers` is: `Barrier` names a
    // subresource RANGE, and a range that stops at level 0 leaves every level below it
    // in TRANSFER_DST while the sampler reads it. The presentation of that would be a
    // texture that is correct until the camera backs away from it, which reads as an
    // LOD bug rather than a barrier one â€” the same trap cube maps set in part 25.
    uint32_t levels = 1;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

// A texture the guest described with a fetch constant, uploaded once and reused.
// Keyed on the fetch constant's own dwords: if any of them changes the texture is a
// different texture, and if none of them changes it is the same one. That is a
// stronger key than the base address alone, which the title reuses.
struct TextureEntry
{
    Image image;
    uint32_t slot = 0;   // index into the bindless heap
    uint64_t key = 0;
    // WHERE the pixels came from, and WHAT THEY WERE. The cache key is the fetch
    // constant's six dwords â€” a DESCRIPTOR â€” and this is the CONTENT that descriptor
    // pointed at when the image was uploaded.
    //
    // Those are not the same thing, and the gap between them is a whole class of
    // rendering defect: a title that streams textures reuses a heap address, so a
    // building's texture can arrive at the address a dropped item's texture used, with
    // the same width, height and format. Every dword of the fetch constant is then
    // identical, the cache says "same texture", and the draw is handed the previous
    // occupant's image. Keeping the source range and a content guard is what lets the
    // cache ask the second question instead of assuming the answer.
    uint32_t va = 0;
    uint64_t srcBytes = 0;
    uint64_t guard = 0;
    // Six for a cube map, one for everything else. Needed on the REFRESH path, which
    // re-copies into the same image and would otherwise write face 0 and leave the
    // other five holding their first-upload pixels.
    uint32_t layers = 1;
    // THE FRAME THIS ENTRY'S CONTENT GUARD WAS LAST COMPUTED IN, plus one, so that the
    // zero-initialised value can never be mistaken for "already validated in frame 0".
    // See `g_texGuardEveryFetch`: the guard is the single largest term in the operator's
    // frame and re-hashing one texture once per draw that samples it is where the 92.9
    // MB/frame goes.
    uint64_t guardFrame = 0;
    // The pre-hashed guard's slot for the NEXT frame, and the frame it was filed for.
    // See PersistEntry::preSlot â€” same mechanism, different subsystem and a different
    // fold bound (`g_texGuardBytes`, which is a separate knob for a separate question).
    uint32_t preSlot = UINT32_MAX;
    uint64_t preFrame = 0;
    // THE FRAME THIS TEXTURE WAS LAST REQUESTED IN â€” the LRU recency stamp. Written by
    // TexFind on every hit and at upload. When the bindless heap fills, the slot
    // reclaimer evicts the entry with the smallest lastUsedFrame that no in-flight
    // frame can still reference (see ReclaimTextureSlot); without this a full session
    // served the 1x1 white dummy for every new texture from the cap onward.
    uint64_t lastUsedFrame = 0;
    // This image was uploaded from an ALL-ZERO source (an opaque-black BC1 block). The
    // golden store (part 94) refreshes such an entry from the last non-zero bytes of its
    // signature on the next cache hit, so a surface frozen black recovers the moment the
    // real texture streams in anywhere.
    bool wasZero = false;
};

// A RESOLVE SNAPSHOT: what one pass left in the EDRAM, kept as a host image under the
// guest address the pass copied it to.
//
// This is the mechanism that makes a post-processing chain work, and it exists because
// of what the resolve trace showed about this title. A title-screen frame issues about
// twenty resolves: a 1280x720 main pass, a 640x360 / 320x180 / ... / 1x1 downsample
// pyramid, some 1024x32 and 1024x1024 surfaces, and finally one resolve to the address
// VdSwap named. Every one of them renders into the SAME EDRAM and clears it afterwards
// (their RB_COPY_CONTROL has both clear bits set; the front-buffer one does not) â€” so
// the EDRAM at the end of a frame holds only the last pass, and the passes communicate
// exclusively through guest memory.
//
// We do not write resolved pixels back into guest memory: that would mean tiling them,
// and the consumer would then untile them again, for a round trip whose only purpose is
// to lose precision. Instead the destination address becomes the key, and a texture
// fetch that names it is served the host image directly.
//
// A resolve's SOURCE is RB_COPY_CONTROL's low three bits â€” 0..3 name a colour target
// and 4 names the DEPTH buffer â€” and 18.4% of this title's resolves are depth ones
// (10,448 of 56,925 in B1, `tools/xtr_resolve_census.py`). They are its shadow
// cascades and the scene depth its depth-of-field pass reads back, so a snapshot taken
// from the colour target for those is not an approximation: it is a different picture
// entirely, handed to a pass that then computes a circle of confusion out of it.
// Set in a snapshot map key to mean "this is the DEPTH resolve to that address, not
// the colour one". A guest address is above 0x1FFFFFFF nowhere in this runtime, so
// bit 31 is free for the purpose and the key stays one integer.
constexpr uint32_t kSnapshotDepthBit = 0x80000000u;

// The largest resolve destination we will build a snapshot image for. This title's
// biggest is its shadow cascade at 4096x1024; the cap exists so a garbage
// RB_COPY_DEST_PITCH cannot ask for a terabyte, and it has its own counter so hitting
// it is visible rather than silent.
constexpr uint32_t kMaxSurfaceExtent = 4096;

// The EDRAM stand-in's size, which is NOT the presented frame's size and was the same
// number for five phases. The scene is 1280x720, but the shadow pass renders a
// 1024x1024 cascade â€” so with a 720-row target every cascade lost its bottom 304 rows
// before anything downstream had a chance to sample them. Heights are what this title
// needs more of; the width stays at the presentation width because no pass here renders
// wider than the screen.
constexpr uint32_t kEdramHeight = 1024;

// A right-sized copy of a snapshot's top-left corner.
//
// A resolve snapshot is created at the destination surface's PITCH, because
// RB_COPY_DEST_PITCH is what the resolve registers carry and its low field IS the pitch.
// The fetch that later samples that address declares the surface's REAL width, and a
// sampler normalises over the image it is handed â€” so whenever pitch != width every
// texture coordinate is scaled by width/pitch and everything past that fraction reads
// the padding, which is zero.
//
// It is invisible while both are multiples of 32, which is the entire scene chain
// (1280, 640, 320, 160), and it destroys the tail of this title's luminance reduction,
// whose surfaces are 80, 40, 20, 10, 5 and 2 wide in pitches of 96, 64, 32, 32, 32, 32.
// Measured lit-column counts of five consecutive links fit that model exactly, and the
// last link â€” the 2x1 scene-average luminance the tone map reads â€” came out EMPTY.
// docs/phase5-notes.md Â§6ao.
//
// The height needs no such treatment: RB_COPY_DEST_PITCH's high field is the real
// height, and the snapshots are already the right height everywhere.
struct SnapshotView
{
    Image image;
    uint32_t slot = 0;
};

struct Snapshot
{
    Image image;
    // THE GUEST EXTENT, which is `image`'s divided by the resolution scale â€” and it is
    // stored rather than derived because it is what every comparison in this file
    // actually means. A fetch declaring 640x360, a sub-region fold looking for a surface
    // "of the same extent", and the resize check that rebuilds a reused address are all
    // asking about the surface the TITLE resolved, not about how many host pixels we
    // chose to keep it in. Deriving it by division would work and would put a scale
    // factor at every one of those sites instead of at the two that create the image.
    uint32_t guestW = 0, guestH = 0;
    // The PASS internal resolution the host image was built at (the scene's, or
    // the shadow tier's). A live change rebuilds each snapshot lazily through the
    // same path a guest-extent change does; comparing the stored pair is what
    // makes that trigger without touching the store eagerly at switch time.
    uint32_t builtW = 1280, builtH = 720;
    uint32_t slot = 0;   // bindless heap index, so a fetch can be served without a copy
    uint64_t frameSeen = 0;
    // Keyed (width << 16) | height. Refreshed from `image` by whatever resolve next
    // writes it, in that resolve's own command buffer â€” so a view never costs a submit
    // after the one that creates it, and it is never staler than its source.
    std::unordered_map<uint32_t, SnapshotView> views;
    // Which buffer this snapshot was taken FROM. It is part of the identity because
    // one guest address can be the destination of both a colour resolve and a depth
    // one (B1's 1812F000 is: 890 depth, 852 colour), and the two need different image
    // formats â€” so a change of source has to rebuild the image, exactly as a change of
    // extent does.
    bool fromDepth = false;
    // RT stage 2 (part 64): the trace pass renders INTO a depth snapshot, and an
    // attachment view must not carry the sampled view's (R,R,R,1) swizzle â€” so the
    // shadow atlas gets one identity-swizzle depth view, created at first trace and
    // retired with the image in the resize path.
    VkImageView rtAttachView = VK_NULL_HANDLE;
};

// A cube map the TITLE RENDERS ITSELF: six resolve snapshots assembled into the six
// layers of one `VK_IMAGE_VIEW_TYPE_CUBE` image in descriptor set 2.
//
// WHY THIS IS ITS OWN TYPE and not a Snapshot with six layers. A Snapshot is created by
// a resolve, keyed on that resolve's destination address, and lives in set 0 where its
// slot means "the 2D texture at this address". A rendered cube is SIX of those addresses
// standing for ONE texture in a different descriptor heap, and nothing about the resolve
// says which â€” the guest only says so later, when a fetch constant names the base
// address with a shader that declares a cube. So the six snapshots stay exactly as they
// are (other passes sample them as 2D surfaces, and do) and this is a second view of
// them, assembled on the first cube fetch and refreshed by each face's own resolve.
//
// It is 55% of all cube sampling in this game's opening hour â€” 409,911 of 746,355
// cube-declared draws in part 25's census â€” and until this existed every one of them
// read the 1x1 white dummy, in BOTH arms of every A/B, because guest memory at a resolve
// destination is whatever the allocator left there and for `06805000` that is zeros.
struct CubeSnapshot
{
    Image image;               // six layers, CUBE view, registered in set 2
    uint32_t slot = 0;         // index into set 2's heap, NOT set 0's
    uint32_t faceExtent = 0;   // one face is faceExtent x faceExtent
    uint32_t builtScale = 1;   // rebuild trigger on a live resolution change
    uint32_t faceStride = 0;   // guest bytes between one face's base and the next
    // Which faces have ever been copied in, as a bitmask. A COUNTER, because "the cube
    // is bound" and "the cube has six faces in it" are different claims and the second
    // is the one a reflection depends on â€” five faces and a stale sixth is a picture
    // defect with no other symptom (gotcha 151).
    uint32_t facesFilled = 0;
    uint64_t frameSeen = 0;
};

// --- one frame's worth of resources, so the CPU can record frame N+1 while the GPU is
// --- still executing frame N ---------------------------------------------------------
//
// Everything in here is something the CPU WRITES and the GPU READS, or the reverse.
// Nothing else needs duplicating: the EDRAM stand-in, the snapshots and the textures are
// only ever touched by the device, and a single queue executes submissions in order, so
// the barriers already in the recorder cover every GPU-to-GPU hazard across the frame
// boundary. It is the host-visible things â€” the command buffer being recorded, the bump
// arena the draws' vertices and constants live in, and the buffer the presented image is
// read back into â€” that would otherwise be rewritten under a frame still in flight.
//
// The per-frame METADATA is here for a reason that is easy to miss and would have made
// every A/B this change is measured with wrong. The present-side instruments (frame
// stats, the PPM dump, the uniform-colour counter) read `presentPixels` and label it with
// `R->frame`, `R->drawFingerprint` and the draw count â€” which with a deferred present
// describe the frame being RECORDED, not the pixels being looked at. `frame_compare.py`
// aligns two runs by exactly those fingerprints, so an off-by-one here does not look like
// a bug, it looks like a picture regression. Captured at submit, read at present.
// An image whose owner replaced it mid-frame, kept alive until every command
// buffer that could reference it has retired. THE REASON THIS EXISTS is the
// shadow-tier freeze (part 60): the snapshot-resize path destroyed the old atlas
// with a vkDeviceWaitIdle, but a wait-idle only covers SUBMITTED work â€” the frame
// being RECORDED had already sampled the old image in earlier passes, so the
// submit that followed referenced destroyed handles. That is undefined behavior
// that presented as a wedged queue (the operator: "it froze the game"), and the
// per-resize wait-idle was ALSO the stutter the live-rescale path documents. A
// retired image is destroyed once `retireFrame + framesInFlight + 1 <= R->frame`,
// by which point its last possible referencing fence has been waited.
struct RetiredImage
{
    uint64_t retireFrame = 0;
    Image image;
};

// ===================================================================================
// PARALLEL COMMAND RECORDING (part 89) â€” design (b): RESOLVE serial, RECORD parallel
// ===================================================================================
//
// The pump keeps everything that touches shared mutable state â€” the PM4 walk, the
// register decode, `UploadStream`'s flat cache / content guard / cross-frame store,
// the bump arena, the pipeline lookup â€” and deposits, per draw, a `DrawCapture`:
// resolved handles, offsets and derived state, nothing that needs re-resolving.
// Chunks of captures become SELF-CONTAINED dynamic-rendering instances recorded by
// the guard pool's workers (13-16% busy at the crowd, Â§6ei) into their own primary
// command buffers; the frame submits as one ordered `vkQueueSubmit` of many buffers.
//
// WHY INSTANCE SPLITS ARE FREE HERE, and why this needs neither secondaries nor
// suspend/resume: every attachment in the main scope is LOAD_OP_LOAD / STORE_OP_STORE
// with no clears (the EDRAM model â€” the title clears with draws), so ending a
// rendering instance and beginning another over the same attachments is semantically
// the identity. A pass under the chunk size therefore never splits and costs nothing
// beyond the capture write; the crowd pass yields ~16 worker chunks.
//
// ORDER IS PRESERVED BY CONSTRUCTION â€” chunks enter the submit list in capture order
// and draws within a chunk replay in capture order â€” and it is GATED, not assumed:
// the part-72 order gate's "submitted" side is now built from the replayed
// instances' own ids (each recomputed from the capture a worker actually consumed),
// and CZ_VK_ORDER_POISON still must fail it.
//
// CZ_VK_PAR_RECORD=1 is the arm (OFF by default until its 3v3 exists â€” part 87 Â§3's
// rule); CZ_VK_RECORD_CHUNK=N sets the chunk size (default 512). Inert without
// workers, and refused (loudly) under CZ_VK_NO_DRIVER_RECORD, whose measurement this
// path would silently distort.
struct DrawCapture
{
    static constexpr uint32_t kMaxBinds = 16;   // == BoundState::kMaxTrackedBindings
    VkPipeline pipeline;
    VkViewport viewport;
    VkRect2D scissor;
    float blend[4];
    uint32_t stencilOn, stencilRef, stencilMask, stencilWriteMask;
    struct { uint64_t vs, ps, shared; uint32_t drawIndex, pad; } push;
    uint32_t bindCount;
    VkBuffer vb[kMaxBinds];
    VkDeviceSize vo[kMaxBinds];
    VkBuffer ib;                 // VK_NULL_HANDLE = non-indexed draw
    VkDeviceSize io;
    VkIndexType it;
    uint32_t drawCount;          // index count (indexed) or vertex count (auto)
    int32_t baseVertex;          // VGT_INDX_OFFSET, already folded to 0 where the
                                 // resolve folded it (rect synth, expansions)
    // The order gate's raw material, filled only when the gate is armed: the replay
    // recomputes the id from THESE â€” the fields it actually consumed â€” so a capture
    // scrambled across draws fails the gate rather than passing on a precomputed id.
    uint64_t vsHash, psHash;
    uint32_t primType, indexVa;
    uint32_t gateCount;          // the RAW index count the capture-side id mixed
    int32_t gateIndxOffset;      // ...and the raw VGT_INDX_OFFSET, pre-folding
};

// A clear a resolve asked for, LATCHED instead of recorded (part 90). The full design
// and its arm are at DoResolve's clear block; the one-line version: the title clears
// EDRAM through the copy block's clear bits, our old mechanism honoured that with a
// whole-image vkCmdClear*Image (7 Mpix written for the ~0.4 the pass rendered, 94.3% of
// a 0.615 ms/frame GPU class), and the deferred form emits the clear as a
// vkCmdClearAttachments rect at the head of the NEXT pass's first instance â€” an
// instance that already exists, so it costs no dedicated render-scope cycle, which is
// what made the part-32 CZ_VK_SCOPED_CLEAR arm a wash on the pump (part 80 Â§1 item 4).
struct PendingClear
{
    bool isColor;        // false = the depth+stencil aspect
    VkClearValue value;
    VkRect2D rect;       // already clamped to the target image's extent at latch time
};

struct ParRecChunk
{
    std::vector<DrawCapture> draws;
    // Deferred clears this chunk must emit BEFORE its first draw â€” only ever non-empty
    // on the FIRST instance of a pass (the pump moves them in at handoff). Owned by the
    // chunk from enqueue to reset, so the worker reads them race-free the same way it
    // reads `draws`.
    std::vector<PendingClear> pendingClears;
    std::vector<uint64_t> orderIds;      // filled by the recorder when the gate is armed
    VkCommandBuffer cb = VK_NULL_HANDLE; // filled by whichever recorder claims it
    std::atomic<uint32_t> state{ 0 };    // 0 free, 1 queued, 2 recorded
    // The pass's attachments, snapshotted at enqueue: stable for the pass by
    // construction, and a chunk must not read R->color at record time â€” the worker
    // runs concurrently with the pump opening the NEXT pass.
    VkImageView colorView = VK_NULL_HANDLE, depthView = VK_NULL_HANDLE;
    uint32_t width = 0, height = 0;
    // Replay-side skip counters, aggregated into R->skips by the pump at the wait â€”
    // workers must not race plain uint64 fields (gotcha 151 needs the counters, the
    // aggregation keeps them honest).
    uint64_t skipPipeline = 0, skipViewport = 0, skipScissor = 0, skipBlend = 0,
             skipStencil = 0, skipSets = 0, skipVertex = 0, skipIndex = 0;
};

struct FrameSlot
{
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    Buffer present;             // the presented image, read back for the window
    bool inFlight = false;      // has been submitted and not yet waited on
    bool presentable = false;   // ...and had a real image copied into it

    uint64_t frame = 0;
    uint64_t draws = 0;
    uint64_t vertices = 0;
    uint64_t drawFingerprint = 0;
    uint64_t cameraFingerprint = 0;
    uint32_t width = 0, height = 0;
    size_t bytes = 0;
    // WAS THE PRESENT READBACK ACTUALLY RECORDED FOR THIS FRAME? Per SLOT, not per
    // frame, and that is the whole reason it exists (part 76).
    //
    // `doReadback` is a decision about the frame being RECORDED; `px` below is read out
    // of the frame being RETIRED, which is one to two frames older. While the predicate
    // was static those were always the same answer and the distinction did not exist.
    // With the readback armed by a KEY PRESS they are not: the first frame after an F8
    // press records a readback while the frame being retired never had one, so testing
    // the current frame's decision would hand the burst a buffer holding a much older
    // frame's pixels â€” a WRONG picture, silently, in the one instrument that exists to
    // be believed about what was on screen when.
    bool hasPixels = false;
    // WHICH FRAME'S PIXELS ARE ACTUALLY IN `present.buffer`. Stamped when the readback is
    // recorded, and checked against `frame` at the retire â€” see the check for why an
    // md5-shaped canary could not do this job.
    uint64_t pixelFrame = ~0ull;

    // Vulkan 1.1 compatibility descriptors are allocated from frame-owned pools.
    // A pool is reset only when this frame slot is reused, after its fence retired;
    // descriptor sets referenced by an in-flight command buffer are therefore never
    // updated or recycled. Modern Vulkan never creates these pools.
    std::vector<VkDescriptorPool> compatDescriptorPools;
    uint32_t compatDescriptorPool = 0;
    uint32_t compatDescriptorDraw = 0;
};
// Two is the whole design: the CPU records one frame while the GPU executes one. Deeper
// pipelining buys nothing here and costs a whole arena each â€” the GPU is 16.5 ms against
// the CPU's 27.7, so one frame of overlap already hides all of it (Â§6ar).
constexpr uint32_t kMaxFramesInFlight = 3;

// THE SWAPCHAIN, when CZ_VK_SWAPCHAIN=1 puts one on the window (part 54, plan Â§7).
//
// Everything here is null in the default arm and not one line of it executes. The
// default present path â€” resolve image -> host buffer -> Host_PresentPixels -> SDL
// texture â€” is untouched, which is what makes these two arms of one A/B.
struct SwapchainState
{
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    uint32_t width = 0, height = 0;
    std::vector<VkImage> images;
    // Layout tracking, because a swapchain image arrives UNDEFINED the first time and
    // PRESENT_SRC every time after, and the barrier before the blit has to name the
    // right old layout or the validation layer objects (and a driver may keep the old
    // contents rather than discarding them).
    std::vector<bool> everPresented;
    // One acquire semaphore per FRAME SLOT, not per image: the acquire happens before we
    // know which image we will get, so the semaphore cannot be indexed by image. One
    // render-finished semaphore per IMAGE, because the present waits on it and the same
    // image may be re-acquired while an older present on a different image is still
    // pending.
    std::vector<VkSemaphore> acquireSem;
    std::vector<VkSemaphore> renderSem;
    uint32_t acquireIndex = 0;
    // The image acquired for the frame currently being submitted, or UINT32_MAX when the
    // acquire failed and this frame will not be presented. A frame that cannot acquire
    // is DROPPED rather than presented stale â€” and it is counted, because a swapchain
    // that silently drops frames is a frame-rate defect wearing a driver's name.
    uint32_t acquired = UINT32_MAX;
    uint64_t presents = 0, acquireFails = 0, rebuilds = 0, suboptimal = 0;
    // CZ_VK_SWAPCHAIN_DUMP â€” the picture gate for this arm. See CreateSwapchain.
    // Set when a present reported SUBOPTIMAL or OUT_OF_DATE; consumed at the top of the
    // next frame's blit, where tearing the old objects down is safe.
    bool rebuildWanted = false;
    // Consecutive creations whose extent disagreed with the window drawable (the
    // launch-stretch guard) â€” reset to 0 by any creation that matches.
    uint32_t mismatchRetries = 0;
    // The F4 debug overlay. All of it is allocated the first time the menu is opened and
    // never in a run that does not open it.
    Image overlay;          // the composited panel, uploaded and blitted
    // ITS OWN UPLOAD BUFFER, not the renderer's shared `staging`. The first version used
    // `staging`, which every texture upload also writes AT OFFSET ZERO â€” and a texture
    // upload records its copy into a command buffer that executes later, so the overlay
    // overwrote bytes a pending texture copy was going to read, and vice versa. The
    // operator saw it as "issue appearing at the top of debug menu from time to time";
    // the other half of it, silently, was TEXTURES getting overlay bytes.
    Buffer overlayStage;
    // The frame BEHIND the panel, captured one frame earlier so the blend has something
    // to blend against. See the composite for why a frame of staleness is the right price.
    Image bgImage;
    Buffer bgBuffer;
    bool bgValid = false;
    uint32_t bgW = 0, bgH = 0;
    const char* dumpDir = nullptr;
    uint64_t dumpEvery = 64;
    bool dumpPending = false;
};

struct Renderer
{
    VkInstance instance = VK_NULL_HANDLE;
    bool wantSwapchain = false;
    SwapchainState swap;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    // Null unless CZ_VK_VALIDATION=1 brought VK_EXT_debug_utils in with the layer. See
    // NameObject: it is what makes a validation message name one of OUR objects.
    PFN_vkSetDebugUtilsObjectNameEXT setObjectName = nullptr;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkPhysicalDeviceMemoryProperties memProps{};

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    // The slot the frame currently being recorded belongs to, and that slot's command
    // buffer and fence hoisted out so the ~30 `R->cmd` call sites do not each have to
    // know the ring exists. Assigned in BeginFrame and nowhere else.
    FrameSlot frames[kMaxFramesInFlight];
    uint32_t frameSlot = 0;
    uint32_t framesInFlight = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool recording = false;
    bool rendering = false;

    // The EDRAM stand-in: one persistent colour target and one depth target.
    Image color;
    Image depth;
    // CZ_VK_MSAA (part 93): when the EDRAM pair above is MULTISAMPLED, these are the
    // single-sample images its samples resolve into. `colorResolve` serves the present
    // fallback (blit/readback cannot read a multisampled image); `depthResolve` serves
    // every depth-buffer resolve, because vkCmdResolveImage is defined for colour
    // formats only â€” the depth path resolves through a zero-draw dynamic-rendering
    // pass with a resolve attachment, then copies out of this image exactly as the
    // single-sample path copies out of R->depth. Both are null at 1x, where not one
    // line of the MSAA paths executes (the same-binary control arm).
    Image colorResolve;
    Image depthResolve;
    // VK_SAMPLE_COUNT_1_BIT unless CZ_VK_MSAA asked for more AND the device's
    // framebuffer limits agreed â€” decided once in CreateDevice, printed there, and
    // constant for the run (the persistent EDRAM cannot change sample count mid-frame).
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
    // SAMPLE_ZERO is the only mode Vulkan 1.2+ guarantees, and it is also the faithful
    // one: hardware's depth resolve hands the consumer one sample's value, not a blend.
    VkResolveModeFlagBits depthResolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
    Buffer readback;

    // Per-frame bump arena for constants, vertex copies and index copies. Device
    // address visible, because the translated shaders reach their constants through
    // vk::RawBufferLoad on a raw 64-bit address rather than through a descriptor.
    // With frames in flight the arena is cut into `framesInFlight` equal regions and a
    // frame bumps inside its own. A region rather than a second buffer, deliberately:
    // `GrowArenaIfNeeded`'s buffer swap, the exhaustion path and every device address
    // the draws record all stay exactly as they were, and the only line that changes is
    // where the cursor starts and stops. The buffer is allocated `framesInFlight` times
    // larger at startup so a frame's own capacity â€” the number that decided the
    // whole-frame black (Â§6ap) â€” is unchanged between the arms.
    Buffer arena;
    VkDeviceSize arenaCursor = 0;
    VkDeviceSize arenaBase = 0;    // this slot's region start
    VkDeviceSize arenaLimit = 0;   // ...and its end
    VkDeviceSize arenaHighWater = 0;
    // Set by ArenaAlloc when it has to refuse a draw; acted on at the next frame
    // boundary, which is the only place the old buffer is provably not in use.
    VkDeviceSize arenaWant = 0;

    // --- THE SHARED-BLOCK SUB-ARENA (part 111, item B1) -------------------------------
    //
    // Every draw needs 2,192 zeroed bytes for its shared-constant block, and until part
    // 111 every draw got them by bumping the arena above and calling `memset` on the
    // pump â€” `__memset_avx2`, 0.44 ms/frame and the 6th-largest symbol on the pump
    // thread at the operator's crowd. That memset has NO SOURCE: it writes a constant
    // into memory only this draw owns, so it is the one piece of per-draw work that can
    // be moved to another core with no race to argue about, which is why item B builds
    // its dispatch machinery on it first.
    //
    // A SEPARATE BUFFER, and for the same reason the cross-frame store below is one: the
    // arena's exhaustion path is load-bearing (Â§6ap), and a worker zeroing ahead of a
    // cursor inside it would have to reason about where the general bump has reached.
    // Here the worker's write region is a different VkBuffer entirely, so "the worker
    // never touches memory the pump is writing" is true by construction rather than by
    // an argument about two cursors in one address range.
    //
    // A FIXED-STRIDE SLOT ARRAY, not a bump allocator. `kSharedStride` is `kSharedSize`
    // rounded to 256, so slot i lives at `sharedBase + i * kSharedStride` and a block
    // NEVER SPANS A PRE-ZERO CHUNK. That is what lets "is this block already zeroed?" be
    // one array read instead of a range test, and it is what makes the watermark
    // bookkeeping incapable of reporting a half-zeroed block as ready.
    Buffer sharedArena;
    VkDeviceSize sharedBase = 0;      // this slot's region start, in bytes
    uint32_t sharedSlotsCapacity = 0; // what the region can hold at all
    uint32_t sharedSlotsPosted = 0;   // ...and how many of them a worker was asked to
                                      // clear this frame. THE TWO ARE DIFFERENT and the
                                      // difference is the item's whole efficiency: the
                                      // first version posted the WHOLE region every
                                      // frame â€” 56.6 MB of write-combined stores for the
                                      // 19.6 MB a crowd frame actually uses, and the
                                      // workers' side of the bill read 4.49 ms against
                                      // the 0.44 the pump was paying. The posted count is
                                      // now last frame's usage plus a margin, which is
                                      // what "a worker zeroes AHEAD OF the bump pointer"
                                      // means (Â§4.1) rather than "a worker zeroes
                                      // everything".
    uint32_t sharedNext = 0;          // next slot index within the region
    VkDeviceSize sharedWant = 0;      // set on exhaustion, acted on at the frame boundary

    // --- the CROSS-FRAME stream store -------------------------------------------------
    //
    // A SECOND buffer, deliberately not a reserved region of the arena. The arena is a
    // bump allocator reset at every swap and its exhaustion path is load-bearing â€” it is
    // what turned a too-small arena into six parts of "view-dependent whole-frame black"
    // (Â§6ap) â€” so carving a persistent region out of the bottom of it would put a new
    // moving floor under that machinery and under `GrowArenaIfNeeded`'s buffer swap. A
    // separate buffer leaves every line of that alone, at the cost of `UploadStream`
    // having to say WHICH buffer it put the bytes in. That is `StreamLoc`.
    //
    // Why this exists at all: 94% of stream lookups already hit within a frame, and the
    // ~2,000 that miss still copy 61-77 MB â€” 94-97% of it byte-identical to what the
    // previous frame copied to the same guest address (Â§6at). The cache was doing its job
    // and being thrown away at the frame boundary.
    Buffer persist;
    VkDeviceSize persistCursor = 0;
    // THE STORE'S DEVICE-LOCAL MIRROR (part 106). Part 106's decomposition put ~4.5 ms
    // of the crowd's 8.9 ms device frame at 1080p in the VERTEX FETCH: the store above
    // is host memory, so every vertex and index of every draw crossed PCIe, every frame,
    // for every tile and cascade pass. `CZ_VK_VRAM_STREAMS=1` (a host-visible VRAM heap)
    // read the GPU frame at 4.4 ms â€” but that heap is Resizable BAR, which a GTX 1060
    // does not have, and its write-combined CPU stores are what lost part 73 its wall
    // time. So the shippable form is a twin: same size, same offsets, plain VRAM, no
    // mapping. The CPU keeps writing the host store exactly as before (cached writes);
    // the ranges it wrote this frame are copied host -> mirror by ONE vkCmdCopyBuffer
    // at the start of the NEXT frame's command buffer (~0.2 MB a frame once the store
    // is warm), and a draw binds the mirror for any slot whose copy is already recorded
    // ahead of it, the host store otherwise (its first frame). The ping-pong twins
    // (`PersistEntry::alt`) make this race-free by the same argument that makes them
    // safe for the host store: a slot rewritten in frame W is read from the mirror only
    // by frames > W, whose command buffers carry its copy ahead of their draws, and a
    // slot cannot be rewritten again until the frame that last read it has retired.
    // CZ_VK_NO_STORE_MIRROR=1 is the same-binary control arm (the part-105 renderer).
    Buffer persistDev;
    std::vector<VkBufferCopy> mirrorPending;
    // Bumped once per BeginFrame AFTER the pending copies are recorded. An entry stamps
    // the generation it was written in; a hit reads the mirror iff that stamp is older
    // than the current one â€” i.e. its copy sits in this or an earlier command buffer.
    uint64_t mirrorGen = 1;
    uint64_t mirrorCopies = 0, mirrorBytes = 0, mirrorHitsDev = 0, mirrorHitsHost = 0;
    // Raised when a persistent allocation did not fit; acted on at the frame boundary,
    // which is the only place the buffer is provably not being read by the GPU.
    VkDeviceSize persistWant = 0;
    struct PersistEntry
    {
        VkDeviceSize at = 0;
        // The PING-PONG TWIN, allocated the first time this stream is caught being
        // rewritten in place and never before.
        //
        // Overwriting `at` was correct for exactly as long as the submit was
        // synchronous: the draws reading it were recorded in an earlier frame whose
        // fence had been waited on. With frames in flight that stops being true, and the
        // failure is a wrong mesh with no error anywhere. Two slots and an alternation
        // are enough â€” when frame N+1 is being recorded, frame N-1 has provably retired
        // (that is the ring's invariant), so the only slot that can still be read is the
        // one frame N used, which is the other one.
        //
        // Lazy because it is rare: ~20 streams a frame go stale out of a store holding
        // thousands, so allocating a twin for every entry would double the store to
        // protect 1% of it.
        VkDeviceSize alt = VkDeviceSize(-1);
        uint64_t mirrorSeq = 0;  // `mirrorGen` when `at` was last written (part 106)
        uint64_t guard = 0;      // StreamGuard over the guest bytes when it was copied
        uint64_t lastFrame = 0;  // for the age report; not an eviction policy yet
        uint32_t bytes = 0;
        // THIS STREAM HAS BEEN SEEN TO CHANGE, so from now on its guard is EXACT.
        //
        // Part 46, open items 00c/00k. The guard is exact up to a byte bound and sampled
        // above it, and the UI text layer â€” one big vertex buffer the guest sub-allocates
        // every run of glyphs out of â€” sits above any affordable bound: raising it to
        // 256 KB left the operator's HUD still dropping out, while part 45's unlimited
        // arm fixed it, which places the buffer above 256 KB. Buying exactness by SIZE
        // therefore costs 121 MB/frame at 256 KB and more above it, to protect a handful
        // of buffers.
        //
        // Size is the wrong discriminator. The distinction that matters is DYNAMIC vs
        // STATIC: the world's geometry is written once and read for the rest of the
        // level, while the UI buffer is rewritten every frame. A stream that has ever
        // been caught changing is dynamic, and dynamic streams are few â€” so hash those
        // exactly and keep sampling everything else. It costs nothing on the population
        // that made the bound expensive, and it cannot be fooled by a buffer's size.
        //
        // The one thing it cannot do is catch a stream whose FIRST change is invisible to
        // the sampled guard. That hole is REAL and the operator hit it on the first
        // session with this policy: "UI did break at the start of being in game but then
        // it seems to be good now" â€” the promotion is earned, so until a stream has been
        // caught changing once it is still sampled, and a miss in that window serves the
        // stale buffer exactly as before. It then self-heals, which is why the defect
        // disappeared without anything else changing.
        //
        // So the presumption is inverted for a NEW entry: a stream is hashed EXACTLY for
        // its first few observations and demoted to the sampled guard only after it has
        // proved static. That closes the window (a dynamic buffer is caught on its first
        // change rather than on its first VISIBLE change) and it is bounded â€” it costs
        // only on entries the store has just met, where the alternative was to copy the
        // whole thing anyway.
        uint32_t probes = 0;
        bool dynamic = false;
        // WHEN it was last caught changing, which is a different question from WHETHER
        // it ever was (part 67).
        //
        // `dynamic` is a latch that never unlatches, and for its original purpose â€” pick
        // the exact guard for this stream from now on â€” that is exactly right: a stream
        // that has changed once can change again and the cost of being wrong is a stale
        // mesh. The RT collector then reused the same flag to mean "this is CPU-deformed
        // smallware, keep it out of the ray structure", and there the latch is far too
        // broad: a static building whose vertex buffer the streaming system recycled ONCE
        // is excluded from every shadow for the rest of the run. That is 41% of
        // everything the collector sees (dyn=10.5M against collected=14.9M), and it is
        // why the part-67 operator session found the ray structure missing the whole
        // foreground while the placement itself was measured correct.
        //
        // A frame stamp lets the RT side ask the question it actually means: has this
        // stream been STILL for a while? A zombie rewritten every frame never settles; a
        // building rewritten at load settles in a second and stays settled â€” and because
        // its guard is then stable, its BLAS key is stable and it costs one build.
        uint64_t dynFrame = 0;
        // ...AND WHETHER IT ACTUALLY NEEDS THE EXACT GUARD, which is a different
        // question from whether it changes.
        //
        // The first version promoted every stream it caught changing, and that is far
        // too broad: a crowd's animated actor meshes are rewritten every frame, so they
        // all promoted, and 436 streams and 35 MB/frame of exact hashing showed up in
        // the operator's session (their frame-time A/B: +22.7% in the 4500-6000 draw
        // bin). But those meshes never needed it â€” they change WHOLESALE, and the cheap
        // sampled guard catches them perfectly well. The only streams that need an exact
        // guard are the ones whose changes the SAMPLED guard misses, which is the UI
        // text buffer's small edits inside a large buffer and almost nothing else.
        //
        // So both guards are computed for a promoted stream â€” the sampled one costs at
        // most kGuardBytes on top, which is noise next to the exact hash â€” and the
        // entry is demoted back to sampling once the sampled guard has proved it can see
        // this stream's changes. `needsExact` latches the moment it is caught missing
        // one, and never unlatches.
        uint64_t sampledGuard = 0;
        uint32_t sampledAgreed = 0;   // consecutive changes the sampled guard also saw
        bool needsExact = false;      // the sampled guard has been caught missing one
        // WHERE NEXT FRAME'S PRE-HASHED GUARD WILL BE (part 53 item 1.1). Written by the
        // pump when it guards this entry, so the next frame's lookup â€” which already does
        // `persistCache.find(key)` â€” costs nothing extra to find its answer. `preFrame`
        // is the frame the slot was filed FOR; anything else is a stale index and is
        // ignored, which is what makes an entry that was erased and re-created safe.
        uint32_t preSlot = UINT32_MAX;
        uint64_t preFrame = 0;
    };
    // Consecutive changes the sampled guard must catch before a stream is demoted back
    // to it. Small: this is evidence, not a warranty, and `needsExact` is permanent.
    static constexpr uint32_t kSampledProof = 3;
    // How many observations a new stream is hashed exactly for before it is trusted to
    // the sampled guard, and A PER-FRAME BYTE BUDGET for doing it.
    //
    // The budget is not a refinement, it is the whole thing working. Without it the
    // probe cost 838 streams and 66.8 MB/frame on the outdoor route -- as much as
    // hashing everything exactly, which is the cost this policy exists to avoid --
    // because a streaming world meets new geometry continuously, so "new entries only"
    // is not a small population at all. With it the probe is a fixed toll paid to the
    // oldest-unprobed entries first, so the hole still closes, just over a few frames
    // instead of instantly, and the worst case is a constant rather than the workload.
    static constexpr uint32_t kGuardProbes = 3;
    static constexpr uint64_t kGuardProbeBudget = 4u << 20;   // bytes per frame
    // THE CROSS-FRAME STORE'S INDEX IS FLAT AS OF PART 55, and it was the LARGER half of
    // the `UploadStream` cost. After the per-frame cache went flat, re-splitting the
    // function by source line said 62% of what was left was STILL hash-map machinery â€”
    // this map, probed once per first-touch stream (~2,000 times a crowd frame) against
    // a table big enough that each probe is two cold misses. Fewer lookups than the
    // per-frame cache by a factor of sixteen, and a comparable share of the thread.
    //
    // ONLY ONE OF THESE TWO IS POPULATED IN A GIVEN RUN, which is different from the
    // stream cache and the shader table, and the reason is worth stating: entries here
    // are MUTATED through the pointer a lookup returns (the guard, the ping-pong slot,
    // the promotion counters), so maintaining a shadow copy would mean mirroring every
    // mutation â€” more new code than the change itself, and a defect in the mirror would
    // look exactly like a defect in the thing being verified. So `CZ_VK_NO_FLAT_CACHE=1`
    // switches which container the accessors below use, and the FlatCache implementation
    // itself is verified where a shadow IS cheap: the stream cache and the shader table
    // check every lookup against their maps (0 of 48.5 M disagreed) and run the same
    // probe, insert and grow code.
    FlatCache<PersistEntry> persistCache;
    std::unordered_map<uint64_t, PersistEntry> persistCacheMap;
    uint64_t probeBudgetLeft = 0;   // refilled each frame; see kGuardProbeBudget
    // Counted, not sampled, because a cache that silently serves stale data looks exactly
    // like a rendering bug twenty frames later and this project has spent whole parts
    // chasing those. Plain adds on a hot struct â€” never Count(), which is a
    // std::map<std::string> lookup and would cost more than the copy it is measuring
    // (gotcha 230).
    struct PersistStats
    {
        uint64_t hits = 0;          // served across the frame boundary: bytes NOT copied
        uint64_t hitBytes = 0;
        uint64_t fills = 0;         // copied into the store for the first time
        uint64_t fillBytes = 0;
        uint64_t stale = 0;         // the guard caught a rewrite and we re-copied
        uint64_t staleBytes = 0;
        uint64_t overflow = 0;      // did not fit; fell back to the per-frame arena
        uint64_t flushes = 0;       // whole store dropped at a frame boundary
        uint64_t guardBytes = 0;    // what the guard itself read, i.e. its own cost
        // A rewritten stream needed a ping-pong twin and the store could not give it
        // one. The entry is DROPPED and the stream falls back to the per-frame arena â€”
        // never overwritten in place, which with frames in flight is the silent wrong
        // mesh this whole mechanism exists to prevent. Counted because "the safe
        // fallback fired" is a performance fact and its absence is the correctness one.
        uint64_t staleEvicted = 0;
    } persistStats;
    // ON by default, with `CZ_VK_NO_PERSIST_STREAMS=1` as the control arm â€” the same
    // shape as CZ_VK_NO_ARENA_GROWTH and CZ_VK_NO_SUBMIT, so one binary is both arms of
    // its own A/B and the old renderer stays reachable for as long as it is useful.
    bool persistOn = true;

    Buffer staging;
    VkDeviceSize stagingCursor = 0;

    // Set by capability classification. It remains false for the released modern
    // route; the Vulkan 1.1 bring-up gate is intentionally still closed below.
    bool compatibilityProfile = false;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayouts[6]{};
    VkDescriptorSet sets[6]{};
    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    // Main EDRAM scope for the Vulkan 1.1 route. Null on the modern dynamic-rendering
    // route, so none of its pipeline or command recording changes.
    VkRenderPass compatEdramRenderPass = VK_NULL_HANDLE;
    VkFramebuffer compatEdramFramebuffer = VK_NULL_HANDLE;

    // --- what is already bound on `cmd`, so a draw can skip re-binding it ------------
    //
    // Vulkan's pipeline binding, dynamic state and descriptor sets are properties of
    // the COMMAND BUFFER and persist across draws and across render-pass instances, so
    // re-issuing them per draw is pure overhead. It went unnoticed for four phases
    // because it is proportional to the draw count and the draw count was ~1,900:
    // `record` was 5% of an 85 ms frame. A Still Creek zombie crowd issues **4,900 to
    // 6,300 draws a frame**, where the same code is 21.6% of a 56 ms frame and the
    // renderer's CPU is the largest single term in it (operator session, part 18).
    //
    // Safe to cache against the command buffer because every pipeline this renderer
    // creates declares the SAME three dynamic states (viewport, scissor, blend
    // constants) and shares ONE pipeline layout â€” so a pipeline bind never invalidates
    // the dynamic state or the descriptor bindings. Reset in BeginFrame, which is the
    // one place a fresh command buffer starts. CZ_VK_NO_STATE_CACHE=1 is the arm.
    struct BoundState
    {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkViewport viewport{};
        VkRect2D scissor{};
        float blend[4]{};
        bool haveViewport = false, haveScissor = false, haveBlend = false;
        // The polygon offset, tracked like the blend constants: it is dynamic state, it
        // changes on ~3% of draws, and re-setting it on the other 97% would be three
        // driver calls a draw for nothing.
        float biasSlope = 0.0f, biasConstant = 0.0f;
        bool haveDepthBias = false;
        uint32_t stencilRef = 0, stencilMask = 0, stencilWriteMask = 0;
        bool haveStencil = false;
        bool setsBound = false;
        // The vertex and index bindings, tracked but NOT yet acted on â€” see
        // `BindSkips` for why the counter comes before the change. 16 is above the
        // highest binding this title has ever used; a draw with more is simply not
        // counted rather than counted wrongly.
        static constexpr uint32_t kMaxTrackedBindings = 16;
        VkDeviceSize vertexOffset[kMaxTrackedBindings]{};
        VkBuffer vertexBuffer[kMaxTrackedBindings]{};
        bool haveVertex[kMaxTrackedBindings]{};
        VkDeviceSize indexOffset = 0;
        VkBuffer indexBuffer = VK_NULL_HANDLE;
        VkIndexType indexType = VK_INDEX_TYPE_MAX_ENUM;
        bool haveIndex = false;
    } bound;
    // --- parallel record (part 89; the block comment is at DrawCapture) -----------
    bool parRec = false;              // CZ_VK_PAR_RECORD=1 and workers exist
    uint32_t parRecChunk = 512;       // CZ_VK_RECORD_CHUNK
    bool capActive = false;           // the OPEN pass is being captured, not recorded
    uint32_t capPassInstances = 0;    // instances this pass has produced so far
    std::vector<DrawCapture> capBuf;  // the pump's accumulating chunk
    VkImageView capColorView = VK_NULL_HANDLE, capDepthView = VK_NULL_HANDLE;
    uint32_t capWidth = 0, capHeight = 0;
    // --- deferred scoped clears (part 90; the block comment is at PendingClear) ----
    // Latched at resolve time, emitted at the head of the next pass's first instance,
    // and FLUSHED early (counted, by reader) if anything reads EDRAM before a pass
    // opens. May legitimately carry across a frame boundary: nothing reads EDRAM
    // between frames except the three flush sites. Dropped on EDRAM recreation.
    std::vector<PendingClear> pendingClears;
    // The frame's command buffers in submission order: (cb, nullptr) for a pump
    // segment, (VK_NULL_HANDLE, chunk) for a worker chunk resolved at submit.
    std::vector<std::pair<VkCommandBuffer, ParRecChunk*>> submitList;
    // The order gate's submitted side: every replayed instance's id vector, appended
    // on the pump thread in submission order (chunk at enqueue, tail at replay).
    std::vector<const std::vector<uint64_t>*> prIdSeq;
    // A deque ON PURPOSE: prIdSeq holds pointers into it, and a vector's growth
    // would silently invalidate every earlier pointer.
    std::deque<std::vector<uint64_t>> prTailIds;
    // How often each bind was skipped, across the whole run. An arm needs a counter or
    // its absence proves nothing (gotcha 151) â€” but the counter must not cost more than
    // the thing it measures. The first version used Count(), which is a
    // std::map<std::string> lookup, so the cached arm paid five map lookups per draw to
    // save five vkCmd calls and the A/B came out a dead heat BY CONSTRUCTION. These are
    // plain adds on a struct that is already hot, printed once with the stats.
    struct BindSkips
    {
        uint64_t pipeline = 0, viewport = 0, scissor = 0, blend = 0, sets = 0, draws = 0;
        uint64_t depthBias = 0, stencil = 0;
        // The vertex and index binds, COUNTED ONLY. `docs/perf-cpu-plan.md` Â§1a
        // hypothesis A is that a crowd â€” many copies of a few zombie meshes â€” rebinds
        // the same buffer at the same offset draw after draw, and that extending the
        // state cache to cover them is a dozen lines. It says to add the counters and
        // run without acting on them, because a low repeat rate kills the idea for
        // free and a high one is the justification. These are that measurement: the
        // `Repeat` counters are what a skip WOULD have saved, and they are reset
        // exactly where `BoundState` is, so they cannot promise a saving that a fresh
        // command buffer would take back.
        uint64_t vertexBinds = 0, vertexBindRepeats = 0;
        uint64_t indexBinds = 0, indexBindRepeats = 0;
    } skips;

    VkSampler linearSampler = VK_NULL_HANDLE;
    VkSampler pointSampler = VK_NULL_HANDLE;
    // 0 = the device has no samplerAnisotropy feature (checked at device creation);
    // otherwise the device's maxSamplerAnisotropy limit, read so a sampler never
    // asks for more than the device names.
    float anisoLimit = 0.0f;
    // RT stage 0 (part 61): the capability probe's answer â€” the device carries the
    // three ray-query extensions. The RT settings rows consult this to show
    // UNSUPPORTED.
    bool rtSupported = false;
    // RT stage 2 (part 64): the probe's answer ACTED ON. When the probe passes and
    // CZ_VK_RT=0 is not set, the device is created WITH the three extensions and
    // the two feature bits, and the five entry points below resolve â€” enabling an
    // extension changes no recorded command, so the OG picture is untouched, and
    // `CZ_VK_RT=0` remains the master arm under which the device is created exactly
    // as before part 64. CZ_VK_RT_FORCE=1 overrides the probe (a diagnostic arm for
    // driver experiments â€” device creation may then fail loudly, which is the
    // point). rtEnabled is the fact new code gates on; rtSupported is only the
    // probe.
    bool rtEnabled = false;
    // How many pixel shaders got a route (b) variant module. Zero means RT shadows
    // cannot engage whatever the settings row says, and the tier read says so once.
    uint32_t rtVariants = 0;
    // ---- PART 71: THE PER-DRAW HOOK FOLD -------------------------------------------
    //
    // Parts 59-70 each hung a probe on `DoDraw`, and with RT parked every one of them
    // exists only to decide not to run. At the operator's soak that is five calls x
    // ~7,000 draws x ~90 frames a second, plus a per-FETCH one: `NoteAtlasFetch` was
    // guarded on `ps.moduleRt` and NOT on whether RT is running, so a 100-second run of
    // the PARKED build did a full `DecodeTextureFetch` register decode 9,482,873 times
    // to feed a diagnostic nothing was going to read.
    //
    // The replacement is one decision per FRAME instead of five (plus one per fetch) per
    // draw. It is behaviour-preserving by construction rather than by hope: the word is
    // the OR of every hook's own arm, so whenever any hook could do work the word is true
    // and every call happens exactly as before. Both inputs are per-frame constants â€”
    // `rtshadow::Active()` is `rtEnabled` (fixed at device creation) AND `TierThisFrame()`
    // (already cached per frame on R->frame), and the two census arms are one-time `Env`
    // reads â€” so the fold cannot drift inside a frame either.
    //
    // `CZ_VK_NO_HOOK_FOLD=1` forces both words true, which restores the pre-part-71 call
    // pattern exactly and is the same-binary control arm. The counters below are what
    // prove the fold engaged (gotcha 151); with RT ON they must read ZERO, and that is
    // the identity gate.
    uint64_t hookFoldFrame = ~0ull;
    bool hooksDraw = true;       // call the five per-draw census/collect hooks
    bool hooksRtFetch = true;    // decode fetch constants for rtshadow::NoteAtlasFetch
    // ---- PART 71: THE PIPELINE CACHE ------------------------------------------------
    //
    // This renderer created every pipeline with `VK_NULL_HANDLE` as the cache from phase
    // 5 until now, so all 503-545 of a session's pipelines were compiled from scratch, on
    // the PUMP THREAD, at the moment a new draw state was first seen. That is the shape
    // of the operator's report â€” "huge stutter after loading the game" â€” and part 71's
    // soak measured a 3,891 ms frame to go with it.
    //
    // The file is keyed on the SHADER CACHE DIRECTORY's name, so the arm caches
    // (`shader_spv_a2m`, the RT variants, the probe caches) each get their own and one
    // cannot poison another. It lives outside the repo, under `$XDG_CACHE_HOME` â€” it is
    // derived data, it is device- and driver-specific, and Vulkan validates its own
    // header (vendor/device/driver UUID) and silently ignores data it cannot use, so a
    // stale or foreign file degrades to an empty cache rather than to a wrong pipeline.
    // `CZ_VK_NO_PIPELINE_CACHE=1` is the same-binary control arm and restores
    // `VK_NULL_HANDLE` exactly; `CZ_VK_PIPELINE_CACHE_FILE=path` overrides the location.
    VkPipelineCache pipeCache = VK_NULL_HANDLE;
    std::string pipeCachePath;
    uint32_t rtScratchAlign = 256;   // minAccelerationStructureScratchOffsetAlignment
    PFN_vkCreateAccelerationStructureKHR pfnCreateAS = nullptr;
    PFN_vkDestroyAccelerationStructureKHR pfnDestroyAS = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR pfnGetASBuildSizes = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR pfnCmdBuildAS = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR pfnGetASAddress = nullptr;
    // Part 41 item 1b: per-fetch samplers. Key = the fetch constant's own
    // mag/min/mip/aniso fields (dword3 bits 19..27); value = the sampler's index in
    // the set-3 heap. Index 0 stays the plain trilinear REPEAT sampler, which is
    // both the fallback and the pre-part-41 behaviour. Samplers live for the
    // process, like every other sampler here.
    //
    // A FLAT ARRAY, not a std::map, as of part 47: the key is nine bits â€” dword3 bits
    // 19..27 â€” so its whole domain is 512 entries, and this is looked up once per
    // texture fetch per draw, ~6,800 times a frame. A red-black tree over a 512-value
    // domain is a tree walk to answer a question an index answers, and the `textures`
    // phase it sits in is 42.9% of the operator's frame. -1 is "no sampler for this spec
    // yet", because 0 is a REAL answer (the plain trilinear REPEAT sampler is both the
    // fallback and the pre-part-41 behaviour), which is exactly the kind of thing a
    // zero-initialised table gets wrong.
    // The initialiser is an immediately-invoked lambda rather than a fill in
    // VkRenderer_Init on purpose: a table whose "empty" value is not zero and whose
    // filling lives somewhere else is a bug waiting for the day someone adds a second
    // construction site. This one cannot be constructed uninitialised.
    // 9 filter bits (part 41) + 3 clamp-x + 3 clamp-y bits (part 108): the fetch
    // constant's address modes are part of the spec now. 32,768 slots of int32.
    static constexpr uint32_t kSamplerSpecs = 1u << 15;
    std::vector<int32_t> samplerBySpec =
        std::vector<int32_t>(kSamplerSpecs, -1);
    uint32_t samplerCount = 1;
    Image dummy2D, dummy3D, dummyCube, dummy1D;

    // THE SHADER TABLE IS FLAT, and it was a `std::map` â€” a red-black tree probed TWICE
    // per draw. `tools/part55_srcline.py` put `std::less` + `_Rb_tree::find` at 17.9% of
    // `DoDraw`, i.e. ~4% of the whole pump thread, for two lookups in a table of a few
    // hundred entries that never changes after start-up. A tree walk of ~9 dependent
    // pointer loads is the worst possible shape for that, and none of it was visible in
    // any phase: `ProfScope(otherShader)` is the enclosing scope and it named the lookup
    // without pricing it.
    //
    // The map is kept for `CZ_VK_NO_FLAT_CACHE=1` and for the verify arm, exactly as the
    // stream cache's is; both are filled at load time, which costs nothing at run time.
    FlatCache<ShaderMeta> shaders;
    std::map<uint64_t, ShaderMeta> shadersMap;
    std::unordered_map<PipelineKey, VkPipeline, PipelineKeyHash> pipelines;
    // ...with a one-entry front cache, because consecutive draws routinely share a
    // pipeline (same shader pair, same blend/depth state) and a 48-byte compare is three
    // SIMD instructions against a hash plus a bucket probe. Its hit rate is COUNTED
    // rather than assumed: if consecutive draws did not repeat, this would be a wasted
    // compare per draw and the counter is the only thing that could say so.
    PipelineKey lastPipelineKey{};
    VkPipeline lastPipeline = VK_NULL_HANDLE;
    bool lastPipelineValid = false;
    // The draw-ID pass: the substitute fragment module, and the frame it is armed for
    // (0 = disarmed, which is every frame unless CZ_VK_DRAW_ID is set and F9 pressed).
    VkShaderModule drawIdModule = VK_NULL_HANDLE;
    // CZ_VK_NULL_PS=1 (part 106): the do-nothing fragment module bound in place of
    // EVERY translated pixel shader â€” the "everything but pixel shading" arm of the
    // GPU decomposition. See tools/null_ps.hlsl.
    VkShaderModule nullPsModule = VK_NULL_HANDLE;
    // pipelineStatisticsQuery was present and enabled (CZ_VK_GPU_STATS needs it).
    bool pipeStats = false;
    // ARMED AS A FLAG, NOT AS A FRAME NUMBER, and that is the whole lesson of building
    // this: `R->frame` is incremented by the SWAP, so the draws of a frame are recorded
    // while the counter still holds the previous frame's value. Arming "frame + 1" from
    // the present path therefore named a number the draw path never saw, and the pass
    // silently never ran â€” for three test runs, while its output was being read as if it
    // were a map. A flag consumed by the first draw that sees it cannot be off by one.
    bool drawIdArmed = false;      // set by F9, cleared when the frame is presented
    bool drawIdActive = false;     // set by the draw path: THIS recorded frame is the map
    uint64_t drawIdRanOnFrame = 0;
    // FLAT as of part 55, for the same reason as the stream caches and measured the same
    // way: after the first three tables went flat, `UploadTexture` was 11.87% of the pump
    // thread and `tools/part55_srcline.py` said ~72% of THAT was still container
    // machinery â€” this map plus two always-on census `std::map`s. Only one of the two
    // containers is populated in a given run (`CZ_VK_NO_FLAT_CACHE=1` chooses), because
    // entries here are mutated through the pointer a lookup returns â€” the guard, its
    // frame stamp, the pre-hash slot â€” exactly as in `persistCache`.
    FlatCache<TextureEntry> textures;
    std::unordered_map<uint64_t, TextureEntry> texturesMap;
    // By resolve destination, with bit 31 of the key set for a DEPTH resolve.
    //
    // The address alone is NOT an identity. `1439B000` is a shadow cascade's depth
    // destination early in a frame and the tone map's colour output late in the SAME
    // frame â€” the trace shows the final compose sampling it â€” and the capture agrees
    // (B1's 1812F000: 890 depth resolves, 852 colour). Keyed on the address alone the
    // two evict each other twice a frame, which is a device-wait and a fresh bindless
    // slot each time, i.e. gotcha 192's descriptor-heap exhaustion. A fetch picks the
    // one it meant by its own FORMAT: `k_24_8` and `k_24_8_FLOAT` are depth surfaces.
    std::unordered_map<uint32_t, Snapshot> snapshots;
    std::deque<RetiredImage> retired;   // see RetiredImage
    uint32_t nextTextureSlot = 1; // slot 0 is the dummy
    // Descriptor set 2 is its OWN unbounded array of TextureCube views, so it has its own
    // slot space. Sharing `nextTextureSlot` would work but would waste the sparser heap's
    // indices against the denser one's exhaustion â€” and this project has already had the
    // 2D heap fill mid-session and serve white (gotcha 192), so the two counters are kept
    // apart to keep that failure legible.
    uint32_t nextCubeSlot = 1;   // slot 0 is the 1x1 white cube dummy

    // CZ_VK_DRAW_CENSUS â€” the frame whose every draw is being listed, and the file it
    // goes to. Zero means disarmed, which is every frame until F9 is pressed.
    uint64_t drawCensusFrame = 0;
    uint64_t capturePictureFrame = 0;   // CZ_CAPTURE_KEY: write this frame's picture
    // THE EDGE-TRIGGERED PRESENT READBACK (part 76, item 1). Frames up to and including
    // this number take the present readback even in the swapchain arm; zero means never,
    // which is every frame of a run until F9 or F8 is pressed. Written by the key
    // handlers at the bottom of SwapBuffers and read by the readback predicate at the
    // top of the next one â€” which is the whole reason it is a frame NUMBER and not a
    // boolean: the press happens after this frame's readback decision has been made, so
    // what it can arm is the frames that follow. See the predicate for why 3.49 ms of
    // the operator's crowd frame was going into a buffer nothing displayed.
    uint64_t readbackUntilFrame = 0;
    // Counted so the arm can be shown to have engaged at all â€” the change is invisible
    // in the picture by construction, so a silent failure here looks exactly like a
    // success (gotcha 151).
    uint64_t readbackArmedPresses = 0;
    // F8's burst â€” see the write site for what it is and why a flicker needs one.
    bool burstActive = false;
    uint32_t burstSeq = 0;          // which press this is, so two bursts never collide
    uint32_t burstFrames = 0;       // frames written in THIS burst
    // Frames of this burst that found no readback pixels. Expected to be 1-2 in the
    // swapchain arm (see the write site); anything larger is the F8 readback arm failing.
    uint64_t burstNoPixelFrames = 0;
    uint64_t burstEndNs = 0;        // steady-clock deadline
    FILE* burstManifest = nullptr;
    // The burst's PER-DRAW CENSUS (part 57). Part 56's burst carried pixels and
    // per-frame fingerprints but no draw list, so a decal seen blinking could not be
    // asked whether its draw was ISSUED that frame â€” which is exactly the question the
    // burst exists to discriminate, and exactly where the analysis stopped. One file per
    // burst, one full census line per draw, each prefixed with its frame number so the
    // lines align with the PPMs exactly. Default-on with the burst (a burst is already a
    // diagnostic act); CZ_BURST_CENSUS=0 declines it, CZ_BURST_CENSUS_EVERY=N thins it.
    FILE* burstCensusFile = nullptr;
    bool burstCensusThisFrame = false;  // decided once per frame at the swap
    uint64_t burstCensusLines = 0;
    uint64_t captureSnapFrame = 0;      // ... and its resolve snapshots
    FILE* drawCensusFile = nullptr;
    uint64_t drawCensusLines = 0;

    // CZ_VK_EXPOSURE_TRACE â€” this frame's spread of the title's own exposure scalar,
    // `pc(14).w`. It exists because part 31 made that number load-bearing and there was
    // no way to read it for a NAMED frame: `CZ_VK_PSBIND` prints it, but it dedupes on
    // a key that includes the constants and caps at 64 lines, so a scalar that drifts by
    // 1e-4 a frame spends the whole budget in the first few hundred frames and says
    // nothing about the frame a snapshot was taken on.
    //
    // Min AND max, not a single value: the tone curve reads `x = colour * pc(14).w`, and
    // whether one exposure is in force for the whole frame or several are decides
    // whether a whole-frame histogram can be inverted at all.
    float expMin = 0.0f;
    float expMax = 0.0f;
    uint32_t expDraws = 0;

    // The cube maps the title renders itself, keyed on the BASE face's address, plus the
    // reverse index a resolve needs: face address -> (base address, face number). The
    // second map is what makes the refresh free â€” a resolve knows only where it wrote,
    // and without it every resolve would have to search six candidate strides.
    std::unordered_map<uint32_t, CubeSnapshot> cubeSnapshots;
    std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> cubeFaceOwner;

    // Per-frame vertex/index stream cache: one guest buffer resolved once per frame
    // however many draws read it. It records WHERE the bytes are, which since the
    // cross-frame store exists is either buffer â€” a stream served across the frame
    // boundary is registered here too, so the second and subsequent draws of that frame
    // do not even pay its guard.
    // FLAT as of part 55: the `std::unordered_map` this used to be was 13.1% of the pump
    // thread all by itself â€” see the FlatCache comment above for the measurement and for
    // why a chained map is that expensive on a lookup this hot. The map is kept beside it
    // and is used by `CZ_VK_NO_FLAT_CACHE=1` (the control arm) and by
    // `CZ_VK_VERIFY_FLAT_CACHE=1` (both structures maintained, every lookup compared).
    FlatCache<StreamLoc> streamCache;
    std::unordered_map<uint64_t, StreamLoc> streamCacheMap;

    // The constant memo's state â€” see the comment at its lookup in DoDraw. All of it is
    // written and read on the pump thread only.
    bool constMemoVsValid = false;
    // WHICH SHADER each memo slot was GATHERED for (perf item C).
    //
    // The memo hit is keyed on (constant version, window base) and NOT on the shader â€”
    // harmless while every miss wrote the whole 256-register window, because any shader
    // could then read any part of it. With the gather it is not: a slot holding shader A's
    // nine registers, served to shader B on a version match, gives B whatever the bump
    // arena left in the registers only B reads. That is a wrong constant read by the wrong
    // shader â€” the hardest defect class in this renderer to see, and one no picture gate
    // would reliably catch.
    //
    // So the slot remembers its occupant and a hit by a DIFFERENT shader re-runs the
    // gather into the same slot. Correct because the version matched (the source registers
    // are unchanged, so writing them again is idempotent), and still ~25 registers rather
    // than 256. `shadersMap` is a std::map, so the pointer is stable for the process.
    const ShaderMeta* constMemoVsFor = nullptr;
    const ShaderMeta* constMemoPsFor = nullptr;
    bool constMemoPsValid = false;
    // The vertex-fetch memo census's own two-field key; see the census block in DoDraw.
    const ShaderMeta* fetchMemoFor = nullptr;
    uint64_t fetchMemoVersion = 0;
    uint64_t fetchMemoHash = 0;
    // The stream-dedup census's per-draw memory; see UploadStream.
    uint64_t dedupDraw = ~0ull;
    uint64_t dedupKeys[16] = {};
    uint32_t dedupN = 0;
    uint64_t constMemoVsVersion = 0;
    uint64_t constMemoPsVersion = 0;
    uint64_t constMemoFrame = ~0ull;
    uint32_t constMemoVsBase = 0;
    uint32_t constMemoPsBase = 0;
    VkDeviceSize constMemoVsAt = 0;
    VkDeviceSize constMemoPsAt = 0;
    // The bound the LAST dynamic VS copy used (0 = it was a full copy). Part 88: the
    // memo verifier compares the arena against a recompute, and above this bound the
    // arena deliberately holds residue â€” a verifier unaware of it would report the
    // feature working as a memo defect (the exact interaction the gather already
    // documents for its own verifier).
    uint32_t constMemoVsDynBound = 0;

    uint64_t frame = 0;
    uint64_t drawsThisFrame = 0;
    // The previous frame's final count â€” the only complete denominator available
    // mid-frame, and what tells an instrument firing at draw N whether N is early.
    uint64_t lastFrameDraws = 0;
    // Per-frame content fingerprints, for the frame-alignment metric.
    //
    // `drawFingerprint` is FNV over every draw's (vs, ps, primitive, index count) â€”
    // it identifies WHAT the guest asked for this frame. `cameraFingerprint` is FNV
    // over the vertex shader's ALU constants at the frame's first draw, which is where
    // the view-projection matrix lives â€” it identifies WHERE the camera was.
    //
    // Both exist because "frame 600" is not a point in this title's animation: the
    // title screen renders a live 3D background driven by guest time, and our frame
    // rate varies with host load, so two runs are looking at different camera angles at
    // the same frame index. Comparing pictures across runs needs frames matched by
    // CONTENT, which is exactly what tools/xtr_determinism.py does to the capture pair.
    uint64_t drawFingerprint = 0;
    uint64_t cameraFingerprint = 0;
    // THE ORDER GATE (part 72) â€” the precondition `perf-plan-part72.md` Â§5 has called owed
    // since part 55, and the only gate that can catch the one defect parallel command
    // recording actually risks.
    //
    // Draw ORDER on this title is SEMANTIC: the PM4 stream's sequence decides blending,
    // depth and the tiled resolve order, so a parallel recorder that emits the same draws
    // in a different order produces a wrong picture that no counter, no frame time and no
    // era median would flag. `drawFingerprint` above cannot serve: it is load-bearing for
    // cross-run frame matching (tools/frame_determinism.py) so its contents cannot change,
    // and it carries neither the pipeline nor the vertex range.
    //
    // `orderLog` is the INTENDED order â€” one identity per draw, appended as `DoDraw` is
    // called, which is PM4 stream order by construction. A parallel path rebuilds the same
    // sequence from its secondaries in EXECUTION order and the two hashes must agree. On
    // the serial path the two orders are identical by definition and the check passes
    // trivially, which is exactly why `CZ_VK_ORDER_POISON` exists: a gate that cannot be
    // shown to fail has not been shown to work (gotcha 30), and this one has to be
    // provably alive BEFORE the item it guards is written, not after.
    std::vector<uint64_t> orderLog;
    uint64_t orderFramesChecked = 0, orderFramesFailed = 0, orderDrawsLogged = 0;
    // The constants the fingerprint above HASHES, kept rather than only summarised.
    // A hash can say two frames differ; only the values can say where the camera was,
    // and reproducing a shot is the whole point of the pose capture (see the .pose
    // writer at the F9 block). 16 float4 = the view-projection plus the world matrices.
    uint32_t camConsts[64] = {};
    // The same constants from the frame's LARGEST draw â€” the scene camera, where
    // camConsts is whatever the first draw used (the shadow pass's light).
    uint32_t camConstsBig[64] = {};
    uint32_t camBigVerts = 0;
    uint64_t verticesThisFrame = 0;
    // Draws recorded since the last resolve, i.e. the size of the pass that resolve is
    // closing. This is the number that separates "the pass rendered nothing because it
    // had no draws" from "the pass had 900 draws and they produced black" â€” two
    // completely different investigations that look identical in a snapshot.
    uint64_t drawsThisPass = 0;
    uint64_t verticesThisPass = 0;
    // Which resolve snapshots the draws of the current pass SAMPLED, and how many
    // textures they took from guest memory instead.
    //
    // This is the one question the existing counters cannot answer. "texture: served
    // from a resolve snapshot" proves snapshots are consumed â€” 450,488 a run â€” but not
    // by WHICH pass, and the whole of step 1 is "does the pass that writes the front
    // buffer sample the scene?". A global counter can never say that; a per-pass set
    // can, and it costs one insert per texture fetch.
    std::vector<uint32_t> snapshotsSampledThisPass;
    uint64_t guestTexturesThisPass = 0;
    // The first few draws of the pass, as (prim, indexCount, vs). A pass of ONE draw is
    // a post-processing blit, and when those are the passes producing nothing, the
    // question is which shader draws them.
    std::vector<std::string> firstDrawsThisPass;
    // The first texture the CURRENT draw bound, for the pass draw list. "This pass is
    // black" and "this pass's input was never produced" are the same picture until you
    // can name the surface each draw sampled (gotcha 140).
    uint32_t lastTexAddr = 0;
    uint32_t lastTexSlot = 0;
    // The first bound texture's DIMENSIONS, so a probe printing normalized texture
    // coordinates can state them in texels. A UV is meaningless without the size it is
    // normalized by â€” that is the whole question when two atlases differ only in size.
    uint32_t lastTexW = 0;
    uint32_t lastTexH = 0;
    uint32_t targetWidth = 1280, targetHeight = 720;
    // The EDRAM stand-in's extent, which is NOT the presented frame's (it is taller â€”
    // this title's shadow cascade is 1024 rows). Set in InitCommon beside the images.
    uint32_t edramWidth = 1280, edramHeight = 720;
    uint32_t frontBuffer = 0;
    uint32_t lastResolveDest = 0;
    uint32_t frontWidth = 0, frontHeight = 0;
    bool haveFrontSnapshot = false;

    // The staging copy of the presented frame. Filled ONLY when a picture instrument is
    // armed (part 53 item 1.3) â€” without one the present path reads the mapped readback
    // buffer directly and this stays as `VkRenderer_Init` sized it. See `wantCachedPixels`
    // in the swap for why the choice is made once from the environment.
    std::vector<uint8_t> presentPixels;
};

Renderer* R = nullptr;

// Vulkan 1.1 compatibility ABI.  These are deliberately separate from the bindless
// g_maxDescriptors heap: every draw owns a small, ordinary descriptor table that can
// be reset only when its frame slot's fence has retired.
// DoDraw publishes Xenos fetch constants 0..15 today. Keeping the fixed tables at
// that same limit cuts the per-stage sampled-image requirement from 128 to 64, which
// matters on Vulkan 1.1 Mali/older Adreno drivers. It changes only the compatibility
// ABI; the modern bindless heaps remain sized by g_maxDescriptors.
constexpr uint32_t kCompatDescriptorSlots = 16;
constexpr uint32_t kCompatDescriptorSetsPerDraw = 6;
constexpr uint32_t kCompatDescriptorDrawsPerPool = 256;

static bool CreateCompatibilityDescriptorPool(FrameSlot& frame)
{
    const VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          kCompatDescriptorSlots * 4 * kCompatDescriptorDrawsPerPool },
        { VK_DESCRIPTOR_TYPE_SAMPLER,
          kCompatDescriptorSlots * kCompatDescriptorDrawsPerPool },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 * kCompatDescriptorDrawsPerPool },
    };
    VkDescriptorPoolCreateInfo pi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pi.maxSets = kCompatDescriptorSetsPerDraw * kCompatDescriptorDrawsPerPool;
    pi.poolSizeCount = uint32_t(std::size(sizes));
    pi.pPoolSizes = sizes;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(R->device, &pi, nullptr, &pool) != VK_SUCCESS)
    {
        fprintf(stderr, "[vk] Vulkan 1.1 fixed-descriptor pool allocation FAILED\n");
        return false;
    }
    frame.compatDescriptorPools.push_back(pool);
    return true;
}

// The actual draw binder will replace only the slots the draw references and will
// write its two constant-buffer bindings in set 5.  Initialising all image/sampler
// entries here is essential: the compatibility shaders are compiled all-resources-
// bound, so an unused fetch must still see a legal white dummy, not an unbound slot.
[[maybe_unused]] static bool AllocateCompatibilityDescriptorSets(
    FrameSlot& frame, VkDescriptorSet (&out)[kCompatDescriptorSetsPerDraw])
{
    while (frame.compatDescriptorPool >= frame.compatDescriptorPools.size())
    {
        if (!CreateCompatibilityDescriptorPool(frame))
            return false;
    }
    if (frame.compatDescriptorDraw == kCompatDescriptorDrawsPerPool)
    {
        ++frame.compatDescriptorPool;
        frame.compatDescriptorDraw = 0;
        return AllocateCompatibilityDescriptorSets(frame, out);
    }

    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = frame.compatDescriptorPools[frame.compatDescriptorPool];
    ai.descriptorSetCount = kCompatDescriptorSetsPerDraw;
    ai.pSetLayouts = R->setLayouts;
    const VkResult ar = vkAllocateDescriptorSets(R->device, &ai, out);
    if (ar != VK_SUCCESS)
    {
        fprintf(stderr, "[vk] Vulkan 1.1 fixed-descriptor set allocation FAILED (%d)\n",
                int(ar));
        return false;
    }
    ++frame.compatDescriptorDraw;

    const Image* dummies[5] = {
        &R->dummy2D, &R->dummy3D, &R->dummyCube, &R->dummy1D, &R->dummy1D,
    };
    const VkDescriptorType types[5] = {
        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLER,
        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
    };
    std::array<VkDescriptorImageInfo, kCompatDescriptorSlots> infos[5]{};
    VkWriteDescriptorSet writes[5]{};
    for (uint32_t set = 0; set < 5; ++set)
    {
        for (VkDescriptorImageInfo& info : infos[set])
        {
            info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            if (set == 3)
                info.sampler = R->linearSampler;
            else
                info.imageView = dummies[set]->view;
        }
        writes[set] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[set].dstSet = out[set];
        writes[set].dstBinding = 0;
        writes[set].descriptorCount = kCompatDescriptorSlots;
        writes[set].descriptorType = types[set];
        writes[set].pImageInfo = infos[set].data();
    }
    vkUpdateDescriptorSets(R->device, uint32_t(std::size(writes)), writes, 0, nullptr);
    return true;
}

// ===================================================================================
// CZ_VK_GPU_PASSES â€” THE FRAME'S GPU TIME, SPLIT BY WHAT THE DEVICE WAS DOING (part 78)
// ===================================================================================
// This project has never had a GPU-side breakdown. `g_tsPool` gives ONE number for the
// whole command buffer â€” 8.65 ms mean on the autonomous route â€” and part 76 measured that
// number as a flat 6.97 -> 12.20 ms across a 4x range of draw counts, i.e. three quarters
// of a crowd frame's device cost is already there in a light scene. That shape says the
// cost is full-screen work rather than the crowd, and NOTHING IN THIS RENDERER COULD SAY
// WHICH full-screen work. Part 78 item 1 is that gap.
//
// WHY INTERVALS AND NOT A CHAIN OF BOUNDARIES. A chain of timestamps partitions the frame
// exactly, which sounds better and is worse: the partition is exact BY CONSTRUCTION, so it
// has no residual and therefore no way to report that it is wrong. Part 77's whole lesson
// was that a split you cannot check is a split you will believe (gotcha 456). These are
// explicit (begin, end) pairs around named regions; whatever falls BETWEEN them is
// unattributed and printed first. A large residual means the instrument is missing a
// region, not that GPU time vanished.
//
// WHAT A SEGMENT MEANS, STATED HONESTLY. Timestamps do not fence. Without a barrier the
// device may still be running work issued before a `BOTTOM_OF_PIPE` timestamp when a later
// region's commands start, so a segment's time is "the wall time between these two points
// in the queue", not "the isolated cost of this region". Every boundary here sits next to a
// layout transition or a render-scope change, which is where this renderer already
// serialises, so the attribution is meaningful â€” but a segment is an upper bound on its own
// region and the SUM is the honest quantity. The instrument is off by default and its own
// bill is measured the only way it can be: run the route with and without and compare
// `g_gpuFrameNs`, which this does not touch.
//
// The pool is per frame-in-flight slot and read back at exactly the point the existing
// frame pair is read â€” after that slot's fence has been waited on â€” so nothing here ever
// blocks to measure a block (gotcha 7).
enum GpCls : uint16_t
{
    kGpPassEmpty,     // a render scope that closed with no draws in it
    kGpPass1,         // exactly one draw
    kGpPassSmall,     // 2..255
    kGpPassBig,       // >= 256 â€” the crowd
    // THE SHADOW CASCADE (part 106): any render scope whose draws bound the 1040-pitch
    // shadow surface (IsShadowSurface), whatever its draw count. Split out because a
    // GTX-1060 budget has to know whether the cascade is a tenth of the frame or half of
    // it, and by draw count alone it hides inside ">=256" next to the scene.
    kGpPassShadow,
    kGpResolveCopy,   // vkCmdCopyImage: EDRAM -> the resolve snapshot
    kGpResolveClear,  // the title's own clear bits, honoured as vkCmdClear*Image
    kGpPresent,       // the swapchain blit, the letterbox clear and the F4 overlay
    kGpReadback,      // vkCmdCopyImageToBuffer of the whole frame (F8/F9 armed only)
    kGpCubeFace,      // a face of the cube map the TITLE renders, blitted out of a snapshot
    kGpSnapView,      // the right-sized views of a resolve snapshot, refreshed in place
    // THE TWO BARRIER CLASSES, split out of the two above them because the first version
    // of this instrument charged them to the copy and to the pass. They are the reason
    // this split was worth building: this renderer transitions the FULL-SIZE EDRAM colour
    // and depth images between attachment and transfer layouts ~49 times a frame, and a
    // layout transition of a compressed 3440x2048 attachment is not an instruction â€” on
    // every desktop GPU it is a metadata decompress over the whole image.
    kGpPassBarrier,   // BeginRendering's colour+depth transitions INTO attachment layout
    kGpResolveBarrier,// the resolve's transitions into transfer layouts and back
    kGpClasses
};
// RT has deliberately NO class. The feature is parked (part 70) and costs nothing in any
// run this instrument will see; a class that is structurally always zero reads as "we
// measured it and it is free", which is a different claim from "it was not running".
const char* const kGpNames[kGpClasses] = {
    "pass: 0 draws",   "pass: 1 draw",   "pass: 2-255 draws", "pass: >=256 draws",
    "pass: shadow cascade",
    "resolve copy",    "resolve clear",  "present blit",      "present readback",
    "cube face refresh", "snapshot views", "pass-begin barriers", "resolve barriers",
};

// 2048 queries a slot. The first version had 512 on the arithmetic "~49 passes and ~49
// resolves a frame, so ~200 timestamps" â€” and then the barrier split took the region count
// past 250, i.e. past 500 timestamps, and 1,454,670 regions were dropped. The symptom was
// NOT a missing number: it was `present blit` reading 0.15 regions a frame where exactly
// one exists, and every truncated class quietly reading low. That is why the overflow
// counter is printed rather than merely kept â€” a silently truncated frame looks like a
// saving (gotcha 3 at instrument scale).
constexpr uint32_t kGpQueriesPerSlot = 2048;
// `ext` is the pass's EXTENT KEY â€” (width << 32) | height of the largest scissor any draw in
// it used, or 0 for a region that is not a render scope. Part 79 item 2 needs it: `pass: 1
// draw` is 29.9 passes a frame at 28 us each and 17% of the device's frame, nobody has ever
// listed what those passes ARE, and the first question is whether 28 us is a full-screen
// shader or pass overhead on a 96x45 bloom target. A millisecond total cannot tell those
// apart; an extent census can, and it is the same shape as `base untile: ns/unit` (Â§6ds Â§10)
// â€” a total divided by the work it covers is what stops an argument.
// `sq` is the pass's PIPELINE-STATISTICS query (CZ_VK_GPU_STATS, part 106), or ~0u.
struct GpSeg { uint32_t q0, q1; uint16_t cls; uint64_t ext; uint32_t sq; };
VkQueryPool g_gpPool = VK_NULL_HANDLE;
std::vector<GpSeg> g_gpSegs[kMaxFramesInFlight];
uint32_t g_gpNext[kMaxFramesInFlight] = {};
uint64_t g_gpFrameOf[kMaxFramesInFlight] = { ~0ull, ~0ull, ~0ull };
uint64_t g_gpNs[kGpClasses] = {};
uint64_t g_gpN[kGpClasses] = {};
uint64_t g_gpFrames = 0, g_gpTotalNs = 0, g_gpAttribNs = 0;
uint64_t g_gpOverflow = 0, g_gpBadRead = 0;
// The resolve copies' PIXEL count, so "resolve copy" can be read as bandwidth rather than
// only as milliseconds â€” a number of pixels is what makes a fix designable.
uint64_t g_gpResolvePixels = 0;
// THE CLEARS' EXTENT, both what they DO clear and what the pass that triggered them
// actually rendered. The resolve honours the title's clear bits by clearing the WHOLE EDRAM
// stand-in â€” one `vkCmdClearColorImage` over the full subresource â€” while the resolve
// itself names the region the pass used. The ratio of these two is the ceiling on scoping
// the clear, and it is a number nobody has ever printed: the depth clear already has a
// scoped form behind `CZ_VK_SCOPED_CLEAR`, kept as an arm since part 32 because nothing
// had priced it.
uint64_t g_gpClearFullPixels = 0, g_gpClearScopedPixels = 0, g_gpClearN = 0;
// THE PASS EXTENT CENSUS (part 79 item 2), keyed by `GpSeg::ext`. Only render scopes are
// entered, and only when the instrument is on, so it costs nothing in a normal run and one
// map lookup per pass in an instrumented one. Held as (count, ns) so the table can be read
// as "N passes of this size, X us each" rather than as a share.
struct GpExtentStat { uint64_t n = 0, ns = 0, draws = 0; };
std::map<uint64_t, GpExtentStat> g_gpExtents[5];   // indexed by kGpPassEmpty..kGpPassShadow
// The render scope currently open, if any. One at a time by construction: BeginRendering
// early-returns when a scope is already open and EndRendering when none is.
int g_gpPassSeg = -1;
// DRAWS IN THE OPEN SCOPE, counted here rather than read off `R->drawsThisPass`. The first
// version of this instrument classified every pass as "0 draws" â€” 65% of the frame in a
// bucket that cannot exist â€” because `DoResolve` zeroes `drawsThisPass` in its histogram
// block BEFORE the `EndRendering` that closes the scope, so the classifier read a counter
// that had already been reset. The Â§4b cycle clock two hundred lines up says exactly that
// in a comment and captures its own class at entry. A counter owned by the thing that reads
// it cannot be reset out from under it.
uint32_t g_gpPassDraws = 0;
// The largest scissor any draw in the open scope has used, as (w << 32) | h. Updated on the
// per-draw path but ONLY when a segment is open, which is only when CZ_VK_GPU_PASSES is set
// â€” `g_gpPassSeg` is already loaded there, so the normal-run cost is a branch on a value
// that is already in a register (gotcha 453: do not put a fresh static-init guard load on
// this path).
uint64_t g_gpPassExt = 0;
uint64_t g_gpPassExtPx = 0;
// Did any draw of the open scope bind the shadow surface? Set on the per-draw path under
// the same `g_gpPassSeg >= 0` guard as the extent, folded into the class at close.
bool g_gpPassShadow = false;

// THE PIPELINE-STATISTICS CENSUS (CZ_VK_GPU_STATS=1, part 106). A GPU millisecond says
// how long a pass took; it cannot say whether the pass was vertex work, pixel work, or
// neither. The device can: a VK_QUERY_TYPE_PIPELINE_STATISTICS query around a render
// scope counts the primitives assembled, the vertex-shader invocations, the primitives
// that survived clipping and the FRAGMENT-SHADER INVOCATIONS â€” and fragment invocations
// divided by the internal resolution's pixel count is the overdraw factor, the one
// number that separates "the shaders are expensive" from "we shade the screen nine
// times". Accumulated per pass class next to the timing, printed with it.
//
// One query per pass, begun and ended inside the pump's own command buffer, which is why
// the arm forces the SERIAL recorder: a query cannot span command buffers, and under
// parallel record a pass's draws live in worker chunks the pump never sees. The GPU
// work is identical in both recorders (same draws, same order â€” the order gate proves
// it), so the census is honest about the frame even though its wall time is the
// part-88 recorder's.
constexpr uint32_t kStQueriesPerSlot = 256;
constexpr uint32_t kStCounters = 6;
const char* const kStNames[kStCounters] = {
    "IA vertices", "IA primitives", "VS invocations", "clip invocations",
    "clip primitives", "FS invocations",
};
VkQueryPool g_stPool = VK_NULL_HANDLE;
uint32_t g_stNext[kMaxFramesInFlight] = {};
uint64_t g_stSum[kGpClasses][kStCounters] = {};
uint64_t g_stN[kGpClasses] = {};
uint64_t g_stOverflow = 0, g_stBadRead = 0;

bool GpuStatsOn()
{
    static const bool on = EnvOn("CZ_VK_GPU_STATS");
    return on;
}
// CZ_VK_SCISSOR_1PX (part 106), read once at renderer init; see DoDraw for the arm.
bool g_scissor1px = false;
// CZ_VK_TRI1=1 (part 106): every draw issues at most its FIRST PRIMITIVE (3 indices).
// Every bind, every push constant, every pipeline switch and every pass stays; the
// vertex and index FETCH traffic and the vertex shading collapse to ~nothing. What the
// device frame reads under it is the per-draw front-end cost of ~9,000 draws â€” the
// third of the three arms that decompose the title's passes (with NULL_PS and
// SCISSOR_1PX). Never a mode; the picture is garbage by design.
bool g_tri1 = false;
inline uint32_t DrawCountArm(uint32_t n) { return (g_tri1 && n > 3) ? 3u : n; }

bool GpuPassesOn()
{
    static const bool on = EnvOn("CZ_VK_GPU_PASSES");
    return on;
}

// Open a segment. Returns an index into this slot's segment list, or -1 when the
// instrument is off, the device cannot timestamp, or this frame has run out of queries.
int GpuSegBegin()
{
    if (!GpuPassesOn() || g_timestampPeriodNs <= 0.0 || !g_gpPool || !R || !R->recording)
        return -1;
    const uint32_t slot = R->frameSlot;
    if (g_gpNext[slot] + 2 > kGpQueriesPerSlot)
    {
        ++g_gpOverflow;
        return -1;
    }
    const uint32_t q = slot * kGpQueriesPerSlot + g_gpNext[slot]++;
    vkCmdWriteTimestamp(R->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_gpPool, q);
    g_gpSegs[slot].push_back(GpSeg{ q, ~0u, uint16_t(kGpClasses), 0, ~0u });
    return int(g_gpSegs[slot].size()) - 1;
}

// Close a segment and give it its class. The class is decided HERE and not at the begin
// because for a render scope it is the draw count, which is only known once the scope has
// closed â€” the same reason the Â§4b cycle clock captures its class at entry and this one
// at exit.
void GpuSegEnd(int idx, GpCls cls, uint64_t ext = 0, uint32_t draws = 0)
{
    if (idx < 0 || !R || !R->recording)
        return;
    const uint32_t slot = R->frameSlot;
    if (size_t(idx) >= g_gpSegs[slot].size())
        return;
    const uint32_t q = slot * kGpQueriesPerSlot + g_gpNext[slot]++;
    vkCmdWriteTimestamp(R->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_gpPool, q);
    g_gpSegs[slot][idx].q1 = q;
    g_gpSegs[slot][idx].cls = uint16_t(cls);
    // The extent is packed into the segment rather than accumulated here because the
    // segment's DURATION is not known until its timestamps are read back, one or two frames
    // later. `draws` rides along in the top bits of nothing â€” it is accumulated at read time
    // out of the same record, so the census can say "N passes of this size, D draws, X us".
    g_gpSegs[slot][idx].ext = ext;
    (void)draws;
}

// A scope guard for the regions that have early returns. `cls` is settable so a render
// scope can name its bucket at close.
struct GpuSeg
{
    int idx;
    GpCls cls;
    explicit GpuSeg(GpCls c) : idx(GpuSegBegin()), cls(c) {}
    ~GpuSeg() { GpuSegEnd(idx, cls); }
    GpuSeg(const GpuSeg&) = delete;
    GpuSeg& operator=(const GpuSeg&) = delete;
};

// ===================================================================================
// SHADOW-RESOLUTION TIERS â€” the Shadow Quality row, wired (part 60 night item 2)
// ===================================================================================
// Until part 60 the shadow cascades rode CZ_VK_RES with everything else: the cascade
// pass renders into the shared EDRAM at the scene scale and the 4096x1024 atlas
// snapshot is created at RS(). The tier gives the SHADOW pass its own scale â€” High is
// the scene scale, Medium half of it, Low a quarter, floored at 1x of the title's own
// base â€” applied to the cascade draws' viewport/scissor and to the atlas resolve's
// copy extents and snapshot size, so the snapshot and its normalized-UV fetches stay
// consistent at any tier. At scale 1 every tier floors to 1x and the knob is honestly
// inert (said so in the panel); at 1440p High=2x Med/Low=1x; at 4K the spread is
// 4x/2x/1x.
//
// THE PASS IS IDENTIFIED BY ITS EDRAM SURFACE PITCH, measured, not assumed: a
// CZ_VK_VIEWPORT_TRACE census over the outdoor DebugJump route shows the cascade
// pass â€” and nothing else in the frame â€” bound to surfacePitch=1040, msaa=0
// (surfaceInfo=10000410; the main draws at depthControl=16/97/B7, the strip and
// full-extent clears included). 1040 is not arbitrary: EDRAM pitch aligns to the
// 80-pixel tile, and 1024 rounds up to 13*80 = 1040 â€” i.e. the pitch MEANS "a
// 1024-wide EDRAM surface", and this title's only 1024-wide surface is the shadow
// cascade. The same predicate serves DoDraw and DoResolve, reading the same live
// register, so a pass can never be drawn at one scale and resolved at another.
inline bool IsShadowSurface(const uint32_t* regs)
{
    return (regs[xenos::kRbSurfaceInfo] & 0x3FFF) == 1040 &&
           ((regs[xenos::kRbSurfaceInfo] >> 16) & 3) == 0;
}

// The shadow pass's own internal resolution this frame. `CZ_VK_SHADOW_TIER=0|1|2`
// is the measurement arm and wins over the settings file (the env-wins rule);
// otherwise the panel's persisted Shadow Quality row drives it, re-read once per
// FRAME so a menu change applies live while draws and resolves within one frame
// always agree. Tier 2 (High) is the scene resolution â€” bit-identical to the
// pre-tier renderer, the control arm. Medium halves the HEIGHT (floored at the
// title's own 720) and Low quarters it; the width follows so the pass keeps the
// scene's aspect.
void ShadowRes(uint32_t& sw, uint32_t& sh)
{
    static const int envTier = [] {
        if (const char* e = Env("CZ_VK_SHADOW_TIER"))
        {
            const long t = strtol(e, nullptr, 10);
            if (t >= 0 && t <= 2)
            {
                fprintf(stderr, "[vk] CZ_VK_SHADOW_TIER=%ld â€” the env arm wins over "
                                "the settings file's shadow_tier\n", t);
                return int(t);
            }
            fprintf(stderr, "[vk] CZ_VK_SHADOW_TIER=%s is not 0..2 â€” IGNORED\n", e);
        }
        return -1;
    }();
    static uint64_t cachedFrame = ~0ull;
    static uint32_t cachedW = 0, cachedH = 0;
    if (!cachedW || cachedFrame != R->frame)
    {
        cachedFrame = R->frame;
        int tier = envTier >= 0 ? envTier : Settings_ShadowTier();
        // CZ_TEST_TIER_FLIP=N â€” cycle the tier 2,1,0 every N frames, headlessly
        // exercising the LIVE flip the panel row performs (the shadow-tier freeze's
        // repro arm: the resize path destroys and rebuilds the atlas mid-run on
        // every boundary this crosses).
        static const long flipEvery = [] {
            const char* e = Env("CZ_TEST_TIER_FLIP");
            return e ? strtol(e, nullptr, 10) : 0;
        }();
        if (flipEvery > 0)
            tier = 2 - int((R->frame / uint64_t(flipEvery)) % 3);
        uint32_t w, h;
        InternalRes(w, h);
        uint32_t th = tier == 2 ? h : tier == 1 ? h / 2 : h / 4;
        if (th < 720)
            th = 720;
        cachedH = th;
        cachedW = uint32_t(uint64_t(w) * th / h);
        if ((cachedW & 1) != 0)
            --cachedW;   // even, so the tile arithmetic stays whole pixels
    }
    sw = cachedW;
    sh = cachedH;
}

// THE RENDERER-SIDE FOV PATCH IS A MEASUREMENT ARM ONLY as of part 62: the
// settings row drives the GAME-SIDE substitution (cpu/camera_fov.cpp), which
// widens rendering AND the title's own CPU culling â€” this renderer patch could
// only widen rendering, and the un-widened culling popped objects at the flanks
// (the operator's report). `CZ_VK_FOV=N` still applies the renderer patch (the
// A/B arm and the composite-analysis instrument), and CZ_TEST_FOV_FLIP still
// exercises it; the Settings_Fov() source is deliberately GONE from here â€” two
// mechanisms driving one slider would double-apply.
float FovHalfRadThisFrame()
{
    static const bool envSet = Env("CZ_VK_FOV") != nullptr;
    static const int envDeg = [] {
        if (const char* e = Env("CZ_VK_FOV"))
        {
            const long d = strtol(e, nullptr, 10);
            if (d >= -10 && d <= 30)
            {
                fprintf(stderr, "[vk] CZ_VK_FOV=%ld â€” the env arm wins over the "
                                "settings file's fov\n", d);
                return int(d);
            }
            fprintf(stderr, "[vk] CZ_VK_FOV=%s is outside -10..+30 â€” IGNORED, "
                            "slider pinned to OG\n", e);
        }
        return 0;
    }();
    // CZ_TEST_FOV_FLIP=N â€” alternate the slider between 0 and +20 every N frames,
    // in ONE process. The part-62 lesson made this mandatory kit: a two-run picture
    // A/B of a projection change is confounded by camera drift between runs (that is
    // exactly how part 61 convinced itself the world moved when only the UI did),
    // where a same-run flip holds the camera and makes "did the WORLD change"
    // answerable from two dumped frames. Same shape as CZ_TEST_TIER_FLIP
    // (live-settings need live verification).
    static const long flipEvery = [] {
        const char* e = Env("CZ_TEST_FOV_FLIP");
        return e ? strtol(e, nullptr, 10) : 0;
    }();
    static uint64_t cachedFrame = ~0ull;
    static float cachedHalfRad = 0.0f;
    if (cachedFrame != R->frame)
    {
        cachedFrame = R->frame;
        int deg = envSet ? envDeg : 0;
        if (flipEvery > 0)
            deg = (R->frame / uint64_t(flipEvery)) % 2 ? 20 : 0;
        cachedHalfRad = float(deg) * 0.00872664626f;   // pi/360: degrees -> half-radians
    }
    return cachedHalfRad;
}

// The cross-frame store's index, through one seam so the container is a runtime choice.
// A raw pointer rather than an iterator because that is what both containers can return
// and because every caller wanted `->second` anyway. It stays valid until the next
// insert into the same container, which is exactly the guarantee the old iterator gave.
Renderer::PersistEntry* PersistFind(uint64_t key)
{
    if (g_pardrawCensus)
        ++g_pdc.persistFinds;
    if (!g_flatCacheOff)
        return R->persistCache.Find(key);
    auto it = R->persistCacheMap.find(key);
    return it != R->persistCacheMap.end() ? &it->second : nullptr;
}
Renderer::PersistEntry* PersistInsert(uint64_t key, const Renderer::PersistEntry& e)
{
    PDC_SCOPE(persistMutNs);
    if (g_pardrawCensus)
        ++g_pdc.persistInserts;
    if (!g_flatCacheOff)
        return R->persistCache.Insert(key, e);
    return &R->persistCacheMap.emplace(key, e).first->second;
}
void PersistErase(uint64_t key)
{
    if (!g_flatCacheOff)
        R->persistCache.Erase(key);
    else
        R->persistCacheMap.erase(key);
}
void PersistClear()
{
    R->persistCache.Clear();
    R->persistCacheMap.clear();
    R->mirrorPending.clear();
}

// Forward declarations for the store mirror (part 106); both are defined further down.
bool CreateBuffer(Buffer& b, VkDeviceSize size, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags props, bool deviceAddress,
                  const char* vramName = nullptr);
VkBufferUsageFlags PersistUsage();

// Queue a host-store range for the mirror and stamp the entry (part 106).
inline void MirrorMark(Renderer::PersistEntry& e, VkDeviceSize at, VkDeviceSize bytes)
{
    if (!R->persistDev.buffer)
        return;
    PDC_SCOPE(persistMutNs);
    e.mirrorSeq = R->mirrorGen;
    if (at + bytes <= R->persistDev.size)
    {
        if (g_pardrawCensus)
            ++g_pdc.mirrorPushes;
        R->mirrorPending.push_back(VkBufferCopy{ at, at, bytes });
    }
}

// Which buffer a persist HIT binds: the mirror when its copy of this slot is already in
// the queue ahead of this draw, the host store otherwise.
inline StreamLoc PersistHitLoc(const Renderer::PersistEntry& e)
{
    if (R->persistDev.buffer && e.mirrorSeq < R->mirrorGen &&
        e.at + e.bytes <= R->persistDev.size)
    {
        ++R->mirrorHitsDev;
        return StreamLoc{ &R->persistDev, e.at };
    }
    ++R->mirrorHitsHost;
    return StreamLoc{ &R->persist, e.at };
}

// Create (or re-create, after a growth) the mirror at the store's size, or as much of
// it as a quarter of the device-local heap allows, or not at all â€” every outcome is
// printed, because a performance number from an unknown memory placement is not
// comparable with anything (the rule the arena's own line follows).
void CreateStoreMirror()
{
    if (R->persistDev.buffer)
    {
        vkDestroyBuffer(R->device, R->persistDev.buffer, nullptr);
        vkFreeMemory(R->device, R->persistDev.memory, nullptr);
        R->persistDev = Buffer{};
    }
    R->mirrorPending.clear();
    static const bool off = EnvOn("CZ_VK_NO_STORE_MIRROR");
    if (off || !R->persistOn || !R->persist.buffer)
    {
        if (off)
            fprintf(stderr, "[vk] CZ_VK_NO_STORE_MIRROR=1 â€” the stream store stays in "
                            "system RAM and the GPU fetches every vertex over PCIe (the "
                            "part-105 renderer, same binary)\n");
        return;
    }
    VkDeviceSize heapBytes = 0;
    for (uint32_t h = 0; h < R->memProps.memoryHeapCount; ++h)
        if (R->memProps.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            heapBytes = std::max(heapBytes, R->memProps.memoryHeaps[h].size);
    VkDeviceSize want = R->persist.size;
    // A quarter of the largest device-local heap at most: the textures, the EDRAM
    // stand-in and the driver's own allocations share it, and a mirror that pages is a
    // regression wearing an optimisation's name. A smaller mirror is still a mirror â€”
    // slots past its end simply stay on the host path.
    while (want > heapBytes / 4 && want > (64ull << 20))
        want /= 2;
    if (!CreateBuffer(R->persistDev, want, PersistUsage() | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, /*deviceAddress=*/true,
                      "cross-frame stream store MIRROR"))
    {
        R->persistDev = Buffer{};
        fprintf(stderr, "[vk] the stream store mirror could not be allocated in video "
                        "memory â€” the GPU reads the store from system RAM this run\n");
        return;
    }
    R->persistDev.shadowMapped = R->persist.mapped;
    fprintf(stderr,
            "[vk] stream store MIRROR: %llu MB of the %llu MB store twinned in video "
            "memory (device-local heap %llu MB); draws bind the mirror once a slot's "
            "copy is queued ahead of them. CZ_VK_NO_STORE_MIRROR=1 is the control arm\n",
            (unsigned long long)(R->persistDev.size >> 20),
            (unsigned long long)(R->persist.size >> 20),
            (unsigned long long)(heapBytes >> 20));
}

// Record this frame's host -> mirror copies at the top of the frame's command buffer
// (no rendering instance is open here, which vkCmdCopyBuffer requires), then bump the
// generation so this frame's hits on those slots bind the mirror.
void MirrorFlush()
{
    if (R->persistDev.buffer && !R->mirrorPending.empty())
    {
        const std::vector<VkBufferCopy>& v = R->mirrorPending;
        for (size_t i = 0; i < v.size(); i += 4096)
            vkCmdCopyBuffer(R->cmd, R->persist.buffer, R->persistDev.buffer,
                            uint32_t(std::min<size_t>(4096, v.size() - i)), v.data() + i);
        VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
                           VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(R->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                                 VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
        R->mirrorCopies += v.size();
        for (const VkBufferCopy& c : v)
            R->mirrorBytes += c.size;
        R->mirrorPending.clear();
    }
    ++R->mirrorGen;
}
size_t PersistSize()
{
    return g_flatCacheOff ? R->persistCacheMap.size() : R->persistCache.Size();
}

// The texture cache, through the same seam and for the same reason.
// THE TEXTURE-RESOLUTION GENERATION (part 109 item 1). `UploadTexture` is called once per
// texture fetch per draw â€” ~13,900 times a frame at the crowd â€” and 0.0014% of those calls
// do any work: the rest hash six fetch-constant dwords, decode them, and look the result
// up. The fetch constants for a slot almost never change between draws, so the answer is
// memoisable per fetch-constant index â€” but ONLY against everything else the answer
// depends on, which is why this counter exists rather than a bare cache.
//
// It is bumped by every mutation of the two things a resolved slot depends on:
//   * the resolve-snapshot set (emplace, erase, and the whole-set clear) â€” a snapshot
//     appearing mid-frame changes a fetch's answer from "the cached upload" to "the
//     snapshot", which is the ordering the big comment in UploadTexture exists to protect;
//   * the texture table (TexInsert, and BOTH arms of ReclaimTextureSlot's eviction) â€” a
//     recycled bindless slot would otherwise be handed out from a stale memo.
// Miss anything that changes an answer and the symptom is a frozen or wrong texture, so
// the memo ships with CZ_VK_TEXMEMO_VERIFY=1, which computes both answers and counts
// disagreements.
uint64_t g_texGen = 1;
inline void TexGenBump() { ++g_texGen; }

TextureEntry* TexFind(uint64_t key)
{
    if (g_pardrawCensus)
        ++g_pdc.texFinds;
    TextureEntry* e = nullptr;
    if (!g_flatCacheOff)
        e = R->textures.Find(key);
    else
    {
        auto it = R->texturesMap.find(key);
        e = it != R->texturesMap.end() ? &it->second : nullptr;
    }
    // Recency stamp for the LRU reclaimer: a lookup this frame IS a use this frame.
    // One uint64 store on the ~40-50k/frame lookup path â€” negligible.
    if (e)
        e->lastUsedFrame = R->frame;
    return e;
}
void TexInsert(uint64_t key, TextureEntry&& e)
{
    PDC_SCOPE(texInsertNs);
    if (g_pardrawCensus)
        ++g_pdc.texInserts;
    TexGenBump();
    if (!g_flatCacheOff)
        R->textures.Insert(key, e);
    else
        R->texturesMap.emplace(key, std::move(e));
}
size_t TexSize()
{
    return g_flatCacheOff ? R->texturesMap.size() : R->textures.Size();
}

#define VK_CHECK(expr, what)                                                           \
    do                                                                                 \
    {                                                                                  \
        const VkResult vkr_ = (expr);                                                  \
        if (vkr_ != VK_SUCCESS)                                                        \
        {                                                                              \
            fprintf(stderr, "[vk] %s failed: VkResult %d\n", what, int(vkr_));         \
            return false;                                                              \
        }                                                                              \
    } while (0)

uint32_t FindMemoryType(uint32_t typeBits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < R->memProps.memoryTypeCount; i++)
        if ((typeBits & (1u << i)) &&
            (R->memProps.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

// The memory properties a READBACK buffer wants â€” which are not the ones every other
// host-visible buffer here wants, and that difference is worth a function.
//
// `FindMemoryType` returns the FIRST type satisfying the mask, and on a discrete GPU
// the first HOST_VISIBLE|HOST_COHERENT type is WRITE-COMBINED: writes stream to the
// device beautifully and reads are UNCACHED, running at a few hundred MB/s. Every other
// mapped buffer in this renderer is write-only from the CPU (the arena, the staging
// buffer), so that type is right for them. The readback buffer is the one buffer the
// CPU READS, and it had inherited the same mask â€” so presenting a frame meant reading
// 3.7 MB back over an uncached mapping. Measured at 15.7% of a 103 ms gameplay frame,
// which is ~230 MB/s and is what uncached reads look like.
//
// HOST_CACHED is the fix and it is the whole fix. Asking for it is a preference rather
// than a requirement because an integrated GPU may not offer the combination, and a
// renderer that refuses to start is worse than one that reads slowly.
// Did the readback buffers end up HOST_CACHED? Read by the present path: with cached
// memory the mapped buffer can be handed straight to the instruments and to the window,
// and the staging copy that used to sit in front of it is pure cost. With UNCACHED
// memory it is not â€” several instruments walk the whole frame, and each walk would be a
// write-combined read. See `stagingCopy` in the swap.
bool g_readbackCached = false;

VkMemoryPropertyFlags ReadbackMemoryProps()
{
    const VkMemoryPropertyFlags base = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    // CZ_VK_READBACK_UNCACHED=1 â€” the pre-fix arm, i.e. the write-combined readback.
    // The same-binary control for the frame-rate claim this change makes.
    if (EnvOn("CZ_VK_READBACK_UNCACHED"))
    {
        fprintf(stderr, "[vk] readback buffer forced UNCACHED (the pre-fix arm)\n");
        return base;
    }
    for (uint32_t i = 0; i < R->memProps.memoryTypeCount; i++)
    {
        const VkMemoryPropertyFlags f = R->memProps.memoryTypes[i].propertyFlags;
        if ((f & base) == base && (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT))
        {
            g_readbackCached = true;
            return base | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        }
    }
    fprintf(stderr, "[vk] no HOST_CACHED memory type â€” the readback stays uncached\n");
    return base;
}

// ===================================================================================
// GEOMETRY IN VRAM â€” the operator's question, and it was a real gap
// ===================================================================================
//
// THEIR QUESTION, closing part 55: "shouldn't we load texture, frame buffers,
// geometry/meshes, shadow maps, lighting and refraction data, shaders and cached assets
// from vram. Especially since it's a old game shouldn't take much vram and then we use ram
// as fallback and page when we really have nothing left in case someone is running it on a
// 2gb laptop with no gpu."
//
// Most of that list already was in VRAM â€” every `VkImage` (textures, render targets,
// shadow maps, the resolve snapshots) allocates `DEVICE_LOCAL`, and shader modules are the
// driver's own. **Geometry was not.** The 512 MB cross-frame stream store and the
// per-frame arena â€” which between them hold every vertex, every index and every ALU
// constant this renderer draws from â€” asked for `HOST_VISIBLE | HOST_COHERENT`, and
// `FindMemoryType` returns the FIRST type matching, which on this machine is
// `memoryTypes[3]` on heap 1: system RAM. So every draw was fetching its vertices across
// PCIe from a buffer the GPU could have owned outright.
//
// WHAT MAKES IT POSSIBLE NOW is a heap that did not exist on hardware of this game's era.
// With Resizable BAR the GPU exposes `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` over
// its whole VRAM (`memoryTypes[5]`, heap 0, 8 GiB on the operator's RTX 3070) â€” CPU-
// writable video memory. Without ReBAR the same type exists but covers only a 256 MB
// window, and on an integrated part there is no separate heap at all. So this is a
// PREFERENCE with a fallback, exactly as the operator framed it, and the same shape as
// the thread budget: ask the machine what it has, take what is sensible, leave the rest.
//
// THE ONE THING THAT WOULD MAKE THIS A DISASTER, checked rather than assumed. CPU-visible
// device memory is WRITE-COMBINED: sequential writes are fine and fast, but a READ is an
// uncached fetch across PCIe and can be a hundred times slower than a cached load. Every
// write into these buffers is `CopySwapped`, which streams forward â€” ideal for WC. The
// only two places that READ back from them (`loc.bytes()` in the rect trace and in the
// index-range census) are both behind diagnostic environment flags that are off by
// default, so no default-path code reads this memory at all. They are named here because
// a future edit that reads a vertex buffer on the CPU would be silently catastrophic
// rather than merely wrong.
//
// `CZ_VK_NO_VRAM_STREAMS=1` is the same-binary control arm and puts them back in RAM.
// Every big buffer prints which heap it landed in and how large that heap is, because a
// performance number taken from an unknown memory type is not comparable with anything â€”
// the same rule the thread budget's start-up line exists for.

// Prefer `props | DEVICE_LOCAL`, fall back to `props`. Returns the type index and, in
// `gotDeviceLocal`, whether the preference was actually honoured â€” never inferred from
// the request, because the whole point is that it can fail.
uint32_t FindMemoryTypePreferDevice(uint32_t typeBits, VkMemoryPropertyFlags props,
                                    VkDeviceSize size, bool* gotDeviceLocal,
                                    bool storeOnlyArm)
{
    *gotDeviceLocal = false;
    // AN ARM, NOT THE DEFAULT, and deliberately so. The placement is a one-line
    // preference; whether it is FASTER is a question about write-combined memory on this
    // machine with this workload, and nothing here has measured it yet. One read hazard
    // is also still open: `SynthRectStream` assembles on the stack and writes once (this
    // part fixed that), but its SOURCE is still a pointer into the buffer, so it reads
    // three vertices per rect draw across the bus in this arm. Shipping a default whose
    // sign is unknown is how a regression arrives wearing an optimisation's name â€” the
    // swapchain took the same route in part 54 and became the default on a measurement,
    // not on an argument.
    static const bool vram = EnvOn("CZ_VK_VRAM_STREAMS");
    // CZ_VK_VRAM_STORE=1 (part 106): the SPLIT. Part 73's arm put the per-frame arena AND
    // the cross-frame store in video memory and lost 14% of WALL time in a CPU-bound
    // regime, because the arena carries ~8 KB of shader constants per draw â€” written
    // once, read once â€” and write-combined PCIe writes of those cost the CPU more than
    // the GPU saved (gotcha 363). The STORE is the opposite shape: written 0.2 MB a
    // frame (only streams whose content changed), read ~100 MB a frame by the vertex
    // fetch across every tile and cascade pass. Part 106's census put the crowd's device
    // frame at ~2 ns per vertex invocation with NO fragments, ten times the vertex
    // throughput â€” the GPU is fetching its geometry over PCIe. This arm moves ONLY the
    // store; the arena stays where the CPU writes it cheaply.
    static const bool vramStore = EnvOn("CZ_VK_VRAM_STORE");
    if (vram || (vramStore && storeOnlyArm))
    {
        const uint32_t t =
            FindMemoryType(typeBits, props | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (t != UINT32_MAX)
        {
            // LEAVE THE MACHINE USABLE, which is the operator's own rule one subsystem
            // over. A 512 MB store into a 256 MB pre-ReBAR window, or into the 2 GB of a
            // laptop that also has to hold every texture, is not an optimisation â€” it is
            // an allocation failure or a driver silently paging it back out, which would
            // read as "the change made it slower" with nothing saying why. Take the heap
            // only if this buffer is at most a quarter of it.
            const uint32_t heap = R->memProps.memoryTypes[t].heapIndex;
            const VkDeviceSize heapSize = R->memProps.memoryHeaps[heap].size;
            if (size * 4 <= heapSize)
            {
                *gotDeviceLocal = true;
                return t;
            }
        }
    }
    return FindMemoryType(typeBits, props);
}

bool CreateBuffer(Buffer& b, VkDeviceSize size, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags props, bool deviceAddress,
                  const char* vramName)
{
    b.size = size;
    VkBufferCreateInfo ci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    ci.size = size;
    ci.usage = usage | (deviceAddress ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0);
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(R->device, &ci, nullptr, &b.buffer), "vkCreateBuffer");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(R->device, b.buffer, &req);
    bool inVram = false;
    const uint32_t type =
        vramName ? FindMemoryTypePreferDevice(req.memoryTypeBits, props, req.size, &inVram,
                                              strcmp(vramName, "cross-frame stream store") == 0)
                 : FindMemoryType(req.memoryTypeBits, props);
    if (type == UINT32_MAX)
    {
        fprintf(stderr, "[vk] no memory type for buffer (props %u)\n", props);
        return false;
    }
    if (vramName)
    {
        const uint32_t heap = R->memProps.memoryTypes[type].heapIndex;
        // "VIDEO MEMORY" is a fact about the TYPE chosen, not about which preference
        // chose it: a plain DEVICE_LOCAL request (the store mirror) lands in VRAM without
        // going through the ReBAR arm, and printed "system RAM" until part 106.
        inVram = (R->memProps.memoryTypes[type].propertyFlags &
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        fprintf(stderr,
                "[vk] %s: %llu MB in %s (memory type %u, heap %u of %llu MB)%s\n",
                vramName, (unsigned long long)(size >> 20),
                inVram ? "VIDEO MEMORY" : "system RAM", type, heap,
                (unsigned long long)(R->memProps.memoryHeaps[heap].size >> 20),
                inVram ? "" : " â€” CZ_VK_VRAM_STREAMS=1 puts geometry in VRAM where a "
                              "CPU-writable device-local heap is big enough "
                              "(CZ_VK_VRAM_STORE=1: the cross-frame store only)");
    }

    VkMemoryAllocateFlagsInfo flags{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.pNext = deviceAddress ? &flags : nullptr;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    VK_CHECK(vkAllocateMemory(R->device, &ai, nullptr, &b.memory), "vkAllocateMemory");
    VK_CHECK(vkBindBufferMemory(R->device, b.buffer, b.memory, 0), "vkBindBufferMemory");

    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        VK_CHECK(vkMapMemory(R->device, b.memory, 0, size, 0,
                             reinterpret_cast<void**>(&b.mapped)),
                 "vkMapMemory");

    if (deviceAddress)
    {
        VkBufferDeviceAddressInfo di{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
        di.buffer = b.buffer;
        b.address = vkGetBufferDeviceAddress(R->device, &di);
    }
    return true;
}

// Attach a human name to a Vulkan object, so the validation layer prints it instead of a
// handle. A no-op without CZ_VK_VALIDATION â€” the function pointer is null then, and the
// arguments are not even formatted.
//
// This exists because of what part 25's first validation session could NOT say. Four of
// its five defects were identifiable by reading the code; the fifth â€” a sampled image
// still VK_IMAGE_LAYOUT_UNDEFINED when a draw reads it â€” names an image, and this
// renderer creates images as EDRAM targets, guest textures, resolve snapshots, sized
// views of snapshots and dummies. A handle distinguishes none of those, and an undefined
// layout is undefined CONTENT: a wrong picture with no counter anywhere.
void NameObject(uint64_t handle, VkObjectType type, const char* fmt, ...)
{
    if (!R->setObjectName || !handle)
        return;
    char name[160];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof name, fmt, ap);
    va_end(ap);
    VkDebugUtilsObjectNameInfoEXT ni{ VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
    ni.objectType = type;
    ni.objectHandle = handle;
    ni.pObjectName = name;
    R->setObjectName(R->device, &ni);
}

void NameImage(const Image& img, const char* fmt, ...)
{
    if (!R->setObjectName)
        return;
    char name[128];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof name, fmt, ap);
    va_end(ap);
    NameObject(uint64_t(img.image), VK_OBJECT_TYPE_IMAGE, "%s", name);
    NameObject(uint64_t(img.view), VK_OBJECT_TYPE_IMAGE_VIEW, "%s view", name);
}

// ===================================================================================
// THE IMAGE MEMORY POOL â€” one VkDeviceMemory per texture was 70% of the texture decode
// ===================================================================================
//
// WHAT WAS MEASURED (part 77, phase5-notes Â§6ds). `CreateImage` is 342.9 ms of the 496.1 ms
// texture decode on the autonomous route, and its own five-call split says where:
//
//     vkCreateImage 3.3 ms | memReq 0.3 | vkAllocateMemory 350.6 | bind 2.2 | view 4.3
//
// **`vkAllocateMemory` is 145 us per call and there are 2,424 of them** â€” one per texture,
// because `CreateImage` has allocated a dedicated `VkDeviceMemory` for every image since
// phase 5. It is a kernel-side buffer-object allocation, not a user-space bookkeeping
// operation, and no amount of parallelising the untile loop (10.5% of the same scope) or
// batching the staging submits touches it.
//
// AND THIS FILE ALREADY KNEW. The RT work built exactly this pool for acceleration
// structures â€” see `AsChunk` and its comment: *"a crowd's ~2,600 BLASes against the
// driver's ~4096 maxMemoryAllocationCount would exhaust the allocator **with textures still
// to serve**"*. The hazard was correctly identified, the pattern was correctly built for
// the NEW subsystem, and the subsystem it was competing with kept the anti-pattern. That is
// part 75's gotcha 446 a second time: a defect fixed at one pointer and never generalised.
//
// SO THE CAP IS A LATENT FAILURE TOO, not only a cost. This 60-second route consumed
// **2,424 of the driver's ~4,096 allocations**. A long play session in a texture-heavy era
// runs out, and `vkAllocateMemory` then fails â€” a VK_CHECK abort, not a degraded picture.
//
// WHY A BUMP POOL IS SAFE HERE, stated so it can be argued with rather than assumed: a
// texture image in this renderer is **never destroyed while the process runs**. The cache
// (`R->textures`, a FlatCache) grows and never evicts; the re-upload path (`refresh`) reuses
// the existing image; `RetireImage` is called for SNAPSHOTS and the RT factor image and for
// nothing else. So there is no free list to build â€” the blocks are released once, at
// shutdown. Snapshots keep their dedicated allocations precisely because they DO get
// retired mid-run, and pooling something whose lifetime is a frame would leak.
//
// A pooled image carries `memory == VK_NULL_HANDLE`, which is what makes this safe against
// the code that already exists: every destroy site in this file is guarded by
// `if (im.memory)`, so a pooled image that somehow reached one would free nothing rather
// than free a block five other live images are bound into.
//
// `CZ_VK_NO_TEX_MEMPOOL=1` is the same-binary control arm â€” one dedicated allocation per
// texture, i.e. the renderer as it was through part 76.
struct ImgBlock
{
    uint32_t typeIndex = 0;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0, used = 0;
};
std::vector<ImgBlock> g_imgBlocks;
// 32 MB. The route places 392 MB of texture images, so this is ~13 allocations where there
// were 2,424, and the worst case wasted is one partly-filled block. Bigger blocks would
// save a handful more allocations and waste more VRAM on a machine that never fills them.
constexpr VkDeviceSize kImgBlockBytes = 32ull << 20;
// ENGAGEMENT EVIDENCE (gotcha 151). `g_imgPooled` is binds served from a block and
// `g_imgDedicated` is images that still took their own allocation â€” a pool whose counter
// reads zero because every image fell down the fallback path would otherwise be invisible.
uint64_t g_imgPooled = 0, g_imgDedicated = 0, g_imgBlockAllocs = 0, g_imgPoolWasteBytes = 0;

bool CreateImage(Image& img, uint32_t w, uint32_t h, VkFormat format,
                 VkImageUsageFlags usage, VkImageAspectFlags aspect,
                 VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D, uint32_t layers = 1,
                 uint32_t depthExtent = 1,
                 VkComponentMapping components = VkComponentMapping{},
                 uint32_t levels = 1, bool poolMemory = false,
                 VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT)
{
    img.width = w;
    img.height = h;
    img.layers = layers;
    img.levels = levels;
    img.format = format;
    img.layout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImageCreateInfo ci{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    // THE IMAGE TYPE COMES FROM THE VIEW TYPE, not from the depth extent. It used to be
    // `depthExtent > 1 ? 3D : 2D`, which is right for every image built out of a guest
    // surface and WRONG for the 1x1 dummies: `makeDummy(dummy3D, VIEW_TYPE_3D, 1, 1, 1)`
    // and `makeDummy(dummy1D, VIEW_TYPE_1D, ...)` both pass depth 1, so both got a
    // VK_IMAGE_TYPE_2D image under a view that Vulkan requires to match. That is
    // VUID-VkImageViewCreateInfo-subResourceRange-01021, one of the five the validation
    // layer reported the hour it was installed (open item 00d) â€” and it had been failing
    // since phase 5, silently, because the layer was not present to say so.
    ci.imageType = (viewType == VK_IMAGE_VIEW_TYPE_3D || depthExtent > 1)
                       ? VK_IMAGE_TYPE_3D
                       : (viewType == VK_IMAGE_VIEW_TYPE_1D ||
                          viewType == VK_IMAGE_VIEW_TYPE_1D_ARRAY)
                             ? VK_IMAGE_TYPE_1D
                             : VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = { w, h, depthExtent };
    ci.mipLevels = levels;
    ci.arrayLayers = layers;
    ci.samples = samples;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (viewType == VK_IMAGE_VIEW_TYPE_CUBE)
        ci.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    // THE FOUR CALLS, TIMED SEPARATELY (part 77). `CreateImage` is 69% of the texture
    // decode â€” 146 us per texture over 2,344 of them â€” and "image creation" is four
    // different driver operations with four different fixes. `vkAllocateMemory` is a
    // kernel-side buffer-object allocation and is the usual answer; guessing which one it
    // is would be the same mistake that put the untile loop at the top of three kickoffs.
    // Unconditional and cheap: `CreateImage` is called ~2,400 times a run, never per draw.
    const uint64_t ciT0 = CycNow();
    VK_CHECK(vkCreateImage(R->device, &ci, nullptr, &img.image), "vkCreateImage");
    const uint64_t ciT1 = CycNow();

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(R->device, img.image, &req);
    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex =
        FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX)
        return false;
    const uint64_t ciT2 = CycNow();
    // THE POOLED PATH. Only callers that promise the image outlives the run ask for it â€”
    // see the ImgBlock comment for why that promise is checkable rather than hopeful.
    static const bool noPool = EnvOn("CZ_VK_NO_TEX_MEMPOOL");
    VkDeviceMemory bindMem = VK_NULL_HANDLE;
    VkDeviceSize bindOff = 0;
    if (poolMemory && !noPool)
    {
        // The alignment is the IMAGE's, not the block's: two images in one allocation are
        // legal exactly when each is bound at an offset satisfying its own
        // `VkMemoryRequirements::alignment` and their ranges do not overlap.
        const VkDeviceSize align = req.alignment ? req.alignment : 1;
        for (ImgBlock& blk : g_imgBlocks)
        {
            if (blk.typeIndex != ai.memoryTypeIndex)
                continue;
            const VkDeviceSize at = (blk.used + align - 1) / align * align;
            if (at + req.size > blk.size)
                continue;
            g_imgPoolWasteBytes += at - blk.used;
            blk.used = at + req.size;
            bindMem = blk.mem;
            bindOff = at;
            break;
        }
        if (!bindMem)
        {
            ImgBlock blk;
            blk.typeIndex = ai.memoryTypeIndex;
            // An image bigger than a block gets a block of its own rather than a failure.
            blk.size = std::max<VkDeviceSize>(kImgBlockBytes, req.size);
            VkMemoryAllocateInfo bi{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
            bi.allocationSize = blk.size;
            bi.memoryTypeIndex = blk.typeIndex;
            // NOT VK_CHECK: a block allocation that fails must fall back to the dedicated
            // path, not abort. Out of VRAM is out of VRAM either way, but a 32 MB request
            // can fail where a 128 KB one succeeds, and the fallback keeps the picture.
            if (vkAllocateMemory(R->device, &bi, nullptr, &blk.mem) == VK_SUCCESS)
            {
                ++g_imgBlockAllocs;
                blk.used = req.size;
                bindMem = blk.mem;
                bindOff = 0;
                g_imgBlocks.push_back(blk);
            }
            else
            {
                Count("image: memory BLOCK allocation failed â€” this image took a "
                      "dedicated allocation");
            }
        }
    }
    if (bindMem)
    {
        // `img.memory` stays NULL: every destroy site in this file is guarded on it, so a
        // pooled image can never free the block its neighbours are bound into.
        ++g_imgPooled;
    }
    else
    {
        VK_CHECK(vkAllocateMemory(R->device, &ai, nullptr, &img.memory), "vkAllocateMemory");
        bindMem = img.memory;
        ++g_imgDedicated;
    }
    const uint64_t ciT3 = CycNow();
    VK_CHECK(vkBindImageMemory(R->device, img.image, bindMem, bindOff),
             "vkBindImageMemory");
    const uint64_t ciT4 = CycNow();

    VkImageViewCreateInfo vi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = img.image;
    vi.viewType = viewType;
    vi.format = format;
    vi.components = components;
    vi.subresourceRange = { aspect, 0, levels, 0, layers };
    VK_CHECK(vkCreateImageView(R->device, &vi, nullptr, &img.view), "vkCreateImageView");
    g_ciN++;
    g_ciCreateNs += ciT1 - ciT0;
    g_ciReqNs += ciT2 - ciT1;
    g_ciAllocNs += ciT3 - ciT2;
    g_ciBindNs += ciT4 - ciT3;
    g_ciViewNs += CycNow() - ciT4;
    g_ciAllocBytes += req.size;
    return true;
}

// True for the depth formats that carry a stencil aspect as well. Used by Barrier: a
// LAYOUT is a property of the whole image, so a barrier on one of these must name both
// aspects even when the caller only cares about depth.
bool FormatHasStencil(VkFormat f)
{
    return f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT ||
           f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_S8_UINT;
}

// The EDRAM depth format. Xenos depth is 24-bit FLOATING POINT (D24FS8), whose precision
// is concentrated near the camera; our historical D24_UNORM spreads precision uniformly,
// so near the camera it is far coarser. That mismatch is the part-92 Chuck-hair flicker:
// 178 layered alpha-blended hair cards a few millimetres apart, all writing depth and
// testing LESS_EQUAL, land on tied/inverted UNORM depths at head distance and the winner
// re-rolls every frame as the mesh animates â€” a flicker hardware's float depth does not
// show because it separates the cards cleanly. CZ_VK_DEPTH_FLOAT=1 selects D32_SFLOAT_S8
// (float depth with stencil), the faithful match, keeping depth-write and occlusion. The
// D24_UNORM default is the same-binary control arm. Everything downstream reads
// R->depth.format, and the depth clear/sample paths are normalised 0..1 floats either way.
// The depth format actually chosen for THIS device, pinned once at init by
// ChooseEdramDepthFormat below. VK_FORMAT_UNDEFINED until then.
static VkFormat g_edramDepthFormat = VK_FORMAT_UNDEFINED;

VkFormat EdramDepthFormat()
{
    if (g_edramDepthFormat != VK_FORMAT_UNDEFINED)
        return g_edramDepthFormat;
    // Pre-init fallback (a caller before ChooseEdramDepthFormat ran): keep the
    // historical behaviour so nothing changes on the path this used to cover.
    static const bool wantFloat = EnvOn("CZ_VK_DEPTH_FLOAT");
    return wantFloat ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_D24_UNORM_S8_UINT;
}

// Decide the EDRAM depth format for this physical device, once, after it is chosen.
//
// D24_UNORM_S8_UINT is the historical default and the format every NVIDIA test used,
// but AMD does not advertise SAMPLED_IMAGE for it (the hardware stores depth as D32
// internally), so on AMD a sampled depth resolve â€” a depth snapshot bound through the
// bindless heap, or the RT depth input â€” would be UNDEFINED rather than merely wrong.
// When the default format is not sampleable we fall back to D32_SFLOAT_S8_UINT, which
// AMD does support sampled and which is ALSO the more faithful match to Xenos float
// depth (see EdramDepthFormat's comment and CZ_VK_DEPTH_FLOAT). NVIDIA keeps
// D24_UNORM_S8_UINT and is unchanged. CZ_VK_DEPTH_FLOAT still forces float everywhere.
// The decision without the side effects: what the format WOULD be on this device and
// why. `--diag` prints it for a device it never creates (part 105).
static VkFormat PickEdramDepthFormat(VkPhysicalDevice phys, bool* d24Sampleable)
{
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(phys, VK_FORMAT_D24_UNORM_S8_UINT, &fp);
    const bool ok = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
    if (d24Sampleable)
        *d24Sampleable = ok;
    if (EnvOn("CZ_VK_DEPTH_FLOAT"))
        return VK_FORMAT_D32_SFLOAT_S8_UINT;
    return ok ? VK_FORMAT_D24_UNORM_S8_UINT : VK_FORMAT_D32_SFLOAT_S8_UINT;
}

void ChooseEdramDepthFormat(VkPhysicalDevice phys)
{
    if (EnvOn("CZ_VK_DEPTH_FLOAT"))
    {
        g_edramDepthFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;
        fprintf(stderr, "[vk] ED×6÷›Ê×¬¢h­µçYH›ÝˆËÈLŽÌŒ[ˆ\È]H[™ML\\[™Hœ˜[YIÜÈš\œÝ[BˆËÈ
ØÚ\ÜÛÜˆ]HÜšYÚ[‹ÙˆLŽÚYJH\ÚÜÈ\È]Y\Ý[ÛˆÙˆ[BˆËÈ]™\žHœ˜[YK‚ˆYˆ
Z[Ý
[JHHZ[Ý
Ý\™•ÊH
ˆÝ\™’
ˆ
BˆÛÛ[YNÂˆÛÛœÝZ[Ì—Ý[HH[HˆLŽÂˆÛÛœÝZ[Ì—ÝH
[H	H[\Ô\”›ÝÊHNÂˆÛÛœÝZ[Ì—ÝHH
[HÈ[\Ô\”›ÝÊHNÂˆYˆ

ÈÛÜUÈˆÝ\™•ÈH
ÈÛÜRˆÝ\™’
BˆÛÛ[YNÂˆËÈ™X\™\Ý˜\ÙH™[ÝËÛÈHÝ\™˜XÙH]\È]Ù[ˆHÝX‹\™YÚ[ÛˆÙˆHšYÙÙ\‚ˆËÈÛ™H›ÛÈ[È]È[[YYX]H\™[˜]\ˆ[ˆHX\›Y\ÝX]Ú‚ˆYˆ
Y›Ý[™ˆˆ™\Ý˜\ÙJBˆÂˆ›Ý[™HYNÂˆ™\Ý˜\ÙHHŽÂˆ™\ÝHÂˆ™\ÝHHNÂˆBˆBˆYˆ
›Ý[™	‰ˆ
™\Ý™\ÝJJBˆÂˆ˜\ÙRÙ^HH™\Ý˜\ÙNÂˆÝH™\ÝÂˆÝHH™\ÝNÂˆÛÝ[
œ™\ÛÛ™Nˆ\Ý[˜][ÛˆY™\ÜÈ›ÛY[È[ˆ^\Ý[™ÈÝ\™˜XÙH\ÈH‚ˆ[HÙ™œÙ]ŠNÂˆBˆB‚ˆËÈHPÒTÒSÓ‹š[Y™^ÈH™YÚ\Ý\œÈ]Ø[YHœ›ÛKˆ]˜Z[\ÈHQSBˆËÈÝ[™Z[‰ÜÈ^[[™]\ÈH\›H]Ú[[H[˜Ø]\ÈHÛÜNˆH\ÜÈX^BˆËÈYÚ][X][H\ÚÈ›Üˆ[Ü™H›ÝÜÈ[ˆÝ\ˆQSH\Ë[™HÛ›HÞ[\ÛH\ÈBˆËÈÛ˜\ÚÝ]\ÈHšYÚÒV‘H[™\H[\H8 %ÚXÚ™XYÈÝÛœÝ™X[H\ÈBˆËÈÚYÝÈX\[Ùˆ™\›ÜËK™Kˆ\È[HØØÛYY˜]\ˆ[ˆ\ÈZ\ÜÚ[™Ë‚ˆYˆ
˜XÙU\Ô\ÜÊBˆœš[ŠÝ\œ‹ˆ–ÝšÜ™\ÛÛ™WHOˆÛ˜\ÚÝ	L	\È	]^	]HÛÜH	]^	]Hœ›ÛHQSJ	]K	]JH‚ˆÈÝ
	]K	]JH]˜Z[	]^	]HÛˆÏIYIY\ÛX\ILˆ‹ˆ˜\ÙRÙ^Kœ›ÛQ\ÈŠ\
Hˆˆˆ‹ËÛÜUËÛÜRÛÜVÛÜVKˆÝÝK]˜Z[Ë]˜Z[[
ÛX\ÛÛÜŠK[
ÛX\‘\
Kˆ™YÜÖÞ[›ÜÎŽšÔ˜‘\ÛX\—JNÂ‚ˆYˆ
È	‰ˆ	‰ˆÛÜUÈ	‰ˆÛÜR
BˆÂˆÛÛœÝZ[Ì—ÝÙ^HH˜\ÙRÙ^H
œ›ÛQ\ÈÔÛ˜\ÚÝ\š]ˆJNÂˆ]]È]H‹OœÛ˜\ÚÝË™š[™
Ù^JNÂˆËÈH\Ý[˜][ÛˆÚÜÙH^[Ú[™ÙY\ÈHY™™\™[Ý\™˜XÙH™]\Ú[™È[‚ˆËÈY™\ÜËÛÈH[XYÙH\È™XZ[˜]\ˆ[ˆ\X[HÝ™\Üš][ˆ8 %BˆËÈ\X[Ý™\Üš]HX]™\ÈH™]š[Ý\ÈÝ\™˜XÙIÜÈ^[È\›Ý[™HYÙHÙ‚ˆËÈH™]ÈÛ™KÚXÚ™XYÈ\ÈHÚÜÝ[™È\Y˜XÝÚ]›ÈØš[Ý\ÈÛÝ\˜ÙK‚ˆYˆ
]OH‹OœÛ˜\ÚÝË™[™

H	‰‚ˆ
]OœÙXÛÛ™™ÝY\ÝÈOHÈ]OœÙXÛÛ™™ÝY\ÝOHˆ]OœÙXÛÛ™˜Z[ÈOH\ÜÕÈ]OœÙXÛÛ™˜Z[OH\ÜÒ
JBˆÂˆËÈ‘UT‘Q›Ý\Ý›ÞYY8 %[™“ÈØZ]ZYKˆ˜]ÜÈ™XÛÜ™YX\›Y\ˆ[‚ˆËÈTÈœ˜[YHX^HØ[\HHÛ[XYÙH
Z\ˆ\ØÜš\ÜœÈÝ^H˜[YˆËÈ™XØ]\ÙHH[XYÙHÝ^\È[]™JK[™HØZ]ZYH\™HÛÝ[›Ý]™BˆËÈ›ÝXÝY[H[ž]Ø^Nˆ]Û›HÛÝ™\œÈÝX›Z]YÛÜšËÚXÚ\ÈBˆËÈÚÛHÚYÝË]Y\ˆœ™Y^™H
ÙYH™]\™Y[XYÙJKˆHÛš[™\ÜÈÛÝÂˆËÈ\™HÝ[›Ý™XÞXÛY
Ü[‹Z][\ÈØŠK‚ˆ™]\™R[XYÙJ]OœÙXÛÛ™š[XYÙJNÂˆ›Üˆ
]]ÉˆÜÚ^™KšY]×Hˆ]OœÙXÛÛ™šY]ÜÊBˆÂˆ
›ÚY
\Ú^™NÂˆ™]\™R[XYÙJšY]Ëš[XYÙJNÂˆBˆYˆ
]OœÙXÛÛ™œ]XÚšY]ÊBˆÂˆËÈH•˜XÙH\ÜÉÜÈ]XÚY[šY]ÈšY\ÈH[XYÙIÜÈY™][YNÂˆËÈÜ˜\]ÛÈH™[˜ÙKX]Ø\™H]Y]YH\Ý›Þ\È]Ú]H[XYÙK‚ˆ[XYÙHžßNÂˆ‹šY]ÈH]OœÙXÛÛ™œ]XÚšY]ÎÂˆ™]\™R[XYÙJŠNÂˆBˆ‹OœÛ˜\ÚÝË™\˜\ÙJ]
NÂˆ^Ù[[\

NÂˆ]H‹OœÛ˜\ÚÝË™[™

NÂˆÛÝ[
œ™\ÛÛ™NˆÛ˜\ÚÝ™\Ú^™YŠNÂˆBˆYˆ
]OH‹OœÛ˜\ÚÝË™[™

H	‰ˆ‹O›™^^\™TÛÝ×ÛX^\ØÜš\ÜœÊBˆÂˆÛ˜\ÚÝÎÂˆËœÛÝH‹O›™^^\™TÛÝ
ÊÎÂˆË™œ›ÛQ\Hœ›ÛQ\ÂˆË™ÝY\ÝÈHÎÂˆË™ÝY\ÝHÂˆË˜Z[ÈH\ÜÕÎÂˆË˜Z[H\ÜÒÂˆËÈH\Û˜\ÚÝÙY\ÈHQSH\Y™™\‰ÜÈÝÛˆ›Ü›X]™XØ]\ÙBˆËÈšÐÛYÛÜR[XYÙH\ÈÛ›HYš[™Y™]ÙY[ˆY[XØ[\›Ü›X]È8 %\™BˆËÈ\È›ÈÛÜHœ›ÛHH\[XYÙH[ÈHÛÛÝ\ˆÛ™Kˆ]\ÈšY]ÙY›ÝYÚˆËÈHT\ÜXÝÚ]]™\žHÛÛ\Û™[™XY[™È]˜[YKÛÈHÚY\‚ˆËÈØ[\[™È]Ù]ÈHXš]\[ˆœˆ
ÚXÚ\ÈÚ]H[›ÜÈ™]ÚˆËÈÙˆH×ÌÎÝ\™˜XÙH™]\›œÊH[™HYš[™Y˜[YH[ˆ™Ø˜H˜]\ˆ[‚ˆËÈ[Ø[‰ÜÈ[™Yš[™Y›Û‹\™YÛÛ\Û™[ÈÙˆH\šY]Ë‚ˆÛÛœÝšÐÛÛ\Û™[X\[™È\ÝÚ^ž›^Âˆ’×ÐÓÓTÓ‘S•ÔÕÒV–“WÔ‹’×ÐÓÓTÓ‘S•ÔÕÒV–“WÔ‹ˆ’×ÐÓÓTÓ‘S•ÔÕÒV–“WÔ‹’×ÐÓÓTÓ‘S•ÔÕÒV–“WÓÓ‘BˆNÂˆYˆ
Ü™X]R[XYÙJËš[XYÙK–ž
ÊK–Š
Kˆœ›ÛQ\È‹O™\™›Ü›X]ˆ’×Ñ“Ô“PUÔŽÎŽNÕS“Ô“Kˆ’×ÒSPQÑWÕTÐQÑWÕS”Ñ‘T—ÑÕÐ’Uˆ’×ÒSPQÑWÕTÐQÑWÕS”Ñ‘T—ÔÔ×Ð’Uˆ’×ÒSPQÑWÕTÐQÑWÔÐSTQÐ’UˆËÈ•ÝYÙHˆ˜XÙ\ÈS•È\Û˜\ÚÝÎÈHš]ˆËÈÛÜÝÈ›Ý[™ÈÚ[ˆ›Ý[™È™[™\œÈ[È]‚ˆ

œ›ÛQ\	‰ˆ‹Oœ[˜X›Y
BˆÈ’×ÒSPQÑWÕTÐQÑWÑTÔÕSÒSÐUPÒQS•Ð’Uˆˆ
Kˆœ›ÛQ\È’×ÒSPQÑWÐTÔPÕÑTÐ’Uˆˆ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’Uˆ’×ÒSPQÑWÕ’QU×ÕTWÌ‘KKˆœ›ÛQ\È\ÝÚ^ž›HˆšÐÛÛ\Û™[X\[™ÞßJJBˆÂˆšÑ\ØÜš\Ü’[XYÙR[™›ÈZ^ßNÂˆZKš[XYÙUšY]ÈHËš[XYÙKšY]ÎÂˆZKš[XYÙS^[Ý]H’×ÒSPQÑWÓVSÕUÔÒQT—Ô‘PQÓÓ“WÓÔSPSÂˆšÕÜš]Q\ØÜš\Ü”Ù]ÜžÈ’×ÔÕ•PÕT‘WÕTWÕÔ’UWÑTÐÔ’TÔ—ÔÑUNÂˆÜ‹™ÝÙ]H‹OœÙ]ÖÌNÂˆÜ‹™Ýš[™[™ÈHÂˆÜ‹™Ý\œ˜^Q[[Y[HËœÛÝÂˆÜ‹™\ØÜš\ÜÛÝ[HNÂˆÜ‹™\ØÜš\Ü•\HH’×ÑTÐÔ’TÔ—ÕTWÔÐSTQÒSPQÑNÂˆÜ‹œ[XYÙR[™›ÈH	šZNÂˆËÈS”ÒUSÓˆU‘Q“Ô‘HS–US‘ÈÐSˆÑQHHTÐÔ’TÔ‹[ˆ]ÈÝÛ‚ˆËÈ[[YYX]HÝX›Z]ˆHÛÜHH™]È[™\È™[ÝÈ[™XYHX]™\È][‚ˆËÈÒQT—Ô‘PQÓÓ“KÛÈ\ÈÛÚÜÈ™Y[™[8 %[™]\È›Ý™XØ]\ÙHBˆËÈÛÜH\È™XÛÜ™Y[ÈH”SQIÜÈÛÛ[X[™Y™™\ˆÚ[HH\ØÜš\Ü‚ˆËÈ™XÛÛY\Èš\ÚX›HÈ]ÚÛHÛÛ[X[™Y™™\ˆH[œÝ[]\ÈÜš][‹‚ˆËÈ]™\žH˜]È™XÛÜ™YPT“QTˆ[ˆHØ[YHœ˜[YH\È›Ý[™ÈHØ[YBˆËÈš[™\ÜÈX\[™H\ØÜš\ÜˆÛZ[Z[™ÈÒQT—Ô‘PQÓÓ“HÛˆ[ˆ[XYÙBˆËÈ]\ÈÝ[S‘Q’S‘Q\È[™Yš[™YÓÓ•S•›Üˆ[ž][™È][™^\ÂˆËÈ]ˆ›Ý[™ÈÙ\È8 %H˜]ÈØ[ˆÛ›HX\›ˆ\ÈÛÝ[X™\ˆœ›ÛHHÛÚÝ\ˆËÈ]ÛÝ[]™HZ\ÜÙY8 %]››Ý[™È[™^\È]ˆ\È[ˆ\™Ý[Y[[™ˆËÈ\È\ÈHÝX\˜[YK‚ˆËÂˆËÈ]\È[ÛÈ[MÙˆšÐÛY˜]ËS›Û™KLMŒH˜[Y][ÛˆY™XÝ\ˆËÈIÜÈ[™[Ù™ˆØZYÈÚ\ÙHš\œÝ
Ü[ˆ][H
KˆH^Y\ˆ˜[YY[BˆËÈÛ˜ÙH[XYÙ\ÈØ\œšYY˜[Y\ÎˆÚ^™\ÛÛ™HÛ˜\ÚÝÈ[ˆH[š[™ÈÚZ[‚ˆËÈ
MžKŒ‹ÌžLKÌžKÌž‹ÌžH8 %H›ÛÛH\˜[ZY
H\ÈZ\‚ˆËÈÙXÛÛ™[™\™ØØÝ\œ™[˜Ù\Ë]™\žHÛ™HÜ™X]YZYYœ˜[YK‚ˆ[’[[YYX]JÉ—JšÐÛÛ[X[™Y™™\ˆØŠHÂˆ˜\œšY\ŠØ‹Ëš[XYÙK’×ÒSPQÑWÓVSÕUÔÒQT—Ô‘PQÓÓ“WÓÔSPSˆœ›ÛQ\È’×ÒSPQÑWÐTÔPÕÑTÐ’Uˆˆ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’U
NÂˆJNÂˆšÕ\]Q\ØÜš\Ü”Ù]Ê‹O™]šXÙKK	Ü‹[ŠNÂˆ˜[YR[XYÙJËš[XYÙKœ™\ÛÛ™HÛ˜\ÚÝ	L	]^	]I\ÈÛÝ	]H‹˜\ÙRÙ^KËˆœ›ÛQ\ÈˆTˆˆˆ‹ËœÛÝ
NÂˆËÈHY\‹Y[™ØYÙ[Y[[™HHÝ™\›šYÚØ]HÜ™\È›ÜŽˆÔÕ^[ËˆËÈHY\ˆØØ[H[™HØÙ[™HØØ[KÛˆHÛ™HÝ\™˜XÙHHY\‚ˆËÈÛÝ™\›œËˆÛ™H[™H\ˆ
™JXÜ™X][Û‹›Ý\ˆœ˜[YK‚ˆYˆ
\ÜÒOH[\›˜[

JBˆœš[ŠÝ\œ‹ˆ–Ýš×HÚYÝË]Y\ˆÛ˜\ÚÝ	Lˆ	]^	]HÜÝ
ÝY\Ý	]^	]K‚ˆœÚYÝÈ\ÜÈ	]^	]KØÙ[™H	]^	]JWˆ‹ˆ˜\ÙRÙ^K–ž
ÊK–Š
KË\ÜÕË\ÜÒ[\›˜[Ê
Kˆ[\›˜[

JNÂˆ]H‹OœÛ˜\ÚÝË™[\XÙJÙ^KÝŽ›[Ý™JÊJK™š\œÝÂˆ^Ù[[\

NÂˆÛÝ[
œ™\ÛÛ™NˆÛ˜\ÚÝÜ™X]YŠNÂˆBˆ[ÙBˆÂˆKT‹O›™^^\™TÛÝÂˆÛÝ[
œ™\ÛÛ™NˆÛ˜\ÚÝ[XYÙHÜ™X][Ûˆ˜Z[YŠNÂˆBˆBˆYˆ
]OH‹OœÛ˜\ÚÝË™[™

JBˆÂˆËÈHÛÝ\˜ÙHQSHY™™\‹[™H\ÜXÝ]ÛÙ\ÈÚ]]ˆH\ˆËÈ™\ÛÛ™HÛÜY\ÈÝ]Ùˆ‹O™\ˆHÚÛHÚ[Ùˆ™XY[™ÂˆËÈÛÜWÜÜ˜×ÜÙ[XÝ\È]\ÙHÛÈ\™HY™™\™[XÝ\™\Ë‚ˆËÂˆËÈÖ—Õ’×ÓTÐPNˆH][\Ø[\Y[XYÙHØ[››Ý™HšÐÛYÛÜR[XYÙIÙ[ÈBˆËÈÚ[™ÛK\Ø[\HÛ˜\ÚÝˆÛÛÝ\ˆ™\ÛÛ™\È[ˆXÙHÙˆHÛÜH™[ÝÎÂˆËÈT\È›ÈšÐÛY™\ÛÛ™R[XYÙKÛÈ]ÛÙ\È›ÝYÚH™\›ËY˜]ÂˆËÈ™[™\š[™È\ÜÈÚÜÙH™\ÛÛ™H]XÚY[Üš]\È‹O™\™\ÛÛ™K[™ˆËÈH^\Ý[™ÈÛÜH[ˆ™XYÈU[XYÙHÚ]HØ[YHÙ™œÙ]È
]\ÂˆËÈHQSIÜÈ^[^XÝJK‚ˆÛÛœÝ›ÛÛ\ØXSÛˆH‹O›\ØXTØ[\\ÈOH’×ÔÐSTWÐÓÕS•ÌWÐ’UÂˆ[XYÙIˆÜ˜ÈHœ›ÛQ\È
\ØXSÛˆÈ‹O™\™\ÛÛ™Hˆ‹O™\
Hˆ‹O˜ÛÛÜŽÂˆÛÛœÝšÒ[XYÙP\ÜXÝ›YÜÈ\ÜXÝBˆœ›ÛQ\È’×ÒSPQÑWÐTÔPÕÑTÐ’Uˆ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’UÂˆ™YÚ[‘œ˜[YJ
NÂˆ[™™[™\š[™Ê
NÂˆËÈHY™\œ™YÛX\ˆÝ[Ý]Ý[™[™È\™HYX[œÈHÛX\‹][‹XÛÜHÚ]›ÂˆËÈ\ÜÈ[ˆ™]ÙY[ŽÈHÛÜH]\ÝÙYHHÛX\™Y^[ËÛÈ[Z]]›ÝË‚ˆ›\Ú[™[™ÐÛX\œÊ˜ÛX\ŽˆY™\œ™Y“TÒQ›ÜˆH™\ÛÛ™HÛÜHŠNÂˆYˆ
\ØXSÛˆ	‰ˆœ›ÛQ\
BˆÂˆËÈH\™\ÛÛ™H\È‘QÒSÓSˆ[˜[ZXÈ™[™\š[™È™\ÛÛ™\È^XÝBˆËÈH™[™\\™XKÛÈÛ›HH™XÝ[™ÛH\ÈÛÜH™YYÈ\ÈZY›Ü‹‚ˆËÈÚ\™ÙYÈH™\ÛÛ™KXÛÜHÛ\ÜÈ8 %]TÈH™\ÛÛ™IÜÈ]šXÙHÛÜšË‚ˆÜTÙYÈÙÙŠÑÜ™\ÛÛ™PÛÜJNÂˆ˜\œšY\Š‹O˜ÛY‹O™\’×ÒSPQÑWÓVSÕUÑTÔÕSÒSÐUPÒQS•ÓÔSPSˆ’×ÒSPQÑWÐTÔPÕÑTÐ’U’×ÒSPQÑWÐTÔPÕÔÕSÒSÐ’U
NÂˆËÈH‘TÓÓ‘HT‘ÑU	ÜÈ˜\œšY\ˆ\ÈÜš][ˆÝ]˜]\ˆ[ˆÛÚ[™È›ÝYÚˆËÈ˜\œšY\Š
K™XØ]\ÙHH™\ÛÛ™KX]XÚY[Ô’UH]šÐÛY[™™[™\š[™È\ÂˆËÈ\™›Ü›YY]ÓÓÔ—ÐUPÒQS•ÓÕUUÚ]ÓÓÔ—ÐUPÒQS•ÕÔ’UHXØÙ\ÜÂˆËÈ8 %]™[ˆ›ÜˆHT™\ÛÛ™H]XÚY[8 %[™^[Ý]X\ÚÜÉÈ\BˆËÈ]XÚY[ØÛÜH
PT“_UHœ˜YÛY[\ÝËÈXØÙ\ÜÊHÙ\È›ÝÛÝ™\‚ˆËÈ]ˆÞ[˜È˜[Y][Ûˆ™\ÜY^XÝH]LÔ’UKPQ•T‹UÔ’UH^˜\™ÂˆËÈYØZ[œÝ\È[XYÙKÛˆ\È\›IÜÈš\œÝØ]H[‹ˆHÝØÛÜH™[ÝÂˆËÈ\ÈH[š[ÛˆÙˆ›Ý[Ù[Ëˆ
\ÈÛ™HÚ]Hž\\ÜÙ\ÈBˆËÈÖ—Õ’×ÐT”’QT—ÔÒTÓÓ‹ÕÒQH\›\È8 %HÝ]YØ]™X]›Ý[ˆÝ™\œÚYÚŠBˆÂˆšÒ[XYÙSY[[ÜžP˜\œšY\ˆ˜žÈ’×ÔÕ•PÕT‘WÕTWÒSPQÑWÓQSSÔ–WÐT”’QTˆNÂˆ˜‹›Û^[Ý]H‹O™\™\ÛÛ™K›^[Ý]Âˆ˜‹›™]Ó^[Ý]H’×ÒSPQÑWÓVSÕUÑTÔÕSÒSÐUPÒQS•ÓÔSPSÂˆ˜‹œÜ˜Ô]Y]YQ˜[Z[R[™^H’×ÔUQUQWÑSRSWÒQÓ“Ô‘QÂˆ˜‹™Ý]Y]YQ˜[Z[R[™^H’×ÔUQUQWÑSRSWÒQÓ“Ô‘QÂˆ˜‹š[XYÙHH‹O™\™\ÛÛ™Kš[XYÙNÂˆ˜‹œÝXœ™\ÛÝ\˜ÙT˜[™ÙHHÈ’×ÒSPQÑWÐTÔPÕÑTÐ’Uˆ’×ÒSPQÑWÐTÔPÕÔÕSÒSÐ’UˆKHNÂˆšÔ\[[™TÝYÙQ›YÜÈÜ˜ÔÝYÙHH’×ÔTSS‘WÔÕQÑWÐSÐÓÓSPS‘×Ð’UÂˆ˜‹œÜ˜ÐXØÙ\ÜÓX\ÚÈHÂˆ^[Ý]X\ÚÜÊ˜‹›Û^[Ý]Ü˜ÔÝYÙK˜‹œÜ˜ÐXØÙ\ÜÓX\ÚÊNÂˆÛÛœÝšÔ\[[™TÝYÙQ›YÜÈÝÝYÙHBˆ’×ÔTSS‘WÔÕQÑWÑPT“WÑ”QÓQS•ÕTÕ×Ð’Uˆ’×ÔTSS‘WÔÕQÑWÓUWÑ”QÓQS•ÕTÕ×Ð’Uˆ’×ÔTSS‘WÔÕQÑWÐÓÓÔ—ÐUPÒQS•ÓÕUUÐ’UÂˆ˜‹™ÝXØÙ\ÜÓX\ÚÈH’×ÐPÐÑTÔ×ÑTÔÕSÒSÐUPÒQS•Ô‘PQÐ’Uˆ’×ÐPÐÑTÔ×ÑTÔÕSÒSÐUPÒQS•ÕÔ’UWÐ’Uˆ’×ÐPÐÑTÔ×ÐÓÓÔ—ÐUPÒQS•ÕÔ’UWÐ’UÂˆšÐÛY\[[™P˜\œšY\Š‹O˜ÛYÜ˜ÔÝYÙKÝÝYÙK[‹ˆ[‹K	˜˜ŠNÂˆ‹O™\™\ÛÛ™K›^[Ý]H˜‹›™]Ó^[Ý]Âˆ
ÊÙ×Ø˜\œšY\“ŽÂˆBˆšÔ™[™\š[™Ð]XÚY[[™›È^È’×ÔÕ•PÕT‘WÕTWÔ‘S‘T’S‘×ÐUPÒQS•ÒS‘“ÈNÂˆKš[XYÙUšY]ÈH‹O™\šY]ÎÂˆKš[XYÙS^[Ý]H’×ÒSPQÑWÓVSÕUÑTÔÕSÒSÐUPÒQS•ÓÔSPSÂˆKœ™\ÛÛ™S[ÙHH‹O™\™\ÛÛ™S[ÙNÂˆKœ™\ÛÛ™R[XYÙUšY]ÈH‹O™\™\ÛÛ™KšY]ÎÂˆKœ™\ÛÛ™R[XYÙS^[Ý]H’×ÒSPQÑWÓVSÕUÑTÔÕSÒSÐUPÒQS•ÓÔSPSÂˆK›ØYÜH’×ÐUPÒQS•ÓÐQÓÔÓÐQÂˆKœÝÜ™SÜH’×ÐUPÒQS•ÔÕÔ‘WÓÔÔÕÔ‘NÂˆšÔ™[™\š[™Ò[™›Èœš^È’×ÔÕ•PÕT‘WÕTWÔ‘S‘T’S‘×ÒS‘“ÈNÂˆœšKœ™[™\\™XHHÈÈ–žJ[Ì—Ý
ÛÜV
JK–žZJ[Ì—Ý
ÛÜVJJHKˆÈ–ž
ÛÜUÊK–ŠÛÜR
HHNÂˆœšK›^Y\ÛÝ[HNÂˆœšKœ\]XÚY[H	™NÂˆœšKœÝ[˜Ú[]XÚY[H	™NÂˆšÐÛY™YÚ[”™[™\š[™Ê‹O˜ÛY	œœšJNÂˆšÐÛY[™™[™\š[™Ê‹O˜ÛY
NÂˆËÈ‹‹˜[™Ý˜ZYÚÈS”Ñ‘T—ÔÔÈ›ÜˆHÛÜH™[ÝËZ\œ›Ü™YˆBˆËÈÓÕTÑHØÛÜH]\Ý[ÛÈ˜[YHH™\ÛÛ™HÜš]IÜÂˆËÈÓÓÔ—ÐUPÒQS•ÓÕUUÐÓÓÔ—ÐUPÒQS•ÕÔ’UH[Ù[ÚXÚˆËÈ^[Ý]X\ÚÜÉÈ\X]XÚY[[žHÙ\È›Ýˆ[Z]Y\™HÛÈBˆËÈÙ[™\šXÈ˜\œšY\Š
H[ˆHÛÜH›ØÚÈX\›K\™]\›œÈÛˆH^[Ý]‚ˆÂˆšÒ[XYÙSY[[ÜžP˜\œšY\ˆ˜žÈ’×ÔÕ•PÕT‘WÕTWÒSPQÑWÓQSSÔ–WÐT”’QTˆNÂˆ˜‹›Û^[Ý]H’×ÒSPQÑWÓVSÕUÑTÔÕSÒSÐUPÒQS•ÓÔSPSÂˆ˜‹›™]Ó^[Ý]H’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÔÔ×ÓÔSPSÂˆ˜‹œÜ˜Ô]Y]YQ˜[Z[R[™^H’×ÔUQUQWÑSRSWÒQÓ“Ô‘QÂˆ˜‹™Ý]Y]YQ˜[Z[R[™^H’×ÔUQUQWÑSRSWÒQÓ“Ô‘QÂˆ˜‹š[XYÙHH‹O™\™\ÛÛ™Kš[XYÙNÂˆ˜‹œÝXœ™\ÛÝ\˜ÙT˜[™ÙHHÈ’×ÒSPQÑWÐTÔPÕÑTÐ’Uˆ’×ÒSPQÑWÐTÔPÕÔÕSÒSÐ’UˆKHNÂˆ˜‹œÜ˜ÐXØÙ\ÜÓX\ÚÈH’×ÐPÐÑTÔ×ÑTÔÕSÒSÐUPÒQS•ÕÔ’UWÐ’Uˆ’×ÐPÐÑTÔ×ÐÓÓÔ—ÐUPÒQS•ÕÔ’UWÐ’UÂˆ˜‹™ÝXØÙ\ÜÓX\ÚÈH’×ÐPÐÑTÔ×ÕS”Ñ‘T—Ô‘PQÐ’UÂˆšÐÛY\[[™P˜\œšY\Š‹O˜ÛYˆ’×ÔTSS‘WÔÕQÑWÑPT“WÑ”QÓQS•ÕTÕ×Ð’Uˆ’×ÔTSS‘WÔÕQÑWÓUWÑ”QÓQS•ÕTÕ×Ð’Uˆ’×ÔTSS‘WÔÕQÑWÐÓÓÔ—ÐUPÒQS•ÓÕUUÐ’Uˆ’×ÔTSS‘WÔÕQÑWÕS”Ñ‘T—Ð’U[‹ˆ[‹K	˜˜ŠNÂˆ‹O™\™\ÛÛ™K›^[Ý]H˜‹›™]Ó^[Ý]Âˆ
ÊÙ×Ø˜\œšY\“ŽÂˆBˆÛÝ[
œ™\ÛÛ™Nˆ\™\ÛÛ™YÝ]ÙˆH][\Ø[\YQSHŠNÂˆBˆËÈHÚÛH™\ÛÛ™IÜÈ]šXÙHÛÜšÈ\ÈÛ™HÙYÛY[˜\œšY\œÈ[˜ÛYYˆHš\œÝˆËÈ™\œÚ[Ûˆ[YYÛ›HHšÐÛYÛÜR[XYÙX[™YHÛÈ^[Ý]˜[œÚ][ÛœÂˆËÈ[ˆH™\ÚYX[Ú\™H^H\™H[™\Ý[™ÝZ\ÚX›Hœ›ÛHÛÜšÈ›Ø›ÙHÜ˜\Y8 %ˆËÈ[™H˜[œÚ][ÛˆÙˆHÍM]XÚY[\È›ÝHœ™YH[œÝXÝ[Û‹‚ˆÂˆÜTÙYÈÙØŠÑÜ™\ÛÛ™P˜\œšY\ŠNÂˆËÈHQSH\[XYÙH\È˜XÚÙYÚ]›Ý\ÜXÝÈ]™\ž]Ú\™H[ÙKÛÂˆËÈ]È˜\œšY\œÈØ\œžH›Ý\™HÛÎÈHÛ˜\ÚÝ\ÈÛ›H\‚ˆ˜\œšY\Š‹O˜ÛYÜ˜Ë’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÔÔ×ÓÔSPSˆœ›ÛQ\È
’×ÒSPQÑWÐTÔPÕÑTÐ’U’×ÒSPQÑWÐTÔPÕÔÕSÒSÐ’U
Bˆˆ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’U
NÂˆ˜\œšY\Š‹O˜ÛY]OœÙXÛÛ™š[XYÙK’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÑÕÓÔSPSˆ\ÜXÝ
NÂˆBˆÂˆÜTÙYÈÙÜ™\ÊÑÜ™\ÛÛ™PÛÜJNÂˆËÈÛÜHHSK]]ÈÝÛˆÜÚ][Ûˆ[ˆXXÚ[XYÙKˆ›ÜˆHØÚ\ÜÛÜ‹[Ù™œÙ]ˆËÈ[HHÛÈÙ™œÙ]È\™HHØ[YK™XØ]\ÙHÝ\ˆQSH\È[\ØÜ™Y[‹\Ú^™YˆËÈ[™HÚ[™ÝÈÙ™œÙ]\È[X™\˜][H›Ý\YYÈHÙ[ÛY]žH
ÙYHBˆËÈØÚ\ÜÛÜˆ›ÝH[ˆÑ˜]ÊKÛÈH[HÚ]È]]ÈYHØÜ™Y[ˆÜÚ][Ûˆ[‚ˆËÈ›Ýˆ›Üˆ[ˆQ‘TÔË[Ù™œÙ]ÝX‹\™YÚ[Ûˆ^HY™™\ŽˆH\ÜÈ™[™\™Y]ˆËÈHQSHÜšYÚ[ˆ[™H^[È™[Û™ÈÛÛY]Ú\™H[ÙH[ˆH\Ý[˜][Û‚ˆËÈÝ\™˜XÙKÚXÚ\ÈÚ]ÝØÝXØ\œžK‚ˆšÒ[XYÙPÛÜHÛÜ^ßNÂˆÛÜKœÜ˜ÔÝXœ™\ÛÝ\˜ÙHHÈ\ÜXÝHNÂˆÛÜKœÜ˜ÓÙ™œÙ]HÈ–žJ[Ì—Ý
ÛÜV
JK–žZJ[Ì—Ý
ÛÜVJJKNÂˆÛÜK™ÝÝXœ™\ÛÝ\˜ÙHHÈ\ÜXÝHNÂˆÛÜK™ÝÙ™œÙ]HÈ–žJ[Ì—Ý
Ý
JK–žZJ[Ì—Ý
ÝJJKNÂˆÛÜK™^[HÈ–ž
ÛÜUÊK–ŠÛÜR
KHNÂˆËÈHVSÛÝ[™\ÚYHHZ[\ÙXÛÛ™ÎˆH˜[™ÚYÛÜÝ\ÈÛ›BˆËÈ\ÚYÛ˜X›HÛ˜ÙH[ÝHÛ›ÝÈÝÈX[žH^[È][Ý™\È8 %ŽH™\ÛÛ™\ÈHœ˜[YH]ˆËÈ[ˆ[šÛ›ÝÛˆ^[\È›ÝH[X™\ˆ[ž[Û™HØ[ˆXÝÛ‹‚ˆ×ÙÜ™\ÛÛ™T^[È
ÏHZ[Ý
–ž
ÛÜUÊJH
ˆZ[Ý
–ŠÛÜR
JNÂˆYˆ
×ØÛÜPÙ[œÝ\ÓÛŠBˆÛÜPÙ[œÝ\ÐÛÜJÙ^K–žJ[Ì—Ý
Ý
JK–žZJ[Ì—Ý
ÝJJKˆ–ž
ÛÜUÊK–ŠÛÜR
KˆZ[Ý
–ž
ÛÜUÊJH
ˆZ[Ý
–ŠÛÜR
JJNÂˆYˆ
\ØXSÛˆ	‰ˆYœ›ÛQ\
BˆÂˆËÈHTÐPH™\ÛÛ™KÚ\™HHÚ[™ÛK\Ø[\H™[™\™\ˆÛÜY\ËˆØ[YBˆËÈ™YÚ[Û‹Ø[YH^[Ý]ÎÈšÒ[XYÙT™\ÛÛ™H\ÈšY[Y›Ü‹YšY[HÛÜBˆËÈÝXÝˆ\È\ÈH[‰ÜÈNŒHX\[™ÈXYH]\˜[ˆHÝY\Ý	ÜÂˆËÈ—ÐÓÔH
š\ÊˆH™\ÛÛ™H
ØÜËÛ\ØXK\[‹›Y0©ÌJK‚ˆšÒ[XYÙT™\ÛÛ™HžßNÂˆ‹œÜ˜ÔÝXœ™\ÛÝ\˜ÙHHÛÜKœÜ˜ÔÝXœ™\ÛÝ\˜ÙNÂˆ‹œÜ˜ÓÙ™œÙ]HÛÜKœÜ˜ÓÙ™œÙ]Âˆ‹™ÝÝXœ™\ÛÝ\˜ÙHHÛÜK™ÝÝXœ™\ÛÝ\˜ÙNÂˆ‹™ÝÙ™œÙ]HÛÜK™ÝÙ™œÙ]Âˆ‹™^[HÛÜK™^[ÂˆšÐÛY™\ÛÛ™R[XYÙJ‹O˜ÛYÜ˜Ëš[XYÙK’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÔÔ×ÓÔSPSˆ]OœÙXÛÛ™š[XYÙKš[XYÙKˆ’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÑÕÓÔSPSK	œŠNÂˆÛÝ[
œ™\ÛÛ™NˆÛÛÝ\ˆ™\ÛÛ™YÝ]ÙˆH][\Ø[\YQSHŠNÂˆBˆ[ÙBˆšÐÛYÛÜR[XYÙJ‹O˜ÛYÜ˜Ëš[XYÙK’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÔÔ×ÓÔSPSˆ]OœÙXÛÛ™š[XYÙKš[XYÙKˆ’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÑÕÓÔSPSK	˜ÛÜJNÂˆËÈ•ÕQÑH‰ÜÈS’‘PÕSÓˆVT’SQS•
\ÈX[™Y›Ý‹\[‹›Y0©ÌËˆËÈ›Ý]H
JJKˆÖ—Õ’×ÔÒQÕ×Ñ’SO\‹ŒOˆÝ™\Üš]\ÈHÚYÝÂˆËÈUTÈÛ˜\ÚÝ	ÜÈ\ÈÚ]HÛÛœÝ[šYÚY\ˆXXÚØ\ØØYBˆËÈ™\ÛÛ™H[™È8 %H[XYÙH\ÈÝ[S”Ñ‘T—ÑÕÛÈHš[ÛÜÝÂˆËÈÛ™HÛX\ˆ[™›È^˜H˜\œšY\œÈ8 %[™H]IÜÈÝÛˆÚYÝÂˆËÈÛÛ\\š\ÛÛˆ[ˆ[œÈYØZ[œÝÕTˆ˜[YH[œÝXYÙˆH˜\Ý\š^™YˆËÈØ\ØØYIÜËˆ\È\ÈHÚX\\ÝÜÜÚX›H\ÝÙˆH[š™XÝ[Û‚ˆËÈÚ[ˆYˆH]\È\ÈÚ\™HHÚYÝÈ\›H™XYËÛ™HÛ\š]BˆËÈ]\ÝÚYÝÈHÚÛHÛÜ›[™HÝ\ˆ]\Ý[œÚYÝÈ][™BˆËÈœ˜[YH][Ý™\È[™\ˆ‘RUTˆ™Y]\ÈH[š™XÝ[ÛˆÚ[]Ù[‚ˆËÈ™Y›Ü™H[ž][™È\ÈZ[Ûˆ]ˆ]ÝX›\È\ÈÝYÙH‰ÜÈÝ[™[™ÂˆËÈÔÒUU‘HÓÓ•“Ó
HÖ—Õ’×ÐÕP‘WÔÒTÓÓˆ]\›ŠNˆH[\ÚYÝÂˆËÈ˜[YH]\Ý\šÙ[ˆHœ˜[YH›Üˆ\ÈÛ™È\ÈH]\È\ÈBˆËÈÛÛ\ÜÚ]H›Ý]KˆHPQÓ“ÔÕPÈT“K™]™\ˆHš^È[œÙ]HÙ™ˆHBˆËÈÚ\Y™[™\™\‹‚ˆÝ]XÈÛÛœÝ›Ø]ÚYÝÑš[H×J
HOˆ›Ø]ÂˆÛÛœÝÚ\ŠˆHH[ŠÖ—Õ’×ÔÒQÕ×Ñ’SŠNÂˆ™]\›ˆHÈ›Ø]
]ÙŠJJHˆLKŒŽÂˆJ
NÂˆYˆ
œ›ÛQ\	‰ˆ\ÔÚYÝÔÝ\™˜XÙJ™YÜÊH	‰ˆÚYÝÑš[HŒŠBˆÂˆšÐÛX\‘\Ý[˜Ú[˜[YHÝžÈÚYÝÑš[NÂˆšÒ[XYÙTÝXœ™\ÛÝ\˜ÙT˜[™ÙH˜[™Ù^È’×ÒSPQÑWÐTÔPÕÑTÐ’UKHNÂˆšÐÛYÛX\‘\Ý[˜Ú[[XYÙJ‹O˜ÛY]OœÙXÛÛ™š[XYÙKš[XYÙKˆ’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÑÕÓÔSPSˆ	˜Ý‹K	œ˜[™ÙJNÂˆÛÝ[
œ™\ÛÛ™NˆÚYÝÈ]\È\’SQ
Ö—Õ’×ÔÒQÕ×Ñ’S
HŠNÂˆBˆËÈ•ÕQÑHˆ
\
Nˆ˜^K]˜XÙH\È\Ý\™\ÛÛ™YØ\ØØYHÛXÙK‚ˆËÈH\ÜÈ™[™\œÈ[ÈHÛXÙIÜÈÝÛˆ™XÝ[™ÛHÚ]H\\ÝˆËÈÙY\[™ÈÚXÚ]™\ˆØØÛY\ˆ\È™X\™\ˆHÝ[‹ÛÈ˜\Ý\ˆÛÛ[ˆËÈ
ÚÚ[›™YXÝÜœË›ÛXYÙK]™\ž][™ÈHTÈ^ÛY\ÊHÝ\š]™\Ë‚ˆËÈ[™\ÈÛ™H\ÝÚ[ˆHY\ˆ\ÈÑË‚ˆËÈ“ÕUH
ŠIÜÈÕSˆUÒ[™]]\Ý\[ˆ‘Q“Ô‘H˜XÙTÛXÙKÚXÚˆËÈÛÛœÝ[Y\È×ÛYÚU˜[YÛˆ›Ý]H
JKˆ\È\ÈHÚÛHš^›ÜˆBˆËÈ››ÈÚYÝÜÈ][ˆHÜ\˜]Ü‰ÜÈÙXÛÛ™Ù\ÜÚ[Ûˆ™\ÜYˆÛ›HBˆËÈX]š^]Ø\ÈØ\\™YÛˆHØ^HÈHÐTÐÐQH‘TÓÓ‘HØ[ˆ™HBˆËÈÝ[‰ÜË[™\Ý]Üš]K]Ú[œÈÝ™\ˆHœ˜[YHØ\ÈXÚÚ[™È\HÜYÝÛ‚ˆËÈÜÈ]\È›ÝÛ™KˆÙYHH×ÜÝ[“HÛÛ[Y[‚ˆYˆ
œ›ÛQ\	‰ˆ\ÔÚYÝÔÝ\™˜XÙJ™YÜÊJBˆÚYÝÎŽ“]ÚÝ[Š˜\ÙRÙ^JNÂˆYˆ
œ›ÛQ\	‰ˆ\ÔÚYÝÔÝ\™˜XÙJ™YÜÊJBˆÚYÝÎŽ•˜XÙTÛXÙJ˜\ÙK]OœÙXÛÛ™–žJ[Ì—Ý
Ý
JKˆ–žZJ[Ì—Ý
ÝJJK–ž
ÛÜUÊK–ŠÛÜR
JNÂˆËÈ“ÕUH
ŠNˆH™\ÛÛ™H[™ÈH\ÜË[™\È]H™[™\œÈ[ˆÛÈ]ÚYBˆËÈ[\ËÛÈH\Y™™\ˆ\ÈX›Ý]È\ØÜšX™HHQ‘‘T‘S•™YÚ[Û‹ˆBˆËÈ˜XÝÜˆÛÛ\]Y›ÜˆH™]š[Ý\È[HÛÝ[™HHXÝ\™HÙˆHÜ›Û™ÂˆËÈ\Ý™\ˆH™]ÈÛ™H8 %™XÛÛ\]H]H™^]\Ë\Ø[\[™È˜]Ë‚ˆ˜XÝÜŽŽ’[˜[Y]J
NÂˆËÈ˜XÚÈÈÒQT—Ô‘PQÓÓ“H[[YYX][NˆH]\ˆ\ÜÈ[ˆ\ÈØ[YHœ˜[YBˆËÈØ[\\È\ÈÝ\™˜XÙK[™H^[Ý]]^XÝÈ\ÈHÛ™HBˆËÈ\ØÜš\ÜˆØ\ÈÜš][ˆÚ]‚ˆBˆÂˆÜTÙYÈÙØŒŠÑÜ™\ÛÛ™P˜\œšY\ŠNÂˆ˜\œšY\Š‹O˜ÛY]OœÙXÛÛ™š[XYÙK’×ÒSPQÑWÓVSÕUÔÒQT—Ô‘PQÓÓ“WÓÔSPSˆ\ÜXÝ
NÂˆBˆËÈHšYÚ\Ú^™YšY]ÜÈÙˆ\ÈÝ\™˜XÙK™Yœ™\ÚY[ˆ\ÈØ[YHÛÛ[X[™ˆËÈY™™\ˆÛÈ^HÛÜÝ›ÈÝX›Z][™Ø[››Ý™HÝ[\ˆ[ˆZ\ˆÛÝ\˜ÙK‚ˆËÈZ\ˆÝÛˆÛ\ÜÎˆ^HšYHH™\ÛÛ™H]^H\™HHÙ\\˜]H›][™BˆËÈÛ\ÜÈ]Y\È[œÚYH[›Ý\ˆØ[››Ý™HXÝYÛ‹‚ˆYˆ
Z]OœÙXÛÛ™šY]ÜË™[\J
JBˆÂˆÜTÙYÈÙÝŠÑÜÛ˜\šY]ÊNÂˆ›Üˆ
]]ÉˆÜÚ^™KšY]×Hˆ]OœÙXÛÛ™šY]ÜÊBˆÂˆ
›ÚY
\Ú^™NÂˆ™Yœ™\ÚÛ˜\ÚÝšY]Ê‹O˜ÛY]OœÙXÛÛ™š[XYÙKšY]Ë\ÜXÝ
NÂˆÛÝ[
œ™\ÛÛ™NˆÛ˜\ÚÝšY]È™Yœ™\ÚYŠNÂˆBˆBˆ]OœÙXÛÛ™™œ˜[YTÙY[ˆH‹O™œ˜[YNÂˆÛÝ[
œ›ÛQ\Èœ™\ÛÛ™NˆÛ˜\ÚÝZÙ[ˆœ›ÛHHTY™™\ˆ‚ˆˆœ™\ÛÛ™NˆÛ˜\ÚÝZÙ[ˆœ›ÛHHÛÛÝ\ˆY™™\ˆŠNÂ‚ˆËÈHPÑHÑˆH‘S‘T‘QÕP‘HPTÛÜYY[È]È^Y\ˆ[ˆ\ÈØ[YHÛÛ[X[™ˆËÈY™™\‹ˆØ[YH™X\ÛÛš[™È\ÈHÚ^™YšY]ÜÈX›Ý™Nˆ]ÛÜÝÈ›ÈÝX›Z][™]ˆËÈØ[››Ý™HÝ[\ˆ[ˆ]ÈÛÝ\˜ÙKˆHÝX™HX\H]H™[™\œÈ\È™Y˜]Ûˆ\ÂˆËÈHÛÜ›Ú[™Ù\ËÛÈHÝX™H\ÜÙ[X›YÛ˜ÙH]]Èš\œÝ™]Ú[™™]™\‚ˆËÈ™Yœ™\ÚYÛÝ[œ™Y^™HÚ]]™\ˆH[š\›Û›Y[ÛÚÙYZÙH]][œÝ[ˆËÈ8 %H^XÝY™XÝHUY[ˆ0©ÍœËÛ™H\ØÜš\ÜˆÙ]Ý™\‹‚ˆYˆ
Yœ›ÛQ\
BˆÂˆ]]ÈÝÛ™\ˆH‹O˜ÝX™Q˜XÙSÝÛ™\‹™š[™
˜\ÙRÙ^JNÂˆYˆ
ÝÛ™\ˆOH‹O˜ÝX™Q˜XÙSÝÛ™\‹™[™

JBˆÂˆ]]ÈÝX™HH‹O˜ÝX™TÛ˜\ÚÝË™š[™
ÝÛ™\‹OœÙXÛÛ™™š\œÝ
NÂˆYˆ
ÝX™HOH‹O˜ÝX™TÛ˜\ÚÝË™[™

JBˆÂˆÜTÙYÈÙØÊÑÜÝX™Q˜XÙJNÂˆËÈHÝX™H\ÜÙ[X›HÛÛœÝ[Y\È\È˜XÙIÜÈÛÜH
HÝX™IÜÈÝÛ‚ˆËÈØ[\[™È\ÈHÙ\\˜]H]Y\Ý[ÛˆHÙ[œÝ\ÈÙ\È›Ý›Û[ŠK‚ˆYˆ
×ØÛÜPÙ[œÝ\ÓÛŠBˆÛÜPÙ[œÝ\ÔØ[\Y
Ù^JNÂˆÛÜQ˜XÙR[ÐÝX™J‹O˜ÛY]OœÙXÛÛ™ÝX™KOœÙXÛÛ™ˆÝÛ™\‹OœÙXÛÛ™œÙXÛÛ™
NÂˆÝX™KOœÙXÛÛ™™œ˜[YTÙY[ˆH‹O™œ˜[YNÂˆÛÝ[
œ™\ÛÛ™Nˆ™Yœ™\ÚYH˜XÙHÙˆH™[™\™YÕP‘HPTŠNÂˆBˆBˆB‚ˆYˆ
Yœ›ÛQ\	‰ˆ‹O™œ›ÛY™™\ˆ	‰ˆÙ^HOH
‹O™œ›ÛY™™\ˆ	ˆQ‘‘‘‘‘‘ŠJBˆÂˆ‹O™œ›ÛÚYHÎÂˆ‹O™œ›ÛZYÚHÂˆ‹Oš]™Qœ›ÛÛ˜\ÚÝHYNÂˆÛÝ[
œ™\ÛÛ™Nˆ\È\ÈHœ˜[YHŠNÂˆBˆBˆB‚ˆYˆ
XÛX\ÛÛÜˆ	‰ˆXÛX\‘\
Bˆ™]\›ŽÂ‚ˆ™YÚ[‘œ˜[YJ
NÂˆ[™™[™\š[™Ê
NÂ‚ˆËÈQ‘T”‘QÐÓÔQÓPT”È8 %H\NLY˜][
\™‹\[‹\\L›Y0©ÌJKˆHÛˆËÈYXÚ[š\ÛHÛ›Ý\™YHÛÜH›ØÚÉÜÈÛX\ˆš]ÈÚ]HÚÛKQQSBˆËÈšÐÛYÛX\žÐÛÛÜ‹\Ý[˜Ú[R[XYÙNˆËŽÛX\œÈHœ˜[YHÜš][™ÈNL\^[›Ü‚ˆËÈHÌËŽHH\ÜÙ\È™[™\™YŒMH\ËÙœ˜[YHÙˆÔH]HÜ›ÝÙ\ÈBˆËÈS”Ñ‘T—ÑÕ^[Ý]›Ý[™]š\XXÚˆÛˆ[›ÜÈHÛÜH›ØÚÉÜÈÛX\ˆÛX\œÈBˆËÈ[\ÈÙˆHÕT”‘S•ÕT‘PÑKÛÈHØÛÜY™XÝ\ÈH[Ü™H˜Z][›Ü›K›ÝˆËÈ\ÜÈ
H\LÌˆØÛÜY\›HYX\Ý\™YØÛÜYOH[È›Ý\ˆXÚ[X[ÈÛˆBˆËÈØ\ØØYHÝ]\ÝXÊKˆHÛX\ˆ\ÈUÒQ\™H[™[Z]Y\ÈBˆËÈšÐÛYÛX\]XÚY[È]HXYÙˆH™^\ÜÉÜÈš\œÝ[œÝ[˜ÙH8 %[‚ˆËÈ[œÝ[˜ÙH][™XYH^\ÝÈ[™\ˆ›Ý\›\ËÚXÚ\ÈÚ]H\LÌˆ\›BˆËÈXÚÙY
]ÈYXØ]YZ[šK\ØÛÜH\ˆÛX\ˆÛÜÝLH\ËÙœ˜[YHÙˆ[\[YKBˆËÈØ\Ú\0©ÌH][H™XÛÜ™ÊKˆ[ž][™È]™XYÈQSHš\œÝ›\Ú\ÈBˆËÈ[™[™ÜÈ
ÙYH›\Ú[™[™ÐÛX\œÊNÈH™\ÛÛ™H^[Ùˆ™\›È]Ú\ÈH•SˆËÈ^[ÛÈH[‹\ØÛÜX›HØ\ÙHYÜ˜Y\ÈÈ^XÝHHÛ^[Ë‚ˆËÈÖ—Õ’×Ó“×ÑQ‘T”‘QÐÓPTLH\ÈHØ[YKXš[˜\žHÛÛ›Û\›H[™™\ÝÜ™\ÈHÛˆËÈ]Èž]H›Üˆž]H
[˜ÛY[™ÈÖ—Õ’×ÔÐÓÔQÐÓPT‰ÜÊK‚ˆÝ]XÈÛÛœÝ›ÛÛY™\œ™YÛX\“Ù™ˆH×HÂˆÛÛœÝ›ÛÛÙ™ˆH[“ÛŠÖ—Õ’×Ó“×ÑQ‘T”‘QÐÓPTˆŠNÂˆœš[ŠÝ\œ‹ˆÙ™ˆÈ–Ýš×HÖ—Õ’×Ó“×ÑQ‘T”‘QÐÓPTLH8 %™\ÛÛ™HÛX\œÈZÙHHÓ‚ˆÚÛKQQSH]
HÛÛ›Û\›JWˆ‚ˆˆ–Ýš×H™\ÛÛ™HÛX\œÈ\™HQ‘T”‘Q[™ÐÓÔQ
\LY˜][È‚ˆÖ—Õ’×Ó“×ÑQ‘T”‘QÐÓPTLH\ÈHÛÛ›Û\›JWˆŠNÂˆ™]\›ˆÙ™ŽÂˆJ
NÂˆËÈÖ—Õ’×ÑQ‘T—Ñ•SÔ‘PÕLH8 %PQÓ“ÔÕPÎˆY™\ˆHÛX\ˆ
Ø[YH]ÚØ[YH[Z\ÜÚ[Û‚ˆËÈÚ[Ø[YHÜ™\š[™ÊH]Ú]H•SZ[XYÙH™XÝK™KˆHÛ^[È›ÝYÚˆËÈH™]ÈYXÚ[š\ÛKˆ\È\ÈHÛËY˜XÝÜˆš\ÙXÝ[Ûˆ›Üˆ[žHXÝ\™HÛÛ\Z[ˆËÈYØZ[œÝHY™\œ™YÛX\œÎˆ[ˆ\Y˜XÝ]˜[š\Ú\È[™\ˆ][™XÝÈBˆËÈÐÓÔS‘ÎÈÛ™H]Ý\š]™\È[™XÝÈHY™\œ˜[ÛÜ™\š[™È]Ù[‹ˆ]PT“‘QUÂˆËÈÑQTÛˆ^HÛ™NˆHÜ\˜]Ü‰ÜÈÚX[\Ý[‹YÛÝÈ›ÝË[Ý]˜[š\ÚY[™\ˆ][™ˆËÈ[™\ˆÖ—Õ’×Ó“×ÑQ‘T”‘QÐÓPTˆÚ[H˜XÚÚ[™ÈHÚ\YY˜][^XÝBˆËÈ
[ˆKÐ‹ÐHžH^YJKÚXÚ\ÈÚ]ÛÛšXÝYH™\ÛÛ™K]Ú[™ÝÈØÛÜ[™È™[ÝË‚ˆÝ]XÈÛÛœÝ›ÛÛY™\‘[™XÝH×HÂˆÛÛœÝ›ÛÛÛˆH[“ÛŠÖ—Õ’×ÑQ‘T—Ñ•SÔ‘PÕŠNÂˆYˆ
ÛŠBˆœš[ŠÝ\œ‹–Ýš×HÖ—Õ’×ÑQ‘T—Ñ•SÔ‘PÕLH8 %Y™\œ™YÛX\œÈ\ÙHH‚ˆ‘•SZ[XYÙH™XÝ
HØÛÜ[™Ë]œË[Ü™\š[™ÈXYÛ›ÜÝXÊWˆŠNÂˆ™]\›ˆÛŽÂˆJ
NÂˆËÈHÐÓÔQ‘PÕTÈHTÕSUSÓˆÕT‘PÑIÔÈ“ÓÕ’S•“ÕH‘TÓÓ‘HÒS‘ÕË‚ˆËÈÛˆ[›ÜÈHÛÜH›ØÚÉÜÈÛX\ˆš]ÈÛX\ˆH[\ÈÙˆHÕT”‘S•ÕT‘PÑH8 %ˆËÈHÚÛHÝ\™˜XÙK]ÚZYÚ›Ý\ÝHÚ[™ÝÈH™\ÛÛ™HÛÜYYˆBˆËÈš\œÝÚ\Y›Ü›HØÛÜYÈH™\ÛÛ™HÚ[™ÝÈ[™HÜ\˜]ÜˆÛÛšXÝY]ˆËÈHØ[YH^HÚ][ˆKÐ‹ÐNˆHšY]ËY\[™[ÚX[Ý[ˆÛÝÈ
H]IÜÂˆËÈÝ[‹]š\ÚXš[]HXXÚ[™\žH™XY[™ÈQSH™YÚ[ÛœÈ[œÚYHHÝ\™˜XÙH›ÛÝš[]ˆËÈÝ]ÚYHH™\ÛÛ™HÚ[™ÝÈ8 %™YÚ[ÛœÈHÚÛKZ[XYÙHÛX\ˆY[Ø^\ÈÚ\Y
K‚ˆËÈHÝ\™˜XÙH›ÛÝš[ÛÛZ[œÈH™\ÛÛ™HÚ[™ÝÈžHÛÛœÝXÝ[Û‚ˆËÈ
ÛÜV
ØÛÜUÈHËÛÜVJØÛÜRH
KÝ\È]HQSHÜšYÚ[ˆÚ\™H]™\žBˆËÈÝ\™˜XÙH\ÈZYÝ][™Ý[™[[Ý™\ÈH[QQSH\œš]ÜžH
HL\›ÝÂˆËÈÚYÝÈ˜[™
H\È]™\ž][™È™^[Û™HÝ\™˜XÙHœ›ÛHHÛ\ÜË‚ˆËÂˆËÈHTÐPHPÕÔˆTÈ“ÕÔSÓSˆHÝ\™˜XÙIÜÈÛÛÜ™[˜]\È\™H[ˆVSÈÚ[BˆËÈÝ\ˆQSHÝ[™Z[ˆ\È]ÐSTH™\ÛÛ][Û‹ÚXÙH\ÈÚYH[™ÚXÙH\È[8 %ˆËÈH˜]È]ØØ[\È]™\žHÚ[™ÝÈÛÛÜ™[˜]HžH^XÝH\È˜XÝÜ‹ÛÈBˆËÈ›ÛÝš[ØØ[\ÈÚ]]
ÛÝÚHLŠK‚ˆÛÛœÝZ[Ì—ÝÛX\“\ØXHH
™YÜÖÞ[›ÜÎŽšÔ˜”Ý\™˜XÙR[™›×HˆMŠH	ˆÎÂˆÛÛœÝZ[Ì—ÝÛX\^\ÔØØ[HHÛX\“\ØXHOHˆÈHˆ]NÂˆÛÛœÝ]]ÈØÛÜY™XÝHÉ—JÛÛœÝ[XYÙIˆ[JHÂˆšÔ™XÝ‘žßNÂˆYˆ
È	‰ˆ	‰ˆYY™\‘[™XÝ
BˆÂˆ‹›Ù™œÙ]HÈNÂˆ‹™^[HÈÝŽ›Z[Š–ž
È
ˆÛX\^\ÔØØ[JK[KÚY
KˆÝŽ›Z[Š–Š
ˆÛX\^\ÔØØ[JK[KšZYÚ
HNÂˆBˆ[ÙBˆ‹™^[HÈ[KÚY[KšZYÚNÂˆ™]\›ˆŽÂˆNÂˆYˆ
ÛX\“\ØXHOHˆ	‰ˆ
ÛX\ÛÛÜˆÛX\‘\
JBˆÓÕS•
œ™\ÛÛ™NˆÛX\ˆ™XÝØØ[Y›ÜˆHTÐPHÝ\™˜XÙHŠNÂ‚ˆYˆ
ÛX\ÛÛÜŠBˆÂˆËÈ—ÐÓÓÔ—ÐÓPTˆÛÈHÛX\ˆ˜[YH[ˆH™[™\ˆ\™Ù]	ÜÈÝÛˆ›Ü›X]ˆ]ˆËÈ\È™XY\È\™NÈH\™Ù][ˆ[›Ý\ˆ›Ü›X]ÛÝ[ÛX\ˆÈHÜ›Û™ÂˆËÈÛÛÝ\‹ÚXÚ\ÈÚHHÛÝ[\ÈÙ\\˜]Hœ›ÛHH™\ÛÛ™HÛÝ[‚ˆÛÛœÝZ[Ì—ÝÈH™YÜÖÞ[›ÜÎŽšÔ˜ÛÛÜÛX\—NÂˆËÈÖ—Õ’×ÐÓPT—ÔÒTÓÓLH8 %ÛX\ˆHÛÛÝ\ˆ\™Ù]ÈPQÑS•H[œÝXYÙˆHÝY\Ý	ÜÂˆËÈ˜[YKˆHÜÚ]]™HÛÛ›Û›Üˆ\ÙH^[ÈÙ\™H™]™\ˆÜš][ˆžH[žH˜]È‹‚ˆËÂˆËÈ\‰ÜÈÜ\˜]Üˆ™\Ü\È\™ÙHÜ›Ý[™]Ú\ÈÙˆHÚ[™ÛH›]ÛÛÝ\‹[™ˆËÈ]ÛÛÝ\ˆ\ÈVPÕH
NNN
HÝ™\ˆÎK‰HÙˆÛ™Hœ˜[YH8 %KNMH^[ÈÙ‚ˆËÈÛ™H˜[YHÚ]HÝ[™\™]šX][ÛˆÙˆËˆHÛÛœÝ[]™XÚ\ÙH\È›ÝBˆËÈ^\™H[™›ÝYÚ[™ÎÈ]\ÈZ]\ˆHÛX\ˆÜˆHÚY\ˆÜš][™ÈHÛÛœÝ[‚ˆËÈHÛÈÛÚÈY[XØ[[ˆHØÜ™Y[œÚÝ[™]™H›Ý[™È[ˆÛÛ[[Ûˆ\ÈYÜËÛÂˆËÈ\ÈÙ\\˜]\È[Nˆ[™\ˆH\›K]™\žH^[Ý[ÚÝÚ[™ÈHÛX\ˆ\ÂˆËÈXYÙ[H[™]™\žH^[ÛÛYH˜]ÈXÝX[HÜ›ÝHÙY\È]ÈÛÛÝ\‹‚ˆËÂˆËÈHÛX\ˆSQH\Èš[YÛ˜ÙH\ˆ\Ý[˜Ý˜[YHÛË™XØ]\ÙHYˆHÝY\Ý	ÜÂˆËÈÝÛˆÛX\ˆ\ÈH]Y\Ý[Ûˆ\È[œÝÙ\™YÚ]Ý][›š[™ÈH\›H][‚ˆÝ]XÈÛÛœÝ›ÛÛÛX\”Ú\ÛÛˆH[“ÛŠÖ—Õ’×ÐÓPT—ÔÒTÓÓˆŠNÂˆÂˆÝ]XÈÝŽ™XÝÜZ[Ì—ÝˆÙY[ÛX\ŽÂˆYˆ
ÝŽ™š[™
ÙY[ÛX\‹˜™YÚ[Š
KÙY[ÛX\‹™[™

KÊHOHÙY[ÛX\‹™[™

H	‰‚ˆÙY[ÛX\‹œÚ^™J
HMŠBˆÂˆÙY[ÛX\‹œ\ÚØ˜XÚÊÊNÂˆœš[ŠÝ\œ‹–Ýš×H—ÐÓÓÔ—ÐÓPTˆH	L
OI]HI]HÏI]HI]JWˆ‹Ëˆ
Èˆ
H	ˆ‘‹
ÈˆMŠH	ˆ‘‹
Èˆ
H	ˆ‘‹È	ˆ‘ŠNÂˆBˆBˆšÐÛX\ÛÛÜ•˜[YH˜[Y^ßNÂˆ˜[YK™›Ø]Ì–ÌHH›Ø]

ÈˆMŠH	ˆ‘ŠHÈMKŒŽÂˆ˜[YK™›Ø]Ì–ÌWHH›Ø]

Èˆ
H	ˆ‘ŠHÈMKŒŽÂˆ˜[YK™›Ø]Ì–Ì—HH›Ø]
È	ˆ‘ŠHÈMKŒŽÂˆ˜[YK™›Ø]Ì–Ì×HH›Ø]

Èˆ
H	ˆ‘ŠHÈMKŒŽÂˆYˆ
ÛX\”Ú\ÛÛŠBˆÂˆ˜[YK™›Ø]Ì–ÌHHKŒŽÂˆ˜[YK™›Ø]Ì–ÌWHHŒŽÂˆ˜[YK™›Ø]Ì–Ì—HHKŒŽÂˆ˜[YK™›Ø]Ì–Ì×HHKŒŽÂˆBˆYˆ
YY™\œ™YÛX\“Ù™ŠBˆÂˆËÈHÙ[œÝ\ÈÛÝ[\œÈÙY\Z\ˆÛYX[š[™È[™\ˆ›Ý\›\Îˆ•S\ÈÚ]ˆËÈHÚÛKZ[XYÙHYXÚ[š\ÛHÛÝ[Üš]KÐÓÔQÚ]H™XÝÛÝ™\œË‚ˆ
ÊÙ×ÙÜÛX\“ŽÂˆ×ÙÜÛX\‘[^[È
ÏHZ[Ý
‹O˜ÛÛÜ‹ÚY
H
ˆZ[Ý
‹O˜ÛÛÜ‹šZYÚ
NÂˆÛÛœÝšÔ™XÝ‘™XÝHØÛÜY™XÝ
‹O˜ÛÛÜŠNÂˆ×ÙÜÛX\”ØÛÜY^[È
ÏHZ[Ý
™XÝ™^[ÚY
H
ˆ™XÝ™^[šZYÚÂˆYˆ
™XÝ™^[ÚY	‰ˆ™XÝ™^[šZYÚ
BˆÂˆ[™[™ÐÛX\ˆßNÂˆš\ÐÛÛÜˆHYNÂˆ˜[YK˜ÛÛÜˆH˜[YNÂˆœ™XÝH™XÝÂˆ‹Oœ[™[™ÐÛX\œËœ\ÚØ˜XÚÊ
NÂˆBˆÓÕS•
œ™\ÛÛ™NˆÛÛÝ\ˆÛX\ˆY™\œ™Y
ØÛÜY
HŠNÂˆBˆ[ÙBˆÂˆËÈH˜[œÚ][ÛˆS•È˜[œÙ™\‹YÝ\È[œÚYHHÙYÛY[Ûˆ\œÜÙNˆHÛX\‚ˆËÈ]›Ü˜Ù\ÈH^[Ý]Ú[™ÙH\È›ÝÙ\\˜X›Hœ›ÛHHÚ[™ÙK[™Ú\™Ú[™ÂˆËÈH˜\œšY\ˆ[Ù]Ú\™HÛÝ[XZÙH\ÈÛ\ÜÈ™XYÝÈ›ÜˆHÜ›Û™È™X\ÛÛ‹‚ˆÜTÙYÈÙÊÑÜ™\ÛÛ™PÛX\ŠNÂˆËÈÚ]\ÈÛX\ˆÔ’UTÈYØZ[œÝÚ]H\ÜÈ]›ÛÝÜÈXÝX[H™[™\™Y‚ˆ
ÊÙ×ÙÜÛX\“ŽÂˆ×ÙÜÛX\‘[^[È
ÏHZ[Ý
‹O˜ÛÛÜ‹ÚY
H
ˆZ[Ý
‹O˜ÛÛÜ‹šZYÚ
NÂˆ×ÙÜÛX\”ØÛÜY^[È
ÏBˆ
ÛÜUÈ	‰ˆÛÜR
HÈZ[Ý
–ž
ÛÜUÊJH
ˆZ[Ý
–ŠÛÜR
JBˆˆZ[Ý
‹O˜ÛÛÜ‹ÚY
H
ˆZ[Ý
‹O˜ÛÛÜ‹šZYÚ
NÂˆYˆ
P˜\œšY\Š‹O˜ÛY‹O˜ÛÛÜ‹’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÑÕÓÔSPSˆ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’U
JBˆ˜[œÙ™\•Üš]P˜\œšY\Š‹O˜ÛY‹O˜ÛÛÜ‹’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’U
NÂˆšÒ[XYÙTÝXœ™\ÛÝ\˜ÙT˜[™ÙH˜[™Ù^È’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’UKHNÂˆšÐÛYÛX\ÛÛÜ’[XYÙJ‹O˜ÛY‹O˜ÛÛÜ‹š[XYÙKˆ’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÑÕÓÔSPS	˜[YKK	œ˜[™ÙJNÂˆÓÕS•
œ™\ÛÛ™NˆÛÛÝ\ˆÛX\™YŠNÂˆBˆBˆYˆ
ÛX\‘\
BˆÂˆšÐÛX\‘\Ý[˜Ú[˜[YH˜[Y^ßNÂˆ˜[YK™\H›Ø]
™YÜÖÞ[›ÜÎŽšÔ˜‘\ÛX\—Hˆ
HÈ›Ø]
‘‘‘‘‘ŠNÂˆ˜[YKœÝ[˜Ú[H™YÜÖÞ[›ÜÎŽšÔ˜‘\ÛX\—H	ˆ‘ŽÂˆËÈÖ—Õ’×ÑTÐÓPT—ÑTLH8 %ÛX\ˆ\ÈKŒÚ]]™\ˆ—ÑTÐÓPTˆØ^\Ë‚ˆËÈHPQÓ“ÔÕPÈT“Nˆ]\ÚÜÈÚ]\ˆHØ\ØØYIÜÈ[\H[ˆ\È[\H™XØ]\ÙBˆËÈHY™™\ˆ]\È\ÝYYØZ[œÝØ\ÈÛX\™YÈ
\È]HX]™\ÂˆËÈ—ÑTÐÓPTˆ]›Üˆ™X\›H]™\žH\ÜË[™Ý\ˆÛX\ˆÛÝ™\œÈBˆËÈÚÛHQSHÝ[™Z[ŠKˆYˆH]\Èš[È[™\ˆ\È\›KHÛX\ˆSQH\ÂˆËÈHÚÛHÝÜžNÈYˆ]Ù\È›ÝH™\›ÜÈÛÛYHœ›ÛHÛÛY]Ú\™H[ÙK‚ˆÝ]XÈÛÛœÝ›ÛÛÛX\‘˜\ˆH[“ÛŠÖ—Õ’×ÑTÐÓPT—ÑTˆŠNÂˆYˆ
ÛX\‘˜\ŠBˆ˜[YK™\HKŒŽÂˆËÈHSQKÛ˜ÙH\ˆ\Ý[˜ÝÛ™K›ÜˆHØ[YH™X\ÛÛˆHÛÛÝ\ˆÛX\ˆš[ÂˆËÈ]ÈÝÛŽˆH\Y™™\ˆ]Ý\È]HÜ›Û™È[™\È›ÝHÜ›Û™ÈXÝ\™KˆËÈ]\ÈH\ÜÈÚÜÙH]™\žHœ˜YÛY[˜Z[ÈH\Ý8 %[™HÞ[\ÛHÙˆ]\ÂˆËÈ[ˆSTHÝ\™˜XÙKÚXÚ™XYÈ\ÈHÙ[ÛY]žHØ\È™]™\ˆÝX›Z]Y‹‚ˆÂˆÝ]XÈÝŽ™XÝÜZ[Ì—ÝˆÙY[‘\ÛX\ŽÂˆÛÛœÝZ[Ì—ÝÈH™YÜÖÞ[›ÜÎŽšÔ˜‘\ÛX\—NÂˆYˆ
ÝŽ™š[™
ÙY[‘\ÛX\‹˜™YÚ[Š
KÙY[‘\ÛX\‹™[™

KÊHOBˆÙY[‘\ÛX\‹™[™

H	‰‚ˆÙY[‘\ÛX\‹œÚ^™J
HMŠBˆÂˆÙY[‘\ÛX\‹œ\ÚØ˜XÚÊÊNÂˆœš[ŠÝ\œ‹–Ýš×H—ÑTÐÓPTˆH	L
\	K™‹Ý[˜Ú[	]JWˆ‹ˆË˜[YK™\˜[YKœÝ[˜Ú[
NÂˆBˆBˆYˆ
YY™\œ™YÛX\“Ù™ŠBˆÂˆ
ÊÙ×ÙÜÛX\“ŽÂˆ×ÙÜÛX\‘[^[È
ÏHZ[Ý
‹O™\ÚY
H
ˆZ[Ý
‹O™\šZYÚ
NÂˆÛÛœÝšÔ™XÝ‘™XÝHØÛÜY™XÝ
‹O™\
NÂˆ×ÙÜÛX\”ØÛÜY^[È
ÏHZ[Ý
™XÝ™^[ÚY
H
ˆ™XÝ™^[šZYÚÂˆYˆ
™XÝ™^[ÚY	‰ˆ™XÝ™^[šZYÚ
BˆÂˆ[™[™ÐÛX\ˆßNÂˆš\ÐÛÛÜˆH˜[ÙNÂˆ˜[YK™\Ý[˜Ú[H˜[YNÂˆœ™XÝH™XÝÂˆ‹Oœ[™[™ÐÛX\œËœ\ÚØ˜XÚÊ
NÂˆBˆÓÕS•
œ™\ÛÛ™Nˆ\ÛX\ˆY™\œ™Y
ØÛÜY
HŠNÂˆ™]\›ŽÂˆBˆYˆ
P˜\œšY\Š‹O˜ÛY‹O™\’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÑÕÓÔSPSˆ’×ÒSPQÑWÐTÔPÕÑTÐ’U’×ÒSPQÑWÐTÔPÕÔÕSÒSÐ’U
JBˆ˜[œÙ™\•Üš]P˜\œšY\Š‹O˜ÛY‹O™\ˆ’×ÒSPQÑWÐTÔPÕÑTÐ’U’×ÒSPQÑWÐTÔPÕÔÕSÒSÐ’U
NÂˆšÒ[XYÙTÝXœ™\ÛÝ\˜ÙT˜[™ÙH˜[™Ù^Âˆ’×ÒSPQÑWÐTÔPÕÑTÐ’U’×ÒSPQÑWÐTÔPÕÔÕSÒSÐ’UKBˆNÂˆËÈÖ—Õ’×ÔÐÓÔQÐÓPTLH8 %ÛX\ˆÛ›HH™YÚ[ÛˆTÈ\ÜÈ™[™\™Y›ÝHÚÛBˆËÈQSHÝ[™Z[‹‚ˆËÂˆËÈÛˆ[›ÜÈHÛÜH›ØÚÉÜÈÛX\ˆš]ÈÛX\ˆH[\ÈÙˆHÕT”‘S•ÕT‘PÑKˆÝ\‚ˆËÈQSH\ÈÛ™HLŽL[XYÙHÚ\™YžH]™\žH\ÜËÛÈÛX\š[™È[Ùˆ]XZÙ\ÂˆËÈHÜÝXÚZ[ˆ\ÜÈÚ\HHLLÚYÝÈØ\ØØYH8 %[™]Ú\\È]ÂˆËÈ—ÑTÐÓPT‹ÚXÚ\È]HX]™\È]›Üˆ[[ÜÝ]™\žH\ÜËˆBˆËÈ\Y™™\ˆ]Ú]HTÔÈ\Ý™Z™XÝÈ]™\žHœ˜YÛY[ÛÈHØ\ØØYHÛÛY\ÂˆËÈÝ][\K[™HÚYÝÈÛÚÝ\]™XYÈ™XYÈ\ÈÐÐÓQQˆ\ÌˆYX\Ý\™YˆËÈ^XÝH]ˆ‹ŽÍIHÙˆ]™\žHØ\ØØYH˜[™\È™\›Ë[™›Ü˜Ú[™ÈHÛÛ\\™HÂˆËÈSÐVTÈ
Ö—Õ’×ÑTÐSÐVTÊHš[È]ÛÈHÙ[ÛY]žHØ\È[Ø^\È™Z[™ÂˆËÈÝX›Z]Y[™[Ø^\È™Z[™È™Z™XÝY‚ˆËÂˆËÈ[ˆT“H[[]\ÈYX\Ý\™Y™XØ]\ÙH]Ý]È›ÝØ^\ÎˆH\ÜÈ]YÚ][X][BˆËÈ^XÝÈHÚÛHÝ\™˜XÙHÛX\™Y›ÝÈÙ]ÈÛ›H]ÈØÚ\ÜÛÜ‰ÜÈÛÜ‚ˆÝ]XÈÛÛœÝ›ÛÛØÛÜYÛX\ˆH[“ÛŠÖ—Õ’×ÔÐÓÔQÐÓPTˆŠNÂˆYˆ
ØÛÜYÛX\ˆ	‰ˆÛÜUÈ	‰ˆÛÜR
BˆÂˆšÐÛX\”™XÝ™XÝßNÂˆ™XÝœ™XÝ›Ù™œÙ]HÈ–žJ[Ì—Ý
ÛÜV
JK–žZJ[Ì—Ý
ÛÜVJJHNÂˆ™XÝœ™XÝ™^[HÈ–ž
ÛÜUÊK–ŠÛÜR
HNÂˆ™XÝ˜˜\ÙP\œ˜^S^Y\ˆHÂˆ™XÝ›^Y\ÛÝ[HNÂˆËÈšÐÛYÛX\]XÚY[È™YYÈH™[™\ˆ\ÜÎÈÝ]ÚYHÛ™HH™YÚ[Ûˆ›Ü›H\ÂˆËÈHÛX\ˆÙˆH[XYÙHÚ]HØÚ\ÜÛÜ‹ÚXÚ[Ø[ˆ\È›È\™XÝØ[›Üˆ8 %ˆËÈÛÈ\ÈÙ\È][œÚYHH™[™\š[™ÈØÛÜHÝ™\ˆ\Ý]™XÝ[™ÛK‚ˆšÔ™[™\š[™Ð]XÚY[[™›È\]Âˆ’×ÔÕ•PÕT‘WÕTWÔ‘S‘T’S‘×ÐUPÒQS•ÒS‘“ÂˆNÂˆ\]š[XYÙUšY]ÈH‹O™\šY]ÎÂˆ\]š[XYÙS^[Ý]H’×ÒSPQÑWÓVSÕUÑTÔÕSÒSÐUPÒQS•ÓÔSPSÂˆ\]›ØYÜH’×ÐUPÒQS•ÓÐQÓÔÐÓPTŽÂˆ\]œÝÜ™SÜH’×ÐUPÒQS•ÔÕÔ‘WÓÔÔÕÔ‘NÂˆ\]˜ÛX\•˜[YK™\Ý[˜Ú[H˜[YNÂˆ˜\œšY\Š‹O˜ÛY‹O™\’×ÒSPQÑWÓVSÕUÑTÔÕSÒSÐUPÒQS•ÓÔSPSˆ’×ÒSPQÑWÐTÔPÕÑTÐ’U’×ÒSPQÑWÐTÔPÕÔÕSÒSÐ’U
NÂˆšÔ™[™\š[™Ò[™›Èš^È’×ÔÕ•PÕT‘WÕTWÔ‘S‘T’S‘×ÒS‘“ÈNÂˆšKœ™[™\\™XHHÈÈ–žJ[Ì—Ý
ÛÜV
JK–žZJ[Ì—Ý
ÛÜVJJHKˆÈ–ž
ÛÜUÊK–ŠÛÜR
HHNÂˆšK›^Y\ÛÝ[HNÂˆšKœ\]XÚY[H	™\]ÂˆšKœÝ[˜Ú[]XÚY[H	™\]ÂˆÜTÙYÈÙÊÑÜ™\ÛÛ™PÛX\ŠNÂˆYˆ
‹O˜ÛÛ\]Xš[]T›Ùš[JBˆÂˆšÔ™[™\”\ÜÐ™YÚ[’[™›Èš^È’×ÔÕ•PÕT‘WÕTWÔ‘S‘T—ÔTÔ×Ð‘QÒS—ÒS‘“ÈNÂˆšKœ™[™\”\ÜÈH‹O˜ÛÛ\]Y˜[T™[™\”\ÜÎÂˆšK™œ˜[YXY™™\ˆH‹O˜ÛÛ\]Y˜[Qœ˜[YXY™™\ŽÂˆšKœ™[™\\™XHHšKœ™[™\\™XNÂˆšÐÛY™YÚ[”™[™\”\ÜÊ‹O˜ÛY	˜šK’×ÔÕP”TÔ×ÐÓÓ•S•×ÒS“S‘JNÂˆšÐÛX\]XÚY[ÛX\žßNÂˆÛX\‹˜\ÜXÝX\ÚÈH’×ÒSPQÑWÐTÔPÕÑTÐ’Uˆ’×ÒSPQÑWÐTÔPÕÔÕSÒSÐ’UÂˆÛX\‹˜ÛX\•˜[YK™\Ý[˜Ú[H˜[YNÂˆÛÛœÝšÐÛX\”™XÝ™XÝÈšKœ™[™\\™XKHNÂˆšÐÛYÛX\]XÚY[Ê‹O˜ÛYK	˜ÛX\‹K	œ™XÝ
NÂˆšÐÛY[™™[™\”\ÜÊ‹O˜ÛY
NÂˆBˆ[ÙBˆÂˆšÐÛY™YÚ[”™[™\š[™Ê‹O˜ÛY	œšJNÂˆšÐÛY[™™[™\š[™Ê‹O˜ÛY
NÂˆBˆÛÝ[
œ™\ÛÛ™Nˆ\ÛX\™Y
ØÛÜYÈH\ÜÊHŠNÂˆBˆ[ÙBˆÂˆÜTÙYÈÙÊÑÜ™\ÛÛ™PÛX\ŠNÂˆ
ÊÙ×ÙÜÛX\“ŽÂˆ×ÙÜÛX\‘[^[È
ÏHZ[Ý
‹O™\ÚY
H
ˆZ[Ý
‹O™\šZYÚ
NÂˆ×ÙÜÛX\”ØÛÜY^[È
ÏBˆ
ÛÜUÈ	‰ˆÛÜR
HÈZ[Ý
–ž
ÛÜUÊJH
ˆZ[Ý
–ŠÛÜR
JBˆˆZ[Ý
‹O™\ÚY
H
ˆZ[Ý
‹O™\šZYÚ
NÂˆšÐÛYÛX\‘\Ý[˜Ú[[XYÙJ‹O˜ÛY‹O™\š[XYÙKˆ’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÑÕÓÔSPS	˜[YKKˆ	œ˜[™ÙJNÂˆÓÕS•
œ™\ÛÛ™Nˆ\ÛX\™YŠNÂˆBˆBŸB‚ŸHËÈ˜[Y\ÜXÙB‚‹ËÈOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOB‹ËÈHX›XÈÙX[B‹ËÈOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOB˜›ÛÛšÔ™[™\™\—Ô]˜Z[X›J
BžÂˆËÈ‘QH‘TURT‘SQS•È\ÈÙˆ\ÌKˆH™X]\™H\ÈÈ™HÑ‘‘T‘Q
]\È\šÙY8 %ˆËÈÙYHÚYÝÎŽ“Y[SÙ™™\œÔ
KH]šXÙH\ÈÈÝ\Ü˜^H]Y\žKS‘HÚY\‚ˆËÈ˜\šX[ØXÚH\ÈÈ™H™\Ù[Ú]™X[˜\šX[È[ˆ]8 %›Ý]H
ŠHXZÙ\ÈBˆËÈÚYÝÜÈ[ˆHX]\šX[ÚY\œËÛÈH[[YHÚ]˜^H]Y\žH[™›ÂˆËÈ\ÜÙ]ËÜÚY\—ÜÜ—ÜÛÝ[Ù™™\ˆ™YH•[™ÜÈ]Ú[™ÙH›Ý[™Ëˆ›Ý]H
JBˆËÈ™YYÈ›È˜\šX[È
]Üš]\ÈH]\ÊKÛÈH]™[Ü\ˆ\›HÝ[ÛÜšÜË‚ˆ™]\›ˆÚYÝÎŽ“Y[SÙ™™\œÔ

H	‰ˆˆ	‰ˆ‹Oœ[˜X›Y	‰‚ˆ
‹Oœ˜\šX[È\ÚYÝÎŽ”›Ý]PŠ
JNÂŸB‚‹ËÈÚHHY\ˆÝÜË›ÜˆH[™[›ÛÝ\ŽˆH]Ù\È›ÝHH›È˜^H]Y\žK‹ËÈˆH˜^H]Y\žH]›ÈÚY\ˆ˜\šX[ËÈHT’ÑQ
\ÌJKˆHÜ™\ˆX]\œÎˆ\šÂ‹ËÈš\œÝ™XØ]\ÙHÛˆHXXÚ[™H]Ø[ˆ[ˆ•\™™XÝHÙ[HÛ™\Ý[œÝÙ\ˆ\ÈÙB‹ËÈ\›™Y]Ù™ˆ‹›Ýž[Ý\ˆ]šXÙHØ[››Ý‹‚š[šÔ™[™\™\—Ô[˜]˜Z[X›T™X\ÛÛŠ
BžÂˆYˆ
\ÚYÝÎŽ“Y[SÙ™™\œÔ

JBˆ™]\›ˆÎÂˆYˆ
TˆT‹Oœ[˜X›Y
Bˆ™]\›ˆNÂˆYˆ
T‹Oœ˜\šX[È	‰ˆÚYÝÎŽ”›Ý]PŠ
JBˆ™]\›ˆŽÂˆ™]\›ˆÂŸB‚˜›ÛÛšÔ™[™\™\—ÐXÝ]™J
HÈ™]\›ˆ×ØXÝ]™NÈB‚‹ËÈ	ÜÈ[œ]Y]YHÙX[KØ[YžHM˜Ü	ÜÈš[™ÚY\ˆÛ˜ÙH\ˆ\Ý[˜Ý\Ú\ˆ[‚‹ËÈ
[œÚYH]È[››Ý[˜ÙK[Û˜ÙH›ØÚËÛÈ\È\È™]™\ˆÛˆH\‹Xš[™]
Kˆ[\™XY‚›ÚYšÔ™[™\™\—ÓÛ”ÚY\š[™
Z[Ì—Ý\KZ[Ý\ÚÛÛœÝZ[Ý
ˆÛÙKˆZ[Ì—ÝÚ^™QÛÜ™ÊBžÂˆYˆ
Y×ØXÝ]™JBˆ™]\›ŽÂˆÚY\šš]Ž“Û‘š\œÝš[™
\K\ÚÛÙKÚ^™QÛÜ™ÊNÂŸB‚›˜[Y\ÜXÙHÂ‚‹ËÈÖ—Õ’×ÓTÐPIÜÈÚ[™ÛK\Ø[\HÛÛ\[š[ÛœËÜ™X]Y™\ÚYHH
›ÝÈ][\Ø[\Y
HQSB‹ËÈZ\ˆ]œš[™Ë]\[™]]™\žH]™H™\ØØ[KˆH›Ë[Ü™]\›š[™ÈYH]^ÛÈB‹ËÈÛÛ›Û\›H[ØØ]\È›Ý[™ËˆÛÛÜ”™\ÛÛ™X\ÈS”Ñ‘T—ÑÕ
šÐÛY™\ÛÛ™R[XYÙB‹ËÈÜš]\È]ÛˆH™\Ù[˜[˜XÚÊH
ÈS”Ñ‘T—ÔÔÈ
H™XY˜XÚËØ›]™XYÈ]
NÂ‹ËÈ\™\ÛÛ™X\ÈHTÔÕSÒSÐUPÒQS•™XØ]\ÙHHÛ›HYš[™YØ^HÈ™\ÛÛ™B‹ËÈH\[XYÙH\ÈH™\ÛÛ™H]XÚY[ÛˆH™[™\š[™È\ÜÈ
šÐÛY™\ÛÛ™R[XYÙH\Â‹ËÈÛÛÝ\‹[Û›JK
ÈS”Ñ‘T—ÔÔÈ›ÜˆHÛ˜\ÚÝÛÜH]›ÛÝÜË‚˜›ÛÛÜ™X]QY˜[T™\ÛÛ™U\™Ù]ÊZ[Ì—ÝÜÝËZ[Ì—ÝÜÝ
BžÂˆYˆ
‹O›\ØXTØ[\\ÈOH’×ÔÐSTWÐÓÕS•ÌWÐ’U
Bˆ™]\›ˆYNÂˆËÈÐSTQÛˆHÛÛÝ\ˆÛÛ\[š[Ûˆ\È›Ý™XYžH[ž][™È8 %]\È\™H™XØ]\ÙBˆËÈÜ™X]R[XYÙH[˜ÛÛ™][Û˜[HZ[ÈHšÒ[XYÙUšY]È[™HšY]È™\]Z\™\È]X\ÝˆËÈÛ™HšY]ËXØ\X›H\ØYÙHš]
•RQUšÒ[XYÙUšY]ÐÜ™X]R[™›ËZ[XYÙKLKØ]YÚžBˆËÈH™\žHš\œÝ˜[Y][Ûˆ[ˆÙˆ\È\›JK‚ˆYˆ
PÜ™X]R[XYÙJ‹O˜ÛÛÜ”™\ÛÛ™KÜÝËÜÝ’×Ñ“Ô“PUÔŽÎŽNÕS“Ô“Kˆ’×ÒSPQÑWÕTÐQÑWÕS”Ñ‘T—ÑÕÐ’U’×ÒSPQÑWÕTÐQÑWÕS”Ñ‘T—ÔÔ×Ð’Uˆ’×ÒSPQÑWÕTÐQÑWÔÐSTQÐ’Uˆ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’U
HˆPÜ™X]R[XYÙJ‹O™\™\ÛÛ™KÜÝËÜÝY˜[Q\›Ü›X]

Kˆ’×ÒSPQÑWÕTÐQÑWÑTÔÕSÒSÐUPÒQS•Ð’Uˆ’×ÒSPQÑWÕTÐQÑWÕS”Ñ‘T—ÔÔ×Ð’Uˆ’×ÒSPQÑWÐTÔPÕÑTÐ’U’×ÒSPQÑWÐTÔPÕÔÕSÒSÐ’U
JBˆ™]\›ˆ˜[ÙNÂˆ˜[YR[XYÙJ‹O˜ÛÛÜ”™\ÛÛ™K‘QSHÛÛÝ\ˆ‘TÓÓ‘H	]^	]H‹ÜÝËÜÝ
NÂˆ˜[YR[XYÙJ‹O™\™\ÛÛ™K‘QSH\‘TÓÓ‘H	]^	]H‹ÜÝËÜÝ
NÂˆ™]\›ˆYNÂŸB‚›ÚY\Ý›ÞPÛÛ\]Xš[]QY˜[T\ÜÊ
BžÂˆYˆ
TˆT‹O™]šXÙJBˆ™]\›ŽÂˆYˆ
‹O˜ÛÛ\]Y˜[Qœ˜[YXY™™\ŠBˆšÑ\Ý›ÞQœ˜[YXY™™\Š‹O™]šXÙK‹O˜ÛÛ\]Y˜[Qœ˜[YXY™™\‹[ŠNÂˆYˆ
‹O˜ÛÛ\]Y˜[T™[™\”\ÜÊBˆšÑ\Ý›ÞT™[™\”\ÜÊ‹O™]šXÙK‹O˜ÛÛ\]Y˜[T™[™\”\ÜË[ŠNÂˆ‹O˜ÛÛ\]Y˜[Qœ˜[YXY™™\ˆH’×Ó•SÒS‘NÂˆ‹O˜ÛÛ\]Y˜[T™[™\”\ÜÈH’×Ó•SÒS‘NÂŸB‚˜›ÛÛÜ™X]PÛÛ\]Xš[]QY˜[T\ÜÊ
BžÂˆYˆ
T‹O˜ÛÛ\]Xš[]T›Ùš[JBˆ™]\›ˆYNÂ‚ˆ\Ý›ÞPÛÛ\]Xš[]QY˜[T\ÜÊ
NÂˆšÐ]XÚY[\ØÜš\[Ûˆ]XÚY[ÖÌ—^ßNÂˆ]XÚY[ÖÌK™›Ü›X]H‹O˜ÛÛÜ‹™›Ü›X]Âˆ]XÚY[ÖÌKœØ[\\ÈH‹O›\ØXTØ[\\ÎÂˆ]XÚY[ÖÌK›ØYÜH’×ÐUPÒQS•ÓÐQÓÔÓÐQÂˆ]XÚY[ÖÌKœÝÜ™SÜH’×ÐUPÒQS•ÔÕÔ‘WÓÔÔÕÔ‘NÂˆ]XÚY[ÖÌKœÝ[˜Ú[ØYÜH’×ÐUPÒQS•ÓÐQÓÔÑÓ•ÐÐT‘NÂˆ]XÚY[ÖÌKœÝ[˜Ú[ÝÜ™SÜH’×ÐUPÒQS•ÔÕÔ‘WÓÔÑÓ•ÐÐT‘NÂˆ]XÚY[ÖÌKš[š]X[^[Ý]H’×ÒSPQÑWÓVSÕUÐÓÓÔ—ÐUPÒQS•ÓÔSPSÂˆ]XÚY[ÖÌK™š[˜[^[Ý]H’×ÒSPQÑWÓVSÕUÐÓÓÔ—ÐUPÒQS•ÓÔSPSÂˆ]XÚY[ÖÌWK™›Ü›X]H‹O™\™›Ü›X]Âˆ]XÚY[ÖÌWKœØ[\\ÈH‹O›\ØXTØ[\\ÎÂˆ]XÚY[ÖÌWK›ØYÜH’×ÐUPÒQS•ÓÐQÓÔÓÐQÂˆ]XÚY[ÖÌWKœÝÜ™SÜH’×ÐUPÒQS•ÔÕÔ‘WÓÔÔÕÔ‘NÂˆ]XÚY[ÖÌWKœÝ[˜Ú[ØYÜH’×ÐUPÒQS•ÓÐQÓÔÓÐQÂˆ]XÚY[ÖÌWKœÝ[˜Ú[ÝÜ™SÜH’×ÐUPÒQS•ÔÕÔ‘WÓÔÔÕÔ‘NÂˆ]XÚY[ÖÌWKš[š]X[^[Ý]H’×ÒSPQÑWÓVSÕUÑTÔÕSÒSÐUPÒQS•ÓÔSPSÂˆ]XÚY[ÖÌWK™š[˜[^[Ý]H’×ÒSPQÑWÓVSÕUÑTÔÕSÒSÐUPÒQS•ÓÔSPSÂ‚ˆÛÛœÝšÐ]XÚY[™Y™\™[˜ÙHÛÛÜ”™YžÂˆ’×ÒSPQÑWÓVSÕUÐÓÓÔ—ÐUPÒQS•ÓÔSPSˆNÂˆÛÛœÝšÐ]XÚY[™Y™\™[˜ÙH\™YžÂˆK’×ÒSPQÑWÓVSÕUÑTÔÕSÒSÐUPÒQS•ÓÔSPSˆNÂˆšÔÝXœ\ÜÑ\ØÜš\[ÛˆÝXœ\ÜÞßNÂˆÝXœ\ÜËœ\[[™Pš[™Ú[H’×ÔTSS‘WÐ’S‘ÔÒS•ÑÔTPÔÎÂˆÝXœ\ÜË˜ÛÛÜ]XÚY[ÛÝ[HNÂˆÝXœ\ÜËœÛÛÜ]XÚY[ÈH	˜ÛÛÜ”™YŽÂˆÝXœ\ÜËœ\Ý[˜Ú[]XÚY[H	™\™YŽÂˆšÔ™[™\”\ÜÐÜ™X]R[™›Èš^È’×ÔÕ•PÕT‘WÕTWÔ‘S‘T—ÔTÔ×ÐÔ‘PUWÒS‘“ÈNÂˆšK˜]XÚY[ÛÝ[HZ[Ì—Ý
ÝŽœÚ^™J]XÚY[ÊJNÂˆšKœ]XÚY[ÈH]XÚY[ÎÂˆšKœÝXœ\ÜÐÛÝ[HNÂˆšKœÝXœ\ÜÙ\ÈH	œÝXœ\ÜÎÂˆ’×ÐÒPÒÊšÐÜ™X]T™[™\”\ÜÊ‹O™]šXÙK	œšK[‹	”‹O˜ÛÛ\]Y˜[T™[™\”\ÜÊKˆšÐÜ™X]T™[™\”\ÜÈ
[Ø[ˆKŒHQSJHŠNÂ‚ˆÛÛœÝšÒ[XYÙUšY]ÈšY]ÜÖ×HHÈ‹O˜ÛÛÜ‹šY]Ë‹O™\šY]ÈNÂˆšÑœ˜[YXY™™\Ü™X]R[™›Èš^È’×ÔÕ•PÕT‘WÕTWÑ”SQP•Q‘‘T—ÐÔ‘PUWÒS‘“ÈNÂˆšKœ™[™\”\ÜÈH‹O˜ÛÛ\]Y˜[T™[™\”\ÜÎÂˆšK˜]XÚY[ÛÝ[HZ[Ì—Ý
ÝŽœÚ^™JšY]ÜÊJNÂˆšKœ]XÚY[ÈHšY]ÜÎÂˆšKÚYH‹O˜ÛÛÜ‹ÚYÂˆšKšZYÚH‹O˜ÛÛÜ‹šZYÚÂˆšK›^Y\œÈHNÂˆ’×ÐÒPÒÊšÐÜ™X]Qœ˜[YXY™™\Š‹O™]šXÙK	™šK[‹	”‹O˜ÛÛ\]Y˜[Qœ˜[YXY™™\ŠKˆšÐÜ™X]Qœ˜[YXY™™\ˆ
[Ø[ˆKŒHQSJHŠNÂˆ™]\›ˆYNÂŸB‚‹ËÈ]šXÙHœš[™Ë]\Ú\™YžH›Ý™YYËˆÙ]È×ØXÝ]™HÛˆÝXØÙ\ÜÎÈÚXÚ™YYÝÛœÂ‹ËÈH™[™\™\ˆ\ÈHÐST‰ÜÈXÛ\˜][Ûˆ
×ÙÙ[ÙJK›ÝXÚYY\™K‚˜›ÛÛ[š]ÛÛ[[ÛŠ
BžÂˆˆH™]È™[™\™\Š
NÂˆYˆ
PÜ™X]Q]šXÙJ
HPÜ™X]Q\ØÜš\Ü”[Xš[™Ê
JBˆÂˆœš[ŠÝ\œ‹–Ýš×H]šXÙHœš[™Ë]\RSQ8 %[›š[™ÈÚ]Ý]H™[™\™\—ˆŠNÂˆ™]\›ˆ˜[ÙNÂˆB‚ˆËÈHQSHÝ[™Z[‹ˆÚ^™YÈHÝY\Ý	ÜÈÝÛˆÝ]Yœ›ÛXY™™\ˆ[Y[œÚ[ÛœËˆËÈÚXÚ™ÝØ\Ø\œšY\È[ˆ]™\žHÝØ\XÚÙ]ÈLŽÌŒ[[Hš\œÝÛ™H\œš]™\Ë‚ˆËÈHQSHÝ[™Z[ˆ\ÈSTˆ[ˆH™\Ù[Yœ˜[YK[™]\ÈHÚ[‚ˆËÈ\™Ù]ÚYÝ\™Ù]ZYÚ\ÈÚ]™ÝØ\Ø^\ÈH”“Ó••Q‘‘Tˆ\ÎÈHQSH\ÂˆËÈÈÛH\™Ù\ÝÝ\™˜XÙH[žHTÔÈ™[™\œÈ[Ë[™\È]IÜÈÚYÝÈ\ÜÂˆËÈ™[™\œÈHLLØ\ØØYKˆ]ÌŒ›ÝÜÈ]™\žHØ\ØØYHÜÝ]È›ÝÛHÌ›ÝÜÂˆËÈ™Y›Ü™H[ž][™ÈÛÝ[Ø[\H[KˆÖ—Õ’×ÔÓPSÑQSOLX™\ÝÜ™\ÈHÛÚ^™H[™ˆËÈ\ÈHØ[YKXš[˜\žHÛÛ›Û\›K‚ˆÝ]XÈÛÛœÝ›ÛÛÛX[Y˜[HH[“ÛŠÖ—Õ’×ÔÓPSÑQSHŠNÂˆÛÛœÝZ[Ì—ÝY˜[RBˆÛX[Y˜[HÈ‹O\™Ù]ZYÚˆÝŽ›X^
‹O\™Ù]ZYÚÑY˜[RZYÚ
NÂˆËÈÙ\ÛˆH™[™\™\ˆ™XØ]\ÙHHÒS‘ÕËPÓÓÔ‘SUH˜]È]™YYÈ]ˆHÚ[™ÝÂˆËÈÛÛÜ™[˜]H\È™[]]™HÈHQSHÝ\™˜XÙK›ÝÈH™\Ù[Yœ˜[YKÛÈBˆËÈÛ\›Û[YH]X\È[È\ÈÈ™HHQSIÜÈ
ÙYHHOOLœ˜[˜Ú
K‚ˆ‹O™Y˜[UÚYH‹O\™Ù]ÚYÂˆ‹O™Y˜[RZYÚHY˜[RÂˆËÈHÔÕVS•TÈÐÐSQÈY˜[UÚYØY˜[RZYÚ™[ÝÈÝ^H[ˆÝY\Ý^[Ë‚ˆYˆ
PÜ™X]R[XYÙJ‹O˜ÛÛÜ‹”Ö
‹O\™Ù]ÚY
K”ÊY˜[R
Kˆ’×Ñ“Ô“PUÔŽÎŽNÕS“Ô“Kˆ’×ÒSPQÑWÕTÐQÑWÐÓÓÔ—ÐUPÒQS•Ð’U’×ÒSPQÑWÕTÐQÑWÕS”Ñ‘T—ÔÔ×Ð’Uˆ’×ÒSPQÑWÕTÐQÑWÕS”Ñ‘T—ÑÕÐ’U’×ÒSPQÑWÕTÐQÑWÔÐSTQÐ’Uˆ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’U’×ÒSPQÑWÕ’QU×ÕTWÌ‘KKˆšÐÛÛ\Û™[X\[™ÞßKK˜[ÙK‹O›\ØXTØ[\\ÊHˆËÈS”Ñ‘T—ÔÔÈ™XØ]\ÙHN	HÙˆ\È]IÜÈ™\ÛÛ™\ÈÛÜHÝ]ÙˆHTˆËÈY™™\ˆ˜]\ˆ[ˆHÛÛÝ\ˆÛ™H
]ÈÚYÝÈØ\ØØY\È[™HØÙ[™H\ˆËÈ]È\[Ù‹YšY[\ÜÈ™XYÈ˜XÚÊH8 %ÙYHÔ™\ÛÛ™K‚ˆËÈÐSTQÛ›HÚ[ˆH]šXÙHØ[YH\Ú]˜^H˜XÚ[™ËÚXÚ\ÈÚ]›Ý]BˆËÈ
ŠIÜÈ˜XÝÜˆ\ÜÈ™YYÈÈ™XYHØÙ[™H\ˆÛÛ™][Û˜[˜]\ˆ[‚ˆËÈ[˜ÛÛ™][Û˜[™XØ]\ÙHH\ØYÙHš]Ø[ˆÚ[™ÙHHš]™\‰ÜÈ\ÛÛ\™\ÜÚ[Û‚ˆËÈXÚ\Ú[Û‹[™Ö—Õ’×Ô•L8 %HX\Ý\ˆ\›HÚ[˜ÙH\8 %]\ÝÝ^HBˆËÈš]Y›Ü‹Xš]\ØÜš\[ÛˆÙˆH™KT•™[™\™\‹‚ˆPÜ™X]R[XYÙJ‹O™\”Ö
‹O\™Ù]ÚY
K”ÊY˜[R
KˆY˜[Q\›Ü›X]

Kˆ’×ÒSPQÑWÕTÐQÑWÑTÔÕSÒSÐUPÒQS•Ð’Uˆ’×ÒSPQÑWÕTÐQÑWÕS”Ñ‘T—ÑÕÐ’Uˆ’×ÒSPQÑWÕTÐQÑWÕS”Ñ‘T—ÔÔ×Ð’Uˆ
‹Oœ[˜X›YÈ’×ÒSPQÑWÕTÐQÑWÔÐSTQÐ’UˆJKˆ’×ÒSPQÑWÐTÔPÕÑTÐ’U’×ÒSPQÑWÐTÔPÕÔÕSÒSÐ’Uˆ’×ÒSPQÑWÕ’QU×ÕTWÌ‘KKšÐÛÛ\Û™[X\[™ÞßKK˜[ÙKˆ‹O›\ØXTØ[\\ÊHˆPÜ™X]QY˜[T™\ÛÛ™U\™Ù]Ê”Ö
‹O\™Ù]ÚY
K”ÊY˜[R
JJBˆÂˆœš[ŠÝ\œ‹–Ýš×H™[™\ˆ\™Ù]Ü™X][ÛˆRSQˆŠNÂˆ™]\›ˆ˜[ÙNÂˆBˆ˜[YR[XYÙJ‹O˜ÛÛÜ‹‘QSHÛÛÝ\ˆ	]^	]H‹”Ö
‹O\™Ù]ÚY
K”ÊY˜[R
JNÂˆ˜[YR[XYÙJ‹O™\‘QSH\	]^	]H‹”Ö
‹O\™Ù]ÚY
K”ÊY˜[R
JNÂˆYˆ
PÜ™X]PÛÛ\]Xš[]QY˜[T\ÜÊ
JBˆÂˆœš[ŠÝ\œ‹–Ýš×H[Ø[ˆKŒHQSH™[™\ˆ\ÜÈÜ™X][ÛˆRSQˆŠNÂˆ™]\›ˆ˜[ÙNÂˆB‚ˆËÈH\Û˜\ÚÝ\ÈØ[\Y›ÝYÚHØ[YHš[™\ÜÈX\[™HØ[YHÚ[™ÛBˆËÈ[™X\ˆØ[\\ˆ\È]™\žHÝ\ˆ^\™KÛÈH]šXÙH\ÈÈ™HX›HÈš[\‚ˆËÈ]›Ü›X]ˆÚXÚÙY˜]\ˆ[ˆ\ÜÝ[YYˆYˆ]Ø[››ÝHXÝ\™HÛÝ[™BˆËÈ[™Yš[™Y˜]\ˆ[ˆÜ›Û™Ë[™Ú[[H8 %Ø^HÛÈÛ˜ÙKÝYK‚ˆÂˆšÑ›Ü›X]›Ü\Y\ÈœßNÂˆšÑÙ]\ÚXØ[]šXÙQ›Ü›X]›Ü\Y\Ê‹Oœ\ÚXØ[‹O™\™›Ü›X]	™œ
NÂˆYˆ
Jœ›Ü[X[[[™Ñ™X]\™\È	ˆ’×Ñ“Ô“PUÑ‘PUT‘WÔÐSTQÒSPQÑWÐ’U
JBˆœš[ŠÝ\œ‹–Ýš×HÐT“’S‘Îˆ	]H\È›ÝØ[\XX›HÛˆ\È]šXÙH8 %‚ˆ™\™\ÛÛ™\ÈÚ[›Ý™H™XYX›HžHHÝY\Ýˆ‹ˆ[œÚYÛ™Y
‹O™\™›Ü›X]
JNÂˆ[ÙHYˆ
Jœ›Ü[X[[[™Ñ™X]\™\È	‚ˆ’×Ñ“Ô“PUÑ‘PUT‘WÔÐSTQÒSPQÑWÑ’ST—ÓS‘PT—Ð’U
JBˆœš[ŠÝ\œ‹–Ýš×H“ÕNˆ\›Ü›X]	]H\È›È[™X\ˆš[\š[™ÎÈ‚ˆ™\Û˜\ÚÝÈ\™HØ[\YÚ]H[™X\ˆØ[\\—ˆ‹ˆ[œÚYÛ™Y
‹O™\™›Ü›X]
JNÂˆB‚ˆËÈLŽPˆÙˆ\‹Yœ˜[YH\™[˜KˆHœ›Û[™	ÜÈÝ™X[\È\™HÛX[ÈØ[Y\^H\ÈBˆËÈ]Y\Ý[Û‹[™HYÚ]Ø]\ˆX\šÈ\Èš[YÚ]HÝ]ÈÛÈH[X™\ˆØ[ˆ™BˆËÈ˜Z\ÙYÛˆ]šY[˜ÙH˜]\ˆ[ˆÝY\ÜÙY]YØZ[‹‚ˆËÂˆËÈ]Ô“ÕÔÈ›ÝÈ
ÙYH™YÚ[‘œ˜[YJKÛÈ\È\ÈHÕT•S‘ÈÚ^™H˜]\ˆ[ˆH[Z]‚ˆËÈLŽ\ÈÙ\\ÈHÝ\[X™\˜][Nˆ]\ÈHÚ^™H]™\žHYX\Ý\™[Y[[ˆ\ÂˆËÈÜ\È\NØ\ÈZÙ[ˆ]ÛÈÖ—Õ’×Ó“×ÐT‘SWÑÔ“ÕÕLX™\›ÙXÙ\ÈHÛˆËÈ™[™\™\ˆ^XÝH[™™[XZ[œÈH\ØX›HÛÛ›Û\›KˆÖ—Õ’×ÐT‘SWÓPSˆÙ]ÈBˆËÈÝ\ÚXÚ\ÈÝÈHLŽ]œËMLLˆKÐˆ]Y[YšYYH›XÚÈœ˜[Y\ÈØ\È[‹‚ˆÝ]XÈÛÛœÝZ[Ý\™[˜SXˆBˆ[ŠÖ—Õ’×ÐT‘SWÓPˆŠHÈÝÝ[
[ŠÖ—Õ’×ÐT‘SWÓPˆŠK[‹L
HˆLŽÂˆËÈHÔ“ÔÔËQ”SQHÝ™X[HÝÜ™KØ[YH\ØYÙH[™Y[[ÜžH\H\ÈH\™[˜H™XØ]\ÙHBˆËÈÔHØ[››Ý[[H\\8 %HÛ›HY™™\™[˜ÙH\È]\ÈÛ™H\È›Ý™\Ù]]ˆËÈHÝØ\ˆ\œÚ\ÝXZ[[˜[˜ÙXÝX›\È]Ú[ˆHœ˜[YHÝ™\œ[œÈ]‚ˆËÈÖ—Õ’×ÔT”ÒTÕÓPSˆÙ]ÈHÝ\‚ˆËÂˆËÈ
Š•HQUSTÈL8 %HÑRSS‘È8 %TÈÑˆT•ÎKS‘UÐTÈLŽ“ÔˆÑS•KUÓÂˆËÈT•ËŠŠˆHÜ›ÝÝ\È›ÝH˜XÚÙÜ›Ý[™ÛÜÝˆ]\ÈØZ][œ˜[Y\ÒYX
ÂˆËÈšÑ]šXÙUØZ]YX
ÈHÜÝ]š\ÚX›H[ØØ][Ûˆ[™PT
Èœ™YZ[™ÈHÛY™™\‹[ˆËÈÛˆH[\[œÚYHÓ‘Hœ˜[YKˆHÜ\˜]Ü‰ÜÈ\MÎHÙ\ÜÚ[ÛˆÜ™]ÈÚXÙKLŽOˆM‚ˆËÈOˆLL‹[™
Š˜›ÝÜ›ÝÝÈÙ\™HHÛÜœÝœ˜[YHÙˆZ\ˆÝÛˆ[‹\ÙXÛÛ™Ú[™ÝÈ[™ˆËÈ›ÝÙ\™HHÛ›H[™ÜÈ^H™[[ˆM‹ŽÙXÛÛ™ÈÙˆ^JŠˆ
0©Í™H0©ÌË0©Í™ŠK‚ˆËÂˆËÈ
Š•H’T”Õ’VÐTÈLLˆS‘UÐTÈÓ“HSˆ’QÒ8 %‘PÓÔ‘QT‘H‘PÐUTÑHHT”“Ô‚ˆËÈTÈHTÑQ•ST•ŠŠˆHÛÜÝØØ[\ÈÚ]H‘UÈY™™\‰ÜÈÚ^™H[™HÝÜ™BˆËÈÕP“TËÛÈ˜Z\Ú[™ÈHÝ\Ù\È›Ý™[[Ý™HHÛ\ÜÎˆ]ÚÚ\ÈHÚX\X\›BˆËÈÜ›ÝÝÈ[™X]™\ÈH^[œÚ]™H]HÛ™Kˆ]LLˆHÜ\˜]Ü‰ÜÈ™^Ù\ÜÚ[ÛˆÜ™]ÂˆËÈÛ˜ÙKLLˆOˆL[™]Ú[™ÛHÜ›ÝÝØ\È
ŠŒÌŽKŒˆ\È[ˆÛ™Hœ˜[YJŠˆ
ØZ]ÈŽH
ÂˆËÈ[ØØ]KÛX\MŽH
Èœ™YK[ÛKŒÊH8 %^H™[][™ÛÜœ™XÝH™\ÜY]BˆËÈY\‹[ØY]ÚØ\ÈÛÛ™H[™Û™H]HÝ]\ˆY\X\™YˆÛÈ]Ú\È]È[™ˆËÈMN\È™XØ[YHÛ™H]ÍL‹‚ˆËÂˆËÈ
ŠŒLTÈÔ\œÚ\ÝÙZ[[™ØÛÈHÝÜ™HØ[ˆ™]™\ˆÜ›ÝÈUSŠŠˆ]\ÈBˆËÈÝXÝ\˜[ÝX\˜[YH˜]\ˆ[ˆHšYÙÙ\ˆÝY\ÜË[™HÛÜÝ\Èš[™XØ]\ÙHÙˆBˆËÈYX\Ý\™Y^\Þ[[Y]žNˆHØ[YH[ØØ][ÛˆÛÜÝÈ
ŠŸŒL\È]›ÛÝ[™MH\ÂˆËÈZY\[ŠŠ‹Ú[˜ÙHZY\[ˆ]]\Ýš[™HÚYØXž]HÚ[HHÛY™™\ˆ\ÈÝ[]™BˆËÈ[™HXXÚ[™H\È[™\ˆØYˆ›ÛÝœ˜[Y\ÈYX\Ý\™YXÜ›ÜÜÈH™YHÝ\È8 %ˆËÈLŽˆŒÍËÈÈŒÍŒ‹LLŽˆŒÈËŒÈÈ‹Ë
ŠŒLˆŒ‹ŽJŠˆ8 %K™KˆHHÐ‚ˆËÈ[ØØ][Ûˆ\È[œÚYHHÜ™XYÙˆH›ÛÝœ˜[YH]\ÈŒÌLL\ÈÚ]]™\ˆÙHË‚ˆËÂˆËÈ][™ÈHÙZ[[™È\È“ÕHÜ›ÝÝˆ\œÚ\ÝXZ[[˜[˜ÙX	ÜÈ[ÙKXœ˜[˜Ú›ÜÈ[™ˆËÈ™Yš[ÈHØXÚKÚXÚÛÜÝÈHØZ][œ˜[Y\ÒYX[™›Ý[™È[ÙK‚ˆËÂˆËÈ›ÝHHØZ]È\™HHÓPSTÕÙˆH™YH\›\È[ˆHÜ›ÝÝˆ™[˜Ú[™ÈHÛˆËÈY™™\ˆ]Ø^H8 %HØš[Ý\Ëš[˜Ú\Y™\Z\ˆ8 %ÛÝ[]™H›ÝYÚNIHÙˆ]‚ˆÝ]XÈÛÛœÝZ[Ý\œÚ\ÝXˆBˆ[ŠÖ—Õ’×ÔT”ÒTÕÓPˆŠHÈÝÝ[
[ŠÖ—Õ’×ÔT”ÒTÕÓPˆŠK[‹L
HˆLÂˆ‹Oœ\œÚ\ÝÛˆHQ[“ÛŠÖ—Õ’×Ó“×ÔT”ÒTÕÔÕ‘PSTÈŠNÂ‚ˆËÈ[››Ý[˜ÙH]Ù[‹™XØ]\ÙH[ˆ\›H›Ø›ÙHØ[ˆÙYH[ˆHÙÈ\È[ˆ\›H]Ø[››Ý™BˆËÈÚÝÛˆÈ]™H[™ØYÙY
ÛÝÚHMLJK‚ˆËÈH›Ý[™[™]ÈÛÈÝ™\œšY\Ëˆ[››Ý[˜ÙYÚ[™]™\ˆ]\È›ÝHY˜][ÛÈ[‚ˆËÈ\›H]Y›Ý[™ØYÙHØ[››Ý™HZ\ÝZÙ[ˆ›ÜˆÛ™H]Y
ÛÝÚHMLJK‚ˆYˆ
ÛÛœÝÚ\ŠˆØˆH[ŠÖ—Õ’×ÔÕ‘PSWÑÕPT‘Ð–UTÈŠJBˆÂˆÛÛœÝ[œÚYÛ™YÛ™ÈÛ™ÈˆHÝÝ[
Ø‹[‹L
NÂˆËÈ™[ÝÈÑÝX\™›ØÚÜÈH›ØÚÈ\š]Y]XÈYÙ[™\˜]\ÎÈ™Y\ÙH˜]\ˆ[‚ˆËÈÚ[[HØ[\[™È›Ý[™Ë‚ˆYˆ
ˆHÑÝX\™›ØÚÜÊBˆÂˆ×ÙÝX\™ž]\ÈHÚ^™WÝ
ŠNÂˆœš[ŠÝ\œ‹–Ýš×HÖ—Õ’×ÔÕ‘PSWÑÕPT‘Ð–UTÏI[H8 %HÝÜ™IÜÈÝX\™\È‚ˆ‘VPÕ\È	[Hž]\È[™Ø[\\ÈX›Ý™H]
Y˜][	^JWˆ‹ˆ‹‹ÑÝX\™ž]\ÑY˜][
NÂˆBˆ[ÙBˆÂˆœš[ŠÝ\œ‹–Ýš×HÖ—Õ’×ÔÕ‘PSWÑÕPT‘Ð–UTÏI[H‘Q•TÑQ8 %]\Ý™HH	^NÈ‚ˆšÙY\[™È	^Wˆ‹‹ÑÝX\™›ØÚÜË×ÙÝX\™ž]\ÊNÂˆBˆBˆ[ÙBˆÂˆœš[ŠÝ\œ‹–Ýš×HÝ™X[HÝX\™^XÝÈ	^Hž]\ËØ[\YX›Ý™H‚ˆŠÖ—Õ’×ÔÕ‘PSWÑÕPT‘Ð–UTÏSˆÈÚ[™ÙK][HÊWˆ‹ˆ×ÙÝX\™ž]\ÊNÂˆB‚ˆ×ÙÝX\™^XÝH[“ÛŠÖ—Õ’×ÔÕ‘PSWÑÕPT‘ÑVPÕŠNÂˆYˆ
×ÙÝX\™^XÝ
Bˆœš[ŠÝ\œ‹–Ýš×HÖ—Õ’×ÔÕ‘PSWÑÕPT‘ÑVPÕLH8 %HÜ›ÜÜËYœ˜[YHÝÜ™IÜÈÝX\™‚ˆš\Ú\ÈU‘T–Hž]KÛÈ]Ø[››ÝZ\ÜÈHÛX[Y][ˆH\™ÙH‚ˆœÝ™X[KˆXYÛ›ÜÝXÈ›ÜˆÜ[ˆ][HË—ˆŠNÂ‚ˆËÈKKHÖ—Õ’×Ñ”SQT×ÒS—Ñ“QÒKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKBˆËÂˆËÈÝÈX[žHœ˜[Y\ÈHÔHX^H™HZXYÙˆHÔKˆH\ÈH™[™\™\ˆ\ÈÜ˜[‚ˆËÈ›ÜˆÙ[K]ÛÈ\ÎˆÝX›Z]Hœ˜[YK›ØÚÈÛˆ]È™[˜ÙK™XY]˜XÚË™\Ù[‚ˆËÂˆËÈÚH[Ü™H[ˆH\ÈH\™Ù\Ý][H[ˆH\™›Ü›X[˜ÙH[ŽˆHÜ›ÝÙœ˜[YH\ÂˆËÈŒËÈ\ÈÙˆÔH›ÛÝÙYžHŒM‹H\ÈÙˆÔKÝšXÝH[ˆÙ\šY\ËÛÈHØ\™\ÂˆËÈYHŽ	HÙˆ]™\žHœ˜[YH[™Hš]™\ˆÛÜœ™XÝHÛÝ™\›œÈ]ÝÛˆÈHZYÛØÚÂˆËÈ
ÛÝÚHŒÌK0©Í˜\ŠKˆÖ—Õ’×Ó“×ÔÕP“RULXYX\Ý\™YHÙZ[[™ÈÛˆ™[[Ýš[™È]ˆËÈÙ\šX[\Ø][Ûˆ8 %ÔK[Û›H[YKŒK^8 %Ú]Ý]Z[[™È]ˆ\È\ÈZ[[™È]‚ˆËÂˆËÈÓÈTÈHQUSS‘‘QHTÈURSP“K[™™Z]\ˆ\ÈHÝY\ÜÈX›Ý]ÚXÚˆËÈÚ[œÎˆÛ™Hœ˜[YHÙˆÝ™\›\[™XYHÛÝ™\œÈHÔHÚÜ\ˆ[ˆHÔKÛÈÈÚÝ[ˆËÈ™XY\È›Ú\ÙK[™Yˆ]Ù\È›Ý[ˆH[Ù[ÙˆÚ\™HH[YHÛÙ\È\ÈÜ›Û™ÂˆËÈ[™]\ÈÛÜÛ›ÝÚ[™Ëˆ›Ý\™HÛ™Hš[˜\žKÚXÚ\ÈÚ]XZÙ\ÈHKÐˆYØ[‚ˆÝ]XÈÛÛœÝÚ\ŠˆšY‘[ˆH[ŠÖ—Õ’×Ñ”SQT×ÒS—Ñ“QÒŠNÂˆ‹O™œ˜[Y\Ò[‘›YÚHšY‘[ˆÈZ[Ì—Ý
ÝÝ[
šY‘[‹[‹L
JHˆŽÂˆYˆ
‹O™œ˜[Y\Ò[‘›YÚJBˆ‹O™œ˜[Y\Ò[‘›YÚHNÂˆYˆ
‹O™œ˜[Y\Ò[‘›YÚˆÓX^œ˜[Y\Ò[‘›YÚ
Bˆ‹O™œ˜[Y\Ò[‘›YÚHÓX^œ˜[Y\Ò[‘›YÚÂ‚ˆËÈH[œÝ[Y[È]ÐS““Õ™HÛ™Hœ˜[YH]K[™\™H\™Y›Ü™H[ÝÙYÈ™]ÂˆËÈH\[[š[™È˜]\ˆ[ˆÚ[[H›ÙXÙHZ\Ø[YÛ™Y]šY[˜ÙK‚ˆËÂˆËÈHY™\œ™Y™\Ù[[™ÈHÚ[™ÝÈœ˜[YH‹LIÜÈ^[ÈÚ[HH™[™\™\‰ÜÂˆËÈÛ˜\ÚÝ[XYÙ\Ë]È™\ÛÛ™HÚZ[ˆ[™]È™YÚ\Ý\ˆÝ]H\™H[œ˜[YH‰ÜËˆBˆËÈœ˜[YK\Ý]È[™H]È\™Hš^Y›Ü\›H™[ÝÈ8 %^HØ\œžHH™\Ù[YˆËÈœ˜[YIÜÈÝÛˆY]Y]H[ˆ]ÈÛÝ8 %]™YH[œÝ[Y[È™XYHU‘H™\ÛÛ™BˆËÈÚZ[ˆ™^ÈH™\Ù[Y^[È[™Ø[››Ý™Hš^Y]Ø^N‚ˆËÈÖ—Õ’×ÔÓTÓÓ—Ð“PÒÈ[™Ö—Õ’×ÔÓTÓÓ—ÑT’ÈšYÙÙ\ˆH[\ÙˆHÝ\œ™[ˆËÈÛ˜\ÚÝÈœ›ÛHH^[\ÝÛˆH™\Ù[Yœ˜[YK[™Ö—Õ’×Ñ”SQWÔÕU×ÔÕT‘PÑBˆËÈ™XYÈ˜XÚÈH˜[YYÛ˜\ÚÝÈÚ][ˆHØ[YHÝ]È[™H\ÈH™\Ù[YÛ™K‚ˆËÈHÛ™KYœ˜[YHÚÙ]È\™H\ÈHXYÛ›ÜÝXÈ]]ZY]H[œÝÙ\œÈX›Ý]HÜ›Û™Èœ˜[YKˆËÈÚXÚ\ÈÛÜœÙH[ˆHÛÝÙ\ˆXYÛ›ÜÝXÈ
ÛÝÚHÉÜÈÛÝ\Ú[Žˆ[ˆ[œÝ[Y[]ˆËÈ™\ÜÈX›Ý]ÛÛY][™ÈÝ\ˆ[ˆÚ]]˜[Y\ÊK‚ˆYˆ
[ŠÖ—Õ’×ÔÓTÓÓ—Ð“PÒÈŠH[ŠÖ—Õ’×ÔÓTÓÓ—ÑT’ÈŠHˆ[ŠÖ—Õ’×Ñ”SQWÔÕU×ÔÕT‘PÑHŠH[ŠÖ—Õ’×ÔÓTÑSTŠHˆ[“ÛŠÖ—Õ’×Ó“×ÔÕP“RUŠJBˆÂˆYˆ
‹O™œ˜[Y\Ò[‘›YÚOHJBˆœš[ŠÝ\œ‹–Ýš×Hœ˜[Y\ËZ[‹Y›YÚ›Ü˜ÙYÈNˆHÛ˜\ÚÝXÚZ[ˆÜˆ‚ˆ››Ë\ÝX›Z][œÝ[Y[\ÈÛˆ[™ÜÙH™XYH™\ÛÛ™HÝ]H‚ˆ›ÙˆHœ˜[YH™Z[™È‘PÓÔ‘Q›ÝHÛ™H™Z[™È™\Ù[YˆŠNÂˆ‹O™œ˜[Y\Ò[‘›YÚHNÂˆBˆœš[ŠÝ\œ‹–Ýš×Hœ˜[Y\È[ˆ›YÚˆ	]I\×ˆ‹‹O™œ˜[Y\Ò[‘›YÚˆ‹O™œ˜[Y\Ò[‘›YÚOHHÈˆ
ÝX›Z][™ØZ]ÈH™K\\LŒÈ™[™\™\ŠHˆˆˆŠNÂ‚ˆËÈŒIÜÈ\›\Ë™XY‘Q“Ô‘HHY™™\œÈ^\Ý™XØ]\ÙHHÛÛ›Û\›H]\Ý›Ý[ØØ]BˆËÈHÝX‹X\™[˜H][8 %[ˆ\›H]Ý[^\È›Üˆ]ÈÝXš™XÝ	ÜÈY[[ÜžH\È›ÝBˆËÈÛÛ›Û›Üˆ]ÈY[[ÜžK‚ˆËÈÑ‘ˆ–HQUS‘PÐUTÑHUÐTÈQPTÕT‘QS‘UTÈH•SˆŒHÙ\È^XÝHÚ]]ˆËÈØ\ÈZ[ÈÈ8 %L	HÙˆ˜]ÜÈÙ\™Y™K^™\›ÙY™\›È˜[˜XÚÜË™\›È˜Z[‹[™ˆËÈ\™˜Ø^\ÈÈ\ÈÙˆ×ÛY[\Ù]Ø]ž˜YH[\8 %[™H[\	ÜÈÔH\ˆœ˜[YBˆËÈY›Ý[Ý™HžHH[™™YÙˆHZ[\ÙXÛÛ™
™YH[œÈ[ˆ\›KÚ^X]ÚY˜]ÂˆËÈ˜[™Ë0©ÌLŒŠKˆHÛÜÝ\ÈÝÜ™HS‘ÒQÚXÚ\ÈXXÚ[™K]ÚYKÛÈ\ÜÝZ[™ÈBˆËÈØ[YHÝÜ™\Èœ›ÛH[›Ý\ˆÛÜ™H^\È›Ý[™Ë‚ˆËÂˆËÈÑTUTˆSˆSUQ[™›ÜˆH™X\ÛÛˆÚ]H]HÛˆ]ˆ\È›Þ\È[ˆXÛÜ™BˆËÈMRˆ\ÚÝÜÚÜÙHY[[ÜžH\H\ÈH›Ý[™ˆHXXÚ[™HÚ]HÛÝÙ\ˆÛÜ™BˆËÈ™[]]™HÈ]ÈY[[ÜžH8 %Hž^™[ˆÈÝ[™Z[ˆÙˆ\LËÜˆHÝX[HXÚÈ8 %X^BˆËÈ]HØ[YHÈ\È˜XÚÈÛˆHÔHÚYHÙˆHYÙ\‹Ú\™H™[ØØ][™È]ÛÝ[ˆËÈ^KˆÖ—Õ’×Ô‘V‘T“ÏLX\ÈÝÈ]Ù]È\ÚÙY[™]ÛÜÝÈÛ™H[‹‚ˆ×Ü™^™\›ÓÙ™ˆHQ[“ÛŠÖ—Õ’×Ô‘V‘T“ÈŠNÂˆ×Ü™^™\›ÔÚ\ÛÛˆH[“ÛŠÖ—Õ’×Ô‘V‘T“×ÔÒTÓÓˆŠNÂˆYˆ
Y×Ü™^™\›ÓÙ™ŠBˆœš[ŠÝ\œ‹ˆ–Ýš×HÖ—Õ’×Ô‘V‘T“ÏLH8 %HÚ\™YX›ØÚÈ™K^™\›È\ÈÓˆ
\LLHŒJKˆ‚ˆ“QPTÕT‘QH•SÛˆ\È›ÞˆHY[\Ù]X]™\ÈH[\[™Hœ˜[YH‚ˆ™Ù\È›Ý[Ý™Kˆ[ˆ\›K›ÝHY˜][—ˆŠNÂˆYˆ
×Ü™^™\›ÔÚ\ÛÛŠBˆœš[ŠÝ\œ‹–Ýš×HÖ—Õ’×Ô‘V‘T“×ÔÒTÓÓLH8 %H™K^™\›ÈÜš]\ÈPH[œÝXY‚ˆ›ÙˆˆHPÕT‘HUTÕ”‘PRÎÈYˆ]Ù\È›ÝH˜\Ý]‚ˆ›™]™\ˆ[™ØYÙY[™]™\žH[X™\ˆœ›ÛH]\ÈYX[š[™Û\Ü×ˆŠNÂˆYˆ
PÜ™X]PY™™\Š‹O˜\™[˜K
\™[˜SXˆŒ
H
ˆ‹O™œ˜[Y\Ò[‘›YÚˆ’×Ð•Q‘‘T—ÕTÐQÑWÕ‘T•VÐ•Q‘‘T—Ð’U’×Ð•Q‘‘T—ÕTÐQÑWÒS‘VÐ•Q‘‘T—Ð’Uˆ’×Ð•Q‘‘T—ÕTÐQÑWÔÕÔQÑWÐ•Q‘‘T—Ð’Uˆ’×ÓQSSÔ–WÔ“ÔT•WÒÔÕÕ’TÒP“WÐ’Uˆ’×ÓQSSÔ–WÔ“ÔT•WÒÔÕÐÓÒT‘S•Ð’UˆÊ™]šXÙPY™\ÜÏJ‹ÝYKœ\‹Yœ˜[YH\™[˜HŠHˆ
×Ý^›Ð˜]ÚH[“ÛŠÖ—Õ’×Ó“×ÕVÐUÒŠK˜[ÙJHˆËÈŒIÜÈÚ\™YÝX‹X\™[˜H
\LLH0©Í
KˆÚ^™Y[ˆÓÕË›ÝYYØXž]\ÎˆH[X™\‚ˆËÈ]X]\œÈ\È˜]ÜÈ\ˆœ˜[YK[™MÍˆÛÝÈH™YÚ[Ûˆ\ÈÙ[\ÝBˆËÈŽKÌHÜ\˜]Ü‰ÜÈÜ›ÝÙ™XXÚ\ËÛÈHÝ™\™›ÝÈ]\ÈHØY™]H™]ˆËÈ˜]\ˆ[ˆH[™ÈHYX\Ý\™[Y[[œÈ›ÝYÚˆM‹ˆPˆÝ[]ˆœ˜[Y\È[‚ˆËÈ›YÚYØZ[œÝ[ˆ\™[˜H]\È[™XYH[™™YË‚ˆ
×Ü™^™\›ÓÙ™ˆÈ˜[ÙBˆˆPÜ™X]PY™™\Š‹OœÚ\™Y\™[˜KˆšÑ]šXÙTÚ^™JÔÚ\™YÝšYJH
ˆMÍˆ
‚ˆ‹O™œ˜[Y\Ò[‘›YÚˆ’×Ð•Q‘‘T—ÕTÐQÑWÔÕÔQÑWÐ•Q‘‘T—Ð’Uˆ’×ÓQSSÔ–WÔ“ÔT•WÒÔÕÕ’TÒP“WÐ’Uˆ’×ÓQSSÔ–WÔ“ÔT•WÒÔÕÐÓÒT‘S•Ð’UˆÊ™]šXÙPY™\ÜÏJ‹ÝYKœÚ\™YÝX‹X\™[˜HŠJHˆËÈHÕQÒS‘ÈT‘SHTÈÕ^\ØYÛÝØÑQÓQS•ÈTÈÑˆT•ÎK[™]Ü™]Èœ›ÛBˆËÈH›]PˆÈÈÌˆPˆÛÈ]\][Ûš[™È]Y›Ý˜\œ›ÝÈH\‹]\ØYˆËÈÙZ[[™È™[ÝÈ[ž][™È™X[ˆHYX\Ý\™Y\™Ù\ÝÚ[™ÛH\ØYÛˆBˆËÈ]]Û›Û[Ý\È›Ý]H\ÈKŒÌÈPŽÈÙYHH\ØY\š[™ÈÛÛ[Y[›ÜˆHÚÛH\™Ý[Y[‚ˆPÜ™X]PY™™\Š‹OœÝYÚ[™ËÕ^\ØYÛÝÈ
ˆÕ^ÛÝž]\Ëˆ’×Ð•Q‘‘T—ÕTÐQÑWÕS”Ñ‘T—ÔÔ×Ð’Uˆ’×ÓQSSÔ–WÔ“ÔT•WÒÔÕÕ’TÒP“WÐ’Uˆ’×ÓQSSÔ–WÔ“ÔT•WÒÔÕÐÓÒT‘S•Ð’Uˆ˜[ÙJHˆPÜ™X]PY™™\Š‹Oœ™XY˜XÚËˆËÈšYÈ[›ÝYÚ›ÜˆH™\Ù[Yœ˜[YHS‘›ÜˆH\™Ù\ÝÛ˜\ÚÝˆËÈÖ—Õ’×ÔÓTÑSTZYÚ™XY˜XÚÈ8 %\È]IÜÈÚYÝÈØ\ØØYBˆËÈ\ÈMžL[™H[\ÒÒTÈ[ž][™È]Ù\È›Ýš]ˆËÈÚXÚÛÝ[]™HXYHHÛ™HÝ\™˜XÙH[™\ˆ[™\ÝYØ][ÛˆBˆËÈÛ™HÝ\™˜XÙHXœÙ[œ›ÛHH\™XÝÜžK‚ˆÝŽ›X^
Z[Ý
”Ö
‹O\™Ù]ÚY
JH
ˆ”Ê‹O\™Ù]ZYÚ
KˆZ[Ý
”Ö
MŠJH
ˆ”ÊL
JH
ˆˆ’×Ð•Q‘‘T—ÕTÐQÑWÕS”Ñ‘T—ÑÕÐ’U™XY˜XÚÓY[[ÜžT›ÜÊ
K˜[ÙJJBˆÂˆœš[ŠÝ\œ‹–Ýš×HY™™\ˆ[ØØ][ÛˆRSQˆŠNÂˆ™]\›ˆ˜[ÙNÂˆB‚ˆËÈHVT‘HTÐQ’S‘È
\ÎH][HJKˆÛ™HÛÛ[X[™Y™™\‹Û™H™[˜ÙH[™Û™BˆËÈÙYÛY[ÙˆHÝYÚ[™È\™[˜H\ˆÛÝÈÙYHHš[™ÉÜÈÛÛ[Y[›ÜˆH\ÚYÛ‹ˆ]\ÂˆËÈÜ™X]YT‘KY\ˆ‹OœÝYÚ[™Ø™XØ]\ÙHHÙYÛY[˜\Ù\È\™HÙ™œÙ]È[È]‚ˆËÂˆËÈÖ—Õ’×ÕVÑ“TÒÕÐRULXÝ[[ØØ]\ÈHš[™È8 %H\›H\ÈÈ™HÛ™Hš[˜\žH[™ˆËÈHÛÝÈÛÜÝ™YHÛÛ[X[™Y™™\œÈ[™™YH™[˜Ù\È8 %]™]™\ˆY˜[˜Ù\È]ÛÂˆËÈ]È\ØYÈ[]™H[ˆÙYÛY[[™]™\žH›\ÚÝX›Z]È[™ØZ]È^XÝH\ÂˆËÈ\ÍÈY‚ˆ×Ý^›\ÚØZ]H[“ÛŠÖ—Õ’×ÕVÑ“TÒÕÐRUŠNÂˆÂˆšÐÛÛ[X[™Y™™\[ØØ]R[™›ÈØš^Âˆ’×ÔÕ•PÕT‘WÕTWÐÓÓSPS‘Ð•Q‘‘T—ÐSÐÐUWÒS‘“ÈNÂˆØšK˜ÛÛ[X[™ÛÛH‹O˜ÛYÛÛÂˆØšK›]™[H’×ÐÓÓSPS‘Ð•Q‘‘T—ÓU‘SÔ’SPT–NÂˆØšK˜ÛÛ[X[™Y™™\ÛÝ[HNÂˆšÑ™[˜ÙPÜ™X]R[™›Èš^È’×ÔÕ•PÕT‘WÕTWÑ‘SÑWÐÔ‘PUWÒS‘“ÈNÂˆ›Üˆ
Z[Ì—ÝHHÈHÕ^\ØYÛÝÎÈ
ÊÚJBˆÂˆ’×ÐÒPÒÊšÐ[ØØ]PÛÛ[X[™Y™™\œÊ‹O™]šXÙK	ØšK	™×Ý^ÛÝÖÚWK˜ØŠKˆšÐ[ØØ]PÛÛ[X[™Y™™\œÈ
^\™H\ØYÛÝ
HŠNÂˆ’×ÐÒPÒÊšÐÜ™X]Q™[˜ÙJ‹O™]šXÙK	šK[‹	™×Ý^ÛÝÖÚWK™™[˜ÙJKˆšÐÜ™X]Q™[˜ÙH
^\™H\ØYÛÝ
HŠNÂˆ×Ý^ÛÝÖÚWK˜˜\ÙHHšÑ]šXÙTÚ^™JJH
ˆÕ^ÛÝž]\ÎÂˆ˜[YSØš™XÝ
Z[Ý
×Ý^ÛÝÖÚWK˜ØŠK’×ÓÐ’‘PÕÕTWÐÓÓSPS‘Ð•Q‘‘T‹ˆ^\™H\ØYÛÝ	]H‹JNÂˆBˆ×Ý^ÛÝHÂˆ×ÜÝYÚ[™ÐÝ\œÛÜˆHÂˆœš[ŠÝ\œ‹ˆ–Ýš×H^\™H\ØYš[™Îˆ	]HÛÝÈ	[HPˆÝYÚ[™Ë›\Ú	\È‚ˆŠÖ—Õ’×ÕVÑ“TÒÕÐRULH\ÈHÛÛ›Û\›JWˆ‹ˆÕ^\ØYÛÝË
[œÚYÛ™YÛ™ÈÛ™ÊJÕ^ÛÝž]\ÈˆŒ
Kˆ×Ý^›\ÚØZ]È”ÕP“RUÈS‘ÐRUÈ
\ÍÈ™Z]š[Ý\ŠH‚ˆˆœÝX›Z]È[™Ù\È“ÕØZ]ŠNÂˆB‚ˆËÈÛ™H™\Ù[™XY˜XÚÈY™™\ˆ\ˆÛÝˆ]\È[X™\˜][H“ÕH™YÚ[ÛˆÙ‚ˆËÈ‹Oœ™XY˜XÚØˆ]Y™™\ˆ\È[ÛÈHÛ˜\ÚÝY[\\™Ù][™Ú\š[™È]ÛÝ[ˆËÈYX[ˆ[ˆ[œÝ[Y[	ÜÈ™XY˜XÚÈÛÝ[[™ÛˆÜÙˆHœ˜[YHHÚ[™ÝÈ\È›ÝˆËÈ™]ÚYY]8 %HÛÜœ\YXÝ\™H]\X\œÈÛ›HÚ[ˆHXYÛ›ÜÝXÈ\ÈÛ‹ÚXÚˆËÈ\ÈHÛÜœÝÚ[™ˆÚ^™YZÙH‹Oœ™XY˜XÚØÛÈHœ›ÛXY™™\ˆ™\ÛÛ™H\™Ù\ˆ[‚ˆËÈHœ˜[YH^[Ý[š]È˜]\ˆ[ˆ™Z[™ÈÚ[[H[˜Ø]Y‚ˆ›Üˆ
Z[Ì—ÝHHÈH‹O™œ˜[Y\Ò[‘›YÚÈ
ÊÚJBˆÂˆYˆ
PÜ™X]PY™™\Š‹O™œ˜[Y\ÖÚWKœ™\Ù[ˆÝŽ›X^
Z[Ý
”Ö
‹O\™Ù]ÚY
JH
ˆ”Ê‹O\™Ù]ZYÚ
KˆZ[Ý
”Ö
MŠJH
ˆ”ÊL
JH
ˆˆ’×Ð•Q‘‘T—ÕTÐQÑWÕS”Ñ‘T—ÑÕÐ’U™XY˜XÚÓY[[ÜžT›ÜÊ
K˜[ÙJJBˆÂˆœš[ŠÝ\œ‹–Ýš×H™\Ù[™XY˜XÚÈY™™\ˆ	]H[ØØ][ÛˆRSQˆ‹JNÂˆ™]\›ˆ˜[ÙNÂˆBˆB‚ˆËÈHÜ›ÜÜËYœ˜[YHÝÜ™H\È[ØØ]YÑTTUSHœ›ÛHHÚZ[ˆX›Ý™K[™]È˜Z[\™BˆËÈQÔQTÈ˜]\ˆ[ˆÚ[ÈH™[™\™\‹ˆ]\È[ˆÜ[Z\Ø][ÛŽˆÚ]Ý]]]™\žBˆËÈÝ™X[HZÙ\ÈH\‹Yœ˜[YH]ÚXÚ\ÈÚ]\ÈÜY›ÜˆÙ[K[Û™H\ÂˆËÈ[™ÚXÚÝ[™[™\œÈHØ[YHÛÜœ™XÝKˆ™Y\Ú[™ÈÈÝ\][™XØ]\ÙHBˆËÈXXÚ[™HÛÝ[›ÝÜ\™H[›Ý\ˆLŽPˆÛÝ[˜YHHÌ	Hœ˜[YK][YHØ]š[™È›ÜˆBˆËÈÚÛHXÝ\™KÚXÚ\È›ÝH˜YH[ž][™È\™HÚÝ[XZÙHÚ[[K‚ˆYˆ
‹Oœ\œÚ\ÝÛˆ	‰‚ˆPÜ™X]PY™™\Š‹Oœ\œÚ\Ý\œÚ\ÝXˆŒ\œÚ\Ý\ØYÙJ
Kˆ’×ÓQSSÔ–WÔ“ÔT•WÒÔÕÕ’TÒP“WÐ’Uˆ’×ÓQSSÔ–WÔ“ÔT•WÒÔÕÐÓÒT‘S•Ð’UˆÊ™]šXÙPY™\ÜÏJ‹ÝYK˜Ü›ÜÜËYœ˜[YHÝ™X[HÝÜ™HŠJBˆÂˆËÈSPÒÈÕÓˆHQTˆ‘Q“Ô‘HÒU’S‘ÈTˆ˜Z\Ú[™ÈHY˜][œ›ÛHLŽPˆÈBˆËÈÚYØXž]H]\Ý›Ý\›ˆ\ÈXXÚ[™H\È\ÜÈSH[ˆZ[™Hˆ[È\ÈXXÚ[™BˆËÈÜÙ\ÈHÝÜ™H[\™[H‹ÚXÚ\ÈHŒÌ	Hœ˜[YK][YH™YÜ™\ÜÚ[Ûˆ[™ÛÝ[™BˆËÈ[š\ÚX›Hœ›ÛHHÐˆ›Þ8 %H˜Z[\™H]ÙˆHšYÙÙ\ˆ™\]Y\Ý\È›ÝBˆËÈØ[YH][™]\ÈHÛ™H›Ø›ÙH\ÝÈ
ÛÝÚHŽJKˆHÛX[\ˆÝÜ™HÝ[ˆËÈÛÜšÜÎÈ]\ÝÜ›ÝÜË[™HÜ›ÝÝ\ÈÝšXÝH™]\ˆ[ˆ›ÈÝÜ™H][‚ˆÝ]XÈÛÛœÝZ[ÝÔ\œÚ\ÝY\–×HHÈLL‹LŽNÂˆ›ÛÛXYHH˜[ÙNÂˆ›Üˆ
Z[ÝXˆˆÔ\œÚ\ÝY\ŠBˆÂˆYˆ
XˆH\œÚ\ÝXŠBˆÛÛ[YNÈËÈ™]™\ˆ™˜[˜XÚÈˆ\Ø\™ˆœš[ŠÝ\œ‹ˆ–Ýš×HH	[HPˆÜ›ÜÜËYœ˜[YHÝ™X[HÝÜ™HÛÝ[›Ý™H[ØØ]Y8 %‚ˆœ™]žZ[™È]	[HP‹ˆ]Ú[Ô“ÕÈœ›ÛH\™K[™HÜ›ÝÝ\ÈHÚÛH‚ˆ™œ˜[YHÙˆ[\[YH
Ìˆ\È›ÜˆHMˆPˆÝ\ÌŽH\È›ÜˆHHÐˆÛ™JNÈ‚ˆÖ—Õ’×ÔT”ÒTÕÓPSˆÙ]ÈHÝ\ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\\œÚ\ÝX‹
[œÚYÛ™YÛ™ÈÛ™Ê[XŠNÂˆYˆ
Ü™X]PY™™\Š‹Oœ\œÚ\ÝXˆŒ\œÚ\Ý\ØYÙJ
Kˆ’×ÓQSSÔ–WÔ“ÔT•WÒÔÕÕ’TÒP“WÐ’Uˆ’×ÓQSSÔ–WÔ“ÔT•WÒÔÕÐÓÒT‘S•Ð’UˆÊ™]šXÙPY™\ÜÏJ‹ÝYK˜Ü›ÜÜËYœ˜[YHÝ™X[HÝÜ™H
˜[˜XÚÊHŠJBˆÂˆXYHHYNÂˆœ™XZÎÂˆBˆBˆYˆ
[XYJBˆÂˆœš[ŠÝ\œ‹–Ýš×HHÜ›ÜÜËYœ˜[YHÝ™X[HÝÜ™HÛÝ[›Ý™H[ØØ]Y]‚ˆ˜[8 %[›š[™ÈÚ]Ý]]ÚXÚ\ÈÛÝÙ\ˆ[™ÛÜœ™XÝˆŠNÂˆ‹Oœ\œÚ\ÝÛˆH˜[ÙNÂˆBˆBˆÜ™X]TÝÜ™SZ\œ›ÜŠ
NÂ‚ˆËÈÛÈØ[\\œË[™Û™HÛØ˜[ÚÚXÙH\ˆ˜]È\ÈHÝ]YÚ[\YšXØ][ÛŽˆBˆËÈ™]ÚÛÛœÝ[Ø\œšY\È\‹]^\™Hš[\ˆ[™Y™\ÜÈ[Ù\È]\ÈÙ\È›ÝˆËÈY]Û›Ý\‹ˆ˜[YY\™HÛÈH™^™XY\ˆÛ›ÝÜÈ]\ÈHØ\Ú]HØØ][Û‚ˆËÈ˜]\ˆ[ˆH^\Ý\žH[ˆHXÝ\™K‚ˆšÔØ[\\Ü™X]R[™›ÈÚ^È’×ÔÕ•PÕT‘WÕTWÔÐSTT—ÐÔ‘PUWÒS‘“ÈNÂˆÚK›XYÑš[\ˆH’×Ñ’ST—ÓS‘PTŽÂˆÚK›Z[‘š[\ˆH’×Ñ’ST—ÓS‘PTŽÂˆÚK›Z\X\[ÙHH’×ÔÐSTT—ÓRTPTÓSÑWÓS‘PTŽÂˆÚK˜Y™\ÜÓ[ÙUHH’×ÔÐSTT—ÐQ‘TÔ×ÓSÑWÔ‘TPUÂˆÚK˜Y™\ÜÓ[ÙUˆH’×ÔÐSTT—ÐQ‘TÔ×ÓSÑWÔ‘TPUÂˆÚK˜Y™\ÜÓ[ÙUÈH’×ÔÐSTT—ÐQ‘TÔ×ÓSÑWÔ‘TPUÂˆÚK›X^ÙH’×ÓÑÐÓSTÓ“Ó‘NÂˆËÈØ[\\ˆ\ÈRSˆ’SS‘PT‹[X™\˜][H[™\›X[™[Kˆ\IÜÈš\œÝˆËÈ][\]Mž[š\ÛÈ\™K™X\ÛÛš[™È™]™\žH™]ÚX›\Ú\È[™^ÛÈ\ÂˆËÈ\ÈÚ\™H[š\ÛÈÛÙ\Èˆ8 %[™H™\žHš\œÝØ\\™HÚÝÙY\šÈÜXÚÛHXÜ›ÜÜÂˆËÈHÚÛHœ˜[YK™XØ]\ÙH[™^[ÛÈÙ\™\ÈHÒQÕÈUTÈÛÚÝ\ËÚXÚˆËÈ\™Ø\™H™]Ú\ÈÚ][š\ÛÏL[™Ú[š[\œËˆH\‹Y™]ÚØ[\\ˆØXÚBˆËÈ
Ø[\\’[™^›Ü‘™]Ú
H\ÈÚ\™HH™]ÚÛÛœÝ[	ÜÈÝÛˆš[\ˆšY[È\™BˆËÈÛ›Ý\™YÈ\ÈØ[\\ˆ™[XZ[œÈH˜[˜XÚÈ[™HÖ—Õ’×Ó“×Ñ‘UÒÔÐSTT”ÂˆËÈ\›IÜÈÚÛHÛÜ›‚ˆœš[ŠÝ\œ‹–Ýš×H\‹Y™]ÚØ[\\œÈ	\È
[š\ÛÈ]šXÙH[Z]	KŒž
Wˆ‹ˆ[ŠÖ—Õ’×Ó“×Ñ‘UÒÔÐSTT”ÈŠHÈ“Ñ‘ˆ
Ö—Õ’×Ó“×Ñ‘UÒÔÐSTT”ÊHˆˆ“Óˆ‹ˆ‹O˜[š\ÛÓ[Z]
NÂˆYˆ
šÐÜ™X]TØ[\\Š‹O™]šXÙK	œÚK[‹	”‹O›[™X\”Ø[\\ŠHOH’×ÔÕPÐÑTÔÊBˆ™]\›ˆ˜[ÙNÂˆÚK›XYÑš[\ˆH’×Ñ’ST—Ó‘PT‘TÕÂˆÚK›Z[‘š[\ˆH’×Ñ’ST—Ó‘PT‘TÕÂˆYˆ
šÐÜ™X]TØ[\\Š‹O™]šXÙK	œÚK[‹	”‹OœÚ[Ø[\\ŠHOH’×ÔÕPÐÑTÔÊBˆ™]\›ˆ˜[ÙNÂ‚ˆËÈH˜]ËRQœ˜YÛY[[Ù[K[X™YY˜]\ˆ[ˆØYY
ÙYH˜]ÚYÜËšÛ
Kˆ]ˆËÈ\ÈÜ™X]Y[˜ÛÛ™][Û˜[H[™ÛÜÝÈH™]È[™™Yž]\Îˆ[ˆ[œÝ[Y[]\ÂˆËÈÈ™H[˜X›Y]•RS[YH\ÈÛ™H›Ø›ÙH\ÈÚ[ˆ^H™YY]‚ˆÂˆšÔÚY\“[Ù[PÜ™X]R[™›ÈÛZ^È’×ÔÕ•PÕT‘WÕTWÔÒQT—ÓSÑSWÐÔ‘PUWÒS‘“ÈNÂˆÛZK˜ÛÙTÚ^™HHÚ^™[ÙˆÑ˜]ÒY^[ÚY\”ÜŽÂˆÛZKœÛÙHHÑ˜]ÒY^[ÚY\”ÜŽÂˆYˆ
šÐÜ™X]TÚY\“[Ù[J‹O™]šXÙK	œÛZK[‹	”‹O™˜]ÒY[Ù[JHOH’×ÔÕPÐÑTÔÊBˆÂˆËÈ›Ý˜][ˆH™[™\™\ˆÛÜšÜÈÚ]Ý]H[œÝ[Y[[™Ø^Z[™ÈÛÈ\ÂˆËÈ™]\ˆ[ˆ™Y\Ú[™ÈÈÝ\™XØ]\ÙHHXYÛ›ÜÝXÈ˜Z[YÈÛÛ\[K‚ˆ‹O™˜]ÒY[Ù[HH’×Ó•SÒS‘NÂˆœš[ŠÝ\œ‹–Ýš×HH˜]ËRQÚY\ˆ[Ù[H˜Z[YÈÜ™X]H8 %‚ˆÖ—Õ’×ÑU×ÒQÚ[›ÝÛÜšÈ\È[—ˆŠNÂˆBˆËÈÖ—Õ’×Ó•SÔÏLH
\LŠNˆ]™\žH˜[œÛ]Y^[ÚY\ˆ™\XÙYžHBˆËÈË[›Ý[™Èœ˜YÛY[ÝYÙH8 %H™]™\ž][™È]^[ÚY[™Èˆ\›HÙˆHÔBˆËÈXÛÛ\ÜÚ][Ûˆ
ÛÛËÛ[ÜËšÛ
KˆÜ™X]YÛ›HÚ[ˆ\ÚÙY[™H[™BˆËÈ™[ÝÈ\ÈH[™ØYÙ[Y[]šY[˜ÙNÈHXÝ\™H\ÈØ\˜˜YÙHžH\ÚYÛ‹‚ˆYˆ
[“ÛŠÖ—Õ’×Ó•SÔÈŠJBˆÂˆÛZK˜ÛÙTÚ^™HHÚ^™[ÙˆÓ[^[ÚY\”ÜŽÂˆÛZKœÛÙHHÓ[^[ÚY\”ÜŽÂˆYˆ
šÐÜ™X]TÚY\“[Ù[J‹O™]šXÙK	œÛZK[‹	”‹O›[Ó[Ù[JHOH’×ÔÕPÐÑTÔÊBˆÂˆ‹O›[Ó[Ù[HH’×Ó•SÒS‘NÂˆœš[ŠÝ\œ‹–Ýš×HÖ—Õ’×Ó•SÔÎˆH[œ˜YÛY[[Ù[H˜Z[YÈ‚ˆ˜Ü™X]H8 %H\›H\È“Õ[™ØYÙYˆŠNÂˆBˆ[ÙBˆœš[ŠÝ\œ‹–Ýš×HÖ—Õ’×Ó•SÔÏLH8 %U‘T–H^[ÚY\ˆ\ÈH‚ˆ™Ë[›Ý[™Èœ˜YÛY[ÝYÙH\È[ˆ
HÔHYX\Ý\™[Y[‚ˆ˜\›NÈHXÝ\™H\ÈØ\˜˜YÙHžH\ÚYÛŠWˆŠNÂˆBˆB‚ˆËÈH[[ZY\ËˆÛÝÙˆ]™\žHX\\ÈHYš[™Y^HÚ]H^[ÛÈHÚY\ˆ]ˆËÈØ[\\ÈHÛÝH[[YHÛÝ[›Ýš[™XYÈÚ]H˜]\ˆ[ˆ[ˆ[˜›Ý[™ˆËÈ\ØÜš\Üˆ8 %[™Yš[™Y™Z]š[Ý\ˆ]™[ˆÚ[ˆH™\Ý[\È\ØØ\™Y‚ˆ]]ÈXZÙQ[[^HHÉ—J[XYÙIˆ[YËšÒ[XYÙUšY]Õ\H\KZ[Ì—Ý^Y\œËˆZ[Ì—Ý\Z[Ì—ÝÙ][™^
HÂˆYˆ
PÜ™X]R[XYÙJ[YËKK’×Ñ“Ô“PUÔŽÎŽNÕS“Ô“Kˆ’×ÒSPQÑWÕTÐQÑWÕS”Ñ‘T—ÑÕÐ’U’×ÒSPQÑWÕTÐQÑWÔÐSTQÐ’Uˆ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’U\K^Y\œË\
JBˆ™]\›ˆ˜[ÙNÂˆ˜[YR[XYÙJ[YË™[[^HÙ]	]H
	]H^Y\œÊH‹Ù][™^^Y\œÊNÂˆËÈÖ—Õ’×ÐÕP‘WÔÒTÓÓLH8 %XZÙHHÕP‘H[[^HÜ\]YHPQÑS•H[œÝXYÙˆÚ]K‚ˆËÂˆËÈHÔÒUU‘HÓÓ•“Ó[™\È[œÝ[Y[^\ÝÈ™XØ]\ÙHHÚ[™ÙH]\ÝÂˆËÈØ[YH˜XÚÈ[š\ÚX›Kˆ\IÜÈXÝ\™HKÐˆ8 %ÝX™HX\È›Ý[™™\œÝ\ÂˆËÈÖ—Õ’×Ó“×ÐÕP‘H8 %Ø\ÈVSRQS•PÐSÛˆ[œ˜[Y\ÈÚ\™H›Ý\›\ÈYBˆËÈØ[YHØ[Y\˜H[™HØ[YH˜]ÈÙ][™H[ZÙH]\ÈÛÈ™XY[™ÜÈ]ˆËÈ›È[[Ý[ÙˆÛÚÚ[™ÈØ[ˆÙ\\˜]NˆHÝX™HØ[\\ÈÈ›Ý™XXÚHÝ]]]ˆËÈ[Üˆ^H™XXÚ][™\[ˆÈ™H[™\Ý[™ÝZ\ÚX›Hœ›ÛHÚ]K‚ˆËÂˆËÈÚ\ÛÛš[™ÈHSSVH[œÝÙ\œÈ]™XØ]\ÙHH[[^H\ÈÚ]HÛ™[™\™\‰ÜÂˆËÈÝX™H™]Ú\È™XYˆYˆHÚ\ÛÛ™Y[ˆ\ÈÝ[Y[XØ[HÝX™HØ[\H\ÂˆËÈ\ØØ\™YÛÛY]Ú\™HÝÛœÝ™X[H[™HÚÛH][H\ÈZ\Ë\ØÛÜYÈYˆHœ˜[YBˆËÈš[ÈÚ]XYÙ[KH]\È]™H[™H[X›Ý™H\ÈHÝ][Y[X›Ý]ˆËÈHÓÓ•S•ÙˆÜÙHÝX™HX\ËˆØ[YHÚ\H\ÈÖ—Õ’×ÕVÑÕPT‘ÔÒTÓÓ‚ˆËÈ
ÛÝÚHÌ
H8 %HÛÛ\\š\ÛÛˆ]\È™]™\ˆ™\ÜYHÜÚ]]™H›Ý™\È›Ý[™ÂˆËÈžH™\Ü[™ÈH™YØ]]™K‚ˆËÈÖ—Õ’×ÑSSVWÔÒTÓÓLHÚ\ÛÛœÈS“ÕTˆ[[ZY\Ë›Ý\ÝHÝX™HÛ™K‚ˆËÂˆËÈÖ—Õ’×ÐÕP‘WÔÒTÓÓˆ[œÝÙ\œÈ™Ù\ÈHÝX™H™]Ú™XXÚHXÝ\™H‹[™\‚ˆËÈ\ÙY]È›Ý™HHÝX™H[[^H[ÈHÜ›ÝÙˆ]Ø[››Ý[œÝÙ\ˆH]Y\Ý[Û‚ˆËÈHÜ›Ý[™]Ú\ÈÜÙK™XØ]\ÙHHÜ›Ý[™™XYÈ›ÈÝX™KˆH^H[[^H[‚ˆËÈU‘T–HX\\ÈÚ]K[™HÚ]H^[[Y\ÈHY™\ÙH\›HÙˆÌˆ\È^XÝBˆËÈH
NNN
HÜÙH]Ú\È\™H8 %KNMH^[ÈÙˆÛ™H˜[YHÚ]HÝ[™\™ˆËÈ]šX][ÛˆÙˆËYØZ[œÝHÝY\ÝÛX\ˆÛÛÝ\ˆÙˆ“PÒËÛÈ^H\™HÜš][ˆžBˆËÈÛÛY][™È˜]\ˆ[ˆY[Üš][‹ˆHXÛ\™YY™]ÚÙ[œÝ\ÈØ[››ÝÙYH\ÂˆËÈØ\ÙNˆHÚY\ˆ™XY[™ÈH\ØÜš\Üˆ[™^H[[YH™]™\ˆÜ›ÝHÙ]ÈÛÝˆËÈÚ[[K™XØ]\ÙHHÚ\™YXÛÛœÝ[›ØÚÈ\ÈY[\Ù]È™\›È]™\žH˜]Ë‚ˆÝ]XÈÛÛœÝ›ÛÛÝX™TÚ\ÛÛˆH[“ÛŠÖ—Õ’×ÐÕP‘WÔÒTÓÓˆŠNÂˆÝ]XÈÛÛœÝ›ÛÛ[Ú\ÛÛˆH[“ÛŠÖ—Õ’×ÑSSVWÔÒTÓÓˆŠNÂˆÛÛœÝ›ÛÛÚ\ÛÛ™YBˆ[Ú\ÛÛˆ
ÝX™TÚ\ÛÛˆ	‰ˆ\HOH’×ÒSPQÑWÕ’QU×ÕTWÐÕP‘JNÂˆÛÛœÝZ[Ì—ÝÚ]HHÚ\ÛÛ™YÈ‘‘‘Œ‘Hˆ‘‘‘‘‘‘‘NÂˆYˆ
Ú\ÛÛ™Y
Bˆœš[ŠÝ\œ‹ˆ–Ýš×H	\ÎˆHÙ]I]H[[^H\ÈPQÑS•H
‘‘‘Œ‘ŠK	]H^Y\ŠÊWˆ‹ˆ[Ú\ÛÛˆÈÖ—Õ’×ÑSSVWÔÒTÓÓˆˆˆÖ—Õ’×ÐÕP‘WÔÒTÓÓˆ‹Ù][™^ˆ^Y\œÊNÂˆËÈSÒVPÑTË›Ý\ÝHš\œÝˆHÛÜH™[ÝÈ\È[Ø^\ÈYˆËÈ^Y\ÛÝ[H^Y\œØ]Û›H›Ý\ˆž]\ÈÙ\™H]™\ˆÜš][ˆ[ÈHÝYÚ[™ÂˆËÈY™™\ˆ8 %ÛÈ˜XÙ\ÈK‹HÙˆ]™\žH^H[[^HÙ\™Hš[Yœ›ÛHÚ]]™\ˆHÝYÚ[™ÂˆËÈY™™\ˆ\Ý[ˆ]Ø\È[š\ÚX›HÚ[HHÛ›H][K[^Y\ˆ[XYÙHØ\ÈH[[^BˆËÈ›Ø›ÙHÛÝ[ÙYK[™]ÛÝ[]™HXYHHÚ\ÛÛ™Y[ˆ™\ÜH›ÛœÙ[œÙBˆËÈÛÛÝ\ˆÛˆš]™H˜XÙ\ÈÝ]ÙˆÚ^‚ˆ›Üˆ
Z[Ì—ÝˆHÈˆ^Y\œÎÈŠÊÊBˆY[XÜJ‹OœÝYÚ[™Ë›X\Y
Èˆ
ˆ	Ú]K
NÂˆ[’[[YYX]JÉ—JšÐÛÛ[X[™Y™™\ˆØŠHÂˆ˜\œšY\ŠØ‹[YË’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÑÕÓÔSPSˆ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’U
NÂˆšÐY™™\’[XYÙPÛÜHÛÜ^ßNÂˆÛÜKš[XYÙTÝXœ™\ÛÝ\˜ÙHHÈ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’U^Y\œÈNÂˆÛÜKš[XYÙQ^[HÈKK\NÂˆšÐÛYÛÜPY™™\•Ò[XYÙJØ‹‹OœÝYÚ[™Ë˜Y™™\‹[YËš[XYÙKˆ’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÑÕÓÔSPSK	˜ÛÜJNÂˆ˜\œšY\ŠØ‹[YË’×ÒSPQÑWÓVSÕUÔÒQT—Ô‘PQÓÓ“WÓÔSPSˆ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’U
NÂˆJNÂˆšÑ\ØÜš\Ü’[XYÙR[™›ÈZ^ßNÂˆZKš[XYÙUšY]ÈH[YËšY]ÎÂˆZKš[XYÙS^[Ý]H’×ÒSPQÑWÓVSÕUÔÒQT—Ô‘PQÓÓ“WÓÔSPSÂˆšÕÜš]Q\ØÜš\Ü”Ù]ÞÈ’×ÔÕ•PÕT‘WÕTWÕÔ’UWÑTÐÔ’TÔ—ÔÑUNÂˆË™ÝÙ]H‹OœÙ]ÖÜÙ][™^NÂˆË™Ýš[™[™ÈHÂˆË™Ý\œ˜^Q[[Y[HÂˆË™\ØÜš\ÜÛÝ[HNÂˆË™\ØÜš\Ü•\HH’×ÑTÐÔ’TÔ—ÕTWÔÐSTQÒSPQÑNÂˆËœ[XYÙR[™›ÈH	šZNÂˆšÕ\]Q\ØÜš\Ü”Ù]Ê‹O™]šXÙKK	Ë[ŠNÂˆ™]\›ˆYNÂˆNÂˆYˆ
[XZÙQ[[^J‹O™[[^L‘’×ÒSPQÑWÕ’QU×ÕTWÌ‘KK
Hˆ[XZÙQ[[^J‹O™[[^LÑ’×ÒSPQÑWÕ’QU×ÕTWÌÑKKJHˆ[XZÙQ[[^J‹O™[[^PÝX™K’×ÒSPQÑWÕ’QU×ÕTWÐÕP‘K‹KŠHˆ[XZÙQ[[^J‹O™[[^LQ’×ÒSPQÑWÕ’QU×ÕTWÌQKK
JBˆÂˆœš[ŠÝ\œ‹–Ýš×H[[^H^\™HÜ™X][ÛˆRSQˆŠNÂˆ™]\›ˆ˜[ÙNÂˆB‚ˆÂˆšÑ\ØÜš\Ü’[XYÙR[™›ÈÚLžßNÂˆÚL‹œØ[\\ˆH‹O›[™X\”Ø[\\ŽÂˆšÕÜš]Q\ØÜš\Ü”Ù]ÞÈ’×ÔÕ•PÕT‘WÕTWÕÔ’UWÑTÐÔ’TÔ—ÔÑUNÂˆË™ÝÙ]H‹OœÙ]ÖÌ×NÂˆË™Ýš[™[™ÈHÂˆË™Ý\œ˜^Q[[Y[HÂˆË™\ØÜš\ÜÛÝ[HNÂˆË™\ØÜš\Ü•\HH’×ÑTÐÔ’TÔ—ÕTWÔÐSTTŽÂˆËœ[XYÙR[™›ÈH	œÚLŽÂˆšÕ\]Q\ØÜš\Ü”Ù]Ê‹O™]šXÙKK	Ë[ŠNÂˆB‚ˆYˆ
SØYÚY\œÊ
JBˆ™]\›ˆ˜[ÙNÂ‚ˆËÈ‘KUÐT“K\™H[™›Ý]\ŽˆY\ˆHÚY\œÈ^\Ý
HÙ^\È˜[YH[HžH\Ú
BˆËÈ[™™Y›Ü™HHÝY\Ý\È˜]Ûˆ[ž][™ËÛÈ]™\žH\[[™H\ÈÙ\ÜÚ[Ûˆ™YYÈ\ÂˆËÈ[™XYHZ[Ú[ˆHš\œÝœ˜[YH\ÚÜÈ›ÜˆÛ™KˆÙYH™]Ø\›T\[[™\Ë‚ˆ™]Ø\›T\[[™\Ê
NÂˆËÈÛ›H›ÝÈX^HÙ]\[[™HÛÈ\Þ[˜Ú›Û›Ý\È
\N
NˆH›ÛÝØ\›HX›Ý™HÝ^\ÂˆËÈÞ[˜Ú›Û›Ý\È8 %]ØYÚ\™HH^Y\ˆ^XÝÈÈØZ]8 %[™]™\ž][™ÈY\ˆ]ˆËÈ\ÈZY\^KÚ\™HHKLŒ\ÈÜ™X]HÛˆHœ˜[YH™XY\ÈH™\ÜYÝ]\‹‚ˆ\[[™Zš]Ž˜›ÛÝÛ™HHYNÂ‚ˆ‹Oœ™\Ù[^[Ëœ™\Ú^™JÚ^™WÝ
”Ö
‹O\™Ù]ÚY
JH
ˆ”Ê‹O\™Ù]ZYÚ
H
ˆ
NÂˆ×Ý^Ù[œÝ\ÈH[“ÛŠÖ—Õ’×ÕVÐÑS”ÕTÈŠNÂˆ×Ù[PÙ[œÝ\ÈH[“ÛŠÖ—Õ’×ÑSWÐÑS”ÕTÈŠNÂˆËÈH™YH™XY\œÈÙˆH\‹\\ÜÈÛ˜\ÚÝZ[œ]\Ý\ÚÙYÛ˜ÙKˆÙYHBˆËÈXÛ\˜][ÛŽÈÚ]Ý]\ÈH\Ý\ÈXZ[Z[™YžHH[™X\ˆØØ[ˆÛˆ]™\žBˆËÈÛ˜\ÚÝ™]Ú›ÜˆHXYÛ›ÜÝXÈ]\ÈÙ™ˆÛˆ\ÜÙ[X[H]™\žH[‹‚ˆËÈÖ—ÐÐTT‘WÒÑVH\È[ˆH\Ý™XØ]\ÙH]\›\ÈH˜]ÈÙ[œÝ\ÈžH[›Ý\ˆÛÜˆ8 %ˆËÈ[ˆŽH™\ÜÈ[™\ˆ]Üš]\ÈØ\\™K˜Ù[œÝ\ØÚ]Ý]Ö—Õ’×ÑU×ÐÑS”ÕTÈ™Z[™ÈÙ]ˆËÈ[™HØ]H]Z\ÜÙ\ÈÛ™HÙˆ]ÈÝÛˆ[˜[˜Ù\ÈÛÝ[Ú[[H›ÜH
Û˜\
XˆËÈÛÛ[[ˆœ›ÛH^XÝHHØ\\™\È[ˆÜ\˜]ÜˆZÙ\Ë‚ˆ×Ü\ÜÒ[œ]ÕØ[YH[ŠÖ—Õ’×ÔÐ’S‘ŠH[ŠÖ—Õ’×ÑU×ÐÑS”ÕTÈŠHˆ[ŠÖ—Õ’×Ô‘TÓÓ‘WÕPÑHŠH[ŠÖ—ÐÐTT‘WÒÑVHŠNÂˆ×ØÛÜPÙ[œÝ\ÓÛˆH[“ÛŠÖ—Õ’×ÐÓÔWÐÑS”ÕTÈŠNÂˆYˆ
×ØÛÜPÙ[œÝ\ÓÛŠBˆœš[ŠÝ\œ‹–Ýš×HÖ—Õ’×ÐÓÔWÐÑS”ÕTÏLH8 %™\ÛÛ™KXÛÜH›ÙXÙY]œË\Ø[\Y‚ˆ˜Ù[œÝ\ÈT“QQ
\L][HŽÈHXYÛ›ÜÝXÈ\›JWˆŠNÂˆYˆ
ÛÛœÝÚ\ŠˆˆH[ŠÖ—Õ’×ÑSWÑTÐQÔ‘QHŠJBˆÂˆ×Ù[Q\ØYÜ™YHHYNÂˆ×Ù[Q\ØYÜ™YSYH]ÚJŠNÈËÈÝÈX[žHÙ]š[Y\È^H\[‚ˆBˆËÈÕPT‘
È‘USQUHT‘HHQUSTÈÑˆT•ÎˆHØXÚH™]š[Ý\ÛH\ØYYBˆËÈ^\™HÓÑH\ˆ
Y™\ÜË^[›Ü›X]
H[™Ù\™Y]›Ü™]™\‹ÚXÚ\ÈÜ›Û™ÂˆËÈH[ÛY[Ý™X[Z[™È™XÞXÛ\È[ˆY™\ÜÈ8 %[™[ˆÜ\˜]ÜˆÙ\ÜÚ[Ûˆ™XÞXÛ\È[BˆËÈÛÛœÝ[NˆH[šÙ\ˆÛÜ™HH”’PÒÈÐS[™˜[[ÜÝ]™\ž][™È\ÛÜÙHÙX\œÈBˆËÈ˜[™ÛH^\™Hˆ
HÜ\˜]Ü‰ÜÈÛÜ™ÊHHÛ™Ù\ˆHÙ\ÜÚ[Ûˆ˜[‹ˆ\ÍIÜÂˆËÈÝ[HÙˆL“H]Èˆ]\ÝYšYYX]š[™ÈH™\Z\ˆÙ™ˆØ\ÈYX\Ý\™YÛˆBˆËÈÈXY\ÜÈ[ˆ]Û™HØØ][Ûˆ8 %H˜XÝX›Ý]]›Ý]K›ÝX›Ý]^BˆËÈ
HÛÝÚKML˜[Z[JKˆH[Ü\˜]ÜˆÙ\ÜÚ[ÛˆÛˆH™\Z\Žˆ]™\žH›ÜˆËÈÛÜœ™XÝ›È™\ÜYÛÝÙÝÛ‹ˆÖ—Õ’×Ó“×ÕVÔ‘USQUOLH\ÈHØ[YKXš[˜\žBˆËÈÛÛ›Û\›H]œš[™ÜÈH˜[™ÛK]^\™HY™XÝ˜XÚË‚ˆÂˆÝ]XÈÛÛœÝ›ÛÛ›Ô™]˜[Y]HH[“ÛŠÖ—Õ’×Ó“×ÕVÔ‘USQUHŠNÂˆ×Ý^ÝX\™H[›Ô™]˜[Y]H[“ÛŠÖ—Õ’×ÕVÑÕPT‘ŠNÂˆ×Ý^™]˜[Y]HH[›Ô™]˜[Y]NÂˆBˆ×Û›ÑÛÛ[ˆH[“ÛŠÖ—Õ’×Ó“×ÑÓÓS—ÕVŠNÂˆÛÛ[“ØY

NÈËÈ™[ØY[žH^\™Hž]\ÈHš[ÜˆÙ\ÜÚ[ÛˆØ\\™YÈ\ÚÂˆËÈH™K\\MÈØY[˜ÙH8 %™]˜[Y]HÛˆ]™\žH™]Ú[œÝXYÙˆÛ˜ÙHHœ˜[YH\‚ˆËÈØXÚH[žKˆHÛÛ›Û\›K›ÝHš^ÈÙYH]ÈXÛ\˜][Û‹‚ˆ×Ý^ÝX\™]™\žQ™]ÚH[“ÛŠÖ—Õ’×ÕVÑÕPT‘ÑU‘T–WÑ‘UÒŠNÂˆËÈH›]Ü[‹XY™\ÜÙYØXÚH[™]ÈÛÈ\›\ËˆÙYHH›]ØXÚHÛÛ[Y[ˆBˆËÈ˜Z[\™H[ÙH\È\ÈÝX\™[™ÈYØZ[œÝ\ÈHÛÚÝ\]™]\›œÈHÔ“Ó‘È[žKˆËÈÚXÚ˜]ÜÈHÜ›Û™ÈY\Ú[™™\ÜÈ›Ý[™ËÛÈHÛÛ›Û\›H™\ÝÜ™\ÈBˆËÈÝŽ[›Ü™\™YÛX\^XÝH[™H™\šYžH\›H[œÈ›Ý[™ÛÛ\\™\Ë‚ˆ×ØÛÛœÝY[[ÓÙ™ˆH[“ÛŠÖ—Õ’×Ó“×ÐÓÓ”ÕÓQSSÈŠNÂˆ×Ù™]ÚY[[ÐÙ[œÝ\ÈH[“ÛŠÖ—Õ’×Ñ‘UÒÓQSS×ÐÑS”ÕTÈŠNÂˆ×Ü™\ÛÛ™TÜ]Ù[œÝ\ÈH[“ÛŠÖ—Õ’×Ô‘TÓÓ‘WÔÔUÐÑS”ÕTÈŠNÂˆËÈTSS‘PÓÔ‘
\JKˆÓˆ–HQUSÚ[˜ÙH]ÈÝŒÈ
ÛZ[˜[Ü›ÝÙ˜[™ˆËÈLËŒOˆLKŒŒ\Ë8¢$ŒLËŽIHÈ8¢$ŒKŽ\ÎÈ0©Í™ZŠH8 %Ö—Õ’×Ó“×ÔT—Ô‘PÓÔ‘LH\ÈBˆËÈØ[YKXš[˜\žHÛÛ›Û\›K\ˆ\È0©ÌÉÜÈ[Kˆ]™Y\Ù\ÈÛÈÛÛXš[˜][ÛœÂˆËÈÝ]ÝY˜]\ˆ[ˆ[‹Y[™ØYÚ[™Îˆ›ÈÛÜšÙ\œÈYX[œÈH[\ÛÝ[Ø\\™BˆËÈ[™™\^H]™\ž][™È]Ù[ˆ›Üˆ\™HÝ™\šXY[™Ö—Õ’×Ó“×Ñ’U‘T—Ô‘PÓÔ‘	ÜÂˆËÈYX\Ý\™[Y[ÛÝ[™HÚ[[H\ÝÜYžHH]ÚÜÙHÚÛHÚ[\È[Ýš[™ÂˆËÈÜÙHØ[Ëˆ
Ö—Õ’×ÔT—Ô‘PÓÔ‘LH\ÈXØÙ\Y[™™Y[™[Ù\ÛÈHÝŒÉÜÂˆËÈ\›HÜ[[™ÈÝ[YX[œÈÚ]]YX[ŠBˆYˆ
Q[“ÛŠÖ—Õ’×Ó“×ÔT—Ô‘PÓÔ‘ŠJBˆÂˆYˆ
‹O˜ÛÛ\]Xš[]T›Ùš[JBˆœš[ŠÝ\œ‹–Ýš×H\˜[[™XÛÜ™Ñ‘Žˆ[Ø[ˆKŒHÛÛ\]Xš[]H‚ˆ\Ù\ÈHÙ\šX[Û\ÜÚXË\™[™\‹\\ÜÈ™XÛÜ™\—ˆŠNÂˆ[ÙHYˆ
›Ñš]™\”™XÛÜ™

JBˆœš[ŠÝ\œ‹–Ýš×H\˜[[™XÛÜ™Ñ‘ŽˆÖ—Õ’×Ó“×Ñ’U‘T—Ô‘PÓÔ‘\È‚ˆœÙ][™HÛÈ[œÝ[Y[ÈYX\Ý\™HHØ[YHØ[×ˆŠNÂˆ[ÙHYˆ
QÝX\™ÛÛÛÜšÙ\œÊ
JBˆœš[ŠÝ\œ‹–Ýš×H\˜[[™XÛÜ™Ñ‘Žˆ›ÈÛÜšÙ\ˆÛÛ‚ˆŠÖ—ÕÓÔ’ÑT”ÏLÜˆÖ—Õ’×Ó“×ÔTSSÑÕPT‘
H8 %HÙ\šX[‚ˆœ]\ÈHÛÛ›Û\›K›ÝHYÜ˜YY[ÙWˆŠNÂˆ[ÙHYˆ
ÜTÝ]ÓÛŠ
JBˆœš[ŠÝ\œ‹–Ýš×H\˜[[™XÛÜ™Ñ‘ŽˆÖ—Õ’×ÑÔWÔÕUÈ\ÈÙ][™H‚ˆœ\[[™K\Ý]\ÝXÜÈ]Y\žHØ[››ÝÜ[ˆHÛÜšÙ\ˆÚ[šÜÈ‚ˆ¸ %HÙ[œÝ\È[œÈÛˆHÙ\šX[™XÛÜ™\ˆ
Ø[YH˜]ÜË‚ˆœØ[YHÜ™\ŽÈHØ[[YH\ÈH\N™XÛÜ™\‰ÜÊWˆŠNÂˆ[ÙBˆÂˆ‹Oœ\”™XÈHYNÂˆÛÛœÝÚ\ŠˆÜÈH[ŠÖ—Õ’×Ô‘PÓÔ‘ÐÒS’ÈŠNÂˆYˆ
ÜÈ	‰ˆ]ÚJÜÊHˆ
Bˆ‹Oœ\”™XÐÚ[šÈHZ[Ì—Ý
]ÚJÜÊJNÂˆ‹O˜Ø\Y‹œ™\Ù\™J‹Oœ\”™XÐÚ[šÊNÂˆœš[ŠÝ\œ‹ˆ–Ýš×H\˜[[ÛÛ[X[™™XÛÜ™[™ÈÓˆ
Y˜][Ú[˜ÙH\JNˆ‚ˆ˜Ú[šÜÈÙˆ	]H˜]ÜÈÈ	]HÚ\™YÝX\™\ÛÛÛÜšÙ\œË™\ÛÛ™H‚ˆœÙ\šX[Ü™\ˆØ]YˆÖ—Õ’×Ó“×ÔT—Ô‘PÓÔ‘LH\ÈHÛÛ›Û\›NÈ‚ˆÖ—Õ’×Ô‘PÓÔ‘ÐÒS’ÏSˆ[™\Ë—ˆ‹ˆ‹Oœ\”™XÐÚ[šËÝX\™ÛÛÛÜšÙ\œÊ
JNÂˆBˆBˆ[ÙBˆœš[ŠÝ\œ‹–Ýš×HÖ—Õ’×Ó“×ÔT—Ô‘PÓÔ‘LH8 %\˜[[™XÛÜ™Ñ‘ˆ
H‚ˆœ\NÙ\šX[™XÛÜ™\‹Ø[YHš[˜\žJWˆŠNÂˆ×Øš[™[Ù[œÝ\ÈH[“ÛŠÖ—Õ’×Ð’S‘Ô•S—ÐÑS”ÕTÈŠNÂˆ×ÜØÚ\ÜÛÜŒ\H[“ÛŠÖ—Õ’×ÔÐÒTÔÓÔ—ÌTŠNÂˆ×ÝšLHH[“ÛŠÖ—Õ’×Õ’LHŠNÂˆYˆ
×ÝšLJBˆœš[ŠÝ\œ‹–Ýš×HÖ—Õ’×Õ’LOLH8 %]™\žH˜]È\ÜÝY\È][ÜÝ]Èš\œÝš[Z]]™H‚ˆ\È[ˆ
HÔHYX\Ý\™[Y[\›NÈHXÝ\™H\ÈØ\˜˜YÙHžH\ÚYÛŠWˆŠNÂˆYˆ
×ÜØÚ\ÜÛÜŒ\
Bˆœš[ŠÝ\œ‹–Ýš×HÖ—Õ’×ÔÐÒTÔÓÔ—ÌTLH8 %]™\žH˜]ÉÜÈØÚ\ÜÛÜˆ\È^H\È[ˆ‚ˆŠHÔHYX\Ý\™[Y[\›NÈHXÝ\™H\ÈØ\˜˜YÙHžH\ÚYÛŠWˆŠNÂˆ×ÙÝX\™Ù[œÝ\ÈH[“ÛŠÖ—Õ’×ÑÕPT‘ÐÑS”ÕTÈŠNÂˆ×Û›Ðš[™˜]ÚH[“ÛŠÖ—Õ’×Ó“×Ð’S‘ÐUÒŠNÂˆ×Ý™\šYžPš[™˜]ÚH[“ÛŠÖ—Õ’×Õ‘T’Q–WÐ’S‘ÐUÒŠNÂˆ×Ý™\šYžPš[™Ú\ÛÛˆH[“ÛŠÖ—Õ’×Õ‘T’Q–WÐ’S‘ÐUÒÔÒTÓÓˆŠNÂˆYˆ
×Ý™\šYžPš[™Ú\ÛÛŠBˆ×Ý™\šYžPš[™˜]ÚHYNÂˆ×ÜÝ™X[QY\Ù[œÝ\ÈH[“ÛŠÖ—Õ’×ÔÕ‘PSWÑQTÐÑS”ÕTÈŠNÂˆ×Ü\™˜]ÐÙ[œÝ\ÈH[“ÛŠÖ—Õ’×ÔT‘U×ÐÑS”ÕTÈŠNÂˆYˆ
×Ü\™˜]ÐÙ[œÝ\ÊBˆœš[ŠÝ\œ‹ˆ–Ýš×HÖ—Õ’×ÔT‘U×ÐÑS”ÕTÏLH8 %H\‹Y˜]ÈUUUSÓˆÙ[œÝ\È\ÈÓˆ
\‚ˆŒLLH0©ÌË][H‰ÜÈ\ÚËYš\œÝÝ\
Kˆ]ÛÝ[ÈHÚ\™Y\Ý]H™XYÈ[™‚ˆÜš]\ÈÛˆH\‹Y˜]È][™SQTÈHÝËYœ™\]Y[˜ÞH]]][ÛœËˆH‚ˆ‘PQÓ“ÔÕPÈT“Nˆ™]™\ˆ][ÝHHœ˜[YH[YHœ›ÛH\È[‹—ˆŠNÂˆ×Ü™]\ÙPÙ[œÝ\ÈH[“ÛŠÖ—Õ’×Ô‘UTÑWÐÑS”ÕTÈŠNÂˆYˆ
×Ü™]\ÙPÙ[œÝ\ÊBˆÂˆ™]\ÙXÙ[œÝ\ÎŽœ™]‹œ™\Ù\™JMŒÎ
NÂˆ™]\ÙXÙ[œÝ\ÎŽ˜Ý\‹œ™\Ù\™JMŒÎ
NÂˆ×Ü™]\ÙPÛÜYYÙ^\Ëœ™\Ù\™JNLŠNÂˆœš[ŠÝ\œ‹ˆ–Ü™]\ÙWHÖ—Õ’×Ô‘UTÑWÐÑS”ÕTÏLH8 %HÜ›ÜÜËYœ˜[YH˜]ËZY[]HÙ[œÝ\È\È‚ˆ“Óˆ
ÕÈXY‰ÜÈ\ÚËYš\œÝÝ\
KˆHPQÓ“ÔÕPÈT“Nˆ]\Ú\ÈŒKLHÐˆ‚ˆ›Ùˆ™YÚ\Ý\œÈ\ˆ˜]ÈÛˆH[\™XYˆ‘U‘Tˆ][ÝHHœ˜[YH[YHœ›ÛH‚ˆ\È[‹—ˆŠNÂˆBˆ×Ü[]PÙ[œÝ\ÈH[“ÛŠÖ—Õ’×ÔSUWÑVS•ÐÑS”ÕTÈŠNÂˆYˆ
×Ü[]PÙ[œÝ\ÊBˆœš[ŠÝ\œ‹ˆ–Ü[Ù[œÝ\×HÖ—Õ’×ÔSUWÑVS•ÐÑS”ÕTÏLH8 %H›Û™K\[]H›Ý[™YH‚ˆ™Ø]\ˆÙ[œÝ\È\ÈÓˆ
\Ý\
KˆHPQÓ“ÔÕPÈT“NÈ]Ú[™Ù\È›È‚ˆ˜ÛÜH[™›Èœ˜[YH[YHœ›ÛH\È[ˆ\È][ÝX›K—ˆŠNÂˆ×ØÛÛœÝY[[Õ™\šYžHH[“ÛŠÖ—Õ’×Õ‘T’Q–WÐÓÓ”ÕÓQSSÈŠNÂˆ×ØÛÛœÝY[[Õ™\šYžTÚ\ÛÛˆH[“ÛŠÖ—Õ’×Õ‘T’Q–WÐÓÓ”ÕÓQSS×ÔÒTÓÓˆŠNÂˆYˆ
×ØÛÛœÝY[[Õ™\šYžTÚ\ÛÛŠBˆ×ØÛÛœÝY[[Õ™\šYžHHYNÂˆ×Ù›]ØXÚSÙ™ˆH[“ÛŠÖ—Õ’×Ó“×Ñ“UÐÐPÒHŠNÂˆ×Ù›]ØXÚU™\šYžHH[“ÛŠÖ—Õ’×Õ‘T’Q–WÑ“UÐÐPÒHŠNÂˆ×Ù›]ØXÚU™\šYžTÚ\ÛÛˆH[“ÛŠÖ—Õ’×Õ‘T’Q–WÑ“UÐÐPÒWÔÒTÓÓˆŠNÂˆYˆ
×Ù›]ØXÚU™\šYžTÚ\ÛÛŠBˆ×Ù›]ØXÚU™\šYžHHYNÈËÈHÚ\ÛÛˆ\ÈYX[š[™Û\ÜÈÚ]Ý]HÚXÚÂˆËÈ‘KTÒV‘KÛÈHÝX›[™È™]™\ˆ[™È[œÚYHHœ˜[YHH^Y\ˆ\ÈÛÚÚ[™È]ˆXXÚˆËÈšYÝ\™H\ÈHYX\Ý\™YYÚ]Ø]\ˆX\šÈÙˆH[Ý]ÛÜˆ[‹›ÝHÝY\ÜÎˆBˆËÈÜ›ÜÜËYœ˜[YHÝÜ™H™XXÚYLKÍL]™H[šY\ÈÚ]HÝ™X[HÝÜ™H]]ÈLLˆP‚ˆËÈÙZ[[™ËH^\™HØXÚHË‹H\‹Yœ˜[YHÝ™X[HØXÚHŒ‹Œš\œÝ]ÝXÚˆËÈÝ™X[\Ë[™HÚY\ˆX›H\ÈÎH[™š^Y]ØYˆÙÙ]\ˆŒÈPˆÙˆ\œ˜^\ÂˆËÈ[ØØ]YÛ˜ÙKYØZ[œÝH™[™\™\ˆ][™XYHÛÈHLLˆPˆÝ™X[HÝÜ™H8 %[™ˆËÈ]ZÙ\ÈH[‰ÜÈÜ›ÝÈš[œ›ÛHÌKH\È[ˆŒÜ›ÝÜÈÈ™\›Ë‚ˆ‹Oœ\œÚ\ÝØXÚK”™\Ù\™JN
NÂˆ‹O^\™\Ë”™\Ù\™JNLŠNÂˆ‹OœÝ™X[PØXÚK”™\Ù\™JNLŠNÂˆ‹OœÚY\œË”™\Ù\™JL
NÂˆ×Ý^ÛÝ\˜Ù\Ë”™\Ù\™JNLŠNÂˆ×Ý^ÝX\™YœË”™\Ù\™JNLŠNÂˆYˆ
ÛÛœÝÚ\ŠˆˆH[ŠÖ—Õ’×ÕVÑÕPT‘Ð–UTÈŠJBˆÂˆËÈÛ[\YÈH][\HÙˆÑÝX\™›ØÚÜÈ[™È]X\ÝÛ™H›ØÚÎˆ™[ÝÈ]ˆËÈ›Ý[™ÈÑÝX\™›ØÚÜØ\È™\›È[™HØ[\Y]›ÛÈ›Ý[™È][ˆËÈÚXÚÛÝ[™XY\ÈHÝX\™]™]™\ˆš\™\È˜]\ˆ[ˆ\ÈH˜YÙ][™Ë‚ˆÛÛœÝÚ^™WÝØ[HÚ^™WÝ
ÝÝ[
‹[‹
JNÂˆ×Ý^ÝX\™ž]\ÈHÝŽ›X^Ú^™WÝŠÑÝX\™›ØÚÜÈ
ˆØ[	ˆœÚ^™WÝ
ÑÝX\™›ØÚÜÈHJJNÂˆœš[ŠÝ\œ‹ˆ–Ýš×HÖ—Õ’×ÕVÑÕPT‘Ð–UTÏI^H8 %H^\™HÛÛ[ÝX\™\È^XÝÈ‚ˆ]X[žHž]\È[™Ø[\\ÈÜ™XY›ØÚÜÈX›Ý™H]
Y˜][MŒÎ
Wˆ‹ˆ×Ý^ÝX\™ž]\ÊNÂˆBˆYˆ
×Ý^ÝX\™]™\žQ™]Ú
Bˆœš[ŠÝ\œ‹ˆ–Ýš×HÖ—Õ’×ÕVÑÕPT‘ÑU‘T–WÑ‘UÒ8 %H^\™HÛÛ[ÝX\™[œÈÛˆ‚ˆ‘U‘T–H™]Ú›ÝÛ˜ÙHHœ˜[YH\ˆ[žH
H™K\\MÈØY[˜ÙJWˆŠNÂˆ×Ý^ÝX\™Ú\ÛÛˆH[“ÛŠÖ—Õ’×ÕVÑÕPT‘ÔÒTÓÓˆŠNÂˆYˆ
×Ý^ÝX\™Ú\ÛÛŠBˆœš[ŠÝ\œ‹–Ýš×H^\™HÝX\™ÒTÓÓ‘Q8 %HÚ[™ÙYÚ\™H]\Ý›ÝÈ™XY‚ˆŒLŒ	INÈ[ž][™È[ÙHYX[œÈHÙ[œÝ\ÈØ[››Ýš\™WˆŠNÂˆYˆ
×Ý^™]˜[Y]JBˆœš[ŠÝ\œ‹–Ýš×H^\™HØXÚH‘USQUTÈÛˆÛÛ[ˆHØXÚH]ÚÜÙH‚ˆ™ÝY\Ýž]\ÈÚ[™ÙY\È™K]\ØYY[ˆXÙWˆŠNÂˆYˆ
[“ÛŠÖ—Õ’×ÕVÔ‘Q”‘TÒÐSŠJBˆœš[ŠÝ\œ‹–Ýš×HU‘T–H^\™H\È™K\™XYÛˆU‘T–H™]Ú‚ˆŠÖ—Õ’×ÕVÔ‘Q”‘TÒÐS
H8 %HXÝ\™H\›KZ[›Ý\ÛHÛÝË[™‚ˆHØXÚHØ[››ÝÙ\™HHÝ[H[XYÙH[™\ˆ]ˆŠNÂˆ×Ü›Ùš[SÛˆH[“ÛŠÖ—Õ’×Ô“Ñ’SHŠNÂˆYˆ
×Ü›Ùš[SÛŠBˆÂˆØ[Xœ˜]T›Ù“›ÝÊ
NÂˆ×Ù^˜TØÛÜ\ÈH[ŠÖ—Õ’×Ô“Ñ’SWÑVWÔÐÓÔTÈŠBˆÈZ[Ì—Ý
]ÚJ[ŠÖ—Õ’×Ô“Ñ’SWÑVWÔÐÓÔTÈŠJJHˆÂˆYˆ
×Ù^˜TØÛÜ\ÊBˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÖ—Õ’×Ô“Ñ’SWÑVWÔÐÓÔTÏI]H8 %HÜÚ]]™HÛÛ›Û›Üˆ‚ˆH[œÝ[Y[[™NÈÝ\˜	ÜÈ™\ÚYX[]\Ýš\ÙHžHX›Ý]	KŒˆ‚ˆ›œËÙ˜]È[™›È˜[YY\ÙHX^H[Ý™Wˆ‹ˆ×Ù^˜TØÛÜ\ËÝX›J×Ù^˜TØÛÜ\ÊH
ˆÝX›J×Ü›Ù“›ÝÓœÌL
HÈLŒ
NÂˆBˆËÈ™\ÜY›ÝYÚH›Ùš[HÚ[™ÝËÛÈ]™YYÈH›Ùš[HÛˆÈØ^H[ž][™Ë‚ˆËÈØ^Z[™ÈÛÈÝ]ÝY˜]\ˆ[ˆÚ[[HÛÝ[[™È[ÈH™\Ü›Ø›ÙHš[Ë‚ˆ×ÜÝ™X[PÙ[œÝ\ÈBˆ[ŠÖ—Õ’×ÔÕ‘PSWÐÑS”ÕTÈŠHÈ]ÚJ[ŠÖ—Õ’×ÔÕ‘PSWÐÑS”ÕTÈŠJHˆÂˆYˆ
×ÜÝ™X[PÙ[œÝ\È	‰ˆY×Ü›Ùš[SÛŠBˆÂˆœš[ŠÝ\œ‹–Ýš×HÖ—Õ’×ÔÕ‘PSWÐÑS”ÕTÈ™YYÈÖ—Õ’×Ô“Ñ’SH8 %]™\ÜÈ‚ˆ›ÝYÚ]Ú[™ÝÎÈÙ[œÝ\ÈÑ‘—ˆŠNÂˆ×ÜÝ™X[PÙ[œÝ\ÈHÂˆBˆ×ÜÝ™X[TÚ\ÛÛˆH[“ÛŠÖ—Õ’×ÔÕ‘PSWÐÑS”ÕT×ÔÒTÓÓˆŠNÂˆYˆ
×ÜÝ™X[TÚ\ÛÛŠBˆœš[ŠÝ\œ‹–Ýš×HÝ™X[HÙ[œÝ\ÈÒTÓÓ‘Q8 %HÛÛ[ÚXÚÈ]\Ý›ÝÈ™XY‚ˆŒŒ	INÈ[ž][™È[ÙHYX[œÈ]Ø[››Ý˜Z[ˆŠNÂˆ×ØXÝ]™HHYNÂˆœš[ŠÝ\œ‹–Ýš×H™[™\™\ˆTˆ	]^	]H\™Ù]	^HÚY\œ×ˆ‹‹O\™Ù]ÚYˆ‹O\™Ù]ZYÚ‹OœÚY\œÓX\œÚ^™J
JNÂˆYˆ
™\ÔØØ[J
HOHJBˆœš[ŠÝ\œ‹ˆ–Ýš×H[\›˜[™\ÛÛ][Ûˆ	]^	]H
	]^H]IÜÈÝÛˆLŽÌŒ
NˆH‚ˆ™ÝY\Ý	ÜÈÙ[ÛY]žH\È[˜Ú[™ÙY[™H˜\Ý\š\Ø][Ûˆ\™Ù]\È›ÝˆH‚ˆœ™\Ù[™XY˜XÚÈ\È	]^Hž]\È8 %	KŒYˆP‹Ùœ˜[YH8 %ÛÈ™XY™XY˜XÚØ‚ˆš[ˆÖ—Õ’×Ô“Ñ’SH™Y›Ü™H][Ý[™ÈHœ˜[YH[YK—ˆ‹ˆ”Ö
‹O\™Ù]ÚY
K”Ê‹O\™Ù]ZYÚ
K™\ÔØØ[J
Kˆ™\ÔØØ[J
H
ˆ™\ÔØØ[J
KˆÝX›J”Ö
‹O\™Ù]ÚY
JH
ˆ”Ê‹O\™Ù]ZYÚ
H
ˆŒÈLMÍ‹Œ
NÂˆYˆ
×Ü›Ùš[SÛŠBˆœš[ŠÝ\œ‹–ÝšÜ›Ù—Hœ˜[YHÔH›Ùš[HÓ—ˆŠNÂˆËÈKŒ‰ÜÈØ[\YÚÛKY[˜Ý[Ûˆ[Y\œÈšYHÚ]H›Ùš[\ŽÈHÛÛ›Û\›BˆËÈ\›œÈ[HÙ™ˆ[œÚYHH›Ùš[Y[ˆÛÈZ\ˆÝÛˆš[Ø[ˆ™HYX\Ý\™Y‚ˆÚYˆÖ—ÕÒÓQ•SÂˆ×ÝÚÛQ[˜ÈHQ[“ÛŠÖ—Õ’×Ó“×ÕÒÓQ•SÈŠNÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÒÓKQ•SÕSÓˆ[Y\œÈÓÓTSQSˆ
QÖ—ÕÒÓQ•SÏLJKˆTÈ‚ˆ•RSTÈŒH\ËÙœ˜[YHÓÕÑTˆSˆHQUSÓ‘HUHÔ“ÕÑ‚ˆ›YX\Ý\™Y
\LL0©Í‹Ž
H8 %™XY]ÈÒT‘TË™]™\ˆ]ÈZ[\ÙXÛÛ™Ë‚ˆ˜[™™]™\ˆ][ÝHHœ˜[YH[YHœ›ÛH]—ˆŠNÂˆÙ[™Y‚ˆ™]\›ˆYNÂŸB‚ŸHËÈ˜[Y\ÜXÙB‚˜›ÛÛšÔ™[™\™\—Ò[š]

BžÂˆYˆ
×Ú[š]šYY
Bˆ™]\›ˆ×ØXÝ]™H	‰ˆY×ÙÙ[ÙNÂˆ×Ú[š]šYYHYNÂ‚ˆYˆ
Q[“ÛŠÖ—Õ’ÑUÈŠJBˆÂˆœš[ŠÝ\œ‹–Ýš×H™[™\™\ˆÑ‘ˆ
Ù]Ö—Õ’ÑUÏLHÈ[˜X›H]
WˆŠNÂˆ™]\›ˆ˜[ÙNÂˆBˆËÈ[š]ÛÛ[[Ûˆ˜[Y\È]ÈÝÛˆ˜Z[\™HÛˆ]™\žH]‚ˆ™]\›ˆ[š]ÛÛ[[ÛŠ
NÂŸB‚›ÚYšÔ™[™\™\—Ñ˜]ÊZ[Ý
ˆ˜\ÙKÛÛœÝM˜]Éˆ˜]ÊBžÂˆYˆ
Y×ØXÝ]™H×ÙÙ[ÙJBˆ™]\›ŽÂˆËÈH™[™\™\‰ÜÈÝÛˆÛÝ[Ùˆ˜]ÜÈ]Ø\ÈS‘Q™^ÈH\‹\š[Z]]™BˆËÈÙ[œÝ\ÈÙˆ˜]ÜÈ]XØÙ\YˆHÛÛ[X[™›ØÙ\ÜÛÜ‰ÜÈš[™Îˆ‹‹ˆ˜]ÜÏXÛÝ[\‚ˆËÈ[™H™[™\™\‰ÜÈš[HÛÝ[\œÈ\ØYÜ™YYžH[ˆ[™\™HØ\È›È[X™\ˆ[‚ˆËÈ™]ÙY[ˆÈØ^HÚ\™HHY™™\™[˜ÙH]™Y8 %HÚZ[ˆ\ÈÈ™HÛÝ[Y[šÈžBˆËÈ[šÈ
ÛÝÚHMŒŠK[˜ÛY[™ÈH[šÈ™]ÙY[ˆÛÈ[Ù[\Ë‚ˆÓÕS•
™˜]Îˆ[™YÈH™[™\™\ˆŠNÂˆÛÛœÝZ[Ì—Ý
ˆ™YÜÈHMÔ™YÚ\Ý\œÊ
NÂˆËÈH™\ÛÛ™H\ØÜš[Z[˜]Ü‹[™HÛ›HÛ™Nˆ—ÓSÑPÓÓ•“Ó	ÜÈY˜[WÛ[ÙK‚ˆYˆ

™YÜÖÌŒŒH	ˆÊHOHŠBˆÂˆÔ™\ÛÛ™J˜\ÙK™YÜÊNÂˆ™]\›ŽÂˆBˆÑ˜]Ê˜\ÙK˜]Ë™YÜËMÐ›Ý[™ÚY\Š
KMÐ›Ý[™ÚY\ŠJJNÂŸB‚‹ËÈ\LMÎˆHØ[YH˜]Ëœ›ÛHÞ‹Y˜]ØÚ]H™YÚ\Ý\ˆš[H™\^YY[™B‹ËÈš[™[™ÜÈHØ[ÈØ\\™Y]HXÚÙ]
ÜKÜ[\ÜÜ]š
K‚›ÚYšÔ™[™\™\—Ñ˜]Ô]Y]YY
Z[Ý
ˆ˜\ÙKÛÛœÝM˜]Éˆ˜]ËÛÛœÝZ[Ì—Ý
ˆ™YÜËˆÛÛœÝMÚY\š[™[™ÉˆœËÛÛœÝMÚY\š[™[™ÉˆÊBžÂˆYˆ
Y×ØXÝ]™H×ÙÙ[ÙJBˆ™]\›ŽÂˆÓÕS•
™˜]Îˆ[™YÈH™[™\™\ˆŠNÂˆYˆ

™YÜÖÌŒŒH	ˆÊHOHŠBˆÂˆÔ™\ÛÛ™J˜\ÙK™YÜÊNÂˆ™]\›ŽÂˆBˆÑ˜]Ê˜\ÙK˜]Ë™YÜËœËÊNÂŸB‚›˜[Y\ÜXÙHÂ‚‹ËÈHÚ\™YÝØ\›ÙH8 %]™\ž][™Èœ›ÛHœ™XÛÜ™Hœ›ÛY™™\ˆˆÈHœ˜[YB‹ËÈÝ]È[™KˆHM™YYØ[È]œ›ÛHHWÔÕÐTXÚÙ]HÑ™YYœ›ÛHB‹ËÈÝØ\ÛÚÎÈHÛÈØ[\œÈØ]HÛˆ×ÙÙ[ÙHÛÈ^XÝHÛ™H\È]™H\ˆ[‹‚›ÚYÔÝØ\[\
Z[Ý
ˆ˜\ÙKZ[Ì—Ýœ›ÛY™™\‹Z[Ì—ÝÚYZ[Ì—ÝZYÚ
BžÂˆ
›ÚY
X˜\ÙNÂˆËÈ™XÛÜ™Y‘Q“Ô‘HHX\›H™]\›œÎˆH™\ÛÛ™H]›ÙXÙ\ÈHœ˜[YH\[œÂˆËÈ™Y›Ü™HHÝØ\][››Ý[˜Ù\È]ÛÈÛˆœ˜[YHˆHÛÛ\\š\ÛÛˆ[ˆÔ™\ÛÛ™H\ÂˆËÈXYHYØZ[œÝHY™\ÜÈœ˜[YH‹LHX›\ÚYˆ]\Èš[™H™XØ]\ÙHHY™\ÜÂˆËÈÙ\È›ÝÚ[™ÙK[™]\ÈH™X\ÛÛˆHš\œÝœ˜[YH\È›ÈÛ˜\ÚÝ˜]\‚ˆËÈ[ˆHÜ›Û™ÈÛ™K‚ˆ‹O™œ›ÛY™™\ˆHœ›ÛY™™\ŽÂˆ
ÊÔ‹O™œ˜[YNÂ‚ˆËÈ\Nˆ[™[žH˜XÚÙÜ›Ý[™XZ[\[[™\È]™[ˆÚ[ˆ›È˜]È\ÈÝ\œ™[BˆËÈZ\ÜÚ[™È[KÛÈZ\ˆÙ^\È™XXÚH\š[ÙXÈØ]™H[™H™]\›š[™ÈX]\šX[ˆËÈ]ÈHX\[œÝXYÙˆH[™[™ÈÙ]ˆÛ™H]ÛZXÈ™XYÚ[ˆ\™H\È›Ý[™Ë‚ˆYˆ
\[[™Zš]Ž\Þ[˜ÓÛŠ
JBˆ\[[™Zš]Ž‘˜Z[Š
NÂ‚ˆËÈUSHKŒNˆÝ\HÛÜšÙ\œÈÛˆHœ˜[YH]\È™YÚ[›š[™Ë\™H[™›Ý[‚ˆËÈ™YÚ[‘œ˜[YXÚXÚÙ\È›Ý[ˆ[[Hš\œÝ˜]Ëˆ]™\ž][™È™]ÙY[ˆ\ÂˆËÈ[™H[™]š\œÝ˜]È8 %H™XY˜XÚËH™\Ù[Hœ˜[YK\Ý]ÈØ[ËBˆËÈXÚÙ]È™Y›Ü™HHš\œÝU×ÒS‘8 %\ÈXYÝ\HÛÛÙ]È›Üˆœ™YK[™]ˆËÈ\ÈHY™™\™[˜ÙH™]ÙY[ˆHX\›H˜]ÜÈ™Z[™ÈÙ\™Y[™\Ú[™È[›[™K‚ˆÝX\™ÛÛ\Ü]Ú

NÂ‚ˆËÈÖ—Ñ”×ÓÑÏSˆ8 %Hœ˜[YH˜]K]™\žHˆÙXÛÛ™Ë[™“ÕS‘ÈSÑK‚ˆËÂˆËÈ]^\ÝÈ™XØ]\ÙH]™\žH[œÝ[Y[\È›Ú™XÝÝÛœÈ]™\ÜÈHœ˜[YH˜]H\ÂˆËÈHš[šYÈ[›ÝYÚÈÚ[™ÙHH[œÝÙ\ŽˆÖ—Õ’×Ô“Ñ’SXÛÜÝÈ‹M\ÈHœ˜[YH[™ˆËÈÖ—Õ’×Ñ”SQWÔÕUØØ[ÜÈ[LŒKŒ^[È›Üˆ[›Ý\ˆKŽKLËŒÈ
ÛÝÚHÌÍÊKˆÛÂˆËÈš\Ý^H][™[YHÝÈ]™Y[Èˆ\È[Ø^\È™Y[ˆHÛ›H[š[œÝ[Y[YˆËÈÛÛ™šYÝ\˜][Û‹[™]›ÙXÙ\È›È[X™\ˆ][8 %ÚXÚXZÙ\ÈHÙ\ÜÚ[Ûˆ]ˆËÈ™\ÜÈš]˜\™\ÈÙ[ˆ[™˜[ÚYšXX›KHÛ™H[™È\È›Ú™XÝÙ\È›ÝXØÙ\‚ˆËÂˆËÈ\È\ÈÛ™HÛÝ[\ˆ[™Û™HÛØÚÈ™XY\ˆ‘TÑS•Qœ˜[YH8 %ŒŒœÈYØZ[œÝBˆËÈLËLŒ\Èœ˜[YKK™KˆÛ™H\[ˆHZ[[Û‹YØZ[œÝH›Ùš[\‰ÜÈÝ\Ø[™ÈÙ‚ˆËÈÛØÚÈ™XYËˆ]\ÈHÚX\\Ý[™È\™H]Ø[ˆÝ[™HÜ›Û™ËÛÈ]\ÈÙ™‚ˆËÈžHY˜][ZÙH]™\ž][™È[ÙK‚ˆËÂˆËÈ]™\ÜÈHQQPSˆ\ÈÙ[\ÈHYX[‹™XØ]\ÙHÛˆ\È]HHYX[ˆYX\Ý\™\ÂˆËÈHXÚ[™È›ÛÜˆ˜]\ˆ[ˆHÚ[™ÙH
ÛÝÚHŒÍÊH8 %[™H[\˜[	ÜÈÝÛ‚ˆËÈœ˜[YHÛÝ[ÛÈHÚ[™ÝÈ]ÛÝ™\œÈHØYØÜ™Y[ˆ\Èš\ÚX›H\ÈÝXÚ˜]\ˆ[‚ˆËÈ]™\˜YÙY[‹‚ˆËÂˆËÈS‘U‘TÔ•ÈHUÈÓÕS•ÚXÚHÜ\˜]Üˆ\ÚÙY›Üˆ[™ÚXÚ\ÈÚ]XZÙ\ÂˆËÈ\È[™H\ØX›H›ÜˆHÛÛ\\š\ÛÛˆ][ˆZ\ˆØš™XÝ[Û‹[ˆZ\ˆÛÜ™Îˆ]]ÐÚXÚÂˆËÈš\Û‰ÝH™Y]\›Z[™Y›Ý]H[™›ÛXšYHÜ]ÛœÈ\™H›Ý[Ø^\ÈHØ[YKÛÈ]ÛÛ‰ÝˆËÈ™HL	HXØÝ\˜]H\ÜXÚX[HYˆ[ˆH[ˆ]Ý^\È[ˆHZ[]\žH›Û™H[™Û™HÛÈÛ‚ˆËÈHXZ[ˆÝ™Y]ˆˆ]\È^XÝHšYÚ[™]\Y\ÈÈH[X[‹Yš]™[ˆÙ\ÜÚ[Û‚ˆËÈÛËˆÚ]Ý]H˜]ÈÛÝ[HÙœ×XÚ[™ÝÈ\ÈHœ˜[YH˜]HÚ]›ÈÝ][Y[Ù‚ˆËÈÚ]Ø\È™Z[™È˜]Û‹ÛÈÛÈÚ[™ÝÜÈØ[››Ý™HX]ÚY[™HY™™\™[˜ÙH™]ÙY[‚ˆËÈH\›\È[™HY™™\™[˜ÙH™]ÙY[ˆÛÈPÑTÈ\™HHØ[YH[X™\‹‚ˆËÂˆËÈHZ[ˆ[™X^ÛÈÝ]™\ÚYHHYYX[ˆ™XØ]\ÙHHÚ[™ÝÈ]ÕQTÈÛÈXÙ\ÂˆËÈ\ÈHØ\ÙH]\ÈÈ™Hš\ÚX›NˆHYYX[ˆÙˆËZ[œ›ÛHL[™‹\ÂˆËÈ›ÝHXÙH][[™H™XY\ˆÚÈÛ›HØ]ÈHYYX[ˆÛÝ[X]Ú]YØZ[œÝBˆËÈÙ[Z[™HËY˜]ÈÚ[™ÝÈ[ˆHÝ\ˆ\›Kˆ]ÛÜÝÈÛ™HÛÝ[\ˆ™XYHœ˜[YK‚ˆÂˆÝ]XÈÛÛœÝ[œÓÙÔÙXÈH[ŠÖ—Ñ”×ÓÑÈŠHÈ]ÚJ[ŠÖ—Ñ”×ÓÑÈŠJHˆÂ‚ˆËÈÖ—Õ’×Ñ”SQWÕPÑHU‘TÈS”ÒQHTÈ“ÐÒÈS‘UTÕ“ÕTS‘ÓˆÖ—Ñ”×ÓÑË‚ˆËÂˆËÈ]\ÙYËˆØÜËÚ[œÝ[Y[Ë›YØÝ[Y[ÈH˜XÙH\ÈHÝ[™[Û™BˆËÈ[œÝ[Y[[™\›Z[™È][Û™H›ÙXÙY[ˆ[\Hš[K›È›ÝÜË[™›Ý]™[‚ˆËÈ]ÈÝÛˆÐS““ÕÔ’UHˆXYÛ›ÜÝXÈ8 %™XØ]\ÙH]Y\ÜØYÙH\È[œÚYHHØ[YBˆËÈXY›ØÚËˆ[ˆÜ\˜]Üˆ^YY›Üˆ[ˆÝ\ˆÈØ\\™HHÝ]\ˆ[™Hš[BˆËÈØ\È™]™\ˆÜ[™Y‚ˆËÂˆËÈ\È\ÈÛÝÚHN	ÜÈ^XÝÚ\KÚXÚHÛÛ[Y[›ÜH[™\È™[ÝÈ˜[Y\Î‚ˆËÈHÛÝ[\ˆ]Ø\ÈØ]Y™Z[™[ˆ^[œÚ]™H[œÝ[Y[[™\™Y›Ü™H™]™\‚ˆËÈÛˆÚ[ˆH[™È]YX\Ý\™\È\[™Y‹ˆ]Ø\ÈÜš][ˆX›Ý]H\[[™BˆËÈ[Y\ˆ[™\YY\™H[››ÝXÙY‚ˆÝ]XÈÛÛœÝ›ÛÛ˜XÙP\›YYH[ŠÖ—Õ’×Ñ”SQWÕPÑHŠH	‰‚ˆ
‘[ŠÖ—Õ’×Ñ”SQWÕPÑHŠHOH	×	ÎÂˆYˆ
œÓÙÔÙXÈˆ˜XÙP\›YY
BˆÂˆ\Ú[™ÈÛÈHÝŽ˜Ú›Û›ÎŽœÝXYWØÛØÚÎÂˆÝ]XÈÛÎŽ[YWÜÚ[Ú[™ÝÔÝ\HÛÎŽ››ÝÊ
NÂˆÝ]XÈÛÎŽ[YWÜÚ[\Ýœ˜[YHHÚ[™ÝÔÝ\ÂˆÝ]XÈZ[Ýœ˜[Y\ÈHÂˆÝ]XÈÝŽ™XÝÜZ[Ì—Ýˆœ˜[YU\ÎÂˆÝ]XÈÝŽ™XÝÜZ[Ì—Ýˆœ˜[YQ˜]ÜÎÂˆÛÛœÝÛÎŽ[YWÜÚ[›ÝÈHÛÎŽ››ÝÊ
NÂˆ
ÊÙœ˜[Y\ÎÂˆœ˜[YQ˜]ÜËœ\ÚØ˜XÚÊZ[Ì—Ý
‹O™˜]ÜÕ\Ñœ˜[YJJNÂˆœ˜[YU\Ëœ\ÚØ˜XÚÊZ[Ì—Ý
ˆÝŽ˜Ú›Û›ÎŽ™\˜][Û—ØØ\ÝÝŽ˜Ú›Û›ÎŽ›ZXÜ›ÜÙXÛÛ™ÏŠ›ÝÈH\Ýœ˜[YJBˆ˜ÛÝ[

JJNÂˆ\Ýœ˜[YHH›ÝÎÂˆËÈÔSˆUSHÉÜÈX›KˆY™™\™[˜ÙYYØZ[œÝH™]š[Ý\Èœ˜[YKÛÈXXÚ›ÝÂˆËÈØ^\ÈÚ]\[™YS”ÒQH]œ˜[YH˜]\ˆ[ˆ\È]‚ˆÂˆÝ]XÈZ[Ý™]•^H™]”\\ÈH™]•^ž]\ÈHˆ™]•^œÈH™]•Ø[ÈH™]”ÛY\Hˆ™]‘™[˜ÙHH™]•^XÈH™]•^›Ü•˜XÙHHˆ™]•^ž]\Ñ›Ü•˜XÙHH™]”\\Ñ›Ü•˜XÙHHˆ™]•^œÑ›Ü•˜XÙHHÂˆËÈH”SQIÔÈPÓÓTÔÒUSÓ‹ˆØ[ÓœØÛ›HXØÝ[][]\ÈÚ[ˆHØ[ÂˆËÈ‘UT“”È[™\È™\Ù[\È\[š[™ÈS”ÒQHÛ™KÛÈH[‹\›ÙÜ™\ÜÂˆËÈÜ[Ûˆ\ÈYY^XÚ]H8 %Ú]Ý]]HÌ\È]Ú\ÈÚ\™ÙYÂˆËÈHœ˜[YHQ•TˆHÛ™H]ÝY™™\™Y]ÚXÚ\ÈHÚ[™ÛHØ\ÙH\ÂˆËÈYX\Ý\™[Y[^\ÝÈ›Ü‹‚ˆÛÛœÝ[\Ý]ÈÈH[\Ý]×Ô™XY

NÂˆÛÛœÝZ[Ý›ÝÓœÈHZ[Ý
ˆÝŽ˜Ú›Û›ÎŽ™\˜][Û—ØØ\ÝÝŽ˜Ú›Û›ÎŽ›˜[›ÜÙXÛÛ™ÏŠˆ›ÝË[YWÜÚ[˜ÙWÙ\ØÚ

JK˜ÛÝ[

JNÂˆÛÛœÝZ[Ý[‘›YÚBˆËØ[ÔÝ\œÈ	‰ˆ›ÝÓœÈˆËØ[ÔÝ\œÈÈ›ÝÓœÈHËØ[ÔÝ\œÈˆÂˆÛÛœÝZ[ÝØ[Ó›ÝÈHËØ[ÓœÈ
È[‘›YÚÂˆÛÛœÝZ[Ì—ÝØ[Õ\ÈHZ[Ì—Ý

Ø[Ó›ÝÈH™]•Ø[ÊHÈL
NÂˆÛÛœÝZ[Ì—ÝÛY\\ÈHZ[Ì—Ý

ËœÛY\œÈH™]”ÛY\
HÈL
NÂˆÛÝÑœ˜[YS›ÝJ‹O™œ˜[YKœ˜[YU\Ë˜˜XÚÊ
KZ[Ì—Ý
‹O™˜]ÜÕ\Ñœ˜[YJKˆZ[Ì—Ý
×Ý^™X[\ØYÈH™]•^
KˆZ[Ì—Ý
×Ü\PÛÝ[H™]”\\ÊKˆZ[Ì—Ý

×Ý^\ØYž]\ÈH™]•^ž]\ÊHÈL
KˆZ[Ì—Ý

×Ý^\ØYœÈH™]•^œÊHÈL
KˆØ[Õ\ËÛY\\Ëˆ[Ì—Ý
œ˜[YU\Ë˜˜XÚÊ
JHH[Ì—Ý
Ø[Õ\ÊHBˆ[Ì—Ý
ÛY\\ÊKˆZ[Ì—Ý

×Ù™[˜ÙUØZ]œÈH™]‘™[˜ÙJHÈL
KˆZ[Ì—Ý
×ÙÜSœÓÙ‘œ˜[YHÈL
KˆZ[Ì—Ý

×Ý^XÛÙSœÈH™]•^XÊHÈL
JNÂˆ™]•^H×Ý^™X[\ØYÎÂˆ™]”\\ÈH×Ü\PÛÝ[Âˆ™]•^ž]\ÈH×Ý^\ØYž]\ÎÂˆ™]•^œÈH×Ý^\ØYœÎÂˆ™]•Ø[ÈHØ[Ó›ÝÎÂˆ™]”ÛY\HËœÛY\œÎÂˆÛÛœÝZ[Ý™[˜ÙQ[HH×Ù™[˜ÙUØZ]œÈH™]‘™[˜ÙNÂˆ™]‘™[˜ÙHH×Ù™[˜ÙUØZ]œÎÂˆÛÛœÝZ[Ý^XÑ[HH×Ý^XÛÙSœÈH™]•^XÎÂˆ™]•^XÈH×Ý^XÛÙSœÎÂˆËÈÖ—Õ’×Ñ”SQWÕPÑOOš[Oˆ8 %Û™H[™H\ˆ™\Ù[Yœ˜[YKÛÈHÝ]\ˆØ[‚ˆËÈ™H›Ý[™[™™XYÑ‘“S‘H[œÝXYÙˆÜ[™È][™È[ˆHÙ[™K\›ÝÂˆËÈX›KˆÛ™Hœš[ˆHœ˜[YNÈHÛÛ[[œÈ\™HHXÛÛ\ÜÚ][ÛˆX›Ý™K‚ˆÝ]XÈ’SJˆ˜XÙHH×H

HOˆ’SJˆÂˆÛÛœÝÚ\ŠˆˆH[ŠÖ—Õ’×Ñ”SQWÕPÑHŠNÂˆYˆ
YˆJ™ŠBˆ™]\›ˆ[ŽÂˆ’SJˆH›Ü[Š‹ÈŠNÂˆËÈS““ÕSÑHUˆ[ˆ[œÝ[Y[]Ø[ˆ™H\›YY[™Ú[[H™XÛÜ™ˆËÈ›Ý[™ÈÛÜÝÈÚÙ]™\ˆ\›YY]H[\™HÙ\ÜÚ[Ûˆ™Y›Ü™H^Hš[™ˆËÈÝ]8 %ÚXÚ\È^XÝHÚ]\[™Y\™K‚ˆœš[ŠÝ\œ‹–Ýš×HÖ—Õ’×Ñ”SQWÕPÑNˆ	\ÈOˆ	\×ˆ‹‹ˆÈ›Ü[‹Û™H›ÝÈ\ˆ™\Ù[Yœ˜[YHˆˆ‘RSQÈÔSˆŠNÂˆYˆ

Bˆœš[Š™œ˜[YH˜]ÜÈØ[\ÈØ[Õ\È™XÛÜ™\È™[˜ÙU\ÈÛY\\È‚ˆœ™\ÚYX[\ÈÜU\È^\ØYÈ^Ðˆ^\\È‚ˆ^XÕ\È\\È™XÔ\ÈÝ•\ÈÝ‘ÝX\™\ÈÛÛœÝ\È‚ˆ^\È™XY˜XÚÕ\È™XÔÝ]U\È™XÕ™\\È™XÒY\È‚ˆ™˜]ÓÝ\•\ÈÒÙ^U\ÈÔ\U\ÈÑ™]Ú\ÈÔÚY\•\È‚ˆ›Ð™YÚ[•\ÈÕZ[\ÈÕœÕ\ÈÔÕ\ÈÔÚ•\ÈÕœÐÜ\È‚ˆ˜ÕœÔU\×ˆŠNÂˆËÈHTÑHÓÓSS”ÈT‘HÓ“HQPS’S‘Ñ•SÒUÖ—Õ’×Ô“Ñ’SHÑU8 %ˆËÈ›Ù”ØÛÜH™XÛÜ™È›Ý[™ÈÚ]Ý]][™^HÚ[[™XY‚ˆËÈ›Ü›X[H]ÛÝ[\Ü]X[YžH[H
H›Ø™HÛÜÝ[™ÈHØ[YHÜ™\‚ˆËÈ\ÈH[™ÈYX\Ý\™YÛÝÚHÊK]Hœ˜[Y\È[™\ˆ[™\ÝYØ][Û‚ˆËÈ\™HÍKLN\È[™H›Ùš[\ˆ\È‹M\ÎˆŒÉHÛˆHÝ]\ˆœ˜[YKˆËÈÚ\™HÛˆHŽ\Èœ˜[YH]ÛÝ[™HH[ˆHš[\ÈÝ]YÛÂˆËÈ›Ø›ÙH][Ý\ÈH“Ô“PSœ˜[YIÜÈ[YHœ›ÛHH[ˆØ\œžZ[™È\Ë‚ˆ[ÙBˆœš[ŠÝ\œ‹–Ýš×HÖ—Õ’×Ñ”SQWÕPÑNˆÐS““ÕÔ’UH	\×ˆ‹ŠNÂˆ™]\›ˆÂˆJ
NÂˆËÈHÔTUÔ‰ÔÈÕUTˆPT’ÑTˆ
ÊKˆÝ[\Y[ÈH˜XÙHS‘HÙËˆËÈ™XØ]\ÙHHÙÈ\ÈÚ]Ù]È™XYš\œÝ[™HX\šÙ\ˆÛ›H[ˆH]BˆËÈš[HÛÝ[™H›Ý[™Û›HžHÛÛY[Û™H[™XYHÛÚÚ[™È›Üˆ]‚ˆYˆ
ÜÝÐÛÛœÝ[YSX\šÔ™\ÜÙY

JBˆÂˆ×ÛX\šÐÛÝ[
ÊÎÂˆœš[ŠÝ\œ‹ˆ–Ýš×H
ŠˆPT’È	[H8 %Ü\˜]Üˆ›YÙÙYHÝ]\ˆ]œ˜[YH	[H‚ˆŠ	KŒYˆ\ÎˆÔ\™XÈ	KŒY‹™[˜ÙH	KŒY‹ÛY\	KŒY‹ÔH	KŒYŽÈ‚ˆ‰]H˜]ÜË	[H^
Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÛX\šÐÛÝ[ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O™œ˜[YKÝX›Jœ˜[YU\Ë˜˜XÚÊ
JHÈLŒˆÝX›J[Ì—Ý
Ø[Õ\ÊHH[Ì—Ý
™[˜ÙQ[HÈL
JHÈLŒˆÝX›J™[˜ÙQ[JHÈYM‹ÝX›JÛY\\ÊHÈLŒˆÝX›J×ÙÜSœÓÙ‘œ˜[YJHÈYM‹Z[Ì—Ý
‹O™˜]ÜÕ\Ñœ˜[YJKˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ×Ý^™X[\ØYÈH™]•^›Ü•˜XÙJJNÂˆYˆ
˜XÙJBˆœš[Š˜XÙKˆÈPT’È	[Hœ˜[YH	[Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÛX\šÐÛÝ[ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O™œ˜[YJNÂˆBˆYˆ
˜XÙJBˆœš[Š˜XÙKˆ‰[H	]H	]H	]H	Y	[H	]H	Y	[H	[H	[H	[H	[H	[H‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O™œ˜[YKZ[Ì—Ý
‹O™˜]ÜÕ\Ñœ˜[YJKˆœ˜[YU\Ë˜˜XÚÊ
KØ[Õ\Ëˆ[Ì—Ý
Ø[Õ\ÊHH[Ì—Ý
™[˜ÙQ[HÈL
Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ™[˜ÙQ[HÈL
KÛY\\Ëˆ[Ì—Ý
œ˜[YU\Ë˜˜XÚÊ
JHH[Ì—Ý
Ø[Õ\ÊHH[Ì—Ý
ÛY\\ÊKˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ×ÙÜSœÓÙ‘œ˜[YHÈL
Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ×Ý^™X[\ØYÈH™]•^›Ü•˜XÙJKˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ
×Ý^\ØYž]\ÈH™]•^ž]\Ñ›Ü•˜XÙJBˆÈL
Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ
×Ý^\ØYœÈH™]•^œÑ›Ü•˜XÙJHÈL
Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ^XÑ[HÈL
Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ×Ü\PÛÝ[H™]”\\Ñ›Ü•˜XÙJJNÂˆ™]•^œÑ›Ü•˜XÙHH×Ý^\ØYœÎÂˆYˆ
˜XÙJBˆÂˆÝ]XÈ›Ùš[T\Ù\È™]”\Ù^ßNÂˆËÈ›ÝÈH™]˜S”ÒQÓ‘QÚ]HÝX\™™XØ]\ÙHHÛÝ[\œÈ]™XYÂˆËÈ\™H‘T“ÑQžHÖ—Õ’×Ô“Ñ’SIÜÈ\š[ÙXÈÚ[™ÝÈ™\ÜˆÚ]Ý]BˆËÈÝX\™Hœ˜[YHY\ˆXXÚÚ[™ÝÈš[ÈNÍÌÍÌH\È8 %BˆËÈÜ˜\Y™YØ]]™H8 %[™H™XY\ˆØØ[›š[™È›ÜˆHÛÜœÝœ˜[Y\Èš[™ÂˆËÈÚ^[\ÜÜÚX›HÛ™\È]HÜÙˆ]™\žHÛÜY\Ýˆ]\[™Y‚ˆËÈ]ÛÜÝH\ÜÈÝ™\ˆ\È]HÈ›ÝXÙHHÜÚ^›ÝÜÈÙ\™HBˆËÈ[œÝ[Y[[™›ÝHØ[YK‚ˆËÂˆËÈ™\›Ë›ÝH˜]ÈÜ˜\ˆY\ˆH™\Ù]HYH[H\È[šÛ›ÝÛ‹[™ˆËÈ\ÈHÛ›H[œÝÙ\ˆ]Ø[››Ý™HZ\ÝZÙ[ˆ›ÜˆHYX\Ý\™[Y[ˆBˆËÈœ˜[YH\ÈÝ[[Z]YÛÈ›Ý[™È\ÈÚ[[H›ÜY‚ˆ]]ÈH×JZ[Ý›ÝËZ[Ý	ˆ™]ŠHÂˆÛÛœÝZ[ÝˆH›ÝÈH™]ˆÈ›ÝÈH™]ˆˆÂˆ™]ˆH›ÝÎÂˆ™]\›ˆ
[œÚYÛ™YÛ™ÈÛ™ÊJˆÈL
NÂˆNÂˆËÈU‘T–H\ÙK›ÝHÚÜÙ[ˆ™]ËˆHš\œÝÝ]š[YÚ^[™^BˆËÈÝ[[YYÈŒ	HÙˆHœ˜[YH8 %™XØ]\ÙH™XÛÜ™\ÈH‘TÒQPSY\‚ˆËÈ™XÛÜ™Ý]KÜ™XÛÜ™™\^Ü™XÛÜ™[™^\™HÝX˜XÝYœ›ÛH][™ˆËÈÜÙH™YHÙ\™H›Ý[ˆH\ÝˆHœ™XZÙÝÛˆ]Ù\È›ÝY\ˆËÈ\ÈHØ[YH˜[ÙKXXœÙ[˜ÙH˜\\ÈHÛ™HH™\ÚYX[ÛÛ[[ˆØ\ÂˆËÈZ[ÈÛÜÙKÛ™H]™[ÝÛŽˆš[[Ùˆ[H[™]BˆËÈ\š]Y]XÈ™HÚXÚØX›K‚ˆœš[Š˜XÙKˆˆ	[H	[H	[H	[H	[H	[H	[H	[H	[H	[H	[H	[H‚ˆ‰[H	[H	[H	[H	[H	[H	[H	[H	[Wˆ‹ˆ
×Ü›Ù‹œ™XÛÜ™™]”\ÙKœ™XÛÜ™
Kˆ
×Ü›Ù‹œÝ™X[\Ë™]”\ÙKœÝ™X[\ÊKˆ
×Ü›Ù‹œÝ™X[QÝX\™™]”\ÙKœÝ™X[QÝX\™
Kˆ
×Ü›Ù‹˜ÛÛœÝ[Ë™]”\ÙK˜ÛÛœÝ[ÊKˆ
×Ü›Ù‹^\™\Ë™]”\ÙK^\™\ÊKˆ
×Ü›Ù‹œ™XY˜XÚË™]”\ÙKœ™XY˜XÚÊKˆ
×Ü›Ù‹œ™XÛÜ™Ý]K™]”\ÙKœ™XÛÜ™Ý]JKˆ
×Ü›Ù‹œ™XÛÜ™™\^™]”\ÙKœ™XÛÜ™™\^
Kˆ
×Ü›Ù‹œ™XÛÜ™[™^™]”\ÙKœ™XÛÜ™[™^
Kˆ
×Ü›Ù‹™˜]ÓÝ\‹™]”\ÙK™˜]ÓÝ\ŠKˆ
×Ü›Ù‹›Ý\’Ù^K™]”\ÙK›Ý\’Ù^JKˆ
×Ü›Ù‹›Ý\”\[[™K™]”\ÙK›Ý\”\[[™JKˆ
×Ü›Ù‹›Ý\‘™]Ú™]”\ÙK›Ý\‘™]Ú
Kˆ
×Ü›Ù‹›Ý\”ÚY\‹™]”\ÙK›Ý\”ÚY\ŠKˆ
×Ü›Ù‹›Ý\™YÚ[‹™]”\ÙK›Ý\™YÚ[ŠKˆ
×Ü›Ù‹›Ý\•Z[™]”\ÙK›Ý\•Z[
Kˆ
×Ü›Ù‹˜ÛÛœÝœË™]”\ÙK˜ÛÛœÝœÊKˆ
×Ü›Ù‹˜ÛÛœÝË™]”\ÙK˜ÛÛœÝÊKˆ
×Ü›Ù‹˜ÛÛœÝÚ\™Y™]”\ÙK˜ÛÛœÝÚ\™Y
Kˆ
×Ü›Ù‹˜ÛÛœÝœÐÛÜK™]”\ÙK˜ÛÛœÝœÐÛÜJKˆ
×Ü›Ù‹˜ÛÛœÝœÔ]Ú™]”\ÙK˜ÛÛœÝœÔ]Ú
JNÂˆBˆ™]•^›Ü•˜XÙHH×Ý^™X[\ØYÎÂˆ™]•^ž]\Ñ›Ü•˜XÙHH×Ý^\ØYž]\ÎÂˆ™]”\\Ñ›Ü•˜XÙHH×Ü\PÛÝ[ÂˆBˆÛÛœÝÝX›H[\ÙYBˆÝŽ˜Ú›Û›ÎŽ™\˜][ÛÝX›OŠ›ÝÈHÚ[™ÝÔÝ\
K˜ÛÝ[

NÂˆYˆ
œÓÙÔÙXÈˆ	‰ˆ[\ÙYHÝX›JœÓÙÔÙXÊH	‰ˆœ˜[Y\ÈˆJBˆÂˆËÈHš\œÝØ[\HÙˆHÚ[™ÝÈ\ÈHØ\PÔ“ÔÔÈHÚ[™ÝÈ›Ý[™\žH[™ˆËÈ™[Û™ÜÈÈ™Z]\ŽÈ›Ü[™È]ÛÜÝÈÛ™Hœ˜[YH[ˆH™]È[™™Y‚ˆÝŽœÛÜ
œ˜[YU\Ë˜™YÚ[Š
H
ÈKœ˜[YU\Ë™[™

JNÂˆÛÛœÝZ[Ì—ÝYY\ÈHœ˜[YU\ÖÊœ˜[YU\ËœÚ^™J
H
ÈJHÈ—NÂˆËÈHRSYY[ˆ\ÌH›ÜˆHÜ\˜]Ü‰ÜÈT“ˆÕUTˆ8 %HÛ™BˆËÈ\™›Ü›X[˜ÙH›Ø›[HÛˆ\ÈÜ]\È™Y[ˆ™\ÜY\È‘S˜]\‚ˆËÈ[ˆYX\Ý\™Y[™HÛ™HHYYX[ˆ\ÈX\ÝX›HÈÙYH
ÛÝÚHŒÍÎ‚ˆËÈ™XYH\ÝšX][Û‹›ÝHÙ[™JKˆHÚ[™ÝÉÜÈœ˜[Y\È\™H[™XYBˆËÈÛÜY›ÜˆHYYX[‹ÛÈNH[™HÚ\™HX›Ý™HžHYYX[ˆ\™HÛÂˆËÈ\œ˜^H™XYÈ[™Û™HÛÜÝ™\ˆ]H[ˆØXÚH8 %›Ý[™È\ÈØ[ˆÛÜÝ\ÂˆËÈš\ÚX›HYØZ[œÝHœ˜[YKˆÚ]Ý][HHÛØZÈ[™HÍŒYYÜ™YH\›‚ˆËÈ›ÙXÙHHØ[YH[™H[™HÝ]\ˆ\È[™˜[ÚYšXX›K‚ˆÛÛœÝÚ^™WÝˆHœ˜[YU\ËœÚ^™J
NÂˆÛÛœÝZ[Ì—ÝNU\ÈHœ˜[YU\ÖÛˆHHH
ˆHJHÈLNÂˆZ[Ì—ÝÛÜœÝ\ÈHœ˜[YU\Ë˜˜XÚÊ
NÂˆÚ^™WÝÝ™\•ÚXÙHHÂˆ›Üˆ
Ú^™WÝHHNÈHŽÈ
ÊÚJBˆYˆ
œ˜[YU\ÖÚWHˆˆ
ˆYY\ÊBˆ
ÊÛÝ™\•ÚXÙNÂˆZ[Ì—ÝZ[ˆHYYHX^HÂˆYˆ
Yœ˜[YQ˜]ÜË™[\J
JBˆÂˆÝŽ™XÝÜZ[Ì—ÝˆHœ˜[YQ˜]ÜÎÂˆÝŽœÛÜ
˜™YÚ[Š
K™[™

JNÂˆZ[ˆH™œ›Û

NÂˆYYHÙœÚ^™J
HÈ—NÂˆX^H˜˜XÚÊ
NÂˆBˆËÈHST	ÔÈÕÓˆÔHTˆ”SQKÛˆHÙœ×H[™H[™\™Y›Ü™BˆËÈ]˜Z[X›H[ˆ[ˆS”“Ñ’SQ[ˆ
\LL0©ÌËŒJK‚ˆËÂˆËÈ]^\ÝÈ™XØ]\ÙHH]X[]H\LL\ÈÈYX\Ý\™H8 %H[\	ÜÂˆËÈÔHZ[\ÙXÛÛ™È\ˆ™\Ù[Yœ˜[YH8 %Ø\ÈÛ›H]™\ˆØZ[˜X›HžBˆËÈÜ›ÜÜÚ[™ÈÛÈ[œÝ[Y[ÈZÙ[ˆÝ™\ˆQ‘‘T‘S•Ú[™ÝÜÎˆ\™˜	ÜÈÜ‚ˆËÈ\LÝ™XYØÜKœX	ÜÈ‰HÙˆÛ™HÛÜ™HˆÝ™\ˆ]ÈÝÛˆMHÈØ[\KˆËÈ]šYY[ÈHœ˜[YH˜]Hœ›ÛHÛÛY]Ú\™H[ÙKˆ\È›Ú™XÝ\ÈBˆËÈ˜[YH›Üˆ]\š]Y]XÈ[™][™[YNHP‹Ùœ˜[YH]™]™\‚ˆËÈ^\ÝY
ÛËXÛÝ[\œËX\™K[›ÝXK\Z\˜
KˆÛ™HÛØÚ×ÙÙ][YX\ˆ”ÂˆËÈÒS‘ÕÈ8 %›Ý\ˆœ˜[YH8 %XZÙ\È]Û™HYX\Ý\™[Y[Ý™\ˆÛ™HÚ[™ÝËˆËÈ˜[™YžHHØ[YH˜]ÈÛÝ[\È]™\ž][™È[ÙHÛˆ\È[™K‚ˆËÂˆËÈœ™YNˆHÙœ×HÚ[™ÝÈ\ÈÙXÛÛ™ÈÛ™ËÛÈ\È\ÈÛ™H‘ÓÈ™XY\‚ˆËÈÙ]™\˜[[™™Yœ˜[Y\ËYØZ[œÝH›Ùš[\‰ÜÈÝ\Ø[™È\ˆœ˜[YK‚ˆËÈ]\Èš[Y[˜ÛÛ™][Û˜[H[™\ˆÖ—Ñ”×ÓÑÈ™XØ]\ÙHH[X™\ˆ]ˆËÈ™YYÈ]ÈÝÛˆ[ˆ˜\ˆ\ÈH[X™\ˆ›Ø›ÙH\ÈÚ[ˆ^H™YY]‚ˆÝX›H[\ÜS\ÈHŒ[\]HHŒÂˆÂˆÝ]XÈZ[Ý\Ý[\ÜSœÈHÂˆÝ]XÈ›ÛÛ]™T[\ÜHH˜[ÙNÂˆ[Y\ÜXÈÞßNÂˆÛØÚ×ÙÙ][YJÓÐÒ×Õ‘PQÐÔUSQWÒQ	œÊNÂˆÛÛœÝZ[Ý›ÝÓœÈBˆZ[Ý
Ë—ÜÙXÊH
ˆL[
ÈZ[Ý
Ë—ÛœÙXÊNÂˆYˆ
]™T[\ÜH	‰ˆœ˜[Y\ÊBˆÂˆÛÛœÝÝX›HœÈHÝX›J›ÝÓœÈH\Ý[\ÜSœÊNÂˆ[\ÜS\ÈHœÈ
ˆYKMˆÈÝX›Jœ˜[Y\ÊNÂˆ[\]HH[\ÙYˆŒÈLŒ
ˆœÈ
ˆYKNHÈ[\ÙYˆŒÂˆBˆ\Ý[\ÜSœÈH›ÝÓœÎÂˆ]™T[\ÜHHYNÂˆBˆËÈHÕQTÕ	ÔÈÕÓˆÔHTˆ”SQKØ[YHÚ[™ÝËØ[YH[™H
\LMŠKˆBˆËÈ]IÜÈXZ[ˆ™XY[™˜]È™XY\™HHŽ\È›ÛÜˆ\LLˆËÈ›Ý[™[™\ˆØ[ˆX^
[\ÝY\ÝÔJXÈHÝY\Ý\ÚYHÚ[™ÙH
BˆËÈ˜]]™HÔ•ÛÚËÓÈÛˆH™XÛÛ\[Y\ÊH[Ý™\ÈTÑHÛÛ[[œÈ[™ˆËÈÚ[HH[\\ÈHÛ™Ù\ˆ\›K›Ý[™È[ÙKˆ™XY›ÝYÚBˆËÈ˜[YY™XY	ÜÈÔHÛØÚÎÈLH[[H]H\È˜[YY]È™XYË‚ˆËÈ\LMÎˆ[™\ˆÖ—ÔSTÔÔULH[\ÜXX›Ý™H\ÈTÈ™XY8 %BˆËÈÛ™HØ[[™ÈÑ˜]ËK™KˆÞ‹Y˜]È8 %[™HØ[ÉÜÈÝÛˆÛÜ™H\È\ÂˆËÈÛÛ[[‹ˆLHÛˆHÛ™K]™XY[\‚ˆÝX›HØ[ÐÜS\ÈHLKŒÂˆÂˆÝ]XÈÝX›H\ÝØ[ÈHLKŒÂˆÛÛœÝÝX›H›ÝÕØ[ÈHÜ]Ž•Ø[ÐÜTÙXÛÛ™Ê
NÂˆYˆ
œ˜[Y\È	‰ˆ\ÝØ[ÈHŒ	‰ˆ›ÝÕØ[ÈHŒ
BˆØ[ÐÜS\ÈH
›ÝÕØ[ÈH\ÝØ[ÊH
ˆYLÈÈÝX›Jœ˜[Y\ÊNÂˆ\ÝØ[ÈH›ÝÕØ[ÎÂˆBˆÝX›HÝY\ÝXZ[“\ÈHLKŒÝY\Ý˜]Ó\ÈHLKŒÂˆÂˆÝ]XÈÝX›H\ÝXZ[ˆHLKŒ\Ý˜]ÈHLKŒÂˆÛÛœÝÝX›H›ÝÓXZ[ˆHÝY\Ý™XYŽÜTÙXÛÛ™ÓÙŠ“XZ[ˆ™XYŠNÂˆÛÛœÝÝX›H›ÝÑ˜]ÈHÝY\Ý™XYŽÜTÙXÛÛ™ÓÙŠ‘˜]È™XYŠNÂˆYˆ
œ˜[Y\È	‰ˆ\ÝXZ[ˆHŒ	‰ˆ›ÝÓXZ[ˆHŒ
BˆÝY\ÝXZ[“\ÈH
›ÝÓXZ[ˆH\ÝXZ[ŠH
ˆYLÈÈÝX›Jœ˜[Y\ÊNÂˆYˆ
œ˜[Y\È	‰ˆ\Ý˜]ÈHŒ	‰ˆ›ÝÑ˜]ÈHŒ
BˆÝY\Ý˜]Ó\ÈH
›ÝÑ˜]ÈH\Ý˜]ÊH
ˆYLÈÈÝX›Jœ˜[Y\ÊNÂˆ\ÝXZ[ˆH›ÝÓXZ[ŽÂˆ\Ý˜]ÈH›ÝÑ˜]ÎÂˆBˆËÈ‹‹˜[™Ú\™HXXÚ™XY	ÜÈ“Ó‹PÔH[YHÙ[ˆ\ËÙœ˜[YH[™Ø[ËÙœ˜[YBˆËÈ[ˆÝ\ˆÙ\›™[	ÜÈØZ]ËžHÚ[™8 %Ú[™ÛK[Øš™XÝØZ]X[žKØ[ÛY\ˆËÈ™[˜ÙH\šÈ
\LMˆ][H
KˆØ[YHÚ[™ÝËØ[YH[›ÛZ[˜]Ü‹‚ˆÚ\ˆØZ][™VÌM—HHˆŽÂˆÂˆÝ]XÈZ[Ý\ÝœÖÌ—VÑÝY\Ý™XYŽšÕØZ]Ú[™×HHßNÂˆÝ]XÈZ[Ý\ÝØ[ÖÌ—VÑÝY\Ý™XYŽšÕØZ]Ú[™×HHßNÂˆÝ]XÈ›ÛÛ]™UØZ]H˜[ÙNÂˆÛÛœÝÚ\Šˆ˜[Y\ÖÌ—HHÈ“XZ[ˆ™XY‹‘˜]È™XYˆNÂˆÛÛœÝÚ\ŠˆÚ[™ÖÑÝY\Ý™XYŽšÕØZ]Ú[™×HHÈœÚ[™ÛH‹›][H‹œÛY\‹™™[˜ÙHˆNÂˆÚ^™WÝÙ™ˆHÂˆ›ÛÛ[žHH˜[ÙNÂˆ›Üˆ
[HÈŽÈ
ÊÊBˆÂˆÛÛœÝÝY\Ý™XYŽ•ØZ]Ý]ÊˆÈHÝY\Ý™XYŽ•ØZ]Ý]ÓÙŠ˜[Y\ÖÝJNÂˆYˆ
]ÊBˆÛÛ[YNÂˆ[žHHYNÂˆÙ™ˆ
ÏHÛœš[ŠØZ][™H
ÈÙ™‹Ú^™[ÙˆØZ][™HHÙ™‹‰\É\Îˆ‹ˆÈˆˆˆˆ‹È™˜]Èˆˆ›XZ[ˆŠNÂˆ›Üˆ
[ÈHÈÈÝY\Ý™XYŽšÕØZ]Ú[™ÎÈÊÊÊBˆÂˆÛÛœÝZ[ÝœÈHËO›œÖÚ×K›ØY
ÝŽ›Y[[ÜžWÛÜ™\—Ü™[^Y
NÂˆÛÛœÝZ[ÝØ[ÈHËO˜Ø[ÖÚ×K›ØY
ÝŽ›Y[[ÜžWÛÜ™\—Ü™[^Y
NÂˆYˆ
]™UØZ]	‰ˆœ˜[Y\È	‰ˆÙ™ˆÚ^™[ÙˆØZ][™JBˆÙ™ˆ
ÏHÛœš[ŠØZ][™H
ÈÙ™‹Ú^™[ÙˆØZ][™HHÙ™‹ˆˆ	\È	KŒ™‹ÉKŒYˆ‹Ú[™ÖÚ×KˆÝX›JœÈH\ÝœÖÝVÚ×JH
ˆYKMˆÈÝX›Jœ˜[Y\ÊKˆÝX›JØ[ÈH\ÝØ[ÖÝVÚ×JHÈÝX›Jœ˜[Y\ÊJNÂˆ\ÝœÖÝVÚ×HHœÎÂˆ\ÝØ[ÖÝVÚ×HHØ[ÎÂˆBˆBˆ]™UØZ]H[žNÂˆBˆœš[ŠÝ\œ‹ˆ–Ùœ×H	KŒYˆœÈYX[ˆ
	KŒ™ˆ\ÊH	KŒYˆœÈYYX[ˆ
	KŒ™ˆ\ÊH‚ˆœNH	KŒ™ˆ\ÈÛÜœÝ	KŒ™ˆ\ÈŒžYY	KŒY‰IH‚ˆ‰[Hœ˜[Y\È[ˆ	KŒYˆÈ˜]ÜÈYY	]H
	]K‹‰]JH‚ˆœ[\ÜH	KŒ™ˆ\ËÙœ˜[YH
	KŒ‰IHÙˆHÛÜ™JHØ[ÈÜH	KŒ™ˆ‚ˆ™ÝY\ÝXZ[ˆ	KŒ™ˆ˜]È	KŒ™ˆ\ËÙœ˜[YWˆ‹ˆÝX›Jœ˜[Y\ÊHÈ[\ÙYLŒ
ˆ[\ÙYÈÝX›Jœ˜[Y\ÊKˆYMˆÈÝX›JYY\ÊKÝX›JYY\ÊHÈLŒˆÝX›JNU\ÊHÈLŒÝX›JÛÜœÝ\ÊHÈLŒˆˆˆHÈLŒ
ˆÝX›JÝ™\•ÚXÙJHÈÝX›JˆHJHˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊYœ˜[Y\Ë[\ÙYYYZ[‹X^ˆ[\ÜS\Ë[\]KØ[ÐÜS\ËÝY\ÝXZ[“\ËÝY\Ý˜]Ó\ÊNÂˆYˆ
ØZ][™VÌJBˆœš[ŠÝ\œ‹–ÙÝY\ÝØZ]H\ËÙœ˜[YHÈØ[ËÙœ˜[YNˆ	\×ˆ‹ØZ][™JNÂˆ™XYYÙ]Ô[”ÝÙY\

NÈËÈÖ—ÑÕQTÕÔSˆ
\LN
NˆH›Ë[Ü[›\ÜÈÙ]ˆËÈ‹‹˜[™H™YÚ\Ý\‹\[ˆÙ[œÝ\È™\ÚYH]Ú[ˆ\›YYTˆÒS‘ÕÈ˜]\‚ˆËÈ[ˆÛ›H]^]ˆH^]]\ÈHšYÚÛYH›ÜˆHÝ[[X\žBˆËÈ
ÛÝÚHMÊH]]\È›ÝH™[XX›HÛ™NˆÛÈ[œÈÛšYÚ[™YˆËÈÚ]Ý]HÒQÕT“H[™\ˆš[[™È[ž][™È][[™HÙ[œÝ\È]ˆËÈÛ›HÜXZÜÈÛˆHØ^HÝ]\ÈHÙ[œÝ\È]ÛÛYH[œÈÚ[\HÈ›ÝˆËÈ]™KˆHÚ[™ÝÙYš[ÛÜÝÈ[ˆ[™\È]™\žH”ÈÚ[™ÝÈÛˆBˆËÈXYÛ›ÜÝXË[Û›H\›H[™Ø[››Ý™HÜÝ‚ˆËÈHØÛÜYÚ\™YX›ØÚÈ™\›ÉÜÈÝÛˆ›ÛÙ‹\ˆÚ[™ÝÈ˜]\ˆ[ˆ]ˆËÈ^]ˆž]\È“ÕÜš][‹[™HÚ\™HÙˆH‹NLˆH˜]È\ÙYÂˆËÈÛÜÝ[˜ÛÛ™][Û˜[H8 %[ˆ\›H]Ø^\È›ÛˆˆÚ]Ý]Ø^Z[™Èœ™XXÚY‚ˆËÈ\ÈÝÈH[Ù]È][ÝY\ÈHØ]š[™Ë‚ˆËÈHÑRSS‘È“Ð‘IÔÈS‘ÐQÑSQS•
\LL0©ÌËŒJK\ˆÚ[™ÝÈ[™ˆËÈ›ÝÛ›H]^]
ÛÝÚHMÎˆÛÈÙˆ\LIÜÈ[œÈ[™YÚ]Ý]ˆËÈHÒQÕT“H[™\ˆš[[™È[ž][™È][
Kˆ]ÈÛÛ›Û\›H8 %ˆËÈ]™\žHÝ\ˆ[ˆ8 %š[È›Ý[™Ë™XØ]\ÙHHÛÝ[\ˆ™]™\ˆ[Ý™\Ë‚ˆYˆ
×Û›ÑÑ˜]ÔÚÚ\Y
Bˆœš[ŠÝ\œ‹ˆ–Û›ÙÙ˜]×HTÕ•PÕU‘Nˆ	[H˜]ÜÈÚÚ\Y[ˆÝ[‚ˆ‰]H\Èœ˜[YH8 %HØ[È˜[‹H˜]ÜÈY›Ýˆ™XYH‚ˆœ[\™XY	ÜÈÔK™]™\ˆ\È[‰ÜÈØ[[YK—ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Û›ÑÑ˜]ÔÚÚ\Yˆ[œÚYÛ™Y
‹O›\Ýœ˜[YQ˜]ÜÊJNÂˆËÈŒIÜÈS‘ÐQÑSQS•
\LLH0©ÍŒÊK\ˆÚ[™ÝË[™“ÕÒQTÈÑˆBˆËÈ’SˆH]˜]HØ^\ÈH˜\Ý]˜[ŽÈHÛÜšÙ\ˆZ[\ÙXÛÛ™ÈØ^BˆËÈÚ\™HHÛÜšÈ]™[[Ý™Yœ›ÛHH[\XÝX[H[™Yˆ\LÈ[Ý™YˆËÈLËŒHÚ[ÈÙ™ˆH[\[™ÌËŒˆ\X\™YÛˆHÛÜšÙ\œËÛÈH[™BˆËÈ]™\ÜÈÛ›HH[\	ÜÈ[ˆ\È›ÝHYX\Ý\™[Y[
ÛÝÚHÍ
K‚ˆËÈHÓÓ•“ÓT“H
Ö—Õ’×Ó“×Ô‘V‘T“ÏLX
Hš[ÈHÜÜÚ]Nˆ]ÈˆËÈ]™\žH˜]È[›[™K‚ˆYˆ
×Ü’]È×Ü“Z\ÜÙ\ÊBˆÂˆÝ]XÈZ[ÝHHHÛœÈHœÈH™HH[ˆHÂˆÛÛœÝZ[ÝH×Ü’]ËHH×Ü“Z\ÜÙ\ÎÂˆÛÛœÝZ[ÝÛœÈH×Ü•ÛÜšÙ\“œÐK›ØY
ÝŽ›Y[[ÜžWÛÜ™\—Ü™[^Y
NÂˆÛÛœÝZ[ÝœÈH×Ü‘˜Z[“œÎÂˆÛÛœÝÝX›H[ˆHKŒÈÝX›Jœ˜[Y\ÊNÂˆœš[ŠÝ\œ‹ˆ–Ü™^™\›×H	KŒY‰IHÙˆ˜]ÜÈÙ\™Y™K^™\›ÙY
	[H]Ë‚ˆ‰[H[›[™H˜[˜XÚÜÈ\ÈÚ[™ÝÊH	KŒ™ˆP‹Ùœ˜[YH[Ý™YÙ™ˆ‚ˆH[\	KŒ™ˆP‹Ùœ˜[YHÝ[[›[™HÛÜšÙ\ˆ	KŒÙˆ\ËÙœ˜[YK‚ˆœ[\˜Z[ˆ	KŒÙˆ\ËÙœ˜[YH\ÝØ]\›X\šÈ	[HÝ™\™›ÝÈ‚ˆ‰[Wˆ‹ˆ

ÈHˆ
ÈJBˆÈLŒ
ˆÝX›JH
HÈÝX›J
H
H
È
HHJJBˆˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊJH
K
[œÚYÛ™YÛ™ÈÛ™ÊJHHJKˆÝX›J×Üž]\Ô™HH™JH
ˆ[ˆÈLMÍ‹ŒˆÝX›J×Üž]\Ò[›[™HH[ŠH
ˆ[ˆÈLMÍ‹ŒˆÝX›JÛœÈHÛœÊH
ˆ[ˆ
ˆYKM‹ˆÝX›JœÈHœÊH
ˆ[ˆ
ˆYKM‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü™^[Û™Ø]\›X\šËˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü“Ý™\™›ÝÊNÂˆHÈHHNÈÛœÈHÛœÎÈœÈHœÎÂˆ™HH×Üž]\Ô™NÈ[ˆH×Üž]\Ò[›[™NÂˆBˆYˆ
×ÜÚ\™Y™\›Ñ˜]ÜÊBˆœš[ŠÝ\œ‹ˆ–ÜÚ\™Y™\›×H	[H˜]ÜË	KŒˆž]\ËÙ˜]È›ÝÜš][ˆ‚ˆŠ	KŒY‰IHÙˆH	]KXž]H›ØÚÊWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÜÚ\™Y™\›Ñ˜]ÜËˆÝX›J×ÜÚ\™Y™\›ÔØ]™Y
HÈÝX›J×ÜÚ\™Y™\›Ñ˜]ÜÊKˆLŒ
ˆÝX›J×ÜÚ\™Y™\›ÔØ]™Y
HÂˆ
ÝX›J×ÜÚ\™Y™\›Ñ˜]ÜÊH
ˆÝX›JÔÚ\™YÚ^™JJKˆÔÚ\™YÚ^™JNÂˆMÔ™YÔ[Ù[œÝ\Ô™\Ü

NÂˆYˆ
Ü]Ž™×ÛÛŠBˆÂˆËÈ\LMÎˆH]Y]YIÜÈX[\ˆÚ[™ÝËˆÜÜXÙX]\ÝÝ^H
ÂˆËÈ™]™\ˆ›ØÚÙY›Üˆ›ÛÛJNÈ\œ]ØZ]\È	ÜÈÝ[]S•T”•TÜË‚ˆÝ]XÈÜ]Ž”Ý]È\ÝßNÂˆÛÛœÝÜ]Ž”Ý]ÈÝHÜ]Ž‘Ù]Ý]Ê
NÂˆœš[ŠÝ\œ‹ˆ–ÜÜ]H\ˆœ˜[YNˆÜÈ	KŒˆ˜]ÜÈ	KŒˆÝÜ™\È	KŒˆ\œH	KŒYˆ‚ˆ›ÙÙÈ	KŒˆ
	KŒˆ[œÈ
È	KŒˆY\™ÙY
HÜÜXÙH	[H[\H	[H\œ]ØZ]	KŒ™ˆ\ËÙœ˜[YH‚ˆŠ	KŒˆ\ÈXXÚ
HYH	KŒ™ˆ\ËÙœ˜[YHØZ]È[›Y]	KŒY‹Ùœ˜[YH‚ˆ›ÙˆÚXÚÛˆÕTˆÝÜ™H	KŒY‹[‹XZXY	KŒY—ˆ‹ˆÝX›JÝ›ÜÈH\Ý›ÜÊHÈÝX›Jœ˜[Y\ÊKˆÝX›JÝ™˜]ÜÈH\Ý™˜]ÜÊHÈÝX›Jœ˜[Y\ÊKˆÝX›JÝœÝÜ™\ÈH\ÝœÝÜ™\ÊHÈÝX›Jœ˜[Y\ÊKˆÝX›JÝš[\œ\ÈH\Ýš[\œ\ÊHÈÝX›Jœ˜[Y\ÊKˆÝX›JÝ›ÙÑÛÜ™ÈH\Ý›ÙÑÛÜ™ÊHÈÝX›Jœ˜[Y\ÊKˆÝX›JÝœ[œÈH\Ýœ[œÊHÈÝX›Jœ˜[Y\ÊKˆÝX›JÝœ[œÓY\™ÙYH\Ýœ[œÓY\™ÙY
HÈÝX›Jœ˜[Y\ÊKˆ
[œÚYÛ™YÛ™ÈÛ™ÊJÝÔÜXÙUØZ]ÈH\ÝÔÜXÙUØZ]ÊKˆ
[œÚYÛ™YÛ™ÈÛ™ÊJÝ™[\UØZ]ÈH\Ý™[\UØZ]ÊKˆÝX›JÝ™\œUØZ]œÈH\Ý™\œUØZ]œÊH
ˆYKMˆÈÝX›Jœ˜[Y\ÊKˆÝš[\œ\Èˆ\Ýš[\œ\ÂˆÈÝX›JÝ™\œUØZ]œÈH\Ý™\œUØZ]œÊH
ˆYKLÈÂˆÝX›JÝš[\œ\ÈH\Ýš[\œ\ÊBˆˆŒˆÝX›JÝ™YSœÈH\Ý™YSœÊH
ˆYKMˆÈÝX›Jœ˜[Y\ÊKˆÝX›JÝØZ]Õ[›Y]H\ÝØZ]Õ[›Y]
HÈÝX›Jœ˜[Y\ÊKˆÝX›JÝØZ]ÓÛ“Ý\”ÝÜ™HH\ÝØZ]ÓÛ“Ý\”ÝÜ™JHÈÝX›Jœ˜[Y\ÊKˆÝX›JÝØZ]ÐžT[™[™ÈH\ÝØZ]ÐžT[™[™ÊHÈÝX›Jœ˜[Y\ÊJNÂˆœš[ŠÝ\œ‹ˆ–ÜÜ]HYHY\Žˆ˜]È	KŒ™ˆÝÜ™H	KŒ™ˆ\œH	KŒ™ˆÝØ\	KŒ™ˆ‚ˆ›Ý\ˆ	KŒ™ˆ\ËÙœ˜[YH[›Y]žHÛÜ™ˆ	L	KŒYˆ	L	KŒYˆ‚ˆ‰L	KŒYˆ	L	KŒYˆÙœ˜[YWˆ‹ˆÝX›JÝ™YPžRÚ[™œÖÌWHH\Ý™YPžRÚ[™œÖÌWJH
ˆYKMˆÈÝX›Jœ˜[Y\ÊKˆÝX›JÝ™YPžRÚ[™œÖÌ—HH\Ý™YPžRÚ[™œÖÌ—JH
ˆYKMˆÈÝX›Jœ˜[Y\ÊKˆÝX›JÝ™YPžRÚ[™œÖÌ×HH\Ý™YPžRÚ[™œÖÌ×JH
ˆYKMˆÈÝX›Jœ˜[Y\ÊKˆÝX›JÝ™YPžRÚ[™œÖÍHH\Ý™YPžRÚ[™œÖÍJH
ˆYKMˆÈÝX›Jœ˜[Y\ÊKˆÝX›JÝ™YPžRÚ[™œÖÌHH\Ý™YPžRÚ[™œÖÌJH
ˆYKMˆÈÝX›Jœ˜[Y\ÊKˆÝØZ]˜VÌKÝX›JÝØZ]˜PÛÝ[ÌHH\ÝØZ]˜PÛÝ[ÌJHÈÝX›Jœ˜[Y\ÊKˆÝØZ]˜VÌWKÝX›JÝØZ]˜PÛÝ[ÌWHH\ÝØZ]˜PÛÝ[ÌWJHÈÝX›Jœ˜[Y\ÊKˆÝØZ]˜VÌ—KÝX›JÝØZ]˜PÛÝ[Ì—HH\ÝØZ]˜PÛÝ[Ì—JHÈÝX›Jœ˜[Y\ÊKˆÝØZ]˜VÌ×KÝX›JÝØZ]˜PÛÝ[Ì×HH\ÝØZ]˜PÛÝ[Ì×JHÈÝX›Jœ˜[Y\ÊJNÂˆ\ÝHÝÂˆBˆËÈ\LÈ][HŽˆH˜]È™XY	ÜÈ™[˜ÙHØZ]\ˆÚ[™ÝË™\ÚYBˆËÈHœ˜[YH˜]H]\ÈYX[È[Ý™H8 %ÛÈHZ[ˆÜ›ÝÙ[ˆ
›È\ÙBˆËÈ›Ùš[\ŠHÝ[Ø^\ÈÚ]\ˆH\šÈS‘ÐQÑQ[™ÝÈXXÚ\\ÛÙBˆËÈ[™YˆH[ˆÚÜÙH\šÜÈ[[™[ˆ[Y[Ý]È\ÈHØZÙH™YXØ]H]ˆËÈ™]™\ˆš\™\ÎÈH[ˆÚ]›È\šÜÈ][\ÈHØZ]]™]™\ˆØZ]Ë‚ˆÂˆÝ]XÈ™[˜ÙUØZ]Ý]È\ÝÎÂˆÛÛœÝ™[˜ÙUØZ]Ý]ÈÈH™[˜ÙUØZ]ÔÝ]Ê
NÂˆÛÛœÝÝX›H[ˆHKŒÈÝX›Jœ˜[Y\ÊNÂˆœš[ŠÝ\œ‹ˆ–Ù™[˜Ù]ØZ]H\ˆœ˜[YNˆ›ÙH	KŒYˆ™XYH	KŒYˆÜ[‹\™\ÛÛ™Y‚ˆ‰KŒYˆ\šÜÈ	KŒYˆ
ÛÚÙ[ˆ	KŒYˆ[Y[Ý]È	KŒYˆRTÔÑQ	KŒYˆXYØZ[ˆ	KŒYŠH‚ˆ˜ÛÛ[™Y	KŒYˆ\ÜÝ›ÝYÚ	KŒYˆÝÜ™\ÈÙY[ˆ	KŒYˆØZÙ\È	KŒY‰\×ˆ‹ˆÝX›JË˜›ÙPØ[ÈH\ÝË˜›ÙPØ[ÊH
ˆ[‹ˆÝX›JËœ™XYP][žHH\ÝËœ™XYP][žJH
ˆ[‹ˆÝX›JËœÜ[”™\ÛÛ™YH\ÝËœÜ[”™\ÛÛ™Y
H
ˆ[‹ˆÝX›JËœ\šÜÈH\ÝËœ\šÜÊH
ˆ[‹ˆÝX›JËœ\šÕÛÚÙ[ˆH\ÝËœ\šÕÛÚÙ[ŠH
ˆ[‹ˆÝX›JËœ\šÕ[Y[Ý]ÈH\ÝËœ\šÕ[Y[Ý]ÊH
ˆ[‹ˆÝX›JËœ\šÓZ\ÜÙYH\ÝËœ\šÓZ\ÜÙY
H
ˆ[‹ˆÝX›JËœ\šÑXYØZ[ˆH\ÝËœ\šÑXYØZ[ŠH
ˆ[‹ˆÝX›JË˜ÛÛ[™YH\ÝË˜ÛÛ[™Y
H
ˆ[‹ˆÝX›JËœ\ÜÝ›ÝYÚH\ÝËœ\ÜÝ›ÝYÚ
H
ˆ[‹ˆÝX›JËœÝÜ™PÚXÚÜÈH\ÝËœÝÜ™PÚXÚÜÊH
ˆ[‹ˆÝX›JËØZÙPØ[ÈH\ÝËØZÙPØ[ÊH
ˆ[‹ˆ™[˜ÙUØZ]Ñ[˜X›Y

HÈˆˆˆˆÐÖ—Ñ‘SÑWÔT’ÏLˆÜ[›š[™×HŠNÂˆ\ÝÈHÎÂˆBˆÚ[™ÝÔÝ\H›ÝÎÂˆœ˜[Y\ÈHÂˆœ˜[YU\Ë˜ÛX\Š
NÂˆœ˜[YQ˜]ÜË˜ÛX\Š
NÂˆBˆBˆB‚ˆYˆ
T‹Oœ™XÛÜ™[™ÊBˆÂˆËÈHœ˜[YHÚ]›È™XÛÜ™YÛÜšÈ][ˆ™\Ù[H™]š[Ý\ÈÛÛ[È˜]\‚ˆËÈ[ˆ›Ý[™ËÛÈHÝ[[ˆH˜]È]ÚÝÜÈ\ÈHœ›Þ™[ˆXÝ\™H[œÝXYˆËÈÙˆH›XÚÙ\ˆ™]ÙY[ˆH™X[œ˜[YH[™›XÚË‚ˆÛÝ[
œÝØ\ˆ›Ý[™È™XÛÜ™YŠNÂˆ™]\›ŽÂˆB‚ˆËÈ™XYHÛÛÝ\ˆ\™Ù]˜XÚÈ[™[™]ÈHÚ[™ÝËˆH™XY˜XÚÈ\ˆœ˜[YH\ÈBˆËÈ™X[ÛÜÝ[™]\ÈÚÜÙ[ˆ[X™\˜][NˆH[\›˜]]™H\ÈH[Ø[ˆÝØ\ÚZ[ˆÛ‚ˆËÈHÑÚ[™ÝËÚXÚÛÝ[][Ø[ˆÛˆHÚ[™ÝÉÜÈ™XY[™ÛÝ\HBˆËÈ™[™\™\ˆÈHÚ[™ÝÚ[™ÈÞ\Ý[H]\ÙHÈ[X™\˜][HÙ\]\›IÜÈ[™Ý‚ˆËÈ]HÝY\Ý	ÜÈÝÛˆŒÌœ˜[Y\ÈHÙXÛÛ™ËHPˆHœ˜[YH\È›ÝÚ][Z]È\Ë‚ˆ[™™[™\š[™Ê
NÂ‚ˆËÈ™XY˜XÚÈHœ›ÛXY™™\ˆÛ˜\ÚÝÚ[ˆ\™H\ÈÛ™K[™H˜]ÈQSHÚ[‚ˆËÈ\™H\È›ÝˆH˜[˜XÚÈ\È[X™\˜]H[™\È[››Ý[˜ÙYžH]ÈÝÛˆÛÝ[\Ž‚ˆËÈ]\ÈÚ]Hœ˜[YHÛÚÜÈZÙH™Y›Ü™HHÝ\™˜XÙHY[]H\ÈÛ›ÝÛ‹[™ÙYZ[™ÂˆËÈ][ˆHÝ]È\ÈÝÈH™\ÛÛ™HX]ÚÝÜYÛÜšÚ[™ÈˆÝ^\Èš\ÚX›BˆËÈ[œÝXYÙˆ\›š[™È[ÈHXÝ\™H]\ÈÝXHHÜ›Û™È\ÜË‚ˆ]]Èœ›ÛÛ˜\H‹OœÛ˜\ÚÝË™š[™
‹O™œ›ÛY™™\ˆ	ˆQ‘‘‘‘‘‘ŠNÂˆYˆ
œ›ÛÛ˜\OH‹OœÛ˜\ÚÝË™[™

JBˆ‹Oš]™Qœ›ÛÛ˜\ÚÝH˜[ÙNÂˆËÈÖ—Õ’×ÓTÐPNˆH˜]ËQQSH˜[˜XÚÈØ[››Ý›]Üˆ™XY˜XÚÈH][\Ø[\YˆËÈ[XYÙKÛÈ]™\Ù[È›ÝYÚHÚ[™ÛK\Ø[\HÛÛ\[š[Û‹š[YžHBˆËÈ™\ÛÛ™HH™]È[™\È™[ÝËˆHÛ˜\ÚÝ]\È[ÝXÚY8 %Û˜\ÚÝÈ\™BˆËÈH™\ÛÛ™HÕUU[™Ý^HÚ[™ÛK\Ø[\K‚ˆÛÛœÝ›ÛÛ\ØXQ˜[˜XÚÈBˆT‹Oš]™Qœ›ÛÛ˜\ÚÝ	‰ˆ‹O›\ØXTØ[\\ÈOH’×ÔÐSTWÐÓÕS•ÌWÐ’UÂˆ[XYÙIˆÛÝ\˜ÙHH‹Oš]™Qœ›ÛÛ˜\ÚÝÈœ›ÛÛ˜\OœÙXÛÛ™š[XYÙBˆˆ\ØXQ˜[˜XÚÈÈ‹O˜ÛÛÜ”™\ÛÛ™Bˆˆ‹O˜ÛÛÜŽÂˆËÈH™\Ù[ÛÛœÝ[Y\ÈHœ›ÛXY™™\ˆÛ˜\ÚÝ	ÜÈ\ÝÛÜK‚ˆYˆ
×ØÛÜPÙ[œÝ\ÓÛˆ	‰ˆ‹Oš]™Qœ›ÛÛ˜\ÚÝ
BˆÛÜPÙ[œÝ\ÔØ[\Y
‹O™œ›ÛY™™\ˆ	ˆQ‘‘‘‘‘‘ŠNÂˆËÈH˜]ËQQSH˜[˜XÚÈ\ÈÛ™HÙˆH™YHQSH™XY\œËÛÈHY™\œ™YÛX\‚ˆËÈÝ[[™[™È]\Ý[™™Y›Ü™HH›]Ü™XY˜XÚÈØ[\\È]‚ˆYˆ
T‹Oš]™Qœ›ÛÛ˜\ÚÝ
Bˆ›\Ú[™[™ÐÛX\œÊ˜ÛX\ŽˆY™\œ™Y“TÒQ›ÜˆHQSH™\Ù[˜[˜XÚÈŠNÂˆËÈH˜]ËQQSH˜[˜XÚÈ™\Ù[ÈH”SQIÜÈ^[›ÝHQSH[XYÙIÜËˆÜÙBˆËÈÝÜY™Z[™ÈHØ[YH[X™\ˆÚ[ˆHQSHÜ™]ÈÈÛHL\›ÝÈÚYÝÂˆËÈØ\ØØYK[™™XY[™È˜XÚÈHÚÛH[XYÙHÛÝ[[™HÚ[™ÝÈHLŽLˆËÈY™™\ˆ\ÈYˆ]Ù\™HHLŽÌŒœ˜[YK‚ˆËÈ[ˆÔÕ^[Îˆœ›ÛÚYØœ›ÛZYÚ\™HÚ]HÝY\Ý™\ÛÛ™Y[™ˆËÈ\™Ù]ÚYØ\™Ù]ZYÚ\™HÚ]][šÜÈHØÜ™Y[ˆ\Ë[™H[XYÙH[‚ˆËÈœ›ÛÙˆ\È\È™Z]\ˆYˆH™\ÛÛ][ÛˆØØ[H\È[ˆ›Ü˜ÙK‚ˆÛÛœÝZ[Ì—ÝÚYH”Ö
‹Oš]™Qœ›ÛÛ˜\ÚÝÈ‹O™œ›ÛÚYˆ‹O\™Ù]ÚY
NÂˆÛÛœÝZ[Ì—ÝZYÚH”Ê‹Oš]™Qœ›ÛÛ˜\ÚÝÈ‹O™œ›ÛZYÚˆ‹O\™Ù]ZYÚ
NÂˆÛÝ[
‹Oš]™Qœ›ÛÛ˜\ÚÝÈœÝØ\ˆ™\Ù[YHœ›ÛXY™™\ˆ™\ÛÛ™H‚ˆˆœÝØ\ˆ™\Ù[Y˜]ÈQSH
›È™\ÛÛ™HX]ÚY
HŠNÂˆYˆ
\ØXQ˜[˜XÚÊBˆÂˆ˜\œšY\Š‹O˜ÛY‹O˜ÛÛÜ‹’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÔÔ×ÓÔSPSˆ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’U
NÂˆ˜\œšY\Š‹O˜ÛY‹O˜ÛÛÜ”™\ÛÛ™K’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÑÕÓÔSPSˆ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’U
NÂˆšÒ[XYÙT™\ÛÛ™HžßNÂˆ‹œÜ˜ÔÝXœ™\ÛÝ\˜ÙHHÈ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’UHNÂˆ‹™ÝÝXœ™\ÛÝ\˜ÙHHÈ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’UHNÂˆ‹™^[HÈÝŽ›Z[ŠÚY‹O˜ÛÛÜ‹ÚY
KˆÝŽ›Z[ŠZYÚ‹O˜ÛÛÜ‹šZYÚ
KHNÂˆšÐÛY™\ÛÛ™R[XYÙJ‹O˜ÛY‹O˜ÛÛÜ‹š[XYÙK’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÔÔ×ÓÔSPSˆ‹O˜ÛÛÜ”™\ÛÛ™Kš[XYÙK’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÑÕÓÔSPSˆK	œŠNÂˆÛÝ[
œÝØ\ˆQSH˜[˜XÚÈ™\ÛÛ™Y›Üˆ™\Ù[
Ö—Õ’×ÓTÐPJHŠNÂˆB‚ˆËÈÒUTˆH‘PQPÒÈÕSTS”ÈUSˆ[ˆHÖ—Õ’×ÔÕÐTÒRSˆ\›HHÚ[™ÝÂˆËÈÙ]È]È^[Èœ›ÛHHÝØ\ÚZ[ˆ›]™[ÝÈ[™›Ý[™È™YYÈ[H[ˆÜÝˆËÈY[[ÜžH8 %^Ù\HXÝ\™H[œÝ[Y[Ë]™\žHÛ™HÙˆÚXÚØ[ÜÈH™\Ù[YˆËÈœ˜[YHÛˆHÔKˆÛÈH™XY˜XÚÈÝ\š]™\È[ˆ]\›H^XÝHÚ[ˆÛ™HÙˆ[H\ÂˆËÈ\›YY[™H[ˆÐVTÈÓË™XØ]\ÙHHÝØ\ÚZ[ˆ[ˆØ\œžZ[™ÈHXÝ\™H[œÝ[Y[ˆËÈ\È^Z[™È›Üˆ›Ý]È[™]È™XY˜XÚØÛÛ[[ˆ\È›ÝH\›IÜÈÛÜÝ‚ˆËÂˆËÈS‘S•ST•ÍˆT“QQˆQPS•“SQQSˆHS•’T“Ó“QS•‹ÒPÒÓÔÕMIHÑˆBˆËÈÔTUÔ‰ÔÈÔ“ÕÑ”SQKˆH\Ý™[ÝÈ\ÙYÈ[˜ÛYHÖ—ÐÐTT‘WÒÑVX[™ˆËÈÖ—Ð•T”ÕÑST[™ÛÛËÜ^WÜÙ\ÜÚ[Û‹œÚÙ]È›Ý[˜ÛÛ™][Û˜[HÛÈ]ŽBˆËÈ[™ŽÛÜšÈ8 %ÛÈ]™\žH^HÙ\ÜÚ[ÛˆÚ[˜ÙH\MZYHÚÛKYœ˜[YBˆËÈšÐÛYÛÜR[XYÙUÐY™™\˜\ÈHNKŽPˆY[XÜX[™\ˆH]]^]™\žHœ˜[YK[ÈBˆËÈY™™\ˆHÝØ\ÚZ[ˆ™]™\ˆ\Ü^\ËÈXZÙHÛÈÙ^\ÈÛÜšÈ]\™H™\ÜÙYBˆËÈ[™[Ùˆ[Y\È[ˆÝ\‹ˆYX\Ý\™Y[ˆH\ÍHÜ\˜]ÜˆÙ\ÜÚ[Ûˆ]
ŠŒËH\ÈÙ‚ˆËÈHŒËŒÌH\ÈÜ›ÝÙœ˜[YJŠˆ8 %H\™Ù\ÝÚ[™ÛHÛÛ[[ˆ[ˆ][™›Û™HÙˆ]BˆËÈØ[YH
ÛÝÚHL
KˆHÝØ\ÚZ[ˆÙˆ\MØ\ÈZ[È[]H\È][™\ÂˆËÈ\ÈÚ]Ø\ÈØ[˜Ù[[™È]‚ˆËÂˆËÈÛÈH[œÝ[Y[È\™HÜ]žH’QÑÑT‹›ÝžH˜[YN‚ˆËÂˆËÈ
ˆÓÓ•S•SÕTÈ8 %^H™XYU‘T–H™\Ù[Yœ˜[YH[™Ø[››Ý™H™YXÝYÛÈBˆËÈ™XY˜XÚÈ[œÈ›ÜˆHÚÛH[ˆÚ[ˆÛ™HÙˆ[H\ÈÙ]ˆ]\ÈH\ÝˆËÈ™[ÝË[™]\È^XÝHHÙ]ÙˆÛÛœÝ[Y\œÈ]\Ý›ÈÝ\ˆ›YË‚ˆËÈ
ˆQÑKU’QÑÑT‘Q8 %Ö—ÐÐTT‘WÒÑVX
ŽJH[™Ö—Ð•T”ÕÑST
Ž
Kˆ›Ý[™È\ÂˆËÈØ[Y[[HÙ^H\È™\ÜÙYÛÈH™\ÜÈT“TÈH‘PQPÒÈ“ÔˆH”SQTÂˆËÈU“ÓÕÈ
‹Oœ™XY˜XÚÕ[[œ˜[YX[™\œÝXÝ]™X›ÜˆH\œÝ	ÜÈÚÛBˆËÈÚ[™ÝÊKˆÛ™Hœ˜[YHÙˆYÈÛˆHÝ[ØÜ™Y[œÚÝ\È›Ý[™È[™H\œÝ\ÈBˆËÈÙXÛÛ™Û™ËÛÈ™Z]\ˆÜÙ\È[ž][™È[ˆÜ\˜]ÜˆØ[ˆÙYK‚ˆËÂˆËÈ™XYÛ˜ÙHœ›ÛHH[š\›Û›Y[ZÙH]™\žHÝ\ˆXÚ\Ú[ÛˆÙˆ\ÈÚ\H\™NˆBˆËÈ\‹Yœ˜[YHÙ][ˆ\ÈHÞ\ØØ[ÛˆHœ˜[YH][™H™YXØ]H]Ø[ˆÚ[™ÙBˆËÈZY\[ˆXZÙ\ÈÛÈÚ[™ÝÜÈÙˆÛ™H›Ùš[H[˜ÛÛ\\˜X›KˆHSSRPÈ[ˆ\È›ÝBˆËÈÙ][ˆ8 %]\ÈÛÈšY[ÈÛˆH™[™\™\‹Üš][ˆžHHÙ^H[™\ˆ™[ÝË‚ˆËÂˆËÈ]™\žHÛÛœÝ[Y\ˆSÓÈ\ÝÈ]Ù[‹[™HÛÛœÝ[Y\ˆ]š[™È]Ù[ˆ\›YYÚ]ˆËÈ›È^[ÈØ^\ÈÛÈžH˜[YH˜]\ˆ[ˆÚ[™È›Ý[™È]ZY]NˆH™YXØ]H]ˆËÈZ\ÜÙ\ÈHØ\ÙH]\Ý›ÙXÙHH™\Ü›ÝHÚ[[˜ÙH
ÛÝÚHMLJK‚ˆËÂˆËÈHTTÈÔS”ËSQQÓÈUTÈ“ÕÐSÑQS•ËˆØ][™ÈÛˆš\È[ˆ[œÝ[Y[ˆËÈ\›YYˆÚ\ÈHY˜][]›ÈØ]H[ˆ\È›Ú™XÝ^\˜Ú\Ù\Ë™XØ]\ÙH]™\žBˆËÈXÝ\™HØ]H\™HÙ]ÈÛ™HÙˆ\ÙH˜\šXX›\È8 %HØ[YH˜\HÝYÚ[™ËXÛÜBˆËÈ›ÝH™[ÝÈ\ØÜšX™\ËˆÛÈ[™ÜÈÛÜÙH]ˆÖ—Õ’×Ô‘TÑS•ÐSÐVTÏLX\ÈBˆËÈØ[YKXš[˜\žHÛÛ›Û\›H]™\ÝÜ™\ÈH™K\\MÍˆ™YXØ]H^XÝK[™ˆËÈÛÛËÜ\Í—Ü™XY˜XÚ×ÙØ]KœÚ[œÈHØ]HÚ]“ÈXÝ\™H[œÝ[Y[][[™ˆËÈÚXÚÜÈHÛÝ[\œÈ[œÝXYÙˆH^[Ë‚ˆÝ]XÈÛÛœÝ›ÛÛØ[ØXÚY^[ÈBˆ[ŠÖ—Õ’×Ñ”SQWÔÕUÈŠH[ŠÖ—Õ’×Ñ”SQWÑSTŠH[ŠÖ—Õ’×ÔÓTÑSTŠHˆ[ŠÖ—Õ’×ÔÓTÓÓ—Ð“PÒÈŠH[ŠÖ—Õ’×ÔÓTÓÓ—ÑT’ÈŠHˆ[ŠÖ—Õ’×ÔÓTÑ”SQHŠH[ŠÖ—Õ’×ÔÒÖWÐTÖSHŠNÂˆËÈHÛÛ›Û\›Nˆ›Ü˜ÙHH™XY˜XÚÈÛˆU‘T–Hœ˜[YHÚ]]™\ˆ\È\›YYˆ]\ÈBˆËÈÝ\\œÙ]ÙˆH™K\\MÍˆ™YXØ]H
ÚXÚ›Ü˜ÙY]Ú[™]™\ˆÖ—ÐÐTT‘WÒÑVXˆËÈÜˆÖ—Ð•T”ÕÑSTØ\ÈY\™[H˜[YY
KÛÈHÙ\ÜÚ[Ûˆ[ˆÚ]]Ûˆ[™Ù™ˆ\È[‚ˆËÈKÐˆÛˆ^XÝH\È][H8 %[™]\È[ÛÈHÛ›HØ^HÈ^\˜Ú\ÙHHÛ]ˆËÈœ›ÛHH[ˆØ\œžZ[™È›ÈXÝ\™H[œÝ[Y[][ÚXÚ\ÈÚ]HØ]HÙ\Ë‚ˆÝ]XÈÛÛœÝ›ÛÛ™\Ù[[Ø^\ÈH[“ÛŠÖ—Õ’×Ô‘TÑS•ÐSÐVTÈŠNÂˆËÈHQÑH\›K™K\™XY]™\žHœ˜[YKˆ\œÝXÝ]™XÛÝ™\œÈH\œÝ	ÜÈÚÛHÚ[™ÝÎÂˆËÈ™XY˜XÚÕ[[œ˜[YXÛÝ™\œÈHÛÈÜˆ™YHœ˜[Y\ÈÛ™HŽH™\ÜÈ™YYË[™]\ÈBˆËÈœ˜[YH•SP‘Tˆ˜]\ˆ[ˆHÛÝ[ÝÛˆÛÈ]ÛÈ™\ÜÙ\È[ˆ]ZXÚÈÝXØÙ\ÜÚ[ÛˆØ[››ÝˆËÈÚÜ[ˆXXÚÝ\‹‚ˆËÈ‹‹˜[™HYË\™\ÜØ\\™H
ÜÝØY×Ü™\Üš
KÚXÚØ[ÈHœ˜[YHÜˆ™YBˆËÈY\ˆŽÑŽHÚ]\ˆÜˆ›ÝH]ˆ[œÝ[Y[È\™H\›YY‚ˆÛÛœÝ›ÛÛYÙP\›YYH‹O˜\œÝXÝ]™H‹O™œ˜[YHH‹Oœ™XY˜XÚÕ[[œ˜[YHˆYÔ™\ÜÕØ[Ô^[Ê
NÂˆÛÛœÝ›ÛÛÔ™XY˜XÚÈBˆT‹OØ[ÝØ\ÚZ[ˆØ[ØXÚY^[È™\Ù[[Ø^\ÈYÙP\›YYÂˆËÈÛÝ[Y›Ý[™\Ë™XØ]\ÙH[ˆ\›HÚ]›ÈÛÝ[\ˆØ[››Ý™HÚÝÛˆÈ]™BˆËÈ[™ØYÙY
ÛÝÚHMLJH8 %[™\ÈÛ™H\È[š\ÚX›H[ˆHXÝ\™HžHÛÛœÝXÝ[ÛŽ‚ˆËÈHÚÛHÚ[\È]›Ý[™ÈÛˆØÜ™Y[ˆÚ[™Ù\Ë‚ˆYˆ
‹OØ[ÝØ\ÚZ[ŠBˆÂˆYˆ
YÔ™XY˜XÚÊBˆÓÕS•
œ™XY˜XÚÎˆÚÚ\Y
ÝØ\ÚZ[‹›ÈXÝ\™H[œÝ[Y[\›YY
HŠNÂˆ[ÙHYˆ
YÙP\›YY	‰ˆ]Ø[ØXÚY^[È	‰ˆ\™\Ù[[Ø^\ÊBˆÓÕS•
œ™XY˜XÚÎˆ˜[ˆ™XØ]\ÙHŽÑŽH\›YY]ŠNÂˆ[ÙBˆÓÕS•
œ™XY˜XÚÎˆ˜[ˆ›ÜˆHÛÛ[[Ý\ÈXÝ\™H[œÝ[Y[ŠNÂˆBˆÝ]XÈ›ÛÛØZYÚHH˜[ÙNÂˆYˆ
‹OØ[ÝØ\ÚZ[ˆ	‰ˆ
Ø[ØXÚY^[È™\Ù[[Ø^\ÊH	‰ˆ\ØZYÚJBˆÂˆØZYÚHHYNÂˆœš[ŠÝ\œ‹ˆ–Ýš×HÝØ\ÚZ[ˆ™\Ù[Ú]HÓÓ•S•SÕTÈXÝ\™H[œÝ[Y[\›YY
Üˆ‚ˆÖ—Õ’×Ô‘TÑS•ÐSÐVTÏLJNˆH™\Ù[‘PQPÒÈTÈ•S“’S‘ÈÓˆU‘T–H‚ˆ‘”SQK™XØ]\ÙHÜÙH[œÝ[Y[ÈØ[ÈHœ˜[YHÛˆHÔKˆ\È[ˆ‚ˆœ^\È›Üˆ›Ý™\Ù[]È[™]È™XY˜XÚØÛÛ[[ˆ\È“Õ\È\›IÜÈ‚ˆ˜ÛÜÝ8 %ZÙHHœ˜[YK][YHKÐˆÚ]Ý]Û™KˆŽÑŽH[Û™H›ÈÛ™Ù\ˆÈ\È‚ˆŠ\ÍŠNˆ^H\›HH™XY˜XÚÈ›ÜˆHœ˜[Y\È^H™YY—ˆŠNÂˆB‚ˆËÈ[ÈTÈÓÕ	ÜÈ™XY˜XÚÈY™™\‹›ÝHÚ\™YÛ™NˆÚ]Hœ˜[YH[ˆ›YÚBˆËÈÚ[™ÝÈ\È›Ý™XÙ\ÜØ\š[H™]ÚYH™]š[Ý\Èœ˜[YIÜÈ^[ÈY]‚ˆœ˜[YTÛÝ	ˆ™XÈH‹O™œ˜[Y\ÖÔ‹O™œ˜[YTÛÝNÂˆYˆ
Ô™XY˜XÚÊBˆÂˆËÈ\ÍˆÜ]\ÈÙ™ˆH[Ø^\Ë[Ûˆ]È]Ý\š]™\ÈÚ[™]™\ˆHXÝ\™BˆËÈ[œÝ[Y[\È\›YY[™]\ÈHÚÛKYœ˜[YHÛÜKÛÈ]\È]ÈÝÛˆÛ\ÜËˆ[ˆBˆËÈZ[ˆ^H[ˆ\ÈÛ\ÜÈ]\Ý™XY‘T“È8 %H›Û‹^™\›ÈÛ™HØ^\È[ˆ[œÝ[Y[\ÂˆËÈ\›YY]H[ˆY›ÝYX[ˆÈ\›KÚXÚ\È^XÝHHY™XÝ\Í‚ˆËÈ›Ý[™[ˆ^WÜÙ\ÜÚ[Û‹œÚ‚ˆÜTÙYÈÙÊÑÜ™XY˜XÚÊNÂˆ˜\œšY\Š‹O˜ÛYÛÝ\˜ÙK’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÔÔ×ÓÔSPSˆ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’U
NÂˆšÐY™™\’[XYÙPÛÜHÛÜ^ßNÂˆÛÜKš[XYÙTÝXœ™\ÛÝ\˜ÙHHÈ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’UHNÂˆÛÜKš[XYÙQ^[HÈÚYZYÚHNÂˆšÐÛYÛÜR[XYÙUÐY™™\Š‹O˜ÛYÛÝ\˜ÙKš[XYÙK’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÔÔ×ÓÔSPSˆ™XËœ™\Ù[˜Y™™\‹K	˜ÛÜJNÂˆBˆËÈHÝØ\ÚZ[ˆ›]ÛÙ\ÈTÕ[ˆHÛÛ[X[™Y™™\‹Y\ˆH™XY˜XÚÈÛÜHÚ[‚ˆËÈ›Ý\™H™\Ù[™XØ]\ÙH]\ÈÚ]HÝX›Z]	ÜÈÙ[X\Ü™HÚYÛ˜[ÈÛ‹‚ˆËÂˆËÈ“ÕPÔURT’S‘ÈTÈHÓ“HÐQ‘HÐVHÈ“Õ‘TÑS•ˆ[ˆXÜ]Z\™Y[XYÙHÛÛY\ÈÚ]BˆËÈÙ[X\Ü™HH™\Ù[][Ûˆ[™Ú[™HÚ[ÚYÛ˜[[™HÛÛ˜XÝ\È]ÛÛY][™ÂˆËÈØZ]ÈÛˆ]ÈX˜[™ÛˆHœ˜[YHY\ˆXÜ]Z\š[™È[™]Ù[X\Ü™H\ÈYÚYÛ˜[YˆËÈÚ]›ÈØZ]\‹ÛÈH™^XÜ]Z\™H™]\Ù\È][ˆ[ˆ[YØ[Ý]Kˆ›ÝØ^\ÈÙ‚ˆËÈX˜[™Ûš[™ÈHœ˜[YH\™H\™HÛ›ÝØX›H‘Q“Ô‘HHXÜ]Z\™H8 %Ö—Õ’×Ó“×ÔÕP“RU™XÛÜ™ÂˆËÈHœ˜[YH[™^XÝ]\È›Û™HÙˆ]
HÙZ[[™È\›JK[™HÛÛ[X[™Y™™\ˆ]\È›ÝˆËÈ™XÛÜ™[™ÈØ[››ÝØ\œžHH›]8 %ÛÈHXÜ]Z\™H\ÈÚ[\H›ÝXYKˆÖ—Õ’×Ó“×ÔÕP“RUˆËÈ\™Y›Ü™H™\Ù[È›Ý[™È][[ˆ\È\›KÚXÚ\ÈÛÜœ™XÝ[™ÛÛœÚ\Ý[Ú]ˆËÈÚ]]\›H[™XYHØÝ[Y[Îˆ]ÈXÝ\™H\ÈÛ›ÝÚ[™ÛH[˜[Y‚ˆÝ]XÈÛÛœÝ›ÛÛ›ÔÝX›Z]\›HH[“ÛŠÖ—Õ’×Ó“×ÔÕP“RUŠNÂˆYˆ
‹OØ[ÝØ\ÚZ[ˆ	‰ˆ‹Oœ™XÛÜ™[™È	‰ˆ[›ÔÝX›Z]\›JBˆÂˆËÈH]\˜›ÞÛX\‹H\ÜXÝYš]›][™HÝ™\›^H8 %[Ùˆ]™XØ]\ÙBˆËÈH™\Ù[ˆ\ÈÛ™H™YÚ[Ûˆ\È˜\ˆ\ÈHš^\ÈÛÛ˜Ù\›™Y‚ˆÜTÙYÈÙÊÑÜ™\Ù[
NÂˆ™XÛÜ™ÝØ\ÚZ[›]
ÛÝ\˜ÙKÚYZYÚ
NÂˆBˆ[ÙHYˆ
‹OØ[ÝØ\ÚZ[ŠBˆÛÝ[
œÝØ\ˆ›ÈXÜ]Z\™H
Ö—Õ’×Ó“×ÔÕP“RUÜˆ›Ý[™È™XÛÜ™Y
H8 %›Ý[™È™\Ù[YŠNÂ‚ˆËÈÚ]\Èœ˜[YHÐTË™XÛÜ™Y™^ÈH^[È]›ÙXÙYˆ]™\žH™\Ù[\ÚYBˆËÈ[œÝ[Y[™[ÝÈ™XYÈ\È[™›Ý‹O™œ˜[YXØ‹O™˜]Ñš[™Ù\œš[ÚXÚœ›ÛBˆËÈ\™HÛˆ\ØÜšX™HHœ˜[YH™Z[™È™XÛÜ™Y˜]\ˆ[ˆHÛ™H™Z[™ÈÚÝÛ‹‚ˆ™XË™œ˜[YHH‹O™œ˜[YNÂˆ™XË™˜]ÜÈH‹O™˜]ÜÕ\Ñœ˜[YNÂˆ™XË™\XÙ\ÈH‹O™\XÙ\Õ\Ñœ˜[YNÂˆ™XË™˜]Ñš[™Ù\œš[H‹O™˜]Ñš[™Ù\œš[Âˆ™XË˜Ø[Y\˜Qš[™Ù\œš[H‹O˜Ø[Y\˜Qš[™Ù\œš[Âˆ™XËÚYHÚYÂˆ™XËšZYÚHZYÚÂˆ™XË˜ž]\ÈHÚ^™WÝ
ÚY
H
ˆZYÚ
ˆÂˆ™XËœ™\Ù[X›HHYNÂˆËÈ™XÛÜ™YÚ]H^[Ë[™™XY]H™]\™H™[ÝÈ˜]\ˆ[ˆ™KY\š]š[™È]‚ˆ™XËš\Ô^[ÈHÔ™XY˜XÚÎÂˆYˆ
Ô™XY˜XÚÊBˆ™XËœ^[œ˜[YHH‹O™œ˜[YNÂ‚ˆÝX›Z]œ˜[YJ
NÂˆËÈ[[YYX][HY\ˆHÝX›Z][™™Y›Ü™HH™[˜ÙHØZ]™[ÝÎˆH™\Ù[ØZ]ÈÛ‚ˆËÈHÝX›Z]	ÜÈÑSPTÔ‘KÛÈHÚ[™ÝÈØ[ˆ™H[™Y\Èœ˜[YHÚ[HHÔH\ÂˆËÈÝ[™]\š[™ÈH™]š[Ý\ÈÛ™Kˆ]Ü™\š[™È\ÈH][H8 %H™XY˜XÚÈ]ˆËÈØ[››ÝÚÝÈHœ˜[YH[[]Èž]\È]™H\œš]™Y[ˆÜÝY[[ÜžK‚ˆYˆ
‹OØ[ÝØ\ÚZ[ŠBˆ™\Ù[ÝØ\ÚZ[Š
NÂˆÛÛœÝ[™\Ù[ÛÝH™]\™SÛ\Ýœ˜[YJ
NÂ‚ˆËÈY˜[˜ÙHHš[™È›ÜˆH™^œ˜[YKˆ]\[œÈT‘KY\ˆHØZ]X›Ý™KÛÂˆËÈ]Ú[ˆ™YÚ[‘œ˜[YXXÚÜÈ\\ÈÛÝ]È™[˜ÙH\È™Y[ˆØœÙ\™YÈÚYÛ˜[8 %ˆËÈÚXÚ\ÈÚ]XZÙ\È™\Ù][™È]ÈÛÛ[X[™Y™™\ˆ[™™]\Ú[™È]È\™[˜H™YÚ[Û‚ˆËÈYØ[Ú]Ý]HÙXÛÛ™ØZ]]HÜÙˆHœ˜[YK‚ˆ‹O™œ˜[YTÛÝH
‹O™œ˜[YTÛÝ
ÈJH	H‹O™œ˜[Y\Ò[‘›YÚÂ‚ˆËÈ™\Ù]H™XÛÜ™Yœ˜[YIÜÈÝÛˆXØÝ[][]ÜœÈ\™H˜]\ˆ[ˆY\ˆHÝ]È[™N‚ˆËÈ^H™[Û™ÈÈHœ˜[YH]\È\Ý™Y[ˆÝX›Z]Y[™HÝ]È[™H™[ÝÈ\ÂˆËÈ›ÝÈX›Ý]HY™™\™[Û™K‚ˆ‹O™˜]Ñš[™Ù\œš[HÂˆ‹O˜Ø[Y\˜Qš[™Ù\œš[HÂˆ‹O™\XÙ\Õ\Ñœ˜[YHHÂ‚ˆËÈÜ›ÝÈH\™[˜HT‘K]Hœ˜[YH›Ý[™\žK˜]\ˆ[ˆ[œÚYHH™^œ˜[YIÜÂˆËÈš\œÝ˜]Ëˆ›ÝÙˆ\ÙHYH]™\žHœ˜[YHÝ[[ˆ›YÚ™Y›Ü™H^HÝXÚˆËÈ[ž][™ËÚXÚ™Y›Ü™H\ŒÈHØ[\‰ÜÈ™[˜ÙHØZ]Y›Üˆ[KˆÙYBˆËÈÜ›ÝÐ\™[˜RY“™YYY›ÜˆÚHHÛXÙ[Y[Ø\ÈHYX\Ý\™[Y[Y™XÝˆ]Ú\™ÙYBˆËÈ]šXÙK]ØZ][™[ˆ[ØØ][ÛˆÈÝ\˜‚ˆÜ›ÝÐ\™[˜RY“™YYY

NÂˆËÈŒH
\LLJNˆHÚ\™YÝX‹X\™[˜HÜ›ÝÜÈ[™\ˆHØ[YH[K[™[ˆ\ÂˆËÈœ˜[YIÜÈ™K^™\›È\ÈÜÝYˆHÔ‘Tˆ\ÈØYX™X\š[™È8 %Ü›ÝÝ\Ý›Þ\ÈHY™™\‚ˆËÈHÛÜšÙ\œÈÜš]H[ËÛÈ]˜Z[œÈš\œÝ[™H\Ü]Ú]›ÛÝÜÈ\ÈYØZ[œÝˆËÈH™]ÈÛ™K‚ˆÜ›ÝÔÚ\™Y\™[˜RY“™YYY

NÂˆ™^™\›×Ñ\Ü]Ú

NÂˆËÈØ[YHÚ]K[™›ÜˆHÝÜ™HH™X\ÛÛˆ\ÈÝ›Û™Ù\Žˆ]ÈÙ™œÙ]È\™H™XÛÜ™Y[ÂˆËÈÛÛ[X[™Y™™\œËÛÈ™]\Ú[™È]ÈY[[ÜžH™Y›Ü™HHÔH\ÈÛ™H[™È[ˆ[‹Y›YÚˆËÈ˜]ÈÛÛYX›ÙH[ÙIÜÈ™\XÙ\Ë‚ˆ\œÚ\ÝXZ[[˜[˜ÙJ
NÂ‚ˆËÈ›Ý[™ÈÈÚÝÈY]8 %Hš\œÝœ˜[YHÙˆH[ˆÚ]ÛÈÛÝË™XØ]\ÙHHÙXÛÛ™ˆËÈÛÝ\È™]™\ˆ™Y[ˆÝX›Z]YˆÛ™Hœ˜[YHÙˆH[‹‚ˆYˆ
™\Ù[ÛÝ
Bˆ™]\›ŽÂˆœ˜[YTÛÝ	ˆ™\ÈH‹O™œ˜[Y\ÖÜ™\Ù[ÛÝNÂˆYˆ
\™\Ëœ™\Ù[X›JBˆ™]\›ŽÂˆÛÛœÝZ[Ì—ÝÚYHH™\ËÚYZYÚHH™\ËšZYÚÂˆÛÛœÝÚ^™WÝž]\ÈH™\Ë˜ž]\ÎÂˆËÈHÓÔHVTÕÈ“ÔˆHS”Õ•SQS•ËS‘Ó“H“ÔˆSH
\LË[ˆ][HKŒÊK‚ˆËÂˆËÈ™\Ëœ™\Ù[›X\Y\ÈÔÕÐÐPÒQ
ÙYH™XY˜XÚÓY[[ÜžT›ÜÈ8 %]Ø\ÈXYHØXÚYˆËÈ[X™\˜][K[™Ö—Õ’×Ô‘PQPÒ×ÕSÐPÒQLH\ÈÝ[H\›H›Üˆ]
KÛÈ™XY[™ÂˆËÈ]ÛÜÝÈÚ]™XY[™È[žHÝ\ˆÜÝY™™\ˆÛÜÝËˆ]™\ž][™ÈÝÛœÝ™X[HÙˆ\™BˆËÈ]ÛÚÜÈ]HXÝ\™H8 %Hœ˜[YHÝ]ËHH[\ËH›XÚËÙ\šÂˆËÈšYÙÙ\œËH[šY›Ü›KXÛÛÝ\ˆÙ[œÝ\È8 %Ø\È™XY[™È™\Ù[^[Ø[™ˆËÈÜÝÔ™\Ù[^[Ø[ˆXZÙ\È]ÈÝÛˆÛÜH[ÈHÚ[™ÝÉÜÈ˜XÚÈY™™\ˆ[™\‚ˆËÈ]ÈÝÛˆØÚËˆÛÈÛˆH[ˆÚ]›ÈXÝ\™H[œÝ[Y[\›YYH[\›YYX]HY™™\‚ˆËÈØ\ÈËHPˆÛÜYY\ˆœ˜[YH›Üˆ›Ý[™ÎˆÈPˆÙˆ˜Y™šXÈÚ\™HËHÙ\Ë‚ˆËÂˆËÈ\È›ÝÈH^[ÈÈ™XY[™]Ú[È]HX\YY™™\ˆ\™XÝKˆBˆËÈÛÛ™][Ûˆ\ÈHQSSÔ–HTK›ÝÚXÚ[œÝ[Y[È\™H\›YY[™]\ÂˆËÈ[X™\˜]NˆØ][™ÈÛˆH[œÝ[Y[ÈÛÝ[YX[ˆHY˜][ÛÛ™šYÝ\˜][ÛˆÛÚÈBˆËÈÛÙH]›ÈØ]H[ˆ\È›Ú™XÝ]™\ˆ^\˜Ú\Ù\Ë™XØ]\ÙH]™\žHXÝ\™HØ]H\™BˆËÈ
Hœ˜[YH[\HLÈÛÜœ™[][Û‹Ö—Õ’×ÔÓTÊŠHÙ]ÈÛ™HÙˆ[KˆØ][™ÈÛ‚ˆËÈÚ]\ˆH™XY˜XÚÈ\ÈØXÚYÙY\ÈHY˜][]HÓ“H][ˆ]™\žBˆËÈÜ™[˜\žH[‹[™X]™\ÈHÝYÚ[™ÈÛÜHÚ\™H]\ÈÙ[Z[™[H™YYY8 %BˆËÈ[˜ØXÚY˜[˜XÚËÚ\™HÙ]™\˜[ÚÛKYœ˜[YHØ[ÜÈÛÝ[XXÚ™HBˆËÈÜš]KXÛÛXš[™Y™XYˆÖ—Õ’×Ô‘TÑS•ÔÕQÒS‘ÏLX\ÈHØ[YKXš[˜\žHÛÛ›Û\›K‚ˆÝ]XÈÛÛœÝ›ÛÛÝYÚ[™ÐÛÜHHY×Ü™XY˜XÚÐØXÚY[“ÛŠÖ—Õ’×Ô‘TÑS•ÔÕQÒS‘ÈŠNÂˆÛÛœÝZ[Ý
ˆH[ŽÂˆËÈ™\Ëš\Ô^[Ø“ÕÔ™XY˜XÚØ8 %ÙYHHšY[ˆHÛÈYÜ™YHÛˆ]™\žHœ˜[YHÙ‚ˆËÈH[ˆÚÜÙH™YXØ]H™]™\ˆÚ[™Ù\È[™\ØYÜ™YHÛˆ^XÝHHœ˜[Y\È[ˆŽÑŽBˆËÈ™\ÜÈÝ˜Y\ËÚXÚ\™HHœ˜[Y\È\ÈÚÛH][H\ÈX›Ý]‚ˆYˆ
\™\Ëš\Ô^[ÊBˆÂˆËÈHÝØ\ÚZ[ˆ\›HÚ]›ÈXÝ\™H[œÝ[Y[ˆ\™H\™H›ÈÜÝ^[È[™ˆËÈ›Ý[™ÈÝÛœÝ™X[HÙˆ\™H\È[ž][™ÈÈÛÚÈ]ˆ]™\ž][™È™[ÝÈ\ÈÚ[ˆËÈ8 %Hœ˜[YHÝ]ËH[\ËH›XÚÈšYÙÙ\œÈ8 %\ÈÝX\™YÛˆ[™ˆËÈ\ÈÛÝ[\ˆ\ÈÚ]XZÙ\ÈHÚÚXÙHš\ÚX›H[ˆHÙ[œÝ\È˜]\ˆ[‚ˆËÈ[™™\˜X›Hœ›ÛH[ˆXœÙ[˜ÙK‚ˆÛÝ[
œÝØ\ˆ™\Ù[Y›ÝYÚHÝØ\ÚZ[ˆ
›ÈÜÝ™XY˜XÚÊHŠNÂˆBˆ[ÙBˆÂˆ›Ù”ØÛÜHÜ
	™×Ü›Ù‹œ™XY˜XÚÊNÂˆYˆ
ÝYÚ[™ÐÛÜJBˆÂˆYˆ
‹Oœ™\Ù[^[ËœÚ^™J
Hž]\ÊBˆ‹Oœ™\Ù[^[Ëœ™\Ú^™Jž]\ÊNÂˆY[XÜJ‹Oœ™\Ù[^[Ë™]J
K™\Ëœ™\Ù[›X\Yž]\ÊNÂˆH‹Oœ™\Ù[^[Ë™]J
NÂˆBˆ[ÙBˆÂˆH™\Ëœ™\Ù[›X\YÂˆBˆËÈHS•T’PS•ÒPÒÑQUTˆSˆTÔÕSQQˆHž]\È[ˆ\ÈÛÝ]\Ý™HBˆËÈœ˜[YH\ÈÛÝ\ØÜšX™\Ë‚ˆËÂˆËÈ]^\ÝÈ™XØ]\ÙHHØš[Ý\ÈØ]H›ÜˆH\MÍˆÙ™‹XžK[Û™H8 %››È\œÝœ˜[YBˆËÈX^H™Hž]KZY[XØ[ÈHŽHØ\\™Hˆ8 %Ø\È•RS•SˆQÐRS”ÕHSP‘TUSBˆËÈ”“ÒÑSˆ•RSS‘TÔÑQˆHÝ[HÛÝÙ\ÈÛ[ˆŽKY\˜Hœ˜[YK]HŽBˆËÈ™\ÜÈ\›\Èœ˜[Y\Ò[‘›YÚ
È˜Ùˆ[H[™HØ\\™HH\ÈÛ›HÛ™NÈBˆËÈÝ[H™XY[™YÛˆHÚX›[™ËˆHÛÛ[Ø[˜\žHZ[YY]Û™H\Y˜XÝØ[››ÝˆËÈÛÝ™\ˆHÚ[™ÝÈ
ÛÝÚHÌ8 %[™HÛÛ›Û\ÈÚ]ØZYÛË›Ý™X\ÛÛš[™ÊK‚ˆËÂˆËÈ\È\ÈH\š]™YYœ›ÛKX[‹Z[˜\šX[›Ü›H[œÝXYˆHÛÝÝ[\ÈHœ˜[YBˆËÈÚÜÙH^[È]ÛË[™H\ØYÜ™Y[Y[\ÈHY™XÝžHÛÛœÝXÝ[Û‹Ú]›ÂˆËÈ›Ý]K›ÈØ[Y\˜H[™›ÈÙXÛÛ™\Y˜XÝ™\]Z\™Yˆ]ÛÜÝÈÛ™HÛÛ\\š\ÛÛˆ\‚ˆËÈ™\Ù[Yœ˜[YKˆÚÝÛˆØ\X›HÙˆš\š[™Îˆ™]™\[™ÈHÝX\™X›Ý™HÂˆËÈÔ™XY˜XÚØXZÙ\È]š\™HÛˆHœ˜[YHY\ˆ]™\žHŽ™\ÜË‚ˆYˆ
™\Ëœ^[œ˜[YHOH™\Ë™œ˜[YJBˆÂˆÛÝ[
”‘TÑS•VSÈT‘H”“ÓHHQ‘‘T‘S•”SQH8 %HÝ[H™XY˜XÚÈÛÝŠNÂˆÝ]XÈ[YHÂˆYˆ
ZÙSÛ™JY
JBˆœš[ŠÝ\œ‹ˆ–Ýš×HHH™\Ù[ÛÝ\ØÜšX™\Èœ˜[YH	[H]ÛÈœ˜[YH	[IÜÈ‚ˆœ^[È8 %]™\žHXÝ\™H[œÝ[Y[™XY[™È\Èœ˜[YH\ÈÛÚÚ[™È‚ˆ˜]HÜ›Û™ÈÛ™Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™œ˜[YKˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ëœ^[œ˜[YJNÂˆBˆÜÝÔ™\Ù[^[ÊÚYKZYÚJNÂˆËÈHYË\™\ÜØ\\™IÜÈœ˜[YJÊKÛÜYYÝ]ÙˆH™XY˜XÚÈ\™H
HÛÝˆËÈ\È™]\ÙY™^œ˜[YJNÈH›Ë[Ü[›\ÜÈHØ\\™H\ÈØZ][™È›Üˆ^[Ë‚ˆYÔ™\ÜÓÙ™™\”^[ÊÚYKZYÚK™\Ë™œ˜[YJNÂˆB‚ˆËÈÖ—Õ’×ÔÓTÓÓ—Ð“PÒÖÏ\ÝH8 %[\HÚÛH™\ÛÛ™HÚZ[ˆÙˆHœ˜[YHHXÝ\™BˆËÈQQÛ‹šYÙÙ\™YžHHXÝ\™HZ[™Ë‚ˆËÂˆËÈHšY]ËY\[™[ÚÛKYœ˜[YH›XÚÈ\ÈHÜ	ÜÈÜ™[™\š[™ÈY™XÝ[™]\ÂˆËÈ™]™\ˆ™Y[ˆØ\\™Y™XØ]\ÙHÖ—Õ’×ÔÓTÑSTš\™\ÈÛˆHœ˜[YH•SP‘Tˆ[™\ÂˆËÈ]™[\[œÈÚ[ˆH[X[ˆ\›œÈHØ[Y\˜Kˆ\ÚÚ[™È[ˆÜ\˜]ÜˆÈ]Hœ˜[YH[™^ˆËÈ\È›ÝH™\]Y\Ý[ž[Û™HØ[ˆ[š[ÛÈ]™\žH™\ÜÙˆ\ÈY™XÝ\È™Y[ˆBˆËÈ›XÚÈœ˜[YH[Û™H8 %ÚXÚ\ÈÛÛœÚ\Ý[Ú]]™\žH\ÜÈ™Z[™ÈÜ›Û™È[™Ú]ˆËÈ^XÝHÛ™H™Z[™ÈÜ›Û™È
H™X\ÛÛˆÖ—Õ’×ÔÓTÑST^\ÝÈ][
K‚ˆËÂˆËÈHšYÙÙ\ˆ\ÈHS”ÒUSÓ‹›ÝH™\ÚÛ[™]\ÈÚ]XZÙ\È]\ØX›N‚ˆËÈ\È[[YH™\Ù[È[HÙˆYÚ][X][H›XÚÈœ˜[Y\È\š[™È›ÛÝ[™ØY[™ËˆËÈÛÈ˜ÛÝ™\˜YÙH™[ÝÈ	Hˆ[Û™HÛÝ[š\™HÛˆHš\œÝÛ™H[™[\HÚZ[ˆ›Ø›ÙBˆËÈØ[Ëˆ™\]Z\š[™ÈHUœ˜[YHš\œÝYX[œÈH[\[™ÈÛˆHœ˜[YHÚ\™HBˆËÈÛÜšÚ[™ÈXÝ\™HÝÜYÛÜšÚ[™ËÚXÚ\ÈHÛ›Hœ˜[YH]Ø[ˆ\Ý[™ÝZ\ÚBˆËÈ\Ý\Ù\Ë‚ˆËÂˆËÈ›Ý™\ÚÛÈ\™HÙ]X›K[™]\ÈÚ]XZÙ\ÈH[œÝ[Y[\ÝX›H]ˆËÈ[ˆHš\œÝ™\œÚ[Ûˆ›ÛY\›Z[™È[™š\š[™È[ÈÛ™HY‹Ù[ÙKÛÈHœ˜[YBˆËÈÛÝ[™]™\ˆÈ›Ý8 %[™]ÈÜÚ]]™HÛÛ›Û
HNIH›ÛÜ‹ÚXÚÚÝ[š\™HÛ‚ˆËÈ\ÜÙ[X[H[žHœ˜[YJHØ]Ú[[›ÝYÚHÚÛH›ÛÝ™XØ]\ÙH™XXÚ[™ÈHš\™BˆËÈœ˜[˜ÚÝ[™\]Z\™YÛÝ™\˜YÙH[™\ˆH\™XÛÙYŒ	H\›Z[™È˜\‹ˆ[ˆ[œÝ[Y[ˆËÈÚÜÙHÛÛ›ÛØ[››Ý™XXÚ]ÈÝÛˆšYÙÙ\ˆ\È›Ý™Y[ˆÚÝÛˆØ\X›HÙˆš\š[™ÂˆËÈ
ÛÝÚHÌ
KˆÚ]Ö—Õ’×ÔÓTÓÓ—Ð“PÒÏNNHÖ—Õ’×ÔÓTÓÓ—Ð“PÒ×ÓULŒHÙXÛÛ™ˆËÈ]œ˜[YHÙˆ[žH[ˆš\™\È]‚ˆËÈ[™\™H\ÈHT‘ÐTÛˆÝ[\\ÛÙ\Ë™XØ]\ÙHH\›Z[™ÈÙÚXÈ™KX\›\ÈÛ‚ˆËÈ]™\žH]œ˜[YH[™\™Y›Ü™H\È›È˜]\˜[›Ý[™ˆ]ÈÝÛˆÜÚ]]™HÛÛ›ÛˆËÈ›Ý™Y]H^[œÚ]™HØ^NˆHNIH›ÛÜˆš\™\ÈÛˆ\ÜÙ[X[H]™\žHœ˜[YKÚXÚˆËÈ[\Y
ŠŽKÌÈ\ÊŠˆ[™™Yš[YH\œÈÚÜÙH^]\Ý[ÛˆÚ[È\ÈXXÚ[™IÜÂˆËÈÚ[ˆHY™XÝ]^\ÝÈÈØ]Ú\[œÈH[™[Ùˆ[Y\È[ˆHÙ\ÜÚ[Û‹ÛÈBˆËÈÝÈØ\ÛÜÝÈ›Ý[™È™X[[™\›œÈHZ\Ë\Ù]™\ÚÛœ›ÛHHš[Y\ÚÈ[ÈBˆËÈ™]ÈØ\ÝYš[\ËˆÖ—Õ’×ÔÓTÓÓ—Ð“PÒ×ÓPV˜Z\Ù\È]‚ˆÝ]XÈÛÛœÝÚ\ŠˆÛ›XÚÑ[ˆH[ŠÖ—Õ’×ÔÓTÓÓ—Ð“PÒÈŠNÂˆÝ]XÈÛÛœÝÝX›H]ÝBˆ[ŠÖ—Õ’×ÔÓTÓÓ—Ð“PÒ×ÓUŠHÈ]ÙŠ[ŠÖ—Õ’×ÔÓTÓÓ—Ð“PÒ×ÓUŠJHˆŒŒÂˆÝ]XÈ[\\ÛÙ\ÓYBˆ[ŠÖ—Õ’×ÔÓTÓÓ—Ð“PÒ×ÓPVŠHÈ]ÚJ[ŠÖ—Õ’×ÔÓTÓÓ—Ð“PÒ×ÓPVŠJHˆÂˆÝ]XÈ›ÛÛØ]Ó]œ˜[YHH˜[ÙNÂˆÝ]XÈ[Û›XÚÐYÙ]HŽÈËÈH˜[œÚ][Ûˆœ˜[YK[™H™^Û™BˆËÈÖ—Õ’×ÔÓTÓÓ—ÑT’ÏOYX[“[XOˆ8 %HØ[YHšYÙÙ\ˆÛˆHY]šXÈHY™XÝPÕPSBˆËÈ[Ý™\Ë[™][\ÈH”’QÒ™Y™\™[˜ÙHÚZ[ˆÈÚ]™\ÚYHH\šÈÛ™K‚ˆËÂˆËÈHš\œÝXY\ÜÈ[ˆÈÝÙY\HØ[Y\˜H[ˆÝ[Ü™YZÈ›Ý[™HY™XÝˆËÈ[[YYX][H[™ÓTÓÓ—Ð“PÒÈÛÝ[›Ýš\™HÛˆ]ˆÝÙY\[™ÈHØ[Y\˜H›ÝYÚˆËÈŒÍŒYÜ™Y\ÈÝÚ[™ÜÈH™\Ù[Yœ˜[YIÜÈYX[ˆ[Z[˜[˜ÙH™]ÙY[ˆŒÈ[™BˆËÈÚ[H]ÈÓÕ‘TQÑHÝ^\ÈÌMÍIKˆHœ˜[YHÛÙ\È™\žH\šË›Ý[\KÛÈBˆËÈÛÝ™\˜YÙH›ÛÜˆÙˆIH™]™\ˆš\ËˆÛÝ™\˜YÙHØ\ÈHšYÚY]šXÈ›ÜˆBˆËÈÜ\˜]Ü‰ÜÈ™\Ü8 %HÚÛKYœ˜[YH›XÚÈ8 %[™]\ÈHÜ›Û™ÈÛ™H›ÜˆH[™ÂˆËÈ]\ÈXÝX[HYX\Ý\˜X›H\™NÈ›Ý\™HÙ\™XØ]\ÙHHÛÈ™\ÚÛÂˆËÈ[œÝÙ\ˆY™™\™[]Y\Ý[ÛœÈ[™[ˆ[œÝ[Y[ÚÜÙHYX[š[™ÈÚ[[HÚ[™ÙYˆËÈÛÝ[[˜[Y]H]™\žH[ˆZÙ[ˆÚ]]‚ˆËÂˆËÈHRTˆ\ÈHÚ[ˆÛ™H\šÈÚZ[ˆ\ÈÛÛœÚ\Ý[Ú]\È\ÜÈ\Èœ›ÚÙ[ˆ‚ˆËÈ[™Ú]HØÙ[™H™X[H\È\šÈ\™HŽÈHœšYÚÚZ[ˆœ›ÛHHØ[YHØØ][Û‚ˆËÈÙXÛÛ™È]\‹Ú]HØ[YHÝ\™˜XÙ\È]HØ[YHY™\ÜÙ\Ë\ÈHÛÛ›Û]ˆËÈÙ\\˜]\È[H
ÛÝÚHLÌÈ8 %Û™Hœ˜[YHÙˆ[ˆ[š[X]YØÙ[™H\ÈÛ™HØ[\JKˆÛÈBˆËÈ\šÈ\\ÛÙHÝÙ\ÈHœšYÚ™Y™\™[˜ÙK[™H™^œ˜[YH]™KX\›\È^\È]‚ˆÝ]XÈÛÛœÝÚ\ŠˆÛ‘\šÑ[ˆH[ŠÖ—Õ’×ÔÓTÓÓ—ÑT’ÈŠNÂˆÝ]XÈÛÛœÝÝX›H\šÓ][XHBˆ[ŠÖ—Õ’×ÔÓTÓÓ—ÑT’×ÓUŠHÈ]ÙŠ[ŠÖ—Õ’×ÔÓTÓÓ—ÑT’×ÓUŠJHˆŒŒÂˆÝ]XÈ[\šÑ\\ÛÙ\ÓYBˆ[ŠÖ—Õ’×ÔÓTÓÓ—ÑT’×ÓPVŠHÈ]ÚJ[ŠÖ—Õ’×ÔÓTÓÓ—ÑT’×ÓPVŠJHˆÎÂˆÝ]XÈ›ÛÛØ]ÐœšYÚœ˜[YHH˜[ÙNÂˆÝ]XÈ›ÛÛÝÙPœšYÚ™Y™\™[˜ÙHH˜[ÙNÂ‚ˆ›ÛÛ›XÚÕ˜[œÚ][ÛˆH˜[ÙNÂˆYˆ
	‰ˆ
Û›XÚÑ[ˆÛ‘\šÑ[ŠJBˆÂˆËÈØ[\Y]™\žHM^[ˆ\È[œÈÛˆH™\Ù[]Ùˆ]™\žHœ˜[YK[™ˆËÈH]X[]H\ÈHÚÛKYœ˜[YHœ˜XÝ[Ûˆ]HKZ[‹LMˆØ[\H\Ý[X]\ÈÈ˜\‚ˆËÈ™]\ˆ[ˆHH\˜Ù[YÙHÚ[È[ž[Û™HØ\™\ÈX›Ý]\™K‚ˆZ[Ý]HÙY[ˆH[XHHÂˆ›Üˆ
Ú^™WÝHHÈH
ÈHž]\ÎÈH
ÏH
BˆÂˆÛÛœÝZ[Ý
ˆH
ÈNÂˆYˆ
ÌHÌWHÌ—JBˆ]
ÊÎÂˆ[XH
ÏH
ÍÝH
ˆÌH
ÈMLH
ˆÌWH
ÈŽ]H
ˆÌ—JHˆÂˆÙY[ŠÊÎÂˆBˆÛÛœÝÝX›HÛÝ”ÝHÙY[ˆÈLŒ
ˆÝX›J]
HÈÝX›JÙY[ŠHˆŒÂˆÛÛœÝÝX›HYX[“[XHHÙY[ˆÈÝX›J[XJHÈÝX›JÙY[ŠHˆŒÂ‚ˆYˆ
Û›XÚÑ[ŠBˆÂˆÛÛœÝÝX›H›ÛÜ”ÝH]ÙŠÛ›XÚÑ[ŠHˆŒÈ]ÙŠÛ›XÚÑ[ŠHˆNÂˆËÈš\™Hš\œÝ[ˆ\›H8 %ÛÈ[™\[™[\ÝÈ˜]\ˆ[ˆ[ˆY‹Ù[ÙKÛÈBˆËÈÛÛ›ÛÚÜÙH™\ÚÛÈÝ™\›\Ø[ˆ^\˜Ú\ÙHHšYÙÙ\‹‚ˆYˆ
Ø]Ó]œ˜[YH	‰ˆÛÝ”Ý›ÛÜ”Ý	‰ˆÛ›XÚÐYÙ]ˆ	‰ˆ\\ÛÙ\ÓYˆ
BˆÂˆ›XÚÕ˜[œÚ][ÛˆHYNÂˆÛ›XÚÐYÙ]KNÂˆ\\ÛÙ\ÓYKNÂˆØ]Ó]œ˜[YHH˜[ÙNÂˆœš[ŠÝ\œ‹ˆ–Ýš×HÓTÓÓ—Ð“PÒÎˆœ˜[YH	[HÙ[›XÚÈ
	KŒÙ‰IH]›ÛÜˆ‚ˆ‰KŒ™‰IJH8 %[\[™ÈH™\ÛÛ™HÚZ[ˆ
	Y\\ÛÙH[\ÈY
Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™œ˜[YKÛÝ”Ý›ÛÜ”Ý\\ÛÙ\ÓY
NÂˆBˆYˆ
ÛÝ”ÝH]Ý
BˆÂˆØ]Ó]œ˜[YHHYNÂˆÛ›XÚÐYÙ]HŽÈËÈ™KX\›KÛÈHÙXÛÛ™\\ÛÙH\ÈØ]YÚÛÂˆBˆB‚ˆYˆ
Û‘\šÑ[ŠBˆÂˆÛÛœÝÝX›H›ÛÜ“[XHH]ÙŠÛ‘\šÑ[ŠHˆŒÈ]ÙŠÛ‘\šÑ[ŠHˆŒÂˆËÈÖ—Õ’×ÔÓTÓÓ—ÑT’×ÐQ•T—ÓTÏSˆ8 %YÛ›Ü™H]™\ž][™È™Y›Ü™Hˆ\ÈÙˆØ[ˆËÈÛØÚËˆ›ÝH™Yš[™[Y[ˆH›ÛÝH]HØÜ™Y[ˆ[™]™\žHØY[™ÂˆËÈØÜ™Y[ˆ˜YKÛÈ[ˆ\\ÛÙHYÙ]Z[YY]Ø[Y\^H\ÈÜ[™Y›Ü™BˆËÈØ[Y\^HÝ\ËˆØ[[YH˜]\ˆ[ˆHœ˜[YH[™^™XØ]\ÙHH™XÚ\\ÂˆËÈ]™XXÚØ[Y\^H\™HÜš][ˆ[ˆÙXÛÛ™È
Ö—ÑRÑWÔÕT•ÓTÈ[\˜[ÊBˆËÈ[™Hœ˜[YH[™^›ÜˆHØ[YH[ÛY[[Ý™\ÈÚ]Hœ˜[YH˜]K‚ˆÝ]XÈÛÛœÝ]]È\šÕHÝŽ˜Ú›Û›ÎŽœÝXYWØÛØÚÎŽ››ÝÊ
NÂˆÝ]XÈÛÛœÝÛ™ÈÛ™ÈY\“\ÈBˆ[ŠÖ—Õ’×ÔÓTÓÓ—ÑT’×ÐQ•T—ÓTÈŠBˆÈ]Û
[ŠÖ—Õ’×ÔÓTÓÓ—ÑT’×ÐQ•T—ÓTÈŠJHˆÂˆÛÛœÝÛ™ÈÛ™È›ÝÓ\ÈHÝŽ˜Ú›Û›ÎŽ™\˜][Û—ØØ\ÝÝŽ˜Ú›Û›ÎŽ›Z[\ÙXÛÛ™ÏŠˆÝŽ˜Ú›Û›ÎŽœÝXYWØÛØÚÎŽ››ÝÊ
HH\šÕ
K˜ÛÝ[

NÂˆÛÛœÝ›ÛÛ]™HH›ÝÓ\ÈHY\“\ÎÂˆYˆ
]™H	‰ˆØ]ÐœšYÚœ˜[YH	‰ˆYX[“[XH›ÛÜ“[XH	‰ˆ\šÑ\\ÛÙ\ÓYˆ
BˆÂˆ›XÚÕ˜[œÚ][ÛˆHYNÂˆ\šÑ\\ÛÙ\ÓYKNÂˆØ]ÐœšYÚœ˜[YHH˜[ÙNÂˆÝÙPœšYÚ™Y™\™[˜ÙHHYNÂˆœš[ŠÝ\œ‹ˆ–Ýš×HÓTÓÓ—ÑT’Îˆœ˜[YH	[HÙ[T’È
YX[ˆ[XH	KŒ™‹›ÛÜˆ‚ˆ‰KŒ™‹	KŒ™‰IH]
H8 %[\[™ÈH™\ÛÛ™HÚZ[ˆ
	YY
Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™œ˜[YKYX[“[XK›ÛÜ“[XKÛÝ”Ýˆ\šÑ\\ÛÙ\ÓY
NÂˆBˆYˆ
YX[“[XHH\šÓ][XJBˆÂˆYˆ
]™H	‰ˆÝÙPœšYÚ™Y™\™[˜ÙJBˆÂˆÝÙPœšYÚ™Y™\™[˜ÙHH˜[ÙNÂˆ›XÚÕ˜[œÚ][ÛˆHYNÂˆœš[ŠÝ\œ‹ˆ–Ýš×HÓTÓÓ—ÑT’Îˆœ˜[YH	[H\ÈH”’QÒ‘Q‘T‘SÑH›Üˆ‚ˆH\\ÛÙHX›Ý™H
YX[ˆ[XH	KŒ™‹	KŒ™‰IH]
Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™œ˜[YKYX[“[XKÛÝ”Ý
NÂˆBˆØ]ÐœšYÚœ˜[YHHYNÂˆBˆBˆB‚ˆËÈÖ—Õ’×Ñ”SQWÑSTO\ˆÜš]\È]™\žHœ˜[YH\ÈHKˆ\È\ÈH[œÝ[Y[ˆËÈ]XZÙ\ÈH™[™\™\ˆÚXÚØX›HÒUÕUHÚ[™ÝËÚXÚX]\œÈ[Ü™H[ˆ]ˆËÈÛÝ[™Îˆ]™\žHÝ\ˆØ]H\È›Ú™XÝÝÛœÈ\ÈHÙÈY™‹[™HXÝ\™H\ÂˆËÈšYÚˆ\ÈHÛ™HÛZ[H]™YYÈ[ˆ[XYÙKˆHXY\ÜÈ[ˆ\ÈH\™XÝÜžHÙ‚ˆËÈœ˜[Y\È\ÈHÙ[‹\Ù\˜X›H™\œÚ[ÛˆÙˆHK\ØÜ™Y[œÚÝÛÛ\\š\ÛÛ‹‚ˆËÈÖ—Õ’×Ñ”SQWÑSTÑU‘T–OSˆÝ™\œšY\ÈHˆHØÜ™Y[ˆHÞ[]XËZ[œ]\›HØ[ÜÂˆËÈ“ÕQÒ˜]\ˆ[ˆ\šÜÈÛˆØ[ˆ™HÚÜ\ˆ[ˆœ˜[Y\Ë[™Û™H[\Ùˆ]\ÂˆËÈÛ™HØ[\HÙˆH˜[œÚ][Ûˆ8 %HØ]™K\ÛÝ[™[™[ÝÈ\X\™Y[ˆ^XÝHÛ™BˆËÈœ˜[YHÙˆHNÈ›ÛÝ‚ˆËÈKKKHŽˆH•T”ÕKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKBˆËÂˆËÈÒHTÈVTÕËˆHÜ\˜]Üˆ\ØÜšX™YHY™XÝ›ÈÚ[™ÛHœ˜[YHØ[ˆÚÝÎˆ
ˆ•BˆËÈXØ[ÈÝÈ]ÛÚÜÈZÙH\È™]H]XÚ›Ü›X[]]\X\œÈ[™\Ø\X\ˆZÙBˆËÈ›XÚÙ\ˆXZÙH]ÛÈÚ[ˆH™\ÜÈŽ]™XÛÜ™È[œ˜[YH›ÜˆHÙXÛÛ™ÛÈ[ÝHØ[ˆÙYBˆËÈ]ˆŠˆHØÜ™Y[œÚÝÙˆH›XÚÙ\ˆ\ÈHØÜ™Y[œÚÝÙˆÛ™HTÑHÙˆ][™ÚXÚ\ÙBˆËÈ[ÝHÙ]\ÈXÚÈ
ÛÝÚHLÌÊKˆŽH[œÝÙ\œÈÚ]Ù\È]ÛÚÈZÙHŽÈŽ[œÝÙ\œÈÚ]ˆËÈÙ\È]ÈÝ™\ˆ[YH‹‚ˆËÂˆËÈS‘UTÈ•RSÈTÐÔ’SRSUK›Ý\ÝÈ[\Ý˜]KˆÛÈYXÚ[š\Û\È›ÙXÙH[‚ˆËÈY[XØ[Ý[[XYÙH[™™YYÜÜÚ]Hš^\Î‚ˆËÂˆËÈ
ˆH˜]È\ÈTÔÕQQ]™\žHœ˜[YH[™ÜÙ\ÈH\šYÚ8 %‹YšYÚ[™ËÚXÚ\ÂˆËÈÚ]HXØ[Ù\ÈÚ[ˆHÝY\Ý	ÜÈÛYÛÛˆÙ™œÙ]\È›ÝÛ›Ý\™Y[™BˆËÈXY[™È\Ý\Ú\È\™H™XØ]\ÙH\È™[™\™\ˆÙ]È›È\šX\Ñ[˜X›X][ÂˆËÈ
ˆH˜]È\È“ÔQÛˆÛÛYHœ˜[Y\È8 %žHHÝY\ÝžH™YXØ][Û‹žHHš[‚ˆËÈX\ÚËÜˆžHÛ™HÙˆÝ\ˆÝÛˆXÛ[™\Ë‚ˆËÂˆËÈ[™\ˆHš\œÝH˜]ÈÛÝ[[™˜]Ñš[™Ù\œš[\™HQS•PÐSœ˜[YHÈœ˜[YBˆËÈÚ[HH^[ÈÚ[™ÙKˆ[™\ˆHÙXÛÛ™^H[Ý™KˆÛÈHX[šY™\ÝØ\œšY\È›ÝˆËÈ\ˆœ˜[YK[™H\œÝ[œÝÙ\œÈH]Y\Ý[Ûˆ˜]\ˆ[ˆY\™[HÚÝÚ[™È]‚ˆËÂˆËÈÖ—Ð•T”ÕÑSTO\˜\›\È]ÈŽš\™\È]ÈÖ—Ð•T”ÕÑSTÓTØ
Y˜][L
H\ÈBˆËÈÚ[™ÝÈ[™Ö—Ð•T”ÕÑSTÓPV
Y˜][Ì
H›Ý[™ÈH\ÚÈ8 %]MHœÈ[™ÌŒBˆËÈÙXÛÛ™\ÈŒŒP‹ÚXÚ\Èš[™HÛˆH\ÚÈ[™ÛÝ[™HH\Ø\Ý\ˆ[ˆÝ\ÛÈBˆËÈ\™XÝÜžH\ÈHÜ\˜]Ü‰ÜÈÈÚÛÜÙH[™HY˜][\È[™\ˆZ\ˆ›ÝX›\ÚÛÝ[™ÂˆËÈ™YH˜]\ˆ[ˆH\œË‚ˆYˆ
[ŠÖ—Ð•T”ÕÑSTŠH	‰ˆÜÝÐÛÛœÝ[YP\œÝ[\™\ÜÙY

JBˆÂˆYˆ
‹O˜\œÝXÝ]™JBˆÂˆËÈHÙXÛÛ™™\ÜÈ\š[™ÈH\œÝS‘È]˜]\ˆ[ˆ™\Ý\[™È]ÛÈBˆËÈÜ\˜]ÜˆØ[ˆ›Ý[™H™XÛÜ™[™È^H]™H[™XYHÙY[ˆ[›ÝYÚÙ‹‚ˆœš[ŠÝ\œ‹–Ýš×H\œÝÉ]NˆÝÜYX\›HžHHÙXÛÛ™Ž
	]Hœ˜[Y\ÊWˆ‹ˆ‹O˜\œÝÙ\K‹O˜\œÝœ˜[Y\ÊNÂˆ‹O˜\œÝ[™œÈHÂˆBˆ[ÙBˆÂˆÝ]XÈZ[Ì—ÝÙ\HHÂˆ‹O˜\œÝÙ\HH
ÊÜÙ\NÂˆ‹O˜\œÝXÝ]™HHYNÂˆ‹O˜\œÝœ˜[Y\ÈHÂˆËÈØ[YH\›H\ÈŽIÜË›ÜˆHØ[YH™X\ÛÛˆ
\Íˆ][HJH8 %\œÝXÝ]™XˆËÈ[Û™H[™XYHÙY\ÈH™XY˜XÚÈÛˆ›ÜˆH\œÝ	ÜÈÚÛHÚ[™ÝËÛÈ\ÂˆËÈÛ›HÛÝ™\œÈHÛÝ\HÙˆœ˜[Y\È]H™\žHÝ\ÚÜÙH™XY˜XÚÈXÚ\Ú[Û‚ˆËÈØ\ÈXYH™Y›Ü™H\È™\ÜÈØ\ÈÛÛœÝ[YYˆÜÙHœ˜[Y\È\™H™\ÜY™[ÝÂˆËÈ˜]\ˆ[ˆÚ[[HZ\ÜÚ[™Ë‚ˆ‹Oœ™XY˜XÚÕ[[œ˜[YHBˆÝŽ›X^
‹Oœ™XY˜XÚÕ[[œ˜[YK‹O™œ˜[YH
È‹O™œ˜[Y\Ò[‘›YÚ
ÈŠNÂˆ
ÊÔ‹Oœ™XY˜XÚÐ\›YY™\ÜÙ\ÎÂˆ‹O˜\œÝ›Ô^[œ˜[Y\ÈHÂˆZ[Ý\ÈHLÂˆYˆ
ÛÛœÝÚ\ŠˆHH[ŠÖ—Ð•T”ÕÑSTÓTÈŠJBˆ\ÈHÝÝ[
K[‹L
NÂˆ‹O˜\œÝ[™œÈHZ[Ý
ÝŽ˜Ú›Û›ÎŽ™\˜][Û—ØØ\ÝÝŽ˜Ú›Û›ÎŽ›˜[›ÜÙXÛÛ™ÏŠˆÝŽ˜Ú›Û›ÎŽœÝXYWØÛØÚÎŽ››ÝÊ
K[YWÜÚ[˜ÙWÙ\ØÚ

JBˆ˜ÛÝ[

JH
Âˆ\È
ˆL[ÂˆÚ\ˆ]ÍLL—NÂˆÛœš[Š]Ú^™[Ùˆ]‰\ËØ\œÝ	LWÛX[šY™\Ý‹[ŠÖ—Ð•T”ÕÑSTŠKˆ‹O˜\œÝÙ\JNÂˆ‹O˜\œÝX[šY™\ÝH›Ü[Š]ÈŠNÂˆYˆ
‹O˜\œÝX[šY™\Ý
Bˆœš[Š‹O˜\œÝX[šY™\ÝˆˆÈš[Hœ˜[YH˜]ÜÈ™\XÙ\È˜]Ñš[™Ù\œš[Ø[Y\˜Qš[™Ù\œš[‚ˆ›YX[“[XH\Ý[˜ÝÛÛÝ\œÈ^[\ÚˆŠNÂˆËÈH\‹Y˜]ÈÙ[œÝ\ËY˜][[Û‹ˆ]ÈXœÙ[˜ÙH\ÈÚ]ÝÜY\M‰ÜÂˆËÈXØ[[˜[\Ú\ÎˆH\œÝÛÝ[ÚÝÈHXØ[›[šÚ[™È]ÛÝ[›ÝØ^BˆËÈÚ]\ˆ]È˜]ÈØ\È\ÜÝYYÛˆH\šÈœ˜[Y\Ëˆ™XY]Ú]ˆËÈÛÛËØ\œÝÜ™XYœKÚXÚ[YÛœÈ]Ú]H\ÈžHœ˜[YH[X™\‹‚ˆÛÛœÝÚ\ŠˆÙ[œÝ\ÕØ[H[ŠÖ—Ð•T”ÕÐÑS”ÕTÈŠNÂˆYˆ
XÙ[œÝ\ÕØ[Ý˜Û\
Ù[œÝ\ÕØ[ŒŠHOH
BˆÂˆÛœš[Š]Ú^™[Ùˆ]‰\ËØ\œÝ	LWØÙ[œÝ\Ë‹ˆ[ŠÖ—Ð•T”ÕÑSTŠK‹O˜\œÝÙ\JNÂˆ‹O˜\œÝÙ[œÝ\Ñš[HH›Ü[Š]ÈŠNÂˆYˆ
‹O˜\œÝÙ[œÝ\Ñš[JBˆœš[Š‹O˜\œÝÙ[œÝ\Ñš[KˆˆÈ]™\žH˜]ÈÙˆ]™\žH\œÝœ˜[YK“ˆÙ[œÝ\È[™O‹ˆØ[YH‚ˆ™šY[È\ÈHØ\\™HÙ[œÝ\ÎÈŒH\ÈHš\œÝ™\^	ÜÈ‚ˆ™š\œÝ™YHÛÜ™ËHY[]H]Ý\š]™\ÈHØ[Y\˜H‚ˆ›[Ý™K—ˆŠNÂˆ[ÙBˆœš[ŠÝ\œ‹–Ýš×H\œÝÉ]NˆØ[››ÝÜ[ˆ]ÈÙ[œÝ\Èš[H8 %H‚ˆš\ÜÝYY[Ü‹Y\ØØ\™Y[ˆÙˆH[œÝÙ\ˆÚ[™H‚ˆ›Z\ÜÚ[™×ˆ‹‹O˜\œÝÙ\JNÂˆ‹O˜\œÝÙ[œÝ\Ó[™\ÈHÂˆBˆœš[ŠÝ\œ‹ˆ–Ýš×H\œÝÉ]HT“QQˆ]™\žH™\Ù[Yœ˜[YH›Üˆ	[H\È[È	\É\×ˆ‹ˆ‹O˜\œÝÙ\K
[œÚYÛ™YÛ™ÈÛ™Ê[\Ë[ŠÖ—Ð•T”ÕÑSTŠKˆ‹O˜\œÝX[šY™\ÝÈˆˆˆˆ8 %HHHX[šY™\ÝÛÝ[›Ý™HÜ[™YÛÈ‚ˆHœ˜[Y\ÈÚ[]™H›È˜]È]H™\ÚYH[HŠNÂˆBˆBˆYˆ
‹O˜\œÝXÝ]™JBˆÂˆZ[Ì—ÝX^œ˜[Y\ÈHÌÂˆYˆ
ÛÛœÝÚ\ŠˆHH[ŠÖ—Ð•T”ÕÑSTÓPVŠJBˆX^œ˜[Y\ÈHZ[Ì—Ý
ÝÝ[
K[‹L
JNÂˆÛÛœÝZ[Ý›ÝÓœÈBˆZ[Ý
ÝŽ˜Ú›Û›ÎŽ™\˜][Û—ØØ\ÝÝŽ˜Ú›Û›ÎŽ›˜[›ÜÙXÛÛ™ÏŠˆÝŽ˜Ú›Û›ÎŽœÝXYWØÛØÚÎŽ››ÝÊ
K[YWÜÚ[˜ÙWÙ\ØÚ

JBˆ˜ÛÝ[

JNÂˆYˆ
\
BˆÂˆËÈ\›YYÚ]›È^[ËˆØ^HÛÈžH˜[YH˜]\ˆ[ˆ™XÛÜ™[™È›Ý[™ËÚXÚˆËÈÛÝ[™XY\ÈHY™XÝY›Ý\[ˆˆ
ÛÝÚHMLJK‚ˆËÂˆËÈÚ[˜ÙH\Íˆ\È\ÈVPÕQ›ÜˆHš\œÝœ˜[YHÜˆÛÈÙˆ]™\žH\œÝˆËÈ[ˆHÝØ\ÚZ[ˆ\›K[™Û›HÜÙNˆŽ\ÈÛÛœÝ[YY\™K]H›ÝÛBˆËÈÙˆHÝØ\ÚÜÙH™XY˜XÚÈXÚ\Ú[ÛˆØ\ÈXYH]HÜÛÈHX\›Y\ÝˆËÈœ˜[YH\È™\ÜÈØ[ˆ\›H\ÈH™^Û™H[™HX\›Y\Ýœ˜[YHÚÜÙBˆËÈ^[È\œš]™H\ÈHÛ™HY\ˆ]ˆÛÝ[Y˜]\ˆ[ˆš[Y\‚ˆËÈœ˜[YK[™š[YÛ˜ÙH]H[™ÙˆH\œÝ8 %H\‹Yœ˜[YH[™H\™BˆËÈÛÝ[™HÛÈ[™\ÈÙˆ›Ú\ÙHÛˆ]™\žH\œÝ[™H\œÝ]ÜÝˆËÈT•Hœ˜[Y\ÈÛÝ[ÛÚÈ^XÝHHØ[YH\ÈÛ™H]ÜÝÛË‚ˆ
ÊÔ‹O˜\œÝ›Ô^[œ˜[Y\ÎÂˆBˆ[ÙBˆÂˆÚ\ˆ]ÍLL—NÂˆÛœš[Š]Ú^™[Ùˆ]‰\ËØ\œÝ	LWÉLWÙ‰L›KœH‹ˆ[ŠÖ—Ð•T”ÕÑSTŠK‹O˜\œÝÙ\K‹O˜\œÝœ˜[Y\Ëˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™œ˜[YJNÂˆYˆ
’SJˆˆH›Ü[Š]ØˆŠJBˆÂˆœš[Š‹”—‰]H	]WŒMWˆ‹ÚYKZYÚJNÂˆ›Üˆ
Ú^™WÝHHÈHž]\ÎÈH
ÏH
BˆÜš]J	œÚWKKËŠNÂˆ˜ÛÜÙJŠNÂˆYˆ
‹O˜\œÝX[šY™\Ý
Bˆœš[Š‹O˜\œÝX[šY™\Ýˆ‰\È	[H	[H	[H	LM›	LM›ˆ‹ˆÝœ˜ÚŠ]	ËÉÊHÈÝœ˜ÚŠ]	ËÉÊH
ÈHˆ]ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™œ˜[YKˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™˜]ÜËˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™\XÙ\Ëˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™˜]Ñš[™Ù\œš[ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë˜Ø[Y\˜Qš[™Ù\œš[
NÂˆ
ÊÔ‹O˜\œÝœ˜[Y\ÎÂˆBˆBˆYˆ
›ÝÓœÈH‹O˜\œÝ[™œÈ‹O˜\œÝœ˜[Y\ÈHX^œ˜[Y\ÊBˆÂˆ‹O˜\œÝXÝ]™HH˜[ÙNÂˆYˆ
‹O˜\œÝX[šY™\Ý
BˆÂˆ˜ÛÜÙJ‹O˜\œÝX[šY™\Ý
NÂˆ‹O˜\œÝX[šY™\ÝH[ŽÂˆBˆYˆ
‹O˜\œÝÙ[œÝ\Ñš[JBˆÂˆ˜ÛÜÙJ‹O˜\œÝÙ[œÝ\Ñš[JNÂˆ‹O˜\œÝÙ[œÝ\Ñš[HH[ŽÂˆœš[ŠÝ\œ‹–Ýš×H\œÝÉ]HÙ[œÝ\Îˆ	[H˜]È[™\×ˆ‹‹O˜\œÝÙ\Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O˜\œÝÙ[œÝ\Ó[™\ÊNÂˆBˆœš[ŠÝ\œ‹ˆ–Ýš×H\œÝÉ]HÓ‘Nˆ	]Hœ˜[Y\È[È	\È8 %™XY]Ú]‚ˆÛÛËØ\œÝÜ™XYœI\È‹ˆ‹O˜\œÝÙ\K‹O˜\œÝœ˜[Y\Ë[ŠÖ—Ð•T”ÕÑSTŠKˆ‹O˜\œÝ›Ô^[œ˜[Y\ÈÈˆˆˆ—ˆŠNÂˆYˆ
‹O˜\œÝ›Ô^[œ˜[Y\ÊBˆœš[ŠÝ\œ‹ˆˆ

É[Hœ˜[YJÊH]HÝ\Ú]›È™XY˜XÚÈ^[ÈY]8 %‚ˆ™^XÝYÙYHH™XY˜XÚÈ™YXØ]NÈ[Ü™H[ˆˆÜˆÈYX[œÈ‚ˆHŽ\›H\È›Ý™XXÚ[™ÈH™YXØ]JWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O˜\œÝ›Ô^[œ˜[Y\ÊNÂˆBˆBˆËÈXÚYHÓÑK]Hœ˜[YH›Ý[™\žKÚ]\ˆH‘Vœ˜[YIÜÈ˜]ÜÈÛÈ[ÈBˆËÈ\œÝÙ[œÝ\È8 %H˜]È]]\Ý›Ý™XY\œÝ[Z[™ÈÝ]HZYYœ˜[YKÜˆBˆËÈÙ[œÝ\ÈÛÝ[Û\ÙˆHœ˜[YH[™™XY\ÈH˜]ËXÛÝ[Ú[™ÙH
ÛÝÚHLIÜÂˆËÈÚ\NˆH\X[\Ý\È›ÝHÛÝ[
KˆÖ—Ð•T”ÕÐÑS”ÕT×ÑU‘T–OSˆ[œÈÈ]™\žHˆËÈ\œÝœ˜[YH›ÜˆÛ™È\œÝÎÈHY˜][\È]™\žHœ˜[YK™XØ]\ÙHš\ÜÝYYÜˆ›Ý‚ˆËÈ\ÈH\‹Yœ˜[YH]Y\Ý[Û‹‚ˆÂˆZ[Ì—Ý]™\žHHNÂˆYˆ
ÛÛœÝÚ\ŠˆHH[ŠÖ—Ð•T”ÕÐÑS”ÕT×ÑU‘T–HŠJBˆ]™\žHHÝŽ›X^
]KZ[Ì—Ý
ÝÝ[
K[‹L
JJNÂˆ‹O˜\œÝÙ[œÝ\Õ\Ñœ˜[YHH‹O˜\œÝXÝ]™H	‰ˆ‹O˜\œÝÙ[œÝ\Ñš[H	‰‚ˆ
‹O˜\œÝœ˜[Y\È	H]™\žJHOHÂˆB‚ˆËÈHÖ—ÐÐTT‘WÒÑVHXÝ\™KÜš][ˆœ›ÛHHØ[YH™XY˜XÚÈH\š[ÙXÈ[\\Ù\Ë‚ˆËÈÙ\\˜]Hœ›ÛHHÛÜ™[ÝÈ˜]\ˆ[ˆ›ÛY[È]È[\˜[\Ý™XØ]\ÙBˆËÈ\ÈÛ™H\ÈÈš\™HÛˆVPÕHH\›YYœ˜[YH8 %H[\˜[\ÝÛÝ[Z]\‚ˆËÈZ\ÜÈ]Ü‹YˆH[\˜[Ù\™H›Ü˜ÙYÈKÜš]H]™\žHœ˜[YHÙˆH[‹‚ˆYˆ
	‰ˆ‹O˜Ø\\™TXÝ\™Qœ˜[YH	‰ˆ™\Ë™œ˜[YHOH‹O˜Ø\\™TXÝ\™Qœ˜[YJBˆÂˆ‹O˜Ø\\™TXÝ\™Qœ˜[YHHÂˆÚ\ˆ]ÍLL—NÂˆÛœš[Š]Ú^™[Ùˆ]‰\ËØØ\\™WÉL›KœH‹[ŠÖ—ÐÐTT‘WÒÑVHŠKˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™œ˜[YJNÂˆYˆ
’SJˆˆH›Ü[Š]ØˆŠJBˆÂˆœš[Š‹”—‰]H	]WŒMWˆ‹ÚYKZYÚJNÂˆ›Üˆ
Ú^™WÝHHÈHž]\ÎÈH
ÏH
BˆÜš]J	œÚWKKËŠNÂˆ˜ÛÜÙJŠNÂˆœš[ŠÝ\œ‹–Ýš×HØ\\™NˆÜ›ÝH	\È
	]^	]JI\×ˆ‹]ÚYKZYÚKˆ‹O™˜]ÒY˜[“Û‘œ˜[YHOH™\Ë™œ˜[YBˆÈˆ8 %“ÕHPÕT‘NˆÖ—Õ’×ÑU×ÒQZ[Y\Èœ˜[YIÜÈ˜]È‚ˆš[™XÙ\ËÛÈ™XYH˜]ÚYÊˆÛ˜\ÚÝ[œÝXY‚ˆˆˆŠNÂˆBˆ[ÙBˆÂˆËÈHØ\\™H]Ú[[HÜš]\È›Ý[™È\ÈÛÜœÙH[ˆ›ÈØ\\™NˆBˆËÈÜ\˜]ÜˆØ[ÜÈ]Ø^H™[Y]š[™ÈH]šY[˜ÙH^\ÝÈ
ÛÝÚ\ÈKMLJK‚ˆœš[ŠÝ\œ‹–Ýš×HØ\\™NˆÐS““ÕÔ’UH	\È8 %HXÝ\™H\ÈÔÕˆ‹]
NÂˆB‚ˆËÈ‹‹S‘HÔÑKÛÈHÚÝØ[ˆ™HZÙ[ˆYØZ[‹‚ˆËÂˆËÈ]™\žHXÝ\™Hš[™[™È[ˆ\ÈÜ\È™Y[ˆ[˜ÚÜ™YÈHÜ\˜]ÜˆØ[ÙYˆËÈÛÛY]Ú\™H[™™\ÜÙYŽH‹ÚXÚ\È›ÝH™\›ÙXÚX›H^\š[Y[ˆ›Ý[™ÂˆËÈXY\ÜÈØ[ˆ™]\›ˆÈ]ÜÝ[™HÝš\Y[X]\šX[Û\ÜÈXÚÜÈBˆËÈY™™\™[]X[]H]™[ÛˆXXÚ›ÛÝÛÈHÙXÛÛ™š\Ú]\ÈHY™™\™[ˆËÈYX\Ý\™[Y[ˆÚ]XZÙ\ÈHÚÝ™\X]X›H\ÈHÐSQTH[™HVQT‹ÛÂˆËÈ›Ý\™H™XÛÜ™Y™\ÚYHHXÝ\™K‚ˆËÂˆËÈ›Ý\™HÜš][ˆUÈ8 %HMˆ›Ø]™\^ÛÛœÝ[ÈHØ[Y\˜Hš[™Ù\œš[ˆËÈ\Ú\È
šY]Ë\›Ú™XÝ[Ûˆ[™ÛÜ›X]šXÙ\ÊH[™HXYÙˆH^Y\‰ÜÈØ[YBˆËÈØš™XÝˆ\š]š[™È[ˆ^YHÜÚ][ÛˆÜˆ˜[Z[™ÈHÜÚ][ÛˆšY[\™HÛÝ[™BˆËÈÝY\ÜÚ[™È]H^[Ý]ÈÛÈœÜÙHš[\ÈZÙ[ˆ[ˆY™™\™[XÙ\È˜[YHÜÙBˆËÈšY[ÈžHÚ]ÒS‘ÑQ[™][˜[\Ú\È™[Û™ÜÈ[ˆHÛÛ]Ø[ˆ™Hš^YˆËÈÚ]Ý]H™XZ[
ÛÛËÜÜÙWÜ™XYœJK‚ˆÛœš[Š]Ú^™[Ùˆ]‰\ËØØ\\™WÉL›KœÜÙH‹[ŠÖ—ÐÐTT‘WÒÑVHŠKˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™œ˜[YJNÂˆYˆ
’SJˆˆH›Ü[Š]ÈŠJBˆÂˆœš[Š‹ˆÈœ˜[YH	[HØ[Y\˜Qš[™Ù\œš[	LM›˜]Ñš[™Ù\œš[	LM›ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™œ˜[YKˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O˜Ø[Y\˜Qš[™Ù\œš[ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O™˜]Ñš[™Ù\œš[
NÂˆœš[Š‹ˆÈ˜ÖÚWHHÛÛœÝ[È]Hœ˜[YIÜÈ’T”Õ˜]È
\ÝX[HHÒQÕÈ‚ˆœ\ÜÎˆ]ÈšY]ÈX]š^\ÈHQÒ	ÜÊWˆŠNÂˆœš[Š‹ˆÈ˜ÖÚWHHÛÛœÝ[È]Hœ˜[YIÜÈ’QÑÑTÕ˜]È
	]H™\ÊH8 %H‚ˆ”ÐÑS‘HØ[Y\˜Kˆ™Y™\ˆ\ÙK—ˆ‹‹O˜Ø[PšYÕ™\ÊNÂˆ›Üˆ
[ÚXÚHÈÚXÚŽÈÚXÚ
ÊÊBˆÂˆÛÛœÝZ[Ì—Ý
ˆÜ˜ÈHÚXÚÈ‹O˜Ø[PÛÛœÝÐšYÈˆ‹O˜Ø[PÛÛœÝÎÂˆ›Üˆ
Z[Ì—ÝHHÈHÈH
ÏH
BˆÂˆ›Ø]–ÍNÂˆ›Üˆ
Z[Ì—ÝÈHÈÈÈÊÊÊBˆÂˆÛÛœÝZ[Ì—Ýš]ÈHÜ˜ÖÚH
È×NÂˆY[XÜJ	–Ú×K	˜š]Ë
NÂˆBˆœš[Š‹‰\ÉKLH	K™ˆ	K™ˆ	K™ˆ	K™—ˆ‹ÚXÚÈ˜˜Èˆˆ˜È‹ˆHÈ–ÌK–ÌWK–Ì—K–Ì×JNÂˆBˆBˆËÈHØš™XÝ[\]™\È[ˆXY×Ý[˜X›\Ë˜ÜÚXÚ[™XYHÝÛœÈÝY\ÝˆËÈY[[ÜžHXØÙ\ÜÈ[™HÚ[\ˆ]Ù[ŽÈÝY\Ý˜[™ÙSÚÈ\™HÛÝ[™HBˆËÈÜ›Û™ÈÚXÚÈ[ž]Ø^KÚ[˜ÙH]˜[Y]\ÈÛ›HH\ÚXØ[^\™H\™[˜BˆËÈ[™HØ[YHØš™XÝ\È[ˆÜ™[˜\žHš\X[Y™\ÜË‚ˆËÈHÔÒUSÓˆUÑS‹ÚXÚ\ÈHÚÛHÚ[ÙˆHÜÙNˆ™XYšXBˆËÈHÝY\Ý	ÜÈÝÛˆÙ]^Y\š[™›È]
Øš‹OX›VÌNJK›Ý[™™\œ™YˆËÈœ›ÛHHØš™XÝ[\™[ÝËˆH[\Ý^\È™XØ]\ÙH]\ÈÚ]˜[YYˆËÈ\ÈšY[	ÜÈ™ZYÚ›Ý\œË[™™XØ]\ÙH[ˆ[™^Z[™YÝXÝ\ÈÛÜˆËÈÙY\[™ÈÚ[HH^[Ý]\ÈÝ[™Z[™ÈX\›™Y‚ˆ›Ø]ÜÖÌ×NÂˆÛ™ÈÛ™ÈYÙS\ÈHLNÂˆYˆ
Ö—ÑXYÔ^Y\”ÜÊÜË	˜YÙS\ÊJBˆœš[Š‹œ^Y\—ÜÜÈ	Kˆ	Kˆ	KˆÈ™XY	[\È™Y›Ü™H\È‚ˆ˜Ø\\™KšXHÙ]^Y\š[™›ÉÜÈX›VÌNWˆ‹ˆÜÖÌKÜÖÌWKÜÖÌ—KYÙS\ÊNÂˆ[ÙBˆœš[Š‹ˆÈ^Y\—ÜÜÈSURSP“H8 %›È]™[[›š[™ËÜˆH‚ˆ›ÛÚÝ\˜Z[YˆŠNÂˆÛÛœÝZ[Ì—ÝØšˆHÖ—ÑXYÕÜš]T^Y\“Øš™XÝ
‹Œ
NÂˆ˜ÛÜÙJŠNÂˆœš[ŠÝ\œ‹–Ýš×HØ\\™NˆÜ›ÝH	\È
Ø[Y\˜H
È^Y\ˆØš™XÝ	L
Wˆ‹ˆ]ØšŠNÂˆBˆ[ÙBˆœš[ŠÝ\œ‹–Ýš×HØ\\™NˆÐS““ÕÔ’UH	\È8 %HÜÙH\ÈÔÕˆ‹]
NÂˆBˆËÈÖ—Õ’×ÔÒÖWÐTÖSOLH8 %ÐÓÔ‘HHS‹TÐÔ‘QSˆÒÖH“PÒÑTˆÒUÕUHSPSˆÐUÒS‘Ë‚ˆËÂˆËÈHÜ\˜]Ü‰ÜÈ™\Ü\È[Ø^\È™Y[ˆHØ[YHÚ\Nˆ
ˆœÚÞH›XÚÙ\ˆœ›ÛH[ˆBˆËÈØÜ™Y[ˆÝÚ]Ú[™Èœ›ÛHšYÚÈY\[™[™ÈÙˆ[ÛY[Š‹ˆ]\ÈHQ•Ô’QÒˆËÈ\Þ[[Y]žH[ˆHÚÞH]ÒS‘ÑTÈ‘UÑQSˆ”SQTÈ8 %\È]H™[™\œÈ[ˆYÜšYÚˆËÈ]ÚYH[\È
ÛÝÚHJKÛÈ]\È^XÝHÚ]H\‹][HY™™\™[˜ÙHÛÚÜÈZÙK‚ˆËÂˆËÈ]™\žH™\™XÝÛˆ\ÈY™XÝÛÈ˜\ˆ\ÈÛÛYHœ›ÛH[ˆ^YH[™H™YK[Z[]H[‹[™ˆËÈ]\ÈÚH]Ø[››Ý™HÙ]YˆÛ™HÛX[ˆ[ˆØ[››Ý™HÛœ›ÛHH[ˆ]YˆËÈ›ÝšYÙÙ\ˆ]ÚXÚ\ÈHÜ\˜]Ü‰ÜÈÝÛˆØš™XÝ[Ûˆ[™\ÈÚH\ÌˆZ[BˆËÈ™]™\\›KˆÛÈ\ÈYX\Ý\™\È]ˆYX[ˆ[XHÙˆHÜÝš\Ü]Y[™šYÚˆËÈ[™HÝ]\ÝXÈ™\ÜY\È›ÝH\Þ[[Y]žH]Ù[ˆ]
ŠšÝÈÙ[ˆ]“TÈÒQÓ‚ˆËÈ™]ÙY[ˆÛÛœÙXÝ]]™Hœ˜[Y\ÊŠ‹™XØ]\ÙHHÝXYH\Þ[[Y]žH\È\ÝHØÙ[™HÚ]BˆËÈœšYÚÚYH[™H›XÚÙ\ˆ\ÈH[\›˜][Û‹‚ˆËÂˆËÈ]™YYÈÜÝ^[ËÛÈ]›Ú[œÈH™XY˜XÚÈ\ÝX›Ý™NÈÛˆHÝØ\ÚZ[ˆ\›H]ˆËÈ›Ü˜Ù\ÈH™XY˜XÚÈHÝØ\ÚZ[ˆ]ÛÝ[Ý\Ú\ÙHÚÚ\ˆÛÜÝ\ÈÛ™H\ÜÈÝ™\ˆBˆËÈÜ]X\\ˆÙˆH[XYÙH\ˆ™\Ù[Yœ˜[YH8 %HXYÛ›ÜÝXÈ\›K™]™\ˆHY˜][‚ˆÝ]XÈÛÛœÝ›ÛÛÚÞP\Þ[HH[“ÛŠÖ—Õ’×ÔÒÖWÐTÖSHŠNÂˆYˆ
ÚÞP\Þ[H	‰ˆ
BˆÂˆÛÛœÝZ[Ì—ÝÝš\HÝŽ›X^
]KZYÚHÈ
NÈËÈHÚÞHØØÝ\Y\ÈHÜˆÝX›HÝ[SHŒÝ[TˆHŒÂˆZ[Ý“H”ˆHÂˆÛÛœÝZ[Ì—Ý[ˆHÚYHÈŽÂˆ›Üˆ
Z[Ì—ÝHHÈHÝš\È
ÊÞJBˆ›Üˆ
Z[Ì—ÝHÈÚYNÈ
ÊÞ
BˆÂˆÛÛœÝZ[Ý
ˆHH
È
Ú^™WÝ
JH
ˆÚYH
È
H
ˆÂˆÛÛœÝÝX›HHŒŒLˆ
ˆVÌH
ÈÌMLˆ
ˆVÌWH
ÈŒÌŒˆ
ˆVÌ—NÂˆYˆ
[ŠHÈÝ[S
ÏHÈ
ÊÛ“ÈBˆ[ÙHÈÝ[Tˆ
ÏHÈ
ÊÛ”ŽÈBˆBˆÛÛœÝÝX›HH
“ÈÝ[SÈÝX›J“
HˆŒ
HH
”ˆÈÝ[TˆÈÝX›J”ŠHˆŒ
NÂˆÝ]XÈÝX›H™]ˆHŒÂˆÝ]XÈ›ÛÛ]™T™]ˆH˜[ÙNÂˆ
ÊÙ×ÜÚÞQœ˜[Y\ÎÂˆ×ÜÚÞPXœÔÝ[H
ÏHÝŽ™˜XœÊ
NÂˆYˆ
]™T™]ŠBˆÂˆËÈH“T\ÈHÚYÛˆÚ[™ÙHÚ]›ÝÚY\ÈYX[š[™Ù[H›Û‹^™\›È8 %H\Þ[[Y]žBˆËÈ]\š[™È\›Ý[™Œ\È›Ú\ÙK›ÝH›XÚÙ\‹[™ÛÝ[[™È]ÛÝ[XZÙBˆËÈ]™\žH\›HÛÚÈY[XØ[‚ˆYˆ


™]ˆˆŒH	‰ˆLŒJH
™]ˆLŒH	‰ˆˆŒJJJBˆ
ÊÙ×ÜÚÞQ›\ÎÂˆ×ÜÚÞTÝ\Ý[H
ÏHÝŽ™˜XœÊH™]ŠNÂˆYˆ
ÝŽ™˜XœÊH™]ŠHˆ×ÜÚÞTÝ\X^
Bˆ×ÜÚÞTÝ\X^HÝŽ™˜XœÊH™]ŠNÂˆBˆËÈS‘HUÈÑT’QTË™XØ]\ÙHHš\œÝÝ]Ùˆ\È[œÝ[Y[Ú\YHÚ[™ÛBˆËÈÝ[[X\žH
ÚYÛˆ›\ÊH[™]Y“Õ\ØÜš[Z[˜]NˆKŒL	HÛˆH[ˆHÜ\˜]Ü‚ˆËÈØ[Y›XÚÙ\š[™ÈYØZ[œÝŽ‰HÛˆÛ™H^HØ[YÛX[‹ˆHÝ]\ÝXÈØ\ÂˆËÈÜ›Û™Ë›ÝHYX\Ý\™[Y[8 %H\›š[™ÈØ[Y\˜HÚ[™Ù\ÈÚXÚÚYH\ÈœšYÚ\‹ÛÂˆËÈÚYÛˆ›\ÈÛÝ[H›Ý]H\È]XÚ\ÈHY™XÝˆÜš][™ÈH\‹Yœ˜[YHÙ\šY\ÂˆËÈYX[œÈHÝ]\ÝXÈØ[ˆ™HÚÜÙ[ˆQ•TˆÛÚÚ[™È]H]H[œÝXYÙˆÝY\ÜÙYˆËÈ™Y›Ü™KÚXÚ\ÈHÚÛH™X\ÛÛˆHÝ[[X\žH\ÈHÜ›Û™È[™ÈÈÛÛXÝš\œÝ‚ˆÝ]XÈ’SJˆÙ\šY\ÈH×H

HOˆ’SJˆÂˆÛÛœÝÚ\ŠˆˆH[ŠÖ—Õ’×ÔÒÖWÐTÖSHŠNÂˆYˆ
YˆJ™ˆ\Ý˜Û\
‹ŒHŠJBˆ™]\›ˆ[ŽÂˆ’SJˆH›Ü[Š‹ÈŠNÂˆYˆ

Bˆœš[Š™œ˜[YH˜]ÜÈ\Þ[WˆŠNÂˆ[ÙBˆœš[ŠÝ\œ‹–Ýš×HÖ—Õ’×ÔÒÖWÐTÖSNˆÐS““ÕÔ’UH	\È8 %›ÈÙ\šY\×ˆ‹ŠNÂˆ™]\›ˆÂˆJ
NÂˆYˆ
Ù\šY\ÊBˆœš[ŠÙ\šY\Ë‰[H	]H	K—ˆ‹
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™œ˜[YKˆZ[Ì—Ý
‹O™˜]ÜÕ\Ñœ˜[YJK
NÂˆ™]ˆHÂˆ]™T™]ˆHYNÂˆBˆÝ]XÈÛÛœÝÚ\Šˆ[\\ˆH[ŠÖ—Õ’×Ñ”SQWÑSTŠNÂˆÝ]XÈÛÛœÝZ[Ý[\]™\žHBˆ[ŠÖ—Õ’×Ñ”SQWÑSTÑU‘T–HŠBˆÈÝŽ›X^Z[ÝŠKÝÝ[
[ŠÖ—Õ’×Ñ”SQWÑSTÑU‘T–HŠK[‹L
JBˆˆÂˆYˆ
	‰ˆ[\\ˆ	‰ˆ
™\Ë™œ˜[YH	H[\]™\žJHOH
BˆÂˆËÈÜ™X]HH\™XÝÜžK[™ÐVHÓÈYˆHœ˜[Y\ÈØ[››Ý™HÜš][‹ˆ\È\ÙYÂˆËÈ™HH˜\™H›Ü[ˆÚÜÙH˜Z[\™HØ\ÈÚ[[ÛÈH[ˆÚ[Y]H\™XÝÜžH]ˆËÈY›Ý^\Ý›ÙXÙY[ˆ[\H™\Ý[]ÛÚÙY^XÝHZÙHH[ˆÚÜÙBˆËÈ™[™\™\ˆ™]È›Ý[™È8 %[™HXÝ\™HÚXÚÈ\ÈHÛ™HØ]H[ˆ\È›Ú™XÝˆËÈ]\È›ÈÙËYY™ˆÝXœÝ]]Kˆ[ˆ[œÝ[Y[]Ø[ˆ›ÙXÙH›Ý[™ÈÚ]Ý]ˆËÈÛÛ\Z[š[™È\È›Ý[ˆ[œÝ[Y[
ÛÝÚ\ÈKMLJK‚ˆÝ]XÈ›ÛÛ\”™XYHH˜[ÙNÂˆÝ]XÈ›ÛÛÛÛ\Z[™YH˜[ÙNÂˆYˆ
Y\”™XYJBˆÂˆÝŽ™\œ›Ü—ØÛÙHXÎÂˆÝŽ™š[\Þ\Ý[NŽ˜Ü™X]WÙ\™XÝÜšY\Ê[\\‹XÊNÂˆ\”™XYHHYNÂˆBˆÚ\ˆ]ÍLL—NÂˆÛœš[Š]Ú^™[Ùˆ]‰\ËÙœ˜[YWÉL›KœH‹[\\‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™œ˜[YJNÂˆYˆ
’SJˆˆH›Ü[Š]ØˆŠJBˆÂˆœš[Š‹”—‰]H	]WŒMWˆ‹ÚYKZYÚJNÂˆ›Üˆ
Ú^™WÝHHÈHž]\ÎÈH
ÏH
BˆÜš]J	œÚWKKËŠNÂˆ˜ÛÜÙJŠNÂˆBˆ[ÙHYˆ
XÛÛ\Z[™Y
BˆÂˆÛÛ\Z[™YHYNÂˆœš[ŠÝ\œ‹–Ýš×HÖ—Õ’×Ñ”SQWÑSTØ[››ÝÜš]H	\È8 %›Èœ˜[Y\ÈÚ[™H‚ˆ™[\Y\È[—ˆ‹ˆ]
NÂˆBˆB‚ˆËÈÖ—Õ’×Ñ”SQWÔÕUÏOš[Oˆ8 %Û™H[™H\ˆœ˜[YNˆÚ]HÝY\Ý\ÚÙY›Ü‹[™Ú]ˆËÈØ[YHÝ]ˆ\È\ÈH˜]ÈX]\šX[›ÜˆÛÛËÙœ˜[YWØÛÛ\\™KœKÚXÚ[YÛœÈÛÂˆËÈ[œÈžHÓÓ•S•[™Û›H[ˆÛÛ\\™\ÈZ\ˆXÝ\™\Ë‚ˆËÂˆËÈHÝ]]YX\Ý\™[Y[È\™H[X™\˜][HÚX\[™ÚÛKZ[XYÙH
ÛÝ™\˜YÙKYX[‚ˆËÈ[Z[˜[˜ÙK\Ý[˜ÝÛÛÝ\œËH^[\Ú
H˜]\ˆ[ˆH\‹\^[[\ˆBˆËÈ]Y\Ý[ÛˆH™[™\™\ˆKÐˆ\ÚÜÈ\È™Y\Èœ˜[YHÚ[™ÙH‹[™›Üˆ]HÛX[ˆËÈ™XÝÜˆÙˆYÙÜ™YØ]\ÈÝ™\ˆHØ[YHÛÛ[\È[›ÝYÚ8 %Ú[HH\‹\^[[\]ˆËÈÌœ˜[Y\ÈHÙXÛÛ™\ÈLPˆH[ˆ›Ø›ÙH™XYË‚ˆËÈÖ—Õ’×ÑVÔÕT‘WÕPÑOOš[Oˆ8 %Û™H[™H\ˆœ˜[YNˆÝÈX[žH˜]ÜÈÙ][ˆ^ÜÝ\™BˆËÈ[™H˜[™ÙHÙˆ˜[Y\È^HÙ]ˆÜš][ˆ\™K]H™\Ù[ÛÈ]Èœ˜[YBˆËÈ[X™\œÈ\™HHØ[YHÛ™\ÈÖ—Õ’×ÔÓTÑ”SQH[™Ö—Õ’×Ñ”SQWÔÕUÈ\ÙK‚ˆÝ]XÈ’SJˆ^š[HH[ŽÂˆÝ]XÈ›ÛÛ^šYYH˜[ÙNÂˆYˆ
Y^šYY
BˆÂˆ^šYYHYNÂˆYˆ
ÛÛœÝÚ\Šˆ]H[ŠÖ—Õ’×ÑVÔÕT‘WÕPÑHŠJBˆÂˆ^š[HH›Ü[Š]ÈŠNÂˆYˆ
^š[JBˆœš[Š^š[KˆÈœ˜[YH˜]ÜÈ^Z[ˆ^X^ˆŠNÂˆ[ÙBˆœš[ŠÝ\œ‹–Ýš×HØ[››ÝÜš]HÖ—Õ’×ÑVÔÕT‘WÕPÑOI\×ˆ‹]
NÂˆBˆBˆYˆ
^š[JBˆÂˆœš[Š^š[K‰[H	]H	K™ˆ	K™—ˆ‹
[œÚYÛ™YÛ™ÈÛ™ÊT‹O™œ˜[YKˆ‹O™^˜]ÜËÝX›J‹O™^Z[ŠKÝX›J‹O™^X^
JNÂˆËÈ™\Ù][˜ÛÛ™][Û˜[K[˜ÛY[™ÈÚ[ˆHš[HÛÝ[›Ý™HÜ[™YÛÈBˆËÈÛÝ[\œÈ™]™\ˆXØÝ[][]HXÜ›ÜÜÈœ˜[Y\È[ˆH[ˆ]\È›Ý˜XÚ[™Ë‚ˆBˆ‹O™^˜]ÜÈHÂ‚ˆÝ]XÈ’SJˆÝ]Ñš[HH[ŽÂˆÝ]XÈ›ÛÛÝ]ÕšYYH˜[ÙNÂˆYˆ
\Ý]ÕšYY
BˆÂˆÝ]ÕšYYHYNÂˆYˆ
ÛÛœÝÚ\Šˆ]H[ŠÖ—Õ’×Ñ”SQWÔÕUÈŠJBˆÂˆÝ]Ñš[HH›Ü[Š]ÈŠNÂˆYˆ
Ý]Ñš[JBˆœš[ŠÝ]Ñš[KˆˆÈœ˜[YH˜]ÜÈ™\XÙ\È˜]Ñš[™Ù\œš[Ø[Y\˜Qš[™Ù\œš[‚ˆÚYZYÚÛÝ™\˜YÙTÝYX[“[XH\Ý[˜ÝÛÛÝ\œÈ^[\Ú‚ˆœÝ\™•ÈÝ\™’Ý\™ÛÝ™\˜YÙTÝÝ\™“YX[“[XHÝ\™‘\Ý[˜Ý‚ˆœÝ\™’\Ú\ÙX×ˆŠNÂˆ[ÙBˆœš[ŠÝ\œ‹–Ýš×HØ[››ÝÜš]HÖ—Õ’×Ñ”SQWÔÕUÏI\×ˆ‹]
NÂˆBˆBˆËÈÖ—Õ’×Ñ”SQWÔÕU×ÔÕT‘PÑOO^ˆ8 %YX\Ý\™HU™\ÛÛ™HÝ\™˜XÙH\ÈÙ[\ÈBˆËÈ™\Ù[Yœ˜[YK‚ˆËÂˆËÈ\È\È›ÝH™Yš[™[Y[]\ÈH[™È]XZÙ\ÈHY]šXÈÛÜšÈ][ˆBˆËÈš\œÝ™\œÚ[ÛˆYX\Ý\™YÛ›HH™\Ù[Yœ›ÛY™™\‹ÚXÚ]H]HØÜ™Y[‚ˆËÈ\ÈHÙÛÈ\˜Nˆ[ÜÝHRK‹LÍ‰HÛÝ™\™Yˆ\ØX›[™ÈHM‹Xš]^ÛÛÜ™ˆËÈ[œÝÚ^ž›H8 %HÚ[™ÙH]ÝXÚ\ÈÍ‹N˜]ÜÈH[ˆ8 %[Ý™Y]žHŒBˆËÈ\˜Ù[YÙHÚ[ËK™KˆHY]šXÈÛÝ[›ÝÙYHHY™XÝ]Ø\ÈZ[ÈØ]ÚˆËÈ™XØ]\ÙHHY™XÝ]™\ÈÛˆHÐÑS‘HÝ\™˜XÙH[™HY]šXÈØ\ÈÛÚÚ[™È]BˆËÈÝ™\›^KˆÛÝÚHÌˆH\Ý]\È™]™\ˆ˜Z[Y\È›Ý™Y[ˆÚÝÛˆØ\X›HÙ‚ˆËÈ˜Z[[™Ë[™\ÈÛ™HØ\ÈÚÝÛˆ[˜Ø\X›K‚ˆËÂˆËÈÙ]]ÈHØÙ[™IÜÈ™\ÛÛ™H\Ý[˜][Û‹ˆ]H]HØÜ™Y[ˆ]\ÂˆËÈ
ŠŒŽŒ
Šˆ8 %“Õ‘MÚXÚØ\ÈÜš][ˆ\™H[™][ÝY›Ú™XÝ]ÚYHœ›ÛBˆËÈ\ÙHHÈ\LÈ[™\ÈHØÙ[™HT	ÜÈš\œÝ[Kˆ][ÛÛÝ\ˆ^[ÂˆËÈÛ›H™XØ]\ÙHÝ\ˆ™\ÛÛ™HÛÜYYHÛÛÝ\ˆY™™\ˆ›Üˆ\™\ÛÛ™\ÈÛËÛÈBˆËÈÜ›Û™ÈX™[Ø\ÈÛÛ™š\›YY]™\žH[YH]Ø\ÈÚXÚÙY
\MÛÝÚHŒJK‚ˆËÈÖ—Õ’×Ô‘TÓÓ‘WÕPÑH˜[Y\ÈHšYÚY™\ÜÈ›Üˆ[žH\˜K‚ˆÝ]XÈÛÛœÝÚ\ŠˆÝ\™˜XÙQ[ˆH[ŠÖ—Õ’×Ñ”SQWÔÕU×ÔÕT‘PÑHŠNÂˆÝ]XÈÛÛœÝZ[Ì—ÝÝ]ÔÝ\™˜XÙHBˆÝ\™˜XÙQ[ˆÈZ[Ì—Ý
ÝÝ[
Ý\™˜XÙQ[‹[‹MŠJH	ˆQ‘‘‘‘‘‘ˆˆÂˆÝŽ™XÝÜZ[ÝˆÝ\™˜XÙT^[ÎÂˆZ[Ì—ÝÝ\™˜XÙUÈHÝ\™˜XÙRHÂˆYˆ
Ý]Ñš[H	‰ˆÝ]ÔÝ\™˜XÙJBˆÂˆËÈ›È\š][ˆHÙ^K[X™\˜][Nˆ˜ÛÝ™\˜YÙHˆ[™›YX[ˆ[Z[˜[˜ÙHˆÈ›ÝˆËÈYX[ˆ[ž][™ÈÝ™\ˆH\Ý\™˜XÙK[™HÛ˜\ÚÝX\	ÜÈÙ^HØ\œšY\ÈBˆËÈ\Ý[˜Ý[ÛˆÛÈHY]šXÈØ[››ÝXØÚY[[H™XYÛ™H›ÝYÚHÜ›Û™ÂˆËÈ[XYÙH\ÜXÝ‚ˆ]]ÈÚ]H‹OœÛ˜\ÚÝË™š[™
Ý]ÔÝ\™˜XÙJNÂˆYˆ
Ú]OH‹OœÛ˜\ÚÝË™[™

JBˆÂˆYˆ
×ØÛÜPÙ[œÝ\ÓÛŠBˆÛÜPÙ[œÝ\ÔØ[\Y
Ý]ÔÝ\™˜XÙJNÂˆÛÛœÝZ[ÝˆBˆZ[Ý
Ú]OœÙXÛÛ™š[XYÙKÚY
H
ˆÚ]OœÙXÛÛ™š[XYÙKšZYÚ
ˆÂˆYˆ
ˆH‹Oœ™XY˜XÚËœÚ^™JBˆÂˆ[’[[YYX]JÉ—JšÐÛÛ[X[™Y™™\ˆØŠHÂˆ˜\œšY\ŠØ‹Ú]OœÙXÛÛ™š[XYÙK’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÔÔ×ÓÔSPSˆ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’U
NÂˆšÐY™™\’[XYÙPÛÜHÞßNÂˆËš[XYÙTÝXœ™\ÛÝ\˜ÙHHÈ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’UHNÂˆËš[XYÙQ^[HÈÚ]OœÙXÛÛ™š[XYÙKÚYÚ]OœÙXÛÛ™š[XYÙKšZYÚˆHNÂˆšÐÛYÛÜR[XYÙUÐY™™\ŠØ‹Ú]OœÙXÛÛ™š[XYÙKš[XYÙKˆ’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÔÔ×ÓÔSPSˆ‹Oœ™XY˜XÚË˜Y™™\‹K	˜ÊNÂˆ˜\œšY\ŠØ‹Ú]OœÙXÛÛ™š[XYÙKˆ’×ÒSPQÑWÓVSÕUÔÒQT—Ô‘PQÓÓ“WÓÔSPSˆ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’U
NÂˆJNÂˆÝ\™˜XÙT^[Ë˜\ÜÚYÛŠ‹Oœ™XY˜XÚË›X\Y‹Oœ™XY˜XÚË›X\Y
ÈŠNÂˆÝ\™˜XÙUÈHÚ]OœÙXÛÛ™š[XYÙKÚYÂˆÝ\™˜XÙRHÚ]OœÙXÛÛ™š[XYÙKšZYÚÂˆBˆBˆB‚ˆYˆ
	‰ˆÝ]Ñš[JBˆÂˆ›Ù”ØÛÜHÙœÊ	™×Ü›Ù‹™œ˜[YTÝ]ÊNÂˆZ[Ý]H[XTÝ[HHHÐ‘ŒŽPÑMŒŒŒÌ][ÂˆËÈ\Ý[˜ÝÛÛÝ\œÈ^XÝKÚ]Ý]H\ÚÙ]ˆHœ˜[YH\È‘ÐN[™BˆËÈ—ŒXš]š]X\\ÈˆP‹ÚXÚ\ÈÚX\\ˆ[ˆH\ÚX›H\ˆœ˜[YH[™ˆËÈÚ]™\È[ˆ^XÝÛÝ[˜]\ˆ[ˆ[ˆ\Ý[X]K‚ˆÝ]XÈÝŽ™XÝÜZ[ÝˆÙY[š]ÎÂˆÙY[š]Ë˜\ÜÚYÛŠ]HN
NÈËÈ—Œš]ÂˆZ[Ý\Ý[˜ÝHÂˆ›Üˆ
Ú^™WÝHHÈHž]\ÎÈH
ÏH
BˆÂˆÛÛœÝZ[Ì—ÝˆHÚWKÈHÚH
ÈWKˆHÚH
È—NÂˆÛÛœÝZ[Ì—Ý™ØˆH
ˆMŠH
È
HŽÂˆYˆ
™ØŠBˆ
ÊÛ]Âˆ[XTÝ[H
ÏH
ˆ
ˆM
ÈÈ
ˆNÈ
Èˆ
ˆNJHˆÂˆÛÛœÝZ[Ì—ÝÛÜ™H™Øˆˆ‹š]H™Øˆ	ˆŒÎÂˆYˆ
JÙY[š]ÖÝÛÜ™H	ˆ
][š]
JJBˆÂˆÙY[š]ÖÝÛÜ™HH][š]Âˆ
ÊÙ\Ý[˜ÝÂˆBˆH™ØŽÂˆ
HLPŒÝ[ÂˆBˆÛÛœÝZ[Ý^[ÈHž]\ÈÈÂˆËÈH˜[YYÝ\™˜XÙKYX\Ý\™YHØ[YHØ^Kˆ™\›ÜÈÚ[ˆ]Ø\È›Ý™\]Y\ÝYÜ‚ˆËÈÙ\È›Ý^\Ý\Èœ˜[YKÚXÚœ˜[YWØÛÛ\\™KœH™XYÈ\È››ÈÝ\™˜XÙH]H‚ˆËÈ˜]\ˆ[ˆ\È[ˆ[\HÝ\™˜XÙK‚ˆZ[ÝÛ]HÛ[XTÝ[HHÜHÐ‘ŒŽPÑMŒŒŒÌ][Ù\Ý[˜ÝHÂˆYˆ
\Ý\™˜XÙT^[Ë™[\J
JBˆÂˆÙY[š]Ë˜\ÜÚYÛŠ]HN
NÂˆ›Üˆ
Ú^™WÝHHÈHÝ\™˜XÙT^[ËœÚ^™J
NÈH
ÏH
BˆÂˆÛÛœÝZ[Ì—ÝˆHÝ\™˜XÙT^[ÖÚWKÈHÝ\™˜XÙT^[ÖÚH
ÈWKˆˆHÝ\™˜XÙT^[ÖÚH
È—NÂˆÛÛœÝZ[Ì—Ý™ØˆH
ˆMŠH
È
HŽÂˆYˆ
™ØŠBˆ
ÊÜÛ]ÂˆÛ[XTÝ[H
ÏH
ˆ
ˆM
ÈÈ
ˆNÈ
Èˆ
ˆNJHˆÂˆÛÛœÝZ[Ì—ÝÛÜ™H™Øˆˆ‹š]H™Øˆ	ˆŒÎÂˆYˆ
JÙY[š]ÖÝÛÜ™H	ˆ
][š]
JJBˆÂˆÙY[š]ÖÝÛÜ™HH][š]Âˆ
ÊÜÙ\Ý[˜ÝÂˆBˆÜH™ØŽÂˆÜ
HLPŒÝ[ÂˆBˆBˆÛÛœÝZ[ÝÜ^[ÈHÝ\™˜XÙT^[ËœÚ^™J
HÈÂˆËÈZ[\ÙXÛÛ™ÈÚ[˜ÙHHš\œÝYX\Ý\™Yœ˜[YH8 %TS‘QÛÈ]™\žHÛÛ[[ˆ[™^ˆËÈ[žH^\Ý[™ÈÛÛ™XYÈ\È[˜Ú[™ÙY‚ˆËÂˆËÈ]^\ÝÈ™XØ]\ÙHHœ˜[YH˜]HÙˆ[ˆTH\È›ÝH[X™\ˆ\È›Ú™XÝÛÝ[ˆËÈ™]š[Ý\ÛHÝ]Kˆ]™\žHœ˜[YK\˜]HšYÝ\™H]ÝÛœÈ]šY\ÈHÚÛH[‰ÜÈœ˜[YBˆËÈÛÝ[žH]ÈØ[[YKÚXÚ›ÜˆH[ˆ]›ÛÝËØ[ÜÈ›Ý\ˆY[\ËØYËˆËÈ[™Û›H[ˆ^\È\È[ˆ]™\˜YÙHÝ™\ˆ\˜\È]Y™™\ˆžH[Ü™H[ˆBˆËÈY™™XÝ[ž[Û™HØ[ÈÈYX\Ý\™KˆŽLLˆœÈ[ˆØ[Y\^HˆØ\È[ˆÜ\˜]Ü‰ÜÂˆËÈÝÜØ]ÚˆÚ]H[Y\Ý[\\ˆœ˜[YK[žH\˜HHØ[Y\˜Hš[™Ù\œš[ÜˆBˆËÈ˜]ÈÛÝ[Ø[ˆ[[Z]\È]ÈÝÛˆYX\Ý\˜X›H˜]Kœ›ÛHH[ˆ]Ø\ÂˆËÈ[™XYH™Z[™ÈXYH›Üˆ[›Ý\ˆ™X\ÛÛ‹‚ˆÝ]XÈÛÛœÝ]]ÈÝ]ÕHÝŽ˜Ú›Û›ÎŽœÝXYWØÛØÚÎŽ››ÝÊ
NÂˆÛÛœÝÛ™ÈÛ™È\ÙXÈHÝŽ˜Ú›Û›ÎŽ™\˜][Û—ØØ\ÝÝŽ˜Ú›Û›ÎŽ›Z[\ÙXÛÛ™ÏŠˆÝŽ˜Ú›Û›ÎŽœÝXYWØÛØÚÎŽ››ÝÊ
HHÝ]Õ
K˜ÛÝ[

NÂˆœš[ŠÝ]Ñš[Kˆ‰[H	[H	[H	LM›	LM›	]H	]H	Kˆ	KŒÙˆ	[H	LM›‚ˆ‰]H	]H	Kˆ	KŒÙˆ	[H	LM›	[ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™œ˜[YK
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™˜]ÜËˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™\XÙ\Ëˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë™˜]Ñš[™Ù\œš[ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™\Ë˜Ø[Y\˜Qš[™Ù\œš[ÚYKZYÚKˆ^[ÈÈLŒ
ˆÝX›J]
HÈÝX›J^[ÊHˆŒˆ^[ÈÈÝX›J[XTÝ[JHÈÝX›J^[ÊHˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊY\Ý[˜Ý
[œÚYÛ™YÛ™ÈÛ™Ê\ˆÝ\™˜XÙUËÝ\™˜XÙRˆÜ^[ÈÈLŒ
ˆÝX›JÛ]
HÈÝX›JÜ^[ÊHˆŒˆÜ^[ÈÈÝX›JÛ[XTÝ[JHÈÝX›JÜ^[ÊHˆŒˆ
[œÚYÛ™YÛ™ÈÛ™Ê\Ù\Ý[˜ÝˆÜ^[ÈÈ
[œÚYÛ™YÛ™ÈÛ™Ê\Üˆ[\ÙXÊNÂˆ™›\Ú
Ý]Ñš[JNÂˆBˆËÈHš[™Ù\œš[[™™\^XØÝ[][]ÜœÈ\™H™\Ù]]HÕP“RUX›Ý™K›Ý\™N‚ˆËÈ^H™[Û™ÈÈHœ˜[YH]Ø\È\Ý™XÛÜ™Y[™]™\ž][™Èœ›ÛHH™\Ù[ˆËÈÛØ\™È\ÈX›Ý]HY™™\™[Û™K‚‚ˆËÈHœ˜[YH]\È[\™[HÛ™HÛÛÝ\ˆ\ÈHÚ[™ÛH[ÜÝÛÛ[[ÛˆÜ›Û™È™\Ý[BˆËÈ™[™\™\ˆ›ÙXÙ\Ë[™]\È[š\ÚX›H[ˆHÙËˆÛÝ[[™È]XZÙ\ÈHXÝ\™BˆËÈ\È›XÚÈˆH[X™\ˆ˜]\ˆ[ˆH™\Ü8 %[™Ù\\˜][™È˜›XÚÈˆœ›ÛHœÛÛYBˆËÈ[šY›Ü›HÛÛÝ\ˆˆÙ\\˜]\ÈHZ\ÜÚ[™È˜]Èœ›ÛHHÛX\ˆ]˜[ˆ[™›Ý[™È[ÙK‚ˆYˆ

BˆÂˆZ[Ì—Ýš\œÝHÂˆY[XÜJ	™š\œÝ
NÂˆ›ÛÛ[šY›Ü›HHYNÂˆ›Üˆ
Ú^™WÝHHÈHž]\È	‰ˆ[šY›Ü›NÈH
ÏH
Bˆ[šY›Ü›HHY[XÛ\
	œÚWK	™š\œÝ
HOHÂˆYˆ
[šY›Ü›JBˆÛÝ[
š\œÝOH‘ŒHš\œÝOHÈ™œ˜[YNˆ[šY›Ü›[H›XÚÈ‚ˆˆ™œ˜[YNˆ[šY›Ü›[HÛ™HÛÛÝ\ˆŠNÂˆ[ÙBˆÛÝ[
™œ˜[YNˆ\ÈÛÛ[ŠNÂˆB‚ˆËÈÖ—Õ’×ÔÓTÑSTO\ˆ8 %Üš]HU‘T–H™\ÛÛ™HÛ˜\ÚÝÙˆÛ™Hœ˜[YH\ÈHK‚ˆËÂˆËÈH]Y\Ý[Ûˆ\È[œÝÙ\œÈ\ÈÚ\™H[ˆHÚZ[ˆYHXÝ\™HÛÏÈ‹[™]\ÂˆËÈHÛ›H[œÝ[Y[]Ø[ŽˆHœ˜[YH\ÈH\Ý[šËÛÈHÜ›Û™Èœ˜[YH\ÂˆËÈÛÛœÚ\Ý[Ú]]™\žH\ÜÈ™Z[™ÈÜ›Û™È[™Ú]^XÝHÛ™H™Z[™ÈÜ›Û™Ëˆ[\[™ÂˆËÈ[Ùˆ[H\›œÈ][ÈH\™XÝÜžH[ÝHØ[ˆÛÚÈ]‚ˆËÈÖ—Õ’×ÔÓTÑ”SQOSˆXÚÜÈHœ˜[YKˆ]Ø\ÈH\™ÛÙYŒ›Üˆ\ÈÛ™È\ÈBˆËÈ[œÝ[Y[^\ÝYÚXÚØ\Èš[™HÚ[H]™\žH]Y\Ý[ÛˆØ\ÈX›Ý]H]BˆËÈØÜ™Y[ˆ8 %[™\Ù[\ÜÈH[ÛY[Û™HØ\È›Ýˆ\ÙHÈ\L‰ÜÈY™XÝ\ÈÛˆBˆËÈY[HÛÈ™\ÜÙ\È\ÝH]KK™Kˆ]Ú]]™\ˆœ˜[YHHÞ[]XËZ[œ]\›BˆËÈ\[œÈÈ[™Û‹[™H\[™[˜ÞHÜ˜\ÙˆHÜ›Û™Èœ˜[YH[œÝÙ\œÈ›Ý[™Ë‚ˆËÈÖ—ÐÐTT‘WÒÑVOO\ˆ8 %Ó‘H‘TÔËÓ‘HPÑKU‘T–HT•QPÕ[ÈÛ™H\™XÝÜžK‚ˆËÂˆËÈH™YH[œÝ[Y[È][œÝÙ\ˆÚ]Ù\È\ÈÝ\™˜XÙHÛÚÈZÙH[™ÚHˆÙ\™BˆËÈ™YH[š\›Û›Y[˜\šXX›\ÈÜš][™ÈÈ™YHXÙ\Ë[™ÛÈÙˆ[Hš\™HÛˆBˆËÈœ˜[YH•SP‘Tˆ˜]\ˆ[ˆÛˆHÜ\˜]Ü‹ˆÝ[™[™È[ˆœ›ÛÙˆHY™XÝÚ]BˆËÈ›ÛXšYHÚ]Ú[™ÈÛˆ[ÝH\È›ÝH[ÛY[È™H™XY[™ÈHœ˜[YHÛÝ[\ˆÝ]ÙˆBˆËÈ]H˜\‹[™H\ÈÙˆ\È›Ú™XÝ]™YY[ˆÜ\˜]Üˆ\™HH\È]ˆËÈ]™HÈÛÜÝ[HHX\Ý
ÛÝÚHNL
KˆÛÈ\ÈÙ]È[™YH]Û˜ÙH[™˜[Y\ÂˆËÈ]™\žHš[HY\ˆHØ[YHœ˜[YN‚ˆËÂˆËÈØ\\™WÏœ˜[YO‹œHH™\Ù[YXÝ\™BˆËÈØ\\™WÏœ˜[YO‹˜Ù[œÝ\È]™\žH˜]ÎˆÚY\œË™]ÚÛÝËY™\ÜÙ\ËSQS”ÒSÓ‚ˆËÈÛ˜\Ê‹œH]™\žH™\ÛÛ™HÛ˜\ÚÝÙˆ]œ˜[YBˆËÂˆËÈHÙ[œÝ\ÈÛÛ[[œÈ\™H[X™\˜][HHØ[YHšY[ÈÛÛËÞ—Ù˜]×Øš[™[™ÜËœXˆËÈš[È›ÜˆH[šXHž˜ÛÈÝ\œÈ[™\™Ø\™IÜÈØ[ˆ™H™XYÚYHžHÚYH8 %ÚXÚˆËÈ\ÈHÚÛHÚ[[™]\ÈÚ]\ÈYÈÈžH[™‚ˆÝ]XÈÛÛœÝÚ\ŠˆØ\\™Q\ˆH[ŠÖ—ÐÐTT‘WÒÑVHŠNÂˆÝ]XÈÛÛœÝ›ÛÛØ\\™Q\”™XYHH×HÂˆYˆ
ÛÛœÝÚ\ŠˆH[ŠÖ—ÐÐTT‘WÒÑVHŠJBˆÂˆÝŽ™\œ›Ü—ØÛÙHXÎÂˆÝŽ™š[\Þ\Ý[NŽ˜Ü™X]WÙ\™XÝÜšY\ÊXÊNÂˆœš[ŠÝ\œ‹–Ýš×HÖ—ÐÐTT‘WÒÑVH\›YYˆ™\ÜÈŽHÈØ\\™HHXÝ\™K‚ˆH\‹Y˜]ÈÙ[œÝ\È[™]™\žH™\ÛÛ™HÛ˜\ÚÝÙˆÛ™Hœ˜[YH‚ˆš[È	\×ˆ‹
NÂˆBˆ™]\›ˆYNÂˆJ
NÂˆ
›ÚY
XØ\\™Q\”™XYNÂˆÝ]XÈÛÛœÝÚ\ŠˆÛ˜\\ˆHØ\\™Q\ˆÈØ\\™Q\ˆˆ[ŠÖ—Õ’×ÔÓTÑSTŠNÂˆÝ]XÈÛÛœÝZ[ÝÛ˜\œ˜[YHBˆ[ŠÖ—Õ’×ÔÓTÑ”SQHŠHÈÝÝ[
[ŠÖ—Õ’×ÔÓTÑ”SQHŠK[‹L
HˆŒÂˆËÈŽH8 %HÜ\˜]Ü‰ÜÈÝÛˆšYÙÙ\‹ÛÛœÝ[YY\™Kˆ\ÚÙY›Üˆœ›ÛH[œÚYHHØ[YKˆËÈÝ[™[™ÈÛˆHY™XÝØZ][™È›ÜˆHœ˜[YHÛÝ[\ˆ[ˆH]H˜\ˆÈ™XXÚBˆËÈ[X™\ˆÚÜÙ[ˆ™Y›Ü™HH[ˆÝ\YˆHš^YÖ—Õ’×ÔÓTÑ”SQX\ÈHš[™HšYÙÙ\‚ˆËÈ›ÜˆH›ÛÝ][YH]Y\Ý[Ûˆ[™HÜ›Û™ÈÛ™H›Üˆ[žH]Y\Ý[ÛˆX›Ý]HPÑKˆHYÙBˆËÈ\ÈÛÛœÝ[YY[˜ÛÛ™][Û˜[HÛÈH™\ÜÈØ[››ÝÚ]]ÚY[™š\™HÛˆÛÛYH]\‚ˆËÈœ˜[YK[™H™\ÜÈÚ]›È\Ý[˜][ÛˆÐVTÈÓÈ˜]\ˆ[ˆÚ[™È›Ý[™Èš\ÚX›H8 %ˆËÈ[ˆ[œÝ[Y[]Ú[[HXÛ[™\È\ÈH˜Z[\™HÚ\H\È›Ú™XÝÙY\È^Z[™ÂˆËÈ›Üˆ
ÛÝÚ\ÈËMLJK‚ˆÛÛœÝ›ÛÛÛ˜\Ù^HHÜÝÐÛÛœÝ[YTÛ˜\[\™\ÜÙY

NÂˆ›ÛÛÛ˜\Ù^S›ÝÈHÛ˜\Ù^NÈËÈÙYHHÖ—ÐÐTT‘WÒÑVH›ÝH]H[\ˆËÈHÐSQH™\ÜÈ[ÛÈ\›\ÈH\‹Y˜]ÈÙ[œÝ\È›ÜˆH‘Vœ˜[YH8 %™^›Ý\ÂˆËÈÛ™K™XØ]\ÙH\Èœ˜[YIÜÈ˜]ÜÈ\™H[™XYH™XÛÜ™YžHH[YHH™\Ù[\ÂˆËÈ™XXÚYˆÛ™H™\ÜÈ\™Y›Ü™HZY[ÈÛÈšY]ÜÈÙˆÛ™HXÙNˆ]™\žHÝ\™˜XÙH[ˆBˆËÈœ˜[YH
HÛ˜\ÚÝÊH[™]™\žH˜]È]Z[]‚ˆÝ]XÈÝŽœÝš[™ÈØ\\™PÙ[œÝ\ÈBˆØ\\™Q\ˆÈÝŽœÝš[™ÊØ\\™Q\ŠH
È‹ØØ\\™K˜Ù[œÝ\ÈˆˆÝŽœÝš[™Ê
NÂˆÝ]XÈÛÛœÝÚ\ŠˆÙ[œÝ\Ô]BˆØ\\™Q\ˆÈØ\\™PÙ[œÝ\Ë˜×ÜÝŠ
Hˆ[ŠÖ—Õ’×ÑU×ÐÑS”ÕTÈŠNÂˆYˆ
Û˜\Ù^H	‰ˆÙ[œÝ\Ô]	‰ˆT‹O™˜]ÐÙ[œÝ\Ñœ˜[YJBˆÂˆ‹O™˜]ÐÙ[œÝ\Ñœ˜[YHH‹O™œ˜[YH
ÈNÂˆËÈHPÕT‘HÙˆHØ[YHœ˜[YKÚXÚ\ÈH\Y˜XÝHÝ\ˆÛÈ^\ÝÂˆËÈ^Z[ˆ[™HÛ›HÛ™H]Ø\ÈÝ[ÛˆHš^Y[\˜[ˆ\›YY›ÜˆBˆËÈ™^œ˜[YH›ÜˆHØ[YH™X\ÛÛˆHÙ[œÝ\È\Îˆ\Èœ˜[YIÜÈ˜]ÜÈ\™H[™XYBˆËÈ™XÛÜ™YžHH[YHH™\Ù[\È™XXÚYÛÈHXÝ\™HZÙ[ˆ›ÝÈ[™HÙ[œÝ\ÂˆËÈZÙ[ˆ™^œ˜[YHÛÝ[™HÛÈY™™\™[[ÛY[È\ØÜšX™Y\ÈÛ™K‚ˆ‹O˜Ø\\™TXÝ\™Qœ˜[YHH‹O™œ˜[YH
ÈNÂˆËÈT“HH‘TÑS•‘PQPÒÈ›ÜˆHœ˜[Y\È\ÈØ\\™H™YYÈ
\Íˆ][HJK‚ˆËÈ[ˆHÝØ\ÚZ[ˆ\›HH™XY˜XÚÈ\ÈÙ™ˆžHY˜][›ÝËÛÈÚ]Ý]\ÈBˆËÈØ\\™HÛÝ[š[™[[™Üš]HHÙ[œÝ\ÈÚ]›ÈXÝ\™H™\ÚYH]‚ˆËÈœ˜[Y\Ò[‘›YÚ
È˜˜]\ˆ[ˆ^XÝHÛ™Hœ˜[YNˆHXÝ\™H\ÈÛÛœÝ[YYˆËÈÚ[ˆH\›YYœ˜[YH‘UT‘TËÚXÚ\ÈÛ™HÈÛÈÝØ\È]\ˆ\[™[™ÈÛˆBˆËÈš[™È\[™H™]È^˜HÚÛKYœ˜[YHÛÜY\ÈÛ˜ÙH\ˆÙ^H™\ÜÈ\È›ÝBˆËÈ]X[]HÛÜ™Z[™È^XÝX›Ý]ˆ]\ÈHX^

HÛÈÛÈ™\ÜÙ\È[ˆ]ZXÚÂˆËÈÝXØÙ\ÜÚ[ÛˆØ[››ÝÚÜ[ˆXXÚÝ\‹‚ˆ‹Oœ™XY˜XÚÕ[[œ˜[YHBˆÝŽ›X^
‹Oœ™XY˜XÚÕ[[œ˜[YK‹O™œ˜[YH
È‹O™œ˜[Y\Ò[‘›YÚ
ÈŠNÂˆ
ÊÔ‹Oœ™XY˜XÚÐ\›YY™\ÜÙ\ÎÂˆËÈÖ—Õ’×ÑU×ÒQLH8 %HÑS”ÕTÈ”SQHUÑSˆZ[È˜]È[™XÙ\È[œÝXYÙ‚ˆËÈÛÛÝ\œË[™]]\Ý™HHØ[YHœ˜[YNˆH˜]È[™^\ÈÛ›HYX[š[™Ù[ˆËÈYØZ[œÝH˜]È\Ý]Ø\È[X™\™Y[‹ˆHš\œÝ™\œÚ[Ûˆ\›YYH‘VˆËÈœ˜[YHÛÈ]Û™H™\ÜÈÛÝ[ZY[›ÝHXÝ\™H[™HX\[™H™\žBˆËÈš\œÝ™XYÚÝÙYÚH]\ÈÜ›Û™È8 %HÜš\ÚX›Hˆ˜]ÜÈ™\ÛÛ™YÂˆËÈÙ[œÝ\È[™\ÈÚ]X\ÚÏL˜]ÜÈ]Üš]H›ÈÛÛÝ\ˆ][™XØ]\ÙH[™^ˆËÈMÙˆÛ™Hœ˜[YH\È›Ý[™^MÙˆH™^ˆÛ™Hœ˜[YKÛ™H[X™\š[™Ë‚ˆËÂˆËÈHÛÜÝ\È]\È™\ÜÈZY[È›È\ØX›HPÕT‘H
HÜÝÚZ[ˆ\ÈXYBˆËÈÙˆ˜]ÜÈÛËÛÈ]Z[È]ÈÝÛˆ[™XÙ\ÈÝ™\ˆ]™\ž][™ÊKˆ]\ÈØZYÝ]ˆËÈÝY™[ÝÈ˜]\ˆ[ˆY›ÜˆÛÛY[Û™HÈ\ØÛÝ™\ˆ[ˆHš[K‚ˆYˆ
[“ÛŠÖ—Õ’×ÑU×ÒQŠH	‰ˆ‹O™˜]ÒY[Ù[JBˆÂˆ‹O™˜]ÒY\›YYHYNÂˆœš[ŠÝ\œ‹–Ýš×HŽNˆH™^™XÛÜ™Yœ˜[YHÚ[™HHUËRQX\‚ˆŠ™XY]Ú]ÛÛËÙ˜]ÚYÜ™XYœJWˆŠNÂˆBˆœš[ŠÝ\œ‹–Ýš×HŽNˆØ\\š[™Èœ˜[YH	[HOˆXÝ\™K	[KY˜]ÈÙ[œÝ\È[™‚ˆ™]™\žH™\ÛÛ™HÛ˜\ÚÝˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O™˜]ÐÙ[œÝ\Ñœ˜[YKˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O™˜]ÜÕ\Ñœ˜[YJNÂˆBˆYˆ
Û˜\Ù^H	‰ˆ\Û˜\\ˆ	‰ˆXÙ[œÝ\Ô]
BˆÂˆÝ]XÈ›ÛÛÛÛ\Z[™YH˜[ÙNÂˆYˆ
XÛÛ\Z[™Y
BˆÂˆÛÛ\Z[™YHYNÂˆœš[ŠÝ\œ‹–Ýš×HŽH™\ÜÙY]™Z]\ˆÖ—Õ’×ÔÓTÑST›Üˆ‚ˆÖ—Õ’×ÑU×ÐÑS”ÕTÈ\ÈÙ]8 %›Ý[™ÈØ\È[\YˆŠNÂˆBˆBˆËÈÛÜÙHHÙ[œÝ\È]HS‘ÙˆHœ˜[YH]ÛÝ™\™Y[™Ø^HÝÈX[žH˜]ÜÈ]Ø]ÂˆËÈ8 %HÙ[œÝ\ÈÚÜÙHš[H^\ÝÈ]\ÈÚÜ\ÈÝ\Ú\ÙH[™\Ý[™ÝZ\ÚX›Hœ›ÛHÛ™BˆËÈ]˜[ˆÛˆHœ˜[YHÚ]›Ý[™È[ˆ]‚ˆYˆ
‹O™˜]ÐÙ[œÝ\Ñœ˜[YH	‰ˆ‹O™œ˜[YHˆ‹O™˜]ÐÙ[œÝ\Ñœ˜[YH	‰ˆ‹O™˜]ÐÙ[œÝ\Ñš[JBˆÂˆ˜ÛÜÙJ‹O™˜]ÐÙ[œÝ\Ñš[JNÂˆ‹O™˜]ÐÙ[œÝ\Ñš[HH[ŽÂˆËÈHÛÝ[\ÈHÙ[œÝ\ÉÜÈÕÓˆ[™HÛÝ[\‹›Ý˜]ÜÕ\Ñœ˜[YX8 %]\ÂˆËÈ[™XYH™Y[ˆ™\Ù]žHHœ˜[YH›Ý[™\žHÙH\™HÝ[™[™ÈÛ‹[™š[[™È]ˆËÈ\™HÛÝ[™\Ü™\›È›ÜˆHÙ[œÝ\È]ÛÜšÙY\™™XÝK‚ˆœš[ŠÝ\œ‹–Ýš×H˜]ÈÙ[œÝ\ÈÜš][Žˆ	[H˜]ÜÈÙˆœ˜[YH	[Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O™˜]ÐÙ[œÝ\Ó[™\Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O™˜]ÐÙ[œÝ\Ñœ˜[YJNÂˆ‹O™˜]ÐÙ[œÝ\Ó[™\ÈHÂˆ‹O™˜]ÐÙ[œÝ\Ñœ˜[YHHÂˆBˆËÈÓ‘H‘TÔÈUTÕQPSˆÓ‘H”SQKˆHÛ˜\ÚÝ[\\ÙYÈš\™HÛˆH™\ÜÈ]Ù[‹ˆËÈK™KˆÛˆHœ˜[YH‘Q“Ô‘HHÛ™HHÙ[œÝ\È[™HXÝ\™HÛÝ™\ˆ8 %ÛÈHØ\\™BˆËÈ›ÙXÙY™YH\Y˜XÝÈX™[Y\ÈÛ™HXÙH[™\ØÜšXš[™ÈÛÈÛÛœÙXÝ]]™BˆËÈœ˜[Y\Ëˆ[ˆHÜ›ÝÙÜÙH\™HÛÈY™™\™[XÝ\™\Ë[™HÚÛH˜[YHÙˆBˆËÈØ\\™H\È]]È™YHšY]ÜÈ\™HÙˆHØ[YH[ÛY[ˆÛÈ[™\ˆÖ—ÐÐTT‘WÒÑVHBˆËÈ™\ÜÈ\›\ÈH[\›ÜˆH™^œ˜[YHZÙHHÝ\ˆÛÎÈHÝ[™[Û™BˆËÈÖ—Õ’×ÔÓTÑST]ÙY\È]ÈÛ[[YYX]H™Z]š[Ý\‹ÚXÚÙ]™\˜[™XÛÜ™YˆËÈYX\Ý\™[Y[ÈÙ\™HZÙ[ˆÚ]‚ˆYˆ
Ø\\™Q\ˆ	‰ˆÛ˜\Ù^JBˆÂˆ‹O˜Ø\\™TÛ˜\œ˜[YHH‹O™œ˜[YH
ÈNÂˆÛ˜\Ù^S›ÝÈH˜[ÙNÂˆBˆYˆ
‹O˜Ø\\™TÛ˜\œ˜[YH	‰ˆ‹O™œ˜[YHOH‹O˜Ø\\™TÛ˜\œ˜[YJBˆÂˆ‹O˜Ø\\™TÛ˜\œ˜[YHHÂˆÛ˜\Ù^S›ÝÈHYNÂˆBˆËÈ[™Hš^YYœ˜[YH[\\ÈÑ‘ˆ[™\ˆÖ—ÐÐTT‘WÒÑVNˆ]˜\šXX›HYX[œÈBˆËÈÜ\˜]ÜˆXÚY\ÈÚ[ˆ‹[™LÌš[\Èœ›ÛHœ˜[YHŒ[ˆHØ\\™H\™XÝÜžH\ÂˆËÈ›Ú\ÙHHÜ\˜]Üˆ[ˆ\ÈÈ[\\œ›ÛHZ\ˆÝÛˆ™\ÜË‚ˆËÈHUËRQ”SQHSTÈUÈÓTÒÕÈÓË[™]]\ÝˆHQ[XYÙHÛ›H^\ÝÂˆËÈ[ˆHÐÑS‘HÓÓÕT‹™Y›Ü™HHÜÝÚZ[‹ˆH™\Ù[YXÝ\™HØ[››ÝØ\œžH]ˆËÈ™XØ]\ÙHHÜÝ\ÜÙ\È\™H˜]ÜÈ\ÈÙ[[™ÛÝ[Z[Z\ˆÝÛˆ[™XÙ\ÈÝ™\‚ˆËÈHÚÛHØÜ™Y[ˆ8 %ÛÈHX\\ÈÈ™H™XYÙ™ˆH™\ÛÛ™K[™\È\ÈBˆËÈ[\]Üš]\È™\ÛÛ™\ÈÝ]‚ˆÛÛœÝ›ÛÛ˜]ÒY›ÝÈH‹O™˜]ÒYXÝ]™NÂˆYˆ
˜]ÒY›ÝÊBˆÂˆ‹O™˜]ÒY\›YYH˜[ÙNÂˆ‹O™˜]ÒYXÝ]™HH˜[ÙNÂˆ‹O™˜]ÒY˜[“Û‘œ˜[YHH‹O™œ˜[YNÂˆBˆYˆ
Û˜\\ˆ	‰ˆ

‹O™œ˜[YHOHÛ˜\œ˜[YH	‰ˆXØ\\™Q\ŠH›XÚÕ˜[œÚ][ÛˆˆÛ˜\Ù^S›ÝÈ˜]ÒY›ÝÊJBˆÂˆYˆ
˜]ÒY›ÝÊBˆœš[ŠÝ\œ‹–Ýš×HUËRQˆœ˜[YH	[H™[™\™Y	[H˜]ÜÈ\È[™XÙ\ÎÈ]È‚ˆœ™\ÛÛ™HÛ˜\ÚÝÈT‘HHQX\ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O™œ˜[YK
[œÚYÛ™YÛ™ÈÛ™ÊT‹O™˜]ÜÕ\Ñœ˜[YJNÂˆYˆ
Û˜\Ù^S›ÝÊBˆœš[ŠÝ\œ‹–Ýš×HŽNˆ[\[™È]™\žH™\ÛÛ™HÛ˜\ÚÝÙˆœ˜[YH	[Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O™œ˜[YJNÂˆËÈÔ‘PUHHT‘PÕÔ–K[™Ø^HÛÈYˆHš\œÝš[HÝ[Ø[››Ý™HÜš][‹‚ˆËÈÚ]Ý]\ÈH[\[››Ý[˜Ù\È™[\[™È]™\žH™\ÛÛ™HÛ˜\ÚÝÙˆœ˜[YHˆ‹ˆËÈ]\˜]\È]™\žHÝ\™˜XÙK[™Üš]\È“ÕS‘ÈÚ[ˆH\™XÝÜžH\ÈXœÙ[8 %ÚXÚˆËÈ\ÈÚ]\[™YHš\œÝ[YH]Ø\ÈÚ[Y]Hœ™\Ú][™]ÛÚÜÂˆËÈ^XÝHZÙHH™[™\™\ˆ]Y›ÈÛ˜\ÚÝÈÈÚ]™KˆÖ—ÔÒQT—ÑSTY\ÂˆËÈØ[YHY™XÝ[™\Hš^Y]\™NÈHš^™[Û™ÜÈ]]™\žH[\Ú]K‚ˆÂˆÝŽ™\œ›Ü—ØÛÙHXÎÂˆÝŽ™š[\Þ\Ý[NŽ˜Ü™X]WÙ\™XÝÜšY\ÊÛ˜\\‹XÊNÂˆBˆ›ÛÛÜ›ÝSÛ™HH˜[ÙNÂˆ›Üˆ
ÛÛœÝ]]ÉˆÙ\ÝÛ˜\š[™[™×Hˆ‹OœÛ˜\ÚÝÊBˆÂˆËÈHZ[ˆ™Y™\™[˜ÙH›ÜˆH[X™H™[ÝÎˆØ\\š[™ÈHÝXÝ\™Yš[™[™È\ÂˆËÈÊÊÌŒ
LLJH]Û[™ÈMH8 %H™[X\ÙIÜÈÛX˜\ÙHÛÛ\[\ˆ8 %™Z™XÝÈ]ˆËÈ[™\ÈØ\ÈHÛ™H[™H[ˆH™YH]YÛÈ
\L
K‚ˆÛÛœÝ]]ÉˆÛ˜\HÛ˜\š[™[™ÎÂˆÛÛœÝÚ^™WÝˆHÚ^™WÝ
Û˜\š[XYÙKÚY
H
ˆÛ˜\š[XYÙKšZYÚ
ˆÂˆYˆ
ˆˆ‹Oœ™XY˜XÚËœÚ^™JBˆÛÛ[YNÂˆÛÛœÝšÒ[XYÙP\ÜXÝ›YÜÈ\ÜXÝHÛ˜\™œ›ÛQ\ˆÈ’×ÒSPQÑWÐTÔPÕÑTÐ’Uˆˆ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’UÂˆ[’[[YYX]JÉ—JšÐÛÛ[X[™Y™™\ˆØŠHÂˆ[XYÙIˆ[YÈHÛÛœÝØØ\Ý[XYÙIŠÛ˜\š[XYÙJNÂˆ˜\œšY\ŠØ‹[YË’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÔÔ×ÓÔSPS\ÜXÝ
NÂˆšÐY™™\’[XYÙPÛÜHÞßNÂˆËš[XYÙTÝXœ™\ÛÝ\˜ÙHHÈ\ÜXÝHNÂˆËš[XYÙQ^[HÈ[YËÚY[YËšZYÚHNÂˆšÐÛYÛÜR[XYÙUÐY™™\ŠØ‹[YËš[XYÙK’×ÒSPQÑWÓVSÕUÕS”Ñ‘T—ÔÔ×ÓÔSPSˆ‹Oœ™XY˜XÚË˜Y™™\‹K	˜ÊNÂˆ˜\œšY\ŠØ‹[YË’×ÒSPQÑWÓVSÕUÔÒQT—Ô‘PQÓÓ“WÓÔSPS\ÜXÝ
NÂˆJNÂˆËÈH”SQH\È[ˆH˜[YH™XØ]\ÙHÖ—Õ’×ÔÓTÓÓ—Ð“PÒÈØ[ˆš\™HÛˆ[Ü™BˆËÈ[ˆÛ™Hœ˜[YK[™HÚZ[ˆ]Ú[[HÝ™\Ü›ÝHH˜[œÚ][Ûˆœ˜[YBˆËÈÚ]HÛ™HY\ˆ]ÛÝ[\Ý›ÞHHÛ›Hœ˜[YHÛÜ]š[™Ë‚ˆÚ\ˆ]ÍLL—NÂˆËÈHQœ˜[YIÜÈš[\È\™H˜[YY\\ÛÈH\™XÝÜžHÙˆÛ˜\ÚÝÈØ[››ÝˆËÈ™HZ\Ü™XYˆ[ˆQX\ÛÚÜÈZÙHHØ\š\ÚÛÛÝ\ˆ›Ú\ÙH[XYÙK[™Z\ÝZÚ[™ÂˆËÈÛ™H›ÜˆHXÝ\™H\È^XÝHHÛÜÙˆÛÛ™\Ú[Ûˆ\È[œÝ[Y[^\ÝÂˆËÈÈ[™‚ˆÛœš[Š]Ú^™[Ùˆ]‰\ËÉ\Ù‰L›WÜÛ˜\ÉLÉ]^	]I\ËœH‹Û˜\\‹ˆ˜]ÒY›ÝÈÈ™˜]ÚYÈˆˆˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O™œ˜[YK\Ý	ˆQ‘‘‘‘‘‘‹Û˜\š[XYÙKÚYˆÛ˜\š[XYÙKšZYÚÛ˜\™œ›ÛQ\È—Ù\ˆˆˆŠNÂˆ’SJˆˆH›Ü[Š]ØˆŠNÂˆYˆ
YŠBˆÂˆÝ]XÈ›ÛÛÛÛ\Z[™YH˜[ÙNÂˆYˆ
XÛÛ\Z[™Y
BˆÂˆÛÛ\Z[™YHYNÂˆœš[ŠÝ\œ‹–Ýš×HÖ—Õ’×ÔÓTÑSTØ[››ÝÜš]H	\È8 %“ÈÛ˜\ÚÝÈ‚ˆÚ[™H[\Y\È[—ˆ‹]
NÂˆBˆBˆYˆ
ŠBˆÂˆÜ›ÝSÛ™HHYNÂˆœš[Š‹”—‰]H	]WŒMWˆ‹Û˜\š[XYÙKÚYÛ˜\š[XYÙKšZYÚ
NÂˆYˆ
Û˜\™œ›ÛQ\
BˆÂˆËÈH\\ÜXÝÛÛY\È˜XÚÈÛ™HÌ‹Xš]ÛÜ™\ˆ^[ˆ›Ü‚ˆËÈÕS“Ô“WÔÎÕRS•H\\ÈHÝÈš]È
H‹Œ—ŒLBˆËÈ[YÙ\ŠNÈ›ÜˆÌ—ÔÑ“ÐUÔÎÕRS•
HSQ]
H]\ÈHÌ‹Xš]ˆËÈ›Ø][ˆ‹ŒKˆ™XY›Ý\ÈH›Ü›X[\ÙY‹ŒHÝX›HÛÈH™\ÝˆËÈÙˆH[\\È›Ü›X]XYÛ›ÜÝXËˆH\œÜXÝ]™H\Y™™\‰ÜÂˆËÈ˜[Y\È[Ú]Ú][ˆHZ\ˆÙˆKŒÛÈH[™X\ˆÜ™^HÛÝ[™HBˆËÈÚ]H™XÝ[™ÛHÚ]]™\ˆ]ÛÛZ[™Y8 %H[XYÙH\È\™Y›Ü™BˆËÈÝ™]ÚY™]ÙY[ˆHÝ\™˜XÙIÜÈÕÓˆZ[ˆ[™X^[™Hš[[˜[YBˆËÈØ^\ÈÙ\ÛÈ›Ø›ÙH™XYÈ]\ÈHÛÛÝ\ˆÝ\™˜XÙK‚ˆÛÛœÝ›ÛÛ\Ñ›Ø]H‹O™\™›Ü›X]OH’×Ñ“Ô“PUÑÌ—ÔÑ“ÐUÔÎÕRS•Âˆ]]È™XY›Ü›HHÉ—JÚ^™WÝJHOˆÝX›HÂˆYˆ
\Ñ›Ø]
BˆÂˆ›Ø]ŽÂˆY[XÜJ	™‹‹Oœ™XY˜XÚË›X\Y
ÈK
NÂˆ™]\›ˆÝX›JŠNÂˆBˆZ[Ì—ÝŽÂˆY[XÜJ	‹‹Oœ™XY˜XÚË›X\Y
ÈK
NÂˆ™]\›ˆÝX›Jˆ	ˆ‘‘‘‘‘JHÈMÍÍÌŒMKŒÂˆNÂˆÝX›HÈHYLÌHHLYLÌÂˆ›Üˆ
Ú^™WÝHHÈHŽÈH
ÏH
BˆÂˆÛÛœÝÝX›HH™XY›Ü›JJNÂˆÈHÝŽ›Z[ŠË
NÂˆHHÝŽ›X^
K
NÂˆBˆÛÛœÝÝX›HÜ[ˆHHˆÈÈ
HHÊHˆKŒÂˆ›Üˆ
Ú^™WÝHHÈHŽÈH
ÏH
BˆÂˆÛÛœÝZ[ÝÈHZ[Ý
MKŒ
ˆ
™XY›Ü›JJHHÊHÈÜ[ŠNÂˆÛÛœÝZ[Ý™Ø–Ì×HHÈËËÈNÂˆÜš]J™Ø‹KËŠNÂˆBˆœš[ŠÝ\œ‹–Ýš×H	L\ÈHTÛ˜\ÚÝ
	\ÊK˜[™ÙH‚ˆ‰K™‹‹‰K™—ˆ‹ˆ\Ý	ˆQ‘‘‘‘‘‘‹\Ñ›Ø]È‘Ì‘ˆˆˆ‘‹ËJNÂˆBˆ[ÙBˆÂˆ›Üˆ
Ú^™WÝHHÈHŽÈH
ÏH
BˆÜš]J‹Oœ™XY˜XÚË›X\Y
ÈKKËŠNÂˆBˆ˜ÛÜÙJŠNÂˆBˆBˆœš[ŠÝ\œ‹–Ýš×H[\Y	^H™\ÛÛ™HÛ˜\ÚÝÈÈ	\É\×ˆ‹ˆ‹OœÛ˜\ÚÝËœÚ^™J
KÛ˜\\‹ˆÜ›ÝSÛ™HÈˆˆˆˆ8 %“Ó‘HÑˆSHÑT‘HÔ’USˆŠNÂˆB‚ˆÝ]XÈÛÛœÝZ[ÝÝ]Ñ]™\žHBˆ[ŠÖ—Õ’×ÔÕUÈŠHÈÝŽ›X^
SÝÛ
[ŠÖ—Õ’×ÔÕUÈŠK[‹L
JHˆÂˆYˆ
Ý]Ñ]™\žH	‰ˆ
‹O™œ˜[YH	HÝ]Ñ]™\žJHOH
BˆšÔ™[™\™\—Ñ[\Ý]Ê
NÂ‚ˆËÈÖ—Õ’×Ô“Ñ’SOSˆ8 %Hœ˜[YIÜÈÔH[YHžH\ÙK]™\žHˆÙXÛÛ™Ë‚ˆËÂˆËÈÛˆHÓÐÒÈ˜]\ˆ[ˆHœ˜[YHÛÝ[[™H™X\ÛÛˆ\È\È›Ú™XÝ	ÜÈÝÛ‚ˆËÈ\ÝÜžNˆH™\Ü]™\žHˆœ˜[Y\ÈØ[\\ÈHY™™\™[[[Ý[ÙˆØ[[YH[ˆ]™\žBˆËÈ\˜KÛÈH›ÛÝ	ÜÈ˜\Ýœ˜[Y\È[™Ø[Y\^IÜÈÛÝÈÛ™\ÈÛÝ[™H]™\˜YÙYžBˆËÈÚ]]™\ˆH[\˜[\[™YÈ^H
ÛÝÚHNŠKˆ™\Ü[™È\ˆÙXÛÛ™XZÙ\ÂˆËÈHœÈÛÛ[[ˆYX[ˆHØ[YH[™È]™\ž]Ú\™KÚXÚ\ÈH[\™HÚ[Ù‚ˆËÈ]š[™È]‚ˆYˆ
×Ü›Ùš[SÛŠBˆÂˆÝ]XÈÛÛœÝ]]ÈHÝŽ˜Ú›Û›ÎŽœÝXYWØÛØÚÎŽ››ÝÊ
NÂˆÝ]XÈ]]È\ÝHÂˆÝ]XÈZ[Ý\Ýœ˜[YHHÂˆÝ]XÈÝX›H\š[ÙBˆ[ŠÖ—Õ’×Ô“Ñ’SHŠHÈÝŽ›X^
KŒ]ÙŠ[ŠÖ—Õ’×Ô“Ñ’SHŠJJHˆKŒÂˆÛÛœÝ]]È›ÝÈHÝŽ˜Ú›Û›ÎŽœÝXYWØÛØÚÎŽ››ÝÊ
NÂˆÛÛœÝÝX›HHÝŽ˜Ú›Û›ÎŽ™\˜][ÛÝX›OŠ›ÝÈH\Ý
K˜ÛÝ[

NÂˆYˆ
H\š[Ù
BˆÂˆÛÛœÝZ[Ýœ˜[Y\ÈH‹O™œ˜[YHH\Ýœ˜[YNÂˆÛÛœÝÝX›H\ÈH
ˆLŒÂˆËÈ]™\žH\ÙH\ÈH\˜Ù[YÙHÙˆHÐS[YHÙˆHÚ[™ÝËÛÈBˆËÈÛÛ[[œÈ\™HÛÛ\\˜X›H[™Z\ˆÚÜ˜[œ›ÛHL	H\ÈHœ˜[YH[YBˆËÈ\È[œÝ[Y[Ù\È›ÝY]XØÛÝ[›Üˆ8 %ÚXÚ\ÈH[X™\ˆÛÜˆËÈÙYZ[™È˜]\ˆ[ˆY[™Ë‚ˆ]]ÈÝHÉ—JZ[ÝœÊHÈ™]\›ˆLŒ
ˆ
ÝX›JœÊH
ˆYKMŠHÈ\ÎÈNÂˆÛÛœÝÝX›H\‘œ˜[YHHœ˜[Y\ÈÈ\ÈÈÝX›Jœ˜[Y\ÊHˆŒÂˆËÈ]™\žH\ÙH™[ÝÈ\ÈVÓTÒU‘HÙˆ]™\žHÝ\ˆ
ÙYH›Ù”ØÛÜJKÛÈBˆËÈÝ[È\™HÝ[\È˜]\ˆ[ˆÝX˜XÝ[ÛœËˆ]\ÈHÚÛHY™™\™[˜ÙN‚ˆËÈHÝX˜XÝ[ÛˆY\È[ˆ\œ›Üˆ[ˆÛ™H\›H[œÚYH[›Ý\ˆ\›IÜÈ™\ÚYX[ˆËÈÚ\™HHÝ[HX]™\È]š\ÚX›H[ˆÝ]ÚYX‚ˆËÂˆËÈÝ\˜\ÈÑ˜]ÉÜÈÝÛˆ[[YYÛÜšÈ8 %H™YÚ\Ý\ˆXÛÙKBˆËÈ\[[™KZÙ^HZ[[™]ÈÛÚÝ\H™]ÚXÛÛœÝ[Ø[Ë[™BˆËÈ[Ø^\Ë[ÛˆÙ[œÝ\Ù\ËˆÝ]ÚYX\È]™\ž][™È]\È›ÝH™[™\™\ˆ]ˆËÈ[ˆHÝY\Ý	ÜÈÚ[][][Û‹HÛÛ[X[™›ØÙ\ÜÛÜ‹[™[žHØZ]™]ÙY[‚ˆËÈœ˜[Y\Ë‚ˆËÈ™XÛÜ™\È›ÝÈH‘TÒQPS[™H™YHÝX‹\\Ù\ÈÚ]™\ÚYH]ÛÂˆËÈH˜]ÈÝ[\ÈÈ[˜ÛYH[HÜˆHÝ]ÚYXÛÛ[[ˆXœÛÜ˜œÈBˆËÈY™™\™[˜ÙH[™™XYÈ\ÈH™YÜ™\ÜÚ[Ûˆ›Ø›ÙHXYKˆ^XÝHH\š]Y]XÂˆËÈ]\ŒÛÝÜ›Û™È[ˆHÝ\ˆ\™XÝ[Ûˆ
ÙYH›Ù”ØÛÜJK‚ˆÛÛœÝZ[Ý™XÛÜ™Ý[H×Ü›Ù‹œ™XÛÜ™
È×Ü›Ù‹œ™XÛÜ™Ý]H
Âˆ×Ü›Ù‹œ™XÛÜ™™\^
È×Ü›Ù‹œ™XÛÜ™[™^
Âˆ×Ü›Ù‹œÝ™X[QÝX\™ÂˆËÈÝ\˜\È›ÝÈH™\ÚYX[[ˆ^XÝHHØ[YHØ^K›ÜˆHØ[YH™X\ÛÛ‹‚ˆÛÛœÝZ[ÝÝ\•Ý[H×Ü›Ù‹™˜]ÓÝ\ˆ
È×Ü›Ù‹›Ý\’Ù^H
Âˆ×Ü›Ù‹›Ý\”\[[™H
È×Ü›Ù‹›Ý\‘™]Ú
Âˆ×Ü›Ù‹›Ý\”ÚY\ˆ
È×Ü›Ù‹›Ý\™YÚ[ˆ
Âˆ×Ü›Ù‹›Ý\•Z[ÂˆËÈÛÛœÝ[Ø\ÈH™\ÚYX[›ÝÈÛÈ
\ÍIÜÈÜ]
K‚ˆÛÛœÝZ[ÝÛÛœÝÝ[H×Ü›Ù‹˜ÛÛœÝ[È
È×Ü›Ù‹˜ÛÛœÝœÈ
Âˆ×Ü›Ù‹˜ÛÛœÝœÐÛÜH
È×Ü›Ù‹˜ÛÛœÝœÔ]Ú
Âˆ×Ü›Ù‹˜ÛÛœÝÈ
È×Ü›Ù‹˜ÛÛœÝÚ\™YÂˆÛÛœÝZ[Ý˜]ÕÝ[HÛÛœÝÝ[
È×Ü›Ù‹œÝ™X[\È
Âˆ×Ü›Ù‹^\™\È
È™XÛÜ™Ý[
ÈÝ\•Ý[ÂˆÛÛœÝZ[ÝÝX›Z]Ý[Bˆ×Ü›Ù‹œÝX›Z]
È×Ü›Ù‹œÝX›Z]Ø[
È×Ü›Ù‹™™[˜ÙUØZ]ÂˆÛÛœÝZ[ÝÛ›ÝÛˆH˜]ÕÝ[
ÈÝX›Z]Ý[
È×Ü›Ù‹œ™XY˜XÚÈ
Âˆ×Ü›Ù‹™œ˜[YTÝ]È
È×Ü›Ù‹œÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—H	KŒYˆœÈ
	KŒYˆ\ËÙœ˜[YK	[H˜]ÜËÙœ˜[YJH˜]È	KŒY‰IH‚ˆ–ØÛÛœÝ[È	KŒYˆ
œÈ	KŒYˆØÛÜH	KŒYˆ]Ú	KŒY—HÈ	KŒYˆ‚ˆœÚ\™Y	KŒYŠHÝ™X[\È	KŒYˆ^\™\È	KŒYˆ™XÛÜ™	KŒYˆÝ\ˆ‚ˆ‰KŒY—HÝX›Z]	KŒY‰IHØØ[	KŒYˆÜH	KŒY—H™XY˜XÚÈ	KŒY‰IH‚ˆœ	KŒY‰IHÝ]ÚYH	KŒY‰IWˆ‹ˆœ˜[Y\ÈÈ\‘œ˜[YKˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈ×Ü›Ù‹™˜]ÜÈÈœ˜[Y\Èˆ
KˆÝ
˜]ÕÝ[
KÝ
ÛÛœÝÝ[
KÝ
×Ü›Ù‹˜ÛÛœÝœÊKˆÝ
×Ü›Ù‹˜ÛÛœÝœÐÛÜJKÝ
×Ü›Ù‹˜ÛÛœÝœÔ]Ú
KˆÝ
×Ü›Ù‹˜ÛÛœÝÊKÝ
×Ü›Ù‹˜ÛÛœÝÚ\™Y
KÝ
×Ü›Ù‹œÝ™X[\ÊKˆÝ
×Ü›Ù‹^\™\ÊKÝ
™XÛÜ™Ý[
KÝ
Ý\•Ý[
KˆÝ
ÝX›Z]Ý[
KÝ
×Ü›Ù‹œÝX›Z]Ø[
KÝ
×Ü›Ù‹™™[˜ÙUØZ]
KˆÝ
×Ü›Ù‹œ™XY˜XÚÊKÝ
×Ü›Ù‹œ
KLŒHÝ
Û›ÝÛŠJNÂ‚ˆÚYˆÖ—ÕÒÓQ•SÂˆËÈKŒˆ8 %ÒUÔÑHTÑTÈÈ“ÕÐVKš[Y[[YYX][H[™\ˆHX›BˆËÈÛÈHØ\\Èš\ÚX›H[ˆHÙÈ˜]\ˆ[ˆÛ›H[ˆH\™˜Ø\\™BˆËÈÛÛYX›ÙH\ÈÈ[šÈÈZÙKˆSÓTÒU‘HÙˆØ[Y\È[™ØØ[YžHBˆËÈØ[\[™È\š[ÙÈH\Ý[X]Üˆ\È
[YYœÊH
Ø[ÈÈØ[\Y
KÚXÚ\ÂˆËÈH\š[Ù^XÝHÚ[ˆHØ[ÛÝ[\ÈH][\HÙˆ][™Ú][ˆÛ™BˆËÈØ[Ùˆ]Ý\Ú\ÙK‚ˆYˆ
×ÝÚÛQ[˜ÊBˆÂˆÝ]XÈÚÛQ[˜ÈÖÌ×NÂˆÚÛQ[˜ÊˆÝ\–Ì×HHÈ	™×ÝÙ”Ý™X[K	™×ÝÙ•^\™K	™×ÝÙ‘˜]ÈNÂˆÛÛœÝÚ\Šˆ›VÌ×HHÈ•\ØYÝ™X[H‹•\ØY^\™H‹‘Ñ˜]ÈˆNÂˆÝX›H\ÝÌ×HHßKØ[ÖÌ×HHßNÂˆZ[ÝØ[\YHÂˆ›Üˆ
[HHÈHÎÈ
ÊÚJBˆÂˆÛÛœÝZ[ÝˆHÝ\–ÚWKO›œÈHÖÚWK›œÎÂˆÛÛœÝZ[ÝÈHÝ\–ÚWKOœØ[\YHÖÚWKœØ[\YÂˆÛÛœÝZ[ÝÈHÝ\–ÚWKO˜Ø[ÈHÖÚWK˜Ø[ÎÂˆÖÚWHH
˜Ý\–ÚWNÂˆØ[\Y
ÏHÎÂˆ\ÝÚWHHÈÈÝX›JŠH
ˆ
ÝX›JÊHÈÝX›JÊJHˆŒÂˆØ[ÖÚWHHœ˜[Y\ÈÈÝX›JÊHÈÝX›Jœ˜[Y\ÊHˆŒÂˆBˆËÈHš[™^ÈH[X™\œÈ˜]\ˆ[ˆ[ˆH›ÛÝ›ÝNˆÛÈÛØÚÂˆËÈ™XYÈ\ˆØ[\YØ[]HØ[YHØ[Xœ˜]YÛÜÝHØÛÜ\È^K‚ˆÛÛœÝÝX›Hš[\ÈBˆÝX›JØ[\Y
H
ˆ‹Œ
ˆ
ÝX›J×Ü›Ù“›ÝÓœÌL
HÈLŒ
H
ˆYKMŽÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÒÓKQ•SÕSÓˆ
HØ[[ˆ	[KSÓTÒU‘HÙˆØ[Y\È‚ˆ¸ %ÛÛ\\™HÚ]H\™˜Þ[X›ÛÔ“ÕT™]™\ˆÛ™HÞ[X›Û	ÜÈÙ[ˆ‚ˆ[YJNˆ	\È	KŒ™ˆ\ËÙœ˜[YH
	KŒY‰IK	KŒˆØ[ËÙœ˜[YJH	\È	KŒ™ˆ‚ˆŠ	KŒY‰IK	KŒŠH	\È	KŒ™ˆ
	KŒY‰IK	KŒŠH8 %HX›HX›Ý™HØ^\È‚ˆœÝ™X[\È	KŒY‰IH^\™\È	KŒY‰IH˜]È	KŒY‰INÈ\È[œÝ[Y[	ÜÈ‚ˆ›ÝÛˆš[	KŒ™ˆ\ÈÝ™\ˆHÚ[™Ý×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊZÕÙ”\š[Ùˆ›VÌKœ˜[Y\ÈÈ\ÝÌH
ˆYKMˆÈÝX›Jœ˜[Y\ÊHˆŒˆÝ
Z[Ý
\ÝÌJJKØ[ÖÌKˆ›VÌWKœ˜[Y\ÈÈ\ÝÌWH
ˆYKMˆÈÝX›Jœ˜[Y\ÊHˆŒˆÝ
Z[Ý
\ÝÌWJJKØ[ÖÌWKˆ›VÌ—Kœ˜[Y\ÈÈ\ÝÌ—H
ˆYKMˆÈÝX›Jœ˜[Y\ÊHˆŒˆÝ
Z[Ý
\ÝÌ—JJKØ[ÖÌ—KˆÝ
×Ü›Ù‹œÝ™X[\ÊKÝ
×Ü›Ù‹^\™\ÊKÝ
˜]ÕÝ[
Kˆš[\ÊNÂˆB‚ˆÙ[™YˆËÈÖ—ÕÒÓQ•SÂˆËÈHS”Õ•SQS•	ÔÈÕÓˆ’SÛˆ]ÈÝÛˆ[™HÛÈ]Ø[ˆ™]™\ˆ™H™XY\ÂˆËÈ\ÙˆHØ[YIÜÈœ˜[YKˆ]\ÈÚ\™ÙYÈH[ˆ]\ÚÙY›Üˆ][™ˆËÈÈ›Ý[™È[ÙK[™]\È‘T“È[ˆH[ˆÚ]Ý]Ö—Õ’×Ñ”SQWÔÕUÈ8 %]ˆËÈ]\È^XÝHH[ˆ[ˆÚXÚ›Èœ˜[YH[YH\È™XÛÜ™YÚXÚ\ÈÚBˆËÈ]YÛÛ™HÌÈ\È[›YX\Ý\™Yˆ][ÝH]Ú[™]™\ˆ][Ý[™ÈHœ˜[YH[YBˆËÈYX\Ý\™YÚ]œ˜[YHÝ]ÈÛ‹K™Kˆ[Ø^\È
ÛÝÚHÌÍIÜÈÚ\KÛ™BˆËÈ[œÝ[Y[\\ˆÝ]
K‚ˆYˆ
×Ü›Ù‹™œ˜[YTÝ]ÊBˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÖ—Õ’×Ñ”SQWÔÕUÈ]Ù[Žˆ	KŒ™ˆ\ËÙœ˜[YH
	KŒY‰IHÙˆ‚ˆ\ÈÚ[™ÝÊH8 %HYX\Ý\™Yœ˜[YH\È\È]XÚÓÕÑTˆ[ˆH‚ˆ›Û™HH^Y\ˆ[œ×ˆ‹ˆœ˜[Y\ÈÈÝX›J×Ü›Ù‹™œ˜[YTÝ]ÊH
ˆYKMˆÈÝX›Jœ˜[Y\ÊHˆŒˆÝ
×Ü›Ù‹™œ˜[YTÝ]ÊJNÂ‚ˆËÈHÕÐTÒRSˆT“IÔÈÕÓˆÓÕS•Tˆ
Ö—Õ’×ÔÕÐTÒRSLJKˆ[ˆ\›HÚ]›ÂˆËÈÛÝ[\ˆØ[››Ý™HÚÝÛˆÈ]™H[™ØYÙY
ÛÝÚHMLJK[™\ÈÛ™H\ÈBˆËÈÜXÚYšXÈØ^HÙˆ[‹Y[™ØYÚ[™È]ÛÝ[Ý\Ú\ÙHÛÚÈZÙHHœ˜[YK\˜]BˆËÈ™YÜ™\ÜÚ[ÛˆÚ]›ÈØ]\ÙNˆHÝØ\ÚZ[ˆ]Ø[››ÝXÜ]Z\™H“ÔÈHœ˜[YKˆËÈÛÈ™\Ù[Ø™[ÝÈHÚ[™ÝÉÜÈœ˜[YHÛÝ[\ÈHÚÛH^[˜][Ûˆ›Ü‚ˆËÈHXÝ\™H]Ý]\œÈÚ[H]™\žHÝ\ˆÛÛ[[ˆ™XYÈ›Ü›X[‚ˆYˆ
‹OØ[ÝØ\ÚZ[ŠBˆÂˆÝ]XÈZ[Ý\Ý™\Ù[ÈH\Ý˜Z[ÈHÂˆÛÛœÝZ[ÝH‹OœÝØ\œ™\Ù[ÈH\Ý™\Ù[ÎÂˆÛÛœÝZ[ÝˆH‹OœÝØ\˜XÜ]Z\™Q˜Z[ÈH\Ý˜Z[ÎÂˆ\Ý™\Ù[ÈH‹OœÝØ\œ™\Ù[ÎÂˆ\Ý˜Z[ÈH‹OœÝØ\˜XÜ]Z\™Q˜Z[ÎÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝØ\ÚZ[ˆ	]^	]H	\Îˆ	[H™\Ù[Y\ÈÚ[™ÝÈÙˆ	[H‚ˆ™œ˜[Y\Ë	[H“ÔQ
XÜ]Z\™KÜ™\Ù[˜Z[Y
K	[H™XZ[Ë‚ˆ‰[HÝX›Ü[X[ˆ‹ˆ‹OœÝØ\ÚY‹OœÝØ\šZYÚˆ‹OœÝØ\›[ÙHOH’×Ô‘TÑS•ÓSÑWÓPRS“ÖÒÒˆÈ“PRS“Öˆˆ‘’Q“È‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY
[œÚYÛ™YÛ™ÈÛ™ÊYœ˜[Y\Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊY‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹OœÝØ\œ™XZ[Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹OœÝØ\œÝX›Ü[X[
NÂˆB‚ˆËÈHÔUÑˆ™XÛÜ™ÚXÚ\ÈH\™Ù\Ý˜]Ë\]\›HÛˆBˆËÈÜ\˜]Ü‰ÜÈœ˜[YH
MKŒˆ\Ë‹ŒMÈ\ÈH˜]ÊH[™Y›Èœ™XZÙÝÛˆ][‚ˆËÈœË\\‹Y˜]È\ÈÙ[\ÈHÚ\™K™XØ]\ÙHHÚ\™H[Ý™\ÈÚ[ˆ[žHÝ\‚ˆËÈ\ÙHÙ\È
ÛÝÚHÌŒ
H[™H\‹Y˜]ÈÛÜÝ\ÈÚ]HÚ[™ÙHÈ\ÂˆËÈÛÙH]XÝX[H[Ý™\Ë‚ˆÂˆÛÛœÝÝX›HHœ˜[Y\ÈÈÝX›J×Ü›Ù‹™˜]ÜÊHˆŒÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—H™XÛÜ™	KŒY‰IHHÝ]H	KŒYˆ
È™\^	KŒYˆ
È[™^	KŒYˆ‚ˆŠÈÕPT‘	KŒYˆ
È™\ÚYX[	KŒYˆ\ˆ˜]Îˆ	KŒˆœÈH	KŒˆ
È‚ˆ‰KŒˆ
È	KŒˆ
È	KŒˆ
È	KŒ—ˆ‹ˆÝ
™XÛÜ™Ý[
KÝ
×Ü›Ù‹œ™XÛÜ™Ý]JKˆÝ
×Ü›Ù‹œ™XÛÜ™™\^
KÝ
×Ü›Ù‹œ™XÛÜ™[™^
KˆÝ
×Ü›Ù‹œÝ™X[QÝX\™
KÝ
×Ü›Ù‹œ™XÛÜ™
KˆÈÝX›J™XÛÜ™Ý[
HÈˆŒˆÈÝX›J×Ü›Ù‹œ™XÛÜ™Ý]JHÈˆŒˆÈÝX›J×Ü›Ù‹œ™XÛÜ™™\^
HÈˆŒˆÈÝX›J×Ü›Ù‹œ™XÛÜ™[™^
HÈˆŒˆÈÝX›J×Ü›Ù‹œÝ™X[QÝX\™
HÈˆŒˆÈÝX›J×Ü›Ù‹œ™XÛÜ™
HÈˆŒ
NÂˆËÈH‘TÓÓ‘HÔU
\HÝ\JKš[Y™\ÚYHH™XÛÜ™Ü]ˆËÈ]XÛÛ\ÜÙ\ËˆœËÙ˜]È\™HYX[œÈœ\ˆÐSTQ˜]È‹ÚXÚ\ÈBˆËÈØ[YHÜ[][ÛˆØØ[YKÌM‹ÛÈHÛÈ[™\ÈÛÛ\\™H\™XÝKˆBˆËÈÙ[œÝ\ÉÜÈÝÛˆÛØÚÈš[\Èš[YÛÈH™XY\ˆÝX˜XÝÈ]8 %ˆËÈ›Û™H›ÝÓœÊ
HØ[[™È[œÚYHXXÚYX\Ý\™YZ\ˆ
ÙYH›Ù”ØÛÜIÜÂˆËÈš[›ÝH›ÜˆH\š]Y]XÊK‚ˆYˆ
×Ü™\ÛÛ™TÜ]Ù[œÝ\ÊBˆÂˆÝ]XÈZ[Ý›ˆH[ˆH˜ÈHXÈHÙHÂˆÛÛœÝZ[Ý›ˆH×Ü™\ÛÛ™U“œÈH›‹[ˆH×Ü™\ÛÛ™RSœÈH[ŽÂˆÛÛœÝZ[Ý˜ÈH×Ü™\ÛÛ™UØ[ÈH˜ËˆXÈH×Ü™\ÛÛ™RPØ[ÈHXÎÂˆÛÛœÝZ[ÝÙH×Ü™\ÛÛ™TØ[\Y˜]ÜÈHÙÂˆ›ˆH×Ü™\ÛÛ™U“œÎÈ[ˆH×Ü™\ÛÛ™RSœÎÂˆ˜ÈH×Ü™\ÛÛ™UØ[ÎÈXÈH×Ü™\ÛÛ™RPØ[ÎÂˆÙH×Ü™\ÛÛ™TØ[\Y˜]ÜÎÂˆYˆ
Ù
Bˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—H™\ÛÛ™HÜ]
H˜]È[ˆM‹	[HØ[\Y
Nˆ‚ˆ™\^\ØYÝ™X[H	KŒˆœËÙ˜]È
	KŒ™ˆØ[ËÙ˜]ÊH
È‚ˆš[™^\ØYÝ™X[H	KŒˆœËÙ˜]È
	KŒ™ˆØ[ËÙ˜]ÊHH‚ˆœ™\ÛÛ™H	KŒˆÙˆH™XÛÜ™œÈX›Ý™NÈÛØÚÈš[[œÚYH‚ˆÜÙH‰KŒˆœËÙ˜]È8 %ÝX˜XÝ]ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊYÙÝX›J›ŠHÈÝX›JÙ
KˆÝX›J˜ÊHÈÝX›JÙ
KÝX›J[ŠHÈÝX›JÙ
KˆÝX›JXÊHÈÝX›JÙ
KˆÝX›J›ˆ
È[ŠHÈÝX›JÙ
KˆÝX›J˜È
ÈXÊHÈÝX›JÙ
H
‚ˆ
ÝX›J×Ü›Ù“›ÝÓœÌL
HÈLŒ
JNÂˆBˆËÈTSS‘PÓÔ‘	ÜÈ[™ØYÙ[Y[Ú[™ÝÙY
ÛÝÚHMLNˆH\›H[™]ÂˆËÈÛÛ›Û]\Ý›Ý™H›Ý˜X›HÚ]^HÛZ[NÈHÛÛ›Ûš[ÂˆËÈ›Ý[™È™XØ]\ÙHHÛÝ[\œÈ™]™\ˆ[Ý™JK‚ˆYˆ
‹Oœ\”™XÊBˆÂˆÝ]XÈZ[ÝÈHHÈHHÈHÂˆÝ]XÈZ[Ý™XÖÚÔ“X^™XÛÜ™\œ×HHßNÂˆZ[ÝÚ[šÜÓ›ÝÈHÂˆÚ\ˆ\–ÌLŽNÂˆÚ^™WÝˆHÂˆ›Üˆ
Z[Ì—Ý’YHÈ’YÔ“X^™XÛÜ™\œÎÈ
ÊÜ’Y
BˆÂˆÛÛœÝZ[ÝŒˆH×ÜÚ[šÜÔ™XÛÜ™YÜ’YHH™XÖÜ’YNÂˆ™XÖÜ’YHH×ÜÚ[šÜÔ™XÛÜ™YÜ’YNÂˆÚ[šÜÓ›ÝÈ
ÏHŒŽÂˆYˆ
Œˆ	‰ˆˆÚ^™[Ùˆ\ˆHMŠBˆˆ
ÏHÚ^™WÝ
Ûœš[Š\ˆ
È‹Ú^™[Ùˆ\ˆH‹ˆ	\É]N‰[H‹ˆ’YOHÔ”[\™XÛÜ™\ˆÈœ[\ˆˆÈ‹ˆ’Y
[œÚYÛ™YÛ™ÈÛ™ÊYŒŠJNÂˆBˆÛÛœÝZ[ÝÌˆH×ÜØ\\™YHËˆH×Ü•Z[˜]ÜÈHÂˆÛÛœÝZ[ÝÌˆH×Ü•ØZ]œÈHËˆH×Ü”[\[YHÂˆÛÛœÝZ[ÝÌˆH×Ü“Ý™\™›ÝÒ[›[™HHÎÂˆÈH×ÜØ\\™YÈH×Ü•Z[˜]ÜÎÈÈH×Ü•ØZ]œÎÂˆH×Ü”[\[YÈÈH×Ü“Ý™\™›ÝÒ[›[™NÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—H\ˆ™XÛÜ™ˆ	KŒYˆÚ[šÜËÙœ˜[YH
	\È
KZ[	KŒˆ‚ˆ™˜]ÜËÙœ˜[YK	KŒˆØ\\™YÙœ˜[YKÝX›Z]ØZ]	KŒˆ‚ˆ\ËÙœ˜[YH
[\[Y	[JKÝ™\™›ÝËZ[›[™H	[I\×ˆ‹ˆœ˜[Y\ÈÈÝX›JÚ[šÜÓ›ÝÊHÈÝX›Jœ˜[Y\ÊHˆŒ\‹ˆœ˜[Y\ÈÈÝX›JŠHÈÝX›Jœ˜[Y\ÊHˆŒˆœ˜[Y\ÈÈÝX›JÌŠHÈÝX›Jœ˜[Y\ÊHˆŒˆœ˜[Y\ÈÈÝX›JÌŠHÈYLÈÈÝX›Jœ˜[Y\ÊHˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊY‹
[œÚYÛ™YÛ™ÈÛ™ÊYÌ‹ˆ×Üš[™Ý™\™›ÝÈÈˆ
ŠŠˆ’S‘Õ‘T‘“ÕÈ8 %Ø\\™\È‚ˆ[˜Ø]YXÝ\™HÝ\ÜXÝ
ŠŠˆ‚ˆˆˆŠNÂˆBˆËÈÒHÕPT‘TÈS”ÒQH™XÛÜ™S‘ÒUUQPS”È“ÔˆHS‹ˆ]\ÂˆËÈHÝ™X[HÛÛ[\Ú[™]Ø\ÈSÐVTÈ[ˆ\ÈÛÛ[[ˆ8 %]\ÝˆËÈY›È˜[YK™XØ]\ÙH›Ù”ØÛÜJÝ™X[\ÊXÜ˜\ÈÛ›HHÛÜHÛÈBˆËÈÜ›ÜÜËYœ˜[YH]ÛÜÝÈÝ™X[\Ø›Ý[™Ëˆ][HK\ÈšXÙYÙ™‚ˆËÈ™XÛÜ™[™][HKŒHÙ™ˆÝX\™›ÛÈ[[\È[™H^\ÝYHØ[YBˆËÈZ[\ÙXÛÛ™ÈÙ\™H[ˆ›ÝšXÙ\ËˆÚ]\ÈYÙˆ™XÛÜ™Y\ˆ\ÂˆËÈÝX˜XÝ[Ûˆ\ÈHXÝX[šÐÛY
˜™XÛÜ™[™Ë[™U\È][HK	ÜÂˆËÈ™X[ÙZ[[™Ë‚ˆËÈ‹‹˜[™HÔUÑˆÝ\˜
\Y\ˆÊK[ˆHØ[YH›Ü›H[™›Ü‚ˆËÈHØ[YH™X\ÛÛŽˆŒNH\ÈÙˆHÜ\˜]Ü‰ÜÈœ˜[YHÚ]›Ý[™ÈØZYˆËÈX›Ý]Ú]\È[ˆ]ˆ][ÝY\ˆ˜]È\ÈÙ[™XØ]\ÙH]\ÈBˆËÈ[X™\ˆHÚ[™ÙHÈ\ÈÛÙH][Ý™\È[™]\ÈÛÛ\\˜X›H™]ÙY[‚ˆËÈZ\ˆœ˜[YH[™Ý\œÈÚ\™HHÚ\™H\È›Ý‚ˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝ\ˆ	KŒˆœËÙ˜]ÈHÚY\ˆ	KŒˆ
ÈÙ^H	KŒˆ
È\[[™H‚ˆ‰KŒˆ
È™YÚ[ˆ	KŒˆ
È™]Ú	KŒˆ
ÈZ[	KŒˆ
È™\ÚYX[	KŒˆ‚ˆŠ	KŒY‰IHÙˆœ˜[YJWˆ‹ˆÈÝX›JÝ\•Ý[
HÈˆŒˆÈÝX›J×Ü›Ù‹›Ý\”ÚY\ŠHÈˆŒˆÈÝX›J×Ü›Ù‹›Ý\’Ù^JHÈˆŒˆÈÝX›J×Ü›Ù‹›Ý\”\[[™JHÈˆŒˆÈÝX›J×Ü›Ù‹›Ý\™YÚ[ŠHÈˆŒˆÈÝX›J×Ü›Ù‹›Ý\‘™]Ú
HÈˆŒˆÈÝX›J×Ü›Ù‹›Ý\•Z[
HÈˆŒˆÈÝX›J×Ü›Ù‹™˜]ÓÝ\ŠHÈˆŒˆÝ
Ý\•Ý[
JNÂ‚ˆËÈÒUH“Ñ’STˆUÑSˆUSˆU‘TÒQPSˆÙYHH›Ù”ØÛÜBˆËÈÛÛ[Y[›ÜˆHYXÚ[š\ÛNˆH™\ÝYØÛÜIÜÈÛÈÛØÚÈ™XYÈ˜[ˆËÈÕUÒQH]ÈÝÛˆYX\Ý\™Y[\˜[[™[œÚYH]È\™[	ÜË[™\™H›ÝˆËÈÝX˜XÝY8 %[™˜]ÓÝ\˜\ÈHÝ]\›[ÜÝ\‹Y˜]ÈØÛÜKÛÈ[Ù‚ˆËÈ[H[™[ˆH[X™\ˆš[Y\È™\ÚYX[\ÝX›Ý™K‚ˆËÂˆËÈH\š]Y]XË\ˆ™\ÝYØÛÜKÛÜšÙY›ÝYÚÛ˜ÙNˆHÛÛœÝXÝÜ‰ÜÂˆËÈ™XY\ÈÜ[™]ÙY[ˆH\™[	ÜÈ[™HÚ[	ÜËÛÈ]\È[ˆBˆËÈ\™[	ÜÈ[\˜[[™\È“Õ[ˆÚ[œØ8 %]\ÈÛ™HÚÛH™XY[ÂˆËÈH\™[	ÜÈ™\ÚYX[ˆHÛÜÙJ
X™XY\È[œÚYHHÚ[	ÜÈÝÛ‚ˆËÈYX\Ý\™YÝ[ÛÈ][™È[ˆHÒS	ÜÈ˜[YY\ÙH[™\ÈÝX˜XÝYˆËÈœ›ÛHH\™[ˆÛÈH[Ù[\È
Š›Û™H™XY\ˆØÛÜH[ÈH™\ÚYX[ˆËÈ[™HÙXÛÛ™Ü™XYXÜ›ÜÜÈH˜[YY\Ù\ÊŠˆ8 %H›Ùš[\‰ÜÈÝ[ˆËÈš[™Z[™ÈÚXÙHÚ]ÚÝÜÈ\\™K‚ˆËÂˆËÈ›Ý\™Hš[Y™XØ]\ÙH^H[œÝÙ\ˆY™™\™[]Y\Ý[ÛœÎˆ™\ÚY\ÈÝÂˆËÈ]XÚÙˆH][HH[ˆØ[ÈÜ]\È[œÝ[Y[][Û‹[™š[\ÂˆËÈÝÈ]XÚ]™\žHœËÙ˜]ÈšYÝ\™H[ˆ\È™\Ü\È[™›]YžHHXÝÙ‚ˆËÈYX\Ý\š[™È]ˆ[™\È\ÈH‘QPÕSÓ‹›ÝHÛÜœ™XÝ[Ûˆ8 %›Ý[™È\ÂˆËÈÝX˜XÝY[ž]Ú\™KˆÖ—Õ’×Ô“Ñ’SWÑVWÔÐÓÔTÏSˆYYXØ]\È]ˆ]ˆËÈYÈˆË[›Ý[™ÈØÛÜ\È\ˆ˜]Ë[™H™\ÚYX[]\Ýš\ÙHžHX›Ý]ˆËÈˆH™XYÛÜÝˆYˆ]Ù\È›Ý\È[Ù[\ÈÜ›Û™Ë‚ˆÛÛœÝÝX›HØÛÜ\Ô\‘˜]ÈHÈÝX›J×Ü›Ù‹œØÛÜ\ÊHÈˆŒÂˆÛÛœÝÝX›H™XYœÈHÝX›J×Ü›Ù“›ÝÓœÌL
HÈLŒÂˆÛÛœÝÝX›H™\ÚYœÈHÈÝX›J×Ü›Ù‹™˜]ÓÝ\ŠHÈˆŒÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—H[œÝ[Y[ˆ	KŒYˆØÛÜ\ËÙ˜]È	KŒYˆœÈ\ˆÛØÚÈ™XYOˆ‚ˆŸ‰KŒˆœËÙ˜]È[ˆÝ\˜	ÜÈ	KŒˆœÈ™\ÚYX[
	KŒ‰IHÙˆ]
KÝ[‚ˆœ›Ùš[\ˆš[‰KŒˆœËÙ˜]×ˆ‹ˆØÛÜ\Ô\‘˜]Ë™XYœËØÛÜ\Ô\‘˜]È
ˆ™XYœË™\ÚYœËˆ™\ÚYœÈˆŒÈLŒ
ˆØÛÜ\Ô\‘˜]È
ˆ™XYœÈÈ™\ÚYœÈˆŒˆØÛÜ\Ô\‘˜]È
ˆ‹Œ
ˆ™XYœÊNÂˆB‚ˆËÈ\[[™HÜ™X][Û‹œ›ÚÙ[ˆÝ]ÙˆÝ\˜ˆš[YÛ›HÚ[ˆ]\[™YˆËÈ™XØ]\ÙHH[™HÙˆ™\›Ù\È]™\žHÚ[™ÝÈÛÝ[˜Z[ˆH^YHÈÚÚ\]8 %ˆËÈ[™HÚÛHÚ[\È]\È\È˜\™H[™^[œÚ]™H˜]\ˆ[‚ˆËÈÝXYKˆÙˆÝ\˜\ÈHÚ\™H]^Z[œÎˆYˆHÜZÙH[ˆÝ\˜\ÂˆËÈÛÛ\[][Û‹][X™\ˆ\È[ÜÝÙˆ][™Yˆ]\È›ÝHÛÝ[\‚ˆËÈØ^\ÈÛÈ\Ý\ÈÛX\›K‚ˆYˆ
×Ü›Ù‹œ\[[™\ÐÜ™X]Y
Bˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—H\[[™\È	[HÜ™X]Y
	KŒ™‹Ùœ˜[YK	KŒYˆ\ÈÝ[‚ˆ‰KŒ™ˆ\ÈXXÚ
HH	KŒY‰IHÙˆœ˜[YK	KŒ‰IHÙˆÝ\˜ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü›Ù‹œ\[[™\ÐÜ™X]Yˆœ˜[Y\ÈÈÝX›J×Ü›Ù‹œ\[[™\ÐÜ™X]Y
HÈÝX›Jœ˜[Y\ÊHˆŒˆÝX›J×Ü›Ù‹œ\[[™SœÊH
ˆYKM‹ˆÝX›J×Ü›Ù‹œ\[[™SœÊH
ˆYKMˆÂˆÝX›J×Ü›Ù‹œ\[[™\ÐÜ™X]Y
KˆÝ
×Ü›Ù‹œ\[[™SœÊKˆÝ\•Ý[ÈLŒ
ˆÝX›J×Ü›Ù‹œ\[[™SœÊHÂˆÝX›JÝ\•Ý[
HˆŒ
NÂ‚ˆËÈH\[[™HÓÒÕTÚXÚ\ÈHY™™\™[]X[]Hœ›ÛHHÜ™X][Û‚ˆËÈX›Ý™H[™\ÈHÛ™H\Lˆ][HËŒˆÚ[™ÙYˆÛ˜ÙH\ˆ˜]ËKL[Y\ÂˆËÈHœ˜[YKšXÙYžHHÝ\˜Ü]]LLLLÈœËÙ˜]È™Y›Ü™HHÚ[™ÙK‚ˆËÈHœ›ÛØXÚIÜÈ]˜]H\ÈH][IÜÈÝÛˆ˜[ÚYšY\ˆ8 %ÛÛœÙXÝ]]™H˜]ÜÂˆËÈÚ\š[™ÈH\[[™H\È[ˆTÔÕSTSÓˆX›Ý]\È]IÜÈ˜]ÈÜ™\‹[™BˆËÈÝÈ˜]H\™HÛÝ[YX[ˆHÛÛ\\™H\ÈHØ\ÝY[œÝXÝ[Ûˆ\ˆ˜]ÂˆËÈ˜]\ˆ[ˆHØ]š[™ËˆÖ—Õ’×Ó“×ÔTSS‘WÐÐPÒLOLH\ÈHÛÛ›Û\›K‚ˆÂˆÝ]XÈZ[Ý\ÝH\ÝHHÂˆÛÛœÝZ[ÝH×Ü\PØXÚLR]ÈH\ÝÂˆÛÛœÝZ[ÝHH×Ü\PØXÚLSZ\ÜÙ\ÈH\ÝNÂˆ\ÝH×Ü\PØXÚLR]ÎÂˆ\ÝHH×Ü\PØXÚLSZ\ÜÙ\ÎÂˆYˆ
JBˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—H\[[™HÛÚÝ\ˆ	KŒY‰IHÙ\™YžHHÛ™KY[žH‚ˆ˜ØXÚH
	[H]È	[HZ\ÜË	[HÛÚÝ\ËÙœ˜[YK	^H[ˆH‚ˆX›JWˆ‹ˆLŒ
ˆÝX›J
HÈÝX›J
ÈJKˆ
[œÚYÛ™YÛ™ÈÛ™ÊY
[œÚYÛ™YÛ™ÈÛ™ÊYKˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈ

ÈJHÈœ˜[Y\Èˆ
Kˆ‹Oœ\[[™\ËœÚ^™J
JNÂˆB‚ˆËÈ‹‹˜[™Ú]Ý]ÚYXXÝX[HTËˆH™[™\™\ˆ[œÈÛˆHÜ˜\XÜÂˆËÈ[\	ÜÈ™XYÛÈ]™\ž][™ÈH[\Ù\È™]ÙY[ˆÛÈ™\Ù[È\È[‚ˆËÈ]ÛÛ[[ˆ8 %[˜ÛY[™ÈHÛY\]HÜÙˆ]ÈÛÜÚXÚ\È›ÝˆËÈÛÜšÈ[™ÚXÚ›ÈÞXÛ\È›Ùš[HØ[ˆÙYH
ÜKÜ[\ÜÝ]Ëš
KˆXÚÜØ\ÂˆËÈH[X™\ˆ]XZÙ\ÈH™\Ý™XYX›NˆHš[™ÈØ[ÈÝÜÈ]]™\žBˆËÈ[œØ]\ÙšYYÐRUÔ‘Q×ÓQSH[™™\Ý[Y\ÈÛˆH‘VXÚËÛÈHœ˜[YBˆËÈÛÜÝÈ]X\ÝÛ™HÛY\\š[Ù\ˆ[™[Ù™ˆØZ][ˆ]‚ˆÝ]XÈ[\Ý]È\Ý[\ßNÂˆÛÛœÝ[\Ý]ÈH[\Ý]×Ô™XY

NÂˆÛÛœÝZ[ÝXÚÜÈHXÚÜÈH\Ý[\XÚÜÎÂˆÛÛœÝZ[ÝØ[ÓœÈHØ[ÓœÈH\Ý[\Ø[ÓœÎÂˆËÈM\ÈHÛÛ[X[™›ØÙ\ÜÛÜ‰ÜÈÕÓˆÛÜÝ[™]™YYÈØ^Z[™È™XØ]\ÙBˆËÈHØ[È\ÈÚ\™HH™[™\™\ˆ\ÈØ[Yœ›ÛNˆØ[ØÛÛZ[œÈ]™\žBˆËÈ˜]Ë]™\žHÝX›Z][™H™XY˜XÚËÛÈ™XY[™È]\ÈHÛÛ[X[™ˆËÈ›ØÙ\ÜÛÜ‰ÜÈÛÜÝÝ™\‹\Ý]\È]žHHÚÛHÙˆH™[™\™\‹ˆ\È\ÂˆËÈHLŽN\È\›HØÜËÜ\™‹XÜK\[‹›Y0©Ìˆ\ÈX›Ý][™[[›ÝÈ]ˆËÈÛÝ[Û›H™HÛÝžHÝX˜XÝ[™ÈÛÈ[™\ÈÙˆ\È™\ÜžH[™‚ˆÛÛœÝZ[ÝMœÈHØ[ÓœÈˆÛ›ÝÛˆÈØ[ÓœÈHÛ›ÝÛˆˆÂˆÛÛœÝZ[ÝÛY\HœÛY\œÈH\Ý[\œÛY\œÎÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—H[\	[HXÚÜÈ
	KŒ™‹Ùœ˜[YJHÛY\	KŒY‰IHØ[È	KŒY‰IH‚ˆ–ÜM	KŒY—H˜›[šËZ\Üˆ	KŒY‰IH[˜XØÛÝ[Y	KŒY‰IWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊYXÚÜËˆœ˜[Y\ÈÈÝX›JXÚÜÊHÈÝX›Jœ˜[Y\ÊHˆŒˆÝ
ÛY\
KÝ
Ø[ÓœÊKÝ
MœÊKˆÝ
š\Ü“œÈH\Ý[\š\Ü“œÊKˆLŒHÝ
ÛY\
ÈØ[ÓœÈ
È
š\Ü“œÈH\Ý[\š\Ü“œÊJJNÂ‚ˆËÈ‹‹˜[™ÝÈ]XÚÙˆ]ÛY\Ø\ÈÓˆHÔ’UPÐSU
\LJKˆBˆËÈ[™HX›Ý™H\È™Y[ˆš[YÚ[˜ÙH\N[™Ø^\ÈÛ›H]H[\Ø\ÂˆËÈÙ™ˆHÔNÈ]Ø[››ÝØ^HÚ]\ˆ[ž][™ÈØ\ÈØZ][™È›Üˆ][™BˆËÈ[œÝÙ\ˆXÚY\ÈÚ]\ˆHÛY\\ÈÛÜœ™XÝ™Z]š[Ý\ˆÜˆœ˜[YH[YK‚ˆËÈÜKÜ[\ÜÝ]ËšYš[™\ÈH\ØÜš[Z[˜]Üˆ
YH™^Ø[ÈY˜[˜ÙHBˆËÈš[™ÈÝ\œÛÜÊH[™ÚHHZ[\ÙXÛÛ™šYÝ\™H\È[ˆTTˆ“ÕS‘8 %š[]ˆËÈÚ]HÛÜ™XÛÈ]Ø[››Ý™H][ÝY\ÈHØ]š[™ÈžHXØÚY[‚ˆÛÛœÝZ[Ý›ÙÈHœ›ÙÜ™\ÜÕXÚÜÈH\Ý[\œ›ÙÜ™\ÜÕXÚÜÎÂˆÛÛœÝZ[Ý›ÙÔÛY\BˆœÛY\™Y›Ü™T›ÙÜ™\ÜÓœÈH\Ý[\œÛY\™Y›Ü™T›ÙÜ™\ÜÓœÎÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÛY\ÛˆHÜš]XØ[]ˆ	[HÙˆ	[HXÚÜÈXYH‚ˆœ›ÙÜ™\ÜÈ
	KŒY‰IJKÛY\™Y›Ü™H[H	KŒYˆ\ÈÙˆ	KŒYˆ\È‚ˆH	KŒ™ˆ\ËÙœ˜[YHÙˆ][˜ÞWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY›ÙË
[œÚYÛ™YÛ™ÈÛ™ÊYXÚÜËˆXÚÜÈÈLŒ
ˆÝX›J›ÙÊHÈÝX›JXÚÜÊHˆŒˆÝX›J›ÙÔÛY\
H
ˆYKM‹ÝX›JÛY\
H
ˆYKM‹ˆœ˜[Y\ÈÈÝX›J›ÙÔÛY\
H
ˆYKMˆÈÝX›Jœ˜[Y\ÊHˆŒ
NÂ‚ˆËÈHÛÈŒ‹LLŽH][˜ÞH][\ËXXÚÚ]]È[™ØYÙ[Y[ÛÝ[\‚ˆËÈ
ÛÝÚHMLJNˆXYÙ\ˆXÚÜÈ
HÛY\ÚÚ\YY\ˆH›ÙXÝ]™HØ[ÊH[™ˆËÈZY]Ø[ÈœˆX›XØ][Ûˆ
HÝY\ÝÙY\Èš[™ÈÛÛœÝ[\[Ûˆ\ˆXÚÙ]
K‚ˆËÈ™\›È™^È[ˆ\›YYY˜][\ÈHY™XÝÈ™\›È[™\ˆHÖ—ÔMÓ“×Ê‚ˆËÈÛÛ›Û\›\È\ÈH\›\ÈÛÜšÚ[™Ë‚ˆÝ]XÈZ[Ý\ÝXYÙ\ˆH\ÝZYØ[ÈH\Ý[˜\ÝHÂˆÛÛœÝZ[ÝXYÙ\ˆH™XYÙ\•XÚÜÈH\ÝXYÙ\ŽÂˆÛÛœÝZ[ÝZYØ[ÈHMÔœ“ZYØ[ÔÝÜ™\Ê
HH\ÝZYØ[ÎÂˆÛÛœÝZ[Ý[˜\ÝHš[˜\ÝXÚÜÈH\Ý[˜\ÝÂˆ\ÝXYÙ\ˆH™XYÙ\•XÚÜÎÂˆ\ÝZYØ[È
ÏHZYØ[ÎÂˆ\Ý[˜\ÝHš[˜\ÝXÚÜÎÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—Hš[™È][˜ÞH\›\ÎˆXYÙ\ˆXÚÜÈ	[HÙˆ	[H
	KŒY‰IJH‚ˆ›ZY]Ø[ÈœˆÝÜ™\È	[H
	KŒY‹Ùœ˜[YJH[Y˜\Ý˜\È	[H‚ˆŠ	KŒY‹Ùœ˜[YJWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊYXYÙ\‹
[œÚYÛ™YÛ™ÈÛ™ÊYXÚÜËˆXÚÜÈÈLŒ
ˆÝX›JXYÙ\ŠHÈÝX›JXÚÜÊHˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊYZYØ[Ëˆœ˜[Y\ÈÈÝX›JZYØ[ÊHÈÝX›Jœ˜[Y\ÊHˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊY[˜\Ýˆœ˜[Y\ÈÈÝX›J[˜\Ý
HÈÝX›Jœ˜[Y\ÊHˆŒ
NÂ‚ˆËÈ\LÈ][HŽˆH˜]È™XY	ÜÈ™[˜ÙHØZ]\šÙYˆ]™\žH\\ÛÙBˆËÈ\ÈÛ\ÜÚYšYYÛÈH\šÈ™]™\ˆ[™ØYÙYˆ
[™XYP][žHÈÜ[ŠH[™ˆËÈHØZÙH™YXØ]H™]™\ˆš\™\Èˆ
\šÜÈOH[Y[Ý]ÊH\™H›Ýš\ÚX›BˆËÈ\™H˜]\ˆ[ˆ[™™\œ™Yœ›ÛHHœ˜[YH[YH
ÛÝÚHMLJKˆHš\œÝ˜YˆËÈØ]ÚYH™XYÚ[\ˆ[œÝXYÙˆH™[˜ÙHÛÜ™[™\È[™H\ÂˆËÈÚ]ØZYÛÎˆ\šÜÈK‹Ùœ˜[YK[Y[Ý]ÈK‹Ùœ˜[YKØZÙ\È‚ˆÂˆÝ]XÈ™[˜ÙUØZ]Ý]È\ÝÎÂˆÛÛœÝ™[˜ÙUØZ]Ý]ÈÈH™[˜ÙUØZ]ÔÝ]Ê
NÂˆÛÛœÝÝX›H[ˆHœ˜[Y\ÈÈKŒÈÝX›Jœ˜[Y\ÊHˆŒÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—H™[˜ÙHØZ]
\LÊNˆ›ÙHØ[È	KŒY‹Ùœ˜[YH‚ˆœ™XYH][žH	KŒYˆÜ[‹\™\ÛÛ™Y	KŒYˆ\šÜÈ	KŒYˆ
ÛÚÙ[ˆ	KŒY‹‚ˆ[Y[Ý]È	KŒY‹RTÔÑQ	KŒY‹XYØZ[ˆ	KŒYŠHÛÛ[™Y	KŒYˆ\ÜÝ›ÝYÚ	KŒYˆ‚ˆ™^XÝ]ÜˆÝÜ™\ÈÙY[ˆÚ[H\šÙY	KŒY‹ØZÙ\È	KŒY‹Ùœ˜[YI\×ˆ‹ˆÝX›JË˜›ÙPØ[ÈH\ÝË˜›ÙPØ[ÊH
ˆ[‹ˆÝX›JËœ™XYP][žHH\ÝËœ™XYP][žJH
ˆ[‹ˆÝX›JËœÜ[”™\ÛÛ™YH\ÝËœÜ[”™\ÛÛ™Y
H
ˆ[‹ˆÝX›JËœ\šÜÈH\ÝËœ\šÜÊH
ˆ[‹ˆÝX›JËœ\šÕÛÚÙ[ˆH\ÝËœ\šÕÛÚÙ[ŠH
ˆ[‹ˆÝX›JËœ\šÕ[Y[Ý]ÈH\ÝËœ\šÕ[Y[Ý]ÊH
ˆ[‹ˆÝX›JËœ\šÓZ\ÜÙYH\ÝËœ\šÓZ\ÜÙY
H
ˆ[‹ˆÝX›JËœ\šÑXYØZ[ˆH\ÝËœ\šÑXYØZ[ŠH
ˆ[‹ˆÝX›JË˜ÛÛ[™YH\ÝË˜ÛÛ[™Y
H
ˆ[‹ˆÝX›JËœ\ÜÝ›ÝYÚH\ÝËœ\ÜÝ›ÝYÚ
H
ˆ[‹ˆÝX›JËœÝÜ™PÚXÚÜÈH\ÝËœÝÜ™PÚXÚÜÊH
ˆ[‹ˆÝX›JËØZÙPØ[ÈH\ÝËØZÙPØ[ÊH
ˆ[‹ˆ™[˜ÙUØZ]Ñ[˜X›Y

HÈˆˆˆˆÐÖ—Ñ‘SÑWÔT’ÏLˆÜ[›š[™×HŠNÂˆ\ÝÈHÎÂˆB‚ˆËÈ‹‹˜[™HÛ™H[™ÈÝ]ÚYX\È™]™\ˆ™Y[ˆX›HÈØ^NˆÝÈ]XÚÙˆ]ˆËÈ\ÈH[\ÓÔ’ÒS‘È[™ÝÈ]XÚ\ÈH[\“Õ•S“’S‘ÈUS‚ˆËÂˆËÈ\Lˆ][HŒK[™]\ÈHØ[YHØ\[ˆHY™™\™[XÙKˆ]™\žBˆËÈÛÛ[[ˆX›Ý™H\ÈHØ[XÛØÚÈ[\˜[YX\Ý\™Yœ›ÛH[œÚYH\È™XYˆËÈ[™HØ[XÛØÚÈ[\˜[Ø[››Ý[ÙHÜ[\ÈÚ[™ÈÛÛY][™Èˆœ›ÛBˆËÈÙHÜ[\È\ØÚY[YØZ][™È›ÜˆÛÛYX›ÙH[ÙH‹ˆ\™‹\[‹\\L›YˆËÈ0©Í˜ÙÈXYH^XÝH]Z\ÝZÙH[ˆ™]™\œÙH8 %]™XYÝ]ÚYX	ÜÈ™\ÚYX[\ÂˆËÈ™ÝY\ÝÚ[][][ÛˆŒÈ\ÈˆÚ[ˆ]ÎIH]HH[\Ø\ÈÚ[\H›ØÚÙY8 %[™ˆËÈH[‰ÜÈÝÛˆ][HŒH\ÚÜÈ›ÜˆHÜ]˜]\ˆ[ˆ[›Ý\ˆÝY\ÜË‚ˆËÂˆËÈÛ™HÛØÚÈ™XY[œÝÙ\œÈ]ˆÓÐÒ×Õ‘PQÐÔUSQWÒQ\ÈTÈ™XY	ÜÈÔBˆËÈ[YKÛÈØ[HÜX\ÈžHYš[š][Ûˆ]™\žH˜[›ÜÙXÛÛ™H[\Ø\ÈÙ™ˆBˆËÈÛÜ™K[™HÛY\ÛÝ[\ˆX›Ý™H[™XYHXØÛÝ[È›ÜˆH[X™\˜]H\‚ˆËÈÚ]]™\ˆ\ÈY\ÈH[\›ØÚÙYÛˆÛÛYX›ÙH[ÙNˆH]]^Hš]™\‹ˆËÈH™[˜ÙKHÝY\Ý‚ˆËÂˆËÈ\ÈÛÜÝÈÛ™HÛØÚ×ÙÙ][YX\ˆ‘TÔ•
]™\žHˆÙXÛÛ™ÊK›Ý\ˆœ˜[YKˆËÈÚXÚ\ÈHÚÛH™X\ÛÛˆ]\ÈØY™HÈX]™HÛŽˆ[ˆ[œÝ[Y[ÛˆHÝˆËÈ]Ø[ˆØ[˜Ù[HY™™XÝ]YX\Ý\™\È
ÛÝÚHŒŒÊK[™\ÈÛ™H\È›ÝÛ‚ˆËÈ[žH]][‚ˆÂˆÝ]XÈZ[Ý\ÝÜSœÈHÂˆÝ]XÈ›ÛÛ]™PÜHH˜[ÙNÂˆ[Y\ÜXÈÝÞßNÂˆÛØÚ×ÙÙ][YJÓÐÒ×Õ‘PQÐÔUSQWÒQ	˜ÝÊNÂˆÛÛœÝZ[ÝÜSœÈBˆZ[Ý
ÝË—ÜÙXÊH
ˆL[
ÈZ[Ý
ÝË—ÛœÙXÊNÂˆÛÛœÝZ[ÝÜHH]™PÜHÈÜSœÈH\ÝÜSœÈˆÂˆÛÛœÝ›ÛÛš\œÝHZ]™PÜNÂˆ\ÝÜSœÈHÜSœÎÂˆ]™PÜHHYNÂˆÛÛœÝÝX›HØ[œÈH\È
ˆYMŽÂˆYˆ
Yš\œÝ	‰ˆØ[œÈˆŒ
BˆÂˆÛÛœÝÝX›HÙ™“œÈHØ[œÈˆÝX›JÜJHÈØ[œÈHÝX›JÜJHˆŒÂˆÛÛœÝÝX›H›ØÚÙYœÈBˆÙ™“œÈˆÝX›JÛY\
HÈÙ™“œÈHÝX›JÛY\
HˆŒÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—H[\™XYˆ	KŒY‰IHÛˆÔHÙ™‹PÔH	KŒ™ˆ\ËÙœ˜[YH‚ˆHÛY\	KŒ™ˆ
È“ÐÒÑQ	KŒ™ˆ8 %H›ØÚÙY\\ÈÝ]ÚYX‚ˆ[YH]\È›ÝÛÜšÈ[™Ø[››Ý™HÜ[Z\ÙY]Ø^Wˆ‹ˆLŒ
ˆÝX›JÜJHÈØ[œËˆœ˜[Y\ÈÈÙ™“œÈ
ˆYKMˆÈÝX›Jœ˜[Y\ÊHˆŒˆœ˜[Y\ÈÈÝX›JÛY\
H
ˆYKMˆÈÝX›Jœ˜[Y\ÊHˆŒˆœ˜[Y\ÈÈ›ØÚÙYœÈ
ˆYKMˆÈÝX›Jœ˜[Y\ÊHˆŒ
NÂ‚ˆËÈKKHKŒNˆHP“IÔÈÕÓˆÓÕ‘TQÑKS‘ÒUÓÕ‘TQÑHÑTÈ“ÕˆËÈQPSˆ
\LL
HKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKBˆËÂˆËÈ]™\žHÛÛ[[ˆX›Ý™H\ÈH\˜Ù[YÙHÙˆÐSˆ›Ý[™ÈX›Ý™HØ^\ÂˆËÈÚ]œ˜XÝ[ÛˆÙˆHST	ÔÈÔHH˜[YY\Ù\ÈXØÛÝ[›Ü‹Ü‚ˆËÈ˜[Y\ÈH\]\È[œÚYH›ÈØÛÜH][8 %[™Ý]ÚYXˆËÈ™XYÈZÙHHØ]YÛÜžH
HØ[ËHÝY\ÝŠHÚ[ˆ]\ÈBˆËÈ™\ÚYX[ˆÛÈÝ]H]ˆ\Ù\ËHØ[È
ÚXÚ\È›ÝBˆËÈ›Ù”ØÛÜH[™\ÈÛ›H]™\ˆØZ[™YžHÝX˜XÝ[ÛŠK[™BˆËÈÙ[Z[™[HS”ÐÓÔQ™[XZ[™\‹\ÈÚ\™\ÈÙˆ\È™XY	ÜÈÔK‚ˆËÂˆËÈS‘HÐT“’S‘ÈTÈHÒS•™XØ]\ÙH\LLYX\Ý\™YBˆËÈÛÝ™\˜YÙH™Y›Ü™HÜš][™È\È[™]\ÈQÒ8 %ÌÉH\Ù\ËŒŽ	BˆËÈØ[ËŒ	H[œØÛÜY8 %ÛˆHØ[YHZ[ÚÜÙHÝ™X[\ØÛÛ[[‚ˆËÈ[™\‹\™\ÜÈ]ÈÝÛˆÝXœÞ\Ý[HžHH˜XÝÜˆÙˆ\Kˆ
ŠH\ÙBˆËÈX›HØ[ˆXØÛÝ[›ÜˆL	HÙˆH™XY[™Ý[™HÜ›Û™ÈX›Ý]ˆËÈ]™\žH›ÝÊŠ‹™XØ]\ÙHHY™XÝ\ÈZ\Ø]šX][Û‹›ÝÛZ\ÜÚ[ÛŽ‚ˆËÈ\ØYÝ™X[X	ÜÈÛÜÝ\ÈÚ\™ÙYÈ™XÛÜ™ÚXÚ\ÈH™X[ˆËÈØÛÜH]™X[HYÛÛZ[ˆ]ˆÛÝ™\˜YÙH\È™XÙ\ÜØ\žH[™]\ÂˆËÈ›ÝÚ\™H™X\ˆÝY™šXÚY[[™H[™H]š[YÛ›HH[X™\‚ˆËÈÛÝ[™HH™^[™ÈÈZ\ÛXYÛÛYX›ÙKˆHÚXÚÈ]ÐS‚ˆËÈØ]Ú]ÛÛ\\™\ÈXXÚ\ÙHÚ]HÖSP“ÓÈ[\[Y[[™ÈBˆËÈÝXœÞ\Ý[H]\È˜[YYY\ˆ8 %ÛÛËÜ\ÙWÝœ×Ü\™‹œX‚ˆÛÛœÝÝX›HÜS\ÈHÝX›JÜJH
ˆYKMŽÂˆÛÛœÝÝX›H\ÙS\ÈHÝX›JÛ›ÝÛŠH
ˆYKMŽÂˆÛÛœÝÝX›HØ[ÓÛ›S\ÈHÝX›JMœÊH
ˆYKMŽÂˆÛÛœÝÝX›H[œØÛÜY\ÈBˆÜS\Èˆ\ÙS\È
ÈØ[ÓÛ›S\ÈÈÜS\ÈH\ÙS\ÈHØ[ÓÛ›S\ÈˆŒÂˆÛÛœÝ]]ÈÜÝHÉ—JÝX›HJHÂˆ™]\›ˆÜS\ÈˆŒÈLŒ
ˆHÈÜS\ÈˆŒÂˆNÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÓÕ‘TQÑNˆ\Ù\È	KŒY‰IHÙˆH[\	ÜÈÔH‚ˆŠ	KŒ™ˆÙˆ	KŒ™ˆ\ËÙœ˜[YJH
ÈMØ[È	KŒY‰IH
›ÝH\ÙNˆ‚ˆ˜Ø[ØZ[\ÈH\Ù\ÊH
ÈS”ÐÓÔQ	KŒY‰IH8 %[œØÛÜY\È‚ˆ““ÕHØ]YÛÜžK]\ÈÛÙH\ÈX›HÙ\È›ÝYX\Ý\™Kˆ‚ˆS‘QÒÓÕ‘TQÑHTÈ“ÕQÔ‘QSQS•ˆH\ÙH˜[Y\ÈHÐÓÔK‚ˆ››ÝHÝXœÞ\Ý[H
ÛÝÚHÍÊKˆ[ˆÛÛËÜ\ÙWÝœ×Ü\™‹œH‚ˆ˜YØZ[œÝH\™˜Ø\\™HÙˆ\È[ˆ™Y›Ü™HšXÚ[™È‚ˆ˜[ž][™ÈÙ™ˆHÛÛ[[ˆX›Ý™K—ˆ‹ˆÜÝ
\ÙS\ÊKœ˜[Y\ÈÈ\ÙS\ÈÈÝX›Jœ˜[Y\ÊHˆŒˆœ˜[Y\ÈÈÜS\ÈÈÝX›Jœ˜[Y\ÊHˆŒÜÝ
Ø[ÓÛ›S\ÊKˆÜÝ
[œØÛÜY\ÊJNÂˆBˆB‚ˆËÈ‹‹˜[™Ú]HØ[ÈØ\ÈÐSÒS‘ËˆMX›Ý™H\ÈH[X™\ˆÙ‚ˆËÈZ[\ÙXÛÛ™ÎÈÛˆ]ÈÝÛˆ]Ý\ÜÈ›È\Ý\Ú\ÈX›Ý]Ú]ÈÚ[™ÙKˆËÈÚXÚ\ÈHÝ]H0©ÌˆÙˆØÜËÜ\™‹XÜK\[‹›Y\ØÜšX™\È\ÂˆËÈ˜ÛÛ\][H[š[œÝ[Y[Y[œÚYH‹ˆ\ÙHÛÈÛÝ[È\›ˆ][ÈHÛÜÝˆËÈ\ˆXÚÙ][™HÛÜÝ\ˆ™YÚ\Ý\‹]Üš]HÛÜ™[™Üš]T™YÚ\Ý\˜8 %BˆËÈÙXÝ[Û‰ÜÈXY[™ÈÝ\ÜXÝØ[YÛ˜ÙH\ˆÛÜ™Ùˆ]™\žHÑUÐÓÓ”ÕS•8 %ˆËÈ\È\ÝX›HH[ÛY[HÛÜ™˜]H\ÈÛ›ÝÛ‹‚ˆËÂˆËÈ™XYÙ™ˆHØ[YHÚ[™ÝÈ\È]™\ž][™ÈX›Ý™KÛÈH\š]Y]XÈ\ÂˆËÈœËÜXÚÙ]HMœÈÈXÚÙ]ÈÚ]›ÈÜ›ÜÜË]Ú[™ÝÈZ^[™Ë‚ˆÝ]XÈZ[Ý\ÝXÚÙ]ÈH\Ý™YÕÜš]\ÈHÂˆÛÛœÝZ[ÝXÚÙ]ÈHMÔXÚÙ]ÛÝ[

NÂˆÛÛœÝZ[Ý™YÕÜš]\ÈHMÔ™YÚ\Ý\•Üš]PÛÝ[

NÂˆÛÛœÝZ[ÝXÚÙ]ÈHXÚÙ]ÈH\ÝXÚÙ]ÎÂˆÛÛœÝZ[Ý™YÕÜš]\ÈH™YÕÜš]\ÈH\Ý™YÕÜš]\ÎÂˆ\ÝXÚÙ]ÈHXÚÙ]ÎÂˆ\Ý™YÕÜš]\ÈH™YÕÜš]\ÎÂˆËÈ‹‹˜[™ÝÈÜÙHÛÜ™ÈÙ\™HÜš][‹ˆH[ÈÚ\™H™X\ˆL	H\ÈÚ]BˆËÈ\MÈ[ˆÛÜH™YXÝÈ›Üˆ\È]H
]ÈÛÛœÝ[˜[šÜÈ]™H]ˆËÈŒ[™X›Ý™K›ÝÚ\™H™X\ˆHØÜ˜]ÚZ\œ›ÜŠNÈHÝÈÚ\™HÛÝ[YX[‚ˆËÈH][H\ÈÛÜ]XÚ\ÜÈ[ˆ0©Ì‹ŒH\Ý[X]\Ë[™Û›HHÛÝ[\ˆØ[‚ˆËÈØ^HÚXÚˆY™™\™[˜ÙYÝ™\ˆHØ[YHÚ[™ÝÈ\È]™\ž][™È[ÙK‚ˆÝ]XÈZ[Ý\Ý[ÈH\ÝÛÝÈHÂˆÛÛœÝZ[Ý[ÈHMÔ™YÔ[[ÑÛÜ™Ê
NÂˆÛÛœÝZ[ÝÛÝÈHMÔ™YÔ[”ÛÝÑÛÜ™Ê
NÂˆÛÛœÝZ[Ý[ÈH[ÈH\Ý[ËÛÝÈHÛÝÈH\ÝÛÝÎÂˆ\Ý[ÈH[ÎÂˆ\ÝÛÝÈHÛÝÎÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HM	[HXÚÙ]È
	[KÙœ˜[YK	KŒˆœÈXXÚ
H	[H‚ˆœ™YÚ\Ý\ˆÛÜ™È
	[KÙœ˜[YK	KŒY‹ÜXÚÙ]	KŒY‰IH[ÊWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊYXÚÙ]Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈXÚÙ]ÈÈœ˜[Y\Èˆ
KˆXÚÙ]ÈÈÝX›JMœÊHÈÝX›JXÚÙ]ÊHˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊY™YÕÜš]\Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈ™YÕÜš]\ÈÈœ˜[Y\Èˆ
KˆXÚÙ]ÈÈÝX›J™YÕÜš]\ÊHÈÝX›JXÚÙ]ÊHˆŒˆ
[È
ÈÛÝÊHÈLŒ
ˆÝX›J[ÊHÈÝX›J[È
ÈÛÝÊBˆˆŒ
NÂˆËÈ[™\ˆÖ—ÔMÕ‘T’Q–WÐ•S×Ô‘QÔÈ\È]\ÝÝ^H]™\›Ëˆš[YÛ›HÚ[‚ˆËÈH™\šYšY\ˆ\ÈÛˆÔˆ]\È›Û‹^™\›ËÛÈ[ˆÜ™[˜\žH[ˆ\È›ÝÚ]™[ˆBˆËÈ[™HØ^Z[™ÈŒZ\ÛX]Ú\Èˆ›ÜˆHÚXÚÈ]™]™\ˆ˜[ˆKHÚXÚÛÝ[™HBˆËÈÛX[ˆ™\Ý[œ›ÛHH\Ý]ÛÝ[›Ý]™H˜Z[Y‚ˆYˆ
ÛÛœÝZ[Ý[HHMÔ™YÔ[“Z\ÛX]Ú\Ê
JBˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HM•SÈ‘QÒTÕTˆRTÓPUÒTÎˆ	[H8 %H[È][™‚ˆH\‹YÛÜ™]TÐQÔ‘QNÈ\È\ÈHY™XÝˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê[[JNÂ‚ˆËÈ‹‹˜[™ÒPÒXÚÙ]ÈÜÙHÙ\™KˆMÕ\PÛÝ[[™MÓÜÛÙPÛÝ[ˆËÈ]™H^\ÝYÚ[˜ÙH\ÙH[™Ù\™H[˜Ü™[Y[YÛˆ]™\žHÚ[™ÛHXÚÙ]ˆËÈ[™[[›ÝÈ^HÙ\™HØ[Yœ›ÛH“ÕÒT‘H[ˆH[[YH8 %ÛÈÚ]\ÂˆËÈHM‹ˆ\ÈÙˆØ[ÈXÝX[HØ[Ú[™ÈˆØ\È[˜[œÝÙ\˜X›HÚ[HH]HØ]ˆËÈ[ˆY[[ÜžKˆ]\ÈHØ[YHØ\™XÛÜ™Ø\È[ˆ[[\ÈÜ]]ˆËÈ[™Ü][™È]\ÈÚ]›Ý[™HÝ™X[HÝX\™
ÛÝÚ\ÈÌKÌŠK‚ˆËÂˆËÈ]X]\œÈ\™HÜXÚYšXØ[H™XØ]\ÙHHÜ\˜]Ü‰ÜÈXÚÙ]Z^Q‘‘T”ÂˆËÈ”“ÓHHPQTÔÈ“ÕUIÔÈSˆÒS‘›Ý\Ý[ˆÚ^™NˆMœÈ\ˆXÚÙ]ˆËÈYØZ[œÝÝ\ˆLLLLLË[™ËŽ™YÚ\Ý\ˆÛÜ™È\ˆXÚÙ]YØZ[œÝKÛÂˆËÈ^HÝX›Z]›ÜÜ[Û˜[H[Ü™H›Û‹\™YÚ\Ý\ˆXÚÙ]È[™\‹TPÒÑUÛÜÝˆËÈÛZ[˜]\ÈZ\ˆØ[Ëˆ›Ý[™È[ˆH[[YH\ØÜšX™Y]Z^È\ÂˆËÈÙ\Ë[™ØÜËÜ\™‹\[‹\\›Y0©ÌÈ˜[šÜÈHØ[È][\ÈÙ™ˆ]‚ˆËÂˆËÈY™™\™[˜ÙY\ˆÚ[™ÝÈZÙH]™\žHÝ\ˆ˜]HÛˆ\ÙH[™\ËÛÈBˆËÈÚ\™\È]šYH[ÈHØ[YHXÚÙ]ØHœË\\‹\XÚÙ]X›Ý™H\ÂˆËÈÛÛ\]Yœ›ÛKˆ\\Èš\œÝ™XØ]\ÙHH\HÜ]\ÈHÛØ\œÙH[œÝÙ\‚ˆËÈ
\HÌH\™H™YÚ\Ý\ˆÜš]\Ë\Hˆ\Èš[™Èš[\‹\HÈ\ÂˆËÈ]™\ž][™ÈHÛÛ[X[™›ØÙ\ÜÛÜˆXÝX[HÙ\ÊK[ˆ]™\žH\KLÂˆËÈÜÛÙHÚ]H›Û‹^™\›È[KÛÜYžHœ™\]Y[˜ÞK‚ˆËÈH\‹]™XYÙ[œÝ\ÉÜÈÝÛˆÛÜœ™XÝ™\ÜÈÚXÚÈ
\][HXŠKˆš[YˆËÈÛ›HÚ[ˆH™\šYšY\ˆ\È[›š[™ÈÜˆHZ\ÛX]Ú^\ÝË›ÜˆHØ[YBˆËÈ™X\ÛÛˆH[Ë\™YÚ\Ý\ˆ[™H\Îˆ[ˆÜ™[˜\žH[ˆ]\Ý›Ý™H[™YBˆËÈ[™HØ^Z[™ÈŒZ\ÛX]Ú\Èˆ›ÜˆHÚXÚÈ]™]™\ˆ˜[‹ˆ™XYØ\Èš[YˆËÈ[Û™ÜÚYH™XØ]\ÙHHÛÛ\\š\ÛÛˆ\ÈÛ›H^XÝÚ[HÛ™H™XYØ[ÜË‚ˆÂˆZ[ÝØ[Ù\œÈHÂˆÛÛœÝZ[Ý˜YHMÐÙ[œÝ\ÓZ\ÛX]Ú\Ê	Ø[Ù\œÊNÂˆYˆ
˜YÙ][ŠÖ—ÔMÕ‘T’Q–WÐÓÕS•T”ÈŠJBˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HMÙ[œÝ\È™\šYžNˆ	[HÙˆLÍHÛÝ[\œÈTÐQÔ‘QH‚ˆŠ\‹]™XYœÈ]ÛZXÊK	[HØ[Ú[™È™XY	\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊX˜Y
[œÚYÛ™YÛ™ÈÛ™Ê]Ø[Ù\œËˆØ[Ù\œÈOHHÈˆˆˆœÈ8 %HÛÛ\\š\ÛÛˆ\È“Õ^XÝX›Ý™HHŠNÂˆBˆÝ]XÈZ[Ý\Ý\\ÖÍHHßK\ÝÜÛÙ\ÖÌLŽHHßNÂˆZ[Ý\\ÖÍNÂˆ›Üˆ
Z[Ì—ÝHÈÈ
ÊÊBˆÂˆÛÛœÝZ[ÝÈHMÕ\PÛÝ[

NÂˆ\\ÖÝHHÈH\Ý\\ÖÝNÂˆ\Ý\\ÖÝHHÎÂˆBˆÛÛœÝ]]ÈXÚÙ]ÝHÉ—JZ[ÝŠHÂˆ™]\›ˆXÚÙ]ÈÈLŒ
ˆÝX›JŠHÈÝX›JXÚÙ]ÊHˆŒÂˆNÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HM\\Îˆ
™YË\[ŠH	KŒY‰IHJ™YË\Z\ŠH	KŒY‰IH‚ˆŠš[\ŠH	KŒY‰IHÊÛÛ[X[™
H	KŒY‰IH	[HËÙœ˜[YWˆ‹ˆXÚÙ]Ý
\\ÖÌJKXÚÙ]Ý
\\ÖÌWJKXÚÙ]Ý
\\ÖÌ—JKˆXÚÙ]Ý
\\ÖÌ×JKˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈ\\ÖÌ×HÈœ˜[Y\Èˆ
JNÂ‚ˆËÈHš[\‹\[ˆÙ[œÝ\È
\L][HXJKˆ˜X›Ý™HØ^\ÈÝÈPS–H›Ë[ÜˆËÈÛÜ™ÈHØ[ÈYY]ÎÈ\ÈØ^\ÈÝÈX[žHÐSÈ^HÛÜÝÚXÚ\ÈBˆËÈÛ›H›Ü›H[ˆÚXÚ˜ÛØ[\ØÙH[Hˆ\ÈH˜[YKˆYX[ˆ[ˆ[™ÝKŒÛÝ[ˆËÈYX[ˆH][HØ]™\È›Ý[™È][[™H\ÝÙÜ˜[HÛÝ[Ø^HÚ\™HBˆËÈÛ™\È\™NÈHš[™ÈÚ\™HÙ\\˜]\Èš]™\ˆš[™ÈY[™Èœ›ÛHY[™È[œÚYBˆËÈH]IÜÈÝÛˆ[™\™XÝY™™\œËÚXÚ\™HY™™\™[›ÙXÙ\œË‚ˆÂˆÝ]XÈZ[Ý\Ý[œÈH\Ýš[™ÈH\Ý\ÝÎHHßNÂˆÛÛœÝZ[Ý[œÈHMÑš[\”[œÊ
NÂˆÛÛœÝZ[Ýš[™ÈHMÑš[\”š[™ÑÛÜ™Ê
NÂˆÛÛœÝZ[Ý[œÈH[œÈH\Ý[œËš[™ÈHš[™ÈH\Ýš[™ÎÂˆ\Ý[œÈH[œÎÂˆ\Ýš[™ÈHš[™ÎÂˆÚ\ˆ\ÝÌM—NÂˆ[ˆHÂˆ›Üˆ
Z[Ì—ÝˆHÈˆÈŠÊÊBˆÂˆÛÛœÝZ[ÝÈHMÑš[\’\Ý
ŠNÂˆÛÛœÝZ[ÝHÈH\Ý\ÝØ—NÂˆ\Ý\ÝØ—HHÎÂˆˆ
ÏHÛœš[Š\Ý
È‹Ú^™[ÙŠ\Ý
HHÚ^™WÝ
ŠK‰\É[H‹ˆˆÈ‹Èˆˆˆ‹
[œÚYÛ™YÛ™ÈÛ™ÊY
NÂˆBˆYˆ
\\ÖÌ—H[œÊBˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HMš[\Žˆ	[HÛÜ™È[ˆ	[H[œÈ
YX[ˆ	KŒY‹‚ˆ‰KŒ‰IHš[™ÊH[œÈžH[™ÝKÌ‹ÍÎÌM‹ÌÌ‹ÍÌLŽ
Îˆ	\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY\\ÖÌ—K
[œÚYÛ™YÛ™ÈÛ™ÊY[œËˆ[œÈÈÝX›J\\ÖÌ—JHÈÝX›J[œÊHˆŒˆ\\ÖÌ—HÈLŒ
ˆÝX›Jš[™ÊHÈÝX›J\\ÖÌ—JHˆŒˆ\Ý
NÂˆB‚ˆËÈHÚY\‹Z\ÚY[[È
\Lˆ][HKŒ
KˆH][IÜÈÛZ[H\È]BˆËÈŒKLNHÚY\ˆØYÈHœ˜[YH™KZ\ÚH[™[Ùˆ\Ý[˜ÝÚY\œËÛÈBˆËÈUUH\ÈHÛZ[K›ÝHÚYH›ÝNˆ™[ÝÈŽL	HHY[[È\È˜\Ú[™ÂˆËÈ[™Hœ˜[YK][YH™\Ý[Ú[™H›Ú\ÙKˆ]šXÝ[ÛœÈÙ\\˜]HHÛÈØ^\ÈBˆËÈÝÈ˜]HØ[ˆ\[ˆ8 %HÛÜšÚ[™ÈÙ]\™Ù\ˆ[ˆHX›KÜˆHÝY\Ý]ˆËÈÙ[Z[™[HÙY\È\ØY[™È™]ÈZXÜ›ØÛÙKˆš[Y[˜ÛÛ™][Û˜[H™XØ]\ÙHBˆËÈ[X™\ˆ›Ø›ÙHš[È\ÈH[X™\ˆ›Ø›ÙHÚXÚÜÈ
\LIÜÈÝ]ÚYX
K‚ˆÂˆÝ]XÈZ[Ý\Ý]ÈH\ÝZ\ÜÙ\ÈH\Ý]šXÝH\ÝÛÛHÂˆÛÛœÝZ[Ý]ÈHMÔÚY\“Y[[Ò]Ê
NÂˆÛÛœÝZ[ÝZ\ÜÙ\ÈHMÔÚY\“Y[[ÓZ\ÜÙ\Ê
NÂˆÛÛœÝZ[Ý]šXÝHMÔÚY\“Y[[Ñ]šXÝ[ÛœÊ
NÂˆÛÛœÝZ[Ý]ÈH]ÈH\Ý]ËZ\ÜÙ\ÈHZ\ÜÙ\ÈH\ÝZ\ÜÙ\ÎÂˆÛÛœÝZ[Ý]šXÝH]šXÝH\Ý]šXÝÂˆÛÛœÝZ[ÝÛÛHMÔÚY\“Y[[ÐÛÛ\Ú[ÛœÊ
NÂˆÛÛœÝZ[ÝÛÛHÛÛH\ÝÛÛÂˆ\ÝÛÛHÛÛÂˆ\Ý]ÈH]ÎÂˆ\ÝZ\ÜÙ\ÈHZ\ÜÙ\ÎÂˆ\Ý]šXÝH]šXÝÂˆYˆ
]ÈZ\ÜÙ\ÊBˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÚY\ˆY[[Îˆ	KŒY‰IH]
	[H]È	[HZ\ÜË‚ˆ‰[H]šXÝY	[HÙˆHZ\ÜÙ\ÈHÛÛ\Ú[ÛŠH‚ˆ‰[HØYËÙœ˜[YI\×ˆ‹ˆLŒ
ˆÝX›J]ÊHÈÝX›J]È
ÈZ\ÜÙ\ÊKˆ
[œÚYÛ™YÛ™ÈÛ™ÊY]Ë
[œÚYÛ™YÛ™ÈÛ™ÊYZ\ÜÙ\Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊY]šXÝ
[œÚYÛ™YÛ™ÈÛ™ÊYÛÛˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈ
]È
ÈZ\ÜÙ\ÊHÈœ˜[Y\Èˆ
KˆMÔÚY\“Y[[ÓZ\ÛX]Ú\Ê
BˆÈˆ
ŠŠˆQSSÈRTÓPUÒTËÙYHÜMHX›Ý™H
ŠŠˆ‚ˆˆˆŠNÂˆB‚ˆËÈHTSSÕPT‘	ÔÈUUH
\LÈ][HKŒJKˆH[ˆÛ\È\ˆËÈÈ‘KT‘QÒTÕTˆ]ˆ™[ÝÈŽ	HÙ\™YH][H\È›ÝÛÜšÚ[™È[™[žBˆËÈœ˜[YK][YH[X™\ˆZÙ[ˆœ›ÛHH[ˆ\È›Ú\ÙKˆ˜Z[˜\ÈHÝ\ˆ[‚ˆËÈÙˆHÝÜžH8 %HÛÛÝ[\Ú[™ÈÚ[ˆH™^œ˜[YHÝØ\È\ÈHÛÛˆËÈ]\ÈÛÈÛX[[™H[\^\È]ØZ]‚ˆYˆ
ÝX\™ÛÛÛÜšÙ\œÊ
H	‰ˆ×ÙÜÝ]Ëœ™\]Y\ÝÊBˆÂˆÛÛœÝÝX\™ÛÛÝ]ÉˆÈH×ÙÜÝ]ÎÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝX\™™Z\Úˆ	KŒY‰IHÙ\™Y
	[HÙˆ	[K	KŒYˆ‚ˆ“P‹Ùœ˜[YH[Ý™YÙ™ˆH[\
HZ\ÜÎˆ[šÛ›ÝÛˆ	[H[™[™È	[H‚ˆ˜\šX[	[H	]HÛÜšÙ\œË	[H\Ü]Ú\Ë	[H›ØÚÙY‚ˆŠ	KŒ™ˆ\ÈÝ[
I\É\×ˆ‹ˆLŒ
ˆÝX›JËœÙ\™Y
HÈÝX›JËœ™\]Y\ÝÊKˆ
[œÚYÛ™YÛ™ÈÛ™Ê\ËœÙ\™Y
[œÚYÛ™YÛ™ÈÛ™Ê\Ëœ™\]Y\ÝËˆœ˜[Y\ÈÈÝX›JË˜ž]\ÔÙ\™Y
HÈÝX›Jœ˜[Y\ÊHÈLMÍ‹ŒˆŒˆ
[œÚYÛ™YÛ™ÈÛ™Ê\Ë›Z\ÜÕ[šÛ›ÝÛ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\Ë›Z\ÜÔ[™[™Ëˆ
[œÚYÛ™YÛ™ÈÛ™Ê\Ë›Z\ÜÕ˜\šX[ÝX\™ÛÛÛÜšÙ\œÊ
Kˆ
[œÚYÛ™YÛ™ÈÛ™Ê\Ë™\Ü]Ú\Ëˆ
[œÚYÛ™YÛ™ÈÛ™Ê\Ë™˜Z[›ØÚÙYˆÝX›JË™˜Z[“œÊHÈYM‹ˆË›Z^\ÈÈˆ
ŠŠˆÓÕRVUTËÙYHÝš×HX›Ý™H
ŠŠˆˆˆˆ‹ˆË™\šYžPÚXÚÙYÈˆˆˆˆŠNÂˆËÈHÓÓ	ÔÈÐÐÕTSÖH8 %\HÝ\ËˆH\Ü]ÚØ›ØÚÙYÛÝ[ÂˆËÈX›Ý™HØ^HHÛÛÑQTÈTÈÛ›H\ÈØ^\ÈÝÈ]XÚÛÜšÙ\‹][YH\ÂˆËÈYÝ™\ˆ›ÜˆH™XÛÜ™ÛÛÈÚ\™KˆÚ[™ÝÙYZÙH]™\žH˜]H\™BˆËÈ
ÛÝÚHŽ
NˆHÝ[][]]™HYX[ˆÛÝ[›[™H›ÛÝ	ÜÈ[\Hœ˜[Y\ÂˆËÈ[ÈHÜ›ÝÙ	ÜË‚ˆYˆ
×ÙÜ
BˆÂˆÝ]XÈZ[Ý\Ý\ÞHHÂˆÛÛœÝZ[Ý\ÞHH×ÙÜO˜\ÞSœË›ØY
ÝŽ›Y[[ÜžWÛÜ™\—Ü™[^Y
NÂˆÛÛœÝZ[Ý\ÞHH\ÞHH\Ý\ÞNÂˆ\Ý\ÞHH\ÞNÂˆÛÛœÝÝX›HÛÛœÈH
ˆYNH
ˆÝX›JÝX\™ÛÛÛÜšÙ\œÊ
JNÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝX\™ÛÛØØÝ\[˜ÞNˆ	KŒ™‰IH\ÞH
	KŒYˆ\ÈÙˆ‚ˆÛÜšÈXÜ›ÜÜÈ	]HÛÜšÙ\œÈ[ˆH	KŒYˆÈÚ[™ÝÎÈH™\Ý\È‚ˆÛÜšÙ\‹][YHHÚ\™Y™XÛÜ™ÛÛÛÝ[ZÙJWˆ‹ˆÛÛœÈˆÈLŒ
ˆÝX›J\ÞJHÈÛÛœÈˆŒˆÝX›J\ÞJHÈYM‹ÝX\™ÛÛÛÜšÙ\œÊ
K
NÂˆBˆYˆ
Ë™\šYžPÚXÚÙY
Bˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝX\™™Z\Ú‘T’Q–Nˆ	[HÙˆ	[HÙ\™YÝX\™È‚ˆ™\ØYÜ™YYÚ][ˆ[›[™H\ÚZÙ[ˆ]H˜]È
	K‰IJH8 %‚ˆ]\ÈHÚY[™Y˜XÙK›ÝHÜ›Û™È[\[Y[][ÛŽÈH‚ˆœÛÝZ^]\ÛÝ[]™Hš[YX›Ý™H[œÝXYˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\Ë™\šYžTÝ[Kˆ
[œÚYÛ™YÛ™ÈÛ™Ê\Ë™\šYžPÚXÚÙYˆLŒ
ˆÝX›JË™\šYžTÝ[JHÈÝX›JË™\šYžPÚXÚÙY
JNÂˆ×ÙÜÝ]ÈHÝX\™ÛÛÝ]ÞßNÂˆB‚ˆËÈH›]Ý™X[HØXÚKˆÛÈ[X™\œÈ[™›Ý\™H™YYYˆ“Ð‘TÈTˆÓÒÕTˆËÈ\ÈHÛ›H[™È]Ø^\ÈHX›H\ÈXÝX[H›][ˆ˜XÝXÙH˜]\‚ˆËÈ[ˆ[ˆš[˜Ú\H8 %H˜Y\ÚÜˆHØY˜XÝÜˆYÛÈYÚ\›œÈ[™X\‚ˆËÈ›Øš[™È[ÈH[™X\ˆØØ[‹[™]ÛÝ[™\Ù[\ÈHÚ[™ÙHYˆËÈ›Ý[™È‹ÚXÚ\È[™\Ý[™ÝZ\ÚX›Hœ›ÛHHÜ›Û™È[ÜžKˆX›Ý™HŒ‹ŒBˆËÈX›H\ÈH›Ø›[KˆH™\šYžH[™H\ÈHÛÜœ™XÝ™\ÜÈ[ˆ[™š[ÂˆËÈÛ›HÚ[ˆH\›H\ÈÛ‹‚ˆËÈ‹‹˜[™]š[È[ˆHÓÓ•“Ó\›HÛËÚ]™\›ÈÛÚÝ\Ë™XØ]\ÙH[ˆ\›BˆËÈ]\ÈÚ[[\È[ˆ\›H›Ø›ÙHØ[ˆ[Ø\ÈÛˆ
ÛÝÚHMLJK‚ˆËÈHÛÛœÝ[Y[[ËˆUUHTÈHÓRSNˆH][IÜÈÚÛH\™Ý[Y[\ÂˆËÈ]HÝY\Ý\ÜÝY\ÈÙ]™\˜[˜]ÜÈ\ˆÛÛœÝ[\]K[™™[ÝÈŒÌ	BˆËÈ]\È›ÝÛÜ]Èš\ÚËˆš[Y[˜ÛÛ™][Û˜[HÛÈH[ˆ]Ù\È›ÝˆËÈ™Z]™H]Ø^HØ^\ÈÛÈ˜]\ˆ[ˆ™Z[™È\ÜÝ[YYË‚ˆYˆ
×ØÛÛœÝY[[Ò]È
È×ØÛÛœÝY[[ÓZ\ÜÙ\ÊBˆÂˆÛÛœÝZ[ÝÝH×ØÛÛœÝY[[Ò]È
È×ØÛÛœÝY[[ÓZ\ÜÙ\ÎÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÛÛœÝY[[Îˆ	KŒY‰IHÙ\™Y
	[HÙˆ	[H[‹XÛÜY\ÊK‚ˆ‰KŒYˆP‹Ùœ˜[YH“ÕÛÜYY	\×ˆ‹ˆLŒ
ˆÝX›J×ØÛÛœÝY[[Ò]ÊHÈÝX›JÝ
Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ØÛÛœÝY[[Ò]Ë
[œÚYÛ™YÛ™ÈÛ™Ê]Ýˆœ˜[Y\ÈÈÝX›J×ØÛÛœÝY[[Ò]ÊH
ˆM‹ŒÈÝX›Jœ˜[Y\ÊHÈLMÍ‹ŒˆˆŒˆ×ØÛÛœÝY[[ÓÙ™ˆÈˆÐÖ—Õ’×Ó“×ÐÓÓ”ÕÓQSSÎˆHÛÜH[œÈ]™\žH˜]×H‚ˆˆˆŠNÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÛÛœÝY[[ÈžH[Žˆ”È	KŒY‰IKÈ	KŒY‰IH
Ùˆ	[H˜]ÜÈ‚ˆ™XXÚ
Wˆ‹ˆŒŒ
ˆÝX›J×ØÛÛœÝY[[ÕœÒ]ÊHÈÝX›JÝ
KˆŒŒ
ˆÝX›J×ØÛÛœÝY[[ÔÒ]ÊHÈÝX›JÝ
Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊJÝÈŠJNÂˆ×ØÛÛœÝY[[ÕœÒ]ÈH×ØÛÛœÝY[[ÔÒ]ÈHÂˆYˆ
×ØÛÛœÝY[[ÐÚXÚÙY
Bˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÛÛœÝY[[È‘T’Q–Nˆ	[HÙˆ	[HÙ\™Y˜]ÜÈY‚ˆ˜ÛÛœÝ[È]\ØYÜ™YYÚ]Hœ™\ÚÛÜH
	K‰IJI\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ØÛÛœÝY[[ÔÝ[Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ØÛÛœÝY[[ÐÚXÚÙYˆLŒ
ˆÝX›J×ØÛÛœÝY[[ÔÝ[JHÈÝX›J×ØÛÛœÝY[[ÐÚXÚÙY
Kˆ×ØÛÛœÝY[[Õ™\šYžTÚ\ÛÛˆÈˆÔÒTÓÓ‘Qˆ\ÈUTÕ™H›Û‹^™\›×H‚ˆˆˆŠNÂˆ×ØÛÛœÝY[[Ò]ÈH×ØÛÛœÝY[[ÓZ\ÜÙ\ÈHÂˆ×ØÛÛœÝY[[ÐÚXÚÙYH×ØÛÛœÝY[[ÔÝ[HHÂˆBˆYˆ
×Ù›]ØXÚSÛÚÝ\È×Ù›]ØXÚSÙ™ŠBˆÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—H›]Ý™X[HØXÚNˆ	[HÛÚÝ\ËÙœ˜[YK	KŒ™ˆ›Ø™\È\ˆ‚ˆ›ÛÚÝ\	\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈ×Ù›]ØXÚSÛÚÝ\ÈÈœ˜[Y\Èˆ
Kˆ×Ù›]ØXÚSÛÚÝ\ÈÈÝX›J×Ù›]ØXÚT›Ø™\ÊHÈÝX›J×Ù›]ØXÚSÛÚÝ\ÊHˆŒˆ×Ù›]ØXÚSÙ™ˆÈˆÐÖ—Õ’×Ó“×Ñ“UÐÐPÒNˆHÝŽ[›Ü™\™YÛX\\ÈÙ\š[™×HˆˆˆŠNÂˆYˆ
×Ù›]Ü›ÝÜÊBˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—H›]ØXÚHÜ›ÝÜÎˆ	[K	KŒ™ˆ\ÈÝ[ÛÜœÝ‚ˆ‰KŒ™ˆ\È8 %HÚÛH•S‹›Ý\ÈÚ[™Ý×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ù›]Ü›ÝÜËÝX›J×Ù›]Ü›ÝÓœÊHÈYM‹ˆÝX›J×Ù›]Ü›ÝÕÛÜœÝœÊHÈYMŠNÂˆ×Ù›]ØXÚSÛÚÝ\ÈHÂˆ×Ù›]ØXÚT›Ø™\ÈHÂˆBˆYˆ
×Ù›]ØXÚPÚXÚÙY
BˆÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—H›]ØXÚH‘T’Q–Nˆ	[HÙˆ	[HÛÚÝ\È\ØYÜ™YYÚ]‚ˆHÝŽ[›Ü™\™YÛX\
	K‰IJI\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ù›]ØXÚQ\ØYÜ™YYˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ù›]ØXÚPÚXÚÙYˆLŒ
ˆÝX›J×Ù›]ØXÚQ\ØYÜ™YY
HÈÝX›J×Ù›]ØXÚPÚXÚÙY
Kˆ×Ù›]ØXÚU™\šYžTÚ\ÛÛˆÈˆÔÒTÓÓ‘Qˆ\ÈUTÕ™H›Û‹^™\›×HˆˆˆŠNÂˆ×Ù›]ØXÚPÚXÚÙYHÂˆ×Ù›]ØXÚQ\ØYÜ™YYHÂˆB‚ˆËÈÛÛXÝÛÜžHÛÝ[\ØÙ[™[™Ëš[ˆLŽÛÝÈ\ÈHš^Y[žBˆËÈ\œ˜^NÈŒIÜÈÙ[œÝ\ÈØ^\È\È]H\Ù\ÈŒHÜÛÙ\ËÛÈ\È\È][ÜÝˆËÈ›Ý\ˆ[™\È[™\ÝX[H™YK‚ˆÝXÝÜÙ[œÝ\ÈÈZ[Ì—ÝÜÈZ[ÝÛÝ[ÈNÂˆÜÙ[œÝ\ÈÝÌLŽNÂˆZ[Ì—Ý’ÝHÂˆ›Üˆ
Z[Ì—ÝÜHÈÜLŽÈÜ
ÊÊBˆÂˆÛÛœÝZ[ÝÈHMÓÜÛÙPÛÝ[
Ü
NÂˆÛÛœÝZ[ÝHÈH\ÝÜÛÙ\ÖÛÜNÂˆ\ÝÜÛÙ\ÖÛÜHHÎÂˆYˆ

BˆÝÛ’Ý
Ê×HHÈÜNÂˆBˆÝŽœÛÜ
ÝÝ
È’Ýˆ×JÛÛœÝÜÙ[œÝ\ÉˆKÛÛœÝÜÙ[œÝ\ÉˆŠHÈ™]\›ˆK˜ÛÝ[ˆ‹˜ÛÝ[ÈJNÂˆ›Üˆ
Z[Ì—ÝHHÈH’ÝÈH
ÏHJBˆÂˆÚ\ˆ[™VÍLL—NÂˆ[ˆHÛœš[Š[™KÚ^™[ÙŠ[™JK–ÝšÜ›Ù—HM	\È‹ˆHÈˆˆˆ›ÜÛÙ\ÎˆŠNÂˆ›Üˆ
Z[Ì—ÝˆHNÈˆ’Ý	‰ˆˆH
ÈNÈŠÊÊBˆÂˆËÈ[ˆ[›˜[YYÜÛÙH\È™\ÜYžH[™^›ÝÚÚ\YˆHØ[ÂˆËÈ[™XYH™X]ÈÛ™H\ÈH™\ÜX›H[›ÛX[H
H\œÙ\ˆ\Þ[˜ÈÜˆBˆËÈXÚÙ]HØ\\™\È™]™\ˆ[
K[™HÙ[œÝ\È]Ú[[BˆËÈ›ÜY]ÛÝ[[œÝÙ\ˆÚXÚXÚÙ]ÈˆÚ]HÛÛ™šY[ÝXœÙ]‚ˆÛÛœÝÚ\Šˆ˜[YHHMÓÜÛÙS˜[YJÝÚ—K›Ü
NÂˆÚ\ˆ˜[YYÌNÂˆYˆ
[˜[YJBˆÂˆÛœš[Š˜[YYÚ^™[ÙŠ˜[YY
K•S’Ó“ÕÓ—ÉL–‹ÝÚ—K›Ü
NÂˆ˜[YHH˜[YYÂˆBˆˆ
ÏHÛœš[Š[™H
È‹Ú^™[ÙŠ[™JHH‹‰\È	[KÙˆ
	KŒY‰IJH‹ˆ˜[YKˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈÝÚ—K˜ÛÝ[Èœ˜[Y\Èˆ
KˆXÚÙ]Ý
ÝÚ—K˜ÛÝ[
JNÂˆBˆœš[ŠÝ\œ‹‰\×ˆ‹[™JNÂˆBˆ\Ý[\HÂ‚ˆËÈHÝ™X[HØXÚKÚ[ˆ\ÚÙY›Ü‹ˆš[Y[œÚYHH›Ùš[HÚ[™ÝÈÛÈBˆËÈ˜]\È\™H\‹Yœ˜[YHÝ™\ˆHÐSQHœ˜[Y\ÈHÝ™X[\Ø\˜Ù[YÙHX›Ý™H\ÂˆËÈ]™\˜YÙYÝ™\ˆ8 %HÙ[œÝ\ÈÛÝ[YÝ™\ˆHÚÛH[ˆ[™H\˜Ù[YÙBˆËÈÛÝ[YÝ™\ˆš]™HÙXÛÛ™ÈØ[››Ý™H]šYY[ÈXXÚÝ\‹ˆH]š\ÛÜˆ\ÂˆËÈ‘TÑS•Qœ˜[Y\ËZÙH]™\žHÝ\ˆ˜]HÛˆ\ÙH[™\Ë›Ýœ˜[Y\È]ˆËÈ™XÛÜ™YH˜]ÎÈHœ˜[YHÚ]›È˜]ÜÈ™]™\ˆØ[È™YÚ[‘œ˜[YH][‚ˆËÈHÜ›ÜÜËYœ˜[YHÝÜ™KSÐVTÈ8 %›Ý™Z[™HÙ[œÝ\Ë™XØ]\ÙH\È\ÈBˆËÈ[™H]Ø^\ÈÚ]\ˆH[™È\ÈÛÜšÚ[™È[™Ú]\ˆ]\ÈÙ\š[™ÈÝ[BˆËÈ]K[™HÛÝ[\ˆ›Ø›ÙHÛÚÜÈ]žHY˜][\ÈHÛÝ[\ˆ]™\ÜÈBˆËÈÚ[[™YÜ™\ÜÚ[ÛˆÈ›Ø›ÙH
ÛÝÚHMLJKˆ˜]\È\™H\ˆ‘TÑS•Qœ˜[YKˆËÈZÙH]™\ž][™È[ÙHÛˆ\ÙH[™\Ë‚ˆYˆ
‹Oœ\œÚ\ÝÛŠBˆÂˆÛÛœÝ™[™\™\ŽŽ”\œÚ\ÝÝ]ÉˆH‹Oœ\œÚ\ÝÝ]ÎÂˆÛÛœÝZ[ÝÝXÚYBˆš]È
È™š[È
ÈœÝ[H
È›Ý™\™›ÝÈ
ÈœÝ[Q]šXÝYÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝÜ™H	[Hš\œÝ]ÝXÚÙœ˜[YNˆ	KŒY‰IHÙ\™YXÜ›ÜÜÈH‚ˆ™œ˜[YH›Ý[™\žK	KŒ™ˆP‹Ùœ˜[YH“ÕÛÜYYš[È	[HÝ[H	[H‚ˆŠ	[H]šXÝY›ÈÚ[ŠHÝ™\™›ÝÈ	[HÝX\™™XY	KŒ™ˆP‹Ùœ˜[YWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈÝXÚYÈœ˜[Y\Èˆ
KˆÝXÚYÈLŒ
ˆÝX›Jš]ÊHÈÝX›JÝXÚY
HˆŒˆœ˜[Y\ÈÈÝX›Jš]ž]\ÊHÈÝX›Jœ˜[Y\ÊHÈLMÍ‹ŒˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈ™š[ÈÈœ˜[Y\Èˆ
Kˆ
[œÚYÛ™YÛ™ÈÛ™Ê\œÝ[Kˆ
[œÚYÛ™YÛ™ÈÛ™Ê\œÝ[Q]šXÝYˆ
[œÚYÛ™YÛ™ÈÛ™Ê\›Ý™\™›ÝËˆœ˜[Y\ÈÈÝX›J™ÝX\™ž]\ÊHÈÝX›Jœ˜[Y\ÊHÈLMÍ‹ŒˆŒ
NÂˆËÈH^ÜÝ\™HH›Ý[™X]™\È™Z[™ˆÝ™X[\ÈÛÈ\™ÙHÈ\ÚˆËÈ^XÝKÚXÚ\™H\™Y›Ü™HÛ›HØ[\Y[™ÐSˆYHHÛX[Y]‚ˆËÈ\È\ÈHÜ[][Ûˆ][HÉÜÈY™XÝ]™Y[‹ÛÈ]\È™\ÜYˆËÈ˜]\ˆ[ˆ\ÜÝ[YYÈ™H[\K‚ˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝX\™^XÝÈ	^HŽÈ	[HÝ™X[\ËÙœ˜[YH^ÙYYY]‚ˆ˜[™Ù\™HÐSTQ
HÛX[Y][œÚYHÛ™HÙˆ\ÙH\È[š\ÚX›H‚ˆ‹KH][HÊWˆ‹ˆ×ÙÝX\™ž]\Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈ×ÙÝX\™Ø[\YÈœ˜[Y\Èˆ
JNÂˆ×ÙÝX\™Ø[\YHÂˆËÈ‹‹˜[™HÒV‘H\ÝšX][ÛˆÙˆ]Ü[][Û‹™XØ]\ÙHHÛÝ[ˆËÈØ[››ÝÚÛÜÙHH›Ý[™[™HÛÜÝÙˆ˜Z\Ú[™ÈÛ™H\È[ˆHž]\Ë‚ˆÂˆÝ]XÈÛÛœÝÚ\ŠˆÓ˜[Y\ÖÚÑÝX\™\ÝXÚÙ]×HBˆÈŒM‹LÌ’È‹ŒÌ‹MÈ‹LLŽÈ‹ŒLŽLM’È‹ŒM‹MLL’È‹ˆLL’ËLSH‹ŒKL“H‹Œ“HˆNÂˆÚ\ˆ[™VÍLL—NÂˆ[]HÛœš[Š[™KÚ^™[Ùˆ[™Kˆ–ÝšÜ›Ù—HØ[\Y\Ý™X[HÚ^™\È
˜Z\ÙHH›Ý[™È‚ˆ˜ÛÝ™\ˆHXÚÙ][™[ÝH^H]ÈP‹Ùœ˜[YJNˆŠNÂˆ›Üˆ
Ú^™WÝˆHÈˆÑÝX\™\ÝXÚÙ]ÎÈ
ÊØŠBˆÂˆYˆ
Y×ÙÝX\™\ÝÛÝ[Ø—H]H[
Ú^™[Ùˆ[™JHH
BˆÛÛ[YNÂˆ]
ÏHÛœš[Š[™H
È]Ú^™[Ùˆ[™HH]ˆ	\ÏI[KÉLŒY“Pˆ‹ˆÓ˜[Y\ÖØ—Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈ×ÙÝX\™\ÝÛÝ[Ø—HÈœ˜[Y\Âˆˆ×ÙÝX\™\ÝÛÝ[Ø—JKˆœ˜[Y\ÈÈÝX›J×ÙÝX\™\Ýž]\ÖØ—JHÈÝX›Jœ˜[Y\ÊBˆÈLMÍ‹ŒˆˆŒ
NÂˆ×ÙÝX\™\ÝÛÝ[Ø—HHÂˆ×ÙÝX\™\Ýž]\ÖØ—HHÂˆBˆœš[ŠÝ\œ‹‰\×ˆ‹[™JNÂˆBˆËÈÚ]H[˜[ZXË\Ý™X[H›Û[Ý[ÛˆXÝX[HÛÜÝËˆ[ˆY\]™HÛXÞBˆËÈÚÜÙHšXÙH\È[šÛ›ÝÛˆ\ÈH[™È]Ù\][HÈ\šÙY‚ˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝX\™“ÓSÕQÈ^XÝ
Ý™X[HÙY[ˆÚ[™Ú[™ÊNˆ‚ˆ‰[KÙœ˜[YK	KŒYˆP‹Ùœ˜[YWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈ×ÙÝX\™[˜[ZXÈÈœ˜[Y\Èˆ
Kˆœ˜[Y\ÈÈÝX›J×ÙÝX\™[˜[ZXÐž]\ÊHÈÝX›Jœ˜[Y\ÊHÈLMÍ‹ŒˆˆŒ
NÂˆ×ÙÝX\™[˜[ZXÈHÂˆ×ÙÝX\™[˜[ZXÐž]\ÈHÂˆËÈ‹‹œÜ]žHHÓÔˆXXÚ›Û[Ý[ÛˆØ[YH›ÝYÚ™XØ]\ÙHH™YBˆËÈ]™HY™™\™[šXÙ\È[™Û›HÛ™HÙˆ[H\È›Ý[™YˆÙYHBˆËÈ×ÙÝX\™›Ý™[ˆXÛ\˜][Ûˆ›ÜˆÚ]\È\È\ÚÚ[™Ë‚ˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝX\™›Û[Ý[ÛˆžH™X\ÛÛŽˆ›Ý™[Š[˜YÙ]Y›Ü™]™\ŠH‚ˆ‰[KÙœ˜[YH	KŒYˆPˆÜXÝ[]]™JYÙ]Y
H	[KÙœ˜[YH	KŒYˆPˆ‚ˆœ›Ø™JYÙ]Y
H	[KÙœ˜[YH	KŒYˆPˆ	[HÙˆ	^H[šY\È]™H‚ˆ‘U‘Tˆ]ÚY›Ý™[ˆÙˆ›Ý™[ˆØœÙ\˜][ÛœÈ	KŒY‰IH›Ý[™H‚ˆœÝ™X[HPÕPSHÒS‘ÑQ
X›Ý™HL	IHHÝX\™ÛÜÝÈH™XYÈ‚ˆ›X\›ˆÚ]HÛÜHÛÝ[]™HÛ\Èœ™YJWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈ×ÙÝX\™›Ý™[ˆÈœ˜[Y\Èˆ
Kˆœ˜[Y\ÈÈÝX›J×ÙÝX\™›Ý™[ž]\ÊHÈÝX›Jœ˜[Y\ÊHÈLMÍ‹ŒˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈ×ÙÝX\™ÜXÈÈœ˜[Y\Èˆ
Kˆœ˜[Y\ÈÈÝX›J×ÙÝX\™ÜXÐž]\ÊHÈÝX›Jœ˜[Y\ÊHÈLMÍ‹ŒˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈ×ÙÝX\™›Ø™HÈœ˜[Y\Èˆ
Kˆœ˜[Y\ÈÈÝX›J×ÙÝX\™›Ø™Pž]\ÊHÈÝX›Jœ˜[Y\ÊHÈLMÍ‹ŒˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙÝX\™›Ý™[‘[šY\Ë\œÚ\ÝÚ^™J
Kˆ×ÙÝX\™›Ý™[“ØœÈÈLŒ
ˆÝX›J×ÙÝX\™›Ý™[Ú[™ÙY
HÂˆÝX›J×ÙÝX\™›Ý™[“ØœÊHˆŒ
NÂˆ×ÙÝX\™›Ý™[“ØœÈH×ÙÝX\™›Ý™[Ú[™ÙYHÂˆ×ÙÝX\™›Ý™[ˆH×ÙÝX\™›Ý™[ž]\ÈHÂˆ×ÙÝX\™ÜXÈH×ÙÝX\™ÜXÐž]\ÈHÂˆ×ÙÝX\™›Ø™HH×ÙÝX\™›Ø™Pž]\ÈHÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝÜ™H	^H[šY\Ë	[HPˆÙˆ	[HPˆ\ÙY	[H›\Ú\È‚ˆˆ\ÈÚ[™Ý×ˆ‹ˆ\œÚ\ÝÚ^™J
Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ‹Oœ\œÚ\ÝÝ\œÛÜˆˆŒ
Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ‹Oœ\œÚ\ÝœÚ^™HˆŒ
Kˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™›\Ú\ÊNÂˆYˆ
‹Oœ\œÚ\Ý]‹˜Y™™\ŠBˆÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝÜ™HZ\œ›ÜŽˆ	KŒ™ˆP‹Ùœ˜[YHÛÜYYÜÝO•”SH[ˆ‚ˆ‰KŒYˆÛÜY\ËÙœ˜[YNÈ]È›Ý[™HRT”“Ôˆ	KŒY‰IHÙˆH[YH‚ˆŠ	[H]‹	[HÜÝ
H\ÈÚ[™Ý×ˆ‹ˆœ˜[Y\ÈÈÝX›J‹O›Z\œ›Üž]\ÊHÈÝX›Jœ˜[Y\ÊHÈLMÍ‹ŒˆŒˆœ˜[Y\ÈÈÝX›J‹O›Z\œ›ÜÛÜY\ÊHÈÝX›Jœ˜[Y\ÊHˆŒˆ
‹O›Z\œ›Ü’]Ñ]ˆ
È‹O›Z\œ›Ü’]ÒÜÝ
BˆÈLŒ
ˆÝX›J‹O›Z\œ›Ü’]Ñ]ŠHÂˆÝX›J‹O›Z\œ›Ü’]Ñ]ˆ
È‹O›Z\œ›Ü’]ÒÜÝ
BˆˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O›Z\œ›Ü’]Ñ]‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O›Z\œ›Ü’]ÒÜÝ
NÂˆ‹O›Z\œ›ÜÛÜY\ÈH‹O›Z\œ›Üž]\ÈHÂˆ‹O›Z\œ›Ü’]Ñ]ˆH‹O›Z\œ›Ü’]ÒÜÝHÂˆBˆ‹Oœ\œÚ\ÝÝ]ÈH™[™\™\ŽŽ”\œÚ\ÝÝ]ÞßNÂˆBˆYˆ
×ÜÝ™X[PÙ[œÝ\ÊBˆÂˆÛÛœÝÝ™X[PÙ[œÝ\ÉˆÈH×ÜÝ™X[PÙ[œÝ\×ØÎÂˆÛÛœÝZ[ÝˆHËš]È
ÈË›Z\ÜÙ\ÎÂˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝ™X[\È	[HÛÚÝ\ËÙœ˜[YNˆ	KŒY‰IH]ÛÜYY‚ˆ‰KŒ™ˆP‹Ùœ˜[YH
	[HZ\ÜÙ\ËÙœ˜[YK	[HˆXXÚ
H]ÈØ]™Y‚ˆ‰KŒ™ˆP‹Ùœ˜[YWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈˆÈœ˜[Y\Èˆ
KˆˆÈLŒ
ˆÝX›JËš]ÊHÈÝX›JŠHˆŒˆœ˜[Y\ÈÈÝX›JË˜ž]\ÐÛÜYY
HÈÝX›Jœ˜[Y\ÊHÈLMÍ‹ŒˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈË›Z\ÜÙ\ÈÈœ˜[Y\Èˆ
KˆË›Z\ÜÙ\ÈÈ
[œÚYÛ™YÛ™ÈÛ™ÊJË˜ž]\ÐÛÜYYÈË›Z\ÜÙ\ÊHˆ[ˆœ˜[Y\ÈÈÝX›JË˜ž]\Ò]
HÈÝX›Jœ˜[Y\ÊHÈLMÍ‹ŒˆŒ
NÂˆËÈÚ]HØXÚH]Ý\š]™YHœ˜[YH›Ý[™\žHÛÝ[]™HÛ™KˆBˆËÈÙXÛÛ™[™HÛ›H\X\œÈ]]™[‹™XØ]\ÙHÛ›H]™[ˆÛ›ÝÜÈÚ]\‚ˆËÈ]ÛÝ[]™H™Y[ˆÓÔ”‘PÕÈÙ\™H]‚ˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝ™X[\ÈÜ›ÜÜËYœ˜[YNˆ	KŒY‰IHÙˆZ\ÜÙ\È™\X]\Ý‚ˆ™œ˜[YIÜÈÙ^H
	KŒ™ˆP‹Ùœ˜[YJI\×ˆ‹ˆË›Z\ÜÙ\ÈÈLŒ
ˆÝX›JËœ™]‘œ˜[YRÙ^R]ÊHÈÝX›JË›Z\ÜÙ\ÊBˆˆŒˆœ˜[Y\ÈÈÝX›JËœ™]‘œ˜[YRÙ^Pž]\ÊHÈÝX›Jœ˜[Y\ÊHÈLMÍ‹ŒˆˆŒˆ×ÜÝ™X[PÙ[œÝ\ÈHˆÈˆˆˆˆÛ]™[ˆ›ÜˆHÛÛ[ÚXÚ×HŠNÂˆÝ]XÈÛÛœÝÚ\ŠˆÚ[™˜[YVÌ×HHÈ™\^‹š[™^‹™™]ÚˆNÂˆ›Üˆ
[ÈHÈÈÎÈ
ÊÚÊBˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝ™X[\È	\Îˆ	[HZ\ÜÙ\ËÙœ˜[YK	KŒ™ˆP‹Ùœ˜[YH‚ˆ˜ÛÜYY	KŒY‰IHÙˆÜÙHž]\È™\X]\Ýœ˜[YWˆ‹ˆÚ[™˜[YVÚ×Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊJœ˜[Y\ÈÈËšÚ[™Z\ÜÙ\ÖÚ×HÈœ˜[Y\Èˆ
Kˆœ˜[Y\ÈÈÝX›JËšÚ[™ž]\ÖÚ×JHÈÝX›Jœ˜[Y\ÊHÈLMÍ‹ŒˆˆŒˆËšÚ[™ž]\ÖÚ×HÈLŒ
ˆÝX›JËšÚ[™™\X]ž]\ÖÚ×JHÂˆÝX›JËšÚ[™ž]\ÖÚ×JBˆˆŒ
NÂˆYˆ
×ÜÝ™X[PÙ[œÝ\ÈHŠBˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝ™X[\ÈÜ›ÜÜËYœ˜[YHÓÓ•S•SÒS‘ÑQˆ	[HÙˆ‚ˆ‰[H™\X]YÙ^\È
	KŒY‰IJK	KŒ™ˆP‹Ùœ˜[YH8 %H™\Ý\ÈH‚ˆ™ÝY\Ý™]Üš][™ÈHY™™\ˆ[ˆXÙWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\Ëœ™]‘œ˜[YTØ[YPÛÛ[ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\Ëœ™]‘œ˜[YRÙ^R]ËˆËœ™]‘œ˜[YRÙ^R]ÈÈLŒ
ˆÝX›JËœ™]‘œ˜[YTØ[YPÛÛ[
HÂˆÝX›JËœ™]‘œ˜[YRÙ^R]ÊBˆˆŒˆœ˜[Y\ÈÈÝX›JËœ™]‘œ˜[YTØ[YPž]\ÊHÈÝX›Jœ˜[Y\ÊHÂˆLMÍ‹ŒˆˆŒ
NÂˆËÈHÕPT‘	ÔÈÕÑT‹ˆH\œÚ\Ý[ÝÜ™HXÚY\ÈÝ[[™\ÜÈÚ]BˆËÈ›Ý[™YXÛÜÝš[™Ù\œš[
][ÜÝLLˆž]\Ë^XÝ™[ÝÈ]
NÈ\ÂˆËÈ[™H\ÈH[\ÚÚXÚÚ[™È]ÈÛÜšËˆ[ž][™È]™\›È\ÈHÝ[BˆËÈ™\^Y™™\ˆ[™YÈH˜]Ë[™\È\ÈHÓ“H[™È[ˆBˆËÈ[[YH]Ø[ˆÙYH]ˆ]™XYÈ™\›ÈÚ[ˆHÝÜ™H\ÈÙ™ˆÛÈ8 %›Ü‚ˆËÈHš]šX[™X\ÛÛˆ]›Ý[™ÈØ\ÈÙ\™YXÜ›ÜÜÈHœ˜[YH8 %ÛÈ™XY]ˆËÈ[Û™ÜÚYHHÝÜ™X[™HX›Ý™K™]™\ˆÛˆ]ÈÝÛ‹‚ˆYˆ
×ÜÝ™X[PÙ[œÝ\ÈHŠBˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝ™X[\ÈÕPT‘RTÔÑQˆ	[HÙˆ	[H™X[ÛÛ[‚ˆ˜Ú[™Ù\ÈÙ\™YÕSHžHHÜ›ÜÜËYœ˜[YHÝÜ™I\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\Ë™ÝX\™Z\ÜÙYˆ
[œÚYÛ™YÛ™ÈÛ™ÊJËœ™]‘œ˜[YRÙ^R]ÈBˆËœ™]‘œ˜[YTØ[YPÛÛ[
Kˆ×ÜÝ™X[TÚ\ÛÛ‚ˆÈˆÔÒTÓÓˆÓˆ8 %H[\ÚØ[È]™\žH™\X]H‚ˆ˜Ú[™ÙHÚ[HHÝX\™ÛÜœ™XÝHÙ\È›ÝÛÈ\È‚ˆ”ÒÕS\]X[H™\X]ÛÝ[H‚ˆˆˆŠNÂˆËÈ‹‹˜[™ÒPÒÛ™\Ë™XØ]\ÙH]\ÈÚ]XÚÜÈH[˜[Y][Û‚ˆËÈYXÚ[š\ÛKˆÝ[][]]™HÝ™\ˆH[‹ÛÈ\È\Ý\ÈH[œÝÙ\ˆÈš\ÂˆËÈH™]Üš][ˆÙ]H™XÝ\œš[™È™]Ë[™\™H^HÛÛYÝ[Ý\È[ˆÝY\ÝˆËÈY[[ÜžH‹ˆØ\Y]Ìˆ[™\ÈÚ]H™[XZ[™\ˆSQQ˜]\ˆ[‚ˆËÈ›ÜYÚ[[H
ÛÝÚHLJH8 %[™HÝY\ÝY™\ÜÈ˜[™ÙH\Èš[YˆËÈÚ]\ˆÜˆ›ÝH\Ý\ÈØ\YÚ[˜ÙHH˜[™ÙH\ÈH[™È[‚ˆËÈ^Û\Ú[Ûˆ[HÛÝ[™HÜš][ˆYØZ[œÝ‚ˆYˆ
×ÜÝ™X[PÙ[œÝ\ÈHˆ	‰ˆY×ÜÝ™X[PÚ[™ÙY™[\J
JBˆÂˆÝŽ™XÝÜÝŽœZ\Z[ÝÝ™X[PÚ[™ÙOˆŠˆ×ÜÝ™X[PÚ[™ÙY˜™YÚ[Š
K×ÜÝ™X[PÚ[™ÙY™[™

JNÂˆÝŽœÛÜ
‹˜™YÚ[Š
K‹™[™

K×JÛÛœÝ]]ÉˆKÛÛœÝ]]ÉˆŠHÂˆ™]\›ˆKœÙXÛÛ™[Y\Èˆ‹œÙXÛÛ™[Y\ÎÂˆJNÂˆZ[Ì—ÝÈH‘‘‘‘‘‘‘KHHÂˆZ[ÝÝ[HÂˆ›Üˆ
ÛÛœÝ]]ÉˆHˆŠBˆÂˆÛÛœÝZ[Ì—Ý˜HHZ[Ì—Ý
K™š\œÝˆÌŠNÂˆÈHÝŽ›Z[ŠË˜JNÂˆHHÝŽ›X^
KZ[Ì—Ý
˜H
ÈKœÙXÛÛ™˜ž]\ÊJNÂˆÝ[
ÏHKœÙXÛÛ™[Y\ÎÂˆBˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝ™X[\È‘UÔ’USˆSˆPÑNˆ	^H\Ý[˜ÝÙ^\Ë	[H‚ˆ›ØØÝ\œ™[˜Ù\ËÝY\Ý˜[™ÙH	L‹‰L	\×ˆ‹ˆ‹œÚ^™J
K
[œÚYÛ™YÛ™ÈÛ™Ê]Ý[ËKˆ×ÜÝ™X[TÚ\ÛÛˆÈˆÔÒTÓÓˆÓˆ8 %]™\žH™\X][™È\™HžH‚ˆ˜ÛÛœÝXÝ[ÛŽÈ\È\Ý\ÈYX[š[™Û\Ü×H‚ˆˆˆŠNÂˆÝ]XÈÛÛœÝÚ\ŠˆÚ[™˜[YL–Ì×HHÈ™\^‹š[™^‹™™]ÚˆNÂˆÛÛœÝÚ^™WÝÚÝÛˆHÝŽ›Z[Ú^™WÝŠ‹œÚ^™J
KÌŠNÂˆ›Üˆ
Ú^™WÝHHÈHÚÝÛŽÈ
ÊÚJBˆÂˆÛÛœÝZ[Ì—Ý˜HHZ[Ì—Ý
–ÚWK™š\œÝˆÌŠNÂˆÛÛœÝÝ™X[PÚ[™ÙIˆÈH–ÚWKœÙXÛÛ™Âˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝ™X[\È˜OILÚ^™OI[H[™X[I[H	\È‚ˆž	[Hœ˜[Y\È	[K‹‰[Wˆ‹ˆ˜K
[œÚYÛ™YÛ™ÈÛ™ÊXË˜ž]\Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ–ÚWK™š\œÝ	ˆÊKÚ[™˜[YL–ØËšÚ[™Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊXË[Y\Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊXË™š\œÝœ˜[YKˆ
[œÚYÛ™YÛ™ÈÛ™ÊXË›\Ýœ˜[YJNÂˆBˆYˆ
‹œÚ^™J
HˆÚÝÛŠBˆœš[ŠÝ\œ‹ˆ–ÝšÜ›Ù—HÝ™X[\È‹‹˜[™	^H[Ü™H\Ý[˜ÝÙ^\È›Ý‚ˆ›\ÝYˆ‹ˆ‹œÚ^™J
HHÚÝÛŠNÂˆBˆ×ÜÝ™X[PÙ[œÝ\×ØÈHÝ™X[PÙ[œÝ\ÞßNÂˆB‚ˆ×Ü›ÙˆH›Ùš[T\Ù\ÞßNÂˆ\ÝH›ÝÎÂˆ\Ýœ˜[YHH‹O™œ˜[YNÂˆBˆB‚ˆËÈHÛ˜\ÚÝ\È\ˆœ˜[YNˆHœ˜[YHÚÜÙH™\ÛÛ™HÚZ[ˆ™]™\ˆ™XXÚ\ÈHœ›ÛˆËÈY™™\ˆ]\Ý›Ý™\Ù[H™]š[Ý\Èœ˜[YIÜÈXÝ\™H\ÈYˆ]Ù\™H\ÈÛ™K‚ˆ‹Oš]™Qœ›ÛÛ˜\ÚÝH˜[ÙNÂˆ
›ÚY
]ÚYÂˆ
›ÚY
ZZYÚÂŸB‚ŸHËÈ˜[Y\ÜXÙB‚‹ËÈÙYHš×Ü™[™\™\‹š8 %HÙ][™ÜÈ[™[	ÜÈ™\ÛÛ][Ûˆ›ÝÈ
\Œ
K‚›ÚYšÔ™[™\™\—Ô™\]Y\Ý™[™\”ØØ[JZ[Ì—ÝØØ[JBžÂˆYˆ
ØØ[HHH	‰ˆØØ[HH
Bˆ×Ü™\ÔØØ[T[™[™ËœÝÜ™JØØ[KÝŽ›Y[[ÜžWÛÜ™\—Ü™[X\ÙJNÂŸB‚‹ËÈÙYHš×Ü™[™\™\‹š8 %H[™[	ÜÈTH™\ÜÈ
\LJK‚›ÚYšÔ™[™\™\—Ô™\]Y\Ý[\›˜[™\ÊZ[Ì—ÝËZ[Ì—Ý
BžÂˆYˆ
È	‰ˆ
Bˆ×Ü™\ÕÒ[™[™ËœÝÜ™J
Z[Ý
ÊHÌŠHÝŽ›Y[[ÜžWÛÜ™\—Ü™[X\ÙJNÂŸB‚›˜[Y\ÜXÙBžÂ‚‹ËÈ\HH[™[™È]™H™\ÛÛ][ÛˆÚ[™ÙK‘UÑQSˆœ˜[Y\ÈÛˆH[\™XY8 %B‹ËÈÛ›H[ÛY[HØØ[HX^H[Ý™K™XØ]\ÙH]™\žH^[[œÚYHHœ˜[YH\ÜÝ[Y\È]‹ËÈ\ÈÛÛœÝ[ˆHQSHZ\ˆ\È™XZ[\™NÈ]™\žH™\ÛÛ™HÛ˜\ÚÝ[™B‹ËÈ™[™\™YÝX™HX\Ø\œžHHØØ[H^HÙ\™HZ[][™™XZ[[\Ù[™\Â‹ËÈ^š[H›ÝYÚZ\ˆ^\Ý[™È™\Ú^™H]ÎÈH™XY˜XÚÈY™™\œÈÜ›ÝÈYˆB‹ËÈ™]Èœ˜[YH›ÈÛ™Ù\ˆš]È
^H™]™\ˆÚš[šÈ8 %Y[[ÜžH\ÈÚX\\ˆ[ˆ[›Ý\‚‹ËÈ™\Ú^™H]
K‚›ÚY\T[™[™Ô™[™\”ØØ[J
BžÂˆÛÛœÝZ[Ì—ÝØ[H×Ü™\ÔØØ[T[™[™Ë™^Ú[™ÙJÝŽ›Y[[ÜžWÛÜ™\—ØXÜWÜ™[
NÂˆÛÛœÝZ[ÝØ[ÒH×Ü™\ÕÒ[™[™Ë™^Ú[™ÙJÝŽ›Y[[ÜžWÛÜ™\—ØXÜWÜ™[
NÂˆZ[Ì—ÝØ[ÈHZ[Ì—Ý
Ø[ÒˆÌŠKØ[HZ[Ì—Ý
Ø[Ò
NÂˆYˆ

]Ø[	‰ˆ]Ø[Ò
HTŠBˆ™]\›ŽÂˆÂˆZ[Ì—ÝÝËÚÂˆ[\›˜[™\ÊÝËÚ
NÂˆYˆ
Ø[Ò	‰ˆØ[ÈOHÝÈ	‰ˆØ[OHÚ
BˆØ[ÈHØ[HÂˆÛÛœÝ›ÛÛØØ[PÚ[™ÙHHØ[OH	‰ˆØ[OH™\ÔØØ[J
NÂˆYˆ
\ØØ[PÚ[™ÙH	‰ˆ]Ø[ÊBˆ™]\›ŽÂˆBˆYˆ
×Ü™\ÔØØ[SØÚÙY
BˆÂˆœš[ŠÝ\œ‹–Ýš×H[\›˜[\™\ÛÛ][ÛˆÚ[™ÙH‘Q•TÑQˆÖ—Õ’×Ô‘TËÈ‚ˆÖ—Õ’×Ô‘T×ÔÐÐSH[ˆ]›Üˆ\È[ˆ
H‚ˆ›YX\Ý\™[Y[\›HÚ[œÈÝ™\ˆHY[JWˆŠNÂˆ™]\›ŽÂˆBˆšÑ]šXÙUØZ]YJ‹O™]šXÙJNÂ‚ˆËÈHœ˜[YXY™™\ˆÝÛœÈ™Y™\™[˜Ù\ÈÈHÛ[XYÙHšY]ÜÈ[™]\Ý™H\Ý›ÞYYˆËÈ™Y›Ü™HÜÙHšY]ÜËˆH™\XÙ[Y[\ÈÜ™X]YY\ˆH™]ÈQSHZ\ˆ™[ÝË‚ˆ\Ý›ÞPÛÛ\]Xš[]QY˜[T\ÜÊ
NÂ‚ˆ]]È\Ý›ÞR[XYÙHHÉ—J[XYÙIˆ[JHÂˆYˆ
[KšY]ÊBˆšÑ\Ý›ÞR[XYÙUšY]Ê‹O™]šXÙK[KšY]Ë[ŠNÂˆYˆ
[Kš[XYÙJBˆšÑ\Ý›ÞR[XYÙJ‹O™]šXÙK[Kš[XYÙK[ŠNÂˆYˆ
[K›Y[[ÜžJBˆšÑœ™YSY[[ÜžJ‹O™]šXÙK[K›Y[[ÜžK[ŠNÂˆ[HH[XYÙ^ßNÂˆNÂˆÛÛœÝZ[Ì—Ý™Y›Ü™HH™\ÔØØ[J
NÂˆËÈY™\œ™YÛX\œÈ]ÚYYØZ[œÝHÓ[XYÙ\ÉÈ^[ÎÈHÛÛ[^HÙ\™BˆËÈØÛÜYÈ\È™Z[™È\Ý›ÞYYÚ]H[XYÙ\Ë‚ˆ‹Oœ[™[™ÐÛX\œË˜ÛX\Š
NÂˆ\Ý›ÞR[XYÙJ‹O˜ÛÛÜŠNÂˆ\Ý›ÞR[XYÙJ‹O™\
NÂˆ\Ý›ÞR[XYÙJ‹O˜ÛÛÜ”™\ÛÛ™JNÂˆ\Ý›ÞR[XYÙJ‹O™\™\ÛÛ™JNÂˆËÈH^XÚ]È›Ü›H
\LNˆH[™[	ÜÈTH™\ÜÊHÚ[œÎÈHYØXÞBˆËÈ[YÙ\‹\ØØ[H›Ü›H™\Ù\™\ÈHÝ\œ™[\ÜXÝ]ÌŒ
œØØ[K‚ˆYˆ
Ø[ÊBˆÂˆ×Ú[\›˜[ËœÝÜ™JØ[È	ˆŒ]KÝŽ›Y[[ÜžWÛÜ™\—Ü™[^Y
NÂˆ×Ú[\›˜[œÝÜ™JØ[ÝŽ›Y[[ÜžWÛÜ™\—Ü™[^Y
NÂˆBˆ[ÙBˆÂˆZ[Ì—ÝËÂˆ[\›˜[™\ÊË
NÂˆÛÛœÝZ[Ì—ÝšHÌŒ
ˆØ[ÂˆÛÛœÝZ[Ì—ÝÈH
Z[Ì—Ý
Z[Ý
ÊH
ˆšÈ
JH	ˆŒ]NÂˆ×Ú[\›˜[ËœÝÜ™JËÝŽ›Y[[ÜžWÛÜ™\—Ü™[^Y
NÂˆ×Ú[\›˜[œÝÜ™JšÝŽ›Y[[ÜžWÛÜ™\—Ü™[^Y
NÂˆB‚ˆÛÛœÝZ[Ì—ÝY˜[RH‹O™Y˜[RZYÚÈËÈÝY\Ý›ÝÜËXÚYY]œš[™Ë]\ˆYˆ
PÜ™X]R[XYÙJ‹O˜ÛÛÜ‹”Ö
‹O\™Ù]ÚY
K”ÊY˜[R
Kˆ’×Ñ“Ô“PUÔŽÎŽNÕS“Ô“Kˆ’×ÒSPQÑWÕTÐQÑWÐÓÓÔ—ÐUPÒQS•Ð’U’×ÒSPQÑWÕTÐQÑWÕS”Ñ‘T—ÔÔ×Ð’Uˆ’×ÒSPQÑWÕTÐQÑWÕS”Ñ‘T—ÑÕÐ’U’×ÒSPQÑWÕTÐQÑWÔÐSTQÐ’Uˆ’×ÒSPQÑWÐTÔPÕÐÓÓÔ—Ð’U’×ÒSPQÑWÕ’QU×ÕTWÌ‘KKˆšÐÛÛ\Û™[X\[™ÞßKK˜[ÙK‹O›\ØXTØ[\\ÊHˆPÜ™X]R[XYÙJ‹O™\”Ö
‹O\™Ù]ÚY
K”ÊY˜[R
KˆY˜[Q\›Ü›X]

Kˆ’×ÒSPQÑWÕTÐQÑWÑTÔÕSÒSÐUPÒQS•Ð’Uˆ’×ÒSPQÑWÕTÐQÑWÕS”Ñ‘T—ÑÕÐ’Uˆ’×ÒSPQÑWÕTÐQÑWÕS”Ñ‘T—ÔÔ×Ð’Uˆ’×ÒSPQÑWÐTÔPÕÑTÐ’U’×ÒSPQÑWÐTÔPÕÔÕSÒSÐ’Uˆ’×ÒSPQÑWÕ’QU×ÕTWÌ‘KKšÐÛÛ\Û™[X\[™ÞßKK˜[ÙKˆ‹O›\ØXTØ[\\ÊHˆPÜ™X]QY˜[T™\ÛÛ™U\™Ù]Ê”Ö
‹O\™Ù]ÚY
K”ÊY˜[R
JJBˆÂˆËÈH™[™\™\ˆÚ]›ÈQSHÝ[™Z[ˆØ[››Ý˜]È][8 %Ø^HÛÈ[™ÝÜˆËÈ™YY[™È]˜]\ˆ[ˆÜ˜\ÚÛˆHš\œÝ\ÜË‚ˆœš[ŠÝ\œ‹–Ýš×HU‘H‘TÐÐSHRSQ]	]^8 %™[™\™\ˆ\ØX›Yˆ‹Ø[
NÂˆ×ØXÝ]™HH˜[ÙNÂˆ™]\›ŽÂˆBˆ˜[YR[XYÙJ‹O˜ÛÛÜ‹‘QSHÛÛÝ\ˆ	]^	]H‹”Ö
‹O\™Ù]ÚY
K”ÊY˜[R
JNÂˆ˜[YR[XYÙJ‹O™\‘QSH\	]^	]H‹”Ö
‹O\™Ù]ÚY
K”ÊY˜[R
JNÂˆYˆ
PÜ™X]PÛÛ\]Xš[]QY˜[T\ÜÊ
JBˆÂˆœš[ŠÝ\œ‹–Ýš×HU‘H‘TÐÐSHÛÝ[›Ý™XZ[H[Ø[ˆKŒHQSH\Ü×ˆŠNÂˆ×ØXÝ]™HH˜[ÙNÂˆ™]\›ŽÂˆB‚ˆËÈ›\ÚU‘T–HÛ˜\ÚÝ[™™[™\™YÝX™H“ÕË[œÚYHHÛ™H]šXÙHYH\ÂˆËÈÝÚ]Ú[™XYHZY›Ü‹ˆHš\œÝ™\œÚ[ÛˆY[HÈH^žH\‹Y[žBˆËÈ™XZ[]8 %ÚXÚØ[ÈšÑ]šXÙUØZ]YHTˆÓTÒÕ[™HÝ™Y]œ˜[YBˆËÈÛÈÞ™[œÈÙˆ]™HÛ˜\ÚÝËÛÈHÙXÛÛ™ÈY\ˆH™\ÛÛ][ÛˆÚ[™ÙBˆËÈÙ\™HHÝ]\ˆ™\Ý]˜[HÜ\˜]Üˆ™XY\ÈHœ™Y^™Kˆ\˜\ÙY[šY\ÂˆËÈ™XÜ™X]Hœ™\ÚÛˆZ\ˆ™^™\ÛÛ™HÚ]›È\\ˆÝ[Ë‚ˆÚ^™WÝ›\ÚYHÂˆ›Üˆ
]]ÉˆÚÙ^KÛ˜\Hˆ‹OœÛ˜\ÚÝÊBˆÂˆšÑ\Ý›ÞR[XYÙUšY]Ê‹O™]šXÙKÛ˜\š[XYÙKšY]Ë[ŠNÂˆšÑ\Ý›ÞR[XYÙJ‹O™]šXÙKÛ˜\š[XYÙKš[XYÙK[ŠNÂˆšÑœ™YSY[[ÜžJ‹O™]šXÙKÛ˜\š[XYÙK›Y[[ÜžK[ŠNÂˆ›Üˆ
]]ÉˆÜÚ^™KšY]×HˆÛ˜\šY]ÜÊBˆÂˆ
›ÚY
\Ú^™NÂˆšÑ\Ý›ÞR[XYÙUšY]Ê‹O™]šXÙKšY]Ëš[XYÙKšY]Ë[ŠNÂˆšÑ\Ý›ÞR[XYÙJ‹O™]šXÙKšY]Ëš[XYÙKš[XYÙK[ŠNÂˆšÑœ™YSY[[ÜžJ‹O™]šXÙKšY]Ëš[XYÙK›Y[[ÜžK[ŠNÂˆBˆ
ÊÙ›\ÚYÂˆBˆ‹OœÛ˜\ÚÝË˜ÛX\Š
NÂˆ^Ù[[\

NÂˆ›Üˆ
]]ÉˆÚÙ^KÝX™WHˆ‹O˜ÝX™TÛ˜\ÚÝÊBˆÂˆšÑ\Ý›ÞR[XYÙUšY]Ê‹O™]šXÙKÝX™Kš[XYÙKšY]Ë[ŠNÂˆšÑ\Ý›ÞR[XYÙJ‹O™]šXÙKÝX™Kš[XYÙKš[XYÙK[ŠNÂˆšÑœ™YSY[[ÜžJ‹O™]šXÙKÝX™Kš[XYÙK›Y[[ÜžK[ŠNÂˆ
ÊÙ›\ÚYÂˆBˆ‹O˜ÝX™TÛ˜\ÚÝË˜ÛX\Š
NÂˆYˆ
›\ÚY
Bˆœš[ŠÝ\œ‹–Ýš×H]™H™\ØØ[H›\ÚY	^HÛ˜\ÚÝÝ\™˜XÙ\È[ˆÛ™H‚ˆœÝ[ˆ‹›\ÚY
NÂ‚ˆÛÛœÝZ[Ý™YYBˆÝŽ›X^
Z[Ý
”Ö
‹O\™Ù]ÚY
JH
ˆ”Ê‹O\™Ù]ZYÚ
KˆZ[Ý
”Ö
MŠJH
ˆ”ÊL
JH
ˆÂˆ]]ÈÜ›ÝÐY™™\ˆHÉ—JY™™\‰ˆ‹ÛÛœÝÚ\ŠˆÚ]
HÂˆYˆ
‹œÚ^™HH™YY
Bˆ™]\›ŽÂˆšÑ\Ý›ÞPY™™\Š‹O™]šXÙK‹˜Y™™\‹[ŠNÂˆšÑœ™YSY[[ÜžJ‹O™]šXÙK‹›Y[[ÜžK[ŠNÂˆˆHY™™\žßNÂˆYˆ
PÜ™X]PY™™\Š‹™YY’×Ð•Q‘‘T—ÕTÐQÑWÕS”Ñ‘T—ÑÕÐ’Uˆ™XY˜XÚÓY[[ÜžT›ÜÊ
K˜[ÙJJBˆœš[ŠÝ\œ‹–Ýš×HU‘H‘TÐÐSNˆ	\È™YÜ›ÝÈRSQ8 %™XY˜XÚË\ÚYH‚ˆ™™X]\™\ÈÚ[[˜Ø]H]\ÈØØ[Wˆ‹Ú]
NÂˆNÂˆÜ›ÝÐY™™\Š‹Oœ™XY˜XÚËœÛ˜\ÚÝ™XY˜XÚÈŠNÂˆ›Üˆ
Z[Ì—ÝHHÈH‹O™œ˜[Y\Ò[‘›YÚÈ
ÊÚJBˆÜ›ÝÐY™™\Š‹O™œ˜[Y\ÖÚWKœ™\Ù[œ™\Ù[™XY˜XÚÈŠNÂˆ‹Oœ™\Ù[^[Ëœ™\Ú^™JÚ^™WÝ
”Ö
‹O\™Ù]ÚY
JH
ˆ”Ê‹O\™Ù]ZYÚ
H
ˆ
NÂ‚ˆ
›ÚY
X™Y›Ü™NÂˆÂˆZ[Ì—ÝËšÂˆ[\›˜[™\ÊËš
NÂˆœš[ŠÝ\œ‹–Ýš×H[\›˜[™\ÛÛ][ÛˆOˆ	]^	]HU‘H
QSHÝ[™Z[ˆ‚ˆ‰]^	]JNÈÛ˜\ÚÝÈ[™HÝX™HX\™XZ[^š[HÝ™\ˆH‚ˆ›™^œ˜[Y\×ˆ‹ˆËš”Ö
‹O\™Ù]ÚY
K”Ê‹O™Y˜[RZYÚ
JNÂˆËÈHÚ[™ÝÙYÚ[™ÝÈ›ÛÝÜÈH\YY™\ÛÛ][Ûˆ
\L
Kˆ[™YÈBˆËÈÚ[™ÝÈ™XYÚXÚÝÛœÈ]™\žHÑØ[È]XÛ[™\È
[™Ø^\ÈÛÊHÚ[ˆBˆËÈÚ[™ÝÈ\È[ØÜ™Y[‹X^[Z\ÙYÜˆ[›™YžHHYX\Ý\™[Y[˜\šXX›K‚ˆÜÝÕÚ[™ÝÑ›ÛÝÒ[\›˜[™\ÊËš
NÂˆBŸB‚ŸHËÈ˜[Y\ÜXÙB‚›ÚYšÔ™[™\™\—ÓÛ”ÝØ\
Z[Ý
ˆ˜\ÙKZ[Ì—Ýœ›ÛY™™\‹Z[Ì—ÝÚYˆZ[Ì—ÝZYÚ
BžÂˆYˆ
Y×ØXÝ]™H×ÙÙ[ÙJBˆ™]\›ŽÂˆËÈ“ÕH
\LJNˆH[™[™È[\›˜[\™\ÛÛ][ÛˆÚ[™ÙH\È\YY]HÔÙ‚ˆËÈ™YÚ[‘œ˜[YK›Ý\™Kˆ\™HÚ]ÈRQQ”SQH8 %Hœ˜[YIÜÈ˜]ÜÈ\™H™XÛÜ™Y]ˆËÈ›ÝÝX›Z]Y[™\Ý›ÞZ[™ÈHQSH[XYÙ\ÈH™XÛÜ™YX]][œÝX›Z]YˆËÈÛÛ[X[™Y™™\ˆ™Y™\™[˜Ù\È\ÈH^XÝ\MŒœ™Y^™HÚ\H
˜HØZ]ZYHÛ›BˆËÈÛÝ™\œÈÕP“RUQÛÜšÈ‹H™]\™Y[XYÙHÛÛ[Y[
Kˆ\ÈZYYœ˜[YHXÙ[Y[\ÂˆËÈHZÙ[H™X\ÛÛˆH]™H]œ›Þ™HHÜ\˜]Ü‰ÜÈXXÚ[™HÚXÙH[™Ü[ˆËÈÌH\È\šÙY‚ˆÔÝØ\[\
˜\ÙKœ›ÛY™™\‹ÚYZYÚ
NÂŸB‚‹ËÈKKHH\ÙHÈ™YYKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKB˜›ÛÛšÔ™[™\™\—ÑÑ[š]

BžÂˆÝ]XÈ›ÛÛšYYH˜[ÙKÚÈH˜[ÙNÂˆYˆ
šYY
Bˆ™]\›ˆÚÎÂˆšYYHYNÂˆYˆ
×ØXÝ]™JBˆÂˆËÈHM™YY[š]X[^™Yš\œÝ
Ö—Õ’ÑUÊKˆÙÙ˜]Ë˜Ü™Y\Ù\ÈBˆËÈÛÛXš[˜][Ûˆ™Y›Ü™HØ[[™È\™KÛÈ™XXÚ[™È\È\ÈHÚ\š[™ÈYË‚ˆœš[ŠÝ\œ‹–Ýš×HÑ™YY™Y\ÙYˆHM™YY[™XYHÝÛœÈH™[™\™\—ˆŠNÂˆ™]\›ˆ˜[ÙNÂˆBˆYˆ
R[š]ÛÛ[[ÛŠ
JBˆ™]\›ˆ˜[ÙNÂˆ×ÙÙ[ÙHHYNÂˆÚÈHYNÂˆœš[ŠÝ\œ‹–Ýš×H™[™\™\ˆ™YYˆÑ˜]ÈÙ\šXÙH
\ÙHÊWˆŠNÂˆ™]\›ˆYNÂŸB‚›ÚYšÔ™[™\™\—ÑÑ˜]ÊZ[Ý
ˆ˜\ÙKÛÛœÝM˜]Éˆ˜]ËÛÛœÝZ[Ì—Ý
ˆ™YÜËˆÛÛœÝMÚY\š[™[™ÉˆœËÛÛœÝMÚY\š[™[™ÉˆÊBžÂˆYˆ
Y×ØXÝ]™HY×ÙÙ[ÙJBˆ™]\›ŽÂˆËÈHØ[YH™\ÛÛ™H\ØÜš[Z[˜]Üˆ\ÈHM™YYÝ™\ˆH’UUH™YÚ\Ý\‚ˆËÈš[NˆHÛÜK[[ÙHÑUÐÓÓ”ÕS•ÈH™\ÛÛ™H›ÙH[Z]È[™\™K‚ˆYˆ

™YÜÖÌŒŒH	ˆÊHOHŠBˆÂˆÔ™\ÛÛ™J˜\ÙK™YÜÊNÂˆ™]\›ŽÂˆBˆÑ˜]Ê˜\ÙK˜]Ë™YÜËœËÊNÂŸB‚›ÚYšÔ™[™\™\—ÑÑÝØ\
Z[Ý
ˆ˜\ÙJBžÂˆYˆ
Y×ØXÝ]™HY×ÙÙ[ÙJBˆ™]\›ŽÂˆËÈHœ›ÛY™™\ˆ\ÈH\Ý[˜][ÛˆÙˆH™\ÛÛ™HH]H\Ý\™›Ü›YYˆËÈ
™TÝØ\™\ÛÛ™H[[YYX][H™XÙY\È]™\žHÝØ\
KÛÈ›ÈÚYHÚ[›™[˜[Y\È]‚ˆÔÝØ\[\
˜\ÙK‹O›\Ý™\ÛÛ™Q\Ý‹O\™Ù]ÚY‹O\™Ù]ZYÚ
NÂŸB‚‹ËÈÙYHš×Ü™[™\™\‹š8 %H]™KU”Þ[˜ÈÙX[H
\Œ
K‚›ÚYšÔ™[™\™\—Ô™\]Y\ÝÝØ\ÚZ[”™XZ[

BžÂˆ×ÜÝØ\™XZ[™\]Y\ÝœÝÜ™JYKÝŽ›Y[[ÜžWÛÜ™\—Ü™[X\ÙJNÂŸB‚‹ËÈH˜XÝÜˆHÐSQIÜÈ›Ø[Z[™ÈØ[Y\˜H]\Ý™HÚY[™YžK[ˆ[ˆÜXÙKÛÈ]]Â‹ËÈÝÛˆMŽŽHÝ[[™Èœ\Ý[HÛÝ™\œÈH™[™\™YšY]ÎˆÈ[ˆÚYH[ÙH
HÜš^›Û[‹ËÈÜ›ÝÜÈžHÊKKÚÈ[ˆ˜\œ›ÝÈ[ÙH
H™\XØ[Ü›ÝÜÈžHKÚÎÈ\L
KH]MŽŽK‚‹ËÈHQSHØ[\HÛÝ[TÈ[ˆ\È\Ú[™È
KˆÜˆ
K›ÜˆH[™[	ÜÈ˜\Y\È]‹ËÈ™^][˜ÚˆÝ\ŽˆHÙ][™ÈØ[ˆY™™\ˆœ›ÛH][[H™[][˜Úˆ™Y›Ü™HB‹ËÈ™[™\™\ˆ^\ÝË‚š[šÔ™[™\™\—Ó\ØXTØ[\\Ê
BžÂˆ™]\›ˆˆÈ[
‹O›\ØXTØ[\\ÊHˆÂŸB‚™›Ø]šÔ™[™\™\—ÕÚYQ›Ý‘˜XÝÜŠ
BžÂˆYˆ
ÚYS[ÙJ
JBˆ™]\›ˆÚYQ›Ý‘˜XÝÜŠ
NÂˆYˆ
˜\œ›ÝÓ[ÙJ
JBˆ™]\›ˆKŒˆÈÚYQ›Ý‘˜XÝÜŠ
NÂˆ™]\›ˆKŒŽÂŸB‚›ÚYšÔ™[™\™\—ÔØ]™T\[[™PØXÚJ
BžÂˆYˆ
Y×ØXÝ]™JBˆ™]\›ŽÂˆØ]™T\[[™PØXÚJ
NÂˆØ]™T\[[™RÙ^\Ê
NÂŸB‚›ÚYšÔ™[™\™\—Ñ[\Ý]Ê
BžÂˆYˆ
Y×ØXÝ]™JBˆ™]\›ŽÂˆœš[ŠÝ\œ‹–Ýš×HKKH™[™\™\ˆÝ]È
œ˜[YH	[JHKKWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O™œ˜[YJNÂˆËÈHQSSÉÔÈÕÓˆ‘TÔ•[™]]™\ÈT‘H™XØ]\ÙH]Èš\œÝÛYHÛÝ[›Ýš\™K‚ˆËÈ\LH]\È[™H[œÚYHHU‘H‘TÐÐSH]8 %H[˜Ý[ÛˆHXY\ÜÈÜ›ÝÙˆËÈ[ˆ™]™\ˆØ[È8 %ÛÈH™\šYšY\ˆ\›H˜[ˆHÚÛH›Ý]H[™š[Y›Ý[™Ë[™ˆËÈŒ\ØYÜ™Y[Y[Èˆ[™H[œÝ[Y[™]™\ˆÜÚÙHˆÙ\™HHØ[YHÝ]]ˆ]™\žBˆËÈ™XÚ\H[ˆ\È›Ú™XÝ[™ÈÛˆH[Y[Ý]ÒQÕT“K[™\È[˜Ý[Ûˆ\ÈÚ]]ˆËÈ[™\ˆØ[ÎÈHÛÝ[\ˆ™\ÜY[ž]Ú\™H[ÙH\ÈHÛÝ[\ˆ›Ø›ÙH™XYË‚ˆMÔ™YÔ[Ù[œÝ\Ô™\Ü

NÂˆYˆ
×Ý^Y[[Ò]È×Ý^Y[[ÓZ\ÜÊBˆœš[ŠÝ\œ‹–Ý^Y[[×H	[H]Ë	[HZ\ÜÙ\È
	KŒY‰IHÙ\™Y
K	[H‚ˆ™\ØYÜ™Y[Y[Ëš[˜[Ù[ˆ	[Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ý^Y[[Ò]Ë
[œÚYÛ™YÛ™ÈÛ™ÊY×Ý^Y[[ÓZ\ÜËˆLŒ
ˆÝX›J×Ý^Y[[Ò]ÊHÈÝX›J×Ý^Y[[Ò]È
È×Ý^Y[[ÓZ\ÜÊKˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ý^Y[[Ñ\ØYÜ™YK
[œÚYÛ™YÛ™ÈÛ™ÊY×Ý^Ù[ŠNÂˆËÈÔSˆUSHÈ8 %HÛÜœÝœ˜[Y\ÈÙˆH[ˆ[™Ú]Ø\È[œÚYH[K‚ˆÂˆÛÝÑœ˜[YT™XÈÌL—NÂˆY[XÜJ×ÜÛÝÕÜÚ^™[Ùˆ
NÂˆÝŽœÛÜ

ÈL‹×JÛÛœÝÛÝÑœ˜[YT™XÉˆKÛÛœÝÛÝÑœ˜[YT™XÉˆŠHÂˆ™]\›ˆK\Èˆ‹\ÎÂˆJNÂˆYˆ
ÌK\ÊBˆÂˆœš[ŠÝ\œ‹–Ýš×HÛÜœÝœ˜[Y\È
Ü[ˆ][HÈ8 %Ú]Ø\ÈSˆ[JN—ˆŠNÂˆ›Üˆ
ÛÛœÝÛÝÑœ˜[YT™XÉˆˆˆ
BˆÂˆYˆ
\‹\ÊBˆœ™XZÎÂˆœš[ŠÝ\œ‹ˆ–Ýš×Hœ˜[YH	NNˆ	NŒYˆ\ÈHÔ\™XÈ	NŒYˆ
È™[˜ÙH	MËŒYˆ
È‚ˆœÛY\	M‹ŒYˆ
È™\ÚY	M‹ŒYˆÔH	MËŒYˆ	MH˜]ÜÈ	MH^‚ˆŠ	MHÐ‹\	MKŒYˆ
ÈXÈ	M‹ŒYˆ\ÊH	MH\Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\‹™œ˜[YKÝX›J‹\ÊHÈLŒˆÝX›J[Ì—Ý
‹Ø[Õ\ÊHH[Ì—Ý
‹™™[˜ÙU\ÊJHÈLŒˆÝX›J‹™™[˜ÙU\ÊHÈLŒÝX›J‹œÛY\\ÊHÈLŒˆÝX›J‹œ™\ÚYX[\ÊHÈLŒÝX›J‹™ÜU\ÊHÈLŒˆ‹™˜]ÜË‹^‹^Ð‹ÝX›J‹^\ÊHÈLŒˆÝX›J‹^XÕ\ÊHÈLŒ‹œ\\ÊNÂˆBˆœš[ŠÝ\œ‹ˆ–Ýš×H
Ô\™XÈHÝ\ˆ™XÛÜ™[™ÎÈ™[˜ÙHHØZ][™È›ÜˆHÔNÈ‚ˆœÛY\H[\YNÈ™\ÚYH›ÝH™[™\™\ˆ][ˆHš\œÝ›Ý\ˆ‚ˆ”ÕSHÈHÐSSQKˆÔH\È]œ˜[YIÜÈÝÛˆ^XÝ][Ûˆ[YHœ›ÛH]È‚ˆ˜ÛÛ[X[™Y™™\‰ÜÈ[Y\Ý[\Ë[™Õ‘T“TÈHÔHÛÛ[[œÈžH\ÚYÛˆ‚ˆ¸ %]\È›Ý\ÙˆHÝ[JWˆŠNÂˆœš[ŠÝ\œ‹ˆ–Ýš×H^\™H\ØYÈÝ™\ˆH[Žˆ	[H\ØYË	KŒYˆP‹‚ˆ‰KŒYˆ\ÈÝYÚ[™ÊÜÝX›Z]
	KŒˆ\ÈXXÚ
KšYÙÙ\ÝÚ[™ÛH\ØY‚ˆ‰KŒ™ˆP—ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ý^™X[\ØYËˆÝX›J×Ý^\ØYž]\ÊHÈLMÍ‹ŒˆÝX›J×Ý^\ØYœÊHÈYM‹ˆ×Ý^™X[\ØYÈÈÝX›J×Ý^\ØYœÊHÈLŒˆÈÝX›J×Ý^™X[\ØYÊBˆˆŒˆÝX›J×Ý^\ØYX^ž]\ÊHÈLMÍ‹Œ
NÂˆœš[ŠÝ\œ‹ˆ–Ýš×H‹‹˜[™	KŒYˆ\ÈPÓÑS‘È[H
[[H
È[™X[ˆÝØ\
È[XYÙH‚ˆ˜Ü™X][ÛŠK	KŒˆ\ÈXXÚ8 %\È\ÈÔHÛÜšÈÛˆH[\[™]\È‚ˆ‰KŒ‰IHÙˆH\ØY]ˆ‹ˆÝX›J×Ý^XÛÙSœÊHÈYM‹ˆ×Ý^™X[\ØYÈÈÝX›J×Ý^XÛÙSœÊHÈLŒˆÈÝX›J×Ý^™X[\ØYÊHˆŒˆ
×Ý^XÛÙSœÈ
È×Ý^\ØYœÊBˆÈLŒ
ˆÝX›J×Ý^XÛÙSœÊBˆÈÝX›J×Ý^XÛÙSœÈ
È×Ý^\ØYœÊHˆŒ
NÂˆËÈHPÓÑIÔÈPÓÓTÔÒUSÓˆ
\ÍÊK[™H‘TÒQPS\Èš[YÚ]]‚ˆËÈ]™\žHÛÛ[[ˆ\™H\ÈHØÛÜH[œÚYHHÛ™HÛØÚÈX›Ý™KÛÈ^H]\ÝÝ[HÂˆËÈ\ÜÈ[ˆ]ÈÚ]\ÈYÝ™\ˆ\È]™\ž][™ÈHÜ]Ù\È›Ý˜[YK[™BˆËÈ\™ÙH™\ÚYX[YX[œÈHÜ]\ÈÜ›Û™È˜]\ˆ[ˆ]HÛÜšÈ˜[š\ÚY‚ˆËÈ]\ÈH˜Ø[››Ý™]\›ˆH˜[ÙHXœÙ[˜ÙHˆÚ\H\ÍYÈZ[™YBˆËÈ[Y\È
0©Í™ˆ0©ÍKÛÝÚHÎJK‚ˆYˆ
×Ý^XÛÙSœÊBˆÂˆÛÛœÝZ[Ý˜[YYH×Ý^XÐ[ØÓœÈ
È×Ý^XÐ˜\ÙSœÈ
È×Ý^XÓZ\œÈ
Âˆ×Ý^XÓZ\ÚÓœÈ
È×Ý^XÔØØ[“œÈ
Âˆ×Ý^XÑÝX\™œÈ
È×Ý^XÒ[XYÙSœÈ
Âˆ×Ý^XÑÛÛ[“œÎÂˆÛÛœÝÝX›HÝHÝX›J×Ý^XÛÙSœÊNÂˆ]]ÈÈHÉ—JZ[ÝŠHÈ™]\›ˆLŒ
ˆÝX›JŠHÈÝÈNÂˆœš[ŠÝ\œ‹ˆ–Ýš×HXÛÙHÜ]
\ÈÈ	IHÙˆXÛÙJNˆ[ØÈ	KŒYˆ
	KŒY‰IJH‚ˆ˜˜\ÙK][[H	KŒYˆ
	KŒY‰IJHZ\][[H	KŒYˆ
	KŒY‰IJH‚ˆ›Z\YÝX\™È	KŒYˆ
	KŒY‰IJHÛÛ[\ØØ[ˆ	KŒYˆ
	KŒY‰IJH‚ˆœÜ˜ËZ\Ú	KŒYˆ
	KŒY‰IJHšÐÜ™X]R[XYÙH	KŒYˆ
	KŒY‰IJH‚ˆ™ÛÛ[ˆ	KŒYˆ
	KŒY‰IJH‘TÒQPS	KŒYˆ
	KŒY‰IJWˆ‹ˆÝX›J×Ý^XÐ[ØÓœÊHÈYM‹Ê×Ý^XÐ[ØÓœÊKˆÝX›J×Ý^XÐ˜\ÙSœÊHÈYM‹Ê×Ý^XÐ˜\ÙSœÊKˆÝX›J×Ý^XÓZ\œÊHÈYM‹Ê×Ý^XÓZ\œÊKˆÝX›J×Ý^XÓZ\ÚÓœÊHÈYM‹Ê×Ý^XÓZ\ÚÓœÊKˆÝX›J×Ý^XÔØØ[“œÊHÈYM‹Ê×Ý^XÔØØ[“œÊKˆÝX›J×Ý^XÑÝX\™œÊHÈYM‹Ê×Ý^XÑÝX\™œÊKˆÝX›J×Ý^XÒ[XYÙSœÊHÈYM‹Ê×Ý^XÒ[XYÙSœÊKˆÝX›J×Ý^XÑÛÛ[“œÊHÈYM‹Ê×Ý^XÑÛÛ[“œÊKˆÝX›J×Ý^XÛÙSœÈHÝŽ›Z[Š˜[YY×Ý^XÛÙSœÊJHÈYM‹ˆÊ×Ý^XÛÙSœÈHÝŽ›Z[Š˜[YY×Ý^XÛÙSœÊJJNÂˆœš[ŠÝ\œ‹ˆ–Ýš×H˜\ÙH[[Nˆ	[H[š]ÈÝ™\ˆ	[H\ØYË	KŒYˆœËÝ[š]‚ˆŠHÝ[Ø[››Ý[HÛÝÈÛÜœ›ÛHHÝÙˆ[š]ÊWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ý^XÐ˜\ÙU[š]Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ý^™X[\ØYËˆ×Ý^XÐ˜\ÙU[š]ÂˆÈÝX›J×Ý^XÐ˜\ÙSœÊHÈÝX›J×Ý^XÐ˜\ÙU[š]ÊHˆŒ
NÂˆBˆYˆ
×ÛYÐÚXÚÙY
Bˆœš[ŠÝ\œ‹ˆ–Ýš×HZ\YÝX\™™\šYžNˆ	[HÙˆ	[HÚXÚÜÈTÐQÔ‘QQÚ]H‚ˆœ™K\\MÍÈÛÛ\]][Ûˆ
	K‰IJH8 %\ÈHÛ›H\ÜÚ[™È˜[YK‚ˆ˜[™Ö—Õ’×Õ‘T’Q–WÓRTÑÕPT‘ÔÒTÓÓLH]\ÝXZÙH]L	IWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÛYÑ\ØYÜ™YKˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÛYÐÚXÚÙYˆLŒ
ˆÝX›J×ÛYÑ\ØYÜ™YJHÈÝX›J×ÛYÐÚXÚÙY
JNÂˆYˆ
×ØÚSŠBˆœš[ŠÝ\œ‹ˆ–Ýš×HÜ™X]R[XYÙH	[H
	KŒYˆPˆ]šXÙJNˆšÐÜ™X]R[XYÙH	KŒYˆ\È‚ˆˆY[T™\H	KŒYˆšÐ[ØØ]SY[[ÜžH	KŒYˆš[™	KŒYˆšY]È	KŒYˆ‚ˆˆOˆ	KŒˆ\ÈXXÚˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ØÚS‹ˆÝX›J×ØÚP[ØÐž]\ÊHÈLMÍ‹ŒˆÝX›J×ØÚPÜ™X]SœÊHÈYM‹ÝX›J×ØÚT™\SœÊHÈYM‹ˆÝX›J×ØÚP[ØÓœÊHÈYM‹ÝX›J×ØÚPš[™œÊHÈYM‹ˆÝX›J×ØÚUšY]ÓœÊHÈYM‹ˆÝX›J×ØÚPÜ™X]SœÈ
È×ØÚT™\SœÈ
È×ØÚP[ØÓœÈ
È×ØÚPš[™œÈ
Âˆ×ØÚUšY]ÓœÊHÈLŒÈÝX›J×ØÚSŠJNÂˆœš[ŠÝ\œ‹ˆ–Ýš×H[XYÙHY[[ÜžNˆ	[HÛÛY[È	[H›ØÚÉ\Ë	[HYXØ]Y‚ˆ˜[ØØ][ÛœÈ
	KŒYˆÐˆÜÝÈ[YÛ›Y[[œÚYH›ØÚÜÊWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ú[YÔÛÛYˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ú[YÐ›ØÚÐ[ØÜËˆ×Ú[YÐ›ØÚÐ[ØÜÈOHHÈˆˆˆœÈ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ú[YÑYXØ]YˆÝX›J×Ú[YÔÛÛØ\ÝPž]\ÊHÈLŒ
NÂˆœš[ŠÝ\œ‹ˆ–Ýš×H^\™H\ØY˜]Úˆ	[H›ØœÈ[ˆ	[H›\Ú\È
	KŒYˆ\ˆ‚ˆ™›\ÚšYÙÙ\Ý	[JK	[HÙˆ[H›Ü˜ÙYžHH[ÝYÚ[™È\™[˜H8 %‚ˆ™XXÚ›\Ú\ÈÓ‘HÝX›Z]Ú\™HXXÚ“Ðˆ\ÙYÈ™HÛ™Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ý^˜]Ú›ØœËˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ý^˜]Ú›\Ú\Ëˆ×Ý^˜]Ú›\Ú\ÈÈÝX›J×Ý^˜]Ú›ØœÊHÈÝX›J×Ý^˜]Ú›\Ú\ÊBˆˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ý^˜]ÚX^›ØœËˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ý^˜]Ú[›\Ú\ÊNÂˆËÈHT•MÎHUSK[™HÛÈ[X™\œÈ]XÚYHÚ]\ˆ]ÛÜšÙYˆBˆËÈ›\ÚÛÜÝ\ÈÚ]HÜ\˜]Ü‰ÜÈÙ\ÜÚ[ÛˆYX\Ý\™Y]MŽ\ÎÈHÝ[ˆËÈÛÝ[\ÈÚ]\ˆ™YHÛÝÈÙ\™H[›ÝYÚˆHš[™È]Ý[ÈÛˆ[ÜÝˆËÈ›\Ú\È\È›Ý™[[Ý™YHØZ]]\È™[˜[YY]‚ˆYˆ
×Ý^›\Ú\ÊBˆœš[ŠÝ\œ‹ˆ–Ýš×H^\™H›\Úˆ	[H›\Ú\Ë	KŒYˆ\ÈÝ[
	KŒˆ\È‚ˆ™XXÚ
H8 %š[™È	\ÎÈ	[HÙˆ[HÕSQØZ][™È›ÜˆHÛÝ‚ˆŠ	KŒYˆ\Ë	KŒˆ\ÈXXÚ
Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ý^›\Ú\ËˆÝX›J×Ý^›\ÚœÊHÈYM‹ˆÝX›J×Ý^›\ÚœÊHÈLŒÈÝX›J×Ý^›\Ú\ÊKˆ×Ý^›\ÚØZ]È‘TÐP“Q
Ö—Õ’×ÕVÑ“TÒÕÐRULH8 %ÝX›Z][™‚ˆšÔ]Y]YUØZ]YKH\MÍÈ™[™\™\ŠH‚ˆˆ™[™ØYÙY›ÈØZ]ÛˆHÝX›Z][™ÈÛÝ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ý^ÛÝÝ[ËˆÝX›J×Ý^ÛÝÝ[œÊHÈYM‹ˆ×Ý^ÛÝÝ[ÈÈÝX›J×Ý^ÛÝÝ[œÊHÈLŒˆÈÝX›J×Ý^ÛÝÝ[ÊBˆˆŒ
NÂˆYˆ
×Ú[[SŠBˆœš[ŠÝ\œ‹ˆ–Ýš×H[[YYX]HÝX›Z]Îˆ	[K	KŒYˆ\ÈÝ[ÙˆÚXÚ‚ˆ‰KŒYˆ\È
	KŒY‰IJH\ÈšÔ]Y]YTÝX›Z]
ÝšÔ]Y]YUØZ]YH‚ˆŠ	KŒˆ\ÈXXÚ
H8 %HØZ]\È›ÜˆHÒÓHUQUQKÛÈ]‚ˆš[˜ÛY\ÈH[‹Y›YÚœ˜[YWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ú[[S‹ÝX›J×Ú[[UÝ[œÊHÈYM‹ˆÝX›J×Ú[[UØZ]œÊHÈYM‹ˆLŒ
ˆÝX›J×Ú[[UØZ]œÊHÈÝX›J×Ú[[UÝ[œÊKˆÝX›J×Ú[[UØZ]œÊHÈLŒÈÝX›J×Ú[[SŠJNÂ‚ˆBˆBˆËÈHTÔËTÒV‘HTÕÑÔSK[™HÛÛ[[ˆ]XÚY\È][HH\ÈHUËUÑRQÒQˆËÈÛ™K›ÝH\ÜÈÛÝ[ˆˆÙˆ\ÜÙ\È\™H[žHˆÛÝ[™È˜][[™\È\œ™[]˜[ˆËÈYˆHÝ\ˆÛÈØ\œžHMIHÙˆH˜]ÜÈ8 %HÛÜšÙ\ˆÛÛ™YYÈÛˆUÔËˆÛÈ›Ý\™BˆËÈš[YÚYHžHÚYH[™HÝ[][]]™H˜]ÈÚ\™H\ÈÜ[YÝ]™XØ]\ÙHBˆËÈXÚ\Ú[Ûˆ\ÈšÝÈX[žH\ÜÙ\ÈÈH™YY™Y›Ü™HH]™H[ÜÝÙˆHœ˜[YH‹‚ˆYˆ
×Ü\ÜÐÛÝ[
BˆÂˆœš[ŠÝ\œ‹ˆ–Ýš×H\ÜÈÚ^™\Îˆ	[H\ÜÙ\Ë	[H˜]ÜËYX[ˆ	KŒ‹X^	[Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü\ÜÐÛÝ[
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü\ÜÑ˜]ÜËˆÝX›J×Ü\ÜÑ˜]ÜÊHÈÝX›J×Ü\ÜÐÛÝ[
Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü\ÜÓX^
NÂˆËÈØ[Èœ›ÛHHT‘ÑTÕXÚÙ]ÝÛŽˆH[œÝÙ\ˆØ[Y\ÈHšYÙÙ\Ýˆ\ÜÙ\ÂˆËÈÛ	HÙˆ[˜]ÜÈ‹ÚXÚ™XYÈ\™XÝHÙ™ˆH\ØÙ[™[™ÈÝ[][]]™K‚ˆZ[ÝÝ[THÝ[QHÂˆ›Üˆ
[ˆHÔ\ÜÐXÚÙ]ÈHNÈˆHÈKXŠBˆÂˆYˆ
Y×Ü\ÜÒ\ÝØ—JBˆÛÛ[YNÂˆÝ[T
ÏH×Ü\ÜÒ\ÝØ—NÂˆÝ[Q
ÏH×Ü\ÜÒ\Ý˜]ÜÖØ—NÂˆÚ\ˆ˜[™ÙVÌÌ—NÂˆYˆ
ˆOH
BˆÛœš[Š˜[™ÙKÚ^™[Ùˆ˜[™ÙK™[\HŠNÂˆ[ÙHYˆ
ˆOHJBˆÛœš[Š˜[™ÙKÚ^™[Ùˆ˜[™ÙKŒHŠNÂˆ[ÙHYˆ
ˆH[
Ô\ÜÐXÚÙ]ÊHHJBˆÛœš[Š˜[™ÙKÚ^™[Ùˆ˜[™ÙKI]H‹]H
Ô\ÜÐXÚÙ]ÈHŠJNÂˆ[ÙBˆÛœš[Š˜[™ÙKÚ^™[Ùˆ˜[™ÙK‰]KI]H‹]H
ˆHJK
]HŠHHJNÂˆœš[ŠÝ\œ‹ˆ–Ýš×H	NÈ˜]ÜÎˆ	NH\ÜÙ\È
	MKŒY‰IJK	LLH˜]ÜÈ
	MKŒY‰IJH‚ˆˆÝ[][]]™Hœ›ÛHHÜˆ	MKŒY‰IHÙˆ\ÜÙ\ÈÛ	MKŒY‰IHÙˆ˜]Ü×ˆ‹ˆ˜[™ÙK
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü\ÜÒ\ÝØ—KˆLŒ
ˆÝX›J×Ü\ÜÒ\ÝØ—JHÈÝX›J×Ü\ÜÐÛÝ[
Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü\ÜÒ\Ý˜]ÜÖØ—Kˆ×Ü\ÜÑ˜]ÜÈÈLŒ
ˆÝX›J×Ü\ÜÒ\Ý˜]ÜÖØ—JHÈÝX›J×Ü\ÜÑ˜]ÜÊHˆŒˆLŒ
ˆÝX›JÝ[T
HÈÝX›J×Ü\ÜÐÛÝ[
Kˆ×Ü\ÜÑ˜]ÜÈÈLŒ
ˆÝX›JÝ[Q
HÈÝX›J×Ü\ÜÑ˜]ÜÊHˆŒ
NÂˆBˆBˆËÈHÔKÑÔHÔUÝ™\ˆHÚÛH[ˆ8 %HXY[™H›ÜˆØ\ÈHœ˜[YHÔHÜˆÔH‹‚ˆYˆ
×ÙÜQœ˜[Y\È×Ù™[˜ÙUØZ]œÊBˆÂˆÛÛœÝÝX›Hœ˜[Y\ÈHÝX›J‹O™œ˜[YHÈ‹O™œ˜[YHˆJNÂˆYˆ
×Ý[Y\Ý[\\š[ÙœÈHŒ
Bˆœš[ŠÝ\œ‹–Ýš×HÔHœ˜[YH[YNˆSURSP“HÛˆ\È]šXÙH
›È‚ˆ[Y\Ý[\Ý\Ü
H8 %H™[˜ÙHØZ]™[ÝÈ\ÈHÛ›H‚ˆ‘ÔK\ÚYH[X™\—ˆŠNÂˆ[ÙBˆœš[ŠÝ\œ‹ˆ–Ýš×HÔHœ˜[YH[YNˆ	KŒ™ˆ\ÈYX[ˆÝ™\ˆ	[Hœ˜[Y\ÈYX\Ý\™Y‚ˆŠœ›ÛHXXÚœ˜[YIÜÈÝÛˆÛÛ[X[™XY™™\ˆ[Y\Ý[\ÊWˆ‹ˆÝX›J×ÙÜQœ˜[YSœÊHÈYMˆÈÝX›J×ÙÜQœ˜[Y\ÈÈ×ÙÜQœ˜[Y\ÈˆJKˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙÜQœ˜[Y\ÊNÂˆœš[ŠÝ\œ‹ˆ–Ýš×H™[˜ÙHØZ]ˆ	KŒ™ˆ\ËÙœ˜[YHYX[ˆ8 %\È\ÈHÔH“ÐÒÑQÓˆH‚ˆ‘ÔK[™]\ÈH[ˆÙˆØ[Ø]\È›ÝÝ\ˆ™XÛÜ™[™×ˆ‹ˆÝX›J×Ù™[˜ÙUØZ]œÊHÈYMˆÈœ˜[Y\ÊNÂˆBˆËÈHT”’QTˆ“Ô“K[™H]šY[˜ÙH]H\›H[™ØYÙYˆ[˜ÛÛ™][Û˜[[™œ™YK‚ˆYˆ
×Ø˜\œšY\“ŠBˆœš[ŠÝ\œ‹ˆ–Ýš×H[XYÙH˜\œšY\œÎˆ	[HÝ™\ˆH[ˆ
	KŒY‹Ùœ˜[YJK	[HÙˆ[HÚ]‚ˆHÒQHSÐÓÓSPS‘ËÓQSSÔ–WÊˆX\ÚÜÈ
	KŒY‰IJH8 %[™\ˆ‚ˆÖ—Õ’×ÕÒQWÐT”’QT”ÏLH]]\Ý™HL	IK[™Ú]Ý]]Û›HÑS‘TS[™‚ˆ[›\ÝY^[Ý]ÎÈ\È	[HÜš]KXY\‹]Üš]H˜\œšY\œÈÛˆ[XYÙ\È[™XYH‚ˆš[ˆH^[Ý]^H™YYYˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ø˜\œšY\“‹ˆÝX›J×Ø˜\œšY\“ŠHÈÝX›J‹O™œ˜[YHÈ‹O™œ˜[YHˆJKˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ø˜\œšY\•ÚYKˆLŒ
ˆÝX›J×Ø˜\œšY\•ÚYJHÈÝX›J×Ø˜\œšY\“ŠKˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÝØ]ÓŠNÂˆËÈHT‹T‘QÒSÓˆÔHÔU
\Î][HJH8 %Hš\œÝœ™XZÙÝÛˆÙˆH]šXÙIÜÈÝÛ‚ˆËÈ[YH\È›Ú™XÝ\ÈYˆH‘TÒQPSTÈ’S•Q’T”ÕS‘ÓˆT”ÔÑNˆ]\ÈBˆËÈœ˜[YIÜÈYX\Ý\™YÔH[YHZ[\È]™\ž][™ÈH™YÚ[ÛœÈXØÛÝ[›Ü‹[™]\ÈBˆËÈÛ›H[™È\™H]Ø[ˆØ^HHÜ]\ÈÜ›Û™ËˆH\™ÙH™\ÚYX[YX[œÈH™YÚ[Ûˆ\ÂˆËÈZ\ÜÚ[™Ë›Ý]ÛÜšÈ˜[š\ÚY
ÛÝÚHŒÍÉÜÈÚ\KÛ™H]™[ÝÛŠK‚ˆYˆ
×ÙÜœ˜[Y\ÊBˆÂˆÛÛœÝÝX›HˆHÝX›J×ÙÜœ˜[Y\ÊNÂˆÛÛœÝÝX›HÝ\ÈHÝX›J×ÙÜÝ[œÊHÈYMˆÈŽÂˆÛÛœÝÝX›H]\ÈHÝX›J×ÙÜ]šX“œÊHÈYMˆÈŽÂˆœš[ŠÝ\œ‹ˆ–Ýš×HÔH\‹\™YÚ[ÛˆÜ]
Ö—Õ’×ÑÔWÔTÔÑTÊHÝ™\ˆ	[Hœ˜[Y\È8 %‚ˆ‰KŒÙˆ\ËÙœ˜[YHYX\Ý\™Y	KŒÙˆ\È]šX]Y‘TÒQPS	KŒÙˆ\È
	KŒY‰IJWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙÜœ˜[Y\ËÝ\Ë]\ËÝ\ÈH]\ËˆÝ\ÈˆŒÈLŒ
ˆ
Ý\ÈH]\ÊHÈÝ\ÈˆŒ
NÂˆ›Üˆ
[ÈHÈÈÑÜÛ\ÜÙ\ÎÈ
ÊØÊBˆÂˆÛÛœÝÝX›H\ÈHÝX›J×ÙÜœÖØ×JHÈYMˆÈŽÂˆœš[ŠÝ\œ‹ˆ–Ýš×H	KLŒÈ	NŒÙˆ\ËÙœ˜[YH
	MKŒY‰IHÙˆHœ˜[YIÜÈÔH[YJH‚ˆ‰NŒ™ˆ™YÚ[ÛœËÙœ˜[YK	MËŒˆœÈXXÚˆ‹ˆÑÜ˜[Y\ÖØ×K\ËÝ\ÈˆŒÈLŒ
ˆ\ÈÈÝ\ÈˆŒˆÝX›J×ÙÜ–Ø×JHÈ‹ˆ×ÙÜ–Ø×HÈÝX›J×ÙÜœÖØ×JHÈÝX›J×ÙÜ–Ø×JHˆŒ
NÂˆBˆËÈHS•“ÐÐUSÓˆÑS”ÕTÈ
Ö—Õ’×ÑÔWÔÕUÊKˆ\ˆ\ÜÈÛ\ÜÎˆš[Z]]™\È[‹™\^ˆËÈ[›ØØ][ÛœËœ˜YÛY[[›ØØ][ÛœÈ8 %[™œ˜YÛY[[›ØØ][ÛœÈ\ˆS•T“SˆËÈVSÚXÚ\ÈHÝ™\™˜]È˜XÝÜˆ[™Hš\œÝ[X™\ˆHÔHYÙ]™YYË‚ˆÂˆ›ÛÛ[žHH˜[ÙNÂˆ›Üˆ
[ÈHÈÈÑÜÛ\ÜÙ\ÎÈ
ÊØÊBˆ[žHH[žH×ÜÝ–Ø×NÂˆYˆ
[žJBˆÂˆËÈH’TÒP“H[\›˜[™\ÛÛ][Û‹›ÝHQSHÝ[™Z[ˆ
ÚXÚØ\œšY\ÈBˆËÈL\›ÝÈÝY\ÝÝ\™˜XÙH™[ÝÈHÌŒH]H™\Ù[Èœ›ÛJNˆÝ™\™˜]ÂˆËÈ\ÈšÝÈX[žH[Y\ÈHØÜ™Y[ˆØ\ÈÚYY‹[™HØÜ™Y[ˆ\ÈNLŒL‚ˆÛÛœÝÝX›HHÝX›J×Ú[\›˜[Ë›ØY

JH
ˆÝX›J×Ú[\›˜[›ØY

JNÂˆZ[ÝÝÚÔÝÛÝ[\œ×HHßNÂˆœš[ŠÝ\œ‹ˆ–Ýš×HS•“ÐÐUSÓˆÑS”ÕTÈ
Ö—Õ’×ÑÔWÔÕUÊH8 %\ˆœ˜[YKÝ™\ˆ	[H‚ˆ™œ˜[Y\ÎÈÝ™\™˜]ÈH”È[›ØØ][ÛœÈÈ	]^	]H[\›˜[^[È‚ˆŠÝ™\™›ÝÈ	[K˜Y™XYÈ	[JWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙÜœ˜[Y\Ë×Ú[\›˜[Ë›ØY

Kˆ×Ú[\›˜[›ØY

Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÜÝÝ™\™›ÝË
[œÚYÛ™YÛ™ÈÛ™ÊY×ÜÝ˜Y™XY
NÂˆ›Üˆ
[ÈHÈÈÑÜÛ\ÜÙ\ÎÈ
ÊØÊBˆÂˆYˆ
Y×ÜÝ–Ø×JBˆÛÛ[YNÂˆ›Üˆ
Z[Ì—ÝÈHÈÈÔÝÛÝ[\œÎÈ
ÊÚÊBˆÝÚ×H
ÏH×ÜÝÝ[VØ×VÚ×NÂˆœš[ŠÝ\œ‹ˆ–Ýš×H	KLŒÈ	MËŒ™ˆ\ÜÙ\ÈPHš[\È	NKŒˆ”È[ˆ	LLŒˆ‚ˆ˜Û\š[\È	NKŒˆ”È[ˆ	LLKŒˆH	KŒ™ˆ^[È‚ˆŠ	KŒYˆ”È[‹Üš[JWˆ‹ˆÑÜ˜[Y\ÖØ×KÝX›J×ÜÝ–Ø×JHÈ‹ÝX›J×ÜÝÝ[VØ×VÌWJHÈ‹ˆÝX›J×ÜÝÝ[VØ×VÌ—JHÈ‹ÝX›J×ÜÝÝ[VØ×VÍJHÈ‹ˆÝX›J×ÜÝÝ[VØ×VÍWJHÈ‹ÝX›J×ÜÝÝ[VØ×VÍWJHÈˆÈˆ×ÜÝÝ[VØ×VÌWHÈÝX›J×ÜÝÝ[VØ×VÌ—JHÈÝX›J×ÜÝÝ[VØ×VÌWJBˆˆŒ
NÂˆBˆœš[ŠÝ\œ‹ˆ–Ýš×H	KLŒÈPHš[\È	NKŒˆ”È[ˆ	LLŒˆ‚ˆ˜Û\š[\È	NKŒˆ”È[ˆ	LLKŒˆH	KŒ™ˆ^[×ˆ‹ˆSTÔÑTÈ‹ÝX›JÝÌWJHÈ‹ÝX›JÝÌ—JHÈ‹ˆÝX›JÝÍJHÈ‹ÝX›JÝÍWJHÈ‹ÝX›JÝÍWJHÈˆÈ
NÂˆ›Üˆ
Z[Ì—ÝÈHÈÈÔÝÛÝ[\œÎÈ
ÊÚÊBˆœš[ŠÝ\œ‹–Ýš×H	KLNÈ	LMŒˆÙœ˜[YWˆ‹ÔÝ˜[Y\ÖÚ×KˆÝX›JÝÚ×JHÈŠNÂˆBˆBˆœš[ŠÝ\œ‹ˆ–Ýš×H™\ÛÛ™HÛÜY\È[Ý™Y	KŒ™ˆ\^[Ùœ˜[YH
	KŒYˆ[	]^	]H‚ˆœØÜ™Y[œÉÈÛÜ
Wˆ‹ˆÝX›J×ÙÜ™\ÛÛ™T^[ÊHÈYMˆÈ‹ˆÝX›J×ÙÜ™\ÛÛ™T^[ÊHÈˆÂˆ
ÝX›J‹O˜ÛÛÜ‹ÚY
H
ˆÝX›J‹O˜ÛÛÜ‹šZYÚ
JKˆ‹O˜ÛÛÜ‹ÚY‹O˜ÛÛÜ‹šZYÚ
NÂˆYˆ
×ÙÜÛX\“ŠBˆœš[ŠÝ\œ‹ˆ–Ýš×H™\ÛÛ™HÛX\œÎˆ[Z[XYÙHYXÚ[š\ÛHÛÝ[Üš]H	KŒ™ˆ‚ˆ“\^[Ùœ˜[YHÝ™\ˆ	KŒYˆÛX\œÎÈHØÛÜY™XÝÈÛÝ™\ˆ	KŒ™ˆ\^[‚ˆŠ	KŒY‰IHÙˆHÛ\ÜÈ	\ÊWˆ‹ˆÝX›J×ÙÜÛX\‘[^[ÊHÈYMˆÈ‹ÝX›J×ÙÜÛX\“ŠHÈ‹ˆÝX›J×ÙÜÛX\”ØÛÜY^[ÊHÈYMˆÈ‹ˆ×ÙÜÛX\‘[^[ÈÈLŒ
ˆ
KŒHÝX›J×ÙÜÛX\”ØÛÜY^[ÊHÂˆÝX›J×ÙÜÛX\‘[^[ÊJBˆˆŒˆ[“ÛŠÖ—Õ’×Ó“×ÑQ‘T”‘QÐÓPTˆŠBˆÈ•ÓÕS‘H‘SSÕ‘QžHHY™\œ™Y\ØÛÜYY˜][
\È[ˆ‚ˆ˜Ø\œšY\ÈHÖ—Õ’×Ó“×ÑQ‘T”‘QÐÓPTˆÛÛ›Û\›JH‚ˆˆ”‘SSÕ‘QžHHY™\œ™Y\ØÛÜYYXÚ[š\ÛK\LŠNÂˆËÈHTÔÈVS•ÑS”ÕTÈ
\ÎH][HŠKˆ\ÜÎˆH˜]Ø\ÈŒÌ\ÜÙ\ÈHœ˜[YH]ˆËÈŒŽ\È[™\È›Ú™XÝ\È™]™\ˆ\ÝYÚ]^HT‘KˆÛÜYžHÝ[[YKÛÂˆËÈHš\œÝ›ÝÜÈ\™HHÛ™\ÈÛÜ\ÚYÛš[™ÈYØZ[œÝ[™XXÚ›ÝÈØ\œšY\È]ÈÝÛ‚ˆËÈ\‹\\ÜÈZXÜ›ÜÙXÛÛ™È8 %ÚXÚ\ÈÚ][ÈH[\ØÜ™Y[ˆÚY\ˆ\\œ›ÛH\ÜÂˆËÈÝ™\šXYÛˆHMžH›ÛÛH\™Ù]ˆ[˜Ø]Y]Mˆ›ÝÜÈ\ˆÛ\ÜÈÚ]HZ[ˆËÈÕSSQQ˜]\ˆ[ˆ›ÜY™XØ]\ÙHHÚ[[H[˜Ø]YÙ[œÝ\È™XYÈ\ÈBˆËÈÛÛ\]HÛ™H
ÛÝÚHÊK‚ˆ›Üˆ
[ÈHÑÜ\ÜÑ[\NÈÈHÑÜ\ÜÔÚYÝÎÈ
ÊØÊBˆÂˆYˆ
×ÙÜ^[ÖØ×K™[\J
JBˆÛÛ[YNÂˆÝŽ™XÝÜÝŽœZ\Z[ÝÜ^[Ý]ˆ›ÝÜÊ×ÙÜ^[ÖØ×K˜™YÚ[Š
Kˆ×ÙÜ^[ÖØ×K™[™

JNÂˆÝŽœÛÜ
›ÝÜË˜™YÚ[Š
K›ÝÜË™[™

Kˆ×JÛÛœÝ]]ÉˆKÛÛœÝ]]ÉˆŠHÈ™]\›ˆKœÙXÛÛ™›œÈˆ‹œÙXÛÛ™›œÎÈJNÂˆœš[ŠÝ\œ‹–Ýš×HVS•ÑS”ÕTÈ›Üˆ	É\ÉÈ8 %	^H\Ý[˜ÝØÚ\ÜÛÜˆÚ^™\×ˆ‹ˆÑÜ˜[Y\ÖØ×K›ÝÜËœÚ^™J
JNÂˆÚ^™WÝÚÝÛˆHÂˆZ[ÝZ[ˆHZ[œÈHÂˆ›Üˆ
ÛÛœÝ]]Éˆˆˆ›ÝÜÊBˆÂˆYˆ
ÚÝÛˆMŠBˆÂˆœš[ŠÝ\œ‹ˆ–Ýš×H	M[^	KM[H	NŒÙˆ\ËÙœ˜[YH	MËŒ™ˆ\ÜÙ\ËÙœ˜[YH‚ˆ‰MËŒˆ\ÈXXÚ	KŒ™ˆ\^[Ùœ˜[YWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ‹™š\œÝˆÌŠKˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ‹™š\œÝ	ˆ‘‘‘‘‘‘‘[
KˆÝX›J‹œÙXÛÛ™›œÊHÈYMˆÈ‹ÝX›J‹œÙXÛÛ™›ŠHÈ‹ˆÝX›J‹œÙXÛÛ™›œÊHÈLŒÈÝX›J‹œÙXÛÛ™›ŠKˆÝX›J‹™š\œÝˆÌŠH
ˆÝX›J‹™š\œÝ	ˆ‘‘‘‘‘‘‘[
H
‚ˆÝX›J‹œÙXÛÛ™›ŠHÈYMˆÈŠNÂˆ
ÊÜÚÝÛŽÂˆBˆ[ÙBˆÂˆZ[ˆ
ÏH‹œÙXÛÛ™›ŽÂˆZ[œÈ
ÏH‹œÙXÛÛ™›œÎÂˆBˆBˆYˆ
Z[ŠBˆœš[ŠÝ\œ‹ˆ–Ýš×H‹‹˜[™	^H[Ü™HÚ^™\Îˆ	KŒÙˆ\ËÙœ˜[YHÝ™\ˆ	KŒ™ˆ‚ˆœ\ÜÙ\ËÙœ˜[YWˆ‹ˆ›ÝÜËœÚ^™J
HHÚÝÛ‹ÝX›JZ[œÊHÈYMˆÈ‹ÝX›JZ[ŠHÈŠNÂˆBˆYˆ
×ÙÜÝ™\™›ÝÈ×ÙÜ˜Y™XY
Bˆœš[ŠÝ\œ‹ˆ–Ýš×H
Šˆ	[H™YÚ[ÛœÈ›ÜY›ÜˆØ[ÙˆH]Y\žHÛÝ[™	[H‚ˆ[œ™XYX›H8 %]™\žHÛ™HÙˆÜÙHS‘T‹\™\ÜÈ]ÈÛ\Ü×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙÜÝ™\™›ÝË
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙÜ˜Y™XY
NÂˆœš[ŠÝ\œ‹ˆ–Ýš×H
H™YÚ[Û‰ÜÈ[YH\ÈHØ[[YH™]ÙY[ˆÛÈ“ÕÓWÓÑ—ÔTH‚ˆ[Y\Ý[\ËÛÈ]\È[ˆ\\ˆ›Ý[™Ûˆ]ÈÝÛˆÛÜÝÚ\™HH]šXÙH‚ˆ›Ý™\›\È™YÚ[ÛœÎÈHÕSH[™H‘TÒQPS\™HHÛ™\Ý]X[]Y\ÊWˆŠNÂˆBˆËÈ0©Íˆ8 %ÒUHH‘PT‹QSTHTÔÑTÈH”SQHPÕPSHÓÔÕˆÜ]žHHÚ^™HÙ‚ˆËÈH\ÜÈ][™Y™XØ]\ÙHHXÚ\Ú[Ûˆ\ÈÚ]\È™XÛÝ™\™YžH›Ý\ÜÝZ[™ÈBˆËÈ™X\‹Y[\HÛ™\È‹›ÝÚ]È™\ÛÛ™\ÈÛÜÝˆ8 %HÝ[\ÈÛZ[˜]YžHHKŒÍBˆËÈšYÈ™\ÛÛ™\ÈHœ˜[YKÚXÚÛ˜\ÚÝH[\ØÜ™Y[ˆÝ\™˜XÙH[™\™H›Ý™[[Ý˜X›K‚ˆYˆ
×ØÞXÓ–ÌH
È×ØÞXÓ–ÌWH
È×ØÞXÓ–Ì—JBˆÂˆÛÛœÝÝX›Hœ˜[Y\ÈHÝX›J‹O™œ˜[YHÈ‹O™œ˜[YHˆJNÂˆÝ]XÈÛÛœÝÚ\ŠˆÓ˜[Y\ÖÚÐÞXÐÛ\ÜÙ\×HHÈ›™X\‹Y[\H
LH˜]ÊH‹ˆœÛX[
‹LMJH‹˜šYÈ
LMŠHˆNÂˆœš[ŠÝ\œ‹ˆ–Ýš×H™\ÛÛ™KØ™YÚ[ˆÞXÛHÛÜÝ
[ˆ0©ÍŠH8 %ÔH‘PÓÔ‘S‘È[YHÛˆH‚ˆœ[\›ÝHÔIÜÈÛÜÝ›ÜˆH^˜H\ÜÈ[œÝ[˜ÙN—ˆŠNÂˆÝX›HÝ[\ÈHŒÂˆ›Üˆ
[ÈHÈÈÐÞXÐÛ\ÜÙ\ÎÈ
ÊØÊBˆÂˆYˆ
Y×ØÞXÓ–Ø×JBˆÛÛ[YNÂˆÛÛœÝÝX›H\ÈHÝX›J×ØÞXÔ™\ÛÛ™SœÖØ×H
È×ØÞXÐ™YÚ[“œÖØ×JHÈYMˆÈœ˜[Y\ÎÂˆÝ[\È
ÏH\ÎÂˆœš[ŠÝ\œ‹ˆ–Ýš×H	KLŒœÈ	NŒ™‹Ùœ˜[YH	MËŒˆœÈXXÚ
™\ÛÛ™H	M‹Œˆ
È™YÚ[ˆ‚ˆ‰M‹ŒŠHH	M‹ŒÙˆ\ËÙœ˜[YHØÛÜHXÝX[Hœ›ÚÙ[ˆ[ˆ	MKŒY‰IWˆ‹ˆÓ˜[Y\ÖØ×KÝX›J×ØÞXÓ–Ø×JHÈœ˜[Y\ËˆÝX›J×ØÞXÔ™\ÛÛ™SœÖØ×H
È×ØÞXÐ™YÚ[“œÖØ×JHÈÝX›J×ØÞXÓ–Ø×JKˆÝX›J×ØÞXÔ™\ÛÛ™SœÖØ×JHÈÝX›J×ØÞXÓ–Ø×JKˆÝX›J×ØÞXÐ™YÚ[“œÖØ×JHÈÝX›J×ØÞXÓ–Ø×JKˆ\ËLŒ
ˆÝX›J×ØÞXÐœ›ÚÙVØ×JHÈÝX›J×ØÞXÓ–Ø×JJNÂˆBˆœš[ŠÝ\œ‹ˆ–Ýš×HÝ[	KŒÙˆ\ËÙœ˜[YNÈH‘PT‹QSTHÛ\ÜÈ\ÈH][IÜÈÙZ[[™È‚ˆ˜[™]\È	KŒÙˆ\ËÙœ˜[YWˆ‹ˆÝ[\ËÝX›J×ØÞXÔ™\ÛÛ™SœÖÌH
È×ØÞXÐ™YÚ[“œÖÌJHÈYMˆÈœ˜[Y\ÊNÂˆœš[ŠÝ\œ‹ˆ–Ýš×H
Yˆ	ÜØÛÜHXÝX[Hœ›ÚÙ[‰È\ÈÛX[›ÜˆH™X\‹Y[\HÛ\ÜË‚ˆ°©Í‰ÜÈ™[Z\ÙH\ÈÜ›Û™ÎˆÜÙH\ÜÙ\È\™H›Ý[™
Ð™YÚ[ˆÞXÛ\È][
WˆŠNÂˆBˆËÈHÓÓ”ÕS•TÓÕPÑHUPÕÔˆ
\Í
H8 %HØ]HHØ]\ˆ\ÈÈ\ÜÈ™Y›Ü™BˆËÈÖ—Õ’×ÐÓÓ”ÕÑÐUTLHØ[ˆ™XÛÛYHHY˜][YØZ[‹‚ˆYˆ
ÛÛœÝ˜XÙSÛŠ
JBˆÂˆœš[ŠÝ\œ‹ˆ–Ýš×HÛÛœÝ˜XÙNˆ	[H˜]ÜÈÚXÚÙYÝ™\ˆ	[Hœ˜[Y\È8 %‚ˆŠŠ‰[HYZ\ˆ‘T•VÚ[™ÝÈÚ[™ÙH™]ÙY[ˆ™XÛÜ™[™ÝX›Z]
Šˆ‚ˆŠ	K‰IJK	[HZ\ˆ^[Ú[™ÝË	[HÙˆ[H[ˆH“Ò‘PÕSÓˆ‚ˆŠÌ‹˜ÌÊK	[HÛˆHÛÝÚ\™YXÜ›ÜÜÈ“ÕSTÎÈ	[HÙˆ	[Hœ˜[Y\È‚ˆ™\Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü˜XÙQ˜]ÜË
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü˜XÙQœ˜[Y\Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü˜XÙUœÐÚ[™ÙYˆ×Ü˜XÙQ˜]ÜÈÈLŒ
ˆÝX›J×Ü˜XÙUœÐÚ[™ÙY
HÈÝX›J×Ü˜XÙQ˜]ÜÊHˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü˜XÙTÐÚ[™ÙYˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü˜XÙT›ÚÚ[™ÙYˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü˜XÙPÜ›ÜÜÕ[Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü˜XÙQ\Qœ˜[Y\Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü˜XÙQœ˜[Y\ÊNÂˆœš[ŠÝ\œ‹ˆ–Ýš×H
ŠˆQ‘‘PÕQ
H[X™\ˆ]XÚY\ÊNˆ	[H˜]ÜÈ™XYH‚ˆœ™YÚ\Ý\ˆRTˆÕÓˆÚY\ˆ\Ù\È][Ý™YY\ˆ^HÙ\™H™XÛÜ™Y‚ˆŠ	K‰IJK	[HÙˆÜÙH[ˆH›Ú™XÝ[Û‹	[HÙˆ	[Hœ˜[Y\ËˆHÚ[™ÙH‚ˆš[ˆ™YÚ\Ý\œÈH˜]È™]™\ˆ™XYÈ\ÈHØ]\ˆÛÜšÚ[™È\È\ÚYÛ™Y—ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü˜XÙPY™™XÝYˆ×Ü˜XÙQ˜]ÜÈÈLŒ
ˆÝX›J×Ü˜XÙPY™™XÝY
HÈÝX›J×Ü˜XÙQ˜]ÜÊHˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü˜XÙPY™™XÝY›Ú‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü˜XÙPY™™XÝYœ˜[Y\Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü˜XÙQœ˜[Y\ÊNÂˆYˆ
Y×Ü˜XÙPY™™XÝY
Bˆœš[ŠÝ\œ‹ˆ–Ýš×HÓPSˆ8 %]HÛX[ˆ[ˆYX[œÈ›Ý[™È[[HÚ\ÛÛˆ\›H‚ˆš\È™Y[ˆÙY[ˆÈØÜ™X[Nˆ™K\[ˆÚ]Ö—Õ’×ÐÓÓ”ÕÔPÑWÔÒTÓÓLH[™‚ˆ˜ÛÛ™š\›H]™\ÜÈ
ÛÝÚHÌ
WˆŠNÂˆBˆYˆ
Y×Ü™\ÚYYPžTÚY\‹™[\J
JBˆÂˆÝŽ™XÝÜÝŽœZ\Z[ÝÛÛœÝ›ÚY
ˆŽÂˆ›Üˆ
ÛÛœÝ]]ÉˆÝˆˆ×Ü™\ÚYYPžTÚY\ŠBˆ‹œ\ÚØ˜XÚÊÈÝ‹œÙXÛÛ™Ý‹™š\œÝJNÂˆÝŽœÛÜ
‹œ˜™YÚ[Š
K‹œ™[™

JNÂˆœš[ŠÝ\œ‹ˆ–Ýš×HÚY\œÈÚÜÙHTÕÛZ]ÈÛÛYHÙˆÌ‹˜ÌÈ8 %	^HÙˆ[KˆÚ[˜ÙH‚ˆœ\ÍHØ]\ˆÛÜY\ÈÌ‹˜ÌÈ[˜ÛÛ™][Û˜[KÛÈ\ÙH\™H›ÈÛ™Ù\ˆ‚ˆœ™XY[™È™\ÚYYNÈHÙ[œÝ\ÈÝ^\È™XØ]\ÙH]\ÈÚ]›Ý[™HY™XÝ—ˆ‹ˆ‹œÚ^™J
JNÂˆ›Üˆ
Ú^™WÝHHÈH‹œÚ^™J
H	‰ˆHMŽÈJÊÊBˆÂˆËÈ™\ÛÛ™HHÚ[\ˆ˜XÚÈÈHÚY\‰ÜÈTÒÚXÚ\ÈHÛ›HY[]BˆËÈ\È™[™\™\ˆÙY\È[™HÛ™HHœÜ‹Ë›Y]KšœÛÛˆš[\È\™H˜[YYžK‚ˆZ[Ý\ÚHÂˆ›Üˆ
Ú^™WÝÈHÈÈ‹OœÚY\œË˜[ËœÚ^™J
NÈÊÊÊBˆYˆ
	”‹OœÚY\œË˜[ÖÚ×HOH–ÚWKœÙXÛÛ™
BˆÂˆ\ÚH‹OœÚY\œËšÙ^\ÖÚ×NÂˆœ™XZÎÂˆBˆœš[ŠÝ\œ‹–Ýš×H	LM›	LLH˜]Ü×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊZ\Ú
[œÚYÛ™YÛ™ÈÛ™Ê]–ÚWK™š\œÝ
NÂˆBˆBˆYˆ
×Ü]Ú™\ÚYYH×Ü]ÚØ]\™Y
Bˆœš[ŠÝ\œ‹ˆ–Ýš×H›Ú™XÝ[Ûˆ]Ú[œ]Îˆ	[HÚ[™ÝÜÈYÌ‹˜ÌÈ[ˆHÚY\‰ÜÈ‚ˆ“ÕÓˆ\Ý	[HY›Ý
HØ]\ˆÛÜY\ÈÌ‹˜ÌÈ™YØ\™\ÜÈÚ[˜ÙH\‚ˆÍÛÈ›Ý\™H™X[˜[Y\È›ÝÊH8 %›Ú™XÝ[Ûˆ™XÛÙÛš^™Y[ˆ	[HÙˆH‚ˆ›]\ˆ
	KŒ™‰IJWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü]ÚØ]\™Yˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü]Ú™\ÚYYKˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü]Ú™\ÚYYT™XÛÙÛš^™Yˆ×Ü]Ú™\ÚYYBˆÈLŒ
ˆÝX›J×Ü]Ú™\ÚYYT™XÛÙÛš^™Y
HÈÝX›J×Ü]Ú™\ÚYYJBˆˆŒ
NÂˆYˆ
×Ü]ÚÜ˜ÐÚXÚÙY
Bˆœš[ŠÝ\œ‹ˆ–Ýš×H›Ú™XÝ[Ûˆ]Ú
ØXÚYÛÝ\˜ÙK\ÍJNˆ	[HÚXÚÙY‚ˆŠŠ‰[HTÐQÔ‘QQÚ]]Ú[™ÈH\™[˜HÛÜJŠˆ
	K‰IJI\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü]ÚÜ˜ÐÚXÚÙYˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü]ÚÜ˜Ð˜YˆLŒ
ˆÝX›J×Ü]ÚÜ˜Ð˜Y
HÈÝX›J×Ü]ÚÜ˜ÐÚXÚÙY
Kˆ×Ü]ÚÜ˜Õ™\šYžTÚ\ÛÛˆÈˆÔÒTÓÓˆT“QQ8 %H™\›È\™H\ÈH“S‘‚ˆ™\šYšY\—HˆˆˆŠNÂˆYˆ
×ÜÚÞQœ˜[Y\ÊBˆœš[ŠÝ\œ‹ˆ–Ýš×HÚÞH\Þ[[Y]žH
Ö—Õ’×ÔÒÖWÐTÖSJNˆ	[Hœ˜[Y\ËYX[ˆTŸ	KŒÙ‹‚ˆ›YX[ˆœ˜[YK]ËYœ˜[YHÝ\	KŒÙ‹X^Ý\	KŒÙ‹
Š‰[HÒQÓˆ“TÈ‚ˆŠ	KŒÙ‰IHÙˆœ˜[Y\ÊJŠ—ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÜÚÞQœ˜[Y\Ë×ÜÚÞPXœÔÝ[HÈÝX›J×ÜÚÞQœ˜[Y\ÊKˆ×ÜÚÞQœ˜[Y\ÈˆHÈ×ÜÚÞTÝ\Ý[HÈÝX›J×ÜÚÞQœ˜[Y\ÈHJHˆŒˆ×ÜÚÞTÝ\X^
[œÚYÛ™YÛ™ÈÛ™ÊY×ÜÚÞQ›\ËˆLŒ
ˆÝX›J×ÜÚÞQ›\ÊHÈÝX›J×ÜÚÞQœ˜[Y\ÊJNÂˆËÈT‘ˆUSHÉÜÈš[Ûˆ]™\žHÝ]È[\8 %Hž]\È“ÕÛÜYYÚXÚ\ÈHÚÛBˆËÈÚ[ÙˆH][K\ÈHÛÈ[X™\œÈ]Ø^HÚ]\ˆ]\ÈØY™NˆÝÈX[žHÝYÙ\ÂˆËÈ™[˜XÚÈÈH[ÛÜK[™Ú]H™\šYšY\ˆ›Ý[™Yˆ]Ø\È\›YY‚ˆÂˆÛÛœÝZ[ÝÝH×ÙØ]\‘[
È×ÙØ]\‘Ø]\™Y
È×ÙØ]\‘[›Ý[™YÂˆYˆ
Ý
BˆÂˆÛÛœÝZ[ÝÛÝ[™HHÝ
ˆMˆ
ˆÂˆÛÛœÝZ[ÝXÝX[Bˆ×ÙØ]\‘ÛÜ™Ñ[
È×ÙØ]\‘ÛÜ™ÐÛÜYY
È×ÙØ]\‘ÛÜ™Ñ[›Ý[™YÂˆœš[ŠÝ\œ‹ˆ–Ýš×HÛÛœÝØ]\Žˆ	KŒY‰IHÙˆÚ[™ÝÈÛÜY\ÈØ]\™Y
	[H[8 %‚ˆ™[˜[ZXÈLÜˆ›È\Ý
K	KŒ™ˆÐˆ›ÝÛÜYYÝ™\ˆH[ˆ
	KŒY‰IHÙˆ‚ˆ‰KŒ™ˆÐŠWˆ‹ˆLŒ
ˆÝX›J×ÙØ]\‘Ø]\™Y
HÈÝX›JÝ
Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙØ]\‘[ˆÝX›JÛÝ[™HHXÝX[
H
ˆŒÈYNKˆLŒ
ˆÝX›JÛÝ[™HHXÝX[
HÈÝX›JÛÝ[™JKˆÝX›JÛÝ[™JH
ˆŒÈYNJNÂˆËÈ\	ÜÈYXÚ[š\ÛH[X™\ŽˆÝÈX[žH[˜[ZXÈÛÜY\ÈÛÚÈH›Ý[™Y]ˆËÈ[™Ú]^HXÝX[H[Ý™YYØZ[œÝHÐˆXXÚÛÜÝ™Y›Ü™KˆBˆËÈÝ\LÈ™XY[™È[HØ[È\È™\ÚYHHœ˜[YH[YK™XØ]\ÙH]BˆËÈ›Ý]IÜÈ0¬L‹ŽIH›ÛÜˆHž]HÛÝ[\ÈH[X™\ˆ]Ø[››Ý™H\™ÝYYˆËÈÚ]‚ˆYˆ
×ÙØ]\‘[›Ý[™Y
Bˆœš[ŠÝ\œ‹ˆ–Ýš×H›Ý[™Y[˜[ZXÎˆ	[HÛÜY\È[Ý™Y	KŒ™ˆÐˆÚ\™H[‚ˆ˜ÛÜY\ÈÙ\™H	KŒ™ˆÐˆ
IKŒY‰IJNÈ	[H[˜[ZXÈÛÜY\ÈÝ[[ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙØ]\‘[›Ý[™YˆÝX›J×ÙØ]\‘ÛÜ™Ñ[›Ý[™Y
H
ˆŒÈYNKˆÝX›J×ÙØ]\‘[›Ý[™Y
H
ˆM‹ŒÈYNKˆLŒ
ˆ
KŒHÝX›J×ÙØ]\‘ÛÜ™Ñ[›Ý[™Y
HÂˆ
ÝX›J×ÙØ]\‘[›Ý[™Y
H
ˆM‹Œ
ˆŒ
JKˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙØ]\‘[˜[ZXÊNÂˆËÈ\][H‰ÜÈYXÚ[š\ÛH[X™\ŽˆH™XÛÙÛš][ÛˆÛÜšÈHY[[È™[[Ý™YˆËÈ\È]È]Ú\™NÈH™\šYžH[™H™\ÚYH]\ÈÚ]XZÙ\ÈHÚ\™HØY™BˆËÈÈ™[Y]™K‚ˆYˆ
×Ü]ÚY[[Ò]È
È×Ü]ÚY[[ÓZ\ÜÙ\ÊBˆœš[ŠÝ\œ‹ˆ–Ýš×H]ÚY[[Îˆ	KŒY‰IHÙˆ	[H”È]Ú\ÈÙ\™Yœ›ÛHH‚ˆ]Ø^HT•H
	[HZ\ÜÙ\ÊWˆ‹ˆLŒ
ˆÝX›J×Ü]ÚY[[Ò]ÊHÂˆÝX›J×Ü]ÚY[[Ò]È
È×Ü]ÚY[[ÓZ\ÜÙ\ÊKˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ×Ü]ÚY[[Ò]È
È×Ü]ÚY[[ÓZ\ÜÙ\ÊKˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü]ÚY[[ÓZ\ÜÙ\ÊNÂˆYˆ
×Ü]ÚY[[ÐÚXÚÙY
Bˆœš[ŠÝ\œ‹ˆ–Ýš×H]ÚY[[È™\šYšYYˆ	[H]È™K\]ÚY[™‚ˆ˜ÛÛ\\™Y
Š‰[H\ØYÜ™YY
Š‰\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü]ÚY[[ÐÚXÚÙYˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü]ÚY[[Ð˜Yˆ]ÚY[[Õ™\šYžTÚ\ÛÛŠ
BˆÈˆ
ÒTÓÓ‘Q8 %H™\›È\™HYX[œÈH™\šYšY\ˆ\È“S‘
H‚ˆˆˆŠNÂˆYˆ
×ÙØ]\ÚXÚÙY
Bˆœš[ŠÝ\œ‹ˆ–Ýš×H™\šYšYYˆ	[HØ]\œÈÚXÚÙYYØZ[œÝH[ÛÜK‚ˆŠŠ‰[H\ØYÜ™YY
Š‰\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙØ]\ÚXÚÙYˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙØ]\˜YˆÛÛœÝØ]\”Ú\ÛÛŠ
BˆÈˆ
ÒTÓÓ‘Q8 %H™\›È\™HYX[œÈH™\šYšY\ˆ\È“S‘
H‚ˆˆˆŠNÂˆYˆ
×ÙØ]\ÚXÚÙY	‰ˆ×ÙØ]\‘[›Ý[™Y
Bˆœš[ŠÝ\œ‹ˆ–Ýš×H‹‹›ÙˆÚXÚ	[HÙ\™H“ÕS‘Q[˜[ZXÈÛÜY\É\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙØ]\˜Y›Ý[™YˆÛÛœÝØ]\”Ú\ÛÛŠ
BˆÈˆ
Ú\ÛÛ™Yˆ™\›È\™HYX[œÈH“ÕS‘Q™\šYšY\ˆ\È›[™
H‚ˆˆˆŠNÂˆBˆBˆËÈ\Ý\	ÜÈ™\™XÝÒS‘ÕÈ˜]\ÈÛ›H
ÛÝÚHŽˆHÝ[][]]™HYX[ˆ\ÈBˆËÈ˜[œÚY[
KˆXXÚš[ÛÝ™\œÈHÛÜY\ÈÚ[˜ÙHH™]š[Ý\ÈÛ™K‚ˆYˆ
×Ü[]PÙ[œÝ\ÊBˆÂˆÛÛœÝ[Ù[œÝ\ÎŽ•Ý	ˆH[Ù[œÝ\ÎŽÂˆ[Ù[œÝ\ÎŽ•Ý	ˆH[Ù[œÝ\ÎŽ›\ÝÂˆÛÛœÝZ[ÝˆH˜ÛÜY\ÈH˜ÛÜY\ÎÂˆYˆ
ŠBˆÂˆÛÛœÝZ[Ý[H˜ž]\Ñ[H˜ž]\Ñ[ÂˆÛÛœÝZ[Ý›™H˜ž]\Ð›Ý[™YH˜ž]\Ð›Ý[™YÂˆÛÛœÝZ[ÝÐˆH˜ž]\ÒYÚØ]\ˆH˜ž]\ÒYÚØ]\ŽÂˆœš[ŠÝ\œ‹ˆ–Ü[Ù[œÝ\×H	[H[˜[ZXÈ”ÈÛÜY\ÈÚ[˜ÙH\Ý[™NˆÛX[‹XÛÝ™\ˆ‚ˆ‰KŒY‰IK\KY˜[˜XÚÈ	KŒY‰IK™]\ÙH	KŒY‰INÈ™]™\‹X›Ý[™	[K‚ˆÚ[™ÝË[[Ý™Y	[NÈ\œÝËØÛÜHÛÝ™\ˆ	KŒ™ˆ\X[	KŒ™—ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê[‹ˆLŒ
ˆÝX›J˜ÛX[ÛÝ™\ˆH˜ÛX[ÛÝ™\ŠHÈÝX›JŠKˆLŒ
ˆÝX›J™\Q˜[˜XÚÈH™\Q˜[˜XÚÊHÈÝX›JŠKˆLŒ
ˆÝX›Jœ™]\ÙHHœ™]\ÙJHÈÝX›JŠKˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ›™]™\›Ý[™H›™]™\›Ý[™
Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊJÚ[™ÝÓ[Ý™YHÚ[™ÝÓ[Ý™Y
KˆÝX›J˜ÛÝ™\\œÝÈH˜ÛÝ™\\œÝÊHÈÝX›JŠKˆÝX›Jœ\X[\œÝÈHœ\X[\œÝÊHÈÝX›JŠJNÂˆœš[ŠÝ\œ‹ˆ–Ü[Ù[œÝ\×Hž]\Îˆ[	KŒYˆPˆOˆ›Ý[™Y	KŒYˆPˆ
ÐU‘TÈ	KŒY‰INÈ‚ˆšÚ[Ì	IJHYÚ]Ø]\‹[Û›H[Ù[	KŒYˆPˆ
Ø]™\È	KŒY‰IJK‚ˆšYÚ]Ø]\ˆÉ]KYX[ˆ›Ý[™	KŒYˆ™YÜ×ˆ‹ˆÝX›J[
HÈYM‹ÝX›J›™
HÈYM‹ˆ[ÈLŒ
ˆÝX›J[H›™
HÈÝX›J[
HˆŒˆÝX›JÐŠHÈYM‹ˆ[ÈLŒ
ˆÝX›J[HÐŠHÈÝX›J[
HˆŒˆ˜]ÕœÔ[]RYÚØ]\Š
KˆÝX›J™^[Ý[HH™^[Ý[JHÈÝX›JŠJNÂˆÝŽœÝš[™ÈÈH–Ü[Ù[œÝ\×H›Ý[™\ÝÙÜ˜[H
™YÜÊNˆŽÂˆ›Üˆ
[ˆHÈˆÌŽÈ
ÊØŠBˆÂˆÛÛœÝZ[ÝÈHš\ÝØ—HHš\ÝØ—NÂˆYˆ
XÊBˆÛÛ[YNÂˆÚ\ˆY–ÍNÂˆÛœš[ŠY‹Ú^™[ÙˆY‹ˆ	YIY‰[H‹ˆ
ˆˆ
ˆ
ÈËˆ
[œÚYÛ™YÛ™ÈÛ™ÊXÊNÂˆÈ
ÏHYŽÂˆBˆœš[ŠÝ\œ‹‰\×ˆ‹Ë˜×ÜÝŠ
JNÂˆHÂˆBˆBˆËÈHÔ‘TˆÐUIÜÈ™\™XÝÛˆ]™\žHÝ]È[\ˆHØ]HÚÜÙH™\Ý[\È›Ýš[Y\ÂˆËÈHY™XÝ\È›Ú™XÝÙY\È™Y\ØÛÝ™\š[™È
\M‰ÜÈÝ[˜Ú[ÚÚ\ÛÝ[\ˆØ\ÂˆËÈÛÛXÝY›ÜˆšYY[ˆ\È[™š[YžH›Ý[™ÊK[™\ÈÛ™HÝX\™ÈH\™Ù\ÝˆËÈ[™š\ÚÚY\Ý][H[ˆH[‹‚ˆYˆ
Ü™\‘Ø]P\›YY

JBˆœš[ŠÝ\œ‹ˆ–ÛÜ™\—H˜]Ë[Ü™\ˆØ]Nˆ	[Hœ˜[Y\ÈÚXÚÙY
Š‰[HRSQ
Š‹‚ˆ‰[H˜]ÜÈÙÙÙY	\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O›Ü™\‘œ˜[Y\ÐÚXÚÙYˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O›Ü™\‘œ˜[Y\Ñ˜Z[Yˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O›Ü™\‘˜]ÜÓÙÙÙYˆ[ŠÖ—Õ’×ÓÔ‘T—ÔÒTÓÓˆŠBˆÈˆ
ÒTÓÓ‘Q8 %H™\›È\™HYX[œÈHØ]H\È“S‘
H‚ˆˆ
ˆ	‰ˆ‹Oœ\”™XÂˆÈˆ
TSS™XÛÜ™[™ÎˆH™\^YY[œÝ[˜Ù\ÉÈÝÛˆYÈ‚ˆ˜YØZ[œÝHØ\\™HÜ™\ˆ8 %™\›È\ÈHÛZ[JH‚ˆˆˆ
Ù\šX[™XÛÜ™[™Îˆ™\›È\ÈHÛ›HÛÜœ™XÝ™\Ý[
HŠJNÂˆËÈ\Ìˆ][HKˆš[YT‘H\ÈÙ[\ÈÛˆHÙ[œÝ\ÉÜÈÝÛˆØY[˜ÙKÛÈHÛØZÂˆËÈ][™ÈÙ™ˆHŒYœ˜[YH›Ý[™\žHÝ[[™ÈH[X™\ˆ8 %HØ[YHY™XÝBˆËÈÝ[˜Ú[ÚÚ\ÛÝ[\ˆY›ÜˆšYY[ˆ\È
ÛÛXÝYÚ[˜ÙH\M‹š[YžBˆËÈ›Ý[™ÊKˆ[™\[™Ú[[[›\ÜÈHÙ[œÝ\È\È\›YY‚ˆšÔ™[™\™\—Ñ[\™\XØ[Ø\ÝJ
NÂˆËÈHÛÛœÝ[Y[[Ëš[YÛˆU‘T–H[ˆ[™›ÝÛ›H[™\ˆH›Ùš[\ˆ8 %BˆËÈÜ\˜]Ü‰ÜÈKÐˆ\›™\ÜÈ[X™\˜][H[œÈÚ]Ý]Ö—Õ’×Ô“Ñ’SX
]ÛÜÝÈ‹M\ÈBˆËÈœ˜[YH[™ÛÝ[Ú[™ÙHH[™È™Z[™ÈYÙY
KÛÈÚ]Ý]\È[™HHÛØZÈÛÝ[ˆËÈ›ÝØ^HÚ]\ˆH\›H[™ØYÙY][
ÛÝÚHMLJKˆ[ˆÝ[Ë›ÝHÚ[™ÝË‚ˆÂˆÛÛœÝZ[ÝÝH×ØÛÛœÝY[[Ô[’]È
È×ØÛÛœÝY[[Ô[“Z\ÜÙ\ÎÂˆYˆ
Ý
Bˆœš[ŠÝ\œ‹ˆ–Ýš×HÛÛœÝY[[Îˆ	KŒY‰IHÙˆ[‹XÛÜY\ÈÙ\™Y
”È	KŒY‰IKÈ‚ˆ‰KŒY‰IJK	KŒYˆÐˆ›ÝÛÜYYÝ™\ˆH[‰\×ˆ‹ˆLŒ
ˆÝX›J×ØÛÛœÝY[[Ô[’]ÊHÈÝX›JÝ
KˆŒŒ
ˆÝX›J×ØÛÛœÝY[[Ô[•œÒ]ÊHÈÝX›JÝ
KˆŒŒ
ˆÝX›J×ØÛÛœÝY[[Ô[”Ò]ÊHÈÝX›JÝ
KˆÝX›J×ØÛÛœÝY[[Ô[’]ÊH
ˆM‹ŒÈLÌÍÍNŒˆ×ØÛÛœÝY[[ÓÙ™ˆÈˆÐÖ—Õ’×Ó“×ÐÓÓ”ÕÓQSSÎˆHÛÜH˜[ˆ]™\žH˜]×HˆˆˆŠNÂˆB‚ˆËÈHÛYÛÛˆÙ™œÙ]š[YÛˆ]™\žH[Žˆ[ˆ\›HÚ]›ÈÛÝ[\ˆØ[››Ý™HÚÝÛˆÂˆËÈ]™H[™ØYÙY
ÛÝÚHMLJK[™\ÈÛ™IÜÈY™™XÝ\ÈHYÙ[Y[X›Ý]HXÝ\™K‚ˆœš[ŠÝ\œ‹–Ýš×HÝ[˜Ú[\Ýˆ	[H˜]ÜÈ[˜X›Y]	\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÜÝ[˜Ú[˜]ÜËˆ[“ÛŠÖ—Õ’×Ó“×ÔÕSÒSŠHÈˆÐÖ—Õ’×Ó“×ÔÕSÒSˆ›Û™HÙ\™HÛ›Ý\™YHˆˆˆŠNÂˆœš[ŠÝ\œ‹–Ýš×HÛYÛÛˆÙ™œÙ]ˆ	[H˜]ÜÈ\ÚÙY›ÜˆÛ™I\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÜÛSÙ™œÙ]˜]ÜËˆ[“ÛŠÖ—Õ’×Ó“×ÔÓWÓÑ‘”ÑUŠHÈˆÐÖ—Õ’×Ó“×ÔÓWÓÑ‘”ÑUˆ›Û™HÙ\™H\YYH‚ˆˆˆŠNÂ‚ˆËÈT•ÌIÔÈTSS‘KPÔ‘PUSÓˆÑS”ÕTËš[YÛˆ]™\žH[‹ˆHÔQ”SQHX›H\ÂˆËÈHÚ[›ÝHÝ[Îˆ›Ý\ˆÙXÛÛ™ÈÙˆÛÛ\[[™ÈÜ™XYÝ™\ˆŒœ˜[Y\È\ÂˆËÈ[š\ÚX›HÈH^Y\ˆ[™›Ý\ˆÙXÛÛ™È[œÚYHÛ™Hœ˜[YH\ÈH[™ÈHÜ\˜]Ü‚ˆËÈ™\ÜY[™Û›HH\‹Yœ˜[YH›Û]\Ù\\˜]\È[K‚ˆÂˆ\Qœ˜[YQ›\Ú

NÈËÈ[˜ÛYHHœ˜[YH[ˆ›ÙÜ™\ÜÂˆœš[ŠÝ\œ‹ˆ–Ýš×H\[[™HÜ™X][ÛŽˆ	[H\[[™\Ë	KŒYˆ\ÈÝ[ÛÜœÝÚ[™ÛH‚ˆ‰KŒYˆ\Èœ˜[YH	[I\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü\PÛÝ[ÝX›J×Ü\SœÊHÈYM‹ˆÝX›J×Ü\UÛÜœÝœÊHÈYM‹
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü\UÛÜœÝœ˜[YKˆ‹Oœ\PØXÚHOH’×Ó•SÒS‘HÈˆÛ›È\[[™HØXÚWHˆˆˆŠNÂˆËÈÛÜYžHÛÜÝÛÜœÝš\œÝ8 %HÝ[H^Y\ˆ™[\ÈH”SQKÛÈ\È\ÂˆËÈHX›H][™\È\Ú]HÙœ×XÚ[™ÝÉÜÈÛÜœÝ‚ˆ\Qœ˜[YT™XÈÜÌL—NÂˆÝŽ˜ÛÜJÝŽ˜™YÚ[Š×Ü\UÜ
KÝŽ™[™
×Ü\UÜ
KÝŽ˜™YÚ[ŠÜ
JNÂˆÝŽœÛÜ
ÝŽ˜™YÚ[ŠÜ
KÝŽ™[™
Ü
Kˆ×H
ÛÛœÝ\Qœ˜[YT™XÉˆKÛÛœÝ\Qœ˜[YT™XÉˆŠHÈ™]\›ˆK›œÈˆ‹›œÎÈJNÂˆ›Üˆ
ÛÛœÝ\Qœ˜[YT™XÉˆˆÜ
BˆYˆ
˜ÛÝ[
Bˆœš[ŠÝ\œ‹–Ýš×Hœ˜[YH	NNˆ	MËŒYˆ\ÈZ[[™È	]H\[[™JÊWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê]™œ˜[YKÝX›J›œÊHÈYM‹˜ÛÝ[
NÂˆB‚ˆËÈT•ÌIÜÈÛÚÈ›Ûš[YÛˆU‘T–H[ˆ›ÜˆHØ[YH™X\ÛÛˆ\ÈHÛÈX›Ý™NˆBˆËÈÜ\˜]Ü‰ÜÈKÐˆ\›™\ÜÈ[œÈÚ]Ý]H›Ùš[\‹ÛÈ\È[™H\ÈHÛ›H[™ÂˆËÈ]Ø[ˆØ^HÚ]\ˆH\›H[™ØYÙYˆHQS•UHÐUHTÈT‘HÓÈ8 %Ú]•Ó‚ˆËÈ›Ý›ÛY[X™\œÈ]\Ý™XY™XØ]\ÙHHÛÜ™\ÈHÔˆÙˆ]™\žHÛÚÉÜÈÝÛ‚ˆËÈ\›H[™Ø[››Ý™H˜[ÙHÚ[H[žHÙˆ[HÛÝ[ÈÛÜšË‚ˆÂˆÛÛœÝZ[ÝH×ÚÛÚÑ›Û›ÛY
È×ÚÛÚÑ›Û]™NÂˆÛÛœÝZ[ÝˆH×ÚÛÚÑ›Û™]Ú›ÛY
È×ÚÛÚÑ›Û™]Ú]™NÂˆœš[ŠÝ\œ‹ˆ–Ýš×HÛÚÈ›Ûˆ	[HÙˆ	[H˜]ÜÈ›ÛY
	KŒY‰IJK	[HÙˆ	[H‚ˆ˜]\ËY™]ÚXÛÙ\È›ÛY
	KŒY‰IJI\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÚÛÚÑ›Û›ÛY
[œÚYÛ™YÛ™ÈÛ™ÊYˆÈLŒ
ˆÝX›J×ÚÛÚÑ›Û›ÛY
HÈÝX›J
HˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÚÛÚÑ›Û™]Ú›ÛY
[œÚYÛ™YÛ™ÈÛ™ÊY‹ˆˆÈLŒ
ˆÝX›J×ÚÛÚÑ›Û™]Ú›ÛY
HÈÝX›JŠHˆŒˆ[“ÛŠÖ—Õ’×Ó“×ÒÓÒ×Ñ“ÓŠBˆÈˆÐÖ—Õ’×Ó“×ÒÓÒ×Ñ“ÓˆH™K\\MÌH\‹Y˜]ÈØ[×H‚ˆˆˆŠNÂˆB‚ˆËÈH›]X›\ÉÈÜ›ÝÈš[š[YÛˆU‘T–H[ˆ˜]\ˆ[ˆÛ›H[™\ˆBˆËÈ›Ùš[\ˆ8 %H^HÙ\ÜÚ[ÛˆHÜ\˜]Üˆš]™\È\È›È›Ùš[\‹[™]\È^XÝBˆËÈH[ˆÚ\™HH]ÚÙ]È™\ÜY‚ˆœš[ŠÝ\œ‹ˆ–Ýš×H›]ØXÚHÜ›ÝÜÎˆ	[K	KŒ™ˆ\ÈÝ[ÛÜœÝ	KŒ™ˆ\É\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ù›]Ü›ÝÜËÝX›J×Ù›]Ü›ÝÓœÊHÈYM‹ˆÝX›J×Ù›]Ü›ÝÕÛÜœÝœÊHÈYM‹ˆ×Ù›]ØXÚSÙ™ˆÈˆÐÖ—Õ’×Ó“×Ñ“UÐÐPÒNˆ›È›]X›HØ\È[ˆ\ÙWHˆˆˆŠNÂˆËÈHÕ‘PSHÕÔ‘IÔÈÔ“ÕÕË™\ÚYHH›]ØXÚIÜÈ›ÜˆHØ[YH™X\ÛÛŽˆ›Ý\™BˆËÈ˜\™K›Ý\[ˆ[\™[H[œÚYHÛ™Hœ˜[YK[™H˜\™HÚ[™ÛKYœ˜[YHÛÜÝ\È^XÝBˆËÈÚ]H[ˆYX[ˆØ[››ÝÙYH[™H^Y\ˆØ[‹ˆ\ÎIÜÈÜ\˜]ÜˆÙ\ÜÚ[ÛˆYÛË[™ˆËÈXXÚØ\ÈHÛÜœÝœ˜[YHÙˆ]ÈÝÛˆL\ÙXÛÛ™Ú[™ÝË‚ˆYˆ
×Ü\œÚ\ÝÜ›ÝÓŠBˆœš[ŠÝ\œ‹ˆ–Ýš×HÝ™X[HÝÜ™HÜ›ÝÜÎˆ	[K	KŒYˆ\ÈÝ[	KŒYˆ\ÈXXÚ8 %XXÚÛ™H\È‚ˆ˜HÚÛHœ˜[YHÙˆST[YH[™]\ÈH]ÚÛ\ÜÈ0©Í™H0©ÌÈ˜[Y\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü\œÚ\ÝÜ›ÝÓ‹ÝX›J×Ü\œÚ\ÝÜ›ÝÓœÊHÈYM‹ˆÝX›J×Ü\œÚ\ÝÜ›ÝÓœÊHÈYMˆÈÝX›J×Ü\œÚ\ÝÜ›ÝÓŠJNÂˆYˆ
×ØØÐÛÜY\ÊBˆÂˆœš[ŠÝ\œ‹ˆ–Ýš×HÛÜHÙ[œÝ\Îˆ	[H™\ÛÛ™HÛÜY\Ë	[HPQ
	KŒY‰IJH8 %‚ˆ‰KŒ™ˆÙˆ	KŒ™ˆÜ^[XY
	KŒY‰IJK	[HØ[\HX\šÜÎÈXYHH‚ˆœØ[YH
Û˜\ÚÝ™XÝ
HÛÜYYYØZ[ˆÚ]›ÈÛÛœÝ[Y\ˆ[ˆ™]ÙY[‹‚ˆ˜ÛÝ[YÛÛœÙ\˜]]™[HÝØ\™U‘Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ØØÐÛÜY\Ë
[œÚYÛ™YÛ™ÈÛ™ÊY×ØØÑXYˆLŒ
ˆÝX›J×ØØÑXY
HÈÝX›J×ØØÐÛÜY\ÊKˆÝX›J×ØØÑXY^[ÊHÈYNKÝX›J×ØØÔ^[ÊHÈYNKˆ×ØØÔ^[ÈÈLŒ
ˆÝX›J×ØØÑXY^[ÊHÈÝX›J×ØØÔ^[ÊHˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ØØÔØ[\SX\šÜÊNÂˆËÈHÙ^\ÈØ\œžZ[™ÈHXY^[ËÛÈH™\™XÝ˜[Y\ÈÝ\™˜XÙ\Ë›ÝHÚ\™K‚ˆÝŽ™XÝÜÝŽ\OZ[ÝZ[ÝZ[Ì—ÝÚ^™WÝˆžQXYÂˆ›Üˆ
ÛÛœÝ]]ÉˆÝˆˆ×ØØÓX\
BˆÂˆZ[ÝHˆHÂˆ›Üˆ
ÛÛœÝ]]ÉˆˆˆÝ‹œÙXÛÛ™
BˆÂˆ
ÏH‹™XYÂˆˆ
ÏH‹™XYŽÂˆBˆžQXY™[\XÙWØ˜XÚÊ‹Ý‹™š\œÝÝ‹œÙXÛÛ™œÚ^™J
JNÂˆBˆÝŽœÛÜ
žQXYœ˜™YÚ[Š
KžQXYœ™[™

JNÂˆ›Üˆ
Ú^™WÝHHÈHžQXYœÚ^™J
H	‰ˆHLÈ
ÊÚJBˆœš[ŠÝ\œ‹ˆ–Ýš×HÛÜHÙ[œÝ\Îˆ	L	\È	[HXYÛÜY\Ë	KŒ™ˆÜ^[XY‚ˆ‰^H\Ý[˜Ý™XÝ×ˆ‹ˆÝŽ™Ù]ŠžQXYÚWJH	ˆQ‘‘‘‘‘‘‹ˆ
ÝŽ™Ù]ŠžQXYÚWJH	ˆÔÛ˜\ÚÝ\š]
HÈˆ
\
Hˆˆˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\ÝŽ™Ù]OŠžQXYÚWJKˆÝX›JÝŽ™Ù]ŠžQXYÚWJJHÈYNKÝŽ™Ù]ÏŠžQXYÚWJJNÂˆBˆYˆ
‹Oœ\œÚ\Ý]‹˜Y™™\ŠBˆœš[ŠÝ\œ‹ˆ–Ýš×HÝÜ™HZ\œ›ÜŽˆ	KŒ™ˆP‹Ùœ˜[YHÛÜYYÜÝO•”SH[ˆ	KŒYˆÛÜY\ËÙœ˜[YH‚ˆ›Ý™\ˆH[ŽÈ\œÚ\Ý]È›Ý[™HRT”“Ôˆ	KŒY‰IHÙˆH[YH
	[H]‹‚ˆ‰[HÜÝ
Wˆ‹ˆ‹O™œ˜[YHÈÝX›J‹O›Z\œ›Üž]\ÊHÈÝX›J‹O™œ˜[YJHÈLMÍ‹ŒˆŒˆ‹O™œ˜[YHÈÝX›J‹O›Z\œ›ÜÛÜY\ÊHÈÝX›J‹O™œ˜[YJHˆŒˆ
‹O›Z\œ›Ü’]Ñ]ˆ
È‹O›Z\œ›Ü’]ÒÜÝ
BˆÈLŒ
ˆÝX›J‹O›Z\œ›Ü’]Ñ]ŠHÂˆÝX›J‹O›Z\œ›Ü’]Ñ]ˆ
È‹O›Z\œ›Ü’]ÒÜÝ
BˆˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O›Z\œ›Ü’]Ñ]‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O›Z\œ›Ü’]ÒÜÝ
NÂˆœš[ŠÝ\œ‹–Ýš×H\[[™\ÏI^HÚY\œÏI^H^\™\ÏI^H\™[˜RYÚØ]\I[HÐ—ˆ‹ˆ‹Oœ\[[™\ËœÚ^™J
K‹OœÚY\œÓX\œÚ^™J
K^Ú^™J
Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ‹O˜\™[˜RYÚØ]\ˆˆL
JNÂˆËÈHÝ]HØXÚIÜÈÝÛˆ[™ØYÙ[Y[\Èœ˜XÝ[ÛœÈÙˆH˜]ÜÈ]Ø\ÈÙ™™\™Y‚ˆËÈš[Y[˜ÛÛ™][Û˜[K[˜ÛY[™ÈÛˆHÖ—Õ’×Ó“×ÔÕUWÐÐPÒH\›HÚ\™H]™\žBˆËÈšYÝ\™H\ÈžHÛÛœÝXÝ[Ûˆ8 %[ˆ\›HÚÜÙH›Ûˆˆ[ˆ\È[™\Ý[™ÝZ\ÚX›Hœ›ÛH]ÂˆËÈ›Ù™ˆˆ[ˆ\È^XÝHÚ]HZ\ÜÚ[™ÈÛÝ[\ˆY\È
ÛÝÚHMLJK‚ˆYˆ
ÛÛœÝZ[ÝH‹OœÚÚ\Ë™˜]ÜÊBˆœš[ŠÝ\œ‹ˆ–Ýš×Hš[™ÈÚÚ\Y\ˆ˜]Îˆ\[[™H	KŒY‰IHšY]ÜÜ	KŒY‰IH‚ˆœØÚ\ÜÛÜˆ	KŒY‰IH›[™	KŒY‰IH\ØÜš\Ü‹\Ù]È	KŒY‰IH
Ùˆ	[H˜]ÜÊWˆ‹ˆLŒ
ˆÝX›J‹OœÚÚ\Ëœ\[[™JHÈÝX›J
KˆLŒ
ˆÝX›J‹OœÚÚ\ËšY]ÜÜ
HÈÝX›J
KˆLŒ
ˆÝX›J‹OœÚÚ\ËœØÚ\ÜÛÜŠHÈÝX›J
KˆLŒ
ˆÝX›J‹OœÚÚ\Ë˜›[™
HÈÝX›J
KˆLŒ
ˆÝX›J‹OœÚÚ\ËœÙ]ÊHÈÝX›J
K
[œÚYÛ™YÛ™ÈÛ™ÊY
NÂˆËÈTSS‘PÓÔ‘	ÜÈ[™ØYÙ[Y[]^]SÓÓ‘USÓSÚ[ˆH™X]\™H\ÈÛˆ8 %ˆËÈHÝšÜ›Ù—H[™HÛ›H^\ÝÈ[™\ˆH›Ùš[\‹[™H[YY[ˆ
ÛÜœ™XÝJBˆËÈÙ\È›ÝØ\œžHÛ™KÚXÚYHÝŒÉÜÈš^\›\È›Ý˜X›HÛ›H›ÝYÚBˆËÈÚÚ\YÙÜ™YØ][Û‹ˆ\È[™H\ÈH\™XÝÝ][Y[
ÛÝÚHMLJK‚ˆYˆ
‹Oœ\”™XÊBˆÂˆZ[ÝÚ[šÜÈHÂˆ›Üˆ
Z[Ì—Ý™XÈHÈ™XÈÔ“X^™XÛÜ™\œÎÈ
ÊÜ™XÊBˆÚ[šÜÈ
ÏH×ÜÚ[šÜÔ™XÛÜ™YÜ™X×NÂˆœš[ŠÝ\œ‹ˆ–Ýš×H\˜[[™XÛÜ™ˆ	[HÚ[šÜÈ
	[HžHH[\]HØZ]
K‚ˆ‰[H˜]ÜÈØ\\™Y	[HZ[˜]ÜÈ[ˆ	[HZ[[œÝ[˜Ù\Ë	[H‚ˆ™[\H[œÝ[˜Ù\ËÝX›Z]ØZ]	KŒYˆ\ÈÝ[Ý™\™›ÝËZ[›[™H	[K‚ˆ˜š[™Ý™\™›ÝÈ	[I\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊXÚ[šÜË
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü”[\[Yˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÜØ\\™Y
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü•Z[˜]ÜËˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü•Z[[œÝ[˜Ù\Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü‘[\R[œÝ[˜Ù\ËÝX›J×Ü•ØZ]œÊHÈYM‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü“Ý™\™›ÝÒ[›[™Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Üš[™Ý™\™›ÝËˆ×Üš[™Ý™\™›ÝÈÈˆ
ŠŠˆÐTT‘TÈ•SÐUQ8 %XÝ\™HÝ\ÜXÝ
ŠŠˆˆˆˆŠNÂˆBˆËÈHÑRSS‘È“Ð‘IÜÈÝÛˆ[™ØYÙ[Y[ˆš[YÛ›HÚ[ˆ]š\™Y]š[YÚ]BˆËÈ\‹Y˜]È˜]H˜]\ˆ[ˆH˜]ÈÝ[™XØ]\ÙHH[X™\ˆH][IÜÈ\š]Y]XÂˆËÈ™YYÈ\È™š]™\ˆØ[È\ˆ˜]Èˆ8 %]\ÈÚ]HÙXÛÛ™\žHÛÛ[X[™Y™™\ˆØ\œšY\ËˆËÈ[™HÝ]HØXÚHYX[œÈ]\È›ÝÚ\™H™X\ˆH[ˆØ[ÈHÛÝ\˜ÙHÝYÙÙ\ÝË‚ˆYˆ
×Ü™]\ÙPÙ[œÝ\ÊBˆ™]\ÙXÙ[œÝ\ÎŽ”š[
ˆÚ[˜ÙHH\Ý[™H8 %’SSŠNÂˆËÈHÛÛœÝ[]Üš]H\ÝÙÜ˜[Kš[Y\™HÛÈ][™È™\ÚYHHØ]\ˆÝ]ÂˆËÈÚÜÙH[XÛÜHÜ[][Ûˆ]^\ÝÈÈ^Z[ˆ
\Ë\ÙMK[›Ý\È0©Í™YÊK‚ˆMÑ[\[UÜš]PÙ[œÝ\Ê
NÂˆYˆ
×Ù™]ÚY[[ÐÙ[œÝ\ÊBˆÂˆÛÛœÝZ[ÝˆH×Ù™]ÚY[[Ò]È
È×Ù™]ÚY[[ÓZ\ÜÙ\ÎÂˆœš[ŠÝ\œ‹ˆ–Ýš×H™\^Y™]ÚY[[ÈÙ[œÝ\Îˆ	KŒY‰IHÙˆ	[H˜]ÜÈÛÝ[™HÑT•‘QžH‚ˆŠÚY\‹™]ÚXÛÛœÝ™\œÚ[ÛŠWˆ‚ˆ–Ýš×HZ\ÜÙ\ÎˆÚY\ˆ	[H
	KŒY‰IK[š\™[
H
È™\œÚ[Ûˆ	[H
	KŒY‰IK‚ˆHÝY\ÝÝXÚYH™]Úš[H™]ÙY[ˆÛÈ˜]ÜÈÙˆÛ™HÚY\ŠWˆ‹ˆˆÈLŒ
ˆÝX›J×Ù™]ÚY[[Ò]ÊHÈÝX›JŠHˆŒˆ
[œÚYÛ™YÛ™ÈÛ™Ê[‹
[œÚYÛ™YÛ™ÈÛ™ÊY×Ù™]ÚY[[ÔÚY\“Z\ÜËˆˆÈLŒ
ˆÝX›J×Ù™]ÚY[[ÔÚY\“Z\ÜÊHÈÝX›JŠHˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ù™]ÚY[[Õ™\œÚ[Û“Z\ÜËˆˆÈLŒ
ˆÝX›J×Ù™]ÚY[[Õ™\œÚ[Û“Z\ÜÊHÈÝX›JŠHˆŒ
NÂˆËÈH[X™\ˆHXÚ\Ú[Ûˆ\›œÈÛ‹Ü[YÝ]˜]\ˆ[ˆYÈ™H][\YYˆËÈžH[™ˆHXÛÙH\ÈLœÈH˜]È
0©Í™Xˆ0©ÌÊKÛÈ\È\ÈÚ]H\™™XÝY[[ÂˆËÈÛÝ[™HÛÜ]HØY\È[ˆXÝX[H™XXÚY‚ˆÛÛœÝZ[Ý™HH×Ù™]ÚY[[Ñ^XÝ]È
È×Ù™]ÚY[[Ñ^XÝZ\ÜÙ\ÎÂˆÛÛœÝÝX›H^XÝH™HÈÝX›J×Ù™]ÚY[[Ñ^XÝ]ÊHÈÝX›J™JHˆŒÂˆœš[ŠÝ\œ‹ˆ–Ýš×HVPÕÙ^H
ÚY\ˆ
ÈH\ÚÙˆÛ›HHÛÜ™È\ÈÚY\‰ÜÈ‚ˆ˜]šX]\È™XY
Nˆ	KŒY‰IHÙˆ	[H˜]ÜÈÑT•‘Qˆ‹ˆLŒ
ˆ^XÝ
[œÚYÛ™YÛ™ÈÛ™Ê[™JNÂˆËÈH[X™\ˆHXÚ\Ú[Ûˆ\›œÈÛ‹Ü[YÝ]˜]\ˆ[ˆYÈ™H][\YYˆËÈžH[™ˆHXÛÙH\ÈLœÈH˜]È
0©Í™Xˆ0©ÌÊKÛÈ\È\ÈÚ]XXÚÙ^HÛÝ[™BˆËÈÛÜ]HÜ\˜]Ü‰ÜÈØY‚ˆœš[ŠÝ\œ‹ˆ–Ýš×H]LœËÙ˜]ÈÙˆXÛÙH[™KÌ˜]ÜÎˆÚÛKYš[HÙ^H	KŒÙˆ\Ë‚ˆ‘VPÕÙ^H	KŒÙˆ\È\ˆœ˜[YH8 %YØZ[œÝ[ˆ][KLHÙZ[[™ÈÙˆ‹ŒÌÈ\È]‚ˆ›™YYYÈ™XYÈ[™[]™\™YŒÚ]HYÙ]\È]Ý[™×ˆ‹ˆKŒÈ
ˆLKLÈ
ˆ
ˆÈÝX›J×Ù™]ÚY[[Ò]ÊHÈÝX›JŠHˆŒ
KˆKŒÈ
ˆLKLÈ
ˆ^XÝ
NÂˆBˆËÈHUÒ	ÔÈÕÓˆQPÒS’TÓH•SP‘T‹[˜ÛÛ™][Û˜[8 %]\ÈHÝ]\ÝXÈH][H\ÂˆËÈYÙYÛˆ[™]Ø[››Ý™H\™ÝYYÚ]Ú\™HHœ˜[YH[YHÛˆ\È›Ý]H\ÈBˆËÈ0¬L‹ŽIH›ÛÜ‹ˆÖ—Õ’×Ó“×Ð’S‘ÐUÒLXÚÝ[™XYŒKÍ\™H[™H˜]ÚŒË‚ˆYˆ
×Øš[™˜]Ú˜]ÜÊBˆœš[ŠÝ\œ‹ˆ–Ýš×H™\^š[™Ø[Îˆ	KŒÙˆ\ˆ˜]ÈÝ™\ˆ	[H˜]ÚY˜]ÜÈ
	[H‚ˆšÐÛYš[™™\^Y™™\œÊI\×ˆ‹ˆÝX›J×Øš[™˜]ÚØ[ÊHÈÝX›J×Øš[™˜]Ú˜]ÜÊKˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Øš[™˜]Ú˜]ÜËˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Øš[™˜]ÚØ[Ëˆ×Û›Ðš[™˜]ÚÈˆÐÖ—Õ’×Ó“×Ð’S‘ÐUÒLKÛ™HØ[\ˆš[™[™×HˆˆˆŠNÂˆYˆ
×Ý™\šYžPš[™˜]Ú
Bˆœš[ŠÝ\œ‹ˆ–Ýš×H’S‘UÒ‘T’Q–Nˆ	[HÙˆ	[H
š[™[™ËY™™\‹Ù™œÙ]
Hš\\È‚ˆ‘TÐQÔ‘QQÚ]Ú]H˜]È\ÚÙY›Üˆ
	K‰IJI\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Øš[™™\šYžP˜Yˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Øš[™™\šYžPÚXÚÙYˆ×Øš[™™\šYžPÚXÚÙYˆÈLŒ
ˆÝX›J×Øš[™™\šYžP˜Y
HÈÝX›J×Øš[™™\šYžPÚXÚÙY
BˆˆŒˆ×Ý™\šYžPš[™Ú\ÛÛˆÈˆÔÒTÓÓˆT“H8 %\ÈUTÕ™XYL	WHˆˆˆŠNÂˆYˆ
×Øš[™[Ù[œÝ\È	‰ˆ×Øœ‘˜]ÜÊBˆÂˆËÈH\Ý˜]ÉÜÈ[ˆ\ÈÝ[Ü[ˆ8 %ÛÜÙH]™Y›Ü™H™XY[™ËÜˆH\ÝÙÜ˜[BˆËÈÚ[[HÜÙ\ÈÛ™H[žH[™H[œËÙ˜]ÈYX[ˆ™XYÈÝË‚ˆš[™[Ù[œÝ\ÐÛÜÙT[Š
NÂˆÛÛœÝÝX›HHÝX›J×Øœ‘˜]ÜÊNÂˆËÈÒUHUÒTˆÓÕSTÔÕQNˆÛ™HØ[\ˆÛÛYÝ[Ý\È[ˆÙˆÚ[™ÙYš[™[™ÜËˆËÈ\È]™\žH[˜XÚÙYš[™ÚXÚ\È™]™\ˆ˜]ÚY‚ˆÛÛœÝÝX›H˜]ÚYHÝX›J×Øœ”[œÈ
È×Øœ•[˜XÚÙY
HÈÂˆÛÛœÝÝX›H›ÝÈHÝX›J×ØœÚ[™ÙY
È×Øœ•[˜XÚÙY
HÈÂˆœš[ŠÝ\œ‹ˆ–Ýš×H’S‘T•SˆÑS”ÕTÈÝ™\ˆ	[H˜]ÜÎˆÙ™™\™Y	KŒÙ‹Ù˜]ËÚ[™ÙY‚ˆ‰KŒÙ‹Ù˜]È
	KŒY‰IJK[œÈÙˆÚ[™ÙY	KŒÙ‹Ù˜]Ë[˜XÚÙY	KŒÙ‹Ù˜]È‚ˆŠ	[H˜]ÜÈYÛ™JWˆ‚ˆ–Ýš×HØ[ËÙ˜]È›ÝÈ	KŒÙˆOˆ˜]ÚY	KŒÙˆ
Ø]š[™È	KŒÙ‹Ù˜]ÈH‚ˆ‰KŒˆœÈH	KŒÙˆ\ËÙœ˜[YH]KÌ˜]ÜÈ[™LˆœÈHš]™\ˆØ[
Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Øœ‘˜]ÜËÝX›J×Øœ“Ù™™\™Y
HÈˆÝX›J×ØœÚ[™ÙY
HÈˆ×Øœ“Ù™™\™YÈLŒ
ˆÝX›J×ØœÚ[™ÙY
HÈÝX›J×Øœ“Ù™™\™Y
HˆŒˆÝX›J×Øœ”[œÊHÈÝX›J×Øœ•[˜XÚÙY
HÈˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Øœ•[˜XÚÙY˜]ÜËˆ›ÝË˜]ÚY›ÝÈH˜]ÚY
›ÝÈH˜]ÚY
H
ˆL‹Œˆ
›ÝÈH˜]ÚY
H
ˆL™KMˆ
ˆLÌŒ
NÂˆËÈHTÕ’P•USÓ‹™XØ]\ÙHHYX[ˆX›Ý™H\ÈÛÛœÚ\Ý[Ú]›Ý\Ý\Ù\Ë‚ˆÚ\ˆ\ÝÌM—NÂˆ[]HÂˆZ[ÝÝHÂˆ›Üˆ
[ÈHNÈÈHÈÊÊÊBˆÝ
ÏH×Øœ”[’\ÝÚ×NÂˆ›Üˆ
[ÈHNÈÈH	‰ˆ][
Ú^™[Ùˆ\Ý
HHÈÊÊÊBˆ]
ÏHÛœš[Š\Ý
È]Ú^™[Ùˆ\ÝH]‰\ÉY‰[J	KŒY‰IJH‹ˆÈˆHÈˆˆˆˆ‹ÈOHÈˆËˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Øœ”[’\ÝÚ×KˆÝÈLŒ
ˆÝX›J×Øœ”[’\ÝÚ×JHÈÝX›JÝ
HˆŒ
NÂˆœš[ŠÝ\œ‹–Ýš×H[‹[[™Ý\ÝÙÜ˜[H
HÜˆ[Ü™JNˆ	\×ˆ‹\Ý
NÂˆBˆYˆ
×ÙÝX\™Ù[œÝ\È	‰ˆ‹O™œ˜[YJBˆÂˆÛÛœÝÝX›HˆHÝX›J‹O™œ˜[YJNÂˆÛÛœÝZ[ÝÝH×ÙØÔ[\ž]\È
È×ÙØÔÛÛž]\ÎÂˆœš[ŠÝ\œ‹ˆ–Ýš×HÕPT‘ÑS”ÕTÈÝ™\ˆ	[Hœ˜[Y\Îˆ	KŒ™ˆP‹Ùœ˜[YH™XY[ˆÝ[8 %‚ˆ”ST	KŒ™ˆP‹Ùœ˜[YHÝ™\ˆ	KŒˆ\Ú\È
	KŒY‰IHÙˆž]\ÊKÓÓ	KŒ™ˆP‹Ùœ˜[YH‚ˆ›Ý™\ˆ	KŒˆ
	KŒY‰IJWˆ‚ˆ–Ýš×HH[\	ÜÈÝÛˆ[ˆÛÜÝ	KŒÙˆ\ËÙœ˜[YH
	[HœÈÝ™\ˆH[‹‚ˆ‰KŒYˆœÈH\Ú	KŒ™ˆÐ‹ÜÊWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O™œ˜[YKÝX›JÝ
HÈˆÈLMÍ‹ŒˆÝX›J×ÙØÔ[\ž]\ÊHÈˆÈLMÍ‹ŒÝX›J×ÙØÔ[\ÛÝ[
HÈ‹ˆÝÈLŒ
ˆÝX›J×ÙØÔ[\ž]\ÊHÈÝX›JÝ
HˆŒˆÝX›J×ÙØÔÛÛž]\ÊHÈˆÈLMÍ‹ŒÝX›J×ÙØÔÛÛÛÝ[
HÈ‹ˆÝÈLŒ
ˆÝX›J×ÙØÔÛÛž]\ÊHÈÝX›JÝ
HˆŒˆÝX›J×ÙØÔ[\œÊHÈˆÈYM‹
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙØÔ[\œËˆ×ÙØÔ[\ÛÝ[ÈÝX›J×ÙØÔ[\œÊHÈÝX›J×ÙØÔ[\ÛÝ[
HˆŒˆ×ÙØÔ[\œÈÈÝX›J×ÙØÔ[\ž]\ÊHÈÝX›J×ÙØÔ[\œÊHˆŒ
NÂˆBˆYˆ

×Ü’]È×Ü“Z\ÜÙ\È×Ü™^™\›ÓÙ™ŠH	‰ˆ‹O™œ˜[YJBˆÂˆÛÛœÝÝX›HˆHÝX›J‹O™œ˜[YJNÂˆÛÛœÝZ[ÝÙ\™YH×Ü’]Ë[›H×Ü“Z\ÜÙ\ÎÂˆœš[ŠÝ\œ‹ˆ–Ýš×HÒT‘QP“ÐÒÈ‘KV‘T“È
\LLHŒJI\Îˆ	KŒY‰IHÙˆ˜]ÜÈÙ\™Y‚ˆœ™K^™\›ÙY
	[HÙˆ	[JWˆ‚ˆ–Ýš×H[Ý™YÙ™ˆH[\	KŒ™ˆP‹Ùœ˜[YNÈÝ[[›[™H	KŒ™ˆP‹Ùœ˜[YWˆ‚ˆ–Ýš×HHÕTˆÚYHÙˆHš[ˆÛÜšÙ\œÈ	KŒÙˆ\ËÙœ˜[YH™\›Ú[™Ë[\‚ˆ‰KŒÙˆ\ËÙœ˜[YH˜Z[š[™È
	[H[Y	[HZY[ÊH[™	KŒÙˆ\ËÙœ˜[YH‚ˆØZ][™ÈÛˆH\ÞHÚ[šÈ
	[HØZ]ÊWˆ‚ˆ–Ýš×HÚ[šÜÎˆ	[HÜÝYÝ™\ˆ	[H\Ü]Ú\È
	KŒYˆHœ˜[YJK	[H‚ˆ˜ÛZ[YYžHHSTš\œÝ
ÛÜšÙ\œÈÚÚ\Y	[JK	[HÛÝÈ\ÝH‚ˆØ]\›X\šË	[H™YÚ[ÛˆÝ™\™›ÝÜ×ˆ‹ˆ×Ü™^™\›ÓÙ™ˆÈˆ8 %Ñ‘ˆ
HY˜][ÈÖ—Õ’×Ô‘V‘T“ÏLH[™ØYÙ\È]
Hˆˆˆ‹ˆ
Ù\™Y
È[›
HÈLŒ
ˆÝX›JÙ\™Y
HÈÝX›JÙ\™Y
È[›
HˆŒˆ
[œÚYÛ™YÛ™ÈÛ™Ê\Ù\™Y
[œÚYÛ™YÛ™ÈÛ™ÊJÙ\™Y
È[›
KˆÝX›J×Üž]\Ô™JHÈˆÈLMÍ‹ŒˆÝX›J×Üž]\Ò[›[™JHÈˆÈLMÍ‹ŒˆÝX›J×Ü•ÛÜšÙ\“œÐK›ØY

JHÈˆÈYM‹ÝX›J×Ü‘˜Z[“œÊHÈˆÈYM‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü‘˜Z[’[Y
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü‘˜Z[•ØZ]ËˆÝX›J×Ü•ØZ]œÊHÈˆÈYM‹
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü•ØZ]Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÜÚ[šÜÔÜÝYˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü‘\Ü]Ú\Ëˆ×Ü‘\Ü]Ú\ÈÈÝX›J×ÜÚ[šÜÔÜÝY
HÈÝX›J×Ü‘\Ü]Ú\ÊHˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü”[\ÛZ[\Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü•ÛÜšÙ\”ÚÚ\Ë›ØY

Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü™^[Û™Ø]\›X\šËˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ü“Ý™\™›ÝÊNÂˆBˆYˆ
×Ü\™˜]ÐÙ[œÝ\È	‰ˆ‹O™œ˜[YJBˆÂˆÛÛœÝÝX›HˆHÝX›J‹O™œ˜[YJNÂˆÛÛœÝÝX›HHÝX›J‹OœÚÚ\Ë™˜]ÜÈÈ‹OœÚÚ\Ë™˜]ÜÈˆJNÂˆÛÛœÝZ[ÝÝ™X[UÝH×ÜËœÝ™X[Qš[™ÎÂˆœš[ŠÝ\œ‹ˆ–Ýš×HT‹QUÈUUUSÓˆÑS”ÕTÈÝ™\ˆ	[Hœ˜[Y\ÈÈ	[H˜]ÜÈ8 %\LLH‚ˆ°©ÌË][H‰ÜÈ\ÚËYš\œÝÝ\ˆ‚ˆ–Ýš×H\™[˜H[\	NKŒ‹Ùœ˜[YH
	KŒ™‹Ù˜]ÊK	MËŒ™ˆP‹Ùœ˜[YH8 %Ý\œË‚ˆœÙ\šX[“Õ[YY
HÛØÚÈ™XY\È^HÜ
Wˆ‚ˆ–Ýš×H\œÚ\Ý[\	NKŒ‹Ùœ˜[YK	MËŒ™ˆP‹Ùœ˜[YWˆ‚ˆ–Ýš×HÝ™X[HØXÚH	NKŒˆš[™ËÙœ˜[YK	MËŒˆ[œÙ\ËÙœ˜[YHH‚ˆ‰KŒ™‰IHUUUSÓ”Ë[œÙ\ÛÜÝ	KŒÙˆ\ËÙœ˜[YWˆ‚ˆ–Ýš×H\œÚ\ÝÝÜ™H	NKŒˆš[™ËÙœ˜[YK	MËŒˆ[œÙ\ËÙœ˜[YH
È	KŒˆ‚ˆ›Z\œ›Üˆ\Ú\Ë]]][ÛˆÛÜÝ	KŒÙˆ\ËÙœ˜[YWˆ‚ˆ–Ýš×H^\™HX›H	NKŒˆš[™ËÙœ˜[YH
PPÒÕSTÈ\Ý\ÙYœ˜[YH8 %H‚ˆœ™XY[[ÙYžKUÔ’UJK	KŒYˆ[œÙ\ËÙœ˜[YK	KŒÙˆ\ËÙœ˜[YWˆ‚ˆ–Ýš×H\ØÜš\ÜˆÙ]	NKŒˆÜš]\ËÙœ˜[YK	KŒÙˆ\ËÙœ˜[YH8 %šÕ\]Jˆ\È‚ˆ™^\›˜[HÞ[˜Ú›Ûš\ÙY[™Ø[››ÝX]™HH[\[™ÝX\™Yˆ‚ˆ–Ýš×H\[[™HØXÚH	NKŒˆš[™ËÙœ˜[YK	KŒ™ˆ[œÙ\ËÙœ˜[YWˆ‚ˆ–Ýš×H‘PQÈÑˆ×Ü™YÜÈ	NKŒˆÛÛœÝ]Ú[™ÝÈÛÜY\ËÙœ˜[YH
	KŒ™‹Ù˜]ÊH
È‚ˆ‰KŒˆ™]ÚØ[ÜËÙœ˜[YH
	KŒ™‹Ù˜]ÊH8 %ŒÉÜÈÛÝ\˜ÙH˜XÙKÛÝ[Yˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹O™œ˜[YK
[œÚYÛ™YÛ™ÈÛ™ÊT‹OœÚÚ\Ë™˜]ÜËˆÝX›J×ÜË˜\™[˜P[ØÜÊHÈ‹ÝX›J×ÜË˜\™[˜P[ØÜÊHÈˆÝX›J×ÜË˜\™[˜Pž]\ÊHÈˆÈLMÍ‹ŒˆÝX›J×ÜËœ\œÚ\Ý[ØÜÊHÈ‹ˆÝX›J×ÜËœ\œÚ\Ý[ØÐž]\ÊHÈˆÈLMÍ‹ŒˆÝX›J×ÜËœÝ™X[Qš[™ÊHÈ‹ÝX›J×ÜËœÝ™X[R[œÙ\ÊHÈ‹ˆÝ™X[UÝÈLŒ
ˆÝX›J×ÜËœÝ™X[R[œÙ\ÊHÈÝX›JÝ™X[UÝ
HˆŒˆÝX›J×ÜËœÝ™X[R[œÙ\œÊHÈˆÈYM‹ˆÝX›J×ÜËœ\œÚ\Ýš[™ÊHÈ‹ÝX›J×ÜËœ\œÚ\Ý[œÙ\ÊHÈ‹ˆÝX›J×ÜË›Z\œ›Ü”\Ú\ÊHÈ‹ÝX›J×ÜËœ\œÚ\Ý]]œÊHÈˆÈYM‹ˆÝX›J×ÜË^š[™ÊHÈ‹ÝX›J×ÜË^[œÙ\ÊHÈ‹ˆÝX›J×ÜË^[œÙ\œÊHÈˆÈYM‹ˆÝX›J×ÜË™\ØÕÜš]\ÊHÈ‹ÝX›J×ÜË™\ØÕÜš]SœÊHÈˆÈYM‹ˆÝX›J×ÜËœ\Qš[™ÊHÈ‹ÝX›J×ÜËœ\R[œÙ\ÊHÈ‹ˆÝX›J×ÜË˜ÛÛœÝÚ[™ÝÐÛÜY\ÊHÈ‹ÝX›J×ÜË˜ÛÛœÝÚ[™ÝÐÛÜY\ÊHÈˆÝX›J×ÜË™™]ÚØ[ÜÊHÈ‹ÝX›J×ÜË™™]ÚØ[ÜÊHÈ
NÂˆËÈHÓ‘HS‘HHSˆTÒÑQ“Ô‹ˆØ\ÈÚ]Ý^\ÈÛˆH[\ÝÙ]™\ˆÛÛÙBˆËÈÚ\™[™È\ÎÈH™XYÚ\™H\ÈÚ]Ø^\ÈÚ]\ˆH™XY[[ÜÝHX›H\È[›ÝYÚ‚ˆÛÛœÝZ[Ý]]œÈH×ÜËœÝ™X[R[œÙ\œÈ
È×ÜËœ\œÚ\Ý]]œÈ
Âˆ×ÜË^[œÙ\œÈ
È×ÜË™\ØÕÜš]SœÎÂˆÛÛœÝZ[Ý™XYÈH×ÜËœÝ™X[Qš[™È
È×ÜËœ\œÚ\Ýš[™È
È×ÜË^š[™ÎÂˆÛÛœÝZ[ÝÜš]\ÈH×ÜËœÝ™X[R[œÙ\È
È×ÜËœ\œÚ\Ý[œÙ\È
Âˆ×ÜË›Z\œ›Ü”\Ú\È
È×ÜË^[œÙ\È
È×ÜË™\ØÕÜš]\ÎÂˆœš[ŠÝ\œ‹ˆ–Ýš×HOˆÈ
SQQ]]][ÛœÈ]]\ÝÝ^HÙ\šX[ÜˆÚ\™
HH	KŒÙˆ‚ˆ›\ËÙœ˜[YNÈÚ\™Y]X›H˜Y™šXÈ\È	KŒY‰IH™XYÈ
	[H™XYË	[HÜš]\È‚ˆœ\ˆ[ŠKˆH\™[˜H[\\ÈÛÝ[Y›Ý[YY[™\È^ÛYYœ›ÛHË—ˆ‹ˆÝX›J]]œÊHÈˆÈYM‹ˆ
™XYÈ
ÈÜš]\ÊHÈLŒ
ˆÝX›J™XYÊHÈÝX›J™XYÈ
ÈÜš]\ÊHˆŒˆ
[œÚYÛ™YÛ™ÈÛ™Ê\™XYË
[œÚYÛ™YÛ™ÈÛ™Ê]Üš]\ÊNÂˆBˆYˆ
×ÜÝ™X[QY\Ù[œÝ\È	‰ˆ‹OœÚÚ\Ë™˜]ÜÊBˆœš[ŠÝ\œ‹ˆ–Ýš×HÝ™X[HÛÚÝ\Îˆ	KŒ™ˆ\ˆ˜]Ë	KŒY‰IHÙˆ[H‘TPUHÙ^H\È‚ˆœØ[YH˜]È[™XYHÛÚÙY\
	[HÙˆ	[NÈ	[H˜]ÜÈ^ÙYYYH‚ˆŒM‹ZÙ^HÚ[™ÝÈ[™Ù\™HÛÝ[Y\È\Ý[˜ÝÚXÚ[™\‹\™\ÜÈ™\X]ÊWˆ‹ˆÝX›J×ÙY\ÛÚÝ\ÊHÈÝX›J‹OœÚÚ\Ë™˜]ÜÊKˆ×ÙY\ÛÚÝ\ÈÈLŒ
ˆÝX›J×ÙY\™\X]ÊHÈÝX›J×ÙY\ÛÚÝ\ÊHˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙY\™\X]Ë
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙY\ÛÚÝ\Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙY\Ý™\™›ÝÊNÂˆYˆ
×Û›Ñš]™\”™XÛÜ™ÚÚ\Y	‰ˆ‹OœÚÚ\Ë™˜]ÜÊBˆœš[ŠÝ\œ‹ˆ–Ýš×HÖ—Õ’×Ó“×Ñ’U‘T—Ô‘PÓÔ‘ˆ	[HšÐÛY
ˆØ[ÈÚÚ\Y	KŒ™ˆ\ˆ˜]È‚ˆ¸ %“ÕS‘ÈÐTÈUÓˆ[ˆ\È[—ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Û›Ñš]™\”™XÛÜ™ÚÚ\YˆÝX›J×Û›Ñš]™\”™XÛÜ™ÚÚ\Y
HÈÝX›J‹OœÚÚ\Ë™˜]ÜÊJNÂˆËÈHÕSÒSÒÒTQQSˆT•Ì‰ÜÈ‘T8 %[™]Ø\ÈÓÓPÕQÒSÑHT•MˆS‘ˆËÈ‘U‘Tˆ’S•QÚXÚ\ÈHY™XÝ\È›Ú™XÝÙY\È™Y\ØÛÝ™\š[™È
HÛÝ[\ˆ[ÝBˆËÈ[™XYH^H›Üˆ]›ÈÙÈØ\œšY\ÊKˆ]\ÈHXÚY[™È[X™\ˆ›Ü‚ˆËÈ\™‹\[‹\\Ì‹›Y0©ÌËH\Ý˜[YYÝ\ÜXÝ›Üˆ\N	ÜÈ
ÌKŒËLKˆ\Îˆ\M‚ˆËÈÛX\œÈ]™TÝ[˜Ú[ÛˆU‘T–H\[[™Hš[™™XØ]\ÙHš[™[™ÈH\[[™H]ˆËÈÜXÚYšY\ÈÝ]HÝ]XØ[HXZÙ\ÈHÛÜœ™\ÜÛ™[™È[˜[ZXÈÝ]H[™Yš[™Y8 %[™ˆËÈ‹	HÙˆ\È]IÜÈ˜]ÜÈš[™H™]È\[[™KˆÛÈH]Y\Ý[Ûˆ\ÈÚ]\ˆBˆËÈ™YHšÐÛYÙ]Ý[˜Ú[
˜Ø[ÈÛÛ\ÙHÈZ\ˆ™X[Ú[™ÙH˜]HÜˆ\™H™KZ\ÜÝYYˆËÈÛÛœÝ[K[™Û›H\È[™HØ[ˆ[œÝÙ\ˆ]‚ˆËÂˆËÈHS“ÓRSUÔˆ\ÈÝ[˜Ú[Y[˜X›Y˜]ÜË›Ý[˜]ÜÎˆH˜]ÈÚ]HÝ[˜Ú[ˆËÈ\ÝÙ™ˆ™Z]\ˆÙ]È›ÜˆÚÚ\Ë[™›Û[™È][ÈHÝ[ÛÝ[™\ÜBˆËÈX[K[ÛÚÚ[™ÈMIH›ÜˆHØXÚH]™]™\ˆÙ\™\È[ž][™Ë‚ˆYˆ
×ÜÝ[˜Ú[˜]ÜÊBˆœš[ŠÝ\œ‹ˆ–Ýš×HÝ[˜Ú[[˜[ZXË\Ý]HÙ]ÈÚÚ\Yˆ	KŒY‰IH
Ùˆ	[H‚ˆœÝ[˜Ú[Y[˜X›Y˜]ÜÊWˆ‹ˆLŒ
ˆÝX›J‹OœÚÚ\ËœÝ[˜Ú[
HÈÝX›J×ÜÝ[˜Ú[˜]ÜÊKˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÜÝ[˜Ú[˜]ÜÊNÂˆËÈ‹‹˜[™HÛÈHØXÚHÓÕ‘T”ÈTÈÑˆT•ËˆH[X™\œÈÙ\™HHYX\Ý\™[Y[ˆËÈ]\ÝYšYYÜš][™È]
LKŒ	H[™ÎK	HÛˆHÜ\˜]Ü‰ÜÈÝÛˆÙ\ÜÚ[Û‹Ý™\‚ˆËÈM‹ŒMÈH˜]ÜÊNÈ^H\™H›ÝÈHÛÝ[ÙˆØ[ÈH™[™\™\ˆY›ÝXZÙK‚ˆËÈ™\ÜY\ÈÛÝ[È\ÈÙ[\È\˜Ù[YÙ\È™XØ]\ÙHHXœÛÛ]H[X™\ˆ\ÈÚ]ˆËÈ][\Y\ÈžHHŒÍœÈHšÐÛY
˜ÛÜÝÈ\™K‚ˆYˆ
‹OœÚÚ\Ë™\^š[™È‹OœÚÚ\Ëš[™^š[™ÊBˆœš[ŠÝ\œ‹ˆ–Ýš×Hš[™ÈÚÚ\YžHHÝ]HØXÚNˆ™\^	[HÙˆ	[H™\X]‚ˆH™]š[Ý\ÈÙ™œÙ]
	KŒY‰IJK[™^	[HÙˆ	[H
	KŒY‰IJWˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹OœÚÚ\Ë™\^š[™™\X]Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹OœÚÚ\Ë™\^š[™Ëˆ‹OœÚÚ\Ë™\^š[™ÂˆÈLŒ
ˆÝX›J‹OœÚÚ\Ë™\^š[™™\X]ÊHÂˆÝX›J‹OœÚÚ\Ë™\^š[™ÊBˆˆŒˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹OœÚÚ\Ëš[™^š[™™\X]Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊT‹OœÚÚ\Ëš[™^š[™Ëˆ‹OœÚÚ\Ëš[™^š[™ÈÈLŒ
ˆÝX›J‹OœÚÚ\Ëš[™^š[™™\X]ÊHÂˆÝX›J‹OœÚÚ\Ëš[™^š[™ÊBˆˆŒ
NÂˆ›Üˆ
ÛÛœÝ]]ÉˆÛ˜[YKÛÝ[Hˆ×ÜÝ]ÊBˆœš[ŠÝ\œ‹–Ýš×H	KMLœÈ	[Wˆ‹˜[YK˜×ÜÝŠ
Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊXÛÝ[
NÂ‚ˆËÈH\‹XY™\ÜÈX›KˆÛ›HH›ÝÜÈ]Ø^HÛÛY][™È\™Hš[YˆHÝ\™˜XÙBˆËÈ\È™[™\™\ˆ™\ÛÛ™YËÜˆ[ˆ\ØY]Ø[YHÝ][\™[H™\›Ëˆ]™\ž][™ÂˆËÈ[ÙH\È[ˆÜ™[˜\žH\ØÈ^\™H[™HYÙÜ™YØ]HÛÝ[\œÈ[™XYHÛÝ™\ˆ]‚ˆYˆ
×Ý^Ù[œÝ\ÊBˆÂˆœš[ŠÝ\œ‹–Ýš×H^\™HÛÝ\˜Ù\È
Y‹^[›]\ØYËÞ™\›Ë‚ˆœÛ˜\ÚÝÛÓÛX^YÙJN—ˆŠNÂˆËÈÛÜYÛÈ\Èš[È^XÝHÚ]HÝŽ›X\š[Y™Y›Ü™H\MBˆËÈXYHHX›H›]ˆHÙ[œÝ\ÈÚÜÙH“ÕÈÔ‘TˆÚ[™Ù\È™XYÈ\ÈHY™™\™[ˆËÈÙ[œÝ\ÈÈ[ž[Û™HY™š[™ÈÛÈ[œÉÈÙÜË‚ˆÝŽ™XÝÜÝŽœZ\Z[Ì—Ý^ÛÝ\˜ÙOˆÜ˜Ô›ÝÜÎÂˆ×Ý^ÛÝ\˜Ù\Ë‘›Ü‘XXÚ
É—H
Z[ÝËÛÛœÝ^ÛÝ\˜ÙIˆŠHÂˆÜ˜Ô›ÝÜË™[\XÙWØ˜XÚÊZ[Ì—Ý
ÊKŠNÂˆJNÂˆÝŽœÛÜ
Ü˜Ô›ÝÜË˜™YÚ[Š
KÜ˜Ô›ÝÜË™[™

Kˆ×H
ÛÛœÝ]]ÉˆKÛÛœÝ]]ÉˆŠHÈ™]\›ˆK™š\œÝ‹™š\œÝÈJNÂˆ›Üˆ
]]ÉˆØY‹×HˆÜ˜Ô›ÝÜÊBˆÂˆYˆ
\Ë™]™\”™\ÛÛ™Y	‰ˆ\Ëž™\›Õ\ØYÊBˆÛÛ[YNÂˆËÈ™K\™XYHÛÝ\˜ÙHž]\È“ÕËˆH›ÝÈ]\ØYY›XÚÈ[™\ÈÝ[ˆËÈ›XÚÈ[ˆÝY\ÝY[[ÜžH\ÈH^\™HHÝY\Ý™]™\ˆÜ›ÝNÈÛ™H]ˆËÈ\ØYY›XÚÈ[™›ÝÈ™XYÈ›Û‹^™\›È\ÈH^\™H]\œš]™YQ•T‚ˆËÈÝ\ˆÛ™H[™Û›H\ØY[™\Èœ›Þ™[ˆ›XÚÈžHHØXÚK‚ˆÛÛœÝÚ\Šˆ›ÝHHˆŽÂˆYˆ
Ëž™\›Õ\ØYÈ	‰ˆËœÜ˜Ðž]\È	‰ˆËœÜ˜ÊBˆÂˆÛÛœÝZ[Ý
ˆHËœÜ˜ÎÂˆ›ÛÛ›ÝÖ™\›ÈHYNÂˆ›Üˆ
Z[ÝHHÈHËœÜ˜Ðž]\ÎÈJÊÊBˆYˆ
ÚWJBˆÂˆ›ÝÖ™\›ÈH˜[ÙNÂˆœ™XZÎÂˆBˆ›ÝHH›ÝÖ™\›ÈÈˆH\ØYY“PÒËÝY\ÝY[[ÜžHÕS™\›È‚ˆˆˆH\ØYY“PÒËÝY\ÝY[[ÜžH\È“Ó‹V‘T“È“ÕÈŽÂˆBˆœš[ŠÝ\œ‹ˆ–Ýš×H	L	KMÜÈ	M^	KMH‰KLH\	[H
™\›È	[JHÛ˜\	[H‚ˆÛÓÛ	[H
X^YÙH	[JI\×ˆ‹ˆYˆ	ˆQ‘‘‘‘‘‘‹
Yˆ	ˆÔÛ˜\ÚÝ\š]
HÈŠ\
Hˆˆˆ‹ˆËÚYËšZYÚË™›Ü›X]
[œÚYÛ™YÛ™ÈÛ™Ê\Ë\ØYËˆ
[œÚYÛ™YÛ™ÈÛ™Ê\Ëž™\›Õ\ØYËˆ
[œÚYÛ™YÛ™ÈÛ™Ê\Ë™œ›ÛTÛ˜\ÚÝˆ
[œÚYÛ™YÛ™ÈÛ™Ê\ËœÛ˜\ÚÝÛÓÛ
[œÚYÛ™YÛ™ÈÛ™Ê\Ë›X^YÙKˆ›ÝJNÂˆBˆB‚ˆËÈ“ÕUH
ŠIÜÈS‘ÐQÑSQS•S‘K[Ø^\ÈÛˆÚ[ˆ[ˆ•Y\ˆ]™\ˆ˜[‹ˆ\Ú\YˆËÈHZ[ÚÜÙHKÐˆYX\Ý\™YH›]Û\ÜÈš^[™Ø\ÈH™X]\™HÚ[[HÝÚ]ÚYˆËÈÙ™ˆ
ÛÝÚHÎŠNÈHÚX\\ÝY™[˜ÙHYØZ[œÝ™\X][™È]\ÈHÛÝ[]ˆËÈØ[››Ý™HÛÛ™\ÙYÚ]HÛÛ™šYÝ\˜][Û‹‚ˆYˆ
˜XÝÜŽŽ™×Ùš\™TØ[\\ÊBˆœš[ŠÝ\œ‹ˆ–Ü—HH˜XÝÜˆ\ÜÈš\™\È]˜]È	[HÙˆ‰[HÛˆ]™\˜YÙH‚ˆŠZ[ˆ	[KX^	[JNÈ	[H]\È˜]ÜÈPÓS‘Q\È‹\™\\ÜÈ‚ˆŠ[\HÛÛÝ\ˆX\ÚÊKˆ\È]H\È“ÈØÙ[™Hˆ™\\ÜÈ‚ˆŠÛÛËÜÙ\ÛÜ™\—ØÙ[œÝ\ËœKŒ˜XÙ\ÊKÛÈ[™\ˆHTÛÝ\˜ÙH‚ˆHØ[\H\™H\È[Ø^\ÈHÛX\ˆ˜[YH8 %˜\‹K™KˆUˆH‚ˆœš[X\žK\˜^HÛÝ\˜ÙHÙ\È›ÝØ\™HÚ\™H[ˆHœ˜[YH\È[X™\ˆ‚ˆ›[™Ë—ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ˜XÝÜŽŽ™×Ùš\™P]Ý[HÈ˜XÝÜŽŽ™×Ùš\™TØ[\\ÊKˆ
[œÚYÛ™YÛ™ÈÛ™ÊJ˜XÝÜŽŽ™×Ùœ˜[YQ˜]ÔÝ[HÈ˜XÝÜŽŽ™×Ùš\™TØ[\\ÊKˆ
[œÚYÛ™YÛ™ÈÛ™Ê\˜XÝÜŽŽ™×Ùš\™P]Z[‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\˜XÝÜŽŽ™×Ùš\™P]X^ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\˜XÝÜŽŽ™×ÙXÛ[™Y™\\ÜÊNÂ‚ˆËÈHUTÈ’S‘S‘Ëš[YÛÈ]\ÈÚXÚØX›H˜]\ˆ[ˆ\ÜÝ[YYˆÛ™H›ÝÈYX[œÂˆËÈHÚYÝË\Ø[\[™ÈÚY\œÈ™]Ú^XÝHÛ™H\Ý\™˜XÙNÈHÚÜÙ[ˆÛ™H\ÂˆËÈX\šÙY[™HÛÛ\][™ÈØ[™Y]H\Èš\ÚX›H[œÝXYÙˆÚ[[HÜÚ[™Ë‚ˆYˆ
ÚYÝÎŽ™×Ø]\ÐØ[™ÛÝ[
BˆÂˆœš[ŠÝ\œ‹–Ü—H\Ý\™˜XÙ\È™]ÚYžHHÚYÝÈÚY\œÈ‚ˆŠH\™Ù\Ý\ÈZÙ[ˆ\ÈHØ\ØØYH]\ÊN—ˆŠNÂˆ›Üˆ
Z[Ì—ÝHHÈHÚYÝÎŽ™×Ø]\ÐØ[™ÛÝ[È
ÊÚJBˆœš[ŠÝ\œ‹–Ü—H	L	M]^	KM]H	[H™]Ú\É\×ˆ‹ˆÚYÝÎŽ™×Ø]\ÐØ[™ÖÚWK˜Y‹ÚYÝÎŽ™×Ø]\ÐØ[™ÖÚWKËˆÚYÝÎŽ™×Ø]\ÐØ[™ÖÚWKšˆ
[œÚYÛ™YÛ™ÈÛ™Ê\ÚYÝÎŽ™×Ø]\ÐØ[™ÖÚWK™™]Ú\ËˆÚYÝÎŽ™×Ø]\ÐØ[™ÖÚWK˜YˆOHÚYÝÎŽ™×Ø]\ÐY‚ˆÈˆKHUTÈˆˆˆŠNÂˆB‚ˆËÈU‘T–HÕSˆT‘PÕSÓˆHUÒÐUË›Ý\ÝH\ÝˆÛ™H›ÝÈ\ÈH[œÝÙ\ŽÂˆËÈÛÈ›ÝÜÈYX[œÈÛÛY][™È]\È›ÝHÝ[‰ÜÈØ\ØØYH\È™Z[™ÈØ\\™YÚXÚˆËÈ\È^XÝHÚ]ÛÜÝHÜ\˜]Ü‰ÜÈÙXÛÛ™Ù\ÜÚ[Ûˆ
0©Í˜ÝÈ0©ÎJK‚ˆYˆ
ÚYÝÎŽ™×ÜÝ[“ØœÐÛÝ[
BˆÂˆœš[ŠÝ\œ‹–Ü—HÝ[ˆ\™XÝ[ÛœÈ]ÚY
	[H]Ú\Ë	]H\Ý[˜Ý‚ˆ‰[H›ÝHÝÚ]Ú\È8 %[Ü™H[ˆÛ™H\™XÝ[Ûˆ\ÈVPÕQ\™H‚ˆ˜[™H\‹Yœ˜[YHXZ›Üš]H\ÈÚ]ÚÛÜÙ\ÊN—ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\ÚYÝÎŽ™×ÜÝ[“]ÚYÚYÝÎŽ™×ÜÝ[“ØœÐÛÝ[ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\ÚYÝÎŽ™×ÜÝ[”ÝÚ]Ú\ÊNÂˆ›Üˆ
Z[Ì—ÝHHÈHÚYÝÎŽ™×ÜÝ[“ØœÐÛÝ[È
ÊÚJBˆœš[ŠÝ\œ‹–Ü—H
	JËŒÙˆ	JËŒÙˆ	JËŒÙŠH›Û[YH	KŒYˆ	[H‚ˆ™œ˜[Y\È	[K‹‰[Wˆ‹ˆÚYÝÎŽ™×ÜÝ[“ØœÖÚWK™\–ÌKÚYÝÎŽ™×ÜÝ[“ØœÖÚWK™\–ÌWKˆÚYÝÎŽ™×ÜÝ[“ØœÖÚWK™\–Ì—KÚYÝÎŽ™×ÜÝ[“ØœÖÚWK›[‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\ÚYÝÎŽ™×ÜÝ[“ØœÖÚWK˜ÛÝ[ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\ÚYÝÎŽ™×ÜÝ[“ØœÖÚWK™š\œÝœ˜[YKˆ
[œÚYÛ™YÛ™ÈÛ™Ê\ÚYÝÎŽ™×ÜÝ[“ØœÖÚWK›\Ýœ˜[YJNÂˆBˆËÈS‘HUIÔÈÕÓˆS”ÕÑTˆ‘TÒQHU8 %HÝ[™[™ÈØ]HÛˆ\Ì	ÜÈÚ[™ÙK‚ˆËÈHÛÈ\™H[™\[™[™XY[™ÜÈÙˆÛ™H\ÚXØ[]X[]KÛÈH\ØYÜ™Y[Y[\ÈBˆËÈY™XÝ[ˆÚXÚ]™\ˆ\È›ÝH]IÜË[™\™Ø\™H\È[™XYHØZYÚXÚ]ˆËÈ\È
ÛÛËÞ—ÜÝ[—ÛÜ˜XÛKœXÙ[HÙˆÙ[H]ŒYÜ™Y\ÊKˆš[Y]™[‚ˆËÈÚ[ˆHØ\ØØYH\›H\ÈÙ[XÝY™XØ]\ÙHHÛÛ\\š\ÛÛˆ\ÈHÚ[‚ˆYˆ
ÚYÝÎŽ™×ÙÝY\ÝÝ[”›Ø™\ÊBˆÂˆœš[ŠÝ\œ‹ˆ–Ü—HHUIÔÈÕÓˆÝ[ˆ
^[ÛÛœÝ[ÌŒËÜ›ÜÜËXÚXÚÙYYØZ[œÝ‚ˆš]ÈÝÛˆØ\ØØYHX]š^
Nˆ	[Hœ˜[Y\È›Ý[™	]H\Ý[˜Ý	[H›Ø™H‚ˆ™˜]ÜË	[H›ØÚÜÈ‘R‘PÕQ›ÜˆÌŒË]œËXØ\ØØYH\ØYÜ™Y[Y[ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\ÚYÝÎŽ™×ÙÝY\ÝÝ[”Ø[\\ËˆÚYÝÎŽ™×ÙÜÝ[“ØœÐÛÝ[ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\ÚYÝÎŽ™×ÙÝY\ÝÝ[”›Ø™\Ëˆ
[œÚYÛ™YÛ™ÈÛ™Ê\ÚYÝÎŽ™×ÙÝY\ÝÝ[“Z\ÛX]Ú
NÂˆ›Üˆ
Z[Ì—ÝHHÈHÚYÝÎŽ™×ÙÜÝ[“ØœÐÛÝ[È
ÊÚJBˆœš[ŠÝ\œ‹–Ü—H
	JËŒÙˆ	JËŒÙˆ	JËŒÙŠH	[Hœ˜[Y\È	[K‹‰[Wˆ‹ˆÚYÝÎŽ™×ÙÜÝ[“ØœÖÚWK™\–ÌKÚYÝÎŽ™×ÙÜÝ[“ØœÖÚWK™\–ÌWKˆÚYÝÎŽ™×ÙÜÝ[“ØœÖÚWK™\–Ì—Kˆ
[œÚYÛ™YÛ™ÈÛ™Ê\ÚYÝÎŽ™×ÙÜÝ[“ØœÖÚWK˜ÛÝ[ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\ÚYÝÎŽ™×ÙÜÝ[“ØœÖÚWK™š\œÝœ˜[YKˆ
[œÚYÛ™YÛ™ÈÛ™Ê\ÚYÝÎŽ™×ÙÜÝ[“ØœÖÚWK›\Ýœ˜[YJNÂˆYˆ
ÚYÝÎŽ™×ÜÝ[‘\ØYÜ™YHHŒŠBˆœš[ŠÝ\œ‹ˆ–Ü—HHÛÈ™XY[™ÜÈY™™\ˆžH	KŒYˆYÜ™Y\Ë[™H\ÜÈ\ÙY‚ˆH	\ÈÛ™Wˆ‹ˆÚYÝÎŽ™×ÜÝ[‘\ØYÜ™YKˆÚYÝÎŽ™×ÜÝ[”Ü˜ÑÝY\ÝÈ•UIÔÈˆˆ˜Ø\ØØYKY\š]™YŠNÂˆBˆ[ÙHYˆ
ÚYÝÎŽ™×ÜÝ[“ØœÐÛÝ[
Bˆœš[ŠÝ\œ‹–Ü—HH]IÜÈÝÛˆÝ[ˆÛÛœÝ[Ø\È‘U‘Tˆ“Ð‘Q8 %Z]\ˆ‚ˆ”•Ø\ÈÙ™ˆ›ÜˆHÚÛH[ˆÜˆ›È˜]ÈØ\œšYYHÛÜ›‚ˆ˜ÛÛœÝ[›ØÚ×ˆŠNÂˆYˆ
˜XÝÜŽŽ™×Ü\ÜÙ\È˜XÝÜŽŽ™×Û›ÔØÙ[™H˜XÝÜŽŽ™×Û›ÓYÚˆ˜XÝÜŽŽ™×Û›Õ\È˜XÝÜŽŽ™×ÜÚ[™Ý[\ŠBˆÂˆÚYÝÎŽ”š[ÛÛXÝÜÙ[œÝ\Ê˜ÛÛXÝÜˆÕSŠNÂˆœš[ŠÝ\œ‹ˆ–Ü—HÕSˆ	[H˜XÝÜˆ\ÜÙ\Ë	[H˜]ÜÈÙ\™Y	]H˜\šX[‚ˆ›[Ù[\ÎÈÚÚ\Y›ÔØÙ[™OI[H›ÓYÚI[H›Õ\ÏI[HÚ[™Ý[\I[Kˆ‚ˆ”ØÙ[™HÛÛ\ÜÚ]Nˆ	[Hœ˜[Y\ÈØ\œšYYÓ‘K	[HØ\œšYYÑU‘TS
[žH‚ˆ‰ÜÙ]™\˜[	ÈYX[œÈHÛÜ›ØØ[Y\˜Hš[™[™È\È›ÝÚ]0©Í˜ÜÈYX\Ý\™Y
Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\˜XÝÜŽŽ™×Ü\ÜÙ\Ëˆ
[œÚYÛ™YÛ™ÈÛ™Ê\˜XÝÜŽŽ™×Ù˜]ÜÔÙ\™Y‹Oœ˜\šX[Ëˆ
[œÚYÛ™YÛ™ÈÛ™Ê\˜XÝÜŽŽ™×Û›ÔØÙ[™Kˆ
[œÚYÛ™YÛ™ÈÛ™Ê\˜XÝÜŽŽ™×Û›ÓYÚˆ
[œÚYÛ™YÛ™ÈÛ™Ê\˜XÝÜŽŽ™×Û›Õ\Ëˆ
[œÚYÛ™YÛ™ÈÛ™Ê\˜XÝÜŽŽ™×ÜÚ[™Ý[\‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\˜XÝÜŽŽ™×Ùœ˜[Y\ÓÛ™SX]š^ˆ
[œÚYÛ™YÛ™ÈÛ™Ê\˜XÝÜŽŽ™×Ùœ˜[Y\ÓX[žSX]š^
NÂˆB‚ˆËÈÒT‘HHSQS”ÒSÓˆU‘TÈSˆH‘UÒÓÓ”ÕS•™XYÙ™ˆHÛÈÛ\ÜÙ\ÈBˆËÈÚY\ˆ\][ÛœÈ]™\žH™]Ú[Ëˆ[Ø^\ÌX\ÈHS‘[Ø^\Ì\ÈBˆËÈÛÛ\[Y[ÙˆHÔŽÈHšY[][˜ÛÙ\ÈH[Y[œÚ[Ûˆ]\Ý™H[œÚYHHš]ÂˆËÈÚ\™HHÛÈÛ\ÜÙ\ÉÈ]\›œÈY™™\‹[™H™\Üš[È]\ØYÜ™Y[Y[ˆËÈX\ÚÈ\™XÝHÛÈH[œÝÙ\ˆ\ÈHš]ÜÚ][Ûˆ˜]\ˆ[ˆ[ˆ\™Ý[Y[‚ˆYˆ
×Ù[PÙ[œÝ\ÊBˆÂˆÝ]XÈÛÛœÝÚ\ŠˆÑ[S˜[YVÍHHÈŒQ‹Œ‘‹ŒÑ‹ÝX™HˆNÂˆœš[ŠÝ\œ‹–Ýš×H™]ÚXÛÛœÝ[[Y[œÚ[ÛˆÙ[œÝ\È8 %ÛÜ™È\ˆ‚ˆœÚY\‹YXÛ\™Y[Y[œÚ[ÛŽ—ˆŠNÂˆ›Üˆ
ÛÛœÝ]]ÉˆÙ[K×Hˆ×Ù[PÛ\ÜÙ\ÊBˆÂˆœš[ŠÝ\œ‹–Ýš×H	KMÈ	[H™]Ú\×ˆ‹ˆ[HÈÑ[S˜[YVÙ[WHˆÈ‹
[œÚYÛ™YÛ™ÈÛ™ÊXË™™]Ú\ÊNÂˆ›Üˆ
Z[Ì—ÝHÈŽÈ
ÊÊBˆœš[ŠÝ\œ‹–Ýš×HÛÜ™	]H[Ø^\ÌOIL[Ø^\ÌILˆ‹ˆË˜[™X\ÚÖÙK˜Ë›Ü“X\ÚÖÙJNÂˆœš[ŠÝ\œ‹–Ýš×HÛÜ™Œˆ
HÝXÚÈ\[šXIÜÈ^[Ý]‚ˆœ™YXÝÈ\ÈH›ÜˆHÝX™JNˆŠNÂˆ›Üˆ
ÛÛœÝ]]ÉˆÝ‹—HˆË™•Ü
Bˆœš[ŠÝ\œ‹ˆ	]H	[H‹‹
[œÚYÛ™YÛ™ÈÛ™Ê[ŠNÂˆœš[ŠÝ\œ‹—ˆŠNÂˆBˆËÈHZ\Ú\ÙH\ØYÜ™Y[Y[ÚXÚ\ÈHXÝX[[œÝÙ\‹ˆÛ›HÛÛ\]Y™]ÙY[‚ˆËÈÛ\ÜÙ\È]›ÝØ]È™]Ú\È8 %HÛ\ÜÈÚ]›Û™H\ÈS‘_Œ[™ÔLÚXÚˆËÈÛÝ[\ØYÜ™YHÚ]]™\ž][™È[™YX[ˆ›Ý[™È
ÛÝÚHÊK‚ˆ›Üˆ
ÛÛœÝ]]ÉˆØKØWHˆ×Ù[PÛ\ÜÙ\ÊBˆ›Üˆ
ÛÛœÝ]]ÉˆØ‹Ø—Hˆ×Ù[PÛ\ÜÙ\ÊBˆÂˆYˆ
HHˆXØK™™]Ú\ÈXØ‹™™]Ú\ÊBˆÛÛ[YNÂˆ›Üˆ
Z[Ì—ÝHÈŽÈ
ÊÊBˆÂˆËÈš]ÈÛ™HÛ\ÜÈ[Ø^\ÈÙ]È[™HÝ\ˆ[Ø^\ÈÛX\œËZ]\‚ˆËÈØ^H›Ý[™ˆ[ž][™È[ÙH˜\šY\ÈÚ][ˆHÛ\ÜÈ[™Ø[››Ý™HBˆËÈÛÛœÝ[\‹Y[Y[œÚ[ÛˆšY[‚ˆÛÛœÝZ[Ì—ÝÙ\H
ØK˜[™X\ÚÖÙH	ˆ˜Ø‹›Ü“X\ÚÖÙJHˆ
Ø‹˜[™X\ÚÖÙH	ˆ˜ØK›Ü“X\ÚÖÙJNÂˆYˆ
Ù\
Bˆœš[ŠÝ\œ‹ˆ–Ýš×H	\ÈœÈ	\ÎˆÛÜ™	]HÙ\\˜]\ÈÛˆš]È	Lˆ‹ˆHÈÑ[S˜[YVØWHˆÈ‹ˆÈÑ[S˜[YVØ—HˆÈ‹ˆÙ\
NÂˆBˆBˆB‚ˆËÈÒPÒÒQT”ÈTÐQÔ‘QHÒURTˆÕÓˆ‘UÒÓÓ”ÕS•Ë[™X›Ý]ÚXÚ^\™K‚ˆËÈ[˜›Ý[™Y[›ZÙHH\‹[ØØÝ\œ™[˜ÙHš[X›Ý™K™XØ]\ÙHHÜ[][Ûˆ\ÈBˆËÈ]Y\Ý[ÛŽˆÛ™HÚY\ˆ\ØYÜ™YZ[™ÈX›Ý]Û™HXÙZÛ\ˆ^\™H[™šYHÚY\œÂˆËÈ\ØYÜ™YZ[™ÈX›Ý]šYH™X[Û™\È\™HHØ[YHÛÝ[\ˆ[™ÛÛ\][HY™™\™[ˆËÈY™XÝËˆÛÛ\\™HHÚY\ˆ\Ú\È\™HYØZ[œÝÛÛËÞ—ØÝX™WØYÜ™Y[Y[œXÛˆBˆËÈØ\\™H8 %HÚY\ˆ]\ØYÜ™Y\È\™H[™YÜ™Y\È\™H\ÈÕTˆ™YÚ\Ý\ˆš[NÈBˆËÈÚY\ˆ]\X\œÈ[ˆ›ÈØ\\™H\ÈHØ\ÙH\™Ø\™H\È™]™\ˆ™Y[ˆ\ÚÙYX›Ý]‚ˆYˆ
×Ù[Q\ØYÜ™YJBˆÂˆÝ]XÈÛÛœÝÚ\ŠˆÑ[S˜[YVÍHHÈŒQ‹Œ‘‹ŒÑ‹ÝX™HˆNÂˆZ[ÝÝ[HÂˆ›Üˆ
ÛÛœÝ]]ÉˆÚËWHˆ×Ù[Q\ØYÜ™Y[Y[ÊBˆÝ[
ÏHK™™]Ú\ÎÂˆœš[ŠÝ\œ‹ˆ–Ýš×HÚY\‹ØÛÛœÝ[[Y[œÚ[Ûˆ\ØYÜ™Y[Y[Îˆ	[H™]Ú\ÈÝ™\ˆ	^H‚ˆ™\Ý[˜Ý
ÚY\‹ÛÝ^\™JHØ\Ù\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™Ê]Ý[×Ù[Q\ØYÜ™Y[Y[ËœÚ^™J
JNÂˆ›Üˆ
ÛÛœÝ]]ÉˆÚËWHˆ×Ù[Q\ØYÜ™Y[Y[ÊBˆœš[ŠÝ\œ‹ˆ–Ýš×HÏILM›œÏILM›ÉKLHÚY\IKMÈÛÛœÝ[IKMÈ‚ˆ‰L	]^	]H›]I]H	[Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊYKœÒ\Ú
[œÚYÛ™YÛ™ÈÛ™ÊYKœÒ\ÚKœÛÝˆKœÚY\‘[HÈÑ[S˜[YVÙKœÚY\‘[WHˆÈ‹ˆK˜ÛÛœÝ[HÈÑ[S˜[YVÙK˜ÛÛœÝ[WHˆÈ‹K˜Y‹KËKšK™›]ˆ
[œÚYÛ™YÛ™ÈÛ™ÊYK™™]Ú\ÊNÂˆB‚ˆËÈH^\™KXÛÛ[ÝX\™ˆH]Y\Ý[Ûˆ\ÈHÜ\˜]Ü‰ÜÎˆ\ÈH˜]È™Z[™ÈÙ\™YˆËÈ[ˆ[XYÙHZ[œ›ÛH^[È]\™H›ÈÛ™Ù\ˆ]]Y™\ÜÏÂˆÛÛ[Üš]\ŽŽ‘˜Z[Š
NÈËÈ\LŽˆš[š\ÚH]Y]YYÛÛ[ˆš[\È™Y›Ü™H™\Ü[™ÂˆYˆ
ÛÛ[Üš]\ŽŽ™›ÜY
Bˆœš[ŠÝ\œ‹–Ýš×HÛÛ[ˆ^\™HÝÜ™Nˆ	[H\œÚ\Ý
ÊH“ÔQ8 %HÜš]\ˆ‚ˆœ]Y]YHØ\È[
	^JNÈ^HÚ[™HØ\\™YYØZ[ˆ™^Ù\ÜÚ[Û—ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊYÛÛ[Üš]\ŽŽ™›ÜYÛÛ[Üš]\ŽŽšÓX^]Y]YY
NÂˆYˆ
Y×Û›ÑÛÛ[ˆ	‰ˆ
×ÙÛÛ[”ÝÜ™Y×ÙÛÛ[”Ù\™Y
JBˆœš[ŠÝ\œ‹ˆ–Ýš×HÛÛ[ˆ^\™HÝÜ™Nˆ	[HÚYÛ˜]\™\È™[Y[X™\™Y	[H[^™\›È‚ˆ\ØYÈÙ\™Y™X[ž]\È
	^H[šY\È[
Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙÛÛ[”ÝÜ™Y
[œÚYÛ™YÛ™ÈÛ™ÊY×ÙÛÛ[”Ù\™Yˆ×ÙÛÛ[•^œÚ^™J
JNÂˆYˆ
Y×Û›ÑÛÛ[ˆ	‰ˆY×Û›ÑÛÛ[”XÚÂˆ	‰ˆ
ÛÛ[Üš]\ŽŽœXÚÐ\[™YÛÛ[Üš]\ŽŽ›ÛÜÙT™[[Ý™Y×ÙÛÛ[”XÚÑ[šY\ÊJBˆœš[ŠÝ\œ‹ˆ–Ýš×HÛÛ[ˆXÚÎˆ	^H[šY\È]ØY
È	[H\[™Y\ÈÙ\ÜÚ[Ûˆ‚ˆŠ	KŒYˆPˆÜš][ŠK	[HÛÜÙHš[\È›ÛY[ˆ[™™[[Ý™Yˆ‹ˆ×ÙÛÛ[”XÚÑ[šY\Ë
[œÚYÛ™YÛ™ÈÛ™ÊYÛÛ[Üš]\ŽŽœXÚÐ\[™YˆÝX›JÛÛ[Üš]\ŽŽœXÚÐ\[™Yž]\ÊHÈ
LŒ
ˆLŒ
Kˆ
[œÚYÛ™YÛ™ÈÛ™ÊYÛÛ[Üš]\ŽŽ›ÛÜÙT™[[Ý™Y
NÂˆYˆ
×Ý^ÝX\™Ý]Ëš]ÊBˆÂˆÛÛœÝ^ÝX\™Ý]ÉˆÈH×Ý^ÝX\™Ý]ÎÂˆœš[ŠÝ\œ‹ˆ–Ýš×H^\™HÝX\™ˆ	[HØXÚH]ÈÚXÚÙY
Š‰[HÙ\™Y[ˆ[XYÙH‚ˆÚÜÙHÝY\Ýž]\ÈYÒS‘ÑQ
	KŒ™‰IJJŠ‹	[H™K]\ØYYÝX\™™XY‚ˆ‰KŒYˆP—ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊYËš]Ë
[œÚYÛ™YÛ™ÈÛ™ÊYË˜Ú[™ÙYˆLŒ
ˆÝX›JË˜Ú[™ÙY
HÈÝX›JËš]ÊKˆ
[œÚYÛ™YÛ™ÈÛ™ÊYËœ™]\ØYYˆÝX›JË™ÝX\™ž]\ÊHÈLMÍ‹Œ
NÂˆËÈÒUHÓÑKTT‹Q”SQHÓPÖHÐU‘Q\ÈHÚ\™HÙˆÚ]H™K\\MÂˆËÈ™[™\™\ˆÛÝ[]™H\ÚY8 %™XØ]\ÙH[ˆ\›HÚ]›ÈÛÝ[\ˆØ[››Ý™HÚÝÛˆÂˆËÈ]™H[™ØYÙY
ÛÝÚHMLJK[™™XØ]\ÙH\È˜][ÈTÈH][Nˆ]\ÈBˆËÈ™Y[™[˜ÞH˜XÝÜˆ™]ÙY[ˆ™]Ú\È[™\Ý[˜Ý^\™\È[ˆHœ˜[YKÚXÚˆËÈ›Ý[™È[ˆ\È[[YHY]™\ˆYX\Ý\™Y‚ˆÂˆÛÛœÝZ[ÝÛÝ[HËš]È
ÈËœÚÚ\YØ[YQœ˜[YNÂˆœš[ŠÝ\œ‹ˆ–Ýš×H^\™HÝX\™ØY[˜ÙNˆ	[HÙˆ	[HÚXÚÜÈÚÚ\Y™XØ]\ÙH‚ˆH[žHØ\ÈS‘PQH˜[Y]Y\Èœ˜[YH
	KŒY‰IKK™Kˆ	KŒYž\ÜÈ‚ˆš\Ú[™ÊI\×ˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊYËœÚÚ\YØ[YQœ˜[YK
[œÚYÛ™YÛ™ÈÛ™Ê]ÛÝ[ˆÛÝ[ÈLŒ
ˆÝX›JËœÚÚ\YØ[YQœ˜[YJHÈÝX›JÛÝ[
HˆŒˆËš]ÈÈÝX›JÛÝ[
HÈÝX›JËš]ÊHˆŒˆ×Ý^ÝX\™]™\žQ™]ÚÈˆÐÖ—Õ’×ÕVÑÕPT‘ÑU‘T–WÑ‘UÒ8 %^XÝH‚ˆˆˆŠNÂˆBˆËÈ‹‹˜[™ÒT‘HÜÙHž]\ÈÙ[žH^\™HÚ^™Kˆ™XY]YØZ[œÝH\‹XY™\ÜÂˆËÈÚ[™ÙYX›H™[ÝÎˆH™Yš^›Ý[™X›Ý™H]™\žHÚ^™H]\X\œÈ\™HÛÜÝÂˆËÈ›Ý[™È[ˆ]XÝ[Ûˆ[™Ø]™\È]™\ž][™ÈX›Ý™H]ˆž]\È\™HÚ]HÝX\™ˆËÈPÕPSH™XY
HØ[\Y][™XYHØ\ÈHšYÈÝ\™˜XÙH]HÝ™X[BˆËÈ›Ý[™
KÛÈHÛÛ[[ˆ\ÈHYHÛÜÝ[™›ÝHÚ^™HÝ[K‚ˆÂˆZ[ÝÝHÂˆ›Üˆ
Ú^™WÝˆHÈˆÕ^ÝX\™\ÝXÚÙ]ÎÈŠÊÊBˆÝ
ÏH×Ý^ÝX\™\Ýž]\ÖØ—NÂˆœš[ŠÝ\œ‹–Ýš×H^\™HÝX\™ž]\ÈžHÓÕTÑHÚ^™H
ÚXÚÜËÓPˆ™XY
NˆŠNÂˆ›Üˆ
Ú^™WÝˆHÈˆÕ^ÝX\™\ÝXÚÙ]ÎÈŠÊÊBˆÂˆYˆ
Y×Ý^ÝX\™\ÝÛÝ[Ø—JBˆÛÛ[YNÂˆÚ\ˆÖÌM—NÂˆYˆ
ˆOH
BˆÛœš[ŠËÚ^™[ÙˆËRÈŠNÂˆ[ÙBˆÛœš[ŠËÚ^™[ÙˆË‰^RÈ‹Ú^™WÝ
JHŠNÂˆœš[ŠÝ\œ‹ˆ	\ÏI[KÉKŒY“PŠ	KŒ‰IJH‹Ëˆ
[œÚYÛ™YÛ™ÈÛ™ÊY×Ý^ÝX\™\ÝÛÝ[Ø—KˆÝX›J×Ý^ÝX\™\Ýž]\ÖØ—JHÈLMÍ‹ŒˆÝÈLŒ
ˆÝX›J×Ý^ÝX\™\Ýž]\ÖØ—JHÈÝX›JÝ
HˆŒ
NÂˆBˆœš[ŠÝ\œ‹—ˆŠNÂˆBˆYˆ
×Ý^ÝX\™Ú\ÛÛŠBˆœš[ŠÝ\œ‹–Ýš×H
ÒTÓÓ‘Qˆ]Ú\™HUTÕ™HLŒ	IH8 %HÙ[œÝ\È‚ˆš\ÈÛ›H\ÝÛÜHYˆ]Ø[ˆ[ÛÈ™\ÜHÜÚ]]™JWˆŠNÂˆËÈHY™\ÜÙ\ËÛÜœÝš\œÝˆH˜][È[Û™HØ[››ÝÙ\\˜]H›Û™H]\ÈHÔBˆËÈ™]Üš]\È]™\žHœ˜[YHˆœ›ÛH˜H\™ÙˆHÛÜ›	ÜÈ^\™\È\™HÜ›Û™È‹[™ˆËÈÜÙH\™HY™™\™[Y™XÝÈÚ]Y™™\™[š^\Ë‚ˆÝŽ™XÝÜÝŽœZ\Z[Ì—Ý^ÝX\™Yˆ›ÝÜÎÂˆ×Ý^ÝX\™YœË‘›Ü‘XXÚ
É—H
Z[ÝËÛÛœÝ^ÝX\™Y‰ˆŠHÂˆ›ÝÜË™[\XÙWØ˜XÚÊZ[Ì—Ý
ÊKŠNÂˆJNÂˆÝŽœÛÜ
›ÝÜË˜™YÚ[Š
K›ÝÜË™[™

K×JÛÛœÝ]]ÉˆKÛÛœÝ]]ÉˆŠHÂˆ™]\›ˆKœÙXÛÛ™˜Ú[™ÙYˆ‹œÙXÛÛ™˜Ú[™ÙYÂˆJNÂˆÚ^™WÝÚÝÛˆHÚ]Ú[™ÙHHÂˆ›Üˆ
ÛÛœÝ]]Éˆˆˆ›ÝÜÊBˆYˆ
‹œÙXÛÛ™˜Ú[™ÙY
Bˆ
ÊÝÚ]Ú[™ÙNÂˆœš[ŠÝ\œ‹–Ýš×H	^HÙˆ	^HØXÚY^\™HY™\ÜÙ\ÈÙ\™YÚ[™ÙY‚ˆ˜ž]\È]X\ÝÛ˜ÙNÈÛÜœÝ—ˆ‹ˆÚ]Ú[™ÙK›ÝÜËœÚ^™J
JNÂˆ›Üˆ
ÛÛœÝ]]ÉˆØY‹WHˆ›ÝÜÊBˆÂˆYˆ
XK˜Ú[™ÙYÚÝÛŠÊÈH
Bˆœ™XZÎÂˆËÈÜ˜Ðž]\È\ÈÛˆ\È[™H\ÈÙˆ\Îˆ]\ÈHÚ^™HH™Yš^›Ý[™\ÂˆËÈÈÛÝ™\ˆÈÙY\ÙYZ[™È\ÈY™\ÜÈÚ[™ÙK[™HÚÛHÜ[][Ûˆ\ÂˆËÈ›ÝÜËÛÈH[œÝÙ\ˆÈÚ]›Ý[™\ÈØY™Hˆ\È™XYX›HÝ˜ZYÚÙ™ˆ]‚ˆœš[ŠÝ\œ‹–Ýš×H	L	M^	KMH‰KLH	MÛHˆ	[HÙˆ	[H]È‚ˆœÝ[H
	KŒY‰IJWˆ‹ˆY‹KÚYKšZYÚK™›Ü›X]ˆ
[œÚYÛ™YÛ™ÈÛ™ÊXKœÜ˜Ðž]\Ë
[œÚYÛ™YÛ™ÈÛ™ÊXK˜Ú[™ÙYˆ
[œÚYÛ™YÛ™ÈÛ™ÊXKš]ËˆLŒ
ˆÝX›JK˜Ú[™ÙY
HÈÝX›JKš]ÊJNÂˆBˆBŸB‚‚‹ËÈOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOB‹ËÈÞ—Ü[[YHKYXYØ8 %H[Ø[ˆ[ˆ
\LKØÜËÜÝX[KYXÚË\[‹›Y0©ÌÈ][HJB‹ËÈOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOB‹ËÈ]™\ž][™Èœš[™Ë]\ÛÝ[XÚYKš[Y›ÜˆH]šXÙH]\È™]™\ˆÜ™X]Yˆ]™\žB‹ËÈ\ÚXØ[]šXÙHHØY\ˆÙY\ÈÚ]]Èš]™\‰ÜÈÝÛˆ˜[YH[™™\œÚ[Û‹HÛ™HB‹ËÈ™[™\™\ˆÛÝ[XÚËH™\]Z\™[Y[ÈX›H™\™XÝÛˆ]HQSH\›Ü›X]‹ËÈHTÐPHØ[\HÛÝ[È[™H]šXÙK[ØØ[Y[[ÜžKˆÛ™H[™H\ˆ˜XÝÛÈH^Y\‚‹ËÈØ[ˆ\ÝHH›ØÚÈ[È[ˆ\ÜÝYKˆ™]\›œÈ˜[ÙHÚ[ˆHØY\ˆ\È›È]šXÙHÜ‚‹ËÈHXÚÈ˜Z[ÈHX›H8 %H^]ÛÙHHØÜš\Ø[ˆ™XY‚˜›ÛÛšÔ™[™\™\—ÑXYÊ
BžÂˆÛÛœÝÚ\ŠˆH–ÙXY×H[Ø[ŽˆŽÂˆZ[Ì—ÝØY\•™\ˆH’×ÐTWÕ‘T”ÒSÓ—ÌWÌÂˆYˆ
šÑ[[Y\˜]R[œÝ[˜ÙU™\œÚ[ÛŠBˆšÑ[[Y\˜]R[œÝ[˜ÙU™\œÚ[ÛŠ	›ØY\•™\ŠNÂˆœš[ŠÝ\œ‹‰\ÈØY\ˆ[œÝ[˜ÙH™\œÚ[Ûˆ	]K‰]K‰]Wˆ‹’×Õ‘T”ÒSÓ—ÓPR“ÔŠØY\•™\ŠKˆ’×Õ‘T”ÒSÓ—ÓRS“ÔŠØY\•™\ŠK’×Õ‘T”ÒSÓ—ÔUÒ
ØY\•™\ŠJNÂ‚ˆšÐ\XØ][Û’[™›È\È’×ÔÕ•PÕT‘WÕTWÐTPÐUSÓ—ÒS‘“ÈNÂˆ\œ\XØ][Û“˜[YHH˜Þ—Ü[[YHKYXYÈŽÂˆ\˜\U™\œÚ[ÛˆH’×ÐTWÕ‘T”ÒSÓ—ÌWÌÎÂˆšÒ[œÝ[˜ÙPÜ™X]R[™›ÈXÚ^È’×ÔÕ•PÕT‘WÕTWÒS”ÕSÑWÐÔ‘PUWÒS‘“ÈNÂˆXÚKœ\XØ][Û’[™›ÈH	˜\ÂˆšÒ[œÝ[˜ÙH[œÝH’×Ó•SÒS‘NÂˆÛÛœÝšÔ™\Ý[\ˆHšÐÜ™X]R[œÝ[˜ÙJ	šXÚK[‹	š[œÝ
NÂˆYˆ
\ˆOH’×ÔÕPÐÑTÔÊBˆÂˆœš[ŠÝ\œ‹‰\ÈšÐÜ™X]R[œÝ[˜ÙHRSQˆšÔ™\Ý[	Y8 %›È[Ø[ˆØY\‹ÒPÑ‚ˆ\ØX›Hœ›ÛH\È›ØÙ\ÜÈ
\ÈH[Ø[ˆš]™\ˆ[œÝ[YÊWˆ‹ˆ[
\ŠJNÂˆ™]\›ˆ˜[ÙNÂˆBˆZ[Ì—ÝÛÝ[HÂˆšÑ[[Y\˜]T\ÚXØ[]šXÙ\Ê[œÝ	˜ÛÝ[[ŠNÂˆÝŽ™XÝÜšÔ\ÚXØ[]šXÙOˆ]šXÙ\ÊÛÝ[
NÂˆšÑ[[Y\˜]T\ÚXØ[]šXÙ\Ê[œÝ	˜ÛÝ[]šXÙ\Ë™]J
JNÂˆœš[ŠÝ\œ‹‰\È	]H\ÚXØ[]šXÙI\×ˆ‹ÛÝ[ÛÝ[OHHÈˆˆˆœÈŠNÂˆYˆ
]šXÙ\Ë™[\J
JBˆÂˆšÑ\Ý›ÞR[œÝ[˜ÙJ[œÝ[ŠNÂˆ™]\›ˆ˜[ÙNÂˆBˆËÈHØ[YHXÚÈ\ÈÜ™X]Q]šXÙNˆHš\œÝ\ØÜ™]HÔK[ÙHHš\œÝ]šXÙK‚ˆšÔ\ÚXØ[]šXÙHXÚÈH]šXÙ\ÖÌNÂˆ›Üˆ
šÔ\ÚXØ[]šXÙHˆ]šXÙ\ÊBˆÂˆšÔ\ÚXØ[]šXÙT›Ü\Y\ÈßNÂˆšÑÙ]\ÚXØ[]šXÙT›Ü\Y\Ê	œ
NÂˆYˆ
™]šXÙU\HOH’×ÔTÒPÐSÑU’PÑWÕTWÑTÐÔ‘UWÑÔJBˆÂˆXÚÈHÂˆœ™XZÎÂˆBˆBˆ›ÛÛÚÈHYNÂˆ›Üˆ
Z[Ì—ÝHHÈHÛÝ[ÈJÊÊBˆÂˆ]šXÙPØ\ÈÎÂˆ]Y\žQ]šXÙPØ\Ê]šXÙ\ÖÚWKÊNÂˆœš[ŠÝ\œ‹‰\ÈÉ]WH	\È
	\ÊH[Ø[ˆ	]K‰]K‰]H™[™Üˆ	HÌž]šXÙH	HÌž	\×ˆ‹ˆKËœ›ÜË™]šXÙS˜[YK]šXÙU\S˜[YJËœ›ÜË™]šXÙU\JKˆ’×Õ‘T”ÒSÓ—ÓPR“ÔŠËœ›ÜË˜\U™\œÚ[ÛŠK’×Õ‘T”ÒSÓ—ÓRS“ÔŠËœ›ÜË˜\U™\œÚ[ÛŠKˆ’×Õ‘T”ÒSÓ—ÔUÒ
Ëœ›ÜË˜\U™\œÚ[ÛŠKËœ›ÜË™[™Ü’QËœ›ÜË™]šXÙRQˆ]šXÙ\ÖÚWHOHXÚÈÈˆHH™[™\™\ˆÛÝ[\ÙH\ÈÛ™HˆˆˆŠNÂˆÝŽœÝš[™ÈYÈHÝŽœÝš[™Ê
H
ÈˆŽÂˆš[š]™\“[™JËYË˜×ÜÝŠ
JNÂˆš[™[™\™\”›Ùš[JËYË˜×ÜÝŠ
JNÂˆYˆ
]šXÙ\ÖÚWHOHXÚÊBˆÛÛ[YNÂ‚ˆYˆ
Ëœ›ÜË˜\U™\œÚ[Ûˆ’×ÐTWÕ‘T”ÒSÓ—ÌWÌÊBˆÂˆœš[ŠÝ\œ‹‰\È‘T‘PÕˆÐS““Õ[ˆH™[™\™\ˆ8 %[Ø[ˆKŒÈ\È™\]Z\™Y‚ˆ˜[™\È]šXÙH™\ÜÈ	]K‰]Wˆ‹ˆ’×Õ‘T”ÒSÓ—ÓPR“ÔŠËœ›ÜË˜\U™\œÚ[ÛŠK’×Õ‘T”ÒSÓ—ÓRS“ÔŠËœ›ÜË˜\U™\œÚ[ÛŠJNÂˆÚÈH˜[ÙNÂˆBˆšÔ\ÚXØ[]šXÙU[Ø[ŒL‘™X]\™\ÈŒLžÂˆ’×ÔÕ•PÕT‘WÕTWÔTÒPÐSÑU’PÑWÕ•SÐS—ÌWÌ—Ñ‘PUT‘TÂˆNÂˆšÔ\ÚXØ[]šXÙU[Ø[ŒLÑ™X]\™\ÈŒLÞÂˆ’×ÔÕ•PÕT‘WÕTWÔTÒPÐSÑU’PÑWÕ•SÐS—ÌWÌ×Ñ‘PUT‘TÂˆNÂˆšÔ\ÚXØ[]šXÙQ™X]\™\Ìˆ™ŒžÈ’×ÔÕ•PÕT‘WÕTWÔTÒPÐSÑU’PÑWÑ‘PUT‘T×ÌˆNÂˆÝŽ™XÝÜÛÛœÝÚ\ŠˆZ\ÜÚ[™ÎÂˆœš[ŠÝ\œ‹‰\È™X]\™\ÈH™[™\™\ˆ\ÚÜÈ›ÜŽ—ˆ‹
NÂˆ]˜[X]T™\]Z\™[Y[ÊË™Œ‹ŒL‹ŒLËZ\ÜÚ[™ËÊ›\Ý[J‹ÝYJNÂˆYˆ
Z\ÜÚ[™Ë™[\J
JBˆœš[ŠÝ\œ‹‰\È]™\žH‘TURT‘Q™X]\™H\È™\Ù[ˆ‹
NÂˆ[ÙBˆÂˆœš[ŠÝ\œ‹‰\È‘T‘PÕˆÐS““Õ[ˆH™[™\™\ˆ8 %Z\ÜÚ[™È‘TURT‘Qˆ‹
NÂˆ›Üˆ
ÛÛœÝÚ\ŠˆHˆZ\ÜÚ[™ÊBˆœš[ŠÝ\œ‹ˆ	\È‹JNÂˆœš[ŠÝ\œ‹—ˆŠNÂˆÚÈH˜[ÙNÂˆBˆœš[ŠÝ\œ‹‰\È’×ÒÒ—ÜÝØ\ÚZ[Žˆ	\È
™YYYÈ™\Ù[[ÈHÚ[™ÝÊWˆ‹ˆË’\Ñ^
’×ÒÒ—ÔÕÐTÒRS—ÑVS”ÒSÓ—ÓSQJHÈœ™\Ù[ˆˆP”ÑS•ŠNÂˆœš[ŠÝ\œ‹‰\È˜^H]Y\žH
\šÙY™X]\™JNˆ	\×ˆ‹ˆ
Ë’\Ñ^
•’×ÒÒ—ØXØÙ[\˜][Û—ÜÝXÝ\™HŠH	‰ˆË’\Ñ^
•’×ÒÒ—Ü˜^WÜ]Y\žHŠBˆ	‰ˆË’\Ñ^
•’×ÒÒ—ÙY™\œ™YÚÜÝÛÜ\˜][ÛœÈŠJBˆÈœÝ\ÜYˆˆ[œÝ\ÜY
š[™NÈ•\È\šÙY
HŠNÂ‚ˆ›ÛÛH˜[ÙNÂˆÛÛœÝšÑ›Ü›X]\HXÚÑY˜[Q\›Ü›X]
]šXÙ\ÖÚWK	™
NÂˆœš[ŠÝ\œ‹‰\ÈÕS“Ô“WÔÎÕRS•Ø[\XX›Nˆ	\ÈOˆQSH\›Ü›X]	\É\×ˆ‹ˆÈžY\Èˆˆ››È‹ˆ\OH’×Ñ“Ô“PUÑÕS“Ô“WÔÎÕRS•È‘ÕS“Ô“WÔÎÕRS•‚ˆˆ‘Ì—ÔÑ“ÐUÔÎÕRS•‹ˆ[“ÛŠÖ—Õ’×ÑTÑ“ÐUŠHÈˆ
Ö—Õ’×ÑTÑ“ÐU
HˆˆˆŠNÂˆšÑ›Ü›X]›Ü\Y\ÈœßNÂˆšÑÙ]\ÚXØ[]šXÙQ›Ü›X]›Ü\Y\Ê]šXÙ\ÖÚWK’×Ñ“Ô“PUÑÌ—ÔÑ“ÐUÔÎÕRS•	™œ
NÂˆYˆ
Jœ›Ü[X[[[™Ñ™X]\™\È	ˆ’×Ñ“Ô“PUÑ‘PUT‘WÑTÔÕSÒSÐUPÒQS•Ð’U
JBˆœš[ŠÝ\œ‹‰\ÈÌ—ÔÑ“ÐUÔÎÕRS•\È›ÝH\]XÚY[\™H8 %H‚ˆ™]šXÙHÚ]™Z]\ˆ\›Ü›X]Ø[››Ý[ˆH™[™\™\—ˆ‹
NÂˆÛÛœÝšÔØ[\PÛÝ[›YÜÈÝ\HËœ›ÜË›[Z]Ë™œ˜[YXY™™\ÛÛÜ”Ø[\PÛÝ[Âˆ	ˆËœ›ÜË›[Z]Ë™œ˜[YXY™™\‘\Ø[\PÛÝ[ÎÂˆœš[ŠÝ\œ‹‰\Èœ˜[YXY™™\ˆØ[\HÛÝ[ÎˆÛÛÝ\ˆ	HÞ\	HÞOˆHžTÐPH‚ˆ™Y˜][	\×ˆ‹ˆ[œÚYÛ™Y
Ëœ›ÜË›[Z]Ë™œ˜[YXY™™\ÛÛÜ”Ø[\PÛÝ[ÊKˆ[œÚYÛ™Y
Ëœ›ÜË›[Z]Ë™œ˜[YXY™™\‘\Ø[\PÛÝ[ÊKˆ
Ý\	ˆŠHÈš\È]˜Z[X›Hˆˆ
Ý\	ˆ
HÈš\È[˜]˜Z[X›NÈÛÝ[™H\ÙY‚ˆˆš\È[˜]˜Z[X›NÈÚ[™ÛK\Ø[\HŠNÂˆšÔ\ÚXØ[]šXÙSY[[ÜžT›Ü\Y\È\ßNÂˆšÑÙ]\ÚXØ[]šXÙSY[[ÜžT›Ü\Y\Ê]šXÙ\ÖÚWK	›\
NÂˆZ[ÝØØ[HÂˆ›Üˆ
Z[Ì—ÝHÈ\›Y[[ÜžRX\ÛÝ[È
ÊÊBˆYˆ
\›Y[[ÜžRX\ÖÚK™›YÜÈ	ˆ’×ÓQSSÔ–WÒPTÑU’PÑWÓÐÐSÐ’U
BˆØØ[
ÏH\›Y[[ÜžRX\ÖÚKœÚ^™NÂˆœš[ŠÝ\œ‹‰\È]šXÙK[ØØ[Y[[ÜžNˆ	[HPˆ[ˆ	]HX\	\ÎÈX^‘[XYÙH	]NÈ‚ˆœØ[\Y[XYÙ\È\ˆÝYÙH	]Wˆ‹ˆ
[œÚYÛ™YÛ™ÈÛ™ÊJØØ[ˆŒ
K\›Y[[ÜžRX\ÛÝ[ˆ\›Y[[ÜžRX\ÛÝ[OHHÈˆˆˆœÈ‹Ëœ›ÜË›[Z]Ë›X^[XYÙQ[Y[œÚ[ÛŒ‘ˆËœ›ÜË›[Z]Ë›X^\”ÝYÙQ\ØÜš\Ü”Ø[\Y[XYÙ\ÊNÂˆœš[ŠÝ\œ‹‰\È[Y\Ý[\Îˆ\š[Ù	KŒÙˆœÉ\×ˆ‹ˆÝX›JËœ›ÜË›[Z]Ë[Y\Ý[\\š[Ù
KˆËœ›ÜË›[Z]Ë[Y\Ý[\\š[ÙˆÈˆˆˆˆ
ÔHœ˜[YH[YH[˜]˜Z[X›JHŠNÂˆBˆœš[ŠÝ\œ‹‰\È‘T‘PÕˆH™[™\™\ˆ	\ÈÛˆ\ÈXXÚ[™IÜÈXÚ×ˆ‹ˆÚÈÈÐSˆ[ˆˆˆÐS““Õ[ˆŠNÂˆšÑ\Ý›ÞR[œÝ[˜ÙJ[œÝ[ŠNÂˆ™]\›ˆÚÎÂŸB