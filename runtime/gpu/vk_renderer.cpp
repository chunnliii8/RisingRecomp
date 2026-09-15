y absorbed,
    // because "the snapshot is the right size" and "the snapshot is FULL" are different
    // claims and only the second one makes a shadow map usable.
    static const bool smallEdram = EnvOn("CZ_VK_SMALL_EDRAM");
    // `edramWidth`/`edramHeight`, not the image's: every extent from here to the copy
    // is in the title's own pixels and is multiplied once, at the vkCmd calls.
    const uint32_t w = smallEdram ? std::min(surfW, R->edramWidth)
                                  : std::min(surfW, kMaxSurfaceExtent);
    const uint32_t h = smallEdram ? std::min(surfH, R->edramHeight)
                                  : std::min(surfH, kMaxSurfaceExtent);
    if (!smallEdram && (surfW > kMaxSurfaceExtent || surfH > kMaxSurfaceExtent))
        Count("resolve: destination surface larger than the snapshot cap");
    const uint32_t copyX = std::min(wx, w);
    const uint32_t copyY = std::min(wy, h);
    uint32_t copyW = wx1 > wx ? std::min(wx1, w) - copyX : w - copyX;
    uint32_t copyH = wy1 > wy ? std::min(wy1, h) - copyY : h - copyY;
    // Bound by the EDRAM we can read from, and say so when that bites.
    const uint32_t availW = copyX < R->edramWidth ? R->edramWidth - copyX : 0;
    const uint32_t availH = copyY < R->edramHeight ? R->edramHeight - copyY : 0;
    if (copyW > availW || copyH > availH)
        Count("resolve: copy region clipped by the EDRAM stand-in's size");
    copyW = std::min(copyW, availW);
    copyH = std::min(copyH, availH);

    // WHERE THE COPY LANDS IN THE SNAPSHOT, which is not always where it was read from.
    //
    // The scissor says where in the EDRAM the pass rendered; it is the SOURCE offset and
    // always right. The DESTINATION offset is where in the destination surface those
    // pixels belong, and this title has TWO ways of saying that — move the scissor, or
    // pre-offset RB_COPY_DEST_BASE. The `baseKey` subtraction above understands the
    // first. This understands the second, and until part 31 nothing did.
    //
    // The case that needs it is the SHADOW ATLAS (§6bc). Four cascades share one
    // 4096x1024 surface; each resolves a 1024x1024 region with the scissor at the
    // origin, and they are told apart only by a destination address 0x20000 apart. In
    // Xenos tiled address space that is exactly +1024 texels in X: a 32bpp macro tile is
    // 32x32 texels = 4096 bytes, a 4096-wide surface is 128 tiles per tile row, so
    // +32 tiles = 32 * 4096 = 0x20000. Without this the four become four disjoint
    // snapshots each holding its own quarter, the consumer fetches the base address, and
    // three quarters of every shadow lookup reads zero. Measured: our atlas was 86.7%
    // zero where hardware's, dumped from the same capture, is 3.5%.
    uint32_t dstX = copyX;
    uint32_t dstY = copyY;
    static const bool noAddrFold = EnvOn("CZ_VK_NO_ADDR_TILE_FOLD");
    // Only when the copy does not cover the surface's width can a horizontal offset
    // exist at all, and only when the scissor is at the origin is the address the thing
    // carrying it — otherwise `baseKey` has already accounted for it and folding again
    // would double-count.
    if (!noAddrFold && copyW < surfW && wx == 0 && wy == 0 && surfW >= 32 && copyW)
    {
        const uint32_t tilesPerRow = surfW >> 5;
        bool found = false;
        uint32_t bestBase = 0, bestX = 0, bestY = 0;
        for (const auto& [k, s] : R->snapshots)
        {
            // Same kind of surface, same extent, at a LOWER address: those three
            // together are what make "this is a sub-region of that" a decode rather
            // than a guess. Two unrelated surfaces of identical shape 0x20000 apart
            // would still fold, so the arm above exists and the fold is counted.
            if (((k & kSnapshotDepthBit) != 0) != fromDepth)
                continue;
            const uint32_t b = k & 0x1FFFFFFF;
            if (b >= baseKey || s.guestW != w || s.guestH != h)
                continue;
            const uint32_t delta = baseKey - b;
            if (delta & 0xFFF)          // not a whole number of macro tiles
                continue;
            // A sub-region has to be inside the allocation. Without this the decode
            // relies on the `ty + copyH > surfH` test below to reject far-apart
            // surfaces of the same shape, which it does — but by arithmetic accident
            // rather than by saying what it means. `0684B000` and `1439B000` are both
            // 1280x720 in this title and 0xD150000 apart, and the frame's first tile
            // (scissor at the origin, 640 of 1280 wide) asks this question of them
            // every frame.
            if (uint64_t(delta) >= uint64_t(surfW) * surfH * 4)
                continue;
            const uint32_t tile = delta >> 12;
            const uint32_t tx = (tile % tilesPerRow) << 5;
            const uint32_t ty = (tile / tilesPerRow) << 5;
            if (tx + copyW > surfW || ty + copyH > surfH)
                continue;
            // Nearest base below, so a surface that is itself a sub-region of a bigger
            // one folds into its immediate parent rather than the earliest match.
            if (!found || b > bestBase)
            {
                found = true;
                bestBase = b;
                bestX = tx;
                bestY = ty;
            }
        }
        if (found && (bestX || bestY))
        {
            baseKey = bestBase;
            dstX = bestX;
            dstY = bestY;
            Count("resolve: destination address folded into an existing surface as a "
                  "tile offset");
        }
    }

    // THE DECISION, printed next to the registers it came from. `avail` is the EDRAM
    // stand-in's extent, and it is the term that silently truncates a copy: a pass may
    // legitimately ask for more rows than our EDRAM has, and the only symptom is a
    // snapshot that is the right SIZE and partly empty — which reads downstream as a
    // shadow map full of zeros, i.e. as fully occluded rather than as missing.
    if (traceThisPass)
        fprintf(stderr,
                "[vkresolve]     -> snapshot %08X%s %ux%u  copy %ux%u from EDRAM(%u,%u) "
                "to dst(%u,%u)  avail %ux%u  clr c=%d d=%d depthClear=%08X\n",
                baseKey, fromDepth ? "(depth)" : "", w, h, copyW, copyH, copyX, copyY,
                dstX, dstY, availW, availH, int(clearColor), int(clearDepth),
                regs[xenos::kRbDepthClear]);

    if (w && h && copyW && copyH)
    {
        const uint32_t key = baseKey | (fromDepth ? kSnapshotDepthBit : 0u);
        auto it = R->snapshots.find(key);
        // A destination whose extent changed is a different surface reusing an
        // address, so the image is rebuilt rather than partially overwritten — a
        // partial overwrite leaves the previous surface's pixels around the edge of
        // the new one, which reads as a ghosting artefact with no obvious source.
        if (it != R->snapshots.end() &&
            (it->second.guestW != w || it->second.guestH != h ||
             it->second.builtW != passW || it->second.builtH != passH))
        {
            // RETIRED, not destroyed — and NO wait-idle. Draws recorded earlier in
            // THIS frame may sample the old image (their descriptors stay valid
            // because the image stays alive), and a wait-idle here could not have
            // protected them anyway: it only covers submitted work, which is the
            // whole shadow-tier freeze (see RetiredImage). The old bindless slots
            // are still not recycled (open-items 3b).
            RetireImage(it->second.image);
            for (auto& [size, view] : it->second.views)
            {
                (void)size;
                RetireImage(view.image);
            }
            if (it->second.rtAttachView)
            {
                // The RT trace pass's attachment view rides the image's lifetime;
                // wrap it so the fence-aware queue destroys it with the image.
                Image v{};
                v.view = it->second.rtAttachView;
                RetireImage(v);
            }
            R->snapshots.erase(it);
            TexGenBump();
            it = R->snapshots.end();
            Count("resolve: snapshot resized");
        }
        if (it == R->snapshots.end() && R->nextTextureSlot < g_maxDescriptors)
        {
            Snapshot s;
            s.slot = R->nextTextureSlot++;
            s.fromDepth = fromDepth;
            s.guestW = w;
            s.guestH = h;
            s.builtW = passW;
            s.builtH = passH;
            // A depth snapshot keeps the EDRAM depth buffer's own format, because
            // vkCmdCopyImage is only defined between identical depth formats — there
            // is no copy from a depth image into a colour one. It is viewed through
            // the DEPTH aspect with every component reading that value, so a shader
            // sampling it gets the 24-bit depth in .r (which is what a Xenos `tfetch`
            // of a `k_24_8` surface returns) and a defined value in .gba rather than
            // Vulkan's undefined non-red components of a depth view.
            const VkComponentMapping depthSwizzle{
                VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
                VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ONE
            };
            if (CreateImage(s.image, RZx(w), RZ(h),
                            fromDepth ? R->depth.format : VK_FORMAT_R8G8B8A8_UNORM,
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                VK_IMAGE_USAGE_SAMPLED_BIT |
                                // RT stage 2 traces INTO depth snapshots; the bit
                                // costs nothing when nothing renders into it.
                                ((fromDepth && R->rtEnabled)
                                     ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                                     : 0),
                            fromDepth ? VK_IMAGE_ASPECT_DEPTH_BIT
                                      : VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_VIEW_TYPE_2D, 1, 1,
                            fromDepth ? depthSwizzle : VkComponentMapping{}))
            {
                VkDescriptorImageInfo ii{};
                ii.imageView = s.image.view;
                ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                VkWriteDescriptorSet wr{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                wr.dstSet = R->sets[0];
                wr.dstBinding = 0;
                wr.dstArrayElement = s.slot;
                wr.descriptorCount = 1;
                wr.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                wr.pImageInfo = &ii;
                // TRANSITION IT BEFORE ANYTHING CAN SEE THE DESCRIPTOR, in its own
                // immediate submit. The copy a few lines below already leaves it in
                // SHADER_READ_ONLY, so this looks redundant — and it is not, because the
                // copy is recorded into the FRAME's command buffer while the descriptor
                // becomes visible to that whole command buffer the instant it is written.
                // Every draw recorded EARLIER in the same frame is bound to the same
                // bindless heap, and a descriptor claiming SHADER_READ_ONLY on an image
                // that is still UNDEFINED is undefined CONTENT for anything that indexes
                // it. Nothing does — a draw can only learn this slot number from a lookup
                // that would have missed — but "nothing indexes it" is an argument and
                // this is a guarantee.
                //
                // It is also all 14 of `vkCmdDraw-None-09600`, the validation defect part
                // 25's hand-off said to chase first (open item 00d). The layer named them
                // once images carried names: six resolve snapshots in a halving chain
                // (96x45, 64x22, 32x11, 32x5, 32x2, 32x1 — a bloom pyramid) plus their
                // second and third occurrences, every one created mid-frame.
                RunImmediate([&](VkCommandBuffer cb) {
                    Barrier(cb, s.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            fromDepth ? VK_IMAGE_ASPECT_DEPTH_BIT
                                      : VK_IMAGE_ASPECT_COLOR_BIT);
                });
                vkUpdateDescriptorSets(R->device, 1, &wr, 0, nullptr);
                NameImage(s.image, "resolve snapshot %08X %ux%u%s slot %u", baseKey, w, h,
                          fromDepth ? " DEPTH" : "", s.slot);
                // The tier-engagement line the overnight gate greps for: HOST extents,
                // the tier scale and the scene scale, on the one surface the tier
                // governs. One line per (re)creation, not per frame.
                if (passH != InternalH())
                    fprintf(stderr,
                            "[vk] shadow-tier snapshot %08X: %ux%u host (guest %ux%u, "
                            "shadow pass %ux%u, scene %ux%u)\n",
                            baseKey, RZx(w), RZ(h), w, h, passW, passH, InternalW(),
                            InternalH());
                it = R->snapshots.emplace(key, std::move(s)).first;
                TexGenBump();
                Count("resolve: snapshot created");
            }
            else
            {
                --R->nextTextureSlot;
                Count("resolve: snapshot image creation failed");
            }
        }
        if (it != R->snapshots.end())
        {
            // The source EDRAM buffer, and the aspect that goes with it. A depth
            // resolve copies out of R->depth: the whole point of reading
            // copy_src_select is that these two are different pictures.
            //
            // CZ_VK_MSAA: a multisampled image cannot be vkCmdCopyImage'd into a
            // single-sample snapshot. Colour resolves in place of the copy below;
            // DEPTH has no vkCmdResolveImage, so it goes through a zero-draw
            // rendering pass whose resolve attachment writes R->depthResolve, and
            // the existing copy then reads THAT image with the same offsets (it is
            // the EDRAM's extent exactly).
            const bool msaaOn = R->msaaSamples != VK_SAMPLE_COUNT_1_BIT;
            Image& src = fromDepth ? (msaaOn ? R->depthResolve : R->depth) : R->color;
            const VkImageAspectFlags aspect =
                fromDepth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
            BeginFrame();
            EndRendering();
            // A deferred clear still outstanding here means a clear-then-copy with no
            // pass in between; the copy must see the cleared pixels, so emit it now.
            FlushPendingClears("clear: deferred FLUSHED for a resolve copy");
            if (msaaOn && fromDepth)
            {
                // The depth resolve is REGIONAL: dynamic rendering resolves exactly
                // the renderArea, so only the rectangle this copy needs is paid for.
                // Charged to the resolve-copy class — it IS the resolve's device work.
                GpuSeg _gdr(kGpResolveCopy);
                Barrier(R->cmd, R->depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
                // The RESOLVE TARGET's barrier is written out rather than going through
                // Barrier(), because the resolve-attachment WRITE at vkCmdEndRendering is
                // performed at COLOR_ATTACHMENT_OUTPUT with COLOR_ATTACHMENT_WRITE access
                // — even for a DEPTH resolve attachment — and LayoutMasks' depth-
                // attachment scope (EARLY|LATE fragment tests, DS access) does not cover
                // it. Sync validation reported exactly that, 10 WRITE-AFTER-WRITE hazards
                // against this image, on this arm's first gate run. The dst scope below
                // is the union of both models. (This one site bypasses the
                // CZ_VK_BARRIER_POISON/WIDE arms — a stated caveat, not an oversight.)
                {
                    VkImageMemoryBarrier bb{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
                    bb.oldLayout = R->depthResolve.layout;
                    bb.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    bb.image = R->depthResolve.image;
                    bb.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT |
                                                VK_IMAGE_ASPECT_STENCIL_BIT,
                                            0, 1, 0, 1 };
                    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                    bb.srcAccessMask = 0;
                    LayoutMasks(bb.oldLayout, srcStage, bb.srcAccessMask);
                    const VkPipelineStageFlags dstStage =
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                    bb.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                    vkCmdPipelineBarrier(R->cmd, srcStage, dstStage, 0, 0, nullptr, 0,
                                         nullptr, 1, &bb);
                    R->depthResolve.layout = bb.newLayout;
                    ++g_barrierN;
                }
                VkRenderingAttachmentInfo da{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
                da.imageView = R->depth.view;
                da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                da.resolveMode = R->depthResolveMode;
                da.resolveImageView = R->depthResolve.view;
                da.resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                da.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                VkRenderingInfo rri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
                rri.renderArea = { { RZxi(int32_t(copyX)), RZyi(int32_t(copyY)) },
                                   { RZx(copyW), RZ(copyH) } };
                rri.layerCount = 1;
                rri.pDepthAttachment = &da;
                rri.pStencilAttachment = &da;
                vkCmdBeginRendering(R->cmd, &rri);
                vkCmdEndRendering(R->cmd);
                // ...and straight to TRANSFER_SRC for the copy below, mirrored: the
                // SOURCE scope must also name the resolve write's
                // COLOR_ATTACHMENT_OUTPUT/COLOR_ATTACHMENT_WRITE model, which
                // LayoutMasks' depth-attachment entry does not. Emitted here so the
                // generic Barrier() in the copy block early-returns on the layout.
                {
                    VkImageMemoryBarrier bb{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
                    bb.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                    bb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    bb.image = R->depthResolve.image;
                    bb.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT |
                                                VK_IMAGE_ASPECT_STENCIL_BIT,
                                            0, 1, 0, 1 };
                    bb.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                    bb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    vkCmdPipelineBarrier(R->cmd,
                                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                             VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                                         0, nullptr, 1, &bb);
                    R->depthResolve.layout = bb.newLayout;
                    ++g_barrierN;
                }
                Count("resolve: depth resolved out of the multisampled EDRAM");
            }
            // The whole resolve's device work is one segment, barriers included. The first
            // version timed only the `vkCmdCopyImage` and left the two layout transitions
            // in the residual, where they are indistinguishable from work nobody wrapped —
            // and a transition of a 3440x1440 attachment is not a free instruction.
            {
            GpuSeg _gb(kGpResolveBarrier);
            // The EDRAM depth image is tracked with both aspects everywhere else, so
            // its barriers carry both here too; the snapshot has only depth.
            Barrier(R->cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    fromDepth ? (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
                              : VK_IMAGE_ASPECT_COLOR_BIT);
            Barrier(R->cmd, it->second.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    aspect);
            }
            {
            GpuSeg _gres(kGpResolveCopy);
            // Copy the TILE, at its own position in each image. For a scissor-offset
            // tile the two offsets are the same, because our EDRAM is full-screen-sized
            // and the window offset is deliberately not applied to the geometry (see the
            // scissor note in DoDraw), so a tile sits at its true screen position in
            // both. For an ADDRESS-offset sub-region they differ: the pass rendered at
            // the EDRAM origin and the pixels belong somewhere else in the destination
            // surface, which is what `dstX`/`dstY` carry.
            VkImageCopy copy{};
            copy.srcSubresource = { aspect, 0, 0, 1 };
            copy.srcOffset = { RZxi(int32_t(copyX)), RZyi(int32_t(copyY)), 0 };
            copy.dstSubresource = { aspect, 0, 0, 1 };
            copy.dstOffset = { RZxi(int32_t(dstX)), RZyi(int32_t(dstY)), 0 };
            copy.extent = { RZx(copyW), RZ(copyH), 1 };
            // The PIXEL count beside the milliseconds: a bandwidth cost is only
            // designable once you know how many pixels it moves — 48.9 resolves a frame at
            // an unknown extent is not a number anyone can act on.
            g_gpResolvePixels += uint64_t(RZx(copyW)) * uint64_t(RZ(copyH));
            if (g_copyCensusOn)
                CopyCensusCopy(key, RZxi(int32_t(dstX)), RZyi(int32_t(dstY)),
                               RZx(copyW), RZ(copyH),
                               uint64_t(RZx(copyW)) * uint64_t(RZ(copyH)));
            if (msaaOn && !fromDepth)
            {
                // The MSAA resolve, where the single-sample renderer copies. Same
                // region, same layouts; VkImageResolve is field-for-field the copy
                // struct. This is the plan's 1:1 mapping made literal: the guest's
                // RB_COPY *is* the resolve (docs/msaa-plan.md §1).
                VkImageResolve rv{};
                rv.srcSubresource = copy.srcSubresource;
                rv.srcOffset = copy.srcOffset;
                rv.dstSubresource = copy.dstSubresource;
                rv.dstOffset = copy.dstOffset;
                rv.extent = copy.extent;
                vkCmdResolveImage(R->cmd, src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  it->second.image.image,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rv);
                Count("resolve: colour resolved out of the multisampled EDRAM");
            }
            else
                vkCmdCopyImage(R->cmd, src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               it->second.image.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            // RT STAGE 2's INJECTION EXPERIMENT (part 64; rt-and-fov-plan.md §3,
            // route (a)). CZ_VK_SHADOW_FILL=<depth 0..1> overwrites the shadow
            // ATLAS snapshot's depths with a constant right after each cascade
            // resolve lands — the image is still TRANSFER_DST, so the fill costs
            // one clear and no extra barriers — and the title's own shadow
            // comparison then runs against OUR value instead of the rasterized
            // cascade's. This is the cheapest possible test of the injection
            // point: if the atlas is where the shadow term reads, one polarity
            // must shadow the whole world and the other must unshadow it, and a
            // frame that moves under NEITHER refutes the injection point itself
            // before anything is built on it. It doubles as stage 2's standing
            // POSITIVE CONTROL (the CZ_VK_CUBE_POISON pattern): the all-shadow
            // value must darken the frame for as long as the atlas is the
            // composite route. A DIAGNOSTIC ARM, never a fix; unset = off = the
            // shipped renderer.
            static const float shadowFill = []() -> float {
                const char* e = Env("CZ_VK_SHADOW_FILL");
                return e ? float(atof(e)) : -1.0f;
            }();
            if (fromDepth && IsShadowSurface(regs) && shadowFill >= 0.0f)
            {
                VkClearDepthStencilValue cv{ shadowFill, 0 };
                VkImageSubresourceRange range{ VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
                vkCmdClearDepthStencilImage(R->cmd, it->second.image.image,
                                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                            &cv, 1, &range);
                Count("resolve: shadow atlas depth FILLED (CZ_VK_SHADOW_FILL)");
            }
            // RT STAGE 2 (part 64): ray-trace this just-resolved cascade slice.
            // The pass renders into the slice's own rectangle with the depth test
            // keeping whichever occluder is nearer the sun, so raster content
            // (skinned actors, foliage, everything the TLAS excludes) survives.
            // Inert to one test when the tier is OG.
            // ROUTE (B)'s SUN LATCH, and it must happen BEFORE TraceSlice, which
            // consumes `g_lightMValid` on route (a). This is the whole fix for the
            // "no shadows at all" the operator's second session reported: only a
            // matrix that was captured on the way to a CASCADE RESOLVE can be the
            // sun's, and last-write-wins over a frame was picking up a top-down
            // ortho that is not one. See the g_sunM comment.
            if (fromDepth && IsShadowSurface(regs))
                rtshadow::LatchSun(baseKey);
            if (fromDepth && IsShadowSurface(regs))
                rtshadow::TraceSlice(base, it->second, RZxi(int32_t(dstX)),
                                     RZyi(int32_t(dstY)), RZx(copyW), RZ(copyH));
            // ROUTE (B): a resolve ends a pass, and this title renders in two 640-wide
            // tiles, so the depth buffer is about to describe a DIFFERENT region. A
            // factor computed for the previous tile would be a picture of the wrong
            // depth over the new one — recompute at the next atlas-sampling draw.
            rtfactor::Invalidate();
            // Back to SHADER_READ_ONLY immediately: a later pass in this same frame
            // samples this surface, and the layout it expects is the one the
            // descriptor was written with.
            }
            {
            GpuSeg _gb2(kGpResolveBarrier);
            Barrier(R->cmd, it->second.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    aspect);
            }
            // The right-sized views of this surface, refreshed in this same command
            // buffer so they cost no submit and cannot be staler than their source.
            // Their own class: they ride the resolve but they are a separate blit, and a
            // class that hides inside another cannot be acted on.
            if (!it->second.views.empty())
            {
                GpuSeg _gv(kGpSnapView);
                for (auto& [size, view] : it->second.views)
                {
                    (void)size;
                    RefreshSnapshotView(R->cmd, it->second.image, view, aspect);
                    Count("resolve: snapshot view refreshed");
                }
            }
            it->second.frameSeen = R->frame;
            Count(fromDepth ? "resolve: snapshot taken from the DEPTH buffer"
                            : "resolve: snapshot taken from the colour buffer");

            // A FACE OF A RENDERED CUBE MAP, copied into its layer in this same command
            // buffer. Same reasoning as the sized views above: it costs no submit and it
            // cannot be staler than its source. A cube map the title renders is redrawn as
            // the world changes, so a cube assembled once at its first fetch and never
            // refreshed would freeze whatever the environment looked like at that instant
            // — the exact defect the LUT had in §6s, one descriptor set over.
            if (!fromDepth)
            {
                auto owner = R->cubeFaceOwner.find(baseKey);
                if (owner != R->cubeFaceOwner.end())
                {
                    auto cube = R->cubeSnapshots.find(owner->second.first);
                    if (cube != R->cubeSnapshots.end())
                    {
                        GpuSeg _gc(kGpCubeFace);
                        // The cube assembly consumes this face's copy (the cube's own
                        // sampling is a separate question the census does not fold in).
                        if (g_copyCensusOn)
                            CopyCensusSampled(key);
                        CopyFaceIntoCube(R->cmd, it->second, cube->second,
                                         owner->second.second);
                        cube->second.frameSeen = R->frame;
                        Count("resolve: refreshed a face of a rendered CUBE MAP");
                    }
                }
            }

            if (!fromDepth && R->frontBuffer && key == (R->frontBuffer & 0x1FFFFFFF))
            {
                R->frontWidth = w;
                R->frontHeight = h;
                R->haveFrontSnapshot = true;
                Count("resolve: this is the frame");
            }
        }
    }

    if (!clearColor && !clearDepth)
        return;

    BeginFrame();
    EndRendering();

    // DEFERRED SCOPED CLEARS — the part-90 default (perf-plan-part90.md §1). The old
    // mechanism honoured the copy block's clear bits with a whole-EDRAM
    // vkCmdClear{Color,DepthStencil}Image: 83.8 clears a frame writing 590 Mpixel for
    // the 33.9 the passes rendered, 0.615 ms/frame of GPU at the crowd, plus a
    // TRANSFER_DST layout round-trip each. On Xenos a copy block's clear clears the
    // tiles of the CURRENT SURFACE, so the scoped rect is the more faithful form, not
    // less (the part-32 scoped arm measured scoped == full to four decimals on the
    // cascade statistic). The clear is LATCHED here and emitted as a
    // vkCmdClearAttachments at the head of the next pass's first instance — an
    // instance that already exists under both arms, which is what the part-32 arm
    // lacked (its dedicated mini-scope per clear cost 0.51 ms/frame of pump time, the
    // wash part 80 §1 item 4 records). Anything that reads EDRAM first flushes the
    // pendings (see FlushPendingClears); a resolve extent of zero latches the FULL
    // extent, so the un-scopable case degrades to exactly the old pixels.
    // CZ_VK_NO_DEFERRED_CLEAR=1 is the same-binary control arm and restores the old
    // paths byte for byte (including CZ_VK_SCOPED_CLEAR's).
    static const bool deferredClearOff = [] {
        const bool off = EnvOn("CZ_VK_NO_DEFERRED_CLEAR");
        fprintf(stderr,
                off ? "[vk] CZ_VK_NO_DEFERRED_CLEAR=1 — resolve clears take the OLD "
                      "whole-EDRAM path (the control arm)\n"
                    : "[vk] resolve clears are DEFERRED and SCOPED (part 90 default; "
                      "CZ_VK_NO_DEFERRED_CLEAR=1 is the control arm)\n");
        return off;
    }();
    // CZ_VK_DEFER_FULL_RECT=1 — DIAGNOSTIC: defer the clear (same latch, same emission
    // point, same ordering) but with the FULL-image rect, i.e. the old pixels through
    // the new mechanism. This is the two-factor bisection for any picture complaint
    // against the deferred clears: an artifact that vanishes under it indicts the
    // SCOPING; one that survives indicts the deferral/ordering itself. It EARNED ITS
    // KEEP on day one: the operator's giant-sun-glow blow-out vanished under it and
    // under CZ_VK_NO_DEFERRED_CLEAR while tracking the shipped default exactly
    // (an A/B/A by eye), which is what convicted the resolve-window scoping below.
    static const bool deferFullRect = [] {
        const bool on = EnvOn("CZ_VK_DEFER_FULL_RECT");
        if (on)
            fprintf(stderr, "[vk] CZ_VK_DEFER_FULL_RECT=1 — deferred clears use the "
                            "FULL-image rect (the scoping-vs-ordering diagnostic)\n");
        return on;
    }();
    // THE SCOPED RECT IS THE DESTINATION SURFACE'S FOOTPRINT, NOT THE RESOLVE WINDOW.
    // On Xenos the copy block's clear bits clear the tiles of the CURRENT SURFACE —
    // the whole surface, pitch x height, not just the window the resolve copied. The
    // first shipped form scoped to the resolve window and the operator convicted it
    // the same day with an A/B/A: a view-dependent giant sun glow (the title's
    // sun-visibility machinery reading EDRAM regions inside the surface footprint but
    // outside the resolve window — regions the whole-image clear had always wiped).
    // The surface footprint contains the resolve window by construction
    // (copyX+copyW <= w, copyY+copyH <= h), starts at the EDRAM origin where every
    // surface is laid out, and still removes the tall-EDRAM territory (the 1024-row
    // shadow band) plus everything beyond the surface from the class.
    //
    // THE MSAA FACTOR IS NOT OPTIONAL: a 4x surface's coordinates are in PIXELS while
    // our EDRAM stand-in is at SAMPLE resolution, twice as wide and twice as tall —
    // the draw path scales every window coordinate by exactly this factor, so the
    // footprint scales with it (gotcha 506).
    const uint32_t clearMsaa = (regs[xenos::kRbSurfaceInfo] >> 16) & 3;
    const uint32_t clearAxisScale = clearMsaa == 2 ? 2u : 1u;
    const auto scopedRect = [&](const Image& im) {
        VkRect2D r{};
        if (w && h && !deferFullRect)
        {
            r.offset = { 0, 0 };
            r.extent = { std::min(RZx(w * clearAxisScale), im.width),
                         std::min(RZ(h * clearAxisScale), im.height) };
        }
        else
            r.extent = { im.width, im.height };
        return r;
    };
    if (clearMsaa == 2 && (clearColor || clearDepth))
        COUNT("resolve: clear rect scaled for a 4x MSAA surface");

    if (clearColor)
    {
        // RB_COLOR_CLEAR holds the clear value in the render target's own format. It
        // is read as 8888 here; a target in another format would clear to the wrong
        // colour, which is why the count is separate from the resolve count.
        const uint32_t c = regs[xenos::kRbColorClear];
        // CZ_VK_CLEAR_POISON=1 — clear the colour target to MAGENTA instead of the guest's
        // value. The positive control for "these pixels were never written by any draw".
        //
        // Part 26's operator report is large ground patches of a single flat colour, and
        // that colour is EXACTLY (180,180,180) over 39.6% of one frame — 49,195 pixels of
        // one value with a standard deviation of 3. A constant that precise is not a
        // texture and not lighting; it is either a clear or a shader writing a constant.
        // The two look identical in a screenshot and have nothing in common as bugs, so
        // this separates them: under the arm, every pixel still showing the clear is
        // magenta and every pixel some draw actually wrote keeps its colour.
        //
        // The clear VALUE is printed once per distinct value too, because if the guest's
        // own clear is 0xB4B4B4 the question is answered without running the arm at all.
        static const bool clearPoison = EnvOn("CZ_VK_CLEAR_POISON");
        {
            static std::vector<uint32_t> seenClear;
            if (std::find(seenClear.begin(), seenClear.end(), c) == seenClear.end() &&
                seenClear.size() < 16)
            {
                seenClear.push_back(c);
                fprintf(stderr, "[vk] RB_COLOR_CLEAR = %08X  (a=%u r=%u g=%u b=%u)\n", c,
                        (c >> 24) & 0xFF, (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF);
            }
        }
        VkClearColorValue value{};
        value.float32[0] = float((c >> 16) & 0xFF) / 255.0f;
        value.float32[1] = float((c >> 8) & 0xFF) / 255.0f;
        value.float32[2] = float(c & 0xFF) / 255.0f;
        value.float32[3] = float((c >> 24) & 0xFF) / 255.0f;
        if (clearPoison)
        {
            value.float32[0] = 1.0f;
            value.float32[1] = 0.0f;
            value.float32[2] = 1.0f;
            value.float32[3] = 1.0f;
        }
        if (!deferredClearOff)
        {
            // The census counters keep their old meaning under both arms: FULL is what
            // the whole-image mechanism would write, SCOPED what the rect covers.
            ++g_gpClearN;
            g_gpClearFullPixels += uint64_t(R->color.width) * uint64_t(R->color.height);
            const VkRect2D rect = scopedRect(R->color);
            g_gpClearScopedPixels += uint64_t(rect.extent.width) * rect.extent.height;
            if (rect.extent.width && rect.extent.height)
            {
                PendingClear p{};
                p.isColor = true;
                p.value.color = value;
                p.rect = rect;
                R->pendingClears.push_back(p);
            }
            COUNT("resolve: colour clear deferred (scoped)");
        }
        else
        {
            // The transition INTO transfer-dst is inside the segment on purpose: a clear
            // that forces a layout change is not separable from the change, and charging
            // the barrier elsewhere would make this class read low for the wrong reason.
            GpuSeg _g(kGpResolveClear);
            // What this clear WRITES against what the pass it follows actually rendered.
            ++g_gpClearN;
            g_gpClearFullPixels += uint64_t(R->color.width) * uint64_t(R->color.height);
            g_gpClearScopedPixels +=
                (copyW && copyH) ? uint64_t(RZx(copyW)) * uint64_t(RZ(copyH))
                                 : uint64_t(R->color.width) * uint64_t(R->color.height);
            if (!Barrier(R->cmd, R->color, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_ASPECT_COLOR_BIT))
                TransferWriteBarrier(R->cmd, R->color, VK_IMAGE_ASPECT_COLOR_BIT);
            VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdClearColorImage(R->cmd, R->color.image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &value, 1, &range);
            COUNT("resolve: colour cleared");
        }
    }
    if (clearDepth)
    {
        VkClearDepthStencilValue value{};
        value.depth = float(regs[xenos::kRbDepthClear] >> 8) / float(0xFFFFFF);
        value.stencil = regs[xenos::kRbDepthClear] & 0xFF;
        // CZ_VK_DEPTH_CLEAR_FAR=1 — clear depth to 1.0 whatever RB_DEPTH_CLEAR says.
        // A DIAGNOSTIC ARM: it asks whether the cascade's empty half is empty because
        // the buffer it is tested against was cleared to 0 (this title leaves
        // RB_DEPTH_CLEAR at 00000000 for nearly every pass, and our clear covers the
        // whole EDRAM stand-in). If the atlas fills under this arm, the clear VALUE is
        // the whole story; if it does not, the zeros come from somewhere else.
        static const bool clearFar = EnvOn("CZ_VK_DEPTH_CLEAR_FAR");
        if (clearFar)
            value.depth = 1.0f;
        // The VALUE, once per distinct one, for the same reason the colour clear prints
        // its own: a depth buffer that starts at the wrong end is not a wrong picture,
        // it is a pass whose every fragment fails the test — and the symptom of that is
        // an EMPTY surface, which reads as "the geometry was never submitted".
        {
            static std::vector<uint32_t> seenDepthClear;
            const uint32_t dc = regs[xenos::kRbDepthClear];
            if (std::find(seenDepthClear.begin(), seenDepthClear.end(), dc) ==
                    seenDepthClear.end() &&
                seenDepthClear.size() < 16)
            {
                seenDepthClear.push_back(dc);
                fprintf(stderr, "[vk] RB_DEPTH_CLEAR = %08X  (depth %.6f, stencil %u)\n",
                        dc, value.depth, value.stencil);
            }
        }
        if (!deferredClearOff)
        {
            ++g_gpClearN;
            g_gpClearFullPixels += uint64_t(R->depth.width) * uint64_t(R->depth.height);
            const VkRect2D rect = scopedRect(R->depth);
            g_gpClearScopedPixels += uint64_t(rect.extent.width) * rect.extent.height;
            if (rect.extent.width && rect.extent.height)
            {
                PendingClear p{};
                p.isColor = false;
                p.value.depthStencil = value;
                p.rect = rect;
                R->pendingClears.push_back(p);
            }
            COUNT("resolve: depth clear deferred (scoped)");
            return;
        }
        if (!Barrier(R->cmd, R->depth, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))
            TransferWriteBarrier(R->cmd, R->depth,
                                 VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
        VkImageSubresourceRange range{
            VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1
        };
        // CZ_VK_SCOPED_CLEAR=1 — clear only the region THIS pass rendered, not the whole
        // EDRAM stand-in.
        //
        // On Xenos a copy block's clear bits clear the tiles of the CURRENT SURFACE. Our
        // EDRAM is one 1280x1024 image shared by every pass, so clearing all of it makes
        // a 64x64 post-chain pass wipe the 1024x1024 shadow cascade — and it wipes it to
        // RB_DEPTH_CLEAR, which this title leaves at 00000000 for almost every pass. A
        // depth buffer at 0 with a LESS test rejects every fragment, so the cascade comes
        // out empty, and a shadow lookup that reads 0 reads as OCCLUDED. Part 32 measured
        // exactly that: 46.875% of every cascade band is zero, and forcing the compare to
        // ALWAYS (CZ_VK_DEPTH_ALWAYS) fills it, so the geometry was always being
        // submitted and always being rejected.
        //
        // An ARM until it is measured, because it cuts both ways: a pass that legitimately
        // expects the whole surface cleared now gets only its scissor's worth.
        static const bool scopedClear = EnvOn("CZ_VK_SCOPED_CLEAR");
        if (scopedClear && copyW && copyH)
        {
            VkClearRect rect{};
            rect.rect.offset = { RZxi(int32_t(copyX)), RZyi(int32_t(copyY)) };
            rect.rect.extent = { RZx(copyW), RZ(copyH) };
            rect.baseArrayLayer = 0;
            rect.layerCount = 1;
            // vkCmdClearAttachments needs a render pass; outside one the region form is
            // a clear of the image with a scissor, which Vulkan has no direct call for —
            // so this does it inside a rendering scope over just that rectangle.
            VkRenderingAttachmentInfo depthAtt{
                VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO
            };
            depthAtt.imageView = R->depth.view;
            depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depthAtt.clearValue.depthStencil = value;
            Barrier(R->cmd, R->depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
            VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
            ri.renderArea = { { RZxi(int32_t(copyX)), RZyi(int32_t(copyY)) },
                              { RZx(copyW), RZ(copyH) } };
            ri.layerCount = 1;
            ri.pDepthAttachment = &depthAtt;
            ri.pStencilAttachment = &depthAtt;
            GpuSeg _g(kGpResolveClear);
            vkCmdBeginRendering(R->cmd, &ri);
            vkCmdEndRendering(R->cmd);
            Count("resolve: depth cleared (scoped to the pass)");
        }
        else
        {
            GpuSeg _g(kGpResolveClear);
            ++g_gpClearN;
            g_gpClearFullPixels += uint64_t(R->depth.width) * uint64_t(R->depth.height);
            g_gpClearScopedPixels +=
                (copyW && copyH) ? uint64_t(RZx(copyW)) * uint64_t(RZ(copyH))
                                 : uint64_t(R->depth.width) * uint64_t(R->depth.height);
            vkCmdClearDepthStencilImage(R->cmd, R->depth.image,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &value, 1,
                                        &range);
            COUNT("resolve: depth cleared");
        }
    }
}

} // namespace

// ===================================================================================
// The public seam
// ===================================================================================
bool VkRenderer_RtAvailable()
{
    // THREE REQUIREMENTS as of part 71. The feature has to be OFFERED (it is parked —
    // see rtshadow::MenuOffersRt), the device has to support ray query, AND the shader
    // variant cache has to be present with real variants in it — route (b) makes the
    // shadows in the material shaders, so a runtime with ray query and no
    // `assets/shader_spv_rt` would offer three RT rungs that change nothing. Route (a)
    // needs no variants (it writes the atlas), so the developer arm still works.
    return rtshadow::MenuOffersRt() && R && R->rtEnabled &&
           (R->rtVariants || !rtshadow::RouteB());
}

// Why the ladder stops, for the panel footer: 0 = it does not, 1 = no ray query,
// 2 = ray query but no shader variants, 3 = PARKED (part 71). The order matters: park
// first, because on a machine that can run RT perfectly well the honest answer is "we
// turned it off", not "your device cannot".
int VkRenderer_RtUnavailableReason()
{
    if (!rtshadow::MenuOffersRt())
        return 3;
    if (!R || !R->rtEnabled)
        return 1;
    if (!R->rtVariants && rtshadow::RouteB())
        return 2;
    return 0;
}

bool VkRenderer_Active() { return g_active; }

// D.4's enqueue seam, called by pm4.cpp's BindShader once per distinct hash per run
// (inside its announce-once block, so this is never on the per-bind path). Pump thread.
void VkRenderer_OnShaderBind(uint32_t type, uint64_t hash, const uint8_t* code,
                             uint32_t sizeDwords)
{
    if (!g_active)
        return;
    shaderjit::OnFirstBind(type, hash, code, sizeDwords);
}

namespace {

// CZ_VK_MSAA's single-sample companions, created beside the (now multisampled) EDRAM
// pair at bring-up and at every live rescale. A no-op returning true at 1x, so the
// control arm allocates nothing. `colorResolve` is TRANSFER_DST (vkCmdResolveImage
// writes it on the present fallback) + TRANSFER_SRC (the readback/blit reads it);
// `depthResolve` is a DEPTH_STENCIL_ATTACHMENT because the only defined way to resolve
// a depth image is a resolve attachment on a rendering pass (vkCmdResolveImage is
// colour-only), + TRANSFER_SRC for the snapshot copy that follows.
bool CreateEdramResolveTargets(uint32_t hostW, uint32_t hostH)
{
    if (R->msaaSamples == VK_SAMPLE_COUNT_1_BIT)
        return true;
    // SAMPLED on the colour companion is not read by anything — it is there because
    // CreateImage unconditionally builds a VkImageView and a view requires at least
    // one view-capable usage bit (VUID-VkImageViewCreateInfo-image-04441, caught by
    // the very first validation run of this arm).
    if (!CreateImage(R->colorResolve, hostW, hostH, VK_FORMAT_R8G8B8A8_UNORM,
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                         VK_IMAGE_USAGE_SAMPLED_BIT,
                     VK_IMAGE_ASPECT_COLOR_BIT) ||
        !CreateImage(R->depthResolve, hostW, hostH, EdramDepthFormat(),
                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                     VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))
        return false;
    NameImage(R->colorResolve, "EDRAM colour RESOLVE %ux%u", hostW, hostH);
    NameImage(R->depthResolve, "EDRAM depth RESOLVE %ux%u", hostW, hostH);
    return true;
}

// Device bring-up, shared by both feeds. Sets g_active on success; which feed owns
// the renderer is the CALLER's declaration (g_d3dMode), not decided here.
bool InitCommon()
{
    R = new Renderer();
    if (!CreateDevice() || !CreateDescriptorPlumbing())
    {
        fprintf(stderr, "[vk] device bring-up FAILED — running without a renderer\n");
        return false;
    }

    // The EDRAM stand-in. Sized to the guest's own stated front-buffer dimensions,
    // which VdSwap carries in every swap packet; 1280x720 until the first one arrives.
    // The EDRAM stand-in is TALLER than the presented frame, and that is the point.
    // `targetWidth/targetHeight` is what VdSwap says the FRONT BUFFER is; the EDRAM has
    // to hold the largest surface any PASS renders into, and this title's shadow pass
    // renders a 1024x1024 cascade. At 720 rows every cascade lost its bottom 304 rows
    // before anything could sample them. `CZ_VK_SMALL_EDRAM=1` restores the old size and
    // is the same-binary control arm.
    static const bool smallEdram = EnvOn("CZ_VK_SMALL_EDRAM");
    const uint32_t edramH =
        smallEdram ? R->targetHeight : std::max(R->targetHeight, kEdramHeight);
    // Kept on the Renderer because the WINDOW-COORDINATE draw path needs it: a window
    // coordinate is relative to the EDRAM surface, not to the presented frame, so the
    // clip volume it maps into has to be the EDRAM's (see the vte==0 branch).
    R->edramWidth = R->targetWidth;
    R->edramHeight = edramH;
    // THE HOST EXTENT IS SCALED; `edramWidth`/`edramHeight` below stay in guest pixels.
    if (!CreateImage(R->color, RSX(R->targetWidth), RS(edramH),
                     VK_FORMAT_R8G8B8A8_UNORM,
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                     VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_VIEW_TYPE_2D, 1, 1,
                     VkComponentMapping{}, 1, false, R->msaaSamples) ||
        // TRANSFER_SRC because 18.4% of this title's resolves copy out of the DEPTH
        // buffer rather than the colour one (its shadow cascades and the scene depth
        // its depth-of-field pass reads back) — see DoResolve.
        // SAMPLED only when the device came up with ray tracing, which is what route
        // (b)'s factor pass needs to read the scene depth. Conditional rather than
        // unconditional because a usage bit can change a driver's depth compression
        // decision, and `CZ_VK_RT=0` — the master arm since part 64 — must stay a
        // bit-for-bit description of the pre-RT renderer.
        !CreateImage(R->depth, RSX(R->targetWidth), RS(edramH),
                     EdramDepthFormat(),
                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                         (R->rtEnabled ? VK_IMAGE_USAGE_SAMPLED_BIT : 0u),
                     VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                     VK_IMAGE_VIEW_TYPE_2D, 1, 1, VkComponentMapping{}, 1, false,
                     R->msaaSamples) ||
        !CreateEdramResolveTargets(RSX(R->targetWidth), RS(edramH)))
    {
        fprintf(stderr, "[vk] render target creation FAILED\n");
        return false;
    }
    NameImage(R->color, "EDRAM colour %ux%u", RSX(R->targetWidth), RS(edramH));
    NameImage(R->depth, "EDRAM depth %ux%u", RSX(R->targetWidth), RS(edramH));

    // A depth snapshot is sampled through the same bindless heap and the same single
    // linear sampler as every other texture, so the device has to be able to filter
    // that format. Checked rather than assumed: if it cannot, the picture would be
    // undefined rather than wrong, and silently — say so once, loudly.
    {
        VkFormatProperties fp{};
        vkGetPhysicalDeviceFormatProperties(R->physical, R->depth.format, &fp);
        if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
            fprintf(stderr, "[vk] WARNING: %u is not sampleable on this device — "
                            "depth resolves will not be readable by the guest\n",
                    unsigned(R->depth.format));
        else if (!(fp.optimalTilingFeatures &
                   VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
            fprintf(stderr, "[vk] NOTE: depth format %u has no linear filtering; "
                            "depth snapshots are sampled with the linear sampler\n",
                    unsigned(R->depth.format));
    }

    // 128 MB of per-frame arena. The frontend's streams are small; gameplay is the
    // question, and the high-water mark is printed with the stats so the number can be
    // raised on evidence rather than guessed at again.
    //
    // It GROWS now (see BeginFrame), so this is the STARTING size rather than the limit.
    // 128 is kept as the start deliberately: it is the size every measurement in this
    // port up to part 18 was taken at, so `CZ_VK_NO_ARENA_GROWTH=1` reproduces the old
    // renderer exactly and remains a usable control arm. CZ_VK_ARENA_MB=N sets the
    // start, which is how the 128-vs-512 A/B that identified the black frames was run.
    static const uint64_t arenaMb =
        Env("CZ_VK_ARENA_MB") ? strtoull(Env("CZ_VK_ARENA_MB"), nullptr, 10) : 128;
    // The CROSS-FRAME stream store, same usage and memory type as the arena because the
    // GPU cannot tell them apart — the only difference is that this one is not reset at
    // the swap. `PersistMaintenance` doubles it when a frame overruns it.
    // CZ_VK_PERSIST_MB=N sets the start.
    //
    // **THE DEFAULT IS 1024 — THE CEILING — AS OF PART 79, AND IT WAS 128 FOR TWENTY-TWO
    // PARTS.** A growth is not a background cost: it is `WaitAllFramesIdle` +
    // `vkDeviceWaitIdle` + a host-visible allocation and MAP + freeing the old buffer, all
    // on the pump inside ONE frame. The operator's part-79 session grew twice, 128 -> 256
    // -> 512, and **both growths were the worst frame of their own ten-second window and
    // both were the only things they felt in 96.8 seconds of play** (§6dy §3, §6dz).
    //
    // **THE FIRST FIX WAS 512 AND IT WAS ONLY HALF RIGHT — RECORDED HERE BECAUSE THE ERROR
    // IS THE USEFUL PART.** The cost scales with the NEW buffer's size and the store
    // DOUBLES, so raising the start does not remove the class: it skips the cheap early
    // growths and leaves the expensive late one. At 512 the operator's next session grew
    // once, 512 -> 1024, and that single growth was **329.2 ms in one frame** (waits 8.9 +
    // allocate/map 254.9 + free-old 65.3) — they felt it, and correctly reported that the
    // after-load hitch was gone and one late stutter had appeared. Two hitches at 87 and
    // 158 ms became one at 352.
    //
    // **1024 IS `kPersistCeiling`, so the store can never grow AT ALL.** That is a
    // structural guarantee rather than a bigger guess, and the cost is nil because of a
    // measured 25x asymmetry: the same allocation costs **~10 ms at boot and 255 ms
    // mid-run**, since mid-run it must find a gigabyte while the old buffer is still live
    // and the machine is under load. Boot frames measured across the three starts —
    // 128: 237.7 / 234.2, 512: 244.0 / 247.3 / 242.7, **1024: 226.9** — i.e. the 1 GB
    // allocation is inside the spread of a boot frame that is 230-250 ms whatever we do.
    //
    // Hitting the ceiling is NOT a growth: `PersistMaintenance`'s else-branch drops and
    // refills the cache, which costs a `WaitAllFramesIdle` and nothing else.
    //
    // Note the waits are the SMALLEST of the three terms in a growth. Fencing the old
    // buffer away — the obvious, principled repair — would have bought 19% of it.
    static const uint64_t persistMb =
        Env("CZ_VK_PERSIST_MB") ? strtoull(Env("CZ_VK_PERSIST_MB"), nullptr, 10) : 1024;
    R->persistOn = !EnvOn("CZ_VK_NO_PERSIST_STREAMS");

    // Announce itself, because an arm nobody can see in the log is an arm that cannot be
    // shown to have engaged (gotcha 151).
    // The bound, and its two overrides. Announced whenever it is not the default, so an
    // arm that did not engage cannot be mistaken for one that did (gotcha 151).
    if (const char* gb = Env("CZ_VK_STREAM_GUARD_BYTES"))
    {
        const unsigned long long v = strtoull(gb, nullptr, 10);
        // Below kGuardBlocks the block arithmetic degenerates; refuse rather than
        // silently sampling nothing.
        if (v >= kGuardBlocks)
        {
            g_guardBytes = size_t(v);
            fprintf(stderr, "[vk] CZ_VK_STREAM_GUARD_BYTES=%llu — the store's guard is "
                            "EXACT up to %llu bytes and samples above it (default %zu)\n",
                    v, v, kGuardBytesDefault);
        }
        else
        {
            fprintf(stderr, "[vk] CZ_VK_STREAM_GUARD_BYTES=%llu REFUSED — must be >= %zu; "
                            "keeping %zu\n", v, kGuardBlocks, g_guardBytes);
        }
    }
    else
    {
        fprintf(stderr, "[vk] stream guard exact to %zu bytes, sampled above "
                        "(CZ_VK_STREAM_GUARD_BYTES=N to change, item 00c)\n",
                g_guardBytes);
    }

    g_guardExact = EnvOn("CZ_VK_STREAM_GUARD_EXACT");
    if (g_guardExact)
        fprintf(stderr, "[vk] CZ_VK_STREAM_GUARD_EXACT=1 — the cross-frame store's guard "
                        "hashes EVERY byte, so it cannot miss a small edit in a large "
                        "stream. Diagnostic for open item 00c.\n");

    // --- CZ_VK_FRAMES_IN_FLIGHT ---------------------------------------------------------
    //
    // How many frames the CPU may be ahead of the GPU. 1 is the renderer this port ran
    // for twenty-two parts: submit the frame, block on its fence, read it back, present.
    //
    // Why more than 1 is the largest item in the performance plan: a crowd frame is
    // ~27.7 ms of CPU followed by ~16.5 ms of GPU, strictly in series, so the card is
    // idle 68% of every frame and the driver correctly governs it down to a mid clock
    // (gotcha 231, §6ar). `CZ_VK_NO_SUBMIT=1` measured the ceiling on removing that
    // serialisation — CPU-only time, ~1.45x — without building it. This is building it.
    //
    // TWO IS THE DEFAULT AND THREE IS AVAILABLE, and neither is a guess about which
    // wins: one frame of overlap already covers a GPU shorter than the CPU, so 3 should
    // read as noise, and if it does not then the model of where the time goes is wrong
    // and that is worth knowing. Both are one binary, which is what makes the A/B legal.
    static const char* fifEnv = Env("CZ_VK_FRAMES_IN_FLIGHT");
    R->framesInFlight = fifEnv ? uint32_t(strtoul(fifEnv, nullptr, 10)) : 2;
    if (R->framesInFlight < 1)
        R->framesInFlight = 1;
    if (R->framesInFlight > kMaxFramesInFlight)
        R->framesInFlight = kMaxFramesInFlight;

    // The instruments that CANNOT be one frame late, and are therefore allowed to veto
    // the pipelining rather than silently produce misaligned evidence.
    //
    // A deferred present hands the window frame N-1's pixels while the renderer's
    // snapshot images, its resolve chain and its register state are all frame N's. The
    // frame-stats and PPM paths are fixed properly below — they carry the presented
    // frame's own metadata in its slot — but three instruments read the LIVE resolve
    // chain next to the presented pixels and cannot be fixed that way:
    // CZ_VK_SNAP_ON_BLACK and CZ_VK_SNAP_ON_DARK trigger a dump of the current
    // snapshots from a pixel test on the presented frame, and CZ_VK_FRAME_STATS_SURFACE
    // reads back a named snapshot to sit in the same stats line as the presented one.
    // A one-frame skew there is a diagnostic that quietly answers about the wrong frame,
    // which is worse than a slower diagnostic (gotcha 7's cousin: an instrument that
    // reports about something other than what it names).
    if (Env("CZ_VK_SNAP_ON_BLACK") || Env("CZ_VK_SNAP_ON_DARK") ||
        Env("CZ_VK_FRAME_STATS_SURFACE") || Env("CZ_VK_SNAP_DUMP") ||
        EnvOn("CZ_VK_NO_SUBMIT"))
    {
        if (R->framesInFlight != 1)
            fprintf(stderr, "[vk] frames-in-flight forced to 1: a snapshot-chain or "
                            "no-submit instrument is on and those read the resolve state "
                            "of the frame being RECORDED, not the one being presented\n");
        R->framesInFlight = 1;
    }
    fprintf(stderr, "[vk] frames in flight: %u%s\n", R->framesInFlight,
            R->framesInFlight == 1 ? " (submit and wait; the pre-part-23 renderer)" : "");

    // B1's arms, read BEFORE the buffers exist because the control arm must not allocate
    // the sub-arena at all — an arm that still pays for its subject's memory is not a
    // control for its memory.
    // OFF BY DEFAULT, BECAUSE IT WAS MEASURED AND IT IS A NULL. B1 does exactly what it
    // was built to do — 100% of draws served pre-zeroed, zero fallbacks, zero drain, and
    // `perf` says 0.43 ms of `__memset_avx2` left the pump — and the pump's CPU per frame
    // did not move by a hundredth of a millisecond (three runs an arm, six matched draw
    // bands, §10.2). The cost is store BANDWIDTH, which is machine-wide, so issuing the
    // same stores from another core buys nothing.
    //
    // KEPT RATHER THAN DELETED, and for a reason with a date on it: this box is an 8-core
    // 4654 MHz desktop whose memory pipe is the bound. A machine with a slower core
    // relative to its memory — the Ryzen 3 stand-in of part 107, or a Steam Deck — may
    // put the same 0.43 ms back on the CPU side of the ledger, where relocating it would
    // pay. `CZ_VK_PREZERO=1` is how that gets asked, and it costs one run.
    g_prezeroOff = !EnvOn("CZ_VK_PREZERO");
    g_prezeroPoison = EnvOn("CZ_VK_PREZERO_POISON");
    if (!g_prezeroOff)
        fprintf(stderr,
                "[vk] CZ_VK_PREZERO=1 — the shared-block pre-zero is ON (part 111 B1). "
                "MEASURED A NULL on this box: the memset leaves the pump and the frame "
                "does not move. An arm, not a default.\n");
    if (g_prezeroPoison)
        fprintf(stderr, "[vk] CZ_VK_PREZERO_POISON=1 — the pre-zero writes 0xAA instead "
                        "of 0. THE PICTURE MUST BREAK; if it does not, the fast path "
                        "never engaged and every number from it is meaningless\n");
    if (!CreateBuffer(R->arena, (arenaMb << 20) * R->framesInFlight,
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      /*deviceAddress=*/true, "per-frame arena") ||
        (g_texNoBatch = EnvOn("CZ_VK_NO_TEX_BATCH"), false) ||
        // B1's shared sub-arena (part 111 §4). Sized in SLOTS, not megabytes: the number
        // that matters is draws per frame, and 24,576 slots a region is well past the
        // ~9,700 the operator's crowd reaches, so the overflow path is a safety net
        // rather than a thing the measurement runs through. 56.6 MB total at 2 frames in
        // flight, against an arena that is already hundreds.
        (g_prezeroOff ? false
                      : !CreateBuffer(R->sharedArena,
                                      VkDeviceSize(kSharedStride) * 24576 *
                                          R->framesInFlight,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      /*deviceAddress=*/true, "shared sub-arena")) ||
        // THE STAGING ARENA IS `kTexUploadSlots` SEGMENTS AS OF PART 79, and it grew from
        // a flat 64 MB to 3 x 32 MB so that partitioning it did not narrow the per-upload
        // ceiling below anything real. The measured largest single upload on the
        // autonomous route is 1.33 MB; see the upload-ring comment for the whole argument.
        !CreateBuffer(R->staging, kTexUploadSlots * kTexSlotBytes,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      false) ||
        !CreateBuffer(R->readback,
                      // Big enough for the presented frame AND for the largest snapshot
                      // CZ_VK_SNAP_DUMP might read back — this title's shadow cascade
                      // is 4096x1024, and the dump SKIPS anything that does not fit,
                      // which would have made the one surface under investigation the
                      // one surface absent from the directory.
                      std::max(uint64_t(RSX(R->targetWidth)) * RS(R->targetHeight),
                               uint64_t(RSX(4096)) * RS(1024)) * 4,
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, ReadbackMemoryProps(), false))
    {
        fprintf(stderr, "[vk] buffer allocation FAILED\n");
        return false;
    }

    // THE TEXTURE UPLOAD RING (part 79 item 1). One command buffer, one fence and one
    // segment of the staging arena per slot; see the ring's comment for the design. It is
    // created HERE, after `R->staging`, because the segment bases are offsets into it.
    //
    // `CZ_VK_TEX_FLUSH_WAIT=1` still allocates the ring — the arm has to be one binary and
    // the slots cost three command buffers and three fences — but never advances it, so
    // its uploads all live in segment 0 and every flush submits and waits exactly as
    // part 77 did.
    g_texFlushWait = EnvOn("CZ_VK_TEX_FLUSH_WAIT");
    {
        VkCommandBufferAllocateInfo tcbi{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        tcbi.commandPool = R->cmdPool;
        tcbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        tcbi.commandBufferCount = 1;
        VkFenceCreateInfo tfi{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        for (uint32_t i = 0; i < kTexUploadSlots; ++i)
        {
            VK_CHECK(vkAllocateCommandBuffers(R->device, &tcbi, &g_texSlots[i].cb),
                     "vkAllocateCommandBuffers (texture upload slot)");
            VK_CHECK(vkCreateFence(R->device, &tfi, nullptr, &g_texSlots[i].fence),
                     "vkCreateFence (texture upload slot)");
            g_texSlots[i].base = VkDeviceSize(i) * kTexSlotBytes;
            NameObject(uint64_t(g_texSlots[i].cb), VK_OBJECT_TYPE_COMMAND_BUFFER,
                       "texture upload slot %u", i);
        }
        g_texSlot = 0;
        g_stagingCursor = 0;
        fprintf(stderr,
                "[vk] texture upload ring: %u slots x %llu MB staging, flush %s "
                "(CZ_VK_TEX_FLUSH_WAIT=1 is the control arm)\n",
                kTexUploadSlots, (unsigned long long)(kTexSlotBytes >> 20),
                g_texFlushWait ? "SUBMITS AND WAITS (part 77 behaviour)"
                               : "submits and does NOT wait");
    }

    // One present readback buffer per slot. It is deliberately NOT a region of
    // `R->readback`: that buffer is also the snapshot-dump target, and sharing it would
    // mean an instrument's readback could land on top of a frame the window has not
    // fetched yet — a corrupted picture that appears only when a diagnostic is on, which
    // is the worst kind. Sized like `R->readback` so a front-buffer resolve larger than
    // the frame extent still fits rather than being silently truncated.
    for (uint32_t i = 0; i < R->framesInFlight; ++i)
    {
        if (!CreateBuffer(R->frames[i].present,
                          std::max(uint64_t(RSX(R->targetWidth)) * RS(R->targetHeight),
                                   uint64_t(RSX(4096)) * RS(1024)) * 4,
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT, ReadbackMemoryProps(), false))
        {
            fprintf(stderr, "[vk] present readback buffer %u allocation FAILED\n", i);
            return false;
        }
    }

    // The cross-frame store is allocated SEPARATELY from the chain above, and its failure
    // DEGRADES rather than kills the renderer. It is an optimisation: without it every
    // stream takes the per-frame path, which is what this port did for twenty-one parts
    // and which still renders the game correctly. Refusing to start at all because a
    // machine could not spare another 128 MB would trade a 30% frame-time saving for the
    // whole picture, which is not a trade anything here should make silently.
    if (R->persistOn &&
        !CreateBuffer(R->persist, persistMb << 20, PersistUsage(),
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      /*deviceAddress=*/true, "cross-frame stream store"))
    {
        // FALL BACK DOWN A LADDER BEFORE GIVING UP. Raising the default from 128 MB to a
        // gigabyte must not turn "this machine has less RAM than mine" into "this machine
        // loses the store entirely", which is a ~30% frame-time regression and would be
        // invisible from a 48 GB box — the failure path of a bigger request is not the
        // same path, and it is the one nobody tests (gotcha 469). A smaller store still
        // works; it just grows, and a growth is strictly better than no store at all.
        static const uint64_t kPersistLadder[] = { 512, 128 };
        bool made = false;
        for (uint64_t mb : kPersistLadder)
        {
            if (mb >= persistMb)
                continue;   // never "fall back" upward
            fprintf(stderr,
                    "[vk] the %llu MB cross-frame stream store could not be allocated — "
                    "retrying at %llu MB. It will GROW from there, and a growth is a whole "
                    "frame of pump time (72 ms for a 256 MB step, 329 ms for a 1 GB one); "
                    "CZ_VK_PERSIST_MB=N sets the start\n",
                    (unsigned long long)persistMb, (unsigned long long)mb);
            if (CreateBuffer(R->persist, mb << 20, PersistUsage(),
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             /*deviceAddress=*/true, "cross-frame stream store (fallback)"))
            {
                made = true;
                break;
            }
        }
        if (!made)
        {
            fprintf(stderr, "[vk] the cross-frame stream store could not be allocated at "
                            "all — running without it, which is slower and correct\n");
            R->persistOn = false;
        }
    }
    CreateStoreMirror();

    // Two samplers, and one global choice per draw is a stated simplification: the
    // fetch constant carries per-texture filter and address modes that this does not
    // yet honour. Named here so the next reader knows it is a gap with a location
    // rather than a mystery in the picture.
    VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.maxLod = VK_LOD_CLAMP_NONE;
    // Sampler 0 is PLAIN TRILINEAR, deliberately and permanently. Part 41's first
    // attempt put 16x aniso here, reasoning "every fetch publishes index 0, so this
    // is where aniso goes" — and the very first capture showed dark speckle across
    // the whole frame, because index 0 also serves the SHADOW ATLAS lookups, which
    // hardware fetches with aniso=0 and point filters. The per-fetch sampler cache
    // (SamplerIndexForFetch) is where the fetch constant's own filter fields are
    // honoured; this sampler remains the fallback and the CZ_VK_NO_FETCH_SAMPLERS
    // arm's whole world.
    fprintf(stderr, "[vk] per-fetch samplers %s (aniso device limit %.0fx)\n",
            Env("CZ_VK_NO_FETCH_SAMPLERS") ? "OFF (CZ_VK_NO_FETCH_SAMPLERS)" : "ON",
            R->anisoLimit);
    if (vkCreateSampler(R->device, &si, nullptr, &R->linearSampler) != VK_SUCCESS)
        return false;
    si.magFilter = VK_FILTER_NEAREST;
    si.minFilter = VK_FILTER_NEAREST;
    if (vkCreateSampler(R->device, &si, nullptr, &R->pointSampler) != VK_SUCCESS)
        return false;

    // The draw-ID fragment module, embedded rather than loaded (see drawid_ps.hlsl). It
    // is created unconditionally and costs a few hundred bytes: an instrument that has
    // to be enabled at BUILD time is one nobody has when they need it.
    {
        VkShaderModuleCreateInfo smi{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smi.codeSize = sizeof kDrawIdPixelShaderSpv;
        smi.pCode = kDrawIdPixelShaderSpv;
        if (vkCreateShaderModule(R->device, &smi, nullptr, &R->drawIdModule) != VK_SUCCESS)
        {
            // Not fatal: the renderer works without the instrument, and saying so is
            // better than refusing to start because a diagnostic failed to compile.
            R->drawIdModule = VK_NULL_HANDLE;
            fprintf(stderr, "[vk] the draw-ID shader module failed to create — "
                            "CZ_VK_DRAW_ID will not work this run\n");
        }
        // CZ_VK_NULL_PS=1 (part 106): every translated pixel shader replaced by the
        // do-nothing fragment stage — the "everything but pixel shading" arm of the GPU
        // decomposition (tools/null_ps.hlsl). Created only when asked, and the line
        // below is the engagement evidence; the picture is garbage by design.
        if (EnvOn("CZ_VK_NULL_PS"))
        {
            smi.codeSize = sizeof kNullPixelShaderSpv;
            smi.pCode = kNullPixelShaderSpv;
            if (vkCreateShaderModule(R->device, &smi, nullptr, &R->nullPsModule) != VK_SUCCESS)
            {
                R->nullPsModule = VK_NULL_HANDLE;
                fprintf(stderr, "[vk] CZ_VK_NULL_PS: the null fragment module failed to "
                                "create — the arm is NOT engaged\n");
            }
            else
                fprintf(stderr, "[vk] CZ_VK_NULL_PS=1 — EVERY pixel shader is the "
                                "do-nothing fragment stage this run (a GPU measurement "
                                "arm; the picture is garbage by design)\n");
        }
    }

    // The dummies. Slot 0 of every heap is a defined 1x1 white texel, so a shader that
    // samples a slot the runtime could not fill reads white rather than an unbound
    // descriptor — undefined behaviour even when the result is discarded.
    auto makeDummy = [&](Image& img, VkImageViewType type, uint32_t layers,
                         uint32_t depth, uint32_t setIndex) {
        if (!CreateImage(img, 1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT, type, layers, depth))
            return false;
        NameImage(img, "dummy set %u (%u layers)", setIndex, layers);
        // CZ_VK_CUBE_POISON=1 — make the CUBE dummy opaque MAGENTA instead of white.
        //
        // THE POSITIVE CONTROL, and this instrument exists because the change it tests
        // came back invisible. Part 25's picture A/B — cube maps bound versus
        // CZ_VK_NO_CUBE — was PIXEL-IDENTICAL on all 44 frames where both arms had the
        // same camera and the same draw set, and a null like that has two readings that
        // no amount of looking can separate: the cube samples do not reach the output at
        // all, or they reach it and happen to be indistinguishable from white.
        //
        // Poisoning the DUMMY answers it, because the dummy is what the old renderer's
        // cube fetches read. If a poisoned run is still identical, the cube sample is
        // discarded somewhere downstream and the whole item is mis-scoped; if the frame
        // fills with magenta, the path is live and the null above is a statement about
        // the CONTENT of those cube maps. Same shape as CZ_VK_TEX_GUARD_POISON
        // (gotcha 30) — a comparison that has never reported a positive proves nothing
        // by reporting a negative.
        // CZ_VK_DUMMY_POISON=1 poisons ALL FOUR dummies, not just the cube one.
        //
        // CZ_VK_CUBE_POISON answers "does a cube fetch reach the picture", and part 26
        // used it to prove the cube dummy tints the crowd. It cannot answer the question
        // the ground patches pose, because the ground reads no cube. The 1x1 dummy in
        // EVERY heap is white, and a white texel times a diffuse term of 0.706 is exactly
        // the (180,180,180) those patches are — 49,195 pixels of one value with a standard
        // deviation of 3, against a guest clear colour of BLACK, so they are written by
        // something rather than left unwritten. The declared-fetch census cannot see this
        // case: a shader reading a descriptor index the runtime never wrote gets slot 0
        // silently, because the shared-constant block is memset to zero every draw.
        static const bool cubePoison = EnvOn("CZ_VK_CUBE_POISON");
        static const bool allPoison = EnvOn("CZ_VK_DUMMY_POISON");
        const bool poisoned =
            allPoison || (cubePoison && type == VK_IMAGE_VIEW_TYPE_CUBE);
        const uint32_t white = poisoned ? 0xFFFF00FFu : 0xFFFFFFFFu;
        if (poisoned)
            fprintf(stderr,
                    "[vk] %s: the set-%u dummy is MAGENTA (0xFFFF00FF), %u layer(s)\n",
                    allPoison ? "CZ_VK_DUMMY_POISON" : "CZ_VK_CUBE_POISON", setIndex,
                    layers);
        // ALL SIX FACES, not just the first. The copy below has always had
        // `layerCount = layers`, but only four bytes were ever written into the staging
        // buffer — so faces 1..5 of every 1x1 dummy were filled from whatever the staging
        // buffer last held. It was invisible while the only multi-layer image was a dummy
        // nobody could see, and it would have made a poisoned run report a nonsense
        // colour on five faces out of six.
        for (uint32_t f = 0; f < layers; f++)
            memcpy(R->staging.mapped + f * 4, &white, 4);
        RunImmediate([&](VkCommandBuffer cb) {
            Barrier(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT);
            VkBufferImageCopy copy{};
            copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers };
            copy.imageExtent = { 1, 1, depth };
            vkCmdCopyBufferToImage(cb, R->staging.buffer, img.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            Barrier(cb, img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_IMAGE_ASPECT_COLOR_BIT);
        });
        VkDescriptorImageInfo ii{};
        ii.imageView = img.view;
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = R->sets[setIndex];
        w.dstBinding = 0;
        w.dstArrayElement = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        w.pImageInfo = &ii;
        vkUpdateDescriptorSets(R->device, 1, &w, 0, nullptr);
        return true;
    };
    if (!makeDummy(R->dummy2D, VK_IMAGE_VIEW_TYPE_2D, 1, 1, 0) ||
        !makeDummy(R->dummy3D, VK_IMAGE_VIEW_TYPE_3D, 1, 1, 1) ||
        !makeDummy(R->dummyCube, VK_IMAGE_VIEW_TYPE_CUBE, 6, 1, 2) ||
        !makeDummy(R->dummy1D, VK_IMAGE_VIEW_TYPE_1D, 1, 1, 4))
    {
        fprintf(stderr, "[vk] dummy texture creation FAILED\n");
        return false;
    }

    {
        VkDescriptorImageInfo si2{};
        si2.sampler = R->linearSampler;
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = R->sets[3];
        w.dstBinding = 0;
        w.dstArrayElement = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        w.pImageInfo = &si2;
        vkUpdateDescriptorSets(R->device, 1, &w, 0, nullptr);
    }

    if (!LoadShaders())
        return false;

    // PRE-WARM, here and not later: after the shaders exist (the keys name them by hash)
    // and before the guest has drawn anything, so every pipeline this session needs is
    // already built when the first frame asks for one. See PrewarmPipelines.
    PrewarmPipelines();
    // Only now may GetPipeline go asynchronous (part 98): the boot warm above stays
    // synchronous — at load, where a player expects to wait — and everything after it
    // is mid-play, where a 1-200 ms create on the frame thread is the reported stutter.
    pipelinejit::bootDone = true;

    R->presentPixels.resize(size_t(RSX(R->targetWidth)) * RS(R->targetHeight) * 4);
    g_texCensus = EnvOn("CZ_VK_TEX_CENSUS");
    g_dimCensus = EnvOn("CZ_VK_DIM_CENSUS");
    // The three readers of the per-pass snapshot-input list, asked once. See the
    // declaration; without this the list is maintained by a linear scan on every
    // snapshot fetch for a diagnostic that is off on essentially every run.
    // CZ_CAPTURE_KEY is in the list because it arms the draw census by another door —
    // an F9 press under it writes `capture.census` without CZ_VK_DRAW_CENSUS being set,
    // and a gate that misses one of its own entrances would silently drop the `(snap)`
    // column from exactly the captures an operator takes.
    g_passInputsWanted = Env("CZ_VK_PSBIND") || Env("CZ_VK_DRAW_CENSUS") ||
                         Env("CZ_VK_RESOLVE_TRACE") || Env("CZ_CAPTURE_KEY");
    g_copyCensusOn = EnvOn("CZ_VK_COPY_CENSUS");
    if (g_copyCensusOn)
        fprintf(stderr, "[vk] CZ_VK_COPY_CENSUS=1 — resolve-copy produced-vs-sampled "
                        "census ARMED (part 90 item 2; a diagnostic arm)\n");
    if (const char* n = Env("CZ_VK_DIM_DISAGREE"))
    {
        g_dimDisagree = true;
        g_dimDisagreeLeft = atoi(n);       // how many get printed as they happen
    }
    // GUARD + REVALIDATE ARE THE DEFAULT AS OF PART 38. The cache previously uploaded a
    // texture ONCE per (address, extent, format) and served it forever, which is wrong
    // the moment streaming recycles an address — and an operator session recycles them
    // constantly: the tanker wore a BRICK WALL, and "almost everything up close wears a
    // random texture" (the operator's words) the longer the session ran. Part 35's
    // "4 stale of 92M hits" that justified leaving the repair off was measured on a
    // 400 s headless run at one location — a fact about that route, not about play
    // (the gotcha-50 family). A full operator session on the repair: every prop
    // correct, no reported slowdown. CZ_VK_NO_TEX_REVALIDATE=1 is the same-binary
    // control arm that brings the random-texture defect back.
    {
        static const bool noRevalidate = EnvOn("CZ_VK_NO_TEX_REVALIDATE");
        g_texGuard = !noRevalidate || EnvOn("CZ_VK_TEX_GUARD");
        g_texRevalidate = !noRevalidate;
    }
    g_noGolden = EnvOn("CZ_VK_NO_GOLDEN_TEX");
    GoldenLoad();   // preload any texture bytes a prior session captured to disk
    // The pre-part-47 cadence — revalidate on every fetch instead of once a frame per
    // cache entry. A control arm, not a fix; see its declaration.
    g_texGuardEveryFetch = EnvOn("CZ_VK_TEX_GUARD_EVERY_FETCH");
    // The flat open-addressed cache and its two arms. See the FlatCache comment: the
    // failure mode this is guarding against is a lookup that returns the WRONG entry,
    // which draws a wrong mesh and reports nothing, so the control arm restores the
    // `std::unordered_map` exactly and the verify arm runs both and compares.
    g_constMemoOff = EnvOn("CZ_VK_NO_CONST_MEMO");
    g_fetchMemoCensus = EnvOn("CZ_VK_FETCH_MEMO_CENSUS");
    g_resolveSplitCensus = EnvOn("CZ_VK_RESOLVE_SPLIT_CENSUS");
    // PARALLEL RECORD (part 89). ON BY DEFAULT since its 3v3 (dominant crowd band
    // 13.00 -> 11.20 ms, −13.9% / −1.80 ms; §6ej) — CZ_VK_NO_PAR_RECORD=1 is the
    // same-binary control arm, per part 87 §3's rule. It refuses two combinations
    // out loud rather than half-engaging: no workers means the pump would capture
    // and replay everything itself for pure overhead, and CZ_VK_NO_DRIVER_RECORD's
    // measurement would be silently distorted by a path whose whole point is moving
    // those calls. (CZ_VK_PAR_RECORD=1 is accepted and redundant, kept so the 3v3's
    // arm spelling still means what it meant.)
    if (!EnvOn("CZ_VK_NO_PAR_RECORD"))
    {
        if (NoDriverRecord())
            fprintf(stderr, "[vk] parallel record OFF: CZ_VK_NO_DRIVER_RECORD is "
                            "set and the two instruments measure the same calls\n");
        else if (!GuardPoolWorkers())
            fprintf(stderr, "[vk] parallel record OFF: no worker pool "
                            "(CZ_WORKERS=0 or CZ_VK_NO_PARALLEL_GUARD) — the serial "
                            "path is the control arm, not a degraded mode\n");
        else if (GpuStatsOn())
            fprintf(stderr, "[vk] parallel record OFF: CZ_VK_GPU_STATS is set and a "
                            "pipeline-statistics query cannot span the worker chunks "
                            "— the census runs on the serial recorder (same draws, "
                            "same order; the wall time is the part-88 recorder's)\n");
        else
        {
            R->parRec = true;
            const char* cs = Env("CZ_VK_RECORD_CHUNK");
            if (cs && atoi(cs) > 0)
                R->parRecChunk = uint32_t(atoi(cs));
            R->capBuf.reserve(R->parRecChunk);
            fprintf(stderr,
                    "[vk] parallel command recording ON (default since part 89): "
                    "chunks of %u draws to %u shared guard-pool workers, resolve "
                    "serial, order gated. CZ_VK_NO_PAR_RECORD=1 is the control arm; "
                    "CZ_VK_RECORD_CHUNK=N tunes.\n",
                    R->parRecChunk, GuardPoolWorkers());
        }
    }
    else
        fprintf(stderr, "[vk] CZ_VK_NO_PAR_RECORD=1 — parallel record OFF (the "
                        "part-88 serial recorder, same binary)\n");
    g_bindRunCensus = EnvOn("CZ_VK_BIND_RUN_CENSUS");
    g_scissor1px = EnvOn("CZ_VK_SCISSOR_1PX");
    g_tri1 = EnvOn("CZ_VK_TRI1");
    if (g_tri1)
        fprintf(stderr, "[vk] CZ_VK_TRI1=1 — every draw issues at most its first primitive "
                        "this run (a GPU measurement arm; the picture is garbage by design)\n");
    if (g_scissor1px)
        fprintf(stderr, "[vk] CZ_VK_SCISSOR_1PX=1 — every draw's scissor is 1x1 this run "
                        "(a GPU measurement arm; the picture is garbage by design)\n");
    g_guardCensus = EnvOn("CZ_VK_GUARD_CENSUS");
    g_noBindBatch = EnvOn("CZ_VK_NO_BIND_BATCH");
    g_verifyBindBatch = EnvOn("CZ_VK_VERIFY_BIND_BATCH");
    g_verifyBindPoison = EnvOn("CZ_VK_VERIFY_BIND_BATCH_POISON");
    if (g_verifyBindPoison)
        g_verifyBindBatch = true;
    g_streamDedupCensus = EnvOn("CZ_VK_STREAM_DEDUP_CENSUS");
    g_pardrawCensus = EnvOn("CZ_VK_PARDRAW_CENSUS");
    if (g_pardrawCensus)
        fprintf(stderr,
                "[vk] CZ_VK_PARDRAW_CENSUS=1 — the per-draw MUTATION census is ON (part "
                "111 §3, item B's ask-first step). It counts the shared-state reads and "
                "writes on the per-draw path and TIMES the low-frequency mutations. A "
                "DIAGNOSTIC ARM: never quote a frame time from this run.\n");
    g_reuseCensus = EnvOn("CZ_VK_REUSE_CENSUS");
    if (g_reuseCensus)
    {
        reusecensus::prev.reserve(16384);
        reusecensus::cur.reserve(16384);
        g_reuseCopiedKeys.reserve(8192);
        fprintf(stderr,
                "[reuse] CZ_VK_REUSE_CENSUS=1 — the cross-frame draw-identity census is "
                "ON (CW lead 2's ask-first step). A DIAGNOSTIC ARM: it hashes ~0.5-1 KB "
                "of registers per draw on the pump thread. NEVER quote a frame time from "
                "this run.\n");
    }
    g_paletteCensus = EnvOn("CZ_VK_PALETTE_EXTENT_CENSUS");
    if (g_paletteCensus)
        fprintf(stderr,
                "[palcensus] CZ_VK_PALETTE_EXTENT_CENSUS=1 — the bone-palette bounded-"
                "gather census is ON (part 88 step 0). A DIAGNOSTIC ARM; it changes no "
                "copy and no frame time from this run is quotable.\n");
    g_constMemoVerify = EnvOn("CZ_VK_VERIFY_CONST_MEMO");
    g_constMemoVerifyPoison = EnvOn("CZ_VK_VERIFY_CONST_MEMO_POISON");
    if (g_constMemoVerifyPoison)
        g_constMemoVerify = true;
    g_flatCacheOff = EnvOn("CZ_VK_NO_FLAT_CACHE");
    g_flatCacheVerify = EnvOn("CZ_VK_VERIFY_FLAT_CACHE");
    g_flatCacheVerifyPoison = EnvOn("CZ_VK_VERIFY_FLAT_CACHE_POISON");
    if (g_flatCacheVerifyPoison)
        g_flatCacheVerify = true;   // the poison is meaningless without the check
    // PRE-SIZE, so a doubling never lands inside a frame the player is looking at. Each
    // figure is the measured high-water mark of a full outdoor run, not a guess: the
    // cross-frame store reached 91,750 live entries with the stream store at its 512 MB
    // ceiling, the texture cache 3,802, the per-frame stream cache ~2,200 first-touch
    // streams, and the shader table is 439 and fixed at load. Together ~27 MB of arrays
    // allocated once, against a renderer that already holds a 512 MB stream store — and
    // it takes the run's grow bill from 31.41 ms in 20 grows to zero.
    R->persistCache.Reserve(180000);
    R->textures.Reserve(8192);
    R->streamCache.Reserve(8192);
    R->shaders.Reserve(1024);
    g_texSources.Reserve(8192);
    g_texGuardAddrs.Reserve(8192);
    if (const char* n = Env("CZ_VK_TEX_GUARD_BYTES"))
    {
        // Clamped to a multiple of kGuardBlocks and to at least one block: below that
        // `bound / kGuardBlocks` is zero and the sampled path folds nothing at all,
        // which would read as a guard that never fires rather than as a bad setting.
        const size_t want = size_t(strtoul(n, nullptr, 0));
        g_texGuardBytes = std::max<size_t>(kGuardBlocks * 8, want & ~size_t(kGuardBlocks - 1));
        fprintf(stderr,
                "[vk] CZ_VK_TEX_GUARD_BYTES=%zu — the texture content guard is exact to "
                "that many bytes and samples 8 spread blocks above it (default 16384)\n",
                g_texGuardBytes);
    }
    if (g_texGuardEveryFetch)
        fprintf(stderr,
                "[vk] CZ_VK_TEX_GUARD_EVERY_FETCH — the texture content guard runs on "
                "EVERY fetch, not once a frame per entry (the pre-part-47 cadence)\n");
    g_texGuardPoison = EnvOn("CZ_VK_TEX_GUARD_POISON");
    if (g_texGuardPoison)
        fprintf(stderr, "[vk] texture guard POISONED — the changed share must now read "
                        "100.0%%; anything else means the census cannot fire\n");
    if (g_texRevalidate)
        fprintf(stderr, "[vk] texture cache REVALIDATES on content: a cache hit whose "
                        "guest bytes changed is re-uploaded in place\n");
    if (EnvOn("CZ_VK_TEX_REFRESH_ALL"))
        fprintf(stderr, "[vk] EVERY texture is re-read on EVERY fetch "
                        "(CZ_VK_TEX_REFRESH_ALL) — a picture arm, ruinously slow, and "
                        "the cache cannot serve a stale image under it\n");
    g_profileOn = EnvOn("CZ_VK_PROFILE");
    if (g_profileOn)
    {
        CalibrateProfNow();
        g_extraScopes = Env("CZ_VK_PROFILE_EXTRA_SCOPES")
                            ? uint32_t(atoi(Env("CZ_VK_PROFILE_EXTRA_SCOPES"))) : 0;
        if (g_extraScopes)
            fprintf(stderr,
                    "[vkprof] CZ_VK_PROFILE_EXTRA_SCOPES=%u — the positive control for "
                    "the instrument line; `other`'s residual must rise by about %.0f "
                    "ns/draw and no named phase may move\n",
                    g_extraScopes, double(g_extraScopes) * double(g_profNowNs10) / 10.0);
    }
    // Reported through the profile window, so it needs the profile on to say anything.
    // Saying so out loud rather than silently counting into a report nobody prints.
    g_streamCensus =
        Env("CZ_VK_STREAM_CENSUS") ? atoi(Env("CZ_VK_STREAM_CENSUS")) : 0;
    if (g_streamCensus && !g_profileOn)
    {
        fprintf(stderr, "[vk] CZ_VK_STREAM_CENSUS needs CZ_VK_PROFILE — it reports "
                        "through that window; census OFF\n");
        g_streamCensus = 0;
    }
    g_streamPoison = EnvOn("CZ_VK_STREAM_CENSUS_POISON");
    if (g_streamPoison)
        fprintf(stderr, "[vk] stream census POISONED — the content check must now read "
                        "0.0%%; anything else means it cannot fail\n");
    g_active = true;
    fprintf(stderr, "[vk] renderer UP: %ux%u target, %zu shaders\n", R->targetWidth,
            R->targetHeight, R->shadersMap.size());
    if (ResScale() != 1)
        fprintf(stderr,
                "[vk] internal resolution %ux%u (%ux the title's own 1280x720): the "
                "guest's geometry is unchanged and the rasterisation target is not. The "
                "present readback is %ux the bytes — %.1f MB/frame — so read `readback` "
                "in CZ_VK_PROFILE before quoting a frame time.\n",
                RSX(R->targetWidth), RS(R->targetHeight), ResScale(),
                ResScale() * ResScale(),
                double(RSX(R->targetWidth)) * RS(R->targetHeight) * 4.0 / 1048576.0);
    if (g_profileOn)
        fprintf(stderr, "[vkprof] frame CPU profile ON\n");
        // A.2's sampled whole-function timers ride with the profiler; the control arm
        // turns them off inside a profiled run so their own bill can be measured.
#if CZ_WHOLEFUNC
        g_wholeFunc = !EnvOn("CZ_VK_NO_WHOLEFUNC");
        fprintf(stderr,
                "[vkprof] WHOLE-FUNCTION timers COMPILED IN (-DCZ_WHOLEFUNC=1). THIS "
                "BUILD IS ~0.5 ms/frame SLOWER THAN A DEFAULT ONE AT THE CROWD, "
                "measured (part 110 §6.8) — read its SHARES, never its milliseconds, "
                "and never quote a frame time from it.\n");
#endif
    return true;
}

} // namespace

bool VkRenderer_Init()
{
    if (g_initTried)
        return g_active && !g_d3dMode;
    g_initTried = true;

    if (!EnvOn("CZ_VKDRAW"))
    {
        fprintf(stderr, "[vk] renderer OFF (set CZ_VKDRAW=1 to enable it)\n");
        return false;
    }
    // InitCommon names its own failure on every path.
    return InitCommon();
}

void VkRenderer_Draw(uint8_t* base, const Pm4Draw& draw)
{
    if (!g_active || g_d3dMode)
        return;
    // The renderer's own count of draws it was HANDED, next to the per-primitive
    // census of draws it accepted. The command processor's `ring: ... draws=` counter
    // and the renderer's prim counters disagreed by half and there was no number in
    // between to say where the difference lived — a chain has to be counted link by
    // link (gotcha 162), including the link between two modules.
    COUNT("draw: handed to the renderer");
    const uint32_t* regs = Pm4_Registers();
    // The resolve discriminator, and the only one: RB_MODECONTROL's edram_mode.
    if ((regs[0x2208] & 7) == 6)
    {
        DoResolve(base, regs);
        return;
    }
    DoDraw(base, draw, regs, Pm4_BoundShader(0), Pm4_BoundShader(1));
}

// Part 117: the same draw, from `cz-draw`, with the register file D replayed and the
// bindings the walk captured at the packet (gpu/pump_split.h).
void VkRenderer_DrawQueued(uint8_t* base, const Pm4Draw& draw, const uint32_t* regs,
                           const Pm4ShaderBinding& vs, const Pm4ShaderBinding& ps)
{
    if (!g_active || g_d3dMode)
        return;
    COUNT("draw: handed to the renderer");
    if ((regs[0x2208] & 7) == 6)
    {
        DoResolve(base, regs);
        return;
    }
    DoDraw(base, draw, regs, vs, ps);
}

namespace {

// The shared swap body — everything from "record the front buffer" to the frame
// stats line. The PM4 feed calls it from the XE_SWAP packet, the D3D feed from the
// Swap hook; the two callers gate on g_d3dMode so exactly one is live per run.
void DoSwapImpl(uint8_t* base, uint32_t frontBuffer, uint32_t width, uint32_t height)
{
    (void)base;
    // Recorded BEFORE the early returns: the resolve that produces the frame happens
    // before the swap that announces it, so on frame N the comparison in DoResolve is
    // made against the address frame N-1 published. That is fine because the address
    // does not change, and it is the reason the first frame has no snapshot rather
    // than the wrong one.
    R->frontBuffer = frontBuffer;
    ++R->frame;

    // Part 98: land any background-built pipelines even when no draw is currently
    // missing them, so their keys reach the periodic save and a returning material
    // hits the map instead of the pending set. One atomic read when there is nothing.
    if (pipelinejit::AsyncOn())
        pipelinejit::Drain();

    // ITEM 1.1: start the workers on the frame that is beginning, here and not in
    // `BeginFrame`, which does not run until the first draw. Everything between this
    // line and that first draw — the readback, the present, the frame-stats walk, the
    // packets before the first DRAW_INDX — is head start the pool gets for free, and it
    // is the difference between the early draws being served and hashing inline.
    GuardPoolDispatch();

    // CZ_FPS_LOG=N — the frame rate, every N seconds, and NOTHING ELSE.
    //
    // It exists because every instrument this project owns that reports a frame rate has
    // a bill big enough to change the answer: `CZ_VK_PROFILE` costs 2-4 ms a frame and
    // `CZ_VK_FRAME_STATS` walks all 921,600 pixels for another 1.9-3.3 (gotcha 337). So
    // "just play it and tell me how it feels" has always been the only uninstrumented
    // configuration, and it produces no number at all — which makes a session that
    // reports "it fares well" unfalsifiable, the one thing this project does not accept.
    //
    // This is one counter and one clock read per PRESENTED frame — ~20 ns against a
    // 13-20 ms frame, i.e. one part in a million, against the profiler's thousands of
    // clock reads. It is the cheapest thing here that can still be wrong, so it is off
    // by default like everything else.
    //
    // It reports the MEDIAN as well as the mean, because on this title a mean measures
    // the pacing floor rather than the change (gotcha 237) — and the interval's own
    // frame count, so a window that covers a load screen is visible as such rather than
    // averaged in.
    //
    // AND IT REPORTS THE DRAW COUNT, which the operator asked for and which is what makes
    // this line usable for a comparison at all. Their objection, in their words: AutoChuck
    // "isn't a predetermined route and zombie spawns are not always the same, so it won't
    // be 100% accurate especially if in a run it stays in the military zone and one go on
    // the main street." That is exactly right, and it applies to a human-driven session
    // too. Without the draw count a `[fps]` window is a frame rate with no statement of
    // what was being drawn, so two windows cannot be matched and the difference between
    // the arms and the difference between two PLACES are the same number.
    //
    // The min and max go out beside the median because a window that STRADDLES two places
    // is the case that has to be visible: a median of 3,000 built from 900 and 6,000 is
    // not a place at all, and a reader who only saw the median would match it against a
    // genuine 3,000-draw window in the other arm. It costs one counter read a frame.
    {
        static const int fpsLogSec = Env("CZ_FPS_LOG") ? atoi(Env("CZ_FPS_LOG")) : 0;

        // CZ_VK_FRAME_TRACE LIVES INSIDE THIS BLOCK AND MUST NOT DEPEND ON CZ_FPS_LOG.
        //
        // It used to. docs/instruments.md documents the trace as a standalone
        // instrument, and arming it alone produced an empty file, no rows, and not even
        // its own "CANNOT WRITE" diagnostic — because that message is inside the same
        // dead block. An operator played for an hour to capture a stutter and the file
        // was never opened.
        //
        // This is gotcha 418's exact shape, which the comment forty lines below names:
        // "the counter that was gated behind an expensive instrument and therefore never
        // on when the thing it measures happened". It was written about the pipeline
        // timer and applied here unnoticed.
        static const bool traceArmed = Env("CZ_VK_FRAME_TRACE") &&
                                       *Env("CZ_VK_FRAME_TRACE") != '\0';
        if (fpsLogSec > 0 || traceArmed)
        {
            using clk = std::chrono::steady_clock;
            static clk::time_point windowStart = clk::now();
            static clk::time_point lastFrame = windowStart;
            static uint64_t frames = 0;
            static std::vector<uint32_t> frameUs;
            static std::vector<uint32_t> frameDraws;
            const clk::time_point now = clk::now();
            ++frames;
            frameDraws.push_back(uint32_t(R->drawsThisFrame));
            frameUs.push_back(uint32_t(
                std::chrono::duration_cast<std::chrono::microseconds>(now - lastFrame)
                    .count()));
            lastFrame = now;
            // OPEN ITEM 0w's table. Differenced against the previous frame, so each row
            // says what happened INSIDE that frame rather than up to it.
            {
                static uint64_t prevTex = 0, prevPipes = 0, prevTexBytes = 0,
                                prevTexNs = 0, prevWalk = 0, prevSleep = 0,
                                prevFence = 0, prevTexDec = 0, prevTexForTrace = 0,
                                prevTexBytesForTrace = 0, prevPipesForTrace = 0,
                                prevTexNsForTrace = 0;
                // THE FRAME'S DECOMPOSITION. `walkNs` only accumulates when a walk
                // RETURNS and this present is happening INSIDE one, so the in-progress
                // portion is added explicitly — without it a 300 ms hitch is charged to
                // the frame AFTER the one that suffered it, which is the single case this
                // measurement exists for.
                const PumpStats ps = PumpStats_Read();
                const uint64_t nowNs = uint64_t(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        now.time_since_epoch()).count());
                const uint64_t inFlight =
                    ps.walkStartNs && nowNs > ps.walkStartNs ? nowNs - ps.walkStartNs : 0;
                const uint64_t walkNow = ps.walkNs + inFlight;
                const uint32_t walkUs = uint32_t((walkNow - prevWalk) / 1000);
                const uint32_t sleepUs = uint32_t((ps.sleepNs - prevSleep) / 1000);
                SlowFrameNote(R->frame, frameUs.back(), uint32_t(R->drawsThisFrame),
                              uint32_t(g_texRealUploads - prevTex),
                              uint32_t(g_pipeCount - prevPipes),
                              uint32_t((g_texUploadBytes - prevTexBytes) / 1024),
                              uint32_t((g_texUploadNs - prevTexNs) / 1000),
                              walkUs, sleepUs,
                              int32_t(frameUs.back()) - int32_t(walkUs) -
                                  int32_t(sleepUs),
                              uint32_t((g_fenceWaitNs - prevFence) / 1000),
                              uint32_t(g_gpuNsOfFrame / 1000),
                              uint32_t((g_texDecodeNs - prevTexDec) / 1000));
                prevTex = g_texRealUploads;
                prevPipes = g_pipeCount;
                prevTexBytes = g_texUploadBytes;
                prevTexNs = g_texUploadNs;
                prevWalk = walkNow;
                prevSleep = ps.sleepNs;
                const uint64_t fenceDelta = g_fenceWaitNs - prevFence;
                prevFence = g_fenceWaitNs;
                const uint64_t texDecDelta = g_texDecodeNs - prevTexDec;
                prevTexDec = g_texDecodeNs;
                // CZ_VK_FRAME_TRACE=<file> — one line per presented frame, so a stutter can
                // be found and read OFFLINE instead of hoping it lands in a twelve-row
                // table. One fprintf a frame; the columns are the decomposition above.
                static FILE* trace = [] () -> FILE* {
                    const char* f = Env("CZ_VK_FRAME_TRACE");
                    if (!f || !*f)
                        return nullptr;
                    FILE* h = fopen(f, "w");
                    // ANNOUNCE IT. An instrument that can be armed and silently record
                    // nothing costs whoever armed it the entire session before they find
                    // out — which is exactly what happened here.
                    fprintf(stderr, "[vk] CZ_VK_FRAME_TRACE: %s -> %s\n", f,
                            h ? "open, one row per presented frame" : "FAILED TO OPEN");
                    if (h)
                        fprintf(h, "frame draws wallUs walkUs recordUs fenceUs sleepUs "
                                   "residualUs gpuUs texUploads texKB texUpUs "
                                   "texDecUs pipes recPhUs strUs strGuardUs constUs "
                                   "texPhUs readbackUs recStateUs recVertUs recIdxUs "
                                   "drawOtherUs oKeyUs oPipeUs oFetchUs oShaderUs "
                                   "oBeginUs oTailUs cVsUs cPsUs cShrUs cVsCpUs "
                                   "cVsPaUs\n");
                    // THE PHASE COLUMNS ARE ONLY MEANINGFUL WITH CZ_VK_PROFILE SET —
                    // ProfScope records nothing without it and they will all read 0.
                    // Normally that would disqualify them (a probe costing the same order
                    // as the thing measured, gotcha 7), but the frames under investigation
                    // are 75-180 ms and the profiler is 2-4 ms: ~3% on a stutter frame,
                    // where on a 28 ms frame it would be a tenth. The bill is stated so
                    // nobody quotes a NORMAL frame's time from a run carrying this.
                    else
                        fprintf(stderr, "[vk] CZ_VK_FRAME_TRACE: CANNOT WRITE %s\n", f);
                    return h;
                }();
                // THE OPERATOR'S STUTTER MARKER (F7). Stamped into the trace AND the log,
                // because the log is what gets read first and a marker only in the data
                // file would be found only by someone already looking for it.
                if (Host_ConsumeMarkPressed())
                {
                    g_markCount++;
                    fprintf(stderr,
                            "[vk] ** MARK %llu — operator flagged a stutter at frame %llu "
                            "(%.1f ms: CPUrec %.1f, fence %.1f, sleep %.1f, GPU %.1f; "
                            "%u draws, %llu tex)\n",
                            (unsigned long long)g_markCount,
                            (unsigned long long)R->frame, double(frameUs.back()) / 1000.0,
                            double(int32_t(walkUs) - int32_t(fenceDelta / 1000)) / 1000.0,
                            double(fenceDelta) / 1e6, double(sleepUs) / 1000.0,
                            double(g_gpuNsOfFrame) / 1e6, uint32_t(R->drawsThisFrame),
                            (unsigned long long)(g_texRealUploads - prevTexForTrace));
                    if (trace)
                        fprintf(trace, "# MARK %llu frame %llu\n",
                                (unsigned long long)g_markCount,
                                (unsigned long long)R->frame);
                }
                if (trace)
                    fprintf(trace,
                            "%llu %u %u %u %d %llu %u %d %llu %llu %llu %llu %llu %llu",
                            (unsigned long long)R->frame, uint32_t(R->drawsThisFrame),
                            frameUs.back(), walkUs,
                            int32_t(walkUs) - int32_t(fenceDelta / 1000),
                            (unsigned long long)(fenceDelta / 1000), sleepUs,
                            int32_t(frameUs.back()) - int32_t(walkUs) - int32_t(sleepUs),
                            (unsigned long long)(g_gpuNsOfFrame / 1000),
                            (unsigned long long)(g_texRealUploads - prevTexForTrace),
                            (unsigned long long)((g_texUploadBytes - prevTexBytesForTrace)
                                                 / 1024),
                            (unsigned long long)((g_texUploadNs - prevTexNsForTrace) / 1000),
                            (unsigned long long)(texDecDelta / 1000),
                            (unsigned long long)(g_pipeCount - prevPipesForTrace));
                prevTexNsForTrace = g_texUploadNs;
                if (trace)
                {
                    static ProfilePhases prevPhase{};
                    // `now - prev` UNSIGNED, with a guard, because the counters it reads
                    // are ZEROED by CZ_VK_PROFILE's periodic window report. Without the
                    // guard the frame after each window prints 18446744073709 us — a
                    // wrapped negative — and a reader scanning for the worst frames finds
                    // six impossible ones at the top of every sorted list. That happened:
                    // it cost a pass over this data to notice the top six rows were the
                    // instrument and not the game.
                    //
                    // Zero, not the raw wrap: after a reset the true delta is unknown, and
                    // 0 is the only answer that cannot be mistaken for a measurement. The
                    // frame is still emitted, so nothing is silently dropped.
                    auto d = [](uint64_t now, uint64_t& prev) {
                        const uint64_t v = now >= prev ? now - prev : 0;
                        prev = now;
                        return (unsigned long long)(v / 1000);
                    };
                    // EVERY phase, not a chosen few. The first cut printed six and they
                    // summed to ~60% of the frame — because `record` is the RESIDUAL after
                    // recordState/recordVertex/recordIndex are subtracted from it, and
                    // those three were not in the list. A breakdown that does not add up
                    // is the same false-absence trap as the one the residual column was
                    // built to close, one level down: print all of them and let the
                    // arithmetic be checkable.
                    fprintf(trace,
                            " %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu "
                            "%llu %llu %llu %llu %llu %llu %llu %llu %llu\n",
                            d(g_prof.record, prevPhase.record),
                            d(g_prof.streams, prevPhase.streams),
                            d(g_prof.streamGuard, prevPhase.streamGuard),
                            d(g_prof.constants, prevPhase.constants),
                            d(g_prof.textures, prevPhase.textures),
                            d(g_prof.readback, prevPhase.readback),
                            d(g_prof.recordState, prevPhase.recordState),
                            d(g_prof.recordVertex, prevPhase.recordVertex),
                            d(g_prof.recordIndex, prevPhase.recordIndex),
                            d(g_prof.drawOther, prevPhase.drawOther),
                            d(g_prof.otherKey, prevPhase.otherKey),
                            d(g_prof.otherPipeline, prevPhase.otherPipeline),
                            d(g_prof.otherFetch, prevPhase.otherFetch),
                            d(g_prof.otherShader, prevPhase.otherShader),
                            d(g_prof.otherBegin, prevPhase.otherBegin),
                            d(g_prof.otherTail, prevPhase.otherTail),
                            d(g_prof.constVs, prevPhase.constVs),
                            d(g_prof.constPs, prevPhase.constPs),
                            d(g_prof.constShared, prevPhase.constShared),
                            d(g_prof.constVsCopy, prevPhase.constVsCopy),
                            d(g_prof.constVsPatch, prevPhase.constVsPatch));
                }
                prevTexForTrace = g_texRealUploads;
                prevTexBytesForTrace = g_texUploadBytes;
                prevPipesForTrace = g_pipeCount;
            }
            const double elapsed =
                std::chrono::duration<double>(now - windowStart).count();
            if (fpsLogSec > 0 && elapsed >= double(fpsLogSec) && frames > 1)
            {
                // The first sample of a window is the gap ACROSS the window boundary and
                // belongs to neither; dropping it costs one frame in a few hundred.
                std::sort(frameUs.begin() + 1, frameUs.end());
                const uint32_t medUs = frameUs[(frameUs.size() + 1) / 2];
                // THE TAIL, added in part 71 for the operator's TURN STUTTER — the one
                // performance problem on this port that has been reported as FELT rather
                // than measured, and the one a median is least able to see (gotcha 237:
                // read the distribution, not the centre). The window's frames are already
                // sorted for the median, so p99 and the share above 2x the median are two
                // array reads and one loop over data in cache — nothing this can cost is
                // visible against a frame. Without them a soak and a 360-degree turn
                // produce the same line and the stutter is unfalsifiable.
                const size_t n = frameUs.size();
                const uint32_t p99Us = frameUs[n - 1 - (n - 1) / 100];
                uint32_t worstUs = frameUs.back();
                size_t overTwice = 0;
                for (size_t i = 1; i < n; ++i)
                    if (frameUs[i] > 2 * medUs)
                        ++overTwice;
                uint32_t dMin = 0, dMed = 0, dMax = 0;
                if (!frameDraws.empty())
                {
                    std::vector<uint32_t> d = frameDraws;
                    std::sort(d.begin(), d.end());
                    dMin = d.front();
                    dMed = d[d.size() / 2];
                    dMax = d.back();
                }
                // THE PUMP'S OWN CPU PER FRAME, on the [fps] line and therefore
                // available in an UNPROFILED run (part 110 §3.1).
                //
                // It exists because the quantity part 110 has to measure — the pump's
                // CPU milliseconds per presented frame — was only ever obtainable by
                // crossing two instruments taken over DIFFERENT windows: `perf`'s or
                // `part50_thread_cpu.py`'s "% of one core" over its own 15 s sample,
                // divided into a frame rate from somewhere else. This project has a
                // name for that arithmetic and it invented 59 MB/frame that never
                // existed (`two-counters-are-not-a-pair`). One `clock_gettime` per FPS
                // WINDOW — not per frame — makes it one measurement over one window,
                // banded by the same draw count as everything else on this line.
                //
                // Free: the [fps] window is seconds long, so this is one vDSO read per
                // several hundred frames, against the profiler's thousands per frame.
                // It is printed unconditionally under CZ_FPS_LOG because a number that
                // needs its own env var is a number nobody has when they need it.
                double pumpCpuMs = 0.0, pumpDuty = 0.0;
                {
                    static uint64_t lastPumpCpuNs = 0;
                    static bool havePumpCpu = false;
                    timespec pts{};
                    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &pts);
                    const uint64_t nowNs =
                        uint64_t(pts.tv_sec) * 1000000000ull + uint64_t(pts.tv_nsec);
                    if (havePumpCpu && frames)
                    {
                        const double dNs = double(nowNs - lastPumpCpuNs);
                        pumpCpuMs = dNs * 1e-6 / double(frames);
                        pumpDuty = elapsed > 0.0 ? 100.0 * dNs * 1e-9 / elapsed : 0.0;
                    }
                    lastPumpCpuNs = nowNs;
                    havePumpCpu = true;
                }
                // THE GUEST'S OWN CPU PER FRAME, same window, same line (part 116). The
                // title's Main Thread and Draw Thread are the 8.8 ms floor part 110
                // found under `wall ~ max(pump, guest, GPU)`; a guest-side change (a
                // native CRT hook, PGO on the recompiled TUs) moves THESE columns and,
                // while the pump is the longer term, nothing else. Read through the
                // named thread's CPU clock; -1 until the title has named its threads.
                // Part 117: under CZ_PUMP_SPLIT=1 `pump cpu` above is THIS thread — the
                // one calling DoDraw, i.e. cz-draw — and the walk's own core is this
                // column. -1 on the one-thread pump.
                double walkCpuMs = -1.0;
                {
                    static double lastWalk = -1.0;
                    const double nowWalk = split::WalkCpuSeconds();
                    if (frames && lastWalk >= 0.0 && nowWalk >= 0.0)
                        walkCpuMs = (nowWalk - lastWalk) * 1e3 / double(frames);
                    lastWalk = nowWalk;
                }
                double guestMainMs = -1.0, guestDrawMs = -1.0;
                {
                    static double lastMain = -1.0, lastDraw = -1.0;
                    const double nowMain = GuestThread::CpuSecondsOf("Main Thread");
                    const double nowDraw = GuestThread::CpuSecondsOf("Draw Thread");
                    if (frames && lastMain >= 0.0 && nowMain >= 0.0)
                        guestMainMs = (nowMain - lastMain) * 1e3 / double(frames);
                    if (frames && lastDraw >= 0.0 && nowDraw >= 0.0)
                        guestDrawMs = (nowDraw - lastDraw) * 1e3 / double(frames);
                    lastMain = nowMain;
                    lastDraw = nowDraw;
                }
                // ...and where each thread's NON-CPU time went: ms/frame and calls/frame
                // in our kernel's waits, by kind — single-object, wait-any/all, sleep,
                // fence park (part 116 item 4). Same window, same denominator.
                char waitLine[256] = "";
                {
                    static uint64_t lastNs[2][GuestThread::kWaitKinds] = {};
                    static uint64_t lastCalls[2][GuestThread::kWaitKinds] = {};
                    static bool haveWait = false;
                    const char* names[2] = { "Main Thread", "Draw Thread" };
                    const char* kinds[GuestThread::kWaitKinds] = { "single", "multi", "sleep", "fence" };
                    size_t off = 0;
                    bool any = false;
                    for (int t = 0; t < 2; t++)
                    {
                        const GuestThread::WaitStats* w = GuestThread::WaitStatsOf(names[t]);
                        if (!w)
                            continue;
                        any = true;
                        off += snprintf(waitLine + off, sizeof waitLine - off, "%s%s:",
                                        t ? " | " : "", t ? "draw" : "main");
                        for (int k = 0; k < GuestThread::kWaitKinds; k++)
                        {
                            const uint64_t ns = w->ns[k].load(std::memory_order_relaxed);
                            const uint64_t calls = w->calls[k].load(std::memory_order_relaxed);
                            if (haveWait && frames && off < sizeof waitLine)
                                off += snprintf(waitLine + off, sizeof waitLine - off,
                                                " %s %.2f/%.1f", kinds[k],
                                                double(ns - lastNs[t][k]) * 1e-6 / double(frames),
                                                double(calls - lastCalls[t][k]) / double(frames));
                            lastNs[t][k] = ns;
                            lastCalls[t][k] = calls;
                        }
                    }
                    haveWait = any;
                }
                fprintf(stderr,
                        "[fps] %.1f fps mean (%.2f ms) | %.1f fps median (%.2f ms) | "
                        "p99 %.2f ms | worst %.2f ms | >2x med %.1f%% | "
                        "%llu frames in %.1f s | draws med %u (%u..%u) | "
                        "pump cpu %.2f ms/frame (%.0f%% of a core) | walk cpu %.2f | "
                        "guest main %.2f draw %.2f ms/frame\n",
                        double(frames) / elapsed, 1000.0 * elapsed / double(frames),
                        1e6 / double(medUs), double(medUs) / 1000.0,
                        double(p99Us) / 1000.0, double(worstUs) / 1000.0,
                        n > 1 ? 100.0 * double(overTwice) / double(n - 1) : 0.0,
                        (unsigned long long)frames, elapsed, dMed, dMin, dMax,
                        pumpCpuMs, pumpDuty, walkCpuMs, guestMainMs, guestDrawMs);
                if (waitLine[0])
                    fprintf(stderr, "[guestwait] ms/frame / calls/frame: %s\n", waitLine);
                ThreadBudget_PinSweep();   // CZ_GUEST_PIN (part 118): a no-op unless set
                // ...and the register-run census beside it when armed, PER WINDOW rather
                // than only at exit. The exit path is the right home for a summary
                // (gotcha 543) but it is not a reliable one: two runs tonight ended
                // without the SIGTERM handler printing anything at all, and a census that
                // only speaks on the way out is a census that some runs simply do not
                // have. A windowed print costs ten lines every FPS window on a
                // diagnostic-only arm and cannot be lost.
                // The scoped shared-block zero's own proof, per window rather than at
                // exit. Bytes NOT written, and the share of the 2,192 a draw used to
                // cost unconditionally — an arm that says "on" without saying "reached"
                // is how a null gets quoted as a saving.
                // THE CEILING PROBE'S ENGAGEMENT (part 110 §3.1), per window and
                // not only at exit (gotcha 543: two of part 109's runs ended without
                // the SIGTERM handler printing anything at all). Its control arm —
                // every other run — prints nothing, because the counter never moves.
                if (g_noDoDrawSkipped)
                    fprintf(stderr,
                            "[nododraw] DESTRUCTIVE: %llu draws skipped in total, "
                            "%u this frame — the walk ran, the draws did not. Read the "
                            "pump thread's CPU, never this run's wall time.\n",
                            (unsigned long long)g_noDoDrawSkipped,
                            unsigned(R->lastFrameDraws));
                // B1's ENGAGEMENT (part 111 §4.3), per window, and BOTH SIDES OF THE
                // BILL. A hit rate says the fast path ran; the worker milliseconds say
                // where the work it removed from the pump actually landed. Part 53 moved
                // 13.1 points off the pump and 33.2 appeared on the workers, so a line
                // that reports only the pump's half is not a measurement (gotcha 344).
                // The CONTROL ARM (`CZ_VK_NO_PREZERO=1`) prints the opposite: hits 0,
                // every draw inline.
                if (g_pzHits || g_pzMisses)
                {
                    static uint64_t lh = 0, lm = 0, lwns = 0, ldns = 0, lpre = 0, lin = 0;
                    const uint64_t h = g_pzHits, m = g_pzMisses;
                    const uint64_t wns = g_pzWorkerNsA.load(std::memory_order_relaxed);
                    const uint64_t dns = g_pzDrainNs;
                    const double inv = 1.0 / double(frames);
                    fprintf(stderr,
                            "[prezero] %.1f%% of draws served pre-zeroed (%llu hits, "
                            "%llu inline fallbacks this window) | %.2f MB/frame moved off "
                            "the pump, %.2f MB/frame still inline | worker %.3f ms/frame, "
                            "pump drain %.3f ms/frame | past watermark %llu | overflow "
                            "%llu\n",
                            (h + m > lh + lm)
                                ? 100.0 * double(h - lh) / double((h - lh) + (m - lm))
                                : 0.0,
                            (unsigned long long)(h - lh), (unsigned long long)(m - lm),
                            double(g_pzBytesPre - lpre) * inv / 1048576.0,
                            double(g_pzBytesInline - lin) * inv / 1048576.0,
                            double(wns - lwns) * inv * 1e-6,
                            double(dns - ldns) * inv * 1e-6,
                            (unsigned long long)g_pzBeyondWatermark,
                            (unsigned long long)g_pzOverflow);
                    lh = h; lm = m; lwns = wns; ldns = dns;
                    lpre = g_pzBytesPre; lin = g_pzBytesInline;
                }
                if (g_sharedZeroDraws)
                    fprintf(stderr,
                            "[sharedzero] %llu draws, %.0f bytes/draw not written "
                            "(%.1f%% of the %u-byte block)\n",
                            (unsigned long long)g_sharedZeroDraws,
                            double(g_sharedZeroSaved) / double(g_sharedZeroDraws),
                            100.0 * double(g_sharedZeroSaved) /
                                (double(g_sharedZeroDraws) * double(kSharedSize)),
                            kSharedSize);
                Pm4_RegRunCensusReport();
                if (split::g_on)
                {
                    // Part 117: the queue's health per window. `wspace` must stay 0 (W
                    // never blocked for room); `irqwait` is D's stall at INTERRUPT ops.
                    static split::Stats last{};
                    const split::Stats st = split::GetStats();
                    fprintf(stderr,
                            "[split] per frame: ops %.0f draws %.0f stores %.0f irq %.1f "
                            "logdw %.0f (%.0f runs + %.0f merged) | wspace %llu dempty %llu irqwait %.2f ms/frame "
                            "(%.0f us each) | didle %.2f ms/frame | waits unmet %.1f/frame "
                            "of which on OUR store %.1f, run-ahead %.1f\n",
                            double(st.ops - last.ops) / double(frames),
                            double(st.draws - last.draws) / double(frames),
                            double(st.stores - last.stores) / double(frames),
                            double(st.interrupts - last.interrupts) / double(frames),
                            double(st.logDwords - last.logDwords) / double(frames),
                            double(st.runs - last.runs) / double(frames),
                            double(st.runsMerged - last.runsMerged) / double(frames),
                            (unsigned long long)(st.wSpaceWaits - last.wSpaceWaits),
                            (unsigned long long)(st.dEmptyWaits - last.dEmptyWaits),
                            double(st.dIrqWaitNs - last.dIrqWaitNs) * 1e-6 / double(frames),
                            st.interrupts > last.interrupts
                                ? double(st.dIrqWaitNs - last.dIrqWaitNs) * 1e-3 /
                                      double(st.interrupts - last.interrupts)
                                : 0.0,
                            double(st.dIdleNs - last.dIdleNs) * 1e-6 / double(frames),
                            double(st.waitsUnmet - last.waitsUnmet) / double(frames),
                            double(st.waitsOnOurStore - last.waitsOnOurStore) / double(frames),
                            double(st.waitsByPending - last.waitsByPending) / double(frames));
                    fprintf(stderr,
                            "[split]   D idle after: draw %.2f store %.2f irq %.2f swap %.2f "
                            "other %.2f ms/frame | unmet by word: %08X %.1f  %08X %.1f  "
                            "%08X %.1f  %08X %.1f /frame\n",
                            double(st.dIdleByKindNs[1] - last.dIdleByKindNs[1]) * 1e-6 / double(frames),
                            double(st.dIdleByKindNs[2] - last.dIdleByKindNs[2]) * 1e-6 / double(frames),
                            double(st.dIdleByKindNs[3] - last.dIdleByKindNs[3]) * 1e-6 / double(frames),
                            double(st.dIdleByKindNs[4] - last.dIdleByKindNs[4]) * 1e-6 / double(frames),
                            double(st.dIdleByKindNs[0] - last.dIdleByKindNs[0]) * 1e-6 / double(frames),
                            st.waitVa[0], double(st.waitVaCount[0] - last.waitVaCount[0]) / double(frames),
                            st.waitVa[1], double(st.waitVaCount[1] - last.waitVaCount[1]) / double(frames),
                            st.waitVa[2], double(st.waitVaCount[2] - last.waitVaCount[2]) / double(frames),
                            st.waitVa[3], double(st.waitVaCount[3] - last.waitVaCount[3]) / double(frames));
                    last = st;
                }
                // Part 107 item 2: the Draw Thread's fence wait, per window, beside
                // the frame rate it is meant to move — so a plain crowd run (no phase
                // profiler) still says whether the park ENGAGED and how each episode
                // ended. A run whose parks all end in timeouts has a wake predicate that
                // never fires; a run with no parks at all has a wait that never waits.
                {
                    static FenceWaitStats lastRw;
                    const FenceWaitStats rw = FenceWait_Stats();
                    const double inv = 1.0 / double(frames);
                    fprintf(stderr,
                            "[fencewait] per frame: body %.1f | ready %.1f | spin-resolved "
                            "%.1f | parks %.1f (woken %.1f timeouts %.1f MISSED %.1f eagain %.1f) | "
                            "contended %.1f passthrough %.1f | stores seen %.1f wakes %.1f%s\n",
                            double(rw.bodyCalls - lastRw.bodyCalls) * inv,
                            double(rw.readyAtEntry - lastRw.readyAtEntry) * inv,
                            double(rw.spinResolved - lastRw.spinResolved) * inv,
                            double(rw.parks - lastRw.parks) * inv,
                            double(rw.parkWoken - lastRw.parkWoken) * inv,
                            double(rw.parkTimeouts - lastRw.parkTimeouts) * inv,
                            double(rw.parkMissed - lastRw.parkMissed) * inv,
                            double(rw.parkEagain - lastRw.parkEagain) * inv,
                            double(rw.contended - lastRw.contended) * inv,
                            double(rw.passthrough - lastRw.passthrough) * inv,
                            double(rw.storeChecks - lastRw.storeChecks) * inv,
                            double(rw.wakeCalls - lastRw.wakeCalls) * inv,
                            FenceWait_Enabled() ? "" : " [CZ_FENCE_PARK=0: spinning]");
                    lastRw = rw;
                }
                windowStart = now;
                frames = 0;
                frameUs.clear();
                frameDraws.clear();
            }
        }
    }

    if (!R->recording)
    {
        // A frame with no recorded work at all: present the previous contents rather
        // than nothing, so a stall in the draw path shows as a frozen picture instead
        // of a flicker between the real frame and black.
        Count("swap: nothing recorded");
        return;
    }

    // Read the colour target back and hand it to the window. A readback per frame is a
    // real cost and it is chosen deliberately: the alternative is a Vulkan swapchain on
    // the SDL window, which would put Vulkan on the window's thread and couple the
    // renderer to the windowing system that phase 3 deliberately kept at arm's length.
    // At the guest's own ~30 frames a second, 3.5 MB a frame is not what limits this.
    EndRendering();

    // Read back the front-buffer snapshot when there is one, and the raw EDRAM when
    // there is not. The fallback is deliberate and is announced by its own counter:
    // it is what a frame looks like before the surface identity is known, and seeing
    // it in the stats is how "the resolve match stopped working" stays visible
    // instead of turning into a picture that is subtly the wrong pass.
    auto frontSnap = R->snapshots.find(R->frontBuffer & 0x1FFFFFFF);
    if (frontSnap == R->snapshots.end())
        R->haveFrontSnapshot = false;
    // CZ_VK_MSAA: the raw-EDRAM fallback cannot blit or read back a multisampled
    // image, so it presents through the single-sample companion, filled by the
    // resolve a few lines below. The snapshot path is untouched — snapshots are
    // the resolve OUTPUT and stay single-sample.
    const bool msaaFallback =
        !R->haveFrontSnapshot && R->msaaSamples != VK_SAMPLE_COUNT_1_BIT;
    Image& source = R->haveFrontSnapshot ? frontSnap->second.image
                    : msaaFallback       ? R->colorResolve
                                         : R->color;
    // The present consumes the front-buffer snapshot's last copy.
    if (g_copyCensusOn && R->haveFrontSnapshot)
        CopyCensusSampled(R->frontBuffer & 0x1FFFFFFF);
    // The raw-EDRAM fallback is one of the three EDRAM readers, so a deferred clear
    // still pending must land before the blit/readback samples it.
    if (!R->haveFrontSnapshot)
        FlushPendingClears("clear: deferred FLUSHED for the EDRAM present fallback");
    // The raw-EDRAM fallback presents the FRAME's extent, not the EDRAM image's. Those
    // stopped being the same number when the EDRAM grew to hold the 1024-row shadow
    // cascade, and reading back the whole image would hand the window a 1280x1024
    // buffer as if it were the 1280x720 frame.
    // In HOST pixels: `frontWidth`/`frontHeight` are what the guest resolved and
    // `targetWidth`/`targetHeight` are what it thinks the screen is, and the image in
    // front of us is neither if a resolution scale is in force.
    const uint32_t width0 = RSX(R->haveFrontSnapshot ? R->frontWidth : R->targetWidth);
    const uint32_t height0 = RS(R->haveFrontSnapshot ? R->frontHeight : R->targetHeight);
    Count(R->haveFrontSnapshot ? "swap: presented the front-buffer resolve"
                               : "swap: presented raw EDRAM (no resolve matched)");
    if (msaaFallback)
    {
        Barrier(R->cmd, R->color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT);
        Barrier(R->cmd, R->colorResolve, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT);
        VkImageResolve rv{};
        rv.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        rv.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        rv.extent = { std::min(width0, R->color.width),
                      std::min(height0, R->color.height), 1 };
        vkCmdResolveImage(R->cmd, R->color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          R->colorResolve.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          1, &rv);
        Count("swap: EDRAM fallback resolved for present (CZ_VK_MSAA)");
    }

    // WHETHER THE READBACK STILL HAPPENS AT ALL. In the CZ_VK_SWAPCHAIN arm the window
    // gets its pixels from the swapchain blit below and nothing needs them in host
    // memory — except the picture instruments, every one of which walks the presented
    // frame on the CPU. So the readback survives in that arm exactly when one of them is
    // armed, and the run SAYS SO, because a swapchain run carrying a picture instrument
    // is paying for both paths and its `readback` column is not the arm's cost.
    //
    // AND UNTIL PART 76 "ARMED" MEANT "NAMED IN THE ENVIRONMENT", WHICH COST 15% OF THE
    // OPERATOR'S CROWD FRAME. The list below used to include `CZ_CAPTURE_KEY` and
    // `CZ_BURST_DUMP`, and `tools/play_session.sh` sets both unconditionally so that F9
    // and F8 work — so every play session since part 54 paid a whole-frame
    // `vkCmdCopyImageToBuffer` plus a 19.8 MB `memcpy` under a mutex, every frame, into a
    // buffer the swapchain never displays, to make two keys work that are pressed a
    // handful of times an hour. Measured in the part 75 operator session at **3.49 ms of
    // a 23.31 ms crowd frame** — the largest single column in it, and none of it the
    // game (gotcha 450). The swapchain of part 54 was built to delete this path and this
    // is what was cancelling it.
    //
    // So the instruments are split by TRIGGER, not by name:
    //
    //   * CONTINUOUS — they read EVERY presented frame and cannot be predicted, so the
    //     readback runs for the whole run when one of them is set. That is the list
    //     below, and it is exactly the set of `px` consumers that test no other flag.
    //   * EDGE-TRIGGERED — `CZ_CAPTURE_KEY` (F9) and `CZ_BURST_DUMP` (F8). Nothing is
    //     wanted until a key is pressed, so the press ARMS THE READBACK FOR THE FRAMES
    //     THAT FOLLOW (`R->readbackUntilFrame`, and `burstActive` for the burst's whole
    //     window). One frame of lag on a still screenshot is nothing and a burst is a
    //     second long, so neither loses anything an operator can see.
    //
    // Read once from the environment, like every other decision of this shape here: a
    // per-frame getenv is a syscall on the frame path, and a predicate that can change
    // mid-run makes two windows of one profile incomparable. The DYNAMIC half is not a
    // getenv — it is two fields on the renderer, written by the key handler below.
    //
    // Every consumer ALSO tests `px` itself, and a consumer that finds itself armed with
    // no pixels says so by name rather than doing nothing quietly: a predicate that
    // misses a case must produce a report, not a silence (gotcha 151).
    //
    // THE TRAP THIS OPENS, NAMED SO IT IS NOT WALKED INTO. Gating on "is an instrument
    // armed" ships a default path no gate in this project exercises, because every
    // picture gate here sets one of these variables — the same trap the staging-copy
    // note below describes. Two things close it: `CZ_VK_PRESENT_ALWAYS=1` is the
    // same-binary control arm that restores the pre-part-76 predicate exactly, and
    // `tools/part76_readback_gate.sh` runs a gate with NO picture instrument at all and
    // checks the counters instead of the pixels.
    static const bool wantCachedPixels =
        Env("CZ_VK_FRAME_STATS") || Env("CZ_VK_FRAME_DUMP") || Env("CZ_VK_SNAP_DUMP") ||
        Env("CZ_VK_SNAP_ON_BLACK") || Env("CZ_VK_SNAP_ON_DARK") ||
        Env("CZ_VK_SNAP_FRAME") || Env("CZ_VK_SKY_ASYM");
    // The control arm: force the readback on EVERY frame whatever is armed. That is a
    // superset of the pre-part-76 predicate (which forced it whenever `CZ_CAPTURE_KEY`
    // or `CZ_BURST_DUMP` was merely named), so a session run with it on and off is an
    // A/B on exactly this item — and it is also the only way to exercise the old path
    // from a run carrying no picture instrument at all, which is what the gate does.
    static const bool presentAlways = EnvOn("CZ_VK_PRESENT_ALWAYS");
    // The EDGE arm, re-read every frame. `burstActive` covers the burst's whole window;
    // `readbackUntilFrame` covers the two or three frames one F9 press needs, and it is a
    // frame NUMBER rather than a countdown so that two presses in quick succession cannot
    // shorten each other.
    // ...and the bug-report capture (host/bug_report.h), which wants a frame or three
    // after F8/F9 whether or not the dev instruments are armed.
    const bool edgeArmed = R->burstActive || R->frame <= R->readbackUntilFrame ||
                           BugReport_WantsPixels();
    const bool doReadback =
        !R->wantSwapchain || wantCachedPixels || presentAlways || edgeArmed;
    // Counted, both halves, because an arm with no counter cannot be shown to have
    // engaged (gotcha 151) — and this one is invisible in the picture by construction:
    // the whole point is that nothing on screen changes.
    if (R->wantSwapchain)
    {
        if (!doReadback)
            COUNT("readback: skipped (swapchain, no picture instrument armed)");
        else if (edgeArmed && !wantCachedPixels && !presentAlways)
            COUNT("readback: ran because F8/F9 armed it");
        else
            COUNT("readback: ran for a continuous picture instrument");
    }
    static bool saidWhy = false;
    if (R->wantSwapchain && (wantCachedPixels || presentAlways) && !saidWhy)
    {
        saidWhy = true;
        fprintf(stderr,
                "[vk] swapchain present with a CONTINUOUS picture instrument armed (or "
                "CZ_VK_PRESENT_ALWAYS=1): the present READBACK IS RUNNING ON EVERY "
                "FRAME, because those instruments walk the frame on the CPU. This run "
                "pays for both present paths and its `readback` column is NOT this arm's "
                "cost — take a frame-time A/B without one. F8/F9 alone no longer do this "
                "(part 76): they arm the readback for the frames they need.\n");
    }

    // Into THIS SLOT's readback buffer, not a shared one: with a frame in flight the
    // window has not necessarily fetched the previous frame's pixels yet.
    FrameSlot& rec = R->frames[R->frameSlot];
    if (doReadback)
    {
        // Part 76 split this off the always-on path; it survives whenever a picture
        // instrument is armed, and it is a whole-frame copy, so it has its own class. In a
        // plain play run this class must read ZERO — a non-zero one says an instrument is
        // armed that the run did not mean to arm, which is exactly the defect part 76
        // found in `play_session.sh`.
        GpuSeg _g(kGpReadback);
        Barrier(R->cmd, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_ASPECT_COLOR_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copy.imageExtent = { width0, height0, 1 };
        vkCmdCopyImageToBuffer(R->cmd, source.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               rec.present.buffer, 1, &copy);
    }
    // The swapchain blit goes LAST in the command buffer, after the readback copy when
    // both are present, because it is what the submit's semaphore signals on.
    //
    // NOT ACQUIRING IS THE ONLY SAFE WAY TO NOT PRESENT. An acquired image comes with a
    // semaphore the presentation engine will signal, and the contract is that something
    // waits on it; abandon the frame after acquiring and that semaphore is left signalled
    // with no waiter, so the next acquire reuses it in an illegal state. Both ways of
    // abandoning a frame here are knowable BEFORE the acquire — `CZ_VK_NO_SUBMIT` records
    // a frame and executes none of it (the ceiling arm), and a command buffer that is not
    // recording cannot carry a blit — so the acquire is simply not made. `CZ_VK_NO_SUBMIT`
    // therefore presents nothing at all in this arm, which is correct and consistent with
    // what that arm already documents: its picture is knowingly invalid.
    static const bool noSubmitArm = EnvOn("CZ_VK_NO_SUBMIT");
    if (R->wantSwapchain && R->recording && !noSubmitArm)
    {
        // The letterbox clear, the aspect-fit blit and the F4 overlay — all of it, because
        // "the present" is one region as far as a fix is concerned.
        GpuSeg _g(kGpPresent);
        RecordSwapchainBlit(source, width0, height0);
    }
    else if (R->wantSwapchain)
        Count("swap: no acquire (CZ_VK_NO_SUBMIT or nothing recorded) — nothing presented");

    // What this frame WAS, recorded next to the pixels it produced. Every present-side
    // instrument below reads this and not `R->frame`/`R->drawFingerprint`, which from
    // here on describe the frame being recorded rather than the one being shown.
    rec.frame = R->frame;
    rec.draws = R->drawsThisFrame;
    rec.vertices = R->verticesThisFrame;
    rec.drawFingerprint = R->drawFingerprint;
    rec.cameraFingerprint = R->cameraFingerprint;
    rec.width = width0;
    rec.height = height0;
    rec.bytes = size_t(width0) * height0 * 4;
    rec.presentable = true;
    // Recorded with the pixels, and read at the retire below rather than re-deriving it.
    rec.hasPixels = doReadback;
    if (doReadback)
        rec.pixelFrame = R->frame;

    SubmitFrame();
    // Immediately after the submit, and before the fence wait below: the present waits on
    // the submit's SEMAPHORE, so the window can be handed this frame while the CPU is
    // still retiring the previous one. That ordering is the item — the readback path
    // cannot show a frame until its bytes have arrived in host memory.
    if (R->wantSwapchain)
        PresentSwapchain();
    const int presentSlot = RetireOldestFrame();

    // Advance the ring for the next frame. It happens HERE, after the wait above, so
    // that when `BeginFrame` picks up this slot its fence has been observed to signal —
    // which is what makes resetting its command buffer and reusing its arena region
    // legal without a second wait at the top of the frame.
    R->frameSlot = (R->frameSlot + 1) % R->framesInFlight;

    // Reset the recorded frame's own accumulators here rather than after the stats line:
    // they belong to the frame that has just been submitted, and the stats line below is
    // now about a different one.
    R->drawFingerprint = 0;
    R->cameraFingerprint = 0;
    R->verticesThisFrame = 0;

    // Grow the arena HERE, at the frame boundary, rather than inside the next frame's
    // first draw. Both of these idle every frame still in flight before they touch
    // anything, which before part 23 the caller's fence wait did for them. See
    // GrowArenaIfNeeded for why the old placement was a measurement defect: it charged a
    // device-wait and an allocation to `other`.
    GrowArenaIfNeeded();
    // B1 (part 111): the shared sub-arena grows under the same rule, and then this
    // frame's pre-zero is posted. The ORDER is load-bearing — growth destroys the buffer
    // the workers write into, so it drains first and the dispatch that follows is against
    // the new one.
    GrowSharedArenaIfNeeded();
    Prezero_Dispatch();
    // Same site, and for the store the reason is stronger: its offsets are recorded into
    // command buffers, so reusing its memory before the GPU is done hands an in-flight
    // draw somebody else's vertices.
    PersistMaintenance();

    // Nothing to show yet — the first frame of a run with two slots, because the second
    // slot has never been submitted. One frame of a run.
    if (presentSlot < 0)
        return;
    FrameSlot& pres = R->frames[presentSlot];
    if (!pres.presentable)
        return;
    const uint32_t width1 = pres.width, height1 = pres.height;
    const size_t bytes = pres.bytes;
    // THE COPY EXISTS FOR THE INSTRUMENTS, AND ONLY FOR THEM (part 53, plan item 1.3).
    //
    // `pres.present.mapped` is HOST_CACHED (see ReadbackMemoryProps — it was made cached
    // deliberately, and CZ_VK_READBACK_UNCACHED=1 is still the arm for that), so reading
    // it costs what reading any other host buffer costs. Everything downstream of here
    // that looks at the picture — the frame stats, the PPM dumps, the black/dark
    // triggers, the uniform-colour census — was reading `presentPixels`, and
    // `Host_PresentPixels` then makes its own copy into the window's back buffer under
    // its own lock. So on a run with no picture instrument armed the intermediate buffer
    // was 3.5 MB copied per frame for nothing: 7 MB of traffic where 3.5 does.
    //
    // `px` is now the pixels to read, and it points at the mapped buffer directly. The
    // condition is the MEMORY TYPE, not which instruments are armed, and that is
    // deliberate: gating on the instruments would mean the default configuration took a
    // code path no gate in this project ever exercises, because every picture gate here
    // (the frame dump, the E3 correlation, CZ_VK_SNAP_*) sets one of them. Gating on
    // whether the readback is cached keeps the default path the ONLY path in every
    // ordinary run, and leaves the staging copy where it is genuinely needed — the
    // uncached fallback, where several whole-frame walks would each be a
    // write-combined read. `CZ_VK_PRESENT_STAGING=1` is the same-binary control arm.
    static const bool stagingCopy = !g_readbackCached || EnvOn("CZ_VK_PRESENT_STAGING");
    const uint8_t* px = nullptr;
    // `pres.hasPixels`, NOT `doReadback` — see the field. The two agree on every frame of
    // a run whose predicate never changes and disagree on exactly the frames an F8/F9
    // press straddles, which are the frames this whole item is about.
    if (!pres.hasPixels)
    {
        // The swapchain arm with no picture instrument: there are no host pixels and
        // nothing downstream of here has anything to look at. Everything below this point
        // — the frame stats, the dumps, the black triggers — is guarded on `px`, and
        // this counter is what makes the choice visible in the census rather than
        // inferable from an absence.
        Count("swap: presented through the swapchain (no host readback)");
    }
    else
    {
        ProfScope _p(&g_prof.readback);
        if (stagingCopy)
        {
            if (R->presentPixels.size() < bytes)
                R->presentPixels.resize(bytes);
            memcpy(R->presentPixels.data(), pres.present.mapped, bytes);
            px = R->presentPixels.data();
        }
        else
        {
            px = pres.present.mapped;
        }
        // THE INVARIANT, CHECKED RATHER THAN ASSUMED: the bytes in this slot must be the
        // frame this slot describes.
        //
        // It exists because the obvious gate for the part-76 off-by-one — "no burst frame
        // may be byte-identical to the F9 capture" — was BUILT, RUN AGAINST A DELIBERATELY
        // BROKEN BUILD, AND PASSED. The stale slot does hold an F9-era frame, but the F9
        // press arms `framesInFlight + 2` of them and the capture PPM is only one; the
        // stale read landed on a sibling. A content canary aimed at one artifact cannot
        // cover a window (gotcha 30 — and the control is what said so, not reasoning).
        //
        // This is the derived-from-an-invariant form instead: the slot stamps the frame
        // whose pixels it holds, and a disagreement is a defect by construction, with no
        // route, no camera and no second artifact required. It costs one comparison per
        // presented frame. Shown capable of firing: reverting the guard above to
        // `doReadback` makes it fire on the frame after every F8 press.
        if (pres.pixelFrame != pres.frame)
        {
            Count("PRESENT PIXELS ARE FROM A DIFFERENT FRAME — a stale readback slot");
            static int left = 8;
            if (TakeOne(left))
                fprintf(stderr,
                        "[vk] !! present slot describes frame %llu but holds frame %llu's "
                        "pixels — every picture instrument reading this frame is looking "
                        "at the wrong one\n",
                        (unsigned long long)pres.frame,
                        (unsigned long long)pres.pixelFrame);
        }
        Host_PresentPixels(px, width1, height1);
        // The bug-report capture's frame(s), copied out of the readback here (the slot
        // is reused next frame); a no-op unless a capture is waiting for pixels.
        BugReport_OfferPixels(px, width1, height1, pres.frame);
    }

    // CZ_VK_SNAP_ON_BLACK[=pct] — dump the whole resolve chain of the frame the picture
    // DIED on, triggered by the picture dying.
    //
    // The view-dependent whole-frame black is the port's top rendering defect and it has
    // never been captured, because CZ_VK_SNAP_DUMP fires on a frame NUMBER and this
    // event happens when a human turns a camera. Asking an operator to hit a frame index
    // is not a request anyone can fulfil, so every report of this defect has been the
    // black frame alone — which is consistent with every pass being wrong and with
    // exactly one being wrong (the reason CZ_VK_SNAP_DUMP exists at all).
    //
    // The trigger is a TRANSITION, not a threshold, and that is what makes it usable:
    // this runtime presents plenty of legitimately black frames during boot and loading,
    // so "coverage below x%" alone would fire on the first one and dump a chain nobody
    // wants. Requiring a LIT frame first means the dump lands on the frame where a
    // working picture stopped working, which is the only frame that can distinguish the
    // hypotheses.
    //
    // Both thresholds are settable, and that is what makes the instrument testable at
    // all. The first version folded arming and firing into one if/else, so a frame
    // could never do both — and its positive control (a 99% floor, which should fire on
    // essentially any frame) sat silent through a whole boot, because reaching the fire
    // branch still required coverage under the hard-coded 20% arming bar. An instrument
    // whose control cannot reach its own trigger has not been shown capable of firing
    // (gotcha 30). With `CZ_VK_SNAP_ON_BLACK=99 CZ_VK_SNAP_ON_BLACK_LIT=20` the second
    // lit frame of any run fires it.
    // And there is a HARD CAP on total episodes, because the arming logic re-arms on
    // every lit frame and therefore has no natural bound. Its own positive control
    // proved that the expensive way: a 99% floor fires on essentially every frame, which
    // dumped **9,833 PPMs** and refilled a tmpfs whose exhaustion kills this machine's
    // shell. The defect it exists to catch happens a handful of times in a session, so a
    // low cap costs nothing real and turns a mis-set threshold from a filled disk into a
    // few wasted files. CZ_VK_SNAP_ON_BLACK_MAX raises it.
    static const char* onBlackEnv = Env("CZ_VK_SNAP_ON_BLACK");
    static const double litPct =
        Env("CZ_VK_SNAP_ON_BLACK_LIT") ? atof(Env("CZ_VK_SNAP_ON_BLACK_LIT")) : 20.0;
    static int episodesLeft =
        Env("CZ_VK_SNAP_ON_BLACK_MAX") ? atoi(Env("CZ_VK_SNAP_ON_BLACK_MAX")) : 4;
    static bool sawLitFrame = false;
    static int onBlackBudget = 2;   // the transition frame, and the next one
    // CZ_VK_SNAP_ON_DARK=<meanLuma> — the same trigger on the metric the defect ACTUALLY
    // moves, and it dumps a BRIGHT reference chain to sit beside the dark one.
    //
    // The first headless run to sweep the camera in Still Creek found the defect
    // immediately and SNAP_ON_BLACK could not fire on it: sweeping the camera through
    // ~360 degrees swings the presented frame's mean luminance between ~27 and ~4.5
    // while its COVERAGE stays 30-75%. The frame goes very dark, not empty, so a
    // coverage floor of 0.5% never trips. Coverage was the right metric for the
    // operator's report — a whole-frame black — and it is the wrong one for the thing
    // that is actually measurable here; both are kept because the two thresholds
    // answer different questions and an instrument whose meaning silently changed
    // would invalidate every run taken with it.
    //
    // The PAIR is the point. One dark chain is consistent with "this pass is broken"
    // and with "the scene really is dark here"; a bright chain from the same location
    // seconds later, with the same surfaces at the same addresses, is the control that
    // separates them (gotcha 133 — one frame of an animated scene is one sample). So a
    // dark episode owes a bright reference, and the next frame that re-arms pays it.
    static const char* onDarkEnv = Env("CZ_VK_SNAP_ON_DARK");
    static const double darkLitLuma =
        Env("CZ_VK_SNAP_ON_DARK_LIT") ? atof(Env("CZ_VK_SNAP_ON_DARK_LIT")) : 20.0;
    static int darkEpisodesLeft =
        Env("CZ_VK_SNAP_ON_DARK_MAX") ? atoi(Env("CZ_VK_SNAP_ON_DARK_MAX")) : 3;
    static bool sawBrightFrame = false;
    static bool oweBrightReference = false;

    bool blackTransition = false;
    if (px && (onBlackEnv || onDarkEnv))
    {
        // Sampled every 16th pixel: this runs on the present path of every frame, and
        // the quantity is a whole-frame fraction that a 1-in-16 sample estimates to far
        // better than the 0.5 percentage points anyone cares about here.
        uint64_t lit = 0, seen = 0, luma = 0;
        for (size_t i = 0; i + 4 <= bytes; i += 64)
        {
            const uint8_t* p = px + i;
            if (p[0] || p[1] || p[2])
                lit++;
            luma += (77u * p[0] + 150u * p[1] + 29u * p[2]) >> 8;
            seen++;
        }
        const double covPct = seen ? 100.0 * double(lit) / double(seen) : 0.0;
        const double meanLuma = seen ? double(luma) / double(seen) : 0.0;

        if (onBlackEnv)
        {
            const double floorPct = atof(onBlackEnv) > 0.0 ? atof(onBlackEnv) : 0.5;
            // Fire first, then arm — two independent tests rather than an if/else, so a
            // control whose thresholds overlap can exercise the trigger.
            if (sawLitFrame && covPct < floorPct && onBlackBudget > 0 && episodesLeft > 0)
            {
                blackTransition = true;
                onBlackBudget--;
                episodesLeft--;
                sawLitFrame = false;
                fprintf(stderr,
                        "[vk] SNAP_ON_BLACK: frame %llu went black (%.3f%% lit, floor "
                        "%.2f%%) — dumping the resolve chain (%d episode dumps left)\n",
                        (unsigned long long)pres.frame, covPct, floorPct, episodesLeft);
            }
            if (covPct >= litPct)
            {
                sawLitFrame = true;
                onBlackBudget = 2;      // re-arm, so a second episode is caught too
            }
        }

        if (onDarkEnv)
        {
            const double floorLuma = atof(onDarkEnv) > 0.0 ? atof(onDarkEnv) : 8.0;
            // CZ_VK_SNAP_ON_DARK_AFTER_MS=N — ignore everything before N ms of wall
            // clock. Not a refinement: the boot, the title screen and every loading
            // screen fade, so an episode budget aimed at gameplay is spent before
            // gameplay starts. Wall time rather than a frame index because the recipes
            // that reach gameplay are written in seconds (CZ_FAKE_START_MS intervals)
            // and a frame index for the same moment moves with the frame rate.
            static const auto darkT0 = std::chrono::steady_clock::now();
            static const long long afterMs =
                Env("CZ_VK_SNAP_ON_DARK_AFTER_MS")
                    ? atoll(Env("CZ_VK_SNAP_ON_DARK_AFTER_MS")) : 0;
            const long long nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::steady_clock::now() - darkT0).count();
            const bool live = nowMs >= afterMs;
            if (live && sawBrightFrame && meanLuma < floorLuma && darkEpisodesLeft > 0)
            {
                blackTransition = true;
                darkEpisodesLeft--;
                sawBrightFrame = false;
                oweBrightReference = true;
                fprintf(stderr,
                        "[vk] SNAP_ON_DARK: frame %llu went DARK (mean luma %.2f, floor "
                        "%.2f, %.2f%% lit) — dumping the resolve chain (%d left)\n",
                        (unsigned long long)pres.frame, meanLuma, floorLuma, covPct,
                        darkEpisodesLeft);
            }
            if (meanLuma >= darkLitLuma)
            {
                if (live && oweBrightReference)
                {
                    oweBrightReference = false;
                    blackTransition = true;
                    fprintf(stderr,
                            "[vk] SNAP_ON_DARK: frame %llu is the BRIGHT REFERENCE for "
                            "the episode above (mean luma %.2f, %.2f%% lit)\n",
                            (unsigned long long)pres.frame, meanLuma, covPct);
                }
                sawBrightFrame = true;
            }
        }
    }

    // CZ_VK_FRAME_DUMP=<dir> writes every 64th frame as a PPM. This is the instrument
    // that makes the renderer checkable WITHOUT a window, which matters more than it
    // sounds: every other gate this project owns is a log diff, and "the picture is
    // right" is the one claim that needs an image. A headless run plus a directory of
    // frames is a self-servable version of the E-screenshot comparison.
    // CZ_VK_FRAME_DUMP_EVERY=N overrides the 64. A screen the synthetic-input arm walks
    // THROUGH rather than parks on can be shorter than 64 frames, and one dump of it is
    // one sample of a transition — the save-slot panel below appeared in exactly one
    // frame of a 180 s boot.
    // ---- F8: THE BURST ----------------------------------------------------------
    //
    // WHY THIS EXISTS. The operator described a defect no single frame can show: *"The
    // decals how it looks like is pretty much normal but it appears and disappear like
    // flicker make it so when I press f8 it records all frame for a second so you can see
    // it."* A screenshot of a flicker is a screenshot of one PHASE of it, and which phase
    // you get is luck (gotcha 133). F9 answers "what does it look like"; F8 answers "what
    // does it do over time".
    //
    // AND IT IS BUILT TO DISCRIMINATE, not just to illustrate. Two mechanisms produce an
    // identical still image and need opposite fixes:
    //
    //   * the draw is ISSUED every frame and loses a depth fight — z-fighting, which is
    //     what a decal does when the guest's polygon offset is not honoured, and the
    //     leading hypothesis here because this renderer sets no `depthBiasEnable` at all;
    //   * the draw is DROPPED on some frames — by the guest, by predication, by a bin
    //     mask, or by one of our own declines.
    //
    // Under the first, the draw count and `drawFingerprint` are IDENTICAL frame to frame
    // while the pixels change. Under the second they move. So the manifest carries both
    // per frame, and the burst answers the question rather than merely showing it.
    //
    // `CZ_BURST_DUMP=<dir>` arms it; F8 fires it; `CZ_BURST_DUMP_MS` (default 1000) is the
    // window and `CZ_BURST_DUMP_MAX` (default 300) bounds the disk — at 95 fps and 720p a
    // second is ~260 MB, which is fine on a disk and would be a disaster in /tmp, so the
    // directory is the operator's to choose and the default is under their troubleshooting
    // tree rather than a tmpfs.
    if (Env("CZ_BURST_DUMP") && Host_ConsumeBurstDumpPressed())
    {
        if (R->burstActive)
        {
            // A second press during a burst ENDS it rather than restarting it, so the
            // operator can bound a recording they have already seen enough of.
            fprintf(stderr, "[vk] burst #%u: stopped early by a second F8 (%u frames)\n",
                    R->burstSeq, R->burstFrames);
            R->burstEndNs = 0;
        }
        else
        {
            static uint32_t seq = 0;
            R->burstSeq = ++seq;
            R->burstActive = true;
            R->burstFrames = 0;
            // Same arm as F9's, for the same reason (part 76 item 1) — `burstActive`
            // alone already keeps the readback on for the burst's whole window, so this
            // only covers the couple of frames at the very start whose readback decision
            // was made before this press was consumed. Those frames are reported below
            // rather than silently missing.
            R->readbackUntilFrame =
                std::max(R->readbackUntilFrame, R->frame + R->framesInFlight + 2);
            ++R->readbackArmedPresses;
            R->burstNoPixelFrames = 0;
            uint64_t ms = 1000;
            if (const char* m = Env("CZ_BURST_DUMP_MS"))
                ms = strtoull(m, nullptr, 10);
            R->burstEndNs = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count()) +
                            ms * 1000000ull;
            char path[512];
            snprintf(path, sizeof path, "%s/burst%02u_manifest.txt", Env("CZ_BURST_DUMP"),
                     R->burstSeq);
            R->burstManifest = fopen(path, "w");
            if (R->burstManifest)
                fprintf(R->burstManifest,
                        "# file frame draws vertices drawFingerprint cameraFingerprint "
                        "meanLuma distinctColours pixelHash\n");
            // The per-draw census, default-on. Its absence is what stopped part 56's
            // decal analysis: the burst could show a decal blinking but could not say
            // whether its draw was issued on the dark frames. Read it with
            // tools/burst_read.py, which aligns it with the PPMs by frame number.
            const char* censusWant = Env("CZ_BURST_CENSUS");
            if (!censusWant || strcmp(censusWant, "0") != 0)
            {
                snprintf(path, sizeof path, "%s/burst%02u_census.txt",
                         Env("CZ_BURST_DUMP"), R->burstSeq);
                R->burstCensusFile = fopen(path, "w");
                if (R->burstCensusFile)
                    fprintf(R->burstCensusFile,
                            "# every draw of every burst frame, fN <census line>. Same "
                            "fields as a capture census; v0= is the first vertex's "
                            "first three dwords, the identity that survives a camera "
                            "move.\n");
                else
                    fprintf(stderr, "[vk] burst #%u: cannot open its census file — the "
                                    "issued-or-discarded half of the answer will be "
                                    "missing\n", R->burstSeq);
                R->burstCensusLines = 0;
            }
            fprintf(stderr,
                    "[vk] burst #%u ARMED: every presented frame for %llu ms into %s%s\n",
                    R->burstSeq, (unsigned long long)ms, Env("CZ_BURST_DUMP"),
                    R->burstManifest ? "" : "  — !! the manifest could not be opened, so "
                                            "the frames will have no draw data beside them");
        }
    }
    if (R->burstActive)
    {
        uint32_t maxFrames = 300;
        if (const char* m = Env("CZ_BURST_DUMP_MAX"))
            maxFrames = uint32_t(strtoul(m, nullptr, 10));
        const uint64_t nowNs =
            uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count());
        if (!px)
        {
            // Armed with no pixels. Say so by name rather than recording nothing, which
            // would read as "the defect did not happen" (gotcha 151).
            //
            // Since part 76 this is EXPECTED for the first frame or two of every burst
            // in the swapchain arm, and only those: F8 is consumed here, at the bottom
            // of a swap whose readback decision was made at the top, so the earliest
            // frame this press can arm is the next one and the earliest frame whose
            // pixels arrive is the one after that. Counted rather than printed per
            // frame, and printed once at the end of the burst — a per-frame line here
            // would be two lines of noise on every burst, and a burst that lost
            // THIRTY frames would look exactly the same as one that lost two.
            ++R->burstNoPixelFrames;
        }
        else
        {
            char path[512];
            snprintf(path, sizeof path, "%s/burst%02u_%04u_f%06llu.ppm",
                     Env("CZ_BURST_DUMP"), R->burstSeq, R->burstFrames,
                     (unsigned long long)pres.frame);
            if (FILE* f = fopen(path, "wb"))
            {
                fprintf(f, "P6\n%u %u\n255\n", width1, height1);
                for (size_t i = 0; i < bytes; i += 4)
                    fwrite(&px[i], 1, 3, f);
                fclose(f);
                if (R->burstManifest)
                    fprintf(R->burstManifest,
                            "%s %llu %llu %llu %016llx %016llx\n",
                            strrchr(path, '/') ? strrchr(path, '/') + 1 : path,
                            (unsigned long long)pres.frame,
                            (unsigned long long)pres.draws,
                            (unsigned long long)pres.vertices,
                            (unsigned long long)pres.drawFingerprint,
                            (unsigned long long)pres.cameraFingerprint);
                ++R->burstFrames;
            }
        }
        if (nowNs >= R->burstEndNs || R->burstFrames >= maxFrames)
        {
            R->burstActive = false;
            if (R->burstManifest)
            {
                fclose(R->burstManifest);
                R->burstManifest = nullptr;
            }
            if (R->burstCensusFile)
            {
                fclose(R->burstCensusFile);
                R->burstCensusFile = nullptr;
                fprintf(stderr, "[vk] burst #%u census: %llu draw lines\n", R->burstSeq,
                        (unsigned long long)R->burstCensusLines);
            }
            fprintf(stderr,
                    "[vk] burst #%u DONE: %u frames into %s — read it with "
                    "tools/burst_read.py%s",
                    R->burstSeq, R->burstFrames, Env("CZ_BURST_DUMP"),
                    R->burstNoPixelFrames ? "" : "\n");
            if (R->burstNoPixelFrames)
                fprintf(stderr,
                        " (+%llu frame(s) at the start with no readback pixels yet — "
                        "expected, see the readback predicate; more than 2 or 3 means "
                        "the F8 arm is not reaching the predicate)\n",
                        (unsigned long long)R->burstNoPixelFrames);
        }
    }
    // Decide ONCE, at the frame boundary, whether the NEXT frame's draws go into the
    // burst census — the draw path must not read burst timing state mid-frame, or the
    // census could hold part of a frame and read as a draw-count change (gotcha 109's
    // shape: a partial list is not a count). CZ_BURST_CENSUS_EVERY=N thins to every Nth
    // burst frame for long bursts; the default is every frame, because "issued or not"
    // is a per-frame question.
    {
        uint32_t every = 1;
        if (const char* e = Env("CZ_BURST_CENSUS_EVERY"))
            every = std::max(1u, uint32_t(strtoul(e, nullptr, 10)));
        R->burstCensusThisFrame = R->burstActive && R->burstCensusFile &&
                                  (R->burstFrames % every) == 0;
    }

    // The CZ_CAPTURE_KEY picture, written from the same readback the periodic dump uses.
    // Separate from the loop below rather than folded into its interval test, because
    // this one has to fire on EXACTLY the armed frame — the interval test would either
    // miss it or, if the interval were forced to 1, write every frame of the run.
    if (px && R->capturePictureFrame && pres.frame == R->capturePictureFrame)
    {
        R->capturePictureFrame = 0;
        char path[512];
        snprintf(path, sizeof path, "%s/capture_%06llu.ppm", Env("CZ_CAPTURE_KEY"),
                 (unsigned long long)pres.frame);
        if (FILE* f = fopen(path, "wb"))
        {
            fprintf(f, "P6\n%u %u\n255\n", width1, height1);
            for (size_t i = 0; i < bytes; i += 4)
                fwrite(&px[i], 1, 3, f);
            fclose(f);
            fprintf(stderr, "[vk] capture: wrote %s (%ux%u)%s\n", path, width1, height1,
                    R->drawIdRanOnFrame == pres.frame
                        ? " — NOT A PICTURE: CZ_VK_DRAW_ID painted this frame's draw "
                          "indices, so read the drawid_* snapshot instead"
                        : "");
        }
        else
        {
            // A capture that silently writes nothing is worse than no capture: the
            // operator walks away believing the evidence exists (gotchas 25, 151).
            fprintf(stderr, "[vk] capture: CANNOT WRITE %s — the picture is LOST\n", path);
        }

        // ...AND THE POSE, so the shot can be taken again.
        //
        // Every picture finding in this port has been anchored to "the operator walked
        // somewhere and pressed F9", which is not a reproducible experiment: nothing
        // headless can return to that spot, and the striped-material class picks a
        // different quality level on each boot, so the second visit is a different
        // measurement. What makes a shot repeatable is the CAMERA and the PLAYER, so
        // both are recorded beside the picture.
        //
        // Both are written RAW — the 16 float4 vertex constants the camera fingerprint
        // hashes (view-projection and world matrices) and the head of the player's game
        // object. Deriving an eye position or naming a position field here would be
        // guessing at a layout; two .pose files taken in different places name those
        // fields by what CHANGED, and that analysis belongs in a tool that can be fixed
        // without a rebuild (tools/pose_read.py).
        snprintf(path, sizeof path, "%s/capture_%06llu.pose", Env("CZ_CAPTURE_KEY"),
                 (unsigned long long)pres.frame);
        if (FILE* f = fopen(path, "w"))
        {
            fprintf(f, "# frame %llu  cameraFingerprint %016llx  drawFingerprint %016llx\n",
                    (unsigned long long)pres.frame,
                    (unsigned long long)R->cameraFingerprint,
                    (unsigned long long)R->drawFingerprint);
            fprintf(f, "# vc[i]  = constants at the frame's FIRST draw (usually the SHADOW "
                       "pass: its view matrix is the LIGHT's)\n");
            fprintf(f, "# bvc[i] = constants at the frame's BIGGEST draw (%u verts) — the "
                       "SCENE camera. Prefer these.\n", R->camBigVerts);
            for (int which = 0; which < 2; which++)
            {
                const uint32_t* src = which ? R->camConstsBig : R->camConsts;
                for (uint32_t i = 0; i < 64; i += 4)
                {
                    float v[4];
                    for (uint32_t k = 0; k < 4; k++)
                    {
                        const uint32_t bits = src[i + k];
                        memcpy(&v[k], &bits, 4);
                    }
                    fprintf(f, "%s%-2u %.6f %.6f %.6f %.6f\n", which ? "bvc" : "vc",
                            i / 4, v[0], v[1], v[2], v[3]);
                }
            }
            // The object dump lives in debug_tunables.cpp, which already owns guest
            // memory access and the pointer itself; GuestRangeOk here would be the
            // wrong check anyway, since it validates only the physical texture arena
            // and a game object is an ordinary virtual address.
            // THE POSITION ITSELF, which is the whole point of the pose: read via
            // the guest's own getplayerinfo path (obj->vtable[0x18]), not inferred
            // from the object dump below. The dump stays because it is what named
            // this field's neighbours, and because an unexplained struct is worth
            // keeping while the layout is still being learned.
            float pos[3];
            long long ageMs = -1;
            if (CZ_DebugPlayerPos(pos, &ageMs))
                fprintf(f, "player_pos %.4f %.4f %.4f   # read %lld ms before this "
                           "capture, via getplayerinfo's vtable[0x18]\n",
                        pos[0], pos[1], pos[2], ageMs);
            else
                fprintf(f, "# player_pos UNAVAILABLE — no level running, or the "
                           "lookup failed\n");
            const uint32_t obj = CZ_DebugWritePlayerObject(f, 2048);
            fclose(f);
            fprintf(stderr, "[vk] capture: wrote %s (camera + player object %08X)\n",
                    path, obj);
        }
        else
            fprintf(stderr, "[vk] capture: CANNOT WRITE %s — the pose is LOST\n", path);
    }
    // CZ_VK_SKY_ASYM=1 — SCORE THE HALF-SCREEN SKY FLICKER WITHOUT A HUMAN WATCHING.
    //
    // The operator's report has always been the same shape: *"sky flicker from half the
    // screen switching from right to left depending of moment"*. That is a LEFT/RIGHT
    // asymmetry in the sky that CHANGES BETWEEN FRAMES — this title renders in left/right
    // 640-wide tiles (gotcha 265), so it is exactly what a per-tile difference looks like.
    //
    // Every verdict on this defect so far has come from an eye and a three-minute run, and
    // that is why it cannot be settled: one clean run cannot be told from a run that did
    // not trigger it, which is the operator's own objection and is why part 72 built a
    // revert arm. So this measures it: mean luma of the top strip, split left and right,
    // and the statistic reported is not the asymmetry itself but **how often it FLIPS SIGN
    // between consecutive frames**, because a steady asymmetry is just a scene with a
    // bright side and a flicker is the alternation.
    //
    // It needs host pixels, so it joins the readback list above; on the swapchain arm that
    // forces a readback the swapchain path would otherwise skip. Cost is one pass over the
    // top quarter of the image per presented frame — a diagnostic arm, never a default.
    static const bool skyAsym = EnvOn("CZ_VK_SKY_ASYM");
    if (skyAsym && px)
    {
        const uint32_t strip = std::max(1u, height1 / 4);      // the sky occupies the top
        double sumL = 0.0, sumR = 0.0;
        uint64_t nL = 0, nR = 0;
        const uint32_t half = width1 / 2;
        for (uint32_t y = 0; y < strip; ++y)
            for (uint32_t x = 0; x < width1; ++x)
            {
                const uint8_t* q = px + (size_t(y) * width1 + x) * 4;
                const double l = 0.2126 * q[0] + 0.7152 * q[1] + 0.0722 * q[2];
                if (x < half) { sumL += l; ++nL; }
                else          { sumR += l; ++nR; }
            }
        const double d = (nL ? sumL / double(nL) : 0.0) - (nR ? sumR / double(nR) : 0.0);
        static double prev = 0.0;
        static bool havePrev = false;
        ++g_skyFrames;
        g_skyAbsSum += std::fabs(d);
        if (havePrev)
        {
            // A FLIP is a sign change with both sides meaningfully non-zero — a asymmetry
            // dithering around 0.0 is noise, not a flicker, and counting it would make
            // every arm look identical.
            if (((prev > 0.25 && d < -0.25) || (prev < -0.25 && d > 0.25)))
                ++g_skyFlips;
            g_skyStepSum += std::fabs(d - prev);
            if (std::fabs(d - prev) > g_skyStepMax)
                g_skyStepMax = std::fabs(d - prev);
        }
        // AND THE RAW SERIES, because the first cut of this instrument shipped a single
        // summary (sign flips) and it did NOT discriminate: 1.10% on a run the operator
        // called flickering against 0.86% on one they called clean. The statistic was
        // wrong, not the measurement — a turning camera changes which side is brighter, so
        // sign flips count the route as much as the defect. Writing the per-frame series
        // means the statistic can be chosen AFTER looking at the data instead of guessed
        // before, which is the whole reason a summary is the wrong thing to collect first.
        static FILE* series = [] () -> FILE* {
            const char* f = Env("CZ_VK_SKY_ASYM");
            if (!f || !*f || !strcmp(f, "1"))
                return nullptr;
            FILE* h = fopen(f, "w");
            if (h)
                fprintf(h, "frame draws asym\n");
            else
                fprintf(stderr, "[vk] CZ_VK_SKY_ASYM: CANNOT WRITE %s — no series\n", f);
            return h;
        }();
        if (series)
            fprintf(series, "%llu %u %.4f\n", (unsigned long long)pres.frame,
                    uint32_t(R->drawsThisFrame), d);
        prev = d;
        havePrev = true;
    }
    static const char* dumpDir = Env("CZ_VK_FRAME_DUMP");
    static const uint64_t dumpEvery =
        Env("CZ_VK_FRAME_DUMP_EVERY")
            ? std::max<uint64_t>(1, strtoull(Env("CZ_VK_FRAME_DUMP_EVERY"), nullptr, 10))
            : 64;
    if (px && dumpDir && (pres.frame % dumpEvery) == 0)
    {
        // Create the directory, and SAY SO if the frames cannot be written. This used to
        // be a bare fopen whose failure was silent, so a run pointed at a directory that
        // did not exist produced an empty result that looked exactly like a run whose
        // renderer drew nothing — and the picture check is the one gate in this project
        // that has no log-diff substitute. An instrument that can produce nothing without
        // complaining is not an instrument (gotchas 25, 151).
        static bool dirReady = false;
        static bool complained = false;
        if (!dirReady)
        {
            std::error_code ec;
            std::filesystem::create_directories(dumpDir, ec);
            dirReady = true;
        }
        char path[512];
        snprintf(path, sizeof path, "%s/frame_%06llu.ppm", dumpDir,
                 (unsigned long long)pres.frame);
        if (FILE* f = fopen(path, "wb"))
        {
            fprintf(f, "P6\n%u %u\n255\n", width1, height1);
            for (size_t i = 0; i < bytes; i += 4)
                fwrite(&px[i], 1, 3, f);
            fclose(f);
        }
        else if (!complained)
        {
            complained = true;
            fprintf(stderr, "[vk] CZ_VK_FRAME_DUMP cannot write %s — no frames will be "
                            "dumped this run\n",
                    path);
        }
    }

    // CZ_VK_FRAME_STATS=<file> — one line per frame: what the guest asked for, and what
    // came out. This is the raw material for tools/frame_compare.py, which aligns two
    // runs by CONTENT and only then compares their pictures.
    //
    // The output measurements are deliberately cheap and whole-image (coverage, mean
    // luminance, distinct colours, a pixel hash) rather than a per-pixel dump: the
    // question a renderer A/B asks is "did this frame change", and for that a small
    // vector of aggregates over the same content is enough — while a per-pixel dump at
    // 30 frames a second is 100 MB a run nobody reads.
    // CZ_VK_EXPOSURE_TRACE=<file> — one line per frame: how many draws set an exposure
    // and the range of values they set. Written here, at the present, so its frame
    // numbers are the same ones CZ_VK_SNAP_FRAME and CZ_VK_FRAME_STATS use.
    static FILE* expFile = nullptr;
    static bool expTried = false;
    if (!expTried)
    {
        expTried = true;
        if (const char* path = Env("CZ_VK_EXPOSURE_TRACE"))
        {
            expFile = fopen(path, "w");
            if (expFile)
                fprintf(expFile, "# frame draws expMin expMax\n");
            else
                fprintf(stderr, "[vk] cannot write CZ_VK_EXPOSURE_TRACE=%s\n", path);
        }
    }
    if (expFile)
    {
        fprintf(expFile, "%llu %u %.6f %.6f\n", (unsigned long long)R->frame,
                R->expDraws, double(R->expMin), double(R->expMax));
        // Reset unconditionally, including when the file could not be opened, so the
        // counters never accumulate across frames in a run that is not tracing.
    }
    R->expDraws = 0;

    static FILE* statsFile = nullptr;
    static bool statsTried = false;
    if (!statsTried)
    {
        statsTried = true;
        if (const char* path = Env("CZ_VK_FRAME_STATS"))
        {
            statsFile = fopen(path, "w");
            if (statsFile)
                fprintf(statsFile,
                        "# frame draws vertices drawFingerprint cameraFingerprint "
                        "width height coveragePct meanLuma distinctColours pixelHash "
                        "surfW surfH surfCoveragePct surfMeanLuma surfDistinct "
                        "surfHash msec\n");
            else
                fprintf(stderr, "[vk] cannot write CZ_VK_FRAME_STATS=%s\n", path);
        }
    }
    // CZ_VK_FRAME_STATS_SURFACE=<hex> — measure THAT resolve surface as well as the
    // presented frame.
    //
    // This is not a refinement, it is the thing that makes the metric work at all. The
    // first version measured only the presented front buffer, which at the title screen
    // is the logo era: mostly UI, 2-36% covered. Disabling the 16-bit texcoord
    // unswizzle — a change that touches 476,858 draws a run — moved it by 0.1
    // percentage points, i.e. the metric could not see a defect it was built to catch,
    // because the defect lives on the SCENE surface and the metric was looking at the
    // overlay. Gotcha 30: a test that has never failed has not been shown capable of
    // failing, and this one was shown incapable.
    //
    // Set it to the scene's resolve destination. At the title screen that is
    // **0684B000** — NOT 06BE4000, which was written here and quoted project-wide from
    // phase 5 to part 13 and is the scene DEPTH's first tile. It held colour pixels
    // only because our resolve copied the colour buffer for depth resolves too, so the
    // wrong label was confirmed every time it was checked (part 14, gotcha 205).
    // CZ_VK_RESOLVE_TRACE names the right address for any era.
    static const char* surfaceEnv = Env("CZ_VK_FRAME_STATS_SURFACE");
    static const uint32_t statsSurface =
        surfaceEnv ? uint32_t(strtoul(surfaceEnv, nullptr, 16)) & 0x1FFFFFFF : 0;
    std::vector<uint8_t> surfacePixels;
    uint32_t surfaceW = 0, surfaceH = 0;
    if (statsFile && statsSurface)
    {
        // No depth bit in the key, deliberately: "coverage" and "mean luminance" do not
        // mean anything over a depth surface, and the snapshot map's key carries the
        // distinction so the metric cannot accidentally read one through the wrong
        // image aspect.
        auto sit = R->snapshots.find(statsSurface);
        if (sit != R->snapshots.end())
        {
            if (g_copyCensusOn)
                CopyCensusSampled(statsSurface);
            const uint64_t n =
                uint64_t(sit->second.image.width) * sit->second.image.height * 4;
            if (n <= R->readback.size)
            {
                RunImmediate([&](VkCommandBuffer cb) {
                    Barrier(cb, sit->second.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_IMAGE_ASPECT_COLOR_BIT);
                    VkBufferImageCopy c{};
                    c.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                    c.imageExtent = { sit->second.image.width, sit->second.image.height,
                                      1 };
                    vkCmdCopyImageToBuffer(cb, sit->second.image.image,
                                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                           R->readback.buffer, 1, &c);
                    Barrier(cb, sit->second.image,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_IMAGE_ASPECT_COLOR_BIT);
                });
                surfacePixels.assign(R->readback.mapped, R->readback.mapped + n);
                surfaceW = sit->second.image.width;
                surfaceH = sit->second.image.height;
            }
        }
    }

    if (px && statsFile)
    {
        ProfScope _fs(&g_prof.frameStats);
        uint64_t lit = 0, lumaSum = 0, ph = 0xCBF29CE484222325ull;
        // Distinct colours exactly, without a hash set: the frame is RGBA8 and a
        // 2^24-bit bitmap is 2 MB, which is cheaper than a hash table per frame and
        // gives an exact count rather than an estimate.
        static std::vector<uint64_t> seenBits;
        seenBits.assign(1u << 18, 0); // 2^24 bits
        uint64_t distinct = 0;
        for (size_t i = 0; i < bytes; i += 4)
        {
            const uint32_t r = px[i], g = px[i + 1], b = px[i + 2];
            const uint32_t rgb = (r << 16) | (g << 8) | b;
            if (rgb)
                ++lit;
            lumaSum += (r * 54 + g * 183 + b * 19) >> 8;
            const uint32_t word = rgb >> 6, bit = rgb & 63;
            if (!(seenBits[word] & (1ull << bit)))
            {
                seenBits[word] |= 1ull << bit;
                ++distinct;
            }
            ph ^= rgb;
            ph *= 0x100000001B3ull;
        }
        const uint64_t pixels = bytes / 4;
        // The named surface, measured the same way. Zeros when it was not requested or
        // does not exist this frame, which frame_compare.py reads as "no surface data"
        // rather than as an empty surface.
        uint64_t slit = 0, slumaSum = 0, sph = 0xCBF29CE484222325ull, sdistinct = 0;
        if (!surfacePixels.empty())
        {
            seenBits.assign(1u << 18, 0);
            for (size_t i = 0; i < surfacePixels.size(); i += 4)
            {
                const uint32_t r = surfacePixels[i], g = surfacePixels[i + 1],
                               b = surfacePixels[i + 2];
                const uint32_t rgb = (r << 16) | (g << 8) | b;
                if (rgb)
                    ++slit;
                slumaSum += (r * 54 + g * 183 + b * 19) >> 8;
                const uint32_t word = rgb >> 6, bit = rgb & 63;
                if (!(seenBits[word] & (1ull << bit)))
                {
                    seenBits[word] |= 1ull << bit;
                    ++sdistinct;
                }
                sph ^= rgb;
                sph *= 0x100000001B3ull;
            }
        }
        const uint64_t spixels = surfacePixels.size() / 4;
        // Milliseconds since the first measured frame — APPENDED, so every column index
        // any existing tool reads is unchanged.
        //
        // It exists because the frame rate of an ERA is not a number this project could
        // previously state. Every frame-rate figure it owns divides a whole run's frame
        // count by its wall time, which for a run that boots, walks four menus, loads,
        // and only then plays is an average over eras that differ by more than the
        // effect anyone wants to measure. "8-12 fps in gameplay" was an operator's
        // stopwatch. With a timestamp per frame, any era the camera fingerprint or the
        // draw count can delimit has its own measurable rate, from a run that was
        // already being made for another reason.
        static const auto statsT0 = std::chrono::steady_clock::now();
        const long long msec = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - statsT0).count();
        fprintf(statsFile,
                "%llu %llu %llu %016llx %016llx %u %u %.4f %.3f %llu %016llx "
                "%u %u %.4f %.3f %llu %016llx %lld\n",
                (unsigned long long)pres.frame, (unsigned long long)pres.draws,
                (unsigned long long)pres.vertices,
                (unsigned long long)pres.drawFingerprint,
                (unsigned long long)pres.cameraFingerprint, width1, height1,
                pixels ? 100.0 * double(lit) / double(pixels) : 0.0,
                pixels ? double(lumaSum) / double(pixels) : 0.0,
                (unsigned long long)distinct, (unsigned long long)ph,
                surfaceW, surfaceH,
                spixels ? 100.0 * double(slit) / double(spixels) : 0.0,
                spixels ? double(slumaSum) / double(spixels) : 0.0,
                (unsigned long long)sdistinct,
                spixels ? (unsigned long long)sph : 0ull, msec);
        fflush(statsFile);
    }
    // The fingerprint and vertex accumulators are reset at the SUBMIT above, not here:
    // they belong to the frame that was just recorded, and everything from the present
    // onwards is about a different one.

    // A frame that is entirely one colour is the single most common wrong result a
    // renderer produces, and it is invisible in a log. Counting it makes "the picture
    // is black" a number rather than a report — and separating "black" from "some
    // uniform colour" separates a missing draw from a clear that ran and nothing else.
    if (px)
    {
        uint32_t first = 0;
        memcpy(&first, px, 4);
        bool uniform = true;
        for (size_t i = 4; i < bytes && uniform; i += 4)
            uniform = memcmp(&px[i], &first, 4) == 0;
        if (uniform)
            Count(first == 0xFF000000u || first == 0 ? "frame: uniformly black"
                                                     : "frame: uniformly one colour");
        else
            Count("frame: has content");
    }

    // CZ_VK_SNAP_DUMP=<dir> — write EVERY resolve snapshot of one frame as a PPM.
    //
    // The question this answers is "where in the chain did the picture go?", and it is
    // the only instrument that can: the frame is the last link, so a wrong frame is
    // consistent with every pass being wrong and with exactly one being wrong. Dumping
    // all of them turns that into a directory you can look at.
    // CZ_VK_SNAP_FRAME=N picks the frame. It was a hardcoded 600 for as long as the
    // instrument existed, which was fine while every question was about the title
    // screen — and useless the moment one was not. Phase C part 12's defect is on a
    // menu two presses past the title, i.e. at whatever frame the synthetic-input arm
    // happens to land on, and a dependency graph of the wrong frame answers nothing.
    // CZ_CAPTURE_KEY=<dir> — ONE PRESS, ONE PLACE, EVERY ARTIFACT, into one directory.
    //
    // The three instruments that answer "what does this surface look like and why" were
    // three environment variables writing to three places, and two of them fire on a
    // frame NUMBER rather than on the operator. Standing in front of a defect with a
    // zombie chewing on you is not the moment to be reading a frame counter out of the
    // title bar, and the parts of this project that need an operator are the parts that
    // have to cost them the least (gotcha 190). So this sets all three at once and names
    // every file after the same frame:
    //
    //   capture_<frame>.ppm        the presented picture
    //   capture_<frame>.census     every draw: shaders, fetch slots, addresses, DIMENSION
    //   snap_*.ppm                 every resolve snapshot of that frame
    //
    // The census columns are deliberately the same fields `tools/xtr_draw_bindings.py`
    // prints for a Xenia `.xtr`, so ours and hardware's can be read side by side — which
    // is the whole point, and it is what part 27 had to do by hand.
    static const char* captureDir = Env("CZ_CAPTURE_KEY");
    static const bool captureDirReady = [] {
        if (const char* d = Env("CZ_CAPTURE_KEY"))
        {
            std::error_code ec;
            std::filesystem::create_directories(d, ec);
            fprintf(stderr, "[vk] CZ_CAPTURE_KEY armed: press F9 to capture the picture, "
                            "the per-draw census and every resolve snapshot of one frame "
                            "into %s\n", d);
        }
        return true;
    }();
    (void)captureDirReady;
    static const char* snapDir = captureDir ? captureDir : Env("CZ_VK_SNAP_DUMP");
    static const uint64_t snapFrame =
        Env("CZ_VK_SNAP_FRAME") ? strtoull(Env("CZ_VK_SNAP_FRAME"), nullptr, 10) : 600;
    // F9 — the operator's own trigger, consumed here. Asked for from inside the game,
    // standing on a defect, waiting for the frame counter in the title bar to reach a
    // number chosen before the run started: a fixed `CZ_VK_SNAP_FRAME` is a fine trigger
    // for a boot-time question and the wrong one for any question about a PLACE. The edge
    // is consumed unconditionally so a press cannot sit latched and fire on some later
    // frame, and a press with no destination SAYS SO rather than doing nothing visible —
    // an instrument that silently declines is the failure shape this project keeps paying
    // for (gotchas 7, 151).
    const bool snapKey = Host_ConsumeSnapDumpPressed();
    bool snapKeyNow = snapKey;   // see the CZ_CAPTURE_KEY note at the dump
    // The SAME press also arms the per-draw census for the NEXT frame — next, not this
    // one, because this frame's draws are already recorded by the time a present is
    // reached. One press therefore yields two views of one place: every surface in the
    // frame (the snapshots) and every draw that built it.
    static std::string captureCensus =
        captureDir ? std::string(captureDir) + "/capture.census" : std::string();
    static const char* censusPath =
        captureDir ? captureCensus.c_str() : Env("CZ_VK_DRAW_CENSUS");
    if (snapKey && censusPath && !R->drawCensusFrame)
    {
        R->drawCensusFrame = R->frame + 1;
        // The PICTURE of the same frame, which is the artifact the other two exist to
        // explain and the only one that was still on a fixed interval. Armed for the
        // next frame for the same reason the census is: this frame's draws are already
        // recorded by the time a present is reached, so a picture taken now and a census
        // taken next frame would be two different moments described as one.
        R->capturePictureFrame = R->frame + 1;
        // ARM THE PRESENT READBACK for the frames this capture needs (part 76 item 1).
        // In the swapchain arm the readback is off by default now, so without this the
        // capture would find `px` null and write a census with no picture beside it.
        // `framesInFlight + 2` rather than exactly one frame: the picture is consumed
        // when the armed frame RETIRES, which is one to two swaps later depending on the
        // ring depth, and a few extra whole-frame copies once per key press is not a
        // quantity worth being exact about. It is a max() so two presses in quick
        // succession cannot shorten each other.
        R->readbackUntilFrame =
            std::max(R->readbackUntilFrame, R->frame + R->framesInFlight + 2);
        ++R->readbackArmedPresses;
        // CZ_VK_DRAW_ID=1 — THE CENSUS FRAME ITSELF paints draw indices instead of
        // colours, and it must be the same frame: a draw index is only meaningful
        // against the draw list it was numbered in. The first version armed the NEXT
        // frame so that one press could yield both a picture and a map, and the very
        // first read showed why that is wrong — the top "visible" draws resolved to
        // census lines with `mask=0`, draws that write no colour at all, because index
        // 254 of one frame is not index 254 of the next. One frame, one numbering.
        //
        // The cost is that this press yields no usable PICTURE (the post chain is made
        // of draws too, so it paints its own indices over everything). That is said out
        // loud below rather than left for someone to discover in the file.
        if (EnvOn("CZ_VK_DRAW_ID") && R->drawIdModule)
        {
            R->drawIdArmed = true;
            fprintf(stderr, "[vk] F9: the next recorded frame will be a DRAW-ID map "
                            "(read it with tools/drawid_read.py)\n");
        }
        fprintf(stderr, "[vk] F9: capturing frame %llu -> picture, %llu-draw census and "
                        "every resolve snapshot\n",
                (unsigned long long)R->drawCensusFrame,
                (unsigned long long)R->drawsThisFrame);
    }
    if (snapKey && !snapDir && !censusPath)
    {
        static bool complained = false;
        if (!complained)
        {
            complained = true;
            fprintf(stderr, "[vk] F9 pressed but neither CZ_VK_SNAP_DUMP nor "
                            "CZ_VK_DRAW_CENSUS is set — nothing was dumped\n");
        }
    }
    // Close the census at the END of the frame it covered, and say how many draws it saw
    // — a census whose file exists but is short is otherwise indistinguishable from one
    // that ran on a frame with nothing in it.
    if (R->drawCensusFrame && R->frame > R->drawCensusFrame && R->drawCensusFile)
    {
        fclose(R->drawCensusFile);
        R->drawCensusFile = nullptr;
        // The count is the census's OWN line counter, not `drawsThisFrame` — that has
        // already been reset by the frame boundary we are standing on, and printing it
        // here would report zero for a census that worked perfectly.
        fprintf(stderr, "[vk] draw census written: %llu draws of frame %llu\n",
                (unsigned long long)R->drawCensusLines,
                (unsigned long long)R->drawCensusFrame);
        R->drawCensusLines = 0;
        R->drawCensusFrame = 0;
    }
    // ONE PRESS MUST MEAN ONE FRAME. The snapshot dump used to fire on the press itself,
    // i.e. on the frame BEFORE the one the census and the picture cover — so a capture
    // produced three artifacts labelled as one place and describing two consecutive
    // frames. In a crowd those are two different pictures, and the whole value of the
    // capture is that its three views are of the same moment. So under CZ_CAPTURE_KEY the
    // press arms the dump for the next frame like the other two; the standalone
    // CZ_VK_SNAP_DUMP path keeps its old immediate behaviour, which several recorded
    // measurements were taken with.
    if (captureDir && snapKey)
    {
        R->captureSnapFrame = R->frame + 1;
        snapKeyNow = false;
    }
    if (R->captureSnapFrame && R->frame == R->captureSnapFrame)
    {
        R->captureSnapFrame = 0;
        snapKeyNow = true;
    }
    // And the fixed-frame dump is OFF under CZ_CAPTURE_KEY: that variable means "the
    // operator decides when", and 130 files from frame 600 in the capture directory is
    // noise the operator then has to tell apart from their own press.
    // THE DRAW-ID FRAME DUMPS ITS SNAPSHOTS TOO, and it must: the ID image only exists
    // in the SCENE COLOUR, before the post chain. The presented picture cannot carry it,
    // because the post passes are draws as well and would paint their own indices over
    // the whole screen — so the map has to be read off the resolve, and this is the
    // dump that writes resolves out.
    const bool drawIdNow = R->drawIdActive;
    if (drawIdNow)
    {
        R->drawIdArmed = false;
        R->drawIdActive = false;
        R->drawIdRanOnFrame = R->frame;
    }
    if (snapDir && ((R->frame == snapFrame && !captureDir) || blackTransition ||
                    snapKeyNow || drawIdNow))
    {
        if (drawIdNow)
            fprintf(stderr, "[vk] DRAW-ID: frame %llu rendered %llu draws as indices; its "
                            "resolve snapshots ARE the ID map\n",
                    (unsigned long long)R->frame, (unsigned long long)R->drawsThisFrame);
        if (snapKeyNow)
            fprintf(stderr, "[vk] F9: dumping every resolve snapshot of frame %llu\n",
                    (unsigned long long)R->frame);
        // CREATE THE DIRECTORY, and say so if the first file still cannot be written.
        // Without this the dump announces "dumping every resolve snapshot of frame N",
        // iterates every surface, and writes NOTHING when the directory is absent — which
        // is what happened the first time it was pointed at a fresh path, and it looks
        // exactly like a renderer that had no snapshots to give. `CZ_SHADER_DUMP` had this
        // same defect and part 25 fixed it there; the fix belongs at every dump site.
        {
            std::error_code ec;
            std::filesystem::create_directories(snapDir, ec);
        }
        bool wroteOne = false;
        for (const auto& [dest, snapBinding] : R->snapshots)
        {
            // A plain reference for the lambda below: capturing a structured binding is
            // C++20 (P1091) but clang 15 — the release's old-base compiler — rejects it,
            // and this was the one line in the tree that did so (part 104).
            const auto& snap = snapBinding;
            const size_t n = size_t(snap.image.width) * snap.image.height * 4;
            if (n > R->readback.size)
                continue;
            const VkImageAspectFlags aspect = snap.fromDepth
                                                  ? VK_IMAGE_ASPECT_DEPTH_BIT
                                                  : VK_IMAGE_ASPECT_COLOR_BIT;
            RunImmediate([&](VkCommandBuffer cb) {
                Image& img = const_cast<Image&>(snap.image);
                Barrier(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, aspect);
                VkBufferImageCopy c{};
                c.imageSubresource = { aspect, 0, 0, 1 };
                c.imageExtent = { img.width, img.height, 1 };
                vkCmdCopyImageToBuffer(cb, img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       R->readback.buffer, 1, &c);
                Barrier(cb, img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, aspect);
            });
            // The FRAME is in the name because CZ_VK_SNAP_ON_BLACK can fire on more
            // than one frame, and a chain that silently overwrote the transition frame
            // with the one after it would destroy the only frame worth having.
            char path[512];
            // The ID frame's files are named apart so a directory of snapshots cannot
            // be misread: an ID map looks like a garish colour noise image, and mistaking
            // one for a picture is exactly the sort of confusion this instrument exists
            // to end.
            snprintf(path, sizeof path, "%s/%sf%06llu_snap_%08X_%ux%u%s.ppm", snapDir,
                     drawIdNow ? "drawid_" : "",
                     (unsigned long long)R->frame, dest & 0x1FFFFFFF, snap.image.width,
                     snap.image.height, snap.fromDepth ? "_depth" : "");
            FILE* f = fopen(path, "wb");
            if (!f)
            {
                static bool complained = false;
                if (!complained)
                {
                    complained = true;
                    fprintf(stderr, "[vk] CZ_VK_SNAP_DUMP cannot write %s — NO snapshots "
                                    "will be dumped this run\n", path);
                }
            }
            if (f)
            {
                wroteOne = true;
                fprintf(f, "P6\n%u %u\n255\n", snap.image.width, snap.image.height);
                if (snap.fromDepth)
                {
                    // The depth aspect comes back one 32-bit word per texel. For
                    // D24_UNORM_S8_UINT the depth is the low 24 bits (a 0..2^24-1
                    // integer); for D32_SFLOAT_S8_UINT (the AMD path) it is a 32-bit
                    // float in 0..1. Read both as a normalised 0..1 double so the rest
                    // of the dump is format-agnostic. A perspective depth buffer's
                    // values all sit within a hair of 1.0, so a linear grey would be a
                    // white rectangle whatever it contained — the image is therefore
                    // stretched between the surface's OWN min and max, and the filename
                    // says `_depth` so nobody reads it as a colour surface.
                    const bool isFloat = R->depth.format == VK_FORMAT_D32_SFLOAT_S8_UINT;
                    auto readNorm = [&](size_t i) -> double {
                        if (isFloat)
                        {
                            float fv;
                            memcpy(&fv, R->readback.mapped + i, 4);
                            return double(fv);
                        }
                        uint32_t v;
                        memcpy(&v, R->readback.mapped + i, 4);
                        return double(v & 0xFFFFFFu) / 16777215.0;
                    };
                    double lo = 1e30, hi = -1e30;
                    for (size_t i = 0; i < n; i += 4)
                    {
                        const double d = readNorm(i);
                        lo = std::min(lo, d);
                        hi = std::max(hi, d);
                    }
                    const double span = hi > lo ? (hi - lo) : 1.0;
                    for (size_t i = 0; i < n; i += 4)
                    {
                        const uint8_t g = uint8_t(255.0 * (readNorm(i) - lo) / span);
                        const uint8_t rgb[3] = { g, g, g };
                        fwrite(rgb, 1, 3, f);
                    }
                    fprintf(stderr, "[vk]   %08X is a DEPTH snapshot (%s), range "
                                    "%.6f..%.6f\n",
                            dest & 0x1FFFFFFF, isFloat ? "D32F" : "D24", lo, hi);
                }
                else
                {
                    for (size_t i = 0; i < n; i += 4)
                        fwrite(R->readback.mapped + i, 1, 3, f);
                }
                fclose(f);
            }
        }
        fprintf(stderr, "[vk] dumped %zu resolve snapshots to %s%s\n",
                R->snapshots.size(), snapDir,
                wroteOne ? "" : "  — NONE OF THEM WERE WRITTEN");
    }

    static const uint64_t statsEvery =
        Env("CZ_VK_STATS") ? std::max(1L, strtol(Env("CZ_VK_STATS"), nullptr, 10)) : 0;
    if (statsEvery && (R->frame % statsEvery) == 0)
        VkRenderer_DumpStats();

    // CZ_VK_PROFILE=N — the frame's CPU time by phase, every N seconds.
    //
    // On a CLOCK rather than a frame count, and the reason is this project's own
    // history: a report every N frames samples a different amount of wall time in every
    // era, so the boot's fast frames and gameplay's slow ones would be averaged by
    // whatever the interval happened to buy (gotcha 186). Reporting per second makes
    // the fps column mean the same thing everywhere, which is the entire point of
    // having it.
    if (g_profileOn)
    {
        static const auto t0 = std::chrono::steady_clock::now();
        static auto last = t0;
        static uint64_t lastFrame = 0;
        static double period =
            Env("CZ_VK_PROFILE") ? std::max(1.0, atof(Env("CZ_VK_PROFILE"))) : 5.0;
        const auto now = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(now - last).count();
        if (dt >= period)
        {
            const uint64_t frames = R->frame - lastFrame;
            const double ms = dt * 1000.0;
            // Every phase as a percentage of the WALL time of the window, so the
            // columns are comparable and their shortfall from 100% is the frame time
            // this instrument does not yet account for — which is a number worth
            // seeing rather than hiding.
            auto pct = [&](uint64_t ns) { return 100.0 * (double(ns) * 1e-6) / ms; };
            const double perFrame = frames ? ms / double(frames) : 0.0;
            // Every phase below is EXCLUSIVE of every other (see ProfScope), so the
            // totals are sums rather than subtractions. That is the whole difference:
            // a subtraction hides an error in one term inside another term's residual,
            // where a sum leaves it visible in `outside`.
            //
            // `other` is DoDraw's own untimed work — the register decode, the
            // pipeline-key build and its lookup, the fetch-constant walk, and the
            // always-on censuses. `outside` is everything that is not the renderer at
            // all: the guest's simulation, the command processor, and any wait between
            // frames.
            // `record` is now a RESIDUAL and the three sub-phases sit beside it, so
            // the draw total has to include them or the `outside` column absorbs the
            // difference and reads as a regression nobody made. Exactly the arithmetic
            // that part 20 got wrong in the other direction (see ProfScope).
            const uint64_t recordTotal = g_prof.record + g_prof.recordState +
                                         g_prof.recordVertex + g_prof.recordIndex +
                                         g_prof.streamGuard;
            // `other` is now a residual in exactly the same way, for the same reason.
            const uint64_t otherTotal = g_prof.drawOther + g_prof.otherKey +
                                        g_prof.otherPipeline + g_prof.otherFetch +
                                        g_prof.otherShader + g_prof.otherBegin +
                                        g_prof.otherTail;
            // `constants` is a residual now too (part 75's split).
            const uint64_t constTotal = g_prof.constants + g_prof.constVs +
                                        g_prof.constVsCopy + g_prof.constVsPatch +
                                        g_prof.constPs + g_prof.constShared;
            const uint64_t drawTotal = constTotal + g_prof.streams +
                                       g_prof.textures + recordTotal + otherTotal;
            const uint64_t submitTotal =
                g_prof.submit + g_prof.submitCall + g_prof.fenceWait;
            const uint64_t known = drawTotal + submitTotal + g_prof.readback +
                                   g_prof.frameStats + g_prof.rt;
            fprintf(stderr,
                    "[vkprof] %.1f fps (%.1f ms/frame, %llu draws/frame) | draw %.1f%% "
                    "[constants %.1f (vs %.1f [copy %.1f patch %.1f] ps %.1f "
                    "shared %.1f) streams %.1f textures %.1f record %.1f other "
                    "%.1f] submit %.1f%% [call %.1f gpu %.1f] readback %.1f%% "
                    "rt %.1f%% outside %.1f%%\n",
                    frames / dt, perFrame,
                    (unsigned long long)(frames ? g_prof.draws / frames : 0),
                    pct(drawTotal), pct(constTotal), pct(g_prof.constVs),
                    pct(g_prof.constVsCopy), pct(g_prof.constVsPatch),
                    pct(g_prof.constPs), pct(g_prof.constShared), pct(g_prof.streams),
                    pct(g_prof.textures), pct(recordTotal), pct(otherTotal),
                    pct(submitTotal), pct(g_prof.submitCall), pct(g_prof.fenceWait),
                    pct(g_prof.readback), pct(g_prof.rt), 100.0 - pct(known));

#if CZ_WHOLEFUNC
            // A.2 — WHAT THOSE PHASES DO NOT SAY, printed immediately under the table
            // so the gap is visible in the log rather than only in a `perf` capture
            // somebody has to think to take. INCLUSIVE of callees and scaled by the
            // sampling period; the estimator is (timed ns) x (calls / sampled), which is
            // the period exactly when the call count is a multiple of it and within one
            // call of it otherwise.
            if (g_wholeFunc)
            {
                static WholeFunc lw[3];
                WholeFunc* cur[3] = { &g_wfStream, &g_wfTexture, &g_wfDraw };
                const char* nm[3] = { "UploadStream", "UploadTexture", "DoDraw" };
                double est[3] = {}, calls[3] = {};
                uint64_t sampled = 0;
                for (int i = 0; i < 3; ++i)
                {
                    const uint64_t dn = cur[i]->ns - lw[i].ns;
                    const uint64_t ds = cur[i]->sampled - lw[i].sampled;
                    const uint64_t dc = cur[i]->calls - lw[i].calls;
                    lw[i] = *cur[i];
                    sampled += ds;
                    est[i] = ds ? double(dn) * (double(dc) / double(ds)) : 0.0;
                    calls[i] = frames ? double(dc) / double(frames) : 0.0;
                }
                // The bill, next to the numbers rather than in a footnote: two clock
                // reads per sampled call, at the same calibrated cost the scopes pay.
                const double billMs =
                    double(sampled) * 2.0 * (double(g_profNowNs10) / 10.0) * 1e-6;
                fprintf(stderr,
                        "[vkprof] WHOLE-FUNCTION (1 call in %llu, INCLUSIVE of callees "
                        "— compare with a `perf` symbol GROUP, never one symbol's self "
                        "time): %s %.2f ms/frame (%.1f%%, %.0f calls/frame) | %s %.2f "
                        "(%.1f%%, %.0f) | %s %.2f (%.1f%%, %.0f) — the table above says "
                        "streams %.1f%% textures %.1f%% draw %.1f%%; this instrument's "
                        "own bill %.2f ms over the window\n",
                        (unsigned long long)kWfPeriod,
                        nm[0], frames ? est[0] * 1e-6 / double(frames) : 0.0,
                        pct(uint64_t(est[0])), calls[0],
                        nm[1], frames ? est[1] * 1e-6 / double(frames) : 0.0,
                        pct(uint64_t(est[1])), calls[1],
                        nm[2], frames ? est[2] * 1e-6 / double(frames) : 0.0,
                        pct(uint64_t(est[2])), calls[2],
                        pct(g_prof.streams), pct(g_prof.textures), pct(drawTotal),
                        billMs);
            }

#endif // CZ_WHOLEFUNC
            // THE INSTRUMENT'S OWN BILL, on its own line so it can never be read as
            // part of the game's frame. It is charged to the run that asked for it and
            // to nothing else, and it is ZERO in a run without CZ_VK_FRAME_STATS — but
            // that is exactly the run in which no frame time is recorded, which is why
            // it had gone 33 parts unmeasured. Quote it whenever quoting a frame time
            // measured with frame stats on, i.e. always (gotcha 335's shape, one
            // instrument further out).
            if (g_prof.frameStats)
                fprintf(stderr,
                        "[vkprof]   CZ_VK_FRAME_STATS itself: %.2f ms/frame (%.1f%% of "
                        "this window) — the measured frame is this much SLOWER than the "
                        "one a player runs\n",
                        frames ? double(g_prof.frameStats) * 1e-6 / double(frames) : 0.0,
                        pct(g_prof.frameStats));

            // THE SWAPCHAIN ARM'S OWN COUNTER (CZ_VK_SWAPCHAIN=1). An arm with no
            // counter cannot be shown to have engaged (gotcha 151), and this one has a
            // specific way of half-engaging that would otherwise look like a frame-rate
            // regression with no cause: a swapchain that cannot acquire DROPS the frame,
            // so `presents` below the window's frame count is the whole explanation for
            // a picture that stutters while every other column reads normal.
            if (R->wantSwapchain)
            {
                static uint64_t lastPresents = 0, lastFails = 0;
                const uint64_t dp = R->swap.presents - lastPresents;
                const uint64_t df = R->swap.acquireFails - lastFails;
                lastPresents = R->swap.presents;
                lastFails = R->swap.acquireFails;
                fprintf(stderr,
                        "[vkprof] swapchain %ux%u %s: %llu presented this window of %llu "
                        "frames, %llu DROPPED (acquire/present failed), %llu rebuilds, "
                        "%llu suboptimal\n",
                        R->swap.width, R->swap.height,
                        R->swap.mode == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX" : "FIFO",
                        (unsigned long long)dp, (unsigned long long)frames,
                        (unsigned long long)df,
                        (unsigned long long)R->swap.rebuilds,
                        (unsigned long long)R->swap.suboptimal);
            }

            // THE SPLIT OF `record`, which is the largest draw-path term on the
            // operator's frame (15.2 ms, 2.17 us a draw) and had no breakdown at all.
            // ns-per-draw as well as a share, because the share moves when any other
            // phase does (gotcha 320) and the per-draw cost is what a change to this
            // code path actually moves.
            {
                const double d = frames ? double(g_prof.draws) : 0.0;
                fprintf(stderr,
                        "[vkprof] record %.1f%% = state %.1f + vertex %.1f + index %.1f "
                        "+ GUARD %.1f + residual %.1f  |  per draw: %.0f ns = %.0f + "
                        "%.0f + %.0f + %.0f + %.0f\n",
                        pct(recordTotal), pct(g_prof.recordState),
                        pct(g_prof.recordVertex), pct(g_prof.recordIndex),
                        pct(g_prof.streamGuard), pct(g_prof.record),
                        d ? double(recordTotal) / d : 0.0,
                        d ? double(g_prof.recordState) / d : 0.0,
                        d ? double(g_prof.recordVertex) / d : 0.0,
                        d ? double(g_prof.recordIndex) / d : 0.0,
                        d ? double(g_prof.streamGuard) / d : 0.0,
                        d ? double(g_prof.record) / d : 0.0);
                // THE RESOLVE SPLIT (part 89 step 0a), printed beside the record split
                // it decomposes. ns/draw here means "per SAMPLED draw", which is the
                // same population scaled 1/16, so the two lines compare directly. The
                // census's own clock bill is printed so the reader subtracts it —
                // ~one NowNs() call lands inside each measured pair (see ProfScope's
                // bill note for the arithmetic).
                if (g_resolveSplitCensus)
                {
                    static uint64_t lvn = 0, lin = 0, lvc = 0, lic = 0, lsd = 0;
                    const uint64_t dvn = g_resolveVNs - lvn, din = g_resolveINs - lin;
                    const uint64_t dvc = g_resolveVCalls - lvc,
                                   dic = g_resolveICalls - lic;
                    const uint64_t dsd = g_resolveSampledDraws - lsd;
                    lvn = g_resolveVNs; lin = g_resolveINs;
                    lvc = g_resolveVCalls; lic = g_resolveICalls;
                    lsd = g_resolveSampledDraws;
                    if (dsd)
                        fprintf(stderr,
                                "[vkprof] resolve split (1 draw in 16, %llu sampled): "
                                "vertex UploadStream %.0f ns/draw (%.2f calls/draw) + "
                                "index UploadStream %.0f ns/draw (%.2f calls/draw) = "
                                "resolve %.0f of the record ns above; clock bill inside "
                                "those ~%.0f ns/draw — subtract it\n",
                                (unsigned long long)dsd, double(dvn) / double(dsd),
                                double(dvc) / double(dsd), double(din) / double(dsd),
                                double(dic) / double(dsd),
                                double(dvn + din) / double(dsd),
                                double(dvc + dic) / double(dsd) *
                                    (double(g_profNowNs10) / 10.0));
                }
                // PARALLEL RECORD's engagement, windowed (gotcha 151: the arm and its
                // control must both be provably what they claim; the control prints
                // nothing because the counters never move).
                if (R->parRec)
                {
                    static uint64_t lc = 0, lt = 0, lw = 0, lh = 0, lo = 0;
                    static uint64_t lrec[kPrMaxRecorders] = {};
                    uint64_t chunksNow = 0;
                    char per[128];
                    size_t pn = 0;
                    for (uint32_t rIdx = 0; rIdx < kPrMaxRecorders; ++rIdx)
                    {
                        const uint64_t dr2 = g_prChunksRecorded[rIdx] - lrec[rIdx];
                        lrec[rIdx] = g_prChunksRecorded[rIdx];
                        chunksNow += dr2;
                        if (dr2 && pn < sizeof per - 16)
                            pn += size_t(snprintf(per + pn, sizeof per - pn, " %s%u:%llu",
                                                  rIdx == kPrPumpRecorder ? "pump" : "w",
                                                  rIdx, (unsigned long long)dr2));
                    }
                    const uint64_t dc2 = g_prCaptured - lc, dt2 = g_prTailDraws - lt;
                    const uint64_t dw2 = g_prWaitNs - lw, dh2 = g_prPumpHelped - lh;
                    const uint64_t do2 = g_prOverflowInline - lo;
                    lc = g_prCaptured; lt = g_prTailDraws; lw = g_prWaitNs;
                    lh = g_prPumpHelped; lo = g_prOverflowInline;
                    fprintf(stderr,
                            "[vkprof] par record: %.1f chunks/frame (%s ), tail %.0f "
                            "draws/frame, %.0f captured/frame, submit wait %.0f "
                            "us/frame (pump helped %llu), overflow-inline %llu%s\n",
                            frames ? double(chunksNow) / double(frames) : 0.0, per,
                            frames ? double(dt2) / double(frames) : 0.0,
                            frames ? double(dc2) / double(frames) : 0.0,
                            frames ? double(dw2) / 1e3 / double(frames) : 0.0,
                            (unsigned long long)dh2, (unsigned long long)do2,
                            g_prBindOverflow ? "  *** BIND OVERFLOW — captures "
                                               "truncated, picture suspect ***"
                                             : "");
                }
                // WHY `GUARD` IS INSIDE `record` AND WHAT IT MEANS FOR THE PLAN. It is
                // the stream content hash, and it was ALWAYS in this column — it just
                // had no name, because `ProfScope(streams)` wraps only the copy so a
                // cross-frame hit costs `streams` nothing. Item 1.4 is priced off
                // `record` and item 1.1 off `GuardFold`; until this line existed the same
                // milliseconds were in both prices. What is left of `record` after this
                // subtraction is the actual `vkCmd*` recording, and THAT is item 1.4's
                // real ceiling.
                // ...and THE SPLIT OF `other` (part 48 tier 3), in the same form and for
                // the same reason: 4.19 ms of the operator's frame with nothing said
                // about what is in it. Quoted per draw as well, because that is the
                // number a change to this code path moves and it is comparable between
                // their frame and ours where a share is not.
                fprintf(stderr,
                        "[vkprof] other %.0f ns/draw = shader %.0f + key %.0f + pipeline "
                        "%.0f + begin %.0f + fetch %.0f + tail %.0f + residual %.0f "
                        "(%.1f%% of frame)\n",
                        d ? double(otherTotal) / d : 0.0,
                        d ? double(g_prof.otherShader) / d : 0.0,
                        d ? double(g_prof.otherKey) / d : 0.0,
                        d ? double(g_prof.otherPipeline) / d : 0.0,
                        d ? double(g_prof.otherBegin) / d : 0.0,
                        d ? double(g_prof.otherFetch) / d : 0.0,
                        d ? double(g_prof.otherTail) / d : 0.0,
                        d ? double(g_prof.drawOther) / d : 0.0,
                        pct(otherTotal));

                // WHAT THE PROFILER ITSELF PUT IN THAT RESIDUAL. See the ProfScope
                // comment for the mechanism: a nested scope's two clock reads fall
                // OUTSIDE its own measured interval and inside its parent's, and are not
                // subtracted — and `drawOther` is the outermost per-draw scope, so all of
                // them land in the number printed as `residual` just above.
                //
                // The arithmetic, per nested scope, worked through once: the constructor's
                // read is spent between the parent's `t0` and the child's, so it is in the
                // parent's interval and is NOT in `childNs` — that is one whole read into
                // the parent's residual. The `Close()` read is inside the child's own
                // measured total, so it lands in the CHILD's named phase and is subtracted
                // from the parent. So the model is **one read per scope into the residual,
                // and a second spread across the named phases** — the profiler's total
                // bill being twice what shows up here.
                //
                // Both are printed because they answer different questions: `resid` is how
                // much of the item the plan wants split is instrumentation, and `bill` is
                // how much every ns/draw figure in this report is inflated by the act of
                // measuring it. And this is a PREDICTION, not a correction — nothing is
                // subtracted anywhere. CZ_VK_PROFILE_EXTRA_SCOPES=N adjudicates it: it
                // adds N do-nothing scopes per draw, and the residual must rise by about
                // N x the read cost. If it does not, this model is wrong.
                const double scopesPerDraw = d ? double(g_prof.scopes) / d : 0.0;
                const double readNs = double(g_profNowNs10) / 10.0;
                const double residNs = d ? double(g_prof.drawOther) / d : 0.0;
                fprintf(stderr,
                        "[vkprof] instrument: %.1f scopes/draw x %.1f ns per clock read => "
                        "~%.0f ns/draw in `other`'s %.0f ns residual (%.0f%% of it), total "
                        "profiler bill ~%.0f ns/draw\n",
                        scopesPerDraw, readNs, scopesPerDraw * readNs, residNs,
                        residNs > 0.0 ? 100.0 * scopesPerDraw * readNs / residNs : 0.0,
                        scopesPerDraw * 2.0 * readNs);
            }

            // Pipeline creation, broken out of `other`. Printed only when it happened,
            // because a line of zeroes every window would train the eye to skip it —
            // and the whole point is that this is rare and expensive rather than
            // steady. `of other` is the share it explains: if a spike in `other` is
            // compilation, that number is most of it, and if it is not, the counter
            // says so just as clearly.
            if (g_prof.pipelinesCreated)
                fprintf(stderr,
                        "[vkprof] pipelines %llu created (%.2f/frame, %.1f ms total, "
                        "%.2f ms each) = %.1f%% of frame, %.0f%% of `other`\n",
                        (unsigned long long)g_prof.pipelinesCreated,
                        frames ? double(g_prof.pipelinesCreated) / double(frames) : 0.0,
                        double(g_prof.pipelineNs) * 1e-6,
                        double(g_prof.pipelineNs) * 1e-6 /
                            double(g_prof.pipelinesCreated),
                        pct(g_prof.pipelineNs),
                        otherTotal ? 100.0 * double(g_prof.pipelineNs) /
                                         double(otherTotal) : 0.0);

            // The pipeline LOOKUP, which is a different quantity from the creation
            // above and is the one part 52 item 3.2 changed: once per draw, ~5,500 times
            // a frame, priced by the `other` split at 108-113 ns/draw before the change.
            // The front cache's hit rate is the item's own falsifier — consecutive draws
            // sharing a pipeline is an ASSUMPTION about this title's draw order, and a
            // low rate here would mean the compare is a wasted instruction per draw
            // rather than a saving. CZ_VK_NO_PIPELINE_CACHE1=1 is the control arm.
            {
                static uint64_t lastH = 0, lastM = 0;
                const uint64_t dH = g_pipeCache1Hits - lastH;
                const uint64_t dM = g_pipeCache1Misses - lastM;
                lastH = g_pipeCache1Hits;
                lastM = g_pipeCache1Misses;
                if (dH || dM)
                    fprintf(stderr,
                            "[vkprof] pipeline lookup: %.1f%% served by the one-entry "
                            "cache (%llu hit / %llu miss, %llu lookups/frame, %zu in the "
                            "table)\n",
                            100.0 * double(dH) / double(dH + dM),
                            (unsigned long long)dH, (unsigned long long)dM,
                            (unsigned long long)(frames ? (dH + dM) / frames : 0),
                            R->pipelines.size());
            }

            // ...and what `outside` actually IS. The renderer runs on the graphics
            // pump's thread, so everything the pump does between two presents is in
            // that column — including the sleep at the top of its loop, which is not
            // work and which no cycles profile can see (gpu/pump_stats.h). `ticks` is
            // the number that makes the rest readable: the ring walk stops at every
            // unsatisfied WAIT_REG_MEM and resumes on the NEXT tick, so a frame
            // costs at least one sleep period per hand-off wait in it.
            static PumpStats lastPump{};
            const PumpStats p = PumpStats_Read();
            const uint64_t dTicks = p.ticks - lastPump.ticks;
            const uint64_t walkNs = p.walkNs - lastPump.walkNs;
            // `pm4` is the command processor's OWN cost, and it needs saying because
            // the walk is where the renderer is called from: `walk` contains every
            // draw, every submit and the readback, so reading it as the command
            // processor's cost over-states that by the whole of the renderer. This is
            // the 10.98 ms term `docs/perf-cpu-plan.md` §2 is about, and until now it
            // could only be got by subtracting two lines of this report by hand.
            const uint64_t pm4Ns = walkNs > known ? walkNs - known : 0;
            const uint64_t dSleep = p.sleepNs - lastPump.sleepNs;
            fprintf(stderr,
                    "[vkprof] pump %llu ticks (%.2f/frame) | sleep %.1f%% walk %.1f%% "
                    "[pm4 %.1f] vblank-isr %.1f%% | unaccounted %.1f%%\n",
                    (unsigned long long)dTicks,
                    frames ? double(dTicks) / double(frames) : 0.0,
                    pct(dSleep), pct(walkNs), pct(pm4Ns),
                    pct(p.isrNs - lastPump.isrNs),
                    100.0 - pct(dSleep + walkNs + (p.isrNs - lastPump.isrNs)));

            // ...and how much of that sleep was ON THE CRITICAL PATH (part 51). The
            // line above has been printed since part 18 and says only that the pump was
            // off the CPU; it cannot say whether anything was waiting for it, and the
            // answer decides whether the sleep is correct behaviour or frame time.
            // gpu/pump_stats.h defines the discriminator (did the next walk advance the
            // ring cursor?) and why the millisecond figure is an UPPER BOUND — print it
            // with the word `<=` so it cannot be quoted as a saving by accident.
            const uint64_t dProg = p.progressTicks - lastPump.progressTicks;
            const uint64_t dProgSleep =
                p.sleepBeforeProgressNs - lastPump.sleepBeforeProgressNs;
            fprintf(stderr,
                    "[vkprof]   sleep on the critical path: %llu of %llu ticks made "
                    "progress (%.1f%%), sleep before them %.1f ms of %.1f ms | "
                    "<= %.2f ms/frame of latency\n",
                    (unsigned long long)dProg, (unsigned long long)dTicks,
                    dTicks ? 100.0 * double(dProg) / double(dTicks) : 0.0,
                    double(dProgSleep) * 1e-6, double(dSleep) * 1e-6,
                    frames ? double(dProgSleep) * 1e-6 / double(frames) : 0.0);

            // The two 2026-08-29 latency items, each with its engagement counter
            // (gotcha 151): eager ticks (the sleep skipped after a productive walk) and
            // mid-walk rptr publication (the guest sees ring consumption per packet).
            // Zero next to an armed default is a defect; zero under the CZ_PM4_NO_*
            // control arms is the arms working.
            static uint64_t lastEager = 0, lastMidwalk = 0, lastHeldFast = 0;
            const uint64_t dEager = p.eagerTicks - lastEager;
            const uint64_t dMidwalk = Pm4_RptrMidwalkStores() - lastMidwalk;
            const uint64_t dHeldFast = p.heldFastTicks - lastHeldFast;
            lastEager = p.eagerTicks;
            lastMidwalk += dMidwalk;
            lastHeldFast = p.heldFastTicks;
            fprintf(stderr,
                    "[vkprof]   ring latency arms: eager ticks %llu of %llu (%.1f%%) | "
                    "mid-walk rptr stores %llu (%.1f/frame) | held-fast naps %llu "
                    "(%.1f/frame)\n",
                    (unsigned long long)dEager, (unsigned long long)dTicks,
                    dTicks ? 100.0 * double(dEager) / double(dTicks) : 0.0,
                    (unsigned long long)dMidwalk,
                    frames ? double(dMidwalk) / double(frames) : 0.0,
                    (unsigned long long)dHeldFast,
                    frames ? double(dHeldFast) / double(frames) : 0.0);

            // Part 107 item 2: the Draw Thread's fence wait, parked. Every episode
            // is classified, so "the park never engaged" (all readyAtEntry / spin) and
            // "the wake predicate never fires" (parks == timeouts) are both visible
            // here rather than inferred from a frame time (gotcha 151). The first draft
            // watched the read pointer instead of the fence word and this line is
            // what said so: parks 5.6/frame, timeouts 5.6/frame, wakes 0.
            {
                static FenceWaitStats lastRw;
                const FenceWaitStats rw = FenceWait_Stats();
                const double inv = frames ? 1.0 / double(frames) : 0.0;
                fprintf(stderr,
                        "[vkprof]   fence wait (part 107): body calls %.1f/frame | "
                        "ready at entry %.1f | spin-resolved %.1f | parks %.1f (woken %.1f, "
                        "timeouts %.1f, MISSED %.1f, eagain %.1f) | contended %.1f passthrough %.1f | "
                        "executor stores seen while parked %.1f, wakes %.1f/frame%s\n",
                        double(rw.bodyCalls - lastRw.bodyCalls) * inv,
                        double(rw.readyAtEntry - lastRw.readyAtEntry) * inv,
                        double(rw.spinResolved - lastRw.spinResolved) * inv,
                        double(rw.parks - lastRw.parks) * inv,
                        double(rw.parkWoken - lastRw.parkWoken) * inv,
                        double(rw.parkTimeouts - lastRw.parkTimeouts) * inv,
                        double(rw.parkMissed - lastRw.parkMissed) * inv,
                        double(rw.parkEagain - lastRw.parkEagain) * inv,
                        double(rw.contended - lastRw.contended) * inv,
                        double(rw.passthrough - lastRw.passthrough) * inv,
                        double(rw.storeChecks - lastRw.storeChecks) * inv,
                        double(rw.wakeCalls - lastRw.wakeCalls) * inv,
                        FenceWait_Enabled() ? "" : "  [CZ_FENCE_PARK=0: spinning]");
                lastRw = rw;
            }

            // ...and the one thing `outside` has never been able to say: how much of it
            // is the pump WORKING and how much is the pump NOT RUNNING AT ALL.
            //
            // Part 52 item 4.1, and it is the same gap in a different place. Every
            // column above is a wall-clock interval measured from inside this thread,
            // and a wall-clock interval cannot tell "we spent 4 ms doing something" from
            // "we spent 4 ms descheduled waiting for somebody else". `perf-plan-part50.md`
            // §6cg made exactly that mistake in reverse — it read `outside`'s residual as
            // "guest simulation ~3 ms" when at 79% duty the pump was simply blocked — and
            // the plan's own item 4.1 asks for the split rather than another guess.
            //
            // One clock read answers it. CLOCK_THREAD_CPUTIME_ID is THIS thread's CPU
            // time, so `wall - cpu` is by definition every nanosecond the pump was off a
            // core, and the sleep counter above already accounts for the deliberate part.
            // Whatever is left is the pump blocked on somebody else: a mutex, the driver,
            // a fence, the guest.
            //
            // This costs one `clock_gettime` per REPORT (every N seconds), not per frame,
            // which is the whole reason it is safe to leave on: an instrument on the hot
            // path can cancel the effect it measures (gotcha 223), and this one is not on
            // any path at all.
            {
                static uint64_t lastCpuNs = 0;
                static bool haveCpu = false;
                timespec cts{};
                clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cts);
                const uint64_t cpuNs =
                    uint64_t(cts.tv_sec) * 1000000000ull + uint64_t(cts.tv_nsec);
                const uint64_t dCpu = haveCpu ? cpuNs - lastCpuNs : 0;
                const bool first = !haveCpu;
                lastCpuNs = cpuNs;
                haveCpu = true;
                const double wallNs = ms * 1e6;
                if (!first && wallNs > 0.0)
                {
                    const double offNs = wallNs > double(dCpu) ? wallNs - double(dCpu) : 0.0;
                    const double blockedNs =
                        offNs > double(dSleep) ? offNs - double(dSleep) : 0.0;
                    fprintf(stderr,
                            "[vkprof]   pump thread: %.1f%% on CPU | off-CPU %.2f ms/frame "
                            "= sleep %.2f + BLOCKED %.2f — the blocked part is `outside` "
                            "time that is not work and cannot be optimised away\n",
                            100.0 * double(dCpu) / wallNs,
                            frames ? offNs * 1e-6 / double(frames) : 0.0,
                            frames ? double(dSleep) * 1e-6 / double(frames) : 0.0,
                            frames ? blockedNs * 1e-6 / double(frames) : 0.0);

                    // --- A.1: THE TABLE'S OWN COVERAGE, AND WHAT COVERAGE DOES NOT
                    // MEAN (part 110) ------------------------------------------------
                    //
                    // Every column above is a percentage of WALL. Nothing above says
                    // what fraction of the PUMP'S CPU the named phases account for, or
                    // names the part that is inside no scope at all — and `outside`
                    // reads like a category ("the walk, the guest") when it is a
                    // residual. So state it: phases, the walk (which is not a
                    // ProfScope and is only ever obtained by subtraction), and the
                    // genuinely UNSCOPED remainder, as shares of this thread's CPU.
                    //
                    // AND THE WARNING IS THE POINT, because part 110 measured the
                    // coverage before writing this and it is HIGH — ~73% phases, ~28%
                    // walk, ~0% unscoped — on the same build whose `streams` column
                    // under-reports its own subsystem by a factor of thirty. **A phase
                    // table can account for 100% of a thread and still be wrong about
                    // every row**, because the defect is misattribution, not omission:
                    // `UploadStream`'s cost is charged to `record`, which is a real
                    // scope that really did contain it. Coverage is necessary and it is
                    // nowhere near sufficient, and a line that printed only the number
                    // would be the next thing to mislead somebody. The check that CAN
                    // catch it compares each phase with the SYMBOLS implementing the
                    // subsystem it is named after — `tools/phase_vs_perf.py`.
                    const double cpuMs = double(dCpu) * 1e-6;
                    const double phaseMs = double(known) * 1e-6;
                    const double walkOnlyMs = double(pm4Ns) * 1e-6;
                    const double unscopedMs =
                        cpuMs > phaseMs + walkOnlyMs ? cpuMs - phaseMs - walkOnlyMs : 0.0;
                    const auto cpct = [&](double m) {
                        return cpuMs > 0.0 ? 100.0 * m / cpuMs : 0.0;
                    };
                    fprintf(stderr,
                            "[vkprof]   COVERAGE: phases %.1f%% of the pump's CPU "
                            "(%.2f of %.2f ms/frame) + PM4 walk %.1f%% (not a phase: "
                            "`walk` minus the phases) + UNSCOPED %.1f%% — unscoped is "
                            "NOT a category, it is code this table does not measure. "
                            "AND HIGH COVERAGE IS NOT AGREEMENT: a phase names a SCOPE, "
                            "not a subsystem (gotcha 343). Run tools/phase_vs_perf.py "
                            "against a `perf` capture of this run before pricing "
                            "anything off a column above.\n",
                            cpct(phaseMs), frames ? phaseMs / double(frames) : 0.0,
                            frames ? cpuMs / double(frames) : 0.0, cpct(walkOnlyMs),
                            cpct(unscopedMs));
                }
            }

            // ...and what the walk was WALKING. `pm4` above is a number of
            // milliseconds; on its own it supports no hypothesis about what to change,
            // which is the state §2 of `docs/perf-cpu-plan.md` describes as
            // "completely uninstrumented inside". These two counts turn it into a cost
            // per packet and a cost per register-write dword, and `WriteRegister` — the
            // section's leading suspect, called once per dword of every SET_CONSTANT —
            // is testable the moment the dword rate is known.
            //
            // Read off the same window as everything above, so the arithmetic is
            // ns/packet = pm4Ns / dPackets with no cross-window mixing.
            static uint64_t lastPackets = 0, lastRegWrites = 0;
            const uint64_t packets = Pm4_PacketCount();
            const uint64_t regWrites = Pm4_RegisterWriteCount();
            const uint64_t dPackets = packets - lastPackets;
            const uint64_t dRegWrites = regWrites - lastRegWrites;
            lastPackets = packets;
            lastRegWrites = regWrites;
            // ...and how those dwords were written. A bulk share near 100% is what the
            // part-47 run copy predicts for this title (its constant banks live at
            // 0x2000 and above, nowhere near the scratch mirror); a low share would mean
            // the item is worth much less than §2.1 estimates, and only the counter can
            // say which. Differenced over the same window as everything else.
            static uint64_t lastBulk = 0, lastSlow = 0;
            const uint64_t bulk = Pm4_RegRunBulkDwords();
            const uint64_t slow = Pm4_RegRunSlowDwords();
            const uint64_t dBulk = bulk - lastBulk, dSlow = slow - lastSlow;
            lastBulk = bulk;
            lastSlow = slow;
            fprintf(stderr,
                    "[vkprof] pm4 %llu packets (%llu/frame, %.0f ns each) | %llu "
                    "register dwords (%llu/frame, %.1f/packet, %.1f%% bulk)\n",
                    (unsigned long long)dPackets,
                    (unsigned long long)(frames ? dPackets / frames : 0),
                    dPackets ? double(pm4Ns) / double(dPackets) : 0.0,
                    (unsigned long long)dRegWrites,
                    (unsigned long long)(frames ? dRegWrites / frames : 0),
                    dPackets ? double(dRegWrites) / double(dPackets) : 0.0,
                    (dBulk + dSlow) ? 100.0 * double(dBulk) / double(dBulk + dSlow)
                                    : 0.0);
            // Under CZ_PM4_VERIFY_BULK_REGS this must stay at zero. Printed only when
            // the verifier is on OR it is non-zero, so an ordinary run is not given a
            // line saying "0 mismatches" for a check it never ran -- which would be a
            // clean result from a test that could not have failed.
            if (const uint64_t mm = Pm4_RegRunMismatches())
                fprintf(stderr,
                        "[vkprof] pm4 BULK REGISTER MISMATCHES: %llu — the bulk path and "
                        "the per-dword path DISAGREE; this is a defect\n",
                        (unsigned long long)mm);

            // ...and WHICH packets those were. `Pm4_TypeCount` and `Pm4_OpcodeCount`
            // have existed since phase 4 and were incremented on every single packet,
            // and until now they were called from NOWHERE in the runtime — so "what is
            // the 16.6 ms of walk actually walking" was unanswerable while the data sat
            // in memory. That is the same gap `record` was in until part 47 split it,
            // and splitting it is what found the stream guard (gotchas 325, 326).
            //
            // It matters here specifically because the operator's packet mix DIFFERS
            // FROM THE HEADLESS ROUTE'S IN KIND, not just in size: 144 ns per packet
            // against our 110-113, and 7.8 register dwords per packet against 9.4, so
            // they submit proportionally more non-register packets and per-PACKET cost
            // dominates their walk. Nothing in the runtime described that mix; this
            // does, and `docs/perf-plan-part48.md` §3 ranks the walk items off it.
            //
            // Differenced per window like every other rate on these lines, so the
            // shares divide into the same `dPackets` the ns-per-packet above is
            // computed from. Types first, because the type split is the coarse answer
            // (type 0/1 are register writes, type 2 is ring filler, type 3 is
            // everything the command processor actually does), then every type-3
            // opcode with a non-zero delta, sorted by frequency.
            // The per-thread census's own correctness check (part 48 item 1b). Printed
            // only when the verifier is running or a mismatch exists, for the same
            // reason the bulk-register line is: an ordinary run must not be handed a
            // line saying "0 mismatches" for a check it never ran. `threads` is printed
            // alongside because the comparison is only exact while one thread walks.
            {
                uint64_t walkers = 0;
                const uint64_t bad = Pm4_CensusMismatches(&walkers);
                if (bad || getenv("CZ_PM4_VERIFY_COUNTERS"))
                    fprintf(stderr,
                            "[vkprof] pm4 census verify: %llu of 135 counters DISAGREE "
                            "(per-thread vs atomic), %llu walking thread%s\n",
                            (unsigned long long)bad, (unsigned long long)walkers,
                            walkers == 1 ? "" : "s — the comparison is NOT exact above 1");
            }
            static uint64_t lastTypes[4] = {}, lastOpcodes[128] = {};
            uint64_t dTypes[4];
            for (uint32_t t = 0; t < 4; t++)
            {
                const uint64_t c = Pm4_TypeCount(t);
                dTypes[t] = c - lastTypes[t];
                lastTypes[t] = c;
            }
            const auto packetPct = [&](uint64_t n) {
                return dPackets ? 100.0 * double(n) / double(dPackets) : 0.0;
            };
            fprintf(stderr,
                    "[vkprof] pm4 types: t0(reg-run) %.1f%% t1(reg-pair) %.1f%% "
                    "t2(filler) %.1f%% t3(command) %.1f%% | %llu t3/frame\n",
                    packetPct(dTypes[0]), packetPct(dTypes[1]), packetPct(dTypes[2]),
                    packetPct(dTypes[3]),
                    (unsigned long long)(frames ? dTypes[3] / frames : 0));

            // The filler-run census (part 50 item 1a). `t2` above says how MANY no-op
            // dwords the walk meets; this says how many CALLS they cost, which is the
            // only form in which "coalesce them" has a value. Mean run length 1.0 would
            // mean the item saves nothing at all and the histogram would say where the
            // ones are; the ring share separates driver ring padding from padding inside
            // the title's own indirect buffers, which are different producers.
            {
                static uint64_t lastRuns = 0, lastRing = 0, lastHist[8] = {};
                const uint64_t runs = Pm4_FillerRuns();
                const uint64_t ring = Pm4_FillerRingDwords();
                const uint64_t dRuns = runs - lastRuns, dRing = ring - lastRing;
                lastRuns = runs;
                lastRing = ring;
                char hist[256];
                int hn = 0;
                for (uint32_t b = 0; b < 8; b++)
                {
                    const uint64_t c = Pm4_FillerHist(b);
                    const uint64_t d = c - lastHist[b];
                    lastHist[b] = c;
                    hn += snprintf(hist + hn, sizeof(hist) - size_t(hn), "%s%llu",
                                   b ? "/" : "", (unsigned long long)d);
                }
                if (dTypes[2] || dRuns)
                    fprintf(stderr,
                            "[vkprof] pm4 filler: %llu dwords in %llu runs (mean %.1f, "
                            "%.0f%% ring) | runs by length 1/2/4/8/16/32/64/128+: %s\n",
                            (unsigned long long)dTypes[2], (unsigned long long)dRuns,
                            dRuns ? double(dTypes[2]) / double(dRuns) : 0.0,
                            dTypes[2] ? 100.0 * double(dRing) / double(dTypes[2]) : 0.0,
                            hist);
            }

            // The shader-hash memo (part 52 item 1.0). The item's claim is that the
            // ~1,919 shader loads a frame re-hash a handful of distinct shaders, so the
            // HIT RATE is the claim, not a side note: below ~90% the memo is thrashing
            // and the frame-time result will be noise. Evictions separate the two ways a
            // low rate can happen — a working set larger than the table, or a guest that
            // genuinely keeps uploading new microcode. Printed unconditionally because a
            // number nobody prints is a number nobody checks (part 51's `outside`).
            {
                static uint64_t lastHits = 0, lastMisses = 0, lastEvict = 0, lastColl = 0;
                const uint64_t hits = Pm4_ShaderMemoHits();
                const uint64_t misses = Pm4_ShaderMemoMisses();
                const uint64_t evict = Pm4_ShaderMemoEvictions();
                const uint64_t dHits = hits - lastHits, dMisses = misses - lastMisses;
                const uint64_t dEvict = evict - lastEvict;
                const uint64_t coll = Pm4_ShaderMemoCollisions();
                const uint64_t dColl = coll - lastColl;
                lastColl = coll;
                lastHits = hits;
                lastMisses = misses;
                lastEvict = evict;
                if (dHits || dMisses)
                    fprintf(stderr,
                            "[vkprof] shader memo: %.1f%% hit (%llu hit / %llu miss, "
                            "%llu evicted, %llu of the misses a collision) | "
                            "%llu loads/frame%s\n",
                            100.0 * double(dHits) / double(dHits + dMisses),
                            (unsigned long long)dHits, (unsigned long long)dMisses,
                            (unsigned long long)dEvict, (unsigned long long)dColl,
                            (unsigned long long)(frames ? (dHits + dMisses) / frames : 0),
                            Pm4_ShaderMemoMismatches()
                                ? "  *** MEMO MISMATCHES, see [pm4] above ***"
                                : "");
            }

            // THE PARALLEL GUARD'S HIT RATE (part 53 item 1.1). The plan told this part
            // to PRE-REGISTER it: below ~80% served, the item is not working and any
            // frame-time number taken from the run is noise. `drain` is the other half
            // of the story — a pool still hashing when the next frame swaps is a pool
            // that is too small, and the pump pays that wait.
            if (GuardPoolWorkers() && g_gpStats.requests)
            {
                const GuardPoolStats& s = g_gpStats;
                fprintf(stderr,
                        "[vkprof] guard prehash: %.1f%% served (%llu of %llu, %.1f "
                        "MB/frame moved off the pump) | miss: unknown %llu pending %llu "
                        "variant %llu | %u workers, %llu dispatches, %llu blocked "
                        "(%.2f ms total)%s%s\n",
                        100.0 * double(s.served) / double(s.requests),
                        (unsigned long long)s.served, (unsigned long long)s.requests,
                        frames ? double(s.bytesServed) / double(frames) / 1048576.0 : 0.0,
                        (unsigned long long)s.missUnknown,
                        (unsigned long long)s.missPending,
                        (unsigned long long)s.missVariant, GuardPoolWorkers(),
                        (unsigned long long)s.dispatches,
                        (unsigned long long)s.drainBlocked,
                        double(s.drainNs) / 1e6,
                        s.mixups ? "  *** SLOT MIX-UPS, see [vk] above ***" : "",
                        s.verifyChecked ? "" : "");
                // THE POOL'S OCCUPANCY — part 89 step 0c. The dispatch/blocked counts
                // above say the pool KEEPS UP; only this says how much worker-time is
                // left over for a record pool to share. Windowed like every rate here
                // (gotcha 428): a cumulative mean would blend the boot's empty frames
                // into the crowd's.
                if (g_gp)
                {
                    static uint64_t lastBusy = 0;
                    const uint64_t busy = g_gp->busyNs.load(std::memory_order_relaxed);
                    const uint64_t dBusy = busy - lastBusy;
                    lastBusy = busy;
                    const double poolNs = dt * 1e9 * double(GuardPoolWorkers());
                    fprintf(stderr,
                            "[vkprof] guard pool occupancy: %.2f%% busy (%.1f ms of "
                            "work across %u workers in a %.1f s window; the rest is "
                            "worker-time a shared record pool could take)\n",
                            poolNs > 0 ? 100.0 * double(dBusy) / poolNs : 0.0,
                            double(dBusy) / 1e6, GuardPoolWorkers(), dt);
                }
                if (s.verifyChecked)
                    fprintf(stderr,
                            "[vkprof] guard prehash VERIFY: %llu of %llu served guards "
                            "disagreed with an inline hash taken at the draw (%.4f%%) — "
                            "that is the widened race, not a wrong implementation; a "
                            "slot mix-up would have printed above instead\n",
                            (unsigned long long)s.verifyStale,
                            (unsigned long long)s.verifyChecked,
                            100.0 * double(s.verifyStale) / double(s.verifyChecked));
                g_gpStats = GuardPoolStats{};
            }

            // The flat stream cache. Two numbers and both are needed: PROBES PER LOOKUP
            // is the only thing that says the table is actually flat in practice rather
            // than in principle — a bad hash or a load factor left too high turns linear
            // probing into a linear scan, and it would present as "the change did
            // nothing", which is indistinguishable from a wrong theory. Above ~2.0 the
            // table is the problem. The verify line is the correctness half and prints
            // only when the arm is on.
            // ...and it prints in the CONTROL arm too, with zero lookups, because an arm
            // that is silent is an arm nobody can tell was on (gotcha 151).
            // The constant memo. HIT RATE IS THE CLAIM: the item's whole argument is
            // that the guest issues several draws per constant update, and below ~30%
            // it is not worth its risk. Printed unconditionally so a run that does not
            // behave that way says so rather than being assumed to.
            if (g_constMemoHits + g_constMemoMisses)
            {
                const uint64_t tot = g_constMemoHits + g_constMemoMisses;
                fprintf(stderr,
                        "[vkprof] const memo: %.1f%% served (%llu of %llu half-copies), "
                        "%.1f MB/frame NOT copied%s\n",
                        100.0 * double(g_constMemoHits) / double(tot),
                        (unsigned long long)g_constMemoHits, (unsigned long long)tot,
                        frames ? double(g_constMemoHits) * 4096.0 / double(frames) / 1048576.0
                               : 0.0,
                        g_constMemoOff ? " [CZ_VK_NO_CONST_MEMO: the copy runs every draw]"
                                       : "");
                fprintf(stderr,
                        "[vkprof] const memo by half: VS %.1f%%, PS %.1f%% (of %llu draws "
                        "each)\n",
                        200.0 * double(g_constMemoVsHits) / double(tot),
                        200.0 * double(g_constMemoPsHits) / double(tot),
                        (unsigned long long)(tot / 2));
                g_constMemoVsHits = g_constMemoPsHits = 0;
                if (g_constMemoChecked)
                    fprintf(stderr,
                            "[vkprof] const memo VERIFY: %llu of %llu served draws had "
                            "constants that disagreed with a fresh copy (%.4f%%)%s\n",
                            (unsigned long long)g_constMemoStale,
                            (unsigned long long)g_constMemoChecked,
                            100.0 * double(g_constMemoStale) / double(g_constMemoChecked),
                            g_constMemoVerifyPoison ? "  [POISONED: this MUST be non-zero]"
                                                    : "");
                g_constMemoHits = g_constMemoMisses = 0;
                g_constMemoChecked = g_constMemoStale = 0;
            }
            if (g_flatCacheLookups || g_flatCacheOff)
            {
                fprintf(stderr,
                        "[vkprof] flat stream cache: %llu lookups/frame, %.2f probes per "
                        "lookup%s\n",
                        (unsigned long long)(frames ? g_flatCacheLookups / frames : 0),
                        g_flatCacheLookups ? double(g_flatCacheProbes) / double(g_flatCacheLookups) : 0.0,
                        g_flatCacheOff ? " [CZ_VK_NO_FLAT_CACHE: the std::unordered_map is serving]" : "");
                if (g_flatGrows)
                    fprintf(stderr,
                            "[vkprof] flat cache grows: %llu, %.2f ms total, worst "
                            "%.2f ms — the whole RUN, not this window\n",
                            (unsigned long long)g_flatGrows, double(g_flatGrowNs) / 1e6,
                            double(g_flatGrowWorstNs) / 1e6);
                g_flatCacheLookups = 0;
                g_flatCacheProbes = 0;
            }
            if (g_flatCacheChecked)
            {
                fprintf(stderr,
                        "[vkprof] flat cache VERIFY: %llu of %llu lookups disagreed with "
                        "the std::unordered_map (%.4f%%)%s\n",
                        (unsigned long long)g_flatCacheDisagreed,
                        (unsigned long long)g_flatCacheChecked,
                        100.0 * double(g_flatCacheDisagreed) / double(g_flatCacheChecked),
                        g_flatCacheVerifyPoison ? "  [POISONED: this MUST be non-zero]" : "");
                g_flatCacheChecked = 0;
                g_flatCacheDisagreed = 0;
            }

            // Collect, sort by count descending, print. 128 slots is a fixed, tiny
            // array; B1's census says this title uses 21 opcodes, so this is at most
            // four lines and usually three.
            struct OpCensus { uint32_t op; uint64_t count; };
            OpCensus hot[128];
            uint32_t nHot = 0;
            for (uint32_t op = 0; op < 128; op++)
            {
                const uint64_t c = Pm4_OpcodeCount(op);
                const uint64_t d = c - lastOpcodes[op];
                lastOpcodes[op] = c;
                if (d)
                    hot[nHot++] = { op, d };
            }
            std::sort(hot, hot + nHot,
                      [](const OpCensus& a, const OpCensus& b) { return a.count > b.count; });
            for (uint32_t i = 0; i < nHot; i += 5)
            {
                char line[512];
                int n = snprintf(line, sizeof(line), "[vkprof] pm4 %s ",
                                 i ? "         " : "opcodes:");
                for (uint32_t j = i; j < nHot && j < i + 5; j++)
                {
                    // An unnamed opcode is reported by index, not skipped: the walk
                    // already treats one as a reportable anomaly (a parser desync or a
                    // packet the captures never held), and a census that silently
                    // dropped it would answer "which packets" with a confident subset.
                    const char* name = Pm4_OpcodeName(hot[j].op);
                    char named[24];
                    if (!name)
                    {
                        snprintf(named, sizeof(named), "UNKNOWN_%02X", hot[j].op);
                        name = named;
                    }
                    n += snprintf(line + n, sizeof(line) - n, "%s %llu/f (%.1f%%)  ",
                                  name,
                                  (unsigned long long)(frames ? hot[j].count / frames : 0),
                                  packetPct(hot[j].count));
                }
                fprintf(stderr, "%s\n", line);
            }
            lastPump = p;

            // The stream cache, when asked for. Printed inside the profile window so the
            // rates are per-frame over the SAME frames the `streams` percentage above is
            // averaged over — a census counted over the whole run and a percentage
            // counted over five seconds cannot be divided into each other. The divisor is
            // PRESENTED frames, like every other rate on these lines, not frames that
            // recorded a draw; a frame with no draws never calls BeginFrame at all.
            // The cross-frame store, ALWAYS — not behind the census, because this is the
            // line that says whether the thing is working and whether it is serving stale
            // data, and a counter nobody looks at by default is a counter that reports a
            // silent regression to nobody (gotcha 151). Rates are per PRESENTED frame,
            // like everything else on these lines.
            if (R->persistOn)
            {
                const Renderer::PersistStats& p = R->persistStats;
                const uint64_t touched =
                    p.hits + p.fills + p.stale + p.overflow + p.staleEvicted;
                fprintf(stderr,
                        "[vkprof] store %llu first-touch/frame: %.1f%% served across the "
                        "frame boundary, %.2f MB/frame NOT copied | fills %llu stale %llu "
                        "(%llu evicted, no twin) overflow %llu | guard read %.2f MB/frame\n",
                        (unsigned long long)(frames ? touched / frames : 0),
                        touched ? 100.0 * double(p.hits) / double(touched) : 0.0,
                        frames ? double(p.hitBytes) / double(frames) / 1048576.0 : 0.0,
                        (unsigned long long)(frames ? p.fills / frames : 0),
                        (unsigned long long)p.stale,
                        (unsigned long long)p.staleEvicted,
                        (unsigned long long)p.overflow,
                        frames ? double(p.guardBytes) / double(frames) / 1048576.0 : 0.0);
                // The exposure the bound leaves behind: streams too large to hash
                // exactly, which are therefore only sampled and CAN hide a small edit.
                // This is the population item 00c's defect lived in, so it is reported
                // rather than assumed to be empty.
                fprintf(stderr,
                        "[vkprof] guard exact to %zu B; %llu streams/frame exceeded it "
                        "and were SAMPLED (a small edit inside one of these is invisible "
                        "-- item 00c)\n",
                        g_guardBytes,
                        (unsigned long long)(frames ? g_guardSampled / frames : 0));
                g_guardSampled = 0;
                // ...and the SIZE distribution of that population, because a count
                // cannot choose a bound and the cost of raising one is in the bytes.
                {
                    static const char* kNames[kGuardHistBuckets] =
                        { "16-32K", "32-64K", "64-128K", "128-256K", "256-512K",
                          "512K-1M", "1-2M", ">2M" };
                    char line[512];
                    int at = snprintf(line, sizeof line,
                                      "[vkprof] sampled-stream sizes (raise the bound to "
                                      "cover a bucket and you pay its MB/frame):");
                    for (size_t b = 0; b < kGuardHistBuckets; ++b)
                    {
                        if (!g_guardHistCount[b] || at >= int(sizeof line) - 48)
                            continue;
                        at += snprintf(line + at, sizeof line - at, "  %s=%llu/%0.1fMB",
                                       kNames[b],
                                       (unsigned long long)(frames ? g_guardHistCount[b] / frames
                                                                   : g_guardHistCount[b]),
                                       frames ? double(g_guardHistBytes[b]) / double(frames)
                                                    / 1048576.0
                                              : 0.0);
                        g_guardHistCount[b] = 0;
                        g_guardHistBytes[b] = 0;
                    }
                    fprintf(stderr, "%s\n", line);
                }
                // What the dynamic-stream promotion actually costs. An adaptive policy
                // whose price is unknown is the thing that kept item 00c parked.
                fprintf(stderr,
                        "[vkprof] guard PROMOTED to exact (stream seen changing): "
                        "%llu/frame, %.1f MB/frame\n",
                        (unsigned long long)(frames ? g_guardDynamic / frames : 0),
                        frames ? double(g_guardDynamicBytes) / double(frames) / 1048576.0
                               : 0.0);
                g_guardDynamic = 0;
                g_guardDynamicBytes = 0;
                // ...split by the DOOR each promotion came through, because the three
                // have different prices and only one of them is bounded. See the
                // g_guardProven declaration for what this is asking.
                fprintf(stderr,
                        "[vkprof] guard promotion by reason: proven(unbudgeted, forever) "
                        "%llu/frame %.1f MB | speculative(budgeted) %llu/frame %.1f MB | "
                        "probe(budgeted) %llu/frame %.1f MB | %llu of %zu entries have "
                        "EVER latched proven | of proven observations %.1f%% found the "
                        "stream ACTUALLY CHANGED (above ~50%% the guard costs a read to "
                        "learn what the copy would have told us free)\n",
                        (unsigned long long)(frames ? g_guardProven / frames : 0),
                        frames ? double(g_guardProvenBytes) / double(frames) / 1048576.0 : 0.0,
                        (unsigned long long)(frames ? g_guardSpec / frames : 0),
                        frames ? double(g_guardSpecBytes) / double(frames) / 1048576.0 : 0.0,
                        (unsigned long long)(frames ? g_guardProbe / frames : 0),
                        frames ? double(g_guardProbeBytes) / double(frames) / 1048576.0 : 0.0,
                        (unsigned long long)g_guardProvenEntries, PersistSize(),
                        g_guardProvenObs ? 100.0 * double(g_guardProvenChanged) /
                                               double(g_guardProvenObs) : 0.0);
                g_guardProvenObs = g_guardProvenChanged = 0;
                g_guardProven = g_guardProvenBytes = 0;
                g_guardSpec = g_guardSpecBytes = 0;
                g_guardProbe = g_guardProbeBytes = 0;
                fprintf(stderr,
                        "[vkprof] store %zu entries, %llu MB of %llu MB used, %llu flushes"
                        " this window\n",
                        PersistSize(),
                        (unsigned long long)(R->persistCursor >> 20),
                        (unsigned long long)(R->persist.size >> 20),
                        (unsigned long long)p.flushes);
                if (R->persistDev.buffer)
                {
                    fprintf(stderr,
                            "[vkprof] store mirror: %.2f MB/frame copied host->VRAM in "
                            "%.1f copies/frame; hits bound the MIRROR %.1f%% of the time "
                            "(%llu dev, %llu host) this window\n",
                            frames ? double(R->mirrorBytes) / double(frames) / 1048576.0 : 0.0,
                            frames ? double(R->mirrorCopies) / double(frames) : 0.0,
                            (R->mirrorHitsDev + R->mirrorHitsHost)
                                ? 100.0 * double(R->mirrorHitsDev) /
                                      double(R->mirrorHitsDev + R->mirrorHitsHost)
                                : 0.0,
                            (unsigned long long)R->mirrorHitsDev,
                            (unsigned long long)R->mirrorHitsHost);
                    R->mirrorCopies = R->mirrorBytes = 0;
                    R->mirrorHitsDev = R->mirrorHitsHost = 0;
                }
                R->persistStats = Renderer::PersistStats{};
            }
            if (g_streamCensus)
            {
                const StreamCensus& s = g_streamCensus_c;
                const uint64_t n = s.hits + s.misses;
                fprintf(stderr,
                        "[vkprof] streams %llu lookups/frame: %.1f%% hit | copied "
                        "%.2f MB/frame (%llu misses/frame, %llu B each) | hits saved "
                        "%.2f MB/frame\n",
                        (unsigned long long)(frames ? n / frames : 0),
                        n ? 100.0 * double(s.hits) / double(n) : 0.0,
                        frames ? double(s.bytesCopied) / double(frames) / 1048576.0 : 0.0,
                        (unsigned long long)(frames ? s.misses / frames : 0),
                        s.misses ? (unsigned long long)(s.bytesCopied / s.misses) : 0ull,
                        frames ? double(s.bytesHit) / double(frames) / 1048576.0 : 0.0);
                // What a cache that survived the frame boundary would have done. The
                // second line only appears at level 2, because only level 2 knows whether
                // it would have been CORRECT to serve it.
                fprintf(stderr,
                        "[vkprof] streams cross-frame: %.1f%% of misses repeat last "
                        "frame's key (%.2f MB/frame)%s\n",
                        s.misses ? 100.0 * double(s.prevFrameKeyHits) / double(s.misses)
                                 : 0.0,
                        frames ? double(s.prevFrameKeyBytes) / double(frames) / 1048576.0
                               : 0.0,
                        g_streamCensus >= 2 ? "" : "  [level 2 for the content check]");
                static const char* kindName[3] = { "vertex", "index ", "vfetch" };
                for (int k = 0; k < 3; ++k)
                    fprintf(stderr,
                            "[vkprof] streams   %s: %llu misses/frame, %.2f MB/frame "
                            "copied, %.1f%% of those bytes repeat last frame\n",
                            kindName[k],
                            (unsigned long long)(frames ? s.kindMisses[k] / frames : 0),
                            frames ? double(s.kindBytes[k]) / double(frames) / 1048576.0
                                   : 0.0,
                            s.kindBytes[k] ? 100.0 * double(s.kindRepeatBytes[k]) /
                                                 double(s.kindBytes[k])
                                           : 0.0);
                if (g_streamCensus >= 2)
                    fprintf(stderr,
                            "[vkprof] streams cross-frame CONTENT UNCHANGED: %llu of "
                            "%llu repeated keys (%.1f%%), %.2f MB/frame — the rest is the "
                            "guest rewriting the buffer in place\n",
                            (unsigned long long)s.prevFrameSameContent,
                            (unsigned long long)s.prevFrameKeyHits,
                            s.prevFrameKeyHits ? 100.0 * double(s.prevFrameSameContent) /
                                                     double(s.prevFrameKeyHits)
                                               : 0.0,
                            frames ? double(s.prevFrameSameBytes) / double(frames) /
                                         1048576.0
                                   : 0.0);
                // THE GUARD'S POWER. The persistent store decides staleness with a
                // bounded-cost fingerprint (at most 512 bytes, exact below that); this
                // line is the full hash checking its work. Anything but zero is a stale
                // vertex buffer handed to a draw, and this is the ONLY thing in the
                // runtime that can see it. It reads zero when the store is off too — for
                // the trivial reason that nothing was served across a frame — so read it
                // alongside the `store` line above, never on its own.
                if (g_streamCensus >= 2)
                    fprintf(stderr,
                            "[vkprof] streams GUARD MISSED: %llu of %llu real content "
                            "changes served STALE by the cross-frame store%s\n",
                            (unsigned long long)s.guardMissed,
                            (unsigned long long)(s.prevFrameKeyHits -
                                                 s.prevFrameSameContent),
                            g_streamPoison
                                ? "  [POISON ON — the full hash calls every repeat a "
                                  "change while the guard correctly does not, so this "
                                  "SHOULD equal the repeat count]"
                                : "");
                // ...and WHICH ones, because that is what picks the invalidation
                // mechanism. Cumulative over the run, so this list is the answer to "is
                // the rewritten set a recurring few, and are they contiguous in guest
                // memory". Capped at 32 lines with the remainder NAMED rather than
                // dropped silently (gotcha 109) — and the guest address range is printed
                // whether or not the list is capped, since a range is the thing an
                // exclusion rule would be written against.
                if (g_streamCensus >= 2 && !g_streamChanged.empty())
                {
                    std::vector<std::pair<uint64_t, StreamChange>> v(
                        g_streamChanged.begin(), g_streamChanged.end());
                    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
                        return a.second.times > b.second.times;
                    });
                    uint32_t lo = 0xFFFFFFFFu, hi = 0;
                    uint64_t total = 0;
                    for (const auto& e : v)
                    {
                        const uint32_t va = uint32_t(e.first >> 32);
                        lo = std::min(lo, va);
                        hi = std::max(hi, uint32_t(va + e.second.bytes));
                        total += e.second.times;
                    }
                    fprintf(stderr,
                            "[vkprof] streams REWRITTEN IN PLACE: %zu distinct keys, %llu "
                            "occurrences, guest range %08X..%08X%s\n",
                            v.size(), (unsigned long long)total, lo, hi,
                            g_streamPoison ? "  [POISON ON — every repeat lands here by "
                                             "construction; this list is meaningless]"
                                           : "");
                    static const char* kindName2[3] = { "vertex", "index ", "vfetch" };
                    const size_t shown = std::min<size_t>(v.size(), 32);
                    for (size_t i = 0; i < shown; ++i)
                    {
                        const uint32_t va = uint32_t(v[i].first >> 32);
                        const StreamChange& c = v[i].second;
                        fprintf(stderr,
                                "[vkprof] streams   va=%08X size=%llu endian=%llu %s  "
                                "x%llu  frames %llu..%llu\n",
                                va, (unsigned long long)c.bytes,
                                (unsigned long long)(v[i].first & 3), kindName2[c.kind],
                                (unsigned long long)c.times,
                                (unsigned long long)c.firstFrame,
                                (unsigned long long)c.lastFrame);
                    }
                    if (v.size() > shown)
                        fprintf(stderr,
                                "[vkprof] streams   ...and %zu more distinct keys not "
                                "listed\n",
                                v.size() - shown);
                }
                g_streamCensus_c = StreamCensus{};
            }

            g_prof = ProfilePhases{};
            last = now;
            lastFrame = R->frame;
        }
    }

    // The snapshot is per frame: a frame whose resolve chain never reaches the front
    // buffer must not present the previous frame's picture as if it were this one.
    R->haveFrontSnapshot = false;
    (void)width;
    (void)height;
}

} // namespace

// See vk_renderer.h — the settings panel's resolution row (part 60).
void VkRenderer_RequestRenderScale(uint32_t scale)
{
    if (scale >= 1 && scale <= 4)
        g_resScalePending.store(scale, std::memory_order_release);
}

// See vk_renderer.h — the panel's APPLY press (part 91).
void VkRenderer_RequestInternalRes(uint32_t w, uint32_t h)
{
    if (w && h)
        g_resWHPending.store((uint64_t(w) << 32) | h, std::memory_order_release);
}

namespace
{

// Apply a pending live resolution change, BETWEEN frames on the pump thread — the
// only moment the scale may move, because every extent inside a frame assumes it
// is constant. The EDRAM pair is rebuilt here; every resolve snapshot and the
// rendered cube map carry the scale they were built at and rebuild themselves
// lazily through their existing resize paths; the readback buffers grow if the
// new frame no longer fits (they never shrink — memory is cheaper than another
// resize path).
void ApplyPendingRenderScale()
{
    const uint32_t want = g_resScalePending.exchange(0, std::memory_order_acq_rel);
    const uint64_t wantWH = g_resWHPending.exchange(0, std::memory_order_acq_rel);
    uint32_t wantW = uint32_t(wantWH >> 32), wantH = uint32_t(wantWH);
    if ((!want && !wantWH) || !R)
        return;
    {
        uint32_t cw, ch;
        InternalRes(cw, ch);
        if (wantWH && wantW == cw && wantH == ch)
            wantW = wantH = 0;
        const bool scaleChange = want != 0 && want != ResScale();
        if (!scaleChange && !wantW)
            return;
    }
    if (g_resScaleLocked)
    {
        fprintf(stderr, "[vk] internal-resolution change REFUSED: CZ_VK_RES/"
                        "CZ_VK_RES_SCALE pin it for this run (the "
                        "measurement arm wins over the menu)\n");
        return;
    }
    vkDeviceWaitIdle(R->device);

    auto destroyImage = [&](Image& im) {
        if (im.view)
            vkDestroyImageView(R->device, im.view, nullptr);
        if (im.image)
            vkDestroyImage(R->device, im.image, nullptr);
        if (im.memory)
            vkFreeMemory(R->device, im.memory, nullptr);
        im = Image{};
    };
    const uint32_t before = ResScale();
    // Deferred clears latched against the OLD images' extents; the content they were
    // scoped to is being destroyed with the images.
    R->pendingClears.clear();
    destroyImage(R->color);
    destroyImage(R->depth);
    destroyImage(R->colorResolve);
    destroyImage(R->depthResolve);
    // The explicit W x H form (part 91: the panel's APPLY press) wins; the legacy
    // integer-scale form preserves the current aspect at 720*scale.
    if (wantW)
    {
        g_internalW.store(wantW & ~1u, std::memory_order_relaxed);
        g_internalH.store(wantH, std::memory_order_relaxed);
    }
    else
    {
        uint32_t w, h;
        InternalRes(w, h);
        const uint32_t nh = 720 * want;
        const uint32_t nw = (uint32_t(uint64_t(w) * nh / h)) & ~1u;
        g_internalW.store(nw, std::memory_order_relaxed);
        g_internalH.store(nh, std::memory_order_relaxed);
    }

    const uint32_t edramH = R->edramHeight;   // guest rows, decided at bring-up
    if (!CreateImage(R->color, RSX(R->targetWidth), RS(edramH),
                     VK_FORMAT_R8G8B8A8_UNORM,
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                     VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_VIEW_TYPE_2D, 1, 1,
                     VkComponentMapping{}, 1, false, R->msaaSamples) ||
        !CreateImage(R->depth, RSX(R->targetWidth), RS(edramH),
                     EdramDepthFormat(),
                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                     VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                     VK_IMAGE_VIEW_TYPE_2D, 1, 1, VkComponentMapping{}, 1, false,
                     R->msaaSamples) ||
        !CreateEdramResolveTargets(RSX(R->targetWidth), RS(edramH)))
    {
        // A renderer with no EDRAM stand-in cannot draw at all — say so and stop
        // feeding it rather than crash on the first pass.
        fprintf(stderr, "[vk] LIVE RESCALE FAILED at %ux — renderer disabled\n", want);
        g_active = false;
        return;
    }
    NameImage(R->color, "EDRAM colour %ux%u", RSX(R->targetWidth), RS(edramH));
    NameImage(R->depth, "EDRAM depth %ux%u", RSX(R->targetWidth), RS(edramH));

    // Flush EVERY snapshot and rendered cube NOW, inside the one device idle this
    // switch already paid for. The first version left them to the lazy per-entry
    // rebuild path — which calls vkDeviceWaitIdle PER SNAPSHOT, and a street frame
    // holds dozens of live snapshots, so the seconds after a resolution change
    // were a stutter festival the operator read as a freeze. Erased entries
    // recreate fresh on their next resolve with no further stalls.
    size_t flushed = 0;
    for (auto& [key, snap] : R->snapshots)
    {
        vkDestroyImageView(R->device, snap.image.view, nullptr);
        vkDestroyImage(R->device, snap.image.image, nullptr);
        vkFreeMemory(R->device, snap.image.memory, nullptr);
        for (auto& [size, view] : snap.views)
        {
            (void)size;
            vkDestroyImageView(R->device, view.image.view, nullptr);
            vkDestroyImage(R->device, view.image.image, nullptr);
            vkFreeMemory(R->device, view.image.memory, nullptr);
        }
        ++flushed;
    }
    R->snapshots.clear();
    TexGenBump();
    for (auto& [key, cube] : R->cubeSnapshots)
    {
        vkDestroyImageView(R->device, cube.image.view, nullptr);
        vkDestroyImage(R->device, cube.image.image, nullptr);
        vkFreeMemory(R->device, cube.image.memory, nullptr);
        ++flushed;
    }
    R->cubeSnapshots.clear();
    if (flushed)
        fprintf(stderr, "[vk] live rescale flushed %zu snapshot surfaces in one "
                        "stall\n", flushed);

    const uint64_t need =
        std::max(uint64_t(RSX(R->targetWidth)) * RS(R->targetHeight),
                 uint64_t(RSX(4096)) * RS(1024)) * 4;
    auto growBuffer = [&](Buffer& b, const char* what) {
        if (b.size >= need)
            return;
        vkDestroyBuffer(R->device, b.buffer, nullptr);
        vkFreeMemory(R->device, b.memory, nullptr);
        b = Buffer{};
        if (!CreateBuffer(b, need, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          ReadbackMemoryProps(), false))
            fprintf(stderr, "[vk] LIVE RESCALE: %s regrow FAILED — readback-side "
                            "features will truncate at this scale\n", what);
    };
    growBuffer(R->readback, "snapshot readback");
    for (uint32_t i = 0; i < R->framesInFlight; ++i)
        growBuffer(R->frames[i].present, "present readback");
    R->presentPixels.resize(size_t(RSX(R->targetWidth)) * RS(R->targetHeight) * 4);

    (void)before;
    {
        uint32_t nw, nh;
        InternalRes(nw, nh);
        fprintf(stderr, "[vk] internal resolution -> %ux%u LIVE (EDRAM stand-in "
                        "%ux%u); snapshots and the cube map rebuild lazily over the "
                        "next frames\n",
                nw, nh, RSX(R->targetWidth), RS(R->edramHeight));
        // A windowed window follows the applied resolution (part 108). Handed to the
        // window thread, which owns every SDL call; it declines (and says so) when the
        // window is fullscreen, maximised or pinned by a measurement variable.
        Host_WindowFollowInternalRes(nw, nh);
    }
}

} // namespace

void VkRenderer_OnSwap(uint8_t* base, uint32_t frontBuffer, uint32_t width,
                       uint32_t height)
{
    if (!g_active || g_d3dMode)
        return;
    // NOTE (part 91): the pending internal-resolution change is applied at the TOP of
    // BeginFrame, not here. Here sits MID-FRAME — the frame's draws are recorded but
    // not submitted, and destroying the EDRAM images a recorded-but-unsubmitted
    // command buffer references is the exact part-60 freeze shape ("a wait-idle only
    // covers SUBMITTED work", the RetiredImage comment). This mid-frame placement is
    // the likely reason the live path froze the operator's machine twice and spent
    // 31 parts parked.
    DoSwapImpl(base, frontBuffer, width, height);
}

// --- the phase C feed --------------------------------------------------------------
bool VkRenderer_D3DInit()
{
    static bool tried = false, ok = false;
    if (tried)
        return ok;
    tried = true;
    if (g_active)
    {
        // The PM4 feed initialized first (CZ_VKDRAW). d3d_draw.cpp refuses the
        // combination before calling here, so reaching this is a wiring bug.
        fprintf(stderr, "[vk] D3D feed refused: the PM4 feed already owns the renderer\n");
        return false;
    }
    if (!InitCommon())
        return false;
    g_d3dMode = true;
    ok = true;
    fprintf(stderr, "[vk] renderer feed: D3D draw service (phase C)\n");
    return true;
}

void VkRenderer_D3DDraw(uint8_t* base, const Pm4Draw& draw, const uint32_t* regs,
                        const Pm4ShaderBinding& vs, const Pm4ShaderBinding& ps)
{
    if (!g_active || !g_d3dMode)
        return;
    // The same resolve discriminator as the PM4 feed, over the PRIVATE register
    // file: the copy-mode SET_CONSTANTs the Resolve body emits land there.
    if ((regs[0x2208] & 7) == 6)
    {
        DoResolve(base, regs);
        return;
    }
    DoDraw(base, draw, regs, vs, ps);
}

void VkRenderer_D3DSwap(uint8_t* base)
{
    if (!g_active || !g_d3dMode)
        return;
    // The front buffer is the destination of the resolve the title just performed
    // (PreSwapResolve immediately precedes every Swap), so no side channel names it.
    DoSwapImpl(base, R->lastResolveDest, R->targetWidth, R->targetHeight);
}

// See vk_renderer.h — the live-VSync seam (part 60).
void VkRenderer_RequestSwapchainRebuild()
{
    g_swapRebuildRequest.store(true, std::memory_order_release);
}

// The factor the GAME's roaming camera must be widened by, in tan space, so that its
// own 16:9 culling frustum covers the rendered view: k in wide mode (the horizontal
// grows by k), 1/k in narrow mode (the vertical grows by 1/k; part 108), 1 at 16:9.
// The EDRAM sample count THIS run is using (1, 2 or 4), for the panel's "applies at
// next launch" star: the setting can differ from it until a relaunch. 0 before the
// renderer exists.
int VkRenderer_MsaaSamples()
{
    return R ? int(R->msaaSamples) : 0;
}

float VkRenderer_WideFovFactor()
{
    if (WideMode())
        return WideFovFactor();
    if (NarrowMode())
        return 1.0f / WideFovFactor();
    return 1.0f;
}

void VkRenderer_SavePipelineCache()
{
    if (!g_active)
        return;
    SavePipelineCache();
    SavePipelineKeys();
}

void VkRenderer_DumpStats()
{
    if (!g_active)
        return;
    fprintf(stderr, "[vk] --- renderer stats (frame %llu) ---\n",
            (unsigned long long)R->frame);
    // THE MEMO'S OWN REPORT, and it lives HERE because its first home could not fire.
    // Part 109 put this line inside the LIVE RESCALE path — a function a headless crowd
    // run never calls — so the verifier arm ran a whole route and printed nothing, and
    // "0 disagreements" and "the instrument never spoke" were the same output. Every
    // recipe in this project ends on a `timeout` SIGTERM, and this function is what that
    // handler calls; a counter reported anywhere else is a counter nobody reads.
    Pm4_RegRunCensusReport();
    if (g_texMemoHits || g_texMemoMiss)
        fprintf(stderr, "[texmemo] %llu hits, %llu misses (%.1f%% served), %llu "
                        "disagreements, final gen %llu\n",
                (unsigned long long)g_texMemoHits, (unsigned long long)g_texMemoMiss,
                100.0 * double(g_texMemoHits) / double(g_texMemoHits + g_texMemoMiss),
                (unsigned long long)g_texMemoDisagree, (unsigned long long)g_texGen);
    // OPEN ITEM 0w — the worst frames of the run and what was inside them.
    {
        SlowFrameRec t[12];
        memcpy(t, g_slowTop, sizeof t);
        std::sort(t, t + 12, [](const SlowFrameRec& a, const SlowFrameRec& b) {
            return a.us > b.us;
        });
        if (t[0].us)
        {
            fprintf(stderr, "[vk]   worst frames (open item 0w — what was IN them):\n");
            for (const SlowFrameRec& r : t)
            {
                if (!r.us)
                    break;
                fprintf(stderr,
                        "[vk]     frame %8llu: %8.1f ms = CPUrec %8.1f + fence %7.1f + "
                        "sleep %6.1f + resid %6.1f   GPU %7.1f  |  %6u draws  %4u tex "
                        "(%6u KB, up %5.1f + dec %6.1f ms)  %4u pipe\n",
                        (unsigned long long)r.frame, double(r.us) / 1000.0,
                        double(int32_t(r.walkUs) - int32_t(r.fenceUs)) / 1000.0,
                        double(r.fenceUs) / 1000.0, double(r.sleepUs) / 1000.0,
                        double(r.residualUs) / 1000.0, double(r.gpuUs) / 1000.0,
                        r.draws, r.tex, r.texKB, double(r.texUs) / 1000.0,
                        double(r.texDecUs) / 1000.0, r.pipes);
            }
            fprintf(stderr,
                    "[vk]     (CPUrec = our recording; fence = waiting for the GPU; "
                    "sleep = pump idle; resid = not the renderer at all. The first four "
                    "SUM TO THE WALL TIME. GPU is that frame's own execution time from its "
                    "command buffer's timestamps, and OVERLAPS the CPU columns by design "
                    "— it is not part of the sum)\n");
            fprintf(stderr,
                    "[vk]     texture uploads over the run: %llu uploads, %.1f MB, "
                    "%.1f ms staging+submit (%.0f us each), biggest single upload "
                    "%.2f MB\n",
                    (unsigned long long)g_texRealUploads,
                    double(g_texUploadBytes) / 1048576.0,
                    double(g_texUploadNs) / 1e6,
                    g_texRealUploads ? double(g_texUploadNs) / 1000.0
                                           / double(g_texRealUploads)
                                     : 0.0,
                    double(g_texUploadMaxBytes) / 1048576.0);
            fprintf(stderr,
                    "[vk]     ...and %.1f ms DECODING them (untile + endian swap + image "
                    "creation), %.0f us each — this is CPU work on the pump and it is "
                    "%.0f%% of the upload path\n",
                    double(g_texDecodeNs) / 1e6,
                    g_texRealUploads ? double(g_texDecodeNs) / 1000.0
                                           / double(g_texRealUploads) : 0.0,
                    (g_texDecodeNs + g_texUploadNs)
                        ? 100.0 * double(g_texDecodeNs)
                              / double(g_texDecodeNs + g_texUploadNs) : 0.0);
            // THE DECODE'S DECOMPOSITION (part 77), and the RESIDUAL is printed with it.
            // Every column here is a scope inside the one clock above, so they must sum to
            // less than it; what is left over is everything the split does not name, and a
            // large residual means the split is wrong rather than that the work vanished.
            // That is the "cannot return a false absence" shape part 74 had to build three
            // times (§6dn §5, gotcha 439).
            if (g_texDecodeNs)
            {
                const uint64_t named = g_texDecAllocNs + g_texDecBaseNs + g_texDecMipNs +
                                       g_texDecMipChkNs + g_texDecScanNs +
                                       g_texDecGuardNs + g_texDecImageNs +
                                       g_texDecGoldenNs;
                const double tot = double(g_texDecodeNs);
                auto pc = [&](uint64_t v) { return 100.0 * double(v) / tot; };
                fprintf(stderr,
                        "[vk]     decode split (ms / %% of decode): alloc %.1f (%.1f%%)  "
                        "base-untile %.1f (%.1f%%)  mip-untile %.1f (%.1f%%)  "
                        "mip-guards %.1f (%.1f%%)  content-scan %.1f (%.1f%%)  "
                        "src-hash %.1f (%.1f%%)  vkCreateImage %.1f (%.1f%%)  "
                        "golden %.1f (%.1f%%)  RESIDUAL %.1f (%.1f%%)\n",
                        double(g_texDecAllocNs) / 1e6, pc(g_texDecAllocNs),
                        double(g_texDecBaseNs) / 1e6, pc(g_texDecBaseNs),
                        double(g_texDecMipNs) / 1e6, pc(g_texDecMipNs),
                        double(g_texDecMipChkNs) / 1e6, pc(g_texDecMipChkNs),
                        double(g_texDecScanNs) / 1e6, pc(g_texDecScanNs),
                        double(g_texDecGuardNs) / 1e6, pc(g_texDecGuardNs),
                        double(g_texDecImageNs) / 1e6, pc(g_texDecImageNs),
                        double(g_texDecGoldenNs) / 1e6, pc(g_texDecGoldenNs),
                        double(g_texDecodeNs - std::min(named, g_texDecodeNs)) / 1e6,
                        pc(g_texDecodeNs - std::min(named, g_texDecodeNs)));
                fprintf(stderr,
                        "[vk]     base untile: %llu units over %llu uploads, %.1f ns/unit "
                        "(a total cannot tell a slow loop from a lot of units)\n",
                        (unsigned long long)g_texDecBaseUnits,
                        (unsigned long long)g_texRealUploads,
                        g_texDecBaseUnits
                            ? double(g_texDecBaseNs) / double(g_texDecBaseUnits) : 0.0);
            }
            if (g_mgChecked)
                fprintf(stderr,
                        "[vk]     mip-guard verify: %llu of %llu checks DISAGREED with the "
                        "pre-part-77 computation (%.4f%%) — 0 is the only passing value, "
                        "and CZ_VK_VERIFY_MIP_GUARD_POISON=1 must make it 100%%\n",
                        (unsigned long long)g_mgDisagree,
                        (unsigned long long)g_mgChecked,
                        100.0 * double(g_mgDisagree) / double(g_mgChecked));
            if (g_ciN)
                fprintf(stderr,
                        "[vk]     CreateImage x%llu (%.1f MB device): vkCreateImage %.1f ms"
                        "  memReq %.1f  vkAllocateMemory %.1f  bind %.1f  view %.1f "
                        " -> %.0f us each\n",
                        (unsigned long long)g_ciN,
                        double(g_ciAllocBytes) / 1048576.0,
                        double(g_ciCreateNs) / 1e6, double(g_ciReqNs) / 1e6,
                        double(g_ciAllocNs) / 1e6, double(g_ciBindNs) / 1e6,
                        double(g_ciViewNs) / 1e6,
                        double(g_ciCreateNs + g_ciReqNs + g_ciAllocNs + g_ciBindNs +
                               g_ciViewNs) / 1000.0 / double(g_ciN));
            fprintf(stderr,
                    "[vk]     image memory: %llu pooled into %llu block%s, %llu dedicated "
                    "allocations (%.1f KB lost to alignment inside blocks)\n",
                    (unsigned long long)g_imgPooled,
                    (unsigned long long)g_imgBlockAllocs,
                    g_imgBlockAllocs == 1 ? "" : "s",
                    (unsigned long long)g_imgDedicated,
                    double(g_imgPoolWasteBytes) / 1024.0);
            fprintf(stderr,
                    "[vk]     texture upload batch: %llu jobs in %llu flushes (%.1f per "
                    "flush, biggest %llu), %llu of them forced by a full staging arena — "
                    "each flush is ONE submit where each JOB used to be one\n",
                    (unsigned long long)g_texBatchJobs,
                    (unsigned long long)g_texBatchFlushes,
                    g_texBatchFlushes ? double(g_texBatchJobs) / double(g_texBatchFlushes)
                                      : 0.0,
                    (unsigned long long)g_texBatchMaxJobs,
                    (unsigned long long)g_texBatchFullFlushes);
            // THE PART-79 ITEM, and the two numbers that decide whether it worked. The
            // flush cost is what the operator's session measured at 568 us; the stall
            // count is whether three slots were enough. A ring that stalls on most
            // flushes has not removed the wait, it has renamed it.
            if (g_texFlushes)
                fprintf(stderr,
                        "[vk]     texture flush: %llu flushes, %.1f ms total (%.0f us "
                        "each) — ring %s; %llu of them STALLED waiting for a slot "
                        "(%.1f ms, %.0f us each)\n",
                        (unsigned long long)g_texFlushes,
                        double(g_texFlushNs) / 1e6,
                        double(g_texFlushNs) / 1000.0 / double(g_texFlushes),
                        g_texFlushWait ? "DISABLED (CZ_VK_TEX_FLUSH_WAIT=1 — submit and "
                                         "vkQueueWaitIdle, the part-77 renderer)"
                                       : "engaged, no wait on the submitting slot",
                        (unsigned long long)g_texSlotStalls,
                        double(g_texSlotStallNs) / 1e6,
                        g_texSlotStalls ? double(g_texSlotStallNs) / 1000.0
                                              / double(g_texSlotStalls)
                                        : 0.0);
            if (g_immN)
                fprintf(stderr,
                        "[vk]     immediate submits: %llu, %.1f ms total, of which "
                        "%.1f ms (%.1f%%) is vkQueueSubmit+vkQueueWaitIdle "
                        "(%.0f us each) — the wait is for the WHOLE QUEUE, so it "
                        "includes the in-flight frame\n",
                        (unsigned long long)g_immN, double(g_immTotalNs) / 1e6,
                        double(g_immWaitNs) / 1e6,
                        100.0 * double(g_immWaitNs) / double(g_immTotalNs),
                        double(g_immWaitNs) / 1000.0 / double(g_immN));

        }
    }
    // THE PASS-SIZE HISTOGRAM, and the column that decides item A is the DRAW-WEIGHTED
    // one, not the pass count. "46 of 48 passes are tiny" sounds fatal and is irrelevant
    // if the other two carry 95% of the draws — a worker pool feeds on DRAWS. So both are
    // printed side by side and the cumulative draw share is spelled out, because the
    // decision is "how many passes do I need before I have most of the frame".
    if (g_passCount)
    {
        fprintf(stderr,
                "[vk]   pass sizes: %llu passes, %llu draws, mean %.0f, max %llu\n",
                (unsigned long long)g_passCount, (unsigned long long)g_passDraws,
                double(g_passDraws) / double(g_passCount),
                (unsigned long long)g_passMax);
        // Walk from the LARGEST bucket down: the answer wanted is "the biggest N passes
        // hold X% of all draws", which reads directly off a descending cumulative.
        uint64_t cumP = 0, cumD = 0;
        for (int b = kPassBuckets - 1; b >= 0; --b)
        {
            if (!g_passHist[b])
                continue;
            cumP += g_passHist[b];
            cumD += g_passHistDraws[b];
            char range[32];
            if (b == 0)
                snprintf(range, sizeof range, "empty");
            else if (b == 1)
                snprintf(range, sizeof range, "1");
            else if (b >= int(kPassBuckets) - 1)
                snprintf(range, sizeof range, ">=%u", 1u << (kPassBuckets - 2));
            else
                snprintf(range, sizeof range, "%u-%u", 1u << (b - 1), (1u << b) - 1);
            fprintf(stderr,
                    "[vk]     %8s draws: %8llu passes (%5.1f%%), %10llu draws (%5.1f%%)"
                    "   cumulative from the top: %5.1f%% of passes hold %5.1f%% of draws\n",
                    range, (unsigned long long)g_passHist[b],
                    100.0 * double(g_passHist[b]) / double(g_passCount),
                    (unsigned long long)g_passHistDraws[b],
                    g_passDraws ? 100.0 * double(g_passHistDraws[b]) / double(g_passDraws) : 0.0,
                    100.0 * double(cumP) / double(g_passCount),
                    g_passDraws ? 100.0 * double(cumD) / double(g_passDraws) : 0.0);
        }
    }
    // THE CPU/GPU SPLIT over the whole run — the headline for "was the frame CPU or GPU".
    if (g_gpuFrames || g_fenceWaitNs)
    {
        const double frames = double(R->frame ? R->frame : 1);
        if (g_timestampPeriodNs <= 0.0)
            fprintf(stderr, "[vk]   GPU frame time: UNAVAILABLE on this device (no "
                            "timestamp support) — the fence wait below is the only "
                            "GPU-side number\n");
        else
            fprintf(stderr,
                    "[vk]   GPU frame time: %.2f ms mean over %llu frames measured "
                    "(from each frame's own command-buffer timestamps)\n",
                    double(g_gpuFrameNs) / 1e6 / double(g_gpuFrames ? g_gpuFrames : 1),
                    (unsigned long long)g_gpuFrames);
        fprintf(stderr,
                "[vk]   fence wait: %.2f ms/frame mean — this is the CPU BLOCKED ON THE "
                "GPU, and it is the half of `walk` that is not our recording\n",
                double(g_fenceWaitNs) / 1e6 / frames);
    }
    // THE BARRIER FORM, and the evidence that the arm engaged. Unconditional and free.
    if (g_barrierN)
        fprintf(stderr,
                "[vk]   image barriers: %llu over the run (%.1f/frame), %llu of them with "
                "the WIDE ALL_COMMANDS/MEMORY_* masks (%.1f%%) — under "
                "CZ_VK_WIDE_BARRIERS=1 that must be 100%%, and without it only GENERAL and "
                "unlisted layouts; plus %llu write-after-write barriers on images already "
                "in the layout they needed\n",
                (unsigned long long)g_barrierN,
                double(g_barrierN) / double(R->frame ? R->frame : 1),
                (unsigned long long)g_barrierWide,
                100.0 * double(g_barrierWide) / double(g_barrierN),
                (unsigned long long)g_wawN);
    // THE PER-REGION GPU SPLIT (part 78 item 1) — the first breakdown of the device's own
    // time this project has had. THE RESIDUAL IS PRINTED FIRST AND ON PURPOSE: it is the
    // frame's measured GPU time minus everything the regions account for, and it is the
    // only thing here that can say the split is wrong. A large residual means a region is
    // missing, not that work vanished (gotcha 237's shape, one level down).
    if (g_gpFrames)
    {
        const double f = double(g_gpFrames);
        const double totMs = double(g_gpTotalNs) / 1e6 / f;
        const double attMs = double(g_gpAttribNs) / 1e6 / f;
        fprintf(stderr,
                "[vk]   GPU per-region split (CZ_VK_GPU_PASSES) over %llu frames — "
                "%.3f ms/frame measured, %.3f ms attributed, RESIDUAL %.3f ms (%.1f%%)\n",
                (unsigned long long)g_gpFrames, totMs, attMs, totMs - attMs,
                totMs > 0.0 ? 100.0 * (totMs - attMs) / totMs : 0.0);
        for (int c = 0; c < kGpClasses; ++c)
        {
            const double ms = double(g_gpNs[c]) / 1e6 / f;
            fprintf(stderr,
                    "[vk]     %-20s %8.3f ms/frame (%5.1f%% of the frame's GPU time)  "
                    "%8.2f regions/frame, %7.0f ns each\n",
                    kGpNames[c], ms, totMs > 0.0 ? 100.0 * ms / totMs : 0.0,
                    double(g_gpN[c]) / f,
                    g_gpN[c] ? double(g_gpNs[c]) / double(g_gpN[c]) : 0.0);
        }
        // THE INVOCATION CENSUS (CZ_VK_GPU_STATS). Per pass class: primitives in, vertex
        // invocations, fragment invocations — and fragment invocations per INTERNAL
        // PIXEL, which is the overdraw factor and the first number a GPU budget needs.
        {
            bool any = false;
            for (int c = 0; c < kGpClasses; ++c)
                any = any || g_stN[c];
            if (any)
            {
                // The VISIBLE internal resolution, not the EDRAM stand-in (which carries the
                // 1024-row guest surface below the 720 the title presents from): overdraw
                // is "how many times the screen was shaded", and the screen is 1920x1080.
                const double px = double(g_internalW.load()) * double(g_internalH.load());
                uint64_t tot[kStCounters] = {};
                fprintf(stderr,
                        "[vk]   INVOCATION CENSUS (CZ_VK_GPU_STATS) — per frame, over %llu "
                        "frames; overdraw = FS invocations / %ux%u internal pixels "
                        "(overflow %llu, bad reads %llu)\n",
                        (unsigned long long)g_gpFrames, g_internalW.load(),
                        g_internalH.load(),
                        (unsigned long long)g_stOverflow, (unsigned long long)g_stBadRead);
                for (int c = 0; c < kGpClasses; ++c)
                {
                    if (!g_stN[c])
                        continue;
                    for (uint32_t k = 0; k < kStCounters; ++k)
                        tot[k] += g_stSum[c][k];
                    fprintf(stderr,
                            "[vk]     %-20s %7.2f passes  IA prims %9.0f  VS inv %10.0f  "
                            "clip prims %9.0f  FS inv %11.0f  = %.2f x pixels  "
                            "(%.1f VS inv/prim)\n",
                            kGpNames[c], double(g_stN[c]) / f, double(g_stSum[c][1]) / f,
                            double(g_stSum[c][2]) / f, double(g_stSum[c][4]) / f,
                            double(g_stSum[c][5]) / f, double(g_stSum[c][5]) / f / px,
                            g_stSum[c][1] ? double(g_stSum[c][2]) / double(g_stSum[c][1])
                                          : 0.0);
                }
                fprintf(stderr,
                        "[vk]     %-20s          IA prims %9.0f  VS inv %10.0f  "
                        "clip prims %9.0f  FS inv %11.0f  = %.2f x pixels\n",
                        "ALL PASSES", double(tot[1]) / f, double(tot[2]) / f,
                        double(tot[4]) / f, double(tot[5]) / f, double(tot[5]) / f / px);
                for (uint32_t k = 0; k < kStCounters; ++k)
                    fprintf(stderr, "[vk]       %-18s %14.0f /frame\n", kStNames[k],
                            double(tot[k]) / f);
            }
        }
        fprintf(stderr,
                "[vk]     resolve copies moved %.2f Mpixel/frame (%.1f full %ux%u "
                "screens' worth)\n",
                double(g_gpResolvePixels) / 1e6 / f,
                double(g_gpResolvePixels) / f /
                    (double(R->color.width) * double(R->color.height)),
                R->color.width, R->color.height);
        if (g_gpClearN)
            fprintf(stderr,
                    "[vk]     resolve clears: full-image mechanism would write %.2f "
                    "Mpixel/frame over %.1f clears; the scoped rects cover %.2f Mpixel "
                    "(%.1f%% of the class %s)\n",
                    double(g_gpClearFullPixels) / 1e6 / f, double(g_gpClearN) / f,
                    double(g_gpClearScopedPixels) / 1e6 / f,
                    g_gpClearFullPixels ? 100.0 * (1.0 - double(g_gpClearScopedPixels) /
                                                             double(g_gpClearFullPixels))
                                        : 0.0,
                    EnvOn("CZ_VK_NO_DEFERRED_CLEAR")
                        ? "WOULD BE REMOVED by the deferred-scoped default (this run "
                          "carries the CZ_VK_NO_DEFERRED_CLEAR control arm)"
                        : "REMOVED by the deferred-scoped mechanism, part 90");
        // THE PASS EXTENT CENSUS (part 79 item 2). `pass: 1 draw` is ~30 passes a frame at
        // ~28 us and this project has never listed what they ARE. Sorted by total time, so
        // the first rows are the ones worth designing against, and each row carries its own
        // per-pass microseconds — which is what tells a full-screen shader apart from pass
        // overhead on a 96x45 bloom target. Truncated at 16 rows per class with the tail
        // SUMMED rather than dropped, because a silently truncated census reads as a
        // complete one (gotcha 3).
        for (int c = kGpPassEmpty; c <= kGpPassShadow; ++c)
        {
            if (g_gpExtents[c].empty())
                continue;
            std::vector<std::pair<uint64_t, GpExtentStat>> rows(g_gpExtents[c].begin(),
                                                                g_gpExtents[c].end());
            std::sort(rows.begin(), rows.end(),
                      [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
            fprintf(stderr, "[vk]     EXTENT CENSUS for '%s' — %zu distinct scissor sizes\n",
                    kGpNames[c], rows.size());
            size_t shown = 0;
            uint64_t tailN = 0, tailNs = 0;
            for (const auto& r : rows)
            {
                if (shown < 16)
                {
                    fprintf(stderr,
                            "[vk]       %5llux%-5llu  %8.3f ms/frame  %7.2f passes/frame  "
                            "%7.0f us each  %.2f Mpixel/frame\n",
                            (unsigned long long)(r.first >> 32),
                            (unsigned long long)(r.first & 0xFFFFFFFFull),
                            double(r.second.ns) / 1e6 / f, double(r.second.n) / f,
                            double(r.second.ns) / 1000.0 / double(r.second.n),
                            double(r.first >> 32) * double(r.first & 0xFFFFFFFFull) *
                                double(r.second.n) / 1e6 / f);
                    ++shown;
                }
                else
                {
                    tailN += r.second.n;
                    tailNs += r.second.ns;
                }
            }
            if (tailN)
                fprintf(stderr,
                        "[vk]       ...and %zu more sizes: %.3f ms/frame over %.2f "
                        "passes/frame\n",
                        rows.size() - shown, double(tailNs) / 1e6 / f, double(tailN) / f);
        }
        if (g_gpOverflow || g_gpBadRead)
            fprintf(stderr,
                    "[vk]     ** %llu regions dropped for want of a query slot and %llu "
                    "unreadable — every one of those UNDER-reports its class\n",
                    (unsigned long long)g_gpOverflow, (unsigned long long)g_gpBadRead);
        fprintf(stderr,
                "[vk]     (a region's time is the wall time between two BOTTOM_OF_PIPE "
                "timestamps, so it is an upper bound on its own cost where the device "
                "overlaps regions; the SUM and the RESIDUAL are the honest quantities)\n");
    }
    // §4b — WHAT THE 41 NEAR-EMPTY PASSES A FRAME ACTUALLY COST. Split by the size of
    // the pass that ended, because the decision is "what is recovered by not issuing the
    // near-empty ones", not "what do resolves cost" — a total is dominated by the 1.35
    // big resolves a frame, which snapshot a full-screen surface and are not removable.
    if (g_cycN[0] + g_cycN[1] + g_cycN[2])
    {
        const double frames = double(R->frame ? R->frame : 1);
        static const char* kNames[kCycClasses] = { "near-empty (<=1 draw)",
                                                   "small (2-255)", "big (>=256)" };
        fprintf(stderr,
                "[vk]   resolve/begin cycle cost (plan §4b) — CPU RECORDING time on the "
                "pump, not the GPU's cost for the extra pass instance:\n");
        double totalMs = 0.0;
        for (int c = 0; c < kCycClasses; ++c)
        {
            if (!g_cycN[c])
                continue;
            const double ms = double(g_cycResolveNs[c] + g_cycBeginNs[c]) / 1e6 / frames;
            totalMs += ms;
            fprintf(stderr,
                    "[vk]     %-22s %8.2f/frame  %7.0f ns each (resolve %6.0f + begin "
                    "%6.0f)  = %6.3f ms/frame   scope actually broken in %5.1f%%\n",
                    kNames[c], double(g_cycN[c]) / frames,
                    double(g_cycResolveNs[c] + g_cycBeginNs[c]) / double(g_cycN[c]),
                    double(g_cycResolveNs[c]) / double(g_cycN[c]),
                    double(g_cycBeginNs[c]) / double(g_cycN[c]),
                    ms, 100.0 * double(g_cycBroke[c]) / double(g_cycN[c]));
        }
        fprintf(stderr,
                "[vk]     total %.3f ms/frame; the NEAR-EMPTY class is the item's ceiling "
                "and it is %.3f ms/frame\n",
                totalMs, double(g_cycResolveNs[0] + g_cycBeginNs[0]) / 1e6 / frames);
        fprintf(stderr,
                "[vk]     (if 'scope actually broken' is small for the near-empty class, "
                "§4b's premise is wrong: those passes are not End+Begin cycles at all)\n");
    }
    // THE CONSTANT-SLOT RACE DETECTOR (part 74) — the gate the gather has to pass before
    // CZ_VK_CONST_GATHER=1 can become the default again.
    if (ConstRaceOn())
    {
        fprintf(stderr,
                "[vk]   const race: %llu draws checked over %llu frames — "
                "**%llu had their VERTEX window change between record and submit** "
                "(%.4f%%), %llu their pixel window, %llu of them in the PROJECTION "
                "(c0..c3), %llu on a slot shared across BOTH TILES; %llu of %llu frames "
                "dirty\n",
                (unsigned long long)g_raceDraws, (unsigned long long)g_raceFrames,
                (unsigned long long)g_raceVsChanged,
                g_raceDraws ? 100.0 * double(g_raceVsChanged) / double(g_raceDraws) : 0.0,
                (unsigned long long)g_racePsChanged,
                (unsigned long long)g_raceProjChanged,
                (unsigned long long)g_raceCrossTile,
                (unsigned long long)g_raceDirtyFrames,
                (unsigned long long)g_raceFrames);
        fprintf(stderr,
                "[vk]     ** AFFECTED (the number that decides): %llu draws read a "
                "register THEIR OWN shader uses that moved after they were recorded "
                "(%.4f%%), %llu of those in the projection, %llu of %llu frames. A change "
                "in registers a draw never reads is the gather working as designed.\n",
                (unsigned long long)g_raceAffected,
                g_raceDraws ? 100.0 * double(g_raceAffected) / double(g_raceDraws) : 0.0,
                (unsigned long long)g_raceAffectedProj,
                (unsigned long long)g_raceAffectedFrames,
                (unsigned long long)g_raceFrames);
        if (!g_raceAffected)
            fprintf(stderr,
                    "[vk]     CLEAN — but a clean run means nothing until the poison arm "
                    "has been seen to scream: re-run with CZ_VK_CONST_RACE_POISON=1 and "
                    "confirm it reports (gotcha 30)\n");
    }
    if (!g_residueByShader.empty())
    {
        std::vector<std::pair<uint64_t, const void*>> v;
        for (const auto& kv : g_residueByShader)
            v.push_back({ kv.second, kv.first });
        std::sort(v.rbegin(), v.rend());
        fprintf(stderr,
                "[vk]   shaders whose LIST omits some of c0..c3 — %zu of them. Since "
                "part 74 the gather copies c0..c3 unconditionally, so these are no longer "
                "reading residue; the census stays because it is what found the defect:\n",
                v.size());
        for (size_t i = 0; i < v.size() && i < 16; i++)
        {
            // Resolve the pointer back to the shader's HASH, which is the only identity
            // this renderer keeps and the one the .spv/.meta.json files are named by.
            uint64_t hash = 0;
            for (size_t k = 0; k < R->shaders.vals.size(); k++)
                if (&R->shaders.vals[k] == v[i].second)
                {
                    hash = R->shaders.keys[k];
                    break;
                }
            fprintf(stderr, "[vk]     %016llx  %10llu draws\n",
                    (unsigned long long)hash, (unsigned long long)v[i].first);
        }
    }
    if (g_patchResidue || g_patchGathered)
        fprintf(stderr,
                "[vk]   projection patch inputs: %llu windows had c0..c3 in the shader's "
                "OWN list, %llu did not (the gather copies c0..c3 regardless since part "
                "74, so both are real values now) — projection recognized in %llu of the "
                "latter (%.2f%%)\n",
                (unsigned long long)g_patchGathered,
                (unsigned long long)g_patchResidue,
                (unsigned long long)g_patchResidueRecognized,
                g_patchResidue
                    ? 100.0 * double(g_patchResidueRecognized) / double(g_patchResidue)
                    : 0.0);
    if (g_patchSrcChecked)
        fprintf(stderr,
                "[vk]   projection patch (cached source, part 75): %llu checked, "
                "**%llu DISAGREED with patching the arena copy** (%.4f%%)%s\n",
                (unsigned long long)g_patchSrcChecked,
                (unsigned long long)g_patchSrcBad,
                100.0 * double(g_patchSrcBad) / double(g_patchSrcChecked),
                g_patchSrcVerifyPoison ? " [POISON ARMED — a zero here is a BLIND "
                                         "verifier]" : "");
    if (g_skyFrames)
        fprintf(stderr,
                "[vk]   sky asymmetry (CZ_VK_SKY_ASYM): %llu frames, mean |L-R| %.3f, "
                "mean frame-to-frame step %.3f, max step %.3f, **%llu SIGN FLIPS "
                "(%.3f%% of frames)**\n",
                (unsigned long long)g_skyFrames, g_skyAbsSum / double(g_skyFrames),
                g_skyFrames > 1 ? g_skyStepSum / double(g_skyFrames - 1) : 0.0,
                g_skyStepMax, (unsigned long long)g_skyFlips,
                100.0 * double(g_skyFlips) / double(g_skyFrames));
    // PERF ITEM C's bill, on every stats dump — the bytes NOT copied, which is the whole
    // point of the item, plus the two numbers that say whether it is safe: how many stages
    // fell back to the full copy, and what the verifier found if it was armed.
    {
        const uint64_t tot = g_gatherFull + g_gatherGathered + g_gatherDynBounded;
        if (tot)
        {
            const uint64_t wouldBe = tot * 256 * 4;
            const uint64_t actual =
                g_gatherDwordsFull + g_gatherDwordsCopied + g_gatherDwordsDynBounded;
            fprintf(stderr,
                    "[vk]   const gather: %.1f%% of window copies gathered (%llu full — "
                    "dynamic a0 or no list), %.2f GB not copied over the run (%.1f%% of "
                    "%.2f GB)\n",
                    100.0 * double(g_gatherGathered) / double(tot),
                    (unsigned long long)g_gatherFull,
                    double(wouldBe - actual) * 4.0 / 1e9,
                    100.0 * double(wouldBe - actual) / double(wouldBe),
                    double(wouldBe) * 4.0 / 1e9);
            // Part 88's mechanism number: how many dynamic copies took the bounded path
            // and what they actually moved against the 4 KB each cost before. The
            // step-3 reading rule wants this beside the frame time, because at the
            // route's ±2.9% floor the byte count is the number that cannot be argued
            // with.
            if (g_gatherDynBounded)
                fprintf(stderr,
                        "[vk]     bounded dynamic: %llu copies moved %.2f GB where full "
                        "copies were %.2f GB (-%.1f%%); %llu dynamic copies still full\n",
                        (unsigned long long)g_gatherDynBounded,
                        double(g_gatherDwordsDynBounded) * 4.0 / 1e9,
                        double(g_gatherDynBounded) * 4096.0 / 1e9,
                        100.0 * (1.0 - double(g_gatherDwordsDynBounded) /
                                           (double(g_gatherDynBounded) * 256.0 * 4.0)),
                        (unsigned long long)g_gatherDynamic);
            // Part 88 item 2's mechanism number: the recognition work the memo removed
            // is its hit share; the verify line beside it is what makes the share safe
            // to believe.
            if (g_patchMemoHits + g_patchMemoMisses)
                fprintf(stderr,
                        "[vk]     patch memo: %.1f%% of %llu VS patches served from the "
                        "4-way MRU (%llu misses)\n",
                        100.0 * double(g_patchMemoHits) /
                            double(g_patchMemoHits + g_patchMemoMisses),
                        (unsigned long long)(g_patchMemoHits + g_patchMemoMisses),
                        (unsigned long long)g_patchMemoMisses);
            if (g_patchMemoChecked)
                fprintf(stderr,
                        "[vk]     patch memo verified: %llu hits re-patched and "
                        "compared, **%llu disagreed**%s\n",
                        (unsigned long long)g_patchMemoChecked,
                        (unsigned long long)g_patchMemoBad,
                        PatchMemoVerifyPoison()
                            ? "  (POISONED — a zero here means the verifier is BLIND)"
                            : "");
            if (g_gatherChecked)
                fprintf(stderr,
                        "[vk]     verified: %llu gathers checked against the full copy, "
                        "**%llu disagreed**%s\n",
                        (unsigned long long)g_gatherChecked,
                        (unsigned long long)g_gatherBad,
                        ConstGatherPoison()
                            ? "  (POISONED — a zero here means the verifier is BLIND)"
                            : "");
            if (g_gatherChecked && g_gatherDynBounded)
                fprintf(stderr,
                        "[vk]     ...of which %llu were BOUNDED dynamic copies%s\n",
                        (unsigned long long)g_gatherBadBounded,
                        ConstGatherPoison()
                            ? " (poisoned: zero here means the BOUNDED verifier is blind)"
                            : "");
        }
    }
    // Part 88 step 0's verdict, WINDOW rates only (gotcha 428: a cumulative mean is a
    // transient). Each print covers the copies since the previous one.
    if (g_paletteCensus)
    {
        const palcensus::Tot& t = palcensus::t;
        palcensus::Tot& l = palcensus::last;
        const uint64_t n = t.copies - l.copies;
        if (n)
        {
            const uint64_t full = t.bytesFull - l.bytesFull;
            const uint64_t bnd = t.bytesBounded - l.bytesBounded;
            const uint64_t hwB = t.bytesHighWater - l.bytesHighWater;
            fprintf(stderr,
                    "[palcensus] %llu dynamic VS copies since last line: clean-cover "
                    "%.1f%%, dirty-fallback %.1f%%, reuse %.1f%%; never-bound %llu, "
                    "window-moved %llu; bursts/copy cover %.2f partial %.2f\n",
                    (unsigned long long)n,
                    100.0 * double(t.cleanCover - l.cleanCover) / double(n),
                    100.0 * double(t.dirtyFallback - l.dirtyFallback) / double(n),
                    100.0 * double(t.reuse - l.reuse) / double(n),
                    (unsigned long long)(t.neverBound - l.neverBound),
                    (unsigned long long)(t.windowMoved - l.windowMoved),
                    double(t.coverBursts - l.coverBursts) / double(n),
                    double(t.partialBursts - l.partialBursts) / double(n));
            fprintf(stderr,
                    "[palcensus]   bytes: full %.1f MB -> bounded %.1f MB (SAVES %.1f%%; "
                    "kill < 30%%) | high-water-only model %.1f MB (saves %.1f%%), "
                    "high-water c%u, mean bound %.1f regs\n",
                    double(full) / 1e6, double(bnd) / 1e6,
                    full ? 100.0 * double(full - bnd) / double(full) : 0.0,
                    double(hwB) / 1e6,
                    full ? 100.0 * double(full - hwB) / double(full) : 0.0,
                    DrawVsPaletteHighWater(),
                    double(t.extentSum - l.extentSum) / double(n));
            std::string hg = "[palcensus]   bound histogram (regs):";
            for (int b = 0; b < 32; ++b)
            {
                const uint64_t c = t.hist[b] - l.hist[b];
                if (!c)
                    continue;
                char buf[64];
                snprintf(buf, sizeof buf, " %d-%d:%llu", b * 8, b * 8 + 7,
                         (unsigned long long)c);
                hg += buf;
            }
            fprintf(stderr, "%s\n", hg.c_str());
            l = t;
        }
    }
    // THE ORDER GATE's verdict, on every stats dump. A gate whose result is not printed is
    // the defect this project keeps rediscovering (part 56's stencil skip counter was
    // collected for fifteen parts and printed by nothing), and this one guards the largest
    // and riskiest item in the plan.
    if (OrderGateArmed())
        fprintf(stderr,
                "[order]   draw-order gate: %llu frames checked, **%llu FAILED**, "
                "%llu draws logged%s\n",
                (unsigned long long)R->orderFramesChecked,
                (unsigned long long)R->orderFramesFailed,
                (unsigned long long)R->orderDrawsLogged,
                Env("CZ_VK_ORDER_POISON")
                    ? "  (POISONED — a zero here means the gate is BLIND)"
                    : (R && R->parRec
                           ? "  (PARALLEL recording: the replayed instances' own ids "
                             "against the capture order — zero is the claim)"
                           : "  (serial recording: zero is the only correct result)"));
    // Part 72 item 1. Printed HERE as well as on the census's own cadence, so a soak
    // that ends off a 600-frame boundary still lands the number — the same defect the
    // stencil skip counter had for fifteen parts (collected since part 56, printed by
    // nothing). Inert and silent unless the census is armed.
    VkRenderer_DumpVerticalWaste();
    // The constant memo, printed on EVERY run and not only under the profiler — the
    // operator's A/B harness deliberately runs without `CZ_VK_PROFILE` (it costs 2-4 ms a
    // frame and would change the thing being judged), so without this line a soak could
    // not say whether the arm engaged at all (gotcha 151). Run totals, not a window.
    {
        const uint64_t tot = g_constMemoRunHits + g_constMemoRunMisses;
        if (tot)
            fprintf(stderr,
                    "[vk]   const memo: %.1f%% of half-copies served (VS %.1f%%, PS "
                    "%.1f%%), %.1f GB not copied over the run%s\n",
                    100.0 * double(g_constMemoRunHits) / double(tot),
                    200.0 * double(g_constMemoRunVsHits) / double(tot),
                    200.0 * double(g_constMemoRunPsHits) / double(tot),
                    double(g_constMemoRunHits) * 4096.0 / 1073741824.0,
                    g_constMemoOff ? " [CZ_VK_NO_CONST_MEMO: the copy ran every draw]" : "");
    }

    // The polygon offset, printed on every run: an arm with no counter cannot be shown to
    // have engaged (gotcha 151), and this one's effect is a judgement about a picture.
    fprintf(stderr, "[vk]   stencil test: %llu draws enabled it%s\n",
            (unsigned long long)g_stencilDraws,
            EnvOn("CZ_VK_NO_STENCIL") ? "  [CZ_VK_NO_STENCIL: none were honoured]" : "");
    fprintf(stderr, "[vk]   polygon offset: %llu draws asked for one%s\n",
            (unsigned long long)g_polyOffsetDraws,
            EnvOn("CZ_VK_NO_POLY_OFFSET") ? "  [CZ_VK_NO_POLY_OFFSET: none were applied]"
                                          : "");

    // PART 71'S PIPELINE-CREATION CENSUS, printed on every run. The TOP-FRAME table is
    // the point, not the totals: four seconds of compiling spread over 20,000 frames is
    // invisible to a player and four seconds inside one frame is the thing the operator
    // reported, and only a per-frame roll-up separates them.
    {
        PipeFrameFlush();          // include the frame in progress
        fprintf(stderr,
                "[vk]   pipeline creation: %llu pipelines, %.1f ms total, worst single "
                "%.1f ms @frame %llu%s\n",
                (unsigned long long)g_pipeCount, double(g_pipeNs) / 1e6,
                double(g_pipeWorstNs) / 1e6, (unsigned long long)g_pipeWorstFrame,
                R->pipeCache == VK_NULL_HANDLE ? "  [no pipeline cache]" : "");
        // Sorted by cost, worst first — a stall the player felt is a FRAME, so this is
        // the table that lines up with a `[fps]` window's `worst`.
        PipeFrameRec top[12];
        std::copy(std::begin(g_pipeTop), std::end(g_pipeTop), std::begin(top));
        std::sort(std::begin(top), std::end(top),
                  [] (const PipeFrameRec& a, const PipeFrameRec& b) { return a.ns > b.ns; });
        for (const PipeFrameRec& t : top)
            if (t.count)
                fprintf(stderr, "[vk]     frame %8llu: %7.1f ms building %u pipeline(s)\n",
                        (unsigned long long)t.frame, double(t.ns) / 1e6, t.count);
    }

    // PART 71's hook fold, printed on EVERY run for the same reason as the two above: the
    // operator's A/B harness runs without the profiler, so this line is the only thing
    // that can say whether the arm engaged. THE IDENTITY GATE IS HERE TOO — with RT ON
    // both `folded` numbers must read 0, because the word is the OR of every hook's own
    // arm and cannot be false while any of them could do work.
    {
        const uint64_t d = g_hookFoldFolded + g_hookFoldLive;
        const uint64_t f = g_hookFoldFetchFolded + g_hookFoldFetchLive;
        fprintf(stderr,
                "[vk]   hook fold: %llu of %llu draws folded (%.1f%%), %llu of %llu "
                "atlas-fetch decodes folded (%.1f%%)%s\n",
                (unsigned long long)g_hookFoldFolded, (unsigned long long)d,
                d ? 100.0 * double(g_hookFoldFolded) / double(d) : 0.0,
                (unsigned long long)g_hookFoldFetchFolded, (unsigned long long)f,
                f ? 100.0 * double(g_hookFoldFetchFolded) / double(f) : 0.0,
                EnvOn("CZ_VK_NO_HOOK_FOLD")
                    ? "  [CZ_VK_NO_HOOK_FOLD: the pre-part-71 per-draw calls]"
                    : "");
    }

    // The flat tables' grow bill, printed on EVERY run rather than only under the
    // profiler — a play session the operator drives has no profiler, and it is exactly
    // the run where a hitch gets reported.
    fprintf(stderr,
            "[vk]   flat cache grows: %llu, %.2f ms total, worst %.2f ms%s\n",
            (unsigned long long)g_flatGrows, double(g_flatGrowNs) / 1e6,
            double(g_flatGrowWorstNs) / 1e6,
            g_flatCacheOff ? " [CZ_VK_NO_FLAT_CACHE: no flat table was in use]" : "");
    // THE STREAM STORE'S GROWTHS, beside the flat cache's for the same reason: both are
    // rare, both happen entirely inside one frame, and a rare single-frame cost is exactly
    // what a run mean cannot see and a player can. Part 79's operator session had two, and
    // each was the worst frame of its own 10-second window.
    if (g_persistGrowN)
        fprintf(stderr,
                "[vk]   stream store grows: %llu, %.1f ms total, %.1f ms each — each one is "
                "a whole frame of PUMP time and it is the hitch class §6dy §3 names\n",
                (unsigned long long)g_persistGrowN, double(g_persistGrowNs) / 1e6,
                double(g_persistGrowNs) / 1e6 / double(g_persistGrowN));
    if (g_ccCopies)
    {
        fprintf(stderr,
                "[vk]   copy census: %llu resolve copies, %llu DEAD (%.1f%%) — "
                "%.2f of %.2f Gpixel dead (%.1f%%), %llu sample marks; dead = the "
                "same (snapshot, rect) copied again with no consumer in between, "
                "counted conservatively toward LIVE\n",
                (unsigned long long)g_ccCopies, (unsigned long long)g_ccDead,
                100.0 * double(g_ccDead) / double(g_ccCopies),
                double(g_ccDeadPixels) / 1e9, double(g_ccPixels) / 1e9,
                g_ccPixels ? 100.0 * double(g_ccDeadPixels) / double(g_ccPixels) : 0.0,
                (unsigned long long)g_ccSampleMarks);
        // The keys carrying the dead pixels, so a verdict names surfaces, not a share.
        std::vector<std::tuple<uint64_t, uint64_t, uint32_t, size_t>> byDead;
        for (const auto& kv : g_ccMap)
        {
            uint64_t dPx = 0, dN = 0;
            for (const auto& r : kv.second)
            {
                dPx += r.deadPx;
                dN += r.deadN;
            }
            byDead.emplace_back(dPx, dN, kv.first, kv.second.size());
        }
        std::sort(byDead.rbegin(), byDead.rend());
        for (size_t i = 0; i < byDead.size() && i < 10; ++i)
            fprintf(stderr,
                    "[vk]     copy census: %08X%s  %llu dead copies, %.2f Gpixel dead, "
                    "%zu distinct rects\n",
                    std::get<2>(byDead[i]) & 0x1FFFFFFF,
                    (std::get<2>(byDead[i]) & kSnapshotDepthBit) ? " (depth)" : "",
                    (unsigned long long)std::get<1>(byDead[i]),
                    double(std::get<0>(byDead[i])) / 1e9, std::get<3>(byDead[i]));
    }
    if (R->persistDev.buffer)
        fprintf(stderr,
                "[vk]   store mirror: %.2f MB/frame copied host->VRAM in %.1f copies/frame "
                "over the run; persist hits bound the MIRROR %.1f%% of the time (%llu dev, "
                "%llu host)\n",
                R->frame ? double(R->mirrorBytes) / double(R->frame) / 1048576.0 : 0.0,
                R->frame ? double(R->mirrorCopies) / double(R->frame) : 0.0,
                (R->mirrorHitsDev + R->mirrorHitsHost)
                    ? 100.0 * double(R->mirrorHitsDev) /
                          double(R->mirrorHitsDev + R->mirrorHitsHost)
                    : 0.0,
                (unsigned long long)R->mirrorHitsDev,
                (unsigned long long)R->mirrorHitsHost);
    fprintf(stderr, "[vk]   pipelines=%zu shaders=%zu textures=%zu arenaHighWater=%llu KB\n",
            R->pipelines.size(), R->shadersMap.size(), TexSize(),
            (unsigned long long)(R->arenaHighWater >> 10));
    // The state cache's own engagement, as fractions of the draws it was offered.
    // Printed unconditionally, including on the CZ_VK_NO_STATE_CACHE arm where every
    // figure is 0 by construction — an arm whose "on" run is indistinguishable from its
    // "off" run is exactly what a missing counter hides (gotcha 151).
    if (const uint64_t d = R->skips.draws)
        fprintf(stderr,
                "[vk]   binds skipped per draw: pipeline %.1f%% viewport %.1f%% "
                "scissor %.1f%% blend %.1f%% descriptor-sets %.1f%% (of %llu draws)\n",
                100.0 * double(R->skips.pipeline) / double(d),
                100.0 * double(R->skips.viewport) / double(d),
                100.0 * double(R->skips.scissor) / double(d),
                100.0 * double(R->skips.blend) / double(d),
                100.0 * double(R->skips.sets) / double(d), (unsigned long long)d);
    // PARALLEL RECORD's engagement at exit, UNCONDITIONAL when the feature is on —
    // the [vkprof] line only exists under the profiler, and a timed run (correctly)
    // does not carry one, which left the 3v3's fix arms provable only through the
    // skip aggregation. This line is the direct statement (gotcha 151).
    if (R->parRec)
    {
        uint64_t chunks = 0;
        for (uint32_t rec = 0; rec < kPrMaxRecorders; ++rec)
            chunks += g_prChunksRecorded[rec];
        fprintf(stderr,
                "[vk]   parallel record: %llu chunks (%llu by the pump at the wait), "
                "%llu draws captured, %llu tail draws in %llu tail instances, %llu "
                "empty instances, submit wait %.1f ms total, overflow-inline %llu, "
                "bind overflow %llu%s\n",
                (unsigned long long)chunks, (unsigned long long)g_prPumpHelped,
                (unsigned long long)g_prCaptured, (unsigned long long)g_prTailDraws,
                (unsigned long long)g_prTailInstances,
                (unsigned long long)g_prEmptyInstances, double(g_prWaitNs) / 1e6,
                (unsigned long long)g_prOverflowInline,
                (unsigned long long)g_prBindOverflow,
                g_prBindOverflow ? "  *** CAPTURES TRUNCATED — picture suspect ***" : "");
    }
    // THE CEILING PROBE's own engagement. Printed only when it fired, but printed with the
    // per-draw rate rather than the raw total, because the number the item's arithmetic
    // needs is "driver calls per draw" — that is what a secondary command buffer carries,
    // and the state cache means it is nowhere near the ten calls the source suggests.
    if (g_reuseCensus)
        reusecensus::Print(" since the last line — FINAL");
    // The constant-write histogram, printed here so it lands beside the gather stats
    // whose full-copy population it exists to explain (part 87, phase5-notes §6eg).
    Pm4_DumpAluWriteCensus();
    if (g_fetchMemoCensus)
    {
        const uint64_t n = g_fetchMemoHits + g_fetchMemoMisses;
        fprintf(stderr,
                "[vk]   vertex-fetch memo census: %.1f%% of %llu draws would be SERVED by "
                "(shader, fetch-const version)\n"
                "[vk]     misses: shader %llu (%.1f%%, inherent) + version %llu (%.1f%%, "
                "the guest touched the fetch file between two draws of one shader)\n",
                n ? 100.0 * double(g_fetchMemoHits) / double(n) : 0.0,
                (unsigned long long)n, (unsigned long long)g_fetchMemoShaderMiss,
                n ? 100.0 * double(g_fetchMemoShaderMiss) / double(n) : 0.0,
                (unsigned long long)g_fetchMemoVersionMiss,
                n ? 100.0 * double(g_fetchMemoVersionMiss) / double(n) : 0.0);
        // The number the decision turns on, spelled out rather than left to be multiplied
        // by hand: the decode is 124 ns a draw (§6eb §3), so this is what a perfect memo
        // would be worth at the load this run actually reached.
        const uint64_t ne = g_fetchMemoExactHits + g_fetchMemoExactMisses;
        const double exact = ne ? double(g_fetchMemoExactHits) / double(ne) : 0.0;
        fprintf(stderr,
                "[vk]   EXACT key (shader + a hash of only the dwords this shader's "
                "attributes read): %.1f%% of %llu draws SERVED\n",
                100.0 * exact, (unsigned long long)ne);
        // The number the decision turns on, spelled out rather than left to be multiplied
        // by hand: the decode is 124 ns a draw (§6eb §3), so this is what each key would be
        // worth at the operator's load.
        fprintf(stderr,
                "[vk]     at 124 ns/draw of decode and 9,300 draws: whole-file key %.3f ms, "
                "EXACT key %.3f ms per frame — against an item-1 ceiling of 2.33 ms that "
                "needed 3 threads and delivered 0.00 with the budget as it stands\n",
                9.3 * 124e-3 * (n ? double(g_fetchMemoHits) / double(n) : 0.0),
                9.3 * 124e-3 * exact);
    }
    // THE BATCH'S OWN MECHANISM NUMBER, unconditional — it is the statistic the item is
    // judged on and it cannot be argued with, where a frame time on this route has a
    // ±2.9% floor. `CZ_VK_NO_BIND_BATCH=1` should read ~1.74 here and the batch ~0.47.
    if (g_bindBatchDraws)
        fprintf(stderr,
                "[vk]   vertex bind calls: %.3f per draw over %llu batched draws (%llu "
                "vkCmdBindVertexBuffers)%s\n",
                double(g_bindBatchCalls) / double(g_bindBatchDraws),
                (unsigned long long)g_bindBatchDraws,
                (unsigned long long)g_bindBatchCalls,
                g_noBindBatch ? " [CZ_VK_NO_BIND_BATCH=1, one call per binding]" : "");
    if (g_verifyBindBatch)
        fprintf(stderr,
                "[vk]   BIND BATCH VERIFY: %llu of %llu (binding, buffer, offset) triples "
                "DISAGREED with what the draw asked for (%.4f%%)%s\n",
                (unsigned long long)g_bindVerifyBad,
                (unsigned long long)g_bindVerifyChecked,
                g_bindVerifyChecked
                    ? 100.0 * double(g_bindVerifyBad) / double(g_bindVerifyChecked)
                    : 0.0,
                g_verifyBindPoison ? "  [POISON ARM — this MUST read 100%]" : "");
    if (g_bindRunCensus && g_brDraws)
    {
        // The last draw's run is still open — close it before reading, or the histogram
        // silently loses one entry and the runs/draw mean reads low.
        BindRunCensusCloseRun();
        const double d = double(g_brDraws);
        // WHAT A BATCHER WOULD ISSUE: one call per contiguous run of changed bindings,
        // plus every untracked bind, which is never batched.
        const double batched = double(g_brRuns + g_brUntracked) / d;
        const double now = double(g_brChanged + g_brUntracked) / d;
        fprintf(stderr,
                "[vk]   BIND-RUN CENSUS over %llu draws: offered %.3f/draw, changed "
                "%.3f/draw (%.1f%%), runs of changed %.3f/draw, untracked %.3f/draw "
                "(%llu draws had one)\n"
                "[vk]     calls/draw now %.3f -> batched %.3f  (saving %.3f/draw = "
                "%.0f ns = %.3f ms/frame at 9,300 draws and 52 ns a driver call)\n",
                (unsigned long long)g_brDraws, double(g_brOffered) / d,
                double(g_brChanged) / d,
                g_brOffered ? 100.0 * double(g_brChanged) / double(g_brOffered) : 0.0,
                double(g_brRuns) / d, double(g_brUntracked) / d,
                (unsigned long long)g_brUntrackedDraws,
                now, batched, now - batched, (now - batched) * 52.0,
                (now - batched) * 52e-6 * 9300.0);
        // THE DISTRIBUTION, because the mean above is consistent with both hypotheses.
        char hist[256];
        int at = 0;
        uint64_t tot = 0;
        for (int k = 1; k <= 8; k++)
            tot += g_brRunHist[k];
        for (int k = 1; k <= 8 && at < int(sizeof hist) - 24; k++)
            at += snprintf(hist + at, sizeof hist - at, "%s%d:%llu(%.1f%%)",
                           k > 1 ? " " : "", k == 8 ? 8 : k,
                           (unsigned long long)g_brRunHist[k],
                           tot ? 100.0 * double(g_brRunHist[k]) / double(tot) : 0.0);
        fprintf(stderr, "[vk]     run-length histogram (8 = 8 or more): %s\n", hist);
    }
    if (g_guardCensus && R->frame)
    {
        const double f = double(R->frame);
        const uint64_t tot = g_gcPumpBytes + g_gcPoolBytes;
        fprintf(stderr,
                "[vk]   GUARD CENSUS over %llu frames: %.2f MB/frame read in total — "
                "PUMP %.2f MB/frame over %.0f hashes (%.1f%% of bytes), POOL %.2f MB/frame "
                "over %.0f (%.1f%%)\n"
                "[vk]     the pump's own half cost %.3f ms/frame (%llu ns over the run, "
                "%.1f ns a hash, %.2f GB/s)\n",
                (unsigned long long)R->frame, double(tot) / f / 1048576.0,
                double(g_gcPumpBytes) / f / 1048576.0, double(g_gcPumpCount) / f,
                tot ? 100.0 * double(g_gcPumpBytes) / double(tot) : 0.0,
                double(g_gcPoolBytes) / f / 1048576.0, double(g_gcPoolCount) / f,
                tot ? 100.0 * double(g_gcPoolBytes) / double(tot) : 0.0,
                double(g_gcPumpNs) / f / 1e6, (unsigned long long)g_gcPumpNs,
                g_gcPumpCount ? double(g_gcPumpNs) / double(g_gcPumpCount) : 0.0,
                g_gcPumpNs ? double(g_gcPumpBytes) / double(g_gcPumpNs) : 0.0);
    }
    if ((g_pzHits || g_pzMisses || g_prezeroOff) && R->frame)
    {
        const double f = double(R->frame);
        const uint64_t served = g_pzHits, inl = g_pzMisses;
        fprintf(stderr,
                "[vk]   SHARED-BLOCK PRE-ZERO (part 111 B1)%s: %.1f%% of draws served "
                "pre-zeroed (%llu of %llu)\n"
                "[vk]     moved off the pump %.2f MB/frame; still inline %.2f MB/frame\n"
                "[vk]     the OTHER side of the bill: workers %.3f ms/frame zeroing, pump "
                "%.3f ms/frame draining (%llu helped, %llu yields) and %.3f ms/frame "
                "waiting on a busy chunk (%llu waits)\n"
                "[vk]     chunks: %llu posted over %llu dispatches (%.1f a frame), %llu "
                "claimed by the PUMP first (workers skipped %llu), %llu slots past the "
                "watermark, %llu region overflows\n",
                g_prezeroOff ? " — OFF (the default; CZ_VK_PREZERO=1 engages it)" : "",
                (served + inl) ? 100.0 * double(served) / double(served + inl) : 0.0,
                (unsigned long long)served, (unsigned long long)(served + inl),
                double(g_pzBytesPre) / f / 1048576.0,
                double(g_pzBytesInline) / f / 1048576.0,
                double(g_pzWorkerNsA.load()) / f / 1e6, double(g_pzDrainNs) / f / 1e6,
                (unsigned long long)g_pzDrainHelped, (unsigned long long)g_pzDrainWaits,
                double(g_pzWaitNs) / f / 1e6, (unsigned long long)g_pzWaits,
                (unsigned long long)g_pzChunksPosted,
                (unsigned long long)g_pzDispatches,
                g_pzDispatches ? double(g_pzChunksPosted) / double(g_pzDispatches) : 0.0,
                (unsigned long long)g_pzPumpClaims,
                (unsigned long long)g_pzWorkerSkips.load(),
                (unsigned long long)g_pzBeyondWatermark,
                (unsigned long long)g_pzOverflow);
    }
    if (g_pardrawCensus && R->frame)
    {
        const double f = double(R->frame);
        const double d = double(R->skips.draws ? R->skips.draws : 1);
        const uint64_t streamTot = g_pdc.streamFinds;
        fprintf(stderr,
                "[vk]   PER-DRAW MUTATION CENSUS over %llu frames / %llu draws — part 111 "
                "§3, item B's ask-first step\n"
                "[vk]     arena bump      %9.0f/frame (%.2f/draw), %7.2f MB/frame — ours, "
                "serial, NOT timed (a clock read is 5x the op)\n"
                "[vk]     persist bump    %9.0f/frame, %7.2f MB/frame\n"
                "[vk]     stream cache    %9.0f finds/frame, %7.0f inserts/frame = "
                "%.2f%% MUTATIONS, insert cost %.3f ms/frame\n"
                "[vk]     persist store   %9.0f finds/frame, %7.0f inserts/frame + %.0f "
                "mirror pushes, mutation cost %.3f ms/frame\n"
                "[vk]     texture table   %9.0f finds/frame (EACH STAMPS lastUsedFrame — a "
                "read-modify-WRITE), %.1f inserts/frame, %.3f ms/frame\n"
                "[vk]     descriptor set  %9.0f writes/frame, %.3f ms/frame — vkUpdate* is "
                "externally synchronised and cannot leave the pump unguarded\n"
                "[vk]     pipeline cache  %9.0f finds/frame, %.2f inserts/frame\n"
                "[vk]     READS OF g_regs %9.0f const-window copies/frame (%.2f/draw) + "
                "%.0f fetch walks/frame (%.2f/draw) — B3's source race, counted\n",
                (unsigned long long)R->frame, (unsigned long long)R->skips.draws,
                double(g_pdc.arenaAllocs) / f, double(g_pdc.arenaAllocs) / d,
                double(g_pdc.arenaBytes) / f / 1048576.0,
                double(g_pdc.persistAllocs) / f,
                double(g_pdc.persistAllocBytes) / f / 1048576.0,
                double(g_pdc.streamFinds) / f, double(g_pdc.streamInserts) / f,
                streamTot ? 100.0 * double(g_pdc.streamInserts) / double(streamTot) : 0.0,
                double(g_pdc.streamInsertNs) / f / 1e6,
                double(g_pdc.persistFinds) / f, double(g_pdc.persistInserts) / f,
                double(g_pdc.mirrorPushes) / f, double(g_pdc.persistMutNs) / f / 1e6,
                double(g_pdc.texFinds) / f, double(g_pdc.texInserts) / f,
                double(g_pdc.texInsertNs) / f / 1e6,
                double(g_pdc.descWrites) / f, double(g_pdc.descWriteNs) / f / 1e6,
                double(g_pdc.pipeFinds) / f, double(g_pdc.pipeInserts) / f,
                double(g_pdc.constWindowCopies) / f, double(g_pdc.constWindowCopies) / d,
                double(g_pdc.fetchWalks) / f, double(g_pdc.fetchWalks) / d);
        // THE ONE LINE THE PLAN ASKED FOR. `S` is what stays on the pump however good the
        // sharding is; the read share is what says whether a read-mostly table is enough.
        const uint64_t mutNs = g_pdc.streamInsertNs + g_pdc.persistMutNs +
                               g_pdc.texInsertNs + g_pdc.descWriteNs;
        const uint64_t reads = g_pdc.streamFinds + g_pdc.persistFinds + g_pdc.texFinds;
        const uint64_t writes = g_pdc.streamInserts + g_pdc.persistInserts +
                                g_pdc.mirrorPushes + g_pdc.texInserts + g_pdc.descWrites;
        fprintf(stderr,
                "[vk]     => S (TIMED mutations that must stay serial or shard) = %.3f "
                "ms/frame; shared-table traffic is %.1f%% reads (%llu reads, %llu writes "
                "per run). The arena bump is counted, not timed, and is excluded from S.\n",
                double(mutNs) / f / 1e6,
                (reads + writes) ? 100.0 * double(reads) / double(reads + writes) : 0.0,
                (unsigned long long)reads, (unsigned long long)writes);
    }
    if (g_streamDedupCensus && R->skips.draws)
        fprintf(stderr,
                "[vk]   stream lookups: %.2f per draw, %.1f%% of them REPEAT a key this "
                "same draw already looked up (%llu of %llu; %llu draws exceeded the "
                "16-key window and were counted as distinct, which under-reports repeats)\n",
                double(g_dedupLookups) / double(R->skips.draws),
                g_dedupLookups ? 100.0 * double(g_dedupRepeats) / double(g_dedupLookups) : 0.0,
                (unsigned long long)g_dedupRepeats, (unsigned long long)g_dedupLookups,
                (unsigned long long)g_dedupOverflow);
    if (g_noDriverRecordSkipped && R->skips.draws)
        fprintf(stderr,
                "[vk]   CZ_VK_NO_DRIVER_RECORD: %llu vkCmd* calls skipped, %.2f per draw "
                "— NOTHING WAS DRAWN in this run\n",
                (unsigned long long)g_noDriverRecordSkipped,
                double(g_noDriverRecordSkipped) / double(R->skips.draws));
    // THE STENCIL SKIP, ADDED IN PART 72's PREP — and it was COLLECTED SINCE PART 56 AND
    // NEVER PRINTED, which is the defect this project keeps rediscovering (a counter you
    // already pay for that no log carries). It is the deciding number for
    // `perf-plan-part72.md` §3, the last named suspect for part 58's +1.3-1.6 ms: part 56
    // clears `haveStencil` on EVERY pipeline bind, because binding a pipeline that
    // specifies state statically makes the corresponding dynamic state undefined — and
    // 26.4% of this title's draws bind a new pipeline. So the question is whether the
    // three `vkCmdSetStencil*` calls collapse to their real change rate or are re-issued
    // constantly, and only this line can answer it.
    //
    // The DENOMINATOR is stencil-enabled draws, not all draws: a draw with the stencil
    // test off neither sets nor skips, and folding it into the total would report a
    // healthy-looking 95% for a cache that never serves anything.
    if (g_stencilDraws)
        fprintf(stderr,
                "[vk]   stencil dynamic-state sets skipped: %.1f%% (of %llu "
                "stencil-enabled draws)\n",
                100.0 * double(R->skips.stencil) / double(g_stencilDraws),
                (unsigned long long)g_stencilDraws);
    // ...and the two the cache COVERS AS OF PART 47. The numbers were the measurement
    // that justified writing it (51.0% and 39.4% on the operator's own session, over
    // 16.17 M draws); they are now the count of calls the renderer did not make.
    // Reported as counts as well as percentages because the absolute number is what
    // multiplies by the ~340 ns a `vkCmd*` costs here.
    if (R->skips.vertexBinds || R->skips.indexBinds)
        fprintf(stderr,
                "[vk]   binds skipped by the state cache: vertex %llu of %llu repeat "
                "the previous offset (%.1f%%), index %llu of %llu (%.1f%%)\n",
                (unsigned long long)R->skips.vertexBindRepeats,
                (unsigned long long)R->skips.vertexBinds,
                R->skips.vertexBinds
                    ? 100.0 * double(R->skips.vertexBindRepeats) /
                          double(R->skips.vertexBinds)
                    : 0.0,
                (unsigned long long)R->skips.indexBindRepeats,
                (unsigned long long)R->skips.indexBinds,
                R->skips.indexBinds ? 100.0 * double(R->skips.indexBindRepeats) /
                                          double(R->skips.indexBinds)
                                    : 0.0);
    for (const auto& [name, count] : g_stats)
        fprintf(stderr, "[vk]   %-52s %llu\n", name.c_str(),
                (unsigned long long)count);

    // The per-address table. Only the rows that say something are printed: a surface
    // this renderer resolved to, or an upload that came out entirely zero. Everything
    // else is an ordinary disc texture and the aggregate counters already cover it.
    if (g_texCensus)
    {
        fprintf(stderr, "[vk]   texture sources (addr, extent, fmt | uploads/zero, "
                        "snapshot, tooOld maxAge):\n");
        // Sorted, so this prints exactly what the `std::map` printed before part 55
        // made the table flat. A census whose ROW ORDER changes reads as a different
        // census to anyone diffing two runs' logs.
        std::vector<std::pair<uint32_t, TexSource>> srcRows;
        g_texSources.ForEach([&] (uint64_t k, const TexSource& v) {
            srcRows.emplace_back(uint32_t(k), v);
        });
        std::sort(srcRows.begin(), srcRows.end(),
                  [] (const auto& a, const auto& b) { return a.first < b.first; });
        for (auto& [addr, s] : srcRows)
        {
            if (!s.everResolved && !s.zeroUploads)
                continue;
            // Re-read the source bytes NOW. A row that uploaded black and is still
            // black in guest memory is a texture the guest never wrote; one that
            // uploaded black and now reads non-zero is a texture that arrived AFTER
            // our one and only upload, and is frozen black by the cache.
            const char* note = "";
            if (s.zeroUploads && s.srcBytes && s.src)
            {
                const uint8_t* p = s.src;
                bool nowZero = true;
                for (uint64_t i = 0; i < s.srcBytes; i++)
                    if (p[i])
                    {
                        nowZero = false;
                        break;
                    }
                note = nowZero ? "   <- uploaded BLACK, guest memory STILL zero"
                               : "   <- uploaded BLACK, guest memory is NON-ZERO NOW";
            }
            fprintf(stderr,
                    "[vk]     %08X%-7s %4ux%-4u f%-2u | up %llu (zero %llu)  snap %llu  "
                    "tooOld %llu (max age %llu)%s\n",
                    addr & 0x1FFFFFFF, (addr & kSnapshotDepthBit) ? "(depth)" : "",
                    s.width, s.height, s.format, (unsigned long long)s.uploads,
                    (unsigned long long)s.zeroUploads,
                    (unsigned long long)s.fromSnapshot,
                    (unsigned long long)s.snapshotTooOld, (unsigned long long)s.maxAge,
                    note);
        }
    }

    // ROUTE (B)'s ENGAGEMENT LINE, always on when an RT tier ever ran. Part 64 shipped
    // a build whose A/B measured a flawless fix and was the feature silently switched
    // off (gotcha 386); the cheapest defence against repeating that is a count that
    // cannot be confused with a configuration.
    if (rtfactor::g_fireSamples)
        fprintf(stderr,
                "[rtb] the factor pass fires at draw %llu of ~%llu on average "
                "(min %llu, max %llu); %llu atlas draws DECLINED as Z-prepass "
                "(empty colour mask). This title has NO scene Z prepass "
                "(tools/rt_depth_order_census.py, 20 traces), so under the DEPTH source "
                "the sample here is always the clear value — far, i.e. LIT. The "
                "primary-ray source does not care where in the frame this number "
                "lands.\n",
                (unsigned long long)(rtfactor::g_fireAtSum / rtfactor::g_fireSamples),
                (unsigned long long)(rtfactor::g_frameDrawSum / rtfactor::g_fireSamples),
                (unsigned long long)rtfactor::g_fireAtMin,
                (unsigned long long)rtfactor::g_fireAtMax,
                (unsigned long long)rtfactor::g_declinedPrepass);

    // THE ATLAS BINDING, printed so it is checkable rather than assumed. One row means
    // the shadow-sampling shaders fetch exactly one depth surface; the chosen one is
    // marked, and a competing candidate is visible instead of silently losing.
    if (rtshadow::g_atlasCandCount)
    {
        fprintf(stderr, "[rtb] depth surfaces fetched by the shadow shaders "
                        "(the largest is taken as the cascade atlas):\n");
        for (uint32_t i = 0; i < rtshadow::g_atlasCandCount; ++i)
            fprintf(stderr, "[rtb]   %08X %5ux%-5u  %llu fetches%s\n",
                    rtshadow::g_atlasCands[i].addr, rtshadow::g_atlasCands[i].w,
                    rtshadow::g_atlasCands[i].h,
                    (unsigned long long)rtshadow::g_atlasCands[i].fetches,
                    rtshadow::g_atlasCands[i].addr == rtshadow::g_atlasAddr
                        ? "   <-- ATLAS" : "");
    }

    // EVERY SUN DIRECTION THE LATCH SAW, not just the last. One row is the answer;
    // two rows means something that is not the sun's cascade is being captured, which
    // is exactly what cost the operator's second session (§6cw §9).
    if (rtshadow::g_sunObsCount)
    {
        fprintf(stderr, "[rtb] sun directions latched (%llu latches, %u distinct, "
                        "%llu vote switches — more than one direction is EXPECTED here "
                        "and the per-frame majority is what chooses):\n",
                (unsigned long long)rtshadow::g_sunLatched, rtshadow::g_sunObsCount,
                (unsigned long long)rtshadow::g_sunSwitches);
        for (uint32_t i = 0; i < rtshadow::g_sunObsCount; ++i)
            fprintf(stderr, "[rtb]   (%+.3f %+.3f %+.3f) volume %.1f  x%llu  "
                            "frames %llu..%llu\n",
                    rtshadow::g_sunObs[i].dir[0], rtshadow::g_sunObs[i].dir[1],
                    rtshadow::g_sunObs[i].dir[2], rtshadow::g_sunObs[i].len,
                    (unsigned long long)rtshadow::g_sunObs[i].count,
                    (unsigned long long)rtshadow::g_sunObs[i].firstFrame,
                    (unsigned long long)rtshadow::g_sunObs[i].lastFrame);
    }
    // AND THE TITLE'S OWN ANSWER BESIDE IT — the standing gate on part 70's change.
    // The two are independent readings of one physical quantity, so a disagreement is a
    // defect in whichever is not the title's, and hardware has already said which that
    // is (`tools/xtr_sun_oracle.py`, twenty of twenty at 0.00 degrees). Printed even
    // when the cascade arm is selected, because the comparison is the point.
    if (rtshadow::g_guestSunProbes)
    {
        fprintf(stderr,
                "[rtb] the TITLE'S OWN sun (pixel constant c23, cross-checked against "
                "its own cascade matrix): %llu frames bound, %u distinct, %llu probe "
                "draws, %llu blocks REJECTED for c23-vs-cascade disagreement\n",
                (unsigned long long)rtshadow::g_guestSunSamples,
                rtshadow::g_gsunObsCount,
                (unsigned long long)rtshadow::g_guestSunProbes,
                (unsigned long long)rtshadow::g_guestSunMismatch);
        for (uint32_t i = 0; i < rtshadow::g_gsunObsCount; ++i)
            fprintf(stderr, "[rtb]   (%+.3f %+.3f %+.3f)  x%llu  frames %llu..%llu\n",
                    rtshadow::g_gsunObs[i].dir[0], rtshadow::g_gsunObs[i].dir[1],
                    rtshadow::g_gsunObs[i].dir[2],
                    (unsigned long long)rtshadow::g_gsunObs[i].count,
                    (unsigned long long)rtshadow::g_gsunObs[i].firstFrame,
                    (unsigned long long)rtshadow::g_gsunObs[i].lastFrame);
        if (rtshadow::g_sunDisagree >= 0.0f)
            fprintf(stderr,
                    "[rtb]   the two readings differ by %.1f degrees, and the pass used "
                    "the %s one\n",
                    rtshadow::g_sunDisagree,
                    rtshadow::g_sunSrcGuest ? "TITLE'S" : "cascade-derived");
    }
    else if (rtshadow::g_sunObsCount)
        fprintf(stderr, "[rtb] the title's own sun constant was NEVER PROBED — either "
                        "RT was off for the whole run or no draw carried the world "
                        "constant block\n");
    if (rtfactor::g_passes || rtfactor::g_noScene || rtfactor::g_noLight ||
        rtfactor::g_noTlas || rtfactor::g_singular)
    {
        rtshadow::PrintCollectorCensus("collector TOTAL");
        fprintf(stderr,
                "[rtb] TOTAL: %llu factor passes, %llu draws served, %u variant "
                "modules; skipped noScene=%llu noLight=%llu noTlas=%llu singular=%llu. "
                "Scene composite: %llu frames carried ONE, %llu carried SEVERAL (any "
                "'several' means the world/camera binding is not what §6cs measured)\n",
                (unsigned long long)rtfactor::g_passes,
                (unsigned long long)rtfactor::g_drawsServed, R->rtVariants,
                (unsigned long long)rtfactor::g_noScene,
                (unsigned long long)rtfactor::g_noLight,
                (unsigned long long)rtfactor::g_noTlas,
                (unsigned long long)rtfactor::g_singular,
                (unsigned long long)rtfactor::g_framesOneMatrix,
                (unsigned long long)rtfactor::g_framesManyMatrix);
    }

    // WHERE THE DIMENSION LIVES IN THE FETCH CONSTANT, read off the two classes the
    // shader partitions every fetch into. `always1` is the AND, `always0` is the
    // complement of the OR; a field that encodes the dimension must be inside the bits
    // where the two classes' patterns differ, and the report prints that disagreement
    // mask directly so the answer is a bit position rather than an argument.
    if (g_dimCensus)
    {
        static const char* kDimName[4] = { "1D", "2D", "3D", "Cube" };
        fprintf(stderr, "[vk]   fetch-constant dimension census — dwords per "
                        "shader-declared dimension:\n");
        for (const auto& [dim, c] : g_dimClasses)
        {
            fprintf(stderr, "[vk]     %-4s  %llu fetches\n",
                    dim < 4 ? kDimName[dim] : "?", (unsigned long long)c.fetches);
            for (uint32_t d = 0; d < 6; d++)
                fprintf(stderr, "[vk]       dword%u always1=%08X always0=%08X\n", d,
                        c.andMask[d], ~c.orMask[d]);
            fprintf(stderr, "[vk]       dword2>>26 (the stack depth Xenia's layout "
                            "predicts is 5 for a cube):");
            for (const auto& [v, n] : c.d2Top)
                fprintf(stderr, " %u x%llu", v, (unsigned long long)n);
            fprintf(stderr, "\n");
        }
        // The pairwise disagreement, which is the actual answer. Only computed between
        // classes that both saw fetches — a class with none has AND=~0 and OR=0, which
        // would disagree with everything and mean nothing (gotcha 3).
        for (const auto& [a, ca] : g_dimClasses)
            for (const auto& [b, cb] : g_dimClasses)
            {
                if (a >= b || !ca.fetches || !cb.fetches)
                    continue;
                for (uint32_t d = 0; d < 6; d++)
                {
                    // Bits one class always sets and the other always clears, either
                    // way round. Anything else varies within a class and cannot be a
                    // constant per-dimension field.
                    const uint32_t sep = (ca.andMask[d] & ~cb.orMask[d]) |
                                         (cb.andMask[d] & ~ca.orMask[d]);
                    if (sep)
                        fprintf(stderr,
                                "[vk]     %s vs %s: dword%u separates on bits %08X\n",
                                a < 4 ? kDimName[a] : "?", b < 4 ? kDimName[b] : "?", d,
                                sep);
                }
            }
    }

    // WHICH SHADERS DISAGREE WITH THEIR OWN FETCH CONSTANTS, and about which texture.
    // Unbounded, unlike the per-occurrence print above, because the population is the
    // question: one shader disagreeing about one placeholder texture and fifty shaders
    // disagreeing about fifty real ones are the same counter and completely different
    // defects. Compare the shader hashes here against `tools/xtr_cube_agreement.py` on a
    // capture — a shader that disagrees here and agrees there is OUR register file; a
    // shader that appears in no capture is a case hardware has never been asked about.
    if (g_dimDisagree)
    {
        static const char* kDimName[4] = { "1D", "2D", "3D", "Cube" };
        uint64_t total = 0;
        for (const auto& [k, e] : g_dimDisagreements)
            total += e.fetches;
        fprintf(stderr,
                "[vk]   shader/constant dimension disagreements: %llu fetches over %zu "
                "distinct (shader, slot, texture) cases\n",
                (unsigned long long)total, g_dimDisagreements.size());
        for (const auto& [k, e] : g_dimDisagreements)
            fprintf(stderr,
                    "[vk]     ps=%016llx vs=%016llx s%-2u shader=%-4s constant=%-4s "
                    "%08X %ux%u fmt=%u  x%llu\n",
                    (unsigned long long)e.psHash, (unsigned long long)e.vsHash, e.slot,
                    e.shaderDim < 4 ? kDimName[e.shaderDim] : "?",
                    e.constDim < 4 ? kDimName[e.constDim] : "?", e.addr, e.w, e.h, e.fmt,
                    (unsigned long long)e.fetches);
    }

    // The texture-content guard. The question is the operator's: is a draw being served
    // an image built from pixels that are no longer at that address?
    goldenwriter::Drain(); // part 102: finish the queued golden files before reporting
    if (goldenwriter::dropped)
        fprintf(stderr, "[vk] golden texture store: %llu persist(s) DROPPED — the writer "
                        "queue was full (%zu); they will be captured again next session\n",
                (unsigned long long)goldenwriter::dropped, goldenwriter::kMaxQueued);
    if (!g_noGolden && (g_goldenStored || g_goldenServed))
        fprintf(stderr,
                "[vk]   golden texture store: %llu signatures remembered, %llu all-zero "
                "uploads served real bytes (%zu entries held)\n",
                (unsigned long long)g_goldenStored, (unsigned long long)g_goldenServed,
                g_goldenTex.size());
    if (!g_noGolden && !g_noGoldenPack
        && (goldenwriter::packAppended || goldenwriter::looseRemoved || g_goldenPackEntries))
        fprintf(stderr,
                "[vk]   golden pack: %zu entries at load + %llu appended this session "
                "(%.1f MB written), %llu loose files folded in and removed\n",
                g_goldenPackEntries, (unsigned long long)goldenwriter::packAppended,
                double(goldenwriter::packAppendedBytes) / (1024.0 * 1024.0),
                (unsigned long long)goldenwriter::looseRemoved);
    if (g_texGuardStats.hits)
    {
        const TexGuardStats& g = g_texGuardStats;
        fprintf(stderr,
                "[vk]   texture guard: %llu cache hits checked, **%llu served an image "
                "whose guest bytes had CHANGED (%.2f%%)**, %llu re-uploaded | guard read "
                "%.1f MB\n",
                (unsigned long long)g.hits, (unsigned long long)g.changed,
                100.0 * double(g.changed) / double(g.hits),
                (unsigned long long)g.reuploaded,
                double(g.guardBytes) / 1048576.0);
        // WHAT THE ONCE-PER-FRAME POLICY SAVED, as a share of what the pre-part-47
        // renderer would have hashed — because an arm with no counter cannot be shown to
        // have engaged (gotcha 151), and because this ratio IS the item: it is the
        // redundancy factor between fetches and distinct textures in a frame, which
        // nothing in this runtime had ever measured.
        {
            const uint64_t would = g.hits + g.skippedSameFrame;
            fprintf(stderr,
                    "[vk]   texture guard cadence: %llu of %llu checks skipped because "
                    "the entry was ALREADY validated this frame (%.1f%%, i.e. %.1fx less "
                    "hashing)%s\n",
                    (unsigned long long)g.skippedSameFrame, (unsigned long long)would,
                    would ? 100.0 * double(g.skippedSameFrame) / double(would) : 0.0,
                    g.hits ? double(would) / double(g.hits) : 0.0,
                    g_texGuardEveryFetch ? "  [CZ_VK_TEX_GUARD_EVERY_FETCH — expect 0]"
                                         : "");
        }
        // ...and WHERE those bytes went, by texture size. Read it against the per-address
        // `changed` table below: a prefix bound above every size that appears there costs
        // nothing in detection and saves everything above it. Bytes are what the guard
        // ACTUALLY read (the sampled path already caps a big surface at the stream
        // bound), so the column is a true cost and not a size sum.
        {
            uint64_t tot = 0;
            for (size_t b = 0; b < kTexGuardHistBuckets; b++)
                tot += g_texGuardHistBytes[b];
            fprintf(stderr, "[vk]   texture guard bytes by SOURCE size (checks/MB read):");
            for (size_t b = 0; b < kTexGuardHistBuckets; b++)
            {
                if (!g_texGuardHistCount[b])
                    continue;
                char lo[16];
                if (b == 0)
                    snprintf(lo, sizeof lo, "<1K");
                else
                    snprintf(lo, sizeof lo, "%zuK", size_t(1) << b);
                fprintf(stderr, "  %s=%llu/%.1fMB(%.0f%%)", lo,
                        (unsigned long long)g_texGuardHistCount[b],
                        double(g_texGuardHistBytes[b]) / 1048576.0,
                        tot ? 100.0 * double(g_texGuardHistBytes[b]) / double(tot) : 0.0);
            }
            fprintf(stderr, "\n");
        }
        if (g_texGuardPoison)
            fprintf(stderr, "[vk]   (POISONED: that share MUST be 100.00%% — the census "
                            "is only trustworthy if it can also report a positive)\n");
        // The addresses, worst first. A ratio alone cannot separate "one atlas the CPU
        // rewrites every frame" from "a third of the world's textures are wrong", and
        // those are different defects with different fixes.
        std::vector<std::pair<uint32_t, TexGuardAddr>> rows;
        g_texGuardAddrs.ForEach([&] (uint64_t k, const TexGuardAddr& v) {
            rows.emplace_back(uint32_t(k), v);
        });
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
            return a.second.changed > b.second.changed;
        });
        size_t shown = 0, withChange = 0;
        for (const auto& r : rows)
            if (r.second.changed)
                ++withChange;
        fprintf(stderr, "[vk]   %zu of %zu cached texture addresses served changed "
                        "bytes at least once; worst 24:\n",
                withChange, rows.size());
        for (const auto& [addr, a] : rows)
        {
            if (!a.changed || shown++ >= 24)
                break;
            // srcBytes is on this line as of part 47: it is the size a prefix bound has
            // to cover to keep seeing this address change, and the whole population is
            // 24 rows, so the answer to "what bound is safe" is readable straight off it.
            fprintf(stderr, "[vk]     %08X %4ux%-4u f%-2u %7llu B  %llu of %llu hits "
                            "stale (%.1f%%)\n",
                    addr, a.width, a.height, a.format,
                    (unsigned long long)a.srcBytes, (unsigned long long)a.changed,
                    (unsigned long long)a.hits,
                    100.0 * double(a.changed) / double(a.hits));
        }
    }
}


// ===================================================================================
// `cz_runtime --diag` — the Vulkan half (part 105, docs/steam-deck-plan.md §3 item 1)
// ===================================================================================
// Everything bring-up would decide, printed for a device that is never created: every
// physical device the loader sees with its driver's own name and version, the one the
// renderer would pick, the requirements table verdict on it, the EDRAM depth format,
// the MSAA sample counts and the device-local memory. One line per fact so a player
// can paste the block into an issue. Returns false when the loader has no device or
// the pick fails the table — the exit code a script can read.
bool VkRenderer_Diag()
{
    const char* T = "[diag] vulkan:";
    uint32_t loaderVer = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion)
        vkEnumerateInstanceVersion(&loaderVer);
    fprintf(stderr, "%s loader instance version %u.%u.%u\n", T, VK_VERSION_MAJOR(loaderVer),
            VK_VERSION_MINOR(loaderVer), VK_VERSION_PATCH(loaderVer));

    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "cz_runtime --diag";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    const VkResult ir = vkCreateInstance(&ici, nullptr, &inst);
    if (ir != VK_SUCCESS)
    {
        fprintf(stderr, "%s vkCreateInstance FAILED: VkResult %d — no Vulkan loader/ICD "
                        "usable from this process (is a Vulkan driver installed?)\n",
                T, int(ir));
        return false;
    }
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(inst, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(inst, &count, devices.data());
    fprintf(stderr, "%s %u physical device%s\n", T, count, count == 1 ? "" : "s");
    if (devices.empty())
    {
        vkDestroyInstance(inst, nullptr);
        return false;
    }
    // The same pick as CreateDevice: the first discrete GPU, else the first device.
    VkPhysicalDevice pick = devices[0];
    for (VkPhysicalDevice d : devices)
    {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            pick = d;
            break;
        }
    }
    bool ok = true;
    for (uint32_t i = 0; i < count; i++)
    {
        DeviceCaps c;
        QueryDeviceCaps(devices[i], c);
        fprintf(stderr, "%s   [%u] %s (%s) Vulkan %u.%u.%u vendor %#06x device %#06x%s\n", T,
                i, c.props.deviceName, DeviceTypeName(c.props.deviceType),
                VK_VERSION_MAJOR(c.props.apiVersion), VK_VERSION_MINOR(c.props.apiVersion),
                VK_VERSION_PATCH(c.props.apiVersion), c.props.vendorID, c.props.deviceID,
                devices[i] == pick ? "  <- the renderer would use this one" : "");
        std::string tag = std::string(T) + "      ";
        PrintDriverLine(c, tag.c_str());
        PrintRendererProfile(c, tag.c_str());
        if (devices[i] != pick)
            continue;

        if (c.props.apiVersion < VK_API_VERSION_1_3)
        {
            fprintf(stderr, "%s   VERDICT: CANNOT run the renderer — Vulkan 1.3 is required "
                            "and this device reports %u.%u\n", T,
                    VK_VERSION_MAJOR(c.props.apiVersion), VK_VERSION_MINOR(c.props.apiVersion));
            ok = false;
        }
        VkPhysicalDeviceVulkan12Features r12{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
        };
        VkPhysicalDeviceVulkan13Features r13{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES
        };
        VkPhysicalDeviceFeatures2 rf2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        std::vector<const char*> missing;
        fprintf(stderr, "%s   features the renderer asks for:\n", T);
        EvaluateRequirements(c, rf2, r12, r13, missing, T, /*listAll=*/true);
        if (missing.empty())
            fprintf(stderr, "%s   every REQUIRED feature is present\n", T);
        else
        {
            fprintf(stderr, "%s   VERDICT: CANNOT run the renderer — missing REQUIRED:", T);
            for (const char* m : missing)
                fprintf(stderr, " %s", m);
            fprintf(stderr, "\n");
            ok = false;
        }
        fprintf(stderr, "%s   VK_KHR_swapchain: %s (needed to present into a window)\n", T,
                c.HasExt(VK_KHR_SWAPCHAIN_EXTENSION_NAME) ? "present" : "ABSENT");
        fprintf(stderr, "%s   ray query (parked feature): %s\n", T,
                (c.HasExt("VK_KHR_acceleration_structure") && c.HasExt("VK_KHR_ray_query")
                 && c.HasExt("VK_KHR_deferred_host_operations"))
                    ? "supported" : "unsupported (fine; RT is parked)");

        bool d24 = false;
        const VkFormat depth = PickEdramDepthFormat(devices[i], &d24);
        fprintf(stderr, "%s   D24_UNORM_S8_UINT sampleable: %s -> EDRAM depth format %s%s\n", T,
                d24 ? "yes" : "no",
                depth == VK_FORMAT_D24_UNORM_S8_UINT ? "D24_UNORM_S8_UINT"
                                                     : "D32_SFLOAT_S8_UINT",
                EnvOn("CZ_VK_DEPTH_FLOAT") ? " (CZ_VK_DEPTH_FLOAT)" : "");
        VkFormatProperties fp{};
        vkGetPhysicalDeviceFormatProperties(devices[i], VK_FORMAT_D32_SFLOAT_S8_UINT, &fp);
        if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
            fprintf(stderr, "%s   D32_SFLOAT_S8_UINT is not a depth attachment here — a "
                            "device with neither depth format cannot run the renderer\n", T);
        const VkSampleCountFlags sup = c.props.limits.framebufferColorSampleCounts
                                       & c.props.limits.framebufferDepthSampleCounts;
        fprintf(stderr, "%s   framebuffer sample counts: colour %#x depth %#x -> the 2x MSAA "
                        "default %s\n", T,
                unsigned(c.props.limits.framebufferColorSampleCounts),
                unsigned(c.props.limits.framebufferDepthSampleCounts),
                (sup & 2) ? "is available" : (sup & 4) ? "is unavailable; 4x would be used"
                                                        : "is unavailable; single-sample");
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(devices[i], &mp);
        uint64_t local = 0;
        for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
            if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                local += mp.memoryHeaps[h].size;
        fprintf(stderr, "%s   device-local memory: %llu MB in %u heap%s; max 2D image %u; "
                        "sampled images per stage %u\n", T,
                (unsigned long long)(local >> 20), mp.memoryHeapCount,
                mp.memoryHeapCount == 1 ? "" : "s", c.props.limits.maxImageDimension2D,
                c.props.limits.maxPerStageDescriptorSampledImages);
        fprintf(stderr, "%s   timestamps: period %.3f ns%s\n", T,
                double(c.props.limits.timestampPeriod),
                c.props.limits.timestampPeriod > 0 ? "" : " (GPU frame time unavailable)");
    }
    fprintf(stderr, "%s VERDICT: the renderer %s on this machine's pick\n", T,
            ok ? "CAN run" : "CANNOT run");
    vkDestroyInstance(inst, nullptr);
    return ok;
}
