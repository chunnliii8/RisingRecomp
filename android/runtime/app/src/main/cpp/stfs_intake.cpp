#include "stfs_intake.h"

#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <fcntl.h>

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kBlockSize = 0x1000;
constexpr uint32_t kBlocksPerHashLevel[3] = {170, 28900, 4913000};
constexpr uint32_t kEndOfChain = 0xFFFFFF;
constexpr uint32_t kEntriesPerDirBlock = kBlockSize / 0x40;
constexpr uint32_t kExpectedTitleId = 0x58410A8D;
constexpr uint32_t kExpectedContentType = 0x000D0000;
constexpr uint32_t kExpectedXexSize = 4538368;
constexpr char kExpectedXexSha256[] =
        "f4d268dd6b3ec9a0b65fb4be61bdf87611ccca97aaf71a69e378914786c79eec";
constexpr char kManifestName[] = "install-manifest.txt";
constexpr char kCompleteName[] = ".install-complete";
constexpr uint64_t kFreeSpaceReserve = 16ull * 1024 * 1024;

constexpr uint32_t kOffHeaderSize = 0x340;
constexpr uint32_t kOffMetadata = 0x344;
constexpr uint32_t kOffContentType = kOffMetadata + 0x00;
constexpr uint32_t kOffContentSize = kOffMetadata + 0x08;
constexpr uint32_t kOffExecutionInfo = kOffMetadata + 0x10;
constexpr uint32_t kOffVolumeDescriptor = kOffMetadata + 0x35;
constexpr uint32_t kOffVolumeType = kOffMetadata + 0x65;
constexpr uint32_t kOffDisplayName = 0x411;

uint32_t BeU32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

uint64_t BeU64(const uint8_t* p) {
    return (uint64_t(BeU32(p)) << 32) | BeU32(p + 4);
}

uint16_t BeU16(const uint8_t* p) {
    return uint16_t((uint16_t(p[0]) << 8) | p[1]);
}

uint16_t LeU16(const uint8_t* p) {
    return uint16_t(p[0] | (uint16_t(p[1]) << 8));
}

uint32_t U24Le(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16);
}

uint64_t BlockIndexToOffset(uint64_t baseOffset, uint32_t blockIndex) {
    uint64_t block = blockIndex;
    for (uint32_t levelBase : kBlocksPerHashLevel) {
        block += (uint64_t(blockIndex) + levelBase) / levelBase;
        if (blockIndex < levelBase) break;
    }
    return baseOffset + (block << 12);
}

uint64_t BlockIndexToHashBlockNumber(uint32_t blockIndex) {
    if (blockIndex < kBlocksPerHashLevel[0]) return 0;
    uint64_t block = uint64_t(blockIndex / kBlocksPerHashLevel[0]) *
                     (kBlocksPerHashLevel[0] + 1);
    block += blockIndex / kBlocksPerHashLevel[1] + 1;
    if (blockIndex < kBlocksPerHashLevel[1]) return block;
    return block + 1;
}

bool SafeName(const std::string& name) {
    if (name.empty() || name == "." || name == ".." ||
        name.find('/') != std::string::npos || name.find('\\') != std::string::npos)
        return false;
    for (unsigned char character : name)
        if (character < 0x20) return false;
    return true;
}

struct Entry {
    std::string path;
    uint32_t length = 0;
    uint32_t startBlock = 0;
    uint32_t blockCount = 0;
};

class Sha256 {
public:
    void Update(const uint8_t* bytes, size_t length) {
        bitCount_ += uint64_t(length) * 8;
        while (length != 0) {
            const size_t count = std::min(length, sizeof buffer_ - buffered_);
            std::memcpy(buffer_ + buffered_, bytes, count);
            buffered_ += count;
            bytes += count;
            length -= count;
            if (buffered_ == sizeof buffer_) {
                Transform(buffer_);
                buffered_ = 0;
            }
        }
    }

    std::string Finish() {
        const uint64_t originalBits = bitCount_;
        const uint8_t one = 0x80;
        Update(&one, 1);
        const uint8_t zero = 0;
        while (buffered_ != 56) Update(&zero, 1);
        uint8_t length[8];
        for (unsigned index = 0; index < 8; ++index)
            length[7 - index] = uint8_t(originalBits >> (index * 8));
        Update(length, sizeof length);
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (uint32_t value : state_) out << std::setw(8) << value;
        return out.str();
    }

private:
    static uint32_t Rotate(uint32_t value, unsigned count) {
        return (value >> count) | (value << (32 - count));
    }

    void Transform(const uint8_t* block) {
        static constexpr uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (unsigned i = 0; i < 16; ++i) w[i] = BeU32(block + i * 4);
        for (unsigned i = 16; i < 64; ++i) {
            const uint32_t s0 = Rotate(w[i - 15], 7) ^ Rotate(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = Rotate(w[i - 2], 17) ^ Rotate(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a=state_[0],b=state_[1],c=state_[2],d=state_[3];
        uint32_t e=state_[4],f=state_[5],g=state_[6],h=state_[7];
        for (unsigned i = 0; i < 64; ++i) {
            const uint32_t s1=Rotate(e,6)^Rotate(e,11)^Rotate(e,25);
            const uint32_t ch=(e&f)^((~e)&g);
            const uint32_t t1=h+s1+ch+k[i]+w[i];
            const uint32_t s0=Rotate(a,2)^Rotate(a,13)^Rotate(a,22);
            const uint32_t maj=(a&b)^(a&c)^(b&c);
            const uint32_t t2=s0+maj;
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
        state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
    }

    uint32_t state_[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                          0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint8_t buffer_[64]{};
    size_t buffered_ = 0;
    uint64_t bitCount_ = 0;
};

class FdReader {
public:
    explicit FdReader(int fd) : fd_(fd) {
        struct stat info {};
        if (fd_ >= 0 && fstat(fd_, &info) == 0 && info.st_size >= 0)
            size_ = uint64_t(info.st_size);
    }

    ~FdReader() {
        if (fd_ >= 0) close(fd_);
    }

    bool valid() const { return fd_ >= 0 && size_ != 0; }
    uint64_t size() const { return size_; }

    bool ReadAt(uint64_t offset, void* destination, size_t length, std::string& error) const {
        if (offset > size_ || length > size_ - offset) {
            std::ostringstream out;
            out << "read at 0x" << std::hex << offset << " exceeds package size";
            error = out.str();
            return false;
        }
        uint8_t* bytes = static_cast<uint8_t*>(destination);
        size_t done = 0;
        while (done < length) {
            const ssize_t count = pread(fd_, bytes + done, length - done,
                                        off_t(offset + done));
            if (count <= 0) {
                error = std::string("package read failed: ") + std::strerror(errno);
                return false;
            }
            done += size_t(count);
        }
        return true;
    }

private:
    int fd_ = -1;
    uint64_t size_ = 0;
};

std::string Hex8(uint32_t value) {
    char text[9];
    std::snprintf(text, sizeof text, "%08X", value);
    return text;
}

std::string Fail(const std::string& reason) {
    return "[XBLA intake]\nINTAKE_STATUS: FAIL\nReason: " + reason + "\n";
}

bool SyncPath(const std::filesystem::path& path, bool directory) {
    const int flags = directory ? (O_RDONLY | O_DIRECTORY) : O_RDONLY;
    const int descriptor = open(path.c_str(), flags);
    if (descriptor < 0) return false;
    const bool success = fsync(descriptor) == 0;
    close(descriptor);
    return success;
}

bool CompleteInstallAt(const std::filesystem::path& root) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::is_regular_file(root / kCompleteName, ec) || ec ||
        !fs::is_regular_file(root / kManifestName, ec) || ec ||
        !fs::is_regular_file(root / "default.xex", ec) || ec)
        return false;
    if (fs::file_size(root / "default.xex", ec) != kExpectedXexSize || ec) return false;

    std::ifstream marker(root / kCompleteName, std::ios::binary);
    const std::string markerText((std::istreambuf_iterator<char>(marker)),
                                 std::istreambuf_iterator<char>());
    if (!marker || markerText.find(std::string("default_xex_sha256=") +
                                   kExpectedXexSha256) == std::string::npos)
        return false;

    std::ifstream xex(root / "default.xex", std::ios::binary);
    Sha256 digest;
    uint8_t buffer[64 * 1024];
    while (xex) {
        xex.read(reinterpret_cast<char*>(buffer), sizeof buffer);
        if (xex.gcount() > 0) digest.Update(buffer, size_t(xex.gcount()));
    }
    return xex.eof() && digest.Finish() == kExpectedXexSha256;
}

}  // namespace

bool IsCaseZeroInstalled(const std::string& installPath) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path root(installPath);
    if (CompleteInstallAt(root)) return true;

    // Recover the only two durable states that a process kill can leave around the
    // final rename. An incomplete staging tree is deliberately never accepted.
    for (const fs::path& candidate : {fs::path(installPath + ".installing"),
                                     fs::path(installPath + ".previous")}) {
        if (!CompleteInstallAt(candidate)) continue;
        fs::remove_all(root, ec);
        if (ec) return false;
        fs::rename(candidate, root, ec);
        return !ec && CompleteInstallAt(root);
    }
    return false;
}

std::string InstallCaseZeroPackage(int fd, const std::string& installPath,
                                   const InstallProgress& progress) {
    Sha256 selfTest;
    constexpr uint8_t kShaTest[] = {'a', 'b', 'c'};
    selfTest.Update(kShaTest, sizeof kShaTest);
    if (selfTest.Finish() !=
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
        close(fd);
        return Fail("internal SHA-256 self-test failed");
    }
    FdReader reader(fd);
    if (!reader.valid()) return Fail("cannot open or stat selected document");

    std::string error;
    uint8_t header[0x1000]{};
    if (!reader.ReadAt(0, header, sizeof header, error)) return Fail(error);

    std::string packageType;
    if (std::memcmp(header, "LIVE", 4) == 0) packageType = "LIVE";
    else if (std::memcmp(header, "CON ", 4) == 0) packageType = "CON";
    else if (std::memcmp(header, "PIRS", 4) == 0) packageType = "PIRS";
    else return Fail("not an Xbox 360 XContent package (LIVE/CON/PIRS magic missing)");

    const uint32_t headerSize = BeU32(header + kOffHeaderSize);
    const uint32_t contentType = BeU32(header + kOffContentType);
    const uint64_t contentSize = BeU64(header + kOffContentSize);
    const uint32_t mediaId = BeU32(header + kOffExecutionInfo);
    const uint32_t version = BeU32(header + kOffExecutionInfo + 4);
    const uint32_t baseVersion = BeU32(header + kOffExecutionInfo + 8);
    const uint32_t titleId = BeU32(header + kOffExecutionInfo + 12);
    const uint32_t volumeType = BeU32(header + kOffVolumeType);

    std::string displayName;
    for (uint32_t index = 0; index < 0x80; index += 2) {
        const uint16_t character = BeU16(header + kOffDisplayName + index);
        if (character == 0) break;
        displayName += character >= 0x20 && character < 0x7F ? char(character) : '?';
    }

    if (titleId != kExpectedTitleId)
        return Fail("wrong title ID " + Hex8(titleId) + " (expected 58410A8D)");
    if (contentType != kExpectedContentType)
        return Fail("wrong content type " + Hex8(contentType) + " (expected 000D0000)");
    if (volumeType != 0) return Fail("package is not an STFS volume");
    if (headerSize < 0x1000 || headerSize > reader.size())
        return Fail("invalid XContent header size");

    const uint8_t* descriptor = header + kOffVolumeDescriptor;
    if (descriptor[0] != 0x24) return Fail("bad STFS volume descriptor length");
    if ((descriptor[2] & 0x01) == 0)
        return Fail("read-write STFS hash layout is unsupported");
    const uint32_t tableBlockCount = LeU16(descriptor + 3);
    uint32_t tableBlockIndex = U24Le(descriptor + 5);
    if (tableBlockCount == 0 || tableBlockCount > 1024)
        return Fail("invalid STFS directory block count");

    const uint64_t baseOffset =
            (uint64_t(headerSize) + kBlockSize - 1) / kBlockSize * kBlockSize;
    auto nextBlock = [&](uint32_t blockIndex, uint32_t* next) {
        const uint64_t tableOffset =
                baseOffset + (BlockIndexToHashBlockNumber(blockIndex) << 12);
        const uint64_t entryOffset =
                tableOffset + (blockIndex % kBlocksPerHashLevel[0]) * 0x18;
        uint8_t info[4]{};
        if (!reader.ReadAt(entryOffset + 0x14, info, sizeof info, error)) return false;
        *next = BeU32(info) & 0xFFFFFF;
        return true;
    };

    std::vector<Entry> files;
    std::map<uint32_t, std::string> directories;
    std::set<uint32_t> directoryBlocks;
    uint32_t ordinal = 0;
    uint64_t totalBytes = 0;
    for (uint32_t table = 0; table < tableBlockCount; ++table) {
        if (!directoryBlocks.insert(tableBlockIndex).second)
            return Fail("directory block chain contains a loop");
        uint8_t block[kBlockSize]{};
        if (!reader.ReadAt(BlockIndexToOffset(baseOffset, tableBlockIndex), block,
                           sizeof block, error))
            return Fail(error);
        for (uint32_t item = 0; item < kEntriesPerDirBlock; ++item) {
            const uint8_t* entry = block + item * 0x40;
            if (entry[0] == 0) break;
            const uint8_t flags = entry[40];
            const uint32_t nameLength = flags & 0x3F;
            if (nameLength == 0 || nameLength > 40)
                return Fail("invalid STFS entry name length");
            const std::string name(reinterpret_cast<const char*>(entry), nameLength);
            if (!SafeName(name)) return Fail("unsafe STFS entry name");
            const uint16_t parent = BeU16(entry + 50);
            std::string parentPath;
            if (const auto found = directories.find(parent); found != directories.end())
                parentPath = found->second;
            else if (parent != 0xFFFF)
                return Fail("STFS entry references a missing parent directory");
            if ((flags & 0x80) != 0) {
                directories[ordinal] = parentPath + name + "/";
            } else {
                Entry file{parentPath + name, BeU32(entry + 52),
                           U24Le(entry + 47), U24Le(entry + 44)};
                totalBytes += file.length;
                files.push_back(std::move(file));
            }
            ++ordinal;
        }
        if (!nextBlock(tableBlockIndex, &tableBlockIndex)) return Fail(error);
        if (tableBlockIndex == kEndOfChain) break;
    }

    const Entry* defaultXex = nullptr;
    for (const Entry& file : files) {
        if (file.path == "default.xex") {
            defaultXex = &file;
            break;
        }
    }
    if (defaultXex == nullptr) return Fail("default.xex is missing from the package");
    if (defaultXex->length != kExpectedXexSize)
        return Fail("unsupported default.xex size " + std::to_string(defaultXex->length) +
                    " (expected 4538368)");

    namespace fs = std::filesystem;
    const fs::path finalRoot(installPath);
    const fs::path parent = finalRoot.parent_path();
    const fs::path staging = finalRoot.string() + ".installing";
    const fs::path backup = finalRoot.string() + ".previous";
    std::error_code ec;
    fs::create_directories(parent, ec);
    if (ec) return Fail("cannot create private install parent: " + ec.message());

    struct statvfs space {};
    if (statvfs(parent.c_str(), &space) != 0)
        return Fail(std::string("cannot query private storage: ") + std::strerror(errno));
    const uint64_t available = uint64_t(space.f_bavail) * uint64_t(space.f_frsize);
    if (totalBytes > UINT64_MAX - kFreeSpaceReserve ||
        available < totalBytes + kFreeSpaceReserve) {
        return Fail("not enough private storage (needs " +
                    std::to_string(totalBytes + kFreeSpaceReserve) + ", available " +
                    std::to_string(available) + ")");
    }

    fs::remove_all(staging, ec);
    ec.clear();
    if (!fs::create_directories(staging, ec) || ec)
        return Fail("cannot create private staging directory: " + ec.message());

    std::ofstream manifest(staging / kManifestName, std::ios::binary | std::ios::trunc);
    if (!manifest) {
        fs::remove_all(staging, ec);
        return Fail("cannot create private install manifest");
    }
    manifest << "RisingRecomp Case Zero package manifest\n"
             << "manifest_version=1\n"
             << "package_type=" << packageType << '\n'
             << "display_name=" << displayName << '\n'
             << "title_id=" << Hex8(titleId) << '\n'
             << "content_type=" << Hex8(contentType) << '\n'
             << "media_id=" << Hex8(mediaId) << '\n'
             << "version=" << Hex8(version) << '\n'
             << "base_version=" << Hex8(baseVersion) << '\n'
             << "package_bytes=" << reader.size() << '\n'
             << "declared_content_bytes=" << contentSize << '\n'
             << "file_count=" << files.size() << '\n'
             << "extracted_bytes=" << totalBytes << "\n\n[files]\n";

    uint64_t completedBytes = 0;
    uint8_t buffer[kBlockSize]{};
    for (const Entry& file : files) {
        const fs::path outputPath = staging / fs::path(file.path);
        fs::create_directories(outputPath.parent_path(), ec);
        if (ec) {
            error = "cannot create install directory: " + ec.message();
            break;
        }
        std::FILE* output = std::fopen(outputPath.c_str(), "wb");
        if (output == nullptr) {
            error = "cannot create private asset " + file.path;
            break;
        }
        Sha256 digest;
        uint32_t remaining = file.length;
        uint32_t currentBlock = file.startBlock;
        std::set<uint32_t> visited;
        bool magicValid = file.path != "default.xex";
        for (uint32_t index = 0;
             index < file.blockCount && remaining != 0 && currentBlock != kEndOfChain;
             ++index) {
            if (!visited.insert(currentBlock).second) {
                error = file.path + " block chain contains a loop";
                break;
            }
            const uint32_t count = std::min(remaining, kBlockSize);
            if (!reader.ReadAt(BlockIndexToOffset(baseOffset, currentBlock), buffer, count, error))
                break;
            if (file.path == "default.xex" && index == 0)
                magicValid = count >= 4 && std::memcmp(buffer, "XEX2", 4) == 0;
            if (std::fwrite(buffer, 1, count, output) != count) {
                error = "failed writing private asset " + file.path;
                break;
            }
            digest.Update(buffer, count);
            remaining -= count;
            completedBytes += count;
            if (progress) progress(completedBytes, totalBytes);
            if (remaining != 0 && !nextBlock(currentBlock, &currentBlock)) break;
        }
        if (error.empty() && remaining != 0) error = file.path + " block chain ended early";
        if (error.empty() && !magicValid) error = "extracted default.xex has no XEX2 magic";
        if (error.empty() && (std::fflush(output) != 0 || fsync(fileno(output)) != 0))
            error = "failed flushing private asset " + file.path;
        const bool closeOk = std::fclose(output) == 0;
        if (error.empty() && !closeOk) error = "failed closing private asset " + file.path;
        if (!error.empty()) break;

        const std::string hash = digest.Finish();
        if (file.path == "default.xex" && hash != kExpectedXexSha256) {
            error = "default.xex SHA-256 mismatch (unsupported Case Zero executable)";
            break;
        }
        manifest << file.length << '\t' << hash << '\t' << file.path << '\n';
    }
    manifest.flush();
    if (error.empty() && !manifest) error = "failed writing private install manifest";
    manifest.close();
    if (error.empty() && !SyncPath(staging / kManifestName, false))
        error = "failed flushing private install manifest";
    if (!error.empty()) {
        fs::remove_all(staging, ec);
        return Fail(error);
    }

    {
        std::ofstream marker(staging / kCompleteName, std::ios::binary | std::ios::trunc);
        marker << "RisingRecomp Case Zero install complete\n"
               << "title_id=58410A8D\n"
               << "default_xex_sha256=" << kExpectedXexSha256 << '\n'
               << "file_count=" << files.size() << '\n'
               << "installed_bytes=" << totalBytes << '\n';
        marker.flush();
        if (!marker) {
            marker.close();
            fs::remove_all(staging, ec);
            return Fail("cannot write completion marker");
        }
    }
    if (!SyncPath(staging / kCompleteName, false) || !SyncPath(staging, true)) {
        fs::remove_all(staging, ec);
        return Fail("cannot durably flush completion marker");
    }

    fs::remove_all(backup, ec);
    ec.clear();
    if (fs::exists(finalRoot, ec)) {
        fs::rename(finalRoot, backup, ec);
        if (ec) {
            fs::remove_all(staging, ec);
            return Fail("cannot preserve previous install: " + ec.message());
        }
    }
    fs::rename(staging, finalRoot, ec);
    if (ec) {
        std::error_code restoreError;
        if (fs::exists(backup, restoreError)) fs::rename(backup, finalRoot, restoreError);
        fs::remove_all(staging, restoreError);
        return Fail("cannot publish completed install: " + ec.message());
    }
    if (!SyncPath(parent, true))
        return Fail("install published but parent directory flush failed");
    fs::remove_all(backup, ec);

    std::ostringstream report;
    report << "[XBLA intake]\n"
           << "Package: " << packageType << " / STFS\n"
           << "Name: " << displayName << '\n'
           << "Title ID: " << Hex8(titleId) << " (Case Zero: PASS)\n"
           << "Content type: " << Hex8(contentType) << " (XBLA: PASS)\n"
           << "Media ID: " << Hex8(mediaId) << '\n'
           << "Version: " << Hex8(version) << " / base " << Hex8(baseVersion) << '\n'
           << "Package size: " << reader.size() << " bytes\n"
           << "Files mapped: " << files.size() << " (reference: 256)\n"
           << "Asset bytes mapped: " << totalBytes << '\n'
           << "Assets installed: " << files.size() << " files / " << totalBytes << " bytes\n"
           << "default.xex: PASS (size, XEX2, SHA-256)\n"
           << "Private manifest + per-file SHA-256: PASS\n"
           << "Atomic completion marker: PASS\n"
           << "INTAKE_STATUS: PASS\n"
           << "Persistent install ready; source XBLA is no longer required.\n"
           << "Execution remains disabled until Stage 5.\n";
    return report.str();
}
