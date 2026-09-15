#include "stfs_intake.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
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

}  // namespace

std::string InspectCaseZeroPackage(int fd, const std::string& xexPath,
                                   const std::string& manifestPath) {
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

    std::ofstream manifest(manifestPath, std::ios::binary | std::ios::trunc);
    if (!manifest) return Fail("cannot create private package manifest");
    manifest << "RisingRecomp Case Zero package manifest\n"
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
    for (const Entry& file : files) manifest << file.length << '\t' << file.path << '\n';
    manifest.close();
    if (!manifest) return Fail("failed writing private package manifest");

    const std::string partialPath = xexPath + ".part";
    std::FILE* output = std::fopen(partialPath.c_str(), "wb");
    if (output == nullptr) return Fail("cannot create private default.xex");
    uint32_t remaining = defaultXex->length;
    uint32_t currentBlock = defaultXex->startBlock;
    std::set<uint32_t> visited;
    bool xexMagicValid = false;
    uint8_t buffer[kBlockSize]{};
    for (uint32_t index = 0;
         index < defaultXex->blockCount && remaining != 0 && currentBlock != kEndOfChain;
         ++index) {
        if (!visited.insert(currentBlock).second) {
            error = "default.xex block chain contains a loop";
            break;
        }
        const uint32_t count = remaining < kBlockSize ? remaining : kBlockSize;
        if (!reader.ReadAt(BlockIndexToOffset(baseOffset, currentBlock), buffer, count, error))
            break;
        if (index == 0) xexMagicValid = count >= 4 && std::memcmp(buffer, "XEX2", 4) == 0;
        if (std::fwrite(buffer, 1, count, output) != count) {
            error = "failed writing private default.xex";
            break;
        }
        remaining -= count;
        if (remaining != 0 && !nextBlock(currentBlock, &currentBlock)) break;
    }
    const bool closeOk = std::fclose(output) == 0;
    if (!error.empty() || remaining != 0 || !closeOk || !xexMagicValid) {
        std::remove(partialPath.c_str());
        if (error.empty()) {
            if (remaining != 0) error = "default.xex block chain ended early";
            else if (!xexMagicValid) error = "extracted file has no XEX2 magic";
            else error = "failed closing private default.xex";
        }
        return Fail(error);
    }
    if (std::rename(partialPath.c_str(), xexPath.c_str()) != 0) {
        std::remove(partialPath.c_str());
        return Fail(std::string("cannot finalize private default.xex: ") +
                    std::strerror(errno));
    }

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
           << "default.xex: PASS (4538368 bytes, XEX2)\n"
           << "Private manifest: PASS\n"
           << "INTAKE_STATUS: PASS\n"
           << "Game data mapped; execution remains disabled.\n";
    return report.str();
}
