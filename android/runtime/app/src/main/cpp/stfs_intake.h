#pragma once

#include <cstdint>
#include <functional>
#include <string>

using InstallProgress = std::function<void(uint64_t completed, uint64_t total)>;

// Installs a player-selected STFS package into app-private persistent storage. The
// function owns and closes fd. It stages the complete install beside installPath and
// publishes it atomically only after every file and the manifest have been flushed.
std::string InstallCaseZeroPackage(int fd, const std::string& installPath,
                                   const InstallProgress& progress = {});

// Checks the durable completion marker without requiring the source XBLA again.
bool IsCaseZeroInstalled(const std::string& installPath);
