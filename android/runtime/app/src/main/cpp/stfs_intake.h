#pragma once

#include <string>

// Inspects a player-selected STFS file descriptor, writes a text-only manifest and
// extracts only default.xex into app-private storage. The function owns and closes fd.
std::string InspectCaseZeroPackage(int fd, const std::string& xexPath,
                                   const std::string& manifestPath);
