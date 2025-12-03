/**
 * Isolate some handy functions out of Mooncake Transfer Engine main code to
 * simplify dependencies.
 */

#include <arpa/inet.h>
#include <bits/stdint-uintn.h>
#include <errno.h>
#include <glog/logging.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static inline std::vector<std::string> parseDeviceList(const char *env_var) {
    std::vector<std::string> devices;
    if (!env_var || strlen(env_var) == 0) return devices;

    std::string devices_str(env_var);
    size_t start = 0;
    size_t end = devices_str.find(',');

    while (end != std::string::npos) {
        std::string device = devices_str.substr(start, end - start);
        // Trim whitespace
        device.erase(0, device.find_first_not_of(" \t"));
        device.erase(device.find_last_not_of(" \t") + 1);
        if (!device.empty()) devices.push_back(device);

        start = end + 1;
        end = devices_str.find(',', start);
    }

    // Last device
    std::string device = devices_str.substr(start);
    device.erase(0, device.find_first_not_of(" \t"));
    device.erase(device.find_last_not_of(" \t") + 1);
    if (!device.empty()) devices.push_back(device);

    return devices;
}

static inline std::vector<std::string> findLocalIpv4Addresses() {
    std::vector<std::string> ips;
    struct ifaddrs *ifaddr, *ifa;

    if (getifaddrs(&ifaddr) == -1) {
        LOG(ERROR) << "getifaddrs failed";
        return ips;
    }

    // Parse device filter from environment variable
    auto device_filter = parseDeviceList(getenv("FLEX_TCP_DEVICE"));

    if (device_filter.empty()) {
        // No filter specified, use all devices except loopback
        for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
            if (ifa->ifa_addr == nullptr) continue;
            if (ifa->ifa_addr->sa_family != AF_INET) continue;
            if (strcmp(ifa->ifa_name, "lo") == 0) continue;

            char host[NI_MAXHOST];
            if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host,
                            NI_MAXHOST, nullptr, 0, NI_NUMERICHOST) == 0) {
                ips.push_back(host);
            }
        }
    } else {
        // Filter by specified devices in order
        for (const auto &device_name : device_filter) {
            for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
                if (ifa->ifa_addr == nullptr) continue;
                if (ifa->ifa_addr->sa_family != AF_INET) continue;
                if (strcmp(ifa->ifa_name, device_name.c_str()) != 0) continue;

                char host[NI_MAXHOST];
                if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host,
                                NI_MAXHOST, nullptr, 0, NI_NUMERICHOST) == 0) {
                    ips.push_back(host);
                }
            }
        }
    }

    freeifaddrs(ifaddr);
    return ips;
}

static inline std::vector<std::string> findLocalIpv6Addresses() {
    std::vector<std::string> ips;
    struct ifaddrs *ifaddr, *ifa;

    if (getifaddrs(&ifaddr) == -1) {
        LOG(ERROR) << "getifaddrs failed";
        return ips;
    }

    // Parse device filter from environment variable
    auto device_filter = parseDeviceList(getenv("FLEX_TCP_DEVICE"));

    if (device_filter.empty()) {
        // No filter specified, use all devices except loopback
        for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
            if (ifa->ifa_addr == nullptr) continue;
            if (ifa->ifa_addr->sa_family != AF_INET6) continue;
            if (strcmp(ifa->ifa_name, "lo") == 0) continue;

            char host[NI_MAXHOST];
            if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in6), host,
                            NI_MAXHOST, nullptr, 0, NI_NUMERICHOST) == 0) {
                ips.push_back(host);
            }
        }
    } else {
        // Filter by specified devices in order
        for (const auto &device_name : device_filter) {
            for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
                if (ifa->ifa_addr == nullptr) continue;
                if (ifa->ifa_addr->sa_family != AF_INET6) continue;
                if (strcmp(ifa->ifa_name, device_name.c_str()) != 0) continue;

                char host[NI_MAXHOST];
                if (getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in6),
                                host, NI_MAXHOST, nullptr, 0,
                                NI_NUMERICHOST) == 0) {
                    ips.push_back(host);
                }
            }
        }
    }

    freeifaddrs(ifaddr);
    return ips;
}

static inline uint16_t findAvailableTcpPort(int &sockfd, bool use_ipv6) {
    static std::random_device rand_gen;
    std::uniform_int_distribution rand_dist;
    const int min_port = 15000;
    const int max_port = 17000;
    const int max_attempts = 500;

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        int port = min_port + rand_dist(rand_gen) % (max_port - min_port + 1);
        int on = 1;

        // Create socket
        sockfd = socket(use_ipv6 ? AF_INET6 : AF_INET, SOCK_STREAM, 0);
        if (sockfd == -1) continue;

        // Set socket options
        struct timeval timeout {
            .tv_sec = 1, .tv_usec = 0
        };
        if (setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                       sizeof(timeout))) {
            goto cleanup;
        }
        if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on))) {
            goto cleanup;
        }

        // Bind to port
        if (use_ipv6) {
            sockaddr_in6 bind_address;
            memset(&bind_address, 0, sizeof(sockaddr_in6));
            bind_address.sin6_family = AF_INET6;
            bind_address.sin6_port = htons(port);
            bind_address.sin6_addr = in6addr_any;
            if (bind(sockfd, (sockaddr *)&bind_address, sizeof(sockaddr_in6)) <
                0) {
                goto cleanup;
            }
        } else {
            sockaddr_in bind_address;
            memset(&bind_address, 0, sizeof(sockaddr_in));
            bind_address.sin_family = AF_INET;
            bind_address.sin_port = htons(port);
            bind_address.sin_addr.s_addr = INADDR_ANY;
            if (bind(sockfd, (sockaddr *)&bind_address, sizeof(sockaddr_in)) <
                0) {
                goto cleanup;
            }
        }

        return port;

    cleanup:
        close(sockfd);
        sockfd = -1;
    }
    return 0;
}

static inline ssize_t readFully(int fd, void *buf, size_t len) {
    char *pos = (char *)buf;
    size_t nbytes = len;
    while (nbytes) {
        ssize_t rc = read(fd, pos, nbytes);
        if (rc < 0 && (errno == EAGAIN || errno == EINTR))
            continue;
        else if (rc < 0) {
            LOG(ERROR) << "Socket read failed";
            return rc;
        } else if (rc == 0) {
            LOG(ERROR) << "Socket read incompleted: expected " << len
                       << " bytes, actual " << len - nbytes << " bytes";
            return len - nbytes;
        }
        pos += rc;
        nbytes -= rc;
    }
    return len;
}

static inline ssize_t writeFully(int fd, const void *buf, size_t len) {
    char *pos = (char *)buf;
    size_t nbytes = len;
    while (nbytes) {
        ssize_t rc = write(fd, pos, nbytes);
        if (rc < 0 && (errno == EAGAIN || errno == EINTR))
            continue;
        else if (rc < 0) {
            LOG(ERROR) << "Socket write failed";
            return rc;
        } else if (rc == 0) {
            LOG(ERROR) << "Socket write incompleted: expected " << len
                       << " bytes, actual " << len - nbytes << " bytes";
            return len - nbytes;
        }
        pos += rc;
        nbytes -= rc;
    }
    return len;
}
