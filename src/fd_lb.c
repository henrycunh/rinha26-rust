#define _GNU_SOURCE

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/uio.h>
#include <time.h>
#include <unistd.h>

#define MAX_UPSTREAMS 32
#define BACKLOG 65535

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

typedef struct {
    char path[108];
    int fd;
    char byte;
    struct iovec iov;
    union {
        char buf[CMSG_SPACE(sizeof(int))];
        struct cmsghdr align;
    } control;
    struct msghdr msg;
    int* fd_slot;
} upstream_t;

static upstream_t upstreams[MAX_UPSTREAMS];
static int upstream_count = 0;
static uint32_t rr_next = 0;
static uint32_t rr_mask = 0;
static int backlog = BACKLOG;
static int quickack = 0;

static int env_enabled(const char* name, int default_value) {
    const char* value = getenv(name);
    if (value == NULL || value[0] == '\0') return default_value;
    return !(value[0] == '0' || value[0] == 'f' || value[0] == 'F');
}

static void lock_current_memory(void) {
    if (env_enabled("MLOCK_CURRENT", 0)) {
        (void)mlockall(MCL_CURRENT);
    }
}

static void sleep_ms(long ms) {
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000L;
    while (nanosleep(&ts, &ts) < 0 && errno == EINTR) {}
}

static void add_upstream(const char* begin, size_t len) {
    while (len > 0 && (*begin == ' ' || *begin == '\t')) {
        ++begin;
        --len;
    }
    while (len > 0 && (begin[len - 1] == ' ' || begin[len - 1] == '\t' ||
                       begin[len - 1] == '\n')) {
        --len;
    }
    if (UNLIKELY(len == 0 || upstream_count >= MAX_UPSTREAMS)) return;
    if (UNLIKELY(len >= sizeof(upstreams[upstream_count].path))) _Exit(1);

    upstream_t* u = &upstreams[upstream_count++];
    memset(u, 0, sizeof(*u));
    memcpy(u->path, begin, len);
    u->path[len] = '\0';
    u->fd = -1;
}

static void parse_upstreams(void) {
    const char* env = getenv("LB_FD_SOCKETS");
    if (UNLIKELY(env == NULL || env[0] == '\0')) env = getenv("UPSTREAMS");
    if (UNLIKELY(env == NULL || env[0] == '\0')) env = "/sockets/api1.sock,/sockets/api2.sock";

    const char* start = env;
    for (const char* p = env;; ++p) {
        if (*p == ',' || *p == '\0') {
            add_upstream(start, (size_t)(p - start));
            if (*p == '\0') break;
            start = p + 1;
        }
    }
    if (UNLIKELY(upstream_count == 0)) _Exit(1);
    uint32_t count = (uint32_t)upstream_count;
    if (LIKELY((count & (count - 1)) == 0)) rr_mask = count - 1;
}

static inline int upstream_index(uint32_t value) {
    if (LIKELY(rr_mask != 0)) return (int)(value & rr_mask);
    return (int)(value % (uint32_t)upstream_count);
}

static void init_upstream_msg(upstream_t* u) {
    u->byte = 1;
    u->iov.iov_base = &u->byte;
    u->iov.iov_len = 1;
    u->msg.msg_iov = &u->iov;
    u->msg.msg_iovlen = 1;
    u->msg.msg_control = u->control.buf;
    u->msg.msg_controllen = CMSG_SPACE(sizeof(int));

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&u->msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    u->msg.msg_controllen = cmsg->cmsg_len;
    u->fd_slot = (int*)CMSG_DATA(cmsg);
}

static int connect_once(const char* path) {
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (UNLIKELY(fd < 0)) return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (UNLIKELY(connect(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0)) {
        close(fd);
        return -1;
    }
    return fd;
}

static int connect_wait(const char* path) {
    for (;;) {
        int fd = connect_once(path);
        if (LIKELY(fd >= 0)) return fd;
        sleep_ms(5);
    }
}

static void connect_all(void) {
    for (int i = 0; i < upstream_count; ++i) {
        init_upstream_msg(&upstreams[i]);
        upstreams[i].fd = connect_wait(upstreams[i].path);
    }
}

static int reconnect_one(int idx) {
    if (upstreams[idx].fd >= 0) close(upstreams[idx].fd);
    upstreams[idx].fd = -1;
    for (int tries = 0; tries < 20; ++tries) {
        int fd = connect_once(upstreams[idx].path);
        if (LIKELY(fd >= 0)) {
            upstreams[idx].fd = fd;
            return 0;
        }
        sleep_ms(2);
    }
    return -1;
}

static inline int send_fd_once(upstream_t* upstream, int client_fd) {
    *upstream->fd_slot = client_fd;

    for (;;) {
        ssize_t sent = sendmsg(upstream->fd, &upstream->msg, MSG_NOSIGNAL);
        if (LIKELY(sent == 1)) return 0;
        if (UNLIKELY(sent < 0 && errno == EINTR)) continue;
        return -1;
    }
}

static int handoff(int idx, int client_fd) {
    upstream_t* upstream = &upstreams[idx];
    if (UNLIKELY(upstream->fd < 0) && reconnect_one(idx) != 0) return -1;
    if (LIKELY(send_fd_once(upstream, client_fd) == 0)) return 0;
    if (UNLIKELY(reconnect_one(idx) != 0)) return -1;
    return send_fd_once(upstream, client_fd);
}

static int listen_tcp(int port) {
    int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (UNLIKELY(fd < 0)) return -1;

    int one = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)port);
    if (UNLIKELY(bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0)) return -1;
    if (UNLIKELY(listen(fd, backlog) != 0)) return -1;
    return fd;
}

static inline void tune_client_fd(int fd) {
    int one = 1;
    (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    if (UNLIKELY(quickack)) {
        (void)setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));
    }
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);
    lock_current_memory();
    parse_upstreams();
    connect_all();

    int port = 9999;
    const char* port_env = getenv("PORT");
    if (UNLIKELY(port_env != NULL && port_env[0] != '\0')) port = atoi(port_env);

    const char* backlog_env = getenv("TCP_BACKLOG");
    if (UNLIKELY(backlog_env != NULL && backlog_env[0] != '\0')) backlog = atoi(backlog_env);
    if (UNLIKELY(backlog < 128)) backlog = 128;
    quickack = env_enabled("TCP_QUICKACK", 1);

    int server_fd = listen_tcp(port);
    if (UNLIKELY(server_fd < 0)) {
        perror("listen");
        return 1;
    }

    for (;;) {
        int client_fd = accept4(server_fd, NULL, NULL, SOCK_CLOEXEC | SOCK_NONBLOCK);
        if (UNLIKELY(client_fd < 0)) {
            if (errno == EINTR) continue;
            continue;
        }
        tune_client_fd(client_fd);

        int first = upstream_index(rr_next++);
        if (UNLIKELY(handoff(first, client_fd) != 0)) {
            for (int offset = 1; offset < upstream_count; ++offset) {
                if (LIKELY(handoff(upstream_index((uint32_t)(first + offset)), client_fd) == 0)) {
                    break;
                }
            }
        }
        close(client_fd);
    }
}
