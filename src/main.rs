use std::env;
use std::io;
use std::mem::MaybeUninit;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use rinha_fraud::fast_tree::score_fast_tree_body;

const DEFAULT_API_WORKERS: usize = 192;
#[cfg(target_os = "linux")]
const DEFAULT_TCP_BACKLOG: i32 = 8192;
const WORKER_STACK_BYTES: usize = 64 * 1024;
const DIRECT_READ_BUFFER_BYTES: usize = 1024;
const KEEP_ALIVE_READ_BUFFER_BYTES: usize = 1024;
const SCM_RIGHTS_CONTROL_BYTES: usize = 32;
#[cfg(target_os = "linux")]
const EPOLL_MAX_EVENTS: usize = 512;
#[cfg(target_os = "linux")]
const EPOLL_MAX_FDS: usize = 65_536;
#[cfg(target_os = "linux")]
const DEFAULT_EPOLL_PREALLOC_FDS: usize = 2_048;
#[cfg(target_os = "linux")]
const DEFAULT_EPOLL_PREALLOC_CONTROLS: usize = 256;
#[cfg(target_os = "linux")]
const EPOLL_LISTENER_TOKEN: u64 = u64::MAX;
#[cfg(target_os = "linux")]
const EPOLL_CONTROL_TOKEN_BIT: u64 = 1u64 << 63;
#[cfg(target_os = "linux")]
const EPOLL_CLIENT_FD_MASK: u64 = u32::MAX as u64;
#[cfg(target_os = "linux")]
const EPOLL_GENERATION_MASK: u32 = 0x7fff_ffff;

struct DirectHttpMessage {
    body_start: usize,
    message_end: usize,
}

#[cfg(target_os = "linux")]
struct EpollConn {
    fd: RawFd,
    generation: u32,
    active: bool,
    buffer: [MaybeUninit<u8>; KEEP_ALIVE_READ_BUFFER_BYTES],
    used: usize,
    pending: &'static [u8],
    pending_offset: usize,
    close_after_pending: bool,
}

#[cfg(target_os = "linux")]
impl EpollConn {
    fn new(fd: RawFd, generation: u32) -> Self {
        Self {
            fd,
            generation,
            active: true,
            buffer: [MaybeUninit::<u8>::uninit(); KEEP_ALIVE_READ_BUFFER_BYTES],
            used: 0,
            pending: b"",
            pending_offset: 0,
            close_after_pending: false,
        }
    }

    fn inactive() -> Self {
        Self {
            fd: -1,
            generation: 0,
            active: false,
            buffer: [MaybeUninit::<u8>::uninit(); KEEP_ALIVE_READ_BUFFER_BYTES],
            used: 0,
            pending: b"",
            pending_offset: 0,
            close_after_pending: false,
        }
    }

    fn prealloc() -> Self {
        let mut conn = Self::inactive();
        conn.touch_buffer_pages();
        conn
    }

    fn touch_buffer_pages(&mut self) {
        if KEEP_ALIVE_READ_BUFFER_BYTES != 0 {
            self.buffer[0].write(0);
            self.buffer[KEEP_ALIVE_READ_BUFFER_BYTES - 1].write(0);
        }
    }

    fn reset(&mut self, fd: RawFd, generation: u32) {
        self.fd = fd;
        self.generation = generation;
        self.active = true;
        self.used = 0;
        self.pending = b"";
        self.pending_offset = 0;
        self.close_after_pending = false;
    }

    fn deactivate(&mut self) {
        self.active = false;
        self.fd = -1;
        self.used = 0;
        self.pending = b"";
        self.pending_offset = 0;
        self.close_after_pending = false;
    }
}

#[cfg(target_os = "linux")]
struct EpollControl {
    fd: RawFd,
    generation: u32,
    byte: MaybeUninit<u8>,
    iov: libc::iovec,
    control: [MaybeUninit<u8>; SCM_RIGHTS_CONTROL_BYTES],
    msg: libc::msghdr,
}

#[cfg(target_os = "linux")]
impl EpollControl {
    fn new(fd: RawFd, generation: u32) -> Box<Self> {
        let mut control = Box::new(Self {
            fd,
            generation,
            byte: MaybeUninit::<u8>::uninit(),
            iov: libc::iovec {
                iov_base: std::ptr::null_mut(),
                iov_len: 1,
            },
            control: [MaybeUninit::<u8>::uninit(); SCM_RIGHTS_CONTROL_BYTES],
            msg: libc::msghdr {
                msg_name: std::ptr::null_mut(),
                msg_namelen: 0,
                msg_iov: std::ptr::null_mut(),
                msg_iovlen: 1,
                msg_control: std::ptr::null_mut(),
                msg_controllen: 0,
                msg_flags: 0,
            },
        });
        control.iov.iov_base = control.byte.as_mut_ptr().cast();
        control.msg.msg_iov = &mut control.iov;
        control.msg.msg_control = control.control.as_mut_ptr().cast();
        control.msg.msg_controllen = std::mem::size_of_val(&control.control) as _;
        control
    }

    fn prealloc() -> Box<Self> {
        Self::new(-1, 0)
    }

    fn reset(&mut self, fd: RawFd, generation: u32) {
        self.fd = fd;
        self.generation = generation;
        self.reset_msg();
    }

    fn deactivate(&mut self) {
        self.fd = -1;
    }

    #[inline(always)]
    fn reset_msg(&mut self) {
        self.msg.msg_controllen = std::mem::size_of_val(&self.control) as _;
        self.msg.msg_flags = 0;
    }
}

#[cfg(target_os = "linux")]
enum FdRecvResult {
    Fd(RawFd),
    WouldBlock,
    Closed,
}

fn main() {
    tune_process_runtime();
    run_main();
}

#[cfg(target_os = "linux")]
fn tune_process_runtime() {
    unsafe {
        let timer_slack_ns = parse_env_i32("TIMER_SLACK_NS", 1).max(1) as libc::c_ulong;
        let _ = libc::prctl(
            libc::PR_SET_TIMERSLACK,
            timer_slack_ns,
            0 as libc::c_ulong,
            0 as libc::c_ulong,
            0 as libc::c_ulong,
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn tune_process_runtime() {}

#[cfg(target_os = "linux")]
fn lock_current_memory() {
    if parse_env_bool("MLOCK_CURRENT", false) {
        unsafe {
            let _ = libc::mlockall(libc::MCL_CURRENT);
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn lock_current_memory() {}

fn run_main() {
    eprintln!("runtime local_tree=true");
    lock_current_memory();

    if let Some(path) = fd_socket_path_from_env() {
        run_fd_api(&path);
    } else {
        run_tcp_api();
    }
}

fn run_fd_api(socket_path: &str) {
    let path = Path::new(socket_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("failed to create socket directory");
    }
    let _ = std::fs::remove_file(path);

    let listener = UnixListener::bind(path).expect("failed to bind fd socket");
    eprintln!("fd api listening on unix:{socket_path}");

    #[cfg(target_os = "linux")]
    if api_epoll_default() {
        run_fd_api_epoll(listener);
        return;
    }

    if api_direct_fd_default() {
        run_fd_api_direct(listener);
        return;
    }

    let sender = start_connection_workers();
    loop {
        let (mut control, _) = listener.accept().expect("failed to accept fd control");
        let sender = sender.clone();
        spawn_named("fd-control", move || loop {
            let Some(fd) = recv_fd(&mut control) else {
                return;
            };

            let stream = unsafe { TcpStream::from_raw_fd(fd) };
            tune_tcp_stream(&stream);

            if sender.send(stream).is_err() {
                return;
            }
        });
    }
}

fn api_direct_fd_default() -> bool {
    parse_env_bool("API_DIRECT_FD", false)
}

#[cfg(target_os = "linux")]
fn api_epoll_default() -> bool {
    parse_env_bool("API_EPOLL", true)
}

fn run_fd_api_direct(listener: UnixListener) {
    eprintln!("fd api direct one-shot mode enabled");
    let mut control_id = 0usize;
    loop {
        let (mut control, _) = listener.accept().expect("failed to accept fd control");
        let name = format!("fd-direct-{control_id}");
        control_id = control_id.wrapping_add(1);
        spawn_named(&name, move || loop {
            let Some(fd) = recv_fd(&mut control) else {
                return;
            };

            handle_fd_once(fd);
        });
    }
}

#[cfg(target_os = "linux")]
fn run_fd_api_epoll(listener: UnixListener) {
    eprintln!("fd api epoll mode enabled");

    let listener_fd = listener.as_raw_fd();
    set_nonblocking_fd(listener_fd).expect("failed to set fd listener nonblocking");
    let epfd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
    if epfd < 0 {
        panic!("failed to create epoll: {}", io::Error::last_os_error());
    }

    epoll_add(
        epfd,
        listener_fd,
        EPOLL_LISTENER_TOKEN,
        (libc::EPOLLIN | libc::EPOLLRDHUP) as u32,
    )
    .expect("failed to add fd listener to epoll");

    let mut conns: Vec<Option<Box<EpollConn>>> = Vec::new();
    conns.resize_with(EPOLL_MAX_FDS, || None);
    prealloc_epoll_conns(&mut conns);
    let mut controls: Vec<Option<Box<EpollControl>>> = Vec::new();
    controls.resize_with(EPOLL_MAX_FDS, || None);
    prealloc_epoll_controls(&mut controls);
    let mut events = vec![libc::epoll_event { events: 0, u64: 0 }; EPOLL_MAX_EVENTS];
    lock_current_memory();

    loop {
        let ready =
            unsafe { libc::epoll_wait(epfd, events.as_mut_ptr(), EPOLL_MAX_EVENTS as i32, -1) };
        if ready < 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            panic!("epoll_wait failed: {error}");
        }

        for event in events.iter().take(ready as usize) {
            let token = event.u64;
            if token == EPOLL_LISTENER_TOKEN {
                accept_epoll_controls(listener_fd, epfd, &mut controls);
            } else if token & EPOLL_CONTROL_TOKEN_BIT != 0 {
                let control_token = token & !EPOLL_CONTROL_TOKEN_BIT;
                let control_fd = (control_token & EPOLL_CLIENT_FD_MASK) as RawFd;
                let generation = (control_token >> 32) as u32;
                drain_epoll_control(control_fd, generation, epfd, &mut controls, &mut conns);
            } else {
                let fd = (token & EPOLL_CLIENT_FD_MASK) as RawFd;
                let generation = (token >> 32) as u32;
                handle_epoll_client_event(fd, generation, event.events, epfd, &mut conns);
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn prealloc_epoll_conns(conns: &mut [Option<Box<EpollConn>>]) {
    let prealloc_fds =
        parse_env_usize("EPOLL_PREALLOC_FDS", DEFAULT_EPOLL_PREALLOC_FDS).min(conns.len());
    for conn in conns.iter_mut().take(prealloc_fds) {
        *conn = Some(Box::new(EpollConn::prealloc()));
    }
}

#[cfg(target_os = "linux")]
fn prealloc_epoll_controls(controls: &mut [Option<Box<EpollControl>>]) {
    let prealloc_controls =
        parse_env_usize("EPOLL_PREALLOC_CONTROLS", DEFAULT_EPOLL_PREALLOC_CONTROLS)
            .min(controls.len());
    for control in controls.iter_mut().take(prealloc_controls) {
        *control = Some(EpollControl::prealloc());
    }
}

#[cfg(target_os = "linux")]
fn accept_epoll_controls(
    listener_fd: RawFd,
    epfd: RawFd,
    controls: &mut [Option<Box<EpollControl>>],
) {
    loop {
        let control_fd = unsafe {
            libc::accept4(
                listener_fd,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
            )
        };
        if control_fd < 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return;
        }

        let index = control_fd as usize;
        if index >= controls.len() {
            unsafe {
                libc::close(control_fd);
            }
            continue;
        }

        let generation = controls[index]
            .as_ref()
            .map(|control| next_epoll_generation(control.generation))
            .unwrap_or(1);
        let token = epoll_control_token(control_fd, generation);

        if epoll_add(
            epfd,
            control_fd,
            token,
            (libc::EPOLLIN | libc::EPOLLRDHUP) as u32,
        )
        .is_err()
        {
            unsafe {
                libc::close(control_fd);
            }
        } else {
            if let Some(control) = controls[index].as_mut() {
                control.reset(control_fd, generation);
            } else {
                controls[index] = Some(EpollControl::new(control_fd, generation));
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn drain_epoll_control(
    control_fd: RawFd,
    generation: u32,
    epfd: RawFd,
    controls: &mut [Option<Box<EpollControl>>],
    conns: &mut [Option<Box<EpollConn>>],
) {
    let index = control_fd as usize;
    if index >= controls.len() || controls[index].is_none() {
        return;
    }
    if controls[index]
        .as_ref()
        .is_some_and(|control| control.fd != control_fd || control.generation != generation)
    {
        return;
    }

    loop {
        let control = unsafe { controls[index].as_mut().unwrap_unchecked() };
        match recv_fd_control(control) {
            FdRecvResult::Fd(client_fd) => register_epoll_client(client_fd, epfd, conns),
            FdRecvResult::WouldBlock => return,
            FdRecvResult::Closed => {
                epoll_del(epfd, control_fd);
                control.deactivate();
                unsafe {
                    libc::close(control_fd);
                }
                return;
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn recv_fd_control(control: &mut EpollControl) -> FdRecvResult {
    control.reset_msg();
    loop {
        let received = unsafe { libc::recvmsg(control.fd, &mut control.msg, libc::MSG_DONTWAIT) };
        if received > 0 {
            break;
        }
        if received == 0 {
            return FdRecvResult::Closed;
        }
        let error = io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::EINTR) => continue,
            Some(libc::EAGAIN) => return FdRecvResult::WouldBlock,
            _ => return FdRecvResult::Closed,
        }
    }

    unsafe {
        let cmsg = libc::CMSG_FIRSTHDR(&control.msg);
        if !cmsg.is_null()
            && (*cmsg).cmsg_level == libc::SOL_SOCKET
            && (*cmsg).cmsg_type == libc::SCM_RIGHTS
        {
            FdRecvResult::Fd(*(libc::CMSG_DATA(cmsg) as *const RawFd))
        } else {
            FdRecvResult::Closed
        }
    }
}

#[cfg(target_os = "linux")]
fn register_epoll_client(client_fd: RawFd, epfd: RawFd, conns: &mut [Option<Box<EpollConn>>]) {
    let index = client_fd as usize;
    if index >= conns.len() {
        unsafe {
            libc::close(client_fd);
        }
        return;
    }

    let generation = conns[index]
        .as_ref()
        .map(|conn| next_epoll_generation(conn.generation))
        .unwrap_or(1);
    let token = epoll_client_token(client_fd, generation);

    if epoll_add(
        epfd,
        client_fd,
        token,
        (libc::EPOLLIN | libc::EPOLLRDHUP) as u32,
    )
    .is_err()
    {
        unsafe {
            libc::close(client_fd);
        }
        return;
    }

    if let Some(conn) = conns[index].as_mut() {
        conn.reset(client_fd, generation);
    } else {
        conns[index] = Some(Box::new(EpollConn::new(client_fd, generation)));
    }
}

#[cfg(target_os = "linux")]
fn handle_epoll_client_event(
    fd: RawFd,
    generation: u32,
    events: u32,
    epfd: RawFd,
    conns: &mut [Option<Box<EpollConn>>],
) {
    let index = fd as usize;
    if index >= conns.len() {
        return;
    }

    let Some(conn) = conns[index].as_mut() else {
        return;
    };
    if !conn.active || conn.fd != fd || conn.generation != generation {
        return;
    }

    if events & (libc::EPOLLERR | libc::EPOLLHUP) as u32 != 0 {
        close_epoll_client(fd, generation, epfd, conns);
        return;
    }

    if events & libc::EPOLLOUT as u32 != 0 && !flush_epoll_response(conn, epfd) {
        close_epoll_client(fd, generation, epfd, conns);
        return;
    }

    if events & libc::EPOLLIN as u32 != 0 && !read_epoll_client(conn, epfd) {
        close_epoll_client(fd, generation, epfd, conns);
        return;
    }

    if events & libc::EPOLLRDHUP as u32 != 0 {
        close_epoll_client(fd, generation, epfd, conns);
    }
}

#[cfg(target_os = "linux")]
fn read_epoll_client(conn: &mut EpollConn, epfd: RawFd) -> bool {
    let mut responded = false;
    loop {
        while conn.pending.is_empty() {
            let initialized = unsafe { initialized_prefix(&conn.buffer, conn.used) };
            let Some(message) = http_message_bounds_direct(initialized)
                .filter(|message| conn.used >= message.message_end)
            else {
                break;
            };

            let (response, close_after_write) = {
                let method = unsafe { *initialized.get_unchecked(0) };
                if method == b'P' {
                    let score =
                        score_request(&initialized[message.body_start..message.message_end]);
                    (http_response_keep_alive(score), false)
                } else if method == b'G' {
                    (HTTP_READY, false)
                } else {
                    (HTTP_NOT_FOUND, true)
                }
            };

            consume_initialized_prefix(&mut conn.buffer, &mut conn.used, message.message_end);
            if !send_epoll_response(conn, response, close_after_write, epfd) {
                return false;
            }
            responded = true;
        }

        if !conn.pending.is_empty() {
            return true;
        }
        if responded {
            return true;
        }

        if conn.used >= conn.buffer.len() {
            return false;
        }

        let read = unsafe {
            libc::recv(
                conn.fd,
                conn.buffer.as_mut_ptr().add(conn.used).cast(),
                conn.buffer.len() - conn.used,
                0,
            )
        };
        if read == 0 {
            return false;
        }
        if read < 0 {
            let error = io::Error::last_os_error();
            match error.raw_os_error() {
                Some(libc::EINTR) => continue,
                Some(libc::EAGAIN) => return true,
                _ => return false,
            }
        } else {
            conn.used += read as usize;
        }
    }
}

#[cfg(target_os = "linux")]
fn send_epoll_response(
    conn: &mut EpollConn,
    response: &'static [u8],
    close_after_write: bool,
    epfd: RawFd,
) -> bool {
    let sent = loop {
        let sent = unsafe {
            libc::send(
                conn.fd,
                response.as_ptr().cast(),
                response.len(),
                libc::MSG_NOSIGNAL,
            )
        };
        if sent >= 0 {
            break sent;
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::EINTR) {
            continue;
        }
        if error.raw_os_error() == Some(libc::EAGAIN) {
            conn.pending = response;
            conn.pending_offset = 0;
            conn.close_after_pending = close_after_write;
            return epoll_mod(
                epfd,
                conn.fd,
                epoll_client_token(conn.fd, conn.generation),
                (libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLRDHUP) as u32,
            )
            .is_ok();
        }
        return false;
    };

    let sent = sent as usize;
    if sent == response.len() {
        return !close_after_write;
    }
    if sent == 0 {
        return false;
    }

    conn.pending = response;
    conn.pending_offset = sent;
    conn.close_after_pending = close_after_write;
    epoll_mod(
        epfd,
        conn.fd,
        epoll_client_token(conn.fd, conn.generation),
        (libc::EPOLLIN | libc::EPOLLOUT | libc::EPOLLRDHUP) as u32,
    )
    .is_ok()
}

#[cfg(target_os = "linux")]
fn flush_epoll_response(conn: &mut EpollConn, epfd: RawFd) -> bool {
    while conn.pending_offset < conn.pending.len() {
        let sent = unsafe {
            libc::send(
                conn.fd,
                conn.pending.as_ptr().add(conn.pending_offset).cast(),
                conn.pending.len() - conn.pending_offset,
                libc::MSG_NOSIGNAL,
            )
        };
        if sent < 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return error.raw_os_error() == Some(libc::EAGAIN);
        }
        if sent == 0 {
            return false;
        }
        conn.pending_offset += sent as usize;
    }

    let close_after_pending = conn.close_after_pending;
    conn.pending = b"";
    conn.pending_offset = 0;
    conn.close_after_pending = false;

    if close_after_pending {
        return false;
    }

    epoll_mod(
        epfd,
        conn.fd,
        epoll_client_token(conn.fd, conn.generation),
        (libc::EPOLLIN | libc::EPOLLRDHUP) as u32,
    )
    .is_ok()
}

#[cfg(target_os = "linux")]
fn close_epoll_client(
    fd: RawFd,
    generation: u32,
    epfd: RawFd,
    conns: &mut [Option<Box<EpollConn>>],
) {
    let index = fd as usize;
    if index >= conns.len() {
        return;
    }

    let Some(conn) = conns[index].as_mut() else {
        return;
    };
    if !conn.active || conn.fd != fd || conn.generation != generation {
        return;
    }

    epoll_del(epfd, fd);
    conn.deactivate();
    unsafe {
        libc::close(fd);
    }
}

fn run_tcp_api() {
    let port = env::var("PORT").unwrap_or_else(|_| "8080".to_owned());
    let listener = bind_tcp_listener(&port).expect("failed to bind api");
    eprintln!("tcp api listening on {}", bind_addr(&port));

    let sender = start_connection_workers();
    loop {
        let (stream, _) = listener.accept().expect("failed to accept connection");
        tune_tcp_stream(&stream);
        if sender.send(stream).is_err() {
            return;
        }
    }
}

fn start_connection_workers() -> mpsc::Sender<TcpStream> {
    let workers = parse_env_usize("API_WORKERS", DEFAULT_API_WORKERS).max(1);
    let (sender, receiver) = mpsc::channel::<TcpStream>();
    let receiver = Arc::new(Mutex::new(receiver));

    for worker in 0..workers {
        let receiver = Arc::clone(&receiver);
        spawn_named(&format!("api-worker-{worker}"), move || loop {
            let stream = {
                let receiver = receiver.lock().expect("connection receiver poisoned");
                receiver.recv()
            };

            let Ok(stream) = stream else {
                return;
            };
            handle_connection(stream);
        });
    }

    sender
}

fn handle_fd_once(fd: RawFd) {
    let mut buffer = [MaybeUninit::<u8>::uninit(); DIRECT_READ_BUFFER_BYTES];
    let Some((message, bytes_read)) = read_http_message_fd_once(fd, &mut buffer) else {
        unsafe {
            libc::close(fd);
        }
        return;
    };

    let buffer = unsafe { initialized_prefix(&buffer, bytes_read) };
    let method = unsafe { *buffer.get_unchecked(0) };
    let response = if method == b'P' {
        let score = score_request(&buffer[message.body_start..message.message_end]);
        http_response_close(score)
    } else if method == b'G' {
        HTTP_READY_CLOSE
    } else {
        HTTP_NOT_FOUND
    };

    let _ = send_response_fd(fd, response);
    unsafe {
        libc::close(fd);
    }
}

fn read_http_message_fd_once(
    fd: RawFd,
    buffer: &mut [MaybeUninit<u8>],
) -> Option<(DirectHttpMessage, usize)> {
    let mut filled = 0usize;
    while filled < buffer.len() {
        let read = unsafe {
            libc::recv(
                fd,
                buffer.as_mut_ptr().add(filled).cast(),
                buffer.len() - filled,
                0,
            )
        };
        if read == 0 {
            return None;
        }
        if read < 0 {
            if io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return None;
        }

        filled += read as usize;
        let initialized = unsafe { initialized_prefix(buffer, filled) };

        if let Some(message) = http_message_bounds_direct(initialized) {
            if filled >= message.message_end {
                return Some((message, filled));
            }
        }
    }
    None
}

#[inline(always)]
fn http_message_bounds_direct(buffer: &[u8]) -> Option<DirectHttpMessage> {
    if buffer.is_empty() {
        return None;
    }

    let (body_start, content_length) = if unsafe { *buffer.get_unchecked(0) } == b'G' {
        let (_, body_start) = find_header_body_boundary(buffer)?;
        (body_start, 0)
    } else {
        fast_known_post_header_bounds(buffer).or_else(|| find_post_header_bounds(buffer))?
    };
    let message_end = body_start.checked_add(content_length)?;
    Some(DirectHttpMessage {
        body_start,
        message_end,
    })
}

#[inline(always)]
fn find_header_body_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    let len = buffer.len();
    let mut cursor = 3usize;
    while cursor < len {
        if unsafe {
            *buffer.get_unchecked(cursor) == b'\n'
                && *buffer.get_unchecked(cursor - 1) == b'\r'
                && *buffer.get_unchecked(cursor - 2) == b'\n'
                && *buffer.get_unchecked(cursor - 3) == b'\r'
        } {
            return Some((cursor - 3, cursor + 1));
        }
        cursor += 1;
    }
    None
}

#[inline(always)]
fn fast_known_post_header_bounds(buffer: &[u8]) -> Option<(usize, usize)> {
    const PREFIX: &[u8] = b"POST /fraud-score HTTP/1.1\r\nHost: localhost:9999\r\nUser-Agent: Grafana k6/2.0.0\r\nContent-Length: ";
    const SUFFIX: &[u8] = b"\r\nContent-Type: application/json\r\n\r\n";

    if buffer.len() < PREFIX.len() || buffer.get(0..PREFIX.len()) != Some(PREFIX) {
        return None;
    }

    let mut cursor = PREFIX.len();
    let mut content_length = 0usize;
    let mut has_digit = false;
    while cursor < buffer.len() {
        let digit = unsafe { *buffer.get_unchecked(cursor) }.wrapping_sub(b'0');
        if digit > 9 {
            break;
        }
        has_digit = true;
        content_length = content_length
            .checked_mul(10)?
            .checked_add(digit as usize)?;
        cursor += 1;
    }

    let suffix_end = cursor.checked_add(SUFFIX.len())?;
    if !has_digit || suffix_end > buffer.len() || buffer.get(cursor..suffix_end) != Some(SUFFIX) {
        return None;
    }

    Some((suffix_end, content_length))
}

#[inline(always)]
fn find_post_header_bounds(buffer: &[u8]) -> Option<(usize, usize)> {
    const HEADER_LEN: usize = b"Content-Length:".len();
    let len = buffer.len();
    let mut cursor = 1usize;
    let mut content_length = None;

    while cursor < len {
        if unsafe {
            *buffer.get_unchecked(cursor) == b'\n' && *buffer.get_unchecked(cursor - 1) == b'\r'
        } {
            if cursor >= 3
                && unsafe {
                    *buffer.get_unchecked(cursor - 2) == b'\n'
                        && *buffer.get_unchecked(cursor - 3) == b'\r'
                }
            {
                return Some((cursor + 1, content_length?));
            }

            let line_start = cursor + 1;
            if line_start + HEADER_LEN <= len {
                let first = unsafe { *buffer.get_unchecked(line_start) };
                if (first == b'C' || first == b'c') && content_length_at(buffer, line_start) {
                    content_length = parse_header_usize(buffer, line_start + HEADER_LEN);
                }
            }
        }

        cursor += 1;
    }

    None
}

unsafe fn initialized_prefix(buffer: &[MaybeUninit<u8>], len: usize) -> &[u8] {
    std::slice::from_raw_parts(buffer.as_ptr().cast(), len)
}

#[inline(always)]
fn send_response_fd(fd: RawFd, response: &[u8]) -> io::Result<()> {
    loop {
        let sent = unsafe {
            libc::send(
                fd,
                response.as_ptr().cast(),
                response.len(),
                libc::MSG_NOSIGNAL,
            )
        };
        if sent < 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return Err(error);
        }
        return if sent as usize == response.len() {
            Ok(())
        } else if sent == 0 {
            Err(io::Error::new(io::ErrorKind::WriteZero, "socket closed"))
        } else {
            send_all_fd(fd, &response[sent as usize..])
        };
    }
}

fn send_all_fd(fd: RawFd, mut response: &[u8]) -> io::Result<()> {
    while !response.is_empty() {
        let sent = unsafe {
            libc::send(
                fd,
                response.as_ptr().cast(),
                response.len(),
                libc::MSG_NOSIGNAL,
            )
        };
        if sent < 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return Err(error);
        }
        if sent == 0 {
            return Err(io::Error::new(io::ErrorKind::WriteZero, "socket closed"));
        }
        response = &response[sent as usize..];
    }
    Ok(())
}

fn handle_connection(stream: TcpStream) {
    let mut buffer = [MaybeUninit::<u8>::uninit(); KEEP_ALIVE_READ_BUFFER_BYTES];
    let mut used = 0usize;
    let fd = stream.as_raw_fd();

    loop {
        let Some(message) = read_http_message(fd, &mut buffer, &mut used) else {
            return;
        };

        let initialized = unsafe { initialized_prefix(&buffer, used) };
        let method = unsafe { *initialized.get_unchecked(0) };
        let response = if method == b'P' {
            let score = score_request(&initialized[message.body_start..message.message_end]);
            http_response_keep_alive(score)
        } else if method == b'G' {
            HTTP_READY
        } else {
            HTTP_NOT_FOUND
        };

        if send_response_fd(fd, response).is_err() {
            return;
        }

        consume_initialized_prefix(&mut buffer, &mut used, message.message_end);
    }
}

fn score_request(body: &[u8]) -> u8 {
    score_fast_tree_body(body).unwrap_or(0)
}

fn fd_socket_path_from_env() -> Option<String> {
    if let Ok(path) = env::var("FD_SOCKET_PATH") {
        let path = path.trim();
        if !path.is_empty() {
            return Some(path.to_owned());
        }
    }

    let dir = env::var("FD_SOCKET_DIR").ok()?;
    let dir = dir.trim();
    if dir.is_empty() {
        return None;
    }

    let hostname = env::var("HOSTNAME").unwrap_or_else(|_| "api".to_owned());
    Some(format!("{}/{}.sock", dir.trim_end_matches('/'), hostname))
}

fn recv_fd(stream: &mut UnixStream) -> Option<RawFd> {
    let mut byte = MaybeUninit::<u8>::uninit();
    let mut iov = libc::iovec {
        iov_base: byte.as_mut_ptr().cast(),
        iov_len: 1,
    };
    let mut control = [MaybeUninit::<u8>::uninit(); SCM_RIGHTS_CONTROL_BYTES];
    let mut msg = libc::msghdr {
        msg_name: std::ptr::null_mut(),
        msg_namelen: 0,
        msg_iov: &mut iov,
        msg_iovlen: 1,
        msg_control: control.as_mut_ptr().cast(),
        msg_controllen: std::mem::size_of_val(&control) as _,
        msg_flags: 0,
    };

    loop {
        let received = unsafe { libc::recvmsg(stream.as_raw_fd(), &mut msg, 0) };
        if received > 0 {
            break;
        }
        if received == 0 {
            return None;
        }
        let error = io::Error::last_os_error();
        if matches!(error.raw_os_error(), Some(libc::EINTR) | Some(libc::EAGAIN)) {
            continue;
        }
        return None;
    }

    unsafe {
        let mut cmsg = libc::CMSG_FIRSTHDR(&msg);
        while !cmsg.is_null() {
            if (*cmsg).cmsg_level == libc::SOL_SOCKET && (*cmsg).cmsg_type == libc::SCM_RIGHTS {
                return Some(*(libc::CMSG_DATA(cmsg) as *const RawFd));
            }
            cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
        }
    }

    None
}

#[inline(always)]
fn read_http_message(
    fd: RawFd,
    buffer: &mut [MaybeUninit<u8>],
    used: &mut usize,
) -> Option<DirectHttpMessage> {
    loop {
        let initialized = unsafe { initialized_prefix(buffer, *used) };
        if let Some(message) = http_message_bounds_direct(initialized) {
            if *used >= message.message_end {
                return Some(message);
            }
        }

        if *used >= buffer.len() {
            return None;
        }

        let read = unsafe {
            libc::recv(
                fd,
                buffer.as_mut_ptr().add(*used).cast(),
                buffer.len() - *used,
                0,
            )
        };
        if read == 0 {
            return None;
        }
        if read < 0 {
            if io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return None;
        }
        *used += read as usize;
    }
}

#[inline(always)]
fn content_length_at(header: &[u8], cursor: usize) -> bool {
    const UPPER_PREFIX: u64 = u64::from_ne_bytes(*b"Content-");
    const UPPER_SUFFIX: u64 = u64::from_ne_bytes(*b"-Length:");
    const LOWER_PREFIX: u64 = u64::from_ne_bytes(*b"content-");
    const LOWER_SUFFIX: u64 = u64::from_ne_bytes(*b"-length:");

    let ptr = unsafe { header.as_ptr().add(cursor) };
    let prefix = unsafe { std::ptr::read_unaligned(ptr.cast::<u64>()) };
    let suffix = unsafe { std::ptr::read_unaligned(ptr.add(7).cast::<u64>()) };
    if unsafe { *ptr } == b'C' {
        prefix == UPPER_PREFIX && suffix == UPPER_SUFFIX
    } else {
        prefix == LOWER_PREFIX && suffix == LOWER_SUFFIX
    }
}

#[inline(always)]
fn parse_header_usize(line: &[u8], cursor: usize) -> Option<usize> {
    let len = line.len();
    if cursor + 4 >= len || unsafe { *line.get_unchecked(cursor) } != b' ' {
        return None;
    }

    let a = unsafe { *line.get_unchecked(cursor + 1) }.wrapping_sub(b'0');
    let b = unsafe { *line.get_unchecked(cursor + 2) }.wrapping_sub(b'0');
    let c = unsafe { *line.get_unchecked(cursor + 3) }.wrapping_sub(b'0');
    let next = unsafe { *line.get_unchecked(cursor + 4) }.wrapping_sub(b'0');
    if a > 9 || b > 9 || c > 9 || next <= 9 {
        return None;
    }
    Some((a as usize) * 100 + (b as usize) * 10 + c as usize)
}

#[inline(always)]
fn consume_initialized_prefix(buffer: &mut [MaybeUninit<u8>], used: &mut usize, count: usize) {
    if count >= *used {
        *used = 0;
    } else {
        unsafe {
            std::ptr::copy(
                buffer.as_ptr().add(count),
                buffer.as_mut_ptr(),
                *used - count,
            );
        }
        *used -= count;
    }
}

#[inline(always)]
fn http_response_close(score: u8) -> &'static [u8] {
    if score == 0 {
        HTTP_SCORE0_CLOSE
    } else {
        HTTP_SCORE5_CLOSE
    }
}

#[inline(always)]
fn http_response_keep_alive(score: u8) -> &'static [u8] {
    if score == 0 {
        HTTP_SCORE0
    } else {
        HTTP_SCORE5
    }
}

fn bind_addr(port: &str) -> SocketAddr {
    let host = env::var("BIND_HOST").unwrap_or_else(|_| "0.0.0.0".to_owned());
    format!("{host}:{port}")
        .parse()
        .expect("invalid bind address")
}

fn bind_tcp_listener(port: &str) -> io::Result<TcpListener> {
    bind_tcp_listener_addr(bind_addr(port))
}

#[cfg(target_os = "linux")]
fn bind_tcp_listener_addr(addr: SocketAddr) -> io::Result<TcpListener> {
    let SocketAddr::V4(addr) = addr else {
        return TcpListener::bind(addr);
    };

    unsafe {
        let fd = libc::socket(
            libc::AF_INET,
            libc::SOCK_STREAM | libc::SOCK_CLOEXEC,
            libc::IPPROTO_TCP,
        );
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }

        let bind_result = (|| {
            set_sockopt_int(fd, libc::SOL_SOCKET, libc::SO_REUSEADDR, 1)?;
            set_sockopt_int(fd, libc::SOL_SOCKET, libc::SO_REUSEPORT, 1)?;

            let sockaddr = libc::sockaddr_in {
                sin_family: libc::AF_INET as libc::sa_family_t,
                sin_port: addr.port().to_be(),
                sin_addr: libc::in_addr {
                    s_addr: u32::from_ne_bytes(addr.ip().octets()),
                },
                sin_zero: [0; 8],
            };

            let result = libc::bind(
                fd,
                &sockaddr as *const _ as *const libc::sockaddr,
                std::mem::size_of::<libc::sockaddr_in>() as libc::socklen_t,
            );
            if result < 0 {
                return Err(io::Error::last_os_error());
            }

            let backlog = parse_env_i32("TCP_BACKLOG", DEFAULT_TCP_BACKLOG).max(128);
            if libc::listen(fd, backlog) < 0 {
                return Err(io::Error::last_os_error());
            }

            Ok(())
        })();

        if let Err(error) = bind_result {
            libc::close(fd);
            return Err(error);
        }

        Ok(TcpListener::from_raw_fd(fd))
    }
}

#[cfg(not(target_os = "linux"))]
fn bind_tcp_listener_addr(addr: SocketAddr) -> io::Result<TcpListener> {
    TcpListener::bind(addr)
}

#[cfg(target_os = "linux")]
fn parse_env_i32(name: &str, default: i32) -> i32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn parse_env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn parse_env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(default)
}

fn spawn_named(name: &str, f: impl FnOnce() + Send + 'static) {
    thread::Builder::new()
        .name(name.to_owned())
        .stack_size(WORKER_STACK_BYTES)
        .spawn(f)
        .expect("failed to spawn worker");
}

fn tune_tcp_stream(stream: &TcpStream) {
    let _ = stream.set_nodelay(true);
    tune_tcp_fd(stream.as_raw_fd());
}

#[cfg(target_os = "linux")]
fn tune_tcp_fd(fd: RawFd) {
    unsafe {
        let _ = set_sockopt_int(fd, libc::IPPROTO_TCP, libc::TCP_QUICKACK, 1);

        let busy_poll = parse_env_i32("TCP_BUSY_POLL_US", 50).max(0);
        if busy_poll > 0 {
            let _ = set_sockopt_int(fd, libc::SOL_SOCKET, libc::SO_BUSY_POLL, busy_poll);
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn tune_tcp_fd(_fd: RawFd) {}

#[cfg(target_os = "linux")]
fn set_nonblocking_fd(fd: RawFd) -> io::Result<()> {
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFL, 0);
        if flags < 0 {
            return Err(io::Error::last_os_error());
        }
        if libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn epoll_add(epfd: RawFd, fd: RawFd, token: u64, events: u32) -> io::Result<()> {
    let mut event = libc::epoll_event { events, u64: token };
    let result = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fd, &mut event) };
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(target_os = "linux")]
#[inline(always)]
fn epoll_client_token(fd: RawFd, generation: u32) -> u64 {
    ((generation as u64) << 32) | (fd as u32 as u64)
}

#[cfg(target_os = "linux")]
#[inline(always)]
fn next_epoll_generation(current: u32) -> u32 {
    let next = (current.wrapping_add(1) & EPOLL_GENERATION_MASK).max(1);
    next
}

#[cfg(target_os = "linux")]
#[inline(always)]
fn epoll_control_token(fd: RawFd, generation: u32) -> u64 {
    EPOLL_CONTROL_TOKEN_BIT | epoll_client_token(fd, generation)
}

#[cfg(target_os = "linux")]
fn epoll_mod(epfd: RawFd, fd: RawFd, token: u64, events: u32) -> io::Result<()> {
    let mut event = libc::epoll_event { events, u64: token };
    let result = unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_MOD, fd, &mut event) };
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn epoll_del(epfd: RawFd, fd: RawFd) {
    let mut event = libc::epoll_event { events: 0, u64: 0 };
    unsafe {
        let _ = libc::epoll_ctl(epfd, libc::EPOLL_CTL_DEL, fd, &mut event);
    }
}

#[cfg(target_os = "linux")]
unsafe fn set_sockopt_int(
    fd: RawFd,
    level: libc::c_int,
    option: libc::c_int,
    value: libc::c_int,
) -> io::Result<()> {
    let result = libc::setsockopt(
        fd,
        level,
        option,
        &value as *const _ as *const _,
        std::mem::size_of::<libc::c_int>() as libc::socklen_t,
    );
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

const HTTP_READY: &[u8] = b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok";
const HTTP_READY_CLOSE: &[u8] = b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok";
const HTTP_NOT_FOUND: &[u8] =
    b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n";
const HTTP_SCORE0: &[u8] =
    b"HTTP/1.1 200 OK\r\ncontent-length: 33\r\n\r\n{\"approved\":true,\"fraud_score\":0}";
const HTTP_SCORE5: &[u8] =
    b"HTTP/1.1 200 OK\r\ncontent-length: 34\r\n\r\n{\"approved\":false,\"fraud_score\":1}";
const HTTP_SCORE0_CLOSE: &[u8] =
    b"HTTP/1.1 200 OK\r\ncontent-length: 33\r\n\r\n{\"approved\":true,\"fraud_score\":0}";
const HTTP_SCORE5_CLOSE: &[u8] =
    b"HTTP/1.1 200 OK\r\ncontent-length: 34\r\n\r\n{\"approved\":false,\"fraud_score\":1}";
