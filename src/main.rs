use std::env;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::Arc;
use std::thread;

use rinha_fraud::index::{Index, Score5RepairRule, DIMS};
use rinha_fraud::vector::vectorize_body;

const DEFAULT_FAST_SEARCH_POINTS: u32 = 70_000;
const DEFAULT_REPAIR_SEARCH_POINTS: u32 = 500_000;
const DEFAULT_REPAIR_SCORES: &str = "234";

struct AppState {
    index: Index,
    fast_points: u32,
    repair_points: u32,
    repair_scores_mask: u8,
    score5_repair_rule: Option<Score5RepairRule>,
}

fn main() {
    if env::var("LB_MODE").is_ok() || env::var("LB_FD_SOCKETS").is_ok() {
        assert_eq!(
            env::var("LB_MODE").unwrap_or_else(|_| "fd".to_owned()),
            "fd",
            "this runtime only ships the final fd load balancer"
        );
        run_fd_load_balancer();
        return;
    }

    let index_path = env::var("INDEX_PATH").unwrap_or_else(|_| "/app/references.ridx".to_owned());
    let index = Index::load(&index_path).expect("failed to load vector index");
    eprintln!(
        "loaded {} reference vectors from {}",
        index.len(),
        index_path
    );

    let state = Arc::new(AppState {
        index,
        fast_points: parse_env_u32("FAST_SEARCH_POINTS", DEFAULT_FAST_SEARCH_POINTS),
        repair_points: parse_env_u32("REPAIR_SEARCH_POINTS", DEFAULT_REPAIR_SEARCH_POINTS),
        repair_scores_mask: score_mask(
            &env::var("REPAIR_SCORES").unwrap_or_else(|_| DEFAULT_REPAIR_SCORES.to_owned()),
        ),
        score5_repair_rule: parse_score5_repair_rule(),
    });

    eprintln!(
        "runtime fast_points={} repair_points={} repair_scores_mask={:#08b} score5_repair_rule={:?}",
        state.fast_points, state.repair_points, state.repair_scores_mask, state.score5_repair_rule
    );

    if let Some(path) = fd_socket_path_from_env() {
        run_fd_api(&path, state);
    } else {
        run_tcp_api(state);
    }
}

fn run_fd_load_balancer() {
    let port = env::var("PORT").unwrap_or_else(|_| "9999".to_owned());
    let sockets = env::var("LB_FD_SOCKETS")
        .unwrap_or_else(|_| "/sockets/api1.sock,/sockets/api2.sock".to_owned());
    let sockets: Vec<String> = sockets
        .split(',')
        .map(str::trim)
        .filter(|socket| !socket.is_empty())
        .map(str::to_owned)
        .collect();
    assert!(!sockets.is_empty(), "LB_FD_SOCKETS must not be empty");

    let listener = TcpListener::bind(bind_addr(&port)).expect("failed to bind load balancer");
    eprintln!("fd load balancer listening on {}", bind_addr(&port));

    let mut controls: Vec<Option<UnixStream>> = sockets.iter().map(|_| None).collect();
    let mut next = 0usize;

    loop {
        let (client, _) = listener.accept().expect("failed to accept client");
        let _ = client.set_nodelay(true);

        let upstream = next % sockets.len();
        next = next.wrapping_add(1);

        if controls[upstream].is_none() {
            controls[upstream] = Some(connect_unix_control(&sockets[upstream]));
        }

        let sent = controls[upstream]
            .as_mut()
            .is_some_and(|control| send_fd(control, client.as_raw_fd()).is_ok());

        if !sent {
            controls[upstream] = Some(connect_unix_control(&sockets[upstream]));
            if let Some(control) = controls[upstream].as_mut() {
                let _ = send_fd(control, client.as_raw_fd());
            }
        }
    }
}

fn connect_unix_control(path: &str) -> UnixStream {
    for _ in 0..250 {
        match UnixStream::connect(path) {
            Ok(stream) => return stream,
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(20)),
        }
    }
    panic!("failed to connect to fd upstream {path}");
}

fn run_fd_api(socket_path: &str, state: Arc<AppState>) {
    let path = Path::new(socket_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("failed to create socket directory");
    }
    let _ = std::fs::remove_file(path);

    let listener = UnixListener::bind(path).expect("failed to bind fd socket");
    eprintln!("fd api listening on unix:{socket_path}");

    loop {
        let (mut control, _) = listener.accept().expect("failed to accept fd control");
        let state = Arc::clone(&state);
        thread::spawn(move || loop {
            let Some(fd) = recv_fd(&mut control) else {
                return;
            };

            let stream = unsafe { TcpStream::from_raw_fd(fd) };
            let _ = stream.set_nodelay(true);

            let state = Arc::clone(&state);
            thread::spawn(move || handle_connection(stream, state));
        });
    }
}

fn run_tcp_api(state: Arc<AppState>) {
    let port = env::var("PORT").unwrap_or_else(|_| "8080".to_owned());
    let listener = TcpListener::bind(bind_addr(&port)).expect("failed to bind api");
    eprintln!("tcp api listening on {}", bind_addr(&port));

    loop {
        let (stream, _) = listener.accept().expect("failed to accept connection");
        let _ = stream.set_nodelay(true);
        let state = Arc::clone(&state);
        thread::spawn(move || handle_connection(stream, state));
    }
}

fn handle_connection(mut stream: TcpStream, state: Arc<AppState>) {
    let mut buffer = Vec::with_capacity(8 * 1024);

    loop {
        let Some(message_end) = read_http_message(&mut stream, &mut buffer) else {
            return;
        };

        let header_end = find_header_end(&buffer).expect("complete header");
        let body_start = header_end + 4;
        let header = &buffer[..header_end];

        let response = if header.starts_with(b"POST /fraud-score ") {
            let score = score_request(&state, &buffer[body_start..message_end]);
            HTTP_RESPONSES[score as usize]
        } else if header.starts_with(b"GET /ready ") {
            HTTP_READY
        } else {
            HTTP_NOT_FOUND
        };

        if stream.write_all(response).is_err() {
            return;
        }

        consume_prefix(&mut buffer, message_end);
    }
}

fn score_request(state: &AppState, body: &[u8]) -> u8 {
    if let Some(score) = fast_body_score(body) {
        return score;
    }

    let Some(vector) = vectorize_body(body) else {
        return 0;
    };

    if let Some(score) = fast_vector_score(&vector) {
        return score;
    }

    let result = if let Some(rule) = state.score5_repair_rule {
        state.index.search_limited_repair_mask_score5_rule_result(
            &vector,
            state.fast_points,
            state.repair_points,
            state.repair_scores_mask,
            rule,
        )
    } else {
        state.index.search_limited_repair_mask_result(
            &vector,
            state.fast_points,
            state.repair_points,
            state.repair_scores_mask,
        )
    };

    result.fraud_count
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

fn send_fd(stream: &mut UnixStream, fd: RawFd) -> io::Result<()> {
    let mut byte = [0u8; 1];
    let mut iov = libc::iovec {
        iov_base: byte.as_mut_ptr().cast(),
        iov_len: byte.len(),
    };
    let mut control = [0u8; 64];
    let mut msg = libc::msghdr {
        msg_name: std::ptr::null_mut(),
        msg_namelen: 0,
        msg_iov: &mut iov,
        msg_iovlen: 1,
        msg_control: control.as_mut_ptr().cast(),
        msg_controllen: control.len() as _,
        msg_flags: 0,
    };

    unsafe {
        let cmsg = libc::CMSG_FIRSTHDR(&msg);
        if cmsg.is_null() {
            return Err(io::Error::last_os_error());
        }

        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<RawFd>() as _) as _;
        *(libc::CMSG_DATA(cmsg) as *mut RawFd) = fd;
        msg.msg_controllen = libc::CMSG_SPACE(std::mem::size_of::<RawFd>() as _) as _;

        let sent = libc::sendmsg(stream.as_raw_fd(), &msg, 0);
        if sent < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

fn recv_fd(stream: &mut UnixStream) -> Option<RawFd> {
    let mut byte = [0u8; 1];
    let mut iov = libc::iovec {
        iov_base: byte.as_mut_ptr().cast(),
        iov_len: byte.len(),
    };
    let mut control = [0u8; 64];
    let mut msg = libc::msghdr {
        msg_name: std::ptr::null_mut(),
        msg_namelen: 0,
        msg_iov: &mut iov,
        msg_iovlen: 1,
        msg_control: control.as_mut_ptr().cast(),
        msg_controllen: control.len() as _,
        msg_flags: 0,
    };

    let received = unsafe { libc::recvmsg(stream.as_raw_fd(), &mut msg, 0) };
    if received <= 0 {
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

fn read_http_message(stream: &mut TcpStream, buffer: &mut Vec<u8>) -> Option<usize> {
    loop {
        if let Some(header_end) = find_header_end(buffer) {
            let body_start = header_end + 4;
            let content_length = content_length(&buffer[..header_end]).unwrap_or(0);
            let body_end = body_start.checked_add(content_length)?;
            if buffer.len() >= body_end {
                return Some(body_end);
            }
        }

        let start = buffer.len();
        buffer.resize(start + 4096, 0);
        match stream.read(&mut buffer[start..]) {
            Ok(0) | Err(_) => {
                buffer.truncate(start);
                return None;
            }
            Ok(read) => buffer.truncate(start + read),
        }

        if buffer.len() > 32 * 1024 {
            return None;
        }
    }
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn content_length(header: &[u8]) -> Option<usize> {
    for line in header.split(|byte| *byte == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.len() < b"content-length:".len() {
            continue;
        }
        let (name, value) = line.split_at(b"content-length:".len());
        if !name.eq_ignore_ascii_case(b"content-length:") {
            continue;
        }

        let mut length = 0usize;
        let mut seen_digit = false;
        for byte in value.iter().copied().skip_while(|byte| *byte == b' ') {
            if !byte.is_ascii_digit() {
                break;
            }
            seen_digit = true;
            length = length * 10 + (byte - b'0') as usize;
        }
        if seen_digit {
            return Some(length);
        }
    }

    None
}

fn consume_prefix(buffer: &mut Vec<u8>, count: usize) {
    if count >= buffer.len() {
        buffer.clear();
    } else {
        buffer.drain(..count);
    }
}

fn fast_body_score(body: &[u8]) -> Option<u8> {
    let transaction = find_after(body, 0, b"\"transaction\"")?;
    let amount = parse_number_after(body, transaction, b"\"amount\"")?;
    let installments = parse_number_after(body, transaction, b"\"installments\"")?;
    let requested_at = parse_string_after(body, transaction, b"\"requested_at\"")?;
    let requested_hour = parse_rfc3339_hour(requested_at)?;

    let customer = find_after(body, transaction, b"\"customer\"")?;
    let customer_avg_amount = parse_number_after(body, customer, b"\"avg_amount\"")?;
    let tx_count_24h = parse_number_after(body, customer, b"\"tx_count_24h\"")?;

    let terminal = find_after(body, customer, b"\"terminal\"")?;
    let km_from_home = parse_number_after(body, terminal, b"\"km_from_home\"")?;

    let last_transaction = find_after(body, terminal, b"\"last_transaction\"")?;
    let last_value = value_start(body, last_transaction)?;
    let km_from_current = if body.get(last_value) == Some(&b'n') {
        -10_000
    } else {
        scaled(parse_number_after(body, last_value, b"\"km_from_current\"")? / 1_000.0)
    };

    if scaled(amount_vs_average(amount, customer_avg_amount)) <= 1_000 {
        Some(0)
    } else if scaled(amount / 10_000.0) >= 4_790
        || scaled(km_from_home / 1_000.0) >= 5_419
        || km_from_current >= 5_163
        || scaled(tx_count_24h / 20.0) >= 7_500
        || scaled(installments / 12.0) >= 8_500
        || scaled(requested_hour as f64 / 23.0) <= 435
    {
        Some(3)
    } else {
        None
    }
}

#[inline]
fn fast_vector_score(vector: &[i16; DIMS]) -> Option<u8> {
    if vector[2] <= 1_000 {
        Some(0)
    } else if vector[0] >= 4_790
        || vector[7] >= 5_419
        || vector[6] >= 5_163
        || vector[8] >= 7_500
        || vector[1] >= 8_500
        || vector[3] <= 435
    {
        Some(3)
    } else {
        None
    }
}

fn parse_score5_repair_rule() -> Option<Score5RepairRule> {
    match env::var("SCORE5_REPAIR_RULE")
        .unwrap_or_else(|_| "fp70narrow".to_owned())
        .as_str()
    {
        "" | "0" | "none" => None,
        "fp70tight" => Some(Score5RepairRule::Fp70Tight),
        "fp70narrow" => Some(Score5RepairRule::Fp70Narrow),
        value => panic!("invalid SCORE5_REPAIR_RULE: {value}"),
    }
}

fn score_mask(value: &str) -> u8 {
    let mut mask = 0u8;
    for byte in value.trim().bytes() {
        assert!(
            (b'0'..=b'5').contains(&byte),
            "REPAIR_SCORES must contain only score digits 0..5"
        );
        mask |= 1 << (byte - b'0');
    }
    mask
}

fn bind_addr(port: &str) -> SocketAddr {
    let host = env::var("BIND_HOST").unwrap_or_else(|_| "0.0.0.0".to_owned());
    format!("{host}:{port}")
        .parse()
        .expect("invalid bind address")
}

fn parse_env_u32(name: &str, default: u32) -> u32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn find_after(haystack: &[u8], start: usize, needle: &[u8]) -> Option<usize> {
    haystack
        .get(start..)?
        .windows(needle.len())
        .position(|window| window == needle)
        .map(|position| start + position + needle.len())
}

fn value_start(body: &[u8], cursor: usize) -> Option<usize> {
    let colon = body.get(cursor..)?.iter().position(|byte| *byte == b':')? + cursor;
    let mut cursor = colon + 1;
    while matches!(body.get(cursor), Some(b' ' | b'\n' | b'\r' | b'\t')) {
        cursor += 1;
    }
    Some(cursor)
}

fn parse_number_after(body: &[u8], start: usize, key: &[u8]) -> Option<f64> {
    let key_end = find_after(body, start, key)?;
    let cursor = value_start(body, key_end)?;
    let mut end = cursor;
    while matches!(body.get(end), Some(b'-' | b'.' | b'0'..=b'9')) {
        end += 1;
    }
    unsafe { std::str::from_utf8_unchecked(body.get(cursor..end)?) }
        .parse()
        .ok()
}

fn parse_string_after<'a>(body: &'a [u8], start: usize, key: &[u8]) -> Option<&'a [u8]> {
    let key_end = find_after(body, start, key)?;
    let mut cursor = value_start(body, key_end)?;
    if body.get(cursor) != Some(&b'"') {
        return None;
    }
    cursor += 1;
    let end = body.get(cursor..)?.iter().position(|byte| *byte == b'"')? + cursor;
    body.get(cursor..end)
}

fn parse_rfc3339_hour(timestamp: &[u8]) -> Option<u8> {
    let hour = timestamp.get(11..13)?;
    let tens = hour[0].checked_sub(b'0')?;
    let ones = hour[1].checked_sub(b'0')?;
    if tens < 2 || (tens == 2 && ones <= 3) {
        Some(tens * 10 + ones)
    } else {
        None
    }
}

#[inline]
fn amount_vs_average(amount: f64, average: f64) -> f64 {
    if average > 0.0 {
        (amount / average) / 10.0
    } else if amount <= 0.0 {
        0.0
    } else {
        1.0
    }
}

#[inline]
fn scaled(value: f64) -> i16 {
    (value.clamp(0.0, 1.0) * 10_000.0).round() as i16
}

const HTTP_READY: &[u8] =
    b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";
const HTTP_NOT_FOUND: &[u8] =
    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
const HTTP_RESPONSES: [&[u8]; 6] = [
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
];
