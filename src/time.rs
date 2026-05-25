#[derive(Clone, Copy)]
pub struct ParsedTime {
    pub epoch_seconds: i64,
    pub hour: u8,
    pub weekday_monday0: u8,
}

pub fn parse_utc_timestamp(value: &str) -> Option<ParsedTime> {
    parse_utc_timestamp_bytes(value.as_bytes())
}

pub fn parse_utc_timestamp_bytes(bytes: &[u8]) -> Option<ParsedTime> {
    if bytes.len() < 20 {
        return None;
    }

    let year = parse_u32(bytes, 0, 4)? as i32;
    let month = parse_u32(bytes, 5, 7)? as u32;
    let day = parse_u32(bytes, 8, 10)? as u32;
    let hour = parse_u32(bytes, 11, 13)? as u8;
    let minute = parse_u32(bytes, 14, 16)? as i64;
    let second = parse_u32(bytes, 17, 19)? as i64;

    let days = days_from_civil(year, month, day);
    let epoch_seconds = days * 86_400 + hour as i64 * 3_600 + minute * 60 + second;
    let weekday_monday0 = (days + 3).rem_euclid(7) as u8;

    Some(ParsedTime {
        epoch_seconds,
        hour,
        weekday_monday0,
    })
}

fn parse_u32(bytes: &[u8], start: usize, end: usize) -> Option<u32> {
    let mut acc = 0u32;
    for &byte in bytes.get(start..end)? {
        if !byte.is_ascii_digit() {
            return None;
        }
        acc = acc * 10 + (byte - b'0') as u32;
    }
    Some(acc)
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let year = year - (month <= 2) as i32;
    let era = div_floor(year, 400);
    let yoe = year - era * 400;
    let month = month as i32;
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day as i32 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146_097 + doe - 719_468) as i64
}

fn div_floor(value: i32, divisor: i32) -> i32 {
    let quotient = value / divisor;
    let remainder = value % divisor;
    if remainder != 0 && (remainder > 0) != (divisor > 0) {
        quotient - 1
    } else {
        quotient
    }
}

#[cfg(test)]
mod tests {
    use super::parse_utc_timestamp;

    #[test]
    fn parses_weekday_with_monday_zero() {
        let parsed = parse_utc_timestamp("2026-03-11T18:45:53Z").unwrap();
        assert_eq!(parsed.hour, 18);
        assert_eq!(parsed.weekday_monday0, 2);
    }
}
