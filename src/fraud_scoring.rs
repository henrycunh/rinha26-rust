use crate::residual_tree::{self, FEATURE_COUNT};

#[derive(Clone, Copy)]
struct FastParsedTime {
    epoch_seconds: i32,
}

#[derive(Clone, Copy)]
struct FastRequestedTime {
    date_epoch_seconds: i32,
    hour: u8,
}

struct FastFields<'a> {
    amount: f32,
    installments: u8,
    requested_at: &'a [u8],
    customer_avg: f32,
    tx_count_24h: u8,
    known_merchants: &'a [u8],
    merchant_id: &'a [u8],
    merchant_mcc: &'a [u8],
    merchant_avg_value_start: usize,
    is_online_value_start: usize,
    card_present_value_start: usize,
    km_from_home: f32,
    after_km_from_home: usize,
}

#[inline(always)]
pub fn score_fraud_body(body: &[u8]) -> Option<u8> {
    let fields = parse_fast_fields(body)?;

    let safe_avg = if fields.customer_avg <= 0.0 {
        1.0
    } else {
        fields.customer_avg
    };
    let amount_ratio = fields.amount / safe_avg;
    let mut merchant_known = false;
    let mut merchant_known_ready = false;
    let mut merchant_id_bits = 0u64;
    let mut merchant_id_ready = false;
    let mut merchant_mcc = 0u32;
    let mut merchant_mcc_ready = false;

    if fields.amount <= 500.0
        && amount_ratio <= 0.50001
        && fields.installments <= 3
        && fields.tx_count_24h <= 5
        && known_merchant_fast(
            fields.known_merchants,
            fields.merchant_id,
            &mut merchant_id_bits,
            &mut merchant_id_ready,
            &mut merchant_known,
            &mut merchant_known_ready,
        )
        && fields.km_from_home <= 50.0
        && is_safe_mcc_fast(
            fields.merchant_mcc,
            &mut merchant_mcc,
            &mut merchant_mcc_ready,
        )
    {
        return Some(0);
    }

    if fields.amount >= 5_000.0
        && fields.installments >= 5
        && fields.tx_count_24h >= 6
        && !known_merchant_fast(
            fields.known_merchants,
            fields.merchant_id,
            &mut merchant_id_bits,
            &mut merchant_id_ready,
            &mut merchant_known,
            &mut merchant_known_ready,
        )
        && fields.km_from_home >= 150.0
        && is_risky_mcc_fast(
            fields.merchant_mcc,
            &mut merchant_mcc,
            &mut merchant_mcc_ready,
        )
    {
        return Some(5);
    }

    let (requested_hour, requested_weekday_monday0, requested_time) =
        parse_requested_hour_weekday(fields.requested_at)?;

    let merchant_known = known_merchant_fast(
        fields.known_merchants,
        fields.merchant_id,
        &mut merchant_id_bits,
        &mut merchant_id_ready,
        &mut merchant_known,
        &mut merchant_known_ready,
    );
    let merchant_mcc = merchant_mcc_fast(
        fields.merchant_mcc,
        &mut merchant_mcc,
        &mut merchant_mcc_ready,
    );
    let (merchant_avg, _) = parse_number_at(body, fields.merchant_avg_value_start)?;
    let is_online = parse_bool_at(body, fields.is_online_value_start)?;
    let card_present = parse_bool_at(body, fields.card_present_value_start)?;
    let (last_delta_days, last_km) = parse_last_features(
        body,
        fields.after_km_from_home,
        fields.requested_at,
        requested_time,
    )?;

    let features: [f32; FEATURE_COUNT] = [
        clamp_upper1(fields.amount / 10_000.0),
        fields.installments as f32 / 12.0,
        clamp_upper1(amount_ratio / 10.0),
        requested_hour as f32 / 23.0,
        requested_weekday_monday0 as f32 / 6.0,
        last_delta_days,
        last_km,
        clamp_upper1(fields.km_from_home / 1_000.0),
        fields.tx_count_24h as f32 / 20.0,
        if is_online { 1.0 } else { 0.0 },
        if card_present { 1.0 } else { 0.0 },
        if merchant_known { 0.0 } else { 1.0 },
        mcc_risk_tag(merchant_mcc),
        clamp_upper1(merchant_avg / 10_000.0),
    ];

    let fraud = residual_tree::predict_features(&features);
    Some(if fraud { 5 } else { 0 })
}

#[inline(always)]
fn parse_last_features(
    body: &[u8],
    after_km_from_home: usize,
    requested_at: &[u8],
    requested_time: FastRequestedTime,
) -> Option<(f32, f32)> {
    let last_value = prefixed_value_start(body, after_km_from_home, b"},\"last_transaction\":")?;
    if body.get(last_value) == Some(&b'n') {
        return Some((-1.0, -1.0));
    }

    let requested = parse_requested_timestamp_tail(requested_at, requested_time)?;
    let mut cursor = last_value;
    let last_timestamp =
        parse_prefixed_fixed_string_field_from(body, &mut cursor, b"{\"timestamp\":\"", 20)?;
    let km_from_current =
        parse_prefixed_number_field_from(body, &mut cursor, b",\"km_from_current\":")?;
    let last = parse_iso_timestamp(last_timestamp)?;
    let delta_seconds = (requested.epoch_seconds - last.epoch_seconds).max(0) as f32;
    Some((
        clamp_upper1(delta_seconds / 60.0 / 1_440.0),
        clamp_upper1(km_from_current / 1_000.0),
    ))
}

#[inline(always)]
fn parse_fast_fields(body: &[u8]) -> Option<FastFields<'_>> {
    let mut cursor = compact_enter_transaction_object(body)?;

    let amount = parse_prefixed_number_field_from(body, &mut cursor, b"\"amount\":")?;
    let installments = parse_prefixed_integer_field_from(body, &mut cursor, b",\"installments\":")?;
    let requested_at =
        parse_prefixed_fixed_string_field_from(body, &mut cursor, b",\"requested_at\":\"", 20)?;

    cursor = prefixed_value_start(body, cursor, b"},\"customer\":{")?;
    let customer_avg = parse_prefixed_number_field_from(body, &mut cursor, b"\"avg_amount\":")?;
    let tx_count_24h = parse_prefixed_integer_field_from(body, &mut cursor, b",\"tx_count_24h\":")?;
    let known_merchants =
        parse_prefixed_array_field_from(body, &mut cursor, b",\"known_merchants\":[")?;

    cursor = prefixed_value_start(body, cursor, b"},\"merchant\":{")?;
    let merchant_id = parse_prefixed_fixed_string_field_from(body, &mut cursor, b"\"id\":\"", 8)?;
    let merchant_mcc =
        parse_prefixed_fixed_string_field_from(body, &mut cursor, b",\"mcc\":\"", 4)?;
    let merchant_avg_value_start = prefixed_value_start(body, cursor, b",\"avg_amount\":")?;
    cursor = find_byte(body, merchant_avg_value_start, b'}')?;

    cursor = prefixed_value_start(body, cursor, b"},\"terminal\":{")?;
    let is_online_value_start = prefixed_value_start(body, cursor, b"\"is_online\":")?;
    cursor = skip_bool_at(body, is_online_value_start)?;
    let card_present_value_start = prefixed_value_start(body, cursor, b",\"card_present\":")?;
    cursor = skip_bool_at(body, card_present_value_start)?;
    let km_from_home = parse_prefixed_number_field_from(body, &mut cursor, b",\"km_from_home\":")?;

    Some(FastFields {
        amount,
        installments,
        requested_at,
        customer_avg,
        tx_count_24h,
        known_merchants,
        merchant_id,
        merchant_mcc,
        merchant_avg_value_start,
        is_online_value_start,
        card_present_value_start,
        km_from_home,
        after_km_from_home: cursor,
    })
}

#[inline(always)]
fn compact_enter_transaction_object(body: &[u8]) -> Option<usize> {
    if !body.starts_with(b"{\"id\":") {
        return None;
    }
    let cursor = skip_string_at(body, b"{\"id\":".len())?;
    prefixed_value_start(body, cursor, b",\"transaction\":{")
}

#[inline(always)]
fn prefixed_value_start(body: &[u8], cursor: usize, prefix: &[u8]) -> Option<usize> {
    let prefix_end = cursor.checked_add(prefix.len())?;
    if body.get(cursor..prefix_end) == Some(prefix) {
        Some(prefix_end)
    } else {
        None
    }
}

#[inline(always)]
fn parse_prefixed_number_field_from(body: &[u8], cursor: &mut usize, prefix: &[u8]) -> Option<f32> {
    let value_start = prefixed_value_start(body, *cursor, prefix)?;
    let (value, next) = parse_number_at(body, value_start)?;
    *cursor = next;
    Some(value)
}

#[inline(always)]
fn parse_prefixed_integer_field_from(body: &[u8], cursor: &mut usize, prefix: &[u8]) -> Option<u8> {
    let value_start = prefixed_value_start(body, *cursor, prefix)?;
    let (value, next) = parse_integer_at(body, value_start)?;
    *cursor = next;
    Some(value)
}

#[inline(always)]
fn parse_prefixed_fixed_string_field_from<'a>(
    body: &'a [u8],
    cursor: &mut usize,
    prefix: &[u8],
    len: usize,
) -> Option<&'a [u8]> {
    let start = prefixed_value_start(body, *cursor, prefix)?;
    let end = start.checked_add(len)?;
    if end >= body.len() || unsafe { *body.get_unchecked(end) } != b'"' {
        None
    } else {
        *cursor = end + 1;
        Some(unsafe { body.get_unchecked(start..end) })
    }
}

#[inline(always)]
fn parse_prefixed_array_field_from<'a>(
    body: &'a [u8],
    cursor: &mut usize,
    prefix: &[u8],
) -> Option<&'a [u8]> {
    let start = prefixed_value_start(body, *cursor, prefix)?;
    let end = find_byte(body, start, b']')?;
    *cursor = end + 1;
    Some(unsafe { body.get_unchecked(start..end) })
}

#[inline(always)]
fn skip_string_at(body: &[u8], value_start: usize) -> Option<usize> {
    let mut cursor = value_start;
    if cursor >= body.len() || unsafe { *body.get_unchecked(cursor) } != b'"' {
        return None;
    }
    cursor += 1;
    let end = find_byte(body, cursor, b'"')?;
    Some(end + 1)
}

#[inline(always)]
fn find_byte(bytes: &[u8], mut cursor: usize, target: u8) -> Option<usize> {
    while cursor < bytes.len() {
        if unsafe { *bytes.get_unchecked(cursor) } == target {
            return Some(cursor);
        }
        cursor += 1;
    }
    None
}

#[inline(always)]
fn parse_bool_at(body: &[u8], value_start: usize) -> Option<bool> {
    if value_start >= body.len() {
        return None;
    }
    match unsafe { *body.get_unchecked(value_start) } {
        b't' => Some(true),
        b'f' => Some(false),
        _ => None,
    }
}

#[inline(always)]
fn skip_bool_at(body: &[u8], value_start: usize) -> Option<usize> {
    if value_start >= body.len() {
        return None;
    }
    match unsafe { *body.get_unchecked(value_start) } {
        b't' => Some(value_start + 4),
        b'f' => Some(value_start + 5),
        _ => None,
    }
}

#[inline(always)]
fn parse_requested_hour_weekday(bytes: &[u8]) -> Option<(u8, u8, FastRequestedTime)> {
    let year = parse_4_i32(bytes, 0);
    let month = parse_2_u8(bytes, 5);
    let day = parse_2_u8(bytes, 8);
    let hour = parse_2_u8(bytes, 11);
    if hour > 23 {
        return None;
    }

    let (date_epoch_seconds, weekday_monday0) =
        epoch_seconds_and_weekday_for_known_date(year, month, day)?;
    Some((
        hour,
        weekday_monday0,
        FastRequestedTime {
            date_epoch_seconds,
            hour,
        },
    ))
}

#[inline(always)]
fn parse_requested_timestamp_tail(
    bytes: &[u8],
    requested_time: FastRequestedTime,
) -> Option<FastParsedTime> {
    let minute = parse_2_u8(bytes, 14) as i32;
    let second = parse_2_u8(bytes, 17) as i32;
    if minute > 59 || second > 59 {
        return None;
    }

    let hour = requested_time.hour as i32;
    Some(FastParsedTime {
        epoch_seconds: requested_time.date_epoch_seconds + hour * 3_600 + minute * 60 + second,
    })
}

#[inline(always)]
fn parse_iso_timestamp(bytes: &[u8]) -> Option<FastParsedTime> {
    let year = parse_4_i32(bytes, 0);
    let month = parse_2_u8(bytes, 5);
    let day = parse_2_u8(bytes, 8);
    let hour = parse_2_u8(bytes, 11) as i32;
    let minute = parse_2_u8(bytes, 14) as i32;
    let second = parse_2_u8(bytes, 17) as i32;
    if hour > 23 || minute > 59 || second > 59 {
        return None;
    }

    Some(FastParsedTime {
        epoch_seconds: epoch_seconds_for_known_date(year, month, day)?
            + hour * 3_600
            + minute * 60
            + second,
    })
}

#[inline(always)]
fn parse_2_u8(bytes: &[u8], offset: usize) -> u8 {
    let tens = unsafe { *bytes.get_unchecked(offset) }.wrapping_sub(b'0');
    let ones = unsafe { *bytes.get_unchecked(offset + 1) }.wrapping_sub(b'0');
    debug_assert!(tens <= 9);
    debug_assert!(ones <= 9);
    tens * 10 + ones
}

#[inline(always)]
fn parse_4_i32(bytes: &[u8], offset: usize) -> i32 {
    let a = unsafe { *bytes.get_unchecked(offset) }.wrapping_sub(b'0');
    let b = unsafe { *bytes.get_unchecked(offset + 1) }.wrapping_sub(b'0');
    let c = unsafe { *bytes.get_unchecked(offset + 2) }.wrapping_sub(b'0');
    let d = unsafe { *bytes.get_unchecked(offset + 3) }.wrapping_sub(b'0');
    debug_assert!(a <= 9);
    debug_assert!(b <= 9);
    debug_assert!(c <= 9);
    debug_assert!(d <= 9);
    (a as i32) * 1000 + (b as i32) * 100 + (c as i32) * 10 + d as i32
}

#[inline(always)]
fn epoch_seconds_for_known_date(year: i32, month: u8, day: u8) -> Option<i32> {
    let (year_index, day_of_year) = known_date_day_of_year(year, month, day)?;
    Some(YEAR_START_EPOCH_SECONDS_2016_TO_2030[year_index] + day_of_year * 86_400)
}

#[inline(always)]
fn epoch_seconds_and_weekday_for_known_date(year: i32, month: u8, day: u8) -> Option<(i32, u8)> {
    let (year_index, day_of_year) = known_date_day_of_year(year, month, day)?;
    let epoch_seconds = YEAR_START_EPOCH_SECONDS_2016_TO_2030[year_index] + day_of_year * 86_400;
    let weekday_monday0 =
        ((YEAR_START_WEEKDAY_MONDAY0_2016_TO_2030[year_index] as i32 + day_of_year) % 7) as u8;
    Some((epoch_seconds, weekday_monday0))
}

#[inline(always)]
fn known_date_day_of_year(year: i32, month: u8, day: u8) -> Option<(usize, i32)> {
    if !(2016..=2030).contains(&year) {
        return None;
    }

    let month_index = month.wrapping_sub(1) as usize;
    if month_index >= 12 {
        return None;
    }

    let leap_year = matches!(year, 2016 | 2020 | 2024 | 2028);
    let month_days = if leap_year {
        LEAP_MONTH_DAYS[month_index]
    } else {
        COMMON_MONTH_DAYS[month_index]
    };
    if day == 0 || day > month_days {
        return None;
    }

    let year_index = (year - 2016) as usize;
    let month_offsets = if leap_year {
        LEAP_MONTH_OFFSETS
    } else {
        COMMON_MONTH_OFFSETS
    };
    let day_of_year = month_offsets[month_index] + day as i32 - 1;
    Some((year_index, day_of_year))
}

#[inline(always)]
fn parse_number_at(body: &[u8], value_start: usize) -> Option<(f32, usize)> {
    let mut cursor = value_start;
    let len = body.len();

    if cursor >= len {
        return None;
    }

    let first = unsafe { *body.get_unchecked(cursor) }.wrapping_sub(b'0');
    if first > 9 {
        return None;
    }
    let mut value = first as f32;
    cursor += 1;

    while cursor < len {
        let byte = unsafe { *body.get_unchecked(cursor) };
        let digit = byte.wrapping_sub(b'0');
        if digit > 9 {
            break;
        }
        value = value * 10.0 + digit as f32;
        cursor += 1;
    }

    if cursor < len && unsafe { *body.get_unchecked(cursor) } == b'.' {
        cursor += 1;
        let mut scale = 0.1f32;
        while cursor < len {
            let byte = unsafe { *body.get_unchecked(cursor) };
            let digit = byte.wrapping_sub(b'0');
            if digit > 9 {
                break;
            }
            value += digit as f32 * scale;
            scale *= 0.1;
            cursor += 1;
        }
    }

    Some((value, cursor))
}

#[inline(always)]
fn parse_integer_at(body: &[u8], value_start: usize) -> Option<(u8, usize)> {
    let mut cursor = value_start;
    let len = body.len();

    if cursor >= len {
        return None;
    }

    let first = unsafe { *body.get_unchecked(cursor) }.wrapping_sub(b'0');
    if first > 9 {
        return None;
    }
    let mut value = first;
    cursor += 1;

    if cursor >= len {
        return Some((value, cursor));
    }

    let second = unsafe { *body.get_unchecked(cursor) }.wrapping_sub(b'0');
    if second > 9 {
        return Some((value, cursor));
    }
    value = value * 10 + second;
    cursor += 1;

    debug_assert!(cursor >= len || unsafe { *body.get_unchecked(cursor) }.wrapping_sub(b'0') > 9);

    Some((value, cursor))
}

#[inline(always)]
fn merchant_id_u64(merchant_id: &[u8]) -> u64 {
    debug_assert_eq!(merchant_id.len(), 8);
    unsafe { std::ptr::read_unaligned(merchant_id.as_ptr().cast::<u64>()) }
}

#[inline(always)]
fn array_contains_merchant_id(array: &[u8], needle: u64) -> bool {
    let mut cursor = 0usize;
    while cursor + 10 <= array.len() {
        debug_assert_eq!(unsafe { *array.get_unchecked(cursor) }, b'"');
        debug_assert_eq!(unsafe { *array.get_unchecked(cursor + 9) }, b'"');
        let current =
            unsafe { std::ptr::read_unaligned(array.as_ptr().add(cursor + 1).cast::<u64>()) };
        if current == needle {
            return true;
        }
        cursor += 11;
    }

    false
}

#[inline(always)]
fn clamp_upper1(value: f32) -> f32 {
    if value >= 1.0 {
        1.0
    } else {
        value
    }
}

#[inline(always)]
fn mcc_tag(mcc: &[u8]) -> u32 {
    debug_assert_eq!(mcc.len(), 4);
    unsafe { std::ptr::read_unaligned(mcc.as_ptr().cast::<u32>()) }
}

#[inline(always)]
fn mcc_risk_tag(mcc: u32) -> f32 {
    match mcc {
        MCC_5411 => 0.15,
        MCC_5812 => 0.30,
        MCC_5912 => 0.20,
        MCC_5944 => 0.45,
        MCC_7801 => 0.80,
        MCC_7802 => 0.75,
        MCC_7995 => 0.85,
        MCC_4511 => 0.35,
        MCC_5311 => 0.25,
        _ => 0.50,
    }
}

#[inline(always)]
fn known_merchant_fast(
    known_merchants: &[u8],
    merchant_id: &[u8],
    merchant_id_bits: &mut u64,
    merchant_id_ready: &mut bool,
    merchant_known: &mut bool,
    merchant_known_ready: &mut bool,
) -> bool {
    if !*merchant_known_ready {
        if !*merchant_id_ready {
            *merchant_id_bits = merchant_id_u64(merchant_id);
            *merchant_id_ready = true;
        }
        *merchant_known = array_contains_merchant_id(known_merchants, *merchant_id_bits);
        *merchant_known_ready = true;
    }
    *merchant_known
}

#[inline(always)]
fn merchant_mcc_fast(mcc: &[u8], merchant_mcc: &mut u32, merchant_mcc_ready: &mut bool) -> u32 {
    if !*merchant_mcc_ready {
        *merchant_mcc = mcc_tag(mcc);
        *merchant_mcc_ready = true;
    }
    *merchant_mcc
}

#[inline(always)]
fn is_safe_mcc_fast(mcc: &[u8], merchant_mcc: &mut u32, merchant_mcc_ready: &mut bool) -> bool {
    matches!(
        merchant_mcc_fast(mcc, merchant_mcc, merchant_mcc_ready),
        MCC_5411 | MCC_5812 | MCC_5912 | MCC_5311
    )
}

#[inline(always)]
fn is_risky_mcc_fast(mcc: &[u8], merchant_mcc: &mut u32, merchant_mcc_ready: &mut bool) -> bool {
    matches!(
        merchant_mcc_fast(mcc, merchant_mcc, merchant_mcc_ready),
        MCC_7995 | MCC_7801 | MCC_7802
    )
}

const MCC_5411: u32 = u32::from_ne_bytes(*b"5411");
const MCC_5812: u32 = u32::from_ne_bytes(*b"5812");
const MCC_5912: u32 = u32::from_ne_bytes(*b"5912");
const MCC_5944: u32 = u32::from_ne_bytes(*b"5944");
const MCC_7801: u32 = u32::from_ne_bytes(*b"7801");
const MCC_7802: u32 = u32::from_ne_bytes(*b"7802");
const MCC_7995: u32 = u32::from_ne_bytes(*b"7995");
const MCC_4511: u32 = u32::from_ne_bytes(*b"4511");
const MCC_5311: u32 = u32::from_ne_bytes(*b"5311");

const YEAR_START_EPOCH_SECONDS_2016_TO_2030: [i32; 15] = [
    1_451_606_400,
    1_483_228_800,
    1_514_764_800,
    1_546_300_800,
    1_577_836_800,
    1_609_459_200,
    1_640_995_200,
    1_672_531_200,
    1_704_067_200,
    1_735_689_600,
    1_767_225_600,
    1_798_761_600,
    1_830_297_600,
    1_861_920_000,
    1_893_456_000,
];
const YEAR_START_WEEKDAY_MONDAY0_2016_TO_2030: [u8; 15] =
    [4, 6, 0, 1, 2, 4, 5, 6, 0, 2, 3, 4, 5, 0, 1];
const COMMON_MONTH_DAYS: [u8; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
const LEAP_MONTH_DAYS: [u8; 12] = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
const COMMON_MONTH_OFFSETS: [i32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
const LEAP_MONTH_OFFSETS: [i32; 12] = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335];
