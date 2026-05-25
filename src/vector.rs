use crate::index::DIMS;
use crate::request::FraudRequest;
use crate::time::{parse_utc_timestamp, parse_utc_timestamp_bytes};

const SCALE: f64 = 10_000.0;

pub fn vectorize(request: &FraudRequest<'_>) -> Option<[i16; DIMS]> {
    let requested_at = parse_utc_timestamp(&request.transaction.requested_at)?;
    let merchant_known = request
        .customer
        .known_merchants
        .iter()
        .any(|merchant| *merchant == request.merchant.id);

    let mut vector = [0i16; DIMS];
    vector[0] = scaled_clamped(request.transaction.amount / 10_000.0);
    vector[1] = scaled_clamped(request.transaction.installments as f64 / 12.0);
    vector[2] = scaled_clamped(amount_vs_average(
        request.transaction.amount,
        request.customer.avg_amount,
    ));
    vector[3] = scaled_clamped(requested_at.hour as f64 / 23.0);
    vector[4] = scaled_clamped(requested_at.weekday_monday0 as f64 / 6.0);

    if let Some(last) = &request.last_transaction {
        let last_at = parse_utc_timestamp(&last.timestamp)?;
        let minutes = (requested_at.epoch_seconds - last_at.epoch_seconds) as f64 / 60.0;
        vector[5] = scaled_clamped(minutes / 1_440.0);
        vector[6] = scaled_clamped(last.km_from_current / 1_000.0);
    } else {
        vector[5] = -10_000;
        vector[6] = -10_000;
    }

    vector[7] = scaled_clamped(request.terminal.km_from_home / 1_000.0);
    vector[8] = scaled_clamped(request.customer.tx_count_24h as f64 / 20.0);
    vector[9] = if request.terminal.is_online {
        10_000
    } else {
        0
    };
    vector[10] = if request.terminal.card_present {
        10_000
    } else {
        0
    };
    vector[11] = if merchant_known { 0 } else { 10_000 };
    vector[12] = scaled_clamped(mcc_risk(&request.merchant.mcc));
    vector[13] = scaled_clamped(request.merchant.avg_amount / 10_000.0);

    Some(vector)
}

pub fn vectorize_body(body: &[u8]) -> Option<[i16; DIMS]> {
    let transaction = find_after(body, 0, b"\"transaction\"")?;
    let (amount, _) = parse_number_after(body, transaction, b"\"amount\"")?;
    let (installments, _) = parse_number_after(body, transaction, b"\"installments\"")?;
    let (requested_at, _) = parse_string_after(body, transaction, b"\"requested_at\"")?;
    let requested_at = parse_utc_timestamp_bytes(requested_at)?;

    let customer = find_after(body, transaction, b"\"customer\"")?;
    let (customer_avg_amount, _) = parse_number_after(body, customer, b"\"avg_amount\"")?;
    let (tx_count_24h, _) = parse_number_after(body, customer, b"\"tx_count_24h\"")?;
    let (known_merchants, _) = parse_array_after(body, customer, b"\"known_merchants\"")?;

    let merchant = find_after(body, customer, b"\"merchant\"")?;
    let (merchant_id, _) = parse_string_after(body, merchant, b"\"id\"")?;
    let (mcc, _) = parse_string_after(body, merchant, b"\"mcc\"")?;
    let (merchant_avg_amount, _) = parse_number_after(body, merchant, b"\"avg_amount\"")?;

    let terminal = find_after(body, merchant, b"\"terminal\"")?;
    let (is_online, _) = parse_bool_after(body, terminal, b"\"is_online\"")?;
    let (card_present, _) = parse_bool_after(body, terminal, b"\"card_present\"")?;
    let (km_from_home, _) = parse_number_after(body, terminal, b"\"km_from_home\"")?;

    let last_transaction = find_after(body, terminal, b"\"last_transaction\"")?;
    let last_value = value_start(body, last_transaction)?;
    let merchant_known = array_contains_string(known_merchants, merchant_id);

    let mut vector = [0i16; DIMS];
    vector[0] = scaled_clamped(amount / 10_000.0);
    vector[1] = scaled_clamped(installments / 12.0);
    vector[2] = scaled_clamped(amount_vs_average(amount, customer_avg_amount));
    vector[3] = scaled_clamped(requested_at.hour as f64 / 23.0);
    vector[4] = scaled_clamped(requested_at.weekday_monday0 as f64 / 6.0);

    if body.get(last_value) == Some(&b'n') {
        vector[5] = -10_000;
        vector[6] = -10_000;
    } else {
        let (last_at, _) = parse_string_after(body, last_value, b"\"timestamp\"")?;
        let last_at = parse_utc_timestamp_bytes(last_at)?;
        let (km_from_current, _) = parse_number_after(body, last_value, b"\"km_from_current\"")?;
        let minutes = (requested_at.epoch_seconds - last_at.epoch_seconds) as f64 / 60.0;
        vector[5] = scaled_clamped(minutes / 1_440.0);
        vector[6] = scaled_clamped(km_from_current / 1_000.0);
    }

    vector[7] = scaled_clamped(km_from_home / 1_000.0);
    vector[8] = scaled_clamped(tx_count_24h / 20.0);
    vector[9] = if is_online { 10_000 } else { 0 };
    vector[10] = if card_present { 10_000 } else { 0 };
    vector[11] = if merchant_known { 0 } else { 10_000 };
    vector[12] = scaled_clamped(mcc_risk_bytes(mcc));
    vector[13] = scaled_clamped(merchant_avg_amount / 10_000.0);

    Some(vector)
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
fn scaled_clamped(value: f64) -> i16 {
    let clamped = value.clamp(0.0, 1.0);
    (clamped * SCALE).round() as i16
}

#[inline]
fn mcc_risk(mcc: &str) -> f64 {
    mcc_risk_bytes(mcc.as_bytes())
}

#[inline]
fn mcc_risk_bytes(mcc: &[u8]) -> f64 {
    match mcc {
        b"5411" => 0.15,
        b"5812" => 0.30,
        b"5912" => 0.20,
        b"5944" => 0.45,
        b"7801" => 0.80,
        b"7802" => 0.75,
        b"7995" => 0.85,
        b"4511" => 0.35,
        b"5311" => 0.25,
        b"5999" => 0.50,
        _ => 0.50,
    }
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

fn parse_number_after(body: &[u8], start: usize, key: &[u8]) -> Option<(f64, usize)> {
    let key_end = find_after(body, start, key)?;
    let cursor = value_start(body, key_end)?;
    let mut end = cursor;
    while matches!(body.get(end), Some(b'-' | b'.' | b'0'..=b'9')) {
        end += 1;
    }
    let value = unsafe { std::str::from_utf8_unchecked(body.get(cursor..end)?) };
    Some((value.parse().ok()?, end))
}

fn parse_bool_after(body: &[u8], start: usize, key: &[u8]) -> Option<(bool, usize)> {
    let key_end = find_after(body, start, key)?;
    let cursor = value_start(body, key_end)?;
    if body.get(cursor..cursor + 4) == Some(b"true") {
        Some((true, cursor + 4))
    } else if body.get(cursor..cursor + 5) == Some(b"false") {
        Some((false, cursor + 5))
    } else {
        None
    }
}

fn parse_string_after<'a>(body: &'a [u8], start: usize, key: &[u8]) -> Option<(&'a [u8], usize)> {
    let key_end = find_after(body, start, key)?;
    let mut cursor = value_start(body, key_end)?;
    if body.get(cursor) != Some(&b'"') {
        return None;
    }
    cursor += 1;
    let end = body.get(cursor..)?.iter().position(|byte| *byte == b'"')? + cursor;
    Some((body.get(cursor..end)?, end + 1))
}

fn parse_array_after<'a>(body: &'a [u8], start: usize, key: &[u8]) -> Option<(&'a [u8], usize)> {
    let key_end = find_after(body, start, key)?;
    let cursor = value_start(body, key_end)?;
    if body.get(cursor) != Some(&b'[') {
        return None;
    }
    let end = body.get(cursor..)?.iter().position(|byte| *byte == b']')? + cursor;
    Some((body.get(cursor + 1..end)?, end + 1))
}

fn array_contains_string(array: &[u8], needle: &[u8]) -> bool {
    let mut cursor = 0;
    while cursor < array.len() {
        if array[cursor] == b'"' {
            cursor += 1;
            let Some(relative_end) = array[cursor..].iter().position(|byte| *byte == b'"') else {
                return false;
            };
            let end = cursor + relative_end;
            if &array[cursor..end] == needle {
                return true;
            }
            cursor = end + 1;
        } else {
            cursor += 1;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use crate::request::FraudRequest;

    use super::{vectorize, vectorize_body};

    #[test]
    fn vectorizes_documented_legit_example() {
        let request: FraudRequest<'_> = serde_json::from_str(
            r#"{
              "id": "tx-1329056812",
              "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
              "customer": { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
              "merchant": { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
              "terminal": { "is_online": false, "card_present": true, "km_from_home": 29.23 },
              "last_transaction": null
            }"#,
        )
        .unwrap();

        let vector = vectorize(&request).unwrap();
        assert_eq!(
            vector,
            [41, 1667, 500, 7826, 3333, -10000, -10000, 292, 1500, 0, 10000, 0, 1500, 60]
        );
    }

    #[test]
    fn direct_body_parser_matches_documented_example() {
        let body = br#"{
          "id": "tx-1329056812",
          "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
          "customer": { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
          "merchant": { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
          "terminal": { "is_online": false, "card_present": true, "km_from_home": 29.23 },
          "last_transaction": null
        }"#;

        assert_eq!(
            vectorize_body(body).unwrap(),
            [41, 1667, 500, 7826, 3333, -10000, -10000, 292, 1500, 0, 10000, 0, 1500, 60]
        );
    }
}
