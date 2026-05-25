use std::env;
use std::io::{self, Read};
use std::time::Instant;

use rinha_fraud::index::{
    order_points_by_bucket, order_points_by_bucket_ivf, parse_ivf_dims, Index, Point, DIMS,
};

fn main() -> io::Result<()> {
    let output = env::args()
        .nth(1)
        .unwrap_or_else(|| "references.ridx".to_owned());

    let started = Instant::now();
    let mut input = Vec::new();
    io::stdin().read_to_end(&mut input)?;

    let mut parser = ReferenceParser::new(&input);
    let mut points = parser.parse_all()?;
    eprintln!(
        "parsed {} reference vectors in {:?}",
        points.len(),
        started.elapsed()
    );

    let ordering_started = Instant::now();
    if let Ok(dims) = env::var("BUILD_INDEX_IVF_DIMS") {
        let dims = dims.trim();
        if dims.is_empty() || dims == "0" || dims.eq_ignore_ascii_case("none") {
            order_points_by_bucket(&mut points);
            eprintln!("ordered buckets in {:?}", ordering_started.elapsed());
        } else {
            let dims = parse_ivf_dims(dims);
            order_points_by_bucket_ivf(&mut points, dims);
            eprintln!(
                "ordered buckets with ivf dims {:?} in {:?}",
                dims,
                ordering_started.elapsed()
            );
        }
    } else {
        order_points_by_bucket(&mut points);
        eprintln!("ordered buckets in {:?}", ordering_started.elapsed());
    }

    Index::write(&output, &points)?;
    eprintln!("wrote {} in {:?}", output, started.elapsed());

    Ok(())
}

struct ReferenceParser<'a> {
    input: &'a [u8],
    offset: usize,
}

impl<'a> ReferenceParser<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self { input, offset: 0 }
    }

    fn parse_all(&mut self) -> io::Result<Vec<Point>> {
        let mut points = Vec::with_capacity(3_000_000);
        while let Some(vector_pos) = self.find(b"\"vector\"") {
            self.offset = vector_pos + 8;
            let mut values = [0i16; DIMS];
            for value in &mut values {
                self.seek_number()?;
                *value = self.parse_scaled_number()?;
            }

            let label_pos = self.find(b"\"label\"").ok_or_else(invalid_data)?;
            self.offset = label_pos + 7;
            let label = if let Some(fraud_pos) = self.find_before(b"fraud", b'}') {
                self.offset = fraud_pos + 5;
                1
            } else {
                self.find_before(b"legit", b'}').ok_or_else(invalid_data)?;
                0
            };

            points.push(Point {
                values,
                label,
                reserved: 0,
            });
        }
        Ok(points)
    }

    fn find(&self, needle: &[u8]) -> Option<usize> {
        self.input[self.offset..]
            .windows(needle.len())
            .position(|window| window == needle)
            .map(|position| self.offset + position)
    }

    fn find_before(&self, needle: &[u8], stop: u8) -> Option<usize> {
        let mut cursor = self.offset;
        while cursor + needle.len() <= self.input.len() {
            if self.input[cursor] == stop {
                return None;
            }
            if &self.input[cursor..cursor + needle.len()] == needle {
                return Some(cursor);
            }
            cursor += 1;
        }
        None
    }

    fn seek_number(&mut self) -> io::Result<()> {
        while self.offset < self.input.len() {
            let byte = self.input[self.offset];
            if byte == b'-' || byte.is_ascii_digit() {
                return Ok(());
            }
            self.offset += 1;
        }
        Err(invalid_data())
    }

    fn parse_scaled_number(&mut self) -> io::Result<i16> {
        let mut sign = 1i32;
        if self.input.get(self.offset) == Some(&b'-') {
            sign = -1;
            self.offset += 1;
        }

        let mut whole = 0i32;
        while let Some(byte) = self.input.get(self.offset) {
            if !byte.is_ascii_digit() {
                break;
            }
            whole = whole * 10 + (byte - b'0') as i32;
            self.offset += 1;
        }

        let mut fraction = 0i32;
        let mut scale = 1i32;
        if self.input.get(self.offset) == Some(&b'.') {
            self.offset += 1;
            while let Some(byte) = self.input.get(self.offset) {
                if !byte.is_ascii_digit() {
                    break;
                }
                if scale < 10_000 {
                    fraction = fraction * 10 + (byte - b'0') as i32;
                    scale *= 10;
                }
                self.offset += 1;
            }
        }

        while scale < 10_000 {
            fraction *= 10;
            scale *= 10;
        }

        let scaled = sign * (whole * 10_000 + fraction);
        Ok(scaled as i16)
    }
}

fn invalid_data() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid references json")
}
