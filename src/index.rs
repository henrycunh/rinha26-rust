use std::cell::RefCell;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::mem::size_of;
use std::path::Path;

use memmap2::{Mmap, MmapOptions};

pub const DIMS: usize = 14;
const MAGIC: &[u8; 8] = b"RINHAO09";
const FLAG_BUCKETS: usize = 16;
const MCC_BUCKETS: usize = 10;
const TX_BUCKETS: usize = 21;
const RATIO_BUCKETS: usize = 8;
const COARSE_BUCKETS: usize = FLAG_BUCKETS * MCC_BUCKETS * TX_BUCKETS * RATIO_BUCKETS;
const BUCKETS: usize = COARSE_BUCKETS;
const HOT_DIMS: usize = 2;
const PREFIX4_EXTRA_DIMS: usize = 2;
const SECTOR_BUCKETS: usize = 8;
const SECTORS: usize = SECTOR_BUCKETS * SECTOR_BUCKETS;
const SECTOR_WIDTH: usize = 1_250;
const SECTORIZE_THRESHOLD: usize = 1_024;
const IVF_SUB_BUCKETS: usize = 4;
const IVF_SUBCELLS: usize = IVF_SUB_BUCKETS * IVF_SUB_BUCKETS;
const IVF_CELLS: usize = SECTORS * IVF_SUBCELLS;
const IVF_SUB_WIDTH: usize = 2_500;
const DEFAULT_IVF_THRESHOLD: usize = 4_096;
const NO_SECTOR: u16 = u16::MAX;
const RATIO_WIDTH: usize = 1_250;
const INDEX_HEADER_LEN: usize = 8 + 4 + (BUCKETS + 1) * size_of::<u32>();
const DEFAULT_REPAIR_SCORES_MASK: u8 = (1u8 << 2) | (1u8 << 3);
const MCC_VALUES: [i16; MCC_BUCKETS] = [
    1_500, 2_000, 2_500, 3_000, 3_500, 4_500, 5_000, 7_500, 8_000, 8_500,
];

thread_local! {
    static CANDIDATES: RefCell<Vec<(i64, usize)>> = RefCell::new(Vec::with_capacity(8_192));
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Point {
    pub values: [i16; DIMS],
    pub label: u8,
    pub reserved: u8,
}

pub struct Index {
    points: PointStorage,
    hot_columns: [Vec<i16>; HOT_DIMS],
    packed_hot2: Option<Vec<u32>>,
    #[allow(dead_code)]
    prefix4_columns: Option<[Vec<i16>; PREFIX4_EXTRA_DIMS]>,
    bucket_offsets: Vec<u32>,
    bucket_approval_counts: Vec<BucketApprovalCounts>,
    sector_approval_counts: Option<Vec<BucketApprovalCounts>>,
    ivf_bucket_map: Vec<u16>,
    ivf_buckets: Vec<IvfBucket>,
    ivf_dims: Option<(usize, usize)>,
    ivf_global_plan: bool,
    bucket_sector_map: Vec<u16>,
    sectorized_buckets: Vec<SectorizedBucket>,
    non_empty_buckets: Vec<BucketEntry>,
    #[allow(dead_code)]
    avx2_prefix4: bool,
    #[allow(dead_code)]
    avx2_prefetch_points: usize,
}

enum PointStorage {
    Owned(Vec<Point>),
    Mapped {
        _mmap: Mmap,
        offset: usize,
        len: usize,
    },
}

impl PointStorage {
    #[inline]
    fn as_slice(&self) -> &[Point] {
        match self {
            Self::Owned(points) => points,
            Self::Mapped { _mmap, offset, len } => unsafe { mapped_points(_mmap, *offset, *len) },
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::Owned(points) => points.len(),
            Self::Mapped { len, .. } => *len,
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Copy)]
pub struct SearchResult {
    pub fraud_count: u8,
}

#[derive(Clone, Copy, Debug)]
pub enum AdaptiveClearRule {
    None,
    TailRatio70Tx45Home40,
    TailMccNonGaming,
}

#[derive(Clone, Copy, Debug)]
pub enum Score5RepairRule {
    Fp70Tight,
    Fp70Narrow,
}

#[derive(Clone, Copy)]
pub struct SearchStats {
    pub buckets_scanned: u32,
    pub points_scanned: u32,
    pub linear_buckets_scanned: u32,
    pub sectorized_buckets_scanned: u32,
    pub linear_points_scanned: u32,
    pub sectorized_points_scanned: u32,
    pub sectors_scanned: u32,
    pub sectors_pruned_by_bound: u32,
    pub candidates_considered: u32,
    pub candidates_pruned_by_bound: u32,
    pub candidates_pruned_by_limit: u32,
    pub worst_distance: i64,
}

#[derive(Clone, Copy)]
pub struct SearchDetails {
    pub primary_points_scanned: u32,
    pub secondary_points_scanned: u32,
    pub primary_bucket_sectorized: u32,
    pub secondary_sectorized_buckets: u32,
    pub largest_bucket_points: u32,
    pub limit_overshoot_points: u32,
    pub distance_steps: [u32; DIMS],
    pub distance_cutoffs: [u32; DIMS],
    pub distance_full: u32,
    pub neighbor_insertions: u32,
}

impl Default for SearchStats {
    fn default() -> Self {
        Self {
            buckets_scanned: 0,
            points_scanned: 0,
            linear_buckets_scanned: 0,
            sectorized_buckets_scanned: 0,
            linear_points_scanned: 0,
            sectorized_points_scanned: 0,
            sectors_scanned: 0,
            sectors_pruned_by_bound: 0,
            candidates_considered: 0,
            candidates_pruned_by_bound: 0,
            candidates_pruned_by_limit: 0,
            worst_distance: i64::MAX,
        }
    }
}

impl Default for SearchDetails {
    fn default() -> Self {
        Self {
            primary_points_scanned: 0,
            secondary_points_scanned: 0,
            primary_bucket_sectorized: 0,
            secondary_sectorized_buckets: 0,
            largest_bucket_points: 0,
            limit_overshoot_points: 0,
            distance_steps: [0; DIMS],
            distance_cutoffs: [0; DIMS],
            distance_full: 0,
            neighbor_insertions: 0,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct SearchTiming {
    pub primary: std::time::Duration,
    pub candidate_build: std::time::Duration,
    pub candidate_sort: std::time::Duration,
    pub bucket_scan: std::time::Duration,
}

#[derive(Clone, Copy)]
struct BucketEntry {
    bucket: usize,
    meta: BucketMeta,
}

#[derive(Clone, Copy)]
struct BucketMeta {
    flags: u8,
    mcc_value: i16,
    tx_value: i16,
    ratio_lo: i16,
    ratio_hi: i16,
}

struct SectorizedBucket {
    offsets: [u32; SECTORS + 1],
}

struct IvfBucket {
    offsets: Box<[u32; IVF_CELLS + 1]>,
}

#[derive(Clone, Copy, Default)]
struct BucketApprovalCounts {
    approved: u32,
    denied: u32,
}

#[derive(Clone, Copy, Default)]
struct SectorScan {
    points_scanned: usize,
    sectors_scanned: u32,
    sectors_pruned_by_bound: u32,
}

impl SearchResult {
    #[inline]
    pub fn approved(self) -> bool {
        self.fraud_count < 3
    }

    #[inline]
    pub fn score_text(self) -> &'static str {
        const SCORES: [&str; 6] = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"];
        SCORES[self.fraud_count as usize]
    }
}

impl Index {
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        if mmap.len() < INDEX_HEADER_LEN || &mmap[..8] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid index magic",
            ));
        }

        let count = u32::from_le_bytes(mmap[8..12].try_into().expect("count bytes")) as usize;
        let points_len = count
            .checked_mul(size_of::<Point>())
            .ok_or_else(|| invalid_index("index point count overflow"))?;
        let expected_len = INDEX_HEADER_LEN
            .checked_add(points_len)
            .ok_or_else(|| invalid_index("index size overflow"))?;
        if mmap.len() < expected_len {
            return Err(invalid_index("truncated index"));
        }

        let mut bucket_offsets = vec![0u32; BUCKETS + 1];
        for (offset, bytes) in bucket_offsets
            .iter_mut()
            .zip(mmap[12..INDEX_HEADER_LEN].chunks_exact(size_of::<u32>()))
        {
            *offset = u32::from_le_bytes(bytes.try_into().expect("offset bytes"));
        }

        if cfg!(target_endian = "big") {
            let mut points = owned_points_from_mmap(&mmap, INDEX_HEADER_LEN, count);
            for point in &mut points {
                for value in &mut point.values {
                    *value = i16::from_le(*value);
                }
            }
            return Ok(Self::from_owned_points(points, bucket_offsets));
        }

        let mapped_points = unsafe { mapped_points(&mmap, INDEX_HEADER_LEN, count) };
        let use_mmap = std::env::var("INDEX_MMAP")
            .map(|value| value != "0")
            .unwrap_or(true);
        let use_ivf_grid = index_env_flag("IVF_GRID");
        if use_mmap {
            if let Some((bucket_sector_map, sectorized_buckets)) =
                sectorized_buckets_from_ordered_points(mapped_points, &bucket_offsets)
            {
                let ivf_dims = use_ivf_grid.then(ivf_dims_from_env);
                let (ivf_bucket_map, ivf_buckets) = if let Some(dims) = ivf_dims {
                    ivf_buckets_from_ordered_points(mapped_points, &bucket_offsets, dims)
                } else {
                    (vec![NO_SECTOR; BUCKETS], Vec::new())
                };
                let ivf_global_plan = index_env_flag("IVF_GLOBAL_PLAN");
                let avx2_prefix4 = index_env_flag("AVX2_PREFIX4");
                let avx2_prefetch_points = index_env_usize("AVX2_PREFETCH_POINTS", 0);
                let use_packed_hot2 = index_env_flag("PACKED_HOT2") && !avx2_prefix4;
                let hot_columns = if use_packed_hot2 {
                    empty_hot_columns()
                } else {
                    hot_columns(mapped_points)
                };
                let packed_hot2 = use_packed_hot2.then(|| packed_hot2_columns(mapped_points));
                let prefix4_columns = avx2_prefix4.then(|| prefix4_columns(mapped_points));
                let bucket_approval_counts = bucket_approval_counts(mapped_points, &bucket_offsets);
                let sector_approval_counts = should_build_sector_decision_counts()
                    .then(|| sector_approval_counts(mapped_points, &bucket_offsets));
                let non_empty_buckets = bucket_entries(&bucket_offsets);
                return Ok(Self {
                    points: PointStorage::Mapped {
                        _mmap: mmap,
                        offset: INDEX_HEADER_LEN,
                        len: count,
                    },
                    hot_columns,
                    packed_hot2,
                    prefix4_columns,
                    bucket_offsets,
                    bucket_approval_counts,
                    sector_approval_counts,
                    ivf_bucket_map,
                    ivf_buckets,
                    ivf_dims,
                    ivf_global_plan,
                    bucket_sector_map,
                    sectorized_buckets,
                    non_empty_buckets,
                    avx2_prefix4,
                    avx2_prefetch_points,
                });
            }
        }

        let points = owned_points_from_mmap(&mmap, INDEX_HEADER_LEN, count);
        Ok(Self::from_owned_points(points, bucket_offsets))
    }

    fn from_owned_points(mut points: Vec<Point>, bucket_offsets: Vec<u32>) -> Self {
        let ivf_dims = index_env_flag("IVF_GRID").then(ivf_dims_from_env);
        let (ivf_bucket_map, ivf_buckets) = if let Some(ivf_dims) = ivf_dims {
            ivf_grid_large_buckets(&mut points, &bucket_offsets, ivf_dims)
        } else {
            (vec![NO_SECTOR; BUCKETS], Vec::new())
        };
        let ivf_global_plan = index_env_flag("IVF_GLOBAL_PLAN");

        let (bucket_sector_map, sectorized_buckets) =
            sectorized_buckets_from_ordered_points(&points, &bucket_offsets)
                .unwrap_or_else(|| sectorize_large_buckets(&mut points, &bucket_offsets));

        let avx2_prefix4 = index_env_flag("AVX2_PREFIX4");
        let avx2_prefetch_points = index_env_usize("AVX2_PREFETCH_POINTS", 0);
        let use_packed_hot2 = index_env_flag("PACKED_HOT2") && !avx2_prefix4;
        let hot_columns = if use_packed_hot2 {
            empty_hot_columns()
        } else {
            hot_columns(&points)
        };
        let packed_hot2 = use_packed_hot2.then(|| packed_hot2_columns(&points));
        let prefix4_columns = avx2_prefix4.then(|| prefix4_columns(&points));
        let bucket_approval_counts = bucket_approval_counts(&points, &bucket_offsets);
        let sector_approval_counts = should_build_sector_decision_counts()
            .then(|| sector_approval_counts(&points, &bucket_offsets));

        let non_empty_buckets = bucket_entries(&bucket_offsets);

        Self {
            points: PointStorage::Owned(points),
            hot_columns,
            packed_hot2,
            prefix4_columns,
            bucket_offsets,
            bucket_approval_counts,
            sector_approval_counts,
            ivf_bucket_map,
            ivf_buckets,
            ivf_dims,
            ivf_global_plan,
            bucket_sector_map,
            sectorized_buckets,
            non_empty_buckets,
            avx2_prefix4,
            avx2_prefetch_points,
        }
    }

    pub fn write(path: impl AsRef<Path>, points: &[Point]) -> io::Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(MAGIC)?;
        writer.write_all(&(points.len() as u32).to_le_bytes())?;

        let bucket_offsets = bucket_offsets(points);
        for offset in bucket_offsets {
            writer.write_all(&offset.to_le_bytes())?;
        }

        if cfg!(target_endian = "little") {
            let raw = unsafe {
                std::slice::from_raw_parts(
                    points.as_ptr() as *const u8,
                    std::mem::size_of_val(points),
                )
            };
            writer.write_all(raw)?;
        } else {
            for point in points {
                for value in point.values {
                    writer.write_all(&value.to_le_bytes())?;
                }
                writer.write_all(&[point.label, point.reserved])?;
            }
        }

        writer.flush()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn bucket_majority_decision(
        &self,
        query: &[i16; DIMS],
        min_total: u32,
        max_error_per_10000: u32,
    ) -> Option<u8> {
        let counts = self.bucket_approval_counts[vector_bucket(query)];
        majority_decision(counts, min_total, max_error_per_10000)
    }

    pub fn sector_majority_decision(
        &self,
        query: &[i16; DIMS],
        min_total: u32,
        max_error_per_10000: u32,
    ) -> Option<u8> {
        let counts = *self
            .sector_approval_counts
            .as_ref()?
            .get(sector_decision_key_query(query))?;
        majority_decision(counts, min_total, max_error_per_10000)
    }

    #[inline]
    pub fn search(&self, query: &[i16; DIMS]) -> SearchResult {
        self.search_sorted(query).0
    }

    pub fn search_checked(&self, query: &[i16; DIMS]) -> (SearchResult, bool) {
        let mut stats = SearchStats::default();
        let result = self.search_internal(query, Some(&mut stats));
        (result, stats.buckets_scanned > 1)
    }

    pub fn search_with_stats(&self, query: &[i16; DIMS]) -> (SearchResult, SearchStats) {
        let mut stats = SearchStats::default();
        let result = self.search_internal(query, Some(&mut stats));
        (result, stats)
    }

    fn search_internal(
        &self,
        query: &[i16; DIMS],
        mut stats: Option<&mut SearchStats>,
    ) -> SearchResult {
        let mut neighbors = NeighborSet::default();
        let scan_query = scan_order(query);
        let primary_bucket = vector_bucket(query);
        let query_flags = vector_flags(query);
        let sector_plan = sector_scan_plan(&scan_query);
        let ivf_sub_plan = self.ivf_sub_plan(&scan_query);
        let ivf_cell_plan = self.ivf_cell_plan(&scan_query);
        self.search_bucket(
            primary_bucket,
            0,
            &scan_query,
            &sector_plan,
            ivf_sub_plan.as_ref(),
            ivf_cell_plan.as_ref(),
            &mut neighbors,
            stats.as_deref_mut(),
        );

        for entry in &self.non_empty_buckets {
            let bucket = entry.bucket;
            if bucket == primary_bucket {
                continue;
            }
            let meta = entry.meta;
            let lower_bound = bucket_lower_bound(query, query_flags, meta);
            if !neighbors.is_full() || lower_bound < neighbors.worst() {
                self.search_bucket(
                    bucket,
                    lower_bound,
                    &scan_query,
                    &sector_plan,
                    ivf_sub_plan.as_ref(),
                    ivf_cell_plan.as_ref(),
                    &mut neighbors,
                    stats.as_deref_mut(),
                );
            }
        }

        if let Some(stats) = stats {
            stats.worst_distance = neighbors.worst();
        }
        neighbors.result()
    }

    pub fn search_brute(&self, query: &[i16; DIMS]) -> SearchResult {
        let scan_query = scan_order(query);
        let mut neighbors = NeighborSet::default();
        self.search_range(0, self.points.len(), &scan_query, &mut neighbors);
        neighbors.result()
    }

    pub fn search_primary(&self, query: &[i16; DIMS]) -> SearchResult {
        let bucket = vector_bucket(query);
        let scan_query = scan_order(query);
        let mut neighbors = NeighborSet::default();
        let sector_plan = sector_scan_plan(&scan_query);
        let ivf_sub_plan = self.ivf_sub_plan(&scan_query);
        let ivf_cell_plan = self.ivf_cell_plan(&scan_query);
        self.search_bucket(
            bucket,
            0,
            &scan_query,
            &sector_plan,
            ivf_sub_plan.as_ref(),
            ivf_cell_plan.as_ref(),
            &mut neighbors,
            None,
        );
        neighbors.result()
    }

    pub fn search_sorted(&self, query: &[i16; DIMS]) -> (SearchResult, SearchStats) {
        CANDIDATES.with_borrow_mut(|candidates| self.search_sorted_with_buffer(query, candidates))
    }

    pub fn search_limited(
        &self,
        query: &[i16; DIMS],
        max_points: u32,
    ) -> (SearchResult, SearchStats) {
        CANDIDATES.with_borrow_mut(|candidates| {
            self.search_sorted_limited_with_buffer(query, max_points, candidates)
        })
    }

    pub fn search_limited_result(&self, query: &[i16; DIMS], max_points: u32) -> SearchResult {
        CANDIDATES.with_borrow_mut(|candidates| {
            self.search_sorted_limited_result_with_buffer(query, max_points, candidates)
        })
    }

    pub fn search_limited_boundary_repair_result(
        &self,
        query: &[i16; DIMS],
        fast_points: u32,
        repair_points: u32,
    ) -> SearchResult {
        CANDIDATES.with_borrow_mut(|candidates| {
            self.search_sorted_limited_boundary_repair_result_with_buffer(
                query,
                fast_points,
                repair_points,
                DEFAULT_REPAIR_SCORES_MASK,
                None,
                None,
                None,
                None,
                None,
                candidates,
            )
        })
    }

    pub fn search_limited_repair_mask_result(
        &self,
        query: &[i16; DIMS],
        fast_points: u32,
        repair_points: u32,
        repair_scores_mask: u8,
    ) -> SearchResult {
        CANDIDATES.with_borrow_mut(|candidates| {
            self.search_sorted_limited_boundary_repair_result_with_buffer(
                query,
                fast_points,
                repair_points,
                repair_scores_mask,
                None,
                None,
                None,
                None,
                None,
                candidates,
            )
        })
    }

    pub fn search_limited_repair_mask_score5_rule_result(
        &self,
        query: &[i16; DIMS],
        fast_points: u32,
        repair_points: u32,
        repair_scores_mask: u8,
        score5_repair_rule: Score5RepairRule,
    ) -> SearchResult {
        CANDIDATES.with_borrow_mut(|candidates| {
            self.search_sorted_limited_boundary_repair_result_with_buffer(
                query,
                fast_points,
                repair_points,
                repair_scores_mask,
                None,
                None,
                Some(score5_repair_rule),
                None,
                None,
                candidates,
            )
        })
    }

    pub fn search_limited_selective_repair_result(
        &self,
        query: &[i16; DIMS],
        fast_points: u32,
        repair_points: u32,
        repair_scores_mask: u8,
        score3_worst_threshold: Option<i64>,
        score4_only_worst_threshold: Option<i64>,
        score5_repair_rule: Option<Score5RepairRule>,
    ) -> SearchResult {
        CANDIDATES.with_borrow_mut(|candidates| {
            self.search_sorted_limited_boundary_repair_result_with_buffer(
                query,
                fast_points,
                repair_points,
                repair_scores_mask,
                None,
                None,
                score5_repair_rule,
                score3_worst_threshold,
                score4_only_worst_threshold,
                candidates,
            )
        })
    }

    pub fn search_limited_repair_mask_high_below_result(
        &self,
        query: &[i16; DIMS],
        fast_points: u32,
        repair_points: u32,
        repair_scores_mask: u8,
        high_score_worst_below_threshold: i64,
    ) -> SearchResult {
        CANDIDATES.with_borrow_mut(|candidates| {
            self.search_sorted_limited_boundary_repair_result_with_buffer(
                query,
                fast_points,
                repair_points,
                repair_scores_mask,
                None,
                Some(high_score_worst_below_threshold),
                None,
                None,
                None,
                candidates,
            )
        })
    }

    pub fn search_limited_score4_repair_result(
        &self,
        query: &[i16; DIMS],
        fast_points: u32,
        repair_points: u32,
    ) -> SearchResult {
        CANDIDATES.with_borrow_mut(|candidates| {
            self.search_sorted_limited_boundary_repair_result_with_buffer(
                query,
                fast_points,
                repair_points,
                DEFAULT_REPAIR_SCORES_MASK,
                Some(i64::MIN),
                None,
                None,
                None,
                None,
                candidates,
            )
        })
    }

    pub fn search_limited_score4_threshold_repair_result(
        &self,
        query: &[i16; DIMS],
        fast_points: u32,
        repair_points: u32,
        score4_worst_threshold: i64,
    ) -> SearchResult {
        CANDIDATES.with_borrow_mut(|candidates| {
            self.search_sorted_limited_boundary_repair_result_with_buffer(
                query,
                fast_points,
                repair_points,
                DEFAULT_REPAIR_SCORES_MASK,
                Some(score4_worst_threshold),
                None,
                None,
                None,
                None,
                candidates,
            )
        })
    }

    pub fn search_limited_adaptive_boundary_repair_result(
        &self,
        query: &[i16; DIMS],
        early_points: u32,
        fast_points: u32,
        repair_points: u32,
        clear_scores_mask: u8,
        repair_scores_mask: u8,
        clear_worst_max: Option<i64>,
        clear_rule: AdaptiveClearRule,
    ) -> SearchResult {
        CANDIDATES.with_borrow_mut(|candidates| {
            self.search_sorted_limited_adaptive_boundary_repair_result_with_buffer(
                query,
                early_points,
                fast_points,
                repair_points,
                clear_scores_mask,
                repair_scores_mask,
                clear_worst_max,
                clear_rule,
                candidates,
            )
        })
    }

    pub fn search_limited_instrumented(
        &self,
        query: &[i16; DIMS],
        max_points: u32,
    ) -> (SearchResult, SearchStats, SearchTiming, SearchDetails) {
        CANDIDATES.with_borrow_mut(|candidates| {
            self.search_sorted_limited_instrumented(query, max_points, candidates)
        })
    }

    fn search_sorted_with_buffer(
        &self,
        query: &[i16; DIMS],
        candidates: &mut Vec<(i64, usize)>,
    ) -> (SearchResult, SearchStats) {
        let mut neighbors = NeighborSet::default();
        let mut stats = SearchStats::default();
        let scan_query = scan_order(query);
        let primary_bucket = vector_bucket(query);
        let query_flags = vector_flags(query);
        let sector_plan = sector_scan_plan(&scan_query);
        let ivf_sub_plan = self.ivf_sub_plan(&scan_query);
        let ivf_cell_plan = self.ivf_cell_plan(&scan_query);
        self.search_bucket(
            primary_bucket,
            0,
            &scan_query,
            &sector_plan,
            ivf_sub_plan.as_ref(),
            ivf_cell_plan.as_ref(),
            &mut neighbors,
            Some(&mut stats),
        );

        let prefiltered_by_bound = self.build_candidate_buckets(
            query,
            query_flags,
            primary_bucket,
            neighbors.is_full().then(|| neighbors.worst()),
            candidates,
        );
        stats.candidates_pruned_by_bound += prefiltered_by_bound;
        candidates.sort_unstable_by_key(|candidate| candidate.0);

        for &(lower_bound, bucket) in candidates.iter() {
            stats.candidates_considered += 1;
            if neighbors.is_full() && lower_bound >= neighbors.worst() {
                stats.candidates_pruned_by_bound +=
                    candidates.len() as u32 - stats.candidates_considered + 1;
                break;
            }
            self.search_bucket(
                bucket,
                lower_bound,
                &scan_query,
                &sector_plan,
                ivf_sub_plan.as_ref(),
                ivf_cell_plan.as_ref(),
                &mut neighbors,
                Some(&mut stats),
            );
        }

        stats.worst_distance = neighbors.worst();
        (neighbors.result(), stats)
    }

    fn search_sorted_limited_with_buffer(
        &self,
        query: &[i16; DIMS],
        max_points: u32,
        candidates: &mut Vec<(i64, usize)>,
    ) -> (SearchResult, SearchStats) {
        let mut neighbors = NeighborSet::default();
        let mut stats = SearchStats::default();
        let scan_query = scan_order(query);
        let primary_bucket = vector_bucket(query);
        let query_flags = vector_flags(query);
        let sector_plan = sector_scan_plan(&scan_query);
        let ivf_sub_plan = self.ivf_sub_plan(&scan_query);
        let ivf_cell_plan = self.ivf_cell_plan(&scan_query);
        self.search_bucket(
            primary_bucket,
            0,
            &scan_query,
            &sector_plan,
            ivf_sub_plan.as_ref(),
            ivf_cell_plan.as_ref(),
            &mut neighbors,
            Some(&mut stats),
        );

        let prefiltered_by_bound = self.build_candidate_buckets(
            query,
            query_flags,
            primary_bucket,
            neighbors.is_full().then(|| neighbors.worst()),
            candidates,
        );
        stats.candidates_pruned_by_bound += prefiltered_by_bound;
        candidates.sort_unstable_by_key(|candidate| candidate.0);

        for &(lower_bound, bucket) in candidates.iter() {
            stats.candidates_considered += 1;
            if neighbors.is_full() && lower_bound >= neighbors.worst() {
                stats.candidates_pruned_by_bound +=
                    candidates.len() as u32 - stats.candidates_considered + 1;
                break;
            }
            if neighbors.is_full() && stats.points_scanned >= max_points {
                stats.candidates_pruned_by_limit =
                    candidates.len() as u32 - stats.candidates_considered + 1;
                break;
            }
            let max_bucket_points = if neighbors.is_full() {
                Some((max_points - stats.points_scanned) as usize)
            } else {
                None
            };
            self.search_bucket_capped(
                bucket,
                lower_bound,
                &scan_query,
                &sector_plan,
                ivf_sub_plan.as_ref(),
                ivf_cell_plan.as_ref(),
                &mut neighbors,
                max_bucket_points,
                Some(&mut stats),
            );
        }

        stats.worst_distance = neighbors.worst();
        (neighbors.result(), stats)
    }

    fn search_sorted_limited_result_with_buffer(
        &self,
        query: &[i16; DIMS],
        max_points: u32,
        candidates: &mut Vec<(i64, usize)>,
    ) -> SearchResult {
        let mut neighbors = NeighborSet::default();
        let scan_query = scan_order(query);
        let primary_bucket = vector_bucket(query);
        let query_flags = vector_flags(query);
        let sector_plan = sector_scan_plan(&scan_query);
        let ivf_sub_plan = self.ivf_sub_plan(&scan_query);
        let ivf_cell_plan = self.ivf_cell_plan(&scan_query);
        let mut points_scanned = self.search_bucket(
            primary_bucket,
            0,
            &scan_query,
            &sector_plan,
            ivf_sub_plan.as_ref(),
            ivf_cell_plan.as_ref(),
            &mut neighbors,
            None,
        ) as u32;

        self.build_candidate_buckets(
            query,
            query_flags,
            primary_bucket,
            neighbors.is_full().then(|| neighbors.worst()),
            candidates,
        );
        candidates.sort_unstable_by_key(|candidate| candidate.0);

        for &(lower_bound, bucket) in candidates.iter() {
            if neighbors.is_full() && lower_bound >= neighbors.worst() {
                break;
            }
            if neighbors.is_full() && points_scanned >= max_points {
                break;
            }
            let max_bucket_points = if neighbors.is_full() {
                Some((max_points - points_scanned) as usize)
            } else {
                None
            };
            points_scanned += self.search_bucket_capped(
                bucket,
                lower_bound,
                &scan_query,
                &sector_plan,
                ivf_sub_plan.as_ref(),
                ivf_cell_plan.as_ref(),
                &mut neighbors,
                max_bucket_points,
                None,
            ) as u32;
        }

        neighbors.result()
    }

    fn search_sorted_limited_boundary_repair_result_with_buffer(
        &self,
        query: &[i16; DIMS],
        fast_points: u32,
        repair_points: u32,
        repair_scores_mask: u8,
        score4_worst_threshold: Option<i64>,
        high_score_worst_below_threshold: Option<i64>,
        score5_repair_rule: Option<Score5RepairRule>,
        score3_worst_threshold: Option<i64>,
        score4_only_worst_threshold: Option<i64>,
        candidates: &mut Vec<(i64, usize)>,
    ) -> SearchResult {
        let mut neighbors = NeighborSet::default();
        let scan_query = scan_order(query);
        let primary_bucket = vector_bucket(query);
        let query_flags = vector_flags(query);
        let sector_plan = sector_scan_plan(&scan_query);
        let ivf_sub_plan = self.ivf_sub_plan(&scan_query);
        let ivf_cell_plan = self.ivf_cell_plan(&scan_query);
        let mut points_scanned = self.search_bucket(
            primary_bucket,
            0,
            &scan_query,
            &sector_plan,
            ivf_sub_plan.as_ref(),
            ivf_cell_plan.as_ref(),
            &mut neighbors,
            None,
        ) as u32;
        let mut limit = fast_points;
        let mut repairing = false;

        self.build_candidate_buckets(
            query,
            query_flags,
            primary_bucket,
            neighbors.is_full().then(|| neighbors.worst()),
            candidates,
        );
        candidates.sort_unstable_by_key(|candidate| candidate.0);

        for &(lower_bound, bucket) in candidates.iter() {
            if neighbors.is_full() && lower_bound >= neighbors.worst() {
                break;
            }
            if neighbors.is_full() && points_scanned >= limit {
                let fraud_count = neighbors.result().fraud_count;
                if !repairing
                    && ((repair_scores_mask & (1u8 << fraud_count)) != 0
                        || (fraud_count >= 4
                            && score4_worst_threshold
                                .is_some_and(|threshold| neighbors.worst() >= threshold))
                        || (fraud_count >= 4
                            && high_score_worst_below_threshold
                                .is_some_and(|threshold| neighbors.worst() <= threshold))
                        || (fraud_count == 3
                            && score3_worst_threshold
                                .is_some_and(|threshold| neighbors.worst() >= threshold))
                        || (fraud_count == 4
                            && score4_only_worst_threshold
                                .is_some_and(|threshold| neighbors.worst() >= threshold))
                        || (fraud_count == 5
                            && score5_repair_rule
                                .is_some_and(|rule| score5_repair_rule_matches(rule, query))))
                {
                    repairing = true;
                    limit = repair_points;
                } else {
                    break;
                }
            }
            let max_bucket_points = if neighbors.is_full() {
                Some((limit - points_scanned) as usize)
            } else {
                None
            };
            points_scanned += self.search_bucket_capped(
                bucket,
                lower_bound,
                &scan_query,
                &sector_plan,
                ivf_sub_plan.as_ref(),
                ivf_cell_plan.as_ref(),
                &mut neighbors,
                max_bucket_points,
                None,
            ) as u32;
        }

        neighbors.result()
    }

    fn search_sorted_limited_adaptive_boundary_repair_result_with_buffer(
        &self,
        query: &[i16; DIMS],
        early_points: u32,
        fast_points: u32,
        repair_points: u32,
        clear_scores_mask: u8,
        repair_scores_mask: u8,
        clear_worst_max: Option<i64>,
        clear_rule: AdaptiveClearRule,
        candidates: &mut Vec<(i64, usize)>,
    ) -> SearchResult {
        let mut neighbors = NeighborSet::default();
        let scan_query = scan_order(query);
        let primary_bucket = vector_bucket(query);
        let query_flags = vector_flags(query);
        let sector_plan = sector_scan_plan(&scan_query);
        let ivf_sub_plan = self.ivf_sub_plan(&scan_query);
        let ivf_cell_plan = self.ivf_cell_plan(&scan_query);
        let mut points_scanned = self.search_bucket(
            primary_bucket,
            0,
            &scan_query,
            &sector_plan,
            ivf_sub_plan.as_ref(),
            ivf_cell_plan.as_ref(),
            &mut neighbors,
            None,
        ) as u32;
        let mut phase = AdaptivePhase::Early;
        let mut limit = early_points;

        self.build_candidate_buckets(
            query,
            query_flags,
            primary_bucket,
            neighbors.is_full().then(|| neighbors.worst()),
            candidates,
        );
        candidates.sort_unstable_by_key(|candidate| candidate.0);

        'bucket: for &(lower_bound, bucket) in candidates.iter() {
            if neighbors.is_full() && lower_bound >= neighbors.worst() {
                break;
            }

            while neighbors.is_full() && points_scanned >= limit {
                let fraud_count = neighbors.result().fraud_count;
                match phase {
                    AdaptivePhase::Early => {
                        if clear_scores_mask & (1 << fraud_count) != 0
                            && clear_worst_max
                                .map_or(true, |threshold| neighbors.worst() <= threshold)
                            && adaptive_clear_rule_matches(clear_rule, query)
                        {
                            break 'bucket;
                        }
                        phase = AdaptivePhase::Fast;
                        limit = fast_points;
                    }
                    AdaptivePhase::Fast => {
                        if repair_scores_mask & (1 << fraud_count) != 0 {
                            phase = AdaptivePhase::Repair;
                            limit = repair_points;
                        } else {
                            break 'bucket;
                        }
                    }
                    AdaptivePhase::Repair => break 'bucket,
                }
            }

            let max_bucket_points = if neighbors.is_full() {
                Some((limit - points_scanned) as usize)
            } else {
                None
            };
            points_scanned += self.search_bucket_capped(
                bucket,
                lower_bound,
                &scan_query,
                &sector_plan,
                ivf_sub_plan.as_ref(),
                ivf_cell_plan.as_ref(),
                &mut neighbors,
                max_bucket_points,
                None,
            ) as u32;
        }

        neighbors.result()
    }

    fn search_sorted_limited_instrumented(
        &self,
        query: &[i16; DIMS],
        max_points: u32,
        candidates: &mut Vec<(i64, usize)>,
    ) -> (SearchResult, SearchStats, SearchTiming, SearchDetails) {
        let mut neighbors = NeighborSet::default();
        let mut stats = SearchStats::default();
        let mut details = SearchDetails::default();
        let mut timing = SearchTiming::default();
        let scan_query = scan_order(query);
        let primary_bucket = vector_bucket(query);
        let query_flags = vector_flags(query);
        let sector_plan = sector_scan_plan(&scan_query);
        let ivf_sub_plan = self.ivf_sub_plan(&scan_query);
        let ivf_cell_plan = self.ivf_cell_plan(&scan_query);

        let before = std::time::Instant::now();
        self.search_bucket_instrumented(
            primary_bucket,
            0,
            &scan_query,
            &sector_plan,
            ivf_sub_plan.as_ref(),
            ivf_cell_plan.as_ref(),
            &mut neighbors,
            &mut stats,
            &mut details,
            true,
        );
        timing.primary = before.elapsed();

        let before = std::time::Instant::now();
        let prefiltered_by_bound = self.build_candidate_buckets(
            query,
            query_flags,
            primary_bucket,
            neighbors.is_full().then(|| neighbors.worst()),
            candidates,
        );
        stats.candidates_pruned_by_bound += prefiltered_by_bound;
        timing.candidate_build = before.elapsed();

        let before = std::time::Instant::now();
        candidates.sort_unstable_by_key(|candidate| candidate.0);
        timing.candidate_sort = before.elapsed();

        let before = std::time::Instant::now();
        for &(lower_bound, bucket) in candidates.iter() {
            stats.candidates_considered += 1;
            if neighbors.is_full() && lower_bound >= neighbors.worst() {
                stats.candidates_pruned_by_bound +=
                    candidates.len() as u32 - stats.candidates_considered + 1;
                break;
            }
            if neighbors.is_full() && stats.points_scanned >= max_points {
                stats.candidates_pruned_by_limit =
                    candidates.len() as u32 - stats.candidates_considered + 1;
                break;
            }
            let max_bucket_points = if neighbors.is_full() {
                Some((max_points - stats.points_scanned) as usize)
            } else {
                None
            };
            self.search_bucket_instrumented_capped(
                bucket,
                lower_bound,
                &scan_query,
                &sector_plan,
                ivf_sub_plan.as_ref(),
                ivf_cell_plan.as_ref(),
                &mut neighbors,
                &mut stats,
                &mut details,
                false,
                max_bucket_points,
            );
        }
        timing.bucket_scan = before.elapsed();
        if stats.points_scanned > max_points {
            details.limit_overshoot_points = stats.points_scanned - max_points;
        }

        stats.worst_distance = neighbors.worst();
        (neighbors.result(), stats, timing, details)
    }

    fn build_candidate_buckets(
        &self,
        query: &[i16; DIMS],
        query_flags: usize,
        primary_bucket: usize,
        worst_bound: Option<i64>,
        candidates: &mut Vec<(i64, usize)>,
    ) -> u32 {
        candidates.clear();
        let mut pruned_by_bound = 0u32;

        for entry in &self.non_empty_buckets {
            if entry.bucket == primary_bucket {
                continue;
            }

            let lower_bound = bucket_lower_bound(query, query_flags, entry.meta);
            if worst_bound.is_some_and(|worst| lower_bound >= worst) {
                pruned_by_bound += 1;
                continue;
            }

            candidates.push((lower_bound, entry.bucket));
        }

        pruned_by_bound
    }

    fn search_bucket(
        &self,
        bucket: usize,
        bucket_lower_bound: i64,
        scan_query: &[i16; DIMS],
        sector_plan: &[(i64, usize); SECTORS],
        ivf_sub_plan: Option<&[(i64, usize); IVF_SUBCELLS]>,
        ivf_cell_plan: Option<&[(i64, usize); IVF_CELLS]>,
        neighbors: &mut NeighborSet,
        stats: Option<&mut SearchStats>,
    ) -> usize {
        self.search_bucket_capped(
            bucket,
            bucket_lower_bound,
            scan_query,
            sector_plan,
            ivf_sub_plan,
            ivf_cell_plan,
            neighbors,
            None,
            stats,
        )
    }

    fn search_bucket_capped(
        &self,
        bucket: usize,
        bucket_lower_bound: i64,
        scan_query: &[i16; DIMS],
        sector_plan: &[(i64, usize); SECTORS],
        ivf_sub_plan: Option<&[(i64, usize); IVF_SUBCELLS]>,
        ivf_cell_plan: Option<&[(i64, usize); IVF_CELLS]>,
        neighbors: &mut NeighborSet,
        max_points: Option<usize>,
        stats: Option<&mut SearchStats>,
    ) -> usize {
        let start = self.bucket_offsets[bucket] as usize;
        let end = self.bucket_offsets[bucket + 1] as usize;
        if start == end || max_points == Some(0) {
            return 0;
        }

        let scanned = if let Some(ivf_index) = self.ivf_index(bucket) {
            if let Some(cell_plan) = ivf_cell_plan {
                self.search_ivf_bucket_global(
                    ivf_index,
                    bucket_lower_bound,
                    scan_query,
                    cell_plan,
                    neighbors,
                    max_points,
                )
            } else {
                self.search_ivf_bucket(
                    ivf_index,
                    bucket_lower_bound,
                    scan_query,
                    sector_plan,
                    ivf_sub_plan.expect("ivf sub plan"),
                    neighbors,
                    max_points,
                )
            }
        } else if let Some(sector_index) = self.sector_index(bucket) {
            self.search_sectorized_bucket(
                sector_index,
                bucket_lower_bound,
                scan_query,
                sector_plan,
                neighbors,
                max_points,
            )
        } else {
            let scan_end = max_points.map_or(end, |max| end.min(start + max));
            self.search_range(start, scan_end, scan_query, neighbors);
            scan_end - start
        };
        if let Some(stats) = stats {
            stats.buckets_scanned += 1;
            stats.points_scanned += scanned as u32;
        }
        scanned
    }

    fn search_bucket_instrumented(
        &self,
        bucket: usize,
        bucket_lower_bound: i64,
        scan_query: &[i16; DIMS],
        sector_plan: &[(i64, usize); SECTORS],
        ivf_sub_plan: Option<&[(i64, usize); IVF_SUBCELLS]>,
        ivf_cell_plan: Option<&[(i64, usize); IVF_CELLS]>,
        neighbors: &mut NeighborSet,
        stats: &mut SearchStats,
        details: &mut SearchDetails,
        is_primary: bool,
    ) {
        self.search_bucket_instrumented_capped(
            bucket,
            bucket_lower_bound,
            scan_query,
            sector_plan,
            ivf_sub_plan,
            ivf_cell_plan,
            neighbors,
            stats,
            details,
            is_primary,
            None,
        );
    }

    fn search_bucket_instrumented_capped(
        &self,
        bucket: usize,
        bucket_lower_bound: i64,
        scan_query: &[i16; DIMS],
        sector_plan: &[(i64, usize); SECTORS],
        ivf_sub_plan: Option<&[(i64, usize); IVF_SUBCELLS]>,
        ivf_cell_plan: Option<&[(i64, usize); IVF_CELLS]>,
        neighbors: &mut NeighborSet,
        stats: &mut SearchStats,
        details: &mut SearchDetails,
        is_primary: bool,
        max_points: Option<usize>,
    ) {
        let start = self.bucket_offsets[bucket] as usize;
        let end = self.bucket_offsets[bucket + 1] as usize;
        if start == end || max_points == Some(0) {
            return;
        }

        let (point_count, sector_scan) = if let Some(ivf_index) = self.ivf_index(bucket) {
            let sector_scan = if let Some(cell_plan) = ivf_cell_plan {
                self.search_ivf_bucket_global_instrumented(
                    ivf_index,
                    bucket_lower_bound,
                    scan_query,
                    cell_plan,
                    neighbors,
                    details,
                    max_points,
                )
            } else {
                self.search_ivf_bucket_instrumented(
                    ivf_index,
                    bucket_lower_bound,
                    scan_query,
                    sector_plan,
                    ivf_sub_plan.expect("ivf sub plan"),
                    neighbors,
                    details,
                    max_points,
                )
            };
            (sector_scan.points_scanned as u32, Some(sector_scan))
        } else if let Some(sector_index) = self.sector_index(bucket) {
            let sector_scan = self.search_sectorized_bucket_instrumented(
                sector_index,
                bucket_lower_bound,
                scan_query,
                sector_plan,
                neighbors,
                details,
                max_points,
            );
            (sector_scan.points_scanned as u32, Some(sector_scan))
        } else {
            let scan_end = max_points.map_or(end, |max| end.min(start + max));
            self.search_range_instrumented(start, scan_end, scan_query, neighbors, details);
            ((scan_end - start) as u32, None)
        };

        stats.buckets_scanned += 1;
        stats.points_scanned += point_count;
        if let Some(sector_scan) = sector_scan {
            stats.sectorized_buckets_scanned += 1;
            stats.sectorized_points_scanned += sector_scan.points_scanned as u32;
            stats.sectors_scanned += sector_scan.sectors_scanned;
            stats.sectors_pruned_by_bound += sector_scan.sectors_pruned_by_bound;
            if is_primary {
                details.primary_bucket_sectorized = 1;
            } else {
                details.secondary_sectorized_buckets += 1;
            }
        } else {
            stats.linear_buckets_scanned += 1;
            stats.linear_points_scanned += point_count;
        }
        details.largest_bucket_points = details.largest_bucket_points.max(point_count);
        if is_primary {
            details.primary_points_scanned += point_count;
        } else {
            details.secondary_points_scanned += point_count;
        }
    }

    #[inline]
    fn sector_index(&self, bucket: usize) -> Option<usize> {
        let index = self.bucket_sector_map[bucket];
        (index != NO_SECTOR).then_some(index as usize)
    }

    #[inline]
    fn ivf_index(&self, bucket: usize) -> Option<usize> {
        let index = self.ivf_bucket_map[bucket];
        (index != NO_SECTOR).then_some(index as usize)
    }

    #[inline]
    fn ivf_sub_plan(&self, scan_query: &[i16; DIMS]) -> Option<[(i64, usize); IVF_SUBCELLS]> {
        self.ivf_dims
            .filter(|_| !self.ivf_buckets.is_empty() && !self.ivf_global_plan)
            .map(|dims| ivf_sub_scan_plan(scan_query, dims))
    }

    #[inline]
    fn ivf_cell_plan(&self, scan_query: &[i16; DIMS]) -> Option<[(i64, usize); IVF_CELLS]> {
        self.ivf_dims
            .filter(|_| !self.ivf_buckets.is_empty() && self.ivf_global_plan)
            .map(|dims| ivf_cell_scan_plan(scan_query, dims))
    }

    fn search_ivf_bucket(
        &self,
        ivf_index: usize,
        bucket_lower_bound: i64,
        scan_query: &[i16; DIMS],
        sector_plan: &[(i64, usize); SECTORS],
        sub_plan: &[(i64, usize); IVF_SUBCELLS],
        neighbors: &mut NeighborSet,
        max_points: Option<usize>,
    ) -> usize {
        let ivf = &self.ivf_buckets[ivf_index];
        let mut points_scanned = 0usize;

        'sector: for &(sector_lower_bound, sector) in sector_plan.iter() {
            for &(sub_lower_bound, subcell) in sub_plan {
                let remaining = max_points.map(|max| max.saturating_sub(points_scanned));
                if remaining == Some(0) {
                    break 'sector;
                }

                let lower_bound = bucket_lower_bound + sector_lower_bound + sub_lower_bound;
                if neighbors.is_full() && lower_bound >= neighbors.worst() {
                    if subcell == sub_plan[0].1 {
                        break 'sector;
                    }
                    break;
                }

                let cell = sector * IVF_SUBCELLS + subcell;
                let start = ivf.offsets[cell] as usize;
                let end = ivf.offsets[cell + 1] as usize;
                if start == end {
                    continue;
                }

                let scan_end = remaining.map_or(end, |max| end.min(start + max));
                self.search_range(start, scan_end, scan_query, neighbors);
                points_scanned += scan_end - start;
                if scan_end < end {
                    break 'sector;
                }
            }
        }

        points_scanned
    }

    fn search_ivf_bucket_global(
        &self,
        ivf_index: usize,
        bucket_lower_bound: i64,
        scan_query: &[i16; DIMS],
        cell_plan: &[(i64, usize); IVF_CELLS],
        neighbors: &mut NeighborSet,
        max_points: Option<usize>,
    ) -> usize {
        let ivf = &self.ivf_buckets[ivf_index];
        let mut points_scanned = 0usize;

        for &(cell_lower_bound, cell) in cell_plan {
            let remaining = max_points.map(|max| max.saturating_sub(points_scanned));
            if remaining == Some(0) {
                break;
            }

            let lower_bound = bucket_lower_bound + cell_lower_bound;
            if neighbors.is_full() && lower_bound >= neighbors.worst() {
                break;
            }

            let start = ivf.offsets[cell] as usize;
            let end = ivf.offsets[cell + 1] as usize;
            if start == end {
                continue;
            }

            let scan_end = remaining.map_or(end, |max| end.min(start + max));
            self.search_range(start, scan_end, scan_query, neighbors);
            points_scanned += scan_end - start;
            if scan_end < end {
                break;
            }
        }

        points_scanned
    }

    fn search_ivf_bucket_instrumented(
        &self,
        ivf_index: usize,
        bucket_lower_bound: i64,
        scan_query: &[i16; DIMS],
        sector_plan: &[(i64, usize); SECTORS],
        sub_plan: &[(i64, usize); IVF_SUBCELLS],
        neighbors: &mut NeighborSet,
        details: &mut SearchDetails,
        max_points: Option<usize>,
    ) -> SectorScan {
        let ivf = &self.ivf_buckets[ivf_index];
        let mut scan = SectorScan::default();

        'sector: for (plan_index, &(sector_lower_bound, sector)) in sector_plan.iter().enumerate() {
            let mut sector_touched = false;
            for &(sub_lower_bound, subcell) in sub_plan {
                let remaining = max_points.map(|max| max.saturating_sub(scan.points_scanned));
                if remaining == Some(0) {
                    break 'sector;
                }

                let lower_bound = bucket_lower_bound + sector_lower_bound + sub_lower_bound;
                if neighbors.is_full() && lower_bound >= neighbors.worst() {
                    if subcell == sub_plan[0].1 {
                        scan.sectors_pruned_by_bound = (SECTORS - plan_index) as u32;
                        break 'sector;
                    }
                    break;
                }

                let cell = sector * IVF_SUBCELLS + subcell;
                let start = ivf.offsets[cell] as usize;
                let end = ivf.offsets[cell + 1] as usize;
                if start == end {
                    continue;
                }

                let scan_end = remaining.map_or(end, |max| end.min(start + max));
                self.search_range_instrumented(start, scan_end, scan_query, neighbors, details);
                scan.points_scanned += scan_end - start;
                sector_touched = true;
                if scan_end < end {
                    break 'sector;
                }
            }
            if sector_touched {
                scan.sectors_scanned += 1;
            }
        }

        scan
    }

    fn search_ivf_bucket_global_instrumented(
        &self,
        ivf_index: usize,
        bucket_lower_bound: i64,
        scan_query: &[i16; DIMS],
        cell_plan: &[(i64, usize); IVF_CELLS],
        neighbors: &mut NeighborSet,
        details: &mut SearchDetails,
        max_points: Option<usize>,
    ) -> SectorScan {
        let ivf = &self.ivf_buckets[ivf_index];
        let mut scan = SectorScan::default();
        let mut sectors_touched = [false; SECTORS];

        for (plan_index, &(cell_lower_bound, cell)) in cell_plan.iter().enumerate() {
            let remaining = max_points.map(|max| max.saturating_sub(scan.points_scanned));
            if remaining == Some(0) {
                break;
            }

            let lower_bound = bucket_lower_bound + cell_lower_bound;
            if neighbors.is_full() && lower_bound >= neighbors.worst() {
                scan.sectors_pruned_by_bound = (IVF_CELLS - plan_index) as u32;
                break;
            }

            let start = ivf.offsets[cell] as usize;
            let end = ivf.offsets[cell + 1] as usize;
            if start == end {
                continue;
            }

            let scan_end = remaining.map_or(end, |max| end.min(start + max));
            self.search_range_instrumented(start, scan_end, scan_query, neighbors, details);
            scan.points_scanned += scan_end - start;
            let sector = cell / IVF_SUBCELLS;
            if !sectors_touched[sector] {
                sectors_touched[sector] = true;
                scan.sectors_scanned += 1;
            }
            if scan_end < end {
                break;
            }
        }

        scan
    }

    fn search_sectorized_bucket(
        &self,
        sector_index: usize,
        bucket_lower_bound: i64,
        scan_query: &[i16; DIMS],
        sector_plan: &[(i64, usize); SECTORS],
        neighbors: &mut NeighborSet,
        max_points: Option<usize>,
    ) -> usize {
        let sectorized = &self.sectorized_buckets[sector_index];
        let mut points_scanned = 0;

        for &(sector_lower_bound, sector) in sector_plan.iter() {
            let remaining = max_points.map(|max| max.saturating_sub(points_scanned));
            if remaining == Some(0) {
                break;
            }
            if neighbors.is_full() && bucket_lower_bound + sector_lower_bound >= neighbors.worst() {
                break;
            }

            let start = sectorized.offsets[sector] as usize;
            let end = sectorized.offsets[sector + 1] as usize;
            if start == end {
                continue;
            }

            let scan_end = remaining.map_or(end, |max| end.min(start + max));
            self.search_range(start, scan_end, scan_query, neighbors);
            points_scanned += scan_end - start;
            if scan_end < end {
                break;
            }
        }

        points_scanned
    }

    fn search_sectorized_bucket_instrumented(
        &self,
        sector_index: usize,
        bucket_lower_bound: i64,
        scan_query: &[i16; DIMS],
        sector_plan: &[(i64, usize); SECTORS],
        neighbors: &mut NeighborSet,
        details: &mut SearchDetails,
        max_points: Option<usize>,
    ) -> SectorScan {
        let sectorized = &self.sectorized_buckets[sector_index];
        let mut scan = SectorScan::default();

        for (plan_index, &(sector_lower_bound, sector)) in sector_plan.iter().enumerate() {
            let remaining = max_points.map(|max| max.saturating_sub(scan.points_scanned));
            if remaining == Some(0) {
                break;
            }
            if neighbors.is_full() && bucket_lower_bound + sector_lower_bound >= neighbors.worst() {
                scan.sectors_pruned_by_bound = (SECTORS - plan_index) as u32;
                break;
            }

            let start = sectorized.offsets[sector] as usize;
            let end = sectorized.offsets[sector + 1] as usize;
            if start == end {
                continue;
            }

            let scan_end = remaining.map_or(end, |max| end.min(start + max));
            self.search_range_instrumented(start, scan_end, scan_query, neighbors, details);
            scan.points_scanned += scan_end - start;
            scan.sectors_scanned += 1;
            if scan_end < end {
                break;
            }
        }

        scan
    }

    fn search_range(
        &self,
        start: usize,
        end: usize,
        query: &[i16; DIMS],
        neighbors: &mut NeighborSet,
    ) {
        let len = end - start;
        if len == 0 {
            return;
        }

        let points = &self.points.as_slice()[start..end];

        if let Some(packed_hot2) = &self.packed_hot2 {
            let hot2 = &packed_hot2[start..end];
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            {
                if len >= 8 {
                    unsafe {
                        search_range_avx2_packed_hot2(
                            hot2,
                            points,
                            query,
                            neighbors,
                            self.avx2_prefetch_points,
                        );
                    }
                    return;
                }
            }

            #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
            {
                if len >= 8 && std::arch::is_x86_feature_detected!("avx2") {
                    unsafe {
                        search_range_avx2_packed_hot2(
                            hot2,
                            points,
                            query,
                            neighbors,
                            self.avx2_prefetch_points,
                        );
                    }
                    return;
                }
            }

            search_range_scalar_packed_hot2(hot2, points, query, neighbors);
            return;
        }

        let c0 = &self.hot_columns[0][start..end];
        let c1 = &self.hot_columns[1][start..end];

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            if len >= 8 {
                unsafe {
                    if self.avx2_prefix4 {
                        let prefix4 = self.prefix4_columns.as_ref().expect("prefix4 columns");
                        let c2 = &prefix4[0][start..end];
                        let c3 = &prefix4[1][start..end];
                        search_range_avx2_prefix4(
                            c0,
                            c1,
                            c2,
                            c3,
                            points,
                            query,
                            neighbors,
                            self.avx2_prefetch_points,
                        );
                    } else {
                        search_range_avx2_hot2(
                            c0,
                            c1,
                            points,
                            query,
                            neighbors,
                            self.avx2_prefetch_points,
                        );
                    }
                }
                return;
            }
        }

        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
        {
            if len >= 8 && std::arch::is_x86_feature_detected!("avx2") {
                unsafe {
                    if self.avx2_prefix4 {
                        let prefix4 = self.prefix4_columns.as_ref().expect("prefix4 columns");
                        let c2 = &prefix4[0][start..end];
                        let c3 = &prefix4[1][start..end];
                        search_range_avx2_prefix4(
                            c0,
                            c1,
                            c2,
                            c3,
                            points,
                            query,
                            neighbors,
                            self.avx2_prefetch_points,
                        );
                    } else {
                        search_range_avx2_hot2(
                            c0,
                            c1,
                            points,
                            query,
                            neighbors,
                            self.avx2_prefetch_points,
                        );
                    }
                }
                return;
            }
        }

        if self.avx2_prefix4 {
            let prefix4 = self.prefix4_columns.as_ref().expect("prefix4 columns");
            let c2 = &prefix4[0][start..end];
            let c3 = &prefix4[1][start..end];
            search_range_scalar_prefix4(c0, c1, c2, c3, points, query, neighbors);
            return;
        }

        'point: for index in 0..len {
            let ceiling = neighbors.best_dist[0];
            let mut acc = 0i64;

            macro_rules! step {
                ($dim:literal, $column:ident) => {{
                    let value = unsafe { *$column.get_unchecked(index) };
                    let delta = query[$dim] as i32 - value as i32;
                    acc += (delta * delta) as i64;
                    if acc >= ceiling {
                        continue 'point;
                    }
                }};
            }

            step!(0, c0);
            step!(1, c1);

            let point = unsafe { points.get_unchecked(index) };
            macro_rules! point_step {
                ($dim:literal) => {{
                    let delta = query[$dim] as i32 - point.values[$dim] as i32;
                    acc += (delta * delta) as i64;
                    if acc >= ceiling {
                        continue 'point;
                    }
                }};
            }

            point_step!(2);
            point_step!(3);
            point_step!(4);
            point_step!(5);
            point_step!(6);
            point_step!(7);
            point_step!(8);
            point_step!(9);
            point_step!(10);
            point_step!(11);
            point_step!(12);
            point_step!(13);

            neighbors.insert(acc, point.label);
        }
    }

    fn search_range_instrumented(
        &self,
        start: usize,
        end: usize,
        query: &[i16; DIMS],
        neighbors: &mut NeighborSet,
        details: &mut SearchDetails,
    ) {
        let len = end - start;
        if len == 0 {
            return;
        }

        let points = &self.points.as_slice()[start..end];

        if let Some(packed_hot2) = &self.packed_hot2 {
            let hot2 = &packed_hot2[start..end];
            search_range_instrumented_packed_hot2(hot2, points, query, neighbors, details);
            return;
        }

        let c0 = &self.hot_columns[0][start..end];
        let c1 = &self.hot_columns[1][start..end];

        'point: for index in 0..len {
            let ceiling = neighbors.best_dist[0];
            let mut acc = 0i64;

            macro_rules! step {
                ($dim:literal, $column:ident) => {{
                    details.distance_steps[$dim] += 1;
                    let value = unsafe { *$column.get_unchecked(index) };
                    let delta = query[$dim] as i32 - value as i32;
                    acc += (delta * delta) as i64;
                    if acc >= ceiling {
                        details.distance_cutoffs[$dim] += 1;
                        continue 'point;
                    }
                }};
            }

            step!(0, c0);
            step!(1, c1);

            let point = unsafe { points.get_unchecked(index) };
            macro_rules! point_step {
                ($dim:literal) => {{
                    details.distance_steps[$dim] += 1;
                    let delta = query[$dim] as i32 - point.values[$dim] as i32;
                    acc += (delta * delta) as i64;
                    if acc >= ceiling {
                        details.distance_cutoffs[$dim] += 1;
                        continue 'point;
                    }
                }};
            }

            point_step!(2);
            point_step!(3);
            point_step!(4);
            point_step!(5);
            point_step!(6);
            point_step!(7);
            point_step!(8);
            point_step!(9);
            point_step!(10);
            point_step!(11);
            point_step!(12);
            point_step!(13);

            details.distance_full += 1;
            details.neighbor_insertions += 1;
            neighbors.insert(acc, point.label);
        }
    }
}

#[inline(always)]
fn unpack_hot2(packed: u32) -> (i16, i16) {
    (packed as u16 as i16, (packed >> 16) as u16 as i16)
}

#[inline(always)]
fn search_range_scalar_packed_hot2(
    hot2: &[u32],
    points: &[Point],
    query: &[i16; DIMS],
    neighbors: &mut NeighborSet,
) {
    let len = points.len();

    'point: for index in 0..len {
        let ceiling = neighbors.best_dist[0];
        let packed = unsafe { *hot2.get_unchecked(index) };
        let (value0, value1) = unpack_hot2(packed);
        let delta0 = query[0] as i32 - value0 as i32;
        let mut acc = (delta0 * delta0) as i64;
        if acc >= ceiling {
            continue 'point;
        }
        let delta1 = query[1] as i32 - value1 as i32;
        acc += (delta1 * delta1) as i64;
        if acc >= ceiling {
            continue 'point;
        }

        let point = unsafe { points.get_unchecked(index) };
        search_point_tail_from_dim2(point, query, acc, ceiling, neighbors);
    }
}

#[inline(always)]
fn search_range_instrumented_packed_hot2(
    hot2: &[u32],
    points: &[Point],
    query: &[i16; DIMS],
    neighbors: &mut NeighborSet,
    details: &mut SearchDetails,
) {
    let len = points.len();

    'point: for index in 0..len {
        let ceiling = neighbors.best_dist[0];
        let packed = unsafe { *hot2.get_unchecked(index) };
        let (value0, value1) = unpack_hot2(packed);
        let delta0 = query[0] as i32 - value0 as i32;
        let delta1 = query[1] as i32 - value1 as i32;
        let acc0 = (delta0 * delta0) as i64;
        details.distance_steps[0] += 1;
        if acc0 >= ceiling {
            details.distance_cutoffs[0] += 1;
            continue 'point;
        }

        let acc = acc0 + (delta1 * delta1) as i64;
        details.distance_steps[1] += 1;
        if acc >= ceiling {
            details.distance_cutoffs[1] += 1;
            continue 'point;
        }

        let point = unsafe { points.get_unchecked(index) };
        let mut acc = acc;
        macro_rules! point_step {
            ($dim:literal) => {{
                details.distance_steps[$dim] += 1;
                let delta = query[$dim] as i32 - point.values[$dim] as i32;
                acc += (delta * delta) as i64;
                if acc >= ceiling {
                    details.distance_cutoffs[$dim] += 1;
                    continue 'point;
                }
            }};
        }

        point_step!(2);
        point_step!(3);
        point_step!(4);
        point_step!(5);
        point_step!(6);
        point_step!(7);
        point_step!(8);
        point_step!(9);
        point_step!(10);
        point_step!(11);
        point_step!(12);
        point_step!(13);

        details.distance_full += 1;
        details.neighbor_insertions += 1;
        neighbors.insert(acc, point.label);
    }
}

#[inline(always)]
fn search_range_scalar_prefix4(
    c0: &[i16],
    c1: &[i16],
    c2: &[i16],
    c3: &[i16],
    points: &[Point],
    query: &[i16; DIMS],
    neighbors: &mut NeighborSet,
) {
    let len = points.len();

    'point: for index in 0..len {
        let ceiling = neighbors.best_dist[0];
        let mut acc = 0i64;

        macro_rules! column_step {
            ($dim:literal, $column:ident) => {{
                let value = unsafe { *$column.get_unchecked(index) };
                let delta = query[$dim] as i32 - value as i32;
                acc += (delta * delta) as i64;
                if acc >= ceiling {
                    continue 'point;
                }
            }};
        }

        column_step!(0, c0);
        column_step!(1, c1);
        column_step!(2, c2);
        column_step!(3, c3);

        let point = unsafe { points.get_unchecked(index) };
        macro_rules! point_step {
            ($dim:literal) => {{
                let delta = query[$dim] as i32 - point.values[$dim] as i32;
                acc += (delta * delta) as i64;
                if acc >= ceiling {
                    continue 'point;
                }
            }};
        }

        point_step!(4);
        point_step!(5);
        point_step!(6);
        point_step!(7);
        point_step!(8);
        point_step!(9);
        point_step!(10);
        point_step!(11);
        point_step!(12);
        point_step!(13);

        neighbors.insert(acc, point.label);
    }
}

#[derive(Clone, Copy)]
enum AdaptivePhase {
    Early,
    Fast,
    Repair,
}

#[inline]
fn adaptive_clear_rule_matches(rule: AdaptiveClearRule, query: &[i16; DIMS]) -> bool {
    match rule {
        AdaptiveClearRule::None => true,
        AdaptiveClearRule::TailRatio70Tx45Home40 => {
            query[9] == 0
                && query[10] != 0
                && query[11] == 0
                && query[5] >= 10_000
                && query[2] <= 7_000
                && query[8] <= 4_500
                && query[7] <= 4_000
        }
        AdaptiveClearRule::TailMccNonGaming => {
            query[9] == 0
                && query[10] != 0
                && query[11] == 0
                && query[5] >= 10_000
                && query[2] <= 8_000
                && query[8] <= 5_000
                && query[12] < 7_500
        }
    }
}

#[inline]
fn score5_repair_rule_matches(rule: Score5RepairRule, query: &[i16; DIMS]) -> bool {
    match rule {
        Score5RepairRule::Fp70Tight => {
            query[2] >= 10_000
                && query[5] >= 10_000
                && (5_000..=6_000).contains(&query[8])
                && query[9] >= 10_000
                && query[10] == 0
                && query[11] >= 10_000
                && query[12] >= 7_500
                && query[6] >= 2_000
                && query[7] >= 3_000
        }
        Score5RepairRule::Fp70Narrow => {
            (2_900..=4_100).contains(&query[0])
                && (5_000..=7_500).contains(&query[1])
                && query[2] >= 10_000
                && query[3] <= 2_700
                && query[5] >= 10_000
                && (2_000..=3_300).contains(&query[6])
                && (3_600..=4_500).contains(&query[7])
                && (5_000..=6_000).contains(&query[8])
                && query[9] >= 10_000
                && query[10] == 0
                && query[11] >= 10_000
                && query[12] >= 7_500
        }
    }
}

pub fn order_points_by_bucket(points: &mut [Point]) {
    points.sort_unstable_by_key(|point| {
        (
            vector_bucket(&point.values),
            sector_id(point.values[5], point.values[6]),
        )
    });
    for point in points {
        point.values = scan_order(&point.values);
    }
}

pub fn order_points_by_bucket_ivf(points: &mut [Point], dims: (usize, usize)) {
    points.sort_unstable_by_key(|point| {
        let scan_values = scan_order(&point.values);
        (vector_bucket(&point.values), ivf_cell(&scan_values, dims))
    });
    for point in points {
        point.values = scan_order(&point.values);
    }
}

#[inline]
fn insert_neighbor(best_dist: &mut [i64; 5], best_label: &mut [u8; 5], dist: i64, label: u8) {
    best_dist[0] = dist;
    best_label[0] = label;

    let mut i = 0;
    while i + 1 < 5 && best_dist[i] < best_dist[i + 1] {
        best_dist.swap(i, i + 1);
        best_label.swap(i, i + 1);
        i += 1;
    }
}

struct NeighborSet {
    best_dist: [i64; 5],
    best_label: [u8; 5],
}

impl NeighborSet {
    fn insert(&mut self, dist: i64, label: u8) {
        insert_neighbor(&mut self.best_dist, &mut self.best_label, dist, label);
    }

    fn is_full(&self) -> bool {
        self.best_dist[0] != i64::MAX
    }

    fn worst(&self) -> i64 {
        self.best_dist[0]
    }

    fn result(&self) -> SearchResult {
        SearchResult {
            fraud_count: self.best_label.iter().copied().sum(),
        }
    }
}

impl Default for NeighborSet {
    fn default() -> Self {
        Self {
            best_dist: [i64::MAX; 5],
            best_label: [0; 5],
        }
    }
}

#[inline(always)]
fn search_point_tail_from_dim2(
    point: &Point,
    query: &[i16; DIMS],
    mut acc: i64,
    ceiling: i64,
    neighbors: &mut NeighborSet,
) {
    if acc >= ceiling {
        return;
    }

    macro_rules! point_step {
        ($dim:literal) => {{
            let delta = query[$dim] as i32 - point.values[$dim] as i32;
            acc += (delta * delta) as i64;
            if acc >= ceiling {
                return;
            }
        }};
    }

    point_step!(2);
    point_step!(3);
    point_step!(4);
    point_step!(5);
    point_step!(6);
    point_step!(7);
    point_step!(8);
    point_step!(9);
    point_step!(10);
    point_step!(11);
    point_step!(12);
    point_step!(13);

    neighbors.insert(acc, point.label);
}

#[inline(always)]
#[allow(dead_code)]
fn search_point_tail_from_dim4(
    point: &Point,
    query: &[i16; DIMS],
    mut acc: i64,
    ceiling: i64,
    neighbors: &mut NeighborSet,
) {
    if acc >= ceiling {
        return;
    }

    macro_rules! point_step {
        ($dim:literal) => {{
            let delta = query[$dim] as i32 - point.values[$dim] as i32;
            acc += (delta * delta) as i64;
            if acc >= ceiling {
                return;
            }
        }};
    }

    point_step!(4);
    point_step!(5);
    point_step!(6);
    point_step!(7);
    point_step!(8);
    point_step!(9);
    point_step!(10);
    point_step!(11);
    point_step!(12);
    point_step!(13);

    neighbors.insert(acc, point.label);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn search_range_avx2_packed_hot2(
    hot2: &[u32],
    points: &[Point],
    query: &[i16; DIMS],
    neighbors: &mut NeighborSet,
    prefetch_points: usize,
) {
    use std::arch::x86_64::*;

    let len = points.len();
    let q0 = _mm256_set1_epi32(query[0] as i32);
    let q1 = _mm256_set1_epi32(query[1] as i32);
    let mut index = 0usize;
    let mut acc_values = [0i32; 8];

    while index + 8 <= len {
        if prefetch_points != 0 {
            let prefetch_index = index + prefetch_points;
            if prefetch_index < len {
                _mm_prefetch(
                    points.as_ptr().add(prefetch_index) as *const i8,
                    _MM_HINT_T0,
                );
            }
        }

        let ceiling = neighbors.best_dist[0];
        let mask = if ceiling > i32::MAX as i64 {
            0xff
        } else {
            let packed = _mm256_loadu_si256(hot2.as_ptr().add(index) as *const _);
            let v0 = _mm256_srai_epi32(_mm256_slli_epi32(packed, 16), 16);
            let v1 = _mm256_srai_epi32(packed, 16);
            let d0 = _mm256_sub_epi32(q0, v0);
            let d1 = _mm256_sub_epi32(q1, v1);
            let s0 = _mm256_mullo_epi32(d0, d0);
            let s1 = _mm256_mullo_epi32(d1, d1);
            let acc = _mm256_add_epi32(s0, s1);
            let ceiling = _mm256_set1_epi32(ceiling as i32);
            let mask =
                _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(ceiling, acc))) as u32;
            if mask != 0 {
                _mm256_storeu_si256(acc_values.as_mut_ptr() as *mut _, acc);
            }
            mask
        };

        let mut lane = 0usize;
        while lane < 8 {
            if (mask & (1 << lane)) != 0 {
                let point = points.get_unchecked(index + lane);
                let acc = if ceiling > i32::MAX as i64 {
                    let (value0, value1) = unpack_hot2(*hot2.get_unchecked(index + lane));
                    let delta0 = query[0] as i32 - value0 as i32;
                    let delta1 = query[1] as i32 - value1 as i32;
                    (delta0 * delta0 + delta1 * delta1) as i64
                } else {
                    acc_values[lane] as i64
                };
                search_point_tail_from_dim2(point, query, acc, neighbors.best_dist[0], neighbors);
            }
            lane += 1;
        }

        index += 8;
    }

    while index < len {
        let ceiling = neighbors.best_dist[0];
        let (value0, value1) = unpack_hot2(*hot2.get_unchecked(index));
        let delta0 = query[0] as i32 - value0 as i32;
        let delta1 = query[1] as i32 - value1 as i32;
        let acc = (delta0 * delta0 + delta1 * delta1) as i64;
        let point = points.get_unchecked(index);
        search_point_tail_from_dim2(point, query, acc, ceiling, neighbors);
        index += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn search_range_avx2_hot2(
    c0: &[i16],
    c1: &[i16],
    points: &[Point],
    query: &[i16; DIMS],
    neighbors: &mut NeighborSet,
    prefetch_points: usize,
) {
    use std::arch::x86_64::*;

    let len = points.len();
    let q0 = _mm256_set1_epi32(query[0] as i32);
    let q1 = _mm256_set1_epi32(query[1] as i32);
    let mut index = 0usize;
    let mut acc_values = [0i32; 8];

    while index + 8 <= len {
        if prefetch_points != 0 {
            let prefetch_index = index + prefetch_points;
            if prefetch_index < len {
                _mm_prefetch(
                    points.as_ptr().add(prefetch_index) as *const i8,
                    _MM_HINT_T0,
                );
            }
        }

        let ceiling = neighbors.best_dist[0];
        let mask = if ceiling > i32::MAX as i64 {
            0xff
        } else {
            let v0 = _mm256_cvtepi16_epi32(_mm_loadu_si128(c0.as_ptr().add(index) as *const _));
            let v1 = _mm256_cvtepi16_epi32(_mm_loadu_si128(c1.as_ptr().add(index) as *const _));
            let d0 = _mm256_sub_epi32(q0, v0);
            let d1 = _mm256_sub_epi32(q1, v1);
            let s0 = _mm256_mullo_epi32(d0, d0);
            let s1 = _mm256_mullo_epi32(d1, d1);
            let acc = _mm256_add_epi32(s0, s1);
            let ceiling = _mm256_set1_epi32(ceiling as i32);
            let mask =
                _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(ceiling, acc))) as u32;
            if mask != 0 {
                _mm256_storeu_si256(acc_values.as_mut_ptr() as *mut _, acc);
            }
            mask
        };

        let mut lane = 0usize;
        while lane < 8 {
            if (mask & (1 << lane)) != 0 {
                let point = points.get_unchecked(index + lane);
                let acc = if ceiling > i32::MAX as i64 {
                    let delta0 = query[0] as i32 - *c0.get_unchecked(index + lane) as i32;
                    let delta1 = query[1] as i32 - *c1.get_unchecked(index + lane) as i32;
                    (delta0 * delta0 + delta1 * delta1) as i64
                } else {
                    acc_values[lane] as i64
                };
                search_point_tail_from_dim2(point, query, acc, neighbors.best_dist[0], neighbors);
            }
            lane += 1;
        }

        index += 8;
    }

    while index < len {
        let ceiling = neighbors.best_dist[0];
        let delta0 = query[0] as i32 - *c0.get_unchecked(index) as i32;
        let delta1 = query[1] as i32 - *c1.get_unchecked(index) as i32;
        let acc = (delta0 * delta0 + delta1 * delta1) as i64;
        let point = points.get_unchecked(index);
        search_point_tail_from_dim2(point, query, acc, ceiling, neighbors);
        index += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn search_range_avx2_prefix4(
    c0: &[i16],
    c1: &[i16],
    c2: &[i16],
    c3: &[i16],
    points: &[Point],
    query: &[i16; DIMS],
    neighbors: &mut NeighborSet,
    prefetch_points: usize,
) {
    use std::arch::x86_64::*;

    let len = points.len();
    let q0 = _mm256_set1_epi32(query[0] as i32);
    let q1 = _mm256_set1_epi32(query[1] as i32);
    let q2 = _mm256_set1_epi32(query[2] as i32);
    let q3 = _mm256_set1_epi32(query[3] as i32);
    let mut index = 0usize;
    let mut acc_values = [0i32; 8];

    while index + 8 <= len {
        if prefetch_points != 0 {
            let prefetch_index = index + prefetch_points;
            if prefetch_index < len {
                _mm_prefetch(
                    points.as_ptr().add(prefetch_index) as *const i8,
                    _MM_HINT_T0,
                );
            }
        }

        let ceiling = neighbors.best_dist[0];
        let mask = if ceiling > i32::MAX as i64 {
            0xff
        } else {
            let v0 = _mm256_cvtepi16_epi32(_mm_loadu_si128(c0.as_ptr().add(index) as *const _));
            let v1 = _mm256_cvtepi16_epi32(_mm_loadu_si128(c1.as_ptr().add(index) as *const _));
            let d0 = _mm256_sub_epi32(q0, v0);
            let d1 = _mm256_sub_epi32(q1, v1);
            let s0 = _mm256_mullo_epi32(d0, d0);
            let s1 = _mm256_mullo_epi32(d1, d1);
            let acc = _mm256_add_epi32(s0, s1);
            let ceiling = _mm256_set1_epi32(ceiling as i32);
            let mask2 =
                _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(ceiling, acc))) as u32;
            if mask2 == 0 {
                0
            } else {
                let v2 = _mm256_cvtepi16_epi32(_mm_loadu_si128(c2.as_ptr().add(index) as *const _));
                let v3 = _mm256_cvtepi16_epi32(_mm_loadu_si128(c3.as_ptr().add(index) as *const _));
                let d2 = _mm256_sub_epi32(q2, v2);
                let d3 = _mm256_sub_epi32(q3, v3);
                let s2 = _mm256_mullo_epi32(d2, d2);
                let s3 = _mm256_mullo_epi32(d3, d3);
                let acc = _mm256_add_epi32(_mm256_add_epi32(acc, s2), s3);
                let mask = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(ceiling, acc)))
                    as u32;
                if mask != 0 {
                    _mm256_storeu_si256(acc_values.as_mut_ptr() as *mut _, acc);
                }
                mask
            }
        };

        let mut lane = 0usize;
        while lane < 8 {
            if (mask & (1 << lane)) != 0 {
                let point = points.get_unchecked(index + lane);
                let acc = if ceiling > i32::MAX as i64 {
                    let delta0 = query[0] as i32 - *c0.get_unchecked(index + lane) as i32;
                    let delta1 = query[1] as i32 - *c1.get_unchecked(index + lane) as i32;
                    (delta0 * delta0 + delta1 * delta1) as i64
                } else {
                    acc_values[lane] as i64
                };
                if ceiling > i32::MAX as i64 {
                    search_point_tail_from_dim2(
                        point,
                        query,
                        acc,
                        neighbors.best_dist[0],
                        neighbors,
                    );
                } else {
                    search_point_tail_from_dim4(
                        point,
                        query,
                        acc,
                        neighbors.best_dist[0],
                        neighbors,
                    );
                }
            }
            lane += 1;
        }

        index += 8;
    }

    while index < len {
        let ceiling = neighbors.best_dist[0];
        let delta0 = query[0] as i32 - *c0.get_unchecked(index) as i32;
        let delta1 = query[1] as i32 - *c1.get_unchecked(index) as i32;
        let mut acc = (delta0 * delta0 + delta1 * delta1) as i64;
        if acc < ceiling {
            let delta2 = query[2] as i32 - *c2.get_unchecked(index) as i32;
            let delta3 = query[3] as i32 - *c3.get_unchecked(index) as i32;
            acc += (delta2 * delta2 + delta3 * delta3) as i64;
            if acc < ceiling {
                let point = points.get_unchecked(index);
                search_point_tail_from_dim4(point, query, acc, ceiling, neighbors);
            }
        }
        index += 1;
    }
}

fn bucket_offsets(points: &[Point]) -> Vec<u32> {
    let mut counts = vec![0u32; BUCKETS];
    for point in points {
        counts[point_bucket(&point.values)] += 1;
    }

    let mut offsets = vec![0u32; BUCKETS + 1];
    for bucket in 0..BUCKETS {
        offsets[bucket + 1] = offsets[bucket] + counts[bucket];
    }
    offsets
}

fn bucket_approval_counts(points: &[Point], offsets: &[u32]) -> Vec<BucketApprovalCounts> {
    let mut counts = vec![BucketApprovalCounts::default(); BUCKETS];
    for bucket in 0..BUCKETS {
        let start = offsets[bucket] as usize;
        let end = offsets[bucket + 1] as usize;
        let mut approved = 0u32;
        let mut denied = 0u32;
        for point in &points[start..end] {
            if point.label < 3 {
                approved += 1;
            } else {
                denied += 1;
            }
        }
        counts[bucket] = BucketApprovalCounts { approved, denied };
    }
    counts
}

fn sector_approval_counts(points: &[Point], offsets: &[u32]) -> Vec<BucketApprovalCounts> {
    let mut counts = vec![BucketApprovalCounts::default(); BUCKETS * SECTORS];
    for bucket in 0..BUCKETS {
        let start = offsets[bucket] as usize;
        let end = offsets[bucket + 1] as usize;
        for point in &points[start..end] {
            let index = bucket * SECTORS + point_sector(&point.values);
            if point.label < 3 {
                counts[index].approved += 1;
            } else {
                counts[index].denied += 1;
            }
        }
    }
    counts
}

#[inline]
fn majority_decision(
    counts: BucketApprovalCounts,
    min_total: u32,
    max_error_per_10000: u32,
) -> Option<u8> {
    let total = counts.approved + counts.denied;
    if total < min_total || total == 0 {
        return None;
    }

    let errors = counts.approved.min(counts.denied);
    if (errors as u64) * 10_000 > (max_error_per_10000 as u64) * (total as u64) {
        return None;
    }

    if counts.approved >= counts.denied {
        Some(0)
    } else {
        Some(3)
    }
}

fn hot_columns(points: &[Point]) -> [Vec<i16>; HOT_DIMS] {
    let mut hot_columns: [Vec<i16>; HOT_DIMS] =
        std::array::from_fn(|_| Vec::with_capacity(points.len()));
    for point in points {
        for (column, value) in hot_columns.iter_mut().zip(point.values) {
            column.push(value);
        }
    }
    hot_columns
}

fn empty_hot_columns() -> [Vec<i16>; HOT_DIMS] {
    std::array::from_fn(|_| Vec::new())
}

fn packed_hot2_columns(points: &[Point]) -> Vec<u32> {
    let mut packed = Vec::with_capacity(points.len());
    for point in points {
        let value0 = point.values[0] as u16 as u32;
        let value1 = point.values[1] as u16 as u32;
        packed.push(value0 | (value1 << 16));
    }
    packed
}

fn prefix4_columns(points: &[Point]) -> [Vec<i16>; PREFIX4_EXTRA_DIMS] {
    let mut columns: [Vec<i16>; PREFIX4_EXTRA_DIMS] =
        std::array::from_fn(|_| Vec::with_capacity(points.len()));
    for point in points {
        columns[0].push(point.values[2]);
        columns[1].push(point.values[3]);
    }
    columns
}

unsafe fn mapped_points(mmap: &Mmap, offset: usize, len: usize) -> &[Point] {
    std::slice::from_raw_parts(mmap.as_ptr().add(offset) as *const Point, len)
}

fn owned_points_from_mmap(mmap: &Mmap, offset: usize, len: usize) -> Vec<Point> {
    unsafe { mapped_points(mmap, offset, len) }.to_vec()
}

fn invalid_index(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn index_env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
}

fn should_build_sector_decision_counts() -> bool {
    index_env_flag("SECTOR_DECISION")
        || std::env::var("FAST_DECISION_RULE").is_ok_and(|value| value == "sector-majority")
}

fn index_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn ivf_dims_from_env() -> (usize, usize) {
    let value = std::env::var("IVF_DIMS").unwrap_or_else(|_| "2,4".to_owned());
    parse_ivf_dims(&value)
}

pub fn parse_ivf_dims(value: &str) -> (usize, usize) {
    let Some((first, second)) = value.split_once(',').or_else(|| value.split_once('-')) else {
        panic!("IVF_DIMS must be formatted as DIM,DIM");
    };
    let first = first.parse::<usize>().expect("valid IVF first dim");
    let second = second.parse::<usize>().expect("valid IVF second dim");
    assert!(first < DIMS && second < DIMS, "IVF_DIMS out of range");
    assert!(
        first != second && first >= 2 && second >= 2,
        "IVF_DIMS must use distinct non-sector scan-order dims"
    );
    (first, second)
}

fn ivf_buckets_from_ordered_points(
    points: &[Point],
    bucket_offsets: &[u32],
    dims: (usize, usize),
) -> (Vec<u16>, Vec<IvfBucket>) {
    let threshold = index_env_usize("IVF_THRESHOLD", DEFAULT_IVF_THRESHOLD);
    let mut bucket_map = vec![NO_SECTOR; BUCKETS];
    let mut buckets = Vec::new();
    let mut skipped_unordered = 0usize;

    for bucket in 0..BUCKETS {
        let start = bucket_offsets[bucket] as usize;
        let end = bucket_offsets[bucket + 1] as usize;
        let len = end - start;
        if len < threshold || !bucket_has_last_transaction(bucket) {
            continue;
        }

        let mut counts = Box::new([0usize; IVF_CELLS]);
        let mut previous_cell = 0usize;
        for (index, point) in points[start..end].iter().enumerate() {
            let cell = ivf_cell(&point.values, dims);
            if index > 0 && cell < previous_cell {
                skipped_unordered += 1;
                counts.fill(0);
                break;
            }
            previous_cell = cell;
            counts[cell] += 1;
        }
        if counts.iter().filter(|count| **count > 0).count() <= 1 {
            continue;
        }

        let mut offsets = Box::new([0u32; IVF_CELLS + 1]);
        offsets[0] = start as u32;
        for cell in 0..IVF_CELLS {
            offsets[cell + 1] = offsets[cell] + counts[cell] as u32;
        }

        bucket_map[bucket] = buckets.len() as u16;
        buckets.push(IvfBucket { offsets });
    }

    eprintln!(
        "ivf_grid mmap threshold={} dims={:?} buckets={} skipped_unordered={}",
        threshold,
        dims,
        buckets.len(),
        skipped_unordered
    );
    (bucket_map, buckets)
}

fn ivf_grid_large_buckets(
    points: &mut [Point],
    bucket_offsets: &[u32],
    dims: (usize, usize),
) -> (Vec<u16>, Vec<IvfBucket>) {
    let threshold = index_env_usize("IVF_THRESHOLD", DEFAULT_IVF_THRESHOLD);
    let mut bucket_map = vec![NO_SECTOR; BUCKETS];
    let mut buckets = Vec::new();
    let mut scratch = Vec::new();

    for bucket in 0..BUCKETS {
        let start = bucket_offsets[bucket] as usize;
        let end = bucket_offsets[bucket + 1] as usize;
        let len = end - start;
        if len < threshold || !bucket_has_last_transaction(bucket) {
            continue;
        }

        let mut counts = Box::new([0usize; IVF_CELLS]);
        for point in &points[start..end] {
            counts[ivf_cell(&point.values, dims)] += 1;
        }
        if counts.iter().filter(|count| **count > 0).count() <= 1 {
            continue;
        }

        let mut offsets = Box::new([0u32; IVF_CELLS + 1]);
        offsets[0] = start as u32;
        for cell in 0..IVF_CELLS {
            offsets[cell + 1] = offsets[cell] + counts[cell] as u32;
        }

        scratch.clear();
        scratch.resize(len, Point::default());
        let mut write_offsets = vec![0usize; IVF_CELLS];
        for cell in 0..IVF_CELLS {
            write_offsets[cell] = offsets[cell] as usize - start;
        }
        for point in &points[start..end] {
            let cell = ivf_cell(&point.values, dims);
            let write = write_offsets[cell];
            scratch[write] = *point;
            write_offsets[cell] += 1;
        }
        points[start..end].copy_from_slice(&scratch);

        bucket_map[bucket] = buckets.len() as u16;
        buckets.push(IvfBucket { offsets });
    }

    eprintln!(
        "ivf_grid enabled threshold={} dims={:?} buckets={}",
        threshold,
        dims,
        buckets.len()
    );
    (bucket_map, buckets)
}

fn sectorize_large_buckets(
    points: &mut [Point],
    bucket_offsets: &[u32],
) -> (Vec<u16>, Vec<SectorizedBucket>) {
    let mut bucket_sector_map = vec![NO_SECTOR; BUCKETS];
    let mut sectorized_buckets = Vec::new();
    let mut scratch = Vec::new();

    for bucket in 0..BUCKETS {
        let start = bucket_offsets[bucket] as usize;
        let end = bucket_offsets[bucket + 1] as usize;
        let len = end - start;
        if len < SECTORIZE_THRESHOLD || !bucket_has_last_transaction(bucket) {
            continue;
        }

        let mut counts = [0usize; SECTORS];
        for point in &points[start..end] {
            counts[point_sector(&point.values)] += 1;
        }
        if counts.iter().filter(|count| **count > 0).count() <= 1 {
            continue;
        }

        let mut offsets = [0u32; SECTORS + 1];
        offsets[0] = start as u32;
        for sector in 0..SECTORS {
            offsets[sector + 1] = offsets[sector] + counts[sector] as u32;
        }

        scratch.clear();
        scratch.resize(len, Point::default());
        let mut write_offsets = [0usize; SECTORS];
        for sector in 0..SECTORS {
            write_offsets[sector] = offsets[sector] as usize - start;
        }
        for point in &points[start..end] {
            let sector = point_sector(&point.values);
            let write = write_offsets[sector];
            scratch[write] = *point;
            write_offsets[sector] += 1;
        }
        points[start..end].copy_from_slice(&scratch);

        bucket_sector_map[bucket] = sectorized_buckets.len() as u16;
        sectorized_buckets.push(SectorizedBucket { offsets });
    }

    (bucket_sector_map, sectorized_buckets)
}

fn sectorized_buckets_from_ordered_points(
    points: &[Point],
    bucket_offsets: &[u32],
) -> Option<(Vec<u16>, Vec<SectorizedBucket>)> {
    let mut bucket_sector_map = vec![NO_SECTOR; BUCKETS];
    let mut sectorized_buckets = Vec::new();

    for bucket in 0..BUCKETS {
        let start = bucket_offsets[bucket] as usize;
        let end = bucket_offsets[bucket + 1] as usize;
        let len = end - start;
        if len < SECTORIZE_THRESHOLD || !bucket_has_last_transaction(bucket) {
            continue;
        }

        let mut counts = [0usize; SECTORS];
        let mut previous_sector = 0usize;
        for (index, point) in points[start..end].iter().enumerate() {
            let sector = point_sector(&point.values);
            if index > 0 && sector < previous_sector {
                return None;
            }
            previous_sector = sector;
            counts[sector] += 1;
        }
        if counts.iter().filter(|count| **count > 0).count() <= 1 {
            continue;
        }

        let mut offsets = [0u32; SECTORS + 1];
        offsets[0] = start as u32;
        for sector in 0..SECTORS {
            offsets[sector + 1] = offsets[sector] + counts[sector] as u32;
        }

        bucket_sector_map[bucket] = sectorized_buckets.len() as u16;
        sectorized_buckets.push(SectorizedBucket { offsets });
    }

    Some((bucket_sector_map, sectorized_buckets))
}

#[inline]
fn bucket_has_last_transaction(bucket: usize) -> bool {
    (bucket % FLAG_BUCKETS) & 1 == 1
}

#[inline]
fn point_sector(values: &[i16; DIMS]) -> usize {
    sector_id(values[0], values[1])
}

#[inline]
fn sector_decision_key_query(values: &[i16; DIMS]) -> usize {
    vector_bucket(values) * SECTORS + sector_id(values[5], values[6])
}

fn sector_scan_plan(query: &[i16; DIMS]) -> [(i64, usize); SECTORS] {
    let mut plan = [(0i64, 0usize); SECTORS];
    let mut sector = 0;
    while sector < SECTORS {
        let s0 = sector / SECTOR_BUCKETS;
        let s1 = sector % SECTOR_BUCKETS;
        plan[sector] = (
            sector_lower_bound(query[0], s0) + sector_lower_bound(query[1], s1),
            sector,
        );
        sector += 1;
    }
    plan.sort_unstable_by_key(|entry| entry.0);
    plan
}

fn ivf_sub_scan_plan(query: &[i16; DIMS], dims: (usize, usize)) -> [(i64, usize); IVF_SUBCELLS] {
    let mut plan = [(0i64, 0usize); IVF_SUBCELLS];
    let mut subcell = 0;
    while subcell < IVF_SUBCELLS {
        let s0 = subcell / IVF_SUB_BUCKETS;
        let s1 = subcell % IVF_SUB_BUCKETS;
        plan[subcell] = (
            ivf_sub_lower_bound(query[dims.0], s0) + ivf_sub_lower_bound(query[dims.1], s1),
            subcell,
        );
        subcell += 1;
    }
    plan.sort_unstable_by_key(|entry| entry.0);
    plan
}

fn ivf_cell_scan_plan(query: &[i16; DIMS], dims: (usize, usize)) -> [(i64, usize); IVF_CELLS] {
    let mut plan = [(0i64, 0usize); IVF_CELLS];
    let mut cell = 0;
    while cell < IVF_CELLS {
        let sector = cell / IVF_SUBCELLS;
        let subcell = cell % IVF_SUBCELLS;
        let sector0 = sector / SECTOR_BUCKETS;
        let sector1 = sector % SECTOR_BUCKETS;
        let sub0 = subcell / IVF_SUB_BUCKETS;
        let sub1 = subcell % IVF_SUB_BUCKETS;
        plan[cell] = (
            sector_lower_bound(query[0], sector0)
                + sector_lower_bound(query[1], sector1)
                + ivf_sub_lower_bound(query[dims.0], sub0)
                + ivf_sub_lower_bound(query[dims.1], sub1),
            cell,
        );
        cell += 1;
    }
    plan.sort_unstable_by_key(|entry| entry.0);
    plan
}

#[inline]
fn sector_id(first: i16, second: i16) -> usize {
    sector_coord(first) * SECTOR_BUCKETS + sector_coord(second)
}

#[inline]
fn ivf_cell(values: &[i16; DIMS], dims: (usize, usize)) -> usize {
    point_sector(values) * IVF_SUBCELLS + ivf_subcell(values[dims.0], values[dims.1])
}

#[inline]
fn ivf_subcell(first: i16, second: i16) -> usize {
    ivf_sub_coord(first) * IVF_SUB_BUCKETS + ivf_sub_coord(second)
}

#[inline]
fn ivf_sub_coord(value: i16) -> usize {
    ((value.max(0) as usize) / IVF_SUB_WIDTH).min(IVF_SUB_BUCKETS - 1)
}

#[inline]
fn ivf_sub_lower_bound(query_value: i16, subcell: usize) -> i64 {
    let lo = (subcell * IVF_SUB_WIDTH) as i16;
    let hi = if subcell + 1 == IVF_SUB_BUCKETS {
        10_000
    } else {
        ((subcell + 1) * IVF_SUB_WIDTH - 1) as i16
    };
    interval_lower_bound(query_value, lo, hi)
}

#[inline]
fn sector_coord(value: i16) -> usize {
    ((value.max(0) as usize) / SECTOR_WIDTH).min(SECTOR_BUCKETS - 1)
}

#[inline]
fn sector_lower_bound(query_value: i16, sector: usize) -> i64 {
    let lo = (sector * SECTOR_WIDTH) as i16;
    let hi = if sector + 1 == SECTOR_BUCKETS {
        10_000
    } else {
        ((sector + 1) * SECTOR_WIDTH - 1) as i16
    };
    interval_lower_bound(query_value, lo, hi)
}

#[inline]
fn scan_order(values: &[i16; DIMS]) -> [i16; DIMS] {
    [
        values[5], values[6], values[0], values[2], values[7], values[8], values[12], values[1],
        values[11], values[9], values[10], values[3], values[4], values[13],
    ]
}

#[inline]
fn point_bucket(values: &[i16; DIMS]) -> usize {
    let flags = point_flags(values);
    let mcc = mcc_bucket(values[6]);
    let tx = tx_bucket(values[5]);
    let ratio = ratio_bucket(values[3]);

    (((ratio * TX_BUCKETS + tx) * MCC_BUCKETS + mcc) * FLAG_BUCKETS) + flags
}

#[inline]
fn point_flags(values: &[i16; DIMS]) -> usize {
    let last_present = (values[0] != -10_000) as usize;
    let is_online = (values[9] != 0) as usize;
    let card_present = (values[10] != 0) as usize;
    let merchant_unknown = (values[8] != 0) as usize;

    last_present | (is_online << 1) | (card_present << 2) | (merchant_unknown << 3)
}

fn bucket_entries(offsets: &[u32]) -> Vec<BucketEntry> {
    let mut buckets = Vec::new();
    for bucket in 0..BUCKETS {
        if offsets[bucket] != offsets[bucket + 1] {
            buckets.push(BucketEntry {
                bucket,
                meta: bucket_meta(bucket),
            });
        }
    }
    buckets
}

#[inline]
fn vector_bucket(values: &[i16; DIMS]) -> usize {
    let flags = vector_flags(values);
    let mcc = mcc_bucket(values[12]);
    let tx = tx_bucket(values[8]);
    let ratio = ratio_bucket(values[2]);

    (((ratio * TX_BUCKETS + tx) * MCC_BUCKETS + mcc) * FLAG_BUCKETS) + flags
}

fn vector_flags(values: &[i16; DIMS]) -> usize {
    let last_present = (values[5] != -10_000) as usize;
    let is_online = (values[9] != 0) as usize;
    let card_present = (values[10] != 0) as usize;
    let merchant_unknown = (values[11] != 0) as usize;

    last_present | (is_online << 1) | (card_present << 2) | (merchant_unknown << 3)
}

fn bucket_lower_bound(query: &[i16; DIMS], query_flags: usize, meta: BucketMeta) -> i64 {
    let flags = meta.flags as usize;
    let mut lower_bound = 0i64;

    if (query_flags & 0b0001) != (flags & 0b0001) {
        lower_bound += if (query_flags & 0b0001) != 0 {
            squared_delta(query[5], -10_000) + squared_delta(query[6], -10_000)
        } else {
            200_000_000
        };
    }

    if (query_flags & 0b0010) != (flags & 0b0010) {
        lower_bound += 100_000_000;
    }
    if (query_flags & 0b0100) != (flags & 0b0100) {
        lower_bound += 100_000_000;
    }
    if (query_flags & 0b1000) != (flags & 0b1000) {
        lower_bound += 100_000_000;
    }

    lower_bound += squared_delta(query[12], meta.mcc_value);
    lower_bound += squared_delta(query[8], meta.tx_value);
    lower_bound += interval_lower_bound(query[2], meta.ratio_lo, meta.ratio_hi);
    lower_bound
}

fn bucket_meta(coarse: usize) -> BucketMeta {
    let (flags, mcc, tx, ratio) = decode_coarse_bucket(coarse);
    let ratio_lo = (ratio * RATIO_WIDTH) as i16;
    let ratio_hi = if ratio + 1 == RATIO_BUCKETS {
        10_000
    } else {
        ((ratio + 1) * RATIO_WIDTH - 1) as i16
    };

    BucketMeta {
        flags: flags as u8,
        mcc_value: MCC_VALUES[mcc],
        tx_value: (tx as i16) * 500,
        ratio_lo,
        ratio_hi,
    }
}

fn decode_coarse_bucket(coarse: usize) -> (usize, usize, usize, usize) {
    let flags = coarse % FLAG_BUCKETS;
    let rem = coarse / FLAG_BUCKETS;
    let mcc = rem % MCC_BUCKETS;
    let rem = rem / MCC_BUCKETS;
    let tx = rem % TX_BUCKETS;
    let ratio = rem / TX_BUCKETS;
    (flags, mcc, tx, ratio)
}

#[inline]
fn mcc_bucket(value: i16) -> usize {
    match value {
        1_500 => 0,
        2_000 => 1,
        2_500 => 2,
        3_000 => 3,
        3_500 => 4,
        4_500 => 5,
        5_000 => 6,
        7_500 => 7,
        8_000 => 8,
        8_500 => 9,
        _ => 6,
    }
}

#[inline]
fn tx_bucket(value: i16) -> usize {
    ((value.max(0) as usize) / 500).min(TX_BUCKETS - 1)
}

#[inline]
fn ratio_bucket(value: i16) -> usize {
    ((value.max(0) as usize) / RATIO_WIDTH).min(RATIO_BUCKETS - 1)
}

#[inline]
fn interval_lower_bound(query_value: i16, lo: i16, hi: i16) -> i64 {
    if query_value < lo {
        squared_delta(query_value, lo)
    } else if query_value > hi {
        squared_delta(query_value, hi)
    } else {
        0
    }
}

#[inline]
fn squared_delta(left: i16, right: i16) -> i64 {
    let delta = left as i32 - right as i32;
    (delta * delta) as i64
}
