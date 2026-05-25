#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAX_DEPTH = 24
MIN_GAIN = 1.0e-12

MCC_RISK = {
    "5411": 0.15,
    "5812": 0.30,
    "5912": 0.20,
    "5944": 0.45,
    "7801": 0.80,
    "7802": 0.75,
    "7995": 0.85,
    "4511": 0.35,
    "5311": 0.25,
    "5999": 0.50,
}

FEATURE_NAMES = [
    "amount_10k",
    "installments_12",
    "amount_vs_customer_avg_10x",
    "requested_hour",
    "requested_weekday",
    "last_delta_days",
    "last_km",
    "km_from_home",
    "tx_count_24h",
    "is_online",
    "card_present",
    "merchant_unknown",
    "mcc_risk",
    "merchant_avg_10k",
]


@dataclass
class TreeNode:
    feature: int
    threshold: np.float32
    left: "TreeNode | None"
    right: "TreeNode | None"
    fraud: bool
    samples: int
    positives: int


def clamp_upper1(value: float) -> float:
    return 1.0 if value >= 1.0 else value


def parse_timestamp(value: str) -> tuple[int, int, int]:
    parsed = dt.datetime(
        int(value[0:4]),
        int(value[5:7]),
        int(value[8:10]),
        int(value[11:13]),
        int(value[14:16]),
        int(value[17:19]),
        tzinfo=dt.timezone.utc,
    )
    return int(parsed.timestamp()), parsed.hour, parsed.weekday()


def request_features(request: dict) -> list[float]:
    transaction = request["transaction"]
    customer = request["customer"]
    merchant = request["merchant"]
    terminal = request["terminal"]
    last_transaction = request.get("last_transaction")

    amount = float(transaction["amount"])
    customer_avg = float(customer["avg_amount"])
    amount_ratio = amount / (customer_avg if customer_avg > 0.0 else 1.0)
    requested_epoch, requested_hour, requested_weekday = parse_timestamp(transaction["requested_at"])

    if last_transaction is None:
        last_delta_days = -1.0
        last_km = -1.0
    else:
        last_epoch, _, _ = parse_timestamp(last_transaction["timestamp"])
        last_delta_days = clamp_upper1(max(0, requested_epoch - last_epoch) / 86_400.0)
        last_km = clamp_upper1(float(last_transaction["km_from_current"]) / 1_000.0)

    merchant_known = merchant["id"] in customer["known_merchants"]

    return [
        clamp_upper1(amount / 10_000.0),
        float(transaction["installments"]) / 12.0,
        clamp_upper1(amount_ratio / 10.0),
        requested_hour / 23.0,
        requested_weekday / 6.0,
        last_delta_days,
        last_km,
        clamp_upper1(float(terminal["km_from_home"]) / 1_000.0),
        float(customer["tx_count_24h"]) / 20.0,
        1.0 if terminal["is_online"] else 0.0,
        1.0 if terminal["card_present"] else 0.0,
        0.0 if merchant_known else 1.0,
        MCC_RISK.get(merchant["mcc"], 0.50),
        clamp_upper1(float(merchant["avg_amount"]) / 10_000.0),
    ]


def load_training_data(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    entries = payload["entries"]
    features = np.asarray([request_features(entry["request"]) for entry in entries], dtype=np.float32)
    labels = np.asarray([not entry["expected_approved"] for entry in entries], dtype=np.uint8)
    return features, labels, hashlib.sha256(raw).hexdigest()


def gini(positives: int, samples: int) -> float:
    if samples == 0:
        return 0.0
    p = positives / samples
    return 2.0 * p * (1.0 - p)


def build_tree(features: np.ndarray, labels: np.ndarray, indices: np.ndarray, depth: int) -> TreeNode:
    sample_count = int(indices.size)
    positive_count = int(labels[indices].sum())
    fraud = positive_count * 2 >= sample_count

    if positive_count == 0 or positive_count == sample_count or depth >= MAX_DEPTH:
        return TreeNode(-1, np.float32(0.0), None, None, fraud, sample_count, positive_count)

    base_impurity = gini(positive_count, sample_count)
    best_feature = -1
    best_threshold = np.float32(0.0)
    best_gain = MIN_GAIN

    for feature_index in range(features.shape[1]):
        values = features[indices, feature_index]
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_labels = labels[indices[order]].astype(np.int32)
        split_possible = sorted_values[:-1] < sorted_values[1:]
        if not bool(split_possible.any()):
            continue

        left_positives = np.cumsum(sorted_labels[:-1])
        left_count = np.arange(1, sample_count, dtype=np.int32)
        right_count = sample_count - left_count
        right_positives = positive_count - left_positives

        left_rate = left_positives / left_count
        right_rate = right_positives / right_count
        impurity = (left_count / sample_count) * (2.0 * left_rate * (1.0 - left_rate))
        impurity += (right_count / sample_count) * (2.0 * right_rate * (1.0 - right_rate))
        gains = base_impurity - impurity
        gains[~split_possible] = -1.0

        split_index = int(np.argmax(gains))
        gain = float(gains[split_index])
        if gain > best_gain:
            best_gain = gain
            best_feature = feature_index
            best_threshold = np.float32(
                (float(sorted_values[split_index]) + float(sorted_values[split_index + 1])) / 2.0
            )

    if best_feature < 0:
        return TreeNode(-1, np.float32(0.0), None, None, fraud, sample_count, positive_count)

    left_mask = features[indices, best_feature] <= best_threshold
    left_indices = indices[left_mask]
    right_indices = indices[~left_mask]
    if left_indices.size == 0 or right_indices.size == 0:
        return TreeNode(-1, np.float32(0.0), None, None, fraud, sample_count, positive_count)

    left = build_tree(features, labels, left_indices, depth + 1)
    right = build_tree(features, labels, right_indices, depth + 1)
    return TreeNode(best_feature, best_threshold, left, right, fraud, sample_count, positive_count)


def predict(node: TreeNode, row: np.ndarray) -> bool:
    while node.feature >= 0:
        node = node.left if row[node.feature] <= node.threshold else node.right
        assert node is not None
    return node.fraud


def flatten_tree(root: TreeNode) -> list[tuple[int, np.float32, int, int, bool]]:
    nodes: list[tuple[int, np.float32, int, int, bool] | None] = []

    def visit(node: TreeNode) -> int:
        index = len(nodes)
        nodes.append(None)
        if node.feature < 0:
            nodes[index] = (-1, np.float32(0.0), 0, 0, node.fraud)
            return index

        assert node.left is not None and node.right is not None
        left = visit(node.left)
        right = visit(node.right)
        nodes[index] = (node.feature, node.threshold, left, right, node.fraud)
        return index

    visit(root)
    return [node for node in nodes if node is not None]


def tree_depth(node: TreeNode) -> int:
    if node.feature < 0:
        return 0
    assert node.left is not None and node.right is not None
    return 1 + max(tree_depth(node.left), tree_depth(node.right))


def format_f32(value: np.float32) -> str:
    text = np.format_float_positional(value, unique=True, trim="-")
    if text in {"0", "-0"}:
        text = "0.0"
    elif "." not in text and "e" not in text:
        text += ".0"
    return f"{text}_f32"


def render_rust(nodes: list[tuple[int, np.float32, int, int, bool]], checksum: str, rows: int, depth: int) -> str:
    node_count = len(nodes)
    if node_count >= 2048:
        raise ValueError(f"packed node format supports fewer than 2048 nodes; got {node_count}")

    lines: list[str] = [
        "// Generated by scripts/train_tree.py from local labelled data.",
        f"// training_sha256: {checksum}",
        f"// rows: {rows}, features: {len(FEATURE_NAMES)}, nodes: {node_count}, max_depth: {depth}",
        "#![allow(clippy::excessive_precision, clippy::manual_range_patterns)]",
        "use std::mem::MaybeUninit;",
        "",
        f"pub const FEATURE_COUNT: usize = {len(FEATURE_NAMES)};",
        "const LEAF_FEATURE: u8 = 255;",
        "const LAST_FEATURE_BITS: u32 = (1 << 5) | (1 << 6);",
        "const TERMINAL_FEATURE_BITS: u32 = (1 << 9) | (1 << 10);",
        "",
        "#[derive(Clone, Copy)]",
        "struct Node {",
        "    threshold: f32,",
        "    left: u16,",
        "    right: u16,",
        "    feature: u8,",
        "    fraud: bool,",
        "}",
        "",
        "#[derive(Clone, Copy)]",
        "struct PackedNode {",
        "    threshold: f32,",
        "    packed: u32,",
        "}",
        "",
        "impl PackedNode {",
        "    const fn new(node: Node) -> Self {",
        "        assert!(node.left < 2_048);",
        "        assert!(node.right < 2_048);",
        "        let fraud_bit = if node.fraud { 1u32 } else { 0u32 };",
        "        Self {",
        "            threshold: node.threshold,",
        "            packed: node.left as u32",
        "                | ((node.right as u32) << 11)",
        "                | ((node.feature as u32) << 22)",
        "                | (fraud_bit << 30),",
        "        }",
        "    }",
        "",
        "    #[inline(always)]",
        "    fn left(self) -> usize {",
        "        (self.packed & 0x7ff) as usize",
        "    }",
        "",
        "    #[inline(always)]",
        "    fn right(self) -> usize {",
        "        ((self.packed >> 11) & 0x7ff) as usize",
        "    }",
        "",
        "    #[inline(always)]",
        "    fn feature(self) -> u8 {",
        "        ((self.packed >> 22) & 0xff) as u8",
        "    }",
        "",
        "    #[inline(always)]",
        "    fn fraud(self) -> bool {",
        "        self.packed & (1 << 30) != 0",
        "    }",
        "}",
        "",
        f"const fn pack_nodes(nodes: [Node; {node_count}]) -> [PackedNode; {node_count}] {{",
        "    let mut packed = [PackedNode {",
        "        threshold: 0.0,",
        "        packed: 0,",
        f"    }}; {node_count}];",
        "    let mut index = 0;",
        f"    while index < {node_count} {{",
        "        packed[index] = PackedNode::new(nodes[index]);",
        "        index += 1;",
        "    }",
        "    packed",
        "}",
        "",
        "#[inline(always)]",
        "pub fn predict_with_lazy_features<F>(",
        "    features: &mut [MaybeUninit<f32>; FEATURE_COUNT],",
        "    mut fill_feature: F,",
        ") -> Option<bool>",
        "where",
        "    F: FnMut(u8, &mut [MaybeUninit<f32>; FEATURE_COUNT]) -> Option<()>,",
        "{",
        "    let mut index = 0usize;",
        "    let mut ready_bits = 0u32;",
        "    loop {",
        "        let node = unsafe { NODES.get_unchecked(index) };",
        "        let feature_index = node.feature();",
        "        if feature_index == LEAF_FEATURE {",
        "            return Some(node.fraud());",
        "        }",
        "        match feature_index {",
        "            5 | 6 => {",
        "                if ready_bits & LAST_FEATURE_BITS == 0 {",
        "                    fill_feature(feature_index, features)?;",
        "                    ready_bits |= LAST_FEATURE_BITS;",
        "                }",
        "            }",
        "            9 | 10 => {",
        "                if ready_bits & TERMINAL_FEATURE_BITS == 0 {",
        "                    fill_feature(feature_index, features)?;",
        "                    ready_bits |= TERMINAL_FEATURE_BITS;",
        "                }",
        "            }",
        "            11 | 12 | 13 => {",
        "                let feature_bit = 1u32 << feature_index;",
        "                if ready_bits & feature_bit == 0 {",
        "                    fill_feature(feature_index, features)?;",
        "                    ready_bits |= feature_bit;",
        "                }",
        "            }",
        "            _ => {}",
        "        }",
        "        let feature = unsafe { features.get_unchecked(feature_index as usize).assume_init() };",
        "        index = if feature <= node.threshold {",
        "            node.left()",
        "        } else {",
        "            node.right()",
        "        };",
        "    }",
        "}",
        "",
        f"const NODES: [PackedNode; {node_count}] = pack_nodes(NODE_SPECS);",
        "",
        f"const NODE_SPECS: [Node; {node_count}] = [",
    ]

    for feature, threshold, left, right, fraud in nodes:
        feature_value = "LEAF_FEATURE" if feature < 0 else str(feature)
        fraud_value = "true" if fraud else "false"
        lines.extend(
            [
                "    Node {",
                f"        feature: {feature_value},",
                f"        threshold: {format_f32(threshold)},",
                f"        left: {left},",
                f"        right: {right},",
                f"        fraud: {fraud_value},",
                "    },",
            ]
        )

    lines.append("];")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    features, labels, checksum = load_training_data(args.input)
    root = build_tree(features, labels, np.arange(labels.size), 0)
    predictions = np.asarray([predict(root, row) for row in features], dtype=np.uint8)
    false_positives = int(((predictions == 1) & (labels == 0)).sum())
    false_negatives = int(((predictions == 0) & (labels == 1)).sum())

    nodes = flatten_tree(root)
    args.output.write_text(
        render_rust(nodes, checksum, int(labels.size), tree_depth(root)),
        encoding="utf-8",
    )

    print(
        f"wrote {args.output}: rows={labels.size} nodes={len(nodes)} "
        f"fp={false_positives} fn={false_negatives}",
        file=sys.stderr,
    )
    return 0 if false_positives == 0 and false_negatives == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
