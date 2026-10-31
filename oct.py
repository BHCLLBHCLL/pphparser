#!/usr/bin/env python3
"""OCT 八叉树文件解析（scFLOW ``*.oct``，CRDL-FLD 容器）。

节布局：

* ``Application`` / ``Dimension`` / ``Date`` / ``UnitOfCoordinates`` — 常规元数据。
* ``LS_CoordinateSystem`` — 坐标系 id（描述符链末值，观测为 0）。
* ``LS_OctLastGenYear`` — 最近一次八叉树生成的年份（0 = 未生成/未知）。
* ``LS_OctRootOctantMinMax`` — 根节点包围盒 ``R8[6]``
  ``(xmin, ymin, zmin, xmax, ymax, zmax)``。
* ``LS_OctOctantRefinement`` — ``U1[n_octants]``，**深度优先前序遍历**的
  八叉树结构位图：``1`` = 内部节点（被细分，其后紧跟 8 个子节点记录），
  ``0`` = 叶子。树是完整八叉树，因此 ``n = 1 + 8 * 内部节点数``。
* ``LS_OctOctantBlockID`` — ``I4[n_octants]``，与位图同序的块 id
  （本样例全为 -1，表示未使用）。

八叉树几何重建：从根包围盒出发，按前序位图递归二分；子节点顺序按
Morton/Z 序约定（bit0=x, bit1=y, bit2=z；低位为 min 半区）。

快照 ``ZIPOCTREE`` 中的附属数组（见 ``sctsnapshot`` / PPH_FORMAT_SPEC §6.3.2）：

* ``OCTREEDIVISION`` — 与本文件 refinement **同一棵树**，但序列化子序为
  ``(1,3,2,0,5,7,6,4)``（非本文件的 ``0..7``），且为 LSB 位打包。
* ``OCTREEREGION`` — **后序** 每节点 1 字节标志（非本文件前序下标）。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from crdlfld import (CrdlFldFile, MAGIC, Section, find_section,
                     iter_data_blocks, iter_descriptors)


@dataclass
class OctModel:
    """解析后的 OCT 八叉树。"""

    root_min: np.ndarray          # (3,) 根包围盒最小角
    root_max: np.ndarray          # (3,) 根包围盒最大角
    n_octants: int                # 位图长度（= 1 + 8*内部节点数）
    n_internal: int               # 内部（被细分）节点数
    n_leaves: int                 # 叶子节点数
    refinement: np.ndarray        # U1[n_octants] 前序位图
    block_id: np.ndarray          # I4[n_octants] 块 id（可能全 -1）
    unit: str = ""
    last_gen_year: int = 0

    def iter_leaves(self, max_leaves: Optional[int] = None
                    ) -> Iterator[tuple[tuple[float, float, float],
                                        tuple[float, float, float], int]]:
        """迭代叶子八分区，产出 ``(min_corner, max_corner, depth)``。

        子节点按 Z 序展开（bit0=x, bit1=y, bit2=z；bit=0 取低半区）。
        注意：单次遍历与 ``refinement`` 同长度，复杂度 O(n_octants)。
        """
        ref = self.refinement
        count = 0
        # 显式栈迭代，避免大深度递归
        stack: list[tuple[float, float, float, float, float, float, int, int]] = []
        x0, y0, z0 = (float(v) for v in self.root_min)
        x1, y1, z1 = (float(v) for v in self.root_max)
        pos = 0
        stack.append((x0, y0, z0, x1, y1, z1, 0, -1))
        while stack:
            ax, ay, az, bx, by, bz, depth, _ = stack.pop()
            r = int(ref[pos])
            pos += 1
            if r == 0:
                yield (ax, ay, az), (bx, by, bz), depth
                count += 1
                if max_leaves is not None and count >= max_leaves:
                    return
                continue
            cx, cy, cz = (ax + bx) / 2.0, (ay + by) / 2.0, (az + bz) / 2.0
            # 子节点 0..7 依次入栈（栈反向，故先压 7）
            for i in range(7, -1, -1):
                nax = cx if i & 1 else ax
                nay = cy if i & 2 else ay
                naz = cz if i & 4 else az
                nbx = bx if i & 1 else cx
                nby = by if i & 2 else cy
                nbz = bz if i & 4 else cz
                stack.append((nax, nay, naz, nbx, nby, nbz, depth + 1, i))

    def leaf_stats(self) -> dict:
        """叶子深度直方图与尺寸统计。"""
        from collections import Counter

        depths: Counter[int] = Counter()
        n = 0
        for _, _, d in self.iter_leaves():
            depths[d] += 1
            n += 1
        return {"n_leaves": n, "depth_histogram": dict(sorted(depths.items()))}

    def block_partition(self) -> dict:
        """LS_OctOctantBlockID 分区语义（O1）。

        box/laptop 样本 block_id 全 -1（未分区）；并行/多块网格下非负值为
        分区（block）id，用于域分解。返回
        {n_octants, partitioned, n_blocks, block_ids, block_id}。
        """
        bids = self.block_id
        uniq = (np.unique(bids[bids >= 0]) if bids.size
                else np.array([], dtype=np.int64))
        return {
            "n_octants": int(bids.size),
            "partitioned": bool(uniq.size > 0),
            "n_blocks": int(uniq.size),
            "block_ids": [int(v) for v in uniq],
            "block_id": bids,
        }


def parse_oct(filepath: str) -> OctModel:
    """解析 OCT 文件，返回 :class:`OctModel`。"""
    with CrdlFldFile.load(filepath) as f:
        data = f.data

        # ── 根包围盒 ─────────────────────────────────────────────────
        root_min = np.zeros(3)
        root_max = np.zeros(3)
        sec = f.get_section("LS_OctRootOctantMinMax")
        if sec:
            for b in iter_data_blocks(data, sec):
                if b.byte_count == 48:
                    vals = b.as_f8(data)
                    root_min = vals[:3]
                    root_max = vals[3:]
                    break

        # ── 位图 ─────────────────────────────────────────────────────
        refinement = np.empty(0, dtype=np.uint8)
        n_octants = 0
        sec = f.get_section("LS_OctOctantRefinement")
        if sec:
            counts = [d.dim0 for d in iter_descriptors(data, sec) if d.dim0 > 1]
            if counts:
                n_octants = counts[0]
            for b in iter_data_blocks(data, sec):
                refinement = b.as_u1(data)
                break
            if n_octants and len(refinement) != n_octants:
                refinement = refinement[:n_octants]
        n_octants = len(refinement)
        n_internal = int(np.count_nonzero(refinement)) if n_octants else 0
        n_leaves = n_octants - n_internal
        if n_octants and (n_octants - 1) % 8 != 0:
            raise ValueError(
                f"{filepath}: 八叉树节点数 {n_octants} 不满足 n = 1 + 8k")
        if n_octants and (n_octants - 1) // 8 != n_internal:
            raise ValueError(
                f"{filepath}: 内部节点数 {n_internal} 与位图长度不一致"
                f"（期望 {(n_octants - 1) // 8}）")

        # ── 块 id ────────────────────────────────────────────────────
        block_id = np.empty(0, dtype=np.int64)
        sec = f.get_section("LS_OctOctantBlockID")
        if sec:
            for b in iter_data_blocks(data, sec):
                if b.byte_count % 4 == 0:
                    block_id = b.as_i4(data)
                    break

        # ── 单位与生成年份 ───────────────────────────────────────────
        # 不能直接用 sec.end：scan_sections 的通用扫描会把 UnitOfCoordinates
        # 体内 unit32 数据块的「[I4=32] + 32 字节 ASCII」误判为节头
        # （两者字节层面同构，见 crdlfld._valid_section_start），把本节截断到
        # 首个记录（只剩那个 f8=1.0 数值块）→ unit 恒为空。
        # 修法：用**下一个具名节**的起点重算 end，再走正规 iter_data_blocks。
        unit = ""
        sec = f.get_section("UnitOfCoordinates")
        if sec:
            end = len(data)
            for nm in ("HeaderDataEnd", "OverlapStart_0", "LS_CoordinateSystem"):
                off = find_section(data, nm)
                if sec.records_start < off < end:
                    end = off
            sub = Section("UnitOfCoordinates", sec.start, end)
            for b in iter_data_blocks(data, sub):
                raw = bytes(data[b.offset : b.offset + b.byte_count])
                if all(x == 0 or 32 <= x < 127 for x in raw):
                    s = raw.decode("ascii", errors="replace").strip("\x00").rstrip()
                    if s:
                        unit = s
                        break
        last_gen_year = 0
        sec = f.get_section("LS_OctLastGenYear")
        if sec:
            vals = [d.dim0 for d in iter_descriptors(data, sec)]
            if vals:
                last_gen_year = vals[-1]

        return OctModel(
            root_min=root_min, root_max=root_max,
            n_octants=n_octants, n_internal=n_internal, n_leaves=n_leaves,
            refinement=refinement, block_id=block_id,
            unit=unit, last_gen_year=last_gen_year,
        )


def oct_leaf_table(model: OctModel, flags=None) -> list[tuple]:
    """叶子表：(preorder_index, min_corner, max_corner, depth, region_flag)。

    与 OctModel.iter_leaves 同序，但额外产出前序下标与可选区域标志
    （flags 为与 refinement 同下标的对齐数组，见 oct_region_map）。
    """
    ref = model.refinement
    rows: list[tuple] = []
    x0, y0, z0 = (float(v) for v in model.root_min)
    x1, y1, z1 = (float(v) for v in model.root_max)
    stack = [(x0, y0, z0, x1, y1, z1, 0)]
    pos = 0
    while stack:
        ax, ay, az, bx, by, bz, d = stack.pop()
        r = int(ref[pos])
        idx = pos
        pos += 1
        if r == 0:
            flag = int(flags[idx]) if flags is not None else -1
            rows.append((idx, (ax, ay, az), (bx, by, bz), d, flag))
            continue
        cx, cy, cz = (ax + bx) / 2.0, (ay + by) / 2.0, (az + bz) / 2.0
        for i in range(7, -1, -1):
            nax = cx if i & 1 else ax
            nay = cy if i & 2 else ay
            naz = cz if i & 4 else az
            nbx = bx if i & 1 else cx
            nby = by if i & 2 else cy
            nbz = bz if i & 4 else cz
            stack.append((nax, nay, naz, nbx, nby, nbz, d + 1))
    return rows


def oct_region_map(snapshot, model: OctModel) -> dict:
    """oct ↔ 快照区域对齐（O2）：OCTREEREGION 后序字节 → .oct 前序下标。

    snapshot 需提供 octree_region_as_oct_order(refinement)（SctSnapshot）。
    返回 {flags, n_leaves, n_active, active_bbox, leaves}；active_bbox 为
    flag=1 叶子的并集包围盒 (lo, hi)。
    """
    flags = None
    if snapshot is not None:
        flags = snapshot.octree_region_as_oct_order(model.refinement)
    leaves = oct_leaf_table(model, flags)
    active = [r for r in leaves if r[4] == 1]
    if active:
        mins = np.array([r[1] for r in active], dtype=float)
        maxs = np.array([r[2] for r in active], dtype=float)
        bbox = (mins.min(axis=0), maxs.max(axis=0))
    else:
        bbox = (None, None)
    return {
        "flags": flags,
        "n_leaves": len(leaves),
        "n_active": len(active),
        "active_bbox": bbox,
        "leaves": leaves,
    }


def oct_cell_mask(model: OctModel, snapshot, centroids) -> np.ndarray:
    """octant → 单元几何链路（O3）：活跃八叉区域内单元质心掩码。

    centroids：(n_cells, 3)。活跃区域 = OCTREEREGION flag=1 叶子并集包围盒；
    返回 bool[n_cells]（质心落在活跃区域内的单元）。区域索引/cvol 语义见快照
    OCTREERESTRRGN / GPH LS_CvolIdOfElements；此处给出几何侧链路。
    """
    rmap = oct_region_map(snapshot, model)
    bbox = rmap["active_bbox"]
    if bbox[0] is None:
        return np.zeros(len(centroids), dtype=bool)
    lo, hi = bbox
    c = np.asarray(centroids, dtype=float).reshape(-1, 3)
    return np.all((c >= lo) & (c <= hi), axis=1)


def _i32(value: int) -> bytes:
    return struct.pack(">i", int(value))


def _descriptor(type_code: int, dim0: int, dim1: int) -> bytes:
    return _i32(12) + _i32(type_code) + _i32(dim0) + _i32(dim1)


def _block(payload: bytes) -> bytes:
    return _i32(12) + _i32(len(payload)) + payload + _i32(len(payload))


#: 非空节尾部的 20 字节「节结束哨兵」（与 gphstats / 宿主实测一致）
_SECTION_SENTINEL = _i32(12) + _i32(0) + _i32(0) + _i32(0) + _i32(12)


def _f64_bytes(value: float) -> bytes:
    """大端 f64 载荷（UnitOfCoordinates 的比例字段）。"""
    return struct.pack(">d", float(value))


def _section(name: str, body: bytes) -> bytes:
    """命名节头（40 字节）：[I4=32][name 32B][I4=32] + 记录流。

    宿主实测（tests/box/meshinggroup1.oct）：name 后有尾随 [I4=32]，
    空节无 20 字节哨兵。此前缺尾随 32（P2-3 修正）。
    """
    out = _i32(32) + name.ljust(32).encode("ascii") + _i32(32) + body
    # 节体自带 20 字节节尾时不再追加哨兵（与 mdl._section 同规则）
    if body and not body.endswith(_SECTION_SENTINEL):
        out += _SECTION_SENTINEL
    return out


def write_oct(filepath: str | Path,
              root_min,
              root_max,
              refinement=None,
              block_id=None,
              app: str = "SCTpre",
              date: int = 20260812,
              unit: str = "m") -> Path:
    """写最小 CRDL-FLD OCT 文件（无宿主兜底，可被 :func:`parse_oct` 读回）。

    ``refinement`` 为前序位图：0=叶子，1=内部（后随 8 个子节点记录）。
    默认单根叶子八叉树。
    """
    root_min = np.asarray(root_min, dtype=float).reshape(3)
    root_max = np.asarray(root_max, dtype=float).reshape(3)
    if refinement is None:
        refinement = np.zeros(1, dtype=np.uint8)
    ref = np.asarray(refinement, dtype=np.uint8).reshape(-1)
    n = len(ref)
    if block_id is None:
        block_id = np.full(n, -1, dtype=np.int32)
    bid = np.asarray(block_id, dtype=np.int32).reshape(-1)
    if len(bid) != n:
        raise ValueError("block_id length must equal refinement length")

    app_block = app.encode("ascii")[:8].ljust(8)
    unit8 = unit.encode("ascii")[:8].ljust(8)
    unit32 = unit.encode("ascii")[:32].ljust(32)

    out = bytearray()
    out += _i32(8) + MAGIC + _i32(8) + _i32(4) + _i32(4) + _i32(4)  # 容器头第 4 个 I4（PPH_FORMAT_SPEC §2：实测 4,4,4）
    # 宿主 OCT 实测：Application = desc(1,8,1) + block(8)（注意与 MDL 的
    # desc(1,1,8) 维序相反）
    out += _section("Application", _descriptor(1, 8, 1) + _block(app_block))
    out += _section("Dimension",
                    _descriptor(4, 1, 1) + _descriptor(4, 3, 4))
    out += _section("Date",
                    _descriptor(4, 1, 1) + _descriptor(4, date, 4))
    # 宿主 OCT 实测：比例 f64(1.0) + 长名 + 短名（本机两种都写单位短名）
    out += _section(
        "UnitOfCoordinates",
        _descriptor(8, 1, 1) + _block(_f64_bytes(1.0)) +
        _descriptor(1, 32, 1) + _block(unit32) +
        _descriptor(1, 32, 1) + _block(unit32))
    out += _section("HeaderDataEnd", b"")
    out += _section("OverlapStart_0", b"")
    coord = (_descriptor(4, 1, 1) + _descriptor(4, 1, 4) +
             _descriptor(4, 1, 1) + _descriptor(4, 0, 4))
    out += _section("LS_CoordinateSystem", coord)
    out += _section("LS_OctLastGenYear", coord)
    root_box = np.concatenate([root_min, root_max]).astype(">f8").tobytes()
    out += _section(
        "LS_OctRootOctantMinMax",
        _descriptor(4, 1, 1) + _descriptor(4, 1, 4) +
        _descriptor(8, 6, 1) + _block(root_box))
    out += _section(
        "LS_OctOctantRefinement",
        _descriptor(4, 1, 1) + _descriptor(4, 1, 4) +
        _descriptor(4, 1, 1) + _descriptor(4, n, 4) +
        _descriptor(1, n, 1) +            # 宿主实测：type=1（非 4）
        _block(ref.tobytes()))
    out += _section(
        "LS_OctOctantBlockID",
        _descriptor(4, 1, 1) + _descriptor(4, 1, 4) +
        _descriptor(4, 1, 1) + _descriptor(4, n, 4) +
        _descriptor(4, n, 1) + _block(bid.astype(">i4").tobytes()))
    out += _section("OverlapEnd", b"")

    path = Path(filepath)
    path.write_bytes(bytes(out))
    return path


def _build_tree(refinement) -> list:
    """把前序位图转成嵌套树 ``[bit, [children...]]``。"""
    ref = list(refinement)
    idx = 0

    def build() -> list:
        nonlocal idx
        r = int(ref[idx])
        idx += 1
        children = [build() for _ in range(8)] if r else []
        return [r, children]

    return build()


def _tree_to_ref(node, out: list[int]) -> None:
    out.append(node[0])
    for child in node[1]:
        _tree_to_ref(child, out)


def _model_from_tree(node, model: OctModel) -> OctModel:
    out: list[int] = []
    _tree_to_ref(node, out)
    ref = np.asarray(out, dtype=np.uint8)
    n = len(ref)
    bid = np.full(n, -1, dtype=np.int32)
    if model.block_id.size == n:
        bid = model.block_id.astype(np.int32)
    return OctModel(
        root_min=model.root_min, root_max=model.root_max,
        n_octants=n, n_internal=int(np.count_nonzero(ref)),
        n_leaves=n - int(np.count_nonzero(ref)),
        refinement=ref, block_id=bid,
        unit=model.unit, last_gen_year=model.last_gen_year)


def refine_all_leaves(model: OctModel) -> OctModel:
    """把所有叶子细分成一层（每个叶子变为内部节点 + 8 个叶子）。"""

    def transform(node: list) -> list:
        if node[0] == 0:
            return [1, [[0, []] for _ in range(8)]]
        return [1, [transform(c) for c in node[1]]]

    return _model_from_tree(transform(_build_tree(model.refinement)), model)


def coarsen_all_leaves(model: OctModel) -> OctModel:
    """把所有“8 个孩子全是叶子”的节点合并为叶子（单层粗化）。"""

    def transform(node: list) -> list:
        if node[0] == 0:
            return [0, []]
        children = [transform(c) for c in node[1]]
        if all(c[0] == 0 for c in children):
            return [0, []]
        return [1, children]

    return _model_from_tree(transform(_build_tree(model.refinement)), model)
