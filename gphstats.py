#!/usr/bin/env python3
"""PPH 内 ``*.gph`` 体网格成员的读写。

只依赖本仓库的 ``crdlfld``（节扫描 / 数据块迭代）：

- LS_Links：面数、单元数、边界面数、npe 范围、多面体标记；
- LS_CvolIdOfElements：单元数 + 唯一 cvol 列表；
- LS_Nodes：顶点数 + 坐标方言（BE f64 / word-reversed f64 / BE f32）；
- LS_SurfaceRegions：面区域名 + 面数；
- LS_VolumeRegions / 其他字符串节：字符串列表；
- LS_Parts：部件名 + cvol 规格（简单单值或复合 frozenset）。

读端按"最常见 I4 块尺寸 → 面数"等启发式统计；写端 ``write_gph`` /
``write_gph_volume`` 按宿主布局重序列化。box 黄金样例：n_faces=3168 /
n_cells=944 / boundary=600 / npe 4..6 / 顶点 1305 / open=600。
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np

import crdlfld


# 同一 buffer 上反复 _find_section 时复用节表（laptop GPH 上 scan≈2s）
_sections_cache: dict[int, list] = {}


@contextmanager
def open_buffer(path: str):
    """读取 GPH 文件；超过阈值时用 mmap。"""
    from pathlib import Path
    import mmap

    size = Path(path).stat().st_size
    # >32 MiB 用 mmap，避免把数百 MB 再拷进进程堆（打开工程时会卡死）
    if size <= 32 * 1024 * 1024:
        with open(path, "rb") as f:
            data = f.read()
        try:
            yield data
        finally:
            _sections_cache.pop(id(data), None)
        return

    f = open(path, "rb")
    try:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            yield mm
        finally:
            _sections_cache.pop(id(mm), None)
            mm.close()
    finally:
        f.close()


def _buffer_fingerprint(data) -> tuple:
    """缓冲区内容指纹（防 id 复用脏缓存）。

    ``_sections_cache`` 以 ``id(data)`` 为键；调用方直接传 ``bytes``
    （公开 API ``gph_cells``/``parse_mesh``）时没有 ``open_buffer``
    的 finally 清理，buffer 被 GC 后新 buffer 复用同一 id 会命中
    脏节表（实测：同进程先后解析两个 GPH，第二个返回 0 cells）。
    ``scan_sections`` 是内容的纯函数——指纹不同必重扫；指纹相同
    （同内容不同对象）时节表本就相同，命中无害。
    """
    return (len(data), bytes(data[:64]), bytes(data[-64:]))


def _all_sections(data) -> list:
    key = id(data)
    fp = _buffer_fingerprint(data)
    entry = _sections_cache.get(key)
    if entry is not None and entry[1] == fp:
        return entry[0]
    secs = crdlfld.scan_sections(data)
    _sections_cache[key] = (secs, fp)
    return secs


def _find_section(data, name: str) -> Optional[crdlfld.Section]:
    """按名定位节，返回 ``Section``（节尾 = 下一节头或文件尾）。"""
    sec_start = crdlfld.find_section(data, name)
    if sec_start < 0:
        return None
    sections = _all_sections(data)
    for i, s in enumerate(sections):
        if s.start == sec_start:
            end = sections[i + 1].start if i + 1 < len(sections) else len(data)
            return crdlfld.Section(name, sec_start, end)
    return crdlfld.Section(name, sec_start, len(data))


def links_summary(data) -> Optional[dict]:
    """LS_Links 拓扑摘要：n_faces / n_cells / boundary / npe / polyhedral。"""
    section = _find_section(data, "LS_Links")
    if section is None:
        return None
    blocks = [(b.offset, b.byte_count)
              for b in crdlfld.iter_data_blocks(data, section) if b.byte_count > 0]
    if not blocks:
        return None
    sizes = [bc for _, bc in blocks]
    n_faces_block_size = None
    for size, count in Counter(sizes).most_common():
        if count >= 3 and size % 4 == 0 and size >= 4:
            n_faces_block_size = size
            break
    if n_faces_block_size is None:
        return None
    n_faces = n_faces_block_size // 4
    triples = [b for b in blocks if b[1] == n_faces_block_size][:3]
    if len(triples) < 3:
        return None
    owner_p, _ = triples[0]
    neigh_p, _ = triples[1]
    npe_p, _ = triples[2]
    npe = np.frombuffer(data, dtype=">u4", count=n_faces, offset=npe_p)
    neigh = np.frombuffer(data, dtype=">u4", count=n_faces, offset=neigh_p)
    owner = np.frombuffer(data, dtype=">u4", count=n_faces, offset=owner_p)
    return {
        "n_faces": int(n_faces),
        "n_cells": int(owner.max()) + 1,
        "boundary_faces": int((neigh == 0xFFFFFFFF).sum()),
        "npe_min": int(npe.min()),
        "npe_max": int(npe.max()),
        "conn_entries": int(npe.sum()),
        "polyhedral": bool(npe.max() > 3),
    }


def cvol_ids(data) -> Optional[np.ndarray]:
    """LS_CvolIdOfElements → I4[n_cells]（节内最大 I4 块）。"""
    section = _find_section(data, "LS_CvolIdOfElements")
    if section is None:
        return None
    best: Optional[tuple[int, int]] = None
    for b in crdlfld.iter_data_blocks(data, section):
        p, bc = b.offset, b.byte_count
        if bc % 4 == 0 and bc >= 4:
            if best is None or bc > best[1]:
                best = (p, bc)
    if best is None:
        return None
    p, bc = best
    return np.frombuffer(data, dtype=">i4", count=bc // 4, offset=p).astype(
        np.int64).copy()


def element_info(data) -> Optional[tuple[int, np.ndarray]]:
    """Element_InformationFlag -> (n_flag_types, flags[n_cells] I4).

    scFLOW 体网格每单元一个「元素信息」标志；节内 (4, 31, 4) 描述符给出
    31 种 flag 类型（box/laptop 均恒为 31），随后 I4[n_cells] 块为每单元
    标志值（box 全 9 = 0b1001）。位语义待 FLDUTIL 对拍钉死（fldutil_bridge）。
    """
    section = _find_section(data, "Element_InformationFlag")
    if section is None:
        return None
    descs = [(d.dim0, d.dim1)
             for d in crdlfld.iter_descriptors(data, section)
             if d.type_code == 4 and d.dim0 > 1]
    flags: Optional[np.ndarray] = None
    for b in crdlfld.iter_data_blocks(data, section):
        if b.byte_count >= 4 and b.byte_count % 4 == 0:
            flags = np.frombuffer(data, dtype=">i4", count=b.byte_count // 4,
                                  offset=b.offset).astype(np.int64).copy()
            break
    if flags is None:
        return None
    n = int(flags.size)
    flag_types = 31
    for d0, d1 in descs:
        if d0 == n and d1 == 4:
            continue  # 数组描述符（dim0 == n_cells）
        if 1 < d0 < n:
            flag_types = int(d0)
    return flag_types, flags


def assemblies_xml(data) -> Optional[str]:
    """LS_Assemblies -> 内嵌 UTF-8 XML 字符串（assembly/part 层级）。

    box 样例：\'<root><assembly name="box" expand="T"><part name="Part"/>\
</assembly></root>\'（单 114B 字符串块）。
    """
    section = _find_section(data, "LS_Assemblies")
    if section is None:
        return None
    for b in crdlfld.iter_data_blocks(data, section):
        raw = bytes(data[b.offset:b.offset + b.byte_count])
        if b"xml" in raw.lower() or raw.lstrip().startswith(b"<"):
            return raw.decode("utf-8", errors="replace").strip("\x00").rstrip()
    return None


def _ls_nodes_elem_bytes(data, sec_start: int, sec_end: int) -> Optional[int]:
    """从 LS_Nodes 类型描述符投票得到坐标元素尺寸（4=f32，8=f64）。"""
    counts = {4: 0, 8: 0}
    pos = sec_start + 40
    n = len(data)
    while pos + 16 <= sec_end and pos + 16 <= n:
        if crdlfld.read_i32_be(data, pos) == 12:
            tc = crdlfld.read_i32_be(data, pos + 4)
            if tc in (4, 8):
                dim0 = crdlfld.read_i32_be(data, pos + 8)
                dim1 = crdlfld.read_i32_be(data, pos + 12)
                if dim0 > 1 and 0 < dim1 < 10_000_000:
                    counts[tc] += 1
        pos += 4
    if counts[8] > counts[4]:
        return 8
    if counts[4] > counts[8]:
        return 4
    return None


def nodes_vertex_count(data) -> tuple[Optional[int], str]:
    """LS_Nodes → (n_vertices, dialect)。轻量实现：最常见的等长三块 → 顶点数。"""
    section = _find_section(data, "LS_Nodes")
    if section is None:
        return None, ""
    blocks = [(b.offset, b.byte_count)
              for b in crdlfld.iter_data_blocks(data, section)
              if b.byte_count >= 4 and b.byte_count % 4 == 0]
    if len(blocks) < 3:
        return None, ""
    sizes = [bc for _, bc in blocks]
    target = max(set(sizes), key=sizes.count)
    trio = [(p, bc) for p, bc in blocks if bc == target][:3]
    if len(trio) < 3:
        return None, ""
    elem = _ls_nodes_elem_bytes(data, section.start, section.end)
    if elem == 8 and target % 8 == 0:
        return target // 8, "standard BE float64"
    if elem == 4 and target % 4 == 0:
        return target // 4, "BE float32"
    if target % 8 == 0:
        return target // 8, "BE float64 (or word-reversed)"
    if target % 4 == 0:
        return target // 4, "BE float32"
    return None, ""


def _iter_surface_region_blocks(data):
    """LS_SurfaceRegions → 迭代 ``(name, face_ids_i32_be_offset, n_faces)``。

    2026-08-17 重写为结构直扫：旧实现走 :func:`crdlfld.iter_data_blocks`
    （4 字节步进回退），追加记录的 ``desc(1,255,1)``（type=1 非 4/8）会被
    误吞成伪块、错过其后的名称块。新实现按名称块 ``[12][255] + 255B 可打印
    + 尾部 255`` 模式定位，再沿后随描述符链取面数组，兼容原文件与
    :func:`append_surface_region` 产物。
    """
    section = _find_section(data, "LS_SurfaceRegions")
    if section is None:
        return
    body = data[section.records_start:section.end]
    base = section.records_start
    pos = 0
    n_body = len(body)
    import struct as _struct
    while pos + 275 <= n_body:
        if body[pos:pos + 4] != b"\x00\x00\x00\x0c" \
                or body[pos + 4:pos + 8] != b"\x00\x00\x00\xff":
            pos += 1
            continue
        raw = body[pos + 8:pos + 8 + 255]
        if not all(b == 0 or 32 <= b < 127 for b in raw):
            pos += 1
            continue
        name = raw.decode("ascii", errors="replace").rstrip()
        if not name:
            pos += 1
            continue
        # 名称块后：desc(4,1,1) desc(4,n,4) desc(4,n,1) block(ids 4n)
        p = pos + 8 + 255 + 4
        if p + 56 > n_body:
            break
        d3 = p + 32
        if body[d3:d3 + 4] != b"\x00\x00\x00\x0c":
            pos += 1
            continue
        n = _struct.unpack(">i", body[d3 + 8:d3 + 12])[0]
        if n < 0 or p + 56 + 4 * n > n_body:
            pos += 1
            continue
        yield name, base + p + 56, n
        pos += 1


def surface_regions_summary(data) -> list[tuple[str, int]]:
    """LS_SurfaceRegions → [(name, n_faces)]。"""
    return [(name, n) for name, _off, n in _iter_surface_region_blocks(data)]


def volume_region_names(data) -> list[str]:
    """LS_VolumeRegions → [名称]（该节无面号数组，只有名称记录）。

    与 :func:`surface_region_face_ids` 对称，供写端往返使用。
    """
    section = _find_section(data, "LS_VolumeRegions")
    if section is None:
        return []
    names: list[str] = []
    pos = section.records_start
    end = min(len(data), section.end)
    while pos + 16 <= end:
        ok_head = (crdlfld.read_i32_be(data, pos) == 12
                   and crdlfld.read_i32_be(data, pos + 4) == 1
                   and crdlfld.read_i32_be(data, pos + 8) == 255
                   and crdlfld.read_i32_be(data, pos + 12) == 1)
        if ok_head:
            p = pos + 16
            ok_name = (p + 12 <= end
                       and crdlfld.read_i32_be(data, p) == 12
                       and crdlfld.read_i32_be(data, p + 4) == 255)
            if ok_name:
                raw = bytes(data[p + 8:p + 8 + 255])
                nm = raw.decode("ascii", "replace").strip()
                if nm:
                    names.append(nm)
                pos = p + 8 + 255 + 4
                continue
        pos += 4
    return names


def surface_region_face_ids(data) -> dict[str, np.ndarray]:
    """LS_SurfaceRegions → ``{name: face_index_i32}``（0-based 面号）。"""
    out: dict[str, np.ndarray] = {}
    for name, off, n in _iter_surface_region_blocks(data):
        out[name] = np.frombuffer(
            data, dtype=">i4", count=n, offset=off).astype(np.int64, copy=True)
    return out


def rename_surface_region(data: bytes, old_name: str,
                         new_name: str) -> bytes:
    """LS_SurfaceRegions 名表原地改名（255B 名称块等长替换）。

    2026-08-17 实机验证矩阵（box.pph + 宿主 OpenProject/QueryFaceRegionByName）：

    - **原地改名宿主安全**：GPH 改名后宿主 `OpenProject` 正常（open=0，
      不触发重建，与追加记录不同）；
    - 但宿主 `QueryFaceRegionByName` 仍解析旧名——宿主 face region 注册表
      的权威来源尚未在文件层定位（main.xml regions/SECTITEM、part/ridge
      MDL 名表、GPH 名表、snapshot FACEGROUPSW 逐一改名实测均不改变宿主
      解析结果），本函数保证字节级名表自洽（本仓 parser 可见新名）。

    ``new_name`` 超 255 字节截断；替换在 LS_SurfaceRegions 节内进行。
    """
    section = _find_section(data, "LS_SurfaceRegions")
    if section is None:
        return data
    body = data[section.records_start:section.end]
    needle = old_name.encode("ascii", errors="replace")[:255].ljust(255, b" ")
    replacement = (new_name.encode("ascii", errors="replace")[:255]
                   .ljust(255, b" "))
    if needle not in body:
        return data
    new_body = body.replace(needle, replacement, 1)
    return data[:section.records_start] + new_body + data[section.end:]


def append_surface_region(data: bytes, name: str,
                          face_ids: Optional[np.ndarray] = None) -> bytes:
    """LS_SurfaceRegions 追加一条面区域记录（格式级写端）。

    **宿主警告（2026-08-17 实机验证）**：追加记录（无论是否带面、count
    是否同步）都会使宿主 `OpenProject` 进入**无界重建**（瞬态实例持续
    60–90% CPU，>5 分钟不完成），因此本函数只保证格式自洽，产物**不建议
    直接交给宿主**——宿主 region 注册表的权威写端仍待定位（见
    REANALYSIS §6.2 负面发现）。原地改名请用 :func:`rename_surface_region`。

    记录模板取自已解析节内的任一现有区域记录（保序复制其完整字节），
    ``face_ids`` 省略时复制该模板的面数组；名称等长约束在 255B 块内
    自动满足。
    """
    section = _find_section(data, "LS_SurfaceRegions")
    if section is None:
        return data
    body = data[section.records_start:section.end]
    names = surface_regions_summary(data)
    if not names:
        return data
    # 模板：最后一个现有区域名块开始（desc(1,255,1) 前 24B）到节尾
    last_name = names[-1][0]
    pos = body.rfind(last_name.encode("ascii")[:255])
    if pos < 24:
        return data
    template = body[pos - 24:]
    # 名称 255B 块替换（新名在块内等长）
    old_block = last_name.encode("ascii", errors="replace")[:255].ljust(
        255, b" ")
    new_block = name.encode("ascii", errors="replace")[:255].ljust(255, b" ")
    if old_block not in template:
        return data
    rec = template.replace(old_block, new_block, 1)
    # count 描述符（第 4 个 desc 的 dim0，节体 +56）
    count_off = 56
    import struct
    if count_off + 4 > len(body):
        return data
    count = struct.unpack(">i", body[count_off:count_off + 4])[0]
    new_body = (body[:count_off] + struct.pack(">i", count + 1)
                + body[count_off + 4:] + rec)
    return data[:section.records_start] + new_body + data[section.end:]


def string_list(data, section_name: str) -> list[str]:
    """任意节（如 LS_VolumeRegions）中的 ASCII 字符串列表。"""
    section = _find_section(data, section_name)
    if section is None:
        return []
    out: list[str] = []
    for b in crdlfld.iter_data_blocks(data, section):
        p, bc = b.offset, b.byte_count
        raw = bytes(data[p:p + bc])
        if all(b == 0 or 32 <= b < 127 for b in raw):
            s = raw.decode("ascii", errors="replace").strip("\x00").rstrip()
            if s:
                out.append(s)
    return out


def _ls_parts_name_blocks(data, sec_start: int, sec_end: int):
    """LS_Parts 内 ASCII 名称块（启发式同 gph_model）。"""
    name_blocks: list[tuple[str, int, int]] = []
    for b in crdlfld.iter_data_blocks(data, crdlfld.Section("", sec_start, sec_end)):
        p, bc = b.offset, b.byte_count
        if bc <= 0 or bc > 512:
            continue
        raw = bytes(data[p:p + bc])
        if not all(b == 0 or 32 <= b < 127 for b in raw):
            continue
        name = raw.decode("ascii", errors="replace").strip("\x00").rstrip()
        if not name or not any(c.isalpha() for c in name):
            continue
        name_blocks.append((name, p, bc))
    return name_blocks


def _scan_cvol_descriptor_chain(data, start: int, end: int) -> list[int]:
    """收集 ``[12,4,X,4]`` 描述符链（简单部件 ``[1,cvol_id]`` / 复合 ``[12,4,N,4]``）。"""
    chain: list[int] = []
    pos = start
    while pos + 16 <= end:
        if (crdlfld.read_i32_be(data, pos) == 12
                and crdlfld.read_i32_be(data, pos + 4) == 4
                and crdlfld.read_i32_be(data, pos + 12) == 4):
            chain.append(crdlfld.read_i32_be(data, pos + 8))
        pos += 4
    return chain


def format_part_cvol_spec(spec) -> str:
    if isinstance(spec, frozenset):
        ids = sorted(spec)
        if len(ids) <= 10:
            return "{" + ", ".join(str(i) for i in ids) + "}"
        return f"{{{ids[0]}..{ids[-1]} ... n={len(ids)}}}"
    return str(spec)


def parts_summary(data, cvol: Optional[np.ndarray] = None) -> list[tuple[str, object]]:
    """LS_Parts → [(name, cvol 规格)]，规格为 int 或 frozenset[int]。"""
    section = _find_section(data, "LS_Parts")
    if section is None:
        return []
    names = _ls_parts_name_blocks(data, section.start, section.end)
    actual_set: Optional[set[int]] = None
    if cvol is not None and len(cvol) > 0:
        actual_set = {int(x) for x in np.unique(cvol)}
    out: list[tuple[str, object]] = []
    for i, (name, p, bc) in enumerate(names):
        scan_end = names[i + 1][1] if i + 1 < len(names) else section.end
        after = p + bc + 4  # 名称块 + 尾随 I4
        chain = _scan_cvol_descriptor_chain(data, after, scan_end)
        if chain and chain[0] == 1:
            out.append((name, int(chain[-1])))
            continue
        # 复合部件：[12,4,N,4] + I4[N] 的 cvol 列表
        spec: Optional[frozenset[int]] = None
        if chain:
            chain_counts = set(chain)
            for b in crdlfld.iter_data_blocks(
                    data, crdlfld.Section("", after, scan_end)):
                bp, bbc = b.offset, b.byte_count
                if bbc < 8 or bbc % 4 != 0:
                    continue
                n = bbc // 4
                if n not in chain_counts:
                    continue
                vals = [int(x) for x in np.frombuffer(
                    data, dtype=">i4", count=n, offset=bp)]
                if len(vals) != n or len(set(vals)) != n:
                    continue
                if actual_set is not None and not all(v in actual_set for v in vals):
                    continue
                if n >= 2:
                    spec = frozenset(vals)
                    break
        out.append((name, spec if spec is not None else 1))
    return out


def summarize(data) -> dict:
    """汇总全部轻量统计（供 CLI 使用）。"""
    links = links_summary(data)
    cvol = cvol_ids(data)
    n_vertices, dialect = nodes_vertex_count(data)
    cells = _cells_summary(data)
    return {
        "links": links,
        "cvol_unique": None if cvol is None else np.unique(cvol).tolist(),
        "n_cells": None if cvol is None else int(cvol.size),
        "n_vertices": n_vertices,
        "dialect": dialect,
        "surface_regions": surface_regions_summary(data),
        "volume_regions": string_list(data, "LS_VolumeRegions"),
        "parts": parts_summary(data, cvol),
        "cells": cells,
    }


def _cells_summary(data) -> Optional[dict]:
    """单元重建摘要（单元类型直方图），供 summarize 使用；失败返回 None。"""
    try:
        mesh = parse_mesh(data)
        if not mesh or not mesh.get("n_faces"):
            return None
        cm = build_cells(mesh["owner"], mesh["neigh"], mesh["npe"])
        return {
            "n_cells": cm["n_cells"],
            "type_histogram": cm["type_histogram"],
        }
    except Exception:
        return None


def summarize_quick(data) -> dict:
    """打开工程用的快速摘要：跳过耗时的节点扫描 / 全量 cvol unique。

    laptop 级 GPH 上 ``nodes_vertex_count`` 可达十余秒；Part Tree 只需
    面/单元数与零件名。
    """
    links = links_summary(data)
    return {
        "links": links,
        "cvol_unique": None,
        "n_cells": links.get("n_cells"),
        "n_vertices": 0,
        "dialect": "",
        "surface_regions": surface_regions_summary(data),
        "volume_regions": string_list(data, "LS_VolumeRegions"),
        "parts": parts_summary(data, None),
    }


def summarize_file(path: str) -> dict:
    with open_buffer(path) as data:
        return summarize(data)


# ── 单元重建与分类（把「轻量统计」补齐为完整单元模型）─────────────
CELL_TETRAHEDRON = "tetrahedron"
CELL_PYRAMID = "pyramid"
CELL_PRISM = "prism"          # 三棱柱 / wedge
CELL_HEXAHEDRON = "hexahedron"
CELL_POLYHEDRAL = "polyhedral"


def _cell_type(n_faces: int, tri: int, quad: int) -> str:
    """按面数与三角形/四边形面数分类单元类型。"""
    if n_faces == 4 and tri == 4:
        return CELL_TETRAHEDRON
    if n_faces == 6 and quad == 6:
        return CELL_HEXAHEDRON
    if n_faces == 5 and tri == 2 and quad == 3:
        return CELL_PRISM
    if n_faces == 5 and tri == 4 and quad == 1:
        return CELL_PYRAMID
    return CELL_POLYHEDRAL


def classify_cell(npe_of_faces) -> str:
    """按各面顶点数（npe）分类单个单元。

    四节点面 = 四边形；三节点面 = 三角形；>4 = 多面体面。
    """
    npe_of_faces = np.asarray(npe_of_faces, dtype=np.int64)
    n = int(npe_of_faces.size)
    tri = int((npe_of_faces == 3).sum())
    quad = int((npe_of_faces == 4).sum())
    return _cell_type(n, tri, quad)


def build_cells(owner, neigh, npe, n_cells=None) -> dict:
    """从 LS_Links 面数据重建单元（单元→面邻接 + 单元类型直方图）。

    ``owner``/``neigh``/``npe``：长度 ``n_faces``；``neigh == 0xFFFFFFFF``
    为边界面。单元 ``c`` 的面 = 所有 ``owner == c`` **或** ``neigh == c``
    的面（GPH 中内部面只存一次，由 owner/neigh 双向归属）。

    返回：``n_cells`` / ``cell_face_offsets``(CSR) / ``cell_faces``(平铺面号)
    / ``cell_face_counts`` / ``cell_types``(list[str]) / ``type_histogram``
    (dict[str,int])。全 numpy 向量化，百万级单元亦可承受。
    """
    owner = np.asarray(owner, dtype=np.int64)
    neigh = np.asarray(neigh, dtype=np.int64)
    npe = np.asarray(npe, dtype=np.int64)
    n_faces = int(owner.size)
    empty = {
        "n_cells": 0,
        "cell_face_offsets": np.zeros(1, dtype=np.int64),
        "cell_faces": np.empty(0, dtype=np.int64),
        "cell_face_counts": np.empty(0, dtype=np.int64),
        "cell_types": [],
        "type_histogram": {},
    }
    if n_faces == 0:
        return empty
    internal = neigh != 0xFFFFFFFF
    if n_cells is None:
        hi = int(owner.max())
        if internal.any():
            hi = max(hi, int(neigh[internal].max()))
        n_cells = hi + 1
    if n_cells <= 0:
        return empty
    # 每个内部面贡献两个 (cell, face) 对，边界面贡献一个
    cells = np.concatenate([owner, neigh[internal]])
    faces = np.concatenate([
        np.arange(n_faces, dtype=np.int64),
        np.arange(n_faces, dtype=np.int64)[internal],
    ])
    order = np.argsort(cells, kind="stable")
    cells = cells[order]
    faces = faces[order]
    counts = np.bincount(cells, minlength=n_cells).astype(np.int64)
    offsets = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    npe_cf = npe[faces]
    tri = np.add.reduceat((npe_cf == 3).astype(np.int64), offsets[:-1])
    quad = np.add.reduceat((npe_cf == 4).astype(np.int64), offsets[:-1])
    types = np.empty(n_cells, dtype=object)
    types[:] = CELL_POLYHEDRAL
    types[(counts == 4) & (tri == 4)] = CELL_TETRAHEDRON
    types[(counts == 6) & (quad == 6)] = CELL_HEXAHEDRON
    types[(counts == 5) & (tri == 2) & (quad == 3)] = CELL_PRISM
    types[(counts == 5) & (tri == 4) & (quad == 1)] = CELL_PYRAMID
    uniq, cnt = np.unique(types, return_counts=True)
    return {
        "n_cells": int(n_cells),
        "cell_face_offsets": offsets,
        "cell_faces": faces,
        "cell_face_counts": counts,
        "cell_types": [str(t) for t in types],
        "type_histogram": dict(zip([str(u) for u in uniq],
                                   [int(c) for c in cnt])),
    }


def prism_layers(owner, neigh, npe, n_cells=None) -> dict:
    """边界层 prism 列分析（G3）。

    体网格的棱柱（wedge）单元以「列/栈」形式堆叠成边界层：列内单元经内部面
    相连，列长即棱柱层数。本函数把 prism 单元按内部面连通性聚成列，返回
    column_lengths（每列层数）与 length_histogram（层数直方图）。

    局限：不区分「哪一侧是壁面」，故给出列深（= 层数）而非「距壁面层号」；
    后者需结合 Element_InformationFlag 位 / 面区域（G4）。
    """
    owner = np.asarray(owner, dtype=np.int64)
    neigh = np.asarray(neigh, dtype=np.int64)
    npe = np.asarray(npe, dtype=np.int64)
    cm = build_cells(owner, neigh, npe, n_cells)
    n = cm["n_cells"]
    types = np.array(cm["cell_types"])
    prism_mask = types == CELL_PRISM
    n_prism = int(prism_mask.sum())
    empty = {"n_prism": 0, "n_columns": 0,
             "column_lengths": [], "length_histogram": {}}
    if n_prism == 0:
        return empty

    # 内部面两端均为 prism 的边
    internal = neigh != 0xFFFFFFFF
    a = owner[internal]
    b = neigh[internal]
    keep = prism_mask[a] & prism_mask[b]
    edges = np.column_stack([a[keep], b[keep]])

    # 并查集求 prism 连通列
    parent = np.arange(n)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v in edges:
        ru, rv = int(find(u)), int(find(v))
        if ru != rv:
            parent[ru] = rv

    from collections import Counter
    roots = [find(int(c)) for c in np.where(prism_mask)[0]]
    lengths = Counter(roots)
    cols = sorted(lengths.values())
    hist = Counter(cols)
    return {
        "n_prism": n_prism,
        "n_columns": len(lengths),
        "column_lengths": cols,
        "length_histogram": {int(k): int(v) for k, v in sorted(hist.items())},
    }


def mesh_cells(mesh: dict) -> dict:
    """对 :func:`parse_mesh` 的结果重建单元。"""
    if not mesh or not mesh.get("n_faces"):
        return build_cells(np.empty(0, np.int64), np.empty(0, np.int64),
                           np.empty(0, np.int64))
    return build_cells(mesh["owner"], mesh["neigh"], mesh["npe"])


def gph_cells(data) -> dict:
    """打开 GPH buffer → 单元模型（单元类型直方图 + 单元→面邻接）。"""
    return mesh_cells(parse_mesh(data))


def parse_mesh(data) -> dict:
    """轻量网格提取：顶点坐标 + 面拓扑（供 3D 渲染）。

    返回：``vertices (n,3) f64``、``owner/neigh/npe``（I4/u4 数组，
    ``neigh == 0xFFFFFFFF`` = 边界面）、``conn``（CSR 连接表）与
    ``face_offsets``。大网格由调用方按面数上限抽样，本函数不切片。
    """
    # ── LS_Links：owner / neigh / npe 三个等长 I4 块 + 连接表 ───────
    section = _find_section(data, "LS_Links")
    if section is None:
        return {}
    blocks = [(b.offset, b.byte_count)
              for b in crdlfld.iter_data_blocks(data, section)
              if b.byte_count > 0 and b.byte_count % 4 == 0]
    if len(blocks) < 4:
        return {}
    sizes = [bc for _, bc in blocks]
    n_faces_block_size = None
    for size, count in Counter(sizes).most_common():
        if count >= 3 and size % 4 == 0 and size >= 4:
            n_faces_block_size = size
            break
    if n_faces_block_size is None:
        return {}
    n_faces = n_faces_block_size // 4
    triples = [b for b in blocks if b[1] == n_faces_block_size][:3]
    if len(triples) < 3:
        return {}
    (owner_p, _), (neigh_p, _), (npe_p, _) = triples
    owner = np.frombuffer(data, dtype=">i4", count=n_faces, offset=owner_p).astype(
        np.int64)
    neigh = np.frombuffer(data, dtype=">u4", count=n_faces, offset=neigh_p).astype(
        np.int64)
    npe = np.frombuffer(data, dtype=">u4", count=n_faces, offset=npe_p).astype(
        np.int64)
    conn_total = int(npe.sum())
    conn_parts = [b for b in blocks
                  if b[1] == conn_total * 4 and b[1] > 0]
    if not conn_parts:
        return {}
    conn = np.concatenate([
        np.frombuffer(data, dtype=">u4", count=conn_total, offset=p)
        for p, bc in conn_parts
        if bc == conn_total * 4])
    if conn.size != conn_total:
        return {}
    offsets = np.empty(n_faces + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(npe, out=offsets[1:])

    # ── LS_Nodes：三个等长 R8 轴块 ──────────────────────────────────
    nsec = _find_section(data, "LS_Nodes")
    vertices = np.empty((0, 3), dtype=np.float64)
    if nsec is not None:
        fblocks = [(b.offset, b.byte_count)
                   for b in crdlfld.iter_data_blocks(data, nsec)
                   if b.byte_count >= 8 and b.byte_count % 8 == 0]
        if len(fblocks) >= 3:
            sizes = [bc for _, bc in fblocks]
            target = max(set(sizes), key=sizes.count)
            trio = [b for b in fblocks if b[1] == target][:3]
            if len(trio) == 3:
                n = target // 8
                axes = [np.frombuffer(data, dtype=">f8", count=n, offset=p)
                        for p, _ in trio]
                vertices = np.column_stack(axes)
    return {
        "n_faces": n_faces,
        "vertices": vertices,
        "owner": owner,
        "neigh": neigh,
        "npe": npe,
        "conn": conn.astype(np.int64),
        "face_offsets": offsets,
        "boundary_mask": neigh == 0xFFFFFFFF,
    }


def _i32(value: int) -> bytes:
    import struct
    return struct.pack(">i", int(value))


def _descriptor(type_code: int, dim0: int, dim1: int) -> bytes:
    return _i32(12) + _i32(type_code) + _i32(dim0) + _i32(dim1)


def _block(payload: bytes) -> bytes:
    return _i32(12) + _i32(len(payload)) + payload + _i32(len(payload))


#: 节尾哨兵（宿主全节实测：I4=12 + 3×0 + I4=12）
_SECTION_SENTINEL_B = _i32(12) + _i32(0) + _i32(0) + _i32(0) + _i32(12)


def _section(name: str, body: bytes) -> bytes:
    """命名节头（40 字节）：[I4=32][name 32B][I4=32] + 记录流。

    实测所有节（FileRevision/LS_* / OverlapEnd）在 name 之后都有一个
    [I4=32] 尾随标记，随后才是描述符/数据块；crdlfld.Section 的
    records_start = start + 40 亦印证 40 字节节头（此前缺尾随 32，导致
    写端首描述符被读端跳过 4 字节——字节对齐修正）。
    """
    out = _i32(32) + name.ljust(32).encode("ascii") + _i32(32) + body
    # 非空节尾部有 20 字节「节结束哨兵」[12][0][0][0][12]（读端按 bc=0
    # 跳过）；空节（OverlapEnd 等）无此哨兵。节体若已自带哨兵则不再追加
    # （模板节 Comments/Cycle/Unused 自带，规则与 mdl/oct._section 一致）。
    if body and not body.endswith(_SECTION_SENTINEL_B):
        out += _SECTION_SENTINEL_B
    return out


def write_gph(filepath,
              vertices,
              faces,
              app: str = "SCTpre",
              date: int = 20260812) -> "Path":
    """写最小 CRDL-FLD GPH（单 owner=0 的边界面集合，可被读回）。"""
    import numpy as _np

    n_faces = len(faces)
    owner = _np.zeros(n_faces, dtype=">i4")
    neigh = _np.full(n_faces, 0xFFFFFFFF, dtype=">u4")
    return write_gph_volume(filepath, vertices, faces, owner, neigh,
                            app=app, date=date)


def write_gph_volume(filepath,
                     vertices,
                     faces,
                     owner,
                     neigh,
                     app: str = "SCTpre",
                     date: int = 20260812,
                     cvol=None,
                     volume_regions=None,
                     surface_regions=None,
                     parts=None,
                     assemblies=None,
                     element_info=None,
                     comments: Optional[str] = "PolyHedra") -> "Path":
    """写完整 CRDL-FLD GPH 体网格（LS_Links 含 owner/neigh）。

    vertices：(n,3) 浮点坐标；faces：多边形顶点索引列表（0-based），
    owner/neigh：与 faces 等长的单元索引；neigh == -1 表示
    边界面（写盘时转存为 0xFFFFFFFF）。

    可选节：assemblies（UTF-8 XML 字符串 → LS_Assemblies）、
    element_info（I4[n_cells] → Element_InformationFlag）、
    comments（ASCII 字符串 → Comments）。
    """
    import numpy as _np

    verts = _np.asarray(vertices, dtype=float).reshape(-1, 3)
    n_vertices = len(verts)
    n_faces = len(faces)
    conn_flat = [int(v) for face in faces for v in face]
    conn_total = len(conn_flat)
    owner = _np.asarray(owner, dtype=">i4").reshape(-1)
    neigh_u4 = _np.asarray(neigh, dtype=">i4").reshape(-1)
    if owner.size != n_faces or neigh_u4.size != n_faces:
        raise ValueError("owner/neigh length must equal faces length")
    neigh_u4 = _np.where(neigh_u4 < 0, _np.uint32(0xFFFFFFFF),
                         neigh_u4.astype(_np.int64)).astype(">u4")

    out = bytearray()
    out += _i32(8) + crdlfld.MAGIC + _i32(8) + _i32(4) + _i32(4) + _i32(4)  # 容器头第 4 个 I4（PPH_FORMAT_SPEC §2：实测 4,4,4）
    # 节序与内容严格对齐宿主 GPH（P2 验收：cradle box 工程实测 23 节）
    out += _section("FileRevision",
                    _descriptor(4, 1, 1) + _descriptor(4, 2025, 4))
    # GPH 的 Application 描述符为 (1,8,1)（与 MDL 的 (1,1,8) 维序相反，实测）
    out += _section("Application",
                    _descriptor(1, 8, 1) +
                    _block(app.encode("ascii")[:8].ljust(8)))
    out += _section("ApplicationVersion",
                    _descriptor(4, 1, 1) + _descriptor(4, 2025, 4))
    out += _section("ReleaseDate",
                    _descriptor(4, 1, 1) + _descriptor(4, 20251217, 4))
    out += _section("GridType",
                    _descriptor(4, 1, 1) + _descriptor(4, 1, 4))
    out += _section("Dimension",
                    _descriptor(4, 1, 1) + _descriptor(4, 3, 4))
    out += _section("Bias",
                    _descriptor(4, 1, 1) + _descriptor(4, 0, 4))
    out += _section("Date",
                    _descriptor(4, 1, 1) + _descriptor(4, date, 4))
    # 节序严格照宿主：Date → Comments → Cycle → Unused → Encoding
    if comments is not None:
        out += _comments_section(comments)
    out += _cycle_section()
    out += _unused_section()
    out += _section("Encoding",
                    _descriptor(1, 32, 1) + _block(b"UTF-8".ljust(32)))
    out += _section("HeaderDataEnd", b"")
    out += _section("OverlapStart_0", b"")
    if cvol is not None:
        out += _cvol_section(cvol)

    npe = _np.array([len(f) for f in faces], dtype=">u4")
    conn = _np.asarray(conn_flat, dtype=">u4")
    links = (
        _descriptor(4, 1, 1) + _descriptor(4, 1, 4) +
        _descriptor(4, 1, 1) + _descriptor(4, n_faces, 4) +
        # 宿主实测：**每个数组块前都重复一次 desc(4,n,1)**（非只写一次）
        _descriptor(4, n_faces, 1) + _block(owner.tobytes()) +
        _descriptor(4, n_faces, 1) + _block(neigh_u4.tobytes()) +
        _descriptor(4, n_faces, 1) + _block(npe.tobytes()) +
        _descriptor(4, 1, 1) + _descriptor(4, conn_total, 4) +
        _descriptor(4, conn_total, 1) + _block(conn.tobytes()))
    out += _section("LS_Links", links)

    nodes = (
        _descriptor(4, 1, 1) + _descriptor(4, 1, 4) +
        _descriptor(4, 1, 1) + _descriptor(4, n_vertices, 4) +
        # 同上：每个坐标轴块前重复 desc(8,n,1)
        _descriptor(8, n_vertices, 1) +
        _block(verts[:, 0].astype(">f8").tobytes()) +
        _descriptor(8, n_vertices, 1) +
        _block(verts[:, 1].astype(">f8").tobytes()) +
        _descriptor(8, n_vertices, 1) +
        _block(verts[:, 2].astype(">f8").tobytes()))
    out += _section("LS_Nodes", nodes)
    if surface_regions is not None:
        out += _surface_regions_section(surface_regions)
    if volume_regions is not None:
        out += _volume_regions_section(volume_regions)
    if parts is not None:
        out += _parts_section(parts)
    if assemblies is not None:
        out += _assemblies_section(assemblies)
    if element_info is not None:
        out += _element_info_section(element_info)
    out += _section("OverlapEnd", b"")

    path = Path(filepath)
    path.write_bytes(bytes(out))
    return path

def _cvol_section(cvol) -> bytes:
    """LS_CvolIdOfElements：每单元 cvol id（I4[n_cells]，大端）。"""
    cvol = np.asarray(cvol, dtype=">i4").reshape(-1)
    n = int(cvol.size)
    body = (_descriptor(4, 1, 1) + _descriptor(4, 1, 4) +
            _descriptor(4, 1, 1) + _descriptor(4, n, 4) +
            _descriptor(4, n, 1) + _block(cvol.tobytes()))
    return _section("LS_CvolIdOfElements", body)


def _element_info_section(flags, flag_types: int = 31) -> bytes:
    """Element_InformationFlag 写端（box 布局：9 描述符 + I4[n] 块）。"""
    flags = np.asarray(flags, dtype=">i4").reshape(-1)
    n = int(flags.size)
    body = (_descriptor(4, 1, 1) + _descriptor(4, 1, 4) +
            _descriptor(4, 1, 1) + _descriptor(4, 1, 4) +
            _descriptor(4, 1, 1) + _descriptor(4, int(flag_types), 4) +
            _descriptor(4, 1, 1) + _descriptor(4, n, 4) +
            _descriptor(4, n, 1) + _block(flags.tobytes()))
    return _section("Element_InformationFlag", bytes(body))


def _assemblies_section(xml) -> bytes:
    """LS_Assemblies 写端（box 布局：4 描述符 + UTF-8 XML 字符串块）。"""
    b = xml.encode("utf-8") if isinstance(xml, str) else bytes(xml)
    n = len(b)
    body = (_descriptor(4, 1, 1) + _descriptor(4, 1, 4) +
            _descriptor(4, 1, 1) + _descriptor(4, n, 4) +
            _descriptor(1, n, 1) +
            _block(b))
    return _section("LS_Assemblies", bytes(body))


#: 宿主固定元数据节模板（2026-09-13 从 cradle box 工程 GPH 逐字节提取）。
#: 三节均为定长模板（含各自的 20 字节节尾哨兵），值与环境相关处保留常量。
_META_COMMENTS_BODY = bytes.fromhex(
    "0000000c0000000100000050000000010000000c00000050506f6c794865647261202020202020202020202020202020"
    "202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020"
    "2020202020202020000000500000000c0000000000000000000000000000000c")
_META_CYCLE_BODY = bytes.fromhex(
    "0000000c0000000400000001000000010000000c0000000400000000000000040000000c000000080000000100000001"
    "0000000c000000080000000000000000000000080000000c0000000000000000000000000000000c00000020556e6974"
    "3a2454454d5020202020202020202020202020202020202020202020000000200000000c000000080000000200000001"
    "0000000c000000103ff00000000000000000000000000000000000100000000c0000000400000001000000010000000c"
    "0000000400000001000000040000000c0000000100000014000000010000000c0000001454454d502020202020202020"
    "2020202020202020000000140000000c0000000000000000000000000000000c")
_META_UNUSED_BODY = bytes.fromhex(
    "0000000c0000000400000001000000010000000c0000000460ad78ec000000040000000c000000080000000100000001"
    "0000000c000000084415af1d78b58c40000000080000000c0000000000000000000000000000000c")


def _cycle_section() -> bytes:
    """Cycle：宿主固定模板（内嵌 `Unit:$TEMP` 记录；本机 TEMP 路径）。"""
    return _section("Cycle", _META_CYCLE_BODY)


def _unused_section() -> bytes:
    """Unused：宿主固定模板（u32 时间戳 + f64 值）。"""
    return _section("Unused", _META_UNUSED_BODY)


def _comments_section(text: str = "PolyHedra") -> bytes:
    """Comments：网格类型名（默认 PolyHedra，宿主 box 实测）。"""
    if text == "PolyHedra":
        return _section("Comments", _META_COMMENTS_BODY)
    body = (_descriptor(1, 80, 1) +
            _block(text.encode("ascii")[:80].ljust(80)))
    return _section("Comments", body)


def _name255(text: str) -> bytes:
    return text.encode("ascii")[:255].ljust(255, b" ")


def _volume_regions_section(names) -> bytes:
    """LS_VolumeRegions：体区域名列表（box 布局：无内部种子点）。"""
    names = list(names)
    body = bytearray()
    body += (_descriptor(4, 1, 1) + _descriptor(4, 1, 4) +
             _descriptor(4, 1, 1) + _descriptor(4, len(names), 4) +
             _descriptor(4, 1, 1) + _descriptor(4, 255, 4))
    for name in names:
        # 宿主实测：每条记录前有 desc(1,255,1)
        body += _descriptor(1, 255, 1) + _block(_name255(name))
        body += (_descriptor(4, 1, 1) + _descriptor(4, 1, 4) +
                 _descriptor(4, 1, 1) + _descriptor(4, 1, 4))
    return _section("LS_VolumeRegions", bytes(body))


def _surface_regions_section(regions) -> bytes:
    """LS_SurfaceRegions：面区域（名称 + 面号数组，大端 I4）。

    ``regions`` 为 ``[(name, face_ids), ...]``，face_ids 为 0-based 面号。
    """
    regions = list(regions)
    body = bytearray()
    body += (_descriptor(4, 1, 1) + _descriptor(4, 1, 4) +
             _descriptor(4, 1, 1) + _descriptor(4, len(regions), 4) +
             _descriptor(4, 1, 1) + _descriptor(4, 255, 4))
    for name, ids in regions:
        ids = np.asarray(ids, dtype=">i4").reshape(-1)
        n = int(ids.size)
        # 宿主实测：每条区域记录前有 desc(1,255,1)（名称块描述符）
        body += _descriptor(1, 255, 1) + _block(_name255(name))
        body += (_descriptor(4, 1, 1) + _descriptor(4, n, 4) +
                 _descriptor(4, n, 1) + _block(ids.tobytes()))
        body += _descriptor(4, n, 1) + _block(
            np.full(n, 1, dtype=">i4").tobytes())
    return _section("LS_SurfaceRegions", bytes(body))


def _parts_section(parts) -> bytes:
    """LS_Parts：部件名 + cvol 规格（简单 int 或复合 id 列表）。"""
    parts = list(parts)
    body = bytearray()
    body += (_descriptor(4, 1, 1) + _descriptor(4, 1, 4) +
             _descriptor(4, 1, 1) + _descriptor(4, len(parts), 4) +
             _descriptor(4, 1, 1) + _descriptor(4, 255, 4))
    for name, spec in parts:
        # 宿主实测：每条记录前有 desc(1,255,1)
        body += _descriptor(1, 255, 1) + _block(_name255(name))
        if isinstance(spec, (list, tuple, set)):
            ids = sorted(set(int(x) for x in spec))
            body += (_descriptor(4, 1, 1) + _descriptor(4, len(ids), 4) +
                     _descriptor(4, 1, 1) + _descriptor(4, len(ids), 4) +
                     _block(np.asarray(ids, dtype=">i4").tobytes()))
        else:
            cid = int(spec)
            body += (_descriptor(4, 1, 1) + _descriptor(4, cid, 4) +
                     _descriptor(4, 1, 1) + _descriptor(4, cid, 4))
    return _section("LS_Parts", bytes(body))
