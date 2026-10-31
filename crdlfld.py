#!/usr/bin/env python3
"""CRDL-FLD 二进制容器核心解析（scFLOW/SCTpre 通用记录格式）。

GPH / OCT / MDL 文件共享同一底层格式：

* 魔数 ``CRDL-FLD``（8 字节 ASCII），前置 ``I4=8`` 长度。
* 全部多字节整数 / 浮点均为 **大端序**（Big-Endian）。
* 文件头之后是一系列 **命名节（named section）**：
  ``[I4=32][名称 C1[32]，空格填充]`` + 记录流。
* 记录流由两种元素构成：

  - **描述符** ``[I4=12][type I4][dim0 I4][dim1 I4]``；
    ``type`` 4=I4 8=R8/字符串；``dim0`` 通常为数组长度。
  - **数据块** ``[I4=12][byte_count I4][payload][I4=byte_count]``
    （尾部 4 字节是与头部相同的 byte_count 哨兵）。

本模块只做通用扫描；各文件类型（MDL/OCT/GPH）在各自模块中解释。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

MAGIC = b"CRDL-FLD"
_LARGE_BYTES = 512 * 1024 * 1024  # 超过此大小改用 mmap


def read_i32_be(data, pos: int) -> int:
    return int.from_bytes(data[pos : pos + 4], "big")


def read_u32_be(data, pos: int) -> int:
    return int.from_bytes(data[pos : pos + 4], "big", signed=False)


def read_f64_be(data, pos: int) -> float:
    return struct.unpack(">d", data[pos : pos + 8])[0]


def open_buffer(filepath):
    """返回 bytes-like 缓冲；>512 MiB 时用 mmap 避免整文件读入内存。"""
    size = Path(filepath).stat().st_size
    if size <= _LARGE_BYTES:
        with open(filepath, "rb") as f:
            return f.read(), None
    import mmap

    f = open(filepath, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return mm, (mm, f)


@dataclass
class DataBlock:
    """一个数据块：``[I4=12][bc][payload][bc]``。"""

    offset: int          # payload 起始偏移
    byte_count: int
    header_offset: int   # [I4=12] 头偏移

    def payload(self, data) -> bytes:
        return bytes(data[self.offset : self.offset + self.byte_count])

    def as_i4(self, data) -> np.ndarray:
        n = self.byte_count // 4
        return np.frombuffer(data, dtype=">i4", count=n, offset=self.offset).astype(np.int64)

    def as_u1(self, data) -> np.ndarray:
        return np.frombuffer(data, dtype=np.uint8, count=self.byte_count, offset=self.offset).copy()

    def as_f8(self, data) -> np.ndarray:
        n = self.byte_count // 8
        return np.frombuffer(data, dtype=">f8", count=n, offset=self.offset).astype(np.float64)

    def as_ascii(self, data) -> str:
        raw = bytes(data[self.offset : self.offset + self.byte_count])
        return raw.decode("ascii", errors="replace").strip("\x00").rstrip()


@dataclass
class Descriptor:
    """一个描述符：``[I4=12][type][dim0][dim1]``。"""

    offset: int
    type_code: int   # 4=I4, 8=R8/C1
    dim0: int
    dim1: int


@dataclass
class Section:
    """命名节：``[I4=32][name 32B]`` + 记录流。"""

    name: str
    start: int           # [I4=32] 标记的偏移
    end: int             # 节结束（下一节开始或文件尾）

    @property
    def records_start(self) -> int:
        return self.start + 40


# 常见节名（用于在无节表情况下推断节边界）。未知节也能被通用扫描发现，
# 此列表仅作为 ``section_end()`` 的边界候选。
KNOWN_SECTION_NAMES = [
    "FileRevision", "Application", "ApplicationVersion", "ReleaseDate",
    "GridType", "Dimension", "Bias", "Date", "Comments", "Cycle",
    "Unused", "Encoding", "UnitOfCoordinates", "HeaderDataEnd", "OverlapStart_0",
    "LS_CvolIdOfElements", "LS_Links", "LS_Nodes", "LS_SurfaceRegions",
    "LS_SolverUnusedRegions", "LS_VolumeRegions", "LS_Parts",
    "LS_Assemblies", "LS_SPHFile", "Element_InformationFlag",
    "LS_CoordinateSystem", "LS_OctLastGenYear", "LS_OctRootOctantMinMax",
    "LS_OctOctantRefinement", "LS_OctOctantBlockID",
    "LS_Faces", "LS_CsidOfFaces", "LS_FridOfFaces", "LS_EdgeStateOfFaces",
    "LS_StateOfNodes", "LS_MdlClosedVolumes", "LS_MdlVolumeRegions",
    "LS_MdlSurfaceRegions",
    # scPOST 求解器 FLD 节（Samples_POST/FLD + flddecoding 仓）
    "LS_Elements", "LS_MatOfElements", "LS_VolumeGeometryArray",
    "LS_SurfaceGeometryArray", "LS_SFile", "LS_STREAMcoc",
    "LS_STREAMmultiblock", "LS_SolverUnusedRegions",
    "Pressure", "Temperature", "CN01", "VECT", "HVEC", "POTENTIAL",
    "OverlapEnd",
]


def find_section(data, name: str) -> int:
    """返回 ``[I4=32]`` 标记偏移，找不到返回 -1。"""
    name_padded = name.ljust(32).encode("ascii")
    idx = data.find(name_padded)
    while idx >= 4:
        if read_i32_be(data, idx - 4) == 32:
            return idx - 4
        idx = data.find(name_padded, idx + 1)
    return -1


def _valid_section_start(data, idx: int, n: int) -> bool:
    """校验通用扫描候选节，排除 ``[I4=32]+ASCII`` 的误报。

    32 字节字符串数据块（如单位名 ``'m'+空格``）在字节层面与节头完全同构，
    必须用后继内容消歧：真实节的名称之后是记录流（``[I4=12]`` 起始）、
    另一个节头（``[I4=32]`` + 可打印名称）、或文件尾/全零填充。
    """
    rec = idx + 40
    if rec + 4 > n:
        return True  # 文件尾
    m2 = read_i32_be(data, rec)
    if m2 == 12:
        return True  # 记录流起始
    if m2 == 32 and rec + 36 <= n:
        raw = bytes(data[rec + 4 : rec + 36])
        if all(b == 32 or 33 <= b < 127 for b in raw):
            return True  # 紧邻的下一个节头（40 字节空节，如 HeaderDataEnd）
    if m2 == 0:
        # 全零填充区（向前看一小段）
        tail = bytes(data[rec : min(rec + 64, n)])
        if not any(tail):
            return True
    return False


def scan_sections(data, names: Optional[list[str]] = None) -> list[Section]:
    """按文件顺序扫描所有命名节（含未知节）。

    通过查找 ``[I4=32]`` + 32 字节可打印 ASCII（含至少一个字母）模式定位节，
    因此不依赖硬编码节名列表；*names* 可用于补充无字母的节名。
    通用候选须经 :func:`_valid_section_start` 校验以排除字符串数据块误报。
    """
    names = names if names is not None else KNOWN_SECTION_NAMES
    found: dict[int, str] = {}
    for name in names:
        off = find_section(data, name)
        if off >= 0:
            found[off] = name
    # 通用模式扫描：\x00\x00\x00\x20 + 32 字节可打印 ASCII
    n = len(data)
    pos = 0
    needle = b"\x00\x00\x00\x20"
    while True:
        idx = data.find(needle, pos)
        if idx < 0 or idx + 36 > n:
            break
        raw = bytes(data[idx + 4 : idx + 36])
        if all(b == 32 or 33 <= b < 127 for b in raw):
            nm = raw.decode("ascii").rstrip()
            if (nm and any(c.isalpha() for c in nm)
                    and _valid_section_start(data, idx, n)):
                found.setdefault(idx, nm)
        pos = idx + 4
    ordered = sorted(found.items())
    sections: list[Section] = []
    for i, (off, nm) in enumerate(ordered):
        end = ordered[i + 1][0] if i + 1 < len(ordered) else n
        sections.append(Section(nm, off, end))
    return sections


def iter_records(data, section: Section) -> Iterator[DataBlock | Descriptor]:
    """按文件顺序产出节内的数据块与描述符。"""
    pos = section.records_start
    n = len(data)
    sec_end = section.end
    while pos + 8 <= sec_end and pos + 8 <= n:
        if read_i32_be(data, pos) != 12:
            pos += 4
            continue
        v = read_i32_be(data, pos + 4)
        if v in (4, 8) and pos + 16 <= sec_end:
            dim0 = read_i32_be(data, pos + 8)
            dim1 = read_i32_be(data, pos + 12)
            if 0 <= dim0 < 100_000_000 and 0 < dim1 < 100_000_000:
                yield Descriptor(pos, v, dim0, dim1)
                pos += 16
                continue
        bc = v
        if bc <= 0 or pos + 8 + bc + 4 > sec_end or pos + 8 + bc + 4 > n:
            pos += 4
            continue
        if read_i32_be(data, pos + 8 + bc) != bc:
            pos += 4
            continue
        yield DataBlock(pos + 8, bc, pos)
        pos = pos + 8 + bc + 4


def iter_data_blocks(data, section: Section) -> Iterator[DataBlock]:
    for rec in iter_records(data, section):
        if isinstance(rec, DataBlock):
            yield rec


def iter_descriptors(data, section: Section) -> Iterator[Descriptor]:
    for rec in iter_records(data, section):
        if isinstance(rec, Descriptor):
            yield rec


@dataclass
class CrdlFldFile:
    """CRDL-FLD 文件：文件头 + 命名节列表。"""

    filepath: str
    header_dims: tuple[int, int, int]
    sections: list[Section] = field(default_factory=list)
    _data: object = None
    _handles: object = None

    @classmethod
    def load(cls, filepath: str) -> "CrdlFldFile":
        data, handles = open_buffer(filepath)
        if bytes(data[4:12]) != MAGIC:
            raise ValueError(f"{filepath}: 不是 CRDL-FLD 文件")
        dims = (read_i32_be(data, 12), read_i32_be(data, 16), read_i32_be(data, 20))
        obj = cls(filepath, dims, [], data, handles)
        obj.sections = scan_sections(data)
        return obj

    def close(self) -> None:
        if self._handles:
            mm, f = self._handles
            mm.close()
            f.close()
            self._handles = None
            self._data = None

    def __enter__(self) -> "CrdlFldFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def data(self):
        return self._data

    def get_section(self, name: str) -> Optional[Section]:
        for s in self.sections:
            if s.name == name:
                return s
        return None

    # ── 元数据便捷访问 ────────────────────────────────────────────────
    def _meta_scalar(self, name: str):
        sec = self.get_section(name)
        if sec is None:
            return None
        for rec in iter_records(self._data, sec):
            if isinstance(rec, DataBlock):
                if rec.byte_count == 4:
                    return read_i32_be(self._data, rec.offset)
                if rec.byte_count % 8 == 0 and rec.byte_count >= 8:
                    return read_f64_be(self._data, rec.offset)
                txt = rec.as_ascii(self._data)
                if txt:
                    return txt
        return None

    def metadata(self) -> dict:
        """提取常见的标量/字符串元数据节。

        标量 I4 节（FileRevision/Dimension/Date/Bias/...）的取值记录
        ``[12][4][value][4]`` 与描述符同构，按惯例取**最后一个描述符的
        dim0** 作为标量值；字符串/数组节取数据块内容。
        """
        out = {"header_dims": self.header_dims}
        for name in ("FileRevision", "Application", "ApplicationVersion",
                     "ReleaseDate", "GridType", "Dimension", "Bias", "Date",
                     "Comments", "Cycle", "Encoding", "UnitOfCoordinates"):
            sec = self.get_section(name)
            if sec is None:
                continue
            blocks = list(iter_data_blocks(self._data, sec))
            vals = []
            for b in blocks:
                raw = bytes(self._data[b.offset : b.offset + b.byte_count])
                # 优先按 ASCII 字符串解释（如 Application="SCTpre  " 恰好 8 字节）
                if all(x == 0 or 32 <= x < 127 for x in raw):
                    s = raw.decode("ascii", errors="replace").strip("\x00").rstrip()
                    if s and any(c.isalpha() for c in s):
                        vals.append(s)
                        continue
                if b.byte_count == 4:
                    vals.append(read_i32_be(self._data, b.offset))
                elif b.byte_count == 8:
                    vals.append(read_f64_be(self._data, b.offset))
            if not vals:
                descs = [d.dim0 for d in iter_descriptors(self._data, sec)]
                if len(descs) >= 2:
                    vals.append(descs[-1])
            if vals:
                out[name] = vals[0] if len(vals) == 1 else vals
        return out
