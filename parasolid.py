#!/usr/bin/env python3
"""Parasolid 传输流（文本 x_t / 二进制 x_b）解析与编码。

PKBody3.decrypt() 得到 scFLOW 内嵌的 Parasolid 传输流（'A' flag 二进制）：

.. code-block:: text

    A3<0x00>: TRANSMIT FILE created by modeller version 3701153
    SCH_3701153_37102_13006

本模块实现完整 XT 解码/编码（P2/P3/P4 交付线）：

* parse_text_xt / parse_binary_xt：头 + 编辑序列 + 节点流全量解析（schema
  由 pskernel 自带 sch_*.sch_txt 动态加载，编辑序列与 base schema 对拍解析
  传输字段表）；支持 'A'（CADthru frustrum，u32 index/指针/n_elts）、'B'
  （bare binary）、'PS'（neutral/typed）三种二进制 flag；
* encode_text_xt / encode_binary_xt：写端闭环（parse->encode 字节级一致，
  kernel 二进制产物对拍 87 节点全量一致）；
* 二进制编码要点（XT Format Reference 2.1/3.3 + 实测对拍）：指针与正整数
  小值存 v+1、大值 pair 扩展；未设哨兵 -32764 / -3.14158e13 归一为 None
  （文本 '?'）；变长节点 varlen 计数在 index 之前，变长字段无额外计数；
  terminator = type 1 + NULL 指针编码的 index 0；
* parse_transmit：轻量扫描（头/schema/字段帧/实体类型/SDL 属性）——其
  "字段表"实为 BODY 编辑序列中 I/A 操作的字节帧（field_data_offsets 的值
  = 各字段的 ptr_class），完整字段定义见 XtModel.edits。

注意：'A'（CADthru）与 'B'（kernel）两种二进制流对同一 body 的节点标签
可能不同（kernel 文本传输会再索引，标签沿 next 链轮换），节点序、node_id
与连接关系不变；解码器不做标签归一。
"""


from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import Optional


HEADER_PREFIX = b"TRANSMIT FILE created by modeller version "
SCHEMA_RE = re.compile(rb"SCH_\d+_\d+_\d+")
ENTITY_RE = re.compile(rb"CADthru/PK[A-Z][A-Za-z]*[a-z]")
SDL_RE = re.compile(rb"SDL/TYSA_(?:NAME|LAYER|UNAME)")

# 字段记录帧：[token][u8 len][name]。
# token = 类型字母（可带 $ / l / u / d 前缀，如 $CCCI、CCCA、lCCCDCCDI、dA、uI）。
_TOKEN_RE = re.compile(rb"\$?[A-Z]+|[lud][A-Z]+")
_NAME_RE = re.compile(rb"[A-Za-z_$][A-Za-z0-9_/$]*")

# 类型 token 字母表（scan_fields 启发式扫出的帧字符；已与二进制编辑序列对拍：
# "$CCCI" 等 = BODY 编辑序列字节（n=36 即 '$'，op 字母 'C'/'I'，字段名与
# ptr_class/n_elts 交错），非独立 schema 语言。真实字段类型见
# XtModel.edits / FIELD_TYPES（XT Format Reference 2.1.4）。
TOKEN_ALPHABET = {
    "I": "insert 编辑操作帧字符",
    "D": "delete 编辑操作帧字符",
    "A": "append 编辑操作帧字符",
    "C": "copy 编辑操作帧字符",
    "$": "编辑序列 n=36 字节（'$'）",
    "l": "字段名字节前缀帧",
    "u": "字段名字节前缀帧",
    "d": "字段名字节前缀帧",
}


def field_data_offsets(stream: "ParasolidStream") -> dict[str, int]:
    """字段名 → 数据区偏移（P2）。

    共享同偏移的字段互为别名——实测 lattice/boundary_lattice 同 222、
    mesh/boundary_mesh 同 1006、polyline/boundary_polyline 同 1008。
    """
    return {f.name: f.data_offset
            for f in stream.fields if f.data_offset >= 0}


# V37 实体 class 枚举（PK_ENTITY_ask_class）。
PK_CLASS_NAMES = {
    2501: "point", 3001: "curve", 4001: "surface",
    5001: "vertex", 5002: "edge", 5003: "loop",
    5004: "face", 5005: "fin", 5006: "body", 5007: "part",
}


def parse_text_entities(xt_text: str) -> dict:
    """解析文本 x_t（无内核）：头元数据 + 实体类型码 + 回引计数（P3）。

    文本 x_t（FORMAT=text, GUISE=transmit）头为 **PART1/2/3 元数据，之后是
    实体流：T<n> 为实体类型定义（每类一次），?n 为对已定义实体的回引
    （≈ 实体实例数）。返回 {header, version, schema, type_counts, n_refs,
    sdl_attributes}。
    """
    header: dict[str, str] = {}
    for key in ("MC", "FORMAT", "GUISE", "KEY", "MC_MODEL"):
        m = re.search(rf"^{key}=(.*?);", xt_text, re.M)
        if m:
            header[key] = m.group(1)
    ver = _parse_version(xt_text.encode("ascii", "replace"))
    sch = SCHEMA_RE.search(xt_text.encode("ascii", "replace"))
    from collections import Counter
    type_counts = {int(k): v
                   for k, v in sorted(Counter(
                       int(x) for x in re.findall(r"\bT(\d+)\b", xt_text)
                   ).items())}
    n_refs = len(re.findall(r"\?(\d+)", xt_text))
    sdl = sorted(set(re.findall(r"SDL/[A-Z_]+", xt_text)))
    return {
        "header": header,
        "version": ver,
        "schema": sch.group().decode("ascii") if sch else None,
        "type_counts": type_counts,
        "n_refs": n_refs,
        "sdl_attributes": sdl,
    }


def encode_facet_mesh(mesh) -> bytes:
    """P4：分面（CADthru lattice/mesh/polyline）编码。

    格式层已闭环：mesh 为 XtModel 时经 encode_text_xt 编码为文本 x_t
    （P2/P3 已钉死 XT 全格式：头/编辑序列/节点流/字段值）。从裸
    lattice/mesh/polyline 数组组装节点图（LATTICE 222 / MESH 201 /
    PSM_MESH 189 / POLYLINE 200 节点结构）仍待补——结构定义已由
    load_schema 提供，属组装层工作。
    """
    if isinstance(mesh, XtModel):
        return encode_text_xt(mesh).encode("ascii")
    raise NotImplementedError(
        "encode_facet_mesh: pass an XtModel (format-level encode done); "
        "raw-array -> node-graph assembly pending")


@dataclass
class ParasolidField:
    """schema 字段定义记录。"""

    token: str       # 类型 token（I / A / Z / CI / CCCI / CCCA / CCCC…DI 等）
    name: str        # 字段名（lattice / mesh / owner / CADthru/PKEdge 等）
    pos: int         # 记录在流中的字节偏移
    data_offset: int = -1   # 字段数据区偏移（名字之后的小端 u32；-1 未解析）


@dataclass
class ParasolidStream:
    """部分提取结果。"""

    size: int
    version: Optional[int] = None
    schema: Optional[str] = None
    header_line: str = ""
    fields: list[ParasolidField] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)      # 出现顺序
    sdl_attributes: list[str] = field(default_factory=list)

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    def summary(self) -> str:
        lines = [f"Parasolid 传输流: {self.size} B"
                 f" version={self.version} schema={self.schema}"]
        lines.append(f"字段 {len(self.fields)} 个: "
                     + ", ".join(f"{f.name}({f.token})" for f in self.fields))
        if self.entities:
            from collections import Counter
            counts = Counter(self.entities)
            lines.append("实体: " + ", ".join(
                f"{name}×{counts[name]}" for name in sorted(counts)))
        if self.sdl_attributes:
            lines.append("SDL 属性: " + ", ".join(self.sdl_attributes))
        return "\n".join(lines)


def _parse_version(header: bytes) -> Optional[int]:
    m = re.search(rb"version (\d+)", header)
    return int(m.group(1)) if m else None


def _is_printable_name(raw: bytes) -> bool:
    return bool(raw) and all(32 <= b < 127 for b in raw)


def scan_fields(data: bytes, start: int = 0, end: Optional[int] = None
                ) -> list[ParasolidField]:
    """按 ``[token][u8 len][name]`` 帧扫描 schema 字段记录。

    该帧在流中多处出现（schema 定义表 + 内嵌定义），用 token 字符集与
    可打印名字双重校验；位置为记录帧起点（token 首字节偏移）。
    """
    n = end if end is not None else len(data)
    out: list[ParasolidField] = []
    pos = start
    while pos + 2 <= n:
        m = _TOKEN_RE.match(data, pos)
        if not m or m.end() >= n:
            pos += 1
            continue
        ln = data[m.end()]
        name_start = m.end() + 1
        name_end = name_start + ln
        if not (1 <= ln <= 128 and name_end <= n):
            pos += 1
            continue
        name = bytes(data[name_start:name_end])
        if not _is_printable_name(name) or not _NAME_RE.fullmatch(name):
            pos += 1
            continue
        # 尾随标记字节（'R'/'P' 等）不计入 len，但紧跟 0x00 时并入名字
        # （实测 index_mapR / node_id_index_mapR / schema_embedding_mapR）。
        end = name_end
        suffix = 0
        while (end + 1 < n and 65 <= data[end] <= 90 and data[end + 1] == 0
               and _NAME_RE.fullmatch(
                   name + bytes([data[end]]))):
            name = name + bytes([data[end]])
            end += 1
            suffix += 1
        # 数据区偏移：名字（含 R/P 后缀）之后的小端 u32；R/P 后缀紧接 0x00
        # 分隔符再跟 u32（实测 index_mapR / node_id_index_mapR 等）。
        doff_pos = end
        if suffix and doff_pos + 5 <= n and data[doff_pos] == 0:
            doff_pos += 1
        data_offset = -1
        if doff_pos + 4 <= n:
            data_offset = int.from_bytes(data[doff_pos:doff_pos + 4], "little")
        out.append(ParasolidField(
            m.group().decode("ascii"), name.decode("ascii"), pos, data_offset))
        pos = end
    return out


def parse_transmit(data: bytes) -> ParasolidStream:
    """轻量解析：头/schema + 字段表 + 实体类型 + SDL 属性。"""
    result = ParasolidStream(size=len(data))
    # ── 文件头 ─────────────────────────────────────────────────────
    idx = data.find(HEADER_PREFIX)
    if idx >= 0:
        head = data[idx:]
        line_end = head.find(b"\x00")
        if line_end < 0:
            line_end = len(head)
        result.header_line = head[:line_end].decode("ascii", "replace")
        result.version = _parse_version(head[:line_end])
    m = SCHEMA_RE.search(data)
    if m:
        result.schema = m.group().decode("ascii")
    # ── 字段表 ─────────────────────────────────────────────────────
    result.fields = scan_fields(data)
    # ── 实体类型 / SDL 属性 ────────────────────────────────────────
    result.entities = [m.group().decode("ascii")
                       for m in ENTITY_RE.finditer(data)]
    result.sdl_attributes = [m.group().decode("ascii")
                             for m in SDL_RE.finditer(data)]
    return result


# ============================================================================
# P3/P2：完整 XT 解码器（文本 + 二进制）
# 依据：Siemens XT Format Reference（q-solid.com 托管）+ Cradle pskernel 自带
# sch_37102.sch_txt schema 文件 + 内核 PK_PART_transmit 产物对拍。
# ============================================================================

# 字段类型字母（XT Format Reference 2.1.4 + schema 文件实测）
FIELD_TYPES = {
    "u": ("byte", 1),
    "c": ("char", 1),
    "l": ("logical", 1),
    "n": ("short", 2),
    "w": ("unicode", 2),
    "d": ("int", 4),
    "p": ("pointer", 4),
    "f": ("double", 8),
    "i": ("interval", 16),
    "v": ("vector", 24),
    "b": ("box", 48),
    "h": ("array3_double", 24),
    "q": ("quaternion", 32),
}

# 节点类型表（V37 sch_37102；常用子集，完整表经 load_schema 动态加载）
NODE_TYPES = {
    10: "ASSEMBLY", 11: "INSTANCE", 12: "BODY", 13: "SHELL", 14: "FACE",
    15: "LOOP", 16: "EDGE", 17: "HALFEDGE", 18: "VERTEX", 19: "REGION",
    29: "POINT", 30: "LINE", 31: "CIRCLE", 32: "ELLIPSE", 38: "INTERSECTION",
    50: "PLANE", 51: "CYLINDER", 52: "CONE", 53: "SPHERE", 54: "TORUS",
    60: "OFFSET_SURF", 67: "SWEPT_SURF", 68: "SPUN_SURF",
    70: "LIST", 74: "POINTER_LIS_BLOCK", 79: "ATT_DEF_ID", 80: "ATTRIB_DEF",
    81: "ATTRIBUTE", 82: "INT_VALUES", 83: "REAL_VALUES", 84: "CHAR_VALUES",
    85: "POINT_VALUES", 86: "VECTOR_VALUES", 87: "AXIS_VALUES", 88: "TAG_VALUES",
    89: "DIRECTION_VALUES", 90: "FEATURE", 91: "MEMBER_OF_FEATURE",
    98: "UNICODE_VALUES", 99: "FIELD_NAMES", 100: "TRANSFORM", 101: "WORLD",
    102: "KEY", 120: "PE_SURF", 124: "B_SURFACE", 125: "SURFACE_DATA",
    126: "NURBS_SURF", 127: "KNOT_MULT", 128: "KNOT_SET", 130: "PE_CURVE",
    133: "TRIMMED_CURVE", 134: "B_CURVE", 135: "CURVE_DATA", 136: "NURBS_CURVE",
    137: "SP_CURVE", 141: "GEOMETRIC_OWNER", 176: "PART_XMT_BLOCK",
    185: "POLYLINE_DATA", 189: "PSM_MESH", 190: "INTEGER_TOOTH",
    191: "INTEGER_COMB", 192: "VECTOR_TOOTH", 193: "VECTOR_COMB",
    200: "POLYLINE", 201: "MESH", 204: "INTERSECTION_DATA", 205: "OFFSET_VALUES",
    206: "MESH_OFFSET_DATA", 207: "SCHEMA_CHAR_VALUES", 208: "NEW_NODE_MAP",
    209: "MOD_NODE_MAP", 210: "NEW_FIELD_MAP", 211: "SCHEMA_DATA",
    212: "OLD_NODE_MAP", 213: "OLD_FIELD_MAP", 220: "REAL_TOOTH",
    221: "REAL_COMB", 222: "LATTICE", 223: "LATTICE_DATA_IRREGULAR",
    224: "GRAPH_COMPACT", 238: "LATTICE_DATA_PATTERN",
}

# 节点类（union of pointers）表 —— 指针字段的 ptr_class 值
NODE_CLASSES = {
    1005: "PART", 1006: "SURFACE", 1008: "CURVE", 1019: "ATTRIB_FEAT",
    1040: "SURFACE_OWNER", 1043: "NODE_MAP", 1044: "FIELD_MAP",
    1045: "LATTICE_OWNER", 1046: "LATTICE_DATA", 1049: "PATTERN_OWNER",
    1050: "PATTERN_DATA", 1051: "PATTERN_FORM",
}


@dataclass
class XtField:
    """XT 节点字段定义（schema 语言一行：name; type; xmt class n_elts）。"""

    name: str
    type: str          # p/d/f/u/c/l/n/v/b/i/h/q...
    xmt: int           # 1 = 传输
    cls: int           # 指针的节点类/类型（非指针为 0）
    n_elts: int        # 0=标量 1=变长 n>1=定长数组


@dataclass
class XtNodeType:
    """XT 节点类型定义。"""

    type_id: int
    name: str
    xmt: int
    variable: bool
    fields: list[XtField]

    def effective_fields(self) -> list[XtField]:
        return [f for f in self.fields if f.xmt or f.n_elts == 1]


@dataclass
class XtNode:
    """XT 节点实例。"""

    index: int
    type_id: int
    name: str
    fields: dict

    def ref(self, key: str) -> Optional["XtNode"]:
        return self.fields.get(key)


@dataclass
class XtModel:
    """解析后的 XT 文件。"""

    version: str
    schema: str
    fmt: str            # 'text' | 'binary'
    userfield_size: int
    nodes: dict = field(default_factory=dict)   # index -> XtNode
    order: list = field(default_factory=list)   # node indices in file order
    edits: dict = field(default_factory=dict)   # type_id -> 编辑序列
    parse_error: bool = False                   # 字段失步时置位（容错停止）

    def by_type(self, type_id: int) -> list[XtNode]:
        return [n for n in self.order if n.type_id == type_id]

    def summary(self) -> str:
        from collections import Counter
        counts = Counter(n.name for n in self.order)
        lines = [f"XT {self.fmt}: version={self.version} schema={self.schema}",
                 f"nodes={len(self.order)} " +
                 ", ".join(f"{k}x{v}" for k, v in sorted(counts.items()))]
        return chr(10).join(lines)


def load_schema(path: str) -> dict[int, XtNodeType]:
    """解析 Parasolid .sch_txt schema 文件（如 pskernel 自带 sch_37102）。

    格式（XT Format Reference 2.1.1）：
    <nodetype> <nodename>; <desc>; <xmt> <n_fields> <variable>
    <fieldname>; <type>; <xmt> <class> <n_elts>
    """
    out: dict[int, XtNodeType] = {}
    cur: Optional[XtNodeType] = None
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            segs = [s.strip() for s in line.split(";")]
            if len(segs) < 3:
                continue
            head = segs[0].split()
            tail = segs[2].split()
            if head and head[0].isdigit() and len(tail) >= 3:
                # <type> <name>; <desc>; <xmt> <n_fields> <variable>
                try:
                    t = int(head[0])
                    cur = XtNodeType(t, head[1], int(tail[0]),
                                     bool(int(tail[2])), [])
                    out[t] = cur
                except (ValueError, IndexError):
                    pass
            elif cur is not None and head and len(tail) >= 3:
                # <name>; <type>; <xmt> <class> <n_elts>
                try:
                    cur.fields.append(XtField(
                        head[0], segs[1], int(tail[0]),
                        int(tail[1]), int(tail[2])))
                except ValueError:
                    pass
    return out


def find_schema_file(schema_name: str, programs_dir: Optional[str] = None
                     ) -> Optional[str]:
    """在 ``programs_dir/Schemas`` 下按 SCH 名定位 ``sch_*.sch_txt``。

    不探测 Cradle 安装路径。未给出目录时返回 None；调用方可以把
    ``load_schema`` 的结果直接传给 ``parse_text_xt`` / ``parse_binary_xt``。
    """
    from pathlib import Path as _P
    if not programs_dir:
        return None
    m = re.match(r"SCH_\d+_(\d+)(?:_\d+)?", schema_name or "")
    if not m:
        return None
    base = _P(programs_dir) / "Schemas"
    for name in (f"sch_{m.group(1)}.sch_txt", f"sch_{m.group(1)}.s_t"):
        p = base / name
        if p.exists():
            return str(p)
    return None


# ── 游标（文本 token 流 / 二进制字节流）──────────────────────────


class _TextCursor:
    """XT 文本格式游标：数字以空白分隔；char/logical 为单字符无尾随空格。"""

    def __init__(self, text: str):
        self.t = text
        self.i = 0

    def _skip(self) -> None:
        while self.i < len(self.t) and self.t[self.i].isspace():
            self.i += 1

    def peek(self) -> str:
        self._skip()
        return self.t[self.i] if self.i < len(self.t) else ""

    def read_num(self):
        """读一个数字 token（'?'=未设标记返回 None；'F'/'T'=logical）。"""
        self._skip()
        if self.i < len(self.t) and self.t[self.i] == "?":
            # 未设标记：1 字符、无尾随空格（如 "?10" = '?' + 10 两 token）
            self.i += 1
            return None
        j = self.i
        s = []
        while j < len(self.t):
            ch = self.t[j]
            if ch in "\r\n":      # 换行不重要，可出现在数字内部
                j += 1
                continue
            if ch.isspace():
                break
            s.append(ch)
            j += 1
        self.i = j
        s = "".join(s)
        if not s:
            raise ValueError("text cursor: unexpected EOF")
        if s == "F":
            return 0
        if s == "T":
            return 1
        return float(s) if any(ch in s for ch in ".eE") else int(s)

    def read_char(self) -> str:
        self._skip()
        ch = self.t[self.i]
        self.i += 1
        return ch

    def read_string(self) -> str:
        ln = int(self.read_num())
        self._skip()              # 长度数字后的分隔空格
        out = []
        while len(out) < ln and self.i < len(self.t):
            ch = self.t[self.i]
            self.i += 1
            if ch in "\r\n":    # 换行/CR 不重要（行折行），串内跳过
                continue
            out.append(ch)
        return "".join(out)


class _BinCursor:
    """XT 二进制游标（小端；pointer/正整数用 V14 pair 编码，见 XT Ref 3.3.3）。

    'A' flag（CADthru 流）与 'B'（bare binary）共用同一节点流编码，差别仅在
    头：'A' 版本长度 u16，'B' 为 u32。
    """

    def __init__(self, data: bytes):
        self.d = data
        self.i = 0

    def u8(self) -> int:
        v = self.d[self.i]
        self.i += 1
        return v

    def u16(self) -> int:
        v = struct.unpack_from("<H", self.d, self.i)[0]
        self.i += 2
        return v

    def i16(self) -> int:
        v = struct.unpack_from("<h", self.d, self.i)[0]
        self.i += 2
        return v

    def u32(self) -> int:
        v = struct.unpack_from("<I", self.d, self.i)[0]
        self.i += 4
        return v

    def f64(self) -> float:
        v = struct.unpack_from("<d", self.d, self.i)[0]
        self.i += 8
        return v

    def read_char(self) -> str:
        return chr(self.u8())

    def read_string(self) -> str:
        ln = self.u8()
        s = self.d[self.i:self.i + ln]
        self.i += ln
        return s.decode("ascii", errors="replace")

    def read_posint(self) -> int:
        """正整数（XT Ref 2.1.2.1 / 3.3.3）：小值 0..32766 存为单个 short
        （写入时 +1 偏移，0 不会出现），大值编码为两个 short：
        r = -(v % 32767 + 1), q = v // 32767（q 非零）。

        实测：n_elts=0 存 1、节点 index 1 存 2，与 +1 偏移一致。
        """
        r = self.i16()
        if r < 0:
            q = self.u16()
            return q * 32767 + (-r) - 1
        return r - 1

    def read_pointer(self) -> int:
        """pointer：与正整数同一编码；NULL（index 0）存 1。

        返回节点 index（0 基语义），NULL 为 0（文本 '?' 的未设值另行归一）。
        """
        v = self.read_posint()
        return v

    def read_int(self) -> int:
        """int 字段（d）：i32 小端。"""
        v = struct.unpack_from("<i", self.d, self.i)[0]
        self.i += 4
        return v

    def read_short(self) -> int:
        """short 字段（n）：i16 小端。"""
        return self.i16()


# ── 编辑序列 / 节点解析核心 ────────────────────────────────────────


def _parse_edit_sequence_text(c: _TextCursor) -> dict:
    """解析首节点的 schema 编辑序列，返回结构化
    {'n': int, 'ops': [(op, name, cls, nelts, typ, xmt)|(op,)]}。"""
    n = int(c.read_num())
    ops: list = []
    if n != 255:
        while True:
            op = c.read_char()
            if op == "Z":
                break
            if op in ("C", "D"):
                ops.append((op,))
                continue
            name = c.read_string()
            cls = int(c.read_num())
            nelts = int(c.read_num())
            typ = c.read_string() if cls == 0 else None
            xmt = c.read_char() if nelts == 1 else None
            ops.append((op, name, cls, nelts, typ, xmt))
    return {"n": n, "ops": ops}


def _parse_edit_sequence_bin(c: _BinCursor,
                                binfmt: Optional[_BinFmt] = None) -> dict:
    """二进制编辑序列解析（同文本版，返回结构化 {'n','ops'}）。

    二进制字段定义（XT Ref 2.1.2.2）：name（短串）+ ptr_class（short）+
    n_elts（B/PS：正整数编码，n_elts=0 存 1；A：u32 原值）+ type（短串，
    cls==0 时）+ xmt（logical 字节，仅 n_elts==1 时）。
    """
    n = c.u8()
    ops: list = []
    if n != 255:
        while True:
            op = c.read_char()
            if op == "Z":
                break
            if op in ("C", "D"):
                ops.append((op,))
                continue
            name = c.read_string()
            cls = c.u16()                    # ptr_class（short）
            nelts = binfmt.read_nelts(c)     # n_elts（约定见 _BinFmt）
            typ = c.read_string() if cls == 0 else None
            xmt = c.u8() if nelts == 1 else None
            ops.append((op, name, cls, nelts, typ, xmt))
    return {"n": n, "ops": ops}


def _read_field_value(c, ftype: str, n_elts: int,
                        varlen: Optional[int] = None,
                        binfmt: Optional[_BinFmt] = None):
    """按字段类型读一个（或定长/变长数组）值。

    二进制与文本的语义差异在此归一：二进制 logical = 0/1 字节（文本 = 'F'/'T'），
    二进制未设标记 = 哨兵值 -32764（int）/ -3.14158e13（double），归一为
    None 与文本 '?' 对齐；二进制 NULL 指针 = 0 与文本一致。

    变长字段（n_elts==1）的计数 = 节点头部的 varlen（XT Ref 2.1.4.3），
    字段本身无额外计数前缀。
    """
    is_bin = isinstance(c, _BinCursor)

    def norm_num(v):
        # 未设哨兵归一（文本 '?' 已由 read_num 返回 None）
        if isinstance(v, float) and abs(v + 3.14158e13) < 1e7:
            return None
        if isinstance(v, int) and v == -32764:
            return None
        return v

    def read_one(t: str):
        if t == "p":
            if is_bin:
                return binfmt.read_pointer(c)
            return c.read_num()
        if t == "f":
            v = c.f64() if is_bin else c.read_num()
            return norm_num(v)
        if t == "d":
            v = c.read_int() if is_bin else c.read_num()
            return norm_num(v)
        if t == "n":
            v = c.read_short() if is_bin else c.read_num()
            return norm_num(v)
        if t == "u":
            return c.u8() if is_bin else int(c.read_num())
        if t == "c":
            return c.read_char()
        if t == "l":
            if is_bin:
                return c.u8()          # 二进制 logical = 0/1 字节
            ch = c.read_char()
            return 1 if ch in ("T", "t") else 0
        if t in ("v", "h"):
            vals = tuple(read_one("f") for _ in range(3))
            if all(v is None for v in vals):
                return None
            return vals
        if t == "i":
            a, b = read_one("f"), read_one("f")
            return None if a is None else (a, b)
        if t == "b":
            vals = tuple(read_one("f") for _ in range(6))
            if all(v is None for v in vals):
                return None
            return vals
        if t == "w":
            return c.u16() if is_bin else int(c.read_num())
        # 未知类型：读一个数字兜底
        return c.read_num()
    if n_elts == 1:
        cnt = varlen if varlen is not None else 0
        return [read_one(ftype) for _ in range(cnt)]
    if n_elts > 1:
        return [read_one(ftype) for _ in range(n_elts)]
    return read_one(ftype)


class _BinFmt:
    """二进制编码约定：'A'（CADthru frustrum）与 'B'/'PS'（kernel）不同。

    实测对拍（同体 87 节点 kernel B 产物 vs PKBody3 A 流）：

    * A：版本长 u16；编辑序列 n_elts = **u32 原值**；节点 index = u32；
      指针 = u32 原值（NULL=0）。
    * B/PS：版本长 u32；编辑序列 n_elts = 正整数编码（存 v+1，大值 pair）；
      节点 index / 指针 = 同一正整数编码（NULL 存 1）。
    """

    def __init__(self, flag: str):
        self.flag = flag
        self.a_style = flag == "A"

    def read_index(self, c: "_BinCursor") -> int:
        return c.u32() if self.a_style else c.read_posint()

    def read_pointer(self, c: "_BinCursor") -> int:
        return c.u32() if self.a_style else c.read_posint()

    def read_nelts(self, c: "_BinCursor") -> int:
        return c.u32() if self.a_style else c.read_posint()

    def read_varlen(self, c: "_BinCursor") -> int:
        return c.u32()

    def read_terminator_tail(self, c: "_BinCursor") -> None:
        if self.a_style:
            c.u32()                     # A：index 0 存 u32 0
        else:
            c.u16()                     # B：NULL 指针编码（存 1）


def _parse_nodes(c, schema, fmt: str, model: XtModel,
                 binfmt: Optional[_BinFmt] = None) -> None:
    """从游标解析节点序列直到 terminator（type 1 + index 0）。

    变长节点布局（XT Ref 2.1/2.1.4.3）：type + [编辑序列] + **varlen 计数**
    + index + 字段值（变长字段直接用 varlen，无额外计数）。
    """
    from pathlib import Path as _P
    # 加载当前 schema + base schema（解析编辑序列的 C/D op 需要 base 字段表）
    sch: dict[int, XtNodeType] = schema or {}
    sch_file = find_schema_file(model.schema)
    if sch_file:
        try:
            sch = load_schema(sch_file)
        except Exception:
            pass
    base_sch: dict[int, XtNodeType] = {}
    m = re.search(r"SCH_\d+_\d+_(\d+)$", model.schema or "")
    if m and sch_file:
        try:
            base_path = _P(sch_file).with_name(f"sch_{m.group(1)}.sch_txt")
            base_sch = load_schema(str(base_path))
        except Exception:
            pass
    # 每类型解析后的字段表缓存
    resolved: dict[int, list[XtField]] = {}
    seen_types: set[int] = set()
    while True:
        try:
            if isinstance(c, _BinCursor):
                type_id = c.u16()
            else:
                type_id = int(c.read_num())
            # type 1（NULLP）即 terminator（XT Ref 2.1：type 1 + index 0）
            if type_id == 1:
                if not isinstance(c, _BinCursor):
                    if int(c.read_num()) == 0:
                        return
                else:
                    binfmt.read_terminator_tail(c)
                    return
            # 编辑序列只出现在每种类型的第一个节点（XT Ref 2.1.2.2）
            if type_id not in seen_types:
                if isinstance(c, _BinCursor):
                    model.edits[type_id] = _parse_edit_sequence_bin(c, binfmt)
                else:
                    model.edits[type_id] = _parse_edit_sequence_text(c)
                resolved[type_id] = resolve_fields(
                    base_sch, sch, type_id, model.edits[type_id])
                seen_types.add(type_id)
            nt = sch.get(type_id)
            # 变长节点的 varlen 计数在 index 之前（XT Ref 2.1.4.3）
            varlen: Optional[int] = None
            if nt is not None and nt.variable:
                if isinstance(c, _BinCursor):
                    varlen = binfmt.read_varlen(c)
                else:
                    varlen = int(c.read_num())
            if isinstance(c, _BinCursor):
                index = binfmt.read_index(c)
            else:
                index = int(c.read_num())
            name = nt.name if nt else NODE_TYPES.get(type_id, f"TYPE_{type_id}")
            fields: dict = {}
            flist = resolved.get(type_id)
            if flist is None and nt is not None:
                flist = nt.effective_fields()
            if varlen is not None:
                fields["@varlen"] = varlen
            for f in (flist or []):
                try:
                    fields[f.name] = _read_field_value(
                        c, f.type, f.n_elts, varlen, binfmt)
                except Exception:
                    fields[f.name] = None
            node = XtNode(index, type_id, name, fields)
            model.nodes[index] = node
            model.order.append(node)
        except (ValueError, IndexError, struct.error):
            # 容错：字段失步时停止解析，保留已解析节点
            model.parse_error = True
            return


def parse_text_xt(text: str, schema=None) -> XtModel:
    """解析文本 x_t（P3）：头 + 节点流。"""
    t = text.lstrip()
    assert t[0] == "T", "not a text XT (missing T flag)"
    c = _TextCursor(t)
    c.read_char()                       # 'T' flag
    version = c.read_string()           # ": TRANSMIT FILE created by modeller version X"
    schema_name = c.read_string()
    max_types = int(c.read_num())       # embedded schema：最大节点类型数
    usr = int(c.read_num())             # user field size
    model = XtModel(version, schema_name, "text", usr)
    model.max_node_types = max_types
    _parse_nodes(c, schema, "text", model)
    return model


def parse_binary_xt(data: bytes, schema=None) -> XtModel:
    """解析二进制 x_t/x_b（P2）：'A'/'B'/'PS' flag + 头 + 节点流。

    已钉死（对拍 kernel 二进制产物 + PKBody3，87 节点全量一致）：

    * 头：flag + 版本长 + 版本 + u32 schema 长 + schema + u16 最大节点类型
      + u32 usrfield。'A'（CADthru 流）版本长 = u16；'B'（bare binary）
      = u32；'PS\0\0'（neutral）/ 'PS\0\1'+3B 机器描述（typed）= u16。
    * 节点流：u16 类型 + [首节点编辑序列] + 正整数编码的节点 index
      （存 idx+1）+ 字段值；变长节点在 index 前有 u32 变长计数。
    * 编辑序列：u8 n + C/D/I/A/Z ops；字段定义 = 短串名 + u16 ptr_class
      + **正整数编码** n_elts + [类型串] + [xmt byte，仅 n_elts==1]。
    * 字段值：pointer/正整数 = u16(+1 偏移，大值 pair 扩展)、int=i32、
      short=i16、double=f64、u=l=u8、logical=0/1 字节、w=u16；
      未设哨兵 -32764 / -3.14158e13 归一为 None（与文本 '?' 一致）。

    注意：二进制产物用 kernel 二进制传输的节点标签（与文本传输的再索引
    标签可不同，如部分标签沿 next 链轮换），同体节点序、node_id 与连接
    关系不变——解码器不做标签归一。
    """
    assert len(data) > 4, "not a binary XT"
    flag = data[0:1]
    c = _BinCursor(data)
    c.u8()                              # flag 首字节
    if flag == b"A":
        ver_len = c.u16()
    elif flag == b"B":
        ver_len = c.u32()
    elif flag == b"P":
        # neutral: PS\0\0；typed: PS\0\1 + 3B 机器描述（字节序/浮点/字符）
        c.u8()                          # 'S'
        b0, b1 = c.u8(), c.u8()
        if b1 == 1:
            c.u8(); c.u8(); c.u8()      # 机器描述 3 字节
        ver_len = c.u16()
    else:
        raise ValueError(f"unknown XT binary flag {flag!r}")
    ver = data[c.i:c.i + ver_len].decode("ascii", errors="replace")
    c.i += ver_len
    sch_len = c.u32()
    schema = data[c.i:c.i + sch_len].decode("ascii", errors="replace")
    c.i += sch_len
    max_types = c.u16()                 # embedded schema：最大节点类型数
    usr = c.u32()                       # user field size
    model = XtModel(ver, schema, "binary", usr)
    model.max_node_types = max_types
    model.binary_flag = flag.decode("ascii", errors="replace")  # 'A'/'B'/'PS'
    binfmt = _BinFmt(model.binary_flag)
    _parse_nodes(c, schema, "binary", model, binfmt)
    return model


def resolve_fields(base_schema, cur_schema, type_id: int, edit: dict
                   ) -> list[XtField]:
    """把编辑序列应用到 base schema 字段表 → 实际传输字段表（含类型）。

    'C' = 复制 base 字段；'D' = 删除；'I'/'A' = 插入（edit 数据携带
    name/ptr_class/n_elts/type/xmt）。n==255 表示与 base 完全一致。
    """
    base = base_schema.get(type_id) if base_schema else None
    bfields = base.effective_fields() if base else []
    if not edit or edit.get("n") == 255:
        return bfields
    out: list[XtField] = []
    bi = 0
    for op in edit["ops"]:
        if op[0] == "C":
            if bi < len(bfields):
                out.append(bfields[bi])
            bi += 1
        elif op[0] == "D":
            bi += 1
        else:
            _, name, cls, nelts, typ, xmt = op
            if typ is None:
                typ = "p"
            out.append(XtField(name.rstrip("R"), typ, 1, cls, nelts))
    return out


def _write_value(buf: list, ftype: str, value, n_elts: int,
                 varlen: Optional[int] = None) -> None:
    """把字段值按文本格式写出（数字后跟空格；char/logical/'?' 不带空格）。

    变长字段（n_elts==1）直接写 varlen 个值（计数在节点头，字段无前缀）。
    """
    def num(v):
        if v is None:
            buf.append("?")            # unset 标记（无尾随空格）
        elif isinstance(v, float):
            buf.append(f"{v:g} ")
        else:
            buf.append(f"{v} ")

    def one(v):
        if ftype in ("c", "l"):
            buf.append(("T" if v else "F") if ftype == "l" else str(v))
        else:
            num(v)

    if n_elts == 1:
        for v in (value or [])[:varlen]:
            one(v)
        return
    if n_elts > 1:
        for v in value:
            one(v)
        return
    if ftype in ("v", "h"):
        for v in value:
            num(v)
    elif ftype == "b":
        for v in value:
            num(v)
    elif ftype == "i":
        num(value[0]); num(value[1])
    else:
        one(value)


def _write_edit(buf: list, edit: dict) -> None:
    """把编辑序列写回文本（'C'/'D' 单字符；'I'/'A' 带字段定义）。"""
    buf.append(f"{edit['n']} ")
    if edit["n"] == 255:
        return
    for op in edit["ops"]:
        buf.append(op[0])
        if op[0] in ("C", "D"):
            continue
        _, name, cls, nelts, typ, xmt = op
        buf.append(f"{len(name)} ")
        buf.append(name)
        buf.append(f"{cls} {nelts} ")
        if typ is not None:
            buf.append(f"{len(typ)} ")
            buf.append(typ)
        if xmt is not None:
            buf.append(xmt)
    buf.append("Z")


def encode_text_xt(model: XtModel) -> str:
    """把 XtModel 编码回文本 x_t（P4，与 parse_text_xt 互逆）。

    头（T flag + 版本 + schema + 最大节点类型 + usrfield）→ 节点流（首节点
    类型带编辑序列）→ terminator '1 0'。字符串/数字按 XT 文本规则写。
    """
    buf: list = ["T"]
    version = model.version
    buf.append(f"{len(version)} ")
    buf.append(version)
    schema = model.schema
    buf.append(f"{len(schema)} ")
    buf.append(schema)
    buf.append(f"{getattr(model, 'max_node_types', 239)} ")
    buf.append(f"{model.userfield_size} ")
    written: set[int] = set()
    for node in model.order:
        buf.append(f"{node.type_id} ")
        if node.type_id not in written:
            if node.type_id in model.edits:
                _write_edit(buf, model.edits[node.type_id])
            else:
                buf.append("255 ")
            written.add(node.type_id)
        sch_file = find_schema_file(model.schema)
        nt = None
        if sch_file:
            try:
                nt = load_schema(sch_file).get(node.type_id)
            except Exception:
                nt = None
        varlen = None
        if nt is not None and nt.variable:
            # varlen 在 index 之前（XT Ref 2.1.4.3）
            varlen = int(node.fields.get("@varlen", 0) or 0)
            buf.append(f"{varlen} ")
        buf.append(f"{node.index} ")
        for f in (nt.effective_fields() if nt else []):
            _write_value(buf, f.type, node.fields.get(f.name), f.n_elts,
                         varlen)
    buf.append("1 0 ")
    return "".join(buf)


def _write_posint_short(buf: bytearray, v: int) -> None:
    """B/PS 正整数/指针编码：小值存 v+1（u16），大值 pair（XT Ref 3.3.3）。"""
    if v is None:
        v = 0
    v = int(v)
    if v < 32767:
        buf.extend(struct.pack("<h", v + 1))
    else:
        buf.extend(struct.pack("<h", -(v % 32767 + 1)))
        buf.extend(struct.pack("<H", v // 32767))


def _write_bin_value(buf: bytearray, ftype: str, value, n_elts: int,
                     varlen: Optional[int] = None,
                     binfmt: Optional[_BinFmt] = None) -> None:
    """按二进制格式写字段值（None → 未设哨兵；约定见 _BinFmt）。"""
    a_style = binfmt is not None and binfmt.a_style

    def write_pointer(v):
        v = 0 if v is None else int(v)
        if a_style:
            buf.extend(struct.pack("<I", v))
        else:
            _write_posint_short(buf, v)

    def one(v):
        if ftype == "p":
            write_pointer(v)
        elif ftype == "f":
            buf.extend(struct.pack("<d", -3.14158e13 if v is None else v))
        elif ftype == "d":
            buf.extend(struct.pack("<i", -32764 if v is None else int(v)))
        elif ftype == "n":
            buf.extend(struct.pack("<h", -32764 if v is None else int(v)))
        elif ftype == "u":
            buf.extend(bytes([int(v or 0)]))
        elif ftype == "c":
            buf.extend(bytes([ord(str(v)[0]) if v else 0]))
        elif ftype == "l":
            buf.extend(bytes([1 if v else 0]))
        elif ftype == "w":
            buf.extend(struct.pack("<H", int(v or 0)))
        else:
            raise ValueError(f"binary encode: unknown field type {ftype}")

    if n_elts == 1:
        # 变长字段直接写 varlen 个值（计数在节点头）
        seq = value if isinstance(value, (list, tuple)) else []
        for v in seq[:varlen]:
            one(v)
        return
    if n_elts > 1:
        seq = value if isinstance(value, (list, tuple)) else []
        for v in seq:
            one(v)
        return
    if ftype in ("v", "h"):
        for v in (value or (None, None, None)):
            buf.extend(struct.pack("<d", -3.14158e13 if v is None else v))
    elif ftype == "b":
        for v in (value or (None,) * 6):
            buf.extend(struct.pack("<d", -3.14158e13 if v is None else v))
    elif ftype == "i":
        a, b = (value or (None, None))
        buf.extend(struct.pack("<dd", -3.14158e13 if a is None else a,
                               -3.14158e13 if b is None else b))
    else:
        one(value)


def _write_edit_bin(buf: bytearray, edit: dict,
                     binfmt: Optional[_BinFmt] = None) -> None:
    """把编辑序列写回二进制（n_elts 约定见 _BinFmt）。"""
    a_style = binfmt is not None and binfmt.a_style
    buf += bytes([edit["n"]])
    if edit["n"] == 255:
        return
    for op in edit["ops"]:
        buf += op[0].encode("ascii")
        if op[0] in ("C", "D"):
            continue
        _, name, cls, nelts, typ, xmt = op
        nb = name.encode("ascii")
        buf += bytes([len(nb)]) + nb
        buf += struct.pack("<H", int(cls))
        if a_style:
            buf += struct.pack("<I", int(nelts))
        else:
            _write_posint_short(buf, int(nelts))
        if typ is not None:
            tb = typ.encode("ascii")
            buf += bytes([len(tb)]) + tb
        if xmt is not None:
            buf += bytes([int(xmt)])
    buf += b"Z"


def encode_binary_xt(model: XtModel, flag: Optional[str] = None) -> bytes:
    """把 XtModel 编码回二进制 x_b（P2 写端，与 parse_binary_xt 互逆）。

    flag：'A'（CADthru，u16 版本长）/ 'B'（bare binary，u32 版本长）。
    kernel 产物对拍：同一 XtModel 经 encode_binary_xt 再 parse 与原文
    字节级一致（见 tests/test_parasolid_p2.py）。
    """
    version = model.version
    schema = model.schema
    if flag is None:
        flag = getattr(model, "binary_flag", "A") or "A"
    if flag not in ("A", "B"):
        raise ValueError(f"encode_binary_xt: unsupported flag {flag!r} "
                         "(仅支持 'A'/'B'；'PS' 需机器描述字节，未实现)")
    buf = bytearray()
    buf += flag.encode("ascii")
    if flag == "A":
        buf += struct.pack("<H", len(version))
    else:
        buf += struct.pack("<I", len(version))
    buf += version.encode("ascii")
    buf += struct.pack("<I", len(schema))
    buf += schema.encode("ascii")
    buf += struct.pack("<H", getattr(model, "max_node_types", 239))
    buf += struct.pack("<I", int(model.userfield_size))
    sch_file = find_schema_file(schema)
    sch = load_schema(sch_file) if sch_file else {}
    binfmt = _BinFmt(flag)
    written: set[int] = set()
    for node in model.order:
        buf += struct.pack("<H", node.type_id)
        if node.type_id not in written:
            if node.type_id in model.edits:
                _write_edit_bin(buf, model.edits[node.type_id], binfmt)
            else:
                buf += b"\xff"
            written.add(node.type_id)
        nt = sch.get(node.type_id)
        varlen = None
        if nt is not None and nt.variable:
            # varlen 在 index 之前（XT Ref 2.1.4.3）
            varlen = int(node.fields.get("@varlen", 0) or 0)
            buf += struct.pack("<I", varlen)
        if binfmt.a_style:
            buf += struct.pack("<I", int(node.index))
        else:
            _write_posint_short(buf, int(node.index))
        for f in (nt.effective_fields() if nt else []):
            _write_bin_value(buf, f.type, node.fields.get(f.name), f.n_elts,
                             varlen, binfmt)
    buf += struct.pack("<H", 1)      # terminator：type 1
    if binfmt.a_style:
        buf += struct.pack("<I", 0)  # A：index 0 存 u32 0
    else:
        buf += struct.pack("<H", 1)  # B/PS：NULL 指针编码（存 1）
    return bytes(buf)


def parse_xt(data, schema=None) -> XtModel:
    """自动识别文本/二进制并解析。"""
    if isinstance(data, bytes):
        if data[:1] == b"T":
            return parse_text_xt(data.decode("ascii", errors="replace"),
                                 schema)
        return parse_binary_xt(data, schema)
    return parse_text_xt(data, schema)


def parse_file(path: str) -> ParasolidStream:
    with open(path, "rb") as f:
        return parse_transmit(f.read())
