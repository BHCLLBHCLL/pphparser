#!/usr/bin/env python3
"""SCTSNAPSHOT 快照文件解析（scFLOW ``main.sctsnapshot``）。

文件是 **小端** 的嵌套记录流（与 CRDL-FLD 的大端不同！）：

.. code-block:: text

    record := TAG[16] (ASCII，空格填充) + LEN (u32le) + PAYLOAD[LEN]

顶层结构（按序）：

1. ``CADTHRUVERSION`` = 8，``TREESTRUCT``   — GUI 树状态（TreeState）
2. ``CADTHRUVERSION`` = 8，``VIEWSTRUCT``   — 视图状态（ViewState，DOUBLEARRAY 18×f64）
3. ``CADTHRUVERSION`` = 3，``TOPASSYSTRUCT`` — 顶层装配 + Parasolid 体
   （``UNIQUEBODYNUMBER`` + N × (``PKBODY_T`` + ``ZIPBODYBYTES``) + ``ASSEMBLY`` 树）
4. ``TOPASSYSTRUCT`` — 八叉树装配（``MESHPRMDLGDATA`` + ``ZIPOCTREE``）
5. ``BSGSEX`` — BodyShapeGroups：网格组 / 八叉树参数 / 区域加密限制
6. N × (``CADTHRUVERSION`` + ``QUEUESTRUCT``) — 其他 GUI 队列

``*STRUCT`` / ``ASSEMBLY`` / ``BODY`` / ``BYTEARRAY`` / ``WRAPBYTEARRAY`` /
``FACEGROUPSW`` / ``FACEINFOMAP`` / ``FFREVERSEMAP`` 等为容器（负载是子记录）。

ZIP 压缩块（``ZIPBODYBYTES`` / ``ZIPOCTREE`` / ``ZIPFACETINGRULES``）整段负载
即为 **Microsoft LZMS** 压缩流（Windows Compression API，
``COMPRESSION_ALGORITHM_LZMS = 4``，``cabinet.dll``）。流首可读字段：

.. code-block:: text

    [magic u32le = 0xC0E5510A][hdr_len u16le = 24][stream_id u16le]
    [uncompressed_size u64le][uncompressed_size 重复 u64le]
    [compressed_size u32le][...LZMS 压缩数据...]

解压后内容：

- ``ZIPBODYBYTES`` → ``CADthru/PKBody3`` 包装：``data`` 为 Blowfish 小端
  变体 ECB 密文（``blowfish_le``，固定密钥），解密后是 Parasolid
  二进制传输流（内嵌 ``SCH_3701153`` schema）
- ``ZIPOCTREE`` / ``ZIPFACETINGRULES`` → 嵌套的快照记录流

``ZIPOCTREE`` 内与八叉树相关的队列（``QUEUEBODY`` 布局均为
``INDEXARRAY`` + ``BYTEARRAY``；``INDEXARRAY`` = ``i32[2] = {count=1, offset=0}``，
表示后续 ``BYTEARRAY`` 为单段负载）：

- ``OCTREEBODY`` — CRDL-FLD，与 ``*.oct`` 字节级一致
- ``OCTREEDIVISION`` — 每 octant 1 bit 的 is-internal 位图（前序，
  子序 ``(1,3,2,0,5,7,6,4)``，LSB-first）；见 :meth:`octree_division`
- ``OCTREEREGION`` — 每 octant 1 字节标志（后序，子序 ``0..7``）；
  见 :meth:`octree_region`
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

ZIP_MAGIC = 0xC0E5510A
PKBODY3_MAGIC = b"CADthru/PKBody3"
PKBODY3_SCHEMA_PREFIX_LEN = 400  # 本样例四个体 data 共享前缀长度
COMPRESSION_ALGORITHM_LZMS = 4
# wimlib 的独立压缩类型常量（WIMLIB_COMPRESSION_TYPE_LZMS = 3）
WIMLIB_COMPRESSION_TYPE_LZMS = 3
PKBODY3_TRAILER_MARK = 0x17DA2940  # 大体可选尾标（非内容 CRC）

# UTF-16-LE 字符串标签
_UTF16_TAGS = {"STRINGW", "NAMESTRINGW", "PRPFILESTRINGW", "SFILESTRINGW"}
# 原始字节串标签（不按 UTF-16 解码）
_BYTES_TAGS = {"LOCATIONSTRING", "REALPOSNAMES", "ORGFILENAMES",
               "DPOINTARRAY", "FINARRAY", "FACETARRAY", "PKBOX"}
# u16 状态数组
_U16_TAGS = {"FACESTATES", "EDGESTATES", "VERTEXSTATES"}
# i32 PK id 数组
_I32_TAGS = {"FIDPKFACE", "EIDPKEDGE", "VIDPKVERTEX"}
# u8 数组
_U8_TAGS = {"EDGEISSEAMLINE", "BODYSELECTION"}
# f64 面片化容差（FACEGROUPW；DLL XML 以 %g 写出）
_MESH_TOL_TAGS = {
    "MESH_CHORDTOL", "MESH_CHORDANG", "MESH_SURFTOL", "MESH_SURFANG",
}
# 带单位的标量/点（见 ValueWithUnit / DPointU）
_VWU_TAGS = {
    "LENGTHVWU", "ANGLEVWU", "AREAVWU", "DENSITYVWU", "ENERGYVWU",
    "FORCEVWU", "TIMEVWU", "VOLUMEVWU",
}
# 4 字节标量（u32/i32 值）标签
_SCALAR4_TAGS = {
    "CADTHRUVERSION", "QUEUEID", "INTEGER", "BOOL", "SGBOOL",
    "UNIQUEBODYNUMBER", "UNIQUEBODYNUM4", "PKBODY_T", "PKASSEMBLY_T",
    "CHILDRENNUMBER", "CHILDRENTYPE", "FACEGROUPNUMBER", "FACEARRAYSIZE",
    "NAMELENGTH", "LOCATIONLENGTH", "VPARTID", "MESHENABLED", "RECALCNODE",
    "COLOR", "COLORINDEX", "DATAKITORGFLG", "SPATIALSEPFLG", "POCTREEASM",
    "OCTREEBALANCING", "CSINFO_CECOUNT", "I777", "DUMMYASSYINFO_UNUSED",
    "FACESTATESLENGTH", "EDGESTATESLENGTH", "VERTEXSTATESLEN",
    "FIDPKFACELENGTH", "EIDPKEDGELENGTH", "VIDPKVERTEXLEN",
    "ZEROLENGTH", "ZEROLENGTH2", "EDGEISSEAMLINE_UNUSED",
}
# 明确的叶子（不尝试按容器展开）
_LEAF_TAGS = (_UTF16_TAGS | _BYTES_TAGS | _U16_TAGS | _I32_TAGS | _U8_TAGS
              | _MESH_TOL_TAGS | _VWU_TAGS | {"DPOINTU"}
              | {"STRING", "DOUBLE", "INTARRAY", "DOUBLEARRAY", "TRANSFORMMATRIX",
                 "ZIPBODYBYTES", "ZIPOCTREE", "ZIPFACETINGRULES"})


def _cabinet_decompress(compressed: bytes,
                        uncompressed_size: Optional[int]) -> bytes:
    """Windows Compression API（``cabinet.dll``）解压 LZMS 流。"""
    import ctypes
    from ctypes import wintypes

    cab = ctypes.WinDLL("cabinet.dll")
    CreateDecompressor = cab.CreateDecompressor
    CreateDecompressor.argtypes = [
        wintypes.DWORD, wintypes.LPVOID, ctypes.POINTER(wintypes.LPVOID)]
    CreateDecompressor.restype = wintypes.BOOL
    Decompress = cab.Decompress
    Decompress.argtypes = [
        wintypes.LPVOID, wintypes.LPCVOID, ctypes.c_size_t,
        wintypes.LPVOID, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
    Decompress.restype = wintypes.BOOL
    CloseDecompressor = cab.CloseDecompressor
    CloseDecompressor.argtypes = [wintypes.LPVOID]
    CloseDecompressor.restype = wintypes.BOOL

    handle = wintypes.LPVOID()
    if not CreateDecompressor(COMPRESSION_ALGORITHM_LZMS, None,
                              ctypes.byref(handle)):
        raise OSError(ctypes.GetLastError(), "CreateDecompressor(LZMS) 失败")
    try:
        needed = ctypes.c_size_t(0)
        ok = Decompress(handle, compressed, len(compressed),
                        None, 0, ctypes.byref(needed))
        err = ctypes.GetLastError()
        # ERROR_INSUFFICIENT_BUFFER = 122：查询输出尺寸时的预期返回
        if not ok and err not in (0, 122):
            raise OSError(err, "Decompress(LZMS) 查询尺寸失败")
        cap = needed.value or (uncompressed_size or 0)
        if uncompressed_size and uncompressed_size > cap:
            cap = uncompressed_size
        if cap <= 0:
            raise OSError(err, "Decompress(LZMS) 无法确定输出尺寸")
        buf = ctypes.create_string_buffer(cap)
        got = ctypes.c_size_t(0)
        if not Decompress(handle, compressed, len(compressed),
                          buf, cap, ctypes.byref(got)):
            raise OSError(ctypes.GetLastError(), "Decompress(LZMS) 失败")
        return buf.raw[:got.value]
    finally:
        CloseDecompressor(handle)


def _wimlib_library():
    """加载 wimlib 动态库（跨平台 C 库），找不到返回 None。"""
    import ctypes
    import ctypes.util

    candidates = ["wimlib.dll", "libwim.so", "libwim.so.15", "libwim.so.14",
                  "libwim.so.13", "libwim.dylib"]
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    found = ctypes.util.find_library("wim")
    if found:
        try:
            return ctypes.CDLL(found)
        except OSError:
            return None
    return None


def _wimlib_error(lib, rc: int) -> str:
    try:
        fn = lib.wimlib_get_error_string
        fn.argtypes = [ctypes.c_int]
        fn.restype = ctypes.c_char_p
        msg = fn(rc)
        return msg.decode("utf-8", errors="replace") if msg else str(rc)
    except Exception:
        return str(rc)


def _wimlib_decompress(compressed: bytes, uncompressed_size: int) -> bytes:
    """用 wimlib 解压 LZMS 流（优先新 API，兼容 1.13 一次性 API）。

    ``compressed`` 必须是完整记录负载（含 MS-LZMS 块首 28 字节）。
    """
    import ctypes

    lib = _wimlib_library()
    if lib is None:
        raise OSError("未找到 wimlib 动态库（wimlib.dll / libwim.so）")
    out = ctypes.create_string_buffer(uncompressed_size)

    # 新 API（wimlib >= 1.14）：create_decompressor + decompress_with_decompressor
    create = getattr(lib, "wimlib_create_decompressor", None)
    if create is not None:
        create.argtypes = [ctypes.c_int, ctypes.c_size_t,
                           ctypes.POINTER(ctypes.c_void_p)]
        create.restype = ctypes.c_int
        decompressor = ctypes.c_void_p()
        rc = create(WIMLIB_COMPRESSION_TYPE_LZMS, uncompressed_size,
                    ctypes.byref(decompressor))
        if rc == 0:
            try:
                fn = lib.wimlib_decompress_with_decompressor
                fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                               ctypes.c_void_p, ctypes.c_size_t]
                fn.restype = ctypes.c_int
                rc = fn(decompressor, compressed, len(compressed),
                        out, uncompressed_size)
            finally:
                free = getattr(lib, "wimlib_free_decompressor", None)
                if free is not None:
                    free.argtypes = [ctypes.c_void_p]
                    free(decompressor)
            if rc != 0:
                raise OSError(rc,
                              f"wimlib_decompress_with_decompressor 失败: "
                              f"{_wimlib_error(lib, rc)}")
            return out.raw

    # 旧 API（wimlib <= 1.13）：一次性 wimlib_decompress(..., compression_type)
    fn = getattr(lib, "wimlib_decompress", None)
    if fn is None:
        raise OSError("wimlib 缺少 wimlib_decompress / wimlib_create_decompressor 符号")
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                   ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    fn.restype = ctypes.c_int
    rc = fn(compressed, len(compressed), out, uncompressed_size,
            WIMLIB_COMPRESSION_TYPE_LZMS)
    if rc != 0:
        raise OSError(rc, f"wimlib_decompress 失败: {_wimlib_error(lib, rc)}")
    return out.raw


def lzms_available() -> bool:
    """LZMS 解压后端是否可用：Windows ``cabinet.dll`` 或 wimlib。"""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.WinDLL("cabinet.dll")
            return True
        except OSError:
            pass
    return _wimlib_library() is not None


def lzms_decompress(compressed: bytes,
                    uncompressed_size: Optional[int] = None) -> bytes:
    """解压 LZMS 流：优先 Windows ``cabinet.dll``，回退 wimlib。

    ``compressed`` 必须是完整记录负载（含流首 28 字节可读字段），
    从偏移 0 整体解压（剥离前缀会使 cabinet.dll 返回 ERROR=605）。
    """
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.WinDLL("cabinet.dll")
            return _cabinet_decompress(compressed, uncompressed_size)
        except OSError:
            pass
    if uncompressed_size is None:
        if len(compressed) >= 16 and struct.unpack("<I", compressed[:4])[0] == ZIP_MAGIC:
            uncompressed_size = struct.unpack("<Q", compressed[8:16])[0]
        else:
            raise RuntimeError("LZMS 解压需要已知的未压缩尺寸（wimlib 路径）")
    return _wimlib_decompress(compressed, uncompressed_size)


def _cabinet_compress(plain: bytes) -> bytes:
    """Windows Compression API（``cabinet.dll``）压缩 LZMS 流。

    输出即完整 MS-LZMS 块（流首 ``0xC0E5510A`` + 24 字节头 + 数据），
    与宿主写入的 ``ZIPBODYBYTES`` / ``ZIPOCTREE`` 负载同构，可被
    :func:`lzms_decompress` / 宿主直接解压。
    """
    import ctypes
    from ctypes import wintypes

    cab = ctypes.WinDLL("cabinet.dll")
    CreateCompressor = cab.CreateCompressor
    CreateCompressor.argtypes = [
        wintypes.DWORD, wintypes.LPVOID, ctypes.POINTER(wintypes.LPVOID)]
    CreateCompressor.restype = wintypes.BOOL
    Compress = cab.Compress
    Compress.argtypes = [
        wintypes.LPVOID, wintypes.LPCVOID, ctypes.c_size_t,
        wintypes.LPVOID, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
    Compress.restype = wintypes.BOOL
    CloseCompressor = cab.CloseCompressor
    CloseCompressor.argtypes = [wintypes.LPVOID]
    CloseCompressor.restype = wintypes.BOOL

    handle = wintypes.LPVOID()
    if not CreateCompressor(COMPRESSION_ALGORITHM_LZMS, None,
                            ctypes.byref(handle)):
        raise OSError(ctypes.GetLastError(), "CreateCompressor(LZMS) 失败")
    try:
        needed = ctypes.c_size_t(0)
        ok = Compress(handle, plain, len(plain), None, 0,
                      ctypes.byref(needed))
        err = ctypes.GetLastError()
        if not ok and err not in (0, 122):
            raise OSError(err, "Compress(LZMS) 查询尺寸失败")
        cap = max(needed.value, len(plain) + 4096)
        buf = ctypes.create_string_buffer(cap)
        got = ctypes.c_size_t(0)
        if not Compress(handle, plain, len(plain), buf, cap,
                        ctypes.byref(got)):
            raise OSError(ctypes.GetLastError(), "Compress(LZMS) 失败")
        return buf.raw[:got.value]
    finally:
        CloseCompressor(handle)


def lzms_compress(data: bytes) -> bytes:
    """压缩 LZMS 流（仅 Windows ``cabinet.dll``；无跨平台开源压缩端）。

    与 :func:`lzms_decompress` 互逆：``lzms_decompress(lzms_compress(x))
    == x``。压缩字节不必与宿主原始压缩位逐字节相同（压缩级别/实现可
    不同），只要解压明文一致即可被宿主接受。
    """
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.WinDLL("cabinet.dll")
            return _cabinet_compress(bytes(data))
        except OSError:
            pass
    raise OSError("LZMS 压缩写端仅支持 Windows cabinet.dll")


@dataclass
class OctreeMdlBody:
    """``ZIPOCTREE`` / ``OCTREEMDLBODY``：八叉树关联的 CAD 面片体。

    布局（子记录顺序）：

    - ``INTEGER`` ×3 = ``n_vertices, n_fins, n_facets``
    - ``DPOINTARRAY`` = ``n_vertices × (f64 x,y,z)``
    - ``FINARRAY`` = ``n_fins × (i32 v0, i32 v1)`` 边/半边端点
    - ``FACETARRAY`` = ``n_facets × 9×i32``：
      ``(v0,v1,v2, facet_index, -1, -1, fin0, fin1, fin2)``
    - ``PKBOX`` = ``6×f64`` AABB ``(xmin,ymin,zmin,xmax,ymax,zmax)``
    - ``BYTEARRAY[n_facets]`` / ``BYTEARRAY[n_fins]`` 标志（本例全 1）
    - ``FACEGROUPSW`` 面组（如 ``open`` / ``$$$-$$$Part``）
    """

    n_vertices: int
    n_fins: int
    n_facets: int
    vertices: np.ndarray          # (n, 3) f64
    fins: np.ndarray              # (n_fins, 2) i32
    facets: np.ndarray            # (n_facets, 9) i32
    bbox_min: np.ndarray          # (3,)
    bbox_max: np.ndarray          # (3,)
    facet_flags: np.ndarray       # (n_facets,) u8
    fin_flags: np.ndarray         # (n_fins,) u8
    face_groups: list[dict]


def _parse_octree_mdl_body(rec: "SnapRecord") -> Optional[OctreeMdlBody]:
    if rec.tag != "OCTREEMDLBODY":
        return None
    ints = [c.value for c in rec.children
            if c.tag == "INTEGER" and isinstance(c.value, int)]
    if len(ints) < 3:
        return None
    nv, nfin, nfac = ints[0], ints[1], ints[2]
    verts = fins = facets = None
    bmin = bmax = None
    facet_flags = fin_flags = np.empty(0, dtype=np.uint8)
    groups: list[dict] = []
    raw_bas: list[bytes] = []
    for c in rec.children:
        if c.tag == "DPOINTARRAY" and isinstance(c.value, bytes):
            n = len(c.value) // 24
            verts = np.frombuffer(c.value, dtype="<f8").reshape(n, 3).copy()
        elif c.tag == "FINARRAY" and isinstance(c.value, bytes):
            n = len(c.value) // 8
            fins = np.frombuffer(c.value, dtype="<i4").reshape(n, 2).copy()
        elif c.tag == "FACETARRAY" and isinstance(c.value, bytes):
            n = len(c.value) // 36
            facets = np.frombuffer(c.value, dtype="<i4").reshape(n, 9).copy()
        elif c.tag == "PKBOX" and isinstance(c.value, bytes) and len(c.value) == 48:
            box = struct.unpack("<6d", c.value)
            bmin = np.array(box[:3]); bmax = np.array(box[3:])
        elif c.tag == "BYTEARRAY" and isinstance(c.value, bytes) and not c.children:
            raw_bas.append(c.value)
        elif c.tag == "FACEGROUPSW":
            for fg in c.find_all("FACEGROUPW"):
                entry: dict = {}
                for x in fg.children:
                    if x.tag == "NAMESTRINGW":
                        entry["name"] = x.value
                    elif x.tag == "FACEARRAYSIZE":
                        entry.setdefault("face_array_sizes", []).append(x.value)
                    elif x.tag == "MESHENABLED":
                        entry["mesh_enabled"] = x.value
                if entry:
                    groups.append(entry)
    for ba in raw_bas:
        if len(ba) == nfac:
            facet_flags = np.frombuffer(ba, dtype=np.uint8).copy()
        elif len(ba) == nfin:
            fin_flags = np.frombuffer(ba, dtype=np.uint8).copy()
    if verts is None or fins is None or facets is None or bmin is None:
        return None
    return OctreeMdlBody(
        nv, nfin, nfac, verts, fins, facets, bmin, bmax,
        facet_flags, fin_flags, groups)


@dataclass
class ValueWithUnit:
    """``LENGTHVWU`` 等：``f64 value`` + ``i32 unit_type``（12 字节）。

    单位类型码与 ``main.xenv`` UNIT / DLL ``ValueWithUnit`` XML 对应；
    本样例常见 ``unit_type=1``（模型长度单位）。
    """

    value: float
    unit_type: int

    @classmethod
    def parse(cls, payload: bytes) -> "ValueWithUnit":
        if len(payload) != 12:
            raise ValueError(f"ValueWithUnit 期望 12 字节，得到 {len(payload)}")
        value, unit_type = struct.unpack("<dI", payload)
        return cls(value, unit_type)


@dataclass
class DPointU:
    """``DPOINTU``：``3×f64`` 坐标 + ``3×i32`` 各轴单位类型（36 字节）。

    二进制布局为值在前、类型在后（与 DLL 导出 XML 的 Type/Value 叙述顺序相反）。
    """

    xyz: tuple[float, float, float]
    unit_types: tuple[int, int, int]

    @classmethod
    def parse(cls, payload: bytes) -> "DPointU":
        if len(payload) != 36:
            raise ValueError(f"DPOINTU 期望 36 字节，得到 {len(payload)}")
        x, y, z = struct.unpack_from("<ddd", payload, 0)
        tx, ty, tz = struct.unpack_from("<iii", payload, 24)
        return cls((x, y, z), (tx, ty, tz))


@dataclass
class PKBody3:
    """``ZIPBODYBYTES`` 解压后的 CADThru Parasolid 体包装。

    布局：``CADthru/PKBody3`` (15B) + ``u32le size`` +
    ``data[ceil8(size)]``（物理密文）。

    **关键修正（2026-08-02）**：``size`` 是逻辑数据长度，物理密文占
    ``ceil8(size)`` 字节，尾部为 Blowfish ECB 的零填充块密文。此前观察
    到的"尾标 ``0x17DA2940``"与"pad ``0xB1``"都是该零填充块的密文碎片：
    固定密钥下 ``E(0^8) = e5 e4 e5 b1 40 29 da 17``，低 32 位恰为
    ``0x17DA2940``。因此仅当 ``size % 8 != 0`` 时"尾标"出现（5 个实测
    体中 7643 / 17604 / 116572 非 8 倍数 → 出现；7824 / 3040 为 8 倍数
    → 不出现）；``0xB1`` 是密文第 ``size`` 字节。该值不是独立存储的
    标记或校验（已排除 CRC32 / Adler / 求和），也不承载任何状态语义。

    ``data`` 整体为 **Blowfish 小端变体 ECB** 密文（见 ``blowfish_le``，
    固定密钥 ``HowDareYouSaySuchAThing``）。``decrypt()`` 后是
    **Parasolid 二进制传输流**（类 ``.x_b``，内嵌 ``SCH_3701153`` schema
    与 ASCII 字段名）。同版本项目间"共享 400 字节前缀"的现象实为
    ECB 下相同 schema 头明文产生相同密文块。
    """

    data: bytes         # 物理密文（ceil8(size) 字节，尾部为零填充块密文）
    logical_size: int   # 声明的逻辑长度（size 字段，明文长度）

    @classmethod
    def parse(cls, raw: bytes) -> "PKBody3":
        if not raw.startswith(PKBODY3_MAGIC):
            raise ValueError("不是 CADthru/PKBody3 包装")
        if len(raw) < 19:
            raise ValueError("PKBody3 过短")
        size = struct.unpack("<I", raw[15:19])[0]
        phys = (size + 7) // 8 * 8  # 物理密文按 8 字节块补齐
        end = 19 + phys
        if end != len(raw):
            raise ValueError(
                f"PKBody3 尺寸不匹配: size={size} ceil8={phys} "
                f"wrapper={len(raw)}")
        return cls(raw[19:end], size)

    @property
    def checksum(self) -> Optional[int]:
        """兼容字段：0x17DA2940 当且仅当密文末尾是零填充块的低 32 位
        （即 ``logical_size % 8 != 0`` 时出现；非独立存储字段）。"""
        if self.data[-4:] == struct.pack("<I", PKBODY3_TRAILER_MARK):
            return PKBODY3_TRAILER_MARK
        return None

    @property
    def pad(self) -> bytes:
        """兼容字段：逻辑长度之后的密文碎片（零填充块密文的一部分）。"""
        tail = self.data[self.logical_size:]
        if self.checksum is not None and len(tail) >= 4:
            return tail[:-4]
        return tail

    def decrypt(self, key: bytes = None) -> bytes:
        """Blowfish-LE ECB 解密 ``data`` → Parasolid 二进制传输流。

        ``key`` 缺省为 scFLOW 固定密钥。输出以
        ``TRANSMIT FILE created by modeller version`` 头（其二进制变体
        前缀为 ``A3.: ``）开头；长度截取到 ``logical_size``（去掉零填充
        尾部，与原始明文逐字节一致）。
        """
        from blowfish_le import DEFAULT_KEY, decrypt_ecb
        plain = decrypt_ecb(self.data, key if key is not None else DEFAULT_KEY)
        return plain[: self.logical_size]

    @property
    def schema_prefix(self) -> bytes:
        """密文前 400 字节（历史名称；ECB 下同版本体间逐字节相同）。"""
        return self.data[:PKBODY3_SCHEMA_PREFIX_LEN]

    @property
    def body_payload(self) -> bytes:
        """密文第 400 字节之后部分（历史名称，语义同 schema_prefix）。"""
        if len(self.data) <= PKBODY3_SCHEMA_PREFIX_LEN:
            return b""
        return self.data[PKBODY3_SCHEMA_PREFIX_LEN:]


@dataclass
class ZipBlob:
    """LZMS 压缩块（整段 ``raw`` 即为可解压流）。"""

    codec_id: int                 # 流首 u16（非独立编解码选择器）
    uncompressed_size: int
    compressed_size: int
    payload: bytes = b""          # 流首 28 字节之后的压缩数据
    raw: bytes = b""              # 完整 LZMS 流（含流首字段）

    @classmethod
    def parse(cls, payload: bytes) -> "ZipBlob":
        if len(payload) >= 28 and struct.unpack("<I", payload[:4])[0] == ZIP_MAGIC:
            codec = struct.unpack("<H", payload[6:8])[0]
            unc = struct.unpack("<Q", payload[8:16])[0]
            comp = struct.unpack("<I", payload[24:28])[0]
            return cls(codec, unc, comp, payload[28:], payload)
        return cls(0, 0, len(payload), payload, payload)

    def decompress(self) -> bytes:
        """LZMS 解压，返回明文。"""
        return lzms_decompress(self.raw, self.uncompressed_size)

    def decompress_body(self) -> PKBody3:
        """``ZIPBODYBYTES`` → ``PKBody3``。"""
        return PKBody3.parse(self.decompress())

    def decompress_records(self, max_depth: int = 24) -> list["SnapRecord"]:
        """``ZIPOCTREE`` / ``ZIPFACETINGRULES`` → 嵌套记录树。"""
        data = self.decompress()
        records, _, _ = _parse_region(data, 0, len(data), 0, max_depth)
        return records


@dataclass
class SnapRecord:
    """快照记录树节点。"""

    tag: str
    offset: int
    length: int
    value: object = None          # 标量/字符串/数组（叶子）
    children: list["SnapRecord"] = field(default_factory=list)
    skipped: int = 0              # 负载中无法对齐跳过的字节数
    override: Optional[bytes] = None  # 写端覆写负载（serialize 优先使用）

    def find_all(self, tag: str) -> Iterator["SnapRecord"]:
        if self.tag == tag:
            yield self
        for c in self.children:
            yield from c.find_all(tag)

    def first(self, tag: str) -> Optional["SnapRecord"]:
        return next(self.find_all(tag), None)

    def text(self, max_value_len: int = 60) -> str:
        """单行显示。"""
        v = self.value
        if v is None and self.children:
            return f"{self.tag} [{self.length}] {{{len(self.children)} 子记录}}"
        if isinstance(v, bytes):
            return f"{self.tag} [{self.length}] <{len(v)} bytes>"
        if isinstance(v, np.ndarray):
            return f"{self.tag} [{self.length}] array{v.shape} {v[:6].tolist()}{'...' if v.size > 6 else ''}"
        if isinstance(v, ZipBlob):
            return (f"{self.tag} [{self.length}] LZMS id={v.codec_id} "
                    f"unc={v.uncompressed_size} comp={v.compressed_size}")
        if isinstance(v, ValueWithUnit):
            return (f"{self.tag} [{self.length}] = {v.value!r} "
                    f"(unit_type={v.unit_type})")
        if isinstance(v, DPointU):
            return (f"{self.tag} [{self.length}] = xyz={v.xyz} "
                    f"units={v.unit_types}")
        s = repr(v)
        if len(s) > max_value_len:
            s = s[:max_value_len] + "..."
        return f"{self.tag} [{self.length}] = {s}"

    def dump(self, depth: int = 0, max_depth: int = 99) -> list[str]:
        pad = "  " * depth
        lines = [pad + f"@{self.offset:#x} " + self.text()]
        if self.skipped:
            lines.append(pad + f"  (+{self.skipped} 字节未对齐填充)")
        if depth < max_depth:
            for c in self.children:
                lines.extend(c.dump(depth + 1, max_depth))
        return lines

    def serialize(self, src: bytes) -> bytes:
        """把本记录重新序列化为 TLV 字节（``[tag 16B][u32 len][payload]``）。

        ``src`` 是 ``self.offset`` 所相对的字节缓冲（顶层为完整文件字节，
        容器递归时为父记录 payload 切片）。已解码叶子值经
        :func:`_encode_scalar` 重编码；容器递归子记录并保留子记录间与
        尾部未对齐填充；未解码值回退到 ``src`` 原始字节。

        写端：``override`` 非空时叶子负载强制取该字节（长度应与
        ``length`` 一致，否则父容器填充错位——见
        :meth:`SctSnapshot.update_octree_region` 的等长约束）。
        """
        tagb = self.tag.encode("ascii")[:16].ljust(16, b" ")
        if self.children:
            sub = src[self.offset + 20:self.offset + 20 + self.length]
            body = bytearray()
            pos = 0
            for c in self.children:
                if c.offset > pos:
                    body += sub[pos:c.offset]
                body += c.serialize(sub)
                pos = c.offset + 20 + c.length
            if pos < len(sub):
                body += sub[pos:]
            payload = bytes(body)
        else:
            payload = None
            if self.override is not None:
                payload = self.override
            elif self.value is not None and not isinstance(self.value, bytes):
                payload = _encode_scalar(self.tag, self.value)
            if payload is None:
                payload = src[self.offset + 20:self.offset + 20 + self.length]
        return tagb + struct.pack("<I", len(payload)) + payload


def _serialize_records(records: list[SnapRecord], src: bytes) -> bytes:
    """把一段缓冲上的记录序列重新序列化（保留记录间与尾部填充）。

    与 :meth:`SctSnapshot.serialize` 的顶层逻辑相同，但作用于解压后
    的 ``ZIPOCTREE`` / ``ZIPFACETINGRULES`` 明文记录流（写回链路）。
    """
    out = bytearray()
    pos = 0
    for r in records:
        if r.offset > pos:
            out += src[pos:r.offset]
        out += r.serialize(src)
        pos = r.offset + 20 + r.length
    if pos < len(src):
        out += src[pos:]
    return bytes(out)


def _decode_scalar(tag: str, payload: bytes):
    """已知叶子标签的类型化解码；返回 None 表示按容器/原始处理。"""
    n = len(payload)
    if tag in _UTF16_TAGS:
        return payload.decode("utf-16-le", errors="replace")
    if tag == "STRING":
        return payload.decode("utf-8", errors="replace")
    if tag in _BYTES_TAGS:
        return payload
    if tag == "DOUBLE" and n == 8:
        return struct.unpack("<d", payload)[0]
    if tag in _MESH_TOL_TAGS and n == 8:
        return struct.unpack("<d", payload)[0]
    if tag in _VWU_TAGS and n == 12:
        return ValueWithUnit.parse(payload)
    if tag == "DPOINTU" and n == 36:
        return DPointU.parse(payload)
    if tag == "INTARRAY" and n % 4 == 0:
        return np.frombuffer(payload, dtype="<i4").astype(np.int64).copy()
    if tag in ("DOUBLEARRAY", "TRANSFORMMATRIX") and n % 8 == 0:
        return np.frombuffer(payload, dtype="<f8").astype(np.float64).copy()
    if tag in _U16_TAGS and n % 2 == 0:
        return np.frombuffer(payload, dtype="<u2").copy()
    if tag in _I32_TAGS and n % 4 == 0:
        return np.frombuffer(payload, dtype="<i4").astype(np.int64).copy()
    if tag in _U8_TAGS:
        return np.frombuffer(payload, dtype=np.uint8).copy()
    if tag in ("ZIPBODYBYTES", "ZIPOCTREE", "ZIPFACETINGRULES"):
        return ZipBlob.parse(payload)
    if tag in _SCALAR4_TAGS and n == 4:
        return struct.unpack("<i", payload)[0]
    return None

def _encode_scalar(tag: str, value) -> Optional[bytes]:
    """``_decode_scalar`` 的逆：把已解码值重新编码为 payload 字节。

    对每条已知叶子标签给出与解码严格互逆的编码（如 ``LENGTHVWU`` →
    ``<d value><I unit_type>``）；无法编码返回 None（调用方回退到原始字节）。
    """
    if tag in _UTF16_TAGS:
        return str(value).encode("utf-16-le")
    if tag == "STRING":
        return str(value).encode("utf-8")
    if tag in _BYTES_TAGS:
        return bytes(value)
    if tag == "DOUBLE" or tag in _MESH_TOL_TAGS:
        return struct.pack("<d", float(value))
    if tag in _VWU_TAGS:
        return struct.pack("<dI", float(value.value), int(value.unit_type))
    if tag == "DPOINTU":
        return struct.pack("<dddiii",
                           float(value.xyz[0]), float(value.xyz[1]),
                           float(value.xyz[2]),
                           int(value.unit_types[0]), int(value.unit_types[1]),
                           int(value.unit_types[2]))
    if tag == "INTARRAY":
        return np.asarray(value, dtype="<i4").tobytes()
    if tag in ("DOUBLEARRAY", "TRANSFORMMATRIX"):
        return np.asarray(value, dtype="<f8").tobytes()
    if tag in _U16_TAGS:
        return np.asarray(value, dtype="<u2").tobytes()
    if tag in _I32_TAGS:
        return np.asarray(value, dtype="<i4").tobytes()
    if tag in _U8_TAGS:
        return np.asarray(value, dtype=np.uint8).tobytes()
    if tag in ("ZIPBODYBYTES", "ZIPOCTREE", "ZIPFACETINGRULES"):
        return value.raw if isinstance(value, ZipBlob) else bytes(value)
    if tag in _SCALAR4_TAGS:
        return struct.pack("<i", int(value))
    return None


def _is_plausible_record(data: bytes, pos: int, end: int) -> bool:
    if pos + 20 > end:
        return False
    tagb = data[pos : pos + 16]
    if not all(32 <= b < 127 for b in tagb):
        return False
    if not tagb.strip():
        return False
    ln = struct.unpack("<I", data[pos + 16 : pos + 20])[0]
    return pos + 20 + ln <= end


def _resync(data: bytes, pos: int, end: int, limit: int = 65536) -> int:
    """从 pos 之后寻找下一条看似合法的记录头；找不到返回 -1。"""
    stop = min(end - 20, pos + limit)
    cand = pos + 1
    while cand <= stop:
        if _is_plausible_record(data, cand, end):
            return cand
        cand += 1
    return -1


def _parse_region(data: bytes, start: int, end: int, depth: int,
                  max_depth: int) -> tuple[list[SnapRecord], int, int]:
    """解析 [start, end) 内的记录序列。

    返回 ``(records, parsed_end, skipped_bytes)``。遇到无法对齐的字节时
    尝试向前重新同步（厂商在部分结构后写入未初始化/保留空间）。
    """
    records: list[SnapRecord] = []
    pos = start
    skipped = 0
    while pos + 20 <= end:
        if not _is_plausible_record(data, pos, end):
            nxt = _resync(data, pos, end)
            if nxt < 0:
                skipped += end - pos
                pos = end
                break
            skipped += nxt - pos
            pos = nxt
            continue
        tag = data[pos : pos + 16].decode("ascii", errors="replace").rstrip()
        ln = struct.unpack("<I", data[pos + 16 : pos + 20])[0]
        payload = data[pos + 20 : pos + 20 + ln]
        rec = SnapRecord(tag, pos, ln)
        value = _decode_scalar(tag, payload)
        if value is not None:
            rec.value = value
        elif (ln >= 20 and depth < max_depth and tag not in _LEAF_TAGS
              and _is_plausible_record(payload, 0, len(payload))):
            children, reached, sub_skipped = _parse_region(
                payload, 0, len(payload), depth + 1, max_depth)
            if children and reached >= len(payload) - 0:
                rec.children = children
                rec.skipped = sub_skipped
            else:
                rec.value = payload
        else:
            rec.value = payload
        records.append(rec)
        pos = pos + 20 + ln
    return records, pos, skipped


def _oct_children_from_refinement(
        refinement: np.ndarray) -> list[Optional[list[int]]]:
    """由 ``*.oct`` 前序位图栈式重建父子表。

    ``children[i]`` = 节点 i 的 8 个孩子前序下标（存储槽序
    ``0..7 = x+2y+4z``）；叶子为 ``None``。前序流中内部节点之后
    紧跟其 8 个孩子的子树，栈式消解即可无损重建。
    """
    n = int(refinement.shape[0])
    children: list[Optional[list[int]]] = [None] * n
    stack: list[list[int]] = []
    if n and refinement[0]:
        children[0] = [-1] * 8
        stack.append([0, 0])
    for pos in range(1, n):
        parent, slot = stack[-1]
        children[parent][slot] = pos
        stack[-1][1] = slot + 1
        if stack[-1][1] == 8:
            stack.pop()
        if refinement[pos]:
            children[pos] = [-1] * 8
            stack.append([pos, 0])
    return children


def _oct_postorder_indices(
        children: list[Optional[list[int]]]) -> list[int]:
    """满八叉树的后序下标序列（子槽序 0..7，节点最后）。

    迭代实现（百万级节点无需递归），与 DLL ``0x89d60`` 写端的
    DFS 发出顺序一致。
    """
    n = len(children)
    post: list[int] = []
    if n == 0:
        return post
    st: list[list[int]] = [[0, 0]]
    while st:
        node, nxt = st[-1]
        ch = children[node]
        if ch is None or nxt == 8:
            post.append(node)
            st.pop()
        else:
            st[-1][1] = nxt + 1
            st.append([ch[nxt], 0])
    return post


def encode_octree_region_postorder(
        refinement: np.ndarray, flags_oct: np.ndarray,
        pad_to: Optional[int] = None) -> np.ndarray:
    """``*.oct`` 前序区域标志 → ``OCTREEREGION`` 后序字节流（写端）。

    DLL ``0x89d60`` 读端的逆：**后序 DFS**（子槽序 ``0..7``、
    节点最后）每节点写 1 字节 ``flags_oct[node]``。与
    :meth:`SctSnapshot.octree_region_as_oct_order` 严格互逆：
    ``encode(ref, as_oct_order(ref)) == 文件原始后序字节``。

    参数：
      - ``refinement``：``*.oct`` / ``OCTREEBODY`` 的 ``U1[n]`` 位图
      - ``flags_oct``：与前序位图同下标的 ``u8[n]`` 区域标志
      - ``pad_to``：输出缓冲长度。宿主 ``BYTEARRAY`` 长于 ``n`` 时
        传原长度，尾部补零（写回等长约束）。
    """
    refinement = np.asarray(refinement, dtype=np.uint8)
    flags_oct = np.asarray(flags_oct, dtype=np.uint8)
    n = int(refinement.shape[0])
    if flags_oct.shape[0] != n:
        raise ValueError(
            f"flags_oct 长度 {flags_oct.shape[0]} != refinement 长度 {n}")
    children = _oct_children_from_refinement(refinement)
    post = _oct_postorder_indices(children)
    out = flags_oct[np.asarray(post, dtype=np.int64)]
    if pad_to is not None:
        if pad_to < n:
            raise ValueError(
                f"pad_to={pad_to} < n_octants={n}（缓冲不足）")
        if pad_to > n:
            out = np.concatenate(
                [out, np.zeros(pad_to - n, dtype=np.uint8)])
    return out


@dataclass
class SctSnapshot:
    """解析后的 sctsnapshot 文件。"""

    filepath: str
    records: list[SnapRecord]
    skipped_bytes: int

    @classmethod
    def load(cls, filepath: str, max_depth: int = 24) -> "SctSnapshot":
        with open(filepath, "rb") as f:
            data = f.read()
        records, _, skipped = _parse_region(data, 0, len(data), 0, max_depth)
        return cls(filepath, records, skipped)

    @classmethod
    def from_bytes(cls, data: bytes, max_depth: int = 24) -> "SctSnapshot":
        """从内存字节解析快照（无需落盘）。"""
        records, _, skipped = _parse_region(data, 0, len(data), 0, max_depth)
        return cls("", records, skipped)

    def find_all(self, tag: str) -> Iterator[SnapRecord]:
        for r in self.records:
            yield from r.find_all(tag)

    def first(self, tag: str) -> Optional[SnapRecord]:
        return next(self.find_all(tag), None)

    def dump(self, max_depth: int = 99) -> str:
        lines: list[str] = []
        for r in self.records:
            lines.extend(r.dump(0, max_depth))
        return "\n".join(lines)

    def serialize(self, original_data: bytes = None) -> bytes:
        """把顶层记录流重新序列化为完整快照字节。

        逐条顶层记录重编码（见 :func:`SnapRecord.serialize`），并保留记录
        之间的未对齐填充与尾部残留；未传 ``original_data`` 时按
        ``filepath`` 重新读取。
        """
        if original_data is None:
            with open(self.filepath, "rb") as f:
                original_data = f.read()
        out = bytearray()
        pos = 0
        for r in self.records:
            if r.offset > pos:
                out += original_data[pos:r.offset]
            out += r.serialize(original_data)
            pos = r.offset + 20 + r.length
        if pos < len(original_data):
            out += original_data[pos:]
        return bytes(out)

    # ── 语义提取 ─────────────────────────────────────────────────────
    def bodies(self) -> list[dict]:
        """Parasolid 体清单：``[{pk_body, zip}]``。"""
        out = []
        for top in self.find_all("TOPASSYSTRUCT"):
            pk = None
            for c in top.children:
                if c.tag == "PKBODY_T":
                    pk = c.value
                elif c.tag == "ZIPBODYBYTES" and isinstance(c.value, ZipBlob):
                    out.append({"pk_body": pk, "zip": c.value})
                    pk = None
        return out

    def zip_blobs(self, tag: str) -> list[ZipBlob]:
        """收集指定标签的全部 LZMS 块。"""
        return [r.value for r in self.find_all(tag)
                if isinstance(r.value, ZipBlob)]

    def decompress_bodies(self) -> list[dict]:
        """解压全部 ``ZIPBODYBYTES`` → ``PKBody3``。"""
        out = []
        for b in self.bodies():
            body = b["zip"].decompress_body()
            out.append({
                "pk_body": b["pk_body"],
                "zip": b["zip"],
                "pkbody3": body,
                "data_size": len(body.data),
            })
        return out

    def decompress_octree(self, max_depth: int = 8) -> list[SnapRecord]:
        """解压 ``ZIPOCTREE`` 为嵌套记录树（含 OCTREEBODY 等）。"""
        blobs = self.zip_blobs("ZIPOCTREE")
        if not blobs:
            return []
        return blobs[0].decompress_records(max_depth=max_depth)

    def octree_crdlfld_bytes(self) -> Optional[bytes]:
        """从 ``ZIPOCTREE`` 提取 ``OCTREEBODY`` 内的 CRDL-FLD 字节
        （与 ``*.oct`` 成员字节级一致）。"""
        records = self.decompress_octree(max_depth=8)

        def walk(recs: list[SnapRecord]) -> Optional[bytes]:
            for r in recs:
                if r.tag == "OCTREEBODY":
                    for qb in r.find_all("QUEUEBODY"):
                        for c in qb.children:
                            if (c.tag == "BYTEARRAY"
                                    and isinstance(c.value, bytes)
                                    and b"CRDL-FLD" in c.value[:16]):
                                return c.value
                found = walk(r.children)
                if found is not None:
                    return found
            return None

        return walk(records)

    def _octree_bytearray(self, tag: str) -> Optional[bytes]:
        """从解压后的 ZIPOCTREE 取指定标签 QUEUEBODY/BYTEARRAY。"""
        records = self.decompress_octree(max_depth=8)

        def walk(recs: list[SnapRecord]) -> Optional[bytes]:
            for r in recs:
                if r.tag == tag:
                    for qb in r.find_all("QUEUEBODY"):
                        for c in qb.children:
                            if c.tag == "BYTEARRAY" and isinstance(c.value, bytes):
                                return c.value
                found = walk(r.children)
                if found is not None:
                    return found
            return None

        return walk(records)

    # DIVISION 序列化子节点访问序（相对存储槽 0..7；存储槽 =
    # ``x + 2*y + 4*z``，与 ``*.oct`` / ``oct.py`` 一致）。
    # 来源：SCTprime_Bx64.dll!0x89be0 内嵌表 ``(1,3,2,0,5,7,6,4)``。
    OCTREE_DIVISION_CHILD_ORDER: tuple[int, ...] = (1, 3, 2, 0, 5, 7, 6, 4)

    def octree_division(self) -> Optional[np.ndarray]:
        """``OCTREEDIVISION`` 原始字节：``ceil(n_octants/8)`` 字节位图。

        语义（DLL 写入端 ``0x89be0`` / ``0x89ce0``）：

        - 每位对应一个八叉体：``1`` = 内部节点（有 8 子），``0`` = 叶子
        - **前序 DFS** 发出；子访问序见 :attr:`OCTREE_DIVISION_CHILD_ORDER`
        - 字节内 **LSB-first**（bit0 = 流中第 1 位）
        - 对满八叉树 ``n = 1+8*n_internal``，长度恰为 ``n_internal+1``

        与 ``*.oct`` refinement 描述同一棵树，仅子遍历置换不同；
        box / laptop 样本上按上述规则重放可达 100% 字节一致。
        """
        raw = self._octree_bytearray("OCTREEDIVISION")
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.uint8).copy()

    def octree_division_bits(self, n_octants: Optional[int] = None
                             ) -> Optional[np.ndarray]:
        """解包 ``OCTREEDIVISION`` 为 ``u8[n_octants]`` 的 is-internal 位流
        （前序 + :attr:`OCTREE_DIVISION_CHILD_ORDER` 序）。"""
        raw = self.octree_division()
        if raw is None:
            return None
        bits = np.unpackbits(raw, bitorder="little")
        if n_octants is not None:
            bits = bits[:n_octants]
        return bits

    def octree_region(self, n_octants: Optional[int] = None
                      ) -> Optional[dict]:
        """``OCTREEREGION``：每 octant 一字节标志（``0/1``）。

        序列化（DLL ``0x89d60``）：

        - **后序 DFS**：先 8 子（存储槽序 ``0..7``），再写本节点
        - 每节点写 ``node[+0x64]`` 的 1 字节；数组尾部为零填充
        - ``flags`` 返回值为**文件后序**下的 ``u8[n_octants]``，
          **不是** ``*.oct`` 前序下标。若需与 refinement 对齐，
          使用 :meth:`octree_region_as_oct_order`。

        语义（box / laptop 已钉死）：

        - ``flag ∈ {0,1}``，两个样例上 ``flag=1`` **全部为叶子**；
        - ``flag=1`` 集中在最深细化层：box 深度 4–5（其最深层，
          883/1968 叶子，位于 y∈[0,0.011] 上半精化板）；laptop 深度
          14–20（3,445,907 / 3,465,218 叶子，位于转子薄柱
          x∈[-54.47,-51.78]、y∈[4.92,5.47]、z∈[-0.24,0.28]）；
        - 区域索引空间与 ``OCTREERESTRRGN`` / MDL ``frid`` / ``csid-1``
          一致（laptop：case1=1、rotation1=2、impeller1=3）。
        """
        raw = self._octree_bytearray("OCTREEREGION")
        if raw is None:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        if n_octants is None:
            nonzero = np.flatnonzero(arr)
            n_octants = int(nonzero[-1]) + 1 if nonzero.size else 0
        flags = arr[:n_octants].copy()
        padding = int(arr.size - n_octants)
        return {
            "flags": flags,
            "order": "postorder",
            "padding": padding,
            "n_active": int(np.count_nonzero(flags)),
            "n_inactive": int(n_octants - np.count_nonzero(flags)),
            "raw_size": int(arr.size),
        }

    def octree_region_as_oct_order(
            self, refinement: np.ndarray) -> Optional[np.ndarray]:
        """把 ``OCTREEREGION`` 后序字节重映射为 ``*.oct`` 前序下标数组。

        ``refinement`` 为 ``*.oct`` / ``OCTREEBODY`` 的 ``U1[n]`` 位图
        （子序 ``0..7 = x+2y+4z``）。返回 ``flags_oct[i]`` 与
        ``refinement[i]`` 同下标。
        """
        n = int(refinement.shape[0])
        reg = self.octree_region(n_octants=n)
        if reg is None:
            return None
        children = _oct_children_from_refinement(refinement)
        post = _oct_postorder_indices(children)
        out = np.empty(n, dtype=np.uint8)
        flags = reg["flags"]
        for k, idx in enumerate(post):
            out[idx] = flags[k]
        return out

    def octree_refinement(self) -> Optional[np.ndarray]:
        """从 ``OCTREEBODY`` 的 CRDL-FLD 提取 ``U1[n]`` 前序位图。

        即 ``*.oct`` 文件 ``LS_OctOctantRefinement`` 节的内容，
        与 ``*.oct`` 字节级一致（写回链路的 refinement 来源）。
        """
        raw = self.octree_crdlfld_bytes()
        if raw is None:
            return None
        import os
        import tempfile
        from crdlfld import CrdlFldFile, iter_data_blocks
        fd, path = tempfile.mkstemp(suffix=".oct")
        try:
            with os.fdopen(fd, "wb") as tf:
                tf.write(raw)
            with CrdlFldFile.load(path) as f:
                sec = f.get_section("LS_OctOctantRefinement")
                if sec is None:
                    return None
                for b in iter_data_blocks(f.data, sec):
                    return np.asarray(b.as_u1(f.data),
                                      dtype=np.uint8).copy()
                return None
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def update_octree_region(self, flags_oct: np.ndarray,
                             refinement: Optional[np.ndarray] = None) -> bool:
        """把 ``*.oct`` 前序区域标志写入 ``ZIPOCTREE`` 的 OCTREEREGION。

        P3-2 写回链路（DLL 写端的本地等价实现）：

        1. ``encode_octree_region_postorder``：前序 → 后序字节
        2. 覆写解压记录树中 ``OCTREEREGION → QUEUEBODY → BYTEARRAY``
           的负载（等长约束：尾部零填充保持 ``BYTEARRAY`` 长度不变，
           避免父容器偏移错位）
        3. 记录重序列化 + ``lzms_compress`` 重压缩 ``ZIPOCTREE``

        成功返回 ``True``；调用方随后 :meth:`serialize` 得到新快照
        字节（顶层 ``ZIPOCTREE`` 记录长度自动重算）。
        """
        zip_rec = next((r for r in self.find_all("ZIPOCTREE")
                        if isinstance(r.value, ZipBlob)), None)
        if zip_rec is None:
            return False
        if refinement is None:
            refinement = self.octree_refinement()
        if refinement is None:
            raise ValueError("快照无 OCTREEBODY refinement，需显式传入")

        blob: ZipBlob = zip_rec.value
        plain = blob.decompress()
        records, _, _ = _parse_region(plain, 0, len(plain), 0, 24)

        def walk_find(recs: list[SnapRecord]) -> Optional[SnapRecord]:
            for r in recs:
                if r.tag == "OCTREEREGION":
                    for qb in r.find_all("QUEUEBODY"):
                        for c in qb.children:
                            if (c.tag == "BYTEARRAY"
                                    and isinstance(c.value, bytes)):
                                return c
                found = walk_find(r.children)
                if found is not None:
                    return found
            return None

        target = walk_find(records)
        if target is None:
            return False
        post = encode_octree_region_postorder(
            refinement, flags_oct, pad_to=len(target.value))
        if len(post) != len(target.value):
            raise ValueError(
                f"后序缓冲 {len(post)} != BYTEARRAY {len(target.value)}"
                "（等长约束破坏）")
        target.override = post.tobytes()

        new_plain = _serialize_records(records, plain)
        new_raw = lzms_compress(new_plain)
        blob.raw = new_raw
        if (len(new_raw) >= 28
                and struct.unpack("<I", new_raw[:4])[0] == ZIP_MAGIC):
            blob.codec_id = struct.unpack("<H", new_raw[6:8])[0]
            blob.uncompressed_size = struct.unpack(
                "<Q", new_raw[8:16])[0]
            blob.compressed_size = struct.unpack(
                "<I", new_raw[24:28])[0]
            blob.payload = new_raw[28:]
        return True

    def octree_restrict_regions(self) -> list[dict]:
        """``BSGSEX → OCTREEPARAM → OCTREERESTR → OCTREERESTRRGN`` 区域清单。

        每个区域记录字段（已钉死）：

        - ``kind``（INTEGER）：0 = open/环境；2 = 指定体区域；
        - ``index``（INTEGER）：区域索引，**与 MDL frid / csid-1 同索引空间**
          （laptop：open=0、case1=1、rotation1=2、impeller1=3）；
        - ``enabled``（BOOL）、附加整数与厚度 ``LENGTHVWU``（本样例 0.0）。

        box 样例 4 个 ``OCTREERESTR`` 全为空（无受限区域）；laptop 的
        meshing group 含 4 个区域（open + 3 个旋转机械体）。
        """
        out: list[dict] = []
        for rstr in self.find_all("OCTREERESTR"):
            for wrap in rstr.find_all("WRAPBYTEARRAY"):
                for rgn in wrap.find_all("OCTREERESTRRGN"):
                    ints = [c.value for c in rgn.children
                            if c.tag == "INTEGER" and isinstance(c.value, int)]
                    name = None
                    for c in rgn.children:
                        if c.tag == "STRING" and isinstance(c.value, str):
                            name = c.value
                            break
                    if len(ints) >= 2 and name is not None:
                        out.append({
                            "name": name,
                            "kind": int(ints[0]),
                            "index": int(ints[1]),
                        })
        return out

    def octree_mdl_body(self) -> Optional[OctreeMdlBody]:
        """解析 ``ZIPOCTREE`` / ``OCTREEMDLBODY`` CAD 面片体。"""
        for root in self.decompress_octree(max_depth=8):
            targets = ([root] if root.tag == "OCTREEMDLBODY"
                       else list(root.find_all("OCTREEMDLBODY")))
            for rec in targets:
                parsed = _parse_octree_mdl_body(rec)
                if parsed is not None:
                    return parsed
        return None

    def decompress_faceting_rules(self, max_depth: int = 12) -> list[SnapRecord]:
        """解压 ``ZIPFACETINGRULES`` 为嵌套记录树。"""
        blobs = self.zip_blobs("ZIPFACETINGRULES")
        if not blobs:
            return []
        return blobs[0].decompress_records(max_depth=max_depth)

    def assembly_tree(self) -> list[dict]:
        """装配树（名称 + 子节点 + 关联 PKBODY_T）。"""

        def walk(rec: SnapRecord) -> dict:
            name = None
            pk = None
            children = []
            node_type = rec.tag
            for c in rec.children:
                if c.tag == "NAMESTRINGW":
                    name = c.value
                elif c.tag in ("PKBODY_T", "PKASSEMBLY_T"):
                    pk = c.value
                elif c.tag in ("ASSEMBLY", "BODY"):
                    children.append(walk(c))
            return {"type": node_type, "name": name, "pk": pk, "children": children}

        return [walk(r) for r in self.find_all("ASSEMBLY")
                if any(p.tag == "TOPASSYSTRUCT" for p in self.records)]

    def face_groups(self) -> list[dict]:
        """所有命名的面组（FACEGROUPW）：所属体 + 面数等。"""
        out = []
        for fg in self.find_all("FACEGROUPW"):
            entry = {}
            for c in fg.children:
                if c.tag == "NAMESTRINGW":
                    entry["name"] = c.value
                elif c.tag == "FACEARRAYSIZE":
                    entry.setdefault("face_array_sizes", []).append(c.value)
                elif c.tag == "COLOR":
                    entry["color"] = c.value
                elif c.tag == "COLORINDEX":
                    entry["color_index"] = c.value
                elif c.tag == "MESHENABLED":
                    entry["mesh_enabled"] = c.value
            if entry:
                out.append(entry)
        return out

    def meshing_groups(self) -> list[dict]:
        """BSGSEX 中的网格组参数摘要。"""
        out = []
        for grp in self.find_all("BODYSHAPEGROUP"):
            entry = {"name": None, "parent": None, "octree_param": {}}
            for c in grp.children:
                if c.tag == "STRINGW":
                    if entry["name"] is None:
                        entry["name"] = c.value
                    elif entry["parent"] is None:
                        entry["parent"] = c.value
                elif c.tag == "OCTREEPARAM":
                    op = c.first("OCTREESIZEBYPRM")
                    if op:
                        vals = [x.value for x in op.children
                                if x.tag in ("DOUBLE", "INTEGER")]
                        entry["octree_param"]["size_by_param"] = vals
                    bal = c.first("OCTREEBALANCING")
                    if bal:
                        entry["octree_param"]["balancing"] = [
                            x.value for x in bal.children]
                    restr = []
                    for rr in c.find_all("OCTREERESTRRGN"):
                        info = {}
                        ints = [x.value for x in rr.children if x.tag == "INTEGER"]
                        nm = rr.first("STRING")
                        info["integers"] = ints
                        if nm:
                            info["region"] = nm.value
                        restr.append(info)
                    if restr:
                        entry["octree_param"]["restrictions"] = restr
            out.append(entry)
        return out
