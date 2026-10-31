#!/usr/bin/env python3
"""PPH 写端：ZIP 打包 + Blowfish 加密 + LZMS 压缩。

仓库原有解码链路（读 → 解密 → 解压 → 解析）的逆操作：

- :func:`lzms_compress` — Windows Compression API（``cabinet.dll``，
  ``COMPRESSION_ALGORITHM_LZMS=4``）压缩，输出与 scFLOW 样本同构的
  ``[magic 0xC0E5510A][hdr_len 24][stream_id][unc u64 ×2][comp u32]``
  流（28 字节头 + LZMS 负载），可直接被 ``sctsnapshot.lzms_decompress`` /
  ``ZipBlob.parse`` 消费；
- :func:`encrypt_pkbody3` — Blowfish-LE ECB（``blowfish_le.encrypt_ecb``）
  加密 + ``CADthru/PKBody3`` 包装（可选尾标 ``0x17DA2940`` / pad）；
- :func:`clone_pph` — 把成员字节写回标准 ZIP/deflate 容器；
- :func:`rewrite_pph` — 读 →（可选替换成员）→ 写，支撑"导出转换与
  文件互操作"闭环。

注意：LZMS 压缩目前依赖 Windows ``cabinet.dll``（与读取端一致）；
非 Windows 上仅 ZIP / Blowfish 路径可用。加密与解压均经过真实样例
round-trip 验证（box.pph 的 PKBody3 体：解密 → 再加密 → 与原始密文
逐字节一致；压缩 → 解压 → 与明文逐字节一致）。
"""

from __future__ import annotations

import struct
import sys
import zipfile
from pathlib import Path
from typing import Optional

import blowfish_le

LZMS_HEADER_LEN = 28
LZMS_MAGIC = 0xC0E5510A
COMPRESSION_ALGORITHM_LZMS = 4
PKBODY3_MAGIC = b"CADthru/PKBody3"

# 说明：历史上把密文末尾的 0x17DA2940 误读为"尾标"。实测它是固定密钥下
# Blowfish E(0^8) = e5 e4 e5 b1 40 29 da 17 的低 32 位——即零填充块密文的
# 一部分，仅当明文长度非 8 倍数时出现；不是独立存储字段（见 sctsnapshot.PKBody3）。
PKBODY3_TRAILER_MARK = 0x17DA2940  # 保留常量名以兼容旧引用


def lzms_compress_available() -> bool:
    """LZMS 压缩是否可用（Windows ``cabinet.dll`` CreateCompressor）。"""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        ctypes.WinDLL("cabinet.dll")
        return True
    except OSError:
        return False


def lzms_compress(data: bytes) -> bytes:
    """LZMS 压缩，返回完整流（28 字节头 + 负载）。"""
    if sys.platform != "win32":
        raise RuntimeError("LZMS 压缩需要 Windows cabinet.dll")
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
        ok = Compress(handle, data, len(data), None, 0, ctypes.byref(needed))
        err = ctypes.GetLastError()
        if not ok and err not in (0, 122):  # 122 = ERROR_INSUFFICIENT_BUFFER
            raise OSError(err, "Compress(LZMS) 查询尺寸失败")
        if needed.value <= 0:
            raise OSError(err, "Compress(LZMS) 无法确定输出尺寸")
        buf = ctypes.create_string_buffer(needed.value)
        got = ctypes.c_size_t(0)
        if not Compress(handle, data, len(data), buf, needed.value,
                        ctypes.byref(got)):
            raise OSError(ctypes.GetLastError(), "Compress(LZMS) 失败")
        return buf.raw[: got.value]
    finally:
        CloseCompressor(handle)


def parse_lzms_header(stream: bytes) -> dict:
    """读 LZMS 流首 28 字节（与 ``ZipBlob.parse`` 同布局）。"""
    if len(stream) < LZMS_HEADER_LEN:
        raise ValueError("LZMS 流过短")
    magic, hdr_len, stream_id = struct.unpack_from("<IHH", stream, 0)
    unc1, unc2 = struct.unpack_from("<QQ", stream, 8)
    comp = struct.unpack_from("<I", stream, 24)[0]
    if magic != LZMS_MAGIC:
        raise ValueError(f"LZMS 魔数不符: {magic:#x}")
    if unc1 != unc2:
        raise ValueError("LZMS 头内未压缩尺寸字段不一致")
    if 28 + comp != len(stream):
        raise ValueError(
            f"LZMS 长度不符: 28+comp={28 + comp} 实际={len(stream)}")
    return {"magic": magic, "hdr_len": hdr_len, "stream_id": stream_id,
            "uncompressed_size": unc1, "compressed_size": comp}


def encrypt_pkbody3(plaintext: bytes) -> bytes:
    """Parasolid 明文 → ``CADthru/PKBody3`` 包装（Blowfish-LE ECB）。

    布局与读取端一致：``CADthru/PKBody3`` + ``u32le len(plaintext)`` +
    零填充到 8 字节倍数后的完整密文。对 ``PKBody3.decrypt()`` 的结果
    再加密可**逐字节复现**原始密文（ECB 确定性 + 原始填充即零字节）。
    """
    pad8 = plaintext + b"\x00" * ((-len(plaintext)) % 8)
    cipher = blowfish_le.encrypt_ecb(pad8)
    out = bytearray()
    out += PKBODY3_MAGIC
    out += struct.pack("<I", len(plaintext))  # size = 逻辑明文长度
    out += cipher
    return bytes(out)


def clone_pph(src: str, dst: str,
              member_overrides: Optional[dict[str, bytes]] = None) -> None:
    """把 src 的所有成员写入新 pph（ZIP/deflate 容器）。

    ``member_overrides`` 可替换指定成员字节（如重写后的 main.xml），
    支撑"读 → 改 → 写回"互操作。源 ZIP 中不存在的 override 键会作为
    **新成员追加**（空工程原生 Execute 写 OCT/GPH 需要这条路径）。
    """
    overrides = member_overrides or {}
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    with zipfile.ZipFile(src) as zin, \
            zipfile.ZipFile(dst_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = overrides.get(info.filename)
            if data is None:
                data = zin.read(info.filename)
            zout.writestr(info.filename, data)
            written.add(info.filename)
        for name, data in overrides.items():
            if name not in written:
                zout.writestr(name, data)


def rewrite_pph(src: str, dst: str,
                member_overrides: Optional[dict[str, bytes]] = None) -> None:
    """与 :func:`clone_pph` 相同语义的别名（读 → 写闭环入口）。"""
    clone_pph(src, dst, member_overrides)
