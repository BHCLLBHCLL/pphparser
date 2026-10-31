# -*- coding: utf-8 -*-
"""Blowfish 小端变体 ECB 加解密（scFLOW ``PKBody3`` 私有流）。

逆向自 Cradle CFD 2025.2 ``SCTprime_Bx64.dll``：

- 标准 Blowfish（16 轮 Feistel、π P/S 表、标准密钥扩展，密钥字按
  big-endian 组装、循环取模密钥长）。
- **唯一差别**：8 字节分块按两个 **little-endian** u32 作为 (L, R)。
  （块函数 RVA 0x57B110 解密 / 0x57B550 加密均 ``dword ptr`` 直读。）
- ECB 模式，无 IV（循环 RVA 0x57B990 逐块原地解密）。
- 数据长度向上取整到 8（``ceil8``），解密后截回原长。

密钥为固定字符串 ``b"HowDareYouSaySuchAThing"``（23 字节），
DllMain 解混淆写入 ``.data`` RVA 0xD20DB8；文件态该处为 3 字节占位
``{e8 16 be 00}``。解密包装 RVA 0xFE9E0 校验 ``CADthru/PKBody3``
15 字节魔数后对整个 ``data`` 字段（size 字节）ECB 解密。

已用 ctypes 直接调用 DLL 内 BF_INIT(0x57BBB0) 做金标准对照：
展开后 18+1024 个 u32 与本实现逐字节一致。
"""

from __future__ import annotations

import struct

from blowfish_tables import TABLES

#: scFLOW 固定密钥（23 字节）
DEFAULT_KEY = b"HowDareYouSaySuchAThing"

_P0 = list(struct.unpack_from("<18I", TABLES, 0))
_S0 = [list(struct.unpack_from("<256I", TABLES, 0x48 + k * 0x400))
       for k in range(4)]
_MASK = 0xFFFFFFFF


def expand_key(key: bytes) -> tuple[list[int], list[list[int]]]:
    """标准 Blowfish 密钥扩展 → (P[18], S[4][256])。"""
    if not key:
        raise ValueError("空密钥")
    P = _P0[:]
    S = [s[:] for s in _S0]
    for i in range(18):
        w = 0
        for j in range(4):
            w = ((w << 8) | key[(4 * i + j) % len(key)]) & _MASK
        P[i] ^= w

    L = R = 0
    for i in range(0, 18, 2):
        L, R = _enc_block_u32(P, S, L, R)
        P[i], P[i + 1] = L, R
    for k in range(4):
        for i in range(0, 256, 2):
            L, R = _enc_block_u32(P, S, L, R)
            S[k][i] = L
            S[k][i + 1] = R
    return P, S


def _f(S: list[list[int]], x: int) -> int:
    return (((S[0][x >> 24] + S[1][(x >> 16) & 0xFF]) & _MASK
             ^ S[2][(x >> 8) & 0xFF]) + S[3][x & 0xFF]) & _MASK


def _enc_block_u32(P, S, L, R):
    for i in range(16):
        L ^= P[i]
        R ^= _f(S, L)
        L, R = R, L
    L, R = R, L
    R ^= P[16]
    L ^= P[17]
    return L, R


def _dec_block_u32(P, S, L, R):
    for i in range(17, 1, -1):
        L ^= P[i]
        R ^= _f(S, L)
        L, R = R, L
    L, R = R, L
    R ^= P[1]
    L ^= P[0]
    return L, R


def decrypt_ecb(data: bytes, key: bytes = DEFAULT_KEY) -> bytes:
    """LE-Blowfish ECB 解密；输入任意长度（内部 ceil8 补零），返回原长。"""
    P, S = expand_key(key)
    n = (len(data) + 7) // 8 * 8
    buf = data + b"\x00" * (n - len(data))
    out = bytearray(n)
    for off in range(0, n, 8):
        L, R = struct.unpack_from("<II", buf, off)
        L, R = _dec_block_u32(P, S, L, R)
        struct.pack_into("<II", out, off, L, R)
    return bytes(out[:len(data)])


def encrypt_ecb(data: bytes, key: bytes = DEFAULT_KEY) -> bytes:
    """LE-Blowfish ECB 加密（测试/对照用），长度处理同 ``decrypt_ecb``。"""
    P, S = expand_key(key)
    n = (len(data) + 7) // 8 * 8
    buf = data + b"\x00" * (n - len(data))
    out = bytearray(n)
    for off in range(0, n, 8):
        L, R = struct.unpack_from("<II", buf, off)
        L, R = _enc_block_u32(P, S, L, R)
        struct.pack_into("<II", out, off, L, R)
    return bytes(out[:len(data)])
