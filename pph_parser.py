#!/usr/bin/env python3
"""PPH（Cradle scFLOW 项目文件）解析器。

PPH 是一个标准 ZIP 归档（deflate，小文件实为 stored 块），按固定角色
组织成员文件：

.. code-block:: text

    main.js            用户子程序脚本（JavaScript 模板/实现）
    main.prp           材料物性数据库（XML：property/group/entry）
    main.sctsnapshot   当前状态快照（CADThru 小端记录流，见 sctsnapshot.py）
    main.xenv          环境/单位/容差（XML：Section/Key）
    main.xml           项目定义（scFLOW XML 方言：<TAG[N]> 索引标签）
    <group>.gph        体网格（CRDL-FLD 大端，见 gphstats.py）
    <group>.oct        八叉树（CRDL-FLD，见 oct.py）
    <group>_part.mdl   显示/零件面片几何（CRDL-FLD，见 mdl.py）
    <group>_ridge.mdl  完整 ridge/细节面片几何（CRDL-FLD，见 mdl.py）

用法：

.. code-block:: text

    python pph_parser.py 项目.pph                 # 全部成员摘要
    python pph_parser.py 项目.pph --extract 目录  # 解包
    python pph_parser.py 项目.pph --snapshot      # 打印 sctsnapshot 记录树
    python pph_parser.py 项目.pph --octree        # 八叉树叶子统计
    python pph_parser.py 项目.pph --xml           # 打印 main.xml 顶层结构
"""

from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 成员角色
ROLE_SCRIPT = "script"
ROLE_PRP = "property_db"
ROLE_SNAPSHOT = "snapshot"
ROLE_XENV = "environment"
ROLE_PROJECT_XML = "project_xml"
ROLE_GPH = "volume_mesh_gph"
ROLE_OCT = "octree"
ROLE_MDL_PART = "surface_part_mdl"
ROLE_MDL_RIDGE = "surface_ridge_mdl"
ROLE_UNKNOWN = "unknown"


def classify_member(name: str) -> tuple[str, str]:
    """``(角色, 说明)`` 按成员文件名分类。"""
    base = name.lower()
    if base == "main.js":
        return ROLE_SCRIPT, "用户子程序脚本"
    if base == "main.prp":
        return ROLE_PRP, "材料物性数据库"
    if base == "main.sctsnapshot":
        return ROLE_SNAPSHOT, "状态快照（CADThru 记录流）"
    if base == "main.xenv":
        return ROLE_XENV, "环境/单位设置"
    if base == "main.xml":
        return ROLE_PROJECT_XML, "项目定义"
    if base.endswith(".gph"):
        return ROLE_GPH, "体网格"
    if base.endswith(".oct"):
        return ROLE_OCT, "八叉树"
    if base.endswith("_part.mdl"):
        return ROLE_MDL_PART, "零件面片几何"
    if base.endswith("_ridge.mdl"):
        return ROLE_MDL_RIDGE, "ridge 细节面片几何"
    if base.endswith(".mdl"):
        return ROLE_MDL_PART, "面片几何"
    return ROLE_UNKNOWN, "未知成员"


@dataclass
class PphMember:
    name: str
    role: str
    description: str
    size: int
    compress_size: int


@dataclass
class PphArchive:
    """PPH 归档（ZIP 容器）。"""

    filepath: str
    members: list[PphMember] = field(default_factory=list)

    @classmethod
    def open(cls, filepath: str) -> "PphArchive":
        if not zipfile.is_zipfile(filepath):
            raise ValueError(f"{filepath}: 不是 ZIP/PPH 归档")
        arch = cls(filepath)
        with zipfile.ZipFile(filepath) as z:
            for info in z.infolist():
                role, desc = classify_member(info.filename)
                arch.members.append(PphMember(
                    info.filename, role, desc, info.file_size, info.compress_size))
        return arch

    def read_member(self, name: str) -> bytes:
        with zipfile.ZipFile(self.filepath) as z:
            return z.read(name)

    def extract(self, out_dir: str) -> list[str]:
        with zipfile.ZipFile(self.filepath) as z:
            z.extractall(out_dir)
        return [str(Path(out_dir) / m.name) for m in self.members]

    def by_role(self, role: str) -> list[PphMember]:
        return [m for m in self.members if m.role == role]

    def surface_mdl_members(self) -> list[PphMember]:
        """显示用面片：优先 ``*_part.mdl``，否则 ``*_ridge.mdl``。

        官方 Org 教程 PPH 常见布局是 ridge + GPH、没有 part；自研网格 /
        Register Region / Select 不能因此判空。
        """
        parts = self.by_role(ROLE_MDL_PART)
        if parts:
            return parts
        return self.by_role(ROLE_MDL_RIDGE)


def group_surface_path(info: Optional[dict]) -> Optional[str]:
    """网格组显示面路径：``part`` 优先，否则 ``ridge``。"""
    paths = (info or {}).get("paths") or {}
    return paths.get("part") or paths.get("ridge")


# ─────────────────────────────────────────────────────────────────────────────
# 摘要报告
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_size(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n} B"


def summarize_text_members(arch: PphArchive, out) -> None:
    """解析并摘要 main.js / main.prp / main.xenv / main.xml。"""
    import pphxml

    for m in arch.members:
        if m.role == ROLE_SCRIPT:
            js = pphxml.parse_main_js(arch.read_member(m.name))
            funcs = js.functions()
            out.append(f"\n[main.js] 用户脚本：{len(funcs)} 个函数，"
                       f"{'含用户实现' if js.has_user_code() else '全部为模板空函数'}")
            out.append(f"  函数: {', '.join(funcs[:12])}"
                       f"{' ...' if len(funcs) > 12 else ''}")
        elif m.role == ROLE_PRP:
            prp = pphxml.parse_prp(arch.read_member(m.name))
            out.append(f"\n[main.prp] 物性库 version={prp.version} groups={len(prp.groups)}")
            for g in prp.groups[:8]:
                entries = prp.entries(g)
                out.append(f"  组 {prp.group_names() and (g.findtext('key') or '')!r}: "
                           f"{len(entries)} 条目")
        elif m.role == ROLE_XENV:
            xenv = pphxml.parse_xenv(arch.read_member(m.name))
            out.append(f"\n[main.xenv] 环境设置：{len(xenv.sections)} 个 Section")
            for sec in ("TYPE", "UNIT", "PROJ_SETTING_FILE", "TOLERANCE", "MESH"):
                if sec in xenv.sections:
                    keys = xenv.sections[sec]
                    sample = ", ".join(f"{k}={v}" for k, v in
                                       list(keys.items())[:4])
                    out.append(f"  [{sec}] {len(keys)} 键: {sample} ...")
        elif m.role == ROLE_PROJECT_XML:
            mx = pphxml.parse_main_xml(arch.read_member(m.name))
            out.append(f"\n[main.xml] 项目定义 version={mx.version} "
                       f"name={mx.project_name!r}")
            out.append("  顶层节: " + ", ".join(c.tag for c in mx.root))
            conds = mx.conditions()
            if conds:
                out.append(f"  边界/求解条件: {len(conds)} 项")
                for c in conds[:6]:
                    s = mx.condition_summary(c)
                    out.append(f"    - {s['name']} (type={s['type']})")


def summarize_snapshot(arch: PphArchive, out, full: bool = False) -> None:
    import sctsnapshot

    for m in arch.by_role(ROLE_SNAPSHOT):
        raw = arch.read_member(m.name)
        tmp = _TempFile(raw)
        snap = sctsnapshot.SctSnapshot.load(tmp.path)
        out.append(f"\n[{m.name}] 快照记录树：顶层 {len(snap.records)} 条记录，"
                   f"未对齐字节 {snap.skipped_bytes}")
        for r in snap.records:
            out.append("  " + r.text())
        bodies = snap.bodies()
        if bodies:
            out.append(f"  Parasolid 体: {len(bodies)} 个（LZMS 压缩）")
            for b in bodies:
                z = b["zip"]
                out.append(f"    PKBODY_T={b['pk_body']} "
                           f"解压后 {z.uncompressed_size} B / 压缩 {z.compressed_size} B")
            if sctsnapshot.lzms_available():
                try:
                    for b in snap.decompress_bodies():
                        pk = b["pkbody3"]
                        ck = pk.checksum
                        ck_s = (f"zero-pad-tail=0x{ck:08x}" if ck is not None
                                else "no-zero-pad-block")
                        pad_s = (f" pad={len(pk.pad)}B" if pk.pad else "")
                        try:
                            plain = pk.decrypt()
                            head = plain[:24]
                            dec_s = (f" decrypt={len(plain)}B"
                                     f" head={head!r}")
                        except Exception as exc:  # noqa: BLE001
                            dec_s = f" decrypt-fail={exc}"
                        out.append(
                            f"    → PKBody3 size={pk.logical_size} B "
                            f"data={len(pk.data)} B "
                            f"{ck_s}{pad_s}{dec_s}")
                        try:
                            import parasolid
                            ps = parasolid.parse_transmit(plain)
                            ps_s = (f" schema={ps.schema} "
                                    f"字段={len(ps.fields)} 实体="
                                    f"{ps.entities}")
                            out.append("      " + ps_s)
                        except Exception:  # noqa: BLE001 - 提取为尽力而为
                            pass
                except (OSError, ValueError) as exc:
                    out.append(f"    （LZMS 解压失败: {exc}）")
        for tag, label in (("ZIPOCTREE", "八叉树块"),
                           ("ZIPFACETINGRULES", "面片规则块")):
            blobs = snap.zip_blobs(tag)
            if not blobs:
                continue
            z = blobs[0]
            out.append(f"  {label} ({tag}): unc={z.uncompressed_size} "
                       f"comp={z.compressed_size}")
        if sctsnapshot.lzms_available():
            try:
                facet = snap.decompress_faceting_rules()
                if facet:
                    out.append(f"  ZIPFACETINGRULES 解压: {facet[0].text()}")
                oct_recs = snap.decompress_octree(max_depth=6)
                if oct_recs:
                    tags = [r.tag for r in oct_recs]
                    out.append(f"  ZIPOCTREE 解压顶层: {', '.join(tags)}")
                    crdl = _octree_bytearray(oct_recs)
                    if crdl:
                        out.append(f"  OCTREEBODY ≡ *.oct CRDL-FLD "
                                   f"({len(crdl):,} B)")
                    div = snap.octree_division()
                    if div is not None:
                        out.append(
                            f"  OCTREEDIVISION: {len(div):,} B "
                            f"(is-internal 位图, perm="
                            f"{sctsnapshot.SctSnapshot.OCTREE_DIVISION_CHILD_ORDER})")
                    reg_raw = snap._octree_bytearray("OCTREEREGION")
                    if reg_raw is not None:
                        out.append(
                            f"  OCTREEREGION: {len(reg_raw):,} B "
                            f"(后序 u8/octant + 尾零填充)")
            except (OSError, ValueError) as exc:
                out.append(f"  （LZMS 嵌套块解压失败: {exc}）")
        groups = snap.meshing_groups()
        if groups:
            out.append(f"  网格组参数 (BSGSEX): {len(groups)} 组")
            for g in groups:
                out.append(f"    - {g['name']} (parent={g['parent']})")
        if full:
            out.append("\n完整记录树:")
            out.append(snap.dump(max_depth=10))
        tmp.close()


def _octree_bytearray(records) -> Optional[bytes]:
    """从已解压的 ZIPOCTREE 记录树提取 OCTREEBODY 内 CRDL-FLD 字节。"""
    for r in records:
        stack = [r]
        while stack:
            cur = stack.pop()
            if cur.tag == "OCTREEBODY":
                for qb in cur.find_all("QUEUEBODY"):
                    for c in qb.children:
                        if (c.tag == "BYTEARRAY"
                                and isinstance(c.value, bytes)
                                and b"CRDL-FLD" in c.value[:16]):
                            return c.value
            stack.extend(cur.children)
    return None


class _TempFile:
    """把 bytes 落到临时文件供基于路径的解析器使用。"""

    def __init__(self, data: bytes, suffix: str = ""):
        import tempfile
        fd, self.path = tempfile.mkstemp(suffix=suffix)
        import os
        with os.fdopen(fd, "wb") as f:
            f.write(data)

    def close(self):
        import os
        try:
            os.unlink(self.path)
        except OSError:
            pass


def summarize_binary_members(arch: PphArchive, out, work_dir: Optional[str] = None,
                             octree: bool = False) -> None:
    """解析并摘要 gph/oct/mdl（从归档解到临时/工作目录）。"""
    import crdlfld
    import mdl as mdl_mod
    import oct as oct_mod

    need_extract = [m for m in arch.members
                    if m.role in (ROLE_GPH, ROLE_OCT, ROLE_MDL_PART, ROLE_MDL_RIDGE)]
    if not need_extract:
        return
    import tempfile
    tmp_ctx = None
    if work_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory()
        work_dir = tmp_ctx.name
    try:
        for m in need_extract:
            target = Path(work_dir) / m.name
            if not target.exists() or target.stat().st_size != m.size:
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, "wb") as f:
                    f.write(arch.read_member(m.name))
            if m.role == ROLE_GPH:
                with crdlfld.CrdlFldFile.load(str(target)) as cf:
                    out.append(f"\n[{m.name}] 体网格 CRDL-FLD 节: "
                               f"{len(cf.sections)} 节")
                    meta = cf.metadata()
                    keep = {k: v for k, v in meta.items() if k != "header_dims"}
                    out.append(f"  元数据: {keep}")
                    for s in cf.sections:
                        if s.name.startswith("LS_") or s.name.startswith("Element"):
                            out.append(f"  节 {s.name}: {_fmt_size(s.end - s.start)}")
                deep = _try_gph_deep(str(target))
                if deep:
                    out.extend("  " + line for line in deep)
            elif m.role == ROLE_OCT:
                model = oct_mod.parse_oct(str(target))
                out.append(f"\n[{m.name}] 八叉树: 节点 {model.n_octants:,} "
                           f"(内部 {model.n_internal:,} / 叶子 {model.n_leaves:,}) "
                           f"单位 {model.unit!r}")
                mn, mx = model.root_min, model.root_max
                out.append(f"  根包围盒: ({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}) .. "
                           f"({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f})")
                if model.block_id.size:
                    import numpy as np
                    uniq = np.unique(model.block_id)
                    out.append(f"  块 id: {uniq[:8].tolist()}"
                               f"{' ...' if uniq.size > 8 else ''}")
                if octree:
                    stats = model.leaf_stats()
                    out.append(f"  叶子深度直方图: {stats['depth_histogram']}")
            else:
                model = mdl_mod.parse_mdl(str(target), load_arrays=False)
                out.append(f"\n[{m.name}] 面片几何 ({m.description}): "
                           f"顶点 {model.n_vertices:,} / 面 {model.n_faces:,}")
                out.append(f"  闭体: {len(model.closed_volumes)} "
                           f"体区域: {model.volume_regions}")
                regs = ", ".join(f"{r.name}(idx={r.index})"
                                 for r in model.surface_regions)
                out.append(f"  面区域: {regs}")
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def _try_gph_deep(gph_path: str) -> Optional[list[str]]:
    """用 ``gphstats`` 统计 GPH 成员的面、单元、顶点和区域。"""
    import gphstats

    try:
        with gphstats.open_buffer(gph_path) as data:
            links = gphstats.links_summary(data)
            cvol = gphstats.cvol_ids(data)
            n_vertices, dialect = gphstats.nodes_vertex_count(data)
            surfs = gphstats.surface_regions_summary(data)
            parts = gphstats.parts_summary(data, cvol)
            vols = gphstats.string_list(data, "LS_VolumeRegions")
        out = []
        if links:
            out.append(
                f"网格: {links['n_faces']:,} 面 / {links['n_cells']:,} 单元 / "
                f"{n_vertices:,} 顶点 ({dialect})"
                + (" 多面体" if links["polyhedral"] else ""))
            out.append(f"边界面: {links['boundary_faces']:,} "
                       f"npe [{links['npe_min']}..{links['npe_max']}]")
        if parts:
            out.append("Parts: " + ", ".join(
                f"{n}(cvol={gphstats.format_part_cvol_spec(c)})"
                for n, c in parts))
        if vols:
            out.append(f"体区域: {vols}")
        if surfs:
            out.append("面区域: " + ", ".join(f"{n}({c:,})" for n, c in surfs))
        return out
    except Exception as exc:  # pragma: no cover - 防御
        return [f"(GPH 统计失败: {exc})"]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="解析 Cradle scFLOW 项目文件 (.pph)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("pph", nargs="?",
                    default=str(Path(__file__).resolve().parent / "tests" / "box.pph"),
                    help="pph 文件路径")
    ap.add_argument("--extract", metavar="DIR", help="解包到目录")
    ap.add_argument("--workdir", metavar="DIR",
                    help="二进制成员解包后使用的目录（默认系统临时目录，自动清理）")
    ap.add_argument("--snapshot", action="store_true", help="打印 sctsnapshot 完整记录树")
    ap.add_argument("--octree", action="store_true", help="统计八叉树叶子深度")
    ap.add_argument("--no-binary", action="store_true", help="跳过 gph/oct/mdl 解析")
    args = ap.parse_args(argv)

    arch = PphArchive.open(args.pph)
    out: list[str] = []
    out.append(f"PPH 归档: {args.pph}")
    out.append(f"成员 {len(arch.members)} 项（ZIP/deflate 容器）:")
    for m in arch.members:
        out.append(f"  {m.name:<28} {m.description:<22} "
                   f"{_fmt_size(m.size):>12} (压缩 {_fmt_size(m.compress_size)})")

    if args.extract:
        paths = arch.extract(args.extract)
        out.append(f"\n已解包 {len(paths)} 个文件到 {args.extract}")

    summarize_text_members(arch, out)
    summarize_snapshot(arch, out, full=args.snapshot)
    if not args.no_binary:
        summarize_binary_members(arch, out, work_dir=args.workdir, octree=args.octree)

    text = "\n".join(out)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
