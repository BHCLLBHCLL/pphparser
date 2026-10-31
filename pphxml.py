#!/usr/bin/env python3
"""PPH 中 XML / 文本成员的解析。

* ``main.xml`` — scFLOWpre 项目定义。**注意**：scFLOW 使用带索引标签的
  XML 方言（``<SECTITEM[0]>``），标准 XML 解析器无法接受；本模块先做
  标签名净化（``TAG[N]`` → ``TAG__IDXN``，索引记录在 ``_index`` 属性）。
* ``main.prp`` — 材料/物性数据库（标准 XML：group/entry 层次）。
* ``main.xenv`` — 环境/单位/容差设置（标准 XML，UTF-8 BOM）。
* ``main.js`` — 用户子程序脚本（JavaScript，``//@FormattedScript`` 段）。

观测到的索引标签家族：``SECTITEM[N]``、``PRISMITEM[N]``、``SMOOTHITEM[N]``
（净化后 ``TAG__IDXN``，``serialize_main_xml`` 可还原）。

快照 ``LENGTHVWU``/``DPOINTU`` 中 ``unit_type`` 码到 xenv 单位的解析见
``UNIT_TYPE_TO_XENV_KEY`` / ``resolve_snapshot_unit``。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

_INDEXED_TAG = re.compile(r"<(/?)([A-Za-z_][\w.]*)\[(\d+)\](>|/>)")
_DIGIT_TAG = re.compile(r"<(/?)(\d[\w.]*)(>|/>)")


def sanitize_scflow_xml(text: str) -> str:
    """把 scFLOW 的 ``<TAG[N]>`` 方言转换为合法 XML。

    ``<SECTITEM[0]>`` → ``<SECTITEM__IDX0>``；调用方可用
    ``restore_index(tag)`` 取回 ``(原名, 索引)``。闭合部分（``>`` 或 ``/>``）
    随标签整体替换，避免向元素文本注入多余 ``>``。

    电池等条件会写出以数字开头的标签（``<1D_spatial_div_...>``），
    标准 XML 不允许，这里改写为 ``<_D1D_spatial_div_...>``。
    """
    text = _INDEXED_TAG.sub(r"<\g<1>\g<2>__IDX\g<3>\g<4>", text)
    return _DIGIT_TAG.sub(r"<\1_D\2\3", text)


def restore_index(tag: str) -> tuple[str, Optional[int]]:
    """把净化后的标签名还原为 ``(原名, 索引或 None)``。"""
    m = re.match(r"^(.*?)__IDX(\d+)$", tag)
    if m:
        name, idx = m.group(1), int(m.group(2))
    else:
        name, idx = tag, None
    if name.startswith("_D") and len(name) > 2 and name[2].isdigit():
        name = name[2:]
    return name, idx


_IDX_SAFE_TAG = re.compile(r"^([A-Za-z_][\w.]*)__IDX(\d+)$")


def serialize_main_xml(root: ET.Element) -> str:
    """把净化后的 ElementTree 写回 scFLOW 方言。

    ``TAG__IDXN`` → ``TAG[N]``，与 :func:`sanitize_scflow_xml` 互逆，
    支撑修改后的 main.xml 写回 .pph（见 pphwriter.py）。
    """
    text = ET.tostring(root, encoding="unicode")
    text = re.sub(r"<(/?)([A-Za-z_][\w.]*)__IDX(\d+)(/?)([^>]*>)",
                  r"<\1\2[\3]\4\5", text)
    text = re.sub(r"<(/?)_D(\d[\w.]*)(/?)([^>]*>)", r"<\1\2\3\4", text)
    return text


# 快照 VWU 标签 → xenv 单位键（量纲感知）。unit_type 码实测恒为 1（SI）；
# 完整「码值→单位系统」枚举需多单位制样例（与 units.VWU_TAG_TO_XENV_KEY 一致）。
VWU_TAG_TO_XENV_KEY: dict[str, str] = {
    "LENGTHVWU": "MODEL_LENGTH_UNIT",
    "ANGLEVWU": "DEFAULT_ANGLE_UNIT",
    "AREAVWU": "DEFAULT_AREA_UNIT",
    "DENSITYVWU": "DEFAULT_DENSITY_UNIT",
    "ENERGYVWU": "DEFAULT_ENERGY_UNIT",
    "FORCEVWU": "DEFAULT_FORCE_UNIT",
    "TIMEVWU": "DEFAULT_TIME_UNIT",
    "VOLUMEVWU": "DEFAULT_VOLUME_UNIT",
    "DPOINTU": "DEFAULT_COORDX_UNIT",
}

UNIT_TYPE_TO_XENV_KEY: dict[int, str] = {1: "MODEL_LENGTH_UNIT"}


def resolve_snapshot_unit(unit_type: int,
                          xenv: "XenvSettings",
                          tag: str = "LENGTHVWU") -> Optional[str]:
    """把快照 ``LENGTHVWU``/``DPOINTU`` 等 VWU 记录的 ``unit_type`` 解析为
    xenv 单位串；``tag`` 决定量纲（取哪个 DEFAULT_*_UNIT 键）。"""
    if unit_type != 1:
        return None
    key = VWU_TAG_TO_XENV_KEY.get(tag, "MODEL_LENGTH_UNIT")
    return xenv.get("UNIT", key)


@dataclass
class XenvSettings:
    """main.xenv：Section/Key 层次设置。"""

    sections: dict[str, dict[str, str]] = field(default_factory=dict)

    def get(self, section: str, key: str, default: Optional[str] = None):
        return self.sections.get(section, {}).get(key, default)


def parse_xenv(data: bytes) -> XenvSettings:
    """解析 main.xenv（UTF-8 BOM 的 Section/Key XML）。"""
    text = data.decode("utf-8-sig")
    root = ET.fromstring(text)
    out = XenvSettings()
    for sec in root.iter("Section"):
        name = sec.get("name", "")
        for key in sec.iter("Key"):
            kname = key.get("name", "")
            # 去掉注释节点文本
            val = (key.text or "").strip()
            out.sections.setdefault(name, {})[kname] = val
    return out


def serialize_xenv(xenv: XenvSettings) -> bytes:
    """将 :class:`XenvSettings` 写回 main.xenv（UTF-8 BOM + CRLF）。

    注释节点不会保留；值与 Section/Key 层次与读端一致，可供 Save As 覆盖。
    """
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<Data type="env">',
    ]
    for sec_name, keys in xenv.sections.items():
        lines.append(f'    <Section name="{_xml_attr(sec_name)}">')
        for kname, val in keys.items():
            lines.append(f'        <Key name="{_xml_attr(kname)}">')
            lines.append(f"            {_xml_text(val)}")
            lines.append("        </Key>")
        lines.append("    </Section>")
    lines.append("</Data>")
    lines.append("")
    body = "\r\n".join(lines)
    return ("\ufeff" + body).encode("utf-8")


def _xml_attr(s: str) -> str:
    return (s.replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def _xml_text(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def set_xenv_value(xenv: XenvSettings, section: str, key: str, value: str) -> None:
    """设置/新增 Section.Key。"""
    xenv.sections.setdefault(section, {})[key] = value


#: OCT_MESH.VOXEL_OCT_REFINE_TYPE 的宿主编码（R30 单变量逐档实测，审计 §44.1）：
#: API setter 取**字符串枚举**（大小写敏感、数值档一律返回 False），
#: 而 main.xenv 落的是**整数码**。`octree` 手册未列，是 getter 读回发现的宿主默认值。
VOXEL_OCT_REFINE_TYPES: dict[str, str] = {
    "speed": "1",   # Speed-weighted
    "shape": "2",   # Model shape-weighted
    "octree": "3",  # 宿主默认（手册未列）
}
VOXEL_OCT_REFINE_CODES: dict[str, str] = {
    v: k for k, v in VOXEL_OCT_REFINE_TYPES.items()}


def voxel_oct_refine_code(name: Optional[str]) -> Optional[str]:
    """枚举名 → xenv 码（未知名返回 None；调用方据此决定是否落盘）。"""
    return VOXEL_OCT_REFINE_TYPES.get((name or "").strip().lower())


def voxel_oct_refine_name(code: Optional[str]) -> Optional[str]:
    """xenv 码 → 枚举名（未知码返回 None）。"""
    return VOXEL_OCT_REFINE_CODES.get((code or "").strip())


@dataclass
class PrpDatabase:
    """main.prp：材料物性库。"""

    version: str = ""
    date: str = ""
    groups: list[ET.Element] = field(default_factory=list)

    def group_names(self) -> list[str]:
        names = []
        for g in self.groups:
            k = g.findtext("key")
            names.append(k if k is not None else "")
        return names

    def entries(self, group: ET.Element) -> list[ET.Element]:
        return group.findall("entry")

    @staticmethod
    def entry_key(entry: ET.Element) -> str:
        return entry.findtext("key") or ""

    @staticmethod
    def entry_properties(entry: ET.Element) -> dict[str, str]:
        """提取 entry 下的简单 ``<tag>值</tag>`` 属性（跳过嵌套容器）。"""
        props: dict[str, str] = {}
        for ch in entry:
            if ch.tag in ("key", "name"):
                continue
            if len(ch) == 0:
                val = (ch.text or "").strip()
                if val:
                    props[ch.tag] = val
        return props


def parse_prp(data: bytes) -> PrpDatabase:
    """解析 main.prp 材料物性库。"""
    root = ET.fromstring(data.decode("utf-8"))
    out = PrpDatabase(version=root.get("version", ""), date=root.get("date", ""))
    out.groups = root.findall("group")
    return out


@dataclass
class MainXml:
    """main.xml：scFLOWpre 项目定义（已净化的 ElementTree）。"""

    root: ET.Element

    @property
    def version(self) -> str:
        return self.root.findtext("version", "")

    @property
    def project_name(self) -> str:
        return self.root.findtext("project/name", "")

    def section(self, name: str) -> Optional[ET.Element]:
        return self.root.find(name)

    def conditions(self) -> list[ET.Element]:
        cond = self.root.find("conditions")
        if cond is None:
            return []
        return cond.findall("condition")

    def all_conditions(
            self, known_types: Optional[set[str]] = None
    ) -> list[tuple[ET.Element, str]]:
        """深度扫描 ``<conditions>`` 子树全部条件实体（P7-1 扩源）。

        覆盖三类形态（直接子级 ``condition`` 之外的嵌套条件在
        ``conditions()`` 中被遗漏，实测 laptop/box 样本漏 8+ 类型）：

        1. 任意深度的 ``<condition>`` 元素（type 非空直接用；
           type 为空时按**父容器标签**推断 ``Cond<CamelCase(tag)>``，
           仅当推断名在 ``known_types`` 目录中才收录——防误报）；
        2. 带 ``<type>CondXxx</type>`` 直接子元素的**条件形容器**
           （如 ``particle_dem/symmetrical_particle_boundary``，标签非
           ``condition``；仅当 type 名在 ``known_types`` 目录中才收录，
           过滤掉 ``<velocity_x><type>VELX`` 一类值槽假阳性）。

        返回 ``[(元素, 类型名)]``；``known_types`` 缺省时仅收录带
        非空 type 的 ``condition`` 元素（无目录交叉核对能力）。
        """
        cond_root = self.root.find("conditions")
        if cond_root is None:
            return []
        known = known_types or set()

        def _infer_from_tag(tag: str) -> Optional[str]:
            """`sted_info` → `CondStedInfo`（复数 s 回退单数）。"""
            if not known:
                return None
            camel = "".join(p.capitalize() for p in tag.split("_"))
            for cand in (f"Cond{camel}", f"Cond{camel[:-1]}"):
                if cand in known:
                    return cand
            return None

        out: list[tuple[ET.Element, str]] = []
        for el in cond_root.iter():
            tname = (el.findtext("type") or "").strip()
            if el.tag == "condition":
                if tname:
                    out.append((el, tname))
                    continue
                parent = self._parent_of(el)
                if parent is not None:
                    inferred = _infer_from_tag(parent.tag)
                    if inferred:
                        out.append((el, inferred))
                continue
            # 条件形容器：type 子元素为目录中的 Cond* 名或其已登记
            # 短名别名（如 ALECancel/SymmetricalBoundary——实测样本中
            # 以短名出现的家族；别名归一由调用方 aliases.get 完成）
            if tname.startswith("Cond") and tname in known:
                out.append((el, tname))
            elif ("Cond" + tname) in known or tname in known:
                out.append((el, tname))
        return out

    def _parent_of(self, target: ET.Element) -> Optional[ET.Element]:
        cond_root = self.root.find("conditions")
        for parent in cond_root.iter():
            for child in parent:
                if child is target:
                    return parent
        return None

    @staticmethod
    def condition_summary(cond: ET.Element) -> dict:
        out = {"type": cond.findtext("type", ""), "name": cond.findtext("name", "")}
        regions = cond.find("regions")
        if regions is not None:
            out["regions"] = [(restore_index(r.tag)[0], (r.text or "").strip())
                              for r in regions]
        return out


def parse_main_xml(data: bytes) -> MainXml:
    """解析 main.xml（自动净化索引标签方言）。"""
    text = data.decode("utf-8")
    root = ET.fromstring(sanitize_scflow_xml(text))
    return MainXml(root)


@dataclass
class JsScript:
    """main.js：用户子程序脚本。"""

    source: str

    def functions(self) -> list[str]:
        """脚本中定义的函数名列表。"""
        return re.findall(r"^\s*function\s+([A-Za-z_]\w*)\s*\(",
                          self.source, re.MULTILINE)

    def has_user_code(self) -> bool:
        """是否存在用户填入的实现（非空函数体）。"""
        for m in re.finditer(r"function\s+\w+\s*\([^)]*\)\s*\{(.*?)\}",
                             self.source, re.DOTALL):
            if m.group(1).strip():
                return True
        return False


def parse_main_js(data: bytes) -> JsScript:
    return JsScript(data.decode("utf-8", errors="replace"))
