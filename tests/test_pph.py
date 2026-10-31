"""box 样例上的 PPH 导入、解析与写回。"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gphstats  # noqa: E402
import mdl  # noqa: E402
import oct as octmod  # noqa: E402
import parasolid  # noqa: E402
import pph_parser  # noqa: E402
import pphwriter  # noqa: E402
import pphxml  # noqa: E402
import sctsnapshot  # noqa: E402

BOX_PPH = ROOT / "tests" / "box.pph"
GOLDEN = ROOT / "tests" / "box"
GPH = GOLDEN / "meshinggroup1.gph"
MDL = GOLDEN / "meshinggroup1_part.mdl"
OCT = GOLDEN / "meshinggroup1.oct"
SNAP = GOLDEN / "main.sctsnapshot"


def _date_of(data: bytes) -> int:
    name = b"Date" + b" " * 28
    i = data.find(name)
    if i < 4:
        return 20260812
    body = i + 36
    return struct.unpack_from(">i", data, body + 16 + 8)[0]


def _load_snap(path: Path) -> sctsnapshot.SctSnapshot:
    if path.suffix.lower() == ".pph":
        with zipfile.ZipFile(path) as z:
            raw = z.read("main.sctsnapshot")
        fd, tmp = tempfile.mkstemp(suffix=".sctsnapshot")
        os.close(fd)
        Path(tmp).write_bytes(raw)
        try:
            return sctsnapshot.SctSnapshot.load(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)
    return sctsnapshot.SctSnapshot.load(str(path))


class TestArchive(unittest.TestCase):
    def test_members(self):
        arch = pph_parser.PphArchive.open(str(BOX_PPH))
        self.assertEqual(
            [(m.name, m.role) for m in arch.members],
            [
                ("main.js", pph_parser.ROLE_SCRIPT),
                ("main.prp", pph_parser.ROLE_PRP),
                ("main.sctsnapshot", pph_parser.ROLE_SNAPSHOT),
                ("main.xenv", pph_parser.ROLE_XENV),
                ("main.xml", pph_parser.ROLE_PROJECT_XML),
                ("meshinggroup1.gph", pph_parser.ROLE_GPH),
                ("meshinggroup1.oct", pph_parser.ROLE_OCT),
                ("meshinggroup1_part.mdl", pph_parser.ROLE_MDL_PART),
                ("meshinggroup1_ridge.mdl", pph_parser.ROLE_MDL_RIDGE),
            ],
        )

    def test_text_members(self):
        arch = pph_parser.PphArchive.open(str(BOX_PPH))
        js = pphxml.parse_main_js(arch.read_member("main.js"))
        self.assertIn("usr_input", js.functions())
        prp = pphxml.parse_prp(arch.read_member("main.prp"))
        self.assertGreater(len(prp.groups), 0)
        xenv = pphxml.parse_xenv(arch.read_member("main.xenv"))
        self.assertEqual(xenv.get("TYPE", "PROJECT_TYPE"), "scflow")
        mx = pphxml.parse_main_xml(arch.read_member("main.xml"))
        self.assertTrue(mx.version)
        self.assertGreater(len(mx.conditions()), 0)

    def test_xml_dialect_roundtrip(self):
        arch = pph_parser.PphArchive.open(str(BOX_PPH))
        raw = arch.read_member("main.xml").decode("utf-8")
        root = pphxml.ET.fromstring(pphxml.sanitize_scflow_xml(raw))
        back = pphxml.serialize_main_xml(root)
        root2 = pphxml.ET.fromstring(pphxml.sanitize_scflow_xml(back))
        self.assertEqual([c.tag for c in root2], [c.tag for c in root])

    def test_xenv_roundtrip_keys(self):
        arch = pph_parser.PphArchive.open(str(BOX_PPH))
        xenv = pphxml.parse_xenv(arch.read_member("main.xenv"))
        again = pphxml.parse_xenv(pphxml.serialize_xenv(xenv))
        self.assertEqual(again.sections, xenv.sections)


class TestSnapshot(unittest.TestCase):
    def test_golden_snapshot_serialize_identity(self):
        raw = SNAP.read_bytes()
        snap = sctsnapshot.SctSnapshot.load(str(SNAP))
        self.assertEqual(snap.serialize(raw), raw)

    def test_pph_snapshot_serialize_identity(self):
        with zipfile.ZipFile(BOX_PPH) as z:
            raw = z.read("main.sctsnapshot")
        fd, tmp = tempfile.mkstemp(suffix=".sctsnapshot")
        os.close(fd)
        path = Path(tmp)
        path.write_bytes(raw)
        try:
            snap = sctsnapshot.SctSnapshot.load(str(path))
            self.assertEqual(snap.skipped_bytes, 0)
            self.assertEqual(snap.serialize(raw), raw)
        finally:
            path.unlink(missing_ok=True)

    def test_pkbody3_and_parasolid(self):
        snap = _load_snap(BOX_PPH)
        body = snap.bodies()[0]["zip"].decompress_body()
        plain = body.decrypt()
        parsed = parasolid.parse_transmit(plain)
        self.assertEqual(parsed.version, 3701153)
        self.assertEqual(parsed.schema, "SCH_3701153_37102_13006")
        wrapped = pphwriter.encrypt_pkbody3(plain)
        self.assertEqual(wrapped, snap.bodies()[0]["zip"].decompress())

    @unittest.skipUnless(os.name == "nt", "LZMS 压缩需要 Windows cabinet.dll")
    def test_lzms_and_body_rebuild(self):
        snap = _load_snap(BOX_PPH)
        zipblob = snap.bodies()[0]["zip"]
        plain = zipblob.decompress_body().decrypt()
        stream = pphwriter.lzms_compress(pphwriter.encrypt_pkbody3(plain))
        blob = sctsnapshot.ZipBlob.parse(stream)
        self.assertEqual(blob.decompress_body().decrypt(), plain)
        self.assertEqual(blob.decompress(), zipblob.decompress())


class TestRewrite(unittest.TestCase):
    def test_clone_and_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, "clone.pph")
            pphwriter.clone_pph(str(BOX_PPH), dst)
            src = pph_parser.PphArchive.open(str(BOX_PPH))
            out = pph_parser.PphArchive.open(dst)
            for m in src.members:
                self.assertEqual(out.read_member(m.name), src.read_member(m.name))
            edited = os.path.join(tmp, "edited.pph")
            pphwriter.rewrite_pph(
                str(BOX_PPH), edited, {"main.xml": b"<scFLOWpre></scFLOWpre>"})
            arch = pph_parser.PphArchive.open(edited)
            self.assertEqual(arch.read_member("main.xml"), b"<scFLOWpre></scFLOWpre>")
            self.assertEqual(
                arch.read_member("main.js"), src.read_member("main.js"))


class TestMeshMembers(unittest.TestCase):
    def test_golden_gph_counts_and_bytes(self):
        raw = GPH.read_bytes()
        with gphstats.open_buffer(str(GPH)) as data:
            summary = gphstats.summarize(data)
            mesh = gphstats.parse_mesh(data)
            cvol = gphstats.cvol_ids(data)
            info = gphstats.element_info(data)
        self.assertEqual(summary["links"]["n_faces"], 3168)
        self.assertEqual(summary["links"]["n_cells"], 944)
        self.assertEqual(summary["n_vertices"], 1305)
        fo = mesh["face_offsets"]
        faces = [mesh["conn"][fo[i]:fo[i + 1]] for i in range(len(fo) - 1)]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "rt.gph"
            gphstats.write_gph_volume(
                out, mesh["vertices"], faces, mesh["owner"], mesh["neigh"],
                app="SCTpre", date=_date_of(raw), cvol=cvol,
                surface_regions=[
                    (n, ids) for n, ids in
                    gphstats.surface_region_face_ids(raw).items()
                ] or None,
                volume_regions=gphstats.volume_region_names(raw) or None,
                parts=gphstats.parts_summary(raw, cvol) or None,
                assemblies=gphstats.assemblies_xml(raw),
                element_info=(info[1] if info else None))
            self.assertEqual(out.read_bytes(), raw)

    def test_golden_mdl_and_oct_bytes(self):
        raw = MDL.read_bytes()
        model = mdl.parse_mdl(MDL)
        faces = [model.face_nodes(i) for i in range(model.n_faces)]
        b1, b2 = model.csid_sides
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "rt.mdl"
            mdl.write_mdl(
                out, model.xyz, faces, csid=(b1, b2), frid=model.frid,
                edge_state=model.edge_state, node_state=model.node_state,
                date=_date_of(raw),
                surface_regions=[(r.name, r.index)
                                 for r in (model.surface_regions or [])] or None,
                closed_volumes=model.closed_volumes,
                volume_regions=model.volume_regions)
            self.assertEqual(out.read_bytes(), raw)

        raw = OCT.read_bytes()
        tree = octmod.parse_oct(OCT)
        self.assertEqual(tree.n_octants, 1 + 8 * tree.n_internal)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "rt.oct"
            octmod.write_oct(
                out, tree.root_min, tree.root_max,
                refinement=tree.refinement, block_id=tree.block_id,
                unit=tree.unit or "m", date=_date_of(raw))
            self.assertEqual(out.read_bytes(), raw)

    def test_pph_mesh_members_parse(self):
        arch = pph_parser.PphArchive.open(str(BOX_PPH))
        with tempfile.TemporaryDirectory() as td:
            gph_path = Path(td) / "m.gph"
            gph_path.write_bytes(arch.read_member("meshinggroup1.gph"))
            with gphstats.open_buffer(str(gph_path)) as data:
                links = gphstats.links_summary(data)
            self.assertGreater(links["n_faces"], 0)
            self.assertGreater(links["n_cells"], 0)

            oct_path = Path(td) / "m.oct"
            oct_path.write_bytes(arch.read_member("meshinggroup1.oct"))
            tree = octmod.parse_oct(oct_path)
            self.assertEqual(tree.n_octants, 1 + 8 * int(tree.n_internal))

            for name in ("meshinggroup1_part.mdl", "meshinggroup1_ridge.mdl"):
                path = Path(td) / name
                path.write_bytes(arch.read_member(name))
                surface = mdl.parse_mdl(path, load_arrays=False)
                self.assertGreater(surface.n_faces, 0)
                self.assertGreater(surface.n_vertices, 0)


if __name__ == "__main__":
    unittest.main()
