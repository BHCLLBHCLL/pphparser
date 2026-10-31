# pphparser

Cradle scFLOW 项目文件（`.pph`）的读写解析。`.pph` 是 ZIP 容器，成员包括 `main.xml` / `main.xenv` / `main.prp` / `main.js`、`main.sctsnapshot`（CADThru 记录流、LZMS、PKBody3），以及网格组的 `.gph` / `.oct` / `.mdl`。

```bash
pip install -r requirements.txt
python pph_parser.py tests/box.pph
python pph_parser.py tests/box.pph --extract out
python -m unittest tests.test_pph -v
```

| 模块 | 作用 |
|------|------|
| `pph_parser.py` | ZIP 容器、成员分类、摘要 |
| `pphxml.py` | `main.xml` 方言、`main.prp`、`main.xenv`、`main.js` |
| `pphwriter.py` | 写回 ZIP、LZMS 压缩、PKBody3 加密 |
| `sctsnapshot.py` | 快照记录流、LZMS 解压、PKBody3 |
| `blowfish_le.py` | PKBody3 的 Blowfish 小端 ECB |
| `parasolid.py` | PKBody3 明文里的 Parasolid 传输流 |
| `crdlfld.py` | GPH / OCT / MDL 共用的 CRDL-FLD 层 |
| `gphstats.py` | `.gph` 体网格 |
| `oct.py` | `.oct` 八叉树 |
| `mdl.py` | `.mdl` 面片几何 |

样例 `tests/box.pph` 是当前工程；`tests/box/` 是与写端逐字节对拍过的宿主成员。LZMS 压缩需要 Windows `cabinet.dll`，解压在 Windows 上走同一接口。
