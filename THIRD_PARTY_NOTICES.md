# 第三方来源与许可

- `vendor/HAST-SEC/`：来自 sola1039/HAST-SEC，固定版本及文件列表见 `UPSTREAM.json`。仅收录运行模块依赖闭包，未修改源码。保留原 LICENSE、第三方声明与许可文本。适用“非商业学习与研究许可证 0.1”；不能据此宣称本项目整体允许商用。可选资源由安装脚本下载，资源来源条款独立适用。
- `server/supermimic_policy.py`：按 SuperMimic 社区复现网络实现可移植推理，原实现版权为 ETH Zurich、Chong Zhang；原 MIT 文本保存在 `licenses/SuperMimic-reproduction-MIT.txt`。不分发预训练权重或原始动作。下载器固定的权重与测试资产声明见 `docs/SUPERMIMIC.md`。
- npm 依赖：Three.js（MIT）、@pixiv/three-vrm（MIT）；构建依赖及完整版本见 `package-lock.json`。上游许可随安装包提供。
- Python 依赖：MCP SDK、NumPy、SciPy 与可选感知/物理运行包保留各自许可；见 `server/requirements-*.txt`。YOLO/Ultralytics 及各感知模型的许可证由上游独立提供，本项目不重新授权。
- `server/assets/core-rest.json` 与 `tests/fixtures/core-rest.json`：ARDY CoreSkeleton27 的中立骨架关节坐标，来源 nv-tlabs/ardy，保留 `licenses/ARDY-Apache-2.0.txt`。
- ARDY：可选上游 nv-tlabs/ardy，代码和权重遵循各自原始条款；本项目只包含本地适配代码。
- AMASS、SMPL、LAFAN1、VRM 角色及其他第三方数据：不包含在源码发布中。用户自行获取合法授权后配置，派生动作并不自动具有再分发权。

主项目原创代码尚未指定额外授权许可。第三方作者署名、来源及许可不属于应清除的开发者个人信息，不应删除。
