# 资源与可复现运行

仓库只发布源码、最小骨架定义和测试夹具。以下目录留在本地，不随 Git 发布：`models/`、`datasets/`、`external/`、`.local-data/`、`public/locomotion/`、`public/motion-table/`、`public/social-actions/` 和 `output/`。

## VRM

将拥有使用授权的模型放入 `models/avatars/`。也可在 `.env` 设置 `MOTION_VRM_DIR` 和可选 `MOTION_VRM_FILES`。空白名单表示列出目录下所有 VRM；存在 Wolferia 时设为默认。不会递归暴露整个磁盘目录，资源服务拒绝越界路径。

## 流明

`npm run setup:lumen` 下载 `vendor/HAST-SEC/UPSTREAM.json` 固定的版本，仅安装动作资源、配置和许可，不安装第三方 node_modules，不保留嵌套 Git。已有资源会保留，不自动覆盖。

构建使用 `vendor/HAST-SEC/` 中未修改的模块；运行数据默认从 `external/HAST-SEC/` 提供，可用 `MOTION_LUMEN_DIR` 覆盖。流明具有非商业限制，资源中的 CMU、BEAT2 等来源条款继续适用。关闭流明资源不会阻止核心服务启动，但对应手势不可执行。

## AMASS 动作库

自行获取 AMASS 及组成数据集所需授权，将目录放入 `datasets/amass/raw/`，或设置 `MOTION_AMASS_DIR`。不要公开上传下载的原始数据或未获授权的派生产物。

```sh
python3 scripts/run_component.py python scripts/build_locomotion.py
python3 scripts/run_component.py python scripts/amass_search.py
python3 scripts/run_component.py python scripts/collect_social_amass.py
npm run table:build
```

采集脚本要求其筛选模式对应的 KIT/ACCAD 等文件存在；这不是全数据集通用下载器。输出写入 `public/locomotion/`、`public/social-actions/`、`public/motion-table/`。清洗与连接质量门槛见 [动作表](MOTION_TABLE.md)。缺少素材时应采集和校验，不应放松门槛伪造行走。

## ARDY

核心 HTTP/MCP 和预设动作无需加载 ARDY 模型。实时生成按需加载，默认 `ARDY_WARMUP=0`。要启用生成：

1. 单独安装上游 `nv-tlabs/ardy` 及其推理依赖和权重，设置 `ARDY_ROOT`。
2. 设置 `ARDY_PYTHON` 指向可导入 ARDY 与本项目核心依赖的 Python。
3. 设置 `ARDY_EMBEDDINGS` 指向可信的 `.npz` 提示词缓存，包含 `prompts` 和 `emb`（每条 4096 维）。服务仅接受缓存提示词，不会在每次请求临时加载大型文本编码器。
4. 如需 SMPL/AMASS 轨迹约束转换，另行获取 SMPL 并设置 `SMPL_MODEL_PATH`。基础规划、播放与通用骨架不需要 SMPL 文件。
5. 启动 `npm run service`。需要启动时预热双模型可设 `ARDY_WARMUP=1`；生成仍固定使用六步扩散。

`server/service.py`、`core.py`、`realtime.py` 是项目内适配代码；不再依赖开发者磁盘中的服务入口。可选旧 TMR 识别需另行准备 `MOTION_TOOLS_ROOT/ardy_service/recognize.py` 和 `TMR_PYTHON`，不属于基础运行要求。生成结果写入 `MOTION_SERVICE_DIR`，默认 `public/service/`。

ARDY 缓存和 SMPL 包含对象序列化格式，只应加载你信任的本地文件。模型、数据和权重许可应分别核查。

## 感知与 SuperMimic

感知：`npm run setup:perception` 安装声明依赖，`npm run perception:download` 下载模型，`npm run perception` 启动。当前 ASR 后端针对 Apple Silicon MLX；麦克风由网页手动开启。人脸/声纹登记保存在被忽略的 `.local-data/`。

SuperMimic：`npm run supermimic:download` 获取固定版本社区权重，`npm run supermimic:assets` 获取可选测试资产。物理与推理依赖使用 `server/requirements-supermimic.txt`，并将 `MOTION_PERCEPTION_PYTHON` 指向对应解释器。与网页主动作的边界见 [SuperMimic](SUPERMIMIC.md)。

## 发布检查

```sh
python3 scripts/audit_release.py --tracked
```

该检查扫描 Git 索引中的路径与文件内容，拒绝本机主目录、密钥特征、模型/缓存目录和超大文件。它不会上传或打印密钥值。日志、截图、身份登记、个人 `.env` 及客户端 `mcp.json` 不提交。
