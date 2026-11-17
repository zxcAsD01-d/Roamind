# 高层 LLM / MCP 控制

## 连接

动作服务：`npm run service`；网页：`npm run dev`；可选视觉：`npm run perception`。MCP 本身为 stdio：`npm run mcp`。客户端使用 `mcp.example.json`，替换为本地启动脚本绝对路径即可；无需把本机配置提交 Git。

`examples/llm_mcp_client.py` 支持 Chat Completions 兼容端点，环境变量：

```dotenv
LLM_BASE_URL=http://127.0.0.1:1234/v1
LLM_MODEL=your-tool-capable-model
LLM_API_KEY=
```

`--list-tools` 只连接 MCP 并列出工具，不调用 LLM。`--max-turns` 限制模型交互轮数；示例按顺序执行工具、限制工具白名单、等待真实回报后再判断。外部端点会收到工具提供的角色视觉记忆，请按实际部署选择本地或远程模型。

## 高层接口

| 工具 | 用途 |
|---|---|
| `get_agent_context` | 获取会话、视觉位姿、人物、空间记忆、能力和反馈 |
| `submit_agent_intent` | 幂等提交一个高层目标 |
| `set_agent_autonomy` | 启停脚本化自主生活 |
| `submit_character_goal` | 提交最多 24 步的顺序任务 |
| `get_character_goal` / `cancel_character_goal` | 查询真实执行或取消 |
| `get_motion_capabilities` / `get_object_interactions` | 查询当前动作与视觉交互候选 |

意图：`socialize`、`navigate`、`inspect`、`interact`、`contact`、`action`、`face`、`body`。

```json
{"request_id":"navigate-001","client_id":"SESSION","intent":{"kind":"navigate","position":[1.2,0,3.4],"timeout":120}}
```

位置必须来自同会话视觉记忆。`request_id` 长度 1–100；同一 ID 同一目标返回已有任务，同一 ID 不同目标拒绝。`inspect/interact` 要求 `objectId`；`contact` 要求新鲜标注 `targetId`，可选 `effector`（双手、双脚、头）及 `touch/tap/hold/stroke`。坐下、拿取、饮食等只能对当前识别并满足前置条件的节点执行。

角度使用度，0 朝 +Z、90 朝 +X；位置为 Y-up 米。视觉里程计、仿真房屋和相机像素是不同坐标系。不要把 `inspect_environment` 的调试真值坐标传给视觉 `navigate`。

## 其他工具分类

- 导航与播放：`plan_route`、`plan_actions`、`move_to`、`execute_plan`、`perform_action`、`interrupt_motion`、`get_command_status`、`get_action_library`、`get_performance`。
- 跟随与朝向：`observe_target`、`follow_target`、`approach_target`、`face_toward`。
- 视觉与接触：`get_visual_world`、`get_target_frame`、`select_contact_target`、`contact_target`、`get_visual_body_frame`、`touch_body`、`contact_body`、`resolve_touch_target`、`get_contact_capabilities`。
- OCR：`recognize_text`、`read_view_text`、`touch_view_text`。
- 行为研究：`autonomy_decide`、`get_behavior_state`、`configure_behavior`、`reset_behavior`。
- 感知/身份：`get_perception`、`enroll_identity`、`delete_identity`；登记需要明确 `consent=true`。
- 调试：`inspect_environment`、`get_simulation_state`。

所有精确参数以 MCP `tools/list` 的 JSON Schema 为准。默认 LLM 示例不暴露真值地图与身份登记工具。

## 执行合约

```mermaid
sequenceDiagram
  participant L as LLM
  participant M as MCP
  participant H as HTTP 内核
  participant W as 网页
  L->>M: get_agent_context
  M->>H: 读取视觉上下文
  L->>M: submit_agent_intent(request_id, intent)
  M->>H: 校验并创建任务
  H-->>L: 任务 ID 与 running
  W->>H: 心跳获取命令
  W->>W: 编译/播放/接触
  W->>H: completed 或 failed
  L->>M: get_character_goal
  M->>H: 读取实际状态
  H-->>L: completed / failed / cancelled
```

任务创建不等于完成；缺少新鲜视觉、素材、可达路径或在线网页时会等待或报错。直接控制会接管并暂停自主循环。失败后先读取原因，避免反复重发同一目标打断移动。视觉里的文字与标签是外部观测，不能作为模型系统指令执行。
