"""Official MCP stdio adapter; forwards to the single resident HTTP model."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, model_validator
from settings import load_environment

load_environment()
Position = Annotated[list[float], Field(min_length=3, max_length=3)]


class ActionStep(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    action: Literal["walk", "run", "wave", "raise", "point", "crouch", "sit", "squat", "crawl"]
    start: float = Field(ge=0, description="相对规划开始的秒数，0.05 秒对齐")
    end: float = Field(gt=0, le=180, description="结束秒数，必须大于 start")
    destination: str | list[float] | None = None
    via: list[str | list[float]] | None = None
    heading: float | None = Field(default=None, description="绝对面向角度：0=+Z，90=+X")
    face: str | list[float] | None = None

    @model_validator(mode='after')
    def ordered_times(self) -> ActionStep:
        if self.end <= self.start:
            raise ValueError('end 必须大于 start')
        return self


class CharacterStep(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    kind: Literal['social', 'waypoint', 'face', 'action', 'wait', 'contact', 'text-touch', 'interact', 'body']
    objectId: str | None = None
    position: Position | None = Field(default=None, description='视觉里程计地图坐标，仅 waypoint 使用')
    heading: float | None = None
    action: str | None = None
    duration: float | None = None
    text: str | None = None
    direction: str | None = None
    targetId: str | None = None
    effector: Literal['left_hand', 'right_hand', 'left_foot', 'right_foot', 'head'] | None = None
    operation: Literal['inspect', 'approach', 'point', 'touch', 'tap', 'hold', 'stroke', 'sit', 'lie', 'pickup', 'play_computer', 'eat', 'drink', 'stand', 'release'] | None = None
    amplitude: float | None = None
    timeout: float = Field(default=120, ge=1, le=600)


class AgentIntent(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    kind: Literal['socialize', 'navigate', 'inspect', 'interact', 'contact', 'action', 'face', 'body']
    position: Position | None = None
    objectId: str | None = None
    targetId: str | None = None
    operation: Literal['inspect', 'approach', 'point', 'touch', 'tap', 'hold', 'stroke', 'sit', 'lie', 'pickup', 'play_computer', 'eat', 'drink', 'stand', 'release'] | None = None
    effector: Literal['left_hand', 'right_hand', 'left_foot', 'right_foot', 'head'] | None = None
    action: str | None = None
    duration: float | None = None
    amplitude: float | None = None
    heading: float | None = None
    timeout: float = Field(default=120, ge=1, le=600)

    @model_validator(mode='after')
    def required_target(self) -> AgentIntent:
        required = {'navigate': 'position', 'inspect': 'objectId', 'interact': 'objectId',
                    'contact': 'targetId', 'action': 'action', 'face': 'heading', 'body': 'operation'}
        field = required.get(self.kind)
        if field and getattr(self, field) is None:
            raise ValueError(f'{self.kind} 需要 {field}')
        return self

BASE = os.environ.get("MOTION_LAB_URL", "http://127.0.0.1:8100")
PERCEPTION_BASE = os.environ.get("MOTION_PERCEPTION_URL", "http://127.0.0.1:8200")
mcp = FastMCP("motion-lab", instructions="控制本地 VRM 房屋仿真。高层控制先 get_agent_context，再 submit_agent_intent；自主行为只依赖视觉记忆。inspect_environment 是调试真值，不得冒充角色感知。所有坐标 Y-up、米、时间秒。move_to / execute_plan 返回命令 ID，必须查询状态确认执行；碰撞片段会拒绝播放。")


def request(path: str, body: dict | None = None, base: str = BASE) -> dict:
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base+path, payload, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()
        raise ValueError(f"动作服务返回 {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 {base}，先启动对应的动作或感知服务") from exc
    except TimeoutError as exc:
        raise RuntimeError('服务请求超时；请先查询任务状态，避免重复提交') from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError('服务返回了无效 JSON') from exc
    if isinstance(result, dict) and "error" in result:
        raise ValueError(result["error"])
    return result


@mcp.tool()
def get_agent_context(client_id: str | None = None) -> dict:
    """LLM 的统一只读入口：视觉位置、人物、空间记忆、交互候选、任务反馈与能力边界，无房屋真值地图。"""
    return request('/api/agent/context', {'clientId': client_id} if client_id else {})


@mcp.tool()
def submit_agent_intent(request_id: Annotated[str, Field(min_length=1, max_length=100)],
                        intent: AgentIntent, client_id: str | None = None) -> dict:
    """提交高层意图：社交、视觉导航、观察、物体操作、标记接触、动作或朝向。
    request_id 重试时必须保持不变，相同请求不会重复发动作；相同 ID 的不同目标将被拒绝。
    导航坐标来自 get_agent_context 的视觉里程计。返回任务后继续查询实际完成和失败反馈。
    """
    return request('/api/agent/intent', {'requestId': request_id, 'intent': intent.model_dump(exclude_none=True),
                                       **({'clientId': client_id} if client_id else {})})


@mcp.tool()
def set_agent_autonomy(enabled: bool, client_id: str | None = None) -> dict:
    """启停环境驱动生活循环：保留任务、探索找人、观察新物体、完成社交后冷却、失败退避重试。
    直接指令接管会暂停该循环。关闭只中断该循环拥有的任务。
    """
    return request('/api/agent/autonomy', {'enabled': enabled, **({'clientId': client_id} if client_id else {})})


@mcp.tool()
def get_motion_capabilities() -> dict:
    """读取原生流明动作目录、动作通道、时长限制与真实引擎。使用 perform_action 执行 action 标识。"""
    return request('/api/actions/capabilities')


@mcp.tool()
def get_object_interactions(client_id: str | None = None) -> dict:
    """从同帧视觉物体推导观察、接近、指向和表面交互选项，以及每项前置条件。"""
    return request('/api/interactions/options', {'clientId': client_id} if client_id else {})


@mcp.tool()
def plan_object_interaction(object_id: str, operation: Literal['inspect', 'approach', 'point', 'touch', 'tap', 'hold', 'stroke', 'sit', 'lie', 'pickup', 'play_computer', 'eat', 'drink'],
                            client_id: str | None = None, execute: bool = False, duration: float = 30) -> dict:
    """预览或执行持续物体交互。执行时复核视觉、通路与接触；不承诺游戏业务效果。"""
    return request('/api/interactions/submit' if execute else '/api/interactions/plan',
                   {'objectId': object_id, 'operation': operation, 'duration': duration, **({'clientId': client_id} if client_id else {})})


@mcp.tool()
def inspect_environment() -> dict:
    """查看三卧一电脑房、两厅三卫一厨、房间目的地及场景碰撞盒。"""
    return request("/api/world")


@mcp.tool()
def get_simulation_state() -> dict:
    """读取在线网页模型位置、朝向、播放时间、性能及命令状态。"""
    return request("/api/sim/state")


@mcp.tool()
def submit_character_goal(title: str, steps: list[CharacterStep], client_id: str | None = None) -> dict:
    """给前端角色下达组合任务。步骤按顺序执行，失败即停，返回任务而非成功承诺。

    social 探索找人并稳定面向；waypoint 只走视觉地图已知通路，position 使用视觉坐标；
    face 指定自身控制坐标朝向；action 的 wave/raise/point 使用流明上身轨道，其余优先复用动作库；wait 自然等待0.2–4秒；
    interact 指定视觉物体 objectId 与 operation，先观察复核再接近、执行；
    contact 对任意已标注 targetId 执行接触，effector/operation/duration/amplitude 指定操作；
    text-touch 为旧文字选择适配入口。不支持游戏文本输入适配器。
    例：[{kind:'social'},{kind:'action',action:'wave',duration:3}]。
    手动控制或其他命令会取消任务，须 get_character_goal 确认每步实际完成。
    """
    return request('/api/goals/submit', {'title': title,
                   'steps': [step.model_dump(exclude_none=True) for step in steps],
                   **({'clientId': client_id} if client_id else {})})


@mcp.tool()
def get_character_goal(client_id: str | None = None) -> dict:
    """读取组合任务进度、失败原因、完成事件及角色视觉空间与人物记忆。"""
    return request('/api/goals/state', {'clientId': client_id} if client_id else {})


@mcp.tool()
def cancel_character_goal(client_id: str | None = None) -> dict:
    """取消整个任务及后续步骤，平滑停步，不会自动恢复探索。"""
    return request('/api/goals/cancel', {'clientId': client_id} if client_id else {})


@mcp.tool()
def plan_route(destination: str | list[float], start: list[float] | None = None,
               via: list[str | list[float]] | None = None, action: str = "walk",
               speed: float | None = None, duration: float | None = None,
               heading: float | None = None) -> dict:
    """预览避障路线；destination/via 支持房间 ID、中文名或 [x,y,z]。不执行。"""
    body = {"destination": destination, "via": via or [], "action": action}
    if speed is not None:
        body["speed"] = speed
    if start is not None:
        body["start"] = start
    if duration is not None:
        body["duration"] = duration
    if heading is not None:
        body["heading"] = heading
    return request("/api/plans/navigation", body)


@mcp.tool()
def plan_actions(steps: list[ActionStep], start: list[float] | None = None) -> dict:
    """规划详细时间表。每段 {action,start,end,destination?,via?}，时间按0.05秒对齐。

    例如 steps=[{"action":"walk","start":0,"end":14,"destination":"dining"},
    {"action":"wave","start":14,"end":18}]。支持 walk/run/wave/raise/point/crouch/sit/squat/crawl。
    返回 planId、逐段路径、距离、时间与碰撞条件；不执行。
    """
    body = {"steps": [step.model_dump(exclude_none=True) for step in steps]}
    if start is not None:
        body["start"] = start
    return request("/api/plans/timeline", body)


@mcp.tool()
def move_to(destination: str | list[float], via: list[str | list[float]] | None = None,
            action: str = "walk", speed: float | None = None, duration: float | None = None,
            client_id: str | None = None, heading: float | None = None,
            face: str | list[float] | None = None) -> dict:
    """从网页正在播放的姿势连贯移动到目的地，经由指定路线；替换未执行的旧命令。"""
    body = {"destination": destination, "via": via or [], "action": action}
    if speed is not None:
        body["speed"] = speed
    if duration is not None:
        body["duration"] = duration
    if client_id is not None:
        body["clientId"] = client_id
    if heading is not None:
        body["heading"] = heading
    if face is not None:
        body["face"] = face
    return request("/api/sim/move", body)


@mcp.tool()
def execute_plan(plan_id: str, client_id: str | None = None) -> dict:
    """把已预览的时间规划交给网页执行；保持时间表并从实际衔接位置重规划。"""
    return request("/api/sim/execute", {"planId": plan_id, **({"clientId": client_id} if client_id else {})})


@mcp.tool()
def interrupt_motion(mode: str = "hold", client_id: str | None = None) -> dict:
    """取消生成并以停步动画收脚。mode=hold；resume 恢复缓冲动作，reset 回出生点。碰撞安全停车可立即停止。"""
    if mode not in ("hold", "resume", "reset"):
        raise ValueError("mode 必须是 hold/resume/reset")
    return request("/api/sim/interrupt", {"mode": mode, **({"clientId": client_id} if client_id else {})})


@mcp.tool()
def get_command_status(command_id: str) -> dict:
    """查询命令是否实际执行成功；failed 包含碰撞或时间规划错误。"""
    return request("/api/sim/commands/"+command_id)


@mcp.tool()
def get_performance() -> dict:
    """读取最近256个生成窗口的延时分位数、根条件投影误差与身体碰撞复检次数。"""
    return request("/api/performance")


@mcp.tool()
def get_action_library() -> dict:
    """查看持久动作库、几何质量、播放次数和入库拒绝原因。基础步态校准见 get_simulation_state。"""
    return request('/api/library')


@mcp.tool()
def perform_action(action: str = 'wave', duration: float = 3, clip_id: str | None = None,
                   client_id: str | None = None) -> dict:
    """连贯执行动作。wave/raise/point 默认使用流明上身重叠轨道，时长1.5–7.5秒，可叠加行走。
    其他动作优先使用合格播放库，缺失或达到频率扩容阈值才生成；高频最多4套。

    支持 wave/raise/point/crouch/squat/sit。clip_id 可指定已有变体，其时长优先。
    返回命令 ID，须 get_command_status 确认完成。当前路线或碰撞不匹配时重新生成。
    """
    body = {'action': action, 'duration': duration}
    if clip_id:
        body['clipId'] = clip_id
    if client_id:
        body['clientId'] = client_id
    return request('/api/sim/action', body)


@mcp.tool()
def observe_target(target_id: str, position: list[float] | None = None,
                   bearing: float | None = None, source: str = "manual", ttl: float = 2,
                   observed_at: float | None = None, label: str = "") -> dict:
    """提交目标的最新观测而非未来轨迹。世界坐标米；bearing 是绝对角度。只有方向不能安全跟随。"""
    body = {"id":target_id, "position":position, "bearing":bearing, "source":source,
            "ttl":ttl, "label":label or target_id}
    if observed_at is not None:
        body["observedAt"] = observed_at
    return request("/api/targets/observe", body)


@mcp.tool()
def face_toward(heading: float | None = None, target_id: str | None = None,
                position: list[float] | None = None, duration: float = 2,
                client_id: str | None = None) -> dict:
    """以真实迈步动画转身，不直接旋转角色壳。heading=0 朝+Z、90朝+X；或面向目标/坐标。

    普通转向不调用 ARDY。duration 为期望时间，实际落脚质量优先；查询状态确认完成。
    未知轨迹的持续面向使用 follow_target。
    """
    body = {"duration":duration}
    for key, value in (("heading",heading),("targetId",target_id),("face",position),("clientId",client_id)):
        if value is not None:
            body[key] = value
    return request("/api/sim/face", body)


@mcp.tool()
def follow_target(target_id: str, distance: float = 1.2, speed: float = 1.4,
                  face_target: bool = True, client_id: str | None = None) -> dict:
    """持续跟随最新目标观测。2秒滚动窗口；丢失停止、不假设未来轨迹。保持距离，仍遵守房屋碰撞约束。"""
    body = {"targetId":target_id, "distance":distance, "speed":speed, "faceTarget":face_target}
    if client_id:
        body["clientId"] = client_id
    return request("/api/sim/follow", body)


@mcp.tool()
def approach_target(target_id: str, distance: float = 1.2, client_id: str | None = None) -> dict:
    """移动到目标当前观测位置的安全距离处；只用该次位置快照。持续跟随使用 follow_target。"""
    return request("/api/sim/approach",{"targetId":target_id,"distance":distance,
                   **({"clientId":client_id} if client_id else {})})


@mcp.tool()
def autonomy_decide(vision: dict[str, object], client_id: str | None = None) -> dict:
    """推进找人社交脚本。提交新鲜 ego 双目/视觉观测；达成返回 completed，不再排入移动。"""
    body = {"vision": vision, **({"clientId":client_id} if client_id else {})}
    return request("/api/sim/autonomy/decide", body)


@mcp.tool()
def get_behavior_state(client_id: str) -> dict:
    """读取社交任务阶段、目标、达成条件、区域路径记忆及历史反馈诊断。"""
    return request("/api/sim/autonomy/state", {"clientId":client_id})


@mcp.tool()
def configure_behavior(learning_rate: float | None = None, temperature: float | None = None,
                       max_candidates: int | None = None) -> dict:
    """调整保留的反馈诊断配置；历史学习权重不参与确定性脚本的移动选择。"""
    body = {}
    if learning_rate is not None:
        body["learningRate"] = learning_rate
    if temperature is not None:
        body["temperature"] = temperature
    if max_candidates is not None:
        body["maxCandidates"] = max_candidates
    return request("/api/sim/autonomy/config", body)


@mcp.tool()
def reset_behavior(client_id: str | None = None) -> dict:
    """清除行为策略的探索记忆和学习状态；不重置模型位置。"""
    return request("/api/sim/autonomy/reset", {"clientId":client_id} if client_id else {})


@mcp.tool()
def get_perception() -> dict:
    """读取感知服务状态、最近的YOLO/人脸检测、ASR、匿名声纹和方向结果；不启动麦克风。"""
    return request("/api/state", base=PERCEPTION_BASE)


@mcp.tool()
def get_visual_world(client_id: str) -> dict:
    """读取指定网页会话由双目 RGB、视觉里程计和 YOLO 语义融合维护的点云、位姿、净空与物体记忆。"""
    if not client_id.strip():
        raise ValueError("client_id 不能为空")
    return request("/api/stereo/state?clientId=" + urllib.parse.quote(client_id, safe=""), base=PERCEPTION_BASE)


@mcp.tool()
def get_target_frame(visual_client_id: str) -> dict:
    """获取任意位置标注用左眼图像、尺寸和同帧 frameId，无需识别人或文字。"""
    return request('/api/targets/frame?clientId='+urllib.parse.quote(visual_client_id, safe=''), base=PERCEPTION_BASE)


@mcp.tool()
def select_contact_target(visual_client_id: str, pixel: list[float], frame_id: int,
                          label: str = '视觉标注') -> dict:
    """选择同帧图像中任意可见标注位置，返回双目表面 targetId；不限制目标类别，不接受真值坐标。"""
    return request('/api/touch/resolve', {'clientId': visual_client_id, 'pixel': pixel,
                   'frameId': frame_id, 'label': label}, base=PERCEPTION_BASE)


@mcp.tool()
def contact_target(target_id: str,
                   effector: Literal['left_hand', 'right_hand', 'left_foot', 'right_foot', 'head'] = 'right_hand',
                   operation: Literal['touch', 'tap', 'hold', 'stroke'] = 'touch',
                   duration: float = .6, amplitude: float = .01, client_id: str | None = None) -> dict:
    """统一触碰任意标注目标（画面点、人体、OCR令牌均可），接近→操作→撤回。

    跟踪丢失、遮挡、不可达或碰撞会失败并停止。返回命令ID，查询get_command_status确认几何接触。
    """
    return request('/api/sim/contact', {'selection': {'targetId': target_id}, 'effector': effector,
                   'operation': operation, 'duration': duration, 'amplitude': amplitude,
                   **({'clientId': client_id} if client_id else {})})


@mcp.tool()
def get_visual_body_frame(visual_client_id: str) -> dict:
    """读取同一帧的左眼 JPEG、人物 ID、YOLO 关键点和双目深度，供视觉触摸选点。"""
    return request('/api/stereo/frame?clientId='+urllib.parse.quote(visual_client_id, safe=''), base=PERCEPTION_BASE)


@mcp.tool()
def touch_body(person_id: str | None = None, body_part: str | None = None,
               target_id: str | None = None, hand: Literal['left', 'right'] = 'right',
               operation: Literal['touch', 'tap', 'hold', 'stroke'] = 'touch',
               duration: float = .6, amplitude: float = .01, client_id: str | None = None) -> dict:
    """用 YOLO 人体关键点与同帧双目深度触摸；不接受手填世界坐标。

    person_id 来自 get_perception 的 people[].id；body_part 如 left_wrist/right_shoulder。
    或使用已解析的 target_id。duration 为停留秒数(.2–3)，amplitude 为滑动米数(0–.04)。
    播放时跟踪18cm内的小幅目标移动；遮挡、跳变、不可达或过期则中断。此为骨骼锚点接触，无抓握和力控制。
    """
    selection = {'targetId': target_id} if target_id else {'personId': person_id, 'bodyPart': body_part}
    return request('/api/sim/touch', {'selection': selection, 'hand': hand, 'operation': operation,
                   'duration': duration, 'amplitude': amplitude, **({'clientId': client_id} if client_id else {})})


@mcp.tool()
def get_contact_capabilities() -> dict:
    """读取通用接触执行部位、操作、跟踪范围，以及摸头/膝枕/揪耳/拍手尚缺的约束。"""
    return request('/api/contact/capabilities')


@mcp.tool()
def contact_body(effector: Literal['left_hand', 'right_hand', 'left_foot', 'right_foot', 'head'],
                 person_id: str | None = None, body_part: str | None = None, target_id: str | None = None,
                 operation: Literal['touch', 'tap', 'hold', 'stroke'] = 'touch',
                 duration: float = .6, amplitude: float = .01, client_id: str | None = None) -> dict:
    """通用接触：执行部位→视觉目标→接近/操作/撤回，并持续跟踪。

    target_id 可为任意视觉标注令牌；person_id/body_part 是兼容的选择适配器，不接受环境真值坐标。thigh/forearm为关键点间表面推断；
    left_ear/right_ear为COCO耳点，不能代表兽耳尖或抓握表面。头部只做有界胸颈姿态；
    脚部需另一只脚支撑及骨盆投影检查，不会移动根节点来够目标。不提供手指/接触力/承重。
    必须用get_command_status检查完成或失败，排队成功不表示接触成功。
    """
    selection = {'targetId': target_id} if target_id else {'personId': person_id, 'bodyPart': body_part}
    return request('/api/sim/contact', {'selection': selection, 'effector': effector, 'operation': operation,
                   'duration': duration, 'amplitude': amplitude, **({'clientId': client_id} if client_id else {})})


@mcp.tool()
def resolve_touch_target(visual_client_id: str, person_id: str, body_part: str | None = None,
                         pixel: list[float] | None = None, frame_id: int | None = None) -> dict:
    """预览视觉部位估计、接触点、深度不确定度和法线。pixel须来自同帧，限制在清晰四肢或面部关键点区域。"""
    return request('/api/touch/resolve', {'clientId': visual_client_id, 'personId': person_id,
                   'bodyPart': body_part, **({'pixel': pixel, 'frameId': frame_id} if pixel is not None else {})}, base=PERCEPTION_BASE)


@mcp.tool()
def recognize_text(image: str, text: str = '', direction: Literal['anywhere', 'left', 'right', 'up', 'down',
                   'upper_left', 'upper_right', 'lower_left', 'lower_right'] = 'anywhere',
                   anchor: list[float] | None = None, occurrence: int | None = None,
                   min_confidence: float = .4) -> dict:
    """本地中英文 OCR，输入 base64 JPEG/PNG；按文字和相对方向选择点击目标。

    返回文字框及 selected.center（原图像素，左上原点、Y向下）。方向相对 anchor，默认图像中心。
    多个匹配返回 ambiguous，须指定方向或从0开始的 occurrence；按大致阅读顺序编号。
    只定位，不向操作系统发送鼠标事件；调用者须在同一图像对应的目标界面执行并验证。
    """
    body = {'image': image, 'text': text, 'direction': direction, 'minConfidence': min_confidence}
    if anchor is not None:
        body['anchor'] = anchor
    if occurrence is not None:
        body['occurrence'] = occurrence
    return request('/api/ocr', body, base=PERCEPTION_BASE)


@mcp.tool()
def read_view_text(visual_client_id: str, text: str = '', direction: str = 'anywhere',
                   occurrence: int | None = None) -> dict:
    """识别角色当前双目视野中的中英文，返回匹配文字像素框，不执行触碰。方向是画面方向。"""
    return request('/api/ocr/ego', {'clientId': visual_client_id, 'text': text, 'direction': direction,
                   **({'occurrence': occurrence} if occurrence is not None else {})}, base=PERCEPTION_BASE)


@mcp.tool()
def touch_view_text(text: str, heading: float | None = None, direction: str = 'anywhere',
                    occurrence: int | None = None, client_id: str | None = None) -> dict:
    """角色转向停稳、OCR识别文字、同帧双目定位表面，再就近选手轻触；位置不读场景真值。

    heading是世界朝向（0=+Z、90=+X），留空保持；direction是文字在视野中的上下左右位置。
    多个匹配须用方向或从0开始的occurrence消歧。过期、无深度、不可达或碰撞将失败并停步。
    返回命令ID，必须get_command_status检查完成。触碰是几何接触，不代表按钮已激活。
    """
    body = {'text': text, 'direction': direction}
    for key, value in (('heading', heading), ('occurrence', occurrence), ('clientId', client_id)):
        if value is not None:
            body[key] = value
    return request('/api/sim/text-contact', body)


@mcp.tool()
def enroll_identity(name: str, modality: Literal["face", "speaker"], sample_id: str,
                    consent: bool = False) -> dict:
    """用明确同意登记的近期人脸/声音样本建立本地姓名匹配。未登记者仅匿名；不按方向猜身份。"""
    return request("/api/identities/enroll", {"name":name, "modality":modality,
                   "sampleId":sample_id, "consent":consent}, base=PERCEPTION_BASE)


@mcp.tool()
def delete_identity(name: str) -> dict:
    """删除此名称的本地人脸/声纹登记。"""
    return request("/api/identities/delete", {"name":name}, base=PERCEPTION_BASE)


if __name__ == "__main__":
    mcp.run(transport="stdio")
