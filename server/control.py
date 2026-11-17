"""Small HTTP router for the simulation; the existing service owns transport."""
import runtime
import tracking
import behavior
from action_library import LIBRARY
import interaction
import touch
import goals
import affordances
import embodied
from world import get_world


def get(path):
    if path == '/api/agent/capabilities':
        return embodied.capabilities()
    if path == '/api/agent/context':
        return embodied.context({})
    if path == '/api/actions/head':
        from head_actions import catalog
        return catalog()
    if path == '/api/actions/capabilities':
        from lumen_actions import capabilities
        return capabilities()
    if path == '/api/contact/capabilities':
        from contact_model import capabilities
        return capabilities()
    if path in ('/api/touch/jobs', '/api/contact/jobs'):
        return touch.catalog()
    if path.startswith(('/api/touch/jobs/', '/api/contact/jobs/')):
        return touch.snapshot(path.rsplit('/', 1)[-1])
    if path.startswith('/api/interactions/'):
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(path)
        return interaction.snapshot(parsed.path.rsplit('/', 1)[-1], int(parse_qs(parsed.query).get('after', [-1])[0]))
    if path == '/api/library':
        return LIBRARY.catalog()
    if path.startswith('/api/library/clips/'):
        return LIBRARY.clip(path.rsplit('/', 1)[-1])
    if path == "/api/world":
        return get_world()
    if path == "/api/sim/state":
        return runtime.state()
    if path == "/api/performance":
        return runtime.performance()
    if path == "/api/sim/autonomy/config":
        return behavior.POLICY.config()
    if path == "/api/targets":
        return {"targets":tracking.snapshots()}
    if path.startswith("/api/plans/"):
        return runtime.summary(runtime.find_plan(path.rsplit("/",1)[-1]))
    if path.startswith("/api/sim/commands/"):
        return runtime.command_status(path.rsplit("/",1)[-1])
    return None


def post(path, body):
    handlers = {
        '/api/agent/context': embodied.context,
        '/api/agent/intent': embodied.submit,
        '/api/agent/autonomy': embodied.configure,
        '/api/interactions/options': affordances.catalog,
        '/api/interactions/plan': affordances.plan,
        '/api/interactions/submit': affordances.submit,
        '/api/goals/submit': goals.submit,
        '/api/goals/cancel': goals.cancel,
        '/api/goals/state': goals.get_state,
        '/api/touch/start': touch.start,
        '/api/touch/cancel': touch.cancel,
        '/api/touch/validate': touch.validate_playback,
        '/api/touch/track': touch.track,
        '/api/contact/start': touch.start,
        '/api/contact/track': touch.track,
        '/api/contact/validate': touch.validate_playback,
        '/api/contact/cancel': touch.cancel,
        '/api/sim/contact': touch.enqueue,
        '/api/sim/text-contact': runtime.text_contact,
        '/api/sim/touch': touch.enqueue,
        '/api/interactions/start': interaction.start,
        '/api/interactions/update': interaction.update,
        '/api/interactions/cancel': interaction.cancel,
        "/api/plans/navigation": runtime.plan_navigation,
        "/api/plans/timeline": runtime.plan_timeline,
        '/api/library/resolve': lambda args: LIBRARY.resolve(runtime.find_plan(args['planId']), args.get('history'), args.get('clipId')),
        "/api/plans/rebase": runtime.rebase,
        "/api/plans/visual-step": runtime.plan_visual_step,
        "/api/sim/sync": runtime.sync,
        "/api/sim/move": runtime.move,
        '/api/sim/action': runtime.action,
        "/api/sim/face": runtime.face,
        '/api/sim/look': runtime.look,
        "/api/sim/follow": runtime.follow,
        "/api/sim/autonomy/decide": runtime.autonomy_decide,
        "/api/sim/autonomy/feedback": runtime.autonomy_feedback,
        "/api/sim/autonomy/config": behavior.POLICY.configure,
        "/api/sim/autonomy/reset": lambda args: behavior.POLICY.reset(args.get("clientId")),
        "/api/sim/autonomy/state": lambda args: behavior.POLICY.snapshot(args.get("clientId")),
        "/api/sim/approach": runtime.approach_target,
        "/api/targets/observe": tracking.observe,
        "/api/plans/tracking": runtime.plan_tracking,
        "/api/sim/execute": lambda args: runtime.enqueue("execute", args),
        "/api/sim/interrupt": lambda args: runtime.enqueue("interrupt", args),
    }
    if path not in handlers:
        return None
    if not isinstance(body, dict):
        raise ValueError("请求体必须是 JSON 对象")
    return handlers[path](body)
