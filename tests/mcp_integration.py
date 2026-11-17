"""Exercise the real MCP SDK transport and browser execution feedback."""
import asyncio
import json
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def call(session, name, arguments=None):
    result = await session.call_tool(name, arguments or {})
    assert not result.isError, result
    return json.loads(result.content[0].text)


async def wait_command(session, command, timeout=45):
    start = time.monotonic()
    states = []
    while time.monotonic()-start < timeout:
        result = await call(session, "get_command_status", {"command_id": command["id"]})
        status = result["status"]
        if not states or states[-1] != status:
            states.append(status)
        if status == "failed":
            raise AssertionError(result)
        if status in ("completed", "interrupted"):
            return states
        await asyncio.sleep(.5)
    raise AssertionError(f"命令超时 {states}")


async def main():
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT/"server/mcp_server.py")])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            listed = await session.list_tools()
            assert {"move_to","follow_target","face_toward","observe_target","get_perception"} <= {tool.name for tool in listed.tools}
            world = await call(session, "inspect_environment")
            assert len(world["rooms"]) == 11
            reset = await call(session, "interrupt_motion", {"mode":"reset"})
            await wait_command(session,reset,10)
            state = await call(session,"get_simulation_state")
            active = min(state["sessions"],key=lambda s:s["ageSeconds"])
            assert active["state"]["houseEnabled"]
            # Preview a route and a timed plan, then execute the plan over MCP.
            route = await call(session,"plan_route",{"destination":"bedroom4"})
            assert route["timeline"][0]["distance"] > 10
            plan = await call(session,"plan_actions",{"steps":[
                {"action":"walk","start":0,"end":14,"destination":"dining"},
                {"action":"wave","start":14,"end":18}]})
            command = await call(session,"execute_plan",{"plan_id":plan["id"]})
            states = await wait_command(session,command)
            after = await call(session,"get_simulation_state")
            current = min(after["sessions"],key=lambda s:s["ageSeconds"])["state"]
            assert abs(current["position"][0]) < .1 and abs(current["position"][2]-3.2) < .1, current
            print(json.dumps({"tools":len(listed.tools),"executionStates":states,"finalPosition":current["position"]},ensure_ascii=False),flush=True)
            # Explicit move_to, then interrupt it while generation/playback is live.
            move = await call(session,"move_to",{"destination":"kitchen","via":["hallway"]})
            await asyncio.sleep(1.5)
            stop = await call(session,"interrupt_motion",{"mode":"hold"})
            await wait_command(session,stop,10)
            final = await call(session,"get_simulation_state")
            state = min(final["sessions"],key=lambda s:s["ageSeconds"])["state"]
            assert not state["playing"] and not state["generating"], state
            print(json.dumps({"moveCommand":move["id"],"interrupt":"passed"}),flush=True)
            (ROOT/"output/performance/mcp-integration.json").write_text(json.dumps({"states":states,"final":current,"moveInterrupted":True},indent=2))


if __name__ == '__main__':
    asyncio.run(main())
