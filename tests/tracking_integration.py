"""Real MCP transport -> browser -> constrained ARDY, moving target has no future path."""
import asyncio
import json
import math
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT=Path(__file__).resolve().parents[1]


async def call(session,name,args=None):
    result=await session.call_tool(name,args or {})
    assert not result.isError,result
    return json.loads(result.content[0].text)


async def state(session):
    result=await call(session,"get_simulation_state")
    return min(result["sessions"],key=lambda entry:entry["ageSeconds"])["state"]


async def completed(session,command):
    for _ in range(40):
        status=await call(session,"get_command_status",{"command_id":command["id"]})
        assert status["status"]!="failed",status
        if status["status"]=="completed": return status
        await asyncio.sleep(.3)
    raise AssertionError(status)


async def main():
    params=StdioServerParameters(command=sys.executable,args=[str(ROOT/"server/mcp_server.py")])
    async with stdio_client(params) as (reader,writer):
        async with ClientSession(reader,writer) as session:
            await session.initialize()
            await completed(session,await call(session,"interrupt_motion",{"mode":"reset"}))
            face=await call(session,"face_toward",{"heading":90,"duration":2})
            await completed(session,face)
            facing=await state(session)
            difference=(facing["heading"]-90+180)%360-180
            assert abs(difference)<18,facing
            await completed(session,await call(session,"interrupt_motion",{"mode":"reset"}))
            await call(session,"observe_target",{"target_id":"integration-moving","position":[-5,.92,0]})
            follow=await call(session,"follow_target",{"target_id":"integration-moving","distance":1.2})
            observed=[]
            for i in range(22):
                position=[-5,.92,0] if i<10 else [-2,.92,0]
                await call(session,"observe_target",{"target_id":"integration-moving","position":position,"ttl":1})
                current=await state(session)
                observed.append({"position":current["position"],"following":current["following"]})
                status=await call(session,"get_command_status",{"command_id":follow["id"]})
                assert status["status"]!="failed",status
                await asyncio.sleep(.35)
            assert any(s["following"] for s in observed),observed
            displacement=math.dist(observed[0]["position"],observed[-1]["position"])
            assert displacement>.5,observed
            await asyncio.sleep(2)
            lost=await state(session)
            assert not lost["playing"] and not lost["generating"],lost
            await completed(session,await call(session,"interrupt_motion",{"mode":"hold"}))
            perception=await call(session,"get_perception")
            assert perception["vision"]["status"]=="ready",perception
            report={"facingDegrees":facing["heading"],"movingTargetDisplacement":displacement,
                    "observations":observed,"lostStops":True,"visionReady":True}
            (ROOT/"output/performance/tracking-integration.json").write_text(json.dumps(report,indent=2))
            print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=="__main__":asyncio.run(main())
