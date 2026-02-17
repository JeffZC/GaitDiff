from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from pathlib import Path
from typing import Dict, Any
import json

from .azure_client import call_azure_chat

app = FastAPI()
DATA_ROOT = Path("runs_chat")
DATA_ROOT.mkdir(exist_ok=True)

# simple in-memory conversation store
conversations: Dict[str, Dict[str, Any]] = {}

class MsgReq(BaseModel):
    conversation_id: str
    message: str


def summarize_results(results: dict) -> str:
    gc = results.get("gait_comparison", {})
    parts = []
    for k in ("cadence","step_length","walking_speed","step_time","total_steps"):
        v = gc.get(k)
        if v is None:
            continue
        if isinstance(v, dict):
            parts.append(f"{k}: a={v.get('video_a')}, b={v.get('video_b')}")
        else:
            parts.append(f"{k}: {v}")
    return " | ".join(parts) or "GaitDiff results (no summary fields found)."


@app.post("/init")
async def init_chat(summary: str = Form(None), file: UploadFile | None = File(None)):
    convo_id = str(uuid4())
    convo_dir = DATA_ROOT / convo_id
    convo_dir.mkdir(parents=True, exist_ok=True)

    stored_summary = summary
    results_path = None
    if file:
        dest = convo_dir / "results.json"
        content = await file.read()
        dest.write_bytes(content)
        results_path = str(dest)
        try:
            results = json.loads(content)
            stored_summary = stored_summary or summarize_results(results)
        except Exception:
            stored_summary = stored_summary or "Uploaded results.json (could not parse)."

    conversations[convo_id] = {
        "summary": stored_summary or "No summary provided.",
        "history": [{"role":"system", "content": stored_summary or "GaitDiff results stored."}],
        "results_path": results_path,
    }
    return {"conversation_id": convo_id, "summary": conversations[convo_id]["summary"]}


@app.post("/message")
async def message(req: MsgReq):
    if req.conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    convo = conversations[req.conversation_id]
    convo["history"].append({"role":"user", "content": req.message})

    # Build messages for model: system(summary) + recent turns
    system = {"role":"system", "content": convo["summary"]}
    recent = convo["history"][-8:]
    messages = [system] + recent

    # Call Azure/OpenAI (wrapper handles lazy imports & errors)
    assistant_text = call_azure_chat(messages)

    convo["history"].append({"role":"assistant", "content": assistant_text})
    return {"reply": assistant_text}
