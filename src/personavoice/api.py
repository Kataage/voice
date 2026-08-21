from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from personavoice.config import PersonaConfig
from personavoice.inference import chat_turn, reenact, repeat, synthesize
from personavoice.project import find_repo_root, get_persona

app = FastAPI(title="PersonaVoice", version="0.2.0")


class TTSRequest(BaseModel):
    persona: str
    text: str
    style: str | None = None
    emotion: str | None = None
    events: list[str] = Field(default_factory=list)
    candidates: int | None = None


class AudioRequest(BaseModel):
    persona: str
    source: str


class ChatRequest(BaseModel):
    persona: str
    prompt: str


def _load(name: str):
    root = find_repo_root()
    paths = get_persona(root, name)
    return root, paths, PersonaConfig.load(paths.config)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/v1/personas")
def personas() -> dict:
    root = find_repo_root()
    base = root / "personas"
    return {"personas": sorted(path.name for path in base.iterdir() if path.is_dir()) if base.exists() else []}


@app.post("/v1/tts")
def tts(request: TTSRequest) -> dict:
    try:
        root, paths, cfg = _load(request.persona)
        outputs = synthesize(
            root, paths, cfg, request.text, style=request.style, emotion=request.emotion,
            events=request.events, candidates=request.candidates,
        )
        return {"outputs": [str(path) for path in outputs]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/voice-convert")
def voice_convert(request: AudioRequest) -> dict:
    try:
        root, paths, cfg = _load(request.persona)
        return {"output": str(reenact(root, paths, cfg, Path(request.source)))}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/repeat")
def repeat_endpoint(request: AudioRequest) -> dict:
    try:
        root, paths, cfg = _load(request.persona)
        return {"outputs": [str(path) for path in repeat(root, paths, cfg, Path(request.source))]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/chat")
def chat_endpoint(request: ChatRequest) -> dict:
    try:
        root, paths, cfg = _load(request.persona)
        return chat_turn(root, paths, cfg, request.prompt)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return """<!doctype html><html lang=ja><meta charset=utf-8><title>PersonaVoice</title>
<style>body{font:16px system-ui;max-width:900px;margin:40px auto;padding:0 20px}input,textarea,select,button{font:inherit;padding:10px;margin:5px 0;width:100%;box-sizing:border-box}button{cursor:pointer}pre{white-space:pre-wrap;background:#eee;padding:16px;border-radius:8px}</style>
<h1>PersonaVoice</h1><label>Persona</label><select id=p></select><label>Text</label><textarea id=t rows=5></textarea>
<label>Style / VoiceDesign caption</label><input id=s placeholder="例: 嬉しそうに、少し早口で"><label>Emotion</label>
<select id=e><option></option><option>HAPPY</option><option>SAD</option><option>ANGRY</option><option>SURPRISED</option><option>NEUTRAL</option></select>
<button onclick=go()>Generate</button><pre id=o></pre><script>
async function init(){let x=await(await fetch('/v1/personas')).json();p.innerHTML=x.personas.map(v=>`<option>${v}</option>`).join('')}
async function go(){o.textContent='Generating...';let r=await fetch('/v1/tts',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({persona:p.value,text:t.value,style:s.value||null,emotion:e.value||null})});o.textContent=JSON.stringify(await r.json(),null,2)}init();</script></html>"""
