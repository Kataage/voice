from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from personavoice.config import PersonaConfig
from personavoice.inference import chat_turn, reenact, repeat, synthesize
from personavoice.project import find_repo_root, get_persona

app = FastAPI(title="PersonaVoice", version="0.3.0")


class TTSRequest(BaseModel):
    persona: str
    text: str
    style: str | None = None
    emotion: str | None = None
    events: list[str] = Field(default_factory=list)
    ref: str | None = None
    candidates: int | None = None
    seed: int | None = None


class AudioRequest(BaseModel):
    persona: str
    source: str
    ref: str | None = None
    transfer_style: bool = True


class ChatRequest(BaseModel):
    persona: str
    prompt: str
    history: list[dict[str, str]] = Field(default_factory=list)


def _load(name: str):
    root = find_repo_root()
    paths = get_persona(root, name)
    return root, paths, PersonaConfig.load(paths.config)


def _source_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input audio does not exist: {path}")
    return path


def _output_item(persona: str, paths, path: Path) -> dict:
    resolved = path.resolve()
    relative = resolved.relative_to(paths.outputs.resolve()).as_posix()
    return {
        "path": str(resolved),
        "url": f"/v1/output/{quote(persona, safe='')}/{quote(relative, safe='/')}",
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/v1/personas")
def personas() -> dict:
    root = find_repo_root()
    base = root / "personas"
    return {
        "personas": sorted(path.name for path in base.iterdir() if path.is_dir())
        if base.exists()
        else []
    }


@app.get("/v1/output/{persona}/{relative_path:path}")
def output_audio(persona: str, relative_path: str):
    try:
        _, paths, _ = _load(persona)
        base = paths.outputs.resolve()
        output = (base / relative_path).resolve()
        output.relative_to(base)
        if not output.is_file():
            raise FileNotFoundError(output)
        return FileResponse(output, media_type="audio/wav", filename=output.name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/tts")
def tts(request: TTSRequest) -> dict:
    try:
        root, paths, cfg = _load(request.persona)
        outputs = synthesize(
            root,
            paths,
            cfg,
            request.text,
            style=request.style,
            emotion=request.emotion,
            events=request.events,
            ref=request.ref,
            candidates=request.candidates,
            seed=request.seed,
        )
        return {"outputs": [_output_item(request.persona, paths, path) for path in outputs]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/voice-convert")
def voice_convert(request: AudioRequest) -> dict:
    try:
        root, paths, cfg = _load(request.persona)
        output = reenact(
            root,
            paths,
            cfg,
            _source_file(request.source),
            ref=request.ref,
            transfer_style=request.transfer_style,
        )
        return {"output": _output_item(request.persona, paths, output)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/repeat")
def repeat_endpoint(request: AudioRequest) -> dict:
    try:
        root, paths, cfg = _load(request.persona)
        outputs = repeat(root, paths, cfg, _source_file(request.source))
        return {"outputs": [_output_item(request.persona, paths, path) for path in outputs]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/chat")
def chat_endpoint(request: ChatRequest) -> dict:
    try:
        root, paths, cfg = _load(request.persona)
        result = chat_turn(root, paths, cfg, request.prompt, request.history)
        audio_path = Path(str(result["audio"]))
        result["audio"] = _output_item(request.persona, paths, audio_path)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>PersonaVoice</title>
<style>
body{font:15px system-ui;max-width:980px;margin:32px auto;padding:0 20px;background:#f6f7f9;color:#16181d}
.card{background:#fff;padding:18px;margin:14px 0;border-radius:12px;box-shadow:0 1px 5px #0001}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.full{grid-column:1/-1}
input,textarea,select,button{font:inherit;padding:10px;width:100%;box-sizing:border-box;border:1px solid #ccd0d7;border-radius:8px}
button{cursor:pointer;background:#17191f;color:#fff;border:0}button:hover{opacity:.88}
pre{white-space:pre-wrap;background:#f0f2f5;padding:12px;border-radius:8px;max-height:260px;overflow:auto}
audio{width:100%;margin-top:10px}.muted{color:#666}@media(max-width:700px){.grid{grid-template-columns:1fr}}
</style></head><body>
<h1>PersonaVoice</h1><p class="muted">Local UI — requests stay on this machine.</p>
<div class="card"><label>Persona</label><select id="persona"></select></div>
<div class="card"><h2>Talk / Voice Design</h2><div class="grid">
<textarea class="full" id="text" rows="4" placeholder="読み上げる文章"></textarea>
<input id="style" placeholder="Style/caption: 嬉しそうに、少し早口で">
<select id="emotion"><option value="">Emotion: Auto</option><option>HAPPY</option><option>SAD</option><option>ANGRY</option><option>SURPRISED</option><option>FEARFUL</option><option>NEUTRAL</option></select>
<input id="events" placeholder="Events: laugh,sigh,cough">
<input id="ref" placeholder="Reference: auto / happy / C:\path\ref.wav">
<button class="full" onclick="tts()">Generate</button></div><div id="ttsAudio"></div></div>
<div class="card"><h2>Audio → Persona</h2><div class="grid">
<input class="full" id="source" placeholder="入力音声のローカルパス">
<button onclick="vc()">Reenact (演技維持)</button><button onclick="repeatAudio()">Repeat (本人として再演)</button>
</div><div id="audioResult"></div></div>
<div class="card"><h2>Chat</h2><textarea id="prompt" rows="3" placeholder="メッセージ"></textarea><button onclick="chat()">Send</button><div id="chatAudio"></div></div>
<div class="card"><h2>Result</h2><pre id="out"></pre></div>
<script>
let history=[];const p=document.getElementById('persona'),o=document.getElementById('out');
async function init(){let x=await(await fetch('/v1/personas')).json();p.replaceChildren(...x.personas.map(v=>{let e=document.createElement('option');e.textContent=v;return e}))}
function player(url){let a=document.createElement('audio');a.controls=true;a.autoplay=true;a.src=url;return a}
function setPlayers(id,items){let d=document.getElementById(id);d.replaceChildren(...items.map(x=>player(x.url)))}
function setChat(text,audio){let d=document.getElementById('chatAudio'),q=document.createElement('p');q.textContent=text||'';d.replaceChildren(q,player(audio.url))}
async function post(url,body){o.textContent='Running...';let r=await fetch(url,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});let x=await r.json();o.textContent=JSON.stringify(x,null,2);if(!r.ok)throw new Error(x.detail||'request failed');return x}
async function tts(){try{let ev=document.getElementById('events').value.split(',').map(x=>x.trim()).filter(Boolean);let x=await post('/v1/tts',{persona:p.value,text:document.getElementById('text').value,style:document.getElementById('style').value||null,emotion:document.getElementById('emotion').value||null,events:ev,ref:document.getElementById('ref').value||null});setPlayers('ttsAudio',x.outputs||[])}catch(e){o.textContent=String(e)}}
async function vc(){try{let x=await post('/v1/voice-convert',{persona:p.value,source:document.getElementById('source').value});setPlayers('audioResult',[x.output])}catch(e){o.textContent=String(e)}}
async function repeatAudio(){try{let x=await post('/v1/repeat',{persona:p.value,source:document.getElementById('source').value});setPlayers('audioResult',x.outputs||[])}catch(e){o.textContent=String(e)}}
async function chat(){try{let prompt=document.getElementById('prompt').value;let x=await post('/v1/chat',{persona:p.value,prompt,history});history.push({role:'user',content:prompt},{role:'assistant',content:JSON.stringify({text:x.text,voice:x.voice})});history=history.slice(-12);setChat(x.text,x.audio)}catch(e){o.textContent=String(e)}}
init();
</script></body></html>"""