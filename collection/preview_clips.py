#!/usr/bin/env python3
# Vendored from h2r_collection (github.com/ravikola02/h2r_collection, MIT) scripts/preview_clips.py at 0783627.
"""Review cut episode clips: head + both wrists, played in sync.

    python3 scripts/preview_clips.py --session box_folding_1

Serves a local page that walks the cut sheet episode by episode and plays the
three camera clips together, so a cut can be checked before the LeRobot build.
Episodes whose clips are not on disk yet (the cutter is still running) are
greyed out; the list refreshes itself.

Flags are written next to the cut sheet as ``<cut-sheet stem>_flags.json`` so a
review survives a restart and can drive a later re-cut.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("EGOUMI_DATA_ROOT", REPO_ROOT / "data"))   # egoumi change: data root override
RAW_ROOT = DATA_ROOT / "raw"
OUTPUT_ROOT = DATA_ROOT / "output"
ROLES = ("wrist_left", "head", "wrist_right")   # display order, head centre

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clip previewer</title>
<style>
 :root{--bg:#14161a;--panel:#1d2026;--line:#2c313a;--fg:#e8eaed;--dim:#9aa3af;
       --accent:#4c8dff;--ok:#36b37e;--flag:#ff6b6b}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;
        gap:16px;align-items:center;flex-wrap:wrap}
 h1{font-size:15px;margin:0;font-weight:650;letter-spacing:.2px}
 .sub{color:var(--dim);font-size:12px}
 main{display:flex;height:calc(100vh - 53px);min-height:0}
 #list{width:230px;flex:none;overflow:auto;border-right:1px solid var(--line);
       background:var(--panel)}
 #list div{padding:6px 12px;border-bottom:1px solid var(--line);cursor:pointer;
           display:flex;justify-content:space-between;gap:8px;font-size:13px}
 #list div:hover{background:#232833}
 #list div.sel{background:var(--accent);color:#fff}
 #list div.missing{color:#5d646e;cursor:default}
 #list div.flagged .nm::after{content:" ⚑";color:var(--flag)}
 #list div.sel.flagged .nm::after{color:#ffd7d7}
 .t{color:var(--dim);font-variant-numeric:tabular-nums}
 #list div.sel .t{color:#dce8ff}
 #stage{flex:1;display:flex;flex-direction:column;min-width:0;padding:12px;gap:10px}
 #vids{display:grid;grid-template-columns:1fr 1.5fr 1fr;gap:8px;min-height:0;flex:1}
 figure{margin:0;display:flex;flex-direction:column;min-height:0;background:#000;
        border:1px solid var(--line);border-radius:6px;overflow:hidden}
 figcaption{padding:4px 8px;font-size:11px;color:var(--dim);background:var(--panel);
            border-bottom:1px solid var(--line);text-transform:uppercase;letter-spacing:.6px}
 video{width:100%;height:100%;min-height:0;object-fit:contain;background:#000;display:block}
 #bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 button{background:#272c36;color:var(--fg);border:1px solid var(--line);
        border-radius:6px;padding:6px 12px;cursor:pointer;font-size:13px}
 button:hover{background:#30363f}
 button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
 button.flag{background:var(--flag);border-color:var(--flag);color:#fff}
 #seek{flex:1;min-width:140px;accent-color:var(--accent)}
 .k{color:var(--dim);font-size:12px}
 #err{color:var(--flag);font-size:12px}
</style>
<header>
  <h1>Clip previewer</h1>
  <span class="sub" id="hdr">loading…</span>
  <span class="sub" id="counts"></span>
  <span class="k" style="margin-left:auto">← → episode · space play/pause · f flag · l loop</span>
</header>
<main>
  <div id="list"></div>
  <div id="stage">
    <div id="vids">
      <figure><figcaption>wrist_left</figcaption><video id="v0" muted playsinline preload="auto"></video></figure>
      <figure><figcaption>head</figcaption><video id="v1" muted playsinline preload="auto"></video></figure>
      <figure><figcaption>wrist_right</figcaption><video id="v2" muted playsinline preload="auto"></video></figure>
    </div>
    <div id="bar">
      <button id="prev">◀ prev</button>
      <button id="play" class="primary">▶ play</button>
      <button id="next">next ▶</button>
      <input type="range" id="seek" min="0" max="1000" value="0">
      <span class="t" id="time">0.00 / 0.00</span>
      <button id="loop">loop: on</button>
      <button id="speed">1×</button>
      <button id="flag">⚑ flag</button>
      <span id="err"></span>
    </div>
  </div>
</main>
<script>
const $=s=>document.querySelector(s);
const V=[$('#v0'),$('#v1'),$('#v2')];
let eps=[], cur=-1, loop=true, speed=1, flags={};

function fmt(x){return (x||0).toFixed(2)}

async function load(){
  const r=await fetch('/api/episodes'); const d=await r.json();
  eps=d.episodes; flags=d.flags||{};
  $('#hdr').textContent=d.session+' · cut sheet '+d.cut_sheet;
  const ready=eps.filter(e=>e.ready).length;
  $('#counts').textContent=ready+' of '+eps.length+' clips on disk'
     +(ready<eps.length?' (cutter still running)':'')
     +' · '+Object.keys(flags).length+' flagged';
  render();
  if(cur<0){const i=eps.findIndex(e=>e.ready); if(i>=0) select(i);}
}
function render(){
  const L=$('#list'); L.innerHTML='';
  eps.forEach((e,i)=>{
    const d=document.createElement('div');
    d.className=(i===cur?'sel ':'')+(e.ready?'':'missing ')+(flags[e.name]?'flagged':'');
    d.innerHTML='<span class="nm">'+e.name+'</span><span class="t">'+fmt(e.end-e.start)+'s</span>';
    if(e.ready) d.onclick=()=>select(i);
    L.appendChild(d);
  });
  const s=L.querySelector('.sel'); if(s) s.scrollIntoView({block:'nearest'});
}
function select(i){
  if(i<0||i>=eps.length||!eps[i].ready) return;
  cur=i; const e=eps[i]; $('#err').textContent='';
  V.forEach((v,k)=>{ v.src='/clip/'+encodeURIComponent(e.name)+'/'+e.roles[k];
                     v.playbackRate=speed; v.load(); });
  render(); updFlag();
}
function updFlag(){
  const e=eps[cur]; if(!e) return;
  $('#flag').className = flags[e.name] ? 'flag' : '';
  $('#flag').textContent = flags[e.name] ? '⚑ flagged' : '⚑ flag';
}
function allPlay(){ V.forEach(v=>v.play().catch(()=>{})); $('#play').textContent='❚❚ pause'; }
function allPause(){ V.forEach(v=>v.pause()); $('#play').textContent='▶ play'; }
function toggle(){ V[1].paused?allPlay():allPause(); }
function seekAll(t){ V.forEach(v=>{ try{v.currentTime=t}catch(_){} }); }

V[1].addEventListener('timeupdate',()=>{
  const t=V[1].currentTime, d=V[1].duration||0;
  $('#time').textContent=fmt(t)+' / '+fmt(d);
  $('#seek').value = d? Math.round(1000*t/d) : 0;
  // keep the wrists locked to the head track
  [V[0],V[2]].forEach(v=>{ if(v.duration && Math.abs(v.currentTime-t)>0.12){
      try{v.currentTime=t}catch(_){} } });
});
V[1].addEventListener('ended',()=>{ if(loop){ seekAll(0); allPlay(); } else allPause(); });
V.forEach(v=>v.addEventListener('error',()=>{ $('#err').textContent='clip failed to load'; }));

$('#seek').oninput=e=>{ const d=V[1].duration||0; seekAll(d*e.target.value/1000); };
$('#play').onclick=toggle;
$('#prev').onclick=()=>{ for(let i=cur-1;i>=0;i--) if(eps[i].ready){select(i);break;} };
$('#next').onclick=()=>{ for(let i=cur+1;i<eps.length;i++) if(eps[i].ready){select(i);break;} };
$('#loop').onclick=()=>{ loop=!loop; $('#loop').textContent='loop: '+(loop?'on':'off'); };
$('#speed').onclick=()=>{ speed = speed===1?0.5 : speed===0.5?0.25 : 1;
  $('#speed').textContent=speed+'×'; V.forEach(v=>v.playbackRate=speed); };
$('#flag').onclick=async()=>{
  const e=eps[cur]; if(!e) return;
  const on=!flags[e.name];
  if(on) flags[e.name]=true; else delete flags[e.name];
  updFlag(); render();
  await fetch('/api/flag',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:e.name,flagged:on})});
};
addEventListener('keydown',ev=>{
  if(ev.key==='ArrowRight'){$('#next').click();ev.preventDefault();}
  else if(ev.key==='ArrowLeft'){$('#prev').click();ev.preventDefault();}
  else if(ev.key===' '){toggle();ev.preventDefault();}
  else if(ev.key==='f'){$('#flag').click();}
  else if(ev.key==='l'){$('#loop').click();}
});
load(); setInterval(()=>{ if(eps.filter(e=>e.ready).length<eps.length) load(); },15000);
</script>
"""


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class State:
    def __init__(self, session: str, cut_sheet: Path, clips_dir: Path):
        self.session = session
        self.cut_sheet = cut_sheet
        self.clips_dir = clips_dir
        self.flags_path = cut_sheet.with_name(cut_sheet.stem + "_flags.json")
        self.flags = {}
        if self.flags_path.is_file():
            try:
                self.flags = json.loads(self.flags_path.read_text())
            except Exception:  # noqa: BLE001 — a corrupt flag file must not block review
                self.flags = {}

    def episodes(self):
        clips = json.loads(self.cut_sheet.read_text()).get("clips", [])
        out = []
        for c in clips:
            roles = [f"{c['name']}_{r}.mp4" for r in ROLES]
            ready = all((self.clips_dir / f).is_file()
                        and (self.clips_dir / f).stat().st_size > 0 for f in roles)
            out.append({"name": c["name"], "start": c["start"], "end": c["end"],
                        "roles": roles, "ready": ready})
        return out

    def set_flag(self, name: str, flagged: bool):
        if flagged:
            self.flags[name] = True
        else:
            self.flags.pop(name, None)
        self.flags_path.write_text(json.dumps(self.flags, indent=1, sort_keys=True))


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == "/":
                return self._send(200, PAGE.encode())
            if path == "/api/episodes":
                payload = {"session": state.session,
                           "cut_sheet": str(state.cut_sheet),
                           "episodes": state.episodes(),
                           "flags": state.flags}
                return self._send(200, json.dumps(payload).encode(),
                                  "application/json")
            m = re.fullmatch(r"/clip/([^/]+)/([^/]+)", path)
            if m:
                return self._serve_clip(m.group(2))
            self._send(404, b"not found", "text/plain")

        def _serve_clip(self, fname: str):
            # the filename comes from the cut sheet, but keep it inside clips_dir
            f = (state.clips_dir / fname).resolve()
            if not str(f).startswith(str(state.clips_dir.resolve())) or not f.is_file():
                return self._send(404, b"no clip", "text/plain")
            size = f.stat().st_size
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng and rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                if a:
                    start = int(a)
                if b:
                    end = min(int(b), size - 1)
                # cap each chunk so seeking stays responsive on 4K clips
                end = min(end, start + 4 * 1024 * 1024 - 1)
            length = max(0, end - start + 1)
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with f.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(262144, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return      # the browser seeked away; not an error
                    remaining -= len(chunk)

        def do_POST(self):
            if urlparse(self.path).path != "/api/flag":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            state.set_flag(str(body.get("name", "")), bool(body.get("flagged")))
            self._send(200, b'{"ok":true}', "application/json")

    return Handler


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--session", help="session / collection name under data/raw")
    p.add_argument("--cut-sheet", help="override the cut-sheet path")
    p.add_argument("--clips-dir", help="override the clips dir")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args(argv)

    session = args.session
    if not session:
        sys.exit("--session is required")
    cut_sheet = Path(args.cut_sheet) if args.cut_sheet else RAW_ROOT / session / "clips.json"
    if not cut_sheet.is_file():
        alt = OUTPUT_ROOT / session / "clips.json"
        if alt.is_file():
            cut_sheet = alt
        else:
            sys.exit(f"no cut sheet at {cut_sheet}")
    sheet = json.loads(cut_sheet.read_text())
    clips_dir = (Path(args.clips_dir) if args.clips_dir
                 else OUTPUT_ROOT / sheet.get("output_dir", f"{session}/episode_clips"))

    state = State(session, cut_sheet, clips_dir)
    eps = state.episodes()
    ready = sum(e["ready"] for e in eps)
    print(f"session   {session}")
    print(f"cut sheet {cut_sheet}  ({len(eps)} episodes)")
    print(f"clips     {clips_dir}  ({ready} ready)")
    if state.flags:
        print(f"flags     {len(state.flags)} from {state.flags_path.name}")

    port = args.port or _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    url = f"http://127.0.0.1:{port}/"
    print(f"\n  {url}\n")
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
