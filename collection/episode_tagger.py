#!/usr/bin/env python3
# Vendored from h2r_collection (github.com/ravikola02/h2r_collection, MIT) scripts/episode_tagger.py at 0783627.
"""Episode tagger — local web app: sync + tag + cut, across one or more sessions.

Computes the audio-refined sync once per session at startup (re-using
``sync_videos.load_session``), then serves a single-page UI from a local
``http.server`` (Python stdlib -- no third-party web deps) and opens it in the
browser. Each session's head MP4 is streamed to a native ``<video>`` element
(an H.264 proxy is built for HEVC footage), so playback and scrubbing are
hardware-decoded and smooth even for 4K/60 video.

Multiple sessions can be tagged into one **collection**: each session is synced
independently and shown one-by-one in a session switcher, and every episode is
cut into a single common ``episode_clips`` dir with one combined cut sheet, so
``convert_to_lerobot`` builds a single dataset. Episodes are numbered
continuously across sessions; the cut sheet records each clip's source session.
The combined cut sheet is additive: re-running with a new session appends its
episodes to the existing collection.

On **Save + Cut** the cut sheet is written and the per-episode triplets are cut
(in a background thread with live progress) into ``data/output/<collection>/
episode_clips/`` via the same frame-exact cutter ``sync_videos --config`` uses.
Each episode is cut with its own session's cameras/offsets. **Done** stops the
server (refused while a cut is running) so the pipeline can continue.

Usage::

    python3 scripts/episode_tagger.py                       # latest session
    python3 scripts/episode_tagger.py --session "video 2"
    python3 scripts/episode_tagger.py --session A --session B --collection combo
    python3 scripts/episode_tagger.py --no-open             # don't auto-open

Hotkeys (in the browser): Space play/pause | Left/Right ±1f |
Shift+Left/Right ±10f | Ctrl+Left/Right ±1s | Home/End jump to overlap
edges | I mark IN | O mark OUT | Backspace clear marks | Enter add episode |
Delete remove selected | Ctrl+S save + cut.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# scripts/ is the script's own dir at runtime, so this import resolves.
import sync_videos as syc

# Repo layout: this file lives at scripts/episode_tagger.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("EGOUMI_DATA_ROOT", REPO_ROOT / "data"))   # egoumi change: data root override
RAW_ROOT = DATA_ROOT / "raw"
OUTPUT_ROOT = DATA_ROOT / "output"

SPEED_CHOICES = (0.5, 1.0, 2.0)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def resolve_session(raw_root: Path, session_name: Optional[str]) -> Path:
    """Return a session dir under raw_root.

    An explicit ``session_name`` is treated as the literal folder name --
    no pattern matching, so arbitrary capture names (e.g. ``"kitchen_take_3"``)
    work. Otherwise fall back to the latest ``video N`` session that has a
    head camera populated.
    """
    if session_name:
        candidate = raw_root / session_name
        if not candidate.is_dir():
            raise SystemExit(f"Session folder not found: {candidate}")
        return candidate
    sessions = syc.discover_sessions(raw_root)
    if not sessions:
        raise SystemExit(f"No 'video N' sessions under {raw_root} "
                         f"(pass --session NAME to use a non-standard folder)")
    for s in reversed(sessions):
        if syc.has_reference_camera(s):
            return s
    raise SystemExit("No session has a head camera populated yet")


def fmt_time(t: float) -> str:
    if t < 0 or t != t:  # NaN
        return "--:--.---"
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:06.3f}"


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    episode_id: int
    name: str
    session_index: int                 # which SessionView (loaded this run)
    session_name: str                  # source session folder name
    in_frame: int
    out_frame: int
    task_label: str = ""
    dominant_hand: str = "right"


@dataclass
class SessionView:
    """One synced session: its cameras, scrub source, and head timeline."""
    index: int
    name: str
    session_dir: Path
    cameras: Dict[str, "syc.Camera"]   # full sync state, used at cut-on-save
    video_path: Path                   # raw head MP4 (cameras['head'].path)
    preview_path: Path                 # browser-playable video (H.264 proxy or original)
    fps: float                         # float(Fraction); display use
    fps_exact: Fraction                # exact rational; export + cut use
    n_frames: int                      # head total frames
    overlap_lo: float                  # unified-time overlap window start
    overlap_hi: float                  # unified-time overlap window end
    lo_frame: int                      # head-frame index for overlap_lo
    hi_frame: int                      # head-frame index for overlap_hi
    mode: str                          # 'head_only' or 'head_wrist'


@dataclass
class TaggerState:
    collection: str                    # combined-output name
    sessions: List[SessionView]
    out_path: Path                     # combined cut sheet
    output_dir: str                    # common clips dir, relative to data/output/
    episodes: List[Episode] = field(default_factory=list)   # editable (loaded sessions)
    carried: List[dict] = field(default_factory=list)       # passthrough clips (other sessions)


# ---------------------------------------------------------------------------
# single-page UI (served verbatim; all dynamic state comes from /api/state)
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Episode tagger</title>
<style>
  :root{
    --bg:#0e1116; --panel:#171b22; --panel2:#1e232c; --line:#2b313c;
    --fg:#e6e9ef; --muted:#8b94a3; --accent:#4f8cff; --accent2:#3a6fd8;
    --in:#2bd576; --out:#ff5c7a; --warn:#ffb454;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  .wrap{max-width:1180px;margin:0 auto;padding:14px 16px 40px}
  header{display:flex;align-items:baseline;gap:12px;margin-bottom:10px}
  header h1{font-size:16px;font-weight:600;margin:0}
  header .sub{color:var(--muted);font-size:13px}
  .badge{font-size:11px;padding:2px 8px;border-radius:999px;
         background:var(--panel2);border:1px solid var(--line);color:var(--muted)}
  .stage{background:#000;border:1px solid var(--line);border-radius:10px;
         overflow:hidden;position:relative}
  video{display:block;width:100%;max-height:60vh;background:#000}
  .statusbar{display:flex;gap:18px;flex-wrap:wrap;align-items:center;
       font-variant-numeric:tabular-nums;color:var(--muted);
       padding:8px 4px;font-size:13px}
  .statusbar b{color:var(--fg);font-weight:600}
  .scrub{position:relative;padding:6px 2px 2px}
  input[type=range]{width:100%;accent-color:var(--accent);height:22px}
  .markline{position:absolute;top:6px;bottom:2px;width:2px;pointer-events:none}
  .panel{background:var(--panel);border:1px solid var(--line);
         border-radius:10px;padding:12px 14px;margin-top:12px}
  .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .row.tight{gap:6px}
  .spacer{flex:1}
  button{background:var(--panel2);color:var(--fg);border:1px solid var(--line);
         border-radius:8px;padding:7px 12px;font-size:13px;cursor:pointer;
         transition:background .12s,border-color .12s}
  button:hover{background:#283040}
  button:active{transform:translateY(1px)}
  button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
  button.primary:hover{background:var(--accent2)}
  button.danger{border-color:#5a2731;color:#ff8ea3}
  button.ghost{background:transparent}
  button:disabled{opacity:.45;cursor:not-allowed}
  .seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
  .seg button{border:0;border-radius:0;padding:7px 12px}
  .seg button.on{background:var(--accent);color:#fff}
  label.fld{display:flex;flex-direction:column;gap:4px;font-size:12px;color:var(--muted)}
  input[type=text],select{background:#0b0e13;color:var(--fg);
         border:1px solid var(--line);border-radius:8px;padding:7px 9px;font-size:13px}
  input[type=text]{min-width:240px}
  .marks{font-variant-numeric:tabular-nums}
  .pill{padding:2px 8px;border-radius:6px;background:var(--panel2);
        border:1px solid var(--line)}
  .pill.in{color:var(--in)} .pill.out{color:var(--out)}
  table{width:100%;border-collapse:collapse;font-size:13px;
        font-variant-numeric:tabular-nums}
  th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line)}
  th{color:var(--muted);font-weight:600;font-size:12px}
  tbody tr{cursor:pointer}
  tbody tr:hover{background:var(--panel2)}
  tbody tr.sel{background:#23314d}
  tbody tr.carried{color:var(--muted);cursor:default}
  td.num,th.num{text-align:right}
  .empty{color:var(--muted);padding:14px 4px}
  .toast{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);
         background:var(--panel2);border:1px solid var(--line);border-radius:10px;
         padding:11px 16px;box-shadow:0 8px 30px rgba(0,0,0,.5);max-width:560px;
         opacity:0;pointer-events:none;transition:opacity .2s}
  .toast.show{opacity:1}
  .toast.err{border-color:#5a2731}
  .toast.ok{border-color:#234e36}
  kbd{background:#0b0e13;border:1px solid var(--line);border-bottom-width:2px;
      border-radius:5px;padding:1px 5px;font:11px ui-monospace,monospace;color:var(--muted)}
  .hint{color:var(--muted);font-size:12px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Episode tagger</h1>
    <span class="sub" id="hsub"></span>
  </header>

  <div class="panel" style="margin-top:0">
    <div class="row tight">
      <strong>session</strong>
      <button id="b-sprev" title="Previous session">◀</button>
      <select id="sess"></select>
      <button id="b-snext" title="Next session">▶</button>
      <span class="badge" id="hmode"></span>
      <span class="sub" id="svid"></span>
      <span class="spacer"></span>
      <span class="hint" id="sess-count"></span>
    </div>
  </div>

  <div class="stage" style="margin-top:12px">
    <video id="vid" preload="auto" playsinline></video>
  </div>

  <div class="statusbar">
    <span>t <b id="st-t">--:--.---</b></span>
    <span>frame <b id="st-f">0</b> / <span id="st-hi">0</span></span>
    <span>overlap <b id="st-ov">[--, --]</b></span>
    <span>fps <b id="st-fps">--</b></span>
  </div>

  <div class="scrub">
    <input id="scrub" type="range" min="0" max="100" value="0" step="1">
    <div class="markline" id="mk-in" style="background:var(--in);display:none"></div>
    <div class="markline" id="mk-out" style="background:var(--out);display:none"></div>
  </div>

  <div class="panel">
    <div class="row">
      <button id="b-home" title="Jump to overlap start (Home)">⏮</button>
      <button id="b-b10" title="-10 frames (Shift+←)">⏪ 10f</button>
      <button id="b-b1" title="-1 frame (←)">◀ 1f</button>
      <button id="b-play" class="primary" title="Play / pause (Space)">▶ Play</button>
      <button id="b-f1" title="+1 frame (→)">1f ▶</button>
      <button id="b-f10" title="+10 frames (Shift+→)">10f ⏩</button>
      <button id="b-end" title="Jump to overlap end (End)">⏭</button>
      <span class="spacer"></span>
      <span class="hint">speed</span>
      <span class="seg" id="speed"></span>
    </div>
  </div>

  <div class="panel">
    <div class="row" style="margin-bottom:10px">
      <span class="marks">in: <span class="pill in" id="m-in">---</span></span>
      <span class="marks">out: <span class="pill out" id="m-out">---</span></span>
      <button id="b-in" title="Mark IN (I)">Mark IN</button>
      <button id="b-out" title="Mark OUT (O)">Mark OUT</button>
      <button id="b-clear" class="ghost" title="Clear marks (Backspace)">Clear</button>
    </div>
    <div class="row">
      <label class="fld">task label
        <input id="f-label" type="text" placeholder="e.g. pick gear, place on peg">
      </label>
      <label class="fld">name (auto)
        <input id="f-name" type="text" placeholder="ep001" style="min-width:140px">
      </label>
      <label class="fld">hand
        <select id="f-hand"><option>right</option><option>left</option></select>
      </label>
      <span class="spacer"></span>
      <button id="b-add" class="primary" title="Add episode (Enter)">+ Add episode</button>
      <button id="b-apply" title="Apply form fields to the selected row">Apply to selected</button>
    </div>
  </div>

  <div class="panel">
    <div class="row" style="margin-bottom:6px">
      <strong>Episodes</strong>
      <span class="hint" id="ep-count"></span>
      <span class="spacer"></span>
      <button id="b-goin" title="Seek to IN of selected">Go to IN</button>
      <button id="b-goout" title="Seek to OUT of selected">Go to OUT</button>
      <button id="b-del" class="danger" title="Delete selected (Delete)">Delete</button>
    </div>
    <table>
      <thead><tr>
        <th class="num">#</th><th>name</th><th>session</th><th class="num">start</th>
        <th class="num">end</th><th class="num">dur</th><th>label</th><th>hand</th>
      </tr></thead>
      <tbody id="ep-body"></tbody>
    </table>
    <div class="empty" id="ep-empty">No episodes yet — mark IN/OUT and add one.</div>
  </div>

  <div class="panel">
    <div class="row">
      <span class="hint">Out: <span id="out-path"></span></span>
      <span class="spacer"></span>
      <button id="b-save" class="primary" title="Save cut sheet + cut clips (Ctrl+S)">
        Save + Cut</button>
      <button id="b-done" title="Stop the server and continue the pipeline">Done</button>
    </div>
    <div class="hint" style="margin-top:8px">
      <kbd>Space</kbd> play ·
      <kbd>←</kbd>/<kbd>→</kbd> ±1f · <kbd>Shift</kbd>+<kbd>←/→</kbd> ±10f ·
      <kbd>Ctrl</kbd>+<kbd>←/→</kbd> ±1s · <kbd>Home</kbd>/<kbd>End</kbd> edges ·
      <kbd>I</kbd>/<kbd>O</kbd> mark · <kbd>Enter</kbd> add · <kbd>Del</kbd> remove ·
      <kbd>Ctrl</kbd>+<kbd>S</kbd> save
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
const $ = s => document.querySelector(s);
const v = $('#vid');
let S=null, act=0, cur=0, inF=null, outF=null, eps=[], carried=[],
    selId=null, playing=false, speed=1.0, cutting=false;

const SV = () => S.sessions[act];
function fmtTime(t){
  if(t==null || t<0 || isNaN(t)) return '--:--.---';
  const m=Math.floor(t/60), s=t-60*m;
  return String(m).padStart(2,'0')+':'+s.toFixed(3).padStart(6,'0');
}
const clampF = f => Math.max(SV().lo_frame, Math.min(Math.round(f), SV().hi_frame));
const fToT = f => (f + 0.5) / SV().fps;        // mid-frame: reliable decode
const tToF = t => Math.round(t * SV().fps);
const fStart = f => f / SV().fps;              // frame's true start (for export)

function toast(msg, kind){
  const el=$('#toast'); el.textContent=msg;
  el.className='toast show '+(kind||'');
  clearTimeout(toast._t);
  toast._t=setTimeout(()=>el.className='toast '+(kind||''), kind==='err'?6000:3200);
}

function pctOf(frame){
  const span=Math.max(1, SV().hi_frame-SV().lo_frame);
  return 100*(frame-SV().lo_frame)/span;
}
function updateMarkers(){
  const mi=$('#mk-in'), mo=$('#mk-out');
  if(inF==null){mi.style.display='none'} else {mi.style.display='block';mi.style.left=pctOf(inF)+'%'}
  if(outF==null){mo.style.display='none'} else {mo.style.display='block';mo.style.left=pctOf(outF)+'%'}
  $('#m-in').textContent = inF==null?'---':('f'+inF+' · '+fmtTime(fStart(inF)));
  $('#m-out').textContent = outF==null?'---':('f'+outF+' · '+fmtTime(fStart(outF)));
}
function updateUI(){
  $('#st-t').textContent=fmtTime(fStart(cur));
  $('#st-f').textContent=cur;
  $('#scrub').value=cur;
  updateMarkers();
}

function seekFrame(f){
  cur=clampF(f);
  v.currentTime=fToT(cur);
  updateUI();
}
function step(d){ pause(); seekFrame(cur+d); }

function play(){
  if(cur>=SV().hi_frame) seekFrame(SV().lo_frame);
  v.playbackRate=speed;
  v.play().catch(()=>{}); playing=true; $('#b-play').textContent='⏸ Pause';
  requestAnimationFrame(tick);
}
function pause(){
  v.pause(); playing=false; $('#b-play').textContent='▶ Play';
}
function togglePlay(){ playing?pause():play(); }
// Drive the readout off the media clock: reliable across browsers (and
// headless), unlike requestVideoFrameCallback which needs painted frames.
function tick(){
  if(!playing) return;
  if(v.currentTime >= fStart(SV().hi_frame)){ pause(); seekFrame(SV().hi_frame); return; }
  cur=clampF(tToF(v.currentTime));
  updateUI();
  requestAnimationFrame(tick);
}

// -- session switching ------------------------------------------------------
function setActive(i, seekTo){
  if(i<0 || i>=S.sessions.length) return;
  pause(); act=i; clearMarks();
  const sv=SV();
  $('#sess').value=String(i);
  $('#hmode').textContent=sv.mode;
  $('#svid').textContent=sv.video_name;
  $('#st-fps').textContent=sv.fps.toFixed(4);
  $('#st-hi').textContent=sv.hi_frame;
  $('#st-ov').textContent='['+fmtTime(sv.overlap_lo)+', '+fmtTime(sv.overlap_hi)+']';
  $('#scrub').min=sv.lo_frame; $('#scrub').max=Math.max(sv.hi_frame, sv.lo_frame+1);
  cur = seekTo!=null ? Math.max(sv.lo_frame, Math.min(seekTo, sv.hi_frame)) : sv.lo_frame;
  v.src='/video?s='+i;
  v.addEventListener('loadedmetadata', ()=>{ seekFrame(cur); }, {once:true});
  updateUI();
}

// -- marks / episodes -------------------------------------------------------
function markIn(){ inF=cur; if(outF!=null && outF<=inF) outF=null; updateMarkers(); }
function markOut(){
  if(inF!=null && cur<=inF){ toast('OUT must be after IN.','err'); return; }
  outF=cur; updateMarkers();
}
function clearMarks(){ inF=null; outF=null; updateMarkers(); }
function nextId(){
  let mx=0;
  for(const e of eps) mx=Math.max(mx, e.episode_id);
  for(const c of carried) mx=Math.max(mx, c.episode_id);
  return mx+1;
}
function nameUsed(n){ return eps.some(e=>e.name===n) || carried.some(c=>c.name===n); }

function addEpisode(){
  if(inF==null || outF==null){ toast('Mark IN and OUT before adding.','err'); return; }
  const id=nextId();
  const name=($('#f-name').value.trim())||('ep'+String(id).padStart(3,'0'));
  if(nameUsed(name)){ toast("name '"+name+"' already used.",'err'); return; }
  eps.push({episode_id:id, name, session_index:act, session_name:SV().name,
            in_frame:inF, out_frame:outF,
            task_label:$('#f-label').value.trim(), dominant_hand:$('#f-hand').value});
  renderTable(); clearMarks(); $('#f-name').value='';
  toast('Added '+name+' to '+SV().name,'ok');
}

function epTimes(e){
  if(e._carried) return [e.start, e.end];
  const fps=S.sessions[e.session_index].fps;
  return [e.in_frame/fps, e.out_frame/fps];
}
function allRows(){
  const rows=eps.map(e=>({...e, _carried:false}))
    .concat(carried.map(c=>({...c, _carried:true})));
  rows.sort((a,b)=>a.episode_id-b.episode_id);
  return rows;
}
function renderTable(){
  const body=$('#ep-body'); body.innerHTML='';
  const rows=allRows();
  for(const e of rows){
    const [s,en]=epTimes(e);
    const tr=document.createElement('tr');
    tr.className=(e._carried?'carried':'')+(e.episode_id===selId?' sel':'');
    tr.innerHTML=`<td class="num">${String(e.episode_id).padStart(3,'0')}</td>`+
      `<td>${esc(e.name)}</td><td>${esc(e.session_name||'')}</td>`+
      `<td class="num">${s.toFixed(3)}</td>`+
      `<td class="num">${en.toFixed(3)}</td><td class="num">${(en-s).toFixed(3)}</td>`+
      `<td>${esc(e.task_label||'')}</td><td>${esc(e.dominant_hand)}</td>`;
    if(!e._carried) tr.onclick=()=>selectRow(e.episode_id);
    else tr.title='from session "'+e.session_name+'" (already cut; not editable this run)';
    body.appendChild(tr);
  }
  $('#ep-empty').style.display = rows.length?'none':'block';
  const n=eps.length, c=carried.length;
  $('#ep-count').textContent = rows.length
    ? ('· '+rows.length+(c?(' ('+n+' editable, '+c+' carried)'):''))
    : '';
}
function esc(s){ return (s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function selectRow(id){
  const e=eps.find(x=>x.episode_id===id); if(!e) return;
  selId=id;
  $('#f-label').value=e.task_label||''; $('#f-name').value=e.name; $('#f-hand').value=e.dominant_hand;
  renderTable();
}
function applyToSelected(){
  const e=eps.find(x=>x.episode_id===selId); if(!e){ toast('Select an editable row first.','err'); return; }
  const nn=$('#f-name').value.trim()||e.name;
  if(nn!==e.name && nameUsed(nn)){ toast("name '"+nn+"' in use.",'err'); return; }
  e.name=nn; e.task_label=$('#f-label').value.trim(); e.dominant_hand=$('#f-hand').value;
  renderTable();
}
function deleteSelected(){
  if(selId==null){ toast('Select an editable row first.','err'); return; }
  if(!eps.some(e=>e.episode_id===selId)){ toast('Carried episodes are not editable.','err'); return; }
  eps=eps.filter(e=>e.episode_id!==selId); selId=null; renderTable();
}
function gotoSel(which){
  const e=eps.find(x=>x.episode_id===selId); if(!e){ toast('Select an editable row first.','err'); return; }
  const tgt = which==='in'?e.in_frame:e.out_frame;
  if(e.session_index!==act) setActive(e.session_index, tgt);
  else { pause(); seekFrame(tgt); }
}

// -- save / done ------------------------------------------------------------
let dirtySaved=false;
function setBusy(b){
  cutting=b;
  $('#b-save').disabled=b; $('#b-done').disabled=b;
  $('#b-save').textContent=b?'Cutting…':'Save + Cut';
}
async function save(){
  if(cutting) return;
  setBusy(true);
  try{
    const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({episodes:eps})});
    const d=await r.json();
    if(!r.ok){ toast('Save failed: '+(d.detail||r.status),'err'); setBusy(false); return; }
    // poll progress until the background cut finishes
    while(true){
      await new Promise(res=>setTimeout(res,400));
      const st=await (await fetch('/api/save_status')).json();
      if(st.total) $('#b-save').textContent=`Cutting ${st.done}/${st.total}…`;
      if(!st.running){
        const res=st.result||{};
        if(res.error){ toast('Cut failed: '+res.error,'err'); }
        else if(res.clips===0){ toast('Wrote empty cut sheet (nothing to cut).','ok'); dirtySaved=true; }
        else {
          dirtySaved=true;
          const carriedNote=res.carried?(' (+'+res.carried+' carried)'):'';
          if(res.failures && res.failures.length)
            toast(`Saved. Cut ${res.cut}/${res.cut_total}${carriedNote}; ${res.failures.length} failed — see terminal.`,'err');
          else
            toast(`Saved + cut ${res.cut} episode(s)${carriedNote}. Sheet has ${res.clips} clip(s).`,'ok');
        }
        break;
      }
    }
  }catch(e){ toast('Save failed: '+e,'err'); }
  finally{ setBusy(false); }
}
async function done(){
  if(cutting){ toast('A cut is still running — wait for it to finish.','err'); return; }
  if((eps.length) && !dirtySaved &&
     !confirm('You have episodes that were not Saved + Cut. Continue anyway?')) return;
  try{
    const r=await fetch('/api/shutdown',{method:'POST'});
    if(!r.ok){ const d=await r.json(); toast(d.detail||'cannot stop now','err'); return; }
  }catch(e){}
  document.body.innerHTML='<div class="wrap"><div class="panel">'+
    'Tagger closed — the pipeline is continuing. You can close this tab.</div></div>';
}

// -- speed control ----------------------------------------------------------
function buildSpeeds(){
  const box=$('#speed');
  for(const sp of [0.5,1.0,2.0]){
    const b=document.createElement('button');
    b.textContent=sp+'×'; b.dataset.sp=sp;
    if(sp===speed) b.className='on';
    b.onclick=()=>{ speed=sp; v.playbackRate=sp;
      [...box.children].forEach(c=>c.className=(+c.dataset.sp===sp)?'on':''); };
    box.appendChild(b);
  }
}

// -- wiring -----------------------------------------------------------------
function typing(){ const t=document.activeElement; return t && (t.tagName==='INPUT'||t.tagName==='SELECT'); }
function keys(e){
  if(e.key==='s' && (e.ctrlKey||e.metaKey)){ e.preventDefault(); save(); return; }
  if(typing()) return;
  const oneSec=Math.round(SV().fps);
  switch(e.key){
    case ' ': e.preventDefault(); togglePlay(); break;
    case 'ArrowLeft':  step(e.shiftKey?-10:(e.ctrlKey?-oneSec:-1)); break;
    case 'ArrowRight': step(e.shiftKey? 10:(e.ctrlKey? oneSec: 1)); break;
    case 'Home': pause(); seekFrame(SV().lo_frame); break;
    case 'End':  pause(); seekFrame(SV().hi_frame); break;
    case 'i': case 'I': markIn(); break;
    case 'o': case 'O': markOut(); break;
    case 'Backspace': clearMarks(); break;
    case 'Enter': addEpisode(); break;
    case 'Delete': deleteSelected(); break;
  }
}

async function init(){
  S=await (await fetch('/api/state')).json();
  $('#hsub').textContent='collection: '+S.collection;
  $('#out-path').textContent=S.out_path;
  eps=S.episodes||[]; carried=S.carried||[];

  const sel=$('#sess');
  S.sessions.forEach(s=>{
    const o=document.createElement('option');
    o.value=String(s.index); o.textContent=`${s.index+1}. ${s.name} [${s.mode}]`;
    sel.appendChild(o);
  });
  $('#sess-count').textContent = S.sessions.length+' session'+(S.sessions.length>1?'s':'');
  sel.disabled = S.sessions.length<2;
  $('#b-sprev').disabled = $('#b-snext').disabled = S.sessions.length<2;

  buildSpeeds(); renderTable();

  sel.addEventListener('change', e=>setActive(+e.target.value));
  $('#b-sprev').onclick=()=>setActive(act-1);
  $('#b-snext').onclick=()=>setActive(act+1);

  $('#scrub').addEventListener('input', e=>{ pause(); seekFrame(+e.target.value); });
  $('#b-home').onclick=()=>{pause();seekFrame(SV().lo_frame)};
  $('#b-end').onclick=()=>{pause();seekFrame(SV().hi_frame)};
  $('#b-b10').onclick=()=>step(-10); $('#b-b1').onclick=()=>step(-1);
  $('#b-f1').onclick=()=>step(1);    $('#b-f10').onclick=()=>step(10);
  $('#b-play').onclick=togglePlay;
  $('#b-in').onclick=markIn; $('#b-out').onclick=markOut; $('#b-clear').onclick=clearMarks;
  $('#b-add').onclick=addEpisode; $('#b-apply').onclick=applyToSelected;
  $('#b-del').onclick=deleteSelected;
  $('#b-goin').onclick=()=>gotoSel('in'); $('#b-goout').onclick=()=>gotoSel('out');
  $('#b-save').onclick=save; $('#b-done').onclick=done;
  document.addEventListener('keydown', keys);
  v.addEventListener('ended', ()=>pause());

  setActive(0);   // loads the first session's video
}
init();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# cut sheet (pure functions)
# ---------------------------------------------------------------------------

def build_clips_json(state: TaggerState) -> dict:
    """Serialise all episodes into one combined cut sheet.

    Editable episodes (from loaded sessions) are converted to ``start``/``end``
    seconds with their own session's exact rational fps; head is the reference
    camera so those are unified-time seconds. Carried entries (clips from
    sessions not loaded this run) are passed through verbatim so the combined
    sheet stays additive. Each clip records its source ``session``.
    """
    by_index = {s.index: s for s in state.sessions}
    clips = []
    for ep in state.episodes:
        fps_exact = by_index[ep.session_index].fps_exact
        clips.append({
            "name": ep.name,
            "start": round(float(Fraction(ep.in_frame) / fps_exact), 6),
            "end": round(float(Fraction(ep.out_frame) / fps_exact), 6),
            "episode_id": ep.episode_id,
            "task_label": ep.task_label,
            "dominant_hand": ep.dominant_hand,
            "session": ep.session_name,
        })
    clips.extend(state.carried)
    clips.sort(key=lambda c: c.get("episode_id", 0))
    return {"output_dir": state.output_dir, "clips": clips}


def split_existing_cut_sheet(
    path: Path, sessions: List[SessionView]
) -> Tuple[List[Episode], List[dict]]:
    """Re-hydrate an existing combined cut sheet against the loaded sessions.

    Returns ``(episodes, carried)``: clips whose ``session`` is loaded this run
    become editable ``Episode`` objects (seconds -> frames via that session's
    fps); clips from any other session are carried through verbatim so they
    survive the next save. A clip with no ``session`` field (an old
    single-session sheet) is attributed to the sole loaded session.
    """
    by_name = {s.name: s for s in sessions}
    cfg = json.loads(path.read_text())
    episodes: List[Episode] = []
    carried: List[dict] = []
    for idx, c in enumerate(cfg.get("clips", [])):
        sess = c.get("session")
        if sess is None and len(sessions) == 1:
            sess = sessions[0].name
        sv = by_name.get(sess)
        if sv is None:
            carried.append(c)
            continue
        fps = float(sv.fps_exact)
        episodes.append(Episode(
            episode_id=int(c.get("episode_id", idx + 1)),
            name=c.get("name", f"ep{idx + 1:03d}"),
            session_index=sv.index,
            session_name=sv.name,
            in_frame=int(round(float(c["start"]) * fps)),
            out_frame=int(round(float(c["end"]) * fps)),
            task_label=c.get("task_label", ""),
            dominant_hand=c.get("dominant_hand", "right"),
        ))
    return episodes, carried


def cut_episodes(state: TaggerState, out_base: Path, progress=None) -> List[str]:
    """Cut each editable episode's triplet with its own session's cameras.

    Returns the list of episode names that failed (empty on full success).
    ``progress(done, total, name)`` is called before each cut for UI ticks.
    Carried clips are already on disk from a previous run and are not re-cut.
    """
    by_index = {s.index: s for s in state.sessions}
    out_base.mkdir(parents=True, exist_ok=True)
    failures: List[str] = []
    total = len(state.episodes)
    for i, ep in enumerate(state.episodes):
        if progress is not None:
            progress(i, total, ep.name)
        sv = by_index[ep.session_index]
        start = float(Fraction(ep.in_frame) / sv.fps_exact)
        end = float(Fraction(ep.out_frame) / sv.fps_exact)
        try:
            syc.create_clip_batch(sv.cameras, start, end, out_base, ep.name,
                                  verbose=False)
        except Exception as exc:  # noqa: BLE001 — isolate per-episode failures
            print(f"ERROR cutting {ep.name}: {exc}", file=sys.stderr)
            failures.append(f"{ep.name}: {exc}")
    if progress is not None:
        progress(total, total, "")
    return failures


# ---------------------------------------------------------------------------
# preview proxy (HEVC -> browser-playable H.264)
# ---------------------------------------------------------------------------

# Codecs a browser <video> element decodes reliably across platforms. DJI
# Osmo footage is HEVC, which Chrome/Firefox (esp. on Linux) generally won't
# play in a <video> tag -- hence the H.264 preview proxy below.
BROWSER_OK_CODECS = {"h264"}
PREVIEW_HEIGHT = 720


def _video_codec(path: Path) -> str:
    """codec_name of the first video stream (e.g. 'hevc', 'h264'), or ''."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1",
             str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        return out.splitlines()[0] if out else ""
    except subprocess.CalledProcessError:
        return ""


def _build_preview(src: Path, dst: Path, codec: str) -> bool:
    """Transcode ``src`` to a downscaled H.264 ``dst`` for browser playback.

    Tries GPU (NVDEC + NVENC) first for HEVC sources -- ~7x realtime on this
    box vs ~0.5x for software HEVC decode -- and falls back to libx264 if the
    GPU path is unavailable or fails. Keeps the source frame rate (``-vsync``
    default copies frame timing), maps only the real video stream (DJI MP4s
    carry a junk thumbnail stream), and writes to a temp file renamed on
    success so a half-built proxy is never cached.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    tail = ["-map", "0:v:0", "-an", "-dn", "-sn", "-movflags", "+faststart"]
    attempts = []
    if codec == "hevc":
        attempts.append(("GPU", [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-c:v", "hevc_cuvid", "-i", str(src), *tail,
            "-vf", f"scale_cuda=-2:{PREVIEW_HEIGHT}",
            "-c:v", "h264_nvenc", "-preset", "fast", "-b:v", "6M", str(tmp)]))
    attempts.append(("CPU", [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
        "-i", str(src), *tail,
        "-vf", f"scale=-2:{PREVIEW_HEIGHT}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", str(tmp)]))

    for label, cmd in attempts:
        print(f"  transcoding preview ({label})…", file=sys.stderr)
        rc = subprocess.run(cmd).returncode
        if rc == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            tmp.replace(dst)
            return True
        print(f"  preview {label} path failed (rc={rc})", file=sys.stderr)
    tmp.unlink(missing_ok=True)
    return False


def ensure_preview(src: Path, session_name: str) -> Path:
    """Return a browser-playable video for ``src``.

    If ``src`` is already H.264 it is served directly. Otherwise a cached,
    downscaled H.264 proxy is built once at ``data/output/<session>/
    _preview_head.mp4`` (reused while it is newer than the source). The proxy
    keeps the source's exact frame rate, so frame index N maps to the same
    instant -- tagging stays frame-accurate. Cutting always uses the original
    via ``sync_videos``, never the proxy.
    """
    codec = _video_codec(src)
    if codec in BROWSER_OK_CODECS:
        return src
    proxy = OUTPUT_ROOT / session_name / "_preview_head.mp4"
    if proxy.is_file() and proxy.stat().st_mtime >= src.stat().st_mtime:
        print(f"Using cached H.264 preview proxy {proxy}", file=sys.stderr)
        return proxy
    print(f"[{session_name}] head video is {codec.upper() or 'a non-H.264 codec'}; "
          f"browsers can't play it in a <video> tag. Building a downscaled "
          f"H.264 preview proxy once (cached for re-runs)…", file=sys.stderr)
    if _build_preview(src, proxy, codec):
        print(f"Preview proxy ready: {proxy}", file=sys.stderr)
        return proxy
    print(f"WARN  preview proxy build failed; serving the original "
          f"{codec.upper()} — it may not display in the browser.",
          file=sys.stderr)
    return src


# ---------------------------------------------------------------------------
# state assembly
# ---------------------------------------------------------------------------

def init_session_view(index: int, session_dir: Path) -> SessionView:
    """Sync one session and build its scrub view (head timeline + preview)."""
    mode = syc.detect_session_mode(session_dir)
    if mode == "head_only":
        print(f"Loading session '{session_dir.name}' (head_only -- "
              f"no sync needed)…", file=sys.stderr)
        cameras = syc.load_session(session_dir, use_audio=False)
    else:
        print(f"Syncing session '{session_dir.name}' (head_wrist)…",
              file=sys.stderr)
        cameras = syc.load_session(session_dir, use_audio=True)
        syc.write_sync_json(session_dir, cameras)
    head = cameras["head"]
    fps_exact = Fraction(head.fps_num, head.fps_den)
    lo, hi = syc.overlap_window(cameras)
    lo_frame = max(0, int(round(lo * float(fps_exact))))
    hi_frame = min(head.n_frames - 1, int(round(hi * float(fps_exact))))
    preview_path = ensure_preview(head.path, session_dir.name)
    return SessionView(
        index=index, name=session_dir.name, session_dir=session_dir,
        cameras=cameras, video_path=head.path, preview_path=preview_path,
        fps=float(fps_exact), fps_exact=fps_exact, n_frames=head.n_frames,
        overlap_lo=float(lo), overlap_hi=float(hi),
        lo_frame=lo_frame, hi_frame=hi_frame, mode=mode,
    )


def init_state(session_dirs: List[Path], collection: str, out: Path,
               output_dir: str) -> TaggerState:
    """Sync every session and assemble the tagger state.

    Loads an existing combined cut sheet at ``out`` (if any) and splits it into
    editable episodes (for the loaded sessions) and carried-through clips from
    other sessions, so re-running appends rather than clobbers.
    """
    sessions = [init_session_view(i, d) for i, d in enumerate(session_dirs)]
    episodes: List[Episode] = []
    carried: List[dict] = []
    if out.is_file():
        try:
            episodes, carried = split_existing_cut_sheet(out, sessions)
            print(f"Loaded cut sheet {out}: {len(episodes)} editable episode(s)"
                  f", {len(carried)} carried from other sessions",
                  file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN  could not parse existing {out}: {exc}",
                  file=sys.stderr)
    return TaggerState(
        collection=collection, sessions=sessions, out_path=out,
        output_dir=output_dir, episodes=episodes, carried=carried,
    )


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------

def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


_RANGE_RE = re.compile(r"bytes=(\d+)-(\d*)")


def _state_payload(state: TaggerState) -> dict:
    return {
        "collection": state.collection,
        "out_path": str(state.out_path),
        "output_dir": state.output_dir,
        "sessions": [
            {
                "index": s.index, "name": s.name,
                "video_name": s.video_path.name, "mode": s.mode,
                "fps": s.fps, "n_frames": s.n_frames,
                "lo_frame": s.lo_frame, "hi_frame": s.hi_frame,
                "overlap_lo": s.overlap_lo, "overlap_hi": s.overlap_hi,
            }
            for s in state.sessions
        ],
        "episodes": [
            {
                "episode_id": ep.episode_id, "name": ep.name,
                "session_index": ep.session_index, "session_name": ep.session_name,
                "in_frame": ep.in_frame, "out_frame": ep.out_frame,
                "task_label": ep.task_label, "dominant_hand": ep.dominant_hand,
            }
            for ep in state.episodes
        ],
        "carried": [
            {
                "episode_id": c.get("episode_id"), "name": c.get("name"),
                "session_name": c.get("session", ""),
                "start": c.get("start"), "end": c.get("end"),
                "task_label": c.get("task_label", ""),
                "dominant_hand": c.get("dominant_hand", "right"),
            }
            for c in state.carried
        ],
    }


def serve(state: TaggerState, port: Optional[int] = None,
          open_browser: bool = True) -> None:
    """Run the tagger web app and block until the operator clicks Done.

    Uses the Python standard library's ``http.server`` (no third-party web
    stack). ``ThreadingHTTPServer`` gives one thread per request. Save + Cut
    runs in a background thread tracked by ``job`` so a long multi-session cut
    reports live progress, survives a closed tab, and can't be aborted by Done
    (which is refused while a cut is in flight).
    """
    done = threading.Event()
    job = {"running": False, "total": 0, "done": 0, "current": "", "result": None}
    job_lock = threading.Lock()

    def run_cut(payload_eps: list) -> None:
        try:
            by_index = {s.index: s for s in state.sessions}
            episodes = []
            for c in payload_eps:
                si = int(c["session_index"])
                if si not in by_index:
                    raise ValueError(f"unknown session_index {si}")
                episodes.append(Episode(
                    episode_id=int(c["episode_id"]),
                    name=str(c["name"]),
                    session_index=si,
                    session_name=by_index[si].name,
                    in_frame=int(c["in_frame"]),
                    out_frame=int(c["out_frame"]),
                    task_label=str(c.get("task_label", "")),
                    dominant_hand=str(c.get("dominant_hand", "right")),
                ))
            state.episodes = episodes
            cfg = build_clips_json(state)
            # Write the cut sheet *before* cutting, so the authored episodes are
            # never lost even if the cut is interrupted.
            state.out_path.parent.mkdir(parents=True, exist_ok=True)
            state.out_path.write_text(json.dumps(cfg, indent=2))

            def prog(d, total, name):
                with job_lock:
                    job.update(done=d, total=total, current=name)

            out_base = OUTPUT_ROOT / state.output_dir
            failures = cut_episodes(state, out_base, progress=prog)
            result = {
                "clips": len(cfg["clips"]),
                "cut_total": len(episodes),
                "cut": len(episodes) - len(failures),
                "failures": failures,
                "carried": len(state.carried),
                "out_path": str(state.out_path),
            }
        except (KeyError, ValueError, TypeError) as exc:
            result = {"error": f"bad episode: {exc}"}
        except Exception as exc:  # noqa: BLE001 — report, don't crash the thread
            result = {"error": str(exc)}
        with job_lock:
            job.update(running=False, current="", result=result)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a):  # silence per-request stderr logging
            pass

        # -- helpers ----------------------------------------------------
        def _send_bytes(self, body: bytes, status: int = 200,
                        content_type: str = "application/octet-stream",
                        extra: Optional[dict] = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def _send_json(self, obj: dict, status: int = 200) -> None:
            self._send_bytes(json.dumps(obj).encode(), status,
                             "application/json")

        def _empty(self, status: int) -> None:
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _query(self) -> dict:
            if "?" not in self.path:
                return {}
            q = {}
            for part in self.path.split("?", 1)[1].split("&"):
                if "=" in part:
                    k, val = part.split("=", 1)
                    q[k] = val
            return q

        def _serve_video(self) -> None:
            try:
                idx = int(self._query().get("s", "0"))
            except ValueError:
                idx = 0
            if not (0 <= idx < len(state.sessions)):
                self._empty(404)
                return
            path = state.sessions[idx].preview_path
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            rng = self.headers.get("Range")
            if rng:
                m = _RANGE_RE.match(rng)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                    end = min(end, size - 1)
                    status = 206
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break  # browser aborted this range (normal on seek)
                    remaining -= len(chunk)

        # -- routes -----------------------------------------------------
        def do_GET(self):  # noqa: N802 — http.server API
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send_bytes(INDEX_HTML.encode(), 200,
                                 "text/html; charset=utf-8")
            elif path == "/favicon.ico":
                self._empty(204)
            elif path == "/video":
                self._serve_video()
            elif path == "/api/state":
                self._send_json(_state_payload(state))
            elif path == "/api/save_status":
                with job_lock:
                    self._send_json(dict(job))
            else:
                self._empty(404)

        def do_POST(self):  # noqa: N802 — http.server API
            path = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            if path == "/api/save":
                with job_lock:
                    if job["running"]:
                        self._send_json({"detail": "a cut is already running"}, 409)
                        return
                    job.update(running=True, total=0, done=0, current="",
                               result=None)
                try:
                    payload = json.loads(raw or b"{}")
                except json.JSONDecodeError as exc:
                    with job_lock:
                        job["running"] = False
                    self._send_json({"detail": f"bad JSON: {exc}"}, 400)
                    return
                threading.Thread(
                    target=run_cut, args=(payload.get("episodes", []),),
                    daemon=True).start()
                self._send_json({"started": True})
            elif path == "/api/shutdown":
                with job_lock:
                    if job["running"]:
                        self._send_json(
                            {"detail": "cut in progress — wait for it"}, 409)
                        return
                done.set()
                self._send_json({"ok": True})
            else:
                self._empty(404)

    port = port or _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/"
    print(f"\n  Episode tagger UI →  {url}")
    print("  Mark episodes, Save + Cut, then click Done to continue.\n")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 — headless / no browser is fine
            pass

    try:
        done.wait()
    except KeyboardInterrupt:
        pass
    httpd.shutdown()
    thread.join(timeout=5)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--session", action="append", dest="sessions",
                   help="session folder name (repeatable). Default: latest "
                        "under data/raw/. Multiple sessions require --collection.")
    p.add_argument("--collection",
                   help="name of the combined output under data/output/. With "
                        "one session, defaults to that session's name.")
    p.add_argument("--out", help="combined cut-sheet path "
                                  "(default derived from the collection)")
    p.add_argument("--output-dir", default=None,
                   help="output_dir field written into the cut sheet — the "
                        "subfolder of data/output/ to cut clips into. "
                        "Default: '<collection>/episode_clips'.")
    p.add_argument("--raw-root", default=str(RAW_ROOT),
                   help="override the data/raw root")
    p.add_argument("--port", type=int, default=None,
                   help="port for the local UI server (default: a free port)")
    p.add_argument("--no-open", action="store_true",
                   help="don't auto-open a browser; just print the URL")
    args = p.parse_args(argv)

    raw_root = Path(args.raw_root)
    names = args.sessions if args.sessions else [None]
    session_dirs = [resolve_session(raw_root, n) for n in names]
    multi = len(session_dirs) > 1

    collection = args.collection or (session_dirs[0].name if not multi else None)
    if collection is None:
        raise SystemExit("Multiple --session require --collection NAME for the "
                         "combined output.")

    if args.out:
        out = Path(args.out)
    elif args.collection or multi:
        out = OUTPUT_ROOT / collection / "clips.json"
    else:
        # Single bare session keeps the legacy cut-sheet location.
        out = session_dirs[0] / "clips.json"
    output_dir = args.output_dir or f"{collection}/episode_clips"

    state = init_state(session_dirs, collection, out, output_dir)

    print(f"\nCollection '{collection}': {len(state.sessions)} session(s)")
    for s in state.sessions:
        win = "Overlap" if s.mode == "head_wrist" else "Head"
        print(f"  [{s.index}] {s.name}: {s.n_frames} frames, fps={s.fps:.4f}, "
              f"{s.mode}; {win} [{s.overlap_lo:.3f}, {s.overlap_hi:.3f}]s "
              f"-> frames [{s.lo_frame}, {s.hi_frame}]")
    print(f"Cut sheet: {state.out_path}")
    if state.carried:
        print(f"Carrying {len(state.carried)} episode(s) from other sessions.")

    serve(state, port=args.port, open_browser=not args.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
