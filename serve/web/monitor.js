"use strict";
const $ = id => document.getElementById(id);
let selected = null, detail = null, pane = "input", records = [], state = null, operating = false, stamp = "", actionError = "";
// #142: inside the web app's Requests tab, the app's header, theme and saved API key stand in for this page's own
let host = null;
try { if (window.parent !== window) host = parent.document.documentElement; } catch (e) { /* framed by another site */ }
let apiKey = sessionStorage.getItem("strata.monitor.key") || (host && JSON.parse(localStorage.getItem("strata.apikey") || '""')) || "";
document.documentElement.dataset.theme = localStorage.getItem("strata.theme") || "dark";
if (host) {
  const sync = () => { document.documentElement.dataset.theme = host.dataset.theme; };
  document.documentElement.classList.add("embedded");
  sync();
  new MutationObserver(sync).observe(host, {attributes: true, attributeFilter: ["data-theme"]});
}
const seconds = n => typeof n === "number" ? `${n.toFixed(2)} s` : "—";
const speed = n => typeof n === "number" ? n.toFixed(1) : "—";
const active = r => !["completed", "error", "disconnected"].includes(r.state);
const isAudio = r => (r.path || "").includes("/audio/");
const audioTask = r => (r.path || "").includes("/audio/translations") ? "translate" :
  (r.path || "").includes("/audio/transcriptions") ? "transcribe" : null;
const millis = n => typeof n === "number" ? `${(n / 1000).toFixed(2)} s` : "—";
function text(id, value) { const el = $(id); if (el.textContent !== String(value)) el.textContent = value; }
async function api(path, body) {
  const response = await fetch(path, {cache:"no-store", headers:{...(apiKey ? {Authorization:`Bearer ${apiKey}`} : {}), ...(body !== undefined ? {"Content-Type":"application/json"} : {})}, ...(body !== undefined ? {method:"POST", body:JSON.stringify(body)} : {})});
  const value = await response.json();
  if (!response.ok) {
    if (response.status === 401) $("auth").hidden = false;
    throw new Error(value.error?.message || `HTTP ${response.status}`);
  }
  return value;
}
function notice(message) { $("error").hidden = !message; text("error", message || ""); }
function list() {
  const query = $("search").value.toLowerCase(), filter = $("filter").value;
  const visible = records.filter(r => `${r.id} ${r.path} ${r.model}`.toLowerCase().includes(query) &&
    (filter === "all" ? true : filter === "active" ? active(r) : filter === "audio" ? isAudio(r) : r.state === filter));
  const nextStamp = JSON.stringify([visible, selected]);
  if (stamp === nextStamp) return;
  stamp = nextStamp;
  const focusId = document.activeElement?.dataset.request;
  $("requests").replaceChildren();
  if (!visible.length) {
    const empty = document.createElement("div"); empty.className = "empty";
    empty.textContent = records.length ? "No matching requests." : "No requests yet. API calls will appear here automatically.";
    $("requests").append(empty);
  }
  for (const r of visible) {
    const button = document.createElement("button"); button.className = `request${selected === r.id ? " selected" : ""}`;
    button.dataset.request = r.id; button.setAttribute("aria-label", `Request ${r.id}`); button.setAttribute("aria-pressed", String(selected === r.id));
    const row = document.createElement("div"); row.className = "row";
    const time = document.createElement("span"); time.textContent = new Date(r.started_at * 1000).toLocaleTimeString();
    const tag = document.createElement("span"); tag.className = `tag ${r.state}`;
    tag.textContent = r.phase || r.state;   // queued / encode / prefill / generate while it runs, then completed/error
    row.append(time, tag);
    const endpoint = document.createElement("div"); endpoint.className = "endpoint"; endpoint.textContent = `POST ${r.path}`;
    const meta = document.createElement("div"); meta.className = "small";
    meta.textContent = `${r.id} · ${seconds(r.wallclock_s)} · ${r.stream ? "stream" : "JSON response"}${r.http_status ? ` · HTTP ${r.http_status}` : ""}`;
    button.append(row, endpoint, meta);
    button.onclick = async () => { selected = r.id; detail = null; list(); await showDetail(); };
    $("requests").append(button);
    if (focusId === r.id) button.focus({preventScroll:true});
  }
}
function content() {
  if (!detail) return;
  let value = detail[pane] || (pane === "response" && detail.stream ? "Streaming response: see Output and the request error, if any." : "No content.");
  if (pane === "output") { try { value = JSON.stringify(JSON.parse(value), null, 2); } catch (_) {} }
  text("content", value);
  const note = pane === "output" && detail.error ? "Raw model output retained for diagnosis. The API returned an error." : pane === "input" ? "Original request body." : pane === "output" ? "Model answer. JSON is formatted here for readability." : pane === "reasoning" ? "Separate reasoning content, when enabled." : "API response body.";
  text("content-note", note + (detail[`${pane}_truncated`] ? " Monitor capture truncated at 256K characters; the API response was not shortened." : ""));
}
async function showDetail() {
  if (!selected) return;
  const id = selected;
  const value = await api(`/api/requests?id=${encodeURIComponent(id)}`);
  if (selected !== id) return;
  detail = value; $("selection").hidden = false; $("no-selection").hidden = true;
  text("detail-title", `${value.id} · ${value.path}`);
  text("detail-status", value.state); $("detail-status").className = `tag ${value.state}`;
  text("detail-time", new Date(value.started_at * 1000).toLocaleString());
  text("detail-wall", seconds(value.wallclock_s ?? ((Date.now() / 1000) - value.started_at)));
  text("detail-load", seconds(value.load_s)); text("detail-queue", seconds(value.queue_s)); text("detail-first", seconds(value.first_token_s));
  text("detail-prompt", value.usage?.prompt_tokens ?? value.usage?.input_tokens ?? "—");
  text("detail-tokens", value.usage?.completion_tokens ?? value.usage?.output_tokens ?? "—");
  text("detail-speed", speed(value.timings?.predicted_per_second)); text("detail-format", value.response_format || "text");
  const a = value.audio || {};
  text("detail-audio-duration", typeof a.duration_s === "number" ? `${a.duration_s.toFixed(2)} s` : "—");
  text("detail-audio-rows", a.rows ?? "—");
  text("detail-audio-encode", millis(a.encode_ms));
  text("detail-audio-prefill", millis(a.prefill_ms));       // the decoder reads the prompt, audio rows included
  text("detail-audio-decode", millis(a.decode_ms));
  text("detail-audio-device", a.device || "—");
  text("detail-audio-rtf", typeof a.rtf === "number" ? `${a.rtf.toFixed(2)}×` : "—");
  text("detail-audio-task", a.task || audioTask(value) || "—");
  text("detail-audio-language", a.language || "—");
  $("detail-error").hidden = !value.error; text("detail-error", value.error?.message || ""); content();
}
async function refresh() {
  try {
    const [status, history] = await Promise.all([api("/v1/status"), api("/api/requests")]);
    state = status; records = history.requests;
    text("retention-note", `Last ${history.retention ?? 50} requests kept in memory until restart. Inputs and outputs stay on this machine.`);
    status.loaded = history.loaded; status.auto_load = history.auto_load;
    if (selected && !records.some(r => r.id === selected)) {
      selected = detail = null; $("selection").hidden = true; $("no-selection").hidden = false;
    }
    const inflight = Math.max(status.activity.in_flight, records.filter(active).length);
    const loading = records.some(r => r.state === "loading");
    text("loaded", loading ? "Loading…" : inflight ? "In use" : status.loaded ? "Loaded" : "Unloaded");
    $("model-state").className = `value ${inflight || loading ? "busy" : status.loaded ? "loaded" : ""}`;
    text("model", status.model); text("count", records.length);
    text("active", inflight ? `${inflight} active / queued` : status.auto_load ? "Loads automatically on the next request" : "No active requests");
    const last = records.find(r => !active(r));
    text("wall", last ? seconds(last.wallclock_s) : "—"); text("speed", speed(last?.timings?.predicted_per_second));
    const audio = status.audio || {};
    const audioRecords = records.filter(isAudio);
    const transcribe = audio.requests?.transcribe ?? audioRecords.filter(r => audioTask(r) === "transcribe").length;
    const translate = audio.requests?.translate ?? audioRecords.filter(r => audioTask(r) === "translate").length;
    const lastAudio = task => audio.last && (!task || audio.last.task === task) ? audio.last :
      audioRecords.filter(r => !task || audioTask(r) === task).map(r => r.audio).find(Boolean);
    const audioLast = task => {
      const x = lastAudio(task);
      if (!x) return "No request yet";
      const bits = [];
      if (typeof x.duration_s === "number") bits.push(`${x.duration_s.toFixed(1)} s audio`);
      if (typeof x.encode_ms === "number") bits.push(`${(x.encode_ms / 1000).toFixed(1)} s encode`);
      if (bits.length) return bits.join(" · ");
      return task ? `${task} done` : "Done";
    };
    const audioState = audio.encoder || (audio.enabled === false ? "off" : audioRecords.length ? "idle" : "");
    text("audio-state", audioState ? audioState[0].toUpperCase() + audioState.slice(1) : "—");
    $("audio-encoder").className = `value ${audioState === "running" || audioState === "busy" ? "loaded" : ""}`;
    text("audio-model", audio.model || "—");
    // Device, live GPU utilization, VRAM, and while a long file runs: what it has done so far.
    const gpuBits = [];
    if (audio.device) gpuBits.push(audio.device);
    if (audio.encoding) {
      const done = typeof audio.rows_done === "number" ? audio.rows_done : 0;
      const totalRows = typeof audio.rows_total === "number" ? audio.rows_total : null;
      let line = `Encoding… ${Math.round((audio.encode_ms_live || 0) / 1000)} s`;
      line += totalRows ? `, ${done}/${totalRows} rows` : done ? `, ${done} rows` : "";
      if (typeof audio.gpu_util_pct === "number") line += `, GPU ${audio.gpu_util_pct}%`;
      gpuBits.unshift(line);
    } else {
      if (typeof audio.gpu_util_pct === "number") gpuBits.push(`GPU ${audio.gpu_util_pct}%`);
      if (typeof audio.vram_mib === "number") gpuBits.push(`${(audio.vram_mib / 1024).toFixed(1)} GB VRAM`);
    }
    text("audio-live", gpuBits.length ? gpuBits.join(" · ") : "—");
    text("audio-transcribe", transcribe); text("audio-transcribe-last", audioLast("transcribe"));
    text("audio-translate", translate); text("audio-translate-last", audioLast("translate"));
    const rtf = lastAudio()?.rtf;
    text("audio-rtf", typeof rtf === "number" ? `${rtf.toFixed(2)}×` : "—");
    const total = audio.audio_seconds ?? audioRecords.reduce((n, r) => n + (r.audio?.duration_s || 0), 0);
    text("audio-total", typeof total === "number" && total > 0 ? `${total.toFixed(1)} s processed` : "Audio processed");
    text("audio-hint", !audio.enabled && !audioRecords.length ? "No audio requests yet" :
      `${audioRecords.length} recent audio request${audioRecords.length === 1 ? "" : "s"}`);
    $("load").disabled = operating || Boolean(inflight) || status.loaded; $("unload").disabled = operating || Boolean(inflight) || !status.loaded;
    list(); if (selected) await showDetail();
    text("updated", `Live · updated ${new Date().toLocaleTimeString()}`);
    if (!operating) notice(actionError);
  } catch (error) { notice(error.message); text("updated", "Disconnected · retrying"); $("load").disabled = $("unload").disabled = true; }
}
async function control(load) {
  actionError = "";
  operating = true; $("load").disabled = $("unload").disabled = true;
  text("loaded", load ? "Loading…" : "Unloading…");
  try { await api(load ? "/load" : "/unload", {}); notice(""); }
  catch (error) { actionError = error.message; notice(actionError); }
  finally { operating = false; await refresh(); }
}
$("load").onclick = () => control(true); $("unload").onclick = () => control(false);
$("key").onclick = () => { $("auth").hidden = !$("auth").hidden; if (!$("auth").hidden) $("api-key").focus(); };
$("auth").onsubmit = async event => { event.preventDefault(); apiKey = $("api-key").value.trim(); sessionStorage.setItem("strata.monitor.key", apiKey); $("auth").hidden = true; await refresh(); };
$("theme").onclick = () => { const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = theme; localStorage.setItem("strata.theme", theme); };
$("search").oninput = list; $("filter").onchange = list;
document.querySelectorAll("[data-pane]").forEach(button => { button.onclick = () => { pane = button.dataset.pane; document.querySelectorAll("[data-pane]").forEach(b => b.setAttribute("aria-selected", String(b === button))); content(); }; });
$("copy").onclick = async () => { try { await navigator.clipboard.writeText($("content").textContent); text("copy", "Copied"); setTimeout(() => text("copy", "Copy"), 1500); } catch (_) { notice("Clipboard unavailable; select the text to copy it."); } };
async function poll() { await refresh(); setTimeout(poll, 2000); }
poll();
