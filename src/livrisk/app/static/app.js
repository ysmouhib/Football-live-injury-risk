// Bench screen client (docs/06_UI_SPEC.md). Every visible text comes from the server: labels
// from /api/wording, values from the payloads. Data is written with textContent only.
"use strict";

const params = new URLSearchParams(window.location.search);
const role = params.get("role") || "analyst";
const token = params.get("token");
let T = {};
let lastPayload = null;
let selected = null;
let retryMs = 1000;

function query(extra) {
  const q = new URLSearchParams({ role: role });
  if (token) q.set("token", token);
  if (extra) Object.entries(extra).forEach(([k, v]) => q.set(k, v));
  return q.toString();
}

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  Object.entries(attrs || {}).forEach(([k, v]) => {
    if (k === "text") node.textContent = v;
    else if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  });
  (children || []).forEach((c) => node.appendChild(c));
  return node;
}

function setText(id, text) {
  document.getElementById(id).textContent = text || "";
}

function table(headers, rows) {
  const head = el("tr", {}, headers.map((h) => el("th", { text: h })));
  return el("table", {}, [el("thead", {}, [head]), el("tbody", {}, rows)]);
}

function setScreen(name, nodes) {
  setText("title", T.title + " | " + T[name]);
  const screen = document.getElementById("screen");
  screen.replaceChildren(...nodes);
}

function header(p) {
  setText("clock", p.clock ? p.clock + " | " + (p.period_text || "") : "");
  setText("age", p.data_age_text || "");
  setText("alerts-used", p.alerts_text || "");
  const banner = document.getElementById("banner");
  banner.hidden = !p.banner;
  banner.textContent = p.banner || "";
  const note = document.getElementById("mode-note");
  note.hidden = !p.mode_note;
  note.textContent = p.mode_note || "";
}

function showDetail(row) {
  const box = document.getElementById("detail");
  if (!row) {
    box.hidden = true;
    return;
  }
  const lines = [
    T.detail_p15 + ": " + row.p15_text,
    T.col_rest + ": " + row.r_rest_text + " (" + T.detail_interval + " " + row.interval_text + ")",
    T.detail_usual + ": " + row.r_usual_text,
  ].concat(row.quality_reasons || []);
  box.replaceChildren(
    el("h2", { text: row.label }),
    el("ul", { class: "lines" }, lines.map((l) => el("li", { text: l }))),
    el("p", { class: "small", text: (window.HELP || {}).interval || "" })
  );
  box.hidden = false;
}

function renderLive(p) {
  header(p);
  const rows = p.rows.map((r) => {
    const tr = el("tr", {
      class: "row " + r.state + (r.alert_level ? " alert-" + r.alert_level : ""),
      tabindex: "0",
    }, [
      el("td", { text: r.label }),
      el("td", {}, [el("span", { class: "chip " + r.prematch_band, text: r.prematch_band_text })]),
      el("td", { class: "num", text: r.rr_text }),
      el("td", { class: "num", text: r.r_rest_text }),
      el("td", { text: r.reason_text }),
      el("td", {}, [
        el("span", { text: r.quality_text }),
        el("span", { class: "age", text: r.age_text ? " " + r.age_text : "" }),
      ]),
    ]);
    tr.addEventListener("click", () => {
      selected = r.player_id;
      showDetail(r);
    });
    return tr;
  });
  setScreen("screen_live", [
    table([T.col_player, T.col_prematch, T.col_now, T.col_rest, T.col_why, T.col_data], rows),
  ]);
  showDetail(p.rows.find((r) => r.player_id === selected));
  renderCards(p.alerts, true);
}

function renderCards(alerts, withButtons) {
  const cards = (alerts || []).map((a) => {
    const parts = [
      el("h2", { text: a.level_text + " | " + a.label }),
      el("p", { text: a.reason_text }),
    ];
    if (a.action_text) parts.push(el("p", { text: a.action_text }));
    if (a.numbers_text) parts.push(el("p", { text: a.numbers_text }));
    if (a.disclaimer) parts.push(el("p", { class: "small", text: a.disclaimer }));
    if (withButtons) {
      (a.buttons || []).forEach((b) => {
        const button = el("button", { type: "button", text: b.text });
        button.addEventListener("click", () => decide(a.alert_id, b.key, button));
        parts.push(button);
      });
    }
    return el("div", { class: "card " + a.level }, parts);
  });
  document.getElementById("cards").replaceChildren(...cards);
}

async function decide(alertId, key, button) {
  const body = { alert_id: alertId, actor_role: role, button: key };
  const res = await fetch("/api/decisions?" + query(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (res.ok) {
    const out = await res.json();
    button.textContent = out.saved;
    button.disabled = true;
  }
}

function renderPre(p) {
  header({});
  const rows = p.rows.map((r) => el("tr", {}, [
    el("td", { text: r.label }),
    el("td", { text: r.starting_text }),
    el("td", {}, [el("span", { class: "chip " + r.prematch_band, text: r.prematch_band_text })]),
    el("td", { class: "num", text: r.r_usual_90_text }),
  ]));
  setScreen("screen_pre_match", [
    table([T.col_player, T.col_starting, T.col_prematch, T.col_usual_90], rows),
  ]);
  showDetail(null);
  renderCards([], false);
}

function renderHalf(p) {
  header({});
  const rows = p.rows.map((r) => el("tr", {}, [
    el("td", { text: r.label }),
    el("td", {}, [el("ul", { class: "lines" }, r.load_lines.map((l) => el("li", { text: l })))]),
    el("td", { class: "num", text: r.r_45_text }),
    el("td", { class: "num", text: r.r_20_text }),
  ]));
  setScreen("screen_half_time", [
    table([T.col_player, T.col_load, T.col_second_45, T.col_second_20], rows),
  ]);
  showDetail(null);
  renderCards([], false);
}

function renderPost(p) {
  header({});
  const alertRows = p.alerts.map((a) => el("tr", {}, [
    el("td", { text: a.label }),
    el("td", { text: a.level_text }),
    el("td", { text: a.reason_text }),
    el("td", { class: "num", text: a.opened + " - " + a.cleared }),
    el("td", { text: a.decisions.join(", ") }),
  ]));
  const playerRows = p.players.map((r) => el("tr", {}, [
    el("td", { text: r.label }),
    el("td", { class: "num", text: String(r.alerts) }),
    el("td", { class: "num", text: r.max_rr_text }),
  ]));
  const nodes = [];
  if (p.none_open) nodes.push(el("p", { text: p.none_open }));
  nodes.push(table([T.col_player, T.col_level, T.col_why, T.col_time, T.col_decision], alertRows));
  nodes.push(table([T.col_player, T.col_alerts, T.col_max_rr], playerRows));
  setScreen("screen_post_match", nodes);
  showDetail(null);
  renderCards([], false);
}

function renderCoach(p) {
  header(p);
  const nodes = p.alerts.length ? [] : [el("p", { text: T.none_open })];
  setScreen("screen_live", nodes);
  showDetail(null);
  renderCards(p.alerts, false);
}

function render(p) {
  lastPayload = p;
  if (p.type === "live_table") renderLive(p);
  else if (p.type === "pre_match") renderPre(p);
  else if (p.type === "half_time") renderHalf(p);
  else if (p.type === "post_match") renderPost(p);
  else if (p.type === "coach_summary") renderCoach(p);
}

function connection(ok) {
  const c = document.getElementById("connection");
  c.textContent = ok ? T.connection_live : T.connection_lost;
  c.className = "connection" + (ok ? "" : " lost");
}

// While the WebSocket is down the page polls the REST snapshot, so the screen keeps updating
// on servers without WebSocket support; the payload carries its own data age either way.
let pollTimer = null;

async function poll() {
  try {
    const res = await fetch("/api/live?" + query());
    if (!res.ok) throw new Error(String(res.status));
    render(await res.json());
    connection(true);
  } catch (err) {
    connection(false);
  }
}

function startPolling() {
  if (pollTimer === null) pollTimer = setInterval(poll, 2000);
}

function stopPolling() {
  if (pollTimer !== null) clearInterval(pollTimer);
  pollTimer = null;
}

function connect() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(scheme + "://" + window.location.host + "/ws/live?" + query());
  ws.onopen = () => {
    retryMs = 1000;
    stopPolling();
    connection(true);
  };
  ws.onmessage = (ev) => render(JSON.parse(ev.data));
  ws.onclose = () => {
    startPolling();
    setTimeout(connect, retryMs);
    retryMs = Math.min(retryMs * 2, 30000);
  };
}

async function start() {
  const words = await (await fetch("/api/wording")).json();
  T = words.ui;
  document.title = T.title;
  window.HELP = words.help;
  await poll();
  connect();
}

start();
