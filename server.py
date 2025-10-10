# server.py - Python network implementation using network.py
from __future__ import annotations

import threading
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, Response

import initialize_network
import network as net_python
import train  # noqa: F401 - referenced by subprocess invocations
from config_store import (
    load_config,
    update_config as store_update_config,
    reset_config as store_reset_config,
)
from param_io import load_params
from training_manager import manager as training_manager

app = Flask(__name__)

# ---------- Global state ----------
PARAM_LOCK = threading.Lock()
PARAMS: Optional[Any] = None


def _timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def load_model_params() -> bool:
    """Refresh PARAMS from disk."""
    global PARAMS
    try:
        params = load_params()
    except FileNotFoundError:
        params = None
    with PARAM_LOCK:
        PARAMS = params
    return params is not None


def get_params_snapshot():
    with PARAM_LOCK:
        return PARAMS


def ensure_params_loaded() -> bool:
    if get_params_snapshot() is not None:
        return True
    return load_model_params()


# ---------- Initial load ----------
_boot_config = load_config()
net_python.refresh_config()
initialize_network.refresh_params()
if load_model_params():
    print("✓ Loaded network parameters")
else:
    print("✗ Network parameters not found (params.pkl)")


# ---------- Network prediction function ----------
def predict(image_data):
    """Use the Python network implementation"""
    params = get_params_snapshot()
    if params is None:
        raise ValueError("Network parameters not loaded")

    outputs = net_python.forward_pass(params, image_data)
    probs = net_python.softmax(outputs)
    return probs


# ---------- Helpers ----------
def _parse_network_size(value: Any) -> list[int]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
        if not parts:
            raise ValueError("network_size cannot be empty")
        try:
            return [int(p) for p in parts]
        except ValueError as err:  # noqa: F841
            raise ValueError("network_size must contain integers")
    if isinstance(value, (list, tuple)):
        try:
            return [int(p) for p in value]
        except ValueError as err:  # noqa: F841
            raise ValueError("network_size must contain integers")
    raise ValueError("network_size must be a list or comma-separated string")


def _coerce_positive_int(name: str, value: Any) -> int:
    if isinstance(value, str) and value.strip() == "":
        raise ValueError(f"{name} cannot be blank")
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as err:  # noqa: F841
        raise ValueError(f"{name} must be an integer")
    if ivalue <= 0:
        raise ValueError(f"{name} must be positive")
    return ivalue


def _coerce_positive_float(name: str, value: Any) -> float:
    if isinstance(value, str) and value.strip() == "":
        raise ValueError(f"{name} cannot be blank")
    try:
        fvalue = float(value)
    except (TypeError, ValueError) as err:  # noqa: F841
        raise ValueError(f"{name} must be a number")
    if fvalue <= 0:
        raise ValueError(f"{name} must be positive")
    return fvalue


def _parse_config_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    if "network_size" in payload:
        updates["network_size"] = _parse_network_size(payload["network_size"])
    if "learning_rate" in payload:
        updates["learning_rate"] = _coerce_positive_float("learning_rate", payload["learning_rate"])
    if "number_of_epochs" in payload:
        updates["number_of_epochs"] = _coerce_positive_int("number_of_epochs", payload["number_of_epochs"])
    if "images_to_train_on" in payload:
        updates["images_to_train_on"] = _coerce_positive_int("images_to_train_on", payload["images_to_train_on"])
    if "data_root" in payload:
        data_root = str(payload["data_root"]).strip()
        if not data_root:
            raise ValueError("data_root cannot be blank")
        updates["data_root"] = data_root
    if "param_path" in payload:
        param_path = str(payload["param_path"]).strip()
        if not param_path:
            raise ValueError("param_path cannot be blank")
        updates["param_path"] = param_path
    if "max_test_examples" in payload:
        value = payload["max_test_examples"]
        if value in (None, ""):
            updates["max_test_examples"] = None
        else:
            updates["max_test_examples"] = _coerce_positive_int("max_test_examples", value)
    return updates


# ---------- HTML Interface ----------
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>28×28 Digit Test UI</title>
<style>
  :root {
    --bg: #ffffff;
    --panel: #ffffff;
    --ink: #111111;
    --muted: #666666;
    --border: #e0e0e0;
    --shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    background: var(--bg);
    font-family: system-ui, sans-serif;
    color: var(--ink);
    display: grid;
    place-items: center;
  }
  .wrap {
    display: grid;
    grid-template-columns: auto 360px;
    gap: 24px;
    align-items: start;
    width: min(1040px, 96vw);
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: var(--shadow);
  }

  /* Left: canvas + toolbar */
  .draw-card {
    padding: 16px;
    display: grid;
    gap: 12px;
    justify-items: center;
  }
  .toolbar {
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
  }
  .btn {
    appearance: none;
    border: 1px solid var(--border);
    background: #fff;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 500;
    transition: background 0.15s ease;
  }
  .btn.primary {
    background: #111111;
    color: #ffffff;
  }
  .btn.danger {
    background: #d32f2f;
    color: #ffffff;
  }
  .btn[disabled] {
    opacity: 0.6;
    cursor: not-allowed;
  }

  #canvas {
    width: 560px;
    height: 560px;
    image-rendering: pixelated;
    border-radius: 6px;
    border: 1px solid var(--border);
    display: block;
    background: #fff;
  }

  .stack {
    display: grid;
    gap: 16px;
  }

  .prob-card {
    padding: 16px;
    display: grid;
    gap: 12px;
  }
  .prob-header {
    display: flex; align-items: baseline; justify-content: space-between;
  }
  .prob-header h2 {
    font-size: 16px; margin: 0;
  }
  .prob-list {
    display: grid; gap: 8px;
  }
  .prob-row {
    display: grid;
    grid-template-columns: 40px 1fr auto;
    gap: 8px; align-items: center;
    padding: 6px 8px;
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  .dot {
    width: 32px; height: 32px; border-radius: 50%;
    border: 1px solid var(--border);
    background: #fff;
  }
  .digit {
    color: var(--ink); font-weight: 600; font-size: 14px;
  }
  .prob {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    font-weight: 500;
  }

  .control-card {
    padding: 16px;
    display: grid;
    gap: 12px;
  }
  .control-card h2 {
    margin: 0;
    font-size: 16px;
  }
  .config-form {
    display: grid;
    gap: 10px;
  }
  .config-field {
    display: grid;
    gap: 4px;
  }
  .config-field label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
  }
  .config-field input {
    padding: 6px 8px;
    border: 1px solid var(--border);
    border-radius: 4px;
    font-size: 14px;
  }
  .actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .msg {
    font-size: 13px;
    min-height: 18px;
  }
  .msg.success { color: #218c5f; }
  .msg.error { color: #d32f2f; }
  .status-block {
    font-size: 13px;
    background: #f7f7f7;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px;
    white-space: pre-line;
  }
  .status-block strong {
    display: block;
    font-size: 12px;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
  }
  .log-block {
    max-height: 220px;
    overflow: auto;
    background: #0d1117;
    color: #c9d1d9;
    border-radius: 6px;
    border: 1px solid #1f242b;
    padding: 8px;
    font-size: 12px;
    line-height: 1.4;
    font-family: ui-monospace, SFMono-Regular, SFMono, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    white-space: pre;
  }

  @media (max-width: 800px) {
    .wrap { grid-template-columns: 1fr; gap: 20px; }
    #canvas { width: 90vw; height: 90vw; max-width: 400px; max-height: 400px; }
  }
</style>
</head>
<body>
  <div class="wrap">
    <div class="card draw-card">
      <canvas id="canvas" width="28" height="28"></canvas>
      <div class="toolbar">
        <button id="clear" class="btn">Clear</button>
        <button id="tick" class="btn">Sample once</button>
      </div>
    </div>

    <div class="stack">
      <div class="card prob-card">
        <div class="prob-header">
          <h2>Probabilities (0–9)</h2>
          <span class="hint">darker = higher</span>
        </div>
        <div id="probs" class="prob-list"></div>
      </div>

      <div class="card control-card">
        <h2>Model Controls</h2>
        <form id="config-form" class="config-form">
          <div class="config-field">
            <label for="cfg-network-size">Network size</label>
            <input id="cfg-network-size" name="network_size" placeholder="784, 16, 16, 10" />
          </div>
          <div class="config-field">
            <label for="cfg-learning-rate">Learning rate</label>
            <input id="cfg-learning-rate" name="learning_rate" type="number" step="0.0001" min="0" />
          </div>
          <div class="config-field">
            <label for="cfg-number-of-epochs">Epochs</label>
            <input id="cfg-number-of-epochs" name="number_of_epochs" type="number" min="1" />
          </div>
          <div class="config-field">
            <label for="cfg-images">Images to train on</label>
            <input id="cfg-images" name="images_to_train_on" type="number" min="1" />
          </div>
          <div class="config-field">
            <label for="cfg-data-root">Data root</label>
            <input id="cfg-data-root" name="data_root" placeholder="data" />
          </div>
          <div class="config-field">
            <label for="cfg-param-path">Parameter path</label>
            <input id="cfg-param-path" name="param_path" placeholder="params.pkl" />
          </div>
          <div class="config-field">
            <label for="cfg-max-test">Max test examples (leave blank for all)</label>
            <input id="cfg-max-test" name="max_test_examples" type="number" min="1" />
          </div>
          <div class="actions">
            <button id="btn-config-save" type="submit" class="btn primary">Save config</button>
            <button id="btn-config-reset" type="button" class="btn">Reset defaults</button>
          </div>
        </form>
        <div class="actions">
          <button id="btn-reinit" type="button" class="btn">Reinitialize weights</button>
          <button id="btn-train" type="button" class="btn">Train model</button>
          <button id="btn-stop" type="button" class="btn danger">Stop training</button>
        </div>
        <div id="config-msg" class="msg"></div>
        <div id="train-status" class="status-block">Status: (loading…)</div>
        <div>
          <strong>Logs</strong>
          <div id="train-logs" class="log-block">(no logs yet)</div>
        </div>
      </div>
    </div>
  </div>

<script>
(function() {
  // ----- Drawing & inference -----
  const W = 28, H = 28;
  const pixels = new Uint8ClampedArray(W * H);
  let isDown = false;
  let lastX = -1, lastY = -1;
  let inFlight = false;

  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const clearBtn = document.getElementById('clear');
  const tickBtn = document.getElementById('tick');
  const probsEl = document.getElementById('probs');

  const rows = [];
  for (let d = 0; d < 10; d++) {
    const row = document.createElement('div');
    row.className = 'prob-row';
    const dot = document.createElement('div');
    dot.className = 'dot';
    const digit = document.createElement('div');
    digit.className = 'digit';
    digit.textContent = String(d);
    const prob = document.createElement('div');
    prob.className = 'prob';
    prob.textContent = '0.0%';
    row.appendChild(dot); row.appendChild(digit); row.appendChild(prob);
    probsEl.appendChild(row);
    rows.push({dot, digit, prob});
  }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  function posToCell(evt) {
    const rect = canvas.getBoundingClientRect();
    const x = ((evt.clientX - rect.left) / rect.width) * W;
    const y = ((evt.clientY - rect.top)  / rect.height) * H;
    return { x, y, i: clamp(Math.floor(x), 0, W-1), j: clamp(Math.floor(y), 0, H-1) };
  }

  function drawDot(i, j, radius, value, erase) {
    const r2 = radius * radius;
    const cx = i, cy = j;
    const i0 = clamp(Math.floor(cx - radius), 0, W-1);
    const i1 = clamp(Math.ceil(cx + radius), 0, W-1);
    const j0 = clamp(Math.floor(cy - radius), 0, H-1);
    const j1 = clamp(Math.ceil(cy + radius), 0, H-1);
    for (let y = j0; y <= j1; y++) {
      for (let x = i0; x <= i1; x++) {
        const dx = x - cx, dy = y - cy;
        if (dx*dx + dy*dy <= r2) {
          const idx = y * W + x;
          if (erase) {
            pixels[idx] = Math.max(0, pixels[idx] - value);
          } else {
            pixels[idx] = Math.min(255, pixels[idx] + value);
          }
        }
      }
    }
  }

  function drawStroke(x0, y0, x1, y1) {
    const radius = 1.2;
    const value  = 128;
    const erase = false;
    const steps = Math.max(Math.abs(x1-x0), Math.abs(y1-y0)) * 1.5 + 1;
    for (let s = 0; s <= steps; s++) {
      const t = s / steps;
      const xi = x0 + (x1 - x0) * t;
      const yi = y0 + (y1 - y0) * t;
      drawDot(xi, yi, radius, value, erase);
    }
  }

  function render() {
    const img = ctx.getImageData(0, 0, W, H);
    const data = img.data;
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const v = pixels[j*W + i];
        const shade = 255 - v;
        const k = (j*W + i) * 4;
        data[k] = shade;
        data[k + 1] = shade;
        data[k + 2] = shade;
        data[k + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    requestAnimationFrame(render);
  }

  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    isDown = true;
    const {x, y} = posToCell(e);
    lastX = x; lastY = y;
    drawStroke(x, y, x, y);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!isDown) return;
    const {x, y} = posToCell(e);
    drawStroke(lastX, lastY, x, y);
    lastX = x; lastY = y;
  });
  window.addEventListener('pointerup', () => { isDown = false; });

  clearBtn.addEventListener('click', () => pixels.fill(0));
  tickBtn.addEventListener('click', () => tick());

  async function tick() {
    if (inFlight) return;
    inFlight = true;
    const arr = new Float32Array(W * H);
    for (let i = 0; i < pixels.length; i++) {
      arr[i] = pixels[i] / 255.0;
    }
    try {
      const res = await fetch('/api/forward', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ pixels: Array.from(arr) }),
        cache: 'no-store',
      });
      if (res.ok) {
        const data = await res.json();
        updateProbs(data.probs);
      }
    } catch (_) {
      // ignore transient errors
    } finally {
      inFlight = false;
    }
  }

  setInterval(tick, 200);

  function updateProbs(p) {
    if (!Array.isArray(p) || p.length !== 10) return;
    for (let d = 0; d < 10; d++) {
      const prob = Math.max(0, Math.min(1, Number(p[d]) || 0));
      const shade = Math.round(255 - prob * 255);
      rows[d].dot.style.backgroundColor = `rgb(${shade}, ${shade}, ${shade})`;
      rows[d].prob.textContent = (prob * 100).toFixed(1) + '%';
    }
  }

  render();

  // ----- Config & training controls -----
  const form = document.getElementById('config-form');
  const msgEl = document.getElementById('config-msg');
  const resetBtn = document.getElementById('btn-config-reset');
  const reinitBtn = document.getElementById('btn-reinit');
  const trainBtn = document.getElementById('btn-train');
  const stopBtn = document.getElementById('btn-stop');
  const statusEl = document.getElementById('train-status');
  const logEl = document.getElementById('train-logs');
  const fieldNetworkSize = document.getElementById('cfg-network-size');
  const fieldLearningRate = document.getElementById('cfg-learning-rate');
  const fieldEpochs = document.getElementById('cfg-number-of-epochs');
  const fieldImages = document.getElementById('cfg-images');
  const fieldDataRoot = document.getElementById('cfg-data-root');
  const fieldParamPath = document.getElementById('cfg-param-path');
  const fieldMaxTest = document.getElementById('cfg-max-test');

  let currentStatus = null;
  let logSeq = 0;

  function setMessage(text, type) {
    msgEl.textContent = text || '';
    msgEl.className = 'msg' + (type ? ' ' + type : '');
  }

  function formatNetworkSize(sizes) {
    if (!Array.isArray(sizes)) return '';
    return sizes.join(', ');
  }

  function fillForm(cfg) {
    if (!cfg) return;
    fieldNetworkSize.value = formatNetworkSize(cfg.network_size);
    fieldLearningRate.value = cfg.learning_rate;
    fieldEpochs.value = cfg.number_of_epochs;
    fieldImages.value = cfg.images_to_train_on;
    fieldDataRoot.value = cfg.data_root;
    fieldParamPath.value = cfg.param_path;
    fieldMaxTest.value = cfg.max_test_examples == null ? '' : cfg.max_test_examples;
  }

  function parseNetworkSizeInput(value) {
    return value
      .split(/[\s,]+/)
      .map((part) => part.trim())
      .filter(Boolean)
      .map((part) => {
        const n = Number(part);
        if (!Number.isInteger(n) || n <= 0) {
          throw new Error('Network size entries must be positive integers');
        }
        return n;
      });
  }

  function readForm() {
    return {
      network_size: parseNetworkSizeInput(fieldNetworkSize.value || ''),
      learning_rate: Number(fieldLearningRate.value),
      number_of_epochs: Number(fieldEpochs.value),
      images_to_train_on: Number(fieldImages.value),
      data_root: fieldDataRoot.value,
      param_path: fieldParamPath.value,
      max_test_examples: fieldMaxTest.value === '' ? null : Number(fieldMaxTest.value),
    };
  }

  function disableForm(disabled) {
    Array.from(form.elements).forEach((field) => { field.disabled = disabled; });
    resetBtn.disabled = disabled;
  }

  function updateButtons() {
    const running = currentStatus && currentStatus.state === 'running';
    trainBtn.disabled = running;
    reinitBtn.disabled = running;
    stopBtn.disabled = !running;
    disableForm(running);
  }

  async function loadConfig() {
    try {
      const res = await fetch('/api/config');
      if (!res.ok) throw new Error('Failed to load config');
      const data = await res.json();
      fillForm(data);
    } catch (err) {
      console.error(err);
      setMessage('Could not load configuration', 'error');
    }
  }

  async function saveConfig(event) {
    event.preventDefault();
    try {
      const payload = readForm();
      disableForm(true);
      const res = await fetch('/api/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || 'Failed to save configuration');
      }
      const cfg = await res.json();
      fillForm(cfg);
      setMessage('Configuration saved', 'success');
    } catch (err) {
      console.error(err);
      setMessage(err.message || 'Failed to save configuration', 'error');
    } finally {
      disableForm(false);
      updateButtons();
    }
  }

  async function resetConfig() {
    try {
      disableForm(true);
      const res = await fetch('/api/config/reset', { method: 'POST' });
      if (!res.ok) throw new Error('Failed to reset configuration');
      const cfg = await res.json();
      fillForm(cfg);
      setMessage('Configuration reset to defaults', 'success');
    } catch (err) {
      console.error(err);
      setMessage(err.message || 'Failed to reset configuration', 'error');
    } finally {
      disableForm(false);
      updateButtons();
    }
  }

  async function reinitializeWeights() {
    try {
      setMessage('Reinitializing weights…', '');
      reinitBtn.disabled = true;
      trainBtn.disabled = true;
      const res = await fetch('/api/reinit', { method: 'POST' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || 'Failed to reinitialize weights');
      }
      await res.json();
      setMessage('Weights reinitialized and saved', 'success');
      await refreshStatus();
    } catch (err) {
      console.error(err);
      setMessage(err.message || 'Failed to reinitialize weights', 'error');
    } finally {
      reinitBtn.disabled = false;
      updateButtons();
    }
  }

  function appendLogs(entries) {
    if (!entries || entries.length === 0) return;
    const atBottom = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 10;
    if (logEl.textContent === '(no logs yet)') {
      logEl.textContent = '';
    }
    entries.forEach((entry) => {
      logEl.textContent += `[${entry.timestamp}] ${entry.line}\n`;
      logSeq = Math.max(logSeq, entry.seq + 1);
    });
    if (atBottom) {
      logEl.scrollTop = logEl.scrollHeight;
    }
  }

  async function fetchLogs() {
    try {
      const res = await fetch(`/api/train/logs?since=${logSeq}`);
      if (!res.ok) return;
      const data = await res.json();
      appendLogs(data.logs || []);
      if (typeof data.next_seq === 'number') {
        logSeq = data.next_seq;
      }
    } catch (err) {
      console.error(err);
    }
  }

  function updateStatusView(status) {
    currentStatus = status;
    const lines = [
      `Status: ${status.state}`,
      status.message ? `Message: ${status.message}` : null,
      status.started_at ? `Started: ${status.started_at}` : null,
      status.finished_at ? `Finished: ${status.finished_at}` : null,
      status.error ? `\n${status.error}` : null,
    ].filter(Boolean);
    statusEl.textContent = lines.join('\n');
    updateButtons();
  }

  async function refreshStatus() {
    try {
      const res = await fetch('/api/train/status');
      if (!res.ok) throw new Error('Failed to fetch status');
      const status = await res.json();
      updateStatusView(status);
    } catch (err) {
      console.error(err);
    }
  }

  async function startTraining() {
    try {
      setMessage('Starting training…', '');
      trainBtn.disabled = true;
      reinitBtn.disabled = true;
      const res = await fetch('/api/train', { method: 'POST' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || 'Failed to start training');
      }
      const status = await res.json();
      logEl.textContent = '(no logs yet)';
      logSeq = 0;
      updateStatusView(status);
      setMessage('Training started', 'success');
    } catch (err) {
      console.error(err);
      setMessage(err.message || 'Failed to start training', 'error');
    }
  }

  async function stopTraining() {
    try {
      stopBtn.disabled = true;
      const res = await fetch('/api/train/stop', { method: 'POST' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || 'Failed to stop training');
      }
      const status = await res.json();
      updateStatusView(status);
      setMessage('Stop signal sent', 'success');
    } catch (err) {
      console.error(err);
      setMessage(err.message || 'Failed to stop training', 'error');
    }
  }

  form.addEventListener('submit', saveConfig);
  resetBtn.addEventListener('click', resetConfig);
  reinitBtn.addEventListener('click', reinitializeWeights);
  trainBtn.addEventListener('click', startTraining);
  stopBtn.addEventListener('click', stopTraining);

  loadConfig();
  refreshStatus();
  setInterval(refreshStatus, 2000);
  setInterval(fetchLogs, 1000);
})();
</script>
</body>
</html>
"""


# ---------- Routes ----------
@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.get("/api/config")
def api_config_get():
    config = load_config()
    return jsonify(config)


@app.post("/api/config")
def api_config_update():
    if training_manager.is_running():
        return jsonify({"error": "Cannot update configuration while training is running"}), 409
    try:
        payload = request.get_json(force=True, silent=False) or {}
        if not isinstance(payload, dict):
            raise ValueError("Payload must be an object")
        updates = _parse_config_payload(payload)
        if not updates:
            return jsonify(load_config())
        config = store_update_config(updates)
        net_python.refresh_config()
        initialize_network.refresh_params()
        return jsonify(config)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/config/reset")
def api_config_reset():
    if training_manager.is_running():
        return jsonify({"error": "Cannot reset configuration while training is running"}), 409
    config = store_reset_config()
    net_python.refresh_config()
    initialize_network.refresh_params()
    return jsonify(config)


@app.post("/api/reinit")
def api_reinit():
    if training_manager.is_running():
        return jsonify({"error": "Cannot reinitialize while training is running"}), 409
    try:
        net_python.refresh_config()
        initialize_network.refresh_params()
        layers = initialize_network.save_new_params()
        load_model_params()
        return jsonify({"status": "ok", "layers": len(layers)})
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.post("/api/train")
def api_train():
    if training_manager.is_running():
        return jsonify({"error": "Training already in progress"}), 409
    try:
        config = load_config()
        status = training_manager.start_training(config, on_complete=load_model_params)
        return jsonify(status)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.post("/api/train/stop")
def api_train_stop():
    try:
        status = training_manager.stop_training()
        return jsonify(status)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 409


@app.get("/api/train/status")
def api_train_status():
    return jsonify(training_manager.status())


@app.get("/api/train/logs")
def api_train_logs():
    try:
        since_raw = request.args.get("since", "0")
        try:
            since = int(since_raw)
        except ValueError:
            since = 0
        logs, next_seq = training_manager.get_logs(since)
        return jsonify({"logs": logs, "next_seq": next_seq})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/forward")
def api_forward():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        raw = payload.get("pixels", [])

        if not isinstance(raw, list) or len(raw) != 28 * 28:
            return jsonify({"error": "Expected 'pixels' as list[784]"}), 400

        arr = []
        for v in raw:
            try:
                f = float(v)
            except Exception:  # noqa: BLE001
                f = 0.0
            arr.append(f / 255.0 if f > 1.0 else f)

        if not ensure_params_loaded():
            return jsonify({"error": "Network parameters not loaded"}), 500

        probs = predict(arr)

        if len(probs) != 10:
            return jsonify({"error": "Network must return 10 probabilities"}), 500

        probs_list = [max(0.0, min(1.0, p)) for p in probs]

        return jsonify({"probs": probs_list})

    except Exception as exc:  # noqa: BLE001
        print(f"ERROR /api/forward: {exc}")
        traceback.print_exc()
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


if __name__ == "__main__":
    print("\n=== Network Status ===")
    print(f"Network: {'✓ Available' if get_params_snapshot() else '✗ Not available'}")

    import socket

    port = 6969
    for p in [6969, 6970, 6971]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", p))
            port = p
            break
        except OSError:
            continue

    print(f"\nStarting server on port {port}")
    print(f"Available at: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
