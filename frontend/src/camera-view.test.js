import { test } from "node:test";
import assert from "node:assert/strict";
import {
  CameraFrameBuffer,
  CameraView,
  drawFaceOverlays,
} from "./camera-view.js";

const frame = (id, capturedAt, width = 1920, height = 1080) => ({
  id,
  capturedAt,
  timestamp: capturedAt / 1000,
  width,
  height,
  bitmap: {
    closed: false,
    close() {
      this.closed = true;
    },
  },
  result: null,
});
const result = (f) => ({
  frame_id: f.id,
  captured_at: f.capturedAt,
  media_timestamp: f.timestamp,
  width: f.width,
  height: f.height,
  overlays: [{ face_bbox: [10, 10, 50, 60], status: "candidate" }],
});

test("playout advances independently of inference; an overlay belongs to exactly one frame", () => {
  const q = new CameraFrameBuffer();
  const a = frame(1, 0),
    b = frame(2, 33),
    c = frame(3, 66);
  q.push(a);
  q.push(b);
  q.push(c);
  assert.equal(q.attach(result(b)), true);
  assert.equal(q.take(300), a);
  assert.equal(a.result, null);
  assert.equal(q.take(333), b);
  assert.equal(b.result.frame_id, 2);
  assert.equal(q.take(366), c);
  assert.equal(c.result, null);
  assert.equal(q.attach(result(a)), false);
  assert.equal(q.displayed, c);
  assert.equal(q.lateResults, 1);
  q.clear();
  assert.ok(a.bitmap.closed && b.bitmap.closed && c.bitmap.closed);
});

test("reject wrong dimensions, PTS, capture time, and old frames", () => {
  const q = new CameraFrameBuffer(),
    a = frame(5, 100);
  q.push(a);
  for (const invalid of [
    { width: 800 },
    { media_timestamp: 0.2 },
    { captured_at: 101 },
    { frame_id: 4 },
  ])
    assert.equal(q.attach({ ...result(a), ...invalid }), false);
  const old = frame(4, 50);
  assert.equal(q.push(old), false);
  assert.ok(old.bitmap.closed);
  assert.equal(q.attach(result(a)), true);
  q.clear();
});

test("memory pressure shortens delay while preserving original resolution", () => {
  const q = new CameraFrameBuffer({ maxBytes: 1920 * 1080 * 4 * 4 });
  for (let i = 0; i < 20; i++) {
    q.push(frame(i + 1, i * 33));
    const displayed = q.take(i * 33);
    if (displayed) {
      assert.equal(displayed.width, 1920);
      assert.equal(displayed.height, 1080);
    }
    assert.ok(q.bytes <= q.maxBytes);
  }
  assert.ok(q.displayed.id >= 17);
  q.clear();
  assert.equal(q.bytes, 0);
});

test("only actual face boxes are drawn; candidate names stay hidden", () => {
  const calls = [],
    context = {
      save() {},
      restore() {},
      strokeRect(...v) {
        calls.push(["box", this.strokeStyle, ...v]);
      },
      fillRect() {},
      fillText(text) {
        calls.push(["text", text]);
      },
      measureText() {
        return { width: 80 };
      },
    };
  const f = frame(1, 0);
  f.result = {
    ...result(f),
    overlays: [
      {
        bbox: [0, 0, 100, 300],
        status: "confirmed",
        person_id: "body",
        name: "Body",
      },
      { face_bbox: [10, 10, 50, 60], status: "candidate", name: "Hidden" },
      {
        face_bbox: [100, 10, 150, 60],
        status: "confirmed",
        person_id: "a",
        name: "Confirmed",
      },
      { face_bbox: [-10, 0, 20, 20], status: "unknown" },
    ],
  };
  assert.equal(drawFaceOverlays(context, f), 2);
  assert.deepEqual(
    calls.filter((c) => c[0] === "box").map((c) => c[1]),
    ["#f05b66", "#0bbf88"],
  );
  assert.ok(!calls.some((c) => c[1] === "Hidden" || c[1] === "Body"));
  f.result.frame_id = 2;
  assert.equal(drawFaceOverlays(context, f), 0);
});

test("playout adapts to 360ms inference while frames keep advancing", () => {
  const q = new CameraFrameBuffer();
  let pending = null,
    attached = 0,
    lastPaint = null,
    lastId = null,
    maxPaintGap = 0;
  for (let now = 0; now < 4000; now += 33) {
    const current = frame(Math.floor(now / 33) + 1, now);
    q.push(current);
    if (pending && now >= pending.due) {
      q.observeLatency(now - pending.frame.capturedAt);
      attached += q.attach(result(pending.frame));
      pending = null;
    }
    if (!pending) pending = { frame: current, due: now + 360 };
    const shown = q.take(now);
    if (shown && shown.id !== lastId) {
      if (lastPaint !== null)
        maxPaintGap = Math.max(maxPaintGap, now - lastPaint);
      lastPaint = now;
      lastId = shown.id;
      if (shown.result) assert.equal(shown.result.frame_id, shown.id);
    }
    assert.ok(q.bytes <= q.maxBytes);
  }
  assert.ok(attached >= 6);
  assert.ok(q.delayMs >= 400 && q.delayMs <= 600);
  assert.ok(maxPaintGap <= 99);
  q.clear();
});

test("only a matching in-flight response can adjust playout delay", () => {
  const view = Object.create(CameraView.prototype);
  view.running = true;
  view.buffer = new CameraFrameBuffer();
  const f = frame(2, performance.now() - 400);
  view.buffer.push(f);
  view.lastRequest = {
    frame_id: f.id,
    captured_at: f.capturedAt,
    timestamp: f.timestamp,
    width: f.width,
    height: f.height,
  };
  assert.equal(view.accept({ ...result(f), width: 800 }), false);
  assert.equal(view.buffer.delayMs, 300);
  assert.equal(view.accept(result(f)), true);
  assert.ok(view.buffer.delayMs >= 440);
  view.buffer.clear();
});

test("CPU visual overlays attach only to their exact captured frames", () => {
  const view = Object.create(CameraView.prototype);
  view.running = true;
  view.buffer = new CameraFrameBuffer();
  const first = frame(7, performance.now() - 250);
  const second = frame(8, performance.now() - 220);
  view.buffer.push(first);
  view.buffer.push(second);
  view.acceptVisual({
    visual_tracking_ms: 4,
    frames: [
      { ...result(first), overlay_origin: "optical_flow" },
      {
        ...result(second),
        captured_at: second.capturedAt - 1,
        overlay_origin: "optical_flow",
      },
    ],
  });
  assert.equal(first.result.overlay_origin, "optical_flow");
  assert.equal(second.result, null);
  assert.equal(view.visualTrackingMs, 4);
  view.buffer.clear();
});

test("tracking renders the existing name only with an explicit continuity source", () => {
  const labels = [],
    colors = [];
  const context = {
    save() {},
    restore() {},
    fillRect() {},
    strokeRect() {
      colors.push(this.strokeStyle);
    },
    measureText() {
      return { width: 100 };
    },
    fillText(value) {
      labels.push(value);
    },
  };
  const f = frame(1, 0);
  const face = {
    face_bbox: [10, 10, 80, 100],
    status: "tracked",
    person_id: "a",
    name: "A",
  };
  f.result = { ...result(f), overlays: [face] };
  assert.equal(drawFaceOverlays(context, f), 1);
  assert.equal(colors.at(-1), "#f05b66");
  assert.ok(!labels.at(-1).includes("A"));
  face.identity_source = "track_continuity";
  drawFaceOverlays(context, f);
  assert.equal(colors.at(-1), "#0bbf88");
  assert.equal(labels.at(-1), "A · theo dõi");
});

