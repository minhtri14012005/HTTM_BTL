import { test } from "node:test";
import assert from "node:assert/strict";
import { frameAt, visibleEvents, mergeEvents, cameraFor, acquireCamera } from "./sync.js";
test("timestamp lookup and seek never use a future/stale overlay", () => {
  const frames = [
    { frame_id: 0, timestamp: 0 },
    { frame_id: 1, timestamp: 0.04 },
    { frame_id: 2, timestamp: 0.08 },
  ];
  assert.equal(frameAt(frames, 0.06, 25).frame_id, 1);
  assert.equal(frameAt(frames, 2, 25), null);
  assert.equal(frameAt([], 0, 25), null);
  assert.equal(frameAt([{ frame_id: 0, timestamp: 0 }], 0.05, 30), null);
  assert.equal(frameAt([{ frame_id: 0, timestamp: 0 }], 0.02, 30).frame_id, 0);

  assert.equal(frameAt(frames, 0.01, 25).frame_id, 0);
  assert.equal(
    frameAt([{ timestamp: 0 }, { timestamp: 1 }], 0.8, 30).timestamp,
    0,
  );
});
test("notification context follows playback, seek replay does not duplicate", () => {
  const events = [
    { id: "a", timestamp: 2 },
    { id: "b", timestamp: 10 },
  ];
  assert.equal(visibleEvents(events, 1).length, 0);
  assert.deepEqual(visibleEvents(events, 4), [events[0]]);
  assert.equal(mergeEvents(events, events).length, 2);
});
test("camera mapping uses actual labels, never fixed indexes", () => {
  const dev = (deviceId, label) => ({ kind: "videoinput", deviceId, label });
  const devices = [
    dev("virtual", "OBS Virtual Camera"),
    dev("mobile", "Camo Camera"),
    dev("integrated", "Integrated Webcam"),
  ];
  assert.equal(cameraFor(devices, "laptop").deviceId, "integrated");
  assert.equal(cameraFor(devices.reverse(), "camo").deviceId, "mobile");
  assert.equal(cameraFor([dev("x", "OBS Virtual Camera")], "laptop"), null);
  assert.equal(cameraFor([dev("x", "")], "camo"), undefined);
});

