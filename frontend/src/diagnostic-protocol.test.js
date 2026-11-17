import { test } from "node:test";
import assert from "node:assert/strict";
import { phaseAt } from "./diagnostic-protocol.js";
const protocol = {
  duration_seconds: 180,
  stages: [0, 30, 60, 90, 120, 150].map((start, i) => ({
    id: String(i),
    start,
    end: start + 30,
    transition_seconds: [0, 3, 3, 8, 3, 8][i],
  })),
};
test("180-second phase clock follows exact boundaries independently of AI frame rate", () => {
  assert.equal(phaseAt(protocol, 29.999).id, "0");
  assert.equal(phaseAt(protocol, 30).id, "1");
  assert.equal(phaseAt(protocol, 30).transition, true);
  assert.equal(phaseAt(protocol, 33).transition, false);
  assert.equal(phaseAt(protocol, 90).transition_remaining_seconds, 8);
  assert.equal(phaseAt(protocol, 98).transition, false);
  assert.equal(phaseAt(protocol, 179.9).id, "5");
  assert.equal(phaseAt(protocol, 180), null);
  assert.equal(phaseAt(protocol, -1), null);
});
test("selective retest stays 30 seconds and cannot repeat itself", () => {
  const targeted = {
    duration_seconds: 30,
    stages: [{ id: "targeted", start: 0, end: 30, transition_seconds: 0 }],
  };
  assert.equal(phaseAt(targeted, 0).remaining_seconds, 30);
  assert.equal(phaseAt(targeted, 30), null);
  assert.equal(phaseAt(targeted, 180), null);
  assert.equal(phaseAt(null, 0), null);
});
