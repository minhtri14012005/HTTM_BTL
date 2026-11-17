export function frameAt(frames, time, fps) {
  let lo = 0,
    hi = frames.length - 1,
    found = null;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].timestamp <= time + 0.001) {
      found = frames[mid];
      lo = mid + 1;
    } else hi = mid - 1;
  }
  const next = frames[hi + 1];
  return found &&
    ((next && time < next.timestamp) ||
      time - found.timestamp < 1 / fps + 0.001)
    ? found
    : null;
}
export function visibleEvents(events, time) {
  return events.filter((e) => e.timestamp <= time + 0.001);
}
export function mergeEvents(previous, incoming) {
  const all = new Map(previous.map((e) => [e.id, e]));
  incoming.forEach((e) => all.set(e.id, e));
  return [...all.values()].sort((a, b) => a.timestamp - b.timestamp);
}
export function cameraFor(devices, kind) {
  const cameras = devices.filter((d) => d.kind === "videoinput");
  if (kind === "camo") return cameras.find((d) => /camo/i.test(d.label));
  const eligible = cameras.filter(
    (d) =>
      d.label &&
      !/iriun|camo|obs|virtual|screen|droidcam|iphone|continuity/i.test(d.label),
  );
  return (
    eligible.find((d) =>
      /integrated|built.?in|laptop|webcam|hd camera|fhd camera|easycamera/i.test(
        d.label,
      ),
    ) || (eligible.length === 1 ? eligible[0] : null)
  );
}

export async function acquireCamera(mediaDevices, selected) {
  return mediaDevices.getUserMedia({
    video: {deviceId: {exact: selected.deviceId}, width: {ideal:1920}, height: {ideal:1080}},
    audio: false,
  });
}
