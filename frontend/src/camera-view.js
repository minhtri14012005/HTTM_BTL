// One playout surface. Every overlay belongs to its own frame and visual evidence.
export class CameraFrameBuffer {
  constructor({
    delayMs = 300,
    maxBytes = 160 * 1024 * 1024,
    maxFrames = 32,
    maxDelayMs = 600,
  } = {}) {
    this.delayMs = delayMs;
    this.baseDelayMs = delayMs;
    this.playoutDelayMs = delayMs;
    this.latencies = [];
    this.lastTakeAt = null;
    this.maxBytes = maxBytes;
    this.maxFrames = maxFrames;
    this.maxDelayMs = maxDelayMs;
    this.frames = [];
    this.displayed = null;
    this.bytes = 0;
    this.interval = 1000 / 30;
    this.lateResults = 0;
  }
  push(frame) {
    const previous = this.frames.at(-1) || this.displayed;
    if (
      previous &&
      (frame.id <= previous.id || frame.timestamp < previous.timestamp)
    ) {
      frame.bitmap.close();
      return false;
    }
    if (previous)
      this.interval =
        0.9 * this.interval +
        0.1 *
          Math.max(8, Math.min(100, frame.capturedAt - previous.capturedAt));
    frame.bytes = frame.width * frame.height * 4;
    this.frames.push(frame);
    this.bytes += frame.bytes;
    while (
      this.frames.length > 1 &&
      (this.bytes > this.maxBytes || this.frames.length > this.maxFrames)
    ) {
      this.discard(this.frames.shift());
    }
    return true;
  }
  discard(frame) {
    this.bytes -= frame.bytes;
    frame.bitmap.close();
  }
  attach(result) {
    const frame =
      this.frames.find((f) => f.id === result.frame_id) ||
      (this.displayed?.id === result.frame_id ? this.displayed : null);
    if (
      !frame ||
      result.width !== frame.width ||
      result.height !== frame.height ||
      !Number.isFinite(result.media_timestamp) ||
      Math.abs(result.media_timestamp - frame.timestamp) > 0.0001 ||
      result.captured_at !== frame.capturedAt
    ) {
      this.lateResults++;
      return false;
    }
    frame.result = result;
    return true;
  }
  observeLatency(elapsedMs) {
    if (!Number.isFinite(elapsedMs) || elapsedMs < 0 || elapsedMs > 30000)
      return;
    this.latencies.push(elapsedMs);
    if (this.latencies.length > 10) this.latencies.shift();
    const sorted = [...this.latencies].sort((a, b) => a - b);
    const p90 = sorted[Math.floor((sorted.length - 1) * 0.9)];
    this.delayMs = Math.max(
      this.baseDelayMs,
      Math.min(this.maxDelayMs, p90 + Math.max(40, 2 * this.interval)),
    );
  }
  take(now) {
    const latest = this.frames.at(-1);
    if (!latest) return this.displayed;
    const capacity = Math.min(
      this.maxFrames,
      Math.max(1, Math.floor(this.maxBytes / latest.bytes)),
    );
    // Reduce buffering time, never image dimensions, when camera resolution uses more memory.
    // Reserve display, current capture and capture-clock jitter slots.
    const memoryDelay = Math.max(0, capacity - 3) * this.interval;
    const target = Math.min(this.delayMs, memoryDelay);
    if (this.lastTakeAt === null) {
      this.playoutDelayMs = target;
    } else {
      // Adjust playback gradually so adapting to inference latency cannot freeze video.
      const adjustment =
        Math.min(100, Math.max(0, now - this.lastTakeAt)) * 0.12;
      this.playoutDelayMs += Math.max(
        -adjustment,
        Math.min(adjustment, target - this.playoutDelayMs),
      );
      this.playoutDelayMs = Math.min(this.playoutDelayMs, memoryDelay);
    }
    this.lastTakeAt = now;
    const delay = this.playoutDelayMs;
    let chosen = -1;
    for (
      let i = 0;
      i < this.frames.length && this.frames[i].capturedAt <= now - delay;
      i++
    )
      chosen = i;
    if (chosen < 0) return this.displayed;
    const frame = this.frames[chosen];
    for (let i = 0; i < chosen; i++) this.discard(this.frames[i]);
    this.frames.splice(0, chosen + 1);
    if (this.displayed) this.discard(this.displayed);
    this.displayed = frame;
    return frame;
  }
  clear() {
    for (const frame of this.frames) this.discard(frame);
    if (this.displayed) this.discard(this.displayed);
    this.frames = [];
    this.displayed = null;
    this.lastTakeAt = null;
    this.latencies = [];
    this.delayMs = this.playoutDelayMs = this.baseDelayMs;
  }
}

export function drawFaceOverlays(context, frame) {
  const result = frame.result;
  if (!result || result.frame_id !== frame.id) return 0;
  let count = 0;
  for (const face of result.overlays || []) {
    // Do not accept a generic/person bbox as face evidence.
    const box = face.face_bbox;
    if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite))
      continue;
    const [x1, y1, x2, y2] = box;
    if (
      x1 < 0 ||
      y1 < 0 ||
      x2 > frame.width ||
      y2 > frame.height ||
      x2 <= x1 ||
      y2 <= y1
    )
      continue;
    const confirmed =
      face.status === "confirmed" &&
      face.person_id &&
      typeof face.name === "string" &&
      face.name.trim();
    const tracked =
      face.status === "tracked" &&
      face.identity_source === "track_continuity" &&
      face.person_id &&
      typeof face.name === "string" &&
      face.name.trim();
    const label = tracked
      ? `${face.name} · theo dõi`
      : confirmed
        ? `${face.name} · xác nhận`
        : face.status === "candidate"
          ? "Đang xác minh"
          : "Chưa xác định";
    const color = confirmed || tracked ? "#0bbf88" : "#f05b66";
    const font = Math.max(12, frame.width / 85);
    context.save();
    context.strokeStyle = color;
    context.lineWidth = Math.max(2, frame.width / 600);
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    context.font = `600 ${font}px system-ui, sans-serif`;
    const width = Math.min(frame.width, context.measureText(label).width + 12);
    const left = Math.min(x1, frame.width - width),
      top = Math.max(0, y1 - font * 1.7);
    context.fillStyle = color;
    context.fillRect(left, top, width, font * 1.7);
    context.fillStyle = "#fff";
    context.fillText(label, left + 5, top + font * 1.2, width - 10);
    context.restore();
    count++;
  }
  return count;
}

export class CameraView {
  constructor(video, canvas, onStats = () => {}, onError = () => {}) {
    this.video = video;
    this.canvas = canvas;
    this.sourceCamera = null;
    try {
      const track = video.srcObject?.getVideoTracks?.()[0];
      const settings = track?.getSettings?.() || {};
      const capabilities = track?.getCapabilities?.() || {};
      this.sourceCamera = {
        width: settings.width ?? null,
        height: settings.height ?? null,
        frame_rate: settings.frameRate ?? null,
        advertised_max_width: capabilities.width?.max ?? null,
        advertised_max_height: capabilities.height?.max ?? null,
        advertised_max_frame_rate: capabilities.frameRate?.max ?? null,
      };
    } catch {
      this.sourceCamera = null;
    }

    this.onStats = onStats;
    this.onError = onError;
    this.buffer = new CameraFrameBuffer();
    this.encoder = document.createElement("canvas");
    this.waiters = [];
    this.nextId = 0;
    this.lastSent = 0;
    this.lastCaptureAt = 0;
    this.running = false;
    this.trackingEnabled = false;
  }
  start() {
    if (!this.video.requestVideoFrameCallback)
      throw new Error("Cần Chrome/Edge hỗ trợ đồng bộ frame camera.");
    this.running = true;
    this.since = performance.now();
    this.captured =
      this.rendered =
      this.skipped =
      this.overlayFrames =
      this.flowFrames =
        0;
    this.lastTimestamp = -1;
    this.sourcePresented = 0;
    this.previousPresented = null;
    this.captureWorkMs = 0;
    this.drawWorkMs = 0;
    const capture = async (_, metadata) => {
      if (!this.running) return;
      try {
        const id = ++this.nextId,
          capturedAt = performance.now();
        const timestamp = metadata.mediaTime;
        if (Number.isFinite(metadata.presentedFrames)) {
          if (this.previousPresented !== null) this.sourcePresented += Math.max(0, metadata.presentedFrames-this.previousPresented);
          this.previousPresented = metadata.presentedFrames;
        }
        const bitmap =
          typeof VideoFrame === "function"
            ? new VideoFrame(this.video, {
                timestamp: Math.round(timestamp * 1e6),
              })
            : await createImageBitmap(this.video);
        if (!this.running) {
          bitmap.close();
          return;
        }
        const width = this.video.videoWidth,
          height = this.video.videoHeight;
        // A source rotation invalidates old geometry, not the camera session.
        const previous = this.buffer.frames.at(-1) || this.buffer.displayed;
        if (
          previous &&
          (previous.width !== width || previous.height !== height)
        )
          this.buffer.clear();
        const frame = {
          id,
          bitmap,
          width,
          height,
          timestamp,
          capturedAt,
          result: null,
        };
        this.captureWorkMs += performance.now()-capturedAt;
        if (this.buffer.push(frame)) {
          this.captured++;
          this.lastCaptureAt = capturedAt;
          for (const resolve of this.waiters.splice(0)) resolve(frame);
        }
        this.captureHandle = this.video.requestVideoFrameCallback(capture);
      } catch (error) {
        if (this.running) this.onError(error.message);
      }
    };
    const paint = () => {
      if (!this.running) return;
      const now = performance.now(),
        frame = this.buffer.take(now);
      if (frame) {
        const changed =
          frame.id !== this.canvasFrameId || frame.result !== this.canvasResult;
        if (changed) {
          if (
            this.canvas.width !== frame.width ||
            this.canvas.height !== frame.height
          ) {
            this.canvas.width = frame.width;
            this.canvas.height = frame.height;
          }
          const drawStarted = performance.now();
          const context = this.canvas.getContext("2d", { alpha: false });
          context.drawImage(frame.bitmap, 0, 0, frame.width, frame.height);
          const boxes = drawFaceOverlays(context, frame);
          this.drawWorkMs += performance.now()-drawStarted;
          if (frame.id !== this.canvasFrameId) {
            this.rendered++;
            if (boxes) this.overlayFrames++;
            if (boxes && frame.result?.overlay_origin === "optical_flow")
              this.flowFrames++;
          }
          this.canvasFrameId = frame.id;
          this.canvasResult = frame.result;
          this.canvas.dataset.frameId = String(frame.id);
          this.canvas.dataset.overlayFrameId = boxes ? String(frame.id) : "";
          this.canvas.dataset.mediaTimestamp = String(frame.timestamp);
          this.canvas.dataset.faceBoxes = String(boxes);
          this.lastTimestamp = frame.timestamp;
        }
      }
      if (now - this.since >= 1000) {
        this.stats = {
          sourcePresentedFps: this.sourcePresented * 1000 / (now-this.since),
          captureCopyMs: this.captureWorkMs / Math.max(1,this.captured),
          renderDrawMs: this.drawWorkMs / Math.max(1,this.rendered),
          previewFps: (this.captured * 1000) / (now - this.since),
          renderFps: (this.rendered * 1000) / (now - this.since),
          faceOverlayFps: (this.overlayFrames * 1000) / (now - this.since),
          flowFps: (this.flowFrames * 1000) / (now - this.since),
          displayDelayMs: frame ? now - frame.capturedAt : null,
          lateResults: this.buffer.lateResults,
          bufferedFrames: this.buffer.frames.length,
          bufferBytes: this.buffer.bytes,
          skippedPreviewFrames: this.skipped,
        };
        this.onStats(this.stats);
        this.sourcePresented = this.captureWorkMs = this.drawWorkMs = 0;
        this.captured =
          this.rendered =
          this.overlayFrames =
          this.flowFrames =
            0;
        this.since = now;
      }
      this.paintHandle = requestAnimationFrame(paint);
    };
    this.captureHandle = this.video.requestVideoFrameCallback(capture);
    this.paintHandle = requestAnimationFrame(paint);
  }
  async encodeNext(onChunk = null) {
    if (!this.running) return null;
    let frame = this.buffer.frames.at(-1) || this.buffer.displayed;
    if (!frame || frame.id <= this.lastSent)
      frame = await new Promise((resolve) => this.waiters.push(resolve));
    if (!frame || !this.running) return null;
    const previousSent = this.lastSent;
    if (previousSent) this.skipped += Math.max(0, frame.id - previousSent - 1);
    this.lastSent = frame.id;
    const intermediate = [];
    let pixels = frame.width * frame.height;
    if (this.trackingEnabled && previousSent) {
      const available = this.buffer.frames.filter(
        (value) => value.id > previousSent && value.id < frame.id,
      );
      for (const value of available.reverse()) {
        if (
          intermediate.length >= 12 ||
          pixels + value.width * value.height > 32000000
        )
          break;
        intermediate.push(value);
        pixels += value.width * value.height;
      }
      intermediate.reverse();
    }
    const encodingStarted = performance.now();
    const selected = [frame, ...intermediate];
    const copies = await Promise.allSettled(
      selected.map(async (value) =>
        typeof value.bitmap.clone === "function"
          ? value.bitmap.clone()
          : createImageBitmap(value.bitmap),
      ),
    );
    if (copies.some((copy) => copy.status === "rejected")) {
      copies
        .filter((copy) => copy.status === "fulfilled")
        .forEach((copy) => copy.value.close());
      throw new Error("Không giữ được frame camera để mã hóa.");
    }
    const header = {
      type: "frame",
      frame_id: frame.id,
      timestamp: frame.timestamp,
      captured_at: frame.capturedAt,
      tracking_frames: intermediate.map((source) => ({
        frame_id: source.id,
        timestamp: source.timestamp,
        captured_at: source.capturedAt,
        width: source.width,
        height: source.height,
      })),
      sampling: {
        source_camera: this.sourceCamera,
        browser_visibility: document.visibilityState,
        browser_focused: document.hasFocus?.() ?? null,
        source_presented_fps: this.stats?.sourcePresentedFps ?? null,
        capture_copy_ms: this.stats?.captureCopyMs ?? null,
        render_draw_ms: this.stats?.renderDrawMs ?? null,
        encoding_ms_previous_packet: this.lastEncodingMs ?? null,
        preview_fps: this.stats?.previewFps ?? null,
        render_fps: this.stats?.renderFps ?? null,
        face_overlay_fps: this.stats?.faceOverlayFps ?? null,
        visual_tracking_fps: this.stats?.flowFps ?? null,
        visual_tracking_ms: this.visualTrackingMs ?? null,
        visual_tracking_summary: this.visualTrackingSummary ?? null,
        display_delay_ms: this.stats?.displayDelayMs ?? null,
        late_overlay_results: this.stats?.lateResults ?? 0,
        buffered_frames: this.stats?.bufferedFrames ?? null,
        buffer_bytes: this.stats?.bufferBytes ?? null,
        presented_frames: frame.id,
        skipped_preview_frames: this.skipped,
        visual_tracking_input_frames: intermediate.length,
      },
    };
    const trackingFrames = [];
    let blob = null;
    try {
      for (let index = 0; index < selected.length; index++) {
        const source = selected[index];
        this.encoder.width = source.width;
        this.encoder.height = source.height;
        const context = this.encoder.getContext("2d", { alpha: false });
        context.filter = index === 0 ? "none" : "grayscale(1)";
        context.drawImage(
          copies[index].value,
          0,
          0,
          source.width,
          source.height,
        );
        const encoded = await new Promise((resolve) =>
          this.encoder.toBlob(resolve, "image/jpeg", 0.97),
        );
        if (!this.running || !encoded) return null;
        if (onChunk && onChunk(index === 0 ? header : null, encoded) === false)
          return null;
        if (index === 0) blob = encoded;
        else
          trackingFrames.push({
            blob: encoded,
            metadata: {
              frame_id: source.id,
              timestamp: source.timestamp,
              captured_at: source.capturedAt,
              width: source.width,
              height: source.height,
            },
          });
      }
    } finally {
      this.lastEncodingMs = performance.now()-encodingStarted;
      for (const copy of copies) copy.value.close();
    }
    this.lastRequest = { ...header, width: frame.width, height: frame.height };
    return { header, blob, trackingFrames, sent: !!onChunk };
  }
  acceptVisual(payload) {
    if (!this.running) return;
    this.visualTrackingMs = payload.visual_tracking_ms;
    this.visualTrackingSummary = payload.visual_tracking_summary;
    for (const tracked of payload.frames || []) {
      if (this.buffer.attach(tracked))
        this.buffer.observeLatency(performance.now() - tracked.captured_at);
    }
  }
  accept(result) {
    if (!this.running) return false;
    for (const tracked of result.tracked_frames || []) {
      if (this.buffer.attach(tracked))
        this.buffer.observeLatency(performance.now() - tracked.captured_at);
    }
    const request = this.lastRequest;
    if (
      request &&
      result.frame_id === request.frame_id &&
      result.captured_at === request.captured_at &&
      result.width === request.width &&
      result.height === request.height &&
      Number.isFinite(result.media_timestamp) &&
      Math.abs(result.media_timestamp - request.timestamp) <= 0.0001
    ) {
      this.buffer.observeLatency(performance.now() - request.captured_at);
    }
    return this.buffer.attach(result);
  }
  stop() {
    this.running = false;
    this.lastRequest = null;
    this.trackingEnabled = false;
    this.video.cancelVideoFrameCallback?.(this.captureHandle);
    cancelAnimationFrame(this.paintHandle);
    for (const resolve of this.waiters.splice(0)) resolve(null);
    this.buffer.clear();
    this.canvas
      .getContext("2d")
      ?.clearRect(0, 0, this.canvas.width, this.canvas.height);
    delete this.canvas.dataset.frameId;
    delete this.canvas.dataset.overlayFrameId;
    delete this.canvas.dataset.faceBoxes;
    delete this.canvas.dataset.mediaTimestamp;
    this.canvasFrameId = null;
    this.canvasResult = null;
    this.encoder.width = this.encoder.height = 1;
  }
}
