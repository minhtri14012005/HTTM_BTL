// Isolated renderer contract tests: generated video + scripted AI replies; no physical camera or user data.
const { chromium } = require("../.tools/qa/node_modules/playwright");
const assert = require("node:assert/strict");
function jpegSize(bytes) {
  const b = Buffer.from(bytes);
  let i = 2;
  while (i < b.length) {
    if (b[i++] !== 255) continue;
    let marker = b[i++];
    while (marker === 255) marker = b[i++];
    if (marker === 216 || marker === 217) continue;
    const length = b.readUInt16BE(i);
    if ([192, 193, 194].includes(marker))
      return { width: b.readUInt16BE(i + 5), height: b.readUInt16BE(i + 3) };
    i += length;
  }
  throw Error("Invalid JPEG fixture");
}
(async () => {
  if (process.argv.includes("--real-ai")) {
    await checkRealAI();
    return;
  }
  const diagnostic = process.argv.includes("--diagnostic");
  const closeTab = process.argv.includes("--close-tab");
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const context = await browser.newContext({
    viewport: { width: 1500, height: 1000 },
  });
  const page = await context.newPage(),
    errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  let delay = 65,
    session = 0,
    mode = "candidate",
    headers = [],
    maxInFlight = 0,
    inFlight = 0;
  let diagnosticState = "opening",
    diagnosticStarted = 0,
    diagnosticCreates = 0;
  const diagnosticView = () => {
    if (
      diagnosticState === "running" &&
      Date.now() - diagnosticStarted >= 30000
    )
      diagnosticState = "complete";
    return {
      id: "qa-run",
      state: diagnosticState,
      prepare_remaining: 5,
      remaining: Math.max(0, 30 - (Date.now() - diagnosticStarted) / 1000),
      summary:
        diagnosticState === "complete"
          ? {
              frames: headers.length,
              duration_seconds: 30,
              false_alerts: 0,
              ground_truth: "QA only",
              verification: "Not a physical camera test",
            }
          : null,
    };
  };
  await page.route("**/api/**", async (route) => {
    const p = new URL(route.request().url()).pathname;
    if (p === "/api/diagnostics" && route.request().method() === "GET") {
      const state = diagnosticView();
      const analysis =
        state.state === "complete" && Date.now() - diagnosticStarted >= 30500
          ? {
              title: "QA auto analysis completed",
              models: {},
              findings: ["Isolated QA fixture"],
              source_sizes: {},
              ground_truth_runs: {},
            }
          : null;
      await route.fulfill({
        json: {
          runs:
            state.state === "complete"
              ? [{ ...state, scenario: "Chính diện 1 m", analysis }]
              : [],
        },
      });
      return;
    }
    if (p === "/api/diagnostics" && route.request().method() === "POST")
      diagnosticCreates++;
    if (p.endsWith("/begin")) {
      diagnosticState = "running";
      diagnosticStarted = Date.now();
    }
    const body =
      p === "/api/health"
        ? {
            ready: true,
            device: "qa",
            models: {},
            settings: { calibrated: false },
          }
        : p === "/api/persons"
          ? []
          : p === "/api/camera/start"
            ? { session_id: `qa-${++session}` }
            : p.startsWith("/api/diagnostics")
              ? diagnosticView()
              : { ok: true };
    await route.fulfill({ json: body });
  });
  await page.addInitScript((diagnostic) => {
    window.__physicalCameraCalls = 0;
    navigator.mediaDevices.enumerateDevices = async () => [
      {
        kind: "videoinput",
        deviceId: "qa-video",
        label: diagnostic ? "Camo Camera QA" : "Integrated QA Camera",
      },
    ];
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      if (constraints.video.deviceId.exact !== "qa-video")
        throw Error("Unexpected camera selection");
      if (constraints.video.width.ideal !== 1920)
        throw Error("Recognition input was reduced");
      const fixture = document.createElement("canvas");
      fixture.width = 1920;
      fixture.height = 1080;
      const stream = fixture.captureStream(30);
      window.__qaFixture = fixture;
      window.__qaStream = stream;
      let i = 0;
      const paint = () => {
        const c = fixture.getContext("2d");
        c.fillStyle = "#c4d7ed";
        c.fillRect(0, 0, fixture.width, fixture.height);
        c.fillStyle = "#245c9f";
        c.fillRect(500 + (i++ % 150), 400, 100, 100);
        if (stream.getVideoTracks()[0].readyState === "live")
          requestAnimationFrame(paint);
      };
      paint();
      return stream;
    };
  }, diagnostic);
  await page.routeWebSocket("**/ws/live", (ws) => {
    let head, hello;
    ws.onMessage((message) => {
      if (typeof message === "string") {
        const data = JSON.parse(message);
        if (data.kind) {
          hello = data;
          assert.equal(data.return_image, false);
          ws.send(
            JSON.stringify({ type: "ready", session_id: hello.session_id }),
          );
        } else head = data;
        return;
      }
      const h = head,
        { width, height } = jpegSize(message);
      headers.push({ ...h, width, height });
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      const status = mode;
      setTimeout(() => {
        inFlight--;
        if (diagnostic && diagnosticState === "opening")
          diagnosticState = "preparing";
        try {
          ws.send(
            JSON.stringify({
              type: "live_frame",
              session_id: hello.session_id,
              frame_id: h.frame_id,
              timestamp: h.timestamp,
              media_timestamp: h.timestamp,
              captured_at: h.captured_at,
              width,
              height,
              processing_ms: delay,
              ai_fps: 1000 / delay,
              detector_ms: 5,
              backlog: 0,
              events: [],
              diagnostic: diagnostic ? diagnosticView() : undefined,
              overlays: [
                {
                  face_bbox: [100, 100, 300, 300],
                  status,
                  person_id: status === "confirmed" ? "qa" : null,
                  name: status === "confirmed" ? "QA CONTRACT" : "HIDDEN NAME",
                },
              ],
            }),
          );
        } catch {}
      }, delay);
    });
  });
  try {
    const mainPage = closeTab ? await context.newPage() : null;
    if (mainPage) await mainPage.goto("http://127.0.0.1:8000/");
    if (closeTab) {
      await page.evaluate(() => location.replace("http://127.0.0.1:8000/?diagnostic=camo&close_after=1"));
      await page.waitForURL("http://127.0.0.1:8000/**");
    } else await page.goto("http://127.0.0.1:8000/");
    if (diagnostic) {
      if (!closeTab) {
        await page.getByRole("button", { name: "Chẩn đoán", exact: true }).click();
        await page.getByRole("button", { name: "Kiểm thử AI bằng Camo Camera", exact: true }).click();
      }
      await page.getByRole("heading", { name: "Bạn đã sẵn sàng?", exact: true }).waitFor();
      assert.equal(await page.locator(".diagnostic-screen").count(), 0);
      await page.getByLabel("Mục tiêu trong khung hình").selectOption("false");
      await page.evaluate(() => {
        window.__qaClock = [];
        new MutationObserver(() => {
          const clock = document.querySelector(".diagnostic-clock");
          if (clock) {
            const value = clock.textContent;
            if (window.__qaClock.at(-1) !== value) window.__qaClock.push(value);
          }
        }).observe(document.body, {
          subtree: true,
          childList: true,
          characterData: true,
        });
      });
      await page
        .getByRole("button", { name: "Tôi đã sẵn sàng – Bắt đầu", exact: true })
        .click();
      await page.waitForFunction(
        () =>
          document
            .querySelector(".diagnostic-clock")
            ?.textContent.includes("CHUẨN BỊ"),
        null,
        { timeout: 8000 },
      );
      const action = page.getByRole("status", {
        name: "Chỉ dẫn động tác kiểm thử",
      });
      await action.getByText("ĐỨNG CÁCH CAMERA 1 M", { exact: true }).waitFor();
      assert.ok(
        (await action
          .locator("strong")
          .first()
          .evaluate((el) => parseFloat(getComputedStyle(el).fontSize))) >= 26,
      );
      const layout = await page.evaluate(() => {
        const camera = document
          .querySelector(".diagnostic-camera")
          .getBoundingClientRect();
        return [".diagnostic-clock", ".diagnostic-action"].map((selector) => {
          const panel = document
            .querySelector(selector)
            .getBoundingClientRect();
          return (
            Math.min(camera.right, panel.right) >
              Math.max(camera.left, panel.left) &&
            Math.min(camera.bottom, panel.bottom) >
              Math.max(camera.top, panel.top)
          );
        });
      });
      assert.deepEqual(
        layout,
        [false, false],
        "Instructions and countdown must never cover camera pixels",
      );
      await page.screenshot({
        path: ".tools/qa/diagnostic-large-instructions.png",
        fullPage: true,
      });
      await page.waitForFunction(
        () =>
          document
            .querySelector(".diagnostic-clock")
            ?.textContent.includes("BẮT ĐẦU KIỂM THỬ"),
        null,
        { timeout: 8000 },
      );
      assert.equal(await page.locator(".diagnostic-camera canvas").count(), 1);
      assert.equal(
        await page.locator(".analysis-inset,.diagnostic-analyzed").count(),
        0,
      );
      await page.waitForFunction(
        () =>
          document
            .querySelector(".diagnostic-clock")
            ?.textContent.includes("HOÀN THÀNH"),
        null,
        { timeout: 35000, polling: "raf" },
      );
      assert.equal(
        await page.evaluate(
          () => window.__qaStream.getVideoTracks()[0].readyState,
        ),
        "ended",
      );
      if (closeTab) {
        const clock = await page.evaluate(() => window.__qaClock);
        await page.waitForEvent("close", { timeout: 4000 });
        assert.equal(mainPage.isClosed(), false);
        assert.ok(clock.includes("HOÀN THÀNH0"));
        assert.equal(diagnosticCreates,1);
        assert.deepEqual(errors,[]);
        console.log(JSON.stringify({ dedicated_test_tab_closed:true,main_tab_preserved:true,camera_released:true,runs:diagnosticCreates,physical_camera_calls:0 }));
        return;
      }
      await page.waitForFunction(
        () => !document.querySelector(".diagnostic-screen"),
        null,
        { timeout: 4000 },
      );
      await page
        .getByText("QA auto analysis completed", { exact: true })
        .waitFor({ timeout: 5000 });
      const clock = await page.evaluate(() => window.__qaClock);
      for (let i = 5; i >= 1; i--) assert.ok(clock.includes(`CHUẨN BỊ${i}`));
      assert.ok(clock.includes("BẮT ĐẦU KIỂM THỬ30"));
      assert.ok(clock.includes("HOÀN THÀNH0"));
      assert.equal(diagnosticCreates, 1);
      assert.deepEqual(errors, []);
      console.log(
        JSON.stringify({
          isolated_diagnostic_ui: true,
          prepare: "5→1",
          duration: "30→0",
          single_surface: true,
          automatic_close: true,
          auto_analysis_without_reload: true,
          camera_released: true,
          runs: diagnosticCreates,
          physical_camera_calls: 0,
        }),
      );
      return;
    }
    await page
      .getByRole("button", { name: "Bắt đầu giám sát", exact: true })
      .click();
    await page.waitForFunction(
      () =>
        Number(
          document.querySelector(".live-stage canvas")?.dataset.faceBoxes,
        ) > 0,
      null,
      { timeout: 20000, polling: "raf" },
    );
    assert.equal(
      await page.locator(".analysis-inset,.diagnostic-analyzed").count(),
      0,
    );
    assert.equal(await page.locator(".live-stage canvas").count(), 1);
    const rect = await page.locator(".live-stage video").boundingBox();
    assert.ok(rect.x + rect.width < 0);
    const read = () =>
      page.locator(".live-stage canvas").evaluate((c) => ({
        id: +c.dataset.frameId,
        overlay: c.dataset.overlayFrameId,
        width: c.width,
        height: c.height,
        timestamp: +c.dataset.mediaTimestamp,
        boxes: +c.dataset.faceBoxes,
      }));
    const first = await read();
    assert.equal(first.width, 1920);
    assert.equal(first.height, 1080);
    assert.ok(headers.every((h) => h.width === 1920 && h.height === 1080));
    await page.waitForFunction(
      () => {
        const c = document.querySelector(".live-stage canvas");
        return (
          c.dataset.faceBoxes === "1" &&
          c.dataset.overlayFrameId === c.dataset.frameId
        );
      },
      null,
      { polling: "raf" },
    );
    await page.screenshot({
      path: ".tools/qa/camera-single-contract.png",
      fullPage: true,
    });
    delay = 360;
    const moderateStart = await read();
    await page.evaluate((startId) => {
      window.__qaModerateMatches = new Set();
      const canvas = document.querySelector(".live-stage canvas");
      window.__qaModerateObserver = new MutationObserver(() => {
        const id = Number(canvas.dataset.frameId);
        if (
          id > startId + 40 &&
          canvas.dataset.faceBoxes === "1" &&
          canvas.dataset.overlayFrameId === String(id)
        )
          window.__qaModerateMatches.add(id);
      });
      window.__qaModerateObserver.observe(canvas, { attributes: true });
    }, moderateStart.id);
    await page.waitForFunction(
      () => window.__qaModerateMatches.size >= 3,
      null,
      { timeout: 15000, polling: "raf" },
    );
    await page.evaluate(() => window.__qaModerateObserver.disconnect());
    const adaptiveMatches = await page.evaluate(
      () => window.__qaModerateMatches.size,
    );
    delay = 900;
    const start = await read();
    await page.waitForFunction(
      (id) =>
        +document.querySelector(".live-stage canvas").dataset.frameId >=
        id + 12,
      start.id,
      { timeout: 10000, polling: "raf" },
    );
    const advanced = await read();
    assert.ok(advanced.timestamp > start.timestamp);
    assert.equal(await page.locator(".diagnostic-entry").count(), 0);
    // Timing counters are now intentionally visible only in diagnostic mode.
    await page.getByRole("button", { name: "Chẩn đoán", exact: true }).click();
    await page.waitForFunction(
      () =>
        /Kết quả đến muộn không vẽ: [1-9]/.test(
          document.querySelector("main").textContent,
        ),
      null,
      { timeout: 10000 },
    );
    // Slow inference must not make an old box follow newer video frames.
    let matched = 0;
    for (let i = 0; i < 20; i++) {
      const f = await read();
      if (f.overlay) {
        assert.equal(f.overlay, String(f.id));
        matched++;
      }
    }
    await page.evaluate(() => {
      window.__qaFixture.width = 1080;
      window.__qaFixture.height = 1920;
    });
    delay = 65;
    await page.waitForFunction(
      () => {
        const c = document.querySelector(".live-stage canvas");
        return c.width === 1080 && c.height === 1920;
      },
      null,
      { timeout: 10000 },
    );
    await page.waitForFunction(
      () =>
        Number(
          document.querySelector(".live-stage canvas")?.dataset.faceBoxes,
        ) > 0,
      null,
      { timeout: 10000, polling: "raf" },
    );
    assert.ok(headers.some((h) => h.width === 1080 && h.height === 1920));
    await page
      .getByRole("button", { name: "Dừng giám sát", exact: true })
      .click();
    assert.equal(
      await page.evaluate(
        () => window.__qaStream.getVideoTracks()[0].readyState,
      ),
      "ended",
    );
    assert.equal(await page.evaluate(() => window.__physicalCameraCalls), 0);
    assert.equal(maxInFlight, 1);
    assert.deepEqual(errors, []);
    console.log(
      JSON.stringify({
        single_surface: true,
        native_input: "1920x1080 → 1080x1920",
        exact_frame_overlay: true,
        overlays_at_360ms: adaptiveMatches,
        slow_ai_video_advanced: advanced.id - start.id,
        max_in_flight: maxInFlight,
        camera_released: true,
        physical_camera_calls: 0,
        js_errors: errors,
      }),
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});

async function checkRealAI() {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const page = await browser.newPage({
      viewport: { width: 1500, height: 1000 },
    }),
    errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/__qa_camera.mp4", (r) =>
    r.fulfill({
      path: require("node:path").resolve(process.env.FACEWATCH_QA_CAMERA_SOURCE || ".tools/qa/camera-source.mp4"),
      contentType: "video/mp4",
    }),
  );
  await page.addInitScript(({ startAt, disableVisual }) => {
    const NativeWebSocket = window.WebSocket;
    window.__qaAiMessages = [];
    window.__qaVisualMessages = [];
    window.WebSocket = class extends NativeWebSocket {
      constructor(...args) {
        super(...args);
        this.addEventListener("message", event => {
          try {
            const value = JSON.parse(event.data);
            if (value.type === "live_frame") window.__qaAiMessages.push({ at: performance.now(), ms: value.processing_ms });
            if (value.type === "visual_frames") window.__qaVisualMessages.push({ at: performance.now(), ms: value.visual_tracking_ms });
          } catch {}
        });
      }
      send(value) {
        if (disableVisual && typeof value === "string") {
          try {
            const header = JSON.parse(value);
            if (header.kind && header.label) value = JSON.stringify({ ...header, track_between_frames: false });
          } catch {}
        }
        return super.send(value);
      }
    };
    navigator.mediaDevices.enumerateDevices = async () => [
      {
        kind: "videoinput",
        deviceId: "qa-prerecorded",
        label: "Integrated QA Camera",
      },
    ];
    navigator.mediaDevices.getUserMedia = async () => {
      const source = document.createElement("video");
      source.src = "/__qa_camera.mp4";
      source.addEventListener("loadedmetadata", () => { source.currentTime = startAt; }, { once: true });
      source.muted = true;
      source.loop = true;
      source.playsInline = true;
      source.style.cssText =
        "position:fixed;left:-10000px;width:1px;height:1px";
      document.body.appendChild(source);
      await Promise.race([
        source.play(),
        new Promise((_, reject) =>
          setTimeout(() => reject(Error("QA clip did not start")), 5000),
        ),
      ]);
      const stream = source.captureStream();
      window.__qaStreamStartedAt = performance.now();
      window.__qaStream = stream;
      return stream;
    };
    window.__qaObserved = [];
    function observe() {
      const c = document.querySelector(".live-stage canvas");
      if (c?.dataset.frameId) {
        const id = +c.dataset.frameId,
          overlay = c.dataset.overlayFrameId;
        if (window.__qaObserved.at(-1)?.id !== id)
          window.__qaObserved.push({
            id,
            overlay,
            timestamp: +c.dataset.mediaTimestamp,
            at: performance.now(),
          });
        else window.__qaObserved.at(-1).overlay = overlay;
      }
      requestAnimationFrame(observe);
    }
    requestAnimationFrame(observe);
  }, { startAt: Number(process.env.FACEWATCH_QA_START || 0), disableVisual: process.argv.includes("--no-visual") });
  try {
    await page.goto("http://127.0.0.1:8000/");
    await page
      .getByRole("button", { name: "Bắt đầu giám sát", exact: true })
      .click({ timeout: 60000 });
    await page.waitForFunction(
      () =>
        window.__qaObserved.filter((f) => f.overlay).length >= 3 ||
        document.querySelector(".alert"),
      null,
      { timeout: 20000, polling: "raf" },
    );
    assert.deepEqual(await page.locator(".alert").allTextContents(), []);
    await page.waitForFunction(() => performance.now()-window.__qaStreamStartedAt >= 8500);
    const performanceSample = await page.evaluate(() => {
      const start = window.__qaStreamStartedAt+2000, end = start+6000;
      const within = frame => frame.at >= start && frame.at < end;
      const frames = window.__qaObserved.filter(within);
      const ai = window.__qaAiMessages.filter(within);
      const visual = window.__qaVisualMessages.filter(within);
      return { render_fps: frames.length/6, face_overlay_fps: frames.filter(frame => frame.overlay).length/6,
        ai_fps: ai.length/6, ai_ms_mean: ai.reduce((sum,frame) => sum+frame.ms,0)/Math.max(1,ai.length),
        visual_ms_mean: visual.reduce((sum,frame) => sum+frame.ms,0)/Math.max(1,visual.length) };
    });
    if(process.env.FACEWATCH_QA_CAMERA_SOURCE && !process.argv.includes("--no-visual")) {
      assert.ok(performanceSample.render_fps >= 13,JSON.stringify(performanceSample));
      assert.ok(performanceSample.face_overlay_fps >= performanceSample.render_fps*.9,JSON.stringify(performanceSample));
    }
    const observed = await page.evaluate(() => window.__qaObserved);
    assert.ok(observed.length >= 3);
    for (const f of observed)
      if (f.overlay) assert.equal(f.overlay, String(f.id));
    assert.equal(
      await page.locator(".analysis-inset,.diagnostic-analyzed").count(),
      0,
    );
    await page.screenshot({
      path: ".tools/qa/camera-single-real.png",
      fullPage: true,
    });
    await page
      .getByRole("button", { name: "Dừng giám sát", exact: true })
      .click();
    assert.equal(
      await page.evaluate(
        () => window.__qaStream.getVideoTracks()[0].readyState,
      ),
      "ended",
    );
    assert.deepEqual(errors, []);
    console.log(
      JSON.stringify({
        real_ai_prerecorded_clip: true,
        performance: performanceSample,
        displayed_frames: observed.length,
        aligned_overlay_frames: observed.filter((f) => f.overlay).length,
        camera_released: true,
        physical_camera_calls: 0,
        js_errors: errors,
      }),
    );
  } finally {
    await browser.close();
  }
}
