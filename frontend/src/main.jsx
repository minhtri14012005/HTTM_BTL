import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Video,
  SquarePlay,
  Users,
  Search,
  Play,
  Pause,
  Square,
  RotateCcw,
  Plus,
  Check,
  Maximize,
  Camera,
  Upload,
  Trash2,
  X,
  ImagePlus,
  ChevronRight,
  LoaderCircle,
  Wifi,
  AlertCircle,
  ScanFace,
} from "lucide-react";
import { api, json, socket } from "./api";
import { acquireCamera, cameraFor, frameAt, mergeEvents, visibleEvents } from "./sync";
import "./styles.css";
import Diagnostic, { RecognitionNotice } from "./Diagnostic";
import { CameraView } from "./camera-view";
import VideoGroundTruth, { VideoEvaluationHistory } from "./VideoGroundTruth";

const titles = {
  live: "Camera trực tiếp",
  video: "Phân tích video",
  people: "Danh sách người",
};
const time = (t) =>
  `${String(Math.floor(Math.max(0, t || 0) / 60)).padStart(2, "0")}:${String(Math.floor(Math.max(0, t || 0) % 60)).padStart(2, "0")}`;
const initial = (name) => name?.trim().charAt(0).toLocaleUpperCase("vi") || "?";
function Badge({ children, green = false }) {
  return <span className={`badge ${green ? "green" : ""}`}>{children}</span>;
}
function Empty({ icon: Icon = ScanFace, title, children }) {
  return (
    <div className="empty">
      <Icon size={30} />
      <strong>{title}</strong>
      <span>{children}</span>
    </div>
  );
}
function Card({ title, badge, children, footer, className = "" }) {
  return (
    <section className={`card ${className}`}>
      <div className="card-head">
        <h2>{title}</h2>
        {badge}
      </div>
      {children}
      {footer && <div className="card-footer">{footer}</div>}
    </section>
  );
}
function Stats({ items }) {
  return (
    <div className="stats">
      {items.map(([label, value, unit]) => (
        <div className="stat" key={label}>
          <span>{label}</span>
          <div>
            {value ?? "—"} <small>{unit}</small>
          </div>
        </div>
      ))}
    </div>
  );
}
function Detections({ events, video = false, calibrated = false }) {
  return (
    <Card
      title={video ? "Phát hiện trong video" : "Thông báo phát hiện"}
      badge={
        <Badge green>
          <Wifi size={10} /> {video ? "CẬP NHẬT NGAY" : "REALTIME"}
        </Badge>
      }
      footer={
        video
          ? "Kết quả hiển thị theo thời điểm đang phát. Ảnh bằng chứng được trích từ frame đã xác nhận."
          : "Thông báo chỉ xuất hiện khi AI xác nhận người thuộc danh sách cần tìm."
      }
    >
      <div className="detections">
        {events.length ? (
          [...events].reverse().map((e) => (
            <div className="event" key={e.id}>
              <img src={e.evidence} alt={`Bằng chứng ${e.name}`} />
              <div>
                <strong>{e.name}</strong>
                <span>
                  {video
                    ? `Video Upload · ${time(e.timestamp)}`
                    : `${new Date(e.occurred_at).toLocaleTimeString("vi-VN")} · ${e.source}`}
                </span>
                <span>Đã tìm thấy người cần tìm</span>
              </div>
              <Badge green>Xác nhận</Badge>
            </div>
          ))
        ) : (
          <Empty title="Chưa có danh tính được xác nhận">
            {calibrated
              ? "Thông báo xuất hiện sau khi danh tính được xác nhận."
              : "Chưa hiệu chuẩn bước xác nhận. Trạng thái đang xác minh chưa tạo thông báo."}
          </Empty>
        )}
      </div>
    </Card>
  );
}
function Watchlist({ people, onToggle }) {
  return (
    <Card
      title="Danh sách đang tìm kiếm"
      badge={<Badge>{people.filter((p) => p.selected).length} người</Badge>}
    >
      <div className="watchlist">
        {people.length ? (
          people.map((p) => (
            <label className="watch-row" key={p.id}>
              <input
                type="checkbox"
                checked={p.selected}
                onChange={() => onToggle(p)}
              />
              <span className="avatar">
                {p.images[0] ? (
                  <img src={p.images[0].url} alt="" />
                ) : (
                  initial(p.name)
                )}
              </span>
              <span>{p.name}</span>
              <small>{p.selected ? "Đã chọn" : "Chưa chọn"}</small>
            </label>
          ))
        ) : (
          <Empty icon={Users} title="Danh sách trống">
            Đăng ký người ở mục Danh sách người.
          </Empty>
        )}
      </div>
    </Card>
  );
}

function Overlay({ frame, review = false }) {
  if (!frame) return null;
  return (
    <svg
      className="bbox-layer"
      viewBox={`0 0 ${frame.width} ${frame.height}`}
      preserveAspectRatio="xMidYMid meet"
      aria-label="Kết quả AI trên frame"
    >
      <>
        {frame.overlays.map((o, i) => {
          let [x1, y1, x2, y2] = o.face_bbox || o.bbox;
          const confirmed = o.status === "confirmed" && o.person_id && o.name;
          const tracked =
            o.status === "tracked" &&
            o.identity_source === "track_continuity" &&
            o.person_id &&
            o.name;
          const color = confirmed || tracked ? "#0bbf88" : "#f05b66";
          const recognitionLabel = tracked
            ? `${o.name} · theo dõi`
            : confirmed
              ? `${o.name} · xác nhận`
              : o.status === "candidate"
                ? "Đang xác minh"
                : "Chưa xác định";
          const label = review
            ? `Mặt ${i + 1}${confirmed || tracked ? ` · ${o.name}${tracked ? " · theo dõi" : ""}` : ""}`
            : recognitionLabel;
          const font = Math.max(12, frame.width / 85);
          return (
            <g key={i}>
              <rect
                x={x1}
                y={y1}
                width={x2 - x1}
                height={y2 - y1}
                fill="none"
                stroke={color}
                strokeWidth={Math.max(2, frame.width / 600)}
              />
              <rect
                x={x1}
                y={Math.max(0, y1 - font * 1.7)}
                width={label.length * font * 0.59 + 12}
                height={font * 1.7}
                fill={color}
                rx="3"
              />
              <text
                x={x1 + 5}
                y={Math.max(font * 1.2, y1 - font * 0.45)}
                fill="white"
                fontSize={font}
                fontWeight="600"
              >
                {label}
              </text>
            </g>
          );
        })}
      </>
    </svg>
  );
}

function Live({ people, onToggle, health, onError, tools }) {
  const [kind, setKind] = useState("laptop"),
    [active, setActive] = useState(false),
    [connecting, setConnecting] = useState(false),
    [device, setDevice] = useState(""),
    [result, setResult] = useState(null),
    [events, setEvents] = useState([]),
    [previewFps, setPreviewFps] = useState(null),
    [latency, setLatency] = useState(null),
    [viewStats, setViewStats] = useState(null);
  const video = useRef(null),
    surface = useRef(null),
    view = useRef(null),
    stream = useRef(null),
    ws = useRef(null),
    running = useRef(false),
    connectId = useRef(0),
    ownedSession = useRef(null);
  const stop = () => {
    connectId.current++;
    running.current = false;
    ws.current?.close();
    ws.current = null;
    view.current?.stop();
    view.current = null;
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
    setActive(false);
    setConnecting(false);
    setResult(null);
    setPreviewFps(null);
    setLatency(null);
    setViewStats(null);
    if (ownedSession.current) {
      const id = ownedSession.current;
      ownedSession.current = null;
      return api("/camera/stop", json("POST", { session_id: id })).catch(
        () => {},
      );
    }
  };
  useEffect(
    () => () => {
      running.current = false;
      connectId.current++;
      ws.current?.close();
      view.current?.stop();
      stream.current?.getTracks().forEach((t) => t.stop());
      if (ownedSession.current) {
        api(
          "/camera/stop",
          json("POST", { session_id: ownedSession.current }),
        ).catch(() => {});
        ownedSession.current = null;
      }
    },
    [],
  );
  async function sendFrame(client) {
    if (
      !running.current ||
      client !== ws.current ||
      client.readyState !== WebSocket.OPEN
    )
      return;
    try {
      await view.current?.encodeNext((header, blob) => {
        if (
          !running.current ||
          client !== ws.current ||
          client.readyState !== WebSocket.OPEN
        )
          return false;
        if (header) client.send(JSON.stringify(header));
        client.send(blob);
        return true;
      });
    } catch (error) {
      if (client === ws.current && running.current) {
        onError(error.message);
        stop();
      }
    }
  }

  async function start() {
    setConnecting(true);
    const attempt = ++connectId.current;
    try {
      if (!health?.ready)
        throw new Error(health?.error || "Model đang khởi động. Vui lòng chờ.");
      const devices = await navigator.mediaDevices.enumerateDevices();
      const selected = cameraFor(devices, kind);
      if (!selected)
        throw new Error(
          kind === "camo"
            ? "Không tìm thấy Camo Camera. Bật Camo, kết nối iPhone và cấp quyền Camera cho localhost trong cài đặt trình duyệt."
            : "Không xác định được webcam laptop. Cấp quyền Camera cho localhost trong cài đặt trình duyệt và kiểm tra thiết bị tích hợp.",
        );
      await api(
        "/camera/validate",
        json("POST", { kind, label: selected.label }),
      );
      const acquired = await acquireCamera(navigator.mediaDevices, selected, kind);
      if (connectId.current !== attempt) {
        acquired.getTracks().forEach((t) => t.stop());
        return;
      }
      stream.current = acquired;
      setDevice(selected.label);
      video.current.srcObject = acquired;
      await video.current.play();
      if (connectId.current !== attempt) return;
      view.current = new CameraView(
        video.current,
        surface.current,
        (stats) => {
          setPreviewFps(Math.round(stats.renderFps));
          setViewStats(stats);
        },
        (message) => {
          onError(message);
          stop();
        },
      );
      view.current.start();
      acquired.getVideoTracks()[0].onended = () => {
        onError("Camera đã ngắt kết nối.");
        stop();
      };
      const reserved = await api(
        "/camera/start",
        json("POST", { kind, label: selected.label }),
      );
      if (connectId.current !== attempt) {
        api(
          "/camera/stop",
          json("POST", { session_id: reserved.session_id }),
        ).catch(() => {});
        acquired.getTracks().forEach((t) => t.stop());
        return;
      }
      ownedSession.current = reserved.session_id;
      const client = socket("/ws/live");
      ws.current = client;
      client.onopen = () =>
        client.send(
          JSON.stringify({
            kind,
            label: selected.label,
            session_id: reserved.session_id,
            return_image: false,
            track_between_frames: true,
          }),
        );
      client.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (client !== ws.current) return;
        if (data.type === "error") {
          onError(data.detail);
          stop();
          return;
        }
        if (data.type === "ready") {
          view.current.trackingEnabled = data.track_between_frames === true;
          running.current = true;
          setActive(true);
          setConnecting(false);
          setEvents([]);
          sendFrame(client);
        }
        if (data.type === "visual_frames") {
          if (data.session_id === ownedSession.current)
            view.current?.acceptVisual(data);
          return;
        }
        if (data.type === "live_frame") {
          if (data.session_id !== ownedSession.current) return;
          view.current?.accept(data);
          setResult(data);
          setLatency(Math.round(performance.now() - data.captured_at));
          setEvents((old) => mergeEvents(old, data.events));
          sendFrame(client);
        }
      };
      client.onerror = () => {
        onError("Mất kết nối WebSocket camera.");
        stop();
      };
      client.onclose = () => {
        if (client === ws.current && running.current) {
          stop();
          onError("Phiên camera đã đóng. Có thể bắt đầu lại.");
        }
      };
    } catch (error) {
      if (connectId.current === attempt) {
        stop();
        onError(error.message);
      }
    }
  }
  return (
    <>
      <div className="toolbar">
        <div className="toolbar-group">
          <label>Nguồn camera</label>
          <select
            aria-label="Nguồn camera"
            value={kind}
            disabled={connecting}
            onChange={async (e) => { const next = e.target.value; await stop(); setKind(next); }}
          >
            <option value="laptop">Webcam laptop</option>
            <option value="camo">Camo Camera (iPhone 13)</option>
          </select>
          <span className="soft-chip">
            <RotateCcw size={12} /> Tự động xoay hình
          </span>
        </div>
        <div className="toolbar-group">
          <Badge green={active}>
            {active
              ? "Đã kết nối"
              : connecting
                ? "Đang kết nối"
                : "Chưa kết nối thiết bị"}
          </Badge>
          <button
            className="primary"
            disabled={connecting || !health?.ready}
            onClick={active ? stop : start}
          >
            {connecting ? (
              <LoaderCircle className="spin" size={15} />
            ) : active ? (
              <Square size={14} />
            ) : (
              <Play size={14} fill="currentColor" />
            )}
            {active ? "Dừng giám sát" : "Bắt đầu giám sát"}
          </button>
        </div>
      </div>
      <RecognitionNotice
        calibrated={health?.settings?.calibrated}
        compact={!tools}
        status={health?.settings?.calibration_status}
      />
      {tools && (
        <Diagnostic
          people={people}
          health={health}
          onError={onError}
          stopLive={stop}
        />
      )}
      <div className="workspace-grid">
        <div>
          <Card
            title="Khung hình giám sát"
            badge={
              <span className="muted">
                {device || "Webcam laptop / Camo Camera"}
              </span>
            }
            footer="Xanh: đã xác nhận người cần tìm · Đỏ: chưa xác định."
          >
            <div className="video-stage live-stage">
              <video
                ref={video}
                muted
                playsInline
                className="camera-capture-source"
                aria-hidden="true"
                tabIndex={-1}
              />
              <canvas
                ref={surface}
                className="camera-ai-surface"
                aria-label="Camera AI trực tiếp"
              />
              {!active && (
                <div className="stage-empty">
                  <Camera size={44} />
                  <strong>Sẵn sàng kết nối camera</strong>
                  <span>Chọn nguồn camera và bắt đầu giám sát</span>
                </div>
              )}
              {active && tools && (
                <span className="stage-note top">
                  Giữ nguyên tỷ lệ · Tự động định hướng
                </span>
              )}
              {active && tools && (
                <span className="stage-note bottom">
                  Camera AI · Frame{" "}
                  {viewStats ? surface.current?.dataset.frameId : "…"} · Đồng bộ
                  frame ID
                </span>
              )}
            </div>
          </Card>
          <Stats
            items={[
              ["Video FPS", previewFps, "FPS hiển thị thực"],
              [
                "AI FPS",
                result?.ai_throughput_fps?.toFixed(1),
                "Frame hoàn thành/giây",
              ],
              ["Độ trễ AI", latency, "ms · capture → UI"],
            ]}
          />
          {tools && (
            <div className="muted">
              Detector: {result ? `${result.detector_ms.toFixed(0)} ms` : "—"} ·
              Backlog: {result?.backlog ?? "—"} · Đệm hiển thị:{" "}
              {viewStats?.displayDelayMs?.toFixed(0) ?? "—"} ms · Kết quả đến
              muộn không vẽ: {viewStats?.lateResults ?? 0} · Frame không được AI
              phân tích: {viewStats?.skippedPreviewFrames ?? 0}
            </div>
          )}
        </div>
        <aside className="right-column">
          <Detections
            events={events}
            calibrated={health?.settings?.calibrated}
          />
          <Watchlist people={people} onToggle={onToggle} />
        </aside>
      </div>
    </>
  );
}

function VideoPage({ people, health, onError, tools }) {
  const [job, setJob] = useState(null),
    [uploading, setUploading] = useState(false),
    [wantPlay, setWantPlay] = useState(false),
    [waiting, setWaiting] = useState(false),
    [current, setCurrent] = useState(0),
    [frame, setFrame] = useState(null),
    [events, setEvents] = useState([]),
    [filename, setFilename] = useState("Chưa chọn video");
  const input = useRef(),
    video = useRef(),
    jobRef = useRef(null),
    ownedVideo = useRef(null),
    cache = useRef([]),
    intent = useRef(false),
    epoch = useRef(0),
    currentRef = useRef(0),
    loading = useRef(false),
    alive = useRef(true);
  const pendingReviewTime = useRef(null);
  const reviewFaces = useRef(
    new URL(window.location.href).searchParams.get("review_faces") === "1",
  );
  useEffect(() => {
    const url = new URL(window.location.href),
      id = url.searchParams.get("video_job");
    if (!id || !/^[a-f0-9]{32}$/.test(id)) return;
    const requestedTime = url.searchParams.has("at")
      ? Number(url.searchParams.get("at"))
      : null;
    url.searchParams.delete("video_job");
    url.searchParams.delete("at");
    url.searchParams.delete("review_faces");
    window.history.replaceState(null, "", url.pathname + url.search + url.hash);
    let closed = false;
    api(`/videos/${id}`)
      .then((result) => {
        if (closed) return;
        jobRef.current = result;
        pendingReviewTime.current =
          Number.isFinite(requestedTime) && requestedTime !== null
            ? Math.max(
                0,
                Math.min(requestedTime, result.duration - 1 / result.fps),
              )
            : null;
        setJob(result);
        setFilename(result.filename || "Video đang đối chứng");
      })
      .catch((e) => {
        if (!closed) onError(e.message);
      });
    return () => {
      closed = true;
    };
  }, []);
  useEffect(() => {
    jobRef.current = job;
  }, [job]);
  useEffect(
    () => () => {
      alive.current = false;
      const j = jobRef.current;
      if (j && ownedVideo.current === j.id)
        api(`/videos/${j.id}`, { method: "DELETE" }).catch(() => {});
    },
    [],
  );
  function applyStatus(status) {
    if (status.id !== jobRef.current?.id) return;
    if ((status.generation ?? 0) < (jobRef.current.generation ?? 0)) return;
    if (status.generation !== jobRef.current.generation) {
      cache.current = [];
      setFrame(null);
    }
    jobRef.current = status;
    setJob(status);
    if (status.events) setEvents((e) => mergeEvents(e, status.events));
    if (status.state === "error") {
      intent.current = false;
      setWantPlay(false);
      video.current?.pause();
      onError(status.error);
    }
  }
  useEffect(() => {
    if (!job?.id) return;
    let closed = false,
      client,
      timer;
    function connect() {
      if (closed) return;
      client = socket("/ws/events");
      client.onmessage = (e) => {
        if (closed) return;
        const message = JSON.parse(e.data);
        if (
          message.session_id === jobRef.current?.id &&
          message.status &&
          (message.generation ?? message.status.generation ?? 0) >=
            (jobRef.current?.generation ?? 0)
        ) {
          applyStatus({ ...message.status, id: message.session_id });
          if (message.events)
            setEvents((old) => mergeEvents(old, message.events));
        }
        if (message.type === "snapshot") {
          const s = message.jobs.find((j) => j.id === jobRef.current?.id);
          if (s) applyStatus(s);
        }
      };
      client.onclose = () => {
        if (!closed) timer = setTimeout(connect, 1200);
      };
    }
    connect();
    return () => {
      closed = true;
      clearTimeout(timer);
      client?.close();
    };
  }, [job?.id]);
  useEffect(() => {
    if (!job?.id) return;
    let closed = false;
    const poll = async () => {
      const j = jobRef.current;
      if (!j || loading.current) return;
      loading.current = true;
      const version = epoch.current,
        at = currentRef.current;
      try {
        const [status, results] = await Promise.all([
          api(`/videos/${j.id}`),
          api(
            `/videos/${j.id}/frames?start=${Math.max(0, at - 0.5)}&end=${at + 3}`,
          ),
        ]);
        if (closed || version !== epoch.current || jobRef.current?.id !== j.id)
          return;
        applyStatus(status);
        cache.current = results.filter(
          (r) => r.generation === status.generation,
        );
      } catch (e) {
        if (
          !closed &&
          version === epoch.current &&
          jobRef.current?.id === j.id
        ) {
          if (e.status === 404) {
            epoch.current++;
            cache.current = [];
            intent.current = false;
            video.current?.pause();
            jobRef.current = null;
            setJob(null);
            setFrame(null);
            setEvents([]);
            setCurrent(0);
            currentRef.current = 0;
            setWantPlay(false);
            setWaiting(false);
            setFilename("Chưa chọn video");
            onError(
              "Phiên video đã đóng. Hãy mở lại video để tiếp tục; số đo đã lưu vẫn có trong lịch sử.",
            );
          } else onError(e.message);
        }
      } finally {
        loading.current = false;
      }
    };
    poll();
    const timer = setInterval(poll, 350);
    return () => {
      closed = true;
      clearInterval(timer);
    };
  }, [job?.id]);
  useEffect(() => {
    if (!job?.id) return;
    let raf,
      closed = false;
    const tick = () => {
      if (closed) return;
      const v = video.current,
        j = jobRef.current;
      if (v && j) {
        const at = v.currentTime;
        currentRef.current = at;
        setCurrent(at);
        const candidateFrame = frameAt(cache.current, at, j.fps);
        const found =
          candidateFrame?.generation === j.generation ? candidateFrame : null;
        setFrame(found);
        const buffered =
          found &&
          (j.state === "complete" ||
            j.analyzed_until - at >= Math.max(1 / j.fps, 0.08));
        if (intent.current && !v.ended) {
          if (buffered) {
            const rate =
              j.state === "complete"
                ? 1
                : Math.max(0.0625, Math.min(1, (j.ai_fps / j.fps) * 0.88));
            if (Math.abs(v.playbackRate - rate) > 0.02) v.playbackRate = rate;
            if (v.paused)
              v.play().catch((e) => {
                if (e.name !== "AbortError") {
                  intent.current = false;
                  setWantPlay(false);
                  onError("Không phát được video. Hãy dùng MP4 H.264 / WebM.");
                }
              });
            setWaiting(false);
          } else {
            v.pause();
            setWaiting(true);
          }
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      closed = true;
      cancelAnimationFrame(raf);
    };
  }, [job?.id]);
  async function upload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    intent.current = false;
    setWantPlay(false);
    epoch.current++;
    cache.current = [];
    setFrame(null);
    setEvents([]);
    try {
      const body = new FormData();
      body.append("file", file);
      const result = await api("/videos", { method: "POST", body });
      ownedVideo.current = result.id;
      jobRef.current = result;
      setJob(result);
      setFilename(file.name);
      setCurrent(0);
      currentRef.current = 0;
    } catch (error) {
      onError(error.message);
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }
  async function toggle() {
    if (!job) return;
    const play = !intent.current;
    if (play && video.current?.ended) {
      epoch.current++;
      cache.current = [];
      setFrame(null);
      video.current.currentTime = 0;
      currentRef.current = 0;
      setCurrent(0);
    }
    intent.current = play;
    setWantPlay(play);
    if (!play) {
      video.current?.pause();
      setWaiting(false);
    }
    try {
      applyStatus(
        await api(
          `/videos/${job.id}/control`,
          json("POST", {
            action: play ? "play" : "pause",
            timestamp: currentRef.current,
          }),
        ),
      );
    } catch (error) {
      intent.current = false;
      setWantPlay(false);
      onError(error.message);
    }
  }
  async function pauseForGroundTruth() {
    intent.current = false;
    setWantPlay(false);
    video.current?.pause();
    setWaiting(false);
    applyStatus(
      await api(
        `/videos/${job.id}/control`,
        json("POST", {
          action: "pause",
          timestamp: currentRef.current,
        }),
      ),
    );
  }
  async function seek(value) {
    if (!video.current || !job) return;
    epoch.current++;
    cache.current = [];
    setFrame(null);
    video.current.pause();
    video.current.currentTime = value;
    currentRef.current = value;
    setCurrent(value);
    setWaiting(intent.current);
    try {
      applyStatus(
        await api(
          `/videos/${job.id}/control`,
          json("POST", {
            action: intent.current ? "seek" : "pause",
            timestamp: value,
          }),
        ),
      );
    } catch (e) {
      onError(e.message);
    }
  }
  const shown = visibleEvents(events, current);
  const foundCount = new Set(shown.map((e) => e.person_id)).size;
  return (
    <>
      <input
        ref={input}
        type="file"
        accept="video/mp4,video/quicktime,video/webm,.m4v"
        hidden
        onChange={upload}
      />
      <div className="toolbar">
        <div className="toolbar-group">
          <label>Video đang chọn</label>
          <div className="filename" title={filename}>
            {filename}
          </div>
          <span className="soft-chip">
            <Check size={12} /> Tự động định hướng
          </span>
        </div>
        <div className="toolbar-group">
          <button disabled={uploading} onClick={() => input.current.click()}>
            <RotateCcw size={14} />
            {uploading ? "Đang tải…" : job ? "Đổi video" : "Chọn video"}
          </button>
          <button
            className="primary"
            disabled={!job || !health?.ready || job.state === "error"}
            onClick={toggle}
          >
            {wantPlay ? (
              <Pause size={14} />
            ) : (
              <Play size={14} fill="currentColor" />
            )}
            {wantPlay ? "Tạm dừng" : "Phát & phân tích"}
          </button>
        </div>
      </div>
      {tools && (
        <VideoEvaluationHistory refreshKey={`${job?.id}:${job?.state}`} />
      )}
      <div className="workspace-grid">
        <div>
          <Card
            title="Video và kết quả AI"
            badge={
              <span className="muted">
                {job
                  ? `${job.width} × ${job.height} · Video ${job.height > job.width ? "dọc" : "ngang"} · Đúng tỷ lệ`
                  : "Giữ nguyên tỷ lệ video gốc"}
              </span>
            }
          >
            <div className="video-stage upload-stage">
              <video
                ref={video}
                src={job?.media_url}
                playsInline
                preload="auto"
                onLoadedMetadata={() => {
                  const at = pendingReviewTime.current ?? 0;
                  pendingReviewTime.current = null;
                  epoch.current++;
                  cache.current = [];
                  video.current.currentTime = at;
                  currentRef.current = at;
                  setCurrent(at);
                  setFrame(null);
                }}
                onEnded={() => {
                  intent.current = false;
                  setWantPlay(false);
                  setWaiting(false);
                }}
                onError={() =>
                  job &&
                  onError(
                    "Trình duyệt không hỗ trợ codec này. Hãy dùng MP4 H.264 hoặc WebM.",
                  )
                }
              />
              {job && <Overlay frame={frame} review={tools} />}
              <span className="stage-note top">
                Tự động nhận diện video dọc · Giữ nguyên tỷ lệ
              </span>
              {!job && (
                <button
                  className="stage-empty upload-empty"
                  onClick={() => input.current.click()}
                  disabled={uploading}
                >
                  <Upload size={40} />
                  <strong>
                    {uploading
                      ? "Đang kiểm tra video…"
                      : "Chọn video để bắt đầu"}
                  </strong>
                  <span>Video chỉ được xử lý trên máy tính này</span>
                </button>
              )}
              {waiting && (
                <div className="buffer-indicator">
                  <LoaderCircle size={16} className="spin" /> AI đang xử lý tới
                  vị trí phát…
                </div>
              )}
            </div>
            <div className="player-controls">
              <div className="timeline-row">
                <button
                  className="icon-button"
                  disabled={!job}
                  onClick={toggle}
                  aria-label={wantPlay ? "Tạm dừng" : "Phát"}
                >
                  {wantPlay ? (
                    <Pause size={18} fill="currentColor" />
                  ) : (
                    <Play size={18} fill="currentColor" />
                  )}
                </button>
                <input
                  aria-label="Vị trí phát video"
                  type="range"
                  min="0"
                  max={job?.duration || 1}
                  step="0.01"
                  value={current}
                  disabled={!job}
                  onChange={(e) => seek(Number(e.target.value))}
                  style={{
                    "--progress": `${(current / (job?.duration || 1)) * 100}%`,
                  }}
                />
                <span>
                  {time(current)} / {time(job?.duration)}
                </span>
                <button
                  className="icon-button"
                  aria-label="Toàn màn hình"
                  onClick={() =>
                    video.current?.parentElement.requestFullscreen()
                  }
                  disabled={!job}
                >
                  <Maximize size={15} />
                </button>
              </div>
              <div className="player-tags">
                <span>
                  Phân tích:{" "}
                  {job
                    ? {
                        ready: "chưa bắt đầu",
                        analyzing: "đang chạy",
                        paused: "tạm dừng",
                        complete: "hoàn tất",
                        error: "có lỗi",
                      }[job.state]
                    : "chờ video"}
                </span>
                {tools && (
                  <span>
                    Đối chứng: {current.toFixed(3)} s · Frame{" "}
                    {frame?.frame_id ?? "đang tải"}
                  </span>
                )}
              </div>
            </div>
          </Card>
          <Stats
            items={[
              [
                "Tiến độ phân tích",
                job ? `${Math.round(job.progress * 100)}%` : null,
                job ? `${job.processed} / ${job.total} frame` : "Chờ video",
              ],
              [
                "Tốc độ xử lý",
                job?.ai_fps ? job.ai_fps.toFixed(1) : null,
                "FPS thực",
              ],
              [
                "Số người tìm thấy",
                job ? foundCount : null,
                "trong đoạn đã phát",
              ],
            ]}
          />
          {tools && job && (
            <VideoGroundTruth
              key={job.id}
              job={job}
              current={current}
              frame={frame}
              onPause={pauseForGroundTruth}
              reviewFaces={reviewFaces.current}
              onError={onError}
            />
          )}
        </div>
        <aside className="right-column">
          <Detections
            events={shown}
            video
            calibrated={health?.settings?.calibrated}
          />
          <Card
            title="Trạng thái tác vụ"
            badge={
              <Badge>
                {waiting
                  ? "Chờ AI"
                  : wantPlay
                    ? "Đang phân tích"
                    : job?.state === "complete"
                      ? "Hoàn tất"
                      : job
                        ? "Sẵn sàng"
                        : "Chưa có video"}
              </Badge>
            }
            footer="Khi AI chậm hơn phát video, tốc độ phát tự điều chỉnh. Tua đến đoạn chưa phân tích sẽ chờ AI, không bỏ qua frame."
          >
            <div className="status-row">
              <span>Phát video đồng thời với nhận diện</span>
              <small>{wantPlay ? "Bật" : "Tạm dừng"}</small>
            </div>
            <div className="status-row">
              <span>Play / Pause / Seek</span>
              <small>Đồng bộ</small>
            </div>
            <div className="status-row">
              <span>Chống sự kiện trùng lặp</span>
              <small>Theo lượt xuất hiện</small>
            </div>
            <div className="status-row">
              <span>Mục tiêu đang chọn</span>
              <small>{people.filter((p) => p.selected).length} người</small>
            </div>
          </Card>
        </aside>
      </div>
    </>
  );
}

function People({ people, reload, onToggle, onError }) {
  const [search, setSearch] = useState(""),
    [selected, setSelected] = useState(null),
    [name, setName] = useState(""),
    [modal, setModal] = useState(false),
    [newName, setNewName] = useState(""),
    [files, setFiles] = useState([]),
    [busy, setBusy] = useState(false),
    [view, setView] = useState(null),
    [confirm, setConfirm] = useState(null);
  const addInput = useRef();
  const person = people.find((p) => p.id === selected) || people[0];
  useEffect(() => {
    setName(person?.name || "");
  }, [person?.id, person?.name]);
  async function action(fn) {
    setBusy(true);
    try {
      await fn();
      await reload();
    } catch (e) {
      onError(e.message);
    } finally {
      setBusy(false);
    }
  }
  async function create(e) {
    e.preventDefault();
    if (!newName.trim() || !files.length) return;
    await action(async () => {
      const body = new FormData();
      body.append("name", newName.trim());
      files.forEach((f) => body.append("files", f));
      const p = await api("/persons", { method: "POST", body });
      setSelected(p.id);
      setModal(false);
      setNewName("");
      setFiles([]);
    });
  }
  async function add(e) {
    const incoming = [...e.target.files];
    if (!incoming.length || !person) return;
    await action(async () => {
      const body = new FormData();
      incoming.forEach((f) => body.append("files", f));
      await api(`/persons/${person.id}/images`, { method: "POST", body });
    });
    e.target.value = "";
  }
  return (
    <>
      <div className="people-toolbar">
        <div className="search-box">
          <Search size={15} />
          <input
            placeholder="Tìm kiếm theo tên người…"
            aria-label="Tìm kiếm người"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <span className="soft-chip">
          <Check size={14} />
          {people.filter((p) => p.selected).length} người đang được chọn
        </span>
        <button className="primary" onClick={() => setModal(true)}>
          <Plus size={14} /> Đăng ký người mới
        </button>
      </div>
      <div className="people-layout">
        <div>
          <div className="person-grid">
            {people
              .filter((p) =>
                p.name
                  .toLocaleLowerCase("vi")
                  .includes(search.toLocaleLowerCase("vi")),
              )
              .map((p, i) => (
                <article
                  className={`person-card ${person?.id === p.id ? "focused" : ""}`}
                  key={p.id}
                >
                  <div
                    className={`person-cover cover-${i % 3}`}
                    onClick={() => setSelected(p.id)}
                  >
                    <img src={p.images[0]?.url} alt={p.name} />
                    {p.selected && (
                      <span className="selected-label">
                        <Check size={11} /> Đang tìm
                      </span>
                    )}
                  </div>
                  <div className="person-body">
                    <h3>{p.name}</h3>
                    <p>
                      {p.images.length} ảnh đăng ký ·{" "}
                      {p.images.some((i) => i.models?.length)
                        ? "Đã trích xuất Hybrid"
                        : "Chờ trích xuất Hybrid"}
                    </p>
                    <div>
                      <button
                        className={`selection-pill ${p.selected ? "chosen" : ""}`}
                        onClick={() => onToggle(p)}
                      >
                        <Check size={11} />
                        {p.selected ? "Đã chọn" : "Chọn tìm"}
                      </button>
                      <button
                        className="small-button"
                        onClick={() => setSelected(p.id)}
                      >
                        Quản lý <ChevronRight size={11} />
                      </button>
                    </div>
                  </div>
                </article>
              ))}
          </div>
          {!people.length && (
            <div className="card people-empty">
              <Empty icon={Users} title="Chưa có người đăng ký">
                Bắt đầu với một ảnh khuôn mặt rõ ràng.
              </Empty>
              <button onClick={() => setModal(true)}>
                <Plus size={14} /> Đăng ký người mới
              </button>
            </div>
          )}
          {people.length > 0 &&
            !people.some((p) =>
              p.name
                .toLocaleLowerCase("vi")
                .includes(search.toLocaleLowerCase("vi")),
            ) && <Empty icon={Search} title="Không tìm thấy tên phù hợp" />}
          <Card
            className="rules"
            title="Quy tắc đăng ký và tìm kiếm"
            badge={<Badge>Một ảnh hợp lệ là đủ đăng ký</Badge>}
            footer="Ảnh đăng ký được lưu trên máy của bạn."
          >
            {[
              "Đăng ký từ tối thiểu 1 ảnh khuôn mặt hợp lệ",
              "Không đặt giới hạn cứng số ảnh bổ sung",
              "Chọn nhiều người làm mục tiêu cùng lúc",
            ].map((text) => (
              <div className="status-row" key={text}>
                <Check size={13} />
                <span>{text}</span>
              </div>
            ))}
          </Card>
        </div>
        <Card
          className="profile"
          title={`Quản lý hồ sơ${person ? " · " + person.name : ""}`}
          badge={
            person && <Badge green>{person.images.length} ảnh hợp lệ</Badge>
          }
          footer="Chọn ảnh có đúng một khuôn mặt rõ nét. Một ảnh chính diện là đủ để đăng ký."
        >
          {person ? (
            <>
              <div className="profile-name">
                <label htmlFor="person-name">Tên người đăng ký</label>
                <input
                  id="person-name"
                  value={name}
                  disabled={busy}
                  onChange={(e) => setName(e.target.value)}
                  maxLength={120}
                />
              </div>
              <div className="photos-title">
                <h3>Ảnh khuôn mặt đã đăng ký</h3>
                <span>Thêm / xem / xóa từng ảnh</span>
              </div>
              <div className="photo-list">
                {person.images.map((img, i) => (
                  <div className="photo-tile" key={img.id}>
                    <button
                      className="photo-open"
                      onClick={() => setView(img.url)}
                    >
                      <img
                        src={img.url}
                        alt={`Ảnh ${i + 1} của ${person.name}`}
                      />
                      <span>Ảnh {String(i + 1).padStart(2, "0")} · hợp lệ</span>
                    </button>
                    <button
                      className="delete-photo"
                      title="Xóa ảnh"
                      disabled={busy}
                      onClick={() =>
                        setConfirm({
                          text: "Xóa ảnh đăng ký này?",
                          run: () =>
                            api(`/persons/${person.id}/images/${img.id}`, {
                              method: "DELETE",
                            }),
                        })
                      }
                    >
                      <X size={12} />
                    </button>
                  </div>
                ))}
                <button
                  className="add-photo"
                  disabled={busy}
                  onClick={() => addInput.current.click()}
                >
                  <Plus size={18} />
                  <span>Thêm ảnh</span>
                </button>
                <input
                  ref={addInput}
                  type="file"
                  multiple
                  accept="image/jpeg,image/png,image/webp"
                  hidden
                  onChange={add}
                />
              </div>
              <div className="profile-actions">
                <label>Trạng thái embedding</label>
                <Badge green={person.images.some((i) => i.models?.length)}>
                  Hybrid:{" "}
                  {[
                    ...new Set(
                      person.images
                        .flatMap((i) => i.models || [])
                        .map((m) => m.split(":")[0]),
                    ),
                  ].join(" / ") || "chưa trích xuất"}
                </Badge>
                <div>
                  <button
                    onClick={() =>
                      document.getElementById("person-name").focus()
                    }
                    disabled={busy}
                  >
                    Sửa hồ sơ
                  </button>
                  <button
                    className="primary"
                    disabled={busy || !name.trim()}
                    onClick={() =>
                      action(() =>
                        api(
                          `/persons/${person.id}`,
                          json("PATCH", { name: name.trim() }),
                        ),
                      )
                    }
                  >
                    {busy ? <LoaderCircle size={14} className="spin" /> : null}
                    Lưu thay đổi
                  </button>
                </div>
                <button
                  className="danger-link"
                  disabled={busy}
                  onClick={() =>
                    setConfirm({
                      text: `Xóa hồ sơ “${person.name}” và toàn bộ ảnh đăng ký?`,
                      run: () =>
                        api(`/persons/${person.id}`, { method: "DELETE" }),
                    })
                  }
                >
                  <Trash2 size={13} />
                  Xóa người này
                </button>
              </div>
            </>
          ) : (
            <Empty icon={Users} title="Chọn một hồ sơ">
              Ảnh đăng ký và thông tin người sẽ hiển thị tại đây.
            </Empty>
          )}
        </Card>
      </div>
      {modal && (
        <div className="modal-backdrop">
          <form className="modal" onSubmit={create}>
            <div className="modal-head">
              <h2>Đăng ký người mới</h2>
              <button
                type="button"
                className="icon-button"
                disabled={busy}
                onClick={() => setModal(false)}
                aria-label="Đóng"
              >
                <X size={19} />
              </button>
            </div>
            <label>
              Tên người cần tìm
              <input
                autoFocus
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                maxLength={120}
                required
                placeholder="Nhập họ tên"
              />
            </label>
            <label className="upload-field">
              <ImagePlus size={28} />
              <strong>Chọn ảnh khuôn mặt</strong>
              <span>Một ảnh hợp lệ là đủ. Có thể thêm nhiều ảnh.</span>
              <input
                type="file"
                multiple
                accept="image/jpeg,image/png,image/webp"
                required
                onChange={(e) => setFiles([...e.target.files])}
              />
            </label>
            <p className="muted">
              {files.length
                ? `${files.length} ảnh đã chọn. AI sẽ kiểm tra từng ảnh trước khi lưu.`
                : "Mỗi ảnh cần đúng một khuôn mặt rõ ràng."}
            </p>
            <button className="primary full" disabled={busy || !files.length}>
              {busy ? (
                <>
                  <LoaderCircle size={16} className="spin" /> Đang kiểm tra &
                  tạo embedding…
                </>
              ) : (
                <>
                  <Plus size={15} />
                  Đăng ký & nhận diện ngay
                </>
              )}
            </button>
          </form>
        </div>
      )}
      {view && (
        <div className="modal-backdrop" onClick={() => setView(null)}>
          <div className="image-dialog">
            <button
              className="icon-button"
              onClick={() => setView(null)}
              aria-label="Đóng ảnh"
            >
              <X size={22} />
            </button>
            <img src={view} alt="Ảnh đăng ký đầy đủ" />
          </div>
        </div>
      )}
      {confirm && (
        <div className="modal-backdrop">
          <div className="modal">
            <h2>Xác nhận xóa</h2>
            <p>{confirm.text}</p>
            <div className="dialog-actions">
              <button onClick={() => setConfirm(null)} disabled={busy}>
                Hủy
              </button>
              <button
                className="danger"
                disabled={busy}
                onClick={() =>
                  action(async () => {
                    await confirm.run();
                    setConfirm(null);
                  })
                }
              >
                Xóa
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function App() {
  const [page, setPage] = useState(() =>
      new URL(window.location.href).searchParams.has("video_job")
        ? "video"
        : "live",
    ),
    [tools, setTools] = useState(() => {
      const query = new URL(window.location.href).searchParams;
      return (
        query.get("tools") === "1" ||
        query.get("diagnostic") === "camo" ||
        query.get("review_faces") === "1"
      );
    }),
    [people, setPeople] = useState([]),
    [health, setHealth] = useState(null),
    [error, setError] = useState(null);
  const timeout = useRef();
  const showError = (message) => {
    setError(message);
    clearTimeout(timeout.current);
    timeout.current = setTimeout(() => setError(null), 12000);
  };
  async function reload() {
    setPeople(await api("/persons"));
  }
  useEffect(() => {
    reload().catch((e) => showError(e.message));
    let alive = true;
    const check = () =>
      api("/health")
        .then((h) => alive && setHealth(h))
        .catch(
          () => alive && setHealth({ error: "Không kết nối được backend." }),
        );
    check();
    const timer = setInterval(check, 3000);
    return () => {
      alive = false;
      clearInterval(timer);
      clearTimeout(timeout.current);
    };
  }, []);
  async function toggle(p) {
    try {
      await api(`/persons/${p.id}`, json("PATCH", { selected: !p.selected }));
      await reload();
    } catch (e) {
      showError(e.message);
    }
  }
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-icon">
            <Search size={19} strokeWidth={3} />
          </span>
          <div>
            <strong>FaceWatch AI</strong>
            <span>
              Local Face Search
              <br />
              System
            </span>
          </div>
        </div>
        <div className="nav-caption">KHÔNG GIAN LÀM VIỆC</div>
        <nav>
          {[
            ["live", Video],
            ["video", SquarePlay],
            ["people", Users],
          ].map(([key, Icon]) => (
            <button
              key={key}
              className={page === key ? "active" : ""}
              onClick={() => {
                setError(null);
                setPage(key);
              }}
            >
              <Icon size={18} />
              <span>{titles[key]}</span>
            </button>
          ))}
        </nav>
        <div className="local-status">
          <strong>
            <i />
            Xử lý cục bộ
          </strong>
          <span>
            Ảnh và video được xử lý trên máy
            <br />
            {health?.ready
              ? `AI inference trên ${health.device === "cuda" ? "NVIDIA GPU" : "CPU"}`
              : "AI đang khởi động"}
          </span>
        </div>
      </aside>
      <div className="main-shell">
        <header>
          <div>
            Hệ thống tìm kiếm người <span>/ {titles[page]}</span>
          </div>
          <div className="header-right">
            <span>Cục bộ · localhost</span>
            <span>
              {health?.ready
                ? `${health.device === "cuda" ? "GPU" : "CPU"} · AI sẵn sàng`
                : "Đang nạp model"}
            </span>
            <b>AI</b>
          </div>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <h1>
                {page === "people" ? "Danh sách người cần tìm" : titles[page]}
              </h1>
              <p>
                {page === "live"
                  ? "Giám sát khu vực và nhận thông báo ngay khi phát hiện người cần tìm."
                  : page === "video"
                    ? "Phát video và phân tích AI đồng thời, đồng bộ kết quả theo từng thời điểm."
                    : "Quản lý ảnh đăng ký và lựa chọn nhiều người cần tìm kiếm."}
              </p>
            </div>
            <div className="page-indicator">
              <i />
              {page === "live"
                ? "CAMERA TRỰC TIẾP"
                : page === "video"
                  ? "VIDEO UPLOAD"
                  : "QUẢN LÝ HỒ SƠ"}
            </div>
          </div>
          {tools && (
            <div className="tools-notice" role="status">
              <span>
                Chế độ chẩn đoán · Công cụ kiểm thử và số đo chi tiết đang mở.
              </span>
              <button
                onClick={() => {
                  setTools(false);
                  const url = new URL(window.location.href);
                  url.searchParams.delete("tools");
                  window.history.replaceState(
                    null,
                    "",
                    url.pathname + url.search + url.hash,
                  );
                }}
              >
                Đóng chẩn đoán
              </button>
            </div>
          )}
          {health?.error && (
            <div className="alert">
              <AlertCircle size={17} />
              <span>AI chưa sẵn sàng: {health.error}</span>
            </div>
          )}
          {error && (
            <div className="alert" role="alert">
              <AlertCircle size={17} />
              <span>{error}</span>
              <button
                className="icon-button"
                onClick={() => setError(null)}
                aria-label="Đóng thông báo"
              >
                <X size={15} />
              </button>
            </div>
          )}
          {page === "live" ? (
            <Live
              people={people}
              onToggle={toggle}
              health={health}
              onError={showError}
              tools={tools}
            />
          ) : page === "video" ? (
            <VideoPage
              people={people}
              health={health}
              onError={showError}
              tools={tools}
            />
          ) : (
            <People
              people={people}
              reload={reload}
              onToggle={toggle}
              onError={showError}
            />
          )}
        </main>
        <footer>
          <span>FaceWatch AI · Dữ liệu được lưu và xử lý cục bộ</span>
          <button
            className="tools-toggle"
            aria-expanded={tools}
            onClick={() => {
              setTools((value) => !value);
              const url = new URL(window.location.href);
              url.searchParams.delete("tools");
              window.history.replaceState(
                null,
                "",
                url.pathname + url.search + url.hash,
              );
            }}
          >
            {tools ? "Ẩn công cụ chẩn đoán" : "Chẩn đoán"}
          </button>
        </footer>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
