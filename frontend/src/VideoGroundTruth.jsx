import React, { useEffect, useRef, useState } from "react";
import { api, json } from "./api";

export function VideoEvaluationHistory({ refreshKey }) {
  const [runs, setRuns] = useState([]),
    [error, setError] = useState("");
  useEffect(() => {
    let closed = false;
    api("/video-evaluations")
      .then((data) => {
        if (!closed) {
          setRuns(data);
          setError("");
        }
      })
      .catch((e) => {
        if (!closed) setError(e.message);
      });
    return () => {
      closed = true;
    };
  }, [refreshKey]);
  if (!runs.length && !error) return null;
  return (
    <details
      className="card video-ground-truth"
      aria-label="Số đo video đã lưu"
    >
      <summary>Số đo video đã lưu · {runs.length} lượt</summary>
      {error && <p>{error}</p>}
      {runs.map((r) => (
        <details key={r.run_id}>
          <summary>
            {r.filename} · {r.frames} frame · lượt {r.run_id.slice(-6)}
          </summary>
          <p>
            {r.annotations.length} khoảng có nhãn · {r.alerts.false_alerts} cảnh
            báo trong khoảng vắng mặt · {r.alerts.unverified} cảnh báo thiếu
            nhãn.
          </p>
          {r.intervals.map((a, i) => (
            <p key={i}>
              {r.enrollment.find((p) => p.person_id === a.person_id)?.name ||
                "Hồ sơ cũ"}
              : {a.start.toFixed(1)}–{a.end.toFixed(1)} s ·{" "}
              {a.confirmation_seen
                ? "có Confirmed trong khoảng"
                : a.fully_analyzed
                  ? "chưa có Confirmed"
                  : "chưa phân tích đủ"}
            </p>
          ))}
          <p>
            {Object.entries(r.models)
              .map(([key, m]) => `${key}: ${m.samples} điểm hợp lệ`)
              .join(" · ")}
          </p>
          <small>{r.caution} Mở lại video gốc để xem frame và sửa nhãn.</small>
        </details>
      ))}
    </details>
  );
}

function VideoFaceTruth({
  job,
  frame,
  onPause,
  onError,
  onSaved,
  requestReview,
}) {
  const [snapshot, setSnapshot] = useState(null),
    [labels, setLabels] = useState({});
  const [busy, setBusy] = useState(false),
    [message, setMessage] = useState("");
  const alive = useRef(true),
    section = useRef(null),
    autoLoaded = useRef(false);
  useEffect(() => {
    if (
      requestReview &&
      Number.isInteger(frame?.frame_id) &&
      !autoLoaded.current
    ) {
      autoLoaded.current = true;
      section.current.open = true;
      load();
    }
  }, [requestReview, frame?.frame_id]);
  useEffect(() => {
    if (requestReview && snapshot)
      section.current?.scrollIntoView({ block: "center" });
  }, [requestReview, snapshot]);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  async function load() {
    const frameId = frame?.frame_id;
    if (!Number.isInteger(frameId)) return;
    setBusy(true);
    setMessage("");
    try {
      await onPause();
      const data = await api(`/videos/${job.id}/face-ground-truth/${frameId}`);
      if (!alive.current) return;
      setSnapshot(data);
      setLabels(
        Object.fromEntries(
          data.labels.map((label) => [
            label.face_index,
            {
              value: label.kind === "outside" ? "outside" : label.person_id,
              different_source: label.different_source,
            },
          ]),
        ),
      );
    } catch (error) {
      if (alive.current) onError(error.message);
    } finally {
      if (alive.current) setBusy(false);
    }
  }
  async function save() {
    setBusy(true);
    try {
      const selected = Object.entries(labels)
        .filter(([, label]) => label.value)
        .map(([index, label]) => ({
          face_index: Number(index),
          kind: label.value === "outside" ? "outside" : "person",
          person_id: label.value === "outside" ? null : label.value,
          different_source: !!label.different_source,
        }));
      await api(
        `/videos/${job.id}/face-ground-truth/${snapshot.frame_id}`,
        json("PUT", { labels: selected }),
      );
      if (!alive.current) return;
      setMessage(`Đã lưu ${selected.length} nhãn mặt do bạn xác nhận.`);
      onSaved();
    } catch (error) {
      if (alive.current) onError(error.message);
    } finally {
      if (alive.current) setBusy(false);
    }
  }
  return (
    <details className="video-face-truth" ref={section}>
      <summary>Xác nhận danh tính từng khuôn mặt trong video</summary>
      <p>
        Chọn vị trí trên thanh phát, rồi mở các crop cùng frame để xác nhận. Mặt
        chưa rõ danh tính được giữ chưa gán nhãn.
      </p>
      <button
        disabled={busy || !Number.isInteger(frame?.frame_id)}
        onClick={load}
      >
        {busy ? "Đang xử lý…" : "Tạm dừng và chọn các mặt ở frame hiện tại"}
      </button>
      {snapshot && (
        <>
          <p>
            Frame {snapshot.frame_id} · {snapshot.timestamp.toFixed(3)} giây ·{" "}
            {snapshot.faces.length} mặt được detector phát hiện.
          </p>
          <div className="truth-face-grid">
            {snapshot.faces.map((face) => {
              const label = labels[face.face_index] || {
                value: "",
                different_source: false,
              };
              return (
                <div className="truth-face" key={face.face_index}>
                  <img
                    src={face.thumbnail}
                    alt={`Crop mặt ${face.face_index + 1} tại ${snapshot.timestamp.toFixed(3)} giây`}
                  />
                  <label>
                    Mặt {face.face_index + 1} · vị trí{" "}
                    {face.bbox.slice(0, 2).map(Math.round).join(", ")}
                    <select
                      value={label.value}
                      disabled={busy}
                      onChange={(e) =>
                        setLabels((previous) => ({
                          ...previous,
                          [face.face_index]: {
                            value: e.target.value,
                            different_source: false,
                          },
                        }))
                      }
                    >
                      <option value="">Chưa xác nhận</option>
                      {snapshot.enrollment.map((person) => (
                        <option key={person.person_id} value={person.person_id}>
                          {person.name}
                        </option>
                      ))}
                      <option value="outside">
                        Người ngoài danh sách đăng ký
                      </option>
                    </select>
                  </label>
                  {label.value && label.value !== "outside" && (
                    <label className="truth-checkbox">
                      <input
                        type="checkbox"
                        checked={label.different_source}
                        disabled={busy}
                        onChange={(e) =>
                          setLabels((previous) => ({
                            ...previous,
                            [face.face_index]: {
                              ...label,
                              different_source: e.target.checked,
                            },
                          }))
                        }
                      />
                      Video khác ảnh đăng ký của người này
                    </label>
                  )}
                </div>
              );
            })}
          </div>
          {!!snapshot.faces.length && (
            <button disabled={busy} onClick={save}>
              Lưu nhãn mặt do tôi xác nhận
            </button>
          )}
        </>
      )}
      {message && <p role="status">{message}</p>}
    </details>
  );
}

export default function VideoGroundTruth({
  job,
  current,
  frame,
  onPause,
  reviewFaces,
  onError,
}) {
  const [annotations, setAnnotations] = useState([]),
    [enrollment, setEnrollment] = useState([]);
  const [kind, setKind] = useState("present"),
    [person, setPerson] = useState("");
  const [start, setStart] = useState("0"),
    [end, setEnd] = useState("");
  const [note, setNote] = useState(""),
    [different, setDifferent] = useState(false);
  const [busy, setBusy] = useState(false),
    [loaded, setLoaded] = useState(false),
    [result, setResult] = useState(null);
  useEffect(() => {
    let closed = false;
    setLoaded(false);
    setResult(null);
    setAnnotations([]);
    setPerson("");
    setEnd("");
    api(`/videos/${job.id}/ground-truth`)
      .then((data) => {
        if (closed) return;
        setAnnotations(data.annotations);
        setEnrollment(data.enrollment);
        setResult(data.evaluation || null);
        setLoaded(true);
      })
      .catch((e) => {
        if (!closed) onError(e.message);
      });
    return () => {
      closed = true;
    };
  }, [job.id]);
  async function save(next) {
    setBusy(true);
    try {
      const data = await api(
        `/videos/${job.id}/ground-truth`,
        json("PUT", { annotations: next }),
      );
      setAnnotations(data.annotations);
      setResult(null);
    } catch (e) {
      onError(e.message);
    } finally {
      setBusy(false);
    }
  }
  async function add(e) {
    e.preventDefault();
    if (start === "" || end === "") return;
    await save([
      ...annotations,
      {
        start: Number(start),
        end: Number(end),
        kind,
        person_id: kind === "outside" ? null : person,
        note,
        different_source: different,
      },
    ]);
  }
  async function evaluate() {
    setBusy(true);
    try {
      setResult(await api(`/videos/${job.id}/evaluate`, json("POST", {})));
    } catch (e) {
      onError(e.message);
    } finally {
      setBusy(false);
    }
  }
  const name = (pid) =>
    enrollment.find((p) => p.person_id === pid)?.name || "Hồ sơ cũ";
  return (
    <section
      className="card video-ground-truth"
      aria-label="Ground truth video"
    >
      <h3>Đối chứng video · Nhãn do bạn xác nhận</h3>
      <p>
        Xem video rồi đánh dấu ai có mặt hoặc vắng mặt trong từng khoảng. Đoạn
        chưa gán nhãn luôn để chưa kiểm chứng. Bạn có thể gán các khoảng chồng
        nhau cho nhiều người.
      </p>
      <VideoFaceTruth
        job={job}
        frame={frame}
        onPause={onPause}
        onError={onError}
        onSaved={() => setResult(null)}
        requestReview={reviewFaces}
      />
      <form onSubmit={add}>
        <div className="truth-fields">
          <label>
            Nhãn xác nhận
            <select value={kind} onChange={(e) => setKind(e.target.value)}>
              <option value="present">Người này có mặt</option>
              <option value="absent">Người này không có mặt</option>
              <option value="outside">Không có ai trong watchlist</option>
            </select>
          </label>
          {kind !== "outside" && (
            <label>
              Người trong video
              <select
                value={person}
                onChange={(e) => setPerson(e.target.value)}
              >
                <option value="">Chọn danh tính thật</option>
                {enrollment.map((p) => (
                  <option key={p.person_id} value={p.person_id}>
                    {p.name} · {p.photo_count} ảnh
                    {p.selected ? "" : " · ngoài watchlist"}
                  </option>
                ))}
              </select>
            </label>
          )}
          <label>
            Từ giây
            <input
              type="number"
              min="0"
              max={job.duration}
              step="0.001"
              value={start}
              required
              onChange={(e) => setStart(e.target.value)}
            />
            <button type="button" onClick={() => setStart(current.toFixed(3))}>
              Lấy vị trí hiện tại làm đầu
            </button>
          </label>
          <label>
            Đến giây
            <input
              type="number"
              min="0"
              max={job.duration}
              step="0.001"
              value={end}
              required
              onChange={(e) => setEnd(e.target.value)}
            />
            <button
              type="button"
              onClick={() => setEnd(Math.min(current, job.duration).toFixed(3))}
            >
              Lấy vị trí hiện tại làm cuối
            </button>
          </label>
        </div>
        <label>
          Ghi chú tình huống / dấu hiệu bạn dùng để đối chiếu
          <input
            value={note}
            maxLength={500}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Ví dụ: bên trái, quay mặt, bị che…"
          />
        </label>
        {kind === "present" && (
          <label className="truth-checkbox">
            <input
              type="checkbox"
              checked={different}
              onChange={(e) => setDifferent(e.target.checked)}
            />
            Tôi xác nhận đây là video khác với ảnh dùng để đăng ký người này
          </label>
        )}
        <button
          disabled={!loaded || busy || (kind !== "outside" && !person)}
          type="submit"
        >
          Lưu khoảng đã xác nhận
        </button>
      </form>
      <div className="diagnostic-model-table">
        <table>
          <thead>
            <tr>
              <th>Khoảng (giây)</th>
              <th>Ground truth</th>
              <th>Ghi chú</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {annotations.map((a, i) => (
              <tr key={i}>
                <td>
                  {a.start.toFixed(3)} – {a.end.toFixed(3)}
                </td>
                <td>
                  {a.kind === "outside"
                    ? `Vắng ${a.scope.map(name).join(", ")}`
                    : `${name(a.person_id)}: ${a.kind === "present" ? "có mặt" : "vắng mặt"}`}
                  {a.different_source && " · khác ảnh đăng ký"}
                </td>
                <td>{a.note}</td>
                <td>
                  <button
                    disabled={busy}
                    aria-label={`Xóa khoảng ${i + 1}`}
                    onClick={() => save(annotations.filter((_, j) => i !== j))}
                  >
                    Xóa nhãn
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!annotations.length && (
        <p>Chưa có nhãn xác thực. Không tính độ chính xác từ dự đoán AI.</p>
      )}
      <button disabled={!loaded || busy || !job.processed} onClick={evaluate}>
        {busy ? "Đang xử lý…" : "Phân tích số đo và đối chiếu nhãn"}
      </button>
      {result && (
        <div aria-label="Kết quả đối chứng video" role="status">
          {result.enrollment_changed && (
            <p>
              Hồ sơ hoặc watchlist đã đổi trong khi phân tích. Không dùng lượt
              này để kết luận thử nghiệm một ảnh đăng ký.
            </p>
          )}
          <p>
            {result.frames} frame · {result.face_observations} lượt phát hiện
            mặt · {result.distinct_track_ids} track khác nhau · Độ trễ P50/P95{" "}
            {result.latency_p50_ms?.toFixed(0) ?? "—"}/
            {result.latency_p95_ms?.toFixed(0) ?? "—"} ms.
          </p>
          {result.face_ground_truth?.labeled_faces > 0 && (
            <div aria-label="Kết quả trên mặt đã gán nhãn">
              <p>
                {result.face_ground_truth.labeled_faces} mặt được bạn gán nhãn ·{" "}
                {result.face_ground_truth.confirmed_correct} Confirmed đúng ·{" "}
                {result.face_ground_truth.false_accepts} nhận nhầm ·{" "}
                {result.face_ground_truth.missed_known_faces} mặt mục tiêu chưa
                xác nhận · {result.face_ground_truth.rejected_outsiders} người
                ngoài được từ chối.
              </p>
              <p>
                {result.face_ground_truth.different_source_known_faces} mặt mục
                tiêu có xác nhận video khác ảnh đăng ký.
              </p>
              {Object.entries(result.face_ground_truth.models).map(
                ([key, measured]) => (
                  <p key={key}>
                    {key} · top-1 đúng {measured.known_top1_correct}/
                    {measured.known_samples} mặt có danh tính thật ·{" "}
                    {measured.outside_reference_passes}/
                    {measured.outside_samples} mặt ngoài vượt ngưỡng tham khảo.
                  </p>
                ),
              )}
              <small>Các số đo này áp dụng cho từng mặt đã gán nhãn.</small>
            </div>
          )}
          {result.stage_rejections && (
            <p>
              Chặn do chất lượng:{" "}
              {result.stage_rejections.quality_rejected_faces ?? 0} ·
              Crop/alignment:{" "}
              {result.stage_rejections.alignment_rejected_faces ?? 0} · Lỗi thực
              thi: {result.stage_rejections.inference_error_faces ?? 0}. Mặt gốc
              trung vị: {result.face_pixels_p50?.toFixed(1) ?? "—"} px.
            </p>
          )}
          <p>
            {result.alerts.false_alerts} cảnh báo trong khoảng đã xác nhận vắng
            mặt · {result.alerts.consistent_with_interval} cảnh báo trùng khoảng
            có mặt · {result.alerts.unverified} cảnh báo chưa có nhãn.
          </p>
          {result.alert_identity_verification && (
            <p>
              Đối chứng đúng khuôn mặt:{" "}
              {result.alert_identity_verification.correct} cảnh báo đúng ·{" "}
              {result.alert_identity_verification.wrong} sai ·{" "}
              {result.alert_identity_verification.unverified} chưa có nhãn mặt
              xác nhận.
            </p>
          )}
          <p>
            {result.intervals.filter((a) => a.no_confirmation).length} khoảng có
            mặt đã phân tích hết nhưng chưa có Confirmed.{" "}
            {result.state !== "complete" &&
              "Video chưa phân tích hết; số đo hiện tại chưa phải kết quả toàn video."}
          </p>
          {result.intervals.map((a, i) => (
            <p key={i}>
              {name(a.person_id)} {a.start.toFixed(2)}–{a.end.toFixed(2)} s:{" "}
              {a.confirmation_seen
                ? `Confirmed đầu sau ${a.time_to_confirmation_seconds.toFixed(2)} s (đối chiếu thời gian)`
                : a.fully_analyzed
                  ? "chưa xác nhận"
                  : "chưa phân tích đủ khoảng"}
            </p>
          ))}
          <div className="diagnostic-model-table">
            <table>
              <thead>
                <tr>
                  <th>Model thực sự chạy</th>
                  <th>Điểm hợp lệ</th>
                  <th>Có hiệu chuẩn</th>
                  <th>Điểm khi danh tính vắng mặt</th>
                  <th>Cosine trung vị</th>
                  <th>Lỗi</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(result.models).map(([key, m]) => (
                  <tr key={key}>
                    <td>{key}</td>
                    <td>{m.samples}</td>
                    <td>{m.calibrated}</td>
                    <td>{m.declared_absence_samples ?? "—"}</td>
                    <td>{m.cosine_p50?.toFixed(3) ?? "—"}</td>
                    <td>{m.errors}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <small>{result.caution}</small>
        </div>
      )}
    </section>
  );
}
