import React, { useEffect, useRef, useState } from "react";
import { api, json, socket } from "./api";
import { acquireCamera, cameraFor } from "./sync";
import { CameraView } from "./camera-view";
import { phaseAt } from "./diagnostic-protocol";

const conditions = [
  ["Chính diện", "NHÌN THẲNG — GIỮ TRỌN MẶT GIỮA KHUNG"],
  [
    "Quay trái/phải",
    "QUAY TRÁI/PHẢI CHẬM TỚI KHOẢNG 90° — GIỮ MỖI BÊN 2–3 GIÂY",
  ],
  [
    "Quay nghiêng/di chuyển",
    "QUAY NGHIÊNG TỚI 90° VÀ ĐI NGANG NHẸ THEO TỪNG GIAI ĐOẠN",
  ],
  ["Giơ tay/di chuyển", "GIƠ TAY — DI CHUYỂN CHẬM — THAY ĐỔI TƯ THẾ"],
  ["Che một phần mặt", "CHE MỘT PHẦN MẶT BẰNG TAY — GIỮ MẮT TRONG KHUNG"],
  ["Đeo khẩu trang", "ĐEO KHẨU TRANG — NHÌN VỀ CAMERA"],
];
const scenarios = [6]
  .map((d) => `Tổng hợp ${d} m`)
  .concat(
    [6].flatMap((distance) =>
      conditions.map(([condition]) => `${condition} ${distance} m`),
    ),
  );
const actionInstructions = Object.fromEntries(
  [1, 3, 6].flatMap((distance) =>
    conditions.map(([condition, instruction]) => [
      `${condition} ${distance} m`,
      [`ĐỨNG CÁCH CAMERA ${distance} M`, instruction],
    ]),
  ),
);
function instructionsFor(scenario) {
  if (scenario === "Tổng hợp 6 m")
    return ["TỔNG HỢP 6 M · 180 GIÂY", "NHÓM A: KHÔNG KHẨU TRANG · NHÓM B: KHẨU TRANG · NHÓM C: THÁO VÀ CHE MẶT", "Mỗi nhóm 60 giây, chuyển thao tác mỗi 20 giây. Chuẩn bị sẵn khẩu trang."];
  if (scenario.startsWith("Tổng hợp "))
    return [
      scenario.toUpperCase(),
      "180 GIÂY · 6 GIAI ĐOẠN — CHUẨN BỊ SẴN KHẨU TRANG",
      "Chuyển động tác có thời gian riêng; đồng hồ vẫn chạy và số đo được đánh dấu chuyển tiếp.",
    ];
  if (actionInstructions[scenario]) return actionInstructions[scenario];
  const condition = conditions.find(([name]) =>
    scenario.startsWith(`${name} `),
  );
  const distance = scenario.match(/ ([0-9]+(?:\.[0-9]+)?) m$/)?.[1];
  return condition && distance
    ? [`ĐỨNG CÁCH CAMERA ${distance} M`, condition[1]]
    : [scenario.toUpperCase(), "GIỮ MẶT TRONG KHUNG CAMERA"];
}
export function RecognitionNotice({ calibrated, compact = false, status }) {
  if (calibrated && status === "preliminary")
    return (
      <div
        className="recognition-notice compact"
        role="status"
        aria-label="Trạng thái xác nhận danh tính"
      >
        Đã bật xác nhận danh tính theo ngưỡng hiệu chuẩn bước đầu. Các tình
        huống mới vẫn cần kiểm chứng; thiếu bằng chứng sẽ giữ Chưa xác định.
      </div>
    );
  if (calibrated) return null;
  if (compact)
    return (
      <div
        className="recognition-notice compact"
        role="status"
        aria-label="Trạng thái xác nhận danh tính"
      >
        Chưa hoàn tất hiệu chuẩn xác nhận danh tính. Hiện AI phát hiện mặt và
        đối chiếu ảnh; chưa phát thông báo đã tìm thấy.
      </div>
    );
  return (
    <div
      className="recognition-notice"
      role="status"
      aria-label="Trạng thái xác nhận danh tính"
    >
      <strong>CHƯA HIỆU CHUẨN — CHƯA PHÁT CẢNH BÁO NHẬN DIỆN</strong>
      <span>
        AI đang phát hiện mặt và đối chiếu ảnh. Hiện chỉ có Candidate/Unknown;
        cần kiểm thử đúng người và người ngoài watchlist để hiệu chuẩn bước xác
        nhận.
      </span>
    </div>
  );
}
function CalibrationTools({ health, locked, onCollect }) {
  const [result, setResult] = useState(null),
    [busy, setBusy] = useState(false),
    [message, setMessage] = useState("");
  const profiles =
    result?.profiles || health?.settings?.calibration_profiles || [];
  async function calculate(apply) {
    setBusy(true);
    setMessage("");
    try {
      const data = await api(
        `/calibration/${apply ? "apply" : "preview"}`,
        json("POST", {}),
      );
      setResult(data.calibration || data);
      setMessage(
        apply
          ? "Đã áp dụng ngưỡng. Các lượt đã lưu sẽ được phân tích lại; chưa phải kiểm chứng bằng camera mới."
          : "Đã tính ngưỡng từ nhãn xác nhận. Chưa thay đổi ngưỡng đang dùng.",
      );
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <details
      className="card diagnostic-coverage"
      aria-label="Hiệu chuẩn danh tính"
    >
      <summary>
        Hiệu chuẩn và xác nhận danh tính ·{" "}
        {health?.settings?.calibrated ? "Đã bật bước đầu" : "Chưa áp dụng"}
      </summary>
      <p>
        Ngưỡng riêng theo model, kích thước mặt và alignment. Dùng các lượt Live
        đã xác nhận để tính ngưỡng, video có nhãn để đối chứng. Không cộng
        cosine thô hoặc lấy kết quả AI làm danh tính thật.
      </p>
      <div className="diagnostic-entry">
        <button
          disabled={busy || locked || !health?.ready}
          onClick={() => calculate(false)}
        >
          Tính ngưỡng từ dữ liệu đã xác nhận
        </button>
        <button disabled={busy || locked || !health?.ready} onClick={onCollect}>
          Bổ sung dữ liệu góc nghiêng bằng Camo
        </button>
        <button
          disabled={busy || locked || !health?.ready}
          onClick={() => calculate(true)}
        >
          Áp dụng ngưỡng
        </button>
      </div>
      {message && <p role="status">{message}</p>}
      <div className="diagnostic-model-table">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Dải mặt</th>
              <th>Ngưỡng</th>
              <th>Margin</th>
              <th>Đối chứng video</th>
            </tr>
          </thead>
          <tbody>
            {profiles.map((p) => {
              const c = p.config || p;
              return (
                <tr key={`${p.model}:${p.band}`}>
                  <td>{p.model.split(":")[0]}</td>
                  <td>
                    {{ normal: "≥90 px", small: "28–89 px", tiny: `${c.minimum_native_face_pixels ?? 14}–27 px` }[
                      p.band
                    ] || p.band}
                  </td>
                  <td>{c.threshold.toFixed(3)}</td>
                  <td>{c.margin.toFixed(3)}</td>
                  <td>
                    {c.validation_negative_observations
                      ? `${c.validation_false_passes}/${c.validation_negative_observations} so sánh âm vượt ngưỡng`
                      : "Chưa có so sánh âm cùng dải"}
                    {c.cross_quality_negative_observations > 0 && (
                      <small>
                        {" "}
                        · Stress alignment lệch ở dải kích thước khác:{" "}
                        {c.cross_quality_false_passes}/
                        {c.cross_quality_negative_observations} vượt ngưỡng
                      </small>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {result?.unavailable?.length > 0 && (
        <p>
          Chưa đủ bằng chứng:{" "}
          {result.unavailable
            .map((p) => `${p.model.split(":")[0]} / ${p.band}`)
            .join(" · ")}
          . Các nhánh này không cấp phiếu xác nhận.
        </p>
      )}
      <p>
        Hiện thích ứng theo kích thước mặt, độ nét và sai lệch alignment. Chưa
        có bộ hiệu chuẩn riêng đầy đủ cho khẩu trang và từng góc nghiêng; không
        tự gán trọng số theo tên model.
      </p>
      <small>
        Dữ liệu Live hiện chủ yếu của một người. Số frame không phải số người
        độc lập; đây chưa phải chứng nhận độ chính xác cho mọi lớp học.
      </small>
    </details>
  );
}

function DiagnosticReview({ record }) {
  const [review, setReview] = useState({}),
    [busy, setBusy] = useState(false),
    [message, setMessage] = useState("");
  useEffect(() => {
    setReview(record.analysis?.user_review || {});
    setMessage("");
  }, [record.id, JSON.stringify(record.analysis?.user_review)]);
  const measured = record.analysis?.reviewed_metrics;
  return (
    <details className="diagnostic-analysis-details">
      <summary>
        Xác nhận ground truth lượt {record.id.slice(-6)} · {record.scenario}
      </summary>
      <p>
        Chỉ đánh dấu điều bạn đã quan sát và chắc chắn. Nếu chưa rõ, để trống;
        hệ thống không tự lấy dự đoán AI làm nhãn thật.
      </p>
      {[
        [
          "only_target",
          "Suốt lượt chỉ có đúng người mục tiêu đã chọn trong khung",
        ],
        [
          "face_visible_throughout",
          "Khuôn mặt mục tiêu luôn ở trong khung suốt lượt đo",
        ],
        [
          "boxes_match_target",
          "Các bounding box hiển thị bám đúng mặt mục tiêu",
        ],
      ].map(([key, label]) => (
        <label className="truth-checkbox" key={key}>
          <input
            type="checkbox"
            checked={!!review[key]}
            onChange={(e) =>
              setReview((r) => ({ ...r, [key]: e.target.checked }))
            }
          />
          {label}
        </label>
      ))}
      {record.context?.protocol?.kind === "composite" &&
        record.context.protocol.stages.map((stage) => (
          <label className="truth-checkbox" key={stage.id}>
            <input
              type="checkbox"
              checked={review.stage_actions?.[stage.id] === true}
              onChange={(e) =>
                setReview((r) => ({
                  ...r,
                  stage_actions: {
                    ...r.stage_actions,
                    [stage.id]: e.target.checked,
                  },
                }))
              }
            />
            Tôi đã thực hiện: {stage.label} ({stage.start}–{stage.end} giây)
          </label>
        ))}
      <button
        disabled={busy}
        onClick={async () => {
          setBusy(true);
          try {
            await api(
              `/diagnostics/${record.id}/ground-truth`,
              json("PUT", {
                ...Object.fromEntries(
                  [
                    "only_target",
                    "face_visible_throughout",
                    "boxes_match_target",
                  ].map((k) => [k, !!review[k]]),
                ),
                stage_actions: review.stage_actions || {},
              }),
            );
            setMessage("Đã lưu xác nhận; số đo sẽ cập nhật tại đây.");
          } catch (e) {
            setMessage(e.message);
          } finally {
            setBusy(false);
          }
        }}
      >
        Lưu xác nhận của tôi
      </button>
      {message && <p role="status">{message}</p>}
      {measured && (
        <div>
          <p>
            Theo xác nhận của bạn: {measured.detected_face_frames}/
            {measured.expected_face_frames} frame đã phân tích có mặt đúng ·{" "}
            {measured.lost_track_episodes} lần mất dấu ·{" "}
            {measured.face_track_changes ?? measured.identity_switches} lần đổi
            track ·{" "}
            {measured.face_track_changes !== undefined
              ? measured.identity_switches
              : "chưa đo"}{" "}
            lần đổi nhãn danh tính.
          </p>
          <p>
            Confirmed đúng: {measured.confirmed_correct_faces} · Confirmed sai:{" "}
            {measured.confirmed_wrong_faces} lượt mặt.
          </p>
          <div className="diagnostic-model-table">
            <table>
              <thead>
                <tr>
                  <th>Nhánh</th>
                  <th>Top-1 đúng mục tiêu</th>
                  <th>Top-1 sai</th>
                  <th>Frame không có điểm hợp lệ</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(measured.models).map(([key, m]) => (
                  <tr key={key}>
                    <td>{key}</td>
                    <td>
                      {m.top1_correct}/{m.samples}
                    </td>
                    <td>{m.top1_wrong}</td>
                    <td>{m.frames_without_valid_score}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <small>{measured.caution}</small>
        </div>
      )}
    </details>
  );
}
function DiagnosticResult({ record }) {
  if (!record?.summary) return null;
  const summary = record.summary,
    analysis = record.analysis;
  return (
    <section
      className="card diagnostic-summary"
      aria-label="Phân tích tự động lượt kiểm thử"
      data-run-id={record.id}
    >
      <strong>
        Phân tích tự động · {record.scenario || "Lượt kiểm thử gần nhất"}
      </strong>
      {record.created_at && (
        <small className="diagnostic-run-time">
          Lượt {record.id.slice(-6)} ·{" "}
          {new Intl.DateTimeFormat("vi-VN", {
            timeZone: "Asia/Bangkok",
            dateStyle: "short",
            timeStyle: "medium",
          }).format(new Date(record.created_at.replace(" ", "T") + "Z"))}
        </small>
      )}
      <p>
        {summary.frames} frame AI / {summary.duration_seconds.toFixed(1)} giây ·{" "}
        {summary.confirmed_frames ?? 0} frame Confirmed
        {summary.tracked_frames > 0 &&
          ` · ${summary.tracked_frames} frame giữ tên bằng tracking`}{" "}
        · Độ trễ P50/P95: {summary.latency_p50_ms?.toFixed(0) ?? "—"}/
        {summary.latency_p95_ms?.toFixed(0) ?? "—"} ms
        {summary.duration_seconds > 0 &&
          ` · AI thực đo ${(summary.frames / summary.duration_seconds).toFixed(2)} FPS`}
      </p>
      {record.context?.feedback?.boxes_during_fast_movement ===
        "not_observed_with_certainty" && (
        <p role="status">
          Bạn chưa quan sát chắc box khi di chuyển nhanh. Lượt này chưa được
          dùng để tính độ chính xác danh tính hoặc hiệu chuẩn. Có tháo kính ở
          phần khẩu trang/che mặt; chưa có nhãn kính theo từng frame.
        </p>
      )}
      {record.context?.feedback?.glasses_changed && (
        <p role="status">
          Có thay đổi kính trong lượt này; chưa xác minh thời điểm đeo/tháo kính
          ở từng frame. Không coi toàn bộ lượt là dữ liệu không đeo kính. Động
          tác trên đồng hồ là lịch hướng dẫn, chỉ đánh giá đúng điều kiện sau
          khi đối chiếu ảnh hoặc xác nhận của bạn.
        </p>
      )}
      {record.context?.grouped_followup?.replay_clock_fix && (
        <p>
          Đã sửa đồng hồ replay cho khớp Live:{" "}
          {record.context.grouped_followup.replay_clock_fix.after_confirmed}{" "}
          Confirmed +{" "}
          {record.context.grouped_followup.replay_clock_fix.after_tracked} duy
          trì tên. Hai cấu hình mặt nhỏ được hiệu chuẩn chặt hơn; thử nghiệm tái
          xác minh không cải thiện đã hoàn nguyên. MaskInv chưa được bật vì thử
          nghiệm không tăng số frame có danh tính. Khẩu trang và che mặt vẫn còn
          hạn chế.
        </p>
      )}
      {analysis?.visual_flow && (
        <p>
          Frame video trung gian có box: {analysis.visual_flow.frames_with_box}/
          {analysis.visual_flow.frames} frame đã ghi nhận · Phục hồi từ đường
          ảnh phía trước: {analysis.visual_flow.forward_rescued_boxes} box.{" "}
          {analysis.visual_flow.caution}
        </p>
      )}
      {record.context?.continuity_followup && (
        <p>
          Phát lại cùng {summary.frames} frame sau sửa liên kết chuyển động:{" "}
          {record.context.continuity_followup.actual_confirmed} Confirmed +{" "}
          {record.context.continuity_followup.actual_tracked} tracking →{" "}
          {record.context.continuity_followup.replay_confirmed} Confirmed +{" "}
          {record.context.continuity_followup.replay_tracked} tracking. Hai
          frame phục hồi chỉ duy trì danh tính đã xác nhận, không thêm thông báo
          hoặc nhận diện mới. Có nhảy nhẹ, mặt vẫn trong video nhưng mất box
          trên preview: phần này còn cần kiểm tra.
        </p>
      )}
      {record.context?.preview_tracking_fix && (
        <p>
          Phát lại video gốc để kiểm tra box trung gian:{" "}
          {record.context.preview_tracking_fix.video_replay.boxes_before} →{" "}
          {record.context.preview_tracking_fix.video_replay.boxes_after} box,
          phục hồi thêm{" "}
          {record.context.preview_tracking_fix.video_replay.forward_rescued} box
          bằng đặc trưng ảnh thực. Đây là kiểm chứng hình học trên video, chưa
          phải độ chính xác danh tính hoặc kết quả Camo mới. Lượt kiểm chứng
          chuyển động 3 m sau sửa vẫn cần xác nhận sẵn sàng riêng.
        </p>
      )}
      {record.context?.motion_validation_followup && (
        <p>
          Kiểm chứng di chuyển 3 m:{" "}
          {record.context.motion_validation_followup.actual.frames} frame AI,{" "}
          {record.context.motion_validation_followup.actual.confirmed} Confirmed
          + {record.context.motion_validation_followup.actual.tracked} duy trì
          tên. Box có ở{" "}
          {record.context.motion_validation_followup.preview.with_box}/
          {record.context.motion_validation_followup.preview.reported_frames}{" "}
          frame trung gian đã ghi nhận; không phải recall toàn bộ video. FPS
          hiển thị trung vị{" "}
          {record.context.motion_validation_followup.performance.render_fps_p50.toFixed(
            2,
          )}
          ; độ trễ AI trung vị{" "}
          {record.context.motion_validation_followup.performance.processing_ms_p50.toFixed(
            0,
          )}{" "}
          ms. Góc nghiêng lớn vẫn còn Unknown; không hạ ngưỡng vì các phương án
          hiệu chuẩn thử còn cho người ngoài vượt qua.
        </p>
      )}
      {record.context?.specialist_probe && (
        <p>
          Đối chứng {record.context.specialist_probe.name}:{" "}
          {record.context.specialist_probe.frames} frame video,{" "}
          {record.context.specialist_probe.inference_errors} lỗi suy luận; xếp
          hạng đúng {record.context.specialist_probe.top1_correct}/
          {record.context.specialist_probe.human_labels} mặt có nhãn. Mẫu mặt
          rất nhỏ còn chồng lấn điểm dữ liệu âm. Chưa đủ để kết luận độ chính
          xác hoặc bật xác nhận; nhánh này chỉ thu số đo, cần kiểm chứng góc
          nghiêng ở 3 m.
        </p>
      )}
      {record.context?.profile_followup && (
        <p>
          Đánh giá kính + góc nghiêng tại 3 m:{" "}
          {record.context.profile_followup.confirmed} frame xác nhận đúng,{" "}
          {record.context.profile_followup.tracked} frame duy trì đúng tên và{" "}
          {record.context.profile_followup.remaining} frame chưa xác nhận. Góc
          khoảng 90° vẫn chưa đạt; theo lựa chọn của bạn, ưu tiên lượt không đeo
          kính tiếp theo. Model đối chứng chưa được bật vào Fusion; ngưỡng cũ
          được giữ nguyên.
        </p>
      )}
      {record.context?.native_detail_replay && (
        <p>
          Phát lại {record.context.native_detail_replay.frames} ảnh Camo gốc với
          cùng detector, landmarks và timestamp. Ngưỡng kích thước đầu vào từ
          {" "}{record.context.native_detail_replay.minimum_before} xuống
          {" "}{record.context.native_detail_replay.minimum_after} pixel cho
          {" "}{record.context.native_detail_replay.before.confirmed} →
          {" "}{record.context.native_detail_replay.after.confirmed} Confirmed và
          {" "}{record.context.native_detail_replay.after.tracked} frame duy trì tên.
          Ngưỡng của LVFace-B, AntelopeV2 và PETALface được giữ nguyên. Nhóm khẩu trang sau chuyển tiếp
          còn {record.context.native_detail_replay.mask_confirmed} Confirmed trên
          {" "}{record.context.native_detail_replay.mask_frames} frame.
          Video đối chứng có {record.context.native_detail_replay.video.frames} frame,
          {" "}{record.context.native_detail_replay.video.absent_named} xác nhận trong
          đoạn đã xác nhận vắng mục tiêu và
          {" "}{record.context.native_detail_replay.video.absent_alerts} thông báo sai
          trong đoạn này. Hồi quy điểm/track cũ tại 1 m và 3 m giữ nguyên kết quả.
          Đây là kết quả phát lại, cần đo lại FPS hiển thị bằng camera thật.
        </p>
      )}
      {record.context?.native_detail_replay?.kprpe_validation && (
        <p>
          Nhánh KP-RPE cho mặt nhỏ đã được chuẩn bị ở chế độ validation.
          Phát lại với nhánh này có
          {" "}{record.context.native_detail_replay.kprpe_validation.after.confirmed} Confirmed và
          {" "}{record.context.native_detail_replay.kprpe_validation.after.tracked} frame duy trì tên;
          khẩu trang ngoài chuyển tiếp có
          {" "}{record.context.native_detail_replay.kprpe_validation.after.evaluated_B_confirmed}/
          {record.context.native_detail_replay.mask_frames} frame Confirmed.
          Hiệu chuẩn bước đầu dựa trên một người và video đối chứng.
          Cần kiểm chứng camera và FPS trước khi đưa nhánh này vào Live chính.
        </p>
      )}
      {record.context?.occlusion_followup && (
        <p>
          Đã bổ sung dải KP-RPE 28–89 pixel cho validation với gate đã được
          đối chứng độc lập. Phát lại cùng lượt có
          {" "}{record.context.occlusion_followup.before.confirmed_faces} →
          {" "}{record.context.occlusion_followup.after.confirmed_faces} Confirmed và
          {" "}{record.context.occlusion_followup.after.tracked_faces} frame giữ tên bằng tracking.
          Dải mới có {record.context.occlusion_followup.false_passes} mẫu âm vượt gate
          trên {record.context.occlusion_followup.negative_observations} mẫu đã đối chứng.
          Nhánh này vẫn ở validation. Khẩu trang còn yếu.
        </p>
      )}
      {record.context?.occlusion_followup?.eye_alignment && (
        <p>
          Căn chỉnh vùng mắt LVFace-B đã đối chứng trên 7 mặt do bạn xác nhận.
          Dùng 3 mặt để tính gate, 4 mặt còn lại có số frame đủ bằng chứng Fusion
          {" "}{record.context.occlusion_followup.eye_alignment.held_out_eligible_before} →
          {" "}{record.context.occlusion_followup.eye_alignment.held_out_eligible_after}/4.
          Đây là bằng chứng từng frame trước bước temporal. Có
          {" "}{record.context.occlusion_followup.eye_alignment.false_identify_passes}/
          {record.context.occlusion_followup.eye_alignment.known_negative_pairs} so sánh âm vượt gate.
          Các mẫu dương được chọn từ cùng một lượt. Nhánh vùng mắt đang đo đối chứng,
          cần validation độc lập trước khi bật xác nhận danh tính.
        </p>
      )}
      {record.context?.independent_occlusion_validation && (
        <p>
          Lượt 6 m mới đã được bạn xác nhận danh tính và box. Trên cùng
          {" "}{record.context.independent_occlusion_validation.frames} frame, phát lại có
          {" "}{record.context.independent_occlusion_validation.native.confirmed_faces} →
          {" "}{record.context.independent_occlusion_validation.kprpe.confirmed_faces} →
          {" "}{record.context.independent_occlusion_validation.eye.confirmed_faces} Confirmed
          khi lần lượt thêm KP-RPE và căn chỉnh vùng mắt. Tổ hợp cuối có
          {" "}{record.context.independent_occlusion_validation.eye.tracked_faces} frame giữ tên bằng tracking.
          Khẩu trang di chuyển còn
          {" "}{record.context.independent_occlusion_validation.stages.mask_movement.confirmed}/
          {record.context.independent_occlusion_validation.stages.mask_movement.faces} Confirmed,
          che mặt còn
          {" "}{record.context.independent_occlusion_validation.stages.partial_occlusion.confirmed}/
          {record.context.independent_occlusion_validation.stages.partial_occlusion.faces} ngoài chuyển tiếp.
          Video đối chứng giữ đúng ba mặt Viet được gán nhãn, chưa có gán tên trong
          phạm vi đã xác nhận mục tiêu vắng mặt. Nhánh vùng mắt chỉ được bật riêng
          trong validation ở dải 14–27 pixel. FPS Live của tổ hợp hiệu chỉnh cần đo tiếp.
        </p>
      )}
      {record.context?.independent_occlusion_validation?.fresh_pipeline_replay && (
        <p>
          Chạy lại đủ detector, alignment, recognition, tracking và Fusion từ
          ảnh gốc đã xác nhận kết quả trên. Thời gian xử lý P50/P95 đạt
          {" "}{record.context.independent_occlusion_validation.fresh_pipeline_replay.processing_ms.p50.toFixed(0)}/
          {record.context.independent_occlusion_validation.fresh_pipeline_replay.processing_ms.p95.toFixed(0)} ms.
          LVFace-B và nhánh vùng mắt dùng chung session trọng số.
        </p>
      )}
      {record.context?.independent_occlusion_validation?.heldout_identity_test && (
        <p>
          Đối chứng gallery một ảnh Viet trong bộ nhớ, dùng
          {" "}{record.context.independent_occlusion_validation.heldout_identity_test.frames} frame tri
          làm người ngoài watchlist, có khẩu trang và che mặt, cho
          {" "}{record.context.independent_occlusion_validation.heldout_identity_test.temporal_replay.alerts} thông báo
          và {record.context.independent_occlusion_validation.heldout_identity_test.temporal_replay.confirmed_faces} Confirmed.
          Phép thử mới bao phủ một danh tính ngoài watchlist.
        </p>
      )}
      {record.context?.technical_followup && (
        <p>
          Đã sửa nguồn thời gian của tracker theo timestamp camera, tách optical
          flow thành duy trì tên, và giữ frame AI gốc trong RAM có giới hạn để
          phát lại lỗi. Phát lại điểm/track của lượt này vẫn có{" "}
          {record.context.technical_followup.score_replay.confirmed} Confirmed
          và {record.context.technical_followup.score_replay.tracked} frame duy
          trì tên; chưa phải kiểm chứng cải thiện chuyển động bằng camera. Các
          ngưỡng nhận diện được giữ nguyên.
        </p>
      )}
      {analysis?.phase_analysis && (
        <details className="diagnostic-analysis-details" open>
          <summary>
            Kết quả theo giai đoạn · {analysis.phase_analysis.evaluated_frames}{" "}
            frame ngoài chuyển tiếp
          </summary>
          <p>{analysis.phase_analysis.interpretation}</p>
          <div className="table-scroll">
            <table className="model-evidence-table">
              <thead>
                <tr>
                  <th>Giai đoạn</th>
                  <th>Frame đo / chuyển tiếp</th>
                  <th>Confirmed / tracking / chưa rõ</th>
                  <th>Phát hiện mặt</th>
                  <th>Độ trễ P95</th>
                  <th>Thông báo</th>
                </tr>
              </thead>
              <tbody>
                {analysis.phase_analysis.stages.map((stage) => (
                  <tr key={stage.id}>
                    <td>
                      {stage.start}–{stage.end}s · {stage.label}
                      <small>
                        {stage.action_confirmed
                          ? "Đã xác nhận động tác"
                          : "Chưa xác nhận động tác"}
                      </small>
                    </td>
                    <td>
                      {stage.evaluated_frames} / {stage.transition_frames}
                    </td>
                    <td>
                      {stage.metrics.statuses.confirmed || 0} /{" "}
                      {stage.metrics.statuses.tracked || 0} /{" "}
                      {stage.fault_observations.unconfirmed}
                    </td>
                    <td>
                      {stage.metrics.frames_with_face}/{stage.evaluated_frames}
                    </td>
                    <td>
                      {stage.metrics.processing_ms?.p95?.toFixed(0) ?? "—"} ms
                    </td>
                    <td>{stage.alerts}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            Chuyển tiếp: {analysis.phase_analysis.transition_frames} frame,{" "}
            {analysis.phase_analysis.transition_alerts} thông báo. Không tính
            các frame này vào hiệu chuẩn từng tình huống.
          </p>
          {analysis.phase_analysis.stages.map((stage) => (
            <details key={stage.id}>
              <summary>{stage.label} · Phân loại lỗi và phát lại</summary>
              <p>
                Không có detection: {stage.fault_observations.detector_missing};
                alignment: {stage.fault_observations.alignment}; thực thi model:{" "}
                {stage.fault_observations.runtime}; chất lượng:{" "}
                {stage.fault_observations.quality}; reset liên kết:{" "}
                {stage.fault_observations.tracking_resets}.
              </p>
              <p>
                Bằng chứng nhận diện chưa đủ:{" "}
                {stage.fault_observations.recognition_insufficient}; đang xác
                minh Fusion/temporal:{" "}
                {stage.fault_observations.fusion_temporal_verification}; mâu
                thuẫn đáng tin cậy:{" "}
                {stage.fault_observations.credible_model_conflicts}. Đây là quan
                sát cần kiểm tra, chưa phải kết luận model hỏng.
              </p>
              <p>{stage.metrics.findings.join(" ")}</p>
              <p>
                Phát lại điểm và track: {stage.replay.statuses.confirmed || 0}{" "}
                Confirmed, {stage.replay.statuses.tracked || 0} duy trì tên. Đây
                không phải chạy lại detector hoặc model trên ảnh gốc.
              </p>
              <pre>{JSON.stringify(stage.metrics.models, null, 2)}</pre>
              <small>
                Frame cần xem:{" "}
                {stage.error_frame_ids.join(", ") ||
                  "Chưa ghi nhận frame mất tên"}
                . Cần kiểm tra nguyên nhân trước lượt tiếp theo; không tự áp
                dụng ngưỡng.
              </small>
            </details>
          ))}
        </details>
      )}
      {record.context?.post_test_correction && (
        <details open className="diagnostic-analysis-details">
          <summary>Khắc phục sau kiểm thử 3 m</summary>
          <p>
            Lượt đo gốc có{" "}
            {record.context.post_test_correction.actual_confirmed} Confirmed,{" "}
            {record.context.post_test_correction.actual_tracked} frame giữ tên
            và {record.context.post_test_correction.actual_unnamed} frame chưa
            hiện tên.
          </p>
          <p>
            Phát lại sau sửa đạt{" "}
            {record.context.post_test_correction.replay.confirmed_faces}{" "}
            Confirmed và{" "}
            {record.context.post_test_correction.replay.tracked_faces} frame giữ
            tên. Còn {record.context.post_test_correction.remaining_unnamed}{" "}
            frame chưa hiện tên, gồm{" "}
            {record.context.post_test_correction.initial_confirmation} frame xác
            nhận ban đầu và{" "}
            {record.context.post_test_correction.remaining_after_link_loss}{" "}
            frame thiếu liên kết hoặc bằng chứng mới đủ tin cậy.
          </p>
          <p>
            Khẩu trang 3 m phục hồi{" "}
            {record.context.post_test_correction.mask_confirmed_before} →{" "}
            {record.context.post_test_correction.mask_confirmed_after}{" "}
            Confirmed. Video đối chứng có{" "}
            {record.context.post_test_correction.video_verified_alerts} cảnh báo
            đúng đã được bạn xác nhận và{" "}
            {record.context.post_test_correction.video_false_alerts} cảnh báo
            trong khoảng mục tiêu vắng mặt.
          </p>
          <p>
            Đã sửa điều kiện duy trì tên trên track và giữ ngưỡng lịch sử theo
            đúng danh tính có nhãn, với kiểm chứng dữ liệu âm. Khi gallery thay
            đổi, nhánh phục hồi ngưỡng cần được xác minh lại.
          </p>
          <details>
            <summary>Phân loại từng frame chưa hiện tên</summary>
            <div className="table-scroll">
              <table className="model-evidence-table">
                <thead>
                  <tr>
                    <th>Frame</th>
                    <th>Giây</th>
                    <th>Sau sửa</th>
                    <th>Căn cứ</th>
                  </tr>
                </thead>
                <tbody>
                  {record.context.post_test_correction.frames?.map((entry) => (
                    <tr key={entry.frame}>
                      <td>{entry.frame}</td>
                      <td>{entry.elapsed.toFixed(3)}</td>
                      <td>{entry.status_label}</td>
                      <td>{entry.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
          <small>
            Kết quả phát lại dùng điểm model và thông tin track đã lưu. Các
            frame còn lại cần cặp ảnh gốc trước và sau khi đứt liên kết để kiểm
            tra tiếp. Lượt tiếp theo sẽ giữ các cặp này trong giới hạn 8 ảnh
            RAM, không ghi thêm video.
          </small>
        </details>
      )}
      {record.context?.continuity_replay && (
        <details open className="diagnostic-analysis-details">
          <summary>Đã kiểm tra 16 frame chưa Confirmed ở 1 m</summary>
          <p>
            Phát lại sau sửa đạt{" "}
            {record.context.continuity_replay.confirmed_faces}/172 Confirmed và{" "}
            {record.context.continuity_replay.tracked_faces} frame giữ tên với
            nhãn “theo dõi”. Hai frame đầu tiếp tục chờ xác nhận.
          </p>
          <p>
            Hai frame 407 và 410 đã khôi phục xác nhận sớm hơn nhờ bằng chứng
            mới từ nhiều model trên cùng track. Mười hai frame góc nghiêng thiếu
            bằng chứng nhận diện mới được duy trì danh tính qua chuỗi liên kết
            ảnh đã đối chứng.
          </p>
          <p>
            {record.context.continuity_replay.alerts} thông báo. Theo nhãn bạn
            xác nhận, {record.context.continuity_replay.wrong_confirmed}{" "}
            Confirmed sai và {record.context.continuity_replay.wrong_tracked}{" "}
            tên theo dõi sai trên lượt này.
          </p>
          <small>
            Số đo camera gốc được giữ để đối chiếu. Liên kết tracking được kiểm
            tra trên video hiện có, cần tiếp tục đối chứng người khác và các
            khoảng cách 3 m, 6 m. Nhận diện mới ở góc 90° vẫn phụ thuộc thông
            tin khuôn mặt.
          </small>
        </details>
      )}
      {record.context?.same_frame_ablation && (
        <details className="diagnostic-analysis-details">
          <summary>
            Validation thực tế và đóng góp KP-RPE trên cùng frame
          </summary>
          <p>
            Thực đo: {record.context.same_frame_ablation.with_expert_confirmed}/
            {record.context.same_frame_ablation.frames} frame AI Confirmed. Khi
            tắt riêng bằng chứng KP-RPE và phát lại đúng những frame này:{" "}
            {record.context.same_frame_ablation.without_expert_confirmed}/
            {record.context.same_frame_ablation.frames}.
          </p>
          <p>
            KP-RPE được gọi ở{" "}
            {record.context.same_frame_ablation.expert_evaluated_frames} frame
            thuộc dải chất lượng đã đối chứng. Các frame khác vẫn được pipeline
            hiện tại phân tích.
          </p>
          {record.context.same_frame_ablation.identity_verified && (
            <p>
              Theo nhãn bạn xác nhận:{" "}
              {record.context.same_frame_ablation.actual_correct_confirmed}{" "}
              Confirmed đúng,{" "}
              {record.context.same_frame_ablation.actual_wrong_confirmed}{" "}
              Confirmed sai. Bộ ngưỡng của lượt validation được giữ nguyên và
              lượt này không được dùng để fit ngưỡng.
            </p>
          )}
          {record.context.unconfirmed_breakdown && (
            <p>
              Chưa Confirmed:{" "}
              {record.context.unconfirmed_breakdown
                .initial_temporal_confirmation || 0}{" "}
              frame đang xác nhận ban đầu,{" "}
              {record.context.unconfirmed_breakdown
                .new_temporal_votes_required || 0}{" "}
              frame đang tích lũy bằng chứng mới,{" "}
              {record.context.unconfirmed_breakdown
                .insufficient_current_calibrated_pair || 0}{" "}
              frame thiếu cặp bằng chứng đã hiệu chuẩn.
            </p>
          )}
          <small>
            Đây là một lượt Camo thực và phép đối chiếu phát lại trên cùng dữ
            liệu; không phải hai lượt Camo. Kết quả một người chưa xác nhận độ
            chính xác trên người mới hoặc toàn bộ góc 90°.
          </small>
        </details>
      )}
      {record.context?.fusion_comparison && (
        <details className="diagnostic-analysis-details">
          <summary>Fusion trước / sau trên cùng dữ liệu đã lưu</summary>
          <p>
            Thực đo trước sửa:{" "}
            {record.context.fusion_comparison.actual_confirmed}/{summary.frames}{" "}
            frame Confirmed.
          </p>
          <p>
            Phát lại sau sửa logic:{" "}
            {record.context.fusion_comparison.logic_only_confirmed}/
            {summary.frames}. Thử thêm nhánh đã hiệu chuẩn:{" "}
            {record.context.fusion_comparison.calibrated_expert_confirmed}/
            {summary.frames}. Số nhận sai theo nhãn:{" "}
            {record.context.fusion_comparison.wrong_confirmed}.
          </p>
          <p>
            Đối chứng video đủ {record.context.fusion_comparison.video_frames}{" "}
            frame: {record.context.fusion_comparison.video_false_alerts} cảnh
            báo trong khoảng vắng danh tính. Đây là phát lại, chưa phải lượt
            Camo mới; dữ liệu này đã tham gia phát triển/hiệu chuẩn.
          </p>
        </details>
      )}
      {(record.context?.candidate_trials || []).map((trial) => (
        <details className="diagnostic-analysis-details" key={trial.version}>
          <summary>
            Model pretrained đối chứng · KP-RPE · Chưa bật xác nhận
          </summary>
          <p>
            Đã đo riêng {trial.camera_samples.length} frame Camo có nhãn,{" "}
            {trial.video_negative_samples.length} khuôn mặt trong đoạn vắng
            watchlist và {trial.video_known_samples.length} mặt video có nhãn.
            Video đối chứng lấy mẫu mỗi giây; chưa phải kiểm thử toàn bộ video
            với model này.
          </p>
          <p>
            GPU P50: {trial.gpu_warm_p50_ms.toFixed(1)} ms/mặt · CPU P50:{" "}
            {trial.cpu_p50_ms.toFixed(1)} ms/mặt · Bộ nhớ CUDA riêng của model
            thử: {trial.candidate_cuda_peak_mb.toFixed(0)} MB.
          </p>
          <p>
            Model chưa hiệu chuẩn và chưa cấp phiếu xác nhận. Các mẫu liên tiếp
            của một người không chứng minh độ chính xác trên người mới.
          </p>
        </details>
      ))}
      {record.context?.reported_motion && (
        <p>Người dùng xác nhận có di chuyển nhẹ trong lượt kiểm thử.</p>
      )}
      {record.context?.series_appearance_note?.note && (
        <p>{record.context.series_appearance_note.note}</p>
      )}
      {analysis ? (
        <>
          <h3>{analysis.title}</h3>
          {analysis.error ? (
            <p>{analysis.error}</p>
          ) : (
            <details className="diagnostic-analysis-details">
              <summary>Xem số đo từng model, nguyên nhân và bước xử lý</summary>
              <ul>
                {(analysis.findings || []).map((text) => (
                  <li key={text}>{text}</li>
                ))}
              </ul>
              {analysis.tracking_methods && (
                <p>
                  Liên kết vị trí khi ngoại hình suy yếu ·{" "}
                  {analysis.geometry_only_associations ?? 0} lần. Làm mới bằng
                  chứng danh tính · {analysis.identity_evidence_resets ?? 0}{" "}
                  lần.
                </p>
              )}
              <div className="diagnostic-model-table">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Mẫu hợp lệ</th>
                      <th>Cosine trung vị</th>
                      <th>P05 – P95</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(analysis.models || {}).map(
                      ([key, model]) => (
                        <tr key={key}>
                          <td>{key}</td>
                          <td>{model.samples}</td>
                          <td>{model.cosine?.p50?.toFixed(3) ?? "—"}</td>
                          <td>
                            {model.cosine
                              ? `${model.cosine.p05.toFixed(3)} – ${model.cosine.p95.toFixed(3)}`
                              : "—"}
                          </td>
                        </tr>
                      ),
                    )}
                  </tbody>
                </table>
              </div>
              <p>
                Nguồn:{" "}
                {Object.keys(analysis.source_sizes || {}).join(", ") || "—"} ·
                Mặt gốc trung vị: {analysis.face_pixels?.p50 ?? "—"} px ·{" "}
                {analysis.ground_truth_runs?.positive ?? 0} lượt có mục tiêu /{" "}
                {analysis.ground_truth_runs?.negative ?? 0} lượt đối chứng.
                {analysis.ground_truth_videos?.sources > 0 &&
                  ` Video có nhãn: ${analysis.ground_truth_videos.sources} nguồn, ${analysis.ground_truth_videos.positive_intervals} khoảng có mặt / ${analysis.ground_truth_videos.negative_intervals} khoảng vắng mặt.`}
              </p>
              {analysis.actions?.length > 0 && (
                <ul>
                  {analysis.actions.map((text) => (
                    <li key={text}>{text}</li>
                  ))}
                </ul>
              )}
              <small>{analysis.caution}</small>
              {analysis.pipeline_revisions && (
                <p>
                  Pipeline của lượt đo:{" "}
                  {Object.keys(analysis.pipeline_revisions).join(", ")}
                </p>
              )}
            </details>
          )}
        </>
      ) : (
        <p role="status">
          Đang tự động phân tích số đo đã lưu… Kết quả cập nhật ngay tại đây.
        </p>
      )}
      {analysis && !analysis.preparation_evidence_isolated && (
        <p>
          Thời điểm cảnh báo của lượt cũ còn chịu ảnh hưởng giai đoạn chuẩn bị;
          chưa dùng để kết luận thời gian xác nhận thực tế.
        </p>
      )}
      {analysis?.calibration_replay && (
        <div className="recognition-notice compact">
          <strong>
            Phát lại số đo với Confirmed mới · Không phải lượt camera mới
          </strong>
          <span>
            {analysis.calibration_replay.confirmed_faces}/
            {analysis.calibration_replay.frames} lượt mặt được xác nhận ·{" "}
            {analysis.calibration_replay.alerts} thông báo dự kiến.{" "}
            {analysis.calibration_replay.tracked_faces > 0 &&
              ` ${analysis.calibration_replay.tracked_faces} lượt mặt giữ tên bằng tracking.`}
            {analysis.calibration_replay.identity_verified
              ? ` Theo nhãn đã xác nhận: ${analysis.calibration_replay.correct_confirmed} đúng, ${analysis.calibration_replay.wrong_confirmed} sai.`
              : " Danh tính đúng/sai chưa có đủ nhãn đối chứng."}
          </span>
          <small>
            {analysis.calibration_replay.used_for_fitting
              ? "Lượt này tham gia tính ngưỡng, không được coi là tập kiểm chứng độc lập. "
              : ""}
            {analysis.calibration_replay.caution}
          </small>
        </div>
      )}
      {record.person_id && record.expected_present ? (
        <DiagnosticReview record={record} />
      ) : null}
    </section>
  );
}
export default function Diagnostic({ people, health, onError, stopLive }) {
  const [confirm, setConfirm] = useState(false),
    [state, setState] = useState("idle");
  const [scenario, setScenario] = useState(scenarios[0]),
    [person, setPerson] = useState("");
  const [purpose, setPurpose] = useState("validation");
  const [present, setPresent] = useState(true),
    [count, setCount] = useState(null);
  const [frame, setFrame] = useState(null),
    [summary, setSummary] = useState(null),
    [fps, setFps] = useState(null);
  const video = useRef(null),
    live = useRef(null),
    client = useRef(null),
    run = useRef(null),
    session = useRef(null);
  const surface = useRef(null),
    view = useRef(null);
  const phase = useRef("idle"),
    deadline = useRef(0),
    timer = useRef(null),
    busy = useRef(false);
  const alive = useRef(true),
    attempt = useRef(0),
    lastAI = useRef(0);
  const closing = useRef(false);
  const closeTestTab = useRef(false);
  const expertProbe = useRef(null);
  const expertMode = useRef("shadow");
  const additionalExperts = useRef([]);
  const eyewearMode = useRef(null);
  const finishTimer = useRef(null),
    backendPhase = useRef(null);
  const [diagnosticError, setDiagnosticError] = useState("");
  const [protocol, setProtocol] = useState(null),
    [currentStage, setCurrentStage] = useState(null);
  const protocolRef = useRef(null);
  const measurementClock = useRef(null);
  const [recent, setRecent] = useState(null);
  const [history, setHistory] = useState([]),
    [selectedRun, setSelectedRun] = useState("");
  const selection = useRef("");
  const [coverage, setCoverage] = useState([]);
  const completedId = useRef(null);
  const completedAt = useRef(0);
  useEffect(() => {
    let closed = false,
      busy = false,
      signature = "";
    const refresh = async () => {
      if (closed || busy) return;
      busy = true;
      const requestedAt = performance.now();
      try {
        const data = await api("/diagnostics");
        if (
          closed ||
          requestedAt < completedAt.current ||
          !Array.isArray(data.runs)
        )
          return;
        setCoverage(data.live_cases || []);
        setHistory(data.runs);
        const latest =
          data.runs.find((r) => r.id === selection.current) ||
          data.runs.find(
            (r) => r.state === "complete" && r.summary?.frames > 0,
          ) ||
          data.runs.find((r) =>
            ["complete", "cancelled", "error"].includes(r.state),
          );
        const next = JSON.stringify(latest || null);
        if (next !== signature) {
          signature = next;
          setRecent(latest || null);
        }
      } catch {
        /* Health status already exposes backend disconnections; retain the last measured result. */
      } finally {
        busy = false;
      }
    };
    refresh();
    const interval = setInterval(refresh, 1500);
    return () => {
      closed = true;
      clearInterval(interval);
    };
  }, []);
  useEffect(() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get("diagnostic") === "camo") {
      eyewearMode.current =
        url.searchParams.get("eyewear") === "none"
          ? "none"
          : url.searchParams.get("eyewear") === "unchanged"
            ? "unchanged"
            : null;
      expertMode.current =
        url.searchParams.get("expert_mode") === "calibrated"
          ? "calibrated"
          : "shadow";
      expertProbe.current = ["kprpe12m", "maskinv_hg", "aroface", "lvface_eye"].includes(
        url.searchParams.get("expert"),
      )
        ? url.searchParams.get("expert")
        : null;
      if (["maskinv_hg", "aroface"].includes(expertProbe.current))
        expertMode.current = "shadow";
      additionalExperts.current = url.searchParams.get("additional_expert") === "lvface_eye"
        ? ["lvface_eye"] : [];
      closeTestTab.current = url.searchParams.get("close_after") === "1";
      if (closeTestTab.current) {
        window.name = "facewatch-camo-test";
        document.title = "Kiểm thử Camo · FaceWatch AI";
      }
      const requestedCase = url.searchParams.get("case");
      const requestedPerson = url.searchParams.get("target");
      if (
        scenarios.includes(requestedCase) ||
        ((requestedCase?.startsWith("Tổng hợp ") ||
          conditions.some(([condition]) =>
            requestedCase?.startsWith(`${condition} `),
          )) &&
          / [0-9]+(?:\.[0-9]+)? m$/.test(requestedCase))
      )
        setScenario(requestedCase);
      if (url.searchParams.get("purpose") === "calibration")
        setPurpose("calibration");
      if (/^[a-f0-9]{32}$/.test(requestedPerson || ""))
        setPerson(requestedPerson);
      setConfirm(true); // Open the consent gate only; never start capture from a URL.
      url.searchParams.delete("diagnostic");
      url.searchParams.delete("case");
      url.searchParams.delete("target");
      url.searchParams.delete("purpose");
      url.searchParams.delete("close_after");
      url.searchParams.delete("expert");
      url.searchParams.delete("expert_mode");
      url.searchParams.delete("additional_expert");
      url.searchParams.delete("eyewear");
      window.history.replaceState(
        null,
        "",
        url.pathname + url.search + url.hash,
      );
    }
  }, []);
  function closeDedicatedTab() {
    if (!closeTestTab.current) return;
    window.close();
    // A new dedicated tab has one history entry. Never try to close another tab.
    setTimeout(() => {
      if (!window.closed && alive.current)
        onError(
          "Camera đã dừng. Trình duyệt giữ tab kiểm thử này; bạn có thể đóng riêng tab này.",
        );
    }, 300);
  }
  function change(next) {
    phase.current = next;
    if (alive.current) setState(next);
  }
  function release() {
    attempt.current++;
    clearInterval(timer.current);
    const ws = client.current;
    client.current = null;
    ws?.close();
    view.current?.stop();
    view.current = null;
    live.current?.getTracks().forEach((t) => t.stop());
    live.current = null;
    if (video.current) video.current.srcObject = null;
    if (session.current) {
      api("/camera/stop", json("POST", { session_id: session.current })).catch(
        () => {},
      );
      session.current = null;
    }
  }
  async function cancel(message) {
    if (closing.current) return;
    closing.current = true;
    release();
    if (run.current)
      api(`/diagnostics/${run.current}/stop`, json("POST", {})).catch(() => {});
    run.current = null;
    if (alive.current) {
      change(message ? "error" : "idle");
      setConfirm(false);
      if (message) {
        setDiagnosticError(message);
        onError(message);
      }
    }
    closing.current = false;
    if (!message) closeDedicatedTab();
  }
  function complete(status) {
    if (closing.current) return;
    closing.current = true;
    release();
    completedId.current = status.id;
    selection.current = "";
    setSelectedRun("");
    completedAt.current = performance.now();
    setSummary(status.summary);
    setRecent({
      id: status.id,
      scenario,
      summary: status.summary,
      analysis: status.analysis || null,
    });
    setCount(0);
    change("complete");
    run.current = null;
    finishTimer.current = setTimeout(() => {
      if (alive.current) change("idle");
      closing.current = false;
      closeDedicatedTab();
    }, 1600);
  }
  useEffect(
    () => () => {
      alive.current = false;
      clearTimeout(finishTimer.current);
      release();
      if (run.current)
        fetch(`/api/diagnostics/${run.current}/stop`, {
          ...json("POST", {}),
          keepalive: true,
        });
    },
    [],
  );
  async function send(ws) {
    if (
      ws !== client.current ||
      ws.readyState !== WebSocket.OPEN ||
      !video.current?.videoWidth
    )
      return;
    try {
      await view.current?.encodeNext((header, blob) => {
        if (ws !== client.current || ws.readyState !== WebSocket.OPEN)
          return false;
        if (header) {
          header.diagnostic_capture_pending = measurementClock.current === null;
          if (measurementClock.current !== null)
            header.diagnostic_elapsed =
              (header.captured_at - measurementClock.current) / 1000;
          ws.send(JSON.stringify(header));
        }
        ws.send(blob);
        return true;
      });
    } catch (error) {
      if (ws === client.current && !closing.current) cancel(error.message);
    }
  }
  async function start() {
    if (phase.current !== "idle" || closing.current) return;
    setConfirm(false);
    setDiagnosticError("");
    setFrame(null);
    setSummary(null);
    setCount(null);
    setFps(null);
    change("opening");
    backendPhase.current = null;
    measurementClock.current = null;
    busy.current = false;
    const token = ++attempt.current;
    try {
      await stopLive();
      const created = await api(
        "/diagnostics",
        json("POST", {
          ready: true,
          purpose,
          scenario,
          person_id: person || null,
          expected_present: present,
          expert: expertProbe.current,
          expert_mode: expertMode.current,
          additional_experts: additionalExperts.current,
          eyewear: eyewearMode.current,
        }),
      );
      if (token !== attempt.current) {
        await api(`/diagnostics/${created.id}/stop`, json("POST", {}));
        return;
      }
      run.current = created.id;
      protocolRef.current = created.protocol;
      setProtocol(created.protocol);
      setCurrentStage(null);
      const devices = await navigator.mediaDevices.enumerateDevices();
      const chosen = cameraFor(devices, "camo");
      if (!chosen)
        throw new Error(
          "Không tìm thấy Camo Camera. Kiểm tra kết nối iPhone và cấp quyền camera cho localhost.",
        );
      const stream = await acquireCamera(navigator.mediaDevices, chosen, "camo");
      if (token !== attempt.current) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      live.current = stream;
      const v = video.current;
      v.srcObject = stream;
      await v.play();
      if (token !== attempt.current) return;
      view.current = new CameraView(
        v,
        surface.current,
        (stats) => {
          setFps(stats.renderFps);
        },
        (message) => cancel(message),
      );
      view.current.start();
      const track = stream.getVideoTracks()[0];
      track.onended = () =>
        cancel("Camera đã ngắt kết nối. Lượt kiểm thử đã dừng.");
      const reserved = await api(
        "/camera/start",
        json("POST", { kind: "camo", label: chosen.label }),
      );
      if (token !== attempt.current) {
        await api(
          "/camera/stop",
          json("POST", { session_id: reserved.session_id }),
        );
        return;
      }
      session.current = reserved.session_id;
      const ws = socket("/ws/live");
      client.current = ws;
      ws.onopen = () =>
        ws.send(
          JSON.stringify({
            kind: "camo",
            label: chosen.label,
            session_id: reserved.session_id,
            diagnostic_id: created.id,
            return_image: false,
            track_between_frames: true,
          }),
        );
      ws.onmessage = (event) => {
        if (client.current !== ws) return;
        const data = JSON.parse(event.data);
        if (data.type === "error") {
          cancel(data.detail);
          return;
        }
        if (data.type === "ready") {
          view.current.trackingEnabled = data.track_between_frames === true;
          send(ws);
          return;
        }
        if (data.type === "visual_frames") {
          if (data.session_id === session.current)
            view.current?.acceptVisual(data);
          return;
        }
        if (data.type === "live_frame") {
          if (data.session_id !== session.current) return;
          view.current?.accept(data);
          lastAI.current = performance.now();
          setFrame(data);
          backendPhase.current = data.diagnostic;
          if (
            data.diagnostic?.state === "preparing" &&
            phase.current === "opening" &&
            view.current?.canvasFrameId
          ) {
            deadline.current = performance.now() + 5000;
            change("preparing");
            setCount(5);
          }
          if (data.diagnostic?.state === "complete") {
            complete(data.diagnostic);
            return;
          }
          send(ws);
        }
      };
      ws.onerror = () => cancel("Mất kết nối backend kiểm thử.");
      ws.onclose = () => {
        if (ws !== client.current || closing.current) return;
        if (
          phase.current === "running" &&
          performance.now() >= deadline.current - 500
        )
          return;
        cancel("Phiên camera đã đóng trước khi hoàn thành.");
      };
      const opened = performance.now();
      let lastPoll = 0;
      timer.current = setInterval(async () => {
        if (closing.current || !run.current) return;
        const now = performance.now();
        if (phase.current === "opening" && now - opened > 25000) {
          cancel("Camera/backend chưa có frame AI sau 25 giây.");
          return;
        }
        if (
          ["preparing", "running"].includes(phase.current) &&
          (now - (view.current?.lastCaptureAt ?? 0) > 3000 ||
            now - lastAI.current > 8000)
        ) {
          cancel(
            "Mất frame camera hoặc backend không phản hồi; đồng hồ đã dừng.",
          );
          return;
        }
        if (phase.current === "preparing") {
          setCount(Math.max(1, Math.ceil((deadline.current - now) / 1000)));
          if (now >= deadline.current && !busy.current) {
            busy.current = true;
            try {
              const status = await api(
                `/diagnostics/${run.current}/begin`,
                json("POST", {}),
              );
              if (token !== attempt.current) return;
              const acknowledgedAt = performance.now();
              measurementClock.current =
                acknowledgedAt -
                (status.protocol.duration_seconds - status.remaining) * 1000;
              deadline.current = acknowledgedAt + status.remaining * 1000;
              change("running");
              setCount(Math.ceil(status.remaining));
              setCurrentStage(phaseAt(protocolRef.current, 0));
            } catch (e) {
              cancel(e.message);
            } finally {
              busy.current = false;
            }
          }
        }
        if (phase.current === "running") {
          setCount(Math.max(0, Math.ceil((deadline.current - now) / 1000)));
          const nextStage = phaseAt(
            protocolRef.current,
            protocolRef.current.duration_seconds -
              Math.max(0, (deadline.current - now) / 1000),
          );
          // Only redraw instructions when a displayed second or action changes.
          setCurrentStage((previous) =>
            previous?.id === nextStage?.id &&
            previous?.transition === nextStage?.transition &&
            Math.ceil(previous?.remaining_seconds) ===
              Math.ceil(nextStage?.remaining_seconds) &&
            Math.ceil(previous?.transition_remaining_seconds) ===
              Math.ceil(nextStage?.transition_remaining_seconds)
              ? previous
              : nextStage,
          );
          if (now - lastPoll > 500 && !busy.current) {
            lastPoll = now;
            busy.current = true;
            try {
              const status = await api(`/diagnostics/${run.current}`);
              if (status.state === "complete") complete(status);
              else if (["error", "cancelled"].includes(status.state))
                cancel("Lượt kiểm thử đã dừng do mất camera/backend.");
            } catch (e) {
              cancel(e.message);
            } finally {
              busy.current = false;
            }
          }
        }
      }, 80);
    } catch (e) {
      if (token === attempt.current) await cancel(e.message);
    }
  }
  const open = ["opening", "preparing", "running", "complete"].includes(state);
  return (
    <>
      <div className="diagnostic-entry">
        <button
          onClick={() => {
            setPurpose("validation");
            setConfirm(true);
          }}
          disabled={!health?.ready || open}
        >
          Kiểm thử AI bằng Camo Camera
        </button>
        <span>
          Live một người · 6 m ·{" "}
          {health?.models
            ? Object.entries(health.models)
                .map(
                  ([key, v]) =>
                    `${key} · ${v.state} (${v.device || v.error || "—"}${v.batch_limit ? ` · batch ${v.batch_limit}` : ""})`,
                )
                .join(" · ")
            : "Đang kiểm tra model"}
        </span>
      </div>
      <CalibrationTools
        health={health}
        locked={open}
        onCollect={() => {
          setPurpose("calibration");
          setScenario("Quay trái/phải 1 m");
          setConfirm(true);
        }}
      />
      <details className="card diagnostic-coverage">
        <summary>Tiến độ Live 3–6–9 m · Dữ liệu 1 m giữ để hồi quy</summary>
        <p>
          Ưu tiên 3 m → 6 m → 9 m. Không cần lặp lại bài tổng hợp 1 m. Lượt tổng
          hợp: 5 giây chuẩn bị + 180 giây đo, gồm 6 giai đoạn. Các lượt ngắn
          được giữ để kiểm chứng riêng lỗi sau sửa. “Đã đo” chưa có nghĩa đạt độ
          chính xác; phải phân tích và xử lý lỗi có căn cứ trước lượt tiếp theo.
        </p>
        <div className="diagnostic-model-table">
          <table>
            <thead>
              <tr>
                <th>Tình huống solo</th>
                {[3, 6, 9].map((d) => (
                  <th key={d}>{d} m</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[["Tổng hợp"], ...conditions].map(([condition]) => (
                <tr key={condition}>
                  <td>{condition}</td>
                  {[3, 6, 9].map((d) => {
                    const name = `${condition} ${d} m`,
                      item = coverage.find((c) => c.scenario === name);
                    return (
                      <td key={d}>
                        <button
                          disabled={!health?.ready || open}
                          onClick={() => {
                            setPurpose("validation");
                            setScenario(name);
                            setConfirm(true);
                          }}
                          aria-label={`${name} — chọn lượt`}
                        >
                          {item?.measured
                            ? `Đã đo (${item.runs.length})`
                            : "Chưa đo"}
                        </button>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      {history.some((r) => r.summary?.frames > 0) && (
        <label className="diagnostic-history-select">
          Xem lượt đã đo
          <select
            value={selectedRun}
            onChange={(e) => {
              selection.current = e.target.value;
              setSelectedRun(e.target.value);
              const record = history.find((r) => r.id === e.target.value);
              if (record) setRecent(record);
              else
                setRecent(
                  history.find(
                    (r) => r.state === "complete" && r.summary?.frames > 0,
                  ) || null,
                );
            }}
          >
            <option value="">Lượt hoàn thành gần nhất</option>
            {history
              .filter((r) => r.summary?.frames > 0)
              .map((r) => (
                <option key={r.id} value={r.id}>
                  {r.scenario} · {r.id.slice(-6)}
                </option>
              ))}
          </select>
        </label>
      )}
      <DiagnosticResult
        record={
          recent ||
          (summary
            ? { id: completedId.current, scenario, summary, analysis: null }
            : null)
        }
      />
      {state === "error" && (
        <div className="modal-backdrop">
          <div className="modal" role="alertdialog" aria-label="Lỗi kiểm thử camera">
            <h2>Kiểm thử đã dừng</h2>
            <p>{diagnosticError}</p>
            <p>Camera đã được giải phóng. Lượt mới cần xác nhận sẵn sàng riêng.</p>
            <button onClick={() => { change("idle"); closeDedicatedTab(); }}>Đóng giao diện kiểm thử</button>
          </div>
        </div>
      )}
      {confirm && (
        <div className="modal-backdrop">
          <div
            className="modal"
            role="dialog"
            aria-label="Xác nhận sẵn sàng kiểm thử"
          >
            <h2>Bạn đã sẵn sàng?</h2>
            <p>
              Các frame gốc đưa qua AI ở lượt mới được lưu để phát lại lỗi sau khi đóng ứng dụng
              (bộ đệm RAM tối đa 96 MB cho lượt tổng hợp, 16 MB cho lượt ngắn).
              Dung lượng lưu trên đĩa phụ thuộc số frame AI thu được. Chọn ground truth trước mỗi lượt. Camera chỉ mở sau
              khi bạn xác nhận. Chỉ cần một mình bạn; nhiều người và người ngoài
              watchlist được kiểm tra bằng video riêng.
            </p>
            <label>
              Mục đích lượt đo
              <select
                aria-label="Mục đích lượt đo"
                value={purpose}
                onChange={(event) => setPurpose(event.target.value)}
              >
                <option value="validation">
                  Kiểm chứng độc lập sau khi sửa
                </option>
                <option value="calibration">
                  Bổ sung dữ liệu hiệu chuẩn — chưa tính là kiểm chứng
                </option>
              </select>
            </label>
            <label>
              Tình huống kiểm thử
              <select
                value={scenario}
                onChange={(e) => setScenario(e.target.value)}
              >
                {!scenarios.includes(scenario) && (
                  <option value={scenario}>{scenario}</option>
                )}
                {scenarios.map((s) => (
                  <option key={s}>{s}</option>
                ))}
              </select>
            </label>
            {(eyewearMode.current === "none" ||
              (eyewearMode.current === null &&
                scenario.startsWith("Tổng hợp "))) && (
              <p role="status">
                <strong>LƯỢT NÀY KHÔNG ĐEO KÍNH</strong> — tháo kính trước khi
                bấm sẵn sàng và giữ nguyên trong suốt lượt.
              </p>
            )}
            {expertProbe.current && (
              <p>
                {expertMode.current === "calibrated"
                  ? "Lượt kiểm chứng Fusion: model bổ trợ chỉ tham gia ở dải chất lượng đã được hiệu chuẩn và đối chứng. Khi bằng chứng chưa đủ, hệ thống vẫn giữ Chưa xác định."
                  : `Đối chứng thêm ${{ maskinv_hg: "MaskInv-HG", aroface: "ARoFace", kprpe12m: "KP-RPE", lvface_eye: "LVFace-B căn chỉnh vùng mắt" }[expertProbe.current]} trên cùng khuôn mặt và frame. Model thử nghiệm chỉ ghi số đo, chưa cấp phiếu xác nhận danh tính.`}
              </p>
            )}
            {additionalExperts.current.includes("lvface_eye") && (
              <p>
                Bổ sung LVFace-B căn chỉnh vùng mắt, dùng gallery và gate riêng.
                Hai cách căn chỉnh LVFace-B được tính chung một nguồn model.
              </p>
            )}
            {person && (
              <p>
                Hồ sơ {people.find((p) => p.id === person)?.name}:{" "}
                {people.find((p) => p.id === person)?.images?.length ?? 0} ảnh
                đăng ký. Số ảnh được lưu cùng lượt đo, không thay đổi ảnh gốc.
              </p>
            )}
            <label>
              Danh tính mục tiêu dự kiến
              <select
                value={person}
                onChange={(e) => setPerson(e.target.value)}
              >
                <option value="">Chọn người mục tiêu</option>
                {people.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </label>
            <div className="diagnostic-instruction-preview">
              {(instructionsFor(scenario) || []).map((line) => (
                <strong key={line}>{line}</strong>
              ))}
            </div>
            <RecognitionNotice
              calibrated={health?.settings?.calibrated}
              status={health?.settings?.calibration_status}
            />
            <label>
              Mục tiêu trong khung hình
              <select
                value={String(present)}
                onChange={(e) => setPresent(e.target.value === "true")}
              >
                <option value="true">Có mặt</option>
                <option value="false">Không có mặt</option>
              </select>
            </label>
            <div className="dialog-actions">
              <button
                onClick={() => {
                  setConfirm(false);
                  closeDedicatedTab();
                }}
              >
                Chưa sẵn sàng
              </button>
              <button
                className="primary"
                disabled={present && !person}
                onClick={start}
              >
                Tôi đã sẵn sàng – Bắt đầu
              </button>
            </div>
          </div>
        </div>
      )}
      {open && (
        <div
          className="diagnostic-screen"
          role="dialog"
          aria-label="Kiểm thử Camo Camera"
        >
          <div className="diagnostic-heading">
            <div>
              <h2>{scenario}</h2>
              <span>
                Camo Camera thật · 5 giây chuẩn bị +{" "}
                {protocol?.duration_seconds ??
                  (scenario.startsWith("Tổng hợp ") ? 180 : 30)}{" "}
                giây đo
              </span>
            </div>
            <button onClick={() => cancel()} disabled={state === "complete"}>
              Hủy / Kết thúc sớm
            </button>
          </div>
          <RecognitionNotice
            calibrated={health?.settings?.calibrated}
            status={health?.settings?.calibration_status}
          />
          <div className="diagnostic-view-grid">
            <div className="diagnostic-camera">
              <video
                ref={video}
                autoPlay
                muted
                playsInline
                className="camera-capture-source"
                aria-hidden="true"
                tabIndex={-1}
              />
              <canvas
                ref={surface}
                className="camera-ai-surface"
                aria-label="Camera AI kiểm thử"
              />
            </div>
            <aside
              className="diagnostic-guidance"
              aria-label="Đồng hồ và chỉ dẫn kiểm thử"
            >
              <div className="diagnostic-clock" aria-live="polite">
                <strong>
                  {state === "opening"
                    ? "MỞ CAMERA · CHỜ FRAME THẬT"
                    : state === "preparing"
                      ? "CHUẨN BỊ"
                      : state === "running"
                        ? "BẮT ĐẦU KIỂM THỬ"
                        : "HOÀN THÀNH"}
                </strong>
                {count !== null && <b>{count}</b>}
              </div>
              <div
                className="diagnostic-action"
                role="status"
                aria-label="Chỉ dẫn động tác kiểm thử"
                aria-live="polite"
              >
                {state === "complete" ? (
                  <strong>ĐÃ XONG — CÓ THỂ RỜI VỊ TRÍ</strong>
                ) : (
                  <>
                    <span>
                      {state === "running"
                        ? "THỰC HIỆN TRONG THỜI GIAN KIỂM THỬ"
                        : "CHUẨN BỊ VÀO VỊ TRÍ"}
                    </span>
                    {state === "running" && currentStage ? (
                      <div
                        key={currentStage.id + String(currentStage.transition)}
                      >
                        <strong>
                          {currentStage.transition
                            ? "CHUYỂN ĐỘNG TÁC — KHÔNG TÍNH VÀO TÌNH HUỐNG"
                            : `GIAI ĐOẠN: ${currentStage.label.toUpperCase()}`}
                        </strong>
                        <strong>{currentStage.instruction}</strong>
                        <strong>
                          {currentStage.transition
                            ? `CHUẨN BỊ: ${Math.ceil(currentStage.transition_remaining_seconds)} GIÂY`
                            : `CÒN ${Math.ceil(currentStage.remaining_seconds)} GIÂY TRONG GIAI ĐOẠN`}
                        </strong>
                      </div>
                    ) : (
                      (instructionsFor(scenario) || []).map((line) => (
                        <strong key={line}>{line}</strong>
                      ))
                    )}
                  </>
                )}
              </div>
            </aside>
          </div>
          <div className="diagnostic-metrics">
            Preview {fps?.toFixed(1) ?? "—"} FPS · AI{" "}
            {frame?.ai_fps?.toFixed(1) ?? "—"} frame/s theo thời gian xử lý ·
            Detector {frame?.detector_ms?.toFixed(0) ?? "—"} ms · Frame{" "}
            {frame?.frame_id ?? 0} · Ảnh gốc{" "}
            {frame ? `${frame.width}×${frame.height}` : "—"} · VRAM{" "}
            {frame?.gpu?.used_mb ?? "—"}/{frame?.gpu?.total_mb ?? "—"} MB ·
            Backlog {frame?.backlog ?? 0} · Frame preview bỏ qua:{" "}
            {frame?.sampling?.skipped_preview_frames ?? 0}
          </div>
          <details className="diagnostic-detail">
            <summary>
              Chất lượng crop / alignment / score từng model / quyết định fusion
            </summary>
            <pre>{JSON.stringify(frame?.overlays ?? [], null, 2)}</pre>
          </details>
        </div>
      )}
    </>
  );
}
