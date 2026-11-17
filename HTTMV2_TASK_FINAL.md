# TASK CODEX — HTTMV2 | XÂY DỰNG LẠI HỆ THỐNG TÌM KIẾM NGƯỜI BẰNG KHUÔN MẶT

**Mô hình Codex đề xuất:** GPT-6.1 Sol — High  
**Thư mục dự án:** `httmv2` (ĐÃ có source code hệ thống cũ; KHÔNG phải workspace trống).  
**Đầu ra cần thiết:** source code ứng dụng chạy được, cấu hình và script khởi động cần thiết. **KHÔNG tạo DOC/DOCX/PDF, báo cáo Markdown, benchmark report hoặc tài liệu phụ.**

> **PROMPT KHỞI CHẠY — có thể sao chép vào Codex:**  
> Hãy đọc toàn bộ `HTTMV2_TASK_CODEX_HYBRID_AI.md` trong workspace `httmv2` và trực tiếp xây dựng lại ứng dụng end-to-end theo task. Source code hiện có là hệ thống cũ với chất lượng nhận diện không đạt; không cần kế thừa kiến trúc hoặc thuật toán AI cũ. Chỉ giữ các thành phần thực sự tốt và tương thích nếu việc giữ là hợp lý; nếu không, thay mới triệt để. Không tự xóa/reset/thay mới dữ liệu hiện có. Ưu tiên hoàn thiện Live Camo Camera, Video Upload vừa phát vừa phân tích, nhận diện nhiều model, chống nhận nhầm người, bảng thông báo realtime và chế độ kiểm thử trực tiếp có đếm ngược. Hãy viết code, chạy, sửa lỗi và kiểm thử thực tế ở mức môi trường cho phép. **Trước MỖI lần kiểm thử Camo có người tham gia, bắt buộc hỏi tôi: “Bạn đã sẵn sàng kiểm thử chưa?”; chỉ khi tôi xác nhận sẵn sàng mới mở giao diện test với camera thật, đếm ngược chuẩn bị 5 giây rõ ràng, báo BẮT ĐẦU rồi kiểm thử 30 giây; hết thời gian tự đóng cửa sổ test.** Không tự chạy khi chưa xác nhận. Không tạo báo cáo/tài liệu thừa, không tuyên bố đã kiểm chứng những điều chưa đo được. Kết thúc chỉ tóm tắt trong chat các chức năng đã hoạt động, lệnh khởi động và vấn đề còn lại.

---

## 1. Mục tiêu và nguyên tắc không được vi phạm

Xây dựng ứng dụng **tìm một hoặc nhiều người mục tiêu trong môi trường lớp học đông người (có thể 30–50 người xuất hiện)**, tìm và thông báo tức thời khi **đã xác nhận** người thuộc watchlist xuất hiện. Hệ thống phải **từ chối nhận nhầm người ngoài watchlist** (`Unknown`), hạn chế cảnh báo sai và không đổi danh tính vô căn cứ khi người giơ tay, quay đầu, di chuyển hoặc che mặt.

- **Một ảnh đăng ký chính diện hợp lệ là đủ để tạo người và sử dụng nhận diện ngay.** Có thể thêm nhiều ảnh bất kỳ sau đó, không giới hạn cứng số ảnh/người hay số người trong watchlist. Không được yêu cầu người dùng phải chụp 5–10 ảnh mới dùng được.
- **Ưu tiên nhận diện ổn định 0–6 m**, trong đó 3–6 m rất quan trọng; 6–10 m là phạm vi cố gắng theo ảnh thật. Phải kiểm tra cả mặt nhỏ/xa, chuyển động, nhìn trái/phải, che nửa mặt, khẩu trang và người khác đi gần. Không được hứa đạt tỷ lệ chính xác hoặc nhận diện khi không còn đủ thông tin sinh trắc học.
- **Chỉ hiển thị bounding box khuôn mặt được detector phát hiện.** Không vẽ khung toàn thân, khung mặt giả hay gán tên trực tiếp từ tư thế/tay/cơ thể. Tracking người nếu có chỉ là phụ trợ nội bộ và không phải bằng chứng sinh trắc học.
- **Không huấn luyện/fine-tune lại, không dịch vụ AI trả phí, không gửi ảnh/video/embedding người dùng ra ngoài.** Chỉ tải pretrained model/dependency từ nguồn chính thức khi cài đặt; inference phải hoàn toàn cục bộ.
- **Chống nhận nhầm quan trọng ngang với không bỏ sót.** Không được ép một khuôn mặt mới vào một tên đã đăng ký chỉ vì là top-1 similarity.
- **Không tối ưu FPS bằng cách hạ chất lượng đầu vào quá mức hoặc âm thầm bỏ qua người mục tiêu.** Live preview phải mượt và phản hồi nhanh, AI phải đo riêng chất lượng/tốc độ/độ trễ thực tế.
- Hệ thống phải chạy được bằng **React + Vite / FastAPI / SQLite**, trên Windows/NVIDIA RTX 4050 sử dụng CUDA khi khả dụng; có fallback CPU rõ ràng, không hardcode GPU index.

## 2. Nguyên tắc xử lý dự án `httmv2` và dữ liệu

1. **Đọc/kiểm kê source hiện tại trước khi thay đổi.** Xác định frontend/backend, cơ chế camera, database, kho ảnh đăng ký, entrypoint, dependency, lỗi và những phần AI cũ.
2. **Không cần cố tái sử dụng** SCRFD-500M, AdaFace IR-101, YOLO11n, BoT-SORT hoặc logic cũ nếu chất lượng thấp/sai kiến trúc. Được thay pipeline AI và tổ chức code; chỉ giữ code đáng tin cậy khi thực sự hữu ích.
3. **Không tự động xóa, reset, làm mới, seed hoặc thay thế dữ liệu.** Không sửa ảnh đăng ký gốc, tên người, watchlist hay xóa SQLite. Không tạo dữ liệu người mẫu giả để giả vờ hệ thống nhận diện được. Nếu cần schema mới, ưu tiên migration nhỏ không phá dữ liệu; chỉ thêm embedding dẫn xuất riêng theo model từ ảnh gốc nếu thực sự cần. Không coi việc làm lại dữ liệu là mục tiêu.
4. Không xóa các file không rõ chức năng. Với mã cũ không còn dùng, loại bỏ khỏi runtime/import trước; chỉ xóa file source đã xác minh là dư thừa và không phải dữ liệu người dùng. Tránh đưa song song hai pipeline lẫn lộn.
5. Giữ phạm vi vừa đủ để demo bằng máy cá nhân, **không** Docker/cloud/microservices/CI phức tạp trừ khi thật sự cần để chạy. Ưu tiên một script khởi động trên Windows sau bước cài dependency ban đầu.

## 3. Các mô hình AI **chốt để triển khai**

### 3.1. Face detection (hai nhánh, có điều kiện)

- **RetinaFace ResNet50**: detector khuôn mặt chính, lấy face bounding box và 5 landmarks.
- **YOLOv5m-Face**: detector bổ sung/fallback cho vùng hoặc frame khó khi RetinaFace có nguy cơ bỏ sót; không bắt buộc chạy cả hai trên mọi frame.
- Không dùng YOLO person detector như nguồn khẳng định danh tính. Không hiển thị person box.
- Đảm bảo NMS/deduplicate giữa hai detector, chuyển hệ tọa độ đúng khi resize/pad/tile, và kiểm tra landmark hợp lệ trước alignment.
- Với mặt xa: thử **full-resolution detect**, ROI/tiled detect có overlap và hợp nhất detection thay vì luôn resize toàn cảnh xuống 640; chọn cấu hình bằng đo thực tế và giới hạn GPU.

### 3.2. Face recognition (ba nhánh mô hình bổ trợ nhau)

1. **LVFace-B Glint360K** — nhánh nhận diện tổng quát chính. Ưu tiên checkpoint ONNX pretrained chính thức.
2. **InsightFace AntelopeV2 — ResNet100@Glint360K (ArcFace)** — nhánh xác minh bổ sung. Khi lấy model pack, phân biệt rõ *recognition ResNet100* với *detection SCRFD* trong cùng pack; **không** vô tình thay toàn bộ detector RetinaFace bằng SCRFD mà không chủ định.
3. **PETALface — Swin CosFace WebFace12M BRIAR (`swin_cosface_webface12m_briar/model.pt`)** — nhánh hỗ trợ độ phân giải thấp/mặt xa; triển khai đầy đủ cơ chế quality-adaptive của checkpoint, không nạp như Swin thông thường rồi bỏ adapter.

Các model này là **ứng viên kiến trúc đã chọn**, không được tự khẳng định là tổ hợp mạnh nhất thế giới hay luôn vượt model đơn lẻ. Nếu checkpoint cụ thể/giấy phép/dependency không thể dùng trong môi trường hiện tại, phải báo rõ trở ngại và giữ phần còn lại hoạt động; **không âm thầm tráo model sang AdaFace IR-101 cũ** hoặc dùng checkpoint sai.

**Nguồn chính thức cần đọc trước khi cài và lập trình:**
- LVFace: https://github.com/bytedance/LVFace ; trọng số: https://huggingface.co/bytedance-research/LVFace/tree/main/LVFace-B_Glint360K
- InsightFace model zoo: https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md ; repo: https://github.com/deepinsight/insightface
- PETALface: https://github.com/Kartik-3004/PETALface ; checkpoint: https://huggingface.co/kartiknarayan/PETALface
- RetinaFace R50: https://github.com/biubug6/Pytorch_Retinaface
- YOLOv5-Face: https://github.com/deepcam-cn/yolov5-face ; tham khảo ONNX phát hành: https://github.com/clibdev/yolov5-face/releases

**Quan trọng về quyền sử dụng:** phân biệt license code và license pretrained weights; LVFace và một số trọng số InsightFace có giới hạn nghiên cứu phi thương mại. Đây là dự án thử nghiệm học tập; không mặc định quyền triển khai thương mại. Không cần viết file báo cáo về license, chỉ giữ notices bắt buộc nếu dùng mã nguồn/trọng số theo điều khoản của chúng.

### 3.3. Pipeline và fusion bắt buộc đúng về mặt kỹ thuật

- Với **mỗi model**, tôn trọng hoàn toàn thông số input size, RGB/BGR, normalization, alignment, thứ tự landmark, checkpoint format, embedding dimensions, L2 normalization và hậu xử lý chính thức. Viết kiểm thử pipeline từng nhánh, không tráo preprocessing giữa các model.
- **Một ảnh đăng ký:** chạy qua mỗi model khả dụng và lưu embedding **riêng** của model đó. Với nhiều ảnh: quản lý embedding cho từng ảnh/model, có thể tổng hợp trong *cùng không gian model* bằng phương pháp đã đánh giá; **không trộn embedding của model khác nhau**.
- **Không cộng/trung bình cosine thô giữa 3 model.** Hiệu chuẩn score và threshold theo từng model/tình huống từ dữ liệu ground-truth có thật, rồi mới quyết định fusion. Khi chưa có dữ liệu hiệu chuẩn, áp dụng xác nhận thận trọng, hiển thị `Candidate/Unknown`, không tự công bố độ tin cậy phần trăm.
- **Open-set identification:** người không thuộc watchlist phải có trạng thái `Unknown`. Kiểm tra score tuyệt đối, top-1/top-2 margin khi có từ hai ứng viên, chất lượng face và mâu thuẫn giữa các nhánh. Trường hợp chỉ có 1 người trong watchlist vẫn phải có cơ chế từ chối người khác.
- **Adaptive inference:** LVFace xử lý thường xuyên; chỉ kích hoạt nhánh AntelopeV2/PETALface khi cần xác minh, chất lượng kém hoặc mặt xa, dựa trên điều kiện có kiểm chứng. Không phải cả 3 chạy mỗi frame. Có batch/chế độ giới hạn concurrency để không OOM hoặc treo preview trên RTX 4050; đo GPU VRAM và latency thực tế.
- Nếu các nhánh bất đồng về danh tính hoặc điểm chưa đủ chắc chắn: `Unknown`/`Candidate`, **không** tự ý chọn tên rồi cảnh báo.
- **Temporal fusion:** tích lũy bằng chứng nhiều frame hợp lệ trên đúng face track; không tái sử dụng bằng chứng cũ sau mất dấu/ID switch. Một frame score cao bất thường không đủ để chuyển sang danh tính khác.
- **Face tracking:** liên kết các face box bằng chuyển động/vị trí/embedding có điều kiện; người xoay mặt hoặc bị che ngắn hạn có thể duy trì track nội bộ. Khi mất dấu hoặc track crossing không chắc chắn thì bỏ khóa nhận diện, bắt xác minh lại; tuyệt đối không giữ nhãn sai qua người khác. Không bịa bounding box khuôn mặt khi không có detection.
- Khi mặt bị che kín hoặc quay lưng và chưa từng xác nhận danh tính, không thể nhận diện chắc chắn — hệ thống phải từ chối thay vì đoán.

## 4. Frontend: đúng giao diện đã chốt

Sử dụng **React + Vite**, ngôn ngữ tiếng Việt, **nền trắng + xanh dương nhạt**, sidebar trái, thẻ bo góc, phân cấp chữ rõ ràng, phong cách hiện đại; tránh dark mode. Ba màn hình chính:

1. **Camera trực tiếp**
2. **Phân tích video**
3. **Danh sách người cần tìm**

Nếu workspace có các ảnh tham chiếu `01_Live_Camera.png`, `02_Video_Upload.png`, `03_Danh_Sach_Nguoi.png`, cần mở xem trực tiếp và bám sát bố cục. Nếu không có ảnh, không giả vờ đã xem; áp dụng yêu cầu layout trong task. Đây là ứng dụng thật, không dùng screenshot làm UI, không hardcode người/thông báo mô phỏng.

- **Bảng thông báo realtime** hiển thị rõ ngay bên video hoặc vùng tương đương, cập nhật qua WebSocket, không refresh.
- Mỗi bản ghi sự kiện cần tên người, nguồn (Webcam/Camo/Video), thời gian phát hiện, timestamp video nếu có, ảnh bằng chứng mặt/frame thực và trạng thái `Confirmed`.
- **Chỉ vẽ khung quanh khuôn mặt thật**: xanh nếu `Confirmed` đúng watchlist, đỏ nếu chưa xác định. Chỉ hiển thị tên khi đã xác nhận; candidate có thể hiển thị trạng thái đang xác minh nhưng không phát alert.
- Có số đo thật preview FPS, AI FPS, detector FPS/latency, thời gian xác nhận và backlog. Không hiển thị số liệu giả.
- UI không bị block khi backend đang inference, không bị update quá dày gây giật React.

## 5. Live Camera — CHỈ hai loại nguồn

Chỉ hiện hai lựa chọn trong UI:
- **Webcam laptop**.
- **Camo Camera** (camera sau iPhone 13).

Không hiển thị danh sách webcam linh tinh. Xác định camera đúng bằng tên/định danh thiết bị trên máy, không hardcode camera index như `0`, `1` cho mọi máy. Nếu thiếu thiết bị, hiện lỗi hướng dẫn chính xác, không tự chuyển qua camera khác gây khó hiểu.

- Hỗ trợ điện thoại đặt **ngang hoặc dọc hoàn toàn tự động**: đọc metadata/rotation khi có, kiểm tra chiều hình và aspect ratio, giữ đúng tỉ lệ, không kéo giãn, không có selector cho người dùng.
- Capture video, AI inference, tracking và render là các công việc tách rời; queue có giới hạn và timestamp/frame ID, tránh backlog vô hạn. Live có thể ưu tiên frame mới để giảm độ trễ nhưng phải có cơ chế theo dõi/kiểm tra lại, đo rõ sự đánh đổi và **không làm mất cơ hội phát hiện người** một cách âm thầm.
- Tận dụng ảnh gốc để phát hiện mặt và crop; không thu nhỏ toàn khung hình khiến mặt xa vốn nhìn thấy rõ trở thành vài pixel.
- Một hoặc nhiều người mục tiêu: xử lý đồng thời với những người khác trong lớp. Không giới hạn cứng watchlist 5 người. Không vẽ person box khi tay di chuyển.
- **Realtime alert:** khi `Confirmed` xuất hiện, backend lập tức push WebSocket; UI cập nhật tên/ảnh/time. Deduplicate theo danh tính/lượt xuất hiện, chống việc mất mặt 1–2 giây tạo hàng loạt event.
- Ngắt camera an toàn, giải phóng handle khi chuyển nguồn/tắt Live; không để Camo bị chiếm cổng khiến test không mở được.

## 6. Video Upload — PHẢI vừa phát vừa phân tích

Người dùng chọn video local. Giao diện **phát video trong lúc AI đang phân tích** đúng timeline, vẽ bounding box mặt và cập nhật bảng thông báo realtime ngay khi có kết quả, chứ không chỉ chạy phân tích ngầm rồi trả kết quả sau.

- Có **Play / Pause / Seek**, thời lượng và thanh tiến độ, tốc độ phân tích, trạng thái xử lý, người đã tìm thấy.
- Đồng bộ playback và inference bằng **video PTS/timestamp + frame ID + analysis generation ID**. Không gắn nhãn của frame cũ lên frame mới.
- Pause: xử lý và overlay nhất quán; Seek: hủy/cách ly kết quả cũ, chuyển đúng vị trí, không phát alert sai do callback trễ; Resume đồng bộ lại.
- Nếu GPU xử lý không kịp thời gian phát, ưu tiên điều tiết playback hoặc hàng đợi có kiểm soát để người dùng xem video đã phân tích, **không âm thầm nhảy qua phần lớn video**. Không ép backend phải quét theo wall-clock 1× nếu có thể xử lý tốt hơn mà vẫn đồng bộ.
- Tự xử lý video ngang/dọc và metadata orientation; giữ tỉ lệ ảnh.
- Tránh thông báo trùng một danh tính trong cùng một lượt; một người xuất hiện tại nhiều thời điểm có thể có các lượt riêng khi thật sự rời rồi quay lại.
- Khởi động lại app: trang Upload ở trạng thái trống, không tự hiển thị video cũ. Không tự xóa dữ liệu đăng ký hoặc database.

## 7. Danh sách người và quản lý đăng ký

- Trang danh sách người có card/thumbnail, tên, số ảnh, checkbox chọn watchlist, nút quản lý, nút thêm người.
- Form tạo/sửa người; tối thiểu **1 ảnh chính diện** để đăng ký. Ảnh phải có khuôn mặt hợp lệ; nếu nhiều mặt, yêu cầu chọn khuôn mặt hoặc báo lỗi rõ ràng, không đăng ký nhầm.
- Thêm nhiều ảnh, xem và xóa từng ảnh; xóa ảnh thì cập nhật embedding liên quan, không xóa toàn bộ người vô ý.
- Xử lý embedding khi model version thay đổi bằng versioned metadata và tái trích xuất từ ảnh gốc **khi cần**, không thay đổi ảnh gốc, không tái tạo dữ liệu thử nghiệm giả.
- Chọn nhiều người cùng lúc để tìm; một người đã lưu không nhất thiết thuộc watchlist đang hoạt động.
- SQLite ở máy local; kiểm tra schema hiện có, không reset hoặc tạo database mới nếu không cần thiết.

## 8. Live AI Diagnostic — kiểm thử trực tiếp bằng Camo khi sửa hệ thống

**Đây là tính năng bắt buộc, không phải tool báo cáo.** Người dùng cần biết chính xác lúc nào kiểm thử bắt đầu và kết thúc để đứng trước iPhone 13 thực hiện hành vi.

- Trong ứng dụng có nút **Kiểm thử AI bằng Camo Camera** (trang Live hoặc khu vực cấu hình gọn). **Không được tự động chạy kiểm thử ngay khi bật app hoặc khi Codex vừa hoàn tất sửa code.**
- **Cổng xác nhận sẵn sàng là bắt buộc trước MỖI lượt test:** Khi cần kiểm thử trực tiếp, Codex phải hỏi rõ trong chat **“Bạn đã sẵn sàng kiểm thử bằng Camo Camera chưa?”** và dừng tại bước thao tác vật lý cho đến khi người dùng đồng ý. Trên ứng dụng cũng phải hiện lựa chọn rõ **“Bạn đã sẵn sàng?”** với hai nút **“Tôi đã sẵn sàng – Bắt đầu”** / **“Chưa sẵn sàng”**. Nếu chưa sẵn sàng thì **không bắt đầu đếm ngược, không ghi dữ liệu test**; người dùng có thể bắt đầu sau mà không cần khởi động lại app.
- Trước lúc người dùng xác nhận sẵn sàng, cho chọn **tình huống kiểm thử**, **danh tính mục tiêu dự kiến** và mục đích (`có mặt`/`không có mặt`) để có ground truth tối thiểu; không tự coi dự đoán của model là nhãn thật.
- **Chỉ sau khi bấm “Tôi đã sẵn sàng – Bắt đầu”**, mới mở **overlay/modal toàn màn hình hoặc cửa sổ kiểm thử** theo theme trắng–xanh nhạt; hiển thị **video Camo thật** và tên tình huống (chính diện 1 m, quay trái/phải, che miệng/khẩu trang, giơ tay/di chuyển, 3 m, 6 m, nhóm đông, người không nằm trong watchlist). Người dùng phải nhìn được hình camera ngay trên cửa sổ này.
- **Đợi camera sẵn sàng và có frame thật**, rồi hiển thị **đếm ngược CHUẨN BỊ 5 → 4 → 3 → 2 → 1** ở vị trí lớn, tương phản cao, dễ thấy khi đang đứng trước điện thoại. Có thể hiện thêm chỉ dẫn hành động của lượt test. Không để đồng hồ chạy khi Camo mất kết nối, chưa có frame hay model/backend chưa sẵn sàng; hiển thị lỗi rõ và cho thử lại.
- Khi hết 5 giây chuẩn bị, hiển thị nổi bật **“BẮT ĐẦU KIỂM THỬ”** rồi chuyển sang **đồng hồ đếm ngược 30 → 0 giây** ngay trên giao diện camera. Định nghĩa rành mạch trạng thái `CHỜ SẴN SÀNG → MỞ CAMERA → CHUẨN BỊ (5s) → ĐANG KIỂM THỬ (30s) → HOÀN THÀNH`. **Chỉ ghi nhận kết quả của 30 giây kiểm thử**, không tính 5 giây chuẩn bị. Đồng hồ dùng monotonic clock, độc lập FPS AI.
- Có nút **Hủy/Kết thúc sớm** và xử lý đóng cửa sổ/camera an toàn ở mọi trạng thái; không tự bấm xác nhận thay người dùng hoặc âm thầm khởi chạy bài test kế tiếp.
- Video live và box/nhãn/trạng thái, thông số GPU, fps, frame count, detection/crop quality, alignment quality, per-model score, similarity candidate và quyết định fusion có thể xem được ở chế độ diagnostic; giao diện không làm nhiễu màn hình dùng thường.
- **Khi đồng hồ về 0, hiện “HOÀN THÀNH” rõ ràng, tự ngừng thu thập dữ liệu, giải phóng camera và tự đóng overlay/cửa sổ kiểm thử**, không đóng toàn bộ app; tóm tắt số đo trong ứng dụng chính nếu cần. Không yêu cầu người dùng bấm dừng, không tự khởi chạy vòng kiểm thử mới.
- Lưu **chỉ số tối thiểu cần cho debug** bằng SQLite hoặc log ngắn có kiểm soát (timestamp, kích thước mặt trước/sau resize, detector nào tìm thấy, chất lượng ảnh, lỗi alignment, model scores, track_id, identity_id, thời gian, sự kiện). Không bắt buộc ghi mọi video/ảnh người dùng và không tạo report DOC/MD.
- Codex phải dùng kết quả này để **xác định đúng lỗi ở 1 m trước**, rồi mới tối ưu 3–6 m và 6–10 m. Không chỉnh threshold tùy tiện, không thêm model rồi tuyên bố xong. Nếu Codex không thể trực tiếp nhìn camera vật lý, hãy đưa ứng dụng vào trạng thái sẵn sàng test và yêu cầu người dùng thực hiện lần test trong UI, sau đó đọc log/metrics từ kết quả.
- Không chạy một stress test lâu hay tự bật camera âm thầm. Chỉ khởi chạy test từ thao tác rõ ràng của người dùng.

## 9. Hệ thống sự kiện, tiêu chuẩn xác nhận và chống nhận nhầm

Thiết kế state machine có phân biệt **Unknown → Candidate → Confirmed → (Mất bằng chứng/đóng track)**.

- `Unknown`: người mới, người không trong watchlist, hoặc thông tin mặt thiếu; không cảnh báo.
- `Candidate`: một hoặc nhiều model gợi ý tên nhưng thiếu điều kiện; không cảnh báo.
- `Confirmed`: có đủ bằng chứng nhất quán sau hiệu chuẩn + temporal verification; chỉ lúc này gửi sự kiện.
- Đã xác nhận A thì **không thể lập tức chuyển sang B** do giơ tay, quay đầu, che mặt, crop sai hoặc ID switch. Khi xung đột, khóa nhãn cũ khỏi việc tuyên bố tên mới và xác minh lại; nếu track không còn đáng tin cậy thì về Unknown/Candidate.
- Chống giả dương: người không đăng ký, người có khuôn mặt tương tự và thời điểm **không có mục tiêu trong khung hình** là bài test bắt buộc.
- Mỗi event có ID ổn định, timestamp, source, video PTS khi có, identity ID, optional evidence frame, confirmation reason; cơ chế dedup theo **identity + continuous appearance/session**, không chỉ timeout 5 giây máy móc.
- WebSocket chỉ thông báo detection xác nhận; **không phát sự kiện mất dấu** vào bảng thông báo. Candidate hiển thị trong overlay nếu cần, không lưu thành "đã tìm thấy".

## 10. Tốc độ, CUDA và ổn định hệ thống

- Bắt đầu bằng kiểm tra CUDA khả dụng, RAM/VRAM và khả năng load từng model (không giả định RTX 4050 luôn có cùng dung lượng VRAM). Không hardcode GPU ID hay cài đổi driver bừa bãi.
- Quản lý model sessions, cấp phát GPU, preprocess, batch và queues có giới hạn; không load lại checkpoint mỗi frame, không khởi tạo model cho từng request.
- Dùng ONNX Runtime CUDA/PyTorch CUDA phù hợp phiên bản; đo và xử lý xung đột dependencies. Có thể dùng TensorRT khi thực sự tăng hiệu năng và không làm méo kết quả; không bắt buộc nếu gây phức tạp không cần.
- Tối ưu có kiểm soát: profile từng tầng capture, decode, detect, landmark, crop/align, recognition từng model, fusion, tracking, render, WebSocket; chỉ sửa nút thắt đo được.
- Ưu tiên ổn định hệ thống khi mở camera lâu: không rò memory, không trễ dồn, không phát event từ frame quá cũ, không block main thread React.
- Nếu không đủ VRAM cho 3 model cùng thường trú, hãy thiết kế staged/lazy loading hoặc giới hạn inference hợp lý; không âm thầm bỏ một nhánh rồi tuyên bố ensemble đầy đủ. Nêu tình trạng từng model ngay trong giao diện diagnostic/console.

## 11. Quy trình thực thi Codex — bắt buộc đi đến code chạy được

**Không dừng ở bản kế hoạch.** Làm tuần tự nhưng tiếp tục triển khai ngay trong workspace:

1. **Inspect:** đọc source, dependency, SQLite và checkpoint cũ trong `httmv2` để biết ảnh hưởng; ghi nhận ngắn trong chat, **không tạo báo cáo file**.
2. **Replace architecture:** thống nhất API/DTO/events, bỏ pipeline AI cũ khỏi đường chạy chính; không phá dữ liệu.
3. **Implement inference:** tải pretrained chính thức, đảm bảo checksum/format theo nguồn, test load và forward từng model, verify preprocessing, alignment, embedding, detection và CUDA thật.
4. **Implement registration:** tạo người bằng 1 ảnh, thêm/xóa ảnh, gallery embedding tách model, watchlist.
5. **Implement Live:** chỉ webcam + Camo, tự orientation, face-only boxes, asynchronous render, realtime WebSocket và bảng cảnh báo.
6. **Implement Video Upload:** vừa phát vừa phân tích có timeline/seek/pause và thông báo đúng mốc.
7. **Implement diagnostic UI:** bắt buộc hỏi người dùng đã sẵn sàng chưa trước từng lượt; nút xác nhận trong app → mở cửa sổ Camo thật → đếm ngược 5 giây chuẩn bị → báo BẮT ĐẦU → đếm ngược 30 giây test → tự đóng, lưu metric ngắn gọn; sửa lỗi đo được.
8. **Verify & fix:** chạy unit/integration test cần thiết và test end-to-end; sửa lỗi thực tế; không khẳng định kết quả không có bằng chứng.

**Bắt buộc kiểm tra:**
- Tạo 1 người bằng 1 ảnh, embeddings của từng model không lỗi; thêm/xóa ảnh và watchlist cập nhật.
- Nhiều người không nằm trong watchlist xuất hiện: **không cảnh báo nhầm**; watchlist một người vẫn có `Unknown`.
- Chuyển động tay/quay đầu không được đổi A→B do lỗi liên kết ID.
- Mặt không phát hiện: không dựng face box giả. Không vẽ khung toàn thân.
- Camo camera ở 1 m: test chính diện, quay trái/phải, che miệng và di chuyển; đo ảnh gốc so với ảnh crop thực tế.
- Kiểm tra 3–6 m và video lớp đông khi người dùng cung cấp tình huống thực; không tự bịa dữ liệu/độ chính xác.
- Video Upload Play/Pause/Seek đúng, overlay và thông báo không lệch timestamp.
- WebSocket dedup và tự kết nối lại phù hợp; UI hiển thị FPS/latency thực đo.
- Kiểm thử Camo **không tự khởi chạy**; Codex hỏi người dùng sẵn sàng trước mỗi lượt, UI phải có hai lựa chọn sẵn sàng/chưa sẵn sàng. Chỉ khi xác nhận, cửa sổ camera thật xuất hiện và chuẩn bị đếm ngược 5 → 1 rõ ràng; hiển thị BẮT ĐẦU và đồng hồ 30 → 0; không đếm khi camera chưa có frame.
- Kết thúc diagnostic tự thông báo HOÀN THÀNH, đóng màn hình kiểm thử và giải phóng Camo, không đóng app; nếu người dùng hủy thì dừng an toàn.
- Startup từ máy Windows qua script đơn giản, cài đặt chỉ một lần; ghi lệnh chạy trong chat khi hoàn thành.

### Tiêu chí đánh giá định lượng (phải đo, không tự giả định)

- **Recall của người mục tiêu** trên từng điều kiện (1 m, 3 m, 6 m, xa hơn, quay đầu, khẩu trang), với nhãn xác thực.
- **False alerts/hour** và **false identifications** khi nhiều người ngoài watchlist xuất hiện.
- **Time-to-confirm/alert P50, P95**, kể từ frame đầu tiên có mặt đủ điều kiện; không tính từ lúc người quay lưng hoàn toàn.
- **Identity switches**, duplicate alerts, detection recall, preview FPS, AI FPS, processing latency, VRAM.
- Hiển thị rõ `chưa kiểm chứng` khi thiếu dữ liệu/thiết bị; không đưa 90%, 95%, 99% như kết quả thật nếu chưa đo.

## 12. Điều cấm và quy tắc bàn giao

- **Không viết bất kỳ báo cáo DOCX, PDF, Markdown, CSV thống kê hoặc tài liệu tổng kết ngoài yêu cầu.** Các file mã nguồn, cấu hình, script, migration, model manifest/notice theo license và test code tối thiểu phục vụ chạy app thì được phép.
- Không xây chức năng ngoài phạm vi; không tạo data giả để demo; không âm thầm reset DB hay thay ảnh/embedding gốc khi không cần.
- Không tự động mở webcam/Camo để quay khi người dùng chưa khởi động Live/test. Đặc biệt **không tự khởi chạy kiểm thử vật lý khi người dùng chưa xác nhận “Tôi đã sẵn sàng”**; mỗi lượt đều có cổng xác nhận riêng.
- Không bắt buộc người dùng cung cấp nhiều ảnh đăng ký; không dùng cloud inference/training.
- Không tự nhận "đã hoạt động chính xác" chỉ dựa trên load model/smoke test hoặc video không có ground truth.
- Tránh hàng loạt file task con, logs dump lớn, notebooks và báo cáo thử nghiệm; dùng code/console/SQLite đủ để debug.
- Không lặp đi lặp lại yêu cầu xác nhận nếu đã rõ; nếu phát sinh blocker vật lý (Camo chưa kết nối, chưa có người đứng test) thì hoàn thành tất cả phần còn lại, báo chính xác bước cần người dùng thực hiện.

### Phản hồi duy nhất khi hoàn thành (trong chat, không tạo file)

- Các màn hình/chức năng đã thực sự chạy được.
- Model nào đã **thực sự load và inference**, model nào chưa và lý do.
- Kết quả kiểm thử thực đo có thể xác minh, điều gì chưa test với Camo/người thật.
- Cách khởi động ngắn gọn.
- Các lỗi còn tồn tại cần xử lý tiếp, không che giấu.

**Hãy triển khai trực tiếp trong `httmv2`, ưu tiên sửa nguyên nhân gây nhận diện kém ở 1 m rồi mới kết luận chất lượng khoảng cách xa.**
