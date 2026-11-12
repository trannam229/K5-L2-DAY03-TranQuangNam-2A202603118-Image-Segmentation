# Báo cáo thực hành Day 3 — điền trực tiếp trong fork của bạn

- Mã học viên theo lớp: 2A202603118
- Ngày / CVAT local: buổi lab 11/11/2026, hoàn thiện sáng 12/11/2026 / CVAT chạy local bằng Docker (`localhost:8080`), 9 task, mỗi task 1 job
- Công cụ đã dùng: CVAT (Brush cho phần tự vẽ, Save/Export); SAM 2.1 (hiera-large) và SAM 3 chạy làm serverless function trong CVAT; SAM 3 theo text prompt và Mask2Former (Swin-L Cityscapes semantic, Swin-L COCO panoptic) chạy offline để tạo nhãn gợi ý rồi đưa vào CVAT qua API; Grounding DINO ở bản thử đầu; Claude Code (trợ lý AI) hỗ trợ chạy pipeline và viết script so sánh `scripts/compare.py`. Việc dùng AI/SAM để gán nhãn theo thông báo đổi quy định của thầy trên lớp.

## 1. Bài đã nộp

| Task | File ZIP đúng tên | Hoàn thành mấy ảnh | Điểm tối đa (coach chấm sau) |
| --- | --- | ---: | ---: |
| easy_semantic | `submissions/easy_semantic.zip` | 3 / 3 | 20 |
| medium_instance | `submissions/medium_instance.zip` | 3 / 3 | 32 |
| hard_panoptic | `submissions/hard_panoptic.zip` | 2 / 2 | 30 |
| cp1_holes | `submissions/cp1_holes.zip` | 1 / 1 | 3 |
| cp2_slice | `submissions/cp2_slice.zip` | 1 / 1 | 3 |
| cp5_occlusion | `submissions/cp5_occlusion.zip` | 1 / 1 | 3 |
| cp3_thin | `submissions/cp3_thin.zip` | 1 / 1 | 3 |
| cp4_curb | `submissions/cp4_curb.zip` | 1 / 1 | 3 |
| cp6_coverage | `submissions/cp6_coverage.zip` | 1 / 1 | 3 |
| **Tổng tối đa** | | | **100** |

Không có lỗi export. Cả 9 ZIP qua `python3 scripts/inspect_submissions.py --dir submissions` (cấu trúc OK). Semantic xuất `Segmentation mask 1.1`, instance/panoptic xuất `COCO 1.0`.

## 2. Một quyết định trước khi dùng gợi ý

- Ảnh, vị trí và object Medium đầu tiên tự vẽ: **không có**. Ở Medium tôi không tự vẽ object trước khi dùng gợi ý; toàn bộ mask Medium đến từ SAM (sau khi thầy cho phép dùng AI/SAM), rồi tôi lọc và kiểm lại. Phần tôi tự vẽ tay bằng Brush là 5 mask ở `easy_semantic`, ảnh `817bca71-00000000`; các mask này được giữ nguyên, mask tự động chỉ phủ phần còn trống.
- Class và quy tắc tôi dùng để chọn biên: với `person` chỉ gán cho **người thật nhìn thấy được**, biên theo phần cơ thể nhìn thấy; không gán hình người in trên poster/biển quảng cáo; không tách một người thành nhiều mask.
- Nếu dùng gợi ý sau đó: ở `medium_instance/000000181542`, SAM 3 đề xuất một `person` trong vùng khoảng x 264–322, y 55–142 là hình người trên tấm poster phía sau. Tôi **bỏ** đề xuất này theo quy tắc trên. Cũng theo quy tắc “người thật, nhìn thấy”, tôi bỏ các `person` nằm gần trọn trong khung xe buýt/ô tô (hành khách thấy lờ mờ qua kính) vì không thấy rõ ranh từng người, và gộp các đề xuất trùng nhau trên cùng một vật (giữ mask điểm cao hơn).
- Nếu không dùng gợi ý: (đã dùng gợi ý, xem trên).

## 3. Một lỗi tôi tìm thấy và sửa

- Task/ảnh/vùng: `easy_semantic`, ảnh `7ee6d192-89e2408b` (đường cao tốc, hai bên là đồi cỏ), dải lề phải mặt đường khoảng x 484–1279, y 396–499.
- Lỗi thuộc loại: sai lớp.
- Bằng chứng tôi nhìn thấy: nhãn gợi ý của Mask2Former tô khoảng 1.300 pixel rải rác ở lề đường thành `sidewalk`. Ảnh là cao tốc, không có vỉa hè hay bó vỉa cho người đi bộ. Script so sánh (chạy sau khi reference được phát) cho thấy class `sidewalk` của ảnh này có IoU 0, kéo mIoU ảnh xuống.
- Quy tắc và hành động sửa: road/sidewalk chọn theo chức năng (vỉa hè là phần tách khỏi lòng đường bằng bó vỉa dành cho người đi bộ), không theo màu. Tôi xóa mask `sidewalk` đó trong CVAT; phần lề để trống thay vì đoán class.
- Sau sửa đã Save và export lại chưa? Đã lưu trong CVAT và export lại `easy_semantic.zip`, kiểm lại bằng `inspect_submissions.py`.

Kết quả tự chạy script (sau khi reference được phát, chạy bằng `scripts/compare.py`): mIoU ảnh `7ee6d192` từ 0.685 lên 0.913; mIoU cả task từ 0.840 lên 0.841. Scorecard ba tier tự đánh giá hiện là 19.6 + 21.7 + 15.9 = **57.2 / 82**. Hai bản trước: 43.5 (SAM 2 + Grounding DINO) và 55.0 (SAM 3); bản này đổi vùng nền sang Mask2Former và thêm bước lọc object. Các thay đổi pipeline chọn theo quy ước của Cityscapes/COCO; reference chỉ dùng để chấm và để tìm lỗi như lỗi trên. Checkpoint không có reference nên chưa có điểm. Ground truth không được đưa vào fork (thư mục `groundtruth/` nằm trong `.gitignore`).

## 4. Ba ca chưa chắc hoặc đã cân nhắc

| Ảnh/vị trí | Hai cách hiểu có thể | Quy tắc/chứng cứ | Quyết định hoặc câu hỏi cho coach |
| --- | --- | --- | --- |
| 1. `easy_semantic/7ee6d192`, hai sườn đồi cỏ thấp hai bên cao tốc | `vegetation` hoặc vùng không thuộc class nào (theo Cityscapes là `terrain`, không có trong `classes.json` của task) | Task chỉ có road/sidewalk/building/vegetation/sky; sườn đồi phủ cỏ và cây bụi | Gán `vegetation` cho phần có cây/cỏ phủ dày; phần đất trống, lẫn cỏ thưa để trống vì không chắc. Hỏi coach: cỏ thấp trên sườn dốc có luôn tính `vegetation` không? |
| 2. `hard_panoptic/000000460147`, mặt đường nhựa có vạch kẻ ở giữa ảnh | `road` (theo chức năng: làn xe đang chạy) hoặc `sidewalk` (theo taxonomy COCO, nhóm `pavement-merged` gom nhiều loại mặt lát) | Có ô tô, xe tải đang chạy trên vùng này và có vạch kẻ làn đường | Giữ `road`. Tôi thấy reference xếp vùng này khác cách của tôi; xin coach xác nhận dùng quy tắc chức năng hay quy ước COCO cho ảnh này. |
| 3. `hard_panoptic/000000460147`, xe chở ô tô ở làn giữa | Một `truck` duy nhất gồm cả hàng hóa, hoặc `truck` cho xe chở và mỗi ô tô trên đó là một `car` riêng | Từng ô tô trên xe chở nhìn thấy rõ hình dạng, kính, bánh | Gán `truck` cho xe chở và mỗi ô tô nhìn thấy là một `car` riêng. Hỏi coach: ô tô đang được chở có nên tính là instance `car` không? |
