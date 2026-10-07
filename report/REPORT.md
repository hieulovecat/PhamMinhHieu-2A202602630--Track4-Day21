# Báo cáo Day 6: Phát hiện vật cản từ LiDAR không dùng deep learning (voxel → RANSAC → DBSCAN)

> Thay **mọi** ô có chữ ĐIỀN nằm trong ngoặc vuông bằng nội dung của bạn, xoá luôn cả dấu ngoặc vuông. Lệnh `python tools/check_submission.py` sẽ báo FAIL nếu còn sót bất kỳ chỗ nào.

- **Họ tên:** Phạm Minh Hiếu
- **MSSV:** 2A202602630
- **Lớp:** [ĐIỀN]
- **Link repo:** https://github.com/hieulovecat/PhamMinhHieu-2A202602630--Track4-Day21
- **Topic:** D — Robot/drone obstacle
- **Dataset:** data/kitti_mini (thí nghiệm chính), data/synthetic (debug)
- **Các frame đã dùng:** toàn bộ 20 frame của kitti_mini (000001 … 000061); demo chính 000011

> Hãy viết ngắn: mỗi mục từ 3 đến 8 dòng, ưu tiên số liệu và hình ảnh.

## 1. Claim

*(Claim nháp CP1, sẽ cập nhật bằng số liệu ở CP3)*: Trên 20 frame KITTI, tăng `distance_threshold` của RANSAC ground từ 0.1 m lên 0.3 m làm recall phát hiện Pedestrian/Cyclist (vật thấp, mảnh) giảm ít nhất 10 điểm %, trong khi recall của Car gần như không đổi.

## 2. Evidence

Bảng hoặc plot số liệu, kèm ảnh/video demo. Ghi rõ đường dẫn file trong `results/`.

| Cấu hình / mức perturb | Metric 1 | Metric 2 | Ghi chú |
|---|---|---|---|
| [ĐIỀN] | | | |

![demo](../results/figures/[ĐIỀN].png)

## 3. Failure case

Nêu khi nào hệ thống hoặc phương pháp fail, vì sao fail, và liên hệ tới lớp nào trong 6 lớp debug: I/O, Geometry, Time, Preprocess, Model, Metric.

![failure](../results/figures/fail_[ĐIỀN].png)

[ĐIỀN]

## 4. Khuyến nghị nếu triển khai thật

Use-case cụ thể (ADAS / robot / drone), trade-off và bước tiếp theo.

[ĐIỀN]

## 5. Cách chạy lại

Các lệnh tái tạo lại toàn bộ kết quả từ repo sạch.

```bash
[ĐIỀN]
```

## 6. Khai báo sử dụng AI

Ghi rõ đã dùng công cụ AI nào, dùng vào việc gì, và bạn đã tự kiểm chứng kết quả đó bằng cách nào. Nếu không dùng AI, ghi "Không sử dụng". Xem quy định ở `RULES.md` mục 2.

| Công cụ | Dùng cho việc gì | Bạn đã kiểm chứng thế nào |
|---|---|---|
| [ĐIỀN] | | |
