# Hướng dẫn Thay đổi Ma trận Cân bằng (Game Balance Matrix)

Trong hệ thống giả lập này, mối quan hệ khắc chế giữa các Lớp nhân vật (Class) được định nghĩa qua một ma trận tỷ lệ thắng kỳ vọng. Hệ thống đánh giá (Fitness Function) sẽ chấm điểm thuật toán dựa trên việc nó có thể chỉnh sửa chỉ số nhân vật sao cho tỷ lệ thắng thực tế tiến gần nhất với ma trận mục tiêu này hay không.

## 1. Vị trí file cần chỉnh sửa
Toàn bộ ma trận mục tiêu nằm trong file:
👉 **`src/simulation/type_chart.py`**

## 2. Cách đọc Ma trận Mục tiêu (Target Matrix)
Ma trận mục tiêu được định nghĩa trong biến `TARGET_MATRIX` (dành cho chế độ cân bằng `symmetric`).

```python
TARGET_MATRIX = {
    "Fighter": {
        "Assassin": 0.60,  # Fighter (hàng) đối đầu với Assassin (cột)
        ...
    },
    # ...
}
```
**Quy tắc đọc:** `Class Hàng (Attacker)` đối đầu với `Class Cột (Defender)` = `Tỷ lệ thắng kỳ vọng`.

Ví dụ: Tại hàng `"Fighter"`, cột `"Assassin"` có giá trị `0.60`.
Điều này có nghĩa là thiết kế game đang mong muốn **Đấu Sĩ (Fighter) có tỷ lệ thắng 60%** khi đối đầu với **Sát Thủ (Assassin)**. Đương nhiên, ở chiều ngược lại (hàng Assassin, cột Fighter), Sát thủ sẽ chỉ có 40% (0.40) cơ hội thắng.

## 3. Cách thay đổi tỷ lệ thắng (Ví dụ 80-20)
Giả sử bạn muốn đổi thiết kế game, làm cho Đấu Sĩ khắc chế cực mạnh Sát Thủ với tỷ lệ thắng 80-20 (Fighter thắng 80%, Assassin thắng 20%).

**Bước 1:** Mở file `src/simulation/type_chart.py`.
**Bước 2:** Tìm đến block `TARGET_MATRIX`.
**Bước 3:** Chỉnh sửa hàng `Fighter`, cột `Assassin` thành `0.80`:
```python
    "Fighter": {
        "Assassin": 0.80, # <--- Đổi từ 0.60 thành 0.80
        "Marksman": 0.55,
        "Mage": 0.40,
        "Fighter": 0.50,
        "Tank": 0.60,
        "Support": 0.65,
    },
```
**Bước 4:** Bắt buộc phải chỉnh sửa tương xứng ở hàng `Assassin`, cột `Fighter` thành `0.20` để tổng bằng 1.0 (100%):
```python
    "Assassin": {
        "Assassin": 0.50,
        "Marksman": 0.65,
        "Mage": 0.60,
        "Fighter": 0.20, # <--- Đổi từ 0.40 thành 0.20
        "Tank": 0.35,
        "Support": 0.70,
    },
```

Lưu file lại. Từ bây giờ, khi chạy thuật toán bằng Launcher (hoặc CLI), hàm Fitness sẽ tự động ép các hệ số chỉ số sao cho thực chiến trong game Đấu Sĩ thắng Sát Thủ với tỷ lệ 80%.

## 4. Chế độ Bất đối xứng (Asymmetric Handicap)
Hệ thống có hỗ trợ chế độ `asymmetric_support_handicap`. Chế độ này dùng biến `ASYMMETRIC_TARGET_MATRIX` (ngay bên dưới `TARGET_MATRIX`). 
Nếu bạn chạy experiment với mode Asymmetric, hãy nhớ thay đổi tỷ lệ ở cả ma trận `ASYMMETRIC_TARGET_MATRIX` này.
