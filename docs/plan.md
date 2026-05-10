Cảm ơn thông tin. Với combination này — đa dạng style (sách + diagram online), không có dataset sẵn, compute thoải mái — tôi có recommendation rõ ràng cho bạn.

## Chiến lược tổng thể

Điểm mấu chốt của case này: **bạn không có dataset nhưng phải xử lý nhiều style khác nhau**. Đây chính xác là tình huống mà **synthetic data generation** tỏa sáng. Bạn có thể tạo hàng triệu mẫu training với label hoàn hảo, không tốn công annotate.

Pipeline tôi khuyến nghị, theo thứ tự ưu tiên:

## Bước 1: Synthetic Dataset Generation (quan trọng nhất)

Đây là nền tảng quyết định mọi thứ. Bạn cần render bàn cờ 2D với độ đa dạng cao để model generalize sang sách thật và diagram online.

**Nguồn piece styles:** Lichess có sẵn ~30 piece sets miễn phí dưới dạng SVG (cburnett, merida, alpha, pirouetti, chessnut, leipzig, fantasy, spatial, reillycraig, riohacha, kosal, gioco, tatiana, maestro, fresca, cardinal, california, pixel, governor, dubrovny, icpieces, mpchess, kiwen-suwi, horsey, anarcandy, caliente, monarchy, companion, staunty, governor). Chess.com có style riêng. python-chess có function `chess.svg.board()` render được nhiều style.

**Variation cần tạo:**
- 30+ piece styles (SVG → PNG)
- Nhiều màu ô sáng/tối: trắng-đen, nâu-be, xanh-trắng, kem-nâu, etc.
- Có/không có border, có/không có coordinates (a-h, 1-8)
- Font coordinates khác nhau
- Kích thước render khác nhau (256px → 1024px)
- Background paper texture (mô phỏng giấy sách): vàng nhạt, xám, có vết ố
- JPEG compression artifacts (mô phỏng scan/chụp)
- Slight blur, noise (Gaussian, salt-pepper)
- Slight rotation (±2°), perspective skew nhẹ
- Print halftone pattern (cho sách in)

**Số lượng:** Khoảng **50,000-100,000 ảnh bàn cờ tổng hợp**, mỗi ảnh có FEN ground truth từ random positions hoặc database PGN. Mỗi ảnh sinh ra 64 ô labeled, vậy là 3.2M-6.4M training samples cho piece classifier.

**Code framework gợi ý:**

```python
import chess
import chess.svg
import cairosvg
from PIL import Image, ImageFilter
import random

# 1. Sample random FEN từ PGN database hoặc sinh ngẫu nhiên hợp lệ
# 2. Render với chess.svg.board() + custom CSS cho colors
# 3. Convert SVG → PNG với cairosvg
# 4. Apply augmentations: blur, JPEG, paper texture, noise
# 5. Save ảnh + FEN label + per-square labels
```

## Bước 2: Board Detection

Với synthetic data có sẵn, bạn có 2 lựa chọn:

**Option A — Classical (đơn giản, đủ tốt cho 2D):**
- Edge detection (Canny) → Hough lines → tìm grid 8×8.
- Hoặc contour detection tìm ô vuông lớn nhất.
- Rectify bằng homography từ 4 góc.

**Option B — YOLO (robust hơn, khuyến nghị):**
- Train YOLOv8 hoặc YOLOv11 trên synthetic data.
- Tạo synthetic "trang sách" bằng cách paste bàn cờ render lên background giấy + text giả + diagram khác.
- Detect bounding box bàn cờ, sau đó refine bằng Hough lines để có pixel-perfect corners.

Tôi khuyến nghị Option B vì sách thật có nhiều bàn cờ trên một trang, có chú thích, có ô khác (như diagram thế cờ nhỏ). YOLO xử lý chuyện này tốt hơn classical methods.

## Bước 3: Square Segmentation

Sau khi có 4 góc bàn cờ chính xác:
- Apply perspective transform để rectify thành hình vuông.
- Chia đều 8×8 = 64 ô.
- Mỗi ô crop ra với padding nhỏ (5-10%) để không miss piece edge.

Đơn giản, không cần ML.

## Bước 4: Piece Classification (giai đoạn quyết định accuracy)

Với compute thoải mái, dùng architecture mạnh:

**Khuyến nghị: EfficientNetV2-S hoặc ConvNeXt-Tiny**
- Input: 96×96 hoặc 128×128 per square
- Output: 13 classes (6 trắng + 6 đen + empty)
- Training: ~20-30 epochs trên synthetic data
- Optimizer: AdamW, cosine schedule
- Augmentation mạnh: RandAugment, MixUp, CutMix

Với synthetic data đa dạng đủ, có thể đạt **>99.5% per-square accuracy**, tương đương **>90% per-board accuracy** (vì 1 board = 64 ô độc lập, sai 1 ô là sai cả board).

**Tip nâng cao:** Dùng **2-stage classifier**:
- Stage 1: Empty vs Occupied (binary, ~99.9% accuracy dễ đạt)
- Stage 2: 12-class piece classifier chỉ chạy trên ô occupied

Cách này hiệu quả vì ~50-70% ô là trống, và phân biệt empty/occupied dễ hơn nhiều phân loại quân.

## Bước 5: Post-processing với Chess Constraints (boost accuracy đáng kể)

Đây là bước thường được bỏ qua nhưng cực kỳ hiệu quả:

**Hard constraints — bắt buộc:**
- Đúng 1 vua trắng, 1 vua đen
- Tốt không ở rank 1 hoặc rank 8
- Tối đa 8 tốt mỗi bên
- Tổng quân mỗi bên ≤ 16

**Soft constraints (kiểm tra hợp lệ về luật):**
- Số bishop sáng/tối ô hợp lý
- Cấu trúc tốt khả thi (không có quá nhiều tốt cùng cột không khớp với capture)

**Cách áp dụng:** Lấy top-3 prediction (với softmax probabilities) cho mỗi ô. Giải bài toán tối ưu: tìm assignment 64 ô sao cho thỏa mãn hard constraints và maximize tổng log-likelihood. Có thể giải bằng:
- **ILP (Integer Linear Programming)** với PuLP hoặc OR-Tools — exact solution
- **Beam search** trên ô-by-ô — gần đúng, nhanh hơn

Bước này thường tăng accuracy thêm 2-5% và loại bỏ những lỗi "ngớ ngẩn" như 3 vua, 9 tốt.

## Bước 6: Orientation Detection

Bàn cờ có thể bị xoay 180° (đặc biệt trong sách phân tích từ phía Đen). Cần xác định hướng:

- Detect coordinate labels (a-h, 1-8) bằng OCR nếu có.
- Nếu không có labels: dùng heuristic — vua trắng thường ở nửa dưới trong opening/middlegame, hoặc check ô a1 phải là ô tối.
- Hoặc: train một small CNN classify orientation (4 lớp: 0°, 90°, 180°, 270°).

## FEN Generation

Sau khi có 8×8 piece grid + orientation đúng:

```python
import chess

board = chess.Board.empty()
for rank in range(8):
    for file in range(8):
        piece = predicted_grid[rank][file]
        if piece != 'empty':
            board.set_piece_at(chess.square(file, 7-rank), 
                              chess.Piece.from_symbol(piece))
fen = board.fen()
```

Một lưu ý: từ ảnh bạn không thể suy ra ai đi, castling rights, en passant. Nên FEN sẽ ở dạng position-only, các trường khác để default.

## Tech stack cụ thể

```
- python-chess: render board, FEN handling
- cairosvg / Pillow: SVG to PNG
- PyTorch + timm: piece classifier (có sẵn EfficientNetV2, ConvNeXt)
- ultralytics: YOLOv8/v11 cho board detection
- OpenCV: Hough lines, perspective transform
- albumentations: augmentation pipeline
- PuLP hoặc OR-Tools: constraint solving
```

## Lộ trình triển khai

1. **Tuần 1-2:** Build synthetic data generator. Render 50K bàn cờ với đa dạng style.
2. **Tuần 3:** Train piece classifier baseline trên synthetic. Đánh giá trên một set diagram online thật.
3. **Tuần 4:** Train YOLO board detector. Build full pipeline end-to-end.
4. **Tuần 5:** Annotate ~100-200 ảnh thật từ sách + diagram online làm test set. Đo accuracy.
5. **Tuần 6+:** Iterate — thêm augmentation/style nào còn thiếu, fine-tune trên một ít ảnh thật nếu cần.

## Reference repos đáng study

- **chesscog** (github.com/georg-wolflein/chesscog) — có paper, code đầy đủ, tuy thiên về 3D nhưng kiến trúc CNN + constraint solving rất giống cái bạn cần.
- **LiveChess2FEN** — pipeline production-ready.
- **Neural Chessboard** của Maciej Czyzewski — board detection robust.

Bạn muốn tôi đi sâu vào phần nào trước? Ví dụ tôi có thể viết code cụ thể cho synthetic data generator, hoặc training script cho piece classifier, hoặc constraint solver post-processing.