📊 1. Phase3_Sutta_Summary.csv (Bản Đồ Toàn Cảnh)
Vai trò: Tóm tắt dữ liệu của toàn bộ 92 cặp bài kinh.
Giá trị: Chấm điểm mức độ biến dạng (Divergence Index) và thống kê số lượng đoạn bị thêm/bớt cho từng bài kinh. Giúp học giả có cái nhìn định lượng về bài kinh nào được bảo tồn tốt nhất, bài nào bị xào nấu nặng nề nhất.
📖 2. Phase3_Pair_Divergence.csv (Hồ Sơ Đối Chiếu Chi Tiết)
Vai trò: File cốt lõi chứa dữ liệu đối chiếu ở cấp độ câu/đoạn (Segment-level) với tỷ lệ ánh xạ 1-1 tinh khiết (TRUE_PAIR).
Giá trị: Hệ thống dán nhãn cụ thể cho TỪNG CẶP CÂU (SEMANTIC_DRIFT, OMISSION, PARALLEL...) dựa trên điểm tương đồng ngữ nghĩa và độ lệch chiều dài. Đây là mạch máu cung cấp dữ liệu hiển thị song ngữ trên Web Dashboard.
🗜️ 3. Phase3_Formulaic_Omission.csv (Báo Cáo "Siêu Nén" Văn Bản)
Vai trò: Bắt quả tang thói quen cắt gọt các đoạn lặp đi lặp lại (trùng tụng) của dịch giả Trung Hoa cổ đại.
Giá trị: Khi 1 câu Hán tự cõng từ 6 câu Pali trở lên (Fanout > 5), hệ thống tự động tính toán Tỷ lệ nén (Compression Ratio). Đây là minh chứng định lượng toán học xuất sắc chứng minh cho luận điểm nghiên cứu văn bản của HT. Thích Minh Châu và Tỳ-kheo Anālayo.
🔍 4. Phase3_Key_Findings.csv (Danh Sách Dị Bản Nghiêm Trọng)
Vai trò: "Bảng phong thần" chắt lọc 100 đoạn văn bị biến dạng nghiêm trọng nhất trên toàn bộ tập dữ liệu (Độ tin cậy cao, điểm sai khác lớn).
Giá trị: "Dọn sẵn" những phát hiện đột phá (VD: bản Hán chèn thêm bài kệ mà Pali không có). Đây là nguồn tư liệu đắt giá để viết các bài báo khoa học chuyên sâu (Case Studies).
🔤 5. Phase2_Dictionary.csv (Từ Điển Tự Động Học)
Vai trò: Bộ từ điển thuật ngữ Pali - Hán được AI tự động trích xuất bằng thuật toán thống kê PMI (Pointwise Mutual Information).
Giá trị: Chứng minh AI có khả năng tự động "học" tiếng cổ đại thông qua tần suất xuất hiện song song (Co-occurrence) mà không cần đến từ điển của con người. Có thể ứng dụng để phát triển công cụ dịch thuật Phật giáo.