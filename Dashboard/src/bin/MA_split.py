## Xóa dòng thừa gây lỗi IndentationError
import pandas as pd
import os
from snownlp import SnowNLP

INPUT_FILE = 'MA_222_Hierarchical_Segmented .csv'
OUTPUT_FILE = os.path.join('source_clean', 'MA', 'split_MA_cleaned_data.csv')

# Đọc file gốc
df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')

output = []
for idx, row in df.iterrows():
    no = row['Sutta_No'] if 'Sutta_No' in row else row[0]
    name = row['Sutta_Name'] if 'Sutta_Name' in row else row[1]
    segment_id = row['Segment_ID'] if 'Segment_ID' in row else row[2]
    content = row['Content'] if 'Content' in row else row[3]
    output.append({
        'No': no,
        'Name': name,
        'Segment_ID': segment_id,
        'Content': content
    })

# Xuất ra file csv
out_df = pd.DataFrame(output, columns=['No', 'Name', 'Segment_ID', 'Content'])
out_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

def extract_bai(segment_id):
    # segment_id dạng MA.x.y.z
    try:
        parts = segment_id.split('.')
        return f'{parts[1]}.{parts[2]}' # chương.bài
    except Exception:
        return None

# Đếm số lượng bài dựa trên sự thay đổi của phần 'bài'
bai_set = set()
for seg_id in out_df['Segment_ID']:
    bai = extract_bai(seg_id)
    if bai:
        bai_set.add(bai)
total_bai = len(bai_set)

print(f'Đã tách xong. File kết quả: {OUTPUT_FILE}')
print(f'Tổng số bài (theo MA.[chương].[bài]): {total_bai}')
if total_bai == 222:
    print('Kết quả đúng: Có 222 bài.')
else:
    print(f'Kết quả sai: Có {total_bai} bài.')
