# Huấn luyện và đối chiếu Word2Vec Pali-Hán thuần Python
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
import os

# Đọc dữ liệu
pali_df = pd.read_csv('final/Hantu_MA_222_Final_Perfect.csv')
if 'Cleaned_Content' in pali_df.columns:
    pali_corpus = pali_df['Cleaned_Content'].astype(str).tolist()
elif 'Content' in pali_df.columns:
    print("File Pali không có cột 'Cleaned_Content', sẽ dùng cột 'Content' thay thế.")
    pali_corpus = pali_df['Content'].astype(str).tolist()
else:
    raise ValueError("File Pali thiếu cả hai cột 'Cleaned_Content' và 'Content'")
pali_sentences = [simple_preprocess(s) for s in pali_corpus]

han_path = 'final/Han_Corpus_Clean.csv'
if os.path.exists(han_path):
    han_df = pd.read_csv(han_path)
    if 'Cleaned_Content' not in han_df.columns:
        raise ValueError('Thiếu cột Cleaned_Content trong file Hán')
    han_sentences = [simple_preprocess(s) for s in han_df['Cleaned_Content'].astype(str)]
    han_data_ready = True
else:
    print('Không tìm thấy file Hán. Chỉ xử lý Pali.')
    han_sentences = []
    han_data_ready = False

# Huấn luyện mô hình
pali_model = Word2Vec(sentences=pali_sentences, vector_size=100, window=5, min_count=1, workers=4)
pali_model.save('pali_word2vec.model')
if han_data_ready and han_sentences:
    han_model = Word2Vec(sentences=han_sentences, vector_size=100, window=5, min_count=1, workers=4)
    han_model.save('han_word2vec.model')
else:
    han_model = None

# Đối chiếu ngữ nghĩa
seed_dict = {'dhamma': '法', 'bhikkhu': '比丘', 'nibbānaṃ': '涅槃'}
results = []
for pali_word, han_word in seed_dict.items():
    if pali_word in pali_model.wv and han_model is not None and han_word in han_model.wv:
        sim = cosine_similarity(
            pali_model.wv[pali_word].reshape(1, -1),
            han_model.wv[han_word].reshape(1, -1)
        )[0][0]
        results.append({'Pali': pali_word, 'Han': han_word, 'Similarity': sim})
    else:
        results.append({'Pali': pali_word, 'Han': han_word, 'Similarity': 'Không có dữ liệu'})

# Xuất kết quả
result_df = pd.DataFrame(results)
result_df.to_csv('pali_han_semantic_mapping.csv', index=False)
print(result_df if len(result_df) <= 20 else result_df.head())