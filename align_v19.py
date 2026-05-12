"""
step3_align_v19.py  —  The Ultimate Fix: Constrained DP (Sakoe-Chiba Band)
============================================================
Giải quyết triệt để lỗi "Nhảy cóc vị trí" (Pos Dist > 0.5)
Bằng cách ép thuật toán Quy hoạch động (DP) chỉ được phép 
tìm kiếm trong một hành lang chéo (±15% độ dài bài kinh).
Kết hợp với Dynamic Merge Penalty để chống Overcollapse.
"""

import pandas as pd
import numpy as np
import torch
import re
import math
import jieba
from sentence_transformers import SentenceTransformer
import faiss
from collections import Counter
import time

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
ANCHOR_MIN        = 0.35
POSITION_ALPHA    = 0.12
GAP_PENALTY       = 0.10       # Phạt bỏ qua
WINDOW_RATIO      = 0.15       # MỚI: Hành lang Sakoe-Chiba (Cho phép lệch tối đa 15%)
MIN_ZONE_SIZE     = 3
SEED_BOOST        = 30
DEDUP_THRESHOLD   = 0.95
INJECT_MAX_TOKENS = 8
BATCH_SIZE        = 64
MAX_SPAN          = 4
SPAN_WEIGHT       = 0.4

# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════
df_p = pd.read_csv('pali_rechunked_2.csv')
df_h = pd.read_csv('han_tagged_2.csv')

HAN_TEXT_COL = 'Cleaned_Content'
if HAN_TEXT_COL not in df_h.columns:
    for fallback in ['masked_content', 'Content']:
        if fallback in df_h.columns:
            HAN_TEXT_COL = fallback
            print(f'⚠️  Cleaned_Content không có, dùng: {HAN_TEXT_COL}')
            break
    else:
        raise ValueError('Không tìm thấy cột text trong han_tagged.csv')

print(f'Pali chunks : {len(df_p):,}  mean={df_p["char_len"].mean():.0f}c')
print(f'Han content : {len(df_h):,}  (toàn bộ, kể cả formulaic)')
print(f'Han text col: {HAN_TEXT_COL}')

# ══════════════════════════════════════════════════════════════
# MAPPING
# ══════════════════════════════════════════════════════════════
MAPPING = {
   61:14
}

# ══════════════════════════════════════════════════════════════
# SEED DICT + TEXT PROCESSING
# ══════════════════════════════════════════════════════════════
SEED = {
    'dukkha':'苦',   'samudaya':'集',  'nirodha':'滅',  'magga':'道',
    'sacca':'諦',    'rupa':'色',      'vedana':'受',   'sanna':'想',
    'sankhara':'行', 'vinnana':'識',   'sila':'戒',     'samadhi':'定',
    'panna':'慧',    'avijja':'無明',  'tanha':'愛',    'upadana':'取',
    'bhava':'有',    'jati':'生',      'marana':'死',   'phassa':'觸',
    'anicca':'無常', 'anatta':'無我',  'bhikkhu':'比丘','nibbana':'涅槃',
    'dhamma':'法',   'sati':'念',      'viriya':'精進', 'metta':'慈',
    'karuna':'悲',   'kamma':'業',     'citta':'心',    'ariya':'聖',
    'jhana':'禪',    'tathagata':'如來','bhagava':'世尊',
    'sariputta':'舍利弗','ananda':'阿難',
    'sammaditthi':'正見','sammasati':'正念','sammasamadhi':'正定',
    'pathavi':'地',  'apo':'水',       'tejo':'火',     'vayo':'風',
    'asava':'漏',    'sotapanna':'須陀洹','arahant':'阿羅漢',
    'namarupa':'名色','salayatana':'六入','jara':'老',
    'satipatthana':'念處','indriya':'根','bala':'力',
    'mudita':'喜',   'upekkha':'捨',   'saddha':'信',   'cetana':'思',
    'upakkilesa':'穢','vattha':'衣',   'rajako':'染',
    'mala':'垢',     'lobha':'貪',     'byapada':'恚',
    'mano':'慢',     'pamada':'逸',    'macchariya':'慳',
    'maya':'誑',     'issa':'嫉',      'kodha':'瞋',
}

def simplify(t):
    t = str(t).lower()
    for k, v in {'ā':'a','ī':'i','ū':'u','ṃ':'m','ṅ':'n','ñ':'n',
                 'ṇ':'n','ṭ':'t','ḍ':'d','ḷ':'l'}.items():
        t = t.replace(k, v)
    return t

PALI_TO_HAN = {}
for p, h in SEED.items():
    ps = simplify(p)
    PALI_TO_HAN[ps] = h
    for length in range(4, min(len(ps)+1, 7)):
        if ps[:length] not in PALI_TO_HAN:
            PALI_TO_HAN[ps[:length]] = h

STOP_P = {'ti','kho','pana','ca','va','pi','hi','na','atha','eva','evam',
           'iti','me','te','so','tam','yam','idam','no','tatra','iti'}
STOP_H = {'之','者','也','而','於','以','得','為','曰','其','則','乃',
           '與','及','亦','此','從','便','已','即','若','彼','我'}

for t in ['比丘','世尊','阿難','舍利弗','無常','無我','涅槃','正念',
          '正定','四念處','八正道','阿羅漢','穢心','心穢','放逸']:
    jieba.add_word(t, freq=10000)

def pali_tokens(text):
    words = []
    for w in str(text).split():
        ws = simplify(re.sub(r'[^a-zA-Zāīūṃṇṭḍḷñṅ]', '', w))
        if len(ws) >= 4 and ws not in STOP_P:
            words.append(ws)
    return words

def han_tokens(text):
    text = str(text).replace(' ', '')
    return [w for w in jieba.cut(text)
            if w.strip() and w not in STOP_H
            and not re.match(r'^[\d\s\W]+$', w)]

def lex_score(pali_text, han_text):
    pw = set(pali_tokens(pali_text))
    if not pw:
        return 0.0
    ht = set(han_tokens(han_text))
    hits = 0
    for w in pw:
        translation = None
        for length in range(min(len(w), 12), 3, -1):
            if w[:length] in PALI_TO_HAN:
                translation = PALI_TO_HAN[w[:length]]
                break
        if translation and translation in ht:
            hits += 1
    return hits / len(pw)

# ══════════════════════════════════════════════════════════════
# KEYWORD INJECTION
# ══════════════════════════════════════════════════════════════
def inject_keywords(pali_text: str, max_tokens: int = INJECT_MAX_TOKENS) -> str:
    tokens = pali_tokens(pali_text)
    if not tokens:
        return pali_text
    translations, seen_han = [], set()
    for w in tokens:
        han = None
        for length in range(min(len(w), 12), 3, -1):
            if w[:length] in PALI_TO_HAN:
                han = PALI_TO_HAN[w[:length]]
                break
        if han and han not in seen_han:
            translations.append(han)
            seen_han.add(han)
        if len(translations) >= max_tokens:
            break
    if not translations:
        return pali_text
    return f'[{" ".join(translations)}] {pali_text}'

def inject_batch(texts: list) -> list:
    return [inject_keywords(t) for t in texts]

# ══════════════════════════════════════════════════════════════
# SPAN EMBEDDING
# ══════════════════════════════════════════════════════════════
def build_han_spans(h_texts: list, max_span: int = MAX_SPAN):
    n = len(h_texts)
    spans, span_index = [], []
    for length in range(1, min(max_span, n) + 1):
        for start in range(n - length + 1):
            spans.append(' '.join(h_texts[start:start+length]))
            span_index.append((start, start+length))
    return spans, span_index

def build_span_sim_matrix(pe_inj: np.ndarray,
                           h_texts_dedup: list,
                           encode_fn,
                           max_span: int = MAX_SPAN) -> np.ndarray:
    n_p    = pe_inj.shape[0]
    n_h_d  = len(h_texts_dedup)

    h_spans, h_span_idx = build_han_spans(h_texts_dedup, max_span)
    if not h_spans:
        return np.zeros((n_p, n_h_d), dtype=np.float32)

    h_span_emb = encode_fn(h_spans)          # (n_spans, 768)
    span_sim   = pe_inj @ h_span_emb.T       # (n_p, n_spans)

    sim_projected = np.zeros((n_p, n_h_d), dtype=np.float32)
    for k, (j_start, j_end) in enumerate(h_span_idx):
        for j in range(j_start, j_end):
            sim_projected[:, j] = np.maximum(
                sim_projected[:, j],
                span_sim[:, k]
            )
    return sim_projected

# ══════════════════════════════════════════════════════════════
# DEDUP HAN SEGMENTS
# ══════════════════════════════════════════════════════════════
def dedup_han_segments(h_texts, h_ids, embeddings,
                       threshold=DEDUP_THRESHOLD):
    n = len(h_texts)
    if n <= 1:
        return h_texts, h_ids, embeddings, {0: [0]}, 0

    sim_mat = embeddings @ embeddings.T
    parent  = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    for i in range(n):
        for j in range(i+1, n):
            if sim_mat[i, j] > threshold:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    dedup_indices = sorted(clusters.keys())
    expand_map    = {new_idx: clusters[rep]
                     for new_idx, rep in enumerate(dedup_indices)}

    dedup_texts = [h_texts[i] for i in dedup_indices]
    dedup_ids   = [h_ids[i]   for i in dedup_indices]
    dedup_emb   = embeddings[dedup_indices]
    n_removed   = n - len(dedup_indices)

    return dedup_texts, dedup_ids, dedup_emb, expand_map, n_removed

def expand_assignment(assign_dedup, expand_map, n_original):
    result = []
    for dedup_hi in assign_dedup:
        originals = expand_map.get(int(dedup_hi), [int(dedup_hi)])
        result.append(min(originals[0], n_original - 1))
    return result

# ══════════════════════════════════════════════════════════════
# PMI
# ══════════════════════════════════════════════════════════════
class PMI:
    def __init__(self):
        self.pair = Counter()
        self.pali = Counter()
        self.han  = Counter()
        self.n    = 0

    def add(self, pw, hw, w=1.0):
        pc, hc = Counter(pw), Counter(hw)
        self.n += w
        for p, pn in pc.items():
            self.pali[p] += w
            for h, hn in hc.items():
                self.pair[(p, h)] += w * min(pn, hn)
        for h in hc:
            self.han[h] += w

    def build(self, min_co=3, min_pmi=0.5, top_n=5):
        N = max(self.n, 1)
        rows = []
        for (pw, hw), cnt in self.pair.items():
            if cnt < min_co:
                continue
            pc = self.pali[pw]; hc = self.han[hw]
            if not pc or not hc:
                continue
            pmi_v = math.log2((cnt/N) / ((pc/N)*(hc/N) + 1e-10))
            if pmi_v <= min_pmi:
                continue
            idf  = math.log(N / (hc + 1))
            prec = cnt / pc; rec = cnt / max(hc, 1)
            f    = 2*prec*rec / max(prec+rec, 1e-10)
            rows.append({
                'Pali_Word': pw, 'Han_Word': hw,
                'Co_Occur': round(cnt, 2), 'PMI': round(pmi_v, 4),
                'IDF': round(idf, 4), 'Score': round(pmi_v*max(idf,0.1), 4),
                'Precision': round(prec, 4), 'Recall': round(rec, 4),
                'F_Score': round(f, 4),
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values('Score', ascending=False)
        return df.groupby('Pali_Word').head(top_n).reset_index(drop=True)

PALI_ROOTS = {
    'dukkh':'dukkha',  'bhikkh':'bhikkhu', 'nibban':'nibbana',
    'vedan':'vedana',  'sann':'sanna',     'vinnan':'vinnana',
    'sankhar':'sankhara','dhamma':'dhamma','kamma':'kamma',
    'samadh':'samadhi','panna':'panna',    'sila':'sila',
    'metta':'metta',   'karun':'karuna',   'anicca':'anicca',
    'anatt':'anatta',  'tanha':'tanha',    'avijj':'avijja',
    'sati':'sati',     'bhava':'bhava',    'jati':'jati',
    'magg':'magga',    'sacca':'sacca',    'rupa':'rupa',
    'upakk':'upakkilesa',
}

def get_root(w):
    for pre, root in PALI_ROOTS.items():
        if w.startswith(pre):
            return root
    return w

def consolidate(df, top_n=5):
    if len(df) == 0:
        return df
    df = df.copy()
    df['Pali_Root'] = df['Pali_Word'].apply(get_root)
    agg = (df.groupby(['Pali_Root', 'Han_Word'])
             .agg(Co_Occur=('Co_Occur','sum'), PMI=('PMI','max'),
                  IDF=('IDF','first'), Score=('Score','max'),
                  Precision=('Precision','max'), Recall=('Recall','max'),
                  F_Score=('F_Score','max'))
             .reset_index().rename(columns={'Pali_Root':'Pali_Word'}))
    return (agg.sort_values('Score', ascending=False)
               .groupby('Pali_Word').head(top_n)
               .reset_index(drop=True))

pmi = PMI()
n_inj = 0
for p_raw, h_w in SEED.items():
    ps = simplify(p_raw)
    if ' ' not in ps:
        for _ in range(SEED_BOOST):
            pmi.add([ps], [h_w])
        n_inj += 1
print(f"Seed: {n_inj} × {SEED_BOOST} = {n_inj*SEED_BOOST:,} virtual pairs")

# ══════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════
print('Loading LaBSE...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
model = SentenceTransformer('sentence-transformers/LaBSE').to(device)

def encode(texts, bs=BATCH_SIZE):
    e = model.encode(texts, batch_size=bs,
                     show_progress_bar=False,
                     convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(e)
    return e

# ══════════════════════════════════════════════════════════════
# ALIGNMENT CORE — BAND-CONSTRAINED MONOTONIC DP
# ══════════════════════════════════════════════════════════════
def compute_mutual(sim_mat, assign):
    best_p = sim_mat.argmax(axis=0)
    return np.array([best_p[assign[pi]] == pi for pi in range(len(assign))])

def find_anchors(sim_mat, assign, mutual, anchor_min=ANCHOR_MIN):
    anchors = []
    for pi in range(len(assign)):
        hi   = int(assign[pi])
        esim = float(sim_mat[pi, hi])
        if mutual[pi] and esim >= anchor_min:
            anchors.append((pi, hi, esim))
    anchors.sort(key=lambda x: x[0])
    return anchors

def conf_tier(esim, mutual, collapse):
    if collapse and esim < 0.28:    return 0   
    if mutual and esim >= 0.35:     return 1   
    if esim >= 0.28:                return 2   
    if collapse:                    return 3   
    return 3

def constrained_dp_align(sim_mat, gap_penalty=GAP_PENALTY, window_ratio=WINDOW_RATIO):
    """ 
    Pure DP with a Sakoe-Chiba Band constraint.
    Ép thuật toán chỉ được dò tìm trong hành lang chéo ±15%.
    Trị dứt điểm bệnh "Nhảy cóc" từ đầu bài xuống cuối bài kinh.
    Kết hợp Dynamic Penalty để trị dứt điểm bệnh Overcollapse.
    """
    n_p, n_h = sim_mat.shape
    
    max_deviation = int(max(n_p, n_h) * window_ratio) + 2

    dp    = np.full((n_p + 1, n_h + 1), -np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    trace = np.zeros((n_p + 1, n_h + 1), dtype=np.int8)
    up_count = np.zeros((n_p + 1, n_h + 1), dtype=np.int32)

    for i in range(1, n_p + 1):
        dp[i, 0] = dp[i-1, 0] - gap_penalty
        trace[i, 0] = 1
        up_count[i, 0] = up_count[i-1, 0] + 1
    for j in range(1, n_h + 1):
        dp[0, j] = dp[0, j-1] - gap_penalty
        trace[0, j] = 2

    for i in range(1, n_p + 1):
        expected_j = int((i / n_p) * n_h)
        
        j_start = max(1, expected_j - max_deviation)
        j_end   = min(n_h + 1, expected_j + max_deviation + 1)
        
        for j in range(j_start, j_end):
            ms = sim_mat[i-1, j-1]
            
            cost_diag = dp[i-1, j-1] + ms
            
            # Dynamic Merge Penalty: Càng dồn toa càng bị trừ điểm mạnh
            current_up_streak = up_count[i-1, j]
            dynamic_merge_penalty = 0.10 + (0.05 * current_up_streak)
            dynamic_merge_penalty = min(dynamic_merge_penalty, 0.40) 
            
            cost_up = dp[i-1, j] + ms - dynamic_merge_penalty
            
            # Cost left (Bỏ qua đoạn Hán rác)
            cost_left = dp[i, j-1] - (gap_penalty * 0.5) 
            
            best = max(cost_diag, cost_up, cost_left)
            dp[i, j] = best
            
            if best == cost_diag:   
                trace[i, j] = 0; up_count[i, j] = 0
            elif best == cost_up:   
                trace[i, j] = 1; up_count[i, j] = current_up_streak + 1
            else:                   
                trace[i, j] = 2; up_count[i, j] = 0

    # Backtracking
    assign = np.zeros(n_p, dtype=np.int32)
    i, j   = n_p, n_h

    while i > 0 and j > 0:
        step = int(trace[i, j])
        if step == 0:
            i -= 1; j -= 1
            assign[i] = j
        elif step == 1:
            i -= 1
            assign[i] = max(j - 1, 0)
        else:
            j -= 1

    while i > 0:
        i -= 1
        assign[i] = 0
        
    assign = np.clip(assign, 0, n_h - 1)
    mutual  = compute_mutual(sim_mat, assign)
    anchors = find_anchors(sim_mat, assign, mutual)
    
    return assign, 'constrained_dp', anchors

# ══════════════════════════════════════════════════════════════
# MAIN ALIGNMENT LOOP
# ══════════════════════════════════════════════════════════════
rows          = []
t0            = time.time()
p_groups      = {int(sno): grp for sno, grp in df_p.groupby('sutta_no')}
h_groups      = {int(sno): grp for sno, grp in df_h.groupby('Sutta_No')}
total         = len(MAPPING)
method_counts = Counter()
n_mutual_fed  = 0
anchor_counts = []
dedup_stats   = []

print(f'\nAligning {total} pairs (v19: Constrained DP + Dynamic Penalty)...\n')

for i, (mn, ma) in enumerate(MAPPING.items(), 1):
    pg = p_groups.get(mn)
    hg = h_groups.get(ma)
    if pg is None or hg is None:
        print(f'  ⚠  MN{mn}↔MA{ma}: missing data, skip')
        continue

    p_texts = pg['chunk_text'].tolist()
    p_ids   = pg['chunk_id'].tolist()
    p_lens  = pg['char_len'].tolist()

    h_texts = hg[HAN_TEXT_COL].fillna('').tolist()
    h_ids   = hg['Segment_ID'].tolist()

    p_texts_injected = inject_batch(p_texts)

    # Encode
    pe_raw = encode(p_texts)
    pe_inj = encode(p_texts_injected)
    he     = encode(h_texts)

    # Dedup Han
    h_texts_d, h_ids_d, he_d, expand_map, n_removed = \
        dedup_han_segments(h_texts, h_ids, he)
    dedup_stats.append(n_removed)

    n_h_d = len(h_texts_d)

    # ── Chunk-level sim (n_p, n_h_dedup) ─────────────────────
    chunk_sim = pe_inj @ he_d.T

    # # Hạ sim của Han formulaic xuống (Masking)
    # h_is_form = np.zeros(len(h_texts), dtype=bool)
    # if 'is_formulaic' in hg.columns:
    #     h_is_form = hg['is_formulaic'].fillna(False).values.astype(bool)
    # h_is_form_d = np.zeros(len(h_texts_d), dtype=bool)
    # for new_idx, orig_list in expand_map.items():
    #     if any(idx < len(h_is_form) and h_is_form[idx] for idx in orig_list):
    #         h_is_form_d[new_idx] = True

    # FORMULAIC_PENALTY = 0.3
    # chunk_sim[:, h_is_form_d] -= FORMULAIC_PENALTY
    # chunk_sim = np.clip(chunk_sim, -1.0, 1.0)

    # ── Span-level sim (n_p, n_h_dedup) ─────────────────────
    span_sim = build_span_sim_matrix(pe_inj, h_texts_d, encode, MAX_SPAN)

    # Kết hợp
    sim_dedup = (1 - SPAN_WEIGHT) * chunk_sim + SPAN_WEIGHT * span_sim

    # ── Constrained Monotonic DP (Sakoe-Chiba Band) ──────────
    assign_d, method, anchors = constrained_dp_align(sim_dedup)
    
    method_counts[method.split('_')[0]] += 1
    anchor_counts.append(len(anchors))

    # Map về original Han indices
    assign_orig = expand_assignment(assign_d, expand_map, len(h_texts))

    # Sim raw trên original space (để report)
    sim_raw = pe_raw @ he.T    # (n_p, n_h_orig)

    usage  = Counter(assign_orig)
    usage_vals = sorted(usage.values(), reverse=True)
    if len(usage_vals) >= 5:
        p90_idx = max(0, int(len(usage_vals) * 0.10) - 1)
        ak = max(usage_vals[p90_idx], math.ceil(len(p_texts) / max(len(h_texts), 1)) + 1)
    else:
        ak = math.ceil(len(p_texts) / max(len(h_texts), 1)) + 2
        
    mutual = np.array([
        sim_raw[pi, assign_orig[pi]] == sim_raw[pi].max()
        for pi in range(len(p_texts))
    ])

    for pi in range(len(p_texts)):
        hi_orig  = assign_orig[pi]
        hi_dedup = int(assign_d[pi])

        esim_inj  = float(sim_dedup[pi, min(hi_dedup, n_h_d - 1)])
        esim_raw  = float(sim_raw[pi, hi_orig])
        esim_span = float(span_sim[pi, min(hi_dedup, n_h_d - 1)])
        lx        = lex_score(p_texts[pi], h_texts[hi_orig])
        hyb       = 0.6 * esim_inj + 0.4 * lx
        mut       = bool(mutual[pi])
        oc        = usage[hi_orig] > ak
        ct        = conf_tier(esim_inj, mut, oc)

        pos_p    = pi / max(len(p_texts) - 1, 1)
        pos_h    = hi_orig / max(len(h_texts) - 1, 1)
        pos_dist = round(abs(pos_p - pos_h), 4)

        rows.append({
            'MN_No'            : mn,
            'MA_No'            : ma,
            'Pali_Chunk_ID'    : p_ids[pi],
            'Han_Seg_ID'       : h_ids[hi_orig],
            'Emb_Sim_Combined' : round(esim_inj,  4),
            'Emb_Sim_Raw'      : round(esim_raw,  4),
            'Emb_Sim_Span'     : round(esim_span, 4),
            'Lex_Overlap'      : round(lx,         4),
            'Hyb_Similarity'   : round(hyb,        4),
            'Is_Mutual'        : mut,
            'Overcollapse'     : oc,
            'Confidence_Tier'  : ct,
            'Han_Usage_Count'  : usage[hi_orig],
            'Adaptive_K'       : ak,
            'Assign_Method'    : method,
            'N_Anchors'        : len(anchors),
            'N_Dedup_Removed'  : n_removed,
            'Position_Distance': pos_dist,
            'P_Char_Len'       : p_lens[pi],
            'H_Char_Len'       : len(h_texts[hi_orig]),
            'Pali_Injected'    : p_texts_injected[pi][:200],
            'Pali_Text'        : p_texts[pi][:300],
            'Han_Text'         : h_texts[hi_orig][:300],
        })

        if mut:
            pw = pali_tokens(p_texts[pi])
            hw = han_tokens(h_texts[hi_orig])
            if pw and hw:
                pmi.add(pw, hw)
                n_mutual_fed += 1

    elapsed = time.time() - t0
    eta     = elapsed / i * (total - i)
    print(f'  [{i:2d}/{total}] MN{mn:>3}↔MA{ma:<3}  '
          f'p={len(p_texts):3d} h={len(h_texts):3d}→{n_h_d:3d}(dedup-{n_removed})  '
          f'{method:<25} anchors={len(anchors):2d}  '
          f'{elapsed/60:.1f}m ETA {eta/60:.1f}m')

# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════
df_out = pd.DataFrame(rows)
df_out.to_csv('Phase2_Alignment_v19_test.csv', index=False, encoding='utf-8-sig')

print(f'\nPMI fed from {n_mutual_fed:,} mutual pairs')
df_dict = consolidate(pmi.build())
df_dict.to_csv('Phase2_Dictionary_v19_test.csv', index=False, encoding='utf-8-sig')

# ══════════════════════════════════════════════════════════════
# REPORTS
# ══════════════════════════════════════════════════════════════
print(f'\n{"="*65}')
print('ALIGNMENT REPORT v19 (Constrained DP + Dynamic Penalty)')
print(f'{"="*65}')
print(f'Total pairs       : {len(df_out):,}')
print(f'Mutual            : {df_out["Is_Mutual"].mean()*100:.1f}%  (target: >20%)')
print(f'Overcollapse      : {df_out["Overcollapse"].mean()*100:.1f}%')
print(f'Avg sim combined  : {df_out["Emb_Sim_Combined"].mean():.4f}  (target: >0.35)')
print(f'Avg sim raw       : {df_out["Emb_Sim_Raw"].mean():.4f}')
print(f'Avg sim span      : {df_out["Emb_Sim_Span"].mean():.4f}')
print(f'Avg lex overlap   : {df_out["Lex_Overlap"].mean():.4f}')
print(f'Avg pos distance  : {df_out["Position_Distance"].mean():.4f}  (target: <0.15)')
print(f'Avg anchors/sutta : {np.mean(anchor_counts):.1f}')
print(f'Avg dedup removed : {np.mean(dedup_stats):.1f} segs/sutta')

print(f'\n── Span embedding effect ───────────────────────────────────')
delta = df_out["Emb_Sim_Span"] - df_out["Emb_Sim_Raw"]
print(f'  Mean span gain  : +{delta.mean():.4f}')
print(f'  Pairs span>raw  : {(delta>0).sum()} ({(delta>0).mean()*100:.1f}%)')

print(f'\n── Confidence Tiers ────────────────────────────────────────')
for t in [1, 2, 3, 0]:
    n   = (df_out['Confidence_Tier'] == t).sum()
    cov = df_out[df_out['Confidence_Tier'] == t]['MN_No'].nunique()
    lbl = {1:'reliable', 2:'usable', 3:'noisy', 0:'discard'}[t]
    print(f'  Tier {t}: {n:5d} ({n/len(df_out)*100:4.1f}%)  MN={cov}  [{lbl}]')

print(f'\n── Assign methods ──────────────────────────────────────────')
for method, cnt in method_counts.most_common():
    print(f'  {method:<30} {cnt:3d} suttas')

print(f'\n── MN7↔MA93 verification ───────────────────────────────────')
mn7 = df_out[(df_out['MN_No']==7) & (df_out['MA_No']==93)].head(5)
if len(mn7):
    print(f'  Avg sim combined: {mn7["Emb_Sim_Combined"].mean():.4f}')
    print(f'  Avg pos dist    : {mn7["Position_Distance"].mean():.4f}')
    for _, r in mn7.iterrows():
        print(f'\n  [{r["Pali_Chunk_ID"]}] → [{r["Han_Seg_ID"]}]')
        print(f'  Pali: {r["Pali_Text"][:70]}')
        print(f'  Han : {r["Han_Text"][:70]}')
        print(f'  comb={r["Emb_Sim_Combined"]:.3f} '
              f'raw={r["Emb_Sim_Raw"]:.3f} '
              f'span={r["Emb_Sim_Span"]:.3f} '
              f'pos={r["Position_Distance"]:.3f} '
              f'Tier={r["Confidence_Tier"]}')

print(f'\n── Seed check (PMI Top-1) ──────────────────────────────────')
SEED_CHECK = {
    'dukkha':'苦',   'dhamma':'法',    'bhikkhu':'比丘','nibbana':'涅槃',
    'anicca':'無常', 'anatta':'無我',  'sati':'念',     'kamma':'業',
    'tanha':'愛',    'avijja':'無明',  'vedana':'受',   'rupa':'色',
    'panna':'慧',    'sila':'戒',      'samadhi':'定',  'metta':'慈',
}
ok = 0
for p, exp in SEED_CHECK.items():
    sub = df_dict[df_dict['Pali_Word'].str.startswith(p[:5], na=False)]
    if len(sub):
        got = sub.nlargest(1, 'Score')['Han_Word'].values[0]
        hit = got == exp
        if hit:
            ok += 1
        print(f'  {"✅" if hit else "❌"} {p:<12} → {got}  (expected {exp})')
    else:
        print(f'  ❌ {p:<12} → not found')

print(f'\n  PMI Top-1: {ok}/{len(SEED_CHECK)} = {ok/len(SEED_CHECK)*100:.0f}%')
print(f'\nTotal time: {(time.time()-t0)/60:.1f} min')
print(f'\n✅ Saved: Phase2_Alignment_v19_test.csv, Phase2_Dictionary_v19_test.csv')