"""
phase3_divergence_v2.py
=======================
Phát hiện và phân loại dị biệt giữa MN (Pali) và MA (Hán cổ).

Input:  Phase2_Alignment_v16.csv, Phase2_Dictionary_v16.csv
Output:
  Phase3_Pair_Divergence.csv      — divergence từng cặp chunk (TRUE_PAIR + SOFT)
  Phase3_Formulaic_Omission.csv   — báo cáo OMISSION do rút gọn formulaic
  Phase3_Sutta_Summary.csv        — tổng hợp theo từng cặp kinh
  Phase3_Key_Findings.csv         — passages dị biệt nổi bật nhất

━━ THAY ĐỔI LỚN SO VỚI v1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[NEW] Han_Fanout — chiều đo mới, quan trọng nhất:
  Số lượng Pali chunks map vào cùng 1 Han segment.
  Phát hiện ra nghịch lý trong v1: Tier 2 (được giữ) có fanout cao →
  mapping sai; Tier 3 (bị loại) có fanout=1 → mapping 1:1 đúng.

[NEW] Working set được tái định nghĩa theo Han_Fanout thay vì Tier:
  TRUE_PAIR  : fanout == 1, Tier 1/2/3  → 2,777 pairs (dùng cho divergence)
  SOFT       : fanout 2–5, Tier 1/2     →   302 pairs (dùng có điều kiện)
  FORMULAIC  : fanout > 5               → 5,245 pairs (classify thành OMISSION)

[NEW] FORMULAIC_OMISSION taxonomy:
  Khi nhiều Pali chunks bị ép vào 1 Han segment (formulaic),
  đây là bằng chứng MA rút gọn cấu trúc lặp của MN.
  compression_ratio = total_pali_chars / han_chars → đo mức độ rút gọn.

[FIX] Tên cột: 'Emb_Similarity' → 'Emb_Sim_Combined' (v16)
[FIX] Z-score per-sutta thay vì absolute emb_drop
[FIX] TRANS dict: exact match only (bỏ prefix loop)
[FIX] Recalibrate: PARALLEL threshold 0.35, HIGH_DIV p75=0.743
[FIX] Case Studies dùng 'true_pairs' thay vì 'align'
"""

import pandas as pd
import numpy as np
import re
import math
import jieba
from collections import Counter, defaultdict

# ── Jieba setup ───────────────────────────────────────────────
for t in ['比丘', '世尊', '阿難', '無常', '無我', '涅槃', '正念', '正定',
          '四念處', '八正道', '阿羅漢', '正見', '念處', '苦集滅道']:
    jieba.add_word(t, freq=10000)

# ── Constants (calibrated on TRUE_PAIR subset) ────────────────
GLOBAL_EMB_MEAN    = 0.1947   # mean Emb_Sim_Combined, TRUE_PAIR
GLOBAL_EMB_STD     = 0.0738   # std  Emb_Sim_Combined, TRUE_PAIR
HIGH_DIV_THRESHOLD = 0.743    # p75 của div_score distribution (TRUE_PAIR)
FORMULAIC_FANOUT   = 5        # fanout > 5 → FORMULAIC bucket
PARALLEL_EMB       = 0.35     # emb > 0.35 → candidate PARALLEL (~2% pairs)

# ══════════════════════════════════════════════════════════════
# 1. LOAD & CLASSIFY MAPPING QUALITY
# ══════════════════════════════════════════════════════════════
print("Loading data...")
align = pd.read_csv('Phase2_Alignment_v19_v2.csv')
dic   = pd.read_csv('Phase2_Dictionary_v19_v2.csv')
print(f"  Alignment: {len(align):,} rows | Dictionary: {len(dic):,} entries")

# Tính Han_Fanout: số Pali chunks trỏ vào cùng Han segment
han_fanout_map = (align
    .groupby('Han_Seg_ID')['Pali_Chunk_ID']
    .nunique()
    .rename('Han_Fanout'))
align = align.join(han_fanout_map, on='Han_Seg_ID')

# Classify Mapping_Quality
def _classify_mapping(row):
    f = row['Han_Fanout']
    t = row['Confidence_Tier']
    if f == 1 and t in [1, 2, 3]:
        return 'TRUE_PAIR'
    elif 2 <= f <= FORMULAIC_FANOUT and t in [1, 2,3]:
        return 'SOFT'
    elif f > FORMULAIC_FANOUT and t in [1, 2, 3]:
        return 'FORMULAIC'
    else:
        return 'DISCARD'   # Tier 0, hoặc T3 với fanout > 1

align['Mapping_Quality'] = align.apply(_classify_mapping, axis=1)

# Working sets
true_pairs = align[align['Mapping_Quality'] == 'TRUE_PAIR'].copy()
soft_pairs = align[align['Mapping_Quality'] == 'SOFT'].copy()
formulaic  = align[align['Mapping_Quality'] == 'FORMULAIC'].copy()
usable     = pd.concat([true_pairs, soft_pairs], ignore_index=True)

print(f"\nWorking set breakdown:")
print(f"  TRUE_PAIR  (fanout=1, T1-3): {len(true_pairs):,} rows | {true_pairs['MN_No'].nunique()} suttas")
print(f"  SOFT       (fanout 2-5, T1-2): {len(soft_pairs):,} rows | {soft_pairs['MN_No'].nunique()} suttas")
print(f"  FORMULAIC  (fanout>5, T1-3): {len(formulaic):,} rows | {formulaic['MN_No'].nunique()} suttas")
print(f"  DISCARD    (T0 or T3+soft): {(align['Mapping_Quality']=='DISCARD').sum():,} rows")

# ══════════════════════════════════════════════════════════════
# 2. TRANSLATION LOOKUP — exact match only
#    (bỏ prefix loop: gây false positive với dasa/dasabala/dasannam)
# ══════════════════════════════════════════════════════════════
def _simplify_pali(t):
    t = str(t).lower()
    for k, v in {'ā': 'a', 'ī': 'i', 'ū': 'u', 'ṃ': 'm', 'ṅ': 'n',
                 'ñ': 'n', 'ṇ': 'n', 'ṭ': 't', 'ḍ': 'd', 'ḷ': 'l'}.items():
        t = t.replace(k, v)
    return t

TRANS = defaultdict(set)
for _, row in dic.iterrows():
    key = _simplify_pali(str(row['Pali_Word']))
    TRANS[key].add(str(row['Han_Word']))

STOP_P = {'ti', 'kho', 'pana', 'ca', 'va', 'pi', 'hi', 'na', 'atha',
          'eva', 'evam', 'iti', 'me', 'te', 'so', 'tam', 'yam', 'idam',
          'no', 'tatra', 'va'}
STOP_H = {'之', '者', '也', '而', '於', '以', '得', '為', '曰', '其', '則',
          '乃', '與', '及', '亦', '此', '從', '便', '已', '即', '若', '彼', '我'}


def pali_key_terms(text):
    terms = {}
    for w in str(text).split():
        ws = _simplify_pali(re.sub(r'[^a-zA-Zāīūṃṇṭḍḷñṅ]', '', w))
        if len(ws) >= 4 and ws not in STOP_P and ws in TRANS:
            terms[ws] = TRANS[ws]
    return terms


def han_token_set(text):
    text = str(text).strip().replace(' ', '')
    if not text:
        return set()
    return {w for w in jieba.cut(text)
            if w and w not in STOP_H and not re.match(r'^[\d\s\W]+$', w)}


# ══════════════════════════════════════════════════════════════
# 3. DIVERGENCE METRICS
# ══════════════════════════════════════════════════════════════

def missing_term_score(pali_text, han_text):
    terms = pali_key_terms(pali_text)
    if not terms:
        return 0.0, [], []
    han_toks = han_token_set(han_text)
    missing, present = [], []
    for pw, expected in terms.items():
        if any(h in han_toks for h in expected):
            present.append(pw)
        else:
            missing.append(pw)
    return len(missing) / len(terms), missing, present


def length_asymmetry(p_len, h_len):
    if h_len == 0:
        return 3.0
    return math.log2(max(p_len, 1) / max(h_len, 1))


def classify_divergence(emb, lex, miss, p_len, h_len,
                        sutta_mean, sutta_std, mapping_quality):
    """
    Phân loại dị biệt.
    - TRUE_PAIR: dùng z-score per-sutta
    - SOFT: dùng global baseline (ít tin cậy hơn)
    """
    len_asym = length_asymmetry(p_len, h_len)

    if sutta_std > 1e-6:
        emb_z = (sutta_mean - emb) / sutta_std
    else:
        emb_z = (GLOBAL_EMB_MEAN - emb) / GLOBAL_EMB_STD

    div_score = (
        0.40 * (1.0 - min(emb, 1.0))        +
        0.30 * miss                          +
        0.20 * (1.0 - min(lex, 1.0))        +
        0.10 * min(abs(len_asym) / 3.0, 1.0)
    )

    # Classification
    if emb_z > 1.5 and miss > 0.5:
        div_type = 'SEMANTIC_DRIFT'
    elif len_asym > 1.5 and miss > 0.3:
        div_type = 'OMISSION'
    elif len_asym < -1.5:
        div_type = 'ADDITION'
    elif miss > 0.6:
        div_type = 'SEMANTIC_DRIFT'
    elif emb > PARALLEL_EMB and lex > 0.08:
        div_type = 'PARALLEL'
    else:
        div_type = 'UNCERTAIN'

    signals = sum([emb_z > 1.0, miss > 0.4, abs(len_asym) > 1.0, lex < 0.05])
    confidence = ['LOW', 'MEDIUM', 'MEDIUM', 'HIGH', 'HIGH'][min(signals, 4)]
    # Downgrade SOFT pairs
    if mapping_quality == 'SOFT' and confidence == 'HIGH':
        confidence = 'MEDIUM'

    return div_type, round(div_score, 4), confidence, round(emb_z, 3)


# ══════════════════════════════════════════════════════════════
# 4. PER-SUTTA BASELINES (chỉ từ TRUE_PAIR)
# ══════════════════════════════════════════════════════════════
print("\nComputing per-sutta baselines (TRUE_PAIR only)...")
sutta_stats = (
    true_pairs
    .groupby(['MN_No', 'MA_No'])['Emb_Sim_Combined']
    .agg(['mean', 'std'])
    .rename(columns={'mean': 'emb_mean', 'std': 'emb_std'})
    .reset_index()
)
sutta_stats['emb_std'] = sutta_stats['emb_std'].fillna(GLOBAL_EMB_STD)
sutta_stat_map = {
    (int(r.MN_No), int(r.MA_No)): (r.emb_mean, r.emb_std)
    for r in sutta_stats.itertuples()
}

# ══════════════════════════════════════════════════════════════
# 5. COMPUTE PAIR-LEVEL DIVERGENCE (TRUE_PAIR + SOFT)
# ══════════════════════════════════════════════════════════════
print("Computing divergence per pair...")

pair_rows = []
for _, row in usable.iterrows():
    mn, ma = int(row['MN_No']), int(row['MA_No'])
    emb_mean, emb_std = sutta_stat_map.get(
        (mn, ma), (GLOBAL_EMB_MEAN, GLOBAL_EMB_STD)
    )
    miss, missing_terms, present_terms = missing_term_score(
        str(row['Pali_Text']), str(row['Han_Text'])
    )
    div_type, div_score, confidence, emb_z = classify_divergence(
        emb             = float(row['Emb_Sim_Combined']),
        lex             = float(row['Lex_Overlap']),
        miss            = miss,
        p_len           = int(row['P_Char_Len']),
        h_len           = int(row['H_Char_Len']),
        sutta_mean      = emb_mean,
        sutta_std       = emb_std,
        mapping_quality = row['Mapping_Quality'],
    )
    pair_rows.append({
        'MN_No'             : mn,
        'MA_No'             : ma,
        'Pali_Chunk_ID'     : row['Pali_Chunk_ID'],
        'Han_Seg_ID'        : row['Han_Seg_ID'],
        'Mapping_Quality'   : row['Mapping_Quality'],
        'Han_Fanout'        : int(row['Han_Fanout']),
        'Confidence_Tier'   : int(row['Confidence_Tier']),
        'Emb_Sim_Combined'  : float(row['Emb_Sim_Combined']),
        'Lex_Overlap'       : float(row['Lex_Overlap']),
        'Hyb_Similarity'    : float(row['Hyb_Similarity']),
        'P_Char_Len'        : int(row['P_Char_Len']),
        'H_Char_Len'        : int(row['H_Char_Len']),
        'Div_Score'         : div_score,
        'Div_Type'          : div_type,
        'Div_Confidence'    : confidence,
        'Missing_Term_Score': round(miss, 4),
        'Len_Asymmetry'     : round(length_asymmetry(int(row['P_Char_Len']), int(row['H_Char_Len'])), 3),
        'Emb_Z'             : emb_z,
        'Sutta_Emb_Mean'    : round(emb_mean, 4),
        'Sutta_Emb_Std'     : round(emb_std, 4),
        'Missing_Terms'     : '|'.join(missing_terms[:8]),
        'Present_Terms'     : '|'.join(present_terms[:5]),
        'Pali_Text'         : str(row['Pali_Text'])[:300],
        'Han_Text'          : str(row['Han_Text'])[:300],
    })

df_pair = pd.DataFrame(pair_rows)
print(f"  Done: {len(df_pair):,} pairs analyzed")

# ══════════════════════════════════════════════════════════════
# 6. FORMULAIC OMISSION — báo cáo riêng
#    Mỗi FORMULAIC bucket = bằng chứng MA rút gọn cấu trúc lặp MN
# ══════════════════════════════════════════════════════════════
print("Analyzing formulaic omissions...")

form_rows = []
for (mn, ma), grp in formulaic.groupby(['MN_No', 'MA_No']):
    n_pali_chunks = grp['Pali_Chunk_ID'].nunique()
    n_han_segs    = grp['Han_Seg_ID'].nunique()
    total_pali    = grp['P_Char_Len'].sum()
    mean_han      = grp['H_Char_Len'].mean()
    total_han_est = mean_han * n_han_segs

    compression = round(total_pali / max(total_han_est, 1), 2)

    # Severity: dựa trên compression ratio
    if compression > 10:
        severity = 'EXTREME'
    elif compression > 5:
        severity = 'HEAVY'
    elif compression > 3:
        severity = 'MODERATE'
    else:
        severity = 'LIGHT'

    # Sample Pali chunk bị collapse
    sample = grp.sort_values('P_Char_Len', ascending=False).iloc[0]

    form_rows.append({
        'MN_No'              : mn,
        'MA_No'              : ma,
        'N_Pali_Chunks'      : n_pali_chunks,
        'N_Han_Segs'         : n_han_segs,
        'Max_Fanout'         : int(grp['Han_Fanout'].max()),
        'Total_Pali_Chars'   : int(total_pali),
        'Mean_Han_Chars'     : round(mean_han, 1),
        'Compression_Ratio'  : compression,
        'Omission_Severity'  : severity,
        'Sample_Pali'        : str(sample['Pali_Text'])[:200],
        'Sample_Han'         : str(sample['Han_Text'])[:200],
        'Mean_Emb_Sim'       : round(grp['Emb_Sim_Combined'].mean(), 4),
    })

df_form = pd.DataFrame(form_rows).sort_values('Compression_Ratio', ascending=False)
print(f"  Formulaic omission events: {len(df_form)} suttas")

# ══════════════════════════════════════════════════════════════
# 7. SUTTA-LEVEL SUMMARY (kết hợp pair divergence + formulaic)
# ══════════════════════════════════════════════════════════════
print("Computing sutta-level summary...")

# Tất cả suttas trong data
all_mn_ma = align[align['Confidence_Tier'].isin([1,2,3])].groupby(['MN_No','MA_No']).size().reset_index()[['MN_No','MA_No']]

sutta_rows = []
for _, sutta in all_mn_ma.iterrows():
    mn, ma = int(sutta['MN_No']), int(sutta['MA_No'])

    # Pair divergence (TRUE_PAIR + SOFT)
    grp = df_pair[(df_pair['MN_No'] == mn) & (df_pair['MA_No'] == ma)]

    # Formulaic omission
    form = df_form[(df_form['MN_No'] == mn) & (df_form['MA_No'] == ma)]
    has_formulaic = len(form) > 0
    form_severity = form.iloc[0]['Omission_Severity'] if has_formulaic else 'NONE'
    compression   = form.iloc[0]['Compression_Ratio'] if has_formulaic else 0.0
    n_formulaic   = int(form.iloc[0]['N_Pali_Chunks']) if has_formulaic else 0

    if len(grp) == 0:
        # Sutta chỉ có formulaic, không có TRUE_PAIR
        div_index = 0.0
        dominant  = 'FORMULAIC_OMISSION' if has_formulaic else 'NO_DATA'
        type_dist = {}
    else:
        type_dist = grp['Div_Type'].value_counts().to_dict()
        weights   = grp['Confidence_Tier'].map({1: 2.0, 2: 1.0, 3: 0.5}).fillna(0.5)
        div_index = float(np.average(grp['Div_Score'], weights=weights))

        content_types = {k: v for k, v in type_dist.items()
                         if k not in ('PARALLEL', 'UNCERTAIN')}
        if has_formulaic and compression > 3:
            content_types['FORMULAIC_OMISSION'] = n_formulaic
        dominant = (max(content_types, key=content_types.get)
                    if content_types else 'PARALLEL')

    # Missing terms
    all_missing = Counter()
    for t in grp['Missing_Terms'].dropna():
        all_missing.update(t.split('|'))
    top_missing = [t for t, _ in all_missing.most_common(5) if t]

    high_div = grp[grp['Div_Score'] > HIGH_DIV_THRESHOLD]

    sutta_rows.append({
        'MN_No'                : mn,
        'MA_No'                : ma,
        'N_True_Pairs'         : len(grp[grp['Mapping_Quality'] == 'TRUE_PAIR']) if len(grp) else 0,
        'N_Soft_Pairs'         : len(grp[grp['Mapping_Quality'] == 'SOFT']) if len(grp) else 0,
        'N_Formulaic_Chunks'   : n_formulaic,
        'Divergence_Index'     : round(div_index, 4),
        'Dominant_Type'        : dominant,
        'N_SEMANTIC_DRIFT'     : type_dist.get('SEMANTIC_DRIFT', 0),
        'N_OMISSION'           : type_dist.get('OMISSION', 0),
        'N_ADDITION'           : type_dist.get('ADDITION', 0),
        'N_PARALLEL'           : type_dist.get('PARALLEL', 0),
        'N_UNCERTAIN'          : type_dist.get('UNCERTAIN', 0),
        'N_High_Div'           : len(high_div),
        'Pct_High_Div'         : round(len(high_div) / len(grp) * 100, 1) if len(grp) else 0,
        'Has_Formulaic_Omission': has_formulaic,
        'Formulaic_Severity'   : form_severity,
        'Compression_Ratio'    : compression,
        'Mean_Emb_Sim'         : round(grp['Emb_Sim_Combined'].mean(), 4) if len(grp) else 0,
        'Mean_Lex'             : round(grp['Lex_Overlap'].mean(), 4) if len(grp) else 0,
        'Mean_Missing_Term'    : round(grp['Missing_Term_Score'].mean(), 4) if len(grp) else 0,
        'Mean_Len_Asym'        : round(grp['Len_Asymmetry'].mean(), 3) if len(grp) else 0,
        'Top_Missing_Terms'    : '|'.join(top_missing),
    })

df_sutta = (pd.DataFrame(sutta_rows)
              .sort_values('Divergence_Index', ascending=False)
              .reset_index(drop=True))

# ══════════════════════════════════════════════════════════════
# 8. KEY FINDINGS
# ══════════════════════════════════════════════════════════════
print("Extracting key findings...")

# Chỉ dùng TRUE_PAIR có div type rõ ràng
true_divs = df_pair[
    (df_pair['Mapping_Quality'] == 'TRUE_PAIR') &
    (df_pair['Div_Type'].isin(['SEMANTIC_DRIFT', 'OMISSION', 'ADDITION']))
].sort_values('Div_Score', ascending=False)

# SOFT tier high confidence
soft_top = df_pair[
    (df_pair['Mapping_Quality'] == 'SOFT') &
    (df_pair['Div_Score'] > HIGH_DIV_THRESHOLD) &
    (df_pair['Div_Confidence'] == 'HIGH') &
    (df_pair['Div_Type'].isin(['SEMANTIC_DRIFT', 'OMISSION', 'ADDITION']))
].sort_values('Div_Score', ascending=False).head(20)

key_findings = (
    pd.concat([true_divs, soft_top])
    .drop_duplicates(subset=['Pali_Chunk_ID', 'Han_Seg_ID'])
    .sort_values('Div_Score', ascending=False)
    .head(100)
)

# ══════════════════════════════════════════════════════════════
# 9. SAVE
# ══════════════════════════════════════════════════════════════
df_pair.to_csv('Phase3_Pair_Divergence.csv',       index=False, encoding='utf-8-sig')
df_form.to_csv('Phase3_Formulaic_Omission.csv',    index=False, encoding='utf-8-sig')
df_sutta.to_csv('Phase3_Sutta_Summary.csv',        index=False, encoding='utf-8-sig')
key_findings.to_csv('Phase3_Key_Findings.csv',     index=False, encoding='utf-8-sig')

# ══════════════════════════════════════════════════════════════
# 10. REPORT
# ══════════════════════════════════════════════════════════════
SEP = '=' * 65
print(f'\n{SEP}')
print('PHASE 3 v2 — DIVERGENCE REPORT')
print(SEP)

print(f'\n── Working set ──')
print(f'TRUE_PAIR  : {len(true_pairs):,} pairs ({true_pairs["MN_No"].nunique()} suttas)')
print(f'SOFT       : {len(soft_pairs):,} pairs ({soft_pairs["MN_No"].nunique()} suttas)')
print(f'FORMULAIC  : {len(formulaic):,} pairs ({formulaic["MN_No"].nunique()} suttas)')

print(f'\n── Divergence type (TRUE_PAIR + SOFT) ──')
type_counts = df_pair['Div_Type'].value_counts()
for dtype, cnt in type_counts.items():
    pct = cnt / len(df_pair) * 100
    bar = '█' * int(pct / 3)
    print(f'  {dtype:<18}: {cnt:5d} ({pct:5.1f}%) {bar}')

print(f'\n── Formulaic Omission severity ──')
sev_counts = df_form['Omission_Severity'].value_counts()
for sev, cnt in sev_counts.items():
    print(f'  {sev:<10}: {cnt} suttas')

print(f'\n── Top 10 most divergent suttas ──')
print(f'{"MN":>4} {"MA":>4}  {"Div_Idx":>7}  {"Dominant":>22}  '
      f'{"DRIFT":>5} {"OMIT":>5} {"FORM":>5}  {"Compress":>8}')
print('-' * 80)
for _, r in df_sutta.head(10).iterrows():
    print(f'MN{r["MN_No"]:>3} MA{r["MA_No"]:>3}  {r["Divergence_Index"]:>7.4f}  '
          f'{r["Dominant_Type"]:>22}  '
          f'{r["N_SEMANTIC_DRIFT"]:>5} {r["N_OMISSION"]:>5} '
          f'{r["N_Formulaic_Chunks"]:>5}  '
          f'{r["Compression_Ratio"]:>7.1f}x')

print(f'\n── Top 5 key findings (TRUE_PAIR, high confidence) ──')
for _, r in key_findings.head(5).iterrows():
    print(f'\n  MN{r["MN_No"]}↔MA{r["MA_No"]}  [{r["Div_Type"]}]  '
          f'div={r["Div_Score"]:.3f}  emb={r["Emb_Sim_Combined"]:.3f}  '
          f'z={r["Emb_Z"]:+.2f}  fanout={r["Han_Fanout"]}')
    print(f'  Missing: {r["Missing_Terms"][:60]}')
    print(f'  Pali: {str(r["Pali_Text"])[:100]}')
    print(f'  Han : {str(r["Han_Text"])[:100]}')

print(f'\n{SEP}')
print(f'✅ Phase3_Pair_Divergence.csv     : {len(df_pair):,} rows')
print(f'✅ Phase3_Formulaic_Omission.csv  : {len(df_form):,} rows')
print(f'✅ Phase3_Sutta_Summary.csv       : {len(df_sutta):,} rows')
print(f'✅ Phase3_Key_Findings.csv        : {len(key_findings):,} rows')
print(f'\nNext: chạy phase3_complete_v2.py')