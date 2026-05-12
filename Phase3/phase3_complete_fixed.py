"""
phase3_complete_v2.py
=====================
Phân tích chuyên sâu — chạy SAU phase3_divergence_v2.py.

Input:
  Phase2_Alignment_v16.csv
  Phase2_Dictionary_v16.csv
  Phase3_Sutta_Summary.csv       ← output của phase3_divergence_v2.py
  Phase3_Formulaic_Omission.csv  ← output của phase3_divergence_v2.py

Output:
  Phase3_Numeric.csv
  Phase3_ProperNoun.csv
  Phase3_CaseStudies.csv
  Phase3_SemanticShift.csv
  Phase3_FinalReport.txt

THAY ĐỔI so với v1:
  [V2] Dùng TRUE_PAIR + SOFT làm working set (thay vì Tier 1+2)
  [V2] Tích hợp Formulaic_Omission vào FinalReport
  [FIX] 'Emb_Similarity' → 'Emb_Sim_Combined'
  [FIX] Case Studies dùng 'usable' (TRUE_PAIR+SOFT) thay vì 'align'
"""

import pandas as pd
import numpy as np
import re
import math
import jieba
from collections import Counter, defaultdict

# ── Jieba setup ───────────────────────────────────────────────
for t in ['比丘', '世尊', '阿難', '無常', '無我', '涅槃', '五蘊', '八正道',
          '四念處', '十二因緣', '舍利弗', '目犍連', '給孤獨', '阿羅漢']:
    jieba.add_word(t, freq=10000)

# ── Load ──────────────────────────────────────────────────────
print("Loading data...")
align  = pd.read_csv('Phase2_Alignment_v16.csv')
dic    = pd.read_csv('Phase2_Dictionary_v16.csv')
phase3 = pd.read_csv('Phase3_Sutta_Summary.csv')
df_form = pd.read_csv('Phase3_Formulaic_Omission.csv')

# [V2] Tái tạo working set giống phase3_divergence_v2.py
FORMULAIC_FANOUT = 5
han_fanout_map = (align
    .groupby('Han_Seg_ID')['Pali_Chunk_ID']
    .nunique()
    .rename('Han_Fanout'))
align = align.join(han_fanout_map, on='Han_Seg_ID')

def _classify_mapping(row):
    f = row['Han_Fanout']
    t = row['Confidence_Tier']
    if f == 1 and t in [1, 2, 3]:
        return 'TRUE_PAIR'
    elif 2 <= f <= FORMULAIC_FANOUT and t in [1, 2]:
        return 'SOFT'
    elif f > FORMULAIC_FANOUT and t in [1, 2, 3]:
        return 'FORMULAIC'
    else:
        return 'DISCARD'

align['Mapping_Quality'] = align.apply(_classify_mapping, axis=1)
usable = align[align['Mapping_Quality'].isin(['TRUE_PAIR', 'SOFT'])].copy()
print(f"  Usable (TRUE_PAIR+SOFT): {len(usable):,} | Dict: {len(dic):,}")

# ══════════════════════════════════════════════════════════════
# A. NUMERIC DISCREPANCY
# ══════════════════════════════════════════════════════════════
print("\n[A] Numeric discrepancy analysis...")

PALI_NUM = {
    'eka'    : {'1', '一'},
    'dve'    : {'2', '二'},
    'tini'   : {'3', '三'},
    'cattari': {'4', '四'},
    'panca'  : {'5', '五'},
    'cha'    : {'6', '六'},
    'satta'  : {'7', '七'},
    'attha'  : {'8', '八'},
    'nava'   : {'9', '九'},
    'dasa'   : {'10', '十'},
}


def extract_pali_nums(text):
    text = str(text).lower()
    found = {}
    for num_word, han_set in PALI_NUM.items():
        if re.search(r'\b' + num_word + r'\b', text):
            found[num_word] = han_set
    return found


def extract_han_nums(text):
    found = set()
    for n in '一二三四五六七八九十':
        if n in str(text):
            found.add(n)
    return found


num_rows = []
for _, row in usable.iterrows():
    p_nums = extract_pali_nums(str(row['Pali_Text']))
    if not p_nums:
        continue
    h_nums = extract_han_nums(str(row['Han_Text']))
    for pali_num, expected_han in p_nums.items():
        if not bool(expected_han & h_nums):
            num_rows.append({
                'MN_No'       : int(row['MN_No']),
                'MA_No'       : int(row['MA_No']),
                'Mapping_Quality': row['Mapping_Quality'],
                'Pali_Num'    : pali_num,
                'Expected_Han': '|'.join(sorted(expected_han)),
                'Found_Han'   : '|'.join(sorted(h_nums)),
                'Missing'     : '|'.join(sorted(expected_han - h_nums)),
                'Pali_Text'   : str(row['Pali_Text'])[:150],
                'Han_Text'    : str(row['Han_Text'])[:150],
            })

df_num = pd.DataFrame(num_rows)
df_num.to_csv('Phase3_Numeric.csv', index=False, encoding='utf-8-sig')
print(f"  Numeric mismatches: {len(df_num)}")
if len(df_num):
    print(f"  Most mismatched: {df_num['Pali_Num'].value_counts().head(5).to_dict()}")

# ══════════════════════════════════════════════════════════════
# B. PROPER NOUN DIVERGENCE
# ══════════════════════════════════════════════════════════════
print("\n[B] Proper noun analysis...")

PROPER_NOUNS = {
    'sariputta'   : ['舍利弗', '舍梨子'],
    'moggallana'  : ['目犍連', '目乾連'],
    'ananda'      : ['阿難', '阿難陀'],
    'kassapa'     : ['迦葉', '大迦葉'],
    'upali'       : ['優波離', '優婆離'],
    'devadatta'   : ['提婆達多', '調達'],
    'brahma'      : ['梵天', '梵'],
    'sakka'       : ['帝釋', '天帝釋'],
    'mara'        : ['魔', '波旬'],
    'nigantha'    : ['尼乾', '尼乾陀'],
    'savatthi'    : ['舍衛', '舍衛國'],
    'rajagaha'    : ['王舍城', '羅閱城'],
    'vesali'      : ['毘舍離', '毗舍離'],
    'kosala'      : ['拘薩羅', '拘薩'],
    'magadha'     : ['摩竭', '摩竭陀'],
    'jetavana'    : ['祇樹', '祇陀林'],
    'nalanda'     : ['那爛陀'],
    'kapilavatthu': ['迦維羅衛', '迦毗羅衛'],
    'rahula'      : ['羅云', '羅睺羅'],
}


def simplify_pali(t):
    t = str(t).lower()
    for k, v in {'ā': 'a', 'ī': 'i', 'ū': 'u', 'ṃ': 'm', 'ṅ': 'n',
                 'ñ': 'n', 'ṇ': 'n', 'ṭ': 't', 'ḍ': 'd', 'ḷ': 'l'}.items():
        t = t.replace(k, v)
    return t


proper_rows = []
for _, row in usable.iterrows():
    p_simp = simplify_pali(str(row['Pali_Text']))
    h_text = str(row['Han_Text'])
    for pali_name, han_variants in PROPER_NOUNS.items():
        if pali_name not in p_simp:
            continue
        found_han = [h for h in han_variants if h in h_text]
        if not found_han:
            status = 'MISSING'
            found  = ''
        elif len(han_variants) > 1 and len(found_han) < len(han_variants):
            status = 'VARIANT'
            found  = '|'.join(found_han)
        else:
            status = 'MATCH'
            found  = '|'.join(found_han)
        proper_rows.append({
            'MN_No'          : int(row['MN_No']),
            'MA_No'          : int(row['MA_No']),
            'Mapping_Quality': row['Mapping_Quality'],
            'Pali_Name'      : pali_name,
            'Expected_Han'   : '|'.join(han_variants),
            'Found_Han'      : found,
            'Status'         : status,
            'Pali_Text'      : str(row['Pali_Text'])[:120],
            'Han_Text'       : str(row['Han_Text'])[:120],
        })

df_proper = pd.DataFrame(proper_rows)
df_proper.to_csv('Phase3_ProperNoun.csv', index=False, encoding='utf-8-sig')
missing_proper = df_proper[df_proper['Status'] == 'MISSING']
print(f"  Total analyzed: {len(df_proper)} | MISSING: {len(missing_proper)}")
if len(missing_proper):
    print(f"  Most missing: {missing_proper['Pali_Name'].value_counts().head(8).to_dict()}")

# ══════════════════════════════════════════════════════════════
# C. CASE STUDIES
#    [V2] Dùng usable (TRUE_PAIR + SOFT), không dùng align
#    [FIX] Emb_Sim_Combined thay Emb_Similarity
# ══════════════════════════════════════════════════════════════
print("\n[C] Case studies...")

CASE_PAIRS = [
    (61,  14,  'Ambalaṭṭhikārāhulovāda', 'Kinh Giáo Giới La-hầu-la'),
    (56,  133, 'Upāli Sutta',            'Kinh Ưu-bà-li'),
    (44,  210, 'Cūḷavedalla Sutta',      'Kinh Tiểu Phân Biệt'),
    (7,   93,  'Vattha Sutta',           'Kinh Vải'),
    (123, 32,  'Acchariyabbhuta Sutta',  'Kinh Hy Hữu Vị Tằng Hữu'),
]


def _summarize_case(mn, ma, p3_row, form_row, missing_terms):
    dom   = p3_row['Dominant_Type']
    div   = float(p3_row['Divergence_Index'])
    n_om  = int(p3_row['N_OMISSION'])
    n_dr  = int(p3_row['N_SEMANTIC_DRIFT'])
    n_fo  = int(p3_row['N_Formulaic_Chunks'])
    terms = ', '.join(missing_terms[:4]) if missing_terms else 'N/A'
    comp  = float(p3_row['Compression_Ratio']) if p3_row['Compression_Ratio'] else 0

    summary = ''
    if n_fo > 0 and comp > 3:
        summary += (f"MA rút gọn {n_fo} đoạn lặp của MN xuống ~{comp:.0f}x "
                    f"(FORMULAIC_OMISSION). ")
    if dom == 'SEMANTIC_DRIFT':
        summary += f"Nội dung khác ý nghĩa trong {n_dr} đoạn. Thiếu: {terms}."
    elif dom == 'OMISSION':
        summary += f"MA lược bỏ {n_om} đoạn. Thiếu: {terms}."
    elif not summary:
        summary = f"Tương đồng tương đối. DI={div:.3f}."
    return summary.strip()


case_rows = []
for mn, ma, en_name, vi_name in CASE_PAIRS:
    p_sub   = usable[(usable['MN_No'] == mn) & (usable['MA_No'] == ma)].copy()
    p3_rows = phase3[(phase3['MN_No'] == mn) & (phase3['MA_No'] == ma)]
    form_rows_sub = df_form[(df_form['MN_No'] == mn) & (df_form['MA_No'] == ma)]

    if p3_rows.empty:
        print(f"  ⚠  MN{mn}↔MA{ma}: không có Phase3 summary, skip")
        continue

    p3   = p3_rows.iloc[0]
    form = form_rows_sub.iloc[0] if len(form_rows_sub) else None
    worst = p_sub.nsmallest(1, 'Emb_Sim_Combined') if len(p_sub) else pd.DataFrame()

    missing_terms = [t for t in str(p3.get('Top_Missing_Terms', '')).split('|')
                     if t and len(t) > 2]

    case_rows.append({
        'MN_No'              : mn,
        'MA_No'              : ma,
        'Sutta_Name_EN'      : en_name,
        'Sutta_Name_VI'      : vi_name,
        'Divergence_Index'   : round(float(p3['Divergence_Index']), 4),
        'Dominant_Type'      : p3['Dominant_Type'],
        'N_True_Pairs'       : int(p3['N_True_Pairs']),
        'N_Soft_Pairs'       : int(p3['N_Soft_Pairs']),
        'N_Formulaic_Chunks' : int(p3['N_Formulaic_Chunks']),
        'Formulaic_Severity' : p3['Formulaic_Severity'],
        'Compression_Ratio'  : float(p3['Compression_Ratio']),
        'N_OMISSION'         : int(p3['N_OMISSION']),
        'N_SEMANTIC_DRIFT'   : int(p3['N_SEMANTIC_DRIFT']),
        'N_ADDITION'         : int(p3['N_ADDITION']),
        'Mean_Emb_Sim'       : round(p_sub['Emb_Sim_Combined'].mean(), 4) if len(p_sub) else 0,
        'Mean_Lex'           : round(p_sub['Lex_Overlap'].mean(), 4) if len(p_sub) else 0,
        'Key_Missing_Terms'  : '|'.join(missing_terms[:6]),
        'Worst_Pali'         : str(worst.iloc[0]['Pali_Text'])[:200] if len(worst) else '',
        'Worst_Han'          : str(worst.iloc[0]['Han_Text'])[:200] if len(worst) else '',
        'Worst_Emb'          : round(float(worst.iloc[0]['Emb_Sim_Combined']), 4) if len(worst) else 0,
        'Summary'            : _summarize_case(mn, ma, p3, form, missing_terms),
    })

df_case = pd.DataFrame(case_rows)
df_case.to_csv('Phase3_CaseStudies.csv', index=False, encoding='utf-8-sig')
print(f"  Case studies saved: {len(df_case)} suttas")

# ══════════════════════════════════════════════════════════════
# D. SEMANTIC SHIFT
# ══════════════════════════════════════════════════════════════
print("\n[D] Semantic shift analysis...")

PALI_FUNCTION_WORDS = {
    'hoti', 'ahosi', 'bhavati', 'atthi', 'natthi', 'iti', 'eva', 'kho',
    'pana', 'ca', 'pi', 'hi', 'nu', 'na', 'atha', 'idha', 'tatra',
    'tassa', 'tesam', 'aham', 'tattha', 'pajanati', 'gahapati',
    'bhagava', 'anuruddha',
}

pali_to_han = defaultdict(list)
for _, row in dic.iterrows():
    pali_to_han[str(row['Pali_Word'])].append({
        'han'      : row['Han_Word'],
        'co_occur' : row['Co_Occur'],
        'precision': row['Precision'],
        'score'    : float(row['Score']),
    })


def safe_entropy(scores):
    pos = [s for s in scores if s > 0]
    if len(pos) < 2:
        return 0.0
    total = sum(pos)
    if total <= 0:
        return 0.0
    return round(-sum((s / total) * math.log2(s / total) for s in pos if s > 0), 3)


def _interpret_shift(pali_word, translations):
    doctrinal = {'dukkha', 'dhamma', 'sati', 'samadhi', 'panna', 'nibbana',
                 'tanha', 'kamma', 'citta', 'ariya', 'jhana', 'sila',
                 'vedana', 'sanna', 'sankhara', 'vinnana', 'rupa'}
    w = pali_word.lower()
    if any(w.startswith(d[:4]) for d in doctrinal):
        return 'Thuật ngữ giáo lý cốt lõi — có thể phản ánh tranh luận giáo phái'
    if len(translations) >= 4:
        return 'Nhiều biến thể dịch — từ không có equiv Hán trực tiếp'
    return 'Biến thể dịch thông thường'


shift_rows = []
for pali_word, translations in pali_to_han.items():
    if len(translations) < 2 or pali_word.lower() in PALI_FUNCTION_WORDS:
        continue
    if len(pali_word) < 5:
        continue
    translations = sorted(translations, key=lambda x: -x['score'])
    pos_t = [t for t in translations if t['score'] > 0]
    if len(pos_t) < 2:
        continue
    if max(t['co_occur'] for t in translations) < 8:
        continue
    entropy = safe_entropy([t['score'] for t in pos_t])
    shift_rows.append({
        'Pali_Word'           : pali_word,
        'N_Han_Variants'      : len(pos_t),
        'Translation_Entropy' : entropy,
        'Primary_Han'         : translations[0]['han'],
        'Primary_Score'       : round(translations[0]['score'], 3),
        'All_Han_Variants'    : '|'.join(t['han'] for t in translations),
        'All_Scores'          : '|'.join(str(round(t['score'], 1)) for t in translations),
        'Interpretation'      : _interpret_shift(pali_word, translations),
    })

df_shift = (pd.DataFrame(shift_rows)
              .sort_values('Translation_Entropy', ascending=False)
              .reset_index(drop=True))
df_shift.to_csv('Phase3_SemanticShift.csv', index=False, encoding='utf-8-sig')
print(f"  Terms with 2+ Han variants: {len(df_shift)}")
if len(df_shift):
    print(df_shift[['Pali_Word','N_Han_Variants','Primary_Han']].head(5).to_string(index=False))

# ══════════════════════════════════════════════════════════════
# E. FINAL REPORT
# ══════════════════════════════════════════════════════════════
print("\n[E] Writing final report...")

SEP = '=' * 65
report = f"""
{SEP}
GIAI ĐOẠN 3 v2 — BÁO CÁO PHÂN TÍCH DỊ BIỆT MN↔MA
{SEP}

━━ I. TỔNG QUAN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Working set: TRUE_PAIR={len(usable[usable['Mapping_Quality']=='TRUE_PAIR']):,} | SOFT={len(usable[usable['Mapping_Quality']=='SOFT']):,}
Phương pháp lọc: Han_Fanout (thay vì Confidence_Tier)
  → TRUE_PAIR: fanout=1, Tier 1-3
  → SOFT:      fanout 2-5, Tier 1-2
  → FORMULAIC: fanout>5 → báo OMISSION riêng

━━ II. PHÂN LOẠI DỊ BIỆT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{phase3['Dominant_Type'].value_counts().to_string()}

OMISSION      : {int(phase3['N_OMISSION'].sum())} cases (pair-level)
ADDITION      : {int(phase3['N_ADDITION'].sum())} cases
FORMULAIC_OM  : {int(phase3['N_Formulaic_Chunks'].sum())} Pali chunks bị rút gọn

━━ III. FORMULAIC OMISSION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{df_form['Omission_Severity'].value_counts().to_string()}
Top 5 compression ratio:
{df_form.nlargest(5,'Compression_Ratio')[['MN_No','MA_No','N_Pali_Chunks','Compression_Ratio','Omission_Severity']].to_string(index=False)}

━━ IV. NUMERIC DISCREPANCY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Số cặp mismatch: {len(df_num)}
{df_num['Pali_Num'].value_counts().to_string() if len(df_num) else '  (không có data)'}

━━ V. PROPER NOUN DIVERGENCE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tổng occurrences: {len(df_proper)} | MISSING: {len(missing_proper)}
{missing_proper['Pali_Name'].value_counts().head(8).to_string() if len(missing_proper) else ''}

━━ VI. SEMANTIC SHIFT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Terms có ≥2 dịch pháp: {len(df_shift)}
{df_shift[['Pali_Word','N_Han_Variants','Primary_Han']].head(8).to_string(index=False) if len(df_shift) else ''}

━━ VII. CASE STUDIES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
for _, r in df_case.iterrows():
    report += f"""
▶ MN{r['MN_No']}↔MA{r['MA_No']} — {r['Sutta_Name_EN']}
  DI={r['Divergence_Index']} | {r['Dominant_Type']}
  TRUE_PAIR={r['N_True_Pairs']} | FORMULAIC={r['N_Formulaic_Chunks']} ({r['Formulaic_Severity']}, {r['Compression_Ratio']:.1f}x)
  OMISSION={r['N_OMISSION']} | DRIFT={r['N_SEMANTIC_DRIFT']} | ADD={r['N_ADDITION']}
  {r['Summary']}
"""

report += f"""
━━ VIII. TOP 15 SUTTAS DỊ BIỆT NHẤT ━━━━━━━━━━━━━━━━━━━━━
"""
for _, r in phase3.head(15).iterrows():
    fo_flag = f" [FORM×{r['Compression_Ratio']:.0f}x]" if r['Has_Formulaic_Omission'] else ''
    report += (f"MN{int(r['MN_No']):>3}↔MA{int(r['MA_No']):<3}  "
               f"DI={r['Divergence_Index']:.3f}  {r['Dominant_Type']}{fo_flag}\n")

report += f"""
{SEP}
Output files:
  Phase3_Pair_Divergence.csv     — pair-level divergence
  Phase3_Formulaic_Omission.csv  — {len(df_form)} sutta-level OMISSION events
  Phase3_Numeric.csv             — {len(df_num)} numeric mismatches
  Phase3_ProperNoun.csv          — {len(df_proper)} proper noun occurrences
  Phase3_CaseStudies.csv         — {len(df_case)} detailed case studies
  Phase3_SemanticShift.csv       — {len(df_shift)} polysemous terms
  Phase3_FinalReport.txt         — this file
{SEP}
"""

with open('Phase3_FinalReport.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print("✅ All Phase 3 v2 outputs saved.")