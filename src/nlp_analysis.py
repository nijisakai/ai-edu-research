#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP/Semantic Analysis for AI Education Product Research
Deep text mining on 教育产品统计_V5.csv
"""

import csv
import json
import re
import os
import warnings
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# ── Paths ──
CSV_PATH = '/Users/sakai/Desktop/产业调研/教育产品统计_V5.csv'
OUTPUT_DIR = '/Users/sakai/Desktop/产业调研/ai-edu-research/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Column indices ──
COL = {
    'case_id': 2, 'case_name': 3, 'region': 4, 'school_level': 5,
    'industry_status_case': 6, 'innovation_case': 7,
    'tech_path_case': 8, 'tech_elements_case': 9,
    'tool_name': 11, 'company': 12, 'product_form': 14, 'product_category': 15,
    'main_scenario': 17, 'sub_scenario': 18,
    'scenario_l1': 19, 'scenario_l2': 20,
    'cultivation': 21, 'potential_effect': 22,
    'tech_path_tool': 23, 'tech_elements_tool': 24,
    'industry_status_tool': 25, 'innovation_tool': 26,
    'subject': 27, 'tool_std_name': 28, 'company_std': 29, 'province': 30,
}

# ── Custom education/tech dictionary for jieba ──
CUSTOM_WORDS = [
    '人工智能', '大语言模型', '自然语言处理', '深度学习', '机器学习',
    '知识图谱', '语音识别', '图像识别', '计算机视觉', '数据挖掘',
    '生成式AI', '智能推荐', '自适应学习', '个性化学习', '精准教学',
    '学情分析', '学情诊断', '智能批改', '智能评测', '智能辅导',
    '智能助手', '教学助手', '虚拟助手', '数字人', '虚拟教师',
    '多模态', '大数据', '云计算', '物联网', '区块链', '元宇宙',
    '增强现实', '虚拟现实', '混合现实', '数字孪生',
    '智慧教育', '智慧课堂', '智慧校园', '在线教育', '远程教育',
    '翻转课堂', '项目式学习', '探究式学习', '协作学习', '混合式教学',
    '核心素养', '计算思维', '信息素养', '数字素养', '创新思维',
    '学习行为', '学习画像', '学习路径', '教育数据', '过程性评价',
    '形成性评价', '教学评一体化', '因材施教', '减负增效',
    '素质教育', '五育并举', '德智体美劳', '跨学科', '学科融合',
    'AIGC', 'ChatGPT', 'RAG', 'Prompt', 'AI赋能', 'AI助手',
    '技术路径', '产品形态', '应用场景', '培养方向', '技术要素',
    '语音交互', '文本生成', '图像生成', '视频生成', '代码生成',
    '智能体', '教育大模型', '垂直大模型', '通用大模型',
    '数据采集', '画像生成', '个性化反馈', '即时反馈', '自动评分',
    '作业批改', '试题生成', '资源推荐', '学习分析', '教学设计',
    '课堂互动', '师生互动', '人机互动', '协同备课', '集体备课',
]

STOPWORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
    '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
    '会', '着', '没有', '看', '好', '自己', '这', '他', '她', '它',
    '们', '那', '些', '什么', '怎么', '如何', '可以', '能', '能够',
    '进行', '通过', '利用', '使用', '实现', '提供', '支持', '开展',
    '基于', '针对', '围绕', '结合', '采用', '运用', '借助', '依托',
    '以及', '同时', '并且', '而且', '但是', '然而', '因此', '所以',
    '如果', '虽然', '不仅', '而是', '还是', '或者', '以及', '等',
    '其中', '对于', '关于', '根据', '按照', '为了', '从而', '进而',
    '不断', '充分', '有效', '积极', '深入', '全面', '具体', '主要',
    '重要', '相关', '一定', '一些', '各种', '其他', '更加', '更好',
    '方面', '过程', '情况', '问题', '方式', '方法', '内容', '形式',
    '工作', '活动', '发展', '建设', '推进', '促进', '加强', '提高',
    '提升', '增强', '优化', '完善', '推动', '引导', '培养', '探索',
    '构建', '打造', '形成', '创新', '改革', '实践', '研究', '分析',
    '设计', '开发', '应用', '服务', '管理', '评价', '教学', '学习',
    '教育', '学生', '教师', '学校', '课堂', '课程', '知识', '能力',
    '素养', '思维', '技术', '工具', '平台', '系统', '模型', '数据',
    '资源', '环境', '功能', '效果', '质量', '水平', '目标', '需求',
    '特点', '优势', '价值', '意义', '作用', '影响', '成效', '成果',
])

# ── Initialize jieba ──
for w in CUSTOM_WORDS:
    jieba.add_word(w)

# ── Helper functions ──

def load_data():
    """Load CSV and return all rows."""
    rows = []
    with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)
    print(f"[DATA] Loaded {len(rows)} rows, {len(set(r[COL['case_id']] for r in rows))} unique cases")
    return header, rows

def get_case_level_data(rows):
    """Deduplicate to case-level (one row per case_id)."""
    seen = set()
    cases = []
    for row in rows:
        cid = row[COL['case_id']]
        if cid not in seen:
            seen.add(cid)
            cases.append(row)
    return cases

def tokenize(text):
    """Segment Chinese text and remove stopwords."""
    if not text or text.strip() == '':
        return []
    words = jieba.lcut(text)
    return [w.strip() for w in words if len(w.strip()) >= 2 and w.strip() not in STOPWORDS]

def safe_col(row, col_idx):
    """Safely get column value."""
    if col_idx < len(row):
        return row[col_idx].strip()
    return ''

def parse_tech_elements(text):
    """Parse JSON-like list of tech elements."""
    if not text or text.strip() == '':
        return []
    text = text.strip()
    # Try JSON parse
    try:
        items = json.loads(text)
        if isinstance(items, list):
            return [str(i).strip() for i in items if str(i).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    # Try manual parse: ["a", "b", "c"]
    m = re.findall(r'"([^"]+)"', text)
    if m:
        return [i.strip() for i in m if i.strip()]
    # Fallback: split by comma or newline
    items = re.split(r'[,，\n]', text)
    return [i.strip().strip('"').strip("'").strip('[]') for i in items if i.strip()]

def parse_tech_path(text):
    """Parse → separated technology pathway chains."""
    if not text or text.strip() == '':
        return []
    # Split by → or ->
    steps = re.split(r'\s*[→\->]+\s*', text)
    return [s.strip() for s in steps if s.strip()]


# ═══════════════════════════════════════════════════════════
# Analysis 1: TF-IDF Keyword Extraction
# ═══════════════════════════════════════════════════════════

def analyze_tfidf_keywords(cases, rows):
    """Extract top keywords via TF-IDF for each major text field."""
    print("\n" + "="*60)
    print("ANALYSIS 1: TF-IDF Keyword Extraction")
    print("="*60)

    fields = {
        'innovation_case': ('优势和创新点(案例级)', COL['innovation_case'], cases),
        'innovation_tool': ('优势和创新点(工具级)', COL['innovation_tool'], rows),
        'tech_path_case': ('关键技术路径(案例级)', COL['tech_path_case'], cases),
        'tech_path_tool': ('关键技术路径(工具级)', COL['tech_path_tool'], rows),
        'industry_status': ('产业应用现状', COL['industry_status_case'], cases),
        'potential_effect': ('潜在成效', COL['potential_effect'], rows),
        'main_scenario': ('主要应用场景', COL['main_scenario'], rows),
    }

    results = {}
    for key, (label, col_idx, data) in fields.items():
        texts = [safe_col(r, col_idx) for r in data if safe_col(r, col_idx)]
        if not texts:
            continue
        # Use jieba TF-IDF
        all_keywords = Counter()
        for t in texts:
            kws = jieba.analyse.extract_tags(t, topK=20, withWeight=True)
            for word, weight in kws:
                if len(word) >= 2 and word not in STOPWORDS:
                    all_keywords[word] += weight

        top50 = all_keywords.most_common(50)
        results[key] = {
            'label': label,
            'total_docs': len(texts),
            'top_keywords': [{'word': w, 'score': round(s, 4)} for w, s in top50]
        }
        print(f"\n  [{label}] ({len(texts)} docs)")
        print(f"  Top 15: {', '.join(w for w, _ in top50[:15])}")

    return results


# ═══════════════════════════════════════════════════════════
# Analysis 2: LDA Topic Modeling
# ═══════════════════════════════════════════════════════════

def analyze_lda_topics(cases, rows):
    """LDA topic modeling on innovation descriptions and tech paths."""
    print("\n" + "="*60)
    print("ANALYSIS 2: LDA Topic Modeling")
    print("="*60)

    results = {}

    for field_name, col_idx, data, n_topics in [
        ('innovation', COL['innovation_case'], cases, 7),
        ('tech_path', COL['tech_path_case'], cases, 6),
        ('potential_effect', COL['potential_effect'], rows, 5),
    ]:
        texts_raw = [safe_col(r, col_idx) for r in data if safe_col(r, col_idx)]
        if len(texts_raw) < 20:
            continue

        # Tokenize
        texts_tok = [' '.join(tokenize(t)) for t in texts_raw]
        texts_tok = [t for t in texts_tok if t.strip()]

        if len(texts_tok) < 20:
            continue

        vectorizer = CountVectorizer(max_features=2000, max_df=0.85, min_df=3)
        try:
            dtm = vectorizer.fit_transform(texts_tok)
        except ValueError:
            continue

        feature_names = vectorizer.get_feature_names_out()

        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42,
            max_iter=30, learning_method='online'
        )
        lda.fit(dtm)

        topics = []
        print(f"\n  [{field_name}] {n_topics} topics from {len(texts_tok)} docs:")
        for idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-15:][::-1]
            top_words = [(feature_names[i], round(float(topic[i]), 2)) for i in top_indices]
            topics.append({
                'topic_id': idx,
                'top_words': [{'word': w, 'weight': wt} for w, wt in top_words]
            })
            print(f"    Topic {idx}: {', '.join(w for w, _ in top_words[:8])}")

        # Get document-topic distribution
        doc_topics = lda.transform(dtm)
        topic_sizes = np.argmax(doc_topics, axis=1)
        topic_dist = Counter(int(t) for t in topic_sizes)

        results[field_name] = {
            'n_topics': n_topics,
            'n_docs': len(texts_tok),
            'topics': topics,
            'topic_distribution': {str(k): v for k, v in sorted(topic_dist.items())}
        }

    return results

# ═══════════════════════════════════════════════════════════
# Analysis 3: Word Co-occurrence Analysis
# ═══════════════════════════════════════════════════════════

def analyze_cooccurrence(cases):
    """Build co-occurrence matrix of top keywords from innovation text."""
    print("\n" + "="*60)
    print("ANALYSIS 3: Word Co-occurrence Analysis")
    print("="*60)

    texts = [safe_col(r, COL['innovation_case']) for r in cases if safe_col(r, COL['innovation_case'])]

    # Get top keywords first
    all_words = Counter()
    doc_words_list = []
    for t in texts:
        words = tokenize(t)
        unique_words = list(set(words))
        doc_words_list.append(unique_words)
        all_words.update(unique_words)

    top_words = [w for w, _ in all_words.most_common(80)]
    top_set = set(top_words)

    # Build co-occurrence
    cooccur = defaultdict(int)
    for doc_words in doc_words_list:
        filtered = [w for w in doc_words if w in top_set]
        for w1, w2 in combinations(sorted(set(filtered)), 2):
            cooccur[(w1, w2)] += 1

    # Top pairs
    top_pairs = sorted(cooccur.items(), key=lambda x: -x[1])[:100]

    print(f"  Top 20 co-occurring word pairs:")
    for (w1, w2), cnt in top_pairs[:20]:
        print(f"    {w1} -- {w2}: {cnt}")

    # Build adjacency for network
    nodes = set()
    edges = []
    for (w1, w2), cnt in top_pairs[:80]:
        nodes.add(w1)
        nodes.add(w2)
        edges.append({'source': w1, 'target': w2, 'weight': cnt})

    node_list = [{'id': w, 'freq': all_words[w]} for w in nodes]

    result = {
        'n_docs': len(texts),
        'top_keywords': [{'word': w, 'freq': c} for w, c in all_words.most_common(80)],
        'top_cooccurrence_pairs': [
            {'word1': w1, 'word2': w2, 'count': cnt} for (w1, w2), cnt in top_pairs
        ],
        'network': {'nodes': node_list, 'edges': edges}
    }
    return result


# ═══════════════════════════════════════════════════════════
# Analysis 4: KMeans Clustering of Cases
# ═══════════════════════════════════════════════════════════

def analyze_clusters(cases):
    """Cluster cases using TF-IDF vectors + KMeans."""
    print("\n" + "="*60)
    print("ANALYSIS 4: Case Clustering (TF-IDF + KMeans)")
    print("="*60)

    # Combine innovation + tech path for richer representation
    texts = []
    valid_cases = []
    for r in cases:
        combined = ' '.join([
            safe_col(r, COL['innovation_case']),
            safe_col(r, COL['tech_path_case']),
            safe_col(r, COL['potential_effect']),
        ])
        tokens = tokenize(combined)
        if len(tokens) >= 3:
            texts.append(' '.join(tokens))
            valid_cases.append(r)

    if len(texts) < 30:
        print("  Not enough data for clustering")
        return {}

    vectorizer = TfidfVectorizer(max_features=1500, max_df=0.8, min_df=3)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Try different k values
    best_k, best_score = 5, -1
    scores = {}
    for k in range(4, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(tfidf_matrix)
        if len(set(labels)) > 1:
            s = silhouette_score(tfidf_matrix, labels, sample_size=min(1000, len(texts)))
            scores[k] = round(s, 4)
            if s > best_score:
                best_score = s
                best_k = k

    print(f"  Silhouette scores: {scores}")
    print(f"  Best k={best_k} (score={best_score:.4f})")

    # Final clustering with best k
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(tfidf_matrix)

    # Characterize each cluster
    clusters = {}
    for c in range(best_k):
        mask = np.where(labels == c)[0]
        cluster_cases = [valid_cases[i] for i in mask]

        # Top terms for this cluster
        center = km.cluster_centers_[c]
        top_idx = center.argsort()[-20:][::-1]
        top_terms = [(feature_names[i], round(float(center[i]), 4)) for i in top_idx]

        # School level distribution
        levels = Counter(safe_col(r, COL['school_level']) for r in cluster_cases)
        # Subject distribution
        subjects = Counter()
        for r in cluster_cases:
            s = safe_col(r, COL['subject'])
            if s:
                subjects[s] += 1

        clusters[str(c)] = {
            'size': len(mask),
            'top_terms': [{'word': w, 'weight': wt} for w, wt in top_terms],
            'school_level_dist': dict(levels.most_common(10)),
            'subject_dist': dict(subjects.most_common(10)),
            'sample_cases': [safe_col(r, COL['case_name']) for r in cluster_cases[:5]],
        }
        print(f"\n  Cluster {c} ({len(mask)} cases):")
        print(f"    Keywords: {', '.join(w for w, _ in top_terms[:10])}")
        print(f"    Levels: {dict(levels.most_common(5))}")

    result = {
        'best_k': best_k,
        'silhouette_scores': scores,
        'best_silhouette': best_score,
        'n_cases': len(texts),
        'clusters': clusters,
    }
    return result

# ═══════════════════════════════════════════════════════════
# Analysis 5: Technology Pathway Pattern Analysis
# ═══════════════════════════════════════════════════════════

def analyze_tech_pathways(cases, rows):
    """Extract and analyze → separated technology pathway chains."""
    print("\n" + "="*60)
    print("ANALYSIS 5: Technology Pathway Patterns")
    print("="*60)

    # Case-level paths
    case_paths = []
    for r in cases:
        path_text = safe_col(r, COL['tech_path_case'])
        if path_text:
            steps = parse_tech_path(path_text)
            if len(steps) >= 2:
                case_paths.append(steps)

    # Tool-level paths
    tool_paths = []
    for r in rows:
        path_text = safe_col(r, COL['tech_path_tool'])
        if path_text:
            steps = parse_tech_path(path_text)
            if len(steps) >= 2:
                tool_paths.append(steps)

    # Full path frequency
    full_path_counter = Counter()
    for p in case_paths:
        full_path_counter[' → '.join(p)] += 1

    # Step frequency
    step_counter = Counter()
    for p in case_paths:
        for s in p:
            step_counter[s] += 1

    # Bigram transitions
    transition_counter = Counter()
    for p in case_paths:
        for i in range(len(p) - 1):
            transition_counter[(p[i], p[i+1])] += 1

    # Path length distribution
    length_dist = Counter(len(p) for p in case_paths)

    # Starting and ending steps
    start_counter = Counter(p[0] for p in case_paths if p)
    end_counter = Counter(p[-1] for p in case_paths if p)

    print(f"  Total case-level paths: {len(case_paths)}")
    print(f"  Total tool-level paths: {len(tool_paths)}")
    print(f"  Unique full paths: {len(full_path_counter)}")
    print(f"\n  Top 10 full pathways:")
    for path, cnt in full_path_counter.most_common(10):
        print(f"    [{cnt}] {path}")
    print(f"\n  Top 15 pathway steps:")
    for step, cnt in step_counter.most_common(15):
        print(f"    [{cnt}] {step}")
    print(f"\n  Top 10 transitions (A → B):")
    for (a, b), cnt in transition_counter.most_common(10):
        print(f"    [{cnt}] {a} → {b}")
    print(f"\n  Path length distribution: {dict(sorted(length_dist.items()))}")

    result = {
        'n_case_paths': len(case_paths),
        'n_tool_paths': len(tool_paths),
        'n_unique_paths': len(full_path_counter),
        'top_full_paths': [
            {'path': p, 'count': c} for p, c in full_path_counter.most_common(30)
        ],
        'step_frequency': [
            {'step': s, 'count': c} for s, c in step_counter.most_common(50)
        ],
        'transitions': [
            {'from': a, 'to': b, 'count': c}
            for (a, b), c in transition_counter.most_common(50)
        ],
        'path_length_distribution': {str(k): v for k, v in sorted(length_dist.items())},
        'starting_steps': [{'step': s, 'count': c} for s, c in start_counter.most_common(20)],
        'ending_steps': [{'step': s, 'count': c} for s, c in end_counter.most_common(20)],
    }
    return result


# ═══════════════════════════════════════════════════════════
# Analysis 6: Cultivation Direction (培养方向) Distribution
# ═══════════════════════════════════════════════════════════

def analyze_cultivation(rows):
    """Analyze cultivation direction distribution."""
    print("\n" + "="*60)
    print("ANALYSIS 6: Cultivation Direction (培养方向)")
    print("="*60)

    cult_counter = Counter()
    cult_by_level = defaultdict(Counter)
    cult_by_subject = defaultdict(Counter)
    cult_by_scenario = defaultdict(Counter)

    for r in rows:
        cult = safe_col(r, COL['cultivation'])
        if not cult:
            continue
        cult_counter[cult] += 1
        level = safe_col(r, COL['school_level'])
        if level:
            cult_by_level[level][cult] += 1
        subj = safe_col(r, COL['subject'])
        if subj:
            cult_by_subject[subj][cult] += 1
        sc = safe_col(r, COL['scenario_l1'])
        if sc:
            cult_by_scenario[sc][cult] += 1

    print(f"  Overall distribution:")
    for cult, cnt in cult_counter.most_common():
        print(f"    {cult}: {cnt} ({cnt/sum(cult_counter.values())*100:.1f}%)")

    result = {
        'overall': {k: v for k, v in cult_counter.most_common()},
        'by_school_level': {
            level: dict(counter.most_common())
            for level, counter in sorted(cult_by_level.items())
        },
        'by_subject': {
            subj: dict(counter.most_common())
            for subj, counter in sorted(cult_by_subject.items(), key=lambda x: -sum(x[1].values()))[:20]
        },
        'by_scenario_l1': {
            sc: dict(counter.most_common())
            for sc, counter in sorted(cult_by_scenario.items(), key=lambda x: -sum(x[1].values()))
        },
    }
    return result

# ═══════════════════════════════════════════════════════════
# Analysis 7: Technology Elements Extraction & Counting
# ═══════════════════════════════════════════════════════════

def analyze_tech_elements(cases, rows):
    """Extract and count technology elements from JSON-like lists."""
    print("\n" + "="*60)
    print("ANALYSIS 7: Technology Elements (技术要素)")
    print("="*60)

    # Case-level tech elements (col 9) - JSON lists
    case_elements = Counter()
    case_element_sets = []
    for r in cases:
        elems = parse_tech_elements(safe_col(r, COL['tech_elements_case']))
        case_elements.update(elems)
        case_element_sets.append(set(elems))

    # Tool-level tech elements (col 24) - layered text
    tool_elements = Counter()
    tool_layer_counter = Counter()
    for r in rows:
        text = safe_col(r, COL['tech_elements_tool'])
        if not text:
            continue
        # Parse layered format: "层名（技术1、技术2）"
        layers = text.split('\n')
        for layer in layers:
            layer = layer.strip()
            if not layer:
                continue
            # Extract layer name
            m = re.match(r'^([^（(]+)[（(]', layer)
            if m:
                layer_name = m.group(1).strip()
                tool_layer_counter[layer_name] += 1
            # Extract items in parentheses
            items_in_parens = re.findall(r'[（(]([^）)]+)[）)]', layer)
            for item_group in items_in_parens:
                items = re.split(r'[、,，;；]', item_group)
                for item in items:
                    item = item.strip()
                    if item and len(item) >= 2:
                        tool_elements[item] += 1

    # Co-occurrence of case-level elements
    elem_cooccur = Counter()
    for elem_set in case_element_sets:
        for e1, e2 in combinations(sorted(elem_set), 2):
            elem_cooccur[(e1, e2)] += 1

    # Element by school level
    elem_by_level = defaultdict(Counter)
    for r in cases:
        level = safe_col(r, COL['school_level'])
        elems = parse_tech_elements(safe_col(r, COL['tech_elements_case']))
        if level and elems:
            elem_by_level[level].update(elems)

    print(f"  Case-level unique elements: {len(case_elements)}")
    print(f"  Top 20 case-level elements:")
    for elem, cnt in case_elements.most_common(20):
        print(f"    [{cnt}] {elem}")

    print(f"\n  Tool-level technology layers:")
    for layer, cnt in tool_layer_counter.most_common(15):
        print(f"    [{cnt}] {layer}")

    print(f"\n  Top 15 tool-level sub-elements:")
    for elem, cnt in tool_elements.most_common(15):
        print(f"    [{cnt}] {elem}")

    print(f"\n  Top 10 element co-occurrences:")
    for (e1, e2), cnt in elem_cooccur.most_common(10):
        print(f"    [{cnt}] {e1} + {e2}")

    result = {
        'case_level_elements': [
            {'element': e, 'count': c} for e, c in case_elements.most_common(100)
        ],
        'tool_level_layers': [
            {'layer': l, 'count': c} for l, c in tool_layer_counter.most_common(30)
        ],
        'tool_level_sub_elements': [
            {'element': e, 'count': c} for e, c in tool_elements.most_common(100)
        ],
        'element_cooccurrence': [
            {'elem1': e1, 'elem2': e2, 'count': c}
            for (e1, e2), c in elem_cooccur.most_common(50)
        ],
        'elements_by_school_level': {
            level: [{'element': e, 'count': c} for e, c in counter.most_common(20)]
            for level, counter in sorted(elem_by_level.items())
        },
    }
    return result


# ═══════════════════════════════════════════════════════════
# Analysis 8: Technology-Subject Association Matrix
# ═══════════════════════════════════════════════════════════

def analyze_tech_subject_matrix(rows):
    """Build a technology-subject association matrix."""
    print("\n" + "="*60)
    print("ANALYSIS 8: Technology-Subject Association Matrix")
    print("="*60)

    # Collect tech elements per row with subject
    tech_subject = defaultdict(Counter)
    subject_counter = Counter()

    for r in rows:
        subj = safe_col(r, COL['subject'])
        if not subj:
            continue
        subject_counter[subj] += 1

        # From case-level tech elements
        elems = parse_tech_elements(safe_col(r, COL['tech_elements_case']))
        for e in elems:
            tech_subject[e][subj] += 1

    # Filter to top subjects and top elements
    top_subjects = [s for s, _ in subject_counter.most_common(20)]
    top_elements = [e for e, _ in sorted(tech_subject.items(),
                    key=lambda x: -sum(x[1].values()))[:30]]

    # Build matrix
    matrix = {}
    for elem in top_elements:
        matrix[elem] = {subj: tech_subject[elem].get(subj, 0) for subj in top_subjects}

    # Also build scenario-tech matrix
    tech_scenario = defaultdict(Counter)
    for r in rows:
        sc = safe_col(r, COL['scenario_l1'])
        if not sc:
            continue
        elems = parse_tech_elements(safe_col(r, COL['tech_elements_case']))
        for e in elems:
            tech_scenario[e][sc] += 1

    top_scenarios = sorted(set(
        sc for r in rows if (sc := safe_col(r, COL['scenario_l1']))
    ), key=lambda x: -sum(1 for r in rows if safe_col(r, COL['scenario_l1']) == x))[:15]

    scenario_matrix = {}
    for elem in top_elements:
        scenario_matrix[elem] = {sc: tech_scenario[elem].get(sc, 0) for sc in top_scenarios}

    print(f"  Matrix dimensions: {len(top_elements)} elements x {len(top_subjects)} subjects")
    print(f"  Top subjects: {', '.join(top_subjects[:10])}")
    print(f"  Top elements: {', '.join(top_elements[:10])}")

    # Print a summary of strongest associations
    print(f"\n  Strongest tech-subject associations:")
    assoc_list = []
    for elem in top_elements:
        for subj in top_subjects:
            val = matrix[elem].get(subj, 0)
            if val > 0:
                assoc_list.append((elem, subj, val))
    assoc_list.sort(key=lambda x: -x[2])
    for elem, subj, val in assoc_list[:15]:
        print(f"    {elem} -- {subj}: {val}")

    result = {
        'subjects': top_subjects,
        'tech_elements': top_elements,
        'tech_subject_matrix': matrix,
        'scenarios': top_scenarios,
        'tech_scenario_matrix': scenario_matrix,
        'top_associations': [
            {'element': e, 'subject': s, 'count': c}
            for e, s, c in assoc_list[:50]
        ],
    }
    return result

# ═══════════════════════════════════════════════════════════
# Analysis 9: Application Scenario Deep Analysis
# ═══════════════════════════════════════════════════════════

def analyze_scenarios(rows):
    """Deep analysis of application scenarios."""
    print("\n" + "="*60)
    print("ANALYSIS 9: Application Scenario Analysis")
    print("="*60)

    main_sc = Counter()
    sub_sc = Counter()
    l1_sc = Counter()
    l2_sc = Counter()
    l1_l2 = defaultdict(Counter)
    sc_by_level = defaultdict(Counter)
    sc_by_subject = defaultdict(Counter)

    for r in rows:
        ms = safe_col(r, COL['main_scenario'])
        ss = safe_col(r, COL['sub_scenario'])
        s1 = safe_col(r, COL['scenario_l1'])
        s2 = safe_col(r, COL['scenario_l2'])
        level = safe_col(r, COL['school_level'])
        subj = safe_col(r, COL['subject'])

        if ms: main_sc[ms] += 1
        if ss: sub_sc[ss] += 1
        if s1:
            l1_sc[s1] += 1
            if level: sc_by_level[level][s1] += 1
            if subj: sc_by_subject[subj][s1] += 1
        if s2: l2_sc[s2] += 1
        if s1 and s2: l1_l2[s1][s2] += 1

    print(f"  Main scenarios (top 15):")
    for sc, cnt in main_sc.most_common(15):
        print(f"    [{cnt}] {sc}")
    print(f"\n  L1 scenarios:")
    for sc, cnt in l1_sc.most_common():
        print(f"    [{cnt}] {sc}")
    print(f"\n  L2 scenarios (top 15):")
    for sc, cnt in l2_sc.most_common(15):
        print(f"    [{cnt}] {sc}")

    result = {
        'main_scenarios': {k: v for k, v in main_sc.most_common(50)},
        'sub_scenarios': {k: v for k, v in sub_sc.most_common(50)},
        'l1_scenarios': {k: v for k, v in l1_sc.most_common()},
        'l2_scenarios': {k: v for k, v in l2_sc.most_common(50)},
        'l1_l2_mapping': {
            l1: dict(counter.most_common(10))
            for l1, counter in sorted(l1_l2.items(), key=lambda x: -sum(x[1].values()))
        },
        'scenarios_by_school_level': {
            level: dict(counter.most_common(10))
            for level, counter in sorted(sc_by_level.items())
        },
        'scenarios_by_subject': {
            subj: dict(counter.most_common(10))
            for subj, counter in sorted(sc_by_subject.items(),
                                        key=lambda x: -sum(x[1].values()))[:15]
        },
    }
    return result

# ═══════════════════════════════════════════════════════════
# Analysis 10: Innovation Text Semantic Patterns
# ═══════════════════════════════════════════════════════════

def analyze_innovation_patterns(cases):
    """Extract semantic patterns from innovation descriptions."""
    print("\n" + "="*60)
    print("ANALYSIS 10: Innovation Semantic Patterns")
    print("="*60)

    # Collect innovation texts
    innov_texts = []
    for r in cases:
        t = safe_col(r, COL['innovation_case'])
        if t and len(t) > 10:
            innov_texts.append(t)

    # Pattern extraction: look for common structural patterns
    # e.g., "通过X实现Y", "将X转化为Y", "从X到Y"
    patterns = {
        'through_achieve': re.compile(r'通过(.{2,20})[，,]?\s*实现(.{2,30})'),
        'transform': re.compile(r'将(.{2,20})转化为(.{2,20})'),
        'from_to': re.compile(r'从(.{2,15})到(.{2,15})'),
        'break_through': re.compile(r'突破(.{2,20})'),
        'innovate': re.compile(r'创新(.{2,20})'),
        'integrate': re.compile(r'融合(.{2,20})'),
    }

    pattern_results = {}
    for pname, pat in patterns.items():
        matches = []
        for t in innov_texts:
            for m in pat.finditer(t):
                matches.append(m.group(0))
        pattern_results[pname] = Counter(matches).most_common(15)

    # Key action verbs in innovation
    action_verbs = ['实现', '突破', '构建', '打造', '融合', '赋能',
                    '驱动', '支撑', '优化', '重构', '变革', '转化',
                    '整合', '嵌入', '贯穿', '覆盖', '联动', '协同']
    verb_counts = Counter()
    for t in innov_texts:
        for v in action_verbs:
            verb_counts[v] += t.count(v)

    # Innovation keyword clusters via TF-IDF on innovation text
    innov_tokenized = [' '.join(tokenize(t)) for t in innov_texts if tokenize(t)]
    if len(innov_tokenized) > 20:
        vec = TfidfVectorizer(max_features=500, max_df=0.8, min_df=3)
        tfidf = vec.fit_transform(innov_tokenized)
        feature_names = vec.get_feature_names_out()
        # Average TF-IDF scores
        avg_tfidf = np.asarray(tfidf.mean(axis=0)).flatten()
        top_idx = avg_tfidf.argsort()[-30:][::-1]
        top_tfidf_words = [(feature_names[i], round(float(avg_tfidf[i]), 5)) for i in top_idx]
    else:
        top_tfidf_words = []

    print(f"  Innovation texts analyzed: {len(innov_texts)}")
    print(f"\n  Action verb frequency:")
    for v, c in verb_counts.most_common():
        if c > 0:
            print(f"    {v}: {c}")
    print(f"\n  Top TF-IDF innovation terms:")
    for w, s in top_tfidf_words[:15]:
        print(f"    {w}: {s:.5f}")

    result = {
        'n_texts': len(innov_texts),
        'action_verb_frequency': {v: c for v, c in verb_counts.most_common() if c > 0},
        'semantic_patterns': {
            pname: [{'text': t, 'count': c} for t, c in matches]
            for pname, matches in pattern_results.items()
        },
        'top_tfidf_terms': [{'word': w, 'score': s} for w, s in top_tfidf_words],
    }
    return result

# ═══════════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════════

def save_json(data, filename):
    """Save data as JSON."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w', encoding='utf-8-sig') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved: {path}")


def main():
    print("=" * 60)
    print("NLP/Semantic Analysis for AI Education Products")
    print(f"Data: {CSV_PATH}")
    print("=" * 60)

    header, rows = load_data()
    cases = get_case_level_data(rows)
    print(f"[DATA] {len(cases)} unique cases, {len(rows)} total rows")

    all_results = {}

    # 1. TF-IDF Keywords
    r1 = analyze_tfidf_keywords(cases, rows)
    save_json(r1, 'nlp_tfidf_keywords.json')
    all_results['tfidf_keywords'] = r1

    # 2. LDA Topics
    r2 = analyze_lda_topics(cases, rows)
    save_json(r2, 'nlp_lda_topics.json')
    all_results['lda_topics'] = r2

    # 3. Co-occurrence
    r3 = analyze_cooccurrence(cases)
    save_json(r3, 'nlp_cooccurrence.json')
    all_results['cooccurrence'] = r3

    # 4. Clustering
    r4 = analyze_clusters(cases)
    save_json(r4, 'nlp_clusters.json')
    all_results['clusters'] = r4

    # 5. Tech Pathways
    r5 = analyze_tech_pathways(cases, rows)
    save_json(r5, 'nlp_tech_pathways.json')
    all_results['tech_pathways'] = r5

    # 6. Cultivation Direction
    r6 = analyze_cultivation(rows)
    save_json(r6, 'nlp_cultivation.json')
    all_results['cultivation'] = r6

    # 7. Tech Elements
    r7 = analyze_tech_elements(cases, rows)
    save_json(r7, 'nlp_tech_elements.json')
    all_results['tech_elements'] = r7

    # 8. Tech-Subject Matrix
    r8 = analyze_tech_subject_matrix(rows)
    save_json(r8, 'nlp_tech_subject_matrix.json')
    all_results['tech_subject_matrix'] = r8

    # 9. Scenarios
    r9 = analyze_scenarios(rows)
    save_json(r9, 'nlp_scenarios.json')
    all_results['scenarios'] = r9

    # 10. Innovation Patterns
    r10 = analyze_innovation_patterns(cases)
    save_json(r10, 'nlp_innovation_patterns.json')
    all_results['innovation_patterns'] = r10

    # Save combined summary
    save_json(all_results, 'nlp_all_results.json')

    # ── Print Key Findings Summary ──
    print("\n" + "=" * 60)
    print("KEY FINDINGS SUMMARY")
    print("=" * 60)

    print(f"\n1. DATA SCOPE: {len(cases)} cases, {len(rows)} tool-rows")

    if r1:
        innov_kw = r1.get('innovation_case', {}).get('top_keywords', [])
        if innov_kw:
            print(f"\n2. TOP INNOVATION KEYWORDS:")
            print(f"   {', '.join(k['word'] for k in innov_kw[:10])}")

    if r2:
        for field, data in r2.items():
            n = data.get('n_topics', 0)
            print(f"\n3. LDA TOPICS ({field}): {n} topics identified")
            for t in data.get('topics', [])[:3]:
                words = ', '.join(w['word'] for w in t['top_words'][:6])
                print(f"   Topic {t['topic_id']}: {words}")

    if r4:
        print(f"\n4. CLUSTERING: {r4.get('best_k', '?')} clusters "
              f"(silhouette={r4.get('best_silhouette', 0):.3f})")
        for cid, cdata in list(r4.get('clusters', {}).items())[:4]:
            terms = ', '.join(w['word'] for w in cdata['top_terms'][:5])
            print(f"   Cluster {cid} ({cdata['size']} cases): {terms}")

    if r5:
        top_paths = r5.get('top_full_paths', [])
        if top_paths:
            print(f"\n5. TOP TECH PATHWAYS ({r5.get('n_unique_paths', 0)} unique):")
            for p in top_paths[:5]:
                print(f"   [{p['count']}] {p['path']}")

    if r6:
        print(f"\n6. CULTIVATION DIRECTION:")
        for k, v in list(r6.get('overall', {}).items())[:5]:
            print(f"   {k}: {v}")

    if r7:
        top_elems = r7.get('case_level_elements', [])
        if top_elems:
            print(f"\n7. TOP TECH ELEMENTS:")
            for e in top_elems[:8]:
                print(f"   [{e['count']}] {e['element']}")

    if r9:
        print(f"\n8. L1 SCENARIOS:")
        for k, v in list(r9.get('l1_scenarios', {}).items())[:6]:
            print(f"   {k}: {v}")

    print(f"\n{'='*60}")
    print(f"All results saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
