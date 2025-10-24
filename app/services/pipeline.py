from typing import Dict, Any, Tuple
import pandas as pd
import time
from .filters import FilterEngine
from .ranking import rank_subset
from .generation import generate_title_and_description, tokenize, STOP
from collections import defaultdict
from datetime import datetime, timezone
# Build query_params reproducible
from urllib.parse import quote
import math
from collections import Counter

class PayloadBuilder:
    def __init__(self, studies_df: pd.DataFrame, id_col: str = 'Study Identifier'):
        self.studies_df = studies_df
        self.id_col = id_col
        # Precompute global token -> set(ids)
        self.global_token_studies = defaultdict(set)
        for _, r in studies_df.iterrows():
            sid = r[id_col]
            combo = ' '.join(str(r.get(c,'')) for c in ['Study Title','Study Description'])
            toks = set(t for t in tokenize(combo) if len(t)>3 and t not in STOP and not t.isdigit())
            for t in toks:
                self.global_token_studies[t].add(sid)
        self.global_token_freq = {k: len(v) for k,v in self.global_token_studies.items()}
        self.filter_engine = FilterEngine(studies_df, id_col, self.global_token_studies)
        # Simple in-memory cache (key tuple) para respuestas repetidas
        self._cache = {}
        # Track LLM status
        self._llm_loaded = False

    def compute_emerging_topics(self, subset_df, top_n=5, max_samples=4):
        if subset_df.empty:
            return []
        subset_token_studies = defaultdict(set)
        for _, r in subset_df.iterrows():
            sidv = r[self.id_col]
            combo = ' '.join(str(r.get(c,'')) for c in ['Study Title','Study Description'])
            toks = set(t for t in tokenize(combo) if len(t)>3 and t not in STOP and not t.isdigit())
            for t in toks:
                subset_token_studies[t].add(sidv)
        candidates = []
        for tok, sids in subset_token_studies.items():
            sub_occ = len(sids)
            glob_occ = self.global_token_freq.get(tok,0)
            if glob_occ == 0:
                continue
            if glob_occ <= 12 and sub_occ <= max(6, int(0.5*len(subset_df))):
                score = (sub_occ / glob_occ) * math.log(1+sub_occ) / (1+math.log(1+glob_occ))
                candidates.append((score, tok, sub_occ, glob_occ, sids))
        candidates.sort(reverse=True, key=lambda x: x[0])
        out = []
        for _, tok, sub_occ, glob_occ, sids in candidates[:top_n]:
            sample_rows = subset_df[subset_df[self.id_col].isin(list(sids))].head(max_samples)
            sample_items = [{'id': r[self.id_col], 'title': r.get('Study Title')} for _, r in sample_rows.iterrows()]
            out.append({
                'topic': tok,
                'subset_occurrences': int(sub_occ),
                'global_occurrences': int(glob_occ),
                'sample_studies': sample_items
            })
        return out

    def frequent_subset_tokens(self, ranked_df, limit=8):
        toks = []
        head_limit = min(120, len(ranked_df))
        for _, r in ranked_df.head(head_limit).iterrows():
            combo = ' '.join(str(r.get(c,'')) for c in ['Study Title','Study Description'])
            toks.extend([t for t in tokenize(combo) if len(t)>3 and t not in STOP and not t.isdigit()])
        cnt = Counter(toks)
        return [{'token': t, 'occurrences': int(c)} for t,c in cnt.most_common(limit)]

    def _cache_key(self, filters: Dict[str,Any], page:int, page_size:int, emerging_topics_n:int, compact: bool, use_llm: bool) -> Tuple:
        # Filters order independent key
        filt_items = tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k,v in filters.items()))
        return (filt_items, page, page_size, emerging_topics_n, compact, use_llm)

    def build_payload(self, filters: Dict[str,Any], page:int=1, page_size:int=20, emerging_topics_n=5, compact: bool=False, use_llm: bool=True) -> Dict[str,Any]:
        t0 = time.time()
        key = self._cache_key(filters, page, page_size, emerging_topics_n, compact, use_llm)
        if key in self._cache:
            out = self._cache[key].copy()
            out['debug']['cache_hit'] = True
            return out

        ids = self.filter_engine.filter_ids(filters)
        subset = self.studies_df[self.studies_df[self.id_col].isin(ids)].copy()
        ranked = rank_subset(subset, self.id_col)
        total = len(ranked)
        page_size = max(1, min(page_size, 200))
        total_pages = max(1, (total + page_size - 1)//page_size)
        page = max(1, min(page, total_pages))
        start = (page-1)*page_size
        end = start + page_size
        page_df = ranked.iloc[start:end]

        # Important / less relevant
        important = ranked.head(10)
        less_rel = ranked.tail(5).sort_values('rank_score') if total >=5 else ranked.tail(3)

        # Generate title & description - PASAR use_llm
        gen = generate_title_and_description(ranked, filters, use_llm=use_llm)

        # Emerging topics
        emerging = self.compute_emerging_topics(ranked, top_n=emerging_topics_n)
        frequent_subset = self.frequent_subset_tokens(ranked, limit=8)

        # Full data export
        full_records = []
        for _, r in ranked.iterrows():
            obj = {}
            for c,v in r.items():
                obj[c] = None if pd.isna(v) else v
            obj['rank_score'] = float(r.get('rank_score',0))
            full_records.append(obj)

        qp_parts = []
        for k in ('organism','project_type','keywords'):
            vals = filters.get(k) or []
            for v in vals:
                qp_parts.append(f"{quote(k)}={quote(str(v))}")
        if filters.get('q'):
            qp_parts.append(f"q={quote(filters['q'])}")
        query_params = '?' + '&'.join(qp_parts) if qp_parts else ''

        # Suggested keywords (top tokens in subset ranked part)
        suggested_kw = [d['token'] for d in frequent_subset[:12]]

        payload = {
            'filters': {
                'organism': filters.get('organism', []),
                'project_type': filters.get('project_type', []),
                'keywords': filters.get('keywords', []),
                'q': filters.get('q'),
                'q_mode': filters.get('q_mode','and'),
                'q_min_match': filters.get('q_min_match'),
                'query_params': query_params
            },
            'generated': gen,
            'counts': {
                'total_studies': int(ranked[self.id_col].nunique()),
                'important': int(len(important)),
                'less_relevant': int(len(less_rel))
            },
            'articles': {
                'important': [
                    {
                        'id': r[self.id_col],
                        'title': r.get('Study Title'),
                        'rank_score': float(r.get('rank_score',0)),
                        'organism': r.get('organism_label'),
                        'project_type': r.get('project_label'),
                        'release_date': str(r.get('release_date')),
                        'DOI': r.get('DOI'),
                        'url': r.get('url')
                    } for _, r in important.iterrows()
                ],
                'less_relevant': [
                    {
                        'id': r[self.id_col],
                        'title': r.get('Study Title'),
                        'rank_score': float(r.get('rank_score',0)),
                        'organism': r.get('organism_label'),
                        'project_type': r.get('project_label'),
                        'release_date': str(r.get('release_date')),
                        'DOI': r.get('DOI'),
                        'url': r.get('url')
                    } for _, r in less_rel.iterrows()
                ],
                'page_items': [
                    {
                        'id': r[self.id_col],
                        'title': r.get('Study Title'),
                        'rank_score': float(r.get('rank_score',0)),
                        'organism': r.get('organism_label'),
                        'project_type': r.get('project_label'),
                        'release_date': str(r.get('release_date')),
                        'DOI': r.get('DOI'),
                        'url': r.get('url')
                    } for _, r in page_df.iterrows()
                ]
            },
            'topics': {
                'emerging': emerging,
                'frequent_subset': frequent_subset,
                'by_topic_index': {et['topic']: [s['id'] for s in et['sample_studies']] for et in emerging}
            },
            'debug': {
                'ranking_preview': [ {'id': r[self.id_col], 'score': float(r.get('rank_score',0))} for _, r in ranked.head(20).iterrows() ],
                'llm_meta': gen.get('meta', {}),
                'generation_time_sec': 0.0  # se actualizará luego
            },
            'data': None if compact else {
                'studies_full': full_records,
                'total_full': len(full_records),
                'suggested_keywords': suggested_kw
            },
            'exported_at': datetime.now(timezone.utc).isoformat()
        }
        # Agregar estadísticas internas del motor de filtros si existen
        if hasattr(self.filter_engine, 'last_stats') and self.filter_engine.last_stats:
            payload['debug']['filter_stats'] = self.filter_engine.last_stats
        payload['debug']['generation_time_sec'] = round(time.time()-t0, 3)
        payload['debug']['cache_hit'] = False
        payload['debug']['studies_full_count'] = len(full_records)
        payload['debug']['fields_per_record'] = (len(full_records[0]) if full_records else 0)

        # Store in cache (shallow copy to prevent mutation issues)
        if len(self._cache) > 256:
            # simple eviction: clear all (could implement LRU)
            self._cache.clear()
        self._cache[key] = payload.copy()
        return payload