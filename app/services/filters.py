from typing import Dict, List, Any, Set
from .spell_checker import SpellChecker, QueryEnhancer
import pandas as pd

class FilterEngine:
    def __init__(self, studies_df: pd.DataFrame, id_col: str, global_token_studies):
        self.studies_df = studies_df
        self.id_col = id_col
        self.global_token_studies = global_token_studies
        self.last_stats: Dict[str,int] = {}
        
        # Inicializar corrector ortográfico con el vocabulario del corpus
        vocabulary_freq = {token: len(ids) for token, ids in global_token_studies.items()}
        self.spell_checker = SpellChecker(vocabulary_freq, min_freq=2)
        self.query_enhancer = QueryEnhancer(vocabulary_freq)
        
        # Stats para corrección ortográfica
        self.spell_corrections: Dict[str, Any] = {}

    @staticmethod
    def _normalize_token(t: str) -> str:
        return ''.join(c.lower() for c in t.strip() if c.isalnum() or c in ('-','_')).strip()

    def _tokens_from_text(self, q: str):
        if not q: return []
        import re
        toks = re.split(r'\W+', q.lower())
        return [self._normalize_token(tok) for tok in toks if len(tok) > 2]

    def filter_ids(self, filters: Dict[str, List[str]] | Dict[str, Any]) -> Set[str]:
        all_ids = set(self.studies_df[self.id_col].unique())
        working = all_ids.copy()
        self.last_stats = {'initial': len(working)}

        # organism
        orgs = [o for o in (filters.get('organism') or []) if o is not None]
        if orgs:
            target = {str(o).strip().lower() for o in orgs if str(o).strip()}
            candidate_cols = [c for c in ['organism_label','Organism','Organism Label','organism'] if c in self.studies_df.columns]
            collected_ids = set()
            for col in candidate_cols:
                tmp = self.studies_df[[col, self.id_col]].copy()
                tmp['__norm'] = tmp[col].astype(str).str.strip().str.lower()
                match_ids = set(tmp[tmp['__norm'].isin(target)][self.id_col])
                collected_ids |= match_ids
            if candidate_cols:
                working &= collected_ids if collected_ids else set()
            self.last_stats['after_organism'] = len(working)
            if not working:
                # debug: valores distintos disponibles
                if candidate_cols:
                    distinct_vals = []
                    for col in candidate_cols:
                        vals = list(self.studies_df[col].dropna().unique())[:40]
                        distinct_vals.extend([f"{col}::"+str(v) for v in vals])
                    self.last_stats['organism_distinct_sample'] = distinct_vals[:60]
                return working

        # project_type
        ptypes = [p for p in (filters.get('project_type') or []) if p is not None]
        if ptypes:
            targetp = {str(p).strip().lower() for p in ptypes if str(p).strip()}
            candidate_cols_pt = [c for c in ['project_label','Project Type','project_type','Project'] if c in self.studies_df.columns]
            collected_ids_pt = set()
            for col in candidate_cols_pt:
                tmp = self.studies_df[[col, self.id_col]].copy()
                tmp['__norm'] = tmp[col].astype(str).str.strip().str.lower()
                match_ids = set(tmp[tmp['__norm'].isin(targetp)][self.id_col])
                collected_ids_pt |= match_ids
            if candidate_cols_pt:
                working &= collected_ids_pt if collected_ids_pt else set()
            self.last_stats['after_project_type'] = len(working)
            if not working:
                if candidate_cols_pt:
                    distinct_vals = []
                    for col in candidate_cols_pt:
                        vals = list(self.studies_df[col].dropna().unique())[:40]
                        distinct_vals.extend([f"{col}::"+str(v) for v in vals])
                    self.last_stats['project_type_distinct_sample'] = distinct_vals[:60]
                return working

        # keywords (AND)
        sel_kws = filters.get('keywords') or []
        for kw in sel_kws:
            k = self._normalize_token(kw)
            if not k: continue
            matches = self.global_token_studies.get(k, set())
            if not matches:
                return set()
            working &= matches
            if not working: return working
        if sel_kws:
            self.last_stats['after_keywords'] = len(working)

        # free text q con corrección ortográfica y modo mejorado
        q_text = filters.get('q') or filters.get('query') or ''
        if q_text:
            original_working = working.copy()
            q_mode = (filters.get('q_mode') or 'and').lower()  # 'and' | 'or' | 'smart'
            
            # 🔥 NUEVA FUNCIONALIDAD: Análisis ortográfico
            spell_check_result = self.spell_checker.check_query(q_text)
            self.spell_corrections = spell_check_result
            
            # Si hay correcciones con alta confianza, usar query corregida
            use_corrected = False
            if spell_check_result['has_errors'] and spell_check_result['confidence'] < 0.7:
                # Hay errores significativos, intentar con query corregida
                corrected_query = spell_check_result['corrected_query']
                self.last_stats['spell_correction_applied'] = True
                self.last_stats['original_query'] = q_text
                self.last_stats['corrected_query'] = corrected_query
                # Intentar primero con query corregida
                q_text_to_use = corrected_query
                use_corrected = True
            else:
                q_text_to_use = q_text
            
            terms = self._tokens_from_text(q_text_to_use)
            
            # Filtrar tokens que existen vs. los que faltan
            existing_terms = [t for t in terms if self.global_token_studies.get(t)]
            missing_terms = [t for t in terms if t not in existing_terms]
            
            # Si hay términos faltantes, intentar corregirlos individualmente
            if missing_terms and not use_corrected:
                corrected_missing = []
                for mterm in missing_terms:
                    suggestions = self.spell_checker.suggest(mterm, top_n=1, min_score=0.5)
                    if suggestions:
                        corrected_missing.append({
                            'original': mterm,
                            'corrected': suggestions[0]['word'],
                            'confidence': suggestions[0]['score']
                        })
                        # Añadir término corregido a existing_terms
                        existing_terms.append(suggestions[0]['word'])
                
                if corrected_missing:
                    self.last_stats['individual_corrections'] = corrected_missing
            
            if missing_terms:
                self.last_stats['ignored_tokens'] = missing_terms
            
            # 🎯 LÓGICA MEJORADA DE q_mode
            # Si no hay términos válidos, fallback a búsqueda de frase
            executed_phrase_fallback = False
            if not existing_terms:
                phrase_ids = self._phrase_match_ids(q_text)
                working &= phrase_ids if phrase_ids else set()
                executed_phrase_fallback = True
                self.last_stats['search_strategy'] = 'phrase_fallback_no_terms'
            else:
                # Tenemos términos válidos, aplicar estrategia según q_mode
                
                if q_mode == 'smart':
                    # 🧠 MODO SMART: Decide automáticamente basado en el contexto
                    # - Si hay filtros de organism/project_type activos: usa AND (más específico)
                    # - Si NO hay filtros: usa OR con min_match inteligente
                    has_filters = bool(filters.get('organism') or filters.get('project_type') or filters.get('keywords'))
                    
                    if has_filters:
                        # Con filtros activos: AND para refinar aún más
                        working = self._apply_and_search(existing_terms, working, original_working)
                        self.last_stats['search_strategy'] = 'smart_and_with_filters'
                    else:
                        # Sin filtros: OR con min_match para amplitud
                        import math
                        min_match = max(1, math.ceil(0.6 * len(existing_terms)))
                        working = self._apply_or_search(existing_terms, working, min_match)
                        self.last_stats['search_strategy'] = f'smart_or_no_filters_min{min_match}'
                
                elif q_mode == 'or':
                    # 🔵 MODO OR: Busca cualquier término
                    min_match = filters.get('q_min_match') or 1
                    working = self._apply_or_search(existing_terms, working, min_match)
                    self.last_stats['search_strategy'] = f'explicit_or_min{min_match}'
                
                else:  # AND (default)
                    # 🟢 MODO AND: Busca todos los términos con fallbacks inteligentes
                    working = self._apply_and_search(existing_terms, working, original_working)
                    self.last_stats['search_strategy'] = 'explicit_and_with_fallbacks'
            
            # Fallback final: fuzzy match si seguimos vacíos
            if q_text and not working:
                fuzzy_ids = self._fuzzy_title_match_ids(q_text, top_k=25, threshold=0.72)
                if fuzzy_ids:
                    working = fuzzy_ids
                    self.last_stats['fuzzy_fallback'] = True
                    self.last_stats['search_strategy'] = 'fuzzy_similarity'
            
            self.last_stats['after_q'] = len(working)
            if executed_phrase_fallback:
                self.last_stats['phrase_fallback'] = True

        self.last_stats['final'] = len(working)
        return working

    def _apply_and_search(self, terms: List[str], working: Set[str], original_working: Set[str]) -> Set[str]:
        """
        Aplica búsqueda AND con fallbacks inteligentes.
        
        Estrategia:
        1. Intentar AND estricto (todos los términos)
        2. Si vacío: AND parcial (al menos 70% de términos)
        3. Si vacío: AND parcial (al menos 50% de términos)
        4. Si vacío: búsqueda de frase
        """
        result = working.copy()
        
        # Intento 1: AND estricto
        for term in terms:
            matches = self.global_token_studies.get(term, set())
            result &= matches
            if not result:
                break
        
        if result:
            self.last_stats['and_strategy'] = 'strict_all_terms'
            return result
        
        # Intento 2: AND parcial (70%)
        import math
        min_match_70 = max(1, math.ceil(0.7 * len(terms)))
        result = self._apply_or_search(terms, working, min_match_70)
        
        if result:
            self.last_stats['and_strategy'] = f'partial_70pct_min{min_match_70}'
            self.last_stats['and_fallback_level'] = 1
            return result
        
        # Intento 3: AND parcial (50%)
        min_match_50 = max(1, math.ceil(0.5 * len(terms)))
        result = self._apply_or_search(terms, working, min_match_50)
        
        if result:
            self.last_stats['and_strategy'] = f'partial_50pct_min{min_match_50}'
            self.last_stats['and_fallback_level'] = 2
            return result
        
        # Último recurso: búsqueda OR con min_match=1
        result = self._apply_or_search(terms, working, 1)
        if result:
            self.last_stats['and_strategy'] = 'fallback_or_any'
            self.last_stats['and_fallback_level'] = 3
        
        return result
    
    def _apply_or_search(self, terms: List[str], working: Set[str], min_match: int = 1) -> Set[str]:
        """
        Aplica búsqueda OR con mínimo de coincidencias.
        
        Args:
            terms: Lista de términos a buscar
            working: Set de IDs donde buscar
            min_match: Mínimo de términos que deben coincidir
        """
        from collections import Counter
        
        term_sets = [self.global_token_studies.get(t, set()) for t in terms]
        counter = Counter()
        
        for tset in term_sets:
            for sid in tset:
                counter[sid] += 1
        
        # Filtrar por min_match y por working set
        valid = {sid for sid, count in counter.items() if count >= min_match}
        result = working & valid
        
        self.last_stats['or_min_match_used'] = min_match
        self.last_stats['or_candidates_found'] = len(valid)
        
        return result
    
    # --- Métodos auxiliares de fallback ---
    def _phrase_match_ids(self, phrase: str) -> Set[str]:
        phrase_norm = phrase.lower().strip()
        if len(phrase_norm) < 4:
            return set()
        cols = [c for c in ['Study Title','Study Description'] if c in self.studies_df.columns]
        if not cols:
            return set()
        matched = set()
        for _, r in self.studies_df[[self.id_col] + cols].iterrows():
            for c in cols:
                val = str(r.get(c,'')).lower()
                if phrase_norm in val:
                    matched.add(r[self.id_col])
                    break
        return matched

    def _fuzzy_title_match_ids(self, text: str, top_k: int = 15, threshold: float = 0.7) -> Set[str]:
        # Implementación ligera con difflib (sin dependencias extra). Si no supera threshold no devuelve nada.
        import difflib
        candidate_pairs = []
        base = text.lower().strip()
        if len(base) < 4:
            return set()
        if 'Study Title' not in self.studies_df.columns:
            return set()
        for _, r in self.studies_df[['Study Title', self.id_col]].iterrows():
            title = str(r['Study Title']).lower()
            ratio = difflib.SequenceMatcher(None, base, title).ratio()
            if ratio >= threshold:
                candidate_pairs.append((ratio, r[self.id_col]))
        candidate_pairs.sort(reverse=True, key=lambda x: x[0])
        top = candidate_pairs[:top_k]
        return {cid for _, cid in top}
