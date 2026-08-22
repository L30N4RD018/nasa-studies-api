"""
Módulo de corrección ortográfica y sugerencias para búsquedas.
Utiliza el corpus de términos científicos del dataset para detectar errores
y sugerir correcciones sin dependencias externas pesadas.
"""
from typing import List, Dict, Tuple, Set, Optional
import re
from collections import defaultdict, Counter


class SpellChecker:
    """
    Corrector ortográfico basado en el vocabulario del corpus.
    Usa distancia de edición y frecuencias de términos para sugerir correcciones.
    """
    
    def __init__(self, vocabulary: Dict[str, int], min_freq: int = 2):
        """
        Args:
            vocabulary: Dict {token: frequency} del corpus
            min_freq: Frecuencia mínima para considerar un término válido
        """
        self.vocabulary = {k: v for k, v in vocabulary.items() if v >= min_freq and len(k) >= 3}
        self.vocab_set = set(self.vocabulary.keys())
        
        # Construir índices para búsqueda rápida
        self._build_ngram_index()
    
    @staticmethod
    def _is_gibberish(word: str) -> bool:
        """
        Detecta si una palabra es ruido/basura (gibberish).
        Criterios:
        - Muy pocas vocales (< 20%)
        - Repetición excesiva de caracteres
        - Patrones aleatorios sin estructura
        """
        if len(word) < 3:
            return False
        
        word_lower = word.lower()
        
        # 1. Detectar palabras con muy pocas vocales
        vowels = set('aeiou')
        vowel_count = sum(1 for c in word_lower if c in vowels)
        vowel_ratio = vowel_count / len(word_lower)
        
        # Si tiene menos de 15% vocales, probablemente es ruido (ej: "asdasdas" = 42% OK, "xcvbnm" = 0%)
        if vowel_ratio < 0.15 and len(word_lower) > 4:
            return True
        
        # 2. Detectar repetición excesiva de secuencias (ej: "asdasdas" = "asd" x3)
        for seq_len in range(2, min(5, len(word_lower) // 2 + 1)):
            sequence = word_lower[:seq_len]
            if word_lower == sequence * (len(word_lower) // seq_len) + word_lower[:len(word_lower) % seq_len]:
                # La palabra es una repetición de una secuencia corta
                return True
        
        # 3. Detectar alternancias raras de consonantes (ej: "qwerty", "zxcvbn")
        # Contar transiciones consonante-consonante consecutivas
        consonants = set('bcdfghjklmnpqrstvwxyz')
        consecutive_consonants = 0
        max_consecutive = 0
        
        for i in range(len(word_lower)):
            if word_lower[i] in consonants:
                consecutive_consonants += 1
                max_consecutive = max(max_consecutive, consecutive_consonants)
            else:
                consecutive_consonants = 0
        
        # Si tiene 5+ consonantes seguidas, probablemente es ruido
        if max_consecutive >= 5:
            return True
        
        # 4. Detectar patrones de teclado (proximidad en QWERTY)
        keyboard_rows = [
            'qwertyuiop',
            'asdfghjkl',
            'zxcvbnm'
        ]
        
        # Si 80%+ de caracteres están en la misma fila del teclado, es sospechoso
        for row in keyboard_rows:
            chars_in_row = sum(1 for c in word_lower if c in row)
            if chars_in_row / len(word_lower) >= 0.8 and len(word_lower) >= 5:
                return True
        
        return False
        
    def _build_ngram_index(self):
        """Construye índice de bigramas para búsqueda rápida de candidatos"""
        self.bigram_index = defaultdict(set)
        for word in self.vocab_set:
            # Añadir bigramas con padding
            padded = f"^{word}$"
            for i in range(len(padded) - 1):
                bigram = padded[i:i+2]
                self.bigram_index[bigram].add(word)
    
    def _get_candidates(self, word: str, max_distance: int = 2) -> Set[str]:
        """Obtiene candidatos usando índice de bigramas"""
        if not word:
            return set()
        
        padded = f"^{word}$"
        bigrams = [padded[i:i+2] for i in range(len(padded) - 1)]
        
        # Palabras que comparten al menos algunos bigramas
        candidates = set()
        for bg in bigrams[:3]:  # Usar solo primeros bigramas para eficiencia
            candidates.update(self.bigram_index.get(bg, set()))
        
        # Filtrar por longitud similar (±2 caracteres)
        min_len = max(3, len(word) - max_distance)
        max_len = len(word) + max_distance
        candidates = {c for c in candidates if min_len <= len(c) <= max_len}
        
        return candidates
    
    @staticmethod
    def _levenshtein_distance(s1: str, s2: str, max_dist: int = 3) -> int:
        """
        Calcula distancia de Levenshtein (edición) con early stopping.
        Optimizada para strings científicos.
        """
        if s1 == s2:
            return 0
        
        len1, len2 = len(s1), len(s2)
        
        # Early exit si la diferencia de longitud ya supera max_dist
        if abs(len1 - len2) > max_dist:
            return max_dist + 1
        
        # Usar solo una fila de la matriz (optimización de espacio)
        if len1 > len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1
        
        current_row = range(len1 + 1)
        
        for i in range(1, len2 + 1):
            previous_row = current_row
            current_row = [i] + [0] * len1
            
            for j in range(1, len1 + 1):
                add = previous_row[j] + 1
                delete = current_row[j - 1] + 1
                change = previous_row[j - 1]
                if s1[j - 1] != s2[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)
            
            # Early stopping: si el mínimo en esta fila supera max_dist, abortar
            if min(current_row) > max_dist:
                return max_dist + 1
        
        return current_row[len1]
    
    def _similarity_score(self, word: str, candidate: str, word_freq: int) -> float:
        """
        Calcula score de similitud combinando distancia de edición y frecuencia.
        Score más alto = mejor candidato.
        """
        distance = self._levenshtein_distance(word.lower(), candidate.lower())
        
        if distance == 0:
            return 1.0
        
        # Penalización por distancia (exponencial)
        distance_score = 1.0 / (1.0 + distance ** 1.5)
        
        # Bonus por frecuencia (normalizado con log)
        import math
        freq_score = math.log(word_freq + 1) / 10.0
        
        # Bonus si comparten prefijo (importante para términos científicos)
        prefix_len = 0
        for i in range(min(len(word), len(candidate))):
            if word[i].lower() == candidate[i].lower():
                prefix_len += 1
            else:
                break
        prefix_bonus = prefix_len / max(len(word), len(candidate))
        
        # Score final ponderado
        final_score = (0.6 * distance_score) + (0.25 * freq_score) + (0.15 * prefix_bonus)
        
        return final_score
    
    def suggest(self, word: str, top_n: int = 5, min_score: float = 0.3, check_even_if_exists: bool = False) -> List[Dict[str, any]]:
        """
        Sugiere correcciones para una palabra.
        
        Args:
            word: Palabra a corregir
            top_n: Número máximo de sugerencias
            min_score: Score mínimo para considerar una sugerencia
            check_even_if_exists: Si True, sugiere incluso si la palabra existe (para detectar variantes comunes)
            
        Returns:
            Lista de dicts con {word, score, frequency, distance}
        """
        word_clean = word.lower().strip()
        
        # Obtener frecuencia de la palabra actual
        current_freq = self.vocabulary.get(word_clean, 0)
        
        # Si la palabra existe y es muy común (freq > 50), probablemente está bien
        if word_clean in self.vocab_set and current_freq > 50 and not check_even_if_exists:
            return []
        
        # Obtener candidatos
        candidates = self._get_candidates(word_clean, max_distance=2)
        
        # Si no hay candidatos cercanos, expandir búsqueda
        if not candidates:
            # Búsqueda más amplia: palabras que empiezan con las mismas 2-3 letras
            prefix = word_clean[:min(3, len(word_clean))]
            candidates = {w for w in self.vocab_set if w.startswith(prefix)}
        
        # Calcular scores
        suggestions = []
        for candidate in candidates:
            freq = self.vocabulary.get(candidate, 0)
            score = self._similarity_score(word_clean, candidate, freq)
            distance = self._levenshtein_distance(word_clean, candidate)
            
            # Solo sugerir si:
            # 1. El score es suficiente Y
            # 2. (El candidato es diferente a la palabra original O check_even_if_exists está activo) Y
            # 3. El candidato es significativamente más frecuente (al menos 3x) O la distancia es <=2
            if score >= min_score and candidate != word_clean:
                # Si la palabra actual existe, solo sugerir candidatos MUCHO más frecuentes
                if word_clean in self.vocab_set:
                    # Sugerir solo si el candidato es al menos 3x más frecuente
                    if freq > current_freq * 3 and distance <= 2:
                        suggestions.append({
                            'word': candidate,
                            'score': round(score, 3),
                            'frequency': freq,
                            'distance': distance
                        })
                else:
                    # Si no existe, sugerir normalmente
                    suggestions.append({
                        'word': candidate,
                        'score': round(score, 3),
                        'frequency': freq,
                        'distance': distance
                    })
        
        # Ordenar por score (descendente) y luego por frecuencia
        suggestions.sort(key=lambda x: (-x['score'], -x['frequency']))
        
        return suggestions[:top_n]
    
    def check_query(self, query: str) -> Dict[str, any]:
        """
        Analiza una query completa y devuelve correcciones sugeridas.
        Filtra automáticamente palabras basura/ruido (gibberish).
        
        Returns:
            {
                'original': query original,
                'has_errors': bool,
                'corrected_query': query corregida,
                'corrections': [{'original': ..., 'suggestions': [...]}],
                'tokens_analyzed': int,
                'tokens_corrected': int,
                'tokens_ignored': int,  # NUEVO
                'ignored_tokens': [...]  # NUEVO
            }
        """
        if not query:
            return {
                'original': '',
                'has_errors': False,
                'corrected_query': '',
                'corrections': [],
                'tokens_analyzed': 0,
                'tokens_corrected': 0,
                'tokens_ignored': 0,
                'ignored_tokens': []
            }
        
        # Tokenizar (similar a filters.py)
        tokens = re.split(r'\W+', query.lower())
        tokens = [t for t in tokens if len(t) > 2]
        
        corrections = []
        corrected_tokens = []
        ignored_tokens = []
        tokens_corrected = 0
        tokens_ignored = 0
        
        for token in tokens:
            # FILTRO DE RUIDO: Detectar y descartar gibberish
            if self._is_gibberish(token):
                ignored_tokens.append(token)
                tokens_ignored += 1
                continue  # NO incluir en la búsqueda
            
            suggestions = self.suggest(token, top_n=5, min_score=0.35, check_even_if_exists=True)
            
            if suggestions:
                # Palabra con error o con una variante más común
                corrections.append({
                    'original': token,
                    'suggestions': suggestions,
                    'best_correction': suggestions[0]['word']
                })
                corrected_tokens.append(suggestions[0]['word'])
                tokens_corrected += 1
            else:
                # Palabra correcta o sin sugerencias confiables
                corrected_tokens.append(token)
        
        corrected_query = ' '.join(corrected_tokens)
        has_errors = tokens_corrected > 0 or tokens_ignored > 0
        
        return {
            'original': query,
            'has_errors': has_errors,
            'corrected_query': corrected_query,
            'corrections': corrections,
            'tokens_analyzed': len(tokens),
            'tokens_corrected': tokens_corrected,
            'tokens_ignored': tokens_ignored,
            'ignored_tokens': ignored_tokens,
            'confidence': round(1.0 - ((tokens_corrected + tokens_ignored) / max(1, len(tokens))), 2)
        }


class QueryEnhancer:
    """
    Mejora queries con expansión de términos relacionados y sinónimos del dominio.
    """
    
    # Sinónimos y variaciones comunes en biología espacial
    DOMAIN_SYNONYMS = {
        'microgravity': ['weightlessness', 'zero-g', 'zerog', 'microg'],
        'spaceflight': ['space-flight', 'space flight', 'orbital', 'mission'],
        'astronaut': ['crew', 'crewmember', 'crew member'],
        'radiation': ['cosmic rays', 'ionizing', 'solar particle'],
        'gene': ['genetic', 'genomic', 'dna'],
        'protein': ['proteomic', 'proteomics'],
        'cell': ['cellular', 'cytoplasm'],
        'muscle': ['muscular', 'skeletal muscle', 'atrophy'],
        'bone': ['skeletal', 'osteo', 'calcium'],
        'immune': ['immunity', 'immunological', 'defense'],
        'bacteria': ['bacterial', 'microbe', 'microbial'],
        'plant': ['vegetation', 'crop', 'botanical'],
        'growth': ['development', 'proliferation'],
        'stress': ['stressor', 'response'],
    }
    
    def __init__(self, vocabulary: Dict[str, int]):
        self.vocabulary = vocabulary
        self.vocab_set = set(vocabulary.keys())
    
    def expand_query(self, query: str, max_expansions: int = 2) -> List[str]:
        """
        Expande query con sinónimos del dominio.
        
        Returns:
            Lista de términos adicionales relacionados
        """
        query_lower = query.lower()
        expansions = []
        
        for main_term, synonyms in self.DOMAIN_SYNONYMS.items():
            if main_term in query_lower:
                # Añadir sinónimos que existan en el vocabulario
                for syn in synonyms[:max_expansions]:
                    if syn in self.vocab_set:
                        expansions.append(syn)
        
        return expansions[:max_expansions * 2]  # Limitar expansiones totales
