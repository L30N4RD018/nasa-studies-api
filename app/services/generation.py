from typing import Dict, Any, List, Optional, Tuple
import re
import os
import time
from dotenv import load_dotenv

load_dotenv()

GGUF_MODEL_PATH = os.getenv('GGUF_MODEL_PATH', '/app/app/models/odr_model_q5_k_m.gguf')

LLM_AVAILABLE = None
_llm_instance = None

try:
    from llama_cpp import Llama
    LLM_AVAILABLE = 'llama-cpp-python'
    print(f"✅ llama-cpp-python disponible")
except ImportError:
    LLM_AVAILABLE = None
    print(f"⚠️ llama-cpp-python no disponible - modo heurístico solamente")

STOP = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
    'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
}

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if len(t) > 2]

def load_llm_model():
    global _llm_instance
    
    if _llm_instance is not None:
        return _llm_instance
    
    if LLM_AVAILABLE != 'llama-cpp-python':
        print("❌ llama-cpp-python no está disponible")
        return None
    
    if not os.path.exists(GGUF_MODEL_PATH):
        print(f"❌ Modelo no encontrado: {GGUF_MODEL_PATH}")
        return None
    
    print(f"📦 Cargando modelo GGUF desde {GGUF_MODEL_PATH}...")
    
    file_size_mb = os.path.getsize(GGUF_MODEL_PATH) / (1024 * 1024)
    print(f"   Tamaño: {file_size_mb:.1f} MB")
    
    try:
        from llama_cpp import Llama
        
        _llm_instance = Llama(
            model_path=GGUF_MODEL_PATH,
            n_ctx=2048,
            n_threads=8,
            n_batch=2048,
            verbose=False,
            n_gpu_layers=0,
            use_mlock=True,
            use_mmap=True
        )
        
        print("✅ Modelo cargado exitosamente")
        return _llm_instance
            
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        import traceback
        traceback.print_exc()
        return None

def mmr_summary(sentences: List[str], target=160) -> str:
    if not sentences:
        return ''
    
    selected = []
    total_words = 0
    used_terms = set()
    
    for sentence in sentences:
        words = sentence.lower().split()
        word_set = set(words)
        new_words = word_set - used_terms
        novelty = len(new_words) / max(len(words), 1)
        
        if not selected or novelty > 0.3:
            if total_words + len(words) > target:
                break
            selected.append(sentence)
            total_words += len(words)
            used_terms.update(word_set)
        
        if len(selected) >= 5:
            break
    
    return ' '.join(selected)

def build_sentences(df) -> List[str]:
    sentences = []
    for _, row in df.head(60).iterrows():
        desc = row.get('Study Description', '') or ''
        if desc:
            for sent in re.split(r'(?<=[.!?])\s+', str(desc)):
                sent = sent.strip()
                if 40 < len(sent) < 400:
                    sentences.append(sent)
    return sentences

def extract_key_themes(df, max_themes=3) -> List[str]:
    keywords = [
        'gene', 'protein', 'cell', 'expression', 'response', 
        'development', 'stress', 'growth', 'pathway', 'regulation',
        'microgravity', 'radiation', 'spaceflight', 'adaptation',
        'transcriptome', 'metabolome', 'genome', 'phenotype'
    ]
    
    keyword_counts = {kw: 0 for kw in keywords}
    
    for _, row in df.head(20).iterrows():
        text = (str(row.get('Study Title', '')) + ' ' + 
                str(row.get('Study Description', ''))).lower()
        for keyword in keywords:
            if keyword in text:
                keyword_counts[keyword] += 1
    
    sorted_themes = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    themes = [theme for theme, count in sorted_themes[:max_themes] if count > 0]
    return themes

def title_heuristic(filters: Dict[str, List[str]], df) -> str:
    organisms = filters.get('organism', [])
    project_types = filters.get('project_type', [])
    keywords = filters.get('keywords', [])
    q = filters.get('q', '')
    
    parts = []
    if q:
        parts.append(q[:50])
    if organisms:
        org_str = ', '.join(organisms[:2])
        if len(organisms) > 2:
            org_str += f" and {len(organisms) - 2} others"
        parts.append(org_str)
    if keywords:
        kw_str = ', '.join(keywords[:2])
        parts.append(f"with {kw_str}")
    if project_types:
        proj_str = ', '.join(project_types[:2])
        parts.append(f"({proj_str})")
    if not parts:
        parts.append(f"{len(df)} Omics Studies")
    
    title = ' '.join(parts)
    if len(title) > 100:
        title = title[:97] + "..."
    return title

def enhance_description(base_description: str, df, filters: Dict) -> str:
    num_studies = len(df)
    organisms = filters.get('organism', [])
    themes = extract_key_themes(df)
    
    parts = [base_description]
    if num_studies > 1:
        parts.append(f"This analysis encompasses {num_studies} studies")
        if organisms:
            parts.append(f"focusing on {', '.join(organisms[:3])}")
    if themes:
        parts.append(f"Key research areas include {', '.join(themes)}")
    
    enhanced = '. '.join(parts)
    if not enhanced.endswith('.'):
        enhanced += '.'
    return enhanced

def is_text_truncated(text: str) -> bool:
    """Detecta si el texto está truncado de forma no natural."""
    if not text:
        return False
    
    text = text.strip()
    
    # Si termina con puntuación válida, probablemente no está truncado
    if text[-1] in '.!?':
        return False
    
    # Si la última palabra es muy corta y no hay puntuación, probablemente está truncado
    words = text.split()
    if words:
        last_word = words[-1]
        # Palabra corta sin puntuación = posible truncamiento
        if len(last_word) < 4 and not any(p in last_word for p in '.,;:!?'):
            return True
    
    return False

def clean_truncated_text(text: str) -> str:
    """Limpia texto truncado eliminando la última frase incompleta."""
    if not text or not is_text_truncated(text):
        return text
    
    # Buscar el último delimitador de frase válido
    last_punct_pos = -1
    for i in range(len(text) - 1, -1, -1):
        if text[i] in '.!?':
            last_punct_pos = i
            break
    
    # Si encontramos puntuación, cortar ahí
    if last_punct_pos > len(text) * 0.3:  # Al menos 30% del texto original
        return text[:last_punct_pos + 1].strip()
    
    # Si no, devolver el texto original (mejor truncado que vacío)
    return text

def generate_with_gguf(ranked_df, filters: Dict[str, List[str]]) -> Optional[Dict[str, Any]]:
    """
    Generación mejorada con LLM local GGUF.
    Genera descripciones completas (200-800 palabras) sin truncamiento artificial.
    
    Parámetros optimizados para modelo de 800MB:
    - max_tokens: 1000 (suficiente para descripciones completas)
    - temperature: 0.6 (balance entre creatividad y coherencia)
    """
    if LLM_AVAILABLE != 'llama-cpp-python':
        return None
    
    llm = load_llm_model()
    if llm is None:
        return None
    
    try:
        organisms = filters.get('organism', [])
        project_types = filters.get('project_type', [])
        keywords = filters.get('keywords', [])
        q = filters.get('q', '')
        
        # Contexto compacto de estudios
        studies = []
        for idx, (_, row) in enumerate(ranked_df.head(3).iterrows(), 1):
            title = row.get('Study Title', 'N/A')[:90]
            studies.append(f"{idx}. {title}")
        
        # Contexto de filtros
        context_parts = []
        if organisms:
            context_parts.append(f"{', '.join(organisms[:2])}")
        if project_types:
            context_parts.append(f"{', '.join(project_types[:2])}")
        if q:
            context_parts.append(f"{q[:60]}")
        
        context = ' - '.join(context_parts) if context_parts else 'omics studies'
        
        # Prompt optimizado para descripciones completas
        prompt = f"""Summarize these {len(ranked_df)} scientific studies about {context}.

Top studies:
{chr(10).join(studies)}

Write a professional summary following this format:

TITLE: [One clear, descriptive sentence (50-100 characters)]

DESCRIPTION: [Write 4-6 complete sentences (200-400 words). Include: (1) main research focus, (2) organisms/systems studied, (3) key methodologies or approaches, (4) scientific significance and findings. Use complete sentences with proper punctuation. Be thorough and detailed.]

TITLE:"""
        
        print("🤖 Generando con modelo GGUF...")
        start = time.time()
        
        # Stop sequences optimizadas
        stop_sequences = [
            "\n\n\n\n",  # Cuádruple newline
            "</s>",
            "<|endoftext|>",
            "<|end|>",
            "###",
            "---",
            "Top studies:",
            "\n\nWrite a professional",
            "\n\nSUMMARY:",
            "\n\nTITLE:",
            "User:",
            "Assistant:"
        ]
        
        # Parámetros optimizados para modelo de 800MB
        max_tokens = 1000  # ← CORREGIDO: Aumentado de 350 a 1000
        temperature = 0.6
        
        # Generar con parámetros balanceados
        result = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.85,
            top_k=40,
            repeat_penalty=1.1,
            stop=stop_sequences,
            echo=False
        )
        
        gen_time = time.time() - start
        generated_text = result['choices'][0]['text'].strip()
        tokens_used = result['usage']['completion_tokens']
        finish_reason = result['choices'][0].get('finish_reason', 'unknown')
        
        print(f"⏱️  Generado en {gen_time:.2f}s ({tokens_used} tokens)")
        print(f"📊 Finish reason: {finish_reason}")
        
        # Mostrar preview del output
        preview_len = min(400, len(generated_text))
        print(f"📄 OUTPUT CRUDO ({len(generated_text)} chars total, preview {preview_len} chars):")
        print(f"{generated_text[:preview_len]}{'...' if len(generated_text) > preview_len else ''}")
        print("---")
        
        # Parsear el resultado
        title, description = parse_llm_output("TITLE:" + generated_text)
        
        # VALIDACIÓN FLEXIBLE - descripciones más largas permitidas
        title_valid = title and 15 <= len(title) <= 300
        # Descripción: mínimo 100 chars, máximo 4000 chars
        desc_valid = description and 100 <= len(description) <= 4000
        
        print(f"📌 Título: '{title[:70]}...' ({len(title) if title else 0} chars) - {'✅' if title_valid else '❌'}")
        print(f"📝 Descripción: {len(description) if description else 0} chars - {'✅' if desc_valid else '❌'}")
        
        if not title_valid:
            print(f"   ⚠️  Título inválido: longitud fuera de rango (15-300)")
        
        if not desc_valid:
            if description:
                print(f"   ⚠️  Descripción inválida: longitud fuera de rango (100-4000)")
                print(f"   📏 Recibida: {len(description)} chars")
            else:
                print(f"   ⚠️  Descripción vacía")
        
        # Expandir descripción MUY corta
        if title_valid and description and len(description) < 100:
            print(f"   🔧 Expandiendo descripción corta...")
            desc_parts = [description]
            desc_parts.append(f"This analysis includes {len(ranked_df)} studies")
            if organisms:
                desc_parts.append(f"focusing on {', '.join(organisms[:2])}")
            if project_types:
                desc_parts.append(f"from {', '.join(project_types[:2])} research")
            description = '. '.join(desc_parts) + '.'
            desc_valid = len(description) >= 100
            print(f"   ✅ Descripción expandida: {len(description)} chars")
        
        # Limitar título si es muy largo
        if title and len(title) > 250:
            # Buscar último espacio antes de 250
            truncate_pos = title[:247].rfind(' ')
            if truncate_pos > 200:
                title = title[:truncate_pos] + '...'
            else:
                title = title[:247] + '...'
            print(f"   ✂️  Título truncado a {len(title)} chars")
        
        # NO truncar descripción - enviar completa al frontend
        
        # Retornar si es válido
        if title_valid and desc_valid:
            # Calcular palabras aproximadas
            word_count = len(description.split())
            
            print(f"✅ Generación exitosa:")
            print(f"   📌 Título: {len(title)} chars")
            print(f"   📝 Descripción: {len(description)} chars (~{word_count} palabras)")
            
            return {
                'title': title,
                'description': description,  # COMPLETA sin truncar
                'generation_time': gen_time,
                'tokens_used': tokens_used,
                'model': 'gguf_local',
                'finish_reason': finish_reason,
                'stats': {
                    'title_length': len(title),
                    'description_length': len(description),
                    'word_count': word_count
                },
                'raw_output_preview': generated_text[:300]
            }
        else:
            print(f"❌ Validación fallida - intentando fallback...")
            
            # FALLBACK inteligente
            if generated_text and len(generated_text) > 30:
                # Extraer título si no es válido
                if not title_valid:
                    lines = [l.strip() for l in generated_text.split('\n') if l.strip()]
                    for line in lines[:3]:
                        clean_line = re.sub(r'^(TITLE|DESCRIPTION)\s*:\s*', '', line, flags=re.IGNORECASE)
                        if 15 <= len(clean_line) <= 300:
                            title = clean_line
                            title_valid = True
                            print(f"   ✓ Título extraído del fallback: '{title[:60]}...'")
                            break
                
                # Construir descripción si no es válida
                if title_valid and not desc_valid:
                    # Tomar todo el texto generado después del título
                    desc_text = generated_text
                    # Remover título si está presente
                    if title in desc_text:
                        desc_text = desc_text.replace(title, '', 1)
                    # Limpiar marcadores
                    desc_text = re.sub(r'^(TITLE|DESCRIPTION)\s*:\s*', '', desc_text, flags=re.IGNORECASE | re.MULTILINE)
                    desc_text = desc_text.strip()
                    
                    if len(desc_text) >= 100:
                        description = desc_text
                        desc_valid = True
                        print(f"   ✓ Descripción extraída del fallback: {len(description)} chars")
                
                if title_valid and desc_valid:
                    word_count = len(description.split())
                    print(f"✅ Generación exitosa con fallback")
                    print(f"   📌 Título: {len(title)} chars")
                    print(f"   📝 Descripción: {len(description)} chars (~{word_count} palabras)")
                    
                    return {
                        'title': title,
                        'description': description,  # COMPLETA
                        'generation_time': gen_time,
                        'tokens_used': tokens_used,
                        'model': 'gguf_local',
                        'finish_reason': finish_reason,
                        'fallback_used': True,
                        'stats': {
                            'title_length': len(title),
                            'description_length': len(description),
                            'word_count': word_count
                        }
                    }
            
            return None
    
    except Exception as e:
        print(f"❌ Error con modelo GGUF: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_llm_output(text: str) -> Tuple[str, str]:
    """
    Parseo robusto del output del LLM.
    NO trunca - devuelve la descripción completa.
    Retorna (title, description)
    """
    if not text:
        return "", ""
    
    text = text.strip()
    
    # Método 1: Parseo con regex
    # Buscar TITLE hasta DESCRIPTION o doble newline
    title_match = re.search(
        r'TITLE\s*:\s*(.+?)(?=\n\s*DESCRIPTION|\n\n|$)', 
        text, 
        re.IGNORECASE | re.DOTALL
    )
    
    # Buscar DESCRIPTION hasta el final (SIN límite de caracteres)
    desc_match = re.search(
        r'DESCRIPTION\s*:\s*(.+?)$',  # Hasta el final del texto
        text, 
        re.IGNORECASE | re.DOTALL
    )
    
    title = ""
    description = ""
    
    if title_match:
        title = clean_text(title_match.group(1))
    
    if desc_match:
        description = clean_text(desc_match.group(1))
        # Solo limpiar excesos obvios (múltiples espacios, etc.)
        # NO truncar por longitud
    
    if title and description:
        return title, description
    
    # Método 2: Parseo por líneas (fallback)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    title_fb = ""
    desc_lines = []
    found_desc = False
    
    for i, line in enumerate(lines):
        line_upper = line.upper()
        
        if 'TITLE:' in line_upper and not title_fb:
            title_fb = line.split(':', 1)[-1].strip()
            title_fb = clean_text(title_fb)
        
        elif 'DESCRIPTION:' in line_upper and not found_desc:
            # Capturar resto de la línea
            desc_start = line.split(':', 1)[-1].strip()
            if desc_start:
                desc_lines.append(desc_start)
            
            # Capturar TODAS las líneas siguientes (sin límite)
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                # Solo detenerse en marcadores obvios de nuevo contenido
                if any(marker in next_line.upper() for marker in [
                    'TITLE:', '###', 'TOP STUDIES:', 'WRITE A PROFESSIONAL'
                ]):
                    break
                desc_lines.append(next_line)
            
            found_desc = True
            break
    
    if desc_lines:
        description_fb = ' '.join(desc_lines)
        description_fb = clean_text(description_fb)
        # NO truncar
    else:
        description_fb = ""
    
    # Usar fallback si el método 1 no funcionó
    if not title and title_fb:
        title = title_fb
    if not description and description_fb:
        description = description_fb
    
    return title, description


def clean_text(text: str) -> str:
    """Limpia y normaliza texto generado SIN truncar."""
    if not text:
        return ""
    
    # Eliminar comillas al inicio y final
    text = re.sub(r'^["\'\`]+|["\'\`]+$', '', text)
    
    # Normalizar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    
    # Eliminar prefijos residuales
    text = re.sub(r'^(TITLE|DESCRIPTION)\s*:\s*', '', text, flags=re.IGNORECASE)
    
    # NO aplicar clean_truncated_text - mantener texto completo
    
    return text.strip()

def generate_title_and_description(ranked_df, filters: Dict[str, List[str]], 
                                   use_llm: bool = True) -> Dict[str, Any]:
    """
    Genera título y descripción usando LLM o fallback heurístico.
    """
    meta = {
        'mode': 'unknown',
        'llm_used': False,
        'generation_time': 0,
        'final_source': 'heuristic',
        'fallback_chain': []
    }
    
    start_time = time.time()
    
    # Intentar con LLM si está disponible
    if use_llm and LLM_AVAILABLE == 'llama-cpp-python':
        meta['fallback_chain'].append({'method': 'llm_gguf', 'attempted': True})
        result = generate_with_gguf(ranked_df, filters)
        
        if result and result.get('title') and result.get('description'):
            meta['mode'] = 'llm'
            meta['llm_used'] = True
            meta['generation_time'] = result['generation_time']
            meta['final_source'] = 'gguf_local'
            meta['tokens_sampled'] = result.get('tokens_used')
            meta['finish_reason'] = result.get('finish_reason')
            meta['fallback_chain'][-1]['success'] = True
            
            # Advertir si fue truncado
            if result.get('truncated'):
                meta['warning'] = 'Generation may be truncated - consider increasing max_tokens'
            
            return {
                'title': result['title'],
                'description': result['description'],
                'meta': meta
            }
        else:
            meta['fallback_chain'][-1]['success'] = False
            meta['fallback_chain'][-1]['reason'] = 'parsing_failed_or_invalid_output'
            print("⚠️  LLM falló, usando método heurístico...")
    
    # Fallback heurístico
    print("📋 Generando con método heurístico...")
    meta['fallback_chain'].append({'method': 'heuristic_enhanced', 'attempted': True})
    
    try:
        title = title_heuristic(filters, ranked_df)
        sentences = build_sentences(ranked_df)
        base_description = mmr_summary(sentences, target=120)
        description = enhance_description(base_description, ranked_df, filters)
        
        if len(description) > 500:
            description = description[:497] + "..."
        
        meta['mode'] = 'heuristic_enhanced'
        meta['final_source'] = 'heuristic'
        meta['generation_time'] = time.time() - start_time
        meta['fallback_chain'][-1]['success'] = True
        
        print(f"✅ Heurístico exitoso: '{title[:50]}...'")
        
        return {
            'title': title,
            'description': description,
            'meta': meta
        }
    except Exception as e:
        print(f"❌ Error en método heurístico: {e}")
        meta['fallback_chain'][-1]['success'] = False
        meta['fallback_chain'][-1]['error'] = str(e)
        
        return {
            'title': 'Scientific Studies Analysis',
            'description': f'Analysis of {len(ranked_df)} scientific studies.',
            'meta': meta
        }