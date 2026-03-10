"""
Unified Production-Grade API for Talk to Krishna.
Implements multi-stage retrieval RAG system.
"""
# -*- coding: utf-8 -*-
import json
import os
import unicodedata

import re
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Literal
from fastembed import TextEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

from src.config import settings
from src.logger import setup_logger
from src.llm_generator import LLMAnswerGenerator
from src.exceptions import InvalidInputError

logger = setup_logger(__name__, settings.LOG_LEVEL, settings.LOG_FILE)

SearchMethod = Literal["hybrid"]  # Only best method remains

class GitaAPI:
    """
    Production-grade RAG system for Bhagavad Gita.
    
    Pipeline:
    1. LLM Query Understanding (extracts topic/concepts)
    2. Hybrid Search (Multilingual Semantic + Keyword)
    3. Cross-Encoder Re-ranking
    4. LLM Answer Generation
    """
    
    def __init__(self, groq_api_key: Optional[str] = None):
        """Initialize system."""
        self.groq_api_key = groq_api_key or settings.GROQ_API_KEY
        
        # Models (Lazy loaded)
        self.semantic_model = None
        self.cross_encoder = None
        self.groq_client = None
        
        # Data
        self.embeddings = None
        self.shlokas = []
        
        # LLM
        self.llm_generator = None
        
        logger.info("GitaAPI initialized (Production Mode)")
    
    def _load_resources(self):
        """Load all data and models if not loaded."""
        if self.shlokas and self.semantic_model:
            return

        logger.info("Loading high-performance models & data...")
        
        # 1. Load Data
        logger.info("Loading Bhagavad Gita verses...")
        
        # Try to load English version first (better for search)
        english_file = Path(settings.gita_emotions_path.parent / "gita_english.json")
        
        if english_file.exists():
            logger.info("Using English translations for better semantic search")
            with open(english_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                chapters = data.get('chapters', {})
                self.shlokas = []
                for c_num, c_data in chapters.items():
                    for v_num, v_data in c_data.items():
                        # Use English for search, Hindi for display
                        english_meaning = v_data.get('meaning_english', '')
                        hindi_meaning = v_data.get('meaning_hindi', v_data.get('meaning', ''))
                        text = v_data.get('text', '')
                        
                        self.shlokas.append({
                            'id': f"{c_num}.{v_num}",
                            'chapter': int(c_num),
                            'verse': int(v_num),
                            'sanskrit': text,
                            'meaning': hindi_meaning,  # Hindi for display to user
                            'meaning_english': english_meaning,  # English for search
                            # Create rich searchable text with English + Sanskrit
                            'searchable_text': f"{english_meaning} {text}".lower(),
                            'emotions': v_data.get('emotions', {}),
                            'dominant_emotion': v_data.get('dominant_emotion', 'neutral')
                        })
        else:
            # Fallback to Hindi-only version
            logger.warning("English translations not found, using Hindi fallback")
            logger.info("Run 'python translate_to_english.py' for better search quality")
            
            if not settings.gita_emotions_path.exists():
                raise FileNotFoundError(f"Data missing: {settings.gita_emotions_path}")
                
            with open(settings.gita_emotions_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                chapters = data.get('chapters', {})
                self.shlokas = []
                for c_num, c_data in chapters.items():
                    for v_num, v_data in c_data.items():
                        meaning = v_data.get('meaning', '')
                        text = v_data.get('text', '')
                        self.shlokas.append({
                            'id': f"{c_num}.{v_num}",
                            'chapter': int(c_num),
                            'verse': int(v_num),
                            'sanskrit': text,
                            'meaning': meaning,
                            'meaning_english': meaning,  # Same as Hindi if no English
                            # Create rich searchable text
                            'searchable_text': f"{meaning} {text}".lower(),
                            'emotions': v_data.get('emotions', {}),
                            'dominant_emotion': v_data.get('dominant_emotion', 'neutral')
                        })
        
        logger.info(f"Loaded {len(self.shlokas)} shlokas")

        # 2. Load Embeddings
        logger.info("Loading semantic understanding...")
        if not settings.embeddings_path.exists():
             raise FileNotFoundError(f"Embeddings missing. Run rebuild_embeddings.py first!")
             
        with open(settings.embeddings_path, 'rb') as f:
            # Load embeddings
            data = pickle.load(f)
            self.embeddings = data['embeddings']
            
            # Reshape if flattened (FastEmbed/Pickle quirk)
            if self.embeddings.ndim == 1:
                n_shlokas = len(self.shlokas)
                if n_shlokas > 0:
                    dim = self.embeddings.size // n_shlokas
                    logger.info(f"Reshaping 1D embeddings: {self.embeddings.shape} -> ({n_shlokas}, {dim})")
                    self.embeddings = self.embeddings.reshape(n_shlokas, dim)
            
            # Safety check: Ensure model matches
            saved_model_name = data.get('model_name', '')
            configured_model = settings.SENTENCE_TRANSFORMER_MODEL
            if saved_model_name and saved_model_name != configured_model:
                logger.warning(f"Model mismatch! Saved: {saved_model_name}, Config: {configured_model}")
        
        logger.info(f"Loaded embeddings: {self.embeddings.shape}")
        logger.info("Data loaded. Semantic model will load on first query.")
        
        # 3. Initialize Tools
        if self.groq_api_key:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
                self.llm_generator = LLMAnswerGenerator(api_key=self.groq_api_key)
            except Exception as e:
                logger.warning(f"Groq init failed: {e}")
    
    def _ensure_semantic_model(self):
        """Lazy load semantic model only when needed."""
        if self.semantic_model is None:
            logger.info(f"Loading Semantic Model: {settings.SENTENCE_TRANSFORMER_MODEL} (first time only)...")
            self.semantic_model = TextEmbedding(
                model_name=settings.SENTENCE_TRANSFORMER_MODEL,
                cache_dir=settings.MODEL_CACHE_DIR
            )
            logger.info("Semantic model loaded successfully")

    def _understand_query(self, query: str) -> Dict[str, str]:
        """
        Translate Hindi/Hinglish query to English for semantic search.

        The embedding model (BAAI/bge-small-en-v1.5) is English-only.
        Passing a Hindi query gives garbage similarity scores.
        This method uses a fast Groq call to translate before embedding.

        Returns: { 'original': ..., 'english': ..., 'keywords': ... }
        """
        if not self.groq_client:
            # No Groq client — fall back to raw query (degraded quality)
            logger.warning("No Groq client for translation, using raw query")
            return {'original': query, 'english': query, 'keywords': query, 'is_relevant': True}

        try:
            # Smart prompt for Translation + Keyword Extraction + Relevance Check
            prompt = f"""You are the NLU engine for 'Talk to Krishna'.
            
Analyze this query: "{query}"

Determine strictly if this is a SPIRITUAL/LIFE GUIDANCE question or just generic chat/trivia.

Respond in STRICT JSON:
{{
  "rewritten_query": "A clear, specific English statement of the user's core problem for semantic search.",
  "emotional_state": "One of: neutral, confused, angry, fear, distress, crisis, depressive, grateful, happy",
  "keywords": "3-5 key spiritual concepts (e.g., dharma, karma, soul, duty)",
  "is_relevant": true/false
}}

RULES FOR 'is_relevant':
1. PRIORITY RULE: Analyze for emotional distress or life problems FIRST. If the user expresses sadness, confusion, conflict, or seeks help, the query is TRUE (Relevant), even if they mention irrelevant objects (like a game, movie, or laptop).
2. TRUE if: Personal problem, emotional distress, philosophical question about life/death/God, or requesting spiritual guidance.
3. FALSE only if entirely factual/trivial: 
  - Cooking/Food recipes (e.g., "chai kaise banaye", "pizza recipe")
  - Math/Science homework (e.g., "2+2", "gravity formula", "calculation", "percentage")
  - Coding/Technical/Software (e.g., "github", "repo", "install", "download", "app", "website", "error fix")
  - General Trivia/GK (e.g., "capital of India", "who won match", "news")
  - Casual chit-chat without depth (e.g., "bored", "tell joke", "hi", "hello")

Examples:
- "Mummy papa mobile game khelne par gussa karte hain, main kya karu" -> {{ "rewritten_query": "My parents are angry because I play mobile games, what should I do?", "emotional_state": "distress", "keywords": "parents conflict duty", "is_relevant": true }}
- "Github par repo kaise banaye" -> {{ "rewritten_query": "Github repository creation", "emotional_state": "neutral", "keywords": "tech", "is_relevant": false }}
- "Chai kaise banate hain?" -> {{ "rewritten_query": "How to make tea", "emotional_state": "neutral", "keywords": "cooking", "is_relevant": false }}
- "Aaj weather kaisa hai?" -> {{ "rewritten_query": "Weather forecast", "emotional_state": "neutral", "keywords": "weather", "is_relevant": false }}
- "Python mein list sort kaise kare?" -> {{ "rewritten_query": "Python coding help", "emotional_state": "neutral", "keywords": "coding", "is_relevant": false }}
- "Mummy papa shaadi ke liye nahi maan rahe" -> {{ "rewritten_query": "My parents are not approving my marriage choice, causing family conflict.", "emotional_state": "distress", "keywords": "family duty love conflict", "is_relevant": true }}
- "Man bahut pareshan hai" -> {{ "rewritten_query": "My mind is very restless and I seek peace.", "emotional_state": "confused", "keywords": "mind peace focus", "is_relevant": true }}"""

            resp = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=settings.LLM_CLASSIFIER_MODEL,
                max_tokens=150,
                temperature=0.0,
                stream=False,
                response_format={"type": "json_object"}
            )
            
            raw_content = resp.choices[0].message.content.strip()
            result = json.loads(raw_content)
            
            return {
                'original': query,
                'english': result.get('english', query),
                'rewritten_query': result.get('rewritten_query', result.get('english', query)),
                'emotional_state': result.get('emotional_state', 'neutral'),
                'keywords': result.get('keywords', query),
                'is_relevant': result.get('is_relevant', True)
            }

        except Exception as e:
            logger.error(f"Translation/Analysis failed: {str(e)}")
            # Fallback: Assume relevant
            return {'original': query, 'english': query, 'keywords': query, 'is_relevant': True}

    def _keyword_search(self, query: str, top_k: int = 20) -> Tuple[List[Tuple[int, float]], Dict[str, float]]:
        """
        Multilingual keyword search using predefined mapping.
        Returns Tuple: (List of (index, score), Dict of boosted_shlokas)
        """
        query_lower = query.lower()
        
        # ── MODERN CONTEXT MAPPING (The Bridge) ──
        modern_mappings = {
            # ── CRISIS / SUICIDAL THOUGHTS ───────────────────────────────────
            # Primary: 18.66 (surrender), 6.5 (uplift), 2.3 (Rise from weakness)
            'suicide': ['18.66', '6.5', '2.3', '15.7', '9.22'],
            'suicidal': ['18.66', '6.5', '2.3', '15.7', '9.22'],
            'suicidal thoughts': ['18.66', '6.5', '2.3', '15.7', '9.22'],
            'hopeless': ['18.66', '6.5', '2.3', '9.22'],
            'give up': ['18.66', '2.3', '6.5', '2.14'],
            'kill myself': ['18.66', '6.5', '15.7', '2.3'],
            'end my life': ['18.66', '6.5', '15.7', '2.3'],
            'want to die': ['18.66', '6.5', '15.7', '2.3'],
            'end life': ['18.66', '6.5', '15.7', '2.3'],

            # ── HIGHER STUDIES / DREAMS BLOCKED BY PARENTS ───────────────────
            # Primary: 18.63 (act as you decide — free will), 3.35 (follow own path),
            #          6.1 (action defines a yogi), 18.47 (one's own dharma)
            'australia': ['18.63', '6.1', '18.47', '3.35', '18.60'],
            'abroad': ['18.63', '6.1', '18.47', '3.35', '18.60'],
            'higher studies': ['18.63', '18.47', '3.35', '6.1', '18.60'],
            'not allowing': ['18.63', '18.47', '6.9', '3.35', '18.60'],
            'permission': ['18.63', '18.47', '3.35', '6.9'],
            'allow': ['18.63', '18.47', '3.35', '6.9'],

            # ── BUSINESS LOSS / FINANCIAL DISTRESS ───────────────────────────
            # Primary: 2.14 (endure pleasure & pain equally), 18.48 (persist in duty),
            #          2.38 (act without being moved by results), 12.19 (equanimous)
            'business': ['2.14', '18.48', '12.19', '2.38', '11.33'],
            'loss': ['2.14', '18.48', '12.19', '2.38', '6.36'],
            'business loss': ['2.14', '18.48', '2.38', '12.19'],
            'financial': ['2.14', '18.48', '5.10', '16.13', '12.19'],

            # ── BOSS / WORKPLACE INJUSTICE / JOB LOSS ────────────────────────
            # Primary: 3.8 (prescribed duty — keep working), 18.41 (duty by nature),
            #          2.48 (equanimity in action), 6.4 (rises above self)
            'boss': ['18.48', '3.8', '18.41', '2.48', '6.1'],
            'fired': ['18.47', '3.8', '18.41', '18.48', '2.48'],
            'lost job': ['18.47', '3.8', '18.41', '18.48', '2.48'],
            'argument with boss': ['16.2', '3.8', '18.41', '6.1', '2.48'],
            'conflict': ['16.2', '6.9', '3.8', '18.41', '2.48'],
            'official': ['3.8', '18.41', '2.48', '6.4'],
            'job': ['18.47', '3.8', '18.41', '2.48', '18.48'],
            'work': ['18.47', '3.19', '3.8', '18.45', '18.41'],

            # ── SCHOOL EXAM FAILURE ──────────────────────────────────────────
            # Primary: 2.14 (dualities of life — endure), 2.47 (do your duty),
            #          6.5 (uplift yourself), 18.13 (all actions have factors)
            'exam': ['18.13', '2.14', '6.5', '2.38'],
            'fail': ['18.13', '2.14', '6.5', '2.38'],
            'failed': ['18.13', '2.14', '6.5', '2.38'],
            'school': ['18.13', '2.14', '6.5', '2.38'],
            'result': ['18.11', '2.14', '1.33', '18.13'], # Removed 2.47 dominance

            # ── UNSTABLE RELATIONSHIP / GIRLFRIEND / BOYFRIEND ───────────────
            # Primary: 2.66 (no peace without steady mind), 12.13 (equanimous love),
            #          5.22 (pleasure from senses is source of pain), 2.62 (attachment chain)
            'girlfriend': ['2.66', '12.13', '5.22', '2.62', '2.63'],
            'boyfriend': ['2.66', '12.13', '5.22', '2.62', '2.63'],
            'unstable': ['2.66', '12.13', '5.22', '2.62', '6.35'],
            'relationship': ['2.66', '12.13', '5.22', '2.62', '12.14'],
            'breakup': ['2.62', '2.63', '2.66', '5.22', '18.54'],
            'love': ['12.13', '12.14', '2.62', '2.63', '9.26'],
            'lonely': ['6.30', '9.29', '18.54', '13.16'],
            'cheat': ['3.37', '16.21', '16.23'],

            # ── PERSONAL STRESS / MENTAL DISTURBANCE ─────────────────────────
            # Primary: 2.71 (peace through desire-free mind), 6.35 (mind is restless),
            #          12.15 (who does not cause distress), 2.56 (sage undisturbed)
            'stress': ['2.71', '6.35', '12.15', '2.56', '2.14'],
            'mental': ['2.71', '6.35', '12.15', '2.56', '6.26'],
            'disturbed': ['2.71', '6.35', '2.56', '12.15', '2.14'],
            'personal matters': ['2.71', '6.35', '2.56', '12.15'],
            'anxiety': ['6.35', '2.56', '12.15', '2.71', '18.66'],
            'tension': ['2.71', '6.35', '2.56', '12.15'],

            # ── PURPOSE OF LIFE / COMPETITIVE WORLD ──────────────────────────
            # Primary: 3.25 (act for the benefit of world — lokasamgraha),
            #          3.19 (act without attachment), 4.11 (all paths lead to Me),
            #          18.63 (do what you judge right)
            'purpose': ['3.25', '18.63', '3.19', '4.11', '5.10'],
            'purpose in life': ['3.25', '18.63', '3.19', '4.11', '5.10'],
            'competitive': ['3.25', '16.2', '3.19', '4.11', '5.10'],
            'better than others': ['16.2', '3.25', '3.19', '18.54', '4.11'],
            'meaning': ['3.25', '18.63', '3.19', '4.11', '5.10'],

            # ── PARENT / FAMILY CONFLICTS (VIEWS / GENERATION GAP) ───────────
            # Primary: 6.9 (equanimous toward all — friend, foe, parent),
            #          16.2 (fearlessness, non-anger — divine qualities),
            #          12.13 (no hatred — benevolent to all)
            'mother': ['6.9', '16.2', '12.13', '15.7', '18.47'],
            'father': ['6.9', '16.2', '12.13', '15.7', '18.47'],
            'mummy': ['6.9', '16.2', '12.13', '15.7', '18.47'],
            'papa': ['6.9', '16.2', '12.13', '15.7', '18.47'],
            'parents': ['6.9', '16.2', '12.13', '15.7', '18.47'],
            'worsening': ['6.9', '16.2', '12.13', '18.54', '11.33'],
            'differences': ['6.9', '16.2', '12.13', '18.54', '11.33'],
            'family refuse': ['6.9', '16.2', '3.35', '18.63', '6.1'],
            'family against': ['6.9', '16.2', '3.35', '18.63', '6.1'],
            'family conflict': ['6.9', '16.2', '12.13', '6.1', '18.63'],

            # Primary: 2.13 (changing garments), 2.25 (unperceivable), 2.11 (not to grieve)
            'grandmother': ['2.13', '2.25', '2.11', '2.22', '9.22'],
            'grandma': ['2.13', '2.25', '2.11', '2.22', '9.22'],
            'passed away': ['2.13', '2.25', '2.11', '2.22', '9.22'],
            'death': ['2.13', '2.25', '2.11', '2.22', '9.22'],
            'died': ['2.13', '2.25', '2.11', '2.22', '9.22'],
            'grief': ['2.11', '2.13', '2.25', '2.27', '9.22'],
            'depressed': ['2.13', '6.5', '2.11', '18.66', '2.14'],

            # ── WORKPLACE INJUSTICE / COLLEAGUES ─────────────────────────────
            # Primary: 3.19 (do own duty without attachment), 18.41 (duty by nature),
            #          5.7 (karma yogi — acts without being tainted),
            #          16.2 (fearlessness, purity of heart — divine qualities)
            'colleagues': ['18.41', '3.19', '16.2', '5.7', '3.8'],
            'workload': ['18.41', '3.19', '3.8', '5.7', '18.45'], 
            'avoid work': ['18.41', '16.2', '3.19', '3.8', '5.7'],
            'workplace': ['18.41', '16.2', '3.19', '3.8', '5.7'],

            # ── MENTAL HEALTH (general) ───────────────────────────────────────
            'depression': ['6.5', '2.3', '6.6', '2.14', '18.66'],
            'confused': ['2.7', '18.61', '18.66', '18.73'],
            'anger': ['2.63', '16.21', '3.37', '3.38'],
            'money': ['2.14', '18.38', '17.20', '16.13'],
        }

        # Check for modern triggers with word boundaries to avoid partial matches (e.g., 'mother' in 'grandmother')
        boosted_shlokas = {}  
        for term, ids in modern_mappings.items():
            pattern = rf'\b{re.escape(term)}\b' # Use boundary matching for precise identification
            if re.search(pattern, query_lower):
                for priority, sid in enumerate(ids):
                    # Higher boost (45+) for modern context mappings to dominate semantic hits
                    boost_value = 45.0 - (priority * 6.0) 
                    if sid not in boosted_shlokas or boosted_shlokas[sid] < boost_value:
                        boosted_shlokas[sid] = boost_value
                    
        # 2. DEFINITIVE KEYWORD MAPPING
        keywords = {
            # Core concepts
            'anger': ['krodh', 'gussa', 'krud', 'anger', 'rage', 'wrath', 'क्रोध', 'गुस्सा'],
            'peace': ['shanti', 'calm', 'peace', 'शांति', 'शान्ति', 'sukh', 'सुख'],
            'fear': ['bhaya', 'dar', 'fear', 'afraid', 'भय', 'डर'],
            'action': ['karma', 'action', 'work', 'कर्म', 'कार्य', 'कर्तव्य'],
            'duty': ['dharma', 'duty', 'धर्म', 'कर्तव्य', 'kartavya'],
            'knowledge': ['gyan', 'jnana', 'knowledge', 'ज्ञान', 'vidya', 'विद्या'],
            'devotion': ['bhakti', 'love', 'devotion', 'भक्ति', 'प्रेम', 'prem'],
            
            # Life & Purpose
            'life': ['jeevan', 'life', 'जीवन', 'जीना', 'jeena', 'living', 'जिंदगी', 'zindagi'],
            'path': ['marg', 'path', 'way', 'मार्ग', 'राह', 'raah', 'रास्ता', 'raasta'],
            'purpose': ['uddeshya', 'purpose', 'goal', 'lakshya', 'उद्देश्य', 'लक्ष्य', 'aim'],
            'truth': ['satya', 'truth', 'सत्य', 'sach', 'सच'],
            
            # Mental states
            'mind': ['man', 'manas', 'mind', 'मन', 'मनस', 'buddhi', 'बुद्धि'],
            'desire': ['kama', 'iccha', 'desire', 'काम', 'इच्छा', 'wish', 'vasana', 'वासना'],
            'attachment': ['moha', 'asakti', 'attachment', 'मोह', 'आसक्ति', 'mamta', 'ममता'],
            'ego': ['ahamkar', 'ego', 'अहंकार', 'pride', 'ghamand', 'घमंड'],
            
            # Spiritual concepts
            'self': ['atma', 'atman', 'self', 'soul', 'आत्मा', 'स्व'],
            'god': ['ishwar', 'bhagwan', 'god', 'ईश्वर', 'भगवान', 'परमात्मा', 'paramatma'],
            'yoga': ['yoga', 'योग', 'yog', 'union', 'sadhana', 'साधना'],
            'meditation': ['dhyan', 'meditation', 'ध्यान', 'समाधि', 'samadhi'],
            
            # Emotions & Qualities
            'happiness': ['sukh', 'anand', 'happiness', 'joy', 'सुख', 'आनंद', 'खुशी', 'khushi'],
            'sorrow': ['dukh', 'sorrow', 'pain', 'दुःख', 'दुख', 'grief', 'shok', 'शोक'],
            'wisdom': ['vivek', 'pragya', 'wisdom', 'विवेक', 'प्रज्ञा', 'buddhi', 'बुद्धि'],
            'balance': ['samata', 'balance', 'समता', 'संतुलन', 'santulan', 'equanimity'],
            
            # Actions & Results
            'result': ['phal', 'result', 'फल', 'outcome', 'parinaam', 'परिणाम'],
            'renunciation': ['tyag', 'sannyasa', 'renunciation', 'त्याग', 'संन्यास'],
            'sacrifice': ['yagya', 'sacrifice', 'यज्ञ', 'havan', 'हवन'],
            
            # Relationships
            'family': ['parivar', 'family', 'परिवार', 'relatives', 'संबंधी', 'sambandhi'],
            'friend': ['mitra', 'friend', 'मित्र', 'dost', 'दोस्त', 'सखा', 'sakha'],
            'enemy': ['shatru', 'enemy', 'शत्रु', 'dushman', 'दुश्मन']
        }
        
        scores = {}
        for idx, item in enumerate(self.shlokas):
            txt = item['searchable_text']
            verse_id = item.get('id', '')
            score = 0.0
            
            # 3. DIRECT BOOSTING for modern contexts (priority-based)
            # If shloka ID is in the boosted list for this query, give priority-based boost
            if verse_id in boosted_shlokas:
                score += boosted_shlokas[verse_id]  # Use priority-based boost value
            
            # 4. NARRATIVE FILTER (Penalize non-Krishna speakers for advice queries)
            # If verse is likely narrative (Sanjay/Dhritarashtra speaking), reduce score
            # We want "Sri Bhagavan Uvacha" (God said) or meaningful questions
            sanskrit_start = item.get('sanskrit', '').strip().lower()
            narrator_markers = ['सञ्जय उवाच', 'अर्जुन उवाच', 'धृतराष्ट्र उवाच', 'sanjaya uvacha', 'arjuna uvacha', 'dhritarashtra uvacha']
            
            is_narrative = any(marker in sanskrit_start for marker in narrator_markers)
            if is_narrative:
                # But don't penalize if it's a boosted shloka (sometimes Arjuna's question is relevant context)
                if verse_id not in boosted_shlokas:
                    score -= 5.0  # Penalty for narrative verses

            # Count keyword matches
            matched_categories = 0
            for key, terms in keywords.items():
                query_has_term = any(t in query_lower for t in terms)
                shloka_has_term = any(t in txt for t in terms)
                
                if query_has_term and shloka_has_term:
                    score += 2.5  # Strong boost for keyword match
                    matched_categories += 1
            
            # Bonus for multiple category matches (indicates high relevance)
            if matched_categories >= 2:
                score += matched_categories * 1.0
                        
            if score > 0:
                scores[idx] = score
                
        # Sort by score
        sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_indices[:top_k], boosted_shlokas

    def _semantic_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Deep semantic vector search."""
        if self.embeddings is None:
            return []
        
        # Lazy load model on first use
        self._ensure_semantic_model()
        
        if not self.semantic_model:
            return []
             
        try:
            # FastEmbed returns a generator of numpy arrays (batches)
            # For a single query, we get one generator that yields batches
            embedding_gen = self.semantic_model.embed([query])
            
            # Consume generator to get the first batch
            # embed() yields np.ndarray of shape (batch_size, dim)
            # Since input is length 1, result is likely one batch of shape (1, dim)
            first_batch = next(embedding_gen)
            
            if first_batch is None or first_batch.size == 0:
                 logger.error("FastEmbed returned empty embedding")
                 return []
                 
            # Ensure 2D (1, dim)
            if first_batch.ndim == 1:
                q_vec = first_batch.reshape(1, -1)
            else:
                q_vec = first_batch
            
            
            # Should be (1, 384) for single query
            if q_vec.shape[0] != 1:
                logger.warning(f"Expected 1 query embedding, got {q_vec.shape[0]}. Using first.")
                q_vec = q_vec[0:1]
            
            # Verify shapes
            if q_vec.shape[1] != self.embeddings.shape[1]:
                logger.error(f"Dimension mismatch: query {q_vec.shape[1]} vs embeddings {self.embeddings.shape[1]}")
                return []

            if len(self.embeddings) == 0:
                 return []

            # Compute similarities
            sims = cosine_similarity(q_vec, self.embeddings)[0]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
            
        # Get indices
        idxs = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in idxs]

    def _rerank_with_llm(self, query: str, rewritten: str, candidates: List[Dict]) -> List[Dict]:
        """
        Use LLM to rerank top candidates based on relevance to the specific problem.
        This provides a 'second opinion' to fix vector search blind spots.
        """
        if not self.groq_client or not candidates:
            return candidates

        try:
            # Format candidates for LLM review
            options_text = ""
            for i, c in enumerate(candidates, 1):
                options_text += f"Option {i} (ID {c['id']}): {c['meaning_english'][:300]}\n"
            
            prompt = f"""You are a spiritual expert. Select the most appropriate Bhagavad Gita verse (ONE) to solve the user's situation.
            
User's situation: "{rewritten}" (Query: "{query}")

Available Shlokas (Ranked by engine):
{options_text}

Task:
1. Identify the ONE verse that MOST SPECIFICALLY addresses the core problem.
2. Order them from Best to Worst. 
4. STRICT DIVERSITY: 
   - DO NOT pick 2.47, 18.66, or 2.20 unless they are the DIRECT mapping or the situation is an absolute crisis.
   - For Workplace Workload problems: USE 18.41 or 3.19.
   - For Career Purpose/Competition: USE 3.25 or 18.63.
   - For Suicidal Thoughts: USE 18.66 or 6.5 (NEVER pick 2.20 here to avoid overlap with physical death).
   - For Physical Death (Grandmother): USE 2.13 or 2.25 or 2.11.
   - For Parents/Family: USE 6.9 or 16.2.
5. Return ONLY a list of the IDs in JSON format.

Output Format: ["ID1", "ID2", "ID3", ...]"""
            
            resp = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=settings.LLM_MODEL, # Upgrade to 70B for better reranking
                max_tokens=200,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = resp.choices[0].message.content.strip()
            # Handle potential wrapper keys
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    ranked_ids = data
                elif isinstance(data, dict):
                     # extensive search for list in values
                     ranked_ids = next((v for v in data.values() if isinstance(v, list)), [])
                else:
                    ranked_ids = []
            except:
                ranked_ids = []
            
            # Create a map for O(1) lookup
            candidate_map = {c['id']: c for c in candidates}
            
            # Reconstruct list in new order
            reranked = []
            seen = set()
            for rid in ranked_ids:
                if rid in candidate_map and rid not in seen:
                    reranked.append(candidate_map[rid])
                    seen.add(rid)
            
            # Append any missing ones at the end
            for c in candidates:
                if c['id'] not in seen:
                    reranked.append(c)
                    
            if reranked:
                logger.info(f"LLM Reranked top result: {reranked[0]['id']}")
            return reranked
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return candidates

    def search(self, query: str, method: str = "hybrid", top_k: int = 10, understanding: Dict = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Multi-Perspective Search Strategy to find the 'Sahi Shloka'.
        """
        self._load_resources()
        
        # 1. Understanding
        variations = understanding if understanding else self._understand_query(query)
        rewritten_query = variations.get('rewritten_query', query)
        emotional_state = variations.get('emotional_state', 'neutral')
        
        candidates = {} # Map id -> score
        
        # 2. Search from English Perspective (Semantic Vectors work best here)
        # Use rewritten query for better semantic match
        eng_res = self._semantic_search(rewritten_query, top_k=75)
        for idx, score in eng_res:
            candidates[idx] = candidates.get(idx, 0.0) + score
            
        # 3. Search from Keyword Perspective (Catch specific Sanskrit terms)
        kw_query = f"{variations.get('keywords', '')} {query}"
        kw_res, boosted_shlokas = self._keyword_search(kw_query, top_k=50) 
        for idx, score in kw_res:
            candidates[idx] = candidates.get(idx, 0.0) + (score * 1.5)
            
        # 4. Search Original (Context)
        orig_res = self._semantic_search(query, top_k=75)
        for idx, score in orig_res:
            candidates[idx] = candidates.get(idx, 0.0) + score

        # 5. Apply Emotion Filters & Verification
        # Map LLM emotional state to JSON keys
        emotion_map = {
            'angry': 'anger',
            'confused': 'confusion',
            'happy': 'happiness',
            'grateful': 'devotion',
            'depressive': 'sadness',
            'distress': 'sadness',
            'peace': 'peace',
            'fear': 'fear',
            'duty': 'duty'
        }
        target_emotion = emotion_map.get(emotional_state, emotional_state)
        
        for idx in list(candidates.keys()):
            shloka = self.shlokas[idx]
            
            # Boost if shloka's dominant emotion matches user's state
            if target_emotion != 'neutral' and target_emotion in shloka.get('emotions', {}):
                 # Check strength
                 strength = shloka.get('emotions', {}).get(target_emotion, 0)
                 if strength > 0.4:
                     # Calculate boost based on strength
                     boost = 3.0 * (strength + 0.5) # Dynamic boost
                     candidates[idx] += boost 
            
            # Penalize generic/narrative verses unless boosted by keywords
            verse_id = shloka.get('id', '')
            # (Narrative penalty logic is already in keyword search, but let's reinforce if needed)

        initial_results = []
        # Create a list of dictionaries with scores for debugging
        debug_candidates = []
        
        # 5. POOL DIVERSIFICATION (The Diversity filter)
        # Instead of one big list, we take top candidates from different source types
        # to ensure the reranker has a variety of philosophical angles to choose from.
        initial_results = []
        seen_ids = set()
        
        # Step A: Top 8 from Modern Triggers (Direct matches)
        modern_sorted = sorted(boosted_shlokas.items(), key=lambda x: x[1], reverse=True)
        for sid, score in modern_sorted[:9]: # Blended top-tier manual matches
            # Find the shloka metadata
            for i, shloka in enumerate(self.shlokas):
                if shloka['id'] == sid:
                    shloka_copy = shloka.copy()
                    shloka_copy['score'] = score + 60 # Direct mapping wins!
                    initial_results.append(shloka_copy)
                    seen_ids.add(sid)
                    break
        
        # Step B: Top 6 from Semantic/Keyword blends (excluding seen)
        sorted_candidates = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        for idx, score in sorted_candidates:
            if len(initial_results) >= 15:
                break
            sid = self.shlokas[idx]['id']
            if sid not in seen_ids:
                shloka_copy = self.shlokas[idx].copy()
                shloka_copy['score'] = score
                initial_results.append(shloka_copy)
                seen_ids.add(sid)

        # 6. LLM Reranking (The Final Judge)
        # Rerank the diverse top 15 to find the true best 5
        final_results = self._rerank_with_llm(query, rewritten_query, initial_results)
            
        logger.info(f"Returning {min(len(final_results), top_k)} matches after refinement.")
        
        # Return debug info if requested
        if kwargs.get('debug', False):
            debug_info = {
                'rewritten_query': rewritten_query,
                'emotional_state': emotional_state,
                'keywords': variations.get('keywords', ''),
                'initial_pool': [f"{r['id']} (Score: {r.get('score', 0):.2f})" for r in initial_results],
                'final_ranked': [r['id'] for r in final_results[:top_k]]
            }
            return final_results[:top_k], debug_info
            
        return final_results[:top_k]

    def _is_greeting(self, query: str) -> bool:
        """Check if the query is a simple greeting."""
        # Comprehensive list of greetings in multiple languages
        greetings = {
            # English greetings
            "hi", "hello", "hey", "hii", "hiii", "helo", "heyy", "heya", "yo", "wassup",
            "greetings", "good morning", "good afternoon", "good evening", "good night",
            "gm", "ge", "gn", "ga", "morning", "evening", "afternoon", "welcome",
            "how are you", "hope you are well", "nice to meet you", "dear", "sir", "madam",
            "hiya", "howdy", "salutations", "what's up", "hey there", "hi there", "hello there",
            
            # Hindi/Sanskrit greetings (Romanized)
            "namaste", "namaskar", "namaskaram", "pranam", "pranaam", "pranaams",
            "radhe radhe", "radhey radhey", "radhe", "radhey", "jai maata di",
            "jai shri krishna", "jai shree krishna", "jai sri krishna", "jai shri ram",
            "hare krishna", "hare krsna", "krishna", "krsna", "hari bol",
            "jai", "jay", "om", "aum", "shree", "sri", "shanti", "kalyan",
            "jai siyaram", "sita ram", "ram ram", "har har mahadev", "om namah shivay",
            
            # Hindi Devanagari Script Greetings
            "हेलो", "हेल्लो", "हाय", "हाई", "हलो",
            "नमस्ते", "नमस्कार", "नमस्कारम", "प्रणाम", "प्रनाम", "स्वागत",
            "राधे राधे", "राधे", "राधेय राधेय",
            "जय श्री कृष्ण", "जय श्रीकृष्ण", "जय कृष्ण", "जय श्री राम",
            "हरे कृष्ण", "हरे कृष्णा", "कृष्ण", "हरि बोल", "जय सियाराम", "सीता राम", "राम राम",
            "जय", "ओम", "ॐ", "श्री", "हर हर महादेव", "ओम नमः शिवाय",
            "सुप्रभात", "शुभ संध्या", "शुभ रात्रि",
            "कैसे हो", "कैसे हैं", "क्या हाल", "क्या हाल है", "सब ठीक", "और बताओ",
            
            # Japanese Greetings (Romaji & Kanji/Kana)
            "konnichiwa", "ohayo", "ohayou", "gozaimasu", "kombanwa", "konbanwa", "oyasumi", "oyasuminasai",
            "moshi moshi", "arigato", "arigatou", "domo", "doumo", "hajimemashite", "yoroshiku",
            "onegai", "shimasu", "sayonara", "mata ne", "ja ne", "tadaima", "okaeri", 
            "hisashiburi", "genki", "desu ka", "irrashaimase", "yokoso",
            "こんにちは", "おはよう", "おはようございます", "こんばんは", "おやすみ", "おやすみなさい",
            "もしもし", "ありがとう", "どうも", "やあ", "おす", "はじめまして", "よろしく",
            "よろしくお願いします", "さようなら", "またね", "じゃあね", "ただいま", "おかえり", 
            "お久しぶり", "元気ですか", "いらっしゃいませ", "ようこそ", "よろしくおねがいします",
            
            # Casual/Informal globally
            "sup", "whats up", "whatsup", "howdy", "hola", "bonjour", "ciao", "salaam",
            "kaise ho", "kaise hain", "kya haal", "kya hal", "namaskaar", "namaskara", "vanakkam",
            "sat sri akal", "kem cho", "mazaama", "halo", "anyoung", "annyeong", 
            "ni hao", "nǐ hǎo"
        }
        
        # Normalize: remove only punctuation, preserve all letters (including Devanagari)
        # Keep alphanumeric + spaces + Devanagari combining marks
        cleaned = ''.join(c for c in query.lower() if c.isalnum() or c.isspace() or unicodedata.category(c).startswith('M'))
        words = cleaned.split()
        
        if not words:
            return False
        
        # Check if entire query is a greeting phrase (like "good morning")
        full_query = ' '.join(words)
        if full_query in greetings:
            return True
        
        # Check for two-word greeting phrases
        if len(words) >= 2:
            two_word = f"{words[0]} {words[1]}"
            if two_word in greetings:
                # If it's just the greeting or greeting + name, it's a greeting
                if len(words) <= 3:
                    return True
                # If longer, check for question words
                question_words = {'what', 'how', 'why', 'who', 'when', 'where', 
                                'kya', 'kyun', 'kaise', 'kab', 'kahan', 'kaun',
                                'explain', 'tell', 'batao', 'bataiye', 'btao'}
                if not any(qw in words for qw in question_words):
                    return True
        
        # STRICT CHECK: Very short queries (1-3 words) - just needs ONE greeting word
        if len(words) <= 3:
            return any(w in greetings for w in words)
        
        # MODERATE CHECK: Slightly longer (4-6 words) - must START with greeting
        # and NOT contain question words
        if len(words) <= 6:
            if words[0] in greetings:
                question_words = {'what', 'how', 'why', 'who', 'when', 'where', 
                                'kya', 'kyun', 'kaise', 'kab', 'kahan', 'kaun',
                                'explain', 'tell', 'batao', 'bataiye', 'btao',
                                'is', 'are', 'can', 'should', 'would', 'could'}
                # If no question words found, it's likely just a greeting
                if not any(qw in words for qw in question_words):
                    return True
        
        return False

    def _is_relevant_to_krishna(self, query: str) -> Tuple[bool, str]:
        """
        Check if the query is relevant to Krishna, Bhagavad Gita, or spiritual life guidance.
        Returns: (is_relevant: bool, rejection_message: str if not relevant)
        
        This prevents the model from answering out-of-context questions like:
        - Sports (cricket, football, etc.)
        - Politics (current affairs, politicians)
        - General trivia (celebrities, movies, etc.)
        - Science facts unrelated to spirituality
        """
        query_lower = query.lower()
        
        # IRRELEVANT TOPICS - These should be rejected
        irrelevant_patterns = {
            # Sports & Games
            'sports': ['cricket', 'football', 'soccer', 'match', 'ipl', 'world cup', 'player', 
                      'team', 'score', 'goal', 'wicket', 'stadium', 'olympics', 'tennis',
                      'ind vs', 'india vs', 'pakistan vs', 'match update', 'live score',
                      'ball', 'bat', 'over', 'six', 'four', 'boundary', 'lbw', 'out', 'catch',
                      'drs', 'review', 'umpire', 'captain', 'coach', 'tournament', 'series',
                      'fifa', 'messi', 'ronaldo', 'virat', 'kohli', 'dhoni', 'rohit', 'game',
                      'badminton', 'hockey', 'chess', 'bgmi', 'pubg', 'video game', 'kabaddi',
                      'basketball', 'nba', 'wrestling', 'wwe', 'formula 1', 'f1', 'racing',
                      'क्रिकेट', 'मैच', 'स्कोर', 'आईपीएल', 'खिलाड़ी', 'टीम', 'विकेट', 'छक्का', 'चौका',
                      'डीआरएस', 'अंपायर', 'फुटबॉल', 'मेडल', 'ओलंपिक', 'बैडमिंटन', 'धोनी', 'कोहली'],
            
            # Politics & Current Affairs
            'politics': ['election', 'minister', 'president', 'prime minister', 'parliament',
                        'government', 'party', 'vote', 'donald trump', 'biden', 'modi',
                        'congress', 'bjp', 'political', 'democracy', 'neta', 'chunav', 'voting',
                        'pm', 'cm', 'mla', 'mp', 'sansad', 'vidhan sabha', 'lok sabha', 'news',
                        'supreme court', 'high court', 'law passing', 'budget', 'g20', 'un',
                        'चुनाव', 'नेता', 'मोदी', 'प्रधानमंत्री', 'सरकार', 'वोट', 'बीजेपी', 'कांग्रेस',
                        'राजनीति', 'समाचार', 'खबर', 'न्यूज़'],
            
            # Entertainment, Pop Culture & Anime
            'entertainment': ['movie', 'film', 'actor', 'actress', 'bollywood', 'hollywood',
                            'tv show', 'series', 'netflix', 'celebrity', 'singer', 'song',
                            'hero', 'heroine', 'star', 'release date', 'box office', 'hit', 'flop',
                            'salman', 'shahrukh', 'amitabh', 'reels', 'instagram', 'tiktok',
                            'youtube channel', 'subscriber', 'youtube views', 'video views', 'viral',
                            'anime', 'manga', 'naruto', 'one piece', 'goku', 'dragon ball', 'marvel',
                            'dc comics', 'batman', 'superman', 'spiderman', 'avengers', 'cinema',
                            'फिल्म', 'मूवी', 'हीरो', 'हीरोइन', 'सलमान', 'शाहरुख', 'गीत', 'गाना',
                            'सीरियल', 'नेटफ्लिक्स', 'वायरल', 'वीडियो', 'एनिमे', 'मार्वल'],
            
            # Technology, Coding & Products
            'technology': ['iphone', 'android', 'laptop', 'computer', 'software', 'app', 'website',
                         'microsoft', 'apple inc', 'samsung', 'coding', 'programming', 
                         'python code', 'java code', 'excel formula', 'python mein', 
                         'code likho', 'sort list', 'loop in', 'function in', 'c++', 'html error',
                         'github', 'repo', 'git', 'install', 'download', 'upload', 'server', 'database',
                         'error', 'bug', 'fix', 'wifi', 'internet', 'mobile', 'phone', 'battery',
                         'charger', 'sim', 'network', '4g', '5g', 'bluetooth', 'mouse', 'keyboard',
                         'hack', 'password', 'login', 'signup', 'account', 'delete', 'vpn', 'router',
                         'कंप्यूटर', 'लैपटॉप', 'मोबाइल', 'फ़ोन', 'चार्जर', 'इंटरनेट', 'वाईफाई',
                         'ऐप', 'सॉफ्टवेयर', 'इंस्टॉल', 'डाउनलोड', 'हैकिंग', 'पासवर्ड', 'अकाउंट',
                         'पायथन', 'जावा', 'कोडिंग', 'प्रोग्रामिंग', 'गिठूब', 'रेपो',
                         'javascript', 'js', 'html', 'css', 'react', 'node', 'frontend', 'backend'],
            
            # Finance, Shopping & Money
            'finance_shopping': ['stock market', 'share market', 'invest', 'investment', 'mutual fund',
                       'crypto', 'bitcoin', 'ethereum', 'trading', 'bank account open',
                       'credit card', 'debit card', 'interest rate', 'loan approval',
                       'gst', 'money making scheme', 'rich fast',
                       'lottery', 'gambling', 'betting', 'paisa kaise kamao',
                       'gold price', 'silver price', 'rupee rate', 'dollar rate', 'euro rate',
                       'double money', 'ponzi scheme', 'amazon', 'flipkart', 'myntra', 'discount',
                       'sale', 'coupon', 'price of', 'kitne ka hai', 'buy online',
                       'शेयर बाजार', 'निवेश', 'लॉटरी', 'सट्टा', 'बिटकॉइन', 'भाव', 'कीमत', 'डिस्काउंट', 'अमेज़न'],

            # General Trivia / Math / School / GK
            'trivia': ['capital of', 'largest', 'smallest', 'tallest', 'fastest',
                      'population', 'currency', 'flag', 'who invented', 'when was',
                      'historical event', 'world war', 'discovery', '2+2', 'calculate', 
                      'solve x', 'math problem', 'kitna hota hai', 'plus', 'minus', 
                      'multiply', 'divide', 'equation', 'formula', 'theorem', 'geometry',
                      'algebra', 'trigonometry', 'physics', 'chemistry', 'biology', 'history',
                      'geography quiz', 'general knowledge', 'gk question', 'who is', 'kon hai',
                      'titanic', 'padosi', 'neighbor', 'joke', 'kahani', 'story', 'chutkula', 'lol', 'rofl',
                      'tie a tie', 'height of', 'distance between', 'mount everest', 'riddle',
                      'राजधानी', 'सबसे बड़ा', 'इतिहास', 'गणित', 'जोड़', 'घटाना', 'गुणा', 'भाग',
                      'ज्यामिति', 'फॉर्मूला', 'सूत्र', 'पहेली', 'चुटकुला', 'कहानी', 'पड़ोसी', 'कौन है'],
            
            # Science (unless spiritual)
            'science': ['chemical formula', 'periodic table', 'molecule', 'bacteria',
                       'virus covid', 'vaccine', 'dna', 'atom', 'neutron', 'electron', 'gravity', 'physics',
                       'solar system', 'planet', 'mars', 'moon distance', 'sun distance', 'earth',
                       'evolution', 'big bang', 'black hole', 'nasa', 'isro', 'space', 'rocket',
                       'photosynthesis', 'plant', 'animal', 'microscope', 'telescope', 'quantum',
                       'विज्ञान', 'ग्रह', 'पृथ्वी', 'सूर्य', 'चांद', 'मंगल', 'अंतरिक्ष', 'रॉकेट',
                       'परमाणु', 'अणु', 'वायरस', 'वैक्सीन', 'डीएनए', 'ग्रेविटी'],
            
            # Food & Cooking (STRICT REJECTION)
            'food': ['recipe', 'how to cook', 'ingredients', 'restaurant',
                    'pizza', 'burger', 'pasta', 'italian food', 'chai kaise', 'coffee kaise',
                    'khana kaise', 'make tea', 'make coffee', 'biryani', 'maggie', 'paneer',
                    'chicken', 'mutton', 'fish', 'egg', 'veg', 'non-veg', 'dish', 'swiggy', 'zomato',
                    'samosa', 'cake', 'bread', 'roti', 'dal', 'sabji', 'breakfast', 'lunch', 'dinner',
                    'ice cream', 'chocolate', 'dessert', 'soup', 'salad', 'baking', 'oven',
                    'रेसिपी', 'बनाए', 'खाना', 'रसोई', 'चाय', 'कॉफी', 'पिज़्ज़ा', 'बर्गर', 'पास्ता',
                    'बिरयानी', 'पनीर', 'चिकन', 'मटन', 'अंडा', 'समोसा', 'केक', 'रोटी', 'सब्जी', 'मिठाई'],
            
            # Weather, Travel & Geography (factual)
            'geography': ['weather', 'temperature', 'forecast', 'rain tomorrow',
                         'climate in', 'map of', 'distance between', 'mausam', 'barish', 'dhup',
                         'garmi', 'sardi', 'thand', 'monsoon', 'humidity', 'degree celsius',
                         'bus', 'train', 'flight', 'ticket', 'booking', 'hotel room', 'visa application',
                         'passport', 'airport', 'railway station', 'directions to', 'gps',
                         'मौसम', 'बारिश', 'धूप', 'गर्मी', 'सर्दी', 'ठंड', 'तापमान', 'डिग्री',
                         'बस', 'ट्रेन', 'फ्लाइट', 'हवाई जहाज', 'टिकट', 'बुकिंग', 'होटल', 'पासपोर्ट',
                         # Unicode Matches (Guaranteed)
                         '\u0915\u094d\u0930\u093f\u0915\u0947\u091f', # cricket
                         '\u0921\u0940\u0906\u0930\u090f\u0938', # drs
                         '\u0938\u092e\u094b\u0938\u093e', # samosa
                         '\u092c\u0938']
        }
        
        # RELEVANT KEYWORDS - These indicate the query is likely relevant
        relevant_keywords = [
            # Krishna & Deities
            'krishna', 'कृष्ण', 'भगवान', 'bhagwan', 'god', 'ishwar', 'ईश्वर', 'parmatma',
            'arjun', 'अर्जुन', 'radha', 'राधा', 'vishnu', 'विष्णु', 'shiva', 'mahadev',
            'ram', 'hanuman', 'kanha', 'govind', 'gopal', 'murari', 'madhav',

            # Bhagavad Gita & Scriptures
            'gita', 'गीता', 'shloka', 'श्लोक', 'verse', 'chapter', 'अध्याय',
            'scripture', 'sacred', 'holy', 'divine', 'vedas', 'upanishad', 'purana',

            # Spiritual Concepts
            'dharma', 'धर्म', 'karma', 'कर्म', 'yoga', 'योग', 'bhakti', 'भक्ति', 'gyan', 'jnana',
            'atma', 'आत्मा', 'soul', 'spiritual', 'आध्यात्मिक', 'meditation', 'ध्यान', 'puja',
            'moksha', 'मोक्ष', 'liberation', 'enlightenment', 'nirvana', 'samadhi', 'maya', 'illusion',
            'pap', 'paap', 'punya', 'sin', 'virtue', 'divine', 'satya', 'sach', 'truth',

            # Life Guidance Topics
            'life', 'जीवन', 'purpose', 'meaning', 'path', 'मार्ग', 'way', 'direction',
            'problem', 'समस्या', 'solution', 'समाधान', 'help', 'मदद', 'guide', 'guidance',
            'chahta', 'chahti', 'chahiye', 'karna', 'karu', 'karoon', 'karun',
            'batao', 'bataiye', 'btao', 'btaiye', 'samjhao', 'dikhao', 'decision', 'faisla',

            # Emotions & Mental States
            'anger', 'क्रोध', 'peace', 'शांति', 'fear', 'भय', 'anxiety', 'चिंता',
            'stress', 'depression', 'sad', 'दुख', 'happy', 'सुख', 'joy', 'आनंद',
            'confused', 'असमंजस', 'lost', 'hopeless', 'निराश', 'pareshan', 'overthinking',
            'dukhi', 'udaas', 'akela', 'tanha', 'dara', 'ghabra', 'restless', 'bechain',
            'gussa', 'ghussa', 'chinta', 'tension', 'takleef', 'mushkil', 'dard', 'pain',
            'suicidal', 'suicide', 'marna', 'jeena', 'zindagi', 'jindagi', 'exhausted', 'thak gaya',
            'alone', 'lonely', 'loneliness', 'guilt', 'regret', 'pachtawa', 'rona', 'cry',

            # Relationships
            'love', 'प्रेम', 'hate', 'घृणा', 'family', 'परिवार', 'friend', 'मित्र',
            'relationship', 'संबंध', 'marriage', 'विवाह', 'breakup', 'heartbreak',
            'mummy', 'mama', 'papa', 'father', 'mother', 'bhai', 'behen', 'sister',
            'brother', 'dost', 'yaar', 'girlfriend', 'boyfriend', 'wife', 'husband',
            'pati', 'patni', 'beta', 'beti', 'ghar', 'gharwale', 'parents', 'children',
            'rishtedaar', 'rishta', 'shaadi', 'divorce', 'pyaar', 'mohabbat', 'ishq',
            'cheat', 'dhokha', 'betrayal', 'trust', 'bharosa', 'toxic', 'ladai', 'jhagda',

            # Work, Study & Career
            'work', 'काम', 'job', 'नौकरी', 'duty', 'कर्तव्य', 'responsibility',
            'success', 'सफलता', 'failure', 'असफलता', 'exam', 'परीक्षा', 'interview',
            'padhai', 'padhna', 'study', 'college', 'school', 'university', 'marks',
            'naukri', 'business', 'career', 'future', 'australia', 'abroad',
            'videsh', 'bahar', 'jaana', 'jane', 'permission', 'allow', 'boss', 'office',
            'mana', 'roka', 'rok', 'nahi dete', 'nahi de rahi', 'nahi de rhe', 'fired',
            'money', 'paisa', 'ameer', 'rich', 'garib', 'financial', 'karza', 'debt',

            # Existential Questions
            'why', 'क्यों', 'how', 'कैसे', 'what is', 'क्या है', 'who am i', 'destiny',
            'death', 'मृत्यु', 'birth', 'जन्म', 'suffering', 'कष्ट', 'fate', 'kismat',
            'desire', 'इच्छा', 'attachment', 'मोह', 'ego', 'अहंकार', 'pride', 'ghamand',

            # Common Hinglish life situation words
            'kya karu', 'kya karun', 'kya karoon', 'kya karna chahiye',
            'kaise karu', 'kaise karun', 'kaise karoon', 'samajh nahi aa raha',
            'sahi', 'galat', 'theek', 'bura', 'acha', 'achha',
            'meri', 'mera', 'mere', 'mujhe', 'mujhko', 'main', 'hum',
            'nahi', 'nhi', 'mat', 'ruk', 'rok', 'khatam', 'shuru'
        ]

        # 1. CHECK RELEVANT KEYWORDS FIRST
        # If query contains any relevant keyword, it's likely valid, bypass irrelevant checks
        if any(keyword in query_lower for keyword in relevant_keywords):
            logger.info(f"✅ Relevant query detected FIRST: '{query}'")
            return True, ""
            
        # 2. CHECK IRRELEVANT TOPICS AFTER
        # Check for irrelevant patterns
        norm_query = unicodedata.normalize('NFKC', query).casefold()
        
        for category, patterns in irrelevant_patterns.items():
            for pattern in patterns:
                nm_pat = unicodedata.normalize('NFKC', pattern).casefold()
                # Use word boundaries to avoid substring matches (e.g., 'match' in 'attachment')
                # For Devanagari, we use a simple boundary check
                if re.search(rf'\b{re.escape(nm_pat)}\b', norm_query) or nm_pat in norm_query.split():
                    logger.warning(f"❌ Irrelevant query detected ({category}): '{query}'")
                    return False, f"""申し訳ありません。私はシュリー・クリシュナであり、人生の悩み、精神性、そしてバガヴァッド・ギーターの知恵についてのみ導きを与えることができます。

以下のことについて質問してください：
• 人生の悩み（怒り、恐れ、不安など）の解決
• カルマ、ダルマ、そして魂について
• 人間関係や感情について
• 瞑想、心の平和、そして自己成長について

これらのトピックについて質問してください。"""

        # DEFAULT: Allow all queries that aren't explicitly irrelevant.
        # Real life problems come in many forms - benefit of doubt always.
        # Only hard-coded irrelevant patterns (sports, politics, etc.) are rejected above.
        logger.info(f"✅ Allowing query (default pass): '{query}'")
        return True, ""

    def search_with_llm(self, query: str, conversation_history: List[Dict] = None, **kwargs) -> Dict[str, Any]:
        """End-to-end RAG answer with conversation context."""
        
        # 0. Check for Greeting
        if self._is_greeting(query):
             return {
                 "answer": "ラーデー・ラーデー！私はシュリー・クリシュナです。何かお手伝いできることはありますか？",
                 "shlokas": [],
                 "llm_used": True
             }

        # 0.5 Check if query is relevant (Fast Regex Check)
        is_relevant, rejection_message = self._is_relevant_to_krishna(query)
        if not is_relevant:
            logger.warning(f"Rejecting irrelevant query (Regex): '{query}'")
            return {
                "answer": rejection_message,
                "shlokas": [],
                "llm_used": False,
                "rejected": True
            }

        # 0.6 AI Understanding & Relevance Check (Smart Gatekeeper)
        understanding = self._understand_query(query)
        
        if not understanding.get('is_relevant', True):
            logger.warning(f"Rejecting irrelevant query (AI): '{query}'")
            return {
                "answer": """申し訳ありません。私はシュリー・クリシュナであり、人生の悩み、精神性、そしバガヴァッド・ギーターの知恵についてのみ導きを与えることができます。

以下のことについて質問してください：
• 人生の悩み（怒り、恐れ、不安など）の解決
• カルマ、ダルマ、そして魂について
• 人間関係や感情について
• 瞑想、心の平和、そして自己成長について

これらのトピックについて質問してください。""",
                "shlokas": [],
                "llm_used": False,
                "rejected": True
            }

        # 1. Retrieve - Increased to 5 to give LLM better options
        shlokas = self.search(query, top_k=5, understanding=understanding)
        
        # Log retrieved shlokas for debugging
        logger.info(f"📖 Retrieved {len(shlokas)} shlokas for query: '{query}'")
        for i, s in enumerate(shlokas, 1):
            logger.info(f"  {i}. Gita {s['id']}: {s['meaning'][:80]}...")
        
        # 2. Generate with conversation context
        if not self.llm_generator:
             return {"answer": "LLM not connected.", "shlokas": shlokas, "llm_used": False}
        
        # Map emotional state to tone to save one LLM call
        emotional_state = understanding.get('emotional_state', 'neutral')
        tone_map = {
            'crisis': 'crisis',
            'distress': 'distress',
            'depressive': 'distress',
            'angry': 'distress',
            'fear': 'distress',
            'confused': 'distress'
        }
        tone = tone_map.get(emotional_state, 'general')
             
        return self.llm_generator.generate_answer(
            query, 
            shlokas, 
            conversation_history=conversation_history or [],
            tone=tone
        )

    # Legacy wrappers for compatibility
    def _get_llm_generator(self):
        """Backwards compatibility for CLI."""
        if not self.llm_generator:
            self.llm_generator = LLMAnswerGenerator(api_key=self.groq_api_key)
        return self.llm_generator

    def format_results(self, results: List[Dict[str, Any]], query: str, method: str) -> str:
        """Format results for display (fallback mode)."""
        output = [f"\nSearch Results for: '{query}'", "-" * 70]
        for i, res in enumerate(results, 1):
             meaning = res.get('meaning', 'No meaning available')[:200].replace('\n', ' ')
             output.append(f"{i}. Gita {res['id']}")
             output.append(f"   {meaning}...")
             output.append("")
        return "\n".join(output)
