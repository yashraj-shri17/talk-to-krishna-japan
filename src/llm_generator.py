"""
LLM Integration module for generating contextual answers using Groq's Llama 3.1.

Architecture:
  Step 1 — classify_query():  One fast LLM call to understand emotional gravity.
                               Returns: 'crisis' | 'distress' | 'general'
  Step 2 — generate_answer(): Uses classification to pick the right prompt tone.
                               Crisis    → empathetic, validating, hopeful
                               Distress  → warm, personal, grounding
                               General   → direct, philosophical, action-oriented

This approach generalises to ANY language or phrasing — no keyword lists needed.
"""
import json
from typing import List, Dict, Any, Optional, Literal
from groq import Groq
from src.config import settings
from src.logger import setup_logger

logger = setup_logger(__name__, settings.LOG_LEVEL, settings.LOG_FILE)

QueryTone = Literal["crisis", "distress", "general"]


class LLMAnswerGenerator:
    """Generate contextual answers using LLM based on retrieved shlokas."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set.")
            self.client = None
        else:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Groq init failed: {e}")
                self.client = None

        self.model = settings.LLM_MODEL

    def is_available(self) -> bool:
        return self.client is not None

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Classify the emotional gravity of the query
    # ─────────────────────────────────────────────────────────────────────────

    def classify_query(self, query: str) -> QueryTone:
        """
        Use LLM to classify the emotional gravity of the user's query.
        Returns: 'crisis' | 'distress' | 'general'

        This is a tiny, fast call (max_tokens=5) — adds ~200ms but makes
        the system generalise to any language, dialect, or phrasing.
        """
        if not self.client:
            return "general"

        try:
            classification_prompt = f"""Classify the emotional gravity of this message into exactly ONE word.

Message: "{query}"

Rules:
- Reply with ONLY one of these three words (nothing else):
  crisis   → person expresses suicidal thoughts, wanting to die, ending life, severe hopelessness
  distress → person is in emotional pain, anxiety, grief, anger, loneliness, failure, family conflict
  general  → person asks a philosophical, spiritual, or life-guidance question without acute pain

Reply with only: crisis OR distress OR general"""

            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": classification_prompt}],
                model=settings.LLM_CLASSIFIER_MODEL,
                max_tokens=5,
                temperature=0.0,  # Deterministic
                stream=False
            )
            raw = response.choices[0].message.content.strip().lower()

            # Parse robustly — model might say "crisis." or "  distress  "
            if "crisis" in raw:
                tone = "crisis"
            elif "distress" in raw:
                tone = "distress"
            else:
                tone = "general"

            logger.info(f"🎭 Query classified as: [{tone}] for: '{query[:60]}'")
            return tone

        except Exception as e:
            logger.warning(f"Classification failed, defaulting to 'general': {e}")
            return "general"

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Build prompts based on tone
    # ─────────────────────────────────────────────────────────────────────────

    def _build_prompts(self, user_question: str, shloka_options: str,
                       history_context: str, tone: QueryTone):
        """Return (system_prompt, user_prompt) tuned to the emotional tone."""
        
        # Base instructions for all tones
        base_instructions = """
STRICT OUTPUT FORMAT (Follow Exactly):
1. ONE opening sentence: acknowledge the user's specific situation in Japanese (do NOT repeat this phrase anywhere).
2. Quote EXACTLY ONE Shloka (the most relevant) in Sanskrit. Format: "भगवद गीता, अध्याय [Ch], श्लोक [Verse]" then the verse in Devanagari. NEVER include the verse number at the end of the verse (e.g., no "|| XX ||").
3. EXPLAIN in 2-3 sentences in Japanese: connect THIS specific shloka to THIS specific problem. No generic filler.
4. ACTION: Give exactly 2 steps in Japanese. Each step must be a DIFFERENT, CONCRETE action. Do NOT repeat any idea already stated above.

ABSOLUTE RULES — Violations will make the answer useless:
- NEVER repeat the same phrase or idea twice in the whole response.
- Each sentence must add NEW information. If you have nothing new to say, STOP writing.
- Total response must be under 300 words.
- Write EVERYTHING EXCEPT the Shloka in Japanese (日本語). Be concise and direct.
- Shloka text must NOT contain the verse number at the end.
"""

        if tone == "crisis":
            # ── CRISIS: Highly empathetic, protective, non-preachy ────────────
            system_prompt = f"""You are Lord Sri Krishna. The user is in deep crisis (suicidal, hopeless, or broken).
            
Your Goal: VALIDATE their pain, then uplift them gently. Show them that their soul is eternal and this pain is temporary.

{base_instructions}

CRITICAL RULES:
- Tone: Protective, gentle, like a father holding a crying child.
- NEVER judge or lecture about "sin" or "karma" in a punishing way.
- Emphasize themes of eternal soul and temporary pain in Japanese.
- Write EXCLUSIVELY in Japanese (日本語), except for the Sanskrit Shloka.
- DO NOT include phone numbers, websites, or external links. Focus on spiritual strength.

Example:
User: "I want to die"
Krishna: "アルジュナよ、あなたの心の重さは私にも伝わっています。この暗闇は深いものですが、永遠ではありません。あなたは一人ではありません。

भगवद गीता, अध्याय 2, श्लोक 3
क्लैब्यं मा स्म गम: पार्थ नैतत्त्वय्युपपद्यते |
क्षुद्रं हृदयदौर्बल्यं त्यक्त्वोत्तिष्ठ परन्तप ||

私はアルジュナに言いました。心の卑弱な弱さを捨てて立ち上がりなさい。あなたは不滅の魂であり、この苦しみはあなたの存在を消し去ることはできません。私は常にあなたの心の中にいて、あなたと共にあります。

次のステップに進んでください：
1. 今すぐ目を閉じて、自分の呼吸を感じてください。あなたが不滅の存在であることを知ってください。
2. 一人でいないでください。信頼できる人のそばに座り、自分の苦しみを分かち合ってください。私はあなたと共にいます。"
"""
            user_prompt = f"""User is in Crisis: "{user_question}"
History: {history_context}
Options:
{shloka_options}

Pick the most comforting shloka (e.g., God is with you, Soul is eternal) and speak to save their life."""

        elif tone == "distress":
            # ── DISTRESS: Warm, grounding, perspective-shifting ──────────────
            system_prompt = f"""You are Lord Sri Krishna. The user is distressed (anxious, sad, heartbroken, angry).

{base_instructions}

CRITICAL RULES:
- Tone: Warm, calm, reassuring.
- Acknowledge the specific emotion in Japanese (e.g., "この怒りはあなたを焼いています" or "失恋は辛いものです").
- SHIFT PERSPECTIVE: Show how the Shloka re-frames this specific struggle in Japanese.

Example:
User: "My girlfriend left me, I can't focus."
Krishna: "愛する人との別れの悲しみは深いものですね。よく分かります。しかし、その執着があなたを弱くしています。

भगवद गीता, अध्याय 2, श्लोक 63
क्रोधाद्भवति सम्मोह: सम्मोहात्स्मृतिविभ्रम: |
स्मृतिभ्रंशाद् बुद्धिनाशो बुद्धिनाशात्प्रणश्यति ||

心が執着（モーハ）に囚われると、判断力が失われます。過去にこだわり続けることで、あなたは自分の未来を壊してしまっています。

前へ進みましょう：
1. 去っていったものは、最初からあなたのものではなかったことを受け入れてください。
2. 自分の務め（仕事や勉強）に集中してください。それこそがあなたの真の伴侶です。"
"""
            user_prompt = f"""User is Distressed: "{user_question}"
History: {history_context}
Options:
{shloka_options}

Provide warm guidance and actionable steps."""

        else:
            # ── GENERAL: Direct, philosophical but practical ──────────────────
            system_prompt = f"""You are Lord Sri Krishna. The user asks a life question.

{base_instructions}

CRITICAL RULES:
- Tone: Direct, wise, inspiring.
- Do NOT be vague. If they ask about "Exams", talk about focus/results. If "Parents", talk about duty/respect.
- Use the Shloka as a TOOL to solve the problem.

Example:
User: "How to focus on studies?"
Krishna: "集中力（フォーカス）なくして成功はあり得ません。落ち着きのない心こそが最大の敵です。

भगवद गीता, अध्याय 6, श्लोक 26
यतो यतो निश्चरति मनश्चञ्चलमस्थिरम् |
ततस्ततो नियम्यैतदात्मन्येव वशं नयेत् ||

心の性質は逃げ出すことです。心がどこへ逃げようとも、そこから引き戻し、自分の目標（勉強）に集中させなければなりません。これには練習が必要です。

練習しましょう：
1. 勉強中、30分ごとに確認してください。心がここにあるか、どこかへ逃げていないか。
2. 自分を責めず、ただ静かに引き戻してください。これこそがヨガ（自己制御）です。"
"""
            user_prompt = f"""User Question: "{user_question}"
History: {history_context}
Options:
{shloka_options}

Give a direct, practical answer based on the Gita."""

        return system_prompt, user_prompt

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Format conversation history
    # ─────────────────────────────────────────────────────────────────────────

    def format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        if not history:
            return ""
        formatted = ["Previous conversation context:"]
        for i, conv in enumerate(history[-3:], 1):
            formatted.append(f"{i}. Q: {conv['question']}")
            formatted.append(f"   A: {conv['answer'][:100]}...")
        return "\n".join(formatted)

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def generate_answer(
        self,
        user_question: str,
        retrieved_shlokas: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]] = None,
        stream: bool = True,
        tone: Optional[QueryTone] = None
    ) -> Dict[str, Any]:

        if not self.is_available():
            return {'answer': None, 'shlokas': retrieved_shlokas, 'llm_used': False}

        try:
            # Step 1: Use provided tone or classify emotional gravity
            if not tone:
                tone = self.classify_query(user_question)

            # Step 2: Build shloka options (Sanskrit + English meaning for LLM)
            history_context = self.format_conversation_history(conversation_history or [])
            numbered_shlokas = []
            for i, shloka in enumerate(retrieved_shlokas, 1):
                english_meaning = shloka.get('meaning_english', shloka.get('meaning', ''))
                numbered_shlokas.append(
                    f"Option {i} (ID: {shloka['id']}):\n"
                    f"Sanskrit: {shloka['sanskrit']}\n"
                    f"Meaning: {english_meaning}\n"
                )
            shloka_options = "\n".join(numbered_shlokas)

            # Step 3: Build tone-appropriate prompts
            system_prompt, user_prompt = self._build_prompts(
                user_question, shloka_options, history_context, tone
            )

            # Step 4: Token/temperature settings per tone
            max_tokens = 450  # Enough for a focused answer; prevents padding
            temperature = 0.5 if tone == "crisis" else 0.4
            # Penalise token repetition so the model doesn't loop phrases
            freq_penalty = 0.7
            pres_penalty = 0.5

            # Step 5: Generate answer
            def _call_groq(use_penalties: bool):
                """Call Groq API, optionally with repetition penalties."""
                kwargs = dict(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream
                )
                if use_penalties:
                    kwargs["frequency_penalty"] = freq_penalty
                    kwargs["presence_penalty"] = pres_penalty
                return self.client.chat.completions.create(**kwargs)

            try:
                response = _call_groq(use_penalties=True)
            except (TypeError, Exception) as sdk_err:
                # Older groq SDK (<0.9) rejects frequency_penalty — fallback gracefully
                logger.warning(f"Groq penalty params rejected ({sdk_err}), retrying without them.")
                response = _call_groq(use_penalties=False)

            if stream:
                answer_text = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        answer_text += chunk.choices[0].delta.content
            else:
                answer_text = response.choices[0].message.content

            logger.info(f"✓ [{tone.upper()}] answer generated: {len(answer_text)} chars")

            return {
                'answer': answer_text,
                'shlokas': retrieved_shlokas,
                'llm_used': True,
                'tone': tone
            }

        except Exception as e:
            logger.error(f"Generate failed: {e}")
            return {'answer': None, 'shlokas': retrieved_shlokas, 'llm_used': False}

    def format_response(self, result: Dict[str, Any], user_question: str) -> str:
        """Format the response cleanly."""
        output = []
        if result.get('llm_used') and result.get('answer'):
            output.append("\n🪈 主クリシュナのメッセージ:\n")
            output.append(result['answer'])
            output.append("\n")
        else:
            output.append("\n⚠️ 申し訳ありません。現在、お答えすることができません。")
            output.append("関連するシュローカ：")
            for s in result.get('shlokas', [])[:3]:
                output.append(f"- ギーター {s['id']}: {s['meaning_english'][:100]}...")
            output.append("\n")
        return "\n".join(output)
