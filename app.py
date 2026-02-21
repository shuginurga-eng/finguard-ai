import streamlit as st
import google.generativeai as genai
import re
import json
import os
import requests
from bs4 import BeautifulSoup
import tldextract
from urllib.parse import urlparse
from typing import Dict, Any, List, Tuple

st.set_page_config(page_title="FinGuard AI", page_icon="🛡️", layout="wide")


# ----------------------------
# HELPERS
# ----------------------------
def pick_model() -> str:
    """Pick an available Gemini model to avoid 404 errors."""
    preferred = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
    ]
    try:
        models = list(genai.list_models())
        available = {m.name.replace("models/", "") for m in models}

        for m in preferred:
            if m in available:
                return m

        for m in models:
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                return m.name.replace("models/", "")
    except Exception:
        pass

    return "gemini-1.0-pro"


def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key

    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return st.session_state.get("GEMINI_API_KEY")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# ----------------------------
# CONFIG
# ----------------------------
GEMINI_API_KEY = get_api_key()

if not GEMINI_API_KEY:
    st.warning("GEMINI_API_KEY табылмады. Сол жақ Sidebar арқылы енгіз немесе secrets.toml қолдан.")
else:
    genai.configure(api_key=GEMINI_API_KEY)


# ----------------------------
# TRIGGERS / RULES (TEXT)
# ----------------------------
TRIGGERS = {
    "гарантированный доход": 25,
    "кепілдендірілген табыс": 25,
    "без риска": 20,
    "тәуекелсіз": 20,
    "300% в день": 30,
    "күніне 300%": 30,
    "удвоим депозит": 25,
    "депозитті екі есе": 25,
    "быстрый заработок": 20,
    "тез ақша": 20,
    "пассивный доход": 15,
    "пассив табыс": 15,
    "приведи друга": 25,
    "досыңды әкел": 25,
    "инвестиционный клуб": 15,
    "инвестиция": 8,
    "крипто-арбитраж": 18,
    "арбитраж": 12,
    "секретная стратегия": 15,
    "жабық стратегия": 15,
    "ограниченные места": 12,
    "орын шектеулі": 12,
    "тек бүгін": 10,
    "срочно": 10,
    "жедел": 10,
    "whatsapp": 8,
    "телеграм": 6,
    "личные сообщения": 8,
    "жеке хабарлама": 8,
    "перевод на карту": 12,
    "картаға аудар": 12,
    "квитанция": 6,
    "чек": 6,
    "инн": 6,
    "бин": 6,
}

SOFT_PATTERNS = [
    (r"\b(гарант|кепіл)\w*", 15),
    (r"\b(табыс|доход|пайда|прибыл)\w*", 8),
    (r"\b(инвест|investment)\w*", 8),
    (r"\b(процент|%|пайыз)\w*", 10),
    (r"\b(крипто|crypto|bitcoin|usdt)\w*", 12),
    (r"\b(арбитраж)\w*", 12),
    (r"\b(тез|быстр)\w*\s+(ақша|заработ)\w*", 15),
    (r"\b(реф|дос|friend)\w*", 12),
]

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\-\s\(\)]{8,}\d)")
MONEY_RE = re.compile(r"(\d[\d\s]{0,6}(?:k|K|млн|млрд|теңге|тг|\$|usd|₸|руб|₽|eur|€))", re.IGNORECASE)


def rule_based_score(text: str) -> Tuple[int, List[str], Dict[str, Any]]:
    t = normalize_text(text)
    found = []
    raw_points = 0

    for phrase, pts in TRIGGERS.items():
        if phrase in t:
            found.append(phrase)
            raw_points += pts

    for pat, pts in SOFT_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            found.append(f"үлгі: {pat}")
            raw_points += pts

    if URL_RE.search(text):
        found.append("сілтеме/URL бар")
        raw_points += 8

    if PHONE_RE.search(text):
        found.append("телефон нөмірі бар")
        raw_points += 6

    if MONEY_RE.search(text):
        found.append("ақша/сомма айтылған")
        raw_points += 6

    rule_score = min(100, raw_points)
    debug = {"raw_points": raw_points, "matched": found}
    return rule_score, found, debug


# ----------------------------
# JSON PARSER (ROBUST)
# ----------------------------
def extract_json(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    return m.group(0).strip() if m else cleaned


def safe_json_loads(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    s = extract_json(text)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)

    if s.count("'") > s.count('"'):
        s = s.replace("'", '"')

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


# ----------------------------
# GEMINI (TEXT ANALYSIS)
# ----------------------------
def gemini_analyze(text: str) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        return {
            "ai_score": 50,
            "confidence": 0.2,
            "red_flags": ["API key жоқ — тек rule-based режим"],
            "verdict": "Gemini қосылмаған. Rule-based нәтижесі ғана қолданылады.",
            "recommendations": ["GEMINI_API_KEY енгізіңіз (secrets.toml немесе terminal env)."],
            "model_used": "—",
        }

    model_name = pick_model()
    model = genai.GenerativeModel(model_name)

    prompt = f"""
Сен Қазақстандағы қаржылық қауіпсіздік және анти-скам сарапшысысың.
Мәтінді талдап, ТЕК қана JSON қайтар.

ҚАТАҢ JSON:
{{
  "ai_score": 0-100,
  "confidence": 0.0-1.0,
  "red_flags": ["қысқа пункттер"],
  "verdict": "1-2 сөйлем",
  "recommendations": ["3-5 ұсыныс"]
}}

Мәтін:
\"\"\"{text}\"\"\"
"""

    try:
        resp = model.generate_content(prompt)
        data = safe_json_loads(resp.text)
        return {
            "ai_score": int(max(0, min(100, data.get("ai_score", 50)))),
            "confidence": float(max(0.0, min(1.0, data.get("confidence", 0.5)))),
            "red_flags": list(data.get("red_flags", []))[:12],
            "verdict": str(data.get("verdict", "Анықталмады")),
            "recommendations": list(data.get("recommendations", []))[:8],
            "model_used": model_name,
        }
    except Exception as e:
        return {
            "ai_score": 50,
            "confidence": 0.2,
            "red_flags": [f"Gemini қатесі: {str(e)}"],
            "verdict": "Gemini уақытша жауап бере алмады. Rule-based арқылы бағалаңыз.",
            "recommendations": ["Кейін қайта тексеріңіз немесе басқа мәтін енгізіңіз."],
            "model_used": model_name,
        }


# ----------------------------
# WEBSITE AUDIT (URL)
# ----------------------------
SUSPICIOUS_TLDS = {"xyz", "top", "site", "live", "click", "icu", "online", "shop", "win", "loan", "buzz", "monster", "pw"}
LEGAL_HINTS = ["бин", "лиценз", "оферта", "terms", "privacy", "policy", "келісім", "құжат", "company", "legal"]
SCAM_HINTS = ["гарант", "кепіл", "тәуекелсіз", "без риска", "300%", "удвоим", "тез ақша", "реферал", "приведи друга", "арбитраж", "usdt", "bitcoin"]

SCAM_TYPE_RULES = {
    "Phishing": ["логин", "пароль", "verify", "verification", "подтвердите", "банк карта", "cvv", "sms", "код", "account locked", "аккаунт бұғат"],
    "Pyramid": ["реферал", "приведи друга", "команда", "матрица", "внеси депозит", "пайыз күніне", "гарант", "кепіл табыс"],
    "Fake broker": ["broker", "trading", "форекс", "forex", "binance support", "инвест платформа", "signal", "сигнал", "аналитик", "менеджер"],
    "Scam shop": ["скидка 90", "төлем алдын ала", "наложенный", "доставка", "только сегодня", "instagram shop", "whatsapp заказ"],
}


def fetch_site_text(url: str) -> Tuple[str, str, str]:
    """
    Returns: (text, final_url, mode)
    mode = "full" | "domain_only"
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7,kk-KZ;q=0.6",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        r = requests.get(url, headers=headers, timeout=12, allow_redirects=True)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = " ".join(soup.get_text(separator=" ").split())
        return text[:20000], r.url, "full"

    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status in (401, 403):
            # сайт контентті бермеді -> домен бойынша ғана
            return "", url, "domain_only"
        raise
    except Exception:
        # басқа жағдайлар: timeout, dns, т.б. -> домен бойынша ғана
        return "", url, "domain_only"


def domain_signals(final_url: str) -> Tuple[int, List[str], str]:
    u = urlparse(final_url)
    ext = tldextract.extract(u.netloc)
    domain = f"{ext.domain}.{ext.suffix}".lower()

    points = 0
    flags = []

    if u.scheme == "http":
        points += 15
        flags.append("HTTPS жоқ (HTTP)")

    if ext.suffix.lower() in SUSPICIOUS_TLDS:
        points += 20
        flags.append(f"Күдікті домен аймағы: .{ext.suffix}")

    if len(ext.domain) <= 4:
        points += 6
        flags.append("Домен аты өте қысқа")

    if re.search(r"\d{3,}", ext.domain):
        points += 10
        flags.append("Домен атауында көп сан бар")

    return min(50, points), flags, domain


def content_signals(text: str) -> Tuple[int, List[str], Dict[str, Any]]:
    t = text.lower()
    points = 0
    flags = []
    debug = {}

    found_scam = [h for h in SCAM_HINTS if h in t]
    if found_scam:
        points += min(40, 8 * len(found_scam))
        flags.append("Күдікті уәде/маркетинг сөздері: " + ", ".join(found_scam[:6]) + ("..." if len(found_scam) > 6 else ""))

    legal_found = [h for h in LEGAL_HINTS if h in t]
    if not legal_found:
        points += 15
        flags.append("Заңды ақпарат әлсіз (terms/privacy/license/bin табылмады)")

    has_phone = bool(PHONE_RE.search(text))
    has_email = bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))
    has_bin = bool(re.search(r"\b\d{12}\b", text))

    if not (has_phone or has_email):
        points += 10
        flags.append("Контакт деректері аз (телефон/email жоқ)")

    if not has_bin:
        points += 8
        flags.append("БИН/реквизит табылмады")

    debug["found_scam_hints"] = found_scam
    debug["legal_hints_found"] = legal_found
    debug["has_phone"] = has_phone
    debug["has_email"] = has_email
    debug["has_bin"] = has_bin

    return min(50, points), flags, debug


def split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[\.\!\?\n])\s+", text)
    return [c.strip() for c in chunks if c.strip()]


def get_quotes(text: str, keywords: List[str], limit: int = 5) -> List[str]:
    sentences = split_sentences(text)
    quotes = []
    for s in sentences:
        low = s.lower()
        if any(k.lower() in low for k in keywords):
            quotes.append(s[:220])
        if len(quotes) >= limit:
            break
    return quotes


def rule_scam_type(text: str) -> Tuple[str, List[str]]:
    t = text.lower()
    hits = {}
    for typ, keys in SCAM_TYPE_RULES.items():
        score = sum(1 for k in keys if k.lower() in t)
        if score > 0:
            hits[typ] = score

    if not hits:
        return "Unknown", []

    typ = sorted(hits.items(), key=lambda x: x[1], reverse=True)[0][0]
    keywords = SCAM_TYPE_RULES.get(typ, [])[:10]
    return typ, keywords


# ----------------------------
# SCORING / LABEL
# ----------------------------
def fuse_scores(rule_score: int, ai_score: int, confidence: float) -> int:
    conf = max(0.15, min(1.0, confidence))
    w_ai = 0.55 * conf
    w_rule = 1.0 - w_ai
    final = int(round(w_ai * ai_score + w_rule * rule_score))
    return max(0, min(100, final))


def risk_label(score: int) -> Tuple[str, str]:
    if score >= 70:
        return "ӨТЕ ЖОҒАРЫ ҚАУІП", "error"
    if score >= 35:
        return "КҮДІКТІ", "warning"
    return "ТӨМЕН ҚАУІП", "success"


def render_checklist():
    st.markdown("""
### ✅ “Қалай тексеруге болады?” Чеклист
**1) Лицензия/заңдылық**
- Компания атауы + БИН бар ма?
- Ресми лицензия/реттеуші органға сілтеме бар ма?

**2) Уәделер**
- “Кепілдендірілген табыс”, “тәуекелсіз”, “күніне 300%” сияқты сөздер бар ма? → күмән

**3) Ақша аудару тәсілі**
- “Картаға таста”, “жеке тұлғаға аудар”, “скрин жібер” → өте қауіпті

**4) Байланыс**
- Тек WhatsApp/Telegram ғана ма?
- Мекенжай, email, келісім-шарт бар ма?

**5) Пікір/іздеу**
- Google/News арқылы атауын ізде
- “scam”, “pyramid”, “алаяқ” сөздерімен тексер
""")


# ----------------------------
# UI
# ----------------------------
st.title("🛡️ FinGuard AI")
st.caption("Қаржылық ұсыныстарды/посттарды және сайттарды тексеру: AI + rule-based триггерлер")

with st.sidebar:
    st.subheader("⚙️ Баптау")
    if not GEMINI_API_KEY:
        key_input = st.text_input("GEMINI_API_KEY енгіз", type="password")
        if st.button("Кілтті сақтау"):
            st.session_state["GEMINI_API_KEY"] = key_input
            st.rerun()

    st.markdown("### 📌 Демо мәтіндер")
    demo_safe = "Банк депозиті: жылдық 15%, тәуекелдер көрсетілген, келісім-шарт бар, ресми сайт бар."
    demo_mid = "Инвестиция жасасаңыз, айына 20% табыс. Орны шектеулі, жедел жазыңыз."
    demo_scam = "Кепілдендірілген табыс! 300% күніне! Досыңды әкел — бонус! Тек бүгін, картаға аудар да скрин жібер."

    choice = st.radio("Таңда:", ["—", "Safe", "Suspicious", "Obvious scam"])
    if choice == "Safe":
        st.session_state["demo_text"] = demo_safe
    elif choice == "Suspicious":
        st.session_state["demo_text"] = demo_mid
    elif choice == "Obvious scam":
        st.session_state["demo_text"] = demo_scam

    st.divider()
    if st.button("✅ Чеклистті көрсету"):
        st.session_state["show_checklist"] = not st.session_state.get("show_checklist", False)

if st.session_state.get("show_checklist", False):
    render_checklist()


tab1, tab2 = st.tabs(["📝 Мәтін тексеру", "🌐 Сайт (URL) тексеру"])


# ----------------------------
# TAB 1: TEXT
# ----------------------------
with tab1:
    text_default = st.session_state.get("demo_text", "")
    user_input = st.text_area("Мәтінді осы жерге қой:", value=text_default, height=180)

    colA, colB = st.columns([1, 1])
    with colA:
        run_btn = st.button("🔍 Мәтінді талдау", use_container_width=True)
    with colB:
        st.button("🧹 Тазалау", on_click=lambda: st.session_state.update({"demo_text": ""}), use_container_width=True)

    if run_btn:
        if not user_input.strip():
            st.error("Мәтін бос!")
        else:
            with st.spinner("Талдау жүріп жатыр..."):
                rule_score, rule_flags, debug = rule_based_score(user_input)
                ai = gemini_analyze(user_input)
                final_score = fuse_scores(rule_score, ai["ai_score"], ai["confidence"])
                label, kind = risk_label(final_score)

                scam_type, kw = rule_scam_type(user_input)
                quotes = get_quotes(user_input, kw or list(TRIGGERS.keys()), limit=5)

            st.divider()
            if kind == "error":
                st.error(f"Қауіп деңгейі: {final_score}% — {label}")
            elif kind == "warning":
                st.warning(f"Қауіп деңгейі: {final_score}% — {label}")
            else:
                st.success(f"Қауіп деңгейі: {final_score}% — {label}")

            st.progress(final_score / 100)
            st.write(f"**Scam type:** `{scam_type}`")

            left, right = st.columns([1.2, 1])

            with left:
                st.subheader("🚩 Табылған белгілер")
                all_flags = []
                all_flags.extend([f"AI: {x}" for x in ai.get("red_flags", [])])
                all_flags.extend([f"RULE: {x}" for x in rule_flags])

                if not all_flags:
                    st.write("✅ Айқын қауіпті триггерлер табылмады.")
                else:
                    for f in all_flags[:20]:
                        st.write(f"- {f}")

                st.subheader("🧾 Дәлел (цитаталар)")
                if quotes:
                    for q in quotes:
                        st.write(f"> {q}")
                else:
                    st.write("Дәлел болатын сөйлем табылмады (мәтін қысқа болуы мүмкін).")

                st.subheader("🧠 Қорытынды")
                st.info(ai.get("verdict", "—"))

            with right:
                st.subheader("📌 Ұсыныстар")
                recs = ai.get("recommendations", [])
                if not recs:
                    recs = [
                        "Компанияның лицензиясын тексеріңіз (ресми орган сайтында).",
                        "Келісім-шартсыз ақша аудармаңыз.",
                        "‘Кепілдендірілген табыс’ сияқты уәделерден сақ болыңыз.",
                    ]
                for r in recs[:8]:
                    st.write(f"✅ {r}")

                st.subheader("📊 Техникалық көрсеткіш")
                st.write(f"- Model used: **{ai.get('model_used','—')}**")
                st.write(f"- AI score: **{ai['ai_score']}%**")
                st.write(f"- Rule score: **{rule_score}%**")
                st.write(f"- Confidence: **{ai['confidence']:.2f}**")


# ----------------------------
# TAB 2: URL
# ----------------------------
with tab2:
    url_input = st.text_input("Сайт сілтемесін енгізіңіз (мысалы: example.com):")

    if st.button("🔎 Сайтты тексеру", use_container_width=True):
        if not url_input.strip():
            st.error("URL бос!")
        else:
            try:
                with st.spinner("Сайттан ақпарат алып жатыр..."):
                    text, final_url, mode = fetch_site_text(url_input)

                    d_score, d_flags, domain = domain_signals(final_url)

                    if mode == "full" and text.strip():
                        c_score, c_flags, dbg = content_signals(text)
                        scam_type, kw = rule_scam_type(text)
                        quotes = get_quotes(text, kw or SCAM_HINTS, limit=5)
                        ai_site = gemini_analyze("Сайт мәтіні:\n" + text[:6000]) if GEMINI_API_KEY else None
                    else:
                        # контент жоқ (403/anti-bot/timeout) -> домен бойынша ғана
                        c_score, c_flags, dbg = 0, ["Контентке қолжетім жоқ (403/anti-bot/timeout). Тек домен бойынша бағаланды."], {}
                        scam_type, kw = "Unknown", []
                        quotes = []
                        ai_site = None

                    website_score = min(100, d_score + c_score)
                    label, kind = risk_label(website_score)

                st.divider()
                if kind == "error":
                    st.error(f"Website Risk: {website_score}% — {label}")
                elif kind == "warning":
                    st.warning(f"Website Risk: {website_score}% — {label}")
                else:
                    st.success(f"Website Risk: {website_score}% — {label}")

                st.progress(website_score / 100)
                st.write(f"**Домен:** {domain}")
                st.write(f"**Final URL:** {final_url}")
                st.write(f"**Scam type:** `{scam_type}`")

                st.subheader("🚩 Табылған белгілер")
                for f in (d_flags + c_flags):
                    st.write(f"- {f}")

                st.subheader("🧾 Дәлел (цитаталар)")
                if quotes:
                    for q in quotes:
                        st.write(f"> {q}")
                else:
                    st.write("Цитата табылмады немесе сайт контенті оқылмады.")

                if ai_site:
                    st.subheader("🧠 AI қорытынды")
                    st.info(ai_site.get("verdict", "—"))

                    st.subheader("📌 AI ұсыныстар")
                    for r in ai_site.get("recommendations", [])[:6]:
                        st.write(f"✅ {r}")

                    st.subheader("📊 AI техникалық")
                    st.write(f"- Model used: **{ai_site.get('model_used','—')}**")
                    st.write(f"- AI score: **{ai_site['ai_score']}%**")
                    st.write(f"- Confidence: **{ai_site['confidence']:.2f}**")

            except Exception as e:
                st.error(f"Сайтты тексеру кезінде қате: {e}")