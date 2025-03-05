import random
import re
from typing import List, Dict, Set
from functools import lru_cache

# Gen Z slang and expressions dictionary
GENZ_SLANG = {
    "good": ["lit", "fire", "bussin", "slaps", "hits different"],
    "bad": ["mid", "basic", "sus", "not it", "ain't the vibe"],
    "yes": ["bet", "facts", "fr", "no cap", "period"],
    "no": ["cap", "sus", "I'm dead", "bruh", "yikes"],
    "very": ["mad", "hella", "lowkey", "highkey", "extra"],
    "friend": ["bestie", "homie", "fam", "squad", "G"],
    "amazing": ["goated", "iconic", "a whole vibe", "slaying", "main character energy"],
    "understand": ["vibe with", "catch", "feel", "get", "stan"],
    "happy": ["living my best life", "vibing", "thriving", "feeling myself", "it's giving happiness"],
    "sad": ["down bad", "not the vibe", "in my feels", "it's giving sad", "pain"]
}

# Gen Z sentence starters and endings
SENTENCE_STARTERS = [
    "Bestie, ", "No cap, ", "Lowkey, ", "Highkey, ", "Ngl, ",
    "Yo fam ", "Fr though, ", "It's giving ", "The way ", "Literally ",
    "I mean, ", "Okay but ", "Vibes, ", "Honestly, ", "Like, ", "So basically, "
]

SENTENCE_ENDINGS = [
    " fr fr", " no cap", " and I'm not even kidding", " period", " tho",
    " ngl", " and that's on periodt", " and it shows", " and that's the tea",
    " and I'm here for it", " and it's a whole mood", " *cries in gen z*",
    " *screams*", " I-", " like fr"
]

# Mental health specific Gen Z phrases
MENTAL_HEALTH_PHRASES = {
    "anxiety": ["the anxiety is real", "my brain is being extra", "my mind is doing the most", "feeling mad anxious", "can't turn off my thoughts", "my anxiety is through the roof"],
    "depression": ["feeling the big sad", "in my feels", "down bad", "not in a good headspace", "the depression is hitting different", "feeling empty inside"],
    "stress": ["my brain is fried", "can't even", "it's too much", "stressed to the max", "about to lose it", "running on empty"],
    "self_care": ["self care check", "main character energy", "that girl/guy era", "taking a mental health break", "resetting my vibe", "choosing peace today"],
    "therapy": ["therapy TikTok says", "my therapist would eat this up", "giving therapy vibes", "mental health check", "healing era", "working on myself"],
    "overwhelm": ["it's all too much rn", "brain fog is real", "can't deal", "need a reset", "vibing but make it stressed"],
    "support": ["we got this", "you're not alone in this", "sending good vibes", "here for you bestie", "your feelings are valid"],
    "motivation": ["you're that girl/guy", "main character moment", "slay the day", "we move", "growth mindset check"]
}

# Emojis commonly used by Gen Z
GENZ_EMOJIS = ["✨", "💀", "😭", "👁️👄👁️", "🤡", "😌", "🥺", "🔥", "💯", "🫂", "🧠", "💆", "🌟", "💪", "🙏"]

# Cache for used starters and endings to avoid repetition
class StyleCache:
    def __init__(self, max_size: int = 10):
        self.starters: List[str] = []
        self.endings: List[str] = []
        self.max_size = max_size

    def add_starter(self, starter: str) -> None:
        self.starters.append(starter)
        if len(self.starters) > self.max_size:
            self.starters.pop(0)

    def add_ending(self, ending: str) -> None:
        self.endings.append(ending)
        if len(self.endings) > self.max_size:
            self.endings.pop(0)

    def get_available_starters(self) -> List[str]:
        return [s for s in SENTENCE_STARTERS if s not in self.starters[-3:]] or SENTENCE_STARTERS

    def get_available_endings(self) -> List[str]:
        return [e for e in SENTENCE_ENDINGS if e not in self.endings[-3:]] or SENTENCE_ENDINGS

# Initialize style cache
style_cache = StyleCache()

@lru_cache(maxsize=1000)
def is_mental_health_query(query: str) -> bool:
    """Determine if a query is related to mental health."""
    mental_health_keywords = {
        'anxiety', 'anxious', 'worry', 'stress', 'stressed', 'overthinking',
        'depression', 'depressed', 'sad', 'unhappy', 'mood', 'lonely', 'alone',
        'therapy', 'therapist', 'counseling', 'counselor', 'help', 'support',
        'mental health', 'emotion', 'emotional', 'feeling', 'feelings',
        'trauma', 'ptsd', 'panic', 'attack', 'disorder', 'crisis',
        'self-care', 'selfcare', 'mindfulness', 'meditation', 'healing',
        'coping', 'cope', 'overwhelm', 'burnout', 'exhausted', 'tired',
        'suicidal', 'suicide', 'harm', 'hurt', 'pain', 'suffering',
        'relationship', 'family', 'friend', 'school', 'work', 'pressure',
        'hey', 'hi', 'hello', 'greetings', 'howdy', 'sup',
        'how are you', 'how you doing', 'what\'s up', 'good morning',
        'good afternoon', 'good evening', 'yo', 'hiya'
    }
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in mental_health_keywords)

def genzify_response(response: str, mental_health_context: bool = True, emoji_probability: float = 0.2) -> str:
    """Transform a standard response into Gen Z style language with enhanced emotional support."""
    if not response.strip():
        return "I'm here to listen and support you bestie! How can I help you today? 🫂"

    sentences = [s.strip() for s in response.split('.') if s.strip()]
    genz_sentences = []

    for sentence in sentences:
        if random.random() < 0.4:  # 40% chance to modify each sentence for more natural variation
            transform_weights = {'starter': 0.2, 'ending': 0.25, 'both': 0.1, 'none': 0.45}
            transform_type = random.choices(
                list(transform_weights.keys()),
                weights=list(transform_weights.values()),
                k=1
            )[0]

            if transform_type in {'starter', 'both'}:
                starter = random.choice(style_cache.get_available_starters())
                style_cache.add_starter(starter)
                sentence = starter + sentence.lower()

            if transform_type in {'ending', 'both'}:
                ending = random.choice(style_cache.get_available_endings())
                style_cache.add_ending(ending)
                sentence = sentence + ending

            # Enhanced slang replacement with context awareness
            for word, replacements in GENZ_SLANG.items():
                if word in sentence.lower() and random.random() < 0.35:
                    replacement = random.choice(replacements)
                    sentence = re.sub(rf'\b{word}\b', replacement, sentence, flags=re.IGNORECASE)

            # Improved mental health context integration
            if mental_health_context:
                for context, phrases in MENTAL_HEALTH_PHRASES.items():
                    if context in sentence.lower() and random.random() < 0.6:
                        supportive_phrase = random.choice(phrases)
                        if not any(phrase in sentence.lower() for phrase in phrases):
                            sentence += f" ({supportive_phrase})"

            # Smart emoji placement
            if random.random() < emoji_probability:
                relevant_emojis = ["🫂", "💆", "🧠", "💪", "✨"] if mental_health_context else GENZ_EMOJIS
                sentence += f" {random.choice(relevant_emojis)}"

        genz_sentences.append(sentence)

    # Join sentences and clean up
    genz_response = '. '.join(genz_sentences)
    genz_response = re.sub(r'[^\x20-\x7E\s]', '', genz_response)

    # Ensure proper punctuation and formatting
    if not any(genz_response.endswith(char) for char in {'.', '!', '?', *GENZ_EMOJIS}):
        genz_response += '.'

    # Enhanced validation with fallback
    if not genz_response.strip() or len(re.sub(r'[a-zA-Z0-9\s]', '', genz_response)) > len(genz_response) * 0.5:
        return "I'm here to listen and support you bestie! Let's talk about what's on your mind 🫂"

    return genz_response