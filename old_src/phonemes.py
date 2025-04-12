phoneme_map = {
    "p": {
        "voicing": "voiceless",
        "place": "bilabial",
        "manner": "stop",
        "example_word": "pat",
        "coaching_tip": "Press your lips together and release a small puff of air."
    },
    "b": {
        "voicing": "voiced",
        "place": "bilabial",
        "manner": "stop",
        "example_word": "bat",
        "coaching_tip": "Press your lips together and make a sound with your voice."
    },
    "m": {
        "voicing": "voiced",
        "place": "bilabial",
        "manner": "nasal",
        "example_word": "mat",
        "coaching_tip": "Press your lips together and hum through your nose."
    },
    "f": {
        "voicing": "voiceless",
        "place": "labiodental",
        "manner": "fricative",
        "example_word": "fun",
        "coaching_tip": "Bite your bottom lip gently and blow out air."
    },
    "v": {
        "voicing": "voiced",
        "place": "labiodental",
        "manner": "fricative",
        "example_word": "van",
        "coaching_tip": "Bite your bottom lip and use your voice while blowing out air."
    },
    "t": {
        "voicing": "voiceless",
        "place": "alveolar",
        "manner": "stop",
        "example_word": "top",
        "coaching_tip": "Tap your tongue quickly behind your top front teeth."
    },
    "d": {
        "voicing": "voiced",
        "place": "alveolar",
        "manner": "stop",
        "example_word": "dog",
        "coaching_tip": "Tap your tongue behind your top front teeth while using your voice."
    },
    "s": {
        "voicing": "voiceless",
        "place": "alveolar",
        "manner": "fricative",
        "example_word": "sun",
        "coaching_tip": "Place your tongue close to the ridge behind your top teeth and blow air gently."
    },
    "z": {
        "voicing": "voiced",
        "place": "alveolar",
        "manner": "fricative",
        "example_word": "zoo",
        "coaching_tip": "Place your tongue near your top teeth ridge and buzz your voice gently."
    },
    "n": {
        "voicing": "voiced",
        "place": "alveolar",
        "manner": "nasal",
        "example_word": "net",
        "coaching_tip": "Place your tongue behind your top teeth and hum through your nose."
    },
    "l": {
        "voicing": "voiced",
        "place": "alveolar",
        "manner": "liquid",
        "example_word": "lip",
        "coaching_tip": "Place the tip of your tongue behind your top teeth and let air flow around it."
    },
    "r": {
        "voicing": "voiced",
        "place": "alveolar",
        "manner": "liquid",
        "example_word": "red",
        "coaching_tip": "Pull your tongue back slightly without touching the top of your mouth."
    },
    "k": {
        "voicing": "voiceless",
        "place": "velar",
        "manner": "stop",
        "example_word": "cat",
        "coaching_tip": "Raise the back of your tongue to the soft part of the roof of your mouth and release air."
    },
    "g": {
        "voicing": "voiced",
        "place": "velar",
        "manner": "stop",
        "example_word": "go",
        "coaching_tip": "Raise the back of your tongue and use your voice as you release air."
    },
    "h": {
        "voicing": "voiceless",
        "place": "glottal",
        "manner": "fricative",
        "example_word": "hat",
        "coaching_tip": "Breathe out gently as if you're fogging a mirror."
    },
    "j": {
        "voicing": "voiced",
        "place": "palatal",
        "manner": "affricate",
        "example_word": "jump",
        "coaching_tip": "Start with a 'd' sound and quickly slide into a 'zh' sound."
    },
    "ch": {
        "voicing": "voiceless",
        "place": "palatal",
        "manner": "affricate",
        "example_word": "chip",
        "coaching_tip": "Push your tongue to the roof of your mouth and release it quickly like a sneeze sound."
    },
    "sh": {
        "voicing": "voiceless",
        "place": "palatal",
        "manner": "fricative",
        "example_word": "shoe",
        "coaching_tip": "Push air between your tongue and the roof of your mouth while keeping your lips rounded."
    },
    "zh": {
        "voicing": "voiced",
        "place": "palatal",
        "manner": "fricative",
        "example_word": "measure",
        "coaching_tip": "Make a buzzing 'sh' sound using your voice."
    },
    "y": {
        "voicing": "voiced",
        "place": "palatal",
        "manner": "glide",
        "example_word": "yes",
        "coaching_tip": "Raise the middle of your tongue toward the roof of your mouth and say 'ee' quickly."
    },
    "w": {
        "voicing": "voiced",
        "place": "bilabial",
        "manner": "glide",
        "example_word": "wet",
        "coaching_tip": "Round your lips and glide into the next sound."
    },
    "a": {
        "voicing": "voiced",
        "place": "open",
        "manner": "vowel",
        "example_word": "apple",
        "coaching_tip": "Open your mouth wide and say 'ah'."
    },
    "e": {
        "voicing": "voiced",
        "place": "front",
        "manner": "vowel",
        "example_word": "egg",
        "coaching_tip": "Smile slightly and say 'eh' with your tongue low and front."
    },
    "i": {
        "voicing": "voiced",
        "place": "front",
        "manner": "vowel",
        "example_word": "igloo",
        "coaching_tip": "Raise your tongue high and close to the front, like saying 'ee'."
    },
    "o": {
        "voicing": "voiced",
        "place": "mid-back",
        "manner": "vowel",
        "example_word": "octopus",
        "coaching_tip": "Round your lips and say 'aw' or 'oh'."
    },
    "u": {
        "voicing": "voiced",
        "place": "back",
        "manner": "vowel",
        "example_word": "umbrella",
        "coaching_tip": "Say 'uh' with your tongue low and lips relaxed."
    }
}

phoneme_similarity_groups = {
    "s": ["s", "ʃ", "z", "tʃ", "dʒ", "ʒ"],
    "ʃ": ["s", "ʃ", "z", "tʃ", "dʒ", "ʒ"],
    "z": ["s", "ʃ", "z", "tʃ", "dʒ", "ʒ"],
    "tʃ": ["s", "ʃ", "z", "tʃ", "dʒ", "ʒ"],
    "dʒ": ["s", "ʃ", "z", "tʃ", "dʒ", "ʒ"],
    "ʒ": ["s", "ʃ", "z", "tʃ", "dʒ", "ʒ"],
    "θ": ["θ", "ð", "f", "v"],
    "ð": ["θ", "ð", "f", "v"],
    "f": ["θ", "ð", "f", "v"],
    "v": ["θ", "ð", "f", "v"],
    "ɛ": ["æ", "ɛ", "ɪ", "ʌ", "ɒ"],
    "ɪ": ["æ", "ɛ", "ɪ", "ʌ", "ɒ"],
    "ʌ": ["æ", "ɛ", "ɪ", "ʌ", "ɒ"],
    "ɒ": ["æ", "ɛ", "ɪ", "ʌ", "ɒ"],
    "iː": ["iː", "uː", "ɔː", "ɑː", "ɜː"],
    "uː": ["iː", "uː", "ɔː", "ɑː", "ɜː"],
    "ɔː": ["iː", "uː", "ɔː", "ɑː", "ɜː"],
    "ɑ": ["æ", "ɛ", "ɪ", "ʌ", "ɒ", "iː", "uː", "ɔː", "ɑː", "a", "ɜː"],
    "ɑː": ["æ", "ɛ", "ɪ", "ʌ", "ɒ", "iː", "uː", "ɔː", "ɑː", "a", "ɜː"],
    "æ": ["æ", "ɛ", "ɪ", "ʌ", "ɒ", "iː", "uː", "ɔː", "ɑː", "a", "ɜː"],
    "ɜː": ["iː", "uː", "ɔː", "ɑː", "ɜː"],
    "aɪ": ["aɪ", "eɪ", "ɔɪ", "aʊ", "ɪə", "eə", "ʊə"],
    "eɪ": ["aɪ", "eɪ", "ɔɪ", "aʊ", "ɪə", "eə", "ʊə"],
    "ɔɪ": ["aɪ", "eɪ", "ɔɪ", "aʊ", "ɪə", "eə", "ʊə"],
    "aʊ": ["aɪ", "eɪ", "ɔɪ", "aʊ", "ɪə", "eə", "ʊə"],
    "ɪə": ["aɪ", "eɪ", "ɔɪ", "aʊ", "ɪə", "eə", "ʊə"],
    "eə": ["aɪ", "eɪ", "ɔɪ", "aʊ", "ɪə", "eə", "ʊə"],
    "ʊə": ["aɪ", "eɪ", "ɔɪ", "aʊ", "ɪə", "eə", "ʊə"],
    "m": ["m", "n", "ŋ"],
    "n": ["m", "n", "ŋ"],
    "ŋ": ["m", "n", "ŋ"],
    "l": ["l", "r"],
    "r": ["l", "r"]
}
