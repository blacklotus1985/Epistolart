from multi_rake import Rake

def extract_keywords(text,lang='it',percentile = 40):


    rake = Rake(
        min_chars=3,
        max_words=10,
        min_freq=1,
        language_code=lang,  # 'en'
        stopwords=None,  # {'and', 'of'}
        lang_detect_threshold=10,
        max_words_unknown_lang=2,
        generated_stopwords_percentile=percentile,
        generated_stopwords_max_len=3,
        generated_stopwords_min_freq=2,
    )
    kw = rake.apply(text)
    return kw

