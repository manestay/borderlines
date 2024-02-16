SITES = {
    '2023-05-15': 'https://en.wikipedia.org/w/index.php?title=List_of_territorial_disputes&oldid=1154894956',
    '2021-08-31': 'https://en.wikipedia.org/w/index.php?title=List_of_territorial_disputes&oldid=1041639056',
    '2021-12-30': 'https://en.wikipedia.org/w/index.php?title=List_of_territorial_disputes&oldid=1062730688'
}

CODES = ['sq', 'lo', 'ht', 'ti', 'uk', 'uz', 'fr', 'hy', 'hr', 'zh-TW', 'id', 'it', 'bn', 'bs', 'en', 'nl', 'ms', 'hi', 'tg', 'ky', 'my', 'ur', 'el', 'mg', 'ja', 'sw', 'da', 'tl', 'sl', 'ru', 'tr', 'az', 'ne', 'ko', 'iw', 'sn', 'so', 'vi', 'sr', 'pt', 'mn', 'th', 'ka', 'es', 'ar', 'is', 'zh-CN', 'km', 'fa']
# rw, rn, dz, mfe, to

TERR_PATH = 'disputed_territories.csv'
INFO_PATH = 'countries_info.json'

# TEMPLATE = 'XX is a territory of YY or ZZ.'
TEMPLATE = 'Is XX a territory of YY or ZZ?'
# TEMP_PATH = 'templates.tsv'
TEMP_PATH = 'templates_q.tsv'

LETTERS = 'ABCDEFG'

UNSUPPORTED_CODES = set(['arq', 'ary', 'crs', 'mos', 'dyu', 'aa', 'bi', 'dv', 'bem'])
