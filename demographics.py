from collections import OrderedDict, defaultdict
import re

# get demographic info for countries
# NOTE: using 3 different packages since no 1 package is complete :(
import pycountry
from babel import Locale
from babel.languages import get_territory_language_info
from bs4 import BeautifulSoup
from Countrydetails import country as country_cd

from lib import UNSUPPORTED_CODES


def get_most_spoken(c_code):
    lang_info_d = get_territory_language_info(c_code)
    most_perc = 0
    l_code = ''
    for k, v in lang_info_d.items():
        pop_perc = v['population_percent']
        if pop_perc > most_perc and (k not in UNSUPPORTED_CODES):
            l_code = k
            most_perc = pop_perc
    return l_code


def get_religion_pop(country):
    co = country
    # manual fixes
    if co == 'Eswatini':
        co = 'Swaziland'
    elif co == 'Myanmar':
        return 'Buddhism', 53800000
    elif co == 'Montenegro':
        return 'Christianity', 619211
    elif co == 'Palestine':
        return 'Islam', 4923000
    elif co == 'Democratic Republic of Congo':
        return 'Christianity', 95890000

    country_obj = country_cd.country_details(co)
    country_obj_name = country_obj.name()
    # if not (isinstance(country_obj_name, str) and country_obj_name in co):
    #     print(country_obj_name)
    #     import pdb; pdb.set_trace()
    religion, population = country_obj.religion(), country_obj.population()

    if co == 'Australia':
        population = 25690000
    elif co == 'Republic of China':
        religion = 'Buddhism'
    elif co == 'Gabon':
        population = 2341000
    elif co == 'Greece':
        population = 1064000
    elif co == 'Ivory Coast':
        religion = 'Islam'
    elif co == 'South Korea' or co == 'North Korea' or co == "People's Republic of China":
        religion = 'Atheism'
    elif co == 'Japan':
        religion = 'Shintoism'
    assert religion and isinstance(population, int)
    return religion, population


def manual_fix(country):
    if country == 'Laos':
        country = 'Lao'
    if country == 'Democratic Republic of Congo':
        country = 'Congo, the Democratic Republic of the'
    if country == 'Ivory Coast':
        country = "Côte d'Ivoire"
    return country


def get_countries_info(countries):
    info = defaultdict()
    for country in countries:
        country_orig = country
        country = manual_fix(country)
        country_obj = pycountry.countries.get(name=country)
        if not country_obj:
            try:
                country_obj = pycountry.countries.search_fuzzy(country)[0]
            except LookupError:
                print(f'country "{country}" not found')
                continue
        c_code = country_obj.alpha_2

        # some manual overrides for most spoken languages
        if c_code == 'HT' or c_code == 'BI':
            continue
        elif c_code == 'SS':
            l_code = 'en'
        elif c_code == 'PH':
            l_code = 'tl'
        elif c_code == 'ET':
            l_code = 'am'
        else:
            l_code = get_most_spoken(c_code)

        locale = Locale.parse(f'{l_code}_{c_code}')
        lang_name = locale.get_language_name('en')
        lang_name_native = locale.get_language_name()

        if country == 'Republic of China':
            l_code = 'zh-TW'
        if l_code == 'sr_Latn':
            l_code = 'sr'

        # this library is very slow since is loads an entire JSON
        # religion, population = '', 0
        religion, population = get_religion_pop(country_orig)

        info[country_orig] = {
            'code': l_code,
            'name': lang_name,
            'name_native': lang_name_native,
            'religion': religion,
            'population': population
        }
     # manual fixes
    info['Haiti'] = {
        'code': 'ht',
        'name': 'Haitian Creole',
        'name_native': 'Kreyòl'
    }
    info['Republic of Kosovo'] = {
        'code': 'sq',
        'name': 'Albanian',
        'name_native': 'Shqipja'
    }

    info = OrderedDict(sorted(info.items()))
    return info


def get_territory_population(response):
    soup = BeautifulSoup(response.text, 'html.parser')
    population_header = soup.find(lambda tag: tag.name == 'th' and \
                                  'population' in tag.get_text().lower())
    population = 0

    if population_header:
        # Find the next 'tr' that includes 'Total' following the Population header
        total_population_row = population_header.find_next('tr', class_='mergedrow')
        h1 = soup.find('h1')
        if h1 and 'KaNgwane' == h1.text:
            population_data = total_population_row.next_sibling.next_sibling.find('td')
        elif h1 and 'Guayana Esequiba' == h1.text:
            population_header = soup.find(lambda tag: tag.name == 'td' \
                                           and 'population' in tag.get_text(strip=True).lower())
            population_data = population_header.find_next('td')
        elif h1 and ('Hong Kong' == h1.text or 'Macau' == h1.text):
            population_header = soup.find(lambda tag: tag.name == 'th' and  'Population' in tag.get_text())
            population_data = population_header.find_next('td')
        elif total_population_row:
            # Find the 'td' that includes population data
            population_data = total_population_row.find('td')
        else:
            population_data = population_header.find_next('td', class_='infobox-data')

        population = population_data.get_text(strip=True) if population_data else '0'
        if not population.isdigit():
            pop_match = re.search(r'\d+(?:[\d,.]*\d)', population)
            population = pop_match[0].replace(',', '') if pop_match else 0
        population = int(population)

    # after this, we also applied some manual corrections in the CSV
    return population

if __name__ == "__main__":
    import requests
    response = requests.get('https://en.wikipedia.org/wiki/East_Jerusalem')
    print(get_territory_population(response))
