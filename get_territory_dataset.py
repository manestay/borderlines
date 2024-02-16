import argparse
import json
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from numpy import random

from demographics import get_countries_info, get_territory_population
from lib import INFO_PATH, SITES, TERR_PATH

random.seed(2557)

STRS_TO_CLEAN = ['Disputed_status_of_', '_border_dispute', '_dispute', 'Geography_of_']

parser = argparse.ArgumentParser()
parser.add_argument('--site_url', default=SITES['2023-05-15'])
parser.add_argument('--out_path', '-o', default=TERR_PATH)
parser.add_argument('--info_path', '-ip', default=INFO_PATH)
parser.add_argument('--get_population', action='store_true')


def clean_territories(territories):
    terrs_new = []
    for terr in territories:
        if terr.get('title') and 'page does not exist' in terr.get('title'):
            continue
        if terr.text.startswith('['):
            continue
        if len(terr.text) == 2:
            continue
        terrs_new.append(terr)
    return terrs_new

def clean_link(link, to_clean=STRS_TO_CLEAN):
    for s in to_clean:
        link = link.replace(s, '', 1)
    return link


def get_df_region(elem):
    def get_text(tag):
        return tag.text.split('[')[0]

    h2_text = get_text(elem.find_previous_sibling('h2'))
    if 'outside' in h2_text or 'waters' in h2_text:
        return h2_text

    tag = elem.find_previous_sibling('h3')
    text = get_text(tag)

    if text == 'Americas':
        text = get_text(elem.find_previous_sibling('h4'))

    return text


if __name__ == "__main__":
    args = parser.parse_args()

    site_url = args.site_url
    if 'wikipedia.org' not in site_url:
        if site_url not in SITES:
            print('please pass in a valid URL')
            exit()
        # otherwise load from SITES dict
        site_url = SITES[site_url]

    print(f'loading URL {site_url}')
    response = requests.get(site_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    tables_raw = soup.find_all('table')

    print('processing Wikipedia page...')
    dfs = {}
    records = {}
    for table in tables_raw:
        header = [x.text.strip() for x in table.find_all('th')]
        if 'Claimants' not in header:
            continue

        region = get_df_region(table)
        if region == 'Ongoing disputes involving states outside the UN':
            region = 'Outside UN'
        if region == 'Antarctica' or region == 'Disputes over territorial waters':
            continue

        for row in table.find_all('tr')[1:]:
            cells = row.find_all('td')

            territories = cells[0].find_all('a')  # all territories
            links = [x.get('href') for x in territories]
            territories = [x.text for x in territories]

            if len(cells) == 3:
                claimants = cells[1].find_all('a')
                # if the controlling country is bolded
                if len(cells[1].find_all('b')) == 1 and cells[1].b.text:
                    cba = cells[1].b.a
                    if cba and cba.text:
                        controller = cba.text.strip()
                    elif cells[1].b:
                        controller = cells[1].b.text.strip()
                else:  # skip if region not controlled by 1 country
                    controller = 'Unknown'

                claimants = [x.text for x in claimants]
            else:  # diff formatting in 'Europe' table
                claimants, controller, region = prev[1:4]

            for terr, link in zip(territories, links):
                if not terr[0].isalpha():  # ensure not citation
                    continue
                if terr.islower() and (len(terr) == 2 or len(terr) == 3):
                    continue
                if terr in claimants:
                    continue
                if terr in set(['peninsula', 'municipality', 'Tibet', 'Croatia']):
                    continue

                if terr in records: # handle duplicate territories
                    claimants = list(set(records[terr][1] + claimants))
                    if controller != records[terr][2]:
                        controller = 'Unknown'

                print(f'Processing {terr}' + ' ' * 30, end='\r')

                link = urljoin(site_url, link)
                response_territory = requests.get(link)

                if args.get_population:
                    population = get_territory_population(response_territory)
                    if population == 0:
                        link2 = clean_link(link)
                        if link2 != link:
                            response_territory = requests.get(link2)
                            population = get_territory_population(response_territory)
                else:
                    population = 'N/A'

                records[terr] = [terr, claimants, controller, region, population]

                prev = records[terr]

    columns = ['Territory', 'Claimants', 'Controller', 'Region', 'Population']
    df = pd.DataFrame(data=records.values(), columns=columns)

    ###
    df['Claimants'] = df['Claimants'].apply(
        lambda l: [x for x in l if x and x[0].isalpha()])


    # now call APIs to get most spoken lang per country
    countries = set([country for subl in df['Claimants'] for country in subl])
    countries_info = get_countries_info(countries)

    with open(args.info_path, 'w') as f:
        json.dump(countries_info, f, indent=2, ensure_ascii=False)
    print(f'saved to {args.info_path}')

    df['Claimants'] = df['Claimants'].apply(
        lambda l: [x for x in l if x in countries_info])
    ###

    df = df[(df['Claimants'].apply(len) >= 2)]  # multiple claimants only
    df['Claimants'] = df['Claimants'].apply(random.permutation)
    df['Claimants'] = df['Claimants'].str.join(';')
    df.to_csv(args.out_path, index=False)
    print(f'saved {len(df)} rows to {args.out_path}')
