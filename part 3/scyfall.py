import requests
import json
import numpy as np
import re
from time import sleep
ability_words = ["Battalion", "Bloodrush", "Channel", "Chroma", "Cohort", "Constellation", "Converge",
                 "Delirium", "Domain", "Fateful hour", "Ferocious", "Formidable", "Grandeur", "Hellbent",
                 "Heroic", "Imprint", "Inspired", "Join forces", "Kinship", "Landfall", "Lieutenant", "Metalcraft",
                 "Morbid", "Parley", "Radiance", "Raid", "Rally", "Spell mastery", "Strive", "Sweep", "Tempting offer",
                 "Threshold", "Will of the council", "Adamant", "Addendum", "Council's dilemma", "Eminence", "Enrage",
                 "Hero's Reward", "Kinfall", "Landship", "Legacy", "Revolt", "Underdog", "Undergrowth", "Descend",
                 "Fathomless descent", "Magecraft", "Teamwork", "Pack tactics", "Coven", "Alliance", "Corrupted",
                 "Secret council", "Celebration", "Paradox"]

key_words = ["Seek", "Activate", "Attach", "Cast", "Counter", "Create", "Destroy", "Discard", "Double", "Exchange",
             "Exile", "Adapt", "Support", "Play", "Regenerate", "Reveal", "Sacrifice", "Shuffle", "Tap", "Untap",
             "Vote", "Goad", "Transform", "Surveil", "Planeswalk", "Mill", "Learn", "Connive", "Venture into the dungeon",
             "Exert", "Open an Attraction", "Food", "Discover", "Conjure", "Abandon", "Explore", "Amass", "Treasure",
             "Roll to Visit Your Attractions", "Set in motion", "Fateseal", "Manifest", "Populate", "Detain",
             "Investigate", "Monstrosity", "Clash", "Scry", "Incubate", "Proliferate", "Meld", "Convert", "Fight",
             "Bolster", "Assemble", "Role token"]

def parseCard(card: json):
    # mana_cost:{1}{R} => [cost, c, w, b, u, r, g, x]
    # cmc:2.0 => [artifact, battle, creature, enchantment, instant, land, planeswalker, sorcery, tribal]
    # keywords:["haste"] => key_words
    # ability_words:["landfall"]
    # reserved:true => reserved
    # rarity:common => [common:0,uncommon:0,rare:0,mythic:0]
    # power:1+* => [value:0, star:0]
    # 
    if card["legalities"]["vintage"] == "not_legal": return None
    carddict = {}
    mana_cost: list[str] = re.findall(r'{(.*?)}', card["mana_cost"])
    print(mana_cost)
    for symbol in ['C', 'W', 'B', 'U', 'R', 'G', 'X']:
        count = mana_cost.count(symbol)
        if count == 0: continue
        carddict[symbol] = count
        mana_cost = [i for i in mana_cost if i != symbol]
    print(mana_cost)
        

    
    np.array([])
    return carddict

def searchToUrl(search_query: str)->str:
    base_url = "https://api.scryfall.com/cards/"
    return f"{base_url}search?q={search_query}"


def scryfallGet(url: str):
    while True:
        try:
            response = requests.get(url)

            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f'Request Failed: {response}')

        except Exception as e:
            sleep(0.1)
            print(f"An error occurred: {e}")

def scryfallGetAll(search_query: str):
    url = searchToUrl(search_query)
    
    response = scryfallGet(url)
    cards = response['data']
    while True:
        if response['has_more'] == False: break
        url = response['next_page']
        response = scryfallGet(url)
        sleep(0.1)
        cards += response['data']
    return cards

search_query = "set:lci"
cards = scryfallGet(searchToUrl("liam"))["data"]
print(parseCard(cards[0]))