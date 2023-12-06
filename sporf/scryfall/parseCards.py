import re
import numpy as np
import pandas as pd
import json


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

card_types = ['artifact', 'battle', 'creature', 'enchantment', 'instant', 'land', 'planeswalker', 'sorcery', 'tribal']

def loadCards(filename: str):
    with open(filename, 'r') as json_file:
        return json.load(json_file)

def parseCard(card: json):
    # mana_cost:{1}{R} => [cost, c, w, b, u, r, g, x]
    # cmc:2.0 => 2.0
    # type = [artifact, battle, creature, enchantment, instant, land, planeswalker, sorcery, tribal]
    # keywords:["haste"] => key_words
    # ability_words:["landfall"]
    # reserved:true => reserved
    # rarity:common => [common:0,uncommon:0,rare:0,mythic:0]
    # power:0 => 0
    # toughness:0 => 0
    # oracle_text:string => mds on the text to map to high dimensional latent space, replace card name with ~
    
    # sort out extra cards that we don't need to worry about (promotional, non-playable, videogame only, etc)
    if card["legalities"]["vintage"] == "not_legal": return None
    if "card_faces" in card: return None

    carddict = {} # holds all values that will go into the dataframe

    mana_cost: list[str] = re.findall(r'{(.*?)}', card["mana_cost"])
    for symbol in ['C', 'W', 'B', 'U', 'R', 'G', 'X']:
        count = sum([1 for s in mana_cost if symbol in s])
        if count == 0:
            carddict[symbol] = 0
            continue
        carddict[symbol] = count
        mana_cost = [i for i in mana_cost if i != symbol]
    if len(mana_cost) == 0: carddict['cost'] = 0
    else: carddict['cost'] = mana_cost[0]

    type_line: str = card["type_line"]
    for card_type in card_types:
        if card_type in type_line:
            carddict[card_type] = 1
        else:
            carddict[card_type] = 0
    
    if carddict["creature"]:
        carddict["toughness"] = int(card["power"])
        carddict["power"] = int(card["power"])
    
    oracle:str = card["oracle_text"]
    oracle = oracle.replace(card["name"], "~")
    carddict["oracle"] = oracle
    carddict["id"] = card["id"]

    np.array([])
    return carddict

def parseCards(filename: str):
    cards = loadCards(filename)
    card = parseCard(cards[0])
    df = pd.DataFrame(columns=card.keys())
    for card in cards:
        card = parseCard(card)
        if card == None:
            continue
        df.loc[len(df)] = card
    df.to_csv('cards.csv', index=False)

# parseCards('oracle-cards-20231203220253.json')