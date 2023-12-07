import requests
import json
from time import sleep
from parseCards import ability_words, key_words, card_types, keyword_abilities

def generateTags():
    otags = []

    with open("rawotags.txt", 'r', encoding='utf-8') as read_file:
        skip = True
        for line in read_file:
            line = line.strip()
            if not skip:
                print(line)
                tags = line.split(' · ')
                otags.extend(tags)
                skip = True
                continue
            if line.endswith('(functional)'):
                skip = False
    writeTags("otags.txt", otags)

def writeTags(filename: str, otags: list[str]):
    with open(filename, 'w', encoding='utf-8') as output_file:
        for tag in otags:
            output_file.write(tag + '\n')

def readTags(filename: str):
    with open(filename, 'r', encoding='utf-8') as read_file:
        return [otag.strip() for otag in read_file]

def filterTags():
    otags = readTags("otags.txt")
    
    def validate(otag:str):
        if otag.startswith('cycle'):
            return False
        
        return True

    filtered_tags = [otag for otag in otags if validate(otag)]

    writeTags("otags.txt", filtered_tags)

def searchToUrl(search_query: str)->str:
    base_url = "https://api.scryfall.com/cards/"
    return f"{base_url}search?q={search_query}"


def scryfallGet(url: str):
    print(f'Scrifall getting url:{url:20}')
    timeOut = 0.1
    while timeOut < 20:
        try:
            response = requests.get(url)
            data = response.json()
            if response.status_code == 200:
                return data
            else:
                details:str = data["details"]
                if (details.startswith("Your query didn’t match any cards")):
                    return {"has_more":False,"data":[]}
                raise Exception(f'Request Failed: {response}')

        except Exception as e:
            print(f"An error occurred: {e}")
            sleep(timeOut)
            timeOut *= 2

def stripIds(cards: list[any])->set[str]:
    return {card["id"] for card in cards}

def scryfallGetEntireTag(otag: str):
    url = searchToUrl(f'otag={otag}')
    response = scryfallGet(url)
    cards = stripIds(response['data'])
    while True:
        if response['has_more'] == False: break
        url = response['next_page']
        response = scryfallGet(url)
        sleep(0.1)
        cards = cards.union(stripIds(response['data']))

    return cards

def writeTagMap(filename: str, jsonObj: any):
    with open(filename, 'w') as json_file:
        json.dump(jsonObj, json_file, indent=2)

def loadTagMap(filename: str)->dict[str, list[str]]:
    with open(filename, 'r') as json_file:
        return json.load(json_file)

def scryfallGetTagMap():
    tagMap: dict[str, set[str]] = {}
    otags = readTags("otags.txt")
    for otag in otags:
        new_cards = scryfallGetEntireTag(otag)
        print(f'getting otag:"{otag}"')
        for id in new_cards:
            if id not in tagMap:
                tagMap[id] = set()
            tagMap[id].add(otag)
    for id, tags in tagMap.items():
        tagMap[id] = list(tags)
    
    writeTagMap('tagmap.json',tagMap)

def isAbilityWord(tag: str):
    for ability in ability_words:
        if tag.startswith(ability.lower()) and "-" in tag:
            return True
    for keyword in key_words:
        if tag.startswith(keyword.lower()) and "-" in tag:
            return True
    for keyword in keyword_abilities:
        if keyword.lower() == "affinity" and tag.startswith("affinity"):
            print(tag.startswith(keyword.lower()) and "-" in tag)

        if tag.startswith(keyword.lower()) and "-" in tag:
            return True
    return False

def filterTags():
    tagMap: dict[str, list[str]] = loadTagMap("tagmap.json")
    otags = set()
    for (card_id, tags) in tagMap.items():
        otags.update(tags)
    filtered = set()
    for tag in otags:
        if not isAbilityWord(tag):
            filtered.add(tag)
    sorted_list = sorted(list(filtered))
    writeTags("filter1tags.txt", sorted_list)
    print(len(filtered))
filterTags()