# /Users/Jack/Documents/Projects/Whatscooking-/src

import pandas as pd 
import nltk
import string
import ast
import re
import unidecode
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter
import config 

# Weigths and measures are words that will not add value to the model. I got these standard words from 
# https://en.wikibooks.org/wiki/Cookbook:Units_of_measurement

# # We lemmatize the words to reduce them to their smallest form (lemmas). 
# lemmatizer = WordNetLemmatizer()
# measures = [lemmatizer.lemmatize(m) for m in measures]
# words_to_remove = [lemmatizer.lemmatize(m) for m in words_to_remove]

nltk.download('wordnet')

def ingredient_parser(ingreds):

    measures = ['teaspoon', 't', 'tsp.', 'tsp','tablespoon', 'T', 'tbl.', 'tb', 'tbsp.', 'fluid ounce', 'fl oz', 'gill', 'cup', 'c', 'pint', 'p', 'pt', 
                'fl pt', 'quart', 'q', 'qt', 'fl qt', 'gallon', 'g', 'gal', 'ml', 'milliliter', 'millilitre', 'cc', 'mL', 'l', 'liter', 'litre',
                  'L', 'dl', 'deciliter', 'decilitre', 'dL', 'bulb', 'level', 'heaped', 'rounded', 'whole', 'pinch', 'medium', 'slice', 'pound',
                    'lb', '#', 'ounce', 'oz', 'mg', 'milligram', 'milligramme', 'g', 'gram', 'gramme', 'kg', 'kilogram', 'kilogramme', 'x', 'of',
                      'mm', 'millimetre', 'millimeter', 'cm', 'centimeter', 'centimetre', 'm', 'meter', 'metre', 'inch', 'in', 'milli', 'centi',
                        'deci', 'hecto', 'kilo','as','required','per','bc','bele','bhaat','Gram','Whole','grams']
    
    words_to_remove = ['fresh', 'oil', 'a', 'red', 'bunch'
                        'and', 'clove', 'or', 'leaf', 'chilli', 'large', 'extra', 'sprig', 'ground',
                       'handful', 'free', 'small', 'pepper', 'virgin', 'range', 'from', 'dried', 'sustainable', 'black', 'peeled', 'higher', 
                       'welfare', 'seed', 'for', 'finely', 'freshly', 'sea', 'quality', 'white', 'ripe', 'few', 'piece', 'source', 'to', 
                       'organic', 'flat', 'smoked', 'ginger', 'sliced', 'green', 'picked', 'the', 'stick', 'plain', 'plus', 'mixed', 'mint', 
                       'bay', 'basil', 'your', 'cumin', 'optional', 'fennel', 'serve', 'mustard', 'unsalted', 'baby', 'paprika', 'fat', 'ask',
                         'natural', 'skin', 'roughly', 'into', 'such', 'cut', 'good', 'brown', 'grated', 'trimmed', 'oregano', 'powder', 
                         'yellow', 'dusting', 'knob', 'frozen', 'on', 'deseeded', 'low', 'runny', 'balsamic', 'cooked', 'streaky', 'nutmeg', 
                         'sage', 'rasher', 'zest', 'pin', 'groundnut', 'breadcrumb', 'turmeric', 'halved', 'grating', 'stalk', 'light', 
                         'tinned', 'dry', 'soft', 'rocket', 'bone', 'colour', 'washed', 'skinless', 'leftover', 'splash', 'removed', 'dijon',
                           'thick', 'big', 'hot', 'drained', 'sized', 'chestnut', 'watercress', 'fishmonger', 'english', 'dill', 'caper', 
                           'raw', 'worcestershire', 'flake', 'cider', 'cayenne', 'tbsp', 'leg', 'pine', 'wild', 'if', 'fine', 'herb', 
                           'almond', 'shoulder', 'cube', 'dressing', 'with', 'chunk', 'spice', 'thumb', 'garam', 'new', 'little', 
                           'punnet', 'peppercorn', 'shelled', 'saffron', 'other''chopped', 'salt', 'olive', 'taste', 'can', 'sauce', 
                           'water', 'diced', 'package', 'italian', 'shredded', 'divided', 'parsley', 'vinegar', 'all', 'purpose', 'crushed', 
                           'juice', 'more', 'coriander', 'bell', 'needed', 'thinly', 'boneless', 'half', 'thyme', 'cubed', 'cinnamon', 
                           'cilantro', 'jar', 'seasoning', 'rosemary', 'extract', 'sweet', 'baking', 'beaten', 'heavy', 'seeded', 'tin', 
                           'vanilla', 'uncooked', 'crumb', 'style', 'thin', 'nut', 'coarsely', 'spring', 'chili', 'cornstarch', 'strip', 
                           'cardamom', 'rinsed', 'honey', 'cherry', 'root', 'quartered', 'head', 'softened', 'container', 'crumbled', 'frying',
                             'lean', 'cooking', 'roasted', 'warm', 'whipping', 'thawed', 'corn', 'pitted', 'sun', 'kosher', 'bite', 'toasted',
                               'lasagna', 'split', 'melted', 'degree', 'lengthwise', 'romano', 'packed', 'pod', 'anchovy', 'rom', 'prepared',
                                 'juiced', 'fluid', 'floret', 'room', 'active', 'seasoned', 'mix', 'deveined', 'lightly', 'anise', 'thai', 
                                 'size', 'unsweetened', 'torn', 'wedge', 'sour', 'basmati', 'marinara', 'dark', 'temperature', 'garnish', 
                                 'bouillon', 'loaf', 'shell', 'reggiano', 'canola', 'parmigiano', 'round', 'canned', 'ghee', 'crust', 'long', 
                                 'broken', 'ketchup', 'bulk', 'cleaned', 'condensed', 'sherry', 'provolone', 'cold', 'soda', 'cottage', 'spray',
                                   'tamarind', 'pecorino', 'shortening', 'part', 'bottle', 'sodium', 'cocoa', 'grain', 'french', 'roast', 'stem',
                                     'link', 'firm', 'asafoetida', 'mild', 'dash', 'boiling']
    # The ingredient list is now a string so we need to turn it back into a list. We use ast.literal_eval
    if isinstance(ingreds, list):
        ingredients = ingreds
    else:
        ingredients = ast.literal_eval(ingreds)

    translator = str.maketrans('', '', string.punctuation)
    lemmatizer = WordNetLemmatizer()
    ingred_list = []
    for i in ingredients:
        i.translate(translator)
        # We split up with hyphens as well as spaces
        items = re.split(' |-', i)
        # Get rid of words containing non alphabet letters
        items = [word for word in items if word.isalpha()]
        # Turn everything to lowercase
        items = [word.lower() for word in items]
        # remove accents
        items = [unidecode.unidecode(word) for word in items] #''.join((c for c in unicodedata.normalize('NFD', items) if unicodedata.category(c) != 'Mn'))
        # Lemmatize words so we can compare words to measuring words
        items = [lemmatizer.lemmatize(word) for word in items]
        # Gets rid of measuring words/phrases, e.g. heaped teaspoon
        items = [word for word in items if word not in measures]
        # Get rid of common easy words
        items = [word for word in items if word not in words_to_remove]
        if items:
            ingred_list.append(' '.join(items)) 
    ingred_list = " ".join(ingred_list)
    return ingred_list

if __name__ == "__main__":
    
    # Main Code
    # recipe_df = pd.read_csv(config.RECIPES_PATH)
    # recipe_df['ingredients_parsed'] = recipe_df['ingredients'].apply(lambda x: ingredient_parser(x))
    # df = recipe_df[['recipe_name', 'ingredients_parsed', 'ingredients', 'recipe_urls']]
    # df = recipe_df.dropna()

    # # remove - Allrecipes.com from end of every recipe title 
    # m = df.recipe_name.str.endswith('Recipe - Allrecipes.com')
    # df['recipe_name'].loc[m] = df.recipe_name.loc[m].str[:-23]        
    # df.to_csv(config.PARSED_PATH, index=False)


    recipe_df = pd.read_csv(config.INDIAN_RECIPES_PATH)
    recipe_df['Ingredients_parsed'] = recipe_df['Ingredients'].apply(lambda x: ingredient_parser(x))
    df = recipe_df[['RecipeName', 'Ingredients_parsed', 'Ingredients', 'Cuisine','Course','Diet']]
    df = recipe_df.dropna()

    m = df.RecipeName.str.endswith('Recipe - Allrecipes.com')
    df['RecipeName'].loc[m] = df.RecipeName.loc[m].str[:-23]   
    print('Parsed_CSV')     
    df.to_csv(config.INDIAN_RECIPES_PARSED_PATH, index=False)