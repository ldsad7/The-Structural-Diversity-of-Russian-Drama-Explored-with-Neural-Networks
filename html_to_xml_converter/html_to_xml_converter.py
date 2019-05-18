# importing necessary libraries
import argparse
from bs4 import BeautifulSoup, NavigableString
from lxml import etree
from copy import deepcopy
import lxml.builder as lb
import requests
import re
import html as html_
import os
from transliterate import translit, get_available_language_codes
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import datetime
from googletrans import Translator
import json
import string
from titlecase import titlecase
import pandas as pd
from pymystem3 import Mystem
import warnings
warnings.filterwarnings('ignore')

common_abbreviations = {'Джордж': 'Дж', 'Джеймс': 'Дж'} # non-standard abbreviations

div_names = ('действие', 'действующие', 'явление', 'сцена', 'картина', 'i')

lower_alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
upper_alphabet = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
from_list = lower_alphabet + upper_alphabet
to_list = 'a,b,v,g,d,e,jo,zh,z,i,y,k,l,m,n,o,p,r,s,t,u,f,kh,c,ch,sh,shh,jhh,ih,jh,eh,ju,ja,A,B,V,G,D,E,JO,ZH,Z,I,Y,K,L,M,N,O,P,R,S,T,U,F,KH,C,CH,SH,SHH,JHH,IH,JH,EH,JU,JA'.split(',')
dashes = '—–−-'

m = Mystem() # to define the sex of a character
translator = Translator() # to translate title if no translation on wikidata.org

exclude = set(string.punctuation) # punctuation marks

def download_wikidata_ids():
    # download wikidata-ids of the authors on the page https://ru.wikipedia.org/wiki/Категория:Драматурги_России
    try: # if file already exists
        with open('titles_to_ids.json', 'r') as f:
            titles_to_ids = json.load(f)
    except FileNotFoundError:
        url = "https://ru.wikipedia.org/w/api.php?action=query&format=json&list=categorymembers&cmtitle=Категория:Драматурги_России&from=Р&cmlimit=max&prop=pageprops&ppprop=wikibase_item"
        jsn1 = requests.get(url).text
        jsn1 = json.loads(jsn1)
        cmcontinue = jsn1['continue']['cmcontinue']
        url += "&cmcontinue=" + cmcontinue
        jsn2 = requests.get(url).text
        jsn2 = json.loads(jsn2)

        titles_to_ids = {}

        jsn = jsn1['query']['categorymembers'] + jsn2['query']['categorymembers']
        for elem in jsn:
            titles_to_ids[elem['title']] = elem['pageid']
        for key, value in titles_to_ids.items():
            url = f"https://ru.wikipedia.org/w/api.php?action=query&format=json&prop=pageprops&ppprop=wikibase_item&pageids={value}"
            jsn = requests.get(url).text
            jsn = json.loads(jsn)
            try:
                titles_to_ids[key] = jsn['query']['pages'][str(value)]['pageprops']['wikibase_item']
            except KeyError:
                continue
        with open('titles_to_ids.json', 'w') as f:
            json.dump(titles_to_ids, f)
    return titles_to_ids

titles_to_ids = download_wikidata_ids()

class HTML():
    def __init__(self, comment='', footnotes='', author='', name='', title='', publication='', \
                 genre='', edition='', mean_length=0, std=0, popular_indent=0, lib_year=0, lines=[]):
        self.comment = comment.strip()
        self.footnotes = footnotes.strip()
        self.author = author.strip()
        self.name = name.strip()
        self.title = title.strip()
        self.publication = publication.strip()
        self.genre = genre.strip()
        self.edition = edition.strip()
        self.mean_length = mean_length
        self.std = std
        self.popular_indent = popular_indent
        self.lib_year = lib_year
        self.lines = lines
        self.play_num = 156 # to be defind somehow (?)

        # prose or poetry
        if self.mean_length > 40 or self.std > self.mean_length:
            # prose
            self.porl = 'p'
        else:
            # poetry
            self.porl = 'l'

    def print(self):
        print("comment:", self.comment)
        print("footnotes:", self.footnotes)
        print("author:", self.author)
        print("name:", self.name)
        print("title:", self.title)
        print("publication:", self.publication)
        print("genre:", self.genre)
        print("edition:", self.edition)
        print("mean_length:", self.mean_length)
        print("std:", self.std)
        print("popular_indent:", self.popular_indent)
        print("lib_year:", self.lib_year)

def preprocess_text(text):
    comment = ''
    footnotes = ''
    author = '' # full name
    name = '' # abbreviated name
    title = ''
    publication = '' # publication information
    genre = ''
    edition = ''
    mean_length = 0 # of lines in a play
    std = 0 # standard deviation of lenghts of lines in a play
    popular_indent = 0 # the most popular indentation (in spaces) of lines in a play
    lib_year = '' # writing year according to lib.ru

    #etree.tostring()
    soup = BeautifulSoup(text, 'lxml')

    """
    # li
    for li in soup.findAll('li'):
        # print('~', li.text, '~')
        strip_low_line = li.text.strip().lower()
        year = re.findall('год:\s*([0-9]{4})$', strip_low_line)
        if year:
            lib_year = year[0]
        if re.match('пьеса:\s*', strip_low_line):
            if 'драма' in strip_low_line:
                genre = 'драма'
            elif 'комеди' in strip_low_line:
                genre = 'комедия'
            elif 'трагеди' in strip_low_line:
                genre = 'трагедия'
        if re.findall('аннотация:\s*(.*)', strip_low_line):
            comment += re.findall('.*?:\s*(.*)', li.text)[0].strip() + ' '
    """

    # remove unnecessary tags with their content (to be revised)
    for tag in 'script form a ul'.split(): # delete or add some tags if necessary
        [s.extract() for s in soup(tag)]
        """
        for s in soup.findAll(tag):
            if not s.text.lower().strip().startswith(tuple(div_names)):
                s.decompose()
        """

    soup_text = soup.text

    # remove start of the line with the name and the title
    soup_text = re.sub('Lib.ru/Классика: ', '', soup_text)

    was_comment = 0
    was_first_line = 0

    lines = []
    for line_index, line in enumerate(soup_text.split('\n')):
        strip_low_line = line.strip().lower()
        if was_first_line == 0 and strip_low_line:
            line = line.split('.') # the line that starts with "Lib.ru/Классика: "
            author = line[0].split()
            if author[1] in common_abbreviations: # name
                name_abbr = common_abbreviations[author[1]]
            else:
                name_abbr = author[1][0] # first letter
            name = name_abbr + '. ' \
                             + author[2][0] \
                             + '. ' \
                             + author[0]
            author[0] += ','
            author = ' '.join(author).strip() # full name
            title = line[1].strip()
            was_first_line = 1
            continue
        if line.startswith('[]') or strip_low_line == 'скачать' or \
            (title and re.match(title.lower() + "(?:$|\s)", strip_low_line.strip('"'))):
            # in order not to capture "Козьма Захарьич Минин, Сухорук, земский староста Нижнего посада."
            # skip lines
            if title and strip_low_line.strip('"').startswith(title.lower()):
                # here can be a year specified
                tmp_lib_year = re.findall("[^0-9]([0-9]{4})[^0-9]", line)
                if tmp_lib_year:
                    lib_year = tmp_lib_year[0]
            continue
        if re.match('\[[0-9]{1,3}\]', strip_low_line): # => footnote
            footnotes += line
            continue
        if strip_low_line == 'комментарии' or  strip_low_line.startswith('======='):
            was_comment = 1
        elif was_comment == 1:
            comment += line # add comment
        if strip_low_line and not was_comment:
            lines.append(line.rstrip())

    # print('\n'.join(lines[:100]))

    was_author = 0
    indices_to_be_removed = []
    for line_index, line in enumerate(lines):
        strip_low_line = line.strip().lower()
        if re.match(re.escape(name.lower()) + "(ъ|$)", strip_low_line): # old orthography
            was_author = 1
            indices_to_be_removed.append(line_index)
            continue
        """
        elif strip_low_line.startswith(div_names):
            was_author = 1
        """
        if not was_author:
            publication += line.strip()
            indices_to_be_removed.append(line_index)
    lines = [line for index_line, line in enumerate(lines) if index_line not in indices_to_be_removed]

    if not lines[0].strip().lower().startswith(div_names):
        genre = lines[0].strip()
        del lines[0]

    was_edition = 0
    if not lines[0].strip().lower().startswith(div_names):
        edition = lines[0].strip()
        was_edition = 1
        del lines[0]

    lengths = np.array([])
    spaces = np.array([])
    for line in lines:
        lengths = np.append(lengths, len(line))
        spaces = np.append(spaces, len(re.findall(r"^\s*", line)[0]))

    popular_indent = int(Counter(spaces).most_common(1)[0][0])
    mean_length = np.mean(lengths)
    std = np.std(lengths)

    # fix possible errors
    if not publication and edition:
        publication = edition
        edition = ''

    html = HTML(comment, footnotes, author, name, title, publication, genre, \
                edition, mean_length, std, popular_indent, lib_year, lines)
    return html

"""
def transliterate(string):
    transliterated_string = ''
    for symbol in string:
        if symbol in from_list:
            index = from_list.index(symbol)
            transliterated_string += to_list[index]
        else:
            transliterated_string += symbol
    return transliterated_string

def detransliterate(string):
    detransliterated_string = ''
    index_of_symbol = 0
    while index_of_symbol < len(string):
        was_combination = 0
        for i in range(3, 0, -1):
            if string[index_of_symbol:index_of_symbol+i] in to_list:
                index = to_list.index(string[index_of_symbol:index_of_symbol+i])
                detransliterated_string += from_list[index]
                index_of_symbol += i
                was_combination = 1
                break
        if not was_combination:
            detransliterated_string += string[index_of_symbol]
            index_of_symbol += 1
    return detransliterated_string
"""

# function that thansliterates in a standard way (as in transliterate.translit)
# translit("Островский", "ru", reversed=True)
eng_trans_small = 'a b v g d e e zh z i j k l m n o p r s t u f h ts ch sh sch ~ y ~ e ju ja'.split()
eng_trans_big = 'A B V G D E E Zh Z I J K L M N O P R S T U F H Ts Ch Sh Sch ~ Y ~ E Ju Ja'.split()
def transliterate(word): # _standard
    translit = ''
    for letter in word:
        if letter in lower_alphabet:
            index = lower_alphabet.index(letter)
            translit += eng_trans_small[index]
        elif letter in upper_alphabet:
            index = upper_alphabet.index(letter)
            translit += eng_trans_big[index]
        else:
            translit += letter
    return translit

# correcting typos due to the automatic recognition of a text
def correct_mistakes(html):
    text = '\n'.join(html.lines)
    # 'о' is frequently recognized as 'C' (in all types of contexts)
    text = re.sub('([а-яё])С', r'\1о', text)
    text = re.sub('([а-яёА-ЯЁ])С([а-яё])', r'\1о\2', text)
    text = re.sub('([а-яёА-ЯЁ])С( [а-яё])', r'\1о\2', text)
    # 'б' is sometimes recognized as '6'
    text = re.sub('([а-яё])6([а-яё])', r'\1б\2', text)

    lines = text.split('\n')

    new_lines = []
    for line_index, line in enumerate(lines):
        if not line.strip(): # empty line
            continue

        # how often is register changed in a line in order to recognize <l>-lines
        # that stick together but not <p>-lines
        register_change = []

        words = line.split()
        for i in range(len(words) - 1):
            if words[i][0] in lower_alphabet and words[i+1][0] in upper_alphabet: # and words[i][-1] not in '.!?'
                register_change.append(i + 1) # index of the presumably next <l>-line

        # some heuristic
        if len(register_change) > 8 and len(line) < 250: # too long line is more probably a <p>-line
            start_index = 0 # start of a sentence
            for i in register_change:
                grs = m.analyze(words[i])[0]['analysis'][0]['gr'] # it takes time because requires mystem call
                if 'фам' in grs or 'имя' in grs or 'гео' in grs: # Probably it is not the start of sent.
                    continue
                new_lines.append(' ' * html.popular_indent + ' '.join(words[start_index:i]) + '\n')
                start_index = i
        else:
            new_lines.append(line)
    html.lines = new_lines
    return html

def get_root():
    # root
    root = etree.Element('TEI', xmlns="http://www.tei-c.org/ns/1.0") # xmlns=https://dracor.org/rus/schema.rng
    root.set("{http://www.w3.org/XML/1998/namespace}lang", "ru")
    root.addprevious(etree.PI('xml-stylesheet', 'type="text/css" href="../css/tei.css"'))
    return root

def add_teiHeader_to_root(root):
    # teiHeader to root
    teiHeader = etree.Element('teiHeader')
    root.append(teiHeader)
    return teiHeader

def add_text_to_root(root):
    # text to root
    text = etree.Element('text')
    root.append(text)
    return text

def add_front_to_text(text):
    # front to text
    front = etree.Element('front')
    text.append(front)
    return front

def add_body_to_text(text):
    # body to text
    body = etree.Element('body')
    text.append(body)
    return body

def add_fileDesk_to_teiHeader(teiHeader):
    # fileDesc to teiHeader
    fileDesc = etree.Element('fileDesc')
    teiHeader.append(fileDesc)
    return fileDesc

def add_profileDesk_to_teiHeader(teiHeader):
    # profileDesc to teiHeader
    profileDesc = etree.Element('profileDesc')
    teiHeader.append(profileDesc)
    return profileDesc

def add_partiDesc_to_profileDesc(profileDesc):
    # partiDesc to profileDesc
    partiDesc = etree.Element('particDesc')
    profileDesc.append(partiDesc)
    return partiDesc

def add_listPerson_to_partiDesc(partiDesc):
    # listPerson to partiDesc
    listPerson = etree.Element('listPerson')
    partiDesc.append(listPerson)
    return listPerson

def add_textClass_to_profileDesc(profileDesc):
    # textClass to profileDesc
    textClass = etree.Element('textClass')
    profileDesc.append(textClass)
    return textClass

def add_keywords_to_textClass(textClass):
    # keywords to textClass
    keywords = etree.Element('keywords')
    textClass.append(keywords)
    return keywords

def add_revisionDesc_to_teiHeader(teiHeader):
    # revisionDesc to teiHeader
    revisionDesc = etree.Element('revisionDesc')
    teiHeader.append(revisionDesc)
    return revisionDesc

def add_listChange_to_revisionDesc(revisionDesc):
    # listChange to revisionDesc
    listChange = etree.Element('listChange')
    revisionDesc.append(listChange)
    return listChange

def add_change_to_listChange(listChange):
    # change to listChange
    change = etree.Element('change', when=datetime.datetime.now().strftime("%Y-%m-%d"))
    change.text = "(%s) convert from source" % ('eg') # replace if necessary
    listChange.append(change)
    return change

def add_docTitle_to_front(front):
    # docTitle to front
    docTitle = etree.Element('docTitle')
    front.append(docTitle)
    return docTitle

def add_titlePart_main_to_docTitle(docTitle, html):
    # titlePart to docTitle
    titlePart = etree.Element('titlePart', type='main')
    titlePart.text = html.title
    docTitle.append(titlePart)
    return titlePart

def add_titlePart_sub_to_docTitle(docTitle, html):
    # titlePart to docTitle
    titlePart = etree.Element('titlePart', type='sub')
    titlePart.text = html.genre.capitalize()
    docTitle.append(titlePart)
    return titlePart

def add_titleStmt_to_fileDesc(fileDesc):
    # titleStmt to fileDesc
    titleStmt = etree.Element('titleStmt')
    fileDesc.append(titleStmt)
    return titleStmt

def add_publicationStmt_to_fileDesc(fileDesc):
    # publicationStmt to fileDesc
    publicationStmt = etree.Element('publicationStmt')
    fileDesc.append(publicationStmt)
    return publicationStmt

def add_sourceDesc_to_fileDesc(fileDesc):
    # sourceDesc to fileDesc
    sourceDesc = etree.Element('sourceDesc')
    fileDesc.append(sourceDesc)
    return sourceDesc

def add_ru_title_to_titleStmt(titleStmt, html):
    # title to titleStmt
    title = etree.Element('title', type="main")
    title.set("{http://www.w3.org/XML/1998/namespace}lang", "ru")
    title.text = html.title
    titleStmt.append(title)
    return title

def find_wikidata_id_of_play(html):
    # wikidata-id of the play
    wikidata_search_url = f"https://www.wikidata.org/w/index.php?search={'+'.join(html.title.split())}"
    html_text = requests.get(wikidata_search_url).text
    qs = re.findall('<span class=\"wb-itemlink-id\">\((.*?)\)<\/span>.*?play (?:written )?by', html_text, re.DOTALL)
    if qs:
        return qs[0]
    return ''

def add_en_title_to_titleStmt(titleStmt, html):
    # title to titleStmt
    title = etree.Element('title', type="main")
    title.set("{http://www.w3.org/XML/1998/namespace}lang", "en")
    wikidata_id_of_play = find_wikidata_id_of_play(html)
    if wikidata_id_of_play:
        wikidata_url = f"https://www.wikidata.org/wiki/{wikidata_id_of_play}"
        html_text = requests.get(wikidata_url).text
        qs = re.findall('<span class=\"wikibase-labelview-text\">(.*?)</span>', html_text, re.DOTALL)
        if qs and qs[0] != 'No label defined':
            en_translation = qs[0]
        else:
            # translate to English
            en_translation = translator.translate(html.title, src='ru', dest='en').text
        title.text = titlecase(en_translation).strip() # capitalize non-functional words
    else:
        # translate to English
        en_translation = translator.translate(html.title, src='ru', dest='en').text
        title.text = titlecase(en_translation).strip() # capitalize non-functional words
    titleStmt.append(title)
    return title, wikidata_id_of_play

def add_ru_sub_title_to_titleStmt(titleStmt, html):
    # title to titleStmt
    title = etree.Element('title', type="sub")
    title.set("{http://www.w3.org/XML/1998/namespace}lang", "ru")
    title.text = html.genre.capitalize()
    titleStmt.append(title)
    return title

def add_en_sub_title_to_titleStmt(titleStmt, html):
    # title to titleStmt
    title = etree.Element('title', type="sub")
    title.set("{http://www.w3.org/XML/1998/namespace}lang", "en")
    title.text = titlecase(translator.translate(html.genre, src='ru', dest='en').text)
    titleStmt.append(title)
    return title

def add_term_to_keywords(keywords, html):
    # term to keywords (to be revised)
    genre_text = html.genre.lower()
    if 'драм' in genre_text:
        subtype = 'drama'
    elif 'комеди' in genre_text:
        subtype = 'comedy'
    elif 'трагеди' in genre_text:
        subtype = 'tragedy'
    else:
        subtype = '???'
    term = etree.Element('term', type="genreTitle", subtype=subtype)
    term.text = html.genre.strip().capitalize()
    keywords.append(term)
    return term

def find_wikidata_id_of_author(html):
    # wikidata_id_of_author (to be revised)
    if html.author in titles_to_ids: # in out databse
        return titles_to_ids[html.author]
    else:
        url_for_request = f"https://www.wikidata.org/w/index.php?search={'+'.join(html.author.split())}"
        html_text = requests.get(url_for_request).text
        qs = re.findall('<span class=\"wb-itemlink-id\">\((.*?)\)<\/span>.*?Russian', html_text, re.DOTALL)
        if qs: # wikidata
            return qs[0]
        return "Q??????"

def add_author_to_titleStmt(titleStmt, html):
    # author to titleStmt
    wikidata_id_of_author = find_wikidata_id_of_author(html)
    key = "wikidata:" + wikidata_id_of_author
    author = etree.Element('author', key=key)
    author.text = html.author
    titleStmt.append(author)
    return author

def add_publisher_to_publicationStmt(publicationStmt):
    # publisher to publicationStmt
    publisher = etree.Element('publisher')
    publisher.set("{http://www.w3.org/XML/1998/namespace}id", "RusDraCor")
    publisher.text = "RusDraCor"
    publicationStmt.append(publisher)
    return publisher

def add_name_idno_to_publicationStmt(publicationStmt, html):
    # name idno to publicationStmt
    # name for publicationStmt
    name_trans = html.name.split()[2].lower()
    if name_trans.endswith('ий'): # Ostrovsky's case
        name_trans = transliterate(name_trans[:-2]) + 'y'
    else:
        name_trans = transliterate(name_trans)
    title_trans = html.title.lower().split()
    for i, elem in enumerate(title_trans):
        title_trans[i] = ''.join(ch for ch in title_trans[i] if ch not in exclude)
    title_trans = transliterate('-'.join(title_trans)).strip()
    name = '<idno type="URL">https://dracor.org/rus/' + name_trans + '-' + title_trans + '</idno>'
    idno = etree.fromstring(name)
    publicationStmt.append(idno)

def add_num_of_play_idno_to_publicationStmt(publicationStmt, html):
    # idno to publicationStmt
    idno = etree.fromstring(f'<idno type="dracor" xml:base="https://dracor.org/id/">rus{html.play_num:06d}</idno>'.format(html.play_num))
    publicationStmt.append(idno)
    return idno

def add_wikidata_idno_to_publicationStmt(publicationStmt, html, wikidata_id_of_play):
    # idno to publicationStmt
    idno = etree.fromstring(f'<idno type="wikidata" xml:base="https://www.wikidata.org/entity/">{wikidata_id_of_play}</idno>')
    publicationStmt.append(idno)
    return idno, wikidata_id_of_play

def add_availability_to_publicationStmt(publicationStmt):
    # availability to publicationStmt
    availability = etree.fromstring('<availability><licence><ab>CC0</ab><ref target="https://creativecommons.org/publicdomain/zero/1.0/">Licence</ref></licence></availability>')
    publicationStmt.append(availability)
    return availability

def add_bibl_to_sourceDesc(sourceDesc):
    # bibl to sourceDesc
    bibl = etree.Element('bibl', type="digitalSource")
    sourceDesc.append(bibl)
    return bibl

def add_name_to_bibl(bibl):
    # name to bibl
    name = etree.Element('name')
    name.text = "Библиотека Максима Мошкова (lib.ru)"
    bibl.append(name)
    return name

def add_idno_to_bibl(bibl, url):
    # idno to bibl
    idno = etree.Element('idno', type="URL")
    idno.text = url
    bibl.append(idno)
    return idno

def add_availability_to_bibl(bibl):
    # availability to bibl
    availability = etree.fromstring('<availability status="free"><p>In the public domain.</p></availability>')
    bibl.append(availability)
    return availability

def add_bibl_to_bibl(bibl):
    # bibl to bibl
    new_bibl = etree.Element('bibl', type="originalSource")
    bibl.append(new_bibl)
    return new_bibl

def add_title_to_bibl(bibl, html):
    # title to bibl
    title = etree.Element('title')
    publication_text = re.sub('OCR.*', '', html.publication.replace('\n', ' '))
    publication_text = re.sub('-? Комментарии.*', '', publication_text).strip()
    title.text = publication_text
    bibl.append(title)
    return title

def year_of_publication_to_bibl(bibl, html, wikidata_id_of_play):
    # comment[0] == publication year? (to be revised)
    comment = '"' + html.comment.replace(u'\xa0', u'\n').strip().split('\n')[0].replace('"', "'") + '"' + ' (lib.ru)'
    years = re.findall('(?:[^0-9]|^)([0-9]{4,4})(?:[^0-9]|$)', comment)
    if years:
        comment_year = min([int(years[i]) for i in range(len(years))])
    else:
        comment_year = 9999

    # url of wikidata page
    wikidata_url = f"https://www.wikidata.org/wiki/{wikidata_id_of_play}"
    # wikidata year
    html_text = requests.get(wikidata_url).text
    years = re.findall('publication date.*?<div class="wikibase-snakview-value wikibase-snakview-variation-valuesnak">([0-9]{4,4})<\/div>', html_text, re.DOTALL)
    if years:
        wikidata_year = int(years[0])
    else:
        wikidata_year = 9999

    # year of publication: comment year or wikidata year
    if comment_year < wikidata_year:
        year = comment_year
    else:
        year = wikidata_year
    if year != 9999:
        date = etree.Element('date', type="print", when=str(year))
    else:
        date = etree.Element('date', type="print")

    # date to bibl
    date.text = comment
    bibl.append(date)
    return date

def year_of_premiere_to_bibl(bibl, html, wikidata_id_of_play):
    # comment[-1] == premiere year? (to be revised)
    comment = '"' + html.comment.replace(u'\xa0', u'\n').strip().strip(string.punctuation).strip().split('\n')[-1].strip().replace('"', "'") + '"' + ' (lib.ru)'
    years = re.findall('(?:[^0-9]|^)([0-9]{4,4})(?:[^0-9]|$)', comment)
    if years:
        comment_year = min([int(years[i]) for i in range(len(years))])
    else:
        comment_year = 9999

    # url of wikidata page
    wikidata_url = f"https://www.wikidata.org/wiki/{wikidata_id_of_play}"
    # wikidata year (to be revised)
    html_text = requests.get(wikidata_url).text
    years = re.findall('premiere date.*?<div class="wikibase-snakview-value wikibase-snakview-variation-valuesnak">([0-9]{4,4})<\/div>', html_text, re.DOTALL)
    if years:
        wikidata_year = int(years[0])
    else:
        wikidata_year = 9999

    # year of premiere: comment year or wikidata year
    if wikidata_year < comment_year:
        year = wikidata_year
    else:
        year = comment_year
    if year != 9999:
        date = etree.Element('date', type="premiere", when=str(year))
    else:
        date = etree.Element('date', type="premiere")

    # date to bibl
    date.text = comment
    bibl.append(date)
    return date

def year_of_written_to_bibl(bibl, html, wikidata_id_of_play, url):
    # comment year
    if html.lib_year:
        comment_year = int(html.lib_year)
    else:
        html_text = requests.get(url).text
        years = re.findall('Год: ([0-9]{4,4})', html_text, re.DOTALL)
        if years:
            comment_year = int(years[0])
        else:
            comment_year = 9999

    # url of wikidata page
    wikidata_url = f"https://www.wikidata.org/wiki/{wikidata_id_of_play}"
    # wikidata year
    html_text = requests.get(wikidata_url).text
    years = re.findall('inception.*?<div class="wikibase-snakview-value wikibase-snakview-variation-valuesnak">([0-9]{4,4})<\/div>', html_text, re.DOTALL)
    if years:
        wikidata_year = int(years[0])
    else:
        wikidata_year = 9999

    # year of inception: comment year or wikidata year
    txt = ''
    if comment_year < wikidata_year or html.lib_year:
        year = comment_year
        txt += '"' + str(year) + '"' + ' (lib.ru)'
    else:
        year = wikidata_year
        txt += '"' + str(year) + '"' + ' (wikidata.org)'
    if year != 9999:
        date = etree.Element('date', type="written", when=str(year))
    else:
        date = etree.Element('date', type="written")

    # date to bibl
    date.text = txt
    bibl.append(date)
    return date

def add_editionStmt_to_fileDesc(fileDesc):
    # editionStmt to fileDesc
    editionStmt = etree.Element('editionStmt')
    fileDesc.append(editionStmt)
    return editionStmt

def add_p_to_editionStmt(editionStmt, html):
    # p to editionStmt
    p = etree.Element('p')
    p.text = html.edition #.strip('[').strip(']')
    editionStmt.append(p)
    return p

def re_sub_names_with_numbers(name):
    tmp3 = re.sub('1-ja', transliterate('первая'), transliterate(name).lower())
    tmp3 = re.sub('2-ja', transliterate('вторая'), tmp3)
    tmp3 = re.sub('3-ja', transliterate('третья'), tmp3)
    tmp3 = re.sub('4-ja', transliterate('четвёртая'), tmp3)
    tmp3 = re.sub('5-ja', transliterate('пятая'), tmp3)
    tmp3 = re.sub('1-j', transliterate('первый'), tmp3)
    tmp3 = re.sub('2-j', transliterate('второй'), tmp3)
    tmp3 = re.sub('3-j', transliterate('третий'), tmp3)
    tmp3 = re.sub('4-j', transliterate('четвёртый'), tmp3)
    tmp3 = re.sub('5-j', transliterate('пятый'), tmp3)
    return tmp3

def common_replacement_fot_p_line(line):
    res = re.sub('\s+', ' ', html_.unescape(re.sub('(\(.*?\))', r'<stage>\1</stage>', line)).strip()).strip()
    res = re.sub(' (\.)', r'\1', res)
    res = re.sub('\. </', '.</', res)
    return res

def main_cycle(html, body, root, listPerson):
    # to save all speakers
    speakers = []

    # footnotes for the play
    footnotes = [footnote.strip() for footnote in re.findall('[0-9]{1,3}\](.*?)(?:\[|$)', html.footnotes.strip())]

    # lines of the play
    lines = html.lines
    
    # some flag (?)
    fl = 1

    # stage flag
    stage_fl = 0

    # first set
    fl_set = 0

    bias = html.popular_indent

    # для определения смыслового содержания I, II, III, V, X, ..
    # предполагаем, что если был act, то это скорее всего Tableau/Scene, иначе: part
    was_act = 0

    """
    depth = 0 # количество уровней, в которое вложено sp
    """

    # stack: body at first
    stack = [body]

    # counter of lines
    i = 0

    while i < len(lines):
        # in case there are still empty lines
        if not lines[i].strip():
            i += 1
            continue

        # no matter what is right
        line = lines[i].rstrip()

        # add <note place="foot"> tags to line
        fns = re.findall('\[[0-9]{1,3}\]', line)
        for fn in fns:
            line = re.sub(re.escape(fn), '<note place="foot">' + footnotes[int(fn[1:-1]) - 1] + '</note>', line)

        # ACT:
        if re.match('ДЕЙСТВИЕ [а-яёА-ЯЁa-zA-Z0-9-]+$', line.strip()) or re.match('Действие [а-яёА-ЯЁa-zA-Z0-9-]+$', line.strip()):
            fl_set = 0
            stage_fl = 0
            if stack[-1].tag == 'sp':
                for j in range(3):
                    del stack[-1]
                if stack[-1].tag == 'div' and 'type' in stack[-1].attrib and stack[-1].attrib['type'] in ['act', 'part', 'epilogue']:
                    # one more time
                    del stack[-1]
            elif stack[-1].tag == 'div' and 'type' in stack[-1].attrib and stack[-1].attrib['type'] in ['scene', 'tableau']:
                for j in range(2):
                    del stack[-1]
                if stack[-1].tag == 'div' and 'type' in stack[-1].attrib and stack[-1].attrib['type'] in ['act', 'part', 'epilogue']:
                    # one more time
                    del stack[-1]
            div = etree.Element('div', type="act")
            head = etree.Element('head')
            head.text = line.strip()
            div.append(head)
            stack[-1].append(div)
            stack.append(div)
            was_act = 1
        # EPILOGUE: (only in drama and almost always the last)
        elif re.match('ЭПИЛОГ', line.strip()) or re.match('Эпилог', line.strip()):
            fl_set = 1
            stage_fl = 0
            if stack[-1].tag == 'sp':
                for j in range(3):
                    del stack[-1]
                if stack[-1].tag == 'div' and 'type' in stack[-1].attrib and stack[-1].attrib['type'] in ['act', 'part', 'epilogue']:
                    del stack[-1]
            elif stack[-1].tag == 'div' and 'type' in stack[-1].attrib and stack[-1].attrib['type'] in ['scene', 'tableau']:
                for j in range(2):
                    del stack[-1]
                if stack[-1].tag == 'div' and 'type' in stack[-1].attrib and stack[-1].attrib['type'] in ['act', 'part', 'epilogue']:
                    del stack[-1]
            div = etree.Element('div', type="epilogue")
            head = etree.Element('head')
            head.text = line.strip()
            div.append(head)
            stack[-1].append(div)
            stack.append(div)
        # SCENE:
        elif re.match(r'\[?ЯВЛЕНИЕ [а-яёА-ЯЁa-zA-Z0-9-]+', line.strip()) or re.match(r'\[?Явление [а-яёА-ЯЁa-zA-Z0-9-]+', line.strip()):
            fl_set = 1
            stage_fl = 0
            if stack[-1].tag == 'sp':
                for j in range(2):
                    del stack[-1]
            elif stack[-1].tag == 'div' and 'type' in stack[-1].attrib and stack[-1].attrib['type'] == 'scene':
                del stack[-1]
            div = etree.Element('div', type="scene")
            head = etree.Element('head')
            head.text = line.strip()
            div.append(head)
            stack[-1].append(div)
            stack.append(div)
        # SCENE/TABLEAU
        elif re.match('КАРТИНА [а-яёА-ЯЁa-zA-Z]+', line.strip()) or re.match('Картина [а-яёА-ЯЁa-zA-Z]+', line.strip()) or re.match('СЦЕНА [а-яёА-ЯЁa-zA-Z]+', line.strip()) or re.match('Сцена [а-яёА-ЯЁa-zA-Z]+', line.strip()) \
        or line.strip() == 'I' or line.strip() == 'II' or line.strip() == 'III' or line.strip() == 'IV' or line.strip() == 'V' or line.strip() == 'X' or re.match('I ', line.strip()) or re.match('V ', line.strip()) or re.match('X ', line.strip()):
            fl_set = 1
            stage_fl = 0
            if stack[-1].tag == 'sp':
                for j in range(2):
                    del stack[-1]
                if was_act:
                    del stack[-1]
            elif stack[-1].tag == 'div' and 'type' in stack[-1].attrib and stack[-1].attrib['type'] == 'scene':
                del stack[-1]
                if was_act:
                    del stack[-1]
            if re.search('СЦЕНА [а-яёА-ЯЁa-zA-Z]+', line.strip()) or re.search('Сцена [а-яёА-ЯЁa-zA-Z]+', line.strip()) or re.search('I', line.strip()) or re.search(r'V', line.strip()) or re.search(r'X', line.strip()):
                if was_act or re.search('СЦЕНА [а-яёА-ЯЁa-zA-Z]+', line.strip()) or re.search('Сцена [а-яёА-ЯЁa-zA-Z]+', line.strip()):
                    div = etree.Element('div', type="tableau")
                else:
                    div = etree.Element('div', type="part")
            else:
                div = etree.Element('div', type="scene")
            head = etree.Element('head')
            head.text = line.strip()
            div.append(head)
            stack[-1].append(div)
            stack.append(div)
        # ДЕЙСТВУЮЩИЕ ЛИЦА
        elif re.match('(ДЕЙСТВУЮЩИЕ )?ЛИЦА(:)?', line.strip()) or re.match('(Действующие )?лица(:)?', line.strip()) or re.match('(действующие )?Лица(:)?', line.strip()):
            fl_set = 1 # предполагаем, что после действующих лиц не может идти set
            stage_fl = 0
            castList = etree.Element('castList')
            stack[-1].append(castList)

            head = etree.Element('head')
            head.text = line.strip()
            castList.append(head)

            i += 1
            fl_castgroup = 0
            # print(lines[i])
            while re.match(fr'\s{{{bias},{bias}}}', lines[i]):
                fns = re.findall('\[[0-9]{1,3}\]', lines[i])
                for fn in fns:
                    lines[i] = re.sub(re.escape(fn), '<note place="foot">' + footnotes[int(fn[1:-1]) - 1] + '</note>', lines[i])
                if '|' in lines[i]:
                    fl_castgroup = 0
                    castGroup = etree.Element('castGroup')
                    castList.append(castGroup)
                    tmp = lines[i].split('}')
                    for elem in tmp[0].split('|'):
                        castItem = etree.Element('castItem')
                        castItem.text = elem.strip()
                        castGroup.append(castItem)
                    roleDesc = etree.Element('roleDesc')
                    roleDesc.text = tmp[1].strip()
                    castGroup.append(roleDesc)
                elif not lines[i].strip().endswith(',') and ', и ' in lines[i] and len([dash for dash in dashes if lines[i].count(dash)]) == 1:
                    fl_castgroup = 0
                    castGroup = etree.Element('castGroup')
                    castList.append(castGroup)
                    for dash in dashes:
                        if len(lines[i].split(dash)) > 1:
                            break
                    tmp = lines[i].split(dash)[0].split(', и ')
                    for elem_ind, elem in enumerate(tmp):
                        castItem = etree.Element('castItem')
                        if elem_ind:
                            if elem_ind != len(tmp) - 1:
                                castItem.text = 'и ' + elem.strip() + ','
                            else:
                                castItem.text = 'и ' + elem.strip()
                        else:
                            castItem.text = elem.strip() + ','
                        castGroup.append(castItem)
                    roleDesc = etree.Element('roleDesc')
                    roleDesc.text = dash + ' ' + lines[i].split(dash)[-1].strip()
                    castGroup.append(roleDesc)
                elif fl_castgroup == 0 and lines[i].strip().endswith(','):
                    fl_castgroup = 1 # X, Y, Z, ..., W, role
                    castGroup = etree.Element('castGroup')
                    castItem = etree.Element('castItem')
                    castItem.text = lines[i].strip()
                    castGroup.append(castItem)
                elif fl_castgroup == 0 and lines[i].strip().endswith(':'):
                    fl_castgroup = 2 # role: X. Y. Z. W. ...
                    castGroup = etree.Element('castGroup')
                    castList.append(castGroup)
                    roleDesc = etree.Element('roleDesc')
                    roleDesc.text = lines[i].strip()
                    castGroup.append(roleDesc)
                elif lines[i].strip().endswith(',') or (fl_castgroup == 2 and ',' not in lines[i].strip()[:-1]):
                    castItem = etree.Element('castItem')
                    castItem.text = lines[i].strip()
                    castGroup.append(castItem)
                elif fl_castgroup == 1:
                    if lines[i].split(',')[-1].strip()[0] in lower_alphabet:
                        castList.append(castGroup)
                        castItem = etree.Element('castItem')
                        castItem.text = ','.join(lines[i].split(',')[:-1]).strip() + ','
                        castGroup.append(castItem)
                        roleDesc = etree.Element('roleDesc')
                        roleDesc.text = lines[i].split(',')[-1].strip()
                        castGroup.append(roleDesc)
                    else:
                        for castItem in castGroup:
                            castList.append(castItem)
                        castItem = etree.Element('castItem')
                        castItem.text = lines[i].strip()
                        castList.append(castItem)
                    fl_castgroup = 0
                else:
                    fl_castgroup = 0
                    castItem = etree.Element('castItem')
                    castItem.text = lines[i].strip()
                    castList.append(castItem)
                i += 1
            i -= 1
        # a set if there is a set
        elif fl_set == 0:
            stage_fl = 0
            set_ = etree.Element('set')
            set_.text = line.strip() #.strip('[').strip(']')
            stack[-1].append(set_)
        # p
        elif html.porl == 'p' and (re.match(fr'\s{{{bias},{bias}}}[-0-9А-Яа-яa-zA-Z]+(?: [-0-9А-Яа-яa-zA-Z]+)?(?: [-0-9А-Яа-яa-zA-Z]+)?\.[^.]+', line) or re.match(fr'\s{{{bias},{bias}}}[-0-9А-Яа-яa-zA-Z]+(?: [-0-9А-Яа-яa-zA-Z]+)? \([^)]*?\)\.[^.]', line)):
            fl_set = 1
            stage_fl = 0
            fl = 0
            if re.match(fr'\s{{{bias},{bias}}}[-0-9А-Яа-яa-zA-Z]+(?: [-0-9А-Яа-яa-zA-Z]+)? \(.*?\)\.', line):
                tmp = re.findall(fr'\s{{{bias},{bias}}}[-0-9А-Яа-яa-zA-Z]+(?: [-0-9А-Яа-яa-zA-Z]+)? (\(.*?\))\.', line)[0]
                stage = etree.Element('stage')
                stage.text = tmp
                tmp = re.findall(fr'\s{{{bias},{bias}}}([-0-9А-Яа-яa-zA-Z]+(?: [-0-9А-Яа-яa-zA-Z]+)?) \(.*?\)\.(.*)', line)[0]
                fl = 1
            else:
                fl3 = 0
                tmp4 = re.findall(fr'\s{{{bias},{bias}}}([-0-9А-Яа-яa-zA-Z]+(?: [-0-9А-Яа-яa-zA-Z]+)?(?: [-0-9А-Яа-яa-zA-Z]+)?)\.', line)[0]
                for elem_ in tmp4.split():
                    if not re.match('[0-9А-Я]', elem_) and elem_ != 'и':
                        fl3 = 1
                        break
                if fl3 == 0 and not re.match(fr'\s{{{bias},{bias}}}[А-ЯA-Z][а-яa-z]*\.', line):
                    tmp = re.findall(fr'\s{{{bias},{bias}}}([-0-9А-Яа-яa-zA-Z]+(?: [-0-9А-Яа-яa-zA-Z]+)?(?: [-0-9А-Яа-яa-zA-Z]+)?)\. (.*)', line)[0]
                else:
                    if re.match(fr'\s{{{bias},{bias}}}([-а-яА-Яa-zA-Z ]+){1,3}\.$', line) or re.match(fr'\s{{{bias},{bias}}}([-а-яА-Яa-zA-Z ]+){1,3}\.{3}', line):
                        p = etree.Element(html.porl)
                        p.text = common_replacement_fot_p_line(line)
                        stack[-1].append(p)
                        i += 1
                        continue
                    elif fl3 == 1 and (re.match(fr'\s{{{bias},{bias}}}[-0-9А-Яа-яa-zA-Z]+(?: [-0-9А-Яа-яa-zA-Z]+)?(?: [-0-9А-Яа-яa-zA-Z]+)?\..*', line) or re.match(fr'\s{{{bias},{bias}}}[-0-9А-Яа-яa-zA-Z]+(?: [-0-9А-Яа-яa-zA-Z]+)?(?: [-0-9А-Яа-яa-zA-Z]+)?\..*', line)):
                        try:
                            qwe = re.findall(fr'\s{{{bias},{bias}}}((?:[-0-9А-Яа-яa-zA-Z]+(?: [-0-9А-Яа-яa-zA-Z]+)?(?: [-0-9А-Яа-яa-zA-Z]+)?\.).*?)(\(.*?\))', line)[0]
                        except:
                            qwe = re.findall(fr'\s{{{bias},{bias}}}[-0-9А-Яа-яa-zA-Z]+(?: [-0-9А-Яа-яa-zA-Z]+)?(?: [-0-9А-Яа-яa-zA-Z]+)?\..*', line)[0]
                        p = etree.Element(html.porl)
                        p.text = common_replacement_fot_p_line(qwe[0])
                        stack[-1].append(p)
                        if qwe[1]:
                            stage = etree.Element('stage')
                            stage.text = qwe[1].strip()
                            stack[-1].append(stage)
                        i += 1
                        continue
                    tmp = re.findall(fr'\s{{{bias},{bias}}}([-0-9А-Яа-яa-zA-Z]+(?: [-0-9А-Яа-яa-zA-Z]+)?(?: [-0-9А-Яа-яa-zA-Z]+)?)\. (.*)', line)[0]

            name = tmp[0].strip()
            if name not in speakers:
                speakers.append(name)
            if stack[-1].tag == 'sp':
                del stack[-1]
            
            tmp3 = re_sub_names_with_numbers(name)
            sp = etree.Element('sp', who="#" + '_'.join(tmp3.split()))
            speaker = etree.Element('speaker')
            speaker.text = name
            sp.append(speaker)
            stack[-1].append(sp)
            stack.append(sp)
            if fl == 1:
                stack[-1].append(stage)
            text = tmp[1].strip()
            if re.match(r'(.*)(\(.*?\.\))$', text.strip()):
                tmp = re.findall(r'(.*)(\(.*?\.\))$', text.strip())[0]
                text = tmp[0].strip()
                if text:
                    p = etree.Element(html.porl)
                    if re.match('\(.*?\)', text):
                        tmp2 = re.findall('(\(.*?\))(.*)', text)[0]
                        stage = etree.Element('stage')
                        stage.text = tmp2[0].strip()
                        stack[-1].append(stage)
                        p.text = common_replacement_fot_p_line(tmp2[1])
                    else:
                        p.text = common_replacement_fot_p_line(text)
                    stack[-1].append(p)
                if tmp[1].strip():
                    stage = etree.Element('stage')
                    stage.text = tmp[1].strip()
                    stack[-1].append(stage)
            else:
                if text:
                    p = etree.Element(html.porl)
                    p.text = common_replacement_fot_p_line(text)
                    stack[-1].append(p)
        elif html.porl == 'p' and stage_fl:
            # Добавить stage для внутренних и отдельный для последнего
            tmp = re.findall('^(\(.*?\)) (.*)', line.strip())
            if tmp:
                stage = etree.Element('stage')
                stage.text = tmp[0][0].strip()
                stack[-1].append(stage)
                line = tmp[0][1]
            p = etree.Element(html.porl)
            p.text = line.strip()
            stack[-1].append(p)
            stage_fl = 0
        # speaker
        elif html.porl == 'l' and re.match('(^[А-ЯЁA-Z0-9][А-ЯЁа-яёa-z--]+(?:(?:,? [а-я]{1,5})?,? [А-ЯЁа-яёa-zA-Z-]+(?: [А-ЯЁа-яёa-zA-Z]+)?(?: [А-ЯЁа-яёa-zA-Z-]+)?)?)(?: (\[.*?\]))?$', line):
            fl_set = 1
            stage_fl = 0
            tmp = re.findall('(^[А-ЯЁA-Z0-9][А-ЯЁа-яёa-z-]+(?:(?:,? [а-я]{1,5})?,? [А-ЯЁа-яёa-zA-Z-]+(?: [А-ЯЁа-яёa-zA-Z]+)?(?: [А-ЯЁа-яёa-zA-Z-]+)?)?)(?: (\[.*?\]))?$', line)
            name = tmp[0][0].strip()
            for a, b in zip('Heo', 'Нео'): # только в имени меняем, т.к. в произвольном случае таким образом можно поменять лишнее
                name = re.sub(a, b, name) # попробовать перенести выше

            tmp3 = re_sub_names_with_numbers(name)
            if name not in speakers:
                if ' и ' in name:
                    tmp_i = name.split(' и ')
                    for elem_name in tmp_i[0].split(','):
                        if elem_name.strip().capitalize() not in speakers:
                            speakers.append(elem_name.strip().capitalize())
                    if tmp_i[1].capitalize() not in speakers:
                        speakers.append(tmp_i[1].capitalize())
                speakers.append(name)
            if stack[-1].tag == 'sp':
                del stack[-1]

            sp = etree.Element('sp', who="#" + '_'.join(tmp3.split()))
            speaker = etree.Element('speaker')
            speaker.text = name
            sp.append(speaker)
            if tmp[0][1]:
                stage = etree.Element('stage')
                stage.text = tmp[0][1]
                sp.append(stage)
            stack[-1].append(sp)
            stack.append(sp)
        # l
        elif html.porl == 'l' and re.match(fr'\s{{{bias},{bias}}}', line):
            fl_set = 1
            stage_fl = 0
            text = line
            if re.match(r'(.*)(\(.*?\.\))$', text.strip()):
                tmp = re.findall(r'(.*)(\(.*?\.\))$', text.strip())[0]
                text = tmp[0].strip()
                if text:
                    p = etree.Element(html.porl)
                    if re.match(fr'\s{{{bias},{bias + 1}}}[а-яёА-ЯЁa-zA-Z"]', line):
                        p.set('part', 'I')
                    else:
                        p.set('part', 'F')
                    p.text = common_replacement_fot_p_line(text)
                    stack[-1].append(p)
                if tmp[1].strip():
                    stage = etree.Element('stage')
                    stage.text = tmp[1].strip()
                    stack[-1].append(stage)
            else:
                if text:
                    p = etree.Element(html.porl)
                    if re.match(fr'\s{{{bias},{bias + 1}}}[а-яёА-ЯЁa-zA-Z"]', line):
                        p.set('part', 'I')
                    else:
                        p.set('part', 'F')
                    if re.match(fr'\s{{{bias},{bias + 1}}}\(.*?\)', text):
                        tmp = re.findall('(\(.*?\))(.*)', text)[0]
                        stage = etree.Element('stage')
                        stage.text = tmp[0].strip()
                        stack[-1].append(stage)
                        p.text = common_replacement_fot_p_line(tmp[1])
                    else:
                        p.text = common_replacement_fot_p_line(text)
                    stack[-1].append(p)
        # tough cases
        elif line.strip() in ['В народе']:
            fl_set = 1
            stage_fl = 0
            old_name = line.strip()
            if line.strip() == 'В народе':
                name = 'Народ'
            if name not in speakers:
                speakers.append(name)
            if stack[-1].tag == 'sp':
                del stack[-1]
            tmp3 = transliterate(name).lower()
            sp = etree.Element('sp', who="#" + '_'.join(tmp3.split()))
            speaker = etree.Element('speaker')
            speaker.text = old_name
            sp.append(speaker)
            stack[-1].append(sp)
            stack.append(sp)
        # Stage
        else:
            fl_set = 1
            if i != len(lines) - 1 and stack[-1].tag == 'sp' and \
            (re.match(fr'\s{{{bias},{bias}}}[-123А-Яа-яa-zA-Z]+(?: [123А-Яа-яa-zA-Z]+)?( \(.*?\))?\.[^\.](.+)?', lines[i+1]) or \
             (re.match(r'ДЕЙСТВИЕ .*?', lines[i+1]) or re.match(r'\[?ЯВЛЕНИЕ .*?', lines[i+1]) or re.match(r'ДЕЙСТВУЮЩИЕ ЛИЦА', lines[i+1]) or re.match('[-123а-яА-Яa-zA-Z]+', lines[i+1]) or re.match('(   )?I', lines[i+1]) or re.match('(   )?V', lines[i+1]) or re.match('(   )?X', lines[i+1]) or re.match('   КОММЕНТАРИИ', lines[i+1]))):
                if re.match(fr'\s{{{bias},{bias}}}[-123А-Яа-яa-zA-Z]+(?: [-123А-Яа-яa-zA-Z]+)?( \(.*?\))?\.(?:\.\.)?(.+)', lines[i+1]):
                    fl3 = 0
                    tmp4 = re.findall(fr'\s{{{bias},{bias}}}([-123А-Яа-яa-zA-Z]+(?: [-123А-Яа-яa-zA-Z]+)?)(?: \(.*?\))?\.(?:\.\.)?(?:.+)?', lines[i+1])[0]
                    for elem_ in tmp4.split():
                        if not re.match('[0-9А-ЯA-Z]', elem_):
                            fl3 = 1
                            break
                    """
                    if fl3 == 0:
                        del stack[-1]
                else:
                    del stack[-1]
                    """
            """
            if stack and stack[-1] and stack[-1][-1].tag == 'stage':
                del stack[-1]
            """
            if stage_fl == 0 and stack[-1].tag == 'body':
                set_ = etree.Element('set')
                set_.text = line.strip()
                stack[-1].append(set_)
            else:
                stage = etree.Element('stage')
                stage.text = line.strip()
                stack[-1].append(stage)
            stage_fl = 1
        i += 1

        # Mystem: определение рода и числа
    out = m.analyze('~'.join(speakers))

    out = [elem for elem in out if elem['text'] != ' ' and elem['text'] != '\n']

    # Удаление ошибочных решений типа "Входит Биркин" -- sp
    i = 0
    count_tilda = 0
    indices_to_be_removed = []
    first_words_to_be_removed = []
    while i < len(out) - 1:
        if (out[i]['text'] == '~' or i == 0) and 'analysis' in out[i + 1] and 'V,' in out[i + 1]['analysis'][0]['gr']:
            indices_to_be_removed.append(i)
            first_words_to_be_removed.append(out[i + 1]['text'])
            i += 1
            sp_to_be_removed = []
            while out[i]['text'] != '~' and i < len(out):
                sp_to_be_removed.append(out[i]['text'])
                indices_to_be_removed.append(i)
                i += 1
            sp_to_be_removed = '#' + transliterate('_'.join(sp_to_be_removed).lower())
            result_sp = root.find('.//div/sp[@who="' + sp_to_be_removed + '"]')
            children_of_removed_sp = result_sp.getchildren()
            parent = result_sp.getparent()
            children_of_parent = parent.getchildren()
            for child_ind, child in enumerate(children_of_parent[:-1]):
                if child_ind == 0 and result_sp == child: # не тестил и не ясно, возможна ли такая ситуация
                    stage = etree.Element('stage')
                    stage.text = children_of_removed_sp[0].text
                    parent.insert(0, stage)
                    for children_of_removed_sp_ind, children_of_removed_sp_elem in enumerate(children_of_removed_sp[1:]):
                        parent.insert(children_of_removed_sp_ind + 1, children_of_removed_sp_elem)
                elif children_of_parent[child_ind + 1] == result_sp:
                    stage = etree.Element('stage')
                    stage.text = children_of_removed_sp[0].text
                    children_of_parent[child_ind].append(stage)
                    for children_of_removed_sp_ind, children_of_removed_sp_elem in enumerate(children_of_removed_sp[1:]):
                        children_of_parent[child_ind].append(children_of_removed_sp_elem)
                    
            parent.remove(result_sp)
                
        else:
            i += 1
    
    for index_to_be_removed in indices_to_be_removed[::-1]:
        del out[index_to_be_removed]
        for i, speaker in enumerate(speakers[::-1]):
            if speaker.split()[0] in first_words_to_be_removed:
                del speakers[len(speakers) - 1 - i]

    i = 0
    genders = ['' for elem in speakers]
    # print(speakers)
    for ind_of_elem, elem in enumerate(out):
        if elem['text'] != '~': # совокупность m, f, +, - даст представление о том, какой тип присваивать
            # print(elem['text'], ind_of_elem, out[ind_of_elem - 1])
            if elem['text'] == 'й' and ind_of_elem and out[ind_of_elem - 1]['text'] == '-':
                genders[i] += 'm'
                continue
            elif elem['text'] == 'я' and ind_of_elem and out[ind_of_elem - 1]['text'] == '-':
                genders[i] += 'f'
                continue
            if 'analysis' not in elem:
                continue
            grs = elem['analysis'][0]['gr']
            grs = re.sub(',', ' ', grs)
            for ii in range(grs.count('|') + grs.count(' ') + 1):
                grs = re.sub('(\(.*?)\s', r'\1,', grs)
            grs = grs.split(' ')
            base = []
            for gr in grs:
                if '=' in gr:
                    base.append(gr.split('=')[0])
                    variable_grs = gr.split('=')[1]
                else:
                    base.append(gr)
            # (пр,ед,жен|дат,ед,жен|род,ед,жен|твор,ед,жен|вин,ед,муж,неод|им,ед,муж)
            for gr in variable_grs.split('|'):
                grs = base[:]
                gr = gr.strip('(').strip(')')
                grs.extend(gr.split(','))
                # print(grs, elem['text'])
                if 'муж' in grs and 'им' in grs and 'мн' not in grs and 'неод' not in grs and ('APRO' not in grs or ('APRO' in grs and (elem['analysis'][0]['wt'] <= 0.9 or out[ind_of_elem + 1]['text'] == '~' or ind_of_elem + 1 == len(out)))):
                    genders[i] += 'm'
                elif 'жен' in grs and 'им' in grs and 'мн' not in grs and 'неод' not in grs and ('APRO' not in grs or ('APRO' in grs and (elem['analysis'][0]['wt'] <= 0.9  or out[ind_of_elem + 1]['text'] == '~' or ind_of_elem + 1 == len(out)))):
                    genders[i] += 'f'
                elif 'мн' in grs or elem['text'] == 'и' or 'NUM' in grs:
                    if 'им' in grs or 'NUM' in grs:
                        genders[i] += '+' # nom pl
                    if 'муж' in grs and 'неод' not in grs:
                        genders[i] += 'm' # pl m anim
                    elif 'жен' in grs and 'неод' not in grs:
                        genders[i] += 'f' # pl f anim
                elif 'муж' not in grs and 'жен' not in grs:
                    genders[i] += '-'
        else:
            i += 1

    for sp_num, name in enumerate(speakers):
        if name.lower() == 'все' or ' и ' in name.lower(): # не добавляем эти случаи
            if ' и ' in name.lower():
                result_sp = root.find('.//div/sp[@who="' + '#' + '_'.join(transliterate(name.lower()).split()) + '"]')
                tmp = name.lower().split(' и ')
                tmp_string = ''
                for part_tmp in tmp[0].split(','):
                    tmp_string += '#' + transliterate(part_tmp.strip()) + ' '
                tmp_string += '#' + transliterate(tmp[1])
                result_sp.set('who', tmp_string)
            continue
        person = etree.Element('person')
        tmp3 = re_sub_names_with_numbers(name)
        # вероятно, нужно добавить 3-й, 4-й и т.д.
        person.set("{http://www.w3.org/XML/1998/namespace}id", '_'.join(tmp3.split()))
        # print(name, genders[sp_num])
        if 'm' in genders[sp_num] and (not(set('f+') & set(genders[sp_num])) or genders[sp_num].count('m') > genders[sp_num].count('+') * 2): # 2?
            gndr = 'MALE'
        elif 'f' in genders[sp_num] and (not(set('m+') & set(genders[sp_num])) or genders[sp_num].count('f') > genders[sp_num].count('+') * 2):
            gndr = 'FEMALE'
        elif (not(len(genders[sp_num])) and name.lower() in ['народ', 'толпа']) or ('+' in genders[sp_num] and (not(('m' in genders[sp_num] or 'f' in genders[sp_num] or len(genders[sp_num]) < 3) and ' iz ' in tmp3) or (' iz ' in tmp3 and genders[sp_num][0] == '+'))): # odin iz tolpy...
            personGrp = etree.Element('personGrp')
            personGrp.set("{http://www.w3.org/XML/1998/namespace}id", '_'.join(tmp3.split()))
            if 'm' in genders[sp_num] and 'f' not in genders[sp_num]:
                gndr = 'MALE'
            elif 'f' in genders[sp_num] and 'm' not in genders[sp_num]:
                gndr = 'FEMALE'
            else:
                gndr = 'UNKNOWN'
            if re.search('(^| )дети($| )', name.lower()) or re.search('(^| )слепые($| )', name.lower()): # necessary костыль
                gndr = 'UNKNOWN'
            personGrp.set('sex', gndr)
            persName = etree.Element('name')
            persName.text = name
            personGrp.append(persName)
            listPerson.append(personGrp)
            continue
        elif '+' in genders[sp_num] and genders[sp_num][0] == 'm':
            gndr = 'MALE'
        elif '+' in genders[sp_num] and genders[sp_num][0] == 'f':
            gndr = 'FEMALE'
        else:
            gndr = 'UNKNOWN'
        if re.search('(^| )голос($| )', name.lower()) or re.search('(^| )голос($| )', name.lower()): # necessary костыль
            gndr = 'UNKNOWN'

        # correcting gender by looking at context of their presentations
        # may cause errors potentially so pay attention
        castItems = root.findall('.//castItem')
        for castItem in castItems:
            castItem_text = castItem.text.lower()
            castItem_text = re.sub('\(.*?\)', '', castItem_text)
            castItem_text = ''.join([symbol for symbol in castItem_text if symbol not in exclude]).split()
            if name.lower() in castItem_text:
                name_index = castItem_text.index(name.lower())
                if name_index + 1 < len(castItem_text):
                    analysis = m.analyze(castItem_text[name_index + 1])[0]['analysis'][0]['gr']
                    if 'жен' in analysis and 'им' in analysis:
                        gndr = 'FEMALE'
                    elif 'муж' in analysis and 'им' in analysis:
                        gndr = "MALE"
                break
        
        person.set('sex', gndr)
        persName = etree.Element('persName')
        persName.text = name
        person.append(persName)
        listPerson.append(person)

    # "#vse" -> "#x #y #z"
    # заменяем все "все" на людей из контекста (довольно неточно)
    divs_scene = root.findall('.//div[@type = "scene"]')
    for scene in divs_scene:
        whos_vse = []
        sp_vse = []
        sp_and = []
        sp_iterator = scene.iter(tag="sp")
        for sp in sp_iterator:
            if sp.attrib['who'] == '#vse':
                sp_vse.append(sp)
            else:
                whos_vse.append(sp.attrib['who'])
            if '_i_' in sp.attrib['who']:
                sp_and.append(sp)
        if sp_vse:
            for sp in sp_vse:
                sp.attrib['who'] = ' '.join(set(whos_vse))
        if sp_and:
            for sp in sp_and:
                tmp_sp = ['#' + elem for elem in sp.attrib['who'][1:].split('_') if elem != 'i']
                sp.attrib['who'] = ' '.join(tmp_sp)

    # удалить пустые l + если part="F" или "I", то очистить следующие значения
    iterator = root.iter(tag="l")
    for l in iterator:
        if not (l.text):
            if 'part' in l.attrib:
                if l.attrib['part'] == 'I':
                    l_next = next(iterator)
                    if l_next.attrib['part'] == 'M':
                        l_next.set('part', 'I')
                    else: # l_next['part'] == 'F'
                        del l_next.attrib['part']
                elif l.attrib['part'] == 'M': # просто удаляем
                    pass
                else: # l.attrib['part'] == 'F'
                    if 'part' in prev.attrib:
                        if prev.attrib['part'] == 'M':
                            prev.set('part', 'F')
                        else: # prev.attrib['part'] == 'I':
                            del prev.attrib['part']
                        
            l.getparent().remove(l)
        prev = l
                
    # <div type="scene">
    # l --> l part="I/M/F"
    iterator = root.iter(tag="l")
    childs = []
    for child in iterator:
        childs.append(child)
    fl_f = 0
    for child in childs[::-1]:
        # print(child.text, child.attrib['part'], fl_f)
        if 'part' not in child.attrib:
            fl_f = 0
        elif fl_f == 0 and child.attrib['part'] == 'I':
            del child.attrib['part']
        elif fl_f == 1 and child.attrib['part'] == 'I':
            fl_f = 0
        elif fl_f == 1 and child.attrib['part'] == 'F':
            child.set('part', 'M')
        elif fl_f == 0 and child.attrib['part'] == 'F':
            fl_f = 1

    # считаем, что в пределах сцены всё по большей части единообразно
    iterator = root.iter(tag="div")
    divs = []
    for div in iterator:
        if div.attrib['type'] in ['scene']: # , 'tableau'
            divs.append(div)

    # l --> p
    for div in divs:
        iterator = div.iter(tag="l")
        fl_l = 0
        for l in iterator:
            if len(l.text) > 2 * html.mean_length and '<note place="foot">' not in l.text:
                fl_l = 1
                break
        if fl_l:
            iterator = div.getiterator(tag="l")
            for l in iterator:
                l.tag = 'p'

    # p --> l
    for div in divs:
        iterator = div.iter(tag="p")
        fl_l = 0
        for p in iterator:
            if fl_l == 0 and 'part' in p.attrib:
                p.tag = 'l'
                fl_l = 1
            elif fl_l == 1 and (len(p.text) < 2 * html.mean_length or '<note place="foot">' in l.text):
                p.tag = 'l'

    # stage на уровень ниже
    for div in divs:
        iterator = div.iter(tag="sp")
        for sp in iterator:
            sp_subtags = list(sp.iter())
            if sp_subtags:
                if sp_subtags[-1].tag == 'stage' and not (sp_subtags[-1].text[0] == '(' and (sp_subtags[-1].text[-1] == ')' or sp_subtags[-1].text[-2] == ')')):
                    stage = etree.Element('stage')
                    stage.text = sp_subtags[-1].text
                    sp.remove(sp_subtags[-1])
                    div.insert(div.index(sp) + 1, stage)

    # удалить пустые l + если part="F" или "I", то очистить следующие значения
    iterator = root.iter(tag="l")
    for l in iterator:
        if not (l.text):
            if 'part' in l.attrib:
                if l.attrib['part'] == 'I':
                    l_next = next(iterator)
                    if l_next.attrib['part'] == 'M':
                        l_next.set('part', 'I')
                    else: # l_next['part'] == 'F'
                        del l_next.attrib['part']
                elif l.attrib['part'] == 'M': # просто удаляем
                    pass
                else: # l.attrib['part'] == 'F'
                    if 'part' in prev.attrib:
                        if prev.attrib['part'] == 'M':
                            prev.set('part', 'F')
                        else: # prev.attrib['part'] == 'I':
                            del prev.attrib['part']
                        
            l.getparent().remove(l)
        prev = l

    # те реплики, которые состоят из нескольких p, на самом деле скорее всего просто l
    iterator = root.iter(tag="sp")
    for elem_iter in iterator:
        count_p = 0
        iterator = elem_iter.getiterator(tag="p")
        for child in iterator:
            if child.tag == 'p':
                count_p += 1
        if count_p > 1:
            iterator = elem_iter.getiterator(tag="p")
            for child in iterator:
                child.tag = 'l'

    result = etree.tostring(root.getroottree(), pretty_print=True, xml_declaration=True, encoding="UTF-8").decode('utf-8')
#     text = html_.unescape(result.replace("'", '"')) # здесь удаляем лишнее как, например, в Vive l'empereur
    text = html_.unescape(result)
    
    text = re.sub('([a-zA-Zа-яА-Я])"([a-zA-Zа-яА-Я"_-])', r'\1\2', text)
    text = re.sub('~', '', text)
    text = re.sub('[^ ] </', '</', text)

    # remove <stage> tag from <note> tag
    text = re.sub('(<note place=\"foot\">.*?)<stage>(.*?)<\/stage>(.*?)(<\/note>)', r'\1\2\3\4', text)

    with open(os.path.join('.', 'xml_piece.xml'), 'w', encoding='utf-8') as f:
        f.write(text)

def convert_html_to_xml(url):
    text = requests.get(url).text
    html = preprocess_text(text)
    html = correct_mistakes(html)
    root = get_root()
    #-------------------
    teiHeader = add_teiHeader_to_root(root)

    #   ----------------
    fileDesc = add_fileDesk_to_teiHeader(teiHeader)
    #      -------------
    titleStmt = add_titleStmt_to_fileDesc(fileDesc)
    #         ----------
    title = add_ru_title_to_titleStmt(titleStmt, html)
    title, wikidata_id_of_play = add_en_title_to_titleStmt(titleStmt, html)
    title = add_ru_sub_title_to_titleStmt(titleStmt, html)
    title = add_en_sub_title_to_titleStmt(titleStmt, html)
    author = add_author_to_titleStmt(titleStmt, html)

    #      -------------
    publicationStmt = add_publicationStmt_to_fileDesc(fileDesc)
    #         -------------
    publisher = add_publisher_to_publicationStmt(publicationStmt)
    idno = add_name_idno_to_publicationStmt(publicationStmt, html)
    idno = add_num_of_play_idno_to_publicationStmt(publicationStmt, html)
    availability = add_availability_to_publicationStmt(publicationStmt)
    idno = add_wikidata_idno_to_publicationStmt(publicationStmt, html, wikidata_id_of_play)

    #      -------------
    sourceDesc = add_sourceDesc_to_fileDesc(fileDesc)
    #         ----------
    bibl = add_bibl_to_sourceDesc(sourceDesc)
    #            -------
    name = add_name_to_bibl(bibl)
    idno = add_idno_to_bibl(bibl, url)
    availability = add_availability_to_bibl(bibl)
    bibl = add_bibl_to_bibl(bibl)
    #            -------
    title = add_title_to_bibl(bibl, html)
    date1 = year_of_publication_to_bibl(bibl, html, wikidata_id_of_play)
    date2 = year_of_premiere_to_bibl(bibl, html, wikidata_id_of_play)
    date3 = year_of_written_to_bibl(bibl, html, wikidata_id_of_play, url)
    # print(date3.attrib['when'])

    #      -------------
    if html.edition:
        editionStmt = add_editionStmt_to_fileDesc(fileDesc)
    #         ----------
        p = add_p_to_editionStmt(editionStmt, html)

    profileDesk = add_profileDesk_to_teiHeader(teiHeader)
    #   ----------------
    
    #      -------------
    partiDesc = add_partiDesc_to_profileDesc(profileDesk)
    #         ----------
    listPerson = add_listPerson_to_partiDesc(partiDesc)
    
    textClass = add_textClass_to_profileDesc(profileDesk)
    #      -------------
    keywords = add_keywords_to_textClass(textClass)
    #         ----------
    term = add_term_to_keywords(keywords, html)


    
    #   ----------------
    revisionDesc = add_revisionDesc_to_teiHeader(teiHeader)
    #      -------------
    listChange = add_listChange_to_revisionDesc(revisionDesc)
    #         ----------
    change = add_change_to_listChange(listChange)

    #-------------------
    text = add_text_to_root(root)
    #   ----------------
    front = add_front_to_text(text)
    #      -------------
    docTitle = add_docTitle_to_front(front)
    #         ----------
    titlePart = add_titlePart_main_to_docTitle(docTitle, html)
    #         ----------
    titlePart = add_titlePart_sub_to_docTitle(docTitle, html)

    #   ----------------
    body = add_body_to_text(text)

    main_cycle(html, body, root, listPerson)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('url', type=str)
    arguments = parser.parse_args()
    convert_html_to_xml(arguments.url)
