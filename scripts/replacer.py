#!/usr/bin/env python
# -*- coding: utf-8 -*-

from country_list import countries_for_language


countries = dict(countries_for_language('en'))
country_ = [country.lower() for country in list(countries.values())]


def replace_countries(sent):
    for country in [country for country in list(countries.values())]:
        sent = sent.replace(country.lower(),country)
    return sent

 


def replacer(sent):
    sent = replace_countries(sent)
    sent = sent.replace('america','American')
    sent_list = sent.split()

    non_end_words = [
        'but', 
        'we'
    ]
    for token in non_end_words:
        if sent_list[-1] == token:
            sent = ' '.join(sent_list[:-1])
            break
    
    sent = sent.replace('Http','http')
    sent = sent.replace('\\"','"')
    sent = sent.replace(' i ',' I ')
    sent = sent.replace(' Korea North ',' North Korea ')
     
        

    return sent


def format(sent):
    out = "# {}"
    

if __name__== "__main__":
    sent = '\\" mexico is very cool, like venezuela, but'
    print(replacer(sent))

