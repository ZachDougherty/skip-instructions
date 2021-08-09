import ipdb

import json
import numpy as np
import sys

def tok(text, ts=False):
    """
    parameters: text, token_list
    If token list is not provided default one will be used instead.
    """
    if not ts:
        ts = [',','.',';','(',')','?','!','&','%',':','*','"']

    for t in ts:
        text = text.replace(t, ' ' + t + ' ')

    return text

def tokenize():
    dets = json.load(open('./data/recipe1M/det_ingrs.json', 'r'))
    layer1 = json.load(open('./data/recipe1M/layer1.json', 'r'))

    idx2ind = {}
    ingrs = []
    for i, entry in enumerate(dets):
        idx2ind[entry['id']] = i

    text = ''
    for i, entry in enumerate(layer1):
        instrs = entry['instructions']

        allinstrs = ''
        for instr in instrs:
            instr = instr['text']
            allinstrs += instr + '\t'
        ipdb.set_trace()

        # find corresponding set of detected ingredients
        det_ingrs = dets[idx2ind[entry['id']]]['ingredients']
        valid = dets[idx2ind[entry['id']]]['valid']

        for j, det_ingr in enumerate(det_ingrs):
            # if detected ingredient matches ingredient text,
            # means it did not work. We skip
            if not valid[j]:
                continue
            # underscore ingredient

            det_ingr_undrs = det_ingr['text'].replace(' ', '_')
            ingrs.append(det_ingr_undrs)
            allinstrs = allinstrs.replace(det_ingr['text'], det_ingr_undrs)
        text += allinstrs + '\n'

    text = tok(text)

    return text

t = tokenize()
ipdb.set_trace()