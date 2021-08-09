from collections import OrderedDict

def build_dictionary(text_file_path):
    """
    Build a dictionary
    text_file_path: path for the tokenized text
    """
    with open(text_file_path, "rt") as f:
        sentences = f.readlines()  # instructions

    text = [s.replace("\n", "") for s in sentences]

    wordcount = {}
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1

    sorted_words = sorted(list(wordcount.keys()), key=lambda x: wordcount[x], reverse=True)

    word_dict = OrderedDict()
    word_dict['__unknown__'] = 0
    word_dict['<eos>'] = 1
    for idx, word in enumerate(sorted_words):
        word_dict[word] = idx + 2  # 0: <__unknown__>, 1: <eos>

    return word_dict, text
