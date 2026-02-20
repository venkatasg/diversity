from typing import List, Tuple, Any, Set, Optional

import stanza
from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline


def _detect_language(data: List[str], sample_size: int = 10) -> str:
    """ Detects the dominant language of a corpus using stanza's multilingual langid.

    Args:
        data (List[str]): Corpus texts to detect language from.
        sample_size (int): Number of texts to sample for detection. Defaults to 10.

    Returns:
        str: ISO 639-1 language code (e.g. 'en', 'fr', 'de'). Defaults to 'en' on failure.
    """
    sample = data[:sample_size] if len(data) > sample_size else data

    stanza.download('multilingual', verbose=False)
    langid_pipeline = Pipeline(lang='multilingual', processors='langid', verbose=False)

    docs = [Document([], text=text) for text in sample]
    langid_pipeline(docs)

    lang_counts: dict = {}
    for doc in docs:
        lang = doc.lang
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    return max(lang_counts, key=lang_counts.get)


def get_pos(
        data: List[str],
        lang: Optional[str] = None,
) -> Tuple[List[str], List[List[Tuple[str, str]]]]:
    """ Turns a sequence into parts of speech using stanza.

    The language of the corpus is auto-detected when *lang* is not supplied.
    Detection uses stanza's multilingual langid model on the first few texts
    and defaults to English ('en') when detection produces no result.

    Args:
        data (List[str]): Data to transform into part-of-speech tags.
        lang (str, optional): BCP-47 / ISO 639-1 language code (e.g. 'en').
            When None the language is detected automatically.

    Returns:
        Tuple[List[str], List[List[Tuple[str, str]]]]:
            - List of space-separated POS tag strings, one per input text.
            - List of lists of (token, POS-tag) tuples, one list per input text.
    """
    if lang is None:
        lang = _detect_language(data)

    stanza.download(lang, verbose=False)
    nlp = stanza.Pipeline(lang=lang, processors='tokenize,pos', verbose=False)

    in_docs = [Document([], text=text) for text in data]
    nlp(in_docs)

    joined_pos: List[str] = []
    pos_tuples: List[List[Tuple[str, str]]] = []

    for doc in in_docs:
        tokens_and_tags: List[Tuple[str, str]] = []
        for sent in doc.sentences:
            for word in sent.words:
                # xpos gives treebank-specific tags (e.g. Penn Treebank for English)
                # which match the tag style previously returned by spaCy's token.tag_
                tag = word.xpos if word.xpos is not None else word.upos
                tokens_and_tags.append((word.text, tag))
        pos_tuples.append(tokens_and_tags)
        joined_pos.append(' '.join(tag for _, tag in tokens_and_tags))

    return joined_pos, pos_tuples


def _find_sub_list(
        sl: List[Any],
        l: List[Any]
) -> List[Any]:
    """ Given a pattern and a list of strings, returns sublists matching the pattern. """

    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind+sll] == sl:
            results.append((ind, ind+sll-1))

    return results


def pos_patterns(
        text: List[List[Tuple[str, str]]],
        pattern: str
) -> Set[str]:
    """ Finds all substrings matching a part of speech pattern.

    Args:
        text (List[List[Tuple[str, str]]]): Text containing words and part-of-speech tags.
        pattern (str): Part-of-speech tag pattern to search for.

    Returns:
        Set[str]: Returns all the strings matching the pattern.
    """

    pos = []
    word = []

    # text is a list of lists of tuples (word, part of speech)
    for doc in text:
        pos.append([i[1] for i in doc])
        word.append([i[0] for i in doc])

    pos = [' '.join(x) for x in pos]
    word = [' '.join(x) for x in word]

    all_matches = []

    # return positions of each tag and the corresponding tokens
    for w, p in zip(word, pos):

        test = _find_sub_list(pattern.split(), p.split())

        if test:
            for occ in test:
                splits = w.split()[int(occ[0]):int(occ[1]+1)]
                all_matches.append(" ".join(splits))

    return set(all_matches)
