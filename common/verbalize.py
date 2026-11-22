"""Triple -> natural-language claim. All generator templates live here (HANDOFF.md section 6).

The claim parser (services/claim_parser) deliberately does NOT import these templates, so
evaluation is not a trivial round trip. Relation names below that do not occur in a dataset are
simply unused; unlisted relations fall back to a generic sentence.
"""
import random
from typing import Optional

FB_TEMPLATES: dict[str, list[str]] = {
    "/people/person/place_of_birth": ["{h} was born in {t}.", "{t} is the birthplace of {h}."],
    "/people/person/nationality": ["{h} is a citizen of {t}.", "{h} holds {t} nationality."],
    "/people/person/gender": ["The gender of {h} is {t}."],
    "/people/person/profession": ["{h} works as a {t}.", "The profession of {h} is {t}."],
    "/people/person/languages": ["{h} speaks {t}."],
    "/people/person/religion": ["The religion of {h} is {t}."],
    "/film/film/genre": ["{h} is a {t} film.", "The genre of {h} is {t}."],
    "/film/film/country": ["{h} was produced in {t}."],
    "/film/film/language": ["The language of {h} is {t}."],
    "/location/location/contains": ["{h} contains {t}.", "{t} is located within {h}."],
    "/location/country/capital": ["The capital of {h} is {t}."],
    "/location/country/form_of_government": ["{h} has a {t} form of government."],
    "/music/artist/genre": ["{h} plays {t} music."],
}

WN_TEMPLATES: dict[str, list[str]] = {
    "_hypernym": ["{h} is a kind of {t}.", "{t} is a broader term for {h}."],
    "_instance_hypernym": ["{h} is an instance of {t}."],
    "_derivationally_related_form": ["{h} is derivationally related to {t}."],
    "_has_part": ["{h} has {t} as a part."],
    "_member_meronym": ["{t} is a member of {h}."],
    "_synset_domain_topic_of": ["{h} belongs to the domain of {t}."],
    "_also_see": ["{h} is also related to {t}."],
    "_verb_group": ["{h} belongs to the same verb group as {t}."],
    "_similar_to": ["{h} is similar to {t}."],
    "_member_of_domain_region": ["{h} is a term used in the region of {t}."],
    "_member_of_domain_usage": ["{h} is a term of the {t} usage domain."],
}

TEMPLATES = {**FB_TEMPLATES, **WN_TEMPLATES}


def humanize_relation(relation: str) -> str:
    """'/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor' -> 'actor'; '_hypernym' -> 'hypernym'."""
    return relation.split("/")[-1].strip("_").replace("_", " ")


def verbalize(head_label: str, relation: str, tail_label: str,
              rng: Optional[random.Random] = None) -> str:
    rng = rng or random.Random(0)
    options = TEMPLATES.get(relation)
    if options:
        sentence = rng.choice(options).format(h=head_label, t=tail_label)
    else:
        sentence = f"The {humanize_relation(relation)} of {head_label} is {tail_label}."
    return sentence[0].upper() + sentence[1:]
