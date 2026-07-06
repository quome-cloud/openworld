"""CLEAN-protocol text parser for the BALROG TextWorld observation channel.

Input: exactly the objective-stripped feedback text the BALROG wrapper serves to agents.
The grammar patterns below encode the TextWorld surface grammar (the 'observation format'),
not per-game instance data; they were developed and verified offline against the dev corpus
(walkthrough replays of all 75 games) with the .json specs as ground truth.
"""
import re

DIRS = ("north", "south", "east", "west")
STATE_ADJ = ("raw", "fried", "grilled", "roasted", "burned", "sliced", "chopped", "diced", "uncut")
KEY_NOUNS = ("key", "keycard", "latchkey", "passkey")
# container noun lexicon (TextWorld house-theme containers)
CONTAINER_NOUNS = (
    "fridge", "refrigerator", "toolbox", "safe", "box", "chest", "drawer", "case", "display",
    "cabinet", "dresser", "trunk", "basket", "portmanteau", "suitcase", "locker", "coffer",
    "crate", "cubby", "bureau", "chest of drawers", "wardrobe", "cupboard", "freezer",
)
APPLIANCES = {"BBQ": "grilled", "oven": "roasted", "stove": "fried"}


def strip_junk(text):
    """Remove z8 status-line artifacts, prompts, banner remains and end-screen."""
    if not text:
        return ""
    out = []
    for line in text.splitlines():
        if re.search(r"-=\s*.+?\s*=-\s*\d+/\d+\s*$", line):
            continue  # z8 status bar
        if line.strip().startswith(">"):
            continue
        if "*** The End ***" in line:
            break
        if "RESTART, RESTORE" in line or re.match(r"^You scored ", line.strip()):
            continue
        out.append(line)
    return "\n".join(out)


def base_name(disp):
    """Strip leading state adjectives: 'raw white tuna' -> 'white tuna'."""
    words = disp.split()
    while words and words[0] in STATE_ADJ:
        words = words[1:]
    return " ".join(words)


def split_list(s):
    """'a A, a B and a C' -> ['A','B','C'] (state adjectives stripped)."""
    s = s.strip().rstrip(".!")
    parts = re.split(r",\s*|\s+and\s+", s)
    out = []
    for p in parts:
        p = p.strip()
        p = re.sub(r"^(a|an|the|some)\s+", "", p)
        if p and p not in ("nothing",):
            out.append(base_name(p))
    return out


def parse(text):
    """Parse one observation into structured events + room snapshot."""
    text = strip_junk(text)
    r = {
        "room": None, "exits": {}, "floor": [], "on": {}, "contains": {},
        "containers": {}, "sightings": set(), "appliances": set(),
        "events": [], "recipe": None, "raw": text,
    }

    # room header: take the LAST one (feedback can precede a room description)
    headers = list(re.finditer(r"-= (.+?) =-", text))
    block = text
    if headers:
        h = headers[-1]
        r["room"] = h.group(1).strip().lower()
        block = text[h.end():]

    # ---------- feedback events (whole text) ----------
    for m in re.finditer(r"You have to unlock (?:the )?(.+?) with (?:the )?(.+?) first", text):
        r["events"].append(("locked", m.group(1).strip(), m.group(2).strip()))
    for m in re.finditer(r"You have to open (?:the )?(.+?) first", text):
        r["events"].append(("must_open", m.group(1).strip()))
    for m in re.finditer(r"You open (?:the )?(.+?), revealing (.+?)[.!]", text):
        r["events"].append(("revealed", m.group(1).strip(), split_list(m.group(2))))
    for m in re.finditer(r"You open (?!the .*?, revealing)(?:the )?([^,.\n]+?)\.", text):
        r["events"].append(("opened", m.group(1).strip()))
    for m in re.finditer(r"You unlock (?:the )?([^,.\n]+?)\.", text):
        r["events"].append(("unlocked", m.group(1).strip()))
    for m in re.finditer(r"You pick up (?:the )?(.+?) from the ground", text):
        r["events"].append(("took", base_name(m.group(1).strip())))
    for m in re.finditer(r"You take (?:the )?(.+?) from (?:the )?([^,.\n]+?)\.", text):
        r["events"].append(("took", base_name(m.group(1).strip())))
    if "You can't see any such thing" in text:
        r["events"].append(("cant_see",))
    if re.search(r"do(?:es)?n(?:'|’)t (?:seem to )?fit|does not fit", text):
        r["events"].append(("wrong_key",))
    if "Which do you mean" in text:
        r["events"].append(("which_mean",))
    if "That's already open" in text:
        r["events"].append(("already_open",))
    if re.search(r"You (?:can't|cannot) go that way", text):
        r["events"].append(("cant_go",))
    for m in re.finditer(r"You (grilled|fried|roasted) (?:the )?(.+?)\.", text):
        r["events"].append(("cooked", m.group(1), base_name(m.group(2).strip())))
    for m in re.finditer(r"You (slice|chop|dice) (?:the )?(.+?)\.", text):
        r["events"].append(("cut", m.group(1), base_name(m.group(2).strip())))
    if "Adding the meal to your inventory" in text:
        r["events"].append(("prepared",))
    if re.search(r"You eat the meal", text):
        r["events"].append(("eaten",))
    if "You are carrying nothing" in text:
        r["events"].append(("carrying", []))
    elif "You are carrying" in text:
        seg = text.split("You are carrying")[1].lstrip(": \n")
        items = []
        for ln in seg.splitlines():
            ln = ln.strip().rstrip(".,")
            m = re.match(r"(?:an?|some|the)\s+(.+)$", ln)
            if m:
                items.append(base_name(m.group(1)))
            elif ln and items:
                break
        r["events"].append(("carrying", items))

    # ---------- recipe ----------
    if "Ingredients:" in text and "Directions:" in text:
        ing_block = text.split("Ingredients:")[1].split("Directions:")[0]
        dir_block = text.split("Directions:")[1]
        ingredients = [ln.strip() for ln in ing_block.splitlines() if ln.strip()]
        cuts, cooks = {}, {}
        for m in re.finditer(r"(slice|chop|dice) the (.+)", dir_block):
            cuts[m.group(2).strip()] = m.group(1)
        for m in re.finditer(r"(fry|grill|roast) the (.+)", dir_block):
            cooks[m.group(2).strip()] = m.group(1)
        r["recipe"] = {"ingredients": ingredients, "cuts": cuts, "cooks": cooks}

    # ---------- room snapshot (block after last header, parsed per sentence) ----------
    if r["room"] is not None:
        cont_last = sorted({c.split()[-1] for c in CONTAINER_NOUNS})
        cont_re = re.compile(r"\b(?:a|an|the) ((?:[A-Za-z'-]+\s+){0,4}(?:%s))\b" % "|".join(cont_last))
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", block) if s.strip()]
        for s in sentences:
            # doors on exits
            m = re.search(r"There is an? (open|closed|locked) (.+?) leading (north|south|east|west)", s)
            if m:
                r["exits"][m.group(3)] = {"door": m.group(2).strip(), "door_state": m.group(1)}
                continue
            # plain exits (several phrasings; can share a sentence with other text)
            for m in re.finditer(r"exit to the (north|south|east|west)", s):
                r["exits"].setdefault(m.group(1), {"door": None, "door_state": None})
            for m in re.finditer(r"try going (north|south|east|west)", s):
                r["exits"].setdefault(m.group(1), {"door": None, "door_state": None})
            # floor items
            m = re.match(r"There(?:'s| is) (.+?) on the floor\.?$", s) or re.match(r"You (?:can )?see (.+?) on the floor\.?$", s)
            if m:
                r["floor"] += split_list(m.group(1))
                continue
            # supporter contents
            m = re.match(r"On the (.+?) you (?:can )?(?:make out|see) (.+?)\.?$", s)
            if m:
                r["on"].setdefault(m.group(1).strip(), []).extend(split_list(m.group(2)))
                continue
            m = re.match(r"You (?:can )?see (.+?) on the (.+?)\.?$", s)
            if m and m.group(2).strip() != "floor":
                r["on"].setdefault(m.group(2).strip(), []).extend(split_list(m.group(1)))
                continue
            # container contents (open container re-described)
            m = re.search(r"The (.+?) contains (.+?)\.?$", s)
            if m:
                r["contains"][m.group(1).strip()] = split_list(m.group(2))
                r["containers"][m.group(1).strip()] = "open"
                continue
            # container sightings with explicit state (not doors: no 'leading' in sentence)
            if " leading " not in s:
                for m in re.finditer(r"\b(open(?:ed)?|closed|locked) ([\w' -]+?)(?=[.,]|$| is\b| in\b| on\b| at\b| near\b| here\b| close\b| nearby\b)", s):
                    st = "open" if m.group(1).startswith("open") else m.group(1)
                    words = m.group(2).strip().split()
                    # truncate trailing flavor text at the first container noun
                    idx = next((i for i, wd in enumerate(words) if wd in cont_last), None)
                    name = " ".join(words[: idx + 1]) if idx is not None else m.group(2).strip()
                    r["containers"].setdefault(name, st)
                # stateless container sightings via noun lexicon
                for m in cont_re.finditer(s):
                    name = m.group(1).strip()
                    if not any(name.startswith(st + " ") for st in ("open", "opened", "closed", "locked")):
                        r["sightings"].add(name)
        # appliances by token over the whole block
        for app in APPLIANCES:
            if re.search(rf"\b{app}\b", block):
                r["appliances"].add(app)
    return r
