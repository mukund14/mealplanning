# -*- coding: utf-8 -*-
"""
Pantry → Plate (AI‑Assisted)
- Rules-first matches you can actually cook
- Optional Chef AI for variations, swaps, and pantry chat
- Streamlit-native analytics (no matplotlib)

This version fixes:
- NameError for ai_variations by defining AI helpers before use
- Potential unterminated strings (clean triple-quoted blocks)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional, Tuple
import json
import numpy as np
import pandas as pd
import streamlit as st

# Optional OpenAI — used only if toggled on with a key
try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

st.set_page_config(page_title="Pantry -> Plate (AI)", page_icon="🍳", layout="wide")

# -----------------------------
# Helpers
# -----------------------------

def norm(s: str) -> str:
    return s.strip().lower().replace("_", " ").replace("-", " ")


def pretty(items: List[str]) -> str:
    return ", ".join(items) if items else "—"


def safe_bool(v) -> bool:
    return bool(v) if isinstance(v, bool) else False


def bullet(xs: List[str]) -> str:
    return "\n".join([f"• {x}" for x in xs])


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


# -----------------------------
# Data model
# -----------------------------
@dataclass
class Recipe:
    id: str
    name: str
    cuisines: List[str]
    meal_types: List[str]
    appliances: List[str]
    time_minutes: int
    diet_tags: List[str]
    required_ingredients: List[str]
    optional_ingredients: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)
    base_servings: int = 2
    flavor: Dict[str, int] = field(default_factory=lambda: {"spicy": 5, "comfort": 5, "healthy": 5, "quick": 5})

    def matches(
        self,
        have_ings: Set[str],
        have_appliances: Set[str],
        meal_filter: Set[str],
        cuisine_filter: Set[str],
        diet_filter: Set[str],
        max_time: Optional[int] = None,
        missing_allow: int = 2,
    ) -> Tuple[bool, Dict]:
        if self.appliances and not (set(map(norm, self.appliances)) & have_appliances):
            return False, {}
        if meal_filter and not (set(map(norm, self.meal_types)) & meal_filter):
            return False, {}
        if cuisine_filter and not (set(map(norm, self.cuisines)) & cuisine_filter):
            return False, {}
        if diet_filter and not set(diet_filter).issubset(set(map(norm, self.diet_tags))):
            return False, {}
        if max_time is not None and self.time_minutes > max_time:
            return False, {}
        req = set(map(norm, self.required_ingredients))
        have_norm = set(map(norm, have_ings))
        missing = sorted(list(req - have_norm))
        can_make = len(missing) == 0
        almost = (0 < len(missing) <= max(0, int(missing_allow)))
        ok = can_make or almost
        return ok, {"missing": missing, "can_make": can_make, "almost": almost}

    def score(self, have_norm: Set[str], craving_weights: Dict[str, int]) -> int:
        req_cov = len(set(map(norm, self.required_ingredients)) & have_norm)
        opt_cov = len(set(map(norm, self.optional_ingredients)) & have_norm)
        time_bonus = max(0, 60 - self.time_minutes) // 5
        base = req_cov * 10 + opt_cov * 2 + time_bonus
        cscore = 0
        for k in ["spicy", "comfort", "healthy", "quick"]:
            cscore += int(craving_weights.get(k, 5)) * int(self.flavor.get(k, 5))
        cscore //= 8
        return base + cscore


# -----------------------------
# Seed data
# -----------------------------
ALL_APPLIANCES = [
    "Stovetop (pan/pot)",
    "Oven",
    "Air Fryer",
    "Instant Pot / Pressure Cooker",
    "Slow Cooker / Crock-Pot",
    "Microwave",
    "Blender",
    "Food Processor",
    "Toaster / Toaster Oven",
    "Rice Cooker",
    "Grill / Grill Pan",
    "Sandwich Press / Panini Press",
    "Waffle Maker",
    "Bread Maker",
]

INGREDIENTS = {
    "Pantry Staples": [
        "olive oil", "vegetable oil", "butter", "ghee",
        "salt", "black pepper", "garlic", "ginger",
        "onion", "tomato", "tomato paste", "canned tomatoes",
        "soy sauce", "vinegar", "lemon", "lime",
        "flour", "cornstarch", "baking powder", "baking soda",
        "sugar", "brown sugar", "honey", "maple syrup",
        "stock/broth", "coconut milk",
    ],
    "Grains & Carbs": [
        "rice", "quinoa", "pasta", "noodles", "bread",
        "tortillas", "naan", "couscous", "oats", "potatoes",
        "sweet potatoes",
    ],
    "Proteins": [
        "eggs", "tofu", "tempeh", "paneer",
        "chicken breast", "chicken thighs", "ground chicken",
        "beef (stew)", "ground beef", "pork shoulder",
        "fish (white)", "salmon", "shrimp", "beans (canned)", "chickpeas", "lentils",
    ],
    "Vegetables": [
        "bell pepper", "carrots", "celery", "spinach",
        "kale", "broccoli", "cauliflower", "zucchini",
        "mushrooms", "peas", "corn", "green beans",
        "cabbage", "eggplant", "beets",
    ],
    "Herbs & Spices": [
        "cilantro", "parsley", "basil", "oregano", "thyme", "rosemary",
        "cumin", "turmeric", "paprika", "chili powder", "garam masala", "curry powder",
    ],
    "Dairy & Alts": [
        "milk", "yogurt", "greek yogurt", "cheddar", "mozzarella", "parmesan",
        "cream", "cream cheese", "sour cream", "feta", "plant milk",
    ],
    "Condiments & Extras": [
        "ketchup", "mustard", "mayonnaise", "sriracha", "hot sauce",
        "bbq sauce", "tahini", "peanut butter", "jam/jelly",
    ],
    "Fruits": [
        "apple", "banana", "berries", "mango", "pineapple", "avocado",
    ],
}

CUISINES = [
    "American", "Italian", "Mexican", "Indian", "Chinese", "Thai",
    "Mediterranean", "Middle Eastern", "Japanese", "Korean", "French", "Spanish",
]

MEAL_TYPES = ["breakfast", "lunch", "dinner", "snack", "dessert"]

DIET_TAGS = [
    "vegan", "vegetarian", "pescatarian", "halal", "kosher",
    "gluten-free", "dairy-free", "keto", "low-carb",
]

ALLERGENS = ["gluten", "dairy", "nuts", "peanuts", "shellfish", "soy", "egg"]

SEED_RECIPES: List[Recipe] = [
    Recipe(
        id="veg_pulao_ip",
        name="Instant Pot Veg Pulao",
        cuisines=["Indian"],
        meal_types=["lunch", "dinner"],
        appliances=["Instant Pot / Pressure Cooker", "Rice Cooker"],
        time_minutes=30,
        diet_tags=["vegetarian"],
        required_ingredients=["rice", "onion", "carrots", "peas", "cumin"],
        optional_ingredients=["garam masala", "ginger", "garlic", "tomato", "cilantro", "ghee"],
        steps=[
            "Rinse rice until water runs clear.",
            "Sauté onion, ginger, garlic in ghee/oil using Sauté mode.",
            "Add carrots, peas, spices; stir 1–2 min.",
            "Add rice and water (1:1), salt. Pressure cook 5 min; natural release 10 min.",
            "Fluff and garnish with cilantro.",
        ],
        tips=["Swap peas/carrot with broccoli/cauliflower.", "Use basmati for best texture."],
        flavor={"spicy": 4, "comfort": 7, "healthy": 6, "quick": 6},
    ),
    Recipe(
        id="airfryer_potato_wedges",
        name="Air Fryer Potato Wedges",
        cuisines=["American"],
        meal_types=["lunch", "dinner", "snack"],
        appliances=["Air Fryer", "Oven"],
        time_minutes=25,
        diet_tags=["vegan", "vegetarian", "gluten-free"],
        required_ingredients=["potatoes", "olive oil", "salt", "paprika"],
        optional_ingredients=["garlic", "oregano", "parmesan"],
        steps=[
            "Cut potatoes into wedges; toss with oil, salt, paprika (and garlic).",
            "Air fry 400°F / 200°C for 18–22 min, shaking once.",
            "Optional: toss with grated parmesan to finish.",
        ],
        tips=["No paprika? Use chili powder.", "Parboil for extra fluffy insides."],
        flavor={"spicy": 5, "comfort": 8, "healthy": 5, "quick": 7},
    ),
    Recipe(
        id="stovetop_tomato_eggs",
        name="Tomato & Eggs (Chinese-style)",
        cuisines=["Chinese"],
        meal_types=["breakfast", "lunch", "dinner"],
        appliances=["Stovetop (pan/pot)"],
        time_minutes=15,
        diet_tags=["vegetarian"],
        required_ingredients=["eggs", "tomato", "green onions", "salt", "sugar"],
        optional_ingredients=["soy sauce"],
        steps=[
            "Beat eggs with a pinch of salt.",
            "Stir-fry tomato with sugar & salt until saucy.",
            "Push aside; scramble eggs softly; combine and finish with soy sauce.",
        ],
        tips=["Great over rice or noodles.", "Add tofu cubes for more protein."],
        flavor={"spicy": 2, "comfort": 7, "healthy": 6, "quick": 9},
    ),
    Recipe(
        id="onepot_pasta",
        name="One-Pot Pantry Pasta",
        cuisines=["Italian"],
        meal_types=["lunch", "dinner"],
        appliances=["Stovetop (pan/pot)"],
        time_minutes=20,
        diet_tags=["vegetarian"],
        required_ingredients=["pasta", "tomato", "garlic", "olive oil", "salt"],
        optional_ingredients=["basil", "parmesan", "chili flakes"],
        steps=[
            "Sauté garlic in oil; add chopped tomato and salt.",
            "Add dry pasta + water to cover; simmer, stirring, until al dente.",
            "Finish with basil/parmesan.",
        ],
        tips=["Use canned tomatoes if fresh are out.", "Stir to avoid sticking."],
        flavor={"spicy": 4, "comfort": 8, "healthy": 5, "quick": 8},
    ),
    Recipe(
        id="airfryer_chickpeas",
        name="Crispy Air Fryer Chickpeas",
        cuisines=["Mediterranean"],
        meal_types=["snack", "lunch"],
        appliances=["Air Fryer"],
        time_minutes=15,
        diet_tags=["vegan", "vegetarian", "gluten-free"],
        required_ingredients=["chickpeas", "olive oil", "salt"],
        optional_ingredients=["paprika", "cumin", "garlic"],
        steps=[
            "Dry chickpeas well; toss with oil, salt, spices.",
            "Air fry 390°F / 200°C for 12–15 min, shaking basket.",
        ],
        tips=["Great on salads or soups."],
        flavor={"spicy": 5, "comfort": 6, "healthy": 8, "quick": 9},
    ),
    Recipe(
        id="baked_salmon",
        name="Sheet-Pan Lemon Salmon",
        cuisines=["American", "Mediterranean"],
        meal_types=["lunch", "dinner"],
        appliances=["Oven"],
        time_minutes=18,
        diet_tags=["pescatarian", "keto", "low-carb", "gluten-free"],
        required_ingredients=["salmon", "lemon", "olive oil", "salt"],
        optional_ingredients=["broccoli", "asparagus", "garlic"],
        steps=[
            "Place salmon & veg on sheet; drizzle oil, salt, lemon.",
            "Bake 425°F / 220°C for 12–15 min.",
        ],
        tips=["Finish with herbs for freshness."],
        flavor={"spicy": 2, "comfort": 7, "healthy": 8, "quick": 9},
    ),
]

# -----------------------------
# AI helpers (define BEFORE UI calls that use them)
# -----------------------------

def call_openai_or_fallback(api_key: Optional[str], model: str, enabled: bool, messages: List[Dict], fallback: List[str]) -> str:
    if enabled and api_key and HAS_OPENAI:
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=350,
            )
            txt = resp.choices[0].message.content or ""
            txt = txt.strip()
            if txt:
                return txt
        except Exception as e:
            return f"_AI error: {e}_\n" + bullet(fallback)
    return bullet(fallback)


def ai_variations(api_key: Optional[str], model: str, enabled: bool, recipe: Recipe, have: List[str], apps: List[str]) -> str:
    system = {"role": "system", "content": "You are Chef AI: practical, fast, realistic home cooking."}
    user = {"role": "user", "content": (
        f"""
Suggest 2 creative but realistic variations. Use mostly what's on hand.
Recipe: {recipe.name}
Have: {have}
Appliances: {apps}
Required: {recipe.required_ingredients}
Optional: {recipe.optional_ingredients}
Keep it ~{recipe.time_minutes} min. Pantry-level add-ins only. Return bullets.
"""
    )}
    return call_openai_or_fallback(api_key, model, enabled, [system, user], [
        "Add chili-lime finish and a handful of corn.",
        "Stir in herbs (cilantro/basil) and a squeeze of lemon.",
    ])


def ai_substitutions(api_key: Optional[str], model: str, enabled: bool, recipe: Recipe, missing: List[str], have: List[str]) -> str:
    system = {"role": "system", "content": "You suggest safe swaps using common pantry items."}
    user = {"role": "user", "content": (
        f"""
Suggest up to 3 substitutions for the missing items.
Missing: {missing}
Have: {have}
Recipe: {recipe.name} needs {recipe.required_ingredients}
Keep it realistic and short.
"""
    )}
    return call_openai_or_fallback(api_key, model, enabled, [system, user], [
        "No cumin → coriander or mild curry powder.",
        "Out of peas → corn or chopped green beans.",
        "No fresh tomato → canned tomatoes or tomato paste + water.",
    ])


# -----------------------------
# Smart combos
# -----------------------------

def smart_combo_recipes(have_ings: Set[str], have_apps: Set[str]) -> List[Recipe]:
    combos: List[Recipe] = []
    if {"tortillas", "beans (canned)"} <= have_ings and ("stovetop (pan/pot)" in have_apps or "toaster / toaster oven" in have_apps):
        combos.append(
            Recipe(
                id="smart_quesadilla",
                name="10-Minute Quesadilla",
                cuisines=["Mexican"],
                meal_types=["lunch", "dinner", "snack"],
                appliances=["Stovetop (pan/pot)", "Toaster / Toaster Oven"],
                time_minutes=10,
                diet_tags=["vegetarian"],
                required_ingredients=["tortillas", "beans (canned)", "cheddar"],
                optional_ingredients=["salsa", "onion", "bell pepper", "hot sauce"],
                steps=[
                    "Warm tortilla, add beans & cheese, fold.",
                    "Toast each side until melty; serve with salsa.",
                ],
                tips=["No cheese? Use hummus for creaminess."],
                flavor={"spicy": 5, "comfort": 8, "healthy": 5, "quick": 10},
            )
        )
        if {"garam masala"} & have_ings or {"curry powder"} & have_ings:
            combos.append(
                Recipe(
                    id="smart_masala_wrap",
                    name="Masala Tortilla Wrap",
                    cuisines=["Indian", "Fusion"],
                    meal_types=["lunch", "dinner"],
                    appliances=["Stovetop (pan/pot)"],
                    time_minutes=12,
                    diet_tags=["vegetarian"],
                    required_ingredients=["tortillas", "onion", "garam masala"],
                    optional_ingredients=["potatoes", "peas", "yogurt", "cilantro"],
                    steps=[
                        "Sauté onion with masala; add pre-cooked potatoes/peas if on hand.",
                        "Fill warm tortillas; optional yogurt + cilantro.",
                    ],
                    tips=["Leftover roasted veg works great here."],
                    flavor={"spicy": 6, "comfort": 8, "healthy": 6, "quick": 9},
                )
            )
    if "rice cooker" in have_apps and "rice" in have_ings:
        combos.append(
            Recipe(
                id="smart_rice_cooker_any",
                name="Rice Cooker One-Pot",
                cuisines=["Any"],
                meal_types=["lunch", "dinner"],
                appliances=["Rice Cooker"],
                time_minutes=35,
                diet_tags=[],
                required_ingredients=["rice", "salt", "oil"],
                optional_ingredients=["frozen veg", "peas", "carrots", "soy sauce", "stock/broth"],
                steps=[
                    "Rinse rice. Add 1:1 water, salt, a splash of oil.",
                    "Top with frozen/chopped veg; cook normally.",
                    "Fluff; season with soy or herbs.",
                ],
                tips=["Fold in leftover protein at the end."],
                flavor={"spicy": 0, "comfort": 7, "healthy": 7, "quick": 6},
            )
        )
    return combos


# -----------------------------
# Session state
# -----------------------------
if "recipes" not in st.session_state:
    st.session_state["recipes"] = {r.id: r for r in SEED_RECIPES}
if "pantry" not in st.session_state:
    st.session_state["pantry"] = set()


# -----------------------------
# Header
# -----------------------------
st.title("🍳 Pantry → Plate")
st.subheader("Rules that never lie. AI that makes it fun.")

with st.container():
    c1, c2 = st.columns([0.62, 0.38])
    with c1:
        st.markdown(
            """
**How it works**  
1) Pick your **appliances** and **ingredients**.  
2) Rules suggest meals you can actually cook.  
3) **Chef AI Studio** adds twists, swaps, and pantry chat.
"""
        )
    with c2:
        st.success(
            "**Chef AI Studio is optional**: toggle it on when you want creative riffs, smart substitutions, or a quick pantry chat.",
            icon="🤖",
        )

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Your Kitchen")
    appliances = st.multiselect("Appliances", ALL_APPLIANCES, default=["Stovetop (pan/pot)"])
    max_time = st.slider("Max cook time (min)", 5, 240, 45, step=5)
    servings = st.number_input("Servings", 1, 12, 2)

    st.header("Diet & Allergies")
    diet = st.multiselect("Diet tags", DIET_TAGS, default=[])
    allergens = st.multiselect("Allergies (filter ingredients)", ALLERGENS, default=[])

    st.header("Meal & Cuisine")
    meal_types = st.multiselect("Meal type(s)", MEAL_TYPES, default=["breakfast", "lunch", "dinner"])
    cuisines = st.multiselect("Preferred cuisines", CUISINES, default=[])

    st.header("Flavor sliders (bias sorting)")
    a, b = st.columns(2)
    with a:
        spicy = st.slider("Spicy", 0, 10, 5)
        comfort = st.slider("Comfort", 0, 10, 5)
    with b:
        healthy = st.slider("Healthy", 0, 10, 5)
        quick = st.slider("Quick", 0, 10, 7)
    cravings = {"spicy": spicy, "comfort": comfort, "healthy": healthy, "quick": quick}

    st.header("Matching Options")
    missing_allow = st.slider("Allowed missing items (for 'Almost There')", 0, 5, 2)

    st.header("AI Companion")
    ai_enabled = st.toggle("Enable Chef AI Studio (OpenAI)", value=False)
    openai_key = None
    openai_model = "gpt-4o-mini"
    if ai_enabled:
        openai_key = st.text_input("OpenAI API Key", type="password", help="sk-…")
        openai_model = st.text_input("Model", value="gpt-4o-mini")
        if not HAS_OPENAI:
            st.info("Install the OpenAI SDK to enable AI: `pip install openai`")

# -----------------------------
# Ingredients UI
# -----------------------------
st.subheader("🧺 Ingredients on hand")
with st.expander("Presets"):
    cA, cB, cC = st.columns(3)
    with cA:
        if st.button("Basic Pantry"):
            st.session_state["pantry"].update(map(norm, [
                "salt", "black pepper", "olive oil", "garlic", "onion", "tomato", "rice", "eggs"
            ]))
        if st.button("Vegan Essentials"):
            st.session_state["pantry"].update(map(norm, [
                "olive oil", "salt", "black pepper", "garlic", "onion", "tomato",
                "rice", "beans (canned)", "lentils", "chickpeas", "broccoli", "carrots", "peas"
            ]))
    with cB:
        if st.button("Italian Night"):
            st.session_state["pantry"].update(map(norm, [
                "pasta", "tomato", "garlic", "basil", "parmesan", "olive oil"
            ]))
        if st.button("Indian Basics"):
            st.session_state["pantry"].update(map(norm, [
                "rice", "onion", "tomato", "ginger", "garlic", "cumin", "turmeric", "garam masala", "peas", "carrots"
            ]))
    with cC:
        if st.button("Breakfast Fixings"):
            st.session_state["pantry"].update(map(norm, [
                "eggs", "bread", "cheddar", "milk", "spinach", "bell pepper"
            ]))
        if st.button("Sheet-Pan Dinner"):
            st.session_state["pantry"].update(map(norm, [
                "salmon", "broccoli", "lemon", "olive oil"
            ]))

selected = set(st.session_state["pantry"])
for cat, items in INGREDIENTS.items():
    defaults = [i for i in items if norm(i) in selected]
    chosen = st.multiselect(cat, items, default=defaults, key=f"ms_{cat}")
    selected |= set(map(norm, chosen))

row = st.columns([0.6, 0.2, 0.2])
with row[0]:
    custom_add = st.text_input("Add a custom ingredient (press Enter)", value="")
    if custom_add.strip():
        selected.add(norm(custom_add))
        st.session_state["pantry"] = selected
        st.success(f"Added: {custom_add}")
with row[1]:
    if st.button("Clear selected"):
        st.session_state["pantry"] = set(); selected = set()
with row[2]:
    if st.button("Save selection"):
        st.session_state["pantry"] = selected
        st.success("Saved for this session.")

# Allergen filter

def filter_allergens(ingredients: Set[str], allergens_selected: Set[str]) -> Set[str]:
    allergen_map = {
        "gluten": {"bread", "tortillas", "naan", "pasta", "flour"},
        "dairy": {"milk", "yogurt", "greek yogurt", "cheddar", "mozzarella", "parmesan", "cream", "cream cheese", "sour cream", "feta", "butter", "ghee"},
        "nuts": {"almonds", "walnuts", "cashews", "pecans"},
        "peanuts": {"peanut butter", "peanuts"},
        "shellfish": {"shrimp"},
        "soy": {"soy sauce", "tofu"},
        "egg": {"eggs"},
    }
    filtered = set(ingredients)
    for a in map(norm, allergens_selected):
        filtered -= allergen_map.get(a, set())
    return filtered

selected = filter_allergens(set(map(norm, selected)), set(map(norm, allergens)))

st.divider()

# -----------------------------
# Add your own recipe
# -----------------------------
with st.expander("➕ Add your own recipe"):
    st.caption("Add a personal recipe and it will be matched like any other.")
    with st.form("add_recipe_form", clear_on_submit=True):
        r_name = st.text_input("Recipe name")
        r_id = norm(r_name).replace(" ", "_")
        r_cuis = st.multiselect("Cuisines", CUISINES)
        r_meals = st.multiselect("Meal types", MEAL_TYPES)
        r_apps = st.multiselect("Appliances", ALL_APPLIANCES)
        r_time = st.number_input("Time (minutes)", 1, 600, 20)
        r_diet = st.multiselect("Diet tags", DIET_TAGS)
        r_req = st.text_area("Required ingredients (comma-separated)")
        r_opt = st.text_area("Optional ingredients (comma-separated)")
        r_steps = st.text_area("Steps (one per line)")
        r_tips = st.text_area("Tips (one per line)")
        ok = st.form_submit_button("Add recipe")
        if ok and r_name and r_req:
            rec = Recipe(
                id=r_id or f"user_{len(st.session_state['recipes'])+1}",
                name=r_name,
                cuisines=r_cuis,
                meal_types=r_meals,
                appliances=r_apps,
                time_minutes=int(r_time),
                diet_tags=list(map(norm, r_diet)),
                required_ingredients=[norm(x) for x in r_req.split(",") if x.strip()],
                optional_ingredients=[norm(x) for x in r_opt.split(",") if x.strip()],
                steps=[s.strip() for s in r_steps.splitlines() if s.strip()],
                tips=[t.strip() for t in r_tips.splitlines() if t.strip()],
            )
            st.session_state["recipes"][rec.id] = rec
            st.success(f"Added: {r_name}")

# -----------------------------
# Matching
# -----------------------------

have_apps = set(map(norm, appliances))
meal_filter = set(map(norm, meal_types))
cuisine_filter = set(map(norm, cuisines))
diet_filter = set(map(norm, diet))

recipes_all: Dict[str, Recipe] = dict(st.session_state["recipes"])
for sr in smart_combo_recipes(selected, have_apps):
    recipes_all[sr.id] = sr

candidates: List[Tuple[Recipe, Dict]] = []
for rec in recipes_all.values():
    ok, meta = rec.matches(
        have_ings=selected,
        have_appliances=have_apps,
        meal_filter=meal_filter,
        cuisine_filter=cuisine_filter,
        diet_filter=diet_filter,
        max_time=max_time,
        missing_allow=missing_allow,
    )
    if ok:
        candidates.append((rec, meta))

have_norm = set(selected)


def sort_key(t: Tuple[Recipe, Dict]) -> int:
    r, _ = t
    return r.score(have_norm, cravings)

make_now = sorted([x for x in candidates if x[1].get("can_make")], key=sort_key, reverse=True)
almost = sorted([x for x in candidates if not safe_bool(x[1].get("can_make"))], key=sort_key, reverse=True)

# -----------------------------
# Tabs
# -----------------------------

t1, t2, t3, t4 = st.tabs([
    f"✅ Matches ({len(make_now)} now / {len(almost)} almost)",
    "🤖 Chef AI Studio",
    "📊 Pantry Analytics",
    "⬇️ Export & Save",
])

# ----- Tab 1: Matches -----

def render_recipe_card(r: Recipe, meta: Dict):
    can_make = meta.get("can_make")
    badge = "✅ Make now" if can_make else "🟡 Almost there"
    missing = meta.get("missing", [])

    with st.container():
        st.markdown(f"### {r.name}  {badge}")
        st.caption(f"{'/'.join(r.cuisines)} • {', '.join(r.meal_types)} • {r.time_minutes} min")
        colL, colR = st.columns([0.62, 0.38])
        with colL:
            st.write("**Needs:** " + pretty(r.required_ingredients))
            if r.optional_ingredients:
                st.write("**Nice-to-have:** " + pretty(r.optional_ingredients))
            if missing:
                st.error("Missing: " + pretty(missing))
            if r.steps:
                with st.expander("Steps"):
                    for i, step in enumerate(r.steps, 1):
                        st.write(f"{i}. {step}")
            if r.tips:
                st.caption("Tips: " + " • ".join(r.tips))
        with colR:
            if st.button("🎨 Variations", key=f"ai_var_{r.id}"):
                text = ai_variations(openai_key, openai_model, ai_enabled, r, list(sorted(have_norm)), list(sorted(have_apps)))
                st.session_state[f"ai_{r.id}"] = text
            if st.button("🔀 Substitutions", key=f"ai_sub_{r.id}"):
                text = ai_substitutions(openai_key, openai_model, ai_enabled, r, missing, list(sorted(have_norm)))
                st.session_state[f"ai_{r.id}"] = text
            out = st.session_state.get(f"ai_{r.id}")
            if out:
                st.markdown("**Chef AI says:**")
                st.write(out)
        st.divider()

with t1:
    if not make_now and not almost:
        st.info("No matches yet — try a preset, widen meal types, or allow 1–2 missing items.")
    for rec, meta in make_now:
        render_recipe_card(rec, meta)
    for rec, meta in almost:
        render_recipe_card(rec, meta)

# ----- Tab 2: Chef AI Studio -----
with t2:
    st.markdown("#### Pantry Chat")
    st.caption("Tell Chef AI what you have and what you crave. Get 3 ideas in under a minute.")
    colA, colB = st.columns([0.7, 0.3])
    with colA:
        pantry_text = st.text_area("What's in your kitchen?", placeholder="onion, tomato, rice, eggs …")
        vibe = st.select_slider("Vibe", options=["quick", "healthy", "comfort", "spicy"], value="quick")
        btn = st.button("Ask Chef AI")
    with colB:
        st.info("AI is optional. The rules engine already works without it.")
        st.write("Model:", openai_model if ai_enabled else "—")
    if btn:
        system = {"role": "system", "content": "You are Chef AI. Offer simple, make-tonight ideas."}
        user = {"role": "user", "content": (
            f"""
Ingredients: {pantry_text}
Appliances: {sorted(list(have_apps))}
Mood: {vibe}
Suggest 3 dishes under 30 min with 1-line directions each.
"""
        )}
        ideas = call_openai_or_fallback(openai_key, openai_model, ai_enabled, [system, user], [
            "Veg fried rice — use leftover rice, egg, soy; 10–12 min.",
            "Masala omelet wrap — eggs + onions + tortilla; 8–10 min.",
            "Tomato garlic pasta — one-pan simmer; 15–20 min.",
        ])
        st.markdown("**Ideas**")
        st.write(ideas)

# ----- Tab 3: Analytics -----
with t3:
    st.markdown("### Pantry Coverage & Match Quality")

    rows = []
    for r, meta in make_now + almost:
        req = set(map(norm, r.required_ingredients))
        opt = set(map(norm, r.optional_ingredients))
        have_req = len(req & have_norm)
        have_opt = len(opt & have_norm)
        miss = meta.get("missing", [])
        rows.append({
            "recipe": r.name,
            "can_make": meta.get("can_make", False),
            "time_min": r.time_minutes,
            "req_have": have_req,
            "req_total": len(req),
            "opt_have": have_opt,
            "opt_total": len(opt),
            "score": r.score(have_norm, cravings),
            "jaccard_req": jaccard(req, have_norm),
            "missing": ", ".join(miss),
        })

    if rows:
        df = pd.DataFrame(rows).sort_values("score", ascending=False)
        st.dataframe(df, use_container_width=True)

        counts, edges = np.histogram(df["score"].values, bins=10)
        centers = (edges[:-1] + edges[1:]) / 2
        hist_df = pd.DataFrame({"bin": centers, "count": counts}).set_index("bin")
        st.markdown("**Recipe Score Distribution**")
        st.bar_chart(hist_df)

        cov = (df["req_have"] / df["req_total"]).fillna(0)
        counts2, edges2 = np.histogram(cov.values, bins=10, range=(0, 1))
        centers2 = (edges2[:-1] + edges2[1:]) / 2
        cov_df = pd.DataFrame({"coverage": centers2, "count": counts2}).set_index("coverage")
        st.markdown("**Required Ingredient Coverage**")
        st.bar_chart(cov_df)

        st.markdown("#### What-if: add one ingredient")
        missing_pool: List[str] = []
        for _, meta in almost:
            missing_pool.extend(meta.get("missing", []))
        cand = pd.Series([norm(x) for x in missing_pool]).value_counts().head(15)
        if not cand.empty:
            pick = st.selectbox("Pick an ingredient to simulate adding", list(cand.index))
            if pick:
                new_make_now = 0
                for r, meta in almost:
                    missing = set(map(norm, meta.get("missing", [])))
                    if missing == {pick} or (pick in missing and len(missing) <= max(1, missing_allow)):
                        new_make_now += 1
                st.info(f"If you add **{pick}**, you could instantly unlock **{new_make_now}** close matches.")

            st.markdown("**Top missing ingredients across 'Almost There'**")
            top_df = pd.DataFrame({"ingredient": cand.index, "count": cand.values}).set_index("ingredient")
            st.bar_chart(top_df)

        st.download_button(
            "Download analytics CSV", data=df.to_csv(index=False), file_name="pantry_analytics.csv", mime="text/csv"
        )
    else:
        st.info("Add ingredients and see matches to view analytics.")

# ----- Tab 4: Export -----
with t4:
    st.markdown("### Results export")

    def candidates_meta(r: Recipe) -> List[str]:
        for rr, meta in make_now + almost:
            if rr.id == r.id:
                return meta.get("missing", [])
        return []

    def export_md(recs: List[Tuple[Recipe, Dict]], servings: int) -> str:
        lines = ["# Pantry -> Plate — Results\n"]
        for r, meta in recs:
            badge = "Make now" if meta.get("can_make") else "Almost there"
            lines += [
                f"## {r.name} ({badge})",
                f"- Cuisines: {', '.join(r.cuisines)}",
                f"- Meals: {', '.join(r.meal_types)}",
                f"- Time: {r.time_minutes} min",
                f"- Servings: {servings}",
                f"- Needs: {pretty(r.required_ingredients)}",
            ]
            miss = candidates_meta(r)
            if r.optional_ingredients:
                lines.append(f"- Nice-to-have: {pretty(r.optional_ingredients)}")
            if miss:
                lines.append(f"- Missing: {pretty(miss)}")
            if r.steps:
                lines.append("\n**Steps**")
                for i, step in enumerate(r.steps, 1):
                    lines.append(f"{i}. {step}")
            if r.tips:
                lines.append("\n**Tips**")
                for t in r.tips:
                    lines.append(f"- {t}")
            lines.append("\n---\n")
        return "\n".join(lines)

    if candidates:
        md_blob = export_md(make_now + almost, servings)
        st.download_button(
            "Download results (Markdown)", data=md_blob, file_name="pantry_to_plate_results.md", mime="text/markdown"
        )
    else:
        st.caption("Once you have matches, export appears here.")

    st.markdown("### Save / Load data")
    colX, colY = st.columns(2)
    with colX:
        if st.button("💾 Prepare JSON for download"):
            payload = {
                "pantry": sorted(list(st.session_state["pantry"])),
                "recipes": [asdict(r) for r in st.session_state["recipes"].values()],
            }
            st.download_button(
                "Download pantry_to_plate_data.json",
                data=json.dumps(payload, indent=2),
                file_name="pantry_to_plate_data.json",
                mime="application/json",
            )
    with colY:
        up = st.file_uploader("Load JSON", type=["json"], label_visibility="collapsed")
        if up is not None:
            try:
                data = json.loads(up.read().decode("utf-8"))
                st.session_state["pantry"] = set(map(norm, data.get("pantry", [])))
                recs = {}
                for rd in data.get("recipes", []):
                    recs[rd["id"]] = Recipe(**rd)
                if recs:
                    st.session_state["recipes"] = recs
                st.success("Loaded pantry & recipes from file.")
            except Exception as e:
                st.error(f"Failed to load: {e}")

# -----------------------------
# Developer sanity tests
# -----------------------------
with st.expander("🔧 Developer Checks (sanity tests)"):
    try:
        r = SEED_RECIPES[0]
        s0 = r.score(set(), {"spicy":5,"comfort":5,"healthy":5,"quick":5})
        s1 = r.score({norm(r.required_ingredients[0])}, {"spicy":5,"comfort":5,"healthy":5,"quick":5})
        st.write("Score overlap test:", "PASS" if s1 > s0 else "FAIL")

        ok1, _ = r.matches(set(r.required_ingredients), set(map(norm, r.appliances)), set(), set(), set(), max_time=60)
        ok2, _ = r.matches(set(r.required_ingredients), {"microwave"}, set(), set(), set(), max_time=60)
        st.write("Appliance filter test:", "PASS" if ok1 and not ok2 else "FAIL")

        ok3, _ = r.matches(set(r.required_ingredients), set(map(norm, r.appliances)), set(), set(), set(), max_time=5)
        st.write("Time filter test:", "PASS" if not ok3 else "FAIL")

        fb = ai_variations(None, "gpt-4o-mini", False, r, list(r.required_ingredients), list(r.appliances))
        st.write("AI fallback test:", "PASS" if "•" in fb else "FAIL")
    except Exception as e:
        st.warning(f"Sanity tests encountered an issue: {e}")

# -----------------------------
# Footer — partners
# -----------------------------
st.divider()
with st.expander("About & Partners"):
    st.markdown(
        """
**Riverside fm** — I record the podcast here. Their AI editor trims filler words and tightens the cut. Get started with Riverside here: https://riverside.sjv.io/jeAW5n 
**RSS com** — Hosting + distribution with a **paid program** so small shows can earn (even with low download counts).  Sign up here: https://rss.com/?via=13bda4.
**Sider ai** — My research buddy for collecting sources fast.Try Sider here: https://vidlineinc.pxf.io/dOJd4M.

*Some links may be affiliates and could support my work at no extra cost to you.*
"""
    )
