# Pantry → Plate (AI‑Assisted)

**An open‑source Streamlit app that turns your pantry into dinner.**

* **Rules first** for reliable, make‑tonight suggestions
* **Chef AI Studio** for creative twists, safe substitutions, and a friendly pantry chat
* **Built‑in analytics** to quantify coverage, scores, and “what‑if I add one ingredient?”

This repo is designed to be **plain‑English, human‑readable**, and easy to extend. No heavy deps. No matplotlib. Just Streamlit + Pandas + Numpy. Optional OpenAI.

---

## ✨ Features

* **Multi‑select inputs** for appliances, ingredients, cuisines, meal types, diets, and allergies
* **Deterministic rules engine** that never hallucinates tools or ingredients you don’t have
* **Chef AI Studio (optional)**

  * *Variations*: 1‑click creative riffs that stay realistic
  * *Substitutions*: safe, pantry‑level swaps when you’re missing something
  * *Pantry Chat*: freeform prompt → 3 ideas with 1‑line directions
* **📊 Pantry Analytics** (Streamlit‑native charts)

  * Score histogram
  * Required‑ingredient coverage histogram
  * **What‑if simulator**: which single ingredient unlocks the most “Make Now” dishes
* **Import/Export**: save your pantry + recipes to JSON
* **Add your own recipes** via UI (stored in session)

---

## 🏎 Quickstart

> Requires **Python 3.10+**

```bash
# 1) Clone
git clone https://github.com/<you>/pantry-to-plate.git
cd pantry-to-plate

# 2) Create a virtual env (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install deps
pip install streamlit pandas numpy openai  # openai is optional

# 4) Run
streamlit run meals.py  # or the filename you used
```

> **No matplotlib needed.** Charts use Streamlit’s `st.bar_chart` to avoid extra installs.

---

## 🔐 (Optional) Enable Chef AI Studio

You can use the app **fully offline**. To enable AI:

* Either paste a key into the sidebar field when you toggle **Enable Chef AI Studio (OpenAI)**
* Or set an env var before launch:

```bash
export OPENAI_API_KEY=sk-your-key
```

Default model in the UI: `gpt-4o-mini` (editable). The app gracefully **falls back** to helpful bullet suggestions if the SDK/key isn’t available.

---

## 🧠 How It Works (Rules + AI)

### Rules Engine (deterministic)

1. **Normalize** all strings (case/format) for reliable matching
2. Appliance filter: require at least one intersection
3. Meal/cuisine/diet filters
4. Time filter (≤ your selected max)
5. Ingredient coverage:

   * **Make Now** → zero missing required ingredients
   * **Almost There** → missing required ingredients ≤ `missing_allow`

### Scoring (for sorting)

```text
score = 10 * (# required overlap)
      +  2 * (# optional overlap)
      + floor((60 - time_minutes) / 5)  # time bonus
      + tiny flavor bias from sliders (spicy/comfort/healthy/quick)
```

### Chef AI Studio (optional)

* **Variations**: realistic riffs that prefer what you already have
* **Substitutions**: short, safe swaps for missing items
* **Pantry Chat**: 3 ideas in under a minute

If the SDK/key is missing or rate‑limited, the app returns sensible **fallback bullets** so you never hit a dead end.

---

## 🗂 Data Model

`Recipe` (Python `@dataclass`):

* `id`, `name`
* `cuisines: List[str]`
* `meal_types: List[str]` (breakfast | lunch | dinner | snack | dessert)
* `appliances: List[str]`
* `time_minutes: int`
* `diet_tags: List[str]` (vegan | vegetarian | pescatarian | halal | kosher | gluten‑free | dairy‑free | keto | low‑carb)
* `required_ingredients: List[str]`
* `optional_ingredients: List[str]`
* `steps: List[str]`, `tips: List[str]`, `base_servings: int`
* `flavor: Dict[str, int]` (spicy/comfort/healthy/quick; default 5..10 scale)

---

## 📦 Import / Export

### Export

* **Markdown** summary (Recipes → steps/tips) via the **Export & Save** tab
* **Analytics CSV** (scores, coverage)
* **JSON** bundle: pantry + all recipes

### JSON Format

```json
{
  "pantry": ["rice", "onion", "garlic"],
  "recipes": [
    {
      "id": "veg_pulao_ip",
      "name": "Instant Pot Veg Pulao",
      "cuisines": ["Indian"],
      "meal_types": ["lunch", "dinner"],
      "appliances": ["Instant Pot / Pressure Cooker", "Rice Cooker"],
      "time_minutes": 30,
      "diet_tags": ["vegetarian"],
      "required_ingredients": ["rice", "onion", "carrots", "peas", "cumin"],
      "optional_ingredients": ["garam masala", "ginger", "garlic", "tomato", "cilantro", "ghee"],
      "steps": ["Rinse rice…", "Sauté onion…"],
      "tips": ["Swap peas…"],
      "base_servings": 2,
      "flavor": {"spicy": 4, "comfort": 7, "healthy": 6, "quick": 6}
    }
  ]
}
```

---

## 🧪 Tests & Developer Checks

Open the **“🔧 Developer Checks (sanity tests)”** expander in the app. It reports **PASS/FAIL** on:

1. Score increases as required‑ingredient overlap increases
2. Appliance filter blocks mismatched tools
3. Time filter blocks over‑limit recipes
4. AI helpers return fallback output when AI is disabled

> Want more tests? Add property tests for matching and scoring (e.g., Hypothesis) or snapshot tests on the exported Markdown.

---

## 🛠 Troubleshooting

* **`ModuleNotFoundError: No module named 'matplotlib'`**

  * This project deliberately **does not** use matplotlib. Use Streamlit’s built‑ins (`st.bar_chart`). If you pasted code from elsewhere, remove `matplotlib` imports.
* **`NameError: name 'ai_variations' is not defined'`**

  * Ensure AI helper functions are **defined before** any UI code that calls them. In this repo, they appear above the Matches render.
* **OpenAI errors / rate limits**

  * You’ll always get fallback bullets so the UI stays usable even without AI.

---

## 🚀 Deploy

### Streamlit Community Cloud

1. Push this repo to GitHub
2. Create an app on Streamlit Cloud → point to `meals.py`
3. (Optional) Add `OPENAI_API_KEY` as a secret if you want Chef AI by default

### Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir streamlit pandas numpy openai
EXPOSE 8501
CMD ["streamlit", "run", "meals.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Run:

```bash
docker build -t pantry-to-plate .
docker run -p 8501:8501 pantry-to-plate
```

---

## 🗺 Roadmap

* Nutrition estimates (simple kcal/macros per serving)
* Multi‑language ingredient normalization
* Saved profiles (cloud sync) and household presets
* Vector search over a larger recipe set
* Unit conversions and smart scaling (by servings)

---

## 🤝 Contributing

PRs welcome! Good first issues:

* Add seed recipes w/ realistic steps and tips
* Expand allergen maps and diet filters
* Improve analytics (e.g., unlock curves vs. pantry size)
* Add tests for corner cases (e.g., no appliances selected)

---

## 📣 Credits & Partners

**RSS**: your podcast, get free transcripts, and earn ad revenue with as few as 10 monthly downloads. Sign up here: https://rss.com/?via=13bda4.
**Sider AI**. AI-powered research and productivity assistant for breaking down job descriptions into keywords. Try Sider here: https://vidlineinc.pxf.io/dOJd4M.
**Riverside FM**: Record your podcast in studio-quality audio and 4K video from anywhere. Get started with Riverside here: https://riverside.sjv.io/jeAW5n.

*Some links may be affiliate links—supporting the project at no extra cost to you.*

---

## 📄 License

Choose a license (MIT/Apache‑2.0). If you’re not sure, MIT is a simple, permissive default. Add a `LICENSE` file at the repo root.

---

## 🔎 SEO Hints for GitHub

* Include terms like **“AI recipe generator”, “pantry meal planner”, “Streamlit app”, “data science cooking analytics”** in the description
* Add topics: `streamlit`, `openai`, `recipes`, `meal-planner`, `ai`, `data-science`
* Add a short demo GIF to the README header
