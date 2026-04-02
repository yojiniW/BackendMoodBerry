from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
import json
import httpx
from typing import Optional, List
from utils.preprocess import preprocess_text
import os
from dotenv import load_dotenv

# ===============================
# Configuration
# ===============================
load_dotenv()
USDA_API_KEY        = os.getenv("MY_API_KEY")
USDA_BASE_URL       = "https://api.nal.usda.gov/fdc/v1"
USDA_PAGE_SIZE      = 50

MODEL_PATH          = "model/limited_emotions_bilstm_model.keras"
TOKENIZER_PATH      = "model/tokenizer.pkl"
LABEL_ENCODER_PATH  = "model/label_encoder.pkl"
KNOWLEDGE_BASE_PATH = "knowledge_base.json"

NUTRIENT_SEARCH_QUERIES: dict[str, dict[str, str]] = {

    # Omega-3 Fatty Acids (1404)
    "1404": {
        "omnivore":    "salmon mackerel sardine tuna flaxseed walnut omega-3",
        "pescatarian": "salmon mackerel sardine tuna herring omega-3",
        "vegetarian":  "flaxseed walnuts chia seeds hemp seeds omega-3",
        "vegan":       "flaxseed walnuts chia seeds hemp seeds seaweed omega-3",
    },

    # Vitamin D (1114)
    "1114": {
        "omnivore":    "salmon tuna egg milk fortified vitamin D",
        "pescatarian": "salmon tuna mackerel fortified milk vitamin D",
        "vegetarian":  "egg milk fortified cereal mushroom vitamin D",
        "vegan":       "fortified plant milk mushroom UV vitamin D",
    },

    # Magnesium (1090)
    "1090": {
        "omnivore":    "spinach pumpkin seeds almonds dark chocolate magnesium",
        "pescatarian": "spinach pumpkin seeds almonds dark chocolate magnesium",
        "vegetarian":  "spinach pumpkin seeds almonds dark chocolate avocado magnesium",
        "vegan":       "spinach pumpkin seeds chia seeds dark chocolate avocado magnesium",
    },

    # Zinc (1095)
    "1095": {
        "omnivore":    "beef oyster pumpkin seeds lentils zinc",
        "pescatarian": "oyster pumpkin seeds lentils chickpeas zinc",
        "vegetarian":  "pumpkin seeds lentils chickpeas hemp seeds zinc",
        "vegan":       "pumpkin seeds lentils chickpeas hemp seeds tofu zinc",
    },

    # Vitamin B12 (1178)
    "1178": {
        "omnivore":    "beef liver clams sardines eggs vitamin B12",
        "pescatarian": "clams sardines eggs fortified milk vitamin B12",
        "vegetarian":  "eggs milk cheese fortified cereal vitamin B12",
        "vegan":       "fortified plant milk nutritional yeast fortified cereal vitamin B12",
    },

    # Selenium (1103)
    "1103": {
        "omnivore":    "brazil nuts tuna halibut sardines selenium",
        "pescatarian": "brazil nuts tuna halibut sardines selenium",
        "vegetarian":  "brazil nuts eggs sunflower seeds mushroom selenium",
        "vegan":       "brazil nuts sunflower seeds mushroom brown rice selenium",
    },

    # Vitamin B6 (1175)
    "1175": {
        "omnivore":    "chicken tuna potato banana vitamin B6",
        "pescatarian": "tuna salmon potato banana vitamin B6",
        "vegetarian":  "potato banana avocado sunflower seeds vitamin B6",
        "vegan":       "potato banana avocado sunflower seeds chickpeas vitamin B6",
    },

    # Vitamin C (1162)
    "1162": {
        "omnivore":    "bell pepper kiwi broccoli orange vitamin C",
        "pescatarian": "bell pepper kiwi broccoli orange vitamin C",
        "vegetarian":  "bell pepper kiwi broccoli orange strawberry vitamin C",
        "vegan":       "bell pepper kiwi broccoli orange strawberry guava vitamin C",
    },

    # Calcium (1087)
    "1087": {
        "omnivore":    "milk cheese yogurt tofu calcium",
        "pescatarian": "milk cheese yogurt tofu sardines calcium",
        "vegetarian":  "milk cheese yogurt tofu fortified calcium",
        "vegan":       "tofu fortified plant milk chia seeds kale calcium",
    },

    # Potassium (1093)
    "1093": {
        "omnivore":    "spinach banana avocado potato potassium",
        "pescatarian": "spinach banana avocado potato potassium",
        "vegetarian":  "spinach banana avocado potato sweet potato potassium",
        "vegan":       "spinach banana avocado potato lentils potassium",
    },

    # Iron (1089)
    "1089": {
        "omnivore":    "beef liver spinach lentils iron",
        "pescatarian": "spinach lentils tofu pumpkin seeds iron",
        "vegetarian":  "spinach lentils tofu pumpkin seeds fortified iron",
        "vegan":       "spinach lentils tofu pumpkin seeds quinoa iron",
    },

    # Glutamine (1223)
    "1223": {
        "omnivore":    "chicken breast beef eggs tofu glutamine protein",
        "pescatarian": "salmon eggs tofu edamame glutamine protein",
        "vegetarian":  "eggs tofu edamame cottage cheese glutamine protein",
        "vegan":       "tofu edamame lentils cabbage beet glutamine protein",
    },

    # Tryptophan (1210)
    "1210": {
        "omnivore":    "turkey chicken eggs milk tryptophan",
        "pescatarian": "salmon tuna eggs milk tryptophan",
        "vegetarian":  "eggs milk cheese pumpkin seeds tryptophan",
        "vegan":       "pumpkin seeds tofu soybeans oats tryptophan",
    },

    # Folate / Vitamin B9 (1177)
    "1177": {
        "omnivore":    "spinach lentils asparagus edamame folate folic acid",
        "pescatarian": "spinach lentils asparagus edamame folate folic acid",
        "vegetarian":  "spinach lentils asparagus avocado folate folic acid",
        "vegan":       "spinach lentils asparagus avocado broccoli folate folic acid",
    },

    # Tyrosine (1214)  
    "1214": {
        "omnivore":    "chicken breast beef eggs pumpkin seeds tyrosine protein",
        "pescatarian": "salmon tuna eggs pumpkin seeds tyrosine protein",
        "vegetarian":  "eggs cheese pumpkin seeds soybeans tyrosine protein",
        "vegan":       "tofu edamame pumpkin seeds soybeans oats tyrosine protein",
    },
}

# Food names containing these keywords will be excluded from results regardless of diet
BLOCKED_NAME_KEYWORDS = [
    "energy drink", "monster", "red bull", "rockstar", "5-hour",
    "supplement", "protein powder", "protein shake",
    "fruit-flavored drink", "drink powder",
    "fish oil",        
    "cod liver oil",
]


DIET_BLOCKED_NAME_KEYWORDS: dict[str, list[str]] = {
    "vegetarian": [
        "salmon", "tuna", "mackerel", "sardine", "cod", "tilapia", "herring",
        "trout", "anchovy", "halibut", "swordfish", "bass", "snapper", "catfish",
        "fish oil", "cod liver", "shrimp", "crab", "lobster", "prawn", "oyster",
        "scallop", "mussel", "clam", "squid", "octopus",
        "chicken", "turkey", "duck", "goose", "quail",
        "beef", "pork", "bacon", "ham", "lamb", "veal", "venison", "bison",
        "hot dog", "sausage", "pepperoni", "salami",
    ],
    "vegan": [
        "salmon", "tuna", "mackerel", "sardine", "cod", "tilapia", "herring",
        "trout", "anchovy", "halibut", "fish oil", "cod liver",
        "shrimp", "crab", "lobster", "prawn", "oyster", "scallop", "mussel", "clam",
        "chicken", "turkey", "duck", "beef", "pork", "bacon", "ham", "lamb", "veal",
        "hot dog", "sausage", "pepperoni", "salami",
        "milk", "cheese", "yogurt", "butter", "cream", "whey", "casein", "lactose",
        "egg",
    ],
    "pescatarian": [
        "chicken", "turkey", "duck", "goose", "quail",
        "beef", "pork", "bacon", "ham", "lamb", "veal", "venison", "bison",
        "hot dog", "sausage", "pepperoni", "salami",
    ],
    "omnivore": [],
}

# ===============================
# Load Model Artifacts
# ===============================

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# ===============================
# FastAPI App
# ===============================

app = FastAPI(
    title="Emotion-Aware Food Recommendation API",
    description="BiLSTM emotion detection with USDA-powered food recommendations and XAI explanations.",
    version="7.0.0"
)

# ===============================
# Request Models
# ===============================

class TextRequest(BaseModel):
    text: str
    diet_type: Optional[str] = "omnivore"
    allergies: Optional[List[str]] = []

# ===============================
# Diet Filter Data
# ===============================

DIET_EXCLUDED_CATEGORIES = {
    "vegetarian":  ["Poultry Products", "Beef Products", "Pork Products", "Finfish and Shellfish Products"],
    "vegan":       ["Poultry Products", "Beef Products", "Pork Products", "Finfish and Shellfish Products", "Dairy and Egg Products"],
    "pescatarian": ["Poultry Products", "Beef Products", "Pork Products", "Lamb, Veal, and Game Products"],
    "omnivore":    []
}

# ===============================
# USDA Fetch & Filter Logic
# ===============================

def knowledge_base_nutrient_name(nutrient_id: str) -> str:
    """Fallback: find nutrient name from knowledge base by USDA ID."""
    for emotion in knowledge_base["emotions"]:
        for n in emotion.get("nutrients", []):
            if str(n.get("usda_nutrient_id")) == str(nutrient_id):
                return n["name"]
    return "healthy food"


async def fetch_usda_foods(
    nutrient_id: str,
    diet_type: str = "omnivore",
    page_size: int = USDA_PAGE_SIZE
) -> List[dict]:
    """
    Searches USDA FoodData Central using a diet-aware, nutrient-specific keyword query.

    FIX: Now accepts diet_type and selects the appropriate query branch so that
    vegetarian/vegan/pescatarian users receive diet-appropriate food candidates
    before any filtering is applied. Previously a single query (e.g. fish-heavy
    for omega-3) dominated the 50-result pool, leaving few or no valid foods
    after the category filter removed meat/fish for plant-based diets.
    """
    query_entry = NUTRIENT_SEARCH_QUERIES.get(str(nutrient_id))

    if isinstance(query_entry, dict):
        # Diet-aware query: pick the branch for the current diet, fall back to omnivore
        query = query_entry.get(diet_type) or query_entry.get("omnivore") or ""
    elif isinstance(query_entry, str):
        # Legacy plain-string fallback (backward compat)
        query = query_entry
    else:
        # Nutrient not in dict at all — use its name from the knowledge base
        query = knowledge_base_nutrient_name(nutrient_id)

    url = f"{USDA_BASE_URL}/foods/search"
    params = {
        "query":    query,
        "dataType": "Foundation,SR Legacy",
        "pageSize": page_size,
        "api_key":  USDA_API_KEY,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, params=params)
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data.get("foods", [])


def is_food_allowed(
    food: dict,
    diet_type: str,
    allergies: List[str],
    allergen_map: dict,
    allergen_keywords: dict
) -> bool:
    """
    Returns True if the food passes all filters:
      1. Global blocked-name keywords (supplements, energy drinks, fish oil, etc.)
      2. USDA category-based diet filter
      3. Diet-based name keyword blocklist  ← FIX: catches fish oil / misc items
         that slip past the category filter due to unexpected USDA categorisation
      4. Allergen filter (category + keyword)
    """
    category        = food.get("foodCategory", "") or ""
    food_name_lower = food.get("description", "").lower()

    # 1. Global blocked names (supplements, energy drinks, fish oil, etc.)
    if any(kw in food_name_lower for kw in BLOCKED_NAME_KEYWORDS):
        return False

    # 2. Category-based diet filter
    excluded_cats = DIET_EXCLUDED_CATEGORIES.get(diet_type, [])
    if any(exc.lower() in category.lower() for exc in excluded_cats):
        return False

    # 3. Diet-based name keyword blocklist
    #    Handles cases like fish oil (USDA category: "Fats and Oils") which pass
    #    the category filter above but should still be excluded for vegetarians/vegans.
    diet_blocked = DIET_BLOCKED_NAME_KEYWORDS.get(diet_type, [])
    if any(kw in food_name_lower for kw in diet_blocked):
        return False

    # 4. Allergen filter
    if allergies:
        # Category-based allergen check
        for cat_key, cat_allergens in allergen_map.items():
            if cat_key.lower() in category.lower():
                for allergen in cat_allergens:
                    if allergen in allergies:
                        return False

        # Keyword-based allergen check (scan food name)
        for allergen in allergies:
            keywords = allergen_keywords.get(allergen, [])
            if any(kw in food_name_lower for kw in keywords):
                return False

    return True


def get_nutrient_value_from_food(food: dict, nutrient_id: str) -> Optional[float]:
    """
    Extracts the amount of a specific nutrient from a USDA food object.
    Handles all field name variations across USDA API endpoints:
      - nutrientNumber (search endpoint string)
      - number (detail endpoint)
      - nutrientId (numeric ID, search endpoint)
    """
    for fn in food.get("foodNutrients", []):
        nnum = str(fn.get("nutrientNumber", fn.get("number", "")))
        nid  = str(fn.get("nutrientId", ""))
        if nnum == str(nutrient_id) or nid == str(nutrient_id):
            return fn.get("value") or fn.get("amount")
    return None


def score_food(food: dict, nutrient_id: str, min_amount: float, priority: int) -> float:
    """
    Scores a food using the knowledge base formula:
      score = (amount_per_100g / min_amount_per_100g) * (1 / priority)
    Returns 0.0 if the nutrient is not present in the food.
    """
    amount = get_nutrient_value_from_food(food, nutrient_id)
    if amount is None or amount == 0 or min_amount <= 0:
        return 0.0
    return (amount / min_amount) * (1 / priority)


def format_food(food: dict, nutrient: dict) -> dict:
    """
    Converts a raw USDA food object into the shape ResultsScreen expects.
    with fields: user_message, psychological_explanation, biological_explanation,
    references.
    """
    food_name   = food.get("description", "Unknown Food")
    category    = food.get("foodCategory", "")
    nutrient_id = str(nutrient["usda_nutrient_id"])

    calories      = get_nutrient_value_from_food(food, "208")   # Energy kcal
    protein       = get_nutrient_value_from_food(food, "203")   # Protein g
    target_amount = get_nutrient_value_from_food(food, nutrient_id)

    calories_str = f"{round(calories)} kcal" if calories is not None else "—"
    protein_str  = f"{round(protein, 1)}g"   if protein  is not None else "—"
    unit         = nutrient.get("unit", "")
    nutrient_str = (
        f"{round(target_amount, 2)}{unit} per 100g"
        if target_amount is not None else "—"
    )

    # Derive allergens from food category —  key: allergen_filter > category_map
    allergen_map   = knowledge_base["metadata"]["allergen_filter"]["category_map"]
    food_allergens = []
    for cat_key, cat_allergens in allergen_map.items():
        if cat_key.lower() in category.lower():
            food_allergens.extend(cat_allergens)
    food_allergens = list(set(food_allergens))

    
    # field names: psychological_explanation, biological_explanation (not psychological/biological)
    xai = nutrient.get("xai", {})

    return {
        "food_name":                 food_name,
        "usda_fdc_id":               food.get("fdcId"),
        "food_category":             category,
        "calories":                  calories_str,
        "protein":                   protein_str,
        "serving_size":              "100g",
        "nutrient_amount":           nutrient_str,
        "allergens":                 food_allergens,
        "psychological_explanation": xai.get("psychological_explanation", ""),
        "biological_explanation":    xai.get("biological_explanation", ""),
        "references":                xai.get("references", []),
        "xai_user_message":          xai.get("user_message", ""),
    }


async def fetch_foods_for_nutrient(
    nutrient: dict,
    diet_type: str,
    allergies: List[str],
    top_n: int = 5
) -> List[dict]:
   
    #  key names
    allergen_map      = knowledge_base["metadata"]["allergen_filter"]["category_map"]
    allergen_keywords = knowledge_base["metadata"]["allergen_filter"]["keyword_map"]

    nutrient_id = str(nutrient["usda_nutrient_id"])
    min_amount  = nutrient.get("min_amount_per_100g", 1)
    priority    = nutrient.get("priority", 1)

    # FIX: pass diet_type so the correct query branch is used
    raw_foods = await fetch_usda_foods(nutrient_id, diet_type=diet_type)

    allowed = [
        f for f in raw_foods
        if is_food_allowed(f, diet_type, allergies, allergen_map, allergen_keywords)
    ]

    scored = sorted(
        allowed,
        key=lambda f: score_food(f, nutrient_id, min_amount, priority),
        reverse=True
    )

    return [format_food(f, nutrient) for f in scored[:top_n]]


# ===============================
# Knowledge Base Helpers
# ===============================

def get_emotion_data(emotion_id: str) -> Optional[dict]:
    for emotion in knowledge_base["emotions"]:
        if emotion["emotion_id"].lower() == emotion_id.lower():
            return emotion
    return None


def format_supportive_compounds(emotion_data: dict) -> List[dict]:
    """
    FIX: Supportive compounds (Polyphenols, Flavonoids, Probiotics, Adaptogens,
    L-Theanine, etc.) are bioactive compounds that are NOT tracked in the USDA
    database. They were previously silently dropped because the /predict endpoint
    only fetched USDA foods for nutrients[].

    These are now returned as informational items with their food sources,
    mechanism, and reference so the Flutter app can display them in a separate
    'Also Consider' section without needing a USDA lookup.
    """
    return [
        {
            "name":      sc.get("name", ""),
            "type":      sc.get("type", ""),
            "found_in":  sc.get("found_in", []),
            "mechanism": sc.get("mechanism", ""),
            "reference": sc.get("reference", ""),
            "note":      sc.get("note", ""),
        }
        for sc in emotion_data.get("supportive_compounds", [])
    ]


# ===============================
# Routes
# ===============================

@app.get("/")
def root():
    return {
        "message":            "Emotion-Aware Food Recommendation API is running",
        "total_emotions":     len(knowledge_base["emotions"]),
        "emotions_supported": [e["emotion_id"] for e in knowledge_base["emotions"]]
    }


@app.get("/emotions")
def list_emotions():
    return {
        "emotions": [
            {
                "emotion_id":   e["emotion_id"],
                "emotion_name": e["emotion_name"],
                "emoji":        e["emoji"],
                "color_hex":    e["color_hex"],
                "valence":      e["valence"],
                "goal":         e["goal"],
                "app_message":  e["app_message"]
            }
            for e in knowledge_base["emotions"]
        ]
    }


@app.get("/emotion/{emotion_id}")
def get_emotion(emotion_id: str):
    emotion_data = get_emotion_data(emotion_id)
    if not emotion_data:
        raise HTTPException(
            status_code=404,
            detail=f"Emotion '{emotion_id}' not found. Supported: "
                   f"{[e['emotion_id'] for e in knowledge_base['emotions']]}"
        )
    return emotion_data


@app.post("/predict")
async def predict_emotion(request: TextRequest):
    """
    Main endpoint. Detects emotion from text, fetches matching foods
    from USDA FoodData Central, filters by diet and allergies, ranks
    by nutrient density, and returns a complete response for ResultsScreen.

    Each nutrient in the response contains:
    - name, usda_nutrient_id, unit, priority, classification
    - xai (full explanation block from knowledge base)
    - foods: list of top 5 foods with food_name, calories, protein,
             serving_size, nutrient_amount, allergens,
             psychological_explanation, biological_explanation,
             references, xai_user_message

    supportive_compounds in the response contains informational items for
    bioactive compounds (L-Theanine, Polyphenols, Probiotics, Adaptogens, etc.)
    that are not in the USDA database. Each item includes found_in food sources,
    mechanism, and a research reference for display in the Flutter UI.
    """

    # 1. Preprocess and predict
    processed_text = preprocess_text(request.text, tokenizer)
    probs    = model.predict(processed_text, verbose=0)
    pred_idx = int(np.argmax(probs, axis=1)[0])

    emotion_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence    = float(np.max(probs))

    # 2. Look up emotion in knowledge base
    emotion_data = get_emotion_data(emotion_label.lower())
    if not emotion_data:
        raise HTTPException(
            status_code=404,
            detail=f"Detected emotion '{emotion_label}' not found in knowledge base."
        )

    # 3. Validate diet_type
    valid_diets = ["omnivore", "vegetarian", "vegan", "pescatarian"]
    diet_type   = request.diet_type.lower() if request.diet_type else "omnivore"
    if diet_type not in valid_diets:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid diet_type '{diet_type}'. Must be one of: {valid_diets}"
        )

    allergies = request.allergies or []

    # 4. Fetch USDA foods for each nutrient (diet-aware queries + name blocklist)
    nutrients_with_foods = []
    for nutrient in emotion_data.get("nutrients", []):
        foods = await fetch_foods_for_nutrient(
            nutrient  = nutrient,
            diet_type = diet_type,
            allergies = allergies,
            top_n     = 5
        )
        nutrients_with_foods.append({
            "name":             nutrient["name"],
            "usda_nutrient_id": nutrient["usda_nutrient_id"],
            "unit":             nutrient.get("unit", ""),
            "priority":         nutrient.get("priority", 1),
            "classification":   nutrient.get("classification", ""),
            # Pass the full xai block so Flutter can access all fields if needed
            "xai_explanation":  nutrient.get("xai", {}),
            "foods":            foods
        })

    # 5. Format supportive compounds (no USDA lookup — informational only)
    supportive_compounds_output = format_supportive_compounds(emotion_data)

    # 6. Return full response matching ResultsScreen expectations
    return {
        "emotion":               emotion_label,
        "confidence":            round(confidence, 2),
        "valence":               emotion_data.get("valence", ""),
        "goal":                  emotion_data.get("goal", ""),
        "brain_state":           emotion_data.get("brain_state", ""),
        "app_message":           emotion_data.get("app_message", ""),
        "avoid":                 emotion_data.get("avoid", []),
        "food_preference_style": emotion_data.get("food_preference_style"),
        "supportive_compounds":  supportive_compounds_output,
        "diet_notes":            emotion_data.get("diet_notes", {}),
        "active_diet_note":      emotion_data.get("diet_notes", {}).get(diet_type, ""),
        "user_diet_type":        diet_type,
        "user_allergies":        allergies,
        "nutrients":             nutrients_with_foods,
        "disclaimer":            knowledge_base["metadata"]["disclaimer"]
    }


@app.get("/metadata")
def get_metadata():
    return knowledge_base["metadata"]