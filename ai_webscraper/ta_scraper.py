import os
import time
import json
import re
import boto3
import requests
import importlib.util
import sys
import random
from pathlib import Path
from datetime import datetime, timezone
from botocore.exceptions import ClientError

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise ImportError(
        "python-dotenv is not installed. Run `pip install python-dotenv` and re-run this script."
    ) from exc

repo_root = Path(__file__).resolve().parents[2]
load_dotenv(repo_root / ".env.local")
load_dotenv(repo_root / ".env")

API_KEY = os.getenv("TA_API_KEY")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_PREFIX = "raw_scrapes/"
TA_SEED_PLACE_ENV = os.getenv("TA_SEED_PLACE")
STATE_S3_KEY = os.getenv("TA_STATE_S3_KEY", "ta_state/ta_state.json")
OLD_STATE_S3_KEY = os.getenv("TA_OLD_STATE_S3_KEY", f"{S3_PREFIX}ta_state.json")
ATTRACTIONS_S3_KEY = os.getenv("TA_ATTRACTIONS_S3_KEY", f"{S3_PREFIX}attractions.json")
PERSIST_ATTRACTIONS = os.getenv("TA_PERSIST_ATTRACTIONS", "").lower() in ("1", "true", "yes")
MAX_NEW_ATTRACTIONS = int(os.getenv("TA_DAILY_LIMIT", "25"))
EXPORT_JSONL = os.getenv("TA_EXPORT_JSONL", "1").lower() not in ("0", "false", "no")
JSONL_PREFIX = os.getenv("TA_JSONL_PREFIX", S3_PREFIX)

if not API_KEY:
    raise RuntimeError("Set TA_API_KEY in your environment.")

DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / "ta_cache"
CACHE_DIR.mkdir(exist_ok=True)

USE_BOUNDING_BOXES = False
BOUNDING_BOXES = [
    {
        "name": "us_northeast",
        "min_lat": 38.0,
        "max_lat": 43.5,
        "min_lon": -78.0,
        "max_lon": -71.0,
        "step_deg": 1.0,
    },
]

SEED_PLACES = []
SEED_PLACES_PATH = DATA_DIR / "seed_places.json"
if SEED_PLACES_PATH.exists():
    try:
        raw_seeds = json.loads(SEED_PLACES_PATH.read_text("utf-8"))
        if isinstance(raw_seeds, list):
            SEED_PLACES = [str(p).strip() for p in raw_seeds if str(p).strip()]
    except json.JSONDecodeError:
        pass

SEED_PLACES_ENV = os.getenv("TA_SEED_PLACES")
if SEED_PLACES_ENV:
    SEED_PLACES = [p.strip() for p in SEED_PLACES_ENV.split(",") if p.strip()]

USE_INTERACTIVE_SEED = os.getenv("TA_NON_INTERACTIVE", "").lower() not in ("1", "true", "yes")
WEB_SCRAPER_PATH = repo_root / "web_scraper" / "code" / "1. main_scraper.py"
WEB_SCRAPER_LIMIT = int(os.getenv("TA_WEB_SCRAPER_LIMIT", "0"))

def get_s3_client():
    if not S3_BUCKET:
        return None
    return boto3.client("s3")

def s3_download_if_exists(s3, bucket, key, dest_path):
    try:
        s3.download_file(bucket, key, str(dest_path))
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey"):
            return False
        raise

def s3_upload_file(s3, bucket, key, source_path):
    s3.upload_file(str(source_path), bucket, key)

def load_web_scraper():
    if not WEB_SCRAPER_PATH.exists():
        return None
    try:
        scraper_dir = str(WEB_SCRAPER_PATH.parent)
        if scraper_dir not in sys.path:
            sys.path.insert(0, scraper_dir)
        spec = importlib.util.spec_from_file_location("main_scraper", WEB_SCRAPER_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        return None

def prompt_seed_place():
    if not USE_INTERACTIVE_SEED:
        return None
    if TA_SEED_PLACE_ENV:
        return TA_SEED_PLACE_ENV.strip()
    try:
        raw = input("Enter seed place (leave blank to use bounding boxes): ").strip()
    except EOFError:
        return None
    return raw or None

def maybe_run_web_scraper(place_name):
    if not place_name:
        return
    if os.getenv("TA_SKIP_WEB_SCRAPER"):
        return
    module = load_web_scraper()
    if not module:
        return
    scraper_fn = getattr(module, "scrape_and_crawl", None)
    if not scraper_fn:
        return
    scraper_fn(place_name)

def collect_web_scraper_destinations(rows, limit):
    if limit <= 0:
        return []
    candidates = []
    seen = set()
    for row in rows:
        city = (row.get("city") or "").strip()
        country = (row.get("country") or "").strip()
        seed_place = (row.get("seed_place") or "").strip()
        if seed_place.lower().startswith("bbox:"):
            seed_place = ""

        if city and country:
            dest = f"{city}, {country}"
        elif seed_place:
            dest = seed_place
        else:
            dest = city

        if not dest:
            continue
        key = dest.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(dest)

    if not candidates and SEED_PLACES:
        for place in SEED_PLACES:
            key = place.lower()
            if key not in seen:
                seen.add(key)
                candidates.append(place)

    if len(candidates) <= limit:
        return candidates

    random.shuffle(candidates)
    return candidates[:limit]

def run_web_scraper_for_new_rows(rows):
    if os.getenv("TA_SKIP_WEB_SCRAPER"):
        return
    if WEB_SCRAPER_LIMIT <= 0:
        return
    destinations = collect_web_scraper_destinations(rows, WEB_SCRAPER_LIMIT)
    if not destinations:
        return
    module = load_web_scraper()
    if not module:
        return
    scraper_fn = getattr(module, "scrape_and_crawl", None)
    if not scraper_fn:
        return
    for dest in destinations:
        scraper_fn(dest)

BASE = "https://api.content.tripadvisor.com/api/v1"
HEADERS = {"accept": "application/json"}

def ta_get(path, params=None, sleep_s=0.25, retries=3):
    params = dict(params or {})
    params["key"] = API_KEY
    url = f"{BASE}{path}"

    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(sleep_s * (2 ** i))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
            time.sleep(sleep_s)
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(sleep_s * (2 ** i))
    raise last_err

def location_search(search_query, category=None, language="en"):
    params = {"searchQuery": search_query, "language": language}
    if category:
        params["category"] = category
    j = ta_get("/location/search", params=params)
    return j.get("data", [])

def pick_best_geo_result(query, candidates):
    q = query.lower()
    scored = []
    for c in candidates:
        name = (c.get("name") or "").lower()
        loc_id = c.get("location_id")
        score = 0
        if loc_id:
            score += 5
        if name and (name in q or q in name):
            score += 10
        addr = c.get("address_obj") or {}
        if addr.get("country"):
            score += 1
        if addr.get("city"):
            score += 1
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else None

def location_details(location_id, language="en", currency="USD"):
    cache_path = CACHE_DIR / f"details_{location_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text("utf-8"))
    j = ta_get(f"/location/{location_id}/details", params={"language": language, "currency": currency})
    cache_path.write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")
    return j

def location_photos(location_id, language="en", limit=5):
    cache_path = CACHE_DIR / f"photos_{location_id}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text("utf-8"))
        return extract_photo_urls(cached, limit)
    try:
        j = ta_get(f"/location/{location_id}/photos", params={"language": language, "limit": limit})
        cache_path.write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")
        return extract_photo_urls(j, limit)
    except Exception as e:
        return []

def extract_photo_urls(photos_response, limit=5):
    urls = []
    data = photos_response.get("data", []) if isinstance(photos_response, dict) else []
    for photo in data[:limit]:
        if not isinstance(photo, dict):
            continue
        images = photo.get("images", {})
        if isinstance(images, dict):
            for size in ["original", "large", "medium", "small"]:
                size_data = images.get(size, {})
                if isinstance(size_data, dict) and size_data.get("url"):
                    urls.append(size_data["url"])
                    break
    return urls

def nearby_search(lat, lon, category="attractions", radius=10, radius_unit="mi", language="en"):
    j = ta_get("/location/nearby_search", params={
        "latLong": f"{lat},{lon}",
        "category": category,
        "radius": radius,
        "radiusUnit": radius_unit,
        "language": language
    })
    return j.get("data", [])

def iter_bbox_points(min_lat, max_lat, min_lon, max_lon, step_deg):
    lat = min_lat
    while lat <= max_lat:
        lon = min_lon
        while lon <= max_lon:
            yield lat, lon
            lon += step_deg
        lat += step_deg

def flatten_details(d, include_photos=True):
    addr = d.get("address_obj") or {}
    flattened = {
        "location_id": d.get("location_id"),
        "name": d.get("name"),
        "web_url": d.get("web_url"),
        "rating": d.get("rating"),
        "num_reviews": d.get("num_reviews"),
        "ranking_string": d.get("ranking_string"),
        "latitude": d.get("latitude"),
        "longitude": d.get("longitude"),
        "address": addr.get("address_string"),
        "city": addr.get("city"),
        "state": addr.get("state"),
        "country": addr.get("country"),
        "phone": d.get("phone"),
        "website": d.get("website"),
        "price_level": d.get("price_level"),
    }

    if include_photos:
        location_id = d.get("location_id")
        photo_count = d.get("photo_count")
        if location_id and photo_count and int(photo_count or 0) > 0:
            flattened["image_urls"] = location_photos(location_id, limit=5)
        else:
            flattened["image_urls"] = []

    return flattened

def map_price_level(raw):
    if not raw:
        return "Unknown"
    s = str(raw).strip()
    if "free" in s.lower():
        return "Free"
    currency_count = len(re.findall(r"[$€£¥]", s))
    if currency_count <= 0:
        return "Unknown"
    if currency_count == 1:
        return "Cheap"
    if currency_count == 2:
        return "Moderate"
    if currency_count == 3:
        return "Expensive"
    return "Luxury"

def build_pre_extracted_attraction(row):
    name = row.get("name") or "Unknown Attraction"
    city = row.get("city") or ""
    country = row.get("country") or ""
    detected_city = ", ".join([p for p in [city, country] if p]).strip()
    price_level = map_price_level(row.get("price_level"))
    ranking = row.get("ranking_string")
    description = f"TripAdvisor listing for {name}"
    if detected_city:
        description = f"{description} in {detected_city}"
    if ranking:
        description = f"{description}. {ranking}"
    else:
        description = f"{description}."

    return {
        "name": name,
        "detected_city": detected_city or row.get("seed_place") or "Unknown",
        "category": "Attraction",
        "vibes": [],
        "price_level": price_level,
        "popularity_keywords": [],
        "rating_score": row.get("rating"),
        "rating_max": 5.0 if row.get("rating") is not None else None,
        "review_count_mentioned": row.get("num_reviews") or 0,
        "logistics": {
            "price_text": row.get("price_level"),
            "hours": None,
            "address": row.get("address"),
            "transport": None,
        },
        "description_summary": description,
        "source_quote_or_summary": description,
    }

def build_jsonl_line(row):
    title = row.get("name") or "Unknown Attraction"
    city = row.get("city") or ""
    country = row.get("country") or ""
    address = row.get("address") or ""
    rating = row.get("rating") or ""
    reviews = row.get("num_reviews") or ""
    price = row.get("price_level") or ""
    website = row.get("website") or ""
    seed_place = row.get("seed_place") or city or ""
    image_urls = row.get("image_urls") or []

    content_body = (
        f"Attraction: {title}\n"
        f"Location: {city}, {country}\n"
        f"Address: {address}\n"
        f"Rating: {rating} (reviews: {reviews})\n"
        f"Price level: {price}\n"
        f"Website: {website}\n"
        f"Seed place: {seed_place}\n"
    ).strip()

    jsonl = {
        "source": "TripAdvisor",
        "title": title,
        "url": row.get("web_url") or website or "",
        "content_body": content_body,
        "location_id": row.get("location_id"),
        "seed_place": seed_place,
        "seed_geo_id": row.get("seed_geo_id"),
        "pre_extracted_attractions": [build_pre_extracted_attraction(row)],
    }

    if image_urls:
        jsonl["image_candidates"] = image_urls

    return jsonl

STATE_PATH = DATA_DIR / "ta_state.json"
ATTRACTIONS_PATH = DATA_DIR / "attractions.json"

s3_client = get_s3_client()
if s3_client:
    downloaded_state = s3_download_if_exists(s3_client, S3_BUCKET, STATE_S3_KEY, STATE_PATH)
    if not downloaded_state and OLD_STATE_S3_KEY:
        if s3_download_if_exists(s3_client, S3_BUCKET, OLD_STATE_S3_KEY, STATE_PATH):
            s3_upload_file(s3_client, S3_BUCKET, STATE_S3_KEY, STATE_PATH)
    if PERSIST_ATTRACTIONS:
        s3_download_if_exists(s3_client, S3_BUCKET, ATTRACTIONS_S3_KEY, ATTRACTIONS_PATH)

existing_rows = []
existing_ids = set()
if ATTRACTIONS_PATH.exists():
    with open(ATTRACTIONS_PATH, "r", encoding="utf-8") as f:
        try:
            existing_rows = json.load(f)
        except json.JSONDecodeError:
            existing_rows = []
    existing_ids = {str(r.get("location_id")) for r in existing_rows if r.get("location_id")}

state = {}
if STATE_PATH.exists():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
    except json.JSONDecodeError:
        state = {}

seen_ids = set(str(x) for x in state.get("seen_attraction_ids", [])) or existing_ids
seed_index = 0
if isinstance(state.get("seed_index"), int):
    seed_index = state["seed_index"]
elif isinstance(state.get("seed_index"), str) and state["seed_index"].isdigit():
    seed_index = int(state["seed_index"])

top_places = {}
unresolved = []
user_seed = prompt_seed_place()
if user_seed:
    USE_BOUNDING_BOXES = False
    SEED_PLACES = [user_seed]
    maybe_run_web_scraper(user_seed)

RADIUS_MILES = 10
attractions_map = {}
seed_errors = []
next_seed_index = seed_index

if USE_BOUNDING_BOXES:
    for box in BOUNDING_BOXES:
        box_name = box["name"]
        for lat, lon in iter_bbox_points(
            box["min_lat"], box["max_lat"], box["min_lon"], box["max_lon"], box["step_deg"]
        ):
            try:
                nearby = nearby_search(lat, lon, category="attractions", radius=RADIUS_MILES, radius_unit="mi")
                for item in nearby:
                    aid = item.get("location_id")
                    if not aid:
                        continue
                    if aid not in attractions_map:
                        attractions_map[aid] = {
                            "name": item.get("name"),
                            "seed_geo_id": None,
                            "seed_place": f"bbox:{box_name}",
                        }
            except Exception as e:
                seed_errors.append({"seed_geo_id": None, "seed": box_name, "error": str(e)})
else:
    if SEED_PLACES:
        seed_count = len(SEED_PLACES)
        if seed_count:
            seed_index = seed_index % seed_count
        processed = 0
        idx = seed_index
        while processed < seed_count and len(attractions_map) < MAX_NEW_ATTRACTIONS:
            place = SEED_PLACES[idx]
            try:
                candidates = location_search(place, category="geos")
                best = pick_best_geo_result(place, candidates)
                if not best or not best.get("location_id"):
                    unresolved.append({"seed": place, "reason": "no_best_match"})
                else:
                    seed_geo_id = best["location_id"]
                    addr = best.get("address_obj") or {}
                    top_places[seed_geo_id] = {
                        "seed": place,
                        "name": best.get("name"),
                        "address_string": addr.get("address_string"),
                    }

                    geo = location_details(seed_geo_id)
                    lat = geo.get("latitude")
                    lon = geo.get("longitude")
                    if lat is None or lon is None:
                        seed_errors.append({
                            "seed_geo_id": seed_geo_id,
                            "seed": place,
                            "error": "missing lat/lon"
                        })
                    else:
                        nearby = nearby_search(lat, lon, category="attractions", radius=RADIUS_MILES, radius_unit="mi")
                        for item in nearby:
                            aid = item.get("location_id")
                            if not aid:
                                continue
                            if str(aid) in seen_ids:
                                continue
                            if aid not in attractions_map:
                                attractions_map[aid] = {
                                    "name": item.get("name"),
                                    "seed_geo_id": seed_geo_id,
                                    "seed_place": place,
                                }
                                if len(attractions_map) >= MAX_NEW_ATTRACTIONS:
                                    break
            except Exception as e:
                unresolved.append({"seed": place, "reason": str(e)})

            processed += 1
            idx = (idx + 1) % seed_count

        next_seed_index = idx

new_attraction_ids = [aid for aid in attractions_map.keys() if str(aid) not in seen_ids]
if MAX_NEW_ATTRACTIONS > 0:
    new_attraction_ids = new_attraction_ids[:MAX_NEW_ATTRACTIONS]
new_attractions_map = {aid: attractions_map[aid] for aid in new_attraction_ids}

rows = []
detail_errors = []
for aid, meta in new_attractions_map.items():
    try:
        d = location_details(aid, language="en", currency="USD")
        row = flatten_details(d)
        row["seed_place"] = meta.get("seed_place")
        row["seed_geo_id"] = meta.get("seed_geo_id")
        rows.append(row)
    except Exception as e:
        detail_errors.append({"location_id": aid, "name": meta.get("name"), "error": str(e)})

new_rows = rows
merged_rows = list(existing_rows)
existing_ids = {str(r.get("location_id")) for r in merged_rows if r.get("location_id")}
for r in new_rows:
    rid = str(r.get("location_id")) if r.get("location_id") is not None else None
    if rid and rid not in existing_ids:
        merged_rows.append(r)
        existing_ids.add(rid)

with open(ATTRACTIONS_PATH, "w", encoding="utf-8") as f:
    json.dump(merged_rows, f, ensure_ascii=False, indent=2)

if detail_errors:
    errors_path = DATA_DIR / "attractions_errors.json"
    with open(errors_path, "w", encoding="utf-8") as f:
        json.dump(detail_errors, f, ensure_ascii=False, indent=2)

all_seen = set(seen_ids)
all_seen.update(str(r.get("location_id")) for r in new_rows if r.get("location_id"))
state = {
    "last_run": datetime.now(timezone.utc).isoformat(),
    "seen_attraction_ids": sorted(all_seen),
    "seed_index": next_seed_index,
}
with open(STATE_PATH, "w", encoding="utf-8") as f:
    json.dump(state, f, ensure_ascii=False, indent=2)

if s3_client:
    s3_upload_file(s3_client, S3_BUCKET, STATE_S3_KEY, STATE_PATH)
    if PERSIST_ATTRACTIONS:
        s3_upload_file(s3_client, S3_BUCKET, ATTRACTIONS_S3_KEY, ATTRACTIONS_PATH)

if EXPORT_JSONL and new_rows:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    jsonl_path = DATA_DIR / f"ta_attractions_{stamp}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps(build_jsonl_line(row), ensure_ascii=False) + "\n")
    if s3_client:
        prefix = JSONL_PREFIX if JSONL_PREFIX.endswith("/") else f"{JSONL_PREFIX}/"
        s3_key = f"{prefix}ta_attractions_{stamp}.jsonl"
        s3_upload_file(s3_client, S3_BUCKET, s3_key, jsonl_path)

    run_web_scraper_for_new_rows(new_rows)