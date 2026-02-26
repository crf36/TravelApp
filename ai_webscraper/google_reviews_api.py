import argparse
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, List

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env.local")
load_dotenv(REPO_ROOT / ".env")

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GOOGLE_REVIEWS_ENABLED = os.getenv("GOOGLE_REVIEWS_ENABLED", "1").lower() in ("1", "true", "yes")
GOOGLE_REVIEWS_MAX_PER_PLACE = int(os.getenv("GOOGLE_REVIEWS_MAX_PER_PLACE", "3"))
GOOGLE_REVIEWS_CACHE_FILE = os.getenv(
	"GOOGLE_REVIEWS_CACHE_FILE",
	str(Path(__file__).resolve().parent / "google_reviews_cache.json"),
)
GOOGLE_REVIEWS_TIMEOUT = int(os.getenv("GOOGLE_REVIEWS_TIMEOUT", "20"))
GOOGLE_REVIEWS_FORCE_REFRESH = os.getenv("GOOGLE_REVIEWS_FORCE_REFRESH", "0").lower() in ("1", "true", "yes")

TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

_CACHE_LOCK = threading.Lock()
_CACHE: Optional[Dict[str, Any]] = None


def _log(message: str) -> None:
	pass


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _normalize(text: Optional[str]) -> str:
	if not text:
		return ""
	cleaned = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
	cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
	return cleaned


def build_cache_key(name: str, place_hint: Optional[str] = None) -> str:
	key = _normalize(name)
	place = _normalize(place_hint)
	return f"{key}::{place}" if place else key


def _load_cache() -> Dict[str, Any]:
	global _CACHE
	if _CACHE is not None:
		return _CACHE

	cache_path = Path(GOOGLE_REVIEWS_CACHE_FILE)
	if cache_path.exists():
		try:
			payload = json.loads(cache_path.read_text("utf-8"))
			if isinstance(payload, dict) and isinstance(payload.get("items"), dict):
				_CACHE = payload
			else:
				_CACHE = {"updated_at": _now_iso(), "items": {}}
		except Exception:
			_CACHE = {"updated_at": _now_iso(), "items": {}}
	else:
		_CACHE = {"updated_at": _now_iso(), "items": {}}
	return _CACHE


def _save_cache() -> None:
	cache = _load_cache()
	cache["updated_at"] = _now_iso()
	cache_path = Path(GOOGLE_REVIEWS_CACHE_FILE)
	cache_path.parent.mkdir(parents=True, exist_ok=True)
	cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _text_search(query: str) -> Optional[Dict[str, Any]]:
	if not GOOGLE_MAPS_API_KEY:
		return None
	params = {
		"query": query,
		"key": GOOGLE_MAPS_API_KEY,
	}
	response = requests.get(TEXT_SEARCH_URL, params=params, timeout=GOOGLE_REVIEWS_TIMEOUT)
	if response.status_code != 200:
		return None
	body = response.json()
	results = body.get("results") or []
	if not results:
		return None
	return results[0]


def _place_details(place_id: str) -> Optional[Dict[str, Any]]:
	if not GOOGLE_MAPS_API_KEY:
		return None
	params = {
		"place_id": place_id,
		"fields": "name,rating,user_ratings_total,formatted_address,url,reviews",
		"reviews_sort": "most_relevant",
		"key": GOOGLE_MAPS_API_KEY,
	}
	response = requests.get(DETAILS_URL, params=params, timeout=GOOGLE_REVIEWS_TIMEOUT)
	if response.status_code != 200:
		return None
	body = response.json()
	return body.get("result")


def _summarize_reviews(reviews: List[Dict[str, Any]]) -> str:
	if not reviews:
		return ""
	snippets = []
	for review in reviews[:GOOGLE_REVIEWS_MAX_PER_PLACE]:
		text = (review.get("text") or "").strip()
		rating = review.get("rating")
		author = review.get("author_name") or "Reviewer"
		if not text:
			continue
		prefix = f"{author}"
		if rating is not None:
			prefix = f"{prefix} ({rating}/5)"
		snippets.append(f"{prefix}: {text}")
	return " | ".join(snippets)


def fetch_google_reviews_for_attraction(
	attraction_name: str,
	place_hint: Optional[str] = None,
	city: Optional[str] = None,
	country: Optional[str] = None,
	force_refresh: bool = False,
) -> Optional[Dict[str, Any]]:
	if not GOOGLE_REVIEWS_ENABLED:
		return None
	if not GOOGLE_MAPS_API_KEY:
		return None
	if not attraction_name:
		return None

	location_hint = place_hint or city
	cache_key = build_cache_key(attraction_name, location_hint)

	with _CACHE_LOCK:
		cache = _load_cache()
		items = cache.setdefault("items", {})
		if not force_refresh and not GOOGLE_REVIEWS_FORCE_REFRESH and cache_key in items:
			return items[cache_key]

	query_parts = [attraction_name]
	if city and city.lower() not in attraction_name.lower():
		query_parts.append(city)
	elif place_hint and place_hint.lower() not in attraction_name.lower():
		query_parts.append(place_hint)
	if country:
		query_parts.append(country)
	query = ", ".join([p for p in query_parts if p])

	try:
		search = _text_search(query)
		if not search:
			return None
		place_id = search.get("place_id")
		if not place_id:
			return None

		details = _place_details(place_id) or {}
		reviews = details.get("reviews") or []
		payload = {
			"attraction_name": attraction_name,
			"query": query,
			"google_place_id": place_id,
			"google_name": details.get("name") or search.get("name"),
			"google_formatted_address": details.get("formatted_address") or search.get("formatted_address"),
			"google_rating": details.get("rating") if details.get("rating") is not None else search.get("rating"),
			"google_user_ratings_total": details.get("user_ratings_total")
			if details.get("user_ratings_total") is not None
			else search.get("user_ratings_total"),
			"google_maps_url": details.get("url"),
			"google_reviews_summary": _summarize_reviews(reviews),
			"google_reviews_raw": reviews[:GOOGLE_REVIEWS_MAX_PER_PLACE],
			"fetched_at": _now_iso(),
		}

		with _CACHE_LOCK:
			cache = _load_cache()
			cache.setdefault("items", {})[cache_key] = payload
			_save_cache()
		return payload
	except Exception as exc:
		return None


def process_jsonl_file(jsonl_path: Path, force_refresh: bool = False) -> Dict[str, int]:
	if not jsonl_path.exists():
		raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

	total = 0
	fetched = 0
	skipped = 0

	with jsonl_path.open("r", encoding="utf-8") as handle:
		for line in handle:
			if not line.strip():
				continue
			total += 1
			try:
				article = json.loads(line)
			except json.JSONDecodeError:
				skipped += 1
				continue

			place_hint = article.get("seed_place")
			country_hint = None
			pre = article.get("pre_extracted_attractions") or []
			if pre and isinstance(pre, list):
				for item in pre:
					name = item.get("name") if isinstance(item, dict) else None
					city = item.get("detected_city") if isinstance(item, dict) else None
					if not name:
						skipped += 1
						continue
					payload = fetch_google_reviews_for_attraction(
						attraction_name=name,
						place_hint=place_hint,
						city=city,
						country=country_hint,
						force_refresh=force_refresh,
					)
					if payload:
						fetched += 1
					else:
						skipped += 1
					time.sleep(0.08)
			else:
				name = article.get("title")
				if not name:
					skipped += 1
					continue
				payload = fetch_google_reviews_for_attraction(
					attraction_name=name,
					place_hint=place_hint,
					city=place_hint,
					country=country_hint,
					force_refresh=force_refresh,
				)
				if payload:
					fetched += 1
				else:
					skipped += 1
				time.sleep(0.08)

	return {"articles": total, "fetched": fetched, "skipped": skipped}


def _resolve_default_jsonl() -> Optional[Path]:
	data_dir = Path(__file__).resolve().parent
	candidates = sorted(data_dir.glob("ta_attractions_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
	return candidates[0] if candidates else None


def main() -> None:
	parser = argparse.ArgumentParser(description="Fetch and cache Google reviews for TripAdvisor attractions.")
	parser.add_argument("--jsonl", type=str, help="Path to TripAdvisor JSONL file.")
	parser.add_argument("--force-refresh", action="store_true", help="Force refresh cached Google review entries.")
	args = parser.parse_args()

	if not GOOGLE_REVIEWS_ENABLED:
		return
	if not GOOGLE_MAPS_API_KEY:
		return

	jsonl_path = Path(args.jsonl) if args.jsonl else _resolve_default_jsonl()
	if not jsonl_path:
		return

	stats = process_jsonl_file(jsonl_path, force_refresh=args.force_refresh)


if __name__ == "__main__":
	main()