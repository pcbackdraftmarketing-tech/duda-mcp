"""
Duda MCP Server
===============
Exposes Duda website-management operations as MCP tools via FastMCP.

Required environment variables (set in a .env file or the process environment):
    DUDA_API_USER         — your Duda API username
    DUDA_API_PASS         — your Duda API password
    ANTHROPIC_API_KEY     — (optional) enables LLM-based template variant selection
    ANTHROPIC_MODEL       — (optional) model id for variant selection
                            (default: 'claude-sonnet-4-6')
    DUDA_TEMPLATES_FILE   — (optional) path to the template registry YAML.
                            Defaults to 'templates.yaml' next to this file.
                            Edit that file to add templates and call the
                            `reload_custom_templates` MCP tool to pick up
                            changes without restarting the server.

Usage:
    python main.py
"""

import base64
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Literal, Optional

import anthropic
import requests
import yaml
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

# ---------------------------------------------------------------------------
# Credentials & shared HTTP session
# ---------------------------------------------------------------------------

USERNAME = os.environ.get("DUDA_API_USER")
PASSWORD = os.environ.get("DUDA_API_PASS")

if not USERNAME or not PASSWORD:
    raise EnvironmentError(
        "DUDA_API_USER and DUDA_API_PASS must be set in the environment or a .env file."
    )

creds = base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()

session = requests.Session()
# POST-REVIEW FIX: dropped the session-level 'Content-Type: application/json' —
# requests sets it automatically when json=... is passed, and sending it on
# bodyless GETs was technically incorrect.
session.headers.update({"Authorization": f"Basic {creds}"})

# POST-REVIEW FIX: retry on transient 429/5xx with exponential backoff so a
# single hiccup on Duda's side doesn't break a whole build workflow.
_retry = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 502, 503, 504],
    allowed_methods=["GET", "POST", "PUT", "DELETE"],
    raise_on_status=False,
)
session.mount("https://", HTTPAdapter(max_retries=_retry))

BASE_URL = "https://api.duda.co/api"
DEFAULT_TIMEOUT = 15   # seconds — for light reads
# POST-REVIEW FIX: duplicate/publish/content operations routinely take
# 20-40s on Duda's side. 15s was too tight.
LONG_TIMEOUT = 60      # seconds — for duplicate, publish, and content updates

# ---------------------------------------------------------------------------
# Anthropic client (module-level, lazy) — used by select_template_variant
# ---------------------------------------------------------------------------

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
_anthropic_client: Optional[anthropic.Anthropic] = None


def _get_anthropic_client() -> Optional[anthropic.Anthropic]:
    """Lazy-init a module-scoped Anthropic client. Returns None if no key set."""
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _call(method: str, path: str, **kwargs) -> dict:
    """
    Make an authenticated request to the Duda API.
    Always returns a dict; includes '_status_code' on non-2xx responses.

    Callers can override the timeout via kwargs (e.g. timeout=LONG_TIMEOUT).

    FIX #7: timeout/connection error handling so tools never crash on network issues.
    POST-REVIEW FIX: guard against non-dict JSON payloads (arrays, strings, null)
    which previously caused TypeError when we tried to attach _status_code.
    """
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    try:
        r = session.request(method, f"{BASE_URL}{path}", **kwargs)
    except requests.Timeout:
        return {"error": "Request timed out. Duda API did not respond in time.", "_status_code": 504}
    except requests.ConnectionError:
        return {"error": "Connection error. Could not reach the Duda API.", "_status_code": 503}
    except requests.RequestException as e:
        return {"error": str(e), "_status_code": 500}

    # Try JSON, but coerce non-dict payloads into a dict wrapper so callers
    # can always rely on a dict-shaped response.
    try:
        parsed = r.json()
    except ValueError:
        parsed = None

    if isinstance(parsed, dict):
        data: dict[str, Any] = parsed
    else:
        # Non-dict JSON (list, string, number, null) or unparseable body
        data = {"raw_response": parsed if parsed is not None else r.text}

    if not r.ok:
        data["_status_code"] = r.status_code
    return data


def _update_content(site_name: str, payload: dict) -> tuple[bool, Optional[str]]:
    """
    POST content library payload to a site.
    Returns (success: bool, error_message: str | None).
    The endpoint returns 204 No Content on success — no JSON body.
    """
    try:
        r = session.post(
            f"{BASE_URL}/sites/multiscreen/{site_name}/content",
            json=payload,
            timeout=LONG_TIMEOUT,  # POST-REVIEW FIX: was DEFAULT_TIMEOUT
        )
    except requests.RequestException as e:
        return False, str(e)
    if r.ok:
        return True, None
    try:
        err_body = r.json()
    except ValueError:
        err_body = r.text
    return False, f"HTTP {r.status_code}: {err_body}"


def _publish_content(site_name: str) -> tuple[bool, Optional[str]]:
    """
    Push content library changes to the live published site.
    Required after updating content on an already-published site.
    """
    try:
        r = session.post(
            f"{BASE_URL}/sites/multiscreen/{site_name}/content/publish",
            timeout=LONG_TIMEOUT,  # POST-REVIEW FIX: was DEFAULT_TIMEOUT
        )
    except requests.RequestException as e:
        return False, str(e)
    if r.ok:
        return True, None
    return False, f"HTTP {r.status_code}: {r.text}"


def _build_content_payload(
    business_name: Optional[str] = None,
    description: Optional[str] = None,
    category: Optional[str] = None,
    logo_url: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    address: Optional[str] = None,
    city: Optional[str] = None,
    region: Optional[str] = None,
    postal_code: Optional[str] = None,
    country: Optional[str] = None,
    facebook: Optional[str] = None,
    instagram: Optional[str] = None,
    twitter: Optional[str] = None,
    linkedin: Optional[str] = None,
    youtube: Optional[str] = None,
    yelp: Optional[str] = None,
    custom_texts: Optional[list[dict]] = None,
    site_images: Optional[list[dict]] = None,
) -> dict:
    """
    Build a well-structured Duda content library payload from flat inputs.

    Duda content library top-level shape:
        {
          "business_data": { "name", "logo_url", "description", "category" },
          "location_data": {
              "phones":          [{ "phoneNumber", "label" }],
              "emails":          [{ "emailAddress", "label" }],
              "address":         { "streetAddress", "city", "region",
                                   "postalCode", "country" },
              "social_accounts": { "facebook", "instagram", "twitter",
                                   "linkedin", "youtube", "yelp" },
          },
          "site_texts":  { "custom": [{ "label", "text" }] },
          "site_images": [{ "label", "url", "alt" }],
        }
    """
    payload: dict = {}

    # --- business_data -------------------------------------------------------
    biz: dict = {}
    if business_name:
        biz["name"] = business_name
    if logo_url:
        biz["logo_url"] = logo_url
    if description:
        biz["description"] = description
    if category:
        biz["category"] = category
    if biz:
        payload["business_data"] = biz

    # --- location_data -------------------------------------------------------
    loc: dict = {}

    if phone:
        loc["phones"] = [{"phoneNumber": phone, "label": "Phone"}]
    if email:
        loc["emails"] = [{"emailAddress": email, "label": "Email"}]

    addr: dict = {}
    if address:
        addr["streetAddress"] = address
    if city:
        addr["city"] = city
    if region:
        addr["region"] = region
    if postal_code:
        addr["postalCode"] = postal_code
    if country:
        addr["country"] = country
    if addr:
        loc["address"] = addr

    social: dict = {}
    for key, val in [
        ("facebook", facebook), ("instagram", instagram), ("twitter", twitter),
        ("linkedin", linkedin), ("youtube", youtube), ("yelp", yelp),
    ]:
        if val:
            social[key] = val
    if social:
        loc["social_accounts"] = social

    if loc:
        payload["location_data"] = loc

    # --- site_texts ----------------------------------------------------------
    if custom_texts:
        payload["site_texts"] = {"custom": custom_texts}

    # --- site_images ---------------------------------------------------------
    if site_images:
        payload["site_images"] = site_images

    return payload


def _duplicate_and_inject(
    source_site_name: str,
    new_default_domain: str,
    content_payload: dict,
    publish_content: bool,
) -> dict:
    """
    POST-REVIEW FIX: shared workflow extracted from build_site_from_custom_template
    and create_site_from_existing. Duplicates a site, optionally injects content
    into its Content Library, and optionally publishes the content changes.

    Returned dict includes an '_error' key when the duplicate step itself failed
    so callers can short-circuit and forward the error to the client.
    """
    new_site = _call(
        "POST",
        f"/sites/multiscreen/duplicate/{source_site_name}",
        params={"new_default_domain": new_default_domain},
        timeout=LONG_TIMEOUT,
    )
    if "_status_code" in new_site:
        return {
            "_error": True,
            "error": "Failed to duplicate template site.",
            "detail": new_site.get("raw_response", new_site),
            "_status_code": new_site["_status_code"],
        }

    new_site_name = new_site.get("site_name")
    if not new_site_name:
        return {
            "_error": True,
            "error": "Duplicate succeeded but site_name was missing from response.",
            "raw": new_site,
        }

    content_injected = False
    content_status: str = "no_fields_provided"   # FIX #5: initialize before branching
    content_warning: Optional[str] = None

    if content_payload:
        ok, err = _update_content(new_site_name, content_payload)
        if ok:
            content_injected = True
            content_status = "injected"
        else:
            content_status = "failed"
            content_warning = f"Site created but content injection failed: {err}"
    else:
        # POST-REVIEW FIX: distinguish "no content requested" from "content
        # injection failed". content_injected=False alone was ambiguous.
        content_status = "no_fields_provided"

    publish_warning: Optional[str] = None
    if publish_content and content_injected:
        pub_ok, pub_err = _publish_content(new_site_name)
        if not pub_ok:
            publish_warning = f"Content injected but publish-content failed: {pub_err}"

    result: dict = {
        "site_name": new_site_name,
        "default_domain": new_site.get("site_default_domain"),
        "edit_url": f"https://dashboard.duda.co/home/site/{new_site_name}",
        "preview_url": f"https://dashboard.duda.co/preview/{new_site_name}",
        "status": "created",
        "content_injected": content_injected,
        "content_status": content_status,
        "fields_sent": list(content_payload.keys()) if content_payload else [],
    }
    if content_warning:
        result["content_warning"] = content_warning
    if publish_warning:
        result["publish_warning"] = publish_warning
    return result


def _fmt_variant(i: int, v: dict) -> str:
    """Format one template variant for the selection prompt.
    POST-REVIEW FIX: pulled out of select_template_variant to module scope."""
    lines = [f"Variant {i + 1}: {v.get('name', v['site_name'])}"]
    if v.get("tagline"):
        lines.append(f"  Tagline: {v['tagline']}")
    if v.get("description"):
        lines.append(f"  Description: {v['description']}")
    if v.get("design_vibe"):
        lines.append(f"  Design vibe: {v['design_vibe']}")
    if v.get("ideal_client"):
        ic = v["ideal_client"]
        lines.append("  Ideal client:")
        for k, val in ic.items():
            lines.append(f"    {k}: {val}")
    if v.get("url"):
        lines.append(f"  Preview: {v['url']}")
    lines.append(f"  site_name: {v['site_name']}")
    return "\n".join(lines)


def _extract_json_object(raw: str) -> dict:
    """Robustly pull the first balanced JSON object out of a model response.

    Uses brace-depth tracking instead of a greedy regex so it handles:
      - Multiple JSON objects in one response (reasoning + answer)
      - Nested objects inside the target payload
      - Stray backticks or markdown fences around the JSON

    Raises ValueError if no valid, balanced JSON object is found.
    """
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(raw[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start : i + 1]
                return json.loads(candidate)

    raise ValueError("No balanced JSON object found in model response.")


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("duda")


# --- SITES ------------------------------------------------------------------


@mcp.tool()
def list_sites(
    limit: int = 75,
    offset: int = 0,
    label_names: Optional[str] = None,
    publish_status: Optional[str] = None,
    site_type: Optional[str] = None,
) -> dict:
    """
    List sites in the Duda account with pagination support.

    Args:
        limit: Number of sites to return per page (max 100, default 75).
        offset: Zero-based offset for pagination. Increment by `limit` to fetch
                the next page. Use `total_responses` in the result to know
                when you have fetched all sites.
        label_names: Optional comma-separated label names to filter by
                     (e.g. 'template,restaurant'). Values are OR'd.
        publish_status: Optional filter — PUBLISHED, UNPUBLISHED, or
                        NOT_PUBLISHED_YET. Multiple values comma-separated.
        site_type: Optional filter — REGULAR to show only regular sites,
                   TEMPLATE to show only template sites.

    Tip — fetch all sites in batches:
        Page 1: list_sites(limit=100, offset=0)
        Page 2: list_sites(limit=100, offset=100)
        ...stop when len(results) < limit or offset >= total_responses.
    """
    # FIX #6: site_type filter to allow filtering regular vs template sites.
    # POST-REVIEW FIX: clamp lower bounds on limit/offset.
    params: dict = {
        "limit": max(1, min(limit, 100)),
        "offset": max(0, offset),
    }
    if label_names:
        params["label_names"] = label_names
    if publish_status:
        params["publish_status"] = publish_status
    if site_type:
        params["site_type"] = site_type
    return _call("GET", "/sites/multiscreen", params=params)


@mcp.tool()
def get_site(site_name: str) -> dict:
    """Get details of a specific site by its site name."""
    return _call("GET", f"/sites/multiscreen/{site_name}")


@mcp.tool()
def publish_site(site_name: str) -> dict:
    """Publish a Duda site to make it live."""
    result = _call("POST", f"/sites/multiscreen/{site_name}/publish", timeout=LONG_TIMEOUT)
    if "_status_code" in result:
        return {"published": False, "site_name": site_name, "error": result}
    return {"published": True, "site_name": site_name}


@mcp.tool()
def unpublish_site(site_name: str) -> dict:
    """Unpublish (take offline) a Duda site."""
    result = _call("POST", f"/sites/multiscreen/{site_name}/unpublish", timeout=LONG_TIMEOUT)
    if "_status_code" in result:
        return {"unpublished": False, "site_name": site_name, "error": result}
    return {"unpublished": True, "site_name": site_name}


@mcp.tool()
def delete_site(site_name: str) -> dict:
    """
    Permanently delete a Duda site. Use with caution — this cannot be undone.

    Args:
        site_name: The site_name ID of the site to delete.

    IMPROVEMENT #10: delete_site tool for cleanup after testing.
    """
    result = _call("DELETE", f"/sites/multiscreen/{site_name}")
    if "_status_code" in result:
        return {"deleted": False, "error": result}
    return {"deleted": True, "site_name": site_name}


@mcp.tool()
def duplicate_site(site_name: str, new_default_domain: str) -> dict:
    """
    Duplicate an existing Duda site.

    Args:
        site_name: The site_name ID of the site to duplicate.
        new_default_domain: Subdomain prefix for the new site (e.g. 'client-abc').
    """
    return _call(
        "POST",
        f"/sites/multiscreen/duplicate/{site_name}",
        params={"new_default_domain": new_default_domain},
        timeout=LONG_TIMEOUT,  # POST-REVIEW FIX: duplication can take 30s+
    )


# --- TEMPLATES --------------------------------------------------------------


@mcp.tool()
def list_templates() -> dict:
    """
    List all available official Duda templates.
    Returns template IDs, names, and preview URLs for use with
    create_site_from_template.
    """
    return _call("GET", "/sites/multiscreen/templates")


@mcp.tool()
def create_site_from_template(
    template_id: int,
    new_default_domain: str,
    lang: str = "en",
    site_data: Optional[dict] = None,
) -> dict:
    """
    Create a new Duda site from an official Duda template.

    Args:
        template_id: The numeric template ID (get these from list_templates).
        new_default_domain: Subdomain for the new site (e.g. 'my-new-site').
        lang: Language code for the site, default 'en'.
        site_data: Optional dict with initial content, e.g.:
                   {"site_business_info": {"business_name": "Acme Co",
                                           "phone": "555-1234"}}
    """
    payload: dict = {
        "template_id": template_id,
        "default_domain_prefix": new_default_domain,
        "lang": lang,
    }
    if site_data:
        payload["site_data"] = site_data

    result = _call(
        "POST",
        "/sites/multiscreen/create",
        json=payload,
        timeout=LONG_TIMEOUT,  # POST-REVIEW FIX: template creation can take 30s+
    )

    # FIX #2: Guard against None site_name in edit_url
    if "_status_code" not in result:
        site_name = result.get("site_name")
        if site_name:
            result["edit_url"] = f"https://dashboard.duda.co/home/site/{site_name}"
    return result


# --- CUSTOM TEMPLATE REGISTRY -----------------------------------------------
#
# CUSTOM_TEMPLATES is loaded at startup from a YAML file so non-developers
# can add templates without editing Python. Default location is
# 'templates.yaml' next to this file; override via DUDA_TEMPLATES_FILE.
#
# YAML schema:
#   collection-name:
#     industry:    <required>  short industry slug (e.g. "plumbing")
#     description: <optional>  short summary of the collection
#     variants:                non-empty list
#       - site_name:    <required>  Duda site_name ID
#         id:           <optional>  short slug
#         name:         <optional>  human-readable label
#         tagline:      <optional>  short pitch
#         description:  <optional>  detailed prose Claude uses when picking
#         design_vibe:  <optional>  short style descriptor
#         url:          <optional>  preview URL
#         ideal_client: <optional>  free-form mapping (company_age,
#                                   target_audience, brand_voice, ...)
#
# Add or edit templates by editing the YAML file, then call the
# `reload_custom_templates` MCP tool — no server restart required. Validation
# runs on every load, so a malformed file is rejected with a clear error and
# the in-memory registry is left untouched.

DEFAULT_TEMPLATES_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "templates.yaml",
)
TEMPLATES_FILE = os.environ.get("DUDA_TEMPLATES_FILE", DEFAULT_TEMPLATES_FILE)


def _validate_templates(data: Any, path: str) -> dict:
    """Validate the parsed YAML shape. Raise ValueError with a clear message
    on any structural problem so misconfiguration fails fast at load time."""
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: top-level must be a mapping of collection-name -> config; "
            f"got {type(data).__name__}"
        )
    for name, meta in data.items():
        if not isinstance(meta, dict):
            raise ValueError(
                f"{path}: collection '{name}' must be a mapping; got {type(meta).__name__}"
            )
        if not meta.get("industry"):
            raise ValueError(
                f"{path}: collection '{name}' missing required field 'industry'"
            )
        variants = meta.get("variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError(
                f"{path}: collection '{name}' must have a non-empty 'variants' list"
            )
        for i, v in enumerate(variants):
            if not isinstance(v, dict):
                raise ValueError(
                    f"{path}: collection '{name}' variant #{i} must be a mapping"
                )
            if not v.get("site_name"):
                raise ValueError(
                    f"{path}: collection '{name}' variant #{i} missing required "
                    "field 'site_name'"
                )
    return data


def _load_templates() -> dict:
    """Load and validate the template registry from TEMPLATES_FILE.

    Behavior:
      - DUDA_TEMPLATES_FILE explicitly set but points nowhere → raise
        (misconfiguration must fail fast).
      - Default templates.yaml missing → warn to stderr and return {} so the
        server can still start and serve non-template tools.
      - File exists but is malformed → raise (fail fast).
    """
    path = TEMPLATES_FILE
    if not os.path.exists(path):
        if os.environ.get("DUDA_TEMPLATES_FILE"):
            raise FileNotFoundError(
                f"DUDA_TEMPLATES_FILE points to '{path}' but no such file exists."
            )
        print(
            f"[duda-mcp] No templates file at {path} — starting with empty "
            "template registry. Create the file and call reload_custom_templates.",
            file=sys.stderr,
        )
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"{path}: YAML parse error: {e}") from e
    return _validate_templates(raw, path)


CUSTOM_TEMPLATES: dict[str, dict] = _load_templates()

# FIX #8: In Lambda (and any packaged deployment), templates.yaml must be bundled
# alongside main.py. Log a prominent warning at startup if it's absent so the
# missing-file problem surfaces immediately in CloudWatch cold-start logs rather
# than appearing as silent "no templates" behaviour at request time.
if not CUSTOM_TEMPLATES and not os.environ.get("DUDA_TEMPLATES_FILE"):
    _expected = DEFAULT_TEMPLATES_FILE
    if not os.path.exists(_expected):
        print(
            f"[duda-mcp] WARNING: templates.yaml not found at '{_expected}'. "
            "If running in AWS Lambda, ensure templates.yaml is included in your "
            "deployment zip alongside main.py. Template selection will be unavailable "
            "until the file is present and reload_custom_templates is called.",
            file=sys.stderr,
        )


@mcp.tool()
def list_custom_templates(industry: Optional[str] = None) -> dict:
    """
    List all registered custom template collections with their variants and
    descriptions. Claude uses this to intelligently pick the best variant
    for a given client brief.

    Args:
        industry: Optional filter — show only templates for a specific
                  industry (e.g. 'plumbing', 'home-care'). Case-insensitive.
                  Leave blank to list all templates.

    Example prompts:
        "List all plumbing templates"
        "Which template should I use for a family-owned rural plumbing business?"
    """
    if not CUSTOM_TEMPLATES:
        return {
            "message": (
                "No custom templates registered yet. "
                "Edit templates.yaml and call reload_custom_templates to add templates."  # FIX #4: was pointing to main.py
            ),
            "templates": {},
        }

    templates = CUSTOM_TEMPLATES

    # Filter by industry if provided
    if industry:
        templates = {
            name: meta for name, meta in CUSTOM_TEMPLATES.items()
            if meta.get("industry", "").lower() == industry.lower()
        }
        if not templates:
            return {
                "message": f"No templates found for industry '{industry}'.",
                "available_industries": sorted({
                    str(m.get("industry", "")) for m in CUSTOM_TEMPLATES.values()
                    if m.get("industry")
                }),
                "templates": {},
            }

    return {
        "total_collections": len(templates),
        "total_variants": sum(len(m.get("variants", [])) for m in templates.values()),
        "collections": {
            name: {
                "industry": meta.get("industry", ""),
                "description": meta.get("description", ""),
                "variants": [
                    {
                        "id": v.get("id", ""),
                        "site_name": v["site_name"],
                        "name": v.get("name", ""),
                        "tagline": v.get("tagline", ""),
                        "url": v.get("url", ""),
                        "description": v.get("description", ""),
                        "ideal_client": v.get("ideal_client", {}),
                        "design_vibe": v.get("design_vibe", ""),
                    }
                    for v in meta.get("variants", [])
                ],
            }
            for name, meta in templates.items()
        },
    }


@mcp.tool()
def reload_custom_templates() -> dict:
    """
    Reload the CUSTOM_TEMPLATES registry from the YAML file (default:
    templates.yaml next to main.py, overridable via DUDA_TEMPLATES_FILE).

    Call this after editing the YAML file to pick up new templates or new
    variants without restarting the MCP server. Validation runs on the new
    file — if it fails, the current in-memory registry is left untouched
    and the validation error is returned so you can fix and retry.

    Returns:
        On success:
          {"success": true, "collections": [...], "total_collections": N,
           "total_variants": N, "source": "<path>"}
        On failure (in-memory registry unchanged):
          {"success": false, "error": "...",
           "current_collections": [...],
           "message": "Existing template registry left unchanged."}
    """
    try:
        new_templates = _load_templates()
    except (FileNotFoundError, ValueError) as e:
        return {
            "success": False,
            "error": f"Failed to reload templates: {e}",
            "current_collections": list(CUSTOM_TEMPLATES.keys()),
            "message": "Existing template registry left unchanged.",
        }

    # In-place mutation so any existing references to CUSTOM_TEMPLATES stay valid.
    CUSTOM_TEMPLATES.clear()
    CUSTOM_TEMPLATES.update(new_templates)
    return {
        "success": True,
        "collections": list(CUSTOM_TEMPLATES.keys()),
        "total_collections": len(CUSTOM_TEMPLATES),
        "total_variants": sum(
            len(m.get("variants", [])) for m in CUSTOM_TEMPLATES.values()
        ),
        "source": TEMPLATES_FILE,
    }


# --- BUILD FROM CUSTOM TEMPLATE (main new feature) --------------------------

@mcp.tool()
def select_template_variant(
    collection: str,
    client_brief: str,
) -> dict:
    """
    Intelligently select the best template variant from a collection based
    on the client brief. Call this BEFORE build_site_from_custom_template
    when a collection has multiple variants.

    Claude reads all variant descriptions and picks the best match for the
    client automatically — no manual selection needed.

    Args:
        collection:   Template collection name (e.g. 'plumbing', 'home-care').
                      Use list_custom_templates() to see available collections.
        client_brief: Short description of the client — business type, target
                      market, style, location, tone, audience, etc.
                      Example: "Family-owned plumbing in a rural town,
                                traditional values, targeting older homeowners."

    Returns:
        The selected variant's site_name and reasoning. Pass the site_name
        directly to build_site_from_custom_template.

    Example prompts:
        "Pick the best plumbing template for a high-end urban plumbing company"
        "Which home-care variant suits a pediatric care agency?"
    """
    meta = CUSTOM_TEMPLATES.get(collection)
    if not meta:
        return {
            "error": f"Collection '{collection}' not found.",
            "available_collections": list(CUSTOM_TEMPLATES.keys()),
        }

    variants = meta.get("variants", [])
    if not variants:
        return {"error": f"Collection '{collection}' has no variants registered."}

    # Single variant — no selection needed
    if len(variants) == 1:
        return {
            "selected_site_name": variants[0]["site_name"],
            "selected_variant_name": variants[0].get("name", ""),
            "selected_variant_id": variants[0].get("id", ""),
            "collection": collection,
            "reasoning": "Only one variant available in this collection.",
            "total_variants_considered": 1,
        }

    variants_text = "\n\n".join(_fmt_variant(i, v) for i, v in enumerate(variants))

    prompt = f"""You are selecting the best website template variant for a client.

Collection: {collection}
Collection description: {meta.get('description', '')}

Client brief:
{client_brief}

Available variants:
{variants_text}

Instructions:
- Carefully read each variant's description, design vibe, and ideal client profile
- Match the best variant to the client brief considering style, tone, target market,
  company age, brand voice, and key priorities
- Return ONLY valid JSON with no preamble or markdown fences:
{{
  "selected_site_name": "<site_name of the best variant>",
  "selected_variant_name": "<name of the chosen variant>",
  "reasoning": "<2-3 sentences explaining specifically why this variant fits the client brief>"
}}"""

    def _fallback(reason: str) -> dict:
        """Deterministic fallback to the first variant when LLM selection is unavailable."""
        first = variants[0]
        return {
            "selected_site_name": first["site_name"],
            "selected_variant_name": first.get("name", ""),
            "selected_variant_id": first.get("id", ""),
            "collection": collection,
            "reasoning": f"Fallback to first variant: {reason}",
            "total_variants_considered": len(variants),
            "fallback": True,
        }

    # POST-REVIEW FIX: fail loudly (via visible 'fallback: true' flag) instead of
    # silently degrading, and skip the API call entirely if there's no key.
    client = _get_anthropic_client()
    if client is None:
        return _fallback("ANTHROPIC_API_KEY not set; using first variant.")

    try:
        message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=500,
            timeout=20.0,  # FIX #9: prevent Lambda from hanging on slow/cold model responses
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:  # noqa: BLE001 — anthropic errors vary; be defensive
        return _fallback(f"Anthropic API error: {e}")

    if not message.content:
        return _fallback("Empty response from Anthropic API.")

    raw = getattr(message.content[0], "text", "") or ""
    try:
        result = _extract_json_object(raw.strip())
    except (ValueError, json.JSONDecodeError) as e:
        return _fallback(f"Could not parse JSON from model response: {e}")

    # POST-REVIEW FIX: resolve to the actual chosen variant's metadata
    # rather than mixing model output with variants[0] fields.
    selected_meta = next(
        (v for v in variants if v["site_name"] == result.get("selected_site_name")),
        None,
    )
    if selected_meta is None:
        return _fallback(
            f"Model chose site_name={result.get('selected_site_name')!r} which is "
            f"not in the '{collection}' collection."
        )

    return {
        "selected_site_name": selected_meta["site_name"],
        "selected_variant_name": selected_meta.get("name", ""),
        "selected_variant_id": selected_meta.get("id", ""),   # FIX #3: was missing from success path
        "collection": collection,
        "reasoning": result.get("reasoning", ""),
        "total_variants_considered": len(variants),
    }


@mcp.tool()
def build_site_from_custom_template(
    template_site_name: str,
    new_default_domain: str,
    # Core business info
    business_name: Optional[str] = None,
    description: Optional[str] = None,
    category: Optional[str] = None,
    logo_url: Optional[str] = None,
    # Contact
    phone: Optional[str] = None,
    email: Optional[str] = None,
    # Address
    address: Optional[str] = None,
    city: Optional[str] = None,
    region: Optional[str] = None,
    postal_code: Optional[str] = None,
    country: Optional[str] = None,
    # Social
    facebook: Optional[str] = None,
    instagram: Optional[str] = None,
    twitter: Optional[str] = None,
    linkedin: Optional[str] = None,
    youtube: Optional[str] = None,
    yelp: Optional[str] = None,
    # Custom text blocks — e.g. hero headlines, taglines, about text
    # Pass as: [{"label": "Hero Headline", "text": "We Fix Pipes Fast"}]
    custom_texts: Optional[list[dict]] = None,
    # Images — e.g. logo, banner, gallery
    # Pass as: [{"label": "Logo", "url": "https://...", "alt": "My Logo"}]
    site_images: Optional[list[dict]] = None,
    # After content is injected, push it to the live site if already published
    publish_content: bool = False,
) -> dict:
    """
    Build a new site from one of your custom template sites and immediately
    replace all placeholder content with the client's real content.

    This is a three-step atomic operation:
      1. Duplicate the template site → new site
      2. Inject all provided content into the new site's Content Library
         (business info, contact details, address, social links, custom text
         blocks, and images — all in a single API call)
      3. Optionally publish the content changes if the site is already live

    Any field left as None is simply skipped — the template's placeholder
    content stays in place for that field.

    Args:
        template_site_name: Friendly name (e.g. 'plumbing') OR raw site_name ID.
                            Use list_custom_templates() to see available names.
        new_default_domain: Subdomain prefix for the new site (e.g. 'acme-plumbing').
        business_name:  Client's business name.
        description:    Short business description / tagline.
        category:       Business category (e.g. 'Plumbing', 'Restaurant').
        logo_url:       Publicly accessible URL of the client's logo image.
        phone:          Primary phone number (e.g. '555-123-4567').
        email:          Primary contact email address.
        address:        Street address (e.g. '123 Main St').
        city:           City name.
        region:         State / province / region code (e.g. 'CA').
        postal_code:    ZIP or postal code.
        country:        Country code (e.g. 'US').
        facebook:       Facebook page handle or URL path (not full URL).
        instagram:      Instagram handle.
        twitter:        Twitter/X handle.
        linkedin:       LinkedIn company page handle.
        youtube:        YouTube channel ID.
        yelp:           Yelp business URL path.
        custom_texts:   List of custom text blocks to replace.
                        Format: [{"label": "Hero Headline", "text": "We Fix Pipes Fast"},
                                 {"label": "About Us",     "text": "Founded in 1998..."}]
                        Labels must match exactly what was set in the template's
                        Content Library (CMS → Business Text in the editor).
        site_images:    List of images to replace.
                        Format: [{"label": "Logo",   "url": "https://...", "alt": "Acme Logo"},
                                 {"label": "Banner", "url": "https://...", "alt": "Banner"}]
                        Labels must match exactly what was set in the template's
                        Content Library (CMS → Business Images in the editor).
        publish_content: Set True to push content changes to the live published
                         site immediately after injection. Only relevant if the
                         new site will be published right away.

    Returns:
        A dict with site_name, edit_url, preview_url, content_injected (bool),
        content_status, and any warnings from the content or publish steps.

    Example prompt to Claude Desktop:
        "Build a new plumbing site for Acme Plumbing — domain acme-plumbing,
         phone 555-999-0000, email info@acme.com, city Portland, region OR."
    """
    # Resolve collection name → variant site_name ID.
    # If a collection with multiple variants is passed here, we fall back to
    # the first variant AND surface a warning so the caller knows the pick
    # wasn't deliberate.
    collection_warning: Optional[str] = None
    template_meta = CUSTOM_TEMPLATES.get(template_site_name)
    if template_meta:
        variants = template_meta.get("variants", [])
        if not variants:
            return {
                "error": f"Template collection '{template_site_name}' has no variants registered.",
            }
        resolved_template = variants[0]["site_name"]
        # POST-REVIEW FIX: warn when we silently defaulted to the first of many.
        if len(variants) > 1:
            first_name = variants[0].get("name", variants[0]["site_name"])
            collection_warning = (
                f"Collection '{template_site_name}' has {len(variants)} variants; "
                f"defaulted to the first one ('{first_name}'). "
                f"Call select_template_variant() first for a targeted match."
            )
    else:
        # Treat as a raw site_name ID (returned by select_template_variant)
        resolved_template = template_site_name

    content_payload = _build_content_payload(
        business_name=business_name,
        description=description,
        category=category,
        logo_url=logo_url,
        phone=phone,
        email=email,
        address=address,
        city=city,
        region=region,
        postal_code=postal_code,
        country=country,
        facebook=facebook,
        instagram=instagram,
        twitter=twitter,
        linkedin=linkedin,
        youtube=youtube,
        yelp=yelp,
        custom_texts=custom_texts,
        site_images=site_images,
    )

    # POST-REVIEW FIX: shared workflow now lives in _duplicate_and_inject
    result = _duplicate_and_inject(
        source_site_name=resolved_template,
        new_default_domain=new_default_domain,
        content_payload=content_payload,
        publish_content=publish_content,
    )

    if result.get("_error"):
        # Surface the duplicate-step error cleanly
        result.pop("_error", None)
        return result

    result["template_source"] = resolved_template
    if collection_warning:
        result["collection_warning"] = collection_warning
    return result


@mcp.tool()
def update_site_content(
    site_name: str,
    business_name: Optional[str] = None,
    description: Optional[str] = None,
    category: Optional[str] = None,
    logo_url: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    address: Optional[str] = None,
    city: Optional[str] = None,
    region: Optional[str] = None,
    postal_code: Optional[str] = None,
    country: Optional[str] = None,
    facebook: Optional[str] = None,
    instagram: Optional[str] = None,
    twitter: Optional[str] = None,
    linkedin: Optional[str] = None,
    youtube: Optional[str] = None,
    yelp: Optional[str] = None,
    custom_texts: Optional[list[dict]] = None,
    site_images: Optional[list[dict]] = None,
    publish_content: bool = False,
) -> dict:
    """
    Update (replace) content library fields on any existing Duda site.
    Use this to re-inject or correct content after a site has been built,
    or to keep a live site's content in sync.

    Only fields you provide are sent to the API — unset fields are untouched.

    Args:
        site_name: The site_name ID of the site to update.
        business_name:  Business name.
        description:    Business description or tagline.
        category:       Business category.
        logo_url:       URL of the logo image.
        phone:          Primary phone number.
        email:          Primary contact email.
        address:        Street address.
        city:           City.
        region:         State / province / region.
        postal_code:    ZIP or postal code.
        country:        Country code.
        facebook / instagram / twitter / linkedin / youtube / yelp:
                        Social media handles or URL paths.
        custom_texts:   [{"label": "Hero Headline", "text": "New headline"}]
        site_images:    [{"label": "Banner", "url": "https://...", "alt": "Banner"}]
        publish_content: Push changes to the live published site immediately.

    Returns:
        Dict with content_updated (bool) and any warnings.
    """
    payload = _build_content_payload(
        business_name=business_name,
        description=description,
        category=category,
        logo_url=logo_url,
        phone=phone,
        email=email,
        address=address,
        city=city,
        region=region,
        postal_code=postal_code,
        country=country,
        facebook=facebook,
        instagram=instagram,
        twitter=twitter,
        linkedin=linkedin,
        youtube=youtube,
        yelp=yelp,
        custom_texts=custom_texts,
        site_images=site_images,
    )

    if not payload:
        return {"content_updated": False, "message": "No content fields provided."}

    ok, err = _update_content(site_name, payload)
    if not ok:
        return {"content_updated": False, "error": err}

    publish_warning: Optional[str] = None
    if publish_content:
        pub_ok, pub_err = _publish_content(site_name)
        if not pub_ok:
            publish_warning = f"Content updated but publish-content failed: {pub_err}"

    response: dict = {
        "content_updated": True,
        "fields_sent": list(payload.keys()),
        "site_name": site_name,
        "edit_url": f"https://dashboard.duda.co/home/site/{site_name}",
    }
    if publish_warning:
        response["publish_warning"] = publish_warning
    return response


# --- LEGACY: create_site_from_existing (kept for backward compat) -----------


@mcp.tool()
def create_site_from_existing(
    template_site_name: str,
    new_default_domain: str,
    business_name: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
) -> dict:
    """
    (Legacy) Duplicate an existing site and optionally pre-fill basic contact
    fields. For full content injection use build_site_from_custom_template.

    Args:
        template_site_name: site_name of the site to use as a template.
        new_default_domain: Subdomain prefix for the new site (e.g. 'client-name').
        business_name: Optional — pre-fill the business name.
        phone: Optional — pre-fill the phone number.
        email: Optional — pre-fill the email address.
    """
    # POST-REVIEW FIX: reuse the shared _duplicate_and_inject workflow so
    # this tool and build_site_from_custom_template can't drift apart.
    content_payload = _build_content_payload(
        business_name=business_name,
        phone=phone,
        email=email,
    )

    result = _duplicate_and_inject(
        source_site_name=template_site_name,
        new_default_domain=new_default_domain,
        content_payload=content_payload,
        publish_content=False,
    )

    if result.get("_error"):
        result.pop("_error", None)
        return result

    result["template_source"] = template_site_name
    return result


# --- CLIENT / CONTENT COLLECTION --------------------------------------------

@mcp.tool()
def create_client_account(
    site_name: str,
    client_email: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
) -> dict:
    """
    Create a client account and grant content collection access for a site.
    After creation, retrieve the login link with get_content_collection_link.

    Args:
        site_name: The site_name ID to grant access to.
        client_email: The client's email address (used as their username).
        first_name: Optional client first name.
        last_name: Optional client last name.
    """
    # Step 1: Create the account
    payload: dict = {"account_name": client_email, "account_type": "CUSTOMER"}
    if first_name:
        payload["first_name"] = first_name
    if last_name:
        payload["last_name"] = last_name

    result = _call("POST", "/accounts/create", json=payload)

    # FIX #4 (revised): only treat this as "already exists" when the response
    # actually signals that. Blindly proceeding on every 400 hid real validation
    # errors and could grant permissions on non-existent accounts.
    if "_status_code" in result:
        status = result.get("_status_code")
        body_str = json.dumps(result, default=str).lower()
        already_exists = (
            status == 409  # conflict — the canonical "already exists" signal
            or (status == 400 and ("already" in body_str or "exist" in body_str))
        )
        if not already_exists:
            return {"error": "Failed to create client account.", "detail": result}
        # Account already exists — safe to continue to permissions step.

    # Step 2: Grant content collection permissions
    perms = _call(
        "POST",
        f"/accounts/permissions/multiscreen/{site_name}/client/{client_email}",
        json={"permissions": ["CONTENT_LIBRARY_CONTENT"]},
    )
    if "_status_code" in perms:
        return {
            "error": "Account created but failed to grant permissions.",
            "detail": perms,
        }

    return {
        "success": True,
        "account_name": client_email,
        "site_name": site_name,
        "message": "Client account created with content collection access. Call get_content_collection_link to get the form URL.",
    }


@mcp.tool()
def get_content_collection_link(
    site_name: str,
    client_email: str,
) -> dict:
    """
    Get the content collection form link to send to the client.
    The client fills in this form to populate the site's Content Library.

    Args:
        site_name: The site_name ID.
        client_email: The client's account email (created via create_client_account).
    """
    # FIX #3: Corrected SSO target from "CONTENT_LIBRARY" to "CONTENT_COLLECTION"
    # per Duda's SSO documentation.
    result = _call(
        "GET",
        f"/accounts/sso/{client_email}",
        params={"site_name": site_name, "target": "CONTENT_COLLECTION"},
    )
    if "_status_code" in result:
        return {"error": "Failed to get content collection link.", "detail": result}

    return {
        "site_name": site_name,
        "client_email": client_email,
        "link": result.get("url"),
        "message": "Send this link to your client to fill in their business details.",
    }


# --- ANALYTICS --------------------------------------------------------------


@mcp.tool()
def get_site_analytics(site_name: str, from_date: str, to_date: str) -> dict:
    """
    Get traffic analytics for a site.

    Args:
        site_name: The site name/ID.
        from_date: Start date in YYYY-MM-DD format.
        to_date: End date in YYYY-MM-DD format.
    """
    parsed_dates = {}
    for label, value in (("from_date", from_date), ("to_date", to_date)):
        try:
            parsed_dates[label] = datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return {
                "error": f"Invalid {label} '{value}'. Expected format: YYYY-MM-DD."
            }

    # FIX #6: guard against reversed or oversized date ranges
    dt_from = parsed_dates["from_date"]
    dt_to = parsed_dates["to_date"]
    if dt_to < dt_from:
        return {"error": "to_date must be on or after from_date."}
    if (dt_to - dt_from).days > 365:
        return {
            "error": (
                f"Date range is {(dt_to - dt_from).days} days. "
                "Duda analytics supports a maximum of 365 days per query. "
                "Split into smaller ranges and combine the results."
            )
        }

    return _call(
        "GET",
        f"/sites/multiscreen/analytics/site/{site_name}",
        params={"from": from_date, "to": to_date},
    )


# --- PAGES ------------------------------------------------------------------


@mcp.tool()
def list_pages(site_name: str) -> dict:
    """List all pages of a Duda site."""
    return _call("GET", f"/sites/multiscreen/{site_name}/pages")


# --- BLOG -------------------------------------------------------------------


@mcp.tool()
def list_blog_posts(site_name: str) -> dict:
    """List all blog posts for a Duda site."""
    return _call("GET", f"/sites/multiscreen/{site_name}/blog/posts")


@mcp.tool()
def create_blog(site_name: str) -> dict:
    """
    Initialize (enable) the blog feature on a Duda site.

    Must be called once before create_blog_post will work on a site that
    has never had a blog. Duda returns 405 on create_blog_post if the blog
    page does not exist yet — call this first to create it.

    Args:
        site_name: The site name/ID.
    """
    try:
        r = session.post(
            f"{BASE_URL}/sites/multiscreen/{site_name}/blog",
            json={},
            timeout=LONG_TIMEOUT,
        )
    except requests.RequestException as e:
        return {"error": str(e), "_status_code": 500}

    if r.ok:
        return {
            "success": True,
            "site_name": site_name,
            "message": "Blog created successfully. You can now create blog posts on this site.",
        }

    # 409 Conflict OR 400 ResourceAlreadyExist — Duda is inconsistent, both mean blog exists
    if r.status_code == 409:
        return {
            "success": True,
            "site_name": site_name,
            "message": "Blog already exists on this site. Ready to create posts.",
        }

    try:
        err_body = r.json()
    except ValueError:
        err_body = r.text

    if r.status_code == 400 and isinstance(err_body, dict) and err_body.get("error_code") == "ResourceAlreadyExist":
        return {
            "success": True,
            "site_name": site_name,
            "message": "Blog already exists on this site. Ready to create posts.",
        }

    return {
        "success": False,
        "error": f"Failed to create blog. HTTP {r.status_code}: {err_body}",
        "_status_code": r.status_code,
    }


@mcp.tool()
def import_blog_post(
    site_name: str,
    title: str,
    body: str,
    description: Optional[str] = None,
    author: Optional[str] = None,
    image_url: Optional[str] = None,
    image_alt: Optional[str] = None,
) -> dict:
    """
    Import a single blog post and append it to an existing site blog.
    Posts are always created in DRAFT mode — publish manually in the Duda
    editor or call publish_blog_post after this.

    Use this as an alternative to create_blog_post if that endpoint returns
    405. Both write to the same blog; import is more permissive about the
    blog's initialization state.

    Duda's import endpoint requires (per the official API reference):
      - content:     base64 encoding of the HTML body  (NOT raw HTML)
      - description: a short summary of the post
      - title:       max 200 chars
    Optional: author (max 200 chars), main_image, thumbnail.

    Args:
        site_name:   The site name/ID.
        title:       Blog post title (truncated to 200 chars).
        body:        HTML content of the blog post (base64-encoded before send).
        description: Short summary. If omitted, derived from the body text.
        author:      Optional author display name (truncated to 200 chars).
        image_url:   Optional URL for the post header image (main_image).
        image_alt:   Optional alt text for the image.
    """
    # content must be a base64 encoding of the HTML, per Duda's API reference.
    content_b64 = base64.b64encode(body.encode("utf-8")).decode("ascii")

    # description is required; derive a plain-text snippet from the body if absent.
    if not description:
        plain = re.sub(r"<[^>]+>", " ", body)
        plain = re.sub(r"\s+", " ", plain).strip()
        description = (plain[:157] + "...") if len(plain) > 160 else plain
        if not description:
            description = title

    payload: dict = {
        "title": title[:200],
        "content": content_b64,
        "description": description,
    }
    if author:
        payload["author"] = author[:200]
    if image_url:
        # main_image is an object per the API reference.
        payload["main_image"] = {"url": image_url}
        if image_alt:
            payload["main_image"]["alt"] = image_alt

    result = _call(
        "POST",
        f"/sites/multiscreen/{site_name}/blog/posts/import",
        json=payload,
        timeout=LONG_TIMEOUT,
    )

    if "_status_code" not in result:
        result.setdefault("note", (
            "Post imported as DRAFT. Publish it manually in the Duda editor "
            "or call publish_blog_post with the returned post id."
        ))
    return result


@mcp.tool()
def publish_blog_post(site_name: str, post_id: str) -> dict:
    """
    Publish a blog post that is currently in DRAFT status.

    Args:
        site_name: The site name/ID.
        post_id:   The blog post ID (returned by list_blog_posts or import_blog_post).
    """
    try:
        r = session.post(
            f"{BASE_URL}/sites/multiscreen/{site_name}/blog/posts/{post_id}/publish",
            timeout=LONG_TIMEOUT,
        )
    except requests.RequestException as e:
        return {"error": str(e), "_status_code": 500}

    if r.ok:
        return {
            "success": True,
            "site_name": site_name,
            "post_id": post_id,
            "message": "Blog post published successfully.",
        }
    try:
        err_body = r.json()
    except ValueError:
        err_body = r.text
    return {
        "success": False,
        "error": f"Failed to publish post. HTTP {r.status_code}: {err_body}",
        "_status_code": r.status_code,
    }


@mcp.tool()
def create_blog_post(
    site_name: str,
    title: str,
    body: str,
    status: Literal["DRAFT", "PUBLISHED"] = "DRAFT",
    auto_wrap_plain_text: bool = False,
) -> dict:
    """
    Create a blog post on a Duda site.

    Args:
        site_name: The site name/ID.
        title: Blog post title.
        body: HTML content of the blog post. Duda expects valid HTML — passing
              plain text or Markdown will render without formatting. Set
              auto_wrap_plain_text=True to have plain text wrapped in <p> tags
              automatically, or convert Markdown to HTML before calling this tool.
        status: 'DRAFT' (default) or 'PUBLISHED'.
        auto_wrap_plain_text: If True and body contains no HTML tags, wraps each
                              non-empty line in <p>...</p> tags so the post renders
                              with paragraph breaks instead of as a single run-on block.
    """
    # POST-REVIEW FIX: Literal is a hint, not a runtime check — validate.
    if status not in ("DRAFT", "PUBLISHED"):
        return {
            "error": f"Invalid status '{status}'. Must be 'DRAFT' or 'PUBLISHED'.",
        }

    # FIX #7: detect plain text / Markdown and warn or auto-wrap.
    final_body = body
    body_looks_like_html = bool(re.search(r"<[a-zA-Z][^>]*>", body))
    if not body_looks_like_html:
        if auto_wrap_plain_text:
            final_body = "\n".join(
                f"<p>{line}</p>" for line in body.splitlines() if line.strip()
            )
        else:
            # Surface a warning in the response so the caller knows what happened.
            result = _call(
                "POST",
                f"/sites/multiscreen/{site_name}/blog/posts",
                json={"title": title, "body": final_body, "status": status},
            )
            result.setdefault("body_warning", (
                "The body appears to be plain text or Markdown, not HTML. "
                "Duda renders the body as-is, so formatting may be lost. "
                "Pass HTML or set auto_wrap_plain_text=True for basic paragraph wrapping."
            ))
            return result

    return _call(
        "POST",
        f"/sites/multiscreen/{site_name}/blog/posts",
        json={"title": title, "body": final_body, "status": status},
    )


# --- CONTENT ----------------------------------------------------------------


@mcp.tool()
def get_content_library(site_name: str) -> dict:
    """
    Get the full content library of a Duda site (business info, text blocks,
    images, social links, address, etc.).

    Use this to inspect what placeholder labels exist in a template before
    calling build_site_from_custom_template, so you know which custom_texts
    and site_images labels to target.
    """
    return _call("GET", f"/sites/multiscreen/{site_name}/content")


# --- CLIENT ACCOUNTS --------------------------------------------------------


@mcp.tool()
def list_client_accounts(site_name: str) -> dict:
    """List all client accounts (permissions) for a Duda site."""
    return _call("GET", f"/accounts/client/{site_name}")


# --- SITE VERSION -----------------------------------------------------------


@mcp.tool()
def get_site_version(site_name: str) -> dict:
    """
    Check whether a Duda site is on the Classic editor or Editor 2.0.
    Call this BEFORE creating blog posts to decide which approach to use.

    Duda's Get Site response exposes an `editor` field:
      - "ADVANCED" or "ADVANCED-2.0"  -> Editor 2.0
      - anything else                  -> Classic editor

    NOTE: Duda may return either "ADVANCED" or "ADVANCED-2.0" for Editor 2.0
    sites depending on account/plan. Both are treated as 2.0 here. The raw
    value is always returned as `editor_raw` for verification.

    Returns 'CLASSIC' or '2.0' in the editor_version field.
    """
    result = _call("GET", f"/sites/multiscreen/{site_name}")

    if "_status_code" in result:
        status = result["_status_code"]
        if status == 404:
            return {
                "success": False,
                "site_name": site_name,
                "error": f"Site '{site_name}' not found. Check the site_name and try again.",
                "_status_code": status,
            }
        if status == 401:
            return {
                "success": False,
                "site_name": site_name,
                "error": "Unauthorized. Check your Duda API credentials.",
                "_status_code": status,
            }
        return {
            "success": False,
            "site_name": site_name,
            "error": "Could not retrieve site info.",
            "detail": result,
            "_status_code": status,
        }

    editor = result.get("editor", "")
    if not editor:
        return {
            "success": False,
            "site_name": site_name,
            "error": "Site response did not include an editor field. Cannot determine version.",
            "editor_raw": editor,
        }

    return {
        "success": True,
        "site_name": site_name,
        "editor_version": "2.0" if editor in ("ADVANCED", "ADVANCED-2.0") else "CLASSIC",
        "editor_raw": editor,
    }


@mcp.tool()
def update_blog_post(
    site_name: str,
    post_id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    author_name: Optional[str] = None,
    meta_title: Optional[str] = None,
    path: Optional[str] = None,
    no_index: Optional[bool] = None,
    publish_date: Optional[str] = None,
    schedule_publish_date: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """
    Update METADATA on an existing blog post (Duda PATCH endpoint).

    IMPORTANT — what this endpoint CANNOT do:
      * It does NOT edit the post body/content. To change body content you must
        delete the post (delete_blog_post) and re-import it (import_blog_post).
      * It does NOT change DRAFT/PUBLISHED status. Use publish_blog_post /
        unpublish_blog_post for that.

    Supported fields (only provided values are sent):
        title:                  Post title.
        description:            Short summary / excerpt.
        author_name:            Author display name (NOTE: 'author_name', not 'author').
        meta_title:             SEO meta title.
        path:                   URL slug / path for the post.
        no_index:               True to add noindex to the post.
        tags:                   List of tag strings.
        publish_date:           Display date (ISO 8601). Does NOT publish.
        schedule_publish_date:  Schedules publishing (ISO 8601).

    Args:
        site_name: The site name/ID.
        post_id:   The blog post ID (from list_blog_posts / import_blog_post).
    """
    payload: dict = {}
    if title is not None:
        payload["title"] = title
    if description is not None:
        payload["description"] = description
    if author_name is not None:
        payload["author_name"] = author_name
    if meta_title is not None:
        payload["meta_title"] = meta_title
    if path is not None:
        payload["path"] = path
    if no_index is not None:
        payload["no_index"] = no_index
    if publish_date is not None:
        payload["publish_date"] = publish_date
    if schedule_publish_date is not None:
        payload["schedule_publish_date"] = schedule_publish_date
    if tags is not None:
        payload["tags"] = tags

    if not payload:
        return {"message": "No update parameters were provided. Post remains unchanged."}

    result = _call(
        "PATCH",
        f"/sites/multiscreen/{site_name}/blog/posts/{post_id}",
        json=payload,
        timeout=LONG_TIMEOUT,
    )
    if not result or "_status_code" in result:
        return {
            "error": f"Failed to update blog post {post_id}.",
            "detail": result if result else "No response from Duda API server.",
        }
    result.setdefault("success", True)
    result.setdefault("fields_updated", list(payload.keys()))
    return result


@mcp.tool()
def unpublish_blog_post(site_name: str, post_id: str) -> dict:
    """
    Unpublish a blog post, returning it to DRAFT status.
    Pairs with publish_blog_post. Use this to take a live post offline,
    since update_blog_post cannot change publish status.

    Args:
        site_name: The site name/ID.
        post_id:   The blog post ID.
    """
    try:
        r = session.post(
            f"{BASE_URL}/sites/multiscreen/{site_name}/blog/posts/{post_id}/unpublish",
            timeout=LONG_TIMEOUT,
        )
    except requests.RequestException as e:
        return {"error": str(e), "_status_code": 500}

    if r.ok:
        return {
            "success": True,
            "site_name": site_name,
            "post_id": post_id,
            "message": "Blog post unpublished (returned to DRAFT).",
        }
    try:
        err_body = r.json()
    except ValueError:
        err_body = r.text
    return {
        "success": False,
        "error": f"Failed to unpublish post. HTTP {r.status_code}: {err_body}",
        "_status_code": r.status_code,
    }


@mcp.tool()
def delete_blog_post(site_name: str, post_id: str) -> dict:
    """
    Permanently delete a blog post. Cannot be undone.

    Because Duda's update endpoint cannot edit post body content, the supported
    way to "edit" content is: delete_blog_post then import_blog_post with the
    corrected HTML.

    Args:
        site_name: The site name/ID.
        post_id:   The blog post ID (from list_blog_posts).
    """
    result = _call(
        "DELETE",
        f"/sites/multiscreen/{site_name}/blog/posts/{post_id}",
        timeout=LONG_TIMEOUT,
    )
    if "_status_code" in result:
        return {"deleted": False, "site_name": site_name, "post_id": post_id, "error": result}
    return {"deleted": True, "site_name": site_name, "post_id": post_id}


@mcp.tool()
def get_blog_post(site_name: str, post_id: str) -> dict:
    """
    Retrieve a single blog post by ID, to verify what was pushed
    (title, status, path, body, etc.).

    Args:
        site_name: The site name/ID.
        post_id:   The blog post ID (from list_blog_posts).
    """
    return _call("GET", f"/sites/multiscreen/{site_name}/blog/posts/{post_id}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()