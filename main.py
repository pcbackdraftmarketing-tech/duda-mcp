"""
Duda MCP Server
===============
Exposes Duda website-management operations as MCP tools via FastMCP.

Required environment variables (set in a .env file or the process environment):
    DUDA_API_USER   — your Duda API username
    DUDA_API_PASS   — your Duda API password

Usage:
    python main.py
"""

import base64
import os
from datetime import datetime
from typing import Any, Literal, Optional

import json
import anthropic
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

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
session.headers.update(
    {
        "Authorization": f"Basic {creds}",
        "Content-Type": "application/json",
    }
)

BASE_URL = "https://api.duda.co/api"
DEFAULT_TIMEOUT = 15  # seconds

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _call(method: str, path: str, **kwargs) -> dict:
    """
    Make an authenticated request to the Duda API.
    Always returns a dict; includes '_status_code' on non-2xx responses.

    FIX #7: Added timeout and connection error handling so tools never crash
    with an unhandled exception on network issues.
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

    data: dict[str, Any]
    try:
        data = r.json()
    except ValueError:
        data = {"raw_response": r.text}
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
            timeout=DEFAULT_TIMEOUT,
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
            timeout=DEFAULT_TIMEOUT,
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
    # FIX #6: Added site_type filter to allow filtering regular vs template sites.
    params: dict = {"limit": min(limit, 100), "offset": offset}
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
    try:
        r = session.post(
            f"{BASE_URL}/sites/multiscreen/{site_name}/publish",
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as e:
        return {"status_code": 500, "message": str(e)}
    return {"status_code": r.status_code, "message": "Published" if r.ok else r.text}


@mcp.tool()
def unpublish_site(site_name: str) -> dict:
    """Unpublish (take offline) a Duda site."""
    try:
        r = session.post(
            f"{BASE_URL}/sites/multiscreen/{site_name}/unpublish",
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as e:
        return {"status_code": 500, "message": str(e)}
    return {"status_code": r.status_code, "message": "Unpublished" if r.ok else r.text}


@mcp.tool()
def delete_site(site_name: str) -> dict:
    """
    Permanently delete a Duda site. Use with caution — this cannot be undone.

    Args:
        site_name: The site_name ID of the site to delete.

    IMPROVEMENT #10: Added delete_site tool for cleanup after testing.
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

    result = _call("POST", "/sites/multiscreen/create", json=payload)

    # FIX #2: Guard against None site_name in edit_url
    if "_status_code" not in result:
        site_name = result.get("site_name")
        if site_name:
            result["edit_url"] = f"https://dashboard.duda.co/home/site/{site_name}"
    return result


# --- CUSTOM TEMPLATE REGISTRY -----------------------------------------------

# Map friendly collection names → template metadata dicts.
#
# Each entry supports MULTIPLE VARIANTS via the "variants" list.
# Claude reads each variant's description and automatically picks the best
# match based on the client brief — no manual selection needed.
#
# Structure:
#   "collection-name": {
#       "industry":    (required) industry vertical for filtering
#       "description": (required) short summary of the collection
#       "variants": [  (required) list of template variants
#           {
#               "site_name":   (required) Duda site_name ID
#               "description": (required) detailed description Claude uses to
#                              decide which variant fits the client best.
#                              Be specific: mention style, colors, tone,
#                              target market, layout, etc.
#           },
#           ...
#       ]
#   }
#
# Single-variant collections are supported — just add one item to "variants".
#
# To add a new template:
#   1. Build the site in Duda and note its site_name ID
#   2. Add a new variant dict under the relevant collection
#   3. Write a detailed description so Claude can match it to client briefs

CUSTOM_TEMPLATES: dict[str, dict] = {
    "plumbing": {
        "industry": "plumbing",
        "description": "Professional plumbing website templates.",
        "variants": [
            {
                "site_name": "efe308d8",
                "description": (
                    "Clean professional layout with a bold hero section. "
                    "Best for established plumbing businesses that want a "
                    "trustworthy, no-frills look. Works well for both urban "
                    "and suburban markets."
                ),
            },
            # Add more plumbing variants below:
            # {
            #     "site_name": "abc12345",
            #     "description": (
            #         "Dark modern theme with large imagery and bold typography. "
            #         "Best for premium plumbing services targeting high-end urban clients."
            #     ),
            # },
        ],
    },
    "home-care": {
        "industry": "home-care",
        "description": "Warm and professional home care website templates.",
        "variants": [
            {
                "site_name": "f9ab6bf6",
                "description": (
                    "Warm, friendly design with soft colors and inviting imagery. "
                    "Best for family-oriented home care agencies focused on "
                    "elderly care or personal care services."
                ),
            },
        ],
    },
}


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
                "Edit the CUSTOM_TEMPLATES dict in main.py to add templates."
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
                    str(m["industry"]) for m in CUSTOM_TEMPLATES.values() if "industry" in m
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
                        "site_name": v["site_name"],
                        "description": v.get("description", ""),
                    }
                    for v in meta.get("variants", [])
                ],
            }
            for name, meta in templates.items()
        },
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
            "collection": collection,
            "variant_description": variants[0].get("description", ""),
            "reasoning": "Only one variant available in this collection.",
            "total_variants_considered": 1,
        }

    # Build variants summary for the selection prompt
    variants_text = "\n\n".join(
        f"Variant {i + 1}:\n  site_name: {v['site_name']}\n  description: {v.get('description', 'No description')}"
        for i, v in enumerate(variants)
    )

    prompt = f"""You are selecting the best website template variant for a client.

Collection: {collection}
Collection description: {meta.get('description', '')}

Client brief:
{client_brief}

Available variants:
{variants_text}

Instructions:
- Read each variant description carefully
- Match the variant to the client brief based on style, tone, target market, and layout
- Return ONLY valid JSON with no preamble or markdown fences:
{{
  "selected_site_name": "<site_name of the best variant>",
  "variant_description": "<description of the chosen variant>",
  "reasoning": "<1-2 sentences explaining why this variant fits the client>"
}}"""

    try:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        message = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = getattr(message.content[0], "text", "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
    except Exception as e:
        # Fallback to first variant if API call fails
        return {
            "selected_site_name": variants[0]["site_name"],
            "collection": collection,
            "variant_description": variants[0].get("description", ""),
            "reasoning": f"Fallback to first variant due to selection error: {e}",
            "total_variants_considered": len(variants),
        }

    return {
        "selected_site_name": result.get("selected_site_name"),
        "collection": collection,
        "variant_description": result.get("variant_description", ""),
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
        and any warnings from the content or publish steps.

    Example prompt to Claude Desktop:
        "Build a new plumbing site for Acme Plumbing — domain acme-plumbing,
         phone 555-999-0000, email info@acme.com, city Portland, region OR."
    """
    # Resolve collection name → best variant site_name ID.
    # Claude should call select_template_variant() first to get the best
    # site_name ID for the client brief, then pass it here directly.
    # If a collection name is passed, we fall back to the first variant.
    # Raw site_name IDs (not in CUSTOM_TEMPLATES) are used as-is.
    template_meta = CUSTOM_TEMPLATES.get(template_site_name)
    if template_meta:
        variants = template_meta.get("variants", [])
        if not variants:
            return {
                "error": f"Template collection '{template_site_name}' has no variants registered.",
            }
        # Fallback to first variant — Claude should use select_template_variant first
        resolved_template = variants[0]["site_name"]
    else:
        # Treat as a raw site_name ID (returned by select_template_variant)
        resolved_template = template_site_name

    # ── Step 1: Duplicate the template ──────────────────────────────────────
    new_site = _call(
        "POST",
        f"/sites/multiscreen/duplicate/{resolved_template}",
        params={"new_default_domain": new_default_domain},
    )
    if "_status_code" in new_site:
        return {
            "error": "Failed to duplicate template site.",
            "detail": new_site.get("raw_response", new_site),
            "_status_code": new_site["_status_code"],
        }

    # FIX #1: Explicit guard — fail clearly if site_name is missing.
    new_site_name = new_site.get("site_name")
    if not new_site_name:
        return {
            "error": "Duplicate succeeded but site_name was missing from response.",
            "raw": new_site,
        }

    # ── Step 2: Build and inject content payload ─────────────────────────────
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

    content_injected = False
    content_warning: Optional[str] = None

    if content_payload:
        ok, err = _update_content(new_site_name, content_payload)
        if ok:
            content_injected = True
        else:
            content_warning = f"Site created but content injection failed: {err}"

    # ── Step 3: Optionally publish content changes ───────────────────────────
    publish_warning: Optional[str] = None
    if publish_content and content_injected:
        pub_ok, pub_err = _publish_content(new_site_name)
        if not pub_ok:
            publish_warning = f"Content injected but publish-content failed: {pub_err}"

    # ── Response ─────────────────────────────────────────────────────────────
    response: dict = {
        "site_name": new_site_name,
        "default_domain": new_site.get("site_default_domain"),
        "edit_url": f"https://dashboard.duda.co/home/site/{new_site_name}",
        "preview_url": f"https://dashboard.duda.co/preview/{new_site_name}",
        "status": "created",
        "template_source": resolved_template,
        "content_injected": content_injected,
        "fields_sent": list(content_payload.keys()) if content_payload else [],
    }
    if content_warning:
        response["content_warning"] = content_warning
    if publish_warning:
        response["publish_warning"] = publish_warning

    return response


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
    new_site = _call(
        "POST",
        f"/sites/multiscreen/duplicate/{template_site_name}",
        params={"new_default_domain": new_default_domain},
    )
    if "_status_code" in new_site:
        return {"error": new_site.get("raw_response", "Unknown error"), **new_site}

    # FIX #8: Explicit guard — same as build_site_from_custom_template.
    new_site_name = new_site.get("site_name")
    if not new_site_name:
        return {
            "error": "Duplicate succeeded but site_name was missing from response.",
            "raw": new_site,
        }

    content_warning: Optional[str] = None
    if any([business_name, phone, email]):
        payload = _build_content_payload(
            business_name=business_name,
            phone=phone,
            email=email,
        )
        ok, err = _update_content(new_site_name, payload)
        if not ok:
            content_warning = f"Site created but content update failed: {err}"

    response: dict = {
        "site_name": new_site_name,
        "default_domain": new_site.get("site_default_domain"),
        "edit_url": f"https://dashboard.duda.co/home/site/{new_site_name}",
        "preview_url": f"https://dashboard.duda.co/preview/{new_site_name}",
        "status": "created",
        "template_source": template_site_name,
    }
    if content_warning:
        response["content_update_warning"] = content_warning
    return response


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

    # FIX #4: If account already exists (400), still proceed to grant permissions.
    if "_status_code" in result:
        if result.get("_status_code") != 400:
            return {"error": "Failed to create client account.", "detail": result}
        # 400 = account already exists — safe to continue to permissions step

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
    for label, value in (("from_date", from_date), ("to_date", to_date)):
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            return {
                "error": f"Invalid {label} '{value}'. Expected format: YYYY-MM-DD."
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
def create_blog_post(
    site_name: str,
    title: str,
    body: str,
    status: Literal["DRAFT", "PUBLISHED"] = "DRAFT",
) -> dict:
    """
    Create a blog post on a Duda site.

    Args:
        site_name: The site name/ID.
        title: Blog post title.
        body: HTML content of the blog post.
        status: 'DRAFT' (default) or 'PUBLISHED'.
    """
    return _call(
        "POST",
        f"/sites/multiscreen/{site_name}/blog/posts",
        json={"title": title, "body": body, "status": status},
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


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()