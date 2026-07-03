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
    """
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    r = session.request(method, f"{BASE_URL}{path}", **kwargs)
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
    r = session.post(
        f"{BASE_URL}/sites/multiscreen/{site_name}/content",
        json=payload,
        timeout=DEFAULT_TIMEOUT,
    )
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
    r = session.post(
        f"{BASE_URL}/sites/multiscreen/{site_name}/content/publish",
        timeout=DEFAULT_TIMEOUT,
    )
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
        loc["phones"] = [{"phoneNumber": phone, "label": "Main"}]
    if email:
        loc["emails"] = [{"emailAddress": email, "label": "Main"}]

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
        # custom_texts expected as [{"label": "...", "text": "..."}, ...]
        payload["site_texts"] = {"custom": custom_texts}

    # --- site_images ---------------------------------------------------------
    if site_images:
        # site_images expected as [{"label": "...", "url": "...", "alt": "..."}, ...]
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

    Tip — fetch all 422 sites in batches:
        Page 1: list_sites(limit=100, offset=0)
        Page 2: list_sites(limit=100, offset=100)
        ...stop when len(results) < limit or offset >= total_responses.
    """
    params: dict = {"limit": min(limit, 100), "offset": offset}
    if label_names:
        params["label_names"] = label_names
    if publish_status:
        params["publish_status"] = publish_status
    return _call("GET", "/sites/multiscreen", params=params)


@mcp.tool()
def get_site(site_name: str) -> dict:
    """Get details of a specific site by its site name."""
    return _call("GET", f"/sites/multiscreen/{site_name}")


@mcp.tool()
def publish_site(site_name: str) -> dict:
    """Publish a Duda site to make it live."""
    r = session.post(
        f"{BASE_URL}/sites/multiscreen/{site_name}/publish",
        timeout=DEFAULT_TIMEOUT,
    )
    return {"status_code": r.status_code, "message": "Published" if r.ok else r.text}


@mcp.tool()
def unpublish_site(site_name: str) -> dict:
    """Unpublish (take offline) a Duda site."""
    r = session.post(
        f"{BASE_URL}/sites/multiscreen/{site_name}/unpublish",
        timeout=DEFAULT_TIMEOUT,
    )
    return {"status_code": r.status_code, "message": "Unpublished" if r.ok else r.text}


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
    if "_status_code" not in result:
        result["edit_url"] = (
            f"https://dashboard.duda.co/home/site/{result.get('site_name')}"
        )
    return result


# --- CUSTOM TEMPLATE REGISTRY -----------------------------------------------

# Map friendly names → site_name IDs of your template sites.
# Run list_sites(label_names="template") to discover them if you label
# template sites in the Duda dashboard, then paste the site_name values here.
CUSTOM_TEMPLATES: dict[str, str] = {
    "plumbing": "efe308d8",
    "home-care": "f9ab6bf6",
}


@mcp.tool()
def list_custom_templates() -> dict:
    """
    List your registered custom site templates (sites you've built and
    designated as reusable starting points).

    Returns a mapping of friendly template name → site_name ID.
    Edit the CUSTOM_TEMPLATES dict in main.py to register your templates,
    or use list_sites(label_names='template') to discover them dynamically
    if you label template sites in the Duda dashboard.
    """
    if not CUSTOM_TEMPLATES:
        return {
            "message": (
                "No custom templates registered yet. "
                "Edit the CUSTOM_TEMPLATES dict in main.py, "
                "or use list_sites(label_names='template') if you label "
                "your template sites in the Duda dashboard."
            ),
            "templates": {},
        }
    return {"templates": CUSTOM_TEMPLATES}


# --- BUILD FROM CUSTOM TEMPLATE (main new feature) --------------------------


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
        template_site_name: site_name ID of your custom template site.
                            Use list_custom_templates() or
                            list_sites(label_names='template') to find these.
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
        "Build a new restaurant site from my restaurant template for
         Joe's Diner — domain joe-diner, phone 555-999-0000,
         email info@joesdiner.com, address 88 Oak Street, city Portland,
         region OR, postal 97201."
    """
    # ── Step 1: Duplicate the template ──────────────────────────────────────
    new_site = _call(
        "POST",
        f"/sites/multiscreen/duplicate/{template_site_name}",
        params={"new_default_domain": new_default_domain},
    )
    if "_status_code" in new_site:
        return {
            "error": "Failed to duplicate template site.",
            "detail": new_site.get("raw_response", new_site),
            "_status_code": new_site["_status_code"],
        }

    new_site_name: str = new_site.get("site_name", "")

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
        "template_source": template_site_name,
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


    new_site_name: str = new_site.get("site_name", "")

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