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
# Internal helper
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
        offset: Zero-based offset for pagination. Increment by `limit` to get
                the next page. Check `total_responses` in the result to know
                when you've fetched all sites.
        label_names: Optional comma-separated label names to filter by
                     (e.g. 'template,restaurant'). Multiple values are OR'd.
        publish_status: Optional filter — one of PUBLISHED, UNPUBLISHED, or
                        NOT_PUBLISHED_YET. Multiple values comma-separated.

    Example — fetch all sites in batches:
        Page 1: list_sites(limit=100, offset=0)
        Page 2: list_sites(limit=100, offset=100)
        Page 3: list_sites(limit=100, offset=200)
        Stop when len(results) < limit or offset >= total_responses.
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
    return {
        "status_code": r.status_code,
        "message": "Published" if r.ok else r.text,
    }


@mcp.tool()
def unpublish_site(site_name: str) -> dict:
    """Unpublish (take offline) a Duda site."""
    r = session.post(
        f"{BASE_URL}/sites/multiscreen/{site_name}/unpublish",
        timeout=DEFAULT_TIMEOUT,
    )
    return {
        "status_code": r.status_code,
        "message": "Unpublished" if r.ok else r.text,
    }


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
    List all available Duda templates.
    Returns template IDs, names, and preview URLs you can use with
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


@mcp.tool()
def create_site_from_existing(
    template_site_name: str,
    new_default_domain: str,
    business_name: Optional[str] = None,
    phone: Optional[str] = None,
    email: Optional[str] = None,
) -> dict:
    """
    Generate a new site by duplicating an existing site as a template.
    Useful when you've built a polished site and want to reuse it as a
    starting point.

    Args:
        template_site_name: site_name of the site to use as a template.
        new_default_domain: Subdomain prefix for the new site (e.g. 'client-name').
        business_name: Optional — pre-fill the business name in the content library.
        phone: Optional — pre-fill the phone number (e.g. '555-123-4567').
        email: Optional — pre-fill the email address.
    """
    # Step 1: Duplicate the site.
    # new_default_domain must be sent as a query param, not a JSON body.
    new_site = _call(
        "POST",
        f"/sites/multiscreen/duplicate/{template_site_name}",
        params={"new_default_domain": new_default_domain},
    )
    if "_status_code" in new_site:
        return {"error": new_site.get("raw_response", "Unknown error"), **new_site}

    new_site_name = new_site.get("site_name")

    # Step 2: Optionally update business info in the content library.
    # The content update endpoint returns 204 No Content on success (no JSON body).
    # - business_name goes under business_data.name (not location_data)
    # - phone goes under location_data.phones as an array of {phoneNumber, label}
    # - email goes under location_data.emails as an array of {emailAddress, label}
    content_warning: Optional[str] = None
    if any([business_name, phone, email]):
        payload: dict = {}

        if business_name:
            payload["business_data"] = {"name": business_name}

        location_data: dict = {}
        if phone:
            location_data["phones"] = [{"phoneNumber": phone, "label": "main"}]
        if email:
            location_data["emails"] = [{"emailAddress": email, "label": "main"}]
        if location_data:
            payload["location_data"] = location_data

        r = session.post(
            f"{BASE_URL}/sites/multiscreen/{new_site_name}/content",
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
        # Success is 204 No Content — no JSON to parse
        if not r.ok:
            try:
                err_body = r.json()
            except ValueError:
                err_body = r.text
            content_warning = (
                f"Site created but content update failed "
                f"(HTTP {r.status_code}): {err_body}"
            )

    response = {
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


# --- CUSTOM TEMPLATE REGISTRY -----------------------------------------------

# Edit this dict to map friendly names → site_name IDs of your template sites.
# Run list_sites() to find the site_name for each template site you've built.
# Naming convention tip: label your template sites in Duda with a "template"
# label so list_sites(label_names="template") returns them automatically.
CUSTOM_TEMPLATES: dict[str, str] = {
    # "restaurant": "abc12345",
    # "law-firm":   "def67890",
    # "real-estate": "ghi11111",
}


@mcp.tool()
def list_custom_templates() -> dict:
    """
    List your saved custom site templates (sites you've built and designated
    as reusable starting points).

    Returns a mapping of friendly template name → site_name ID.
    Edit the CUSTOM_TEMPLATES dict in main.py to register your templates.

    Tip: you can also use list_sites(label_names='template') to discover
    template sites dynamically if you label them in the Duda dashboard.
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
    """Get the content library of a Duda site (text, images, business info)."""
    return _call("GET", f"/sites/multiscreen/{site_name}/content")


# --- CLIENT ACCOUNTS --------------------------------------------------------


@mcp.tool()
def list_client_accounts(site_name: str) -> dict:
    """List all client accounts (permissions) for a Duda site."""
    return _call("GET", f"/accounts/client/{site_name}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()