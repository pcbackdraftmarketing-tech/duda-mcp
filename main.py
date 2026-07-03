import base64
import os
from typing import Optional
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# Build Basic Auth header from API credentials
USERNAME = os.environ["DUDA_API_USER"]
PASSWORD = os.environ["DUDA_API_PASS"]
creds = base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()
HEADERS = {
    "Authorization": f"Basic {creds}",
    "Content-Type": "application/json"
}
BASE_URL = "https://api.duda.co/api"

mcp = FastMCP("duda")


# --- SITES ---

@mcp.tool()
def list_sites() -> dict:
    """List all sites in the Duda account."""
    r = requests.get(f"{BASE_URL}/sites/multiscreen", headers=HEADERS)
    return r.json()


@mcp.tool()
def get_site(site_name: str) -> dict:
    """Get details of a specific site by its site name."""
    r = requests.get(f"{BASE_URL}/sites/multiscreen/{site_name}", headers=HEADERS)
    return r.json()


@mcp.tool()
def publish_site(site_name: str) -> dict:
    """Publish a Duda site to make it live."""
    r = requests.post(f"{BASE_URL}/sites/multiscreen/{site_name}/publish", headers=HEADERS)
    return {"status_code": r.status_code, "message": "Published" if r.ok else r.text}


@mcp.tool()
def unpublish_site(site_name: str) -> dict:
    """Unpublish (take offline) a Duda site."""
    r = requests.post(f"{BASE_URL}/sites/multiscreen/{site_name}/unpublish", headers=HEADERS)
    return {"status_code": r.status_code, "message": "Unpublished" if r.ok else r.text}


@mcp.tool()
def duplicate_site(site_name: str, new_default_domain: str) -> dict:
    """Duplicate an existing Duda site."""
    params = {"new_default_domain": new_default_domain}
    r = requests.post(
        f"{BASE_URL}/sites/multiscreen/duplicate/{site_name}",
        headers=HEADERS,
        params=params   # <-- query param, not json body
    )
    return r.json()


# --- TEMPLATES ---

@mcp.tool()
def list_templates() -> dict:
    """
    List all available Duda templates.
    Returns template IDs, names, and preview URLs you can use with create_site_from_template.
    """
    r = requests.get(f"{BASE_URL}/sites/multiscreen/templates", headers=HEADERS)
    return r.json()


@mcp.tool()
def create_site_from_template(
    template_id: int,
    new_default_domain: str,
    lang: str = "en",
    site_data: Optional[dict] = None
) -> dict:
    """
    Create a new Duda site from an official Duda template.
    Args:
        template_id: The numeric template ID (get these from list_templates)
        new_default_domain: Subdomain for the new site (e.g. 'my-new-site')
        lang: Language code for the site, default 'en'
        site_data: Optional dict with initial content, e.g.:
                   {"site_business_info": {"business_name": "Acme Co", "phone": "555-1234"}}
    """
    payload = {
        "template_id": template_id,
        "default_domain_prefix": new_default_domain,
        "lang": lang,
    }
    if site_data:
        payload["site_data"] = site_data

    r = requests.post(
        f"{BASE_URL}/sites/multiscreen/create",
        headers=HEADERS,
        json=payload
    )
    result = r.json()
    if r.ok:
        result["edit_url"] = f"https://dashboard.duda.co/home/site/{result.get('site_name')}"
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
    Useful when you've built a polished site and want to reuse it as a starting point.
    Args:
        template_site_name: site_name of the site to use as a template
        new_default_domain: Subdomain prefix for the new site (e.g. 'client-name')
        business_name: Optional — pre-fill the business name in the content library
        phone: Optional — pre-fill the phone number
        email: Optional — pre-fill the email address
    """
    # Step 1: Duplicate the site
    payload = {"new_default_domain": new_default_domain}
    r = requests.post(
        f"{BASE_URL}/sites/multiscreen/duplicate/{template_site_name}",
        headers=HEADERS,
        json=payload
    )
    if not r.ok:
        return {"error": r.text, "status_code": r.status_code}

    new_site = r.json()
    new_site_name = new_site.get("site_name")

    # Step 2: Optionally update business info in the content library
    if any([business_name, phone, email]):
        business_info = {}
        if business_name:
            business_info["business_name"] = business_name
        if phone:
            business_info["phone"] = phone
        if email:
            business_info["email"] = email

        requests.post(
            f"{BASE_URL}/sites/multiscreen/{new_site_name}/content",
            headers=HEADERS,
            json={"location_data": business_info}
        )

    return {
        "site_name": new_site_name,
        "default_domain": new_site.get("site_default_domain"),
        "edit_url": f"https://dashboard.duda.co/home/site/{new_site_name}",
        "preview_url": f"https://dashboard.duda.co/preview/{new_site_name}",
        "status": "created",
        "template_source": template_site_name,
    }


# --- ANALYTICS ---

@mcp.tool()
def get_site_analytics(site_name: str, from_date: str, to_date: str) -> dict:
    """
    Get traffic analytics for a site.
    Args:
        site_name: The site name/ID
        from_date: Start date in format YYYY-MM-DD
        to_date: End date in format YYYY-MM-DD
    """
    params = {"from": from_date, "to": to_date}
    r = requests.get(
        f"{BASE_URL}/sites/multiscreen/analytics/site/{site_name}",
        headers=HEADERS,
        params=params
    )
    return r.json()


# --- PAGES ---

@mcp.tool()
def list_pages(site_name: str) -> dict:
    """List all pages of a Duda site."""
    r = requests.get(f"{BASE_URL}/sites/multiscreen/{site_name}/pages", headers=HEADERS)
    return r.json()


# --- BLOG ---

@mcp.tool()
def list_blog_posts(site_name: str) -> dict:
    """List all blog posts for a Duda site."""
    r = requests.get(f"{BASE_URL}/sites/multiscreen/{site_name}/blog/posts", headers=HEADERS)
    return r.json()


@mcp.tool()
def create_blog_post(site_name: str, title: str, body: str, status: str = "DRAFT") -> dict:
    """
    Create a blog post on a Duda site.
    Args:
        site_name: The site name/ID
        title: Blog post title
        body: HTML content of the blog post
        status: DRAFT or PUBLISHED
    """
    payload = {"title": title, "body": body, "status": status}
    r = requests.post(
        f"{BASE_URL}/sites/multiscreen/{site_name}/blog/posts",
        headers=HEADERS,
        json=payload
    )
    return r.json()


# --- CONTENT ---

@mcp.tool()
def get_content_library(site_name: str) -> dict:
    """Get the content library of a Duda site (text, images, business info)."""
    r = requests.get(f"{BASE_URL}/sites/multiscreen/{site_name}/content", headers=HEADERS)
    return r.json()


# --- CLIENT ACCOUNTS ---

@mcp.tool()
def list_client_accounts(site_name: str) -> dict:
    """List all client accounts (permissions) for a Duda site."""
    r = requests.get(
        f"{BASE_URL}/accounts/client/{site_name}",
        headers=HEADERS
    )
    return r.json()


if __name__ == "__main__":
    mcp.run()