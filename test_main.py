"""
test_main.py
============
Full pytest suite for all 30 public MCP tool functions in main.py.

Mocking strategy:
  - Functions that use _call()         → patch main._call
  - Functions that use session directly → patch main.session.post / .request
  - No real HTTP calls are made; no real credentials are needed.
"""

import base64
import json
import os
import sys
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Stub credentials before importing main so it doesn't raise EnvironmentError
# ---------------------------------------------------------------------------
os.environ.setdefault("DUDA_API_USER", "test_user")
os.environ.setdefault("DUDA_API_PASS", "test_pass")

sys.path.insert(0, "/mnt/user-data/uploads")
import main  # noqa: E402


# ===========================================================================
# Shared helpers
# ===========================================================================

def ok_response(json_data: dict | None = None, text: str = "") -> MagicMock:
    """Fake a successful requests.Response."""
    r = MagicMock()
    r.status_code = 200
    r.ok = True
    r.text = text
    if json_data is not None:
        r.json.return_value = json_data
    else:
        r.json.side_effect = ValueError("no json")
    return r


def err_response(status: int, json_data: dict | None = None, text: str = "") -> MagicMock:
    """Fake a failed requests.Response."""
    r = MagicMock()
    r.status_code = status
    r.ok = False
    r.text = text
    if json_data is not None:
        r.json.return_value = json_data
    else:
        r.json.side_effect = ValueError("no json")
    return r


def call_ok(data: dict = None) -> dict:
    """Fake a successful _call() return."""
    return data or {"ok": True}


def call_err(status: int = 500, msg: str = "error") -> dict:
    """Fake a failed _call() return (includes _status_code)."""
    return {"_status_code": status, "message": msg}


# ===========================================================================
# list_sites
# ===========================================================================

class TestListSites:

    def test_basic_call_passes_default_params(self):
        with patch.object(main, "_call", return_value=call_ok({"results": []})) as m:
            main.list_sites()
        params = m.call_args.kwargs["params"]
        assert params["limit"] == 75
        assert params["offset"] == 0

    def test_limit_clamped_to_100(self):
        with patch.object(main, "_call", return_value=call_ok()) as m:
            main.list_sites(limit=999)
        assert m.call_args.kwargs["params"]["limit"] == 100

    def test_offset_clamped_to_zero(self):
        with patch.object(main, "_call", return_value=call_ok()) as m:
            main.list_sites(offset=-5)
        assert m.call_args.kwargs["params"]["offset"] == 0

    def test_optional_filters_included_when_provided(self):
        with patch.object(main, "_call", return_value=call_ok()) as m:
            main.list_sites(label_names="template", publish_status="PUBLISHED", site_type="REGULAR")
        params = m.call_args.kwargs["params"]
        assert params["label_names"] == "template"
        assert params["publish_status"] == "PUBLISHED"
        assert params["site_type"] == "REGULAR"

    def test_optional_filters_omitted_when_not_provided(self):
        with patch.object(main, "_call", return_value=call_ok()) as m:
            main.list_sites()
        params = m.call_args.kwargs["params"]
        assert "label_names" not in params
        assert "publish_status" not in params
        assert "site_type" not in params


# ===========================================================================
# get_site
# ===========================================================================

class TestGetSite:

    def test_returns_site_data(self):
        data = {"site_name": "abc123", "editor": "ADVANCED-2.0"}
        with patch.object(main, "_call", return_value=data):
            result = main.get_site("abc123")
        assert result["site_name"] == "abc123"

    def test_api_error_passed_through(self):
        with patch.object(main, "_call", return_value=call_err(404)):
            result = main.get_site("bad")
        assert result["_status_code"] == 404


# ===========================================================================
# publish_site
# ===========================================================================

class TestPublishSite:

    def test_success(self):
        with patch.object(main, "_call", return_value={}):
            result = main.publish_site("abc123")
        assert result["published"] is True
        assert result["site_name"] == "abc123"

    def test_failure(self):
        with patch.object(main, "_call", return_value=call_err(500)):
            result = main.publish_site("abc123")
        assert result["published"] is False
        assert "error" in result


# ===========================================================================
# unpublish_site
# ===========================================================================

class TestUnpublishSite:

    def test_success(self):
        with patch.object(main, "_call", return_value={}):
            result = main.unpublish_site("abc123")
        assert result["unpublished"] is True

    def test_failure(self):
        with patch.object(main, "_call", return_value=call_err(500)):
            result = main.unpublish_site("abc123")
        assert result["unpublished"] is False


# ===========================================================================
# delete_site
# ===========================================================================

class TestDeleteSite:

    def test_success(self):
        with patch.object(main, "_call", return_value={}):
            result = main.delete_site("abc123")
        assert result["deleted"] is True
        assert result["site_name"] == "abc123"

    def test_failure(self):
        with patch.object(main, "_call", return_value=call_err(404)):
            result = main.delete_site("abc123")
        assert result["deleted"] is False


# ===========================================================================
# duplicate_site
# ===========================================================================

class TestDuplicateSite:

    def test_passes_correct_params(self):
        with patch.object(main, "_call", return_value={"site_name": "new-site"}) as m:
            main.duplicate_site("abc123", "new-domain")
        assert "/duplicate/abc123" in m.call_args.args[1]
        assert m.call_args.kwargs["params"]["new_default_domain"] == "new-domain"

    def test_api_error_passed_through(self):
        with patch.object(main, "_call", return_value=call_err(500)):
            result = main.duplicate_site("abc123", "new-domain")
        assert result["_status_code"] == 500


# ===========================================================================
# list_templates
# ===========================================================================

class TestListTemplates:

    def test_returns_templates(self):
        data = {"results": [{"template_id": 1, "name": "Business"}]}
        with patch.object(main, "_call", return_value=data):
            result = main.list_templates()
        assert "results" in result


# ===========================================================================
# create_site_from_template
# ===========================================================================

class TestCreateSiteFromTemplate:

    def test_success_adds_edit_url(self):
        with patch.object(main, "_call", return_value={"site_name": "new123"}):
            result = main.create_site_from_template(1234, "my-domain")
        assert "edit_url" in result
        assert "new123" in result["edit_url"]

    def test_site_data_included_in_payload(self):
        site_data = {"site_business_info": {"business_name": "Acme"}}
        with patch.object(main, "_call", return_value={"site_name": "new123"}) as m:
            main.create_site_from_template(1234, "my-domain", site_data=site_data)
        payload = m.call_args.kwargs["json"]
        assert payload["site_data"] == site_data

    def test_api_error_no_edit_url(self):
        with patch.object(main, "_call", return_value=call_err(400)):
            result = main.create_site_from_template(1234, "my-domain")
        assert "edit_url" not in result

    def test_default_lang_is_en(self):
        with patch.object(main, "_call", return_value={"site_name": "new123"}) as m:
            main.create_site_from_template(1234, "my-domain")
        payload = m.call_args.kwargs["json"]
        assert payload["lang"] == "en"


# ===========================================================================
# list_custom_templates
# ===========================================================================

class TestListCustomTemplates:

    def test_returns_all_when_no_filter(self):
        main.CUSTOM_TEMPLATES = {
            "plumbing": {"industry": "plumbing", "variants": [{"site_name": "abc"}]},
            "hvac": {"industry": "hvac", "variants": [{"site_name": "def"}]},
        }
        result = main.list_custom_templates()
        assert result["total_collections"] == 2
        assert "collections" in result

    def test_filters_by_industry(self):
        main.CUSTOM_TEMPLATES = {
            "plumbing": {"industry": "plumbing", "variants": [{"site_name": "abc"}]},
            "hvac": {"industry": "hvac", "variants": [{"site_name": "def"}]},
        }
        result = main.list_custom_templates(industry="plumbing")
        assert result["total_collections"] == 1
        assert "plumbing" in result["collections"]

    def test_no_match_returns_available_industries(self):
        main.CUSTOM_TEMPLATES = {
            "plumbing": {"industry": "plumbing", "variants": [{"site_name": "abc"}]},
        }
        result = main.list_custom_templates(industry="roofing")
        assert "available_industries" in result
        assert result["templates"] == {}

    def test_empty_registry_returns_message(self):
        main.CUSTOM_TEMPLATES = {}
        result = main.list_custom_templates()
        assert "message" in result
        assert result["templates"] == {}


# ===========================================================================
# reload_custom_templates
# ===========================================================================

class TestReloadCustomTemplates:

    def test_reload_success(self):
        new_templates = {
            "roofing": {"industry": "roofing", "variants": [{"site_name": "xyz"}]}
        }
        with patch.object(main, "_load_templates", return_value=new_templates):
            result = main.reload_custom_templates()
        assert result["success"] is True
        assert result["total_collections"] == 1
        assert "roofing" in result["collections"]

    def test_reload_failure_leaves_registry_unchanged(self):
        main.CUSTOM_TEMPLATES.clear()
        main.CUSTOM_TEMPLATES["existing"] = {
            "industry": "hvac", "variants": [{"site_name": "abc"}]
        }
        with patch.object(main, "_load_templates", side_effect=ValueError("bad yaml")):
            result = main.reload_custom_templates()
        assert result["success"] is False
        assert "bad yaml" in result["error"]
        assert "existing" in main.CUSTOM_TEMPLATES  # registry left unchanged


# ===========================================================================
# update_site_content
# ===========================================================================

class TestUpdateSiteContent:

    def test_no_fields_returns_no_op(self):
        result = main.update_site_content("abc123")
        assert result["content_updated"] is False
        assert "No content fields" in result["message"]

    def test_success_returns_fields_sent(self):
        with patch.object(main, "_update_content", return_value=(True, None)):
            result = main.update_site_content("abc123", business_name="Acme")
        assert result["content_updated"] is True
        assert "fields_sent" in result
        assert len(result["fields_sent"]) > 0

    def test_update_failure_returns_error(self):
        with patch.object(main, "_update_content", return_value=(False, "API error")):
            result = main.update_site_content("abc123", business_name="Acme")
        assert result["content_updated"] is False
        assert "error" in result

    def test_publish_content_triggers_publish(self):
        with patch.object(main, "_update_content", return_value=(True, None)), \
             patch.object(main, "_publish_content", return_value=(True, None)) as mp:
            main.update_site_content("abc123", business_name="Acme", publish_content=True)
        mp.assert_called_once_with("abc123")

    def test_publish_failure_adds_warning(self):
        with patch.object(main, "_update_content", return_value=(True, None)), \
             patch.object(main, "_publish_content", return_value=(False, "pub failed")):
            result = main.update_site_content("abc123", business_name="Acme", publish_content=True)
        assert "publish_warning" in result


# ===========================================================================
# create_site_from_existing
# ===========================================================================

class TestCreateSiteFromExisting:

    def test_success_includes_template_source(self):
        with patch.object(main, "_duplicate_and_inject", return_value={"site_name": "new123"}):
            result = main.create_site_from_existing("tmpl123", "new-domain")
        assert result["template_source"] == "tmpl123"

    def test_duplicate_error_propagated(self):
        with patch.object(main, "_duplicate_and_inject",
                          return_value={"_error": True, "error": "dup failed"}):
            result = main.create_site_from_existing("tmpl123", "new-domain")
        assert "error" in result
        assert "_error" not in result


# ===========================================================================
# create_client_account
# ===========================================================================

class TestCreateClientAccount:

    def test_success(self):
        with patch.object(main, "_call", side_effect=[{}, {}]):
            result = main.create_client_account("abc123", "client@example.com")
        assert result["success"] is True
        assert result["account_name"] == "client@example.com"

    def test_409_account_exists_still_grants_permissions(self):
        with patch.object(main, "_call", side_effect=[
            {"_status_code": 409},
            {},
        ]):
            result = main.create_client_account("abc123", "client@example.com")
        assert result["success"] is True

    def test_400_already_exists_still_grants_permissions(self):
        with patch.object(main, "_call", side_effect=[
            {"_status_code": 400, "message": "account already exists"},
            {},
        ]):
            result = main.create_client_account("abc123", "client@example.com")
        assert result["success"] is True

    def test_400_other_error_returns_error(self):
        with patch.object(main, "_call", return_value={"_status_code": 400, "message": "bad input"}):
            result = main.create_client_account("abc123", "client@example.com")
        assert "error" in result

    def test_permissions_failure_returns_error(self):
        with patch.object(main, "_call", side_effect=[
            {},
            {"_status_code": 403},
        ]):
            result = main.create_client_account("abc123", "client@example.com")
        assert "error" in result

    def test_optional_name_fields_included(self):
        with patch.object(main, "_call", side_effect=[{}, {}]) as m:
            main.create_client_account("abc123", "c@example.com",
                                       first_name="Jane", last_name="Doe")
        payload = m.call_args_list[0].kwargs["json"]
        assert payload["first_name"] == "Jane"
        assert payload["last_name"] == "Doe"


# ===========================================================================
# get_content_collection_link
# ===========================================================================

class TestGetContentCollectionLink:

    def test_success_returns_link(self):
        with patch.object(main, "_call", return_value={"url": "https://example.com/form"}):
            result = main.get_content_collection_link("abc123", "client@example.com")
        assert result["link"] == "https://example.com/form"
        assert result["site_name"] == "abc123"

    def test_api_error_returns_error(self):
        with patch.object(main, "_call", return_value=call_err(404)):
            result = main.get_content_collection_link("abc123", "client@example.com")
        assert "error" in result


# ===========================================================================
# get_site_analytics
# ===========================================================================

class TestGetSiteAnalytics:

    def test_valid_dates_calls_api(self):
        with patch.object(main, "_call", return_value={"visits": 100}) as m:
            result = main.get_site_analytics("abc123", "2024-01-01", "2024-01-31")
        assert result["visits"] == 100

    def test_invalid_from_date_returns_error(self):
        result = main.get_site_analytics("abc123", "01-01-2024", "2024-01-31")
        assert "error" in result
        assert "from_date" in result["error"]

    def test_invalid_to_date_returns_error(self):
        result = main.get_site_analytics("abc123", "2024-01-01", "31/01/2024")
        assert "error" in result
        assert "to_date" in result["error"]

    def test_reversed_dates_returns_error(self):
        result = main.get_site_analytics("abc123", "2024-12-31", "2024-01-01")
        assert "error" in result
        assert "after" in result["error"]

    def test_range_over_365_days_returns_error(self):
        result = main.get_site_analytics("abc123", "2022-01-01", "2024-01-01")
        assert "error" in result
        assert "365" in result["error"]


# ===========================================================================
# list_pages
# ===========================================================================

class TestListPages:

    def test_returns_pages(self):
        data = {"results": [{"page_name": "Home"}]}
        with patch.object(main, "_call", return_value=data):
            result = main.list_pages("abc123")
        assert "results" in result

    def test_api_error_passed_through(self):
        with patch.object(main, "_call", return_value=call_err(404)):
            result = main.list_pages("abc123")
        assert result["_status_code"] == 404


# ===========================================================================
# list_blog_posts
# ===========================================================================

class TestListBlogPosts:

    def test_returns_posts(self):
        data = {"results": [{"id": "post-001", "title": "Hello"}]}
        with patch.object(main, "_call", return_value=data):
            result = main.list_blog_posts("abc123")
        assert result["results"][0]["id"] == "post-001"

    def test_api_error_passed_through(self):
        with patch.object(main, "_call", return_value=call_err(404)):
            result = main.list_blog_posts("abc123")
        assert result["_status_code"] == 404


# ===========================================================================
# get_content_library
# ===========================================================================

class TestGetContentLibrary:

    def test_returns_library(self):
        data = {"business_data": {"business_name": "Acme"}}
        with patch.object(main, "_call", return_value=data):
            result = main.get_content_library("abc123")
        assert "business_data" in result

    def test_api_error_passed_through(self):
        with patch.object(main, "_call", return_value=call_err(404)):
            result = main.get_content_library("abc123")
        assert result["_status_code"] == 404


# ===========================================================================
# list_client_accounts
# ===========================================================================

class TestListClientAccounts:

    def test_returns_accounts(self):
        data = {"results": [{"account_name": "client@example.com"}]}
        with patch.object(main, "_call", return_value=data):
            result = main.list_client_accounts("abc123")
        assert result["results"][0]["account_name"] == "client@example.com"


# ===========================================================================
# get_site_version
# ===========================================================================

class TestGetSiteVersion:

    def test_classic(self):
        with patch.object(main, "_call", return_value={"editor": "CLASSIC"}):
            result = main.get_site_version("abc123")
        assert result["editor_version"] == "CLASSIC"

    def test_advanced_is_2_0(self):
        with patch.object(main, "_call", return_value={"editor": "ADVANCED"}):
            result = main.get_site_version("abc123")
        assert result["editor_version"] == "2.0"

    def test_advanced_2_0_is_2_0(self):
        with patch.object(main, "_call", return_value={"editor": "ADVANCED-2.0"}):
            result = main.get_site_version("abc123")
        assert result["editor_version"] == "2.0"

    def test_404(self):
        with patch.object(main, "_call", return_value=call_err(404)):
            result = main.get_site_version("bad")
        assert result["success"] is False
        assert result["_status_code"] == 404

    def test_401(self):
        with patch.object(main, "_call", return_value=call_err(401)):
            result = main.get_site_version("abc123")
        assert result["success"] is False
        assert "unauthorized" in result["error"].lower()

    def test_missing_editor_field(self):
        with patch.object(main, "_call", return_value={"site_name": "abc123"}):
            result = main.get_site_version("abc123")
        assert result["success"] is False


# ===========================================================================
# create_blog
# ===========================================================================

class TestCreateBlog:

    def test_success(self):
        with patch.object(main.session, "post", return_value=ok_response({})):
            result = main.create_blog("abc123")
        assert result["success"] is True

    def test_409_is_success(self):
        with patch.object(main.session, "post", return_value=err_response(409, {})):
            result = main.create_blog("abc123")
        assert result["success"] is True

    def test_400_resource_already_exist_is_success(self):
        with patch.object(main.session, "post",
                          return_value=err_response(400, {"error_code": "ResourceAlreadyExist"})):
            result = main.create_blog("abc123")
        assert result["success"] is True

    def test_other_400_is_failure(self):
        with patch.object(main.session, "post",
                          return_value=err_response(400, {"message": "bad"})):
            result = main.create_blog("abc123")
        assert result["success"] is False

    def test_network_error(self):
        import requests as req
        with patch.object(main.session, "post", side_effect=req.RequestException("err")):
            result = main.create_blog("abc123")
        assert "error" in result


# ===========================================================================
# import_blog_post
# ===========================================================================

class TestImportBlogPost:

    SITE = "abc123"
    TITLE = "My Post"
    BODY = "<p>Hello</p>"

    def test_success_has_note(self):
        with patch.object(main, "_call", return_value={"id": "p1"}):
            result = main.import_blog_post(self.SITE, self.TITLE, self.BODY)
        assert "note" in result

    def test_body_base64_encoded(self):
        with patch.object(main, "_call", return_value={"id": "p1"}) as m:
            main.import_blog_post(self.SITE, self.TITLE, self.BODY)
        payload = m.call_args.kwargs["json"]
        assert payload["content"] == base64.b64encode(self.BODY.encode()).decode()

    def test_description_derived_when_omitted(self):
        body = "<p>Auto generated description content here.</p>"
        with patch.object(main, "_call", return_value={"id": "p1"}) as m:
            main.import_blog_post(self.SITE, self.TITLE, body)
        payload = m.call_args.kwargs["json"]
        assert len(payload["description"]) > 0

    def test_explicit_description_used(self):
        with patch.object(main, "_call", return_value={"id": "p1"}) as m:
            main.import_blog_post(self.SITE, self.TITLE, self.BODY, description="Custom desc")
        payload = m.call_args.kwargs["json"]
        assert payload["description"] == "Custom desc"

    def test_title_truncated_to_200(self):
        with patch.object(main, "_call", return_value={"id": "p1"}) as m:
            main.import_blog_post(self.SITE, "X" * 250, self.BODY)
        payload = m.call_args.kwargs["json"]
        assert len(payload["title"]) == 200

    def test_author_included(self):
        with patch.object(main, "_call", return_value={"id": "p1"}) as m:
            main.import_blog_post(self.SITE, self.TITLE, self.BODY, author="Jane")
        payload = m.call_args.kwargs["json"]
        assert payload["author"] == "Jane"

    def test_author_omitted_when_not_provided(self):
        with patch.object(main, "_call", return_value={"id": "p1"}) as m:
            main.import_blog_post(self.SITE, self.TITLE, self.BODY)
        payload = m.call_args.kwargs["json"]
        assert "author" not in payload

    def test_image_url_and_alt(self):
        with patch.object(main, "_call", return_value={"id": "p1"}) as m:
            main.import_blog_post(self.SITE, self.TITLE, self.BODY,
                                  image_url="https://img.com/x.jpg", image_alt="Alt")
        payload = m.call_args.kwargs["json"]
        assert payload["main_image"]["url"] == "https://img.com/x.jpg"
        assert payload["main_image"]["alt"] == "Alt"

    def test_api_error_passed_through(self):
        with patch.object(main, "_call", return_value=call_err(500)):
            result = main.import_blog_post(self.SITE, self.TITLE, self.BODY)
        assert result["_status_code"] == 500


# ===========================================================================
# publish_blog_post
# ===========================================================================

class TestPublishBlogPost:

    def test_success(self):
        with patch.object(main.session, "post", return_value=ok_response({})):
            result = main.publish_blog_post("abc123", "p1")
        assert result["success"] is True
        assert result["post_id"] == "p1"

    def test_failure(self):
        with patch.object(main.session, "post", return_value=err_response(404, {"msg": "nf"})):
            result = main.publish_blog_post("abc123", "p1")
        assert result["success"] is False

    def test_network_error(self):
        import requests as req
        with patch.object(main.session, "post", side_effect=req.RequestException("err")):
            result = main.publish_blog_post("abc123", "p1")
        assert "error" in result


# ===========================================================================
# create_blog_post
# ===========================================================================

class TestCreateBlogPost:

    def test_invalid_status_returns_error(self):
        result = main.create_blog_post("abc123", "Title", "<p>body</p>", status="INVALID")
        assert "error" in result

    def test_plain_text_adds_warning(self):
        with patch.object(main, "_call", return_value={"id": "p1"}):
            result = main.create_blog_post("abc123", "Title", "no html here")
        assert "body_warning" in result

    def test_auto_wrap_wraps_lines(self):
        with patch.object(main, "_call", return_value={"id": "p1"}) as m:
            main.create_blog_post("abc123", "Title", "line1\nline2", auto_wrap_plain_text=True)
        payload = m.call_args.kwargs["json"]
        assert "<p>line1</p>" in payload["body"]
        assert "<p>line2</p>" in payload["body"]

    def test_html_body_sent_as_is(self):
        body = "<h2>Title</h2><p>content</p>"
        with patch.object(main, "_call", return_value={"id": "p1"}) as m:
            main.create_blog_post("abc123", "Title", body)
        payload = m.call_args.kwargs["json"]
        assert payload["body"] == body

    def test_status_passed_in_payload(self):
        with patch.object(main, "_call", return_value={"id": "p1"}) as m:
            main.create_blog_post("abc123", "Title", "<p>body</p>", status="PUBLISHED")
        payload = m.call_args.kwargs["json"]
        assert payload["status"] == "PUBLISHED"


# ===========================================================================
# update_blog_post
# ===========================================================================

class TestUpdateBlogPost:

    def test_no_params_no_op(self):
        result = main.update_blog_post("abc123", "p1")
        assert "No update parameters" in result["message"]

    def test_only_provided_fields_sent(self):
        with patch.object(main, "_call", return_value={"id": "p1"}) as m:
            main.update_blog_post("abc123", "p1", title="New")
        payload = m.call_args.kwargs["json"]
        assert list(payload.keys()) == ["title"]

    def test_fields_updated_in_response(self):
        with patch.object(main, "_call", return_value={"id": "p1"}):
            result = main.update_blog_post("abc123", "p1", title="New", description="Desc")
        assert set(result["fields_updated"]) == {"title", "description"}

    def test_api_error_returns_error(self):
        with patch.object(main, "_call", return_value=call_err(404)):
            result = main.update_blog_post("abc123", "p1", title="New")
        assert "error" in result

    def test_all_optional_fields_sent(self):
        with patch.object(main, "_call", return_value={"id": "p1"}) as m:
            main.update_blog_post("abc123", "p1",
                                  title="T", description="D", author_name="A",
                                  meta_title="M", path="/slug", no_index=True,
                                  tags=["tag1"], publish_date="2024-01-01T00:00:00")
        payload = m.call_args.kwargs["json"]
        assert payload["title"] == "T"
        assert payload["author_name"] == "A"
        assert payload["no_index"] is True
        assert payload["tags"] == ["tag1"]


# ===========================================================================
# unpublish_blog_post
# ===========================================================================

class TestUnpublishBlogPost:

    def test_success(self):
        with patch.object(main.session, "post", return_value=ok_response({})):
            result = main.unpublish_blog_post("abc123", "p1")
        assert result["success"] is True
        assert "DRAFT" in result["message"]

    def test_failure(self):
        with patch.object(main.session, "post", return_value=err_response(404, {})):
            result = main.unpublish_blog_post("abc123", "p1")
        assert result["success"] is False

    def test_network_error(self):
        import requests as req
        with patch.object(main.session, "post", side_effect=req.RequestException("err")):
            result = main.unpublish_blog_post("abc123", "p1")
        assert "error" in result


# ===========================================================================
# delete_blog_post
# ===========================================================================

class TestDeleteBlogPost:

    def test_success(self):
        with patch.object(main, "_call", return_value={}):
            result = main.delete_blog_post("abc123", "p1")
        assert result["deleted"] is True

    def test_failure(self):
        with patch.object(main, "_call", return_value=call_err(404)):
            result = main.delete_blog_post("abc123", "p1")
        assert result["deleted"] is False


# ===========================================================================
# get_blog_post
# ===========================================================================

class TestGetBlogPost:

    def test_success(self):
        with patch.object(main, "_call", return_value={"id": "p1", "status": "DRAFT"}):
            result = main.get_blog_post("abc123", "p1")
        assert result["id"] == "p1"

    def test_api_error(self):
        with patch.object(main, "_call", return_value=call_err(404)):
            result = main.get_blog_post("abc123", "bad")
        assert result["_status_code"] == 404
