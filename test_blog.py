"""
test_blog.py
============
Pytest suite for all blog-related functions in main.py.

Mocking strategy: patch `main._call` or `main.session` directly so tests
never touch the Duda API. Functions that bypass _call and use session
directly (create_blog, publish_blog_post, unpublish_blog_post) are mocked
at the session level.
"""

import base64
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out env vars BEFORE importing main.py so it doesn't raise on missing
# credentials.
# ---------------------------------------------------------------------------
os.environ.setdefault("DUDA_API_USER", "test_user")
os.environ.setdefault("DUDA_API_PASS", "test_pass")

sys.path.insert(0, "/mnt/user-data/uploads")
import main  # noqa: E402  (import after env setup)


# ===========================================================================
# Helpers
# ===========================================================================

def make_response(status_code: int, json_data: dict | None = None, text: str = "") -> MagicMock:
    """Return a mock requests.Response."""
    r = MagicMock()
    r.status_code = status_code
    r.ok = 200 <= status_code < 300
    r.text = text
    if json_data is not None:
        r.json.return_value = json_data
    else:
        r.json.side_effect = ValueError("no json")
    return r


# ===========================================================================
# get_site_version
# ===========================================================================

class TestGetSiteVersion:

    def test_returns_classic_for_non_advanced_editor(self):
        with patch.object(main, "_call", return_value={"editor": "CLASSIC_EDITOR"}):
            result = main.get_site_version("abc123")
        assert result["success"] is True
        assert result["editor_version"] == "CLASSIC"
        assert result["editor_raw"] == "CLASSIC_EDITOR"

    def test_returns_2_0_for_advanced(self):
        with patch.object(main, "_call", return_value={"editor": "ADVANCED"}):
            result = main.get_site_version("abc123")
        assert result["success"] is True
        assert result["editor_version"] == "2.0"

    def test_returns_2_0_for_advanced_2_0(self):
        with patch.object(main, "_call", return_value={"editor": "ADVANCED-2.0"}):
            result = main.get_site_version("abc123")
        assert result["success"] is True
        assert result["editor_version"] == "2.0"

    def test_404_returns_not_found_error(self):
        with patch.object(main, "_call", return_value={"_status_code": 404}):
            result = main.get_site_version("bad-site")
        assert result["success"] is False
        assert result["_status_code"] == 404
        assert "not found" in result["error"].lower()

    def test_401_returns_unauthorized_error(self):
        with patch.object(main, "_call", return_value={"_status_code": 401}):
            result = main.get_site_version("abc123")
        assert result["success"] is False
        assert result["_status_code"] == 401
        assert "unauthorized" in result["error"].lower()

    def test_other_non_2xx_returns_generic_error(self):
        with patch.object(main, "_call", return_value={"_status_code": 500, "message": "oops"}):
            result = main.get_site_version("abc123")
        assert result["success"] is False
        assert result["_status_code"] == 500

    def test_missing_editor_field_returns_error(self):
        with patch.object(main, "_call", return_value={"site_name": "abc123"}):
            result = main.get_site_version("abc123")
        assert result["success"] is False
        assert "editor" in result["error"].lower()


# ===========================================================================
# create_blog
# ===========================================================================

class TestCreateBlog:

    def test_success(self):
        mock_r = make_response(200, {})
        with patch.object(main.session, "post", return_value=mock_r):
            result = main.create_blog("abc123")
        assert result["success"] is True
        assert "created" in result["message"].lower()

    def test_409_conflict_treated_as_success(self):
        mock_r = make_response(409, {})
        with patch.object(main.session, "post", return_value=mock_r):
            result = main.create_blog("abc123")
        assert result["success"] is True
        assert "already exists" in result["message"].lower()

    def test_400_resource_already_exist_treated_as_success(self):
        mock_r = make_response(400, {"error_code": "ResourceAlreadyExist"})
        with patch.object(main.session, "post", return_value=mock_r):
            result = main.create_blog("abc123")
        assert result["success"] is True

    def test_other_400_returns_error(self):
        mock_r = make_response(400, {"message": "bad request"})
        with patch.object(main.session, "post", return_value=mock_r):
            result = main.create_blog("abc123")
        assert result["success"] is False

    def test_network_error_returns_error(self):
        import requests as req
        with patch.object(main.session, "post", side_effect=req.RequestException("timeout")):
            result = main.create_blog("abc123")
        assert "error" in result
        assert result["_status_code"] == 500


# ===========================================================================
# import_blog_post
# ===========================================================================

class TestImportBlogPost:

    SITE = "abc123"
    TITLE = "Test Post"
    BODY = "<p>Hello world</p>"

    def test_success_returns_post_id_and_note(self):
        api_response = {"id": "post-001", "title": self.TITLE}
        with patch.object(main, "_call", return_value=api_response) as mock_call:
            result = main.import_blog_post(self.SITE, self.TITLE, self.BODY)
        assert result["id"] == "post-001"
        assert "note" in result
        assert "DRAFT" in result["note"]

    def test_body_is_base64_encoded_in_payload(self):
        with patch.object(main, "_call", return_value={"id": "post-001"}) as mock_call:
            main.import_blog_post(self.SITE, self.TITLE, self.BODY)
        _, kwargs = mock_call.call_args[0], mock_call.call_args[1]
        payload = mock_call.call_args.kwargs.get("json") or mock_call.call_args[1].get("json")
        expected_b64 = base64.b64encode(self.BODY.encode()).decode()
        assert payload["content"] == expected_b64

    def test_description_derived_from_body_when_omitted(self):
        body = "<p>This is the first paragraph of a blog post.</p>"
        with patch.object(main, "_call", return_value={"id": "post-001"}) as mock_call:
            main.import_blog_post(self.SITE, self.TITLE, body)
        payload = mock_call.call_args.kwargs.get("json") or mock_call.call_args[1].get("json")
        assert "first paragraph" in payload["description"]

    def test_explicit_description_used_when_provided(self):
        with patch.object(main, "_call", return_value={"id": "post-001"}) as mock_call:
            main.import_blog_post(self.SITE, self.TITLE, self.BODY, description="My custom desc")
        payload = mock_call.call_args.kwargs.get("json") or mock_call.call_args[1].get("json")
        assert payload["description"] == "My custom desc"

    def test_title_truncated_to_200_chars(self):
        long_title = "A" * 250
        with patch.object(main, "_call", return_value={"id": "post-001"}) as mock_call:
            main.import_blog_post(self.SITE, long_title, self.BODY)
        payload = mock_call.call_args.kwargs.get("json") or mock_call.call_args[1].get("json")
        assert len(payload["title"]) == 200

    def test_author_included_when_provided(self):
        with patch.object(main, "_call", return_value={"id": "post-001"}) as mock_call:
            main.import_blog_post(self.SITE, self.TITLE, self.BODY, author="Jane Doe")
        payload = mock_call.call_args.kwargs.get("json") or mock_call.call_args[1].get("json")
        assert payload["author"] == "Jane Doe"

    def test_author_omitted_when_not_provided(self):
        with patch.object(main, "_call", return_value={"id": "post-001"}) as mock_call:
            main.import_blog_post(self.SITE, self.TITLE, self.BODY)
        payload = mock_call.call_args.kwargs.get("json") or mock_call.call_args[1].get("json")
        assert "author" not in payload

    def test_image_url_included_when_provided(self):
        with patch.object(main, "_call", return_value={"id": "post-001"}) as mock_call:
            main.import_blog_post(self.SITE, self.TITLE, self.BODY,
                                  image_url="https://example.com/img.jpg",
                                  image_alt="A photo")
        payload = mock_call.call_args.kwargs.get("json") or mock_call.call_args[1].get("json")
        assert payload["main_image"]["url"] == "https://example.com/img.jpg"
        assert payload["main_image"]["alt"] == "A photo"

    def test_api_error_returned_as_is(self):
        with patch.object(main, "_call", return_value={"_status_code": 500, "error": "server error"}):
            result = main.import_blog_post(self.SITE, self.TITLE, self.BODY)
        assert result["_status_code"] == 500


# ===========================================================================
# publish_blog_post
# ===========================================================================

class TestPublishBlogPost:

    def test_success(self):
        mock_r = make_response(200, {})
        with patch.object(main.session, "post", return_value=mock_r):
            result = main.publish_blog_post("abc123", "post-001")
        assert result["success"] is True
        assert result["post_id"] == "post-001"

    def test_failure_returns_error(self):
        mock_r = make_response(404, {"message": "post not found"})
        with patch.object(main.session, "post", return_value=mock_r):
            result = main.publish_blog_post("abc123", "bad-post")
        assert result["success"] is False
        assert "404" in result["error"]

    def test_network_error(self):
        import requests as req
        with patch.object(main.session, "post", side_effect=req.RequestException("conn error")):
            result = main.publish_blog_post("abc123", "post-001")
        assert "error" in result


# ===========================================================================
# unpublish_blog_post
# ===========================================================================

class TestUnpublishBlogPost:

    def test_success(self):
        mock_r = make_response(200, {})
        with patch.object(main.session, "post", return_value=mock_r):
            result = main.unpublish_blog_post("abc123", "post-001")
        assert result["success"] is True
        assert "DRAFT" in result["message"]

    def test_failure_returns_error(self):
        mock_r = make_response(404, {"message": "not found"})
        with patch.object(main.session, "post", return_value=mock_r):
            result = main.unpublish_blog_post("abc123", "bad-post")
        assert result["success"] is False

    def test_network_error(self):
        import requests as req
        with patch.object(main.session, "post", side_effect=req.RequestException("timeout")):
            result = main.unpublish_blog_post("abc123", "post-001")
        assert "error" in result


# ===========================================================================
# delete_blog_post
# ===========================================================================

class TestDeleteBlogPost:

    def test_success(self):
        with patch.object(main, "_call", return_value={}):
            result = main.delete_blog_post("abc123", "post-001")
        assert result["deleted"] is True
        assert result["post_id"] == "post-001"

    def test_failure_returns_error(self):
        with patch.object(main, "_call", return_value={"_status_code": 404}):
            result = main.delete_blog_post("abc123", "bad-post")
        assert result["deleted"] is False
        assert "error" in result


# ===========================================================================
# get_blog_post
# ===========================================================================

class TestGetBlogPost:

    def test_success(self):
        api_response = {"id": "post-001", "title": "Hello", "status": "DRAFT"}
        with patch.object(main, "_call", return_value=api_response):
            result = main.get_blog_post("abc123", "post-001")
        assert result["id"] == "post-001"
        assert result["status"] == "DRAFT"

    def test_api_error_passed_through(self):
        with patch.object(main, "_call", return_value={"_status_code": 404}):
            result = main.get_blog_post("abc123", "bad-post")
        assert result["_status_code"] == 404


# ===========================================================================
# update_blog_post
# ===========================================================================

class TestUpdateBlogPost:

    def test_no_params_returns_no_op_message(self):
        result = main.update_blog_post("abc123", "post-001")
        assert "No update parameters" in result["message"]

    def test_sends_only_provided_fields(self):
        with patch.object(main, "_call", return_value={"id": "post-001"}) as mock_call:
            main.update_blog_post("abc123", "post-001", title="New Title")
        payload = mock_call.call_args.kwargs.get("json") or mock_call.call_args[1].get("json")
        assert payload == {"title": "New Title"}

    def test_multiple_fields_sent_together(self):
        with patch.object(main, "_call", return_value={"id": "post-001"}) as mock_call:
            main.update_blog_post("abc123", "post-001",
                                  title="New Title",
                                  description="New desc",
                                  author_name="Jane")
        payload = mock_call.call_args.kwargs.get("json") or mock_call.call_args[1].get("json")
        assert payload["title"] == "New Title"
        assert payload["description"] == "New desc"
        assert payload["author_name"] == "Jane"

    def test_success_sets_fields_updated(self):
        with patch.object(main, "_call", return_value={"id": "post-001"}):
            result = main.update_blog_post("abc123", "post-001", title="New Title")
        assert result["fields_updated"] == ["title"]

    def test_api_error_returns_error(self):
        with patch.object(main, "_call", return_value={"_status_code": 404}):
            result = main.update_blog_post("abc123", "post-001", title="New Title")
        assert "error" in result


# ===========================================================================
# create_blog_post
# ===========================================================================

class TestCreateBlogPost:

    def test_invalid_status_returns_error(self):
        result = main.create_blog_post("abc123", "Title", "<p>body</p>", status="INVALID")
        assert "error" in result
        assert "INVALID" in result["error"]

    def test_plain_text_body_adds_warning(self):
        with patch.object(main, "_call", return_value={"id": "post-001"}):
            result = main.create_blog_post("abc123", "Title", "plain text no tags")
        assert "body_warning" in result

    def test_plain_text_with_auto_wrap_wraps_in_p_tags(self):
        with patch.object(main, "_call", return_value={"id": "post-001"}) as mock_call:
            main.create_blog_post("abc123", "Title", "line one\nline two",
                                  auto_wrap_plain_text=True)
        payload = mock_call.call_args.kwargs.get("json") or mock_call.call_args[1].get("json")
        assert "<p>line one</p>" in payload["body"]
        assert "<p>line two</p>" in payload["body"]

    def test_html_body_sent_as_is(self):
        body = "<p>Hello <strong>world</strong></p>"
        with patch.object(main, "_call", return_value={"id": "post-001"}) as mock_call:
            main.create_blog_post("abc123", "Title", body)
        payload = mock_call.call_args.kwargs.get("json") or mock_call.call_args[1].get("json")
        assert payload["body"] == body

    def test_draft_status_passed_in_payload(self):
        with patch.object(main, "_call", return_value={"id": "post-001"}) as mock_call:
            main.create_blog_post("abc123", "Title", "<p>body</p>", status="DRAFT")
        payload = mock_call.call_args.kwargs.get("json") or mock_call.call_args[1].get("json")
        assert payload["status"] == "DRAFT"
