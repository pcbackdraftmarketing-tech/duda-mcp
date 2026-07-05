# duda-mcp

A Python-based MCP (Model Context Protocol) integration for the Duda website platform. Used by Backdraft Marketing to interact with the Duda API directly from Claude — enabling site creation, content updates, and business info management without manual intervention.

---

## What It Does

- Connects Claude to the Duda API via MCP
- Creates new client sites from a master template
- Pushes business info (name, phone, email, socials) to Duda
- Updates the Duda Content Library with AI-generated copy
- Supports multiple site management via site name identifiers

---

## Project Structure

```
duda-mcp/
├── main.py            # Main MCP server entry point
├── templates.yaml     # Duda template configuration
├── test.py            # Manual test scripts
├── pyproject.toml     # Project metadata and dependencies
├── .python-version    # Python version pin
└── .env               # Local secrets (never committed)
```

---

## Setup

```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment and install dependencies
uv sync

# Copy env template and fill in your credentials
cp .env.example .env
```

---

## Environment Variables

Store these in `.env` locally — never commit them.

| Variable | Description |
|---|---|
| `DUDA_API_USER` | Duda API username |
| `DUDA_API_PASS` | Duda API password |

---

## Key Site IDs

| Label | Site Name |
|---|---|
| Master template | `dfe8c020` |
| Test site | `8ef881b1` |

---

## Related

- [duda-webhook-lambda](../duda-webhook-lambda) — AWS Lambda webhook that handles form submissions and triggers site builds
