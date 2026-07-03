import base64
import os
import sys
import requests
import json
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.environ["DUDA_API_USER"]
PASSWORD = os.environ["DUDA_API_PASS"]
creds = base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()
HEADERS = {
    "Authorization": f"Basic {creds}",
    "Content-Type": "application/json"
}
BASE_URL = "https://api.duda.co/api"

site_name = sys.argv[1]  # ← reads from terminal
response = requests.get(f"{BASE_URL}/sites/multiscreen/{site_name}", headers=HEADERS)
print(json.dumps(response.json(), indent=2))