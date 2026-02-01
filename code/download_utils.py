import http.cookiejar
import os
import re
import urllib.parse
import urllib.request

try:
    import wget
except ImportError:
    wget = None


def _ensure_parent_dir(file_path):
    parent = os.path.dirname(file_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def is_google_drive_url(url):
    return "drive.google.com" in (url or "")


def extract_google_drive_file_id(url):
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.endswith("drive.google.com"):
        match = re.search(r"/d/([^/]+)", parsed.path)
        if match:
            return match.group(1)
        query = urllib.parse.parse_qs(parsed.query or "")
        file_ids = query.get("id", [])
        if file_ids:
            return file_ids[0]
    return None


def _save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as file_handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            file_handle.write(chunk)


def download_from_google_drive(file_id, output_path):
    _ensure_parent_dir(output_path)
    base_url = "https://drive.google.com/uc?export=download"
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    response = opener.open(f"{base_url}&id={file_id}")
    confirm_token = None
    for cookie in cookie_jar:
        if cookie.name.startswith("download_warning"):
            confirm_token = cookie.value
            break
    if confirm_token:
        response = opener.open(f"{base_url}&id={file_id}&confirm={confirm_token}")
    _save_response_content(response, output_path)


def download_file(url, output_path):
    _ensure_parent_dir(output_path)
    if is_google_drive_url(url):
        file_id = extract_google_drive_file_id(url)
        if not file_id:
            raise ValueError(f"Unable to extract Google Drive file id from: {url}")
        download_from_google_drive(file_id, output_path)
        return
    if wget is None:
        raise ImportError("The 'wget' package is required to download non-Google Drive URLs.")
    wget.download(url, output_path)
