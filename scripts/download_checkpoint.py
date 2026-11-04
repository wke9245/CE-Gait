"""Download the versioned CE-Gait checkpoint from GitHub Releases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        return None


OPENER = urllib.request.build_opener(NoRedirect())


def open_url(url, headers=None):
    headers = dict(headers or {})
    for _ in range(6):
        try:
            return OPENER.open(urllib.request.Request(url, headers=headers), timeout=90)
        except urllib.error.HTTPError as error:
            if error.code not in {301, 302, 303, 307, 308} or not error.headers.get("Location"):
                raise
            target = urllib.parse.urljoin(url, error.headers["Location"])
            parsed = urllib.parse.urlsplit(target)
            if parsed.scheme != "https":
                raise ValueError("The release returned a non-HTTPS download endpoint")
            if parsed.netloc != urllib.parse.urlsplit(url).netloc:
                headers.pop("Authorization", None)
            url = target
    raise ValueError("Too many download redirects")


def read_json(url, headers=None):
    with open_url(url, headers) as response:
        return json.load(response)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def download(url, destination, digest, size, headers):
    if destination.is_file() and destination.stat().st_size == size and file_digest(destination) == digest:
        print("Verified existing " + destination.name, flush=True)
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="." + destination.name + "-", suffix=".part", delete=False) as target:
            temporary = Path(target.name)
            checksum, count = hashlib.sha256(), 0
            with open_url(url, headers) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    target.write(block)
                    checksum.update(block)
                    count += len(block)
        if count != size or "sha256:" + checksum.hexdigest() != digest:
            raise ValueError("Downloaded file checksum or byte count does not match: " + destination.name)
        temporary.replace(destination)
        temporary = None
        print("Downloaded and verified " + destination.name, flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--token-file", type=Path, help="Read a GitHub access token from a local file")
    args = parser.parse_args()
    if args.token_file:
        matches = re.findall(r"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]+", args.token_file.read_text(encoding="utf-8-sig"))
        if len(matches) != 1:
            parser.error("The token file must contain one GitHub access token")
        token = matches[0]
    else:
        token = (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "").strip()
    if not token:
        parser.error("Set GH_TOKEN or GITHUB_TOKEN to an access token with repository read access, or provide --token-file")
    release = json.loads((ROOT / "checkpoints/release.json").read_text(encoding="utf-8"))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", release["repository"]):
        raise ValueError("Invalid release repository")
    base = "https://api.github.com/repos/" + release["repository"]
    headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "CEGait-Checkpoint-Download"}
    remote = read_json(base + "/releases/" + str(release["release_id"]), headers)
    if remote["tag_name"] != release["tag"]:
        raise ValueError("Release tag does not match the pinned version")
    expected = {item["name"]: item for item in release["assets"]}
    if set(expected) != {"best.pt", "best.json", "SHA256SUMS"}:
        raise ValueError("Unexpected checkpoint release file list")
    actual = {item["id"]: item for item in remote["assets"]}
    for filename, item in expected.items():
        asset = actual.get(item["asset_id"])
        if asset is None or asset["name"] != filename or asset["size"] != item["size"] or asset["state"] != "uploaded":
            raise ValueError("Release file metadata does not match: " + filename)
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Invalid release file checksum")
        if asset.get("digest") and asset["digest"] != "sha256:" + item["sha256"]:
            raise ValueError("Release asset digest does not match: " + filename)
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, item in expected.items():
        print("Retrieving " + filename, flush=True)
        download(base + "/releases/assets/" + str(item["asset_id"]), args.output / filename, "sha256:" + item["sha256"], item["size"], {**headers, "Accept": "application/octet-stream"})
    print("Checkpoint release verified at " + str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as error:
        raise SystemExit("GitHub Releases request failed (HTTP " + str(error.code) + ")")
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit("Checkpoint download failed: " + str(error))
