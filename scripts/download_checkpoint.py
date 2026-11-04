"""Download the versioned CE-Gait checkpoint from GitHub Packages."""
import argparse
import base64
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
                raise ValueError("The registry returned a non-HTTPS download endpoint")
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
        parser.error("Set GH_TOKEN or GITHUB_TOKEN to an access token with package read access, or provide --token-file")
    package = json.loads((ROOT / "checkpoints/package.json").read_text(encoding="utf-8"))
    if package["registry"] != "ghcr.io":
        raise ValueError("Unsupported package registry")
    account = read_json("https://api.github.com/user", {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json", "User-Agent": "CEGait-Checkpoint-Download"})
    name = package["namespace"] + "/" + package["name"]
    basic = base64.b64encode((account["login"] + ":" + token).encode()).decode("ascii")
    query = urllib.parse.urlencode({"service": "ghcr.io", "scope": "repository:" + name + ":pull"})
    grant = read_json("https://ghcr.io/token?" + query, {"Authorization": "Basic " + basic})
    registry_token = grant.get("token") or grant["access_token"]
    headers = {"Authorization": "Bearer " + registry_token, "User-Agent": "CEGait-Checkpoint-Download"}
    manifest_url = "https://ghcr.io/v2/" + name + "/manifests/" + package["digest"]
    with open_url(manifest_url, {**headers, "Accept": "application/vnd.oci.image.manifest.v1+json"}) as response:
        body = response.read()
    if "sha256:" + hashlib.sha256(body).hexdigest() != package["digest"]:
        raise ValueError("Package manifest digest does not match the pinned version")
    manifest = json.loads(body)
    expected = {item["name"]: item for item in package["files"]}
    if set(expected) != {"best.pt", "best.json"}:
        raise ValueError("Unexpected checkpoint package file list")
    actual = {}
    for layer in manifest["layers"]:
        filename = layer.get("annotations", {}).get("org.opencontainers.image.title")
        if filename not in expected or filename in actual:
            raise ValueError("Unexpected file in the checkpoint package")
        if layer["digest"] != expected[filename]["digest"] or layer["size"] != expected[filename]["size"]:
            raise ValueError("Package file metadata does not match: " + filename)
        actual[filename] = layer
    if set(actual) != set(expected):
        raise ValueError("Checkpoint package is incomplete")
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, layer in actual.items():
        print("Retrieving " + filename, flush=True)
        download("https://ghcr.io/v2/" + name + "/blobs/" + layer["digest"], args.output / filename, layer["digest"], layer["size"], headers)
    print("Checkpoint package verified at " + str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as error:
        raise SystemExit("GitHub Packages request failed (HTTP " + str(error.code) + ")")
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit("Checkpoint download failed: " + str(error))
