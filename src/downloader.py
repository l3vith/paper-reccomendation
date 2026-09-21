"""
Module for downloading research paper PDFs using aria2 (accelerated multi-connection).
Falls back gracefully to requests streaming if aria2 is unavailable.
"""

import os
import shutil
import logging
import subprocess
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def find_aria2_executable() -> Optional[str]:
    """
    Locate the aria2c executable in PATH or standard installation locations.

    Returns:
        Full path to aria2c.exe or None if not found.
    """
    # 1. Check PATH
    which_path = shutil.which("aria2c")
    if which_path:
        return which_path

    # 2. Check WinGet standard package location
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        winget_base = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages")
        if os.path.exists(winget_base):
            for root, _, files in os.walk(winget_base):
                if "aria2c.exe" in files:
                    return os.path.join(root, "aria2c.exe")

    # 3. Check Program Files
    program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
    candidate = os.path.join(program_files, "aria2", "aria2c.exe")
    if os.path.exists(candidate):
        return candidate

    return None


def download_paper_aria2(
    url: str,
    filename: Optional[str] = None,
    output_dir: str = "downloads",
    max_connections: int = 4
) -> Tuple[bool, str, int]:
    """
    Download a research paper PDF using aria2c with multi-connection acceleration.

    Args:
        url: Direct URL to the paper PDF or arXiv page.
        filename: Optional desired filename (e.g. '2304.00501.pdf').
        output_dir: Directory where the file will be saved.
        max_connections: Number of simultaneous connections per server.

    Returns:
        Tuple of (success: bool, filepath_or_message: str, file_size_bytes: int)
    """
    # Clean up arXiv URL to ensure it points to the direct PDF
    if "arxiv.org/abs/" in url:
        url = url.replace("/abs/", "/pdf/")
    if "arxiv.org/pdf/" in url and not url.endswith(".pdf"):
        url = f"{url}.pdf"

    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        # Generate filename from URL or default
        base = url.split("?")[0].rstrip("/").split("/")[-1]
        filename = base if base.endswith(".pdf") else f"{base}.pdf"

    # Sanitize filename
    safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
    target_path = os.path.join(output_dir, safe_filename)

    # Check if already downloaded and non-empty
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        return True, target_path, os.path.getsize(target_path)

    aria2_path = find_aria2_executable()

    if aria2_path:
        try:
            cmd = [
                aria2_path,
                "-x", str(max_connections),
                "-s", str(max_connections),
                "-k", "1M",
                "--dir", os.path.abspath(output_dir),
                "--out", safe_filename,
                "--auto-file-renaming=false",
                "--allow-overwrite=true",
                "--timeout=30",
                url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                return True, target_path, os.path.getsize(target_path)
            else:
                logger.warning(f"aria2c returned code {result.returncode}: {result.stderr}")
        except Exception as e:
            logger.warning(f"aria2c execution failed: {e}. Falling back to requests.")

    # Fallback to requests if aria2 is unavailable or failed
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PaperDownloader/1.0"}
        resp = requests.get(url, headers=headers, stream=True, timeout=30)
        if resp.status_code == 200:
            with open(target_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                return True, target_path, os.path.getsize(target_path)
        return False, f"HTTP status {resp.status_code}", 0
    except Exception as e:
        return False, str(e), 0


def extract_text_from_pdf(pdf_path: str, max_pages: int = 10) -> str:
    """
    Extract readable text from a downloaded PDF file using pypdf.

    Args:
        pdf_path: Local path to the PDF.
        max_pages: Maximum number of initial pages to extract.

    Returns:
        Extracted plain text of the paper.
    """
    if not os.path.exists(pdf_path):
        return ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages = []
        for i in range(min(len(reader.pages), max_pages)):
            txt = reader.pages[i].extract_text()
            if txt and txt.strip():
                pages.append(txt.strip())
        return "\n\n".join(pages)
    except Exception as e:
        logger.warning(f"Failed to extract text from {pdf_path}: {e}")
        return ""

