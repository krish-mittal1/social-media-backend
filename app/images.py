from dotenv import load_dotenv
from imagekitio import ImageKit
import os
import logging
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger(__name__)

# FIX: Add validation for required ImageKit environment variables
# This prevents silent failures and provides clear error messages
private_key = os.getenv("IMAGEKIT_PRIVATE_KEY")
public_key = os.getenv("IMAGEKIT_PUBLIC_KEY")
url_endpoint = os.getenv("IMAGEKIT_URL_ENDPOINT")

if not all([private_key, public_key, url_endpoint]):
    logger.warning(
        "ImageKit credentials not fully configured. "
        "Set IMAGEKIT_PRIVATE_KEY, IMAGEKIT_PUBLIC_KEY, and IMAGEKIT_URL_ENDPOINT environment variables. "
        "File uploads will fail until configured."
    )

imagekit = ImageKit(
    private_key=private_key or "",
    public_key=public_key or "",
    url_endpoint=url_endpoint or "",
)
