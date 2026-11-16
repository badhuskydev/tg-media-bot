#!/bin/sh
# Keep yt-dlp current without rebuilding the image: sites change often and a
# pinned yt-dlp silently breaks. Best-effort — failure falls back to the
# bundled version. Disable with YTDLP_AUTO_UPDATE=false.
set -e

if [ "${YTDLP_AUTO_UPDATE:-true}" = "true" ]; then
    echo "Updating yt-dlp to the latest release..."
    pip install --no-cache-dir --upgrade "yt-dlp[default,curl-cffi]" \
        || echo "yt-dlp update failed; continuing with the bundled version."
fi

# Same for gallery-dl (image-post fallback). Follows YTDLP_AUTO_UPDATE unless
# GALLERYDL_AUTO_UPDATE is set explicitly.
if [ "${GALLERYDL_AUTO_UPDATE:-${YTDLP_AUTO_UPDATE:-true}}" = "true" ]; then
    echo "Updating gallery-dl to the latest release..."
    pip install --no-cache-dir --upgrade gallery-dl \
        || echo "gallery-dl update failed; continuing with the bundled version."
fi

exec python main.py
