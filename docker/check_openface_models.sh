#!/bin/bash
# OpenFace's download_models.sh fetches its CEN patch experts from Dropbox without "?dl=1", so it can
# save a Dropbox HTML page instead of the model, and FeatureExtraction then aborts with
# std::bad_alloc. Re-download each file that is not a model, and fail if one still is not.
# Usage: check_openface_models.sh <OpenFace install path>
set -euo pipefail
dir="$1/build/bin/model/patch_experts"
[ -d "$dir" ] || { echo "error: $dir not found" >&2; exit 1; }
is_html() { head -c 200 "$1" | grep -qi "<!doctype\|<html"; }
for f in 7na5qsjzz8yfoer/cen_patches_0.25_of.dat k7bj804cyiu474t/cen_patches_0.35_of.dat \
         ixt4vkbmxgab1iu/cen_patches_0.50_of.dat 2t5t1sdpshzfhpj/cen_patches_1.00_of.dat; do
    path="$dir/$(basename "$f")"
    if [ ! -s "$path" ] || is_html "$path"; then
        echo "re-downloading $(basename "$f")"
        wget -q -O "$path.part" "https://www.dropbox.com/s/$f?dl=1"
        mv "$path.part" "$path"
    fi
    # The real files are 60 to 155 MB; an error page is a few hundred kB.
    if is_html "$path" || [ "$(stat -c %s "$path")" -lt 10000000 ]; then
        echo "error: $path is not an OpenFace model" >&2
        exit 1
    fi
done
echo "OpenFace CEN models OK"
