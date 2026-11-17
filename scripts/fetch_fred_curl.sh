#!/usr/bin/env bash
# Fetch missing FRED-MD vintages with plain curl (its normal, honest client identity).
# One file at a time, 25 s apart. An HTML answer (content-type text/html) means that
# vintage is not published at this URL -> logged and skipped. Any real connection
# failure / unexpected status stops the run so we never hammer the host.
cd "$(dirname "$0")/.." || exit 1
DIR=data/raw/fred_md
mkdir -p "$DIR"
BASE="https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly"
absent=()
for v in 2025-01 2025-02 2025-03 2025-04 2025-05 2025-07 2025-08 2025-09 2025-10 2025-11 2026-04 2026-06 2026-07; do
  f="$DIR/$v-md.csv"
  [ -s "$f" ] && { echo "$v already have"; continue; }
  tmp="$DIR/.$v.tmp"
  meta=$(curl -s -L --max-time 90 -o "$tmp" -w "%{http_code} %{content_type}" "$BASE/$v-md.csv")
  code=${meta%% *}; ctype=${meta#* }
  if [ "$code" = "200" ] && head -c 7 "$tmp" | grep -q "sasdate" && [ "$(wc -l < "$tmp")" -gt 100 ]; then
    mv "$tmp" "$f"; echo "$v OK ($(wc -c < "$f") bytes)"
  elif [ "$code" = "200" ] && echo "$ctype" | grep -qi "text/html"; then
    echo "$v NOT PUBLISHED at this URL (HTML page) -> skipped"; absent+=("$v"); rm -f "$tmp"
  else
    echo "$v FAILED http=$code type=$ctype -> stopping"; rm -f "$tmp"; exit 2
  fi
  sleep 25
done
echo "done: $(ls $DIR/*-md.csv | wc -l) vintages; not published: ${absent[*]:-none}"
