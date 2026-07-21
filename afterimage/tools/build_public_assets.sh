#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUBLIC="$ROOT/public"
LANDSCAPE="$PUBLIC/assets/source/key-art-master-final.png"
PORTRAIT="$PUBLIC/assets/source/poster-art-master.png"
GENERATED="$PUBLIC/assets/generated"
SOCIAL="$PUBLIC/assets/social"
BRAND="$PUBLIC/assets/brand"

command -v magick >/dev/null || { echo "public-assets: ImageMagick 'magick' is required" >&2; exit 2; }
test -f "$LANDSCAPE" || { echo "public-assets: missing landscape master" >&2; exit 2; }
test -f "$PORTRAIT" || { echo "public-assets: missing portrait master" >&2; exit 2; }

mkdir -p "$GENERATED" "$SOCIAL"

magick "$LANDSCAPE" -strip -resize '1920x1080^' -gravity center -extent 1920x1080 \
  -quality 88 -define webp:method=6 "$GENERATED/hero-1920x1080.webp"
magick "$LANDSCAPE" -strip -resize '2400x1350^' -gravity center -extent 2400x1350 \
  -sampling-factor 4:2:0 -quality 91 "$GENERATED/hero-2400x1350.jpg"
magick "$LANDSCAPE" -strip -quality 90 -define webp:method=6 \
  "$GENERATED/key-art-editorial-1536x1024.webp"
magick "$PORTRAIT" -strip -resize '1080x1920^' -gravity center -extent 1080x1920 \
  -quality 89 -define webp:method=6 "$GENERATED/poster-1080x1920.webp"
magick "$PORTRAIT" -strip -resize '1600x2000^' -gravity center -extent 1600x2000 \
  -quality 89 -define webp:method=6 "$GENERATED/poster-1600x2000.webp"
magick "$PORTRAIT" -strip -resize '2480x3508^' -gravity center -extent 2480x3508 \
  -sampling-factor 4:2:0 -quality 92 "$GENERATED/poster-a4-blank.jpg"

magick -background none "$BRAND/mark.svg" -resize 512x512 "$GENERATED/mark-512.png"
magick -background none "$BRAND/wordmark-light.svg" -resize 1600x369 \
  -gravity center -extent 1600x369 "$GENERATED/wordmark-light-1600.png"
magick -background none "$BRAND/wordmark-dark.svg" -resize 1600x369 \
  -gravity center -extent 1600x369 "$GENERATED/wordmark-dark-1600.png"

make_social_set() {
  local locale="$1"
  local font="$2"
  local headline="$3"
  local kicker="$4"

  magick "$LANDSCAPE" -strip -resize '1200x630^' -gravity center -extent 1200x630 \
    -fill '#08131fdd' -draw 'rectangle 0,0 790,630' \
    -fill '#f7f4eb' -font Noto-Sans-Bold -pointsize 38 -gravity northwest -annotate +70+65 'AFTERIMAGE' \
    -fill '#a7bac7' -font Noto-Sans-Medium -pointsize 17 -annotate +72+112 'THE COUNTERFACTUAL CITY' \
    -fill '#f7f4eb' -font "$font" -pointsize 58 -interline-spacing 10 -annotate +70+230 "$headline" \
    -fill '#63c7d8' -font "$font" -pointsize 22 -annotate +72+540 "$kicker" \
    -quality 94 "$SOCIAL/afterimage-og-$locale.png"

  magick "$LANDSCAPE" -strip -resize '1080x1080^' -gravity east -extent 1080x1080 \
    -fill '#08131f99' -draw 'rectangle 0,0 1080,1080' \
    -fill '#08131fee' -draw 'rectangle 0,0 1080,395' \
    -fill '#f7f4eb' -font Noto-Sans-Bold -pointsize 45 -gravity northwest -annotate +72+65 'AFTERIMAGE' \
    -fill '#f7f4eb' -font "$font" -pointsize 55 -interline-spacing 10 -annotate +72+175 "$headline" \
    -fill '#63c7d8' -font "$font" -pointsize 21 -gravity southwest -annotate +72+65 "$kicker" \
    -quality 94 "$SOCIAL/afterimage-square-$locale.png"

  magick "$PORTRAIT" -strip -resize '1080x1920^' -gravity center -extent 1080x1920 \
    -fill '#08131fdd' -draw 'rectangle 0,0 1080,560' \
    -fill '#08131fbb' -draw 'rectangle 0,1660 1080,1920' \
    -fill '#f7f4eb' -font Noto-Sans-Bold -pointsize 48 -gravity northwest -annotate +72+70 'AFTERIMAGE' \
    -fill '#a7bac7' -font Noto-Sans-Medium -pointsize 19 -annotate +74+128 'THE COUNTERFACTUAL CITY' \
    -fill '#f7f4eb' -font "$font" -pointsize 62 -interline-spacing 12 -annotate +72+270 "$headline" \
    -fill '#63c7d8' -font "$font" -pointsize 24 -gravity southwest -annotate +72+96 "$kicker" \
    -quality 94 "$SOCIAL/afterimage-story-$locale.png"
}

make_social_set \
  en Noto-Sans-Bold \
  $'Build a causal evaluator.\nDebug tomorrow.' \
  '75 CASES · 8 FAMILIES · OFFLINE'
make_social_set \
  ja Noto-Sans-CJK-JP-Bold \
  $'因果評価器を実装し、\n明日をデバッグせよ。' \
  '全75問 · 8ファミリー · オフライン'
make_social_set \
  zh-Hans Noto-Sans-CJK-SC-Bold \
  $'构建因果求值器，\n调试明天。' \
  '75道题 · 8个系列 · 离线运行'
make_social_set \
  de Noto-Sans-Bold \
  $'Implementiere einen kausalen\nAuswerter. Debugge morgen.' \
  '75 AUFGABEN · 8 FAMILIEN · OFFLINE'

python3 "$ROOT/tools/render_public_previews.py"
python3 "$ROOT/tools/build_public_manifest.py"

echo "public-assets: PASS: editorial, brand, social, site-preview, and press assets"
