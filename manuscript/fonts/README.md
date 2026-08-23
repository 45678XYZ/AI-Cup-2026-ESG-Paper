# Droid Sans Fallback AI CUP subset

`DroidSansFallback-ai-cup-subset.ttf` is a glyph subset of
`DroidSansFallbackFull.ttf` from Debian/Ubuntu package `fonts-droid-fallback`
version `1:6.0.1r16-1.1build1` (source package `fonts-android`). The upstream
project is the Android Open Source Project:
<https://android.googlesource.com/platform/frameworks/base/>.

Exact upstream files at tag `android-6.0.1_r16`:

- Font: <https://android.googlesource.com/platform/frameworks/base/+/android-6.0.1_r16/data/fonts/DroidSansFallbackFull.ttf>
- NOTICE: <https://android.googlesource.com/platform/frameworks/base/+/android-6.0.1_r16/data/fonts/NOTICE>

- Upstream copyright: Copyright 2006--2010 Google Corp.
- License: Apache License 2.0; see `LICENSE-APACHE-2.0.txt`.
- Applicable upstream NOTICE attribution: see `NOTICE-AOSP.txt`.
- Upstream SHA-256:
  `acb6440a713d880a13a21b468ba7cd43f5a2b2934972e51be791c880730777b8`.
- Subset SHA-256:
  `6ef7774b36575a9fdea9a39ca606ac8677595b8e75eb1b941c336824eb50c9ed`.
- Included Unicode characters: space and `競賽提交格式說明` (U+0020, U+7AF6,
  U+8CFD, U+63D0, U+4EA4, U+683C, U+5F0F, U+8AAA, U+660E).

The file was mechanically subset with fontTools `pyftsubset`. It differs from
the upstream font by retaining only the glyphs required to reproduce the exact
official guide title in the manuscript bibliography.
