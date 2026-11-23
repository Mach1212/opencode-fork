#!/usr/bin/env bash
# Builds one stable OpenCode 1.x release with the changes in patches/. With nothing to
# carry, the result is upstream's own Linux x64 baseline archive, unchanged.
#
#   build.sh <tag> <out-dir>        for example: build.sh v1.19.6 out
#
# Writes opencode-linux-x64-baseline.tar.gz, SHA256SUMS and notes.md to <out-dir>.
# Exit 3 means a patch does not fit this release; any other failure is the build's.
#
# Needs curl, jq and tar. To build from source, also needs git, the release's Bun, Node 22
# or newer, python3, make, and a C/C++ compiler for native dependencies.
# GH_TOKEN, when set, signs the calls to GitHub's API.
set -euo pipefail
export LC_ALL=C
shopt -s nullglob

tag=${1:?usage: build.sh <tag> <out-dir>}
[[ $tag =~ ^v1\.[0-9]+\.[0-9]+$ ]] || { echo "Unexpected OpenCode 1.x release tag: $tag" >&2; exit 1; }
mkdir -p "${2:?usage: build.sh <tag> <out-dir>}"
out=$(cd "$2" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
upstream=anomalyco/opencode
version=${tag#v}
stem=opencode-linux-x64-baseline
archive=$stem.tar.gz
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

api() {
  curl -fsSL ${GH_TOKEN:+-H "Authorization: Bearer $GH_TOKEN"} \
    -H "Accept: application/vnd.github+json" "https://api.github.com/$1"
}
say() { printf '\n== %s\n' "$*"; }

# A patch stands when the packages it touches still typecheck and its test files pass.
stands() {
  local path dir paths
  local -a dirs=() tests=()
  local -A seen=()
  git diff --cached --name-only --no-renames -z > "$work/paths" || return 1
  mapfile -d '' -t paths < "$work/paths"
  for path in "${paths[@]}"; do
    [[ $path == packages/*/* ]] || continue
    dir=${path#packages/}
    dir=packages/${dir%%/*}
    if [[ ! -v 'seen[$dir]' ]]; then
      seen[$dir]=1
      dirs+=("$dir")
    fi
  done
  for dir in "${dirs[@]}"; do
    echo "$dir: typecheck"
    (cd "$dir" && bun run typecheck) || return 1
    tests=()
    for path in "${paths[@]}"; do
      if [[ $path == "$dir"/test/*.test.ts && -f $path ]]; then
        tests+=("./${path#"$dir/"}")
      fi
    done
    if ((${#tests[@]})); then
      (cd "$dir" && bun test --timeout 30000 "${tests[@]}") || return 1
    fi
  done
}

dirs=("$here"/patches/*/)
printf 'Upstream release: [%s](https://github.com/%s/releases/tag/%s).\n\n' \
  "$tag" "$upstream" "$tag" > "$work/notes.md"

if ((${#dirs[@]} == 0)); then
  say "nothing to carry: upstream's archive, checked against its release digest"
  digest=$(api "repos/$upstream/releases/tags/$tag" |
    jq -er --arg archive "$archive" '.assets[] | select(.name == $archive) | .digest')
  [[ $digest =~ ^sha256:[0-9a-f]{64}$ ]] || { echo "No SHA-256 digest for $archive in $tag." >&2; exit 1; }
  curl -fsSL -o "$work/$archive" "https://github.com/$upstream/releases/download/$tag/$archive"
  (cd "$work" && printf '%s  %s\n' "${digest#sha256:}" "$archive" | sha256sum -c -)
  echo "Upstream's own archive of $tag, unchanged: there is nothing to carry." >> "$work/notes.md"
else
  say "the source of $tag"
  git clone --quiet --depth 1 --branch "$tag" "https://github.com/$upstream.git" "$work/src"
  cd "$work/src"
  git config user.name "opencode-fork build"
  git config user.email "build@opencode-fork.invalid"

  say "Bun as the release names it, and the release's dependencies"
  bun_wanted=$(jq -er '.packageManager | capture("^bun@(?<version>[^+]+)").version' package.json)
  bun_here=$(bun --version)
  if [ "$bun_here" != "$bun_wanted" ]; then
    echo "$tag builds with Bun $bun_wanted; this is $bun_here." >&2
    exit 1
  fi
  bun install --frozen-lockfile

  echo "Built from upstream's $tag with these changes:" >> "$work/notes.md"
  echo >> "$work/notes.md"
  for dir in "${dirs[@]}"; do
    name=$(basename "$dir")
    title=$(head -n 1 "$dir/title")
    say "patches/$name: $title"
    fitted=
    for patch in "$dir"*.patch; do
      : > "$work/check.log"
      if git apply --index --3way "$patch" > "$work/apply.log" 2>&1 &&
        [ -z "$(git diff --name-only --diff-filter=U)" ] && stands > "$work/check.log" 2>&1; then
        fitted=$(basename "$patch")
        break
      fi
      echo "$(basename "$patch") does not fit:"
      tail -n 20 "$work/apply.log" "$work/check.log"
      git reset --quiet --hard
      git clean --quiet -fd
    done
    if [ -z "$fitted" ]; then
      echo "No patch in patches/$name fits $tag. Add one for this release." >&2
      exit 3
    fi
    git -c commit.gpgsign=false commit --quiet --no-verify -m "carry patches/$name ($fitted)"
    echo "carried as $fitted"
    printf -- "- %s (\`patches/%s\`)\n" "$title" "$name" >> "$work/notes.md"
  done

  say "the Linux x64 baseline executable, from $tag with ${#dirs[@]} change(s)"
  # With only OPENCODE_VERSION set, upstream picks latest, which opens opencode.db.
  # A preview channel would hide existing sessions, and OPENCODE_RELEASE would upload to GitHub.
  unset OPENCODE_CHANNEL OPENCODE_RELEASE
  cd packages/opencode
  OPENCODE_VERSION=$version bun run script/build.ts --single --baseline || exit 1

  say "the archive, packaged as upstream packages it"
  tar -czf "$work/$archive" -C "dist/$stem/bin" opencode

  say "the archived executable's version and database"
  mkdir -p "$work"/{unpacked,home,data,config,state,cache}
  tar -xzf "$work/$archive" -C "$work/unpacked"
  export HOME="$work/home" XDG_DATA_HOME="$work/data" XDG_CONFIG_HOME="$work/config"
  export XDG_STATE_HOME="$work/state" XDG_CACHE_HOME="$work/cache"
  unset OPENCODE_DB OPENCODE_DISABLE_CHANNEL_DB
  actual=$("$work/unpacked/opencode" --version) || exit 1
  if [ "$actual" != "$version" ]; then
    echo "Expected version $version; the archive prints $actual." >&2
    exit 1
  fi
  (cd "$work/unpacked" && ./opencode session list) || exit 1
  databases=("$XDG_DATA_HOME"/opencode/opencode*.db)
  if ((${#databases[@]} != 1)) || [ ! -f "$XDG_DATA_HOME/opencode/opencode.db" ]; then
    echo "Expected only opencode.db in $XDG_DATA_HOME/opencode; found: ${databases[*]:-none}." >&2
    exit 1
  fi
  echo "$actual opens opencode.db."
fi

cp "$work/$archive" "$out/$archive"
cp "$work/notes.md" "$out/notes.md"
(cd "$out" && sha256sum "$archive" > SHA256SUMS)
say "done: $(cat "$out/SHA256SUMS")"
