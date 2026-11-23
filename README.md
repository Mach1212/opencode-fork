# OpenCode 1.x, with what upstream's releases do not have

This is a fork of [anomalyco/opencode](https://github.com/anomalyco/opencode). It exists to run
stable OpenCode 1.x releases with changes that those releases do not have. Upstream takes no
pull requests for 1.x, so the changes are carried here.

## What it publishes

For a stable 1.x release of upstream, a [release here](../../releases) named
`<upstream tag>-t3kube.<n>` with:

- `opencode-linux-x64-baseline.tar.gz`: built from that release's source plus the changes in
  [`patches/`](patches), with `opencode` at the root of the archive, as in upstream's archive.
- `SHA256SUMS` for it.

Its notes list what that build carries. [`builds.md`](builds.md) lists the builds.
When nothing is carried, the release holds upstream's archive unchanged.

[Mach1212/t3kube](https://github.com/Mach1212/t3kube) pins a build in `machine/nix/sources.nix`.

## What it carries

- `patches/compaction-fits/`: Compaction that fits the model's context window: the summary is
  first asked for with the conversation's own request, which reuses its prompt cache, while
  that request leaves room for the summary. Otherwise, or when that reply is unusable, the
  history goes as a text transcript without the model's reasoning, shortened oldest first to
  fit, and shortened further each time the model refuses it as too long.

## How it follows upstream

[A workflow](.github/workflows/follow-upstream.yml) runs every three hours. It finds upstream's
newest stable 1.x release and, if this fork has not tried it with the current `patches/` and
`build.sh`, runs [`build.sh`](build.sh) and publishes the result. Only the build of upstream's
newest 1.x release is marked latest.

If no patch in a folder fits the release, that run fails, leaves a draft, and opens an issue
assigned to the owner. A build that fails for another reason is recorded the same way. Earlier
builds stay as they are.

## Changing what it carries

- **Carry another change**: add a folder in [`patches/`](patches) with a `title` file and a
  `.patch` file from `git diff` against the release it is for.
- **A change does not fit a new release**: add another `.patch` file to its folder with the
  change written for that release.
- **Stop carrying a change**: delete its folder.

A change to `patches/` or `build.sh` makes the next run try the newest 1.x release again, as
build `n+1`. To build another 1.x release, run the workflow by hand with its tag.

## Branches

`t3kube` holds what makes the builds. `dev` is where upstream's `dev` was when the fork was
made and is not updated. The `fix/…` branches hold changes on upstream's code.
