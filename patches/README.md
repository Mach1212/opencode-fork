# Patches

Each folder holds one change of this fork. The first line of `title` names it in the release
notes. Each `.patch` file is a `git diff` of the same change against the release it was
written for.

The folders and the patch files in each folder are tried in name order. The first patch in
a folder that applies without conflicts, leaves every package it touches passing its
typecheck, and passes the test files it changes is carried. If none fits, the build fails
and names the folder. With no folders, the build packages upstream's archive unchanged.
