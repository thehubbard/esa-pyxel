# How to issue a Pyxel release

Time required: ???

These instructions assume that `upstream` refers to the main repository:

```fish
$ git remote -v
{...}
upstream	https://gitlab.com/esa/pyxel.git (fetch)
upstream	https://gitlab.com/esa/pyxel.git (push)
```

1. Ensure your master branch is synced to upstream:

   ```fish
   $ git switch master
   $ git pull upstream master
   ```

1. Confirm there is no commits on stable that are not yet merged.

   ```fish
   $ git merge upstream stable
   ```

1. Write a release summary: ~50 words describing the high level features. This will be used in the release emails, GitLab release notes, blog, etc.

1. Update release notes in `CHANGELOG.rst` and add release summary at the top.

1. If possible, open a merge request with the release summary and changes.

   :interrobang: The user should work in a branch and not master

1. After merging, again ensure your master branch is synced to upstream:

   ```fish
   $ git pull upstream master
   ```

1. If you have any doubts, run the full test suite one final time !

   ```fish
   $ pytest

   or

   $ tox
   ```

1. On the master branch, commit the release in Git:

   ```fish
   $ git commit -am 'Release v{X.Y}'
   ```

   :interrobang: This could be done directly inside Gitlab

1. Tag the release

   ```fish
   $ git tag -a v{X.Y} -m 'v{X.Y}'
   ```

   :interrobang: This could be done directly inside Gitlab

1. Push your changes to master:
   ```fish
   $ git push upstream master
   $ git push upstream --tags
   ```
   :interrobang: This could be done directly inside Gitlab
