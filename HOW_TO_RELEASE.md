# How to issue a Pyxel release

~~Time required: ???~~

These instructions assume that `upstream` refers to the main repository:

```fish
$ git remote -v
{...}
upstream	https://gitlab.com/esa/pyxel.git (fetch)
upstream	https://gitlab.com/esa/pyxel.git (push)
```

1. Write a release summary: ~50 words describing the high level features. This will be used in the release emails, GitLab release notes, blog, etc.

1. Ensure your master branch is synced to upstream:

   ```fish
   $ git switch master
   $ git pull upstream master
   ```

1. Create a branch 'new_release{X}.{Y}' from 'master' for the new release.

   ```fish
   $ git checkout -b new_release{X}.{Y}
   ```

1. Open a merge request linked to the new release branch with the release summary and changes.

1. Update release notes in `CHANGELOG.rst` in the branch and add release summary at the top.

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

1. Tag the release from https://gitlab.com/esa/pyxel/-/tags with the following actions:

   a. Click the button 'new_tag'

   b. In the field 'Tag name', enter `v{X.Y}`

   c. In the field 'Create from', choose `master` (this is the default value)

   d. In the field 'Message', enter `v{X.Y}`

   e. In the field 'Release notes', enter the content of `CHANGELOG.rst` only for this release.

1. Push your changes to master:
   ```fish
   $ git push upstream master
   $ git push upstream --tags
   ```
   :interrobang: This could be done directly inside Gitlab

1. Add a section for the next release {X:Y+1} to `CHANGELOG.rst`

    ```fish
    version {X:Y+1} / 2020-MM-DD
    ----------------------------

    Core
    ~~~~

    Documentation
    ~~~~~~~~~~~~~
    ```

1. Commit you changes and push to master again:
    ```fish
    $ git commit -am "New Changelog section"
    $ git push upstream master
    ```

1. Issue the release on GitLab.
   Click on https://gitlab.com/esa/pyxel/-/releases . Type in the version number and paste the release summary in the notes.

1. Issue the release announcement to the mailing list pyxel-dev@googlegroups.com and to the Pyxel blog.
