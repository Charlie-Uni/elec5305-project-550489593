# Publishing the project website

The `Publish verified project website` workflow publishes the `main` branch to the
existing GitHub Pages address. Pushes to `main` and manual workflow dispatches use
the same process. The old `gh-pages` branch is retained as a historical snapshot;
it is no longer the publishing source once the repository uses GitHub Actions.

`scripts/prepare_pages.py` copies the page's tracked resources into `_site`:
the HTML and JSON, both listening collections, report and presentation deliveries,
documentation, and the displayed figure collections. It preserves these resource paths,
checks all four targets and 48 conditions, verifies the SHA-256 and WAV format of
all 96 playback files, and checks the source hashes of the explanatory context.
Generated `deployment.json` identifies the exact deployed commit and audio hashes.
Large intermediate trial tables and unrelated historical outputs stay in GitHub;
the page's research-documentation links open the repository, where their relative
links remain valid. Context source hashes refer to those repository files, and
are checked there before packaging. Models, local datasets and environments are
untracked and are not packaged.

The earlier branch-based publication reached `syncing_files` but was cancelled by
the deploy action's ten-minute timeout. A larger configured timeout was tried, but
the official action caps it at ten minutes; increasing that input does not extend
the limit. The workflow instead reduces the publication payload, serialises releases
and validates the artifact before uploading. A completed build alone does not prove publication: check the public
`deployment.json`, page contents and actual audio playback after deployment.

Local packaging check, from the repository root:

```sh
python3 scripts/prepare_pages.py --output /path/to/an/empty/directory
```

Workflow dependencies are pinned to verified official action commits. See the
[custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
and [deploy-pages inputs](https://github.com/actions/deploy-pages#inputs-).
