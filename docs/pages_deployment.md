# Publishing the project website

The `Publish verified project website` workflow publishes the `main` branch to the
existing GitHub Pages address. Pushes to `main` and manual workflow dispatches use
the same process. The old `gh-pages` branch is retained as a historical snapshot;
it is no longer the publishing source once the repository uses GitHub Actions.

`scripts/prepare_pages.py` copies only tracked, non-hidden public files, plus
`.nojekyll`, into `_site`. It preserves the existing public document/result paths,
checks all four targets and 48 conditions, verifies the SHA-256 and WAV format of
all 96 playback files, and checks the source hashes of the explanatory context.
Generated `deployment.json` identifies the exact deployed commit and audio hashes.
Models, local datasets and environments are untracked and are not packaged.

The earlier branch-based publication reached `syncing_files` but was cancelled by
the deploy action's default ten-minute timeout. The custom workflow gives the
deployment thirty minutes, serialises releases and validates the artifact before
uploading. A completed build alone does not prove publication: check the public
`deployment.json`, page contents and actual audio playback after deployment.

Local packaging check, from the repository root:

```sh
python3 scripts/prepare_pages.py --output /path/to/an/empty/directory
```

Workflow dependencies are pinned to verified official action commits. See the
[custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
and [deploy-pages inputs](https://github.com/actions/deploy-pages#inputs-).
