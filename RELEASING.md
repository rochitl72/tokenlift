# Release checklist

1. `python -m pip install build twine && python -m build && twine check dist/*`
2. Tag and push: `git tag -a v0.1.0 -m "TokenLift v0.1.0" && git push origin v0.1.0`
3. On GitHub: Releases, Draft a new release, choose `v0.1.0`, paste the matching CHANGELOG section.
4. Zenodo DOI: sign in at zenodo.org with GitHub, switch this repository on under GitHub settings,
   then publish the GitHub release. Zenodo reads `.zenodo.json` and issues the DOI.
5. Add the DOI badge to the README and the DOI to `CITATION.cff`.
6. PyPI (optional): create a PyPI account with 2FA, make an API token, then `twine upload dist/*`.
