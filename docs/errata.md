# Errata

This page is the **single place** where corrected errors in NumSDE are
documented: wrong numerical results, wrong convergence orders or formulas on
theory pages, and flawed tests that let an error pass.

The changelog, release notes, issues and theory pages never describe a
correction themselves. They link to the entry here by its ID, for example
`Fixed: wrong Lévy-area variance for m ≥ 3, see E-0004.` This keeps every
correction in one place and avoids contradicting descriptions.

## How to add an entry

1. Take the next free ID. IDs are never reused, even if an entry turns out to
   be invalid; such an entry is marked as withdrawn instead.
2. Copy the template below to the top of the list of entries (newest first).
3. Fill in every field. If a field does not apply, write "none" and say why.
4. Reference the ID in the changelog entry and in the pull request that fixes
   the error.

### Template

```markdown
### E-XXXX: <short title>

| Field | Content |
| --- | --- |
| Found | YYYY-MM-DD, by <name or issue link> |
| Fixed in | <version>, <pull request link> |
| Affected versions | <first affected> to <last affected> |
| Component | <scheme, module or documentation page> |
| Description | <what was wrong and why> |
| Impact | <which results change; whether published numbers or figures need re-running> |
| Guarding test | <test that now fails on the old behavior> |
```

## Entries

No errata have been recorded yet.
