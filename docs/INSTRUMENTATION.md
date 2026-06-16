# Instrumentation

How we measure floe-guard's adoption — and the one thing we deliberately *don't* do.

## No runtime telemetry (deliberate)

floe-guard does **not** phone home. It sends no usage events, no install pings, no
identifiers — nothing leaves your process at runtime except hosted-budget reads
you explicitly opt into by setting `FLOE_API_KEY` (the *Upgrade to hosted Floe*
path) — never otherwise.

This is a choice, not an oversight. A guardrail's whole value is trust: a library
that silently exfiltrates usage from people's agents is the opposite of a tool you
hand a budget to. Runtime telemetry in OSS dev tools reliably triggers backlash
and forks. So we measure adoption only from **public, outside-in signals** — the
ones below — and accept that they're coarser than telemetry. That trade is correct
for this project.

## The signals we track

| Runbook KPI | How it's measured | Source |
|---|---|---|
| Star velocity | GitHub stargazers over time | `scripts/metrics.py` → GitHub API |
| Fork count | GitHub forks | `scripts/metrics.py` |
| Install velocity (`pip install`) | PyPI download counts (day/week/month) | `scripts/metrics.py` → pypistats.org |
| README reach | repo views & unique clones | `scripts/metrics.py` (needs `GITHUB_TOKEN`) |
| README → site click-through | UTM params on outbound links | Floe web analytics |
| Install → hosted signup | UTM `utm_source=floe-guard` lands in the signup funnel | Floe web analytics |
| "Built with floe-guard" usage | badge backlinks | GitHub code/badge search |

### Snapshot script

```bash
python scripts/metrics.py                    # stars, forks, PyPI downloads
GITHUB_TOKEN=ghp_xxx python scripts/metrics.py   # + repo views/clones
```

It prints real numbers only — if a source is unreachable it says so rather than
guessing. Run it on a schedule and diff snapshots to get *velocity* (the runbook's
actual target), not just totals.

### UTM scheme

Outbound links in the README carry:

```
?utm_source=floe-guard&utm_medium=readme&utm_campaign=oss
```

So clicks from this repo into `dev-dashboard.floelabs.xyz` / `floelabs.xyz` are
attributable in Floe's web analytics, and any session that converts to a hosted
signup keeps `utm_source=floe-guard` — that's the **install → signup** handoff the
runbook calls the critical metric. The conversion side lives in the Floe web app's
funnel, not in this repo.

### "Built with floe-guard" badge

Downstream projects can advertise usage with the badge in the README's *Built with
floe-guard* section. Each embed is a backlink — searchable on GitHub — and is the
honest version of a "powered by" usage signal (opt-in, public, no tracking).

## What's still manual

- Posting the launch content and seeding (the runbook's Week-1/2 GTM motion).
- Wiring the UTM-tagged sessions into a dashboard on the Floe web side.
