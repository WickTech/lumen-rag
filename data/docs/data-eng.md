# Engineering handbook — data & platform

## Data retention policy
Customer data is retained for the duration of the account plus 90 days
after deletion, to allow for accidental-deletion recovery. Application logs
are retained for 30 days; security audit logs are retained for 1 year to
satisfy compliance requirements. Backups are encrypted at rest and rotated
every 24 hours with a 14-day retention window.

## Database migrations
Schema migrations must be backward-compatible with the previous application
version to support zero-downtime deploys: add columns as nullable first,
backfill, then enforce constraints in a follow-up migration. Migrations
affecting tables over 10 million rows require a review from the database
team and must run online without locking writes.

## API versioning
Public APIs are versioned in the URL path (/v1/, /v2/) and each version is
supported for at least 12 months after the next version ships. Breaking
changes require a new major version; additive changes (new optional fields)
can ship within the current version. Deprecation notices go out at least 90
days before a version is sunset.

## Feature flags
New user-facing features must ship behind a feature flag unless the change
is a pure bugfix. Flags default to off in production and are rolled out
gradually: 1% -> 10% -> 50% -> 100%, with at least a 24-hour soak at each
stage for risky changes. Stale flags older than 90 days are flagged for
cleanup in the quarterly flag audit.

## Internal dashboards
Engineering metrics dashboards refresh every 15 minutes and pull from the
same warehouse tables used for the quarterly business review. Dashboard
access is open to all engineers; editing dashboard definitions requires the
data-platform team's review.

## Data warehouse costs
Ad-hoc warehouse queries over 1TB scanned trigger an automatic Slack alert
to the requester and the data-platform on-call, since large ad-hoc queries
are the leading cause of unexpected warehouse cost spikes.

## Team communication norms
Teams default to public channels over DMs for anything work-related, so
context stays searchable. Cross-team requests go through a dedicated
request channel rather than pinging individuals directly, and response-time
expectations there are best-effort, not an SLA.

## Tooling procurement
New SaaS tool requests over $500/year go through a lightweight procurement
review covering security and data-handling questions before purchase.
Renewals under the same terms skip the review and are approved by finance
automatically.

## Internal wiki hygiene
Wiki pages without an update in 12 months are flagged stale and surfaced in
a quarterly cleanup pass; owners either refresh or archive them. Search
ranking on the wiki favors recently-edited pages over older ones.
