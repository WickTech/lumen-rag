# Engineering handbook — support & postmortems

## Authentication
All internal services authenticate via short-lived OAuth2 tokens issued by
the identity provider; tokens expire after 60 minutes and must be refreshed
by the client. Service-to-service calls use mutual TLS in addition to
tokens. Personal access tokens for CLI tools expire after 90 days and are
scoped to a single project.

## Customer support escalation
Support tickets tagged "urgent" must be triaged within 1 hour during
business hours. Enterprise customers on the premium support tier get a
15-minute first-response SLA, the same acknowledgment window as internal
on-call pages. Escalations to engineering go through the on-call engineer,
not directly to individual contributors.

## Postmortem template
Every postmortem must include a timeline, root cause, blast radius, and at
least three concrete follow-up action items with owners and due dates.
Postmortems for severity-1 incidents are reviewed in the weekly engineering
sync; severity-2 postmortems are reviewed asynchronously. Templates live in
the incident-response wiki space.

## Documentation standards
Public-facing API docs are generated from OpenAPI specs and rebuilt on
every merge to main; internal runbooks are written in markdown and stored
alongside the service they document, not in a separate wiki.

## Support tooling
Support agents use a shared ticketing queue with automatic tagging based on
keyword rules; tickets that go untagged for more than 10 minutes are
flagged for manual triage by a team lead.

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
