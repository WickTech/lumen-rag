# Engineering handbook — code & security

## Code review
Every pull request requires at least one approval. Changes to billing or
auth code require two approvals, including one from a senior engineer. CI
must be green before merge.

## Billing system changes
Any change touching the billing pipeline, invoicing, or payment provider
integration requires sign-off from the billing tech lead in addition to the
standard two code-review approvals. Billing migrations must be dry-run
against a staging replica before running in production. Refund logic
changes require a finance stakeholder review.

## Security incident response
Suspected security incidents (credential leaks, unauthorized access, data
exposure) must be reported to the security channel within 15 minutes of
discovery, faster than the standard on-call acknowledgment window. The
security lead triages severity and decides whether to invoke the incident
commander process. All security incidents get a postmortem regardless of
severity, unlike ordinary incidents which only require one at severity-1.

## Secrets management
Secrets (API keys, database credentials, signing keys) are stored in the
central secrets manager, never in source control or environment files
committed to git. Secrets are rotated automatically every 90 days;
production database credentials rotate every 30 days. Access to production
secrets requires a just-in-time approval, logged and reviewed weekly.

## Linter and formatting rules
All Python code is formatted with ruff and must pass linting in CI before
merge; JavaScript code uses prettier with the shared team config. Style
disagreements that aren't caught by the linter are left to reviewer
discretion rather than escalated.

## Dependency updates
Automated dependency update bots open pull requests weekly for minor and
patch version bumps; major version bumps require a manual review from the
package owner and a changelog read-through before merge.

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
