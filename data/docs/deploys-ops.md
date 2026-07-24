# Engineering handbook — deploys & operations

## Deployments
We deploy to production every weekday at 4pm using a blue-green strategy.
Rollbacks are automatic if error rates exceed 2% within the first five
minutes. Hotfixes may be deployed outside the window with approval from an
on-call lead.

## On-call
On-call rotations last one week and run Monday to Monday. The primary
on-call engineer must acknowledge pages within 15 minutes. Secondary on-call
is the fallback after 30 minutes of no response.

## Incident response
Severity-1 incidents require a written postmortem within 48 hours.
Postmortems are blameless and focus on systemic fixes, not individual fault.

## Disaster recovery
The disaster recovery plan targets a recovery time objective (RTO) of 4
hours and a recovery point objective (RPO) of 15 minutes for the primary
database. Full DR drills, including a simulated region failover, run twice a
year. Runbooks are stored outside the primary cloud region so they remain
accessible during a regional outage.

## Internal tooling access
Engineers request access to internal admin dashboards through the access
portal; approval routes to the resource owner and typically completes
within one business day. Read-only access to production dashboards is
granted by default to all engineers on day one.

## Office network maintenance
Scheduled network maintenance windows run the first Sunday of each month
from 2am to 4am local time. Engineers relying on VPN for weekend work
should check the maintenance calendar before starting a task.

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
