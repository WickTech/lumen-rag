# Deployments

We deploy to production every weekday at 4pm using a blue-green strategy.
Rollbacks are automatic if error rates exceed 2% within the first five minutes.
Hotfixes may be deployed outside the window with approval from an on-call lead.
