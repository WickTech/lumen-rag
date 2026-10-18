# Backup policy
Production databases are backed up continuously with point-in-time recovery for 35 days, and a full snapshot is copied weekly to a second region and kept for 12 weeks. Restore tests run every quarter and the results are reported to the head of infrastructure. The recovery point objective is 5 minutes and the recovery time objective is 4 hours for the core platform.
