# Testing standards
New code needs automated tests, and the pipeline blocks a merge if coverage of changed lines falls below 80 percent. Services with robot control logic must also run a simulated warehouse regression suite that takes about 25 minutes. Flaky tests are quarantined within 2 days and fixed within 2 weeks. Manual testing is documented in the pull request for changes to the dashboard.
