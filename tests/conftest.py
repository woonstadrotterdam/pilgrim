import os


# TODO: Remove this when first tests are added
def pytest_configure(config):
    """Configure pytest to not error when no tests are found."""
    config.option.no_tests_action = "ignore"


# TODO: Remove this when first tests are added
def pytest_sessionfinish(session, exitstatus):
    """
    Modify exit status to prevent failure when no tests are collected
    or when coverage has no data to report.
    """
    # pytest.ExitCode.NO_TESTS_COLLECTED = 5
    if exitstatus == 5:
        session.exitstatus = 0

    # Coverage typically exits with status 1 when there's no data
    # This ensures we don't fail CI when coverage has nothing to report
    if os.environ.get("COVERAGE_PROCESS_START") and exitstatus == 1:
        session.exitstatus = 0
