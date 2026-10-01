def financial_profile_down(current, baseline):
    return current < baseline

def test_financial_profile_down():
    assert financial_profile_down(80, 100)

def test_financial_profile_ok():
    assert not financial_profile_down(120, 100)
