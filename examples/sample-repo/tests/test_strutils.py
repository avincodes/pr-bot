from strutils import reverse_string, shout


def test_shout():
    assert shout("hi") == "HI!"


def test_reverse_string_basic():
    assert reverse_string("abc") == "cba"


def test_reverse_string_empty():
    assert reverse_string("") == ""


def test_reverse_string_unicode():
    assert reverse_string("héllo") == "olléh"
