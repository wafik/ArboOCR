// tests/test_main.cpp — provides doctest's main()/runtime implementation.
// Kept separate from the actual test files so none of them need to know
// about this; every *_test.cpp just #include <doctest/doctest.h> and adds
// TEST_CASE blocks.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
