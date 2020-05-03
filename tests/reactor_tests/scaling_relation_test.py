"""
Tests for the scaling relationships.
"""

import unittest

from rt_eqcorrscan.reactor.scaling_relations import (
    set_scaling_relation, SCALING_RELATIONS, SCALING_RELATIONS_ORIGINAL,
    get_scaling_relation, register_scaling_relation)


class ScalingRelationTests(unittest.TestCase):
    def test_default_registered(self):
        self.assertIn("default", SCALING_RELATIONS.keys())

    def test_get_scaling_relation(self):
        scaling_relation = get_scaling_relation("wells_coppersmith_surface")
        self.assertTrue(callable(scaling_relation))

    def test_register_relation(self):
        def scalar(x):
            return x * 10
        register_scaling_relation(scalar)
        self.assertIn("scalar", SCALING_RELATIONS)
        self.assertNotIn("scalar", SCALING_RELATIONS_ORIGINAL)

    def test_set_default(self):
        def scalar(x):
            return x * 10
        old_default = get_scaling_relation()
        register_scaling_relation(scalar)
        set_scaling_relation("scalar")
        self.assertTrue(get_scaling_relation() == scalar)
        set_scaling_relation(old_default)

    def test_set_default_context_manager(self):
        def scalar(x):
            return x * 10
        old_default = get_scaling_relation()
        register_scaling_relation(scalar)
        with set_scaling_relation("scalar"):
            self.assertTrue(get_scaling_relation() == scalar)
        self.assertTrue(get_scaling_relation() == old_default)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level="DEBUG")

    unittest.main()
