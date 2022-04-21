# This file is part production_cost_analysis module for Tryton.
# The COPYRIGHT file at the top level of this repository contains
# the full copyright notices and license terms.
import unittest
import doctest

from trytond.tests.test_tryton import ModuleTestCase
from trytond.tests.test_tryton import suite as test_suite
from trytond.tests.test_tryton import doctest_teardown
from trytond.tests.test_tryton import doctest_checker

class ProductionCostAnalysisTestCase(ModuleTestCase):
    'Test Production Cost Analysis module'
    module = 'production_cost_analysis'


def suite():
    suite = test_suite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(
            ProductionCostAnalysisTestCase))
    suite.addTests(doctest.DocFileSuite('scenario_production.rst',
            tearDown=doctest_teardown,
            encoding='utf-8',
            checker=doctest_checker,
            optionflags=doctest.REPORT_ONLY_FIRST_FAILURE))
    return suite
