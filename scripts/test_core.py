"""Dependency-light regression suite; never requires private motion data or weights."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
PATTERNS = ('test_overlap.py', 'test_command_reports.py', 'test_world.py',
            'test_tracking.py', 'test_intent_route.py', 'test_social_engine.py',
            'test_life_policy.py', 'test_navigation_memory.py',
            'test_mcp_contract.py')
suite = unittest.TestSuite()
for pattern in PATTERNS:
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern=pattern))
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
