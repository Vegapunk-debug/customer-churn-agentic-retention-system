import unittest
import sys
import os

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.retention_automation import RetentionAgent

class TestRetentionAutomation(unittest.TestCase):
    def setUp(self):
        self.agent = RetentionAgent()
        self.sample_profile = {
            "customerID": "TST-001",
            "churn_probability": 0.85,
            "tenure": 2
        }

    def test_generate_strategy_returns_dict(self):
        strategy = self.agent.generate_strategy(self.sample_profile)
        self.assertIsInstance(strategy, dict)
        self.assertEqual(strategy["customer_id"], "TST-001")
        self.assertIn("recommended_action", strategy)

    def test_trigger_outreach(self):
        strategy = {"customer_id": "TST-001"}
        result = self.agent.trigger_automated_outreach(strategy)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
