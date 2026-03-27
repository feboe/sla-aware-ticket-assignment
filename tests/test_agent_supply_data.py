from __future__ import annotations

import csv
import unittest
from collections import Counter
from pathlib import Path

from scripts.generate_ticket_assignment_data import SEED, generate_dataset


AGENTS_PATH = Path("data/agents.csv")
EXPECTED_HEADER = [
    "agent_id",
    "agent_role",
    "languages",
    "queue_permissions",
    "priority_scope",
    "skill_tags",
    "shift_start",
    "shift_end",
    "capacity_min_per_day",
    "scarce_resource",
]
EXPECTED_SKILL_TAGS = {
    "GEN-01": "account_access|billing|product_support|de_language|enterprise_handling",
    "GEN-02": "account_access|billing|product_support|de_language|enterprise_handling",
    "GEN-03": "account_access|billing|product_support|de_language|enterprise_handling",
    "PS-01": "product_support|product_specialist|enterprise_handling|de_language",
    "PS-02": "product_support|product_specialist|enterprise_handling",
    "INT-01": "integrations_api|api_specialist|integration_specialist|enterprise_handling|de_language",
    "INT-02": "integrations_api|api_specialist|integration_specialist|enterprise_handling",
    "SIC-01": "account_access|billing|product_support|integrations_api|api_specialist|incident_escalation|enterprise_handling|de_language",
}


def parse_set(value: str) -> set[str]:
    return set(value.split("|")) if value else set()


class TestAgentSupplyData(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with AGENTS_PATH.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            cls.header = reader.fieldnames
            cls.rows = list(reader)
        cls.tickets = generate_dataset(SEED)

    def test_has_exact_header(self) -> None:
        self.assertEqual(self.header, EXPECTED_HEADER)

    def test_has_exactly_eight_agents(self) -> None:
        self.assertEqual(len(self.rows), 8)

    def test_role_counts_match_roster_design(self) -> None:
        role_counts = Counter(row["agent_role"] for row in self.rows)
        self.assertEqual(
            role_counts,
            Counter(
                {
                    "generalist": 3,
                    "product_specialist": 2,
                    "integration_specialist": 2,
                    "senior_incident_coordinator": 1,
                }
            ),
        )

    def test_language_coverage_matches_plan(self) -> None:
        generalists = [row for row in self.rows if row["agent_role"] == "generalist"]
        self.assertTrue(all(row["languages"] == "EN|DE" for row in generalists))

        product_specialists = [
            row for row in self.rows if row["agent_role"] == "product_specialist"
        ]
        self.assertEqual(
            sum(row["languages"] == "EN|DE" for row in product_specialists),
            1,
        )

        integration_specialists = [
            row for row in self.rows if row["agent_role"] == "integration_specialist"
        ]
        self.assertEqual(
            sum(row["languages"] == "EN|DE" for row in integration_specialists),
            1,
        )

        senior = next(
            row for row in self.rows if row["agent_role"] == "senior_incident_coordinator"
        )
        self.assertEqual(senior["languages"], "EN|DE")

    def test_queue_coverage_has_de_support_everywhere(self) -> None:
        expected_queues = {
            "Account Access",
            "Billing",
            "Product Support",
            "Integrations/API",
        }
        de_covered_queues = set()

        for row in self.rows:
            languages = set(row["languages"].split("|"))
            if "DE" not in languages:
                continue
            de_covered_queues.update(row["queue_permissions"].split("|"))

        self.assertTrue(expected_queues.issubset(de_covered_queues))

    def test_skill_tags_match_refined_role_design(self) -> None:
        observed = {row["agent_id"]: row["skill_tags"] for row in self.rows}
        self.assertEqual(observed, EXPECTED_SKILL_TAGS)
        self.assertTrue(all("senior_review" not in tags for tags in observed.values()))

    def test_shifts_and_capacity_are_uniform(self) -> None:
        for row in self.rows:
            self.assertEqual(row["shift_start"], "08:00")
            self.assertEqual(row["shift_end"], "16:00")
            self.assertEqual(row["capacity_min_per_day"], "480")

    def test_only_senior_is_marked_scarce(self) -> None:
        scarce_agents = [
            row["agent_id"] for row in self.rows if row["scarce_resource"] == "1"
        ]
        self.assertEqual(scarce_agents, ["SIC-01"])

    def test_every_ticket_has_v1_feasible_agents(self) -> None:
        for ticket in self.tickets:
            eligible_agents = []
            for agent in self.rows:
                if ticket.queue not in parse_set(agent["queue_permissions"]):
                    continue
                if ticket.language not in parse_set(agent["languages"]):
                    continue
                if ticket.priority not in parse_set(agent["priority_scope"]):
                    continue
                eligible_agents.append(agent["agent_id"])

            self.assertTrue(
                eligible_agents,
                msg=f"No feasible agents for {ticket.ticket_id}",
            )

    def test_de_integration_tickets_are_covered_by_bilingual_specialist(self) -> None:
        bilingual_integration_specialist = next(
            row for row in self.rows if row["agent_id"] == "INT-01"
        )
        for ticket in self.tickets:
            if ticket.queue != "Integrations/API" or ticket.language != "DE":
                continue

            self.assertIn(
                "Integrations/API",
                parse_set(bilingual_integration_specialist["queue_permissions"]),
            )
            self.assertIn("DE", parse_set(bilingual_integration_specialist["languages"]))
            self.assertIn(
                ticket.priority,
                parse_set(bilingual_integration_specialist["priority_scope"]),
            )


if __name__ == "__main__":
    unittest.main()
