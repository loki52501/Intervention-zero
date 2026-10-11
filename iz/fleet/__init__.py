"""Fleet layer: several humanoid robots sharing one world (D10).

Honest claim: base motion / locomotion is a scripted parametric engine, not a learned walking
policy.  Learning happens at behaviour, environment and fleet level (shared hazards, task
claiming, right-of-way).  The simulator is authored by us.
"""
from iz.fleet.coordinator import Coordinator
from iz.fleet.scene import add_fleet_robots

__all__ = ["Coordinator", "add_fleet_robots"]
