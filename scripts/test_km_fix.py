"""Verify the Kaplan-Meier censoring fix."""
from ttr_tracker.survival import kaplan_meier

# Test 1: tied events and censored at the same time
times = [5, 5, 5, 5, 5]
events = [True, True, True, False, False]
curve = kaplan_meier(times, events)
print("Test 1: tied events+censored at t=5")
for pt in curve:
    print(f"  t={pt['time']} S={pt['survival']} at_risk={pt['at_risk']} events={pt['events']}")
assert len(curve) == 1, f"Expected 1 row, got {len(curve)}"
assert curve[0]["survival"] == 0.4, f"S=0.4, got {curve[0]['survival']}"
print("  PASS\n")

# Test 2: only events
times2 = [1, 2, 3]
events2 = [True, True, True]
curve2 = kaplan_meier(times2, events2)
print("Test 2: all events")
for pt in curve2:
    print(f"  t={pt['time']} S={pt['survival']} at_risk={pt['at_risk']}")
assert len(curve2) == 3
assert abs(curve2[0]["survival"] - 0.6667) < 0.001, f"Expected 0.6667, got {curve2[0]['survival']}"
assert abs(curve2[1]["survival"] - 0.3333) < 0.001
assert abs(curve2[2]["survival"] - 0.0) < 0.001
print("  PASS\n")

# Test 3: only censored
times3 = [1, 2, 3]
events3 = [False, False, False]
curve3 = kaplan_meier(times3, events3)
print("Test 3: all censored")
for pt in curve3:
    print(f"  t={pt['time']} S={pt['survival']} at_risk={pt['at_risk']}")
assert all(pt["survival"] == 1.0 for pt in curve3)
print("  PASS\n")

# Test 4: events and censored at different times
times4 = [1, 1, 2, 3, 3]
events4 = [True, False, True, True, False]
curve4 = kaplan_meier(times4, events4)
print("Test 4: mix at different times")
for pt in curve4:
    print(f"  t={pt['time']} S={pt['survival']} at_risk={pt['at_risk']} events={pt['events']}")
assert len(curve4) == 3
# t=1: at_risk=5, events=1, c=1, S=(5-1)/5=0.8, next risk=5-1-1=3
# t=2: at_risk=3, events=1, S=0.8*(3-1)/3=0.5333, next risk=3-1=2
# t=3: at_risk=2, events=1, c=1, S=0.5333*(2-1)/2=0.2667, next risk=2-1-1=0
assert abs(curve4[0]["survival"] - 0.8) < 0.001
assert abs(curve4[1]["survival"] - 0.5333) < 0.001
assert abs(curve4[2]["survival"] - 0.2667) < 0.001
assert curve4[0]["at_risk"] == 5
assert curve4[1]["at_risk"] == 3
assert curve4[2]["at_risk"] == 2
assert curve4[0]["events"] == 1
assert curve4[1]["events"] == 1
assert curve4[2]["events"] == 1
print("  PASS\n")

print("All Kaplan-Meier tests passed!")
