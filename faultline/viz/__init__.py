"""Figure generation for FaultLine runs, drawn from persisted report data.

Everything here reads plain records (a room layout, a graph snapshot, a refinement
trace), never a live environment, so any saved run can be redrawn without retraining.
Matplotlib is imported lazily, so importing the package never needs the reports extra.
"""
