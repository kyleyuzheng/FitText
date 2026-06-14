#!/usr/bin/env python3
import json, os

for fname in os.listdir("."):
    if fname.endswith(".json"):
        with open(fname) as f:
            data = json.load(f)
        print(f"{fname}: {len(data)}")

"""
G1_category.json: 153
G1_instruction.json: 163
G1_tool.json: 158

G2_category.json: 124
G2_instruction.json: 106

G3_instruction.json: 61
"""