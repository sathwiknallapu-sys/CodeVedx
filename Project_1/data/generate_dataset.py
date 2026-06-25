"""
generate_dataset.py
--------------------
One-off helper script used during development to create the synthetic
household utility usage dataset (utility_usage_data.csv).

This is NOT part of the runtime application — it was used once to produce
the CSV that ships with the project. Kept here for transparency / so the
dataset generation process can be reviewed or reproduced.

Logic: electricity units consumed is modeled as a function of household
size, home area, AC usage hours, number of appliances, and season, with
added random noise to keep it realistic (not a perfectly clean formula).
"""

import csv
import random

random.seed(42)

SEASONS = ["Summer", "Winter", "Monsoon", "Spring"]
HOME_TYPES = ["Apartment", "Independent House", "Villa"]

def generate_row(record_id):
    household_size = random.randint(1, 7)
    home_area_sqft = random.randint(400, 3200)
    home_type = random.choice(HOME_TYPES)
    season = random.choice(SEASONS)
    ac_usage_hours = round(random.uniform(0, 10), 1)
    num_appliances = random.randint(3, 18)
    avg_temp_c = {
        "Summer": random.uniform(32, 45),
        "Winter": random.uniform(5, 18),
        "Monsoon": random.uniform(24, 30),
        "Spring": random.uniform(20, 28),
    }[season]
    prev_month_units = round(random.uniform(80, 650), 1)

    # base load from household size and appliances
    base = household_size * 18 + num_appliances * 6.5
    # area contributes to lighting/fans load
    area_factor = home_area_sqft * 0.018
    # AC is the biggest swing factor, amplified by heat
    ac_factor = ac_usage_hours * (9 + max(0, avg_temp_c - 25) * 0.6)
    # winter heating bump (independent houses/villas lose more heat)
    heating_bump = 0
    if season == "Winter" and home_type in ("Independent House", "Villa"):
        heating_bump = random.uniform(15, 40)

    noise = random.uniform(-25, 25)

    units_consumed = max(20, base + area_factor + ac_factor + heating_bump + noise)
    units_consumed = round(units_consumed, 1)

    # rough INR slab-based billing approximation for realism
    def bill_from_units(u):
        if u <= 100:
            return u * 4.0
        elif u <= 300:
            return 100 * 4.0 + (u - 100) * 6.5
        else:
            return 100 * 4.0 + 200 * 6.5 + (u - 300) * 8.2

    bill_amount = round(bill_from_units(units_consumed), 2)

    return {
        "record_id": record_id,
        "household_size": household_size,
        "home_area_sqft": home_area_sqft,
        "home_type": home_type,
        "season": season,
        "avg_temp_c": round(avg_temp_c, 1),
        "ac_usage_hours": ac_usage_hours,
        "num_appliances": num_appliances,
        "prev_month_units": prev_month_units,
        "units_consumed": units_consumed,
        "bill_amount": bill_amount,
    }


def main():
    rows = [generate_row(i) for i in range(1, 601)]
    fieldnames = list(rows[0].keys())

    with open("utility_usage_data.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows -> utility_usage_data.csv")


if __name__ == "__main__":
    main()
