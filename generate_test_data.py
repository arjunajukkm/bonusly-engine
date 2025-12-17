import random
import string
import pandas as pd

ROWS = 50_000
OUTPUT_FILE = "test_input_50k.xlsx"

VALUES = ["collaboration", "execution", "disruption", "integrity", "inclusion"]
REGIONS = {
    "India": "JAPAC",
    "Singapore": "JAPAC",
    "Japan": "JAPAC",
    "Germany": "Europe, Middle East & Africa",
    "France": "Europe, Middle East & Africa",
    "United Kingdom": "Europe, Middle East & Africa",
    "United States of America": "Americas",
    "Brazil": "LATAM",
}
COUNTRIES = list(REGIONS.keys())
GENDERS = ["Male", "Female"]
TEAMS = ["Engineering", "Product", "HR", "Finance", "Sales", "Marketing"]

BAD_RATINGS = [
    "APPROACHING EXPECTATIONS",
    "UNSATISFACTORY",
    "LOA",
    "TOO NEW TO RATE",
]
GOOD_RATINGS = ["MEETS EXPECTATIONS", "EXCEEDS EXPECTATIONS"]

def random_email(name):
    return f"{name.lower()}@company.com"

def random_name():
    return "".join(random.choices(string.ascii_lowercase, k=7)).capitalize()

def build_message():
    tags = random.sample(VALUES, random.randint(1, 3))
    mentions = random.randint(0, 5)
    msg = "Great work "
    msg += " ".join(f"#{t}" for t in tags)
    msg += " "
    msg += " ".join(f"@user{random.randint(1,999)}" for _ in range(mentions))
    return msg.strip()

# ------------------------
# Base Data
# ------------------------
base_rows = []
emails = []

for i in range(ROWS):
    first = random_name()
    last = random_name()
    email = random_email(first + last)
    emails.append(email)

    base_rows.append({
        "FromEmail": f"sender{i%200}@company.com",
        "toFirstName": first,
        "toLastName": last,
        "toEmail": email,
        "toLocation": random.choice(COUNTRIES),
        "Message": build_message(),
        "Recognition": "Peer to Peer" if random.random() > 0.15 else "Manager Award",
        "Gender": random.choice(GENDERS),
        "Team": random.choice(TEAMS),
    })

base_df = pd.DataFrame(base_rows)

# ------------------------
# Previous Winners
# ------------------------
prev_rows = []
for email in random.sample(emails, int(ROWS * 0.1)):
    prev_rows.append({
        "toEmail": email,
        "FY": random.choice(["FY23", "FY24", "FY25"]),
        "Quarter": random.randint(1, 4),
    })

prev_df = pd.DataFrame(prev_rows)

# ------------------------
# Serving Notice
# ------------------------
sn_rows = [{"toEmail": email} for email in random.sample(emails, int(ROWS * 0.05))]
sn_df = pd.DataFrame(sn_rows)

# ------------------------
# Rating
# ------------------------
rating_rows = []
for email in emails:
    rating_rows.append({
        "toEmail": email,
        "Year End Rating": random.choice(GOOD_RATINGS + BAD_RATINGS),
        "Mid Year Rating": random.choice(GOOD_RATINGS + BAD_RATINGS),
    })

rating_df = pd.DataFrame(rating_rows)

# ------------------------
# Write Excel
# ------------------------
with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
    base_df.to_excel(writer, index=False, sheet_name="Base Data")
    prev_df.to_excel(writer, index=False, sheet_name="Previous Winners")
    sn_df.to_excel(writer, index=False, sheet_name="Serving Notice")
    rating_df.to_excel(writer, index=False, sheet_name="Rating")

print(f"✅ Generated {OUTPUT_FILE} with {ROWS} Base Data rows")

