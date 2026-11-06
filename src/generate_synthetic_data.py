"""
TeqoAI — Synthetic Data Generator

Generates auth.txt.gz and redteam.txt.gz in the EXACT LANL format, so they
drop straight into data/raw/ and work with src/data_prep.py unchanged.

Why this exists: the real LANL dataset is gated behind a usage-description
form and is very large. This generator produces realistic normal login
behavior, realistic-but-benign "weird" activity (so your model has to
actually learn what's normal, not just spot rare events), and a set of
red-team-style compromise chains (lateral movement through a small org)
with matching ground-truth labels.

Swap this out for the real LANL files later — data_prep.py doesn't change.

Usage:
    python src/generate_synthetic_data.py

Output:
    data/raw/auth.txt.gz
    data/raw/redteam.txt.gz
"""

import gzip
import csv
import os
import random

random.seed(42)

RAW_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "raw")
)
AUTH_OUT = os.path.join(RAW_DIR, "auth.txt.gz")
REDTEAM_OUT = os.path.join(RAW_DIR, "redteam.txt.gz")

# ---- Config ------------------------------------------------------------
NUM_USERS = 60
NUM_DAYS = 10
SECONDS_PER_DAY = 86_400
BUSINESS_START = 8 * 3600     # 08:00
BUSINESS_END = 18 * 3600      # 18:00
DOMAIN = "DOM1"

# A small org topology, matching the User -> Laptop -> Server -> Application
# -> Database story from the proposal.
WORKSTATIONS = [f"C{i}" for i in range(1, NUM_USERS + 1)]   # one laptop per user
SERVERS = ["SRV1", "SRV2", "SRV3"]
APPLICATIONS = ["APP-HR", "APP-FIN"]
DATABASE = "DB-CORE"

USERS = [f"U{i}@{DOMAIN}" for i in range(1, NUM_USERS + 1)]
# each user's home workstation
USER_HOME_PC = {USERS[i]: WORKSTATIONS[i] for i in range(NUM_USERS)}
# each user's habitual servers (1-2 of the 3, consistently, so "new server" is meaningful)
USER_SERVERS = {u: random.sample(SERVERS, k=random.choice([1, 2])) for u in USERS}

AUTH_TYPES = ["Kerberos", "NTLM", "Negotiate"]
LOGON_TYPES_NORMAL = ["Network", "Interactive"]

NUM_COMPROMISED_USERS = 4       # how many users get a red-team attack chain
NUM_BENIGN_ANOMALIES = 300      # realistic "weird but not malicious" events
TEST_DAY_START = max(1, int(NUM_DAYS * 0.8))

# ---- Helpers -------------------------------------------------------------

def business_hour_time(day):
    """A random timestamp within business hours on the given day (0-indexed)."""
    offset = random.randint(BUSINESS_START, BUSINESS_END)
    return day * SECONDS_PER_DAY + offset


def off_hour_time(day):
    """A random timestamp outside business hours (nights/very early)."""
    offset = random.choice(
        list(range(0, BUSINESS_START)) + list(range(BUSINESS_END, SECONDS_PER_DAY))
    )
    return day * SECONDS_PER_DAY + offset


def auth_row(time_, user, dest_user, src_pc, dst_pc, auth_type, logon_type, result):
    return [time_, user, dest_user, src_pc, dst_pc, auth_type, logon_type, "LogOn", result]


# ---- Generate normal behavior ---------------------------------------------

def generate_normal_events():
    events = []
    for day in range(NUM_DAYS):
        for user in USERS:
            home_pc = USER_HOME_PC[user]

            # 1) User logs into their own laptop each business day (skip some days randomly)
            if random.random() < 0.9:
                t = business_hour_time(day)
                events.append(auth_row(
                    t, user, user, home_pc, home_pc,
                    random.choice(AUTH_TYPES), random.choice(LOGON_TYPES_NORMAL), "Success"
                ))

                # 2) Sometimes authenticates onward to one of their habitual servers
                if random.random() < 0.6:
                    server = random.choice(USER_SERVERS[user])
                    t2 = t + random.randint(60, 1800)
                    events.append(auth_row(
                        t2, user, user, home_pc, server,
                        random.choice(AUTH_TYPES), "Network", "Success"
                    ))

            # 3) Occasional normal failed login (mistyped password, etc.) — benign
            if random.random() < 0.05:
                t3 = business_hour_time(day)
                events.append(auth_row(
                    t3, user, user, home_pc, home_pc,
                    random.choice(AUTH_TYPES), "Interactive", "Fail"
                ))

    return events


def generate_benign_anomalies():
    """Unusual but NOT malicious events — off-hours work, a new device now and
    then, a one-off server access. These are NOT in redteam.txt.gz. Their job
    is to make sure your model has to learn real patterns, not just flag
    every rare event as an attack."""
    events = []
    for _ in range(NUM_BENIGN_ANOMALIES):
        user = random.choice(USERS)
        home_pc = USER_HOME_PC[user]
        day = random.randint(0, NUM_DAYS - 1)
        t = off_hour_time(day)
        kind = random.choice(["off_hours_own_pc", "unfamiliar_server"])
        if kind == "off_hours_own_pc":
            events.append(auth_row(
                t, user, user, home_pc, home_pc,
                random.choice(AUTH_TYPES), "Interactive", "Success"
            ))
        else:
            server = random.choice(SERVERS)
            events.append(auth_row(
                t, user, user, home_pc, server,
                random.choice(AUTH_TYPES), "Network", "Success"
            ))
    return events


# ---- Generate red-team compromise chains -----------------------------------

def generate_redteam_events():
    """Each compromised user gets a short burst of lateral-movement events:
    laptop -> server -> application -> database, at an off-hour time, using
    an unfamiliar path. Every hop is written to BOTH auth events and
    redteam labels, using matching (time, user, src, dst) keys so
    data_prep.py's label-matching logic finds them."""
    auth_events = []
    redteam_events = []

    compromised_users = random.sample(USERS, k=NUM_COMPROMISED_USERS)

    test_attack_count = max(1, NUM_COMPROMISED_USERS // 2)
    train_attack_count = NUM_COMPROMISED_USERS - test_attack_count
    train_days = range(TEST_DAY_START)
    test_days = tuple(range(TEST_DAY_START, NUM_DAYS))
    if not test_days:
        raise ValueError("NUM_DAYS must include at least one held-out day.")

    for index, user in enumerate(compromised_users):
        home_pc = USER_HOME_PC[user]
        if index < train_attack_count:
            day = random.choice(train_days)
        else:
            test_index = index - train_attack_count
            day = test_days[test_index % len(test_days)]
        t = off_hour_time(day)

        # Attack chain: own laptop -> a server they don't normally use
        #            -> an application -> the core database
        unfamiliar_servers = [s for s in SERVERS if s not in USER_SERVERS[user]]
        server = random.choice(unfamiliar_servers) if unfamiliar_servers else random.choice(SERVERS)
        app = random.choice(APPLICATIONS)

        chain = [
            (home_pc, server),
            (server, app),
            (app, DATABASE),
        ]

        hop_time = t
        for src, dst in chain:
            row = auth_row(
                hop_time, user, user, src, dst,
                "NTLM", "Network", "Success"
            )
            auth_events.append(row)
            redteam_events.append([hop_time, user, src, dst])
            hop_time += random.randint(30, 180)  # fast lateral movement

    return auth_events, redteam_events


# ---- Write output -----------------------------------------------------------

def write_gz_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    print("[info] generating normal login behavior...")
    normal_events = generate_normal_events()

    print("[info] generating benign anomalies (not malicious, just unusual)...")
    benign_events = generate_benign_anomalies()

    print("[info] generating red-team compromise chains...")
    redteam_auth_events, redteam_labels = generate_redteam_events()

    all_auth_events = normal_events + benign_events + redteam_auth_events
    random.shuffle(all_auth_events)
    all_auth_events.sort(key=lambda r: r[0])  # sort by time, like real LANL data

    print(f"[info] writing {len(all_auth_events):,} auth events to {AUTH_OUT}")
    write_gz_csv(AUTH_OUT, all_auth_events)

    print(f"[info] writing {len(redteam_labels):,} red-team labels to {REDTEAM_OUT}")
    write_gz_csv(REDTEAM_OUT, redteam_labels)

    print("\n[done] synthetic dataset ready.")
    print(f"  - {len(normal_events):,} normal events")
    print(f"  - {len(benign_events):,} benign-but-unusual events (not labeled malicious)")
    print(f"  - {len(redteam_auth_events):,} malicious events across "
          f"{NUM_COMPROMISED_USERS} compromised users")
    print("\nNext: run `python src/data_prep.py` to clean and map this into "
          "the common schema.")
