import pandas as pd
import random
import os

real_templates = [
    "The government announced a new policy regarding {}.",
    "Scientists discovered {} after years of research.",
    "The Supreme Court delivered a verdict on {}.",
    "The Reserve Bank released a report on {}.",
    "Researchers published findings about {}.",
    "The Ministry of Health launched {}.",
    "Engineers successfully tested {}.",
    "NASA announced progress in {}.",
    "The education department introduced {}.",
    "A new technology for {} was unveiled."
]

fake_templates = [
    "Aliens secretly control {} according to leaked documents.",
    "{} cures every disease in just one day.",
    "Scientists confirm the Earth will stop rotating next week because of {}.",
    "A hidden civilization was found beneath {}.",
    "{} allows humans to become invisible instantly.",
    "Government admits {} was planned decades ago.",
    "Time travelers revealed the future of {}.",
    "NASA confirms dragons were found near {}.",
    "Doctors hide the truth about {}.",
    "{} gives people superpowers instantly."
]

topics = [
    "artificial intelligence",
    "space exploration",
    "renewable energy",
    "quantum computing",
    "healthcare",
    "education",
    "cybersecurity",
    "climate change",
    "financial markets",
    "robotics",
    "medicine",
    "vaccination",
    "elections",
    "sports",
    "Olympics",
    "cricket",
    "football",
    "economy",
    "banking",
    "blockchain"
]

records = []

for _ in range(25000):
    topic = random.choice(topics)

    text = random.choice(real_templates).format(topic)

    text += (
        " Experts verified the information. "
        "Official reports confirmed the announcement."
    )

    records.append({
        "text": text,
        "label": 0
    })

for _ in range(25000):
    topic = random.choice(topics)

    text = random.choice(fake_templates).format(topic)

    text += (
        " The story spread rapidly on social media without any official confirmation."
    )

    records.append({
        "text": text,
        "label": 1
    })

random.shuffle(records)

df = pd.DataFrame(records)

os.makedirs("dataset", exist_ok=True)

df.to_csv("dataset/fake_news.csv", index=False)

print("Dataset created successfully!")

print(df.shape)