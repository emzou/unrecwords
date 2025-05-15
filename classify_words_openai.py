import pandas as pd
from openai import OpenAI
import time
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="2013_2023_pol_unrecword_list.txt")
parser.add_argument("--output", type=str, default="round6_classified_words.csv")
parser.add_argument("--batch", type=int, default=10)
parser.add_argument("--model", type=str, default="gpt-4o")
parser.add_argument("--start", type=int, default=0)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = OpenAI(api_key="")

with open(args.input, "r") as f:
    all_words = [line.strip() for line in f if line.strip()]

try:
    df_out = pd.read_csv(args.output)
    processed_words = set(df_out['word'])
except FileNotFoundError:
    df_out = pd.DataFrame(columns=["word", "label"])
    processed_words = set()

remaining_words = [w for w in all_words if w not in processed_words]

for i in range(args.start * args.batch, len(remaining_words), args.batch):
    batch = remaining_words[i:i + args.batch]
    logging.info(f"Processing batch {i // args.batch + 1} with {len(batch)} words")

    numbered_words = [f"{j+1}. {word}" for j, word in enumerate(batch)]
    prompt = (
        "Classify the following words. Return 1 if the word is a formal or conceptual compound (like 'africanamerican', 'prodiversity'), "
        "and 0 if it is slang, memetic, informal, or offensive (like 'shitalian', 'fashwave').\n"
        "Respond ONLY with the labels (0 or 1), one per line, in the same order as the input list. Do not repeat the words.\n\n"
        + "\n".join(numbered_words)
    )

    try:
        res = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000
        )
        result = res.choices[0].message.content
    except Exception as e:
        logging.error(f"API error: {e}")
        continue

    if hasattr(res, 'moderation_result') and res.moderation_result.flagged:
        logging.warning("Moderation flagged this batch. Skipping.")
        continue

    lines = result.strip().split("\n")
    if len(lines) != len(batch):
        logging.warning("Mismatch between number of words and labels. Skipping batch.")
        continue

    try:
        data = [(word, int(label.strip())) for word, label in zip(batch, lines)]
    except Exception as e:
        logging.warning(f"Label parsing error: {e}. Skipping batch.")
        continue

    pd.DataFrame(data, columns=["word", "label"]).to_csv(
        args.output, mode='a', header=not processed_words, index=False
    )
    processed_words.update(w for w, _ in data)
    logging.info(f"Saved {len(data)} words to {args.output}")

    time.sleep(1)