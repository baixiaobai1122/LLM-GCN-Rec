"""
Generate GPT-4o-mini content profiles for items based on metadata.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import os
from openai import OpenAI

# 初始化OpenAI client (新版API)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API_KEY = "sk-kQ0JDIHMVVHesdRhS1QTumF4sPNC7R9eC4rt3eudqt74ziVp" 
# BASE_URL = "https://api.sydney-ai.com"                       

# client = OpenAI(
#     api_key=API_KEY,
#     base_url=BASE_URL
# )


CONTENT_PROFILE_PROMPT = """You are a book recommendation expert. Based on the book metadata below, create a rich semantic profile that captures the essence of this book for recommendation purposes.

Book Information:
- Title: {title}
- Authors: {authors}
- Subjects/Topics: {subjects}
- Description: {description}

Create a comprehensive semantic profile (100-120 words) that includes:
1. Core themes and main topics
2. Genre, style, and literary approach
3. Key content elements and narrative features
4. Comparable works and positioning in genre
5. Distinctive characteristics that define this book

Focus on semantic richness to help match with similar books and reader preferences.

Semantic Profile:"""


def load_metadata(metadata_path):
    """Load book metadata from JSON."""
    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} books")
    return metadata


def load_item_mapping(item_list_path):
    """Load item ID to ISBN mapping."""
    print(f"Loading item mapping from {item_list_path}")
    id_to_isbn = {}

    with open(item_list_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                isbn, remap_id = parts
                id_to_isbn[int(remap_id)] = isbn

    print(f"Loaded {len(id_to_isbn)} item mappings")
    return id_to_isbn


def format_book_info(book_data):
    """Format book metadata for prompt."""
    title = book_data.get('title', 'Unknown')

    authors = book_data.get('authors', [])
    if authors:
        authors_str = ", ".join(authors)
    else:
        authors_str = "Unknown"

    subjects = book_data.get('subjects', [])
    if subjects:
        # Handle both dict and string formats
        subject_names = []
        for subj in subjects[:5]:
            if isinstance(subj, dict) and 'name' in subj:
                subject_names.append(subj['name'])
            elif isinstance(subj, str):
                subject_names.append(subj)
        subjects_str = ", ".join(subject_names) if subject_names else "Not specified"
    else:
        subjects_str = "Not specified"

    description = book_data.get('description', '')
    if description and len(description) > 300:
        description = description[:300] + "..."
    elif not description:
        description = "No description available"

    return {
        'title': title,
        'authors': authors_str,
        'subjects': subjects_str,
        'description': description
    }


def generate_profile(book_info, model="gpt-4o-mini", max_retries=3):
    """Generate content profile using GPT."""
    prompt = CONTENT_PROFILE_PROMPT.format(**book_info)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a book recommendation expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=210,    #标注代码的截断长度
                top_p=0.9
            )

            profile = response.choices[0].message.content.strip()
            return profile, response.usage

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"\nError: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\nFailed after {max_retries} attempts: {e}")
                return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Generate GPT content profiles for items"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="GPT model to use"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Process in batches (for progress tracking)"
    )
    parser.add_argument(
        "--rate_limit_delay",
        type=float,
        default=0.05,
        help="Delay between API calls (seconds)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gpt_item_profiles.json",
        help="Output filename"
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Maximum number of items to process (for testing)"
    )

    args = parser.parse_args()

    # Paths
    data_path = Path(args.data_path)
    metadata_path = data_path / "book_metadata.json"
    item_list_path = data_path / "item_list.txt"
    output_path = data_path / args.output

    # Load data
    metadata = load_metadata(metadata_path)
    id_to_isbn = load_item_mapping(item_list_path)

    # Generate profiles
    total_items = len(id_to_isbn)
    num_items = min(args.max_items, total_items) if args.max_items else total_items
    profiles = {}
    total_input_tokens = 0
    total_output_tokens = 0
    successful = 0
    failed = 0

    print(f"\n{'='*70}")
    print(f"Generating GPT content profiles for {num_items} items...")
    if args.max_items and args.max_items < total_items:
        print(f"(Testing mode: processing first {num_items} of {total_items} items)")
    print(f"Model: {args.model}")
    print(f"Rate limit delay: {args.rate_limit_delay}s")
    print(f"{'='*70}\n")

    for item_id in tqdm(range(num_items), desc="Generating profiles"):
        isbn = id_to_isbn.get(item_id)

        if isbn and isbn in metadata:
            book_data = metadata[isbn]
            book_info = format_book_info(book_data)

            profile, usage = generate_profile(book_info, model=args.model)

            if profile:
                profiles[item_id] = profile
                if usage:
                    # 单次调用统计 
                    input_tokens = usage.prompt_tokens
                    output_tokens = usage.completion_tokens
                    total_tokens = usage.total_tokens

                    # 成本计算 (基于OpenAI官方定价)
                    # GPT-4o-mini: $0.150/1M input, $0.600/1M output
                    call_cost = (input_tokens * 0.150 + output_tokens * 0.600) / 1_000_000

                    # 累计统计
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens

                    # 打印单次调用统计 (真实消耗)
                    tqdm.write(f"[Item {item_id}] tokens from API: {input_tokens}+{output_tokens}={total_tokens} | Calculated cost: ${call_cost:.6f}")

                successful += 1
            else:
                # Fallback: use simple description
                profiles[item_id] = f"Book: {book_info['title']}"
                failed += 1
        else:
            # No metadata available
            profiles[item_id] = f"Book item {item_id}"
            failed += 1

        # Rate limiting
        time.sleep(args.rate_limit_delay)

        # Progress update every 100 items
        if (item_id + 1) % 100 == 0:
            current_cost = (total_input_tokens * 0.150 +
                          total_output_tokens * 0.600) / 1_000_000
            avg_input = total_input_tokens / successful if successful > 0 else 0
            avg_output = total_output_tokens / successful if successful > 0 else 0
            tqdm.write(f"\n{'='*70}")
            tqdm.write(f"Progress: {item_id + 1}/{num_items} ({(item_id+1)/num_items*100:.1f}%)")
            tqdm.write(f"Successful: {successful}, Failed: {failed}")
            tqdm.write(f"Total tokens: {total_input_tokens:,} input + {total_output_tokens:,} output = {total_input_tokens+total_output_tokens:,}")
            tqdm.write(f"Avg tokens/item: {avg_input:.0f} input + {avg_output:.0f} output")
            tqdm.write(f"Cost so far: ${current_cost:.4f}")
            
            tqdm.write(f"Checkpoint: Saving {len(profiles)} profiles to {output_path}...")
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(profiles, f, indent=2, ensure_ascii=False)
            except Exception as e:
                tqdm.write(f"[ERROR] Failed to save checkpoint: {e}")

            tqdm.write(f"{'='*70}\n")

    # Save profiles
    print(f"\nSaving profiles to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

    # Calculate cost
    total_cost = (total_input_tokens * 0.150 +
                 total_output_tokens * 0.600) / 1_000_000

    # Save metadata
    metadata_output = {
        "model": args.model,
        "num_items": num_items,
        "successful": successful,
        "failed": failed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_cost": total_cost,
        "description": "GPT-4o-mini content profiles based on book metadata"
    }

    metadata_path_out = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(metadata_path_out, 'w') as f:
        json.dump(metadata_output, f, indent=2)

    print("\n" + "="*70)
    print("✅ Profile generation completed!")
    print("="*70)
    print(f"Total items: {num_items}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Input tokens: {total_input_tokens:,}")
    print(f"Output tokens: {total_output_tokens:,}")
    print(f"Estimated cost: ${total_cost:.4f}")
    print(f"Output: {output_path}")
    print(f"Metadata: {metadata_path_out}")
    print("="*70)


if __name__ == "__main__":
    main()
