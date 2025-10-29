"""
Generate structured GPT-4o-mini content profiles for items based on metadata.
Uses OpenAI structured outputs to ensure consistent format across 6 semantic aspects.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import os
from openai import OpenAI
from pydantic import BaseModel

# 初始化OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class BookProfile(BaseModel):
    """Structured book profile with 6 semantic aspects."""
    core_themes: str  # Core themes and main topics
    genre_style: str  # Genre, style, and literary approach
    content_features: str  # Key content elements and narrative features
    comparable_works: str  # Comparable works and positioning in genre
    distinctive_traits: str  # Distinctive characteristics that define this book
    target_readers: str  # Suitable reader types and demographics


STRUCTURED_PROFILE_PROMPT = """You are a book recommendation expert. Based on the book metadata below, create a rich semantic profile that captures the essence of this book for recommendation purposes.

Book Information:
- Title: {title}
- Authors: {authors}
- Subjects/Topics: {subjects}
- Description: {description}

Analyze this book and provide responses for each of the following 6 aspects. Keep each response concise (15-30 words), focused on semantic meaning without filler words:

1. **core_themes**: Identify the central themes, main topics, and conceptual focus of the book
2. **genre_style**: Describe the genre classification, writing style, and literary approach
3. **content_features**: Highlight key content elements, narrative structure, and storytelling features
4. **comparable_works**: Name similar books, authors, or position within the genre landscape
5. **distinctive_traits**: Identify unique characteristics, special features, or what sets this book apart
6. **target_readers**: Describe the ideal reader profile, demographics, interests, or reading preferences

Provide dense, information-rich responses without connecting phrases like "this book is" or "the reader will"."""


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


def generate_structured_profile(book_info, model="gpt-4o-mini", max_retries=3):
    """Generate structured content profile using GPT with structured outputs."""
    prompt = STRUCTURED_PROFILE_PROMPT.format(**book_info)

    for attempt in range(max_retries):
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a book recommendation expert creating structured semantic profiles."},
                    {"role": "user", "content": prompt}
                ],
                response_format=BookProfile,
                temperature=0.7,
                max_tokens=300,
                top_p=0.9
            )

            profile_obj = response.choices[0].message.parsed

            # # Debug: print what we got
            # if attempt == 0:
            #     print(f"\n[DEBUG] Response type: {type(profile_obj)}")
            #     print(f"[DEBUG] Response content: {profile_obj}")

            # Use model_dump() for Pydantic v2 or dict() for v1
            try:
                profile_dict = profile_obj.model_dump()  # Pydantic v2
            except AttributeError:
                profile_dict = profile_obj.dict()  # Pydantic v1

            # Validate all required fields are present
            required_fields = ["core_themes", "genre_style", "content_features",
                             "comparable_works", "distinctive_traits", "target_readers"]
            missing_fields = [f for f in required_fields if f not in profile_dict]

            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")

            return profile_dict, response.usage

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
        description="Generate structured GPT content profiles for items"
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
        default="gpt_item_profiles_structured.json",
        help="Output filename"
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Maximum number of items to process (for testing)"
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="Start from this item ID (for resuming)"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
        help="Save checkpoint every N items (default: 100)"
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

    # Load existing profiles if resuming
    profiles = {}
    if args.start_from > 0 and output_path.exists():
        print(f"Loading existing profiles from {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_profiles = json.load(f)
            # Convert string keys to int
            profiles = {int(k): v for k, v in existing_profiles.items()}
        print(f"Loaded {len(profiles)} existing profiles")

    # Generate profiles
    total_items = len(id_to_isbn)
    num_items = min(args.max_items, total_items) if args.max_items else total_items
    total_input_tokens = 0
    total_output_tokens = 0
    successful = 0
    failed = 0

    print(f"\n{'='*70}")
    print(f"Generating structured GPT content profiles for {num_items} items...")
    print(f"Output format: 6 semantic aspects (structured)")
    if args.start_from > 0:
        print(f"(Resuming from item {args.start_from})")
    if args.max_items and args.max_items < total_items:
        print(f"(Testing mode: processing first {num_items} of {total_items} items)")
    print(f"Model: {args.model}")
    print(f"Rate limit delay: {args.rate_limit_delay}s")
    print(f"Checkpoint interval: every {args.checkpoint_interval} items")
    print(f"{'='*70}\n")

    for item_id in tqdm(range(args.start_from, num_items), desc="Generating profiles", initial=args.start_from, total=num_items):
        isbn = id_to_isbn.get(item_id)

        if isbn and isbn in metadata:
            book_data = metadata[isbn]
            book_info = format_book_info(book_data)

            profile_dict, usage = generate_structured_profile(book_info, model=args.model)

            if profile_dict:
                profiles[item_id] = profile_dict
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

                    # 打印单次调用统计
                    tqdm.write(f"[Item {item_id}] tokens: {input_tokens}+{output_tokens}={total_tokens} | Cost: ${call_cost:.6f}")

                successful += 1
            else:
                # Fallback: use simple structured format
                profiles[item_id] = {
                    "core_themes": f"Book: {book_info['title']}",
                    "genre_style": "Unknown",
                    "content_features": "No metadata available",
                    "comparable_works": "N/A",
                    "distinctive_traits": "N/A",
                    "target_readers": "General readers"
                }
                failed += 1
        else:
            # No metadata available
            profiles[item_id] = {
                "core_themes": f"Book item {item_id}",
                "genre_style": "Unknown",
                "content_features": "No metadata available",
                "comparable_works": "N/A",
                "distinctive_traits": "N/A",
                "target_readers": "General readers"
            }
            failed += 1

        # Rate limiting
        time.sleep(args.rate_limit_delay)

        # Checkpoint: Save progress at regular intervals
        if (item_id + 1) % args.checkpoint_interval == 0 or (item_id + 1) == num_items:
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

            tqdm.write(f"\n[CHECKPOINT] Saving {len(profiles)} profiles to {output_path}...")
            try:
                # Save main output file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(profiles, f, indent=2, ensure_ascii=False)

                # Save checkpoint metadata
                checkpoint_meta = {
                    "last_item_id": item_id,
                    "total_items_processed": item_id + 1,
                    "successful": successful,
                    "failed": failed,
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "current_cost": current_cost,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                checkpoint_meta_path = output_path.parent / f"{output_path.stem}_checkpoint.json"
                with open(checkpoint_meta_path, 'w') as f:
                    json.dump(checkpoint_meta, f, indent=2)

                tqdm.write(f"[SUCCESS] Checkpoint saved successfully!")

            except Exception as e:
                tqdm.write(f"[ERROR] Failed to save checkpoint: {e}")

            tqdm.write(f"{'='*70}\n")

    # Save profiles
    print(f"\nSaving structured profiles to {output_path}")
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
        "format": "structured_6_aspects",
        "aspects": [
            "core_themes",
            "genre_style",
            "content_features",
            "comparable_works",
            "distinctive_traits",
            "target_readers"
        ],
        "description": "Structured GPT-4o-mini profiles with 6 semantic aspects for recommendation systems"
    }

    metadata_path_out = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(metadata_path_out, 'w') as f:
        json.dump(metadata_output, f, indent=2)

    print("\n" + "="*70)
    print("✅ Structured profile generation completed!")
    print("="*70)
    print(f"Total items: {num_items}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Input tokens: {total_input_tokens:,}")
    print(f"Output tokens: {total_output_tokens:,}")
    print(f"Estimated cost: ${total_cost:.4f}")
    print(f"Output: {output_path}")
    print(f"Metadata: {metadata_path_out}")
    print(f"Format: 6 structured semantic aspects per item")
    print("="*70)


if __name__ == "__main__":
    main()