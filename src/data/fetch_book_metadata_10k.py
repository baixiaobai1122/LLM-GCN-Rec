"""
获取Amazon-Book数据集的书籍元数据 - 10000本书版本
使用Open Library API (免费，无需API key)
输出: book_metadata_10k.json
"""
import requests
import json
import time
from tqdm import tqdm
import os
import sys

class BookMetadataFetcher:
    def __init__(self, cache_file='../../datasets/amazon-book_10k/book_metadata.json'):
        self.cache_file = cache_file
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        self.metadata = self.load_cache()

    def load_cache(self):
        """加载已缓存的元数据"""
        if os.path.exists(self.cache_file):
            print(f"📂 Loading cached metadata from {self.cache_file}")
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_cache(self):
        """保存元数据到缓存"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(self.metadata)} books to {self.cache_file}")

    def fetch_from_openlibrary(self, isbn):
        """从Open Library API获取书籍信息"""
        try:
            url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                key = f"ISBN:{isbn}"

                if key in data:
                    book_data = data[key]

                    # 提取关键信息
                    metadata = {
                        'isbn': isbn,
                        'title': book_data.get('title', 'Unknown'),
                        'authors': [author.get('name', '') for author in book_data.get('authors', [])],
                        'publish_date': book_data.get('publish_date', ''),
                        'publishers': [p.get('name', '') for p in book_data.get('publishers', [])],
                        'subjects': book_data.get('subjects', [])[:10],
                        'cover_url': book_data.get('cover', {}).get('medium', None),
                        'description': self._extract_description(book_data),
                    }
                    return metadata

            return None

        except Exception as e:
            return None

    def _extract_description(self, book_data):
        """提取书籍描述"""
        if 'description' in book_data:
            desc = book_data['description']
            if isinstance(desc, dict) and 'value' in desc:
                return desc['value']
            elif isinstance(desc, str):
                return desc

        subjects = book_data.get('subjects', [])[:5]
        if subjects:
            subject_names = [s.get('name', s) if isinstance(s, dict) else s for s in subjects]
            return f"Topics: {', '.join(subject_names)}"

        return "No description available"

    def fetch_all_books(self, isbn_list, batch_size=100, delay=0.5):
        """批量获取书籍元数据"""
        total = len(isbn_list)
        fetched = 0
        failed = []

        print(f"\n📚 Fetching metadata for {total} books...")
        print(f"   Already cached: {len(self.metadata)}")
        print(f"   To fetch: {total - len([i for i in isbn_list if i in self.metadata])}")
        print(f"   Estimated time: ~{(total * delay / 60):.1f} minutes\n")

        for i, isbn in enumerate(tqdm(isbn_list, desc="Fetching")):
            # 跳过已缓存的
            if isbn in self.metadata:
                continue

            # 获取元数据
            metadata = self.fetch_from_openlibrary(isbn)

            if metadata:
                self.metadata[isbn] = metadata
                fetched += 1
            else:
                failed.append(isbn)

            # 定期保存
            if (i + 1) % batch_size == 0:
                self.save_cache()

            # 延迟，避免限流
            time.sleep(delay)

        # 最终保存
        self.save_cache()

        print(f"\n✅ Fetching complete!")
        print(f"   Newly fetched: {fetched}")
        print(f"   Total cached: {len(self.metadata)}")
        print(f"   Failed: {len(failed)}")

        if failed:
            failed_file = self.cache_file.replace('.json', '_failed.txt')
            with open(failed_file, 'w') as f:
                f.write('\n'.join(failed))
            print(f"   Failed ISBNs saved to: {failed_file}")

        return self.metadata

def load_isbn_list(item_list_file, n_books=10000):
    """从item_list.txt加载前N本书的ISBN"""
    isbns = []
    with open(item_list_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:  # 跳过header
                continue
            if len(isbns) >= n_books:
                break
            parts = line.strip().split()
            if len(parts) >= 1:
                isbns.append(parts[0])  # org_id (ISBN)

    print(f"📖 Loaded {len(isbns)} ISBNs from {item_list_file}")
    return isbns

def main():
    """主流程"""
    import argparse
    parser = argparse.ArgumentParser(description='Fetch 10K book metadata')
    parser.add_argument('--n_books', type=int, default=10000,
                        help='Number of books to fetch (default: 10000)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between requests in seconds (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Save every N books (default: 100)')
    args = parser.parse_args()

    print("="*70)
    print(f"📚 Book Metadata Fetcher - {args.n_books} Books")
    print("="*70)

    # 1. 加载ISBN列表
    item_list = os.path.join(os.path.dirname(__file__),
                             '../../datasets/amazon-book/item_list.txt')
    isbns = load_isbn_list(item_list, n_books=args.n_books)

    # 2. 创建fetcher
    cache_file = os.path.join(os.path.dirname(__file__),
                             f'../../datasets/amazon-book_{args.n_books}/book_metadata.json')
    fetcher = BookMetadataFetcher(cache_file=cache_file)

    # 3. 获取元数据
    metadata = fetcher.fetch_all_books(isbns,
                                      batch_size=args.batch_size,
                                      delay=args.delay)

    print(f"\n✅ Done! Metadata saved to {fetcher.cache_file}")

    # 4. 显示统计
    print(f"\n📊 Statistics:")
    print(f"   Total books: {len(metadata)}")

    with_desc = sum(1 for m in metadata.values()
                   if m.get('description') and 'No description' not in m.get('description', ''))
    with_authors = sum(1 for m in metadata.values() if m.get('authors'))
    with_cover = sum(1 for m in metadata.values() if m.get('cover_url'))

    print(f"   With description: {with_desc} ({with_desc/len(metadata)*100:.1f}%)")
    print(f"   With authors: {with_authors} ({with_authors/len(metadata)*100:.1f}%)")
    print(f"   With cover: {with_cover} ({with_cover/len(metadata)*100:.1f}%)")

    # 5. 显示示例
    print(f"\n📄 Sample metadata (first 3 books):")
    for isbn, meta in list(metadata.items())[:3]:
        print(f"\n  ISBN: {isbn}")
        print(f"    Title: {meta.get('title', 'N/A')}")
        print(f"    Authors: {', '.join(meta.get('authors', [])) or 'N/A'}")
        desc = meta.get('description', 'N/A')
        print(f"    Description: {desc[:80]}..." if len(desc) > 80 else f"    Description: {desc}")

if __name__ == "__main__":
    main()
