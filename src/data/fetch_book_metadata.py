"""
获取Amazon-Book数据集的书籍元数据
使用Open Library API (免费，无需API key)
输出: book_metadata.json
"""
import requests
import json
import time
from tqdm import tqdm
import os

class BookMetadataFetcher:
    def __init__(self, cache_file='../../datasets/amazon-book/book_metadata.json'):
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
        print(f"💾 Saved metadata to {self.cache_file}")

    def fetch_from_openlibrary(self, isbn):
        """
        从Open Library API获取书籍信息
        API: https://openlibrary.org/dev/docs/api/books
        """
        try:
            # Open Library API (免费，无需key)
            url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
            response = requests.get(url, timeout=5)

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
                        'subjects': book_data.get('subjects', [])[:10],  # 只取前10个主题
                        'cover_url': book_data.get('cover', {}).get('medium', None),
                        'description': self._extract_description(book_data),
                    }

                    return metadata

            return None

        except Exception as e:
            print(f"❌ Error fetching {isbn}: {e}")
            return None

    def _extract_description(self, book_data):
        """提取书籍描述"""
        # Open Library有多种描述格式
        if 'description' in book_data:
            desc = book_data['description']
            if isinstance(desc, dict) and 'value' in desc:
                return desc['value']
            elif isinstance(desc, str):
                return desc

        # 如果没有描述，使用主题作为替代
        subjects = book_data.get('subjects', [])[:5]
        if subjects:
            subject_names = [s.get('name', s) if isinstance(s, dict) else s for s in subjects]
            return f"Topics: {', '.join(subject_names)}"

        return "No description available"

    def fetch_all_books(self, isbn_list, batch_size=100, delay=0.5):
        """
        批量获取书籍元数据

        Args:
            isbn_list: ISBN列表
            batch_size: 每个batch保存一次
            delay: 请求间隔（秒），避免被限流
        """
        total = len(isbn_list)
        fetched = 0
        failed = []

        print(f"📚 Fetching metadata for {total} books...")
        print(f"   Already cached: {len(self.metadata)}")

        for i, isbn in enumerate(tqdm(isbn_list, desc="Fetching books")):
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
                print(f"\n   Progress: {i+1}/{total}, Fetched: {fetched}, Failed: {len(failed)}")

            # 延迟，避免限流
            time.sleep(delay)

        # 最终保存
        self.save_cache()

        print(f"\n✅ Fetching complete!")
        print(f"   Total fetched: {fetched}")
        print(f"   Total cached: {len(self.metadata)}")
        print(f"   Failed: {len(failed)}")

        if failed:
            print(f"\n❌ Failed ISBNs (first 10): {failed[:10]}")
            failed_file = self.cache_file.replace('.json', '_failed.txt')
            with open(failed_file, 'w') as f:
                f.write('\n'.join(failed))
            print(f"   Failed ISBNs saved to: {failed_file}")

        return self.metadata

def load_isbn_list(item_list_file='../../datasets/amazon-book/item_list.txt', n_books=None):
    """从item_list.txt加载ISBN列表"""
    isbns = []
    with open(item_list_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:  # 跳过header
                continue
            if n_books and len(isbns) >= n_books:
                break
            parts = line.strip().split()
            if len(parts) >= 1:
                isbns.append(parts[0])  # org_id (ISBN)

    print(f"📖 Loaded {len(isbns)} ISBNs from {item_list_file}")
    return isbns

def main():
    """主流程"""
    import argparse
    parser = argparse.ArgumentParser(description='Fetch book metadata from Open Library')
    parser.add_argument('--n_books', type=int, default=1000,
                        help='Number of books to fetch (default: 1000)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: datasets/amazon-book_N)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between requests (default: 0.5s)')
    args = parser.parse_args()

    # 1. 加载ISBN列表
    isbns = load_isbn_list(n_books=args.n_books)

    # 2. 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f'../../datasets/amazon-book_{args.n_books}'

    cache_file = os.path.join(os.path.dirname(__file__), output_dir, 'book_metadata.json')

    # 3. 创建fetcher
    fetcher = BookMetadataFetcher(cache_file=cache_file)

    # 4. 获取元数据
    print(f"📌 Fetching {len(isbns)} books")
    print(f"📁 Output: {cache_file}")
    print(f"⏱️  Estimated time: ~{len(isbns) * args.delay / 60:.1f} minutes\n")

    # 开始获取
    metadata = fetcher.fetch_all_books(isbns, batch_size=100, delay=args.delay)

    print(f"\n✅ Done! Metadata saved to {fetcher.cache_file}")

    # 4. 显示示例
    print("\n📄 Sample metadata:")
    for isbn, meta in list(metadata.items())[:3]:
        print(f"\nISBN: {isbn}")
        print(f"  Title: {meta.get('title', 'N/A')}")
        print(f"  Authors: {', '.join(meta.get('authors', []))}")
        print(f"  Description: {meta.get('description', 'N/A')[:100]}...")
        print(f"  Cover: {meta.get('cover_url', 'N/A')}")

if __name__ == "__main__":
    main()
