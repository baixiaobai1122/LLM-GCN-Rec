"""
为 amazon-book-2023 数据集获取书籍元数据
从 amazon_item.json 读取 ASIN，使用 Open Library API 获取书籍信息
输出: book_metadata.json
"""
import requests
import json
import time
from tqdm import tqdm
import os
from pathlib import Path

class BookMetadataFetcher2023:
    def __init__(self, data_dir='../../datasets/amazon-book-2023'):
        self.data_dir = Path(data_dir)
        self.cache_file = self.data_dir / 'book_metadata.json'
        os.makedirs(self.data_dir, exist_ok=True)
        self.metadata = self.load_cache()

    def load_cache(self):
        """加载已缓存的元数据"""
        if self.cache_file.exists():
            print(f"📂 Loading cached metadata from {self.cache_file}")
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_cache(self):
        """保存元数据到缓存"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved metadata to {self.cache_file}")

    def fetch_from_openlibrary_by_isbn(self, isbn_or_asin):
        """
        从 Open Library API 获取书籍信息 (使用 ISBN)
        API: https://openlibrary.org/api/books
        """
        try:
            # 尝试 ISBN 查询
            url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn_or_asin}&format=json&jscmd=data"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                key = f"ISBN:{isbn_or_asin}"

                if key in data and data[key]:
                    return self._extract_metadata(data[key], isbn_or_asin)

            return None

        except Exception as e:
            # print(f"❌ Error fetching {isbn_or_asin}: {e}")
            return None

    def fetch_from_openlibrary_search(self, asin):
        """
        使用 Open Library Search API 查询
        当 ISBN lookup 失败时使用此方法
        """
        try:
            # 使用 search API
            url = f"https://openlibrary.org/search.json?q={asin}&limit=1"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()

                if data.get('numFound', 0) > 0 and len(data.get('docs', [])) > 0:
                    book = data['docs'][0]

                    # 提取信息
                    metadata = {
                        'isbn': asin,
                        'title': book.get('title', 'Unknown'),
                        'authors': book.get('author_name', ['Unknown']),
                        'publish_date': str(book.get('first_publish_year', '')),
                        'publishers': book.get('publisher', [])[:3],  # 取前3个
                        'subjects': [{'name': s, 'url': ''} for s in book.get('subject', [])[:10]],
                        'cover_url': f"https://covers.openlibrary.org/b/id/{book.get('cover_i', '')}-M.jpg" if book.get('cover_i') else None,
                        'description': self._format_subjects_as_description(book.get('subject', [])[:5]),
                    }

                    return metadata

            return None

        except Exception as e:
            # print(f"❌ Search error for {asin}: {e}")
            return None

    def _extract_metadata(self, book_data, isbn_or_asin):
        """从 Open Library 数据提取元数据"""
        metadata = {
            'isbn': isbn_or_asin,
            'title': book_data.get('title', 'Unknown'),
            'authors': [author.get('name', '') for author in book_data.get('authors', [])],
            'publish_date': book_data.get('publish_date', ''),
            'publishers': [p.get('name', '') for p in book_data.get('publishers', [])],
            'subjects': book_data.get('subjects', [])[:10],  # 只取前10个主题
            'cover_url': book_data.get('cover', {}).get('medium', None),
            'description': self._extract_description(book_data),
        }
        return metadata

    def _extract_description(self, book_data):
        """提取书籍描述"""
        # Open Library 有多种描述格式
        if 'description' in book_data:
            desc = book_data['description']
            if isinstance(desc, dict) and 'value' in desc:
                return desc['value']
            elif isinstance(desc, str):
                return desc

        # 如果没有描述，使用主题作为替代
        subjects = book_data.get('subjects', [])[:5]
        return self._format_subjects_as_description(subjects)

    def _format_subjects_as_description(self, subjects):
        """将主题格式化为描述"""
        if subjects:
            subject_names = [s.get('name', s) if isinstance(s, dict) else s for s in subjects]
            return f"Topics: {', '.join(subject_names)}"
        return "No description available"

    def fetch_all_books(self, asin_list, batch_size=100, delay=0.5):
        """
        批量获取书籍元数据

        Args:
            asin_list: [(iid, asin), ...] 列表
            batch_size: 每个batch保存一次
            delay: 请求间隔（秒），避免被限流
        """
        total = len(asin_list)
        fetched = 0
        failed = []

        # 创建 ITEM_X 到 metadata 的映射
        result_metadata = {}

        print(f"📚 Fetching metadata for {total} books...")
        print(f"   Already cached: {len(self.metadata)}")

        for i, (iid, asin) in enumerate(tqdm(asin_list, desc="Fetching books")):
            item_key = f"ITEM_{iid}"

            # 跳过已缓存的
            if item_key in self.metadata:
                result_metadata[item_key] = self.metadata[item_key]
                continue

            # 先尝试 ISBN lookup
            metadata = self.fetch_from_openlibrary_by_isbn(asin)

            # 如果失败，尝试 search API
            if not metadata:
                time.sleep(0.2)  # 短暂延迟
                metadata = self.fetch_from_openlibrary_search(asin)

            if metadata:
                result_metadata[item_key] = metadata
                self.metadata[item_key] = metadata
                fetched += 1
            else:
                # 创建最小元数据作为 fallback
                result_metadata[item_key] = {
                    'isbn': asin,
                    'title': f"Book {iid}",
                    'authors': ['Unknown'],
                    'publish_date': '',
                    'publishers': [],
                    'subjects': [],
                    'cover_url': None,
                    'description': f"Amazon ASIN: {asin}"
                }
                self.metadata[item_key] = result_metadata[item_key]
                failed.append((iid, asin))

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
        print(f"   Total in cache: {len(self.metadata)}")
        print(f"   Failed: {len(failed)}")

        if failed:
            print(f"\n❌ Failed ASINs (first 10): {failed[:10]}")
            failed_file = self.data_dir / 'failed_asins.txt'
            with open(failed_file, 'w') as f:
                for iid, asin in failed:
                    f.write(f"{iid}\t{asin}\n")
            print(f"   Failed ASINs saved to: {failed_file}")

        return self.metadata


def load_asin_list(data_dir='../../datasets/amazon-book-2023'):
    """从 amazon_item.json 加载 ASIN 列表"""
    data_dir = Path(data_dir)
    item_file = data_dir / 'amazon_item.json'

    asin_list = []
    with open(item_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            asin_list.append((item['iid'], item['asin']))

    print(f"📖 Loaded {len(asin_list)} ASINs from {item_file}")
    return asin_list


def main():
    """主流程"""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch book metadata for amazon-book-2023")
    parser.add_argument('--data_dir', type=str,
                       default='../../datasets/amazon-book-2023',
                       help='Data directory path')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Save cache every N books')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between requests (seconds)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of books to fetch (for testing)')

    args = parser.parse_args()

    print("="*60)
    print("Amazon Book 2023 Metadata Fetcher")
    print("="*60)

    # 加载 ASIN 列表
    asin_list = load_asin_list(args.data_dir)

    if args.limit:
        print(f"⚠️  Limiting to first {args.limit} books for testing")
        asin_list = asin_list[:args.limit]

    # 创建 fetcher 并获取元数据
    fetcher = BookMetadataFetcher2023(args.data_dir)
    metadata = fetcher.fetch_all_books(
        asin_list,
        batch_size=args.batch_size,
        delay=args.delay
    )

    print("\n" + "="*60)
    print("✓ Metadata fetching completed!")
    print(f"  Output: {fetcher.cache_file}")
    print(f"  Total items: {len(metadata)}")
    print("="*60)


if __name__ == '__main__':
    main()