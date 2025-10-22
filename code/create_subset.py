"""
创建Amazon-Book的子数据集
只保留前N本书及其相关交互
"""
import os
import json
from collections import defaultdict

def create_subset(n_items=1500, min_interactions=5):
    """
    创建子数据集

    Args:
        n_items: 保留前N个物品
        min_interactions: 用户最少交互次数（过滤掉交互太少的用户）
    """
    data_dir = '../data/amazon-book'

    print(f"📊 Creating subset with first {n_items} items")
    print(f"   Min user interactions: {min_interactions}")

    # 1. 读取train.txt，过滤交互
    print("\n📖 Processing train.txt...")
    train_data = {}
    total_interactions = 0
    filtered_interactions = 0

    with open(f'{data_dir}/train.txt', 'r') as f:
        for user_id, line in enumerate(f):
            items = list(map(int, line.strip().split()))
            total_interactions += len(items)

            # 只保留 < n_items 的物品
            filtered_items = [item for item in items if item < n_items]

            # 过滤掉交互太少的用户
            if len(filtered_items) >= min_interactions:
                train_data[user_id] = filtered_items
                filtered_interactions += len(filtered_items)

    print(f"   Original interactions: {total_interactions}")
    print(f"   Filtered interactions: {filtered_interactions}")
    print(f"   Retention rate: {filtered_interactions/total_interactions*100:.1f}%")

    # 2. 读取test.txt，过滤交互
    print("\n📖 Processing test.txt...")
    test_data = {}
    total_test = 0
    filtered_test = 0

    with open(f'{data_dir}/test.txt', 'r') as f:
        for user_id, line in enumerate(f):
            items = list(map(int, line.strip().split()))
            total_test += len(items)

            # 只保留在train_data中的用户
            if user_id in train_data:
                # 只保留 < n_items 的物品
                filtered_items = [item for item in items if item < n_items]

                if len(filtered_items) > 0:
                    test_data[user_id] = filtered_items
                    filtered_test += len(filtered_items)

    print(f"   Original test interactions: {total_test}")
    print(f"   Filtered test interactions: {filtered_test}")
    print(f"   Retention rate: {filtered_test/total_test*100:.1f}%")

    # 3. 重新编号用户ID（连续化）
    print("\n🔄 Remapping user IDs...")
    old_to_new_user = {}
    new_user_id = 0
    for old_user_id in sorted(train_data.keys()):
        old_to_new_user[old_user_id] = new_user_id
        new_user_id += 1

    n_users = len(old_to_new_user)
    print(f"   Total users: {n_users}")

    # 4. 创建输出目录
    subset_dir = f'{data_dir}_subset_{n_items}'
    os.makedirs(subset_dir, exist_ok=True)
    print(f"\n📁 Creating subset in: {subset_dir}")

    # 5. 写入train.txt（新格式）
    with open(f'{subset_dir}/train.txt', 'w') as f:
        for old_user_id in sorted(train_data.keys()):
            new_user_id = old_to_new_user[old_user_id]
            items = train_data[old_user_id]
            f.write(f"{new_user_id} " + " ".join(map(str, items)) + "\n")

    print(f"✅ Created {subset_dir}/train.txt")

    # 6. 写入test.txt
    with open(f'{subset_dir}/test.txt', 'w') as f:
        for old_user_id in sorted(test_data.keys()):
            new_user_id = old_to_new_user[old_user_id]
            items = test_data[old_user_id]
            f.write(f"{new_user_id} " + " ".join(map(str, items)) + "\n")

    print(f"✅ Created {subset_dir}/test.txt")

    # 7. 创建item_list.txt（前n_items个）
    with open(f'{data_dir}/item_list.txt', 'r') as f_in:
        with open(f'{subset_dir}/item_list.txt', 'w') as f_out:
            for i, line in enumerate(f_in):
                if i == 0:  # header
                    f_out.write(line)
                elif i <= n_items:  # 前n_items行
                    f_out.write(line)
                else:
                    break

    print(f"✅ Created {subset_dir}/item_list.txt")

    # 8. 创建user_list.txt
    with open(f'{subset_dir}/user_list.txt', 'w') as f:
        f.write("org_id remap_id\n")
        for old_user_id, new_user_id in sorted(old_to_new_user.items(), key=lambda x: x[1]):
            f.write(f"{old_user_id} {new_user_id}\n")

    print(f"✅ Created {subset_dir}/user_list.txt")

    # 9. 复制元数据文件
    import shutil
    if os.path.exists(f'{data_dir}/book_metadata.json'):
        shutil.copy(f'{data_dir}/book_metadata.json', f'{subset_dir}/book_metadata.json')
        print(f"✅ Copied book_metadata.json")

    # 10. 生成统计报告
    print("\n" + "="*50)
    print("📊 SUBSET STATISTICS")
    print("="*50)
    print(f"Users:                 {n_users:,}")
    print(f"Items:                 {n_items:,}")
    print(f"Train interactions:    {filtered_interactions:,}")
    print(f"Test interactions:     {filtered_test:,}")
    print(f"Total interactions:    {filtered_interactions + filtered_test:,}")
    print(f"Sparsity:              {(filtered_interactions + filtered_test) / (n_users * n_items) * 100:.4f}%")
    print(f"Avg interactions/user: {(filtered_interactions + filtered_test) / n_users:.1f}")
    print("="*50)

    # 11. 保存配置
    config = {
        'n_users': n_users,
        'n_items': n_items,
        'train_interactions': filtered_interactions,
        'test_interactions': filtered_test,
        'min_interactions': min_interactions,
        'original_dataset': 'amazon-book',
        'subset_dir': subset_dir
    }

    with open(f'{subset_dir}/subset_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Subset created successfully!")
    print(f"📂 Location: {subset_dir}")
    print(f"\n🚀 To train LightGCN on this subset, run:")
    print(f"   python main.py --model=lgn --dataset=amazon-book_subset_{n_items} --layer=3 --recdim=64")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-items', type=int, default=1500,
                        help='Number of items to keep (default: 1500)')
    parser.add_argument('--min-interactions', type=int, default=5,
                        help='Minimum interactions per user (default: 5)')
    args = parser.parse_args()

    create_subset(n_items=args.n_items, min_interactions=args.min_interactions)

if __name__ == "__main__":
    main()
