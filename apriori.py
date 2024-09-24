import itertools

# Function to read transactions from the file
def read_transactions(filename):
    transactions = []
    with open(filename, 'r') as file:
        for line in file:
            transactions.append(set(line.strip().split()))
    return transactions

# Function to count the support of each itemset in the transactions
def get_support_count(transactions, itemset):
    return sum(1 for transaction in transactions if itemset.issubset(transaction))

# Function to get frequent itemsets that meet the minimum support
def get_frequent_itemsets(transactions, candidates, min_support):
    frequent_itemsets = []
    itemset_counts = {}
    
    for itemset in candidates:
        support_count = get_support_count(transactions, itemset)
        if support_count >= min_support:
            frequent_itemsets.append(itemset)
            itemset_counts[itemset] = support_count
    
    return frequent_itemsets, itemset_counts

# The main Apriori algorithm function
def apriori(filename, min_support_percentage):
    # Step 1: Read transactions
    transactions = read_transactions(filename)
    num_transactions = len(transactions)
    
    # Step 2: Convert percentage to actual support count
    min_support = (min_support_percentage / 100) * num_transactions
    
    # Step 3: Generate frequent 1-itemsets (single items)
    items = set(itertools.chain(*transactions))
    candidates = [frozenset([item]) for item in items]
    
    frequent_itemsets, itemset_counts = get_frequent_itemsets(transactions, candidates, min_support)
    all_frequent_itemsets = frequent_itemsets[:]
    
    # Step 4: Generate and prune candidate k-itemsets for k = 2, 3, ...
    k = 2
    while frequent_itemsets:
        # Generate k-itemset candidates
        candidates = [frozenset(x) for x in itertools.combinations(set(itertools.chain(*frequent_itemsets)), k)]
        frequent_itemsets, counts = get_frequent_itemsets(transactions, candidates, min_support)
        all_frequent_itemsets.extend(frequent_itemsets)
        itemset_counts.update(counts)
        k += 1
    
    # Step 5: Output frequent itemsets and their relative support
    print("Frequent Itemsets:")
    for itemset, count in itemset_counts.items():
        print(f"{set(itemset)}: {count/num_transactions:.2%}")

if __name__ == '__main__':
    filename = input("Enter the filename: ")
    min_support = float(input("Enter the minimum support percentage (e.g., 10, 20): "))
    apriori(filename, min_support)
