from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        tokens = list(corpus)

        # 2. For each merge step:
        merges = []
        for _ in range(num_merges):
            if len(tokens) < 2:
                break

            # a. Count frequency of all adjacent token pairs
            pairs = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1

            if not pairs:
                break

            # b. Find the most frequent pair (break ties lexicographically)
            best_count = max(pairs.values())
            candidates = sorted(p for p, c in pairs.items() if c == best_count)
            best = candidates[0]

            # c. Record the merge as [token_a, token_b]
            merges.append([best[0], best[1]])

            # d. Merge all non-overlapping occurrences left to right
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best[0] and tokens[i + 1] == best[1]:
                    new_tokens.append(best[0] + best[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # 3. Return the list of merges performed
        return merges
