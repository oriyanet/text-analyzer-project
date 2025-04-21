import json


class SentenceGrouping:
    """Task 9: Groups sentences based on shared words."""

    def __init__(self, processed_sentences: list, threshold: int) -> None:
        """Initializes with processed sentences and threshold."""
        self.processed_sentences = processed_sentences
        self.threshold = threshold
        self.graph = self.build_graph()

    def build_graph(self) -> dict:
        """Builds a graph of sentences connected by shared words."""
        graph = {}
        for i, sentence1 in enumerate(self.processed_sentences):
            for j, sentence2 in enumerate(self.processed_sentences):
                if i >= j:  # Avoid self-comparisons
                    continue
                shared_words = len(set(sentence1).intersection(set(sentence2)))
                if shared_words >= self.threshold:
                    if i not in graph:
                        graph[i] = []  # Initialize an empty list if the sentence node is not in the graph.
                    if j not in graph:
                        graph[j] = []
                    graph[i].append(j)  # Add a bidirectional connection between the sentences.
                    graph[j].append(i)
        return graph

    def find_groups(self) -> list:
        """Finds groups of connected sentences."""
        visited = set()
        groups = []

        def dfs(node, group):
            visited.add(node)
            group.append(node)
            for neighbor in self.graph.get(node, []):  # Explore connected sentences
                if neighbor not in visited:
                    dfs(neighbor, group)

        for node in range(len(self.processed_sentences)):
            if node not in visited:  # Start a new group if the sentence is unvisited.
                group = []
                dfs(node, group)
                groups.append(group)

        # Sort groups by size and alphabetically by sentence text
        groups = [sorted(group, key=lambda idx: self.processed_sentences[idx]) for group in groups]
        groups.sort(key=lambda g: (len(g), self.processed_sentences[g[0]]))
        return groups

    def group_sentences(self) -> list:
        groups = self.find_groups()  # Identify sentence groups based on shared words.
        group_list = []
        # Iterate through each group to format the output.
        for idx, group in enumerate(groups, start=1):
            group_name = f"Group {idx}"
            sentences = [self.processed_sentences[i] for i in group]
            group_list.append([group_name, sentences])

        return group_list

    @staticmethod
    def generate_output(grouped_sentences: list) -> None:
        """Prints the grouped sentences in the required JSON format."""
        output = {
            "Question 9": {
                "group Matches": grouped_sentences
            }
        }
        print(json.dumps(output, indent=4))
