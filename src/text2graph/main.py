from modules import PostgreSQLConnection


def main():
    print("Hello from text2graph!")
    pg = PostgreSQLConnection(
        min_connections=1,
        max_connections=5,
    )


if __name__ == "__main__":
    main()


X = """
You are a top student and very good at making your notes to understand a topic. Make a list of topics and relations between the topics used in this text, that can be used as part of a knowlefge graph. Replace references to a preceeding sentence with the topic beeing referred to. Produce a list in the JSON format:
Standard Retrieval-Augmented Generation (RAG) relies on chunk-based retrieval, whereas GraphRAG advances this approach by graph-based knowledge representation. However, existing graph-based RAG approaches are constrained by binary relations, as each edge in an ordinary graph connects only two entities, limiting their ability to represent the n-ary relations (n >= 2) in real-world knowledge. In this work, we propose HyperGraphRAG, a novel hypergraph-based RAG method that represents n-ary relational facts via hyperedges, and consists of knowledge hypergraph construction, retrieval, and generation. Experiments across medicine, agriculture, computer science, and law demonstrate that HyperGraphRAG outperforms both standard RAG and previous graph-based RAG methods in answer accuracy, retrieval efficiency, and generation quality. Our data and code are publicly available at this https URL.
"""
