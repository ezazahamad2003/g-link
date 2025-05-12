# g-link

## About The Project

g-link is a research project focused on generating novel insights by integrating knowledge graphs from disparate scientific domains. This project specifically explores the intersection of **physics** and **biology**, aiming to uncover new knowledge by forming connections between their respective knowledge graphs.

## Methodology

The core methodology involves:
1.  **Knowledge Graph Construction:** Developing or utilizing existing knowledge graphs for the domains of physics and biology.
2.  **Inter-Domain Linking with GNNs:** Employing Graph Neural Networks (GNNs) to identify and create new nodes or edges that bridge these two distinct knowledge graphs.
3.  **Knowledge Discovery via Reasoning:** Utilizing a reasoning model to analyze the augmented, interconnected graph structure to discover emergent knowledge and novel relationships.

## Goals

-   To develop a framework for integrating knowledge graphs from different domains.
-   To leverage GNNs for identifying potential cross-domain links.
-   To apply reasoning models to extract new, previously unrecognised knowledge from the combined graph.
-   To demonstrate the potential of this approach by finding new connections between physics and biology.

## Wikipedia Knowledge Graph Builder

The project includes a Wikipedia knowledge graph builder (`wiki_knowledge_graph.py`) that constructs domain-specific knowledge graphs for physics and biology by crawling Wikipedia pages.

### Features

- Breadth-first search (BFS) exploration of Wikipedia pages
- Domain classification using Wikipedia categories and content analysis
- Directed graph representation of page relationships
- Visualization capabilities for the resulting knowledge graph
- Checkpoint system to resume interrupted crawls

### Limitations

- **Completeness**: Due to Wikipedia's vast size, the script cannot guarantee capturing the complete graph of all physics and biology pages. The exploration is limited by:
  - Wikipedia's rate limits (no more than 200 requests per minute)
  - The effectiveness of the domain classification methods
  - The connectivity of the Wikipedia link structure