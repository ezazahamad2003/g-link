import requests
import networkx as nx
import matplotlib.pyplot as plt
import json
import time
from tqdm import tqdm
import os

class WikiKnowledgeGraph:
    def __init__(self):
        """Initialize the knowledge graph with empty structures"""
        self.graph = nx.DiGraph()
        self.visited_pages = set()
        self.physics_pages = set()
        self.biology_pages = set()
        self.api_url = "https://en.wikipedia.org/w/api.php"
        self.checkpoint_dir = "checkpoints"
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
    def get_links(self, page_title):
        """Get all links from a Wikipedia page
        
        Args:
            page_title (str): Title of the Wikipedia page
            
        Returns:
            list: List of page titles linked from the given page
        """
        params = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "links",
            "pllimit": "500"
        }
        
        links = []
        continue_params = True
        
        while continue_params:
            response = requests.get(self.api_url, params=params)
            data = response.json()
            
            # Extract page id
            page_id = list(data["query"]["pages"].keys())[0]
            
            # Check if the page exists and has links
            if page_id != "-1" and "links" in data["query"]["pages"][page_id]:
                for link in data["query"]["pages"][page_id]["links"]:
                    # Skip links to special pages (those with colons in the title)
                    if ":" not in link["title"]:
                        links.append(link["title"])
            
            # Check if there are more links to fetch (pagination)
            if "continue" in data and "plcontinue" in data["continue"]:
                params["plcontinue"] = data["continue"]["plcontinue"]
            else:
                continue_params = False
                
            # Be nice to Wikipedia API by adding a delay
            time.sleep(1)
            
        return links
    
    def is_physics_page(self, page_title):
        """Check if a page is related to physics using Wikipedia's category system
        
        Args:
            page_title (str): Title of the Wikipedia page
            
        Returns:
            bool: True if the page is related to physics, False otherwise
        """
        # Check direct categories
        params = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "categories",
            "cllimit": "500"
        }
        
        response = requests.get(self.api_url, params=params)
        data = response.json()
        
        # Extract page id
        page_id = list(data["query"]["pages"].keys())[0]
        
        # Check if the page exists and has categories
        if page_id != "-1" and "categories" in data["query"]["pages"][page_id]:
            for category in data["query"]["pages"][page_id]["categories"]:
                category_title = category["title"].lower()
                # Check for physics-related categories
                if any(term in category_title for term in ["physics", "physical", "physicist"]):
                    return True
        
        # Check if the page is in physics-related subcategories
        physics_subcategories = [
            "Category:Physics",
            "Category:Classical mechanics",
            "Category:Quantum mechanics",
            "Category:Thermodynamics",
            "Category:Electromagnetism",
            "Category:Nuclear physics",
            "Category:Particle physics",
            "Category:Astrophysics",
            "Category:Theoretical physics",
            "Category:Applied physics",
            "Category:Physical quantities"
        ]
        
        for category in physics_subcategories:
            subcategory_params = {
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmtitle": category,
                "cmprop": "title",
                "cmtype": "page",
                "cmlimit": "500"
            }
            
            response = requests.get(self.api_url, params=subcategory_params)
            data = response.json()
            
            if "query" in data and "categorymembers" in data["query"]:
                for member in data["query"]["categorymembers"]:
                    if member["title"] == page_title:
                        return True
            
            # Be nice to Wikipedia API
            time.sleep(0.5)
        
        # As a fallback, check page content for physics-related terms
        content_params = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True
        }
        
        response = requests.get(self.api_url, params=content_params)
        data = response.json()
        
        # Check if the page has an extract
        if page_id != "-1" and "extract" in data["query"]["pages"][page_id]:
            extract = data["query"]["pages"][page_id]["extract"].lower()
            physics_terms = [
                "physics", "physical", "quantum", "relativity", "mechanics", 
                "particle", "energy", "force", "mass", "electromagnetic", 
                "nuclear", "thermodynamic", "kinetic", "potential energy"
            ]
            # Check if multiple physics terms appear in the extract
            term_count = sum(1 for term in physics_terms if term in extract)
            if term_count >= 3:  # Require at least 3 terms for content-based classification
                return True
        
        return False
    
    def is_biology_page(self, page_title):
        """Check if a page is related to biology using Wikipedia's category system
        
        Args:
            page_title (str): Title of the Wikipedia page
            
        Returns:
            bool: True if the page is related to biology, False otherwise
        """
        # Check direct categories
        params = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "categories",
            "cllimit": "500"
        }
        
        response = requests.get(self.api_url, params=params)
        data = response.json()
        
        # Extract page id
        page_id = list(data["query"]["pages"].keys())[0]
        
        # Check if the page exists and has categories
        if page_id != "-1" and "categories" in data["query"]["pages"][page_id]:
            for category in data["query"]["pages"][page_id]["categories"]:
                category_title = category["title"].lower()
                # Check for biology-related categories
                if any(term in category_title for term in ["biology", "biological", "biologist"]):
                    return True
        
        # Check if the page is in biology-related subcategories
        biology_subcategories = [
            "Category:Biology",
            "Category:Molecular biology",
            "Category:Cell biology",
            "Category:Genetics",
            "Category:Biochemistry",
            "Category:Ecology",
            "Category:Evolution",
            "Category:Physiology",
            "Category:Microbiology",
            "Category:Neuroscience",
            "Category:Organisms"
        ]
        
        for category in biology_subcategories:
            subcategory_params = {
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmtitle": category,
                "cmprop": "title",
                "cmtype": "page",
                "cmlimit": "500"
            }
            
            response = requests.get(self.api_url, params=subcategory_params)
            data = response.json()
            
            if "query" in data and "categorymembers" in data["query"]:
                for member in data["query"]["categorymembers"]:
                    if member["title"] == page_title:
                        return True
            
            # Be nice to Wikipedia API
            time.sleep(0.5)
        
        # As a fallback, check page content for biology-related terms
        content_params = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True
        }
        
        response = requests.get(self.api_url, params=content_params)
        data = response.json()
        
        # Check if the page has an extract
        if page_id != "-1" and "extract" in data["query"]["pages"][page_id]:
            extract = data["query"]["pages"][page_id]["extract"].lower()
            biology_terms = [
                "biology", "biological", "cell", "dna", "rna", "protein", 
                "gene", "organism", "species", "evolution", "ecology", 
                "molecular", "tissue", "enzyme", "chromosome"
            ]
            # Check if multiple biology terms appear in the extract
            term_count = sum(1 for term in biology_terms if term in extract)
            if term_count >= 3:  # Require at least 3 terms for content-based classification
                return True
        
        return False
    
    def build_graph(self, physics_seed_pages, biology_seed_pages, max_pages=None, checkpoint_interval=50):
        """Build a knowledge graph starting from seed pages using BFS
        
        Args:
            physics_seed_pages (list): List of physics seed page titles
            biology_seed_pages (list): List of biology seed page titles
            max_pages (int, optional): Maximum number of pages to process. If None, continue indefinitely.
            checkpoint_interval (int, optional): Save intermediate results every this many pages
        """
        # Add seed pages to the graph
        for page in physics_seed_pages:
            self.graph.add_node(page, domain="physics")
            self.physics_pages.add(page)
        
        for page in biology_seed_pages:
            self.graph.add_node(page, domain="biology")
            self.biology_pages.add(page)
        
        # Queue for BFS - store (page_title, depth) pairs
        queue = [(page, 0) for page in physics_seed_pages + biology_seed_pages]
        
        # Track pages already in queue to avoid duplicates
        in_queue = set(physics_seed_pages + biology_seed_pages)
        
        # Process pages using BFS
        with tqdm() as pbar:  # Progress bar without total (indefinite)
            while queue:
                # Check if we've reached the maximum number of pages
                if max_pages and len(self.visited_pages) >= max_pages:
                    print(f"Reached maximum number of pages ({max_pages})")
                    break
                
                # Get the next page from the queue (BFS order)
                current_page, depth = queue.pop(0)
                in_queue.remove(current_page)
                
                # Skip if already visited
                if current_page in self.visited_pages:
                    continue
                
                # Mark as visited and update progress
                self.visited_pages.add(current_page)
                pbar.update(1)
                pbar.set_description(f"Processing: {current_page}")
                
                # Get links from the current page
                try:
                    links = self.get_links(current_page)
                    
                    for link in links:
                        # Skip if already visited or in queue
                        if link in self.visited_pages or link in in_queue:
                            continue
                        
                        # Determine domain of the linked page
                        if link in self.physics_pages:
                            domain = "physics"
                        elif link in self.biology_pages:
                            domain = "biology"
                        elif self.is_physics_page(link):
                            domain = "physics"
                            self.physics_pages.add(link)
                        elif self.is_biology_page(link):
                            domain = "biology"
                            self.biology_pages.add(link)
                        else:
                            # Skip pages that are neither physics nor biology
                            continue
                        
                        # Add node and edge to the graph
                        self.graph.add_node(link, domain=domain)
                        self.graph.add_edge(current_page, link)
                        
                        # Add to queue for further exploration
                        queue.append((link, depth + 1))
                        in_queue.add(link)
                    
                    # Save checkpoint periodically
                    if len(self.visited_pages) % checkpoint_interval == 0:
                        checkpoint_file = os.path.join(self.checkpoint_dir, f"checkpoint_{len(self.visited_pages)}")
                        self.save_checkpoint(checkpoint_file)
                        self.save_graph(
                            os.path.join(self.checkpoint_dir, f"nodes_{len(self.visited_pages)}.csv"),
                            os.path.join(self.checkpoint_dir, f"edges_{len(self.visited_pages)}.csv")
                        )
                        print(f"\nCheckpoint saved with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
                        print(f"Physics pages: {len(self.physics_pages)}, Biology pages: {len(self.biology_pages)}")
                
                except Exception as e:
                    print(f"\nError processing page {current_page}: {e}")
    
    def save_checkpoint(self, filename):
        """Save the current state of the graph to a checkpoint file
        
        Args:
            filename (str): Base filename for the checkpoint
        """
        checkpoint = {
            "visited_pages": list(self.visited_pages),
            "physics_pages": list(self.physics_pages),
            "biology_pages": list(self.biology_pages)
        }
        
        with open(f"{filename}.json", "w", encoding="utf-8") as f:
            json.dump(checkpoint, f)
    
    def load_checkpoint(self, filename):
        """Load a checkpoint to resume graph building
        
        Args:
            filename (str): Base filename for the checkpoint
        """
        with open(f"{filename}.json", "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        
        self.visited_pages = set(checkpoint["visited_pages"])
        self.physics_pages = set(checkpoint["physics_pages"])
        self.biology_pages = set(checkpoint["biology_pages"])
        
        # Rebuild the graph from the checkpoint data
        self.graph = nx.DiGraph()
        
        # Load nodes
        nodes_file = filename.replace("checkpoint", "nodes") + ".csv"
        with open(nodes_file, "r", encoding="utf-8") as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    node_id = parts[0].strip('"')
                    domain = parts[2].strip('"')
                    self.graph.add_node(node_id, domain=domain)
        
        # Load edges
        edges_file = filename.replace("checkpoint", "edges") + ".csv"
        with open(edges_file, "r", encoding="utf-8") as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    source = parts[0].strip('"')
                    target = parts[1].strip('"')
                    self.graph.add_edge(source, target)
    
    def save_graph(self, nodes_file="nodes.csv", edges_file="edges.csv"):
        """Save the graph to CSV files
        
        Args:
            nodes_file (str): Filename for nodes CSV
            edges_file (str): Filename for edges CSV
        """
        # Save nodes
        with open(nodes_file, "w", encoding="utf-8") as f:
            f.write("id,label,domain\n")
            for node in self.graph.nodes():
                domain = self.graph.nodes[node].get("domain", "unknown")
                f.write(f'"{node}","{node}","{domain}"\n')
        
        # Save edges
        with open(edges_file, "w", encoding="utf-8") as f:
            f.write("source,target\n")
            for edge in self.graph.edges():
                f.write(f'"{edge[0]}","{edge[1]}"\n')
    
    def visualize_graph(self, output_file="knowledge_graph.png", max_nodes=500):
        """Visualize the graph
        
        Args:
            output_file (str): Filename for the output image
            max_nodes (int): Maximum number of nodes to include in visualization
        """
        # If graph is too large, take a sample
        if len(self.graph.nodes()) > max_nodes:
            print(f"Graph is too large ({len(self.graph.nodes())} nodes). Visualizing a sample of {max_nodes} nodes.")
            # Take a connected subgraph centered on important nodes
            important_nodes = list(self.physics_pages)[:10] + list(self.biology_pages)[:10]
            subgraph = nx.ego_graph(self.graph, important_nodes[0], radius=3)
            for node in important_nodes[1:]:
                if len(subgraph) < max_nodes and node in self.graph:
                    subgraph = nx.compose(subgraph, nx.ego_graph(self.graph, node, radius=2))
            graph_to_draw = subgraph
        else:
            graph_to_draw = self.graph
        
        plt.figure(figsize=(16, 16))
        
        # Define node colors based on domain
        node_colors = []
        for node in graph_to_draw.nodes():
            if graph_to_draw.nodes[node].get("domain") == "physics":
                node_colors.append("blue")
            elif graph_to_draw.nodes[node].get("domain") == "biology":
                node_colors.append("green")
            else:
                node_colors.append("gray")
        
        # Draw the graph
        pos = nx.spring_layout(graph_to_draw, seed=42)
        nx.draw_networkx_nodes(graph_to_draw, pos, node_size=100, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(graph_to_draw, pos, width=0.5, alpha=0.5, arrows=True)
        
        # Only draw labels for nodes with degree > 1 to reduce clutter
        labels = {node: node for node in graph_to_draw.nodes() if graph_to_draw.degree(node) > 1}
        nx.draw_networkx_labels(graph_to_draw, pos, labels=labels, font_size=8)
        
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        print(f"Graph visualization saved to {output_file}")
    
    def analyze_graph(self):
        """Analyze the graph and print statistics"""
        print("\n=== Knowledge Graph Analysis ===")
        print(f"Total nodes: {len(self.graph.nodes())}")
        print(f"Total edges: {len(self.graph.edges())}")
        print(f"Physics pages: {len(self.physics_pages)}")
        print(f"Biology pages: {len(self.biology_pages)}")
        
        # Calculate degree statistics
        degrees = [d for n, d in self.graph.degree()]
        if degrees:
            print(f"Average degree: {sum(degrees)/len(degrees):.2f}")
            print(f"Maximum degree: {max(degrees)}")
        
        # Find most connected nodes
        most_connected = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)[:10]
        print("\nMost connected pages:")
        for page, degree in most_connected:
            domain = self.graph.nodes[page].get("domain", "unknown")
            print(f"  - {page} ({domain}): {degree} connections")
        
        # Find bridge nodes (nodes that connect physics and biology)
        bridge_nodes = []
        for node in self.graph.nodes():
            physics_neighbors = 0
            biology_neighbors = 0
            
            for neighbor in self.graph.neighbors(node):
                if self.graph.nodes[neighbor].get("domain") == "physics":
                    physics_neighbors += 1
                elif self.graph.nodes[neighbor].get("domain") == "biology":
                    biology_neighbors += 1
            
            if physics_neighbors > 0 and biology_neighbors > 0:
                bridge_nodes.append((node, physics_neighbors, biology_neighbors))
        
        # Sort bridge nodes by total connections
        bridge_nodes.sort(key=lambda x: x[1] + x[2], reverse=True)
        
        print("\nBridge pages (connecting physics and biology):")
        for node, physics_count, biology_count in bridge_nodes[:10]:
            domain = self.graph.nodes[node].get("domain", "unknown")
            print(f"  - {node} ({domain}): {physics_count} physics connections, {biology_count} biology connections")


if __name__ == "__main__":
    # Create knowledge graph
    kg = WikiKnowledgeGraph()
    
    # Define seed pages - comprehensive starting points
    physics_seeds = [
        "Physics", "Quantum mechanics", "Relativity", "Thermodynamics",
        "Classical mechanics", "Electromagnetism", "Nuclear physics",
        "Particle physics", "Astrophysics", "Condensed matter physics"
    ]
    
    biology_seeds = [
        "Biology", "Cell (biology)", "DNA", "Evolution",
        "Genetics", "Ecology", "Molecular biology", "Biochemistry",
        "Physiology", "Microbiology", "Neuroscience"
    ]
    
    print("=== Wikipedia Knowledge Graph Builder ===")
    print("This script will build a knowledge graph of physics and biology pages from Wikipedia.")
    print(f"Starting with {len(physics_seeds)} physics seed pages and {len(biology_seeds)} biology seed pages.")
    
    # Ask user for parameters
    try:
        max_pages = input("Enter maximum number of pages to process (leave empty for unlimited): ")
        max_pages = int(max_pages) if max_pages.strip() else None
        
        checkpoint_interval = input("Save checkpoint every how many pages? (default: 50): ")
        checkpoint_interval = int(checkpoint_interval) if checkpoint_interval.strip() else 50
        
        print("\nBuilding knowledge graph...")
        kg.build_graph(physics_seeds, biology_seeds, max_pages=max_pages, checkpoint_interval=checkpoint_interval)
        
        print("\nGraph building complete!")
        kg.analyze_graph()
        
        # Save the final graph
        kg.save_graph("final_nodes.csv", "final_edges.csv")
        print("Graph saved to final_nodes.csv and final_edges.csv")
        
        # Visualize the graph
        kg.visualize_graph("final_knowledge_graph.png")
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Saving current progress...")
        kg.save_graph("interrupted_nodes.csv", "interrupted_edges.csv")
        print("Partial graph saved to interrupted_nodes.csv and interrupted_edges.csv")
        kg.analyze_graph()