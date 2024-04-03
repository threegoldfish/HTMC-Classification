import networkx as nx
import matplotlib.pyplot as plt


def create_label_graph(hierarchy_path, class_num):
    G = nx.DiGraph()
    for i in range(class_num):
        G.add_node(i)

    with open(hierarchy_path) as file:
        for line in file:
            line = line.replace('\n', "")
            from_index = int(line.split('\t')[0])
            to_index = int(line.split('\t')[1])
            G.add_edge(from_index, to_index)

    return G

def get_first_level(G):
    return [node for node in G.nodes() if G.in_degree(node) == 0]

def get_second_level(G, first_level):
    nodes = []
    for target_node in first_level:
        nodes.extend([node for node in G.nodes() if G.has_edge(target_node, node)])
    return nodes

def get_third_level(G, second_level):
    nodes = []
    for target_node in second_level:
        nodes.extend([node for node in G.nodes() if G.has_edge(target_node, node)])
    return nodes

def get_labels(label_path):
    labels = []
    with open(label_path) as file:
        for line in file:
            line = line.replace("\n", "")
            labels.append(line.split('\t')[1])
    return labels

def plot_graph(G):
    pos = nx.spring_layout(G, seed=42)  # Position nodes using a spring layout
    nx.draw(G, pos, with_labels=True, node_size=800, node_color="lightblue", font_size=12, font_weight="bold", arrows=True)
    plt.title("Directed Graph")
    plt.show()

    # Get information about the directed graph
    print("Nodes:", G.nodes())
    print("Edges:", G.edges())
    print("In-Degree of C:", G.in_degree(0))
    print("Out-Degree of B:", G.out_degree(3))

#helper function such that given a list of nodes,
# returns the child of all nodes in a nested list based on level
def get_childs(G, nodes, level):
    childs = []
    for node in nodes:
        childs.extend([n for n in G.nodes() if G.has_edge(node, n)])
    return childs


if __name__ == '__main__':
    G = create_label_graph("./TaxoClass-dataset/Amazon-531/label_hierarchy_small.txt", 21)
    """first_level = get_first_level(G)
    second_level = get_second_level(G, firstevel)
    third_level = get_third_level(G, second_level)
    
    print(len(first_level))
    #print(first_level)
    print(len(second_level))
    #print(second_level)
    print(len(third_level))
    #print(third_level)"""