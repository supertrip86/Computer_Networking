import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import fnss
import warnings


plt.style.use('seaborn-whitegrid')
warnings.filterwarnings("ignore", category=FutureWarning)


def generate_erdos_renyi_random_graph(k, p, use_nx = False):
    """
    Generate an Erdős-Rényi random graph with k nodes and edge probability p.

    Parameters:
    -----------
    k : int
        The number of nodes in the graph.
    p : float
        The probability of having an edge between any two nodes.
    use_nx : bool, optional
        Whether to use NetworkX to generate the graph instead of NumPy. Default is False.

    Returns:
    --------
    A : ndarray or NetworkX Graph
        An adjacency matrix or Graph representing the generated random graph.

    Examples:
    ---------
    >>> A = generate_erdos_renyi_random_graph(10, 0.5)
    >>> A.shape == (10, 10)
    True

    >>> G = generate_erdos_renyi_random_graph(20, 0.3, use_nx=True)
    >>> len(G.nodes) == 20 and len(G.edges) > 0
    True
    """

    
    if use_nx:
        return nx.erdos_renyi_graph(k, p)

    # Create an empty adjacency matrix
    adj_matrix = np.zeros((k, k))
    
    # Loop through each pair of nodes
    for i in range(k):
        for j in range(i+1, k):
            # Generate a random number between 0 and 1
            rand = np.random.uniform(0, 1)
            # If the random number is less than the probability, add an edge between the nodes
            if rand < p:
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1

    return adj_matrix


def generate_r_regular_graph(k, r, use_nx = True):
    
    """
    Generate a random r-regular graph with k nodes using NumPy.

    Parameters:
    -----------
    k : int
        The number of nodes in the graph.
    r : int
        The degree of each node in the graph.

    Returns:
    --------
    A : ndarray
        An adjacency matrix representing the generated graph.

    Raises:
    -------
    ValueError:
        If r is not even or r < 0, or if r*k is odd or r*k > k*(k-1).

    Examples:
    ---------
    >>> A = generate_r_regular_graph(10, 2)
    >>> A.shape == (10, 10)
    True
    """
    
    if r >= k or k % 2 != 0 or r % 2 != 0:
        raise ValueError("Invalid k or r value for r-regular graph generation.")
    
    if use_nx:
        return nx.random_regular_graph(r, k, seed=None)

    # Create an empty adjacency matrix
    A = np.zeros((k, k), dtype=int)

    # Create a list of all possible edges
    edges = []
    for i in range(k):
        for j in range(i+1, k):
            edges.append((i, j))

    # Shuffle the list of edges
    np.random.shuffle(edges)

    # Loop over the edges and add them to the graph
    degree = np.zeros(k, dtype=int)
    for edge in edges:
        if degree[edge[0]] < r and degree[edge[1]] < r:
            A[edge[0], edge[1]] = 1
            A[edge[1], edge[0]] = 1
            degree[edge[0]] += 1
            degree[edge[1]] += 1
        if np.all(degree >= r):
            break

    return A


def is_connected(adj_matrix, type):

    """
    Check the connectivity of a graph represented by its adjacency matrix using different methods.

    Parameters:
    -----------
    adj_matrix : ndarray
        An adjacency matrix representing the graph.
    type : str
        The type of connectivity checking method to use. Can be "Irreducibility", "Laplacian", or "BFS".

    Returns:
    --------
    str
        A string indicating whether the graph is connected or not.

    Raises:
    -------
    ValueError:
        If type is not one of "Irreducibility", "Laplacian", or "BFS".

    Examples:
    ---------
    >>> A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> is_connected(A, "Irreducibility")
    'is connected'

    >>> B = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> is_connected(B, "Laplacian")
    'is NOT connected'
    """

    result = False

    if type == "Irreducibility":
        result = is_connected_irreducibility(adj_matrix)
    elif type == "Laplacian":
        result = is_connected_laplacian(adj_matrix)
    elif type == "BFS":
        result = is_connected_bfs(adj_matrix)
    else:
        raise ValueError("Select a valid connectivity checking method.")

    return "is connected" if result else "is NOT connected"


def is_connected_irreducibility(adj_matrix):
    """
    Checks if a graph is connected or not using the irreducibility of its adjacency matrix.

    Parameters:
    -----------
    adj_matrix : np.ndarray
        The adjacency matrix of the graph.
        
    Returns:
    --------
    bool
        True if the graph is connected, False otherwise.
    
    Raises:
    -------
    ValueError:
        If the adjacency matrix is not square.

    Examples:
    ---------
    >>> adj_matrix = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    >>> is_connected_irreducibility(adj_matrix)
    True
    """
    
    if isinstance(adj_matrix, nx.Graph):
        adj_matrix = nx.to_numpy_array(adj_matrix)

    k = adj_matrix.shape[0]
    
    if k != adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
        
    result = np.identity(k)

    for d in range(1, k):
        result += np.linalg.matrix_power(adj_matrix, d)
        
    return result.all() > 0


def is_connected_laplacian(adj_matrix):
    """
    Checks if a graph is connected or not using the eigenvalues of its Laplacian matrix.

    Parameters:
    -----------
    adj_matrix : numpy.ndarray
        The adjacency matrix of the graph.

    Returns:
    --------
    bool:
        True if the graph is connected, False otherwise.

    Raises:
    -------
    ValueError:
        If the matrix is not square.

    Examples:
    ---------
    >>> A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> D = np.diag(np.sum(A, axis=1))
    >>> L = D - A
    >>> is_connected_laplacian(L)
    True
    """

    if isinstance(adj_matrix, nx.Graph):
        adj_matrix = nx.to_numpy_array(adj_matrix)

    # Check if the matrix is square
    n = adj_matrix.shape[0]
    
    if n != adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
        
    # Compute the Laplacian matrix
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    laplacian_matrix = degree_matrix - adj_matrix
    
    # Compute the eigenvalues of the Laplacian matrix
    eigenvalues, _ = np.linalg.eig(laplacian_matrix)
    
    # Check if the second smallest eigenvalue is positive
    sorted_eigenvalues = np.sort(eigenvalues)
    
    return sorted_eigenvalues[1] > 0


def is_connected_bfs(adj_matrix):
    
    """
    Checks if a graph is connected or not using the breadth-first search algorithm.

    Parameters:
    -----------
    adj_matrix : np.ndarray
        The adjacency matrix of the graph.

    Returns:
    --------
    bool
        True if the graph is connected, False otherwise.

    Raises:
    -------
    ValueError:
        If the adjacency matrix is not square.

    Examples:
    ---------
    >>> adj_matrix = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    >>> is_connected_bfs(adj_matrix)
    True
    >>> adj_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> is_connected_bfs(adj_matrix)
    False
    """

    if isinstance(adj_matrix, nx.Graph):
        adj_matrix = nx.to_numpy_array(adj_matrix)

    k = adj_matrix.shape[0]
    
    if k != adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
        
    # Choose an arbitrary starting node
    start_node = 0
    
    # Initialize a queue to store the nodes to be visited
    queue = [start_node]
    
    # Initialize a list to store the visited nodes
    visited = [start_node]
    
    # Traverse the graph using BFS
    while queue:
        # Dequeue a node from the queue
        node = queue.pop(0)
        # Get the neighbors of the node
        neighbors = np.nonzero(adj_matrix[node])[0]
        # Add unvisited neighbors to the queue
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
                
    # Check if all nodes have been visited
    return len(visited) == k


def run_experiment(K_range, p, generate_graph_fn):
    """
    Run experiments to compare the runtimes of the three algorithms for checking the connectivity of a graph:
    - is_connected_irreducibility
    - is_connected_laplacian
    - is_connected_bfs

    The experiments generate random graphs using a specified function, and vary the number of nodes from K_range.
    For each graph, the three algorithms are timed and their runtimes are recorded.

    Parameters:
    -----------
    K_range : list of int
        The range of the number of nodes in the graphs to be generated.
    p : float
        The edge probability for each pair of nodes in the generated graphs.
    generate_graph_fn : function
        A function that generates a random graph with the signature: generate_graph_fn(k, p),
        where k is the number of nodes in the graph, and p is the edge probability.

    Returns:
    --------
    runtimes_irreducibility : list of float
        The runtimes of is_connected_irreducibility for each generated graph.
    runtimes_laplacian : list of float
        The runtimes of is_connected_laplacian for each generated graph.
    runtimes_bfs : list of float
        The runtimes of is_connected_bfs for each generated graph.


    Examples:
    ---------
    >>> K_range = [50, 100, 150]
    >>> p = 0.5
    >>> G_fn = generate_erdos_renyi_nx
    >>> runtimes_irr, runtimes_lap, runtimes_bfs = run_experiment(K_range, p, G_fn)
    """

    runtimes_irreducibility = []
    runtimes_laplacian = []
    runtimes_bfs = []
    
    for K in K_range:
        # Generate random graph using the specified function
        G = generate_graph_fn(K, p)
        
        # Time is_connected_irreducibility_nx
        start_time = time.time()
        is_connected_irreducibility(G)
        end_time = time.time()
        runtime_irreducibility = end_time - start_time
        runtimes_irreducibility.append(runtime_irreducibility)
        
        # Time is_connected_laplacian_nx
        start_time = time.time()
        is_connected_laplacian(G)
        end_time = time.time()
        runtime_laplacian = end_time - start_time
        runtimes_laplacian.append(runtime_laplacian)
        
        # Time is_connected_bfs_nx
        start_time = time.time()
        is_connected_bfs(G)
        end_time = time.time()
        runtime_bfs = end_time - start_time
        runtimes_bfs.append(runtime_bfs)
    
    return runtimes_irreducibility, runtimes_laplacian, runtimes_bfs


def plot_runtime_complexity(K_range, p, generate_graph_fn, title_graph, title_patch):

    """
    Plot the runtime complexity of three different connectivity checking methods as a function of the number of nodes in a graph.

    Parameters:
    -----------
    K_range : list
        A list of integers representing the number of nodes in the graph for which to measure the runtime complexity.
    p : float
        The probability of generating an edge between any two nodes in the graph (for an Erdos-Renyi graph).
    generate_graph_fn : function
        A function that generates a graph given the number of nodes and the probability of generating an edge.
    title_graph : str
        A string indicating the type of graph being generated (e.g., "Erdos-Renyi").
    title_patch : str
        A string indicating the parameter used to generate the graph (e.g., "p" for an Erdos-Renyi graph).

    Returns:
    --------
    None

    Examples:
    ---------
    >>> K_range = [10, 20, 30]
    >>> p = 0.5
    >>> plot_runtime_complexity(K_range, p, generate_erdos_renyi_random_graph, "Erdos-Renyi", "p")
    """

    runtimes_irreducibility_erdos_renyi, runtimes_laplacian_erdos_renyi, runtimes_bfs_erdos_renyi = run_experiment(K_range, p, generate_graph_fn)
    # Plot the results
    fig, ax = plt.subplots(dpi=100)

    # Plot the results with custom colors and line styles
    ax.plot(K_range, runtimes_irreducibility_erdos_renyi, color='#1f77b4', linewidth=2, linestyle='-', label='Irreducibility')
    ax.plot(K_range, runtimes_laplacian_erdos_renyi, color='#ff7f0e', linewidth=2, linestyle='-', label='Laplacian')
    ax.plot(K_range, runtimes_bfs_erdos_renyi, color='#2ca02c', linewidth=2, linestyle='-', label='BFS')

    # Set title and axis labels with larger font sizes
    ax.set_title(f'{title_graph} graph with {title_patch} = {p}', fontsize = 16)
    ax.set_xlabel('Number of Nodes ($K$)', fontsize=14)
    ax.set_ylabel('Time Taken (seconds)', fontsize=14)

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major')

    # Add legend with larger font size and transparent background
    ax.legend(fontsize=12, fancybox=True, facecolor='none')

    # Add grid lines and tighten layout
    ax.grid(alpha=0.3)
    fig.tight_layout()

    # Display the plot
    return plt.show()


def estimate_p_connected_er(K, p_range, num_trials):
    """
    Estimates the probability that an Erdos-Renyi random graph with K nodes is connected
    for different edge probabilities using Monte Carlo simulations.

    Parameters:
    -----------
    K : int
        The number of nodes in the graph.
    p_range : numpy.ndarray
        An array of edge probabilities to test.
    num_trials : int
        The number of trials to run for each edge probability.

    Returns:
    --------
    p_connected : numpy.ndarray
        An array of estimated probabilities that the graph is connected for each edge probability.

    """

    p_connected = np.zeros(len(p_range))

    for i, p in enumerate(p_range):
        num_connected = 0

        for j in range(num_trials):
            G = generate_erdos_renyi_random_graph(K, p)

            if is_connected_bfs(G):
                num_connected += 1

        p_connected[i] = num_connected / num_trials

    return p_connected


def estimate_p_connected_r_regular(r, K_range, num_trials):
    """
    Estimates the probability that an R-Regular random graph with K nodes is connected
    for different edge probabilities using Monte Carlo simulations.

    Parameters:
    -----------
    r : int
        The number of degrees.
    K_range : numpy.ndarray
        An array containing the amounts of nodes to be generated.
    num_trials : int
        The number of trials to run for each random graph with number of nodes k.

    Returns:
    --------
    p_connected : numpy.ndarray
        An array of estimated probabilities that the graph is connected for each edge probability.

    """

    p_connected = np.zeros(len(K_range))

    for k in enumerate(K_range):
        num_connected = 0

        for j in range(num_trials):
            G = generate_r_regular_graph(k[1], r)

            if is_connected_bfs(G):
                num_connected += 1

        p_connected[k[0]] = num_connected / num_trials

    return p_connected


def plot_estimated_p_connected_er(K, p_range, num_trials):

    """
    Plot the estimated probability of connectivity for a p-ER random graph with K nodes and a range of edge probabilities.

    Parameters:
    -----------
    K : int
        The number of nodes in the graph.
    p_range : array-like
        A range of edge probabilities to test.
    num_trials : int
        The number of trials to run for each edge probability.

    Returns:
    --------
    A plot showing the estimated probability of connectivity for each edge probability in the given range, as well as a dashed line showing the value of p at which the estimated probability of connectivity reaches 1.

    Examples:
    ---------
    >>> plot_estimated_p_connected_er(20, np.linspace(0, 0.2, 11), 100)
    """

    p_connected = estimate_p_connected_er(K, p_range, num_trials)
    connected = np.interp(.9999999999999, p_connected, p_range)

    # Plot the results
    fig, ax = plt.subplots(dpi=100)

    # Plot the results with custom colors and line styles
    ax.plot(p_range, p_connected, color='#1f77b4', linewidth=2, linestyle='-')

    ax.vlines(connected, ymin = 0, ymax = 1, color = '#ff7f0e', linewidth = 1, linestyle='--')
    # Set title and axis labels with larger font sizes
    ax.set_title(f'Probability of Connectivity for $p-ER$ random graph with $K$ = {K}', fontsize = 16)
    ax.set_xlabel('Edge Probability ($p$)', fontsize=14)
    ax.set_ylabel('Probability of Connectivity', fontsize=14)
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major')

    # Add grid lines and tighten layout
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plt.xlim([0, 0.12])

    ticks = list(plt.xticks()[0])
    extraticks=[connected]
    plt.xticks(ticks + extraticks)

    # Display the plot
    return plt.show()


def plot_estimated_p_connected_r_regular(r1, r2, K_range, num_trials):
    """
    Plots the estimated probability of connectivity for R-Regular random graphs with parameters r1 and r2,
    over a range of number of nodes (K) using Monte Carlo simulations.

    Parameters
    ----------
    r1 : int
        The degree of each node for the first R-Regular graph.
    r2 : int
        The degree of each node for the second R-Regular graph.
    K_range : numpy.ndarray
        An array of integers representing the range of number of nodes to consider.
    num_trials : int
        The number of Monte Carlo trials to perform for each value of K.

    Returns
    -------
    Displays the plot of estimated probability of connectivity for the two R-Regular graphs.

    """
    p_connected_1 = estimate_p_connected_r_regular(r1, K_range, num_trials)
    p_connected_2 = estimate_p_connected_r_regular(r2, K_range, num_trials)

    # Plot the results
    fig, ax = plt.subplots(dpi=100)

    # Plot the results with custom colors and line styles
    ax.plot(K_range, p_connected_1, color='#1f77b4', linewidth=2, linestyle='-', label=f'r = {r1}')
    ax.plot(K_range, p_connected_2, color='#ff7f0e', linewidth=2, linestyle='-', label=f'r = {r2}')

    # Set title and axis labels with larger font sizes
    ax.set_title(f'Probability of Connectivity for R-Regular random graphs', fontsize = 16)
    ax.set_xlabel('Number of nodes ($K$)', fontsize=14)
    ax.set_ylabel('Probability of Connectivity', fontsize=14)
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major')

    # Add legend with larger font size and transparent background
    ax.legend(fontsize=12, fancybox=True, facecolor='none')

    # Add grid lines and tighten layout
    ax.grid(alpha=0.3)
    fig.tight_layout()

    # Display the plot
    return plt.show()


def generate_fat_tree_topology(n, seed = 42):
    """
    Generates a Fat-Tree topology with n pods.

    Parameters
    ----------
    n : int
        The number of pods in the Fat-Tree topology. Must be an even number greater than 0.
    seed : int, optional
        Seed for the random number generator (default is 42).

    Returns
    -------
    fnss.Topology
        A Fat-Tree topology with n pods.

    Raises
    ------
    ValueError
        If n is not an even number greater than 0.

    Notes
    -----
    A Fat-Tree is a popular topology used in data centers. It is a tree with four levels:
    core, aggregation, edge and host. The core layer consists of core switches which are
    fully interconnected. The aggregation layer consists of aggregation switches which
    are connected to all core switches. The edge layer consists of edge switches which are
    connected to all aggregation switches. Finally, the host layer consists of hosts which
    are connected to all edge switches. Fat-Tree topologies are designed to provide high
    bandwidth and low latency for data center applications.
    """
    if n <= 4 or n % 2 != 0:
        raise ValueError("n must be an even integer greater than 4")
    np.random.seed(seed)

    return fnss.fat_tree_topology(n)


def generate_jellyfish_topology(num_servers, num_switches, r, seed = 42):

    """
    Generates a Jellyfish topology with the given parameters.

    Parameters:
    -----------
    num_servers : int
        The number of servers in the topology.
    num_switches : int
        The number of switches in the topology.
    r : int
        The degree of each switch in the topology. Must be an even number.
    seed : int, optional
        Seed for the random number generator.

    Returns:
    --------
    graph : NetworkX graph
        The Jellyfish topology.

    Raises:
    -------
    ValueError
        If r is not an even number.

    """

    if r % 2 != 0:
        raise ValueError("r must be an even number.")
    
    graph = nx.random_regular_graph(r, num_switches, seed)

    attribute_update = {node: "switch" for node in graph.nodes()}

    nx.set_node_attributes(graph, attribute_update, name="type")

    hosts = ['h'+str(i) for i in range(0, num_servers)]
    graph.add_nodes_from(hosts, type="host")

    switches = [node for node in graph.nodes() if node not in hosts]

    for i in range(0, num_switches):
        switch = switches[i]
        for j in range(r*i,r*(i+1)):
            graph.add_edge(switch, hosts[j])

    return graph


def get_closest_hosts_fat_tree(ft, A, N):

    """Compute the N closest hosts to node A in a Fat-Tree topology graph.

    Parameters
    ----------
    ft : networkx.Graph
        A Fat-Tree topology graph.
    A : int or str
        The source node from which to compute distances.
    N : int
        The number of closest hosts to return.

    Returns
    -------
    dict
        A dictionary where keys are the indexes of the closest hosts and values
        are their corresponding shortest path distances from node A.

    Raises
    ------
    ValueError
        If node A is not in the Fat-Tree topology graph.
    TypeError
        If the provided Fat-Tree topology graph is not a networkx.Graph object.

    """
    if A not in ft.nodes():
        raise ValueError("Node A is not in the Fat-Tree topology graph.")
    if not isinstance(ft, nx.Graph):
        raise TypeError("The provided Fat-Tree topology graph is not a networkx.Graph object.")
    # Compute shortest path distances from node A to all other nodes
    # the "shortest_path_length" method implements Dijkstra's algorithm --> super efficient!
    dists = nx.shortest_path_length(ft, source=A)
    
    results = [(node, dists[node]) for node in dists.keys() if ft.node[node]['type'] == 'host' and node != A]

    results = dict(results[:N])
    
    closest_hosts = {}

    for i, node in enumerate(results.keys()):
        closest_hosts[i + 1] = results[node]

    return closest_hosts


def get_closest_hosts_jellyfish(G, A, N):

    """
    Returns the N closest hosts to node A in the given Jellyfish topology G, based on shortest path distances.
    
    Parameters
    ----------
    G : networkx.Graph
        The Jellyfish topology to search for closest hosts.
    A : str
        The node ID for which to find the closest hosts.
    N : int
        The number of closest hosts to return.
    
    Returns
    -------
    dict
        A dictionary containing the N closest hosts to node A, along with their shortest path distances.
        The keys of the dictionary are the rank of the host, starting from 1 (closest), and the values are
        tuples containing the host node ID and its shortest path distance to node A.
    
    Raises
    ------
    ValueError
        If A is not a valid node ID in the Jellyfish topology G.
    """

    if A not in G.nodes():
        raise ValueError("Invalid node ID.")
    # Compute shortest path distances from node A to all other nodes
    # the "shortest_path_length" method implements Dijkstra's algorithm --> super efficient!
    dists = nx.shortest_path_length(G, source=A)
    
    results = [(node, dists[node]) for node in dists.keys() if str(node).startswith('h') and node != A]

    results = dict(results[:N])
    
    closest_hosts = {}

    for i, node in enumerate(results.keys()):
        closest_hosts[i + 1] = results[node]

    return closest_hosts


def get_thetas(closest_hosts, tau, C):

    """
    Computes the average throughput to send data between a server A and a set of closest servers.

    Parameters
    ----------
    closest_hosts : dict
        A dictionary of the N closest hosts to server A, containing the host names as keys
        and the shortest path distances as values.
    tau : float
        The time it takes to transmit one packet (in seconds).
    C : float
        The capacity of the link (in bits per second).

    Returns
    -------
    np.ndarray
        An array containing the average throughput to send data between server A and
        each of the closest servers.

    Raises
    ------
    ValueError
        If tau or C are non-positive.
    TypeError
        If closest_hosts is not a dictionary.
    """

    # Check if tau and C are positive
    if tau <= 0 or C <= 0:
        raise ValueError("tau and C must be positive.")

    # Check if closest_hosts is a dictionary
    if not isinstance(closest_hosts, dict):
        raise TypeError("closest_hosts must be a dictionary.")
    

    t_i = np.array([(h * tau * 2) for h in closest_hosts.values()])

    theta_i = (C * (1 / t_i)) / sum(1 / t_i)

    return theta_i #Average throughput to send data between server A and server i


def get_response_time_fat_tree(ft, N, A, tau = 0.5 * 10**-6, C = 10, L_f = 4 * 8000, L_o = 4 * 8000, E_x = 2880, f = 48/1500, T_0 = 30):

    """
    Compute the mean response time for a fat-tree topology.

    Parameters
    ----------
    ft : networkx.Graph
        The fat-tree topology.
    N : int
        The number of closest servers to consider.
    A : str
        The ID of the source server.
    tau : float, optional
        The propagation delay of a link, in seconds. Default is 0.5 * 10 ** -6.
    C : int, optional
        The bandwidth of a link, in Gbps. Default is 10.
    L_f : int, optional
        The size of the data that is transmitted to each server, in bits. Default is 4 * 8000.
    L_o : int, optional
        The size of the output data from each server, in bits. Default is 4 * 8000.
    E_x : int, optional
        The execution time for a job, in cycles. Default is 2880.
    f : float, optional
        The overhead factor, used to account for protocol overhead. Default is 48 / 1500.
    T_0 : int, optional
        The time needed for a server to process its input data, in cycles. Default is 30.

    Returns
    -------
    tuple of float
        The maximum response time and the sum of all server computation times, both in seconds.

    Raises
    ------
    ValueError
        If `A` is not a valid node ID in the graph.
    """

    if A not in ft:
        raise ValueError("Invalid source node ID.")

    # get the N closest servers to server A, along with the number of hops
    closest_hosts = get_closest_hosts_fat_tree(ft, A, N)

    # calculate the the average throughput for each of the N servers, in Gbit/s
    average_throughputs = get_thetas(closest_hosts, tau, C)

    # compute the amount of time in seconds needed to transmit each data fraction to each server (overhead included)
    time_forth = ((L_f / N) / average_throughputs)
    
    cpt = []
    otp = []
    
    for s in range(100):
        # calculate the time each server needs to perform its share of the computation (in seconds)
        cpt.append(np.random.exponential(scale=E_x / N, size=N) + T_0)

        # calculate the amount of data produced by each server
        otp.append(np.random.uniform(0, ((2 * L_o) / N), N))

    # compute the average of the results obtained with the previous simulation
    avg_cpt = np.array([sum(x) / len(x) for x in zip(*cpt)])
    avg_opt = np.array([sum(x) / len(x) for x in zip(*otp)])
    
    # calculate the time in seconds needed to send each fraction of processed data back to server A (overhead included)
    time_back = ((avg_opt + (avg_opt * f)) / average_throughputs)
    
    # compute the mean for each server
    mrt = time_forth + avg_cpt + time_back

    # the mean response time is calculated as the longest time taken to return a processed portion of data back to A. As well, we return the average sum of all computations
    return np.max(mrt), np.sum(avg_cpt)


def get_response_time_jellyfish(G, N, A, tau = 0.5 * 10**-6, C = 10, L_f = 4 * 8000, L_o = 4 * 8000, E_x = 2880, f = 48/1500, T_0 = 30):

    """
    Compute the mean response time for a jelly-fish topology.

    Parameters
    ----------
    G : networkx.Graph
        The jellyfish topology.
    N : int
        The number of closest servers to consider.
    A : str
        The ID of the source server.
    tau : float, optional
        The propagation delay of a link, in seconds. Default is 0.5 * 10 ** -6.
    C : int, optional
        The bandwidth of a link, in Gbps. Default is 10.
    L_f : int, optional
        The size of the data that is transmitted to each server, in bits. Default is 4 * 8000.
    L_o : int, optional
        The size of the output data from each server, in bits. Default is 4 * 8000.
    E_x : int, optional
        The execution time for a job, in cycles. Default is 2880.
    f : float, optional
        The overhead factor, used to account for protocol overhead. Default is 48 / 1500.
    T_0 : int, optional
        The time needed for a server to process its input data, in cycles. Default is 30.

    Returns
    -------
    tuple of float
        The maximum response time and the sum of all server computation times, both in seconds.

    Raises
    ------
    ValueError
        If `A` is not a valid node ID in the graph.
    """

    if A not in G:
        raise ValueError("Invalid source node ID.")


    # get the N closest servers to server A, along with the number of hops
    closest_hosts = get_closest_hosts_jellyfish(G, A, N)

    # calculate the the average throughput for each of the N servers, in Gbit/s
    average_throughputs = get_thetas(closest_hosts, tau, C)

    # compute the amount of time in seconds needed to transmit each data fraction to each server (overhead included)
    time_forth = ((L_f / N) / average_throughputs)
    
    cpt = []
    otp = []
    
    for s in range(100):
        # calculate the time each server needs to perform its share of the computation (in seconds)
        cpt.append(np.random.exponential(scale=E_x / N, size=N) + T_0)

        # calculate the amount of data produced by each server
        otp.append(np.random.uniform(0, ((2 * L_o) / N), N))

    # compute the average of the results obtained with the previous simulation
    avg_cpt = np.array([sum(x) / len(x) for x in zip(*cpt)])
    avg_opt = np.array([sum(x) / len(x) for x in zip(*otp)])
    
    # calculate the time in seconds needed to send each fraction of processed data back to server A (overhead included)
    time_back = ((avg_opt + (avg_opt * f)) / average_throughputs)
    
    # compute the mean for each server
    mrt = time_forth + avg_cpt + time_back

    # the mean response time is calculated as the longest time taken to return a processed portion of data back to A. As well, we return the average sum of all computations
    return np.max(mrt), np.sum(avg_cpt)


def get_response_time_plot(response_time_list_fat_tree, response_time_list_jellyfish, baseline):
    
    """
    Generate a plot of the mean response time (E[R]) normalized by R_baseline for Fat-Tree and Jellyfish topologies.
    
    Parameters
    ----------
    response_time_list_fat_tree : list
        A list of response times for Fat-Tree topology.
    response_time_list_jellyfish : list
        A list of response times for Jellyfish topology.
    baseline : float
        The baseline response time for normalization.
        
    Returns
    -------
    None
        The function plots the data and displays the plot.

    Notes
    -----
    The function creates a plot that shows the the mean response time (E[R]) normalized by the baseline response time (R_{baseline})
    for both Fat-Tree and Jellyfish network architectures. The x-axis represents the number of hosts (N), while the y-axis
    represents the time taken (seconds). The plot also shows vertical lines at hops 4 and 6 and includes a legend, a title,
    and axis labels.

    Examples
    --------
    >>> get_job_running_cost_plot(job_running_costs_fat_tree, job_running_costs_jellyfish, baseline)
    """

    # Define colors
    fat_tree_color = '#ff7f0e'
    jellyfish_color = '#2ca02c'

    # Set font size
    plt.rcParams.update({'font.size': 14})

    # Create figure and axis objects
    fig, ax = plt.subplots(dpi=100)

    # Extract data from dataframes
    plot_upper_bound_ft = len(response_time_list_fat_tree) + 1
    plot_upper_bound_jf = len(response_time_list_jellyfish) + 1
    x_axis_fat_tree = range(1, plot_upper_bound_ft)
    x_axis_jellyfish = range(1, plot_upper_bound_jf)
    y_axis_fat_tree = response_time_list_fat_tree / baseline
    y_axis_jellyfish = response_time_list_jellyfish / baseline

    # Plot data
    ax.plot(x_axis_fat_tree, y_axis_fat_tree, color=fat_tree_color, linewidth=1.5, linestyle='-', label="Fat-Tree")
    ax.plot(x_axis_jellyfish, y_axis_jellyfish, color=jellyfish_color, linewidth=1.5, linestyle='-', label="Jellyfish")

    ax.vlines(32, ymin = 0.11, ymax=1.3, color = 'black', linewidth = 0.8, linestyle='--')
    ax.vlines(1024, ymin = 0.11, ymax=1.3, color = 'black', linewidth = 0.8, linestyle='--')
    ax.vlines(1055, ymin = 0.11, ymax=1.3, color = 'black', linewidth = 0.8, linestyle='--')

    # Set axis labels and title
    ax.set_title("Mean Response Time $E[R]$ normalized by $R_{baseline}$", fontsize=16)
    ax.set_xlabel('Number of Hosts ($N$), log-scaled', fontsize=14)
    ax.set_ylabel('Time Taken', fontsize=14)

    ax.set_xscale('log')

    # Set axis ticks
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.gcf().text(0.44, 0.2, "32", fontsize=12)
    plt.gcf().text(0.675, 0.2, "1024", fontsize=12)
    plt.gcf().text(0.745, 0.2, "1055", fontsize=12)

    # Add legend with larger font size and transparent background
    ax.legend(fontsize=12, fancybox=True, facecolor='none', loc='lower left')

    # Add grid lines
    ax.grid(which='both', alpha=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle='--')
    ax.yaxis.grid(True, linestyle='--')

    # Add margin and tighten layout
    plt.margins(0.02)
    fig.tight_layout()

    # Display the plot
    return plt.show()


def get_job_running_cost_plot(job_running_costs_fat_tree, job_running_costs_jellyfish, baseline):
    
    """
    Plot job running costs for a Fat-Tree and Jellyfish network architecture.

    Parameters
    ----------
    job_running_costs_fat_tree : pandas.core.series.Series
        A pandas series containing the job running costs for a Fat-Tree network.
    job_running_costs_jellyfish : pandas.core.series.Series
        A pandas series containing the job running costs for a Jellyfish network.
    baseline : float
        The baseline job running cost to normalize the data.

    Returns
    -------
    None
        Displays the plot.

    Notes
    -----
    The function creates a plot that shows the job running cost (S) normalized by the baseline job running cost (S_baseline)
    for both Fat-Tree and Jellyfish network architectures. The x-axis represents the number of hosts (N), while the y-axis
    represents the time taken (seconds). The plot also shows vertical lines at hops 4 and 6 and includes a legend, a title,
    and axis labels.

    Examples
    --------
    >>> get_job_running_cost_plot(job_running_costs_fat_tree, job_running_costs_jellyfish, baseline)
    """
    # Define colors
    fat_tree_color = '#ff7f0e'
    jellyfish_color = '#2ca02c'

    # Set font size
    plt.rcParams.update({'font.size': 14})

    # Create figure and axis objects
    fig, ax = plt.subplots(dpi=100)

    # Extract data from dataframes
    plot_upper_bound_ft = len(job_running_costs_fat_tree) + 1
    plot_upper_bound_jf = len(job_running_costs_jellyfish) + 1
    x_axis_fat_tree = range(1, plot_upper_bound_ft)
    x_axis_jellyfish = range(1, plot_upper_bound_jf)
    y_axis_fat_tree = job_running_costs_fat_tree / baseline
    y_axis_jellyfish = job_running_costs_jellyfish / baseline

    # Plot data
    ax.plot(x_axis_fat_tree, y_axis_fat_tree, color=fat_tree_color, linewidth=1.5, linestyle='-', label="Fat-Tree")
    ax.plot(x_axis_jellyfish, y_axis_jellyfish, color=jellyfish_color, linewidth=1.5, linestyle='-', label="Jellyfish")

    ax.vlines(32, ymin = 0.11, ymax=1.3, color = 'black', linewidth = 0.8, linestyle='--')
    ax.vlines(1024, ymin = 0.11, ymax=1.3, color = 'black', linewidth = 0.8, linestyle='--')
    ax.vlines(1055, ymin = 0.11, ymax=1.3, color = 'black', linewidth = 0.8, linestyle='--')

    # Set axis labels and title
    ax.set_title("Job Running Cost (S) normalized by $S_{baseline}$", fontsize=16)
    ax.set_xlabel('Number of Hosts ($N$), log-scaled', fontsize=14)
    ax.set_ylabel('Time Taken', fontsize=14)

    ax.set_xscale('log')

    # Set axis ticks
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.gcf().text(0.44, 0.2, "32", fontsize=12)
    plt.gcf().text(0.675, 0.2, "1024", fontsize=12)
    plt.gcf().text(0.745, 0.2, "1055", fontsize=12)

    x_min_ft = np.argmin(job_running_costs_fat_tree)
    y_min_ft = min(job_running_costs_fat_tree) / baseline
    x_min_jf = np.argmin(job_running_costs_jellyfish)
    y_min_jf = min(job_running_costs_jellyfish) / baseline

    if np.argmin(job_running_costs_fat_tree) == np.argmin(job_running_costs_jellyfish):
        plt.scatter(x_min_ft, y_min_ft, color="red", s=50)
        plt.annotate(x_min_ft, (x_min_ft, y_min_ft-.05))
    else:
        plt.scatter(x_min_ft, y_min_ft, color="red", s=50)
        plt.scatter(x_min_jf, y_min_jf, color="blue", s=50)

    # Add legend with larger font size and transparent background
    ax.legend(fontsize=12, fancybox=True, facecolor='none', loc='lower left')

    # Add grid lines
    ax.grid(which='both', alpha=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle='--')
    ax.yaxis.grid(True, linestyle='--')

    # Add margin and tighten layout
    plt.margins(0.02)
    fig.tight_layout()

    # Display the plot
    return plt.show()