use rustc_hash::FxHashMap;

/// The Graph struct represents a graph as an adjacency matrix
pub struct Graph {
    /// The adjacency matrix
    pub adjacency_matrix: Vec<Vec<usize>>,

    /// Label dictionary for indexes, of adjacency matrix. Keys are valid numbers from 0 to n-1.
    pub node_index_dict: FxHashMap<usize, i32>,

    /// The set of vertices corresponding to the edge_dictionary representation
    pub n_vertices: usize,
}

impl Clone for Graph {
    fn clone(&self) -> Self {
        Graph {
            adjacency_matrix: self.adjacency_matrix.clone(),
            node_index_dict: self.node_index_dict.clone(),
            n_vertices: self.n_vertices,
        }
    }
}
