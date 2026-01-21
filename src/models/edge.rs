#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use surrealdb::sql::Thing;
use chrono::{DateTime, Utc};

/// Tipo de relación entre nodos
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EdgeType {
    /// Relación padre-hijo en la jerarquía fractal
    ParentChild,
    /// Relación semántica entre nodos del mismo nivel
    Semantic,
    /// Relación temporal (secuencia de eventos)
    Temporal,
    /// Relación causal (causa-efecto)
    Causal,
    /// Referencia cruzada entre namespaces (personal ↔ global)
    CrossNamespace,
    /// Relación de similitud
    Similar,
}

/// Arista en el grafo fractal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalEdge {
    /// Identificador único de la arista
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Thing>,

    /// Nodo origen
    pub from: Thing,

    /// Nodo destino
    pub to: Thing,

    /// Tipo de relación
    pub edge_type: EdgeType,

    /// Peso de la arista (usado para shortest path)
    /// Calculado como: 1.0 / similitud_semántica
    /// Menor peso = mayor relevancia
    pub weight: f32,

    /// Similitud semántica entre nodos (0.0 - 1.0)
    pub similarity: f32,

    /// Confianza en la relación (0.0 - 1.0)
    pub confidence: f32,

    /// Timestamp de creación
    pub created_at: DateTime<Utc>,

    /// Timestamp de última actualización
    pub updated_at: DateTime<Utc>,
}

impl FractalEdge {
    /// Crea una nueva arista padre-hijo
    pub fn new_parent_child(from: Thing, to: Thing, similarity: f32) -> Self {
        let weight = if similarity > 0.0 {
            1.0 / similarity
        } else {
            f32::MAX
        };

        Self {
            id: None,
            from,
            to,
            edge_type: EdgeType::ParentChild,
            weight,
            similarity,
            confidence: 1.0,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    /// Crea una arista semántica entre nodos del mismo nivel
    pub fn new_semantic(from: Thing, to: Thing, similarity: f32) -> Self {
        let weight = if similarity > 0.0 {
            1.0 / similarity
        } else {
            f32::MAX
        };

        Self {
            id: None,
            from,
            to,
            edge_type: EdgeType::Semantic,
            weight,
            similarity,
            confidence: 0.8, // Menor confianza que parent-child
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    /// Crea una arista cross-namespace (personal ↔ global)
    pub fn new_cross_namespace(from: Thing, to: Thing, similarity: f32, confidence: f32) -> Self {
        let weight = if similarity > 0.0 {
            1.0 / similarity
        } else {
            f32::MAX
        };

        Self {
            id: None,
            from,
            to,
            edge_type: EdgeType::CrossNamespace,
            weight,
            similarity,
            confidence,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    /// Actualiza el peso basado en nueva similitud
    pub fn update_weight(&mut self, new_similarity: f32) {
        self.similarity = new_similarity;
        self.weight = if new_similarity > 0.0 {
            1.0 / new_similarity
        } else {
            f32::MAX
        };
        self.updated_at = Utc::now();
    }

    /// Verifica si la arista es válida (peso finito, similitud razonable)
    pub fn is_valid(&self) -> bool {
        self.weight.is_finite()
            && self.similarity >= 0.0
            && self.similarity <= 1.0
            && self.confidence >= 0.0
            && self.confidence <= 1.0
    }
}

/// Camino en el grafo (usado para shortest path)
#[derive(Debug, Clone)]
pub struct GraphPath {
    /// Nodos en el camino (ordenados)
    pub nodes: Vec<Thing>,

    /// Aristas en el camino
    pub edges: Vec<FractalEdge>,

    /// Costo total del camino (suma de pesos)
    pub total_cost: f32,

    /// Similitud promedio del camino
    pub avg_similarity: f32,
}

impl GraphPath {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            total_cost: 0.0,
            avg_similarity: 0.0,
        }
    }

    /// Agrega una arista al camino
    pub fn add_edge(&mut self, edge: FractalEdge, target_node: Thing) {
        self.total_cost += edge.weight;
        self.edges.push(edge);
        self.nodes.push(target_node);
        self.recalculate_avg_similarity();
    }

    /// Recalcula la similitud promedio
    fn recalculate_avg_similarity(&mut self) {
        if self.edges.is_empty() {
            self.avg_similarity = 0.0;
        } else {
            let sum: f32 = self.edges.iter().map(|e| e.similarity).sum();
            self.avg_similarity = sum / self.edges.len() as f32;
        }
    }

    /// Retorna la longitud del camino (número de aristas)
    pub fn length(&self) -> usize {
        self.edges.len()
    }
}

impl Default for GraphPath {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_thing(id: &str) -> Thing {
        Thing {
            tb: "nodes".to_string(),
            id: surrealdb::sql::Id::String(id.to_string()),
        }
    }

    #[test]
    fn test_new_parent_child_edge() {
        let from = create_test_thing("node1");
        let to = create_test_thing("node2");

        let edge = FractalEdge::new_parent_child(from, to, 0.8);

        assert_eq!(edge.edge_type, EdgeType::ParentChild);
        assert_eq!(edge.similarity, 0.8);
        assert!((edge.weight - 1.25).abs() < 1e-6); // 1.0 / 0.8 = 1.25
        assert_eq!(edge.confidence, 1.0);
    }

    #[test]
    fn test_update_weight() {
        let from = create_test_thing("node1");
        let to = create_test_thing("node2");

        let mut edge = FractalEdge::new_semantic(from, to, 0.5);
        assert!((edge.weight - 2.0).abs() < 1e-6); // 1.0 / 0.5 = 2.0

        edge.update_weight(0.9);
        assert_eq!(edge.similarity, 0.9);
        assert!((edge.weight - 1.111).abs() < 0.01); // 1.0 / 0.9 ≈ 1.111
    }

    #[test]
    fn test_edge_is_valid() {
        let from = create_test_thing("node1");
        let to = create_test_thing("node2");

        let edge = FractalEdge::new_semantic(from, to, 0.7);
        assert!(edge.is_valid());
    }

    #[test]
    fn test_graph_path() {
        let from = create_test_thing("node1");
        let to = create_test_thing("node2");

        let mut path = GraphPath::new();
        assert_eq!(path.length(), 0);

        let edge = FractalEdge::new_semantic(from.clone(), to.clone(), 0.8);
        path.add_edge(edge, to);

        assert_eq!(path.length(), 1);
        assert!((path.total_cost - 1.25).abs() < 1e-6);
        assert_eq!(path.avg_similarity, 0.8);
    }
}
