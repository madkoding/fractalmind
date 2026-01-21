//! Similarity computation utilities for graph algorithms.

#![allow(dead_code)]

use crate::models::EmbeddingVector;

/// Computes cosine similarity between two embedding vectors.
pub fn cosine_similarity(a: &EmbeddingVector, b: &EmbeddingVector) -> f32 {
    a.cosine_similarity(b)
}

/// Computes the centroid (average) of multiple embedding vectors.
pub fn compute_centroid(embeddings: &[&EmbeddingVector]) -> Option<EmbeddingVector> {
    if embeddings.is_empty() {
        return None;
    }

    let dimension = embeddings[0].dimension;
    if dimension == 0 {
        return None;
    }

    // Verify all embeddings have the same dimension
    if !embeddings.iter().all(|e| e.dimension == dimension) {
        return None;
    }

    let mut centroid = vec![0.0f32; dimension];
    let count = embeddings.len() as f32;

    for embedding in embeddings {
        for (i, val) in embedding.vector.iter().enumerate() {
            centroid[i] += val / count;
        }
    }

    let model = embeddings[0].model.clone();
    Some(EmbeddingVector::new(centroid, model))
}

/// Computes the average pairwise similarity within a group of embeddings.
pub fn average_pairwise_similarity(embeddings: &[&EmbeddingVector]) -> f32 {
    if embeddings.len() < 2 {
        return 1.0;
    }

    let mut total_similarity = 0.0;
    let mut count = 0;

    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            total_similarity += cosine_similarity(embeddings[i], embeddings[j]);
            count += 1;
        }
    }

    if count == 0 {
        1.0
    } else {
        total_similarity / count as f32
    }
}

/// Computes the similarity between a point and a centroid.
pub fn similarity_to_centroid(point: &EmbeddingVector, centroid: &EmbeddingVector) -> f32 {
    cosine_similarity(point, centroid)
}

/// Finds the most similar embedding to a query from a list.
pub fn find_most_similar<'a>(
    query: &EmbeddingVector,
    candidates: &'a [&EmbeddingVector],
) -> Option<(usize, f32)> {
    if candidates.is_empty() {
        return None;
    }

    let mut best_idx = 0;
    let mut best_similarity = f32::NEG_INFINITY;

    for (idx, candidate) in candidates.iter().enumerate() {
        let sim = cosine_similarity(query, candidate);
        if sim > best_similarity {
            best_similarity = sim;
            best_idx = idx;
        }
    }

    Some((best_idx, best_similarity))
}

/// Finds the k most similar embeddings to a query.
pub fn find_k_most_similar<'a>(
    query: &EmbeddingVector,
    candidates: &'a [&EmbeddingVector],
    k: usize,
) -> Vec<(usize, f32)> {
    if candidates.is_empty() || k == 0 {
        return Vec::new();
    }

    let mut similarities: Vec<(usize, f32)> = candidates
        .iter()
        .enumerate()
        .map(|(idx, c)| (idx, cosine_similarity(query, c)))
        .collect();

    // Sort by similarity descending
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    similarities.into_iter().take(k).collect()
}

/// Computes a similarity matrix for a list of embeddings.
pub fn similarity_matrix(embeddings: &[&EmbeddingVector]) -> Vec<Vec<f32>> {
    let n = embeddings.len();
    let mut matrix = vec![vec![0.0f32; n]; n];

    for i in 0..n {
        matrix[i][i] = 1.0; // Self-similarity is 1
        for j in (i + 1)..n {
            let sim = cosine_similarity(embeddings[i], embeddings[j]);
            matrix[i][j] = sim;
            matrix[j][i] = sim; // Symmetric
        }
    }

    matrix
}

/// Converts similarity to distance (for shortest path algorithms).
/// Returns 1/similarity, with a minimum distance to avoid division by zero.
pub fn similarity_to_distance(similarity: f32) -> f32 {
    if similarity <= 0.0 {
        f32::MAX
    } else {
        1.0 / similarity
    }
}

/// Converts distance back to similarity.
pub fn distance_to_similarity(distance: f32) -> f32 {
    if distance <= 0.0 {
        1.0
    } else if distance >= f32::MAX {
        0.0
    } else {
        1.0 / distance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::EmbeddingModel;

    fn make_embedding(values: Vec<f32>) -> EmbeddingVector {
        EmbeddingVector::new(values, EmbeddingModel::NomicEmbedTextV15)
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = make_embedding(vec![1.0, 0.0, 0.0]);
        let b = make_embedding(vec![1.0, 0.0, 0.0]);
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = make_embedding(vec![1.0, 0.0, 0.0]);
        let b = make_embedding(vec![0.0, 1.0, 0.0]);
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_compute_centroid() {
        let e1 = make_embedding(vec![1.0, 0.0, 0.0]);
        let e2 = make_embedding(vec![0.0, 1.0, 0.0]);
        let embeddings: Vec<&EmbeddingVector> = vec![&e1, &e2];

        let centroid = compute_centroid(&embeddings).unwrap();
        let values = &centroid.vector;

        assert!((values[0] - 0.5).abs() < 0.001);
        assert!((values[1] - 0.5).abs() < 0.001);
        assert!((values[2] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_centroid_empty() {
        let embeddings: Vec<&EmbeddingVector> = vec![];
        assert!(compute_centroid(&embeddings).is_none());
    }

    #[test]
    fn test_average_pairwise_similarity_single() {
        let e1 = make_embedding(vec![1.0, 0.0, 0.0]);
        let embeddings: Vec<&EmbeddingVector> = vec![&e1];
        let sim = average_pairwise_similarity(&embeddings);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_average_pairwise_similarity() {
        let e1 = make_embedding(vec![1.0, 0.0]);
        let e2 = make_embedding(vec![0.707, 0.707]); // 45 degrees
        let e3 = make_embedding(vec![0.0, 1.0]);

        let embeddings: Vec<&EmbeddingVector> = vec![&e1, &e2, &e3];
        let sim = average_pairwise_similarity(&embeddings);

        // Should be average of 3 pairwise similarities
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn test_find_most_similar() {
        let query = make_embedding(vec![1.0, 0.0]);
        let c1 = make_embedding(vec![0.0, 1.0]); // Orthogonal
        let c2 = make_embedding(vec![0.9, 0.1]); // Similar
        let c3 = make_embedding(vec![0.5, 0.5]); // Medium

        let candidates: Vec<&EmbeddingVector> = vec![&c1, &c2, &c3];
        let (idx, sim) = find_most_similar(&query, &candidates).unwrap();

        assert_eq!(idx, 1); // c2 is most similar
        assert!(sim > 0.9);
    }

    #[test]
    fn test_find_k_most_similar() {
        let query = make_embedding(vec![1.0, 0.0]);
        let c1 = make_embedding(vec![0.0, 1.0]);
        let c2 = make_embedding(vec![0.9, 0.1]);
        let c3 = make_embedding(vec![0.5, 0.5]);

        let candidates: Vec<&EmbeddingVector> = vec![&c1, &c2, &c3];
        let results = find_k_most_similar(&query, &candidates, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1); // c2 first
        assert_eq!(results[1].0, 2); // c3 second
    }

    #[test]
    fn test_similarity_matrix() {
        let e1 = make_embedding(vec![1.0, 0.0]);
        let e2 = make_embedding(vec![0.0, 1.0]);
        let e3 = make_embedding(vec![1.0, 0.0]);

        let embeddings: Vec<&EmbeddingVector> = vec![&e1, &e2, &e3];
        let matrix = similarity_matrix(&embeddings);

        assert_eq!(matrix.len(), 3);
        assert!((matrix[0][0] - 1.0).abs() < 0.001); // Self-similarity
        assert!((matrix[0][2] - 1.0).abs() < 0.001); // e1 and e3 identical
        assert!(matrix[0][1].abs() < 0.001); // e1 and e2 orthogonal
    }

    #[test]
    fn test_similarity_to_distance() {
        assert!((similarity_to_distance(1.0) - 1.0).abs() < 0.001);
        assert!((similarity_to_distance(0.5) - 2.0).abs() < 0.001);
        assert!(similarity_to_distance(0.0) > 1000000.0);
    }

    #[test]
    fn test_distance_to_similarity() {
        assert!((distance_to_similarity(1.0) - 1.0).abs() < 0.001);
        assert!((distance_to_similarity(2.0) - 0.5).abs() < 0.001);
        assert!((distance_to_similarity(f32::MAX) - 0.0).abs() < 0.001);
    }
}
