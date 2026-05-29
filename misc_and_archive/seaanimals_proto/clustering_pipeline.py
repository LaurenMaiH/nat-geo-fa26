import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

# ==========================================
# 1. PLACEHOLDER MODEL (MOCKING)
# ==========================================
class MockBioClipModel:
    """
    A dummy model to satisfy the pipeline without needing the heavy weights.
    When ready for production, swap this class out for the real BioClip model.
    """
    def __init__(self):
        self.embedding_dim = 512 # Standard dimension size for models like CLIP

    def get_embedding(self, crop_image_path):
        # Returns a random normalized vector to simulate an embedding
        vec = np.random.rand(self.embedding_dim)
        return vec / np.linalg.norm(vec)

# ==========================================
# 2. INCREMENTAL CLUSTERING
# ==========================================
class IncrementalClusterer:
    def __init__(self, distance_threshold=0.15):
        # distance_threshold: How close a new track must be to an existing 
        # centroid to join it. Lower = stricter matching.
        self.distance_threshold = distance_threshold
        self.global_centroids = [] # Stores dicts: { 'vector': array, 'count': int, 'cluster_id': int }
        self.next_cluster_id = 0
        self.cluster_assignments = [] # Tracks which crop goes to which cluster

    def process_new_track(self, track_vector, video_name, track_id, representative_crop):
        """Compares a new track point against established global centroids."""
        
        # If the scatterplot is totally empty (first video, first track)
        if not self.global_centroids:
            self._create_new_centroid(track_vector, video_name, track_id, representative_crop)
            return

        # Calculate distance between the new track and all existing centroids
        distances = [cosine(track_vector, c['vector']) for c in self.global_centroids]
        best_match_idx = np.argmin(distances)
        best_distance = distances[best_match_idx]

        if best_distance <= self.distance_threshold:
            # MATCH FOUND: Assign to existing cluster and update the centroid's coordinates
            matched_centroid = self.global_centroids[best_match_idx]
            self._update_existing_centroid(matched_centroid, track_vector)
            
            self.cluster_assignments.append({
                "video_id": video_name,
                "track_id": track_id,
                "cluster_id": matched_centroid['cluster_id'],
                "representative_crop": representative_crop
            })
        else:
            # NO MATCH: The distance is too high. Create a brand new cluster.
            self._create_new_centroid(track_vector, video_name, track_id, representative_crop)

    def _create_new_centroid(self, vector, video_name, track_id, crop_path):
        """Spawns a new centroid point in the mathematical space."""
        self.global_centroids.append({
            "cluster_id": self.next_cluster_id,
            "vector": vector,
            "count": 1 # This centroid currently represents 1 track
        })
        self.cluster_assignments.append({
            "video_id": video_name,
            "track_id": track_id,
            "cluster_id": self.next_cluster_id,
            "representative_crop": crop_path
        })
        self.next_cluster_id += 1

    def _update_existing_centroid(self, centroid_dict, new_vector):
        """Pulls the centroid slightly toward the newly added track (Moving Average)."""
        current_vec = centroid_dict['vector']
        n = centroid_dict['count']
        
        # New Average = (Old Average * N + New Value) / (N + 1)
        updated_vec = ((current_vec * n) + new_vector) / (n + 1)
        
        # Re-normalize to ensure the vector stays on the unit sphere
        centroid_dict['vector'] = updated_vec / np.linalg.norm(updated_vec)
        centroid_dict['count'] += 1

# ==========================================
# 3. THE SEQUENTIAL VIDEO PIPELINE
# ==========================================
def run_video_by_video_pipeline(csv_filepath):
    print("Loading overnight mission data...")
    df = pd.read_csv(csv_filepath)
    
    # Initialize our mock model and the sequential clustering engine
    model = MockBioClipModel()
    clusterer = IncrementalClusterer(distance_threshold=0.15)
    
    # Process sequentially: Video by Video
    unique_videos = df['video_id'].unique()
    
    for video in unique_videos:
        print(f"--> Processing {video}...")
        video_data = df[df['video_id'] == video]
        
        # Group by Track ID to collapse the 3 crops (Early, Best, Late)
        for track_id, track_group in video_data.groupby('track_id'):
            
            # Note: In production, you would parse the strings back to numpy arrays.
            # Here, we simulate generating them with our mock model.
            crop_paths = track_group['crop_path'].tolist()
            embeddings = [model.get_embedding(path) for path in crop_paths]
            
            # Collapse the 3 crops into a single "Track Point"
            track_mean_vector = np.mean(embeddings, axis=0)
            track_mean_vector = track_mean_vector / np.linalg.norm(track_mean_vector)
            
            # Find the 'best' crop image for the UI gallery
            best_crop_row = track_group[track_group['crop_type'] == 'best']
            if not best_crop_row.empty:
                rep_crop = best_crop_row['crop_path'].values[0]
            else:
                rep_crop = crop_paths[0] # Fallback
            
            # Feed the collapsed point into the incremental scatterplot!
            clusterer.process_new_track(track_mean_vector, video, track_id, rep_crop)

    # ==========================================
    # 4. EXPORT RESULTS FOR THE UI
    # ==========================================
    print(f"\nPipeline Complete! Discovered {clusterer.next_cluster_id} unique groups.")
    
    results_df = pd.DataFrame(clusterer.cluster_assignments)
    results_df.to_csv("final_incremental_clusters.csv", index=False)
    print("Saved to final_incremental_clusters.csv")

if __name__ == "__main__":
    # Point this to a dummy CSV to test the logic locally on your machine
    run_video_by_video_pipeline("Mission_Galapagos_Output/marine_tracking_results.csv")