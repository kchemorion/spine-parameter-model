import numpy as np
import pandas as pd
import trimesh
import os
import wandb
import joblib
import logging
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

class SpineParametricModel:
    def __init__(self, stl_directory, wandb_project="spine-parametric-model", n_components=5):
        """
        Initialize the parametric spine model
        
        Args:
            stl_directory (str): Directory containing STL files
            wandb_project (str): W&B project name
            n_components (int): Number of PCA components
        """
        self.stl_directory = stl_directory
        self.wandb_project = wandb_project
        self.n_components = n_components
        
        # Model components
        self.pca_model = None
        self.mean_shape = None
        self.parameters_df = None
        self.vertex_pca = None
        self.scaler = StandardScaler()
        self.param_to_pca = None
        self.pca_to_param = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Define parameters
        self.PRIMARY_PARAMS = ['PI', 'PT', 'SS', 'LL', 'GT', 'LDI', 'TPA', 'cobb_angle']
        self.DERIVED_PARAMS = ['LL-PI', 'RPV', 'RLL', 'RSA', 'GAP']
        self.ALL_PARAMS = self.PRIMARY_PARAMS + self.DERIVED_PARAMS
        
        # Parameter relationships
        self.PARAM_RELATIONSHIPS = {
            'PI': lambda p: p['PT'] + p['SS'],
            'PT': lambda p: p['PI'] - p['SS'],
            'SS': lambda p: p['PI'] - p['PT'],
            'LL-PI': lambda p: p['LL'] - p['PI'],
            'RPV': lambda p: p['SS'] - (p['PI'] * 0.59),
            'RLL': lambda p: p['LL'] - (p['PI'] * 0.62 + 9),
            'RSA': lambda p: p['GT'] - (p['PI'] * 0.48 - 15),
            'GAP': lambda p: p['RPV'] + p['RLL'] + p['LDI'] + p['RSA']
        }
        
        # Training metrics
        self.metrics = {
            'train_reconstruction_error': [],
            'val_reconstruction_error': [],
            'train_r2': [],
            'val_r2': []
        }

    def preprocess_mesh(self, mesh):
        """Preprocess a mesh by centering and normalizing"""
        mesh.vertices -= mesh.vertices.mean(axis=0)
        max_dist = np.max(np.abs(mesh.vertices))
        mesh.vertices /= max_dist
        return mesh

    def load_parameters(self, parameters_file):
        """Load and validate parameter data"""
        try:
            # Load parameters
            df = pd.read_csv(parameters_file)
            
            # Print the first few rows and IDs for debugging
            self.logger.info("First few IDs from parameters file:")
            self.logger.info(df['IDs'].head())
            
            # List files in STL directory for debugging
            stl_files = os.listdir(self.stl_directory)
            self.logger.info("First few STL files:")
            self.logger.info(stl_files[:5])
            if 'Cobb angle (scoliosis)' in df.columns:
                df = df.rename(columns={'Cobb angle (scoliosis)': 'cobb_angle'})
            
            # Verify required columns
            required_columns = ['IDs'] + [p for p in self.PRIMARY_PARAMS if p != 'cobb_angle']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Filter to only include rows with existing files
            self.logger.info("Verifying STL files...")
            df = self.verify_data_files(df)
            
            # Check if df is empty after verification
            if df.empty:
                raise ValueError("No valid STL files found after verification.")
            
            # Calculate derived parameters
            validated_data = []
            for _, row in df.iterrows():
                params = {col: row[col] for col in df.columns if col in self.ALL_PARAMS}
                complete_params = self.calculate_dependent_parameters(params)
                complete_params['IDs'] = row['IDs']
                validated_data.append(complete_params)
            
            self.parameters_df = pd.DataFrame(validated_data)
            
            # Debugging: Print the shape of parameters_df
            print(f"Loaded parameters DataFrame shape: {self.parameters_df.shape}")
            
            # Compute statistics
            self.param_stats = {}
            for param in self.ALL_PARAMS:
                values = self.parameters_df[param].values
                self.param_stats[param] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            
            self.logger.info(f"Successfully loaded {len(self.parameters_df)} valid parameter sets")
            
            # Verify minimum dataset size
            if len(self.parameters_df) < 2:
                raise ValueError("Need at least 2 valid samples for training")
                
        except Exception as e:
            self.logger.error(f"Error loading parameters: {str(e)}")
            raise

    def calculate_dependent_parameters(self, params):
        """Calculate all dependent parameters based on relationships"""
        complete_params = params.copy()
        
        # Calculate primary relationships if missing
        if 'SS' not in complete_params and 'PI' in complete_params and 'PT' in complete_params:
            complete_params['SS'] = self.PARAM_RELATIONSHIPS['SS'](complete_params)
        if 'PT' not in complete_params and 'PI' in complete_params and 'SS' in complete_params:
            complete_params['PT'] = self.PARAM_RELATIONSHIPS['PT'](complete_params)
        if 'PI' not in complete_params and 'PT' in complete_params and 'SS' in complete_params:
            complete_params['PI'] = self.PARAM_RELATIONSHIPS['PI'](complete_params)
            
        # Calculate derived parameters
        if all(p in complete_params for p in ['LL', 'PI']):
            complete_params['LL-PI'] = self.PARAM_RELATIONSHIPS['LL-PI'](complete_params)
            
        if all(p in complete_params for p in ['SS', 'PI']):
            complete_params['RPV'] = self.PARAM_RELATIONSHIPS['RPV'](complete_params)
            
        if all(p in complete_params for p in ['LL', 'PI']):
            complete_params['RLL'] = self.PARAM_RELATIONSHIPS['RLL'](complete_params)
            
        if all(p in complete_params for p in ['GT', 'PI']):
            complete_params['RSA'] = self.PARAM_RELATIONSHIPS['RSA'](complete_params)
            
        if all(p in complete_params for p in ['RPV', 'RLL', 'LDI', 'RSA']):
            complete_params['GAP'] = self.PARAM_RELATIONSHIPS['GAP'](complete_params)
            
        return complete_params

    def verify_data_files(self, df):
        """Verify which files exist and return filtered dataframe"""
        valid_rows = []
        
        # List all files in the directory
        existing_files = os.listdir(self.stl_directory)
        self.logger.info(f"Found {len(existing_files)} files in {self.stl_directory}")
        self.logger.info(f"First few files: {existing_files[:5]}")
        
        for idx, row in df.iterrows():
            # Try different possible filename formats
            possible_filenames = [
                f"{int(row['IDs'])}.stl",  # 1.stl
                f"{row['IDs']}.stl",       # 1.0.stl
                f"{int(row['IDs'])}.0.stl", # 1.0.stl
                f"model_{int(row['IDs'])}.stl"  # model_1.stl
            ]
            
            file_found = False
            for filename in possible_filenames:
                stl_path = os.path.join(self.stl_directory, filename)
                if os.path.exists(stl_path):
                    self.logger.info(f"Found file: {stl_path}")
                    valid_rows.append(row)
                    file_found = True
                    break
                    
            if not file_found:
                self.logger.warning(f"No matching file found for ID {row['IDs']}")
                self.logger.warning(f"Tried filenames: {possible_filenames}")
        
        if not valid_rows:
            raise ValueError("No valid STL files found in the specified directory")
        
        return pd.DataFrame(valid_rows)

    def load_mesh_by_id(self, id_):
        """Load and preprocess a mesh by its ID with memory optimization"""
        try:
            if isinstance(id_, (float, np.float64)):
                id_ = int(id_)
            
            possible_filenames = [
                f"{id_}.stl",
                f"{int(id_)}.stl",
                f"{id_}.0.stl"
            ]
            
            for filename in possible_filenames:
                stl_path = os.path.join(self.stl_directory, filename)
                if os.path.exists(stl_path):
                    # Use process=False to prevent automatic processing
                    mesh = trimesh.load(stl_path, process=False, encoding='utf-8')
                    # Free up memory
                    mesh.process = False
                    mesh.merge_vertices()
                    # Normalize
                    mesh.vertices -= mesh.vertices.mean(axis=0)
                    max_dist = np.max(np.abs(mesh.vertices))
                    mesh.vertices /= max_dist
                    return mesh.vertices.flatten()  # Return flattened vertices directly
                    
            raise FileNotFoundError(f"No mesh file found for ID {id_}")
            
        except Exception as e:
            self.logger.error(f"Error loading mesh {id_}: {str(e)}")
            raise

    def prepare_training_data(self, test_size=0.2, batch_size=32):
        """Prepare training data with proper batch handling"""
        meshes = []
        parameters = []
        total_rows = len(self.parameters_df)
        
        self.logger.info(f"Loading {total_rows} meshes in batches of {batch_size}...")
        
        # Calculate number of batches
        n_batches = (total_rows + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(n_batches), desc="Loading meshes"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = self.parameters_df.iloc[start_idx:end_idx]
            
            batch_meshes = []
            batch_params = []
            
            for _, row in batch_df.iterrows():
                try:
                    vertices = self.load_mesh_by_id(row['IDs'])
                    if vertices is not None:  # Only add if mesh loading succeeded
                        batch_meshes.append(vertices)
                        batch_params.append([row[param] for param in self.ALL_PARAMS])
                except Exception as e:
                    self.logger.warning(f"Skipping ID {row['IDs']}: {str(e)}")
                    continue
            
            if batch_meshes:
                meshes.extend(batch_meshes)
                parameters.extend(batch_params)
            
            # Clear batch data
            del batch_meshes
            del batch_params
        
        if not meshes:
            raise ValueError("No valid meshes loaded")
        
        self.logger.info(f"Successfully loaded {len(meshes)} meshes")
        
        # Convert to arrays
        X = np.array(meshes)
        y = np.array(parameters)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        return X_train, X_val, y_train, y_val

    def train_model(self, wandb_config=None):
        """Train the model with optimized batch processing"""
        if wandb_config is None:
            wandb_config = {
                'n_components': self.n_components,
                'ridge_alpha': 1.0,
                'test_size': 0.2,
                'batch_size': 32,
                'min_pca_batch_size': 10  # Ensure minimum batch size for PCA
            }
        
        # Initialize wandb
        wandb.init(project=self.wandb_project, config=wandb_config)
        
        try:
            # Prepare data
            X_train, X_val, y_train, y_val = self.prepare_training_data(
                test_size=wandb_config['test_size'],
                batch_size=max(wandb_config['batch_size'], wandb_config['n_components'] + 5)
            )
            
            # Verify we have enough samples
            if len(X_train) < wandb_config['n_components']:
                raise ValueError(f"Not enough training samples ({len(X_train)}) for {wandb_config['n_components']} components")
            
            # Scale parameters
            y_train_scaled = self.scaler.fit_transform(y_train)
            y_val_scaled = self.scaler.transform(y_val)
            
            # Train PCA model
            self.logger.info("Training PCA model...")
            self.pca_model = PCA(n_components=wandb_config['n_components'])
            
            # Fit PCA on full training data
            self.pca_model.fit(X_train)
            train_pca = self.pca_model.transform(X_train)
            val_pca = self.pca_model.transform(X_val)
            
            # Train parameter mappings
            self.logger.info("Training parameter mappings...")
            self.param_to_pca = MultiOutputRegressor(
                Ridge(alpha=wandb_config['ridge_alpha'])
            )
            self.pca_to_param = MultiOutputRegressor(
                Ridge(alpha=wandb_config['ridge_alpha'])
            )
            
            # Fit models
            self.param_to_pca.fit(y_train_scaled, train_pca)
            self.pca_to_param.fit(train_pca, y_train_scaled)
            
            # Compute metrics
            train_metrics = self.compute_metrics_batched(
                X_train, y_train_scaled, train_pca, 
                batch_size=wandb_config['batch_size']
            )
            val_metrics = self.compute_metrics_batched(
                X_val, y_val_scaled, val_pca, 
                batch_size=wandb_config['batch_size']
            )
            
            # Log metrics
            self.logger.info("Logging metrics...")
            wandb.log({
                'train_reconstruction_error': train_metrics['reconstruction_error'],
                'val_reconstruction_error': val_metrics['reconstruction_error'],
                'train_r2': train_metrics['r2'],
                'val_r2': val_metrics['r2'],
                'n_training_samples': len(X_train),
                'n_validation_samples': len(X_val)
            })
            
            # Log PCA explained variance
            for i, var in enumerate(self.pca_model.explained_variance_ratio_):
                wandb.log({f'PC{i+1}_explained_variance': var})
            
            return train_metrics, val_metrics
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
    
    def compute_metrics_batched(self, X, y_scaled, pca_coords, batch_size):
        """Compute metrics in batches to save memory"""
        total_mse = 0
        total_samples = 0
        
        for i in range(0, len(X), batch_size):
            batch_end = min(i + batch_size, len(X))
            
            # Get batch data
            X_batch = X[i:batch_end]
            pca_batch = pca_coords[i:batch_end]
            y_batch = y_scaled[i:batch_end]
            
            # Reconstruction error
            X_reconstructed = self.pca_model.inverse_transform(pca_batch)
            batch_mse = np.sum(np.square(X_batch - X_reconstructed))
            total_mse += batch_mse
            total_samples += len(X_batch)
            
            # Clear memory
            del X_reconstructed
        
        # Parameter prediction R2
        y_pred = self.pca_to_param.predict(pca_coords)
        r2 = r2_score(y_scaled, y_pred)
        
        return {
            'reconstruction_error': total_mse / total_samples,
            'r2': r2
        }

    def compute_metrics(self, X, y_scaled, pca_coords):
        """Compute model performance metrics"""
        # Reconstruction error
        X_reconstructed = self.pca_model.inverse_transform(pca_coords)
        reconstruction_error = mean_squared_error(X, X_reconstructed)
        
        # Parameter prediction R2
        y_pred = self.pca_to_param.predict(pca_coords)
        r2 = r2_score(y_scaled, y_pred)
        
        return {
            'reconstruction_error': reconstruction_error,
            'r2': r2
        }

    def save_model(self, output_dir):
        """Save the trained model and configuration"""
        os.makedirs(output_dir, exist_ok=True)
        
        model_components = {
            'pca_model': self.pca_model,
            'mean_shape': self.mean_shape,
            'scaler': self.scaler,
            'param_to_pca': self.param_to_pca,
            'pca_to_param': self.pca_to_param,
            'param_stats': self.param_stats,
            'n_components': self.n_components,
            'metrics': self.metrics
        }
        
        model_path = os.path.join(output_dir, 'spine_model.joblib')
        joblib.dump(model_components, model_path)
        
        # Log model artifact to wandb
        if wandb.run is not None:
            artifact = wandb.Artifact('spine_model', type='model')
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            
        self.logger.info(f"Model saved to {model_path}")

    @classmethod
    def load_model(cls, model_path, stl_directory):
        """Load a trained model"""
        components = joblib.load(model_path)
        
        model = cls(
            stl_directory=stl_directory,
            n_components=components['n_components']
        )
        
        model.pca_model = components['pca_model']
        model.mean_shape = components['mean_shape']
        model.scaler = components['scaler']
        model.param_to_pca = components['param_to_pca']
        model.pca_to_param = components['pca_to_param']
        model.param_stats = components['param_stats']
        model.metrics = components['metrics']
        
        return model

def main():
    try:
        # Configure with appropriate batch sizes
        wandb_config = {
            'n_components': 5,
            'ridge_alpha': 1.0,
            'test_size': 0.2,
            'batch_size': 64,  # Increased batch size
            'min_pca_batch_size': 10
        }
        
        # Initialize model
        model = SpineParametricModel(
            stl_directory="stl/",
            n_components=wandb_config['n_components']
        )
        
        # Load parameters
        model.load_parameters("parameters.csv")
        
        # Train model
        train_metrics, val_metrics = model.train_model(wandb_config)
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"trained_models/{timestamp}"
        model.save_model(output_dir)
        
        # Print results
        print("\nTraining Results:")
        print(f"Train Reconstruction Error: {train_metrics['reconstruction_error']:.4f}")
        print(f"Validation Reconstruction Error: {val_metrics['reconstruction_error']:.4f}")
        print(f"Train R2 Score: {train_metrics['r2']:.4f}")
        print(f"Validation R2 Score: {val_metrics['r2']:.4f}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()