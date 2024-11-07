import numpy as np
from sklearn.decomposition import PCA
import trimesh
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
import os
import wandb
import joblib
from datetime import datetime
import logging

class SpineParametricModel:
    def __init__(self, stl_directory, wandb_project="spine-parametric-model"):
        """Initialize the parametric spine model"""
        self.stl_directory = stl_directory
        self.wandb_project = wandb_project
        self.pca_model = None
        self.mean_shape = None
        self.parameters_df = None
        self.vertex_pca = None
        self.scaler = StandardScaler()
        self.param_to_pca = None
        self.pca_to_param = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Parameters setup remains the same
        self.PRIMARY_PARAMS = ['PI', 'PT', 'SS', 'LL', 'GT', 'LDI', 'TPA', 'cobb_angle']
        self.DERIVED_PARAMS = ['LL-PI', 'RPV', 'RLL', 'RSA', 'GAP']
        self.ALL_PARAMS = self.PRIMARY_PARAMS + self.DERIVED_PARAMS
        
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
        
    def load_mesh_by_id(self, id):
        """Load a mesh by its ID from the STL directory"""
        # Remove .0 from the ID if present
        if isinstance(id, str):
            id = id.replace('.0', '')
        elif isinstance(id, float):
            id = int(id)
        
        # Try different possible filename formats
        possible_filenames = [
            f"{id}.stl",
            f"{id}.0.stl",
            f"{int(id)}.stl",
            f"{int(id)}.0.stl"
        ]
        
        for filename in possible_filenames:
            stl_path = os.path.join(self.stl_directory, filename)
            if os.path.exists(stl_path):
                mesh = trimesh.load(stl_path)
                # Center the mesh
                mesh.vertices -= mesh.vertices.mean(axis=0)
                return mesh
                
        # If no file is found, list the directory contents for debugging
        dir_contents = os.listdir(self.stl_directory)
        self.logger.error(f"Available files in directory: {dir_contents[:5]}...")
        raise FileNotFoundError(f"Mesh file not found for ID: {id}. Tried: {possible_filenames}")

    def load_training_data(self):
        """Load all meshes and their corresponding parameters"""
        print("Loading training data...")
        meshes = []
        parameters = []
        
        for idx, row in self.parameters_df.iterrows():
            try:
                mesh = self.load_mesh_by_id(row['IDs'])
                meshes.append(mesh.vertices.flatten())
                parameters.append([row[param] for param in self.ALL_PARAMS])
                
                if (idx + 1) % 100 == 0:
                    print(f"Loaded {idx + 1} meshes...")
            except Exception as e:
                print(f"Error loading data for ID {row['IDs']}: {str(e)}")
                continue
        
        # Debugging: Print the number of loaded meshes and parameters
        print(f"Number of meshes loaded: {len(meshes)}")
        print(f"Number of parameters loaded: {len(parameters)}")
        
        if not meshes or not parameters:  # Check if any data was loaded
            raise ValueError("No meshes or parameters loaded. Check your data files.")
        
        return np.array(meshes), np.array(parameters)

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

    def train_model(self, n_components=5, wandb_config=None):
        """Train the statistical shape model with wandb logging"""
        if self.parameters_df is None:
            raise ValueError("Parameters not loaded. Call load_parameters first.")
        
        # Initialize wandb
        if wandb_config is None:
            wandb_config = {
                'n_components': n_components,
                'ridge_alpha': 1.0,
                'dataset_size': len(self.parameters_df)
            }
        
        wandb.init(project=self.wandb_project, config=wandb_config)
        
        # Load training data
        print("Loading meshes and parameters...")
        X, Y = self.load_training_data()
        wandb.log({'dataset_size': len(X)})
        
        # Scale parameters
        print("Scaling parameters...")
        Y_scaled = self.scaler.fit_transform(Y)
        
        # Perform PCA
        print("Performing PCA...")
        self.pca_model = PCA(n_components=n_components)
        self.vertex_pca = self.pca_model.fit_transform(X)
        self.mean_shape = self.pca_model.mean_
        
        # Log PCA metrics
        explained_variance_ratio = self.pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
            wandb.log({
                f'pc_{i+1}_variance_ratio': var_ratio,
                f'pc_{i+1}_cumulative_variance': cum_var
            })
        
        # Train parameter mappings
        print("Training parameter mappings...")
        self.param_to_pca = MultiOutputRegressor(Ridge(alpha=wandb_config['ridge_alpha']))
        self.pca_to_param = MultiOutputRegressor(Ridge(alpha=wandb_config['ridge_alpha']))
        
        # Fit and evaluate mappings
        self.param_to_pca.fit(Y_scaled, self.vertex_pca)
        self.pca_to_param.fit(self.vertex_pca, Y_scaled)
        
        # Compute and log reconstruction errors
        pca_pred = self.param_to_pca.predict(Y_scaled)
        vertices_reconstructed = self.pca_model.inverse_transform(pca_pred)
        reconstruction_error = np.mean(np.square(X - vertices_reconstructed))
        wandb.log({
            'reconstruction_error': reconstruction_error,
            'total_variance_explained': cumulative_variance[-1]
        })
        
        variance_explained = self.pca_model.explained_variance_ratio_.sum() * 100
        print(f"\nModel trained with {n_components} components explaining {variance_explained:.2f}% of variance")
        
        # Log summary metrics
        wandb.run.summary['final_reconstruction_error'] = reconstruction_error
        wandb.run.summary['total_variance_explained'] = cumulative_variance[-1]
        
        return reconstruction_error

    def save_model(self, output_dir):
        """Save the trained model and all necessary components"""
        os.makedirs(output_dir, exist_ok=True)
        
        model_components = {
            'pca_model': self.pca_model,
            'mean_shape': self.mean_shape,
            'scaler': self.scaler,
            'param_to_pca': self.param_to_pca,
            'pca_to_param': self.pca_to_param,
            'param_stats': self.param_stats,
            'parameters_df': self.parameters_df
        }
        
        model_path = os.path.join(output_dir, 'spine_model.joblib')
        joblib.dump(model_components, model_path)
        print(f"Model saved to {model_path}")
        
        # Log model artifact to wandb
        if wandb.run is not None:
            artifact = wandb.Artifact('spine_model', type='model')
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

def main():
    try:
        # Get absolute path to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set up paths relative to script directory
        stl_directory = os.path.join(script_dir, "stl")
        parameters_file = os.path.join(script_dir, "parameters.csv")
        
        print(f"Looking for STL files in: {stl_directory}")
        print(f"Using parameters file: {parameters_file}")
        
        # Verify directory exists
        if not os.path.exists(stl_directory):
            raise ValueError(f"STL directory not found: {stl_directory}")
        
        # List contents of STL directory
        print("\nContents of STL directory:")
        for file in os.listdir(stl_directory):
            print(f"- {file}")
            
        # Initialize model
        model = SpineParametricModel(stl_directory)
        
        # Print first few rows of parameters file
        print("\nFirst few rows of parameters file:")
        df = pd.read_csv(parameters_file)
        print(df.head())
        
        # Load parameters (this will filter for only existing files)
        model.load_parameters(parameters_file)
        
        # Configure wandb
        wandb_config = {
            'n_components': min(5, len(model.parameters_df) - 1),  # Ensure n_components is valid
            'ridge_alpha': 1.0,
            'learning_rate': 0.001,
            'batch_size': min(32, len(model.parameters_df))
        }
        
        # Train model
        reconstruction_error = model.train_model(
            n_components=wandb_config['n_components'],
            wandb_config=wandb_config
        )
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"trained_models/{timestamp}"
        model.save_model(output_dir)
        
        print(f"Training completed with reconstruction error: {reconstruction_error}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()