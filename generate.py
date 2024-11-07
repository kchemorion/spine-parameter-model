import numpy as np
import trimesh
import os
import logging
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class SpineGenerator:
    def __init__(self, model_path, stl_directory):
        """
        Initialize spine generator with trained model
        
        Args:
            model_path (str): Path to trained model file (.joblib)
            stl_directory (str): Directory containing reference STL files
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load model components
        self.logger.info(f"Loading model from {model_path}")
        model_components = joblib.load(model_path)
        
        self.pca_model = model_components['pca_model']
        self.scaler = model_components['scaler']
        self.param_to_pca = model_components['param_to_pca']
        self.param_stats = model_components['param_stats']
        
        # Load reference mesh for topology and mean shape
        self.logger.info("Loading reference mesh...")
        ref_mesh_path = os.path.join(stl_directory, "1.stl")
        if not os.path.exists(ref_mesh_path):
            raise FileNotFoundError(f"Reference mesh not found at {ref_mesh_path}")
        
        self.reference_mesh = trimesh.load(ref_mesh_path)
        self.n_vertices = len(self.reference_mesh.vertices)
        
        # Store paths
        self.stl_directory = stl_directory
        
        # Define parameters
        self.PRIMARY_PARAMS = ['PI', 'PT', 'SS', 'LL', 'GT', 'LDI', 'TPA', 'cobb_angle']
        self.DERIVED_PARAMS = ['LL-PI', 'RPV', 'RLL', 'RSA', 'GAP']
        self.ALL_PARAMS = self.PRIMARY_PARAMS + self.DERIVED_PARAMS
        
        self.logger.info(f"Initialized with {self.n_vertices} vertices")
        
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

    def generate_spine(self, parameters, output_path=None, validate=True):
        """
        Generate a new spine mesh from parameters
        
        Args:
            parameters (dict): Spine parameters
            output_path (str, optional): Path to save generated STL file
            validate (bool): Whether to validate the generated mesh is different
        """
        try:
            # Generate new vertices as before
            param_vector = np.array([[
                parameters.get(param, self.param_stats[param]['mean']) 
                for param in self.ALL_PARAMS
            ]])
            param_vector_scaled = self.scaler.transform(param_vector)
            pca_coords = self.param_to_pca.predict(param_vector_scaled)
            new_vertices = self.pca_model.inverse_transform(pca_coords)
            vertices = new_vertices.reshape((1, self.n_vertices, 3))[0]
            
            # Create new mesh
            new_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=self.reference_mesh.faces,
                process=False
            )
            
            if validate:
                # Verify the mesh is different from reference
                vertex_diff = np.mean(np.abs(new_mesh.vertices - self.reference_mesh.vertices))
                self.logger.info(f"Average vertex difference from reference: {vertex_diff:.4f}")
                
                if vertex_diff < 1e-6:
                    raise ValueError("Generated mesh is identical to reference - generation failed!")
                
                # Compare mesh statistics
                stats_ref = {
                    'bounds': self.reference_mesh.bounds,
                    'center_mass': self.reference_mesh.center_mass,
                }
                
                stats_new = {
                    'bounds': new_mesh.bounds,
                    'center_mass': new_mesh.center_mass,
                }
                
                self.logger.info("\nMesh Statistics:")
                self.logger.info("Reference Mesh:")
                self.logger.info(f"Bounds: {stats_ref['bounds']}")
                self.logger.info(f"Center of Mass: {stats_ref['center_mass']}")
                
                self.logger.info("\nGenerated Mesh:")
                self.logger.info(f"Bounds: {stats_new['bounds']}")
                self.logger.info(f"Center of Mass: {stats_new['center_mass']}")
            
            # Save if requested
            if output_path:
                new_mesh.export(output_path)
                self.logger.info(f"Saved generated mesh to {output_path}")
            
            return new_mesh, parameters
            
        except Exception as e:
            self.logger.error(f"Error generating spine: {str(e)}")
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

    def interpolate_spines(self, params1, params2, t, output_path=None):
        """
        Interpolate between two parameter sets
        
        Args:
            params1 (dict): First parameter set
            params2 (dict): Second parameter set
            t (float): Interpolation factor (0 to 1)
            output_path (str, optional): Path to save interpolated mesh
        """
        interpolated_params = {}
        for param in self.PRIMARY_PARAMS:
            if param in params1 and param in params2:
                interpolated_params[param] = (1-t) * params1[param] + t * params2[param]
        
        return self.generate_spine(interpolated_params, output_path)

def main():
    # Paths
    model_path = "trained_models/20241107_160609/spine_model.joblib"
    stl_dir = "stl/"
    
    # Initialize generator
    generator = SpineGenerator(model_path, stl_dir)
    
    # Generate multiple spines with different parameters
    test_params_list = [
        {
            'PI': 45.0, 'PT': 20.0, 'LL': -40.0,
            'GT': 30.0, 'LDI': 100.0, 'TPA': 25.0
        },
        {
            'PI': 55.0, 'PT': 25.0, 'LL': -45.0,
            'GT': 35.0, 'LDI': 110.0, 'TPA': 30.0
        },
        {
            'PI': 35.0, 'PT': 15.0, 'LL': -35.0,
            'GT': 25.0, 'LDI': 90.0, 'TPA': 20.0
        }
    ]
    
    # Generate and validate multiple different spines
    for i, params in enumerate(test_params_list):
        print(f"\nGenerating spine {i+1}...")
        mesh, _ = generator.generate_spine(
            params,
            output_path=f"generated_spine_{i+1}.stl",
            validate=True
        )

if __name__ == "__main__":
    main()