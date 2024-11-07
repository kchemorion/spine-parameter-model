import streamlit as st
import numpy as np
import trimesh
import joblib
import os
import pyvista as pv
from datetime import datetime
from stpyvista import stpyvista

class SpineVisualizer:
    def __init__(self, model_path, reference_stl_path):
        """Initialize the spine visualizer"""
        # Load model components
        self.model = joblib.load(model_path)
        self.pca_model = self.model['pca_model']
        self.scaler = self.model['scaler']
        self.param_to_pca = self.model['param_to_pca']
        self.param_stats = self.model['param_stats']
        
        # Load reference topology
        ref_mesh = trimesh.load(reference_stl_path)
        self.faces = ref_mesh.faces
        self.n_vertices = len(ref_mesh.vertices)
        
        # Parameter definitions
        self.ALL_PARAMS = ['PI', 'PT', 'SS', 'LL', 'GT', 'LDI', 'TPA', 
                          'cobb_angle', 'LL-PI', 'RPV', 'RLL', 'RSA', 'GAP']

    def generate_mesh(self, parameters):
        """Generate mesh from parameters"""
        param_vector = np.array([[
            parameters.get(param, self.param_stats[param]['mean']) 
            for param in self.ALL_PARAMS
        ]])
        
        param_vector_scaled = self.scaler.transform(param_vector)
        pca_coords = self.param_to_pca.predict(param_vector_scaled)
        new_vertices = self.pca_model.inverse_transform(pca_coords)
        vertices = new_vertices.reshape((1, self.n_vertices, 3))[0]
        
        return vertices, self.faces

    def convert_to_pyvista(self, vertices, faces):
        """Convert to PyVista mesh"""
        faces_with_count = np.column_stack((np.full(len(faces), 3), faces))
        faces_with_count = faces_with_count.flatten()
        mesh = pv.PolyData(vertices, faces_with_count)
        return mesh

def get_parameter_ranges():
    """Get parameter ranges with defaults"""
    return {
        'PI': (30.0, 70.0, 45.0),
        'PT': (5.0, 35.0, 20.0),
        'LL': (-60.0, -20.0, -40.0),
        'GT': (20.0, 40.0, 30.0),
        'LDI': (80.0, 120.0, 100.0),
        'TPA': (15.0, 35.0, 25.0),
        'cobb_angle': (0.0, 30.0, 15.0)
    }

def setup_plotter(view_angle='front'):
    """Setup plotter with fixed view"""
    plotter = pv.Plotter(notebook=True, window_size=[800, 600])
    plotter.set_background('white')
    
    # Define camera positions
    camera_positions = {
        'front': [(0, -1, 0), (0, 0, 0), (0, 0, 1)],
        'side': [(1, 0, 0), (0, 0, 0), (0, 0, 1)],
        'top': [(0, 0, 1), (0, 0, 0), (0, 1, 0)]
    }
    
    # Set camera position
    plotter.camera_position = camera_positions.get(view_angle, camera_positions['front'])
    plotter.camera.zoom(1.2)
    
    return plotter

def main():
    st.set_page_config(layout="wide")
    st.title("Interactive Parameter Model")
    
    # Initialize session state for view angle
    if 'view_angle' not in st.session_state:
        st.session_state.view_angle = 'front'
    
    # Initialize visualizer
    model_path = "trained_models/20241107_160609/spine_model.joblib"
    reference_path = "stl/1.stl"
    
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = SpineVisualizer(model_path, reference_path)
    
    # Layout
    col1, col2 = st.columns([1, 2])
    
    # Parameters panel
    with col1:
        st.header("Parameters")
        
        # Initialize parameters
        parameters = {}
        ranges = get_parameter_ranges()
        
        # Create sliders
        for param, (min_val, max_val, default) in ranges.items():
            # Use session state for initial values
            if f"param_{param}" not in st.session_state:
                st.session_state[f"param_{param}"] = default
                
            value = st.slider(
                param,
                min_value=min_val,
                max_value=max_val,
                value=st.session_state[f"param_{param}"],
                step=0.1,
                key=f"slider_{param}"
            )
            parameters[param] = value
        
        # Reset button - using session state correctly
        if st.button("Reset Parameters"):
            for param, (_, _, default) in ranges.items():
                st.session_state[f"param_{param}"] = default
            st.experimental_rerun()
        
        # View selector
        st.session_state.view_angle = st.radio(
            "View Angle",
            ["front", "side", "top"],
            horizontal=True
        )
    
    # Visualization panel
    with col2:
        st.header("Spine Model")
        
        try:
            # Generate mesh
            vertices, faces = st.session_state.visualizer.generate_mesh(parameters)
            pv_mesh = st.session_state.visualizer.convert_to_pyvista(vertices, faces)
            
            # Setup plotter with current view
            plotter = setup_plotter(st.session_state.view_angle)
            
            # Add mesh
            plotter.add_mesh(
                pv_mesh,
                color='lightblue',
                show_edges=True,
                edge_color='black',
                opacity=0.8,
                smooth_shading=True
            )
            
            # Display
            stpyvista(plotter)
            
            # Export option
            if st.button("Export Model"):
                export_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"exported_spines/spine_{timestamp}.stl"
                os.makedirs("exported_spines", exist_ok=True)
                export_mesh.export(export_path)
                st.success(f"Model exported to {export_path}")
                
        except Exception as e:
            st.error(f"Error in visualization: {str(e)}")
    
    # Parameter display
    with st.expander("Current Parameters"):
        st.json(parameters)

if __name__ == "__main__":
    # Fix for inotify watch limit
    import sys
    if sys.platform.startswith('linux'):
        import resource
        resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
    
    main()