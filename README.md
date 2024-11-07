# Parametric Spine Model Viewer

A machine learning-powered interactive visualization tool for generating and manipulating 3D spine models based on clinical parameters. This project combines statistical shape modeling with an interactive web interface to allow real-time manipulation of spine morphology.

## 🌟 Features

- Real-time 3D spine visualization
- Interactive parameter adjustment via sliders
- Multiple viewing angles (Front, Side, Top)
- Model export functionality
- Parameter validation
- Consistent camera positioning
- STL file generation

## 📊 Parameters

The model uses the following clinical parameters:
- **PI (Pelvic Incidence)**: 30-70°
- **PT (Pelvic Tilt)**: 5-35°
- **SS (Sacral Slope)**: Derived from PI - PT
- **LL (Lumbar Lordosis)**: -60 to -20°
- **GT (Global Tilt)**: 20-40°
- **LDI (Lumbar Distribution Index)**: 80-120
- **TPA (T1 Pelvic Angle)**: 15-35°
- **Cobb Angle**: 0-30°

Additional derived parameters:
- LL-PI
- RPV (Relative Pelvic Version)
- RLL (Relative Lumbar Lordosis)
- RSA (Relative Spinopelvic Alignment)
- GAP (Global Alignment and Proportion)

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **3D Visualization**: PyVista, VTK
- **Machine Learning**: scikit-learn (PCA, Ridge Regression)
- **3D Processing**: Trimesh
- **Data Management**: NumPy, Pandas
- **Model Persistence**: Joblib

## 🚀 Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd parametric-spine-model
```

2. Create and activate virtual environment:
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
# or
.\env\Scripts\activate  # Windows
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. For Linux users, increase inotify watch limit:
```bash
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## 📦 Project Structure

```
parametric-spine-model/
├── trained_models/           # Trained statistical shape models
│   └── spine_model.joblib
├── stl/                     # Reference mesh files
│   └── 1.stl               # Reference topology
├── exported_spines/         # Generated models
├── requirements.txt         # Project dependencies
├── spine_app.py            # Main Streamlit application
└── README.md               # This file
```

## 🎮 Usage

1. Start the application:
```bash
streamlit run spine_app.py
```

2. Access the web interface at `http://localhost:8501`

3. Use the sliders to adjust spine parameters

4. Change view angles using the radio buttons

5. Export models using the "Export Model" button

## 🔬 Model Architecture

The system uses a two-stage approach:
1. **Statistical Shape Model**: PCA-based model capturing spine shape variations
2. **Parameter Mapping**: Ridge regression mapping between clinical parameters and PCA space

## 🎯 Features in Development

- [ ] Parameter presets for common spine configurations
- [ ] Advanced measurement tools
- [ ] Cross-sectional visualization
- [ ] Parameter relationship visualization
- [ ] Animation between states
- [ ] Additional view angles
- [ ] Export of parameter reports

## 📄 License

[Your chosen license]

## 🙏 Acknowledgments

Based on research and development from:
- [Your Institution/Lab]
- Statistical Shape Model methodology from [Reference]
- Clinical parameter definitions from [Reference]

## 📧 Contact

[Your Contact Information]

## 📚 Citation

If you use this software in your research, please cite:

