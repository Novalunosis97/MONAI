import os
import argparse
import json
import numpy as np
import torch
from monai.bundle import ConfigParser, download
from monai.transforms import LoadImage, EnsureChannelFirst, Orientation, Compose
from skimage import measure
import vtk

class DragInteractorStyle(vtk.vtkInteractorStyleTrackballActor):
    def __init__(self, parent=None):
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release_event)
        self.AddObserver("MouseMoveEvent", self.mouse_move_event)
        self.AddObserver("KeyPressEvent", self.key_press_event)
        self.dragging = False
        self.last_picked_actor = None
        self.actor_initial_positions = {}
        self.renderer = None

    def set_renderer(self, renderer):
        self.renderer = renderer

    def left_button_press_event(self, obj, event):
        click_pos = self.GetInteractor().GetEventPosition()
        picker = vtk.vtkPropPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        self.last_picked_actor = picker.GetActor()
        if self.last_picked_actor:
            self.dragging = True
        self.OnLeftButtonDown()
        return

    def left_button_release_event(self, obj, event):
        self.dragging = False
        self.last_picked_actor = None
        self.OnLeftButtonUp()
        return

    def mouse_move_event(self, obj, event):
        if self.dragging and self.last_picked_actor:
            click_pos = self.GetInteractor().GetEventPosition()
            picker = vtk.vtkPropPicker()
            picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
            new_pos = picker.GetPickPosition()
            self.last_picked_actor.SetPosition(new_pos)
            self.GetInteractor().Render()
        self.OnMouseMove()
        return

    def key_press_event(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == 'r':
            for actor, initial_pos in self.actor_initial_positions.items():
                actor.SetPosition(initial_pos)
            self.GetInteractor().Render()
        elif key == 'Up':
            camera = self.renderer.GetActiveCamera()
            camera.Elevation(10)
            self.renderer.ResetCameraClippingRange()
            self.GetInteractor().Render()
        elif key == 'Down':
            camera = self.renderer.GetActiveCamera()
            camera.Elevation(-10)
            self.renderer.ResetCameraClippingRange()
            self.GetInteractor().Render()
        elif key == 'Left':
            camera = self.renderer.GetActiveCamera()
            camera.Azimuth(-10)
            self.renderer.ResetCameraClippingRange()
            self.GetInteractor().Render()
        elif key == 'Right':
            camera = self.renderer.GetActiveCamera()
            camera.Azimuth(10)
            self.renderer.ResetCameraClippingRange()
            self.GetInteractor().Render()

def save_vtk_polydata(mesh, filename):
    # Ensure normals are properly calculated
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.ConsistencyOn()  # Ensure consistency of normals
    normals.SplittingOff()  # Disable splitting
    normals.AutoOrientNormalsOn()  # Automatically orient normals
    normals.Update()
    mesh_with_normals = normals.GetOutput()

    writer = vtk.vtkOBJWriter()
    writer.SetFileName(filename)
    writer.SetInputData(mesh_with_normals)
    writer.Write()
    
def visualize_3d_multilabel(segmentation, output_directory):
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    if segmentation.ndim == 4 and segmentation.shape[0] == 1:
        segmentation = np.squeeze(segmentation, axis=0)

    if segmentation.ndim != 3:
        raise ValueError(f'Input volume should be a 3D numpy array. Got shape: {segmentation.shape}')

    segmentation = np.asarray(segmentation, dtype=np.uint8)

    color_map = {
        0: [0, 0, 0],              # Background - Black
        1: [0.8, 0.3, 0.3],        # Spleen - Light Red
        2: [0.6, 0.2, 0.2],        # Right Kidney - Medium Red
        3: [0.6, 0.2, 0.2],        # Left Kidney - Medium Red
        4: [0.7, 0.4, 0.3],        # Gallbladder - Light Brown
        5: [0.5, 0.1, 0.1],        # Liver - Dark Red
        6: [0.8, 0.6, 0.5],        # Stomach - Light Brown
        7: [0.9, 0.1, 0.1],        # Aorta - Bright Red
        8: [0.4, 0.6, 0.8],        # Inferior Vena Cava - Light Blue
        9: [0.4, 0.6, 0.8],        # Portal Vein and Splenic Vein - Light Blue
        10: [0.5, 0.3, 0.3],       # Pancreas - Medium Brown
        11: [0.7, 0.4, 0.4],       # Adrenal Gland Right - Medium Red
        12: [0.7, 0.4, 0.4],       # Adrenal Gland Left - Medium Red
        13: [0.4, 0.3, 0.3],       # Lung Upper Lobe Left - Dark Brown
        14: [0.4, 0.3, 0.3],       # Lung Lower Lobe Left - Dark Brown
        15: [0.4, 0.3, 0.3],       # Lung Upper Lobe Right - Dark Brown
        16: [0.4, 0.3, 0.3],       # Lung Middle Lobe Right - Dark Brown
        17: [0.4, 0.3, 0.3],       # Lung Lower Lobe Right - Dark Brown
        18: [0.3, 0.3, 0.3],       # Vertebrae L5 - Dark Grey
        19: [0.3, 0.3, 0.3],       # Vertebrae L4 - Dark Grey
        20: [0.3, 0.3, 0.3],       # Vertebrae L3 - Dark Grey
        21: [0.3, 0.3, 0.3],       # Vertebrae L2 - Dark Grey
        22: [0.3, 0.3, 0.3],       # Vertebrae L1 - Dark Grey
        23: [0.3, 0.3, 0.3],       # Vertebrae T12 - Dark Grey
        24: [0.3, 0.3, 0.3],       # Vertebrae T11 - Dark Grey
        25: [0.3, 0.3, 0.3],       # Vertebrae T10 - Dark Grey
        26: [0.3, 0.3, 0.3],       # Vertebrae T9 - Dark Grey
        27: [0.3, 0.3, 0.3],       # Vertebrae T8 - Dark Grey
        28: [0.3, 0.3, 0.3],       # Vertebrae T7 - Dark Grey
        29: [0.3, 0.3, 0.3],       # Vertebrae T6 - Dark Grey
        30: [0.3, 0.3, 0.3],       # Vertebrae T5 - Dark Grey
        31: [0.3, 0.3, 0.3],       # Vertebrae T4 - Dark Grey
        32: [0.3, 0.3, 0.3],       # Vertebrae T3 - Dark Grey
        33: [0.3, 0.3, 0.3],       # Vertebrae T2 - Dark Grey
        34: [0.3, 0.3, 0.3],       # Vertebrae T1 - Dark Grey
        35: [0.3, 0.3, 0.3],       # Vertebrae C7 - Dark Grey
        36: [0.3, 0.3, 0.3],       # Vertebrae C6 - Dark Grey
        37: [0.3, 0.3, 0.3],       # Vertebrae C5 - Dark Grey
        38: [0.3, 0.3, 0.3],       # Vertebrae C4 - Dark Grey
        39: [0.3, 0.3, 0.3],       # Vertebrae C3 - Dark Grey
        40: [0.3, 0.3, 0.3],       # Vertebrae C2 - Dark Grey
        41: [0.3, 0.3, 0.3],       # Vertebrae C1 - Dark Grey
        42: [0.4, 0.3, 0.3],       # Esophagus - Dark Brown
        43: [0.5, 0.4, 0.4],       # Trachea - Brown
        44: [1, 0, 0],             # Heart Myocardium - Red
        45: [1, 0.4, 0.4],         # Heart Atrium Left - Light Red
        46: [1, 0, 0],             # Heart Ventricle Left - Red
        47: [1, 0.4, 0.4],         # Heart Atrium Right - Light Red
        48: [1, 0, 0],             # Heart Ventricle Right - Red
        49: [0, 0, 1],             # Pulmonary Artery - Blue
        50: [0.5, 0.5, 1],         # Brain - Light Blue
        51: [0.6, 0.3, 0],         # Iliac Artery Left - Orange
        52: [0.6, 0.3, 0],         # Iliac Artery Right - Orange
        53: [0.3, 0.6, 0.6],       # Iliac Vein Left - Teal
        54: [0.3, 0.6, 0.6],       # Iliac Vein Right - Teal
        55: [0.5, 0.3, 0.3],       # Small Bowel - Light Brown
        56: [0.7, 0.2, 0.2],       # Duodenum - Light Red
        57: [0.2, 0.2, 0.7],       # Colon - Dark Blue
        58: [0.3, 0.3, 0.3],       # Rib Left 1 - Dark Grey
        59: [0.3, 0.3, 0.3],       # Rib Left 2 - Dark Grey
        60: [0.3, 0.3, 0.3],       # Rib Left 3 - Dark Grey
        61: [0.3, 0.3, 0.3],       # Rib Left 4 - Dark Grey
        62: [0.3, 0.3, 0.3],       # Rib Left 5 - Dark Grey
        63: [0.3, 0.3, 0.3],       # Rib Left 6 - Dark Grey
        64: [0.3, 0.3, 0.3],       # Rib Left 7 - Dark Grey
        65: [0.3, 0.3, 0.3],       # Rib Left 8 - Dark Grey
        66: [0.3, 0.3, 0.3],       # Rib Left 9 - Dark Grey
        67: [0.3, 0.3, 0.3],       # Rib Left 10 - Dark Grey
        68: [0.3, 0.3, 0.3],       # Rib Left 11 - Dark Grey
        69: [0.3, 0.3, 0.3],       # Rib Left 12 - Dark Grey
        70: [0.3, 0.3, 0.3],       # Rib Right 1 - Dark Grey
        71: [0.3, 0.3, 0.3],       # Rib Right 2 - Dark Grey
        72: [0.3, 0.3, 0.3],       # Rib Right 3 - Dark Grey
        73: [0.3, 0.3, 0.3],       # Rib Right 4 - Dark Grey
        74: [0.3, 0.3, 0.3],       # Rib Right 5 - Dark Grey
        75: [0.3, 0.3, 0.3],       # Rib Right 6 - Dark Grey
        76: [0.3, 0.3, 0.3],       # Rib Right 7 - Dark Grey
        77: [0.3, 0.3, 0.3],       # Rib Right 8 - Dark Grey
        78: [0.3, 0.3, 0.3],       # Rib Right 9 - Dark Grey
        79: [0.3, 0.3, 0.3],       # Rib Right 10 - Dark Grey
        80: [0.3, 0.3, 0.3],       # Rib Right 11 - Dark Grey
        81: [0.3, 0.3, 0.3],       # Rib Right 12 - Dark Grey
        82: [0.5, 0.2, 0.2],       # Humerus Left - Dark Red
        83: [0.6, 0.3, 0.3],       # Humerus Right - Medium Red
        84: [0.6, 0.3, 0.3],       # Scapula Left - Medium Red
        85: [0.7, 0.4, 0.4],       # Scapula Right - Medium Red
        86: [0.7, 0.4, 0.4],       # Clavicula Left - Medium Red
        87: [0.8, 0.5, 0.5],       # Clavicula Right - Light Red
        88: [0.3, 0.2, 0.2],       # Femur Left - Dark Brown
        89: [0.4, 0.3, 0.3],       # Femur Right - Dark Brown
        90: [0.3, 0.2, 0.2],       # Hip Left - Dark Brown
        91: [0.4, 0.3, 0.3],       # Hip Right - Dark Brown
        92: [0.3, 0.2, 0.2],       # Sacrum - Dark Brown
        93: [0.3, 0.2, 0.2],       # Face - Dark Brown
        94: [0.5, 0.3, 0.3],       # Gluteus Maximus Left - Medium Brown
        95: [0.5, 0.3, 0.3],       # Gluteus Maximus Right - Medium Brown
        96: [0.5, 0.3, 0.3],       # Gluteus Medius Left - Medium Brown
        97: [0.5, 0.3, 0.3],       # Gluteus Medius Right - Medium Brown
        98: [0.5, 0.3, 0.3],       # Gluteus Minimus Left - Medium Brown
        99: [0.5, 0.3, 0.3],       # Gluteus Minimus Right - Medium Brown
        100: [0.5, 0.3, 0.3],      # Autochthon Left - Medium Brown
        101: [0.5, 0.3, 0.3],      # Autochthon Right - Medium Brown
        102: [0.4, 0.2, 0.2],      # Iliopsoas Left - Dark Brown
        103: [0.4, 0.2, 0.2],      # Iliopsoas Right - Dark Brown
        104: [0.3, 0.3, 0.8],      # Urinary Bladder - Dark Blue
    }

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetAlphaBitPlanes(1)  # Enable alpha channel
    render_window.SetMultiSamples(0)    # Disable multisampling
    render_window.SetNumberOfLayers(2)  # Use multiple layers

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor_style = DragInteractorStyle()
    interactor.SetInteractorStyle(interactor_style)
    interactor_style.set_renderer(renderer)

    for label_idx in np.unique(segmentation):
        if label_idx == 0:
            continue

        verts, faces, _, _ = measure.marching_cubes(segmentation == label_idx, level=0.5)
        mesh = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        triangles = vtk.vtkCellArray()

        for i, vert in enumerate(verts):
            points.InsertNextPoint(vert)
        for face in faces:
            triangle = vtk.vtkTriangle()
            for j in range(3):
                triangle.GetPointIds().SetId(j, face[j])
            triangles.InsertNextCell(triangle)

        mesh.SetPoints(points)
        mesh.SetPolys(triangles)

        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(mesh)
        smoother.SetNumberOfIterations(30)
        smoother.SetRelaxationFactor(0.1)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOn()
        smoother.Update()
        mesh = smoother.GetOutput()

        # Clean the mesh to remove any duplicate or coincident points
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(mesh)
        cleaner.Update()
        mesh = cleaner.GetOutput()

        # Compute normals
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(mesh)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOn()
        normals.Update()
        mesh = normals.GetOutput()

        # Add texture coordinates
        texture_map = vtk.vtkTextureMapToCylinder()
        texture_map.SetInputData(mesh)
        texture_map.PreventSeamOn()
        texture_map.Update()
        mesh = texture_map.GetOutput()

        # Save each label's mesh as an OBJ file
        output_path = os.path.join(output_directory, f'label_{label_idx}.obj')
        save_vtk_polydata(mesh, output_path)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        color = color_map.get(label_idx, [1, 1, 1])

        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(1.0)
        actor.GetProperty().SetRepresentationToSurface()

        renderer.AddActor(actor)
        interactor_style.actor_initial_positions[actor] = actor.GetPosition()

    renderer.SetBackground(0, 0, 0)  # Background color in RGB (black)
    renderer.SetBackgroundAlpha(0.0)  # Set the background alpha (transparency)

    render_window.SetSize(800, 800)

    interactor.Initialize()
    render_window.Render()
    interactor.Start()

def save_metadata(metadata, output_directory):
    metadata_path = os.path.join(output_directory, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

def main(input_directory, output_directory):
    image_loader = LoadImage(image_only=True)
    CT = image_loader(input_directory)

    preprocessing_pipeline = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes='LPS')
    ])
    CT = preprocessing_pipeline(input_directory)

    datadir = "/Users/raja/Downloads"
    model_name = "wholeBody_ct_segmentation"
    download(name=model_name, bundle_dir=datadir)
    model_path = os.path.join(datadir, 'wholeBody_ct_segmentation', 'models', 'model_lowres.pt')
    config_path = os.path.join(datadir, 'wholeBody_ct_segmentation', 'configs', 'inference.json')

    config = ConfigParser()
    config.read_config(config_path)

    preprocessing = config.get_parsed_content("preprocessing")
    data = preprocessing({'image': input_directory})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config.get_parsed_content("network")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    inferer = config.get_parsed_content("inferer")
    postprocessing = config.get_parsed_content("postprocessing")

    with torch.no_grad():
        data['pred'] = inferer(data['image'].unsqueeze(0).to(device), network=model)
    data['pred'] = data['pred'][0]
    data['image'] = data['image'][0]
    data = postprocessing(data)

    metadata = {'color_map': [
        {"label": 0, "color": [0, 0, 0]},             # Background - Black
        {"label": 1, "color": [0.8, 0.3, 0.3]},       # Spleen - Light Red
        {"label": 2, "color": [0.6, 0.2, 0.2]},       # Right Kidney - Medium Red
        {"label": 3, "color": [0.6, 0.2, 0.2]},       # Left Kidney - Medium Red
        {"label": 4, "color": [0.7, 0.4, 0.3]},       # Gallbladder - Light Brown
        {"label": 5, "color": [0.5, 0.1, 0.1]},       # Liver - Dark Red
        {"label": 6, "color": [0.8, 0.6, 0.5]},       # Stomach - Light Brown
        {"label": 7, "color": [0.9, 0.1, 0.1]},       # Aorta - Bright Red
        {"label": 8, "color": [0.4, 0.6, 0.8]},       # Inferior Vena Cava - Light Blue
        {"label": 9, "color": [0.4, 0.6, 0.8]},       # Portal Vein and Splenic Vein - Light Blue
        {"label": 10, "color": [0.5, 0.3, 0.3]},      # Pancreas - Medium Brown
        {"label": 11, "color": [0.7, 0.4, 0.4]},      # Adrenal Gland Right - Medium Red
        {"label": 12, "color": [0.7, 0.4, 0.4]},      # Adrenal Gland Left - Medium Red
        {"label": 13, "color": [0.4, 0.3, 0.3]},      # Lung Upper Lobe Left - Dark Brown
        {"label": 14, "color": [0.4, 0.3, 0.3]},      # Lung Lower Lobe Left - Dark Brown
        {"label": 15, "color": [0.4, 0.3, 0.3]},      # Lung Upper Lobe Right - Dark Brown
        {"label": 16, "color": [0.4, 0.3, 0.3]},      # Lung Middle Lobe Right - Dark Brown
        {"label": 17, "color": [0.4, 0.3, 0.3]},      # Lung Lower Lobe Right - Dark Brown
        {"label": 18, "color": [0.3, 0.3, 0.3]},      # Vertebrae L5 - Dark Grey
        {"label": 19, "color": [0.3, 0.3, 0.3]},      # Vertebrae L4 - Dark Grey
        {"label": 20, "color": [0.3, 0.3, 0.3]},      # Vertebrae L3 - Dark Grey
        {"label": 21, "color": [0.3, 0.3, 0.3]},      # Vertebrae L2 - Dark Grey
        {"label": 22, "color": [0.3, 0.3, 0.3]},      # Vertebrae L1 - Dark Grey
        {"label": 23, "color": [0.3, 0.3, 0.3]},      # Vertebrae T12 - Dark Grey
        {"label": 24, "color": [0.3, 0.3, 0.3]},      # Vertebrae T11 - Dark Grey
        {"label": 25, "color": [0.3, 0.3, 0.3]},      # Vertebrae T10 - Dark Grey
        {"label": 26, "color": [0.3, 0.3, 0.3]},      # Vertebrae T9 - Dark Grey
        {"label": 27, "color": [0.3, 0.3, 0.3]},      # Vertebrae T8 - Dark Grey
        {"label": 28, "color": [0.3, 0.3, 0.3]},      # Vertebrae T7 - Dark Grey
        {"label": 29, "color": [0.3, 0.3, 0.3]},      # Vertebrae T6 - Dark Grey
        {"label": 30, "color": [0.3, 0.3, 0.3]},      # Vertebrae T5 - Dark Grey
        {"label": 31, "color": [0.3, 0.3, 0.3]},      # Vertebrae T4 - Dark Grey
        {"label": 32, "color": [0.3, 0.3, 0.3]},      # Vertebrae T3 - Dark Grey
        {"label": 33, "color": [0.3, 0.3, 0.3]},      # Vertebrae T2 - Dark Grey
        {"label": 34, "color": [0.3, 0.3, 0.3]},      # Vertebrae T1 - Dark Grey
        {"label": 35, "color": [0.3, 0.3, 0.3]},      # Vertebrae C7 - Dark Grey
        {"label": 36, "color": [0.3, 0.3, 0.3]},      # Vertebrae C6 - Dark Grey
        {"label": 37, "color": [0.3, 0.3, 0.3]},      # Vertebrae C5 - Dark Grey
        {"label": 38, "color": [0.3, 0.3, 0.3]},      # Vertebrae C4 - Dark Grey
        {"label": 39, "color": [0.3, 0.3, 0.3]},      # Vertebrae C3 - Dark Grey
        {"label": 40, "color": [0.3, 0.3, 0.3]},      # Vertebrae C2 - Dark Grey
        {"label": 41, "color": [0.3, 0.3, 0.3]},      # Vertebrae C1 - Dark Grey
        {"label": 42, "color": [0.4, 0.3, 0.3]},      # Esophagus - Dark Brown
        {"label": 43, "color": [0.5, 0.4, 0.4]},      # Trachea - Brown
        {"label": 44, "color": [1, 0, 0]},            # Heart Myocardium - Red
        {"label": 45, "color": [1, 0.4, 0.4]},        # Heart Atrium Left - Light Red
        {"label": 46, "color": [1, 0, 0]},            # Heart Ventricle Left - Red
        {"label": 47, "color": [1, 0.4, 0.4]},        # Heart Atrium Right - Light Red
        {"label": 48, "color": [1, 0, 0]},            # Heart Ventricle Right - Red
        {"label": 49, "color": [0, 0, 1]},            # Pulmonary Artery - Blue
        {"label": 50, "color": [0.5, 0.5, 1]},        # Brain - Light Blue
        {"label": 51, "color": [0.6, 0.3, 0]},        # Iliac Artery Left - Orange
        {"label": 52, "color": [0.6, 0.3, 0]},        # Iliac Artery Right - Orange
        {"label": 53, "color": [0.3, 0.6, 0.6]},      # Iliac Vein Left - Teal
        {"label": 54, "color": [0.3, 0.6, 0.6]},      # Iliac Vein Right - Teal
        {"label": 55, "color": [0.5, 0.3, 0.3]},      # Small Bowel - Light Brown
        {"label": 56, "color": [0.7, 0.2, 0.2]},      # Duodenum - Light Red
        {"label": 57, "color": [0.2, 0.2, 0.7]},      # Colon - Dark Blue
        {"label": 58, "color": [0.3, 0.3, 0.3]},      # Rib Left 1 - Dark Grey
        {"label": 59, "color": [0.3, 0.3, 0.3]},      # Rib Left 2 - Dark Grey
        {"label": 60, "color": [0.3, 0.3, 0.3]},      # Rib Left 3 - Dark Grey
        {"label": 61, "color": [0.3, 0.3, 0.3]},      # Rib Left 4 - Dark Grey
        {"label": 62, "color": [0.3, 0.3, 0.3]},      # Rib Left 5 - Dark Grey
        {"label": 63, "color": [0.3, 0.3, 0.3]},      # Rib Left 6 - Dark Grey
        {"label": 64, "color": [0.3, 0.3, 0.3]},      # Rib Left 7 - Dark Grey
        {"label": 65, "color": [0.3, 0.3, 0.3]},      # Rib Left 8 - Dark Grey
        {"label": 66, "color": [0.3, 0.3, 0.3]},      # Rib Left 9 - Dark Grey
        {"label": 67, "color": [0.3, 0.3, 0.3]},      # Rib Left 10 - Dark Grey
        {"label": 68, "color": [0.3, 0.3, 0.3]},      # Rib Left 11 - Dark Grey
        {"label": 69, "color": [0.3, 0.3, 0.3]},      # Rib Left 12 - Dark Grey
        {"label": 70, "color": [0.3, 0.3, 0.3]},      # Rib Right 1 - Dark Grey
        {"label": 71, "color": [0.3, 0.3, 0.3]},      # Rib Right 2 - Dark Grey
        {"label": 72, "color": [0.3, 0.3, 0.3]},      # Rib Right 3 - Dark Grey
        {"label": 73, "color": [0.3, 0.3, 0.3]},      # Rib Right 4 - Dark Grey
        {"label": 74, "color": [0.3, 0.3, 0.3]},      # Rib Right 5 - Dark Grey
        {"label": 75, "color": [0.3, 0.3, 0.3]},      # Rib Right 6 - Dark Grey
        {"label": 76, "color": [0.3, 0.3, 0.3]},      # Rib Right 7 - Dark Grey
        {"label": 77, "color": [0.3, 0.3, 0.3]},      # Rib Right 8 - Dark Grey
        {"label": 78, "color": [0.3, 0.3, 0.3]},      # Rib Right 9 - Dark Grey
        {"label": 79, "color": [0.3, 0.3, 0.3]},      # Rib Right 10 - Dark Grey
        {"label": 80, "color": [0.3, 0.3, 0.3]},      # Rib Right 11 - Dark Grey
        {"label": 81, "color": [0.3, 0.3, 0.3]},      # Rib Right 12 - Dark Grey
        {"label": 82, "color": [0.5, 0.2, 0.2]},      # Humerus Left - Dark Red
        {"label": 83, "color": [0.6, 0.3, 0.3]},      # Humerus Right - Medium Red
        {"label": 84, "color": [0.6, 0.3, 0.3]},      # Scapula Left - Medium Red
        {"label": 85, "color": [0.7, 0.4, 0.4]},      # Scapula Right - Medium Red
        {"label": 86, "color": [0.7, 0.4, 0.4]},      # Clavicula Left - Medium Red
        {"label": 87, "color": [0.8, 0.5, 0.5]},      # Clavicula Right - Light Red
        {"label": 88, "color": [0.3, 0.2, 0.2]},      # Femur Left - Dark Brown
        {"label": 89, "color": [0.4, 0.3, 0.3]},      # Femur Right - Dark Brown
        {"label": 90, "color": [0.3, 0.2, 0.2]},      # Hip Left - Dark Brown
        {"label": 91, "color": [0.4, 0.3, 0.3]},      # Hip Right - Dark Brown
        {"label": 92, "color": [0.3, 0.2, 0.2]},      # Sacrum - Dark Brown
        {"label": 93, "color": [0.3, 0.2, 0.2]},      # Face - Dark Brown
        {"label": 94, "color": [0.5, 0.3, 0.3]},      # Gluteus Maximus Left - Medium Brown
        {"label": 95, "color": [0.5, 0.3, 0.3]},      # Gluteus Maximus Right - Medium Brown
        {"label": 96, "color": [0.5, 0.3, 0.3]},      # Gluteus Medius Left - Medium Brown
        {"label": 97, "color": [0.5, 0.3, 0.3]},      # Gluteus Medius Right - Medium Brown
        {"label": 98, "color": [0.5, 0.3, 0.3]},      # Gluteus Minimus Left - Medium Brown
        {"label": 99, "color": [0.5, 0.3, 0.3]},      # Gluteus Minimus Right - Medium Brown
        {"label": 100, "color": [0.5, 0.3, 0.3]},     # Autochthon Left - Medium Brown
        {"label": 101, "color": [0.5, 0.3, 0.3]},     # Autochthon Right - Medium Brown
        {"label": 102, "color": [0.4, 0.2, 0.2]},     # Iliopsoas Left - Dark Brown
        {"label": 103, "color": [0.4, 0.2, 0.2]},     # Iliopsoas Right - Dark Brown
        {"label": 104, "color": [0.3, 0.3, 0.8]}      # Urinary Bladder - Dark Blue
    ]}
    
    save_metadata(metadata, output_directory)

    visualize_3d_multilabel(data['pred'], output_directory)

if __name__ == "__main__":
    input_directory = '/Users/raja/Downloads/Task07_Pancreas/imagesTr/pancreas_080.nii'
    output_directory = "/Users/raja/Downloads/output_pancreas"
    
    os.makedirs(output_directory, exist_ok=True)
    main(input_directory, output_directory)