"""
Web UI for Face Detection using Gradio
Upload an image and get detected faces cropped to 128x128 with landmarks
"""

from typing import List, Optional, Tuple, Union
import os

import gradio as gr
from PIL import Image
import numpy as np

from .detector import get_detector
from .landmark_model import get_landmark_predictor, LandmarkPredictor


# Path to checkpoint (relative to face_detection folder)
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'best_hourglass_model.pth')


def process_image(image: Optional[Union[np.ndarray, Image.Image]]) -> Tuple[
    Optional[np.ndarray], 
    List[Tuple[np.ndarray, str]], 
    List[Tuple[np.ndarray, str]],
    List[Tuple[np.ndarray, str]], 
    str
]:
    """
    Process uploaded image: detect faces, crop them, predict landmarks, and align faces.
    
    Args:
        image: Input image (numpy array from Gradio or PIL Image)
        
    Returns:
        tuple of (annotated_image, cropped_faces_gallery, faces_with_landmarks_gallery, aligned_faces_gallery, status_message)
    """
    if image is None:
        return None, [], [], [], "Upload an image to detect faces."
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Get detector and process
    detector = get_detector()
    annotated_image, cropped_faces = detector.detect_and_draw(pil_image)
    
    # Prepare gallery items for cropped faces
    cropped_gallery = []
    for i, face in enumerate(cropped_faces):
        face_array = np.array(face)
        cropped_gallery.append((face_array, f"Face {i+1} (128x128)"))
    
    # Prepare gallery items for faces with landmarks and aligned faces
    landmarks_gallery = []
    aligned_gallery = []
    
    # Check if checkpoint exists
    if os.path.exists(CHECKPOINT_PATH):
        try:
            landmark_predictor = get_landmark_predictor(CHECKPOINT_PATH)
            
            for i, face in enumerate(cropped_faces):
                # Predict landmarks and align face
                aligned_face, original_keypoints, aligned_keypoints = landmark_predictor.align_and_predict(face)
                
                # Draw landmarks on original face
                face_with_landmarks = landmark_predictor.draw_landmarks(face, original_keypoints)
                
                # Draw landmarks on aligned face
                aligned_with_landmarks = landmark_predictor.draw_landmarks(aligned_face, aligned_keypoints)
                
                # Convert to numpy for gallery
                face_landmarks_array = np.array(face_with_landmarks)
                aligned_landmarks_array = np.array(aligned_with_landmarks)
                
                # Create labels
                landmarks_gallery.append((face_landmarks_array, f"Face {i+1} with landmarks"))
                aligned_gallery.append((aligned_landmarks_array, f"Face {i+1} aligned"))
                
        except Exception as e:
            print(f"Error predicting landmarks: {e}")
            import traceback
            traceback.print_exc()
            # If landmark prediction fails, just show faces without landmarks
            for i, face in enumerate(cropped_faces):
                face_array = np.array(face)
                landmarks_gallery.append((face_array, f"Face {i+1} (no landmarks)"))
                aligned_gallery.append((face_array, f"Face {i+1} (alignment failed)"))
    else:
        # No checkpoint - show message
        for i, face in enumerate(cropped_faces):
            face_array = np.array(face)
            landmarks_gallery.append((face_array, f"Face {i+1} (checkpoint not found)"))
            aligned_gallery.append((face_array, f"Face {i+1} (no checkpoint)"))
    
    # Convert annotated image to numpy
    annotated_array = np.array(annotated_image)
    
    # Create status message
    num_faces = len(cropped_faces)
    if num_faces == 0:
        status = "No faces detected in the image."
    elif num_faces == 1:
        status = "1 face detected and cropped to 128x128."
    else:
        status = f"{num_faces} faces detected and cropped to 128x128."
    
    if os.path.exists(CHECKPOINT_PATH):
        status += " Landmarks predicted. Faces aligned."
    else:
        status += f" Warning: Checkpoint not found at {CHECKPOINT_PATH}"
    
    return annotated_array, cropped_gallery, landmarks_gallery, aligned_gallery, status


def create_demo() -> gr.Blocks:
    """Create and configure the Gradio demo interface."""
    
    with gr.Blocks(
        title="Face Detection - Crop Faces to 128x128",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown(
            """
            # Face Detection, Landmark Prediction & Alignment
            
            Upload an image to detect faces. Each detected face will be:
            1. Cropped and resized to **128x128** pixels
            2. Annotated with **5 facial landmarks** (eyes, nose, mouth corners)
            3. **Aligned** using affine transformation based on eye positions
            
            **Face Detector:** MTCNN  
            **Landmark Model:** Stacked Hourglass Network  
            **Alignment:** Affine transformation (rotation, scale, translation)
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy"
                )
                
                detect_btn = gr.Button(
                    "Detect Faces & Landmarks",
                    variant="primary"
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    placeholder="Upload an image and click 'Detect Faces & Landmarks'"
                )
            
            with gr.Column(scale=1):
                # Output section - annotated image
                output_image = gr.Image(
                    label="Detected Faces (with bounding boxes)",
                    type="numpy"
                )
        
        gr.Markdown("### Cropped Faces (128x128)")
        
        # Gallery for cropped faces
        face_gallery = gr.Gallery(
            label="Cropped Faces",
            columns=4,
            height=250,
            object_fit="contain",
            show_label=False
        )
        
        gr.Markdown("### Faces with Landmarks")
        gr.Markdown("*Landmarks: Left Eye (red), Right Eye (blue), Nose (green), Left Mouth (yellow), Right Mouth (magenta)*")
        
        # Gallery for faces with landmarks
        landmarks_gallery = gr.Gallery(
            label="Faces with Landmarks",
            columns=4,
            height=250,
            object_fit="contain",
            show_label=False
        )
        
        gr.Markdown("### Aligned Faces with Landmarks")
        gr.Markdown("*Faces aligned using affine transformation based on eye positions*")
        
        # Gallery for aligned faces with landmarks
        aligned_gallery = gr.Gallery(
            label="Aligned Faces with Landmarks",
            columns=4,
            height=250,
            object_fit="contain",
            show_label=False
        )
        
        # Connect button to processing function
        detect_btn.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[output_image, face_gallery, landmarks_gallery, aligned_gallery, status_text]
        )
        
        # Also trigger on image upload
        input_image.change(
            fn=process_image,
            inputs=[input_image],
            outputs=[output_image, face_gallery, landmarks_gallery, aligned_gallery, status_text]
        )
        
        gr.Markdown(
            """
            ---
            **Notes:**
            - Faces are detected using MTCNN with confidence threshold of 0.9
            - Each face is cropped with a margin and resized to exactly 128x128 pixels
            - Landmarks are predicted using a Stacked Hourglass Network trained on CelebA
            - Face alignment uses affine transformation to normalize eye positions
            - Aligned faces have eyes at standardized positions (38, 48) and (90, 48)
            - You can download individual images by right-clicking on them
            """
        )
    
    return demo


def main():
    """Entry point for the application."""
    print("Starting Face Detection Web UI...")
    print("Loading MTCNN model...")
    
    # Pre-load the detector
    _ = get_detector()
    
    # Pre-load landmark predictor if checkpoint exists
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading landmark model from {CHECKPOINT_PATH}...")
        _ = get_landmark_predictor(CHECKPOINT_PATH)
    else:
        print(f"Warning: Landmark checkpoint not found at {CHECKPOINT_PATH}")
        print("Landmarks will not be predicted.")
    
    # Create and launch the demo
    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
