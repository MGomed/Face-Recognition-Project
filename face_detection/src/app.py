from typing import List, Optional, Tuple, Union
import os

import gradio as gr
from PIL import Image
import numpy as np

from .processor import FaceProcessor
from .config import load_config


_processor: Optional[FaceProcessor] = None


def get_processor(config_path: Optional[str] = None) -> FaceProcessor:
    global _processor

    if _processor is None:
        _processor = FaceProcessor.from_config(config_path)

    return _processor


def process_image(image: Optional[Union[np.ndarray, Image.Image]]) -> Tuple[
    Optional[np.ndarray], 
    List[Tuple[np.ndarray, str]], 
    List[Tuple[np.ndarray, str]],
    List[Tuple[np.ndarray, str]], 
    str
]:
    if image is None:
        return None, [], [], [], "Upload an image to detect faces."

    processor = get_processor()
    result = processor.process(image, align_faces=True)
    
    cropped_gallery = []
    landmarks_gallery = []
    aligned_gallery = []
    
    for face_result in result.faces:
        i = face_result.index

        face_array = np.array(face_result.face)
        cropped_gallery.append((face_array, f"Face {i+1} (128x128)"))

        if face_result.landmarks is not None:
            face_with_landmarks = processor.draw_landmarks(
                face_result.face, 
                face_result.landmarks
            )
            landmarks_gallery.append((
                np.array(face_with_landmarks), 
                f"Face {i+1} landmarks"
            ))
        else:
            landmarks_gallery.append((face_array, f"Face {i+1} (no landmarks)"))

        if face_result.aligned_face is not None and face_result.aligned_landmarks is not None:
            aligned_with_landmarks = processor.draw_landmarks(
                face_result.aligned_face,
                face_result.aligned_landmarks
            )
            aligned_gallery.append((
                np.array(aligned_with_landmarks),
                f"Face {i+1} aligned"
            ))
        else:
            aligned_gallery.append((face_array, f"Face {i+1} (not aligned)"))

    annotated_array = np.array(result.annotated_image)
    
    return annotated_array, cropped_gallery, landmarks_gallery, aligned_gallery, result.status


def create_demo() -> gr.Blocks:
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
                output_image = gr.Image(
                    label="Detected Faces (with bounding boxes)",
                    type="numpy"
                )
        
        gr.Markdown("### Cropped Faces (128x128)")
        face_gallery = gr.Gallery(
            label="Cropped Faces",
            columns=4,
            height=250,
            object_fit="contain",
            show_label=False
        )
        
        gr.Markdown("### Faces with Landmarks")
        gr.Markdown("*Landmarks: Left Eye (red), Right Eye (blue), Nose (green), Left Mouth (yellow), Right Mouth (magenta)*")
        landmarks_gallery = gr.Gallery(
            label="Faces with Landmarks",
            columns=4,
            height=250,
            object_fit="contain",
            show_label=False
        )
        
        gr.Markdown("### Aligned Faces with Landmarks")
        gr.Markdown("*Faces aligned using affine transformation based on eye positions*")
        aligned_gallery = gr.Gallery(
            label="Aligned Faces with Landmarks",
            columns=4,
            height=250,
            object_fit="contain",
            show_label=False
        )
        
        # Connect events
        detect_btn.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[output_image, face_gallery, landmarks_gallery, aligned_gallery, status_text]
        )
        
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
            - Landmarks are predicted using a Stacked Hourglass Network
            - Face alignment normalizes eye positions to (38, 48) and (90, 48)
            """
        )
    
    return demo


def main():
    print("Starting Face Detection Web UI...")
    
    _ = get_processor()
    
    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
