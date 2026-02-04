from typing import List, Optional, Tuple, Union, Generator
import os
from pathlib import Path

import gradio as gr
from PIL import Image
import numpy as np

from .processor import FaceProcessor
from .config import load_config, get_config


_processor: Optional[FaceProcessor] = None


def get_processor(config_path: Optional[str] = None) -> FaceProcessor:
    global _processor
    if _processor is None:
        _processor = FaceProcessor.from_config(config_path)
    return _processor


def init_processor_with_progress() -> Generator[str, None, FaceProcessor]:
    global _processor
    
    if _processor is not None:
        yield "Processor already initialized"

        return _processor
    
    yield "Loading configuration..."
    config = load_config()
    
    yield "Initializing face detector..."

    _processor = FaceProcessor(
        landmark_checkpoint_path=config.get_landmark_model_path(),
        recognition_checkpoint_path=config.get_face_recognition_model_path(),
        cache_path=config.get_embeddings_cache_path(),
        images_directory=config.get_images_directory(),
        device=config.device,
        auto_save=config.auto_save,
        auto_load=False
    )

    if _processor._database is not None:
        cache_path = config.get_embeddings_cache_path()

        if cache_path and os.path.exists(cache_path):
            yield "Loading embeddings from cache..."
            try:
                _processor._database.load_cache(cache_path)
                yield f"Loaded {_processor.database_size} faces from cache"

                if _processor._database.needs_rebuild():
                    yield "Directory changed, rebuilding..."
                else:
                    yield f"Database ready: {_processor.database_size} faces"
                    return _processor
            except Exception as e:
                yield f"Cache load failed: {e}, will rebuild..."

        if _processor._database.needs_rebuild() and _processor.has_recognition:
            images = _processor._database.get_image_files()
            total = len(images)
            
            if total > 0:
                yield f"Building database from {total} images..."
                
                def progress_callback(current, total_count, path):
                    pass

                _processor._database.set_embedding_function(_processor._compute_embedding_from_path)
                
                processed = 0
                for i, image_path in enumerate(images):
                    try:
                        embedding = _processor._compute_embedding_from_path(image_path)
                        _processor._database.add_face(image_path, embedding, auto_save=False)
                        processed += 1
                        if (i + 1) % 10 == 0 or i == total - 1:
                            yield f"Processing images: {i + 1}/{total} ({processed} successful)"
                    except Exception as e:
                        yield f"Error processing {Path(image_path).name}: {e}"

                _processor._database._directory_hash = _processor._database._compute_directory_hash()
                if cache_path:
                    _processor._database.save_cache()
                    yield f"Saved {processed} embeddings to cache"
                
                yield f"Database ready: {processed} faces"
            else:
                yield "No images found in directory"
    
    yield f"Ready! Database: {_processor.database_size if _processor._database else 0} faces"

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


def search_similar_faces(
    image: Optional[Union[np.ndarray, Image.Image]],
    top_k: int = 5,
    threshold: float = 0.0
) -> Tuple[List[Tuple[np.ndarray, str]], str]:
    if image is None:
        return [], "Upload an image to search for similar faces."

    processor = get_processor()

    if not processor.has_recognition:
        return [], "Face recognition model not loaded."

    if processor.database_size == 0:
        return [], "Database is empty. Add faces first."

    result = processor.process(image, align_faces=True)

    if result.num_faces == 0:
        return [], "No faces detected in the image."

    gallery = []
    status_parts = []

    for face_result in result.faces:
        if face_result.aligned_face is None:
            continue

        embedding = processor.get_embedding(face_result.aligned_face)
        matches = processor.find_similar_faces(
            embedding,
            top_k=top_k,
            threshold=threshold if threshold > 0 else None
        )

        face_idx = face_result.index + 1
        status_parts.append(f"Face {face_idx}: {len(matches)} matches found")

        query_array = np.array(face_result.aligned_face)
        gallery.append((query_array, f"Query Face {face_idx}"))

        for j, match in enumerate(matches):
            try:
                match_img = Image.open(match.path).convert('RGB')
                match_array = np.array(match_img)
                name = Path(match.path).stem
                gallery.append((
                    match_array, 
                    f"#{j+1}: {name} ({match.similarity:.2%})"
                ))
            except Exception as e:
                gallery.append((
                    np.zeros((128, 128, 3), dtype=np.uint8),
                    f"Error: {e}"
                ))

    status = "\n".join(status_parts)

    return gallery, status


def add_face_to_db(
    image: Optional[Union[np.ndarray, Image.Image]],
    name: str
) -> str:
    if image is None:
        return "Upload an image to add to database."

    if not name or not name.strip():
        return "Please enter a name for this face."

    processor = get_processor()

    if not processor.has_recognition:
        return "Face recognition model not loaded."

    if processor._database is None:
        return "Database not initialized."

    result = processor.process(image, align_faces=True)

    if result.num_faces == 0:
        return "No faces detected in the image."

    added = 0
    name = name.strip()

    for i, face_result in enumerate(result.faces):
        if face_result.aligned_face is None:
            continue

        if result.num_faces > 1:
            face_id = f"{name}_{i+1}"
        else:
            face_id = name

        embedding = processor.get_embedding(face_result.aligned_face)

        success = processor._database.add_face(face_id, embedding, overwrite=True)
        if success:
            added += 1

    return f"Added {added} face(s) to database as '{name}'. Total: {processor.database_size} faces."


def add_folder_to_db(folder_path: str) -> Generator[str, None, None]:
    if not folder_path or not folder_path.strip():
        yield "Please enter a folder path."
        return

    folder = Path(folder_path.strip())

    if not folder.exists():
        yield f"Folder not found: {folder}"
        return

    if not folder.is_dir():
        yield f"Not a directory: {folder}"
        return

    processor = get_processor()

    if not processor.has_recognition:
        yield "Face recognition model not loaded."
        return

    if processor._database is None:
        yield "Database not initialized."
        return

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    images = []
    for ext in extensions:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))

    images = sorted(set(images))
    total = len(images)

    if total == 0:
        yield f"No images found in {folder}"
        return

    yield f"Found {total} images. Starting processing..."

    added = 0
    errors = 0

    for i, img_path in enumerate(images):
        try:
            img = Image.open(img_path).convert('RGB')

            embedding = processor.get_embedding(img)

            success = processor._database.add_face(
                str(img_path), 
                embedding, 
                overwrite=True,
                auto_save=False
            )
            
            if success:
                added += 1
            
            if (i + 1) % 5 == 0 or i == total - 1:
                yield f"Processing: {i + 1}/{total} (added: {added})"
                
        except Exception as e:
            errors += 1
            yield f"Error with {img_path.name}: {e}"

    if added > 0:
        processor._database.save_cache()
        yield f"Saved to cache."

    yield f"Done! Added {added} faces, {errors} errors. Total in DB: {processor.database_size}"


def rebuild_database() -> Generator[str, None, None]:
    processor = get_processor()

    if processor._database is None:
        yield "Database not initialized."
        return

    if not processor.has_recognition:
        yield "Face recognition model not loaded."
        return

    images_dir = processor._database.images_directory
    if not images_dir or not os.path.isdir(images_dir):
        yield f"Images directory not configured or not found: {images_dir}"
        return

    images = processor._database.get_image_files()
    total = len(images)

    if total == 0:
        yield f"No images found in {images_dir}"
        return

    yield f"Rebuilding database from {total} images..."

    processor._database.clear()

    added = 0
    errors = 0

    for i, img_path in enumerate(images):
        try:
            embedding = processor._compute_embedding_from_path(img_path)
            processor._database.add_face(img_path, embedding, auto_save=False)
            added += 1

            if (i + 1) % 10 == 0 or i == total - 1:
                yield f"Processing: {i + 1}/{total} (added: {added})"

        except Exception as e:
            errors += 1
            if errors <= 5:
                yield f"Error: {Path(img_path).name}: {e}"

    processor._database._directory_hash = processor._database._compute_directory_hash()
    processor._database.save_cache()

    yield f"Done! Rebuilt database with {added} faces ({errors} errors)."


def get_database_stats() -> str:
    processor = get_processor()

    if processor._database is None:
        return "Database not initialized"

    db = processor._database
    stats = [
        f"Total faces: {db.size}",
        f"Cache path: {db.cache_path or 'Not set'}",
        f"Images directory: {db.images_directory or 'Not set'}",
        f"Auto-save: {db.auto_save}",
    ]

    if db.images_directory and os.path.isdir(db.images_directory):
        image_count = len(db.get_image_files())
        stats.append(f"Images in directory: {image_count}")

        if db.needs_rebuild():
            stats.append("Status: Needs rebuild (directory changed)")
        else:
            stats.append("Status: Up to date")

    return "\n".join(stats)


def create_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Face Detection & Recognition",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown(
            """
            # Face Detection & Recognition System
            
            **Features:** Face detection, landmarks, alignment, similarity search, database management
            """
        )
        
        with gr.Tabs():
            # ========== TAB 1: Detection ==========
            with gr.TabItem("Detection & Landmarks"):
                gr.Markdown("### Detect faces and predict landmarks")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        detect_input = gr.Image(label="Upload Image", type="numpy")
                        detect_btn = gr.Button("Detect Faces", variant="primary")
                        detect_status = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Column(scale=1):
                        detect_output = gr.Image(label="Detected Faces", type="numpy")
                
                gr.Markdown("### Cropped Faces (128x128)")
                cropped_gallery = gr.Gallery(
                    label="Cropped", columns=4, height=200, 
                    object_fit="contain", show_label=False
                )
                
                gr.Markdown("### Faces with Landmarks")
                landmarks_gallery = gr.Gallery(
                    label="Landmarks", columns=4, height=200,
                    object_fit="contain", show_label=False
                )
                
                gr.Markdown("### Aligned Faces")
                aligned_gallery = gr.Gallery(
                    label="Aligned", columns=4, height=200,
                    object_fit="contain", show_label=False
                )
                
                detect_btn.click(
                    fn=process_image,
                    inputs=[detect_input],
                    outputs=[detect_output, cropped_gallery, landmarks_gallery, aligned_gallery, detect_status]
                )
            
            # ========== TAB 2: Search ==========
            with gr.TabItem("Search Similar Faces"):
                gr.Markdown(
                    """
                    ### Find similar faces in database
                    Upload an image with a face to find the most similar faces from the database.
                    """
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        search_input = gr.Image(label="Upload Face Image", type="numpy")
                        
                        with gr.Row():
                            top_k_slider = gr.Slider(
                                minimum=1, maximum=20, value=5, step=1,
                                label="Number of results"
                            )
                            threshold_slider = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                                label="Min similarity (0 = no filter)"
                            )
                        
                        search_btn = gr.Button("Search", variant="primary")
                        search_status = gr.Textbox(label="Status", interactive=False, lines=3)
                    
                    with gr.Column(scale=2):
                        search_gallery = gr.Gallery(
                            label="Results (Query + Matches)",
                            columns=6, height=400,
                            object_fit="contain"
                        )
                
                search_btn.click(
                    fn=search_similar_faces,
                    inputs=[search_input, top_k_slider, threshold_slider],
                    outputs=[search_gallery, search_status]
                )
            
            # ========== TAB 3: Database Management ==========
            with gr.TabItem("Database"):
                gr.Markdown("### Manage Face Database")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Database Status")
                        db_stats = gr.Textbox(
                            label="Statistics", 
                            interactive=False, 
                            lines=7,
                            value=get_database_stats
                        )
                        refresh_stats_btn = gr.Button("Refresh Stats")
                        
                        refresh_stats_btn.click(
                            fn=get_database_stats,
                            outputs=[db_stats]
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### Rebuild Database")
                        gr.Markdown("*Rebuild from configured images directory*")
                        rebuild_btn = gr.Button("Rebuild Database", variant="secondary")
                        rebuild_status = gr.Textbox(label="Progress", interactive=False, lines=5)
                        
                        rebuild_btn.click(
                            fn=rebuild_database,
                            outputs=[rebuild_status]
                        )
                
                gr.Markdown("---")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Add Single Face")
                        add_single_image = gr.Image(label="Face Image", type="numpy")
                        add_single_name = gr.Textbox(label="Name / ID", placeholder="e.g. John_Doe")
                        add_single_btn = gr.Button("Add to Database", variant="primary")
                        add_single_status = gr.Textbox(label="Status", interactive=False)
                        
                        add_single_btn.click(
                            fn=add_face_to_db,
                            inputs=[add_single_image, add_single_name],
                            outputs=[add_single_status]
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### Add Folder of Faces")
                        gr.Markdown("*Add all aligned face images from a folder*")
                        add_folder_path = gr.Textbox(
                            label="Folder Path",
                            placeholder="/path/to/aligned_faces"
                        )
                        add_folder_btn = gr.Button("Add Folder", variant="primary")
                        add_folder_status = gr.Textbox(label="Progress", interactive=False, lines=5)
                        
                        add_folder_btn.click(
                            fn=add_folder_to_db,
                            inputs=[add_folder_path],
                            outputs=[add_folder_status]
                        )
        
        gr.Markdown(
            """
            ---
            **Models:** MTCNN (detection), Stacked Hourglass (landmarks), Custom CNN (recognition)
            """
        )
    
    return demo


def main():
    """Run the web UI."""
    print("=" * 50)
    print("Face Detection & Recognition Web UI")
    print("=" * 50)
    
    # Initialize with progress (shown in console)
    print("\nInitializing...")
    for status in init_processor_with_progress():
        print(f"  {status}")
    
    print("\nStarting Gradio server...")
    
    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
