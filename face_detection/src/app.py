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

    config = load_config()

    _processor = FaceProcessor(
        landmark_checkpoint_path=config.get_landmark_model_path(),
        recognition_checkpoint_path=config.get_face_recognition_model_path(),
        cache_path=config.get_embeddings_cache_path(),
        images_directory=config.get_images_directory(),
        device=config.device,
        auto_save=config.auto_save
    )

    if _processor._database is not None:
        cache_path = config.get_embeddings_cache_path()

        if cache_path and os.path.exists(cache_path):
            try:
                _processor._database.load_cache(cache_path)

                return _processor
            except Exception as e:
                yield f"Ошибка загрузки кеша: {e}"

        if _processor._database.needs_rebuild() and _processor.has_recognition:
            images = _processor._database.get_image_files()
            total = len(images)
            
            if total > 0:
                yield f"Сборка базы данных из {total} изображений..."

                _processor._database.set_embedding_function(_processor._compute_embedding_from_path)
                
                processed = 0
                for i, image_path in enumerate(images):
                    try:
                        embedding = _processor._compute_embedding_from_path(image_path)
                        _processor._database.add_face(image_path, embedding, auto_save=False)
                        processed += 1
                        if (i + 1) % 10 == 0 or i == total - 1:
                            yield f"Обработано изображений: {i + 1}/{total} ({processed} успешно)"
                    except Exception as e:
                        yield f"Ошибка обработки {Path(image_path).name}: {e}"

                if cache_path:
                    _processor._database.save_cache()
                    yield f"Сохранено {processed} эмбеддингов в кеш"
                
                yield f"База данных готова: {processed} лиц"
            else:
                yield "Изображения не найдены в директории"
    
    yield f"Готово! База данных: {_processor.database_size if _processor._database else 0} лиц"

    return _processor


def process_image(image: Optional[Union[np.ndarray, Image.Image]]) -> Tuple[
    Optional[np.ndarray], 
    List[Tuple[np.ndarray, str]], 
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
    aligned_gallery_with_landmarks = []
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
            aligned_gallery.append((
                np.array(face_result.aligned_face),
                f"Face {i+1} aligned"
            ))
            aligned_with_landmarks = processor.draw_landmarks(
                face_result.aligned_face,
                face_result.aligned_landmarks
            )
            aligned_gallery_with_landmarks.append((
                np.array(aligned_with_landmarks),
                f"Face {i+1} aligned with landmarks"
            ))
        else:
            aligned_gallery.append((face_array, f"Face {i+1} (not aligned)"))
            aligned_gallery_with_landmarks.append((face_array, f"Face {i+1} (not aligned)"))

    annotated_array = np.array(result.annotated_image)

    return annotated_array, cropped_gallery, landmarks_gallery, aligned_gallery_with_landmarks, aligned_gallery, result.status


def search_similar_faces(
    image: Optional[Union[np.ndarray, Image.Image]],
    top_k: int = 5,
    threshold: float = 0.0
) -> Tuple[List[Tuple[np.ndarray, str]], str]:
    if image is None:
        return [], "Загрузите изображение для поиска похожих лиц."

    processor = get_processor()

    if not processor.has_recognition:
        return [], "Модель распознавания лиц не загружена."

    if processor.database_size == 0:
        return [], "База данных пуста. Добавьте лица сначала."

    result = processor.process(image, align_faces=True)

    if result.num_faces == 0:
        return [], "На изображении не обнаружены лица."

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

def search_for_all_faces(aligned_faces, threshold):
    final_gallery = []
    status_parts = []

    for idx, face in enumerate(aligned_faces):
        gallery, status = search_similar_faces(
            face[0],
            top_k=5,
            threshold=threshold
        )

        status_parts.append(f"Face {idx + 1}: {status}")

        final_gallery.extend(gallery)

    return final_gallery, "\n".join(status_parts)

def create_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Face Detection and Recognition",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown(
            """
            # Система детекции и распознавания лиц
            
            **Возможности:** Детекция лиц, предсказание ключевых точек, выравнивание, поиск похожих лиц
            """
        )
        
        with gr.Tabs():
            with gr.TabItem("Детекция и предсказание ключевых точек"):
                gr.Markdown("### Детекция лиц, предсказание ключевых точек и выравнивание")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        detect_input = gr.Image(label="Загрузить изображение", type="numpy")
                        detect_btn = gr.Button("Обнаружить лица", variant="primary")
                        detect_status = gr.Textbox(label="Статус", interactive=False)
                    
                    with gr.Column(scale=1):
                        detect_output = gr.Image(label="Обнаруженные лица", type="numpy")
                
                gr.Markdown("### Обрезанные лица")
                cropped_gallery = gr.Gallery(
                    label="Cropped", columns=4, height=600, 
                    object_fit="contain", show_label=False
                )
                
                gr.Markdown("### Лица с ключевыми точками")
                landmarks_gallery = gr.Gallery(
                    label="Landmarks", columns=4, height=600,
                    object_fit="contain", show_label=False
                )
                
                gr.Markdown("### Выровненные лица")
                aligned_gallery_with_landmarks = gr.Gallery(
                    label="Aligned", columns=4, height=600,
                    object_fit="contain", show_label=False
                )

                aligned_gallery = gr.State([])
                
                detect_btn.click(
                    fn=process_image,
                    inputs=[detect_input],
                    outputs=[detect_output, cropped_gallery, landmarks_gallery, aligned_gallery_with_landmarks, aligned_gallery, detect_status]
                )

                with gr.Row():
                    search_btn = gr.Button("Поиск похожих лиц", variant="primary")
                    threshold_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                        label="Минимальное сходство (0 = без фильтра)"
                    )
                    search_status = gr.Textbox(label="Status", interactive=False, lines=3)
                
                results_gallery = gr.Gallery(
                    columns=5 + 1,
                    object_fit="contain",
                    height=800
                )

                search_btn.click(
                    fn=search_for_all_faces,
                    inputs=[aligned_gallery, threshold_slider],
                    outputs=[results_gallery, search_status]
                )
   
        gr.Markdown(
            """
            ---
            **Модели:** MTCNN (detection), Custom Stacked Hourglass (landmarks), Custom CNN (recognition)
            """
        )
    
    return demo


def main():
    print("Face Detection & Recognition Web UI")

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
