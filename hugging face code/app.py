import gradio as gr
import spaces
from huggingface_hub import hf_hub_download
import yolov9
from PIL import Image


def download_models(model_id):
    """Download a model from Hugging Face Hub and return the local path."""
    return hf_hub_download(repo_id="da-xiong/tku2024", filename=model_id, local_dir="./")


def predict(img_path, model_id, image_size, conf_threshold, iou_threshold):
    """
    Perform object detection using YOLOv9 model on CPU.
    """
    model_path = hf_hub_download(repo_id="da-xiong/tku2024", filename="best.pt", local_dir="./")
    model = yolov9.load(model_path, device='cpu') 
    model.conf = conf_threshold
    model.iou = iou_threshold
    results = model(img_path, size=image_size)
    output_image = Image.fromarray(results.render()[0])
    return output_image


def app():
    """Define the Gradio app."""
    with gr.Blocks() as gradio_app:
        with gr.Row():
            with gr.Column():
                img_path = gr.Image(type="filepath", label="Image")
                model_id = gr.Dropdown(
                    label="Model",
                    choices=["da-xiong/tku2024/best.pt"],  # Ensure your model filename is correct
                    value="da-xiong/tku2024/best.pt"
                )
                image_size = gr.Slider(label="Image Size", minimum=320, maximum=1280, step=32, value=640)
                conf_threshold = gr.Slider(label="Confidence Threshold", minimum=0.1, maximum=1.0, step=0.1, value=0.25)
                iou_threshold = gr.Slider(label="IoU Threshold", minimum=0.1, maximum=1.0, step=0.1, value=0.5)
                infer_button = gr.Button(value="Submit")
                    
            with gr.Column():
                output_image = gr.Image(type="numpy", label="Output")

            infer_button.click(
                fn=predict,
                inputs=[img_path, model_id, image_size, conf_threshold, iou_threshold],
                outputs=[output_image]
            )

               
        gr.Examples(
            examples=[
                ["test_images/test_images_MR.jpg", "da-xiong/tku2024/best.pt", 640, 0.25, 0.5],
                ["test_images/test_images_PR.jpg", "da-xiong/tku2024/best.pt", 640, 0.25, 0.5],
                ["test_images/test_images_TR.jpg", "da-xiong/tku2024/best.pt", 640, 0.25, 0.5],
                ["test_images/test_images_AR.jpg", "da-xiong/tku2024/best.pt", 640, 0.25, 0.5]
            ],
            fn=predict,
            inputs=[img_path, model_id, image_size, conf_threshold, iou_threshold],
            outputs=[output_image],
            cache_examples=True,            

        )
        

        gr.HTML(
            """
        <h1 style='text-align: center'>
        YOLOv9 project to TKU
        </h1>
        """)
        
    return gradio_app


gradio_app = app()
gradio_app.launch(debug=True)
