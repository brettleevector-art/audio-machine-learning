
import gradio as gr

from predict import predict_audio


demo = gr.Interface(
    fn=predict_audio,
    inputs=gr.Audio(
        type="filepath",
        sources=["upload"],
        label="Upload a sound recording",
    ),
    outputs=gr.Label(
        num_top_classes=5,
        label="Predicted sound categories",
    ),
    flagging_mode="never",
    title="WILDLIGHT | Environmental Audio Intelligence",
    description=(
        "Classifies a five-second audio window into one of "
        "the 50 ESC-50 environmental sound categories."
    ),
)


if __name__ == "__main__":
    demo.launch(share=True)
