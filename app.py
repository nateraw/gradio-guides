import gradio as gr

import onnx_guide.app
import chatbot.app
import pictionary.app

gr.TabbedInterface(
    interface_list=[
        chatbot.app.interface,
        pictionary.app.interface,
        onnx_guide.app.interface,
    ],
    tab_names=['chatbot', 'pictionary'],
).launch(debug=True)
