import gradio as gr

import chatbot.app
import pictionary.app

gr.TabbedInterface(
    interface_list=[
        chatbot.app.interface,
        pictionary.app.interface,
    ],
    tab_names=['chatbot', 'pictionary'],
).launch(debug=True)
