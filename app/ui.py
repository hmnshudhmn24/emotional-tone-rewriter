import gradio as gr
from src.inference import rewrite_tone

tones=['friendly','angry','sad','corporate','romantic','sarcastic']

def go(s,t): return rewrite_tone(s,t)

gr.Interface(fn=go, inputs=[gr.Textbox(),gr.Dropdown(tones)], outputs='text').launch()
