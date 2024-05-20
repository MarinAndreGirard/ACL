from IPython.display import HTML
from PIL import Image

def display_gif(gif_path):
    # Create HTML code to display the GIF
    html_code = f'<img src="{gif_path}">'
    # Display the GIF
    display(HTML(html_code))
