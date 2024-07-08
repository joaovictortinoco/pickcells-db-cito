import datetime
from io import StringIO
import random

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from eval import predict_image
import asyncio
from pathlib import Path
from PIL import Image

# Show app title and description.
st.set_page_config(page_title="PickCells", page_icon=":material/network_intelligence_update:")
st.title("PickCells")
st.write(
    """
    Prova de conceito implementada junto com a DB Diagnósticos.
    """
)
st.info(
    "Você poderá submeter as imagens de uma lâmina e iremos analisá-las.",
    icon=":material/biotech:",
)

st.header("Upload de lâminas")

with st.form("add_images_cito"):
    uploaded_files = st.file_uploader("Enviar Arquivo(s)", accept_multiple_files=True)
    submitted = st.form_submit_button("Analisar")

if submitted:
    progress_bar = st.progress(0, text="Analisando... Por favor aguarde.")
    for index, uploaded_file in enumerate(uploaded_files):
        # bytes_data = uploaded_files.read()
        # Save uploaded file to 'F:/tmp' folder.
        save_folder = './model/test/test'
        save_path = Path(save_folder, uploaded_file.name)
        with open(save_path, mode='wb') as w:
            w.write(uploaded_file.getvalue())

        if save_path.exists():
            st.success(f'Analisando arquivo {uploaded_file.name}:')
            asyncio.run(predict_image())
            
            # st.write("filename:", uploaded_file.name)
            # st.write(bytes_data)
        
        # To read file as string:
        # string_data = stringio.read()
        # st.write(string_data)

        # Show a little success message.
        # st.write(f"Imagens submetidas! Aguarde retorno da IA.")
        file_name = uploaded_file.name
        file_name = file_name.split('.jpg')[0] 
        image = Image.open(f'./runs/classification/predict/{file_name}_pred.jpg')
        st.image(image, caption=f'{file_name}')
        progress_value = ((index+1)/len(uploaded_files))
        progress_bar.progress(progress_value,text=f"{round(progress_value*100, 0)}% concluído. \n Analisando amostras. Por favor aguarde.")


#displaying the image on streamlit app

# st.header("Últimas análises")
# st.write(f"Número de lâminas: 0")

# st.info(
#     "You can edit the tickets by double clicking on a cell. Note how the plots below "
#     "update automatically! You can also sort the table by clicking on the column headers.",
#     icon="✍️",
# )

# Show some metrics and charts about the ticket.
# st.header("Statistics")

# Show metrics side by side using `st.columns` and `st.metric`.
# col1, col2, col3 = st.columns(3)
# num_open_tickets = len(st.session_state.df[st.session_state.df.Status == "Open"])
# col1.metric(label="Number of open tickets", value=num_open_tickets, delta=10)
# col2.metric(label="First response time (hours)", value=5.2, delta=-1.5)
# col3.metric(label="Average resolution time (hours)", value=16, delta=2)

# Show two Altair charts using `st.altair_chart`.
# st.write("")
# st.write("##### Ticket status per month")
# status_plot = (
#     alt.Chart(edited_df)
#     .mark_bar()
#     .encode(
#         x="month(Date Submitted):O",
#         y="count():Q",
#         xOffset="Status:N",
#         color="Status:N",
#     )
#     .configure_legend(
#         orient="bottom", titleFontSize=14, labelFontSize=14, titlePadding=5
#     )
# )
# st.altair_chart(status_plot, use_container_width=True, theme="streamlit")

# st.write("##### Current ticket priorities")
# priority_plot = (
#     alt.Chart(edited_df)
#     .mark_arc()
#     .encode(theta="count():Q", color="Priority:N")
#     .properties(height=300)
#     .configure_legend(
#         orient="bottom", titleFontSize=14, labelFontSize=14, titlePadding=5
#     )
# )
# st.altair_chart(priority_plot, use_container_width=True, theme="streamlit")
