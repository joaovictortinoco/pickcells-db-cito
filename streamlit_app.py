import streamlit as st

from eval import predict_image
import asyncio
from pathlib import Path
from PIL import Image
import datetime
from utils import config

def analyze_classes_from_prediction(report, final_report: bool | None = False):
    hsil = 0
    lsil = 0
    normal = 0
    if final_report:
        for r in report:
            for classes in r:
                if classes[0] == 'HSIL':
                    hsil += classes[1]
                elif classes[0] == 'LSIL':
                    lsil += classes[1]
                else:
                    normal += classes[1]
        end_time_analysis = datetime.datetime.now()
        overall_time = int((end_time_analysis-init_time_analysis).total_seconds()/60)
        if overall_time > 0:
            final_report_container.write(f'Tempo decorrido para análise da lâmina: {int((end_time_analysis-init_time_analysis).total_seconds()/60)} minuto(s)')
        else:
            final_report_container.write(f'Tempo decorrido para análise da lâmina: {int((end_time_analysis-init_time_analysis).total_seconds())} segundos')
        final_report_container.write("Adequação amostra: Satisfatória")
        # final_report_container.write("Zona de transformação: N/A")
        final_report_container.write("Organismos: Detectáveis")
        # final_report_container.write(f"Quantidade HSIL: {hsil}")
        # final_report_container.write(f"Quantidade LSIL: {lsil}")
        # final_report_container.write(f"Quantidade NORMAL: {normal}")
        print(hsil, lsil, normal)
        if (hsil/(hsil+lsil+normal)) > 0.5:
            final_report_container.warning("Interpretação e Resultado: Positivo para malignidade HSIL")
            final_report_container.warning("Complexidade: Alta complexidade")
            final_report_container.warning("Sugestivo: GRUPO 4")
        elif lsil/(hsil+lsil+normal) > 0.5:
            final_report_container.warning("Interpretação e Resultado: Positivo para malignidade LSIL")
            final_report_container.warning("Complexidade: Baixa complexidade")
            final_report_container.warning("Sugestivo: GRUPO 3")
        else:
            final_report_container.warning("Interpretação e Resultado: NEGATIVO")
            final_report_container.warning("Complexidade: N/A")
            final_report_container.warning("Sugestivo: GRUPO 1 e 2")

    else:
        for classes in report:
                if classes[0] == 'HSIL':
                    hsil += classes[1]
                elif classes[0] == 'LSIL':
                    lsil += classes[1]
                else:
                    normal += classes[1]
        
st.set_page_config(page_title="PickCells", page_icon='icon.png', layout='wide')
st.image(image='pickcells-logo.png')
title_col1,mid, title_col2 = st.columns([1,2,35])
col1, col2 = st.columns(2, gap='large')

with col1:
    st.info(
        "Envie as imagens de uma lâmina para que a IA possa interpretar as amostras e apontar os resultados",
        icon=":material/biotech:",
    )

    st.header("Upload de lâmina")
    with st.form("add_images_cito", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Enviar imagens que representam uma lâmina a ser analisada", 
            accept_multiple_files=True, 
            type=['jpg', 'png', 'jpeg'],
        )
        submitted = st.form_submit_button("Iniciar Análise")

    final_report_container = st.container(border=True)

with col2.container(height=700, border=False):
    final_report = []
    if submitted:
        if len(uploaded_files) == 0:
            final_report_container.warning('Você precisa submeter ao menos um arquivo para continuar.')
        else:
            progress_bar = st.progress(0, text="Analisando... Por favor aguarde.")
            init_time_analysis = datetime.datetime.now()
            st.write(f'Início da análise às: {(init_time_analysis - datetime.timedelta(hours=3)).strftime("%H:%M:%S")}')
            for index, uploaded_file in enumerate(uploaded_files):
                expand = st.expander(f"Análise da amostra #{index+1}", icon=":material/network_intelligence_update:")
                with expand:
                    save_folder = f"{config.getPath()}/production"
                    save_path = Path(save_folder, uploaded_file.name)
                    with open(save_path, mode='wb') as w:
                        w.write(uploaded_file.getvalue())

                    if save_path.exists():
                        st.write(f'Análise do arquivo {uploaded_file.name}:')
                        report = asyncio.run(predict_image())
                    
                    file_name = uploaded_file.name
                    file_name = file_name.split('.jpg')[0] 
                    image = Image.open(f'{config.getPath()}/runs/classification/predict/{file_name}_pred.jpg')
                    st.image(image, caption=f'{file_name}')
                    progress_value = ((index+1)/len(uploaded_files))

                    analyze_classes_from_prediction(report)

                    final_report.append(report)

                    if round(progress_value*100, 0) >= 100 :
                        progress_bar.progress(progress_value,text=f"{int(progress_value*100)}% concluído.")
                        st.success("Análise realizada com sucesso.")
                    else:
                        progress_bar.progress(progress_value,text=f"{int(progress_value*100)}% concluído. \n Analisando amostras. Por favor aguarde.")
            analyze_classes_from_prediction(final_report, final_report=True)
