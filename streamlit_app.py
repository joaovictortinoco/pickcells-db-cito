import streamlit as st

from eval import predict_image
import asyncio
from pathlib import Path
from PIL import Image
import datetime
from utils import config
import pandas as pd

def sum_categories(data):
    category_sums = {'HSIL': 0, 'LSIL': 0, 'NORMAL': 0}
    for entry in data:
        for category, value in entry:
            category_sums[category] += value
    
    return category_sums

def input_data(hsil,lsil,normal,hsil_rate,lsil_rate,normal_rate,real_result,predicted_result,time):
    try:
        experiments = pd.read_csv('experiments.csv')
        new_row = pd.DataFrame({
                "hsil": [hsil],
                "lsil": [lsil],
                "normal": [normal],
                "hsil_rate": [hsil_rate],
                "lsil_rate": [lsil_rate],
                "normal_rate": [normal_rate],
                "real_result": [real_result],
                "predicted_result": [predicted_result],
                "time": [time]
            })
        experiments = pd.concat([experiments, new_row], ignore_index=False)
        experiments.to_csv('experiments.csv')
    except:
        new_row = pd.DataFrame({
                "hsil": [hsil],
                "lsil": [lsil],
                "normal": [normal],
                "hsil_rate": [hsil_rate],
                "lsil_rate": [lsil_rate],
                "normal_rate": [normal_rate],
                "real_result": [real_result],
                "predicted_result": [predicted_result],
                "time": [time]
            })
        new_row.to_csv('experiments.csv')

def calculate_rates(hsil, lsil, normal, real_result, time):
    hsil_rate = hsil / (hsil+lsil)
    lsil_rate = lsil / (hsil+lsil)
    normal_rate = normal / (hsil + lsil + normal)
    
    print(f'HSIL:{hsil} \n LSIL: {lsil} \n NORMAL: {normal}\n')
    print(f'HSIL_RATE:{hsil_rate} \n LSIL_RATE: {lsil_rate} \n NORMAL_RATE: {normal_rate}')

    if hsil_rate >= 0.40:
        final_report_container.warning("Interpretação e Resultado: Possível malignidade HSIL")
        final_report_container.warning("Complexidade: Alta complexidade")
        final_report_container.warning("Sugestivo: GRUPO 4")
        input_data(hsil, lsil, normal, hsil_rate, lsil_rate, normal_rate, real_result, 'HSIL', time)
    elif lsil_rate >= 0.35:
        final_report_container.warning("Interpretação e Resultado: Possível malignidade LSIL")
        final_report_container.warning("Complexidade: Baixa complexidade")
        final_report_container.warning("Sugestivo: GRUPO 3")
        input_data(hsil, lsil, normal, hsil_rate, lsil_rate, normal_rate, real_result, 'LSIL', time)
    else:
        final_report_container.warning("Interpretação e Resultado: Possível NEGATIVO")
        final_report_container.warning("Complexidade: Baixa Complexidade")
        final_report_container.warning("Sugestivo: GRUPO 1 e 2")
        input_data(hsil, lsil, normal, hsil_rate, lsil_rate, normal_rate, real_result, 'NEGATIVO', time)


def analyze_classes_from_prediction(report, real_result:str | None = None, final_report: bool | None = False):
    print('\n\n\n========================REPORT========================\n', report, '\n\n\n')
    if final_report:
        end_time_analysis = datetime.datetime.now()
        classes_amount = sum_categories(report)
        overall_time = int((end_time_analysis-init_time_analysis).total_seconds()/60)
        if overall_time > 0:
            final_report_container.write(f'Tempo decorrido para análise da lâmina: {int((end_time_analysis-init_time_analysis).total_seconds()/60)} minuto(s)')
        else:
            final_report_container.write(f'Tempo decorrido para análise da lâmina: {int((end_time_analysis-init_time_analysis).total_seconds())} segundos')
        final_report_container.write("Adequação amostra: Satisfatória")
        final_report_container.write("Organismos: Detectáveis")
        hsil = classes_amount.get('HSIL')
        lsil = classes_amount.get('LSIL')
        normal = classes_amount.get('NORMAL')
        calculate_rates(hsil, lsil, normal, real_result=real_result, time=(end_time_analysis-init_time_analysis).total_seconds())
        
            
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
            real_result = ''
            expand = st.expander('')
            progress_bar = st.progress(0, text="Analisando... Por favor aguarde.")
            init_time_analysis = datetime.datetime.now()
            st.write(f'Início da análise às: {(init_time_analysis - datetime.timedelta(hours=3)).strftime("%H:%M:%S")}')
            for index, uploaded_file in enumerate(uploaded_files):
                expand = st.expander(f"Análise da amostra #{index+1}", icon=":material/network_intelligence_update:")
                with expand:
                    save_folder = f"{config.getPath()}"
                    save_path = Path(save_folder, uploaded_file.name)
                    with open(save_path, mode='wb') as w:
                        w.write(uploaded_file.getvalue())

                    if save_path.exists():
                        st.write(f'Análise do arquivo {uploaded_file.name}:')
                        report = asyncio.run(predict_image())
                    
                    file_name = uploaded_file.name
                    if 'HSIL' in file_name: real_result = 'HSIL'
                    elif 'LSIL' in file_name: real_result = 'LSIL'
                    elif 'NEGATIVO' in file_name: real_result = 'NEGATIVO'
                    else: real_result = 'NOT_IDENTIFIED'

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
            analyze_classes_from_prediction(final_report, real_result=real_result, final_report=True)
