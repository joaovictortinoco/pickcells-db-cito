import streamlit as st

from eval import predict_image
import asyncio
from pathlib import Path
from PIL import Image
import datetime
from utils import config

def sum_categories(data):
    category_sums = {'HSIL': 0, 'LSIL': 0, 'NORMAL': 0}
    for entry in data:
        for category, value in entry:
            category_sums[category] += value
    
    return category_sums

def calculate_rates_opt1(hsil, lsil, normal):
    hsil_rate = hsil / (hsil+lsil+normal)
    lsil_rate = lsil / (hsil+lsil+normal)
    normal_rate = normal / (hsil + lsil + normal)
    print(f'HSIL:{hsil_rate} \n LSIL: {lsil_rate} \n NORMAL: {normal_rate}')
    if hsil_rate >= 0.40 or ((hsil_rate + normal_rate > 0.6) and hsil_rate > 0.15):
        final_report_container.warning("Interpretação e Resultado: Possível malignidade HSIL")
        final_report_container.warning("Complexidade: Alta complexidade")
        final_report_container.warning("Sugestivo: GRUPO 4")
    elif lsil_rate >= 0.35 or ((lsil_rate + normal_rate > 0.6) and lsil_rate > 0.15):
        final_report_container.warning("Interpretação e Resultado: Possível malignidade LSIL")
        final_report_container.warning("Complexidade: Baixa complexidade")
        final_report_container.warning("Sugestivo: GRUPO 3")
    elif normal_rate > 0.6:
        final_report_container.warning("Interpretação e Resultado: Possível NEGATIVO")
        final_report_container.warning("Complexidade: Baixa Complexidade")
        final_report_container.warning("Sugestivo: GRUPO 1 e 2")

def calculate_rates_opt2(hsil, lsil, normal):
    hsil_rate = hsil / (hsil+lsil)
    lsil_rate = lsil / (hsil+lsil)
    normal_rate = normal / (hsil + lsil + normal)
    print(f'HSIL:{hsil_rate} \n LSIL: {lsil_rate} \n NORMAL: {normal_rate}')

    if hsil_rate >= 0.40:
        final_report_container.warning("Interpretação e Resultado: Possível malignidade HSIL")
        final_report_container.warning("Complexidade: Alta complexidade")
        final_report_container.warning("Sugestivo: GRUPO 4")
    elif lsil_rate >= 0.35:
        final_report_container.warning("Interpretação e Resultado: Possível malignidade LSIL")
        final_report_container.warning("Complexidade: Baixa complexidade")
        final_report_container.warning("Sugestivo: GRUPO 3")
    else:
        final_report_container.warning("Interpretação e Resultado: Possível NEGATIVO")
        final_report_container.warning("Complexidade: Baixa Complexidade")
        final_report_container.warning("Sugestivo: GRUPO 1 e 2")

def calculate_rates_opt3(hsil, lsil, normal):
    hsil_rate_prevalence = hsil / (hsil+lsil)
    lsil_rate_prevalence = lsil / (hsil+lsil)
    normal_rate_prevalence = normal / (hsil + lsil + normal)
    hsil_rate_general = hsil / (hsil+lsil+normal)
    lsil_rate_general = lsil / (hsil+lsil+normal)
    
    print(f'HSIL:{hsil_rate_prevalence} \n LSIL: {lsil_rate_prevalence} \n NORMAL: {normal_rate_prevalence}')

    hsil_final_rate = hsil_rate_prevalence + hsil_rate_general / normal_rate_prevalence
    lsil_final_rate = lsil_rate_prevalence + lsil_rate_general / normal_rate_prevalence
    
    print(hsil_final_rate, lsil_final_rate)
    
    if hsil_final_rate > lsil_final_rate:
        final_report_container.warning("Interpretação e Resultado: Possível malignidade HSIL")
        final_report_container.warning("Complexidade: Alta complexidade")
        final_report_container.warning("Sugestivo: GRUPO 4")
    elif lsil_final_rate > normal_rate_prevalence:
        final_report_container.warning("Interpretação e Resultado: Possível malignidade LSIL")
        final_report_container.warning("Complexidade: Baixa complexidade")
        final_report_container.warning("Sugestivo: GRUPO 3")
    else:
        final_report_container.warning("Interpretação e Resultado: Possível NEGATIVO")
        final_report_container.warning("Complexidade: Baixa Complexidade")
        final_report_container.warning("Sugestivo: GRUPO 1 e 2")

def analyze_classes_from_prediction(report, final_report: bool | None = False):
    print('\n\n\n========================REPORT========================\n\n\n', report, '\n')
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
        final_report_container.text("=====================================================")
        calculate_rates_opt1(hsil, lsil, normal)
        final_report_container.text("=====================================================")
        calculate_rates_opt2(hsil, lsil, normal)
        final_report_container.text("=====================================================")
        calculate_rates_opt3(hsil, lsil, normal)
        final_report_container.text("=====================================================")
        
        
            
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
