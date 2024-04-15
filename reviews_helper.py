import pandas as pd
import numpy as np
import sys
import os
import streamlit as st
import json
import pyperclip
from string import Template
import markdownify

from pydantic import BaseModel
from typing import Union, Tuple
import graphviz

start_comment_student_template = Template("""<div style="border:solid green 2px; padding: 20px">
    
<font color='green' style='font-size:18px; font-weight:bold'>Приветствую, ${student_name}!</font>

Меня зовут Владислав и я буду проводить ревью твоего проекта. Предлагаю обращаться друг к другу на «ты», если для тебя это будет комфортно. Иначе дай знать, и мы сразу перейдём на «вы».

Моя основная цель — поделиться своим опытом и помочь тебе стать отличным специалистом по Data Science. Тобой проделана огромная работа над проектом и я предлагаю сделать его еще лучше. Ниже ты найдешь мои комментарии - **пожалуйста, не перемещай, не изменяй и не удаляй их**. Увидев у тебя ошибку, я лишь укажу на ее наличие и дам тебе возможность самостоятельно найти и исправить ее. <br>
    
Мои комментарии будут в <font color='green'>зеленой</font>, <font color='orange'>жёлтой</font> или <font color='red'>красной</font> рамках:<br>
<div class="alert alert-block alert-success">
<b>Комментарий ревьювера ✅:</b> Так я выделяю верные действия, когда все сделано правильно.
</div>

<div class="alert alert-warning" role="alert">
<b>Комментарий ревьювера ⚠️: </b> Так выделены небольшие замечания или предложения по улучшению. Я надеюсь, что их ты тоже учтешь - твой проект от этого станет только лучше.
</div>

<div class="alert alert-block alert-danger">
<b>Комментарий ревьювера ⛔️:</b> Если требуются исправления. Работа не может быть принята с красными комментариями.
</div>

После получения ревью работы постарайся внести изменения в исследование в соответствии с моими комментариями. Это позволит сделать твою работу еще лучше и помни, что у нас общая цель - подготовить тебя к успешной работе Data Science специалистом!
</div>""")

start_comment_command_template = Template("""<div style="border:solid green 2px; padding: 20px">
    
<font color='green' style='font-size:18px; font-weight:bold'>Приветствую, ${student_name}!</font>

Меня зовут Владислав, я буду проводить ревью вашего проекта и моя основная цель — поделиться своим опытом и помочь вам стать отличными специалистами по Data Science. Вами проделана огромная работа над проектом и я предлагаю сделать его еще лучше. Ниже вы найдете мои комментарии - **пожалуйста, не перемещайте, не изменяйте и не удаляйте их**. Увидев у вас ошибку, я лишь укажу на ее наличие и дам вам возможность самостоятельно найти и исправить ее. <br>
    
Мои комментарии будут в <font color='green'>зеленой</font>, <font color='orange'>жёлтой</font> или <font color='red'>красной</font> рамках:<br>
<div class="alert alert-block alert-success">
<b>Комментарий ревьювера ✅:</b> Так я выделяю верные действия, когда все сделано правильно.
</div>

<div class="alert alert-warning" role="alert">
<b>Комментарий ревьювера ⚠️: </b> Так выделены небольшие замечания или предложения по улучшению. Я надеюсь, что их вы тоже учтете - вам проект от этого станет только лучше.
</div>

<div class="alert alert-block alert-danger">
<b>Комментарий ревьювера ⛔️:</b> Если требуются исправления. Работа не может быть принята с красными комментариями.
</div>

После получения ревью работы постарайтест внести изменения в исследование в соответствии с моими комментариями. Это позволит сделать вашу работу еще лучше и помните, что у нас общая цель - подготовка к успешной работе Data Science специалистом!
</div>""")

finish_comment_init = {
    "Вступление для идеального проекта": "Шикарный проект, одно удовольствие такие проверять:) Подробный анализ, вдумчивые выводы, наглядные графики — все супер! Спасибо за вложенные усилия, работа выполнена на отлично! Здорово, что удалось попробовать разные инструменты при подготовке данных и модели. Надеюсь, этот опыт был полезным!",
    "Вступление для хорошего проекта": "Поздравляю - проект выполнен и получены хорошие результаты! Основные этапы выполнены, протестированы несколько моделей. В проектах советую всегда определять baseline и двигаться по пути усложнения, чтобы понимать какие действия к какому результату приводят 😉 Проект нужно немного доработать: в тетрадке оставлены советы как все быстро и легко поправить, чтобы внести необходимые изменения и добавить классный артефакт в портфолио 😉 Уверен всё получится!",
    "Вступление для слабого проекта": "Спасибо за выполненный проект и старания! Проект немного сырой: ячейки кода выполнены непоследовательно, повторить вычисления не получается из-за блоков с ошибками. Практически отсутствует EDA и есть критические недочеты в предобработке данных, нет итоговых выводов. В тетрадке оставлены советы как все быстро и легко поправить, чтобы чуть-чуть доработать проект и добавить классный артефакт в портфолио 😉 Уверен всё получится!"
}

finish_comment_student_template = Template("""<div style="border:solid green 2px; padding: 40px">
<font color='green' style='font-size:24px; font-weight:bold'>Итоговый комментарий</font><br><br>
${finish_comment_init}<br><br>
<b>Положительные моменты проекта</b>:
    ${success_comment}
<b>Что необходимо исправить</b>:
    ${error_comment}     
<b>На что еще стоит обратить внимание</b>:
    ${attention_comment}<br>
    Остальные комментарии можно найти в проекте. Желаю удачи и побед в соревнованиях!😉
</div>""")

start_chat_gpt_template = Template("""Выполни ревью кода:
${code}

Напиши короткое описание, выдели ошибки в коде, что получилось хорошо, а на что нужно обратить внимание.""")

def session_state_init()-> bool:
    if "start_variables" not in st.session_state:
        st.session_state["start_variables"] = ["command_state", "student_name"]
    for variable in st.session_state["start_variables"]:
        if variable not in st.session_state:
            st.session_state[variable] = None
    return True

def get_student_name(tab)-> bool:
    with tab:
        st.radio(
            label = "Проект выполнялся одним студентом или в команде?",
            options = ["Проект выполнил один студент", "Проект выполнялся в команде"],
            horizontal = True,
            key = "command_state",
            index = 0
        )
        if st.session_state.command_state == "Проект выполнил один студент":
            st.session_state["student_name"] = st.text_input(label = "Введите имя студента:", value = "Студент")
        elif st.session_state.command_state == "Проект выполнялся в команде":
            st.session_state["student_name"] = st.text_input(label = "Введите название команды:", value = "Команда")
    return True if st.session_state.student_name else False

class WorkBook():
    def __init__(self)-> None:
        self.schema_df = None
        self.regular_sections = []
        self.special_sections_available = []
        return None
    
    def load_schema(self, file_path: str = os.path.dirname(os.path.realpath(__file__)) + "/template.xlsx")-> None:
        self.schema_df = pd.read_excel(file_path)        
        self.regular_sections = self.schema_df.loc[self.schema_df["is_special_block"] == 0, "Section"].unique().tolist()
        self.special_sections_available = self.schema_df.loc[self.schema_df["is_special_block"] == 1, "Section"].unique().tolist()
        return None
    
    def check_progress(self, tab_name)-> bool:
        if tab_name == "Основное":
            variable_list = st.session_state["start_variables"]
        else:
            variable_list = self.schema_df.loc[self.schema_df["Section"] == tab_name, "Key"].unique()
        for key in variable_list:
            if key not in st.session_state or st.session_state[key] == None:
                return False
        return True
    
    def get_start_comment(self)->None:
        if st.session_state["command_state"] == "Проект выполнил один студент":
            comment = start_comment_student_template.substitute(st.session_state)
        else:            
            comment = start_comment_command_template.substitute(st.session_state)
        if st.button("Копировать", key = "copy_button_get_start_comment"):
            pyperclip.copy(comment)
        st.markdown(comment, unsafe_allow_html = True)
        st.text("")
        return None
    
    def get_finish_comment(self)->None:
        global finish_comment_init
        finish_comment_init = st.radio(
            label = "Выбери вступление:",
            options = finish_comment_init.values(),
            captions = finish_comment_init.keys()
        )
        success_comment_list = []
        attention_comment_list = []
        error_comment_list = []
        for key in self.schema_df["Key"].unique():
            if key in st.session_state:
                cond = (self.schema_df["Key"] == key) & (self.schema_df["Answers"] == st.session_state[key])                
                success_comment = self.schema_df.loc[cond, "Success comment"]
                attention_comment = self.schema_df.loc[cond, "Attention comment"]
                error_comment = self.schema_df.loc[cond, "Error comment"]
                if success_comment.notna().values:
                    success_comment_list.append(str(success_comment.values[0]))
                if attention_comment.notna().values:
                    attention_comment_list.append(str(attention_comment.values[0]))
                if error_comment.notna().values:
                    error_comment_list.append(str(error_comment.values[0]))
        success_comment = "<ol>\n" + "\n".join([f"<li>{elem}</li>" for elem in success_comment_list]) + "</ol>"
        attention_comment = "<ol>\n" + "\n".join([f"<li>{elem}</li>" for elem in attention_comment_list]) + "</ol>"
        error_comment = "<ol>\n" + "\n".join([f"<li>{elem}</li>" for elem in error_comment_list]) + "</ol>"
        comment = finish_comment_student_template.substitute(
            {
                "finish_comment_init" : finish_comment_init,
                "success_comment" : success_comment,
                "attention_comment" : attention_comment,
                "error_comment" : error_comment
            }
        )        
        if st.button("Копировать", key = "copy_button_get_finish_comment"):
            pyperclip.copy(comment)
        st.markdown(comment, unsafe_allow_html = True)
        return None
    
    def get_summary(self)-> None:
        text = []
        for key in self.schema_df["Key"].unique():
            if key in st.session_state:
                summary = self.schema_df.loc[(self.schema_df["Key"] == key) & (self.schema_df["Answers"] == st.session_state[key]), "Summary"]
                if summary.notna().values:
                    text.append(str(summary.values[0]))
        text = "\n".join([f"{num + 1}. {elem};" for elem, num in zip(text, range(len(text)))])
        if st.button("Копировать", key = "copy_button_get_summary"):
            pyperclip.copy(text)
        st.markdown(text)
        st.text("")
        return None
    
    def get_code_analysis(self)-> None:
        notebook = st.file_uploader('Загрузите тетрадь студента', type='ipynb', accept_multiple_files = False)
        if notebook:
            content = json.loads(notebook.read())
            code = ""
            for cell in content["cells"]:
                if cell["cell_type"] == "code":
                    code += "".join(cell["source"])
            if st.button("Начать общение с ChatGPT", type = "primary"):
                pyperclip.copy(start_chat_gpt_template.substitute({"code" : code}))
            if st.button("Код соответствует стандартам PEP8?"):
                pyperclip.copy("Код соответствует стандартам PEP8?")
            if st.button('В предоставленном коде может быть "утечка данных" при обучении моделей?'):
                pyperclip.copy('В предоставленном коде может быть "утечка данных" при обучении моделей?')
            if st.button("Напиши советы для улучшения представленного кода?"):
                pyperclip.copy("Напиши советы для улучшения представленного кода?")
            if st.button("Проанализируй код на предмет оптимизации и улучшения производительности?"):
                pyperclip.copy("Проанализируй код на предмет оптимизации и улучшения производительности?")
        return None
    
    def render_regular_tab(self, tab, tab_name)-> None:
        question_cond = self.schema_df["Section"] == tab_name
        for question in self.schema_df.loc[question_cond, "Question"].unique():
            answer_cond = (question_cond) & (self.schema_df["Question"] == question)
            answers =  self.schema_df.loc[answer_cond, "Answers"].values
            key = self.schema_df.loc[answer_cond, "Key"].values[0]
            tab.radio(
                label = question,
                options = answers,
                index = None,
                horizontal = True,
                key = key
            )
            if self.schema_df.loc[answer_cond, "Tip"].nunique() > 0:
                tip = self.schema_df.loc[(answer_cond) & (self.schema_df["Answers"] == st.session_state[key]), "Tip"]
                if tip.notna().values:
                    if tab.button("Подсказка!", key = "tip_button_" + key):
                        pyperclip.copy(tip.values[0])
                else:
                    tab.button("Подсказка!", disabled = True, key = "tip_button_" + key)
        return None
    
    def render_general_tab(self, tab)-> None:
        get_student_name(tab)
        return None
    
    def render_stack_tab(self, tab)-> None:        
        special_sections_choosen = tab.multiselect(
            "Выберите специальные разделы:",
            self.special_sections_available
        )
        for special_section in special_sections_choosen:
            self.render_regular_tab(tab, special_section)
        return None        
    
    def render_tabs(self)-> None:
        tabs_names = ["Основное"] + self.regular_sections
        tab_labels = []
        for tab_name in tabs_names:
            if self.check_progress(tab_name):
                tab_labels.append(tab_name + "✅")
            else:
                tab_labels.append(tab_name + "⛔")
        tabs_names += ["Стек"]
        tab_labels += ["Стек"]
        for tab, name in zip(st.tabs(tab_labels), tabs_names):
            if name == "Основное":
                self.render_general_tab(tab)
            elif name == "Стек":
                self.render_stack_tab(tab)
            else:
                self.render_regular_tab(tab, name)
        return None
    
    def render_expanders(self)->None:
        st.divider()
        with st.expander("Анализ кода"):
            self.get_code_analysis()
        with st.expander("Стартовый комментарий"):
            self.get_start_comment()
        with st.expander("Итоговый комментарий"):
            self.get_finish_comment()
        with st.expander("SUMMARY"):
            self.get_summary()

def main()-> None:
    st.title("Let's make _review_ :green[easier] :sunglasses:")
    session_state_init()
    wb = WorkBook()
    wb.load_schema()
    wb.render_tabs()
    wb.render_expanders()
    return None

if __name__ == "__main__":
    main()