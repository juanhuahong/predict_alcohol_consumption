from lib.LR import LinearRegression
from lib.SVM import SVMPredict, SVMTrain
import pandas as pd
import streamlit as st

TRAIN_DATA = "data/combined_student_data.xlsx"
SVM_TRAIN_DATA = "data/SVMdata.xlsx"
WALC = "週末飲酒量"
DALC = "平日飲酒量"


def LR():
    tab1, tab2 = st.tabs([WALC, DALC])

    with tab1:
        target = "Walc"
        with st.form(key=target):
            nursery = st.selectbox("是否上過幼兒園", ["no", "yes"])
            Fjob = st.selectbox(
                "父親的職業", ["teacher", "health", "services", "at_home", "other"]
            )
            absences = st.number_input("缺席次數", min_value=0, max_value=100, value=0)
            guardian = st.selectbox("監護人", ["mother", "father", "other"])
            Medu = st.selectbox("母親的教育程度", [0, 1, 2, 3, 4])
            Fedu = st.selectbox("父親的教育程度", [0, 1, 2, 3, 4])
            studytime = st.selectbox("每週學習時間", [1, 2, 3, 4])
            health = st.selectbox("健康狀況", [1, 2, 3, 4, 5])
            sex = st.selectbox("性別", ["F", "M"])
            famrel = st.selectbox("家庭關係", [1, 2, 3, 4, 5])
            goout = st.selectbox("每週出去玩的時間", [1, 2, 3, 4, 5])
            Dalc = st.selectbox("平日飲酒量", [1, 2, 3, 4, 5])

            submit_button = st.form_submit_button(label="提交")

        if submit_button:
            inputs = {
                "nursery[T.yes]": 1 if nursery == "yes" else 0,
                "Fjob[T.other]": 1 if Fjob == "other" else 0,
                "absences": absences,
                "Fjob[T.services]": 1 if Fjob == "services" else 0,
                "guardian[T.other]": 1 if guardian == "other" else 0,
                "Medu": Medu,
                "Fedu": Fedu,
                "studytime": studytime,
                "health": health,
                "sex[T.M]": 1 if sex == "M" else 0,
                "famrel": famrel,
                "goout": goout,
                "Dalc": Dalc,
            }

            df = pd.read_excel(TRAIN_DATA)
            lr = LinearRegression(df, target)
            prediction = lr.predict(inputs)
            show_result(prediction, target, [lambda: st.code(lr.model.summary())])

        with tab2:
            target = "Dalc"
            with st.form(key=target):
                Fedu = st.selectbox("父親的教育程度", [0, 1, 2, 3, 4])
                freetime = st.selectbox("空閒時間", [1, 2, 3, 4, 5])
                Mjob = st.selectbox(
                    "母親的職業", ["teacher", "health", "services", "at_home", "other"]
                )
                Medu = st.selectbox("母親的教育程度", [0, 1, 2, 3, 4])
                reason = st.selectbox(
                    "選擇學校的理由", ["home", "reputation", "course", "other"]
                )
                sex = st.selectbox("性別", ["F", "M"])
                Walc = st.selectbox("週末飲酒量", [1, 2, 3, 4, 5])

                submit_button = st.form_submit_button(label="提交")

            if submit_button:
                inputs = {
                    "Fedu": Fedu,
                    "freetime": freetime,
                    "Mjob[T.health]": 1 if Mjob == "health" else 0,
                    "Medu": Medu,
                    "reason[T.other]": 1 if reason == "other" else 0,
                    "sex[T.M]": 1 if sex == "M" else 0,
                    "Walc": Walc,
                }

                df = pd.read_excel(TRAIN_DATA)
                lr = LinearRegression(df, target)
                prediction = lr.predict(inputs)
                show_result(prediction, target, [lambda: st.code(lr.model.summary())])


def SVM():
    option = st.selectbox("預測：", options=[WALC, DALC])
    target = "Walc" if option == WALC else "Dalc"

    with st.form(key=target):
        sex_options = {"F": 0, "M": 1}
        guardian_options = {"mother": 0, "father": 1, "other": 2}
        Fjob_options = {"teacher": 0, "health": 1, "services": 2, "at_home": 3, "other": 4}

        sex = st.selectbox("性別", options=sex_options.keys())
        guardian = st.selectbox("監護人", options=guardian_options.keys())
        Fjob = st.selectbox("父親的職業", options=Fjob_options.keys())
        Fedu = st.selectbox("父親的教育程度", [0, 1, 2, 3, 4])
        Medu = st.selectbox("母親的教育程度", [0, 1, 2, 3, 4])
        studytime = st.selectbox("每週學習時間", [1, 2, 3, 4])
        famrel = st.selectbox("家庭關係", [1, 2, 3, 4, 5])
        goout = st.selectbox("每週出去玩的時間", [1, 2, 3, 4, 5])
        health = st.selectbox("健康狀況", [1, 2, 3, 4, 5])
        absences = st.number_input("缺席次數", min_value=0, max_value=100, value=0)

        submit_button = st.form_submit_button(label="提交")

    if submit_button:
        inputs = {
            "sex": [sex_options[sex]],
            "guardian": [guardian_options[guardian]],
            "Fjob": [Fjob_options[Fjob]],
            "Fedu": [Fedu],
            "Medu": [Medu],
            "studytime": [studytime],
            "famrel": [famrel],
            "goout": [goout],
            "health": [health],
            "absences": [absences],
        }

        # @st.cache_resource
        def train():
            svm_train = SVMTrain(pd.read_excel(SVM_TRAIN_DATA), target)
            return svm_train, *svm_train.run()

        svm_train, r, img = train()
        show_obj = [lambda: st.write(r), lambda: st.pyplot(img)]

        using = SVMPredict(svm_train, inputs)
        predictions = using.predict_new_data()[0]
        show_result(predictions, target, show_obj)


def RF(): ...


def KNN(): ...


def show_result(prediction: int | float, target: str, show_obj=None):
    st.write(f"根據您的回答，預測的分析結果如下：")
    container = st.container(border=True)
    c1, c2 = container.columns(2)
    alc_name = WALC if target == "Walc" else DALC
    c1.metric(alc_name, f"{prediction:.2f}")

    if show_obj:
        with st.expander("訓練結果"):
            for i in show_obj:
                i()

    with st.expander("數據集總覽"):
        st.image("data/heatmap.png")
        st.image(f"data/boxplots_vs_{target}.png")


algorithm = {"線性回歸": LR, "SVM": SVM, "隨機森林": RF, "KNN": KNN}

st.title("🍺 飲酒量預測")
st.write("請回答以下問題來預測您的飲酒量。")
option = st.selectbox("演算法：", algorithm)

algorithm[option]()
