import streamlit as st
import requests
import pyowm
import joblib
import pandas as pd
import numpy as np
import google.generativeai as genai

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import requests
YOUTUBE_API_KEY = "AIzaSyDYEeSTrT7pPpVzpmaJ491gxogVxfWwpvM"

def fetch_youtube_videos(query, max_results=6):
    url = f"https://www.googleapis.com/youtube/v3/search"
    params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'key': YOUTUBE_API_KEY,
        'maxResults': max_results
    }
    response = requests.get(url, params=params)
    videos = []
    if response.status_code == 200:
        data = response.json()
        for item in data['items']:
            video_id = item['id']['videoId']
            video_title = item['snippet']['title']
            videos.append({'video_id': video_id, 'title': video_title})
    return videos

from sklearn.ensemble import RandomForestClassifier
owm = pyowm.OWM('11081b639d8ada3e97fc695bcf6ddb20')
import time
st.set_page_config(page_title = 'Smart Irrigarion System', 
        layout='wide',page_icon=":mag_right:")
st.sidebar.title('Navigation')
selected = st.sidebar.radio("", ['Home', 'Crop Recommendations','Yield Prediction','Weather Forecasting','ChatBot'])
#
# One-hot encoding for categorical variables
if selected=='Home':
        st.markdown(f"<h1 style='text-align: center;font-size:60px;color:#33ccff;'> Predictive Analytics for Optimal Crop Growth</h1>", unsafe_allow_html=True)
        lottie_url = "https://assets8.lottiefiles.com/packages/lf20_CgexnTerux.json"
        st.image('agri.jpeg',caption='Agriculture',use_column_width=True)
if selected=='Weather Forecasting':
        st.markdown(f"<h1 style='text-align: center; color:skyblue;'>Weather Forecasting</h1>", unsafe_allow_html=True)
        id = st.text_input("Enter City")
        if len(id)==0:
            col1,col2,col3 = st.columns([1,5,1])
            col2.image("https://static.vecteezy.com/system/resources/previews/011/651/324/non_2x/newyork-city-highrise-skyline-simplicity-flat-design-free-png.png", caption="Weather",use_column_width=True)
        else:
            try:
                col1,col2,col3 = st.columns([3,3,2])
                if col2.button('Submit',type='primary'):
                    owm = pyowm.OWM('11081b639d8ada3e97fc695bcf6ddb20')
                    mgr = owm.weather_manager()
                    observation = mgr.weather_at_place(id)
                    weather = observation.weather
                    t1 = weather.temperature('celsius')['temp']
                    h1 = weather.humidity
                    w1 = weather.wind()
                    p1=weather.pressure['press']
                    num_weekdays = 5
                    count_weekdays = 0
                    weekday_names = []
                    now = time.time()
                    now1 = time.localtime()
                    us_date = time.strftime("%m/%d/%Y", now1)
                    while count_weekdays < num_weekdays:
                        now += 86400
                        local_time = time.localtime(now)
                        weekday = local_time.tm_wday
                        wn = time.strftime("%a", local_time)
                        if count_weekdays!=5:
                            count_weekdays += 1
                            weekday_names.append(time.strftime("%a", local_time))
                    col1, col2,col3,col4,col5= st.columns([4,4,4,4,4])
                    forecaster = mgr.forecast_at_place(id, '3h', limit=40)

                    c=0
                    l=[]
                    
                    for weather in forecaster.forecast:
                        temperature = weather.temperature('celsius')['temp']
                        c+=1
                        if c==8 or c==16 or c==24 or c==32 or c==40:
                            l.append(temperature)
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 15px; background-color: #f0f0f0; border-radius: 15px; border: 1.5px solid black; margin-bottom: 20px;">
                            <!-- Left Section (Date and Temperature) -->
                            <div style="text-align: left;">
                                <h5 style="margin: 5px 0;">Date: {us_date}</h5>
                                <h4 style="color:red; margin: 5px 0;">{t1}°C</h4>
                            </div>
                            <!-- Right Section (Cloud Image) -->
                            <div style="text-align: right;">
                                <p style="margin: 5px 0;">{id}</p> 
                                <img src="https://static.vecteezy.com/system/resources/thumbnails/022/287/830/small_2x/3d-rendering-sun-ahead-of-the-clouds-icon-3d-render-weather-sun-cloud-icon-sun-ahead-of-the-clouds-png.png" alt="Cloud" style="width: 150px; height: 150px;">
                            </div>
                        </div>
                        <div style="text-align: center; padding: 15px; background-color: #e5f2ae; border-radius: 15px; border: 1.5px solid black; margin-bottom: 20px;">
                            <!-- Additional Info -->
                            <p style="margin: 5px 0;"><b>Humidity:</b> {h1}%</p>
                            <p style="margin: 5px 0;"><b>Pressure:</b> {p1} hPa</p>
                            <p style="margin: 5px 0;"><b>Wind Speed:</b> {w1['speed']} hPa</p>
                            <hr style="border: 1px solid #ccc; margin: 10px 0;" />
                            <!-- Forecast Columns -->
                            <div style="display: flex; justify-content: space-between; text-align: center;">
                                <div style="flex: 1; margin: 5px; font-size: 16px; font-family: Arial;">
                                    <h4 style="color:#EE82EE;">{weekday_names[0]}</h4>
                                    <p>{l[0]}°C</p>
                                </div>
                                <div style="flex: 1; margin: 5px;">
                                    <h4 style="color:blue;">{weekday_names[1]}</h4>
                                    <p>{l[1]}°C</p>
                                </div>
                                <div style="flex: 1; margin: 5px;">
                                    <h4 style="color:green;">{weekday_names[2]}</h4>
                                    <p>{l[2]}°C</p>
                                </div>
                                <div style="flex: 1; margin: 5px;">
                                    <h4 style="color:orange;">{weekday_names[3]}</h4>
                                    <p>{l[3]}°C</p>
                                </div>
                                <div style="flex: 1; margin: 5px;">
                                    <h4 style="color:red;">{weekday_names[4]}</h4>
                                    <p>{l[4]}°C</p>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            except:
                st.error("City not found. Please enter a valid city name.")
if selected=='Crop Recommendations':
    st.markdown(f"<h1 style='text-align: center; color:red;'>Crop Recomendation</h1>", unsafe_allow_html=True)
    df=pd.read_csv("Crop_recommendation.csv")
    model= 'new_rf_model.pickle'
    model = joblib.load(model)
    features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
    target = df['label']
    labels = df['label']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
    RF = RandomForestClassifier(n_estimators=25, random_state=42)
    RF.fit(Xtrain,Ytrain)
    col1, col2,col3= st.columns([5,5,5])
    with col1:
        a=st.number_input('Enter N (Nitrogen) (Range: 0-120)',min_value=0,max_value=120)
    with col2:
        b=st.number_input('Enter P (Phosphorus) (Range: 0-120)',min_value=0,max_value=120)
    with col3:
        c1=st.number_input('Enter K (Potassium) (Range: 0-120)',min_value=0,max_value=120)
    col1, col2,col3,col4= st.columns([5,5,5,5])
    with col1:
        d=st.number_input('Temperature °C')
    with col2:
        e=st.number_input('Humidity %')
    with col3:
        f=st.number_input('pH')
    with col4:
        g=st.number_input('Rainfall mm')
    col1,col2,col3=st.columns([5,5,1])
    button=col2.button('Predict',type='primary',key='predict')
    if button:
        data = np.array([[a,b,c1,d,e,f,g]])
        prediction = RF.predict(data)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.write(f'Your recommended crop is {prediction[0]}')
            if prediction[0]=='apple':
                st.image("images/apple.jpg",caption='Apple')
            if prediction[0]=='banana':
                st.image("images/banana.jpg",caption='Banana')
            if prediction[0]=='blackgram':
                st.image("images/blackgram.jpg",caption='Blackgram')
            if prediction[0]=='chickpea':
                st.image("images/chickpea.jpg",caption='Chickpea')
            if prediction[0]=='coconut':
                st.image("images/coconut.jpg",caption='Coconut')
            if prediction[0]=='coffee':
                st.image("images/coffee.jpg",caption='Coffee')
            if prediction[0]=='cotton':
                st.image("images/cotton.jpg",caption='Cotton')
            if prediction[0]=='grapes':
                st.image("images/grapes.jpg",caption='Grapes')
            if prediction[0]=='jute':
                st.image("images/jute.jpg",caption='Jute')
            if prediction[0]=='kidneybeans':
                st.image("images/kidneybeans.jpg",caption='Kidneybeans')
            if prediction[0]=='lentil':
                st.image("images/lentil.jpg",caption='Lentil')
            if prediction[0]=='maize':
                st.image("images/maize.jpg" ,caption='Maize')
            if prediction[0]=='mango':
                st.image("images/mango.jpg",caption='Mango')
            if prediction[0]=='mothbeans':
                st.image("images/mothbeans.jpg",caption='Mothbeans')
            if prediction[0]=='mungbean':
                st.image("images/mungbeans.jpg" ,caption='Mungbean')
            if prediction[0]=='muskmelon':
                st.image("images/muskmelon.jpg",caption='Muskmelon')
            if prediction[0]=='orange':
                st.image("images/orange.jpg",caption='Orange')
            if prediction[0]=='papaya':
                st.image("images/papaya.jpg",caption='Papaya')
            if prediction[0]=='pomegranate':
                st.image("images/pomogranate.jpg",caption='Pomegranate')
            if prediction[0]=='pigeonpeas':
                st.image("images/pigeonpeas.jpg",caption='Pigeonpeas')
            if prediction[0]=='rice':
                st.image("images/rice.jpg",caption='Rice')
            if prediction[0]=='watermelon':
                st.image("images/watermelon.jpg",caption='Watermelon')
        with col3:
            st.write(' ')
elif selected=='Yield Prediction':
    yield_model = 'yield_prediction_model.pkl'
    model_yield = joblib.load(yield_model)

    data = pd.read_csv('crop_data.csv') 
    X = data.drop(['CROP_PRICE', 'CROP'], axis=1)  # Features
    y = data['CROP_PRICE']  # Target variable

    # One-hot encoding for categorical variables
    X = pd.get_dummies(X)
    st.markdown(
        """
        <style>
        /* Apply background image to the main content area with transparency */
        .main {
            background-image: url('https://img.freepik.com/free-photo/detail-rice-plant-sunset-valencia-with-plantation-out-focus-rice-grains-plant-seed_181624-25838.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-color: rgba(255, 255, 255, 0.8); /* Add a semi-transparent overlay */
            background-blend-mode: overlay; /* Blend the image with the overlay */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(f"<h1 style='text-align: center; color:green;'>Crop Yield Prediction</h1>", unsafe_allow_html=True)
    input=st.selectbox("Select State",('Andaman and Nicobar Islands', 'Andhra Pradesh',
    'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh',
    'Chhattisgarh', 'Dadra and Nagar Haveli', 'Daman and Diu', 'Delhi',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh',
    'Jammu and Kashmir', 'Jharkhand', 'Karnataka', 'Kerala',
    'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
    'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan',
    'Sikkim', 'Tamil Nadu', 'Tripura', 'Uttar Pradesh'))
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        a=st.number_input('Enter N (Nitrogen) (Range: 0-120)',min_value=0,max_value=126,value=10)
    with col2:
        b=st.number_input('Enter P (Phosphorus) (Range: 0-120)',min_value=0,max_value=54,value=7)
    with col3:
        c1=st.number_input('Enter K (Potassium) (Range: 0-120)',min_value=0,max_value=59,value=8)
    with col4:
        soil_type=st.selectbox("Select Soil Type",('Alluvial', 'Black', 'Clayey', 'Loamy', 'Red', 'Sandy', 'Silt'))
    with col1:
        temp=st.number_input('Temperature °C',min_value=0.0,max_value=50.0,value=25.0)
    with col2:
        humd=st.number_input('Humidity %',min_value=0.0,max_value=100.0,value=50.0)
    with col3:
        rain=st.number_input('Rainfall mm',min_value=0.0,max_value=300.0,value=100.0)
    with col4:
        f=st.number_input('pH',min_value=0.0,max_value=14.0,value=7.0)
    d=temp
    e=humd
    col1, col2, col3, col4 = st.columns(4)
    g=rain
    col1,col2=st.columns([1,1])
    area=col1.number_input('Enter Area in Hectares',min_value=0.1,max_value=100.0,value=1.0)
    crop_name=col2.selectbox('Crop',['Barley','Cotton','Ground Nuts','Maize','Millets','Oil seeds','Paddy','Pulses','Sugarcane','Tobacco','Wheat','coffee','kidneybeans','orange','pomegranate','rice','watermelon'])
    new_data = pd.DataFrame({
    'STATE': [input],
    'SOIL_TYPE': [soil_type],
    'N_SOIL': [a],
    'P_SOIL': [b],
    'K_SOIL': [c1],
    'TEMPERATURE': [d],
    'HUMIDITY': [e],
    'ph': [f],
    'RAINFALL': [g],
    })
    new_data_encoded = pd.get_dummies(new_data)
    new_data_encoded = new_data_encoded.reindex(columns=X.columns, fill_value=0)  # Ensure same set of dummy variables
    predicted_price = model_yield.predict(new_data_encoded)
    col1,col2,col3 = st.columns([4,6,1])
    #get crop yield= profit+total cost/predicted price
    profit=0.3*predicted_price
    total_cost=0.7*predicted_price
    yi=profit+total_cost/predicted_price[0]
    yid=yi/100
    yid=yid*area

    if col2.button('Predict Crop Yield',type='primary'):
        output=f"The predicted {crop_name} yield is: {int(yid)} quintals/ha."
        st.success(output)
        query = f"{crop_name} fertilizers"
        videos = fetch_youtube_videos(query)

        # Display videos in rows of 3
        for i in range(0, len(videos), 2):
            cols = st.columns(2)  # Create 3 columns
            for j, video in enumerate(videos[i:i+2]):  # Iterate over videos for the current row
                with cols[j]:
                    st.video(f"https://www.youtube.com/watch?v={video['video_id']}")
elif selected=='ChatBot':
    api_key = "AIzaSyCEHqEUnURAuH54Tng8IjlWSR6LyzzEpCI"
    genai.configure(api_key=api_key)

    # Initialize the model
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

    # Streamlit UI
    st.title("AI-Powered Chatbot 🤖")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    user_input = st.chat_input("Ask me anything...")

    if user_input:
        # Append user message to session
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response from Gemini API
        response = model.generate_content([user_input, ""])
        bot_response = response.text

        # Append bot response to session
        st.session_state.messages.append({"role": "bot", "content": bot_response})
        with st.chat_message("bot"):
            st.markdown(bot_response)
