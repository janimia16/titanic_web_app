import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ওয়েবসাইটের টাইটেল এবং ডিজাইন
st.set_page_config(page_title="Titanic AI", page_icon="🚢")
st.title("Titanic survived and ticket class cheking AI")
st.write("Ask the AI with 'age' 'fare' 'ticket number'. He can reply servival status and ticket class")

# ডাটা লোড করার ফাংশন (যাতে ওয়েবসাইট ফাস্ট কাজ করে)
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Age'] = df['Age'].fillna(df['Age'].median())
    return df

try:
    df = load_data()
    
    # --- মডেল ট্রেইনিং শুরু ---
    # ১. সারভাইভাল মডেল
    X_surv = df[['Age', 'Fare']]
    y_surv = df['Survived']
    model_surv = RandomForestClassifier(n_estimators=100, random_state=42)
    model_surv.fit(X_surv, y_surv)
    
    # ২. টিকিটের ক্লাস মডেল
    le_ticket = LabelEncoder()
    df['Ticket_Encoded'] = le_ticket.fit_transform(df['Ticket'].astype(str))
    X_class = df[['Ticket_Encoded']]
    y_class = df['Pclass']
    model_class = RandomForestClassifier(n_estimators=100, random_state=42)
    model_class.fit(X_class, y_class)
    # --- মডেল ট্রেইনিং শেষ ---

    # ইউজারের কাছ থেকে ইনপুট নেওয়া (UI)
    st.subheader("Please provide the following information:")
    age = st.number_input('Enter the passengers age:', min_value=1, max_value=100, value=00)
    fare = st.number_input('Enter the ticket fare:', min_value=0.0, value=0.0)
    ticket = st.text_input('Enter the ticket number:', '0')

    # রেজাল্ট দেখার বাটন
    if st.button('Check'):
        # সারভাইভাল প্রেডিকশন
        user_surv = pd.DataFrame({'Age': [age], 'Fare': [fare]})
        pred_surv = model_surv.predict(user_surv)[0]
        
        # ক্লাস প্রেডিকশন
        try:
            ticket_encoded = le_ticket.transform([str(ticket)])[0]
        except ValueError:
            ticket_encoded = 0
        user_class = pd.DataFrame({'Ticket_Encoded': [ticket_encoded]})
        pred_class = model_class.predict(user_class)[0]
        
        st.markdown("---")
        st.subheader("The Result:")
        
        # সারভাইভাল রেজাল্ট দেখানো
        if pred_surv == 1:
            st.success("He was rescued alive.")
        else:
            st.error("He was recovered dead.")
            
        # ক্লাস রেজাল্ট দেখানো
        if pred_class == 1:
            st.info("First Class Ticket")
        elif pred_class == 2:
            st.info("Second Class Ticket")
        else:
            st.info("Third Class Ticket")

except FileNotFoundError:
    st.error("⚠️ 'train.csv' ফাইলটি পাওয়া যাচ্ছে না! দয়া করে GitHub-এ ডাটাবেস ফাইলটি আপলোড করুন।")
