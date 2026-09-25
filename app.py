import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ওয়েবসাইটের টাইটেল এবং ডিজাইন
st.set_page_config(page_title="Titanic AI", page_icon="🚢")
st.title("🚢 টাইটানিক সারভাইভাল এআই")
st.write("মেশিন লার্নিং ব্যবহার করে চেক করুন টাইটানিক জাহাজে থাকলে আপনার কী হতো!")

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
    st.subheader("আপনার তথ্য দিন:")
    age = st.number_input('আপনার বয়স (Age):', min_value=1, max_value=100, value=20)
    fare = st.number_input('টিকিটের ভাড়া (Fare - ডলারে):', min_value=0.0, value=8.05)
    ticket = st.text_input('টিকিট নম্বর (যেমন: A/5 2151):', '12345')

    # রেজাল্ট দেখার বাটন
    if st.button('রেজাল্ট দেখুন'):
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
        st.subheader("🤖 এআই (AI) এর রেজাল্ট:")
        
        # সারভাইভাল রেজাল্ট দেখানো
        if pred_surv == 1:
            st.success("🟢 রেজাল্ট: এই যাত্রী সম্ভবত বেঁচে ফিরেছিলেন!")
        else:
            st.error("🔴 রেজাল্ট: এই যাত্রী সম্ভবত মারা গিয়েছিলেন।")
            
        # ক্লাস রেজাল্ট দেখানো
        if pred_class == 1:
            st.info("🥇 টিকিটের ক্লাস: First Class")
        elif pred_class == 2:
            st.info("🥈 টিকিটের ক্লাস: Second Class")
        else:
            st.info("🥉 টিকিটের ক্লাস: Third Class")

except FileNotFoundError:
    st.error("⚠️ 'train.csv' ফাইলটি পাওয়া যাচ্ছে না! দয়া করে GitHub-এ ডাটাবেস ফাইলটি আপলোড করুন।")
