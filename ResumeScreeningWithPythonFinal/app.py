import streamlit as st
import re
import nltk
import fickling

nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf=fickling.load(open('clf.pkl','rb'))
tfidfd=fickling.load(open('tfidf.pkl','rb'))

#web app
# st.title("Resume Screening Web App")
def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text
def main():
    st.title("Resume screening web app")
    upload_file = st.file_uploader("Upload Resume", type=['txt', 'pdf', 'docx'])
    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])  # Transform the text using the TF-IDF vectorizer
        prediction_id = clf.predict(input_features)[0]  # Pass the 2D array to the predict function
        st.write(prediction_id)
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }


        category_name=category_mapping.get(prediction_id,"Unknown")
        st.write("Predicted category",category_name)




#python main
if __name__=="__main__":
    main()
