import streamlit as st
import pickle
import docx
import PyPDF2
import re
import spacy

# Load spaCy NER model
# Make sure to run: python -m spacy download en_core_web_sm
nlp_ner = spacy.load("en_core_web_sm")

# Load pre-trained model and TF-IDF vectorizer
svc_model = pickle.load(open('models/clf.pkl', 'rb'))
tfidf = pickle.load(open('models/tfidf.pkl', 'rb'))
le = pickle.load(open('models/encoder.pkl', 'rb'))


# Clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\\S+\\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\\S+\\s', ' ', cleanText)
    cleanText = re.sub('@\\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\\s+', ' ', cleanText)
    return cleanText


# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)


# Extract text from TXT
def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')


# Decide extractor based on file
def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type!")


# Prediction function
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vec = tfidf.transform([cleaned_text]).toarray()
    pred_label = svc_model.predict(vec)
    return le.inverse_transform(pred_label)[0]


# Extract entities using spaCy
def extract_entities(text):
    doc = nlp_ner(text)
    entities = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, set()).add(ent.text)
    return {label: sorted(list(vals)) for label, vals in entities.items()}


# Streamlit app UI
def app():
    st.set_page_config(page_title="Resume Scanner with NER", page_icon="📄", layout="wide")
    st.title("Resume Scanner (Category Prediction + NER)")
    st.markdown("Upload a resume and extract job category + important info!")

    uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

    if uploaded_file:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("Resume text extracted!")

            if st.checkbox("Show Extracted Text"):
                st.text_area("Resume Text", resume_text, height=250)

            st.subheader("Predicted Job Category")
            category = pred(resume_text)
            st.markdown(f"🎯 **{category}**")

            # Apply NER
            if st.checkbox("Show Named Entity Recognition (NER)"):
                st.subheader("Extracted Resume Entities")
                entities = extract_entities(resume_text)

                if not entities:
                    st.info("No entities found!")
                else:
                    for label, items in entities.items():
                        st.markdown(f"**{label}**")
                        st.write(", ".join(items))

        except Exception as e:
            st.error(f"Processing Error: {str(e)}")


if __name__ == "__main__":
    app()
