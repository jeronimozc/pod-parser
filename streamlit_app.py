import streamlit as st
import tempfile
import os
from agentic_doc.parse import parse_documents
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# get api keys from secrets
if not os.environ.get("VISION_AGENT_API_KEY"):
  os.environ["VISION_AGENT_API_KEY"] = st.secrets["VISION_AGENT_API_KEY"]

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Show title and description.
st.title("Proof of Delivery Parser")
st.write(
    "Upload a Bill of Lading or Proof of Delivery document below and the agent will attempt " \
    "to infer whether the corresponding shipment was delivered."
)

# We are using Streamlit's secrets feature
# see https://docs.streamlit.io/develop/concepts/connections/secrets-management

# Initiate the Google GenAI model.
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai", temperature=0.1, timeout=120)

# Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
    "Upload a document (.pdf)", type=("pdf")
)

if uploaded_file:

    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())

    # Process the uploaded file
    results = parse_documents([path])
    parsed_doc = results[0]
    pod_text = parsed_doc.markdown

    # Prepare LLM prompt
    system_template = """
    Answer the following questions, in order, based on the Input Text. Repeat each question before its answer, separate each question-answer pair by a line break. Write all dates in MM/DD/YYYY format, and assume all dates belong to the year range 2024-2026.

    1. Is this a Bill of Lading or a Proof of Delivery? If the answer is negative, ignore the rest of the questions and output your answer, followed by a reasoning for you answer, without using any semi-colons.
    3. What is the B/L date?
    4. What is the B/L number? 
    5. To whom was it sold? 
    6. To whom was it shipped? 
    7. Is this marked "fully delivered", "partially delivered" or is this unknown?
    8. Is the Driver Signature present? 
    9. Is there a Ship Date? If so, what is the Ship Date?
    10. Is the Reciever Signature present?
    11. What is there a Delivery Date? If so, what is the Delivery Date?
    12. Is the Consignor Signature present?
    13. Is the Shipper's Certification Signature present?

    After answering these questions, use the answers to questions 7, 10, 11 and/or 12 to answer the following question and fully justify your answer.
    Consider the following when answering:
    * If the answer to question 7 is "fully delivered", the shipment was delivered.
    * If the answers to question 10 or 12 are "yes", the shipment was delivered. 
    * If the answer to question 7 is "partially delivered", mention that in you response.
    * If the answer to both questions 10 and 12 are "no", the shipment was *not* delivered.

    14. Was this shipment delivered? 
    Answer "Yes" if there is clear evidence that this shipment was delivered; 
    Answer "No" if there is clear evidence that the shipment was not delivered, for example if no signatures are present and it's not marked as "partially delivered" nor "fully delivered"
    ; or answer "Unclear" if there is no clear evidence either way.

    Finally, place the answers to all the questions at the end, on a single line, separated by semicolons, including any justification.
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), 
        ("user", "<Input Text>\n{input_text}\n</Input Text>")]
    )

    prompt = prompt_template.invoke({"input_text": pod_text})

    # Invoke LLM    
    response = model.invoke(prompt)

    # Stream the response to the app using `st.write_stream`.
    response_data = response.content.split("\n")[-1].split(";")

    disclaimer = "Please consider I'm a demo app. My responses can still improve with some work, so any feedback is appreciated!"

    output = f"""
    **Was this shipment delivered?**  
    :robot_face: {response_data[-1].strip()}

    :material/info: *{disclaimer}*
    """

    st.write(output)
