import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from streamlit_mic_recorder import speech_to_text  # Import speech-to-text function
import fitz  # PyMuPDF for capturing screenshots
import pdfplumber  # For searching text in PDF
import torch
from sentence_transformers import SentenceTransformer

# Initialize API key variables
groq_api_key = "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"
google_api_key = "AIzaSyDdAiOdIa2I28sphYw36Genb4D--2IN1tU"

# Change the page title and icon
st.set_page_config(
    page_title="BGC ChatBot",  # Page title
    page_icon="BGC Logo Colored.svg",  # New page icon
    layout="wide"  # Page layout
)

# Function to apply CSS based on language direction
def apply_css_direction(direction):
    st.markdown(
        f"""
        <style>
            .stApp {{
                direction: {direction};
                text-align: {direction};
            }}
            .stChatInput {{
                direction: {direction};
            }}
            .stChatMessage {{
                direction: {direction};
                text-align: {direction};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# PDF Search and Screenshot Class
class PDFSearchAndDisplay:
    def __init__(self):
        pass

    def search_and_highlight(self, pdf_path, search_term):
        highlighted_pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text()
                if search_term in text:
                    highlighted_pages.append((page_number, text))
        return highlighted_pages

    def capture_screenshots(self, pdf_path, pages):
        doc = fitz.open(pdf_path)
        screenshots = []
        for page_number, _ in pages:
            page = doc.load_page(page_number)
            pix = page.get_pixmap()
            screenshot_path = f"screenshot_page_{page_number}.png"
            pix.save(screenshot_path)
            screenshots.append(screenshot_path)
        return screenshots

# Sidebar configuration
with st.sidebar:
    # Language selection dropdown
    interface_language = st.selectbox("Interface Language", ["English", "العربية"])

    # Apply CSS direction based on selected language
    if interface_language == "العربية":
        apply_css_direction("rtl")  # Right-to-left for Arabic
        st.title("الإعدادات")  # Sidebar title in Arabic
    else:
        apply_css_direction("ltr")  # Left-to-right for English
        st.title("Settings")  # Sidebar title in English

    # Validate API key inputs and initialize components if valid
    if groq_api_key and google_api_key:
        # Set Google API key as environment variable
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Initialize ChatGroq with the provided Groq API key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

        # Define the chat prompt template with memory
        prompt = ChatPromptTemplate.from_messages([
              ("system", """
            You are a helpful assistant for Basrah Gas Company (BGC). Your task is to answer questions based on the provided context about BGC. The context is supplied as a documented resource (e.g., a multi-page manual or database) that is segmented by pages. Follow these rules strictly:

            1. **Language Handling:**
               - If the question is in English, answer in English.
               - If the question is in Arabic, answer in Arabic.
               - If the user explicitly requests a response in a specific language, respond in that language.
               - If the user’s interface language conflicts with the language of the available context (e.g., context is only in English but the interface is Arabic), provide the best possible answer in the available language while noting any limitations if needed.

            2. **Understanding and Using Context:**
               - The “provided context” refers to the complete set of documents, manuals, or data provided, segmented into pages.
               - When answering a question, refer only to the relevant section or page of this context.
               - If the question spans multiple sections or pages, determine which page is most directly related to the question. If ambiguity remains, ask the user for clarification.

            3. **Handling Unclear, Ambiguous, or Insufficient Information:**
               - If a question is unclear or lacks sufficient context, respond with:
                 - In English: "I'm sorry, I couldn't understand your question. Could you please provide more details?"
                 - In Arabic: "عذرًا، لم أتمكن من فهم سؤالك. هل يمكنك تقديم المزيد من التفاصيل؟"
               - If the question cannot be answered with the available context, respond with:
                 - In English: "I'm sorry, I don't have enough information to answer that question."
                 - In Arabic: "عذرًا، لا أملك معلومات كافية للإجابة على هذا السؤال."

            4. **User Interface Language and Content Availability:**
               - Prioritize the user's interface language when formulating answers unless the question explicitly specifies another language.
               - If the context is only available in one language, answer in that language while ensuring clarity.

            5. **Professional Tone:**
               - Maintain a professional, respectful, and factual tone in all responses.
               - Avoid making assumptions or providing speculative answers.

            6. **Page-Specific Answers and Source Referencing:**
               - When a question relates directly to content found on a specific page, use that page’s content exclusively for your answer.
               - For topics that consistently come from a specific page, always refer to that page. For example:
                   - **Life Saving Rules:** If asked "What are the life saving rules?", respond with:
                         1. Bypassing Safety Controls  
                         2. Confined Space  
                         3. Driving  
                         4. Energy Isolation  
                         5. Hot Work  
                         6. Line of Fire  
                         7. Safe Mechanical Lifting  
                         8. Work Authorisation  
                         9. Working at Height  
                     (This answer is sourced from page 5.)
                   - **PTW Explanation:** If asked "What is PTW?", respond with:
                         "BGC’s PTW is a formal documented system that manages specific work within BGC’s locations and activities. PTW aims to ensure Hazards and Risks are identified, and Controls are in place to prevent harm to People, Assets, Community, and the Environment (PACE)."
                     (This answer is sourced from page 213.)
               - Optionally, you may append a note such as " (Source: Page X)" if it aids clarity, but only do so if it does not conflict with other instructions or if the user explicitly requests source details.

            7. **Handling Overlapping or Multiple Relevant Contexts:**
               - If a question might be answered by content on multiple pages, determine the most directly relevant page. If the relevance is unclear, request clarification from the user before providing an answer.
               - When in doubt, state that the topic may span multiple sections and ask the user to specify which aspect they are interested in.

            8. **Addressing Updates and Content Discrepancies:**
               - In case the context or page content has been updated and there are discrepancies between the provided examples and current content, rely on the latest available context.
               - If there is uncertainty due to updates, mention that the answer is based on the most recent information available from the provided context.

            9. **Additional Examples and Clarifications:**
               - Besides the examples provided above, ensure you handle edge cases where the question may not exactly match any example. Ask for clarification if necessary.
               - Always double-check that your answer strictly adheres to the information found on the relevant page in the context.
               
            10. **Section-Specific Answers and Source Referencing:**
               - If the answer is derived from a particular section within a page, indicate this by referencing the section number (e.g., Section 2.14) rather than the page number.
               - Ensure that when a section is mentioned, you use the term "Section" followed by the appropriate identifier, avoiding the term "Page" if the context is organized by sections.
               - In cases where both page and section references are relevant, include both details appropriately to maintain clarity for the user.
               
            By following these guidelines, you will provide accurate, context-based answers while maintaining clarity, professionalism, and consistency with the user’s language preferences.
"""
),
            MessagesPlaceholder(variable_name="history"),  # Add chat history to the prompt
            ("human", "{input}"),
            ("system", "Context: {context}"),
        ])

        # Load existing embeddings from files
        if "vectors" not in st.session_state:
            with st.spinner("جارٍ تحميل التضميدات... الرجاء الانتظار." if interface_language == "العربية" else "Loading embeddings... Please wait."):
                # Initialize embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )

                # Load existing FAISS index with safe deserialization
                embeddings_path = "embeddings"  # Path to your embeddings folder
                embeddings_path_2 = "embeddingsocr"
                
                try:
                    # Load first FAISS index
                    vectors_1 = FAISS.load_local(
                    embeddings_path, embeddings, allow_dangerous_deserialization=True
                    )

                    # Load second FAISS index
                    vectors_2 = FAISS.load_local(
                    embeddings_path_2, embeddings, allow_dangerous_deserialization=True
                    )

                    # Merge both FAISS indexes
                    vectors_1.merge_from(vectors_2)

                    # Store in session state
                    st.session_state.vectors = vectors_1

                except Exception as e:
                    st.error(f"Error loading embeddings: {str(e)}")
                    st.session_state.vectors = None
            # Microphone button in the sidebar
        st.markdown("### الإدخال الصوتي" if interface_language == "العربية" else "### Voice Input")
        input_lang_code = "ar" if interface_language == "العربية" else "en"  # Set language code based on interface language
        voice_input = speech_to_text(
            start_prompt="🎤",
            stop_prompt="⏹️ إيقاف" if interface_language == "العربية" else "⏹️ Stop",
            language=input_lang_code,  # Language (en for English, ar for Arabic)
            use_container_width=True,
            just_once=True,
            key="mic_button",
        )

        # Reset button in the sidebar
        if st.button("إعادة تعيين الدردشة" if interface_language == "العربية" else "Reset Chat"):
            st.session_state.messages = []  # Clear chat history
            st.session_state.memory.clear()  # Clear memory
            st.success("تمت إعادة تعيين الدردشة بنجاح." if interface_language == "العربية" else "Chat has been reset successfully.")
            st.rerun()  # Rerun the app to reflect changes immediately
    else:
        st.error("الرجاء إدخال مفاتيح API للمتابعة." if interface_language == "العربية" else "Please enter both API keys to proceed.")

# Initialize the PDFSearchAndDisplay class with the default PDF file
pdf_path = "BGC.pdf"
pdf_searcher = PDFSearchAndDisplay()

# Main area for chat interface
# Use columns to display logo and title side by side
col1, col2 = st.columns([1, 4])  # Adjust the ratio as needed

# Display the logo in the first column
with col1:
    st.image("BGC Logo Colored.svg", width=100)  # Adjust the width as needed

# Display the title and description in the second column
with col2:
    if interface_language == "العربية":
        st.title("بوت الدردشة BGC")
        st.write("""
        **مرحبًا!**  
        هذا بوت الدردشة الخاص بشركة غاز البصرة (BGC). يمكنك استخدام هذا البوت للحصول على معلومات حول الشركة وأنشطتها.  
        **كيفية الاستخدام:**  
        - اكتب سؤالك في مربع النص أدناه.  
        - أو استخدم زر المايكروفون للتحدث مباشرة.  
        - سيتم الرد عليك بناءً على المعلومات المتاحة.  
        """)
    else:
        st.title("BGC ChatBot")
        st.write("""
        **Welcome!**  
        This is the Basrah Gas Company (BGC) ChatBot. You can use this bot to get information about the company and its activities.  
        **How to use:**  
        - Type your question in the text box below.  
        - Or use the microphone button to speak directly.  
        - You will receive a response based on the available information.  
        """)

# Initialize session state for chat messages if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for page references if not already done
if "page_references" not in st.session_state:
    st.session_state.page_references = {}

# Initialize memory if not already done
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

# List of negative phrases to check for unclear or insufficient answers
negative_phrases = [
    "I'm sorry",
    "عذرًا",
    "لا أملك معلومات كافية",
    "I don't have enough information",
    "لم أتمكن من فهم سؤالك",
    "I couldn't understand your question",
    "لا يمكنني الإجابة على هذا السؤال",
    "I cannot answer this question",
    "يرجى تقديم المزيد من التفاصيل",
    "Please provide more details",
    "غير واضح",
    "Unclear",
    "غير متأكد",
    "Not sure",
    "لا أعرف",
    "I don't know",
    "غير متاح",
    "Not available",
    "غير موجود",
    "Not found",
    "غير معروف",
    "Unknown",
    "غير محدد",
    "Unspecified",
    "غير مؤكد",
    "Uncertain",
    "غير كافي",
    "Insufficient",
    "غير دقيق",
    "Inaccurate",
    "غير مفهوم",
    "Not clear",
    "غير مكتمل",
    "Incomplete",
    "غير صحيح",
    "Incorrect",
    "غير مناسب",
    "Inappropriate",
    "Please provide me",  # إضافة هذه العبارة
    "يرجى تزويدي",  # إضافة هذه العبارة
    "Can you provide more",  # إضافة هذه العبارة
    "هل يمكنك تقديم المزيد"  # إضافة هذه العبارة
]

# Function to capture and display a page from PDF with better error handling
def capture_and_display_page(page_num):
    """Capture and display a page from PDF with better error handling"""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        screenshot_path = f"screenshot_page_{page_num}.png"
        pix.save(screenshot_path)
        doc.close()
        st.image(screenshot_path)
        return True
    except Exception as e:
        st.error(f"Error capturing page {page_num}: {str(e)}")
        return False

# Function to calculate semantic similarity between two texts using all-MiniLM-L6-v2
def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity between two texts using all-MiniLM-L6-v2"""
    try:
        # Convert texts to vectors (embeddings)
        embedding1 = sentence_model.encode(text1, convert_to_tensor=True)
        embedding2 = sentence_model.encode(text2, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
        return similarity.item()
    except Exception as e:
        st.error(f"Error calculating similarity: {str(e)}")
        return 0.0

# Function to find relevant pages using semantic similarity with improved scoring
def get_relevant_pages(context_docs, query_text, min_similarity=0.5):  # Increased minimum similarity threshold
    """
    Find relevant pages using semantic similarity with improved scoring.
    Only returns pages with similarity score >= 0.5 for better accuracy.
    """
    page_scores = {}
    
    try:
        for doc in context_docs:
            page_number = doc.metadata.get("page", "unknown")
            if page_number != "unknown" and str(page_number).isdigit():
                page_number = int(page_number)
                content = doc.page_content.strip()
                
                # Skip very short or empty content
                if len(content) < 10:
                    continue
                
                # Calculate semantic similarity
                similarity_score = calculate_semantic_similarity(query_text, content)
                
                # Store if score is above threshold (0.5)
                if similarity_score >= min_similarity:
                    if page_number not in page_scores or similarity_score > page_scores[page_number]["score"]:
                        page_scores[page_number] = {
                            "score": similarity_score,
                            "content": content
                        }
        
        # Sort pages by score in descending order
        # Limit to top 5 most relevant pages if we have too many
        sorted_pages = dict(sorted(page_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:5])
        return sorted_pages
    except Exception as e:
        st.error(f"Error finding relevant pages: {str(e)}")
        return {}

# Function to display page references with improved visualization and higher relevance threshold
def display_page_references(message_index, relevant_pages):
    """Display page references with improved visualization and higher relevance threshold"""
    try:
        with st.expander("مراجع الصفحات" if interface_language == "العربية" else "Page References", expanded=False):
            # Display page numbers
            page_numbers = sorted(relevant_pages.keys())
            page_numbers_str = ", ".join(map(str, page_numbers))
            st.write(f"{'هذه الإجابة وفقًا للصفحات' if interface_language == 'العربية' else 'This answer references pages'}: {page_numbers_str}")
            
            # Create two columns for better organization
            col1, col2 = st.columns([1, 2])
            
            # Show similarity scores in first column
            with col1:
                st.write("Semantic Similarity Scores:")
                for page_num in page_numbers:
                    score = relevant_pages[page_num]["score"]
                    # Updated color coding for higher threshold
                    if score >= 0.8:
                        score_color = "green"
                        confidence = "High Relevance"
                    elif score >= 0.65:
                        score_color = "orange"
                        confidence = "Medium Relevance"
                    else:
                        score_color = "blue"
                        confidence = "Moderate Relevance"
                    st.markdown(f"Page {page_num}: <span style='color: {score_color}'>{score:.3f}</span> ({confidence})", unsafe_allow_html=True)
            
            # Show page tabs in second column
            with col2:
                tabs = st.tabs([f"Page {page}" for page in page_numbers])
                for tab, page_num in zip(tabs, page_numbers):
                    with tab:
                        score = relevant_pages[page_num]["score"]
                        # Only show screenshots for highly relevant pages (score > 0.5)
                        # if score >= 0.5:
                        #     # Show content preview
                        #     st.write("Content Preview:")
                        #     content = relevant_pages[page_num]["content"]
                        #     st.text(content[:200] + "..." if len(content) > 200 else content)
                            
                            # Show page screenshot
                            st.write("Page Screenshot:")
                            capture_and_display_page(page_num)
                        else:
                            st.write("Content score below threshold. Screenshot not displayed.")
            
            return True
    except Exception as e:
        st.error(f"Error displaying page references: {str(e)}")
        return False

# Function to process response and update chat
def process_response(input_text, response, is_voice=False):
    """Process response with enhanced page reference handling"""
    try:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": input_text})
        with st.chat_message("user"):
            st.markdown(input_text)

        # Get response and context
        assistant_response = response["answer"]
        context = response.get("context", [])
        current_msg_index = len(st.session_state.messages)

        # Check for uncertainty in response
        uncertain_response = any(phrase.lower() in assistant_response.lower() for phrase in negative_phrases)
        
        # Override uncertainty if we have good context
        if uncertain_response and context:
            has_tables = any("table" in doc.page_content.lower() for doc in context)
            if has_tables:
                for phrase in negative_phrases:
                    assistant_response = assistant_response.replace(phrase, "")
                uncertain_response = False

        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Update memory
        st.session_state.memory.chat_memory.add_user_message(input_text)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)
        
        # Process page references
        if context:
            # Combine question and answer for better context matching
            query_text = f"{input_text} {assistant_response}"
            
            # Get relevant pages with semantic scoring
            relevant_pages = get_relevant_pages(context, query_text)
            
            # Display references if found
            if relevant_pages:
                page_numbers_str = ", ".join(map(str, sorted(relevant_pages.keys())))
                st.session_state.page_references[current_msg_index] = {
                    "pages": page_numbers_str,
                    "scores": relevant_pages
                }
                
                # Display page references
                with st.chat_message("assistant"):
                    display_page_references(current_msg_index, relevant_pages)
        
        return True
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")
        return False

# Initialize sentence model for semantic similarity
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Display chat history with page references
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display page references for each message if they exist
        if i in st.session_state.page_references and st.session_state.page_references[i]:
            with st.expander("مراجع الصفحات" if interface_language == "العربية" else "Page References", expanded=False):
                page_numbers_str = st.session_state.page_references[i]["pages"]
                st.write(f"{'هذه الإجابة وفقًا للصفحات' if interface_language == 'العربية' else 'This answer references pages'}: {page_numbers_str}")
                relevant_pages = st.session_state.page_references[i]["scores"]
                page_numbers = sorted(relevant_pages.keys())
                for page_num in page_numbers:
                    st.write(f"Page {page_num}: {relevant_pages[page_num]['score']:.3f}")
                highlighted_pages = [(int(page), "") for page in page_numbers]
                screenshots = pdf_searcher.capture_screenshots(pdf_path, highlighted_pages)
                for screenshot in screenshots:
                    st.image(screenshot)

# Process voice input
if voice_input:
    if "vectors" in st.session_state and st.session_state.vectors is not None:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({
            "input": voice_input,
            "context": retriever.get_relevant_documents(voice_input),
            "history": st.session_state.memory.chat_memory.messages
        })
        
        process_response(voice_input, response, is_voice=True)
    else:
        assistant_response = (
            "لم يتم تحميل التضميدات. يرجى التحقق مما إذا كان مسار التضميدات صحيحًا." if interface_language == "العربية" else "Embeddings not loaded. Please check if the embeddings path is correct."
        )
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

# Text input field
if interface_language == "العربية":
    human_input = st.chat_input("اكتب سؤالك هنا...")
else:
    human_input = st.chat_input("Type your question here...")

# Process text input
if human_input:
    if "vectors" in st.session_state and st.session_state.vectors is not None:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({
            "input": human_input,
            "context": retriever.get_relevant_documents(human_input),
            "history": st.session_state.memory.chat_memory.messages
        })
        
        process_response(human_input, response)
    else:
        assistant_response = (
            "لم يتم تحميل التضميدات. يرجى التحقق مما إذا كان مسار التضميدات صحيحًا." if interface_language == "العربية" else "Embeddings not loaded. Please check if the embeddings path is correct."
        )
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
