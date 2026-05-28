import streamlit as st

from io import BytesIO

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

from rag.pipeline import AdvancedRAG

from config.setting import (
    APP_TITLE,
    APP_DESCRIPTION
)


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide"
)

st.title(APP_TITLE)
st.markdown(APP_DESCRIPTION)


# =====================================================
# SESSION STATE
# =====================================================

if "rag_system" not in st.session_state:
    st.session_state.rag_system = AdvancedRAG()

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =====================================================
# PDF EXPORT FUNCTION
# =====================================================

def generate_chat_pdf(chat_history):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "Advanced RAG Chat History",
            styles['Title']
        )
    )

    elements.append(
        Spacer(1, 20)
    )

    for message in chat_history:

        role = message["role"].upper()

        content = message["content"]

        elements.append(
            Paragraph(
                f"<b>{role}:</b> {content}",
                styles['BodyText']
            )
        )

        elements.append(
            Spacer(1, 12)
        )

    doc.build(elements)

    buffer.seek(0)

    return buffer


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose PDF File",
        type=["pdf"]
    )

    if uploaded_file:

        if st.button("⚡ Process Document"):

            with st.spinner("Processing PDF..."):

                st.session_state.rag_system.process_document(
                    uploaded_file
                )

                st.session_state.document_processed = True

            st.success(
                "✅ Document processed successfully!"
            )


# =====================================================
# TABS
# =====================================================

chat_tab, history_tab = st.tabs([
    "💬 Chat",
    "📜 History"
])


# =====================================================
# CHAT TAB
# =====================================================

with chat_tab:

    if st.session_state.document_processed:

        st.subheader("💬 Ask Questions")

        # ---------------------------------------------
        # Chat Messages Area
        # ---------------------------------------------

        chat_container = st.container(height=400)

        with chat_container:

            for message in st.session_state.chat_history:

                with st.chat_message(message["role"]):

                    st.markdown(message["content"])

        # ---------------------------------------------
        # Fixed Input at Bottom
        # ---------------------------------------------

        st.divider()

        user_query = st.chat_input(
            "Ask a question about your document..."
        )

        if user_query:

            # Save User Message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query
            })

            # Display User Message
            with chat_container:

                with st.chat_message("user"):

                    st.markdown(user_query)

            # -----------------------------------------
            # Generate Response
            # -----------------------------------------

            with chat_container:

                with st.chat_message("assistant"):

                    with st.spinner("Thinking..."):

                        history_text = "\n".join([
                            f"{msg['role']}: {msg['content']}"
                            for msg in st.session_state.chat_history[-6:]
                        ])

                        result = (
                            st.session_state.rag_system.ask_question(
                                query=user_query,
                                chat_history=history_text
                            )
                        )

                        response = result["response"]

                        st.markdown(response)

            # Save Assistant Response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })

            st.rerun()

    else:

        st.info(
            "📄 Upload and process a PDF document to begin."
        )

# =====================================================
# HISTORY TAB
# =====================================================

with history_tab:

    st.subheader("📜 Conversation History")

    if st.session_state.chat_history:

        # ---------------------------------------------
        # Export PDF Button
        # ---------------------------------------------

        pdf_file = generate_chat_pdf(
            st.session_state.chat_history
        )

        st.download_button(
            label="📥 Export Chat as PDF",
            data=pdf_file,
            file_name="rag_chat_history.pdf",
            mime="application/pdf"
        )

        st.divider()

        # ---------------------------------------------
        # Clear History
        # ---------------------------------------------

        if st.button("🗑️ Clear History"):

            st.session_state.chat_history = []

            st.rerun()

        st.divider()

        # ---------------------------------------------
        # Filter Casual Messages
        # ---------------------------------------------

        skip_messages = [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good evening",
            "thank you",
            "thanks",
            "thx",
            "how are you",
            "how r u",
            "ok",
            "okay",
            "cool",
            "nice"
        ]

        # ---------------------------------------------
        # Display Conversations
        # ---------------------------------------------

        conversation_pairs = []

        temp_question = None

        for message in st.session_state.chat_history:

            content = (
                message["content"]
                .lower()
                .strip()
            )

            # Skip casual messages
            if content in skip_messages:
                continue

            if len(content.split()) <= 1:
                continue

            # Store question
            if message["role"] == "user":

                temp_question = message["content"]

            # Store answer
            elif (
                message["role"] == "assistant"
                and temp_question
            ):

                conversation_pairs.append({
                    "question": temp_question,
                    "answer": message["content"]
                })

                temp_question = None

        # ---------------------------------------------
        # Show Conversation History
        # ---------------------------------------------

        if conversation_pairs:

            for i, chat in enumerate(
                reversed(conversation_pairs)
            ):

                with st.expander(
                    f"Conversation {len(conversation_pairs)-i}"
                ):

                    st.markdown(
                        f"### 👤 Question\n{chat['question']}"
                    )

                    st.markdown(
                        f"### 🤖 Answer\n{chat['answer']}"
                    )

        else:

            st.info(
                "No meaningful conversations available."
            )

    else:

        st.info(
            "No conversation history available."
        )