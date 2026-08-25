from fastapi import FastAPI, UploadFile, File, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from io import BytesIO
import os
import shutil
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_core.prompts import PromptTemplate

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY is not set")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")


groq_client = Groq(
    api_key=GROQ_API_KEY
)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "openai/gpt-oss-120b"

vector_store = None

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY
)

# model = ChatGoogleGenerativeAI(
#     model=LLM_MODEL,
#     temperature=0.3,
#     google_api_key=GOOGLE_API_KEY
# )

# prompt = PromptTemplate(
#     template="""
# Answer the question as accurately as possible using ONLY
# the information provided in the context.

# If the answer is not available in the context, say:

# "Answer is not available in the context."

# Do not make up information.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """,
#     input_variables=["context", "question"]
# )


@app.get("/")
def read_root():
    return {
        "message": "PDF Q&A API is running"
    }


def get_pdf_text(pdf_contents):
    text = ""

    for pdf in pdf_contents:
        pdf_reader = PdfReader(
            BytesIO(pdf)
        )

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )

    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    global vector_store

    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    vector_store.save_local(
        FAISS_INDEX_PATH
    )

    return vector_store


@app.post("/process_pdf")
async def process_pdf(
    pdf_docs: list[UploadFile] = File(...)
):
    try:
        pdf_contents = []

        for pdf in pdf_docs:
            contents = await pdf.read()

            if contents:
                pdf_contents.append(contents)

        if not pdf_contents:
            raise HTTPException(
                status_code=400,
                detail="No PDF files were uploaded."
            )

        raw_text = get_pdf_text(
            pdf_contents
        )

        if not raw_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF."
            )

        text_chunks = get_text_chunks(
            raw_text
        )

        if not text_chunks:
            raise HTTPException(
                status_code=400,
                detail="No text chunks were created."
            )

        get_vector_store(
            text_chunks
        )

        return {
            "status": "success",
            "message": "PDF processing completed",
            "chunks": len(text_chunks)
        }

    except HTTPException:
        raise

    except Exception as e:
        print(
            "Error processing PDF:",
            e
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/answer_question")
async def answer_question(
    user_question: str = Body(..., embed=True)
):
    global vector_store

    try:
        if vector_store is None:

            if not os.path.exists(
                FAISS_INDEX_PATH
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Please upload and process a PDF first."
                )

            vector_store = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )

        docs = vector_store.similarity_search(
            user_question,
            k=3
        )

        if not docs:
            return {
                "answer": "Answer is not available in the context."
            }

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )


        # final_prompt = prompt.format(
        #     context=context,
        #     question=user_question
        # )

        # response = model.invoke(
        #     final_prompt
        # )

        # content = response.content

        prompt = f"""
You are a PDF question-answering assistant.

Answer the question using ONLY the information
provided in the context.

If the answer is not available in the context,
say exactly:

"Answer is not available in the context."

Do not make up information.

Context:
{context}

Question:
{user_question}

Answer:
"""

        completion = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_completion_tokens=2048,
            top_p=1,
            reasoning_effort="low",
            stream=False
        )

        answer = completion.choices[0].message.content

        # if isinstance(content, list):
        #     answer = ""

        #     for item in content:
        #         if isinstance(item, dict):
        #             answer += item.get(
        #                 "text",
        #                 ""
        #             )
        #         elif isinstance(item, str):
        #             answer += item
        # else:
        #     answer = str(content)

        return {
            "answer": answer
        }

    except HTTPException:
        raise

    except Exception as e:
        print(
            "Error answering question:",
            e
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )