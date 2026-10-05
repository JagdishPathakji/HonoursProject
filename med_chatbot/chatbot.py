import os
import asyncio
import cv2
import numpy as np
import mediapipe as mp
import torch
import time
import concurrent.futures
from collections import deque
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from helper.general_dictionary import MED_VIDEO_IDS
from gpu_landmark_renderer import GPULandmarkDetector


async def load_video_frames(video_path):
    """Asynchronously loads video frames into memory and returns them as a list."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return []

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (500, 500))  # Resize for consistency
        frames.append(frame)

    cap.release()
    return frames


def process_video(frames, detector, word):
    """Processes a video by performing landmark detection on each frame."""
    prev_time = 0

    for frame in frames:
        # Detect landmarks
        landmark_canvas = detector.detect_landmarks(frame)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # Display FPS on the video frame
        cv2.putText(
            landmark_canvas,
            f"FPS: {int(fps)}",
            (10, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        
        cv2.putText(
            landmark_canvas,
            f"Word: {word}",
            (10, 60),  # Position below the FPS
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),  # Blue text
            1,
            cv2.LINE_AA,
        )

        # Show the frame
        cv2.imshow("Landmark Canvas", landmark_canvas)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


async def buffer_videos(queue, words, word_to_video_map):
    """Continuously buffers video frames into the queue while maintaining sequence."""
    buffer_index = 0  # Tracks which word is being buffered

    while buffer_index < len(words):
        if queue.full():
            await asyncio.sleep(0.1)  # Wait until there's space in the buffer
            continue

        # Buffer the next video
        word = words[buffer_index]
        video_path = word_to_video_map.get(word)
        if video_path:
            frames = await load_video_frames(video_path)
            await queue.put((frames, word))  # Add frames and word to the queue
        else:
            print(f"No video found for word: {word}")
            await queue.put(([], word))  # Placeholder for missing videos

        buffer_index += 1  # Move to the next word

    # Signal that buffering is complete
    await queue.put(None)  # None signals end of buffering


async def stream_videos(queue, detector):
    """Streams videos from the buffer asynchronously."""
    with concurrent.futures.ThreadPoolExecutor() as pool:
        while True:
            item = await queue.get()  # Wait for frames and word from the buffer
            if item is None:
                break  # Exit when buffering is done

            frames, word = item
            if not frames:
                print(f"Skipping empty frames for word: {word}")
                continue

            print(f"Processing video for word: {word}")
            # Process the video in a separate thread
            await asyncio.get_event_loop().run_in_executor(pool, process_video, frames, detector, word)


async def process_sentence(words, word_to_video_map):
    """Processes a sentence and handles buffering and streaming concurrently."""
    detector = GPULandmarkDetector()

    # Shared queue for buffering and streaming
    queue = asyncio.Queue(maxsize=3)  # Buffer size of 3 videos

    # Create tasks for buffering and streaming
    buffer_task = asyncio.create_task(buffer_videos(queue, words, word_to_video_map))
    stream_task = asyncio.create_task(stream_videos(queue, detector))

    # Run both tasks concurrently
    await asyncio.gather(buffer_task, stream_task)


def init_chatbot(vectorstore, qa_system_prompt=None):
    """qa_system_prompt: optional answer instructions (must contain {context}); defaults to ISL-style word lists."""
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-120b"
    )
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = (
        "Given the chat history and the latest user question about medical issues or health-related topics, "
        "reformulate the question to make it standalone and clear. "
        "The reformulated question should be short, to the point, and avoid complex sentence structures or grammar. "
        "Ensure that the key medical concepts are captured concisely, as the response will be in Indian Sign Language (ISL)."
        "Do NOT answer the question, just reformulate it. "
        "If it's already clear and concise, return it as is."
    )

    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),  # The user's latest input
        ]
    )

    # Create a history-aware retriever that will use LLM to contextualize queries
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # System prompt for answering the user's question based on retrieved documents
    qa_system_prompt = qa_system_prompt or (
        "You are a knowledgeable and concise medical assistant. "
        "Use the provided medical context to answer the health-related question accurately. "
        "Respond in simple, short, and direct words suitable for Indian Sign Language (ISL). "
        "Use as few words as possible while conveying the key medical concepts. "
        "Do not use grammar and punctuations as ISL does not follow grammatical structures. "
        "Lemmatize the words using part-of-speech tagging (POS), to make them easier to sign. Eg: 'running' -> 'run', 'swelling' -> 'swell'."
        "If you are unsure or do not know the answer, say 'don't know'. "
        "Do not use jargon or complex medical terms."
        "Provide a concise, evidence-based response in a single sentence in no more than easily tokenisable sentence or short phrases. "
        "\n\n"
        "{context}"
)


    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),  # Optional chat history
            ("human", "{input}"),  # User's query
        ]
    )

    # Combine documents and generate an answer using the LLM
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


def chat(rag_chain):
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "bye"]:
            break
        
        # Process the user's query through the retrieval chain
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # Display the AI's response
        print(f"AI: {result['answer']}")
        
        words = []
        for word in result["answer"].lower().replace(",","").replace(".","").replace("  "," ").split():
            if word not in MED_VIDEO_IDS.keys():
                for letter in word:
                    words.append(letter.upper())
            else:
                words.append(word)
        print(words)
        
        asyncio.run(process_sentence(words, MED_VIDEO_IDS))
        
        # Update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["answer"]))
