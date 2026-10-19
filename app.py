"""Streamlit chat for the Royal Enfield document collection."""

from __future__ import annotations

import streamlit as st

from rag import answer_question


def main() -> None:
    st.set_page_config(page_title="Royal Enfield assistant", page_icon="🏍️", layout="centered")
    st.title("Royal Enfield assistant")
    st.caption("Answers come only from the brochure and the dealer directory.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for item in message["sources"]:
                        heading = item["title"] or item["source"]
                        place = item.get("location") or ""
                        st.markdown(f"**{heading}** — page {item['page']} {place}")
                        st.text(item["text"])

    question = st.chat_input("Ask about a bike or a dealer")
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            result = answer_question(question)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources"):
                for item in result["sources"]:
                    heading = item["title"] or item["source"]
                    place = item.get("location") or ""
                    st.markdown(f"**{heading}** — page {item['page']} {place}")
                    st.text(item["text"])
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )


if __name__ == "__main__":
    main()
