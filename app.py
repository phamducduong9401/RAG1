from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, models
import os
import streamlit as st
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

def load_db(embedding_function):
    return Chroma(persist_directory="dblast1.db",embedding_function=embedding_function)

def augment_multiple_query(query,llm):
    messages = [
        {
            "role": "system",
            "content": "あなたは役に立つ専門家の校則調査アシスタントです。ユーザーは校則について質問しています。"
            "提供された質問に対して、彼らが必要な情報を見つけるのを助けるために、必ず3つ以上5つ以下の追加の関連する質問を提案してください。"
            "複文のない短い質問だけを提案してください。トピックのさまざまな側面をカバーするさまざまな質問を提案してください。"
            "完全な質問であること、元の質問に関連していることを確認してください。"
            "1行に1つの質問を出力する。質問に番号をつけないでください。"
        },
        {"role": "user", "content": query}
    ]
    response = llm.invoke(
        input=messages,
    )
    content = response
    content = content.split("\n")
    return content 

def rag_c(question,context,llm):
    c=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>まず、ユーザーからのクエリを見て、そのクエリが挨拶、感謝の言葉、謝罪の言葉や質問したい気持ちを表す内容であれば、適切な言葉で返すしてください。参考情報関連の話を入れないでください。
                            そうではない場合は、参考情報を元に、ユーザーからの質問にできるだけ正確かつ具体的に答えてください。できるだけ回答に改行を入れて回答を見やすくしてください。
                            ユーザーからの質問と関連する適切な情報が参考情報にない場合は、"適切な情報が見つかりませんでした。"と答えてください。
                            なお、「ユーザーからのクエリが挨拶、感謝の言葉、謝罪の言葉や質問したい気持ちを表す内容でない場合、参考情報に基づいて正確かつ詳細に回答します。」といった内容は回答に入れないでください。
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {context}\n質問: {question} \n<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    result = llm.invoke(c).split("assistant<|end_header_id|>")[-1].strip()
    return result

def init_page():
    st.set_page_config(
        page_title='オリジナルチャットボット',
        page_icon='🧑‍💻',
    )
def qa(original_query,db,llm):
    # cross_encoder = CrossEncoder('cross_rank')
    MODEL_NAME = "hotchpotch/japanese-reranker-cross-encoder-large-v1"
    cross_encoder = CrossEncoder(MODEL_NAME)
    augmented_queries = augment_multiple_query(original_query,llm)
    augmented_queries = [augmented_querie.strip() for augmented_querie in augmented_queries if augmented_querie.strip()]
    augmented_queries.append(original_query)
    results=[]
    for augmented_query in augmented_queries:
        result = db.similarity_search(augmented_query,k=3,)
        results.append(result)
    # Deduplicate the retrieved documents
    unique_documents = set()
    for documents in results:
        for document in documents:
            unique_documents.add(document.page_content)
    unique_documents = list(unique_documents)
    pairs = []
    for doc in unique_documents:
        pairs.append([original_query, doc])
    scores = cross_encoder.predict(pairs)
    scored_docs = zip(scores, unique_documents)
    sorted_docs = sorted(scored_docs, reverse=True)
    reranked_docs = [doc for _, doc in sorted_docs][0:10]
    context1=''
    for i in range(0, 3): 
        context1+=reranked_docs[i]
        context1+="\n\n"
    return rag_c(original_query,context1,llm), context1, reranked_docs

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # # ローカルに保存されているモデルのパスを指定
    # model_path = '/home/RAG/model'

    # try:
    #     # モデルとトークナイザをロード
    #     tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    #     model = AutoModel.from_pretrained(model_path, local_files_only=True)

    #     # SentenceTransformer用にラップ
    #     transformer_model = models.Transformer(model_path)
    #     pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension())
    #     model = SentenceTransformer(modules=[transformer_model, pooling_model])   

    #     # 埋め込み関数を定義
    #     class SentenceTransformerEmbeddingFunction:
    #         def __init__(self, model):
    #             self.model = model
            
    #         def embed_documents(self, texts):
    #             return self.model.encode(texts).tolist()
            
    #         def embed_query(self, text):
    #             return self.model.encode([text])[0].tolist()

    #     # 埋め込み関数のインスタンスを作成
    #     embedding_function = SentenceTransformerEmbeddingFunction(model)

    # except Exception as e:
    #     print(f"エラーが発生しました: {e}")
    embedding_function = SentenceTransformer("intfloat/multilingual-e5-large")
    db = load_db(embedding_function)
    init_page()

    from langchain_ollama import OllamaLLM
    local_llm1 = OllamaLLM(base_url="http://csink12.nda.ac.jp:11434",
        model="elyza:70b",
        temperature=0.01
    )
    local_llm2 = OllamaLLM(base_url="http://csink12.nda.ac.jp:11434",
        model="elyza:70b",
        temperature=0
    )

    if "messages" not in st.session_state:
      st.session_state.messages = []
    if user_input := st.chat_input('質問しよう！'):
        # 以前のチャットログを表示
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        print(user_input)
        with st.chat_message('user'):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message('assistant'):
            with st.spinner('防大チャットボットが検索中 ...'):
                response = qa(user_input,db,local_llm2)
                st.markdown(response[0])
                print(response[0])
                st.markdown("\n\n")
                st.markdown(f"参考元：{response[2][0]}")
        st.session_state.messages.append({"role": "assistant", "content": response[0]})

if __name__ == "__main__":
  main()
