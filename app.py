import streamlit as st #for creating web apps for ML
from langchain.prompts import PromptTemplate #translate user input and parameters into instructions for LLms
from langchain_community.llms import CTransformers

## Function to get response from LLAma 2 model

def getLLamaresponse(input_text, no_words, blog_style):
    ## LLama model
    llm = CTransformers(model='Model\llama-2-7b-chat.ggmlv3.q8_0.bin',
                    
                        model_type = 'llama',
                        config={'max_new_tokens':256,
                                'temperature':0.01})
    
    ## Prompt Template
    template="""
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.

        """
    prompt = PromptTemplate(input_variables=['blog_style', "input_text", 'no_words'],
                           template = template)
    
    ## Generate the response from the LLama 2 model
    response = llm.invoke(prompt.format(blog_style=blog_style, input_text=input_text, no_words = no_words))
    print(response)
    return response




st.set_page_config(page_title='Generate Blogs',
                      page_icon='✨',
                      layout='centered',
                      initial_sidebar_state='collapsed')
st.header("Generate Blogs ✨ ")

input_text = st.text_input("Write the Blog Topic")

##creating two more columns for additional 2 fields

col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input('Number of Words')

with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientists', 'Common People', 'Doctors', 'Software Engineers', 'Children'), index=0 )

submit = st.button('Generate')
# Final response
if submit:
    st.write(getLLamaresponse(input_text, no_words, blog_style))