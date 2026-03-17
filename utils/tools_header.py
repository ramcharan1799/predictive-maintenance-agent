import pandas as pd
import numpy as np
import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=get_api_key())
