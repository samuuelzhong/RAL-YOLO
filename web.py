import streamlit as st
import os
import uuid

# 设置一个本地目录来存储上传的视频文件
UPLOAD_FOLDER = "uploads"

# 确保上传目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

st.title('视频上传与播放')

# 文件上传组件
uploaded_file = st.file_uploader("上传视频文件", type=["mp4", "avi", "mov"])

# 打开视频文件
#video_file = open(uploaded_file, 'rb')
#video_bytes = video_file.read()
col1, col2, col3 = st.columns(3)
# 使用st.video函数播放视频
with col1:
    st.video(uploaded_file)
with col2:
    st.video('test.avi')
