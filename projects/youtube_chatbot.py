from langchain_community.document_loaders import YoutubeLoader

url=''
loader=YoutubeLoader.from_youtube_url('url')

loader.load()

