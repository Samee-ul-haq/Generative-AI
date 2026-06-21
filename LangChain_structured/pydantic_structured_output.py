from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel,Field

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

#Schema
# There is no data validation in TypedDict means despiste 
# giving saying string the ouput can be any other type also/  --- Pydantic is solution.
class Review(BaseModel):

    key_themes:list[str]=Field(description="Write down all the key themes discussed int the review in a list")
    summary:str=Field(description="A brief summary of the review")
    sentiment:Literal["pos","neg"]=Field("Return sentiment")
    pros:Optional[list[str]]=Field(description="Write down all the pros")
    cons:Optional[list[str]]=Field(description="Write down all the cons")
    name:Optional[str]=Field(description="Write down the name of reviewer")


structured_model=model.with_structured_output(Review)

response=structured_model.invoke(
    """"
    I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don’t use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
- Insanely powerful processor (great for gaming and productivity)
- Stunning 200MP camera with incredible zoom capabilities
- Long battery life with fast charging
- S-Pen support is unique and useful

Cons:
- Bulky and heavy—not great for one-handed use
- Bloatware still exists in One UI
- Expensive compared to competitors
      """)

print(response)
print(response.summary)
print(response.name)